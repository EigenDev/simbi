# =============================================================================
# test_immersed_body_validation.py
#
# immersed-body configuration is validated at construction and at the execution
# -dict boundary, so no unhonored knob or out-of-range dial reaches the backend
# (which reads every key with a silent default). covers:
#   - AccretionProperties physical bounds
#   - a capability bit requires its property block
#   - unwired capabilities (ELASTIC / DEFORMABLE) are rejected
#   - raw body dicts are rejected in favor of the validated dataclass
# =============================================================================

import pytest

from simbi.simulation.problem import ConfigError
from simbi.simulation.runner import to_execution_dict
from simbi.simulation.tests.fixtures.fofc_periodic_blast import FofcPeriodicBlast
from simbi.types.bodies import (
    AccretionProperties,
    BodyCapability,
    GravitationalProperties,
    ImmersedBodyConfig,
    RigidProperties,
)


def _grav() -> GravitationalProperties:
    return GravitationalProperties(softening_length=0.1)


def _accretion() -> AccretionProperties:
    return AccretionProperties(accretion_radius=0.5, sink_rate=1.0)


# -- AccretionProperties bounds -------------------------------------------------


def test_zero_accretion_radius_rejected() -> None:
    with pytest.raises(ConfigError, match="accretion_radius"):
        AccretionProperties(accretion_radius=0.0)


def test_negative_sink_rate_rejected() -> None:
    with pytest.raises(ConfigError, match="sink_rate"):
        AccretionProperties(accretion_radius=0.5, sink_rate=-1.0)


@pytest.mark.parametrize("porosity", [-0.1, 1.5, 3.7])
def test_porosity_outside_unit_interval_rejected(porosity: float) -> None:
    with pytest.raises(ConfigError, match="porosity"):
        AccretionProperties(accretion_radius=0.5, porosity=porosity)


@pytest.mark.parametrize("porosity", [0.0, 0.5, 1.0])
def test_porosity_in_unit_interval_accepted(porosity: float) -> None:
    AccretionProperties(accretion_radius=0.5, porosity=porosity)


def test_negative_surface_friction_rejected() -> None:
    with pytest.raises(ConfigError, match="k_eta"):
        AccretionProperties(accretion_radius=0.5, k_eta_n=-1.0)


@pytest.mark.parametrize("xi", [-0.1, 1.5])
def test_torque_free_xi_outside_unit_interval_rejected(xi: float) -> None:
    with pytest.raises(ConfigError, match="torque_free_xi"):
        AccretionProperties(accretion_radius=0.5, torque_free_xi=xi)


@pytest.mark.parametrize("xi", [0.0, 0.5, 1.0])
def test_torque_free_xi_in_unit_interval_accepted(xi: float) -> None:
    AccretionProperties(accretion_radius=0.5, torque_free_xi=xi)


def test_torque_free_and_porous_are_mutually_exclusive() -> None:
    with pytest.raises(ConfigError, match="both porous and torque-free"):
        AccretionProperties(
            accretion_radius=0.5, torque_free_xi=1.0, porosity=0.5
        )


def test_torque_free_xi_serializes_into_the_execution_dict() -> None:
    from simbi.simulation.runner import to_execution_dict

    prob = FofcPeriodicBlast.from_cli([])

    def _bodies(self):  # noqa: ANN001
        return [
            ImmersedBodyConfig(
                capability=BodyCapability.ACCRETION | BodyCapability.GRAVITATIONAL,
                mass=1.0,
                velocity=(0.0, 0.0, 0.0),
                position=(0.0, 0.0, 0.0),
                radius=0.0,
                gravitational=_grav(),
                accretion=AccretionProperties(
                    accretion_radius=0.5, torque_free_xi=1.0
                ),
            )
        ]

    import pytest as _pytest

    with _pytest.MonkeyPatch.context() as mp:
        mp.setattr(type(prob), "immersed_bodies", property(_bodies), raising=False)
        exec_dict = to_execution_dict(prob)
    accretion = exec_dict["immersed_bodies"][0]["accretion"]
    assert accretion["torque_free_xi"] == 1.0


# -- capability requires its property block -------------------------------------


def test_accretion_without_block_rejected() -> None:
    with pytest.raises(ConfigError, match="ACCRETION"):
        ImmersedBodyConfig(
            capability=BodyCapability.ACCRETION | BodyCapability.GRAVITATIONAL,
            mass=1.0,
            velocity=(0.0, 0.0, 0.0),
            position=(0.0, 0.0, 0.0),
            radius=0.0,
            gravitational=_grav(),
            accretion=None,
        )


def test_gravitational_without_block_rejected() -> None:
    with pytest.raises(ConfigError, match="GRAVITATIONAL"):
        ImmersedBodyConfig(
            capability=BodyCapability.GRAVITATIONAL,
            mass=1.0,
            velocity=(0.0, 0.0, 0.0),
            position=(0.0, 0.0, 0.0),
            radius=0.0,
            gravitational=None,
        )


# -- unwired capabilities -------------------------------------------------------


@pytest.mark.parametrize(
    "cap",
    [BodyCapability.ELASTIC, BodyCapability.DEFORMABLE],
)
def test_unwired_capability_rejected(cap: BodyCapability) -> None:
    with pytest.raises(ConfigError, match="not wired"):
        ImmersedBodyConfig(
            capability=BodyCapability.GRAVITATIONAL | cap,
            mass=1.0,
            velocity=(0.0, 0.0, 0.0),
            position=(0.0, 0.0, 0.0),
            radius=0.0,
            gravitational=_grav(),
        )


def test_rigid_without_block_rejected() -> None:
    with pytest.raises(ConfigError, match="requires a `rigid` property block"):
        ImmersedBodyConfig(
            capability=BodyCapability.RIGID,
            mass=0.0,
            velocity=(0.0, 0.0, 0.0),
            position=(0.0, 0.0, 0.0),
            radius=0.3,
        )


@pytest.mark.parametrize("k_eta_n", [0.0, -1.0])
def test_rigid_permeable_normal_wall_rejected(k_eta_n: float) -> None:
    # k_eta_n is the normal (no-penetration) rate dial; zero or negative leaves
    # the wall permeable, which is not a rigid boundary.
    with pytest.raises(ConfigError, match="k_eta_n must be > 0"):
        RigidProperties(inertia=1.0, apply_no_slip=True, k_eta_n=k_eta_n)


def test_valid_rigid_obstacle_constructs() -> None:
    ImmersedBodyConfig(
        capability=BodyCapability.RIGID,
        mass=0.0,
        velocity=(0.0, 0.0, 0.0),
        position=(0.0, 0.0, 0.0),
        radius=0.3,
        rigid=RigidProperties(inertia=1.0, apply_no_slip=False),
    )


def test_valid_accretor_constructs() -> None:
    ImmersedBodyConfig(
        capability=BodyCapability.ACCRETION | BodyCapability.GRAVITATIONAL,
        mass=1.0,
        velocity=(0.0, 0.0, 0.0),
        position=(0.0, 0.0, 0.0),
        radius=0.0,
        gravitational=_grav(),
        accretion=_accretion(),
    )


# -- raw dict rejected at the execution-dict boundary ---------------------------


def test_raw_body_dict_rejected(monkeypatch) -> None:
    prob = FofcPeriodicBlast.from_cli([])
    monkeypatch.setattr(
        type(prob),
        "immersed_bodies",
        property(lambda self: [{"mass": 1.0, "capability": 1}]),
        raising=False,
    )
    with pytest.raises(ConfigError, match="ImmersedBodyConfig"):
        to_execution_dict(prob)
