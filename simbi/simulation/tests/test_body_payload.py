# =============================================================================
# test_body_payload.py
#
# the immersed-body serialization ssot: `ImmersedBodyConfig.to_backend` /
# `GravitationalSystemConfig.to_backend` / `body_payload` emit exactly the dict
# tree the rust `BodyParams` reader consumes. the backend reads every key with an
# unwrap_or default, so a silently dropped or renamed key becomes a wrong-physics
# default with no error raised -- these tests pin the key set so that such drift
# fails loudly here.
# =============================================================================
import math

import pytest

from simbi.simulation.problem import ConfigError
from simbi.types.bodies import (
    AccretionProperties,
    BinaryComponentConfig,
    BinaryConfig,
    BodyCapability,
    GravitationalProperties,
    GravitationalSystemConfig,
    ImmersedBodyConfig,
    RigidProperties,
    body_payload,
)
from simbi.types.shape import Shape

# the exact keys rust `parse_bodies` reads from each body dict (symbi-py BodyParams):
# top-level scalars + the three nested property groups. pinned so a config-side
# rename cannot silently drift into a backend default.
_TOP_LEVEL_KEYS = {
    "capability",
    "mass",
    "radius",
    "position",
    "velocity",
    "two_way_coupling",
}
_RIGID_KEYS = {
    "inertia",
    "apply_no_slip",
    "k_eta_n",
    "k_eta_t",
    "shape",
    "omega",
    "spin_axis",
    "inertia_principal",
}
_ACCRETION_KEYS = {
    "accretion_radius",
    "sink_rate",
    "porosity",
    "torque_free_xi",
}


def _rigid_body() -> ImmersedBodyConfig:
    # a spinning rigid shaped wall (omega != 0 requires a shape).
    return ImmersedBodyConfig(
        capability=BodyCapability.RIGID,
        mass=1.0,
        velocity=(0.0, 0.0, 0.0),
        position=(1.0, 0.5, 0.0),
        radius=0.3,
        rigid=RigidProperties(
            inertia=1.0,
            apply_no_slip=True,
            shape=Shape.box((0.0, 0.0, 0.0), (0.5, 0.2, 1.0)),
            omega=2.0,
            spin_axis=(0.0, 0.0, 1.0),
            inertia_principal=(1.0, 2.0, 3.0),
        ),
    )


def _accretor() -> ImmersedBodyConfig:
    return ImmersedBodyConfig(
        capability=BodyCapability.ACCRETION,
        mass=1.0,
        velocity=(0.0, 0.0, 0.0),
        position=(0.0, 0.0, 0.0),
        radius=0.05,
        accretion=AccretionProperties(
            accretion_radius=0.1, porosity=0.3
        ),
    )


def test_rigid_to_backend_emits_the_backend_key_tree() -> None:
    wire = _rigid_body().to_backend()
    assert _TOP_LEVEL_KEYS <= set(wire)
    assert int(wire["capability"]) == int(BodyCapability.RIGID)
    assert _RIGID_KEYS <= set(wire["rigid"])
    # the CSG shape crosses as `rigid.shape.wire` (SdfExpr json the backend reads).
    assert wire["rigid"]["shape"]["wire"]["kind"] == "box"
    assert wire["rigid"]["omega"] == 2.0
    assert wire["rigid"]["inertia_principal"] == (1.0, 2.0, 3.0)


def test_accretion_to_backend_emits_the_accretion_group() -> None:
    wire = _accretor().to_backend()
    assert _ACCRETION_KEYS <= set(wire["accretion"])
    assert wire["accretion"]["accretion_radius"] == 0.1
    assert wire["accretion"]["porosity"] == 0.3
    assert wire["accretion"]["torque_free_xi"] is None


def test_spin_axis_is_normalized_before_serialization() -> None:
    body = ImmersedBodyConfig(
        capability=BodyCapability.RIGID,
        mass=1.0,
        velocity=(0.0, 0.0, 0.0),
        position=(0.0, 0.0, 0.0),
        radius=0.3,
        rigid=RigidProperties(
            inertia=1.0,
            apply_no_slip=False,
            shape=Shape.sphere((0.0, 0.0, 0.0), 1.0),
            omega=1.0,
            spin_axis=(0.0, 3.0, 4.0),  # |.| = 5
        ),
    )
    axis = body.to_backend()["rigid"]["spin_axis"]
    assert math.isclose(math.sqrt(sum(a * a for a in axis)), 1.0)
    assert math.isclose(axis[1], 0.6) and math.isclose(axis[2], 0.8)


def test_body_payload_composes_present_kinds_and_omits_absent() -> None:
    empty = body_payload(None, [])
    assert empty == {}

    only_bodies = body_payload(None, [_rigid_body(), _accretor()])
    assert set(only_bodies) == {"immersed_bodies"}
    assert len(only_bodies["immersed_bodies"]) == 2

    system = GravitationalSystemConfig(
        prescribed_motion=True,
        reference_frame="com",
        system_type="binary",
        binary_config=BinaryConfig(
            semi_major=1.0,
            eccentricity=0.0,
            mass_ratio=1.0,
            total_mass=2.0,
            components=[
                BinaryComponentConfig(
                    mass=1.0,
                    radius=0.1,
                    is_an_accretor=True,
                    softening_length=0.05,
                    two_way_coupling=False,
                    accretion_radius=0.1,
                ),
                BinaryComponentConfig(
                    mass=1.0,
                    radius=0.1,
                    is_an_accretor=True,
                    softening_length=0.05,
                    two_way_coupling=False,
                    accretion_radius=0.1,
                ),
            ],
        ),
    )
    both = body_payload(system, [_accretor()])
    assert set(both) == {"body_system", "immersed_bodies"}
    assert both["body_system"]["system_type"] == "binary"
    assert len(both["body_system"]["binary_config"]["components"]) == 2


def test_body_payload_rejects_a_raw_dict_body() -> None:
    # a raw dict bypasses every field validation; the backend would read it through
    # silent unwrap_or defaults. the ssot refuses it loudly.
    with pytest.raises(ConfigError, match="not an\n?.*ImmersedBodyConfig|ImmersedBodyConfig"):
        body_payload(None, [{"capability": BodyCapability.RIGID, "mass": 1.0}])
