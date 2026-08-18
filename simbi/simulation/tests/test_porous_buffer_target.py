# =============================================================================
# test_porous_buffer_target.py
#
# the porous accretor's reservoir must relax to a static isentrope, at every porosity.
#
# two properties are being protected, and they fail differently.
#
# the sponge must impose no mass current. its inflow variant builds a velocity target from
# the analytic Bondi coefficient lambda = 1/4, which writes the analytic accretion rate into
# the boundary condition of the experiment whose measurement is the accretion rate. the
# static target fixes the reservoir's thermodynamic state and leaves the throughput to be
# selected by the interior — and it is not limiting, supplying a measured 2.13-2.29x the
# Bondi rate against 2.38-2.62x for the inflow variant.
#
# and the target must be the same at every porosity. the claim under test is a step in the
# surface; a boundary condition that changed with porosity would step alongside it, and any
# measured step would be partly attributable to the boundary rather than the surface. that
# failure is silent — every run completes and the profiles look plausible.
# =============================================================================
import pytest

import simbi.expression as expr
from simbi_configs.science.simbi_projects.porous_turbulent_accretor import (
    PorousTurbulentAccretor,
)

# the sweep's endpoints and interior: sealed, partial, and pure drain.
POROSITIES = [0.0, 0.125, 0.5, 1.0]


def velocity_reference(problem: PorousTurbulentAccretor) -> list[float]:
    """the sponge's velocity target sampled inside the damping shell, where kappa > 0.

    the reference is a primitive state, so a zero here is a zero mass current at every
    density the reservoir holds -- the target is kinematic rather than a momentum that
    would have to be read against the local density to know what it prescribes."""
    graph = expr.ExprGraph()
    axes = [expr.variable(f"x{ii}", graph) for ii in (1, 2, 3)]
    outputs = problem.buffer_sponge_terms(*axes)
    assert len(outputs) == 6, "sponge needs [kappa, rho_ref, vel_x..z, pre_ref]"
    return graph.compile(outputs[2:5]).evaluate(x1=0.9, x2=0.0, x3=0.0)


@pytest.mark.parametrize("porosity", POROSITIES)
def test_the_reservoir_imposes_no_mass_current(porosity: float) -> None:
    # checked on the emitted source rather than the model flag: a wire that dropped the
    # setting would leave the flag reading correctly while the run carried a momentum target.
    problem = PorousTurbulentAccretor(porosity=porosity)
    assert problem.buffer_bondi_inflow is False
    velocity = velocity_reference(problem)
    assert all(abs(v) < 1.0e-12 for v in velocity), (
        f"porosity={porosity}: the reservoir carries a velocity target {velocity}, which "
        "prescribes a throughput in the experiment that measures throughput"
    )


def test_the_boundary_condition_does_not_step_with_porosity() -> None:
    # the one that makes the sweep a one-variable experiment.
    targets = {p: PorousTurbulentAccretor(porosity=p).buffer_bondi_inflow for p in POROSITIES}
    assert len(set(targets.values())) == 1, (
        f"the buffer's kinematic target varies across the sweep ({targets}); a step measured "
        "across porosity would then be partly a step in the boundary condition"
    )


def test_the_inflow_variant_is_still_reachable() -> None:
    # the A/B that measured the supply rate has to remain runnable, or the default cannot be
    # re-justified later.
    problem = PorousTurbulentAccretor(porosity=1.0, buffer_bondi_inflow=True)
    assert problem.buffer_bondi_inflow is True
    velocity = velocity_reference(problem)
    assert any(abs(v) > 1.0e-12 for v in velocity), (
        "the inflow variant produced no velocity target, so the A/B compares nothing"
    )
    # and it must point inward: sampled on the +x axis, the x-velocity is negative.
    assert velocity[0] < 0.0, f"the bondi target points outward: {velocity}"


@pytest.mark.parametrize("porosity", POROSITIES)
def test_the_thermodynamic_reference_is_the_isentrope(porosity: float) -> None:
    # the reservoir still has to supply something: density and pressure follow the hydrostatic
    # isentrope, which is what makes it a reservoir rather than a wall.
    problem = PorousTurbulentAccretor(porosity=porosity)
    graph = expr.ExprGraph()
    axes = [expr.variable(f"x{ii}", graph) for ii in (1, 2, 3)]
    outputs = problem.buffer_sponge_terms(*axes)
    radius = 0.9
    (den_ref,) = graph.compile([outputs[1]]).evaluate(x1=radius, x2=0.0, x3=0.0)
    gamma = problem.adiabatic_index
    # the sponge guards its radius against a division by zero at the origin with a fixed
    # 1e-10 offset; the reference is evaluated at that same guarded radius so the comparison
    # is of the profile and not of the guard, which shifts the result by 7e-11 relative here.
    guarded = radius + 1.0e-10
    expected = problem.ambient_density * (
        1.0 + (gamma - 1.0) * problem.central_mass / (problem.ambient_sound_speed**2 * guarded)
    ) ** (1.0 / (gamma - 1.0))
    assert den_ref == pytest.approx(expected, rel=1.0e-12), (
        f"the reservoir's density target is {den_ref}, not the isentrope's {expected}"
    )
