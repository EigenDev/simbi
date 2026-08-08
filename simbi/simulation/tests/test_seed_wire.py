# =============================================================================
# test_seed_wire.py
#
# the python -> rust initial-PERTURBATION wire. the payload is a state DELTA
# carrying one expression per primitive component, read positionally by the
# backend, so a missing or extra component silently shifts every one after it.
#
# also the residual gate that stands in for an explicit momentum projection: the
# backend applies the declared delta and nothing else, so whether the seeded
# state carries a coherent drift or spin is a property of the MODE TABLE. random
# phases make both negligible (measured |L_net|/L_pool = 4.7e-3 and
# |P_net|/P_pool = 1.1e-2 on the composite grid, giving a coherent xi of ~6e-4
# and an S contribution of ~4e-7 against a lock value near unity). a table built
# coherently would break that, and this is what would catch it.
# =============================================================================
import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest

from simbi.simulation.runner import _validate_perturbation_payload

_CONFIG = (
    Path(__file__).resolve().parents[3]
    / "simbi_configs"
    / "science"
    / "simbi_projects"
    / "porous_turbulent_accretor.py"
)


def valid_payload(dim: int = 3, *, isothermal: bool = False) -> dict:
    """the shape a config emits: one expression per evolved primitive component."""
    import simbi.expression as expr

    graph = expr.ExprGraph()
    zero = expr.constant(0.0, graph)
    outputs = [zero] + [expr.variable(f"x{ax + 1}", graph) for ax in range(dim)]
    if not isothermal:
        outputs.append(zero)
    payload = graph.compile(outputs).serialize_equilibrium(dim=dim)
    return {
        "perturbation_expressions": payload,
        "dimensionality": dim,
        "isothermal": isothermal,
        "is_mhd": False,
    }


def test_valid_payload_passes() -> None:
    _validate_perturbation_payload(valid_payload())


def test_absent_perturbation_is_the_common_case() -> None:
    # every config emits an empty payload by default; it must cross untouched.
    _validate_perturbation_payload({})
    _validate_perturbation_payload({"perturbation_expressions": {}})


def test_mhd_is_rejected() -> None:
    bad = valid_payload() | {"is_mhd": True}
    with pytest.raises(ValueError, match="div\\(B\\)"):
        _validate_perturbation_payload(bad)


def test_component_count_must_match_the_regime() -> None:
    bad = valid_payload()
    bad["perturbation_expressions"]["outputs"] = bad["perturbation_expressions"][
        "outputs"
    ][:-1]
    with pytest.raises(ValueError, match="expected 5 primitive components"):
        _validate_perturbation_payload(bad)


def test_isothermal_carries_no_pressure_slot() -> None:
    _validate_perturbation_payload(valid_payload(dim=2, isothermal=True))
    # the same payload on an energy-bearing run is one component short.
    short = valid_payload(dim=2, isothermal=True) | {"isothermal": False}
    with pytest.raises(ValueError, match="expected 4 primitive components"):
        _validate_perturbation_payload(short)


def test_dimension_mismatch_is_rejected() -> None:
    bad = valid_payload(dim=2) | {"dimensionality": 3}
    with pytest.raises(ValueError, match="2-dimensional"):
        _validate_perturbation_payload(bad)


def test_a_non_dict_payload_is_rejected() -> None:
    with pytest.raises(ValueError, match="serialized expression dictionary"):
        _validate_perturbation_payload({"perturbation_expressions": [1, 2, 3]})


def _problem():
    if not _CONFIG.is_file():
        pytest.skip("the porous accretor config is not present in this checkout")
    spec = importlib.util.spec_from_file_location("_pta_wire", _CONFIG)
    module = importlib.util.module_from_spec(spec)
    sys.modules["_pta_wire"] = module
    spec.loader.exec_module(module)
    return module.PorousTurbulentAccretor(
        seed_epsilon=1.0 / 16.0, initial_profile="hydrostatic"
    )


def test_the_seed_carries_no_coherent_drift_or_spin() -> None:
    """the backend applies the declared delta verbatim, so a net linear or angular
    momentum in the seed survives into the run -- and net L is CONSERVED while the
    random pool decays, so a coherent axis only grows in relative terms. random
    phases keep both far below the scale where they could contaminate the support
    measurement; a mode table built with correlated phases or directions would not,
    and would show up here rather than as a slow spin-up mid-run."""
    problem = _problem()
    rows = np.asarray(problem._seed_mode_table())
    onset, width = problem._seed_taper()
    gamma, gm = problem.adiabatic_index, problem.central_mass
    rho_inf, cs_inf = problem.ambient_density, problem.ambient_sound_speed

    def ramp(t):
        t = np.clip(t, 0.0, 1.0)
        return t * t * (3.0 - 2.0 * t), 6.0 * t * (1.0 - t)

    def field(points):
        radius = np.linalg.norm(points, axis=1)
        step, dstep = ramp((radius - onset) / width)
        envelope, d_envelope = 1.0 - step, -dstep / width
        rhat = points / np.maximum(radius, 1e-300)[:, None]
        out = np.zeros_like(points)
        for row in rows:
            kk, ee, amp, phase, r_cut = row[:3], row[3:6], row[6], row[7], row[8]
            ff, dff = envelope.copy(), d_envelope.copy()
            if r_cut > 0.0:
                half = 0.5 * r_cut
                cut, dcut = ramp((radius - half) / half)
                dff = dff * (1.0 - cut) + ff * (-dcut / half)
                ff = ff * (1.0 - cut)
            theta = points @ kk + phase
            out += amp * (
                ff[:, None] * np.cross(kk, ee) * np.cos(theta)[:, None]
                + dff[:, None] * np.cross(rhat, ee) * np.sin(theta)[:, None]
            )
        return out

    # the composite the backend seeds: root plus each level, minus covered cells.
    regions = problem.refinement_regions or []
    box = problem.domain_radius * problem.bondi_radius
    cells = 24  # coarse per level; the ratio is a bulk statistic, not a resolved one
    l_net, l_pool = np.zeros(3), 0.0
    p_net, p_pool = np.zeros(3), 0.0
    for level in range(len(regions) + 1):
        half = box if level == 0 else regions[level - 1][1]
        dx = 2.0 * half / cells
        axis = -half + (np.arange(cells) + 0.5) * dx
        zz, yy, xx = np.meshgrid(axis, axis, axis, indexing="ij")
        points = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)
        if level < len(regions):
            points = points[~np.all(np.abs(points) < regions[level][1], axis=1)]
        radius = np.linalg.norm(points, axis=1)
        rho = rho_inf * (
            1.0 + (gamma - 1.0) * gm / (cs_inf**2 * np.maximum(radius, 1e-12))
        ) ** (1.0 / (gamma - 1.0))
        mass = rho * dx**3
        vel = field(points)
        ang = np.cross(points, vel)
        l_net += (mass[:, None] * ang).sum(axis=0)
        l_pool += (mass * np.linalg.norm(ang, axis=1)).sum()
        p_net += (mass[:, None] * vel).sum(axis=0)
        p_pool += (mass * np.linalg.norm(vel, axis=1)).sum()

    spin = np.linalg.norm(l_net) / l_pool
    drift = np.linalg.norm(p_net) / p_pool
    assert l_pool > 0.0 and p_pool > 0.0, "the seed carries no circulation to measure"
    assert spin < 0.05, (
        f"the seed carries a coherent angular momentum of {spin:.3%} of its "
        "circulation pool; net L is conserved while the pool decays, so this sets a "
        "floor on the measured support"
    )
    assert drift < 0.10, (
        f"the seed carries a coherent drift of {drift:.3%} of its momentum pool, "
        "which acts as an imposed wind on the accretor"
    )
