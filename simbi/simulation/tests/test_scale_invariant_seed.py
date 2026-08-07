# =============================================================================
# test_scale_invariant_seed.py
#
# the scale-invariant velocity seed's DESIGN PROPERTY, measured on the mode table
# the config emits.
#
# the seed exists to remove a scale mismatch: a band-limited seed lives at
# wavelengths >= r_0/16, while convection at radius r needs perturbations at scale
# ~ r, so a deep sink sits in gas the seed never reached. the cure is one octave
# per level with per-octave rms v = eps v_K(lambda) -- the E(k) ~ k^0 spectrum
# anchored at every scale rather than normalized once. paired with the isentrope
# (where cs^2 -> (gamma-1) GM/r), the dimensionless state
#
#     xi = v_perp sqrt(r / GM)  and  Ma = v / cs
#
# is then the SAME in every radial decade, so an ignition outcome cannot depend on
# r_acc/R_B -- which is the property that makes one endpoint pair sufficient and a
# ladder protocol unnecessary.
#
# these gates measure that invariance directly (xi(r) and Ma(r) flat across the
# science window and beyond) rather than checking the amplitude law that is
# supposed to produce it, because the law holding while the field is not
# scale-free is exactly the failure worth catching.
# =============================================================================

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest

_CONFIG = (
    Path(__file__).resolve().parents[3]
    / "simbi_configs"
    / "science"
    / "simbi_projects"
    / "porous_turbulent_accretor.py"
)

_EPSILON = 1.0 / 16.0
# the science window is 5-35 r_acc; the invariance claim is broader than the window
# it is used in, so it is measured over the two decades the taper admits.
_SCALES = (5.0, 10.0, 20.0, 35.0, 70.0, 150.0, 300.0)
_SAMPLES = 3000


def _problem():
    if not _CONFIG.is_file():
        pytest.skip("the porous accretor config is not present in this checkout")
    spec = importlib.util.spec_from_file_location("_pta_seed", _CONFIG)
    module = importlib.util.module_from_spec(spec)
    sys.modules["_pta_seed"] = module
    spec.loader.exec_module(module)
    return module.PorousTurbulentAccretor(
        seed_epsilon=_EPSILON, initial_profile="hydrostatic"
    )


def _ramp(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t), 6.0 * t * (1.0 - t)


def _evaluate(problem, points: np.ndarray) -> np.ndarray:
    """the seed field v = curl(f(r) A) at `points`, mirroring the backend's
    evaluation of the mode table (symbi-py `SeedField::eval`)."""
    rows = np.asarray(problem.seed_modes)
    onset, width = problem.seed_taper
    radius = np.linalg.norm(points, axis=1)
    step, dstep = _ramp((radius - onset) / width)
    envelope, d_envelope = 1.0 - step, -dstep / width
    rhat = points / np.maximum(radius, 1e-300)[:, None]
    field = np.zeros_like(points)
    for row in rows:
        kk, ee, amp, phase, r_cut = row[:3], row[3:6], row[6], row[7], row[8]
        ff, dff = envelope.copy(), d_envelope.copy()
        if r_cut > 0.0:
            half = 0.5 * r_cut
            cut, dcut = _ramp((radius - half) / half)
            dff = dff * (1.0 - cut) + ff * (-dcut / half)
            ff = ff * (1.0 - cut)
        theta = points @ kk + phase
        field += amp * (
            ff[:, None] * np.cross(kk, ee) * np.cos(theta)[:, None]
            + dff[:, None] * np.cross(rhat, ee) * np.sin(theta)[:, None]
        )
    return field


def _shell(rng, radius: float) -> np.ndarray:
    directions = rng.normal(size=(_SAMPLES, 3))
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    return directions * radius


def _sound_speed(problem, radius: float) -> float:
    """the isentropic hydrostatic atmosphere's sound speed at `radius`."""
    gamma = problem.adiabatic_index
    cs_inf, rho_inf = problem.ambient_sound_speed, problem.ambient_density
    rho = rho_inf * (
        1.0 + (gamma - 1.0) * problem.central_mass / (cs_inf**2 * radius)
    ) ** (1.0 / (gamma - 1.0))
    pre = (rho_inf * cs_inf**2 / gamma) * (rho / rho_inf) ** gamma
    return math.sqrt(gamma * pre / rho)


def test_specific_angular_momentum_is_radius_independent() -> None:
    """xi = |r x v| / sqrt(GM r) is the theory's own dimensionless variable, and a
    scale-free seed carries the same value in every decade. a band-limited seed
    instead delivers xi rising with radius and essentially nothing inside the
    science window, which is the mismatch this design removes."""
    problem = _problem()
    r_acc = problem.accretor_radius
    rng = np.random.default_rng(11)
    values = []
    for scale in _SCALES:
        radius = scale * r_acc
        points = _shell(rng, radius)
        angular = np.cross(points, _evaluate(problem, points))
        values.append(
            math.sqrt(np.mean(np.sum(angular**2, axis=1)))
            / math.sqrt(problem.central_mass * radius)
        )
    values = np.array(values)
    spread = np.ptp(values) / values.mean()
    assert spread < 0.15, (
        f"xi varies by {spread:.1%} across {_SCALES[0]:g}-{_SCALES[-1]:g} r_acc "
        f"({np.array2string(values, precision=4)}); the seed is not scale-free, so "
        "an ignition outcome could depend on r_acc/R_B"
    )


def test_seed_mach_number_follows_the_atmosphere_at_fixed_amplitude() -> None:
    """the seed holds v/v_K fixed, so its LOCAL mach number is the product of that
    constant with the atmosphere's own v_K/cs. for the isentrope
    cs^2 = cs_inf^2 + (gamma-1) GM/r exactly, giving

        v_K / cs = sqrt(1 / (r + (gamma-1) R_B)),

    which is flat only deep inside the Bondi radius and falls off as the atmosphere
    relaxes toward the ambient state -- 20 percent between 5 and 300 r_acc here.
    asserting a flat mach profile would therefore be asserting a property of the
    atmosphere, so the gate gives the seed exactly one free number (its amplitude)
    and requires the whole measured profile to follow from it.

    the seed also has to stay WEAK everywhere: the claim is that convection makes
    the turbulence, which a seed that steepens on arrival would supply instead."""
    problem = _problem()
    gamma, r_acc = problem.adiabatic_index, problem.accretor_radius
    r_bondi, cs_inf = problem.bondi_radius, problem.ambient_sound_speed
    rng = np.random.default_rng(5)

    radii = np.array([scale * r_acc for scale in _SCALES])
    mach, predicted = [], []
    for radius in radii:
        field = _evaluate(problem, _shell(rng, radius))
        speed = math.sqrt(np.mean(np.sum(field**2, axis=1)))
        mach.append(speed / _sound_speed(problem, radius))
        predicted.append(
            math.sqrt(problem.central_mass / radius) / _sound_speed(problem, radius)
        )
    mach, predicted = np.array(mach), np.array(predicted)

    assert mach.max() < 0.35, (
        f"the seed reaches mach {mach.max():.3f}; a seed that steepens on arrival "
        "supplies the turbulence instead of letting convection make it"
    )
    # the atmosphere factor must actually VARY over the sampled range, or the test
    # would pass on a flat profile and prove nothing about the decomposition.
    assert np.ptp(predicted) / predicted.mean() > 0.1, (
        "the sampled radii see an essentially constant v_K/cs, so this gate cannot "
        "distinguish a scale-free seed from a merely flat one -- extend the range"
    )
    # exact closed form for the isentrope's v_K/cs, independent of the sampler.
    closed_form = np.sqrt(1.0 / (radii + (gamma - 1.0) * r_bondi * cs_inf**2))
    assert np.abs(predicted / closed_form - 1.0).max() < 1.0e-12

    amplitude = float(np.mean(mach / predicted))
    residual = np.abs(mach / (amplitude * predicted) - 1.0).max()
    assert residual < 0.1, (
        f"the mach profile departs from (constant amplitude {amplitude:.4f}) x "
        f"(v_K/cs) by {residual:.1%}: measured {np.array2string(mach, precision=4)} "
        f"against {np.array2string(amplitude * predicted, precision=4)}. the seed "
        "carries a radial trend of its own"
    )


def test_amplitude_scales_linearly_with_epsilon() -> None:
    """eps is the seed's single knob: the field is linear in it, so an ignition
    threshold found by scanning eps is a statement about one number."""
    problem = _problem()
    half = type(problem)(seed_epsilon=0.5 * _EPSILON, initial_profile="hydrostatic")
    rng = np.random.default_rng(2)
    points = _shell(rng, 20.0 * problem.accretor_radius)
    full_speed = np.linalg.norm(_evaluate(problem, points), axis=1).mean()
    half_speed = np.linalg.norm(_evaluate(half, points), axis=1).mean()
    assert abs(half_speed / full_speed - 0.5) < 1.0e-12, (
        f"halving eps scaled the field by {half_speed / full_speed:.6f}"
    )


def test_mode_table_is_deterministic_and_well_formed() -> None:
    """the same seed integer must give the same realization -- a run's initial
    condition has to be reproducible from its config alone -- and every mode must
    be transverse (k . e = 0), or the field the backend builds as a curl carries a
    compressive part the construction claims it does not."""
    problem = _problem()
    again = type(problem)(seed_epsilon=_EPSILON, initial_profile="hydrostatic")
    first, second = np.asarray(problem.seed_modes), np.asarray(again.seed_modes)
    assert first.shape == second.shape
    assert np.array_equal(first, second), "the mode table is not reproducible"

    other = type(problem)(
        seed_epsilon=_EPSILON, initial_profile="hydrostatic", turb_seed=43
    )
    assert not np.array_equal(first, np.asarray(other.seed_modes)), (
        "turb_seed does not change the realization"
    )

    wave = first[:, :3]
    direction = first[:, 3:6]
    k_hat = wave / np.linalg.norm(wave, axis=1)[:, None]
    assert np.abs(np.einsum("ij,ij->i", k_hat, direction)).max() < 1.0e-12
    assert np.abs(np.linalg.norm(direction, axis=1) - 1.0).max() < 1.0e-12


def test_seed_requires_the_stratified_start() -> None:
    """on a constant medium the same seed is mach 0.12 at the outer boundary and
    supersonic near the accretor, because cs no longer tracks v_K. the config must
    refuse that pairing rather than run a seed whose weakness is radius-dependent."""
    problem = _problem()
    with pytest.raises(Exception, match="hydrostatic"):
        type(problem)(seed_epsilon=_EPSILON, initial_profile="uniform")
