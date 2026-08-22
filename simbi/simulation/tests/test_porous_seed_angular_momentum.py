# =============================================================================
# test_porous_seed_angular_momentum.py
#
# the scale-invariant seed must break the symmetry without spinning the cloud.
#
# the experiment asks what a NON-ROTATING atmosphere does when a surface drains it, so
# net angular momentum in the initial data is a second physical ingredient nobody
# declared. a random solenoidal field carries none in expectation, but one realization
# keeps a residual falling only as the square root of the mode count, and that residual
# is a global rotation that survives for the whole run: a sealed free-slip surface in a
# point-mass field exerts no torque, so nothing removes it.
#
# it fails silently. the flow still looks turbulent, the profiles still look plausible,
# and the measured angular-momentum support carries a spin the seed put there rather than
# one the dynamics built.
#
# the correction is the one field that carries angular momentum and no shear, so every
# eddy is left intact. it wears the seed's own envelope, which keeps it inside the seeded
# region, and that costs no solenoidality because `Omega x r` is orthogonal to `r`:
#
#     div(f(r) (Omega x r)) = f'(r) rhat . (Omega x r) = 0.
# =============================================================================
import numpy as np
import pytest

from simbi_configs.science.projects.porous_turbulent_accretor import (
    PorousTurbulentAccretor,
)

# realizations: the residual is a random draw, so a single seed proves nothing about
# the next one.
TURB_SEEDS = [1, 7, 42]
# the two columns whose sound speed tracks v_K, which is what admits the seed at all.
COLUMNS = ["power_law", "hydrostatic"]


def _problem(profile: str, turb_seed: int, eps: float = 0.25):
    flags = [
        "--porosity", "1",
        "--no-pilot",
        "--initial-profile", profile,
        "--seed-epsilon", str(eps),
        "--turb-seed", str(turb_seed),
    ]
    if profile == "power_law":
        flags += ["--entropy-slope", "0.1666666666666667"]
    else:
        flags += ["--initial-index", "1.25"]
    return PorousTurbulentAccretor.from_cli(flags)


@pytest.mark.parametrize("profile", COLUMNS)
@pytest.mark.parametrize("turb_seed", TURB_SEEDS)
def test_the_corrected_seed_carries_no_net_angular_momentum(profile, turb_seed):
    """|L| after correction is machine zero against the realization's own residual."""
    p = _problem(profile, turb_seed)
    raw = p.seed_net_angular_momentum(corrected=False)
    fixed = p.seed_net_angular_momentum(corrected=True)
    raw_mag = float(np.linalg.norm(raw))

    # the gate reports its own irrelevance: a realization that happened to arrive with no
    # net angular momentum would pass the assertion below while testing nothing.
    assert raw_mag > 1.0e-6, (
        f"seed {turb_seed} on {profile} carries |L| = {raw_mag:e} before correction; "
        "the correction has nothing to remove and this case is vacuous"
    )
    assert float(np.linalg.norm(fixed)) < 1.0e-12 * raw_mag, (
        f"corrected |L| = {np.linalg.norm(fixed):e} against a residual of {raw_mag:e}"
    )


@pytest.mark.parametrize("profile", COLUMNS)
def test_the_correction_removes_rotation_and_leaves_the_eddies(profile):
    """the delivered xi is the seed's, not the correction's.

    solid-body rotation carries no shear, so subtracting it must not move the local
    specific angular momentum the loss-cone amplitude is chosen against. a correction
    that moved xi would be removing turbulence rather than spin."""
    p = _problem(profile, turb_seed=7)
    xi = p._seed_xi_rms()
    omega = np.asarray(p.seed_solid_body_correction())
    # the correction's own rotational speed at the seeded edge, against the seed's.
    onset, _ = p._seed_taper()
    spin_speed = float(np.linalg.norm(omega)) * onset
    seed_speed = xi * np.sqrt(p.central_mass / onset)
    assert spin_speed < 0.05 * seed_speed, (
        f"the solid-body correction moves at {spin_speed:e} against a seed of "
        f"{seed_speed:e}: it is removing structure rather than spin"
    )
    assert xi > 0.0, "the seed delivers no specific angular momentum at all"


@pytest.mark.parametrize("profile", COLUMNS)
def test_the_corrected_field_stays_divergence_free(profile):
    """`div(f(r)(Omega x r)) = 0` analytically; this checks the emitted field numerically.

    the seed is built as a curl so it is solenoidal by construction, and the correction is
    solenoidal by the orthogonality above. a sampled divergence catches a wiring error that
    the algebra cannot."""
    p = _problem(profile, turb_seed=7)
    table = p._seed_mode_table()
    omega = np.asarray(p.seed_solid_body_correction())
    onset, _ = p._seed_taper()
    h = onset * 1.0e-4
    rng = np.random.default_rng(3)
    pts = rng.normal(size=(64, 3))
    pts *= (0.4 * onset) / np.linalg.norm(pts, axis=1)[:, None]

    def corrected(q):
        v = p._sample_seed_velocity(q, table)
        rad = np.linalg.norm(q, axis=1)
        env = p.seed_envelope(rad)[:, None]
        spin = np.cross(np.broadcast_to(omega, q.shape), q)
        return v - env * spin

    div = np.zeros(len(pts))
    for ax in range(3):
        step = np.zeros(3)
        step[ax] = h
        div += (corrected(pts + step)[:, ax] - corrected(pts - step)[:, ax]) / (2.0 * h)
    # scale by the field's own velocity gradient, so this is a relative statement
    scale = np.linalg.norm(corrected(pts), axis=1).mean() / (0.4 * onset)
    assert np.max(np.abs(div)) < 1.0e-4 * scale, (
        f"max |div v| = {np.max(np.abs(div)):e} against a gradient scale of {scale:e}"
    )


@pytest.mark.parametrize("porosity", ["0", "1"])
def test_the_seeded_power_law_column_receives_the_scale_invariant_seed(porosity):
    """the perturbation dispatch follows the seed rather than the profile's name.

    the admission gate and the perturbation dispatch guard the same doorway from two
    sides, and they fail worst when they disagree: a power-law run with seed_epsilon
    dialed up would carry /eps in its path while laying down the zero-angular-momentum
    symmetry break underneath, and the free-fall outcome would read as a null result of
    an experiment the run never performed. the serialized wires are deterministic under
    turb_seed, so equality against the seed builder is exact."""
    p = _problem("power_law", turb_seed=7, eps=0.25)
    assert p.perturbation_expressions == p._seed_expression()

    # the seedless power-law column keeps its linear symmetry break, so the in-flight
    # runs' semantics are untouched by the doorway widening.
    flags = [
        "--porosity", porosity, "--no-pilot",
        "--initial-profile", "power_law",
        "--entropy-slope", "0.1666666666666667",
    ]
    bare = PorousTurbulentAccretor.from_cli(flags)
    assert bare.perturbation_expressions == bare._symmetry_break_expression()


def test_the_seeded_power_law_path_is_its_own_directory(tmp_path, monkeypatch):
    """the seeded experiment writes beside the seedless one rather than over it.

    the /eps segment separates the two series, and the index it names is the column's
    operative density slope -- on the power-law profile that is the derived
    n = (1 - s)/(gamma - 1) rather than the initial_index knob the profile ignores."""
    monkeypatch.setenv("SCRATCH", str(tmp_path))
    seeded = _problem("power_law", turb_seed=7, eps=0.25)
    seeded.setup()
    bare_flags = [
        "--porosity", "1", "--no-pilot",
        "--initial-profile", "power_law",
        "--entropy-slope", "0.1666666666666667",
    ]
    bare = PorousTurbulentAccretor.from_cli(bare_flags)
    bare.setup()
    assert seeded.data_directory != bare.data_directory
    assert "/eps0.25_n1.25/" in str(seeded.data_directory) + "/"
