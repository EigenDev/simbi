# =============================================================================
# test_porous_power_law_column.py
#
# the porous accretor's entropy-stratified power-law start. the branch exists so the
# physics of the initial condition is settled before the run begins, which rests on
# four properties of the emitted state:
# - hydrostatic balance dp/dr = -rho GM/r^2 holds identically
# - the entropy slope of the emitted column is the declared s
# - the density index is derived from s and never declared beside it
# - the reservoir the sponge holds is the same column, so the boundary is passive
# the velocity payload is a symmetry break: solenoidal, and the same tiny fraction of
# the local sound speed at every radius.
# =============================================================================
import importlib.util
import inspect
import math
import sys
from pathlib import Path

import numpy as np
import pytest

import simbi.expression as expr

_CONFIG = (
    Path(__file__).resolve().parents[3]
    / "simbi_configs"
    / "science"
    / "simbi_projects"
    / "porous_turbulent_accretor.py"
)

# the tabulated members of the family at gamma = 5/3: s = 0 is the isentrope, s = 0.47 the
# slope a self-similar convective accretion flow settles on.
SLOPE_INDEX_TABLE = [(0.0, 1.5), (1.0 / 6.0, 1.25), (1.0 / 3.0, 1.0), (0.47, 0.795)]


@pytest.fixture(scope="module")
def porous_module():
    if not _CONFIG.exists():
        pytest.skip("science config tree not present")
    spec = importlib.util.spec_from_file_location("porous_power_law", _CONFIG)
    module = importlib.util.module_from_spec(spec)
    sys.modules["porous_power_law"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def porous_cls(porous_module):
    (cls,) = [
        c
        for _, c in inspect.getmembers(porous_module, inspect.isclass)
        if c.__module__ == "porous_power_law"
    ]
    return cls


def _problem(cls, **kwargs):
    """a derived sealed-surface problem on the power-law start. the sealed branch retains
    its gas by construction, so it derives without tripping the loss-cone premise a
    draining surface has to satisfy."""
    kwargs.setdefault("porosity", 0.0)
    kwargs.setdefault("initial_profile", "power_law")
    problem = cls(**kwargs)
    problem.setup()
    return problem


def _column(problem):
    """the emitted column as a callable r -> (rho, p), evaluated through the traced graph
    the backend receives."""
    graph = expr.ExprGraph()
    x = expr.variable("x1", graph)
    y = expr.variable("x2", graph)
    z = expr.variable("x3", graph)
    radius = expr.sqrt(x * x + y * y + z * z)
    primitives = problem._column_primitives(graph, radius)
    compiled = graph.compile([primitives[0], primitives[-1]])

    def state(r: float) -> tuple[float, float]:
        rho, pre = compiled.evaluate(x1=r, x2=0.0, x3=0.0)
        return rho, pre

    return state


# radii spanning the science window and the reservoir, in code units where R_B = 1.
SAMPLE_RADII = (0.05, 0.2, 0.5, 1.0, 1.3)


@pytest.mark.parametrize("slope,index", SLOPE_INDEX_TABLE)
def test_the_density_index_follows_from_the_entropy_slope(porous_cls, slope, index):
    # n = (1 - s)/(gamma - 1) is the only index at which a power law solves the balance.
    problem = _problem(porous_cls, entropy_slope=slope)
    assert problem.power_law_index == pytest.approx(index, rel=1e-12)


@pytest.mark.parametrize("slope", [0.0, 1.0 / 6.0, 1.0 / 3.0, 0.47])
def test_the_emitted_column_is_in_hydrostatic_balance(porous_cls, slope):
    problem = _problem(porous_cls, entropy_slope=slope)
    state = _column(problem)
    for r in SAMPLE_RADII:
        h = r * 1.0e-6
        (rho, _), (_, p_up), (_, p_dn) = state(r), state(r + h), state(r - h)
        weight = rho * problem.central_mass / r**2
        residual = (p_up - p_dn) / (2.0 * h) + weight
        assert abs(residual) / weight < 1.0e-8, f"balance broken at r = {r}"


@pytest.mark.parametrize("slope", [0.0, 1.0 / 6.0, 1.0 / 3.0, 0.47])
def test_the_emitted_column_carries_the_declared_entropy_slope(porous_cls, slope):
    problem = _problem(porous_cls, entropy_slope=slope)
    gamma = problem.adiabatic_index
    state = _column(problem)

    def log_entropy(r: float) -> float:
        rho, pre = state(r)
        return math.log(pre) - gamma * math.log(rho)

    for r in SAMPLE_RADII:
        h = r * 1.0e-6
        measured = (log_entropy(r + h) - log_entropy(r - h)) / (
            math.log(r + h) - math.log(r - h)
        )
        assert measured == pytest.approx(-slope, abs=1.0e-8)


@pytest.mark.parametrize("slope", [0.0, 1.0 / 6.0, 0.47])
def test_the_column_is_virial_at_every_radius(porous_cls, slope):
    # the pressure coefficient is forced by the balance, so cs^2 = gamma GM/((n + 1) r)
    # and the sound speed tracks the keplerian speed everywhere on the family.
    problem = _problem(porous_cls, entropy_slope=slope)
    gamma = problem.adiabatic_index
    index = problem.power_law_index
    state = _column(problem)
    for r in SAMPLE_RADII:
        rho, pre = state(r)
        expected = gamma * problem.central_mass / ((index + 1.0) * r)
        assert gamma * pre / rho == pytest.approx(expected, rel=1.0e-12)


def test_the_float_column_transcribes_the_traced_one(porous_cls):
    # the summary and the cell generator read the float twin; a drift between the two would
    # report one atmosphere while running another.
    problem = _problem(porous_cls, entropy_slope=1.0 / 3.0)
    state = _column(problem)
    for r in SAMPLE_RADII:
        traced, floats = state(r), problem.column_state(r)
        assert floats[0] == pytest.approx(traced[0], rel=1.0e-14)
        assert floats[1] == pytest.approx(traced[1], rel=1.0e-14)


def test_the_reservoir_holds_the_starting_column(porous_cls):
    # a sponge holding a different profile from the one the interior starts in drives the
    # outer shell from the first step, which is the standing force this branch exists to
    # exclude.
    problem = _problem(porous_cls, entropy_slope=1.0 / 6.0)
    graph = expr.ExprGraph()
    axes = [expr.variable(f"x{ax + 1}", graph) for ax in range(3)]
    terms = problem.buffer_sponge_terms(*axes)
    compiled = graph.compile([terms[1], terms[-1]])
    gamma = problem.adiabatic_index
    state = _column(problem)
    buffer_radius = problem.buffer_parameters["buffer_radius"]
    for r in (buffer_radius, 1.05 * buffer_radius, 1.2 * buffer_radius):
        rho_ref, nrg_ref = compiled.evaluate(x1=r, x2=0.0, x3=0.0)
        rho, pre = state(r)
        # the sponge's radius carries a 1e-10 regularization against the origin, which the
        # column's does not; at the buffer radius that is a 2e-10 relative offset in
        # density, and the tolerance sits an order and a half above it.
        assert rho_ref == pytest.approx(rho, rel=1.0e-8)
        assert nrg_ref == pytest.approx(pre / (gamma - 1.0), rel=1.0e-8)


def test_a_declared_density_index_is_refused(porous_cls):
    # the two slopes name one quantity; accepting both would let them drift apart.
    with pytest.raises(ValueError, match="derives its density"):
        _problem(porous_cls, entropy_slope=1.0 / 6.0, initial_index=1.4)


def test_the_loss_cone_seed_is_refused(porous_cls):
    # the instability supplies its own driving, so the amplitude on this branch is set by
    # linearity rather than by the loss cone.
    with pytest.raises(ValueError, match="seed_epsilon requires"):
        _problem(porous_cls, seed_epsilon=0.05)


def test_the_entropy_slope_keys_the_output_path(porous_cls, tmp_path, monkeypatch):
    # two slopes are two experiments: the slope fixes the density index, the schwarzschild
    # discriminant and the growth rate.
    monkeypatch.setenv("SCRATCH", str(tmp_path))
    first = _problem(porous_cls, entropy_slope=1.0 / 6.0).data_directory
    second = _problem(porous_cls, entropy_slope=0.47).data_directory
    assert first != second
    assert "power_law_s0.47" in str(second)


def _symmetry_break_velocity(module, problem):
    """the emitted symmetry break as a callable (x, y, z) -> (v1, v2, v3), rebuilt from the
    same pieces the perturbation expression is assembled from."""
    graph = expr.ExprGraph()
    axes = [expr.variable(f"x{ax + 1}", graph) for ax in range(3)]
    radius = expr.sqrt(
        sum(a * a for a in axes) + expr.constant(1.0e-300, graph)
    )
    onset, width = problem._seed_taper()
    step, dstep = module._smoothstep(graph, radius, onset, onset + width)
    envelope = expr.constant(1.0, graph) - step
    d_envelope = expr.constant(0.0, graph) - dstep
    index = problem.power_law_index
    r_0, _, _ = problem.power_law_reference
    cs_0 = math.sqrt(
        problem.adiabatic_index * problem.central_mass / ((index + 1.0) * r_0)
    )
    amplitude = expr.constant(problem.symmetry_break_mach * cs_0, graph)
    scaled = radius * expr.constant(1.0 / r_0, graph)
    shape = scaled ** (-0.5)
    d_shape = expr.constant(-0.5 / r_0, graph) * scaled ** (-1.5)
    gain = (
        amplitude * shape * envelope,
        amplitude * (d_shape * envelope + shape * d_envelope),
    )
    velocity = problem._curl_of_modes(
        graph, axes, radius, problem._symmetry_break_modes(), gain
    )
    compiled = graph.compile(velocity)

    def field(point) -> np.ndarray:
        return np.array(compiled.evaluate(x1=point[0], x2=point[1], x3=point[2]))

    return field


def test_the_symmetry_break_is_solenoidal(porous_module, porous_cls):
    # the field is written as a curl, so it carries no compression under any radial
    # envelope and the seed launches no sound.
    problem = _problem(porous_cls)
    field = _symmetry_break_velocity(porous_module, problem)
    rng = np.random.default_rng(11)
    for radius in (0.1, 0.4, 0.9):
        for _ in range(8):
            direction = rng.normal(size=3)
            point = radius * direction / np.linalg.norm(direction)
            h = radius * 1.0e-5
            divergence = 0.0
            magnitude = 0.0
            for axis in range(3):
                offset = np.zeros(3)
                offset[axis] = h
                up, down = field(point + offset), field(point - offset)
                divergence += (up[axis] - down[axis]) / (2.0 * h)
                magnitude += abs(up[axis] - down[axis]) / (2.0 * h)
            assert abs(divergence) < 1.0e-6 * magnitude


def test_the_symmetry_break_is_mach_anchored(porous_module, porous_cls):
    # the rms rides the local sound speed, so the linear phase is entered at the same
    # amplitude at every radius rather than at the same absolute speed.
    problem = _problem(porous_cls)
    field = _symmetry_break_velocity(porous_module, problem)
    index = problem.power_law_index
    gamma = problem.adiabatic_index
    rng = np.random.default_rng(3)
    for radius in (0.1, 0.4, 0.9):
        samples = []
        for _ in range(256):
            direction = rng.normal(size=3)
            samples.append(field(radius * direction / np.linalg.norm(direction)))
        rms = math.sqrt(float(np.mean(np.sum(np.asarray(samples) ** 2, axis=1))))
        sound_speed = math.sqrt(
            gamma * problem.central_mass / ((index + 1.0) * radius)
        )
        assert rms / sound_speed == pytest.approx(problem.symmetry_break_mach, rel=0.5)
