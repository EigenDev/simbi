# =============================================================================
# test_buffer_bondi_inflow.py
#
# what the damping shell supplies to the science window.
#
# the porous accretor's damping shell begins at r_0 = (2/3) R_B. that is close enough to
# the bondi radius that the steady solution there carries a real inward velocity: mass
# conservation through a sphere gives
#
#     v(r) = lambda rho_inf R_B^2 cs / (rho_ref r^2),
#
# which is 0.199 cs at r_0 for gamma = 5/3. relaxing the shell's momentum to zero holds
# that inflow at rest, which raises the question this measures: whether the reservoir the
# interior draws from is instead a wall it has to push through.
#
# the quantity is the net radial mass flux crossing r = r_0, the total supply the science
# window receives. it has a closed-form scale to be read against -- the bondi rate
# 4 pi lambda rho_inf R_B^2 cs, equal to pi in these units at gamma = 5/3 -- and it settles
# to a constant by about two bondi times, so the steady value is what is compared rather
# than any point in the initial collapse transient.
#
# the grid is coarse and unrefined on purpose. r_0 sits at two thirds of the domain radius
# and is resolved on the root grid; the ladder exists to resolve the sink, which this
# boundary measurement does not touch. that makes the whole comparison a few tens of
# seconds rather than cluster time.
# =============================================================================

import importlib.util
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

pytestmark = pytest.mark.simulation

_BASE_RESOLUTION = 20
# the flux at r_0 is flat to four digits from about two bondi times onward, once the
# collapse of the uniform initial state off the point mass has passed through the boundary.
_BONDI_TIMES = 2.5
# the step budget that buys that. the timestep is set by the source CFL against the point
# mass rather than by the sound crossing, so it does not follow from the cell width; the
# runs are asserted below to have actually reached the time and the steadiness they need.
_MAX_STEPS = 2400
_MIN_BONDI_TIMES = 2.4
# two consecutive samples this close are on the plateau rather than still in the transient.
_STEADY_TOL = 0.01


def _problem_class():
    if not _CONFIG.is_file():
        pytest.skip("the porous accretor config is not present in this checkout")
    spec = importlib.util.spec_from_file_location("_pta", _CONFIG)
    module = importlib.util.module_from_spec(spec)
    sys.modules["_pta"] = module
    spec.loader.exec_module(module)
    return module.PorousTurbulentAccretor


def _radial_mass_flux(path: Path, radius: float) -> tuple[float, float]:
    """
    net mass flux crossing the sphere `radius`, positive INWARD, and the time it is read at.

    a shell one cell wide either side of the sphere stands in for the surface integral: for
    a shell of thickness `w`, `(1/w) * integral_shell rho v_r dV` converges to
    `surface_integral rho v_r dA` as `w -> 0`, and on a uniform cartesian grid the volume
    element is the constant `dx^3`.
    """
    from simbi.reader import read_checkpoint

    chk = read_checkpoint(str(path)).unwrap()
    assert chk.num_levels == 1, (
        f"the checkpoint carries {chk.num_levels} levels; this reads a single grid and "
        "would be integrating over only part of the shell on a refined one"
    )
    level = chk.levels[0]
    assert level.num_partitions == 1, "this reads an undecomposed grid"

    halo = level.mesh.halo_radius
    prims = level.partitions[0].hydro.primitives
    rho = np.asarray(prims["rho"].interior(halo).data)
    vel = [np.asarray(prims[f"v{ii}"].interior(halo).data) for ii in (1, 2, 3)]

    # the domain is a symmetric cube, so the per-axis coordinate arrays are identical and
    # the storage order of the axes does not enter. axis 2 carries v1, axis 0 carries v3.
    lo, hi = level.mesh.dims[0]
    n = rho.shape[0]
    dx = (hi - lo) / n
    centers = lo + (np.arange(n) + 0.5) * dx
    coord = [
        centers[np.newaxis, np.newaxis, :],
        centers[np.newaxis, :, np.newaxis],
        centers[:, np.newaxis, np.newaxis],
    ]
    r = np.sqrt(sum(np.broadcast_to(c, rho.shape) ** 2 for c in coord))

    v_r = sum(v * np.broadcast_to(c, rho.shape) for v, c in zip(vel, coord)) / r
    shell = np.abs(r - radius) < dx
    assert shell.sum() > 100, (
        f"the shell at r = {radius:.4g} holds only {shell.sum()} cells; the flux integral "
        "is too poorly sampled to mean anything"
    )
    # dV / w = dx^3 / (2 dx) = dx^2 / 2. inward is -v_r.
    mdot = float(-(rho * v_r)[shell].sum() * dx * dx / 2.0)
    return mdot, float(chk.metadata.time)


def _run(tmp_path: Path, bondi_inflow: bool):
    from simbi.simulation import runner

    cls = _problem_class()
    data_dir = tmp_path / ("bondi" if bondi_inflow else "static")
    problem = cls(
        buffer_bondi_inflow=bondi_inflow,
        base_resolution=_BASE_RESOLUTION,
        refinement_enabled=False,
        refinement_max_levels=1,
        turb_mach=0.0,
        total_bondi_times=_BONDI_TIMES,
        data_directory=data_dir,
    )
    runner.run(problem, compute_mode="cpu", max_steps=_MAX_STEPS)
    return problem, data_dir


def _steady_flux(problem, data_dir: Path, label: str) -> float:
    """the plateau value of the flux at r_0, with the plateau itself asserted."""
    r_0 = problem.buffer_parameters["buffer_radius"]
    history = sorted(
        (_radial_mass_flux(f, r_0) for f in data_dir.rglob("*.chkpt.*.h5")),
        key=lambda row: row[1],
    )
    assert len(history) > 2, f"the {label} run wrote {len(history)} checkpoints"

    (last, t_last), (prev, t_prev) = history[-1], history[-2]
    assert t_last > _MIN_BONDI_TIMES * problem.bondi_time, (
        f"the {label} run reached only t = {t_last:.4g} in {_MAX_STEPS} steps. the flux at "
        "r_0 is still in the collapse transient there, so it is not the steady supply"
    )
    # non-vacuity: a plateau is what makes the two runs comparable at all. two samples
    # still drifting would mean the numbers below are two arbitrary points on two curves.
    drift = abs(last - prev) / abs(last)
    assert drift < _STEADY_TOL, (
        f"the {label} flux at r_0 moved {drift:.2%} between t = {t_prev:.3f} and "
        f"t = {t_last:.3f}; it has not reached a steady value, so comparing it against the "
        "other buffer target compares two transients"
    )
    return last


@pytest.fixture(scope="module")
def _flux(tmp_path_factory):
    """both runs, reduced to the one number each that the comparison rests on."""
    tmp_path = tmp_path_factory.mktemp("buffer")
    static_problem, static_dir = _run(tmp_path, bondi_inflow=False)
    bondi_problem, bondi_dir = _run(tmp_path, bondi_inflow=True)

    return {
        "r_0": static_problem.buffer_parameters["buffer_radius"],
        "target": (
            4.0
            * np.pi
            * static_problem.accretion_coefficient()
            * static_problem.bondi_radius**2
            * static_problem.ambient_sound_speed
        ),
        "static": _steady_flux(static_problem, static_dir, "static"),
        "bondi": _steady_flux(bondi_problem, bondi_dir, "bondi"),
    }


def test_the_static_buffer_supplies_at_least_the_bondi_rate(_flux):
    # the load-bearing one. if relaxing the shell's momentum to zero held the interior's
    # supply below what a bondi accretor consumes, the science window would be starved by
    # its own boundary condition and nothing measured inside it would be trustworthy.
    static, bondi, target = _flux["static"], _flux["bondi"], _flux["target"]
    print(
        f"\nr_0 = {_flux['r_0']:.4f}   bondi rate 4 pi lambda = {target:.4f}\n"
        f"  static target: steady Mdot(r_0) = {static:+.4f}  ({static / target:.2f} x bondi)\n"
        f"  bondi  target: steady Mdot(r_0) = {bondi:+.4f}  ({bondi / target:.2f} x bondi)\n"
        f"  attributable to the momentum target: {bondi - static:+.4f}  "
        f"({(bondi - static) / target:.2f} x bondi)"
    )
    assert static > target, (
        f"the static buffer supplies only {static / target:.2f} times the bondi rate across "
        f"r_0 ({static:+.4f} against {target:.4f}) in steady state. relaxing the shell's "
        "momentum to zero holds the inflow the steady solution carries at that radius at "
        "rest, and the interior is then fed no faster than it drains itself"
    )


def test_the_bondi_target_adds_supply_in_the_direction_it_pushes(_flux):
    # the wiring, and its sign. the two targets differ by exactly the inward bondi velocity,
    # so the flux they differ by is bounded by 4 pi r_0^2 rho v_bondi(r_0), which is the
    # bondi rate itself to the extent rho at the boundary is near rho_ref.
    static, bondi, target = _flux["static"], _flux["bondi"], _flux["target"]
    excess = bondi - static

    assert excess > 0.0, (
        f"the bondi target REDUCED the steady supply across r_0 by {-excess:.4f}. it relaxes "
        "the shell toward an inward v(r) = -lambda rho_inf R_B^2 cs / (rho_ref r^2), so this "
        "is backwards and the sign of v_r in the momentum target is the first thing to check"
    )
    # non-vacuity: the flag has to change the run at all. were the momentum target dropped
    # on the way to the backend, both runs would be one run and the comparison would be a
    # statement about nothing.
    assert excess > 0.01 * target, (
        f"the two buffer targets reached the same steady supply to {excess / target:.1%} of "
        f"the bondi rate (static {static:+.4f}, bondi {bondi:+.4f}). the momentum target is "
        "not reaching the backend, so this comparison is vacuous"
    )
    assert excess < 5.0 * target, (
        f"the bondi target added {excess / target:.1f} times the bondi rate across r_0 "
        f"({excess:+.4f}). the momentum target carries at most about that flux by "
        "construction, so an excess this large is the shell driving the interior rather "
        "than supplying it"
    )
