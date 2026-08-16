# =============================================================================
# test_synge_seeding.py
#
# the initial condition of a synge (taub-mathews) run must be the one the config
# declared. seeding converts the config's primitives to conserved variables through
# the closure the sim state carries, while every cons->prim recovery afterwards runs
# the closure the kernels were baked with; if the two name different gases the run
# starts somewhere other than where it was pointed. the signature is specific —
# D = rho W needs no closure and survives, so the corruption lands entirely in the
# rho/W split and in the pressure.
#
# the probe is a uniform hot relativistic state (theta = p/rho = 20, W = 20), which
# a conservative scheme preserves exactly: whatever comes back differs from what
# went in only through the seeding conversion. the enthalpies of the two closures
# stand ~36% apart there, so the round trip has something to fail.
# requires the built cpu_ext backend; skipped in its absence.
# =============================================================================
import glob
import tempfile

import h5py
import numpy as np
import pytest

from simbi.simulation import runner

_BACKEND = runner._load_backend("cpu")
needs_backend = pytest.mark.skipif(
    _BACKEND is None, reason="rust cpu_ext backend not built"
)

# the seeded state: rho, three-velocity, pressure. W = 20 and theta = 20 put the gas
# deep in the taub-mathews walk, far from both the 5/3 and 4/3 gamma-law limits.
RHO = 1.0
PRE = 20.0
LORENTZ = 20.0
VEL = np.sqrt(1.0 - 1.0 / LORENTZ**2)
RESOLUTION = 64


def _taub_mathews_enthalpy(theta: float) -> float:
    """h = 2.5 theta + sqrt(2.25 theta^2 + 1), the taub-inequality-saturating gas."""
    return 2.5 * theta + np.sqrt(2.25 * theta**2 + 1.0)


def _gamma_law_enthalpy(theta: float, gamma: float) -> float:
    return 1.0 + gamma / (gamma - 1.0) * theta


def _synge_uniform_problem(data_dir: str):
    from simbi.simulation import CoordSystem, ProblemParam, Regime, SimbiProblem
    from simbi.types.input import Eos

    class SyngeUniform(SimbiProblem):
        resolution: int = ProblemParam(64, cli=True, description="resolution")
        bounds: tuple[tuple[float, float]] = ProblemParam(
            ((0.0, 1.0),), description="domain bounds"
        )
        coord_system: CoordSystem = ProblemParam(
            CoordSystem.CARTESIAN, description="coordinate system"
        )
        regime: Regime = ProblemParam(Regime.RHD, description="physics regime")
        eos: Eos = ProblemParam(Eos.SYNGE, cli=True, description="closure")

        def initial_primitive_state(self):
            def gas_state():
                for _ in range(self.resolution):
                    yield (RHO, VEL, PRE)

            return gas_state

    p = SyngeUniform.from_cli([])
    p.end_time = 0.05
    p.data_directory = data_dir
    p.checkpoint_interval = 1.0e30
    return p


def _final_checkpoint(data_dir: str):
    final = glob.glob(data_dir + "*final*.h5")
    assert final, "no final checkpoint written"
    return final[0]


def _interior(f, name: str) -> np.ndarray:
    """a checkpoint field over the owned cells. the primitive groups are stored WITH the
    halo band while the conserved ones are not, so the band is stripped by length rather
    than by which group the field came from."""
    halo = int(f["level_0/mesh"].attrs.get("halo_width", 0))
    arr = np.asarray(f[name]).ravel()
    return arr[halo:-halo] if halo and arr.size > RESOLUTION else arr


@needs_backend
@pytest.mark.simulation
def test_synge_run_recovers_the_primitives_it_was_seeded_with() -> None:
    d = tempfile.mkdtemp() + "/"
    runner.run(_synge_uniform_problem(d), compute_mode="cpu", max_steps=8)
    group = "level_0/partition_0/hydro/primitives/"
    with h5py.File(_final_checkpoint(d)) as f:
        got = {
            "rho": (_interior(f, group + "rho"), RHO),
            "v1": (_interior(f, group + "v1"), VEL),
            "pre": (_interior(f, group + "pre"), PRE),
        }
    for name, (field, seeded) in got.items():
        assert field.size == RESOLUTION, (
            f"{name} covers {field.size} cells, expected the {RESOLUTION} owned ones"
        )
        err = float(np.abs(field - seeded).max()) / abs(seeded)
        assert err < 1e-9, (
            f"a uniform synge state did not survive its own seeding: {name} is "
            f"{err:.3e} off the seeded {seeded} (worst cell {field[np.argmax(np.abs(field - seeded))]})"
        )


@needs_backend
@pytest.mark.simulation
def test_synge_conserved_energy_is_the_taub_mathews_one() -> None:
    # tau = rho h W^2 - p - D reads the closure directly, so it separates the two
    # candidate gases without inverting anything.
    theta = PRE / RHO
    den = RHO * LORENTZ
    tau_tm = RHO * _taub_mathews_enthalpy(theta) * LORENTZ**2 - PRE - den
    tau_ideal = RHO * _gamma_law_enthalpy(theta, 5.0 / 3.0) * LORENTZ**2 - PRE - den
    # the gate is only worth running where the closures disagree.
    assert abs(tau_ideal - tau_tm) / tau_tm > 0.1, (
        f"the two closures agree on this state (tau_tm = {tau_tm}, "
        f"tau_ideal = {tau_ideal}); the seeding check below is vacuous"
    )

    d = tempfile.mkdtemp() + "/"
    runner.run(_synge_uniform_problem(d), compute_mode="cpu", max_steps=8)
    with h5py.File(_final_checkpoint(d)) as f:
        tau_field = _interior(f, "level_0/conserved/nrg")
        den_field = _interior(f, "level_0/conserved/den")

    err_den = float(np.abs(den_field - den).max()) / den
    assert err_den < 1e-9, (
        f"D = rho W is closure-free and must be exact: {err_den:.3e} off {den}"
    )
    tau = float(tau_field[tau_field.size // 2])
    err_tm = float(np.abs(tau_field - tau_tm).max()) / tau_tm
    err_ideal = abs(tau - tau_ideal) / tau_ideal
    assert err_tm < 1e-9, (
        f"the seeded energy is not the taub-mathews one: tau = {tau}, "
        f"taub-mathews = {tau_tm} ({err_tm:.3e} off), gamma-law 5/3 = {tau_ideal} "
        f"({err_ideal:.3e} off)"
    )
