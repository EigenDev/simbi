# =============================================================================
# test_silent_config_loss.py
#
# regression gates against SILENT config loss: a typo'd kwarg, an out-of-range
# assignment, a restart flag conflict, and the van leer limiter spelling must all
# fail loudly or round-trip correctly, so a run with a default value cannot
# silently swallow the config.
# =============================================================================

import pytest
from pydantic import ValidationError

from simbi.simulation.checkpoint import _values_agree, metadata_to_config_dict
from simbi.simulation.problem import ConfigError
from simbi.types.input import Limiter
from simbi_configs.examples.grhd.gr_bondi_ks import GrBondiKS


def test_typoed_constructor_kwarg_is_rejected():
    # extra="forbid": cfl_numbr must not vanish while cfl_number keeps its default.
    with pytest.raises(ValidationError, match="cfl_numbr"):
        GrBondiKS(cfl_numbr=0.9)


def test_typoed_cli_flag_is_rejected():
    with pytest.raises(ConfigError, match="unrecognized"):
        GrBondiKS.from_cli(["--cfl-numbr", "0.9"])


def test_out_of_range_assignment_is_rejected():
    # field constraints re-validate on ASSIGNMENT: a setup() or
    # user-code mutation cannot smuggle an invalid value to the backend.
    p = GrBondiKS.from_cli([])
    with pytest.raises(ValueError, match="cfl_number"):
        p.cfl_number = -5.0
    with pytest.raises(ValueError, match="plm_theta"):
        p.plm_theta = 3.5


def test_valid_assignment_still_works():
    p = GrBondiKS.from_cli([])
    p.cfl_number = 0.25
    assert p.cfl_number == 0.25


def test_vanleer_limiter_keeps_the_model_theta_positive():
    # the -1 kernel spelling lives at the execution-dict boundary; the validated
    # model keeps the user's compression so restore/assignment never violate gt=0.
    p = GrBondiKS.from_cli(["--limiter", "vanleer"])
    assert p.limiter == Limiter.VAN_LEER
    assert p.plm_theta > 0.0


def test_ppm_rejects_stray_plm_knobs():
    # ppm carries its own monotonicity constraint; a plm_theta or limiter moved off
    # its default alongside ppm is dead configuration — rejected, never silently
    # ignored. (a knob AT its default is indistinguishable from the passthrough and
    # is inert either way, so it passes.)
    with pytest.raises((ValidationError, ConfigError), match="PPM"):
        GrBondiKS.from_cli(["--reconstruction", "ppm", "--plm-theta", "1.8"])
    with pytest.raises((ValidationError, ConfigError), match="PPM"):
        GrBondiKS.from_cli(["--reconstruction", "ppm", "--limiter", "vanleer"])


def test_ppm_alone_is_accepted_by_the_model():
    from simbi.types.input import Reconstruction

    p = GrBondiKS.from_cli(["--reconstruction", "ppm"])
    assert p.reconstruction == Reconstruction.PPM


def test_order_three_maps_to_ppm_rk3():
    from simbi.types.input import Reconstruction, TimeStepping

    p = GrBondiKS.from_cli(["--order", "3"])
    assert p.reconstruction == Reconstruction.PPM
    assert p.timestepping == TimeStepping.RK3


def test_negative_checkpoint_theta_maps_back_to_vanleer():
    # a checkpoint written by a van leer run stores the kernel spelling
    # plm_theta = -1; the restore path must recover the limiter selection.
    # feeding -1 into the gt=0 field would fail revalidation.
    class _Meta:
        time = 1.0
        gamma = 4.0 / 3.0
        coord_system = "spherical"
        regime = "rhd"
        solver = "hlle"
        reconstruction = "plm"
        timestepping = "rk2"
        plm_theta = -1.0
        cfl = 0.3
        checkpoint_index = 3
        checkpoint_interval = 0.5
        x1_spacing = "log"
        x2_spacing = "linear"
        x3_spacing = "linear"
        boundary_conditions = ["outflow", "outflow"]
        level_dts = ()
        level_substeps = ()
        subcycling_mode = "none"

    config = metadata_to_config_dict(_Meta(), (128,))
    assert config["limiter"] == Limiter.VAN_LEER
    assert "plm_theta" not in config

    _Meta.plm_theta = 1.8
    config = metadata_to_config_dict(_Meta(), (128,))
    assert config["plm_theta"] == 1.8
    assert "limiter" not in config


def test_checkpoint_restores_every_geometric_spacing_ratio():
    class _Meta:
        time = 1.0
        gamma = 5.0 / 3.0
        coord_system = "cartesian"
        regime = "newtonian"
        solver = "hllc"
        reconstruction = "plm"
        timestepping = "rk2"
        plm_theta = 1.5
        cfl = 0.3
        checkpoint_index = 2
        checkpoint_interval = 0.5
        x1_spacing = "geometric"
        x1_spacing_ratio = 0.97
        x2_spacing = "geometric"
        x2_spacing_ratio = 1.03
        x3_spacing = "geometric"
        x3_spacing_ratio = 0.99
        boundary_conditions = ["outflow"] * 6
        level_dts = ()
        level_substeps = ()
        subcycling_mode = "none"

    config = metadata_to_config_dict(_Meta(), (8, 10, 12))

    assert config["x1_spacing_ratio"] == 0.97
    assert config["x2_spacing_ratio"] == 1.03
    assert config["x3_spacing_ratio"] == 0.99


def test_values_agree_is_container_and_enum_insensitive():
    from simbi.types.input import Solver

    assert _values_agree((1.5, 100.0), [1.5, 100.0])
    assert _values_agree(Solver.HLLE, "hlle")
    assert _values_agree(1.0, 1)
    assert not _values_agree(Solver.HLLE, "hllc")
    assert not _values_agree((1.5, 100.0), [1.5, 200.0])


def test_from_cli_records_explicit_flags():
    # the checkpoint merge distinguishes a demanded override from a class default
    # via the argv-provided flag set.
    p = GrBondiKS.from_cli(["--solver", "hllc"])
    assert "solver" in p._cli_explicit
    assert "cfl_number" not in p._cli_explicit


def test_restart_conflict_on_explicit_immutable_flag(monkeypatch, tmp_path):
    # restarting an hlle checkpoint with an EXPLICIT --solver hllc must refuse
    # loudly; without the flag the checkpoint value wins silently.
    from simbi.simulation import checkpoint as cp

    class _Meta:
        time = 1.0
        gamma = 4.0 / 3.0
        coord_system = "spherical"
        regime = "rhd"
        solver = "hlle"
        reconstruction = "plm"
        timestepping = "rk2"
        plm_theta = 1.5
        cfl = 0.3
        checkpoint_index = 3
        checkpoint_interval = 0.5
        x1_spacing = "log"
        x2_spacing = "linear"
        x3_spacing = "linear"
        boundary_conditions = ["outflow", "outflow"]
        level_dts = ()
        level_substeps = ()
        subcycling_mode = "none"

    monkeypatch.setattr(
        cp, "load_checkpoint_metadata", lambda _p: (_Meta(), (512,))
    )

    demanded = GrBondiKS.from_cli(["--solver", "hllc"])
    with pytest.raises(ConfigError, match="solver"):
        cp.merge_with_checkpoint(demanded, tmp_path / "fake.h5")

    silent = GrBondiKS.from_cli([])
    merged = cp.merge_with_checkpoint(silent, tmp_path / "fake.h5")
    assert str(merged.solver.value) == "hlle"


def test_restart_continues_checkpoint_index_not_reset_to_zero(monkeypatch, tmp_path):
    # restarting from chkpt.030 must number the next dump 031. the checkpoint index is
    # checkpoint_safe (serializable), so a generic field merge would take the config default
    # (0) and renumber every restart from zero, overwriting earlier files on disk while
    # start_time still looked correct. the merge must force the index from the checkpoint.
    from simbi.simulation import checkpoint as cp

    class _Meta:
        time = 5.0
        gamma = 4.0 / 3.0
        coord_system = "spherical"
        regime = "rhd"
        solver = "hlle"
        reconstruction = "plm"
        timestepping = "rk2"
        plm_theta = 1.5
        cfl = 0.3
        checkpoint_index = 30
        checkpoint_interval = 0.5
        x1_spacing = "log"
        x2_spacing = "linear"
        x3_spacing = "linear"
        boundary_conditions = ["outflow", "outflow"]
        level_dts = ()
        level_substeps = ()
        subcycling_mode = "none"

    monkeypatch.setattr(
        cp, "load_checkpoint_metadata", lambda _p: (_Meta(), (512,))
    )

    # a fresh config carries the default checkpoint_index (0) and start_time (0); the restart must
    # override both from the checkpoint.
    fresh = GrBondiKS.from_cli([])
    assert fresh.checkpoint_index == 0
    merged = cp.merge_with_checkpoint(fresh, tmp_path / "fake.h5")
    assert merged.checkpoint_index == 30, "restart reset the checkpoint index to the config default"
    assert merged.start_time == 5.0


def _synge_problem_class():
    """a minimal rhd problem on the parameter-free (taub-mathews) closure."""
    from simbi.simulation import (
        CoordSystem,
        ProblemParam,
        Regime,
        SimbiProblem,
    )
    from simbi.types.input import Eos

    class SyngeBlast(SimbiProblem):
        resolution: int = ProblemParam(512, cli=True, description="resolution")
        bounds: tuple[tuple[float, float]] = ProblemParam(
            ((1.0, 2.0),), description="domain bounds"
        )
        coord_system: CoordSystem = ProblemParam(
            CoordSystem.SPHERICAL, description="coordinate system"
        )
        regime: Regime = ProblemParam(Regime.RHD, description="physics regime")
        eos: Eos = ProblemParam(Eos.SYNGE, cli=True, description="closure")

        def initial_primitive_state(self):
            def gas_state():
                for _ in range(self.resolution):
                    yield (1.0, 0.0, 1.0)

            return gas_state

    return SyngeBlast


def _synge_meta(**overrides):
    """checkpoint metadata for a synge run: `gamma` on disk is the INERT placeholder
    the closure's validator wrote, not a declared adiabatic index."""

    class _Meta:
        time = 1.0
        gamma = 5.0 / 3.0
        eos = "synge"
        coord_system = "spherical"
        regime = "rhd"
        solver = "hlle"
        reconstruction = "plm"
        timestepping = "rk2"
        plm_theta = 1.5
        cfl = 0.3
        checkpoint_index = 3
        checkpoint_interval = 0.5
        x1_spacing = "linear"
        x2_spacing = "linear"
        x3_spacing = "linear"
        boundary_conditions = ["outflow", "outflow"]
        level_dts = ()
        level_substeps = ()
        subcycling_mode = "none"

    for key, value in overrides.items():
        setattr(_Meta, key, value)
    return _Meta


def test_synge_restart_does_not_choke_on_placeholder_gamma(monkeypatch, tmp_path):
    # the synge closure REJECTS a declared adiabatic_index and then writes the inert
    # placeholder 5/3 itself; the backend records it as the checkpoint's `gamma`. handing
    # that back to the constructor read as a user declaration and killed every synge
    # restart on the validator. the merge must drop it and let the closure re-supply it.
    from simbi.simulation import checkpoint as cp
    from simbi.types.input import Eos

    monkeypatch.setattr(
        cp, "load_checkpoint_metadata", lambda _p: (_synge_meta(), (512,))
    )

    merged = cp.merge_with_checkpoint(
        _synge_problem_class().from_cli([]), tmp_path / "fake.h5"
    )
    assert merged.eos == Eos.SYNGE
    assert merged.start_time == 1.0
    assert merged.checkpoint_index == 3


def test_synge_restart_from_checkpoint_predating_the_eos_attr(monkeypatch, tmp_path):
    # an OLDER checkpoint records no eos at all. the merge must fall back to the config's
    # declared closure rather than reading the absent attribute as "ideal" — which would
    # both revive the placeholder gamma and silently swap the equations of state.
    from simbi.simulation import checkpoint as cp
    from simbi.types.input import Eos

    monkeypatch.setattr(
        cp, "load_checkpoint_metadata", lambda _p: (_synge_meta(eos=""), (512,))
    )

    merged = cp.merge_with_checkpoint(
        _synge_problem_class().from_cli([]), tmp_path / "fake.h5"
    )
    assert merged.eos == Eos.SYNGE


def test_gamma_law_restart_keeps_the_checkpoint_adiabatic_index(monkeypatch, tmp_path):
    # the drop is scoped to synge: on an ideal run the checkpoint gamma is real physics
    # and must still survive the restart as an immutable field.
    from simbi.simulation import checkpoint as cp

    monkeypatch.setattr(
        cp,
        "load_checkpoint_metadata",
        lambda _p: (_synge_meta(eos="ideal", gamma=4.0 / 3.0), (512,)),
    )

    merged = cp.merge_with_checkpoint(GrBondiKS.from_cli([]), tmp_path / "fake.h5")
    assert merged.adiabatic_index == 4.0 / 3.0


def test_restart_refuses_a_closure_swapped_by_a_config_edit(monkeypatch, tmp_path):
    # the real bmk failure: a checkpoint integrated at ideal gamma = 4/3, restarted from a
    # config later edited to declare synge. no explicit flag exists to catch, so the silent
    # checkpoint-wins rule would resume gamma-law under a source file that reads synge.
    from simbi.simulation import checkpoint as cp

    monkeypatch.setattr(
        cp,
        "load_checkpoint_metadata",
        lambda _p: (_synge_meta(eos="ideal", gamma=4.0 / 3.0), (512,)),
    )

    with pytest.raises(ConfigError, match="closure cannot change on restart") as exc:
        cp.merge_with_checkpoint(
            _synge_problem_class().from_cli([]), tmp_path / "fake.h5"
        )

    # the repair it names must be a CONFIG edit carrying the recorded gamma. `--eos ideal`
    # alone cannot be the advice: the synge config declares no adiabatic_index, so the flag
    # dies on the model validator before the merge is reached.
    assert "adiabatic_index = 1.3333333333333333" in str(exc.value)
    with pytest.raises(ConfigError, match="require an adiabatic_index"):
        _synge_problem_class().from_cli(["--eos", "ideal"])

    # a config that DOES declare the recorded closure resumes without complaint.
    merged = cp.merge_with_checkpoint(GrBondiKS.from_cli([]), tmp_path / "fake.h5")
    assert merged.adiabatic_index == 4.0 / 3.0


def test_viz_config_rejects_typoed_kwargs():
    from simbi.viz.config import FigureConfig

    with pytest.raises(ValidationError):
        FigureConfig(fig_dims=(5, 4))  # the field is fig_size


def test_unknown_props_component_gets_did_you_mean():
    from simbi.viz.config_loader import load_component_props

    with pytest.raises(ValueError, match="did you mean"):
        load_component_props(None, ["qaud.cmap=inferno"])
