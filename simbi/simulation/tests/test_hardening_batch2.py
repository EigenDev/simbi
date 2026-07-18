# =============================================================================
# test_hardening_batch2.py
#
# regression gates for the silent-config-loss hardening: a typo'd kwarg, an
# out-of-range assignment, a restart flag conflict, and the van leer limiter
# spelling must all fail loudly (or round-trip correctly) instead of silently
# running with a default.
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
    # field constraints hold on ASSIGNMENT, not just construction: a setup() or
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


def test_negative_checkpoint_theta_maps_back_to_vanleer():
    # a checkpoint written by a van leer run stores the kernel spelling
    # plm_theta = -1; the restore path must recover the limiter selection
    # instead of feeding -1 into the gt=0 field (which fails revalidation).
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
    # loudly; without the flag the checkpoint value wins silently, as before.
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
    # restarting from chkpt.030 must number the next dump 031, not restart from 0. the checkpoint
    # index is checkpoint_safe (serializable), so the generic field merge preferred the config
    # default (0) and every restart silently renumbered from zero, overwriting earlier files —
    # while start_time (force-restored) looked correct. the merge must force the index too.
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


def test_viz_config_rejects_typoed_kwargs():
    from simbi.viz.config import FigureConfig

    with pytest.raises(ValidationError):
        FigureConfig(fig_dims=(5, 4))  # the field is fig_size


def test_unknown_props_component_gets_did_you_mean():
    from simbi.viz.config_loader import load_component_props

    with pytest.raises(ValueError, match="did you mean"):
        load_component_props(None, ["qaud.cmap=inferno"])
