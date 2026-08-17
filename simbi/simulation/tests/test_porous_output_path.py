# =============================================================================
# test_porous_output_path.py
#
# the porous accretor's checkpoint directory is the run's identity: two
# parameter points sharing one path silently mix two experiments in a single
# checkpoint series. these gates pin the segment-free default path (every
# archived series wrote to it), the swept segments, and the rule that sweep
# segments key on the model's own declared defaults rather than on hardcoded
# copies of them -- the copy is what let a default edit alias two experiments.
# =============================================================================
import importlib.util
import inspect
import sys
from pathlib import Path

import pytest

_CONFIG = (
    Path(__file__).resolve().parents[3]
    / "simbi_configs"
    / "science"
    / "simbi_projects"
    / "porous_turbulent_accretor.py"
)


@pytest.fixture(scope="module")
def porous_cls():
    if not _CONFIG.exists():
        pytest.skip("science config tree not present")
    spec = importlib.util.spec_from_file_location("porous_path_gate", _CONFIG)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["porous_path_gate"] = mod
    spec.loader.exec_module(mod)
    (cls,) = [
        c
        for _, c in inspect.getmembers(mod, inspect.isclass)
        if c.__module__ == "porous_path_gate"
    ]
    return cls


def _path(cls, tmp_path, monkeypatch, **kwargs):
    # the sealed branch retains its gas by construction, so it derives paths without
    # tripping the loss-cone premise a draining surface has to satisfy.
    monkeypatch.setenv("SCRATCH", str(tmp_path))
    kwargs.setdefault("porosity", 0.0)
    problem = cls(**kwargs)
    problem.setup()
    return str(problem.data_directory.relative_to(tmp_path))


def test_the_default_path_is_the_archived_one(porous_cls, tmp_path, monkeypatch) -> None:
    # the segment-free layout every existing series wrote to; a drift here orphans
    # the archive.
    assert (
        _path(porous_cls, tmp_path, monkeypatch)
        == "porous_turbulent/uniform/p0/freeslip/mach0.0625/fmr8/racc4dx/hllc_lm"
    )


def test_swept_dials_append_their_segments(porous_cls, tmp_path, monkeypatch) -> None:
    got = _path(
        porous_cls,
        tmp_path,
        monkeypatch,
        buffer_time_fraction=0.3,
        buffer_index=1.25,
    )
    assert got.endswith("hllc_lm/sponge0.3/bufn1.25"), got


def test_segments_key_on_the_declared_defaults(porous_cls, tmp_path, monkeypatch) -> None:
    # the comparison must read the model's own default, so passing the default
    # value explicitly still lands on the segment-free path -- the same run point
    # is the same directory however it was spelled on the command line.
    for name in ("buffer_time_fraction", "buffer_index", "mach_limit"):
        default = porous_cls.model_fields[name].default
        got = _path(porous_cls, tmp_path, monkeypatch, **{name: default})
        assert "sponge" not in got and "bufn" not in got and "/ml" not in got, (name, got)


def test_a_swept_mach_limit_gets_its_own_directory(porous_cls, tmp_path, monkeypatch) -> None:
    # the fleischmann saturation threshold is the instrument knob the machsweep arm
    # varies; without its own segment the three sweep members would interleave one
    # checkpoint series and the sweep would silently compare a run with itself.
    got = _path(porous_cls, tmp_path, monkeypatch, mach_limit=0.05)
    assert got.endswith("/ml0.05"), got


def test_telescoping_ladder_carries_the_halving_cap(porous_cls, tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("SCRATCH", str(tmp_path))
    problem = porous_cls(porosity=0.0)
    problem.setup()
    r_prev = problem.domain_radius * problem.bondi_radius
    for box in problem.refinement_regions:
        assert box[1] <= 0.5 * r_prev + 1e-15
        r_prev = box[1]


def test_the_loss_cone_fraction_tracks_the_seed_amplitude(porous_cls, tmp_path, monkeypatch) -> None:
    # xi/xi_crit is a constant of the motion, so the seed fixes the retained pool once
    # and permanently: gas inside the cone circularizes within the accretor and drains.
    # a draining run whose window starts inside the cone therefore relaxes to laminar
    # inflow whatever the surface does, which is decidable from the initial data alone.
    # measured at the window's inner edge: the band-limited fft seed retains nothing,
    # and the scale-invariant seed clears the cone from eps ~ 0.125 upward.
    monkeypatch.setenv("SCRATCH", str(tmp_path))
    inner = 20.0

    bare = porous_cls(porosity=0.0)
    bare.setup()
    assert bare.frustrated_fraction(inner) == 0.0

    seen = []
    for eps in (0.0625, 0.125, 0.25, 0.5):
        p = porous_cls(porosity=0.0, initial_profile="hydrostatic", seed_epsilon=eps)
        p.setup()
        seen.append(p.frustrated_fraction(inner))
    assert seen == sorted(seen), seen
    assert seen[0] < 0.10, seen[0]
    assert seen[1] > 0.35, seen[1]
    assert seen[-1] > 0.90, seen[-1]
