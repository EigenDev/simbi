# =============================================================================
# test_porous_output_path.py
#
# the porous accretor's checkpoint directory is the run's identity: two
# parameter points sharing one path silently mix two experiments in a single
# checkpoint series. these gates pin the current production default path,
# distinct paths for distinct supported configurations, the swept segments, and
# the rule that sweep segments key on the model's own declared defaults rather
# than on hardcoded copies of them -- the copy is what let a default edit alias
# two experiments. the endpoint contract rides here too: the sealed surface runs
# KM reconstruction without a fixed reference, the draining surface neither, for
# every solver, and setup refuses any other representation rather than rewrite it.
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
    / "projects"
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
    # the sealed surface runs with the KM pressure-gravity pairing its marginal null needs
    kwargs.setdefault("wb_reconstruction", True)
    problem = cls(**kwargs)
    problem.setup()
    return str(problem.data_directory.relative_to(tmp_path))


def test_the_default_production_path(porous_cls, tmp_path, monkeypatch) -> None:
    # the default sealed run: KM reconstruction, no fixed reference, the HLLC+ solver. the
    # path names each of the run's declared dials and nothing the run does not declare.
    assert (
        _path(porous_cls, tmp_path, monkeypatch)
        == "porous_turbulent/uniform/p0/freeslip/mach0.0625/fmr8/racc4dx/hllc_plus"
    )


def _refusal(porous_cls, **kwargs) -> str:
    """the message the config raises for `kwargs`, or "" when it accepts them"""
    try:
        porous_cls(**kwargs).setup()
    except Exception as err:  # noqa: BLE001 -- pydantic wraps the config's own ValueError
        return str(err)
    return ""


@pytest.mark.parametrize("solver", ["hllc_plus", "hllc"])
def test_the_endpoints_refuse_every_other_representation(porous_cls, tmp_path, monkeypatch, solver) -> None:
    # the sealed endpoint pairs pressure and gravity by KM reconstruction and integrates the
    # state itself; the draining endpoint uses neither mechanism. the contract is the same
    # for every solver: the solver chooses the flux, never the representation.
    monkeypatch.setenv("SCRATCH", str(tmp_path))
    sealed_bare = _refusal(porous_cls, porosity=0.0, solver=solver, wb_reconstruction=False)
    assert "porosity 0 requires --wb-reconstruction" in sealed_bare, sealed_bare
    sealed_ref = _refusal(porous_cls, porosity=0.0, solver=solver, wb_reconstruction=True, well_balanced=True)
    assert "porosity 0 forbids --well-balanced" in sealed_ref, sealed_ref
    for flags in ({"wb_reconstruction": True}, {"well_balanced": True}):
        draining = _refusal(porous_cls, porosity=1.0, solver=solver, **flags)
        assert "porosity 1 requires ordinary conservative reconstruction" in draining, (flags, draining)


@pytest.mark.parametrize("solver", ["hllc_plus", "hllc"])
def test_an_accepted_endpoint_keeps_its_declared_representation(porous_cls, tmp_path, monkeypatch, solver) -> None:
    # setup validates the declared representation and leaves it as declared: the solver
    # and both balance flags come out of setup exactly as they went in.
    monkeypatch.setenv("SCRATCH", str(tmp_path))
    for porosity, km in ((0.0, True), (1.0, False)):
        problem = porous_cls(porosity=porosity, solver=solver, wb_reconstruction=km, well_balanced=False)
        problem.setup()
        assert problem.solver.value == solver
        assert problem.wb_reconstruction is km
        assert problem.well_balanced is False


def test_distinct_supported_configurations_write_distinct_paths(porous_cls, tmp_path, monkeypatch) -> None:
    # every pair below differs in a dial that changes what the scheme integrates, and every
    # member is a configuration the config accepts; two of them sharing a directory would mix
    # two experiments in one checkpoint series. the intermediate porosity is a research arm the
    # endpoint contract leaves free, so both balance mechanisms are live dials there.
    points = {
        "sealed": {"porosity": 0.0},
        "draining": {"porosity": 1.0, "wb_reconstruction": False},
        "sealed, hllc": {"porosity": 0.0, "solver": "hllc"},
        "half, neither": {"porosity": 0.5, "wb_reconstruction": False},
        "half, km": {"porosity": 0.5, "wb_reconstruction": True},
        "half, fixed reference": {"porosity": 0.5, "wb_reconstruction": False, "well_balanced": True},
        "half, km and fixed reference": {"porosity": 0.5, "wb_reconstruction": True, "well_balanced": True},
    }
    paths = {label: _path(porous_cls, tmp_path, monkeypatch, **kw) for label, kw in points.items()}
    shared = {}
    for label, path in paths.items():
        shared.setdefault(path, []).append(label)
    collisions = {path: labels for path, labels in shared.items() if len(labels) > 1}
    assert not collisions, f"supported configurations share an output path: {collisions}"


def test_swept_dials_append_their_segments(porous_cls, tmp_path, monkeypatch) -> None:
    got = _path(
        porous_cls,
        tmp_path,
        monkeypatch,
        buffer_time_fraction=0.3,
        buffer_index=1.25,
    )
    assert got.endswith("hllc_plus/sponge0.3/bufn1.25"), got


def test_segments_key_on_the_declared_defaults(porous_cls, tmp_path, monkeypatch) -> None:
    # the comparison must read the model's own default, so passing the default
    # value explicitly still lands on the segment-free path -- the same run point
    # is the same directory however it was spelled on the command line.
    for name in ("buffer_time_fraction", "buffer_index"):
        default = porous_cls.model_fields[name].default
        got = _path(porous_cls, tmp_path, monkeypatch, **{name: default})
        assert "sponge" not in got and "bufn" not in got, (name, got)


def test_telescoping_ladder_carries_the_halving_cap(porous_cls, tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("SCRATCH", str(tmp_path))
    problem = porous_cls(porosity=0.0, wb_reconstruction=True)
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

    bare = porous_cls(porosity=0.0, wb_reconstruction=True)
    bare.setup()
    assert bare.frustrated_fraction(inner) == 0.0

    seen = []
    for eps in (0.0625, 0.125, 0.25):
        p = porous_cls(porosity=0.0, wb_reconstruction=True, initial_profile="hydrostatic", seed_epsilon=eps)
        p.setup()
        seen.append(p.frustrated_fraction(inner))
    assert seen == sorted(seen), seen
    assert seen[0] < 0.10, seen[0]

    # the retained pool at each amplitude, measured: 0.032 at eps = 1/16, 0.423 at 1/8,
    # 0.806 at 1/4. the middle bound says the cone is cleared by 1/8, and the endpoint
    # bound says a quarter amplitude retains essentially all of the window.
    assert seen[1] > 0.35, seen[1]
    assert seen[-1] > 0.75, seen[-1]

    # the sweep stops where the physics does. retention keeps rising with amplitude, but
    # the delivered field goes supersonic first, and a seed that shocks on arrival is no
    # longer the weak symmetry break whose retention this trend describes. eps = 1/4 sits
    # just inside at mach 0.59; eps = 1/2 is refused at mach 1.12. the boundary is
    # asserted rather than avoided, so the sweep documents where it ends instead of
    # appearing to have been truncated for convenience -- the previous endpoint bound of
    # 0.90 was measured at eps = 1/2, which the mach condition now excludes.
    top = porous_cls(porosity=0.0, wb_reconstruction=True, initial_profile="hydrostatic", seed_epsilon=0.25)
    top.setup()
    assert top.seed_mach_number() < 1.0, top.seed_mach_number()
    with pytest.raises(Exception, match="reaches mach"):
        porous_cls(porosity=0.0, wb_reconstruction=True, initial_profile="hydrostatic", seed_epsilon=0.5).setup()
