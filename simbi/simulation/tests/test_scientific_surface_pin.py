# =============================================================================
# test_scientific_surface_pin.py
#
# Phase 7 Pass 1 pins for the scientific surface. these lock the contract the
# eventual vocabulary/geography renames must preserve, before any rename:
#   - the config surface exposes no compiler-internal concept (a leak ratchet
#     that can only shrink), with the one documented graph-handle leak tracked
#     explicitly so it cannot proliferate;
#   - the representative problems express their science and serialize to the
#     backend wire without a config author touching an IR node, manifest,
#     effect, admission witness, kernel name, or buffer order;
#   - invalid physical combinations fail at construction, not at run.
# =============================================================================

from pathlib import Path

import pytest

from simbi.simulation.runner import to_execution_dict

CONFIGS = Path(__file__).resolve().parents[3] / "simbi_configs" / "examples"

# raw compiler/graph manipulation a scientific config must not perform. the
# ergonomic source layer mints the graph for the author; a config that reaches
# for the graph itself is the documented Pass 1 leak.
RAW_GRAPH_TOKENS = (".graph", "ExprGraph(", "add_node(", "NodeId", "NodeDef")

# compiler-internal identities that must never appear in a scientific config.
FORBIDDEN_INTERNALS = (
    "GvKernel",
    "KernelProgram",
    "AdmittedSources",
    "UserVocabulary",
    "MemorySpace",
    "Effects",
    "Dependence",
)

# the known graph-handle leaks, tracked so the ratchet fails if a fourth
# appears. two patterns: the disk sponge passes `x1.graph` to a constant, and the
# atmosphere/equilibrium configs call `primitives[0].graph.compile(..)`. both
# close when the expression builder mints the graph in a later pass.
DOCUMENTED_GRAPH_LEAKS = {
    "isothermal/dittmann_single_disk.py",
    "newtonian/refined_atmosphere.py",
    "newtonian/refined_isothermal_atmosphere.py",
}


def _config_sources() -> list[Path]:
    return sorted(CONFIGS.rglob("*.py"))


def test_no_config_names_a_compiler_internal():
    offenders = []
    for path in _config_sources():
        text = path.read_text()
        for token in FORBIDDEN_INTERNALS:
            if token in text:
                offenders.append(f"{path.relative_to(CONFIGS)}: {token}")
    assert not offenders, (
        "a scientific config named a compiler-internal identity:\n"
        + "\n".join(offenders)
    )


def test_no_new_raw_graph_leak_is_introduced():
    # the ratchet law is observed_leaks subset documented_legacy_leaks: a new
    # leak fails, closing a known one is an improvement that never breaks the
    # suite. remaining known leaks are reported diagnostically, not asserted.
    leaking = {
        str(path.relative_to(CONFIGS))
        for path in _config_sources()
        if any(tok in path.read_text() for tok in RAW_GRAPH_TOKENS)
    }
    new_leaks = leaking - DOCUMENTED_GRAPH_LEAKS
    assert not new_leaks, (
        "a config introduced a new raw expression-graph leak — the ergonomic "
        f"source layer must mint it instead:\n{sorted(new_leaks)}"
    )
    remaining = sorted(leaking & DOCUMENTED_GRAPH_LEAKS)
    if remaining:
        print(f"[pass 1] {len(remaining)} documented graph leak(s) still to close: {remaining}")


# ---- acceptance: the exemplars express their science and reach the wire ------


def _payload(problem):
    return to_execution_dict(problem)


def test_blast_exemplar_states_geometry_and_boundaries():
    from simbi_configs.examples.newtonian.sedov import SedovTaylor

    payload = _payload(SedovTaylor())
    assert payload["coord_system"] in ("spherical", "SPHERICAL")
    assert payload["regime"] in ("newtonian", "NEWTONIAN")
    # a config author never sees a kernel, manifest, or effect in the wire keys.
    assert not any(
        k in payload for k in ("kernels", "manifest", "effects", "buffers")
    )


def test_refined_blast_exemplar_carries_refinement_without_ir():
    from simbi_configs.examples.newtonian.refined_blast import RefinedBlast

    payload = _payload(RefinedBlast())
    assert payload.get("refinement_enabled") is True
    assert payload.get("refinement_max_levels", 0) >= 1


def test_accretor_exemplar_carries_a_source_and_a_body():
    from simbi_configs.examples.newtonian.bondi import SphericalBondiTest

    payload = _payload(SphericalBondiTest())
    # the sponge source and the accretor body reach the wire as physics, not IR.
    assert payload.get("source_expressions")
    assert payload.get("immersed_bodies") or payload.get("body_system")


# ---- construction-rejection laws: invalid physics fails at construction ------


def test_energy_regime_requires_a_gamma():
    from simbi_configs.examples.newtonian.sedov import SedovTaylor

    with pytest.raises((ValueError, Exception)):
        SedovTaylor(adiabatic_index=None)  # an energy regime with no closure


def test_isothermal_requires_a_sound_speed():
    from simbi import SimbiProblem
    from simbi.types import CoordSystem, Regime

    with pytest.raises((ValueError, Exception)):
        # an isothermal regime declares no closure and no sound speed.
        class _Bad(SimbiProblem):
            regime: Regime = Regime.ISOTHERMAL
            coord_system: CoordSystem = CoordSystem.CARTESIAN
            resolution: tuple[int, int] = (16, 16)
            bounds: list = [(0.0, 1.0), (0.0, 1.0)]

            def initial_primitive_state(self):
                return lambda: (1.0, 0.0, 0.0)

        _Bad().model_validate(_Bad())
