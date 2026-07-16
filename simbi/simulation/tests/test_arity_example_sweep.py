# =============================================================================
# test_arity_example_sweep.py
#
# the arity guard must accept EVERY shipped example config (zero false
# positives): each config is discovered, instantiated with defaults, its gas
# generator's first tuple peeked, and the tuple passed through the same
# first-tuple contract check the runner applies. a config whose actual yield
# the guard rejects is either a broken config (the guard caught real rot) or a
# mis-scoped guard (an exact width enforced on a regime that does not fix it) —
# both must fail loudly here rather than at a user's run.
# =============================================================================
import importlib
import inspect
from pathlib import Path

import pytest

import simbi_configs.examples
from simbi.simulation import SimbiProblem
from simbi.simulation.runner import _check_first_tuple, _get_iterators


def _example_modules() -> list[str]:
    """dotted module names for every example config file, in sorted order."""
    mods = []
    # a namespace package has no __file__; walk every __path__ entry.
    for entry in simbi_configs.examples.__path__:
        root = Path(entry)
        for path in sorted(root.rglob("*.py")):
            if path.stem.startswith("_"):
                continue
            rel = path.relative_to(root).with_suffix("")
            mods.append("simbi_configs.examples." + ".".join(rel.parts))
    return sorted(set(mods))


def _problem_classes(module_name: str) -> list[type[SimbiProblem]]:
    """the SimbiProblem subclasses DEFINED in the module (imported bases are
    excluded via __module__, mirroring the run executor's discovery)."""
    module = importlib.import_module(module_name)
    return [
        obj
        for obj in vars(module).values()
        if inspect.isclass(obj)
        and issubclass(obj, SimbiProblem)
        and obj is not SimbiProblem
        and obj.__module__ == module.__name__
    ]


@pytest.mark.parametrize("module_name", _example_modules())
def test_every_example_config_passes_the_arity_guard(module_name: str) -> None:
    classes = _problem_classes(module_name)
    if not classes:
        pytest.skip(f"{module_name} defines no SimbiProblem subclass")
    for cls in classes:
        problem = cls()
        # a default-constructed config must be a runnable grid: computed-field
        # placeholders (zero zones, zero-extent bounds) that survive the model
        # validator surface as backend allocation failures at a user's run.
        # 1d configs carry a scalar resolution and a flat (lo, hi) bounds pair;
        # normalize both to per-axis lists.
        res = problem.resolution
        zones = [res] if isinstance(res, int) else list(res)
        for ax, nn in enumerate(zones):
            assert nn > 0, f"{cls.__name__}: zero zones on axis {ax}: {res}"
        bounds = problem.bounds
        pairs = [bounds] if not hasattr(bounds[0], "__len__") else list(bounds)
        for ax, (lo, hi) in enumerate(pairs):
            assert hi > lo, f"{cls.__name__}: zero-extent bounds on axis {ax}: {bounds}"
        # the exact-width branch, when the regime fixes it, must be internally
        # consistent: rho + at least one velocity + pressure.
        expected = problem.expected_primitive_arity()
        if expected is not None:
            assert expected[0] >= 3, f"{cls.__name__}: nonsense exact arity {expected}"
        # the decisive check: the config's OWN first yielded tuple passes the
        # guard and replays intact through the returned iterator.
        prim_iterator, _ = _get_iterators(problem)
        out = _check_first_tuple(problem, prim_iterator)
        first = next(out)
        assert len(first) >= 2
