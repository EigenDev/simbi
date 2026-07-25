# =============================================================================
# runner.py
#
# functional simulation runner for simbi.
# converts SimbiProblem to backend format and executes simulation.
#
# usage:
#   from simbi.simulation import run
#   problem = SodProblem(resolution=1000, end_time=0.2)
#   run(problem, compute_mode="cpu")
# =============================================================================
from __future__ import annotations

import importlib
import os
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any, Optional, Sequence

from simbi.types.bodies import body_payload
from simbi.types.input import BoundaryCondition

from simbi.types.typing import (
    GasStateFunction,
    GasStateGenerator,
    StaggeredBFieldGenerator,
)

if TYPE_CHECKING:
    from .problem import SimbiProblem


# =============================================================================
# execution dict conversion
# =============================================================================
def _format_param_value(v: Any) -> Optional[str]:
    """render a custom-param value for the dashboard, or None to skip (callables / complex objects)."""
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        return f"{v:.4g}"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, str):
        return v if len(v) <= 32 else v[:29] + "..."
    if (
        isinstance(v, (list, tuple))
        and 0 < len(v) <= 4
        and all(isinstance(x, (int, float, bool)) for x in v)
    ):
        return "[" + ", ".join(_format_param_value(x) or "?" for x in v) + "]"
    return None


def _collect_custom_params(problem: SimbiProblem) -> list[list[str]]:
    """the config author's OWN params (subclass fields beyond the SimbiProblem base), for the live
    dashboard's grouped 'problem setup' panel. each row is [group, humanized name, value]; the group
    comes from `ProblemParam(group=...)` (default 'Parameters')."""
    from .problem import SimbiProblem
    from .param import get_param_metadata

    base_fields = set(SimbiProblem.model_fields)
    rows: list[list[str]] = []
    for fname, finfo in type(problem).model_fields.items():
        if fname in base_fields:
            continue
        formatted = _format_param_value(getattr(problem, fname, None))
        if formatted is None:
            continue
        group = get_param_metadata(finfo).group or "Parameters"
        rows.append([group, fname.replace("_", " "), formatted])
    # the config's DERIVED quantities (the summary() hook): same panel, own
    # groups — the declared dials and the numbers computed from them side by
    # side, rendered by the dashboard.
    for group, label, value in problem.summary():
        rows.append([str(group), str(label), str(value)])
    return rows


def to_execution_dict(problem: SimbiProblem) -> dict[str, Any]:
    """
    convert problem config to dict for C++ backend.

    this produces the exact format expected by backend.run_simulation().
    """
    import warnings

    # silence the benign pydantic serialization warnings for numpy scalar types (e.g. np.uint64
    # cell counts) crossing into the rust backend. the warning is emitted from
    # `pydantic._internal._serializers` (a module filter on `pydantic.main` would miss it), so
    # match by message across all modules and scope it to the dump so nothing else is suppressed.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r".*[Pp]ydantic serializer warnings.*",
            category=UserWarning,
        )
        # get all model fields
        model_dict = problem.model_dump()

    # the problem's class name — shown in the rust run dashboard header.
    model_dict["name"] = type(problem).__name__

    # ensure data_directory is string with trailing slash
    data_dir = model_dict.get("data_directory", "data/")
    if isinstance(data_dir, Path):
        data_dir = str(data_dir)
    if not data_dir.endswith("/"):
        data_dir += "/"
    model_dict["data_directory"] = data_dir

    # add computed fields (honor explicit problem attributes, then derive sensible defaults)
    computed_fields = [
        "dimensionality",
        "is_mhd",
        "isothermal",
        "nvars",
        "is_relativistic",
        "mesh_motion",
        "is_homologous",
        "dlogt",
        "locally_isothermal",
        "ambient_sound_speed",
        "shakura_sunyaev_alpha",
        "viscosity",
        "resistivity",
    ]
    for field in computed_fields:
        if hasattr(problem, field):
            model_dict[field] = getattr(problem, field)

    # the config author's own params, grouped, for the live dashboard's problem-setup panel.
    model_dict["custom_params"] = _collect_custom_params(problem)

    # normalize regime to a lowercase string and put back into model_dict
    regime = model_dict.get("regime")
    if hasattr(regime, "value"):
        regime = regime.value
    if isinstance(regime, bytes):
        try:
            regime = regime.decode("utf-8")
        except Exception:
            regime = str(regime)
    regime = (regime or "").lower()
    model_dict["regime"] = regime

    # derive is_mhd / is_relativistic from regime when not explicitly provided
    # - any regime string containing 'mhd' is considered mhd
    # - regimes starting with 'sr' or containing 'relativ' or 'rmhd' are considered relativistic
    model_dict.setdefault("is_mhd", "mhd" in regime)
    model_dict.setdefault(
        "is_relativistic",
        bool(
            regime.startswith("sr") or "relativ" in regime or "rmhd" in regime
        ),
    )


    # ensure dimensionality exists (prefer explicit dimensionality, else infer from resolution)
    if (
        "dimensionality" not in model_dict
        or model_dict.get("dimensionality") is None
    ):
        res = model_dict.get("resolution")
        if isinstance(res, (list, tuple)):
            # if resolution provided as [nx, ny, nz] or similar
            # effective dimensionality is number of entries >1 up to 3
            eff = sum(1 for x in res if x and int(x) > 1)
            model_dict["dimensionality"] = max(1, eff)
        else:
            model_dict["dimensionality"] = int(
                model_dict.get("dimensionality", 1)
            )

    # derive a conservative nvars if missing:
    # density (1) + momentum (dim) + energy (1) [+ bfields (3) if mhd] [+ chi (1) if present]
    if "nvars" not in model_dict or model_dict.get("nvars") is None:
        dim = int(model_dict.get("dimensionality", 1))
        nvars = 1 + dim + 1
        if model_dict.get("is_mhd"):
            nvars += 3
        if "chi" in model_dict or hasattr(problem, "chi"):
            nvars += 1
        model_dict["nvars"] = nvars

    # bodies: model_dump carries the raw computed-field values; replace them with
    # the backend wire from the single serialization SSOT (simbi.types.bodies).
    model_dict.pop("body_system", None)
    model_dict.pop("immersed_bodies", None)
    model_dict.pop("bonded_assembly", None)
    model_dict.update(
        body_payload(
            problem.body_system,
            problem.immersed_bodies,
            getattr(problem, "bonded_assembly", None),
        )
    )

    # process bounds to separate x1, x2, x3 bounds
    bounds = problem.bounds
    dim = problem.dimensionality

    x1bounds = bounds[0] if len(bounds) > 0 else (0.0, 1.0)
    x2bounds = bounds[1] if len(bounds) > 1 else (0.0, 1.0)
    x3bounds = bounds[2] if len(bounds) > 2 else (0.0, 1.0)

    model_dict.pop("bounds", None)
    model_dict["x1_bounds"] = x1bounds
    model_dict["x2_bounds"] = x2bounds
    model_dict["x3_bounds"] = x3bounds

    # process boundary conditions
    bcs = _process_boundary_conditions(problem.boundary_conditions, dim)
    model_dict["boundary_conditions"] = bcs

    # convert paths to strings
    for key, value in list(model_dict.items()):
        if isinstance(value, Path):
            model_dict[key] = str(value)

    # nullify callables (c++ has own implementations)
    for key, value in list(model_dict.items()):
        if callable(value):
            model_dict[key] = None

    # normalize resolution to 3d array
    resolution = model_dict["resolution"]
    if isinstance(resolution, int):
        model_dict["resolution"] = [resolution, 1, 1]
    elif len(resolution) == 1:
        model_dict["resolution"] = [resolution[0], 1, 1]
    elif len(resolution) == 2:
        model_dict["resolution"] = [resolution[0], resolution[1], 1]

    return model_dict


def _process_boundary_conditions(
    boundary_conditions: str | Sequence[str],
    effective_dim: int,
) -> list[str]:
    """process and normalize boundary conditions."""
    if isinstance(boundary_conditions, str):
        return [boundary_conditions] * (2 * effective_dim)
    # a single non-list bc (a lone Neumann/Robin gradient wall) applies to every face.
    if not isinstance(boundary_conditions, (list, tuple)):
        return [boundary_conditions] * (2 * effective_dim)

    bcs = list(boundary_conditions)
    num_bcs = len(bcs)
    num_faces = 2 * effective_dim

    # one bc per dimension (same for inner and outer)
    if num_bcs == effective_dim:
        return [bc for bc in bcs for _ in range(2)]

    # one bc for each face
    if num_bcs == num_faces:
        return bcs

    # single bc for all faces
    if num_bcs == 1:
        return bcs * num_faces

    # extrapolate missing
    missing = num_faces - num_bcs
    return bcs + [BoundaryCondition.OUTFLOW] * missing


# =============================================================================
# generator handling
# =============================================================================
def _is_mhd_generator(gen: Any) -> bool:
    """check if generator is mhd (tuple of 4 generators)."""
    return not callable(gen) and len(gen) == 4


def _get_iterators(
    problem: SimbiProblem,
) -> tuple[GasStateGenerator, list[StaggeredBFieldGenerator]]:
    """the gas iterator + the staggered-B iterators (empty for hydro), from ONE
    initial_primitive_state() call. the one-call contract matters: a STOCHASTIC
    initial condition invoked twice hands the gas and the magnetic field
    different random draws — silently inconsistent data unless the config
    happens to fix its rng seed."""
    initial_state = problem.initial_primitive_state()

    if _is_mhd_generator(initial_state):
        gas_gen_func, bx_gen, by_gen, bz_gen = initial_state
        return gas_gen_func(), [bx_gen(), by_gen(), bz_gen()]
    gas_gen_func: GasStateFunction = initial_state
    return gas_gen_func(), []


# =============================================================================
# backend loading
# =============================================================================
def _enable_gpu_page_migration() -> None:
    """make managed allocations device-resident on amd.

    fields are allocated as unified/managed memory, which only migrates onto the gpu
    when the device can fault on an absent page. that mechanism is XNACK, and it is
    disabled by default on gfx90a (MI250X). without it a managed allocation stays
    host-resident for the life of the run and every kernel access crosses the host
    bus instead of hitting hbm -- a ~24x throughput loss on a memory-bound stencil,
    with no error and no warning. the amd runtime reads this variable once when it
    initializes, so it must be set before the gpu extension is imported.

    nvidia ignores the variable entirely and page-migrates on fault unconditionally,
    so setting it is inert there. an explicit value from the environment is left
    alone, which keeps the disabling case reachable.
    """
    os.environ.setdefault("HSA_XNACK", "1")


def _load_backend(compute_mode: str) -> Optional[ModuleType]:
    """load the appropriate backend extension. a MISSING extension (never built for this
    mode) returns None -> the caller's demo mode (config inspection, CI without a build). an
    extension that EXISTS but fails to load -- typically a GPU runtime version mismatch
    between build and run, surfacing as an undefined symbol -- raises: silently demoting a
    scheduled GPU job to a config dump wastes the whole allocation."""
    lib_mode = "cpu" if compute_mode in ["cpu", "omp"] else "gpu"
    if lib_mode == "gpu":
        _enable_gpu_page_migration()
    try:
        return importlib.import_module(f"simbi.libs.{lib_mode}_ext")
    except ModuleNotFoundError as e:
        print(f"warning: {lib_mode} backend not built ({e}); entering demo mode")
        return None
    except ImportError as e:
        raise RuntimeError(
            f"the {lib_mode} backend is built but failed to load: {e}\n"
            f"this is almost always a GPU runtime version mismatch between build and run -- "
            f"load the SAME ROCm/CUDA module the extension was built against (e.g. the exact "
            f"`rocm/<version>` from the session where the build + a test run succeeded), so "
            f"the HIP/HSA (or CUDA/NVRTC) runtime versions agree."
        ) from e


def _configure_gpu_blocks(dimensionality: int) -> tuple[int, int, int]:
    """configure gpu block dimensions if not already set."""
    dims = {1: (128, 1, 1), 2: (16, 16, 1), 3: (4, 4, 4)}
    dim = min(3, dimensionality)
    default_blocks = dims[dim]

    if "BLOCK_X" not in os.environ:
        os.environ["BLOCK_X"] = str(default_blocks[0])
        os.environ["BLOCK_Y"] = str(default_blocks[1])
        os.environ["BLOCK_Z"] = str(default_blocks[2])

    return (
        int(os.environ.get("BLOCK_X", default_blocks[0])),
        int(os.environ.get("BLOCK_Y", default_blocks[1])),
        int(os.environ.get("BLOCK_Z", default_blocks[2])),
    )


# =============================================================================
# main entry point
# =============================================================================
def validate_problem(problem: SimbiProblem, compute_mode: str = "cpu") -> None:
    """validate a complete run without allocating its grid or writing output."""
    errors = _validate_generator(problem)
    if errors:
        raise ValueError(f"generator validation failed: {errors}")

    exec_dict = to_execution_dict(problem)
    prim_iterator, bfield_iterators = _get_iterators(problem)
    _check_first_tuple(problem, prim_iterator)
    for name, iterator in zip(("Bx", "By", "Bz"), bfield_iterators):
        _check_first_scalar(type(problem).__name__, name, iterator)
    chi_field = problem.passive_scalar()
    if chi_field is not None:
        _check_first_scalar(type(problem).__name__, "passive scalar", chi_field)

    backend = _load_backend(compute_mode)
    if backend is None:
        raise RuntimeError(
            f"{compute_mode} backend is required for production Rust validation"
        )
    backend.validate_simulation(sim_info=exec_dict)
    print(f"{type(problem).__name__}: validation passed")


def _check_first_scalar(problem_name: str, field_name: str, iterator: Any) -> float:
    """validate one scalar generator value without weakening finite-state rules."""
    import math

    try:
        raw = next(iterator)
    except StopIteration:
        raise ValueError(
            f"{problem_name}.{field_name}: generator yielded nothing"
        ) from None
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{problem_name}.{field_name}: generator values must be numeric: {exc}"
        ) from None
    if not math.isfinite(value):
        raise ValueError(
            f"{problem_name}.{field_name}: first value must be finite, got {value}"
        )
    return value


def run(
    problem: SimbiProblem,
    compute_mode: str = "cpu",
    validate: bool = False,
    live_monitor: bool = False,
    max_steps: int = 0,
) -> None:
    """
    run a simulation with the given problem configuration.

    args:
        problem: the problem configuration
        compute_mode: "cpu", "omp", or "gpu"
        validate: if True, validate generator output before running
        live_monitor: if True, write a read-only snapshot each cadence so
            `simbi attach <data_directory>` can monitor a headless run
        max_steps: stop after this many steps (0 = run to end_time); the final
            checkpoint is written either way — a bounded run is a truncated
            but otherwise ordinary run (smoke tests, profiling probes)

    example:
        >>> from simbi.simulation import SimbiProblem, ProblemParam, run
        >>> class Sod(SimbiProblem):
        ...     resolution: int = ProblemParam(1000, cli=True)
        ...     # ... other fields ...
        ...     def initial_primitive_state(self):
        ...         def gen():
        ...             for i in range(self.resolution):
        ...                 yield (1.0, 0.0, 1.0) if i < 500 else (0.125, 0.0, 0.1)
        ...         return gen
        >>> run(Sod(bounds=[(0, 1)], ...), compute_mode="cpu")
    """
    from .checkpoint import merge_with_checkpoint

    # handle checkpoint if specified
    if problem.checkpoint_file:
        checkpoint_path = Path(problem.checkpoint_file)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
        problem = merge_with_checkpoint(problem, checkpoint_path)

    # ensure data directory exists
    data_dir = Path(problem.data_directory)
    if not data_dir.exists():
        data_dir.mkdir(parents=True, exist_ok=True)

    # deep generator validation stays opt-in (it consumes fresh iterators);
    # the zero-cost first-tuple check below always runs.
    if validate:
        errors = _validate_generator(problem)
        if errors:
            raise ValueError(f"generator validation failed: {errors}")

    # convert to execution dict
    exec_dict = to_execution_dict(problem)
    exec_dict["live_monitor"] = live_monitor
    if max_steps < 0:
        raise ValueError(f"max_steps must be >= 0, got {max_steps}")
    exec_dict["max_steps"] = max_steps

    # configure gpu if needed
    gpu_blocks = None
    if compute_mode == "gpu":
        gpu_blocks = _configure_gpu_blocks(problem.dimensionality)

    # load backend
    backend = _load_backend(compute_mode)
    if backend is None:
        print("demo mode: simulation would execute with:")
        for key, value in sorted(exec_dict.items()):
            print(f"  {key}: {value}")
        return

    # forward gpu block dims for the (future) gpu backend; the run dashboard —
    # problem setup, live benchmarks, progress, messages — is rendered by the
    # rust backend (symbi_display::Table), so no python-side summary is printed.
    if gpu_blocks is not None:
        exec_dict["gpu_block_dims"] = tuple(gpu_blocks)

    # get fresh iterators — ONE initial_primitive_state() call for both, so a
    # stochastic IC seeds gas and B from the same draw.
    prim_iterator, bfield_iterators = _get_iterators(problem)
    # first-tuple contract check, always on: one tuple costs nothing and catches
    # the classic generator-contract violations (calling gen() where gen itself
    # must be passed, wrong tuple arity, NaN or non-positive density/pressure)
    # with a message naming the contract.
    prim_iterator = _check_first_tuple(problem, prim_iterator)

    # get scale factor functions
    scale_factor = problem.scale_factor or (lambda t: 1.0)
    scale_factor_derivative = problem.scale_factor_derivative or (lambda t: 0.0)

    # run simulation
    backend.run_simulation(
        prim_gen=prim_iterator,
        staggered_bfields=bfield_iterators,
        sim_info=exec_dict,
        a=scale_factor,
        adot=scale_factor_derivative,
        chi_field=problem.passive_scalar(),
    )


def _check_first_tuple(problem: SimbiProblem, it: GasStateGenerator) -> GasStateGenerator:
    """peek the generator's first yielded tuple, validate the contract, and
    return an iterator that replays it: `initial_primitive_state` must return a
    zero-argument callable (or the 4-tuple of them for MHD) whose iterator
    yields per-cell numeric tuples (rho, v.., p) with finite entries and
    positive density (and pressure, when the regime carries energy)."""
    import itertools

    name = type(problem).__name__
    try:
        first = next(it)
    except StopIteration:
        raise ValueError(
            f"{name}.initial_primitive_state: the gas generator yielded nothing — "
            "it must yield one (rho, v.., p) tuple per cell"
        ) from None
    except TypeError as exc:
        raise ValueError(
            f"{name}.initial_primitive_state: expected an ITERATOR of per-cell "
            f"tuples; check that the config returns the generator function itself "
            f"(not gen()) or vice versa at the call boundary: {exc}"
        ) from None
    # where the regime fixes the width exactly, pin it: the reader maps the tuple
    # positionally, so a too-long tuple silently shifts a trailing field (e.g.
    # pressure) into an ignored slot without erroring. regimes whose width is
    # not uniquely determined (isothermal's optional p, relativistic hydro's
    # chart-dependent velocity dof) return None and get a lower-bound check.
    if not hasattr(first, "__len__"):
        raise ValueError(
            f"{name}.initial_primitive_state: each yield must be a (rho, v.., p) "
            f"sequence, got a scalar {first!r}"
        )
    arity_fn = getattr(problem, "expected_primitive_arity", lambda: None)
    expected = arity_fn()
    if expected is not None:
        arity, signature = expected
        if len(first) != arity:
            raise ValueError(
                f"{name}.initial_primitive_state: each yielded tuple must have "
                f"EXACTLY {arity} entries {signature} for a "
                f"{problem.dimensionality}d {problem.regime} run — got "
                f"{len(first)}: {first!r}. mhd always carries the full 3-velocity "
                f"even in 2.5d; pure hydro carries one velocity per spatial "
                f"dimension. a longer tuple silently shifts the trailing field "
                f"(e.g. pressure) into an ignored slot."
            )
    else:
        # velocities are never optional: every regime carries at least one velocity
        # per spatial dimension (curvilinear/relativistic charts may carry MORE —
        # transverse components — which is why this is a floor).
        # only the trailing pressure is optional, and only for isothermal.
        isothermal = getattr(problem, "isothermal", False)
        ndim = getattr(problem, "dimensionality", 1)
        min_len = 1 + ndim + (0 if isothermal else 1)
        shape = "(rho, v..)" if isothermal else "(rho, v.., p)"
        if len(first) < min_len:
            raise ValueError(
                f"{name}.initial_primitive_state: each yield must be a {shape} "
                f"sequence of >= {min_len} numbers for a {ndim}d run, got {first!r}"
            )
    try:
        vals = [float(v) for v in first]
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{name}.initial_primitive_state: non-numeric entry in the first "
            f"yielded tuple {first!r}: {exc}"
        ) from None
    import math

    if any(not math.isfinite(v) for v in vals):
        raise ValueError(
            f"{name}.initial_primitive_state: non-finite entry in the first "
            f"yielded tuple {tuple(vals)}"
        )
    if vals[0] <= 0.0:
        raise ValueError(
            f"{name}.initial_primitive_state: the first cell's density is "
            f"{vals[0]} — density must be positive"
        )
    return itertools.chain([first], it)


def _validate_generator(
    problem: SimbiProblem, num_samples: int = 10
) -> Optional[str]:
    """validate that generator produces correctly-shaped output."""
    try:
        initial_state = problem.initial_primitive_state()

        expected = problem.expected_primitive_arity()

        if _is_mhd_generator(initial_state):
            gas_gen_func, bx_gen, by_gen, bz_gen = initial_state
            gas_iter = gas_gen_func()

            for _ in range(num_samples):
                values = next(gas_iter)
                if not hasattr(values, "__len__"):
                    return f"generator must yield sequences, got {type(values)}"
                if expected is not None and len(values) != expected[0]:
                    return (
                        f"mhd generator must yield exactly {expected[0]} values "
                        f"{expected[1]}, got {len(values)}"
                    )
                if expected is None and len(values) < 4:
                    return f"mhd generator must yield at least 4 values, got {len(values)}"
                try:
                    [float(v) for v in values]
                except (ValueError, TypeError) as e:
                    return f"all values must be numeric: {e}"

            for name, gen_func in [
                ("Bx", bx_gen),
                ("By", by_gen),
                ("Bz", bz_gen),
            ]:
                b_iter = gen_func()
                for _ in range(num_samples):
                    value = next(b_iter)
                    try:
                        float(value)
                    except (ValueError, TypeError):
                        return f"{name} generator must yield numeric values"
        else:
            gas_gen_func: GasStateFunction = initial_state
            gas_iter = gas_gen_func()

            # isothermal carries no mandatory pressure (min rho + dim velocities);
            # an energy regime adds pressure (min rho + dim velocities + p).
            expected_min = problem.dimensionality + (
                1 if getattr(problem, "isothermal", False) else 2
            )
            for _ in range(num_samples):
                values = next(gas_iter)
                if not hasattr(values, "__len__"):
                    return f"generator must yield sequences, got {type(values)}"
                if expected is not None and len(values) != expected[0]:
                    return (
                        f"generator must yield exactly {expected[0]} values "
                        f"{expected[1]}, got {len(values)}"
                    )
                if expected is None and len(values) < expected_min:
                    return f"generator must yield at least {expected_min} values, got {len(values)}"
                try:
                    [float(v) for v in values]
                except (ValueError, TypeError) as e:
                    return f"all values must be numeric: {e}"

        return None

    except StopIteration:
        return f"generator exhausted after {num_samples} samples"
    except Exception as e:
        return f"unexpected error: {e}"
