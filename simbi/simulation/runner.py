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

import dataclasses
import importlib
import os
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any, Optional, Sequence

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
def to_execution_dict(problem: SimbiProblem) -> dict[str, Any]:
    """
    convert problem config to dict for C++ backend.

    this produces the exact format expected by backend.run_simulation().
    """
    # get all model fields
    model_dict = problem.model_dump()

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
    ]
    for field in computed_fields:
        if hasattr(problem, field):
            model_dict[field] = getattr(problem, field)

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

    # derive isothermal from adiabatic_index (gamma) if not explicitly provided
    gamma_val = model_dict.get("adiabatic_index", model_dict.get("gamma", None))
    try:
        if gamma_val is not None:
            model_dict.setdefault("isothermal", float(gamma_val) == 1.0)
    except Exception:
        # leave isothermal absent if gamma cannot be parsed
        pass

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

    # add body system if present
    body_system = problem.body_system
    if body_system and dataclasses.is_dataclass(body_system):
        model_dict["body_system"] = dataclasses.asdict(body_system)
    elif not body_system:
        model_dict.pop("body_system", None)

    # add immersed bodies if present
    immersed = problem.immersed_bodies
    if immersed:
        model_dict["immersed_bodies"] = [
            dataclasses.asdict(b) if dataclasses.is_dataclass(b) else b
            for b in immersed
        ]

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


def _get_primitive_iterator(problem: SimbiProblem) -> GasStateGenerator:
    """get fresh primitive state iterator from problem."""
    initial_state = problem.initial_primitive_state()

    if _is_mhd_generator(initial_state):
        gas_gen_func = initial_state[0]
        return gas_gen_func()
    else:
        gas_gen_func: GasStateFunction = initial_state
        return gas_gen_func()


def _get_bfield_iterators(
    problem: SimbiProblem,
) -> list[StaggeredBFieldGenerator]:
    """get fresh b-field iterators for mhd, or empty list for hydro."""
    initial_state = problem.initial_primitive_state()

    if _is_mhd_generator(initial_state):
        _, bx_gen, by_gen, bz_gen = initial_state
        return [bx_gen(), by_gen(), bz_gen()]

    return []


# =============================================================================
# backend loading
# =============================================================================
def _load_backend(compute_mode: str) -> Optional[ModuleType]:
    """load the appropriate backend module."""
    lib_mode = "cpu" if compute_mode in ["cpu", "omp"] else "gpu"
    try:
        return importlib.import_module(f"simbi.libs.{lib_mode}_ext")
    except ImportError as e:
        print(f"warning: could not load {lib_mode} backend: {e}")
        return None


def _configure_gpu_blocks(dimensionality: int) -> tuple[int, int, int]:
    """configure gpu block dimensions if not already set."""
    dims = {1: (128, 1, 1), 2: (16, 16, 1), 3: (4, 4, 4)}
    dim = min(3, dimensionality)
    default_blocks = dims[dim]

    if "BLOCK_X" not in os.environ:
        os.environ["BLOCK_X"] = str(default_blocks[0])
        os.environ["GPU_BLOCK_Y"] = str(default_blocks[1])
        os.environ["GPU_BLOCK_Z"] = str(default_blocks[2])

    return (
        int(os.environ.get("BLOCK_X", default_blocks[0])),
        int(os.environ.get("GPU_BLOCK_Y", default_blocks[1])),
        int(os.environ.get("GPU_BLOCK_Z", default_blocks[2])),
    )


# =============================================================================
# main entry point
# =============================================================================
def run(
    problem: SimbiProblem,
    compute_mode: str = "cpu",
    validate: bool = False,
) -> None:
    """
    run a simulation with the given problem configuration.

    args:
        problem: the problem configuration
        compute_mode: "cpu", "omp", or "gpu"
        validate: if True, validate generator output before running

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

    # validate generator if requested
    if validate:
        errors = _validate_generator(problem)
        if errors:
            raise ValueError(f"generator validation failed: {errors}")

    # convert to execution dict
    exec_dict = to_execution_dict(problem)

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

    # print simulation info
    _print_simulation_info(exec_dict, gpu_blocks)

    # get fresh iterators
    prim_iterator = _get_primitive_iterator(problem)
    bfield_iterators = _get_bfield_iterators(problem)

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
    )


def _validate_generator(
    problem: SimbiProblem, num_samples: int = 10
) -> Optional[str]:
    """validate that generator produces correctly-shaped output."""
    try:
        initial_state = problem.initial_primitive_state()

        if _is_mhd_generator(initial_state):
            gas_gen_func, bx_gen, by_gen, bz_gen = initial_state
            gas_iter = gas_gen_func()

            for _ in range(num_samples):
                values = next(gas_iter)
                if not hasattr(values, "__len__"):
                    return f"generator must yield sequences, got {type(values)}"
                if len(values) < 5:
                    return f"mhd generator must yield at least 5 values, got {len(values)}"
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

            expected_min = problem.dimensionality + 2
            for _ in range(num_samples):
                values = next(gas_iter)
                if not hasattr(values, "__len__"):
                    return f"generator must yield sequences, got {type(values)}"
                if len(values) < expected_min:
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


def _print_simulation_info(
    exec_dict: dict[str, Any],
    gpu_blocks: Optional[tuple[int, int, int]] = None,
) -> None:
    """print simulation parameters."""
    # defer to existing print function if available
    try:
        from simbi.functional.helpers import print_progress
        from simbi.reader.rich_summary import print_rich_simulation_parameters

        params = exec_dict
        params["gpu_block_dims"] = gpu_blocks
        print_rich_simulation_parameters(params)
        print_progress()  # articificial progress bar for startup
    except ImportError:
        # fallback to simple print
        print("=" * 60)
        print("simulation parameters:")
        print("=" * 60)
        for key in [
            "resolution",
            "regime",
            "coord_system",
            "end_time",
            "cfl_number",
        ]:
            if key in exec_dict:
                print(f"  {key}: {exec_dict[key]}")
        print("=" * 60)
