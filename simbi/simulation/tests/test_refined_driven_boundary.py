# =============================================================================
# test_refined_driven_boundary.py
#
# driven (DYNAMIC) boundaries on a REFINED run, end to end through the wheel:
# the config's boundary prescriptions must register on the base AND every fine
# kernel set (a fine level flush against a driven face inherits Driven(id) and
# resolves it against its own registration). the invariant is exact: uniform gas
# with every face driven at the same uniform state must stay uniform on both
# levels — any ordering or registration defect breaks uniformity or panics.
# gated for the energy (newtonian) and isothermal build macros separately;
# they register the dags in different code paths.
# =============================================================================
import glob
import tempfile
from pathlib import Path
from typing import Annotated

import h5py
import numpy as np
from pydantic import computed_field

import simbi.expression as expr
from simbi import ProblemParam, SimbiProblem
from simbi.simulation import runner
from simbi.types import BoundaryCondition, CoordSystem, Regime, Solver
from simbi.types.typing import ExpressionDict, GasStateGenerator, InitialStateType

RHO0 = 2.0
PRE0 = 1.0
DYN4 = [BoundaryCondition.DYNAMIC] * 4


def _uniform_prescription(values: list[float], dim: int = 2) -> ExpressionDict:
    g = expr.ExprGraph()
    _x1 = expr.variable("x1", g)
    _x2 = expr.variable("x2", g)
    outs = [expr.constant(v, g) for v in values]
    return g.compile(outs).serialize_boundary(dim=dim)


class _RefinedDrivenBase(SimbiProblem):
    resolution: Annotated[tuple[int, int], ProblemParam((32, 32))]
    bounds: Annotated[
        list[tuple[float, float]], ProblemParam([(0.0, 1.0), (0.0, 1.0)])
    ]
    coord_system: Annotated[
        CoordSystem, ProblemParam(CoordSystem.CARTESIAN)
    ]
    solver: Annotated[Solver, ProblemParam(Solver.HLLE)]
    boundary_conditions: Annotated[
        list[BoundaryCondition], ProblemParam(DYN4)
    ]
    refinement_enabled: Annotated[bool, ProblemParam(True)]
    refinement_max_levels: Annotated[int, ProblemParam(2)]
    refinement_regions: Annotated[
        list[list[float]], ProblemParam([[0.25, 0.75, 0.25, 0.75]])
    ]
    refinement_ratios: Annotated[list[int], ProblemParam([2])]
    end_time: Annotated[float, ProblemParam(1.0, checkpoint_safe=True)]
    checkpoint_interval: Annotated[
        float, ProblemParam(1.0e30, cli=True, checkpoint_safe=True)
    ]
    data_directory: Annotated[
        Path,
        ProblemParam(Path("data/_test"), cli=True, checkpoint_safe=True),
    ]


class _RefinedDrivenAdiabatic(_RefinedDrivenBase):
    regime: Annotated[Regime, ProblemParam(Regime.NEWTONIAN)]
    adiabatic_index: Annotated[float, ProblemParam(1.4)]

    @computed_field
    @property
    def bx1_inner_expressions(self) -> ExpressionDict:
        return _uniform_prescription([RHO0, 0.0, 0.0, PRE0])

    @computed_field
    @property
    def bx1_outer_expressions(self) -> ExpressionDict:
        return _uniform_prescription([RHO0, 0.0, 0.0, PRE0])

    @computed_field
    @property
    def bx2_inner_expressions(self) -> ExpressionDict:
        return _uniform_prescription([RHO0, 0.0, 0.0, PRE0])

    @computed_field
    @property
    def bx2_outer_expressions(self) -> ExpressionDict:
        return _uniform_prescription([RHO0, 0.0, 0.0, PRE0])

    def initial_primitive_state(self) -> InitialStateType:
        def gas_state() -> GasStateGenerator:
            nx, ny = self.resolution
            for _ in range(nx * ny):
                yield (RHO0, 0.0, 0.0, PRE0)

        return gas_state


class _RefinedDrivenIso(_RefinedDrivenBase):
    regime: Annotated[Regime, ProblemParam(Regime.ISOTHERMAL)]

    # the base declares ambient_sound_speed as a computed field; override the
    # property, not the field.
    @computed_field
    @property
    def ambient_sound_speed(self) -> float:
        return 1.0

    @computed_field
    @property
    def bx1_inner_expressions(self) -> ExpressionDict:
        return _uniform_prescription([RHO0, 0.0, 0.0])

    @computed_field
    @property
    def bx1_outer_expressions(self) -> ExpressionDict:
        return _uniform_prescription([RHO0, 0.0, 0.0])

    @computed_field
    @property
    def bx2_inner_expressions(self) -> ExpressionDict:
        return _uniform_prescription([RHO0, 0.0, 0.0])

    @computed_field
    @property
    def bx2_outer_expressions(self) -> ExpressionDict:
        return _uniform_prescription([RHO0, 0.0, 0.0])

    def initial_primitive_state(self) -> InitialStateType:
        def gas_state() -> GasStateGenerator:
            nx, ny = self.resolution
            for _ in range(nx * ny):
                yield (RHO0, 0.0, 0.0)

        return gas_state


def _run_and_check_uniform(problem) -> None:
    d = tempfile.mkdtemp() + "/"
    problem.data_directory = Path(d)
    runner.run(problem, compute_mode="cpu", max_steps=10)
    final = glob.glob(d + "*final*.h5")
    assert final, "no final checkpoint written"
    with h5py.File(final[0]) as h:
        for lvl in (0, 1):
            arr = h[f"level_{lvl}/conserved/den"][:]
            nc = int(h[f"level_{lvl}/mesh/global_cells"][0])
            ng = int((arr.shape[0] - nc) // 2)
            interior = arr[(slice(ng, -ng),) * arr.ndim]
            drift = np.abs(interior - RHO0).max()
            assert drift < 1e-12, (
                f"level {lvl}: den drifted by {drift:e} under refined driven faces"
            )


class _RefinedDrivenMhd(SimbiProblem):
    # mhd refinement is 3d-only (the CT reflux assumes 3d curl coefficients).
    regime: Annotated[Regime, ProblemParam(Regime.NMHD)]
    adiabatic_index: Annotated[float, ProblemParam(1.4)]
    resolution: Annotated[tuple[int, int, int], ProblemParam((16, 16, 16))]
    bounds: Annotated[
        list[tuple[float, float]],
        ProblemParam([(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]),
    ]
    coord_system: Annotated[CoordSystem, ProblemParam(CoordSystem.CARTESIAN)]
    solver: Annotated[Solver, ProblemParam(Solver.HLLE)]
    boundary_conditions: Annotated[
        list[BoundaryCondition], ProblemParam([BoundaryCondition.DYNAMIC] * 6)
    ]
    refinement_enabled: Annotated[bool, ProblemParam(True)]
    refinement_max_levels: Annotated[int, ProblemParam(2)]
    refinement_regions: Annotated[
        list[list[float]],
        ProblemParam([[0.25, 0.75, 0.25, 0.75, 0.25, 0.75]]),
    ]
    refinement_ratios: Annotated[list[int], ProblemParam([2])]
    end_time: Annotated[float, ProblemParam(1.0, checkpoint_safe=True)]
    checkpoint_interval: Annotated[
        float, ProblemParam(1.0e30, cli=True, checkpoint_safe=True)
    ]
    data_directory: Annotated[
        Path,
        ProblemParam(Path("data/_test"), cli=True, checkpoint_safe=True),
    ]

    @computed_field
    @property
    def bx1_inner_expressions(self) -> ExpressionDict:
        return _uniform_prescription([RHO0, 0.0, 0.0, 0.0, PRE0, 0.0, 0.0, 0.0], dim=3)

    @computed_field
    @property
    def bx1_outer_expressions(self) -> ExpressionDict:
        return _uniform_prescription([RHO0, 0.0, 0.0, 0.0, PRE0, 0.0, 0.0, 0.0], dim=3)

    @computed_field
    @property
    def bx2_inner_expressions(self) -> ExpressionDict:
        return _uniform_prescription([RHO0, 0.0, 0.0, 0.0, PRE0, 0.0, 0.0, 0.0], dim=3)

    @computed_field
    @property
    def bx2_outer_expressions(self) -> ExpressionDict:
        return _uniform_prescription([RHO0, 0.0, 0.0, 0.0, PRE0, 0.0, 0.0, 0.0], dim=3)

    @computed_field
    @property
    def bx3_inner_expressions(self) -> ExpressionDict:
        return _uniform_prescription([RHO0, 0.0, 0.0, 0.0, PRE0, 0.0, 0.0, 0.0], dim=3)

    @computed_field
    @property
    def bx3_outer_expressions(self) -> ExpressionDict:
        return _uniform_prescription([RHO0, 0.0, 0.0, 0.0, PRE0, 0.0, 0.0, 0.0], dim=3)

    def initial_primitive_state(self) -> InitialStateType:
        nx, ny, nz = self.resolution

        def gas_state() -> GasStateGenerator:
            for _ in range(nx * ny * nz):
                yield (RHO0, 0.0, 0.0, 0.0, PRE0)

        def b_field(bn: str):
            def gen():
                counts = {
                    "bx": (nx + 1) * ny * nz,
                    "by": nx * (ny + 1) * nz,
                    "bz": nx * ny * (nz + 1),
                }[bn]
                for _ in range(counts):
                    yield 0.0

            return gen

        return (gas_state, b_field("bx"), b_field("by"), b_field("bz"))


def test_refined_driven_uniform_preservation_adiabatic() -> None:
    _run_and_check_uniform(_RefinedDrivenAdiabatic())


def test_refined_driven_uniform_preservation_iso() -> None:
    _run_and_check_uniform(_RefinedDrivenIso())


def test_refined_driven_uniform_preservation_mhd() -> None:
    _run_and_check_uniform(_RefinedDrivenMhd())
