import numpy as np
from dataclasses import dataclass
from ..config.constants import (
    Regime,
    CoordSystem,
    TimeStepping,
    SpatialOrder,
    CellSpacing,
    Solver,
    BoundaryCondition,
)
from ..protocol import StateGenerator
from numpy.typing import NDArray
from ...functional.maybe import Maybe
from typing import Sequence, Any, Callable


class ConfigValidator:
    """validates configuration properties"""

    def __init__(self) -> None:
        self.validators: dict[str, Callable[[Any], Maybe[Any]]] = {
            "regime": self.validate_regime,
            "coord_system": self.validate_coord_system,
            "temporal_order": self.validate_temporal_order,
            "spatial_order": self.validate_spatial_order,
            "x1_spacing": self.validate_cell_spacing,
            "x2_spacing": self.validate_cell_spacing,
            "x3_spacing": self.validate_cell_spacing,
            "solver": self.validate_solver,
            "boundary_conditions": self.validate_boundary_conditions,
        }

    def validate_regime(self, regime: str) -> Maybe[str]:
        """validate regime setting"""
        try:
            return Maybe.of(Regime(regime.lower()).value)
        except ValueError as e:
            return Maybe.save_failure(f"{e}. Available options: {Regime.list()}")

    def validate_coord_system(self, coord_system: str) -> Maybe[str]:
        """validate coordinate system"""
        try:
            return Maybe.of(CoordSystem(coord_system.lower()).value)
        except ValueError as e:
            return Maybe.save_failure(f"{e}. Available options: {CoordSystem.list()}")

    def validate_temporal_order(self, temporal_order: str) -> Maybe[str]:
        """validate time stepping scheme"""
        try:
            return Maybe.of(TimeStepping(temporal_order.lower()).value)
        except ValueError as e:
            return Maybe.save_failure(f"{e}. Available options: {TimeStepping.list()}")

    def validate_spatial_order(self, space_order: str) -> Maybe[str]:
        """validate space order"""
        try:
            return Maybe.of(SpatialOrder(space_order.lower()).value)
        except ValueError as e:
            return Maybe.save_failure(f"{e}. Available options: {SpatialOrder.list()}")

    def validate_spacing(self, cell_spacing: str) -> Maybe[str]:
        """validate cell spacing"""
        try:
            return Maybe.of(CellSpacing(cell_spacing.lower()).value)
        except ValueError as e:
            return Maybe.save_failure(f"{e}. Available options: {CellSpacing.list()}")

    def validate_cell_spacing(self, x_spacing: str) -> Maybe[str]:
        """validate x spacing"""
        try:
            return Maybe.of(CellSpacing(x_spacing.lower()).value)
        except ValueError as e:
            return Maybe.save_failure(f"{e}. Available options: {CellSpacing.list()}")

    def validate_solver(self, solver: str) -> Maybe[str]:
        """validate the given solver"""
        try:
            return Maybe.of(Solver(solver).value)
        except ValueError as e:
            return Maybe.save_failure(f"{e}. Available options: {Solver.list()}")

    def validate_boundary_conditions(
        self, conditions: Sequence[str]
    ) -> Maybe[Sequence[str]]:
        """validate the given set of boundary conditions"""
        try:
            return Maybe.of([BoundaryCondition(c).value for c in conditions])
        except ValueError as e:
            return Maybe.save_failure(
                f"{e}. Available options: {BoundaryCondition.list()}"
            )

    def validate(self, settings: Any) -> Maybe[Any]:
        validated = settings.copy()
        for name, value in settings.items():
            if name in self.validators:
                result = self.validators[name](value)
                validated[name] = result.unwrap()

        return Maybe.of(validated)
