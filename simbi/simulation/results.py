# =============================================================================
# results.py
#
# the public run-result surface. a completed run returns a frozen `RunResult`
# tree, assembled explicitly from the backend's typed diagnostics transport by
# named attribute access — never a dictionary, tuple, reflection, or read of a
# process-global counter. every value is the run's own accepted evidence.
# =============================================================================
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = [
    "CellCount",
    "Injection",
    "GuardDiagnostics",
    "ProjectionDiagnostics",
    "RunDiagnostics",
    "RunResult",
    "from_native_diagnostics",
]


@dataclass(frozen=True)
class CellCount:
    """a count of cells and the subset inside a configured horizon."""

    total: int
    inside_horizon: int


@dataclass(frozen=True)
class Injection:
    """a projection contribution to a conserved total: the signed net and the
    gross (L1) magnitude of the activity."""

    signed: float
    gross: float


@dataclass(frozen=True)
class GuardDiagnostics:
    """the FOFC guard acts accepted over the surviving solution: the cells the
    recovery flagged troubled and the cells the correcting select froze."""

    troubled_cells: CellCount
    frozen_cells: CellCount


@dataclass(frozen=True)
class ProjectionDiagnostics:
    """the admissible-boundary projection's accepted intervention totals."""

    passes_fired: int
    projected_cells: int
    min_theta: float
    injected_den: Injection
    injected_nrg: Injection


@dataclass(frozen=True)
class RunDiagnostics:
    """one run's accepted evidence: the projection and guard interventions on the
    states that survived into the solution."""

    projection: ProjectionDiagnostics
    guards: GuardDiagnostics


@dataclass(frozen=True)
class RunResult:
    """the result of a completed run: where its data landed and what the scheme
    did to keep the solution admissible."""

    data_directory: Path
    diagnostics: RunDiagnostics


def _cell_count(native: Any) -> CellCount:
    return CellCount(total=native.total, inside_horizon=native.inside_horizon)


def from_native_diagnostics(native: Any, data_directory: Path) -> RunResult:
    """assemble the public `RunResult` from the backend's `_NativeRunDiagnostics`
    transport, reading each value by its named getter. the transport is a private
    read-only carrier; this is the one place it becomes the public surface."""
    projection = ProjectionDiagnostics(
        passes_fired=native.projection.passes_fired,
        projected_cells=native.projection.projected_cells,
        min_theta=native.projection.min_theta,
        injected_den=Injection(
            signed=native.projection.injected_den_signed,
            gross=native.projection.injected_den_gross,
        ),
        injected_nrg=Injection(
            signed=native.projection.injected_nrg_signed,
            gross=native.projection.injected_nrg_gross,
        ),
    )
    guards = GuardDiagnostics(
        troubled_cells=_cell_count(native.guards.troubled_cells),
        frozen_cells=_cell_count(native.guards.frozen_cells),
    )
    return RunResult(
        data_directory=data_directory,
        diagnostics=RunDiagnostics(projection=projection, guards=guards),
    )
