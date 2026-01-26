# =============================================================================
# memory_estimator.py
#
# accurate memory estimation for simbi simulations.
# accounts for:
#   - ghost zones (halo cells) based on reconstruction method
#   - all allocated arrays: cons, prim, flux, bfield, efield
#   - runge-kutta workspace arrays
#   - mesh refinement levels
#
# usage:
#   from simbi.core.memory_estimator import estimate_memory, MemoryEstimate
#   estimate = estimate_memory(problem)  # or pass a dict
#   print(f"total: {estimate.total_gb:.2f} GB")
# =============================================================================
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ArrayInfo:
    """info about a single allocated array"""

    name: str
    shape: tuple[int, ...]
    dtype_bytes: int = 8
    count: int = 1

    @property
    def bytes(self) -> int:
        return self.count * math.prod(self.shape) * self.dtype_bytes

    @property
    def mb(self) -> float:
        return self.bytes / (1024**2)


@dataclass
class MemoryEstimate:
    """complete memory estimate for a simulation"""

    arrays: list[ArrayInfo] = field(default_factory=list)
    resolution: tuple[int, int, int] = (1, 1, 1)
    halo_width: int = 2
    dimensionality: int = 1
    is_mhd: bool = False
    timestepping: str = "rk2"

    @property
    def total_bytes(self) -> int:
        return sum(a.bytes for a in self.arrays)

    @property
    def total_mb(self) -> float:
        return self.total_bytes / (1024**2)

    @property
    def total_gb(self) -> float:
        return self.total_bytes / (1024**3)

    @property
    def active_cells(self) -> int:
        return math.prod(self.resolution[: self.dimensionality])

    @property
    def allocated_cells(self) -> int:
        """cells including ghost zones"""
        h = self.halo_width
        return math.prod(r + 2 * h for r in self.resolution[: self.dimensionality])

    @property
    def ghost_overhead_percent(self) -> float:
        if self.active_cells == 0:
            return 0.0
        return 100.0 * (self.allocated_cells - self.active_cells) / self.active_cells

    def breakdown(self) -> dict[str, float]:
        """return memory breakdown by category in GB"""
        categories: dict[str, float] = {}
        for arr in self.arrays:
            cat = arr.name.split("[")[0]
            categories[cat] = categories.get(cat, 0.0) + arr.bytes / (1024**3)
        return categories


def get_halo_width(reconstruction: str) -> int:
    """return ghost zone width for reconstruction method"""
    rec = reconstruction.lower()
    if rec == "pcm":
        return 1
    elif rec == "plm":
        return 2
    elif rec == "ppm":
        return 3
    else:
        return 2


def get_nvars(dimensionality: int, is_mhd: bool) -> int:
    """return number of conserved/primitive variables"""
    if is_mhd:
        return 2 * dimensionality + 3
    return dimensionality + 3


def estimate_memory(
    config: dict[str, Any] | Any,
) -> MemoryEstimate:
    """
    estimate memory usage for a simulation.

    accepts either:
    - a dict with simulation parameters
    - a SimbiProblem instance (extracts relevant fields)

    returns a MemoryEstimate with detailed breakdown.
    """
    if hasattr(config, "model_dump"):
        params = config.model_dump()
    elif hasattr(config, "__dict__") and not isinstance(config, dict):
        params = vars(config)
    else:
        params = config

    resolution = params.get("resolution", (1, 1, 1))
    if isinstance(resolution, int):
        resolution = (resolution, 1, 1)
    resolution = tuple(resolution) + (1,) * (3 - len(resolution))
    ni, nj, nk = resolution

    dimensionality = params.get("dimensionality", sum(1 for r in resolution if r > 1))
    if dimensionality == 0:
        dimensionality = 1

    is_mhd = params.get("is_mhd", False)
    regime = params.get("regime", "")
    if hasattr(regime, "value"):
        regime = regime.value
    if "mhd" in str(regime).lower():
        is_mhd = True

    reconstruction = params.get("reconstruction", "plm")
    if hasattr(reconstruction, "value"):
        reconstruction = reconstruction.value
    halo_width = get_halo_width(str(reconstruction))

    timestepping = params.get("timestepping", "rk2")
    if hasattr(timestepping, "value"):
        timestepping = timestepping.value
    timestepping = str(timestepping).lower()

    nvars = get_nvars(dimensionality, is_mhd)

    estimate = MemoryEstimate(
        resolution=(ni, nj, nk),
        halo_width=halo_width,
        dimensionality=dimensionality,
        is_mhd=is_mhd,
        timestepping=timestepping,
    )

    allocated_shape = (ni + 2 * halo_width, nj + 2 * halo_width, nk + 2 * halo_width)
    if dimensionality == 1:
        allocated_shape = (ni + 2 * halo_width,)
    elif dimensionality == 2:
        allocated_shape = (ni + 2 * halo_width, nj + 2 * halo_width)

    cons_bytes = nvars * 8

    estimate.arrays.append(
        ArrayInfo("cons", allocated_shape, dtype_bytes=cons_bytes)
    )
    estimate.arrays.append(
        ArrayInfo("prim", allocated_shape, dtype_bytes=cons_bytes)
    )

    active_shape = resolution[:dimensionality]
    for dd in range(dimensionality):
        flux_shape = list(active_shape)
        flux_shape[dd] += 1
        if is_mhd:
            for tt in range(dimensionality):
                if tt != dd:
                    flux_shape[tt] += 2
        estimate.arrays.append(
            ArrayInfo(f"flux[{dd}]", tuple(flux_shape), dtype_bytes=cons_bytes)
        )

    if is_mhd:
        for dd in range(dimensionality):
            bfield_shape = list(active_shape)
            bfield_shape[dd] += 1
            estimate.arrays.append(
                ArrayInfo(f"bfield[{dd}]", tuple(bfield_shape), dtype_bytes=8)
            )

        for dd in range(dimensionality):
            efield_shape = list(active_shape)
            for tt in range(dimensionality):
                if tt != dd:
                    efield_shape[tt] += 1
            estimate.arrays.append(
                ArrayInfo(f"efield[{dd}]", tuple(efield_shape), dtype_bytes=8)
            )

    if timestepping in ("rk2", "rk3", "rk4"):
        estimate.arrays.append(
            ArrayInfo("workspace.u_n", allocated_shape, dtype_bytes=cons_bytes)
        )
        estimate.arrays.append(
            ArrayInfo("workspace.prim_n", allocated_shape, dtype_bytes=cons_bytes)
        )
        estimate.arrays.append(
            ArrayInfo("workspace.u_star", allocated_shape, dtype_bytes=cons_bytes)
        )

        if is_mhd:
            for dd in range(dimensionality):
                efield_shape = list(active_shape)
                for tt in range(dimensionality):
                    if tt != dd:
                        efield_shape[tt] += 1
                estimate.arrays.append(
                    ArrayInfo(f"workspace.e_n[{dd}]", tuple(efield_shape), dtype_bytes=8)
                )

    refinement_enabled = params.get("refinement_enabled", False)
    if refinement_enabled:
        max_levels = params.get("refinement_max_levels", 1)
        ratios = params.get("refinement_ratios", [])
        regions = params.get("refinement_regions", [])

        base_estimate = estimate.total_bytes
        refined_memory = 0

        # get base domain size for calculating region fractions
        # handle both "bounds" format and split "x1_bounds/x2_bounds/x3_bounds" format
        bounds = params.get("bounds", None)
        if bounds and len(bounds) >= dimensionality:
            domain_size = [abs(b[1] - b[0]) for b in bounds[:dimensionality]]
        else:
            # try split format (x1_bounds, x2_bounds, x3_bounds)
            domain_size = []
            for dd in range(dimensionality):
                key = f"x{dd + 1}_bounds"
                b = params.get(key, (0.0, 1.0))
                if b:
                    domain_size.append(abs(b[1] - b[0]))
                else:
                    domain_size.append(1.0)

        for lvl in range(1, max_levels):
            if lvl <= len(ratios):
                ratio = ratios[lvl - 1]
            else:
                ratio = 2

            # calculate actual cell count for this level
            if regions and lvl <= len(regions):
                # use actual region bounds to compute cell count
                # regions are [xmin, xmax, ymin, ymax, zmin, zmax]
                region = regions[lvl - 1]
                level_cells = 1
                for dd in range(dimensionality):
                    region_lo = region[2 * dd]
                    region_hi = region[2 * dd + 1]
                    region_size = abs(region_hi - region_lo)
                    # cells at this level = region_size / dx_level
                    # dx_level = domain_size / (base_res * cumulative_ratio)
                    cumulative_ratio = 1
                    for rr in range(lvl):
                        cumulative_ratio *= ratios[rr] if rr < len(ratios) else 2
                    base_res = resolution[dd] if dd < len(resolution) else 1
                    dx_level = domain_size[dd] / (base_res * cumulative_ratio)
                    cells_this_dim = int(region_size / dx_level) if dx_level > 0 else 0
                    # add halo
                    cells_this_dim += 2 * halo_width
                    level_cells *= max(cells_this_dim, 1)

                # memory for this level = cells * bytes_per_cell
                bytes_per_cell = base_estimate / estimate.allocated_cells if estimate.allocated_cells > 0 else 0
                refined_memory += level_cells * bytes_per_cell
            else:
                # fallback: assume each level covers ~same volume as finest box
                # this is more accurate for FMR than geometric decay
                level_factor = 1.0 / (ratio ** dimensionality)
                refined_memory += base_estimate * level_factor

        estimate.arrays.append(
            ArrayInfo(
                "refinement_overhead",
                (1,),
                dtype_bytes=int(refined_memory),
            )
        )

    return estimate


def format_memory_report(estimate: MemoryEstimate) -> str:
    """format a human-readable memory report"""
    lines = []
    lines.append("memory estimate")
    lines.append("=" * 50)
    lines.append(f"resolution: {estimate.resolution[:estimate.dimensionality]}")
    lines.append(f"dimensionality: {estimate.dimensionality}D")
    lines.append(f"mhd: {estimate.is_mhd}")
    lines.append(f"halo width: {estimate.halo_width} cells")
    lines.append(f"timestepping: {estimate.timestepping}")
    lines.append("")
    lines.append(f"active cells: {estimate.active_cells:,}")
    lines.append(f"allocated cells (with ghosts): {estimate.allocated_cells:,}")
    lines.append(f"ghost overhead: {estimate.ghost_overhead_percent:.1f}%")
    lines.append("")
    lines.append("array breakdown:")
    lines.append("-" * 50)

    for arr in estimate.arrays:
        if arr.bytes > 0:
            lines.append(f"  {arr.name:25s} {arr.mb:10.2f} MB")

    lines.append("-" * 50)
    lines.append(f"  {'TOTAL':25s} {estimate.total_gb:10.3f} GB")

    breakdown = estimate.breakdown()
    if len(breakdown) > 1:
        lines.append("")
        lines.append("by category:")
        for cat, gb in sorted(breakdown.items(), key=lambda x: -x[1]):
            lines.append(f"  {cat:25s} {gb:10.3f} GB")

    return "\n".join(lines)
