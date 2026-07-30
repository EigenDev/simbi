# =============================================================================
# census.py
#
# reading binned reductions back out of a checkpoint. a census stores, per sample, one
# accumulated value per (bucket, accumulator); this turns that flat table into named,
# bin-shaped arrays and forms the derived quantities the accumulators cannot store.
#
# WHY THE READER FORMS THE MEANS. an accumulator has to be a commutative monoid — the
# reduction runs in parallel, over blocks, across restart segments, so the combine must be
# associative and order-agnostic. sums and extrema are; means, variances and percentiles are
# not, being functions of sums. a census therefore registers `m*v` and `m` and the division
# happens HERE, once, on the reduced totals. a mean accumulated per-block and averaged again
# is weighted by block occupancy rather than by mass, which is a different number that looks
# entirely reasonable.
#
# the same argument gives the Favre (density-weighted) moments for free: the mass-weighted
# mean of q is `sum(m q) / sum(m)`, and its variance `sum(m q^2)/sum(m) - favre(q)^2`, both
# built from sums the run already carries.
#
# usage:
#   c = read_census("chkpt.h5", "shells")
#   r = c.bin_centers(0)
#   mdot = c.value("mass_flux")[-1]            # the last sample, shaped by the bin axes
#   vr = c.favre("mass_vr", "mass")[-1]        # sum(m v) / sum(m), per bin
# =============================================================================

from __future__ import annotations

from dataclasses import dataclass, replace

import h5py
import numpy as np

Array = np.ndarray

__all__ = ["Census", "CensusError", "read_census", "census_names"]


class CensusError(Exception):
    """a census could not be read or a derived quantity was asked for that its accumulators
    cannot support. raised rather than returning a sentinel: a silently empty profile is
    indistinguishable from a physical one."""


@dataclass(frozen=True)
class Census:
    """one census's whole recorded history.

    `values` is (n_rows, *bin_shape, n_values) — the stored segment axis reshaped to the
    per-axis bin counts, last axis varying fastest, matching the order the segment index was
    built in. with no bin axes `bin_shape` is empty and each row is a single global total.

    every row is self-describing. `level` says which refinement level produced it, `n_samples` how
    many samples were folded into it, and `t_start` / `time` the span it covers.

    an ACCUMULATING census stores one row per level folded from many samples with its own reduce
    op, rather than a row apiece: the time series is traded for a whole-segment reduction, so
    `n_samples` is what makes a running sum divisible back into a time average.

    a PER-LEVEL census records each level's own subcycle rather than one composite row per root
    step, so rows from different levels carry different times and cover different volumes. summing
    across levels is therefore the reader's decision, not one baked into the file — use
    `for_level`.
    """

    name: str
    time: Array
    values: Array
    value_names: tuple[str, ...]
    axis_names: tuple[str, ...]
    axis_edges: tuple[Array, ...]
    op: str
    dropped: Array
    accumulated: bool = False
    cadence: str = "root_step"
    level: Array | None = None
    n_samples: Array | None = None
    t_start: Array | None = None

    @property
    def bin_shape(self) -> tuple[int, ...]:
        return tuple(len(e) - 1 for e in self.axis_edges)

    @property
    def n_rows(self) -> int:
        """rows stored in `values`."""
        return int(self.time.size)

    @property
    def levels(self) -> tuple[int, ...]:
        """the refinement levels present, ascending."""
        if self.level is None:
            return (0,)
        return tuple(int(v) for v in np.unique(self.level))

    def for_level(self, level: int) -> "Census":
        """the rows produced by one refinement level.

        rows from different levels are NOT interchangeable: they carry different times, cover
        different volumes, and on a per-level cadence arrive at different rates. selecting before
        any time-series analysis is what keeps a level-2 row taken four times per root step from
        being averaged against a level-0 row taken once.
        """
        if self.level is None:
            raise CensusError(
                f"census '{self.name}' carries no level column; it predates per-level sampling"
            )
        keep = self.level == level
        if not keep.any():
            raise CensusError(
                f"census '{self.name}' has no rows from level {level}; it has {self.levels}"
            )
        return replace(
            self,
            time=self.time[keep],
            values=self.values[keep],
            dropped=self.dropped[keep],
            level=self.level[keep],
            n_samples=None if self.n_samples is None else self.n_samples[keep],
            t_start=None if self.t_start is None else self.t_start[keep],
        )

    def time_average(self, name: str) -> Array:
        """one accumulator averaged over the samples of this run segment, shaped `bin_shape`.

        this is what an accumulating census exists to produce: the stored row is the running
        sum over samples, so the mean is that row divided by the count — and the count is what
        makes it recoverable without the samples themselves. a per-sample census averages its
        rows directly, so the same call means the same thing either way.

        only meaningful for an additive census: a running max over time is already the answer
        for the whole segment and dividing it by a sample count is not an average of anything.
        """
        self._require_additive("a time average")
        if len(self.levels) > 1:
            raise CensusError(
                f"census '{self.name}' holds rows from levels {self.levels}. they cover different "
                "volumes at different rates, so averaging them together is not a time average of "
                "anything; select one with for_level() first."
            )
        if not self.accumulated:
            return self.value(name).mean(axis=0)
        counts = self.n_samples
        if counts is None or counts.sum() == 0:
            raise CensusError(
                f"census '{self.name}' is accumulating but records no sample count; a running "
                "sum cannot be divided back into an average without it"
            )
        # the count-weighted mean over rows: with one row this is the row divided by its count,
        # and it stays right if a level ever contributes more than one.
        return self.value(name).sum(axis=0) / float(counts.sum())

    def bin_centers(self, axis: int = 0) -> Array:
        """midpoints of one axis's bins. these are plotting coordinates, NOT the radii the
        accumulators were evaluated at — each bucket holds a sum over a finite-width bin, so
        the representative coordinate of the contents is whatever the census was asked to
        accumulate (register `mass_r` alongside `mass` if the mass-weighted radius matters)."""
        e = self.axis_edges[self._axis(axis)]
        return 0.5 * (e[:-1] + e[1:])

    def value(self, name: str) -> Array:
        """one accumulator across every stored row, shaped (n_rows, *bin_shape)."""
        return self.values[..., self._value(name)]

    def favre(self, moment: str, weight: str) -> Array:
        """the weight-averaged mean `sum(w q) / sum(w)` per bin, per sample.

        `moment` is the accumulator holding `sum(w q)` and `weight` the one holding `sum(w)`.
        with `weight` the binned mass this is the Favre (density-weighted) mean, which is the
        average a finite-volume scheme actually evolves — a volume-weighted mean of the same
        field is a different quantity and diverges from it wherever the density is structured.

        empty bins return NaN rather than zero: no matter landed there, so the mean is not
        defined, and zero would read as a physical measurement of zero.
        """
        self._require_additive("a weighted mean")
        num = self.value(moment)
        den = self.value(weight)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(den != 0.0, num / np.where(den != 0.0, den, 1.0), np.nan)

    def favre_variance(self, second: str, moment: str, weight: str) -> Array:
        """the weight-averaged variance `sum(w q^2)/sum(w) - favre(q)^2` per bin, per sample.

        `second` holds `sum(w q^2)`. this is the reason a census stores raw powers: the
        variance is a function of two sums, so it cannot be accumulated directly, but it
        falls out of them exactly. the subtraction is cancellation-prone where the spread is
        far below the mean, so a negative result (pure roundoff) is clamped to zero rather
        than returned as a nonsensical negative variance.
        """
        self._require_additive("a weighted variance")
        mean = self.favre(moment, weight)
        with np.errstate(invalid="ignore", divide="ignore"):
            den = self.value(weight)
            m2 = np.where(den != 0.0, self.value(second) / np.where(den != 0.0, den, 1.0), np.nan)
        return np.maximum(m2 - mean * mean, 0.0)

    def total(self, name: str) -> Array:
        """one accumulator summed over every bin, per sample — the global reduction the
        binning refines. only meaningful for an additive census: combining extrema across
        bins is a max, not a sum, and is not what this returns."""
        self._require_additive("a total over bins")
        axes = tuple(range(1, 1 + len(self.bin_shape)))
        return self.values[..., self._value(name)].sum(axis=axes) if axes else self.value(name)

    def assert_fully_binned(self, sample: int | None = None) -> None:
        """every cell landed in a bin.

        a cell whose coordinate falls outside the declared edges is DROPPED, and a census
        that under-covers its domain produces a profile that looks entirely physical while
        omitting an arbitrary part of the grid. the shortfall travels with the numbers so
        this can be checked rather than assumed.
        """
        d = self.dropped if sample is None else self.dropped[sample : sample + 1]
        worst = int(d.max()) if d.size else 0
        if worst:
            where = "" if sample is not None else f" (worst of {self.n_rows} row(s))"
            raise CensusError(
                f"census '{self.name}' dropped {worst} cell(s){where}: their bin coordinate "
                "fell outside the declared edges, so the profile omits part of the grid. "
                "widen the outermost edges or exclude those cells deliberately."
            )

    # ---- internals ----------------------------------------------------------------

    def _axis(self, axis: int) -> int:
        if not -len(self.axis_edges) <= axis < len(self.axis_edges):
            raise CensusError(
                f"census '{self.name}' has {len(self.axis_edges)} bin axis/axes; no axis {axis}"
            )
        return axis % len(self.axis_edges)

    def _value(self, name: str) -> int:
        try:
            return self.value_names.index(name)
        except ValueError:
            raise CensusError(
                f"census '{self.name}' has no accumulator '{name}'; it registered "
                f"{list(self.value_names)}"
            ) from None

    def _require_additive(self, what: str) -> None:
        if self.op != "add":
            raise CensusError(
                f"census '{self.name}' reduces with '{self.op}', so {what} is not defined "
                "over its accumulators — only an additive census carries the sums these are "
                "built from."
            )


def census_names(path: str) -> tuple[str, ...]:
    """the censuses recorded in a checkpoint, in file order."""
    with h5py.File(path, "r") as h:
        return tuple(h["census"].keys()) if "census" in h else ()


def read_census(path: str, name: str) -> Census:
    """read one census's history out of a checkpoint.

    the stored `values` is (n_rows, n_segments, n_values) with the segment axis flat; it
    is reshaped here to the per-axis bin counts in registration order, last axis varying
    fastest — the same order the segment index was built in, so a bin's position in the
    returned array matches its edges.
    """
    with h5py.File(path, "r") as h:
        if "census" not in h or name not in h["census"]:
            raise CensusError(
                f"checkpoint '{path}' records no census '{name}'; it has {list(census_names(path))}"
            )
        g = h["census"][name]
        attrs = g.attrs
        value_names = tuple(_attr_str(attrs, "value_names").split(","))
        n_axes = sum(1 for k in attrs.keys() if k.startswith("axis") and k.endswith("_name"))
        axis_names = tuple(_attr_str(attrs, f"axis{k}_name") for k in range(n_axes))
        axis_edges = tuple(np.asarray(g[f"axis{k}_edges"][...], dtype=float) for k in range(n_axes))
        time = np.asarray(g["time"][...], dtype=float)
        raw = np.asarray(g["values"][...], dtype=float)
        dropped = np.asarray(g["dropped"][...], dtype=np.int64)
        op = _attr_str(attrs, "op")
        n_segments = int(attrs["n_segments"])
        n_values = int(attrs["n_values"])
        # a file written before accumulation existed is, by definition, not accumulated, and its
        # sample count is its row count — so these default rather than fail.
        accumulated = bool(int(attrs.get("accumulated", 0)))
        cadence = _attr_str(attrs, "cadence") if "cadence" in attrs else "root_step"
        level = np.asarray(g["level"][...], dtype=np.int64) if "level" in g else None
        n_folded = np.asarray(g["n_samples"][...], dtype=np.int64) if "n_samples" in g else None
        t_start = np.asarray(g["t_start"][...], dtype=float) if "t_start" in g else None

    bin_shape = tuple(len(e) - 1 for e in axis_edges)
    expected = int(np.prod(bin_shape)) if bin_shape else 1
    if expected != n_segments:
        raise CensusError(
            f"census '{name}': {n_segments} stored segments but the axis edges describe "
            f"{expected} bin(s) {bin_shape}. the file's segment axis and its edges disagree, "
            "so no reshape of it is trustworthy."
        )
    if len(value_names) != n_values:
        raise CensusError(
            f"census '{name}': {n_values} stored accumulators but {len(value_names)} name(s) "
            f"{list(value_names)}; a column cannot be named."
        )

    n_levels = 1 if level is None else int(np.unique(level).size)
    if accumulated and time.size != n_levels:
        raise CensusError(
            f"census '{name}' is marked accumulating but stores {time.size} rows across "
            f"{n_levels} level(s); an accumulating history folds every sample into one row per "
            "level, so the file is inconsistent with its own metadata."
        )

    return Census(
        name=name,
        time=time,
        values=raw.reshape((time.size, *bin_shape, n_values)),
        value_names=value_names,
        axis_names=axis_names,
        axis_edges=axis_edges,
        op=op,
        dropped=dropped,
        accumulated=accumulated,
        cadence=cadence,
        level=level,
        n_samples=n_folded,
        t_start=t_start,
    )


def _attr_str(attrs, key: str) -> str:
    if key not in attrs:
        raise CensusError(f"census group is missing the '{key}' attribute")
    v = attrs[key]
    return v.decode() if isinstance(v, bytes) else str(v)
