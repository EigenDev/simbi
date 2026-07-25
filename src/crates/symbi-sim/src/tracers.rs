// =============================================================================
// tracers.rs
//
// lagrangian tracer particles: massless points advected by the interpolated
// gas velocity, each carrying an id, a statistical mass weight, and provenance
// flags. the INTEGRATOR is grid-free — it advances positions with rk2
// (midpoint) against any velocity sampler `fn([f64; D]) -> [f64; D]` — and the
// grid enters only through the bilinear/trilinear sampler adapter over the
// cell-centered primitive velocity. this split keeps the integrator gated by
// analytic flows (uniform line, solid-body rotation) and the adapter gated by
// exactness on linear fields, independently.
//
// seeding is MASS-WEIGHTED and DETERMINISTIC: per-cell tracer counts by
// golden-ratio stratified inversion of the cumulative mass (no rng anywhere —
// restart and cross-driver bitwise gates come free), positions stratified on a
// per-cell sub-lattice. the known bias of velocity-field tracers (clustering
// at convergence zones; Genel et al. 2013) is a property of the METHOD, not
// the seeding — mass-weighted seeding fixes the initial condition only, and
// shock statistics carry the accumulated bias regardless.
//
// a tracer that leaves the domain freezes with its exit state (`escaped`);
// each tracer's mass weight is `m_total_sampled / n`, so counts of crossing
// events convert directly to mass fluxes for the provenance ledgers.
//
// usage:
//  let counts = systematic_counts(&cell_masses, n_tracers);
//  let mut set = TracerSet::<2>::seed_stratified(&cells, &counts, mass_total / n as f64);
//  set.advance_rk2(dt, |x| velocity_sampler(x));
// =============================================================================

/// one tracer's provenance record: the crossing events the accretion ledgers
/// consume. positions/ids live in the parallel SoA vectors of [`TracerSet`].
#[derive(Clone, Copy, Debug, Default)]
pub struct TracerFlags {
    /// left the domain: frozen at its exit state, no further advection.
    pub escaped: bool,
    /// crossed inside a sink/horizon radius: frozen, with the crossing time.
    pub crossed_sink: bool,
    pub crossing_time: f64,
}

/// the tracer population, SoA. `weight` is the statistical mass one tracer
/// represents (total sampled mass / population), shared by construction.
#[derive(Clone, Debug, Default)]
pub struct TracerSet<const D: usize> {
    pub x: Vec<[f64; D]>,
    pub id: Vec<u64>,
    pub flags: Vec<TracerFlags>,
    pub weight: f64,
}

/// stratified apportionment of `n` tracers over cells proportional to their
/// masses: sample points from the sorted GOLDEN-RATIO low-discrepancy sequence
/// `fract((k+1) phi) * M` along the cumulative mass and count how many land in
/// each cell's interval. deterministic, no rng, exact total (sum == n),
/// per-interval deviation O(log n) — and immune to two failure modes of the
/// simpler schemes: largest-remainder degenerates to winner-take-all in the
/// sparse regime (every quota below 1), and UNIFORM strata alias against
/// periodic mass structure in the cell-walk order (a striped density walked
/// column-wise at a rational strata-per-column ratio biases systematically).
pub fn systematic_counts(masses: &[f64], n: usize) -> Vec<usize> {
    let total: f64 = masses.iter().sum();
    let mut counts = vec![0usize; masses.len()];
    if total <= 0.0 || n == 0 {
        return counts;
    }
    const PHI: f64 = 0.618_033_988_749_894_9;
    let mut pts: Vec<f64> = (0..n).map(|k| ((k as f64 + 1.0) * PHI).fract() * total).collect();
    pts.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mut cum = 0.0;
    let mut k = 0usize;
    for (i, m) in masses.iter().enumerate() {
        cum += m;
        while k < n && pts[k] < cum {
            counts[i] += 1;
            k += 1;
        }
    }
    // float shortfall lands the trailing points in the last massive cell.
    if let Some(last) = masses.iter().rposition(|&m| m > 0.0) {
        counts[last] += n - k;
    }
    counts
}

impl<const D: usize> TracerSet<D> {
    /// seed `counts[c]` tracers in each cell, stratified on the cell's own
    /// sub-lattice (deterministic: k of m tracers sits at fraction
    /// (k + 1/2)/m along axis 0, centered on the other axes). `cells` gives
    /// each cell's low corner and widths.
    pub fn seed_stratified(
        cells: &[([f64; D], [f64; D])],
        counts: &[usize],
        weight: f64,
    ) -> Self {
        assert_eq!(cells.len(), counts.len(), "cells/counts length mismatch");
        let mut set = Self { weight, ..Default::default() };
        let mut next_id = 0u64;
        for (ci, &(lo, dx)) in cells.iter().enumerate() {
            let m = counts[ci];
            for k in 0..m {
                let mut p = [0.0; D];
                for a in 0..D {
                    let frac = if a == 0 { (k as f64 + 0.5) / m as f64 } else { 0.5 };
                    p[a] = lo[a] + frac * dx[a];
                }
                set.x.push(p);
                set.id.push(next_id);
                set.flags.push(TracerFlags::default());
                next_id += 1;
            }
        }
        set
    }

    pub fn len(&self) -> usize {
        self.x.len()
    }

    pub fn is_empty(&self) -> bool {
        self.x.is_empty()
    }

    /// advance every live tracer by one rk2 (midpoint) step against the
    /// velocity sampler. escaped/crossed tracers stay frozen.
    pub fn advance_rk2(&mut self, dt: f64, vel: impl Fn([f64; D]) -> [f64; D]) {
        for (i, p) in self.x.iter_mut().enumerate() {
            let f = &self.flags[i];
            if f.escaped || f.crossed_sink {
                continue;
            }
            let v1 = vel(*p);
            let mut mid = *p;
            for a in 0..D {
                mid[a] += 0.5 * dt * v1[a];
            }
            let v2 = vel(mid);
            for a in 0..D {
                p[a] += dt * v2[a];
            }
        }
    }

    /// freeze tracers outside the domain box (exit state kept) and tracers
    /// inside the sink radius about `center` (recording the crossing time).
    /// event scans are separate from advection so a single step cannot both
    /// move and mis-classify a tracer against inconsistent states.
    pub fn scan_events(
        &mut self,
        lo: [f64; D],
        hi: [f64; D],
        sink: Option<([f64; D], f64)>,
        time: f64,
    ) {
        for (i, p) in self.x.iter().enumerate() {
            let f = &mut self.flags[i];
            if f.escaped || f.crossed_sink {
                continue;
            }
            if (0..D).any(|a| p[a] < lo[a] || p[a] > hi[a]) {
                f.escaped = true;
                continue;
            }
            if let Some((c, r)) = sink {
                let mut d2 = 0.0;
                for a in 0..D {
                    let d = p[a] - c[a];
                    d2 += d * d;
                }
                if d2 < r * r {
                    f.crossed_sink = true;
                    f.crossing_time = time;
                }
            }
        }
    }

    /// wrap live tracer positions across PERIODIC axes back into `[lo, hi)`: a tracer that
    /// advected past a periodic boundary re-enters from the far side, so the population is
    /// conserved on a periodic domain instead of freezing at the edge. non-periodic
    /// (outflow) axes are untouched, left for `scan_events` to flag as escaped. call BEFORE
    /// `scan_events` so a periodic crossing is never miscounted as an escape.
    pub fn wrap_periodic(&mut self, lo: [f64; D], hi: [f64; D], periodic: [bool; D]) {
        for (i, p) in self.x.iter_mut().enumerate() {
            if self.flags[i].escaped || self.flags[i].crossed_sink {
                continue;
            }
            for a in 0..D {
                let l = hi[a] - lo[a];
                if periodic[a] && l > 0.0 {
                    // (x - lo) mod l, into [0, l), then shift back -> [lo, hi).
                    p[a] = lo[a] + (p[a] - lo[a]).rem_euclid(l);
                }
            }
        }
    }

    /// the accreted tracer mass: crossing count times the per-tracer weight —
    /// the quantity the G-flux gate compares against the sink's Mdot ledger.
    pub fn crossed_mass(&self) -> f64 {
        self.flags.iter().filter(|f| f.crossed_sink).count() as f64 * self.weight
    }
}

/// bilinear/trilinear sampler over cell-centered per-axis velocity grids on a
/// uniform cartesian chart: `vels[a]` is component a, flattened axis-0-fastest
/// over `n` cells per axis (INCLUDING any ghost band the caller sliced in);
/// `x_lo`/`dx` describe that same box. positions are clamped to the outermost
/// cell centers, so sampling at the domain edge extrapolates constantly rather
/// than reading out of bounds.
pub fn grid_velocity_sampler<'a, const D: usize>(
    vels: [&'a [f64]; D],
    n: [usize; D],
    x_lo: [f64; D],
    dx: [f64; D],
) -> impl Fn([f64; D]) -> [f64; D] + 'a {
    move |p: [f64; D]| {
        // fractional cell-center coordinates, clamped inside the sampled box.
        let mut base = [0usize; D];
        let mut frac = [0.0f64; D];
        for a in 0..D {
            let u = ((p[a] - x_lo[a]) / dx[a] - 0.5).clamp(0.0, (n[a] - 1) as f64);
            let b = (u.floor() as usize).min(n[a] - 2);
            base[a] = b;
            frac[a] = u - b as f64;
        }
        let stride = |c: [usize; D]| -> usize {
            let mut lin = 0;
            let mut s = 1;
            for a in 0..D {
                lin += c[a] * s;
                s *= n[a];
            }
            lin
        };
        let mut out = [0.0; D];
        for a in 0..D {
            let mut acc = 0.0;
            // corners of the D-cube around the position.
            for corner in 0..(1usize << D) {
                let mut c = base;
                let mut w = 1.0;
                for ax in 0..D {
                    if corner & (1 << ax) != 0 {
                        c[ax] += 1;
                        w *= frac[ax];
                    } else {
                        w *= 1.0 - frac[ax];
                    }
                }
                acc += w * vels[a][stride(c)];
            }
            out[a] = acc;
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn apportionment_is_exact_and_proportional() {
        let masses = vec![1.0, 2.0, 3.0, 4.0];
        let counts = systematic_counts(&masses, 1000);
        assert_eq!(counts.iter().sum::<usize>(), 1000);
        // the golden-ratio sequence's discrepancy bounds each interval to a
        // few points of its exact quota (O(log n), not the uniform-strata 1).
        for (m, c) in masses.iter().zip(&counts) {
            let quota = 1000.0 * m / 10.0;
            assert!((*c as f64 - quota).abs() <= 4.0, "count {c} vs quota {quota}");
        }
        assert_eq!(counts, systematic_counts(&masses, 1000));
    }

    #[test]
    fn sparse_apportionment_stays_proportional() {
        // n far below the cell count — the regime where largest-remainder
        // degenerates to winner-take-all on the densest cells: 2/5 of the
        // cells carry rho 3 (2/3 of the mass) and must get ~2/3 of a sparse
        // population, spread across the WHOLE band, not stacked densest-first.
        let mut masses = vec![1.0; 2000];
        for m in masses.iter_mut().take(800) {
            *m = 3.0;
        }
        let counts = systematic_counts(&masses, 300);
        assert_eq!(counts.iter().sum::<usize>(), 300);
        let in_band: usize = counts[..800].iter().sum();
        assert!((195..=205).contains(&in_band), "band got {in_band}, expected ~200");
        // and the band allocation is spread, not concentrated at its head.
        assert!(counts[..800].iter().filter(|&&c| c > 0).count() > 150);
    }

    #[test]
    fn uniform_flow_carries_tracers_on_the_analytic_line() {
        let cells = vec![([0.0, 0.0], [1.0, 1.0])];
        let mut set = TracerSet::<2>::seed_stratified(&cells, &[4], 1.0);
        let x0 = set.x.clone();
        let dt = 0.01;
        for _ in 0..100 {
            set.advance_rk2(dt, |_| [0.3, -0.2]);
        }
        for (p, p0) in set.x.iter().zip(&x0) {
            assert!((p[0] - (p0[0] + 0.3)).abs() < 1e-12, "x drifted: {p:?} from {p0:?}");
            assert!((p[1] - (p0[1] - 0.2)).abs() < 1e-12, "y drifted");
        }
    }

    #[test]
    fn solid_body_rotation_orbits_close_at_second_order() {
        // v = omega x r about the origin: rk2 closes an orbit to O(dt^2 per
        // step, dt^2 accumulated over the period).
        const OMEGA: f64 = 1.0;
        let mut set = TracerSet::<2> {
            x: vec![[1.0, 0.0]],
            id: vec![0],
            flags: vec![TracerFlags::default()],
            weight: 1.0,
        };
        let period = 2.0 * std::f64::consts::PI / OMEGA;
        let n = 2000;
        let dt = period / n as f64;
        for _ in 0..n {
            set.advance_rk2(dt, |p| [-OMEGA * p[1], OMEGA * p[0]]);
        }
        let err = ((set.x[0][0] - 1.0).powi(2) + set.x[0][1].powi(2)).sqrt();
        assert!(err < 5e-3, "orbit failed to close: err = {err}");
        // radius drift bounds the integrator's dissipation/growth.
        let r = (set.x[0][0].powi(2) + set.x[0][1].powi(2)).sqrt();
        assert!((r - 1.0).abs() < 5e-3, "radius drifted to {r}");
    }

    #[test]
    fn linear_velocity_field_interpolates_exactly() {
        // bilinear interpolation is exact on fields linear in each coordinate:
        // v_x = 2 + x + 2y, v_y = -1 + 0.5x - y, sampled far from the clamp.
        const N: usize = 8;
        let (x_lo, dx) = ([0.0, 0.0], [1.0, 1.0]);
        let mut vx = vec![0.0; N * N];
        let mut vy = vec![0.0; N * N];
        for j in 0..N {
            for i in 0..N {
                let (x, y) = (x_lo[0] + (i as f64 + 0.5) * dx[0], x_lo[1] + (j as f64 + 0.5) * dx[1]);
                vx[i + j * N] = 2.0 + x + 2.0 * y;
                vy[i + j * N] = -1.0 + 0.5 * x - y;
            }
        }
        let sampler = grid_velocity_sampler::<2>([&vx, &vy], [N, N], x_lo, dx);
        for &(px, py) in &[(2.3, 3.7), (4.0, 4.0), (1.6, 5.9)] {
            let v = sampler([px, py]);
            assert!((v[0] - (2.0 + px + 2.0 * py)).abs() < 1e-12, "vx at ({px},{py}): {}", v[0]);
            assert!((v[1] - (-1.0 + 0.5 * px - py)).abs() < 1e-12, "vy at ({px},{py}): {}", v[1]);
        }
    }

    #[test]
    fn escaped_and_crossed_tracers_freeze() {
        let mut set = TracerSet::<2> {
            x: vec![[0.5, 0.5], [5.0, 0.5], [0.05, 0.05]],
            id: vec![0, 1, 2],
            flags: vec![TracerFlags::default(); 3],
            weight: 2.5,
        };
        set.scan_events([0.0, 0.0], [1.0, 1.0], Some(([0.0, 0.0], 0.1)), 3.25);
        assert!(!set.flags[0].escaped && !set.flags[0].crossed_sink);
        assert!(set.flags[1].escaped);
        assert!(set.flags[2].crossed_sink);
        assert_eq!(set.flags[2].crossing_time, 3.25);
        assert_eq!(set.crossed_mass(), 2.5);
        let frozen = [set.x[1], set.x[2]];
        set.advance_rk2(0.1, |_| [1.0, 1.0]);
        assert_eq!(set.x[1], frozen[0], "escaped tracer moved");
        assert_eq!(set.x[2], frozen[1], "crossed tracer moved");
        assert!(set.x[0] != [0.5, 0.5], "live tracer failed to move");
    }

    // THE periodic-conservation contract: a tracer that advects past a periodic boundary
    // must WRAP back into the domain (population conserved), not be marked escaped and lost.
    // this reproduces the periodic-KH leak -- ambient tracers drift off one edge and never
    // reappear on the far side -- and locks the fix.
    #[test]
    fn periodic_tracers_wrap_and_do_not_escape() {
        let lo = [0.0, 0.0];
        let hi = [1.0, 1.0];
        // x periodic, y outflow. one tracer past +x, one past -x, one past the +y (outflow)
        // edge, one interior.
        let mut set = TracerSet::<2> {
            x: vec![[1.2, 0.5], [-0.3, 0.5], [0.5, 1.4], [0.4, 0.6]],
            id: vec![0, 1, 2, 3],
            flags: vec![TracerFlags::default(); 4],
            weight: 1.0,
        };
        set.wrap_periodic(lo, hi, [true, false]);
        // periodic x wraps into [0, 1); y (non-periodic) is left alone for the scan.
        assert!((set.x[0][0] - 0.2).abs() < 1e-12, "x=1.2 did not wrap to 0.2");
        assert!((set.x[1][0] - 0.7).abs() < 1e-12, "x=-0.3 did not wrap to 0.7");
        assert_eq!(set.x[2][1], 1.4, "outflow y wrongly wrapped");

        set.scan_events(lo, hi, None, 0.0);
        // the two periodic crossers survived (in-domain after wrap); only the outflow-y
        // crosser escaped. population conserved on the periodic axis.
        assert!(!set.flags[0].escaped, "periodic +x crosser lost");
        assert!(!set.flags[1].escaped, "periodic -x crosser lost");
        assert!(set.flags[2].escaped, "outflow-y crosser should escape");
        assert!(!set.flags[3].escaped);
        assert_eq!(
            set.flags.iter().filter(|f| !f.escaped).count(),
            3,
            "exactly the 3 in-domain tracers remain live"
        );
    }

    // a fully-periodic box loses NO tracers however far they advect: many wrap-arounds keep
    // every tracer live and in-domain.
    #[test]
    fn fully_periodic_box_conserves_every_tracer() {
        let mut set = TracerSet::<2> {
            x: (0..50).map(|k| [k as f64 * 0.02, 0.5]).collect(),
            id: (0..50).collect(),
            flags: vec![TracerFlags::default(); 50],
            weight: 1.0,
        };
        // advect a big uniform drift many steps (several box-lengths), wrapping each step.
        for _ in 0..200 {
            set.advance_rk2(0.05, |_| [1.0, 1.0]);
            set.wrap_periodic([0.0, 0.0], [1.0, 1.0], [true, true]);
            set.scan_events([0.0, 0.0], [1.0, 1.0], None, 0.0);
        }
        assert_eq!(set.flags.iter().filter(|f| f.escaped).count(), 0, "a periodic box lost tracers");
        for p in &set.x {
            assert!((0.0..1.0).contains(&p[0]) && (0.0..1.0).contains(&p[1]), "tracer left the box: {p:?}");
        }
    }
}

// =============================================================================
// grid-coupled layer: mass-weighted seeding from the conserved density and the
// once-per-step advance against the post-step primitive velocity. host-side
// (the sampler reads the field buffers directly); uniform cartesian charts
// (constant cell volume, so per-cell mass is proportional to den).
// =============================================================================

use crate::state::FieldStore;
use symbi_xpu::MemorySpace;

/// seed `n` tracers mass-weighted over the interior density, stratified per
/// cell, deterministic. the per-tracer weight is the total sampled mass over n.
pub fn seed_mass_weighted<const D: usize, const DOF: usize, Mem: MemorySpace>(
    sim: &FieldStore<D, DOF, Mem, f64>,
    n: usize,
) -> TracerSet<D> {
    let interior = sim.geom.interior.clone();
    let mut masses = Vec::new();
    let mut cells: Vec<([f64; D], [f64; D])> = Vec::new();
    for c in interior.iter() {
        masses.push(*sim.fields.cons.den.view().at(c));
        let mut lo = [0.0; D];
        let mut dxs = [0.0; D];
        for a in 0..D {
            lo[a] = sim.geom.x_lo[a]
                + (c[a] - interior.spaces[a].lo) as f64 * sim.geom.dx[a];
            dxs[a] = sim.geom.dx[a];
        }
        cells.push((lo, dxs));
    }
    let counts = systematic_counts(&masses, n);
    let vol: f64 = sim.geom.dx[..D].iter().product();
    let weight = if n == 0 { 0.0 } else { masses.iter().sum::<f64>() * vol / n as f64 };
    TracerSet::seed_stratified(&cells, &counts, weight)
}

/// advance the tracer population by one fluid step against the CURRENT
/// primitive velocity (post-step, ghost bands filled), then scan events:
/// domain escape, and crossing of the first accreting body's sink radius.
/// shared by every driver — the per-driver call sites are guarded by the
/// cross-driver bitwise trajectory gate rather than the recorder law (this is
/// driver-level shared code, invisible to the kernel-set seam by design).
pub fn advance_tracers<const D: usize, const DOF: usize, Mem: MemorySpace>(
    sim: &mut FieldStore<D, DOF, Mem, f64>,
) {
    let Some(mut tr) = sim.tracers.take() else { return };
    advance_positions(sim, &mut tr);
    let (lo, hi) = interior_box(sim);
    // periodic crossings wrap (conserved population); the escape scan then only fires on
    // non-periodic (outflow) axes.
    tr.wrap_periodic(lo, hi, periodic_axes(sim));
    tr.scan_events(lo, hi, sink_of(sim), sim.time);
    sim.tracers = Some(tr);
}

/// per-axis periodicity from the store's boundaries: an axis is periodic when its lo face is
/// (a periodic axis is periodic on both faces). for a decomposed run the low-corner tile
/// carries the physical lo-face boundary, so reading tile 0 gives the GLOBAL periodicity.
fn periodic_axes<const D: usize, const DOF: usize, Mem: MemorySpace>(
    sim: &FieldStore<D, DOF, Mem, f64>,
) -> [bool; D] {
    std::array::from_fn(|a| matches!(sim.boundaries.0[a][0], crate::state::BoundaryType::Periodic))
}

/// advect the population one step against `sim`'s post-step primitive velocity, sampled
/// over the allocated grid (ghost bands current). the POSITION update only -- event scans
/// (escape / sink) are separate so a decomposed step can advance per tile, MIGRATE across
/// cuts, then scan once against the global domain.
fn advance_positions<const D: usize, const DOF: usize, Mem: MemorySpace>(
    sim: &FieldStore<D, DOF, Mem, f64>,
    tr: &mut TracerSet<D>,
) {
    let alloc = &sim.geom.allocated;
    let mut n = [0usize; D];
    let mut x_lo_alloc = [0.0; D];
    let mut dxs = [0.0; D];
    let ng = sim.geom.ng as f64;
    let mut volume = 1usize;
    for a in 0..D {
        n[a] = (alloc.spaces[a].hi - alloc.spaces[a].lo) as usize;
        dxs[a] = sim.geom.dx[a];
        x_lo_alloc[a] = sim.geom.x_lo[a] - ng * dxs[a];
        volume *= n[a];
    }
    // the first D velocity components advect positions; a DOF-lifted swirl
    // component is out-of-plane and does not move a tracer on this grid.
    let slices: [&[f64]; D] = std::array::from_fn(|a| unsafe {
        std::slice::from_raw_parts(sim.fields.prim.vel[a].as_ptr(), volume)
    });
    let sampler = grid_velocity_sampler(slices, n, x_lo_alloc, dxs);
    tr.advance_rk2(sim.dt, sampler);
}

/// the sim's interior physical box `[x_lo, x_lo + n_int*dx)` per axis -- the escape test
/// domain. for a single-grid run this IS the whole domain; for a decomposed tile it is only
/// the tile's slab, so the decomposed step scans against the GLOBAL box instead.
fn interior_box<const D: usize, const DOF: usize, Mem: MemorySpace>(
    sim: &FieldStore<D, DOF, Mem, f64>,
) -> ([f64; D], [f64; D]) {
    let mut lo = [0.0; D];
    let mut hi = [0.0; D];
    for a in 0..D {
        let n_int = (sim.geom.interior.spaces[a].hi - sim.geom.interior.spaces[a].lo) as f64;
        lo[a] = sim.geom.x_lo[a];
        hi[a] = sim.geom.x_lo[a] + n_int * sim.geom.dx[a];
    }
    (lo, hi)
}

/// the first accreting body's sink mask (center, radius) in cartesian world coordinates, or
/// None. tiles replicate the bodies at their global positions, so any tile yields the sink.
fn sink_of<const D: usize, const DOF: usize, Mem: MemorySpace>(
    sim: &FieldStore<D, DOF, Mem, f64>,
) -> Option<([f64; D], f64)> {
    sim.immersed.as_ref().and_then(|im| {
        im.bodies.bodies().iter().find(|b| b.has_accretion()).and_then(|b| {
            b.accretion_radius().map(|r| {
                let mut c = [0.0; D];
                for a in 0..D {
                    c[a] = b.position[a];
                }
                (c, r)
            })
        })
    })
}

/// advance the tracers of a DECOMPOSED run one step: advect each tile's population against
/// its own (halo-filled) velocity, MIGRATE tracers that crossed a cut into the tile that now
/// owns them, then scan escape/sink against the GLOBAL domain. a tracer near a cut is
/// advected by its owning tile whose ghost band the halo exchange filled from the neighbor,
/// so the trajectory is bit-identical to the monolithic sampler; migration only re-homes it.
/// tiles carry EQUAL cell counts on a uniform Cartesian grid, so the owning tile of a
/// position is a floor-divide -- decomposed tracers are gated to that case upstream.
pub fn advance_tracers_decomposed<const D: usize, const DOF: usize, Mem: MemorySpace>(
    stores: &mut [&mut FieldStore<D, DOF, Mem, f64>],
    counts: [usize; D],
) {
    if stores.iter().all(|s| !s.has_tracers()) {
        return;
    }
    // advect per tile (position only), leaving the tracers in place.
    for s in stores.iter_mut() {
        if let Some(mut tr) = s.tracers.take() {
            advance_positions(*s, &mut tr);
            s.tracers = Some(tr);
        }
    }
    // the global domain box = tile (0,..,0)'s lower corner extended by the full cell count.
    let (mut glo, mut ghi) = ([0.0; D], [0.0; D]);
    for a in 0..D {
        let n_int = (stores[0].geom.interior.spaces[a].hi - stores[0].geom.interior.spaces[a].lo)
            as f64;
        glo[a] = stores[0].geom.x_lo[a];
        ghi[a] = glo[a] + n_int * counts[a] as f64 * stores[0].geom.dx[a];
    }
    // wrap periodic crossings into the GLOBAL box BEFORE migration, so a tracer that left one
    // periodic edge re-homes to the tile at the opposite edge. tile 0 is the low corner, so it
    // carries the physical (global) boundary.
    let periodic = periodic_axes(&*stores[0]);
    for s in stores.iter_mut() {
        if let Some(mut tr) = s.tracers.take() {
            tr.wrap_periodic(glo, ghi, periodic);
            s.tracers = Some(tr);
        }
    }
    migrate_tracers(stores, counts);
    let sink = stores.iter().find_map(|s| sink_of(*s));
    for s in stores.iter_mut() {
        if let Some(mut tr) = s.tracers.take() {
            tr.scan_events(glo, ghi, sink, s.time);
            s.tracers = Some(tr);
        }
    }
}

/// the flat tile index owning a physical position, or None if outside the global domain. the
/// tile grid is uniform, so it is a floor-divide by the per-tile extent; the flat index is the
/// SAME `flatten` the decomposition addresses tiles by, so a tracer lands in the store the
/// decomposition calls its owner.
fn tile_owner<const D: usize>(
    x: &[f64; D],
    glo: [f64; D],
    extent: [f64; D],
    counts: [usize; D],
) -> Option<usize> {
    let mut tc = [0usize; D];
    for a in 0..D {
        if x[a] < glo[a] || x[a] >= glo[a] + extent[a] * counts[a] as f64 {
            return None;
        }
        let idx = ((x[a] - glo[a]) / extent[a]).floor() as isize;
        tc[a] = idx.clamp(0, counts[a] as isize - 1) as usize;
    }
    Some(crate::decomp::flatten(tc, counts))
}

/// the global lower corner and per-tile physical extent of a uniform decomposition, read from
/// tile 0 (its interior cell count is the per-tile count `m`).
fn tile_geometry<const D: usize, const DOF: usize, Mem: MemorySpace>(
    tile0: &FieldStore<D, DOF, Mem, f64>,
) -> ([f64; D], [f64; D]) {
    let mut glo = [0.0; D];
    let mut extent = [0.0; D];
    for a in 0..D {
        let m = (tile0.geom.interior.spaces[a].hi - tile0.geom.interior.spaces[a].lo) as usize;
        glo[a] = tile0.geom.x_lo[a];
        extent[a] = m as f64 * tile0.geom.dx[a];
    }
    (glo, extent)
}

/// seed `n` tracers from `global`'s density (the monolithic seeding) and split them across the
/// `counts` tiles by initial position, returning one set per tile in flat order. the monolithic
/// and decomposed runs thus start from the IDENTICAL population (same ids, same positions), so a
/// decomposed run can be gated against the single-grid trajectories.
pub fn seed_and_partition<const D: usize, const DOF: usize, Mem: MemorySpace>(
    global: &FieldStore<D, DOF, Mem, f64>,
    n: usize,
    counts: [usize; D],
) -> Vec<TracerSet<D>> {
    let set = seed_mass_weighted(global, n);
    // the per-tile extent from the FULL-SIZE global grid: interior cells / tile counts, times dx.
    let mut glo = [0.0; D];
    let mut extent = [0.0; D];
    for a in 0..D {
        let n_int =
            (global.geom.interior.spaces[a].hi - global.geom.interior.spaces[a].lo) as usize;
        glo[a] = global.geom.x_lo[a];
        extent[a] = (n_int / counts[a]) as f64 * global.geom.dx[a];
    }
    let ntiles: usize = counts.iter().product();
    let mut per_tile: Vec<TracerSet<D>> = (0..ntiles)
        .map(|_| TracerSet { weight: set.weight, ..Default::default() })
        .collect();
    for i in 0..set.x.len() {
        // the seed is inside the domain by construction, so `tile_owner` is always Some.
        let dest = tile_owner(&set.x[i], glo, extent, counts).unwrap_or(0);
        per_tile[dest].x.push(set.x[i]);
        per_tile[dest].id.push(set.id[i]);
        per_tile[dest].flags.push(set.flags[i]);
    }
    per_tile
}

fn migrate_tracers<const D: usize, const DOF: usize, Mem: MemorySpace>(
    stores: &mut [&mut FieldStore<D, DOF, Mem, f64>],
    counts: [usize; D],
) {
    let (glo, extent) = tile_geometry(&*stores[0]);
    let owner = |x: &[f64; D]| -> Option<usize> { tile_owner(x, glo, extent, counts) };

    // collect (dest_tile, x, id, flags) for every tracer that must move, removing it from its
    // source tile by swap_remove (SoA-consistent). done in one pass per tile so a tracer is
    // considered exactly once this step.
    let mut moved: Vec<(usize, [f64; D], u64, TracerFlags)> = Vec::new();
    for (src, s) in stores.iter_mut().enumerate() {
        let Some(tr) = s.tracers.as_mut() else { continue };
        let mut i = 0;
        while i < tr.x.len() {
            let f = tr.flags[i];
            // a frozen tracer (escaped/crossed) stays; otherwise its owner is by position, and
            // an owner outside the domain (None) stays too, for the global scan to flag it.
            let dest = if f.escaped || f.crossed_sink { Some(src) } else { owner(&tr.x[i]) };
            match dest {
                Some(d) if d != src => {
                    let x = tr.x.swap_remove(i);
                    let id = tr.id.swap_remove(i);
                    let fl = tr.flags.swap_remove(i);
                    moved.push((d, x, id, fl));
                    // do not advance i: swap_remove moved the last element into slot i.
                }
                _ => i += 1,
            }
        }
    }
    for (dest, x, id, fl) in moved {
        let tr = stores[dest].tracers.as_mut().expect("dest tile carries a tracer set");
        tr.x.push(x);
        tr.id.push(id);
        tr.flags.push(fl);
    }
}
