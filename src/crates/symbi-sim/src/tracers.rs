// =============================================================================
// tracers.rs
//
// discrete mass-transport tracers owned by fluid cells or material reservoirs.
// accepted finite-volume mass fluxes define transition probabilities, and a
// deterministic low-variance sampler advances ownership at every ssp stage.
//
// seeding is mass-weighted and deterministic: per-cell tracer counts by
// golden-ratio stratified inversion of the cumulative mass (no rng anywhere —
// restart and cross-driver bitwise gates come free), positions stratified on a
// per-cell sub-lattice. position is derived display state at the owning cell
// centroid; it never determines transport.
//
// usage:
//  let counts = systematic_counts(&cell_masses, n_tracers);
//  let mut set = TracerSet::<2>::seed_stratified(&cells, &counts, mass_total / n as f64);
// =============================================================================

/// one tracer's provenance record: the crossing events the accretion ledgers
/// consume. positions/ids live in the parallel SoA vectors of [`TracerSet`].
#[derive(Clone, Copy, Debug, Default, PartialEq)]
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
    /// material container that owns each tracer. cell containers use the
    /// interior's axis-0-fastest linear index.
    pub owner: Vec<crate::mass_transport::ContainerId>,
    /// step-entry ancestry retained across all ssp stages.
    pub step_owner: Vec<crate::mass_transport::ContainerId>,
    pub step_flags: Vec<TracerFlags>,
    pub run_seed: u64,
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
    let mut pts: Vec<f64> = (0..n)
        .map(|k| ((k as f64 + 1.0) * PHI).fract() * total)
        .collect();
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
    pub fn seed_stratified(cells: &[([f64; D], [f64; D])], counts: &[usize], weight: f64) -> Self {
        assert_eq!(cells.len(), counts.len(), "cells/counts length mismatch");
        let mut set = Self {
            weight,
            ..Default::default()
        };
        let mut next_id = 0u64;
        for (ci, &(lo, dx)) in cells.iter().enumerate() {
            let m = counts[ci];
            for k in 0..m {
                let mut p = [0.0; D];
                for a in 0..D {
                    let frac = if a == 0 {
                        (k as f64 + 0.5) / m as f64
                    } else {
                        0.5
                    };
                    p[a] = lo[a] + frac * dx[a];
                }
                set.x.push(p);
                set.id.push(next_id);
                set.flags.push(TracerFlags::default());
                set.owner
                    .push(crate::mass_transport::ContainerId(ci as u64));
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

    /// the accreted tracer mass: crossing count times the per-tracer weight —
    /// the quantity the G-flux gate compares against the sink's Mdot ledger.
    pub fn crossed_mass(&self) -> f64 {
        self.flags.iter().filter(|f| f.crossed_sink).count() as f64 * self.weight
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
            assert!(
                (*c as f64 - quota).abs() <= 4.0,
                "count {c} vs quota {quota}"
            );
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
        assert!(
            (195..=205).contains(&in_band),
            "band got {in_band}, expected ~200"
        );
        // and the band allocation is spread, not concentrated at its head.
        assert!(counts[..800].iter().filter(|&&c| c > 0).count() > 150);
    }

    #[test]
    fn accepted_mass_flux_moves_the_low_variance_quota() {
        use crate::state::{Boundaries, BoundaryType, SimState, Timestepping};
        use symbi_geometry::Cartesian;
        use symbi_hydro::eos::IdealGas;
        use symbi_hydro::newtonian::Newtonian;
        use symbi_xpu::{CpuSpace, HostMemory};

        let mut sim =
            SimState::<Newtonian, 1, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>::new(
                Newtonian,
                IdealGas { gamma: 1.4 },
                Cartesian,
                [2],
                [0.0],
                [1.0],
                2,
                Boundaries::uniform(BoundaryType::Outflow),
                0.4,
                Timestepping::Euler,
                0,
            )
            .unwrap();
        for coord in sim.geom.interior.iter() {
            sim.fields.cons.den.view_mut().set(coord, 1.0);
            sim.workspace.u_stage.den.view_mut().set(coord, 1.0);
        }
        for coord in sim.geom.allocated.iter() {
            sim.fields.flux[0].den.view_mut().set(coord, 0.0);
        }
        let internal_face = [sim.geom.interior.spaces[0].lo + 1];
        sim.fields.flux[0].den.view_mut().set(internal_face, 0.5);
        sim.dt = 0.5;
        let cells = [([0.0], [1.0]), ([1.0], [1.0])];
        sim.tracers = Some(TracerSet::seed_stratified(&cells, &[100, 100], 0.01));
        snapshot_transport_state(&mut sim);

        advance_stage_mass_transport(&mut sim, 0.0, 1.0, 0).unwrap();

        let tracers = sim.tracers.as_ref().unwrap();
        let in_left = tracers.owner.iter().filter(|owner| owner.0 == 0).count();
        let in_right = tracers.owner.iter().filter(|owner| owner.0 == 1).count();
        assert_eq!((in_left, in_right), (75, 125));
        assert_eq!(tracers.flags.iter().filter(|flag| flag.escaped).count(), 0);
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
            lo[a] = sim.geom.x_lo[a] + (c[a] - interior.spaces[a].lo) as f64 * sim.geom.dx[a];
            dxs[a] = sim.geom.dx[a];
        }
        cells.push((lo, dxs));
    }
    let counts = systematic_counts(&masses, n);
    let vol: f64 = sim.geom.dx[..D].iter().product();
    let weight = if n == 0 {
        0.0
    } else {
        masses.iter().sum::<f64>() * vol / n as f64
    };
    let mut tracers = TracerSet::seed_stratified(&cells, &counts, weight);
    tracers.step_owner = tracers.owner.clone();
    tracers.step_flags = tracers.flags.clone();
    tracers
}

/// retain the ownership and provenance state that supplies the `a0*u_n`
/// ancestry branch of every ssp stage in the step.
pub fn snapshot_transport_state<const D: usize, const DOF: usize, Mem: MemorySpace>(
    sim: &mut FieldStore<D, DOF, Mem, f64>,
) {
    let Some(tracers) = sim.tracers.as_mut() else {
        return;
    };
    tracers.step_owner.clone_from(&tracers.owner);
    tracers.step_flags.clone_from(&tracers.flags);
}

/// advance cell-owned tracers through one accepted forward-euler mass-flux
/// kernel, then apply the ssp convex ancestry selection.
pub fn advance_stage_mass_transport<R, const D: usize, const DOF: usize, M, E, S, Mem>(
    sim: &mut crate::state::SimStateGeneric<R, D, DOF, M, E, S, Mem>,
    a0: f64,
    ac: f64,
    stage: usize,
) -> Result<(), String>
where
    R: symbi_hydro::regime::Regime<f64, D>,
    M: symbi_geometry::Metric<f64, D> + Copy,
    E: symbi_hydro::eos::Eos<f64>,
    S: symbi_xpu::ExecutionSpace,
    Mem: MemorySpace,
{
    let geometry = sim.geom.block_geometry(sim.physics.metric);
    let layout = TransportLayout::single(&sim.geom.interior);
    advance_stage_mass_transport_store(&mut sim.store, &geometry, layout, a0, ac, stage)
}

/// global cell addressing for a single grid or one tile of a decomposition.
#[derive(Clone, Copy, Debug)]
pub struct TransportLayout<const D: usize> {
    pub global_cells: [usize; D],
    pub tile_offset: [usize; D],
}

impl<const D: usize> TransportLayout<D> {
    pub fn single(domain: &symbi_algebra::Domain<D>) -> Self {
        Self {
            global_cells: std::array::from_fn(|dd| domain.spaces[dd].size()),
            tile_offset: [0; D],
        }
    }
}

/// advance one transport stage when the driver owns a field store and its
/// material-volume geometry separately.
pub fn advance_stage_mass_transport_store<const D: usize, const DOF: usize, M, Mem>(
    sim: &mut FieldStore<D, DOF, Mem, f64>,
    geometry: &symbi_geometry::BlockGeometry<M, f64, D>,
    layout: TransportLayout<D>,
    a0: f64,
    ac: f64,
    stage: usize,
) -> Result<(), String>
where
    M: symbi_geometry::Metric<f64, D> + Copy,
    Mem: MemorySpace,
{
    use crate::mass_transport::{
        ContainerId, MassTransfer, SamplingKey, TransportKernel, sample_convex_blend,
        sample_systematic,
    };
    use std::collections::BTreeMap;

    let Some(mut tracers) = sim.tracers.take() else {
        return Ok(());
    };
    if tracers.owner.len() != tracers.id.len() {
        return Err("tracer ownership length does not match tracer ids".to_string());
    }
    if tracers.step_owner.len() != tracers.id.len() {
        return Err("tracer step snapshot is missing".to_string());
    }
    if sim.motion.homologous {
        return Err("mass-transport tracers with mesh motion are not wired".to_string());
    }

    let interior = sim.geom.interior.clone();
    let stage_input = sim.stage_input();
    let key = SamplingKey {
        run_seed: tracers.run_seed,
        epoch: sim.iteration.wrapping_mul(4).wrapping_add(stage as u64),
    };

    let mut by_source = BTreeMap::<ContainerId, Vec<u64>>::new();
    for (ii, &owner) in tracers.owner.iter().enumerate() {
        if !tracers.flags[ii].escaped && !tracers.flags[ii].crossed_sink {
            by_source.entry(owner).or_default().push(tracers.id[ii]);
        }
    }

    let mut candidate = BTreeMap::<u64, ContainerId>::new();
    for coord in interior.iter() {
        let source = cell_container(coord, &interior, layout);
        let Some(ids) = by_source.get(&source) else {
            continue;
        };
        let source_mass = *stage_input.den.view().at(coord) * geometry.volume(coord);
        let mut transfers = Vec::with_capacity(2 * D);
        for dd in 0..D {
            let mut high = coord;
            high[dd] += 1;
            let low_flux = *sim.fields.flux[dd].den.view().at(coord);
            let high_flux = *sim.fields.flux[dd].den.view().at(high);
            if low_flux < 0.0 {
                transfers.push(MassTransfer {
                    destination: face_destination(
                        coord,
                        dd,
                        false,
                        &interior,
                        &sim.boundaries,
                        layout,
                    ),
                    mass: -low_flux * geometry.face_area(coord, dd) * sim.dt,
                });
            }
            if high_flux > 0.0 {
                transfers.push(MassTransfer {
                    destination: face_destination(
                        coord,
                        dd,
                        true,
                        &interior,
                        &sim.boundaries,
                        layout,
                    ),
                    mass: high_flux * geometry.face_area(high, dd) * sim.dt,
                });
            }
        }
        let kernel = TransportKernel::new(source, source_mass, transfers)?;
        candidate.extend(sample_systematic(&kernel, ids, key));
    }

    for (ii, id) in tracers.id.iter().enumerate() {
        if let Some(&destination) = candidate.get(id) {
            tracers.owner[ii] = destination;
            if is_exterior(destination) {
                tracers.flags[ii].escaped = true;
            }
        }
    }

    if a0 != 0.0 || ac != 1.0 {
        let selections = sample_convex_blend(&tracers.id, ac, key)?;
        for (ii, (_, choose_candidate)) in selections.into_iter().enumerate() {
            if !choose_candidate {
                tracers.owner[ii] = tracers.step_owner[ii];
                tracers.flags[ii] = tracers.step_flags[ii];
            }
        }
    }

    for (ii, &owner) in tracers.owner.iter().enumerate() {
        if !is_exterior(owner) {
            if let Some(coord) = container_cell(owner, &interior, layout) {
                let center = geometry.centroid(coord);
                for dd in 0..D {
                    tracers.x[ii][dd] = center[dd];
                }
            }
        }
    }
    sim.tracers = Some(tracers);
    Ok(())
}

const EXTERIOR_BIT: u64 = 1 << 63;

fn is_exterior(container: crate::mass_transport::ContainerId) -> bool {
    container.0 & EXTERIOR_BIT != 0
}

fn exterior_container(axis: usize, high: bool) -> crate::mass_transport::ContainerId {
    crate::mass_transport::ContainerId(EXTERIOR_BIT | ((axis as u64) << 1) | high as u64)
}

fn cell_container<const D: usize>(
    coord: [isize; D],
    domain: &symbi_algebra::Domain<D>,
    layout: TransportLayout<D>,
) -> crate::mass_transport::ContainerId {
    let mut linear = 0usize;
    let mut stride = 1usize;
    for dd in 0..D {
        let local = (coord[dd] - domain.spaces[dd].lo) as usize;
        linear += (layout.tile_offset[dd] + local) * stride;
        stride *= layout.global_cells[dd];
    }
    crate::mass_transport::ContainerId(linear as u64)
}

fn container_cell<const D: usize>(
    container: crate::mass_transport::ContainerId,
    domain: &symbi_algebra::Domain<D>,
    layout: TransportLayout<D>,
) -> Option<[isize; D]> {
    let mut linear = container.0 as usize;
    let global: [usize; D] = std::array::from_fn(|dd| {
        let index = linear % layout.global_cells[dd];
        linear /= layout.global_cells[dd];
        index
    });
    let local: [isize; D] = std::array::from_fn(|dd| {
        global[dd] as isize - layout.tile_offset[dd] as isize + domain.spaces[dd].lo
    });
    domain.contains(local).then_some(local)
}

fn face_destination<const D: usize>(
    coord: [isize; D],
    axis: usize,
    high: bool,
    domain: &symbi_algebra::Domain<D>,
    boundaries: &crate::state::Boundaries<D>,
    layout: TransportLayout<D>,
) -> crate::mass_transport::ContainerId {
    let local: [usize; D] = std::array::from_fn(|dd| (coord[dd] - domain.spaces[dd].lo) as usize);
    let mut global: [isize; D] =
        std::array::from_fn(|dd| (layout.tile_offset[dd] + local[dd]) as isize);
    global[axis] += if high { 1 } else { -1 };
    if global[axis] >= 0 && global[axis] < layout.global_cells[axis] as isize {
        let mut linear = 0usize;
        let mut stride = 1usize;
        for dd in 0..D {
            linear += global[dd] as usize * stride;
            stride *= layout.global_cells[dd];
        }
        return crate::mass_transport::ContainerId(linear as u64);
    }
    match boundaries.0[axis][high as usize] {
        crate::state::BoundaryType::Periodic => {
            global[axis] = if high {
                0
            } else {
                layout.global_cells[axis] as isize - 1
            };
            let mut linear = 0usize;
            let mut stride = 1usize;
            for dd in 0..D {
                linear += global[dd] as usize * stride;
                stride *= layout.global_cells[dd];
            }
            crate::mass_transport::ContainerId(linear as u64)
        }
        crate::state::BoundaryType::Reflect => cell_container(coord, domain, layout),
        _ => exterior_container(axis, high),
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
        .map(|_| TracerSet {
            weight: set.weight,
            ..Default::default()
        })
        .collect();
    for i in 0..set.x.len() {
        // the seed is inside the domain by construction, so `tile_owner` is always Some.
        let dest = tile_owner(&set.x[i], glo, extent, counts).unwrap_or(0);
        per_tile[dest].x.push(set.x[i]);
        per_tile[dest].id.push(set.id[i]);
        per_tile[dest].flags.push(set.flags[i]);
        per_tile[dest].owner.push(set.owner[i]);
        per_tile[dest].step_owner.push(set.step_owner[i]);
        per_tile[dest].step_flags.push(set.step_flags[i]);
        per_tile[dest].run_seed = set.run_seed;
    }
    per_tile
}
