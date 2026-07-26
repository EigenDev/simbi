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
    /// immutable initial-material provenance label.
    pub cohort: Vec<u16>,
    pub flags: Vec<TracerFlags>,
    pub weight: f64,
    /// material container that owns each tracer. cell containers use the
    /// interior's axis-0-fastest linear index.
    pub owner: Vec<crate::mass_transport::ContainerId>,
    /// step-entry ancestry retained across all ssp stages.
    pub step_owner: Vec<crate::mass_transport::ContainerId>,
    pub step_flags: Vec<TracerFlags>,
    pub run_seed: u64,
    /// first unused stable identity for tracers spawned by material injection.
    pub next_id: u64,
    /// injected mass below one tracer quantum, carried exactly between steps.
    pub injection_remainder: f64,
}

/// continuous-position passive tracers in execution-space-accessible soa storage.
///
/// position and its ssp step snapshot are authoritative. ownership is derived
/// after advancement for migration and material-reservoir accounting.
pub struct ContinuousTracerSet<const D: usize, Mem: symbi_xpu::MemorySpace> {
    pub order: crate::mass_transport::ItoOrder,
    pub x: [symbi_xpu::MemoryBlock<Mem>; D],
    pub step_x: [symbi_xpu::MemoryBlock<Mem>; D],
    pub id: symbi_xpu::MemoryBlock<Mem>,
    pub cohort: symbi_xpu::MemoryBlock<Mem>,
    pub owner: symbi_xpu::MemoryBlock<Mem>,
    pub escaped: symbi_xpu::MemoryBlock<Mem>,
    pub crossed_sink: symbi_xpu::MemoryBlock<Mem>,
    pub crossing_time: symbi_xpu::MemoryBlock<Mem>,
    pub random_counter: symbi_xpu::MemoryBlock<Mem>,
    pub len: usize,
    pub capacity: usize,
    pub weight: f64,
    pub run_seed: u64,
    pub next_id: u64,
    pub injection_remainder: f64,
}

/// cell-centered flux-derived coefficients interpolated by continuous tracers.
pub struct ItoCoefficientFields<const D: usize, Mem: symbi_xpu::MemorySpace> {
    pub drift: [symbi_grid::Field<f64, D, Mem>; D],
    pub variance: [symbi_grid::Field<f64, D, Mem>; D],
    pub third: [symbi_grid::Field<f64, D, Mem>; D],
}

impl<const D: usize, Mem: symbi_xpu::MemorySpace> ItoCoefficientFields<D, Mem> {
    pub fn zeros(domain: &symbi_algebra::Domain<D>) -> Result<Self, String> {
        Ok(Self {
            drift: crate::state::array_field_zeros(domain).map_err(|err| err.to_string())?,
            variance: crate::state::array_field_zeros(domain)
                .map_err(|err| err.to_string())?,
            third: crate::state::array_field_zeros(domain).map_err(|err| err.to_string())?,
        })
    }

    /// interpolate every directional moment rate to a physical particle position.
    pub fn interpolate(
        &self,
        geometry: &crate::state::PartitionGeometry<D>,
        position: [f64; D],
    ) -> Result<[crate::mass_transport::JumpMomentRates; D], String> {
        let stencil = cic_stencil(geometry, &self.drift[0].domain, position)?;
        Ok(std::array::from_fn(|dd| {
            crate::mass_transport::JumpMomentRates {
                drift: stencil.interpolate(&self.drift[dd]),
                variance: stencil.interpolate(&self.variance[dd]),
                third: stencil.interpolate(&self.third[dd]),
            }
        }))
    }
}

/// tensor-product linear interpolation from cell centers.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CicStencil<const D: usize> {
    lower: [isize; D],
    upper_weight: [f64; D],
}

impl<const D: usize> CicStencil<D> {
    pub fn interpolate<Mem: symbi_xpu::MemorySpace>(
        &self,
        field: &symbi_grid::Field<f64, D, Mem>,
    ) -> f64 {
        let mut sum = 0.0;
        for corner in 0..(1usize << D) {
            let mut coord = self.lower;
            let mut weight = 1.0;
            for dd in 0..D {
                let upper = (corner & (1 << dd)) != 0;
                if upper {
                    coord[dd] += 1;
                    weight *= self.upper_weight[dd];
                } else {
                    weight *= 1.0 - self.upper_weight[dd];
                }
            }
            sum += weight * *field.view().at(coord);
        }
        sum
    }
}

/// construct the enclosing cell-center stencil on uniform or mapped axes.
pub fn cic_stencil<const D: usize>(
    geometry: &crate::state::PartitionGeometry<D>,
    domain: &symbi_algebra::Domain<D>,
    position: [f64; D],
) -> Result<CicStencil<D>, String> {
    let mut lower = [0isize; D];
    let mut upper_weight = [0.0; D];
    for dd in 0..D {
        let containing = match geometry.maps {
            Some(maps) => maps[dd].index_at(position[dd]),
            None => ((position[dd] - geometry.x_lo[dd]) / geometry.dx[dd]).floor() as isize,
        };
        let center = |ii| match geometry.maps {
            Some(maps) => maps[dd].center(ii),
            None => geometry.x_lo[dd] + (ii as f64 + 0.5) * geometry.dx[dd],
        };
        let base = if position[dd] < center(containing) {
            containing - 1
        } else {
            containing
        };
        let space = &domain.spaces[dd];
        if base < space.lo || base + 1 >= space.hi {
            return Err(format!(
                "particle coordinate {} on axis {} lies outside the interpolation domain",
                position[dd], dd
            ));
        }
        let lo = center(base);
        let hi = center(base + 1);
        if !lo.is_finite() || !hi.is_finite() || hi <= lo {
            return Err(format!(
                "invalid cell-center interval [{lo}, {hi}] on axis {dd}"
            ));
        }
        lower[dd] = base;
        upper_weight[dd] = ((position[dd] - lo) / (hi - lo)).clamp(0.0, 1.0);
    }
    Ok(CicStencil {
        lower,
        upper_weight,
    })
}

/// advance continuous tracers on host-accessible storage.
///
/// this is the correctness oracle for accelerator kernels: one counter value
/// identifies a complete dimensional update, while the axis selects independent
/// samples without making results depend on traversal order.
pub fn advance_continuous_tracers_host<const D: usize, Mem: symbi_xpu::MemorySpace>(
    tracers: &mut ContinuousTracerSet<D, Mem>,
    coefficients: &ItoCoefficientFields<D, Mem>,
    geometry: &crate::state::PartitionGeometry<D>,
    dt: f64,
) -> Result<(), String> {
    if !Mem::IS_HOST_ACCESSIBLE {
        return Err("host tracer advancement requires host-accessible memory".to_string());
    }
    if !dt.is_finite() || dt <= 0.0 {
        return Err("ito tracer timestep must be positive and finite".to_string());
    }
    unsafe {
        let ids = std::slice::from_raw_parts(tracers.id.as_ptr::<u64>(), tracers.len);
        let escaped = std::slice::from_raw_parts(tracers.escaped.as_ptr::<u8>(), tracers.len);
        let counters = std::slice::from_raw_parts_mut(
            tracers.random_counter.as_mut_ptr::<u64>(),
            tracers.len,
        );
        for ii in 0..tracers.len {
            if escaped[ii] != 0 {
                continue;
            }
            let position =
                std::array::from_fn(|dd| *tracers.x[dd].as_ptr::<f64>().add(ii));
            let rates = coefficients.interpolate(geometry, position)?;
            for dd in 0..D {
                let unit = crate::mass_transport::ito_unit_sample(
                    tracers.run_seed,
                    ids[ii],
                    counters[ii],
                    dd,
                );
                let displacement = match tracers.order {
                    crate::mass_transport::ItoOrder::Two => {
                        crate::mass_transport::ito2_displacement(rates[dd], dt, unit)?
                    }
                    crate::mass_transport::ItoOrder::Three => {
                        crate::mass_transport::ito3_displacement(rates[dd], dt, unit)?
                    }
                };
                *tracers.x[dd].as_mut_ptr::<f64>().add(ii) = position[dd] + displacement;
            }
            counters[ii] = counters[ii].wrapping_add(1);
        }
    }
    Ok(())
}

fn particle_blocks<const D: usize, Mem: symbi_xpu::MemorySpace, T>(
    capacity: usize,
) -> Result<[symbi_xpu::MemoryBlock<Mem>; D], String> {
    let mut blocks = Vec::with_capacity(D);
    for _dd in 0..D {
        blocks.push(
            symbi_xpu::MemoryBlock::<Mem>::for_elements::<T>(capacity)
                .map_err(|err| err.to_string())?,
        );
    }
    match blocks.try_into() {
        Ok(array) => Ok(array),
        Err(_) => unreachable!("particle block count equals the dimension"),
    }
}

impl<const D: usize, Mem: symbi_xpu::MemorySpace> ContinuousTracerSet<D, Mem> {
    pub fn allocate(
        capacity: usize,
        order: crate::mass_transport::ItoOrder,
    ) -> Result<Self, String> {
        Ok(Self {
            order,
            x: particle_blocks::<D, Mem, f64>(capacity)?,
            step_x: particle_blocks::<D, Mem, f64>(capacity)?,
            id: symbi_xpu::MemoryBlock::<Mem>::for_elements::<u64>(capacity)
                .map_err(|err| err.to_string())?,
            cohort: symbi_xpu::MemoryBlock::<Mem>::for_elements::<u16>(capacity)
                .map_err(|err| err.to_string())?,
            owner:
                symbi_xpu::MemoryBlock::<Mem>::for_elements::<crate::mass_transport::ContainerId>(
                    capacity,
                )
                .map_err(|err| err.to_string())?,
            escaped: symbi_xpu::MemoryBlock::<Mem>::for_elements::<u8>(capacity)
                .map_err(|err| err.to_string())?,
            crossed_sink: symbi_xpu::MemoryBlock::<Mem>::for_elements::<u8>(capacity)
                .map_err(|err| err.to_string())?,
            crossing_time: symbi_xpu::MemoryBlock::<Mem>::for_elements::<f64>(capacity)
                .map_err(|err| err.to_string())?,
            random_counter: symbi_xpu::MemoryBlock::<Mem>::for_elements::<u64>(capacity)
                .map_err(|err| err.to_string())?,
            len: 0,
            capacity,
            weight: 0.0,
            run_seed: 0,
            next_id: 0,
            injection_remainder: 0.0,
        })
    }

    pub fn from_discrete(
        seed: &TracerSet<D>,
        order: crate::mass_transport::ItoOrder,
    ) -> Result<Self, String> {
        if !Mem::IS_HOST_ACCESSIBLE {
            return Err("continuous tracer seeding requires host-accessible memory".to_string());
        }
        let mut set = Self::allocate(seed.len(), order)?;
        set.len = seed.len();
        set.weight = seed.weight;
        set.run_seed = seed.run_seed;
        set.next_id = seed.next_id;
        set.injection_remainder = seed.injection_remainder;
        unsafe {
            for dd in 0..D {
                let position =
                    std::slice::from_raw_parts_mut(set.x[dd].as_mut_ptr::<f64>(), set.len);
                let step_position =
                    std::slice::from_raw_parts_mut(set.step_x[dd].as_mut_ptr::<f64>(), set.len);
                for ii in 0..set.len {
                    position[ii] = seed.x[ii][dd];
                    step_position[ii] = seed.x[ii][dd];
                }
            }
            std::ptr::copy_nonoverlapping(seed.id.as_ptr(), set.id.as_mut_ptr(), set.len);
            std::ptr::copy_nonoverlapping(seed.cohort.as_ptr(), set.cohort.as_mut_ptr(), set.len);
            std::ptr::copy_nonoverlapping(seed.owner.as_ptr(), set.owner.as_mut_ptr(), set.len);
            let escaped = std::slice::from_raw_parts_mut(set.escaped.as_mut_ptr::<u8>(), set.len);
            let crossed_sink =
                std::slice::from_raw_parts_mut(set.crossed_sink.as_mut_ptr::<u8>(), set.len);
            let crossing_time =
                std::slice::from_raw_parts_mut(set.crossing_time.as_mut_ptr::<f64>(), set.len);
            let random_counter =
                std::slice::from_raw_parts_mut(set.random_counter.as_mut_ptr::<u64>(), set.len);
            for ii in 0..set.len {
                escaped[ii] = seed.flags[ii].escaped as u8;
                crossed_sink[ii] = seed.flags[ii].crossed_sink as u8;
                crossing_time[ii] = seed.flags[ii].crossing_time;
                random_counter[ii] = 0;
            }
        }
        Ok(set)
    }
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
        let owners: Vec<_> = (0..cells.len())
            .map(|ii| crate::mass_transport::ContainerId(ii as u64))
            .collect();
        Self::seed_stratified_owned(cells, &owners, counts, weight)
    }

    /// seed stratified tracers into explicitly addressed material cells.
    pub fn seed_stratified_owned(
        cells: &[([f64; D], [f64; D])],
        owners: &[crate::mass_transport::ContainerId],
        counts: &[usize],
        weight: f64,
    ) -> Self {
        assert_eq!(cells.len(), counts.len(), "cells/counts length mismatch");
        assert_eq!(cells.len(), owners.len(), "cells/owners length mismatch");
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
                set.cohort.push(0);
                set.flags.push(TracerFlags::default());
                set.owner.push(owners[ci]);
                next_id += 1;
            }
        }
        set.next_id = next_id;
        set
    }

    pub fn len(&self) -> usize {
        self.x.len()
    }

    pub fn is_empty(&self) -> bool {
        self.x.is_empty()
    }

    /// assign initial-material labels from the owning cell's linear index.
    pub fn assign_cell_cohorts(&mut self, cell_cohorts: &[u16]) -> Result<(), String> {
        if self.owner.len() != self.cohort.len() {
            return Err("tracer owner/cohort length mismatch".to_string());
        }
        for (cohort, owner) in self.cohort.iter_mut().zip(&self.owner) {
            let linear = usize::try_from(owner.0)
                .map_err(|_| format!("initial tracer owner {} is not a cell", owner.0))?;
            *cohort = *cell_cohorts
                .get(linear)
                .ok_or_else(|| format!("missing cohort for initial cell {linear}"))?;
        }
        Ok(())
    }

    /// the accreted tracer mass: crossing count times the per-tracer weight —
    /// the quantity the G-flux gate compares against the sink's Mdot ledger.
    pub fn crossed_mass(&self) -> f64 {
        self.flags.iter().filter(|f| f.crossed_sink).count() as f64 * self.weight
    }
}

/// spawn fixed-weight tracers for newly injected material. total represented
/// mass plus the carried remainder equals injected mass plus the old remainder.
pub fn spawn_injected_tracers<const D: usize>(
    tracers: &mut TracerSet<D>,
    injections: impl IntoIterator<Item = crate::mass_transport::MassTransfer>,
    positions: impl Fn(crate::mass_transport::ContainerId) -> [f64; D],
    key: crate::mass_transport::SamplingKey,
) -> Result<usize, String> {
    use crate::mass_transport::{ContainerId, MassTransfer, TransportKernel, sample_systematic};
    use std::collections::BTreeMap;

    if !tracers.weight.is_finite() || tracers.weight <= 0.0 {
        return Err(format!(
            "injected tracer spawning requires positive finite weight, got {:?}",
            tracers.weight
        ));
    }
    let mut combined = BTreeMap::<ContainerId, f64>::new();
    for injection in injections {
        if !injection.mass.is_finite() || injection.mass < 0.0 {
            return Err(format!("invalid injected mass {:?}", injection.mass));
        }
        *combined.entry(injection.destination).or_insert(0.0) += injection.mass;
    }
    let injected_mass: f64 = combined.values().sum();
    let available = tracers.injection_remainder + injected_mass;
    let count = (available / tracers.weight).floor() as usize;
    tracers.injection_remainder = available - count as f64 * tracers.weight;
    if count == 0 {
        return Ok(0);
    }

    const INJECTION_SOURCE: ContainerId = ContainerId((1 << 62) | 1);
    let kernel = TransportKernel::new(
        INJECTION_SOURCE,
        injected_mass,
        combined
            .into_iter()
            .map(|(destination, mass)| MassTransfer { destination, mass }),
    )?;
    let ids: Vec<u64> = (0..count)
        .map(|_| {
            let id = tracers.next_id;
            tracers.next_id = tracers.next_id.checked_add(1).expect("tracer id overflow");
            id
        })
        .collect();
    for (id, owner) in sample_systematic(&kernel, &ids, key) {
        tracers.x.push(positions(owner));
        tracers.id.push(id);
        tracers.cohort.push(u16::MAX);
        tracers.flags.push(TracerFlags::default());
        tracers.owner.push(owner);
        tracers.step_owner.push(owner);
        tracers.step_flags.push(TracerFlags::default());
    }
    Ok(count)
}

/// accumulate one stage's injected masses through the same shu-osher
/// recurrence as the conserved state.
pub fn fold_injection_ledger(
    ledger: &mut std::collections::BTreeMap<crate::mass_transport::ContainerId, f64>,
    stage: impl IntoIterator<Item = crate::mass_transport::MassTransfer>,
    ac: f64,
) {
    for mass in ledger.values_mut() {
        *mass *= ac;
    }
    for transfer in stage {
        *ledger.entry(transfer.destination).or_insert(0.0) += ac * transfer.mass;
    }
}

/// accepted external mass entering through physical domain faces during one
/// forward-euler candidate.
pub fn boundary_injection_transfers<R, const D: usize, const DOF: usize, M, E, S, Mem>(
    sim: &crate::state::SimStateGeneric<R, D, DOF, M, E, S, Mem>,
) -> Vec<crate::mass_transport::MassTransfer>
where
    R: symbi_hydro::regime::Regime<f64, D>,
    M: symbi_geometry::Metric<f64, D> + Copy,
    E: symbi_hydro::eos::Eos<f64>,
    S: symbi_xpu::ExecutionSpace,
    Mem: MemorySpace,
{
    let interior = sim.geom.interior.clone();
    let geometry = sim.geom.block_geometry(sim.physics.metric);
    let layout = TransportLayout::single(&interior);
    boundary_injection_transfers_store(&sim.store, &geometry, layout)
}

pub fn boundary_injection_transfers_store<const D: usize, const DOF: usize, M, Mem>(
    sim: &FieldStore<D, DOF, Mem, f64>,
    geometry: &symbi_geometry::BlockGeometry<M, f64, D>,
    layout: TransportLayout<D>,
) -> Vec<crate::mass_transport::MassTransfer>
where
    M: symbi_geometry::Metric<f64, D> + Copy,
    Mem: MemorySpace,
{
    let interior = sim.geom.interior.clone();
    let mut transfers = Vec::new();
    for dd in 0..D {
        for high in [false, true] {
            if matches!(
                sim.boundaries.0[dd][high as usize],
                crate::state::BoundaryType::Periodic
                    | crate::state::BoundaryType::Reflect
                    | crate::state::BoundaryType::CoarseFine
            ) {
                continue;
            }
            let face_index = if high {
                interior.spaces[dd].hi
            } else {
                interior.spaces[dd].lo
            };
            let slab = interior.slab(
                dd,
                (face_index - high as isize, face_index + !high as isize),
            );
            for cell in slab.iter() {
                let mut face = cell;
                if high {
                    face[dd] += 1;
                }
                let flux = *sim.fields.flux[dd].den.view().at(face);
                let inward_flux = if high { -flux } else { flux };
                if inward_flux > 0.0 {
                    transfers.push(crate::mass_transport::MassTransfer {
                        destination: cell_container(cell, &interior, layout),
                        mass: inward_flux * geometry.face_area(face, dd) * sim.dt,
                    });
                }
            }
        }
    }
    transfers
}

/// positive density supplied by non-flux stage operators, recovered as the
/// conservative residual of the accepted stage update.
pub fn source_injection_transfers<R, const D: usize, const DOF: usize, M, E, S, Mem>(
    sim: &crate::state::SimStateGeneric<R, D, DOF, M, E, S, Mem>,
    a0: f64,
    ac: f64,
) -> Vec<crate::mass_transport::MassTransfer>
where
    R: symbi_hydro::regime::Regime<f64, D>,
    M: symbi_geometry::Metric<f64, D> + Copy,
    E: symbi_hydro::eos::Eos<f64>,
    S: symbi_xpu::ExecutionSpace,
    Mem: MemorySpace,
{
    let interior = sim.geom.interior.clone();
    let geometry = sim.geom.block_geometry(sim.physics.metric);
    let layout = TransportLayout::single(&interior);
    source_injection_transfers_store(&sim.store, &geometry, layout, a0, ac)
}

pub fn source_injection_transfers_store<const D: usize, const DOF: usize, M, Mem>(
    sim: &FieldStore<D, DOF, Mem, f64>,
    geometry: &symbi_geometry::BlockGeometry<M, f64, D>,
    layout: TransportLayout<D>,
    a0: f64,
    ac: f64,
) -> Vec<crate::mass_transport::MassTransfer>
where
    M: symbi_geometry::Metric<f64, D> + Copy,
    Mem: MemorySpace,
{
    let interior = sim.geom.interior.clone();
    let stage_input = sim.stage_input();
    let mut transfers = Vec::new();
    for coord in interior.iter() {
        let volume = geometry.volume(coord);
        let mut divergence = 0.0;
        for dd in 0..D {
            let mut high = coord;
            high[dd] += 1;
            divergence += *sim.fields.flux[dd].den.view().at(high) * geometry.face_area(high, dd)
                - *sim.fields.flux[dd].den.view().at(coord) * geometry.face_area(coord, dd);
        }
        let expected_mass = a0 * *sim.workspace.u_n.den.view().at(coord) * volume
            + ac * (*stage_input.den.view().at(coord) * volume - sim.dt * divergence);
        let actual_mass = *sim.fields.cons.den.view().at(coord) * volume;
        let residual = actual_mass - expected_mass;
        let tolerance = 128.0 * f64::EPSILON * actual_mass.abs().max(expected_mass.abs()).max(1.0);
        if residual > tolerance && ac > 0.0 {
            transfers.push(crate::mass_transport::MassTransfer {
                destination: cell_container(coord, &interior, layout),
                mass: residual / ac,
            });
        }
    }
    transfers
}

pub fn spawn_boundary_injection<R, const D: usize, const DOF: usize, M, E, S, Mem>(
    sim: &mut crate::state::SimStateGeneric<R, D, DOF, M, E, S, Mem>,
    ledger: std::collections::BTreeMap<crate::mass_transport::ContainerId, f64>,
) -> Result<usize, String>
where
    R: symbi_hydro::regime::Regime<f64, D>,
    M: symbi_geometry::Metric<f64, D> + Copy,
    E: symbi_hydro::eos::Eos<f64>,
    S: symbi_xpu::ExecutionSpace,
    Mem: MemorySpace,
{
    let interior = sim.geom.interior.clone();
    let geometry = sim.geom.block_geometry(sim.physics.metric);
    let layout = TransportLayout::single(&interior);
    spawn_boundary_injection_store(&mut sim.store, &geometry, layout, ledger)
}

pub fn spawn_boundary_injection_store<const D: usize, const DOF: usize, M, Mem>(
    sim: &mut FieldStore<D, DOF, Mem, f64>,
    geometry: &symbi_geometry::BlockGeometry<M, f64, D>,
    layout: TransportLayout<D>,
    ledger: std::collections::BTreeMap<crate::mass_transport::ContainerId, f64>,
) -> Result<usize, String>
where
    M: symbi_geometry::Metric<f64, D> + Copy,
    Mem: MemorySpace,
{
    let interior = sim.geom.interior.clone();
    let Some(mut tracers) = sim.tracers.take() else {
        return Ok(0);
    };
    let key = crate::mass_transport::SamplingKey {
        run_seed: tracers.run_seed,
        epoch: sim.iteration | (1 << 62),
    };
    let count = spawn_injected_tracers(
        &mut tracers,
        ledger
            .into_iter()
            .map(|(destination, mass)| crate::mass_transport::MassTransfer { destination, mass }),
        |owner| {
            let coord = container_cell(owner, &interior, layout)
                .expect("boundary injection destination belongs to this grid");
            let center = geometry.centroid(coord);
            std::array::from_fn(|dd| center[dd])
        },
        key,
    )?;
    sim.tracers = Some(tracers);
    Ok(count)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn continuous_storage_preserves_seed_state_and_allocates_ssp_snapshot() {
        use symbi_xpu::HostMemory;

        let seed = TracerSet::<2>::seed_stratified(&[([0.0, 1.0], [2.0, 4.0])], &[3], 0.25);
        let continuous = ContinuousTracerSet::<2, HostMemory>::from_discrete(
            &seed,
            crate::mass_transport::ItoOrder::Three,
        )
        .unwrap();

        assert_eq!(continuous.order, crate::mass_transport::ItoOrder::Three);
        assert_eq!(continuous.len, 3);
        assert_eq!(continuous.capacity, 3);
        assert_eq!(continuous.weight, 0.25);
        unsafe {
            let x0 = std::slice::from_raw_parts(continuous.x[0].as_ptr::<f64>(), 3);
            let x1 = std::slice::from_raw_parts(continuous.x[1].as_ptr::<f64>(), 3);
            let step_x0 = std::slice::from_raw_parts(continuous.step_x[0].as_ptr::<f64>(), 3);
            let ids = std::slice::from_raw_parts(continuous.id.as_ptr::<u64>(), 3);
            let counters = std::slice::from_raw_parts(continuous.random_counter.as_ptr::<u64>(), 3);
            for ii in 0..3 {
                assert_eq!(x0[ii], seed.x[ii][0]);
                assert_eq!(x1[ii], seed.x[ii][1]);
                assert_eq!(step_x0[ii], seed.x[ii][0]);
                assert_eq!(ids[ii], seed.id[ii]);
                assert_eq!(counters[ii], 0);
            }
        }
    }

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
    fn injected_mass_spawns_fixed_weight_tracers_and_carries_the_remainder() {
        use crate::mass_transport::{ContainerId, MassTransfer, SamplingKey};

        let mut tracers = TracerSet::<1> {
            weight: 0.25,
            next_id: 10,
            ..Default::default()
        };
        let spawned = spawn_injected_tracers(
            &mut tracers,
            [
                MassTransfer {
                    destination: ContainerId(2),
                    mass: 0.4,
                },
                MassTransfer {
                    destination: ContainerId(3),
                    mass: 0.2,
                },
            ],
            |owner| [owner.0 as f64],
            SamplingKey {
                run_seed: 7,
                epoch: 11,
            },
        )
        .unwrap();
        assert_eq!(spawned, 2);
        assert_eq!(tracers.id, [10, 11]);
        assert_eq!(tracers.cohort, [u16::MAX; 2]);
        assert_eq!(tracers.next_id, 12);
        assert!((tracers.injection_remainder - 0.1).abs() < 1.0e-15);
        assert_eq!(tracers.owner.len(), 2);

        let spawned = spawn_injected_tracers(
            &mut tracers,
            [MassTransfer {
                destination: ContainerId(3),
                mass: 0.15,
            }],
            |owner| [owner.0 as f64],
            SamplingKey {
                run_seed: 7,
                epoch: 12,
            },
        )
        .unwrap();
        assert_eq!(spawned, 1);
        assert!(tracers.injection_remainder.abs() < 1.0e-15);
        assert_eq!(tracers.id, [10, 11, 12]);
        assert_eq!(tracers.cohort, [u16::MAX; 3]);
    }

    #[test]
    fn rk2_injection_ledger_weights_predictor_and_corrector_equally() {
        use crate::mass_transport::{ContainerId, MassTransfer};
        use std::collections::BTreeMap;

        let destination = ContainerId(4);
        let mut ledger = BTreeMap::new();
        fold_injection_ledger(
            &mut ledger,
            [MassTransfer {
                destination,
                mass: 2.0,
            }],
            1.0,
        );
        fold_injection_ledger(
            &mut ledger,
            [MassTransfer {
                destination,
                mass: 6.0,
            }],
            0.5,
        );
        assert_eq!(ledger[&destination], 4.0);
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
        // the accepted state includes the finite-volume divergence of the imposed flux.
        for (ii, coord) in sim.geom.interior.iter().enumerate() {
            let density = if ii == 0 { 0.75 } else { 1.25 };
            sim.fields.cons.den.view_mut().set(coord, density);
        }
        let cells = [([0.0], [1.0]), ([1.0], [1.0])];
        sim.tracers = Some(TracerSet::seed_stratified(&cells, &[100, 100], 0.01));
        snapshot_transport_state(&mut sim);
        let geometry = sim.geom.block_geometry(Cartesian);
        materialize_ito_coefficients_store(&mut sim, &geometry).unwrap();
        let coefficients = sim.ito_coefficients.as_ref().unwrap();
        let left = [sim.geom.interior.spaces[0].lo];
        let right = [left[0] + 1];
        assert_eq!(*coefficients.drift[0].view().at(left), 0.5);
        assert_eq!(*coefficients.variance[0].view().at(left), 0.375);
        assert_eq!(*coefficients.third[0].view().at(left), 0.1875);
        assert_eq!(*coefficients.drift[0].view().at(right), 0.0);
        assert_eq!(*coefficients.variance[0].view().at(right), 0.0);
        assert_eq!(*coefficients.third[0].view().at(right), 0.0);

        advance_stage_mass_transport(&mut sim, 0.0, 1.0, 0).unwrap();

        let tracers = sim.tracers.as_ref().unwrap();
        let in_left = tracers.owner.iter().filter(|owner| owner.0 == 0).count();
        let in_right = tracers.owner.iter().filter(|owner| owner.0 == 1).count();
        assert_eq!((in_left, in_right), (75, 125));
        assert_eq!(tracers.flags.iter().filter(|flag| flag.escaped).count(), 0);
    }

    #[test]
    fn cic_reproduces_affine_fields_on_mapped_axes() {
        use crate::state::{Boundaries, BoundaryType, SimState, Timestepping};
        use symbi_geometry::{AxisMap, Cartesian};
        use symbi_hydro::eos::IdealGas;
        use symbi_hydro::newtonian::Newtonian;
        use symbi_xpu::{CpuSpace, HostMemory};

        let mut sim =
            SimState::<Newtonian, 2, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>::new(
                Newtonian,
                IdealGas { gamma: 1.4 },
                Cartesian,
                [4, 4],
                [1.0, 2.0],
                [0.25, 0.5],
                2,
                Boundaries::uniform(BoundaryType::Outflow),
                0.4,
                Timestepping::Euler,
                0,
            )
            .unwrap();
        sim.geom.set_maps([
            AxisMap::Geometric {
                start: 1.0,
                width: 0.2,
                ratio: 1.2,
            },
            AxisMap::Log {
                start: 2.0,
                log_slope: 0.08,
            },
        ]);
        let fields = ItoCoefficientFields::<2, HostMemory>::zeros(&sim.geom.allocated).unwrap();
        for coord in sim.geom.allocated.iter() {
            let [x, y] = sim.geom.cell_coord(coord);
            fields.drift[0].view_mut().set(coord, 2.0 * x - 3.0 * y + 4.0);
            fields.variance[0].view_mut().set(coord, -x + 0.5 * y + 2.0);
            fields.third[0].view_mut().set(coord, 3.0 * x + y - 1.0);
            fields.drift[1].view_mut().set(coord, x + y);
            fields.variance[1].view_mut().set(coord, 2.0 * x + y);
            fields.third[1].view_mut().set(coord, x + 2.0 * y);
        }
        let lo = [
            sim.geom.interior.spaces[0].lo + 1,
            sim.geom.interior.spaces[1].lo + 1,
        ];
        let hi = [lo[0] + 1, lo[1] + 1];
        let lo_position = sim.geom.cell_coord(lo);
        let hi_position = sim.geom.cell_coord(hi);
        let position = [
            lo_position[0] + 0.37 * (hi_position[0] - lo_position[0]),
            lo_position[1] + 0.61 * (hi_position[1] - lo_position[1]),
        ];
        let rates = fields.interpolate(&sim.geom, position).unwrap();
        let close = |actual: f64, expected: f64| {
            assert!((actual - expected).abs() < 1.0e-12, "{actual} != {expected}");
        };
        close(rates[0].drift, 2.0 * position[0] - 3.0 * position[1] + 4.0);
        close(rates[0].variance, -position[0] + 0.5 * position[1] + 2.0);
        close(rates[0].third, 3.0 * position[0] + position[1] - 1.0);
        close(rates[1].drift, position[0] + position[1]);
        close(rates[1].variance, 2.0 * position[0] + position[1]);
        close(rates[1].third, position[0] + 2.0 * position[1]);
    }

    #[test]
    fn continuous_host_step_advects_live_particles_and_consumes_one_counter() {
        use crate::state::{Boundaries, BoundaryType, SimState, Timestepping};
        use symbi_geometry::Cartesian;
        use symbi_hydro::eos::IdealGas;
        use symbi_hydro::newtonian::Newtonian;
        use symbi_xpu::{CpuSpace, HostMemory};

        let sim =
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
        let seed = TracerSet::<1>::seed_stratified(&[([0.0], [1.0])], &[2], 0.5);
        let mut tracers = ContinuousTracerSet::<1, HostMemory>::from_discrete(
            &seed,
            crate::mass_transport::ItoOrder::Two,
        )
        .unwrap();
        tracers.run_seed = 17;
        unsafe {
            *tracers.escaped.as_mut_ptr::<u8>().add(1) = 1;
        }
        let coefficients =
            ItoCoefficientFields::<1, HostMemory>::zeros(&sim.geom.allocated).unwrap();
        for coord in sim.geom.allocated.iter() {
            coefficients.drift[0].view_mut().set(coord, 2.0);
            coefficients.variance[0].view_mut().set(coord, 0.0);
            coefficients.third[0].view_mut().set(coord, 0.0);
        }

        advance_continuous_tracers_host(&mut tracers, &coefficients, &sim.geom, 0.25).unwrap();

        unsafe {
            let position = std::slice::from_raw_parts(tracers.x[0].as_ptr::<f64>(), 2);
            let counters =
                std::slice::from_raw_parts(tracers.random_counter.as_ptr::<u64>(), 2);
            assert_eq!(position, [0.75, 0.75]);
            assert_eq!(counters, [1, 0]);
        }
    }

    #[test]
    fn covered_coarse_neighbor_is_deferred_to_the_interface_operator() {
        use crate::state::{Boundaries, BoundaryType, SimState, Timestepping};
        use symbi_algebra::{Domain, Space};
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
            sim.workspace.u_stage.den.view_mut().set(coord, 1.0);
        }
        let left = [sim.geom.interior.spaces[0].lo];
        let right = [left[0] + 1];
        sim.fields.cons.den.view_mut().set(left, 0.75);
        sim.fields.cons.den.view_mut().set(right, 1.25);
        for coord in sim.geom.allocated.iter() {
            sim.fields.flux[0].den.view_mut().set(coord, 0.0);
        }
        sim.fields.flux[0].den.view_mut().set(right, 0.5);
        sim.dt = 0.5;
        sim.tracers = Some(TracerSet::seed_stratified(
            &[([0.0], [1.0]), ([1.0], [1.0])],
            &[100, 0],
            0.01,
        ));
        snapshot_transport_state(&mut sim);
        let inactive = Domain::new([Space {
            name: "i",
            lo: right[0],
            hi: right[0] + 1,
        }]);
        let geometry = sim.geom.block_geometry(Cartesian);
        let layout = TransportLayout::single(&sim.geom.interior);

        advance_stage_mass_transport_store_masked(
            &mut sim.store,
            &geometry,
            layout,
            Some(&inactive),
            0.0,
            1.0,
            0,
        )
        .unwrap();

        assert_eq!(
            sim.tracers.as_ref().unwrap().owner,
            [crate::mass_transport::ContainerId(0); 100]
        );
    }

    #[test]
    fn accepted_boundary_inflow_spawns_tracers_in_the_receiving_cell() {
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
        sim.dt = 0.25;
        let low_face = [sim.geom.interior.spaces[0].lo];
        sim.fields.flux[0].den.view_mut().set(low_face, 1.0);
        sim.tracers = Some(TracerSet {
            weight: 0.1,
            next_id: 4,
            ..Default::default()
        });

        let mut ledger = std::collections::BTreeMap::new();
        fold_injection_ledger(&mut ledger, boundary_injection_transfers(&sim), 1.0);
        let spawned = spawn_boundary_injection(&mut sim, ledger).unwrap();
        let tracers = sim.tracers.as_ref().unwrap();
        assert_eq!(spawned, 2);
        assert_eq!(tracers.owner, [crate::mass_transport::ContainerId(0); 2]);
        assert!((tracers.injection_remainder - 0.05).abs() < 1.0e-15);
    }

    #[test]
    fn accepted_density_source_residual_spawns_tracers() {
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
                [1],
                [0.0],
                [1.0],
                2,
                Boundaries::uniform(BoundaryType::Outflow),
                0.4,
                Timestepping::Euler,
                0,
            )
            .unwrap();
        let cell = [sim.geom.interior.spaces[0].lo];
        sim.workspace.u_stage.den.view_mut().set(cell, 1.0);
        sim.fields.cons.den.view_mut().set(cell, 1.2);
        sim.dt = 0.5;
        sim.tracers = Some(TracerSet {
            weight: 0.05,
            next_id: 8,
            ..Default::default()
        });

        let mut ledger = std::collections::BTreeMap::new();
        fold_injection_ledger(&mut ledger, source_injection_transfers(&sim, 0.0, 1.0), 1.0);
        let spawned = spawn_boundary_injection(&mut sim, ledger).unwrap();
        assert_eq!(spawned, 3);
        let tracers = sim.tracers.as_ref().unwrap();
        assert!((tracers.injection_remainder - 0.05).abs() < 1.0e-14);
        assert_eq!(tracers.owner, [crate::mass_transport::ContainerId(0); 3]);
    }

    #[test]
    fn accepted_negative_density_source_moves_tracers_to_removal_reservoir() {
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
                [1],
                [0.0],
                [1.0],
                2,
                Boundaries::uniform(BoundaryType::Outflow),
                0.4,
                Timestepping::Euler,
                0,
            )
            .unwrap();
        let cell = [sim.geom.interior.spaces[0].lo];
        sim.workspace.u_stage.den.view_mut().set(cell, 1.0);
        sim.fields.cons.den.view_mut().set(cell, 0.8);
        sim.dt = 0.5;
        sim.tracers = Some(TracerSet::seed_stratified(&[([0.0], [1.0])], &[100], 0.01));
        snapshot_transport_state(&mut sim);

        advance_stage_mass_transport(&mut sim, 0.0, 1.0, 0).unwrap();
        let tracers = sim.tracers.as_ref().unwrap();
        let removed = tracers
            .owner
            .iter()
            .filter(|&&owner| owner == MATERIAL_REMOVAL_RESERVOIR)
            .count();
        assert_eq!(removed, 20);
        assert_eq!(
            tracers.owner.iter().filter(|owner| owner.0 == 0).count(),
            80
        );
    }

    #[test]
    fn cell_container_namespace_preserves_root_ids_and_separates_levels() {
        let root = cell_container_id(42, 0);
        let fine = cell_container_id(42, 1);

        assert_eq!(root.0, 42);
        assert_ne!(root, fine);
        assert_eq!(cell_container_address(root), Some((0, 42)));
        assert_eq!(cell_container_address(fine), Some((1, 42)));
        assert_eq!(cell_container_address(ACCRETION_RESERVOIR), None);
        assert_eq!(cell_container_address(MATERIAL_REMOVAL_RESERVOIR), None);
        assert_eq!(cell_container_address(exterior_container(0, false)), None);
    }

    #[test]
    fn body_accretion_reservoirs_preserve_identity_and_aggregate_queries() {
        let first = body_accretion_reservoir(0);
        let second = body_accretion_reservoir(1);

        assert_ne!(first, second);
        assert_eq!(accretion_reservoir_body(first), Some(0));
        assert_eq!(accretion_reservoir_body(second), Some(1));
        assert!(is_accretion_reservoir(first));
        assert!(is_accretion_reservoir(second));
        assert!(is_accretion_reservoir(ACCRETION_RESERVOIR));
        assert!(!is_accretion_reservoir(MATERIAL_REMOVAL_RESERVOIR));
    }

    #[test]
    fn accepted_body_receipts_partition_accreted_tracers_by_body() {
        use crate::state::{Boundaries, BoundaryType, SimState, Timestepping};
        use symbi_algebra::Tensor;
        use symbi_geometry::Cartesian;
        use symbi_hydro::eos::IdealGas;
        use symbi_hydro::newtonian::Newtonian;
        use symbi_xpu::{CpuSpace, HostMemory};

        let mut sim =
            SimState::<Newtonian, 1, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>::new(
                Newtonian,
                IdealGas { gamma: 1.4 },
                Cartesian,
                [1],
                [0.0],
                [1.0],
                2,
                Boundaries::uniform(BoundaryType::Outflow),
                0.4,
                Timestepping::Euler,
                0,
            )
            .unwrap();
        let bodies = (0..2).fold(symbi_ib::BodyCollection::new(), |bodies, ii| {
            bodies.add(symbi_ib::Body::black_hole(
                ii,
                Tensor::new([0.5]),
                Tensor::zeros(),
                1.0,
                0.1,
                0.05,
                0.5,
                0.0,
                0.1,
            ))
        });
        sim.attach_bodies(bodies);
        let cell = [sim.geom.interior.spaces[0].lo];
        sim.fields.cons.den.view_mut().set(cell, 0.5);
        sim.tracers = Some(TracerSet::seed_stratified(&[([0.0], [1.0])], &[100], 0.01));
        let immersed = sim.immersed.as_ref().unwrap();
        immersed.reset_accretion_receipts(1);
        immersed.record_accretion_receipt(0, [0.25]);
        immersed.record_accretion_receipt(1, [0.25]);

        advance_accretion_transport(&mut sim, &[1.0]).unwrap();

        let tracers = sim.tracers.as_ref().unwrap();
        let first = body_accretion_reservoir(0);
        let second = body_accretion_reservoir(1);
        assert_eq!(
            tracers
                .owner
                .iter()
                .filter(|&&owner| owner == first)
                .count(),
            25
        );
        assert_eq!(
            tracers
                .owner
                .iter()
                .filter(|&&owner| owner == second)
                .count(),
            25
        );
        assert_eq!(
            tracers.owner.iter().filter(|&&owner| owner.0 == 0).count(),
            50
        );
        assert_eq!(tracers.crossed_mass(), 0.5);
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
        let center = sim.geom.centroid(c);
        for a in 0..D {
            lo[a] = center[a] - 0.5 * sim.geom.dx[a];
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

pub fn seed_mass_weighted_with_cohorts<const D: usize, const DOF: usize, Mem: MemorySpace>(
    sim: &FieldStore<D, DOF, Mem, f64>,
    n: usize,
    cell_cohorts: &[u16],
) -> Result<TracerSet<D>, String> {
    if cell_cohorts.len() != sim.geom.interior.volume() {
        return Err(format!(
            "tracer_cohort yielded {} values for {} interior cells",
            cell_cohorts.len(),
            sim.geom.interior.volume()
        ));
    }
    let mut tracers = seed_mass_weighted(sim, n);
    tracers.assign_cell_cohorts(cell_cohorts)?;
    Ok(tracers)
}

/// seed a fixed-size tracer population over explicitly addressed cells whose
/// masses may come from different refinement levels.
pub fn seed_weighted_cells<const D: usize>(
    owners: &[crate::mass_transport::ContainerId],
    cells: &[([f64; D], [f64; D])],
    masses: &[f64],
    n: usize,
) -> TracerSet<D> {
    assert_eq!(owners.len(), cells.len(), "owners/cells length mismatch");
    assert_eq!(masses.len(), cells.len(), "masses/cells length mismatch");
    let counts = systematic_counts(masses, n);
    let weight = if n == 0 {
        0.0
    } else {
        masses.iter().sum::<f64>() / n as f64
    };
    let mut tracers = TracerSet::seed_stratified_owned(cells, owners, &counts, weight);
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
    pub level: u8,
}

impl<const D: usize> TransportLayout<D> {
    pub fn single(domain: &symbi_algebra::Domain<D>) -> Self {
        Self {
            global_cells: std::array::from_fn(|dd| domain.spaces[dd].size()),
            tile_offset: [0; D],
            level: 0,
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
    advance_stage_mass_transport_store_masked(sim, geometry, layout, None, a0, ac, stage)
}

/// materialize continuous-tracer moment rates from the final accepted stage flux.
pub fn materialize_ito_coefficients_store<const D: usize, const DOF: usize, M, Mem>(
    sim: &mut FieldStore<D, DOF, Mem, f64>,
    geometry: &symbi_geometry::BlockGeometry<M, f64, D>,
) -> Result<(), String>
where
    M: symbi_geometry::Metric<f64, D> + Copy,
    Mem: MemorySpace,
{
    let interior = sim.geom.interior.clone();
    let coefficients = ItoCoefficientFields::zeros(&sim.geom.allocated)?;
    let stage_input = sim.stage_input();
    let scale = if sim.motion.homologous {
        sim.motion.a
    } else {
        1.0
    };
    for coord in interior.iter() {
        let source_mass =
            *stage_input.den.view().at(coord) * geometry.labframe_volume(coord, scale);
        for dd in 0..D {
            let mut high = coord;
            high[dd] += 1;
            let low_flux = *sim.fields.flux[dd].den.view().at(coord);
            let high_flux = *sim.fields.flux[dd].den.view().at(high);
            let mass_to_minus = (-low_flux).max(0.0)
                * geometry.labframe_face_area(coord, dd, scale)
                * sim.dt;
            let mass_to_plus = high_flux.max(0.0)
                * geometry.labframe_face_area(high, dd, scale)
                * sim.dt;
            let width = geometry.cell_width(coord, dd) * scale;
            let rates = crate::mass_transport::accepted_face_moment_rates(
                source_mass,
                mass_to_minus,
                mass_to_plus,
                width,
                sim.dt,
            )?;
            coefficients.drift[dd]
                .view_mut()
                .set(coord, rates.drift);
            coefficients.variance[dd]
                .view_mut()
                .set(coord, rates.variance);
            coefficients.third[dd]
                .view_mut()
                .set(coord, rates.third);
        }
    }
    sim.ito_coefficients = Some(coefficients);
    Ok(())
}

/// advance one transport stage while excluding cells replaced by a finer
/// representation of the same material volume.
pub fn advance_stage_mass_transport_store_masked<const D: usize, const DOF: usize, M, Mem>(
    sim: &mut FieldStore<D, DOF, Mem, f64>,
    geometry: &symbi_geometry::BlockGeometry<M, f64, D>,
    layout: TransportLayout<D>,
    inactive: Option<&symbi_algebra::Domain<D>>,
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
    let interior = sim.geom.interior.clone();
    let volume_scale = if sim.motion.homologous {
        sim.motion.a.powi(D as i32)
    } else {
        1.0
    };
    let area_scale = if sim.motion.homologous {
        sim.motion.a.powi(D.saturating_sub(1) as i32)
    } else {
        1.0
    };
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
        if inactive.is_some_and(|domain| domain.contains(coord)) {
            continue;
        }
        let source = cell_container(coord, &interior, layout);
        let Some(ids) = by_source.get(&source) else {
            continue;
        };
        let source_mass = *stage_input.den.view().at(coord) * geometry.volume(coord) * volume_scale;
        let mut transfers = Vec::with_capacity(2 * D);
        for dd in 0..D {
            let mut high = coord;
            high[dd] += 1;
            let low_flux = *sim.fields.flux[dd].den.view().at(coord);
            let high_flux = *sim.fields.flux[dd].den.view().at(high);
            let mut low_cell = coord;
            low_cell[dd] -= 1;
            let low_inactive = inactive.is_some_and(|domain| domain.contains(low_cell));
            let high_inactive = inactive.is_some_and(|domain| domain.contains(high));
            if low_flux < 0.0 && !low_inactive {
                transfers.push(MassTransfer {
                    destination: face_destination(
                        coord,
                        dd,
                        false,
                        &interior,
                        &sim.boundaries,
                        layout,
                    ),
                    mass: -low_flux * geometry.face_area(coord, dd) * area_scale * sim.dt,
                });
            }
            if high_flux > 0.0 && !high_inactive {
                transfers.push(MassTransfer {
                    destination: face_destination(
                        coord,
                        dd,
                        true,
                        &interior,
                        &sim.boundaries,
                        layout,
                    ),
                    mass: high_flux * geometry.face_area(high, dd) * area_scale * sim.dt,
                });
            }
        }
        if ac > 0.0 && !sim.motion.homologous {
            let mut divergence = 0.0;
            for dd in 0..D {
                let mut high = coord;
                high[dd] += 1;
                divergence += *sim.fields.flux[dd].den.view().at(high)
                    * geometry.face_area(high, dd)
                    - *sim.fields.flux[dd].den.view().at(coord) * geometry.face_area(coord, dd);
            }
            let expected_mass =
                a0 * *sim.workspace.u_n.den.view().at(coord) * geometry.volume(coord)
                    + ac * (source_mass - sim.dt * divergence);
            let actual_mass = *sim.fields.cons.den.view().at(coord) * geometry.volume(coord);
            let residual = actual_mass - expected_mass;
            let tolerance =
                128.0 * f64::EPSILON * actual_mass.abs().max(expected_mass.abs()).max(1.0);
            if residual < -tolerance {
                transfers.push(MassTransfer {
                    destination: MATERIAL_REMOVAL_RESERVOIR,
                    mass: -residual / ac,
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
const ACCRETION_RESERVOIR_ID: u64 = 1 << 62;
const BODY_ACCRETION_BIT: u64 = 1 << 61;
const MATERIAL_REMOVAL_RESERVOIR_ID: u64 = (1 << 62) | 1;
const CELL_LEVEL_SHIFT: u32 = 56;
const CELL_LEVEL_MASK: u64 = 0x3f << CELL_LEVEL_SHIFT;
const CELL_LINEAR_MASK: u64 = (1 << CELL_LEVEL_SHIFT) - 1;

pub const ACCRETION_RESERVOIR: crate::mass_transport::ContainerId =
    crate::mass_transport::ContainerId(ACCRETION_RESERVOIR_ID);
pub const MATERIAL_REMOVAL_RESERVOIR: crate::mass_transport::ContainerId =
    crate::mass_transport::ContainerId(MATERIAL_REMOVAL_RESERVOIR_ID);

pub fn body_accretion_reservoir(body: usize) -> crate::mass_transport::ContainerId {
    crate::mass_transport::ContainerId(ACCRETION_RESERVOIR_ID | BODY_ACCRETION_BIT | body as u64)
}

pub fn accretion_reservoir_body(container: crate::mass_transport::ContainerId) -> Option<usize> {
    let prefix = ACCRETION_RESERVOIR_ID | BODY_ACCRETION_BIT;
    ((container.0 & prefix) == prefix).then_some((container.0 & (BODY_ACCRETION_BIT - 1)) as usize)
}

pub fn is_accretion_reservoir(container: crate::mass_transport::ContainerId) -> bool {
    container == ACCRETION_RESERVOIR || accretion_reservoir_body(container).is_some()
}

fn is_exterior(container: crate::mass_transport::ContainerId) -> bool {
    container.0 & EXTERIOR_BIT != 0
}

/// capture the conserved density immediately before a post-step material
/// removal operator.
pub fn snapshot_accretion_density<const D: usize, const DOF: usize, Mem: MemorySpace>(
    sim: &FieldStore<D, DOF, Mem, f64>,
) -> Vec<f64> {
    sim.geom
        .interior
        .iter()
        .map(|coord| *sim.fields.cons.den.view().at(coord))
        .collect()
}

/// transfer tracers with the accepted cellwise mass removed by post-step
/// immersed-body penalization into the aggregate accretion reservoir.
pub fn advance_accretion_transport<R, const D: usize, const DOF: usize, M, E, S, Mem>(
    sim: &mut crate::state::SimStateGeneric<R, D, DOF, M, E, S, Mem>,
    density_before: &[f64],
) -> Result<(), String>
where
    R: symbi_hydro::regime::Regime<f64, D>,
    M: symbi_geometry::Metric<f64, D> + Copy,
    E: symbi_hydro::eos::Eos<f64>,
    S: symbi_xpu::ExecutionSpace,
    Mem: MemorySpace,
{
    let interior = sim.geom.interior.clone();
    let geometry = sim.geom.block_geometry(sim.physics.metric);
    let layout = TransportLayout::single(&interior);
    advance_accretion_transport_store(&mut sim.store, &geometry, layout, density_before)
}

pub fn advance_accretion_transport_store<const D: usize, const DOF: usize, M, Mem>(
    sim: &mut FieldStore<D, DOF, Mem, f64>,
    geometry: &symbi_geometry::BlockGeometry<M, f64, D>,
    layout: TransportLayout<D>,
    density_before: &[f64],
) -> Result<(), String>
where
    M: symbi_geometry::Metric<f64, D> + Copy,
    Mem: MemorySpace,
{
    use crate::mass_transport::{MassTransfer, SamplingKey, TransportKernel, sample_systematic};
    use std::collections::BTreeMap;

    let interior = sim.geom.interior.clone();
    if density_before.len() != interior.volume() {
        return Err("accretion density snapshot has the wrong cell count".to_string());
    }
    let body_receipts = sim
        .immersed
        .as_ref()
        .map(|immersed| immersed.accretion_receipts())
        .unwrap_or_default();
    let Some(mut tracers) = sim.tracers.take() else {
        return Ok(());
    };
    let mut by_source = BTreeMap::new();
    for (ii, &owner) in tracers.owner.iter().enumerate() {
        if !tracers.flags[ii].escaped && !tracers.flags[ii].crossed_sink {
            by_source
                .entry(owner)
                .or_insert_with(Vec::new)
                .push(tracers.id[ii]);
        }
    }
    let key = SamplingKey {
        run_seed: tracers.run_seed,
        epoch: sim.iteration | (1 << 63),
    };
    let mut assignments = BTreeMap::new();
    for (linear, coord) in interior.iter().enumerate() {
        let source = cell_container(coord, &interior, layout);
        let Some(ids) = by_source.get(&source) else {
            continue;
        };
        let before = density_before[linear];
        let after = *sim.fields.cons.den.view().at(coord);
        let volume = geometry.volume(coord);
        let mut transfers: Vec<MassTransfer> = body_receipts
            .iter()
            .enumerate()
            .filter_map(|(body, receipt)| {
                let mass = receipt.get(linear).copied().unwrap_or(0.0).max(0.0);
                (mass > 0.0).then_some(MassTransfer {
                    destination: body_accretion_reservoir(body),
                    mass,
                })
            })
            .collect();
        if transfers.is_empty() {
            let removed_density = (before - after).max(0.0);
            transfers.push(MassTransfer {
                destination: ACCRETION_RESERVOIR,
                mass: removed_density * volume,
            });
        }
        let kernel = TransportKernel::new(source, before * volume, transfers)?;
        assignments.extend(sample_systematic(&kernel, ids, key));
    }
    for (ii, id) in tracers.id.iter().enumerate() {
        if let Some(&owner) = assignments
            .get(id)
            .filter(|&&owner| is_accretion_reservoir(owner))
        {
            tracers.owner[ii] = owner;
            tracers.flags[ii].crossed_sink = true;
            tracers.flags[ii].crossing_time = sim.time + sim.dt;
        }
    }
    sim.tracers = Some(tracers);
    Ok(())
}

fn exterior_container(axis: usize, high: bool) -> crate::mass_transport::ContainerId {
    crate::mass_transport::ContainerId(EXTERIOR_BIT | ((axis as u64) << 1) | high as u64)
}

pub fn cell_container_id(linear: usize, level: u8) -> crate::mass_transport::ContainerId {
    assert!(level < 64, "tracer refinement level {level} exceeds 63");
    assert!(
        linear as u64 <= CELL_LINEAR_MASK,
        "tracer cell index {linear} exceeds the 56-bit cell-address space"
    );
    crate::mass_transport::ContainerId(((level as u64) << CELL_LEVEL_SHIFT) | linear as u64)
}

pub fn cell_container_address(
    container: crate::mass_transport::ContainerId,
) -> Option<(u8, usize)> {
    if container.0 & (EXTERIOR_BIT | ACCRETION_RESERVOIR_ID) != 0 {
        return None;
    }
    let level = ((container.0 & CELL_LEVEL_MASK) >> CELL_LEVEL_SHIFT) as u8;
    let linear = (container.0 & CELL_LINEAR_MASK) as usize;
    Some((level, linear))
}

pub fn refresh_derived_positions_store<const D: usize, const DOF: usize, M, Mem>(
    sim: &mut FieldStore<D, DOF, Mem, f64>,
    geometry: &symbi_geometry::BlockGeometry<M, f64, D>,
    layout: TransportLayout<D>,
) where
    M: symbi_geometry::Metric<f64, D> + Copy,
    Mem: MemorySpace,
{
    let interior = sim.geom.interior.clone();
    let Some(tracers) = sim.tracers.as_mut() else {
        return;
    };
    for (ii, &owner) in tracers.owner.iter().enumerate() {
        let Some(coord) = container_cell(owner, &interior, layout) else {
            continue;
        };
        let mut position: [f64; D] = geometry.centroid(coord).into();
        if sim.motion.homologous {
            for value in &mut position {
                *value *= sim.motion.a;
            }
        } else if D > 0 {
            position[0] += sim.motion.a_dot * sim.time;
        }
        tracers.x[ii] = position;
    }
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
    cell_container_id(linear, layout.level)
}

fn container_cell<const D: usize>(
    container: crate::mass_transport::ContainerId,
    domain: &symbi_algebra::Domain<D>,
    layout: TransportLayout<D>,
) -> Option<[isize; D]> {
    let (level, mut linear) = cell_container_address(container)?;
    if level != layout.level {
        return None;
    }
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
        return cell_container_id(linear, layout.level);
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
            cell_container_id(linear, layout.level)
        }
        crate::state::BoundaryType::Reflect => cell_container(coord, domain, layout),
        crate::state::BoundaryType::CoarseFine => cell_container(coord, domain, layout),
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
    partition_seeded(global, set, counts)
}

pub fn seed_and_partition_with_cohorts<const D: usize, const DOF: usize, Mem: MemorySpace>(
    global: &FieldStore<D, DOF, Mem, f64>,
    n: usize,
    counts: [usize; D],
    cell_cohorts: &[u16],
) -> Result<Vec<TracerSet<D>>, String> {
    let set = seed_mass_weighted_with_cohorts(global, n, cell_cohorts)?;
    Ok(partition_seeded(global, set, counts))
}

fn partition_seeded<const D: usize, const DOF: usize, Mem: MemorySpace>(
    global: &FieldStore<D, DOF, Mem, f64>,
    set: TracerSet<D>,
    counts: [usize; D],
) -> Vec<TracerSet<D>> {
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
        per_tile[dest].cohort.push(set.cohort[i]);
        per_tile[dest].flags.push(set.flags[i]);
        per_tile[dest].owner.push(set.owner[i]);
        per_tile[dest].step_owner.push(set.step_owner[i]);
        per_tile[dest].step_flags.push(set.step_flags[i]);
        per_tile[dest].run_seed = set.run_seed;
        per_tile[dest].next_id = set.next_id;
    }
    per_tile
}
