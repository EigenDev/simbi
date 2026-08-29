// =============================================================================
// hierarchy.rs
//
// the static-mesh-refinement (SMR) hierarchy. each
// level is a complete SimStateGeneric + its KernelSet; the hierarchy adds only
// inter-level coordination: recursive berger-oliger subcycling with
// time-interpolated coarse-fine ghost prolongation, conservative restriction,
// and flux-register refluxing. the single-level engine is untouched —
//
// single-coverage cap (this is SMR): each level refines one box
// (`coverage: Option<Domain>`), with one FluxRegister per coarse-fine
// level-pair. the refined region is fixed at setup and stays where it was
// placed; there is a single patch per level and no berger-rigoutsos
// clustering. multi-patch adaptive refinement (a level as a disjoint cover of
// Domains) lies outside this implementation.
//
// advance_level re-sequences the SSP stage loop (sim/evolve.rs::step) so the
// register accumulation slots between flux() and the stage update.
//
// levels share absolute index space: a fine level covering coarse cells
// [cov_lo, cov_hi) lives at fine indices [2*cov_lo, 2*cov_hi), and every level
// keeps the same global physical origin — `geom.centroid` is correct on every
// level with the same formula, and no coverage-relative translation exists.
//
// two invariants pin the construction: a 1-level hierarchy reproduces the uni-grid
// evolve() bit-for-bit, and a 2-level static nesting conserves the composite-grid
// totals to machine precision.
//
// usage:
//  let mut hier = Hierarchy::with_refinement(sim, kernels, &regions, order, make)?;
//  hier.evolve(t_final)?;
// =============================================================================

use symbi_algebra::{Domain, Space};
use symbi_geometry::Metric;
use symbi_hydro::eos::Eos;
use symbi_hydro::regime::Regime;
use symbi_xpu::{ExecutionSpace, MemorySpace};

use symbi_grid::Field;

use super::emf_register::EmfRegister;
use super::equilibrium::{
    EquilibriumFlux, accumulate as equilibrium_accumulate, imbalance_from_stage,
    overwrite as equilibrium_overwrite, residual_components, restore_gas_state, save_gas_state,
    snapshot_flux,
};
use super::flux_register::FluxRegister;
use super::tracer_interface::{
    InterfaceFace, InterfaceTransfer, interface_faces, interface_faces_with_layout,
    interface_mass_transfers, interface_transport_kernels,
};
use super::transfer::{
    ProlongOrder, ProlongSweepScratch, bcell_from_bface_region, bface_cf_halo_slabs,
    cf_ghost_slabs, copy_field, copy_field_region, prolong_face_field, prolong_field,
    prolong_prims_balanced, prolong_prims_swept, prolong_prims_targeted, restrict_band_balanced,
    restrict_bface, restrict_cell_field, restrict_cons,
};
use std::ops::ControlFlow;
use symbi_hydro::state::PrimFromSlots;
use symbi_sim::decomp::{HaloTransport, drain_devices, exchange_grid, flatten, unflatten};
use symbi_sim::driver::{
    advance_clock, advance_state_clock, book_horizon_receipt, check_dt_or_panic, evolve_bodies,
    horizon_request, needs_step_snapshot, prof, stage_time_fractions,
};
use symbi_sim::hydro_ops::scan_c2p_errors;
use symbi_sim::stage::{HookPoint, StageArgs, fold_stage};
use symbi_sim::state::{
    Boundaries, BoundaryType, ConsFieldsGeneric, FieldDecimation, FieldKind, FieldStore,
    PrimFieldsGeneric, SimStateGeneric, Timestepping, array_field_zeros, axis_name,
};
use symbi_sim::substrate_seam::KernelSet;

/// the refinement ratio. fixed at 2 (the transfer kernels are baked at 2 in
/// the aot registry; the builders accept any ratio when that changes).
const RATIO: usize = 2;

// =============================================================================
// refinement regions
// =============================================================================

/// a box in physical space that one finer level will cover (static nesting:
/// region k refines level k). snapped to coarse cell faces by rounding.
pub struct RefinementRegion<const D: usize> {
    pub x_lo: [f64; D],
    pub x_hi: [f64; D],
}

// =============================================================================
// per-level data
// =============================================================================

/// one refinement level: the state, the kernel set that advances it, and the
/// inter-level metadata (present only where a finer level exists).
pub struct LevelData<R, const NDIM: usize, const DOF: usize, M, E, S, Mem, K>
where
    R: Regime<f64, NDIM>,
    M: Metric<f64, NDIM> + Copy,
    E: Eos<f64>,
    S: ExecutionSpace,
    Mem: MemorySpace,
    K: KernelSet<NDIM, DOF, Mem, f64>,
{
    pub state: SimStateGeneric<R, NDIM, DOF, M, E, S, Mem>,
    pub kernels: K,
    /// primitives at this level's step start, for the time-interpolated
    /// coarse-fine ghost prolongation of the finer level. None on the finest.
    pub prim_old: Option<PrimFieldsGeneric<NDIM, DOF, Mem>>,
    /// coarse scratch for the lerp-then-prolong split: `field_lerp` writes
    /// `(1-alpha)*prim_old + alpha*prim` here once per coarse cell, and the
    /// single-snapshot prolong reads it — half the time-pair kernel's gather
    /// traffic. allocated alongside prim_old (None on the finest) and reused on
    /// every call, so the step loop allocates nothing.
    pub prim_lerp: Option<PrimFieldsGeneric<NDIM, DOF, Mem>>,
    /// cell-centered B at this level's step start (mhd only) — the magnetic
    /// counterpart of prim_old for the finer level's ghost prolongation.
    pub bcell_old: Option<[Field<f64, NDIM, Mem>; NDIM]>,
    /// staggered face B at this level's step start (mhd only) — feeds the
    /// finer level's bface transverse-halo prolongation (per-component
    /// staggered domains, cloned from this level's bface).
    pub bface_old: Option<[Field<f64, NDIM, Mem>; NDIM]>,
    /// per-slab intermediates of the axis-split prolongation into this level,
    /// in `cf_ghost_slabs` order. SMR slabs are static, so
    /// the shapes are too: lazily allocated on the first prolongation, reused
    /// on every call (the step loop allocates nothing). uninitialized on the
    /// root.
    pub prolong_sweep: std::sync::OnceLock<Vec<ProlongSweepScratch<NDIM, DOF, Mem>>>,
    /// this level's pressure departures from the composite-lattice equilibrium
    /// under its parent's seam bands — the balanced restriction's fine scratch.
    /// lazily allocated over this level's allocated domain on the first
    /// balanced restriction, reused on every call. unused on the root and on
    /// plain hierarchies.
    pub band_departure: std::sync::OnceLock<Field<f64, NDIM, Mem>>,
    /// the region of this level covered by the next finer level, in absolute
    /// indices of this level. None on the finest. single-coverage cap: one
    /// refined box per level (static refinement / SMR — a single `Domain`,
    /// placed at setup).
    pub coverage: Option<Domain<NDIM>>,
    /// this level's numerical flux of the run's stationary target state, `F(qt)`, when the run
    /// declares one. the flux registers subtract it from both sides of the coarse-fine
    /// difference, so what is refluxed is the deviation from the target and a state sitting
    /// exactly on the target accumulates nothing. per level, because the mismatch the register
    /// would otherwise see is precisely the difference between the two grids' reconstructions of
    /// the same exact solution.
    pub flux_eq: Option<EquilibriumFlux<NDIM, DOF, Mem>>,
    /// this level's discrete imbalance of that same target, `R = div_h F_h(qt) - s_h(qt)`, per
    /// unit time and per cell. a steady state solves the continuum equations, so the scheme leaves
    /// this residual at truncation order and an atmosphere seeded on the exact hydrostatic profile
    /// starts moving. every stage adds it back, which makes the target an exact fixed point.
    pub residual_eq: Option<ConsFieldsGeneric<NDIM, DOF, Mem>>,
    /// the target's own conserved state on this level — the reference the deviation is measured
    /// from. covered cells hold the volume-weighted restriction of the finer level's target, so
    /// the restriction the run performs every parent step leaves the target where it found it.
    /// defining it by an independent evaluation of the profile per level would move it.
    pub cons_eq: Option<ConsFieldsGeneric<NDIM, DOF, Mem>>,
    /// the target primitives on this level after the same c2p and ghost-fill path
    /// used by the live state. coarse-fine transfer prolongs live departures from
    /// the parent target and decodes them against this fixed fine target, so the
    /// boundary operator never reads the fine interior solution.
    pub prim_eq: Option<PrimFieldsGeneric<NDIM, DOF, Mem>>,
}

// =============================================================================
// hierarchy
// =============================================================================

/// static-refinement hierarchy: levels[0] = coarsest, levels[n-1] = finest.
/// a fatal crash: either the CFL watchdog (the wave speed went NaN or collapsed — an unphysical
/// c2p, e.g. V -> 1 near a boundary — so the next dt is NaN / non-positive / blown up) or a panic
/// caught inside the step (the FOFC freeze-streak halt, a poisoned-cell assertion). the evolve loop
/// halts on the last computed state and the driver snapshots a `.crashed` checkpoint + reports it,
/// so every crash of either class reaches the user with its state on disk.
#[derive(Clone, Debug)]
pub struct CrashReport {
    pub iter: u64,
    pub time: f64,
    pub dt_cfl: f64,
    pub dt_prev: f64,
    /// the payload of a panic caught inside `step_root`; the dt fields carry NaN in
    /// that case, and the watchdog's own reports leave this empty.
    pub panic: Option<String>,
}

struct HierarchyRetrySidecars<const D: usize> {
    discrete: Vec<Option<symbi_sim::tracers::TracerSet<D>>>,
    continuous: Vec<Option<symbi_sim::tracers::ContinuousTracerSnapshot<D>>>,
    bodies: Vec<Option<symbi_ib::BodyCollection<f64, D>>>,
    motion: Vec<symbi_geometry::MotionState<f64>>,
    clocks: Vec<(f64, u64, f64)>,
    censuses: Vec<Vec<(symbi_sim::census::CensusHistory, Vec<Option<f64>>)>>,
}

pub struct Hierarchy<R, const NDIM: usize, const DOF: usize, M, E, S, Mem, K>
where
    R: Regime<f64, NDIM>,
    M: Metric<f64, NDIM> + Copy,
    E: Eos<f64>,
    S: ExecutionSpace,
    Mem: MemorySpace,
    K: KernelSet<NDIM, DOF, Mem, f64>,
{
    pub levels: Vec<LevelData<R, NDIM, DOF, M, E, S, Mem, K>>,
    injection_ledger: std::collections::BTreeMap<symbi_sim::mass_transport::ContainerId, f64>,
    /// coarse-fine prolongation order (one above the evolution reconstruction).
    pub prolong_order: ProlongOrder,
    /// flux_registers[ll] corrects the interface between level ll and ll+1.
    pub flux_registers: Vec<FluxRegister<NDIM, DOF, Mem>>,
    /// emf_registers[ll] corrects the coarse bface at the same interface (mhd
    /// pairs only; None for pure hydro).
    pub emf_registers: Vec<Option<EmfRegister<NDIM, Mem>>>,
    tracer_interfaces: Vec<Vec<InterfaceFace<NDIM>>>,
    tracer_interface_ledgers: Vec<std::cell::RefCell<Vec<InterfaceTransfer>>>,
    tracer_root_global_cells: [usize; NDIM],
    tracer_root_offset: [usize; NDIM],
    /// set by `step_root` when the cfl dt is fatal (NaN / non-positive / a sudden blowup from a
    /// collapsed wave speed). `Some` halts the march at the last computed state; the driver writes a
    /// `.crashed` checkpoint and reports it, halting before advancing past t_final on garbage.
    pub crash: Option<CrashReport>,

    /// cells within this radius of the origin are excluded from the stationarity diagnostic.
    ///
    /// a declared target is a steady state of the equations the stage pipeline applies. inside a
    /// sink the drain removes mass and energy, so the target's imbalance there measures the drain
    /// rather than truncation error and reports non-convergence however exact the target is
    /// elsewhere. the well-balancing correction still lands on every cell; this radius decides
    /// which cells testify about convergence.
    ///
    /// `None` resolves to the largest accretion radius among the attached bodies, so a run with a
    /// sink gets the exclusion by default and a run with no bodies gets a zero radius that keeps
    /// every cell.
    pub equilibrium_mask_radius: Option<f64>,

    /// the previous step's raw cfl-derived dt — the quantity the collapse guard compares against.
    /// the level's `state.dt` is the accepted dt, which the `t_final` clamp and an explicit-step
    /// rejection both shrink; comparing a fresh cfl estimate against it reports a "collapse" every
    /// time a rejected step is replayed at a smaller dt and then recovers. 0 before the first step.
    pub prev_dt_cfl: f64,

    /// whether the coarse-fine ghost transfer prolongs departures from the local hydrostatic
    /// equilibrium; when off it prolongs the raw primitive state. `None` follows the kernel set:
    /// active precisely when the set reconstructs balanced (`KernelSet::hydrostatic_balance`) and
    /// a gravitating body supplies the potential. `Some(x)` forces the choice — the knob that
    /// measures the transfer as the load-bearing piece. see `cf_transfer_balanced`.
    pub balance_aware_transfer: Option<bool>,
}

impl<R, const NDIM: usize, const DOF: usize, M, E, S, Mem, K>
    Hierarchy<R, NDIM, DOF, M, E, S, Mem, K>
where
    R: Regime<f64, NDIM> + Copy,
    M: Metric<f64, NDIM> + Copy + Send + Sync,
    E: Eos<f64> + Copy + Send + Sync,
    S: ExecutionSpace,
    Mem: MemorySpace + Sync,
    K: KernelSet<NDIM, DOF, Mem, f64>,
{
    /// seed tracers from the hierarchy's active composite mass: covered coarse
    /// cells are omitted and their finest available children supply the mass.
    pub fn seed_mass_tracers(&self, n: usize) -> symbi_sim::tracers::TracerSet<NDIM> {
        assert!(
            self.levels[0].state.geom.coords == symbi_geometry::Geometry::Cartesian,
            "refined mass tracers require cartesian geometry"
        );
        assert!(
            self.levels[0]
                .state
                .geom
                .interior
                .spaces
                .iter()
                .all(|space| space.lo == 0),
            "refined mass tracers require a root grid beginning at global index zero"
        );

        let root_cells: [usize; NDIM] =
            std::array::from_fn(|aa| self.levels[0].state.geom.interior.spaces[aa].size());
        let mut owners = Vec::new();
        let mut cells = Vec::new();
        let mut masses = Vec::new();
        for (ll, level) in self.levels.iter().enumerate() {
            let scale = 1usize
                .checked_shl(ll as u32)
                .expect("refinement level exceeds machine index width");
            let global_cells: [usize; NDIM] = std::array::from_fn(|aa| root_cells[aa] * scale);
            let geometry = level.state.geom.block_geometry(level.state.physics.metric);
            for coord in level.state.geom.interior.iter() {
                if !level.state.composite_ownership.owns_leaf(coord) {
                    continue;
                }
                let mut linear = 0usize;
                let mut stride = 1usize;
                for aa in 0..NDIM {
                    linear += coord[aa] as usize * stride;
                    stride *= global_cells[aa];
                }
                let center = geometry.centroid(coord);
                let widths = level.state.geom.dx;
                let lo = std::array::from_fn(|aa| center[aa] - 0.5 * widths[aa]);
                owners.push(symbi_sim::tracers::cell_container_id(linear, ll as u8));
                cells.push((lo, widths));
                masses.push(*level.state.fields.cons.den.view().at(coord) * geometry.volume(coord));
            }
        }
        symbi_sim::tracers::seed_weighted_cells(&owners, &cells, &masses, n)
    }

    /// seed the active composite hierarchy once and partition the resulting
    /// globally identified population onto the levels that own its cells.
    pub fn attach_mass_tracers(&mut self, n: usize) {
        let seeded = self.seed_mass_tracers(n);
        let mut per_level: Vec<symbi_sim::tracers::TracerSet<NDIM>> = (0..self.levels.len())
            .map(|_| symbi_sim::tracers::TracerSet {
                weight: seeded.weight,
                run_seed: seeded.run_seed,
                next_id: seeded.next_id,
                ..Default::default()
            })
            .collect();
        for ii in 0..seeded.len() {
            let (level, _) = symbi_sim::tracers::cell_container_address(seeded.owner[ii])
                .expect("composite seed produced a non-cell owner");
            let target = per_level
                .get_mut(level as usize)
                .expect("composite seed owner names an absent refinement level");
            target.x.push(seeded.x[ii]);
            target.id.push(seeded.id[ii]);
            target.cohort.push(seeded.cohort[ii]);
            target.flags.push(seeded.flags[ii]);
            target.owner.push(seeded.owner[ii]);
            target.step_owner.push(seeded.step_owner[ii]);
            target.step_flags.push(seeded.step_flags[ii]);
        }
        for (level, tracers) in self.levels.iter_mut().zip(per_level) {
            level.state.tracers = Some(tracers);
        }
    }

    /// relocate continuous particles onto the finest active level containing
    /// their authoritative physical position.
    pub fn migrate_continuous_tracers_to_finest(&mut self) -> Result<usize, String> {
        let descriptors: Vec<_> = self
            .levels
            .iter()
            .enumerate()
            .map(|(level, data)| {
                (
                    level,
                    symbi_sim::tracers::partition_physical_bounds(&data.state.geom),
                    data.state.geom.maps,
                    data.state.geom.x_lo,
                    data.state.geom.dx,
                    data.state.geom.interior.clone(),
                    self.tracer_layout(level),
                )
            })
            .collect();
        let metadata = self
            .levels
            .iter()
            .filter_map(|level| level.state.continuous_tracers.as_ref())
            .max_by_key(|tracers| usize::from(tracers.len > 0))
            .map(|tracers| {
                (
                    tracers.order,
                    tracers.weight,
                    tracers.run_seed,
                    tracers.next_id,
                    tracers.injection_remainder,
                )
            })
            .ok_or_else(|| "continuous tracer population is missing".to_string())?;
        let locate = |position: [f64; NDIM]| {
            descriptors.iter().rev().find_map(
                |(level, bounds, maps, x_lo, dx, interior, layout)| {
                    if (0..NDIM)
                        .any(|dd| position[dd] < bounds[dd].0 || position[dd] >= bounds[dd].1)
                    {
                        return None;
                    }
                    let coord: [isize; NDIM] = std::array::from_fn(|dd| match maps {
                        Some(maps) => maps[dd].index_at(position[dd]),
                        None => ((position[dd] - x_lo[dd]) / dx[dd]).floor() as isize,
                    });
                    if !interior.contains(coord) {
                        return None;
                    }
                    let mut linear = 0usize;
                    let mut stride = 1usize;
                    for dd in 0..NDIM {
                        let local = (coord[dd] - interior.spaces[dd].lo) as usize;
                        linear += (layout.tile_offset[dd] + local) * stride;
                        stride *= layout.global_cells[dd];
                    }
                    Some((
                        *level,
                        symbi_sim::tracers::cell_container_id(linear, layout.level),
                    ))
                },
            )
        };
        for level in &mut self.levels {
            if let Some(tracers) = level.state.continuous_tracers.as_mut() {
                tracers.order = metadata.0;
                tracers.weight = metadata.1;
                tracers.run_seed = metadata.2;
                tracers.next_id = metadata.3;
                tracers.injection_remainder = metadata.4;
            }
        }
        let mut moved = Vec::new();
        for (source_level, level) in self.levels.iter_mut().enumerate() {
            let Some(tracers) = level.state.continuous_tracers.as_mut() else {
                continue;
            };
            let mut ii = 0;
            while ii < tracers.len {
                let inactive = unsafe {
                    *tracers.escaped.as_ptr::<u8>().add(ii) != 0
                        || *tracers.crossed_sink.as_ptr::<u8>().add(ii) != 0
                };
                if inactive {
                    ii += 1;
                    continue;
                }
                let position: [f64; NDIM] =
                    unsafe { std::array::from_fn(|dd| *tracers.x[dd].as_ptr::<f64>().add(ii)) };
                let Some((target_level, owner)) = locate(position) else {
                    ii += 1;
                    continue;
                };
                if target_level == source_level {
                    unsafe {
                        *tracers
                            .owner
                            .as_mut_ptr::<symbi_sim::mass_transport::ContainerId>()
                            .add(ii) = owner;
                    }
                    ii += 1;
                } else {
                    let mut record = tracers.swap_remove_host(ii)?;
                    record.owner = owner;
                    moved.push((target_level, record));
                }
            }
        }
        let count = moved.len();
        for (target_level, record) in moved {
            if self.levels[target_level].state.continuous_tracers.is_none() {
                let mut tracers = symbi_sim::tracers::ContinuousTracerSet::allocate(0, metadata.0)?;
                tracers.weight = metadata.1;
                tracers.run_seed = metadata.2;
                tracers.next_id = metadata.3;
                tracers.injection_remainder = metadata.4;
                self.levels[target_level].state.continuous_tracers = Some(tracers);
            }
            self.levels[target_level]
                .state
                .continuous_tracers
                .as_mut()
                .expect("continuous tracer destination was initialized")
                .push_host(record)?;
        }
        Ok(count)
    }

    fn tracer_cell(
        &self,
        owner: symbi_sim::mass_transport::ContainerId,
    ) -> Option<(usize, [isize; NDIM])> {
        let (level, mut linear) = symbi_sim::tracers::cell_container_address(owner)?;
        let level = level as usize;
        let scale = 1usize.checked_shl(level as u32)?;
        let global_cells: [usize; NDIM] =
            std::array::from_fn(|aa| self.tracer_root_global_cells[aa] * scale);
        let global_coord: [isize; NDIM] = std::array::from_fn(|aa| {
            let index = linear % global_cells[aa];
            linear /= global_cells[aa];
            index as isize
        });
        let coord = std::array::from_fn(|aa| {
            global_coord[aa] - (self.tracer_root_offset[aa] * scale) as isize
        });
        self.levels
            .get(level)?
            .state
            .geom
            .interior
            .contains(coord)
            .then_some((level, coord))
    }

    fn tracer_layout(&self, level: usize) -> symbi_sim::tracers::TransportLayout<NDIM> {
        let scale = 1usize << level;
        let interior = &self.levels[level].state.geom.interior;
        symbi_sim::tracers::TransportLayout {
            global_cells: std::array::from_fn(|aa| self.tracer_root_global_cells[aa] * scale),
            tile_offset: std::array::from_fn(|aa| {
                self.tracer_root_offset[aa] * scale + interior.spaces[aa].lo as usize
            }),
            level: level as u8,
        }
    }

    pub fn set_tracer_root_layout(
        &mut self,
        global_cells: [usize; NDIM],
        tile_offset: [usize; NDIM],
    ) {
        self.tracer_root_global_cells = global_cells;
        self.tracer_root_offset = tile_offset;
        let parent_cells: [usize; NDIM] =
            std::array::from_fn(|aa| self.levels[0].state.geom.interior.spaces[aa].size());
        self.tracer_interfaces = (0..self.levels.len().saturating_sub(1))
            .map(|ll| {
                interface_faces_with_layout(
                    self.levels[ll].coverage.as_ref().unwrap(),
                    parent_cells,
                    tile_offset,
                    global_cells,
                    ll as u8,
                )
            })
            .collect();
    }

    fn tracer_owner_is_active(&self, owner: symbi_sim::mass_transport::ContainerId) -> bool {
        self.tracer_cell(owner).is_some_and(|(level, coord)| {
            self.levels[level]
                .state
                .composite_ownership
                .owns_leaf(coord)
        })
    }

    fn sync_tracer_spawn_state(&mut self, source_level: usize) {
        let Some(source) = self.levels[source_level].state.tracers.as_ref() else {
            return;
        };
        let next_id = source.next_id;
        let injection_remainder = source.injection_remainder;
        for level in &mut self.levels {
            if let Some(tracers) = level.state.tracers.as_mut() {
                tracers.next_id = next_id;
                tracers.injection_remainder = injection_remainder;
            }
        }
    }

    fn set_tracer_spawn_state(&mut self, next_id: u64, injection_remainder: f64) {
        for level in &mut self.levels {
            if let Some(tracers) = level.state.tracers.as_mut() {
                tracers.next_id = next_id;
                tracers.injection_remainder = injection_remainder;
            }
        }
    }

    fn spawn_pending_injection(
        &mut self,
        level: usize,
        next_id: u64,
        injection_remainder: f64,
    ) -> Result<(u64, f64), String> {
        if level >= self.levels.len() || !self.levels[level].state.has_tracers() {
            return Ok((next_id, injection_remainder));
        }
        self.set_tracer_spawn_state(next_id, injection_remainder);
        let ledger = std::mem::take(&mut self.injection_ledger);
        let layout = self.tracer_layout(level);
        let geometry = self.levels[level]
            .state
            .geom
            .block_geometry(self.levels[level].state.physics.metric);
        symbi_sim::tracers::spawn_boundary_injection_store(
            &mut self.levels[level].state.store,
            &geometry,
            layout,
            ledger,
        )?;
        self.sync_tracer_spawn_state(level);
        let tracers = self.levels[level].state.tracers.as_ref().unwrap();
        Ok((tracers.next_id, tracers.injection_remainder))
    }

    fn tracer_cell_mass(&self, owner: symbi_sim::mass_transport::ContainerId) -> Option<f64> {
        let (level, coord) = self.tracer_cell(owner)?;
        let state = &self.levels[level].state;
        let geometry = state.geom.block_geometry(state.physics.metric);
        Some(*state.fields.cons.den.view().at(coord) * geometry.volume(coord))
    }

    fn apply_interface_event(
        &mut self,
        transfers: &[InterfaceTransfer],
        epoch: u64,
    ) -> Result<(), String> {
        use std::collections::BTreeMap;
        use symbi_sim::mass_transport::{SamplingKey, sample_systematic};

        if transfers.is_empty() {
            return Ok(());
        }
        let mut post_mass = BTreeMap::new();
        for transfer in transfers {
            for owner in [transfer.source, transfer.destination] {
                if let std::collections::btree_map::Entry::Vacant(entry) = post_mass.entry(owner) {
                    let mass = self.tracer_cell_mass(owner).ok_or_else(|| {
                        format!("interface receipt names inactive cell {}", owner.0)
                    })?;
                    entry.insert(mass);
                }
            }
        }
        let kernels = interface_transport_kernels(transfers, &post_mass)?;
        let run_seed = self
            .levels
            .iter()
            .find_map(|level| level.state.tracers.as_ref().map(|tracers| tracers.run_seed))
            .unwrap_or(0);
        let key = SamplingKey { run_seed, epoch };
        let mut assignments = BTreeMap::new();
        for kernel in kernels {
            let ids: Vec<u64> = self
                .levels
                .iter()
                .filter_map(|level| level.state.tracers.as_ref())
                .flat_map(|tracers| {
                    tracers
                        .id
                        .iter()
                        .zip(&tracers.owner)
                        .filter_map(|(&id, &owner)| (owner == kernel.source()).then_some(id))
                })
                .collect();
            assignments.extend(sample_systematic(&kernel, &ids, key));
        }
        for level in &mut self.levels {
            if let Some(tracers) = level.state.tracers.as_mut() {
                for (ii, id) in tracers.id.iter().enumerate() {
                    if let Some(&destination) = assignments.get(id) {
                        tracers.owner[ii] = destination;
                        tracers.step_owner[ii] = destination;
                    }
                }
            }
        }

        let mut migrating = Vec::new();
        for (level_index, level) in self.levels.iter_mut().enumerate() {
            let Some(tracers) = level.state.tracers.as_mut() else {
                continue;
            };
            let mut ii = 0usize;
            while ii < tracers.len() {
                let destination_level =
                    symbi_sim::tracers::cell_container_address(tracers.owner[ii])
                        .map(|address| address.0 as usize)
                        .unwrap_or(level_index);
                if destination_level == level_index {
                    ii += 1;
                    continue;
                }
                tracers.x.swap_remove(ii);
                migrating.push((
                    destination_level,
                    tracers.id.swap_remove(ii),
                    tracers.cohort.swap_remove(ii),
                    tracers.flags.swap_remove(ii),
                    tracers.owner.swap_remove(ii),
                    tracers.step_owner.swap_remove(ii),
                    tracers.step_flags.swap_remove(ii),
                ));
            }
        }
        for (destination_level, id, cohort, flags, owner, step_owner, step_flags) in migrating {
            let (_, coord) = self
                .tracer_cell(owner)
                .ok_or_else(|| format!("interface destination {} is not active", owner.0))?;
            let state = &self.levels[destination_level].state;
            let x = state.geom.centroid(coord);
            let tracers = self.levels[destination_level]
                .state
                .tracers
                .as_mut()
                .expect("every refined level carries a tracer set");
            tracers.x.push(x);
            tracers.id.push(id);
            tracers.cohort.push(cohort);
            tracers.flags.push(flags);
            tracers.owner.push(owner);
            tracers.step_owner.push(step_owner);
            tracers.step_flags.push(step_flags);
        }
        Ok(())
    }

    /// decimate the hierarchy to a screen-sized density heatmap, compositing the
    /// nested refinement levels: each root cell descends to the finest level whose
    /// `coverage` box contains it (SMR = single nested box per level, ratio 2), so
    /// the refined region shows its fine detail and the rest shows the coarse grid.
    /// a single-level hierarchy is just the root decimation. cost is screen-bounded.
    pub fn field_slice_composite(
        &self,
        max_dim: usize,
        index: usize,
        orient: usize,
        zoom: usize,
    ) -> Option<FieldDecimation> {
        // single grid: the root is the only level, so reuse the plain per-state decimation.
        if self.levels.len() <= 1 {
            return self.levels[0]
                .state
                .field_slice_oriented(max_dim, index, orient, zoom);
        }
        if !Mem::IS_HOST_ACCESSIBLE {
            return None;
        }
        let (kind, name) = self.levels[0].state.nth_field(index);
        let interior = &self.levels[0].state.geom.interior;
        // the display axes per orientation (the same law as field_slice_oriented):
        // (horizontal, vertical) of the picture; the remaining 3D axis holds its
        // mid-plane index. 2D shows (0, 1).
        let (ah, av) = if NDIM >= 3 {
            match orient % 3 {
                1 => (0usize, 2usize),
                2 => (1, 2),
                _ => (0, 1),
            }
        } else {
            (0, 1)
        };
        // the zoomed window: a centered span of size/2^zoom, at least 4 cells.
        let windowed = |sp: &symbi_algebra::Space| -> symbi_algebra::Space {
            if zoom == 0 {
                return sp.clone();
            }
            let size = sp.size() as isize;
            let span = (size >> zoom.min(4)).max(4.min(size));
            let mid = sp.lo + size / 2;
            symbi_algebra::Space {
                name: sp.name,
                lo: mid - span / 2,
                hi: mid - span / 2 + span,
            }
        };
        let sp0 = windowed(&interior.spaces[ah]);
        let sp1 = windowed(interior.spaces.get(av)?); // None on a 1D grid
        let (nx, ny) = (sp0.size(), sp1.size());
        if nx == 0 || ny == 0 {
            return None;
        }
        let m = max_dim.max(1);
        let sx = ((nx + m - 1) / m).max(1);
        let sy = ((ny + m - 1) / m).max(1);
        let out_w = (nx + sx - 1) / sx;
        let out_h = (ny + sy - 1) / sy;
        let (lo0, hi0, lo1, hi1) = (sp0.lo, sp0.hi, sp1.lo, sp1.hi);
        // the base root index: mid-plane on every non-display axis.
        let base: [isize; NDIM] = std::array::from_fn(|ax| {
            let s = &interior.spaces[ax];
            s.lo + (s.size() / 2) as isize
        });

        let mut data = Vec::with_capacity(out_w * out_h);
        let mut vmin = f64::INFINITY;
        let mut vmax = f64::NEG_INFINITY;
        for j in 0..out_h {
            let y0 = lo1 + (j * sy) as isize;
            let y1 = (y0 + sy as isize).min(hi1);
            for i in 0..out_w {
                let x0 = lo0 + (i * sx) as isize;
                let x1 = (x0 + sx as isize).min(hi0);
                // block-average the root-cell footprint, each root cell resolved to
                // the finest level covering it.
                let mut sum = 0.0_f64;
                let mut cnt = 0u32;
                let mut ry = y0;
                while ry < y1 {
                    let mut rx = x0;
                    while rx < x1 {
                        let mut idx = base;
                        idx[ah] = rx;
                        idx[av] = ry;
                        sum += self.sample_finest_at(idx, kind);
                        cnt += 1;
                        rx += 1;
                    }
                    ry += 1;
                }
                let v = sum / cnt.max(1) as f64;
                vmin = vmin.min(v);
                vmax = vmax.max(v);
                data.push(v as f32);
            }
        }
        Some(FieldDecimation {
            width: out_w,
            height: out_h,
            data,
            vmin,
            vmax,
            name: name.into(),
            preserve_aspect: false,
        })
    }

    /// resolve a root-level index to the finest level covering it and read the
    /// field there — the axis-general coverage descent (the display slice builds
    /// the root index per its own orientation/zoom and hands it here).
    fn sample_finest_at(&self, mut idx: [isize; NDIM], kind: FieldKind) -> f64 {
        let mut lvl = 0usize;
        while lvl + 1 < self.levels.len() {
            let cov = match &self.levels[lvl].coverage {
                Some(c) if domain_contains(c, &idx) => c,
                _ => break,
            };
            let finer = &self.levels[lvl + 1].state.geom.interior;
            idx = std::array::from_fn(|ax| finer.spaces[ax].lo + (idx[ax] - cov.spaces[ax].lo) * 2);
            lvl += 1;
        }
        self.levels[lvl].state.field_value(idx, kind)
    }

    /// seed every fine level's interior from its parent by conservative
    /// prolongation at the hierarchy's `prolong_order` (= interior reconstruction
    /// order + 1, the same order the coarse-fine ghost prolongation uses). the IC
    /// fill for a hierarchy whose coarse level was seeded but whose fine levels are
    /// still empty — a coarse-only IC, e.g., a python-driven `prim_gen`. each
    /// conserved component is prolonged coarse -> fine interior; the fine levels
    /// then refine the solution as they evolve.
    ///
    /// the conserved components are prolonged independently, and a component-wise
    /// high-order prolongation is free to break the admissibility inequality
    /// E >= |m|^2 / (2 rho): near an extremum of the momentum the
    /// non-monotone stencil overshoots `m` while `E` interpolates low, and the
    /// fine cell's internal energy E - |m|^2/(2 rho) goes negative wherever the
    /// kinetic energy density is comparable to the internal — e.g. a velocity
    /// field whose local mach approaches unity on a cold background. each level
    /// is therefore c2p-audited after its fill, and every inadmissible cell is
    /// re-seeded by piecewise-constant injection of its covering parent cell,
    /// which is unconditionally admissible (the children are copies of an
    /// admissible parent) and preserves the coarse-cell integral exactly. the
    /// audit repairs hydro components only; returns the count of re-seeded
    /// cells so callers can assert how often the fallback engaged.
    ///
    /// the seeded state is hydro-complete plus the mhd cell-centered B. the staggered
    /// fine `bface` comes from the mhd refinement path, which supplies the face
    /// prolongation a face field requires.
    pub fn seed_fine_from_coarse(&self) -> symbi_xpu::Result<usize> {
        let mut reseeded = 0usize;
        for ll in 1..self.levels.len() {
            let (lo, hi) = self.levels.split_at(ll);
            let coarse = &lo[ll - 1].state;
            let fine = &hi[0].state;
            let region = fine.geom.interior.clone();
            let order = self.prolong_order;
            // the prolong takes a time pair (old, new) and forms (1-alpha)*old +
            // alpha*new. the IC has no second time level, so bind a disjoint zero
            // as `new` (the kernel forbids old/new aliasing) and alpha = 0, which
            // reads `old` only.
            let zero = Field::<f64, NDIM, Mem>::zeros(&coarse.geom.allocated)?;
            let (cc, fc) = (&coarse.fields.cons, &fine.fields.cons);
            prolong_field(&cc.den, &zero, &fc.den, &region, order, 0.0);
            for k in 0..DOF {
                prolong_field(&cc.mom[k], &zero, &fc.mom[k], &region, order, 0.0);
            }
            if let (Some(cn), Some(fn_nrg)) = (cc.nrg_field(), fc.nrg_field()) {
                prolong_field(cn, &zero, fn_nrg, &region, order, 0.0);
            }
            // mhd: cell-centered B + the staggered faces (divergence-free
            // prolongation per normal axis — Balsara). seeding the faces div-free
            // is what lets the fine CT start at div(B)=0 and coarse-fine
            // consistent; the cell B is prolonged alongside for the flux.
            if let (Some(cm), Some(fm)) = (coarse.fields.mhd.as_ref(), fine.fields.mhd.as_ref()) {
                for k in 0..DOF {
                    prolong_field(&cm.bcell[k], &zero, &fm.bcell[k], &region, order, 0.0);
                }
                for dd in 0..NDIM {
                    let face_region = fine.geom.interior.extend(dd, 0, 1);
                    let zero_face = Field::<f64, NDIM, Mem>::zeros(cm.bface[dd].domain())?;
                    prolong_face_field(
                        dd,
                        &cm.bface[dd],
                        &zero_face,
                        &fm.bface[dd],
                        &face_region,
                        0.0,
                    );
                }
                fm.bface_initialized
                    .store(true, std::sync::atomic::Ordering::Relaxed);
            }

            // admissibility audit of the freshly filled level. the audit runs
            // inside the level loop so a repaired cell is what the next level
            // prolongs from — every level prolongs from an admissible parent.
            let lvl = &self.levels[ll];
            lvl.kernels.c2p(&lvl.state);
            symbi_substrate::regimes::substrate_gpu::device_sync::<Mem>();
            if scan_c2p_errors(&lvl.state).is_err() {
                let flagged: Vec<[isize; NDIM]> = lvl
                    .state
                    .geom
                    .interior
                    .iter()
                    .filter(|coord| *lvl.state.fields.c2p_error.view().at(*coord) != 0.0)
                    .collect();
                for coord in &flagged {
                    let cell = Domain::new(std::array::from_fn(|aa| Space {
                        name: lvl.state.geom.interior.spaces[aa].name,
                        lo: coord[aa],
                        hi: coord[aa] + 1,
                    }));
                    prolong_field(&cc.den, &zero, &fc.den, &cell, ProlongOrder::Pcm, 0.0);
                    for kk in 0..DOF {
                        prolong_field(
                            &cc.mom[kk],
                            &zero,
                            &fc.mom[kk],
                            &cell,
                            ProlongOrder::Pcm,
                            0.0,
                        );
                    }
                    if let (Some(cn), Some(fn_nrg)) = (cc.nrg_field(), fc.nrg_field()) {
                        prolong_field(cn, &zero, fn_nrg, &cell, ProlongOrder::Pcm, 0.0);
                    }
                }
                reseeded += flagged.len();
                // refresh the primitives and the error field; a cell still
                // inadmissible after parent injection is a defect of the coarse
                // state itself and is left for the caller's c2p gate to report.
                lvl.kernels.c2p(&lvl.state);
                symbi_substrate::regimes::substrate_gpu::device_sync::<Mem>();
                eprintln!(
                    "seed_fine_from_coarse: level {} re-seeded {} cell(s) by parent \
                     injection (high-order prolongation of independent conserved \
                     components broke E >= |m|^2/2rho)",
                    ll,
                    flagged.len()
                );
            }
        }
        Ok(reseeded)
    }

    /// a 1-level hierarchy: the degenerate case, which reproduces evolve()
    /// bit-for-bit.
    pub fn single(state: SimStateGeneric<R, NDIM, DOF, M, E, S, Mem>, kernels: K) -> Self {
        let tracer_root_global_cells =
            std::array::from_fn(|aa| state.geom.interior.spaces[aa].size());
        Hierarchy {
            levels: vec![LevelData {
                state,
                kernels,
                prim_old: None,
                prim_lerp: None,
                prolong_sweep: std::sync::OnceLock::new(),
                band_departure: std::sync::OnceLock::new(),
                bcell_old: None,
                bface_old: None,
                coverage: None,
                flux_eq: None,
                residual_eq: None,
                cons_eq: None,
                prim_eq: None,
            }],
            injection_ledger: std::collections::BTreeMap::new(),
            prolong_order: ProlongOrder::Plm,
            flux_registers: Vec::new(),
            emf_registers: Vec::new(),
            tracer_interfaces: Vec::new(),
            tracer_interface_ledgers: Vec::new(),
            tracer_root_global_cells,
            tracer_root_offset: [0; NDIM],
            crash: None,
            equilibrium_mask_radius: None,
            prev_dt_cfl: 0.0,
            balance_aware_transfer: None,
        }
    }

    /// build a statically nested hierarchy: region k refines level k at ratio 2.
    /// fine levels live in absolute indices (interior at 2x the covered cells)
    /// sharing the global physical origin; coarse-fine faces get
    /// BoundaryType::CoarseFine, faces flush with the parent interior inherit
    /// the parent's boundary. `make_kernels` builds each fine level's kernel
    /// set from its constructed state.
    pub fn with_refinement(
        coarse: SimStateGeneric<R, NDIM, DOF, M, E, S, Mem>,
        coarse_kernels: K,
        regions: &[RefinementRegion<NDIM>],
        prolong_order: ProlongOrder,
        make_kernels: impl Fn(&SimStateGeneric<R, NDIM, DOF, M, E, S, Mem>) -> K,
    ) -> symbi_xpu::Result<Self> {
        let root_ng = coarse.geom.ng;
        let mut levels = vec![LevelData {
            state: coarse,
            kernels: coarse_kernels,
            prim_old: None,
            prim_lerp: None,
            prolong_sweep: std::sync::OnceLock::new(),
            band_departure: std::sync::OnceLock::new(),
            bcell_old: None,
            bface_old: None,
            coverage: None,
            flux_eq: None,
            residual_eq: None,
            cons_eq: None,
            prim_eq: None,
        }];

        for region in regions {
            let parent = &levels.last().unwrap().state;
            if parent.fields.mhd.is_some() {
                assert!(
                    NDIM == 3,
                    "mhd refinement: the staggered CT substrate is 3d-only"
                );
                assert!(
                    parent.geom.coords == symbi_geometry::Geometry::Cartesian,
                    "mhd refinement: cartesian only (the emf-reflux curl coefficients are 1/dx)"
                );
            }
            let coverage = region_to_domain(region, &parent.geom.x_lo, &parent.geom.dx);
            // fine ghost depth: one above the root's (the prolongation stencil
            // is one order above the evolution reconstruction).
            let ng = root_ng + 1;
            validate_coverage(&coverage, parent, ng, prolong_order);

            let n_cells: [usize; NDIM] =
                std::array::from_fn(|ax| coverage.spaces[ax].size() * RATIO);
            let interior_lo: [isize; NDIM] =
                std::array::from_fn(|ax| coverage.spaces[ax].lo * RATIO as isize);
            let dx: [f64; NDIM] = std::array::from_fn(|ax| parent.geom.dx[ax] / RATIO as f64);

            let mut boundaries = Boundaries::<NDIM>::uniform(BoundaryType::CoarseFine);
            for ax in 0..NDIM {
                if coverage.spaces[ax].lo == parent.geom.interior.spaces[ax].lo {
                    boundaries.0[ax][0] = parent.boundaries.lo(ax);
                }
                if coverage.spaces[ax].hi == parent.geom.interior.spaces[ax].hi {
                    boundaries.0[ax][1] = parent.boundaries.hi(ax);
                }
            }

            let fine = SimStateGeneric::new_at(
                parent.physics.regime,
                parent.physics.eos,
                parent.physics.metric,
                interior_lo,
                n_cells,
                parent.geom.x_lo,
                dx,
                ng,
                boundaries,
                parent.cfl,
                parent.timestepping,
                0,
            )?;
            assert!(
                parent.geom.maps.is_none(),
                "refinement: non-uniform axis maps need a per-level map split (not built)"
            );
            // the passive scalar is a run-level opt-in taken on the root, and a fine level is
            // built fresh, so the dye slots have to be carried down explicitly. the slots are what
            // hold the concentration a fine level's own cells advect.
            let fine = if parent.fields.cons.chi_field().is_some() {
                fine.with_passive_scalar()?
            } else {
                fine
            };

            let kernels = make_kernels(&fine);
            let last = levels.last_mut().unwrap();
            last.prim_old = Some(PrimFieldsGeneric::zeros_with_pressure(
                &last.state.geom.allocated,
                last.state.fields.prim.pre_field().is_some(),
            )?);
            last.prim_lerp = Some(PrimFieldsGeneric::zeros_with_pressure(
                &last.state.geom.allocated,
                last.state.fields.prim.pre_field().is_some(),
            )?);
            // the dye concentration is time-interpolated onto the fine coarse-fine ghosts like the
            // rest of the primitive state, so the parent's step-start copy has to carry it.
            if last.state.fields.prim.chi_field().is_some() {
                let alloc = last.state.geom.allocated.clone();
                last.prim_old.as_mut().unwrap().alloc_chi(&alloc)?;
                last.prim_lerp.as_mut().unwrap().alloc_chi(&alloc)?;
            }
            if let Some(pmhd) = last.state.fields.mhd.as_ref() {
                last.bcell_old = Some(array_field_zeros(&last.state.geom.allocated)?);
                // per-component staggered domains (face axis + transverse halo).
                let mut bf: Vec<Field<f64, NDIM, Mem>> = Vec::with_capacity(NDIM);
                for dd in 0..NDIM {
                    bf.push(Field::zeros(pmhd.bface[dd].domain())?);
                }
                last.bface_old = Some(bf.try_into().unwrap_or_else(|_| unreachable!()));
            }
            last.coverage = Some(coverage);
            levels.push(LevelData {
                state: fine,
                kernels,
                prim_old: None,
                prim_lerp: None,
                prolong_sweep: std::sync::OnceLock::new(),
                band_departure: std::sync::OnceLock::new(),
                bcell_old: None,
                bface_old: None,
                coverage: None,
                flux_eq: None,
                residual_eq: None,
                cons_eq: None,
                prim_eq: None,
            });
        }

        let mut flux_registers = Vec::with_capacity(levels.len().saturating_sub(1));
        let mut emf_registers = Vec::with_capacity(levels.len().saturating_sub(1));
        let root_cells: [usize; NDIM] =
            std::array::from_fn(|aa| levels[0].state.geom.interior.spaces[aa].size());
        let mut tracer_interfaces = Vec::with_capacity(levels.len().saturating_sub(1));
        for level in &mut levels {
            let donor_width = (level.kernels.reconstruction_reach() as usize)
                .max(prolong_order.ghost_width())
                + 1;
            level.state.composite_ownership = symbi_sim::state::CompositeOwnership::new(
                level.coverage.clone(),
                &level.state.geom.interior,
                donor_width,
            );
        }
        for ll in 0..levels.len().saturating_sub(1) {
            flux_registers.push(FluxRegister::new(
                levels[ll].coverage.as_ref().unwrap(),
                &levels[ll].state.geom.interior,
                levels[ll].state.fields.cons.has_energy(),
                levels[ll].state.fields.cons.chi_field().is_some(),
            )?);
            emf_registers.push(if levels[ll].state.fields.mhd.is_some() {
                Some(EmfRegister::new(
                    levels[ll].coverage.as_ref().unwrap(),
                    &levels[ll].state.geom.interior,
                )?)
            } else {
                None
            });
            tracer_interfaces.push(interface_faces(
                levels[ll].coverage.as_ref().unwrap(),
                root_cells,
                ll as u8,
            ));
        }
        let tracer_interface_ledgers = (0..tracer_interfaces.len())
            .map(|_| std::cell::RefCell::new(Vec::new()))
            .collect();

        Ok(Hierarchy {
            levels,
            injection_ledger: std::collections::BTreeMap::new(),
            prolong_order,
            flux_registers,
            emf_registers,
            tracer_interfaces,
            tracer_interface_ledgers,
            tracer_root_global_cells: root_cells,
            tracer_root_offset: [0; NDIM],
            crash: None,
            equilibrium_mask_radius: None,
            prev_dt_cfl: 0.0,
            balance_aware_transfer: None,
        })
    }

    /// attach an immersed body collection to every level: the finest level
    /// carries the full collection (the sink and the accretion diagnostics
    /// have a single owner — the resolution truth), every coarser level a
    /// gravity-only proxy of each body (same mass / softening / motion, sink
    /// disabled), so the drain acts on the finest cells alone and the
    /// restriction then sets the covered coarse cells. body motion advances
    /// once per root step on the finest and is synced outward. the sink region
    /// lies inside the finest level — asserted every step as the bodies move.
    pub fn with_bodies(mut self, bodies: symbi_ib::BodyCollection<f64, NDIM>) -> Self {
        let n = self.levels.len();
        for ll in 0..n {
            let coll = if ll + 1 == n {
                bodies.clone()
            } else {
                gravity_only(&bodies)
            };
            self.levels[ll].state.attach_bodies(coll);
        }
        self.assert_sinks_inside_finest();
        self
    }

    /// exclude cells within `radius` of the origin from the stationarity diagnostic.
    ///
    /// overrides the default, which is the largest accretion radius among the attached bodies.
    /// pass `0.0` to consider every cell.
    pub fn with_equilibrium_mask(mut self, radius: f64) -> Self {
        self.equilibrium_mask_radius = Some(radius);
        self
    }

    /// force the coarse-fine transfer's equilibrium decomposition on or off, overriding the
    /// kernel-set default. `false` on a balanced gravitating hierarchy reproduces the seam
    /// entropy drain the decomposition removes — the measurement that identifies the transfer,
    /// over the reconstruction, as the load-bearing piece.
    pub fn balance_aware_transfer(mut self, on: bool) -> Self {
        self.balance_aware_transfer = Some(on);
        self
    }

    /// the radius the stationarity diagnostic ignores inside: the configured value, or the
    /// largest accretion radius among the bodies, or zero when neither exists.
    fn resolved_equilibrium_mask(&self) -> f64 {
        if let Some(r) = self.equilibrium_mask_radius {
            return r;
        }
        self.levels
            .last()
            .map(|level| {
                level
                    .state
                    .immersed
                    .as_ref()
                    .map(|im| {
                        im.bodies
                            .bodies()
                            .iter()
                            .filter_map(|b| b.accretion_radius())
                            .fold(0.0_f64, f64::max)
                    })
                    .unwrap_or(0.0)
            })
            .unwrap_or(0.0)
    }

    /// attach per-body immersed-boundary shapes to every level. rigid walls remain active on
    /// uncovered coarse cells when a body crosses a coarse-fine boundary, so every level must
    /// evaluate the same shape. `None` entries keep the analytic sphere.
    pub fn attach_body_shapes(&mut self, shapes: Vec<Option<symbi_ib::sdf::SdfExpr<f64, 3>>>) {
        for level in &mut self.levels {
            level.state.attach_body_shapes(shapes.clone());
        }
    }

    /// every accreting body's sink sphere must lie inside the finest level's
    /// interior — a sink straddling a coarse-fine boundary corrupts the mass
    /// accounting the refluxing protects.
    fn assert_sinks_inside_finest(&self) {
        // a 1-level hierarchy has no coarse-fine boundary for a sink to straddle.
        if self.levels.len() < 2 {
            return;
        }
        self.assert_sinks_inside_finest_clipped();
    }

    /// the sink containment invariant, clipped to this hierarchy's root: the part of the
    /// sink sphere overlapping the root interior lies inside the finest level — on a
    /// decomposed tile the sphere may span cuts (each owning tile drains its own cells),
    /// so the constraint binds exactly the tiles the sphere touches. a 1-level hierarchy
    /// is a single grid with no coarse-fine boundary to straddle; the decomposed driver
    /// separately forbids sphere overlap on an unrefined tile of a refined run (refluxing
    /// protects drains inside refined regions).
    pub fn assert_sinks_inside_finest_clipped(&self) {
        if self.levels.len() < 2 {
            return;
        }
        let root = &self.levels[0].state;
        let finest = &self.levels[self.levels.len() - 1].state;
        let Some(im) = finest.immersed.as_ref() else {
            return;
        };
        let rg = &root.geom;
        let fg = &finest.geom;
        im.bodies.visit_accretion(|body| {
            let racc: f64 = body.accretion_radius().unwrap_or(0.0);
            // sphere-box overlap against the root interior, per axis.
            let mut clip_lo = [0.0f64; NDIM];
            let mut clip_hi = [0.0f64; NDIM];
            let mut overlaps = true;
            for ax in 0..NDIM {
                let rlo = rg.x_lo[ax] + rg.interior.spaces[ax].lo as f64 * rg.dx[ax];
                let rhi = rg.x_lo[ax] + rg.interior.spaces[ax].hi as f64 * rg.dx[ax];
                let p: f64 = body.position[ax];
                clip_lo[ax] = (p - racc).max(rlo);
                clip_hi[ax] = (p + racc).min(rhi);
                overlaps &= clip_lo[ax] < clip_hi[ax];
            }
            if !overlaps {
                return;
            }
            for ax in 0..NDIM {
                let flo = fg.x_lo[ax] + fg.interior.spaces[ax].lo as f64 * fg.dx[ax];
                let fhi = fg.x_lo[ax] + fg.interior.spaces[ax].hi as f64 * fg.dx[ax];
                assert!(
                    clip_lo[ax] >= flo && clip_hi[ax] <= fhi,
                    "refinement x decomposition: body {} sink sphere clip [{:.4}, {:.4}] leaves                      this tile's finest level [{flo:.4}, {fhi:.4}] on axis {ax}",
                    body.idx,
                    clip_lo[ax],
                    clip_hi[ax]
                );
            }
        });
    }

    /// true when any accretion sphere overlaps this hierarchy's root interior — the
    /// decomposed driver's guard input: on a refined run, draining is confined to tiles
    /// carrying a fine patch, since reflux protection covers the fine regions alone.
    pub fn accretion_overlaps_root(&self) -> bool {
        let root = &self.levels[0].state;
        let Some(im) = root.immersed.as_ref() else {
            return false;
        };
        let rg = &root.geom;
        let mut hit = false;
        im.bodies.visit_accretion(|body| {
            let racc: f64 = body.accretion_radius().unwrap_or(0.0);
            let mut overlaps = true;
            for ax in 0..NDIM {
                let rlo = rg.x_lo[ax] + rg.interior.spaces[ax].lo as f64 * rg.dx[ax];
                let rhi = rg.x_lo[ax] + rg.interior.spaces[ax].hi as f64 * rg.dx[ax];
                let p: f64 = body.position[ax];
                overlaps &= (p - racc).max(rlo) < (p + racc).min(rhi);
            }
            hit |= overlaps;
        });
        hit
    }

    /// the local half of the decomposed body step: reduce this tile's finest-level
    /// backward feedback (force/torque/accreted mass receipts) into its diagnostics
    /// accumulator. the caller consolidates the partials across tiles and applies the
    /// identical global delta everywhere via `apply_global_body_deltas`.
    pub fn finest_body_feedback(&mut self, dt: f64) {
        if !self.levels[0].state.has_bodies() {
            return;
        }
        let fi = self.levels.len() - 1;
        let finest = &mut self.levels[fi];
        let needs_fb = finest
            .state
            .immersed
            .as_ref()
            .is_some_and(|im| im.bodies.needs_feedback());
        if needs_fb {
            prof("body_feedback", || {
                finest.kernels.body_feedback(&finest.state, dt)
            });
        }
    }

    /// drain this tile's finest-level body-delta partials (consolidated diagnostics),
    /// resetting the accumulator. empty when the tile carries no bodies.
    pub fn take_body_deltas(&mut self) -> Vec<symbi_ib::BodyDelta<f64, NDIM>> {
        let fi = self.levels.len() - 1;
        let Some(im) = self.levels[fi].state.immersed.as_mut() else {
            return Vec::new();
        };
        let deltas = im.diagnostics.consolidate();
        im.diagnostics.reset();
        deltas
    }

    /// the global half of the decomposed body step: apply the cross-tile-summed deltas +
    /// the prescribed orbit advance to this tile's finest bodies (identical input on every
    /// tile -> identical body state, the lockstep contract), record the global per-step
    /// exchange in the history, sync the advanced positions/velocities to the coarser
    /// levels, and re-check the clipped sink containment.
    pub fn apply_global_body_deltas(
        &mut self,
        deltas: &[symbi_ib::BodyDelta<f64, NDIM>],
        dt: f64,
        time: f64,
    ) {
        if !self.levels[0].state.has_bodies() {
            return;
        }
        let fi = self.levels.len() - 1;
        {
            let finest = &mut self.levels[fi].state;
            let im = finest.immersed.as_mut().unwrap();
            symbi_ib::apply_body_deltas(&mut im.bodies, deltas, dt);
            im.history.push(time, dt, deltas);
        }
        let truth: Vec<_> = {
            let bodies = &self.levels[fi].state.immersed.as_ref().unwrap().bodies;
            (0..bodies.len())
                .map(|bb| (bodies.get(bb).position, bodies.get(bb).velocity))
                .collect()
        };
        for ll in 0..fi {
            if let Some(im) = self.levels[ll].state.immersed.as_mut() {
                for (bb, (pos, vel)) in truth.iter().enumerate() {
                    let body = im.bodies.get_mut(bb);
                    body.position = *pos;
                    body.velocity = *vel;
                }
            }
        }
        self.assert_sinks_inside_finest_clipped();
    }

    /// march the hierarchy to t_final: the root's cfl sets dt, advance_level
    /// recurses through the levels, then the root advances its clock and body
    /// state (mirroring evolve_with_callback).
    pub fn evolve(&mut self, t_final: f64) -> symbi_xpu::Result<()> {
        self.evolve_with_callback(t_final, u64::MAX, |_| std::ops::ControlFlow::Continue(()))
    }

    /// evolve with a periodic observer: the callback fires every `interval`
    /// root iterations and once after the final step, with the device queue
    /// drained first so it may read fields from the host (the same contract
    /// as the single-level sim::evolve::evolve_with_callback).
    pub fn evolve_with_callback(
        &mut self,
        t_final: f64,
        interval: u64,
        callback: impl FnMut(&Self) -> std::ops::ControlFlow<()>,
    ) -> symbi_xpu::Result<()> {
        self.evolve_with_callback_impl(t_final, interval, true, callback)
    }

    /// continue a checkpoint-restored hierarchy whose primitive and staggered
    /// magnetic interiors represent the accepted step tail. physical ghost
    /// bands are reconstructed without re-running c2p or bcell recovery.
    pub fn resume_with_callback(
        &mut self,
        t_final: f64,
        interval: u64,
        callback: impl FnMut(&Self) -> std::ops::ControlFlow<()>,
    ) -> symbi_xpu::Result<()> {
        for level in &self.levels {
            level.kernels.ghost_fill(&level.state);
        }
        self.evolve_with_callback_impl(t_final, interval, false, callback)
    }

    fn evolve_with_callback_impl(
        &mut self,
        t_final: f64,
        interval: u64,
        prepare_initial_state: bool,
        mut callback: impl FnMut(&Self) -> std::ops::ControlFlow<()>,
    ) -> symbi_xpu::Result<()> {
        // homologous mesh motion applies on a single grid: the hierarchy's flux
        // registers and transfer operators are written for a fixed mesh. a
        // 1-level hierarchy runs without registers or transfer, so motion is
        // safe there (it reproduces the single-grid evolve); refined runs are
        // refused.
        assert!(
            self.levels.len() == 1
                || self
                    .levels
                    .iter()
                    .all(|l| { l.state.motion.a_dot == 0.0 && l.state.motion.a == 1.0 }),
            "refinement: mesh motion is uni-grid only (the registers carry no scale factor)"
        );
        if prepare_initial_state {
            self.init_levels();
        }
        // the uni-grid march is the one-tile case of the shared driver: no decomposition
        // context, so the exchange schedule is empty and the step runs the uni-grid
        // transaction. the driver owns the watchdog, the step-panic catch, and the
        // observer cadence for every shape.
        evolve_tiles::<R, NDIM, DOF, M, E, S, Mem, K, symbi_sim::decomp::LocalCopy, _>(
            std::slice::from_mut(self),
            None,
            t_final,
            interval,
            |ts| callback(&ts[0]),
        );
        Ok(())
    }

    /// run the one-time IC preparation at the current time: bcell-from-bface,
    /// c2p to populate the primitive buffer from the seeded conserved state, the
    /// coarse-fine prolong, and the ghost fill. the drivers call this internally
    /// at evolve start; a caller that snapshots state at t=0 (the binding's
    /// initial-condition checkpoint) must call it first, so that snapshot carries
    /// recovered primitives and cell-centered B rather than zeroed scratch
    /// buffers. idempotent.
    pub fn prime(&mut self) {
        self.init_levels();
    }

    /// advance exactly `nsteps` root steps (the smoke/diagnostic driver).
    pub fn evolve_steps(&mut self, nsteps: u64) -> symbi_xpu::Result<()> {
        self.init_levels();
        for _ in 0..nsteps {
            self.step_root(f64::INFINITY);
        }
        symbi_substrate::regimes::substrate_gpu::device_sync::<Mem>();
        Ok(())
    }

    /// initial setup. mhd pairs first restrict the staggered B downward
    /// (finest pair first) so the invariant "coarse face = average of its fine
    /// faces" holds from step 0 — the emf reflux preserves it thereafter, and
    /// divB exactness across the interface depends on it. then coarsest-first:
    /// c2p, coarse-fine prolong (alpha = 1: the current coarse prims; prim_old
    /// is still zeroed and contributes (1-1)*0), then the physical ghost fill,
    /// with the loud c2p check. idempotent.
    fn init_levels(&mut self) {
        for ll in (0..self.levels.len().saturating_sub(1)).rev() {
            if self.levels[ll].state.fields.mhd.is_none() {
                continue;
            }
            let (lo, hi) = self.levels.split_at(ll + 1);
            let coarse = &lo[ll];
            let fine = &hi[0];
            let cov = coarse.coverage.as_ref().unwrap();
            let cmhd = coarse.state.fields.mhd.as_ref().unwrap();
            let fmhd = fine.state.fields.mhd.as_ref().unwrap();
            for aa in 0..NDIM {
                restrict_cell_field(&fmhd.bcell[aa], &cmhd.bcell[aa], cov);
            }
            restrict_bface(&fmhd.bface, &cmhd.bface, cov);
            bcell_from_bface_region(cmhd, coarse.state.fields.cons.nrg_field(), cov);
        }
        for ll in 0..self.levels.len() {
            let lvl = &self.levels[ll];
            lvl.kernels.c2p(&lvl.state);
            if ll > 0 {
                self.prolong_cf(ll, 1.0);
            }
            let lvl = &self.levels[ll];
            lvl.kernels.ghost_fill(&lvl.state);
            // the error scan reads c2p_error from the host mid-queue.
            symbi_substrate::regimes::substrate_gpu::device_sync::<Mem>();
            let err = scan_c2p_errors(&lvl.state);
            if err.is_err() {
                panic!(
                    "hierarchy: c2p failed on initial conditions at level {} (time {:.4e}): {}",
                    ll, lvl.state.time, err
                );
            }
        }
    }

    /// restore every level from a checkpoint, adding levels beyond the file's own depth.
    ///
    /// this is the bootstrap ladder: a rung converges at its own resolution and the next rung
    /// resumes it with one more level, so the expensive early transient is paid once at the
    /// coarsest depth instead of once per depth. the levels the file carries are loaded; the levels
    /// beyond it are injected from their parents.
    ///
    /// the split between the two is taken from the file's own level count rather than the config:
    /// a run that guessed would either inject over converged data (replacing a level with a coarser
    /// copy of itself) or try to load a level absent from the file, and the first of those is
    /// silent.
    ///
    /// each loaded level's grid is verified against the file before its data is read. a deeper
    /// restart is only meaningful when level `i` occupies the same region regardless of the total
    /// depth, which is a property of a refinement schedule rather than of this code.
    pub fn restore_from_checkpoint(&mut self, path: &str) -> Result<usize, String> {
        let stored =
            symbi_sim::checkpoint::checkpoint_level_count(path).map_err(|e| format!("{e}"))?;
        if stored > self.levels.len() {
            return Err(format!(
                "checkpoint '{path}' carries {stored} refinement level(s) but this run builds {}. \
                 a restart may add levels to a checkpoint, never drop them: discarding a level \
                 means restricting its data onto its parent, which is a different operation and \
                 loses the resolution the checkpoint was run to obtain.",
                self.levels.len()
            ));
        }
        for (level_index, level) in self.levels.iter_mut().enumerate().take(stored) {
            symbi_sim::checkpoint::verify_checkpoint_level_geometry(
                &level.state,
                path,
                level_index,
            )
            .map_err(|e| format!("{e}"))?;
            symbi_sim::checkpoint::load_checkpoint_level(&mut level.state, path, level_index)
                .map_err(|e| format!("{e}"))?;
        }
        for level_index in stored..self.levels.len() {
            self.inject_level_from_parent(level_index)?;
            // the injected level inherits its parent's clock: it enters the run fresh and takes the
            // parent's elapsed history as its own. matching level times keep the subcycle
            // synchronized from the very first root step.
            let (time, dt, iteration, motion) = {
                let parent = &self.levels[level_index - 1].state;
                (parent.time, parent.dt, parent.iteration, parent.motion)
            };
            let fresh = &mut self.levels[level_index].state;
            fresh.time = time;
            fresh.dt = dt;
            fresh.iteration = iteration;
            fresh.motion = motion;
        }
        Ok(stored)
    }

    /// initialize `level` from its parent by piecewise-constant injection: every fine cell takes
    /// the conserved state of the coarse cell containing it.
    ///
    /// exactly conservative on cell averages — a coarse average replicated to its `RATIO^D`
    /// children preserves the integral over the coarse cell — so the new level enters the run
    /// carrying precisely the mass, momentum and energy its parent held over the same region.
    ///
    /// that exactness is the justification for injection over a higher-order
    /// prolongation. adding a level to a converged solution adds resolution, and the
    /// structure the finer cells can then represent is generated by the flow itself within a few
    /// crossing times of the new level's own (small) region, which is negligible against the time
    /// such a level is then run for. a smoother initial transient is all a higher-order operator
    /// would buy.
    ///
    /// refused for MHD: a face-centered field needs a divergence-preserving prolongation, and
    /// injecting it cell-wise seeds a `div B` that constrained transport then preserves for the
    /// rest of the run.
    pub fn inject_level_from_parent(&mut self, level: usize) -> Result<(), String> {
        if level == 0 || level >= self.levels.len() {
            return Err(format!(
                "level {level} cannot be injected from a parent: the hierarchy has {} level(s) and \
                 the root has none",
                self.levels.len()
            ));
        }
        if self.levels[level].state.fields.mhd.is_some() {
            return Err(format!(
                "level {level} carries a magnetic field: initializing it by cell-wise injection \
                 would seed a divergence the constrained transport cannot remove. a \
                 divergence-preserving prolongation of the face field is required."
            ));
        }

        let (coarser, finer) = self.levels.split_at_mut(level);
        let parent = &coarser[level - 1].state;
        let fine = &mut finer[0].state;

        // piecewise-constant prolongation is precisely parent injection: the stencil is the single
        // covering coarse cell, so a coarse average lands unchanged on each of its children and the
        // integral over the coarse cell is preserved exactly.
        //
        // routed through the same rendered kernel every other coarse-to-fine transfer uses, so a
        // device-resident hierarchy injects on the device. the source and destination buffers are
        // the same field, which collapses the time interpolation the coarse-fine ghost path needs:
        // the parent's state serves as both the old and the new level.
        let region = fine.geom.interior.clone();
        let inject = |src: &Field<f64, NDIM, Mem>, dst: &Field<f64, NDIM, Mem>| {
            crate::refinement::transfer::prolong_field(
                src,
                src,
                dst,
                &region,
                ProlongOrder::Pcm,
                0.0,
            );
        };

        inject(&parent.fields.cons.den, &fine.fields.cons.den);
        for dd in 0..DOF {
            inject(&parent.fields.cons.mom[dd], &fine.fields.cons.mom[dd]);
        }
        if let (Some(pn), Some(fnn)) =
            (parent.fields.cons.nrg_field(), fine.fields.cons.nrg_field())
        {
            inject(pn, fnn);
        }
        if let (Some(pc), Some(fc)) = (parent.fields.cons.chi_field(), fine.fields.cons.chi_field())
        {
            inject(pc, fc);
        }

        // the census, the diagnostics and the first flux evaluation all read primitives; injection
        // writes the conserved state alone, so the recovery has to run before the level is stepped.
        finer[0].kernels.c2p(&fine.store);
        Ok(())
    }

    /// the cfl-limited root dt this hierarchy would take, ahead of the t_final clamp: the min over
    /// every level of `cfl(level) * RATIO^level` (covered coarse cells are conservative averages,
    /// so a fast fine-only feature is diluted out of the root cfl; level l subcycles ratio^l times,
    /// so its limit enters scaled by ratio^l). exposed for the decomposed driver: it takes the
    /// global min of this across tiles, then drives each tile with `evolve(t + global_dt)` -- since
    /// the global dt is the min, each tile's internal `dt_cfl.min(global_dt)` collapses to
    /// global_dt, so the existing clamp alone produces a lockstep root step.
    /// the raw cfl estimate over every level: `cfl(level) * RATIO^level` (level l subcycles
    /// ratio^l times, so its limit enters scaled by ratio^l), minimized across levels. a
    /// non-finite or non-positive per-level candidate propagates out unmodified so the
    /// callers' crash heuristics see it.
    fn raw_root_cfl(&self) -> f64 {
        let mut dt = f64::INFINITY;
        for (ll, lvl) in self.levels.iter().enumerate() {
            let scale = RATIO.pow(ll as u32) as f64;
            let candidate = lvl.kernels.cfl(&lvl.state) * scale;
            if !candidate.is_finite() || candidate <= 0.0 {
                return candidate;
            }
            dt = dt.min(candidate);
        }
        dt
    }

    pub fn root_cfl_dt(&self) -> f64 {
        // same full-grid wave-speed pass as `step_root`; instrumented under the same phase name so a
        // decomposed run attributes it identically.
        let dt_cfl = prof("cfl", || self.raw_root_cfl());
        // the user clamp (max_dt > 0): pins the dt sequence across runs whose CFL
        // estimators differ. applied after the raw-cfl crash heuristics elsewhere.
        let clamp = self.levels[0].state.max_dt;
        if clamp > 0.0 {
            dt_cfl.min(clamp)
        } else {
            dt_cfl
        }
    }

    fn snapshot_retry_sidecars(&self) -> HierarchyRetrySidecars<NDIM> {
        HierarchyRetrySidecars {
            discrete: self
                .levels
                .iter()
                .map(|level| level.state.tracers.clone())
                .collect(),
            continuous: self
                .levels
                .iter()
                .map(|level| {
                    level
                        .state
                        .continuous_tracers
                        .as_ref()
                        .map(|tracers| tracers.snapshot_retry())
                        .transpose()
                        .unwrap_or_else(|detail| {
                            panic!("continuous tracer retry snapshot: {detail}")
                        })
                })
                .collect(),
            bodies: self
                .levels
                .iter()
                .map(|level| {
                    level
                        .state
                        .immersed
                        .as_ref()
                        .map(|immersed| immersed.bodies.clone())
                })
                .collect(),
            motion: self.levels.iter().map(|level| level.state.motion).collect(),
            clocks: self
                .levels
                .iter()
                .map(|level| (level.state.time, level.state.iteration, level.state.dt))
                .collect(),
            censuses: self
                .levels
                .iter()
                .map(|level| {
                    level
                        .state
                        .censuses
                        .iter()
                        .map(|registered| {
                            (registered.history.clone(), registered.last_sample.clone())
                        })
                        .collect()
                })
                .collect(),
        }
    }

    fn restore_retry_sidecars(&mut self, snapshot: &HierarchyRetrySidecars<NDIM>) {
        for (ii, level) in self.levels.iter_mut().enumerate() {
            level.state.tracers.clone_from(&snapshot.discrete[ii]);
            if let (Some(tracers), Some(saved)) = (
                level.state.continuous_tracers.as_mut(),
                snapshot.continuous[ii].as_ref(),
            ) {
                tracers
                    .restore_retry(saved)
                    .unwrap_or_else(|detail| panic!("continuous tracer retry restore: {detail}"));
            }
            if let (Some(immersed), Some(bodies)) =
                (level.state.immersed.as_mut(), snapshot.bodies[ii].as_ref())
            {
                immersed.bodies.clone_from(bodies);
            }
            level.state.motion = snapshot.motion[ii];
            level.state.time = snapshot.clocks[ii].0;
            level.state.iteration = snapshot.clocks[ii].1;
            level.state.dt = snapshot.clocks[ii].2;
            assert_eq!(
                level.state.censuses.len(),
                snapshot.censuses[ii].len(),
                "census registrations changed during a retryable step"
            );
            for (registered, (history, last_sample)) in
                level.state.censuses.iter_mut().zip(&snapshot.censuses[ii])
            {
                registered.history.clone_from(history);
                registered.last_sample.clone_from(last_sample);
            }
        }
        self.injection_ledger.clear();
        for ledger in &self.tracer_interface_ledgers {
            ledger.borrow_mut().clear();
        }
    }

    /// one root step: cfl-limited dt (clamped to t_final), the recursive level
    /// advance, then the root clock + body state. every level limits the root
    /// step — covered coarse cells are conservative averages of fine data, so
    /// a fast feature resolved only on the fine level is diluted out of the
    /// root's own cfl; level l subcycles ratio^l times, so its limit enters
    /// scaled by ratio^l.
    fn step_root(&mut self, t_final: f64) {
        // the coarse-fine invariant: the prolongation's smooth-data polynomial degree
        // must be at least the evolution reconstruction's stencil reach (= its degree
        // plus one: pcm evolution -> plm prolong, plm -> ppm, ppm -> quartic). the
        // prolonged ghost averages then carry error one order above the interior
        // truncation, so the interface layer's flux-divergence order loss leaves
        // the interior order intact. a shallower prolongation would degrade it
        // silently at every refinement boundary, so the pairing is asserted here.
        // a single-level hierarchy is one grid with no coarse-fine boundary, and
        // carries any pairing.
        if self.levels.len() > 1 {
            for lvl in &self.levels {
                assert!(
                    lvl.kernels.reconstruction_reach() <= self.prolong_order.degree(),
                    "evolution reconstruction reach {} exceeds the {:?} prolongation's \
                     degree {}: the coarse-fine ghost averages would carry a lower-order \
                     error than the interior truncation and the refinement boundary would \
                     degrade the interior order; pair this reconstruction with a \
                     higher-degree prolongation (ppm evolution -> quartic)",
                    lvl.kernels.reconstruction_reach(),
                    self.prolong_order,
                    self.prolong_order.degree(),
                );
            }
        }
        // the per-root-step wave-speed pass + global min reduction. instrumented because it is a
        // full-grid read of prim on every level, once per step, and sits outside the substage loop:
        // at a small domain / high step count it is a large fraction of the step that per-phase
        // timing inside the substage loop would leave unattributed.
        let Some(dt) = self.watchdog_root_dt(t_final) else {
            return;
        };
        self.step_root_with_dt(dt);
    }

    /// the watchdog-screened root timestep. the crash detection runs ahead of the `t_final`
    /// clamp: the clamp `dt_cfl.min(t_final - time)` silently replaces a NaN dt with the
    /// remaining time (f64::min returns the non-NaN operand) and pulls a collapsed-wave-speed
    /// blowup (an unphysical c2p cell — e.g. V->1 at the inner boundary — drives the cfl speed
    /// -> 0, so dt -> huge) down to the remaining time; either way the run would "finish" at
    /// t_final on garbage. a physical flow grows dt smoothly (cfl-limited), so a crash shows
    /// as: NaN / non-positive, or a sudden >1000x one-step jump in the raw cfl dt. a genuinely
    /// static state (dt_cfl = +inf) arises from the rest state at step 0 alone (dt_prev = 0,
    /// skipped) -> the clamp takes the run end. a fatal estimate records the crash and yields
    /// nothing, so the caller halts on the last computed state.
    fn watchdog_root_dt(&mut self, t_final: f64) -> Option<f64> {
        let dt_cfl = prof("cfl", || self.raw_root_cfl());
        let (iter, time) = {
            let r = &self.levels[0].state;
            (r.iteration, r.time)
        };
        // compare cfl estimate against cfl estimate. `state.dt` is the accepted dt, which both the
        // `t_final` clamp and a rejected-step replay shrink below the rate the wave speeds imply —
        // measuring a fresh estimate against it reports a collapse whenever a reduced step recovers.
        let dt_prev = self.prev_dt_cfl; // 0.0 before the first step
        let crashed =
            dt_cfl.is_nan() || dt_cfl <= 0.0 || (dt_prev > 0.0 && dt_cfl > 1.0e3 * dt_prev);
        if crashed {
            // record the crash and halt on this state: the evolve loop reports it and the driver
            // snapshots `.crashed.h5` from the last computed state and stops. the run therefore
            // terminates by report, in place of a panic or a march past t_final on garbage.
            self.crash = Some(CrashReport {
                iter,
                time,
                dt_cfl,
                dt_prev,
                panic: None,
            });
            return None;
        }
        // the guard's reference for the next step: the rate the wave speeds imply, before the
        // user clamp, the `t_final` clamp, or any rejection reduces it.
        self.prev_dt_cfl = dt_cfl;
        let root = &self.levels[0];
        let user_clamp = root.state.max_dt;
        let dt_cfl = if user_clamp > 0.0 {
            dt_cfl.min(user_clamp)
        } else {
            dt_cfl
        };
        let dt = dt_cfl.min(t_final - root.state.time);
        check_dt_or_panic(dt, root.state.iteration, root.state.time);
        Some(dt)
    }

    /// one root step at an already-selected dt: the retry-snapshot transaction, the stage
    /// recursion with its rejection replay, and the accepted-step tail (tracer migration,
    /// census, clock and motion, body feedback). callable with any admissible dt, which is
    /// what lets a multi-tile driver select a global minimum and drive each tile with it
    /// while the uni-grid path arrives through `step_root`'s watchdog.
    pub fn step_root_with_dt(&mut self, mut dt: f64) {
        let time = self.levels[0].state.time;
        // A rejection can originate in the second (or deeper) fine substep after an earlier
        // substep has spawned/migrated tracers or booked a horizon receipt.  Field snapshots alone
        // cannot invert those host-side commits, so the root attempt owns a complete transaction
        // snapshot of every mutable side-car.
        let retry_sidecars = self.snapshot_retry_sidecars();
        loop {
            for level in &self.levels {
                if level.kernels.fofc_active() {
                    level.kernels.snapshot_retry(&level.state);
                }
            }
            if !self.advance_level(0, dt, 0.0) {
                break;
            }
            for level in &mut self.levels {
                if level.kernels.fofc_active() {
                    level.kernels.restore_step(&level.state);
                }
            }
            self.restore_retry_sidecars(&retry_sidecars);
            dt = symbi_sim::driver::retry_timestep(dt, time)
                .unwrap_or_else(|err| panic!("{}", err.detail));
        }
        if self
            .levels
            .iter()
            .any(|level| level.state.continuous_tracers.is_some())
        {
            self.migrate_continuous_tracers_to_finest()
                .unwrap_or_else(|detail| panic!("continuous tracer refinement transfer: {detail}"));
        }

        // horizon excision runs once per step after the full RK combination, the same
        // point the single-grid loop applies it: the causally disconnected cells inside
        // the excision sphere take a zero-gradient fill. viscous / excise /
        // horizon-ledger / penalize run in the finest level's tail (level_step_tail),
        // in the uni-grid driver's exact order.

        self.census_sample_root_step();

        let root = &mut self.levels[0];
        // homologous linear advance, taken when the run leaves the motion law untraced.
        advance_state_clock(&mut root.state, dt);

        self.root_body_step(dt);
    }

    /// the registered binned reductions, at the tail of the accepted step: the stage sequence has
    /// finished and its last stage ended in a conserved-to-primitive recovery, so the primitives
    /// every census reads belong to the state at this time. sampling mid-stage would bin a
    /// partially advanced state.
    ///
    /// every level contributes its leaf cells: a cell a finer level resolves is excluded here and
    /// counted there, so the refined volume enters the reduction exactly once at the finest
    /// resolution that covers it. counting a covered coarse cell as well would inflate every
    /// extensive total by the refined volume — a wrong number that looks entirely reasonable,
    /// since it is smooth, positive, and of the right order.
    pub fn census_sample_root_step(&mut self) {
        if self.levels[0].state.censuses.is_empty() {
            return;
        }
        let ops: Vec<_> = self.levels[0]
            .state
            .censuses
            .iter()
            .map(|r| r.evaluator.spec().op())
            .collect();
        // the registrations live on the root, which owns the history; every level is reduced
        // against that one list. a level carrying its own (empty) list would contribute
        // nothing while its volume stayed excluded from the parent.
        let registrations = &self.levels[0].state.store.censuses;
        let root_step = Some(symbi_sim::census::Cadence::RootStep);
        let mut totals = symbi_substrate::census_sample::level_partials(
            &self.levels[0].state,
            registrations,
            0,
            root_step,
            self.levels[0].state.composite_ownership.coverage.as_ref(),
        );
        for level in 1..self.levels.len() {
            let partial = symbi_substrate::census_sample::level_partials(
                &self.levels[level].state,
                registrations,
                0,
                root_step,
                self.levels[level]
                    .state
                    .composite_ownership
                    .coverage
                    .as_ref(),
            );
            symbi_substrate::census_sample::combine_partials(&mut totals, partial, &ops);
        }
        let time = self.levels[0].state.time;
        symbi_substrate::census_sample::record_samples(&mut self.levels[0].state, time, totals);
    }

    /// body feedback + motion at the tail of an accepted root step: the finest level owns the
    /// sink and the diagnostics; the (prescribed) motion advances there once at the root dt,
    /// then positions and velocities sync outward so every level's gravity sees the same bodies.
    pub fn root_body_step(&mut self, dt: f64) {
        if !self.levels[0].state.has_bodies() {
            return;
        }
        let fi = self.levels.len() - 1;
        let finest = &mut self.levels[fi];
        // the backward feedback reduction (force/torque/accreted-mass) is only
        // needed for bodies whose dynamics consume it — two-way-coupled or
        // accreting. a one-way fixed gravitational mass skips the entire pass.
        let needs_fb = finest
            .state
            .immersed
            .as_ref()
            .is_some_and(|im| im.bodies.needs_feedback());
        if needs_fb {
            // the backward reduction sweeps the full domain (~11 outputs per
            // body: force/torque/mass/energy) — a real per-step cost, instrumented
            // so the profile accounts for it explicitly.
            prof("body_feedback", || {
                finest.kernels.body_feedback(&finest.state, dt)
            });
        }
        finest.state.dt = dt;
        prof("body_motion", || evolve_bodies(&mut finest.state));

        let truth: Vec<_> = {
            let bodies = &self.levels[fi].state.immersed.as_ref().unwrap().bodies;
            (0..bodies.len())
                .map(|bb| (bodies.get(bb).position, bodies.get(bb).velocity))
                .collect()
        };
        for ll in 0..fi {
            if let Some(im) = self.levels[ll].state.immersed.as_mut() {
                for (bb, (pos, vel)) in truth.iter().enumerate() {
                    let body = im.bodies.get_mut(bb);
                    body.position = *pos;
                    body.velocity = *vel;
                }
            }
        }
        self.assert_sinks_inside_finest();
    }

    /// advance one level by dt, then subcycle the finer level, restrict, and
    /// reflux. `alpha0` is this level's substep start as a fraction of the
    /// parent's step; 0.0 on the root. the coarse-fine ghosts are
    /// stage-correct in time: the prolong feeding stage k reconstructs at the
    /// shu-osher stage time `alpha0 + c_k / RATIO` (see
    /// stage_time_fractions), which restores second-order temporal coupling
    /// at the interface — substep-start-frozen ghosts measurably collapse the
    /// boundary to first order. the stage loop mirrors sim/evolve.rs::step, and a
    /// 1-level hierarchy reproduces it bit-for-bit.
    fn advance_level(&mut self, level: usize, dt: f64, alpha0: f64) -> bool {
        self.level_step_begin(level, dt);
        let n = self.levels[level].state.timestepping.stages().len();
        for ii in 0..n {
            if self.level_stage(level, ii, dt, alpha0) {
                return true;
            }
        }
        self.level_step_tail(level, dt, alpha0)
    }

    /// step prologue: snapshot this level's prims (for the finer level's time-interpolated ghost
    /// prolongation) + the rk u_n snapshot. `advance_level` calls begin / stage* / tail in order;
    /// splitting them out lets the decomposed root driver drive the root stages one at a time with
    /// a root halo exchange between stages (rk2-root requires the
    /// corrector to read each neighbor's stage-1 update, exactly like the single-level exchange).
    pub fn level_step_begin(&mut self, level: usize, dt: f64) {
        let has_finer = level + 1 < self.levels.len();
        if has_finer {
            prof("refine_save_prim", || save_prim_old(&self.levels[level]));
        }
        self.levels[level].state.dt = dt;
        if self.levels[level].state.continuous_tracers.is_some() {
            let geometry = self.levels[level]
                .state
                .geom
                .block_geometry(self.levels[level].state.physics.metric);
            symbi_sim::tracers::begin_ito_transport_store(
                &mut self.levels[level].state.store,
                &geometry,
            )
            .unwrap_or_else(|detail| panic!("ito transport initialization: {detail}"));
        }
        let stages = self.levels[level].state.timestepping.stages();
        if needs_step_snapshot(stages) {
            let l = &self.levels[level];
            prof("snapshot", || l.kernels.snapshot(&l.state));
        }
        if self.levels[level].state.has_tracers() {
            symbi_sim::tracers::snapshot_transport_state(&mut self.levels[level].state);
            self.injection_ledger.clear();
            if has_finer {
                self.tracer_interface_ledgers[level].borrow_mut().clear();
            }
        }
    }

    /// one SSP stage `ii` of level `level` -- the body of `advance_level`'s stage loop. `alpha0` is
    /// this level's substep start as a fraction of the parent step (0 on the root). pure extraction.
    pub fn level_stage(&mut self, level: usize, ii: usize, dt: f64, alpha0: f64) -> bool {
        // the phase sequence comes from the shared table (symbi-sim::stage); this
        // driver contributes two structural interleaves through the hook points —
        // flux-register sampling (on the high-order fluxes, before a fofc splice)
        // and the coarse-fine ghost re-prolongation. everything the hooks touch is
        // reached by shared reference (the registers and field writes go through
        // interior mutability), so the fold's borrow of the level state stays
        // compatible.
        let has_finer = level + 1 < self.levels.len();
        let has_coarser = level > 0;
        let stages = self.levels[level].state.timestepping.stages();
        let n = stages.len();
        let weights = flux_weights(stages);
        let stage_time = stage_time_fractions(stages);
        let (a0, ac) = stages[ii];
        let receipt_start = has_coarser
            .then(|| self.tracer_interface_ledgers[level - 1].borrow().len())
            .unwrap_or(0);

        let this = &*self;
        let l = &this.levels[level];
        let mut hook = |hp: HookPoint| match hp {
            HookPoint::AfterFlux => {
                prof("refine_flux_reg", || {
                    // when the run declares a stationary target, that same accumulation runs a
                    // second time on that target's flux with the weight negated, so the register
                    // holds the coarse-fine mismatch of `F(Q) - F(qt)` rather than of `F(Q)`. a
                    // state sitting on the target then accumulates zero, while the two grids'
                    // reconstructions of the target itself — which differ, and which the plain
                    // register would apply to the coarse cells as a force — cancel.
                    if has_finer {
                        let reg = &this.flux_registers[level];
                        if uniform_cartesian(&l.state) {
                            if ii == 0 {
                                reg.zero_uniform();
                            }
                            for dd in 0..NDIM {
                                reg.accumulate_coarse_uniform(
                                    &l.state.fields.flux,
                                    &l.state.geom.dx,
                                    dd,
                                    weights[ii] * dt,
                                );
                                if let Some(feq) = l.flux_eq.as_ref() {
                                    reg.accumulate_coarse_uniform(
                                        feq,
                                        &l.state.geom.dx,
                                        dd,
                                        -weights[ii] * dt,
                                    );
                                }
                            }
                        } else {
                            if ii == 0 {
                                reg.zero();
                            }
                            let geo = l.state.geom.block_geometry(l.state.physics.metric);
                            for dd in 0..NDIM {
                                reg.accumulate_coarse(
                                    &l.state.fields.flux,
                                    &geo,
                                    dd,
                                    weights[ii] * dt,
                                );
                                if let Some(feq) = l.flux_eq.as_ref() {
                                    reg.accumulate_coarse(feq, &geo, dd, -weights[ii] * dt);
                                }
                            }
                        }
                    }
                    if has_coarser {
                        let reg = &this.flux_registers[level - 1];
                        if uniform_cartesian(&l.state) {
                            for dd in 0..NDIM {
                                reg.accumulate_fine_uniform(
                                    &l.state.fields.flux,
                                    &l.state.geom.dx,
                                    dd,
                                    weights[ii] * dt,
                                );
                                if let Some(feq) = l.flux_eq.as_ref() {
                                    reg.accumulate_fine_uniform(
                                        feq,
                                        &l.state.geom.dx,
                                        dd,
                                        -weights[ii] * dt,
                                    );
                                }
                            }
                        } else {
                            let geo = l.state.geom.block_geometry(l.state.physics.metric);
                            for dd in 0..NDIM {
                                reg.accumulate_fine(
                                    &l.state.fields.flux,
                                    &geo,
                                    dd,
                                    weights[ii] * dt,
                                    RATIO,
                                );
                                if let Some(feq) = l.flux_eq.as_ref() {
                                    reg.accumulate_fine(feq, &geo, dd, -weights[ii] * dt, RATIO);
                                }
                            }
                        }
                        if l.state.has_tracers() {
                            symbi_substrate::regimes::substrate_gpu::device_sync::<Mem>();
                            let geometry = l.state.geom.block_geometry(l.state.physics.metric);
                            let transfers = interface_mass_transfers(
                                &this.tracer_interfaces[level - 1],
                                &l.state.fields.flux,
                                &geometry,
                                weights[ii] * dt,
                            );
                            this.tracer_interface_ledgers[level - 1]
                                .borrow_mut()
                                .extend(transfers);
                        }
                    }
                });
            }
            HookPoint::BeforeC2p => {
                // only the covered shell next to a coarse-fine boundary participates in the
                // composite update: it supplies stage-time prolongation data and the coarse face
                // flux sampled by the register. the deep covered core is replaced by restriction
                // after the fine subcycle, so evolving it is both unnecessary and, for a compact
                // source resolved only on the fine grids, mathematically undefined. in particular,
                // a point-sampled softened force can drive the central cell of an under-resolved
                // covered level outside the admissible set before restriction gets a chance to
                // overwrite it.
                //
                // preserve one cell beyond both stencils at the seam. restoring the remainder
                // from this stage's input makes that inactive volume an identity update while
                // leaving every coarse value observable by reconstruction or prolongation alone.
                if let Some(inactive) = l.state.composite_ownership.inactive.as_ref() {
                    restore_cons_region(l.state.stage_input(), &l.state.fields.cons, inactive);
                    // CT fields are part of the state, not diagnostics.  A covered-core identity
                    // update must therefore restore both representations of B as well as U.
                    // `*_old` is the level-step entry field; after every earlier stage the same
                    // core was restored to it, so it is also this stage's core input.
                    if let (Some(mhd), Some(bcell_old)) =
                        (l.state.fields.mhd.as_ref(), l.bcell_old.as_ref())
                    {
                        for comp in 0..DOF {
                            copy_field_region(&bcell_old[comp], &mhd.bcell[comp], inactive);
                        }
                    }
                    if let (Some(mhd), Some(bface_old)) =
                        (l.state.fields.mhd.as_ref(), l.bface_old.as_ref())
                    {
                        for dir in 0..NDIM {
                            let face_core = inactive.extend(dir, 0, 1);
                            copy_field_region(&bface_old[dir], &mhd.bface[dir], &face_core);
                        }
                    }
                }
                // the ssp recombination leaves `a0*u_n + ac*(qt - dt R) = qt - ac dt R` when the
                // stage input is the target, so adding `ac dt R` back returns the target exactly.
                // the weights are the stage's own, so this holds at every stage of every scheme.
                if let Some(res) = l.residual_eq.as_ref() {
                    prof("refine_equilibrium", || {
                        equilibrium_accumulate(
                            res,
                            &l.state.fields.cons,
                            &l.state.geom.interior,
                            ac * dt,
                        );
                    });
                }
            }
            HookPoint::BeforeGhostFill => {
                // c2p over the full allocated domain recomputed the coarse-fine
                // prim ghosts from stale cons; re-prolong them at the time of
                // the state entering the next stage before the physical fill
                // reads corners.
                if has_coarser {
                    this.prolong_cf(level, alpha0 + stage_time[ii] / RATIO as f64);
                }
            }
        };
        let outcome = fold_stage(
            &l.state,
            &l.kernels,
            StageArgs {
                dt,
                a0,
                ac,
                stage: ii,
                n_stages: n,
                allow_elision: true,
            },
            &mut hook,
        );
        if outcome == symbi_sim::stage::StageOutcome::RetryStep {
            return true;
        }
        if self.levels[level].state.continuous_tracers.is_some() {
            symbi_substrate::regimes::substrate_gpu::device_sync::<Mem>();
            let geometry = self.levels[level]
                .state
                .geom
                .block_geometry(self.levels[level].state.physics.metric);
            symbi_sim::tracers::accumulate_ito_transport_stage_store(
                &mut self.levels[level].state.store,
                &geometry,
                ac,
            )
            .unwrap_or_else(|detail| panic!("ito transport accumulation: {detail}"));
        }
        if self.levels[level].state.has_tracers() {
            symbi_substrate::regimes::substrate_gpu::device_sync::<Mem>();
            let layout = self.tracer_layout(level);
            let geometry = self.levels[level]
                .state
                .geom
                .block_geometry(self.levels[level].state.physics.metric);
            let mut injections = symbi_sim::tracers::boundary_injection_transfers_store(
                &self.levels[level].state,
                &geometry,
                layout,
            );
            injections.extend(symbi_sim::tracers::source_injection_transfers_store(
                &self.levels[level].state,
                &geometry,
                layout,
                a0,
                ac,
            ));
            injections.retain(|transfer| self.tracer_owner_is_active(transfer.destination));
            symbi_sim::tracers::fold_injection_ledger(&mut self.injection_ledger, injections, ac);
            prof("tracers", || {
                let coverage = self.levels[level].coverage.clone();
                symbi_sim::tracers::advance_stage_mass_transport_store_masked(
                    &mut self.levels[level].state,
                    &geometry,
                    layout,
                    coverage.as_ref(),
                    a0,
                    ac,
                    ii,
                )
                .unwrap_or_else(|detail| panic!("tracer transport: {detail}"))
            });
        }
        if has_coarser && self.levels[level].state.has_tracers() {
            let transfers = {
                let ledger = self.tracer_interface_ledgers[level - 1].borrow();
                ledger[receipt_start..].to_vec()
            };
            let epoch = self.levels[level]
                .state
                .iteration
                .wrapping_mul(64)
                .wrapping_add((level as u64) << 4)
                .wrapping_add(ii as u64);
            self.apply_interface_event(&transfers, epoch)
                .unwrap_or_else(|detail| panic!("tracer interface transport: {detail}"));
        }
        false
    }

    /// step epilogue: emf bookkeeping (mhd) + the finer-level subcycle + restrict + reflux + the
    /// level clock. pure extraction from `advance_level`. for the decomposed root the driver calls
    /// this after the (exchanged) root stages; the fine subcycle here is tile-local (the patch lives
    /// inside one tile), so it reuses the recursive `advance_level` on the finer level unchanged.
    pub fn level_step_tail(&mut self, level: usize, dt: f64, alpha0: f64) -> bool {
        let has_finer = level + 1 < self.levels.len();
        if self.levels[level].state.has_tracers() {
            let ledger = std::mem::take(&mut self.injection_ledger);
            let layout = self.tracer_layout(level);
            let geometry = self.levels[level]
                .state
                .geom
                .block_geometry(self.levels[level].state.physics.metric);
            symbi_sim::tracers::spawn_boundary_injection_store(
                &mut self.levels[level].state.store,
                &geometry,
                layout,
                ledger,
            )
            .unwrap_or_else(|detail| panic!("tracer boundary injection: {detail}"));
            self.sync_tracer_spawn_state(level);
        }
        self.level_tail_emf(level, dt);
        // the IBM surface physics on the finest level, once per its substep,
        // after the full RK combination (receipt == removal) and ahead of the
        // parent's restriction (so the coarse covered cells sync to the
        // drained state — restriction consistency).
        // per-step operator order matches the uni-grid driver exactly:
        // viscous -> excise -> horizon ledger -> penalize. transport first, then the
        // causally-disconnected fill, then the shell-flux booking (the flux fields still hold the
        // last stage's flux), then the IBM surface physics whose receipt must equal the removal.
        if !has_finer {
            let l = &self.levels[level];
            prof("viscous", || l.kernels.viscous(&l.state, dt));
        }
        // excise runs on every level: the excised (causally-disconnected) region is owned by
        // whichever level contains it, and the refinement request gate keeps every finer patch clear
        // of it, so a refined root still owns — and still excises — its singular core. gating this
        // on `!has_finer` silently skipped excision wherever the excised region sat under a refined
        // level (the root of an off-core refined run), leaving the core evolving. ordered after
        // viscous (bit-identical on the finest, where both run); a no-op at excision_radius = 0, so
        // unexcised and uni-grid runs are unchanged. the excised fill survives the finer subcycle +
        // restriction below because the excised region stays outside every covered (finer-patch)
        // region.
        self.level_tail_excise(level);
        if !has_finer {
            if let Some((index, diagnostic_radius)) = horizon_request(&self.levels[level].state) {
                let l = &self.levels[level];
                let (mdot, edot) = prof("horizon_accretion", || {
                    l.kernels.horizon_accretion(&l.state, diagnostic_radius)
                });
                book_horizon_receipt(&mut self.levels[level].state, index, mdot, edot, dt);
            }
        }
        // rigid surfaces act on every level: a wall may cross a coarse-fine boundary, and its
        // uncovered segment belongs to the coarse level. covered coarse writes are replaced by
        // restriction after the fine subcycle. accretion remains finest-only because coarse
        // proxies have their sink capability removed.
        let l = &self.levels[level];
        let has_rigid = l
            .state
            .immersed
            .as_ref()
            .is_some_and(|immersed| immersed.bodies.rigid_count() > 0);
        if l.state.has_bodies() && (!has_finer || has_rigid) {
            let accretion_density = if l.state.has_tracers() {
                symbi_substrate::regimes::substrate_gpu::device_sync::<Mem>();
                Some(symbi_sim::tracers::snapshot_accretion_density(&l.state))
            } else {
                None
            };
            prof("penalize", || l.kernels.penalize(&l.state, dt));
            if let Some(density_before) = accretion_density.as_deref() {
                symbi_substrate::regimes::substrate_gpu::device_sync::<Mem>();
                let layout = self.tracer_layout(level);
                let geometry = self.levels[level]
                    .state
                    .geom
                    .block_geometry(self.levels[level].state.physics.metric);
                let crossing_time = self.levels[level].state.time + dt;
                symbi_sim::tracers::advance_accretion_transport_store(
                    &mut self.levels[level].state.store,
                    &geometry,
                    layout,
                    density_before,
                    crossing_time,
                )
                .unwrap_or_else(|detail| panic!("tracer accretion transport: {detail}"));
                symbi_sim::tracers::advance_continuous_accretion_transport_store(
                    &mut self.levels[level].state.store,
                    &geometry,
                    layout,
                    density_before,
                    crossing_time,
                )
                .unwrap_or_else(|detail| panic!("continuous tracer accretion transport: {detail}"));
            }
        }
        if self.levels[level].state.continuous_tracers.is_some() {
            self.prepare_level_ito_coefficients(level);
        }
        if has_finer {
            if self.level_subcycle(level, dt) {
                return true;
            }
            self.level_restrict_reflux(level, alpha0);
        }
        if self.levels[level].state.continuous_tracers.is_some() {
            self.advance_level_continuous_tracers(level, dt);
        }
        self.sample_per_level_censuses(level);
        self.level_clock(level, dt);
        false
    }

    /// record a per-level census sample from this level's own accepted subcycle step.
    ///
    /// levels are time-aligned only at root-step boundaries, so this row is this level's alone: it
    /// covers this level's leaf cells at this level's clock, and carries the level index so a
    /// consumer can tell it from a root row or from another level's. summing rows across levels is
    /// therefore left to the consumer.
    ///
    /// the registrations live on the root, which owns the history; a finer level defers to it.
    fn sample_per_level_censuses(&mut self, level: usize) {
        if self.levels[0].state.store.censuses.is_empty() {
            return;
        }
        let time = self.levels[level].state.time;
        let partial = symbi_substrate::census_sample::level_partials(
            &self.levels[level].state,
            &self.levels[0].state.store.censuses,
            level,
            Some(symbi_sim::census::Cadence::PerLevelStep),
            self.levels[level]
                .state
                .composite_ownership
                .coverage
                .as_ref(),
        );
        if partial.iter().all(|(values, _)| values.is_empty()) {
            return;
        }
        symbi_substrate::census_sample::record_samples_at_level(
            &mut self.levels[0].state,
            time,
            level,
            partial,
        );
    }

    /// derive one level's accepted-step coefficients before its finer level
    /// subcycles so fine coarse/fine ghosts can sample the parent field.
    fn prepare_level_ito_coefficients(&mut self, level: usize) {
        let geometry = self.levels[level]
            .state
            .geom
            .block_geometry(self.levels[level].state.physics.metric);
        symbi_sim::tracers::materialize_ito_coefficients_store(
            &mut self.levels[level].state.store,
            &geometry,
        )
        .unwrap_or_else(|detail| panic!("ito coefficient materialization: {detail}"));
        symbi_sim::tracers::fill_ito_coefficient_boundaries_host(
            self.levels[level]
                .state
                .ito_coefficients
                .as_ref()
                .expect("ito coefficients were materialized"),
            &self.levels[level].state.geom,
            self.levels[level].state.boundaries,
        )
        .unwrap_or_else(|detail| panic!("ito coefficient boundaries: {detail}"));
        if level > 0 {
            self.prolong_ito_cf(level);
        }
    }

    /// prolong accepted parent coefficients into a fine level's coarse/fine
    /// ghost slabs. the coefficients are fixed over the enclosing parent step.
    fn prolong_ito_cf(&self, level: usize) {
        let parent = &self.levels[level - 1];
        let fine = &self.levels[level];
        let parent_coefficients = parent
            .state
            .ito_coefficients
            .as_ref()
            .expect("the parent coefficients precede fine subcycling");
        let fine_coefficients = fine
            .state
            .ito_coefficients
            .as_ref()
            .expect("fine coefficients were materialized");
        let zero = Field::<f64, NDIM, Mem>::zeros(&parent.state.geom.allocated)
            .expect("ito coefficient prolongation scratch allocation failed");
        for slab in cf_ghost_slabs(
            &fine.state.geom.allocated,
            &fine.state.geom.interior,
            &fine.state.boundaries,
        ) {
            for dd in 0..NDIM {
                prolong_field(
                    &parent_coefficients.drift[dd],
                    &zero,
                    &fine_coefficients.drift[dd],
                    &slab,
                    self.prolong_order,
                    0.0,
                );
                prolong_field(
                    &parent_coefficients.variance[dd],
                    &zero,
                    &fine_coefficients.variance[dd],
                    &slab,
                    self.prolong_order,
                    0.0,
                );
                prolong_field(
                    &parent_coefficients.third[dd],
                    &zero,
                    &fine_coefficients.third[dd],
                    &slab,
                    self.prolong_order,
                    0.0,
                );
            }
        }
    }

    /// advance particles owned by one level over exactly that level's accepted
    /// interval. hierarchy ownership remains frozen until the root step ends.
    fn advance_level_continuous_tracers(&mut self, level: usize, dt: f64) {
        let mut tracers = self.levels[level]
            .state
            .continuous_tracers
            .take()
            .expect("continuous tracers remain attached through the level step");
        let (scale_start, scale_end, offset_start, offset_end) =
            symbi_sim::tracers::continuous_tracer_mesh_step(&self.levels[level].state.store, dt);
        symbi_sim::tracers::advance_continuous_tracers(
            &mut tracers,
            self.levels[level]
                .state
                .ito_coefficients
                .as_ref()
                .expect("ito coefficients were materialized"),
            &self.levels[level].state.geom,
            scale_start,
            scale_end,
            offset_start,
            offset_end,
            dt,
        )
        .unwrap_or_else(|detail| panic!("ito tracer advancement: {detail}"));
        let root_bounds = symbi_sim::tracers::map_continuous_tracer_bounds(
            symbi_sim::tracers::partition_physical_bounds(&self.levels[0].state.geom),
            scale_end,
            offset_end,
        );
        symbi_sim::tracers::apply_continuous_boundaries_host(
            &mut tracers,
            root_bounds,
            self.levels[0].state.boundaries,
        )
        .unwrap_or_else(|detail| panic!("ito tracer boundaries: {detail}"));
        self.levels[level].state.continuous_tracers = Some(tracers);
    }

    /// emf register bookkeeping (mhd) after the stage loop: the efield buffers hold the effective
    /// per-step EMF (post_godunov wrote the rk2 time-average in place before the single curl; euler
    /// keeps the raw stage EMF) -- sample it into the coarse-side register (a finer level exists)
    /// and the fine-side register (a coarser level exists). a no-op for hydro. a pure extraction
    /// from level_step_tail so the decomposed driver can interpose a fine-level halo exchange
    /// between the root stages and the (driver-controlled) fine subcycle.
    /// the excise pass for one level, as a pure extraction from `level_step_tail` — the same reason
    /// `level_tail_emf` is extracted: the decomposed driver builds the step itself and runs this
    /// piece of the tail on its own, apart from the finest-level pieces (viscous, horizon ledger,
    /// penalize) that belong to the finest level alone.
    ///
    /// it runs on every level: the excised region is owned by whichever level contains it, and the
    /// refinement request gate keeps every finer patch clear of it, so a refined root still owns
    /// and still excises its singular core. a decomposed driver that omitted this would evolve
    /// un-excised gas inside the horizon forever — silent, since the interior is causally
    /// disconnected and reports nothing.
    ///
    /// the halo survives it: the fill writes cells inside the excised surface, and the refinement
    /// gate keeps every fine patch off that surface, so the prolongation that follows reads
    /// live cells only.
    pub fn level_tail_excise(&self, level: usize) {
        let l = &self.levels[level];
        l.kernels.excise(&l.state);
    }

    pub fn level_tail_emf(&self, level: usize, dt: f64) {
        let has_finer = level + 1 < self.levels.len();
        let has_coarser = level > 0;
        let is_mhd = self.levels[level].state.fields.mhd.is_some();
        if has_finer && is_mhd {
            prof("refine_emf", || {
                let reg = self.emf_registers[level]
                    .as_ref()
                    .expect("mhd pair carries an emf register");
                reg.zero();
                let l = &self.levels[level];
                reg.accumulate_coarse(&l.state.fields.mhd.as_ref().unwrap().efield, dt);
            });
        }
        if has_coarser && is_mhd {
            prof("refine_emf", || {
                let reg = self.emf_registers[level - 1]
                    .as_ref()
                    .expect("mhd pair carries an emf register");
                let l = &self.levels[level];
                reg.accumulate_fine(&l.state.fields.mhd.as_ref().unwrap().efield, dt);
            });
        }
    }

    /// the monolithic finer-level subcycle: ratio substeps of level `level+1`, each prolonged from
    /// this level's time-interpolated prims. the decomposed driver replicates this loop but drives
    /// each fine substep's stages with a fine-level halo exchange between them (the fine patch may
    /// span a tile cut), so this method is the single-tile reference. pure extraction.
    fn level_subcycle(&mut self, level: usize, dt: f64) -> bool {
        let fine_dt = dt / RATIO as f64;
        for sub in 0..RATIO {
            let alpha = sub as f64 / RATIO as f64;
            self.prolong_cf(level + 1, alpha);
            let f = &self.levels[level + 1];
            prof("ghost_fill", || f.kernels.ghost_fill(&f.state));
            if self.advance_level(level + 1, fine_dt, alpha) {
                return true;
            }
        }
        false
    }

    /// restrict the finer level into this level's coverage + apply the flux/emf reflux + re-derive
    /// prim (and, if this level has a coarser parent, re-prolong its coarse-fine ghosts). runs after
    /// the finer-level subcycle. the flux/emf registers are tile-local (a coarse cell + the fine
    /// cells at its face are co-located), so the decomposed driver calls this per tile unchanged.
    /// a pure extraction.
    pub fn level_restrict_reflux(&mut self, level: usize, alpha0: f64) {
        let has_coarser = level > 0;
        let is_mhd = self.levels[level].state.fields.mhd.is_some();
        prof("refine_restrict", || {
            let (lo, hi) = self.levels.split_at(level + 1);
            let coarse = &lo[level];
            let fine = &hi[0];
            let cov = coarse.coverage.as_ref().unwrap();
            restrict_cons(&fine.state.fields.cons, &coarse.state.fields.cons, cov);
            // mhd: restrict the staggered B (interface faces included),
            // reflux the outside faces from the edge-EMF mismatch, then
            // re-derive cell B + the magnetic-energy correction over the
            // coverage plus the one-cell shell whose faces just changed.
            if is_mhd {
                let cmhd = coarse.state.fields.mhd.as_ref().unwrap();
                let fmhd = fine.state.fields.mhd.as_ref().unwrap();
                for aa in 0..NDIM {
                    restrict_cell_field(&fmhd.bcell[aa], &cmhd.bcell[aa], cov);
                }
                restrict_bface(&fmhd.bface, &cmhd.bface, cov);
                let inv_dx: [f64; NDIM] = std::array::from_fn(|ax| 1.0 / coarse.state.geom.dx[ax]);
                self.emf_registers[level]
                    .as_ref()
                    .unwrap()
                    .apply(&cmhd.bface, &inv_dx);
                let shell = shell_around(cov, &coarse.state.geom.interior);
                bcell_from_bface_region(cmhd, coarse.state.fields.cons.nrg_field(), &shell);
            }
        });
        prof("refine_reflux_apply", || {
            let l = &self.levels[level];
            if uniform_cartesian(&l.state) {
                self.flux_registers[level].apply_uniform(&l.state.fields.cons, &l.state.geom.dx);
            } else {
                let geo = l.state.geom.block_geometry(l.state.physics.metric);
                self.flux_registers[level].apply(&l.state.fields.cons, &geo);
            }
        });
        let l = &self.levels[level];
        prof("c2p", || l.kernels.c2p(&l.state));
        if self.levels[level].cons_eq.is_none() && self.cf_transfer_balanced(level + 1) {
            // SYMBI_WB_BAND=0 withholds the band rewrite so the seam carries the
            // conservative restriction alone; the identical binary then runs the
            // counterfactual arm of the seam admissibility census, attributing
            // decode abstentions between the band's anchor rewrite and the
            // turbulent density contrast.
            if wb_band_enabled() {
                prof("refine_restrict_balanced", || {
                    self.restrict_band_balanced(level)
                });
            }
        }
        if has_coarser {
            self.prolong_cf(level, alpha0 + 1.0 / RATIO as f64);
        }
        let l = &self.levels[level];
        prof("ghost_fill", || l.kernels.ghost_fill(&l.state));
    }

    /// the balanced restriction of level `level`'s seam bands from level `level + 1`:
    /// the covered coarse cells within the evolution reach of each coarse-fine seam
    /// are rewritten onto the coarse mechanical chain from the uncovered cell beyond
    /// the seam, carrying the fine departure. runs after the conservative restriction
    /// and the coarse c2p, and rebuilds the band's conserved energy, which requires
    /// the gamma-law closure the kernel set reports.
    fn restrict_band_balanced(&self, level: usize) {
        let coarse = &self.levels[level];
        let fine = &self.levels[level + 1];
        let gamma = coarse.kernels.gamma_law().unwrap_or_else(|| {
            panic!(
                "the balanced coarse-fine restriction rebuilds the seam band's energy under the \
                 gamma law; level {level}'s kernel set reports another energy closure"
            )
        });
        let departure = fine.band_departure.get_or_init(|| {
            Field::zeros(&fine.state.geom.allocated).expect("band departure scratch")
        });
        // the coarse level's own scratch holds the restricted departures; its other
        // role -- the fine encode target of the seam one level up -- writes the
        // interior edge strips, disjoint from the coverage strips written here.
        let coarse_departure = coarse.band_departure.get_or_init(|| {
            Field::zeros(&coarse.state.geom.allocated).expect("band departure scratch")
        });
        restrict_band_balanced(
            &fine.state.fields.prim,
            &fine.state.geom.interior,
            departure,
            coarse_departure,
            &coarse.state.fields.prim,
            coarse
                .state
                .fields
                .cons
                .nrg_field()
                .expect("the balanced restriction needs cons.nrg"),
            &coarse.state.geom.interior,
            coarse.coverage.as_ref().unwrap(),
            // the uncovered cell's update over one step depends on cells within
            // `reach` per stage; the band holds the cells its own stencils read
            // and two more, each of which removes a limiter-slope hop from the
            // path the conservatively restricted interior's class offset takes
            // to reach it (measured ~10x attenuation per cell on the sealed
            // class column). capped by the decode's unrolled chain.
            (coarse.kernels.reconstruction_reach() as usize + 2)
                .min(symbi_discretize::WB_CF_CHAIN_MAX as usize),
            gamma,
            &fine.state.geom.x_lo,
            &fine.state.geom.dx,
            &coarse.state.geom.x_lo,
            &coarse.state.geom.dx,
            &coarse.state.immersed.as_ref().unwrap().bodies,
        );
    }

    /// advance this level's clock (fine levels only; the root clock is the driver's). pure extraction.
    fn level_clock(&mut self, level: usize, dt: f64) {
        if level > 0 {
            let s = &mut self.levels[level].state;
            (s.time, s.iteration) = advance_clock(s.time, s.iteration, dt);
        }
    }

    /// whether level `level`'s coarse-fine ghost transfer rides the hydrostatic
    /// equilibrium decomposition: the explicit override when set, otherwise the level's
    /// own kernel set. active only where the decomposition is defined —
    /// a gravitating body supplies the potential and the prim set carries pressure
    /// (gamma-law; asserted at the transfer). the encode/decode are baked kernels,
    /// so host and device hierarchies take the same path.
    fn cf_transfer_balanced(&self, level: usize) -> bool {
        // SYMBI_WB_GHOST=0 withholds the balanced ghost transfer alone: the seam
        // ghosts take the plain prolongation while the in-level balanced flux,
        // the balanced body source, and the restriction band keep their own
        // switches. the ghost decode anchors on the fine interior, so a run
        // with the switch off carries seam boundary data built from the coarse
        // level only, which is the single-variable arm for attributing a seam
        // failure to that anchoring.
        if !wb_ghost_enabled() {
            return false;
        }
        let want = self
            .balance_aware_transfer
            .unwrap_or_else(|| self.levels[level].kernels.hydrostatic_balance());
        if !want {
            return false;
        }
        let fine = &self.levels[level];
        let has_gravity = fine
            .state
            .immersed
            .as_ref()
            .is_some_and(|im| (0..im.bodies.len()).any(|b| im.bodies.get(b).has_gravity()));
        has_gravity && fine.state.fields.prim.pre_field().is_some()
    }

    /// fill level `level`'s coarse-fine prim ghosts from its parent's
    /// time-interpolated prims: `(1 - alpha)*prim_old + alpha*prim_new`. pub so the decomposed
    /// driver can drive the fine subcycle (prolong -> fine ghost -> fine stages with exchange).
    pub fn prolong_cf(&self, level: usize, alpha: f64) {
        let parent = &self.levels[level - 1];
        let fine = &self.levels[level];
        let prim_old = parent
            .prim_old
            .as_ref()
            .expect("the parent of a fine level carries prim_old");
        let prim_lerp = parent
            .prim_lerp
            .as_ref()
            .expect("the parent of a fine level carries prim_lerp");
        let slabs = cf_ghost_slabs(
            &fine.state.geom.allocated,
            &fine.state.geom.interior,
            &fine.state.boundaries,
        );
        let sweep_scratch = fine.prolong_sweep.get_or_init(|| {
            let has_pre = fine.state.fields.prim.pre_field().is_some();
            slabs
                .iter()
                .map(|s| ProlongSweepScratch::for_slab(s, self.prolong_order, has_pre))
                .collect()
        });
        let target = parent.prim_eq.as_ref().zip(fine.prim_eq.as_ref());
        let balanced = self.cf_transfer_balanced(level);
        for (slab, scratch) in slabs.iter().zip(sweep_scratch) {
            if let Some((coarse_target, fine_target)) = target {
                prolong_prims_targeted(
                    scratch,
                    prim_lerp,
                    prim_old,
                    &parent.state.fields.prim,
                    coarse_target,
                    fine_target,
                    &fine.state.fields.prim,
                    slab,
                    self.prolong_order,
                    alpha,
                );
            } else if balanced {
                // the equilibrium decomposition: encode the lerped coarse slab's
                // pressure as its departure from the mechanical equilibrium chained
                // out of the interior, prolong the departures with the unchanged
                // kernels, rebuild each fine ghost's pressure on the fine chain from
                // its nearest interior cell. baked kernel pair in the parent's lerp
                // scratch, with the slab's sweep intermediates carrying the departures.
                prolong_prims_balanced(
                    scratch,
                    prim_old,
                    &parent.state.fields.prim,
                    prim_lerp,
                    &fine.state.fields.prim,
                    slab,
                    &fine.state.geom.interior,
                    self.prolong_order,
                    alpha,
                    &parent.state.geom.x_lo,
                    &parent.state.geom.dx,
                    &fine.state.geom.x_lo,
                    &fine.state.geom.dx,
                    &parent.state.immersed.as_ref().unwrap().bodies,
                    &fine.state.immersed.as_ref().unwrap().bodies,
                );
            } else {
                prolong_prims_swept(
                    scratch,
                    prim_lerp,
                    prim_old,
                    &parent.state.fields.prim,
                    &fine.state.fields.prim,
                    slab,
                    self.prolong_order,
                    alpha,
                );
            }
            // the dye rides the same time-interpolated prolongation as the rest of the primitive
            // state. it sits outside the swept pass because that pass carries a positional
            // component count (rho, DOF velocities, optional pressure) sized by its scratch.
            if let (Some(chi_old), Some(pchi), Some(fchi)) = (
                prim_old.chi_field(),
                parent.state.fields.prim.chi_field(),
                fine.state.fields.prim.chi_field(),
            ) {
                prolong_field(chi_old, pchi, fchi, slab, self.prolong_order, alpha);
            }
            // mhd: the cell-centered B ghosts feed the fine reconstruction +
            // the boundary-edge UCT emf. the fine CT evolves its own owned
            // bface, and CT preserves divB for whatever emf values it is
            // handed, so those faces stand on their own.
            if let (Some(fmhd), Some(bcell_old)) =
                (fine.state.fields.mhd.as_ref(), parent.bcell_old.as_ref())
            {
                let pmhd = parent.state.fields.mhd.as_ref().unwrap();
                for aa in 0..NDIM {
                    prof("refine_prolong_face", || {
                        prolong_field(
                            &bcell_old[aa],
                            &pmhd.bcell[aa],
                            &fmhd.bcell[aa],
                            slab,
                            self.prolong_order,
                            alpha,
                        )
                    });
                }
            }
        }
        // the bface transverse halo at coarse-fine sides: the scalar ghost
        // fill skips CF faces, so the transversely-extended flux sweep would
        // read stale normal-B there — prolong it from the coarse face lattice
        // (divB and conservation are indifferent to these values; this is the
        // fine boundary-edge EMF quality).
        if let (Some(fmhd), Some(bface_old)) =
            (fine.state.fields.mhd.as_ref(), parent.bface_old.as_ref())
        {
            let pmhd = parent.state.fields.mhd.as_ref().unwrap();
            for dd in 0..NDIM {
                for slab in
                    bface_cf_halo_slabs(&fine.state.geom.interior, &fine.state.boundaries, dd)
                {
                    prof("refine_prolong_face", || {
                        prolong_face_field(
                            dd,
                            &bface_old[dd],
                            &pmhd.bface[dd],
                            &fmhd.bface[dd],
                            &slab,
                            alpha,
                        )
                    });
                }
            }
        }
    }
}

// =============================================================================
// well-balancing against a stationary target
// =============================================================================

impl<R, const NDIM: usize, const DOF: usize, M, E, S, Mem, K>
    Hierarchy<R, NDIM, DOF, M, E, S, Mem, K>
where
    R: Regime<f64, NDIM> + Regime<f64, DOF> + Copy,
    M: Metric<f64, NDIM> + Metric<f64, DOF> + Copy + Send + Sync,
    E: Eos<f64> + Copy + Send + Sync,
    S: ExecutionSpace,
    Mem: MemorySpace + Sync,
    K: KernelSet<NDIM, DOF, Mem, f64>,
    <R as Regime<f64, DOF>>::Cons: symbi_hydro::state::SeedableCons<f64, DOF>,
{
    /// declare the run's stationary target state `qt`, making it an exact fixed point of the
    /// refined scheme.
    ///
    /// a steady state solves the continuum equations, so on any grid the scheme leaves a residual
    /// `R = div_h F_h(qt) - s_h(qt)` at truncation order, and an atmosphere seeded on the exact
    /// hydrostatic profile starts moving. worse, `R` is grid-dependent, so the coarse-fine flux
    /// register differences two unequal reconstructions of one exact solution and applies the
    /// difference to the coarse cells at the interface as a force. this captures both halves of
    /// the cure: `F(qt)` per level, which the registers subtract from both sides of their
    /// difference, and `R` per level, which every stage adds back.
    ///
    /// the target reaches the flux phase through the preparation an evolving state gets —
    /// conserved-to-primitive recovery, coarse-fine prolongation, physical ghost fill — and `R` is
    /// read off one explicit stage of the shared pipeline, so both quantities are what the scheme
    /// genuinely produces when handed the target, ghost band and every source included.
    ///
    /// the target must be time-independent: both quantities are evaluated once here and reused
    /// every stage of every step. each level's conserved and primitive state is left exactly as it
    /// was found, so this may be called at any point after the initial condition is seeded.
    pub fn with_equilibrium(
        mut self,
        target: impl Fn([f64; NDIM]) -> <R as Regime<f64, DOF>>::Prim,
    ) -> symbi_xpu::Result<Self> {
        for (ll, level) in self.levels.iter().enumerate() {
            assert!(
                level.state.fields.mhd.is_none(),
                "level {ll}: a cell-centered primitive cannot seed the staggered face field, so a \
                 magnetized target's interface flux is undefined"
            );
            assert!(
                level.state.fields.cons.chi_field().is_none(),
                "level {ll}: the passive scalar carries no declared target concentration, so its \
                 interface flux has nothing to difference against"
            );
        }

        let saved = self
            .levels
            .iter()
            .map(|level| save_gas_state(&level.state))
            .collect::<symbi_xpu::Result<Vec<_>>>()?;

        // the copy above runs on the device queue while the seeding below writes the same fields
        // from the host. field storage is unified memory, so both reach the same bytes, but a host
        // write is unordered against an in-flight kernel, so this barrier is what keeps the copy
        // from reading cells the target had already overwritten and "restoring" the run to the
        // target it was asked to measure.
        symbi_substrate::regimes::substrate_gpu::device_sync::<Mem>();
        for level in &self.levels {
            level.state.seed_cells(&target);
        }
        // the target has to be hierarchy-consistent: a coarse target cell equals the
        // volume-weighted average of the fine target cells covering it, or the run's own
        // restriction moves a level that was sitting on its target off it, twice per parent step.
        // an analytic profile evaluated independently per level breaks that — the two differ at
        // the profile's curvature — so the coarse target is defined as the restriction of the fine
        // one wherever a finer level exists.
        for ll in (0..self.levels.len().saturating_sub(1)).rev() {
            let (lo, hi) = self.levels.split_at(ll + 1);
            let coarse = &lo[ll];
            let fine = &hi[0];
            restrict_cons(
                &fine.state.fields.cons,
                &coarse.state.fields.cons,
                coarse.coverage.as_ref().unwrap(),
            );
        }
        self.init_levels();

        // the target's conserved state, held aside as the anchor the probe stage is differenced
        // against.
        let anchor = self
            .levels
            .iter()
            .map(|level| save_gas_state(&level.state))
            .collect::<symbi_xpu::Result<Vec<_>>>()?;

        let mut flux_eq = Vec::with_capacity(self.levels.len());
        let mut residual_eq = Vec::with_capacity(self.levels.len());
        for (level, anchored) in self.levels.iter().zip(&anchor) {
            level.kernels.wave_speeds(&level.state);
            for dd in 0..NDIM {
                level.kernels.flux(&level.state, dd);
            }
            flux_eq.push(snapshot_flux(&level.state)?);

            // one forward-euler stage of the shared pipeline, evaluated entirely at the target:
            // every flux and every source sees the target and no other state, so the advanced
            // state is `qt - dt R` and the probe step cancels out of the quotient. the level's own
            // cfl step is the scale that keeps that true — a far larger step drives the probe
            // state unphysical and the response stops being linear, a far smaller one loses the
            // difference `qt - advanced` to cancellation.
            let probe_dt = level.kernels.cfl(&level.state);
            assert!(
                probe_dt.is_finite() && probe_dt > 0.0,
                "the stationary target has no finite cfl step ({probe_dt:.3e}), so it cannot be \
                 advanced to read off its imbalance"
            );
            let outcome = fold_stage(
                &level.state,
                &level.kernels,
                StageArgs {
                    dt: probe_dt,
                    a0: 0.0,
                    ac: 1.0,
                    stage: 0,
                    n_stages: 1,
                    allow_elision: false,
                },
                &mut |_hp: HookPoint| {},
            );
            assert!(
                outcome == symbi_sim::stage::StageOutcome::Accepted,
                "the stationary target was rejected by the admissibility redo, so it is not a \
                 state the scheme can carry"
            );
            // an inadmissible probe state means the update was clipped somewhere, and a clipped
            // update departs from `qt - dt R`: what would be read off is a limiter's response in
            // place of the scheme's imbalance.
            symbi_substrate::regimes::substrate_gpu::device_sync::<Mem>();
            let err = scan_c2p_errors(&level.state);
            assert!(
                err.is_ok(),
                "advancing the stationary target by one cfl step left an unrecoverable state \
                 ({err}), so its imbalance cannot be read off a linear response"
            );
            residual_eq.push(imbalance_from_stage(
                anchored.cons(),
                &level.state.fields.cons,
                &level.state.geom.allocated,
                probe_dt,
            )?);
        }

        for (level, state) in self.levels.iter().zip(&saved) {
            restore_gas_state(&level.state, state);
        }
        for (((level, flux), residual), anchored) in self
            .levels
            .iter_mut()
            .zip(flux_eq)
            .zip(residual_eq)
            .zip(anchor)
        {
            level.flux_eq = Some(flux);
            level.residual_eq = Some(residual);
            let (cons, prim) = anchored.into_parts();
            level.cons_eq = Some(cons);
            level.prim_eq = Some(prim);
        }
        self.report_target_stationarity();
        Ok(self)
    }

    /// the declared target must actually be a steady state of the equations being solved.
    ///
    /// the method takes the declaration as given: the imbalance is measured and subtracted whatever
    /// it is, so a state merely asserted to be stationary gets held motionless and the run proceeds
    /// silently on the wrong problem. that silence is this feature's sharpest edge.
    ///
    /// the discriminator is how the imbalance behaves under refinement. for a genuine steady state
    /// `R` is pure truncation error, `C(x) dx^p`, so halving the cell width cuts its norm over a
    /// fixed region by `2^p`. for a state that departs from the equations, `R` converges to the
    /// continuum residual, a property of the state alone that holds flat under refinement. every
    /// level pair measures this for free, over the region they both cover.
    ///
    /// the test is source-agnostic — gravity, rotation, the curvilinear geometric terms and any
    /// user source all enter `R` through the stage that produced it.
    /// report how the declared target's discrete imbalance behaves under refinement.
    ///
    /// this is advisory: it reports and lets the run continue. the deviation method carries it as a
    /// diagnostic, because declaring a non-stationary state is a silent error — the scheme holds
    /// whatever it is given, so the run proceeds quietly on the wrong problem.
    ///
    /// it is advisory because soundness and completeness pull apart here. a genuine steady state's
    /// imbalance is truncation error, which converges where the grid resolves the target; a
    /// non-stationary state leaves the continuum residual, which holds flat everywhere. for a
    /// strongly stratified target those two populations overlap — the imbalance is large where the
    /// profile is steep, and steep is where convergence stays undemonstrable either way. a check
    /// that refused on that basis would reject correct equilibria for being sharp, which is the
    /// more damaging error: it blocks valid science, where a missed warning leaves a mistake the
    /// user can still find by other means.
    ///
    /// what it reports is therefore evidence for the reader to weigh. a median near 1 on resolved
    /// cells is strong evidence the target departs from these equations; anything else is
    /// inconclusive.
    fn report_target_stationarity(&self) {
        for ll in 0..self.levels.len().saturating_sub(1) {
            let Some(measured) = self.target_imbalance_convergence(ll) else {
                continue;
            };
            let table: Vec<String> = (0..measured.ratio.len())
                .map(|cc| {
                    format!(
                        "[{cc}] ratio {:.3} scale {:.3e} cells {}",
                        measured.ratio[cc], measured.scale[cc], measured.sampled[cc]
                    )
                })
                .collect();
            // no component with broadly distributed signal on resolved cells: nothing was
            // measured. say so, because silence would otherwise read as "checked and clean".
            if (0..measured.ratio.len()).all(|cc| measured.sampled[cc] < MIN_CONVERGENCE_SAMPLE) {
                eprintln!(
                    "note: the declared stationary target between levels {ll} and {} was not \
                     checked. its imbalance is concentrated where the grid does not resolve it \
                     ({} of {} overlapping cells resolved), so no component carries signal on \
                     cells that could show convergence either way. this is expected for a sharply \
                     stratified target and is NOT evidence of a problem; it is also not evidence \
                     of correctness. every component: {}",
                    ll + 1,
                    measured.resolved,
                    measured.considered,
                    table.join("  ")
                );
                continue;
            }
            let suspect = (0..measured.ratio.len()).any(|cc| {
                measured.sampled[cc] >= MIN_CONVERGENCE_SAMPLE && measured.ratio[cc] <= 1.5
            });
            if suspect {
                eprintln!(
                    "warning: the declared stationary target may not be a steady state of these \
                     equations. between levels {ll} and {}, a conserved component's discrete \
                     imbalance did not shrink when the cell width halved, on cells where the grid \
                     resolves the target. truncation error falls by at least 2; what does not fall \
                     is the continuum residual. if that is so, well-balancing this state would \
                     hold the run motionless in something that is not an equilibrium. \
                     {} of {} overlapping cells resolved; every component: {}",
                    ll + 1,
                    measured.resolved,
                    measured.considered,
                    table.join("  ")
                );
            }
        }
    }

    /// how the target's discrete imbalance behaves under refinement, measured per cell and reduced
    /// with a median — the raw evidence behind the steady-state check, threshold-free.
    ///
    /// each coarse cell in the overlap is compared against the mean of its own children, so the
    /// statistic is local. that is what makes it usable: a ratio of summed norms is dominated by
    /// wherever the imbalance happens to be largest, and for a target with an unresolved feature —
    /// the `1/r` cusp of a point-mass atmosphere at the center of a refinement box, say — that is a
    /// handful of cells whose error stays flat because both grids under-resolve them. those cells
    /// are real, and they are silent on whether the declared state solves the equations. a median
    /// over cells is robust to them.
    ///
    /// `None` when the two levels share too little interior to compare. the region is the coverage
    /// trimmed by two coarse cells on every side: the fine level's outermost cells reconstruct
    /// through prolonged coarse-fine ghosts, whose error converges at the prolongation's order
    /// rather than the scheme's, and a mixture of the two measures neither.
    pub fn target_imbalance_convergence(&self, pair: usize) -> Option<ImbalanceConvergence> {
        const TRIM: isize = 2;
        // the imbalance is written by device kernels and read here on the host; the read goes
        // through unified memory, which leaves the barrier below to order it against the queue.
        symbi_substrate::regimes::substrate_gpu::device_sync::<Mem>();
        let coarse = self.levels.get(pair)?;
        let fine = self.levels.get(pair + 1)?;
        let coverage = coarse.coverage.as_ref()?;

        let inner = Domain::new(std::array::from_fn(|aa| {
            let s = &coverage.spaces[aa];
            Space {
                name: s.name,
                lo: s.lo + TRIM,
                hi: s.hi - TRIM,
            }
        }));
        if (0..NDIM).any(|aa| inner.spaces[aa].hi <= inner.spaces[aa].lo) {
            return None;
        }

        // the cells the question can be asked of. truncation error is a statement about a
        // solution the grid resolves; where the target changes by a large factor across one cell
        // the reconstruction is clipping rather than approximating, and its error carries no
        // order. such cells are real and are corrected like any other — they are silent on whether
        // the declared state solves the equations, so they stay out of the statistic instead of
        // setting it.
        let target = coarse.cons_eq.as_ref()?;
        let den = target.den.view();
        // a cell testifies about convergence when the grid resolves the target there: where the
        // density turns over inside one width the reconstruction is clipping rather than
        // approximating, and its error carries no order either way.
        // inside a sink the drain removes mass and energy, so the target departs from a steady
        // state of the applied equations there and those cells report the drain rather than
        // truncation, setting the statistic from a region the declaration left out.
        let mask = self.resolved_equilibrium_mask();
        let geom = &coarse.state.geom;
        let offered: Vec<[isize; NDIM]> = inner
            .iter()
            .filter(|c| {
                if mask <= 0.0 {
                    return true;
                }
                let x = geom.cell_coord(*c);
                let r2: f64 = (0..NDIM).map(|ax| x[ax] * x[ax]).sum();
                r2 >= mask * mask
            })
            .collect();
        let resolved: Vec<[isize; NDIM]> = offered
            .iter()
            .copied()
            .filter(|c| {
                let here = den.at(*c).abs().max(f64::MIN_POSITIVE);
                (0..NDIM).all(|ax| {
                    [-1isize, 1].iter().all(|step| {
                        let mut nb = *c;
                        nb[ax] += step;
                        (den.at(nb) - den.at(*c)).abs() / here < RESOLVED_VARIATION
                    })
                })
            })
            .collect();

        let coarse_comps = residual_components(coarse.residual_eq.as_ref()?);
        let fine_comps = residual_components(fine.residual_eq.as_ref()?);

        // the scale is the component's peak over the whole overlap, the resolved subset included.
        // it sets what counts as real signal, and a floor derived from the quiet cells alone would
        // admit their own roundoff as evidence.
        let scale: Vec<f64> = coarse_comps
            .iter()
            .map(|field| {
                let view = field.view();
                inner
                    .iter()
                    .map(|c| view.at(c).abs())
                    .fold(0.0_f64, f64::max)
            })
            .collect();

        let mut ratio = Vec::with_capacity(coarse_comps.len());
        let mut sampled = Vec::with_capacity(coarse_comps.len());
        for (cc, (cf, ff)) in coarse_comps.iter().zip(&fine_comps).enumerate() {
            let (cv, fv) = (cf.view(), ff.view());
            let mut ratios = Vec::new();
            for c in &resolved {
                let numerator = cv.at(*c).abs();
                // a cell whose own imbalance sits at roundoff reports noise over noise, so the
                // sample is restricted to cells that are resolved and carrying signal. a cell
                // whose residual is a millionth of this component's peak is reporting roundoff,
                // and a median of such cells reads 1 whatever the target is — the same reading a
                // genuine non-convergence gives.
                if numerator < SIGNAL_FLOOR * scale[cc].max(f64::MIN_POSITIVE) {
                    continue;
                }
                let mut child_sum = 0.0;
                let mut children = 0usize;
                for_each_child::<NDIM>(*c, RATIO, |fc| {
                    child_sum += fv.at(fc).abs();
                    children += 1;
                });
                let child_mean = child_sum / children as f64;
                if child_mean > 0.0 {
                    ratios.push(numerator / child_mean);
                }
            }
            sampled.push(ratios.len());
            ratio.push(median(&mut ratios));
        }
        Some(ImbalanceConvergence {
            ratio,
            scale,
            sampled,
            resolved: resolved.len(),
            considered: offered.len(),
        })
    }

    /// declare the stationary target as an expression of position rather than as a closure.
    ///
    /// this is the form a configured run supplies, and the form that survives a restart: the
    /// target is re-derived from the expression at whatever resolution the restarted hierarchy
    /// has, so a restart that adds a refinement level gets a target on the newly created cells.
    /// sampled field data covers the cells it was written from alone.
    ///
    /// the expression's outputs are the primitive components in order — density, one velocity
    /// component per momentum degree of freedom, then pressure when the regime carries energy —
    /// and are evaluated at each cell center at time zero, because the target is stationary.
    pub fn with_equilibrium_expression(
        self,
        config: &symbi_hydro::EquilibriumConfig,
    ) -> symbi_xpu::Result<Self>
    where
        <R as Regime<f64, DOF>>::Prim: symbi_hydro::state::PrimFromSlots<f64, DOF>,
    {
        let has_energy = self.levels[0].state.fields.cons.has_energy();
        let expected = 1 + DOF + usize::from(has_energy);
        assert_eq!(
            config.outputs.len(),
            expected,
            "the declared stationary target supplies {} primitive components; this run needs \
             {expected} — density, {DOF} velocity component(s){}",
            config.outputs.len(),
            if has_energy {
                ", and pressure"
            } else {
                " (no pressure: the regime carries no energy)"
            }
        );
        assert_eq!(
            config.dim, NDIM,
            "the declared stationary target was built for a {}-dimensional grid, but this run is \
             {NDIM}-dimensional",
            config.dim
        );
        let expression = config
            .to_expression()
            .unwrap_or_else(|err| panic!("the declared stationary target does not load: {err:?}"));

        self.with_equilibrium(|x| {
            let at = |aa: usize| x.get(aa).copied().unwrap_or(0.0);
            // a stationary target is evaluated once, at t = 0; it is the same state at every later
            // time by definition.
            let slots = expression.eval(at(0), at(1), at(2), 0.0);
            <R as Regime<f64, DOF>>::Prim::from_slots(&slots)
        })
    }

    /// seed every level's conserved state from the declared target.
    ///
    /// the target a refined hierarchy holds exactly has covered cells carrying the restriction of
    /// the finer level's target, because that is what the run's own restriction produces and
    /// re-produces every parent step. a run seeded from the pointwise profile instead starts a
    /// truncation-order distance off the state the well-balancing preserves, and that distance
    /// evolves like any other perturbation. build a perturbed initial condition on top of this
    /// seeding pass.
    pub fn seed_equilibrium(&mut self) {
        for (ll, level) in self.levels.iter().enumerate() {
            let target = level.cons_eq.as_ref().unwrap_or_else(|| {
                panic!("level {ll} has no declared stationary target to seed from")
            });
            equilibrium_overwrite(
                target,
                &level.state.fields.cons,
                &level.state.geom.allocated,
            );
        }
        self.init_levels();
    }

    /// rewrite every level's interior in primitive space: `f` receives each cell's physical
    /// center and its recovered primitive and returns the primitive to store. every level
    /// evaluates that one closure at its own cell centers, so a perturbation carries each
    /// level's full resolvable content in place of the coarse grid's prolongation — the entry
    /// point for laying a multi-scale velocity field over a prolonged smooth base state.
    ///
    /// may be called repeatedly (a measured correction pass after a seeding pass, say);
    /// call `sync_perturbed` once after the last pass.
    pub fn perturb_cells(
        &self,
        f: impl Fn([f64; NDIM], <R as Regime<f64, DOF>>::Prim) -> <R as Regime<f64, DOF>>::Prim,
    ) where
        <R as Regime<f64, DOF>>::Prim: symbi_hydro::state::PrimFromSlots<f64, DOF>,
    {
        for (ll, level) in self.levels.iter().enumerate() {
            assert!(
                level.state.fields.mhd.is_none(),
                "level {ll}: a cell-centered primitive rewrite cannot update the staggered \
                 face field, so a magnetized state's div(B) = 0 would not survive it"
            );
        }
        // field storage is unified memory; this barrier is what orders a host write against an
        // in-flight kernel.
        symbi_substrate::regimes::substrate_gpu::device_sync::<Mem>();
        for level in &self.levels {
            level.state.perturb_cells(&f);
        }
    }

    /// restore hierarchy consistency after `perturb_cells`: covered coarse cells are
    /// re-defined as the restriction of the fine state — what the run's own sync produces
    /// every parent step — and the primitive/ghost state is re-initialized on every level.
    pub fn sync_perturbed(&mut self) {
        for ll in (0..self.levels.len().saturating_sub(1)).rev() {
            let (lo, hi) = self.levels.split_at(ll + 1);
            let coarse = &lo[ll];
            let fine = &hi[0];
            restrict_cons(
                &fine.state.fields.cons,
                &coarse.state.fields.cons,
                coarse.coverage.as_ref().unwrap(),
            );
        }
        self.init_levels();
    }
}

// =============================================================================
// helpers
// =============================================================================

/// how a declared stationary target's discrete imbalance behaves under one halving of the cell
/// width, per conserved component.
pub struct ImbalanceConvergence {
    /// median over resolved coarse cells of `|R_coarse(cell)| / mean |R_fine(its children)|`.
    /// truncation error gives `2^p` (4 for a second-order scheme); a continuum residual gives 1.
    pub ratio: Vec<f64>,
    /// the largest single-cell `|R|` over the resolved cells — the scale below which a
    /// component's ratios are quotients of roundoff and say nothing.
    pub scale: Vec<f64>,
    /// how many resolved cells carried enough signal to contribute a ratio, per component.
    pub sampled: Vec<usize>,
    /// how many cells the target is resolved on, out of `considered`.
    pub resolved: usize,
    /// how many cells the overlap offered before the resolution filter.
    pub considered: usize,
}

/// the fewest cells a median may be taken over before it is a statistic rather than an accident.
const MIN_CONVERGENCE_SAMPLE: usize = 8;

/// the largest fractional change in the target's density across one cell for that cell to count as
/// resolved. beyond it a limited reconstruction is clipping rather than approximating and its
/// error carries no order, leaving the cell silent about convergence either way. a stratified
/// atmosphere spread over its grid sits far below this; the cells it excludes are the ones where
/// the profile turns over inside a single width — a `1/r` cusp at a box center, typically.
const RESOLVED_VARIATION: f64 = 0.25;

/// the smallest share of a component's peak imbalance a cell must carry to vote on convergence.
/// below it the cell reports roundoff, and a median of such cells reads 1 whatever the target is —
/// which is exactly the value a genuinely non-stationary target gives, so admitting them makes the
/// two indistinguishable.
const SIGNAL_FLOOR: f64 = 1.0e-3;

/// the median of `values`, or 0 when empty. sorts in place; the caller owns the buffer.
fn median(values: &mut [f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = values.len() / 2;
    if values.len() % 2 == 0 {
        0.5 * (values[mid - 1] + values[mid])
    } else {
        values[mid]
    }
}

/// visit the `ratio^D` fine cells covering one coarse cell. levels share absolute index space, so
/// the children of coarse cell `c` start at `ratio * c`.
fn for_each_child<const D: usize>(c: [isize; D], ratio: usize, mut f: impl FnMut([isize; D])) {
    let base: [isize; D] = std::array::from_fn(|ax| c[ax] * ratio as isize);
    let total = ratio.pow(D as u32);
    for n in 0..total {
        let mut child = base;
        let mut rem = n;
        for slot in child.iter_mut() {
            *slot += (rem % ratio) as isize;
            rem /= ratio;
        }
        f(child);
    }
}

/// the per-stage effective flux weights of an ssp scheme: stage i's operator
/// enters the step total with `ac_i * prod_{k>i} ac_k` (each later convex
/// combine rescales the accumulated state by its ac). euler -> [1]; rk2 ->
/// [1/2, 1/2]; rk3 -> [1/6, 1/6, 2/3]. the weights sum to 1; the register
/// accumulates `weight * dt * F * A` so refluxing matches the actual flux the
/// scheme pushed through each face.
fn flux_weights(stages: &[(f64, f64)]) -> Vec<f64> {
    (0..stages.len())
        .map(|ii| {
            let mut w = stages[ii].1;
            for kk in ii + 1..stages.len() {
                w *= stages[kk].1;
            }
            w
        })
        .collect()
}

/// is this level on a uniform cartesian grid — the flux register's kernel
/// path (cpu + gpu, constant area/volume scales)? curvilinear levels keep
/// the per-coordinate host loops.
fn uniform_cartesian<R, const NDIM: usize, const DOF: usize, M, E, S, Mem>(
    state: &SimStateGeneric<R, NDIM, DOF, M, E, S, Mem>,
) -> bool
where
    R: Regime<f64, NDIM>,
    M: Metric<f64, NDIM>,
    E: Eos<f64>,
    S: ExecutionSpace,
    Mem: MemorySpace,
{
    state.geom.coords == symbi_geometry::Geometry::Cartesian && state.geom.maps.is_none()
}

/// snapshot a level's prims (+ cell/face B for mhd) into the *_old buffers
/// via the copy kernel — parallel on the host, device-side on a gpu backend
/// (no host touch on unified memory). a serial host memcpy here was a
/// measured per-root-step stall at production sizes (~0.7 gb per step at
/// n = 256); the dispatch setup only matters on demo grids where the copy
/// is microseconds anyway.
fn save_prim_old<R, const NDIM: usize, const DOF: usize, M, E, S, Mem, K>(
    lvl: &LevelData<R, NDIM, DOF, M, E, S, Mem, K>,
) where
    R: Regime<f64, NDIM>,
    M: Metric<f64, NDIM> + Copy,
    E: Eos<f64>,
    S: ExecutionSpace,
    Mem: MemorySpace,
    K: KernelSet<NDIM, DOF, Mem, f64>,
{
    let po = lvl
        .prim_old
        .as_ref()
        .expect("save_prim_old: prim_old allocated");
    let prim = &lvl.state.fields.prim;

    copy_field(&prim.rho, &po.rho);
    for dd in 0..DOF {
        copy_field(&prim.vel[dd], &po.vel[dd]);
    }
    if let (Some(p), Some(pp)) = (prim.pre_field(), po.pre_field()) {
        copy_field(p, pp);
    }
    if let (Some(c), Some(pc)) = (prim.chi_field(), po.chi_field()) {
        copy_field(c, pc);
    }
    if let (Some(mhd), Some(bo)) = (lvl.state.fields.mhd.as_ref(), lvl.bcell_old.as_ref()) {
        for dd in 0..NDIM {
            copy_field(&mhd.bcell[dd], &bo[dd]);
        }
    }
    if let (Some(mhd), Some(bfo)) = (lvl.state.fields.mhd.as_ref(), lvl.bface_old.as_ref()) {
        for dd in 0..NDIM {
            copy_field(&mhd.bface[dd], &bfo[dd]);
        }
    }
}

/// remove accretion from coarser-level body proxies. rigid surfaces remain active because a
/// body may cross a coarse-fine boundary and must act on uncovered coarse cells.
fn gravity_only<const D: usize>(
    bodies: &symbi_ib::BodyCollection<f64, D>,
) -> symbi_ib::BodyCollection<f64, D> {
    let mut coll = bodies.clone();
    coll.visit_all_mut(|body| {
        if let symbi_ib::BodyKind::BlackHole { softening, .. } = body.kind {
            body.kind = symbi_ib::BodyKind::Gravitational { softening };
        }
    });
    coll
}

/// restore the hydrodynamic conserved state over an inactive composite-grid region.
fn restore_cons_region<const D: usize, const DOF: usize, Mem: MemorySpace>(
    src: &ConsFieldsGeneric<D, DOF, Mem>,
    dst: &ConsFieldsGeneric<D, DOF, Mem>,
    region: &Domain<D>,
) {
    copy_field_region(&src.den, &dst.den, region);
    for dd in 0..DOF {
        copy_field_region(&src.mom[dd], &dst.mom[dd], region);
    }
    if let (Some(sn), Some(dn)) = (src.nrg_field(), dst.nrg_field()) {
        copy_field_region(sn, dn, region);
    }
    if let (Some(sc), Some(dc)) = (src.chi_field(), dst.chi_field()) {
        copy_field_region(sc, dc, region);
    }
}

/// whether an absolute index is inside a domain (per-axis half-open [lo, hi)).
fn domain_contains<const D: usize>(dom: &Domain<D>, idx: &[isize; D]) -> bool {
    (0..D).all(|ax| idx[ax] >= dom.spaces[ax].lo && idx[ax] < dom.spaces[ax].hi)
}

/// the coverage extended by one cell per axis, clamped to the interior — the
/// cells whose faces the restriction + emf reflux changed.
fn shell_around<const D: usize>(cov: &Domain<D>, interior: &Domain<D>) -> Domain<D> {
    Domain::new(std::array::from_fn(|ax| Space {
        name: cov.spaces[ax].name,
        lo: (cov.spaces[ax].lo - 1).max(interior.spaces[ax].lo),
        hi: (cov.spaces[ax].hi + 1).min(interior.spaces[ax].hi),
    }))
}

/// snap a physical refinement region to parent cells: absolute parent indices
/// from the shared global origin.
fn region_to_domain<const D: usize>(
    region: &RefinementRegion<D>,
    x_lo: &[f64; D],
    dx: &[f64; D],
) -> Domain<D> {
    Domain::new(std::array::from_fn(|ax| Space {
        name: axis_name(ax),
        lo: ((region.x_lo[ax] - x_lo[ax]) / dx[ax]).round() as isize,
        hi: ((region.x_hi[ax] - x_lo[ax]) / dx[ax]).round() as isize,
    }))
}

/// the coverage must sit inside the parent interior, and on every coarse-fine
/// (non-touching) side leave enough parent cells that the prolongation's
/// deepest read — the parent of the outermost fine ghost minus the stencil
/// halfwidth — stays inside the parent allocated domain.
fn validate_coverage<R, const D: usize, const DOF: usize, M, E, S, Mem>(
    coverage: &Domain<D>,
    parent: &SimStateGeneric<R, D, DOF, M, E, S, Mem>,
    fine_ng: usize,
    order: ProlongOrder,
) where
    R: Regime<f64, D>,
    M: Metric<f64, D> + Copy,
    E: Eos<f64>,
    S: ExecutionSpace,
    Mem: MemorySpace,
{
    let reach = (fine_ng.div_ceil(RATIO) + order.ghost_width()) as isize;
    for ax in 0..D {
        let (clo, chi) = (coverage.spaces[ax].lo, coverage.spaces[ax].hi);
        let int = &parent.geom.interior.spaces[ax];
        let alloc = &parent.geom.allocated.spaces[ax];
        assert!(
            clo >= int.lo && chi <= int.hi && clo < chi,
            "refinement: coverage [{clo}, {chi}) outside parent interior [{}, {}) on axis {ax}",
            int.lo,
            int.hi
        );
        if clo != int.lo {
            assert!(
                clo - reach >= alloc.lo,
                "refinement: coverage lo {clo} on axis {ax} leaves the prolongation stencil \
                 (reach {reach}) outside the parent allocated domain (lo {})",
                alloc.lo
            );
        }
        if chi != int.hi {
            assert!(
                chi + reach <= alloc.hi,
                "refinement: coverage hi {chi} on axis {ax} leaves the prolongation stencil \
                 (reach {reach}) outside the parent allocated domain (hi {})",
                alloc.hi
            );
        }
    }
}

// =============================================================================
// decomposed hierarchy driver (refinement x decomposition)
// =============================================================================

/// the sub-grid of tiles that carry the first fine level. for SMR (one refined box per level) the
/// refined tiles form a contiguous rectangle; `order[k]` is the tile index at fine-flatten position
/// `k`, and `devices[k]`/`counts` give that sub-grid's device map + shape for `exchange_grid`. when
/// the patch is tile-local the sub-grid is 1x1, so the fine exchange is a no-op; when
/// the patch spans cuts the sub-grid has internal cuts and the fine halos are exchanged.
pub struct FineSubgrid<const NDIM: usize> {
    pub counts: [usize; NDIM],
    pub order: Vec<usize>,
    pub devices: Vec<i32>,
}

/// derive the first-fine-level sub-grid from which tiles carry a fine level. None if no tile is
/// refined. asserts the refined tiles fill a rectangle (the SMR single-box invariant). pub so the
/// seed every tile's fine levels from its coarse level, decomposition-aware: the root
/// conserved cut halos are exchanged first, then each tile prolongs.
///
/// a refined patch spanning a cut needs this exchange on top of the per-tile
/// [`Hierarchy::seed_fine_from_coarse`]: the seed prolongs conserved components, its stencil at
/// the coverage edge reads the root's cut ghosts, and this exchange is the sole filler of
/// conserved ghosts -- the evolve loop's exchange moves primitives, on the (correct there)
/// invariant that the flux stages read primitive ghosts alone. seeded per tile, the cut-side fine
/// interior would be prolonged from each tile's standalone boundary fill in place of its
/// neighbor's data, and the decomposed hierarchy would differ from the monolithic one at step
/// zero -- measured 7.2e-3 on a smooth bump centered on the cut, decaying as the flow smooths it,
/// and growing with the prolongation order (a higher-degree interpolant weights the ghost cells
/// more heavily). tile-local patches stay clear of every cut and behave identically either way.
pub fn seed_decomposed_fine_from_coarse<
    R,
    const NDIM: usize,
    const DOF: usize,
    M,
    E,
    S,
    Mem,
    K,
    T,
>(
    tiles: &[Hierarchy<R, NDIM, DOF, M, E, S, Mem, K>],
    counts: [usize; NDIM],
    devices: &[i32],
    transport: &T,
) -> symbi_xpu::Result<usize>
where
    R: Regime<f64, NDIM> + Copy,
    M: Metric<f64, NDIM> + Copy + Send + Sync,
    E: Eos<f64> + Copy + Send + Sync,
    S: ExecutionSpace,
    Mem: MemorySpace + Sync,
    K: KernelSet<NDIM, DOF, Mem, f64>,
    T: HaloTransport,
{
    drain_devices::<Mem>(devices);
    {
        let states: Vec<&FieldStore<NDIM, DOF, Mem, f64>> =
            tiles.iter().map(|t| &*t.levels[0].state).collect();
        symbi_sim::decomp::exchange_grid_cons(&states, counts, devices, transport);
    }
    let mut reseeded = 0usize;
    for (i, t) in tiles.iter().enumerate() {
        if t.levels.len() > 1 {
            reseeded += symbi_xpu::with_device(devices[i], || t.seed_fine_from_coarse())?;
        }
    }
    Ok(reseeded)
}

/// python decomposed-refinement loop can gather the fine level with the same tile order + counts.
pub fn fine_subgrid<R, const NDIM: usize, const DOF: usize, M, E, S, Mem, K>(
    tiles: &[Hierarchy<R, NDIM, DOF, M, E, S, Mem, K>],
    counts: [usize; NDIM],
    devices: &[i32],
) -> Option<FineSubgrid<NDIM>>
where
    R: Regime<f64, NDIM>,
    M: Metric<f64, NDIM> + Copy,
    E: Eos<f64>,
    S: ExecutionSpace,
    Mem: MemorySpace,
    K: KernelSet<NDIM, DOF, Mem, f64>,
{
    let refined: Vec<usize> = (0..tiles.len())
        .filter(|&i| tiles[i].levels.len() > 1)
        .collect();
    if refined.is_empty() {
        return None;
    }
    // the tile-coord bounding box of the refined tiles -> the fine sub-grid shape.
    let mut lo = [usize::MAX; NDIM];
    let mut hi = [0usize; NDIM];
    for &i in &refined {
        let tc = unflatten(i, counts);
        for ax in 0..NDIM {
            lo[ax] = lo[ax].min(tc[ax]);
            hi[ax] = hi[ax].max(tc[ax]);
        }
    }
    let sub_counts: [usize; NDIM] = std::array::from_fn(|ax| hi[ax] - lo[ax] + 1);
    let nsub: usize = sub_counts.iter().product();
    // place each refined tile at its position in the sub-grid's flatten order.
    let mut order = vec![usize::MAX; nsub];
    for &i in &refined {
        let tc = unflatten(i, counts);
        let rc: [usize; NDIM] = std::array::from_fn(|ax| tc[ax] - lo[ax]);
        order[flatten(rc, sub_counts)] = i;
    }
    assert!(
        order.iter().all(|&i| i != usize::MAX),
        "refinement x decomposition: refined tiles do not form a rectangle (SMR is single-box)"
    );
    let sub_devices: Vec<i32> = order.iter().map(|&i| devices[i]).collect();
    Some(FineSubgrid {
        counts: sub_counts,
        order,
        devices: sub_devices,
    })
}

pub fn seed_decomposed_hierarchy_tracers<R, const NDIM: usize, const DOF: usize, M, E, S, Mem, K>(
    global: &Hierarchy<R, NDIM, DOF, M, E, S, Mem, K>,
    tiles: &mut [Hierarchy<R, NDIM, DOF, M, E, S, Mem, K>],
    count: usize,
) where
    R: Regime<f64, NDIM> + Copy,
    M: Metric<f64, NDIM> + Copy + Send + Sync,
    E: Eos<f64> + Copy + Send + Sync,
    S: ExecutionSpace,
    Mem: MemorySpace + Sync,
    K: KernelSet<NDIM, DOF, Mem, f64>,
{
    let seeded = global.seed_mass_tracers(count);
    for tile in tiles.iter_mut() {
        for level in &mut tile.levels {
            level.state.tracers = Some(symbi_sim::tracers::TracerSet {
                weight: seeded.weight,
                run_seed: seeded.run_seed,
                next_id: seeded.next_id,
                ..Default::default()
            });
        }
    }
    for ii in 0..seeded.len() {
        let owner = seeded.owner[ii];
        let destination = tiles
            .iter()
            .position(|tile| tile.tracer_owner_is_active(owner))
            .expect("composite tracer seed has no decomposed owner");
        let (level, coord) = tiles[destination]
            .tracer_cell(owner)
            .expect("decomposed tracer owner must resolve");
        let x = tiles[destination].levels[level].state.geom.centroid(coord);
        let tracers = tiles[destination].levels[level]
            .state
            .tracers
            .as_mut()
            .unwrap();
        tracers.x.push(x);
        tracers.id.push(seeded.id[ii]);
        tracers.cohort.push(seeded.cohort[ii]);
        tracers.flags.push(seeded.flags[ii]);
        tracers.owner.push(owner);
        tracers.step_owner.push(seeded.step_owner[ii]);
        tracers.step_flags.push(seeded.step_flags[ii]);
    }
}

pub fn gather_decomposed_hierarchy_tracers<
    R,
    const NDIM: usize,
    const DOF: usize,
    M,
    E,
    S,
    Mem,
    K,
>(
    global: &mut Hierarchy<R, NDIM, DOF, M, E, S, Mem, K>,
    tiles: &[Hierarchy<R, NDIM, DOF, M, E, S, Mem, K>],
) where
    R: Regime<f64, NDIM> + Copy,
    M: Metric<f64, NDIM> + Copy + Send + Sync,
    E: Eos<f64> + Copy + Send + Sync,
    S: ExecutionSpace,
    Mem: MemorySpace + Sync,
    K: KernelSet<NDIM, DOF, Mem, f64>,
{
    for level in 0..global.levels.len() {
        let mut records = Vec::new();
        let mut metadata = None;
        for tile in tiles {
            let Some(tracers) = tile
                .levels
                .get(level)
                .and_then(|data| data.state.tracers.as_ref())
            else {
                continue;
            };
            metadata.get_or_insert((
                tracers.weight,
                tracers.run_seed,
                tracers.next_id,
                tracers.injection_remainder,
            ));
            records.extend((0..tracers.len()).map(|ii| {
                (
                    tracers.id[ii],
                    tracers.x[ii],
                    tracers.cohort[ii],
                    tracers.flags[ii],
                    tracers.owner[ii],
                    tracers.step_owner[ii],
                    tracers.step_flags[ii],
                )
            }));
        }
        records.sort_unstable_by_key(|record| record.0);
        let (weight, run_seed, next_id, injection_remainder) = metadata.unwrap_or_default();
        let mut gathered = symbi_sim::tracers::TracerSet {
            weight,
            run_seed,
            next_id,
            injection_remainder,
            ..Default::default()
        };
        for (id, x, cohort, flags, owner, step_owner, step_flags) in records {
            gathered.id.push(id);
            gathered.x.push(x);
            gathered.cohort.push(cohort);
            gathered.flags.push(flags);
            gathered.owner.push(owner);
            gathered.step_owner.push(step_owner);
            gathered.step_flags.push(step_flags);
        }
        global.levels[level].state.tracers = Some(gathered);
    }
}

/// exchange one level's halos across an ordered set of tiles (a sub-grid of shape `counts`), then
/// per-tile `ghost_fill` at that level. `order[k]` is the tile index at flatten position `k`;
/// `devices[k]` is order-aligned. cut faces are `CoarseFine` (so `ghost_fill` leaves them for the
/// exchange); the sub-grid's outer faces are the patch's coarse-fine boundary (prolong-filled) or a
/// physical boundary, neither of which `exchange_grid` touches (it moves only internal cuts).
fn exchange_level_halos<R, const NDIM: usize, const DOF: usize, M, E, S, Mem, K, T>(
    tiles: &[Hierarchy<R, NDIM, DOF, M, E, S, Mem, K>],
    level: usize,
    order: &[usize],
    counts: [usize; NDIM],
    devices: &[i32],
    transport: &T,
) where
    R: Regime<f64, NDIM> + Copy,
    M: Metric<f64, NDIM> + Copy + Send + Sync,
    E: Eos<f64> + Copy + Send + Sync,
    S: ExecutionSpace,
    Mem: MemorySpace + Sync,
    K: KernelSet<NDIM, DOF, Mem, f64>,
    T: HaloTransport,
{
    drain_devices::<Mem>(devices);
    {
        let states: Vec<&FieldStore<NDIM, DOF, Mem, f64>> = order
            .iter()
            .map(|&i| &*tiles[i].levels[level].state)
            .collect();
        exchange_grid(&states, counts, devices, transport);
    }
    for (k, &i) in order.iter().enumerate() {
        symbi_xpu::with_device(devices[k], || {
            tiles[i].levels[level]
                .kernels
                .ghost_fill(&tiles[i].levels[level].state)
        });
    }
}

fn migrate_level_tracers<R, const NDIM: usize, const DOF: usize, M, E, S, Mem, K>(
    tiles: &mut [Hierarchy<R, NDIM, DOF, M, E, S, Mem, K>],
    level: usize,
) where
    R: Regime<f64, NDIM> + Copy,
    M: Metric<f64, NDIM> + Copy + Send + Sync,
    E: Eos<f64> + Copy + Send + Sync,
    S: ExecutionSpace,
    Mem: MemorySpace + Sync,
    K: KernelSet<NDIM, DOF, Mem, f64>,
{
    let mut migrating = Vec::new();
    for source in 0..tiles.len() {
        if level >= tiles[source].levels.len() {
            continue;
        }
        let destinations: std::collections::BTreeMap<u64, usize> = tiles[source].levels[level]
            .state
            .tracers
            .as_ref()
            .into_iter()
            .flat_map(|tracers| {
                tracers
                    .id
                    .iter()
                    .copied()
                    .zip(tracers.owner.iter().copied())
            })
            .filter_map(|(id, owner)| {
                tiles
                    .iter()
                    .enumerate()
                    .find_map(|(index, tile)| {
                        (index != source
                            && tile
                                .tracer_cell(owner)
                                .is_some_and(|(owner_level, _)| owner_level == level))
                        .then_some(index)
                    })
                    .map(|destination| (id, destination))
            })
            .collect();
        let Some(tracers) = tiles[source].levels[level].state.tracers.as_mut() else {
            continue;
        };
        let mut ii = 0usize;
        while ii < tracers.len() {
            let Some(&destination) = destinations.get(&tracers.id[ii]) else {
                ii += 1;
                continue;
            };
            tracers.x.swap_remove(ii);
            migrating.push((
                destination,
                tracers.id.swap_remove(ii),
                tracers.cohort.swap_remove(ii),
                tracers.flags.swap_remove(ii),
                tracers.owner.swap_remove(ii),
                tracers.step_owner.swap_remove(ii),
                tracers.step_flags.swap_remove(ii),
            ));
        }
    }
    for (destination, id, cohort, flags, owner, step_owner, step_flags) in migrating {
        let (_, coord) = tiles[destination]
            .tracer_cell(owner)
            .expect("tracer destination tile owns the addressed cell");
        let x = tiles[destination].levels[level].state.geom.centroid(coord);
        let tracers = tiles[destination].levels[level]
            .state
            .tracers
            .as_mut()
            .expect("every decomposed hierarchy level carries tracers");
        tracers.x.push(x);
        tracers.id.push(id);
        tracers.cohort.push(cohort);
        tracers.flags.push(flags);
        tracers.owner.push(owner);
        tracers.step_owner.push(step_owner);
        tracers.step_flags.push(step_flags);
    }
}

fn spawn_decomposed_injections<R, const NDIM: usize, const DOF: usize, M, E, S, Mem, K>(
    tiles: &mut [Hierarchy<R, NDIM, DOF, M, E, S, Mem, K>],
    order: &[usize],
    level: usize,
) where
    R: Regime<f64, NDIM> + Copy,
    M: Metric<f64, NDIM> + Copy + Send + Sync,
    E: Eos<f64> + Copy + Send + Sync,
    S: ExecutionSpace,
    Mem: MemorySpace + Sync,
    K: KernelSet<NDIM, DOF, Mem, f64>,
{
    let Some((mut next_id, mut injection_remainder)) =
        tiles.iter().flat_map(|tile| &tile.levels).find_map(|data| {
            data.state
                .tracers
                .as_ref()
                .map(|tracers| (tracers.next_id, tracers.injection_remainder))
        })
    else {
        return;
    };
    for &index in order {
        (next_id, injection_remainder) = tiles[index]
            .spawn_pending_injection(level, next_id, injection_remainder)
            .unwrap_or_else(|detail| panic!("decomposed tracer injection: {detail}"));
    }
    for tile in tiles {
        tile.set_tracer_spawn_state(next_id, injection_remainder);
    }
}

/// the one hierarchy march: N tiles advance in lockstep behind an agreed timestep, with
/// the per-tile watchdog, the step-panic catch, the crash report, and the observer
/// cadence shared by every shape. a single tile with no decomposition context steps
/// through the uni-grid transaction (`step_root_with_dt`); a decomposed set steps
/// through the collective attempt with halo exchanges. the observer fires every
/// `interval` root iterations, once when a crash is recorded (the driver snapshots
/// the `.crashed` state), and once after the final step, devices drained first.
pub fn evolve_tiles<R, const NDIM: usize, const DOF: usize, M, E, S, Mem, K, T, F>(
    tiles: &mut [Hierarchy<R, NDIM, DOF, M, E, S, Mem, K>],
    decomp: Option<(&[i32], [usize; NDIM], &T)>,
    t_final: f64,
    interval: u64,
    mut callback: F,
) where
    R: Regime<f64, NDIM> + Copy,
    M: Metric<f64, NDIM> + Copy + Send + Sync,
    E: Eos<f64> + Copy + Send + Sync,
    S: ExecutionSpace,
    Mem: MemorySpace + Sync,
    K: KernelSet<NDIM, DOF, Mem, f64>,
    T: HaloTransport,
    F: FnMut(&[Hierarchy<R, NDIM, DOF, M, E, S, Mem, K>]) -> ControlFlow<()>,
{
    let n = tiles.len();
    let drain = |decomp: Option<(&[i32], [usize; NDIM], &T)>| match decomp {
        Some((devices, _, _)) => drain_devices::<Mem>(devices),
        None => symbi_substrate::regimes::substrate_gpu::device_sync::<Mem>(),
    };
    let root_order: Vec<usize> = (0..n).collect();
    let nstages = tiles[0].levels[0].state.timestepping.stages().len();
    let fine = match decomp {
        Some((devices, counts, _)) => fine_subgrid(tiles, counts, devices),
        None => None,
    };
    // cut halos current before the first flux.
    if let Some((devices, counts, transport)) = decomp {
        exchange_level_halos(tiles, 0, &root_order, counts, devices, transport);
    }

    let mut last_cb = tiles[0].levels[0].state.iteration;
    while tiles[0].levels[0].state.time < t_final {
        // the agreed timestep: every tile runs the watchdog-screened selection, and the
        // minimum drives the lockstep step. a fatal estimate records the crash on its
        // tile and the march halts on the last computed state below.
        let mut gdt = f64::INFINITY;
        let mut dt_crashed = false;
        for i in 0..n {
            let d = match decomp {
                Some((devices, _, _)) => {
                    symbi_xpu::with_device(devices[i], || tiles[i].watchdog_root_dt(t_final))
                }
                None => tiles[i].watchdog_root_dt(t_final),
            };
            match d {
                Some(v) => gdt = gdt.min(v),
                None => {
                    dt_crashed = true;
                    break;
                }
            }
        }
        if !dt_crashed {
            // a panic inside the step (the FOFC freeze-streak halt, a poisoned-cell
            // assertion) carries the same diagnostic value as a watchdog crash and
            // deserves the same exit: catch it, convert it into a crash report carrying
            // the panic message, and let the observer snapshot the `.crashed`
            // checkpoint. the default panic hook has already printed the message and
            // backtrace to stderr at the panic site, so the report reaches the log and
            // the state reaches disk. the caught state is the mid-step one the panic
            // fired on, which is exactly the state worth inspecting.
            let step = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                match decomp {
                    Some((devices, counts, transport)) => {
                        let retry_sidecars: Vec<_> = tiles
                            .iter()
                            .map(Hierarchy::snapshot_retry_sidecars)
                            .collect();
                        let t = tiles[0].levels[0].state.time;
                        let adt = decomposed_root_attempt(
                            tiles,
                            counts,
                            devices,
                            transport,
                            &fine,
                            &root_order,
                            nstages,
                            &retry_sidecars,
                            gdt,
                            t,
                        );
                        decomposed_root_post(
                            tiles, counts, devices, transport, &root_order, adt, t,
                        );
                    }
                    None => tiles[0].step_root_with_dt(gdt),
                }
            }));
            if let Err(payload) = step {
                let msg = payload
                    .downcast_ref::<&'static str>()
                    .map(|s| s.to_string())
                    .or_else(|| payload.downcast_ref::<String>().cloned())
                    .unwrap_or_else(|| "panic with a non-string payload".to_string());
                let (iter, time) = {
                    let r = &tiles[0].levels[0].state;
                    (r.iteration, r.time)
                };
                tiles[0].crash = Some(CrashReport {
                    iter,
                    time,
                    dt_cfl: f64::NAN,
                    dt_prev: tiles[0].prev_dt_cfl,
                    panic: Some(msg),
                });
            }
        }
        // a crashed step records the crash and leaves the clock where it was: let the
        // observer snapshot the `.crashed` checkpoint + report it, then stop the march.
        // marching on would spin on the frozen time and clamp dt to t_final on garbage.
        if tiles.iter().any(|h| h.crash.is_some()) {
            drain(decomp);
            let _ = callback(tiles);
            return;
        }
        let it = tiles[0].levels[0].state.iteration;
        if it.saturating_sub(last_cb) >= interval {
            last_cb = it;
            drain(decomp);
            // a `Break` from the observer (e.g., a caught signal) stops the march early;
            // the observer has already snapshotted state for restart. drain the queue so
            // the host read in the caller is coherent, then return.
            if callback(tiles).is_break() {
                drain(decomp);
                return;
            }
        }
    }
    // the caller reads fields next: drain the device queue (the host-read barrier;
    // no-op on a host backend).
    drain(decomp);
    let _ = callback(tiles);
}

/// one collective root step of the decomposed march at an agreed timestep: every tile
/// snapshots its retry state, the root stages advance in lockstep with a halo exchange
/// between each, the fine subcycle runs with its own exchanges, and a rejection anywhere
/// rolls every tile back to the same entry state and replays the attempt at a reduced dt.
/// returns the accepted timestep the tiles advanced by.
#[allow(clippy::too_many_arguments)]
fn decomposed_root_attempt<R, const NDIM: usize, const DOF: usize, M, E, S, Mem, K, T>(
    tiles: &mut [Hierarchy<R, NDIM, DOF, M, E, S, Mem, K>],
    counts: [usize; NDIM],
    devices: &[i32],
    transport: &T,
    fine: &Option<FineSubgrid<NDIM>>,
    root_order: &[usize],
    nstages: usize,
    retry_sidecars: &[HierarchyRetrySidecars<NDIM>],
    mut gdt: f64,
    t: f64,
) -> f64
where
    R: Regime<f64, NDIM> + Copy,
    M: Metric<f64, NDIM> + Copy + Send + Sync,
    E: Eos<f64> + Copy + Send + Sync,
    S: ExecutionSpace,
    Mem: MemorySpace + Sync,
    K: KernelSet<NDIM, DOF, Mem, f64>,
    T: HaloTransport,
{
    let n = tiles.len();
        'attempt: loop {
            for i in 0..n {
                for level in &tiles[i].levels {
                    if level.kernels.fofc_active() {
                        symbi_xpu::with_device(devices[i], || {
                            level.kernels.snapshot_retry(&level.state)
                        });
                    }
                }
                symbi_xpu::with_device(devices[i], || tiles[i].level_step_begin(0, gdt));
            }
            let mut retry = false;
            // the root SSP stages, with a root halo exchange between each.
            for ii in 0..nstages {
                for i in 0..n {
                    retry |= symbi_xpu::with_device(devices[i], || {
                        tiles[i].level_stage(0, ii, gdt, 0.0)
                    });
                }
                if retry {
                    break;
                }
                migrate_level_tracers(tiles, 0);
                exchange_level_halos(tiles, 0, &root_order, counts, devices, transport);
            }
            if retry {
                // handled by the common collective rollback at the end of the attempt.
            } else {
                spawn_decomposed_injections(tiles, &root_order, 0);
                // root emf bookkeeping (mhd; no-op for hydro + unrefined tiles).
                for i in 0..n {
                    symbi_xpu::with_device(devices[i], || tiles[i].level_tail_emf(0, gdt));
                }
                // the root excise, in the same position the uni-grid tail puts it (after the emf
                // bookkeeping). the rest of the tail — viscous, horizon ledger, penalize — belongs to the
                // finest level and is driven with the fine subcycle; the excise runs per level, because
                // the excised region is owned by whichever level contains it and a refined root still
                // owns its core.
                for i in 0..n {
                    symbi_xpu::with_device(devices[i], || tiles[i].level_tail_excise(0));
                }
                // rigid walls also act on uncovered root cells when their shape crosses the refined
                // patch boundary. covered root writes are replaced by restriction after the fine
                // subcycle; accretion remains absent from root proxies.
                for i in 0..n {
                    symbi_xpu::with_device(devices[i], || {
                        let level = &tiles[i].levels[0];
                        let has_rigid = level
                            .state
                            .immersed
                            .as_ref()
                            .is_some_and(|immersed| immersed.bodies.rigid_count() > 0);
                        if has_rigid {
                            prof("penalize", || level.kernels.penalize(&level.state, gdt));
                        }
                    });
                }
                // the fine subcycle, decomposed across the refined tiles: ratio substeps, each fine substep
                // driven stage-by-stage with a fine halo exchange (a no-op for a tile-local 1x1 sub-grid).
                if let Some(fg) = &fine {
                    let fine_dt = gdt / RATIO as f64;
                    'fine_subcycle: for sub in 0..RATIO {
                        let alpha = sub as f64 / RATIO as f64;
                        for (k, &i) in fg.order.iter().enumerate() {
                            symbi_xpu::with_device(fg.devices[k], || {
                                tiles[i].prolong_cf(1, alpha);
                                tiles[i].levels[1]
                                    .kernels
                                    .ghost_fill(&tiles[i].levels[1].state);
                            });
                        }
                        // fill the fine cut halo before the fine flux reads it.
                        exchange_level_halos(
                            tiles,
                            1,
                            &fg.order,
                            fg.counts,
                            &fg.devices,
                            transport,
                        );
                        for (k, &i) in fg.order.iter().enumerate() {
                            symbi_xpu::with_device(fg.devices[k], || {
                                tiles[i].level_step_begin(1, fine_dt)
                            });
                        }
                        for jj in 0..nstages {
                            for (k, &i) in fg.order.iter().enumerate() {
                                retry |= symbi_xpu::with_device(fg.devices[k], || {
                                    tiles[i].level_stage(1, jj, fine_dt, alpha)
                                });
                            }
                            if retry {
                                break 'fine_subcycle;
                            }
                            migrate_level_tracers(tiles, 1);
                            exchange_level_halos(
                                tiles,
                                1,
                                &fg.order,
                                fg.counts,
                                &fg.devices,
                                transport,
                            );
                        }
                        spawn_decomposed_injections(tiles, &fg.order, 1);
                        for (k, &i) in fg.order.iter().enumerate() {
                            retry |= symbi_xpu::with_device(fg.devices[k], || {
                                tiles[i].level_step_tail(1, fine_dt, alpha)
                            });
                        }
                        if retry {
                            break 'fine_subcycle;
                        }
                    }
                    // tile-local restrict + flux/emf reflux + c2p + ghost on each refined tile's root.
                    if !retry {
                        for (k, &i) in fg.order.iter().enumerate() {
                            symbi_xpu::with_device(fg.devices[k], || {
                                tiles[i].level_restrict_reflux(0, 0.0)
                            });
                        }
                    }
                }
                // the root clock; the root post-step carries no mesh motion.
                if !retry {
                    for i in 0..n {
                        symbi_xpu::with_device(devices[i], || {
                            let s = &mut tiles[i].levels[0].state;
                            (s.time, s.iteration) = advance_clock(s.time, s.iteration, gdt);
                        });
                    }
                }
            }
            if !retry {
                break 'attempt;
            }
            // Collective rollback: every tile returns to the same root-step entry before any halo is
            // exchanged or the reduced timestep is replayed.  This includes host-side tracer/body
            // state as well as each kernel set's conserved and magnetic retry snapshot.
            for i in 0..n {
                for level in &tiles[i].levels {
                    if level.kernels.fofc_active() {
                        symbi_xpu::with_device(devices[i], || {
                            level.kernels.restore_step(&level.state)
                        });
                    }
                }
                tiles[i].restore_retry_sidecars(&retry_sidecars[i]);
            }
            drain_devices::<Mem>(devices);
            exchange_level_halos(tiles, 0, &root_order, counts, devices, transport);
            if let Some(fg) = &fine {
                exchange_level_halos(tiles, 1, &fg.order, fg.counts, &fg.devices, transport);
            }
            gdt = symbi_sim::driver::retry_timestep(gdt, t)
                .unwrap_or_else(|err| panic!("{}", err.detail));
        }
    gdt
}

/// the accepted-step tail of the decomposed march: the per-step immersed-body bookkeeping
/// (each tile's finest level reduces its local feedback partials — the tile finest interiors
/// partition the sink region — the deltas sum across tiles into the true global reaction,
/// and the identical global delta + prescribed advance applies to every tile's bodies, so
/// all tiles stay in lockstep), then the root halo refresh for the next step's stage 0.
fn decomposed_root_post<R, const NDIM: usize, const DOF: usize, M, E, S, Mem, K, T>(
    tiles: &mut [Hierarchy<R, NDIM, DOF, M, E, S, Mem, K>],
    counts: [usize; NDIM],
    devices: &[i32],
    transport: &T,
    root_order: &[usize],
    gdt: f64,
    t: f64,
)
where
    R: Regime<f64, NDIM> + Copy,
    M: Metric<f64, NDIM> + Copy + Send + Sync,
    E: Eos<f64> + Copy + Send + Sync,
    S: ExecutionSpace,
    Mem: MemorySpace + Sync,
    K: KernelSet<NDIM, DOF, Mem, f64>,
    T: HaloTransport,
{
    let n = tiles.len();
        // per-step immersed-body bookkeeping, the hierarchy form of the flat decomposed
        // body step: each tile's finest level reduces its local feedback partials (the
        // tile finest interiors partition the sink region — coarser levels carry a
        // gravity-only proxy, so the drain stays on the finest), the deltas are summed
        // across tiles into the true global reaction, and the identical global delta +
        // prescribed advance is applied to every tile's bodies. identical input ->
        // identical body state, so all tiles stay in lockstep.
        if tiles.iter().any(|h| h.levels[0].state.has_bodies()) {
            for i in 0..n {
                symbi_xpu::with_device(devices[i], || tiles[i].finest_body_feedback(gdt));
            }
            drain_devices::<Mem>(devices);
            let mut global: Vec<symbi_ib::BodyDelta<f64, NDIM>> = Vec::new();
            for i in 0..n {
                for d in tiles[i].take_body_deltas() {
                    match global.iter_mut().find(|g| g.idx == d.idx) {
                        Some(g) => {
                            g.force_delta = g.force_delta + d.force_delta;
                            g.torque_delta = g.torque_delta + d.torque_delta;
                            g.mass_delta += d.mass_delta;
                        }
                        None => global.push(d),
                    }
                }
            }
            let t_now = t + gdt;
            let any_refined = tiles.iter().any(|h| h.levels.len() > 1);
            for i in 0..n {
                symbi_xpu::with_device(devices[i], || {
                    tiles[i].apply_global_body_deltas(&global, gdt, t_now)
                });
                assert!(
                    !(any_refined
                        && tiles[i].levels.len() == 1
                        && tiles[i].accretion_overlaps_root()),
                    "refinement x decomposition: a sink sphere overlaps UNREFINED tile {i} — the \
                     refined patch must cover the sink on every tile it touches"
                );
            }
        }
        // the restrict/c2p recomputed root prim over the allocated domain (incl. the cut halo from
        // stale halo cons, and the cut-adjacent reflux ghosts); refresh for the next step's stage 0.
        exchange_level_halos(tiles, 0, &root_order, counts, devices, transport);
}

/// the decomposed hierarchy driver (refinement x decomposition): lockstep-advance N
/// per-tile hierarchies, decomposing the root and the first fine level. the root stages run with a
/// root halo exchange between them (rk2 corrector reads each neighbor's stage-1 update); the fine
/// subcycle is driven here so its stages can exchange the fine halos between fine tiles when a
/// patch spans a tile cut. for a tile-local patch the fine sub-grid is 1x1, so the fine exchange
/// is a no-op and this reduces to the tile-local case exactly. the flux/emf reflux registers stay
/// tile-local (a coarse cell + the fine cells at its face are co-located; any register write to a
/// cut-adjacent ghost is overwritten by the next root exchange). global dt = min over tiles of
/// `root_cfl_dt()`. `decomposed == monolithic` holds for both tile-local and cut-spanning patches.
///
/// levels 2+ are advanced tile-locally (inside the level-1 fine tile, via the recursive
/// `level_step_tail(1)`); a patch spanning a cut on a level >= 2 lies outside this driver's scope.
/// hydro only in the root post-step (the root clock is advanced directly; mesh motion and immersed
/// bodies belong to the single-grid path). `on_checkpoint(iteration, time, &tiles)` fires every
/// `interval` root steps and once at the end (devices drained first).
#[allow(clippy::too_many_arguments)]
pub fn evolve_hierarchy_decomposed<R, const NDIM: usize, const DOF: usize, M, E, S, Mem, K, T, F>(
    tiles: &mut [Hierarchy<R, NDIM, DOF, M, E, S, Mem, K>],
    counts: [usize; NDIM],
    devices: &[i32],
    transport: &T,
    ts: Timestepping,
    start_time: f64,
    t_final: f64,
    interval: u64,
    mut on_checkpoint: F,
) where
    R: Regime<f64, NDIM> + Copy,
    M: Metric<f64, NDIM> + Copy + Send + Sync,
    E: Eos<f64> + Copy + Send + Sync,
    S: ExecutionSpace,
    Mem: MemorySpace + Sync,
    K: KernelSet<NDIM, DOF, Mem, f64>,
    T: HaloTransport,
    F: FnMut(u64, f64, &[Hierarchy<R, NDIM, DOF, M, E, S, Mem, K>]) -> ControlFlow<()>,
{
    let n = tiles.len();
    // the finest-level tail owns the IBM surface physics, and the refined-
    // decomposed step drives the fine tail alone, so a refined run needs at
    // least one tile carrying a fine patch for its bodies to be penalized at
    // all — a silent physics loss otherwise, refused here.
    assert!(
        tiles.iter().any(|h| h.levels.len() > 1)
            || tiles.iter().all(|h| !h.levels[0].state.has_bodies()),
        "refined-decomposed: no tile built a fine level while immersed bodies are active; \
         the finest-level penalize would silently never run",
    );
    debug_assert_eq!(n, devices.len(), "tiles/devices length mismatch");
    debug_assert_eq!(
        ts.stages().len(),
        tiles[0].levels[0].state.timestepping.stages().len(),
        "the driver's declared timestepping matches the tiles' own"
    );
    // the tiles' own root clock is the march's time authority; the caller's start_time
    // is the same value by construction (the restart path restores both from the
    // checkpoint), asserted here so a drifting caller fails loudly.
    debug_assert!(
        (tiles[0].levels[0].state.time - start_time).abs()
            <= start_time.abs().max(1.0) * 1e-12,
        "start_time disagrees with the tiles' root clock"
    );
    // the decomposed march is the multi-tile case of the shared driver, which owns the
    // watchdog screening, the step-panic catch, the crash report, and the observer
    // cadence. the callback keeps its historical clock: iterations counted from this
    // call's entry, alongside the root time the tiles carry.
    let iter0 = tiles[0].levels[0].state.iteration;
    evolve_tiles(
        tiles,
        Some((devices, counts, transport)),
        t_final,
        interval,
        |march_tiles| {
            let s = &march_tiles[0].levels[0].state;
            on_checkpoint(s.iteration - iter0, s.time, march_tiles)
        },
    );
}

fn wb_ghost_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("SYMBI_WB_GHOST")
            .map(|v| v != "0")
            .unwrap_or(true)
    })
}

fn wb_band_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("SYMBI_WB_BAND")
            .map(|v| v != "0")
            .unwrap_or(true)
    })
}
