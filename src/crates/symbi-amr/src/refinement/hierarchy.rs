// =============================================================================
// hierarchy.rs
//
// the static-mesh-refinement (SMR) hierarchy. each
// level is a complete SimStateGeneric + its KernelSet; the hierarchy adds only
// inter-level coordination: recursive berger-oliger subcycling with
// time-interpolated coarse-fine ghost prolongation, conservative restriction,
// and flux-register refluxing. the single-level engine is untouched —
//
// SINGLE-COVERAGE CAP (this is SMR): each level refines exactly
// ONE box (`coverage: Option<Domain>`), with ONE FluxRegister per coarse-fine
// level-pair. the refined region is fixed at setup and is not re-flagged from the
// solution; there is no patch graph and no berger-rigoutsos clustering.
// multi-patch adaptive refinement (a level as a disjoint cover of Domains) is
// not implemented.
//
// advance_level re-sequences the SSP stage loop (sim/evolve.rs::step) so the
// register accumulation slots between flux() and the stage update.
//
// levels share ABSOLUTE index space: a fine level covering coarse cells
// [cov_lo, cov_hi) lives at fine indices [2*cov_lo, 2*cov_hi), and every level
// keeps the same global physical origin — `geom.centroid` is correct on every
// level with the same formula, and no coverage-relative translation exists.
//
// the bit-for-bit gate: a 1-level hierarchy must reproduce evolve() exactly
// (crates/symbi/tests/refine_hierarchy.rs); the conservation gate: a 2-level
// static nesting conserves the composite-grid totals to machine precision
// (crates/symbi/tests/refine_conservation.rs).
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
use super::flux_register::FluxRegister;
use super::transfer::{
    ProlongOrder, bcell_from_bface_region, bface_cf_halo_slabs, cf_ghost_slabs, copy_field,
    prolong_face_field, prolong_field, prolong_prims_swept,
    restrict_bface, restrict_cell_field, ProlongSweepScratch,
    restrict_cons,
};
use symbi_sim::stage::{fold_stage, HookPoint, StageArgs};
use symbi_sim::driver::{check_dt_or_panic, evolve_bodies, prof, stage_time_fractions};
use symbi_sim::hydro_ops::scan_c2p_errors;
use symbi_sim::decomp::{drain_devices, exchange_grid, flatten, unflatten, HaloTransport};
use symbi_sim::state::{
    Boundaries, BoundaryType, FieldDecimation, FieldKind, FieldStore, PrimFieldsGeneric,
    SimStateGeneric, Timestepping, array_field_zeros, axis_name,
};
use symbi_sim::substrate_seam::KernelSet;
use std::ops::ControlFlow;

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
    /// traffic. allocated with prim_old (None on the finest); reused every
    /// call, never allocated in the step loop.
    pub prim_lerp: Option<PrimFieldsGeneric<NDIM, DOF, Mem>>,
    /// cell-centered B at this level's step start (mhd only) — the magnetic
    /// counterpart of prim_old for the finer level's ghost prolongation.
    pub bcell_old: Option<[Field<f64, NDIM, Mem>; NDIM]>,
    /// staggered face B at this level's step start (mhd only) — feeds the
    /// finer level's bface transverse-halo prolongation (per-component
    /// staggered domains, cloned from this level's bface).
    pub bface_old: Option<[Field<f64, NDIM, Mem>; NDIM]>,
    /// per-slab intermediates of the axis-split prolongation INTO this level,
    /// in `cf_ghost_slabs` order. SMR slabs are static, so
    /// the shapes are too: lazily allocated on the first prolongation, reused
    /// every call (the step loop allocates nothing). None-equivalent
    /// (uninitialized) on the root.
    pub prolong_sweep: std::sync::OnceLock<Vec<ProlongSweepScratch<NDIM, DOF, Mem>>>,
    /// the region of THIS level covered by the next finer level, in absolute
    /// indices of this level. None on the finest. single-coverage cap: exactly
    /// ONE refined box per level (this is static refinement / SMR — no disjoint-cover
    /// `Vec<Domain>`, no clustering).
    pub coverage: Option<Domain<NDIM>>,
}

// =============================================================================
// hierarchy
// =============================================================================

/// static-refinement hierarchy: levels[0] = coarsest, levels[n-1] = finest.
/// a fatal CFL crash: the wave speed went NaN or collapsed (an unphysical c2p — e.g. V -> 1 near a
/// boundary), so the next dt is NaN / non-positive / blown up. the evolve loop stops at the LAST
/// computed state (no further advance) and the driver snapshots a `.crashed` checkpoint + reports
/// it, so the crash is never masked by clamping dt to t_final and "finishing" on garbage.
#[derive(Clone, Copy, Debug)]
pub struct CrashReport {
    pub iter: u64,
    pub time: f64,
    pub dt_cfl: f64,
    pub dt_prev: f64,
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
    /// coarse-fine prolongation order (one above the evolution reconstruction).
    pub prolong_order: ProlongOrder,
    /// flux_registers[ll] corrects the interface between level ll and ll+1.
    pub flux_registers: Vec<FluxRegister<NDIM, DOF, Mem>>,
    /// emf_registers[ll] corrects the coarse bface at the same interface (mhd
    /// pairs only; None for pure hydro).
    pub emf_registers: Vec<Option<EmfRegister<NDIM, Mem>>>,
    /// set by `step_root` when the cfl dt is fatal (NaN / non-positive / a sudden blowup from a
    /// collapsed wave speed). `Some` halts the march at the last computed state; the driver writes a
    /// `.crashed` checkpoint and reports it, halting before advancing past t_final on garbage.
    pub crash: Option<CrashReport>,
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
    /// decimate the hierarchy to a screen-sized density heatmap, compositing the
    /// nested refinement levels: each root cell descends to the FINEST level whose
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
        // single grid: no coverage to descend, reuse the plain per-state decimation.
        if self.levels.len() <= 1 {
            return self.levels[0].state.field_slice_oriented(max_dim, index, orient, zoom);
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
            symbi_algebra::Space { name: sp.name, lo: mid - span / 2, hi: mid - span / 2 + span }
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

    /// resolve a ROOT-level index to the finest level covering it and read the
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
            idx = std::array::from_fn(|ax| {
                finer.spaces[ax].lo + (idx[ax] - cov.spaces[ax].lo) * 2
            });
            lvl += 1;
        }
        self.levels[lvl].state.field_value(idx, kind)
    }

    /// seed every fine level's interior from its parent by CONSERVATIVE
    /// prolongation at the hierarchy's `prolong_order` (= interior reconstruction
    /// order + 1, the same order the coarse-fine ghost prolongation uses). the IC
    /// fill for a hierarchy whose coarse level was seeded but whose fine levels are
    /// still empty — a coarse-only IC, e.g., a python-driven `prim_gen`. each
    /// conserved component is prolonged coarse -> fine interior; the fine levels
    /// then refine the solution as they evolve.
    ///
    /// note: hydro-complete + mhd cell-centered B. the staggered fine `bface` is
    /// NOT seeded here — mhd refinement needs face prolongation, wired alongside
    /// the mhd AMR path.
    pub fn seed_fine_from_coarse(&self) -> symbi_xpu::Result<()> {
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
            // mhd: cell-centered B + the staggered FACES (divergence-free
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
        }
        Ok(())
    }

    /// a 1-level hierarchy: the degenerate case that must reproduce evolve()
    /// bit-for-bit.
    pub fn single(state: SimStateGeneric<R, NDIM, DOF, M, E, S, Mem>, kernels: K) -> Self {
        Hierarchy {
            levels: vec![LevelData {
                state,
                kernels,
                prim_old: None,
                prim_lerp: None,
                prolong_sweep: std::sync::OnceLock::new(),
                bcell_old: None,
                bface_old: None,
                coverage: None,
            }],
            prolong_order: ProlongOrder::Plm,
            flux_registers: Vec::new(),
            emf_registers: Vec::new(),
            crash: None,
        }
    }

    /// build a statically nested hierarchy: region k refines level k at ratio 2.
    /// fine levels live in absolute indices (interior at 2x the covered cells)
    /// with the SAME global physical origin; coarse-fine faces get
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
            bcell_old: None,
            bface_old: None,
            coverage: None,
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
                bcell_old: None,
                bface_old: None,
                coverage: None,
            });
        }

        let mut flux_registers = Vec::with_capacity(levels.len().saturating_sub(1));
        let mut emf_registers = Vec::with_capacity(levels.len().saturating_sub(1));
        for ll in 0..levels.len().saturating_sub(1) {
            flux_registers.push(FluxRegister::new(
                levels[ll].coverage.as_ref().unwrap(),
                &levels[ll].state.geom.interior,
                levels[ll].state.fields.cons.has_energy(),
            )?);
            emf_registers.push(if levels[ll].state.fields.mhd.is_some() {
                Some(EmfRegister::new(
                    levels[ll].coverage.as_ref().unwrap(),
                    &levels[ll].state.geom.interior,
                )?)
            } else {
                None
            });
        }

        Ok(Hierarchy {
            levels,
            prolong_order,
            flux_registers,
            emf_registers,
            crash: None,
        })
    }

    /// attach an immersed body collection to every level: the FINEST level
    /// carries the full collection (the sink and the accretion diagnostics
    /// have a single owner — the resolution truth), every coarser level a
    /// gravity-only proxy of each body (same mass / softening / motion, sink
    /// disabled), so covered coarse cells are never sink-drained under the
    /// restriction. body motion advances once per root step on the finest and
    /// is synced outward. the sink region must lie inside the finest level —
    /// asserted every step as the bodies move.
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

    /// attach per-body immersed-boundary shapes to the FINEST level, which owns the full
    /// (wall / accreting) bodies; coarser levels carry gravity-only proxies with no surface, so
    /// they need no shape. `None` entries keep the analytic sphere.
    pub fn attach_body_shapes(&mut self, shapes: Vec<Option<symbi_ib::sdf::SdfExpr<f64, 3>>>) {
        let finest = self.levels.len() - 1;
        self.levels[finest].state.attach_body_shapes(shapes);
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

    /// the sink containment invariant, clipped to this hierarchy's root: only the part of
    /// the sink sphere that overlaps the root interior must lie inside the finest level —
    /// on a decomposed tile the sphere may span cuts (each owning tile drains its own
    /// cells), and a tile the sphere does not touch carries no constraint. a 1-level
    /// hierarchy has no coarse-fine boundary to straddle; the decomposed driver separately
    /// forbids sphere overlap on an UNREFINED tile of a refined run (a coarse drain the
    /// refluxing does not protect).
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
            // sphere-box overlap against the ROOT interior, per axis.
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
    /// decomposed driver's guard input: on a refined run an UNREFINED tile must not
    /// drain (its coarse cells are outside every reflux-protected fine region).
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

    /// the LOCAL half of the decomposed body step: reduce this tile's finest-level
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
            prof("body_feedback", || finest.kernels.body_feedback(&finest.state, dt));
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

    /// the GLOBAL half of the decomposed body step: apply the cross-tile-summed deltas +
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
        mut callback: impl FnMut(&Self) -> std::ops::ControlFlow<()>,
    ) -> symbi_xpu::Result<()> {
        // homologous mesh motion is single-grid only: the hierarchy's flux
        // registers and transfer operators have no a(t) bookkeeping yet. a
        // 1-level hierarchy has no registers/transfer, so motion is safe there
        // (it reproduces the single-grid evolve); only REFINED runs are forbidden.
        assert!(
            self.levels.len() == 1
                || self
                    .levels
                    .iter()
                    .all(|l| { l.state.motion.a_dot == 0.0 && l.state.motion.a == 1.0 }),
            "refinement: mesh motion is uni-grid only (the registers carry no scale factor)"
        );
        self.init_levels();
        let mut last_cb = self.levels[0].state.iteration;
        while self.levels[0].state.time < t_final {
            self.step_root(t_final);
            // a crashed step recorded the crash WITHOUT advancing the clock: let the observer
            // snapshot the `.crashed` checkpoint + report it, then stop the march (don't spin on the
            // frozen time, and don't clamp dt to t_final on garbage).
            if self.crash.is_some() {
                symbi_substrate::regimes::substrate_gpu::device_sync::<Mem>();
                let _ = callback(self);
                return Ok(());
            }
            if self.levels[0].state.iteration.saturating_sub(last_cb) >= interval {
                last_cb = self.levels[0].state.iteration;
                symbi_substrate::regimes::substrate_gpu::device_sync::<Mem>();
                // a `Break` from the observer (e.g., a caught signal) stops the
                // march early; the observer has already snapshotted state for
                // restart. drain the queue so the host read in the caller is
                // coherent, then return.
                if callback(self).is_break() {
                    symbi_substrate::regimes::substrate_gpu::device_sync::<Mem>();
                    return Ok(());
                }
            }
        }
        // the caller reads fields next: drain the device queue (the
        // host-read barrier; no-op on a host backend).
        symbi_substrate::regimes::substrate_gpu::device_sync::<Mem>();
        let _ = callback(self);
        Ok(())
    }

    /// run the one-time IC preparation WITHOUT advancing time: bcell-from-bface,
    /// c2p to populate the primitive buffer from the seeded conserved state, the
    /// coarse-fine prolong, and the ghost fill. the drivers call this internally
    /// at evolve start; a caller that snapshots state at t=0 (the binding's
    /// initial-condition checkpoint) must call it first, else that snapshot
    /// captures the ZEROED primitive + cell-centered-B scratch buffers. idempotent.
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

    /// the cfl-limited root dt this hierarchy would take (UNCLAMPED by t_final): the min over every
    /// level of `cfl(level) * RATIO^level` (covered coarse cells are conservative averages, so a
    /// fast fine-only feature is diluted out of the root cfl; level l subcycles RATIO^l times, so
    /// its limit enters scaled by RATIO^l). exposed for the DECOMPOSED driver: it takes the global
    /// min of this across tiles, then drives each tile with `evolve(t + global_dt)` -- since the
    /// global dt is the min, each tile's internal `dt_cfl.min(global_dt)` collapses to global_dt,
    /// giving a lockstep root step without a separate dt-injection path.
    pub fn root_cfl_dt(&self) -> f64 {
        // same full-grid wave-speed pass as `step_root`; instrumented under the same phase name so a
        // decomposed run attributes it identically.
        let dt_cfl = prof("cfl", || {
            let mut dt = f64::INFINITY;
            for (ll, lvl) in self.levels.iter().enumerate() {
                let scale = RATIO.pow(ll as u32) as f64;
                dt = dt.min(lvl.kernels.cfl(&lvl.state) * scale);
            }
            dt
        });
        // the user clamp (max_dt > 0): pins the dt sequence across runs whose CFL
        // estimators differ. applied AFTER the raw-cfl crash heuristics elsewhere.
        let clamp = self.levels[0].state.max_dt;
        if clamp > 0.0 { dt_cfl.min(clamp) } else { dt_cfl }
    }

    /// one root step: cfl-limited dt (clamped to t_final), the recursive level
    /// advance, then the root clock + body state. EVERY level limits the root
    /// step — covered coarse cells are conservative averages of fine data, so
    /// a fast feature resolved only on the fine level is diluted out of the
    /// root's own cfl; level l subcycles RATIO^l times, so its limit enters
    /// scaled by RATIO^l (tests/refine_per_level_cfl.rs pins this).
    fn step_root(&mut self, t_final: f64) {
        // the per-root-step wave-speed pass + global min reduction. instrumented because it is a
        // FULL-GRID read of prim on every level, once per step, and sits OUTSIDE the substage loop:
        // at a small domain / high step count it is a large fraction of the step that no per-phase
        // timing would otherwise attribute.
        let dt_cfl = prof("cfl", || {
            let mut dt = f64::INFINITY;
            for (ll, lvl) in self.levels.iter().enumerate() {
                let scale = RATIO.pow(ll as u32) as f64;
                dt = dt.min(lvl.kernels.cfl(&lvl.state) * scale);
            }
            dt
        });
        // a crashed state must HALT the run and must not be masked by the `t_final` clamp below. the clamp
        // `dt_cfl.min(t_final - time)` silently replaces a NaN dt with the remaining time (f64::min
        // returns the non-NaN operand) AND clamps a collapsed-wave-speed BLOWUP (an unphysical c2p
        // cell — e.g. V->1 at the inner boundary — drives the cfl speed -> 0, so dt -> huge) down to
        // the remaining time; either way the run would "finish" at t_final on garbage. a physical
        // flow grows dt SMOOTHLY (cfl-limited), so detect a crash as: NaN / non-positive, or a sudden
        // >1000x one-step jump in the RAW cfl dt. a genuinely static state (dt_cfl = +inf) only
        // arises from the rest state at step 0 (dt_prev = 0, skipped) -> the clamp takes the run end.
        let (iter, time, dt_prev) = {
            let r = &self.levels[0].state;
            (r.iteration, r.time, r.dt) // dt_prev = 0.0 before the first step
        };
        let crashed = dt_cfl.is_nan()
            || dt_cfl <= 0.0
            || (dt_prev > 0.0 && dt_cfl > 1.0e3 * dt_prev);
        if crashed {
            // record the crash + STOP without advancing: the evolve loop reports it and the driver
            // snapshots `.crashed.h5` from this (last computed) state and stops, without panicking or
            // marching past t_final on garbage.
            self.crash = Some(CrashReport { iter, time, dt_cfl, dt_prev });
            return;
        }
        let root = &mut self.levels[0];
        let user_clamp = root.state.max_dt;
        let dt_cfl = if user_clamp > 0.0 { dt_cfl.min(user_clamp) } else { dt_cfl };
        let dt = dt_cfl.min(t_final - root.state.time);
        check_dt_or_panic(dt, root.state.iteration, root.state.time);
        root.state.dt = dt;

        self.advance_level(0, dt, 0.0);

        // horizon excision, ONCE per step after the full RK combination (the same
        // point the single-grid loop applies it): overwrite the causally
        // disconnected cells inside the excision sphere with a zero-gradient
        // viscous / excise / horizon-ledger / penalize now run in the finest
        // level's tail (level_step_tail), in the uni-grid driver's exact order.

        let root = &mut self.levels[0];
        // homologous linear advance ONLY when there is no traced motion law.
        if root.state.motion_law.is_none() && root.state.motion.homologous {
            root.state.motion.a += root.state.motion.a_dot * dt;
        }
        root.state.time += dt;
        // expression motion: set a / a_dot EXACTLY at the new time (for output and the next root
        // step's cfl + stages); a constant-rate extrapolation would overshoot a
        // decelerating mesh.
        let tnew = root.state.time;
        if let Some((a, ad)) = root
            .state
            .motion_law
            .as_ref()
            .map(|ml| (ml.a_at(tnew), ml.adot_at(tnew)))
        {
            root.state.motion.a = a;
            root.state.motion.a_dot = ad;
        }
        root.state.iteration += 1;

        // body feedback + motion: the FINEST level owns the sink and the
        // diagnostics; advance the (prescribed) motion there once per root
        // step at the root dt, then sync positions/velocities outward so
        // every level's gravity sees the same bodies.
        if self.levels[0].state.has_bodies() {
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
                // body: force/torque/mass/energy) — a real per-step cost that must
                // appear in the profile, so it does not hide in the wall-vs-instrumented gap.
                prof("body_feedback", || finest.kernels.body_feedback(&finest.state, dt));
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

        // lagrangian tracers, ONCE per root step against the post-step
        // primitive velocity — the same slot and order as the uni-grid driver.
        // single-level only: a refined run's root velocity is coarse under the
        // fine patches, so tracer advection there silently degrades — refuse.
        if self.levels[0].state.has_tracers() {
            assert!(
                self.levels.len() == 1,
                "tracers on a refined hierarchy are not wired (root-level advection \
                 would silently use coarse velocities under the fine patches)",
            );
            let root = &mut self.levels[0];
            prof("tracers", || symbi_sim::tracers::advance_tracers(&mut root.state));
        }
    }

    /// advance one level by dt, then subcycle the finer level, restrict, and
    /// reflux. `alpha0` is this level's substep start as a fraction of the
    /// PARENT's step; 0.0 on the root. the coarse-fine ghosts are
    /// STAGE-CORRECT in time: the prolong feeding stage k reconstructs at the
    /// shu-osher stage time `alpha0 + c_k / RATIO` (see
    /// stage_time_fractions), which restores second-order temporal coupling
    /// at the interface — substep-start-frozen ghosts measurably collapse the
    /// boundary to first order (tests/refine_temporal_convergence.rs). the stage
    /// loop mirrors sim/evolve.rs::step — the bit-for-bit gate holds the two
    /// in lockstep.
    fn advance_level(&mut self, level: usize, dt: f64, alpha0: f64) {
        self.level_step_begin(level, dt);
        let n = self.levels[level].state.timestepping.stages().len();
        for ii in 0..n {
            self.level_stage(level, ii, dt, alpha0);
        }
        self.level_step_tail(level, dt, alpha0);
    }

    /// step prologue: snapshot this level's prims (for the finer level's time-interpolated ghost
    /// prolongation) + the rk u_n snapshot. extracted from `advance_level` (which calls begin /
    /// stage* / tail in order -- bit-for-bit unchanged) so the DECOMPOSED root driver can drive the
    /// root stages stage-by-stage with a root halo exchange BETWEEN stages (rk2-root requires the
    /// corrector to read each neighbor's stage-1 update, exactly like the single-level exchange).
    pub fn level_step_begin(&mut self, level: usize, dt: f64) {
        let has_finer = level + 1 < self.levels.len();
        if has_finer {
            prof("refine_save_prim", || save_prim_old(&self.levels[level]));
        }
        self.levels[level].state.dt = dt;
        let stages = self.levels[level].state.timestepping.stages();
        if stages.len() > 1 {
            let l = &self.levels[level];
            prof("snapshot", || l.kernels.snapshot(&l.state));
        }
    }

    /// one SSP stage `ii` of level `level` -- the body of `advance_level`'s stage loop. `alpha0` is
    /// this level's substep start as a fraction of the parent step (0 on the root). pure extraction.
    pub fn level_stage(&mut self, level: usize, ii: usize, dt: f64, alpha0: f64) {
        // the phase sequence is THE shared table (symbi-sim::stage); this
        // driver contributes only its two structural interleaves through the
        // hook points — flux-register sampling (on the high-order fluxes,
        // before a fofc splice) and the coarse-fine ghost re-prolongation.
        // everything the hooks touch is reached by shared reference (the
        // registers and field writes go through interior mutability), so the
        // fold's borrow of the level state never conflicts.
        let has_finer = level + 1 < self.levels.len();
        let has_coarser = level > 0;
        let stages = self.levels[level].state.timestepping.stages();
        let n = stages.len();
        let weights = flux_weights(stages);
        let stage_time = stage_time_fractions(stages);
        let (a0, ac) = stages[ii];

        let this = &*self;
        let l = &this.levels[level];
        let mut hook = |hp: HookPoint| match hp {
            HookPoint::AfterFlux => {
                prof("refine_flux_reg", || {
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
                            }
                        } else {
                            if ii == 0 {
                                reg.zero();
                            }
                            let geo = l.state.geom.block_geometry(l.state.physics.metric);
                            for dd in 0..NDIM {
                                reg.accumulate_coarse(&l.state.fields.flux, &geo, dd, weights[ii] * dt);
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
                            }
                        }
                    }
                });
            }
            HookPoint::BeforeGhostFill => {
                // c2p over the full allocated domain recomputed the coarse-fine
                // prim ghosts from stale cons; re-prolong them at the time of
                // the state entering the NEXT stage before the physical fill
                // reads corners.
                if has_coarser {
                    this.prolong_cf(level, alpha0 + stage_time[ii] / RATIO as f64);
                }
            }
        };
        fold_stage(
            &l.state,
            &l.kernels,
            StageArgs { dt, a0, ac, stage: ii, n_stages: n, allow_elision: true },
            &mut hook,
        );
    }

    /// step epilogue: emf bookkeeping (mhd) + the finer-level subcycle + restrict + reflux + the
    /// level clock. pure extraction from `advance_level`. for the DECOMPOSED root the driver calls
    /// this AFTER the (exchanged) root stages; the fine subcycle here is tile-local (the patch lives
    /// inside one tile), so it reuses the recursive `advance_level` on the finer level unchanged.
    pub fn level_step_tail(&mut self, level: usize, dt: f64, alpha0: f64) {
        let has_finer = level + 1 < self.levels.len();
        self.level_tail_emf(level, dt);
        // the IBM surface physics on the FINEST level, once per its substep,
        // AFTER the full RK combination (receipt == removal) and BEFORE the
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
        // EXCISE runs on EVERY level, not only the finest: the excised (causally-disconnected) region
        // is owned by whichever level contains it, and the refinement request gate forbids a finer
        // patch overlapping it, so a REFINED ROOT still owns — and must excise — its singular core.
        // gating this on `!has_finer` silently skipped excision wherever the excised region sat under
        // a refined level (the root of an off-core refined run), leaving the core evolving. ordered
        // after viscous (bit-identical on the finest, where both run); a no-op at excision_radius = 0,
        // so unexcised and uni-grid runs are unchanged. the excised fill survives the finer subcycle +
        // restriction below because the excised region is never a covered (finer-patch) region.
        {
            let l = &self.levels[level];
            l.kernels.excise(&l.state);
        }
        if !has_finer {
            let horizon = self.levels[level].state.immersed.as_ref().and_then(|im| {
                im.bodies.bodies().iter().enumerate().find_map(|(i, b)| match b.kind {
                    symbi_ib::BodyKind::Horizon { diagnostic_radius, .. } => {
                        Some((i, diagnostic_radius))
                    }
                    _ => None,
                })
            });
            if let Some((idx, r_d)) = horizon {
                let l = &self.levels[level];
                let (mdot, edot) =
                    prof("horizon_accretion", || l.kernels.horizon_accretion(&l.state, r_d));
                if let Some(im) = self.levels[level].state.immersed.as_mut() {
                    if let symbi_ib::BodyKind::Horizon {
                        total_accreted_mass, total_accreted_energy, mdot: m, edot: e, ..
                    } = &mut im.bodies.get_mut(idx).kind
                    {
                        *total_accreted_mass += mdot * dt;
                        *total_accreted_energy += edot * dt;
                        *m = mdot;
                        *e = edot;
                    }
                }
            }
            let l = &self.levels[level];
            if l.state.has_bodies() {
                prof("penalize", || l.kernels.penalize(&l.state, dt));
            }
        }
        if has_finer {
            self.level_subcycle(level, dt);
            self.level_restrict_reflux(level, alpha0);
        }
        self.level_clock(level, dt);
    }

    /// emf register bookkeeping (mhd) after the stage loop: the efield buffers hold the EFFECTIVE
    /// per-step EMF (post_godunov wrote the rk2 time-average in place before the single curl; euler
    /// keeps the raw stage EMF) -- sample it into the coarse-side register (a finer level exists)
    /// and the fine-side register (a coarser level exists). no-op for hydro. PURE EXTRACTION from
    /// level_step_tail so the DECOMPOSED driver can interpose a fine-level halo exchange between the
    /// root stages and the (driver-controlled) fine subcycle.
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

    /// the MONOLITHIC finer-level subcycle: RATIO substeps of level `level+1`, each prolonged from
    /// this level's time-interpolated prims. the DECOMPOSED driver REPLICATES this loop but drives
    /// each fine substep's stages with a fine-level halo exchange between them (the fine patch may
    /// span a tile cut), so this method is the single-tile reference. PURE EXTRACTION.
    fn level_subcycle(&mut self, level: usize, dt: f64) {
        let fine_dt = dt / RATIO as f64;
        for sub in 0..RATIO {
            let alpha = sub as f64 / RATIO as f64;
            self.prolong_cf(level + 1, alpha);
            let f = &self.levels[level + 1];
            prof("ghost_fill", || f.kernels.ghost_fill(&f.state));
            self.advance_level(level + 1, fine_dt, alpha);
        }
    }

    /// restrict the finer level into this level's coverage + apply the flux/emf reflux + re-derive
    /// prim (and, if this level has a coarser parent, re-prolong its coarse-fine ghosts). runs AFTER
    /// the finer-level subcycle. the flux/emf registers are TILE-LOCAL (a coarse cell + the fine
    /// cells at its face are co-located), so the decomposed driver calls this per tile unchanged.
    /// PURE EXTRACTION.
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
                let inv_dx: [f64; NDIM] =
                    std::array::from_fn(|ax| 1.0 / coarse.state.geom.dx[ax]);
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
                self.flux_registers[level]
                    .apply_uniform(&l.state.fields.cons, &l.state.geom.dx);
            } else {
                let geo = l.state.geom.block_geometry(l.state.physics.metric);
                self.flux_registers[level].apply(&l.state.fields.cons, &geo);
            }
        });
        let l = &self.levels[level];
        prof("c2p", || l.kernels.c2p(&l.state));
        if has_coarser {
            self.prolong_cf(level, alpha0 + 1.0 / RATIO as f64);
        }
        let l = &self.levels[level];
        prof("ghost_fill", || l.kernels.ghost_fill(&l.state));
    }

    /// advance this level's clock (fine levels only; the root clock is the driver's). PURE EXTRACTION.
    fn level_clock(&mut self, level: usize, dt: f64) {
        if level > 0 {
            let s = &mut self.levels[level].state;
            s.time += dt;
            s.iteration += 1;
        }
    }

    /// fill level `level`'s coarse-fine prim ghosts from its parent's
    /// time-interpolated prims: `(1 - alpha)*prim_old + alpha*prim_new`. pub so the DECOMPOSED
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
        for (slab, scratch) in slabs.iter().zip(sweep_scratch) {
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
            // mhd: the cell-centered B ghosts feed the fine reconstruction +
            // the boundary-edge UCT emf. the fine OWNED bface needs no
            // prolongation (the fine CT evolves its own boundary faces, and
            // CT preserves divB for ANY emf values).
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
        // the bface TRANSVERSE HALO at coarse-fine sides: the scalar ghost
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
// helpers
// =============================================================================

/// the per-stage EFFECTIVE flux weights of an ssp scheme: stage i's operator
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

/// is this level on a uniform cartesian grid — the flux register's KERNEL
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
/// measured per-root-step stall at production sizes (~0.7 GB per step at
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

/// a gravity-only proxy of a body collection for the coarser levels: accreting
/// bodies keep their mass / softening / motion but lose the sink (the kernels
/// then see sink_rate = 0 and remove no mass); collection-level capabilities
/// (binary params, frame) are preserved.
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
/// halfwidth — stays inside the parent ALLOCATED domain.
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
/// the patch is TILE-LOCAL the sub-grid is 1x1, so the fine exchange is a no-op; when
/// the patch SPANS cuts the sub-grid has internal cuts and the fine halos are exchanged.
pub struct FineSubgrid<const NDIM: usize> {
    pub counts: [usize; NDIM],
    pub order: Vec<usize>,
    pub devices: Vec<i32>,
}

/// derive the first-fine-level sub-grid from which tiles carry a fine level. None if no tile is
/// refined. asserts the refined tiles fill a rectangle (the SMR single-box invariant). pub so the
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
    let refined: Vec<usize> = (0..tiles.len()).filter(|&i| tiles[i].levels.len() > 1).collect();
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
    Some(FineSubgrid { counts: sub_counts, order, devices: sub_devices })
}

/// exchange one LEVEL's halos across an ordered set of tiles (a sub-grid of shape `counts`), then
/// per-tile `ghost_fill` at that level. `order[k]` is the tile index at flatten position `k`;
/// `devices[k]` is order-aligned. cut faces are `CoarseFine` (so `ghost_fill` leaves them for the
/// exchange); the sub-grid's OUTER faces are the patch's coarse-fine boundary (prolong-filled) or a
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
        let states: Vec<&FieldStore<NDIM, DOF, Mem, f64>> =
            order.iter().map(|&i| &*tiles[i].levels[level].state).collect();
        exchange_grid(&states, counts, devices, transport);
    }
    for (k, &i) in order.iter().enumerate() {
        symbi_xpu::with_device(devices[k], || {
            tiles[i].levels[level].kernels.ghost_fill(&tiles[i].levels[level].state)
        });
    }
}

/// the DECOMPOSED hierarchy driver (refinement x decomposition): lockstep-advance N
/// per-tile hierarchies, decomposing the ROOT and the FIRST FINE level. the root stages run with a
/// root halo exchange BETWEEN them (rk2 corrector reads each neighbor's stage-1 update); the fine
/// subcycle is driven here so its stages can exchange the FINE halos between fine tiles when a patch
/// SPANS a tile cut. for a TILE-LOCAL patch the fine sub-grid is 1x1, so the fine exchange
/// is a no-op and this reduces to the tile-local case exactly. the flux/emf reflux registers stay TILE-LOCAL
/// (a coarse cell + the fine cells at its face are co-located; any register write to a cut-adjacent
/// GHOST is overwritten by the next root exchange). global dt = min over tiles of `root_cfl_dt()`.
/// proven `decomposed == monolithic` by `decomp_refine_equivalence.rs` + its spanning-cut variant.
///
/// LEVELS 2+ are advanced TILE-LOCALLY (inside the level-1 fine tile, via the recursive
/// `level_step_tail(1)`); decomposing a patch that spans a cut on a level >= 2 is not supported.
/// hydro only in the root post-step (mesh motion / immersed bodies + refinement-decomp deferred;
/// the root clock is advanced directly). `on_checkpoint(iteration, time, &tiles)` fires every
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
    // the finest-level tail owns the IBM surface physics; the refined-
    // decomposed step never runs the root tail, so a refined run whose tiles
    // ALL failed to build a fine patch would penalize its bodies NOWHERE —
    // silent physics loss, refused here.
    assert!(
        tiles.iter().any(|h| h.levels.len() > 1)
            || tiles.iter().all(|h| !h.levels[0].state.has_bodies()),
        "refined-decomposed: no tile built a fine level while immersed bodies are active; \
         the finest-level penalize would silently never run",
    );
    let nstages = ts.stages().len();
    debug_assert_eq!(n, devices.len(), "tiles/devices length mismatch");
    let root_order: Vec<usize> = (0..n).collect();
    let fine = fine_subgrid(tiles, counts, devices);

    // cut halos current before the first flux.
    exchange_level_halos(tiles, 0, &root_order, counts, devices, transport);

    let mut t = start_time;
    let mut iter: u64 = 0;
    let mut last_cb: u64 = 0;
    while t < t_final {
        // global dt = min over tiles' root cfl, clamped to land exactly on t_final.
        let mut gdt = t_final - t;
        for i in 0..n {
            gdt = gdt.min(symbi_xpu::with_device(devices[i], || tiles[i].root_cfl_dt()));
        }
        for i in 0..n {
            symbi_xpu::with_device(devices[i], || tiles[i].level_step_begin(0, gdt));
        }
        // the root SSP stages, with a root halo exchange between each.
        for ii in 0..nstages {
            for i in 0..n {
                symbi_xpu::with_device(devices[i], || tiles[i].level_stage(0, ii, gdt, 0.0));
            }
            exchange_level_halos(tiles, 0, &root_order, counts, devices, transport);
        }
        // root emf bookkeeping (mhd; no-op for hydro + unrefined tiles).
        for i in 0..n {
            symbi_xpu::with_device(devices[i], || tiles[i].level_tail_emf(0, gdt));
        }
        // the FINE subcycle, decomposed across the refined tiles: RATIO substeps, each fine substep
        // driven stage-by-stage with a fine halo exchange (a no-op for a tile-local 1x1 sub-grid).
        if let Some(fg) = &fine {
            let fine_dt = gdt / RATIO as f64;
            for sub in 0..RATIO {
                let alpha = sub as f64 / RATIO as f64;
                for (k, &i) in fg.order.iter().enumerate() {
                    symbi_xpu::with_device(fg.devices[k], || {
                        tiles[i].prolong_cf(1, alpha);
                        tiles[i].levels[1].kernels.ghost_fill(&tiles[i].levels[1].state);
                    });
                }
                // fill the fine cut halo before the fine flux reads it.
                exchange_level_halos(tiles, 1, &fg.order, fg.counts, &fg.devices, transport);
                for (k, &i) in fg.order.iter().enumerate() {
                    symbi_xpu::with_device(fg.devices[k], || tiles[i].level_step_begin(1, fine_dt));
                }
                for jj in 0..nstages {
                    for (k, &i) in fg.order.iter().enumerate() {
                        symbi_xpu::with_device(fg.devices[k], || {
                            tiles[i].level_stage(1, jj, fine_dt, alpha)
                        });
                    }
                    exchange_level_halos(tiles, 1, &fg.order, fg.counts, &fg.devices, transport);
                }
                for (k, &i) in fg.order.iter().enumerate() {
                    symbi_xpu::with_device(fg.devices[k], || {
                        tiles[i].level_step_tail(1, fine_dt, alpha)
                    });
                }
            }
            // tile-local restrict + flux/emf reflux + c2p + ghost on each refined tile's root.
            for (k, &i) in fg.order.iter().enumerate() {
                symbi_xpu::with_device(fg.devices[k], || tiles[i].level_restrict_reflux(0, 0.0));
            }
        }
        // the root clock (mesh motion in the root post-step remains deferred).
        for i in 0..n {
            symbi_xpu::with_device(devices[i], || {
                tiles[i].levels[0].state.time += gdt;
                tiles[i].levels[0].state.iteration += 1;
            });
        }
        // per-step immersed-body bookkeeping, the hierarchy form of the flat decomposed
        // body step: each tile's FINEST level reduces its LOCAL feedback partials (the
        // tile finest interiors partition the sink region — coarser levels carry a
        // gravity-only proxy and never drain), the deltas are summed across tiles into
        // the true global reaction, and the identical global delta + prescribed advance
        // is applied to every tile's bodies. identical input -> identical body state,
        // so all tiles stay in lockstep.
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
                    !(any_refined && tiles[i].levels.len() == 1 && tiles[i].accretion_overlaps_root()),
                    "refinement x decomposition: a sink sphere overlaps UNREFINED tile {i} — the \
                     refined patch must cover the sink on every tile it touches"
                );
            }
        }
        // the restrict/c2p recomputed root prim over the allocated domain (incl. the cut halo from
        // stale halo cons, and the cut-adjacent reflux ghosts); refresh for the next step's stage 0.
        exchange_level_halos(tiles, 0, &root_order, counts, devices, transport);

        t += gdt;
        iter += 1;
        if iter - last_cb >= interval {
            last_cb = iter;
            drain_devices::<Mem>(devices);
            if on_checkpoint(iter, t, tiles).is_break() {
                return;
            }
        }
    }
    drain_devices::<Mem>(devices);
    let _ = on_checkpoint(iter, t, tiles);
}
