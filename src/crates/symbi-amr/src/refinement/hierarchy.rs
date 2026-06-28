// =============================================================================
// hierarchy.rs
//
// the static-mesh-refinement (SMR) hierarchy (docs/design/22 phases 0+2). each
// level is a complete SimStateGeneric + its KernelSet; the hierarchy adds only
// inter-level coordination: recursive berger-oliger subcycling with
// time-interpolated coarse-fine ghost prolongation, conservative restriction,
// and flux-register refluxing. the single-level engine is untouched —
//
// SINGLE-COVERAGE CAP (this is SMR, not adaptive AMR): each level refines exactly
// ONE box (`coverage: Option<Domain>`), with ONE FluxRegister per coarse-fine
// level-pair. the refined region is fixed at setup, not re-flagged from the
// solution; there is no patch graph and no berger-rigoutsos clustering.
// multi-patch adaptive refinement (a level as a disjoint cover of Domains) is
// future work — see docs/design/21_amr.md and findings/ADVERSARIAL_REVIEW_2026-06-16.md §5.
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
    prolong_face_field, prolong_field, prolong_prims, restrict_bface, restrict_cell_field,
    restrict_cons,
};
use symbi_sim::driver::{check_dt_or_panic, evolve_bodies, prof, stage_tag, stage_time_fractions};
use symbi_sim::hydro_ops::scan_c2p_errors;
use symbi_sim::state::{
    Boundaries, BoundaryType, PrimFieldsGeneric, SimStateGeneric, array_field_zeros, axis_name,
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
    /// cell-centered B at this level's step start (mhd only) — the magnetic
    /// counterpart of prim_old for the finer level's ghost prolongation.
    pub bcell_old: Option<[Field<f64, NDIM, Mem>; NDIM]>,
    /// staggered face B at this level's step start (mhd only) — feeds the
    /// finer level's bface transverse-halo prolongation (per-component
    /// staggered domains, cloned from this level's bface).
    pub bface_old: Option<[Field<f64, NDIM, Mem>; NDIM]>,
    /// the region of THIS level covered by the next finer level, in absolute
    /// indices of this level. None on the finest. single-coverage cap: exactly
    /// ONE refined box per level (this is static refinement / SMR, not multi-patch
    /// adaptive AMR — no disjoint-cover `Vec<Domain>`, no clustering).
    pub coverage: Option<Domain<NDIM>>,
}

// =============================================================================
// hierarchy
// =============================================================================

/// static-refinement hierarchy: levels[0] = coarsest, levels[n-1] = finest.
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
                bcell_old: None,
                bface_old: None,
                coverage: None,
            }],
            prolong_order: ProlongOrder::Plm,
            flux_registers: Vec::new(),
            emf_registers: Vec::new(),
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

    /// every accreting body's sink sphere must lie inside the finest level's
    /// interior — a sink straddling a coarse-fine boundary corrupts the mass
    /// accounting the refluxing protects.
    fn assert_sinks_inside_finest(&self) {
        let finest = &self.levels[self.levels.len() - 1].state;
        let Some(im) = finest.immersed.as_ref() else {
            return;
        };
        let geom = &finest.geom;
        im.bodies.visit_accretion(|body| {
            let racc: f64 = body.accretion_radius().unwrap_or(0.0);
            for ax in 0..NDIM {
                let lo = geom.x_lo[ax] + geom.interior.spaces[ax].lo as f64 * geom.dx[ax];
                let hi = geom.x_lo[ax] + geom.interior.spaces[ax].hi as f64 * geom.dx[ax];
                let p: f64 = body.position[ax];
                assert!(
                    p - racc >= lo && p + racc <= hi,
                    "refinement: body {} sink sphere [{:.4}, {:.4}] leaves the finest level \
                     [{lo:.4}, {hi:.4}] on axis {ax}",
                    body.idx,
                    p - racc,
                    p + racc
                );
            }
        });
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
        // the caller reads fields next: drain the device queue (the B12
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
        let mut dt_cfl = f64::INFINITY;
        for (ll, lvl) in self.levels.iter().enumerate() {
            let scale = RATIO.pow(ll as u32) as f64;
            dt_cfl = dt_cfl.min(lvl.kernels.cfl(&lvl.state) * scale);
        }
        dt_cfl
    }

    /// one root step: cfl-limited dt (clamped to t_final), the recursive level
    /// advance, then the root clock + body state. EVERY level limits the root
    /// step — covered coarse cells are conservative averages of fine data, so
    /// a fast feature resolved only on the fine level is diluted out of the
    /// root's own cfl; level l subcycles RATIO^l times, so its limit enters
    /// scaled by RATIO^l (tests/refine_per_level_cfl.rs pins this).
    fn step_root(&mut self, t_final: f64) {
        let mut dt_cfl = f64::INFINITY;
        for (ll, lvl) in self.levels.iter().enumerate() {
            let scale = RATIO.pow(ll as u32) as f64;
            dt_cfl = dt_cfl.min(lvl.kernels.cfl(&lvl.state) * scale);
        }
        let root = &mut self.levels[0];
        let dt = dt_cfl.min(t_final - root.state.time);
        check_dt_or_panic(dt, root.state.iteration, root.state.time);
        root.state.dt = dt;

        self.advance_level(0, dt, 0.0);

        let root = &mut self.levels[0];
        // homologous linear advance ONLY when there is no traced motion law.
        if root.state.motion_law.is_none() && root.state.motion.homologous {
            root.state.motion.a += root.state.motion.a_dot * dt;
        }
        root.state.time += dt;
        // expression motion: set a / a_dot EXACTLY at the new time (for output and the next root
        // step's cfl + stages), instead of a constant-rate extrapolation that overshoots a
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
                finest.kernels.body_feedback(&finest.state, dt);
            }
            finest.state.dt = dt;
            evolve_bodies(&mut finest.state);

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
        let has_finer = level + 1 < self.levels.len();
        let has_coarser = level > 0;

        // snapshot this level's prims at its step start for the finer level's
        // time-interpolated ghost prolongation.
        if has_finer {
            prof("refine_save_prim", || save_prim_old(&self.levels[level]));
        }

        self.levels[level].state.dt = dt;
        let stages = self.levels[level].state.timestepping.stages();
        let n = stages.len();
        let weights = flux_weights(stages);
        let stage_time = stage_time_fractions(stages);
        if n > 1 {
            let l = &self.levels[level];
            prof("snapshot", || l.kernels.snapshot(&l.state));
        }
        let additive_source = self.levels[level].kernels.has_additive_source();
        for (ii, &(a0, ac)) in stages.iter().enumerate() {
            let l = &self.levels[level];
            if additive_source {
                prof("snapshot_stage", || l.kernels.snapshot_stage(&l.state));
            }
            l.kernels.wave_speeds(&l.state);
            prof("flux", || {
                for dd in 0..NDIM {
                    l.kernels.flux(&l.state, dd);
                }
            });
            // register accumulation between flux and the stage update. the
            // per-stage weight is the stage's EFFECTIVE flux contribution
            // after the ssp convex recombination, so the step total is dt.
            prof("refine_flux_reg", || {
                if has_finer {
                    let reg = &self.flux_registers[level];
                    let l = &self.levels[level];
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
                    let reg = &self.flux_registers[level - 1];
                    let l = &self.levels[level];
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
            let l = &self.levels[level];
            prof("efield", || l.kernels.efield(&l.state));
            prof("godunov_stage", || {
                l.kernels.godunov_stage(&l.state, dt, a0, ac)
            });
            prof("post_godunov", || {
                l.kernels.post_godunov(&l.state, dt, stage_tag(ii, n))
            });
            if additive_source {
                prof("source_apply", || l.kernels.source_apply(&l.state, ac * dt));
            }
            if l.state.has_bodies() {
                prof("body_source", || l.kernels.body_source(&l.state, ac * dt));
            }
            prof("c2p", || l.kernels.c2p(&l.state));
            // c2p over the full allocated domain recomputed the coarse-fine
            // prim ghosts from stale cons; re-prolong them at the time of the
            // state entering the NEXT stage (the last stage's tail lands on
            // the substep end = the next substep's start) before the physical
            // fill reads corners.
            if has_coarser {
                prof("refine_prolong", || {
                    self.prolong_cf(level, alpha0 + stage_time[ii] / RATIO as f64)
                });
            }
            let l = &self.levels[level];
            prof("ghost_fill", || l.kernels.ghost_fill(&l.state));
        }

        // emf bookkeeping (mhd): after the stage loop the efield buffers hold
        // the EFFECTIVE per-step EMF (post_godunov wrote the rk2 time-average
        // in place before the single curl application; euler keeps the raw
        // stage EMF) — sample it for the registers on both sides.
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

        // subcycle the finer level, then restrict + reflux.
        if has_finer {
            let fine_dt = dt / RATIO as f64;
            for sub in 0..RATIO {
                let alpha = sub as f64 / RATIO as f64;
                prof("refine_prolong", || self.prolong_cf(level + 1, alpha));
                let f = &self.levels[level + 1];
                prof("ghost_fill", || f.kernels.ghost_fill(&f.state));
                self.advance_level(level + 1, fine_dt, alpha);
            }

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
                prof("refine_prolong", || {
                    self.prolong_cf(level, alpha0 + 1.0 / RATIO as f64)
                });
            }
            let l = &self.levels[level];
            prof("ghost_fill", || l.kernels.ghost_fill(&l.state));
        }

        if level > 0 {
            let s = &mut self.levels[level].state;
            s.time += dt;
            s.iteration += 1;
        }
    }

    /// fill level `level`'s coarse-fine prim ghosts from its parent's
    /// time-interpolated prims: `(1 - alpha)*prim_old + alpha*prim_new`.
    fn prolong_cf(&self, level: usize, alpha: f64) {
        let parent = &self.levels[level - 1];
        let fine = &self.levels[level];
        let prim_old = parent
            .prim_old
            .as_ref()
            .expect("the parent of a fine level carries prim_old");
        let slabs = cf_ghost_slabs(
            &fine.state.geom.allocated,
            &fine.state.geom.interior,
            &fine.state.boundaries,
        );
        for slab in &slabs {
            prolong_prims(
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
                    prolong_field(
                        &bcell_old[aa],
                        &pmhd.bcell[aa],
                        &fmhd.bcell[aa],
                        slab,
                        self.prolong_order,
                        alpha,
                    );
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
                    prolong_face_field(
                        dd,
                        &bface_old[dd],
                        &pmhd.bface[dd],
                        &fmhd.bface[dd],
                        &slab,
                        alpha,
                    );
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
