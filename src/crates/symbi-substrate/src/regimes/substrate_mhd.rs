// =============================================================================
// regimes/substrate_mhd.rs
//
// MhdSubstrateKernelSet<R, Mem, Sc, D> — the ONE ideal-MHD KernelSet, generic over the
// regime `R` (RMHD / NewtonianMhd / IsothermalMhd; DOF = 3 fixed).
//
// the three MHD families differ ONLY by data read off `R::SPEC`:
//   - `Self::kernel_prefix()`                     -> the AOT kernel-name prefix (rmhd / nmhd / imhd)
//   - `R::SPEC.has_energy`               -> whether the pre / nrg field rows are bound
//                                           (gamma EOS scalar vs the iso `cs`)
//   - `R::SPEC.materializes_wave_speeds` -> RMHD writes the quartic ws_l/ws_r in a
//                                           `wave_speeds` pass for the HLLE flux to read;
//                                           NMHD/iMHD compute the magnetosonic speed inline
//
// on a CURVED chart that pass is conditional rather than regime-fixed, because two independent
// kernels may consume its output: the HLL face flux (which reads the per-cell speeds instead of
// solving its own fan — unlike the HLLD and rusanov arms) and the UCT edge EMF. the condition is
// read off the flux kernel's OWN manifest, so it cannot drift from what that kernel actually does.
// the gas godunov + the ENTIRE constrained-transport stack are regime-agnostic and delegate
// to `mhd_substrate` (the SAME AOT kernels). the per-regime structs (`RmhdSubstrateKernelSet`
// etc.) are now back-compat type aliases of this one.
//
// usage:
//  let sub = MhdSubstrateKernelSet::<Rmhd, HostMemory, f64, 3>::new(gamma, cfl, theta, &alloc);
//  // or, identically, via the back-compat alias:
//  let sub = RmhdSubstrateKernelSet::<HostMemory, f64, 3>::new(gamma, cfl, theta, &alloc);
// =============================================================================

use std::marker::PhantomData;
use std::sync::atomic::{AtomicU64, Ordering};

use symbi_algebra::{Domain, OrderedNumeric};
use symbi_grid::Field;
use symbi_hydro::regime::Regime;
use symbi_ir::algebra::Scalar;
use symbi_ir::{FieldRef, ScalarRef};
use symbi_xpu::MemorySpace;

use crate::kernels::support::cfl_from_lambda;
use std::sync::Arc;

use crate::regimes::substrate_kernels::{
    RegimeKind, RuntimeSource, ScalarBind, Solver, dispatch_c2p_status, dispatch_driven_boundaries,
    dispatch_named, dispatch_runtime_source, geom_scalar, kernel_bindings, kernel_geom,
    mhd_flux_suffix, mhd_geom_suffix, motion_scalar, scalars_for, spacetime_slug,
};
use symbi_hydro::source_spec::BuiltSource;
use symbi_sim::state::CtMethod;
use symbi_sim::state::FieldStore;
use symbi_sim::substrate_seam::KernelSet;

static MHD_CFL_DIAGNOSTIC_CALLS: AtomicU64 = AtomicU64::new(0);

/// the per-coordinate metric scale factor `h` (in f64): the
/// physical width along an axis is `h * coordinate_width`. cartesian = 1; spherical theta = r,
/// phi = r*sin(theta); cylindrical phi = r.
#[inline]
fn host_scale_factor(coords: symbi_geometry::Geometry, ci: usize, pos: &[f64; 3]) -> f64 {
    use symbi_geometry::Geometry::*;
    match (coords, ci) {
        (Cartesian, _) => 1.0,
        (Spherical, 1) => pos[0],
        (Spherical, 2) => pos[0] * pos[1].sin(),
        (Spherical, _) => 1.0,
        (Cylindrical, 1) => pos[0],
        (Cylindrical, _) => 1.0,
    }
}

/// the MAX inverse PHYSICAL cell width over the interior, `max_{cell, axis} 1/(h_a * dx_a)` (i.e. the
/// reciprocal of the SMALLEST physical cell). the diagonal-metric scale factors are extremized at the
/// coordinate boundaries (min r, pole-closest theta), so the smallest physical cell sits at a domain
/// CORNER — evaluate the `2^D` corner cells, O(1). the resistive diffusion crosses PHYSICAL widths, so
/// a curvilinear grid (small r / near the poles) is stiffer than its coordinate spacing suggests;
/// Cartesian falls through to `1/dx` (h = 1).
pub(crate) fn max_inv_physical_width<const D: usize>(
    geom: &symbi_sim::state::PartitionGeometry<D>,
) -> f64 {
    let mut max_inv = 0.0f64;
    for corner in 0..(1usize << D) {
        let coord: [isize; D] = std::array::from_fn(|d| {
            let s = &geom.interior.spaces[d];
            if (corner >> d) & 1 == 0 {
                s.lo
            } else {
                s.hi - 1
            }
        });
        // physical cell-center per COORDINATE axis (for the scale factors), mirroring driven_inflow_lambda.
        let mut pos = [0.0f64; 3];
        for d in 0..D {
            let c = match &geom.maps {
                Some(m) => m[d].center(coord[d]),
                None => geom.x_lo[d] + (coord[d] as f64 + 0.5) * geom.dx[d],
            };
            pos[geom.axes[d]] = c;
        }
        for d in 0..D {
            let h = host_scale_factor(geom.coords, geom.axes[d], &pos);
            let iw = 1.0 / (h * geom.cell_width(coord, d));
            if iw > max_inv {
                max_inv = iw;
            }
        }
    }
    max_inv
}

/// the relativistic driven-inflow CFL bound: `max` over the interior cells adjacent to every
/// DRIVEN boundary face of `1 / (h_d * width_d)` (the inverse physical cell width, the steepest
/// per-axis). a relativistic signal travels at most `c = 1`, so this is the `lambda` a wind
/// injected at that face imposes — invisible to the interior-only wave-speed map. returns 0 when
/// there are no driven faces (the `max` is then a no-op).
fn driven_inflow_lambda<const D: usize, const DOF: usize, Mem, Sc>(
    sim: &FieldStore<D, DOF, Mem, Sc>,
) -> f64
where
    Sc: Scalar + OrderedNumeric,
    Mem: MemorySpace,
{
    let geom = &sim.geom;
    let mut max_inv_w = 0.0f64;
    for a in 0..D {
        for side in 0..2 {
            let bt = if side == 0 {
                sim.boundaries.lo(a)
            } else {
                sim.boundaries.hi(a)
            };
            if !matches!(bt, symbi_sim::state::BoundaryType::Driven(_)) {
                continue;
            }
            let s = if side == 0 {
                symbi_algebra::Side::Lo
            } else {
                symbi_algebra::Side::Hi
            };
            for coord in geom.interior.boundary(a, s, 1).iter() {
                // physical cell-center per COORDINATE axis (for the scale factors).
                let mut pos = [0.0f64; 3];
                for d in 0..D {
                    let c = match &geom.maps {
                        Some(m) => m[d].center(coord[d]),
                        None => geom.x_lo[d] + (coord[d] as f64 + 0.5) * geom.dx[d],
                    };
                    pos[geom.axes[d]] = c;
                }
                let mut inv_w = 0.0f64;
                for d in 0..D {
                    let h = host_scale_factor(geom.coords, geom.axes[d], &pos);
                    let iw = 1.0 / (h * geom.cell_width(coord, d));
                    if iw > inv_w {
                        inv_w = iw;
                    }
                }
                if inv_w > max_inv_w {
                    max_inv_w = inv_w;
                }
            }
        }
    }
    max_inv_w
}

/// the unified D-dimensional ideal-MHD `KernelSet`, regime supplied as `R` (carries `SPEC`).
pub struct MhdSubstrateKernelSet<R, Mem: MemorySpace, Sc: Scalar + OrderedNumeric, const D: usize> {
    /// the EOS scalar: `gamma` (ideal-gas regimes) or `cs` (isothermal). bound to the flux/cfl
    /// kernel's eos param by name (`gamma` when `has_energy`, else `cs`).
    pub eos_param: f64,
    pub cfl_number: f64,
    /// theta-MC limiter compression in [1,2]: 1 = minmod, 2 = monotonized-central.
    pub theta: f64,
    pub cfl_scratch: Field<Sc, D, Mem>,
    /// the joint constraint blend and the index of the member that bound it, written by the
    /// state-constraint projection. `binding` is what lets the injection budget name the constraint
    /// it charges rather than reporting one aggregate number.
    pub constraint_theta: Field<Sc, D, Mem>,
    pub constraint_binding: Field<Sc, D, Mem>,
    /// the run's DECLARED constraint family parameters. neutral by default — nothing is floored
    /// unless a caller asks, so there is no hidden default floor.
    pub constraints: crate::regimes::substrate_kernels::ConstraintParams,
    /// Riemann solver — HLLE (default) / HLLC / HLLD; validated against the regime at attach.
    pub solver: Solver,
    /// constrained-transport edge-EMF scheme: Contact (Gardiner-Stone, default) or Uct (Del Zanna
    /// 2007 / Mignone-Del Zanna 2021 HLL-weighted, kills the odd-even checkerboard). Uct needs the
    /// per-cell Riemann wave speeds materialized (RMHD only).
    pub ct_method: CtMethod,
    /// a runtime user source (python -> json `SourceConfig` -> `build_user_source`),
    /// applied two-pass via the regime-agnostic `dispatch_runtime_source`. nmhd/imhd
    /// take the newtonian force/cooling/relax lifts; rmhd is relativistic so only
    /// `kind="raw"` reaches here. targets the hydro conserved slots (den/mom/nrg);
    /// B is evolved by CT. None for source-free runs.
    pub runtime_source: Option<Arc<RuntimeSource>>,
    /// driven (DYNAMIC) boundary prescriptions, indexed by `BoundaryType::Driven(id)`. each is a
    /// complete prim DAG `[rho, vel.., pre, B..]`; the standard ghost-fill skips the driven face,
    /// then `dispatch_driven_boundaries` assigns its ghost state. empty => no driven faces.
    pub boundary_dags: Vec<Arc<RuntimeSource>>,
    /// consecutive substages on which the FOFC last-resort freeze tier fired; a genuine unrecoverable
    /// poison freezes every stage, while the rare correct parachute is isolated (see fofc.rs).
    pub freeze_streak: std::sync::atomic::AtomicU32,
    /// the Ohmic resistivity `eta` (the induction diffusivity `dB/dt += eta * lap(B)`, added as a
    /// resistive edge EMF before the CT curl). 0 = ideal MHD (bit-identical). the resistive CFL
    /// `dt <= 1/2 dx^2 / eta` folds into the wave-speed reduction.
    pub resistivity: f64,
    /// the horizon-excision radius on a cartesian kerr-schild background (0 = no
    /// excision). the gas fill + magnetized conserved rebuild run once per step;
    /// the staggered faces stay ct-owned, so div(sqrt(gamma) b) is untouched.
    pub excision_radius: f64,
    /// density and pressure of the absorbing atmosphere inside the excision surface.
    pub excision_rho: f64,
    pub excision_pre: f64,
    /// shakura-sunyaev alpha with the LOCAL sound speed (energy regimes only);
    /// takes precedence over the constant nu when positive.
    pub alpha: f64,
    /// constant kinematic viscosity `nu` (the Navier-Stokes shear on the velocity, ORTHOGONAL to the
    /// resistive diffusion of B). 0 = inviscid. >0 runs the viscous force + (energy regime) the viscous
    /// heating onto the total energy; B is untouched, so the heat warms the gas with 1/2 B^2 preserved.
    /// finite magnetic Prandtl number Pm = nu / eta. cartesian; full 3D (D==DOF), 2.5D via the
    /// DOF-aware kernel. caps dt at C_visc dx^2 / nu.
    pub viscosity: f64,
    _r: PhantomData<R>,
}

impl<R, Mem, Sc, const D: usize> MhdSubstrateKernelSet<R, Mem, Sc, D>
where
    R: Regime<Sc, D>,
    Mem: MemorySpace,
    Sc: Scalar + OrderedNumeric,
{
    pub fn new(eos_param: f64, cfl_number: f64, theta: f64, alloc_domain: &Domain<D>) -> Self {
        let cfl_scratch = Field::<Sc, D, Mem>::zeros(alloc_domain)
            .unwrap_or_else(|_| panic!("failed to allocate {} CFL scratch", Self::kernel_prefix()));
        let alloc = |what: &str| {
            Field::<Sc, D, Mem>::zeros(alloc_domain)
                .unwrap_or_else(|_| panic!("failed to allocate {} {what}", Self::kernel_prefix()))
        };
        Self {
            eos_param,
            cfl_number,
            theta,
            cfl_scratch,
            constraint_theta: alloc("constraint theta"),
            constraint_binding: alloc("constraint binding"),
            constraints: Default::default(),
            solver: Solver::Hlle,
            ct_method: CtMethod::Contact,
            runtime_source: None,
            boundary_dags: Vec::new(),
            freeze_streak: std::sync::atomic::AtomicU32::new(0),
            resistivity: 0.0,
            viscosity: 0.0,
            excision_radius: 0.0,
            excision_rho: 1.0,
            excision_pre: 1.0,
            alpha: 0.0,
            _r: PhantomData,
        }
    }

    /// register a DRIVEN (DYNAMIC) boundary. the returned id (registration order) is what the
    /// sim's `Boundaries` carries as `BoundaryType::Driven(id)` on the prescribed face. build
    /// `built` from `expr_bridge::build_boundary_dag(&cfg, R::SPEC)` — a complete prim prescription
    /// `[rho, vel.., pre, B..]`. for a purely toroidal injection the in-plane B is 0 and the
    /// out-of-plane B_phi is the injected toroidal field (cell-centered, div-free by axisymmetry).
    pub fn with_driven_boundary(
        mut self,
        built: Vec<(String, BuiltSource)>,
        params: Vec<f64>,
    ) -> (Self, u16) {
        let id = self.boundary_dags.len() as u16;
        self.boundary_dags
            .push(RuntimeSource::new(built, params, R::SPEC.has_energy));
        (self, id)
    }

    /// attach a runtime user source from already-lowered `(target, BuiltSource)` pairs
    /// (the `build_user_source` output of a `SourceConfig`). applied two-pass in
    /// `source_apply`. has_energy is read from the regime spec (rmhd/nmhd carry it,
    /// imhd does not), so the source pass writes only the slots the regime owns.
    /// declare the Ohmic resistivity `eta` (fluent; default 0 = ideal MHD). the resistive edge EMF
    /// `eta * J` rides the CT curl and the resistive CFL `dt <= 1/2 dx^2 / eta` bounds the step.
    pub fn with_resistivity(mut self, eta: f64) -> Self {
        self.resistivity = eta;
        self
    }

    pub fn with_runtime_source(
        mut self,
        built: Vec<(String, BuiltSource)>,
        params: Vec<f64>,
    ) -> Self {
        self.runtime_source = Some(RuntimeSource::new(built, params, R::SPEC.has_energy));
        self
    }

    /// pick the Riemann solver; rejects one invalid for this regime. fluent.
    pub fn with_solver(mut self, solver: Solver) -> Result<Self, symbi_sim::state::ConfigError> {
        let regime = RegimeKind::of::<Sc, D, R>();
        if !solver.valid_for(regime) {
            return Err(symbi_sim::state::ConfigError::SolverRegimeMismatch { solver, regime });
        }
        self.solver = solver;
        Ok(self)
    }

    /// set the theta-MC limiter compression in [1,2] (1 = minmod, 2 = monotonized-central). fluent.
    pub fn theta(mut self, theta: f64) -> Self {
        self.theta = theta;
        self
    }

    /// select the constrained-transport edge-EMF scheme (Contact / Uct). fluent.
    pub fn ct_method(mut self, m: CtMethod) -> Self {
        self.ct_method = m;
        self
    }

    // the AOT kernel-name PREFIX (build.rs's family token). distinct from `SPEC.name` (the
    // canonical/diagnostic name): the MHD kernels were emitted under abbreviations (`nmhd`/`imhd`)
    // that the canonical names (`newtonian_mhd`/`iso_mhd`) do not match. one bridge, here.
    #[inline]
    fn kernel_prefix() -> &'static str {
        match R::SPEC.name {
            "newtonian_mhd" => "nmhd",
            "iso_mhd" => "imhd",
            n => n, // "rmhd" passes through unchanged
        }
    }
}

impl<R, Mem, Sc, const D: usize> MhdSubstrateKernelSet<R, Mem, Sc, D>
where
    R: Regime<Sc, D>,
    Mem: MemorySpace + Sync,
    Sc: Scalar + OrderedNumeric,
{
    /// the face-flux kernel name for one sweep direction. flat charts key on the geometry + the
    /// solver's own suffix; curved charts key on the metric-aware valencia family, where the solver
    /// tag is "" (HLLE), "_hlld" (tetrad MUB09) or "_rusanov" (the light-cone Lax-Friedrichs fan the
    /// first-order flux correction falls back to). the SINGLE spelling of the name — anything that
    /// needs to know what the sweep will run asks here rather than re-deriving it.
    fn flux_kernel_name(
        sim: &FieldStore<D, 3, Mem, Sc>,
        dir: usize,
        flat_suffix: &str,
        gr_solver: &str,
    ) -> String {
        // ONE composition, both backgrounds. these were two `format!`s with the chart and the
        // solver in OPPOSITE orders -- flat spelled `{chart}{solver}`, curved `{solver}{chart}`
        // -- inside this one function. they never collided because the flat branch's chart is
        // non-empty on exactly one grid (cylindrical r-z), so the disagreement was invisible
        // until a solver was selected there. the SEGMENTS still differ by background (flat keys
        // the chart on the r-z plane alone; curved keys it on the full grid-axis set, since a
        // curved kernel is baked per chart), and that difference is real; the ORDER is not, and
        // now lives in `face_flux_name` with every other flux.
        let flat = spacetime_slug(sim.geom.spacetime).is_empty();
        let (solver_sfx, chart_sfx) = if flat {
            (
                flat_suffix,
                mhd_flux_suffix(sim.geom.coords, &sim.geom.axes),
            )
        } else {
            (
                gr_solver,
                mhd_geom_suffix(sim.geom.coords, &sim.geom.axes),
            )
        };
        symbi_discretize::kernel_slug::face_flux_name(
            Self::kernel_prefix(),
            solver_sfx,
            "",
            "",
            "",
            chart_sfx,
            sim.geom.spacetime,
            D,
            dir,
        )
    }

    /// the face-flux kernel the PRODUCTION sweep runs for `dir` under the configured solver.
    fn face_flux_kernel(&self, sim: &FieldStore<D, 3, Mem, Sc>, dir: usize) -> String {
        let gr_solver = if matches!(self.solver, Solver::Hlld) {
            "_hlld"
        } else {
            ""
        };
        Self::flux_kernel_name(sim, dir, self.solver.kernel_suffix(), gr_solver)
    }

    /// the flux sweep, parameterized by the flat solver suffix, the GR HLLD toggle, and the slope
    /// limiter `theta` — so FOFC can re-run it at FIRST ORDER (HLLE + theta = 0) through the same
    /// code path the production sweep uses. the production `flux` calls this with the configured
    /// solver + `self.theta`.
    fn flux_impl(
        &self,
        sim: &FieldStore<D, 3, Mem, Sc>,
        dir: usize,
        flat_suffix: &str,
        gr_solver: &str,
        theta: f64,
    ) {
        // face domain extended +1 on the sweep hi + 1 on each transverse axis (CT corners).
        let mut face = sim.geom.interior.extend(dir, 0, 1);
        for ax in 0..D {
            if ax != dir {
                face = face.expand(ax, 1);
            }
        }
        let flux_name = Self::flux_kernel_name(sim, dir, flat_suffix, gr_solver);
        let (x_lo_k, dx_k) = kernel_geom(
            &sim.geom.x_lo,
            &sim.geom.dx,
            &sim.geom.maps,
            sim.geom.coords,
            sim.motion.a,
        );
        let scalars = scalars_for(&flux_name, |bind| match bind {
            ScalarBind::Ref(ScalarRef::Gamma) | ScalarBind::Ref(ScalarRef::Cs) => {
                Sc::from_f64(self.eos_param)
            }
            ScalarBind::Ref(ScalarRef::Theta) => Sc::from_f64(theta),
            ScalarBind::Ref(ScalarRef::SchwarzschildMass) => Sc::from_f64(
                sim.geom
                    .spacetime_scalars
                    .iter()
                    .find(|(n, _)| n == "schwarzschild_mass")
                    .map(|(_, v)| *v)
                    .expect("GR MHD flux needs schwarzschild_mass"),
            ),
            ScalarBind::Ref(ScalarRef::KerrSpin) => Sc::from_f64(
                sim.geom
                    .spacetime_scalars
                    .iter()
                    .find(|(n, _)| n == "kerr_spin")
                    .map(|(_, v)| *v)
                    .expect("GR MHD flux needs kerr_spin"),
            ),
            ScalarBind::Ref(other) => Sc::from_f64(
                geom_scalar(&x_lo_k, &dx_k, &sim.geom.maps, *other).unwrap_or_else(|| {
                    panic!(
                        "{} flux: unexpected scalar {other:?}",
                        Self::kernel_prefix()
                    )
                }),
            ),
            o => panic!("{} flux: unexpected scalar {o:?}", Self::kernel_prefix()),
        });
        let pre_bind = if R::SPEC.has_energy {
            sim.fields
                .prim
                .pre_field()
                .expect("MHD energy regime requires prim.pre")
        } else {
            &sim.fields.cons.den
        };
        dispatch_named(sim, pre_bind, None, dir, &flux_name, &face, &[], &scalars);
    }

    /// FIRST-ORDER FLUX CORRECTION for the MHD regimes: the shared face-based gas redo (first-order
    /// flux = the light-cone Lax-Friedrichs / Rusanov fan) PLUS the constrained-transport re-sync.
    /// the induction/CT subsystem is INVARIANT under a gas FOFC — B evolves by curl-of-EMF,
    /// independent of the gas c2p — so the face field, EMF, and curl stay HIGH-ORDER; the redo only
    /// (a) restores the HO induction flux for the cell-B predictor so its magnetic-energy patch is
    /// the small HO reconciliation (not the FO-vs-HO shock), and (b) re-runs `bcell_from_bface` on the
    /// patch stages to re-attach `bcell = interp(bface_HO)` + the patch onto the corrected gas.
    ///
    /// on a CURVED background the redo continues into the conservative source replay: with the
    /// spliced flux and EMF fixed, the pointwise geometric source is scaled per cell to the
    /// largest fraction of the source ray that stays inside the GRMHD admissible set (Wu & Tang,
    /// arXiv:1709.05838, theorem 2.1). nothing shared between neighbors moves, so the update
    /// telescopes and `div(B)` is untouched; the only clipped quantity is a local, already
    /// non-conservative metric source. an anchor that is inadmissible with NO source at all is a
    /// timestep failure and rejects the step (`true`).
    fn fofc_impl(
        &self,
        sim: &FieldStore<D, 3, Mem, Sc>,
        dt: f64,
        a0: f64,
        ac: f64,
        stage: u8,
    ) -> bool {
        let pre_bind = if R::SPEC.has_energy {
            sim.fields
                .prim
                .pre_field()
                .expect("MHD energy regime requires prim.pre")
        } else {
            &sim.fields.cons.den
        };
        // `bcell_from_bface` (the face->cell interp + magnetic-energy patch) runs on the SINGLE
        // (Euler, tag 0) and CORRECTOR (rk2, tag 2) stages, never the predictor (tag 1, which leaves
        // bcell flux-predicted). re-sync exactly there.
        let patch_stage = stage == 0 || stage == 2;
        let has_energy = R::SPEC.has_energy;
        let rebuild_spliced_ct = || {
            use crate::regimes::mhd_substrate as ct;
            let prefix = Self::kernel_prefix();
            let flag = &sim.workspace.fofc_flag;
            ct::fofc_splice_induction(sim, prefix, flag);
            if patch_stage {
                ct::efield(
                    sim,
                    CtMethod::Contact,
                    self.solver,
                    prefix,
                    self.eos_param,
                    0.0,
                );
                ct::fofc_emf_splice(sim, flag);
                ct::fofc_restore_bface_n(sim);
                ct::ct_curl(sim, dt);
            }
        };
        let resync_ct = || {
            if patch_stage {
                crate::regimes::mhd_substrate::bcell_from_bface(sim, has_energy);
            }
        };
        crate::regimes::fofc::fofc_orchestrate(
            sim,
            Self::kernel_prefix(),
            "", // MHD momentum is always a 3-vector; no DOF-lift tag
            self.has_additive_source(),
            &self.cfl_scratch,
            pre_bind,
            &self.freeze_streak,
            |dir| self.flux_impl(sim, dir, "", "_rusanov", 0.0),
            || self.c2p(sim),
            || self.godunov_stage(sim, dt, a0, ac),
            || self.source_apply(sim, ac * dt),
            || {}, // MHD has no immersed-body source (trait-default no-op)
            || {
                // TIER 2, below the conservative source limiter: the limiter SCALES a source and
                // therefore cannot act on a cell that is already outside G with no source left to
                // scale. this projects such a cell onto the boundary of `G \cap C` — the admissible
                // set intersected with the run's DECLARED constraint family — along the segment to
                // the admissible stage-input anchor. admissibility is simply the always-present
                // member, so floors and the sufficient condition are enforced in ONE operation
                // rather than as two projections that could disagree.
                // exact passthrough on an admissible cell (theta = 1, bit-for-bit), so it is a
                // no-op everywhere except the states nothing above it can resolve. the cell-B
                // enters the residual but is never blended — constrained transport owns the
                // staggered field, so div(B) survives by construction. curved GRMHD only.
                if Self::kernel_prefix() == "rmhd"
                    && has_energy
                    && sim.geom.spacetime != symbi_geometry::Spacetime::Minkowski
                {
                    let mhd = sim
                        .fields
                        .mhd
                        .as_ref()
                        .expect("the GRMHD projection requires magnetic fields");
                    // the state-constraint family projection (`constraint_projection`) anchors on
                    // the stage-input CONSERVED state, which is wrong here: constrained transport
                    // has already advanced B, so `u_stage` paired with the CANDIDATE's cell B is a
                    // hybrid with no admissibility guarantee, and the projection cannot recover the
                    // cell at any blend. a sound anchor is rebuilt by p2c from the stage-input
                    // PRIMITIVES with the candidate's B (and raised to the margin), exactly as this
                    // kernel does — substituting the family without that reconstruction collapses
                    // the magnetized torus from t = 4.02 to a dt underflow at t = 2.286.
                    crate::regimes::substrate_kernels::fofc_project(
                        sim,
                        "rmhd",
                        // MHD keys its chart on the grid-axis set: B is always a 3-vector,
                        // so the momentum-DOF lift cannot separate the two cylindrical planes.
                        symbi_discretize::kernel_slug::ChartKeying::GridAxes,
                        self.eos_param,
                        sim.stage_input(),
                        &sim.fields.cons,
                        &sim.fields.prim,
                        Some([&mhd.bcell[0], &mhd.bcell[1], &mhd.bcell[2]]),
                    );
                }
            },
            None, // no body-evolved freeze parachute (no MHD body source)
            || crate::regimes::mhd_substrate::fofc_ct_save(sim),
            || crate::regimes::mhd_substrate::fofc_restore_bcell_stage(sim),
            rebuild_spliced_ct,
            || {
                use crate::regimes::fofc::SourceReplay;
                // the source limiter is defined against the GRMHD admissible set (Wu & Tang
                // theorem 2.1), which needs the energy slot and a curved metric to have a
                // geometric source at all.
                let curved_rmhd = Self::kernel_prefix() == "rmhd"
                    && has_energy
                    && sim.geom.spacetime != symbi_geometry::Spacetime::Minkowski;
                if !curved_rmhd {
                    return SourceReplay::NotApplicable;
                }
                use crate::regimes::mhd_substrate as ct;
                // the cfl scratch carries the per-cell source weight (first the anchor's zero,
                // then the inadmissibility mask, then theta). the CFL reduction runs between
                // steps and the freeze probe re-uses the same buffer only AFTER this replay has
                // consumed theta, so the three lifetimes never overlap.
                let source_weight = &self.cfl_scratch;
                let cons = &sim.fields.cons;
                let prim = &sim.fields.prim;
                let mhd = sim
                    .fields
                    .mhd
                    .as_ref()
                    .expect("the GRMHD source replay requires magnetic fields");
                let resync = resync_ct;
                let restore_stage = || {
                    crate::regimes::fofc::fofc_copy(
                        sim,
                        "rmhd",
                        "",
                        "restore",
                        (sim.stage_input(), prim),
                        (cons, prim),
                    );
                    ct::fofc_restore_bcell_stage(sim);
                    self.c2p(sim);
                };
                let apply_additive = || {
                    if self.has_additive_source() {
                        self.source_apply(sim, ac * dt);
                    }
                };

                // the ANCHOR: the same conservative flux + CT update with the geometric source
                // switched OFF (weight 0). that operator is the low-order physical-constraint-
                // preserving one, so under the CFL its output is admissible — unless the timestep
                // itself is too large, which no source fraction can repair.
                ct::fill_cell_field(sim, source_weight, 0.0);
                ct::godunov_stage_pcp(sim, self.eos_param, dt, a0, ac, source_weight);
                resync();
                apply_additive();
                self.c2p(sim);
                crate::regimes::fofc::fofc_probe(sim, "rmhd", "", pre_bind, source_weight);
                // cells inside the excision surface are causally disconnected and their state is
                // overwritten by the horizon fill, so they must not veto the step. the c2p error
                // field is the scratch for the masked count: the next c2p rewrites it before any
                // reader, on both the reject and the continue path.
                crate::regimes::substrate_kernels::fofc_exterior_mask(
                    sim,
                    self.excision_radius,
                    source_weight,
                    &sim.fields.c2p_error,
                );
                if crate::regimes::fofc::fofc_flag_count(sim, &sim.fields.c2p_error) != 0 {
                    // the source-free low-order anchor is itself inadmissible, so there is no
                    // admissible endpoint to measure a source fraction against — this tier cannot
                    // act. hand the substage to the ORDINARY redo, whose projection maps the cell
                    // onto partial-G. shrinking the timestep is NOT the answer here: the anchor is
                    // inadmissible because the state cannot be represented, not because the step
                    // was too long, so rejecting merely replays the same failure at half dt.
                    restore_stage();
                    return SourceReplay::NotApplicable;
                }
                // every face splice has already consumed flux_ho; its first component group is
                // dead for the remainder of this fallback and safely retains the cell anchor.
                let anchor = &sim.workspace.flux_ho[0];
                crate::regimes::fofc::fofc_copy(
                    sim,
                    "rmhd",
                    "",
                    "restore",
                    (cons, prim),
                    (anchor, prim),
                );

                // evaluate the matching full-source candidate from the identical stage input.
                restore_stage();
                self.godunov_stage(sim, dt, a0, ac);
                resync();
                apply_additive();
                crate::regimes::substrate_kernels::fofc_source_theta(
                    sim,
                    anchor,
                    cons,
                    [&mhd.bcell[0], &mhd.bcell[1], &mhd.bcell[2]],
                    source_weight,
                );

                // replay once more with only the local metric source scaled by theta. the flux
                // divergence and the CT curl are re-run bit-identically, so every face still
                // carries ONE flux and div(B) is untouched — only the pointwise, non-conservative
                // geometric source is clipped, and only on the cells that needed it.
                restore_stage();
                ct::godunov_stage_pcp(sim, self.eos_param, dt, a0, ac, source_weight);
                resync();
                apply_additive();
                SourceReplay::Completed
            },
            resync_ct,
            // curved GRMHD alone has the projection tier; where it ran and still left an exterior
            // cell outside G, replaying the step beats waiving that cell's conservation.
            Self::kernel_prefix() == "rmhd"
                && has_energy
                && sim.geom.spacetime != symbi_geometry::Spacetime::Minkowski,
        )
    }
}

impl<R, Mem, Sc, const D: usize> symbi_sim::substrate_seam::WithViscosity
    for MhdSubstrateKernelSet<R, Mem, Sc, D>
where
    R: Regime<Sc, D>,
    Mem: MemorySpace,
    Sc: Scalar + OrderedNumeric,
{
    fn with_viscosity(mut self, nu: f64) -> Self {
        self.viscosity = nu;
        self
    }
    fn with_alpha(mut self, alpha: f64) -> Self {
        // nu = alpha (gamma p/rho)/Omega_K with the LOCAL sound speed — energy
        // regimes only (iso MHD has no pressure field to read cs^2 from).
        assert!(
            R::SPEC.has_energy,
            "alpha viscosity on MHD needs the energy regime (local cs^2 = gamma p/rho); \
             iso MHD has no pressure field — use constant-nu with_viscosity"
        );
        self.alpha = alpha;
        self
    }
}

impl<R, Mem, Sc, const D: usize> symbi_sim::substrate_seam::WithResistivity
    for MhdSubstrateKernelSet<R, Mem, Sc, D>
where
    R: Regime<Sc, D>,
    Mem: MemorySpace,
    Sc: Scalar + OrderedNumeric,
{
    fn with_resistivity(mut self, eta: f64) -> Self {
        self.resistivity = eta;
        self
    }
}

impl<R, Mem, Sc, const D: usize> symbi_sim::substrate_seam::WithExcision
    for MhdSubstrateKernelSet<R, Mem, Sc, D>
where
    R: Regime<Sc, D>,
    Mem: MemorySpace,
    Sc: Scalar + OrderedNumeric,
{
    fn with_excision(mut self, r_exc: f64, rho_scale: f64, pre_scale: f64) -> Self {
        self.excision_radius = r_exc;
        self.excision_rho = rho_scale * 1e-10;
        self.excision_pre = pre_scale * 1e-12;
        self
    }
}

impl<R, Mem, Sc, const D: usize> KernelSet<D, 3, Mem, Sc> for MhdSubstrateKernelSet<R, Mem, Sc, D>
where
    R: Regime<Sc, D>,
    Mem: MemorySpace + Sync,
    Sc: Scalar + OrderedNumeric,
{
    fn viscous(&self, sim: &FieldStore<D, 3, Mem, Sc>, dt: f64) {
        // alpha (local-cs shakura-sunyaev) takes precedence over the constant nu;
        // the 2.5D DOF-aware kernel diffuses the toroidal momentum too.
        if self.alpha > 0.0 {
            // cartesian 2.5D + the curvilinear 2.5D planes + full 3D charts all
            // route through dispatch_viscous_alpha's (D, DOF, chart) match, which
            // fails loud on an unbaked combination.
            crate::regimes::substrate_kernels::dispatch_viscous_alpha(
                sim,
                dt,
                self.alpha,
                self.eos_param,
            );
            return;
        }
        if self.viscosity <= 0.0 {
            return;
        }
        // viscosity acts on the velocity (orthogonal to the resistive diffusion of B); the energy
        // regime also books the viscous heating onto the total energy (B untouched -> the gas heats,
        // 1/2 B^2 preserved). full 3D MHD (D=DOF=3) uses the 3D kernel (cartesian flat / chart
        // orthogonal); 2.5D MHD (D=2, DOF=3) the DOF-aware plane kernel (cartesian, cyl r-phi,
        // cyl r-z, spherical meridian) so the toroidal velocity diffuses with its metric.
        crate::regimes::substrate_kernels::dispatch_viscous(sim, dt, self.viscosity);
    }

    fn flux(&self, sim: &FieldStore<D, 3, Mem, Sc>, dir: usize) {
        // the production sweep: the configured solver + slope limiter.
        let gr_solver = if matches!(self.solver, Solver::Hlld) {
            "_hlld"
        } else {
            ""
        };
        self.flux_impl(sim, dir, self.solver.kernel_suffix(), gr_solver, self.theta);
    }

    fn fofc(&self, sim: &FieldStore<D, 3, Mem, Sc>, dt: f64, a0: f64, ac: f64, stage: u8) -> bool {
        self.fofc_impl(sim, dt, a0, ac, stage)
    }

    fn fofc_active(&self) -> bool {
        true
    }

    fn snapshot_retry(&self, sim: &FieldStore<D, 3, Mem, Sc>) {
        crate::regimes::mhd_substrate::snapshot_retry(sim);
    }

    fn restore_step(&self, sim: &FieldStore<D, 3, Mem, Sc>) {
        crate::regimes::mhd_substrate::restore_step(sim);
        self.c2p(sim);
        self.ghost_fill(sim);
    }

    fn penalize(&self, sim: &FieldStore<D, 3, Mem, Sc>, dt: f64) {
        // the immersed-body penalization under MHD, via the 1/2|B|^2 sandwich: strip
        // the magnetic energy so the (unchanged) hydro drain acts on the GAS energy alone, run
        // it, then restore the field energy. the drain never touches bcell, so B and 1/2|B|^2
        // are exactly invariant and the flux is left to constrained transport. host-only (the
        // immersed dispatch is host + f64); with no body the shifts would be a wasted no-op.
        if !Mem::IS_HOST_ACCESSIBLE || sim.immersed.is_none() {
            return;
        }
        crate::regimes::mhd_substrate::shift_magnetic_energy(sim, -1.0);
        // eos_param is gamma (has_energy MHD); c_drain uses the adiabatic default 1.0.
        crate::regimes::substrate_kernels::dispatch_penalize(sim, dt, self.eos_param, 1.0);
        crate::regimes::mhd_substrate::shift_magnetic_energy(sim, 1.0);
    }

    fn c2p(&self, sim: &FieldStore<D, 3, Mem, Sc>) {
        // the primitives now hold a state recovered from the conserved fields; anything
        // reading prim.* outside the evolve loop checks this before trusting it.
        sim.mark_primitives_recovered();
        let st = spacetime_slug(sim.geom.spacetime);
        let cname = if st.is_empty() {
            format!("{}_c2p_{D}d", Self::kernel_prefix())
        } else {
            // the metric-aware KKC recovery: gamma at the volume-weighted centroid, so the name
            // carries the chart + spacing + spacetime slugs and the kernel reads mass + grid scalars.
            let gsfx = mhd_geom_suffix(sim.geom.coords, &sim.geom.axes);
            format!("{}_c2p{gsfx}{st}_{D}d", Self::kernel_prefix())
        };
        let (x_lo_k, dx_k) = kernel_geom(
            &sim.geom.x_lo,
            &sim.geom.dx,
            &sim.geom.maps,
            sim.geom.coords,
            sim.motion.a,
        );
        // iso c2p declares no scalars -> scalars_for returns [] (resolver never called).
        let scalars = scalars_for(&cname, |bind| match bind {
            ScalarBind::Ref(ScalarRef::Gamma) | ScalarBind::Ref(ScalarRef::Cs) => {
                Sc::from_f64(self.eos_param)
            }
            ScalarBind::Ref(ScalarRef::SchwarzschildMass) => Sc::from_f64(
                sim.geom
                    .spacetime_scalars
                    .iter()
                    .find(|(n, _)| n == "schwarzschild_mass")
                    .map(|(_, v)| *v)
                    .expect("GR MHD c2p needs schwarzschild_mass"),
            ),
            ScalarBind::Ref(ScalarRef::KerrSpin) => Sc::from_f64(
                sim.geom
                    .spacetime_scalars
                    .iter()
                    .find(|(n, _)| n == "kerr_spin")
                    .map(|(_, v)| *v)
                    .expect("GR MHD c2p needs kerr_spin"),
            ),
            ScalarBind::Ref(other) => Sc::from_f64(
                geom_scalar(&x_lo_k, &dx_k, &sim.geom.maps, *other).unwrap_or_else(|| {
                    panic!("{} c2p: unexpected scalar {other:?}", Self::kernel_prefix())
                }),
            ),
            o => panic!("{} c2p: unexpected scalar {o:?}", Self::kernel_prefix()),
        });
        // bind BY MANIFEST: cons.{den,mom,nrg?} + bcell(3) reads -> prim.{rho,vel,pre?} writes.
        // the energy regimes' `prim.pre` is an OUTPUT here, so `pre` binds the real field.
        let pre_bind = if R::SPEC.has_energy {
            sim.fields
                .prim
                .pre_field()
                .expect("MHD energy regime requires prim.pre")
        } else {
            &sim.fields.cons.den
        };
        dispatch_named(
            sim,
            pre_bind,
            None,
            0,
            &cname,
            &sim.geom.interior,
            &[],
            &scalars,
        );
        dispatch_c2p_status(sim, pre_bind, Self::kernel_prefix(), "");
    }

    fn wave_speeds(&self, sim: &FieldStore<D, 3, Mem, Sc>) {
        // materializing regimes (RMHD always; NMHD under UCT, whose edge-EMF coefficients read
        // wave_speed_l/r) run the per-cell pass; others compute speeds inline in the flux.
        let materialize = R::SPEC.materializes_wave_speeds
            || (self.ct_method == CtMethod::Uct
                && matches!(Self::kernel_prefix(), "nmhd" | "imhd"));
        let st = spacetime_slug(sim.geom.spacetime);
        // GR: the per-cell speeds have TWO consumers — the GR-UCT edge EMF (its corner
        // coefficients) and the GR HLL flux fan, which reads `wave_speed_{l,r}[dir]` from the two
        // cells sharing each face (davis estimate). the HLLD arm solves its own five-wave fan and
        // the rusanov fallback uses the state-independent light-cone bound, so neither reads them.
        // ASK THE KERNEL rather than re-deriving which arm reads what: a flux that consumes the
        // fields while this pass is skipped sees their zero initialization, which collapses the fan
        // onto the shift alone and leaves ZERO dissipation on every axis whose shift component
        // vanishes — a one-sided, odd-even-decoupled sweep that grows a grid-scale checkerboard.
        if !st.is_empty() {
            let reads_speeds = |name: &str| {
                kernel_bindings(name).iter().any(|(field, is_output)| {
                    !is_output && matches!(field, FieldRef::WaveSpeedL(_) | FieldRef::WaveSpeedR(_))
                })
            };
            let flux_consumes = (0..D).any(|dir| reads_speeds(&self.face_flux_kernel(sim, dir)));
            if !flux_consumes && self.ct_method != CtMethod::Uct {
                return;
            }
            let gsfx = mhd_geom_suffix(sim.geom.coords, &sim.geom.axes);
            let wsname = format!("{}_wave_speeds_cell{gsfx}{st}_{D}d", Self::kernel_prefix());
            let (x_lo_k, dx_k) = kernel_geom(
                &sim.geom.x_lo,
                &sim.geom.dx,
                &sim.geom.maps,
                sim.geom.coords,
                sim.motion.a,
            );
            let scalars = scalars_for(&wsname, |bind| match bind {
                ScalarBind::Ref(ScalarRef::Gamma) | ScalarBind::Ref(ScalarRef::Cs) => {
                    Sc::from_f64(self.eos_param)
                }
                ScalarBind::Ref(ScalarRef::SchwarzschildMass) => Sc::from_f64(
                    sim.geom
                        .spacetime_scalars
                        .iter()
                        .find(|(n, _)| n == "schwarzschild_mass")
                        .map(|(_, v)| *v)
                        .expect("GR UCT wave speeds need schwarzschild_mass"),
                ),
                ScalarBind::Ref(ScalarRef::KerrSpin) => Sc::from_f64(
                    sim.geom
                        .spacetime_scalars
                        .iter()
                        .find(|(n, _)| n == "kerr_spin")
                        .map(|(_, v)| *v)
                        .expect("GR UCT wave speeds need kerr_spin"),
                ),
                ScalarBind::Ref(other) => Sc::from_f64(
                    geom_scalar(&x_lo_k, &dx_k, &sim.geom.maps, *other).unwrap_or_else(|| {
                        panic!(
                            "{} wave_speeds: unexpected scalar {other:?}",
                            Self::kernel_prefix()
                        )
                    }),
                ),
                o => panic!(
                    "{} wave_speeds: unexpected scalar {o:?}",
                    Self::kernel_prefix()
                ),
            });
            let pre_bind = sim
                .fields
                .prim
                .pre_field()
                .expect("GR MHD requires prim.pre");
            dispatch_named(
                sim,
                pre_bind,
                None,
                0,
                &wsname,
                &sim.geom.allocated,
                &[],
                &scalars,
            );
            return;
        }
        if !materialize {
            return;
        }
        let wsname = format!("{}_wave_speeds_cell_{D}d", Self::kernel_prefix());
        let scalars = scalars_for(&wsname, |bind| match bind {
            ScalarBind::Ref(ScalarRef::Gamma) | ScalarBind::Ref(ScalarRef::Cs) => {
                Sc::from_f64(self.eos_param)
            }
            o => panic!(
                "{} wave_speeds: unexpected scalar {o:?}",
                Self::kernel_prefix()
            ),
        });
        // bind BY MANIFEST: prim + bcell reads -> the per-axis `wave_speed_{l,r}[k]` writes (typed
        // `WaveSpeedL/R(k)`). energy regimes have prim.pre; isothermal (no pressure) passes den as
        // the leading window field (mirrors the cfl dispatch).
        let pre_bind = if R::SPEC.has_energy {
            sim.fields
                .prim
                .pre_field()
                .expect("MHD energy regime requires prim.pre")
        } else {
            &sim.fields.cons.den
        };
        dispatch_named(
            sim,
            pre_bind,
            None,
            0,
            &wsname,
            &sim.geom.allocated,
            &[],
            &scalars,
        );
    }

    // horizon excision as the sweep/finalize pieces (see the RHD set): the gas
    // primitives freeze at the vacuum floor; the magnetized p2c folds the cell B into
    // the conserved rebuild; the staggered faces are never written. inert at
    // zero radius; the dispatch asserts the baked combination fail-loud.
    fn excise_pass_count(&self, sim: &FieldStore<D, 3, Mem, Sc>) -> usize {
        crate::regimes::substrate_kernels::excise_pass_count_for(sim, self.excision_radius)
    }

    fn excise_sweep(&self, sim: &FieldStore<D, 3, Mem, Sc>) {
        if self.excision_radius > 0.0 {
            crate::regimes::substrate_kernels::dispatch_excise_sweep(
                sim,
                self.eos_param,
                self.excision_radius,
                self.excision_rho,
                self.excision_pre,
            );
        }
    }

    fn excise_finalize(&self, sim: &FieldStore<D, 3, Mem, Sc>) {
        if self.excision_radius > 0.0 {
            crate::regimes::substrate_kernels::dispatch_excise_finalize(
                sim,
                self.eos_param,
                self.excision_radius,
                self.excision_rho,
                self.excision_pre,
            );
        }
    }

    fn cfl(&self, sim: &FieldStore<D, 3, Mem, Sc>) -> f64 {
        let geom = &sim.geom;
        let st = spacetime_slug(geom.spacetime);
        // the geometry / spacing / spacetime slugs all ride the name: a log-radial grid selects the
        // geometric-mean CFL-width map (`_logr`); uniform grids get sp = "" so the name is unchanged.
        let wname = format!(
            "{}_wave_speed_map{}{st}_{D}d",
            Self::kernel_prefix(),
            mhd_geom_suffix(geom.coords, &geom.axes)
        );
        // scalars BY NAME (the kernel's declared set drives it): eos param + the per-axis CFL
        // widths (cartesian `inv_dx_d`, curvilinear `x_lo_d`/`dx_d`); the mhd substrates run static,
        // so the motion rates bind 0. `kernel_geom` gives the log-aware per-axis scalars the in-kernel
        // `gv_axis_face_at` reads via `map_kind`; on a uniform static grid it is bit-identical to the
        // physical geometry.
        let (x_lo_phys, dx_phys) =
            kernel_geom(&geom.x_lo, &geom.dx, &geom.maps, geom.coords, sim.motion.a);
        let resolve_scalar = |bind: &ScalarBind| -> Sc {
            let sref = match bind {
                ScalarBind::Ref(sref) => sref,
                // the source-cfl kernel zeroes its admissibility rate on the excised
                // r_ks < r_exc level set; 0 on an unexcised run (empty mask).
                ScalarBind::Spec(sp) if &**sp == "excision_radius" => {
                    return Sc::from_f64(self.excision_radius);
                }
                other => panic!(
                    "{} cfl: unexpected spec scalar {other:?}",
                    Self::kernel_prefix()
                ),
            };
            match *sref {
                ScalarRef::Gamma | ScalarRef::Cs => Sc::from_f64(self.eos_param),
                ScalarRef::SchwarzschildMass => Sc::from_f64(
                    geom.spacetime_scalars
                        .iter()
                        .find(|(n, _)| n == "schwarzschild_mass")
                        .map(|(_, v)| *v)
                        .expect("GR MHD cfl needs schwarzschild_mass"),
                ),
                ScalarRef::KerrSpin => Sc::from_f64(
                    geom.spacetime_scalars
                        .iter()
                        .find(|(n, _)| n == "kerr_spin")
                        .map(|(_, v)| *v)
                        .expect("GR MHD cfl needs kerr_spin"),
                ),
                other => Sc::from_f64(
                    motion_scalar(&sim.motion, geom.coords, D, other)
                        .or_else(|| geom_scalar(&x_lo_phys, &dx_phys, &sim.geom.maps, other))
                        .unwrap_or_else(|| {
                            panic!("{} cfl: unexpected scalar {other:?}", Self::kernel_prefix())
                        }),
                ),
            }
        };
        let scalars = scalars_for(&wname, &resolve_scalar);
        // bind BY MANIFEST: prim + bcell reads -> the `scratch` lambda write (the cfl_scratch
        // field, supplied as the scratch override). iso passes a dummy pre (reads cs^2*rho).
        let pre_bind = if R::SPEC.has_energy {
            sim.fields
                .prim
                .pre_field()
                .expect("MHD energy regime requires prim.pre")
        } else {
            &sim.fields.cons.den
        };
        dispatch_named(
            sim,
            pre_bind,
            Some(&self.cfl_scratch),
            0,
            &wname,
            &geom.interior,
            &[],
            &scalars,
        );
        let diagnose_cfl = std::env::var_os("SIMBI_CFL_DIAGNOSTICS").is_some();
        let flux_max = diagnose_cfl.then(|| {
            crate::regimes::substrate_gpu::field_max_reduce(&self.cfl_scratch, &geom.interior)
        });
        // the wu 2017 (arXiv:1708.07267) source-admissibility rate: on a curved background the
        // geometric source S advances U -> U + dt S and can push the conserved state out of the
        // physical-constraint set; the light-cone LxF FOFC flux is only physical-constraint-preserving
        // under a timestep that also keeps U + dt S admissible. this kernel reads the flux light-cone
        // rate already in the scratch and adds the source characteristic rate
        // lambda_S = (|S_tau| + ||S_mom||_gamma)/q(U) in place, so the reduction sizes dt against both
        // (dt (lambda_flux + lambda_S) < 1). baked for GR-RMHD only.
        //
        // KEPT for GRMHD even though the projection enforces the SUFFICIENT admissibility condition
        // (wu & tang, arXiv:1709.05838, theorem 2.1), unlike the GR-hydro path where the same
        // projection retires it. enforcing sufficiency is necessary for the retirement but not enough
        // to license it, because an a-posteriori projection is not the physical-constraint-preserving
        // LIMITER those theorems assume: the limiter bounds the RECONSTRUCTION at every quadrature
        // point of every cell and the CFL still carries lambda_S alongside it, whereas this projection
        // maps the already-updated state and only runs when some cell has tripped the correction flag.
        // dropping the rate leaves two residual violations, both measured on the magnetized torus at
        // native resolution: the recovered gas pressure reaches -8.3e-10, which is what projecting ONTO
        // the admissible boundary gives once downstream rounding crosses it, and the field mirror
        // symmetry degrades to 5.3e-13 against a 1e-14 stencil bound, because the boundary crossing is
        // ill-conditioned where psi approaches it tangentially and amplifies a roundoff-scale asymmetry
        // in the two mirror states into a discrete difference in the blend factor.
        //
        // the sufficient condition still does most of the work: under the B-free necessary cone the
        // same retirement produced NaN within a few steps, so psi lifts the failure from a blowup to a
        // rounding-scale boundary violation. what remains is a limiter-versus-projection gap, not a
        // missing magnetic term.
        if !st.is_empty() {
            let sname = format!(
                "{}_source_cfl{}{st}_{D}d",
                Self::kernel_prefix(),
                mhd_geom_suffix(geom.coords, &geom.axes)
            );
            let sscalars = scalars_for(&sname, &resolve_scalar);
            dispatch_named(
                sim,
                pre_bind,
                Some(&self.cfl_scratch),
                0,
                &sname,
                &geom.interior,
                &[],
                &sscalars,
            );
        }
        let mut lambda_max =
            crate::regimes::substrate_gpu::field_max_reduce(&self.cfl_scratch, &geom.interior);
        if diagnose_cfl {
            let call = MHD_CFL_DIAGNOSTIC_CALLS.fetch_add(1, Ordering::Relaxed);
            // `SIMBI_CFL_DIAGNOSTICS=all` reports EVERY call. the sampled cadence shows the
            // steady state but never the excursion that actually halts a run: a rate spike lasts
            // one step, so sampling every hundredth call is guaranteed to miss it.
            let every = std::env::var("SIMBI_CFL_DIAGNOSTICS").is_ok_and(|v| v == "all");
            if every || call == 0 || call.is_multiple_of(100) {
                let flux_max = flux_max.unwrap_or(0.0);
                let minimum_source = (lambda_max - flux_max).max(0.0);
                let owner = if Mem::IS_DEVICE_ACCESSIBLE {
                    String::new()
                } else {
                    let rates = self.cfl_scratch.view();
                    let rho = sim.fields.prim.rho.view();
                    let pre = sim.fields.prim.pre_field().map(Field::view);
                    let mut best = None;
                    for c in geom.interior.iter() {
                        let rate = (*rates.at(c)).to_f64();
                        if best.is_none_or(|(_, best_rate): (_, f64)| rate > best_rate) {
                            best = Some((c, rate));
                        }
                    }
                    best.map(|(c, rate)| {
                        let x: [f64; D] = std::array::from_fn(|aa| {
                            geom.x_lo[aa] + (c[aa] as f64 + 0.5) * geom.dx[aa]
                        });
                        let radius = x.iter().map(|xx| xx * xx).sum::<f64>().sqrt();
                        let pressure = pre.as_ref().map(|view| (*view.at(c)).to_f64());
                        format!(
                            " owner={c:?} x={x:?} r={radius:.9e} rate={rate:.9e} \
                             rho={:.9e} pre={pressure:?}",
                            (*rho.at(c)).to_f64(),
                        )
                    })
                    .unwrap_or_default()
                };
                eprintln!(
                    "mhd cfl diagnostic: call={call} flux_max={flux_max:.9e} \
                     total_max={lambda_max:.9e} minimum_source={minimum_source:.9e} \
                     dt={:.9e}{owner}",
                    cfl_from_lambda(lambda_max, self.cfl_number),
                );
            }
        }
        // OHMIC RESISTIVE CFL: explicit induction diffusion is stable for `dt <= dx^2 / (2 D eta)`;
        // fold the equivalent rate `2 D eta / min(dx)^2` (an inverse timescale, like the wave rate)
        // into lambda_max so the shared `dt = cfl / lambda_max` bounds the step. 0 off resistive MHD.
        // the resistive rate is set by the STIFFEST diffusivity present: the uniform bulk resistivity
        // OR any immersed body's localized `MagneticSpec::Resistive` eta (chi <= 1, so the body eta is
        // its own upper bound). fold the max so a resistive sink cannot outrun the diffusion limit.
        let body_eta_max = sim.immersed.as_ref().map_or(0.0, |im| {
            (0..im.bodies.len())
                .filter_map(|b| match im.bodies.get(b).spec.magnetic {
                    symbi_ib::MagneticSpec::Resistive { eta } => Some(eta),
                    _ => None,
                })
                .fold(0.0_f64, f64::max)
        });
        let eta_eff = self.resistivity.max(body_eta_max);
        if eta_eff > 0.0 {
            // the diffusion crosses PHYSICAL cell widths h*dx (the coordinate dx alone omits the metric factor h); on a curvilinear grid
            // the smallest physical cell (min r / near the poles) sets the tightest bound. inv_w =
            // 1/min_physical_width, so 2 D eta / width_min^2 = 2 D eta * inv_w^2.
            let inv_w = max_inv_physical_width(geom);
            lambda_max = lambda_max.max(2.0 * (D as f64) * eta_eff * inv_w * inv_w);
        }
        // VISCOUS CFL: the momentum diffusion obeys the same parabolic limit as the
        // resistive one; fold the equivalent rate for the constant nu OR the alpha
        // bound (largest local sound speed at the slowest orbit).
        let nu_eff = if self.alpha > 0.0 {
            crate::regimes::substrate_kernels::adiabatic_alpha_nu_max(
                sim,
                self.alpha,
                self.eos_param,
            )
        } else {
            self.viscosity
        };
        if nu_eff > 0.0 {
            let inv_w = max_inv_physical_width(geom);
            lambda_max = lambda_max.max(2.0 * (D as f64) * nu_eff * inv_w * inv_w);
        }
        // DRIVEN-INFLOW CFL CAP: the per-cell wave-speed map only scans the INTERIOR, so a driven
        // boundary's inflow state (which lives in the ghost band) is invisible to it — a relativistic
        // wind from a cold-ambient inner boundary would size dt off the slow interior and then get
        // pulled across the first inner face at a dt ~1e4x too large -> NaN. a relativistic signal
        // can travel at most c = 1, so bound dt by the light-crossing of the interior cell adjacent
        // to each driven face: lambda >= max_d (1 / (h_d * width_d)). only relativistic regimes get
        // the c = 1 bound.
        if R::SPEC.is_relativistic {
            lambda_max = lambda_max.max(driven_inflow_lambda(sim));
        }
        // GHOST-BAND FAIL-LOUD: a poisoned boundary leaves a non-finite ghost that FOFC never touches;
        // force the rate to +inf (dt -> 0, halt) if any zone in the allocated domain is non-finite.
        if !crate::regimes::substrate_kernels::state_finite_over_allocated(
            sim,
            pre_bind,
            &self.cfl_scratch,
        ) {
            lambda_max = f64::INFINITY;
        }
        let dt = cfl_from_lambda(lambda_max, self.cfl_number);
        // the parabolic viscous cap dt <= C_visc dx^2 / nu (C_visc = 0.1), ON TOP of the wave +
        // resistive rate. cartesian 3D only (where MHD viscosity is built), so coordinate dx is
        // physical. the resistive and viscous limits stack: a resistive-viscous run is bounded by
        // whichever diffusion is stiffer.
        let dt = if self.viscosity > 0.0 {
            const C_VISC: f64 = 0.1;
            let min_dx = geom.dx.iter().copied().fold(f64::INFINITY, f64::min);
            dt.min(C_VISC * min_dx * min_dx / self.viscosity)
        } else {
            dt
        };
        crate::regimes::substrate_kernels::body_gravity_limited_dt(
            sim,
            dt,
            self.cfl_number,
            max_inv_physical_width(geom).recip(),
        )
    }

    // ---- regime-agnostic tails: the gas godunov + the full CT stack (shared AOT kernels) ----
    fn godunov_stage(&self, sim: &FieldStore<D, 3, Mem, Sc>, dt: f64, a0: f64, ac: f64) {
        crate::regimes::mhd_substrate::godunov_stage(
            sim,
            R::SPEC.has_energy,
            Self::kernel_prefix(),
            self.eos_param,
            dt,
            a0,
            ac,
        );
    }
    fn ghost_fill(&self, sim: &FieldStore<D, 3, Mem, Sc>) {
        crate::regimes::mhd_substrate::ghost_fill(sim, R::SPEC.has_energy);
        // the standard fill skips Driven faces (Driven -> Skip); prescribe
        // their ghost prim state (incl. the cell B_phi) from the registered boundary DAGs.
        if !self.boundary_dags.is_empty() {
            dispatch_driven_boundaries(sim, &self.boundary_dags);
        }
    }
    fn snapshot(&self, sim: &FieldStore<D, 3, Mem, Sc>) {
        crate::regimes::mhd_substrate::snapshot(sim, R::SPEC.has_energy);
    }
    fn efield(&self, sim: &FieldStore<D, 3, Mem, Sc>) {
        // UCT needs the per-cell Riemann wave speeds; only regimes that materialize them (RMHD)
        // can use it today, so fall back to Contact otherwise (nmhd/imhd).
        let materialized = R::SPEC.materializes_wave_speeds
            || (self.ct_method == CtMethod::Uct
                && matches!(Self::kernel_prefix(), "nmhd" | "imhd"));
        let method = if self.ct_method == CtMethod::Uct && !materialized {
            CtMethod::Contact
        } else {
            self.ct_method
        };
        crate::regimes::mhd_substrate::efield(
            sim,
            method,
            self.solver,
            Self::kernel_prefix(),
            self.eos_param,
            self.theta,
        );
    }
    fn post_godunov(&self, sim: &FieldStore<D, 3, Mem, Sc>, dt: f64, stage: u8) {
        crate::regimes::mhd_substrate::post_godunov(
            sim,
            R::SPEC.has_energy,
            dt,
            stage,
            self.resistivity,
        );
    }

    fn has_additive_source(&self) -> bool {
        self.runtime_source.is_some()
    }

    fn snapshot_stage(&self, sim: &FieldStore<D, 3, Mem, Sc>) {
        // capture the stage-input gas cons into u_stage so source_apply reads the
        // pre-godunov state. without this, u_stage is zero and the
        // force lift S_mom = rho*a reads rho=0 -> NaN.
        crate::regimes::mhd_substrate::snapshot_stage(sim, R::SPEC.has_energy);
    }

    fn source_apply(&self, sim: &FieldStore<D, 3, Mem, Sc>, weight: f64) {
        // two-pass: add the user source increment to the hydro conserved slots after
        // the gas godunov. dispatch_runtime_source is regime-agnostic (targets
        // den/mom/nrg by name); B stays under CT, untouched here.
        if let Some(rs) = &self.runtime_source {
            dispatch_runtime_source(sim, rs, weight);
        }
    }
}
