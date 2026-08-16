// =============================================================================
// regimes/substrate_newton.rs
//
// AdiabaticSubstrateKernelSet<Mem, Sc, const D> — the D-GENERIC adiabatic (ideal-gas
// Newtonian Euler) KernelSet, every method dispatched to a build-time AOT substrate
// kernel through the structured binding ABI, the instance resolved by name (regime,
// ndim, dir) via the generated `kernel_by_name` registry. one struct serves 1D/2D/3D.
//
// it shares the EOS-generic substrate kernels with the RHD set — the godunov /
// snapshot / rk2 are the SAME `{prefix}_*` builders — and adds the genuinely
// regime-specific pieces: the closed-form adiabatic c2p (`p` from the energy) and
// the adiabatic face flux. the CFL wave-speed map is the iso one
// (`iso_wave_speed_map_{D}d`: cs = sqrt(gamma*p/rho), folding ALL D velocity
// components + per-axis inv_dx — D-correct), and the ghost fill is the SHARED
// lattice-map pullback (`iso_ghost_fill_{D}d`).
//
// a Newtonian sim allocates cons.nrg, prim.pre, flux.nrg (has_energy), so the
// dispatch wires sim.fields directly — no substrate-owned pressure field.
//
// the scheme is validated by the adiabatic (ideal-gas) Euler Sod shock-tube
// through the real evolve() loop in tests/substrate_adiabatic_sod.rs.
// =============================================================================

use symbi_algebra::{Domain, OrderedNumeric};
use symbi_grid::Field;
use symbi_hydro::source_spec::BuiltSource;
use symbi_ir::ScalarRef;
use symbi_ir::algebra::Scalar;
use symbi_xpu::MemorySpace;

use std::sync::{Arc, OnceLock};

use crate::kernels::support::{GhostFillDriver, to_bc_array};
use crate::regimes::substrate_kernels::{
    FusedCpuKernel, FusedSourceBinding, GradientBc, RuntimeSource, ScalarBind, Solver,
    body_fused_in, cfl_wave_speed, dispatch_body_feedback, dispatch_body_source,
    dispatch_body_source_wb,
    dispatch_c2p_status, dispatch_driven_boundaries, dispatch_fields, dispatch_flux,
    dispatch_fused_runtime_cpu, dispatch_godunov_maybe_fused, dispatch_gradient_boundaries,
    dispatch_named, dispatch_penalize, dispatch_runtime_source, dispatch_source_apply,
    FluxSpec, dof_lift_suffix, fused_runtime_cpu_kernel, geom_scalar, resolve_body_only_fused,
    resolve_params,
    motion_scalar,
    scalars_for,
};
use symbi_discretize::gv::GeoSource;
use symbi_sim::state::FieldStore;
use symbi_sim::substrate_seam::{KernelSet, WithViscosity};

/// a D-generic adiabatic (ideal-gas Euler) `KernelSet`, every method substrate-generated.
// the ADIABATIC viscous operator books BOTH the Navier-Stokes shear force AND the viscous heating
// (div(tau.v) onto the total energy); constant-nu, cartesian 2D.
impl<Mem: MemorySpace, Sc: Scalar + OrderedNumeric, const D: usize> WithViscosity
    for AdiabaticSubstrateKernelSet<Mem, Sc, D>
{
    fn with_viscosity(mut self, nu: f64) -> Self {
        self.viscosity = nu;
        self
    }
    fn with_alpha(mut self, alpha: f64) -> Self {
        // nu = alpha (gamma p/rho) / Omega_K with the LOCAL sound speed — the
        // shakura-sunyaev prescription on a varying-cs gas.
        self.alpha = alpha;
        self
    }
}

// horizon excision is a GR-chart operation; the adiabatic (flat) set ignores it.
impl<Mem: MemorySpace, Sc: Scalar + OrderedNumeric, const D: usize>
    symbi_sim::substrate_seam::WithExcision for AdiabaticSubstrateKernelSet<Mem, Sc, D>
{
}

// resistivity is an MHD operation; the pure-hydro adiabatic set has no magnetic field, ignores it.
impl<Mem: MemorySpace, Sc: Scalar + OrderedNumeric, const D: usize>
    symbi_sim::substrate_seam::WithResistivity for AdiabaticSubstrateKernelSet<Mem, Sc, D>
{
}

pub struct AdiabaticSubstrateKernelSet<
    Mem: MemorySpace,
    Sc: Scalar + OrderedNumeric,
    const D: usize,
> {
    pub gamma: f64,
    pub cfl_number: f64,
    /// the theta-MC reconstruction compression (regime-generic; 1 == plain minmod).
    pub theta: f64,
    pub cfl_scratch: Field<Sc, D, Mem>,
    /// declarative routing to the AOT-baked fused godunov
    /// (`adiabatic_godunov_euler_with_{source_id}_{D}d`). `None` => the unfused
    /// kernel (the original default; bit-identical for existing callers). `Some`
    /// => `godunov_euler` / `godunov_rk2` route through the fused variant.
    pub fused_source: Option<FusedSourceBinding>,
    /// the NON-fused (additive) source overlay — plain godunov + a
    /// per-stage `adiabatic_source_with_{slug}_{D}d` pass. proven bit-for-bit
    /// equal to `fused_source` with the same binding. see the iso analogue.
    pub additive_source: Option<FusedSourceBinding>,
    /// a RUNTIME-loaded user source (python -> json -> `build_user_source` ->
    /// `SourceEvaluator::from_built`). `Some` => `source_apply` runs a per-cell CPU pass that
    /// interprets the user `BuiltSource`(s) and adds `weight*S` to cons — NO recompile, NO
    /// AOT-baked kernel. CPU-only (host memory); the gpu path is runtime NVRTC (future).
    pub runtime_source: Option<Arc<RuntimeSource>>,
    /// when true AND a `runtime_source` is attached, the runtime user source is
    /// FUSED into the godunov stage as ONE Cranelift-JIT'd host kernel (cooling -> RT MZCS) instead
    /// of the two-pass (plain godunov + per-cell `apply_runtime_source`). opt-in + gated: falls back
    /// to the two-pass on non-f64 carriers, device memory, or an out-of-JIT-subset source. the
    /// two-pass remains the default; the fused path is bit-for-bit identical to it.
    pub fuse_runtime: bool,
    /// the drain timescale dial tau = c_drain dx / c_s: a
    /// convergence-study parameter, never tuned to a target rate.
    pub c_drain: f64,
    /// the BODY-ONLY fused godunov kernel (godunov + geo + immersed-body wrap, no user source),
    /// resolved once + cached here on the kernel-set — a gravity/accretion run with no runtime source
    /// to carry the body. built lazily on first `godunov_stage` (geometry + bodies known then), gated
    /// host+f64 + `fuse_runtime`. `Some(None)` = out-of-JIT-subset -> the two-pass body pass runs.
    pub(crate) fused_rhs: OnceLock<Option<FusedCpuKernel>>,
    /// driven-boundary DAGs, indexed by the `BoundaryType::Driven(id)` id the
    /// sim's `Boundaries` carries. each prescribes a face's ghost prim state. empty => no driven
    /// faces (the standard ghost-fill is the whole story). reuses `RuntimeSource` as the holder (same
    /// data); the `(Coord, Assign)` dispatch is what makes it a boundary.
    pub boundary_dags: Vec<Arc<RuntimeSource>>,
    /// gradient-boundary coefficients (Neumann / Robin), indexed by the `BoundaryType::Neumann(id)` /
    /// `Robin(id)` id. the convenience short-circuit for prescribed-gradient / mixed walls; empty =>
    /// none. the general path is a driven boundary.
    pub gradient_bcs: Vec<GradientBc>,
    /// Riemann solver — HLLE (default, two-wave) or HLLC (contact-resolving).
    /// see [[Solver]]. tunable via `.with_solver(Solver::Hllc)`.
    pub solver: Solver,
    /// evolution face reconstruction — the plm family (default; runtime theta
    /// selects theta-MC / van leer / pcm-at-zero) or the ppm monotonized
    /// parabola (`_ppm` kernels, -3..+2 stencil, requires an ng >= 3 allocation
    /// and a uniform cartesian grid). tunable via `.reconstruction(Recon::Ppm)`.
    pub recon: symbi_discretize::Recon,
    /// ppm flatten dials (onset, full); (0, 0) = pure parabola. see `ppm_flatten`.
    pub flatten: (f64, f64),
    /// the reference mach number the PUBLISHED low-mach ramp saturates at. read only by
    /// `Solver::HllcLm`; inert on every other solver. see `mach_limit`.
    pub mach_limit: f64,
    /// whether the face reconstruction limits the STATE or its departure from local hydrostatic
    /// equilibrium. see `balance`.
    pub balance: symbi_discretize::coords::Balance,
    /// consecutive substages the FOFC freeze tier fired (persistent-freeze fail-loud; see fofc.rs).
    pub freeze_streak: std::sync::atomic::AtomicU32,
    /// constant kinematic viscosity nu. 0 = inviscid. >0 runs the Navier-Stokes shear PLUS the
    /// viscous heating (div(tau.v) onto the total energy) and caps dt at C_visc dx^2 / nu. cartesian
    /// 2D (the adiabatic energy-flux kernel).
    pub viscosity: f64,
    /// shakura-sunyaev alpha with the LOCAL sound speed: nu = alpha (gamma p/rho) / Omega_K
    /// about immersed body 0. takes precedence over the constant nu when positive.
    pub alpha: f64,
}

impl<Mem: MemorySpace, Sc: Scalar + OrderedNumeric, const D: usize>
    AdiabaticSubstrateKernelSet<Mem, Sc, D>
{
    pub fn new(gamma: f64, cfl_number: f64, alloc_domain: &Domain<D>) -> Self {
        let cfl_scratch = Field::<Sc, D, Mem>::zeros(alloc_domain)
            .expect("failed to allocate adiabatic CFL scratch field");
        // theta defaults to 1.0 (plain minmod) — exact prior behavior; set `theta` to tune.
        Self {
            gamma,
            cfl_number,
            theta: 1.0,
            cfl_scratch,
            fused_source: None,
            additive_source: None,
            runtime_source: None,
            fuse_runtime: false,
            c_drain: 1.0,
            fused_rhs: OnceLock::new(),
            boundary_dags: Vec::new(),
            gradient_bcs: Vec::new(),
            solver: Solver::Hlle,
            flatten: (0.0, 0.0),
            mach_limit: symbi_hydro::dissipation::MACH_LIMIT,
            balance: symbi_discretize::coords::Balance::Plain,
            recon: symbi_discretize::Recon::Plm,
            freeze_streak: std::sync::atomic::AtomicU32::new(0),
            viscosity: 0.0,
            alpha: 0.0,
        }
    }

    /// attach a RUNTIME-loaded user source from already-lowered `(target, BuiltSource)`
    /// pairs (the `build_user_source` output of a python -> json `SourceConfig`). ONE source of
    /// truth: the substrate derives BOTH the CPU per-cell interpreter (`SourceEvaluator::from_built`)
    /// AND — lazily, on first device dispatch — the GPU IR (the SAME `source_apply_from_built_gv`
    /// builder build.rs AOT-bakes), so the host runs the interpreter and the device runs an
    /// NVRTC-JIT kernel, both from the same DAG, no recompile. `params` are the DAG's `p{i}` knobs.
    /// `let built = build_user_source(&cfg, has_energy)?;`
    /// `let sub = sim.substrate().with_runtime_source(built, cfg.params.clone());`
    pub fn with_runtime_source(
        mut self,
        built: Vec<(String, BuiltSource)>,
        params: Vec<f64>,
    ) -> Self {
        // has_energy = true is the AUTHORITY here (Newtonian's RegimeSpec) —
        // the set IS the regime. validation of the source vs the regime happened at
        // `build_user_source(cfg, &NEWTONIAN_SPEC)`.
        self.runtime_source = Some(RuntimeSource::new(built, params, true));
        self
    }

    /// attach a runtime user source AND route it through the FUSED host path — the
    /// source rides INSIDE the Cranelift-JIT'd godunov stage (one launch).
    /// same source, faster execution; bit-for-bit identical to `with_runtime_source` (the two-pass).
    /// host + f64 only; otherwise it transparently falls back
    /// to the two-pass.
    pub fn with_fused_runtime_source(
        mut self,
        built: Vec<(String, BuiltSource)>,
        params: Vec<f64>,
    ) -> Self {
        self.runtime_source = Some(RuntimeSource::new(built, params, true));
        self.fuse_runtime = true;
        self
    }

    /// enable source fusion WITHOUT a user source: an immersed body (gravity + accretion) folds into
    /// the godunov stage on host+f64, sparing a separate `body_source` pass. the body-only twin of
    /// `with_fused_runtime_source`; bit-identical to the two-pass, falls back off-host / non-f64 /
    /// JIT-miss. the production (py) path sets this so a pure-gravity run fuses; the direct-construction
    /// test path leaves it off (the two-pass body reference).
    pub fn with_source_fusion(mut self) -> Self {
        self.fuse_runtime = true;
        self
    }

    /// the body-only fused-kernel build state, for tests/introspection: `None` = never attempted (no
    /// bodies / fusion off / a user source carried the body instead); `Some(true)` = the body-only
    /// godunov+body kernel JIT-compiled and is live; `Some(false)` = out-of-JIT-subset -> the two-pass
    /// body ran. the fused-vs-two-pass equivalence check asserts `Some(true)` so a silent fallback
    /// can't make `fused == two-pass` pass vacuously.
    pub fn body_only_fused_state(&self) -> Option<bool> {
        self.fused_rhs.get().map(|o| o.is_some())
    }

    /// register a DRIVEN boundary. the returned id (registration order, 0-based)
    /// is what the sim's `Boundaries` must carry as `BoundaryType::Driven(id)` on the prescribed
    /// face. build `built` from `expr_bridge::build_boundary_dag(&cfg, &NEWTONIAN_SPEC)` (a complete
    /// prim prescription `[rho, vel.., pre]`). after the standard ghost-fill skips the driven faces,
    /// `ghost_fill` prescribes their ghost state from this DAG.
    /// `let sub = sim.substrate().with_driven_boundary(built, cfg.params.clone()).0;`
    pub fn with_driven_boundary(
        mut self,
        built: Vec<(String, BuiltSource)>,
        params: Vec<f64>,
    ) -> (Self, u16) {
        let id = self.boundary_dags.len() as u16;
        // has_energy = true (Newtonian) — the boundary prescribes prim.pre as well as rho/vel.
        self.boundary_dags
            .push(RuntimeSource::new(built, params, true));
        (self, id)
    }

    /// register a NEUMANN / ROBIN gradient boundary (the convenience short-circuit). the returned id
    /// (registration order) is what the sim's `Boundaries` carries as `BoundaryType::Neumann(id)` /
    /// `Robin(id)` on the prescribed face. `coeffs` are the per-variable coefficients in prim order
    /// `[rho, vel.., pre]`. after the standard ghost-fill skips these faces, `ghost_fill` fills them
    /// from the boundary-adjacent interior cell + these coefficients.
    /// `let sub = sim.substrate().with_gradient_boundary(GradientBc::Neumann(qs)).0;`
    pub fn with_gradient_boundary(mut self, coeffs: GradientBc) -> (Self, u16) {
        let id = self.gradient_bcs.len() as u16;
        self.gradient_bcs.push(coeffs);
        (self, id)
    }

    /// bind a fused-source AOT kernel for this kernel-set.
    /// fluent builder; chain off `new(..)`:
    /// `AdiabaticSubstrateKernelSet::new(gamma, cfl, alloc).with_fused_source(b)`.
    pub fn with_fused_source(mut self, binding: FusedSourceBinding) -> Self {
        self.fused_source = Some(binding);
        self
    }

    /// bind the SAME source as a NON-fused additive pass instead of
    /// fusing it into godunov. bit-for-bit equal to `with_fused_source` for a
    /// baked family. fluent builder; chain off `new(..)`.
    pub fn with_additive_source(mut self, binding: FusedSourceBinding) -> Self {
        self.additive_source = Some(binding);
        self
    }

    /// pick the Riemann solver. default is HLLE (unsuffixed kernels); HLLC
    /// routes to the `adiabatic_face_flux_hllc_*` AOT variants. fluent builder.
    /// rejects a solver that is invalid for the Newtonian regime (e.g., HLLD).
    pub fn with_solver(mut self, solver: Solver) -> Result<Self, symbi_sim::state::ConfigError> {
        let regime = crate::regimes::substrate_kernels::RegimeKind::of::<
            Sc,
            D,
            symbi_hydro::newtonian::Newtonian,
        >();
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

    /// select the evolution face reconstruction (plm family or ppm). fluent. ppm
    /// dispatches the `_ppm` kernel twins: flat cartesian only, allocation ng >= 3
    /// (`.ghosts(3)` at sim build); a refinement hierarchy requires the quartic
    /// coarse-fine prolongation (ppm evolution -> quartic, the degree ladder).
    /// theta is ignored under ppm — the parabola carries its own monotonicity
    /// constraint.
    pub fn reconstruction(mut self, recon: symbi_discretize::Recon) -> Self {
        self.recon = recon;
        self
    }

    /// the ppm convergence-gated flatten dials (onset, full) in units of
    /// compression per cell crossing over the isothermal sound speed. the
    /// default (0, 0) is the PURE parabola; a gravity-sink run declares its own
    /// dials to close the smooth-infall entropy vent (full flatten by the
    /// standing-layer strength), and a trans-sonic turbulence run leaves them
    /// off — an active flatten there degrades the parabola to first order in
    /// every eddy collision. inert under plm (the kernels never declare the
    /// scalars). fluent.
    /// set the reference mach number the PUBLISHED low-mach ramp saturates at (default
    /// `MACH_LIMIT` = 0.1, the value used throughout Fleischmann, Adami & Adams 2020).
    /// the ramp reduces acoustic dissipation only BELOW this number, so it decides how much
    /// of the flow the reduction reaches: a deeply subsonic problem whose entire range sits
    /// under 0.1 is untouched by the published value and needs it raised to meet the flow.
    /// 0 reduces nothing and recovers classical HLLC; 1 reduces all the way to the sonic
    /// point. read only by `Solver::HllcLm` -- the retired clamped arm froze its own
    /// pressure ceiling from the compile-time constant and holds them consistent. fluent.
    /// reconstruct each cell's DEPARTURE from the local hydrostatic profile rather than the
    /// state, so a discretely balanced atmosphere presents no face jump and a low-dissipation
    /// riemann solver has no residual to leave undamped.
    ///
    /// orthogonal to the solver: it composes with any of them, and the first-order FOFC redo
    /// inherits it (that redo runs HLLE at theta = 0, and a piecewise-constant reconstruction of
    /// departures is exactly balanced). exact on a locally isentropic hydrostatic column;
    /// degrades linearly in the entropy variation off one. costs a per-stencil body-potential
    /// evaluation, so it is OFF by default and worth measuring before a long run. fluent.
    pub fn well_balanced_reconstruction(mut self, on: bool) -> Self {
        self.balance = if on {
            symbi_discretize::coords::Balance::Hydrostatic
        } else {
            symbi_discretize::coords::Balance::Plain
        };
        self
    }

    pub fn mach_limit(mut self, mach_limit: f64) -> Self {
        assert!(
            (0.0..=1.0).contains(&mach_limit),
            "mach_limit must lie in [0, 1]; got {mach_limit}. above 1 the ramp would scale \
             acoustic dissipation on supersonic faces, where the star states it multiplies \
             are no longer the subsonic intermediate flux"
        );
        self.mach_limit = mach_limit;
        self
    }

    pub fn ppm_flatten(mut self, onset: f64, full: f64) -> Self {
        self.flatten = (onset, full);
        self
    }

    /// eos closure selector. the newtonian family is gamma-law only — the synge
    /// (taub-mathews) closure is relativistic — so this validates rather than
    /// stores; a shared regime-generic build path may call it unconditionally.
    pub fn with_eos(self, eos: symbi_discretize::EosArm) -> Self {
        assert!(
            eos == symbi_discretize::EosArm::IdealGamma,
            "the synge-gas closure applies to the rhd regime only"
        );
        self
    }
}

impl<Mem: MemorySpace + Sync, Sc: Scalar + OrderedNumeric, const D: usize>
    AdiabaticSubstrateKernelSet<Mem, Sc, D>
{
    /// the evolution face-flux scheme assembled from this set's dials.
    fn flux_spec(&self) -> FluxSpec {
        FluxSpec {
            theta: self.theta,
            solver: self.solver,
            recon: self.recon,
            eos: symbi_discretize::EosArm::IdealGamma,
            flatten: self.flatten,
            mach_limit: self.mach_limit,
            balance: self.balance,
            rusanov: false,
        }
    }
}

impl<Mem: MemorySpace + Sync, Sc: Scalar + OrderedNumeric, const D: usize, const DOF: usize>
    KernelSet<D, DOF, Mem, Sc> for AdiabaticSubstrateKernelSet<Mem, Sc, D>
{
    fn flux(&self, sim: &FieldStore<D, DOF, Mem, Sc>, dir: usize) {
        let pre = sim
            .fields
            .prim
            .pre_field()
            .expect("Newtonian requires prim.pre");
        dispatch_flux(sim, pre, "adiabatic", dir, self.gamma, self.flux_spec());
    }

    fn hydrostatic_balance(&self) -> bool {
        self.balance == symbi_discretize::coords::Balance::Hydrostatic
    }

    fn reconstruction_reach(&self) -> u8 {
        match self.recon {
            symbi_discretize::Recon::Plm => 2,
            symbi_discretize::Recon::Ppm => 3,
        }
    }

    fn c2p(&self, sim: &FieldStore<D, DOF, Mem, Sc>) {
        // the primitives now hold a state recovered from the conserved fields; anything
        // reading prim.* outside the evolve loop checks this before trusting it.
        sim.mark_primitives_recovered();
        // cons -> prim, all DOF velocities; the manifest binds cons.* -> prim.* (the cyl
        // c2p writes prim.vel_0..2, automatic). "prim.pre" resolves to the prim pressure.
        let pre = sim
            .fields
            .prim
            .pre_field()
            .expect("Newtonian requires prim.pre");
        let sfx = dof_lift_suffix(sim.geom.coords, DOF, D);
        let name = format!("adiabatic_c2p{sfx}_{D}d");
        let scalars = scalars_for(&name, |bind| match bind {
            ScalarBind::Ref(ScalarRef::Gamma) => Sc::from_f64(self.gamma),
            o => panic!("adiabatic c2p: unexpected scalar {o:?}"),
        });
        dispatch_named(sim, pre, None, 0, &name, &sim.geom.interior, &[], &scalars);
        dispatch_c2p_status(sim, pre, "adiabatic", sfx);
    }

    fn godunov_stage(&self, sim: &FieldStore<D, DOF, Mem, Sc>, dt: f64, a0: f64, ac: f64) {
        let pre = sim
            .fields
            .prim
            .pre_field()
            .expect("Newtonian requires prim.pre");
        // the FUSED runtime-source path: one Cranelift-JIT'd godunov+source launch, replacing
        // (plain godunov + the separate `apply_runtime_source` pass). gated host+f64; the source's
        // own pass is skipped in `source_apply` under the SAME predicate. geo = Hydro{inertial:true}
        // matches the AOT `adiabatic` godunov (geo_source), so the two-pass fallback is exact.
        // a dyed run stays OFF the fused body path: the fused godunov carries den/mom/nrg only,
        // so folding the body in there would drain mass without its dye and raise the concentration
        // of the gas left behind. fusion is a launch-count optimization, not semantics, so the
        // two-pass body kernels (which have dyed twins) carry a dye correctly at a small cost.
        if self.fuse_runtime && !sim.has_passive_scalar() {
            let geo = GeoSource::Hydro { inertial: true };
            match &self.runtime_source {
                Some(rs) => {
                    if let Some(fk) = fused_runtime_cpu_kernel(sim, rs, geo, true) {
                        dispatch_fused_runtime_cpu(sim, pre, fk, Some(rs), dt, a0, ac, self.gamma);
                        return;
                    }
                }
                // no user source, but an immersed body still folds into godunov (gravity + drain).
                None => {
                    if let Some(fk) = resolve_body_only_fused(sim, &self.fused_rhs, true, geo) {
                        dispatch_fused_runtime_cpu(sim, pre, fk, None, dt, a0, ac, self.gamma);
                        return;
                    }
                }
            }
        }
        dispatch_godunov_maybe_fused(
            sim,
            pre,
            "adiabatic",
            dt,
            a0,
            ac,
            self.gamma,
            self.fused_source.as_ref(),
        );
    }

    fn cfl(&self, sim: &FieldStore<D, DOF, Mem, Sc>) -> f64 {
        // adiabatic shares the iso wave-speed map (cs = sqrt(gamma*p/rho), gamma=1.4 vs 1);
        // the SHARED cfl dispatch binds the field buffers by manifest + owns the reduction.
        let pre = sim
            .fields
            .prim
            .pre_field()
            .expect("Newtonian requires prim.pre");
        let dt = cfl_wave_speed(
            sim,
            pre,
            &self.cfl_scratch,
            "iso",
                        symbi_discretize::EosArm::IdealGamma,
            self.gamma,
            self.cfl_number,
            None,
            0.0,
        );
        // the parabolic viscous cap: an explicit momentum-diffusion step is stable for
        // dt <= C_visc dx^2 / nu (C_visc = 0.1 below the ~0.21 von-Neumann limit of the 4/3 normal
        // stress). cartesian 2D only, so the coordinate dx IS the physical cell width. inert inviscid.
        // alpha: nu grows with radius AND with the local cs^2, so the cap bounds it
        // by the largest sound speed at the slowest orbit (adiabatic_alpha_nu_max).
        let nu_max = if self.alpha > 0.0 {
            crate::regimes::substrate_kernels::adiabatic_alpha_nu_max(sim, self.alpha, self.gamma)
        } else {
            self.viscosity
        };
        let dt = if nu_max > 0.0 {
            const C_VISC: f64 = 0.1;
            let min_dx = sim.geom.dx.iter().copied().fold(f64::INFINITY, f64::min);
            dt.min(C_VISC * min_dx * min_dx / nu_max)
        } else {
            dt
        };
        let min_physical_width = super::substrate_mhd::max_inv_physical_width(&sim.geom).recip();
        crate::regimes::substrate_kernels::body_gravity_limited_dt(
            sim,
            dt,
            self.cfl_number,
            min_physical_width,
        )
    }

    fn viscous(&self, sim: &FieldStore<D, DOF, Mem, Sc>, dt: f64) {
        // alpha (local-cs shakura-sunyaev) takes precedence over the constant nu.
        if self.alpha > 0.0 {
            crate::regimes::substrate_kernels::dispatch_viscous_alpha(
                sim, dt, self.alpha, self.gamma,
            );
            return;
        }
        if self.viscosity <= 0.0 {
            return;
        }
        // the adiabatic viscous operator (shear force + viscous heating onto the
        // total energy): cartesian 2D/3D flat kernels, curvilinear 2D / 2.5D / 3D
        // through the shared scale-factor operator family. dispatch's
        // (chart, D, DOF) match fails loud on an unbaked combination.
        crate::regimes::substrate_kernels::dispatch_viscous(sim, dt, self.viscosity);
    }

    fn ghost_fill(&self, sim: &FieldStore<D, DOF, Mem, Sc>) {
        // the SHARED lattice-map pullback (iso_ghost_fill{sfx}_{D}d): rho/vel/pre, in-place,
        // bound by manifest. (DOF>NDIM: the cyl ghost manifest carries no per-axis BC entry,
        // but the dispatch path is regime-uniform.)
        let bc = to_bc_array::<D>(&sim.boundaries);
        let pre = sim
            .fields
            .prim
            .pre_field()
            .expect("Newtonian requires prim.pre");
        let sfx = dof_lift_suffix(sim.geom.coords, DOF, D);
        // under a BALANCED reconstruction the ghosts must satisfy the same premise the
        // interior does: a mirrored copy of a stratified column is not its continuation, so
        // reflect/outflow ghosts extend (rho, p) along the local isentrope to the ghost's own
        // potential (velocity mirrors as always). without this the wall face presents pure
        // boundary-artifact departures and the balanced scheme fights the wall -- measured as
        // a 1.5e-2 floor loss against the interior's 2.2e-8.
        let name = if self.balance == symbi_discretize::coords::Balance::Hydrostatic {
            assert!(
                DOF == D,
                "the balance-aware ghost fill is baked for DOF == D cartesian grids only"
            );
            // a periodic cut identifies two positions whose body potentials generally
            // differ, and the isentrope extension from one to the other is a real state
            // change smuggled through a boundary that claims to be an identification.
            // exact only when the potential is periodic too, which a body potential is
            // not -- refused rather than approximated.
            let has_bodies = sim
                .immersed
                .as_ref()
                .is_some_and(|im| !im.bodies.is_empty());
            if has_bodies {
                for axis_bc in bc.iter() {
                    for face_bc in axis_bc.iter() {
                    assert!(
                        *face_bc != symbi_grid::ghost::BcType::Periodic,
                        "balance-aware ghost fill under a PERIODIC boundary with a body \
                         potential: the periodic images sit at different potentials, so \
                         the isentrope extension across the cut is unsound. use outflow \
                         or reflect walls, or run the plain reconstruction"
                    );
                    }
                }
            }
            format!("wb_ghost_fill_{D}d")
        } else {
            format!("iso_ghost_fill{sfx}_{D}d")
        };

        GhostFillDriver::<D>::new(&sim.geom.allocated, &sim.geom.interior, bc).drive_sweep(
            |region, p| {
                // params BY NAME via the type-sorted manifest: map_type/arg are INT lanes (the
                // `ints` tail), vel_sign FLOAT (the `scalars` tail) — each routed by the kernel's
                // declared sort. the int \sqcup float coproduct that defeated positional scalars_for.
                let (ints, scalars) = resolve_params(
                    &name,
                    |bind| match bind {
                        ScalarBind::Ref(ScalarRef::MapType(ax)) => p.map_type[*ax as usize] as i32,
                        ScalarBind::Ref(ScalarRef::Arg(ax)) => p.arg[*ax as usize],
                        o => panic!("ghost_fill: unexpected int param {o:?}"),
                    },
                    |bind| match bind {
                        ScalarBind::Ref(ScalarRef::VelSign(ax)) => {
                            Sc::from_f64(p.vel_sign[*ax as usize])
                        }
                        ScalarBind::Ref(ScalarRef::Gamma) => Sc::from_f64(self.gamma),
                        // the balance-aware fill evaluates the body potential at the ghost
                        // and source centroids; same slots the body source binds.
                        ScalarBind::Ref(ScalarRef::Body { idx, field }) => {
                            let bodies = sim.immersed.as_ref().map(|im| &im.bodies);
                            Sc::from_f64(crate::regimes::substrate_kernels::body_scalar::<D>(
                                bodies, *idx, *field,
                            ))
                        }
                        // the balance-aware fill evaluates cell centroids through the
                        // face-position ladder, which declares the per-axis grid origin
                        // and spacing like any position-reading kernel.
                        ScalarBind::Ref(other) => Sc::from_f64(
                            geom_scalar(&sim.geom.x_lo, &sim.geom.dx, &sim.geom.maps, *other)
                                .unwrap_or_else(|| {
                                    panic!("ghost_fill: unexpected scalar {other:?}")
                                }),
                        ),
                        o => panic!("ghost_fill: unexpected scalar {o:?}"),
                    },
                );
                dispatch_named(sim, pre, None, 0, &name, &region.domain, &ints, &scalars);
            },
        );

        // the standard pullback skips any Driven faces (Driven -> BcType::Skip);
        // prescribe their ghost prim state from the registered boundary DAGs.
        if !self.boundary_dags.is_empty() {
            dispatch_driven_boundaries(sim, &self.boundary_dags);
        }
        // and the Neumann/Robin gradient faces (also skipped by the pullback), filled from the edge cell.
        if !self.gradient_bcs.is_empty() {
            dispatch_gradient_boundaries(sim, pre, &self.gradient_bcs, None);
        }
        // the dye concentration ghost band: a true scalar (reflect sign +1)
        // through the field-agnostic single-scalar pullback. gradient faces resolve to a
        // zero-derivative copy here, since a prescribed normal derivative is a per-primitive-variable
        // quantity and the dye carries none.
        if let Some(chi) = sim.fields.prim.chi_field() {
            crate::regimes::mhd_substrate::flag_ghost_fill(
                sim,
                chi,
                crate::kernels::support::to_bc_array_scalar::<D>(&sim.boundaries),
            );
        }
    }

    fn snapshot(&self, sim: &FieldStore<D, DOF, Mem, Sc>) {
        // u_n = cons (all DOF components), bound by manifest over the full allocated domain.
        let pre = sim
            .fields
            .prim
            .pre_field()
            .expect("Newtonian requires prim.pre");
        let sfx = dof_lift_suffix(sim.geom.coords, DOF, D);
        let name = format!("adiabatic_snapshot{sfx}_{D}d");
        dispatch_named(sim, pre, None, 0, &name, &sim.geom.allocated, &[], &[]);
        // the dye snapshot rides alongside: u_n.chi = cons.chi, the rk combine's
        // step-start state.
        if sim.has_passive_scalar() {
            let cname = format!("chi_snapshot_{D}d");
            dispatch_named(sim, pre, None, 0, &cname, &sim.geom.allocated, &[], &[]);
        }
    }

    fn chi_flux(&self, sim: &FieldStore<D, DOF, Mem, Sc>) {
        if !sim.has_passive_scalar() {
            return;
        }
        let pre = sim
            .fields
            .prim
            .pre_field()
            .expect("Newtonian requires prim.pre");
        // the divergence at an interior cell reads the flux at offset +1, so each axis is swept
        // over the interior grown by one cell on its own hi side.
        for dir in 0..D {
            let mut band = sim.geom.interior.clone();
            band.spaces[dir].hi += 1;
            let fname = format!("chi_flux_{dir}_{D}d");
            dispatch_named(sim, pre, None, dir, &fname, &band, &[], &[]);
        }
    }

    fn chi_update(&self, sim: &FieldStore<D, DOF, Mem, Sc>, dt: f64, a0: f64, ac: f64) {
        if !sim.has_passive_scalar() {
            return;
        }
        let pre = sim
            .fields
            .prim
            .pre_field()
            .expect("Newtonian requires prim.pre");
        // the dye divergence divides by the PHYSICAL cell width, so its geom scalars resolve
        // through the same motion-aware path the gas godunov uses: on a homologously expanding
        // mesh `dx` carries a(t) on the expanding axes, and the comoving width would be short by
        // exactly that factor. reproduces the raw linear (x_lo, dx) bit-identically on a static grid.
        let (x_lo_k, dx_k) = crate::regimes::substrate_kernels::kernel_geom(
            &sim.geom.x_lo,
            &sim.geom.dx,
            &sim.geom.maps,
            sim.geom.coords,
            sim.motion.a,
        );
        let name = format!("chi_godunov_{D}d");
        let scalars = scalars_for(&name, |bind| {
            let ScalarBind::Ref(sref) = bind else {
                panic!("chi_godunov: unexpected spec scalar {bind:?}");
            };
            Sc::from_f64(match *sref {
                ScalarRef::Dt => dt,
                ScalarRef::A0 => a0,
                ScalarRef::Ac => ac,
                other => motion_scalar(&sim.motion, sim.geom.coords, D, other)
                    .or_else(|| geom_scalar(&x_lo_k, &dx_k, &sim.geom.maps, other))
                    .unwrap_or_else(|| panic!("chi_godunov: unexpected scalar {other:?}")),
            })
        });
        dispatch_named(sim, pre, None, 0, &name, &sim.geom.interior, &[], &scalars);
        // the dye concentration recovery: prim.chi = cons.chi / den, against the
        // stage-final density (fofc has already spliced by this phase).
        let cname = format!("chi_c2p_{D}d");
        dispatch_named(sim, pre, None, 0, &cname, &sim.geom.interior, &[], &[]);
    }

    fn has_additive_source(&self) -> bool {
        self.additive_source.is_some() || self.runtime_source.is_some()
    }

    fn fofc_active(&self) -> bool {
        // the first-order flux correction covers the DOF == D charts (the fofc select's momentum
        // count is baked to ncomp = D).
        DOF == D
    }

    fn fofc(
        &self,
        sim: &FieldStore<D, DOF, Mem, Sc>,
        dt: f64,
        a0: f64,
        ac: f64,
        _stage: u8,
    ) -> bool {
        // FOFC covers the DOF == D charts only (the fofc kernels are baked at ncomp = D). `fofc` is
        // called unconditionally by the driver, so
        // the gate lives here (fofc_active only guards the stage-input snapshot).
        if DOF != D {
            return false;
        }
        // the first-order redo runs HLLE at theta = 0 (PCM) — the positivity-preserving Einfeldt
        // fan — regardless of the production solver (HLLC can undershoot in a strong rarefaction).
        let pre = sim
            .fields
            .prim
            .pre_field()
            .expect("Newtonian requires prim.pre");
        crate::regimes::fofc::fofc_orchestrate(
            sim,
            "adiabatic",
            "", // the DOF != D early-return means this path is always DOF == D
            <Self as KernelSet<D, DOF, Mem, Sc>>::has_additive_source(self),
            &self.cfl_scratch,
            pre,
            &self.freeze_streak,
            |dir| {
                dispatch_flux(
                    sim,
                    pre,
                    "adiabatic",
                    dir,
                    self.gamma,
                    self.flux_spec().first_order(),
                )
            },
            || self.c2p(sim),
            || self.godunov_stage(sim, dt, a0, ac),
            || self.source_apply(sim, ac * dt),
            || {
                if sim.immersed.is_some() {
                    self.body_source(sim, ac * dt)
                }
            },
            || {}, // newtonian: no admissible-boundary projection (keeps the freeze parachute)
            // freeze parachute evolves by the body source (adiabatic has the _with_body kernel).
            sim.immersed.is_some().then(|| (ac * dt, self.gamma)),
            crate::regimes::fofc::CtHooks::none(),
            || crate::regimes::fofc::SourceReplay::NotApplicable, // hydro: no source replay
            false, // no projection tier below the freeze; keep the parachute
        )
    }

    fn snapshot_stage(&self, sim: &FieldStore<D, DOF, Mem, Sc>) {
        // u_stage = cons (pure copy), positional [den, mom_0.., nrg] — the snapshot kernel's
        // manifest order (snapshot_gv) — over the interior the source pass iterates. explicit
        // buffers (not dispatch_named, whose manifest fixes the output to u_n).
        let nrg = sim.fields.cons.nrg_field().expect("adiabatic cons.nrg");
        let u_nrg = sim.workspace.u_stage.nrg_field().expect("u_stage.nrg");
        let mut inputs: Vec<&Field<Sc, D, Mem>> = vec![&sim.fields.cons.den];
        for k in 0..DOF {
            inputs.push(&sim.fields.cons.mom[k]);
        }
        inputs.push(nrg);
        let mut outputs: Vec<&Field<Sc, D, Mem>> = vec![&sim.workspace.u_stage.den];
        for k in 0..DOF {
            outputs.push(&sim.workspace.u_stage.mom[k]);
        }
        outputs.push(u_nrg);
        let sfx = dof_lift_suffix(sim.geom.coords, DOF, D);
        let name = format!("adiabatic_snapshot{sfx}_{D}d");
        dispatch_fields::<Sc, Mem, D>(
            &name,
            &sim.geom.allocated,
            &sim.geom.interior,
            &inputs,
            &outputs,
            &[],
            &[],
        );
    }

    fn source_apply(&self, sim: &FieldStore<D, DOF, Mem, Sc>, weight: f64) {
        if let Some(b) = &self.additive_source {
            dispatch_source_apply(sim, "adiabatic", b, weight);
        }
        if let Some(rs) = &self.runtime_source {
            // when the fused path is live (same predicate godunov_stage used), the source already
            // rode inside the godunov launch — skip the separate pass to avoid double-counting.
            if self.fuse_runtime
                && !sim.has_passive_scalar()
                && fused_runtime_cpu_kernel(sim, rs, GeoSource::Hydro { inertial: true }, true)
                    .is_some()
            {
                return;
            }
            dispatch_runtime_source(sim, rs, weight);
        }
    }

    fn body_source(&self, sim: &FieldStore<D, DOF, Mem, Sc>, dt: f64) {
        // when the fused stage carried the immersed body inside godunov (one launch), running the
        // standalone pass here would double-apply it. same predicate + geo the godunov stage used —
        // via the user-source fused kernel, or the body-only fused kernel when there is no user source.
        // the FUSED stage applies the ANALYTIC rho*g; under a balanced reconstruction that is the
        // wrong pairing, so balance forces the standalone equilibrium-difference pass.
        if self.balance == symbi_discretize::coords::Balance::Plain
            && self.fuse_runtime
            && !sim.has_passive_scalar()
        {
            let geo = GeoSource::Hydro { inertial: true };
            let absorbed = match &self.runtime_source {
                Some(rs) => body_fused_in(sim, rs, geo, true),
                None => resolve_body_only_fused(sim, &self.fused_rhs, true, geo).is_some(),
            };
            if absorbed {
                return;
            }
        }
        // forward immersed-body source: cons += dt*S, in-place. under a BALANCED
        // reconstruction the momentum source must be the equilibrium-pressure difference at
        // the cell faces, or the flux/source mismatch re-accelerates the very column the
        // reconstruction balanced -- measured as an O((dx/H)^2) per-step drift that walks a
        // sealed stagnant column to |v| ~ 3e-2 in 400 steps. the pair is the scheme.
        if self.balance == symbi_discretize::coords::Balance::Hydrostatic {
            dispatch_body_source_wb(sim, dt, self.gamma);
        } else {
            dispatch_body_source(sim, dt, self.gamma);
        }
    }

    fn penalize(&self, sim: &FieldStore<D, DOF, Mem, Sc>, dt: f64) {
        // the penalization drain is the sole accretion mechanism on
        // cartesian grids — the in-godunov sink resolves its rate to
        // zero under the same predicate (params::penalize_owns_accretion).
        dispatch_penalize(sim, dt, self.gamma, self.c_drain);
    }

    fn body_feedback(&self, sim: &FieldStore<D, DOF, Mem, Sc>, dt: f64) {
        // backward feedback: reduce per-body force/torque/mass -> diagnostics.
        dispatch_body_feedback(sim, dt, self.gamma);
    }
}
