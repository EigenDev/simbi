// =============================================================================
// regimes/substrate_rhd.rs
//
// RhdSubstrateKernelSet<Mem, Sc, const D> — the D-generic RHD (special-
// relativistic Euler) KernelSet, every method dispatched to a build-time AOT
// substrate kernel through the structured binding ABI (a KernelInvocation of
// ordered Buf handles, inputs-then-outputs, run_cpu / GPU JIT). the kernel instance
// is resolved by name (regime, ndim, dir) via the generated `kernel_by_name`
// registry — one struct serves 1D/2D/3D, no per-dimension copy.
//
// the relativistic pieces: the masked-Newton c2p (`rhd_c2p_{D}d`), the HLLE flux
// with relativistic U/F/wave speeds per sweep dir (`rhd_face_flux_{D}d_{dir}`), and
// the relativistic CFL wave-speed map (`rhd_wave_speed_map_{D}d`). the godunov /
// snapshot / rk2 are the same EOS-generic kernels as the Newtonian sets (D/S_k/tau ==
// den/mom/nrg in structure), and the ghost_fill is the shared lattice-map pullback
// (`iso_ghost_fill_{D}d`, the EOS-generic 3-field prim pullback).
//
// the wave-speed map is D-generic: it folds the per-axis relativistic Davis speed
// (vn = vel[d], shared cs) over all D axes with a per-axis inv_dx, so the CFL is the
// anisotropic-correct `cfl_from_lambda` — matching the iso/Newton convention via a
// per-axis wave-speed projection.
//
// the RHD scheme is validated by the Marti & Mueller relativistic Sod through
// the real evolve() loop in tests/substrate_rhd_sod.rs.
// =============================================================================

use symbi_algebra::{Domain, OrderedNumeric};
use symbi_carrier::Scalar;
use symbi_grid::Field;
use symbi_ir::ScalarRef;
use symbi_xpu::MemorySpace;

use std::sync::Arc;

use crate::kernels::support::{GhostFillDriver, to_bc_array};
use crate::regimes::substrate_kernels::{
    FluxSpec, GradientBc, RuntimeSource, ScalarBind, Solver, cfl_wave_speed,
    dispatch_driven_boundaries, dispatch_fields, dispatch_flux, dispatch_fused_runtime_cpu,
    dispatch_godunov, dispatch_gradient_boundaries, dispatch_runtime_source, dof_lift_suffix,
    fused_runtime_cpu_kernel, geom_scalar, gr_chart_dof_tag, kernel_geom, resolve_params,
    scalars_for, shell_accretion_rates, spacetime_slug,
};
use symbi_discretize::gv::GeoSource;
use symbi_geometry::Spacetime;
use symbi_sim::state::FieldStore;
use symbi_sim::substrate_seam::{KernelSet, WithExcision, WithViscosity};
use symbi_source_compile::source_spec::SourceProgram;

/// a D-generic RHD `KernelSet`, every method substrate-generated.
// viscosity is isothermal-only; RHD uses the no-op default.
impl<Mem: MemorySpace, Sc: Scalar + OrderedNumeric, const D: usize> WithViscosity
    for RhdSubstrateKernelSet<Mem, Sc, D>
{
}

// non-ideal resistivity is a Newtonian-MHD operation; the RHD set ignores it (no-op default).
impl<Mem: MemorySpace, Sc: Scalar + OrderedNumeric, const D: usize>
    symbi_sim::substrate_seam::WithResistivity for RhdSubstrateKernelSet<Mem, Sc, D>
{
}

impl<Mem: MemorySpace, Sc: Scalar + OrderedNumeric, const D: usize> WithExcision
    for RhdSubstrateKernelSet<Mem, Sc, D>
{
    fn with_excision(mut self, r_exc: f64, rho_scale: f64, pre_scale: f64) -> Self {
        self.excision_radius = r_exc;
        self.excision_rho = rho_scale * 1e-10;
        self.excision_pre = pre_scale * 1e-12;
        self
    }
}

pub struct RhdSubstrateKernelSet<Mem: MemorySpace, Sc: Scalar + OrderedNumeric, const D: usize> {
    pub gamma: f64,
    pub cfl_number: f64,
    /// the theta-MC reconstruction compression (regime-generic; 1 == plain minmod).
    pub theta: f64,
    pub cfl_scratch: Field<Sc, D, Mem>,
    /// Riemann solver — HLLE (default) or HLLC (contact-resolving, Mignone-Bodo
    /// 2005). tunable via `.with_solver(Solver::Hllc)`.
    pub solver: Solver,
    /// a runtime user source (python -> json `SourceConfig` -> `build_user_source`).
    /// RHD is relativistic, so only `kind="raw"` reaches here (the bridge rejects
    /// the newtonian force/cooling/relax lifts); the user supplies the conserved
    /// increment directly and it is added in `source_apply` via the regime-agnostic
    /// `dispatch_runtime_source`. None for source-free runs.
    pub runtime_source: Option<Arc<RuntimeSource>>,
    /// when true and a `runtime_source` is attached on a flat (Minkowski) background, the raw user
    /// source is fused into the godunov stage as one Cranelift-JIT'd host kernel, replacing the
    /// two-pass. host+f64 only; falls back to the two-pass off-host / non-f64 / JIT-miss / GR (the
    /// fused builder traces the flat geo, so it cannot match a curved godunov kernel).
    pub fuse_runtime: bool,
    /// driven (dynamic) boundary prescriptions, indexed by `BoundaryType::Driven(id)` — the
    /// complete prim state [rho, vel_0..DOF-1, pre] as coordinate DAGs, evaluated over the
    /// face's ghost band after the standard pullback skips it. a rotating (theta-stratified)
    /// equilibrium requires this: no local rule (mirror or copy) can represent the state
    /// beyond a wedge wall — only the analytic continuation can.
    pub boundary_dags: Vec<Arc<RuntimeSource>>,
    /// gradient-boundary (Neumann / Robin) coefficients, indexed by the `BoundaryType::Neumann(id)` /
    /// `Robin(id)` id — the convenience prescribed-gradient / mixed walls.
    pub gradient_bcs: Vec<GradientBc>,
    /// consecutive substages the FOFC freeze tier fired (persistent-freeze fail-loud; see fofc.rs).
    pub freeze_streak: std::sync::atomic::AtomicU32,
    /// the eos closure arm: `IdealGamma` (default, gamma-law) or `TaubMathews`
    /// (the synge-gas approximation, parameter-free — `gamma` stays bound but
    /// inert on that arm). selects the `_tm` kernel twins, baked for the flat
    /// cartesian rhd family only.
    pub eos: symbi_discretize::EosArm,
    /// the horizon-excision sphere radius about the chart origin (cartesian
    /// kerr-schild only); 0 disables the excision pass entirely.
    pub excision_radius: f64,
    /// density and pressure of the absorbing atmosphere inside the excision surface.
    pub excision_rho: f64,
    pub excision_pre: f64,
}

impl<Mem: MemorySpace, Sc: Scalar + OrderedNumeric, const D: usize>
    RhdSubstrateKernelSet<Mem, Sc, D>
{
    pub fn new(gamma: f64, cfl_number: f64, alloc_domain: &Domain<D>) -> Self {
        let cfl_scratch = Field::<Sc, D, Mem>::zeros(alloc_domain)
            .expect("failed to allocate RHD CFL scratch field");
        Self {
            gamma,
            cfl_number,
            theta: 1.0,
            cfl_scratch,
            solver: Solver::Hlle,
            runtime_source: None,
            fuse_runtime: false,
            boundary_dags: Vec::new(),
            gradient_bcs: Vec::new(),
            freeze_streak: std::sync::atomic::AtomicU32::new(0),
            eos: symbi_discretize::EosArm::IdealGamma,
            excision_radius: 0.0,
            excision_rho: 1.0,
            excision_pre: 1.0,
        }
    }

    /// ppm flatten dials. the rhd flux family is plm-only, so this validates
    /// rather than stores; a shared regime-generic build path may call it
    /// unconditionally with the dials off.
    pub fn ppm_flatten(self, onset: f64, full: f64) -> Self {
        assert!(
            full <= onset,
            "the ppm flatten applies to the newtonian adiabatic flux family only"
        );
        self
    }

    /// select the eos closure arm. the taub-mathews (synge-gas) closure is baked
    /// for the flat cartesian rhd family only — a curved spacetime or a
    /// non-cartesian chart refuses at dispatch. fluent.
    pub fn with_eos(mut self, eos: symbi_discretize::EosArm) -> Self {
        self.eos = eos;
        self
    }

    /// register a driven boundary. the returned id (registration order,
    /// 0-based) is what the sim's `Boundaries` must carry as `BoundaryType::Driven(id)` on the
    /// prescribed face. build `built` from `expr_bridge::build_boundary_dag(&cfg, RHD_SPEC)` —
    /// a complete prim prescription `[rho, vel.., pre]`.
    pub fn with_driven_boundary(
        mut self,
        built: Vec<(String, SourceProgram)>,
        params: Vec<f64>,
    ) -> (Self, u16) {
        let id = self.boundary_dags.len() as u16;
        self.boundary_dags
            .push(RuntimeSource::new(built, params, true));
        (self, id)
    }

    /// register a neumann / robin gradient boundary (the convenience short-circuit). the id
    /// (registration order) is what the sim's `Boundaries` carries as `BoundaryType::Neumann(id)` /
    /// `Robin(id)`. `coeffs` are the per-variable coefficients in prim order `[rho, vel.., pre]`.
    pub fn with_gradient_boundary(mut self, coeffs: GradientBc) -> (Self, u16) {
        let id = self.gradient_bcs.len() as u16;
        self.gradient_bcs.push(coeffs);
        (self, id)
    }

    /// attach a runtime user source from already-lowered `(target, SourceProgram)`
    /// pairs (the `build_user_source` output of a `kind="raw"` RHD `SourceConfig`).
    /// applied two-pass in `source_apply` (plain godunov + `dispatch_runtime_source`).
    pub fn with_runtime_source(
        mut self,
        built: Vec<(String, SourceProgram)>,
        params: Vec<f64>,
    ) -> Self {
        // has_energy = true (RHD carries tau); validation happened at
        // `build_user_source(cfg, &RHD_SPEC)`.
        self.runtime_source = Some(RuntimeSource::new(built, params, true));
        self
    }

    /// attach a runtime user source and fuse it into the godunov stage (flat host+f64); the two-pass
    /// twin `with_runtime_source` forces the separate pass. bit-identical (falls back off-host / non-f64
    /// / GR). no immersed-body fold — RHD has no Newtonian body.
    pub fn with_fused_runtime_source(
        mut self,
        built: Vec<(String, SourceProgram)>,
        params: Vec<f64>,
    ) -> Self {
        self.runtime_source = Some(RuntimeSource::new(built, params, true));
        self.fuse_runtime = true;
        self
    }

    /// pick the Riemann solver. default is HLLE; HLLC routes to the
    /// `rhd_face_flux_hllc_*` AOT variants. fluent builder.
    /// rejects a solver that is invalid for the RHD regime (e.g., HLLD).
    pub fn with_solver(mut self, solver: Solver) -> Result<Self, symbi_sim::state::ConfigError> {
        let regime =
            crate::regimes::substrate_kernels::RegimeKind::of::<Sc, D, symbi_hydro::rhd::Rhd>();
        if !solver.valid_for(regime) {
            return Err(symbi_sim::state::ConfigError::SolverRegimeMismatch { solver, regime });
        }
        self.solver = solver;
        Ok(self)
    }

    /// set the theta-MC limiter compression in [1,2] (1 = minmod, 2 = monotonized-central). fluent.
    /// set the reference mach number the published low-mach ramp saturates at. accepted for
    /// builder uniformity; the relativistic LM arm is the clamped one, so no relativistic
    /// kernel reads it. fluent.
    /// accepted for builder-chain uniformity and refused when set: the well-balanced
    /// reconstruction limits departures from the newtonian polytropic isentrope, and no
    /// relativistic kernel carries it. silently ignoring the request would run a plain
    /// reconstruction under a config that asked for a balanced one.
    pub fn well_balanced_reconstruction(self, on: bool) -> Self {
        assert!(
            !on,
            "well-balanced reconstruction is newtonian-only: the balanced kernels limit \
             departures from the polytropic isentrope, which no relativistic flux carries"
        );
        self
    }

    pub fn theta(mut self, theta: f64) -> Self {
        self.theta = theta;
        self
    }

    /// evolution reconstruction selector. the rhd flux family is plm-only — no `_ppm`
    /// kernel twins are baked for it — so this validates rather than stores; a shared
    /// regime-generic build path may call it unconditionally.
    pub fn reconstruction(self, recon: symbi_discretize::Recon) -> Self {
        assert!(
            recon == symbi_discretize::Recon::Plm,
            "ppm reconstruction is baked for the newtonian adiabatic flux family only"
        );
        self
    }
}

impl<Mem: MemorySpace + Sync, Sc: Scalar + OrderedNumeric, const D: usize>
    RhdSubstrateKernelSet<Mem, Sc, D>
{
    /// the evolution face-flux scheme assembled from this set's dials. the rhd
    /// family is plm-only, plain-balanced, and carries its eos closure arm.
    fn flux_spec(&self) -> FluxSpec {
        FluxSpec {
            theta: self.theta,
            solver: self.solver,
            recon: symbi_discretize::Recon::Plm,
            eos: self.eos,
            flatten: (0.0, 0.0),
            balance: symbi_discretize::coords::Balance::Plain,
            rusanov: false,
        }
    }
}

impl<Mem: MemorySpace + Sync, Sc: Scalar + OrderedNumeric, const D: usize, const DOF: usize>
    KernelSet<D, DOF, Mem, Sc> for RhdSubstrateKernelSet<Mem, Sc, D>
{
    fn flux(&self, sim: &FieldStore<D, DOF, Mem, Sc>, dir: usize) {
        let pre = sim.fields.prim.pre_field().expect("Rhd requires prim.pre");
        dispatch_flux(sim, pre, "rhd", dir, self.gamma, self.flux_spec());
    }

    fn c2p(&self, sim: &FieldStore<D, DOF, Mem, Sc>) {
        // the primitives now hold a state recovered from the conserved fields; anything
        // reading prim.* outside the evolve loop checks this before trusting it.
        sim.mark_primitives_recovered();
        let cnrg = sim.fields.cons.nrg_field().expect("Rhd requires cons.nrg");
        let pre = sim.fields.prim.pre_field().expect("Rhd requires prim.pre");

        // inputs: cons den, mom_0..mom_{DOF-1}, nrg. outputs: prim rho, vel_0.., pre.
        let mut inputs: Vec<&Field<Sc, D, Mem>> = vec![&sim.fields.cons.den];
        for k in 0..DOF {
            inputs.push(&sim.fields.cons.mom[k]);
        }
        inputs.push(cnrg);
        let mut outputs: Vec<&Field<Sc, D, Mem>> = vec![&sim.fields.prim.rho];
        for k in 0..DOF {
            outputs.push(&sim.fields.prim.vel[k]);
        }
        outputs.push(pre);
        // the c2p status channel: the recovery kernel writes its accept/reject
        // fact alongside the candidate, so the channel has one producer.
        outputs.push(&sim.fields.c2p_error);

        // the GR path uses the metric-aware Valencia recovery (`|S|^2 = gamma^{ij} S_i S_j`,
        // contravariant `v^i`); its name carries the spacetime slug and it reads the lapse
        // mass M + the log-aware radial grid scalars (the metric is evaluated at the cell centroid).
        // flat keeps the plain `rhd_c2p_{D}d` (gamma only), bit-identical. the DOF-lift tag
        // (spherical swirl) selects the instance whose manifest carries the extra momentum.
        // the chart/DOF tag: the spherical swirl (DOF != D) rides geom_suffix; the cartesian GR
        // chart rides `_cart` (distinct from the implicit spherical GR default); flat + spherical GR
        // stay untagged here.
        let geom_sfx = gr_chart_dof_tag(sim.geom.coords, sim.geom.spacetime, DOF, D);
        let st_sfx = spacetime_slug(sim.geom.spacetime);
        if self.eos == symbi_discretize::EosArm::TaubMathews {
            assert!(
                geom_sfx.is_empty() && st_sfx.is_empty(),
                "the taub-mathews c2p is baked for the flat rhd family \
                 (minkowski, ncomp == ndim) only"
            );
        }
        let (name, scalars) = if st_sfx.is_empty() {
            let name = format!("rhd_c2p{}{geom_sfx}_{D}d", self.eos.suffix());
            let scalars = scalars_for(&name, |bind| match bind {
                ScalarBind::Ref(ScalarRef::Gamma) => Sc::from_f64(self.gamma),
                o => panic!("rhd c2p: unexpected scalar {o:?}"),
            });
            (name, scalars)
        } else {
            let name = format!("rhd_c2p{geom_sfx}{st_sfx}_{D}d");
            let (x_lo, dx) = kernel_geom(
                &sim.geom.x_lo,
                &sim.geom.dx,
                &sim.geom.maps,
                sim.geom.coords,
                sim.motion.a,
            );
            let scalars = scalars_for(&name, |bind| {
                let ScalarBind::Ref(sref) = bind else {
                    panic!("rhd GR c2p: unexpected spec scalar {bind:?}");
                };
                match *sref {
                    ScalarRef::Gamma => Sc::from_f64(self.gamma),
                    ScalarRef::SchwarzschildMass => Sc::from_f64(
                        sim.geom
                            .spacetime_scalars
                            .iter()
                            .find(|(n, _)| n == "schwarzschild_mass")
                            .map(|(_, v)| *v)
                            .expect(
                                "rhd GR c2p needs schwarzschild_mass but the metric supplied none",
                            ),
                    ),
                    ScalarRef::KerrSpin => Sc::from_f64(
                        sim.geom
                            .spacetime_scalars
                            .iter()
                            .find(|(n, _)| n == "kerr_spin")
                            .map(|(_, v)| *v)
                            .expect("rhd GR c2p needs kerr_spin but the metric supplied none"),
                    ),
                    other => Sc::from_f64(
                        geom_scalar(&x_lo, &dx, &sim.geom.maps, other)
                            .unwrap_or_else(|| panic!("rhd GR c2p: unexpected scalar {other:?}")),
                    ),
                }
            });
            (name, scalars)
        };
        dispatch_fields::<Sc, Mem, D>(
            &name,
            &sim.geom.allocated,
            &sim.geom.interior,
            &inputs,
            &outputs,
            &[],
            &scalars,
        );
    }

    fn godunov_stage(&self, sim: &FieldStore<D, DOF, Mem, Sc>, dt: f64, a0: f64, ac: f64) {
        let pre = sim.fields.prim.pre_field().expect("Rhd requires prim.pre");
        // fuse the raw user source into godunov on a flat background, host+f64. the fused builder
        // traces the flat geo (GeoSource::Hydro, matching the `rhd` AOT godunov's geo_source), so it
        // matches only the Minkowski kernel; GR (a spacetime suffix) keeps the two-pass. fold_body =
        // false — RHD has no Newtonian immersed body.
        if self.fuse_runtime && matches!(sim.geom.spacetime, Spacetime::Minkowski) {
            if let Some(rs) = &self.runtime_source {
                if let Some(fk) =
                    fused_runtime_cpu_kernel(sim, rs, GeoSource::Hydro { inertial: true }, false)
                {
                    dispatch_fused_runtime_cpu(sim, pre, fk, Some(rs), dt, a0, ac, self.gamma);
                    return;
                }
            }
        }
        dispatch_godunov(sim, pre, "rhd", dt, a0, ac, self.gamma);
    }

    fn has_additive_source(&self) -> bool {
        self.runtime_source.is_some()
    }

    fn source_apply(&self, sim: &FieldStore<D, DOF, Mem, Sc>, weight: f64) {
        // two-pass: plain godunov already ran; add the raw user source increment to
        // the conserved fields. dispatch_runtime_source is regime-agnostic (it adds
        // the SourceProgram outputs to their target conserved slots), so no RHD-specific
        // source code — the relativistic conservation law lives in the godunov stage.
        if let Some(rs) = &self.runtime_source {
            // when the fused stage rode the source inside godunov (same predicate godunov_stage used),
            // the separate pass would double-count — skip it.
            if self.fuse_runtime
                && matches!(sim.geom.spacetime, Spacetime::Minkowski)
                && fused_runtime_cpu_kernel(sim, rs, GeoSource::Hydro { inertial: true }, false)
                    .is_some()
            {
                return;
            }
            dispatch_runtime_source(sim, rs, weight);
        }
    }

    // horizon excision as the sweep/finalize pieces: the trait-default excise()
    // composes them for the monolithic loop; the decomposed loop drives the
    // sweeps itself with a halo exchange between them. inert at zero radius;
    // the dispatch asserts the baked combination (2d/3d cartesian kerr-schild
    // charts) fail-loud.
    fn excise_pass_count(&self, sim: &FieldStore<D, DOF, Mem, Sc>) -> usize {
        crate::regimes::substrate_kernels::excise_pass_count_for(sim, self.excision_radius)
    }

    fn excise_sweep(&self, sim: &FieldStore<D, DOF, Mem, Sc>) {
        if self.excision_radius > 0.0 {
            crate::regimes::substrate_kernels::dispatch_excise_sweep(
                sim,
                self.gamma,
                self.excision_radius,
                self.excision_rho,
                self.excision_pre,
            );
        }
    }

    fn excise_finalize(&self, sim: &FieldStore<D, DOF, Mem, Sc>) {
        if self.excision_radius > 0.0 {
            crate::regimes::substrate_kernels::dispatch_excise_finalize(
                sim,
                self.gamma,
                self.excision_radius,
                self.excision_rho,
                self.excision_pre,
            );
        }
    }

    fn cfl(&self, sim: &FieldStore<D, DOF, Mem, Sc>) -> f64 {
        // the shared cfl dispatch binds the field buffers by manifest + owns the reduction.
        // RHD's only contribution is the "rhd" map (its relativistic wave speed). on a curved
        // background (DOF == D, where FOFC is active) the source-admissibility rate lambda_S folds in
        // after the map: the geodesic source must not push U + dt S out of the physical cone.
        let pre = sim.fields.prim.pre_field().expect("Rhd requires prim.pre");
        // the wu 2017 source-admissibility rate lambda_S is now subsumed by the admissible-boundary
        // projection: the projection guarantees U in G post-hoc for any dt, so lambda_S no longer has
        // to shrink the step to keep U + dt S admissible. dropping it lets the near-horizon cusp cell
        // (near-vacuum ultrarelativistic infall, where lambda_S -> inf and collapsed the step) floor
        // via the projection instead of dictating the global dt — the HARM discipline. dt is then set
        // by the flux light-cone; a resolved cell's geodesic source is well within it, an unresolvable
        // cusp cell is floored. flat SRHD never had a source rate.
        let source_cfl: Option<String> = None;
        cfl_wave_speed(
            sim,
            pre,
            &self.cfl_scratch,
            "rhd",
            self.eos,
            self.gamma,
            self.cfl_number,
            source_cfl.as_deref(),
            self.excision_radius,
        )
    }

    fn horizon_accretion(
        &self,
        sim: &FieldStore<D, DOF, Mem, Sc>,
        diagnostic_radius: f64,
    ) -> (f64, f64) {
        // the shell-flux kernels are baked for the cartesian kerr-schild chart (the only chart that
        // excises). reuse the (already-consumed) cfl scratch for the per-quantity Add-reduction.
        if sim.geom.spacetime != Spacetime::SchwarzschildKS
            || sim.geom.coords != symbi_geometry::Geometry::Cartesian
        {
            return (0.0, 0.0);
        }
        let pre = sim.fields.prim.pre_field().expect("Rhd requires prim.pre");
        shell_accretion_rates(sim, pre, &self.cfl_scratch, diagnostic_radius)
    }

    fn ghost_fill(&self, sim: &FieldStore<D, DOF, Mem, Sc>) {
        // the shared lattice-map pullback (iso_ghost_fill_{D}d): the EOS-generic
        // prim pullback (rho/vel_0..DOF-1/pre), in-place, per ghost region. the DOF-lift
        // tag (spherical swirl) selects the instance carrying the extra velocity. the
        // spinning-kerr instance copies the azimuthal ghost through the angular-momentum
        // variable w = v^phi + (gamma_{r phi}/gamma_{phi phi}) v^r (dragging-consistent at
        // the ghost's own radius), so it reads the metric scalars + the radial grid map.
        let bc = to_bc_array::<D>(&sim.boundaries);
        let pre = sim.fields.prim.pre_field().expect("Rhd requires prim.pre");
        let geom_sfx = dof_lift_suffix(sim.geom.coords, DOF, D);
        // the dragging-consistent w-copy is a spherical-azimuth construct (gamma_{r phi} on the
        // swirl DOF); the cartesian kerr chart has DOF == D and copies the raw prims like any
        // other background.
        let is_kerr = matches!(sim.geom.spacetime, symbi_geometry::Spacetime::KerrKS)
            && sim.geom.coords == symbi_geometry::Geometry::Spherical;
        let name = if is_kerr {
            format!("rhd_ghost_fill{geom_sfx}_kerr_{D}d")
        } else {
            format!("iso_ghost_fill{geom_sfx}_{D}d")
        };
        let (x_lo, dx) = kernel_geom(
            &sim.geom.x_lo,
            &sim.geom.dx,
            &sim.geom.maps,
            sim.geom.coords,
            sim.motion.a,
        );

        GhostFillDriver::<D>::new(&sim.geom.allocated, &sim.geom.interior, bc).drive_sweep(
            |region, p| {
                let mut outputs: Vec<&Field<Sc, D, Mem>> = vec![&sim.fields.prim.rho];
                for k in 0..DOF {
                    outputs.push(&sim.fields.prim.vel[k]);
                }
                outputs.push(pre);
                // ints: map_type_0..{D-1}, arg_0..{D-1}. scalars: vel_sign_0..{D-1} (+ the
                // metric mass/spin and the log-aware grid scalars on the kerr instance).
                // params by name via the type-sorted manifest: map_type/arg are INT lanes, vel_sign
                // FLOAT — each routed to its ABI tail by the kernel's declared sort (the int \sqcup float
                // coproduct).
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
                        ScalarBind::Ref(ScalarRef::SchwarzschildMass) => Sc::from_f64(
                            sim.geom
                                .spacetime_scalars
                                .iter()
                                .find(|(n, _)| n == "schwarzschild_mass")
                                .map(|(_, v)| *v)
                                .expect("kerr ghost fill needs schwarzschild_mass"),
                        ),
                        ScalarBind::Ref(ScalarRef::KerrSpin) => Sc::from_f64(
                            sim.geom
                                .spacetime_scalars
                                .iter()
                                .find(|(n, _)| n == "kerr_spin")
                                .map(|(_, v)| *v)
                                .expect("kerr ghost fill needs kerr_spin"),
                        ),
                        ScalarBind::Ref(other) => Sc::from_f64(
                            geom_scalar(&x_lo, &dx, &sim.geom.maps, *other).unwrap_or_else(|| {
                                panic!("ghost_fill: unexpected scalar {other:?}")
                            }),
                        ),
                        o => panic!("ghost_fill: unexpected scalar {o:?}"),
                    },
                );
                dispatch_fields::<Sc, Mem, D>(
                    &name,
                    &sim.geom.allocated,
                    &region.domain,
                    &[],
                    &outputs,
                    &ints,
                    &scalars,
                );
            },
        );

        // the standard pullback skips any Driven faces (Driven ->
        // BcType::Skip); prescribe their ghost prim state from the registered boundary DAGs.
        if !self.boundary_dags.is_empty() {
            dispatch_driven_boundaries(sim, &self.boundary_dags);
        }
        // the Neumann/Robin gradient faces (also skipped by the pullback), filled from the edge cell.
        if !self.gradient_bcs.is_empty() {
            dispatch_gradient_boundaries(sim, pre, &self.gradient_bcs, None);
        }
    }

    fn snapshot(&self, sim: &FieldStore<D, DOF, Mem, Sc>) {
        let cnrg = sim.fields.cons.nrg_field().expect("cons.nrg");
        let unrg = sim.workspace.u_n.nrg_field().expect("u_n.nrg");

        // inputs: cons den, mom_0.., nrg. outputs: u_n den, mom_0.., nrg. the DOF-lift tag
        // (spherical swirl) selects the instance copying the extra momentum.
        let mut inputs: Vec<&Field<Sc, D, Mem>> = vec![&sim.fields.cons.den];
        for k in 0..DOF {
            inputs.push(&sim.fields.cons.mom[k]);
        }
        inputs.push(cnrg);
        let mut outputs: Vec<&Field<Sc, D, Mem>> = vec![&sim.workspace.u_n.den];
        for k in 0..DOF {
            outputs.push(&sim.workspace.u_n.mom[k]);
        }
        outputs.push(unrg);

        let geom_sfx = dof_lift_suffix(sim.geom.coords, DOF, D);
        let name = format!("rhd_snapshot{geom_sfx}_{D}d");
        dispatch_fields::<Sc, Mem, D>(
            &name,
            &sim.geom.allocated,
            &sim.geom.allocated,
            &inputs,
            &outputs,
            &[],
            &[],
        );
    }

    fn snapshot_stage(&self, sim: &FieldStore<D, DOF, Mem, Sc>) {
        // u_stage = cons (den, mom_0.., nrg), the pre-godunov state FOFC restores to reconstruct
        // the first-order redo from. mirrors `snapshot` (which targets u_n).
        let cnrg = sim.fields.cons.nrg_field().expect("cons.nrg");
        let unrg = sim.workspace.u_stage.nrg_field().expect("u_stage.nrg");
        let mut inputs: Vec<&Field<Sc, D, Mem>> = vec![&sim.fields.cons.den];
        for k in 0..DOF {
            inputs.push(&sim.fields.cons.mom[k]);
        }
        inputs.push(cnrg);
        let mut outputs: Vec<&Field<Sc, D, Mem>> = vec![&sim.workspace.u_stage.den];
        for k in 0..DOF {
            outputs.push(&sim.workspace.u_stage.mom[k]);
        }
        outputs.push(unrg);
        let geom_sfx = dof_lift_suffix(sim.geom.coords, DOF, D);
        let name = format!("rhd_snapshot{geom_sfx}_{D}d");
        dispatch_fields::<Sc, Mem, D>(
            &name,
            &sim.geom.allocated,
            &sim.geom.allocated,
            &inputs,
            &outputs,
            &[],
            &[],
        );
    }

    fn fofc_active(&self) -> bool {
        true
    }

    fn fofc(
        &self,
        sim: &FieldStore<D, DOF, Mem, Sc>,
        dt: f64,
        a0: f64,
        ac: f64,
        _stage: u8,
    ) -> bool {
        // the first-order redo runs at theta = 0 (PCM). flat (Minkowski) SRHD uses HLLE — the
        // positivity-preserving Einfeldt fan — regardless of the production solver (HLLC can
        // undershoot in a strong rarefaction). the curved (GR) background uses the light-cone rusanov
        // fan: the state-dependent HLLE speeds can under-bound near the physical-set boundary on a
        // curved metric, whereas alpha sqrt(gamma^{nn}) bounds every fluid characteristic. the GR
        // source-admissibility timestep (the lambda_S folded into the CFL) keeps U + dt S in the cone.
        // the spherical-swirl DOF-lift (DOF > D) selects the ncomp = DOF fofc/rusanov kernels via the
        // geom tag; the ghost-band finiteness halt provides the FOFC-surviving fail-loud.
        let pre = sim.fields.prim.pre_field().expect("Rhd requires prim.pre");
        let curved = sim.geom.spacetime != symbi_geometry::Spacetime::Minkowski;
        let dof_sfx = dof_lift_suffix(sim.geom.coords, DOF, D);
        crate::regimes::fofc::fofc_orchestrate(
            sim,
            "rhd",
            dof_sfx,
            <Self as KernelSet<D, DOF, Mem, Sc>>::has_additive_source(self),
            &self.cfl_scratch,
            pre,
            &self.freeze_streak,
            |dir| {
                // the redo keeps the run's own eos closure (`first_order` carries it);
                // a curved background swaps the fallback to the rusanov form.
                let mut redo = self.flux_spec().first_order();
                redo.rusanov = curved;
                dispatch_flux(sim, pre, "rhd", dir, self.gamma, redo)
            },
            || self.c2p(sim),
            || self.godunov_stage(sim, dt, a0, ac),
            || self.source_apply(sim, ac * dt),
            || {}, // rhd has no immersed-body source (trait-default no-op)
            || {
                // admissible-boundary projection: on a curved background, project every spliced cell
                // onto partial-G along the segment to the admissible stage input, replacing the freeze
                // parachute with a provable map into the physical set. flat SRHD keeps the freeze.
                if curved {
                    crate::regimes::substrate_kernels::fofc_project(
                        sim,
                        "rhd",
                        // hydro keys its chart on the momentum-DOF lift: the 2-axis spherical
                        // wedge carrying azimuthal momentum is a different kernel from the plain
                        // 2D spherical one.
                        symbi_discretize::kernel_slug::ChartKeying::MomentumDof,
                        self.gamma,
                        sim.stage_input(),
                        &sim.fields.cons,
                        &sim.fields.prim,
                        None, // hydro: the admissible cone carries no magnetic term
                    );
                }
            },
            None, // no body-evolved freeze parachute (no rhd body source)
            crate::regimes::fofc::CtHooks::none(),
            || crate::regimes::fofc::SourceReplay::NotApplicable, // hydro: no source replay
            false, // no projection tier below the freeze; keep the parachute
        )
    }
}
