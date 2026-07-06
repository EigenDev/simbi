// =============================================================================
// regimes/substrate_newton.rs
//
// AdiabaticSubstrateKernelSet<Mem, Sc, const D> — the D-GENERIC adiabatic (ideal-gas
// Newtonian Euler) KernelSet, every method dispatched to a build-time AOT substrate
// kernel through the structured binding ABI, the instance resolved by name (regime,
// ndim, dir) via the generated `kernel_by_name` registry. one struct serves 1D/2D/3D
// (the kepler/blast gap).
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
use symbi_ir::algebra::Scalar;
use symbi_ir::ScalarRef;
use symbi_grid::Field;
use symbi_hydro::source_spec::BuiltSource;
use symbi_xpu::MemorySpace;

use std::sync::Arc;

use crate::kernels::support::{to_bc_array, GhostFillDriver};
use crate::regimes::substrate_kernels::{
    cfl_wave_speed, dispatch_body_feedback, dispatch_body_source, dispatch_fields, dispatch_flux,
    dispatch_driven_boundaries, dispatch_fused_runtime_cpu, dispatch_godunov_maybe_fused,
    dispatch_named, dispatch_runtime_source, dispatch_source_apply, fused_runtime_cpu_kernel,
    geom_suffix, resolve_params, scalars_for, FusedSourceBinding, RuntimeSource, ScalarBind, Solver,
};
use symbi_discretize::gv::GeoSource;
use symbi_sim::substrate_seam::KernelSet;
use symbi_sim::state::FieldStore;

/// a D-generic adiabatic (ideal-gas Euler) `KernelSet`, every method substrate-generated.
pub struct AdiabaticSubstrateKernelSet<Mem: MemorySpace, Sc: Scalar + OrderedNumeric, const D: usize> {
    pub gamma: f64,
    pub cfl_number: f64,
    /// the theta-MC reconstruction compression (regime-generic; 1 == plain minmod).
    pub theta: f64,
    pub cfl_scratch: Field<Sc, D, Mem>,
    /// **B6-iv Phase 4b**: declarative routing to the AOT-baked fused godunov
    /// (`adiabatic_godunov_euler_with_{source_id}_{D}d`). `None` => the unfused
    /// kernel (the original default; bit-identical for existing callers). `Some`
    /// => `godunov_euler` / `godunov_rk2` route through the fused variant.
    pub fused_source: Option<FusedSourceBinding>,
    /// **S3b**: the NON-fused (additive) source overlay — plain godunov + a
    /// per-stage `adiabatic_source_with_{slug}_{D}d` pass. proven bit-for-bit
    /// equal to `fused_source` with the same binding. see the iso analogue.
    pub additive_source: Option<FusedSourceBinding>,
    /// **Gap B**: a RUNTIME-loaded user source (python -> json -> `build_user_source` ->
    /// `SourceEvaluator::from_built`). `Some` => `source_apply` runs a per-cell CPU pass that
    /// interprets the user `BuiltSource`(s) and adds `weight*S` to cons — NO recompile, NO
    /// AOT-baked kernel. CPU-only (host memory); the gpu path is runtime NVRTC (future).
    pub runtime_source: Option<Arc<RuntimeSource>>,
    /// **v2 inc 3+4**: when true AND a `runtime_source` is attached, the runtime user source is
    /// FUSED into the godunov stage as ONE Cranelift-JIT'd host kernel (cooling -> RT MZCS) instead
    /// of the two-pass (plain godunov + per-cell `apply_runtime_source`). opt-in + gated: falls back
    /// to the two-pass on non-f64 carriers, device memory, or an out-of-JIT-subset source. the
    /// two-pass remains the default. proven bit-for-bit by `jit_fused_equals_two_pass`.
    pub fuse_runtime: bool,
    /// **docs/design/33** — driven-boundary DAGs, indexed by the `BoundaryType::Driven(id)` id the
    /// sim's `Boundaries` carries. each prescribes a face's ghost prim state. empty => no driven
    /// faces (the standard ghost-fill is the whole story). reuses `RuntimeSource` as the holder (same
    /// data); the `(Coord, Assign)` dispatch is what makes it a boundary.
    pub boundary_dags: Vec<Arc<RuntimeSource>>,
    /// Riemann solver — HLLE (default, two-wave) or HLLC (contact-resolving).
    /// see [[Solver]]. tunable via `.with_solver(Solver::Hllc)`.
    pub solver: Solver,
}

impl<Mem: MemorySpace, Sc: Scalar + OrderedNumeric, const D: usize> AdiabaticSubstrateKernelSet<Mem, Sc, D> {
    pub fn new(gamma: f64, cfl_number: f64, alloc_domain: &Domain<D>) -> Self {
        let cfl_scratch = Field::<Sc, D, Mem>::zeros(alloc_domain)
            .expect("failed to allocate adiabatic CFL scratch field");
        // theta defaults to 1.0 (plain minmod) — exact prior behavior; set `theta` to tune.
        Self { gamma, cfl_number, theta: 1.0, cfl_scratch, fused_source: None, additive_source: None, runtime_source: None, fuse_runtime: false, boundary_dags: Vec::new(), solver: Solver::Hlle }
    }

    /// **Gap B**: attach a RUNTIME-loaded user source from already-lowered `(target, BuiltSource)`
    /// pairs (the `build_user_source` output of a python -> json `SourceConfig`). ONE source of
    /// truth: the substrate derives BOTH the CPU per-cell interpreter (`SourceEvaluator::from_built`)
    /// AND — lazily, on first device dispatch — the GPU IR (the SAME `source_apply_from_built_gv`
    /// builder build.rs AOT-bakes), so the host runs the interpreter and the device runs an
    /// NVRTC-JIT kernel, both from the same DAG, no recompile. `params` are the DAG's `p{i}` knobs.
    /// `let built = build_user_source(&cfg, has_energy)?;`
    /// `let sub = sim.substrate().with_runtime_source(built, cfg.params.clone());`
    pub fn with_runtime_source(mut self, built: Vec<(String, BuiltSource)>, params: Vec<f64>) -> Self {
        // has_energy = true is the AUTHORITY here (Newtonian's RegimeSpec), not caller-supplied —
        // the set IS the regime. validation of the source vs the regime happened at
        // `build_user_source(cfg, &NEWTONIAN_SPEC)`.
        self.runtime_source = Some(RuntimeSource::new(built, params, true));
        self
    }

    /// **v2 inc 3+4**: attach a runtime user source AND route it through the FUSED host path — the
    /// source rides INSIDE the Cranelift-JIT'd godunov stage (one launch), not as a separate pass.
    /// same source, faster execution; bit-for-bit identical to `with_runtime_source` (the two-pass),
    /// proven by `jit_fused_equals_two_pass`. host + f64 only; otherwise it transparently falls back
    /// to the two-pass.
    pub fn with_fused_runtime_source(mut self, built: Vec<(String, BuiltSource)>, params: Vec<f64>) -> Self {
        self.runtime_source = Some(RuntimeSource::new(built, params, true));
        self.fuse_runtime = true;
        self
    }

    /// **docs/design/33**: register a DRIVEN boundary. the returned id (registration order, 0-based)
    /// is what the sim's `Boundaries` must carry as `BoundaryType::Driven(id)` on the prescribed
    /// face. build `built` from `expr_bridge::build_boundary_dag(&cfg, &NEWTONIAN_SPEC)` (a complete
    /// prim prescription `[rho, vel.., pre]`). after the standard ghost-fill skips the driven faces,
    /// `ghost_fill` prescribes their ghost state from this DAG.
    /// `let sub = sim.substrate().with_driven_boundary(built, cfg.params.clone()).0;`
    pub fn with_driven_boundary(mut self, built: Vec<(String, BuiltSource)>, params: Vec<f64>) -> (Self, u16) {
        let id = self.boundary_dags.len() as u16;
        // has_energy = true (Newtonian) — the boundary prescribes prim.pre as well as rho/vel.
        self.boundary_dags.push(RuntimeSource::new(built, params, true));
        (self, id)
    }

    /// **B6-iv Phase 4b**: bind a fused-source AOT kernel for this kernel-set.
    /// fluent builder; chain off `new(..)`:
    /// `AdiabaticSubstrateKernelSet::new(gamma, cfl, alloc).with_fused_source(b)`.
    pub fn with_fused_source(mut self, binding: FusedSourceBinding) -> Self {
        self.fused_source = Some(binding);
        self
    }

    /// **S3b**: bind the SAME source as a NON-fused additive pass instead of
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
        let regime = crate::regimes::substrate_kernels::RegimeKind::of::<Sc, D, symbi_hydro::newtonian::Newtonian>();
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
}

impl<Mem: MemorySpace + Sync, Sc: Scalar + OrderedNumeric, const D: usize, const DOF: usize>
    KernelSet<D, DOF, Mem, Sc>
    for AdiabaticSubstrateKernelSet<Mem, Sc, D>
{
    fn flux(&self, sim: &FieldStore<D, DOF, Mem, Sc>, dir: usize) {
        let pre = sim.fields.prim.pre_field().expect("Newtonian requires prim.pre");
        dispatch_flux(sim, pre, "adiabatic", dir, self.gamma, self.theta, self.solver, false);
    }

    fn c2p(&self, sim: &FieldStore<D, DOF, Mem, Sc>) {
        // cons -> prim, all DOF velocities; the manifest binds cons.* -> prim.* (the cyl
        // c2p writes prim.vel_0..2, automatic). "prim.pre" resolves to the prim pressure.
        let pre = sim.fields.prim.pre_field().expect("Newtonian requires prim.pre");
        let sfx = if DOF != D { geom_suffix(sim.geom.coords, DOF, D) } else { "" };
        let name = format!("adiabatic_c2p{sfx}_{D}d");
        let scalars = scalars_for(&name, |bind| match bind {
            ScalarBind::Ref(ScalarRef::Gamma) => Sc::from_f64(self.gamma),
            o => panic!("adiabatic c2p: unexpected scalar {o:?}"),
        });
        dispatch_named(sim, pre, None, 0, &name, &sim.geom.interior, &[], &scalars);
    }

    fn godunov_stage(&self, sim: &FieldStore<D, DOF, Mem, Sc>, dt: f64, a0: f64, ac: f64) {
        let pre = sim.fields.prim.pre_field().expect("Newtonian requires prim.pre");
        // the FUSED runtime-source path: one Cranelift-JIT'd godunov+source launch, replacing
        // (plain godunov + the separate `apply_runtime_source` pass). gated host+f64; the source's
        // own pass is skipped in `source_apply` under the SAME predicate. geo = Hydro{inertial:true}
        // matches the AOT `adiabatic` godunov (geo_source), so the two-pass fallback is exact.
        if self.fuse_runtime {
            if let Some(rs) = &self.runtime_source {
                if let Some(fk) = fused_runtime_cpu_kernel(sim, rs, GeoSource::Hydro { inertial: true }) {
                    dispatch_fused_runtime_cpu(sim, pre, fk, rs, dt, a0, ac);
                    return;
                }
            }
        }
        dispatch_godunov_maybe_fused(sim, pre, "adiabatic", dt, a0, ac, self.fused_source.as_ref());
    }

    fn cfl(&self, sim: &FieldStore<D, DOF, Mem, Sc>) -> f64 {
        // adiabatic shares the iso wave-speed map (cs = sqrt(gamma*p/rho), gamma=1.4 vs 1);
        // the SHARED cfl dispatch binds the field buffers by manifest + owns the reduction.
        let pre = sim.fields.prim.pre_field().expect("Newtonian requires prim.pre");
        cfl_wave_speed(sim, pre, &self.cfl_scratch, "iso", self.gamma, self.cfl_number, None)
    }

    fn ghost_fill(&self, sim: &FieldStore<D, DOF, Mem, Sc>) {
        // the SHARED lattice-map pullback (iso_ghost_fill{sfx}_{D}d): rho/vel/pre, in-place,
        // bound by manifest. (DOF>NDIM: the cyl ghost manifest is the pending axis-BC work —
        // docs/design/18 D3 — but the dispatch path is regime-uniform.)
        let bc = to_bc_array::<D>(&sim.boundaries);
        let pre = sim.fields.prim.pre_field().expect("Newtonian requires prim.pre");
        let sfx = if DOF != D { geom_suffix(sim.geom.coords, DOF, D) } else { "" };
        let name = format!("iso_ghost_fill{sfx}_{D}d");

        GhostFillDriver::<D>::new(&sim.geom.allocated, &sim.geom.interior, bc).drive_sweep(|region, p| {
            // params BY NAME via the type-sorted manifest: map_type/arg are INT lanes (the
            // `ints` tail), vel_sign FLOAT (the `scalars` tail) — each routed by the kernel's
            // declared sort. the int ⊔ float coproduct that defeated positional scalars_for.
            let (ints, scalars) = resolve_params(
                &name,
                |bind| match bind {
                    ScalarBind::Ref(ScalarRef::MapType(ax)) => p.map_type[*ax as usize] as i32,
                    ScalarBind::Ref(ScalarRef::Arg(ax)) => p.arg[*ax as usize],
                    o => panic!("ghost_fill: unexpected int param {o:?}"),
                },
                |bind| match bind {
                    ScalarBind::Ref(ScalarRef::VelSign(ax)) => Sc::from_f64(p.vel_sign[*ax as usize]),
                    o => panic!("ghost_fill: unexpected scalar {o:?}"),
                },
            );
            dispatch_named(sim, pre, None, 0, &name, &region.domain, &ints, &scalars);
        });

        // docs/design/33: the standard pullback above SKIPPED any Driven faces (Driven -> BcType::Skip);
        // now prescribe their ghost prim state from the registered boundary DAGs.
        if !self.boundary_dags.is_empty() {
            dispatch_driven_boundaries(sim, &self.boundary_dags);
        }
    }

    fn snapshot(&self, sim: &FieldStore<D, DOF, Mem, Sc>) {
        // u_n = cons (all DOF components), bound by manifest over the full allocated domain.
        let pre = sim.fields.prim.pre_field().expect("Newtonian requires prim.pre");
        let sfx = if DOF != D { geom_suffix(sim.geom.coords, DOF, D) } else { "" };
        let name = format!("adiabatic_snapshot{sfx}_{D}d");
        dispatch_named(sim, pre, None, 0, &name, &sim.geom.allocated, &[], &[]);
    }

    fn has_additive_source(&self) -> bool {
        self.additive_source.is_some() || self.runtime_source.is_some()
    }

    fn fofc_active(&self) -> bool {
        // the first-order flux correction covers the DOF == D charts (the fofc select's momentum
        // count is baked to ncomp = D); the spherical-swirl DOF-lift is a follow-on.
        DOF == D
    }

    fn fofc(&self, sim: &FieldStore<D, DOF, Mem, Sc>, dt: f64, a0: f64, ac: f64) {
        // FOFC covers the DOF == D charts only (the fofc kernels are baked at ncomp = D); the
        // spherical-swirl DOF-lift is a follow-on. `fofc` is called unconditionally by the driver, so
        // the gate lives here (fofc_active only guards the stage-input snapshot).
        if DOF != D {
            return;
        }
        // the first-order redo runs HLLE at theta = 0 (PCM) — the positivity-preserving Einfeldt
        // fan — regardless of the production solver (HLLC can undershoot in a strong rarefaction).
        let pre = sim.fields.prim.pre_field().expect("Newtonian requires prim.pre");
        crate::regimes::fofc::fofc_orchestrate(
            sim,
            "adiabatic",
            "", // the DOF != D early-return above means this path is always DOF == D
            <Self as KernelSet<D, DOF, Mem, Sc>>::has_additive_source(self),
            &self.cfl_scratch,
            pre,
            |dir| dispatch_flux(sim, pre, "adiabatic", dir, self.gamma, 0.0, Solver::Hlle, false),
            || self.c2p(sim),
            || self.godunov_stage(sim, dt, a0, ac),
            || self.source_apply(sim, ac * dt),
        );
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
        let sfx = if DOF != D { geom_suffix(sim.geom.coords, DOF, D) } else { "" };
        let name = format!("adiabatic_snapshot{sfx}_{D}d");
        dispatch_fields::<Sc, Mem, D>(
            &name, &sim.geom.allocated, &sim.geom.interior, &inputs, &outputs, &[], &[],
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
                && fused_runtime_cpu_kernel(sim, rs, GeoSource::Hydro { inertial: true }).is_some()
            {
                return;
            }
            dispatch_runtime_source(sim, rs, weight);
        }
    }

    fn body_source(&self, sim: &FieldStore<D, DOF, Mem, Sc>, dt: f64) {
        // forward immersed-body source (gravity + accretion, docs/design/19): cons += dt*S, in-place.
        dispatch_body_source(sim, dt, self.gamma);
    }

    fn body_feedback(&self, sim: &FieldStore<D, DOF, Mem, Sc>, dt: f64) {
        // backward feedback (docs/design/19 P3): reduce per-body force/torque/mass -> diagnostics.
        dispatch_body_feedback(sim, dt, self.gamma);
    }
}
