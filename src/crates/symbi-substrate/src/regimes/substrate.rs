// =============================================================================
// regimes/substrate.rs
//
// IsoSubstrateKernelSet<Mem, Sc, const D> — the D-GENERIC isothermal (Newtonian
// Euler, p = cs^2*rho) KernelSet, every method dispatched to a build-time AOT
// substrate kernel through the structured binding ABI, the instance resolved by name
// (regime, ndim, dir) via the generated `kernel_by_name` registry. one struct serves
// 1D/2D/3D.
//
// the isothermal regime carries no energy law: cons / flux / u_n have den + momentum
// only. the closure p = cs^2*rho is materialized as a SUBSTRATE-OWNED pressure
// primitive (`self.pre`) — kept off the global iso field ABI, which does not allocate
// prim.pre on CPU — so the flux + cfl read a per-cell sound speed sqrt(gamma*p/rho)
// (gamma = ISO_GAMMA = 1) rather than a global cs constant. c2p writes self.pre, the
// ghost fill pulls it back, flux + cfl read it.
//
// the godunov / snapshot / rk2 are the EOS-generic `iso_*` kernels (no energy law);
// the flux + wave-speed map + ghost fill are the iso ones shared with the Newtonian
// CFL path. validated by the full isothermal Euler evolution through the real
// evolve() loop in tests/substrate_evolve_smoke.rs.
// =============================================================================

use symbi_algebra::{Domain, OrderedNumeric};
use symbi_grid::Field;
use symbi_ir::ScalarRef;
use symbi_ir::algebra::Scalar;
use symbi_xpu::MemorySpace;

use std::sync::Arc;
use symbi_hydro::source_spec::BuiltSource;

use crate::kernels::support::{GhostFillDriver, to_bc_array};
use crate::regimes::substrate_kernels::{
    FusedSourceBinding, RuntimeSource, ScalarBind, Solver, cfl_wave_speed,
    dispatch_body_feedback_iso, dispatch_body_source_iso, dispatch_fields, dispatch_flux,
    dispatch_fused_runtime_cpu, dispatch_godunov_maybe_fused, dispatch_godunov_with_body_source,
    dispatch_runtime_source, dispatch_source_apply, fused_runtime_cpu_kernel, resolve_params,
};
use symbi_discretize::gv::GeoSource;
use symbi_geometry::Geometry;
use symbi_sim::state::FieldStore;
use symbi_sim::substrate_seam::KernelSet;

/// the adiabatic index of the isothermal limit: `cs^2 = gamma*p/rho` with the
/// closure `p = cs^2*rho` recovers the isothermal sound speed when `gamma = 1`. the
/// sound speed is derived per cell from `p`, never global.
const ISO_GAMMA: f64 = 1.0;

/// a D-generic isothermal `KernelSet`, every method substrate-generated.
pub struct IsoSubstrateKernelSet<Mem: MemorySpace, Sc: Scalar + OrderedNumeric, const D: usize> {
    pub cs: f64,
    pub cfl_number: f64,
    /// the theta-MC reconstruction compression (regime-generic; 1 == plain minmod).
    pub theta: f64,
    /// the PRESCRIBED per-cell sound-speed-squared `cs^2` (read-only; the local
    /// "temperature"). c2p reads it -> `prim.pre = cs2*rho`. `new` fills it uniformly with
    /// `cs^2` (globally isothermal); for a LOCALLY isothermal run, overwrite it at IC time
    /// (e.g., `cs2(x) ~ 1/r` for a disk) before the first step.
    pub cs2: Field<Sc, D, Mem>,
    /// the substrate-owned pressure primitive (off the global iso field ABI). c2p writes
    /// the interior (= cs2*rho), ghost_fill the ghosts; flux + cfl reconstruct/read it.
    pub pre: Field<Sc, D, Mem>,
    pub cfl_scratch: Field<Sc, D, Mem>,
    /// **B6-iv Phase 4b**: declarative routing to the AOT-baked fused godunov
    /// (`iso_godunov_euler_with_{source_id}_{D}d`). `None` => the unfused kernel
    /// (the original default; bit-identical behavior for all existing callers).
    /// `Some(...)` => `godunov_euler` / `godunov_rk2` route to the fused variant
    /// in ONE launch with the binding's `scalars` filled into the kernel.
    pub fused_source: Option<FusedSourceBinding>,
    /// **S3b**: the NON-fused (additive) source overlay. `Some` => the step loop
    /// snapshots `u_stage` + runs the standalone `iso_source_with_{slug}_{D}d`
    /// pass after each godunov stage (`cons += ac*dt*S`). proven bit-for-bit
    /// equal to `fused_source` with the same binding (S2 + the evolve-level
    /// `additive_source_matches_fused_trajectory`). mutually exclusive with
    /// `fused_source` in practice — the same physics, two execution strategies.
    pub additive_source: Option<FusedSourceBinding>,
    /// **Gap B**: a RUNTIME-loaded user source (python -> json -> `build_user_source`). regime-
    /// agnostic mechanism (shared with the energy regimes); iso stamps `has_energy = false`, so a
    /// `cooling` or `nrg`-targeted source was already rejected at `build_user_source`.
    pub runtime_source: Option<Arc<RuntimeSource>>,
    /// **v2 inc 3+4**: when true AND a `runtime_source` is attached, the runtime user source is
    /// FUSED into the godunov stage as ONE Cranelift-JIT'd host kernel instead of the two-pass.
    /// opt-in + gated (host + f64); falls back to the two-pass otherwise. proven bit-for-bit by
    /// `jit_fused_equals_two_pass`.
    pub fuse_runtime: bool,
}

impl<Mem: MemorySpace, Sc: Scalar + OrderedNumeric, const D: usize>
    IsoSubstrateKernelSet<Mem, Sc, D>
{
    pub fn new(cs: f64, cfl_number: f64, alloc_domain: &Domain<D>) -> Self {
        let cs2 =
            Field::<Sc, D, Mem>::zeros(alloc_domain).expect("failed to allocate iso cs2 field");
        // globally isothermal default: cs2(x) = cs^2 everywhere. a locally isothermal run
        // overwrites cs2 at IC time.
        let cs_sq = Sc::from_f64(cs * cs);
        for coord in alloc_domain.iter() {
            cs2.set(coord, cs_sq);
        }
        let pre = Field::<Sc, D, Mem>::zeros(alloc_domain)
            .expect("failed to allocate substrate pressure field");
        let cfl_scratch = Field::<Sc, D, Mem>::zeros(alloc_domain)
            .expect("failed to allocate iso CFL scratch field");
        Self {
            cs,
            cfl_number,
            theta: 1.0,
            cs2,
            pre,
            cfl_scratch,
            fused_source: None,
            additive_source: None,
            runtime_source: None,
            fuse_runtime: false,
        }
    }

    /// **Gap B**: attach a RUNTIME-loaded user source (the regime-agnostic mechanism). build it
    /// against the ISO spec so ill-posed configs (cooling / `nrg` target — iso has no energy) are
    /// rejected up front:
    /// `let built = build_user_source(&cfg, &ISO_NEWTONIAN_SPEC)?;`
    /// `let sub = sim.substrate().with_runtime_source(built, cfg.params.clone());`
    pub fn with_runtime_source(
        mut self,
        built: Vec<(String, BuiltSource)>,
        params: Vec<f64>,
    ) -> Self {
        // has_energy = false is the AUTHORITY here (iso's RegimeSpec): no `nrg` write is emitted.
        self.runtime_source = Some(RuntimeSource::new(built, params, false));
        self
    }

    /// **v2 inc 3+4**: attach a runtime user source AND route it through the FUSED host path (the
    /// source rides inside the Cranelift-JIT'd godunov stage, one launch). same physics as
    /// `with_runtime_source`, bit-for-bit, proven by `jit_fused_equals_two_pass`; host + f64 only,
    /// else it falls back to the two-pass.
    pub fn with_fused_runtime_source(
        mut self,
        built: Vec<(String, BuiltSource)>,
        params: Vec<f64>,
    ) -> Self {
        self.runtime_source = Some(RuntimeSource::new(built, params, false));
        self.fuse_runtime = true;
        self
    }

    /// set the theta-MC limiter compression in [1,2] (1 = minmod, 2 = monotonized-central). fluent.
    pub fn theta(mut self, theta: f64) -> Self {
        self.theta = theta;
        self
    }

    /// **B6-iv Phase 4b**: bind a fused-source AOT kernel for this kernel-set.
    /// fluent builder, intended to chain off `new(..)`:
    /// `IsoSubstrateKernelSet::new(cs, cfl, alloc).with_fused_source(binding)`. one
    /// AOT-baked kernel covers `div(F) + spec source(s) + integrator` per step.
    pub fn with_fused_source(mut self, binding: FusedSourceBinding) -> Self {
        self.fused_source = Some(binding);
        self
    }

    /// **S3b**: bind the SAME source as a NON-fused additive pass (plain godunov +
    /// per-stage `source_apply`) instead of fusing it into the godunov kernel.
    /// the general execution path; bit-for-bit equal to `with_fused_source` for a
    /// baked family (the dispatch in step 4 prefers fused-when-baked, falls back
    /// here). fluent builder; chain off `new(..)`.
    pub fn with_additive_source(mut self, binding: FusedSourceBinding) -> Self {
        self.additive_source = Some(binding);
        self
    }

    /// the isothermal sound speed squared, `cs^2(x) = p(x) / rho(x)` — just physics. fills
    /// the (read-only) per-cell `cs2` field from a pressure + density profile. globally
    /// isothermal is the uniform-`p/rho` special case; a varying `p/rho` is locally
    /// isothermal. call once after the initial density + pressure are set (cs^2 is then held
    /// fixed for the run). `domain` is where the profile is valid (typically the interior).
    pub fn compute_isothermal_cs2(
        &self,
        rho: &Field<Sc, D, Mem>,
        pre: &Field<Sc, D, Mem>,
        domain: &Domain<D>,
    ) {
        for coord in domain.iter() {
            self.cs2.set(coord, *pre.at(coord) / *rho.at(coord));
        }
    }
}

impl<Mem: MemorySpace + Sync, Sc: Scalar + OrderedNumeric, const D: usize> KernelSet<D, D, Mem, Sc>
    for IsoSubstrateKernelSet<Mem, Sc, D>
{
    fn flux(&self, sim: &FieldStore<D, D, Mem, Sc>, dir: usize) {
        // the gv iso flux IS the Newtonian flux at gamma->1: it reconstructs prim.pre
        // (= cs^2(x)*rho) and takes cs = sqrt(gamma*p/rho) = sqrt(p/rho) = the LOCAL sound
        // speed, so a locally isothermal cs2(x) flows through naturally. gamma = ISO_GAMMA = 1;
        // no flux.nrg (the energy U/F is dead-code-eliminated). + the theta-MC limiter.
        // iso is HLLE-only by physics (no contact wave); the substrate enforces it
        // here by hardcoding the solver rather than exposing a per-set knob.
        dispatch_flux(
            sim,
            &self.pre,
            "iso",
            dir,
            ISO_GAMMA,
            self.theta,
            Solver::Hlle,
        );
    }

    fn c2p(&self, sim: &FieldStore<D, D, Mem, Sc>) {
        // inputs (manifest order): cons den, mom_0.., then the prescribed cs2 field.
        // outputs: prim rho, vel_0.., self.pre (= cs2*rho). cs2 is a FIELD (read-only),
        // not a scalar — so the run can be locally isothermal. no scalar params.
        let mut inputs: Vec<&Field<Sc, D, Mem>> = vec![&sim.fields.cons.den];
        for k in 0..D {
            inputs.push(&sim.fields.cons.mom[k]);
        }
        inputs.push(&self.cs2);
        let mut outputs: Vec<&Field<Sc, D, Mem>> = vec![&sim.fields.prim.rho];
        for k in 0..D {
            outputs.push(&sim.fields.prim.vel[k]);
        }
        outputs.push(&self.pre);
        let name = format!("iso_c2p_{D}d");
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

    fn godunov_stage(&self, sim: &FieldStore<D, D, Mem, Sc>, dt: f64, a0: f64, ac: f64) {
        // immersed bodies: gravity + accretion are FUSED into the godunov update
        // (one launch, additive convention), so `body_source` is a no-op. cs feeds
        // the accretion rate cap (iso passes the constant self.cs). the fused kernel
        // is baked for Cartesian; curvilinear falls back to the separate body_source
        // pass (the fused cyl/sph bake is a follow-up).
        if sim.immersed.is_some() && sim.geom.coords == Geometry::Cartesian {
            dispatch_godunov_with_body_source(sim, &self.pre, "iso", dt, a0, ac, self.cs);
            return;
        }
        // the geometric-source pressure is the substrate-owned self.pre (= cs^2*rho).
        // fused_source: None => unfused kernel (the prior default), Some => AOT-baked
        // fused variant in one launch (`iso_godunov_stage_with_{source_id}_{D}d`).
        // the FUSED runtime-source path (one JIT'd godunov+source launch); gated host+f64, the
        // source's separate pass skipped in `source_apply` under the same predicate. iso reads the
        // substrate-owned pressure `&self.pre` (= cs^2*rho); geo = Hydro{inertial:true} matches the
        // AOT `iso` godunov.
        if self.fuse_runtime {
            if let Some(rs) = &self.runtime_source {
                if let Some(fk) =
                    fused_runtime_cpu_kernel(sim, rs, GeoSource::Hydro { inertial: true })
                {
                    dispatch_fused_runtime_cpu(sim, &self.pre, fk, rs, dt, a0, ac);
                    return;
                }
            }
        }
        dispatch_godunov_maybe_fused(
            sim,
            &self.pre,
            "iso",
            dt,
            a0,
            ac,
            self.fused_source.as_ref(),
        );
    }

    fn cfl(&self, sim: &FieldStore<D, D, Mem, Sc>) -> f64 {
        // iso wave-speed map (cs = sqrt(gamma*p/rho) from the substrate-owned self.pre);
        // the SHARED cfl dispatch binds the field buffers by manifest + owns the reduction.
        cfl_wave_speed(
            sim,
            &self.pre,
            &self.cfl_scratch,
            "iso",
            ISO_GAMMA,
            self.cfl_number,
        )
    }

    fn ghost_fill(&self, sim: &FieldStore<D, D, Mem, Sc>) {
        // the lattice-map pullback (iso_ghost_fill_{D}d): rho/vel/pre, in-place. the
        // pressure is pulled back too (a grade-0 scalar, like density).
        let bc = to_bc_array::<D>(&sim.boundaries);
        let name = format!("iso_ghost_fill_{D}d");
        GhostFillDriver::<D>::new(&sim.geom.allocated, &sim.geom.interior, bc).drive_sweep(
            |region, p| {
                let mut outputs: Vec<&Field<Sc, D, Mem>> = vec![&sim.fields.prim.rho];
                for k in 0..D {
                    outputs.push(&sim.fields.prim.vel[k]);
                }
                outputs.push(&self.pre);
                // params BY NAME via the type-sorted manifest: map_type/arg are INT lanes, vel_sign
                // FLOAT — each routed to its ABI tail by the kernel's declared sort (the int ⊔ float
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
    }

    fn snapshot(&self, sim: &FieldStore<D, D, Mem, Sc>) {
        // u_n = cons (pure copy), no energy law, over the full allocated domain.
        let mut inputs: Vec<&Field<Sc, D, Mem>> = vec![&sim.fields.cons.den];
        for k in 0..D {
            inputs.push(&sim.fields.cons.mom[k]);
        }
        let mut outputs: Vec<&Field<Sc, D, Mem>> = vec![&sim.workspace.u_n.den];
        for k in 0..D {
            outputs.push(&sim.workspace.u_n.mom[k]);
        }
        let name = format!("iso_snapshot_{D}d");
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

    fn has_additive_source(&self) -> bool {
        self.additive_source.is_some() || self.runtime_source.is_some()
    }

    fn fofc_active(&self) -> bool {
        true
    }

    fn fofc(&self, sim: &FieldStore<D, D, Mem, Sc>, dt: f64, a0: f64, ac: f64) {
        // isothermal is HLLE-only by physics; the first-order redo is the same fan at theta = 0
        // (PCM) — the positivity-preserving Einfeldt fan. the substrate-owned pressure (cs^2*rho)
        // feeds the flux as in the production sweep.
        crate::regimes::fofc::fofc_orchestrate(
            sim,
            "iso",
            self.has_additive_source(),
            &self.cfl_scratch,
            &sim.fields.cons.den,
            |dir| dispatch_flux(sim, &self.pre, "iso", dir, ISO_GAMMA, 0.0, Solver::Hlle),
            || self.c2p(sim),
            || self.godunov_stage(sim, dt, a0, ac),
            || self.source_apply(sim, ac * dt),
        );
    }

    fn snapshot_stage(&self, sim: &FieldStore<D, D, Mem, Sc>) {
        // u_stage = cons (pure copy via the snapshot kernel), positional buffers (no nrg).
        // identical to `snapshot` but targets u_stage — the stage-input state the additive
        // source pass reads. snapshot over the interior (where source_apply iterates).
        let mut inputs: Vec<&Field<Sc, D, Mem>> = vec![&sim.fields.cons.den];
        for k in 0..D {
            inputs.push(&sim.fields.cons.mom[k]);
        }
        let mut outputs: Vec<&Field<Sc, D, Mem>> = vec![&sim.workspace.u_stage.den];
        for k in 0..D {
            outputs.push(&sim.workspace.u_stage.mom[k]);
        }
        let name = format!("iso_snapshot_{D}d");
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

    fn source_apply(&self, sim: &FieldStore<D, D, Mem, Sc>, weight: f64) {
        if let Some(b) = &self.additive_source {
            dispatch_source_apply(sim, "iso", b, weight);
        }
        if let Some(rs) = &self.runtime_source {
            if self.fuse_runtime
                && fused_runtime_cpu_kernel(sim, rs, GeoSource::Hydro { inertial: true }).is_some()
            {
                return; // already fused into the godunov stage
            }
            dispatch_runtime_source(sim, rs, weight);
        }
    }

    fn body_source(&self, sim: &FieldStore<D, D, Mem, Sc>, dt: f64) {
        // Cartesian: fused into the godunov stage (no-op here). curvilinear: the
        // fused kernel isn't baked yet, so run the separate frame-correct pass.
        if sim.geom.coords != Geometry::Cartesian {
            dispatch_body_source_iso(sim, &self.pre, dt);
        }
    }

    fn body_feedback(&self, sim: &FieldStore<D, D, Mem, Sc>, dt: f64) {
        // backward feedback (force/torque/accreted-mass -> diagnostics), isothermal.
        dispatch_body_feedback_iso(sim, &self.pre, dt);
    }
}
