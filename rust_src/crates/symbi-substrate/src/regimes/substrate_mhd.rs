// =============================================================================
// regimes/substrate_mhd.rs
//
// MhdSubstrateKernelSet<R, Mem, Sc, D> — the ONE ideal-MHD KernelSet, generic over the
// regime `R` (RMHD / NewtonianMhd / IsothermalMhd; DOF = 3 fixed). docs/design/35 R1.
//
// the three MHD families differ ONLY by data read off `R::SPEC`:
//   - `Self::kernel_prefix()`                     -> the AOT kernel-name prefix (rmhd / nmhd / imhd)
//   - `R::SPEC.has_energy`               -> whether the pre / nrg field rows are bound
//                                           (gamma EOS scalar vs the iso `cs`)
//   - `R::SPEC.materializes_wave_speeds` -> RMHD writes the quartic ws_l/ws_r in a
//                                           `wave_speeds` pass for the HLLE flux to read;
//                                           NMHD/iMHD compute the magnetosonic speed inline
// the gas godunov + the ENTIRE constrained-transport stack are regime-agnostic and delegate
// to `mhd_substrate` (the SAME AOT kernels). the per-regime structs (`RmhdSubstrateKernelSet`
// etc.) are now back-compat type aliases of this one. see docs/design/29, /30.
//
// usage:
//  let sub = MhdSubstrateKernelSet::<Rmhd, HostMemory, f64, 3>::new(gamma, cfl, theta, &alloc);
//  // or, identically, via the back-compat alias:
//  let sub = RmhdSubstrateKernelSet::<HostMemory, f64, 3>::new(gamma, cfl, theta, &alloc);
// =============================================================================

use std::marker::PhantomData;

use symbi_algebra::{Domain, OrderedNumeric};
use symbi_ir::algebra::Scalar;
use symbi_ir::ScalarRef;
use symbi_grid::Field;
use symbi_hydro::regime::Regime;
use symbi_xpu::MemorySpace;

use crate::kernels::support::cfl_from_lambda;
use std::sync::Arc;

use crate::regimes::substrate_kernels::{
    dispatch_named, dispatch_runtime_source, geom_scalar, mhd_flux_suffix, mhd_geom_suffix,
    motion_scalar, physical_geom, scalars_for, RegimeKind, RuntimeSource, ScalarBind, Solver,
};
use symbi_hydro::source_spec::BuiltSource;
use symbi_sim::substrate_seam::KernelSet;
use symbi_sim::state::FieldStore;

/// the unified D-dimensional ideal-MHD `KernelSet`, regime supplied as `R` (carries `SPEC`).
pub struct MhdSubstrateKernelSet<R, Mem: MemorySpace, Sc: Scalar + OrderedNumeric, const D: usize> {
    /// the EOS scalar: `gamma` (ideal-gas regimes) or `cs` (isothermal). bound to the flux/cfl
    /// kernel's eos param by name (`gamma` when `has_energy`, else `cs`).
    pub eos_param: f64,
    pub cfl_number: f64,
    /// theta-MC limiter compression in [1,2]: 1 = minmod, 2 = monotonized-central.
    pub theta: f64,
    pub cfl_scratch: Field<Sc, D, Mem>,
    /// Riemann solver — HLLE (default) / HLLC / HLLD; validated against the regime at attach.
    pub solver: Solver,
    /// a runtime user source (python -> json `SourceConfig` -> `build_user_source`),
    /// applied two-pass via the regime-agnostic `dispatch_runtime_source`. nmhd/imhd
    /// take the newtonian force/cooling/relax lifts; rmhd is relativistic so only
    /// `kind="raw"` reaches here. targets the hydro conserved slots (den/mom/nrg);
    /// B is evolved by CT, not a cell source. None for source-free runs.
    pub runtime_source: Option<Arc<RuntimeSource>>,
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
        Self { eos_param, cfl_number, theta, cfl_scratch, solver: Solver::Hlle, runtime_source: None, _r: PhantomData }
    }

    /// attach a runtime user source from already-lowered `(target, BuiltSource)` pairs
    /// (the `build_user_source` output of a `SourceConfig`). applied two-pass in
    /// `source_apply`. has_energy is read from the regime spec (rmhd/nmhd carry it,
    /// imhd does not), so the source pass writes only the slots the regime owns.
    pub fn with_runtime_source(mut self, built: Vec<(String, BuiltSource)>, params: Vec<f64>) -> Self {
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

impl<R, Mem, Sc, const D: usize> KernelSet<D, 3, Mem, Sc>
    for MhdSubstrateKernelSet<R, Mem, Sc, D>
where
    R: Regime<Sc, D>,
    Mem: MemorySpace + Sync,
    Sc: Scalar + OrderedNumeric,
{
    fn flux(&self, sim: &FieldStore<D, 3, Mem, Sc>, dir: usize) {
        // face domain extended +1 on the sweep hi + 1 on each transverse axis (CT corners).
        let mut face = sim.geom.interior.extend(dir, 0, 1);
        for ax in 0..D {
            if ax != dir {
                face = face.expand(ax, 1);
            }
        }
        let gsfx = mhd_flux_suffix(sim.geom.coords, &sim.geom.axes);
        let flux_name = format!("{}_face_flux{gsfx}{}_{D}d_{dir}", Self::kernel_prefix(), self.solver.kernel_suffix());
        let scalars = scalars_for(&flux_name, |bind| match bind {
            // gamma (energy regimes) and cs (isothermal) are the EOS param's two names.
            ScalarBind::Ref(ScalarRef::Gamma) | ScalarBind::Ref(ScalarRef::Cs) => {
                Sc::from_f64(self.eos_param)
            }
            ScalarBind::Ref(ScalarRef::Theta) => Sc::from_f64(self.theta),
            o => panic!("{} flux: unexpected scalar {o:?}", Self::kernel_prefix()),
        });
        // bind BY MANIFEST: the staggered `bface_n` (-> bface[dir]) and the per-axis wave speeds
        // resolve through the typed `resolve_path` like every cell field, and the per-buffer
        // dispatch layout binds bface with its OWN (staggered) domain. no hand-ordered list.
        let pre_bind = if R::SPEC.has_energy {
            sim.fields.prim.pre_field().expect("MHD energy regime requires prim.pre")
        } else {
            &sim.fields.cons.den
        };
        dispatch_named(sim, pre_bind, None, dir, &flux_name, &face, &[], &scalars);
    }

    fn c2p(&self, sim: &FieldStore<D, 3, Mem, Sc>) {
        let cname = format!("{}_c2p_{D}d", Self::kernel_prefix());
        // iso c2p declares no scalars -> scalars_for returns [] (resolver never called).
        let scalars = scalars_for(&cname, |bind| match bind {
            ScalarBind::Ref(ScalarRef::Gamma) | ScalarBind::Ref(ScalarRef::Cs) => {
                Sc::from_f64(self.eos_param)
            }
            o => panic!("{} c2p: unexpected scalar {o:?}", Self::kernel_prefix()),
        });
        // bind BY MANIFEST: cons.{den,mom,nrg?} + bcell(3) reads -> prim.{rho,vel,pre?} writes.
        // the energy regimes' `prim.pre` is an OUTPUT here, so `pre` binds the real field.
        let pre_bind = if R::SPEC.has_energy {
            sim.fields.prim.pre_field().expect("MHD energy regime requires prim.pre")
        } else {
            &sim.fields.cons.den
        };
        dispatch_named(sim, pre_bind, None, 0, &cname, &sim.geom.interior, &[], &scalars);
    }

    fn wave_speeds(&self, sim: &FieldStore<D, 3, Mem, Sc>) {
        // only materializing regimes (RMHD) run the quartic pass; others compute inline (no-op).
        if !R::SPEC.materializes_wave_speeds {
            return;
        }
        let wsname = format!("{}_wave_speeds_cell_{D}d", Self::kernel_prefix());
        let scalars = scalars_for(&wsname, |bind| match bind {
            ScalarBind::Ref(ScalarRef::Gamma) => Sc::from_f64(self.eos_param),
            o => panic!("{} wave_speeds: unexpected scalar {o:?}", Self::kernel_prefix()),
        });
        // bind BY MANIFEST: prim + bcell reads -> the per-axis `wave_speed_{l,r}[k]` writes
        // (typed `WaveSpeedL/R(k)`). materializing regimes carry energy, so prim.pre is real.
        let pre = sim.fields.prim.pre_field().expect("MHD energy regime requires prim.pre");
        dispatch_named(sim, pre, None, 0, &wsname, &sim.geom.allocated, &[], &scalars);
    }

    fn cfl(&self, sim: &FieldStore<D, 3, Mem, Sc>) -> f64 {
        let geom = &sim.geom;
        let wname = format!("{}_wave_speed_map{}_{D}d", Self::kernel_prefix(), mhd_geom_suffix(geom.coords, &geom.axes));
        // scalars BY NAME (the kernel's declared set drives it): eos param + the per-axis CFL
        // widths (cartesian `inv_dx_d`, curvilinear `x_lo_d`/`dx_d`); the mhd substrates run
        // static, so the motion rates bind 0.
        let (x_lo_phys, dx_phys) = physical_geom(&geom.x_lo, &geom.dx, geom.coords, sim.motion.a);
        let scalars = scalars_for(&wname, |bind| {
            let ScalarBind::Ref(sref) = bind else {
                panic!("{} cfl: unexpected spec scalar {bind:?}", Self::kernel_prefix());
            };
            match *sref {
                ScalarRef::Gamma | ScalarRef::Cs => Sc::from_f64(self.eos_param),
                other => Sc::from_f64(
                    motion_scalar(&sim.motion, geom.coords, D, other)
                        .or_else(|| geom_scalar(&x_lo_phys, &dx_phys, other))
                        .unwrap_or_else(|| panic!("{} cfl: unexpected scalar {other:?}", Self::kernel_prefix())),
                ),
            }
        });
        // bind BY MANIFEST: prim + bcell reads -> the `scratch` lambda write (the cfl_scratch
        // field, supplied as the scratch override). iso passes a dummy pre (reads cs^2*rho).
        let pre_bind = if R::SPEC.has_energy {
            sim.fields.prim.pre_field().expect("MHD energy regime requires prim.pre")
        } else {
            &sim.fields.cons.den
        };
        dispatch_named(sim, pre_bind, Some(&self.cfl_scratch), 0, &wname, &geom.interior, &[], &scalars);
        let lambda_max = crate::regimes::substrate_gpu::field_max_reduce(&self.cfl_scratch, &geom.interior);
        cfl_from_lambda(lambda_max, self.cfl_number)
    }

    // ---- regime-agnostic tails: the gas godunov + the full CT stack (shared AOT kernels) ----
    fn godunov_stage(&self, sim: &FieldStore<D, 3, Mem, Sc>, dt: f64, a0: f64, ac: f64) {
        crate::regimes::mhd_substrate::godunov_stage(sim, R::SPEC.has_energy, Self::kernel_prefix(), self.eos_param, dt, a0, ac);
    }
    fn ghost_fill(&self, sim: &FieldStore<D, 3, Mem, Sc>) {
        crate::regimes::mhd_substrate::ghost_fill(sim, R::SPEC.has_energy);
    }
    fn snapshot(&self, sim: &FieldStore<D, 3, Mem, Sc>) {
        crate::regimes::mhd_substrate::snapshot(sim, R::SPEC.has_energy);
    }
    fn efield(&self, sim: &FieldStore<D, 3, Mem, Sc>) {
        crate::regimes::mhd_substrate::efield(sim);
    }
    fn post_godunov(&self, sim: &FieldStore<D, 3, Mem, Sc>, dt: f64, stage: u8) {
        crate::regimes::mhd_substrate::post_godunov(sim, R::SPEC.has_energy, dt, stage);
    }

    fn has_additive_source(&self) -> bool {
        self.runtime_source.is_some()
    }

    fn snapshot_stage(&self, sim: &FieldStore<D, 3, Mem, Sc>) {
        // capture the stage-input gas cons into u_stage so source_apply reads the
        // pre-godunov state (S2 invariant). without this, u_stage is zero and the
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
