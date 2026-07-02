// =============================================================================
// regimes/substrate_rhd.rs
//
// RhdSubstrateKernelSet<Mem, Sc, const D> — the D-GENERIC RHD (special-
// relativistic Euler) KernelSet, every method dispatched to a build-time AOT
// substrate kernel through the structured binding ABI (a KernelInvocation of
// ordered Buf handles, inputs-then-outputs, run_cpu / GPU JIT). the kernel instance
// is resolved by name (regime, ndim, dir) via the generated `kernel_by_name`
// registry — one struct serves 1D/2D/3D, no per-dimension copy.
//
// the relativistic pieces: the masked-Newton c2p (`rhd_c2p_{D}d`), the HLLE flux
// with relativistic U/F/wave speeds per sweep dir (`rhd_face_flux_{D}d_{dir}`), and
// the relativistic CFL wave-speed map (`rhd_wave_speed_map_{D}d`). the godunov /
// snapshot / rk2 are the SAME EOS-generic kernels as the Newtonian sets (D/S_k/tau ==
// den/mom/nrg in structure), and the ghost_fill is the SHARED lattice-map pullback
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
use symbi_grid::Field;
use symbi_ir::ScalarRef;
use symbi_ir::algebra::Scalar;
use symbi_xpu::MemorySpace;

use std::sync::Arc;

use crate::kernels::support::{GhostFillDriver, to_bc_array};
use crate::regimes::substrate_kernels::{
    RuntimeSource, ScalarBind, Solver, cfl_wave_speed, dispatch_fields, dispatch_flux, dispatch_rhd_ks_shift_flux,
    dispatch_godunov, dispatch_runtime_source, geom_scalar, kernel_geom, resolve_params, scalars_for,
    spacing_suffix,
};
use symbi_hydro::source_spec::BuiltSource;
use symbi_sim::state::FieldStore;
use symbi_sim::substrate_seam::KernelSet;

/// a D-generic RHD `KernelSet`, every method substrate-generated.
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
        }
    }

    /// attach a runtime user source from already-lowered `(target, BuiltSource)`
    /// pairs (the `build_user_source` output of a `kind="raw"` RHD `SourceConfig`).
    /// applied two-pass in `source_apply` (plain godunov + `dispatch_runtime_source`).
    pub fn with_runtime_source(
        mut self,
        built: Vec<(String, BuiltSource)>,
        params: Vec<f64>,
    ) -> Self {
        // has_energy = true (RHD carries tau); validation happened at
        // `build_user_source(cfg, &RHD_SPEC)`.
        self.runtime_source = Some(RuntimeSource::new(built, params, true));
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
    pub fn theta(mut self, theta: f64) -> Self {
        self.theta = theta;
        self
    }
}

impl<Mem: MemorySpace + Sync, Sc: Scalar + OrderedNumeric, const D: usize> KernelSet<D, D, Mem, Sc>
    for RhdSubstrateKernelSet<Mem, Sc, D>
{
    fn flux(&self, sim: &FieldStore<D, D, Mem, Sc>, dir: usize) {
        let pre = sim.fields.prim.pre_field().expect("Rhd requires prim.pre");
        dispatch_flux(sim, pre, "rhd", dir, self.gamma, self.theta, self.solver);
    }

    fn ks_shift(&self, sim: &FieldStore<D, D, Mem, Sc>, dir: usize) {
        // ingoing-Kerr-Schild shift-advection added to the radial face flux; no-op unless the
        // background is KerrSchild and dir == 0 (the dispatch gates both).
        let pre = sim.fields.prim.pre_field().expect("Rhd requires prim.pre");
        dispatch_rhd_ks_shift_flux(sim, pre, dir);
    }

    fn c2p(&self, sim: &FieldStore<D, D, Mem, Sc>) {
        let cnrg = sim.fields.cons.nrg_field().expect("Rhd requires cons.nrg");
        let pre = sim.fields.prim.pre_field().expect("Rhd requires prim.pre");

        // inputs: cons den, mom_0..mom_{D-1}, nrg. outputs: prim rho, vel_0.., pre.
        let mut inputs: Vec<&Field<Sc, D, Mem>> = vec![&sim.fields.cons.den];
        for k in 0..D {
            inputs.push(&sim.fields.cons.mom[k]);
        }
        inputs.push(cnrg);
        let mut outputs: Vec<&Field<Sc, D, Mem>> = vec![&sim.fields.prim.rho];
        for k in 0..D {
            outputs.push(&sim.fields.prim.vel[k]);
        }
        outputs.push(pre);

        // the GR path uses the metric-aware Valencia recovery (`|S|^2 = gamma^{ij} S_i S_j`,
        // contravariant `v^i`); its name carries the spacing + spacetime slug and it reads the lapse
        // mass M + the LOG-AWARE radial grid scalars (the metric is evaluated at the cell centroid).
        // flat keeps the plain `rhd_c2p_{D}d` (gamma only), bit-identical.
        let st_sfx = match sim.geom.spacetime {
            symbi_geometry::Spacetime::Minkowski => "",
            symbi_geometry::Spacetime::Schwarzschild => "_schw",
            symbi_geometry::Spacetime::KerrSchild => "_ks",
        };
        let (name, scalars) = if st_sfx.is_empty() {
            let name = format!("rhd_c2p_{D}d");
            let scalars = scalars_for(&name, |bind| match bind {
                ScalarBind::Ref(ScalarRef::Gamma) => Sc::from_f64(self.gamma),
                o => panic!("rhd c2p: unexpected scalar {o:?}"),
            });
            (name, scalars)
        } else {
            let sp_sfx = spacing_suffix(&sim.geom.maps);
            let name = format!("rhd_c2p{sp_sfx}{st_sfx}_{D}d");
            let (x_lo, dx) = kernel_geom(&sim.geom.x_lo, &sim.geom.dx, &sim.geom.maps, sim.geom.coords, sim.motion.a);
            let scalars = scalars_for(&name, |bind| {
                let ScalarBind::Ref(sref) = bind else {
                    panic!("rhd GR c2p: unexpected spec scalar {bind:?}");
                };
                match *sref {
                    ScalarRef::Gamma => Sc::from_f64(self.gamma),
                    ScalarRef::SchwarzschildMass => Sc::from_f64(
                        sim.geom.spacetime_scalars.iter()
                            .find(|(n, _)| n == "schwarzschild_mass")
                            .map(|(_, v)| *v)
                            .expect("rhd GR c2p needs schwarzschild_mass but the metric supplied none"),
                    ),
                    other => Sc::from_f64(
                        geom_scalar(&x_lo, &dx, other)
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

    fn godunov_stage(&self, sim: &FieldStore<D, D, Mem, Sc>, dt: f64, a0: f64, ac: f64) {
        let pre = sim.fields.prim.pre_field().expect("Rhd requires prim.pre");
        dispatch_godunov(sim, pre, "rhd", dt, a0, ac);
    }

    fn has_additive_source(&self) -> bool {
        self.runtime_source.is_some()
    }

    fn source_apply(&self, sim: &FieldStore<D, D, Mem, Sc>, weight: f64) {
        // two-pass: plain godunov already ran; add the raw user source increment to
        // the conserved fields. dispatch_runtime_source is regime-agnostic (it adds
        // the BuiltSource outputs to their target conserved slots), so no RHD-specific
        // source code — the relativistic conservation law lives in the godunov stage.
        if let Some(rs) = &self.runtime_source {
            dispatch_runtime_source(sim, rs, weight);
        }
    }

    fn cfl(&self, sim: &FieldStore<D, D, Mem, Sc>) -> f64 {
        // the SHARED cfl dispatch binds the field buffers by manifest + owns the reduction.
        // RHD's only contribution is the "rhd" map (its relativistic wave speed).
        let pre = sim.fields.prim.pre_field().expect("Rhd requires prim.pre");
        cfl_wave_speed(
            sim,
            pre,
            &self.cfl_scratch,
            "rhd",
            self.gamma,
            self.cfl_number,
        )
    }

    fn ghost_fill(&self, sim: &FieldStore<D, D, Mem, Sc>) {
        // the SHARED lattice-map pullback (iso_ghost_fill_{D}d): the EOS-generic
        // 3-field prim pullback (rho/vel/pre), in-place, per ghost region.
        let bc = to_bc_array::<D>(&sim.boundaries);
        let pre = sim.fields.prim.pre_field().expect("Rhd requires prim.pre");
        let name = format!("iso_ghost_fill_{D}d");

        GhostFillDriver::<D>::new(&sim.geom.allocated, &sim.geom.interior, bc).drive_sweep(
            |region, p| {
                let mut outputs: Vec<&Field<Sc, D, Mem>> = vec![&sim.fields.prim.rho];
                for k in 0..D {
                    outputs.push(&sim.fields.prim.vel[k]);
                }
                outputs.push(pre);
                // ints: map_type_0..{D-1}, arg_0..{D-1}. scalars: vel_sign_0..{D-1}.
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
        let cnrg = sim.fields.cons.nrg_field().expect("cons.nrg");
        let unrg = sim.workspace.u_n.nrg_field().expect("u_n.nrg");

        // inputs: cons den, mom_0.., nrg. outputs: u_n den, mom_0.., nrg.
        let mut inputs: Vec<&Field<Sc, D, Mem>> = vec![&sim.fields.cons.den];
        for k in 0..D {
            inputs.push(&sim.fields.cons.mom[k]);
        }
        inputs.push(cnrg);
        let mut outputs: Vec<&Field<Sc, D, Mem>> = vec![&sim.workspace.u_n.den];
        for k in 0..D {
            outputs.push(&sim.workspace.u_n.mom[k]);
        }
        outputs.push(unrg);

        let name = format!("rhd_snapshot_{D}d");
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
}
