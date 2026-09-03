// =============================================================================
// regimes/substrate.rs
//
// IsoSubstrateKernelSet<Mem, Sc, const D> — the D-generic isothermal (Newtonian
// Euler, p = cs^2*rho) KernelSet, every method dispatched to a build-time AOT
// substrate kernel through the structured binding ABI, the instance resolved by name
// (regime, ndim, dir) via the generated `kernel_by_name` registry. one struct serves
// 1D/2D/3D.
//
// the isothermal regime carries no energy law: cons / flux / u_n have den + momentum
// only. the closure p = cs^2*rho is materialized as a substrate-owned pressure
// primitive (`self.pre`) — kept off the global iso field ABI, which does not allocate
// prim.pre on CPU — so the flux + cfl read a per-cell sound speed sqrt(gamma*p/rho)
// (gamma = ISO_GAMMA = 1), letting a locally-varying sound speed hold. c2p writes self.pre, the
// ghost fill pulls it back, flux + cfl read it.
//
// the godunov / snapshot / rk2 are the EOS-generic `iso_*` kernels (no energy law);
// the flux + wave-speed map + ghost fill are the iso ones shared with the Newtonian
// CFL path. validated by the full isothermal Euler evolution through the real
// evolve() loop in tests/substrate_evolve_smoke.rs.
// =============================================================================

use symbi_algebra::{Domain, OrderedNumeric};
use symbi_carrier::Scalar;
use symbi_grid::Field;
use symbi_ir::ScalarRef;
use symbi_xpu::MemorySpace;

use std::sync::Arc;
use symbi_source_compile::source_spec::SourceProgram;

use crate::kernels::support::{GhostFillDriver, to_bc_array};
use crate::regimes::substrate_kernels::{
    FluxSpec, FusedSourceBinding, GradientBc, RuntimeSource, ScalarBind, Solver, cfl_wave_speed,
    dispatch_body_feedback_iso, dispatch_body_source_iso, dispatch_driven_boundaries,
    dispatch_fields, dispatch_flux, dispatch_fused_runtime_cpu, dispatch_godunov_maybe_fused,
    dispatch_godunov_with_body_source, dispatch_gradient_boundaries, dispatch_named,
    dispatch_runtime_source, dispatch_source_apply, fused_runtime_cpu_kernel, geom_scalar,
    motion_scalar, resolve_params, scalars_for,
};
use symbi_discretize::gv::GeoSource;
use symbi_geometry::Geometry;
use symbi_sim::state::FieldStore;
use symbi_sim::substrate_seam::{KernelSet, WithViscosity};

/// the adiabatic index of the isothermal limit: `cs^2 = gamma*p/rho` with the
/// closure `p = cs^2*rho` recovers the isothermal sound speed when `gamma = 1`. the
/// sound speed is derived per cell from `p`, never global.
const ISO_GAMMA: f64 = 1.0;

/// a D-generic isothermal `KernelSet`, every method substrate-generated.
pub struct IsoSubstrateKernelSet<Mem: MemorySpace, Sc: Scalar + OrderedNumeric, const D: usize> {
    pub cs: f64,
    /// dimensionless acoustic crossing factor: tau = c_drain dx / signal_speed.
    pub c_drain: f64,
    /// constant kinematic viscosity nu. 0 = inviscid (the
    /// viscous pass and its CFL cap are inert). >0 selects the Navier-Stokes
    /// shear operator and caps dt at C_visc dx^2 / nu.
    pub viscosity: f64,
    /// Shakura-Sunyaev alpha. >0 selects the alpha viscous
    /// operator nu(x) = alpha cs^2 / Omega_k(r) (takes precedence over the
    /// constant-nu `viscosity`); requires a central body. 0 = off.
    pub alpha: f64,
    pub cfl_number: f64,
    /// the theta-MC reconstruction compression (regime-generic; 1 == plain minmod).
    pub theta: f64,
    /// the prescribed per-cell sound-speed-squared `cs^2` (read-only; the local
    /// "temperature"). c2p reads it -> `prim.pre = cs2*rho`. `new` fills it uniformly with
    /// `cs^2` (globally isothermal); for a locally isothermal run, overwrite it at IC time
    /// (e.g., `cs2(x) ~ 1/r` for a disk) before the first step.
    pub cs2: Field<Sc, D, Mem>,
    /// the substrate-owned pressure primitive (off the global iso field ABI). c2p writes
    /// the interior (= cs2*rho), ghost_fill the ghosts; flux + cfl reconstruct/read it.
    pub pre: Field<Sc, D, Mem>,
    pub cfl_scratch: Field<Sc, D, Mem>,
    /// declarative routing to the AOT-baked fused godunov
    /// (`iso_godunov_euler_with_{source_id}_{D}d`). `None` => the unfused kernel
    /// (the original default; bit-identical behavior for all existing callers).
    /// `Some(...)` => `godunov_euler` / `godunov_rk2` route to the fused variant
    /// in one launch with the binding's `scalars` filled into the kernel.
    pub fused_source: Option<FusedSourceBinding>,
    /// the non-fused (additive) source overlay. `Some` => the step loop
    /// snapshots `u_stage` + runs the standalone `iso_source_with_{slug}_{D}d`
    /// pass after each godunov stage (`cons += ac*dt*S`). proven bit-for-bit
    /// equal to `fused_source` with the same binding (proven at the evolve level by
    /// `additive_source_matches_fused_trajectory`). mutually exclusive with
    /// `fused_source` in practice — the same physics, two execution strategies.
    pub additive_source: Option<FusedSourceBinding>,
    /// **Gap B**: a runtime-loaded user source (python -> json -> `build_user_source`). regime-
    /// agnostic mechanism (shared with the energy regimes); iso stamps `has_energy = false`, so a
    /// `cooling` or `nrg`-targeted source was already rejected at `build_user_source`.
    pub runtime_source: Option<Arc<RuntimeSource>>,
    /// when true and a `runtime_source` is attached, the runtime user source is
    /// fused into the godunov stage as one Cranelift-JIT'd host kernel, replacing the two-pass.
    /// opt-in + gated (host + f64); falls back to the two-pass otherwise. bit-for-bit identical
    /// to the two-pass path.
    pub fuse_runtime: bool,
    /// gradient-boundary (Neumann / Robin) coefficients, indexed by the `BoundaryType::Neumann(id)` /
    /// `Robin(id)` id. the shared fills, fed `cs^2` so the ghost honours `pre = cs^2*rho`.
    pub gradient_bcs: Vec<GradientBc>,
    /// driven-boundary DAGs, indexed by the `BoundaryType::Driven(id)` id. each prescribes a
    /// face's ghost `[rho, vel..]` (no pressure slot — the isothermal closure re-derives the
    /// ghost pressure as `cs2 * rho` after the fill). empty => no driven faces.
    pub boundary_dags: Vec<Arc<RuntimeSource>>,
    /// consecutive substages the FOFC freeze tier fired (persistent-freeze fail-loud; see fofc.rs).
    pub freeze_streak: std::sync::atomic::AtomicU32,
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
            c_drain: 1.0,
            viscosity: 0.0,
            alpha: 0.0,
            cfl_number,
            theta: 1.0,
            cs2,
            pre,
            cfl_scratch,
            fused_source: None,
            additive_source: None,
            runtime_source: None,
            fuse_runtime: false,
            gradient_bcs: Vec::new(),
            boundary_dags: Vec::new(),
            freeze_streak: std::sync::atomic::AtomicU32::new(0),
        }
    }

    /// nu_max for the alpha viscous CFL cap: nu(r) = alpha cs^2 / Omega_k(r)
    /// grows with r (Omega_k = sqrt(GM/r^3)), so the maximum over the interior is
    /// at the corner farthest from the central body. returns 0 (cap inert) with
    /// no body — the alpha dispatch then fails loud.
    fn alpha_nu_max(&self, sim: &FieldStore<D, D, Mem, Sc>) -> f64 {
        let geom = &sim.geom;
        let Some(im) = sim.immersed.as_ref() else {
            return 0.0;
        };
        if im.bodies.is_empty() {
            return 0.0;
        }
        let b = im.bodies.get(0);
        let gm = b.mass;
        if gm <= 0.0 {
            return 0.0;
        }
        // nu grows outward (nu ~ R^{3/2}), so nu_max sits at the largest orbital
        // radius. on a cylindrical (R, phi) grid R is that radius, so nu_max is at
        // the outer R edge. cartesian forms it as the farthest domain corner from
        // the body in the disk plane (the first two axes; the vertical z in 3D does
        // not enter Omega_k) — matching the kernel's nu(x, y) exactly.
        let r_max = if geom.coords == Geometry::Cylindrical || geom.coords == Geometry::Spherical {
            let sp = &geom.interior.spaces[0];
            geom.x_lo[0] + geom.dx[0] * (sp.hi as f64)
        } else {
            let plane = D.min(2);
            let mut r_max = 0.0_f64;
            for corner in 0..(1usize << D) {
                let mut d2 = 0.0;
                for a in 0..plane {
                    let sp = &geom.interior.spaces[a];
                    let idx = if corner & (1 << a) != 0 { sp.hi } else { sp.lo };
                    let x = geom.x_lo[a] + geom.dx[a] * (idx as f64);
                    let d = x - b.position[a];
                    d2 += d * d;
                }
                r_max = r_max.max(d2.sqrt());
            }
            r_max
        };
        if r_max <= 0.0 {
            return 0.0;
        }
        let omega_min = (gm / (r_max * r_max * r_max)).sqrt();
        self.alpha * self.cs * self.cs / omega_min
    }

    /// register a neumann / robin gradient boundary (the convenience short-circuit). the id
    /// (registration order) is what the sim's `Boundaries` carries as `BoundaryType::Neumann(id)` /
    /// `Robin(id)`. `coeffs` are the per-variable coefficients in prim order `[rho, vel.., pre]` (the
    /// pressure coefficient is ignored — iso re-derives pre = cs^2*rho at the ghost).
    pub fn with_gradient_boundary(mut self, coeffs: GradientBc) -> (Self, u16) {
        let id = self.gradient_bcs.len() as u16;
        self.gradient_bcs.push(coeffs);
        (self, id)
    }

    /// **Gap B**: attach a runtime-loaded user source (the regime-agnostic mechanism). build it
    /// against the iso spec so ill-posed configs (cooling / `nrg` target — iso has no energy) are
    /// rejected up front:
    /// `let built = build_user_source(&cfg, &ISO_NEWTONIAN_SPEC)?;`
    /// `let sub = sim.substrate().with_runtime_source(built, cfg.params.clone());`
    pub fn with_runtime_source(
        mut self,
        built: Vec<(String, SourceProgram)>,
        params: Vec<f64>,
    ) -> Self {
        // has_energy = false is the authority here (iso's RegimeSpec): no `nrg` write is emitted.
        self.runtime_source = Some(RuntimeSource::new(built, params, false));
        self
    }

    /// attach a runtime user source and route it through the fused host path (the
    /// source rides inside the Cranelift-JIT'd godunov stage, one launch). same physics as
    /// `with_runtime_source`, bit-for-bit identical; host + f64 only,
    /// else it falls back to the two-pass.
    pub fn with_fused_runtime_source(
        mut self,
        built: Vec<(String, SourceProgram)>,
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

    /// bind a fused-source AOT kernel for this kernel-set.
    /// fluent builder, intended to chain off `new(..)`:
    /// `IsoSubstrateKernelSet::new(cs, cfl, alloc).with_fused_source(binding)`. one
    /// AOT-baked kernel covers `div(F) + spec source(s) + integrator` per step.
    pub fn with_fused_source(mut self, binding: FusedSourceBinding) -> Self {
        self.fused_source = Some(binding);
        self
    }

    /// bind the same source as a non-fused additive pass (plain godunov +
    /// per-stage `source_apply`).
    /// the general execution path; bit-for-bit equal to `with_fused_source` for a
    /// baked family (the dispatch prefers fused-when-baked, falls back
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

    /// register a driven boundary DAG; the returned id (registration order) is what the sim's
    /// `Boundaries` must carry as `BoundaryType::Driven(id)` on the prescribed face. build
    /// `built` from `expr_bridge::build_boundary_dag(&cfg, &ISO_NEWTONIAN_SPEC)` — the
    /// prescription is `[rho, vel..]` only (no pressure slot; the isothermal closure re-derives
    /// the ghost pressure as `cs2 * rho` after the fill).
    pub fn with_driven_boundary(
        mut self,
        built: Vec<(String, SourceProgram)>,
        params: Vec<f64>,
    ) -> (Self, u16) {
        let id = self.boundary_dags.len() as u16;
        // has_energy = false: no prim.pre assignment rides the dag.
        self.boundary_dags
            .push(RuntimeSource::new(built, params, false));
        (self, id)
    }

    /// extend the per-cell `cs2` into every ghost cell by clamping each axis into the
    /// interior — the zero-gradient continuation of the fixed temperature field, covering
    /// faces, edges, and corners in one pass. without this a locally isothermal run keeps
    /// the constructor's uniform `cs^2` in the ghosts (the interior-only derive never
    /// touches them), and the `p = cs2 * rho` ghost-pressure pass then books that alien
    /// temperature into every boundary flux — for a cold disk edge (`cs2 ~ 1e-3`) against
    /// the default `cs = 1`, a ~1000x spurious boundary pressure.
    pub fn extend_cs2_into_ghosts(&self, allocated: &Domain<D>, interior: &Domain<D>) {
        for coord in allocated.iter() {
            if interior.contains(coord) {
                continue;
            }
            let clamped: [isize; D] = std::array::from_fn(|ax| {
                coord[ax].clamp(interior.spaces[ax].lo, interior.spaces[ax].hi - 1)
            });
            self.cs2.set(coord, *self.cs2.at(clamped));
        }
    }
}

impl<Mem: MemorySpace + Sync, Sc: Scalar + OrderedNumeric, const D: usize>
    IsoSubstrateKernelSet<Mem, Sc, D>
{
    /// the evolution face-flux scheme. iso is hlle-only by physics (no contact
    /// wave), plm-only, plain-balanced; the mach limit is inert on hlle.
    fn flux_spec(&self) -> FluxSpec {
        FluxSpec {
            theta: self.theta,
            solver: Solver::Hlle,
            recon: symbi_discretize::Recon::Plm,
            eos: symbi_discretize::EosArm::IdealGamma,
            flatten: (0.0, 0.0),
            balance: symbi_discretize::coords::Balance::Plain,
            rusanov: false,
        }
    }
}

impl<Mem: MemorySpace + Sync, Sc: Scalar + OrderedNumeric, const D: usize> KernelSet<D, D, Mem, Sc>
    for IsoSubstrateKernelSet<Mem, Sc, D>
{
    fn flux(&self, sim: &FieldStore<D, D, Mem, Sc>, dir: usize) {
        // the gv iso flux is the Newtonian flux at gamma->1: it reconstructs prim.pre
        // (= cs^2(x)*rho) and takes cs = sqrt(gamma*p/rho) = sqrt(p/rho) = the local sound
        // speed, so a locally isothermal cs2(x) flows through naturally. gamma = ISO_GAMMA = 1;
        // no flux.nrg (the energy U/F is dead-code-eliminated). + the theta-MC limiter.
        // iso is HLLE-only by physics (no contact wave); the substrate enforces it
        // here by hardcoding the solver.
        dispatch_flux(sim, &self.pre, "iso", dir, ISO_GAMMA, self.flux_spec());
    }

    fn c2p(&self, sim: &FieldStore<D, D, Mem, Sc>) {
        // the primitives now hold a state recovered from the conserved fields; anything
        // reading prim.* outside the evolve loop checks this before trusting it.
        sim.mark_primitives_recovered();
        // inputs (manifest order): cons den, mom_0.., then the prescribed cs2 field.
        // outputs: prim rho, vel_0.., self.pre (= cs2*rho). cs2 is a read-only field,
        // so the run can be locally isothermal. no scalar params.
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
        // the c2p status channel: the recovery kernel writes its accept/reject
        // fact alongside the candidate, so the channel has one producer.
        outputs.push(&sim.fields.c2p_error);
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
        // immersed bodies: gravity + accretion are fused into the godunov update
        // (one launch, additive convention), so `body_source` is a no-op. cs feeds
        // the accretion rate cap (iso passes the constant self.cs). the fused kernel
        // is baked for Cartesian; curvilinear falls back to the separate body_source
        // pass (no fused cyl/sph bake exists).
        if sim.immersed.is_some() && sim.geom.coords == Geometry::Cartesian {
            dispatch_godunov_with_body_source(sim, &self.pre, "iso", dt, a0, ac, self.cs);
            return;
        }
        // the geometric-source pressure is the substrate-owned self.pre (= cs^2*rho).
        // fused_source: None => unfused kernel (the default), Some => AOT-baked
        // fused variant in one launch (`iso_godunov_stage_with_{source_id}_{D}d`).
        // the fused runtime-source path (one JIT'd godunov+source launch); gated host+f64, the
        // source's separate pass skipped in `source_apply` under the same predicate. iso reads the
        // substrate-owned pressure `&self.pre` (= cs^2*rho); geo = Hydro{inertial:true} matches the
        // AOT `iso` godunov.
        if self.fuse_runtime {
            if let Some(rs) = &self.runtime_source {
                if let Some(fk) =
                    fused_runtime_cpu_kernel(sim, rs, GeoSource::Hydro { inertial: true }, false)
                {
                    // iso reaches this path body-free (the Cartesian-body case early-returns above),
                    // so the fused kernel carries no body fold; self.cs is the inert `Cs` binding.
                    dispatch_fused_runtime_cpu(sim, &self.pre, fk, Some(rs), dt, a0, ac, self.cs);
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
            // iso carries no energy, so the covariant-energy godunov never binds the EOS scalar;
            // the sound speed is passed only to satisfy the (unreached) Gamma/Cs arm.
            self.cs,
            self.fused_source.as_ref(),
        );
    }

    fn cfl(&self, sim: &FieldStore<D, D, Mem, Sc>) -> f64 {
        // iso wave-speed map (cs = sqrt(gamma*p/rho) from the substrate-owned self.pre);
        // the shared cfl dispatch binds the field buffers by manifest + owns the reduction.
        let dt_hydro = cfl_wave_speed(
            sim,
            &self.pre,
            &self.cfl_scratch,
            "iso",
            symbi_discretize::EosArm::IdealGamma,
            ISO_GAMMA,
            self.cfl_number,
            None,
            0.0,
        );
        // the parabolic viscous cap: an explicit momentum-diffusion step is
        // stable for dt <= C_visc dx^2 / nu_max. the 2D Navier-Stokes normal
        // stress carries a 4/3 factor (tau_xx = rho nu (4/3 d_x v_x - ...)), so
        // the von-Neumann limit is ~0.21 dx^2/nu for the self-diffusion and lower
        // once the cross terms couple x and y; C_visc = 0.1 sits safely below it
        // (0.25, the plain-Laplacian value, is unstable and blows the velocity
        // up). inert when inviscid. for alpha the viscosity grows with radius, so
        // nu_max is at the domain corner farthest from the body.
        let nu_max = if self.alpha > 0.0 {
            self.alpha_nu_max(sim)
        } else {
            self.viscosity
        };
        let dt = if nu_max > 0.0 {
            const C_VISC: f64 = 0.1;
            // the diffusion cap uses the physical min cell size. cylindrical (R, phi)
            // has a coordinate azimuthal width dphi, so its physical extent is R dphi,
            // smallest at the inner edge — using the raw angle would leave the inner
            // annulus under-resolved and unstable. cartesian widths are already
            // physical.
            let curvilinear =
                sim.geom.coords == Geometry::Cylindrical || sim.geom.coords == Geometry::Spherical;
            let min_dx = if curvilinear {
                // h2 = the radial coordinate on both curvilinear charts, so the x2
                // physical width is r*dx2, smallest at the inner edge.
                let dr = sim.geom.dx[0];
                let r_min = sim.geom.x_lo[0] + 0.5 * dr; // innermost cell centroid
                dr.min((r_min * sim.geom.dx[1]).abs())
            } else {
                sim.geom.dx.iter().copied().fold(f64::INFINITY, f64::min)
            };
            dt_hydro.min(C_VISC * min_dx * min_dx / nu_max)
        } else {
            dt_hydro
        };
        let min_physical_width = super::substrate_mhd::max_inv_physical_width(&sim.geom).recip();
        crate::regimes::substrate_kernels::body_gravity_limited_dt(
            sim,
            dt,
            self.cfl_number,
            min_physical_width,
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
        // the driven faces (skipped by the pullback: Driven -> Skip): prescribe their ghost
        // [rho, vel..] from the registered boundary DAGs. the pressure is not prescribed —
        // the eos pass below re-derives it as cs2 * rho over every ghost the fill wrote.
        if !self.boundary_dags.is_empty() {
            dispatch_driven_boundaries(sim, &self.boundary_dags);
        }
        // the Neumann/Robin gradient faces (skipped by the pullback), filled from the edge cell over
        // the substrate-owned pressure field; Some(cs^2) makes the shared kernel honour the
        // isothermal closure pre = cs^2*rho at the ghost.
        if !self.gradient_bcs.is_empty() {
            dispatch_gradient_boundaries(
                sim,
                &self.pre,
                &self.gradient_bcs,
                Some(self.cs * self.cs),
            );
        }
        // re-derive the eos law p = cs^2 * rho over the full allocated lattice. the
        // substrate pressure lives outside the prim batch, so the coarse-fine ghost
        // prolongation (which fills prim rho/vel) never writes it, and the pullback above
        // skips coarse-fine faces — without this pass the face reconstruction at a level
        // seam reads pre = 0 and injects a spurious vacuum pressure (a density ring at
        // the refinement boundary). idempotent where c2p already wrote pre (rho = den
        // there, same product), and the single source of pressure in every ghost the
        // reconstruction can reach.
        {
            let name = format!("iso_pre_{D}d");
            dispatch_fields::<Sc, Mem, D>(
                &name,
                &sim.geom.allocated,
                &sim.geom.allocated,
                &[&sim.fields.prim.rho, &self.cs2],
                &[&self.pre],
                &[],
                &[],
            );
        }
        // the dye concentration ghost band: a true scalar (reflect sign +1). gradient faces
        // resolve to a zero-derivative copy, since a prescribed normal derivative is a
        // per-primitive-variable quantity and the dye carries none.
        if let Some(chi) = sim.fields.prim.chi_field() {
            crate::regimes::mhd_substrate::flag_ghost_fill(
                sim,
                chi,
                crate::kernels::support::to_bc_array_scalar::<D>(&sim.boundaries),
            );
        }
    }

    /// the interface dye flux, written during the flux phase so the coarse-fine registers sample it
    /// alongside the gas fluxes. the dye kernels read only the mass flux and `prim.chi`, so the same
    /// baked instances serve every regime.
    fn chi_flux(&self, sim: &FieldStore<D, D, Mem, Sc>) {
        if !sim.has_passive_scalar() {
            return;
        }
        for dir in 0..D {
            let mut band = sim.geom.interior.clone();
            band.spaces[dir].hi += 1;
            let fname = format!("chi_flux_{dir}_{D}d");
            dispatch_named(sim, &self.pre, None, dir, &fname, &band, &[], &[]);
        }
    }

    fn chi_update(&self, sim: &FieldStore<D, D, Mem, Sc>, dt: f64, a0: f64, ac: f64) {
        if !sim.has_passive_scalar() {
            return;
        }
        // the dye divergence divides by the physical cell width, so its geom scalars resolve
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
        dispatch_named(
            sim,
            &self.pre,
            None,
            0,
            &name,
            &sim.geom.interior,
            &[],
            &scalars,
        );
        let cname = format!("chi_c2p_{D}d");
        dispatch_named(
            sim,
            &self.pre,
            None,
            0,
            &cname,
            &sim.geom.interior,
            &[],
            &[],
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

    fn fofc(
        &self,
        sim: &FieldStore<D, D, Mem, Sc>,
        dt: f64,
        a0: f64,
        ac: f64,
        _stage: u8,
    ) -> symbi_sim::substrate_seam::FofcReport {
        // isothermal is HLLE-only by physics; the first-order redo is the same fan at theta = 0
        // (PCM) — the positivity-preserving Einfeldt fan. the substrate-owned pressure (cs^2*rho)
        // feeds the flux as in the production sweep.
        crate::regimes::fofc::fofc_orchestrate(
            sim,
            "iso",
            "", // iso is DOF == D (no swirl lift)
            self.has_additive_source(),
            &self.freeze_streak,
            |dir| {
                dispatch_flux(
                    sim,
                    &self.pre,
                    "iso",
                    dir,
                    ISO_GAMMA,
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
            || None, // iso: no admissible-boundary projection (density-only admissibility; keeps the freeze)
            // freeze parachute evolves by the iso body source (eos param = cs, no energy field).
            sim.immersed.is_some().then(|| (ac * dt, self.cs)),
            crate::regimes::fofc::CtHooks::none(),
            || symbi_sim::substrate_seam::SourceReplayOutcome::SharedRedo, // hydro: no source replay
            false, // no projection tier below the freeze; keep the parachute
        )
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
                && fused_runtime_cpu_kernel(sim, rs, GeoSource::Hydro { inertial: true }, false)
                    .is_some()
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

    fn penalize(&self, sim: &FieldStore<D, D, Mem, Sc>, dt: f64) {
        // the [Drain] stack: the sole accretion mechanism on
        // cartesian grids; the iso kernel reads `cs` through the eos-param slot.
        crate::regimes::substrate_kernels::dispatch_penalize(sim, dt, self.cs, self.c_drain);
    }

    fn viscous(&self, sim: &FieldStore<D, D, Mem, Sc>, dt: f64) {
        // the Navier-Stokes shear; inert when inviscid. baked
        // for 2D and 3D cartesian — fail loud otherwise (a silent drop would discard
        // the transport a viscous run declared). alpha (spatially varying nu)
        // takes precedence over the constant-nu viscosity.
        if self.alpha <= 0.0 && self.viscosity <= 0.0 {
            return;
        }
        let coords = sim.geom.coords;
        assert!(
            coords == Geometry::Cartesian
                || coords == Geometry::Cylindrical
                || coords == Geometry::Spherical,
            "viscosity is baked for cartesian (2D/3D) and curvilinear (2D, the general \
             orthogonal operator) only"
        );
        // both dispatches select the cartesian or cylindrical kernel by coords and
        // assert the supported dimension (alpha: cartesian 2D/3D, cylindrical 2D).
        if self.alpha > 0.0 {
            crate::regimes::substrate_kernels::dispatch_viscous_alpha(sim, dt, self.alpha, self.cs);
        } else {
            crate::regimes::substrate_kernels::dispatch_viscous(sim, dt, self.viscosity);
        }
    }

    fn body_feedback(&self, sim: &FieldStore<D, D, Mem, Sc>, dt: f64) {
        // backward feedback (force/torque/accreted-mass -> diagnostics), isothermal.
        dispatch_body_feedback_iso(sim, &self.pre, dt);
    }
}

impl<Mem: MemorySpace, Sc: Scalar + OrderedNumeric, const D: usize> WithViscosity
    for IsoSubstrateKernelSet<Mem, Sc, D>
{
    fn with_viscosity(mut self, nu: f64) -> Self {
        self.viscosity = nu;
        self
    }
    fn with_alpha(mut self, alpha: f64) -> Self {
        self.alpha = alpha;
        self
    }
}

// horizon excision is a GR-chart operation; the isothermal (flat) set ignores it.
impl<Mem: MemorySpace, Sc: Scalar + OrderedNumeric, const D: usize>
    symbi_sim::substrate_seam::WithExcision for IsoSubstrateKernelSet<Mem, Sc, D>
{
}

// resistivity is an MHD operation; the pure-hydro isothermal set has no magnetic field, ignores it.
impl<Mem: MemorySpace, Sc: Scalar + OrderedNumeric, const D: usize>
    symbi_sim::substrate_seam::WithResistivity for IsoSubstrateKernelSet<Mem, Sc, D>
{
}
