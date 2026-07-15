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
    dispatch_driven_boundaries, dispatch_named, dispatch_runtime_source, geom_scalar,
    kernel_geom, mhd_flux_suffix, mhd_geom_suffix, motion_scalar, scalars_for,
    spacetime_slug, RegimeKind, RuntimeSource, ScalarBind, Solver,
};
use symbi_hydro::source_spec::BuiltSource;
use symbi_sim::substrate_seam::KernelSet;
use symbi_sim::state::FieldStore;
use symbi_sim::state::CtMethod;

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
            let bt = if side == 0 { sim.boundaries.lo(a) } else { sim.boundaries.hi(a) };
            if !matches!(bt, symbi_sim::state::BoundaryType::Driven(_)) {
                continue;
            }
            let s = if side == 0 { symbi_algebra::Side::Lo } else { symbi_algebra::Side::Hi };
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
    /// B is evolved by CT, not a cell source. None for source-free runs.
    pub runtime_source: Option<Arc<RuntimeSource>>,
    /// driven (DYNAMIC) boundary prescriptions, indexed by `BoundaryType::Driven(id)`. each is a
    /// complete prim DAG `[rho, vel.., pre, B..]`; the standard ghost-fill skips the driven face,
    /// then `dispatch_driven_boundaries` assigns its ghost state. empty => no driven faces.
    pub boundary_dags: Vec<Arc<RuntimeSource>>,
    /// consecutive substages on which the FOFC last-resort freeze tier fired; a genuine unrecoverable
    /// poison freezes every stage, while the rare correct parachute is isolated (see fofc.rs).
    pub freeze_streak: std::sync::atomic::AtomicU32,
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
        Self { eos_param, cfl_number, theta, cfl_scratch, solver: Solver::Hlle, ct_method: CtMethod::Contact, runtime_source: None, boundary_dags: Vec::new(), freeze_streak: std::sync::atomic::AtomicU32::new(0), _r: PhantomData }
    }

    /// register a DRIVEN (DYNAMIC) boundary. the returned id (registration order) is what the
    /// sim's `Boundaries` carries as `BoundaryType::Driven(id)` on the prescribed face. build
    /// `built` from `expr_bridge::build_boundary_dag(&cfg, R::SPEC)` — a complete prim prescription
    /// `[rho, vel.., pre, B..]`. for a purely toroidal injection the in-plane B is 0 and the
    /// out-of-plane B_phi is the injected toroidal field (cell-centered, div-free by axisymmetry).
    pub fn with_driven_boundary(mut self, built: Vec<(String, BuiltSource)>, params: Vec<f64>) -> (Self, u16) {
        let id = self.boundary_dags.len() as u16;
        self.boundary_dags.push(RuntimeSource::new(built, params, R::SPEC.has_energy));
        (self, id)
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
    /// the flux sweep, parameterized by the flat solver suffix, the GR HLLD toggle, and the slope
    /// limiter `theta` — so FOFC can re-run it at FIRST ORDER (HLLE + theta = 0) through the same
    /// code path the production sweep uses. the production `flux` calls this with the configured
    /// solver + `self.theta`.
    fn flux_impl(&self, sim: &FieldStore<D, 3, Mem, Sc>, dir: usize, flat_suffix: &str, gr_solver: &str, theta: f64) {
        // face domain extended +1 on the sweep hi + 1 on each transverse axis (CT corners).
        let mut face = sim.geom.interior.extend(dir, 0, 1);
        for ax in 0..D {
            if ax != dir {
                face = face.expand(ax, 1);
            }
        }
        let st = spacetime_slug(sim.geom.spacetime);
        let flux_name = if st.is_empty() {
            let gsfx = mhd_flux_suffix(sim.geom.coords, &sim.geom.axes);
            format!("{}_face_flux{gsfx}{flat_suffix}_{D}d_{dir}", Self::kernel_prefix())
        } else {
            // the metric-aware valencia flux (RmhdGr). "" = HLLE, "_hlld" = tetrad MUB09, "_rusanov"
            // = the light-cone Lax-Friedrichs fan (the FOFC first-order fallback).
            let solver = gr_solver;
            let gsfx = mhd_geom_suffix(sim.geom.coords, &sim.geom.axes);
            format!("{}_face_flux{solver}{gsfx}{st}_{D}d_{dir}", Self::kernel_prefix())
        };
        let (x_lo_k, dx_k) = kernel_geom(&sim.geom.x_lo, &sim.geom.dx, &sim.geom.maps, sim.geom.coords, sim.motion.a);
        let scalars = scalars_for(&flux_name, |bind| match bind {
            ScalarBind::Ref(ScalarRef::Gamma) | ScalarBind::Ref(ScalarRef::Cs) => Sc::from_f64(self.eos_param),
            ScalarBind::Ref(ScalarRef::Theta) => Sc::from_f64(theta),
            ScalarBind::Ref(ScalarRef::SchwarzschildMass) => Sc::from_f64(
                sim.geom.spacetime_scalars.iter().find(|(n, _)| n == "schwarzschild_mass")
                    .map(|(_, v)| *v).expect("GR MHD flux needs schwarzschild_mass"),
            ),
            ScalarBind::Ref(ScalarRef::KerrSpin) => Sc::from_f64(
                sim.geom.spacetime_scalars.iter().find(|(n, _)| n == "kerr_spin")
                    .map(|(_, v)| *v).expect("GR MHD flux needs kerr_spin"),
            ),
            ScalarBind::Ref(other) => Sc::from_f64(
                geom_scalar(&x_lo_k, &dx_k, &sim.geom.maps, *other)
                    .unwrap_or_else(|| panic!("{} flux: unexpected scalar {other:?}", Self::kernel_prefix())),
            ),
            o => panic!("{} flux: unexpected scalar {o:?}", Self::kernel_prefix()),
        });
        let pre_bind = if R::SPEC.has_energy {
            sim.fields.prim.pre_field().expect("MHD energy regime requires prim.pre")
        } else {
            &sim.fields.cons.den
        };
        dispatch_named(sim, pre_bind, None, dir, &flux_name, &face, &[], &scalars);
    }

    /// FIRST-ORDER FLUX CORRECTION for the MHD regimes: the shared face-based gas redo (first-order
    /// flux = the light-cone Lax-Friedrichs / Rusanov fan) PLUS the C2 constrained-transport re-sync
    /// (§3'). the induction/CT subsystem is INVARIANT under a gas FOFC — B evolves by curl-of-EMF,
    /// independent of the gas c2p — so the face field, EMF, and curl stay HIGH-ORDER; the redo only
    /// (a) restores the HO induction flux for the cell-B predictor so its magnetic-energy patch is
    /// the small HO reconciliation (not the FO-vs-HO shock), and (b) re-runs `bcell_from_bface` on the
    /// patch stages to re-attach `bcell = interp(bface_HO)` + the patch onto the corrected gas.
    fn fofc_impl(&self, sim: &FieldStore<D, 3, Mem, Sc>, dt: f64, a0: f64, ac: f64, stage: u8) {
        let pre_bind = if R::SPEC.has_energy {
            sim.fields.prim.pre_field().expect("MHD energy regime requires prim.pre")
        } else {
            &sim.fields.cons.den
        };
        // `bcell_from_bface` (the face->cell interp + magnetic-energy patch) runs on the SINGLE
        // (Euler, tag 0) and CORRECTOR (rk2, tag 2) stages, never the predictor (tag 1, which leaves
        // bcell flux-predicted). re-sync exactly there.
        let patch_stage = stage == 0 || stage == 2;
        let has_energy = R::SPEC.has_energy;
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
            None,  // no body-evolved freeze parachute (no MHD body source)
            || crate::regimes::mhd_substrate::fofc_ct_save(sim),
            || crate::regimes::mhd_substrate::fofc_restore_bcell_stage(sim),
            || {
                use crate::regimes::mhd_substrate as ct;
                let prefix = Self::kernel_prefix();
                let flag = &sim.workspace.fofc_flag;
                // splice the induction flux at EVERY firing stage (the cell-B predictor reads it);
                ct::fofc_splice_induction(sim, prefix, flag);
                // the CT curl runs only on the patch stages (euler / corrector). there the redo
                // recomputes the FIRST-ORDER edge EMF (Contact/HLL — no UCT per-face wave-speed
                // dependency), splices it by the edge flag (HO off the fallback region, FO on it),
                // restores the pre-curl face field, and re-curls -> flagged cells get diffused,
                // recoverable B; non-flagged faces are unchanged; div(B) stays zero.
                if patch_stage {
                    ct::efield(sim, CtMethod::Contact, self.solver, prefix, self.eos_param, 0.0);
                    ct::fofc_emf_splice(sim, flag);
                    ct::fofc_restore_bface_n(sim);
                    ct::ct_curl(sim, dt);
                }
            },
            || {
                if patch_stage {
                    crate::regimes::mhd_substrate::bcell_from_bface(sim, has_energy);
                }
            },
        );
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
        // the production sweep: the configured solver + slope limiter.
        let gr_solver = if matches!(self.solver, Solver::Hlld) { "_hlld" } else { "" };
        self.flux_impl(sim, dir, self.solver.kernel_suffix(), gr_solver, self.theta);
    }

    fn fofc(&self, sim: &FieldStore<D, 3, Mem, Sc>, dt: f64, a0: f64, ac: f64, stage: u8) {
        self.fofc_impl(sim, dt, a0, ac, stage);
    }

    fn fofc_active(&self) -> bool {
        true
    }

    fn penalize(&self, sim: &FieldStore<D, 3, Mem, Sc>, dt: f64) {
        // the design-50 immersed-body penalization under MHD, via the 1/2|B|^2 sandwich: strip
        // the magnetic energy so the (unchanged) hydro drain acts on the GAS energy alone, run
        // it, then restore the field energy. the drain never touches bcell, so B and 1/2|B|^2
        // are exactly invariant and the flux is left to constrained transport. host-only (the
        // immersed dispatch is host + f64); with no body the shifts would be a wasted no-op.
        if !Mem::IS_HOST_ACCESSIBLE || sim.immersed.is_none() {
            return;
        }
        crate::regimes::mhd_substrate::shift_magnetic_energy(sim, -1.0);
        // eos_param is gamma (has_energy MHD); c_drain uses the adiabatic default 1.0 (plumbing
        // it from config is a follow-on).
        crate::regimes::substrate_kernels::dispatch_penalize(sim, dt, self.eos_param, 1.0);
        crate::regimes::mhd_substrate::shift_magnetic_energy(sim, 1.0);
    }

    fn c2p(&self, sim: &FieldStore<D, 3, Mem, Sc>) {
        let st = spacetime_slug(sim.geom.spacetime);
        let cname = if st.is_empty() {
            format!("{}_c2p_{D}d", Self::kernel_prefix())
        } else {
            // the metric-aware KKC recovery: gamma at the volume-weighted centroid, so the name
            // carries the chart + spacing + spacetime slugs and the kernel reads mass + grid scalars.
            let gsfx = mhd_geom_suffix(sim.geom.coords, &sim.geom.axes);
            format!("{}_c2p{gsfx}{st}_{D}d", Self::kernel_prefix())
        };
        let (x_lo_k, dx_k) = kernel_geom(&sim.geom.x_lo, &sim.geom.dx, &sim.geom.maps, sim.geom.coords, sim.motion.a);
        // iso c2p declares no scalars -> scalars_for returns [] (resolver never called).
        let scalars = scalars_for(&cname, |bind| match bind {
            ScalarBind::Ref(ScalarRef::Gamma) | ScalarBind::Ref(ScalarRef::Cs) => {
                Sc::from_f64(self.eos_param)
            }
            ScalarBind::Ref(ScalarRef::SchwarzschildMass) => Sc::from_f64(
                sim.geom.spacetime_scalars.iter()
                    .find(|(n, _)| n == "schwarzschild_mass")
                    .map(|(_, v)| *v)
                    .expect("GR MHD c2p needs schwarzschild_mass"),
            ),
            ScalarBind::Ref(ScalarRef::KerrSpin) => Sc::from_f64(
                sim.geom.spacetime_scalars.iter()
                    .find(|(n, _)| n == "kerr_spin")
                    .map(|(_, v)| *v)
                    .expect("GR MHD c2p needs kerr_spin"),
            ),
            ScalarBind::Ref(other) => Sc::from_f64(
                geom_scalar(&x_lo_k, &dx_k, &sim.geom.maps, *other)
                    .unwrap_or_else(|| panic!("{} c2p: unexpected scalar {other:?}", Self::kernel_prefix())),
            ),
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
        // materializing regimes (RMHD always; NMHD under UCT, whose edge-EMF coefficients read
        // wave_speed_l/r) run the per-cell pass; others compute speeds inline in the flux.
        let materialize = R::SPEC.materializes_wave_speeds
            || (self.ct_method == CtMethod::Uct && matches!(Self::kernel_prefix(), "nmhd" | "imhd"));
        let st = spacetime_slug(sim.geom.spacetime);
        // GR: the flux computes its bound speeds inline, so the ONLY consumer of the materialized
        // per-cell speeds is the GR-UCT edge EMF. materialize (the cheap SHIFTED BF bound) exactly
        // when UCT is requested on the curved background; skip on GR otherwise.
        if !st.is_empty() {
            if self.ct_method != CtMethod::Uct {
                return;
            }
            let gsfx = mhd_geom_suffix(sim.geom.coords, &sim.geom.axes);
            let wsname = format!("{}_wave_speeds_cell{gsfx}{st}_{D}d", Self::kernel_prefix());
            let (x_lo_k, dx_k) = kernel_geom(&sim.geom.x_lo, &sim.geom.dx, &sim.geom.maps, sim.geom.coords, sim.motion.a);
            let scalars = scalars_for(&wsname, |bind| match bind {
                ScalarBind::Ref(ScalarRef::Gamma) | ScalarBind::Ref(ScalarRef::Cs) => Sc::from_f64(self.eos_param),
                ScalarBind::Ref(ScalarRef::SchwarzschildMass) => Sc::from_f64(
                    sim.geom.spacetime_scalars.iter().find(|(n, _)| n == "schwarzschild_mass")
                        .map(|(_, v)| *v).expect("GR UCT wave speeds need schwarzschild_mass"),
                ),
                ScalarBind::Ref(ScalarRef::KerrSpin) => Sc::from_f64(
                    sim.geom.spacetime_scalars.iter().find(|(n, _)| n == "kerr_spin")
                        .map(|(_, v)| *v).expect("GR UCT wave speeds need kerr_spin"),
                ),
                ScalarBind::Ref(other) => Sc::from_f64(
                    geom_scalar(&x_lo_k, &dx_k, &sim.geom.maps, *other)
                        .unwrap_or_else(|| panic!("{} wave_speeds: unexpected scalar {other:?}", Self::kernel_prefix())),
                ),
                o => panic!("{} wave_speeds: unexpected scalar {o:?}", Self::kernel_prefix()),
            });
            let pre_bind = sim.fields.prim.pre_field().expect("GR MHD requires prim.pre");
            dispatch_named(sim, pre_bind, None, 0, &wsname, &sim.geom.allocated, &[], &scalars);
            return;
        }
        if !materialize {
            return;
        }
        let wsname = format!("{}_wave_speeds_cell_{D}d", Self::kernel_prefix());
        let scalars = scalars_for(&wsname, |bind| match bind {
            ScalarBind::Ref(ScalarRef::Gamma) | ScalarBind::Ref(ScalarRef::Cs) => Sc::from_f64(self.eos_param),
            o => panic!("{} wave_speeds: unexpected scalar {o:?}", Self::kernel_prefix()),
        });
        // bind BY MANIFEST: prim + bcell reads -> the per-axis `wave_speed_{l,r}[k]` writes (typed
        // `WaveSpeedL/R(k)`). energy regimes have prim.pre; isothermal (no pressure) passes den as
        // the leading window field (mirrors the cfl dispatch).
        let pre_bind = if R::SPEC.has_energy {
            sim.fields.prim.pre_field().expect("MHD energy regime requires prim.pre")
        } else {
            &sim.fields.cons.den
        };
        dispatch_named(sim, pre_bind, None, 0, &wsname, &sim.geom.allocated, &[], &scalars);
    }

    fn cfl(&self, sim: &FieldStore<D, 3, Mem, Sc>) -> f64 {
        let geom = &sim.geom;
        let st = spacetime_slug(geom.spacetime);
        // the geometry / spacing / spacetime slugs all ride the name: a log-radial grid selects the
        // geometric-mean CFL-width map (`_logr`); uniform grids get sp = "" so the name is unchanged.
        let wname = format!(
            "{}_wave_speed_map{}{st}_{D}d",
            Self::kernel_prefix(), mhd_geom_suffix(geom.coords, &geom.axes)
        );
        // scalars BY NAME (the kernel's declared set drives it): eos param + the per-axis CFL
        // widths (cartesian `inv_dx_d`, curvilinear `x_lo_d`/`dx_d`); the mhd substrates run static,
        // so the motion rates bind 0. `kernel_geom` gives the log-aware per-axis scalars the in-kernel
        // `gv_axis_face_at` reads via `map_kind`; on a uniform static grid it is bit-identical to the
        // physical geometry.
        let (x_lo_phys, dx_phys) = kernel_geom(&geom.x_lo, &geom.dx, &geom.maps, geom.coords, sim.motion.a);
        let resolve_scalar = |bind: &ScalarBind| -> Sc {
            let ScalarBind::Ref(sref) = bind else {
                panic!("{} cfl: unexpected spec scalar {bind:?}", Self::kernel_prefix());
            };
            match *sref {
                ScalarRef::Gamma | ScalarRef::Cs => Sc::from_f64(self.eos_param),
                ScalarRef::SchwarzschildMass => Sc::from_f64(
                    geom.spacetime_scalars.iter()
                        .find(|(n, _)| n == "schwarzschild_mass")
                        .map(|(_, v)| *v)
                        .expect("GR MHD cfl needs schwarzschild_mass"),
                ),
                ScalarRef::KerrSpin => Sc::from_f64(
                    geom.spacetime_scalars.iter()
                        .find(|(n, _)| n == "kerr_spin")
                        .map(|(_, v)| *v)
                        .expect("GR MHD cfl needs kerr_spin"),
                ),
                other => Sc::from_f64(
                    motion_scalar(&sim.motion, geom.coords, D, other)
                        .or_else(|| geom_scalar(&x_lo_phys, &dx_phys, &sim.geom.maps, other))
                        .unwrap_or_else(|| panic!("{} cfl: unexpected scalar {other:?}", Self::kernel_prefix())),
                ),
            }
        };
        let scalars = scalars_for(&wname, &resolve_scalar);
        // bind BY MANIFEST: prim + bcell reads -> the `scratch` lambda write (the cfl_scratch
        // field, supplied as the scratch override). iso passes a dummy pre (reads cs^2*rho).
        let pre_bind = if R::SPEC.has_energy {
            sim.fields.prim.pre_field().expect("MHD energy regime requires prim.pre")
        } else {
            &sim.fields.cons.den
        };
        dispatch_named(sim, pre_bind, Some(&self.cfl_scratch), 0, &wname, &geom.interior, &[], &scalars);
        // the wu 2017 (arXiv:1708.07267) source-admissibility rate: on a curved background the
        // geometric source S advances U -> U + dt S and can push the conserved state out of the
        // physical-constraint cone; the light-cone LxF FOFC flux is only physical-constraint-
        // preserving under a timestep that also keeps U + dt S admissible. this kernel reads the
        // flux light-cone rate already in the scratch and adds the source characteristic rate
        // lambda_S = (|S_tau| + ||S_mom||_gamma)/q(U) in place, so the reduction sizes dt against
        // both (dt (lambda_flux + lambda_S) < 1). the source-CFL kernel is baked for GR-RMHD only.
        if !st.is_empty() {
            let sname = format!(
                "{}_source_cfl{}{st}_{D}d",
                Self::kernel_prefix(), mhd_geom_suffix(geom.coords, &geom.axes)
            );
            let sscalars = scalars_for(&sname, &resolve_scalar);
            dispatch_named(sim, pre_bind, Some(&self.cfl_scratch), 0, &sname, &geom.interior, &[], &sscalars);
        }
        let mut lambda_max = crate::regimes::substrate_gpu::field_max_reduce(&self.cfl_scratch, &geom.interior);
        // DRIVEN-INFLOW CFL CAP: the per-cell wave-speed map only scans the INTERIOR, so a driven
        // boundary's inflow state (which lives in the ghost band) is invisible to it — a relativistic
        // wind from a cold-ambient inner boundary would size dt off the slow interior and then get
        // pulled across the first inner face at a dt ~1e4x too large -> NaN. a relativistic signal
        // can travel at most c = 1, so bound dt by the light-crossing of the interior cell adjacent
        // to each driven face: lambda >= max_d (1 / (h_d * width_d)). only relativistic regimes get
        // the c = 1 bound; non-relativistic driven inflow is a future refinement.
        if R::SPEC.is_relativistic {
            lambda_max = lambda_max.max(driven_inflow_lambda(sim));
        }
        // GHOST-BAND FAIL-LOUD: a poisoned boundary leaves a non-finite ghost that FOFC never touches;
        // force the rate to +inf (dt -> 0, halt) if any zone in the allocated domain is non-finite.
        if !crate::regimes::substrate_kernels::state_finite_over_allocated(sim, pre_bind, &self.cfl_scratch) {
            lambda_max = f64::INFINITY;
        }
        cfl_from_lambda(lambda_max, self.cfl_number)
    }

    // ---- regime-agnostic tails: the gas godunov + the full CT stack (shared AOT kernels) ----
    fn godunov_stage(&self, sim: &FieldStore<D, 3, Mem, Sc>, dt: f64, a0: f64, ac: f64) {
        crate::regimes::mhd_substrate::godunov_stage(sim, R::SPEC.has_energy, Self::kernel_prefix(), self.eos_param, dt, a0, ac);
    }
    fn ghost_fill(&self, sim: &FieldStore<D, 3, Mem, Sc>) {
        crate::regimes::mhd_substrate::ghost_fill(sim, R::SPEC.has_energy);
        // docs/design/33: the standard fill skipped Driven faces (Driven -> Skip); prescribe
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
            || (self.ct_method == CtMethod::Uct && matches!(Self::kernel_prefix(), "nmhd" | "imhd"));
        let method = if self.ct_method == CtMethod::Uct && !materialized {
            CtMethod::Contact
        } else {
            self.ct_method
        };
        crate::regimes::mhd_substrate::efield(sim, method, self.solver, Self::kernel_prefix(), self.eos_param, self.theta);
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
