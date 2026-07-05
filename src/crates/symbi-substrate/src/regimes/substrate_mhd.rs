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
    dispatch_driven_boundaries, dispatch_fields_each, dispatch_named, dispatch_runtime_source, geom_scalar,
    kernel_field_binds, kernel_geom, mhd_flux_suffix, mhd_geom_suffix, motion_scalar, physical_geom, scalars_for,
    spacetime_slug, spacing_suffix, RegimeKind, RuntimeSource, ScalarBind, Solver,
};
use symbi_hydro::source_spec::BuiltSource;
use symbi_sim::substrate_seam::KernelSet;
use symbi_sim::state::{ConsFieldsGeneric, FieldStore, PrimFieldsGeneric};
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
    /// per-cell Riemann wave speeds materialized (RMHD only for now).
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
        Self { eos_param, cfl_number, theta, cfl_scratch, solver: Solver::Hlle, ct_method: CtMethod::Contact, runtime_source: None, boundary_dags: Vec::new(), _r: PhantomData }
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
            let sp = spacing_suffix(&sim.geom.maps);
            format!("{}_face_flux{solver}{gsfx}{sp}{st}_{D}d_{dir}", Self::kernel_prefix())
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
                geom_scalar(&x_lo_k, &dx_k, *other)
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

    /// FIRST-ORDER FLUX CORRECTION (see the trait doc). the flow: snapshot the high-order cons+prim
    /// into `u_fofc`/`prim_fofc`; restore `cons <- u_stage` and c2p so the redo reconstructs from the
    /// PHYSICAL stage-input state; re-flux at HLLE + theta = 0 -> re-godunov -> re-c2p (the first-order
    /// update, in place on cons/prim); then select `physical(prim_fofc) ? high_order : first_order`
    /// per cell. B / CT are untouched (the gas conserved is corrected, div(B) preserved). v1 runs
    /// unconditionally; the host-gate on a failure reduction is a perf overlay added on top.
    fn fofc_impl(&self, sim: &FieldStore<D, 3, Mem, Sc>, dt: f64, a0: f64, ac: f64) {
        // (1) snapshot the HIGH-ORDER cons+prim -> u_fofc/prim_fofc (whole domain, ghosts too, so the
        //     first-order reconstruction has valid ghost inputs).
        self.fofc_copy(sim, "snap", true,
            (&sim.fields.cons, &sim.fields.prim), (&sim.workspace.u_fofc, &sim.workspace.prim_fofc));
        // (2) restore cons <- u_stage (cons only): the redo reconstructs from the PHYSICAL stage input.
        self.fofc_copy(sim, "restore", false,
            (&sim.workspace.u_stage, &sim.fields.prim), (&sim.fields.cons, &sim.fields.prim));
        // (3) prim <- c2p(stage-input cons); (4) re-flux at first order (light-cone Rusanov, theta = 0
        //     -> admissibility-preserving); (5) re-godunov; (6) re-c2p.
        self.c2p(sim);
        for dir in 0..D {
            self.flux_impl(sim, dir, "", "_rusanov", 0.0);
        }
        self.godunov_stage(sim, dt, a0, ac);
        if self.has_additive_source() {
            self.source_apply(sim, ac * dt);
        }
        self.c2p(sim);
        // (7) select physical(prim_fofc high-order) ? high-order : first-order, over the interior.
        self.fofc_select(sim);
    }

    /// component field for a FOFC copy/select slot name (`den`/`mom_k`/`nrg`/`rho`/`vel_k`/`pre`).
    fn fofc_comp<'a>(
        cons: &'a ConsFieldsGeneric<D, 3, Mem, Sc>,
        prim: &'a PrimFieldsGeneric<D, 3, Mem, Sc>,
        name: &str,
    ) -> &'a Field<Sc, D, Mem> {
        match name {
            "den" => &cons.den,
            "nrg" => cons.nrg_field().expect("fofc: energy field"),
            "rho" => &prim.rho,
            "pre" => prim.pre_field().expect("fofc: pressure field"),
            s if s.starts_with("mom_") => &cons.mom[s[4..].parse::<usize>().unwrap()],
            s if s.starts_with("vel_") => &prim.vel[s[4..].parse::<usize>().unwrap()],
            o => panic!("fofc_comp: unknown component '{o}'"),
        }
    }

    /// dispatch the componentwise FOFC copy kernel `{prefix}_fofc_{tag}_{D}d` (src `s_*` -> dst `d_*`).
    fn fofc_copy(
        &self,
        sim: &FieldStore<D, 3, Mem, Sc>,
        tag: &str,
        include_prim: bool,
        src: (&ConsFieldsGeneric<D, 3, Mem, Sc>, &PrimFieldsGeneric<D, 3, Mem, Sc>),
        dst: (&ConsFieldsGeneric<D, 3, Mem, Sc>, &PrimFieldsGeneric<D, 3, Mem, Sc>),
    ) {
        let _ = include_prim;
        let name = format!("{}_fofc_{tag}_{D}d", Self::kernel_prefix());
        let slot = |s: &str| -> &Field<Sc, D, Mem> {
            let comp = &s[2..]; // strip "s_" / "d_"
            if s.starts_with("s_") {
                Self::fofc_comp(src.0, src.1, comp)
            } else {
                Self::fofc_comp(dst.0, dst.1, comp)
            }
        };
        let mut inputs: Vec<&Field<Sc, D, Mem>> = Vec::new();
        let mut outputs: Vec<&Field<Sc, D, Mem>> = Vec::new();
        for (bind, is_out) in kernel_field_binds(&name).iter() {
            let fld = slot(&bind.name());
            if *is_out { outputs.push(fld); } else { inputs.push(fld); }
        }
        dispatch_fields_each::<Sc, Mem, D>(&name, &sim.geom.allocated, &inputs, &outputs, &[], &[]);
    }

    /// dispatch the FOFC select kernel `{prefix}_fofc_select_{D}d`: `physical(ho_prim) ? ho : fo`, with
    /// `ho_*` = the high-order snapshot (u_fofc/prim_fofc), `fo_*` = out_* = the live cons/prim (the
    /// first-order redo, corrected in place), over the interior.
    fn fofc_select(&self, sim: &FieldStore<D, 3, Mem, Sc>) {
        let name = format!("{}_fofc_select_{D}d", Self::kernel_prefix());
        let (u_fofc, prim_fofc) = (&sim.workspace.u_fofc, &sim.workspace.prim_fofc);
        let (cons, prim) = (&sim.fields.cons, &sim.fields.prim);
        let slot = |s: &str| -> &Field<Sc, D, Mem> {
            if let Some(c) = s.strip_prefix("ho_") {
                Self::fofc_comp(u_fofc, prim_fofc, c)
            } else if let Some(c) = s.strip_prefix("x_") {
                // the in-place cons/prim: read (first-order) + write (select result), one binding.
                Self::fofc_comp(cons, prim, c)
            } else {
                panic!("fofc_select: unknown slot '{s}'")
            }
        };
        let mut inputs: Vec<&Field<Sc, D, Mem>> = Vec::new();
        let mut outputs: Vec<&Field<Sc, D, Mem>> = Vec::new();
        for (bind, is_out) in kernel_field_binds(&name).iter() {
            let fld = slot(&bind.name());
            if *is_out { outputs.push(fld); } else { inputs.push(fld); }
        }
        dispatch_fields_each::<Sc, Mem, D>(&name, &sim.geom.interior, &inputs, &outputs, &[], &[]);
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

    fn fofc(&self, sim: &FieldStore<D, 3, Mem, Sc>, dt: f64, a0: f64, ac: f64) {
        self.fofc_impl(sim, dt, a0, ac);
    }

    fn fofc_active(&self) -> bool {
        true
    }

    fn c2p(&self, sim: &FieldStore<D, 3, Mem, Sc>) {
        let st = spacetime_slug(sim.geom.spacetime);
        let cname = if st.is_empty() {
            format!("{}_c2p_{D}d", Self::kernel_prefix())
        } else {
            // the metric-aware KKC recovery: gamma at the volume-weighted centroid, so the name
            // carries the chart + spacing + spacetime slugs and the kernel reads mass + grid scalars.
            let gsfx = mhd_geom_suffix(sim.geom.coords, &sim.geom.axes);
            let sp = spacing_suffix(&sim.geom.maps);
            format!("{}_c2p{gsfx}{sp}{st}_{D}d", Self::kernel_prefix())
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
                geom_scalar(&x_lo_k, &dx_k, *other)
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
            let sp = spacing_suffix(&sim.geom.maps);
            let gsfx = mhd_geom_suffix(sim.geom.coords, &sim.geom.axes);
            let wsname = format!("{}_wave_speeds_cell{gsfx}{sp}{st}_{D}d", Self::kernel_prefix());
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
                    geom_scalar(&x_lo_k, &dx_k, *other)
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
        let wname = if st.is_empty() {
            format!("{}_wave_speed_map{}_{D}d", Self::kernel_prefix(), mhd_geom_suffix(geom.coords, &geom.axes))
        } else {
            // curved background: the coordinate light-cone bound (state-independent).
            let sp = spacing_suffix(&geom.maps);
            format!("{}_wave_speed_map{}{sp}{st}_{D}d", Self::kernel_prefix(), mhd_geom_suffix(geom.coords, &geom.axes))
        };
        // scalars BY NAME (the kernel's declared set drives it): eos param + the per-axis CFL
        // widths (cartesian `inv_dx_d`, curvilinear `x_lo_d`/`dx_d`); the mhd substrates run
        // static, so the motion rates bind 0.
        // the GR light-cone map builds positions through gv_axis_face_at, so it takes the
        // LOG-AWARE kernel scalars; the flat maps keep the physical geometry (identical on a
        // uniform static mesh).
        let (x_lo_phys, dx_phys) = if st.is_empty() {
            physical_geom(&geom.x_lo, &geom.dx, geom.coords, sim.motion.a)
        } else {
            kernel_geom(&geom.x_lo, &geom.dx, &geom.maps, geom.coords, sim.motion.a)
        };
        let scalars = scalars_for(&wname, |bind| {
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
