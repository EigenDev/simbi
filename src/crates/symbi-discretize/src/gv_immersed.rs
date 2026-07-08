// =============================================================================
// gv_immersed.rs
//
// the immersed-boundary body source terms traced at S = Gv (docs/design/19):
//   - body_source_gv  (FORWARD, bodies -> fluid): softened gravity + Bondi-Hoyle accretion,
//     cons -> cons in-place (`cons += dt * S`).
//   - body_feedback_gv (BACKWARD, fluid -> bodies): per-cell per-body force / torque / accreted
//     mass -> scratch fields a device reduction sums into each body's BodyDelta.
//
// GENERIC over coordinate system (docs/design/19): the physics is done in CARTESIAN
// (coord-free) — `cell_scaffold` supplies the cell's cartesian position + gas velocity via the
// gv to_cartesian / vector_to_cartesian transforms; the forward source PROJECTS gravity + sink
// velocity onto the physical momentum frame (vector_from_cartesian), the feedback keeps the
// cartesian force/torque (matching symbi_ib's BodyDelta). the physics is a function of `cons`
// ALONE (no c2p); the 0..n_bodies loop is BRANCH-FREE (inactive body: mass=0 / sink_rate=0).
//
// kept OUT of the (already large) gv.rs: this rides the PUBLIC gv API only — Gv arithmetic +
// the transcendentals (exp) + `cell_geometry_gv` + begin/end_trace.
// =============================================================================

use symbi_ir::graph::NodeId;

use symbi_ir::algebra::Scalar;
use symbi_ir::{FieldBind, FieldRef};
use symbi_algebra::algebra::Numeric;

use super::coords::{Coords, Spacing};
use super::gv::{cell_geometry_gv, CellGeometryGv};
use symbi_ir::{begin_trace, end_trace, Gv, GvKernel};

use symbi_hydro::regime_spec::law_params;
use symbi_hydro::source_spec::{lift_to_built, source_params, BuiltSource};
use symbi_hydro::source_term::BodySource;

type Writes = Vec<(String, FieldBind, NodeId)>;

#[inline]
fn sq(a: Gv) -> Gv {
    a * a
}
#[inline]
fn cube(a: Gv) -> Gv {
    a * a * a
}

/// per-cell, per-body physics in CARTESIAN (coord-free): softened gravity `g`, accretion rate
/// `den_dot`, sink velocity `vstar`, and `rvec = cell - body` (all 3-vectors).
struct BodyContributionGv {
    g: [Gv; 3],
    den_dot: Gv,
    vstar: [Gv; 3],
    rvec: [Gv; 3],
}

/// the CARTESIAN axes (0=x,1=y,2=z) a body's ndim-D position/velocity components map to — the
/// grid-plane convention (docs/design/19). identical to `immersed::body_cart_axes`.
fn body_cart_axes(coords: Coords, ndim: usize, axes: &[usize]) -> Vec<usize> {
    match coords {
        Coords::Cartesian => (0..ndim).collect(),
        Coords::Cylindrical => axes.to_vec(),
        Coords::Spherical => {
            if ndim >= 3 {
                vec![0, 1, 2]
            } else {
                vec![0, 2]
            }
        }
    }
}

// ---- gv coordinate transforms (the metric's to_cartesian / orthonormal frame) --------------
// only the immersed source uses these; co-located with their consumer.

/// the cell cartesian position from its coordinate-basis position (3D embedding).
fn to_cartesian_gv(coords: Coords, coord: &[Gv; 3]) -> [Gv; 3] {
    match coords {
        Coords::Cartesian => *coord,
        Coords::Cylindrical => {
            let (r, phi) = (coord[0], coord[1]);
            [r * phi.cos(), r * phi.sin(), coord[2]]
        }
        Coords::Spherical => {
            let (r, th, phi) = (coord[0], coord[1], coord[2]);
            let rst = r * th.sin();
            [rst * phi.cos(), rst * phi.sin(), r * th.cos()]
        }
    }
}

/// the orthonormal (unit) basis vector e_comp(coord) in cartesian, for coordinate index `comp`.
fn basis_vec_gv(coords: Coords, coord: &[Gv; 3], comp: usize) -> [Gv; 3] {
    let (z, o) = (Gv::ZERO, Gv::ONE);
    match (coords, comp) {
        (Coords::Cartesian, 0) => [o, z, z],
        (Coords::Cartesian, 1) => [z, o, z],
        (Coords::Cartesian, _) => [z, z, o],
        (Coords::Cylindrical, 0) => [coord[1].cos(), coord[1].sin(), z], // r_hat
        (Coords::Cylindrical, 1) => [-coord[1].sin(), coord[1].cos(), z], // phi_hat
        (Coords::Cylindrical, _) => [z, z, o],                           // z_hat
        (Coords::Spherical, 0) => {
            let (st, ct, cp, sp) = (coord[1].sin(), coord[1].cos(), coord[2].cos(), coord[2].sin());
            [st * cp, st * sp, ct] // r_hat
        }
        (Coords::Spherical, 1) => {
            let (st, ct, cp, sp) = (coord[1].sin(), coord[1].cos(), coord[2].cos(), coord[2].sin());
            [ct * cp, ct * sp, -st] // theta_hat
        }
        (Coords::Spherical, _) => [-coord[2].sin(), coord[2].cos(), z], // phi_hat
    }
}

/// project a cartesian vector `w` onto the orthonormal coordinate frame: the `ncomp` physical
/// components `w . e_comp`.
fn vector_from_cartesian_gv(coords: Coords, coord: &[Gv; 3], w: &[Gv; 3], ncomp: usize) -> Vec<Gv> {
    (0..ncomp)
        .map(|comp| {
            let e = basis_vec_gv(coords, coord, comp);
            w[0] * e[0] + w[1] * e[1] + w[2] * e[2]
        })
        .collect()
}

/// expand physical components `v` (coordinate frame) back to a cartesian vector `sum v[c] e_c`.
fn vector_to_cartesian_gv(coords: Coords, coord: &[Gv; 3], v: &[Gv]) -> [Gv; 3] {
    let mut acc = [Gv::ZERO; 3];
    for (comp, &vc) in v.iter().enumerate() {
        let e = basis_vec_gv(coords, coord, comp);
        for ax in 0..3 {
            acc[ax] = acc[ax] + vc * e[ax];
        }
    }
    acc
}

// ---- gas state + per-cell scaffolding + per-body contribution -------------------------------

// a body's 3D cartesian `pos`/`vel`: place its ndim components at `cart_axes`, 0 elsewhere.
fn body_vec3(b: usize, ndim: usize, cart_axes: &[usize], name: &str) -> [Gv; 3] {
    let mut v = [Gv::ZERO; 3];
    for g in 0..ndim {
        v[cart_axes[g]] = Gv::scalar(&format!("body_{b}_{name}_{g}"));
    }
    v
}

// gas primitives from cons: (vel_physical[ncomp], cs, e_int).
fn gas_state(ncomp: usize, gamma: Gv, den: Gv, mom: &[Gv], nrg: Gv) -> (Vec<Gv>, Gv, Gv) {
    let inv_den = Gv::ONE / den;
    let vel: Vec<Gv> = (0..ncomp).map(|comp| mom[comp] * inv_den).collect();
    let ke = Gv::from_f64(0.5) * (0..ncomp).map(|comp| mom[comp] * vel[comp]).sum::<Gv>();
    let nrg_minus_ke = nrg - ke;
    let p = (gamma - Gv::ONE) * nrg_minus_ke;
    let cs = (gamma * p / den).sqrt();
    let e_int = nrg_minus_ke / den;
    (vel, cs, e_int)
}

/// the per-cell scaffolding both kernels open with: the cell's coordinate position `coord3` (3D,
/// gridded coords from the centroid via the axis-role map, ungridded at symmetry 0), the cell
/// CARTESIAN position, the gas velocity in CARTESIAN, min width, cs, e_int.
#[allow(clippy::too_many_arguments)]
fn cell_scaffold(
    coords: Coords,
    ndim: usize,
    ncomp: usize,
    axes: &[usize],
    gamma: Gv,
    den: Gv,
    mom: &[Gv],
    nrg: Gv,
) -> ([Gv; 3], [Gv; 3], [Gv; 3], Gv, Gv, Gv) {
    let geo = cell_geometry_gv(coords, &vec![Spacing::Uniform; ndim], axes, ndim);
    // coord3 in NATURAL coordinate order: gridded coords from the centroid, ungridded = 0.
    let mut coord3 = [Gv::ZERO; 3];
    for (g, &coord_idx) in axes.iter().enumerate() {
        if coord_idx < 3 {
            coord3[coord_idx] = geo.centroid[g];
        }
    }
    let cell_cart = to_cartesian_gv(coords, &coord3);
    let (vel_phys, cs, e_int) = gas_state(ncomp, gamma, den, mom, nrg);
    let vel_cart = vector_to_cartesian_gv(coords, &coord3, &vel_phys);

    let mut min_w = Gv::scalar("dx_0");
    for ax in 1..ndim {
        min_w = min_w.min(Gv::scalar(&format!("dx_{ax}")));
    }
    (coord3, cell_cart, vel_cart, min_w, cs, e_int)
}

#[allow(clippy::too_many_arguments)]
fn body_contribution(
    b: usize,
    ndim: usize,
    cart_axes: &[usize],
    cell_cart: &[Gv; 3],
    vel_cart: &[Gv; 3],
    den: Gv,
    cs: Gv,
    min_w: Gv,
    inv_dt: Gv,
) -> BodyContributionGv {
    let (tiny, eps_r) = (Gv::from_f64(1e-30), Gv::from_f64(1e-24));
    let mass = Gv::scalar(&format!("body_{b}_mass"));
    let soft = Gv::scalar(&format!("body_{b}_soft"));
    let bpos = body_vec3(b, ndim, cart_axes, "pos");

    // r_vec = cell_cart - body_pos; r_dist2 = |r_vec|^2; r_eff = sqrt(.. + soft^2).
    let rvec: [Gv; 3] = std::array::from_fn(|i| cell_cart[i] - bpos[i]);
    let r_dist2 = sq(rvec[0]) + sq(rvec[1]) + sq(rvec[2]);
    let r_eff = (r_dist2 + sq(soft)).sqrt();
    let r_mag = r_dist2.sqrt();

    // gravity (cartesian): g = -mass * r_vec / r_eff^3.
    let grav_fac = -mass / cube(r_eff);
    let g: [Gv; 3] = std::array::from_fn(|i| rvec[i] * grav_fac);

    // accretion rate (scalar): den_dot = den*min(sink, 1/t_nat, 1/dt)*weight.
    let r_acc = Gv::scalar(&format!("body_{b}_racc"));
    let sink_rate = Gv::scalar(&format!("body_{b}_sink"));
    let delta = Gv::scalar(&format!("body_{b}_delta"));
    let r_norm = r_mag / (Gv::from_f64(0.5) * r_acc);
    let weight = (-sq(r_norm)).exp();
    let sound_crossing = min_w / cs;
    let t_ff = (cube(r_mag) / (Gv::from_f64(2.0) * mass + tiny)).sqrt();
    let nat_rate = Gv::ONE / (sound_crossing.min(t_ff) + tiny);
    let sr_base = sink_rate.min(nat_rate).min(inv_dt);
    let den_dot = den * sr_base * weight;

    // sink velocity (cartesian): v_star = v_rad + delta*v_ang + v_body, v_rel = v_gas - v_body.
    let bvel = body_vec3(b, ndim, cart_axes, "vel");
    let inv_safe_r = Gv::ONE / (r_dist2 + eps_r).sqrt();
    let rhat: [Gv; 3] = std::array::from_fn(|i| rvec[i] * inv_safe_r);
    let vrel: [Gv; 3] = std::array::from_fn(|i| vel_cart[i] - bvel[i]);
    let vrad_comp = vrel[0] * rhat[0] + vrel[1] * rhat[1] + vrel[2] * rhat[2];
    let vstar: [Gv; 3] = std::array::from_fn(|i| {
        let vrad = vrad_comp * rhat[i];
        let vang = vrel[i] - vrad;
        vrad + delta * vang + bvel[i]
    });

    BodyContributionGv { g, den_dot, vstar, rvec }
}

/// FORWARD source: `cons += dt * (S_grav + S_accretion)`, generic over coordinate system.
/// reads cons (den/mom/nrg) in place; declares dt/gamma + per-axis grid scalars + the MAX_BODIES
/// body params (resolved by name at dispatch). returns the in-place conserved writes.
/// per-cell body-evolved ADIABATIC conserved state: `(den, mom, nrg)` -> the state after `dt *` the
/// forward immersed-body source (gravity + accretion sink) over all `n_bodies` slots. a PURE
/// function of the cell state in registers (NO field reads / writes), so it composes into ANY kernel
/// that already holds the conserved state — the standalone body pass AND the FOFC freeze parachute
/// — with no materialized buffer. declares `dt` / `gamma` + the per-body scalars via
/// `body_contribution`; unused body slots contribute zero.
pub(crate) fn body_evolved_gv(
    den: Gv,
    mom: &[Gv],
    nrg: Gv,
    dt: Gv,
    gamma: Gv,
    n_bodies: usize,
    coords: Coords,
    ndim: usize,
    ncomp: usize,
    axes: &[usize],
) -> (Gv, Vec<Gv>, Gv) {
    let inv_dt = Gv::ONE / dt;
    let cart_axes = body_cart_axes(coords, ndim, axes);
    let (coord3, cell_cart, vel_cart, min_w, cs, e_int) =
        cell_scaffold(coords, ndim, ncomp, axes, gamma, den, mom, nrg);

    let mut d_den = Gv::ZERO;
    let mut d_mom: Vec<Gv> = vec![Gv::ZERO; ncomp];
    let mut d_nrg = Gv::ZERO;
    for b in 0..n_bodies {
        let bc = body_contribution(b, ndim, &cart_axes, &cell_cart, &vel_cart, den, cs, min_w, inv_dt);
        // project cartesian gravity + sink velocity onto the physical momentum components.
        let g_phys = vector_from_cartesian_gv(coords, &coord3, &bc.g, ncomp);
        let vstar_phys = vector_from_cartesian_gv(coords, &coord3, &bc.vstar, ncomp);
        d_den = d_den - bc.den_dot;
        let mut vstar2 = Gv::ZERO;
        for comp in 0..ncomp {
            d_mom[comp] = d_mom[comp] + den * g_phys[comp]; // gravity
            d_nrg = d_nrg + mom[comp] * g_phys[comp];
            d_mom[comp] = d_mom[comp] - vstar_phys[comp] * bc.den_dot; // accreted momentum
            vstar2 = vstar2 + sq(vstar_phys[comp]);
        }
        let nrg_dot = Gv::from_f64(0.5) * vstar2 * bc.den_dot + e_int * bc.den_dot;
        d_nrg = d_nrg - nrg_dot;
    }

    let den_new = den + dt * d_den;
    let mom_new: Vec<Gv> = (0..ncomp).map(|comp| mom[comp] + dt * d_mom[comp]).collect();
    let nrg_new = nrg + dt * d_nrg;
    (den_new, mom_new, nrg_new)
}

pub fn body_source_gv(
    n_bodies: usize,
    coords: Coords,
    ndim: usize,
    ncomp: usize,
    axes: &[usize],
) -> (GvKernel, Writes) {
    begin_trace();
    let dt = Gv::scalar("dt");
    let gamma = Gv::scalar("gamma");
    let den = Gv::field("den", FieldRef::cons_den());
    let mom: Vec<Gv> = (0..ncomp)
        .map(|comp| Gv::field(&format!("mom_{comp}"), FieldRef::cons_mom(comp as u8)))
        .collect();
    let nrg = Gv::field("nrg", FieldRef::cons_nrg());
    let (den_new, mom_new, nrg_new) =
        body_evolved_gv(den, &mom, nrg, dt, gamma, n_bodies, coords, ndim, ncomp, axes);

    let mut writes = vec![("den_new".to_string(), FieldRef::cons_den().into(), den_new.node())];
    for (comp, m) in mom_new.iter().enumerate() {
        writes.push((format!("mom_{comp}_new"), FieldRef::cons_mom(comp as u8).into(), m.node()));
    }
    writes.push(("nrg_new".to_string(), FieldRef::cons_nrg().into(), nrg_new.node()));
    (end_trace(), writes)
}

/// BACKWARD feedback: per cell, per body, the CARTESIAN force / 3D torque / accreted mass each
/// body receives -> the MAX_BODIES*(ndim+4) reduction-scratch writes (`fb_{b}_force_{ax}` /
/// `fb_{b}_torque_{t}` / `fb_{b}_mass`, the order the runtime sums). generic over coord system.
pub fn body_feedback_gv(
    n_bodies: usize,
    coords: Coords,
    ndim: usize,
    ncomp: usize,
    axes: &[usize],
) -> (GvKernel, Writes) {
    begin_trace();
    let dt = Gv::scalar("dt");
    let gamma = Gv::scalar("gamma");
    let inv_dt = Gv::ONE / dt;
    let cart_axes = body_cart_axes(coords, ndim, axes);
    let den = Gv::field("den", FieldRef::cons_den());
    let mom: Vec<Gv> = (0..ncomp)
        .map(|comp| Gv::field(&format!("mom_{comp}"), FieldRef::cons_mom(comp as u8)))
        .collect();
    let nrg = Gv::field("nrg", FieldRef::cons_nrg());
    let (_coord3, cell_cart, vel_cart, min_w, cs, _e_int) =
        cell_scaffold(coords, ndim, ncomp, axes, gamma, den, &mom, nrg);

    // cell volume dv = 1 / inv_volume (cell_geometry recomputed here — CSE collapses it).
    let geo: CellGeometryGv = cell_geometry_gv(coords, &vec![Spacing::Uniform; ndim], axes, ndim);
    let dv = Gv::ONE / geo.inv_volume;
    let dv_dt = dv * dt;

    let mut writes: Writes = Vec::new();
    for b in 0..n_bodies {
        let bc = body_contribution(b, ndim, &cart_axes, &cell_cart, &vel_cart, den, cs, min_w, inv_dt);
        // body force (cartesian, 3D) = -(den*g + v_star*den_dot)*dv. accretion force `fa` alone
        // carries the torque (gravity is central -> no torque about the body).
        let mut force_cart = [Gv::ZERO; 3];
        let mut fa = [Gv::ZERO; 3];
        for i in 0..3 {
            let vd = bc.vstar[i] * bc.den_dot;
            force_cart[i] = -(den * bc.g[i] + vd) * dv;
            fa[i] = -vd * dv;
        }
        // the body's ndim force components live at its cartesian axes.
        for ax in 0..ndim {
            let fc = force_cart[cart_axes[ax]];
            writes.push((format!("b{b}_f{ax}"), format!("fb_{b}_force_{ax}").into(), fc.node()));
        }
        // torque = r x F_accretion (full 3D cross; 2D yields (0,0,tz) automatically).
        let cross = |a: usize, bb: usize| bc.rvec[a] * fa[bb] - bc.rvec[bb] * fa[a];
        for (t, tc) in [cross(1, 2), cross(2, 0), cross(0, 1)].into_iter().enumerate() {
            writes.push((format!("b{b}_t{t}"), format!("fb_{b}_torque_{t}").into(), tc.node()));
        }
        writes.push((format!("b{b}_m"), format!("fb_{b}_mass").into(), (bc.den_dot * dv_dt).node()));
    }
    (end_trace(), writes)
}

// =============================================================================
// isothermal variants (docs/design/19; no energy equation)
//
// the immersed-body PHYSICS is EOS-independent — softened gravity + Bondi-Hoyle
// accretion are functions of (den, mom, cs) only via the SHARED `body_contribution`.
// the single difference from the adiabatic kernels is the closure for `cs` and
// the absence of an energy update:
//   - adiabatic: cs / e_int are recovered from `cons.nrg` (`gas_state`), and the
//     source updates `cons.nrg` (gravity work + accreted internal+kinetic energy).
//   - isothermal: there is no `cons.nrg`. cs comes from `prim.pre` (= cs^2(x)*rho,
//     the substrate's iso pressure encoding — EXACTLY what the iso FLUX reads), so
//     a locally-isothermal cs^2(x) flows through identically; no energy is updated.
// future media (porous / deformable) extend `body_contribution`, so BOTH the
// adiabatic and isothermal kernels inherit them with no further duplication.
// =============================================================================

// iso gas state from cons + the substrate pressure: (vel_physical[ncomp], cs).
// `pre = cs^2(x)*rho` so `cs = sqrt(pre/rho)`. no internal energy (isothermal).
fn gas_state_iso(ncomp: usize, den: Gv, mom: &[Gv], pre: Gv) -> (Vec<Gv>, Gv) {
    let inv_den = Gv::ONE / den;
    let vel: Vec<Gv> = (0..ncomp).map(|comp| mom[comp] * inv_den).collect();
    let cs = (pre * inv_den).sqrt();
    (vel, cs)
}

/// the iso counterpart of `cell_scaffold`: identical geometry + velocity scaffold,
/// but cs comes from `prim.pre` and there is no e_int.
#[allow(clippy::too_many_arguments)]
fn cell_scaffold_iso(
    coords: Coords,
    ndim: usize,
    ncomp: usize,
    axes: &[usize],
    den: Gv,
    mom: &[Gv],
    pre: Gv,
) -> ([Gv; 3], [Gv; 3], [Gv; 3], Gv, Gv) {
    let geo = cell_geometry_gv(coords, &vec![Spacing::Uniform; ndim], axes, ndim);
    let mut coord3 = [Gv::ZERO; 3];
    for (g, &coord_idx) in axes.iter().enumerate() {
        if coord_idx < 3 {
            coord3[coord_idx] = geo.centroid[g];
        }
    }
    let cell_cart = to_cartesian_gv(coords, &coord3);
    let (vel_phys, cs) = gas_state_iso(ncomp, den, mom, pre);
    let vel_cart = vector_to_cartesian_gv(coords, &coord3, &vel_phys);
    let mut min_w = Gv::scalar("dx_0");
    for ax in 1..ndim {
        min_w = min_w.min(Gv::scalar(&format!("dx_{ax}")));
    }
    (coord3, cell_cart, vel_cart, min_w, cs)
}

/// FORWARD iso source: `cons += dt * (S_grav + S_accretion)` with NO energy update.
/// reads cons (den/mom) + prim.pre; writes den/mom (no nrg). gravity + accretion via
/// the SHARED `body_contribution`.
pub fn body_source_iso_gv(
    n_bodies: usize,
    coords: Coords,
    ndim: usize,
    ncomp: usize,
    axes: &[usize],
) -> (GvKernel, Writes) {
    begin_trace();
    let dt = Gv::scalar("dt");
    let den = Gv::field("den", FieldRef::cons_den());
    let mom: Vec<Gv> = (0..ncomp)
        .map(|comp| Gv::field(&format!("mom_{comp}"), FieldRef::cons_mom(comp as u8)))
        .collect();
    let pre = Gv::field("pre", FieldRef::PrimPre);
    let (den_new, mom_new) =
        body_evolved_iso_gv(den, &mom, pre, dt, n_bodies, coords, ndim, ncomp, axes);

    let mut writes = vec![("den_new".to_string(), FieldRef::cons_den().into(), den_new.node())];
    for (comp, m) in mom_new.iter().enumerate() {
        writes.push((format!("mom_{comp}_new"), FieldRef::cons_mom(comp as u8).into(), m.node()));
    }
    (end_trace(), writes)
}

/// the isothermal immersed-body evolution as a PURE per-cell function (no field reads): the
/// energy-free twin of `body_evolved_gv`. given the cell's conserved (den, mom) and its isothermal
/// pressure `pre` (which sets the sound speed), returns the state advanced by `dt` of softened
/// Newtonian gravity + Bondi-Hoyle accretion from `n_bodies` point masses. shared by the standalone
/// iso body source and the FOFC freeze-select-with-body composition.
pub(crate) fn body_evolved_iso_gv(
    den: Gv,
    mom: &[Gv],
    pre: Gv,
    dt: Gv,
    n_bodies: usize,
    coords: Coords,
    ndim: usize,
    ncomp: usize,
    axes: &[usize],
) -> (Gv, Vec<Gv>) {
    let inv_dt = Gv::ONE / dt;
    let cart_axes = body_cart_axes(coords, ndim, axes);
    let (coord3, cell_cart, vel_cart, min_w, cs) =
        cell_scaffold_iso(coords, ndim, ncomp, axes, den, mom, pre);

    let mut d_den = Gv::ZERO;
    let mut d_mom: Vec<Gv> = vec![Gv::ZERO; ncomp];
    for b in 0..n_bodies {
        let bc = body_contribution(b, ndim, &cart_axes, &cell_cart, &vel_cart, den, cs, min_w, inv_dt);
        let g_phys = vector_from_cartesian_gv(coords, &coord3, &bc.g, ncomp);
        let vstar_phys = vector_from_cartesian_gv(coords, &coord3, &bc.vstar, ncomp);
        d_den = d_den - bc.den_dot;
        for comp in 0..ncomp {
            d_mom[comp] = d_mom[comp] + den * g_phys[comp]; // gravity
            d_mom[comp] = d_mom[comp] - vstar_phys[comp] * bc.den_dot; // accreted momentum
        }
    }

    let den_new = den + dt * d_den;
    let mom_new: Vec<Gv> = (0..ncomp).map(|comp| mom[comp] + dt * d_mom[comp]).collect();
    (den_new, mom_new)
}

/// BACKWARD iso feedback: identical force/torque/mass writes as the adiabatic
/// kernel; cs comes from prim.pre rather than nrg. 2D+ only.
pub fn body_feedback_iso_gv(
    n_bodies: usize,
    coords: Coords,
    ndim: usize,
    ncomp: usize,
    axes: &[usize],
) -> (GvKernel, Writes) {
    begin_trace();
    let dt = Gv::scalar("dt");
    let inv_dt = Gv::ONE / dt;
    let cart_axes = body_cart_axes(coords, ndim, axes);
    let den = Gv::field("den", FieldRef::cons_den());
    let mom: Vec<Gv> = (0..ncomp)
        .map(|comp| Gv::field(&format!("mom_{comp}"), FieldRef::cons_mom(comp as u8)))
        .collect();
    let pre = Gv::field("pre", FieldRef::PrimPre);
    let (_coord3, cell_cart, vel_cart, min_w, cs) =
        cell_scaffold_iso(coords, ndim, ncomp, axes, den, &mom, pre);

    let geo: CellGeometryGv = cell_geometry_gv(coords, &vec![Spacing::Uniform; ndim], axes, ndim);
    let dv = Gv::ONE / geo.inv_volume;
    let dv_dt = dv * dt;

    let mut writes: Writes = Vec::new();
    for b in 0..n_bodies {
        let bc = body_contribution(b, ndim, &cart_axes, &cell_cart, &vel_cart, den, cs, min_w, inv_dt);
        let mut force_cart = [Gv::ZERO; 3];
        let mut fa = [Gv::ZERO; 3];
        for i in 0..3 {
            let vd = bc.vstar[i] * bc.den_dot;
            force_cart[i] = -(den * bc.g[i] + vd) * dv;
            fa[i] = -vd * dv;
        }
        for ax in 0..ndim {
            let fc = force_cart[cart_axes[ax]];
            writes.push((format!("b{b}_f{ax}"), format!("fb_{b}_force_{ax}").into(), fc.node()));
        }
        let cross = |a: usize, bb: usize| bc.rvec[a] * fa[bb] - bc.rvec[bb] * fa[a];
        for (t, tc) in [cross(1, 2), cross(2, 0), cross(0, 1)].into_iter().enumerate() {
            writes.push((format!("b{b}_t{t}"), format!("fb_{b}_torque_{t}").into(), tc.node()));
        }
        writes.push((format!("b{b}_m"), format!("fb_{b}_mass").into(), (bc.den_dot * dv_dt).node()));
    }
    (end_trace(), writes)
}

// =============================================================================
// fused additive body source (the frame-correct twin of `body_source_gv`)
//
// re-expresses the body source as `BuiltSource`s so it FUSES into the godunov
// stage (one launch, no separate pass) via `godunov_stage_gv_with_fused_built`.
// frame-correct for any geometry: the cell coord is lifted to Cartesian, the
// shared `BodySource` carrier does the physics, and the Cartesian momentum source
// is projected onto the physical coordinate basis. additive convention: reads the
// SSP-stage prim (rho/vel/pre), applied with the `ac*dt` weight by the composer.
// =============================================================================

/// declare the shared per-cell leaves (INSIDE `lift_to_built`'s trace): gas state
/// (rho, vel in CARTESIAN, cs), the cell CARTESIAN position, and the grid (min
/// width, 1/dt). `has_energy` picks the cs closure: adiabatic = sqrt(gamma*pre/rho),
/// iso = the constant `cs` scalar (exact for globally-isothermal; the cs only enters
/// the accretion rate cap).
fn body_scaffold(
    coords: Coords,
    ndim: usize,
    ncomp: usize,
    axes: &[usize],
    has_energy: bool,
) -> (Gv, [Gv; 3], [Gv; 3], Gv, Gv, Gv, [Gv; 3]) {
    let rho = Gv::scalar(law_params::RHO);
    let vel_phys: Vec<Gv> = (0..ncomp).map(|k| Gv::scalar(&law_params::vel(k))).collect();
    let mut coord3 = [Gv::ZERO; 3];
    for (g, &coord_idx) in axes.iter().enumerate() {
        if coord_idx < 3 {
            coord3[coord_idx] = Gv::scalar(&source_params::x(g));
        }
    }
    let x_cart = to_cartesian_gv(coords, &coord3);
    let vel_cart = vector_to_cartesian_gv(coords, &coord3, &vel_phys);
    let cs = if has_energy {
        let gamma = Gv::scalar("gamma");
        let pre = Gv::scalar(law_params::PRE);
        (gamma * pre / rho).sqrt()
    } else {
        Gv::scalar("cs")
    };
    let mut min_w = Gv::scalar("dx_0");
    for g in 1..ndim {
        min_w = min_w.min(Gv::scalar(&format!("dx_{g}")));
    }
    let inv_dt = Gv::ONE / Gv::scalar("dt");
    (rho, vel_cart, x_cart, cs, min_w, inv_dt, coord3)
}

/// the b-th body's carrier (Cartesian Gv leaves) — params resolved by name from
/// the immersed side-car at dispatch. `body_vec3` places the ndim pos/vel
/// components on their Cartesian axes (the grid-plane convention).
fn body_at(b: usize, ndim: usize, cart_axes: &[usize]) -> BodySource<Gv> {
    BodySource {
        mass: Gv::scalar(&format!("body_{b}_mass")),
        xm: body_vec3(b, ndim, cart_axes, "pos"),
        vm: body_vec3(b, ndim, cart_axes, "vel"),
        soft: Gv::scalar(&format!("body_{b}_soft")),
        racc: Gv::scalar(&format!("body_{b}_racc")),
        sink: Gv::scalar(&format!("body_{b}_sink")),
        delta: Gv::scalar(&format!("body_{b}_delta")),
    }
}

/// the immersed-body source as FUSED-source `BuiltSource`s (gravity + Bondi
/// accretion). returns (target_field, BuiltSource): "mom" (ncomp), "den" (1), and
/// for energy-bearing regimes "nrg" (1).
pub fn body_source_built(
    coords: Coords,
    ndim: usize,
    ncomp: usize,
    axes: &[usize],
    n_bodies: usize,
    has_energy: bool,
) -> Vec<(String, BuiltSource)> {
    let axes = axes.to_vec();
    let cart_axes = body_cart_axes(coords, ndim, &axes);

    let mom = {
        let (axes, cart_axes) = (axes.clone(), cart_axes.clone());
        lift_to_built(move || {
            let (rho, vel_cart, x_cart, cs, min_w, inv_dt, coord3) =
                body_scaffold(coords, ndim, ncomp, &axes, has_energy);
            let mut s_cart = [Gv::ZERO; 3];
            for b in 0..n_bodies {
                let sm = body_at(b, ndim, &cart_axes)
                    .momentum_cartesian(rho, &vel_cart, &x_cart, cs, min_w, inv_dt);
                for k in 0..3 {
                    s_cart[k] = s_cart[k] + sm[k];
                }
            }
            vector_from_cartesian_gv(coords, &coord3, &s_cart, ncomp)
        })
    };

    let den = {
        let (axes, cart_axes) = (axes.clone(), cart_axes.clone());
        lift_to_built(move || {
            let (rho, _v, x_cart, cs, min_w, inv_dt, _c3) =
                body_scaffold(coords, ndim, ncomp, &axes, has_energy);
            let mut s = Gv::ZERO;
            for b in 0..n_bodies {
                s = s + body_at(b, ndim, &cart_axes).density(rho, cs, &x_cart, min_w, inv_dt);
            }
            vec![s]
        })
    };

    let mut out = vec![("mom".to_string(), mom), ("den".to_string(), den)];

    if has_energy {
        let (axes, cart_axes) = (axes.clone(), cart_axes.clone());
        let nrg = lift_to_built(move || {
            let (rho, vel_cart, x_cart, cs, min_w, inv_dt, _c3) =
                body_scaffold(coords, ndim, ncomp, &axes, has_energy);
            let gamma = Gv::scalar("gamma");
            let pre = Gv::scalar(law_params::PRE);
            let e_int = pre / ((gamma - Gv::ONE) * rho);
            let mut d_nrg = Gv::ZERO;
            for b in 0..n_bodies {
                let bsrc = body_at(b, ndim, &cart_axes);
                let a = bsrc.accel(&x_cart);
                let den_dot = bsrc.accretion_rate(rho, cs, &x_cart, min_w, inv_dt);
                let vstar = bsrc.sink_velocity(&vel_cart, &x_cart);
                let work = rho * (a[0] * vel_cart[0] + a[1] * vel_cart[1] + a[2] * vel_cart[2]);
                let vstar2 = vstar[0] * vstar[0] + vstar[1] * vstar[1] + vstar[2] * vstar[2];
                d_nrg = d_nrg + work - (Gv::from_f64(0.5) * vstar2 + e_int) * den_dot;
            }
            vec![d_nrg]
        });
        out.push(("nrg".to_string(), nrg));
    }
    out
}
