// =============================================================================
// gv_immersed.rs
//
// the immersed-boundary body source terms traced at S = Gv (docs/design/19):
//   - body_source_gv  (FORWARD, bodies -> fluid): softened gravity + Bondi-Hoyle accretion,
//     cons -> cons in-place (`cons += dt * S`).
//   - body_feedback_gv (BACKWARD, fluid -> bodies): per-cell per-body force / torque / accreted
//     mass -> scratch fields a device reduction sums into each body's BodyDelta.
//
// GENERIC over coordinate system (docs/design/19 P4): the physics is done in CARTESIAN
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
/// grid-plane convention (docs/design/19 P4). identical to `immersed::body_cart_axes`.
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
    let inv_dt = Gv::ONE / dt;
    let den = Gv::field("den", FieldRef::cons_den());
    let mom: Vec<Gv> = (0..ncomp)
        .map(|comp| Gv::field(&format!("mom_{comp}"), FieldRef::cons_mom(comp as u8)))
        .collect();
    let nrg = Gv::field("nrg", FieldRef::cons_nrg());
    let cart_axes = body_cart_axes(coords, ndim, axes);
    let (coord3, cell_cart, vel_cart, min_w, cs, e_int) =
        cell_scaffold(coords, ndim, ncomp, axes, gamma, den, &mom, nrg);

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
