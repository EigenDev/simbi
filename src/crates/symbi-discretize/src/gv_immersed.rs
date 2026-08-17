// =============================================================================
// gv_immersed.rs
//
// the immersed-boundary body source terms traced at S = Gv:
//   - body_source_gv  (forward, bodies -> fluid): softened gravity + bondi-hoyle accretion,
//     cons -> cons in-place (`cons += dt * S`).
//   - body_feedback_gv (backward, fluid -> bodies): per-cell per-body force / torque / accreted
//     mass -> scratch fields a device reduction sums into each body's BodyDelta.
//
// generic over coordinate system: the physics is done in cartesian
// (coord-free) — `cell_scaffold` supplies the cell's cartesian position + gas velocity via the
// gv to_cartesian / vector_to_cartesian transforms; the forward source projects gravity + sink
// velocity onto the physical momentum frame (vector_from_cartesian), the feedback keeps the
// cartesian force/torque (matching symbi_ib's BodyDelta). the physics is a function of `cons`
// alone (c2p-free); the 0..n_bodies loop is branch-free (inactive body: mass=0 / sink_rate=0).
//
// this module sits outside the (already large) gv module and rides the public gv API alone —
// Gv arithmetic + the transcendentals (exp) + `cell_geometry_gv` + begin/end_trace.
// =============================================================================

use symbi_ir::graph::NodeId;

use symbi_algebra::algebra::Numeric;
use symbi_ir::algebra::Scalar;
use symbi_ir::{FieldBind, FieldRef};

use super::coords::{Coords, Spacing};
use super::gv::{CellGeometryGv, cell_geometry_gv};
use symbi_ir::{Gv, GvKernel, begin_trace, end_trace};

type Writes = Vec<(String, FieldBind, NodeId)>;

#[inline]
fn sq(a: Gv) -> Gv {
    a * a
}

/// per-cell, per-body physics in cartesian (coord-free): softened gravity `g`, the well-posed
/// drain rate `drain_rate` (1/time; the fluid in the mask relaxes by `exp(-drain_rate*dt)`), and
/// `rvec = cell - body`. the drain replaces the KMK04 mass-only sink: a
/// uniform exponential scaling of every conserved component leaves the intensive primitive state
/// invariant (acoustically silent, positivity-preserving for any dt) and the accretion rate is
/// emergent (the reduced `U(1 - exp(-rate*dt))`).
struct BodyContributionGv {
    g: [Gv; 3],
    drain_rate: Gv,
    rvec: [Gv; 3],
}

/// the cartesian axes (0=x,1=y,2=z) a body's ndim-D position/velocity components map to — the
/// grid-plane convention. identical to `immersed::body_cart_axes`.
pub(crate) fn body_cart_axes(coords: Coords, ndim: usize, axes: &[usize]) -> Vec<usize> {
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
pub(crate) fn to_cartesian_gv(coords: Coords, coord: &[Gv; 3]) -> [Gv; 3] {
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
            let (st, ct, cp, sp) = (
                coord[1].sin(),
                coord[1].cos(),
                coord[2].cos(),
                coord[2].sin(),
            );
            [st * cp, st * sp, ct] // r_hat
        }
        (Coords::Spherical, 1) => {
            let (st, ct, cp, sp) = (
                coord[1].sin(),
                coord[1].cos(),
                coord[2].cos(),
                coord[2].sin(),
            );
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
pub(crate) fn body_vec3(b: usize, ndim: usize, cart_axes: &[usize], name: &str) -> [Gv; 3] {
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
/// cartesian position, the gas velocity in cartesian, min width, cs, e_int.
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
    // coord3 in natural coordinate order: gridded coords from the centroid, ungridded = 0.
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
    let mass = Gv::scalar(&format!("body_{b}_mass"));
    let soft = Gv::scalar(&format!("body_{b}_soft"));
    let soft_kind = Gv::scalar(&format!("body_{b}_softkind"));
    let bpos = body_vec3(b, ndim, cart_axes, "pos");

    // r_vec = cell_cart - body_pos; r_mag = |r_vec| (for the drain mask).
    let rvec: [Gv; 3] = std::array::from_fn(|i| cell_cart[i] - bpos[i]);
    let r_mag = (sq(rvec[0]) + sq(rvec[1]) + sq(rvec[2])).sqrt();

    // the body's gravitational field, family selected by `softkind` — the carrier-generic form,
    // proven conservative (g = -grad phi) + bounded in the well-posedness suite (`ibm.rs`).
    let g = crate::ibm::body_gravity(rvec, mass, soft, soft_kind);

    // the well-posed drain rate: chi * min(sink, cs/dx), the mollified mask chi =
    // 0.5(1 - tanh((r - r_mask)/w)) (w = one cell) times the sound-crossing-capped sink. `sink_rate`
    // (per body) is the user dial: 0 for a non-accreting body (drain_rate = 0, exact no-op), large ->
    // the full sound-crossing drain. carrier-generic form proven nonnegative -> f in (0,1] (`ibm.rs`).
    let r_mask = Gv::scalar(&format!("body_{b}_racc"));
    let sink_rate = Gv::scalar(&format!("body_{b}_sink"));
    // spatial gate at the mask's exact support (ibm::DRAIN_SUPPORT_WIDTHS): beyond it the
    // ungated rate is exactly zero (tanh saturation), so the lazy branch skips the
    // tanh + divisions on the far field — ~all cells for a sink of a few cell widths — for a
    // bit-identical result. the branch is spatially coherent, hence well predicted.
    let r_cut = r_mask + Gv::from_f64(crate::ibm::DRAIN_SUPPORT_WIDTHS) * min_w;
    let drain_rate = Gv::cond(
        r_mag.cmp_lt(r_cut),
        || crate::ibm::drain_rate(r_mag, r_mask, min_w, sink_rate, cs),
        || Gv::ZERO,
    );
    let _ = (inv_dt, vel_cart, den); // unused by the drain (kept for the shared signature)
    BodyContributionGv {
        g,
        drain_rate,
        rvec,
    }
}

/// per-cell body-evolved adiabatic conserved state as a standalone forward kick: `(den, mom, nrg)`
/// -> the state after `dt *` the immersed-body source (gravity + accretion sink) over all
/// `n_bodies` slots, with the gravity half exact to second order in `dt` by construction.
///
/// the energy carries `0.5 rho |g|^2 dt^2` explicitly, which is what makes the isolated kick
/// preserve internal energy exactly: momentum gains `rho g dt`, so the kinetic energy it implies
/// gains `m.g dt + 0.5 rho |g|^2 dt^2`, and the energy is credited exactly that. this is the right
/// operator for a consumer that applies the whole kick to one state and takes the result as the
/// answer — the FOFC freeze parachute, which evolves the stage input and uses it as that cell's
/// entire update.
///
/// applied on top of a state some other operator has already advanced, this becomes sequential
/// composition: an exact flux update composed with an exact source update is first order in `dt`
/// however accurately either half is integrated, and the residue is one-signed in the internal
/// energy. a consumer that runs after the flux divergence wants [`body_applied_gv`].
///
/// a pure function of the cell state in registers, touching registers alone, so it composes into
/// any kernel that already holds the conserved state with no materialized buffer. declares
/// `dt` / `gamma` + the per-body scalars via `body_contribution`; unused body slots contribute zero.
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
) -> (Gv, Vec<Gv>, Gv, Gv) {
    let inv_dt = Gv::ONE / dt;
    let cart_axes = body_cart_axes(coords, ndim, axes);
    let (coord3, cell_cart, vel_cart, min_w, cs, _e_int) =
        cell_scaffold(coords, ndim, ncomp, axes, gamma, den, mom, nrg);

    // gravity is an additive momentum + energy source; the drain is the total rate over all bodies,
    // applied as one uniform multiplicative factor (the exact-exponential
    // relaxation). the two operators split cleanly: gravity accelerates, then the mask drains.
    let mut d_mom: Vec<Gv> = vec![Gv::ZERO; ncomp];
    let mut total_rate = Gv::ZERO;
    for b in 0..n_bodies {
        let bc = body_contribution(
            b, ndim, &cart_axes, &cell_cart, &vel_cart, den, cs, min_w, inv_dt,
        );
        let g_phys = vector_from_cartesian_gv(coords, &coord3, &bc.g, ncomp);
        for comp in 0..ncomp {
            d_mom[comp] = d_mom[comp] + den * g_phys[comp]; // gravity force
        }
        total_rate = total_rate + bc.drain_rate;
    }
    // average-momentum work across the finite kick preserves internal energy:
    // delta E = m.g dt + 0.5 rho |g|^2 dt^2.
    let mut gravity_work = Gv::ZERO;
    let mut force_sq = Gv::ZERO;
    for comp in 0..ncomp {
        gravity_work = gravity_work + mom[comp] * d_mom[comp] / den;
        force_sq = force_sq + d_mom[comp] * d_mom[comp];
    }

    // f = exp(-total_rate * dt) in (0, 1]: uniform scaling of den, mom, nrg (intensive state
    // invariant, positivity-preserving for any dt). f = 1 outside every mask -> exact no-op.
    // outside every mask total_rate is exactly zero and the ungated factor exactly
    // one — the lazy branch skips the exp without changing a bit.
    let f = Gv::cond(
        total_rate.cmp_gt(Gv::ZERO),
        || crate::ibm::drain_factor(total_rate, dt),
        || Gv::ONE,
    );
    let den_new = den * f;
    let mom_new: Vec<Gv> = (0..ncomp)
        .map(|comp| (mom[comp] + dt * d_mom[comp]) * f)
        .collect();
    let nrg_new = (nrg + dt * gravity_work + Gv::from_f64(0.5) * dt * dt * force_sq / den) * f;
    // `f` is returned so a dyed run can drain the conserved dye by the same factor. a sink removes
    // gas together with the dye dissolved in it, leaving the concentration of what remains
    // untouched: `D_chi = rho chi` scales exactly as `rho` does. recomputing the factor at the
    // call site would duplicate the per-cell mask and rate, so it is handed out instead.
    (den_new, mom_new, nrg_new, f)
}

/// the immersed-body operator for a consumer that runs after the flux divergence: the body
/// contribution is evaluated at `src` (the stage input) and applied to `dst` (the flux-advanced
/// conserved state).
///
/// splitting the evaluation point from the application point is what makes this composable. an
/// explicit scheme advances `cons = a0 u_n + ac (cons - dt div F + dt S)`, in which the flux and
/// the source are both evaluated at the stage input and summed into one convex update. applying a
/// complete source operator on top of an already-flux-advanced state is a different scheme —
/// sequential composition — and it is first order in `dt` no matter how accurately either half is
/// integrated. the residue lands in the internal energy `e = E - |m|^2/2rho` with a fixed sign,
/// so it accumulates, and it survives every Runge-Kutta order because each stage repeats the
/// same composition.
///
/// two consequences for the form below, both differences from [`body_evolved_gv`]:
///
/// - the gravity force, the gravity work and the drain rate all read `src`, so the source is the
///   one the stage's flux was evaluated against.
/// - the energy carries `m.g dt` alone. the `0.5 rho |g|^2 dt^2` second-order term belongs to a
///   standalone kick; here the stage weights reconstruct it, so adding it explicitly would
///   double-count it.
///
/// the accretion drain is unchanged: `f = exp(-rate dt)` still multiplies the whole conserved
/// vector, which is exact for the relaxation it solves and positivity-preserving for any `dt`
/// regardless of the state it acts on. gravity accelerates, then the mask drains.
#[allow(clippy::too_many_arguments)]
pub(crate) fn body_applied_gv(
    dst_den: Gv,
    dst_mom: &[Gv],
    dst_nrg: Gv,
    src_den: Gv,
    src_mom: &[Gv],
    src_nrg: Gv,
    dt: Gv,
    gamma: Gv,
    n_bodies: usize,
    coords: Coords,
    ndim: usize,
    ncomp: usize,
    axes: &[usize],
) -> (Gv, Vec<Gv>, Gv, Gv) {
    let inv_dt = Gv::ONE / dt;
    let cart_axes = body_cart_axes(coords, ndim, axes);
    let (coord3, cell_cart, vel_cart, min_w, cs, _e_int) = cell_scaffold(
        coords, ndim, ncomp, axes, gamma, src_den, src_mom, src_nrg,
    );

    let mut d_mom: Vec<Gv> = vec![Gv::ZERO; ncomp];
    let mut total_rate = Gv::ZERO;
    for b in 0..n_bodies {
        let bc = body_contribution(
            b, ndim, &cart_axes, &cell_cart, &vel_cart, src_den, cs, min_w, inv_dt,
        );
        let g_phys = vector_from_cartesian_gv(coords, &coord3, &bc.g, ncomp);
        for comp in 0..ncomp {
            d_mom[comp] = d_mom[comp] + src_den * g_phys[comp];
        }
        total_rate = total_rate + bc.drain_rate;
    }

    // the work rate `rho (v . g)` at the state the force was evaluated at, so the momentum source
    // and the energy source describe the same acceleration acting on the same gas.
    let mut gravity_work = Gv::ZERO;
    for comp in 0..ncomp {
        gravity_work = gravity_work + src_mom[comp] * d_mom[comp] / src_den;
    }

    let f = Gv::cond(
        total_rate.cmp_gt(Gv::ZERO),
        || crate::ibm::drain_factor(total_rate, dt),
        || Gv::ONE,
    );
    let den_new = dst_den * f;
    let mom_new: Vec<Gv> = (0..ncomp)
        .map(|comp| (dst_mom[comp] + dt * d_mom[comp]) * f)
        .collect();
    let nrg_new = (dst_nrg + dt * gravity_work) * f;
    (den_new, mom_new, nrg_new, f)
}

pub fn body_source_gv(
    n_bodies: usize,
    coords: Coords,
    ndim: usize,
    ncomp: usize,
    axes: &[usize],
    has_dye: bool,
) -> (GvKernel, Writes) {
    begin_trace();
    let dt = Gv::scalar("dt");
    let gamma = Gv::scalar("gamma");
    let den = Gv::field("den", FieldRef::cons_den());
    let mom: Vec<Gv> = (0..ncomp)
        .map(|comp| Gv::field(&format!("mom_{comp}"), FieldRef::cons_mom(comp as u8)))
        .collect();
    let nrg = Gv::field("nrg", FieldRef::cons_nrg());
    // this pass runs after the godunov stage has advanced `cons`, so the body contribution is
    // evaluated at the stage input — the state the stage's flux divergence was also evaluated at —
    // and applied to the advanced `cons`.
    let us_den = Gv::field("us_den", FieldRef::ustage_den());
    let us_mom: Vec<Gv> = (0..ncomp)
        .map(|comp| Gv::field(&format!("us_mom_{comp}"), FieldRef::ustage_mom(comp as u8)))
        .collect();
    let us_nrg = Gv::field("us_nrg", FieldRef::ustage_nrg());
    let (den_new, mom_new, nrg_new, drain) = body_applied_gv(
        den, &mom, nrg, us_den, &us_mom, us_nrg, dt, gamma, n_bodies, coords, ndim, ncomp, axes,
    );

    let mut writes = vec![(
        "den_new".to_string(),
        FieldRef::cons_den().into(),
        den_new.node(),
    )];
    for (comp, m) in mom_new.iter().enumerate() {
        writes.push((
            format!("mom_{comp}_new"),
            FieldRef::cons_mom(comp as u8).into(),
            m.node(),
        ));
    }
    writes.push((
        "nrg_new".to_string(),
        FieldRef::cons_nrg().into(),
        nrg_new.node(),
    ));
    // the dye drains with the mass it is dissolved in, so the concentration the surviving gas
    // carries is unchanged. the drain alone touches the dye; gravity only accelerates the gas.
    if has_dye {
        let chi = Gv::field("chi", FieldRef::cons_chi());
        writes.push((
            "chi_new".to_string(),
            FieldRef::cons_chi().into(),
            (chi * drain).node(),
        ));
    }
    (end_trace(), writes)
}

/// backward feedback, gravity-reaction half (single body slot): per cell, the reaction
/// force the gas exerts on the body, `f_grav[ax] = -(den * g_cart[ax]) * dv`. genuinely
/// global support — every gas cell pulls on the body — so the runtime reduces it over
/// the full interior. reads `cons.den` alone, so the pass streams a single field.
/// slot-0 scalar names (`body_0_*`); the dispatch rebinds them per active body.
pub fn body_feedback_grav_gv(coords: Coords, ndim: usize, axes: &[usize]) -> (GvKernel, Writes) {
    begin_trace();
    let cart_axes = body_cart_axes(coords, ndim, axes);
    let den = Gv::field("den", FieldRef::cons_den());
    let geo: CellGeometryGv = cell_geometry_gv(coords, &vec![Spacing::Uniform; ndim], axes, ndim);
    let mut coord3 = [Gv::ZERO; 3];
    for (g, &coord_idx) in axes.iter().enumerate() {
        if coord_idx < 3 {
            coord3[coord_idx] = geo.centroid[g];
        }
    }
    let cell_cart = to_cartesian_gv(coords, &coord3);
    let dv = Gv::ONE / geo.inv_volume;
    let mass = Gv::scalar("body_0_mass");
    let soft = Gv::scalar("body_0_soft");
    let soft_kind = Gv::scalar("body_0_softkind");
    let bpos = body_vec3(0, ndim, &cart_axes, "pos");
    let rvec: [Gv; 3] = std::array::from_fn(|i| cell_cart[i] - bpos[i]);
    let g = crate::ibm::body_gravity(rvec, mass, soft, soft_kind);
    let mut writes: Writes = Vec::new();
    for ax in 0..ndim {
        let fc = -(den * g[cart_axes[ax]]) * dv;
        writes.push((
            format!("b0_f{ax}"),
            format!("fb_0_force_{ax}").into(),
            fc.node(),
        ));
    }
    (end_trace(), writes)
}

/// backward feedback, drain half (single body slot): the sink-weighted quantities —
/// drag force (absorbed momentum / dt), torque, absorbed mass and energy. every output
/// is proportional to `frac = 1 - exp(-rate*dt)`, which is exactly zero outside the
/// mask support (tanh saturation, `ibm::DRAIN_SUPPORT_WIDTHS`), so the runtime
/// dispatches and reduces this kernel over the body's support bounding box only: an
/// omitted cell contributes an exact zero to every sum. slot-0 scalar names; the
/// dispatch rebinds them per active body.
pub fn body_feedback_drain_gv(
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
    let geo: CellGeometryGv = cell_geometry_gv(coords, &vec![Spacing::Uniform; ndim], axes, ndim);
    let dv = Gv::ONE / geo.inv_volume;
    let mom_cart: [Gv; 3] = std::array::from_fn(|i| den * vel_cart[i]);

    let bc = body_contribution(
        0, ndim, &cart_axes, &cell_cart, &vel_cart, den, cs, min_w, inv_dt,
    );
    // the saturation lemma at the drain seam: the cond-gated rate is exactly
    // zero outside |x - body_pos| > racc + DRAIN_SUPPORT_WIDTHS*min(dx) (the
    // gate radius is the tanh saturation radius), so every feedback write —
    // all multiples of the rate — derives this ball. cartesian only: the ball
    // lives in cartesian space, the one chart whose index box contains it.
    if matches!(coords, Coords::Cartesian) {
        use symbi_ir::ParamExpr;
        symbi_ir::tag_support_ball(
            &bc.drain_rate,
            (0..ndim)
                .map(|ax| ParamExpr::param(&format!("body_0_pos_{}", cart_axes[ax])))
                .collect(),
            ParamExpr::param("body_0_racc")
                + ParamExpr::constant(crate::ibm::DRAIN_SUPPORT_WIDTHS)
                    * ParamExpr::min_of(
                        (0..ndim)
                            .map(|ax| ParamExpr::param(&format!("dx_{ax}")))
                            .collect(),
                    ),
        );
    }
    let frac = Gv::cond(
        bc.drain_rate.cmp_gt(Gv::ZERO),
        || Gv::ONE - (Gv::ZERO - bc.drain_rate * dt).exp(),
        || Gv::ZERO,
    );
    let mut fa = [Gv::ZERO; 3];
    for i in 0..3 {
        fa[i] = mom_cart[i] * frac * dv * inv_dt;
    }
    let mut writes: Writes = Vec::new();
    for ax in 0..ndim {
        writes.push((
            format!("b0_f{ax}"),
            format!("fb_0_force_{ax}").into(),
            fa[cart_axes[ax]].node(),
        ));
    }
    let cross = |a: usize, bb: usize| bc.rvec[a] * fa[bb] - bc.rvec[bb] * fa[a];
    for (t, tc) in [cross(1, 2), cross(2, 0), cross(0, 1)]
        .into_iter()
        .enumerate()
    {
        writes.push((
            format!("b0_t{t}"),
            format!("fb_0_torque_{t}").into(),
            tc.node(),
        ));
    }
    writes.push((
        "b0_m".to_string(),
        "fb_0_mass".into(),
        (den * frac * dv).node(),
    ));
    writes.push((
        "b0_e".to_string(),
        "fb_0_energy".into(),
        (nrg * frac * dv).node(),
    ));
    let kernel = end_trace().with_derived_support(&writes);
    (kernel, writes)
}

/// backward feedback: per cell, per body, the cartesian force / 3D torque / absorbed mass / absorbed
/// energy each body receives -> the MAX_SOURCE_BODIES*(ndim+5) reduction-scratch writes (`fb_{b}_force_{ax}`
/// / `fb_{b}_torque_{t}` / `fb_{b}_mass` / `fb_{b}_energy`, the order the runtime sums). generic over
/// coord system.
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

    let _ = dv_dt;
    // the gas momentum in cartesian (den * v_cart): what the uniform drain removes proportionally.
    let mom_cart: [Gv; 3] = std::array::from_fn(|i| den * vel_cart[i]);

    let mut writes: Writes = Vec::new();
    for b in 0..n_bodies {
        let bc = body_contribution(
            b, ndim, &cart_axes, &cell_cart, &vel_cart, den, cs, min_w, inv_dt,
        );
        // the fraction of this cell drained by body b this step: frac = 1 - exp(-rate*dt). exact for
        // non-overlapping masks (each cell in at most one mask -> matches the forward's total-rate
        // factor); overlapping masks slightly over-attribute in this diagnostic reduction.
        let frac = Gv::cond(
            bc.drain_rate.cmp_gt(Gv::ZERO),
            || Gv::ONE - (Gv::ZERO - bc.drain_rate * dt).exp(),
            || Gv::ZERO,
        );
        // body force (cartesian, 3D) = gravity reaction (-den*g*dv) + accretion drag
        // (+ absorbed momentum / dt). the drag `fa` alone carries the torque (gravity is central).
        let mut force_cart = [Gv::ZERO; 3];
        let mut fa = [Gv::ZERO; 3];
        for i in 0..3 {
            fa[i] = mom_cart[i] * frac * dv * inv_dt; // absorbed momentum / dt = drag force
            force_cart[i] = -(den * bc.g[i]) * dv + fa[i];
        }
        // the body's ndim force components live at its cartesian axes.
        for ax in 0..ndim {
            let fc = force_cart[cart_axes[ax]];
            writes.push((
                format!("b{b}_f{ax}"),
                format!("fb_{b}_force_{ax}").into(),
                fc.node(),
            ));
        }
        // torque = r x F_drag (full 3D cross; 2D yields (0,0,tz) automatically).
        let cross = |a: usize, bb: usize| bc.rvec[a] * fa[bb] - bc.rvec[bb] * fa[a];
        for (t, tc) in [cross(1, 2), cross(2, 0), cross(0, 1)]
            .into_iter()
            .enumerate()
        {
            writes.push((
                format!("b{b}_t{t}"),
                format!("fb_{b}_torque_{t}").into(),
                tc.node(),
            ));
        }
        // absorbed mass = den * frac * dv (the emergent accretion, a functional of the flow).
        writes.push((
            format!("b{b}_m"),
            format!("fb_{b}_mass").into(),
            (den * frac * dv).node(),
        ));
        // absorbed total (internal + kinetic) energy = nrg * frac * dv -- the accretion power,
        // closing the gas+body energy ledger. adiabatic only (the iso state carries den + mom).
        writes.push((
            format!("b{b}_e"),
            format!("fb_{b}_energy").into(),
            (nrg * frac * dv).node(),
        ));
    }
    (end_trace(), writes)
}

// =============================================================================
// isothermal variants (no energy equation)
//
// the immersed-body physics is EOS-independent — softened gravity + bondi-hoyle
// accretion are functions of (den, mom, cs) only, through the shared `body_contribution`.
// the single difference from the adiabatic kernels is the closure for `cs` and
// the reach of the update:
//   - adiabatic: cs / e_int are recovered from `cons.nrg` (`gas_state`), and the
//     source updates `cons.nrg` (gravity work + accreted internal+kinetic energy).
//   - isothermal: the state carries den + mom, and cs comes from `prim.pre` (= cs^2(x)*rho,
//     the substrate's iso pressure encoding, exactly what the iso flux reads), so
//     a locally-isothermal cs^2(x) flows through identically and the update stays in den + mom.
// future media (porous / deformable) extend `body_contribution`, so both the
// adiabatic and isothermal kernels inherit them with no further duplication.
// =============================================================================

// iso gas state from cons + the substrate pressure: (vel_physical[ncomp], cs).
// `pre = cs^2(x)*rho` so `cs = sqrt(pre/rho)`. the isothermal closure returns velocity and cs.
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

/// forward iso source: `cons += dt * (S_grav + S_accretion)` over den and mom.
/// reads cons (den/mom) + prim.pre; writes den/mom. gravity + accretion via
/// the shared `body_contribution`.
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
    // evaluated at the stage input, applied to the godunov-advanced cons — see `body_applied_gv`.
    let us_den = Gv::field("us_den", FieldRef::ustage_den());
    let us_mom: Vec<Gv> = (0..ncomp)
        .map(|comp| Gv::field(&format!("us_mom_{comp}"), FieldRef::ustage_mom(comp as u8)))
        .collect();
    let (den_new, mom_new) = body_applied_iso_gv(
        den, &mom, us_den, &us_mom, pre, dt, n_bodies, coords, ndim, ncomp, axes,
    );

    let mut writes = vec![(
        "den_new".to_string(),
        FieldRef::cons_den().into(),
        den_new.node(),
    )];
    for (comp, m) in mom_new.iter().enumerate() {
        writes.push((
            format!("mom_{comp}_new"),
            FieldRef::cons_mom(comp as u8).into(),
            m.node(),
        ));
    }
    (end_trace(), writes)
}

/// the isothermal immersed-body evolution as a pure per-cell function of its arguments alone: the
/// energy-free twin of `body_evolved_gv`. given the cell's conserved (den, mom) and its isothermal
/// pressure `pre` (which sets the sound speed), returns the state advanced by `dt` of softened
/// newtonian gravity + bondi-hoyle accretion from `n_bodies` point masses. shared by the standalone
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

    // gravity additive, then the uniform drain (den + mom; no energy equation). see the adiabatic
    // `body_evolved_gv` for the operator-split rationale.
    let mut d_mom: Vec<Gv> = vec![Gv::ZERO; ncomp];
    let mut total_rate = Gv::ZERO;
    for b in 0..n_bodies {
        let bc = body_contribution(
            b, ndim, &cart_axes, &cell_cart, &vel_cart, den, cs, min_w, inv_dt,
        );
        let g_phys = vector_from_cartesian_gv(coords, &coord3, &bc.g, ncomp);
        for comp in 0..ncomp {
            d_mom[comp] = d_mom[comp] + den * g_phys[comp]; // gravity force
        }
        total_rate = total_rate + bc.drain_rate;
    }

    let f = Gv::cond(
        total_rate.cmp_gt(Gv::ZERO),
        || (Gv::ZERO - total_rate * dt).exp(),
        || Gv::ONE,
    );
    let den_new = den * f;
    let mom_new: Vec<Gv> = (0..ncomp)
        .map(|comp| (mom[comp] + dt * d_mom[comp]) * f)
        .collect();
    (den_new, mom_new)
}

/// host-testable trace of the standalone body kick [`body_evolved_gv`], which reaches production
/// inlined into the FOFC freeze parachute, wholly inside that kernel's boundary. reads and
/// writes cons; the law it carries is that the kick leaves internal energy exactly fixed, because
/// the energy it credits is exactly the kinetic energy its own momentum update implies.
pub fn body_evolved_probe_gv(
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
    let (den_new, mom_new, nrg_new, _drain) = body_evolved_gv(
        den, &mom, nrg, dt, gamma, n_bodies, coords, ndim, ncomp, axes,
    );
    let mut writes = vec![(
        "den_new".to_string(),
        FieldRef::cons_den().into(),
        den_new.node(),
    )];
    for (comp, m) in mom_new.iter().enumerate() {
        writes.push((
            format!("mom_{comp}_new"),
            FieldRef::cons_mom(comp as u8).into(),
            m.node(),
        ));
    }
    writes.push((
        "nrg_new".to_string(),
        FieldRef::cons_nrg().into(),
        nrg_new.node(),
    ));
    (end_trace(), writes)
}

/// the isothermal twin of [`body_applied_gv`]: the body contribution is evaluated at `src` (the
/// stage input) and applied to `dst` (the flux-advanced conserved state). there is no energy
/// equation here, so the entropy the adiabatic form loses has nowhere to show up — but the
/// momentum source is subject to the same composition: a complete source operator applied on top
/// of an already-advanced state is a sequential composition, first order in `dt` at any
/// Runge-Kutta order. evaluating it alongside the flux keeps the isothermal and adiabatic bodies
/// the same discrete operator.
#[allow(clippy::too_many_arguments)]
pub(crate) fn body_applied_iso_gv(
    dst_den: Gv,
    dst_mom: &[Gv],
    src_den: Gv,
    src_mom: &[Gv],
    src_pre: Gv,
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
        cell_scaffold_iso(coords, ndim, ncomp, axes, src_den, src_mom, src_pre);

    let mut d_mom: Vec<Gv> = vec![Gv::ZERO; ncomp];
    let mut total_rate = Gv::ZERO;
    for b in 0..n_bodies {
        let bc = body_contribution(
            b, ndim, &cart_axes, &cell_cart, &vel_cart, src_den, cs, min_w, inv_dt,
        );
        let g_phys = vector_from_cartesian_gv(coords, &coord3, &bc.g, ncomp);
        for comp in 0..ncomp {
            d_mom[comp] = d_mom[comp] + src_den * g_phys[comp];
        }
        total_rate = total_rate + bc.drain_rate;
    }

    let f = Gv::cond(
        total_rate.cmp_gt(Gv::ZERO),
        || (Gv::ZERO - total_rate * dt).exp(),
        || Gv::ONE,
    );
    let den_new = dst_den * f;
    let mom_new: Vec<Gv> = (0..ncomp)
        .map(|comp| (dst_mom[comp] + dt * d_mom[comp]) * f)
        .collect();
    (den_new, mom_new)
}

/// backward iso feedback: identical force/torque/mass writes as the adiabatic
/// kernel; cs comes from prim.pre. 2D+ only.
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
    let mom_cart: [Gv; 3] = std::array::from_fn(|i| den * vel_cart[i]);

    let mut writes: Writes = Vec::new();
    for b in 0..n_bodies {
        let bc = body_contribution(
            b, ndim, &cart_axes, &cell_cart, &vel_cart, den, cs, min_w, inv_dt,
        );
        let frac = Gv::cond(
            bc.drain_rate.cmp_gt(Gv::ZERO),
            || Gv::ONE - (Gv::ZERO - bc.drain_rate * dt).exp(),
            || Gv::ZERO,
        );
        let mut force_cart = [Gv::ZERO; 3];
        let mut fa = [Gv::ZERO; 3];
        for i in 0..3 {
            fa[i] = mom_cart[i] * frac * dv * inv_dt; // drag = absorbed momentum / dt
            force_cart[i] = -(den * bc.g[i]) * dv + fa[i];
        }
        for ax in 0..ndim {
            let fc = force_cart[cart_axes[ax]];
            writes.push((
                format!("b{b}_f{ax}"),
                format!("fb_{b}_force_{ax}").into(),
                fc.node(),
            ));
        }
        let cross = |a: usize, bb: usize| bc.rvec[a] * fa[bb] - bc.rvec[bb] * fa[a];
        for (t, tc) in [cross(1, 2), cross(2, 0), cross(0, 1)]
            .into_iter()
            .enumerate()
        {
            writes.push((
                format!("b{b}_t{t}"),
                format!("fb_{b}_torque_{t}").into(),
                tc.node(),
            ));
        }
        writes.push((
            format!("b{b}_m"),
            format!("fb_{b}_mass").into(),
            (den * frac * dv).node(),
        ));
    }
    (end_trace(), writes)
}

/// the cartesian position of a point displaced `half_cells` half-cell widths from the current
/// cell's lower face along the sweep axis: `half_cells = 2k` lands on the lower face of cell
/// `i+k`, `2k+1` on that cell's centre.
///
/// natural-order coordinates: every axis at its own centroid, then the sweep axis replaced by the
/// displaced position. an even half-cell offset lands on a face of the runtime spacing map; an odd
/// one lands on the cell center between the two bracketing faces, through the same map-aware
/// center the host `stagger_coord(Center)` uses (geometric mean on a log axis, arithmetic midpoint
/// otherwise) — the position the initial condition seeds the hydrostatic column at, which is what
/// makes the anchor departures exactly zero. the transverse coordinates hold at the current cell's
/// own centre: the reconstruction is one-dimensional along `dir`, so only the sweep axis moves.
///
/// every consumer of a body's geometry along a sweep reads the same ladder, so the potential a
/// well-balanced reconstruction cancels and the mask indicator a dissipation floor keys on are
/// evaluated at one and the same point.
pub(crate) fn stencil_position_cartesian_gv(
    coords: Coords,
    ndim: usize,
    dir: usize,
    axes: &[usize],
    spacing: &[Spacing],
    half_cells: i64,
) -> [Gv; 3] {
    let geo = cell_geometry_gv(coords, spacing, axes, ndim);
    let mut coord3 = [Gv::ZERO; 3];
    for (g, &coord_idx) in axes.iter().enumerate() {
        if coord_idx < 3 {
            coord3[coord_idx] = geo.centroid[g];
        }
    }
    let sweep_coord = axes[dir];
    if sweep_coord < 3 {
        let lo = crate::gv::gv_axis_face_at(sweep_coord, spacing[dir], half_cells.div_euclid(2));
        coord3[sweep_coord] = if half_cells.rem_euclid(2) == 0 {
            lo
        } else {
            let hi =
                crate::gv::gv_axis_face_at(sweep_coord, spacing[dir], half_cells.div_euclid(2) + 1);
            crate::gv::gv_axis_center_between(sweep_coord, lo, hi)
        };
    }
    to_cartesian_gv(coords, &coord3)
}

/// the total gravitational potential at a point displaced `half_cells` half-cell widths from the
/// current cell's lower face along the sweep axis: `half_cells = 2k` lands on the lower face of
/// cell `i+k`, `2k+1` on that cell's centre.
///
/// built from the same `body_{b}_*` scalars and the same `body_potential` the immersed-body source
/// applies, which is the whole point of sourcing it from the source itself. a
/// well-balanced reconstruction is exact only when the face states it produces cancel the
/// discrete force the scheme exerts; a potential taken from an idealized profile agrees with that
/// force only to truncation order, and the leftover is precisely the residual the balancing exists
/// to remove. `body_potential` is the antiderivative of `body_gravity` under the same softening
/// selector, proven by autodiff (`ibm_wellposedness.rs`, theorem 1), so the pairing is exact for
/// every softening family a body can declare.
///
/// the transverse coordinates are the current cell's own centre: the reconstruction is
/// one-dimensional along `dir`, so only the sweep axis moves.
pub fn stencil_potential_gv(
    n_bodies: usize,
    coords: Coords,
    ndim: usize,
    dir: usize,
    axes: &[usize],
    spacing: &[Spacing],
    half_cells: i64,
) -> Gv {
    let cart_axes = body_cart_axes(coords, ndim, axes);
    let cart = stencil_position_cartesian_gv(coords, ndim, dir, axes, spacing, half_cells);
    (0..n_bodies)
        .map(|b| {
            let bpos = body_vec3(b, ndim, &cart_axes, "pos");
            let rvec: [Gv; 3] = std::array::from_fn(|i| cart[i] - bpos[i]);
            crate::ibm::body_potential(
                rvec,
                Gv::scalar(&format!("body_{b}_mass")),
                Gv::scalar(&format!("body_{b}_soft")),
                Gv::scalar(&format!("body_{b}_softkind")),
            )
        })
        .sum::<Gv>()
}

/// the well-balanced forward body source, gravity only, per chart. the momentum-d source is the
/// area-weighted difference of equilibrium pressures at the cell's own faces,
///
///   S_m[d] = [ A_hi,d (p_eq(phi_hi) - p_eq(phi_c)) - A_lo,d (p_eq(phi_lo) - p_eq(phi_c)) ] / V,
///   S_E    = sum_d v[d] S_m[d],
///
/// with `p_eq` the isentrope through the cell's own stage-input state, `phi` the total body
/// potential at the face and centroid positions (Kaeppeli & Mishra, J. Comput. Phys. 259:199,
/// 2014), and A/V the same `cell_geometry_gv` factors the godunov divergence and the geometric
/// pressure source use. on a discretely balanced column the three cancel by telescoping: the
/// balanced reconstruction's face states make the pressure flux divergence
/// `(A_hi p_eq(phi_hi) - A_lo p_eq(phi_lo))/V`, the geometric source contributes
/// `p_eq(phi_c)(A_hi - A_lo)/V`, and this source is exactly their difference. `p_eq(phi_c)`
/// is the cell's own stage-input pressure bit-exactly (the isentrope is anchored there), so
/// the reference term costs no transcendental. on a transverse axis the two face potentials
/// coincide and the source vanishes in the same float arithmetic. cartesian keeps its landed
/// `(p_eq(phi_hi) - p_eq(phi_lo))/dx` spelling — value-equal (A_hi = A_lo, V = A dx) but a
/// different traced graph, so the chart match preserves the baked cartesian kernels byte-for-
/// byte. off equilibrium the source differs from `rho g` at second order, so smooth dynamics
/// are unchanged at the scheme's order.
///
/// gravity alone, on purpose: every accreting surface in this codebase drains through the
/// penalization stack (`penalize_owns_accretion` = true), so the legacy in-source sink is
/// already inert on production configs; a balanced source preserves an equilibrium, and an
/// active sink is precisely what removes one. the dispatch refuses that pairing outright.
pub fn body_source_wb_gv(
    n_bodies: usize,
    coords: Coords,
    ndim: usize,
    ncomp: usize,
    axes: &[usize],
) -> (GvKernel, Writes) {
    use symbi_hydro::hydrostatic::LocalEquilibrium;
    begin_trace();
    let dt = Gv::scalar("dt");
    let gamma = Gv::scalar("gamma");
    let mom: Vec<Gv> = (0..ncomp)
        .map(|comp| Gv::field(&format!("mom_{comp}"), FieldRef::cons_mom(comp as u8)))
        .collect();
    let nrg = Gv::field("nrg", FieldRef::cons_nrg());
    // evaluated at the stage input, exactly like the analytic body source: the stage's flux
    // divergence was reconstructed from this state, and the cancellation is a statement about
    // the pair.
    let us_den = Gv::field("us_den", FieldRef::ustage_den());
    let us_mom: Vec<Gv> = (0..ncomp)
        .map(|comp| Gv::field(&format!("us_mom_{comp}"), FieldRef::ustage_mom(comp as u8)))
        .collect();
    let us_nrg = Gv::field("us_nrg", FieldRef::ustage_nrg());
    let (us_vel, _cs, e_int) = gas_state(ncomp, gamma, us_den, &us_mom, us_nrg);
    let p_us = (gamma - Gv::ONE) * us_den * e_int;

    // the bake-time spacing enum is vestigial in the traced face map: `gv_axis_face_at_index`
    // selects uniform/log/geometric at runtime through the per-axis `map_kind_{ax}` scalar,
    // so this one kernel serves every grading.
    let spacing = vec![Spacing::Uniform; ndim];
    // the curvilinear form needs the per-axis face areas and inverse volume; traced only on
    // the curvilinear arms so the cartesian graph — and the baked cartesian kernels — carry
    // not one extra node.
    let geo = (coords != Coords::Cartesian).then(|| cell_geometry_gv(coords, &spacing, axes, ndim));

    let mut mom_new: Vec<Gv> = mom.clone();
    let mut nrg_new = nrg;
    for ax in 0..ndim {
        // total body potential at this cell's two faces and centre along `ax`: half-cells
        // 0 (lower face), 1 (centre), 2 (upper face) on the face ladder.
        let phi_lo = stencil_potential_gv(n_bodies, coords, ndim, ax, axes, &spacing, 0);
        let phi_c = stencil_potential_gv(n_bodies, coords, ndim, ax, axes, &spacing, 1);
        let phi_hi = stencil_potential_gv(n_bodies, coords, ndim, ax, axes, &spacing, 2);
        let eq = LocalEquilibrium::through(us_den, p_us, phi_c, gamma);
        let (_, p_lo) = eq.state_at(phi_lo);
        let (_, p_hi) = eq.state_at(phi_hi);
        let s_m = match &geo {
            // at equilibrium `rho g = dp_eq/dx`, so the source is the discrete equilibrium
            // pressure gradient -- upper face minus lower. the flipped difference doubles the
            // force the flux divergence carries where the correct sign cancels it, and a 400-step
            // stagnant column measured |v| = 8.1 under it against 2.9e-2 with the analytic
            // source: sign errors here announce themselves as detonations.
            // the width is the cell's own, through the runtime spacing map (`gv_axis_width`
            // reduces to the `dx_{ax}` scalar on an unmapped axis) -- a graded axis carries a
            // distinct width per cell, and the flux divergence this source telescopes against
            // differences its faces over that same per-cell width.
            None => (p_hi - p_lo) / crate::gv::gv_axis_width(ax, spacing[ax]),
            // the area-weighted form: `p_eq(phi_c)` is `p_us` bit-exactly (the isentrope's
            // anchor point), so the reference term is the raw stage-input pressure. on a
            // radial column the transverse axes see equal face potentials, equal `p_eq`,
            // and a source of exactly zero.
            Some(geo) => (geo.area_hi[ax] * (p_hi - p_us) - geo.area_lo[ax] * (p_lo - p_us))
                * geo.inv_volume,
        };
        mom_new[ax] = mom_new[ax] + dt * s_m;
        nrg_new = nrg_new + dt * us_vel[ax] * s_m;
    }

    let mut writes = Vec::with_capacity(ncomp + 1);
    for (comp, m) in mom_new.iter().enumerate() {
        writes.push((
            format!("mom_{comp}_new"),
            FieldRef::cons_mom(comp as u8).into(),
            m.node(),
        ));
    }
    writes.push((
        "nrg_new".to_string(),
        FieldRef::cons_nrg().into(),
        nrg_new.node(),
    ));
    (end_trace(), writes)
}
