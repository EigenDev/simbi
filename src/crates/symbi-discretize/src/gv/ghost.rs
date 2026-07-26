// =============================================================================
// ghost.rs
//
// lattice-map ghost-fill kernel builders (the boundary pullback).
// =============================================================================

use super::*;

/// the isothermal lattice-map ghost fill — pull back rho/vel/pre at the per-axis source coord,
/// write IN PLACE; the velocity component whose coordinate is a GRID axis picks up that axis's
/// wall-normal `vel_sign` (an ungridded swirl coordinate has no wall map -> unflipped). rho/pre
/// are grade-0 copies. `ncomp` velocity components, `ndim` gridded axes; `axes[d]` = the coord
/// of grid axis d. the EOS-generic 3-field pullback the iso/newton/rhd ghost fill share.
pub fn iso_ghost_fill_gv(
    ndim: usize,
    ncomp: usize,
    axes: &[usize],
) -> (GvKernel, Vec<(String, FieldBind, NodeId)>) {
    begin_trace();
    let src = gv_lattice_source(ndim);
    let vel_sign: Vec<Gv> = (0..ndim)
        .map(|ax| Gv::scalar(&format!("vel_sign_{ax}")))
        .collect();
    let rho = gv_load_at("prim_rho", "prim.rho", &src);
    let mut writes = vec![("prim_rho".to_string(), FieldRef::PrimRho.into(), rho.node())];
    for k in 0..ncomp {
        let v = gv_load_at(&format!("prim_v{k}"), FieldRef::PrimVel(k as u8), &src);
        // grade-1 wall flip on the grid axis whose coordinate IS k; ungridded -> unflipped.
        let v = match axes.iter().position(|&c| c == k) {
            Some(ax) => v * vel_sign[ax],
            None => v,
        };
        writes.push((
            format!("prim_v{k}"),
            FieldRef::PrimVel(k as u8).into(),
            v.node(),
        ));
    }
    let pre = gv_load_at("prim_pre", "prim.pre", &src);
    writes.push(("prim_pre".to_string(), FieldRef::PrimPre.into(), pre.node()));
    (end_trace(), writes)
}

/// the OUTWARD edge->ghost separation used by the prescribed-gradient fills: `sum_ax |centroid(ghost
/// index) - centroid(source index)|`. the source index is the outflow EDGE (map_type >= 3), so a
/// PASSTHROUGH axis (source == cell coord) contributes 0 and a single-axis face pass yields exactly
/// that face's outward distance — a corner (multi-axis) composes the per-axis distances. the
/// centroid is the midpoint of the cell's two faces (uniform or log, via `gv_axis_face_at_index`).
fn gv_outward_dist(ndim: usize, spacing: &[Spacing], src: &[NodeId]) -> Gv {
    let centroid = |ax: usize, i: Gv| -> Gv {
        Gv::from_f64(0.5)
            * (gv_axis_face_at_index(ax, spacing[ax], i)
                + gv_axis_face_at_index(ax, spacing[ax], i + Gv::ONE))
    };
    let mut dist = Gv::ZERO;
    for ax in 0..ndim {
        let ghost = centroid(ax, Gv::coord(ax as u8));
        let edge = centroid(ax, Gv::of(src[ax]));
        dist = dist + (ghost - edge).abs();
    }
    dist
}

/// the NEUMANN lattice-map ghost fill: prescribe the OUTWARD normal derivative `dU/dn = q` per
/// primitive variable. reuses the outflow EDGE source coord (map_type >= 3 -> arg), reads the
/// boundary-adjacent interior value, and extrapolates `U_ghost = u_edge + q * dist` (per-variable
/// coefficient `neu_q_*`, outward separation `dist`). `q = 0` recovers the plain outflow copy, so
/// outflow is the homogeneous member of this family. `ncomp` velocity components, `ndim` gridded
/// axes; the energy regime additionally prescribes `pre`.
pub fn neumann_ghost_fill_gv(
    ndim: usize,
    ncomp: usize,
    has_energy: bool,
    spacing: &[Spacing],
) -> (GvKernel, Vec<(String, FieldBind, NodeId)>) {
    begin_trace();
    let src = gv_lattice_source(ndim);
    let dist = gv_outward_dist(ndim, spacing, &src);
    let neumann =
        |u: Gv, q: &str| symbi_hydro::boundary_term::neumann_ghost(u, Gv::scalar(q), dist);
    let rho = neumann(gv_load_at("prim_rho", "prim.rho", &src), "neu_q_rho");
    let mut writes = vec![("prim_rho".to_string(), FieldRef::PrimRho.into(), rho.node())];
    for k in 0..ncomp {
        let v = neumann(
            gv_load_at(&format!("prim_v{k}"), FieldRef::PrimVel(k as u8), &src),
            &format!("neu_q_v{k}"),
        );
        writes.push((
            format!("prim_v{k}"),
            FieldRef::PrimVel(k as u8).into(),
            v.node(),
        ));
    }
    if has_energy {
        let pre = neumann(gv_load_at("prim_pre", "prim.pre", &src), "neu_q_pre");
        writes.push(("prim_pre".to_string(), FieldRef::PrimPre.into(), pre.node()));
    }
    (end_trace(), writes)
}

/// the ROBIN lattice-map ghost fill: prescribe `a*U_face + b*(dU/dn) = c` per primitive variable at
/// the boundary FACE, with the face midway between the edge cell and the ghost (separation `dist`).
/// reuses the outflow EDGE source; the per-variable coefficients are `rob_{a,b,c}_*`. degenerates to
/// Dirichlet (`b = 0`) and Neumann (`a = 0`) per `symbi_hydro::boundary_term::robin_ghost`.
pub fn robin_ghost_fill_gv(
    ndim: usize,
    ncomp: usize,
    has_energy: bool,
    spacing: &[Spacing],
) -> (GvKernel, Vec<(String, FieldBind, NodeId)>) {
    begin_trace();
    let src = gv_lattice_source(ndim);
    let h = gv_outward_dist(ndim, spacing, &src);
    let robin = |u: Gv, a: &str, b: &str, c: &str| {
        symbi_hydro::boundary_term::robin_ghost(u, Gv::scalar(a), Gv::scalar(b), Gv::scalar(c), h)
    };
    let rho = robin(
        gv_load_at("prim_rho", "prim.rho", &src),
        "rob_a_rho",
        "rob_b_rho",
        "rob_c_rho",
    );
    let mut writes = vec![("prim_rho".to_string(), FieldRef::PrimRho.into(), rho.node())];
    for k in 0..ncomp {
        let v = robin(
            gv_load_at(&format!("prim_v{k}"), FieldRef::PrimVel(k as u8), &src),
            &format!("rob_a_v{k}"),
            &format!("rob_b_v{k}"),
            &format!("rob_c_v{k}"),
        );
        writes.push((
            format!("prim_v{k}"),
            FieldRef::PrimVel(k as u8).into(),
            v.node(),
        ));
    }
    if has_energy {
        let pre = robin(
            gv_load_at("prim_pre", "prim.pre", &src),
            "rob_a_pre",
            "rob_b_pre",
            "rob_c_pre",
        );
        writes.push(("prim_pre".to_string(), FieldRef::PrimPre.into(), pre.node()));
    }
    (end_trace(), writes)
}

/// the SINGLE-SCALAR lattice-map ghost fill: pull back one field "f" at the per-axis
/// integer source coord, times the runtime grade `sign` (+1 for a scalar copy or a
/// tangential staggered component; -1 for a wall-normal component under a reflect
/// map). the staggered `bface` transverse-halo fill dispatches this per component —
/// the field resolves the region's absolute coords against its OWN staggered lo, so
/// the same kernel serves any cell- or face-anchored scalar.
pub fn scalar_ghost_fill_gv(ndim: usize) -> (GvKernel, Vec<(String, FieldBind, NodeId)>) {
    begin_trace();
    let src = gv_lattice_source(ndim);
    let sign = Gv::scalar("sign");
    let v = gv_load_at("f", "f", &src) * sign;
    let writes = vec![("f".to_string(), "f".into(), v.node())];
    (end_trace(), writes)
}

// the per-vector-component wall-map sign: the in-plane components (k < ndim) pick up the
// boundary axis's reflect sign (B/vel are grade-1 vectors under the wall map); the out-of-
// plane components (k >= ndim, e.g., Bz/vz in 1.5D/2.5D) are tangential to every grid-axis
// wall, so they copy unchanged (sign = +1). this is why ghost fill loops 0..ncomp (DOF),
// NOT 0..ndim — else the out-of-plane ghosts stay zero and drain the boundary.
fn gv_ghost_sign(k: usize, ndim: usize, vel_sign: &[Gv]) -> Gv {
    if k < ndim { vel_sign[k] } else { Gv::ONE }
}

/// the RMHD lattice-map ghost fill — `iso_ghost_fill_gv` plus the cell-centered B: pull back
/// rho/vel/pre + `mhd.bcell[k]`, the velocity AND B (DOF-vectors) picking up the per-axis
/// `vel_sign` for in-plane components and copying the out-of-plane ones. `ndim` = grid axes
/// (the lattice source + reflect signs), `ncomp` = vector components (DOF).
pub fn rmhd_ghost_fill_gv(
    ndim: usize,
    ncomp: usize,
) -> (GvKernel, Vec<(String, FieldBind, NodeId)>) {
    begin_trace();
    let src = gv_lattice_source(ndim);
    let vel_sign: Vec<Gv> = (0..ndim)
        .map(|k| Gv::scalar(&format!("vel_sign_{k}")))
        .collect();
    let rho = gv_load_at("prim_rho", "prim.rho", &src);
    let mut writes = vec![("prim_rho".to_string(), FieldRef::PrimRho.into(), rho.node())];
    for k in 0..ncomp {
        let v = gv_load_at(&format!("prim_v{k}"), FieldRef::PrimVel(k as u8), &src)
            * gv_ghost_sign(k, ndim, &vel_sign);
        writes.push((
            format!("prim_v{k}"),
            FieldRef::PrimVel(k as u8).into(),
            v.node(),
        ));
    }
    let pre = gv_load_at("prim_pre", "prim.pre", &src);
    writes.push(("prim_pre".to_string(), FieldRef::PrimPre.into(), pre.node()));
    for k in 0..ncomp {
        let b = gv_load_at(&format!("bcell_{k}"), &format!("mhd.bcell[{k}]"), &src)
            * gv_ghost_sign(k, ndim, &vel_sign);
        writes.push((
            format!("bcell_{k}"),
            format!("mhd.bcell[{k}]").into(),
            b.node(),
        ));
    }
    (end_trace(), writes)
}

/// the ISOTHERMAL lattice-map ghost fill — `rmhd_ghost_fill_gv` minus the `pre` field
/// (isothermal MHD has no pressure to fill). rho + vel + bcell only.
pub fn imhd_ghost_fill_gv(
    ndim: usize,
    ncomp: usize,
) -> (GvKernel, Vec<(String, FieldBind, NodeId)>) {
    begin_trace();
    let src = gv_lattice_source(ndim);
    let vel_sign: Vec<Gv> = (0..ndim)
        .map(|k| Gv::scalar(&format!("vel_sign_{k}")))
        .collect();
    let rho = gv_load_at("prim_rho", "prim.rho", &src);
    let mut writes = vec![("prim_rho".to_string(), FieldRef::PrimRho.into(), rho.node())];
    for k in 0..ncomp {
        let v = gv_load_at(&format!("prim_v{k}"), FieldRef::PrimVel(k as u8), &src)
            * gv_ghost_sign(k, ndim, &vel_sign);
        writes.push((
            format!("prim_v{k}"),
            FieldRef::PrimVel(k as u8).into(),
            v.node(),
        ));
    }
    for k in 0..ncomp {
        let b = gv_load_at(&format!("bcell_{k}"), &format!("mhd.bcell[{k}]"), &src)
            * gv_ghost_sign(k, ndim, &vel_sign);
        writes.push((
            format!("bcell_{k}"),
            format!("mhd.bcell[{k}]").into(),
            b.node(),
        ));
    }
    (end_trace(), writes)
}

/// the SPINNING-KERR lattice-map ghost fill — `iso_ghost_fill_gv` (2D grid, swirl DOF = 3)
/// with the azimuthal ghost copied through the ANGULAR-MOMENTUM variable
/// w = v^phi + (gamma_{r phi}/gamma_{phi phi}) v^r. a frame-dragging
/// state (S_phi = 0) satisfies w = 0 at every radius; a raw v^phi copy plants the source
/// cell's dragging velocity at the ghost's DIFFERENT (r, theta), violating the dragging
/// relation there and generating boundary S_phi at truncation scale. the w copy keeps the
/// pulled-back state on the dragging manifold exactly:
///   v^phi(ghost) = [v^phi(src) + q(src) v^r(src)] - q(ghost) v^r(ghost),
/// with q = gamma_{r phi}/gamma_{phi phi} evaluated at each cell's VOLUME-WEIGHTED centroid
/// (the c2p metric point, so the cellwise cancellation transfers at roundoff) and
/// v^r(ghost) carrying the wall map's vel_sign. q(src) needs the source cell's position, an
/// integer map expression — `gv_axis_face_at_index` evaluates the coordinate map there.
/// reduces to the plain copy when gamma_{r phi} = 0, so it is baked for Kerr only.
pub fn rhd_kerr_ghost_fill_gv(spacing: &[Spacing]) -> (GvKernel, Vec<(String, FieldBind, NodeId)>) {
    use symbi_geometry::{KerrKS, Metric};
    begin_trace();
    let ndim = 2usize;
    let src = gv_lattice_source(ndim);
    let vel_sign: Vec<Gv> = (0..ndim)
        .map(|ax| Gv::scalar(&format!("vel_sign_{ax}")))
        .collect();
    let rho = gv_load_at("prim_rho", "prim.rho", &src);
    let mut writes = vec![("prim_rho".to_string(), FieldRef::PrimRho.into(), rho.node())];
    let v0_src = gv_load_at("prim_v0", FieldRef::PrimVel(0), &src);
    let v0 = v0_src * vel_sign[0];
    writes.push((
        "prim_v0".to_string(),
        FieldRef::PrimVel(0).into(),
        v0.node(),
    ));
    let v1 = gv_load_at("prim_v1", FieldRef::PrimVel(1), &src) * vel_sign[1];
    writes.push((
        "prim_v1".to_string(),
        FieldRef::PrimVel(1).into(),
        v1.node(),
    ));
    // q at the volume-weighted centroid of the cell at integer indices (i, j):
    // r_c = 0.75 (rh^4 - rl^4)/(rh^3 - rl^3), theta_c = [sin - t cos]_{tl}^{th}/(cos tl - cos th).
    let mass = Gv::scalar("schwarzschild_mass");
    let spin = Gv::scalar("kerr_spin");
    let q_at = |i: Gv, j: Gv| -> Gv {
        let rl = gv_axis_face_at_index(0, spacing[0], i);
        let rh = gv_axis_face_at_index(0, spacing[0], i + Gv::ONE);
        let r_c = Gv::from_f64(0.75) * (gv_powi(rh, 4) - gv_powi(rl, 4))
            / (gv_powi(rh, 3) - gv_powi(rl, 3));
        let tl = gv_axis_face_at_index(1, spacing[1], j);
        let th = gv_axis_face_at_index(1, spacing[1], j + Gv::ONE);
        let th_c =
            ((th.sin() - th * th.cos()) - (tl.sin() - tl * tl.cos())) / (tl.cos() - th.cos());
        let m = KerrKS { mass, spin };
        let gm = <KerrKS<Gv> as Metric<Gv, 3>>::spatial_metric(
            &m,
            Tensor::<Gv, 3>::new([r_c, th_c, Gv::ZERO]),
        );
        gm[(0, 2)] / gm[(2, 2)]
    };
    let v2_src = gv_load_at("prim_v2", FieldRef::PrimVel(2), &src);
    let w_src = v2_src + q_at(Gv::of(src[0]), Gv::of(src[1])) * v0_src;
    let v2 = w_src - q_at(Gv::coord(0), Gv::coord(1)) * v0;
    writes.push((
        "prim_v2".to_string(),
        FieldRef::PrimVel(2).into(),
        v2.node(),
    ));
    let pre = gv_load_at("prim_pre", "prim.pre", &src);
    writes.push(("prim_pre".to_string(), FieldRef::PrimPre.into(), pre.node()));
    (end_trace(), writes)
}

/// the SPINNING-KERR MHD lattice-map ghost fill — `rhd_kerr_ghost_fill_gv` (the velocity
/// w = v^phi + q v^r copy) PLUS the cell-centered B, whose out-of-plane component gets the SAME
/// frame-dragging treatment: the covariant B_phi = gamma_{phi phi} B^phi + gamma_{phi r} B^r is the
/// magnetic angular-momentum density, so a B_phi = 0 state satisfies w_B = B^phi + q B^r = 0 at every
/// radius (q = gamma_{r phi}/gamma_{phi phi}). copying B^phi raw plants the source cell's dragging
/// profile at the ghost's DIFFERENT (r, theta) and generates a boundary B_phi (a spurious toroidal-
/// field / azimuthal-tension source) at truncation; the w_B copy keeps the pulled-back B on the
/// dragging manifold exactly:
///   B^phi(ghost) = [B^phi(src) + q(src) B^r(src)] - q(ghost) B^r(ghost).
/// the in-plane B^r/B^theta pick up the wall map's vel_sign like the velocity; q is at the same
/// volume-weighted centroid the velocity copy + the c2p use. reduces to the plain copy at
/// gamma_{r phi} = 0, so it is baked for Kerr only. DOF = 3 swirl (2D grid).
pub fn rmhd_kerr_ghost_fill_gv(
    spacing: &[Spacing],
) -> (GvKernel, Vec<(String, FieldBind, NodeId)>) {
    use symbi_geometry::{KerrKS, Metric};
    begin_trace();
    let ndim = 2usize;
    let src = gv_lattice_source(ndim);
    let vel_sign: Vec<Gv> = (0..ndim)
        .map(|ax| Gv::scalar(&format!("vel_sign_{ax}")))
        .collect();
    let mass = Gv::scalar("schwarzschild_mass");
    let spin = Gv::scalar("kerr_spin");
    let q_at = |i: Gv, j: Gv| -> Gv {
        let rl = gv_axis_face_at_index(0, spacing[0], i);
        let rh = gv_axis_face_at_index(0, spacing[0], i + Gv::ONE);
        let r_c = Gv::from_f64(0.75) * (gv_powi(rh, 4) - gv_powi(rl, 4))
            / (gv_powi(rh, 3) - gv_powi(rl, 3));
        let tl = gv_axis_face_at_index(1, spacing[1], j);
        let th = gv_axis_face_at_index(1, spacing[1], j + Gv::ONE);
        let th_c =
            ((th.sin() - th * th.cos()) - (tl.sin() - tl * tl.cos())) / (tl.cos() - th.cos());
        let gm = <KerrKS<Gv> as Metric<Gv, 3>>::spatial_metric(
            &KerrKS { mass, spin },
            Tensor::<Gv, 3>::new([r_c, th_c, Gv::ZERO]),
        );
        gm[(0, 2)] / gm[(2, 2)]
    };
    let q_src = q_at(Gv::of(src[0]), Gv::of(src[1]));
    let q_gh = q_at(Gv::coord(0), Gv::coord(1));

    let rho = gv_load_at("prim_rho", "prim.rho", &src);
    let mut writes = vec![("prim_rho".to_string(), FieldRef::PrimRho.into(), rho.node())];
    // velocity: v^r/v^theta reflect, v^phi via the angular-momentum variable w = v^phi + q v^r.
    let v0_src = gv_load_at("prim_v0", FieldRef::PrimVel(0), &src);
    let v0 = v0_src * vel_sign[0];
    writes.push((
        "prim_v0".to_string(),
        FieldRef::PrimVel(0).into(),
        v0.node(),
    ));
    let v1 = gv_load_at("prim_v1", FieldRef::PrimVel(1), &src) * vel_sign[1];
    writes.push((
        "prim_v1".to_string(),
        FieldRef::PrimVel(1).into(),
        v1.node(),
    ));
    let v2_src = gv_load_at("prim_v2", FieldRef::PrimVel(2), &src);
    let v2 = (v2_src + q_src * v0_src) - q_gh * v0;
    writes.push((
        "prim_v2".to_string(),
        FieldRef::PrimVel(2).into(),
        v2.node(),
    ));
    let pre = gv_load_at("prim_pre", "prim.pre", &src);
    writes.push(("prim_pre".to_string(), FieldRef::PrimPre.into(), pre.node()));
    // cell B: B^r/B^theta reflect, B^phi via w_B = B^phi + q B^r (the magnetic dragging manifold).
    let b0_src = gv_load_at("bcell_0", "mhd.bcell[0]", &src);
    let b0 = b0_src * vel_sign[0];
    writes.push(("bcell_0".to_string(), "mhd.bcell[0]".into(), b0.node()));
    let b1 = gv_load_at("bcell_1", "mhd.bcell[1]", &src) * vel_sign[1];
    writes.push(("bcell_1".to_string(), "mhd.bcell[1]".into(), b1.node()));
    let b2_src = gv_load_at("bcell_2", "mhd.bcell[2]", &src);
    let b2 = (b2_src + q_src * b0_src) - q_gh * b0;
    writes.push(("bcell_2".to_string(), "mhd.bcell[2]".into(), b2.node()));
    (end_trace(), writes)
}
