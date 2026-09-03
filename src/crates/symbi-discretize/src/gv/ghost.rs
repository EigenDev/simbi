// =============================================================================
// ghost.rs
//
// lattice-map ghost-fill kernel builders (the boundary pullback).
// =============================================================================

use super::*;
use symbi_ir::{KernelProgram, KernelWrite, trace_kernel};

/// the isothermal lattice-map ghost fill — pull back rho/vel/pre at the per-axis source coord,
/// write in place; the velocity component whose coordinate is a grid axis picks up that axis's
/// wall-normal `vel_sign` (an ungridded swirl coordinate keeps its sign, having no wall
/// map). rho/pre are grade-0 copies. `ncomp` velocity components, `ndim` gridded axes;
/// `axes[d]` = the coord of grid axis d. the EOS-generic 3-field pullback the
/// iso/newton/rhd ghost fill share.
pub fn iso_ghost_fill_gv(ndim: usize, ncomp: usize, axes: &[usize]) -> KernelProgram {
    trace_kernel(|cx| {
        let src = gv_lattice_source(cx, ndim);
        let vel_sign: Vec<Gv> = (0..ndim)
            .map(|ax| cx.scalar(&format!("vel_sign_{ax}")))
            .collect();
        let rho = gv_load_at(cx, "prim_rho", "prim.rho", &src);
        let mut writes = vec![KernelWrite::new("prim_rho", FieldRef::PrimRho, rho.node())];
        for k in 0..ncomp {
            let v = gv_load_at(cx, &format!("prim_v{k}"), FieldRef::PrimVel(k as u8), &src);
            // grade-1 wall flip on the grid axis whose coordinate is k; ungridded keeps its sign.
            let v = match axes.iter().position(|&c| c == k) {
                Some(ax) => v * vel_sign[ax],
                None => v,
            };
            writes.push(KernelWrite::new(
                format!("prim_v{k}"),
                FieldRef::PrimVel(k as u8),
                v.node(),
            ));
        }
        let pre = gv_load_at(cx, "prim_pre", "prim.pre", &src);
        writes.push(KernelWrite::new("prim_pre", FieldRef::PrimPre, pre.node()));
        writes
    })
}

/// the outward edge->ghost separation used by the prescribed-gradient fills: `sum_ax |centroid(ghost
/// index) - centroid(source index)|`. the source index is the outflow edge (map_type >= 3), so a
/// passthrough axis (source == cell coord) contributes 0 and a single-axis face pass yields exactly
/// that face's outward distance — a corner (multi-axis) composes the per-axis distances. the
/// centroid is the midpoint of the cell's two faces (uniform or log, via `gv_axis_face_at_index`).
fn gv_outward_dist<'t>(
    cx: TraceCx<'t>,
    ndim: usize,
    spacing: &[Spacing],
    src: &[NodeId],
) -> Gv<'t> {
    let centroid = |ax: usize, i| {
        Gv::from_f64(0.5)
            * (gv_axis_face_at_index(cx, ax, spacing[ax], i)
                + gv_axis_face_at_index(cx, ax, spacing[ax], i + Gv::ONE))
    };
    let mut dist = Gv::ZERO;
    for ax in 0..ndim {
        let ghost = centroid(ax, cx.coord(ax as u8));
        let edge = centroid(ax, cx.gv(src[ax]));
        dist = dist + (ghost - edge).abs();
    }
    dist
}

/// the neumann lattice-map ghost fill: prescribe the outward normal derivative `dU/dn = q` per
/// primitive variable. reuses the outflow edge source coord (map_type >= 3 -> arg), reads the
/// boundary-adjacent interior value, and extrapolates `U_ghost = u_edge + q * dist` (per-variable
/// coefficient `neu_q_*`, outward separation `dist`). `q = 0` recovers the plain outflow copy, so
/// outflow is the homogeneous member of this family. `ncomp` velocity components, `ndim` gridded
/// axes; the energy regime additionally prescribes `pre`.
pub fn neumann_ghost_fill_gv(
    ndim: usize,
    ncomp: usize,
    has_energy: bool,
    spacing: &[Spacing],
) -> KernelProgram {
    trace_kernel(|cx| {
        let src = gv_lattice_source(cx, ndim);
        let dist = gv_outward_dist(cx, ndim, spacing, &src);
        let neumann = |u, q: &str| symbi_hydro::boundary_term::neumann_ghost(u, cx.scalar(q), dist);
        let rho = neumann(gv_load_at(cx, "prim_rho", "prim.rho", &src), "neu_q_rho");
        let mut writes = vec![KernelWrite::new("prim_rho", FieldRef::PrimRho, rho.node())];
        for k in 0..ncomp {
            let v = neumann(
                gv_load_at(cx, &format!("prim_v{k}"), FieldRef::PrimVel(k as u8), &src),
                &format!("neu_q_v{k}"),
            );
            writes.push(KernelWrite::new(
                format!("prim_v{k}"),
                FieldRef::PrimVel(k as u8),
                v.node(),
            ));
        }
        if has_energy {
            let pre = neumann(gv_load_at(cx, "prim_pre", "prim.pre", &src), "neu_q_pre");
            writes.push(KernelWrite::new("prim_pre", FieldRef::PrimPre, pre.node()));
        }
        writes
    })
}

/// the robin lattice-map ghost fill: prescribe `a*U_face + b*(dU/dn) = c` per primitive variable at
/// the boundary face, with the face midway between the edge cell and the ghost (separation `dist`).
/// reuses the outflow edge source; the per-variable coefficients are `rob_{a,b,c}_*`. degenerates to
/// dirichlet (`b = 0`) and neumann (`a = 0`) per `symbi_hydro::boundary_term::robin_ghost`.
pub fn robin_ghost_fill_gv(
    ndim: usize,
    ncomp: usize,
    has_energy: bool,
    spacing: &[Spacing],
) -> KernelProgram {
    trace_kernel(|cx| {
        let src = gv_lattice_source(cx, ndim);
        let h = gv_outward_dist(cx, ndim, spacing, &src);
        let robin = |u, a: &str, b: &str, c: &str| {
            symbi_hydro::boundary_term::robin_ghost(u, cx.scalar(a), cx.scalar(b), cx.scalar(c), h)
        };
        let rho = robin(
            gv_load_at(cx, "prim_rho", "prim.rho", &src),
            "rob_a_rho",
            "rob_b_rho",
            "rob_c_rho",
        );
        let mut writes = vec![KernelWrite::new("prim_rho", FieldRef::PrimRho, rho.node())];
        for k in 0..ncomp {
            let v = robin(
                gv_load_at(cx, &format!("prim_v{k}"), FieldRef::PrimVel(k as u8), &src),
                &format!("rob_a_v{k}"),
                &format!("rob_b_v{k}"),
                &format!("rob_c_v{k}"),
            );
            writes.push(KernelWrite::new(
                format!("prim_v{k}"),
                FieldRef::PrimVel(k as u8),
                v.node(),
            ));
        }
        if has_energy {
            let pre = robin(
                gv_load_at(cx, "prim_pre", "prim.pre", &src),
                "rob_a_pre",
                "rob_b_pre",
                "rob_c_pre",
            );
            writes.push(KernelWrite::new("prim_pre", FieldRef::PrimPre, pre.node()));
        }
        writes
    })
}

/// the single-scalar lattice-map ghost fill: pull back one field "f" at the per-axis
/// integer source coord, times the runtime grade `sign` (+1 for a scalar copy or a
/// tangential staggered component; -1 for a wall-normal component under a reflect
/// map). the staggered `bface` transverse-halo fill dispatches this per component —
/// the field resolves the region's absolute coords against its own staggered lo, so
/// the same kernel serves any cell- or face-anchored scalar.
pub fn scalar_ghost_fill_gv(ndim: usize) -> KernelProgram {
    trace_kernel(|cx| {
        let src = gv_lattice_source(cx, ndim);
        let sign = cx.scalar("sign");
        let v = gv_load_at(cx, "f", "f", &src) * sign;
        let writes = vec![KernelWrite::new("f", "f", v.node())];
        writes
    })
}

// the per-vector-component wall-map sign: the in-plane components (k < ndim) pick up the
// boundary axis's reflect sign (B/vel are grade-1 vectors under the wall map); the out-of-
// plane components (k >= ndim, e.g., Bz/vz in 1.5D/2.5D) are tangential to every grid-axis
// wall, so they copy unchanged (sign = +1). this is why ghost fill loops 0..ncomp (DOF):
// a 0..ndim loop leaves the out-of-plane ghosts at zero, which drains the boundary.
fn gv_ghost_sign<'t>(k: usize, ndim: usize, vel_sign: &[Gv<'t>]) -> Gv<'t> {
    if k < ndim { vel_sign[k] } else { Gv::ONE }
}

/// the RMHD lattice-map ghost fill — `iso_ghost_fill_gv` plus the cell-centered B: pull back
/// rho/vel/pre + `mhd.bcell[k]`, the velocity and B (DOF-vectors) picking up the per-axis
/// `vel_sign` for in-plane components and copying the out-of-plane ones. `ndim` = grid axes
/// (the lattice source + reflect signs), `ncomp` = vector components (DOF).
pub fn rmhd_ghost_fill_gv(ndim: usize, ncomp: usize) -> KernelProgram {
    trace_kernel(|cx| {
        let src = gv_lattice_source(cx, ndim);
        let vel_sign: Vec<Gv> = (0..ndim)
            .map(|k| cx.scalar(&format!("vel_sign_{k}")))
            .collect();
        let rho = gv_load_at(cx, "prim_rho", "prim.rho", &src);
        let mut writes = vec![KernelWrite::new("prim_rho", FieldRef::PrimRho, rho.node())];
        for k in 0..ncomp {
            let v = gv_load_at(cx, &format!("prim_v{k}"), FieldRef::PrimVel(k as u8), &src)
                * gv_ghost_sign(k, ndim, &vel_sign);
            writes.push(KernelWrite::new(
                format!("prim_v{k}"),
                FieldRef::PrimVel(k as u8),
                v.node(),
            ));
        }
        let pre = gv_load_at(cx, "prim_pre", "prim.pre", &src);
        writes.push(KernelWrite::new("prim_pre", FieldRef::PrimPre, pre.node()));
        for k in 0..ncomp {
            let b = gv_load_at(cx, &format!("bcell_{k}"), &format!("mhd.bcell[{k}]"), &src)
                * gv_ghost_sign(k, ndim, &vel_sign);
            writes.push(KernelWrite::new(
                format!("bcell_{k}"),
                format!("mhd.bcell[{k}]"),
                b.node(),
            ));
        }
        writes
    })
}

/// the isothermal lattice-map ghost fill — the `rmhd_ghost_fill_gv` field set at the
/// isothermal state: rho + vel + bcell, the pressure coming from the closure.
pub fn imhd_ghost_fill_gv(ndim: usize, ncomp: usize) -> KernelProgram {
    trace_kernel(|cx| {
        let src = gv_lattice_source(cx, ndim);
        let vel_sign: Vec<Gv> = (0..ndim)
            .map(|k| cx.scalar(&format!("vel_sign_{k}")))
            .collect();
        let rho = gv_load_at(cx, "prim_rho", "prim.rho", &src);
        let mut writes = vec![KernelWrite::new("prim_rho", FieldRef::PrimRho, rho.node())];
        for k in 0..ncomp {
            let v = gv_load_at(cx, &format!("prim_v{k}"), FieldRef::PrimVel(k as u8), &src)
                * gv_ghost_sign(k, ndim, &vel_sign);
            writes.push(KernelWrite::new(
                format!("prim_v{k}"),
                FieldRef::PrimVel(k as u8),
                v.node(),
            ));
        }
        for k in 0..ncomp {
            let b = gv_load_at(cx, &format!("bcell_{k}"), &format!("mhd.bcell[{k}]"), &src)
                * gv_ghost_sign(k, ndim, &vel_sign);
            writes.push(KernelWrite::new(
                format!("bcell_{k}"),
                format!("mhd.bcell[{k}]"),
                b.node(),
            ));
        }
        writes
    })
}

/// the spinning-kerr lattice-map ghost fill — `iso_ghost_fill_gv` (2D grid, swirl DOF = 3)
/// with the azimuthal ghost copied through the angular-momentum variable
/// w = v^phi + (gamma_{r phi}/gamma_{phi phi}) v^r. a frame-dragging
/// state (S_phi = 0) satisfies w = 0 at every radius; a raw v^phi copy plants the source
/// cell's dragging velocity at the ghost's own (r, theta), violating the dragging
/// relation there and generating boundary S_phi at truncation scale. the w copy keeps the
/// pulled-back state on the dragging manifold exactly:
///   v^phi(ghost) = [v^phi(src) + q(src) v^r(src)] - q(ghost) v^r(ghost),
/// with q = gamma_{r phi}/gamma_{phi phi} evaluated at each cell's volume-weighted centroid
/// (the c2p metric point, so the cellwise cancellation transfers at roundoff) and
/// v^r(ghost) carrying the wall map's vel_sign. q(src) needs the source cell's position, an
/// integer map expression — `gv_axis_face_at_index` evaluates the coordinate map there.
/// reduces to the plain copy when gamma_{r phi} = 0, so it is baked for kerr only.
pub fn rhd_kerr_ghost_fill_gv(spacing: &[Spacing]) -> KernelProgram {
    use symbi_geometry::{KerrKS, Metric};
    trace_kernel(|cx| {
        let ndim = 2usize;
        let src = gv_lattice_source(cx, ndim);
        let vel_sign: Vec<Gv> = (0..ndim)
            .map(|ax| cx.scalar(&format!("vel_sign_{ax}")))
            .collect();
        let rho = gv_load_at(cx, "prim_rho", "prim.rho", &src);
        let mut writes = vec![KernelWrite::new("prim_rho", FieldRef::PrimRho, rho.node())];
        let v0_src = gv_load_at(cx, "prim_v0", FieldRef::PrimVel(0), &src);
        let v0 = v0_src * vel_sign[0];
        writes.push(KernelWrite::new("prim_v0", FieldRef::PrimVel(0), v0.node()));
        let v1 = gv_load_at(cx, "prim_v1", FieldRef::PrimVel(1), &src) * vel_sign[1];
        writes.push(KernelWrite::new("prim_v1", FieldRef::PrimVel(1), v1.node()));
        // q at the volume-weighted centroid of the cell at integer indices (i, j):
        // r_c = 0.75 (rh^4 - rl^4)/(rh^3 - rl^3), theta_c = [sin - t cos]_{tl}^{th}/(cos tl - cos th).
        let mass = cx.scalar("schwarzschild_mass");
        let spin = cx.scalar("kerr_spin");
        let q_at = |i, j| {
            let rl = gv_axis_face_at_index(cx, 0, spacing[0], i);
            let rh = gv_axis_face_at_index(cx, 0, spacing[0], i + Gv::ONE);
            // the volume-weighted centroid, the same text `cell_geometry_gv` evaluates:
            // the c2p inverted the metric at that centroid, and the zero-angular-momentum
            // cancellation transfers to the stencil when the coefficient is evaluated at
            // that bit-identical position.
            let r_c = symbi_geometry::volume_weighted_centroid(
                symbi_geometry::Geometry::Spherical,
                0,
                rl,
                rh,
            );
            let tl = gv_axis_face_at_index(cx, 1, spacing[1], j);
            let th = gv_axis_face_at_index(cx, 1, spacing[1], j + Gv::ONE);
            let th_c = symbi_geometry::volume_weighted_centroid(
                symbi_geometry::Geometry::Spherical,
                1,
                tl,
                th,
            );
            let m = KerrKS { mass, spin };
            let gm = <KerrKS<Gv> as Metric<Gv, 3>>::spatial_metric(
                &m,
                Tensor::<Gv, 3>::new([r_c, th_c, Gv::ZERO]),
            );
            gm[(0, 2)] / gm[(2, 2)]
        };
        let v2_src = gv_load_at(cx, "prim_v2", FieldRef::PrimVel(2), &src);
        let w_src = v2_src + q_at(cx.gv(src[0]), cx.gv(src[1])) * v0_src;
        let v2 = w_src - q_at(cx.coord(0), cx.coord(1)) * v0;
        writes.push(KernelWrite::new("prim_v2", FieldRef::PrimVel(2), v2.node()));
        let pre = gv_load_at(cx, "prim_pre", "prim.pre", &src);
        writes.push(KernelWrite::new("prim_pre", FieldRef::PrimPre, pre.node()));
        writes
    })
}

/// the spinning-kerr MHD lattice-map ghost fill — `rhd_kerr_ghost_fill_gv` (the velocity
/// w = v^phi + q v^r copy) plus the cell-centered B, whose out-of-plane component gets the same
/// frame-dragging treatment: the covariant B_phi = gamma_{phi phi} B^phi + gamma_{phi r} B^r is the
/// magnetic angular-momentum density, so a B_phi = 0 state satisfies w_B = B^phi + q B^r = 0 at every
/// radius (q = gamma_{r phi}/gamma_{phi phi}). copying B^phi raw plants the source cell's dragging
/// profile at the ghost's own (r, theta) and generates a boundary B_phi (a spurious toroidal-
/// field / azimuthal-tension source) at truncation; the w_B copy keeps the pulled-back B on the
/// dragging manifold exactly:
///   B^phi(ghost) = [B^phi(src) + q(src) B^r(src)] - q(ghost) B^r(ghost).
/// the in-plane B^r/B^theta pick up the wall map's vel_sign like the velocity; q is at the same
/// volume-weighted centroid the velocity copy + the c2p use. reduces to the plain copy at
/// gamma_{r phi} = 0, so it is baked for kerr only. DOF = 3 swirl (2D grid).
pub fn rmhd_kerr_ghost_fill_gv(spacing: &[Spacing]) -> KernelProgram {
    use symbi_geometry::{KerrKS, Metric};
    trace_kernel(|cx| {
        let ndim = 2usize;
        let src = gv_lattice_source(cx, ndim);
        let vel_sign: Vec<Gv> = (0..ndim)
            .map(|ax| cx.scalar(&format!("vel_sign_{ax}")))
            .collect();
        let mass = cx.scalar("schwarzschild_mass");
        let spin = cx.scalar("kerr_spin");
        let q_at = |i, j| {
            let rl = gv_axis_face_at_index(cx, 0, spacing[0], i);
            let rh = gv_axis_face_at_index(cx, 0, spacing[0], i + Gv::ONE);
            // the volume-weighted centroid, the same text `cell_geometry_gv` evaluates:
            // the c2p inverted the metric at that centroid, and the zero-angular-momentum
            // cancellation transfers to the stencil when the coefficient is evaluated at
            // that bit-identical position.
            let r_c = symbi_geometry::volume_weighted_centroid(
                symbi_geometry::Geometry::Spherical,
                0,
                rl,
                rh,
            );
            let tl = gv_axis_face_at_index(cx, 1, spacing[1], j);
            let th = gv_axis_face_at_index(cx, 1, spacing[1], j + Gv::ONE);
            let th_c = symbi_geometry::volume_weighted_centroid(
                symbi_geometry::Geometry::Spherical,
                1,
                tl,
                th,
            );
            let gm = <KerrKS<Gv> as Metric<Gv, 3>>::spatial_metric(
                &KerrKS { mass, spin },
                Tensor::<Gv, 3>::new([r_c, th_c, Gv::ZERO]),
            );
            gm[(0, 2)] / gm[(2, 2)]
        };
        let q_src = q_at(cx.gv(src[0]), cx.gv(src[1]));
        let q_gh = q_at(cx.coord(0), cx.coord(1));

        let rho = gv_load_at(cx, "prim_rho", "prim.rho", &src);
        let mut writes = vec![KernelWrite::new("prim_rho", FieldRef::PrimRho, rho.node())];
        // velocity: v^r/v^theta reflect, v^phi via the angular-momentum variable w = v^phi + q v^r.
        let v0_src = gv_load_at(cx, "prim_v0", FieldRef::PrimVel(0), &src);
        let v0 = v0_src * vel_sign[0];
        writes.push(KernelWrite::new("prim_v0", FieldRef::PrimVel(0), v0.node()));
        let v1 = gv_load_at(cx, "prim_v1", FieldRef::PrimVel(1), &src) * vel_sign[1];
        writes.push(KernelWrite::new("prim_v1", FieldRef::PrimVel(1), v1.node()));
        let v2_src = gv_load_at(cx, "prim_v2", FieldRef::PrimVel(2), &src);
        let v2 = (v2_src + q_src * v0_src) - q_gh * v0;
        writes.push(KernelWrite::new("prim_v2", FieldRef::PrimVel(2), v2.node()));
        let pre = gv_load_at(cx, "prim_pre", "prim.pre", &src);
        writes.push(KernelWrite::new("prim_pre", FieldRef::PrimPre, pre.node()));
        // cell B: B^r/B^theta reflect, B^phi via w_B = B^phi + q B^r (the magnetic dragging manifold).
        let b0_src = gv_load_at(cx, "bcell_0", "mhd.bcell[0]", &src);
        let b0 = b0_src * vel_sign[0];
        writes.push(KernelWrite::new("bcell_0", "mhd.bcell[0]", b0.node()));
        let b1 = gv_load_at(cx, "bcell_1", "mhd.bcell[1]", &src) * vel_sign[1];
        writes.push(KernelWrite::new("bcell_1", "mhd.bcell[1]", b1.node()));
        let b2_src = gv_load_at(cx, "bcell_2", "mhd.bcell[2]", &src);
        let b2 = (b2_src + q_src * b0_src) - q_gh * b0;
        writes.push(KernelWrite::new("bcell_2", "mhd.bcell[2]", b2.node()));
        writes
    })
}

/// the well-balanced lattice-map ghost fill, per chart: the velocity pulls back with the
/// wall-normal `vel_sign` flip exactly as the plain fill, and density and pressure are
/// extended along the local isentrope from the source cell to the ghost position,
///
///   (rho, p)_ghost = LocalEquilibrium::through((rho, p)_src, phi_src).state_at(phi_ghost),
///
/// with `phi` the total body potential at each cell's position — the runtime spacing map's
/// own cell centers (geometric mean of the faces on a log axis, arithmetic midpoint
/// otherwise), the same anchor ladder the balanced reconstruction evaluates, mapped to
/// cartesian through the chart embedding on a curvilinear grid. the continuation of a
/// stratified column is its hydrostatic extension, so a plain reflect ghost presents
/// the balanced reconstruction with departures that are pure boundary artifact -- measured
/// as a 1.5e-2 entropy-floor loss on a sealed column that the interior scheme holds to
/// 2.2e-8. the extension makes the wall face balanced by construction, the same statement
/// the interior reconstruction makes at every other face.
///
/// the extension is exact wherever the source and ghost potentials coincide: a skip axis
/// (source == cell) gives `phi_src == phi_ghost` as identical traced nodes, the enthalpy
/// ratio is exactly one, and the fill reduces to the plain pullback bit-for-bit. on an
/// outflow edge it extends the column hydrostatically, which is the well-balanced outflow
/// fill (the plain fill flat-copies). a periodic cut across an asymmetric potential would be
/// mis-extended, and dispatch refuses it.
pub fn wb_ghost_fill_gv(
    ndim: usize,
    ncomp: usize,
    axes: &[usize],
    n_bodies: usize,
    coords: Coords,
) -> KernelProgram {
    use symbi_hydro::hydrostatic::LocalEquilibrium;
    trace_kernel(|cx| {
        let src = gv_lattice_source(cx, ndim);
        let vel_sign: Vec<Gv> = (0..ndim)
            .map(|ax| cx.scalar(&format!("vel_sign_{ax}")))
            .collect();
        // the bake-time spacing enum is vestigial: face positions and the cell center both come
        // from the runtime per-axis map (`map_kind_{ax}`), so this one kernel serves every
        // grading. the center is the map's own (geometric mean on a log axis, arithmetic midpoint
        // otherwise) — the same position `set_initial` seeds at and the balanced reconstruction's
        // potential ladder anchors on, which is what makes the wall-face extension exact on the
        // seeded column.
        let spacing = vec![Spacing::Uniform; ndim];
        let centroid = |ax: usize, i| {
            let lo = gv_axis_face_at_index(cx, ax, spacing[ax], i);
            let hi = gv_axis_face_at_index(cx, ax, spacing[ax], i + Gv::ONE);
            crate::gv::gv_axis_center_between(cx, ax, lo, hi)
        };
        let (phi_src, phi_ghost) = match coords {
            Coords::Cartesian => {
                // cartesian: grid axis positions are the cartesian coordinates; ungridded components 0.
                let mut ghost_pos = [Gv::ZERO, Gv::ZERO, Gv::ZERO];
                let mut src_pos = [Gv::ZERO, Gv::ZERO, Gv::ZERO];
                for (g, &coord_idx) in axes.iter().enumerate().take(ndim) {
                    if coord_idx < 3 {
                        ghost_pos[coord_idx] = centroid(g, cx.coord(g as u8));
                        src_pos[coord_idx] = centroid(g, cx.gv(src[g]));
                    }
                }
                let phi_at = |pos: &[_; 3]| {
                    (0..n_bodies)
                        .map(|b| {
                            let mut bpos = [Gv::ZERO, Gv::ZERO, Gv::ZERO];
                            for (g, &coord_idx) in axes.iter().enumerate().take(ndim) {
                                if coord_idx < 3 {
                                    bpos[coord_idx] = cx.scalar(&format!("body_{b}_pos_{g}"));
                                }
                            }
                            let rvec: [Gv; 3] = std::array::from_fn(|i| pos[i] - bpos[i]);
                            crate::ibm::body_potential(
                                rvec,
                                cx.scalar(&format!("body_{b}_mass")),
                                cx.scalar(&format!("body_{b}_soft")),
                                cx.scalar(&format!("body_{b}_softkind")),
                            )
                        })
                        .sum::<Gv>()
                };
                let phi_src = phi_at(&src_pos);
                let phi_ghost = phi_at(&ghost_pos);
                (phi_src, phi_ghost)
            }
            // curvilinear: the per-axis midpoints are chart coordinates (r, theta, ...); the
            // potential is evaluated at their cartesian embedding, against body positions on
            // the chart's grid-plane cartesian axes — the same convention the balanced
            // reconstruction's potential ladder and the wb body source use, so the wall face
            // the extension constructs is the one the interior scheme balances against.
            _ => {
                let cart_axes = crate::gv_immersed::body_cart_axes(coords, ndim, axes);
                let mut ghost_c3 = [Gv::ZERO, Gv::ZERO, Gv::ZERO];
                let mut src_c3 = [Gv::ZERO, Gv::ZERO, Gv::ZERO];
                for (g, &coord_idx) in axes.iter().enumerate().take(ndim) {
                    if coord_idx < 3 {
                        ghost_c3[coord_idx] = centroid(g, cx.coord(g as u8));
                        src_c3[coord_idx] = centroid(g, cx.gv(src[g]));
                    }
                }
                let phi_at = |coord3: &[_; 3]| {
                    let pos = crate::gv_immersed::to_cartesian_gv(coords, coord3);
                    (0..n_bodies)
                        .map(|b| {
                            let bpos =
                                crate::gv_immersed::body_vec3(cx, b, ndim, &cart_axes, "pos");
                            let rvec: [Gv; 3] = std::array::from_fn(|i| pos[i] - bpos[i]);
                            crate::ibm::body_potential(
                                rvec,
                                cx.scalar(&format!("body_{b}_mass")),
                                cx.scalar(&format!("body_{b}_soft")),
                                cx.scalar(&format!("body_{b}_softkind")),
                            )
                        })
                        .sum::<Gv>()
                };
                (phi_at(&src_c3), phi_at(&ghost_c3))
            }
        };

        let rho_src = gv_load_at(cx, "prim_rho", "prim.rho", &src);
        let pre_src = gv_load_at(cx, "prim_pre", "prim.pre", &src);
        // the mechanical extension: the ghost inherits the source cell's density (the
        // equilibrium density is piecewise constant) and its pressure follows the one
        // segment `p_src + rho_src (phi_src - phi_ghost)`. the extension spans one known
        // excursion, so the validity weight comes from that excursion itself: a ghost
        // sitting above the source by more of the segment's positive domain than it
        // carries is placed by a faded profile, and at zero weight the fill is the plain
        // pullback of the source state — the continuation an unstratified column has.
        let rise = symbi_hydro::hydrostatic::potential_rise(phi_src, phi_ghost, phi_ghost);
        let eq = LocalEquilibrium::faded(rho_src, pre_src, phi_src, rise);
        let (rho_g, pre_g) = eq.state_at(phi_ghost);

        let mut writes = vec![KernelWrite::new(
            "prim_rho",
            FieldRef::PrimRho,
            rho_g.node(),
        )];
        for k in 0..ncomp {
            let v = gv_load_at(cx, &format!("prim_v{k}"), FieldRef::PrimVel(k as u8), &src);
            let v = match axes.iter().position(|&c| c == k) {
                Some(ax) => v * vel_sign[ax],
                None => v,
            };
            writes.push(KernelWrite::new(
                format!("prim_v{k}"),
                FieldRef::PrimVel(k as u8),
                v.node(),
            ));
        }
        writes.push(KernelWrite::new(
            "prim_pre",
            FieldRef::PrimPre,
            pre_g.node(),
        ));
        writes
    })
}
