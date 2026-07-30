// =============================================================================
// gv_penalize.rs
//
// the traced immersed-boundary penalization kernel:
// per cell, the sphere SDF's signed distance -> the mollified chi -> the
// property stack's Relax accumulation -> the SAME carrier-generic
// `penalize_cell` that runs at f64, evaluated at Gv. the [Drain] stack is
// the p = 1 anchor: chi/tau on the rho channel only, every other channel's
// correction an exact arithmetic zero, so the kernel reduces bit-for-bit to
// `drain_cell`'s uniform scaling.
//
// buffers: cons (den, mom_0.., nrg) IN PLACE, plus per-body delta scratch
// (pen_0_mass, pen_0_force_{ax}, pen_0_energy) for the feedback reduction.
// scalars: dt, gamma, the grid (x_lo/dx/map_kind per axis), body_0_pos_*,
// body_0_racc (the mask radius), and the open spec knob `c_drain`
// (tau = c_drain dx / c_s, the convergence dial — never tuned to a rate).
// the mask width is one minimum cell. outputs declare the support ball
// body_0_pos +- (body_0_racc + DRAIN_SUPPORT_WIDTHS min dx): beyond it tanh
// saturation makes chi exactly zero and the update an exact no-op.
//
// usage (build.rs):
//   let (k, writes) = penalize_drain_gv(ndim);
//   emit_gv(out, KernelId::PenalizeDrain { ndim }.name(), ndim, &k, &writes);
// =============================================================================

use symbi_algebra::algebra::Numeric;
use symbi_algebra::{Embedded, Physical, Tensor};
use symbi_geometry::{Cylindrical, CylindricalRPhi, DiagonalMetric, Metric, Spherical};
use symbi_hydro::energy::{Adiabatic, Dyed};
use symbi_hydro::state::ConsG;
use symbi_ib::penalize::{BodyKin, Property, Relax, penalize_cell};
use symbi_ib::sdf::SdfExpr;
use symbi_ir::algebra::Scalar;
use symbi_ir::gv::Writes;
use symbi_ir::{Gv, GvKernel, ParamExpr};

/// the wall/drain relaxation signal speed: the FAST MAGNETOSONIC speed `sqrt(c_s^2 + c_a^2)`,
/// with `c_a^2 = |B|^2 / rho` bound as the runtime `c_a2` scalar (the max over the interior, so the
/// wall stays a signal-crossing stiff in the low-beta regions a magnetized sink accumulates).
/// `c_a2 = 0` off MHD reduces it to the sound speed exactly, so a hydro run is unchanged.
fn signal_speed(cs: Gv) -> Gv {
    (cs * cs + Gv::scalar("c_a2")).sqrt()
}

use crate::coords::{Coords, Spacing};
use crate::gv::gv_field_at;
use crate::gv::{cell_geometry_gv, gv_axis_width};
use symbi_ir::{begin_trace, end_trace};

/// the 3-space axes whose torque component can be nonzero at dimension
/// `ndim`: rotation needs a plane, so 1d has none, 2d only the z moment,
/// 3d all three.
pub fn torque_axes(ndim: usize) -> std::ops::Range<usize> {
    match ndim {
        3 => 0..3,
        2 => 2..3,
        _ => 0..0,
    }
}

/// map the coordinate cell centroid `(r, theta, phi)` / `(R, phi, z)` to cartesian
/// so the sphere SDF measures the PHYSICAL distance to the body. a coordinate-space
/// subtraction is meaningless on a curved grid (`sqrt((r - r_b)^2 + (theta -
/// theta_b)^2)` is not a distance); on a Cartesian grid this is the identity. the
/// 2D cylindrical chart is the `(R, phi)` disk plane (`CylindricalRPhi`), matching
/// the geometric-source dispatch — one metric for the whole codebase.
/// the 2d cylindrical (r, z) axisymmetric section: the phi = 0 half-plane is
/// isometric to a 2d cartesian half-plane (h_r = h_z = 1), so an ON-AXIS body's
/// mask distance is the plain euclidean |(r, z - z0)|, the section frame
/// rotations are identities, and the mask region is a genuine coordinate-space
/// ball. the (r, phi) disk (axes [0, 1]) keeps the curved-chart machinery.
fn is_cyl_rz(coords: Coords, ndim: usize, axes: &[usize]) -> bool {
    coords == Coords::Cylindrical && ndim == 2 && axes[..2] == [0, 2]
}

fn centroid_to_cartesian(coords: Coords, ndim: usize, axes: &[usize], centroid: &[Gv]) -> Vec<Gv> {
    if is_cyl_rz(coords, ndim, axes) {
        return centroid[..ndim].to_vec();
    }
    fn run<M, const D: usize>(m: M, x: &[Gv]) -> Vec<Gv>
    where
        M: Metric<Gv, D>,
    {
        let xc = m.to_cartesian(Tensor::from_fn(|i| x[i]));
        (0..D).map(|i| xc[i]).collect()
    }
    match (coords, ndim) {
        (Coords::Cartesian, _) => centroid[..ndim].to_vec(),
        (Coords::Spherical, 1) => run::<_, 1>(Spherical, centroid),
        (Coords::Spherical, 2) => run::<_, 2>(Spherical, centroid),
        (Coords::Spherical, 3) => run::<_, 3>(Spherical, centroid),
        (Coords::Cylindrical, 1) => run::<_, 1>(Cylindrical, centroid),
        (Coords::Cylindrical, 2) => run::<_, 2>(CylindricalRPhi, centroid),
        (Coords::Cylindrical, 3) => run::<_, 3>(Cylindrical, centroid),
        (c, d) => panic!("penalize centroid_to_cartesian: unsupported (coords {c:?}, ndim {d})"),
    }
}

/// rotate a CARTESIAN vector into the cell's PHYSICAL (orthonormal) frame — the
/// frame the substrate stores momentum in. the surface normal is a geometric
/// cartesian direction; the wall / torque-free split projects the physical
/// momentum onto it, so it must be rotated here. `x` is the COORDINATE centroid
/// (the rotation depends on the local basis). identity on a cartesian grid.
fn vector_from_cartesian(
    coords: Coords,
    ndim: usize,
    axes: &[usize],
    x: &[Gv],
    v_cart: &[Gv],
) -> Vec<Gv> {
    if is_cyl_rz(coords, ndim, axes) {
        return v_cart[..ndim].to_vec();
    }
    fn run<M, const D: usize>(m: M, x: &[Gv], v: &[Gv]) -> Vec<Gv>
    where
        M: Metric<Gv, D> + DiagonalMetric<Gv, D>,
    {
        let p = m.vector_from_cartesian(
            Tensor::from_fn(|i| x[i]),
            Embedded::new(Tensor::from_fn(|i| v[i])),
        );
        (0..D).map(|i| p[i]).collect()
    }
    match (coords, ndim) {
        (Coords::Cartesian, _) => v_cart[..ndim].to_vec(),
        (Coords::Spherical, 1) => run::<_, 1>(Spherical, x, v_cart),
        (Coords::Spherical, 2) => run::<_, 2>(Spherical, x, v_cart),
        (Coords::Spherical, 3) => run::<_, 3>(Spherical, x, v_cart),
        (Coords::Cylindrical, 1) => run::<_, 1>(Cylindrical, x, v_cart),
        (Coords::Cylindrical, 2) => run::<_, 2>(CylindricalRPhi, x, v_cart),
        (Coords::Cylindrical, 3) => run::<_, 3>(Cylindrical, x, v_cart),
        (c, d) => panic!("penalize vector_from_cartesian: unsupported (coords {c:?}, ndim {d})"),
    }
}

/// rotate a PHYSICAL-frame vector into the global cartesian frame — used to book
/// the lab-frame torque `r_cart x F_cart` (the accreted force receipt is in the
/// physical frame; the cross product needs both vectors in one frame). identity
/// on a cartesian grid, so the cartesian torque is bit-unchanged.
fn vector_to_cartesian(
    coords: Coords,
    ndim: usize,
    axes: &[usize],
    x: &[Gv],
    v_phys: &[Gv],
) -> Vec<Gv> {
    if is_cyl_rz(coords, ndim, axes) {
        return v_phys[..ndim].to_vec();
    }
    fn run<M, const D: usize>(m: M, x: &[Gv], v: &[Gv]) -> Vec<Gv>
    where
        M: Metric<Gv, D> + DiagonalMetric<Gv, D>,
    {
        let e = m.vector_to_cartesian(
            Tensor::from_fn(|i| x[i]),
            Physical::new(Tensor::from_fn(|i| v[i])),
        );
        (0..D).map(|i| e[i]).collect()
    }
    match (coords, ndim) {
        (Coords::Cartesian, _) => v_phys[..ndim].to_vec(),
        (Coords::Spherical, 1) => run::<_, 1>(Spherical, x, v_phys),
        (Coords::Spherical, 2) => run::<_, 2>(Spherical, x, v_phys),
        (Coords::Spherical, 3) => run::<_, 3>(Spherical, x, v_phys),
        (Coords::Cylindrical, 1) => run::<_, 1>(Cylindrical, x, v_phys),
        (Coords::Cylindrical, 2) => run::<_, 2>(CylindricalRPhi, x, v_phys),
        (Coords::Cylindrical, 3) => run::<_, 3>(Cylindrical, x, v_phys),
        (c, d) => panic!("penalize vector_to_cartesian: unsupported (coords {c:?}, ndim {d})"),
    }
}

/// append the FORM-DRAG receipt: the surface force projected onto the outward SDF normal,
/// `force_normal = (F.n_hat) n_hat`, in the cartesian frame the reduction sums (both `f_cart` and
/// `n_cart` are already cartesian). the tangential (skin-friction) part is recovered downstream as
/// `force - force_normal`. a bare drain passes `n_cart = 0`, so its form drag is exactly zero. this
/// is the LAST receipt block, appended after mass/force/energy/torque so no existing slot shifts.
fn push_force_normal(writes: &mut Writes, ndim: usize, f_cart: &Tensor<Gv, 3>, n_cart: &[Gv]) {
    let mut f_dot_n = Gv::ZERO;
    for a in 0..ndim {
        f_dot_n = f_dot_n + f_cart[a] * n_cart[a];
    }
    for a in 0..ndim {
        writes.push((
            format!("pen_force_normal_{a}"),
            format!("pen_0_force_normal_{a}").into(),
            (f_dot_n * n_cart[a]).node(),
        ));
    }
}

/// the cell's force receipt in the CARTESIAN world frame plus its lab-frame
/// moment `r_cart x F_cart` about the body center. penalization acts in the
/// cell's physical orthonormal basis, which rotates from cell to cell on a
/// curvilinear chart — only cartesian components sum across cells to a
/// meaningful net force on the body (identity rotation on a cartesian grid).
/// returns `(f_cart, torque)`; 2d books only the z moment, 1d none.
fn cartesian_receipt(
    coords: Coords,
    ndim: usize,
    axes: &[usize],
    centroid: &[Gv],
    x_cart: &[Gv; 3],
    center: &[Gv; 3],
    force: &Tensor<Gv, 3>,
) -> (Tensor<Gv, 3>, Tensor<Gv, 3>) {
    if is_cyl_rz(coords, ndim, axes) {
        // the 2d cell stands for a full ring: the ring-radial force cancels
        // around the ring identically, so the net world force is z only, and
        // the axis torque is the moment arm r times the out-of-plane (phi)
        // momentum receipt — nonzero only when the swirl slot exists (dof 3).
        let f = Tensor::<Gv, 3>::new([Gv::ZERO, force[1], Gv::ZERO]);
        let torque = Tensor::<Gv, 3>::new([Gv::ZERO, Gv::ZERO, centroid[0] * force[2]]);
        return (f, torque);
    }
    let f_cart = vector_to_cartesian(
        coords,
        ndim,
        axes,
        centroid,
        &(0..ndim).map(|a| force[a]).collect::<Vec<_>>(),
    );
    let f = Tensor::<Gv, 3>::new(std::array::from_fn(|a| {
        if a < ndim { f_cart[a] } else { Gv::ZERO }
    }));
    let x_rel = Tensor::<Gv, 3>::new(std::array::from_fn(|a| x_cart[a] - center[a]));
    let torque = symbi_ib::moment(&x_rel, &f);
    (f, torque)
}

/// a cartesian world-frame velocity expressed in the cell's physical
/// orthonormal basis. gas momentum is stored in physical components, so a
/// wall's velocity target (the body's translational velocity, or the local
/// rigid surface velocity u + omega x r) must be rotated into the cell frame
/// before the normal/tangential decomposition (identity on a cartesian grid).
fn solid_velocity_phys(
    coords: Coords,
    ndim: usize,
    axes: &[usize],
    centroid: &[Gv],
    u_cart: &Tensor<Gv, 3>,
) -> Tensor<Gv, 3> {
    let u_phys = vector_from_cartesian(
        coords,
        ndim,
        axes,
        centroid,
        &(0..ndim).map(|a| u_cart[a]).collect::<Vec<_>>(),
    );
    Tensor::new(std::array::from_fn(|a| {
        if a < ndim { u_phys[a] } else { Gv::ZERO }
    }))
}

/// the immersed body's mask geometry as a traced SDF, centered at `center` (the body
/// position `body_0_pos_*` in cartesian): a sphere of radius `body_0_racc`. this is the
/// ONE seam every penalization kernel (drain / porous-wall / torque-free, adiabatic and
/// iso) shares — the mask cannot drift between kernels.
fn body_mask_sdf(center: [Gv; 3]) -> SdfExpr<Gv, 3> {
    body_mask_sdf_shaped(center, None)
}

/// the mask geometry with an optional config CSG. `None`: a sphere of runtime radius
/// `body_0_racc` (the AOT kernel). `Some(shape)`: the shape — authored in the body-LOCAL
/// frame as f64 constants — lifted to Gv constants and TRANSLATED to the runtime center. so a
/// MOVING body rides the same kernel: only `body_0_pos_*` changes per step; the shape geometry
/// is baked, never a runtime knob. (rotation would compose a runtime orientation transform on
/// `x` before the shape; translation alone is the moving-body core.)
///
/// `center` is the body's CENTER OF MASS AND its mask/geometric center at once — they coincide for
/// a symmetric body (sphere, symmetric CSG). everything dynamical (translation, gravity, the
/// omega x r wall velocity, the torque moment arm `x - center`) is referenced to the COM, so an
/// asymmetric mass distribution would offset the MASK PLACEMENT alone.
fn body_mask_sdf_shaped(center: [Gv; 3], shape: Option<&SdfExpr<f64, 3>>) -> SdfExpr<Gv, 3> {
    match shape {
        None => SdfExpr::<Gv, 3>::Sphere {
            center,
            radius: Gv::scalar("body_0_racc"),
        },
        Some(s) => s.lift(&|c| Gv::from_f64(c)).translated(center),
    }
}

/// the SPINNING body's mask: the shape lifted to Gv constants, rotated by the RUNTIME orientation
/// matrix `R` (the body's full 3D orientation, integrated on the CPU from its angular velocity and
/// read here as the 9 row-major scalars `body_0_rot_0..8`), then translated to the runtime body
/// position. one kernel handles ANY orientation, so a freely-tumbling body reuses it; the mask + its
/// Dual-autodiff normal track `R` as it evolves.
fn body_mask_sdf_spinning(center: [Gv; 3], shape: &SdfExpr<f64, 3>) -> SdfExpr<Gv, 3> {
    let rot: [[Gv; 3]; 3] = std::array::from_fn(|i| {
        std::array::from_fn(|j| Gv::scalar(&format!("body_0_rot_{}", i * 3 + j)))
    });
    shape
        .lift(&|c| Gv::from_f64(c))
        .rotated(rot)
        .translated(center)
}

/// the saturation lemma at the mask seam: `chi = 0.5(1 - tanh(phi/w))` is
/// exactly zero in f64 once `phi > DRAIN_SUPPORT_WIDTHS * w`, so the mask is
/// supported by its geometry's bounding ball padded by that many cell widths.
/// tagging chi HERE (where the mask is built) lets `with_derived_support`
/// propagate the ball to every write — support and mask cannot drift apart.
/// the ball is a COORDINATE-space region: only the identity chart can spell
/// the cartesian mask ball in grid coordinates, so curvilinear kernels stay
/// untagged and derive Everywhere (dispatch already falls back to the whole
/// interior off-cartesian). a SPINNING shape sweeps every orientation about
/// the body position, so its ball is position-centered with the body-local
/// offset folded into the radius.
fn tag_body_mask(
    chi: &Gv,
    coords: Coords,
    ndim: usize,
    axes: &[usize],
    shape: Option<&SdfExpr<f64, 3>>,
    spin: bool,
) {
    // the (r, z) section's on-axis mask region is a genuine coordinate ball
    // (identity embedding), so the sphere mask tags there too; every other
    // curvilinear chart derives Everywhere. shaped CSG stays cartesian-only.
    let rz_sphere = is_cyl_rz(coords, ndim, axes) && shape.is_none();
    if coords != Coords::Cartesian && !rz_sphere {
        return;
    }
    let pad = ParamExpr::constant(crate::ibm::DRAIN_SUPPORT_WIDTHS)
        * ParamExpr::min_of(
            (0..ndim)
                .map(|a| ParamExpr::param(&format!("dx_{a}")))
                .collect(),
        );
    let pos = |a: usize| ParamExpr::param(&format!("body_0_pos_{a}"));
    let (center, radius) = match shape {
        None => (
            (0..ndim).map(pos).collect(),
            ParamExpr::param("body_0_racc") + pad,
        ),
        Some(s) => {
            let (lc, lr) = s.bounding_ball().expect(
                "a shaped immersed body must be bounded (a complement has no support ball)",
            );
            if spin {
                let lc_norm = (lc[0] * lc[0] + lc[1] * lc[1] + lc[2] * lc[2]).sqrt();
                (
                    (0..ndim).map(pos).collect(),
                    ParamExpr::constant(lr + lc_norm) + pad,
                )
            } else {
                (
                    (0..ndim)
                        .map(|a| pos(a) + ParamExpr::constant(lc[a]))
                        .collect(),
                    ParamExpr::constant(lr) + pad,
                )
            }
        }
    };
    symbi_ir::tag_support_ball(chi, center, radius);
}

/// the 2.5D immersed-body OHMIC RESISTIVE edge EMF: adds a body-LOCALIZED resistive current
/// `eta * chi(x) * J_z` to the out-of-plane edge EMF `ez` (efield[0]) IN PLACE, where `J_z =
/// dB_y/dx - dB_x/dy` is the SAME adjoint current as the generic resistive kernel and `chi` is the
/// body mask (`0.5(1 - tanh(phi/w))`, 1 inside the body, 0 outside, mollified over one cell). the
/// localized resistivity dissipates the magnetic field THREADING the body while leaving the exterior
/// flux untouched. div-B-clean: the same curl consumes the augmented EMF (`div(curl) = 0`). STABILITY:
/// the composed operator `-curl(eta chi J)` is `-C diag(eta chi) C^T`, negative-definite for ANY
/// `eta chi >= 0` — the mask only reweights the edges of the already-adjoint current, so the body can
/// only DISSIPATE, never amplify. `chi` is sampled at the E_z CORNER (the edge location), half a cell
/// below the traced cell centroid on each in-plane axis.
pub fn body_resistive_emf_2d_gv(coords: Coords) -> (GvKernel, Writes) {
    let ndim = 2usize;
    let axes: &[usize] = &[0, 1, 2][..ndim];
    begin_trace();
    let ez = Gv::field("ez", "ez");
    let eta = Gv::scalar("eta");
    let bx = Gv::field("bx", "bx");
    let by = Gv::field("by", "by");
    let bx_jm = gv_field_at("bx", "bx", 2, &[0, -1]); // B_x at the neighbor below in y
    let by_im = gv_field_at("by", "by", 2, &[-1, 0]); // B_y at the neighbor behind in x
    let dx0 = Gv::scalar("dx_0");
    let dx1 = Gv::scalar("dx_1");
    let jz = (Gv::ONE / dx0) * (by - by_im) - (Gv::ONE / dx1) * (bx - bx_jm);

    // the geometry scaffold yields the cell centroid; the E_z corner is half a cell below it on each
    // in-plane axis. sample the body mask at that corner so the dissipation registers with the edge.
    let geo = cell_geometry_gv(coords, &vec![Spacing::Uniform; ndim], &axes[..ndim], ndim);
    let half = Gv::from_f64(0.5);
    let corner = vec![geo.centroid[0] - half * dx0, geo.centroid[1] - half * dx1];
    let x_cart = centroid_to_cartesian(coords, ndim, axes, &corner);
    let x: [Gv; 3] = std::array::from_fn(|a| if a < ndim { x_cart[a] } else { Gv::ZERO });
    let center: [Gv; 3] = std::array::from_fn(|a| {
        if a < ndim {
            Gv::scalar(&format!("body_0_pos_{a}"))
        } else {
            Gv::ZERO
        }
    });
    let min_w = dx0.min(dx1);
    let phi = body_mask_sdf(center).dist(x);
    let chi = symbi_ib::sdf::chi(phi, min_w);
    tag_body_mask(&chi, coords, ndim, axes, None, false);

    let ez_new = ez + eta * chi * jz;
    let writes: Writes = vec![("ez_new".to_string(), "ez".into(), ez_new.node())];
    let kernel = end_trace().with_derived_support(&writes);
    (kernel, writes)
}

/// the 3D cartesian body-masked ohmic resistive edge EMF along edge `dir`: adds `eta * chi * J_dir`
/// to that edge's EMF in place, where `chi` is the body indicator sampled at the dir-edge and
/// `J_dir = dB_p2/dx_p1 - dB_p1/dx_p2` (p1=(dir+1)%3, p2=(dir+2)%3) is the current from the two
/// transverse faces. it is the bulk 3D resistive EMF (`rmhd_resistive_emf_3d_dir_gv`) gated by the
/// mask, so the same div-B-clean 3D curl consumes it and the composed operator is the mask-weighted
/// negative-definite laplacian: the body can only shed the field threading it.
pub fn body_resistive_emf_3d_dir_gv(dir: usize, coords: Coords) -> (GvKernel, Writes) {
    let ndim = 3usize;
    let axes: &[usize] = &[0, 1, 2][..ndim];
    let p1 = (dir + 1) % 3;
    let p2 = (dir + 2) % 3;
    begin_trace();
    let emf = Gv::field("emf", "emf");
    let eta = Gv::scalar("eta");
    let b_p1 = Gv::field("b_p1", "b_p1");
    let b_p2 = Gv::field("b_p2", "b_p2");
    let back = |ax: usize| -> [i32; 3] {
        let mut o = [0, 0, 0];
        o[ax] = -1;
        o
    };
    let b_p1_m = gv_field_at("b_p1", "b_p1", 3, &back(p2)); // dB_p1/dx_p2, backward
    let b_p2_m = gv_field_at("b_p2", "b_p2", 3, &back(p1)); // dB_p2/dx_p1, backward
    let dxp1 = Gv::scalar(&format!("dx_{p1}"));
    let dxp2 = Gv::scalar(&format!("dx_{p2}"));
    let j = (Gv::ONE / dxp1) * (b_p2 - b_p2_m) - (Gv::ONE / dxp2) * (b_p1 - b_p1_m);

    // the dir-edge sits at the cell centroid shifted half a cell back on the two TRANSVERSE axes
    // (p1, p2) and centered along dir. sample the body mask at that edge so the dissipation registers
    // with the edge EMF.
    let dx: Vec<Gv> = (0..ndim).map(|a| Gv::scalar(&format!("dx_{a}"))).collect();
    let geo = cell_geometry_gv(coords, &vec![Spacing::Uniform; ndim], &axes[..ndim], ndim);
    let half = Gv::from_f64(0.5);
    let corner: Vec<Gv> = (0..ndim)
        .map(|a| {
            if a == dir {
                geo.centroid[a]
            } else {
                geo.centroid[a] - half * dx[a]
            }
        })
        .collect();
    let x_cart = centroid_to_cartesian(coords, ndim, axes, &corner);
    let x: [Gv; 3] = std::array::from_fn(|a| if a < ndim { x_cart[a] } else { Gv::ZERO });
    let center: [Gv; 3] = std::array::from_fn(|a| {
        if a < ndim {
            Gv::scalar(&format!("body_0_pos_{a}"))
        } else {
            Gv::ZERO
        }
    });
    let min_w = dx.iter().copied().reduce(|a, b| a.min(b)).unwrap();
    let phi = body_mask_sdf(center).dist(x);
    let chi = symbi_ib::sdf::chi(phi, min_w);
    tag_body_mask(&chi, coords, ndim, axes, None, false);

    let emf_new = emf + eta * chi * j;
    let writes: Writes = vec![("emf_new".to_string(), "emf".into(), emf_new.node())];
    let kernel = end_trace().with_derived_support(&writes);
    (kernel, writes)
}

/// trace the [Drain]-stack penalization for the adiabatic regime, cartesian. `ndim` is the SPATIAL
/// grid dimension (geometry, mask, force receipt); `dof` is the MOMENTUM count (the conserved
/// 3-vector's active components). `dof == ndim` for hydro and full 3D MHD; `dof = 3, ndim = 2` for
/// 2.5D MHD, where the out-of-plane momentum must be drained too (else its velocity blows up as the
/// density is evacuated). dimension-generic over 1..=3.
pub fn penalize_drain_gv(
    coords: Coords,
    ndim: usize,
    dof: usize,
    axes: &[usize],
    has_dye: bool,
) -> (GvKernel, Writes) {
    assert!(
        (1..=3).contains(&ndim) && (ndim..=3).contains(&dof),
        "penalize_drain_gv: need 1<=ndim<=dof<=3"
    );
    begin_trace();
    let dt = Gv::scalar("dt");
    let gamma = Gv::scalar("gamma");
    let c_drain = Gv::scalar("c_drain");

    // conserved reads (in-place: the same fields are the writes). the momentum runs over dof (all
    // active components), so a 2.5D MHD sink drains the out-of-plane momentum and its kinetic energy.
    let den = Gv::field("den", symbi_ir::FieldRef::cons_den());
    let mom: Vec<Gv> = (0..dof)
        .map(|c| Gv::field(&format!("mom_{c}"), symbi_ir::FieldRef::cons_mom(c as u8)))
        .collect();
    let nrg = Gv::field("nrg", symbi_ir::FieldRef::cons_nrg());

    // the cell centroid + volume from the shared geometry scaffold — the coordinate
    // map every other body kernel of this chart evaluates.
    let geo = cell_geometry_gv(coords, &vec![Spacing::Uniform; ndim], &axes[..ndim], ndim);
    let dv = Gv::ONE / geo.inv_volume;
    let mut min_w = gv_axis_width(0, Spacing::Uniform);
    for ax in 1..ndim {
        min_w = min_w.min(gv_axis_width(ax, Spacing::Uniform));
    }

    // the mask geometry as a traced SDF: phi = |x - body_pos| - r_mask, and
    // chi = 0.5 (1 - tanh(phi / w)). the distance is PHYSICAL, so the coordinate
    // centroid is mapped to cartesian first (identity on a cartesian grid).
    let center: [Gv; 3] = std::array::from_fn(|a| {
        if a < ndim {
            Gv::scalar(&format!("body_0_pos_{a}"))
        } else {
            Gv::ZERO
        }
    });
    let x_cart = centroid_to_cartesian(coords, ndim, axes, &geo.centroid);
    let x: [Gv; 3] = std::array::from_fn(|a| if a < ndim { x_cart[a] } else { Gv::ZERO });
    let sphere = body_mask_sdf(center);
    let phi = sphere.dist(x);
    let chi = symbi_ib::sdf::chi(phi, min_w);
    tag_body_mask(&chi, coords, ndim, axes, None, false);

    // tau = c_drain dx / c_s with c_s from the just-updated conserved state
    // (the drain runs post-godunov, pre-c2p — the stored primitive is stale).
    let mut mom_sq = Gv::ZERO;
    for m in &mom {
        mom_sq = mom_sq + *m * *m;
    }
    let cs = symbi_ib::drain::sound_speed_from_cons(den, mom_sq, nrg, gamma);
    let inv_tau = signal_speed(cs) / (c_drain * min_w);

    // the property stack: [Drain]. contribute at Gv, then
    // the SAME integrator that runs at f64.
    let kin = BodyKin::<Gv, 3> {
        u_solid: Tensor::zeros(),
        omega: Tensor::zeros(),
        e_wall: Gv::ZERO,
    };
    let mut acc = Relax::<Gv, 3>::none();
    Property::Drain { inv_tau }.contribute(chi, &kin, &mut acc);
    let cons = ConsG::<Gv, 3, Adiabatic, Dyed> {
        // an undyed kernel traces a constant-zero dye: `penalize_cell` still scales it by the
        // drain factor, the result feeds no write, and the tracer eliminates the dead arithmetic.
        chi: if has_dye {
            Gv::field("chi", symbi_ir::FieldRef::cons_chi())
        } else {
            Gv::ZERO
        },
        den,
        mom: Tensor::new(std::array::from_fn(
            |a| if a < dof { mom[a] } else { Gv::ZERO },
        )),
        nrg,
    };
    let (out, delta) = penalize_cell(&cons, &acc, Tensor::zeros(), dt, dv, 0);
    // a bare drain has no wall surface -> a zero normal -> zero form drag (all force is accretion).
    let n_cart: Vec<Gv> = vec![Gv::ZERO; ndim];
    // the angular-momentum receipt: the lab-frame moment r_cart x F_cart of the
    // cell's force receipt about the body center. exactly zero beyond the support
    // ball (F vanishes there). 3d books all three components; 2d only z; 1d none.
    let (f_cart, torque) = cartesian_receipt(
        coords,
        ndim,
        axes,
        &geo.centroid,
        &x,
        &center,
        &delta.force_delta,
    );

    let mut writes: Writes = Vec::new();
    writes.push((
        "den_out".to_string(),
        symbi_ir::FieldRef::cons_den().into(),
        out.den.node(),
    ));
    for a in 0..dof {
        writes.push((
            format!("mom_out_{a}"),
            symbi_ir::FieldRef::cons_mom(a as u8).into(),
            out.mom[a].node(),
        ));
    }
    writes.push((
        "nrg_out".to_string(),
        symbi_ir::FieldRef::cons_nrg().into(),
        out.nrg.node(),
    ));
    // the sink swallows gas and the dye dissolved in it together; `penalize_cell` applied the
    // same drain factor to both, so the concentration of the surviving gas is unchanged.
    if has_dye {
        writes.push((
            "chi_out".to_string(),
            symbi_ir::FieldRef::cons_chi().into(),
            out.chi.node(),
        ));
    }
    writes.push((
        "pen_mass".to_string(),
        "pen_0_mass".into(),
        delta.mass_delta.node(),
    ));
    for a in 0..ndim {
        writes.push((
            format!("pen_force_{a}"),
            format!("pen_0_force_{a}").into(),
            f_cart[a].node(),
        ));
    }
    writes.push((
        "pen_energy".to_string(),
        "pen_0_energy".into(),
        delta.energy_delta.node(),
    ));
    for a in torque_axes(ndim) {
        writes.push((
            format!("pen_torque_{a}"),
            format!("pen_0_torque_{a}").into(),
            torque[a].node(),
        ));
    }
    push_force_normal(&mut writes, ndim, &f_cart, &n_cart);

    // the delta outputs vanish exactly beyond the tanh saturation radius; the
    // in-place cons writes are unchanged-value there, so the ball bounds
    // everything the reduction needs (dispatch may clip to it).
    let kernel = end_trace().with_derived_support(&writes);
    (kernel, writes)
}

/// trace the [PorousAccretor]-stack penalization for the adiabatic regime,
/// cartesian, DOF = ndim. the porosity dial `p` scales the drain channel and
/// `(1 - p)` the wall channels; the wall rates are `k_eta_n/t * c_s / dx`
/// (sound-crossings per cell width — MULTIPLICATIVE dials so zero is an EXACT
/// off switch: `k_eta_t = 0` is free-slip with the tangential velocity
/// bit-untouched). the velocity target is the body's translational velocity
/// (`body_0_vel_*`); the surface normal is the sphere's, `x_rel / |x_rel|`,
/// with the division guarded by a subnormal floor so the body-center cell
/// (|x_rel| = 0) degrades to a zero normal — its whole du is treated as
/// tangential, exact and finite. at p = 1 every wall factor carries an exact
/// (1 - p) = 0 and the kernel reduces bit-for-bit to `penalize_drain`.
pub fn penalize_porous_gv(
    coords: Coords,
    ndim: usize,
    dof: usize,
    axes: &[usize],
    has_dye: bool,
) -> (GvKernel, Writes) {
    penalize_porous_inner(coords, ndim, dof, None, false, axes, has_dye)
}

/// the arbitrary-shape porous wall: the same relaxation stack as `penalize_porous_gv`, but the
/// mask AND the surface normal come from the config CSG `shape` (body-local, translated to the
/// runtime body position). built at sim SETUP, once the body's shape is
/// known — the AOT bake cannot know a per-body CSG.
pub fn penalize_porous_gv_shaped(
    coords: Coords,
    ndim: usize,
    dof: usize,
    shape: &SdfExpr<f64, 3>,
    has_dye: bool,
) -> (GvKernel, Writes) {
    penalize_porous_inner(coords, ndim, dof, Some(shape), false, &[0, 1, 2][..ndim], has_dye)
}

/// the SPINNING arbitrary-shape porous wall: like `penalize_porous_gv_shaped`, but the mask is
/// rotated by the runtime orientation matrix (the 9 scalars `body_0_rot_0..8`) and the surface
/// velocity carries the spin `omega x r` (the vector `body_0_omega_0..2`), so the wall drags the
/// gas around as it turns — an arbitrary spin axis.
pub fn penalize_porous_gv_spinning(
    coords: Coords,
    ndim: usize,
    dof: usize,
    shape: &SdfExpr<f64, 3>,
    has_dye: bool,
) -> (GvKernel, Writes) {
    penalize_porous_inner(coords, ndim, dof, Some(shape), true, &[0, 1, 2][..ndim], has_dye)
}

fn penalize_porous_inner(
    coords: Coords,
    ndim: usize,
    dof: usize,
    shape: Option<&SdfExpr<f64, 3>>,
    spin: bool,
    axes: &[usize],
    has_dye: bool,
) -> (GvKernel, Writes) {
    assert!(
        (1..=3).contains(&ndim) && (ndim..=3).contains(&dof),
        "penalize_porous_gv: need 1<=ndim<=dof<=3"
    );
    begin_trace();
    let dt = Gv::scalar("dt");
    let gamma = Gv::scalar("gamma");
    let c_drain = Gv::scalar("c_drain");
    let porosity = Gv::scalar("porosity");
    let k_eta_n = Gv::scalar("k_eta_n");
    let k_eta_t = Gv::scalar("k_eta_t");

    // the wall normal + velocity target are ndim (in-plane); the momentum runs over dof so the
    // out-of-plane component is carried as a purely TANGENTIAL velocity (relaxed by k_eta_t) and its
    // kinetic energy enters c_s. drained/walled in 2.5D MHD, absent for hydro (dof == ndim).
    let den = Gv::field("den", symbi_ir::FieldRef::cons_den());
    let mom: Vec<Gv> = (0..dof)
        .map(|c| Gv::field(&format!("mom_{c}"), symbi_ir::FieldRef::cons_mom(c as u8)))
        .collect();
    let nrg = Gv::field("nrg", symbi_ir::FieldRef::cons_nrg());

    let geo = cell_geometry_gv(coords, &vec![Spacing::Uniform; ndim], &axes[..ndim], ndim);
    let dv = Gv::ONE / geo.inv_volume;
    let mut min_w = gv_axis_width(0, Spacing::Uniform);
    for ax in 1..ndim {
        min_w = min_w.min(gv_axis_width(ax, Spacing::Uniform));
    }

    let center: [Gv; 3] = std::array::from_fn(|a| {
        if a < ndim {
            Gv::scalar(&format!("body_0_pos_{a}"))
        } else {
            Gv::ZERO
        }
    });
    // the mask distance is PHYSICAL: map the coordinate centroid to cartesian.
    let x_cart = centroid_to_cartesian(coords, ndim, axes, &geo.centroid);
    let x: [Gv; 3] = std::array::from_fn(|a| if a < ndim { x_cart[a] } else { Gv::ZERO });
    let sdf = if spin {
        body_mask_sdf_spinning(center, shape.expect("a spinning wall must have a shape"))
    } else {
        body_mask_sdf_shaped(center, shape)
    };
    let phi = sdf.dist(x);
    let chi = symbi_ib::sdf::chi(phi, min_w);
    tag_body_mask(&chi, coords, ndim, axes, shape, spin);

    let mut mom_sq = Gv::ZERO;
    for m in &mom {
        mom_sq = mom_sq + *m * *m;
    }
    let cs = symbi_ib::drain::sound_speed_from_cons(den, mom_sq, nrg, gamma);
    let inv_tau = signal_speed(cs) / (c_drain * min_w);
    let rate_scale = signal_speed(cs) / min_w;

    // the outward surface normal in the cell's PHYSICAL frame (the cartesian normal rotated into
    // the orthonormal basis; identity on a cartesian grid). the sphere path is r_hat =
    // x_rel/|x_rel|, with |x_rel| = 0 guarded to a zero normal (its whole du is tangential,
    // finite). the shaped path is the exact CSG gradient — the SDF outward unit normal via Dual
    // autodiff — which is the arbitrary surface's normal everywhere the distance is smooth.
    let n_cart: Vec<Gv> = match shape {
        None => {
            let x_rel = Tensor::<Gv, 3>::new(std::array::from_fn(|a| x[a] - center[a]));
            let r = x_rel.dot(&x_rel).sqrt();
            let nonzero = r.cmp_gt(Gv::ZERO);
            let divisor = Gv::select(nonzero, r, Gv::ONE);
            let inv_r = Gv::select(nonzero, Gv::ONE / divisor, Gv::ZERO);
            (0..ndim).map(|a| x_rel[a] * inv_r).collect()
        }
        Some(_) => {
            let n = sdf.normal(x);
            (0..ndim).map(|a| n[a]).collect()
        }
    };
    let n_phys = vector_from_cartesian(coords, ndim, axes, &geo.centroid, &n_cart);
    let normal = Tensor::<Gv, 3>::new(std::array::from_fn(|a| {
        if a < ndim { n_phys[a] } else { Gv::ZERO }
    }));

    let u_solid = Tensor::<Gv, 3>::new(std::array::from_fn(|a| {
        if a < ndim {
            Gv::scalar(&format!("body_0_vel_{a}"))
        } else {
            Gv::ZERO
        }
    }));
    // a spinning wall's surface moves at u_solid + omega x r with the FULL world-frame
    // angular-velocity vector (an arbitrary rotation axis; z-spin is the [0, 0, w] case).
    let omega = if spin {
        // the full angular-velocity vector (world frame): the surface drags the gas at omega x r
        // about the (evolving) rotation axis.
        Tensor::<Gv, 3>::new([
            Gv::scalar("body_0_omega_0"),
            Gv::scalar("body_0_omega_1"),
            Gv::scalar("body_0_omega_2"),
        ])
    } else {
        Tensor::zeros()
    };
    let base_kin = BodyKin::<Gv, 3> {
        u_solid,
        omega,
        e_wall: Gv::ZERO,
    };
    // a spinning wall's velocity target is the LOCAL rigid-motion velocity u_solid + omega x r_cell;
    // `at` bakes omega x r into u_solid per cell (the contribute path reads u_solid directly). the
    // static path keeps the bare translational u_solid, bit-identical to before.
    let kin_cart = if spin {
        let x_rel = Tensor::<Gv, 3>::new(std::array::from_fn(|a| x[a] - center[a]));
        base_kin.at(&x_rel)
    } else {
        base_kin
    };
    // the surface velocity is assembled in the cartesian world frame; the gas
    // momentum is stored in physical components, so the target rotates into the
    // cell's orthonormal basis before the normal/tangential decomposition.
    let kin = BodyKin::<Gv, 3> {
        u_solid: solid_velocity_phys(coords, ndim, axes, &geo.centroid, &kin_cart.u_solid),
        ..kin_cart
    };
    let mut acc = Relax::<Gv, 3>::none();
    Property::PorousAccretor {
        p: porosity,
        inv_tau,
        inv_eta_n: k_eta_n * rate_scale,
        inv_eta_t: k_eta_t * rate_scale,
    }
    .contribute(chi, &kin, &mut acc);
    let cons = ConsG::<Gv, 3, Adiabatic, Dyed> {
        // an undyed kernel traces a constant-zero dye: `penalize_cell` still scales it by the
        // drain factor, the result feeds no write, and the tracer eliminates the dead arithmetic.
        chi: if has_dye {
            Gv::field("chi", symbi_ir::FieldRef::cons_chi())
        } else {
            Gv::ZERO
        },
        den,
        mom: Tensor::new(std::array::from_fn(
            |a| if a < dof { mom[a] } else { Gv::ZERO },
        )),
        nrg,
    };
    let (out, delta) = penalize_cell(&cons, &acc, normal, dt, dv, 0);
    let (f_cart, torque) = cartesian_receipt(
        coords,
        ndim,
        axes,
        &geo.centroid,
        &x,
        &center,
        &delta.force_delta,
    );

    let mut writes: Writes = Vec::new();
    writes.push((
        "den_out".to_string(),
        symbi_ir::FieldRef::cons_den().into(),
        out.den.node(),
    ));
    for a in 0..dof {
        writes.push((
            format!("mom_out_{a}"),
            symbi_ir::FieldRef::cons_mom(a as u8).into(),
            out.mom[a].node(),
        ));
    }
    writes.push((
        "nrg_out".to_string(),
        symbi_ir::FieldRef::cons_nrg().into(),
        out.nrg.node(),
    ));
    // the sink swallows gas and the dye dissolved in it together; `penalize_cell` applied the
    // same drain factor to both, so the concentration of the surviving gas is unchanged.
    if has_dye {
        writes.push((
            "chi_out".to_string(),
            symbi_ir::FieldRef::cons_chi().into(),
            out.chi.node(),
        ));
    }
    writes.push((
        "pen_mass".to_string(),
        "pen_0_mass".into(),
        delta.mass_delta.node(),
    ));
    for a in 0..ndim {
        writes.push((
            format!("pen_force_{a}"),
            format!("pen_0_force_{a}").into(),
            f_cart[a].node(),
        ));
    }
    writes.push((
        "pen_energy".to_string(),
        "pen_0_energy".into(),
        delta.energy_delta.node(),
    ));
    for a in torque_axes(ndim) {
        writes.push((
            format!("pen_torque_{a}"),
            format!("pen_0_torque_{a}").into(),
            torque[a].node(),
        ));
    }
    push_force_normal(&mut writes, ndim, &f_cart, &n_cart);

    let kernel = end_trace().with_derived_support(&writes);
    (kernel, writes)
}

/// the ISOTHERMAL torque-free accretor: the drain plus a
/// tangential ANTI-relaxation `lambda_t = -xi lambda_rho` about the sphere
/// normal, so the accreted mass carries no net angular momentum to the body
/// (the dittmann & ryan 2021 torque-free sink, coordinate-free via the SDF
/// normal). the retention floor (`Relax.ut_growth_cap`, set by the property)
/// bounds the growing tangential factor at the evacuation limit. `xi = 0`
/// reduces bit-for-bit to `penalize_drain_iso`; `xi = 1` is torque-free. no
/// energy channel (iso, the thin-disk regime); delta outputs mass + force +
/// torque. the velocity target is the body's translational velocity
/// `body_0_vel_*` (the accretion is torque-free RELATIVE to the moving sink).
pub fn penalize_torque_free_iso_gv(
    coords: Coords,
    ndim: usize,
    dof: usize,
    axes: &[usize],
) -> (GvKernel, Writes) {
    use symbi_hydro::energy::IsoModel;
    assert!(
        (1..=3).contains(&ndim),
        "penalize_torque_free_iso_gv: ndim must be 1..=3"
    );
    begin_trace();
    let dt = Gv::scalar("dt");
    let cs = Gv::scalar("cs");
    let c_drain = Gv::scalar("c_drain");
    let xi = Gv::scalar("xi");

    let den = Gv::field("den", symbi_ir::FieldRef::cons_den());
    let mom: Vec<Gv> = (0..dof)
        .map(|c| Gv::field(&format!("mom_{c}"), symbi_ir::FieldRef::cons_mom(c as u8)))
        .collect();

    let geo = cell_geometry_gv(coords, &vec![Spacing::Uniform; ndim], &axes[..ndim], ndim);
    let dv = Gv::ONE / geo.inv_volume;
    let mut min_w = gv_axis_width(0, Spacing::Uniform);
    for ax in 1..ndim {
        min_w = min_w.min(gv_axis_width(ax, Spacing::Uniform));
    }

    let center: [Gv; 3] = std::array::from_fn(|a| {
        if a < ndim {
            Gv::scalar(&format!("body_0_pos_{a}"))
        } else {
            Gv::ZERO
        }
    });
    // the mask distance is PHYSICAL: map the coordinate centroid to cartesian.
    let x_cart = centroid_to_cartesian(coords, ndim, axes, &geo.centroid);
    let x: [Gv; 3] = std::array::from_fn(|a| if a < ndim { x_cart[a] } else { Gv::ZERO });
    let sphere = body_mask_sdf(center);
    let chi = symbi_ib::sdf::chi(sphere.dist(x), min_w);
    tag_body_mask(&chi, coords, ndim, axes, None, false);
    let inv_tau = signal_speed(cs) / (c_drain * min_w);

    // the outward surface normal in the cell's PHYSICAL frame: the cartesian
    // r_hat from the body center rotated into the orthonormal basis (identity on
    // a cartesian grid; e_r for a centered accretor). the torque-free channel's
    // radial/tangential split is about this normal, so it must match the frame
    // the momentum lives in.
    let x_rel = Tensor::<Gv, 3>::new(std::array::from_fn(|a| x[a] - center[a]));
    let r = x_rel.dot(&x_rel).sqrt();
    let nonzero = r.cmp_gt(Gv::ZERO);
    let divisor = Gv::select(nonzero, r, Gv::ONE);
    let inv_r = Gv::select(nonzero, Gv::ONE / divisor, Gv::ZERO);
    let n_cart: Vec<Gv> = (0..ndim).map(|a| x_rel[a] * inv_r).collect();
    let n_phys = vector_from_cartesian(coords, ndim, axes, &geo.centroid, &n_cart);
    let normal = Tensor::<Gv, 3>::new(std::array::from_fn(|a| {
        if a < ndim { n_phys[a] } else { Gv::ZERO }
    }));

    let u_solid = Tensor::<Gv, 3>::new(std::array::from_fn(|a| {
        if a < ndim {
            Gv::scalar(&format!("body_0_vel_{a}"))
        } else {
            Gv::ZERO
        }
    }));
    // the body's translational velocity is a cartesian world vector; rotate it
    // into the cell's physical basis to match the stored momentum components.
    let kin = BodyKin::<Gv, 3> {
        u_solid: solid_velocity_phys(coords, ndim, axes, &geo.centroid, &u_solid),
        omega: Tensor::zeros(),
        e_wall: Gv::ZERO,
    };
    let mut acc = Relax::<Gv, 3>::none();
    Property::TorqueFreeAccretor { inv_tau, xi }.contribute(chi, &kin, &mut acc);
    let cons = ConsG::<Gv, 3, IsoModel> {
        chi: Default::default(),
        den,
        mom: Tensor::new(std::array::from_fn(
            |a| if a < dof { mom[a] } else { Gv::ZERO },
        )),
        nrg: Default::default(),
    };
    let (out, delta) = penalize_cell(&cons, &acc, normal, dt, dv, 0);
    let (f_cart, torque) = cartesian_receipt(
        coords,
        ndim,
        axes,
        &geo.centroid,
        &x,
        &center,
        &delta.force_delta,
    );

    let mut writes: Writes = Vec::new();
    writes.push((
        "den_out".to_string(),
        symbi_ir::FieldRef::cons_den().into(),
        out.den.node(),
    ));
    for a in 0..dof {
        writes.push((
            format!("mom_out_{a}"),
            symbi_ir::FieldRef::cons_mom(a as u8).into(),
            out.mom[a].node(),
        ));
    }
    writes.push((
        "pen_mass".to_string(),
        "pen_0_mass".into(),
        delta.mass_delta.node(),
    ));
    for a in 0..ndim {
        writes.push((
            format!("pen_force_{a}"),
            format!("pen_0_force_{a}").into(),
            f_cart[a].node(),
        ));
    }
    for a in torque_axes(ndim) {
        writes.push((
            format!("pen_torque_{a}"),
            format!("pen_0_torque_{a}").into(),
            torque[a].node(),
        ));
    }
    push_force_normal(&mut writes, ndim, &f_cart, &n_cart);

    let kernel = end_trace().with_derived_support(&writes);
    (kernel, writes)
}

/// the ISOTHERMAL [PorousAccretor] twin: the same porous surface as
/// `penalize_porous_gv` but for the isothermal regime — the sound speed is the
/// constant `cs` param (no energy channel), delta outputs mass + force only.
/// `p = 1` reduces bit-for-bit to `penalize_drain_iso`.
pub fn penalize_porous_iso_gv(
    coords: Coords,
    ndim: usize,
    dof: usize,
    axes: &[usize],
) -> (GvKernel, Writes) {
    penalize_porous_iso_inner(coords, ndim, dof, None, false, axes)
}

/// the arbitrary-shape ISO porous wall: the energy-free counterpart of
/// `penalize_porous_gv_shaped` — mask + normal from the config CSG, no energy channel.
pub fn penalize_porous_iso_gv_shaped(
    coords: Coords,
    ndim: usize,
    dof: usize,
    shape: &SdfExpr<f64, 3>,
) -> (GvKernel, Writes) {
    penalize_porous_iso_inner(coords, ndim, dof, Some(shape), false, &[0, 1, 2][..ndim])
}

/// the SPINNING ISO porous wall: the energy-free counterpart of `penalize_porous_gv_spinning`.
pub fn penalize_porous_iso_gv_spinning(
    coords: Coords,
    ndim: usize,
    dof: usize,
    shape: &SdfExpr<f64, 3>,
) -> (GvKernel, Writes) {
    penalize_porous_iso_inner(coords, ndim, dof, Some(shape), true, &[0, 1, 2][..ndim])
}

fn penalize_porous_iso_inner(
    coords: Coords,
    ndim: usize,
    dof: usize,
    shape: Option<&SdfExpr<f64, 3>>,
    spin: bool,
    axes: &[usize],
) -> (GvKernel, Writes) {
    use symbi_hydro::energy::IsoModel;
    assert!(
        (1..=3).contains(&ndim) && (ndim..=3).contains(&dof),
        "penalize_porous_iso_gv: need 1<=ndim<=dof<=3"
    );
    begin_trace();
    let dt = Gv::scalar("dt");
    let cs = Gv::scalar("cs");
    let c_drain = Gv::scalar("c_drain");
    let porosity = Gv::scalar("porosity");
    let k_eta_n = Gv::scalar("k_eta_n");
    let k_eta_t = Gv::scalar("k_eta_t");

    let den = Gv::field("den", symbi_ir::FieldRef::cons_den());
    let mom: Vec<Gv> = (0..dof)
        .map(|c| Gv::field(&format!("mom_{c}"), symbi_ir::FieldRef::cons_mom(c as u8)))
        .collect();

    let geo = cell_geometry_gv(coords, &vec![Spacing::Uniform; ndim], &axes[..ndim], ndim);
    let dv = Gv::ONE / geo.inv_volume;
    let mut min_w = gv_axis_width(0, Spacing::Uniform);
    for ax in 1..ndim {
        min_w = min_w.min(gv_axis_width(ax, Spacing::Uniform));
    }

    let center: [Gv; 3] = std::array::from_fn(|a| {
        if a < ndim {
            Gv::scalar(&format!("body_0_pos_{a}"))
        } else {
            Gv::ZERO
        }
    });
    let x_cart = centroid_to_cartesian(coords, ndim, axes, &geo.centroid);
    let x: [Gv; 3] = std::array::from_fn(|a| if a < ndim { x_cart[a] } else { Gv::ZERO });
    let sdf = if spin {
        body_mask_sdf_spinning(center, shape.expect("a spinning wall must have a shape"))
    } else {
        body_mask_sdf_shaped(center, shape)
    };
    let chi = symbi_ib::sdf::chi(sdf.dist(x), min_w);
    tag_body_mask(&chi, coords, ndim, axes, shape, spin);
    let inv_tau = signal_speed(cs) / (c_drain * min_w);
    let rate_scale = signal_speed(cs) / min_w;

    // sphere normal r_hat (guarded), or the CSG SDF gradient for a shaped wall (see the adiabatic
    // `penalize_porous_gv`).
    let n_cart: Vec<Gv> = match shape {
        None => {
            let x_rel = Tensor::<Gv, 3>::new(std::array::from_fn(|a| x[a] - center[a]));
            let r = x_rel.dot(&x_rel).sqrt();
            let nonzero = r.cmp_gt(Gv::ZERO);
            let divisor = Gv::select(nonzero, r, Gv::ONE);
            let inv_r = Gv::select(nonzero, Gv::ONE / divisor, Gv::ZERO);
            (0..ndim).map(|a| x_rel[a] * inv_r).collect()
        }
        Some(_) => {
            let n = sdf.normal(x);
            (0..ndim).map(|a| n[a]).collect()
        }
    };
    let n_phys = vector_from_cartesian(coords, ndim, axes, &geo.centroid, &n_cart);
    let normal = Tensor::<Gv, 3>::new(std::array::from_fn(|a| {
        if a < ndim { n_phys[a] } else { Gv::ZERO }
    }));

    let u_solid = Tensor::<Gv, 3>::new(std::array::from_fn(|a| {
        if a < ndim {
            Gv::scalar(&format!("body_0_vel_{a}"))
        } else {
            Gv::ZERO
        }
    }));
    let omega = if spin {
        // the full angular-velocity vector (world frame): the surface drags the gas at omega x r
        // about the (evolving) rotation axis.
        Tensor::<Gv, 3>::new([
            Gv::scalar("body_0_omega_0"),
            Gv::scalar("body_0_omega_1"),
            Gv::scalar("body_0_omega_2"),
        ])
    } else {
        Tensor::zeros()
    };
    let base_kin = BodyKin::<Gv, 3> {
        u_solid,
        omega,
        e_wall: Gv::ZERO,
    };
    // a spinning wall's velocity target is the LOCAL rigid-motion velocity u_solid + omega x r_cell;
    // `at` bakes omega x r into u_solid per cell (the contribute path reads u_solid directly). the
    // static path keeps the bare translational u_solid, bit-identical to before.
    let kin_cart = if spin {
        let x_rel = Tensor::<Gv, 3>::new(std::array::from_fn(|a| x[a] - center[a]));
        base_kin.at(&x_rel)
    } else {
        base_kin
    };
    // the surface velocity is assembled in the cartesian world frame; the gas
    // momentum is stored in physical components, so the target rotates into the
    // cell's orthonormal basis before the normal/tangential decomposition.
    let kin = BodyKin::<Gv, 3> {
        u_solid: solid_velocity_phys(coords, ndim, axes, &geo.centroid, &kin_cart.u_solid),
        ..kin_cart
    };
    let mut acc = Relax::<Gv, 3>::none();
    Property::PorousAccretor {
        p: porosity,
        inv_tau,
        inv_eta_n: k_eta_n * rate_scale,
        inv_eta_t: k_eta_t * rate_scale,
    }
    .contribute(chi, &kin, &mut acc);
    let cons = ConsG::<Gv, 3, IsoModel> {
        chi: Default::default(),
        den,
        mom: Tensor::new(std::array::from_fn(
            |a| if a < dof { mom[a] } else { Gv::ZERO },
        )),
        nrg: Default::default(),
    };
    let (out, delta) = penalize_cell(&cons, &acc, normal, dt, dv, 0);
    let (f_cart, torque) = cartesian_receipt(
        coords,
        ndim,
        axes,
        &geo.centroid,
        &x,
        &center,
        &delta.force_delta,
    );

    let mut writes: Writes = Vec::new();
    writes.push((
        "den_out".to_string(),
        symbi_ir::FieldRef::cons_den().into(),
        out.den.node(),
    ));
    for a in 0..ndim {
        writes.push((
            format!("mom_out_{a}"),
            symbi_ir::FieldRef::cons_mom(a as u8).into(),
            out.mom[a].node(),
        ));
    }
    writes.push((
        "pen_mass".to_string(),
        "pen_0_mass".into(),
        delta.mass_delta.node(),
    ));
    for a in 0..ndim {
        writes.push((
            format!("pen_force_{a}"),
            format!("pen_0_force_{a}").into(),
            f_cart[a].node(),
        ));
    }
    for a in torque_axes(ndim) {
        writes.push((
            format!("pen_torque_{a}"),
            format!("pen_0_torque_{a}").into(),
            torque[a].node(),
        ));
    }
    push_force_normal(&mut writes, ndim, &f_cart, &n_cart);

    let kernel = end_trace().with_derived_support(&writes);
    (kernel, writes)
}

/// the ADIABATIC torque-free twin: the same torque-free surface as
/// `penalize_torque_free_iso_gv` but for the adiabatic regime — the sound speed
/// is recovered from the conserved state and the energy channel is carried (delta
/// outputs mass + force + energy + torque). `xi = 0` reduces to `penalize_drain`.
pub fn penalize_torque_free_gv(
    coords: Coords,
    ndim: usize,
    dof: usize,
    axes: &[usize],
    has_dye: bool,
) -> (GvKernel, Writes) {
    assert!(
        (1..=3).contains(&ndim),
        "penalize_torque_free_gv: ndim must be 1..=3"
    );
    begin_trace();
    let dt = Gv::scalar("dt");
    let gamma = Gv::scalar("gamma");
    let c_drain = Gv::scalar("c_drain");
    let xi = Gv::scalar("xi");

    let den = Gv::field("den", symbi_ir::FieldRef::cons_den());
    let mom: Vec<Gv> = (0..dof)
        .map(|c| Gv::field(&format!("mom_{c}"), symbi_ir::FieldRef::cons_mom(c as u8)))
        .collect();
    let nrg = Gv::field("nrg", symbi_ir::FieldRef::cons_nrg());

    let geo = cell_geometry_gv(coords, &vec![Spacing::Uniform; ndim], &axes[..ndim], ndim);
    let dv = Gv::ONE / geo.inv_volume;
    let mut min_w = gv_axis_width(0, Spacing::Uniform);
    for ax in 1..ndim {
        min_w = min_w.min(gv_axis_width(ax, Spacing::Uniform));
    }

    let center: [Gv; 3] = std::array::from_fn(|a| {
        if a < ndim {
            Gv::scalar(&format!("body_0_pos_{a}"))
        } else {
            Gv::ZERO
        }
    });
    let x_cart = centroid_to_cartesian(coords, ndim, axes, &geo.centroid);
    let x: [Gv; 3] = std::array::from_fn(|a| if a < ndim { x_cart[a] } else { Gv::ZERO });
    let sphere = body_mask_sdf(center);
    let chi = symbi_ib::sdf::chi(sphere.dist(x), min_w);
    tag_body_mask(&chi, coords, ndim, axes, None, false);

    let mut mom_sq = Gv::ZERO;
    for m in &mom {
        mom_sq = mom_sq + *m * *m;
    }
    let cs = symbi_ib::drain::sound_speed_from_cons(den, mom_sq, nrg, gamma);
    let inv_tau = signal_speed(cs) / (c_drain * min_w);

    let x_rel = Tensor::<Gv, 3>::new(std::array::from_fn(|a| x[a] - center[a]));
    let r = x_rel.dot(&x_rel).sqrt();
    let nonzero = r.cmp_gt(Gv::ZERO);
    let divisor = Gv::select(nonzero, r, Gv::ONE);
    let inv_r = Gv::select(nonzero, Gv::ONE / divisor, Gv::ZERO);
    let n_cart: Vec<Gv> = (0..ndim).map(|a| x_rel[a] * inv_r).collect();
    let n_phys = vector_from_cartesian(coords, ndim, axes, &geo.centroid, &n_cart);
    let normal = Tensor::<Gv, 3>::new(std::array::from_fn(|a| {
        if a < ndim { n_phys[a] } else { Gv::ZERO }
    }));

    let u_solid = Tensor::<Gv, 3>::new(std::array::from_fn(|a| {
        if a < ndim {
            Gv::scalar(&format!("body_0_vel_{a}"))
        } else {
            Gv::ZERO
        }
    }));
    // the body's translational velocity is a cartesian world vector; rotate it
    // into the cell's physical basis to match the stored momentum components.
    let kin = BodyKin::<Gv, 3> {
        u_solid: solid_velocity_phys(coords, ndim, axes, &geo.centroid, &u_solid),
        omega: Tensor::zeros(),
        e_wall: Gv::ZERO,
    };
    let mut acc = Relax::<Gv, 3>::none();
    Property::TorqueFreeAccretor { inv_tau, xi }.contribute(chi, &kin, &mut acc);
    let cons = ConsG::<Gv, 3, Adiabatic, Dyed> {
        // an undyed kernel traces a constant-zero dye: `penalize_cell` still scales it by the
        // drain factor, the result feeds no write, and the tracer eliminates the dead arithmetic.
        chi: if has_dye {
            Gv::field("chi", symbi_ir::FieldRef::cons_chi())
        } else {
            Gv::ZERO
        },
        den,
        mom: Tensor::new(std::array::from_fn(
            |a| if a < dof { mom[a] } else { Gv::ZERO },
        )),
        nrg,
    };
    let (out, delta) = penalize_cell(&cons, &acc, normal, dt, dv, 0);
    let (f_cart, torque) = cartesian_receipt(
        coords,
        ndim,
        axes,
        &geo.centroid,
        &x,
        &center,
        &delta.force_delta,
    );

    let mut writes: Writes = Vec::new();
    writes.push((
        "den_out".to_string(),
        symbi_ir::FieldRef::cons_den().into(),
        out.den.node(),
    ));
    for a in 0..ndim {
        writes.push((
            format!("mom_out_{a}"),
            symbi_ir::FieldRef::cons_mom(a as u8).into(),
            out.mom[a].node(),
        ));
    }
    writes.push((
        "nrg_out".to_string(),
        symbi_ir::FieldRef::cons_nrg().into(),
        out.nrg.node(),
    ));
    // the sink swallows gas and the dye dissolved in it together; `penalize_cell` applied the
    // same drain factor to both, so the concentration of the surviving gas is unchanged.
    if has_dye {
        writes.push((
            "chi_out".to_string(),
            symbi_ir::FieldRef::cons_chi().into(),
            out.chi.node(),
        ));
    }
    writes.push((
        "pen_mass".to_string(),
        "pen_0_mass".into(),
        delta.mass_delta.node(),
    ));
    for a in 0..ndim {
        writes.push((
            format!("pen_force_{a}"),
            format!("pen_0_force_{a}").into(),
            f_cart[a].node(),
        ));
    }
    writes.push((
        "pen_energy".to_string(),
        "pen_0_energy".into(),
        delta.energy_delta.node(),
    ));
    for a in torque_axes(ndim) {
        writes.push((
            format!("pen_torque_{a}"),
            format!("pen_0_torque_{a}").into(),
            torque[a].node(),
        ));
    }
    push_force_normal(&mut writes, ndim, &f_cart, &n_cart);

    let kernel = end_trace().with_derived_support(&writes);
    (kernel, writes)
}

/// the ISOTHERMAL twin: no energy channel (the drain scales den + mom; the
/// sound speed is the constant `cs` param), delta
/// outputs mass + force only. same [Drain] stack, same integrator — the iso
/// energy slot discards the e-channel by construction.
pub fn penalize_drain_iso_gv(
    coords: Coords,
    ndim: usize,
    dof: usize,
    axes: &[usize],
) -> (GvKernel, Writes) {
    use symbi_hydro::energy::IsoModel;
    assert!(
        (1..=3).contains(&ndim),
        "penalize_drain_iso_gv: ndim must be 1..=3"
    );
    begin_trace();
    let dt = Gv::scalar("dt");
    let cs = Gv::scalar("cs");
    let c_drain = Gv::scalar("c_drain");

    let den = Gv::field("den", symbi_ir::FieldRef::cons_den());
    let mom: Vec<Gv> = (0..dof)
        .map(|c| Gv::field(&format!("mom_{c}"), symbi_ir::FieldRef::cons_mom(c as u8)))
        .collect();

    let geo = cell_geometry_gv(coords, &vec![Spacing::Uniform; ndim], &axes[..ndim], ndim);
    let dv = Gv::ONE / geo.inv_volume;
    let mut min_w = gv_axis_width(0, Spacing::Uniform);
    for ax in 1..ndim {
        min_w = min_w.min(gv_axis_width(ax, Spacing::Uniform));
    }

    let center: [Gv; 3] = std::array::from_fn(|a| {
        if a < ndim {
            Gv::scalar(&format!("body_0_pos_{a}"))
        } else {
            Gv::ZERO
        }
    });
    // the mask distance is the PHYSICAL distance to the body: map the coordinate
    // centroid to cartesian (identity on a cartesian grid), then the euclidean SDF.
    let x_cart = centroid_to_cartesian(coords, ndim, axes, &geo.centroid);
    let x: [Gv; 3] = std::array::from_fn(|a| if a < ndim { x_cart[a] } else { Gv::ZERO });
    let sphere = body_mask_sdf(center);
    let chi = symbi_ib::sdf::chi(sphere.dist(x), min_w);
    tag_body_mask(&chi, coords, ndim, axes, None, false);
    let inv_tau = signal_speed(cs) / (c_drain * min_w);

    let kin = BodyKin::<Gv, 3> {
        u_solid: Tensor::zeros(),
        omega: Tensor::zeros(),
        e_wall: Gv::ZERO,
    };
    let mut acc = Relax::<Gv, 3>::none();
    Property::Drain { inv_tau }.contribute(chi, &kin, &mut acc);
    let cons = ConsG::<Gv, 3, IsoModel> {
        chi: Default::default(),
        den,
        mom: Tensor::new(std::array::from_fn(
            |a| if a < dof { mom[a] } else { Gv::ZERO },
        )),
        nrg: Default::default(),
    };
    let (out, delta) = penalize_cell(&cons, &acc, Tensor::zeros(), dt, dv, 0);
    // a bare drain has no wall surface -> a zero normal -> zero form drag (all force is accretion).
    let n_cart: Vec<Gv> = vec![Gv::ZERO; ndim];
    // the angular-momentum receipt, identical booking to the adiabatic twin.
    let (f_cart, torque) = cartesian_receipt(
        coords,
        ndim,
        axes,
        &geo.centroid,
        &x,
        &center,
        &delta.force_delta,
    );

    let mut writes: Writes = Vec::new();
    writes.push((
        "den_out".to_string(),
        symbi_ir::FieldRef::cons_den().into(),
        out.den.node(),
    ));
    for a in 0..dof {
        writes.push((
            format!("mom_out_{a}"),
            symbi_ir::FieldRef::cons_mom(a as u8).into(),
            out.mom[a].node(),
        ));
    }
    writes.push((
        "pen_mass".to_string(),
        "pen_0_mass".into(),
        delta.mass_delta.node(),
    ));
    for a in 0..ndim {
        writes.push((
            format!("pen_force_{a}"),
            format!("pen_0_force_{a}").into(),
            f_cart[a].node(),
        ));
    }
    for a in torque_axes(ndim) {
        writes.push((
            format!("pen_torque_{a}"),
            format!("pen_0_torque_{a}").into(),
            torque[a].node(),
        ));
    }
    push_force_normal(&mut writes, ndim, &f_cart, &n_cart);

    let kernel = end_trace().with_derived_support(&writes);
    (kernel, writes)
}

#[cfg(test)]
mod shaped_tests {
    use super::*;
    use symbi_ib::sdf::SdfExpr;

    // a shaped porous kernel bakes the CSG geometry as CONSTANTS and keeps only the body
    // POSITION as a runtime scalar — so a moving body rides the same kernel, updating just
    // `body_0_pos_*` per step. the sphere kernel instead reads a runtime `body_0_racc`.
    #[test]
    fn shaped_porous_kernel_bakes_geometry_and_keeps_position_runtime() {
        let shape = SdfExpr::<f64, 3>::cuboid([0.0, 0.0, 0.0], [0.5, 0.3, 0.2])
            .union(SdfExpr::sphere([0.6, 0.0, 0.0], 0.25));
        let (kernel, writes) = penalize_porous_gv_shaped(Coords::Cartesian, 3, 3, &shape, false);
        assert!(
            !kernel.graph.has_errors(),
            "shaped porous kernel traced with graph errors"
        );
        assert!(
            kernel.scalar_params.iter().any(|p| p == "body_0_pos_0"),
            "the runtime body position must remain a scalar (a moving body updates it)",
        );
        assert!(
            !kernel.scalar_params.iter().any(|p| p == "body_0_racc"),
            "shaped kernel must bake geometry, not read the sphere radius: {:?}",
            kernel.scalar_params,
        );
        for p in ["porosity", "k_eta_n", "k_eta_t", "body_0_vel_0", "dt"] {
            assert!(
                kernel.scalar_params.iter().any(|s| s == p),
                "missing runtime param {p}"
            );
        }
        assert!(!writes.is_empty(), "the shaped kernel emitted no writes");
    }

    // the unshaped path is the sphere: `body_0_racc` IS a runtime scalar (the AOT kernel).
    #[test]
    fn unshaped_porous_kernel_reads_the_runtime_radius() {
        let (kernel, _) = penalize_porous_gv(Coords::Cartesian, 3, 3, &[0, 1, 2], false);
        assert!(kernel.scalar_params.iter().any(|p| p == "body_0_racc"));
    }
}
