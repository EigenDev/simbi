// =============================================================================
// gv_penalize.rs
//
// the traced immersed-boundary penalization kernel:
// per cell, the sphere SDF's signed distance -> the mollified chi -> the
// property stack's PenalizationRelaxation accumulation -> the same carrier-generic
// `penalize_cell` that runs at f64, evaluated at Gv. the [Drain] stack is
// the p = 1 anchor: chi/tau on the rho channel only, every other channel's
// correction an exact arithmetic zero, so the kernel reduces bit-for-bit to
// `drain_cell`'s uniform scaling.
//
// buffers: cons (den, mom_0.., nrg) in place, plus per-body delta scratch
// (pen_0_mass, pen_0_force_{ax}, pen_0_energy) for the feedback reduction.
// scalars: dt, gamma, the grid (x_lo/dx/map_kind per axis), body_0_pos_*,
// body_0_racc (the mask radius), and the dimensionless crossing factor `c_drain`.
// A spherical drain uses the faster of its acoustic cell-crossing rate and the
// free-fall rate at the mask; a shaped porous wall uses the acoustic rate.
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
use symbi_carrier::Scalar;
use symbi_geometry::{Cylindrical, CylindricalRPhi, DiagonalMetric, Metric, Spherical};
use symbi_hydro::energy::{Adiabatic, Dyed, EnergyModel, EnergySlot, IsoModel};
use symbi_hydro::state::ConsG;
use symbi_ib::penalize::{BodyKin, PenalizationRelaxation, Property, penalize_cell};
use symbi_ib::sdf::SdfExpr;
use symbi_ir::gv::{KernelWrite, KernelWrites};
use symbi_ir::{CtCellCt, CtEdgeCt, CtFaceCt, CtScratchKey, CtWireName, PhysComp, Transverse};
use symbi_ir::{Gv, KernelProgram, ParamExpr, TraceCx};

/// the wall/drain relaxation signal speed: the fast magnetosonic speed `sqrt(c_s^2 + c_a^2)`,
/// with `c_a^2 = |B|^2 / rho` bound as the runtime `c_a2` scalar (the max over the interior, so the
/// wall stays a signal-crossing stiff in the low-beta regions a magnetized sink accumulates).
/// `c_a2 = 0` off MHD reduces it to the sound speed exactly, so a hydro run is unchanged.
fn signal_speed<'t>(cx: TraceCx<'t>, cs: Gv<'t>) -> Gv<'t> {
    (cs * cs + cx.scalar("c_a2")).sqrt()
}

use crate::coords::{Coords, Spacing};
use crate::gv::{cell_geometry_gv, gv_axis_width};
use symbi_ir::trace;

/// the 3-space axes whose torque component can be nonzero at dimension
/// `ndim`: rotation needs a plane, so 1d carries zero, 2d the z moment alone,
/// 3d all three.
pub fn torque_axes(ndim: usize) -> std::ops::Range<usize> {
    match ndim {
        3 => 0..3,
        2 => 2..3,
        _ => 0..0,
    }
}

/// map the coordinate cell centroid `(r, theta, phi)` / `(R, phi, z)` to cartesian
/// so the sphere SDF measures the physical distance to the body. on a curved grid the
/// cartesian difference is the one that carries a length — `sqrt((r - r_b)^2 + (theta -
/// theta_b)^2)` mixes a radius with an angle; on a cartesian grid the map is the identity. the
/// 2D cylindrical chart is the `(R, phi)` disk plane (`CylindricalRPhi`), matching
/// the geometric-source dispatch — one metric for the whole codebase.
/// the 2d cylindrical (r, z) axisymmetric section: the phi = 0 half-plane is
/// isometric to a 2d cartesian half-plane (h_r = h_z = 1), so an on-axis body's
/// mask distance is the plain euclidean |(r, z - z0)|, the section frame
/// rotations are identities, and the mask region is a genuine coordinate-space
/// ball. the (r, phi) disk (axes [0, 1]) keeps the curved-chart machinery.
fn is_cyl_rz(coords: Coords, ndim: usize, axes: &[usize]) -> bool {
    coords == Coords::Cylindrical && ndim == 2 && axes[..2] == [0, 2]
}

fn centroid_to_cartesian<'t>(
    coords: Coords,
    ndim: usize,
    axes: &[usize],
    centroid: &[Gv<'t>],
) -> Vec<Gv<'t>> {
    if is_cyl_rz(coords, ndim, axes) {
        return centroid[..ndim].to_vec();
    }
    fn run<'t, M, const D: usize>(m: M, x: &[Gv<'t>]) -> Vec<Gv<'t>>
    where
        M: Metric<Gv<'t>, D>,
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

/// rotate a cartesian vector into the cell's physical (orthonormal) frame — the
/// frame the substrate stores momentum in. the surface normal is a geometric
/// cartesian direction; the wall / torque-free split projects the physical
/// momentum onto it, so it must be rotated here. `x` is the coordinate centroid
/// (the rotation depends on the local basis). identity on a cartesian grid.
fn vector_from_cartesian<'t>(
    coords: Coords,
    ndim: usize,
    axes: &[usize],
    x: &[Gv<'t>],
    v_cart: &[Gv<'t>],
) -> Vec<Gv<'t>> {
    if is_cyl_rz(coords, ndim, axes) {
        return v_cart[..ndim].to_vec();
    }
    fn run<'t, M, const D: usize>(m: M, x: &[Gv<'t>], v: &[Gv<'t>]) -> Vec<Gv<'t>>
    where
        M: Metric<Gv<'t>, D> + DiagonalMetric<Gv<'t>, D>,
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

/// rotate a physical-frame vector into the global cartesian frame — used to book
/// the lab-frame torque `r_cart x F_cart` (the accreted force receipt is in the
/// physical frame; the cross product needs both vectors in one frame). identity
/// on a cartesian grid, so the cartesian torque is bit-unchanged.
fn vector_to_cartesian<'t>(
    coords: Coords,
    ndim: usize,
    axes: &[usize],
    x: &[Gv<'t>],
    v_phys: &[Gv<'t>],
) -> Vec<Gv<'t>> {
    if is_cyl_rz(coords, ndim, axes) {
        return v_phys[..ndim].to_vec();
    }
    fn run<'t, M, const D: usize>(m: M, x: &[Gv<'t>], v: &[Gv<'t>]) -> Vec<Gv<'t>>
    where
        M: Metric<Gv<'t>, D> + DiagonalMetric<Gv<'t>, D>,
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

/// append the form-drag receipt: the surface force projected onto the outward SDF normal,
/// `force_normal = (F.n_hat) n_hat`, in the cartesian frame the reduction sums (both `f_cart` and
/// `n_cart` are already cartesian). the tangential (skin-friction) part is recovered downstream as
/// `force - force_normal`. a bare drain passes `n_cart = 0`, so its form drag is exactly zero. this
/// is the last receipt block, appended after mass/force/energy/torque so every existing slot
/// keeps its index.
fn push_force_normal<'t>(
    writes: &mut KernelWrites,
    ndim: usize,
    f_cart: &Tensor<Gv<'t>, 3>,
    n_cart: &[Gv<'t>],
) {
    let mut f_dot_n = Gv::ZERO;
    for a in 0..ndim {
        f_dot_n = f_dot_n + f_cart[a] * n_cart[a];
    }
    for a in 0..ndim {
        writes.push(KernelWrite::new(
            format!("pen_force_normal_{a}"),
            format!("pen_0_force_normal_{a}"),
            (f_dot_n * n_cart[a]).node(),
        ));
    }
}

/// the cell's force receipt in the cartesian world frame plus its lab-frame
/// moment `r_cart x F_cart` about the body center. penalization acts in the
/// cell's physical orthonormal basis, which rotates from cell to cell on a
/// curvilinear chart — only cartesian components sum across cells to a
/// meaningful net force on the body (identity rotation on a cartesian grid).
/// returns `(f_cart, torque)`; 2d books only the z moment, 1d none.
fn cartesian_receipt<'t>(
    coords: Coords,
    ndim: usize,
    axes: &[usize],
    centroid: &[Gv<'t>],
    x_cart: &[Gv<'t>; 3],
    center: &[Gv<'t>; 3],
    force: &Tensor<Gv<'t>, 3>,
) -> (Tensor<Gv<'t>, 3>, Tensor<Gv<'t>, 3>) {
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
fn solid_velocity_phys<'t>(
    coords: Coords,
    ndim: usize,
    axes: &[usize],
    centroid: &[Gv<'t>],
    u_cart: &Tensor<Gv<'t>, 3>,
) -> Tensor<Gv<'t>, 3> {
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
/// single seam every penalization kernel (drain / porous-wall / torque-free, adiabatic and
/// iso) shares, so one mask definition serves them all.
fn body_mask_sdf<'t>(cx: TraceCx<'t>, center: [Gv<'t>; 3]) -> SdfExpr<Gv<'t>, 3> {
    body_mask_sdf_shaped(cx, center, None)
}

/// the mask geometry with an optional config CSG. `None`: a sphere of runtime radius
/// `body_0_racc` (the AOT kernel). `Some(shape)`: the shape — authored in the body-local
/// frame as f64 constants — lifted to Gv constants and translated to the runtime center. so a
/// moving body rides the same kernel: only `body_0_pos_*` changes per step; the shape geometry
/// is baked in as constants. (rotation would compose a runtime orientation transform on
/// `x` before the shape; translation alone is the moving-body core.)
///
/// `center` is the body's center of mass and its mask/geometric center at once — they coincide for
/// a symmetric body (sphere, symmetric CSG). everything dynamical (translation, gravity, the
/// omega x r wall velocity, the torque moment arm `x - center`) is referenced to the com, so an
/// asymmetric mass distribution would offset the mask placement alone.
/// the drain rate for a spherical accretor: the faster of one acoustic-cell
/// crossing and free fall through the mask,
/// `max(signal_speed / (c_drain dx), sqrt(GM / r_acc^3))`.
///
/// `c_drain = 1` gives one signal-crossing time and the gravitational arm
/// prevents a resolved accretor from becoming less absorbing than free fall.
/// The state-dependent acoustic arm is non-self-amplifying under the pure
/// drain: uniform conserved scaling preserves velocity, specific internal
/// energy, and hence sound speed exactly.
fn body_mask_sdf_shaped<'t>(
    cx: TraceCx<'t>,
    center: [Gv<'t>; 3],
    shape: Option<&SdfExpr<f64, 3>>,
) -> SdfExpr<Gv<'t>, 3> {
    match shape {
        None => SdfExpr::<Gv, 3>::Sphere {
            center,
            radius: cx.scalar("body_0_racc"),
        },
        Some(s) => s.lift(&|c| Gv::from_f64(c)).translated(center),
    }
}

/// the spinning body's mask: the shape lifted to Gv constants, rotated by the runtime orientation
/// matrix `R` (the body's full 3D orientation, integrated on the CPU from its angular velocity and
/// read here as the 9 row-major scalars `body_0_rot_0..8`), then translated to the runtime body
/// position. one kernel handles any orientation, so a freely-tumbling body reuses it; the mask + its
/// Dual-autodiff normal track `R` as it evolves.
fn body_mask_sdf_spinning<'t>(
    cx: TraceCx<'t>,
    center: [Gv<'t>; 3],
    shape: &SdfExpr<f64, 3>,
) -> SdfExpr<Gv<'t>, 3> {
    let rot: [[Gv; 3]; 3] = std::array::from_fn(|i| {
        std::array::from_fn(|j| cx.scalar(&format!("body_0_rot_{}", i * 3 + j)))
    });
    shape
        .lift(&|c| Gv::from_f64(c))
        .rotated(rot)
        .translated(center)
}

/// the saturation lemma at the mask seam: `chi = 0.5(1 - tanh(phi/w))` is
/// exactly zero in f64 once `phi > DRAIN_SUPPORT_WIDTHS * w`, so the mask is
/// supported by its geometry's bounding ball padded by that many cell widths.
/// tagging chi where the mask is built lets `with_derived_support`
/// propagate the ball to every write, keeping support and mask in step.
/// the ball is a coordinate-space region: the identity chart alone can spell
/// the cartesian mask ball in grid coordinates, so curvilinear kernels stay
/// untagged and derive Everywhere (dispatch already falls back to the whole
/// interior off-cartesian). a spinning shape sweeps every orientation about
/// the body position, so its ball is position-centered with the body-local
/// offset folded into the radius.
fn tag_body_mask<'t>(
    cx: TraceCx<'t>,
    chi: &Gv<'t>,
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
    cx.tag_support_ball(chi, center, radius);
}

/// the 2.5D immersed-body ohmic resistive edge EMF: adds a body-localized resistive current
/// `eta * chi(x) * J_z` to the out-of-plane edge EMF `ez` (efield[0]) in place, where `J_z =
/// dB_y/dx - dB_x/dy` is the same adjoint current as the generic resistive kernel and `chi` is the
/// body mask (`0.5(1 - tanh(phi/w))`, 1 inside the body, 0 outside, mollified over one cell). the
/// localized resistivity dissipates the magnetic field threading the body while leaving the exterior
/// flux untouched. div-B-clean: the same curl consumes the augmented EMF (`div(curl) = 0`). stability:
/// the composed operator `-curl(eta chi J)` is `-C diag(eta chi) C^T`, negative-definite for any
/// `eta chi >= 0` — the mask reweights the edges of the already-adjoint current, so the body's
/// action is purely dissipative. `chi` is sampled at the E_z corner (the edge location), half a cell
/// below the traced cell centroid on each in-plane axis.
pub fn body_resistive_emf_2d_gv(coords: Coords) -> KernelProgram {
    let ndim = 2usize;
    let axes: &[usize] = &[0, 1, 2][..ndim];
    let (kernel, writes) = trace(|cx| {
        let ez = crate::gv::ct_field(cx, CtScratchKey::edge(CtEdgeCt::Emf, CtWireName::Ez));
        let eta = cx.scalar("eta");
        let bx = crate::gv::ct_field(
            cx,
            CtScratchKey::face(CtFaceCt::BFace(Transverse::A), CtWireName::Bx),
        );
        let by = crate::gv::ct_field(
            cx,
            CtScratchKey::face(CtFaceCt::BFace(Transverse::B), CtWireName::By),
        );
        let bx_jm = crate::gv::ct_field_at(
            cx,
            CtScratchKey::face(CtFaceCt::BFace(Transverse::A), CtWireName::Bx),
            2,
            &[0, -1],
        ); // B_x at the neighbor below in y
        let by_im = crate::gv::ct_field_at(
            cx,
            CtScratchKey::face(CtFaceCt::BFace(Transverse::B), CtWireName::By),
            2,
            &[-1, 0],
        ); // B_y at the neighbor behind in x
        let dx0 = cx.scalar("dx_0");
        let dx1 = cx.scalar("dx_1");
        let jz = (Gv::ONE / dx0) * (by - by_im) - (Gv::ONE / dx1) * (bx - bx_jm);

        // the geometry scaffold yields the cell centroid; the E_z corner is half a cell below it on each
        // in-plane axis. sample the body mask at that corner so the dissipation registers with the edge.
        let geo = cell_geometry_gv(
            cx,
            coords,
            &vec![Spacing::Uniform; ndim],
            &axes[..ndim],
            ndim,
        );
        let half = Gv::from_f64(0.5);
        let corner = vec![geo.centroid[0] - half * dx0, geo.centroid[1] - half * dx1];
        let x_cart = centroid_to_cartesian(coords, ndim, axes, &corner);
        let x: [Gv; 3] = std::array::from_fn(|a| if a < ndim { x_cart[a] } else { Gv::ZERO });
        let center: [Gv; 3] = std::array::from_fn(|a| {
            if a < ndim {
                cx.scalar(&format!("body_0_pos_{a}"))
            } else {
                Gv::ZERO
            }
        });
        let min_w = dx0.min(dx1);
        let phi = body_mask_sdf(cx, center).dist(x);
        let chi = symbi_ib::sdf::chi(phi, min_w);
        tag_body_mask(cx, &chi, coords, ndim, axes, None, false);

        let ez_new = ez + eta * chi * jz;
        vec![KernelWrite::new(
            "ez_new",
            CtScratchKey::edge(CtEdgeCt::Emf, CtWireName::Ez),
            ez_new.node(),
        )]
    });
    let kernel = kernel.with_derived_support(&writes);
    KernelProgram::new(kernel, writes)
}

/// the 3D cartesian body-masked ohmic resistive edge EMF along edge `dir`: adds `eta * chi * J_dir`
/// to that edge's EMF in place, where `chi` is the body indicator sampled at the dir-edge and
/// `J_dir = dB_p2/dx_p1 - dB_p1/dx_p2` (p1=(dir+1)%3, p2=(dir+2)%3) is the current from the two
/// transverse faces. it is the bulk 3D resistive EMF (`rmhd_resistive_emf_3d_dir_gv`) gated by the
/// mask, so the same div-B-clean 3D curl consumes it and the composed operator is the mask-weighted
/// negative-definite laplacian: the body can only shed the field threading it.
pub fn body_resistive_emf_3d_dir_gv(dir: usize, coords: Coords) -> KernelProgram {
    let ndim = 3usize;
    let axes: &[usize] = &[0, 1, 2][..ndim];
    let p1 = (dir + 1) % 3;
    let p2 = (dir + 2) % 3;
    let (kernel, writes) = trace(|cx| {
        let emf = crate::gv::ct_field(cx, CtEdgeCt::Emf);
        let eta = cx.scalar("eta");
        let b_p1 = crate::gv::ct_field(
            cx,
            CtScratchKey::face(CtFaceCt::BFace(Transverse::A), CtWireName::BP1),
        );
        let b_p2 = crate::gv::ct_field(
            cx,
            CtScratchKey::face(CtFaceCt::BFace(Transverse::B), CtWireName::BP2),
        );
        let back = |ax: usize| -> [i32; 3] {
            let mut o = [0, 0, 0];
            o[ax] = -1;
            o
        };
        let b_p1_m = crate::gv::ct_field_at(
            cx,
            CtScratchKey::face(CtFaceCt::BFace(Transverse::A), CtWireName::BP1),
            3,
            &back(p2),
        ); // dB_p1/dx_p2, backward
        let b_p2_m = crate::gv::ct_field_at(
            cx,
            CtScratchKey::face(CtFaceCt::BFace(Transverse::B), CtWireName::BP2),
            3,
            &back(p1),
        ); // dB_p2/dx_p1, backward
        let dxp1 = cx.scalar(&format!("dx_{p1}"));
        let dxp2 = cx.scalar(&format!("dx_{p2}"));
        let j = (Gv::ONE / dxp1) * (b_p2 - b_p2_m) - (Gv::ONE / dxp2) * (b_p1 - b_p1_m);

        // the dir-edge sits at the cell centroid shifted half a cell back on the two transverse axes
        // (p1, p2) and centered along dir. sample the body mask at that edge so the dissipation registers
        // with the edge EMF.
        let dx: Vec<Gv> = (0..ndim).map(|a| cx.scalar(&format!("dx_{a}"))).collect();
        let geo = cell_geometry_gv(
            cx,
            coords,
            &vec![Spacing::Uniform; ndim],
            &axes[..ndim],
            ndim,
        );
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
                cx.scalar(&format!("body_0_pos_{a}"))
            } else {
                Gv::ZERO
            }
        });
        let min_w = dx.iter().copied().reduce(|a, b| a.min(b)).unwrap();
        let phi = body_mask_sdf(cx, center).dist(x);
        let chi = symbi_ib::sdf::chi(phi, min_w);
        tag_body_mask(cx, &chi, coords, ndim, axes, None, false);

        let emf_new = emf + eta * chi * j;
        vec![KernelWrite::new("emf_new", CtEdgeCt::Emf, emf_new.node())]
    });
    let kernel = kernel.with_derived_support(&writes);
    KernelProgram::new(kernel, writes)
}

/// the magnetic-slip cell pass (3D cartesian): the per-cell quadrature vector
/// `F_q = A(B_q) (R J)_q`, the intermediate the edge pass scatters to the oriented edge EMF.
/// `B_q` is the cell-centered field `bcell` (the forward face-average). `(R J)_q[d]` gathers the
/// four `d`-edge currents bounding the cell (forward transverse offsets, weight 1/4), each edge
/// current the discrete `curl B` from backward face differences. the dyad coefficient
/// `a_B chi_B` closes on the local drain rate `lambda_rho` (the same `spherical_drain_rate` the
/// material drain applies) and the shell mask `chi_B`, so `a_B = ell_B^2 lambda_rho / ((|B|^2 +
/// B_0^2) D_B)`. writes `fq_0..2`. explicit oracle: it validates the operator's signs, staggering,
/// and energy law and carries the diffusive stability limit `dt <~ dx^2 / eta_B`.
/// the gathered midpoint current `(R J)_q[d]` at the cell, `d = 0..3`: 1/4 the sum over the four
/// d-edges bounding the cell (forward transverse offsets) of the edge current
/// `(curl B)_d = dB_p2/dx_p1 - dB_p1/dx_p2`, each from backward differences of the staggered
/// face field `bf_c` (`CtFaceCt::BFaceComp`). the slip cell pass forms its dyad on this gather and
/// the dissipation pass contracts it with the dyad output, so `<(R J)_q, F_q>` per cell is the
/// midpoint quadratic of the operator.
fn gathered_current_gv<'t>(cx: TraceCx<'t>, inv_dx: &[Gv<'t>]) -> Vec<Gv<'t>> {
    let bface = |c: usize, off: &[i32]| {
        crate::gv::gv_field_at(cx, &format!("bf_{c}"), CtFaceCt::BFaceComp(PhysComp::new(c)), 3, off)
    };
    let quarter = Gv::from_f64(0.25);
    (0..3)
        .map(|d| {
            let p1 = (d + 1) % 3;
            let p2 = (d + 2) % 3;
            let curl_at = |eoff: [i32; 3]| -> Gv<'t> {
                let mut o_mp1 = eoff;
                o_mp1[p1] -= 1;
                let mut o_mp2 = eoff;
                o_mp2[p2] -= 1;
                (bface(p2, &eoff) - bface(p2, &o_mp1)) * inv_dx[p1]
                    - (bface(p1, &eoff) - bface(p1, &o_mp2)) * inv_dx[p2]
            };
            let mut e1 = [0, 0, 0];
            e1[p1] = 1;
            let mut e2 = [0, 0, 0];
            e2[p2] = 1;
            let mut e12 = [0, 0, 0];
            e12[p1] = 1;
            e12[p2] = 1;
            (curl_at([0, 0, 0]) + curl_at(e1) + curl_at(e2) + curl_at(e12)) * quarter
        })
        .collect()
}

/// the entering gas internal energy density of the magnetic-slip predictor,
/// `e_g^0 = E - |mom|^2/(2 rho) - |B_cell|^2/2`, written to the `slip_ge` cell scratch the slip
/// coefficient's sound speed reads. 3D cartesian adiabatic MHD.
pub fn slip_gas_energy_3d_gv() -> KernelProgram {
    let (kernel, writes) = trace(|cx| {
        let den = cx.field("den", symbi_ir::FieldRef::cons_den());
        let mom: Vec<Gv> = (0..3)
            .map(|c| cx.field(&format!("mom_{c}"), symbi_ir::FieldRef::cons_mom(c as u8)))
            .collect();
        let nrg = cx.field("nrg", symbi_ir::FieldRef::cons_nrg());
        let b_q: Vec<Gv> = (0..3)
            .map(|c| cx.field(&format!("bc_{c}"), symbi_ir::FieldRef::BCell(c as u8)))
            .collect();
        let half = Gv::from_f64(0.5);
        let mom_sq = mom.iter().fold(Gv::ZERO, |s, m| s + *m * *m);
        let b_sq = b_q.iter().fold(Gv::ZERO, |s, b| s + *b * *b);
        let e_g = nrg - half * mom_sq / den - half * b_sq;
        vec![KernelWrite::new("slip_ge_new", CtCellCt::SlipGasEnergy, e_g.node())]
    });
    KernelProgram::new(kernel, writes)
}

/// the per-cell magnetic dissipation rate of the slip operator on the state the quadrature was
/// formed on, `qdot_c = (R J)_c . F_q,c`, with `(R J)_c` gathered from the staggered field the
/// kernel is bound to and `F_q` the slip cell pass output. nonnegative cell by cell, since
/// `F_q = A (R J)` with `A` positive semidefinite. 3D cartesian.
pub fn slip_dissipation_3d_gv() -> KernelProgram {
    let (kernel, writes) = trace(|cx| {
        let inv_dx: Vec<Gv> = (0..3)
            .map(|a| Gv::ONE / cx.scalar(&format!("dx_{a}")))
            .collect();
        let j_q = gathered_current_gv(cx, &inv_dx);
        let f_q: Vec<Gv> = (0..3)
            .map(|c| crate::gv::ct_field(cx, CtCellCt::SlipQuadrature(PhysComp::new(c))))
            .collect();
        let qdot = (0..3).fold(Gv::ZERO, |s, c| s + j_q[c] * f_q[c]);
        vec![KernelWrite::new("slip_qdot_new", CtCellCt::SlipDissipation, qdot.node())]
    });
    let kernel = kernel.with_derived_support(&writes);
    KernelProgram::new(kernel, writes)
}

/// the magnetic-slip commit's energy deposit through the face-to-cell defect bridge:
/// `E += (M_cc^1 - M_cc^0) + dt qdot`, with `M_cc^1 = |B_cell|^2/2` from the committed cell field,
/// `M_cc^0 = |interp(B^0)|^2/2` from the substep-input faces the kernel is bound to (the forward
/// face average), and `qdot` the midpoint dissipation rate. the gas internal energy then gains
/// exactly `dt qdot` per cell and the cell-summed total energy changes by the operator's
/// face-Hodge dissipation alone. 3D cartesian adiabatic MHD; writes the total energy in place.
pub fn slip_commit_energy_3d_gv() -> KernelProgram {
    let (kernel, writes) = trace(|cx| {
        let dt = cx.scalar("dt");
        let nrg = cx.field("nrg", symbi_ir::FieldRef::cons_nrg());
        let b_new: Vec<Gv> = (0..3)
            .map(|c| cx.field(&format!("bc_{c}"), symbi_ir::FieldRef::BCell(c as u8)))
            .collect();
        let half = Gv::from_f64(0.5);
        let b_old: Vec<Gv> = (0..3)
            .map(|c| {
                let mut up = [0, 0, 0];
                up[c] = 1;
                let lo = crate::gv::gv_field_at(
                    cx,
                    &format!("bf_{c}"),
                    CtFaceCt::BFaceComp(PhysComp::new(c)),
                    3,
                    &[0, 0, 0],
                );
                let hi = crate::gv::gv_field_at(
                    cx,
                    &format!("bf_{c}"),
                    CtFaceCt::BFaceComp(PhysComp::new(c)),
                    3,
                    &up,
                );
                half * (lo + hi)
            })
            .collect();
        let qdot = crate::gv::ct_field(cx, CtCellCt::SlipDissipation);
        let m_new = b_new.iter().fold(Gv::ZERO, |s, b| s + *b * *b) * half;
        let m_old = b_old.iter().fold(Gv::ZERO, |s, b| s + *b * *b) * half;
        let e = nrg + (m_new - m_old) + dt * qdot;
        vec![KernelWrite::new("nrg_new", symbi_ir::FieldRef::cons_nrg(), e.node())]
    });
    let kernel = kernel.with_derived_support(&writes);
    KernelProgram::new(kernel, writes)
}

/// the frozen dyad coefficient `a_B chi_B` of the slip operator at the cell, from the coefficient
/// state `b_q` (the cell-centered field the operator is frozen on) and the cell gas state. the
/// local drain rate lambda_rho = 1/tau_rho is the one the material drain reads: the fast-
/// magnetosonic cell-crossing rate on the cell's own local Alfven term |B|^2/rho lifted by free
/// fall, on the gas sound speed from the predicted midpoint gas internal energy density e_g* staged
/// in `slip_ge` (cs^2 = gamma(gamma-1) e_g*/rho), so the drain clock tracks the substep-midpoint gas
/// state without consulting the endpoint-reconciled total energy. the shell mask chi_B reads the
/// physical distance from the cell center to the body surface, and
/// a_B = ell_B^2 / ((|B|^2 + B_0^2) D_B tau_rho) with ell_B = slip_length_ratio w.
fn slip_dyad_coefficient_gv<'t>(
    cx: TraceCx<'t>,
    coords: Coords,
    ndim: usize,
    axes: &[usize],
    b_q: &[Gv<'t>],
) -> Gv<'t> {
    let gamma = cx.scalar("gamma");
    let den = cx.field("den", symbi_ir::FieldRef::cons_den());
    let b_sq = b_q.iter().fold(Gv::ZERO, |s, b| s + *b * *b);
    let e_g = crate::gv::ct_field(cx, CtCellCt::SlipGasEnergy);
    let cs = (gamma * (gamma - Gv::ONE) * e_g / den).sqrt();
    let c_drain = cx.scalar("c_drain");
    let mut min_w = gv_axis_width(cx, 0, Spacing::Uniform);
    for ax in 1..ndim {
        min_w = min_w.min(gv_axis_width(cx, ax, Spacing::Uniform));
    }
    let inv_cd_dx = Gv::ONE / (c_drain * min_w);
    let lambda_rho = symbi_ib::drain::local_drain_rate(
        cs,
        b_sq,
        den,
        inv_cd_dx,
        cx.scalar("body_0_mass"),
        cx.scalar("body_0_racc"),
    );

    let geo = cell_geometry_gv(cx, coords, &vec![Spacing::Uniform; ndim], &axes[..ndim], ndim);
    // the body position carries one scalar per gridded axis; an ungridded slot sits at zero for
    // both the center and the cell, so the mask distance of a 2.5D run is the in-plane distance:
    // the removal region is a cylinder along the missing axis.
    let center: [Gv; 3] = std::array::from_fn(|a| {
        if a < ndim {
            cx.scalar(&format!("body_0_pos_{a}"))
        } else {
            Gv::ZERO
        }
    });
    let x_cart = centroid_to_cartesian(coords, ndim, axes, &geo.centroid);
    let x: [Gv; 3] = std::array::from_fn(|a| if a < x_cart.len() { x_cart[a] } else { Gv::ZERO });
    let phi = body_mask_sdf(cx, center).dist(x);
    let w = cx.scalar("slip_w");
    let placement = cx.scalar("slip_placement");
    let chi_b = symbi_ib::magnetic_slip::chi_shell(phi, w, placement);
    tag_body_mask(cx, &chi_b, coords, ndim, axes, None, false);

    let ell_b = cx.scalar("slip_ell_ratio") * w;
    let a_b = symbi_ib::magnetic_slip::slip_coefficient(
        ell_b,
        b_sq,
        cx.scalar("slip_b0"),
        cx.scalar("slip_d_b"),
        Gv::ONE / lambda_rho,
    );
    a_b * chi_b
}

/// the 2.5D current of the operand at the cell, `R B = (D_y B_z, -D_x B_z, G_z(B_x, B_y))`: central
/// differences of the operand's cell `B_z` (`slip_bz`) and the 1/4-weighted gather of the four corner
/// z-edge currents from backward differences of the staggered faces `bf_c`.
fn mixed_current_2d_gv<'t>(cx: TraceCx<'t>, inv_dx: &[Gv<'t>]) -> [Gv<'t>; 3] {
    let half = Gv::from_f64(0.5);
    let bz = |off: &[i32]| crate::gv::gv_field_at(cx, "slip_bz", CtCellCt::SlipOperandBz, 2, off);
    let bface = |c: usize, off: &[i32]| {
        crate::gv::gv_field_at(cx, &format!("bf_{c}"), CtFaceCt::BFaceComp(PhysComp::new(c)), 2, off)
    };
    let j_x = (bz(&[0, 1]) - bz(&[0, -1])) * half * inv_dx[1];
    let j_y = Gv::ZERO - (bz(&[1, 0]) - bz(&[-1, 0])) * half * inv_dx[0];
    let curl_at = |eoff: [i32; 2]| -> Gv<'t> {
        let mx = [eoff[0] - 1, eoff[1]];
        let my = [eoff[0], eoff[1] - 1];
        (bface(1, &eoff) - bface(1, &mx)) * inv_dx[0] - (bface(0, &eoff) - bface(0, &my)) * inv_dx[1]
    };
    let j_z = (curl_at([0, 0]) + curl_at([1, 0]) + curl_at([0, 1]) + curl_at([1, 1]))
        * Gv::from_f64(0.25);
    [j_x, j_y, j_z]
}

/// the 2.5D per-cell magnetic dissipation rate `qdot = (R B) . F_q` with the mixed current gathered
/// from the bound faces and the operand's cell `B_z`: all three components of the current enter,
/// so the predicted heat and the committed heat carry the out-of-plane channel.
pub fn slip_dissipation_2d_gv() -> KernelProgram {
    let (kernel, writes) = trace(|cx| {
        let inv_dx: Vec<Gv> = (0..2)
            .map(|a| Gv::ONE / cx.scalar(&format!("dx_{a}")))
            .collect();
        let j = mixed_current_2d_gv(cx, &inv_dx);
        let f_q: Vec<Gv> = (0..3)
            .map(|c| crate::gv::ct_field(cx, CtCellCt::SlipQuadrature(PhysComp::new(c))))
            .collect();
        let qdot = (0..3).fold(Gv::ZERO, |s, c| s + j[c] * f_q[c]);
        vec![KernelWrite::new("slip_qdot_new", CtCellCt::SlipDissipation, qdot.node())]
    });
    let kernel = kernel.with_derived_support(&writes);
    KernelProgram::new(kernel, writes)
}

/// the 2.5D commit's energy deposit through the face-to-cell defect bridge:
/// `E += dM_cell + dt qdot` with `dM_cell = dM_xy,interp + (B_z^1)^2/2 - (B_z^0)^2/2`. the in-plane
/// part is the change of the interpolated cell energy, `M_xy^1` from the committed cell field and
/// `M_xy^0` from the substep-input faces the kernel is bound to; the out-of-plane part carries no
/// defect, `B_z` being cell-centered in both states, with `B_z^0` read from the bound `slip_bz`.
pub fn slip_commit_energy_2d_gv() -> KernelProgram {
    let (kernel, writes) = trace(|cx| {
        let dt = cx.scalar("dt");
        let nrg = cx.field("nrg", symbi_ir::FieldRef::cons_nrg());
        let b_new: Vec<Gv> = (0..3)
            .map(|c| cx.field(&format!("bc_{c}"), symbi_ir::FieldRef::BCell(c as u8)))
            .collect();
        let half = Gv::from_f64(0.5);
        let b_old_xy: Vec<Gv> = (0..2)
            .map(|c| {
                let mut up = [0, 0];
                up[c] = 1;
                let lo = crate::gv::gv_field_at(cx, &format!("bf_{c}"), CtFaceCt::BFaceComp(PhysComp::new(c)), 2, &[0, 0]);
                let hi = crate::gv::gv_field_at(cx, &format!("bf_{c}"), CtFaceCt::BFaceComp(PhysComp::new(c)), 2, &up);
                half * (lo + hi)
            })
            .collect();
        let bz_old = crate::gv::ct_field(cx, CtCellCt::SlipOperandBz);
        let qdot = crate::gv::ct_field(cx, CtCellCt::SlipDissipation);
        let m_new = b_new.iter().fold(Gv::ZERO, |s, b| s + *b * *b) * half;
        let m_old = (b_old_xy[0] * b_old_xy[0] + b_old_xy[1] * b_old_xy[1] + bz_old * bz_old) * half;
        let e = nrg + (m_new - m_old) + dt * qdot;
        vec![KernelWrite::new("nrg_new", symbi_ir::FieldRef::cons_nrg(), e.node())]
    });
    let kernel = kernel.with_derived_support(&writes);
    KernelProgram::new(kernel, writes)
}

/// the 2.5D slip cell pass (cartesian x-y grid, three vector components). the operand lives on the
/// mixed complex F_x + F_y + C_z: the in-plane components on staggered faces, the out-of-plane
/// `B_z` at cell centers (`slip_bz`). its current at the cell is
///   R B = (D_y B_z, -D_x B_z, G_z(B_x, B_y)),
/// with D_x, D_y central differences of the cell `B_z` and G_z the 1/4-weighted gather of the four
/// corner z-edge currents (backward face differences), so the dissipative operator R* A R closes on
/// the face-plus-cell inner product with R* the exact transpose. the dyad coefficient is frozen on
/// the cell field `bc_c` (the interpolated in-plane components and the coefficient state's `B_z`),
/// which coincides with the operand for the explicit operator. writes F_q = a_B chi_B (|B|^2 I - B B) R B.
pub fn body_slip_quadrature_2d_gv(coords: Coords) -> KernelProgram {
    let ndim = 2usize;
    let axes: &[usize] = &[0, 1];
    let (kernel, writes) = trace(|cx| {
        let b_q: Vec<Gv> = (0..3)
            .map(|c| cx.field(&format!("bc_{c}"), symbi_ir::FieldRef::BCell(c as u8)))
            .collect();
        let inv_dx: Vec<Gv> = (0..2)
            .map(|a| Gv::ONE / cx.scalar(&format!("dx_{a}")))
            .collect();
        let [j_x, j_y, j_z] = mixed_current_2d_gv(cx, &inv_dx);

        let coeff = slip_dyad_coefficient_gv(cx, coords, ndim, axes, &b_q);
        let b_vec = Tensor::new([b_q[0], b_q[1], b_q[2]]);
        let j_vec = Tensor::new([j_x, j_y, j_z]);
        let f_q = symbi_ib::magnetic_slip::slip_apply(coeff, &b_vec, &j_vec);
        (0..3)
            .map(|c| {
                KernelWrite::new(
                    format!("fq_{c}_new"),
                    CtCellCt::SlipQuadrature(PhysComp::new(c)),
                    f_q[c].node(),
                )
            })
            .collect::<Vec<_>>()
    });
    let kernel = kernel.with_derived_support(&writes);
    KernelProgram::new(kernel, writes)
}

/// the 2.5D slip edge pass: scatter the out-of-plane quadrature component to the corner z-edge EMF,
/// `efield += (R^* F)_z`, reading `fq_2` at the four cells sharing the corner (backward offsets,
/// weight 1/4): the exact transpose of the cell pass's corner gather. the shared CT curl then
/// consumes the augmented EMF, so the in-plane transport is div-B-clean.
pub fn body_slip_emf_2d_gv() -> KernelProgram {
    let (kernel, writes) = trace(|cx| {
        let emf = crate::gv::ct_field(cx, CtEdgeCt::Emf);
        let fq = |off: &[i32]| {
            crate::gv::gv_field_at(cx, "fq_2", CtCellCt::SlipQuadrature(PhysComp::new(2)), 2, off)
        };
        let scattered = (fq(&[0, 0]) + fq(&[-1, 0]) + fq(&[0, -1]) + fq(&[-1, -1])) * Gv::from_f64(0.25);
        let emf_new = emf + scattered;
        vec![KernelWrite::new("emf_new", CtEdgeCt::Emf, emf_new.node())]
    });
    let kernel = kernel.with_derived_support(&writes);
    KernelProgram::new(kernel, writes)
}

/// the 2.5D slip out-of-plane update, `B_z -= dt (D_x F_y - D_y F_x)`: the transpose of the cell
/// current gather `(D_y B_z, -D_x B_z)` applied to the in-plane quadrature components, in flux form
/// (the central difference is the divergence of the face-averaged flux `(F_y, -F_x)`, so the update
/// telescopes). in place on the operand's `B_z` slot.
pub fn body_slip_bz_2d_gv() -> KernelProgram {
    let (kernel, writes) = trace(|cx| {
        let dt = cx.scalar("dt");
        let inv_dx: Vec<Gv> = (0..2)
            .map(|a| Gv::ONE / cx.scalar(&format!("dx_{a}")))
            .collect();
        let half = Gv::from_f64(0.5);
        let bz = crate::gv::ct_field(cx, CtCellCt::SlipOperandBz);
        let fq = |c: usize, off: &[i32]| {
            crate::gv::gv_field_at(cx, &format!("fq_{c}"), CtCellCt::SlipQuadrature(PhysComp::new(c)), 2, off)
        };
        let dx_fy = (fq(1, &[1, 0]) - fq(1, &[-1, 0])) * half * inv_dx[0];
        let dy_fx = (fq(0, &[0, 1]) - fq(0, &[0, -1])) * half * inv_dx[1];
        let bz_new = bz - dt * (dx_fy - dy_fx);
        vec![KernelWrite::new("bz_new", CtCellCt::SlipOperandBz, bz_new.node())]
    });
    let kernel = kernel.with_derived_support(&writes);
    KernelProgram::new(kernel, writes)
}

pub fn body_slip_quadrature_3d_gv(coords: Coords) -> KernelProgram {
    let ndim = 3usize;
    let axes: &[usize] = &[0, 1, 2];
    let (kernel, writes) = trace(|cx| {
        // B_q: the cell-centered magnetic field (the forward face-average bcell).
        let b_q: Vec<Gv> = (0..3)
            .map(|c| cx.field(&format!("bc_{c}"), symbi_ir::FieldRef::BCell(c as u8)))
            .collect();

        let inv_dx: Vec<Gv> = (0..3)
            .map(|a| Gv::ONE / cx.scalar(&format!("dx_{a}")))
            .collect();

        let j_q = gathered_current_gv(cx, &inv_dx);

        let coeff = slip_dyad_coefficient_gv(cx, coords, ndim, axes, &b_q);
        let b_vec = Tensor::new([b_q[0], b_q[1], b_q[2]]);
        let j_vec = Tensor::new([j_q[0], j_q[1], j_q[2]]);
        let f_q = symbi_ib::magnetic_slip::slip_apply(coeff, &b_vec, &j_vec);

        (0..3)
            .map(|c| {
                KernelWrite::new(
                    format!("fq_{c}_new"),
                    CtCellCt::SlipQuadrature(PhysComp::new(c)),
                    f_q[c].node(),
                )
            })
            .collect::<Vec<_>>()
    });
    let kernel = kernel.with_derived_support(&writes);
    KernelProgram::new(kernel, writes)
}

/// the magnetic-slip edge pass (3D cartesian) along edge `dir`: scatter the cell-quadrature vector
/// to the oriented edge EMF, `efield[dir] += (R^* F)_dir`. the weighted-adjoint scatter reads the
/// `dir`-component `fq_dir` at the four cells sharing the edge (backward transverse offsets, the
/// exact transpose of the cell pass's forward gather), weight 1/4. the augmented EMF rides the
/// shared CT curl, so the slip transport is div-B-clean. explicit oracle.
pub fn body_slip_emf_3d_dir_gv(dir: usize, _coords: Coords) -> KernelProgram {
    let p1 = (dir + 1) % 3;
    let p2 = (dir + 2) % 3;
    let (kernel, writes) = trace(|cx| {
        let emf = crate::gv::ct_field(cx, CtEdgeCt::Emf);
        let fq = |off: &[i32]| {
            crate::gv::gv_field_at(cx, &format!("fq_{dir}"), CtCellCt::SlipQuadrature(PhysComp::new(dir)), 3, off)
        };
        let mut om1 = [0, 0, 0];
        om1[p1] = -1;
        let mut om2 = [0, 0, 0];
        om2[p2] = -1;
        let mut om12 = [0, 0, 0];
        om12[p1] = -1;
        om12[p2] = -1;
        let scattered =
            (fq(&[0, 0, 0]) + fq(&om1) + fq(&om2) + fq(&om12)) * Gv::from_f64(0.25);
        let emf_new = emf + scattered;
        vec![KernelWrite::new("emf_new", CtEdgeCt::Emf, emf_new.node())]
    });
    let kernel = kernel.with_derived_support(&writes);
    KernelProgram::new(kernel, writes)
}

/// trace the [Drain]-stack penalization for the adiabatic regime, cartesian. `ndim` is the spatial
/// grid dimension (geometry, mask, force receipt); `dof` is the momentum count (the conserved
/// 3-vector's active components). `dof == ndim` for hydro and full 3D MHD; `dof = 3, ndim = 2` for
/// 2.5D MHD, where the out-of-plane momentum must be drained too (else its velocity blows up as the
/// density is evacuated). dimension-generic over 1..=3.
pub fn penalize_drain_gv(
    coords: Coords,
    ndim: usize,
    dof: usize,
    axes: &[usize],
    has_dye: bool,
) -> KernelProgram {
    penalize_drain_impl::<Adiabatic>(coords, ndim, dof, axes, has_dye)
}

/// the isothermal twin: the drain scales den + mom and the sound speed is the
/// constant `cs` param, so the delta outputs are mass + force. same [Drain]
/// stack, same integrator — the iso energy slot is zero-sized, so the e-channel
/// drops out by construction.
pub fn penalize_drain_iso_gv(
    coords: Coords,
    ndim: usize,
    dof: usize,
    axes: &[usize],
    has_dye: bool,
) -> KernelProgram {
    penalize_drain_impl::<IsoModel>(coords, ndim, dof, axes, has_dye)
}

/// the second-order material drain for the magnetic-slip path (3D cartesian, adiabatic). the drain
/// rate `lambda = max(sqrt(cs^2 + |B|^2/rho) inv_cd_dx, sqrt(GM/r_acc^3))` stiffens as the density
/// drains at fixed field, so evaluating it at the entering state is first-order in time. this kernel
/// takes the local Alfven speed from the cell's own `bcell` (not the domain-wide max) and integrates
/// the masked step with `event_split_drain_factor`: exact free-fall to the acoustic/free-fall crossing
/// density, then a midpoint on the acoustic remainder, so the drain stays second-order accurate even
/// for a cell that crosses branches mid-step. the resulting factor `f` scales den/mom/energy (and the
/// dissolved dye) uniformly; `B` is untouched. `penalize_cell` applies `exp(-chi dt inv_tau)`, so the
/// effective rate reproducing `f` is `-ln(f)/(chi dt)`, with a guard leaving `chi = 0` an exact no-op.
pub fn penalize_drain_midpoint_gv(coords: Coords, has_dye: bool) -> KernelProgram {
    let ndim = 3usize;
    let dof = 3usize;
    let axes: &[usize] = &[0, 1, 2];
    assert!(
        coords == Coords::Cartesian,
        "the midpoint material drain is a 3D cartesian slip-path kernel; the {coords:?} chart uses the \
         global-c_a2 penalize_drain"
    );
    let (kernel, writes) = trace(|cx| {
        let dt = cx.scalar("dt");
        let gamma = cx.scalar("gamma");
        let c_drain = cx.scalar("c_drain");

        let den = cx.field("den", symbi_ir::FieldRef::cons_den());
        let mom: Vec<Gv> = (0..dof)
            .map(|c| cx.field(&format!("mom_{c}"), symbi_ir::FieldRef::cons_mom(c as u8)))
            .collect();
        let nrg: <Adiabatic as EnergyModel>::Slot<Gv> =
            <<Adiabatic as EnergyModel>::Slot<Gv> as EnergySlot<Gv>>::from_scalar(
                cx.field("nrg", symbi_ir::FieldRef::cons_nrg()),
            );
        // the cell-centered field: the local Alfven speed c_a^2 = |B|^2/rho, the density-dependent
        // stiffness the midpoint tracks.
        let b_q: Vec<Gv> = (0..3)
            .map(|c| cx.field(&format!("bc_{c}"), symbi_ir::FieldRef::BCell(c as u8)))
            .collect();
        let b_sq = b_q.iter().fold(Gv::ZERO, |s, b| s + *b * *b);

        let geo = cell_geometry_gv(cx, coords, &vec![Spacing::Uniform; ndim], &axes[..ndim], ndim);
        let dv = Gv::ONE / geo.inv_volume;
        let mut min_w = gv_axis_width(cx, 0, Spacing::Uniform);
        for ax in 1..ndim {
            min_w = min_w.min(gv_axis_width(cx, ax, Spacing::Uniform));
        }

        let center: [Gv; 3] = std::array::from_fn(|a| cx.scalar(&format!("body_0_pos_{a}")));
        let x_cart = centroid_to_cartesian(coords, ndim, axes, &geo.centroid);
        let x: [Gv; 3] = std::array::from_fn(|a| x_cart[a]);
        let phi = body_mask_sdf(cx, center).dist(x);
        let chi = symbi_ib::sdf::chi(phi, min_w);
        tag_body_mask(cx, &chi, coords, ndim, axes, None, false);

        // the sound speed is invariant under the uniform drain (e_int is preserved), so the drain rate
        // shifts only through the local Alfven term |B|^2/rho.
        let mom_sq = mom.iter().fold(Gv::ZERO, |sum, m| sum + *m * *m);
        let cs = symbi_ib::drain::sound_speed_from_cons(den, mom_sq, nrg.value(), gamma);
        let mass = cx.scalar("body_0_mass");
        let r_acc = cx.scalar("body_0_racc");
        let inv_cd_dx = Gv::ONE / (c_drain * min_w);
        let lambda_ff = (mass / (r_acc * r_acc * r_acc)).sqrt();

        // the event-split second-order factor: exact free-fall to the acoustic/free-fall crossing, then
        // midpoint on the acoustic remainder. penalize_cell applies exp(-chi dt inv_tau), so the
        // effective rate reproducing the factor f is -ln(f)/(chi dt); the guard keeps chi = 0 an exact
        // no-op (f = 1, inv_tau = 0).
        let f = symbi_ib::drain::event_split_drain_factor(den, chi, dt, cs, b_sq, inv_cd_dx, lambda_ff);
        let tiny = Gv::from_f64(1e-300);
        let inv_tau_eff = (Gv::ZERO - f.max(tiny).ln()) / (chi * dt).max(tiny);

        let kin = BodyKin::<Gv, 3> {
            u_solid: Tensor::zeros(),
            omega: Tensor::zeros(),
            e_wall: Gv::ZERO,
        };
        let mut acc = PenalizationRelaxation::<Gv, 3>::none();
        Property::Drain { inv_tau: inv_tau_eff }.contribute(chi, &kin, &mut acc);
        let chi_val = if has_dye {
            cx.field("chi", symbi_ir::FieldRef::cons_chi())
        } else {
            Gv::ZERO
        };
        let mut slots: Vec<Gv> = vec![den];
        slots.extend((0..3).map(|a| if a < dof { mom[a] } else { Gv::ZERO }));
        slots.push(nrg.value());
        let cons = ConsG::<Gv, 3, Adiabatic, Dyed>::from_slots(&slots).with_chi(chi_val);
        let (out, delta) = penalize_cell(&cons, &acc, Tensor::zeros(), dt, dv, 0);
        let n_cart: Vec<Gv> = vec![Gv::ZERO; ndim];
        let (f_cart, torque) = cartesian_receipt(
            coords,
            ndim,
            axes,
            &geo.centroid,
            &x,
            &center,
            &delta.force_delta,
        );
        penalize_writes::<Adiabatic>(ndim, dof, has_dye, &out, &delta, &f_cart, &torque, &n_cart)
    });
    let kernel = kernel.with_derived_support(&writes);
    KernelProgram::new(kernel, writes)
}

/// the [Drain] stack traced once, generic over the energy model: the adiabatic and
/// isothermal kernels are the same graph up to the energy channel, so one builder
/// carries both. `E::HAS_ENERGY` gates the `nrg` field read/write, the sound-speed
/// recovery, and the `pen_0_energy` receipt; the iso energy slot is zero-sized, so
/// the iso trace carries den + mom alone.
fn penalize_drain_impl<E: EnergyModel>(
    coords: Coords,
    ndim: usize,
    dof: usize,
    axes: &[usize],
    has_dye: bool,
) -> KernelProgram {
    assert!(
        (1..=3).contains(&ndim) && (ndim..=3).contains(&dof),
        "penalize_drain_gv: need 1<=ndim<=dof<=3"
    );
    let (kernel, writes) = trace(|cx| {
        let dt = cx.scalar("dt");
        let eos = cx.scalar(if E::HAS_ENERGY { "gamma" } else { "cs" });
        let c_drain = cx.scalar("c_drain");

        // conserved reads (in-place: the same fields are the writes). the momentum runs over dof (all
        // active components), so a 2.5D MHD sink drains the out-of-plane momentum and its kinetic energy.
        // the energy field exists only in the adiabatic regime; the iso slot is zero-sized.
        let den = cx.field("den", symbi_ir::FieldRef::cons_den());
        let mom: Vec<Gv> = (0..dof)
            .map(|c| cx.field(&format!("mom_{c}"), symbi_ir::FieldRef::cons_mom(c as u8)))
            .collect();
        let nrg: E::Slot<Gv> = if E::HAS_ENERGY {
            <E::Slot<Gv> as EnergySlot<Gv>>::from_scalar(
                cx.field("nrg", symbi_ir::FieldRef::cons_nrg()),
            )
        } else {
            <E::Slot<Gv> as EnergySlot<Gv>>::zero()
        };

        // the cell centroid + volume from the shared geometry scaffold — the coordinate
        // map every other body kernel of this chart evaluates.
        let geo = cell_geometry_gv(
            cx,
            coords,
            &vec![Spacing::Uniform; ndim],
            &axes[..ndim],
            ndim,
        );
        let dv = Gv::ONE / geo.inv_volume;
        let mut min_w = gv_axis_width(cx, 0, Spacing::Uniform);
        for ax in 1..ndim {
            min_w = min_w.min(gv_axis_width(cx, ax, Spacing::Uniform));
        }

        // the mask geometry as a traced SDF: phi = |x - body_pos| - r_mask, and
        // chi = 0.5 (1 - tanh(phi / w)). the distance is physical, so the coordinate
        // centroid is mapped to cartesian first (identity on a cartesian grid).
        let center: [Gv; 3] = std::array::from_fn(|a| {
            if a < ndim {
                cx.scalar(&format!("body_0_pos_{a}"))
            } else {
                Gv::ZERO
            }
        });
        let x_cart = centroid_to_cartesian(coords, ndim, axes, &geo.centroid);
        let x: [Gv; 3] = std::array::from_fn(|a| if a < ndim { x_cart[a] } else { Gv::ZERO });
        let sphere = body_mask_sdf(cx, center);
        let phi = sphere.dist(x);
        let chi = symbi_ib::sdf::chi(phi, min_w);
        tag_body_mask(cx, &chi, coords, ndim, axes, None, false);

        let cs = if E::HAS_ENERGY {
            let mom_sq = mom.iter().fold(Gv::ZERO, |sum, m| sum + *m * *m);
            symbi_ib::drain::sound_speed_from_cons(den, mom_sq, nrg.value(), eos)
        } else {
            eos
        };
        let sound_rate = signal_speed(cx, cs) / (c_drain * min_w);
        let inv_tau = symbi_ib::drain::spherical_drain_rate(
            sound_rate,
            cx.scalar("body_0_mass"),
            cx.scalar("body_0_racc"),
        );

        // the property stack: [Drain]. contribute at Gv, then
        // the same integrator that runs at f64.
        let kin = BodyKin::<Gv, 3> {
            u_solid: Tensor::zeros(),
            omega: Tensor::zeros(),
            e_wall: Gv::ZERO,
        };
        let mut acc = PenalizationRelaxation::<Gv, 3>::none();
        Property::Drain { inv_tau }.contribute(chi, &kin, &mut acc);
        // an undyed kernel traces a constant-zero dye: `penalize_cell` still scales it by the
        // drain factor, the result feeds no write, and the tracer eliminates the dead arithmetic.
        let chi_val = if has_dye {
            cx.field("chi", symbi_ir::FieldRef::cons_chi())
        } else {
            Gv::ZERO
        };
        let mut slots: Vec<Gv> = vec![den];
        slots.extend((0..3).map(|a| if a < dof { mom[a] } else { Gv::ZERO }));
        slots.push(nrg.value());
        let cons = ConsG::<Gv, 3, E, Dyed>::from_slots(&slots).with_chi(chi_val);
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

        penalize_writes::<E>(ndim, dof, has_dye, &out, &delta, &f_cart, &torque, &n_cart)
    });

    // the delta outputs vanish exactly beyond the tanh saturation radius; the
    // in-place cons writes are unchanged-value there, so the ball bounds
    // everything the reduction needs (dispatch may clip to it).
    let kernel = kernel.with_derived_support(&writes);
    KernelProgram::new(kernel, writes)
}

/// the shared write list every penalization surface emits: the in-place conserved
/// updates (den, mom over dof, the adiabatic energy, the optional dye) followed by
/// the body receipts (mass, cartesian force, the adiabatic energy receipt, torque,
/// form drag). the momentum block runs over DOF, so a 2.5D MHD kernel (dof = 3,
/// ndim = 2) writes the drained out-of-plane momentum back — leaving it unwritten
/// while the density drains would grow its velocity as 1/rho inside the mask.
#[allow(clippy::too_many_arguments)]
fn penalize_writes<'t, E: EnergyModel>(
    ndim: usize,
    dof: usize,
    has_dye: bool,
    out: &ConsG<Gv<'t>, 3, E, Dyed>,
    delta: &symbi_ib::BodyDelta<Gv<'t>, 3>,
    f_cart: &Tensor<Gv<'t>, 3>,
    torque: &Tensor<Gv<'t>, 3>,
    n_cart: &[Gv<'t>],
) -> KernelWrites {
    let mut writes: KernelWrites = Vec::new();
    writes.push(KernelWrite::new(
        "den_out",
        symbi_ir::FieldRef::cons_den(),
        out.den().node(),
    ));
    for a in 0..dof {
        writes.push(KernelWrite::new(
            format!("mom_out_{a}"),
            symbi_ir::FieldRef::cons_mom(a as u8),
            out.mom()[a].node(),
        ));
    }
    if E::HAS_ENERGY {
        writes.push(KernelWrite::new(
            "nrg_out",
            symbi_ir::FieldRef::cons_nrg(),
            out.nrg().value().node(),
        ));
    }
    // the sink swallows gas and the dye dissolved in it together; `penalize_cell` applied the
    // same drain factor to both, so the concentration of the surviving gas is unchanged. the
    // iso regime has no energy write, so there the dye rides directly after the momentum block.
    if has_dye {
        writes.push(KernelWrite::new(
            "chi_out",
            symbi_ir::FieldRef::cons_chi(),
            out.chi().node(),
        ));
    }
    writes.push(KernelWrite::new(
        "pen_mass",
        "pen_0_mass",
        delta.mass_delta.node(),
    ));
    for a in 0..ndim {
        writes.push(KernelWrite::new(
            format!("pen_force_{a}"),
            format!("pen_0_force_{a}"),
            f_cart[a].node(),
        ));
    }
    if E::HAS_ENERGY {
        writes.push(KernelWrite::new(
            "pen_energy",
            "pen_0_energy",
            delta.energy_delta.node(),
        ));
    }
    for a in torque_axes(ndim) {
        writes.push(KernelWrite::new(
            format!("pen_torque_{a}"),
            format!("pen_0_torque_{a}"),
            torque[a].node(),
        ));
    }
    push_force_normal(&mut writes, ndim, f_cart, n_cart);
    writes
}

/// trace the [PorousAccretor]-stack penalization for the adiabatic regime,
/// cartesian, DOF = ndim. the porosity dial `p` scales the drain channel and
/// `(1 - p)` the wall channels; the wall rates are `k_eta_n/t * c_s / dx`
/// (sound-crossings per cell width — multiplicative dials so zero is an exact
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
) -> KernelProgram {
    penalize_porous_impl::<Adiabatic>(coords, ndim, dof, None, false, axes, has_dye)
}

/// the arbitrary-shape porous wall: the same relaxation stack as `penalize_porous_gv`, but the
/// mask and the surface normal come from the config CSG `shape` (body-local, translated to the
/// runtime body position). built at sim setup, once the body's shape is
/// known — a per-body CSG is a runtime fact, so it reaches the kernel after the AOT bake.
pub fn penalize_porous_gv_shaped(
    coords: Coords,
    ndim: usize,
    dof: usize,
    shape: &SdfExpr<f64, 3>,
    has_dye: bool,
) -> KernelProgram {
    penalize_porous_impl::<Adiabatic>(
        coords,
        ndim,
        dof,
        Some(shape),
        false,
        &[0, 1, 2][..ndim],
        has_dye,
    )
}

/// the spinning arbitrary-shape porous wall: like `penalize_porous_gv_shaped`, but the mask is
/// rotated by the runtime orientation matrix (the 9 scalars `body_0_rot_0..8`) and the surface
/// velocity carries the spin `omega x r` (the vector `body_0_omega_0..2`), so the wall drags the
/// gas around as it turns — an arbitrary spin axis.
pub fn penalize_porous_gv_spinning(
    coords: Coords,
    ndim: usize,
    dof: usize,
    shape: &SdfExpr<f64, 3>,
    has_dye: bool,
) -> KernelProgram {
    penalize_porous_impl::<Adiabatic>(
        coords,
        ndim,
        dof,
        Some(shape),
        true,
        &[0, 1, 2][..ndim],
        has_dye,
    )
}

/// the isothermal [PorousAccretor] twin: the same porous surface as
/// `penalize_porous_gv` for the isothermal regime — the sound speed is the
/// constant `cs` param and the delta outputs are mass + force.
/// `p = 1` reduces bit-for-bit to `penalize_drain_iso`.
pub fn penalize_porous_iso_gv(
    coords: Coords,
    ndim: usize,
    dof: usize,
    axes: &[usize],
    has_dye: bool,
) -> KernelProgram {
    penalize_porous_impl::<IsoModel>(coords, ndim, dof, None, false, axes, has_dye)
}

/// the arbitrary-shape iso porous wall: the energy-free counterpart of
/// `penalize_porous_gv_shaped` — mask + normal from the config CSG, den + mom channels.
pub fn penalize_porous_iso_gv_shaped(
    coords: Coords,
    ndim: usize,
    dof: usize,
    shape: &SdfExpr<f64, 3>,
    has_dye: bool,
) -> KernelProgram {
    penalize_porous_impl::<IsoModel>(
        coords,
        ndim,
        dof,
        Some(shape),
        false,
        &[0, 1, 2][..ndim],
        has_dye,
    )
}

/// the spinning iso porous wall: the energy-free counterpart of `penalize_porous_gv_spinning`.
pub fn penalize_porous_iso_gv_spinning(
    coords: Coords,
    ndim: usize,
    dof: usize,
    shape: &SdfExpr<f64, 3>,
    has_dye: bool,
) -> KernelProgram {
    penalize_porous_impl::<IsoModel>(
        coords,
        ndim,
        dof,
        Some(shape),
        true,
        &[0, 1, 2][..ndim],
        has_dye,
    )
}

/// the porous surface traced once, generic over the energy model (see
/// `penalize_drain_impl` for the gating rule).
fn penalize_porous_impl<E: EnergyModel>(
    coords: Coords,
    ndim: usize,
    dof: usize,
    shape: Option<&SdfExpr<f64, 3>>,
    spin: bool,
    axes: &[usize],
    has_dye: bool,
) -> KernelProgram {
    assert!(
        (1..=3).contains(&ndim) && (ndim..=3).contains(&dof),
        "penalize_porous_gv: need 1<=ndim<=dof<=3"
    );
    let (kernel, writes) = trace(|cx| {
        let dt = cx.scalar("dt");
        // the regime's eos handle: adiabatic reads `gamma` (the sound speed is recovered from the
        // conserved state below); isothermal reads the constant sound speed `cs` directly.
        let eos = cx.scalar(if E::HAS_ENERGY { "gamma" } else { "cs" });
        let c_drain = cx.scalar("c_drain");
        let porosity = cx.scalar("porosity");
        let k_eta_n = cx.scalar("k_eta_n");
        let k_eta_t = cx.scalar("k_eta_t");

        // the wall normal + velocity target are ndim (in-plane); the momentum runs over dof so the
        // out-of-plane component is carried as a purely tangential velocity (relaxed by k_eta_t) and its
        // kinetic energy enters c_s. it exists in 2.5D MHD, where the drain and the wall both act on
        // it; hydro runs at dof == ndim.
        let den = cx.field("den", symbi_ir::FieldRef::cons_den());
        let mom: Vec<Gv> = (0..dof)
            .map(|c| cx.field(&format!("mom_{c}"), symbi_ir::FieldRef::cons_mom(c as u8)))
            .collect();
        let nrg: E::Slot<Gv> = if E::HAS_ENERGY {
            <E::Slot<Gv> as EnergySlot<Gv>>::from_scalar(
                cx.field("nrg", symbi_ir::FieldRef::cons_nrg()),
            )
        } else {
            <E::Slot<Gv> as EnergySlot<Gv>>::zero()
        };

        let geo = cell_geometry_gv(
            cx,
            coords,
            &vec![Spacing::Uniform; ndim],
            &axes[..ndim],
            ndim,
        );
        let dv = Gv::ONE / geo.inv_volume;
        let mut min_w = gv_axis_width(cx, 0, Spacing::Uniform);
        for ax in 1..ndim {
            min_w = min_w.min(gv_axis_width(cx, ax, Spacing::Uniform));
        }

        let center: [Gv; 3] = std::array::from_fn(|a| {
            if a < ndim {
                cx.scalar(&format!("body_0_pos_{a}"))
            } else {
                Gv::ZERO
            }
        });
        // the mask distance is physical: map the coordinate centroid to cartesian.
        let x_cart = centroid_to_cartesian(coords, ndim, axes, &geo.centroid);
        let x: [Gv; 3] = std::array::from_fn(|a| if a < ndim { x_cart[a] } else { Gv::ZERO });
        let sdf = if spin {
            body_mask_sdf_spinning(
                cx,
                center,
                shape.expect("a spinning wall must have a shape"),
            )
        } else {
            body_mask_sdf_shaped(cx, center, shape)
        };
        let phi = sdf.dist(x);
        let chi = symbi_ib::sdf::chi(phi, min_w);
        tag_body_mask(cx, &chi, coords, ndim, axes, shape, spin);

        // tau = c_drain dx / c_s. the adiabatic c_s comes from the just-updated conserved
        // state (post-godunov, pre-c2p); the isothermal c_s is the constant scalar.
        let cs = if E::HAS_ENERGY {
            let mut mom_sq = Gv::ZERO;
            for m in &mom {
                mom_sq = mom_sq + *m * *m;
            }
            symbi_ib::drain::sound_speed_from_cons(den, mom_sq, nrg.value(), eos)
        } else {
            eos
        };
        let sound_rate = signal_speed(cx, cs) / (c_drain * min_w);
        let inv_tau = if shape.is_none() {
            symbi_ib::drain::spherical_drain_rate(
                sound_rate,
                cx.scalar("body_0_mass"),
                cx.scalar("body_0_racc"),
            )
        } else {
            sound_rate
        };
        let rate_scale = signal_speed(cx, cs) / min_w;

        // the outward surface normal in the cell's physical frame (the cartesian normal rotated into
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
                cx.scalar(&format!("body_0_vel_{a}"))
            } else {
                Gv::ZERO
            }
        }));
        // a spinning wall's surface moves at u_solid + omega x r with the full world-frame
        // angular-velocity vector (an arbitrary rotation axis; z-spin is the [0, 0, w] case).
        let omega = if spin {
            // the full angular-velocity vector (world frame): the surface drags the gas at omega x r
            // about the (evolving) rotation axis.
            Tensor::<Gv, 3>::new([
                cx.scalar("body_0_omega_0"),
                cx.scalar("body_0_omega_1"),
                cx.scalar("body_0_omega_2"),
            ])
        } else {
            Tensor::zeros()
        };
        let base_kin = BodyKin::<Gv, 3> {
            u_solid,
            omega,
            e_wall: Gv::ZERO,
        };
        // a spinning wall's velocity target is the local rigid-motion velocity u_solid + omega x r_cell;
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
        let mut acc = PenalizationRelaxation::<Gv, 3>::none();
        Property::PorousAccretor {
            p: porosity,
            inv_tau,
            inv_eta_n: k_eta_n * rate_scale,
            inv_eta_t: k_eta_t * rate_scale,
        }
        .contribute(chi, &kin, &mut acc);
        // an undyed kernel traces a constant-zero dye: `penalize_cell` still scales it by the
        // drain factor, the result feeds no write, and the tracer eliminates the dead arithmetic.
        let chi_val = if has_dye {
            cx.field("chi", symbi_ir::FieldRef::cons_chi())
        } else {
            Gv::ZERO
        };
        let mut slots: Vec<Gv> = vec![den];
        slots.extend((0..3).map(|a| if a < dof { mom[a] } else { Gv::ZERO }));
        slots.push(nrg.value());
        let cons = ConsG::<Gv, 3, E, Dyed>::from_slots(&slots).with_chi(chi_val);
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

        penalize_writes::<E>(ndim, dof, has_dye, &out, &delta, &f_cart, &torque, &n_cart)
    });

    let kernel = kernel.with_derived_support(&writes);
    KernelProgram::new(kernel, writes)
}

/// the adiabatic torque-free accretor: the drain plus a tangential
/// anti-relaxation `lambda_t = -xi lambda_rho` about the sphere normal, so the
/// accreted mass delivers zero net angular momentum to the body (the dittmann &
/// ryan 2021 torque-free sink, coordinate-free via the SDF normal). the
/// retention floor (`PenalizationRelaxation.ut_growth_cap`, set by the property) bounds the
/// growing tangential factor at the evacuation limit. the sound speed is
/// recovered from the conserved state and the energy channel is carried (delta
/// outputs mass + force + energy + torque). `xi = 0` reduces to
/// `penalize_drain`; `xi = 1` is torque-free. the velocity target is the
/// body's translational velocity `body_0_vel_*` (the accretion is torque-free
/// relative to the moving sink).
pub fn penalize_torque_free_gv(
    coords: Coords,
    ndim: usize,
    dof: usize,
    axes: &[usize],
    has_dye: bool,
) -> KernelProgram {
    penalize_torque_free_impl::<Adiabatic>(coords, ndim, dof, axes, has_dye)
}

/// the isothermal torque-free twin: the same torque-free surface as
/// `penalize_torque_free_gv` on the den + mom channels (iso, the thin-disk
/// regime) — the sound speed is the constant `cs` param, delta outputs
/// mass + force + torque. `xi = 0` reduces bit-for-bit to `penalize_drain_iso`.
pub fn penalize_torque_free_iso_gv(
    coords: Coords,
    ndim: usize,
    dof: usize,
    axes: &[usize],
    has_dye: bool,
) -> KernelProgram {
    penalize_torque_free_impl::<IsoModel>(coords, ndim, dof, axes, has_dye)
}

/// the torque-free surface traced once, generic over the energy model (see
/// `penalize_drain_impl` for the gating rule).
fn penalize_torque_free_impl<E: EnergyModel>(
    coords: Coords,
    ndim: usize,
    dof: usize,
    axes: &[usize],
    has_dye: bool,
) -> KernelProgram {
    assert!(
        (1..=3).contains(&ndim) && (ndim..=3).contains(&dof),
        "penalize_torque_free_gv: need 1<=ndim<=dof<=3"
    );
    let (kernel, writes) = trace(|cx| {
        let dt = cx.scalar("dt");
        let eos = cx.scalar(if E::HAS_ENERGY { "gamma" } else { "cs" });
        let c_drain = cx.scalar("c_drain");
        let xi = cx.scalar("xi");

        let den = cx.field("den", symbi_ir::FieldRef::cons_den());
        let mom: Vec<Gv> = (0..dof)
            .map(|c| cx.field(&format!("mom_{c}"), symbi_ir::FieldRef::cons_mom(c as u8)))
            .collect();
        let nrg: E::Slot<Gv> = if E::HAS_ENERGY {
            <E::Slot<Gv> as EnergySlot<Gv>>::from_scalar(
                cx.field("nrg", symbi_ir::FieldRef::cons_nrg()),
            )
        } else {
            <E::Slot<Gv> as EnergySlot<Gv>>::zero()
        };

        let geo = cell_geometry_gv(
            cx,
            coords,
            &vec![Spacing::Uniform; ndim],
            &axes[..ndim],
            ndim,
        );
        let dv = Gv::ONE / geo.inv_volume;
        let mut min_w = gv_axis_width(cx, 0, Spacing::Uniform);
        for ax in 1..ndim {
            min_w = min_w.min(gv_axis_width(cx, ax, Spacing::Uniform));
        }

        let center: [Gv; 3] = std::array::from_fn(|a| {
            if a < ndim {
                cx.scalar(&format!("body_0_pos_{a}"))
            } else {
                Gv::ZERO
            }
        });
        // the mask distance is physical: map the coordinate centroid to cartesian.
        let x_cart = centroid_to_cartesian(coords, ndim, axes, &geo.centroid);
        let x: [Gv; 3] = std::array::from_fn(|a| if a < ndim { x_cart[a] } else { Gv::ZERO });
        let sphere = body_mask_sdf(cx, center);
        let chi = symbi_ib::sdf::chi(sphere.dist(x), min_w);
        tag_body_mask(cx, &chi, coords, ndim, axes, None, false);

        let cs = if E::HAS_ENERGY {
            let mom_sq = mom.iter().fold(Gv::ZERO, |sum, m| sum + *m * *m);
            symbi_ib::drain::sound_speed_from_cons(den, mom_sq, nrg.value(), eos)
        } else {
            eos
        };
        let sound_rate = signal_speed(cx, cs) / (c_drain * min_w);
        let inv_tau = symbi_ib::drain::spherical_drain_rate(
            sound_rate,
            cx.scalar("body_0_mass"),
            cx.scalar("body_0_racc"),
        );

        // the outward surface normal in the cell's physical frame: the cartesian
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
                cx.scalar(&format!("body_0_vel_{a}"))
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
        let mut acc = PenalizationRelaxation::<Gv, 3>::none();
        Property::TorqueFreeAccretor { inv_tau, xi }.contribute(chi, &kin, &mut acc);
        // an undyed kernel traces a constant-zero dye: `penalize_cell` still scales it by the
        // drain factor, the result feeds no write, and the tracer eliminates the dead arithmetic.
        let chi_val = if has_dye {
            cx.field("chi", symbi_ir::FieldRef::cons_chi())
        } else {
            Gv::ZERO
        };
        let mut slots: Vec<Gv> = vec![den];
        slots.extend((0..3).map(|a| if a < dof { mom[a] } else { Gv::ZERO }));
        slots.push(nrg.value());
        let cons = ConsG::<Gv, 3, E, Dyed>::from_slots(&slots).with_chi(chi_val);
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

        penalize_writes::<E>(ndim, dof, has_dye, &out, &delta, &f_cart, &torque, &n_cart)
    });

    let kernel = kernel.with_derived_support(&writes);
    KernelProgram::new(kernel, writes)
}

#[cfg(test)]
mod shaped_tests {
    use super::*;
    use symbi_ib::sdf::SdfExpr;

    // a shaped porous kernel bakes the CSG geometry as constants and keeps only the body
    // position as a runtime scalar — so a moving body rides the same kernel, updating just
    // `body_0_pos_*` per step. the sphere kernel instead reads a runtime `body_0_racc`.
    #[test]
    fn shaped_porous_kernel_bakes_geometry_and_keeps_position_runtime() {
        let shape = SdfExpr::<f64, 3>::cuboid([0.0, 0.0, 0.0], [0.5, 0.3, 0.2])
            .union(SdfExpr::sphere([0.6, 0.0, 0.0], 0.25));
        let program =
            penalize_porous_gv_shaped(Coords::Cartesian, 3, 3, &shape, false);
        let kernel = program.kernel();
        let writes = program.writes();
        assert!(
            !kernel.graph().has_errors(),
            "shaped porous kernel traced with graph errors"
        );
        assert!(
            kernel.scalar_params().iter().any(|p| p == "body_0_pos_0"),
            "the runtime body position must remain a scalar (a moving body updates it)",
        );
        assert!(
            !kernel.scalar_params().iter().any(|p| p == "body_0_racc"),
            "shaped kernel must bake geometry, not read the sphere radius: {:?}",
            kernel.scalar_params(),
        );
        for p in ["porosity", "k_eta_n", "k_eta_t", "body_0_vel_0", "dt"] {
            assert!(
                kernel.scalar_params().iter().any(|s| s == p),
                "missing runtime param {p}"
            );
        }
        assert!(!writes.is_empty(), "the shaped kernel emitted no writes");
    }

    // the unshaped path is the sphere: `body_0_racc` is a runtime scalar (the AOT kernel).
    #[test]
    fn unshaped_porous_kernel_reads_the_runtime_radius() {
        let program =
            penalize_porous_gv(Coords::Cartesian, 3, 3, &[0, 1, 2], false);
        let kernel = program.kernel();
        assert!(kernel.scalar_params().iter().any(|p| p == "body_0_racc"));
    }
}

#[cfg(test)]
mod twin_tests {
    use super::*;

    type Builder = fn(Coords, usize, usize, &[usize], bool) -> KernelProgram;

    const PAIRS: [(&str, Builder, Builder); 3] = [
        ("drain", penalize_drain_gv, penalize_drain_iso_gv),
        ("porous", penalize_porous_gv, penalize_porous_iso_gv),
        (
            "torque_free",
            penalize_torque_free_gv,
            penalize_torque_free_iso_gv,
        ),
    ];

    // each iso surface differs from its adiabatic twin only by the energy channel
    // and by the lawful source of sound speed (`gamma` recovery versus fixed `cs`).
    #[test]
    fn iso_twin_is_the_adiabatic_surface_minus_the_energy_channel() {
        for (name, adiabatic, iso) in PAIRS {
            let program_a = adiabatic(Coords::Cartesian, 2, 2, &[0, 1], true);
            let ka = program_a.kernel();
            let wa = program_a.writes();
            let program_i = iso(Coords::Cartesian, 2, 2, &[0, 1], true);
            let ki = program_i.kernel();
            let wi = program_i.writes();
            let names =
                |w: &KernelWrites| w.iter().map(|write| write.key.clone()).collect::<Vec<_>>();
            let expected: Vec<symbi_ir::OutputKey> = names(&wa)
                .into_iter()
                .filter(|n| n != "nrg_out" && n != "pen_energy")
                .collect();
            assert_eq!(
                names(&wi),
                expected,
                "{name}: the iso write list must be the adiabatic list minus the energy channel"
            );
            assert!(
                ka.scalar_params().iter().any(|p| p == "gamma"),
                "{name}: every adiabatic spherical sink recovers its local sound speed from gamma"
            );
            assert!(
                ki.scalar_params().iter().any(|p| p == "cs"),
                "{name}: every isothermal spherical sink reads its prescribed sound speed"
            );
            assert!(
                !ki.scalar_params().iter().any(|p| p == "gamma"),
                "{name}: the iso kernel must not read gamma"
            );
        }
    }

    // every 2.5D bake (grid ndim = 2, momentum dof = 3) must write the out-of-plane
    // momentum back: the sink drains the density while the momentum rides along, so
    // an unwritten mom_out_2 leaves the out-of-plane velocity growing as 1/rho
    // inside the mask.
    #[test]
    fn dof3_kernels_write_the_out_of_plane_momentum() {
        for (name, adiabatic, iso) in PAIRS {
            for (regime, build) in [("adiabatic", adiabatic), ("iso", iso)] {
                let program = build(Coords::Cartesian, 2, 3, &[0, 1], false);
                let writes = program.writes();
                assert!(
                    writes.iter().any(|write| write.key == "mom_out_2"),
                    "{name} ({regime}): the dof=3 kernel must write mom_out_2"
                );
            }
        }
    }

    // the momentum count sits at or above the grid dimension: the cons momentum
    // fill and the write loop both index 0..dof, so dof < ndim would silently zero
    // in-plane momentum.
    #[test]
    #[should_panic(expected = "1<=ndim<=dof<=3")]
    fn iso_drain_rejects_dof_below_ndim() {
        let _ = penalize_drain_iso_gv(Coords::Cartesian, 2, 1, &[0, 1], false);
    }
}
