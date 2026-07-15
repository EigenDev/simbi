// =============================================================================
// gv_penalize.rs
//
// the traced immersed-boundary penalization kernel (docs/design/50 layer 3):
// per cell, the sphere SDF's signed distance -> the mollified chi -> the
// property stack's Relax accumulation -> the SAME carrier-generic
// `penalize_cell` the f64 oracle runs, evaluated at Gv. the [Drain] stack is
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
use symbi_hydro::energy::Adiabatic;
use symbi_hydro::state::ConsG;
use symbi_ib::penalize::{penalize_cell, BodyKin, Property, Relax};
use symbi_ib::sdf::SdfExpr;
use symbi_ir::gv::Writes;
use symbi_ir::{Gv, GvKernel, ParamExpr, Support};

/// the wall/drain relaxation signal speed: the FAST MAGNETOSONIC speed `sqrt(c_s^2 + c_a^2)`,
/// with `c_a^2 = |B|^2 / rho` bound as the runtime `c_a2` scalar (the max over the interior, so the
/// wall stays a signal-crossing stiff in the low-beta regions a magnetized sink accumulates).
/// `c_a2 = 0` off MHD reduces it to the sound speed exactly, so a hydro run is unchanged.
fn signal_speed(cs: Gv) -> Gv {
    (cs * cs + Gv::scalar("c_a2")).sqrt()
}

use crate::coords::{Coords, Spacing};
use crate::gv::cell_geometry_gv;
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

/// map the coordinate cell centroid `(r, theta, phi)` / `(R, phi, z)` to Cartesian
/// so the sphere SDF measures the PHYSICAL distance to the body. a coordinate-space
/// subtraction is meaningless on a curved grid (`sqrt((r - r_b)^2 + (theta -
/// theta_b)^2)` is not a distance); on a Cartesian grid this is the identity. the
/// 2D cylindrical chart is the `(R, phi)` disk plane (`CylindricalRPhi`), matching
/// the geometric-source dispatch — one metric for the whole codebase.
fn centroid_to_cartesian(coords: Coords, ndim: usize, centroid: &[Gv]) -> Vec<Gv> {
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
/// Cartesian direction; the wall / torque-free split projects the physical
/// momentum onto it, so it must be rotated here. `x` is the COORDINATE centroid
/// (the rotation depends on the local basis). identity on a Cartesian grid.
fn vector_from_cartesian(coords: Coords, ndim: usize, x: &[Gv], v_cart: &[Gv]) -> Vec<Gv> {
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

/// rotate a PHYSICAL-frame vector into the global Cartesian frame — used to book
/// the lab-frame torque `r_cart x F_cart` (the accreted force receipt is in the
/// physical frame; the cross product needs both vectors in one frame). identity
/// on a Cartesian grid, so the Cartesian torque is bit-unchanged.
fn vector_to_cartesian(coords: Coords, ndim: usize, x: &[Gv], v_phys: &[Gv]) -> Vec<Gv> {
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

/// the lab-frame torque receipt `r_cart x F_cart` for a body at `center` (Cartesian):
/// rotate the physical-frame `force` to Cartesian, cross with the Cartesian
/// displacement. returns the 3-moment (2d books only z). `x_cart` is the cell's
/// Cartesian position, `centroid` its coordinate position (drives the rotation).
fn lab_torque(
    coords: Coords,
    ndim: usize,
    centroid: &[Gv],
    x_cart: &[Gv; 3],
    center: &[Gv; 3],
    force: &Tensor<Gv, 3>,
) -> Tensor<Gv, 3> {
    let f_cart = vector_to_cartesian(
        coords,
        ndim,
        centroid,
        &(0..ndim).map(|a| force[a]).collect::<Vec<_>>(),
    );
    let f = Tensor::<Gv, 3>::new(std::array::from_fn(|a| {
        if a < ndim { f_cart[a] } else { Gv::ZERO }
    }));
    let x_rel = Tensor::<Gv, 3>::new(std::array::from_fn(|a| x_cart[a] - center[a]));
    symbi_ib::moment(&x_rel, &f)
}

/// the immersed body's mask geometry as a traced SDF, centered at `center` (the body
/// position `body_0_pos_*` in Cartesian): a sphere of radius `body_0_racc`. this is the
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
fn body_mask_sdf_shaped(center: [Gv; 3], shape: Option<&SdfExpr<f64, 3>>) -> SdfExpr<Gv, 3> {
    match shape {
        None => SdfExpr::<Gv, 3>::Sphere { center, radius: Gv::scalar("body_0_racc") },
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
    shape.lift(&|c| Gv::from_f64(c)).rotated(rot).translated(center)
}

/// the dispatch output support for a body penalization kernel: the mask ball padded by
/// `DRAIN_SUPPORT_WIDTHS` cell widths, beyond which the tanh chi (and hence every delta and
/// the in-place cons write) is unchanged-valued. derived from the SAME geometry the mask SDF
/// traces, so support and mask describe one region.
fn body_support(ndim: usize) -> Support {
    body_support_shaped(ndim, None)
}

/// the support for an optionally-shaped body. `None`: the sphere ball at `body_0_pos_*` of
/// runtime radius `body_0_racc`. `Some(shape)`: the shape's bounding ball — a body-LOCAL
/// constant center + radius — translated to the runtime body position. a complement is
/// unbounded and has no ball, so a shaped body must be bounded.
fn body_support_shaped(ndim: usize, shape: Option<&SdfExpr<f64, 3>>) -> Support {
    let pad = ParamExpr::constant(crate::ibm::DRAIN_SUPPORT_WIDTHS)
        * ParamExpr::min_of((0..ndim).map(|a| ParamExpr::param(&format!("dx_{a}"))).collect());
    match shape {
        None => {
            let center_p: Vec<ParamExpr> =
                (0..ndim).map(|a| ParamExpr::param(&format!("body_0_pos_{a}"))).collect();
            Support::ball(center_p, ParamExpr::param("body_0_racc") + pad)
        }
        Some(s) => {
            let (lc, lr) = s
                .bounding_ball()
                .expect("a shaped immersed body must be bounded (a complement has no support ball)");
            let center_p: Vec<ParamExpr> = (0..ndim)
                .map(|a| ParamExpr::param(&format!("body_0_pos_{a}")) + ParamExpr::constant(lc[a]))
                .collect();
            Support::ball(center_p, ParamExpr::constant(lr) + pad)
        }
    }
}

/// trace the [Drain]-stack penalization for the adiabatic regime, cartesian,
/// DOF = ndim. dimension-generic over 1..=3.
pub fn penalize_drain_gv(coords: Coords, ndim: usize) -> (GvKernel, Writes) {
    assert!((1..=3).contains(&ndim), "penalize_drain_gv: ndim must be 1..=3");
    begin_trace();
    let dt = Gv::scalar("dt");
    let gamma = Gv::scalar("gamma");
    let c_drain = Gv::scalar("c_drain");

    // conserved reads (in-place: the same fields are the writes).
    let den = Gv::field("den", symbi_ir::FieldRef::cons_den());
    let mom: Vec<Gv> = (0..ndim)
        .map(|c| Gv::field(&format!("mom_{c}"), symbi_ir::FieldRef::cons_mom(c as u8)))
        .collect();
    let nrg = Gv::field("nrg", symbi_ir::FieldRef::cons_nrg());

    // the cell centroid + volume from the shared geometry scaffold — the coordinate
    // map every other body kernel of this chart evaluates.
    let geo = cell_geometry_gv(coords, &vec![Spacing::Uniform; ndim], &(0..ndim).collect::<Vec<_>>(), ndim);
    let dv = Gv::ONE / geo.inv_volume;
    let mut min_w = Gv::scalar("dx_0");
    for ax in 1..ndim {
        min_w = min_w.min(Gv::scalar(&format!("dx_{ax}")));
    }

    // the mask geometry as a traced SDF: phi = |x - body_pos| - r_mask, and
    // chi = 0.5 (1 - tanh(phi / w)). the distance is PHYSICAL, so the coordinate
    // centroid is mapped to Cartesian first (identity on a Cartesian grid).
    let center: [Gv; 3] = std::array::from_fn(|a| {
        if a < ndim { Gv::scalar(&format!("body_0_pos_{a}")) } else { Gv::ZERO }
    });
    let x_cart = centroid_to_cartesian(coords, ndim, &geo.centroid);
    let x: [Gv; 3] = std::array::from_fn(|a| {
        if a < ndim { x_cart[a] } else { Gv::ZERO }
    });
    let sphere = body_mask_sdf(center);
    let phi = sphere.dist(x);
    let chi = symbi_ib::sdf::chi(phi, min_w);

    // tau = c_drain dx / c_s with c_s from the just-updated conserved state
    // (the drain runs post-godunov, pre-c2p — the stored primitive is stale).
    let mut mom_sq = Gv::ZERO;
    for m in &mom {
        mom_sq = mom_sq + *m * *m;
    }
    let cs = symbi_ib::drain::sound_speed_from_cons(den, mom_sq, nrg, gamma);
    let inv_tau = signal_speed(cs) / (c_drain * min_w);

    // the property stack (docs/design/50): [Drain]. contribute at Gv, then
    // the SAME integrator the f64 oracle runs.
    let kin = BodyKin::<Gv, 3> {
        u_solid: Tensor::zeros(),
        omega: Tensor::zeros(),
        e_wall: Gv::ZERO,
    };
    let mut acc = Relax::<Gv, 3>::none();
    Property::Drain { inv_tau }.contribute(chi, &kin, &mut acc);
    let cons = ConsG::<Gv, 3, Adiabatic> {
        den,
        mom: Tensor::new(std::array::from_fn(|a| if a < ndim { mom[a] } else { Gv::ZERO })),
        nrg,
    };
    let (out, delta) = penalize_cell(&cons, &acc, Tensor::zeros(), dt, dv, 0);
    // the angular-momentum receipt: the lab-frame moment r_cart x F_cart of the
    // cell's force receipt about the body center. exactly zero beyond the support
    // ball (F vanishes there). 3d books all three components; 2d only z; 1d none.
    let torque = lab_torque(coords, ndim, &geo.centroid, &x, &center, &delta.force_delta);

    let mut writes: Writes = Vec::new();
    writes.push(("den_out".to_string(), symbi_ir::FieldRef::cons_den().into(), out.den.node()));
    for a in 0..ndim {
        writes.push((
            format!("mom_out_{a}"),
            symbi_ir::FieldRef::cons_mom(a as u8).into(),
            out.mom[a].node(),
        ));
    }
    writes.push(("nrg_out".to_string(), symbi_ir::FieldRef::cons_nrg().into(), out.nrg.node()));
    writes.push(("pen_mass".to_string(), "pen_0_mass".into(), delta.mass_delta.node()));
    for a in 0..ndim {
        writes.push((
            format!("pen_force_{a}"),
            format!("pen_0_force_{a}").into(),
            delta.force_delta[a].node(),
        ));
    }
    writes.push(("pen_energy".to_string(), "pen_0_energy".into(), delta.energy_delta.node()));
    for a in torque_axes(ndim) {
        writes.push((
            format!("pen_torque_{a}"),
            format!("pen_0_torque_{a}").into(),
            torque[a].node(),
        ));
    }

    // the delta outputs vanish exactly beyond the tanh saturation radius; the
    // in-place cons writes are unchanged-value there, so the ball bounds
    // everything the reduction needs (dispatch may clip to it).
    let kernel = end_trace().with_output_support(body_support(ndim));
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
pub fn penalize_porous_gv(coords: Coords, ndim: usize) -> (GvKernel, Writes) {
    penalize_porous_inner(coords, ndim, None, false)
}

/// the arbitrary-shape porous wall: the same relaxation stack as `penalize_porous_gv`, but the
/// mask AND the surface normal come from the config CSG `shape` (body-local, translated to the
/// runtime body position) instead of the sphere. built at sim SETUP, once the body's shape is
/// known — the AOT bake cannot know a per-body CSG.
pub fn penalize_porous_gv_shaped(
    coords: Coords,
    ndim: usize,
    shape: &SdfExpr<f64, 3>,
) -> (GvKernel, Writes) {
    penalize_porous_inner(coords, ndim, Some(shape), false)
}

/// the SPINNING arbitrary-shape porous wall: like `penalize_porous_gv_shaped`, but the mask is
/// rotated by the runtime orientation `R(body_0_angle)` and the surface velocity carries the spin
/// `omega x r` (`body_0_omega`), so the wall drags the gas around as it turns.
pub fn penalize_porous_gv_spinning(
    coords: Coords,
    ndim: usize,
    shape: &SdfExpr<f64, 3>,
) -> (GvKernel, Writes) {
    penalize_porous_inner(coords, ndim, Some(shape), true)
}

fn penalize_porous_inner(
    coords: Coords,
    ndim: usize,
    shape: Option<&SdfExpr<f64, 3>>,
    spin: bool,
) -> (GvKernel, Writes) {
    assert!((1..=3).contains(&ndim), "penalize_porous_gv: ndim must be 1..=3");
    begin_trace();
    let dt = Gv::scalar("dt");
    let gamma = Gv::scalar("gamma");
    let c_drain = Gv::scalar("c_drain");
    let porosity = Gv::scalar("porosity");
    let k_eta_n = Gv::scalar("k_eta_n");
    let k_eta_t = Gv::scalar("k_eta_t");

    let den = Gv::field("den", symbi_ir::FieldRef::cons_den());
    let mom: Vec<Gv> = (0..ndim)
        .map(|c| Gv::field(&format!("mom_{c}"), symbi_ir::FieldRef::cons_mom(c as u8)))
        .collect();
    let nrg = Gv::field("nrg", symbi_ir::FieldRef::cons_nrg());

    let geo = cell_geometry_gv(coords, &vec![Spacing::Uniform; ndim], &(0..ndim).collect::<Vec<_>>(), ndim);
    let dv = Gv::ONE / geo.inv_volume;
    let mut min_w = Gv::scalar("dx_0");
    for ax in 1..ndim {
        min_w = min_w.min(Gv::scalar(&format!("dx_{ax}")));
    }

    let center: [Gv; 3] = std::array::from_fn(|a| {
        if a < ndim { Gv::scalar(&format!("body_0_pos_{a}")) } else { Gv::ZERO }
    });
    // the mask distance is PHYSICAL: map the coordinate centroid to Cartesian.
    let x_cart = centroid_to_cartesian(coords, ndim, &geo.centroid);
    let x: [Gv; 3] = std::array::from_fn(|a| {
        if a < ndim { x_cart[a] } else { Gv::ZERO }
    });
    let sdf = if spin {
        body_mask_sdf_spinning(center, shape.expect("a spinning wall must have a shape"))
    } else {
        body_mask_sdf_shaped(center, shape)
    };
    let phi = sdf.dist(x);
    let chi = symbi_ib::sdf::chi(phi, min_w);

    let mut mom_sq = Gv::ZERO;
    for m in &mom {
        mom_sq = mom_sq + *m * *m;
    }
    let cs = symbi_ib::drain::sound_speed_from_cons(den, mom_sq, nrg, gamma);
    let inv_tau = signal_speed(cs) / (c_drain * min_w);
    let rate_scale = signal_speed(cs) / min_w;

    // the outward surface normal in the cell's PHYSICAL frame (the Cartesian normal rotated into
    // the orthonormal basis; identity on a Cartesian grid). the sphere path is r_hat =
    // x_rel/|x_rel|, with |x_rel| = 0 guarded to a zero normal (its whole du is tangential,
    // finite). the shaped path is the exact CSG gradient — the SDF outward unit normal via Dual
    // autodiff — which is the arbitrary surface's normal everywhere the distance is smooth.
    let n_cart: Vec<Gv> = match shape {
        None => {
            let x_rel = Tensor::<Gv, 3>::new(std::array::from_fn(|a| x[a] - center[a]));
            let r = x_rel.dot(&x_rel).sqrt();
            let inv_r = Gv::ONE / r.max(Gv::from_f64(1e-300));
            (0..ndim).map(|a| x_rel[a] * inv_r).collect()
        }
        Some(_) => {
            let n = sdf.normal(x);
            (0..ndim).map(|a| n[a]).collect()
        }
    };
    let n_phys = vector_from_cartesian(coords, ndim, &geo.centroid, &n_cart);
    let normal = Tensor::<Gv, 3>::new(std::array::from_fn(|a| {
        if a < ndim { n_phys[a] } else { Gv::ZERO }
    }));

    let u_solid = Tensor::<Gv, 3>::new(std::array::from_fn(|a| {
        if a < ndim { Gv::scalar(&format!("body_0_vel_{a}")) } else { Gv::ZERO }
    }));
    // a spinning wall's surface moves at u_solid + omega x r; omega is about z (`body_0_omega`).
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
    let base_kin = BodyKin::<Gv, 3> { u_solid, omega, e_wall: Gv::ZERO };
    // a spinning wall's velocity target is the LOCAL rigid-motion velocity u_solid + omega x r_cell;
    // `at` bakes omega x r into u_solid per cell (the contribute path reads u_solid directly). the
    // static path keeps the bare translational u_solid, bit-identical to before.
    let kin = if spin {
        let x_rel = Tensor::<Gv, 3>::new(std::array::from_fn(|a| x[a] - center[a]));
        base_kin.at(&x_rel)
    } else {
        base_kin
    };
    let mut acc = Relax::<Gv, 3>::none();
    Property::PorousAccretor {
        p: porosity,
        inv_tau,
        inv_eta_n: k_eta_n * rate_scale,
        inv_eta_t: k_eta_t * rate_scale,
    }
    .contribute(chi, &kin, &mut acc);
    let cons = ConsG::<Gv, 3, Adiabatic> {
        den,
        mom: Tensor::new(std::array::from_fn(|a| if a < ndim { mom[a] } else { Gv::ZERO })),
        nrg,
    };
    let (out, delta) = penalize_cell(&cons, &acc, normal, dt, dv, 0);
    let torque = lab_torque(coords, ndim, &geo.centroid, &x, &center, &delta.force_delta);

    let mut writes: Writes = Vec::new();
    writes.push(("den_out".to_string(), symbi_ir::FieldRef::cons_den().into(), out.den.node()));
    for a in 0..ndim {
        writes.push((
            format!("mom_out_{a}"),
            symbi_ir::FieldRef::cons_mom(a as u8).into(),
            out.mom[a].node(),
        ));
    }
    writes.push(("nrg_out".to_string(), symbi_ir::FieldRef::cons_nrg().into(), out.nrg.node()));
    writes.push(("pen_mass".to_string(), "pen_0_mass".into(), delta.mass_delta.node()));
    for a in 0..ndim {
        writes.push((
            format!("pen_force_{a}"),
            format!("pen_0_force_{a}").into(),
            delta.force_delta[a].node(),
        ));
    }
    writes.push(("pen_energy".to_string(), "pen_0_energy".into(), delta.energy_delta.node()));
    for a in torque_axes(ndim) {
        writes.push((
            format!("pen_torque_{a}"),
            format!("pen_0_torque_{a}").into(),
            torque[a].node(),
        ));
    }

    let kernel = end_trace().with_output_support(body_support_shaped(ndim, shape));
    (kernel, writes)
}

/// the ISOTHERMAL torque-free accretor: the drain plus a
/// tangential ANTI-relaxation `lambda_t = -xi lambda_rho` about the sphere
/// normal, so the accreted mass carries no net angular momentum to the body
/// (the Dittmann & Ryan 2021 torque-free sink, coordinate-free via the SDF
/// normal). the retention floor (`Relax.ut_growth_cap`, set by the property)
/// bounds the growing tangential factor at the evacuation limit. `xi = 0`
/// reduces bit-for-bit to `penalize_drain_iso`; `xi = 1` is torque-free. no
/// energy channel (iso, the thin-disk regime); delta outputs mass + force +
/// torque. the velocity target is the body's translational velocity
/// `body_0_vel_*` (the accretion is torque-free RELATIVE to the moving sink).
pub fn penalize_torque_free_iso_gv(coords: Coords, ndim: usize) -> (GvKernel, Writes) {
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
    let mom: Vec<Gv> = (0..ndim)
        .map(|c| Gv::field(&format!("mom_{c}"), symbi_ir::FieldRef::cons_mom(c as u8)))
        .collect();

    let geo = cell_geometry_gv(coords, &vec![Spacing::Uniform; ndim], &(0..ndim).collect::<Vec<_>>(), ndim);
    let dv = Gv::ONE / geo.inv_volume;
    let mut min_w = Gv::scalar("dx_0");
    for ax in 1..ndim {
        min_w = min_w.min(Gv::scalar(&format!("dx_{ax}")));
    }

    let center: [Gv; 3] = std::array::from_fn(|a| {
        if a < ndim { Gv::scalar(&format!("body_0_pos_{a}")) } else { Gv::ZERO }
    });
    // the mask distance is PHYSICAL: map the coordinate centroid to Cartesian.
    let x_cart = centroid_to_cartesian(coords, ndim, &geo.centroid);
    let x: [Gv; 3] = std::array::from_fn(|a| {
        if a < ndim { x_cart[a] } else { Gv::ZERO }
    });
    let sphere = body_mask_sdf(center);
    let chi = symbi_ib::sdf::chi(sphere.dist(x), min_w);
    let inv_tau = signal_speed(cs) / (c_drain * min_w);

    // the outward surface normal in the cell's PHYSICAL frame: the Cartesian
    // r_hat from the body center rotated into the orthonormal basis (identity on
    // a Cartesian grid; e_r for a centered accretor). the torque-free channel's
    // radial/tangential split is about this normal, so it must match the frame
    // the momentum lives in.
    let x_rel = Tensor::<Gv, 3>::new(std::array::from_fn(|a| x[a] - center[a]));
    let r = x_rel.dot(&x_rel).sqrt();
    let inv_r = Gv::ONE / r.max(Gv::from_f64(1e-300));
    let n_cart: Vec<Gv> = (0..ndim).map(|a| x_rel[a] * inv_r).collect();
    let n_phys = vector_from_cartesian(coords, ndim, &geo.centroid, &n_cart);
    let normal = Tensor::<Gv, 3>::new(std::array::from_fn(|a| {
        if a < ndim { n_phys[a] } else { Gv::ZERO }
    }));

    let u_solid = Tensor::<Gv, 3>::new(std::array::from_fn(|a| {
        if a < ndim { Gv::scalar(&format!("body_0_vel_{a}")) } else { Gv::ZERO }
    }));
    let kin = BodyKin::<Gv, 3> { u_solid, omega: Tensor::zeros(), e_wall: Gv::ZERO };
    let mut acc = Relax::<Gv, 3>::none();
    Property::TorqueFreeAccretor { inv_tau, xi }.contribute(chi, &kin, &mut acc);
    let cons = ConsG::<Gv, 3, IsoModel> {
        den,
        mom: Tensor::new(std::array::from_fn(|a| if a < ndim { mom[a] } else { Gv::ZERO })),
        nrg: Default::default(),
    };
    let (out, delta) = penalize_cell(&cons, &acc, normal, dt, dv, 0);
    let torque = lab_torque(coords, ndim, &geo.centroid, &x, &center, &delta.force_delta);

    let mut writes: Writes = Vec::new();
    writes.push(("den_out".to_string(), symbi_ir::FieldRef::cons_den().into(), out.den.node()));
    for a in 0..ndim {
        writes.push((
            format!("mom_out_{a}"),
            symbi_ir::FieldRef::cons_mom(a as u8).into(),
            out.mom[a].node(),
        ));
    }
    writes.push(("pen_mass".to_string(), "pen_0_mass".into(), delta.mass_delta.node()));
    for a in 0..ndim {
        writes.push((
            format!("pen_force_{a}"),
            format!("pen_0_force_{a}").into(),
            delta.force_delta[a].node(),
        ));
    }
    for a in torque_axes(ndim) {
        writes.push((
            format!("pen_torque_{a}"),
            format!("pen_0_torque_{a}").into(),
            torque[a].node(),
        ));
    }

    let kernel = end_trace().with_output_support(body_support(ndim));
    (kernel, writes)
}

/// the ISOTHERMAL [PorousAccretor] twin: the same porous surface as
/// `penalize_porous_gv` but for the isothermal regime — the sound speed is the
/// constant `cs` param (no energy channel), delta outputs mass + force only.
/// `p = 1` reduces bit-for-bit to `penalize_drain_iso`.
pub fn penalize_porous_iso_gv(coords: Coords, ndim: usize) -> (GvKernel, Writes) {
    penalize_porous_iso_inner(coords, ndim, None, false)
}

/// the arbitrary-shape ISO porous wall: the energy-free counterpart of
/// `penalize_porous_gv_shaped` — mask + normal from the config CSG, no energy channel.
pub fn penalize_porous_iso_gv_shaped(
    coords: Coords,
    ndim: usize,
    shape: &SdfExpr<f64, 3>,
) -> (GvKernel, Writes) {
    penalize_porous_iso_inner(coords, ndim, Some(shape), false)
}

/// the SPINNING ISO porous wall: the energy-free counterpart of `penalize_porous_gv_spinning`.
pub fn penalize_porous_iso_gv_spinning(
    coords: Coords,
    ndim: usize,
    shape: &SdfExpr<f64, 3>,
) -> (GvKernel, Writes) {
    penalize_porous_iso_inner(coords, ndim, Some(shape), true)
}

fn penalize_porous_iso_inner(
    coords: Coords,
    ndim: usize,
    shape: Option<&SdfExpr<f64, 3>>,
    spin: bool,
) -> (GvKernel, Writes) {
    use symbi_hydro::energy::IsoModel;
    assert!((1..=3).contains(&ndim), "penalize_porous_iso_gv: ndim must be 1..=3");
    begin_trace();
    let dt = Gv::scalar("dt");
    let cs = Gv::scalar("cs");
    let c_drain = Gv::scalar("c_drain");
    let porosity = Gv::scalar("porosity");
    let k_eta_n = Gv::scalar("k_eta_n");
    let k_eta_t = Gv::scalar("k_eta_t");

    let den = Gv::field("den", symbi_ir::FieldRef::cons_den());
    let mom: Vec<Gv> = (0..ndim)
        .map(|c| Gv::field(&format!("mom_{c}"), symbi_ir::FieldRef::cons_mom(c as u8)))
        .collect();

    let geo = cell_geometry_gv(coords, &vec![Spacing::Uniform; ndim], &(0..ndim).collect::<Vec<_>>(), ndim);
    let dv = Gv::ONE / geo.inv_volume;
    let mut min_w = Gv::scalar("dx_0");
    for ax in 1..ndim {
        min_w = min_w.min(Gv::scalar(&format!("dx_{ax}")));
    }

    let center: [Gv; 3] = std::array::from_fn(|a| {
        if a < ndim { Gv::scalar(&format!("body_0_pos_{a}")) } else { Gv::ZERO }
    });
    let x_cart = centroid_to_cartesian(coords, ndim, &geo.centroid);
    let x: [Gv; 3] = std::array::from_fn(|a| if a < ndim { x_cart[a] } else { Gv::ZERO });
    let sdf = if spin {
        body_mask_sdf_spinning(center, shape.expect("a spinning wall must have a shape"))
    } else {
        body_mask_sdf_shaped(center, shape)
    };
    let chi = symbi_ib::sdf::chi(sdf.dist(x), min_w);
    let inv_tau = signal_speed(cs) / (c_drain * min_w);
    let rate_scale = signal_speed(cs) / min_w;

    // sphere normal r_hat (guarded), or the CSG SDF gradient for a shaped wall (see the adiabatic
    // `penalize_porous_gv`).
    let n_cart: Vec<Gv> = match shape {
        None => {
            let x_rel = Tensor::<Gv, 3>::new(std::array::from_fn(|a| x[a] - center[a]));
            let r = x_rel.dot(&x_rel).sqrt();
            let inv_r = Gv::ONE / r.max(Gv::from_f64(1e-300));
            (0..ndim).map(|a| x_rel[a] * inv_r).collect()
        }
        Some(_) => {
            let n = sdf.normal(x);
            (0..ndim).map(|a| n[a]).collect()
        }
    };
    let n_phys = vector_from_cartesian(coords, ndim, &geo.centroid, &n_cart);
    let normal = Tensor::<Gv, 3>::new(std::array::from_fn(|a| {
        if a < ndim { n_phys[a] } else { Gv::ZERO }
    }));

    let u_solid = Tensor::<Gv, 3>::new(std::array::from_fn(|a| {
        if a < ndim { Gv::scalar(&format!("body_0_vel_{a}")) } else { Gv::ZERO }
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
    let base_kin = BodyKin::<Gv, 3> { u_solid, omega, e_wall: Gv::ZERO };
    // a spinning wall's velocity target is the LOCAL rigid-motion velocity u_solid + omega x r_cell;
    // `at` bakes omega x r into u_solid per cell (the contribute path reads u_solid directly). the
    // static path keeps the bare translational u_solid, bit-identical to before.
    let kin = if spin {
        let x_rel = Tensor::<Gv, 3>::new(std::array::from_fn(|a| x[a] - center[a]));
        base_kin.at(&x_rel)
    } else {
        base_kin
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
        den,
        mom: Tensor::new(std::array::from_fn(|a| if a < ndim { mom[a] } else { Gv::ZERO })),
        nrg: Default::default(),
    };
    let (out, delta) = penalize_cell(&cons, &acc, normal, dt, dv, 0);
    let torque = lab_torque(coords, ndim, &geo.centroid, &x, &center, &delta.force_delta);

    let mut writes: Writes = Vec::new();
    writes.push(("den_out".to_string(), symbi_ir::FieldRef::cons_den().into(), out.den.node()));
    for a in 0..ndim {
        writes.push((format!("mom_out_{a}"), symbi_ir::FieldRef::cons_mom(a as u8).into(), out.mom[a].node()));
    }
    writes.push(("pen_mass".to_string(), "pen_0_mass".into(), delta.mass_delta.node()));
    for a in 0..ndim {
        writes.push((format!("pen_force_{a}"), format!("pen_0_force_{a}").into(), delta.force_delta[a].node()));
    }
    for a in torque_axes(ndim) {
        writes.push((format!("pen_torque_{a}"), format!("pen_0_torque_{a}").into(), torque[a].node()));
    }

    let kernel = end_trace().with_output_support(body_support_shaped(ndim, shape));
    (kernel, writes)
}

/// the ADIABATIC torque-free twin: the same torque-free surface as
/// `penalize_torque_free_iso_gv` but for the adiabatic regime — the sound speed
/// is recovered from the conserved state and the energy channel is carried (delta
/// outputs mass + force + energy + torque). `xi = 0` reduces to `penalize_drain`.
pub fn penalize_torque_free_gv(coords: Coords, ndim: usize) -> (GvKernel, Writes) {
    assert!((1..=3).contains(&ndim), "penalize_torque_free_gv: ndim must be 1..=3");
    begin_trace();
    let dt = Gv::scalar("dt");
    let gamma = Gv::scalar("gamma");
    let c_drain = Gv::scalar("c_drain");
    let xi = Gv::scalar("xi");

    let den = Gv::field("den", symbi_ir::FieldRef::cons_den());
    let mom: Vec<Gv> = (0..ndim)
        .map(|c| Gv::field(&format!("mom_{c}"), symbi_ir::FieldRef::cons_mom(c as u8)))
        .collect();
    let nrg = Gv::field("nrg", symbi_ir::FieldRef::cons_nrg());

    let geo = cell_geometry_gv(coords, &vec![Spacing::Uniform; ndim], &(0..ndim).collect::<Vec<_>>(), ndim);
    let dv = Gv::ONE / geo.inv_volume;
    let mut min_w = Gv::scalar("dx_0");
    for ax in 1..ndim {
        min_w = min_w.min(Gv::scalar(&format!("dx_{ax}")));
    }

    let center: [Gv; 3] = std::array::from_fn(|a| {
        if a < ndim { Gv::scalar(&format!("body_0_pos_{a}")) } else { Gv::ZERO }
    });
    let x_cart = centroid_to_cartesian(coords, ndim, &geo.centroid);
    let x: [Gv; 3] = std::array::from_fn(|a| if a < ndim { x_cart[a] } else { Gv::ZERO });
    let sphere = body_mask_sdf(center);
    let chi = symbi_ib::sdf::chi(sphere.dist(x), min_w);

    let mut mom_sq = Gv::ZERO;
    for m in &mom {
        mom_sq = mom_sq + *m * *m;
    }
    let cs = symbi_ib::drain::sound_speed_from_cons(den, mom_sq, nrg, gamma);
    let inv_tau = signal_speed(cs) / (c_drain * min_w);

    let x_rel = Tensor::<Gv, 3>::new(std::array::from_fn(|a| x[a] - center[a]));
    let r = x_rel.dot(&x_rel).sqrt();
    let inv_r = Gv::ONE / r.max(Gv::from_f64(1e-300));
    let n_cart: Vec<Gv> = (0..ndim).map(|a| x_rel[a] * inv_r).collect();
    let n_phys = vector_from_cartesian(coords, ndim, &geo.centroid, &n_cart);
    let normal = Tensor::<Gv, 3>::new(std::array::from_fn(|a| {
        if a < ndim { n_phys[a] } else { Gv::ZERO }
    }));

    let u_solid = Tensor::<Gv, 3>::new(std::array::from_fn(|a| {
        if a < ndim { Gv::scalar(&format!("body_0_vel_{a}")) } else { Gv::ZERO }
    }));
    let kin = BodyKin::<Gv, 3> { u_solid, omega: Tensor::zeros(), e_wall: Gv::ZERO };
    let mut acc = Relax::<Gv, 3>::none();
    Property::TorqueFreeAccretor { inv_tau, xi }.contribute(chi, &kin, &mut acc);
    let cons = ConsG::<Gv, 3, Adiabatic> {
        den,
        mom: Tensor::new(std::array::from_fn(|a| if a < ndim { mom[a] } else { Gv::ZERO })),
        nrg,
    };
    let (out, delta) = penalize_cell(&cons, &acc, normal, dt, dv, 0);
    let torque = lab_torque(coords, ndim, &geo.centroid, &x, &center, &delta.force_delta);

    let mut writes: Writes = Vec::new();
    writes.push(("den_out".to_string(), symbi_ir::FieldRef::cons_den().into(), out.den.node()));
    for a in 0..ndim {
        writes.push((format!("mom_out_{a}"), symbi_ir::FieldRef::cons_mom(a as u8).into(), out.mom[a].node()));
    }
    writes.push(("nrg_out".to_string(), symbi_ir::FieldRef::cons_nrg().into(), out.nrg.node()));
    writes.push(("pen_mass".to_string(), "pen_0_mass".into(), delta.mass_delta.node()));
    for a in 0..ndim {
        writes.push((format!("pen_force_{a}"), format!("pen_0_force_{a}").into(), delta.force_delta[a].node()));
    }
    writes.push(("pen_energy".to_string(), "pen_0_energy".into(), delta.energy_delta.node()));
    for a in torque_axes(ndim) {
        writes.push((format!("pen_torque_{a}"), format!("pen_0_torque_{a}").into(), torque[a].node()));
    }

    let kernel = end_trace().with_output_support(body_support(ndim));
    (kernel, writes)
}

/// the ISOTHERMAL twin: no energy channel (the drain scales den + mom; the
/// sound speed is the constant `cs` param, not recovered from cons), delta
/// outputs mass + force only. same [Drain] stack, same integrator — the iso
/// energy slot discards the e-channel by construction.
pub fn penalize_drain_iso_gv(coords: Coords, ndim: usize) -> (GvKernel, Writes) {
    use symbi_hydro::energy::IsoModel;
    assert!((1..=3).contains(&ndim), "penalize_drain_iso_gv: ndim must be 1..=3");
    begin_trace();
    let dt = Gv::scalar("dt");
    let cs = Gv::scalar("cs");
    let c_drain = Gv::scalar("c_drain");

    let den = Gv::field("den", symbi_ir::FieldRef::cons_den());
    let mom: Vec<Gv> = (0..ndim)
        .map(|c| Gv::field(&format!("mom_{c}"), symbi_ir::FieldRef::cons_mom(c as u8)))
        .collect();

    let geo = cell_geometry_gv(coords, &vec![Spacing::Uniform; ndim], &(0..ndim).collect::<Vec<_>>(), ndim);
    let dv = Gv::ONE / geo.inv_volume;
    let mut min_w = Gv::scalar("dx_0");
    for ax in 1..ndim {
        min_w = min_w.min(Gv::scalar(&format!("dx_{ax}")));
    }

    let center: [Gv; 3] = std::array::from_fn(|a| {
        if a < ndim { Gv::scalar(&format!("body_0_pos_{a}")) } else { Gv::ZERO }
    });
    // the mask distance is the PHYSICAL distance to the body: map the coordinate
    // centroid to Cartesian (identity on a Cartesian grid), then the euclidean SDF.
    let x_cart = centroid_to_cartesian(coords, ndim, &geo.centroid);
    let x: [Gv; 3] = std::array::from_fn(|a| {
        if a < ndim { x_cart[a] } else { Gv::ZERO }
    });
    let sphere = body_mask_sdf(center);
    let chi = symbi_ib::sdf::chi(sphere.dist(x), min_w);
    let inv_tau = signal_speed(cs) / (c_drain * min_w);

    let kin = BodyKin::<Gv, 3> {
        u_solid: Tensor::zeros(),
        omega: Tensor::zeros(),
        e_wall: Gv::ZERO,
    };
    let mut acc = Relax::<Gv, 3>::none();
    Property::Drain { inv_tau }.contribute(chi, &kin, &mut acc);
    let cons = ConsG::<Gv, 3, IsoModel> {
        den,
        mom: Tensor::new(std::array::from_fn(|a| if a < ndim { mom[a] } else { Gv::ZERO })),
        nrg: Default::default(),
    };
    let (out, delta) = penalize_cell(&cons, &acc, Tensor::zeros(), dt, dv, 0);
    // the angular-momentum receipt, identical booking to the adiabatic twin.
    let torque = lab_torque(coords, ndim, &geo.centroid, &x, &center, &delta.force_delta);

    let mut writes: Writes = Vec::new();
    writes.push(("den_out".to_string(), symbi_ir::FieldRef::cons_den().into(), out.den.node()));
    for a in 0..ndim {
        writes.push((
            format!("mom_out_{a}"),
            symbi_ir::FieldRef::cons_mom(a as u8).into(),
            out.mom[a].node(),
        ));
    }
    writes.push(("pen_mass".to_string(), "pen_0_mass".into(), delta.mass_delta.node()));
    for a in 0..ndim {
        writes.push((
            format!("pen_force_{a}"),
            format!("pen_0_force_{a}").into(),
            delta.force_delta[a].node(),
        ));
    }
    for a in torque_axes(ndim) {
        writes.push((
            format!("pen_torque_{a}"),
            format!("pen_0_torque_{a}").into(),
            torque[a].node(),
        ));
    }

    let kernel = end_trace().with_output_support(body_support(ndim));
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
        let (kernel, writes) = penalize_porous_gv_shaped(Coords::Cartesian, 3, &shape);
        assert!(!kernel.graph.has_errors(), "shaped porous kernel traced with graph errors");
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
            assert!(kernel.scalar_params.iter().any(|s| s == p), "missing runtime param {p}");
        }
        assert!(!writes.is_empty(), "the shaped kernel emitted no writes");
    }

    // the unshaped path is the sphere: `body_0_racc` IS a runtime scalar (the AOT kernel).
    #[test]
    fn unshaped_porous_kernel_reads_the_runtime_radius() {
        let (kernel, _) = penalize_porous_gv(Coords::Cartesian, 3);
        assert!(kernel.scalar_params.iter().any(|p| p == "body_0_racc"));
    }
}
