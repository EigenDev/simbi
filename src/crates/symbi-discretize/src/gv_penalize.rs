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
use symbi_algebra::Tensor;
use symbi_hydro::energy::Adiabatic;
use symbi_hydro::state::ConsG;
use symbi_ib::penalize::{penalize_cell, BodyKin, Property, Relax};
use symbi_ib::sdf::SdfExpr;
use symbi_ir::gv::Writes;
use symbi_ir::{Gv, GvKernel, ParamExpr, Support};

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

/// trace the [Drain]-stack penalization for the adiabatic regime, cartesian,
/// DOF = ndim. dimension-generic over 1..=3.
pub fn penalize_drain_gv(ndim: usize) -> (GvKernel, Writes) {
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

    // the cell centroid + volume from the shared geometry scaffold — the same
    // coordinate map every other cartesian body kernel evaluates.
    let geo = cell_geometry_gv(Coords::Cartesian, &vec![Spacing::Uniform; ndim], &(0..ndim).collect::<Vec<_>>(), ndim);
    let dv = Gv::ONE / geo.inv_volume;
    let mut min_w = Gv::scalar("dx_0");
    for ax in 1..ndim {
        min_w = min_w.min(Gv::scalar(&format!("dx_{ax}")));
    }

    // the mask geometry as a traced SDF: phi = |x - body_pos| - r_mask, and
    // chi = 0.5 (1 - tanh(phi / w)) == drain_mask(|x - c|, r_mask, w) bit for
    // bit (same subtraction, same division, same tanh).
    let center: [Gv; 3] = std::array::from_fn(|a| {
        if a < ndim { Gv::scalar(&format!("body_0_pos_{a}")) } else { Gv::ZERO }
    });
    let x: [Gv; 3] = std::array::from_fn(|a| {
        if a < ndim { geo.centroid[a] } else { Gv::ZERO }
    });
    let r_mask = Gv::scalar("body_0_racc");
    let sphere = SdfExpr::<Gv, 3>::Sphere { center, radius: r_mask };
    let phi = sphere.dist(x);
    let chi = symbi_ib::sdf::chi(phi, min_w);

    // tau = c_drain dx / c_s with c_s from the just-updated conserved state
    // (the drain runs post-godunov, pre-c2p — the stored primitive is stale).
    let mut mom_sq = Gv::ZERO;
    for m in &mom {
        mom_sq = mom_sq + *m * *m;
    }
    let cs = symbi_ib::drain::sound_speed_from_cons(den, mom_sq, nrg, gamma);
    let inv_tau = cs / (c_drain * min_w);

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
    // the angular-momentum receipt: the moment of the cell's force receipt
    // about the body center, r x F with r = x - body_pos. exactly zero beyond
    // the support ball (F vanishes there), so the same reduction box applies.
    // 3d books all three components; 2d only z; 1d none.
    let x_rel = Tensor::<Gv, 3>::new(std::array::from_fn(|a| x[a] - center[a]));
    let torque = symbi_ib::moment(&x_rel, &delta.force_delta);

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
    let center_p: Vec<ParamExpr> =
        (0..ndim).map(|a| ParamExpr::param(&format!("body_0_pos_{a}"))).collect();
    let radius_p = ParamExpr::param("body_0_racc")
        + ParamExpr::constant(crate::ibm::DRAIN_SUPPORT_WIDTHS)
            * ParamExpr::min_of((0..ndim).map(|a| ParamExpr::param(&format!("dx_{a}"))).collect());
    let kernel = end_trace().with_output_support(Support::ball(center_p, radius_p));
    (kernel, writes)
}

/// the ISOTHERMAL twin: no energy channel (the drain scales den + mom; the
/// sound speed is the constant `cs` param, not recovered from cons), delta
/// outputs mass + force only. same [Drain] stack, same integrator — the iso
/// energy slot discards the e-channel by construction.
pub fn penalize_drain_iso_gv(ndim: usize) -> (GvKernel, Writes) {
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

    let geo = cell_geometry_gv(Coords::Cartesian, &vec![Spacing::Uniform; ndim], &(0..ndim).collect::<Vec<_>>(), ndim);
    let dv = Gv::ONE / geo.inv_volume;
    let mut min_w = Gv::scalar("dx_0");
    for ax in 1..ndim {
        min_w = min_w.min(Gv::scalar(&format!("dx_{ax}")));
    }

    let center: [Gv; 3] = std::array::from_fn(|a| {
        if a < ndim { Gv::scalar(&format!("body_0_pos_{a}")) } else { Gv::ZERO }
    });
    let x: [Gv; 3] = std::array::from_fn(|a| {
        if a < ndim { geo.centroid[a] } else { Gv::ZERO }
    });
    let r_mask = Gv::scalar("body_0_racc");
    let sphere = SdfExpr::<Gv, 3>::Sphere { center, radius: r_mask };
    let chi = symbi_ib::sdf::chi(sphere.dist(x), min_w);
    let inv_tau = cs / (c_drain * min_w);

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
    let x_rel = Tensor::<Gv, 3>::new(std::array::from_fn(|a| x[a] - center[a]));
    let torque = symbi_ib::moment(&x_rel, &delta.force_delta);

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

    let center_p: Vec<ParamExpr> =
        (0..ndim).map(|a| ParamExpr::param(&format!("body_0_pos_{a}"))).collect();
    let radius_p = ParamExpr::param("body_0_racc")
        + ParamExpr::constant(crate::ibm::DRAIN_SUPPORT_WIDTHS)
            * ParamExpr::min_of((0..ndim).map(|a| ParamExpr::param(&format!("dx_{a}"))).collect());
    let kernel = end_trace().with_output_support(Support::ball(center_p, radius_p));
    (kernel, writes)
}
