// =============================================================================
// support_inference.rs
//
// the derived-support parity law: every body kernel's output support is
// derived — the mask
// seam tags chi's saturation ball and propagation carries it through the
// property algebra to the write roots — and must reproduce the geometry the
// hand declarations stated: ball(body_0_pos, racc + DRAIN_SUPPORT_WIDTHS *
// min(dx)) for sphere masks, the shape's bounding ball for CSG masks. gates:
// - every cartesian penalize / feedback / resistive-EMF builder derives that
//   exact ball (numerically, under a fixed scalar table);
// - a spinning shape derives the position-centered swept ball |lc| + lr + pad
//   — strictly wider than the static shape ball, since it contains the mask at
//   every rotation angle, a half turn included;
// - a curvilinear chart derives Everywhere (a cartesian mask region is a
//   coordinate ball on the identity chart alone) — the fail-safe, matching
//   the whole-interior dispatch fallback;
// - a kernel whose write chain breaks the mask algebra derives Everywhere, so
//   a broken chain widens the support (gated in symbi-ir's support_infer unit
//   tests).
// =============================================================================

use symbi_discretize::coords::Coords;
use symbi_discretize::{
    body_feedback_drain_gv, body_resistive_emf_2d_gv, body_resistive_emf_3d_dir_gv,
    penalize_drain_gv, penalize_drain_iso_gv, penalize_porous_gv, penalize_porous_gv_shaped,
    penalize_porous_gv_spinning, penalize_porous_iso_gv, penalize_torque_free_gv,
    penalize_torque_free_iso_gv,
};
use symbi_ib::sdf::SdfExpr;
use symbi_ir::Support;

const POS: [f64; 3] = [0.13, -0.07, 0.4];
const RACC: f64 = 0.1;
const DX: [f64; 3] = [0.025, 0.05, 0.04];
const PAD: f64 = 20.0 * 0.025; // DRAIN_SUPPORT_WIDTHS * min(dx)

fn resolve(name: &str) -> f64 {
    match name {
        "dx_0" => DX[0],
        "dx_1" => DX[1],
        "dx_2" => DX[2],
        "body_0_pos_0" => POS[0],
        "body_0_pos_1" => POS[1],
        "body_0_pos_2" => POS[2],
        "body_0_racc" => RACC,
        other => panic!("support parity resolver: unexpected param '{other}'"),
    }
}

fn expect_ball(support: Option<&Support>, what: &str, center: &[f64], radius: f64) {
    let s = support.unwrap_or_else(|| panic!("{what}: no derived support"));
    let (c, r) = s
        .eval_ball(&resolve)
        .unwrap_or_else(|| panic!("{what}: derived support is not a Ball: {s:?}"));
    assert_eq!(c.len(), center.len(), "{what}: center rank");
    for (a, b) in c.iter().zip(center) {
        assert!((a - b).abs() < 1e-14, "{what}: center {c:?} != {center:?}");
    }
    assert!((r - radius).abs() < 1e-14, "{what}: radius {r} != {radius}");
}

#[test]
fn sphere_kernels_derive_the_declared_ball() {
    macro_rules! check {
        ($what:literal, $f:ident) => {{
            let (k, _) = $f(Coords::Cartesian, 2, 2, &[0, 1], false);
            expect_ball(k.output_support(), $what, &POS[..2], RACC + PAD);
        }};
    }
    check!("drain", penalize_drain_gv);
    check!("porous", penalize_porous_gv);
    check!("torque_free", penalize_torque_free_gv);
    check!("torque_free_iso", penalize_torque_free_iso_gv);
    check!("porous_iso", penalize_porous_iso_gv);
    check!("drain_iso", penalize_drain_iso_gv);
    // 2.5d (dof 3 on a 2d grid) shares the in-plane ball.
    let (k, _) = penalize_drain_gv(Coords::Cartesian, 2, 3, &[0, 1], false);
    expect_ball(k.output_support(), "drain 2.5d", &POS[..2], RACC + PAD);
    // 3d.
    let (k, _) = penalize_drain_gv(Coords::Cartesian, 3, 3, &[0, 1, 2], false);
    expect_ball(k.output_support(), "drain 3d", &POS, RACC + PAD);
    // the (r, z) axisymmetric section: the on-axis mask region is a coordinate
    // ball (identity embedding), so the sphere mask carries its ball there too.
    let (k, _) = penalize_drain_gv(Coords::Cylindrical, 2, 3, &[0, 2], false);
    expect_ball(k.output_support(), "drain rz", &POS[..2], RACC + PAD);
}

#[test]
fn resistive_emf_kernels_derive_the_declared_ball() {
    let (k, _) = body_resistive_emf_2d_gv(Coords::Cartesian);
    expect_ball(k.output_support(), "emf 2d", &POS[..2], RACC + PAD);
    for dir in 0..3 {
        let (k, _) = body_resistive_emf_3d_dir_gv(dir, Coords::Cartesian);
        // 3d pad uses min over all three widths — still dx_0 here.
        expect_ball(k.output_support(), "emf 3d", &POS, RACC + PAD);
    }
}

#[test]
fn feedback_drain_derives_the_declared_ball() {
    let (k, _) = body_feedback_drain_gv(Coords::Cartesian, 2, 2, &[0, 1]);
    expect_ball(k.output_support(), "feedback drain", &POS[..2], RACC + PAD);
}

#[test]
fn shaped_kernel_derives_the_shape_bounding_ball() {
    let shape = SdfExpr::<f64, 3>::cuboid([0.3, 0.0, 0.0], [0.5, 0.3, 0.2])
        .union(SdfExpr::sphere([0.6, 0.0, 0.0], 0.25));
    let (lc, lr) = shape.bounding_ball().expect("bounded shape");
    let (k, _) = penalize_porous_gv_shaped(Coords::Cartesian, 2, 2, &shape, false);
    expect_ball(
        k.output_support(),
        "shaped static",
        &[POS[0] + lc[0], POS[1] + lc[1]],
        lr + PAD,
    );
}

#[test]
fn spinning_kernel_derives_the_position_centered_swept_ball() {
    // an offset shape: rotation sweeps its bounding ball around the body
    // position, so the support is position-centered with |lc| + lr radius —
    // the radius that still covers the mask at half a turn.
    let shape = SdfExpr::<f64, 3>::cuboid([0.3, 0.1, 0.0], [0.5, 0.3, 0.2]);
    let (lc, lr) = shape.bounding_ball().expect("bounded shape");
    let lc_norm = (lc[0] * lc[0] + lc[1] * lc[1] + lc[2] * lc[2]).sqrt();
    assert!(
        lc_norm > 0.0,
        "the offset shape must have an off-center bounding ball"
    );
    let (k, _) = penalize_porous_gv_spinning(Coords::Cartesian, 2, 2, &shape, false);
    expect_ball(
        k.output_support(),
        "shaped spinning",
        &POS[..2],
        lc_norm + lr + PAD,
    );
}

#[test]
fn curvilinear_kernels_derive_no_ball() {
    // the mask ball is a cartesian region, a coordinate ball on the identity
    // chart alone, so a curvilinear chart derives Everywhere — dispatch already
    // runs the whole interior off-cartesian.
    for coords in [Coords::Cylindrical, Coords::Spherical] {
        let (k, _) = penalize_drain_gv(coords, 2, 2, &[0, 1], false);
        assert!(
            k.output_support().is_none(),
            "{coords:?}: a curvilinear kernel must not carry a cartesian ball"
        );
        let (k, _) = body_feedback_drain_gv(coords, 2, 2, &[0, 1]);
        assert!(
            k.output_support().is_none(),
            "{coords:?}: feedback ball leaked off-cartesian"
        );
    }
}
