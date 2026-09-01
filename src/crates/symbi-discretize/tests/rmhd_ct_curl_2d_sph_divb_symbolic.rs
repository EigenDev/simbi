// =============================================================================
// rmhd_ct_curl_2d_sph_divb_symbolic.rs
//
// the symbolic proof that the 2D spherical poloidal (r, theta) constrained-transport
// curl (rmhd_ct_curl_2d_sph_gv) preserves the area-weighted div(B) = 0 exactly, by
// rational-function cancellation on the traced IR DAG. the structural counterpart to
// the numerical rmhd_ct_curl_2d_sph_poloidal_divb gate. the heaviest 2D case: the coeff
// ring needs the affine r together with the opaque sin(theta) symbols.
//
// the 2D curl from the single out-of-plane corner EMF E (`ez` = E_phi):
//   dir=0 (B_r,   r-face):    dB_r/dt  = -(1/(r_f sin th_c)) d_th(sin th E)
//   dir=1 (B_th, theta-face): dB_th/dt = +(1/r_c) d_r(r E)
// the point-form area-weighted divergence (sph poloidal: (1/r^2)d_r(r^2 B_r) +
// (1/(r sin th))d_th(sin th B_th)), the rmhd_ct_curl_2d_sph_poloidal_divb form:
//   div(B) = (r_hi^2 sin_c B_r[+r] - r_lo^2 sin_c B_r) dth          [/(r^2) folded into area]
//          + (r_c sin_hi B_th[+th] - r_c sin_lo B_th) dr
// with B = dt*curl(E), every E read cancels to the zero rational function. r is affine,
// sin(theta@offset) opaque (half-unit keys: sin_c = sin@1, sin-faces = sin@0/sin@2).
// =============================================================================

use symbi_discretize::{Spacing, rmhd_ct_curl_2d_sph_gv};
use symbi_ir::proof::{LinFormR, Poly, RatFun};

const FIELDS: &[&str] = &["ez", "b"];
const SCALARS: &[&str] = &["dt", "x_lo_0", "dx_0", "x_lo_1", "dx_1"];

fn curl_only(mut lf: LinFormR) -> LinFormR {
    lf.terms.retain(|(key, _), _| key != "b");
    lf
}

// r at r-face offset `off`: x_lo_0 + (c_0 + off) dx_0.
fn r_at(off: i64) -> Poly {
    let mut p = Poly::var("x_lo_0").plus(&Poly::var("c_0").times(&Poly::var("dx_0")));
    if off != 0 {
        p = p.plus(&Poly::constant(off).times(&Poly::var("dx_0")));
    }
    p
}

// r_c, the cell-center radius x_lo_0 + (c_0 + 1/2) dx_0, as a RatFun (denominator 2).
fn r_center() -> RatFun {
    let mut num = Poly::var("x_lo_0").times(&Poly::constant(2));
    num = num.plus(
        &Poly::var("c_0")
            .times(&Poly::var("dx_0"))
            .times(&Poly::constant(2)),
    );
    num = num.plus(&Poly::var("dx_0"));
    RatFun::new(num, Poly::constant(2))
}

// sin(theta) at the integer theta-face offset `off`: the opaque symbol sin_th@(2*off).
fn sin_face(off: i64) -> Poly {
    Poly::sin_sym(2 * off)
}
// sin(theta) at the cell-center (offset +1/2): the opaque symbol sin_th@1.
fn sin_center() -> Poly {
    Poly::sin_sym(1)
}

fn dx(ax: usize) -> Poly {
    Poly::var(&format!("dx_{ax}"))
}

fn curl(dir: usize) -> LinFormR {
    let (kernel, writes) = rmhd_ct_curl_2d_sph_gv(dir, &[Spacing::Uniform; 2]);
    assert_eq!(writes.len(), 1, "curl builder must write exactly b_new");
    let lf = curl_only(LinFormR::extract_rat(
        &kernel.graph,
        writes[0].value,
        FIELDS,
        SCALARS,
    ));
    assert!(
        !lf.is_zero(),
        "dir {dir}: curl is empty — extractor saw no emf reads"
    );
    lf
}

#[test]
fn divb_2d_sph_symbolic_telescoping() {
    let curl_r = curl(0); // B_r update
    let curl_th = curl(1); // B_theta update

    // r-face area weights: r_f^2 sin_c dth (sin at cell-center, r at the r-face).
    let w_r_hi = RatFun::new(
        r_at(1).times(&r_at(1)).times(&sin_center()).times(&dx(1)),
        Poly::constant(1),
    );
    let w_r_lo = RatFun::new(
        r_at(0).times(&r_at(0)).times(&sin_center()).times(&dx(1)),
        Poly::constant(1),
    );
    // theta-face area weights: r_c sin_f dr (r at cell-center, sin at the theta-face).
    let w_th_hi = r_center().mul(&RatFun::new(sin_face(1).times(&dx(0)), Poly::constant(1)));
    let w_th_lo = r_center().mul(&RatFun::new(sin_face(0).times(&dx(0)), Poly::constant(1)));

    // div(B) = (area_r(+r) B_r[+r] - area_r B_r) + (area_th(+th) B_th[+th] - area_th B_th).
    let mut div = LinFormR::default();
    div.add(&curl_r.shifted(&[1, 0]).scale_rat(&w_r_hi));
    div.add(&curl_r.scale_rat(&w_r_lo).neg_form());
    div.add(&curl_th.shifted(&[0, 1]).scale_rat(&w_th_hi));
    div.add(&curl_th.scale_rat(&w_th_lo).neg_form());

    assert!(
        div.is_zero(),
        "2d spherical poloidal div(curl B) != 0 symbolically — residual edge-emf numerators:\n{:#?}",
        div.residual()
    );
}

#[test]
fn divb_2d_sph_symbolic_detects_residual() {
    let mut lf = LinFormR::default();
    lf.add(&LinFormR::single_var(("ez".into(), vec![0, 0]), "x_lo_0"));
    lf.add(&LinFormR::single_var(("ez".into(), vec![1, 0]), "dx_0"));
    assert!(!lf.is_zero(), "mismatched coefficients must NOT cancel");
    assert_eq!(lf.residual().len(), 2);
}
