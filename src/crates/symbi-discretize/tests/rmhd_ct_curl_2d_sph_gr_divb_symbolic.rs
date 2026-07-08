// =============================================================================
// rmhd_ct_curl_2d_sph_gr_divb_symbolic.rs
//
// the SYMBOLIC proof that the 2D SPHERICAL GR (Schwarzschild) constrained-transport curl
// (rmhd_ct_curl_2d_sph_gr_gv) preserves the AREA-WEIGHTED div(B) = 0 EXACTLY — by
// rational-function cancellation on the traced IR DAG, NOT the numeric 1e-12 evolve gate
// (rmhd_ct_curl_2d_sph_gr_divb.rs is that one). this closes the M10 gap: the GR curls had NO
// symbolic proof because the extractor could not represent the lapse `sqrt(f)`.
//
// the GR densitized weight is the FLAT spherical area divided by the lapse:
//   sqrt(gamma)_Schw = r^2 sin(theta) / sqrt(f),   f = 1 - 2M/r
// the `sqrt(f)` is an OPAQUE, radially-keyed symbol `sqrt_f@<2m>` (proof/extract.rs): identical at
// a shared radial face (so the div-weight's 1/sqrt(f) cancels the curl's sqrt(f) there — a rational
// num/den cancellation), distinct across faces, and REMAPPED by the divergence's radial shift. once
// it cancels, the residual is the SAME flat r^2 sin(theta) telescoping as the Minkowski proof.
//
// the 2D poloidal curl from the single densitized corner EMF Etilde (`ez`):
//   dir=0 (B_r,   r-face):    dB_r/dt  = -(1/w_r)  d_th(Etilde),  w_r  = sqrt(gamma)(r_f, th_c) dth
//   dir=1 (B_th, theta-face): dB_th/dt = +(1/w_th) d_r(Etilde),   w_th = sqrt(gamma)(r_c, th_f) dr
// weighting each curl by its face area w (= the SAME expression) collapses the metric — the lapse
// AND the r^2 sin — so the edge-EMF reads telescope to the ZERO rational function.
// =============================================================================

use symbi_discretize::{rmhd_ct_curl_2d_sph_gr_gv, Coords, Spacetime, Spacing};
use symbi_ir::proof::{LinFormR, Poly, RatFun};

const FIELDS: &[&str] = &["ez", "b"];
// the GR curl adds the lapse mass parameter `schwarzschild_mass` to the flat grid scalars.
const SCALARS: &[&str] =
    &["dt", "x_lo_0", "dx_0", "x_lo_1", "dx_1", "schwarzschild_mass"];

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
    num = num.plus(&Poly::var("c_0").times(&Poly::var("dx_0")).times(&Poly::constant(2)));
    num = num.plus(&Poly::var("dx_0"));
    RatFun::new(num, Poly::constant(2))
}

fn sin_face(off: i64) -> Poly {
    Poly::sin_sym(2 * off)
}
fn sin_center() -> Poly {
    Poly::sin_sym(1)
}
fn dx(ax: usize) -> Poly {
    Poly::var(&format!("dx_{ax}"))
}

// 1/sqrt(f) at the radial half-unit offset `two_m` — the reciprocal lapse factor the densitized
// GR area carries; it cancels the extracted curl's sqrt(f) at the SAME radial offset.
fn inv_sqrt_f(two_m: i64) -> RatFun {
    RatFun::new(Poly::constant(1), Poly::sqrt_f_sym(two_m))
}

fn curl(dir: usize) -> LinFormR {
    let (kernel, writes) = rmhd_ct_curl_2d_sph_gr_gv(
        dir,
        Spacetime::Schwarzschild,
        Coords::Spherical,
        &[Spacing::Uniform; 2],
        &[0, 1],
    );
    assert_eq!(writes.len(), 1, "curl builder must write exactly b_new");
    let lf = curl_only(LinFormR::extract_rat(&kernel.graph, writes[0].2, FIELDS, SCALARS));
    assert!(!lf.is_zero(), "dir {dir}: curl is empty — extractor saw no emf reads");
    lf
}

#[test]
fn divb_2d_sph_gr_symbolic_telescoping() {
    let curl_r = curl(0); // B_r update
    let curl_th = curl(1); // B_theta update

    // r-face area = r_f^2 sin_c dth / sqrt(f(r_f)): flat area / lapse at the r-FACE (offset 0 / +1,
    // so sqrt_f@0 / sqrt_f@2). the reciprocal-lapse factor cancels the curl's sqrt(f) at that face.
    let w_r_hi = RatFun::new(
        r_at(1).times(&r_at(1)).times(&sin_center()).times(&dx(1)),
        Poly::constant(1),
    )
    .mul(&inv_sqrt_f(2));
    let w_r_lo = RatFun::new(
        r_at(0).times(&r_at(0)).times(&sin_center()).times(&dx(1)),
        Poly::constant(1),
    )
    .mul(&inv_sqrt_f(0));
    // theta-face area = r_c^2 sin_f dr / sqrt(f(r_c)) = the FULL sqrt(gamma)(r_c, th_f) dr (the GR
    // curl is the flux form (1/area) d(Etilde), so the weight is the whole densitized area, r^2 sin
    // NOT the flat scale-factor r). the lapse is at the r-CENTER (offset +1/2 -> sqrt_f@1) for BOTH
    // theta faces (same radial position), untouched by the theta shift.
    let r_c_sq = r_center().mul(&r_center());
    let w_th_hi = r_c_sq
        .mul(&RatFun::new(sin_face(1).times(&dx(0)), Poly::constant(1)))
        .mul(&inv_sqrt_f(1));
    let w_th_lo = r_c_sq
        .mul(&RatFun::new(sin_face(0).times(&dx(0)), Poly::constant(1)))
        .mul(&inv_sqrt_f(1));

    // div(B) = (area_r(+r) B_r[+r] - area_r B_r) + (area_th(+th) B_th[+th] - area_th B_th).
    let mut div = LinFormR::default();
    div.add(&curl_r.shifted(&[1, 0]).scale_rat(&w_r_hi));
    div.add(&curl_r.scale_rat(&w_r_lo).neg_form());
    div.add(&curl_th.shifted(&[0, 1]).scale_rat(&w_th_hi));
    div.add(&curl_th.scale_rat(&w_th_lo).neg_form());

    assert!(
        div.is_zero(),
        "2d spherical GR (Schwarzschild) div(curl B) != 0 symbolically — residual edge-emf \
         numerators (a nonzero here means the lapse or the r^2 sin weight did not cancel):\n{:#?}",
        div.residual()
    );
}

// bug-injection: a WRONG lapse offset (using the reciprocal lapse at the r-face +1 for the LOW
// r-face weight, i.e. sqrt_f@2 where sqrt_f@0 belongs) must NOT cancel — the sqrt_f atoms no longer
// match at the shared face, so the residual survives. proves the proof is sensitive to the lapse.
#[test]
fn divb_2d_sph_gr_symbolic_detects_wrong_lapse_offset() {
    let curl_r = curl(0);
    let curl_th = curl(1);

    let w_r_hi = RatFun::new(
        r_at(1).times(&r_at(1)).times(&sin_center()).times(&dx(1)),
        Poly::constant(1),
    )
    .mul(&inv_sqrt_f(2));
    // INJECTED BUG: the low r-face weight uses the lapse at the WRONG radial offset (2 instead of 0).
    let w_r_lo_bug = RatFun::new(
        r_at(0).times(&r_at(0)).times(&sin_center()).times(&dx(1)),
        Poly::constant(1),
    )
    .mul(&inv_sqrt_f(2));
    let r_c_sq = r_center().mul(&r_center());
    let w_th_hi = r_c_sq
        .mul(&RatFun::new(sin_face(1).times(&dx(0)), Poly::constant(1)))
        .mul(&inv_sqrt_f(1));
    let w_th_lo = r_c_sq
        .mul(&RatFun::new(sin_face(0).times(&dx(0)), Poly::constant(1)))
        .mul(&inv_sqrt_f(1));

    let mut div = LinFormR::default();
    div.add(&curl_r.shifted(&[1, 0]).scale_rat(&w_r_hi));
    div.add(&curl_r.scale_rat(&w_r_lo_bug).neg_form());
    div.add(&curl_th.shifted(&[0, 1]).scale_rat(&w_th_hi));
    div.add(&curl_th.scale_rat(&w_th_lo).neg_form());

    assert!(
        !div.is_zero(),
        "the wrong-lapse-offset weight must leave a nonzero residual — the proof is lapse-blind"
    );
}
