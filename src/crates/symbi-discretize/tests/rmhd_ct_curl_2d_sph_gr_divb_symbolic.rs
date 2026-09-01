// =============================================================================
// rmhd_ct_curl_2d_sph_gr_divb_symbolic.rs
//
// the symbolic proof that the 2D spherical GR constrained-transport curl
// (rmhd_ct_curl_2d_sph_gr_gv) preserves the area-weighted div(B) = 0 exactly, by
// rational-function cancellation on the traced IR DAG at graph-build time; the numeric 1e-12 evolve
// gate lives in rmhd_ct_curl_2d_sph_gr_divb.rs. closes the M10 gap for the spherical GR charts: the
// proof rests on the extractor's opaque representation of the lapse `sqrt(f)` / `sqrt(h)`.
//
// the GR densitized weight is the flat spherical area times the lapse factor:
//   schwarzschild: sqrt(gamma) = r^2 sin(theta) / sqrt(f),   f = 1 - 2M/r   (lapse in the denominator)
//   kerr-schild:   sqrt(gamma) = r^2 sin(theta) * sqrt(h),   h = 1 + 2M/r   (lapse in the numerator)
// the lapse is an opaque, radially-keyed symbol `sqrt_f@<2m>` (proof/extract.rs): identical at a
// shared radial face (so the div-weight's lapse cancels the curl's inverse lapse there — a rational
// num/den cancellation), distinct across faces, and remapped by the divergence's radial shift. once
// it cancels, the residual is the same flat r^2 sin(theta) telescoping as the minkowski proof.
//
// the 2D poloidal curl from the single densitized corner EMF Etilde (`ez`), flux form:
//   dir=0 (B_r,   r-face):    dB_r/dt  = -(1/w_r)  d_th(Etilde),  w_r  = sqrt(gamma)(r_f, th_c) dth
//   dir=1 (B_th, theta-face): dB_th/dt = +(1/w_th) d_r(Etilde),   w_th = sqrt(gamma)(r_c, th_f) dr
// weighting each curl by its face area w (the same expression the kernel divides by) collapses the
// metric — both the lapse and the r^2 sin — so the edge-EMF reads telescope to the zero rational
// function.
// =============================================================================

use symbi_discretize::{Coords, Spacetime, Spacing, rmhd_ct_curl_2d_sph_gr_gv};
use symbi_ir::proof::{LinFormR, Poly, RatFun};

const FIELDS: &[&str] = &["ez", "b"];
const SCALARS: &[&str] = &[
    "dt",
    "x_lo_0",
    "dx_0",
    "x_lo_1",
    "dx_1",
    "schwarzschild_mass",
];

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

fn sin_face(off: i64) -> Poly {
    Poly::sin_sym(2 * off)
}
fn sin_center() -> Poly {
    Poly::sin_sym(1)
}
fn dx(ax: usize) -> Poly {
    Poly::var(&format!("dx_{ax}"))
}

/// the radial lapse factor of `sqrt(gamma)` at the radial half-unit offset `two_m`. schwarzschild
/// carries `1/sqrt(f)` (lapse in the denominator); kerr-schild carries `sqrt(h)` (numerator). the
/// same opaque `sqrt_f@<2m>` atom either way — it cancels the extracted curl's reciprocal factor at
/// the same radial offset.
fn lapse(two_m: i64, in_numerator: bool) -> RatFun {
    if in_numerator {
        RatFun::new(Poly::sqrt_f_sym(two_m), Poly::constant(1))
    } else {
        RatFun::new(Poly::constant(1), Poly::sqrt_f_sym(two_m))
    }
}

fn curl(dir: usize, spacetime: Spacetime) -> LinFormR {
    let (kernel, writes) = rmhd_ct_curl_2d_sph_gr_gv(
        dir,
        spacetime,
        Coords::Spherical,
        &[Spacing::Uniform; 2],
        &[0, 1],
    );
    assert_eq!(writes.len(), 1, "curl builder must write exactly b_new");
    let lf = curl_only(LinFormR::extract_rat(
        kernel.graph(),
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

/// build the point-form area-weighted divergence of the 2D poloidal GR curl and return it. the flat
/// r^2 sin area is multiplied by the chart's radial lapse factor at the same face the curl uses:
///   r-face:     lapse at the r-face (offset 0 / +1 -> sqrt_f@0 / sqrt_f@2)
///   theta-face: lapse at the r-center (sqrt_f@1) for both theta faces (same radius, theta-shift inert)
fn div(spacetime: Spacetime, lapse_in_numerator: bool) -> LinFormR {
    let curl_r = curl(0, spacetime);
    let curl_th = curl(1, spacetime);
    let l = |two_m| lapse(two_m, lapse_in_numerator);

    let w_r_hi = RatFun::new(
        r_at(1).times(&r_at(1)).times(&sin_center()).times(&dx(1)),
        Poly::constant(1),
    )
    .mul(&l(2));
    let w_r_lo = RatFun::new(
        r_at(0).times(&r_at(0)).times(&sin_center()).times(&dx(1)),
        Poly::constant(1),
    )
    .mul(&l(0));
    let r_c_sq = r_center().mul(&r_center());
    let w_th_hi = r_c_sq
        .mul(&RatFun::new(sin_face(1).times(&dx(0)), Poly::constant(1)))
        .mul(&l(1));
    let w_th_lo = r_c_sq
        .mul(&RatFun::new(sin_face(0).times(&dx(0)), Poly::constant(1)))
        .mul(&l(1));

    let mut d = LinFormR::default();
    d.add(&curl_r.shifted(&[1, 0]).scale_rat(&w_r_hi));
    d.add(&curl_r.scale_rat(&w_r_lo).neg_form());
    d.add(&curl_th.shifted(&[0, 1]).scale_rat(&w_th_hi));
    d.add(&curl_th.scale_rat(&w_th_lo).neg_form());
    d
}

#[test]
fn divb_2d_sph_kerr_schild_symbolic_telescoping() {
    // the evolved chart: sqrt(gamma) = r^2 sin(theta) sqrt(1 + 2M/r) -> lapse in the numerator.
    let d = div(Spacetime::SchwarzschildKS, true);
    assert!(
        d.is_zero(),
        "2d sph Kerr-Schild div(curl B) != 0 symbolically — residual:\n{:#?}",
        d.residual()
    );
}

// bug-injection: pairing the low r-face weight with the lapse from the wrong radius leaves the
// sqrt_f atoms disagreeing at the shared face, so a residual survives — the proof is lapse-aware
// (schwarzschild chart; the kerr-schild control is identical up to the numerator/denominator side).
#[test]
fn divb_2d_sph_gr_symbolic_detects_wrong_lapse_offset() {
    let curl_r = curl(0, Spacetime::SchwarzschildKS);
    let curl_th = curl(1, Spacetime::SchwarzschildKS);

    let w_r_hi = RatFun::new(
        r_at(1).times(&r_at(1)).times(&sin_center()).times(&dx(1)),
        Poly::constant(1),
    )
    .mul(&lapse(2, false));
    // injected bug: the low r-face weight uses the lapse at radial offset 2; the correct low r-face offset is 0.
    let w_r_lo_bug = RatFun::new(
        r_at(0).times(&r_at(0)).times(&sin_center()).times(&dx(1)),
        Poly::constant(1),
    )
    .mul(&lapse(2, false));
    let r_c_sq = r_center().mul(&r_center());
    let w_th_hi = r_c_sq
        .mul(&RatFun::new(sin_face(1).times(&dx(0)), Poly::constant(1)))
        .mul(&lapse(1, false));
    let w_th_lo = r_c_sq
        .mul(&RatFun::new(sin_face(0).times(&dx(0)), Poly::constant(1)))
        .mul(&lapse(1, false));

    let mut d = LinFormR::default();
    d.add(&curl_r.shifted(&[1, 0]).scale_rat(&w_r_hi));
    d.add(&curl_r.scale_rat(&w_r_lo_bug).neg_form());
    d.add(&curl_th.shifted(&[0, 1]).scale_rat(&w_th_hi));
    d.add(&curl_th.scale_rat(&w_th_lo).neg_form());

    assert!(
        !d.is_zero(),
        "the wrong-lapse-offset weight must leave a nonzero residual — the proof is lapse-blind"
    );
}
