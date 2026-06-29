// =============================================================================
// rmhd_ct_curl_cyl_rphi_divb_symbolic.rs
//
// the SYMBOLIC proof that the 2D CYLINDRICAL (r, phi) DISK constrained-transport
// curl (rmhd_ct_curl_cyl_rphi_gv) preserves the AREA-WEIGHTED div(B) = 0 EXACTLY —
// rational-function cancellation on the traced IR DAG. the structural guard for the
// disk-plane induction (the nmhd_rotor_cyl_rphi numerical test only approximates it).
//
// the 2D curl from the single out-of-plane corner EMF E (`ez` = E_z):
//   dir=0 (B_r, r-face):    dB_r/dt   = -(1/r_f) d_phi E   (1/r metric, r_f = r-face)
//   dir=1 (B_phi, phi-face): dB_phi/dt = +d_r E             (flat, NO metric)
// the point-form area-weighted divergence (cyl r-phi: (1/r)d_r(r B_r) + (1/r)d_phi B_phi):
//   div(B) = (1/(r_c dr))(r_hi B_r[+r] - r_lo B_r) + (1/(r_c dphi))(B_phi[+phi] - B_phi)
// with B = dt*curl(E), every E read cancels to the ZERO rational function. r is AFFINE; no sin.
// =============================================================================

use symbi_discretize::{rmhd_ct_curl_cyl_rphi_gv, Spacing};
use symbi_ir::proof::{LinFormR, Poly, RatFun};

const FIELDS: &[&str] = &["ez", "b"];
const SCALARS: &[&str] = &["dt", "x_lo_0", "dx_0", "x_lo_1", "dx_1"];

fn curl_only(mut lf: LinFormR) -> LinFormR {
    lf.terms.retain(|(key, _), _| key != "b");
    lf
}

fn r_at(off: i64) -> Poly {
    let mut p = Poly::var("x_lo_0").plus(&Poly::var("c_0").times(&Poly::var("dx_0")));
    if off != 0 {
        p = p.plus(&Poly::constant(off).times(&Poly::var("dx_0")));
    }
    p
}

// 2 r_c = 2 x_lo_0 + (2 c_0 + 1) dx_0.
fn two_rc() -> Poly {
    Poly::var("x_lo_0")
        .times(&Poly::constant(2))
        .plus(&Poly::var("c_0").times(&Poly::var("dx_0")).times(&Poly::constant(2)))
        .plus(&Poly::var("dx_0"))
}

fn dx(ax: usize) -> Poly {
    Poly::var(&format!("dx_{ax}"))
}

fn curl(dir: usize) -> LinFormR {
    let (kernel, writes) = rmhd_ct_curl_cyl_rphi_gv(dir, &[Spacing::Uniform; 2]);
    assert_eq!(writes.len(), 1, "curl builder must write exactly b_new");
    let lf = curl_only(LinFormR::extract_rat(&kernel.graph, writes[0].2, FIELDS, SCALARS));
    assert!(!lf.is_zero(), "dir {dir}: curl is empty — extractor saw no emf reads");
    lf
}

#[test]
fn divb_cyl_rphi_symbolic_telescoping() {
    let curl_r = curl(0); // B_r update
    let curl_phi = curl(1); // B_phi update

    // common denominator (2 r_c) * width keeps the integer-poly ring exact.
    let den_r = two_rc().times(&dx(0)); // (2 r_c) dr
    let den_phi = two_rc().times(&dx(1)); // (2 r_c) dphi
    let w_r_hi = RatFun::new(r_at(1).times(&Poly::constant(2)), den_r.clone()); // r_hi/(r_c dr)
    let w_r_lo = RatFun::new(r_at(0).times(&Poly::constant(2)), den_r);
    let w_phi = RatFun::new(Poly::constant(2), den_phi); // 1/(r_c dphi)

    // div(B) = (1/(r_c dr))(r_hi B_r[+r] - r_lo B_r) + (1/(r_c dphi))(B_phi[+phi] - B_phi).
    let mut div = LinFormR::default();
    div.add(&curl_r.shifted(&[1, 0]).scale_rat(&w_r_hi));
    div.add(&curl_r.scale_rat(&w_r_lo).neg_form());
    div.add(&curl_phi.shifted(&[0, 1]).scale_rat(&w_phi));
    div.add(&curl_phi.scale_rat(&w_phi).neg_form());

    assert!(
        div.is_zero(),
        "cyl r-phi div(curl B) != 0 symbolically — residual edge-emf numerators:\n{:#?}",
        div.residual()
    );
}

#[test]
fn divb_cyl_rphi_symbolic_detects_residual() {
    let mut lf = LinFormR::default();
    lf.add(&LinFormR::single_var(("ez".into(), vec![0, 0]), "x_lo_0"));
    lf.add(&LinFormR::single_var(("ez".into(), vec![1, 0]), "dx_0"));
    assert!(!lf.is_zero(), "mismatched coefficients must NOT cancel");
    assert_eq!(lf.residual().len(), 2);
}
