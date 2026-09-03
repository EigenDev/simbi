// =============================================================================
// rmhd_ct_curl_cyl_rz_divb_symbolic.rs
//
// the symbolic proof that the 2D cylindrical (r, z) constrained-transport curl
// (rmhd_ct_curl_cyl_rz_gv) preserves the area-weighted div(B) = 0 exactly, by
// rational-function coefficient cancellation on the traced IR DAG. the structural
// counterpart to the numerical nmhd_rotor_cyl_rz gate.
//
// this is the curl that carried a sign bug: dir=1 / B_z used +d_r(r E) where the correct
// term is -d_r(r E); div(B) blew to O(1) in one step. this proof would have caught it: the
// telescoping leaves the E reads at 2x where they should cancel to zero. it is the instant,
// always-true guard the numerical rotor test only approximates.
//
// the 2D curl from the single out-of-plane corner EMF E (`ez`):
//   dir=0 (B_r, r-face):  dB_r/dt = +d_z E              (flat: h_z = 1)
//   dir=1 (B_z, z-face):  dB_z/dt = -(1/r_c) d_r(r E)   (cylindrical radial metric)
// the point-form area-weighted divergence over a cell (the nmhd_rotor_cyl_rz form):
//   div(B) = (1/(r_c dr)) (r_hi B_r[+r] - r_lo B_r) + (1/dz) (B_z[+z] - B_z)
// with B = dt*curl(E). substituting curl, every E read cancels to the zero rational
// function — for every input field. r is affine (x_lo_0 + (c_0+off) dx_0), so r symbols
// alone span the coefficient ring.
// =============================================================================

use symbi_discretize::{Spacing, rmhd_ct_curl_cyl_rz_gv};
use symbi_ir::SweepAxis;
use symbi_ir::proof::{LinFormR, Poly, RatFun};

// fields: the single out-of-plane corner EMF `ez` + the in-place `b`. scalars: dt + the
// 2D grid geometry (axis 0 = r, axis 1 = z).
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

// 2 r_c (cell-center radius x2, an integer-coeff Poly): 2 x_lo_0 + (2 c_0 + 1) dx_0.
fn two_rc() -> Poly {
    Poly::var("x_lo_0")
        .times(&Poly::constant(2))
        .plus(
            &Poly::var("c_0")
                .times(&Poly::var("dx_0"))
                .times(&Poly::constant(2)),
        )
        .plus(&Poly::var("dx_0"))
}

fn dx(ax: usize) -> Poly {
    Poly::var(&format!("dx_{ax}"))
}

// extract the b-stripped dt*curl rational linear form for one face axis.
fn curl(dir: usize) -> LinFormR {
    let program =
        rmhd_ct_curl_cyl_rz_gv(SweepAxis::new(dir, 2), &[Spacing::Uniform; 2]);
    let kernel = program.kernel();
    let writes = program.writes();
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

#[test]
fn divb_cyl_rz_symbolic_telescoping() {
    let curl_r = curl(0); // B_r update
    let curl_z = curl(1); // B_z update

    // radial term weights: r_hi/(r_c dr) on B_r[+r], r_lo/(r_c dr) on B_r. written over the
    // common denominator (2 r_c) dr so the integer-poly ring stays exact (factor of 2 from r_c).
    let den_r = two_rc().times(&dx(0));
    let w_r_hi = RatFun::new(r_at(1).times(&Poly::constant(2)), den_r.clone());
    let w_r_lo = RatFun::new(r_at(0).times(&Poly::constant(2)), den_r);
    // axial term weight: 1/dz (flat, no metric).
    let w_z = RatFun::new(Poly::constant(1), dx(1));

    // div(B) = (1/(r_c dr))(r_hi B_r[+r] - r_lo B_r) + (1/dz)(B_z[+z] - B_z).
    let mut div = LinFormR::default();
    div.add(&curl_r.shifted(&[1, 0]).scale_rat(&w_r_hi));
    div.add(&curl_r.scale_rat(&w_r_lo).neg_form());
    div.add(&curl_z.shifted(&[0, 1]).scale_rat(&w_z));
    div.add(&curl_z.scale_rat(&w_z).neg_form());

    // the proof: the area-weighted edge-EMF contributions telescope to zero.
    assert!(
        div.is_zero(),
        "cyl r-z div(curl B) != 0 symbolically — residual edge-emf numerators:\n{:#?}",
        div.residual()
    );
}

// negative control: a mismatched pair leaves a residual, so the checker reports a broken
// cancellation.
#[test]
fn divb_cyl_rz_symbolic_detects_residual() {
    let mut lf = LinFormR::default();
    lf.add(&LinFormR::single_var(("ez".into(), vec![0, 0]), "x_lo_0"));
    lf.add(&LinFormR::single_var(("ez".into(), vec![1, 0]), "dx_0"));
    assert!(!lf.is_zero(), "mismatched coefficients must NOT cancel");
    assert_eq!(lf.residual().len(), 2);
}
