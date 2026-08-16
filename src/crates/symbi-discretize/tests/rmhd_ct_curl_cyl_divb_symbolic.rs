// =============================================================================
// rmhd_ct_curl_cyl_divb_symbolic.rs
//
// the symbolic proof that the 3D cylindrical constrained-transport curl
// (rmhd_ct_curl_3d_dir_gv under Coords::Cylindrical) preserves the area-weighted
// div(B) = 0 exactly, by rational-function coefficient cancellation on the traced
// IR DAG. the cylindrical analog of rmhd_ct_curl_sph_divb_symbolic.rs.
//
// cylindrical (r, phi, z) scale factors are h_r=1, h_phi=r, h_z=1: the radius r
// (axis 0) carries the whole curvature. the coefficients are therefore rational
// functions of the affine r alone (x_lo_0 + (c_0+off) dx_0), where the spherical case
// also needs the opaque sin(theta) — the simplest curvilinear case the `RatFun` /
// `LinFormR` ring covers. the covariant coord shift (c_0 -> c_0 + delta_0 under a +r
// step) stays essential, since the curl coefficients depend on the cell radius.
//
// the point-form area-weighted divergence over a cell (the same 1/inv_pref * widths
// rule as the spherical test):
//   dir=0 (r-face):   A_r   = r(0) dphi dz     (h_phi h_z = r * 1, r at the r-face)
//   dir=1 (phi-face): A_phi = dr dz            (h_z h_r = 1 — r-independent)
//   dir=2 (z-face):   A_z   = r_c dr dphi      (h_r h_phi = 1 * r, r at the cell center)
// weighting the curl by A_dir collapses the metric so the edge-EMF reads telescope
// to the zero rational function — the proof, and it holds for every input field.
// =============================================================================

use std::collections::HashMap;

use symbi_discretize::{Coords, Spacing, rmhd_ct_curl_3d_dir_gv};
use symbi_ir::proof::{LinFormR, Poly, RatFun};

// the curl reads exactly these edge-emf fields plus the in-place b; the scalars are
// dt and the geometry grid scalars (x_lo_N / dx_N, coordinate order: 0=r,1=phi,2=z).
const FIELDS: &[&str] = &["e_p1", "e_p2", "b"];
const SCALARS: &[&str] = &["dt", "x_lo_0", "dx_0", "x_lo_1", "dx_1", "x_lo_2", "dx_2"];

// strip the old-field `b` leaf: it reproduces div(B_old), which the update leaves
// invariant; the proof concerns the increment the update adds.
fn curl_only(mut lf: LinFormR) -> LinFormR {
    lf.terms.retain(|(key, _), _| key != "b");
    lf
}

// the cylindrical geometry is built from the absolute axis scalars x_lo_N/dx_N, so
// only the per-dir generic field keys e_p1/e_p2 need canonicalizing to physical axes.
fn physical_rename(dir: usize) -> HashMap<String, String> {
    let p1 = (dir + 1) % 3;
    let p2 = (dir + 2) % 3;
    HashMap::from([
        ("e_p1".to_string(), format!("e_{p1}")),
        ("e_p2".to_string(), format!("e_{p2}")),
    ])
}

// r at face offset `off` from the cell: x_lo_0 + (c_0 + off) dx_0.
fn r_at(off: i64) -> Poly {
    let mut p = Poly::var("x_lo_0");
    p = p.plus(&Poly::var("c_0").times(&Poly::var("dx_0")));
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

fn dx(ax: usize) -> Poly {
    Poly::var(&format!("dx_{ax}"))
}

// the point-form face areas at the cell's low dir-face (offset 0).
fn area(dir: usize) -> RatFun {
    match dir {
        // r-face: r(0) dphi dz. h_phi (= r) at the r-face, h_z = 1.
        0 => RatFun::new(r_at(0).times(&dx(1)).times(&dx(2)), Poly::constant(1)),
        // phi-face: dr dz. h_z h_r = 1 — r-independent.
        1 => RatFun::new(dx(0).times(&dx(2)), Poly::constant(1)),
        // z-face: r_c dr dphi. h_phi (= r) at the cell center, h_r = 1.
        _ => r_center().mul(&RatFun::new(dx(0).times(&dx(1)), Poly::constant(1))),
    }
}

#[test]
fn divb_cyl_symbolic_telescoping() {
    // accumulate the per-dir area-weighted divergence contribution; the cylindrical
    // area-weighted div(B)=0 holds iff the sum is the identically-zero rational linear
    // form (all numerators cancel).
    let mut div_contribution = LinFormR::default();

    for dir in 0..3usize {
        let (kernel, writes) =
            rmhd_ct_curl_3d_dir_gv(Coords::Cylindrical, &[Spacing::Uniform; 3], dir);
        assert_eq!(writes.len(), 1, "curl builder must write exactly b_new");
        let root = writes[0].2;

        let raw = curl_only(LinFormR::extract_rat(&kernel.graph, root, FIELDS, SCALARS));
        let curl = raw.canonicalize_keys(&physical_rename(dir));
        assert!(
            !curl.is_zero(),
            "dir {dir}: curl is empty — extractor saw no emf reads"
        );

        // weight by the dir-face area, then the covariant forward difference along dir.
        let weighted = curl.scale_rat(&area(dir));
        let mut e_dir = [0i32; 3];
        e_dir[dir] = 1;
        let mut diff = weighted.shifted(&e_dir);
        diff.add(&weighted.shifted(&[0, 0, 0]).neg_form());
        div_contribution.add(&diff);
    }

    // the proof: the area-weighted edge-EMF contributions telescope to the zero
    // rational function; every coefficient numerator cancels exactly.
    assert!(
        div_contribution.is_zero(),
        "cylindrical div(curl B) != 0 symbolically — residual edge-emf numerators:\n{:#?}",
        div_contribution.residual()
    );
}

// a negative control: a single uncancelled area-weighted edge read survives as a
// residual, so the rational checker reports a broken cancellation when one exists.
#[test]
fn divb_cyl_symbolic_detects_residual() {
    let mut lf = LinFormR::default();
    lf.add(&LinFormR::single_var(
        ("e_0".into(), vec![0, 0, 0]),
        "x_lo_0",
    ));
    lf.add(&LinFormR::single_var(("e_0".into(), vec![1, 0, 0]), "dx_0"));
    assert!(!lf.is_zero(), "mismatched coefficients must NOT cancel");
    assert_eq!(lf.residual().len(), 2);
}
