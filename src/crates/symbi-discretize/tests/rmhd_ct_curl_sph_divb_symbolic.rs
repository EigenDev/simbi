// =============================================================================
// rmhd_ct_curl_sph_divb_symbolic.rs
//
// the SYMBOLIC keystone proof that the 3D SPHERICAL constrained-transport curl
// (rmhd_ct_curl_3d_dir_gv under Coords::Spherical) preserves the AREA-WEIGHTED
// div(B) = 0 EXACTLY — by RATIONAL-FUNCTION coefficient cancellation on the traced
// IR DAG, NOT a numerical 1e-10 evolve test (rmhd_ct_curl_sph_divb.rs is that one).
//
// the curvilinear curl multiplies edge EMFs by scale-factor weights h_p (= r,
// r sin(theta)) and divides by the face-center prefactor 1/(h_p1c h_p2c) and the
// transverse widths, so its coefficients are RATIONAL FUNCTIONS, not integer
// polynomials. `symbi_ir::proof::LinFormR` carries those: r-values are AFFINE in
// {x_lo_0, dx_0, c_0} (a true polynomial — r^2 in an area must algebraically equal
// r*r in an h-product); sin(theta at offset m) is an OPAQUE symbol keyed by the
// resolved offset (sin at distinct offsets have NO polynomial relation). the cell
// coord c_N is a polynomial variable, and the divergence's coord shift is
// COVARIANT — shifting the cell by e_dir shifts the field reads AND the c_N / sin
// the coefficients depend on (the geometry at c+e_dir differs from c), mirroring
// the numerical test's absolute-index area weights.
//
// the point-form area-weighted divergence over a cell (the SAME stencil as
// rmhd_ct_curl_sph_divb.rs):
//   div(B) = sum_dir [ A_dir(+e_dir) curl_dir(+e_dir) - A_dir(+0) curl_dir(+0) ]
// each A_dir is exactly 1/inv_pref(dir) * the transverse widths; weighting the
// curl by it collapses the metric so the edge-EMF reads telescope to the ZERO
// rational function — that is the proof, for ANY input field.
// =============================================================================

use std::collections::HashMap;

use symbi_discretize::{rmhd_ct_curl_3d_dir_gv, Coords, Spacing};
use symbi_ir::proof::{LinFormR, Poly, RatFun};

// the curl reads exactly these edge-emf fields plus the in-place b; the scalars
// are dt and the geometry grid scalars (x_lo_N / dx_N, in coordinate order).
const FIELDS: &[&str] = &["e_p1", "e_p2", "b"];
const SCALARS: &[&str] =
    &["dt", "x_lo_0", "dx_0", "x_lo_1", "dx_1", "x_lo_2", "dx_2"];

// strip the old-field `b` leaf: it reproduces div(B_old), invariant under the
// update — not part of the "the update preserves div" proof.
fn curl_only(mut lf: LinFormR) -> LinFormR {
    lf.terms.retain(|(key, _), _| key != "b");
    lf
}

// map the per-dir generic emf keys to physical-axis identities. for face axis dir,
// p1=(dir+1)%3 and p2=(dir+2)%3. (the geometry coefficients are built from the
// ABSOLUTE axis scalars x_lo_N/dx_N, not per-dir id_pN — so ONLY the field keys
// need canonicalizing, unlike the cartesian proof.)
fn physical_rename(dir: usize) -> HashMap<String, String> {
    let p1 = (dir + 1) % 3;
    let p2 = (dir + 2) % 3;
    HashMap::from([
        ("e_p1".to_string(), format!("e_{p1}")),
        ("e_p2".to_string(), format!("e_{p2}")),
    ])
}

// ---- symbolic geometry primitives (rational functions in the grid vars) ----

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
    // 2 r_c = 2 x_lo_0 + (2 c_0 + 1) dx_0.
    let mut num = Poly::var("x_lo_0").times(&Poly::constant(2));
    num = num.plus(&Poly::var("c_0").times(&Poly::var("dx_0")).times(&Poly::constant(2)));
    num = num.plus(&Poly::var("dx_0"));
    RatFun::new(num, Poly::constant(2))
}

// sin(theta) at the integer face offset `off`: the opaque symbol sin_th@(2*off).
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

// the point-form face areas at the cell's LO dir-face (offset 0). these are exactly
// 1/inv_pref(dir) * the transverse widths — see rmhd_ct_curl_sph_divb.rs.
//   dir=0 (r-face):   A_r  = r(0)^2 sin(th_c) dth dph
//   dir=1 (theta-face): A_th = r_c sin(th(0)) dr dph
//   dir=2 (phi-face):  A_ph = r_c dr dth          (h_th = r, theta-free)
fn area(dir: usize) -> RatFun {
    match dir {
        0 => {
            let num = r_at(0).times(&r_at(0)).times(&sin_center()).times(&dx(1)).times(&dx(2));
            RatFun::new(num, Poly::constant(1))
        }
        1 => {
            // r_c sin(th(0)) dr dph.
            let widths = sin_face(0).times(&dx(0)).times(&dx(2));
            r_center().mul(&RatFun::new(widths, Poly::constant(1)))
        }
        _ => {
            // r_c dr dth.
            let widths = dx(0).times(&dx(1));
            r_center().mul(&RatFun::new(widths, Poly::constant(1)))
        }
    }
}

#[test]
fn divb_sph_symbolic_telescoping() {
    // accumulate the per-dir area-weighted divergence contribution; the spherical
    // area-weighted div(B)=0 holds iff the sum is the identically-zero rational
    // linear form (all numerators cancel).
    let mut div_contribution = LinFormR::default();

    for dir in 0..3usize {
        let (kernel, writes) =
            rmhd_ct_curl_3d_dir_gv(Coords::Spherical, &[Spacing::Uniform; 3], dir);
        assert_eq!(writes.len(), 1, "curl builder must write exactly b_new");
        let root = writes[0].2;

        // the dt*curl rational linear form (b stripped), keys -> physical axes.
        let raw = curl_only(LinFormR::extract_rat(&kernel.graph, root, FIELDS, SCALARS));
        let curl = raw.canonicalize_keys(&physical_rename(dir));
        assert!(!curl.is_zero(), "dir {dir}: curl is empty — extractor saw no emf reads");

        // weight the curl by the dir-face area at the cell, then form the
        // covariant forward difference along `dir`:
        //   A_dir(+e_dir) curl(+e_dir) - A_dir(+0) curl(+0).
        let weighted = curl.scale_rat(&area(dir));
        let mut e_dir = [0i32; 3];
        e_dir[dir] = 1;
        let mut diff = weighted.shifted(&e_dir);
        diff.add(&weighted.shifted(&[0, 0, 0]).neg_form());
        div_contribution.add(&diff);
    }

    // THE PROOF: the area-weighted edge-EMF contributions telescope to the zero
    // rational function (every coefficient numerator cancels EXACTLY).
    assert!(
        div_contribution.is_zero(),
        "spherical div(curl B) != 0 symbolically — residual edge-emf numerators:\n{:#?}",
        div_contribution.residual()
    );
}

// a NEGATIVE control: a single uncancelled area-weighted edge read does NOT vanish,
// so the rational checker is not vacuously green.
#[test]
fn divb_sph_symbolic_detects_residual() {
    let mut lf = LinFormR::default();
    lf.add(&LinFormR::single_var(("e_0".into(), vec![0, 0, 0]), "x_lo_0"));
    lf.add(&LinFormR::single_var(("e_0".into(), vec![1, 0, 0]), "dx_0"));
    assert!(!lf.is_zero(), "mismatched coefficients must NOT cancel");
    assert_eq!(lf.residual().len(), 2);
}
