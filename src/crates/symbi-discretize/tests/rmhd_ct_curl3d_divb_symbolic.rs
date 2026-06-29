// =============================================================================
// rmhd_ct_curl3d_divb_symbolic.rs
//
// the SYMBOLIC keystone proof that the cartesian 3D constrained-transport curl
// preserves div(B) = 0 — by polynomial-coefficient cancellation on the traced IR
// DAG, NOT a numerical 1e-12 evolve test (rmhd_ct_curl3d_divb.rs is that one).
//
// for each face axis `dir`, `rmhd_ct_curl_3d_dir_gv` traces `b_new = b + dt*curl`
// whose leaves are edge-emf field reads at KNOWN integer stencil offsets.
// `symbi_ir::proof::LinForm` extracts the exact symbolic linear combination of
// those reads. the divergence stencil over a cell is
//   d(div B)/dt = sum_dir id_dir * (curl[dir][+e_dir] - curl[dir][+0]),
// applied symbolically by SHIFTING each curl LinForm's offsets. the edge-emf
// contributions must cancel to the ZERO linear form for div(B) to vanish by
// construction, for ANY input field — that is the proof.
//
// canonicalization: the per-dir builder reuses the GENERIC keys e_p1/e_p2 and
// scalars id_p1/id_p2 for all three dirs, but they denote different PHYSICAL emf
// components / inverse widths (p1=(dir+1)%3, p2=(dir+2)%3). the runtime dispatch
// binds the real buffers positionally; the symbolic proof must rename them to
// their physical-axis identity (e_<axis>, id_<axis>) before the three dirs can
// telescope. the b (old-field) leaf is stripped — it reproduces div(B_old), not
// part of the UPDATE's preservation property.
// =============================================================================

use std::collections::HashMap;

use symbi_discretize::{rmhd_ct_curl_3d_dir_gv, Coords, Spacing};
use symbi_ir::proof::{LinForm, Poly};

// the curl reads exactly these edge-emf fields plus the in-place b; scalars are
// dt and the per-dir generic cartesian inverse widths.
const FIELDS: &[&str] = &["e_p1", "e_p2", "b"];
const SCALARS: &[&str] = &["dt", "id_p1", "id_p2"];

// strip the old-field `b` leaf: it contributes div(B_old), invariant under the
// update, so it is not part of the "the update preserves div" proof.
fn curl_only(mut lf: LinForm) -> LinForm {
    lf.terms.retain(|(key, _), _| key != "b");
    lf
}

// map the per-dir generic names to physical-axis identities. for face axis dir,
// p1=(dir+1)%3 and p2=(dir+2)%3, so the generic e_p1/id_p1 ARE the physical
// component p1, e_p2/id_p2 the component p2.
fn physical_rename(dir: usize) -> HashMap<String, String> {
    let p1 = (dir + 1) % 3;
    let p2 = (dir + 2) % 3;
    HashMap::from([
        ("e_p1".to_string(), format!("e_{p1}")),
        ("e_p2".to_string(), format!("e_{p2}")),
        ("id_p1".to_string(), format!("id_{p1}")),
        ("id_p2".to_string(), format!("id_{p2}")),
    ])
}

#[test]
fn divb_symbolic_telescoping() {
    // accumulate the per-dir divergence contribution; div(B)=0 holds iff the sum
    // is the identically-zero linear form.
    let mut div_contribution = LinForm::default();

    for dir in 0..3usize {
        let (kernel, writes) =
            rmhd_ct_curl_3d_dir_gv(Coords::Cartesian, &[Spacing::Uniform; 3], dir);
        assert_eq!(writes.len(), 1, "curl builder must write exactly b_new");
        let root = writes[0].2;

        // the dt*curl linear form (b stripped), canonicalized to physical axes.
        let raw = curl_only(LinForm::extract(&kernel.graph, root, FIELDS, SCALARS));
        let curl = raw.canonicalize(&physical_rename(dir));
        assert!(!curl.is_zero(), "dir {dir}: curl is empty — extractor saw no emf reads");

        // the divergence's id_dir-weighted forward difference along `dir`:
        //   id_dir * (curl[+e_dir] - curl[+0]).
        let id_div = format!("id_{dir}");
        let mut e_dir = [0i32; 3];
        e_dir[dir] = 1;
        let mut diff = curl.shifted(&e_dir);
        diff.add(&curl.shifted(&[0, 0, 0]).neg_form());
        div_contribution.add(&diff.scale_var(&id_div));
    }

    // THE PROOF: the edge-emf contributions telescope to the zero polynomial.
    assert!(
        div_contribution.is_zero(),
        "div(curl B) != 0 symbolically — residual edge-emf coefficients:\n{:#?}",
        div_contribution.residual()
    );
}

// a NEGATIVE control: mismatched coefficients do NOT cancel, so the checker is
// not vacuously green — exactly the residual a sign bug in the curl would leave.
#[test]
fn divb_symbolic_detects_broken_telescoping() {
    let mut lf = LinForm::default();
    lf.add(&LinForm::single(("e_0".into(), vec![0, 0, 0]), Poly::var("id_0")));
    lf.add(&LinForm::single(("e_0".into(), vec![1, 0, 0]), Poly::var("id_1")));
    assert!(!lf.is_zero(), "mismatched coefficients must NOT cancel");
    assert_eq!(lf.residual().len(), 2);
}
