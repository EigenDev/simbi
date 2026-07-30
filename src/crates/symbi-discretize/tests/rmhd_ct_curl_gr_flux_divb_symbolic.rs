// =============================================================================
// rmhd_ct_curl_gr_flux_divb_symbolic.rs
//
// the SYMBOLIC proof that the NON-AFFINE-radius GR constrained-transport curls preserve the
// area-weighted div(B) = 0 EXACTLY — the charts whose lapse `sqrt(gamma)` cannot be keyed by a simple
// radial offset: cartesian kerr-schild (r = sqrt(x^2 + y^2)), cylindrical r-z kerr-schild
// (r = sqrt(R^2 + z^2)), and spinning kerr (Sigma = r^2 + a^2 cos^2 theta). the extractor keys these
// metric factors by their exact CANONICAL argument form (proof/extract.rs), so identical factors at
// the same face share the atom.
//
// EXTRACT-THE-WEIGHT method. the GR curls are the FLUX form
//   b_new = b -+ dt (Etilde[+corner] - Etilde) / w,   w = sqrt(gamma)(face) * dtransverse
// so the edge-emf coefficient on the near corner `ez[0,0]` is exactly `-+ dt / w`. the proof RECOVERS the
// face area `w = dt / coeff(ez[0,0])` straight from the traced curl and scale it back out — the
// (possibly nested / transcendental) metric atom cancels LOCALLY (num/den, same face), leaving the
// metric-free bare EMF difference `-+ dt (ez[+corner] - ez)`. the area-weighted divergence of the
// bare differences telescopes to the ZERO rational function (the corner reads cancel), for ANY field.
// no offset key, no shift-remap, no hand-built weight — the weight comes from the kernel itself, so
// the proof binds whatever `sqrt(gamma)` the metric actually produced.
//
// cancellation is exact even WITHOUT gcd reduction: `is_zero` cross-multiplies, so an unreduced
// `(dt w)/w` compares equal to `dt` and the shared corner reads across directions cancel.
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
    "kerr_spin",
];

fn curl_only(mut lf: LinFormR) -> LinFormR {
    lf.terms.retain(|(key, _), _| key != "b");
    lf
}

/// the metric-free "bare" EMF difference for face-axis `dir`: extract the flux-form curl, read the
/// face area `w = dt / coeff(ez[0,0])` off it, and scale the metric back out.
fn bare(dir: usize, spacetime: Spacetime, coords: Coords, axes: &[usize]) -> LinFormR {
    let (kernel, writes) =
        rmhd_ct_curl_2d_sph_gr_gv(dir, spacetime, coords, &[Spacing::Uniform; 2], axes);
    assert_eq!(writes.len(), 1, "curl builder must write exactly b_new");
    let curl = curl_only(LinFormR::extract_rat(
        &kernel.graph,
        writes[0].2,
        FIELDS,
        SCALARS,
    ));
    let c00 = curl
        .terms
        .get(&("ez".to_string(), vec![0, 0]))
        .expect("the flux-form curl must read the near corner ez[0,0]");
    // w = dt / c00 (c00 = -+ dt/w); scaling the curl by it cancels the metric atom in every coeff.
    let w = RatFun::new(Poly::var("dt"), Poly::constant(1)).mul(&c00.reciprocal());
    curl.scale_rat(&w)
}

/// the point-form area-weighted divergence of the 2D poloidal GR curl. dir 0 is `b - ..`, dir 1 is
/// `b + ..`, so extracting `w = dt/c00` yields `+bare` for dir 0 and `-bare` for dir 1 — hence the
/// dir-1 difference enters SUBTRACTED, recovering the physical `+area` weighting of both faces.
fn div(spacetime: Spacetime, coords: Coords, axes: &[usize]) -> LinFormR {
    let q0 = bare(0, spacetime, coords, axes);
    let q1 = bare(1, spacetime, coords, axes);
    let mut d = LinFormR::default();
    d.add(&q0.shifted(&[1, 0]));
    d.add(&q0.neg_form());
    d.add(&q1.shifted(&[0, 1]).neg_form());
    d.add(&q1);
    d
}

#[test]
fn divb_cartesian_kerr_schild_symbolic() {
    // r = sqrt(x^2 + y^2): a NESTED sqrt in sqrt(gamma) = sqrt(1 + 2M/r).
    let d = div(Spacetime::SchwarzschildKS, Coords::Cartesian, &[0, 1]);
    assert!(
        d.is_zero(),
        "cartesian KS div(curl B) != 0:\n{:#?}",
        d.residual()
    );
}

#[test]
fn divb_cylindrical_rphi_kerr_schild_symbolic() {
    // the (R, phi) equatorial disk: r = R on the equator, so this chart is diagonal + affine, but the
    // extract-the-weight method binds it uniformly with the rest.
    let d = div(Spacetime::SchwarzschildKS, Coords::Cylindrical, &[0, 1]);
    assert!(
        d.is_zero(),
        "cylindrical r-phi KS div(curl B) != 0:\n{:#?}",
        d.residual()
    );
}

#[test]
fn divb_cylindrical_rz_kerr_schild_symbolic() {
    // the (R, z) poloidal chart: r = sqrt(R^2 + z^2), NON-diagonal spatial metric (gamma_Rz), nested
    // sqrt in the measure. axes = [R=0, z=2].
    let d = div(Spacetime::SchwarzschildKS, Coords::Cylindrical, &[0, 2]);
    assert!(
        d.is_zero(),
        "cylindrical r-z KS div(curl B) != 0:\n{:#?}",
        d.residual()
    );
}

#[test]
fn divb_spherical_kerr_symbolic() {
    // spinning kerr (r, theta): sqrt(gamma) = Sigma sin(theta) sqrt(1 + b), Sigma = r^2 + a^2 cos^2,
    // b = 2Mr/Sigma — non-affine radius AND a transcendental (cos) argument.
    let d = div(Spacetime::KerrKS, Coords::Spherical, &[0, 1]);
    assert!(
        d.is_zero(),
        "spherical Kerr div(curl B) != 0:\n{:#?}",
        d.residual()
    );
}

// negative control: dropping the weight (using the raw curl, metric NOT cancelled) leaves the lapse
// atoms uncancelled across the two directions, so the shared corner reads do NOT telescope to zero.
#[test]
fn divb_gr_flux_symbolic_detects_missing_weight() {
    let raw = |dir: usize| -> LinFormR {
        let (kernel, writes) = rmhd_ct_curl_2d_sph_gr_gv(
            dir,
            Spacetime::SchwarzschildKS,
            Coords::Cartesian,
            &[Spacing::Uniform; 2],
            &[0, 1],
        );
        curl_only(LinFormR::extract_rat(
            &kernel.graph,
            writes[0].2,
            FIELDS,
            SCALARS,
        ))
    };
    let q0 = raw(0);
    let q1 = raw(1);
    let mut d = LinFormR::default();
    d.add(&q0.shifted(&[1, 0]));
    d.add(&q0.neg_form());
    d.add(&q1.shifted(&[0, 1]).neg_form());
    d.add(&q1);
    assert!(
        !d.is_zero(),
        "without cancelling the face-area weight the metric atoms must NOT telescope away"
    );
}
