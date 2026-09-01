// =============================================================================
// godunov_mass_conservation_sph_symbolic.rs
//
// the symbolic proof that the spherical godunov mass update conserves mass exactly,
// by the shared-face area-consistency condition
//   area_hi_0(c) == area_lo_0(c + e_r)
// the high r-face area of cell c equals the low r-face area of its outward neighbor
// c+e_r, so the shared face is single-valued, the volume-weighted flux divergence
// telescopes, and mass conserves globally. the spherical counterpart of the
// cylindrical proof — it exercises the opaque cos-symbol extraction (the solid-angle
// measure), an ingredient the spherical chart adds on top of the cylindrical one.
//
// spherical (r, theta, phi), h=(1, r, r sin theta): the r-face area is
//   area_lo_0 = r_lo^2 * Omega,   area_hi_0 = r_hi^2 * Omega
// with Omega = (cos(theta_lo) - cos(theta_hi)) * dphi the solid-angle measure and
// r_lo = x_lo_0 + c_0 dx_0, r_hi = x_lo_0 + (c_0+1) dx_0. Omega is c_0-independent
// (it lives entirely on axes 1 and 2) and is a common factor of both faces, so under
// the r-shift e_r = [1,0,0] it is untouched (the shift remaps trig symbols only along
// axis 1) and cancels identically in the equality — no pi, no Omega survives. the
// consistency reduces to r_hi(c)^2 == r_lo(c+1)^2, i.e. the squared shared-edge radius
// is single-valued: r_lo(c_0+1) = x_lo_0 + (c_0+1) dx_0 = r_hi(c_0). the cos symbols
// (cos_th@0, cos_th@2) are theta-keyed and identical on both sides — a structural
// (not numerical) cancellation.
// =============================================================================

use symbi_discretize::{Coords, Spacing, geometry_probe_gv};
use symbi_ir::proof::{RatFun, extract_scalar};

const NDIM: usize = 3;
// the geometry factors are field-free; the symbols are the grid scalars (coordinate
// order 0=r, 1=theta, 2=phi) plus the opaque cos_th@<2m> the extractor synthesizes.
const SCALARS: &[&str] = &["x_lo_0", "dx_0", "x_lo_1", "dx_1", "x_lo_2", "dx_2"];

// the r-direction unit cell shift e_r.
const E_R: [i64; NDIM] = [1, 0, 0];

// extract the dir-0 lo/hi face-area RatFuns from a fresh spherical geometry probe.
fn sph_areas() -> (RatFun, RatFun) {
    let (kernel, writes) = geometry_probe_gv(Coords::Spherical, &[Spacing::Uniform; NDIM], NDIM);
    // probe writes: 0=inv_volume, 1=area_lo_0, 2=area_hi_0, 3=centroid_0.
    let area_lo = extract_scalar(kernel.graph(), writes[1].value, SCALARS);
    let area_hi = extract_scalar(kernel.graph(), writes[2].value, SCALARS);
    (area_lo, area_hi)
}

#[test]
fn godunov_mass_conservation_sph_symbolic() {
    let (area_lo, area_hi) = sph_areas();

    // the proof: the cell's high r-face area equals its outward neighbor's low r-face
    // area. exact symbolic equality — r_hi(c)^2 Omega == r_lo(c+1)^2 Omega, with Omega
    // (the cos solid angle) cancelling structurally. => the r-flux divergence
    // telescopes => mass conserves.
    assert!(
        area_hi.equals(&area_lo.shift_coords(&E_R)),
        "spherical r-face area inconsistency: area_hi_0(c) != area_lo_0(c+e_r) — does NOT conserve"
    );
}

// negative control: the unshifted low face area differs from the high face area (r_lo^2
// vs r_hi^2 differ for a positive-width cell), so the cos-symbol checker has real content.
#[test]
fn conservation_sph_symbolic_detects_inconsistency() {
    let (area_lo, area_hi) = sph_areas();
    assert!(
        !area_hi.equals(&area_lo),
        "area_hi_0 == area_lo_0 of the SAME cell would mean a degenerate (zero-width) cell"
    );
}
