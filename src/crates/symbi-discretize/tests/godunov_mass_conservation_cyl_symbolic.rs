// =============================================================================
// godunov_mass_conservation_cyl_symbolic.rs
//
// the symbolic proof that the cylindrical godunov mass update conserves mass
// exactly, by the shared-face area-consistency condition
//   area_hi_0(c) == area_lo_0(c + e_r)
// the high r-face area of cell c equals the low r-face area of its outward
// neighbor c+e_r. the face is single-valued, so the volume-weighted flux
// divergence is a discrete divergence that telescopes globally (interior faces
// cancel equal-and-opposite, only the domain boundary survives). this is the
// curvilinear analog of the cartesian godunov_mass_conservation_symbolic.rs (which
// proves directly that the flat flux update is a discrete divergence).
//
// cylindrical (r, phi, z), h=(1,r,1): the r-face area is
//   area_lo_0 = r_lo * dphi * dz,   area_hi_0 = r_hi * dphi * dz
// with r_lo = x_lo_0 + c_0 dx_0, r_hi = x_lo_0 + (c_0+1) dx_0. a pure polynomial in
// the affine r and the transverse widths, free of transcendentals (h_phi=r is the only
// curvature, and it is the radial coordinate itself, already a poly var). the
// consistency reduces to r_hi(c) == r_lo(c+1), which holds because the cell-edge
// radius is single-valued: r_lo(c_0+1) = x_lo_0 + (c_0+1) dx_0 = r_hi(c_0). the
// transverse factor dphi*dz is c_0-independent, so the r-shift leaves it unchanged.
// =============================================================================

use symbi_discretize::{Coords, Spacing, geometry_probe_gv};
use symbi_ir::proof::{RatFun, extract_scalar};

const NDIM: usize = 3;
// the geometry factors are field-free; the only symbols are the grid scalars
// (coordinate order 0=r, 1=phi, 2=z). dt is irrelevant to a face area.
const SCALARS: &[&str] = &["x_lo_0", "dx_0", "x_lo_1", "dx_1", "x_lo_2", "dx_2"];

// the r-direction unit cell shift e_r.
const E_R: [i64; NDIM] = [1, 0, 0];

// extract the dir-0 lo/hi face-area RatFuns from a fresh cylindrical geometry probe.
fn cyl_areas() -> (RatFun, RatFun) {
    let program =
        geometry_probe_gv(Coords::Cylindrical, &[Spacing::Uniform; NDIM], NDIM);
    let kernel = program.kernel();
    let writes = program.writes();
    // probe writes: 0=inv_volume, 1=area_lo_0, 2=area_hi_0, 3=centroid_0.
    let area_lo = extract_scalar(kernel.graph(), writes[1].value, SCALARS);
    let area_hi = extract_scalar(kernel.graph(), writes[2].value, SCALARS);
    (area_lo, area_hi)
}

#[test]
fn godunov_mass_conservation_cyl_symbolic() {
    let (area_lo, area_hi) = cyl_areas();

    // the proof: the cell's high r-face area equals its outward neighbor's low r-face
    // area. exact symbolic equality — the shared face is single-valued, so the
    // area-weighted flux divergence telescopes => mass conserves.
    assert!(
        area_hi.equals(&area_lo.shift_coords(&E_R)),
        "cylindrical r-face area inconsistency: area_hi_0(c) != area_lo_0(c+e_r) — does NOT conserve"
    );
}

// negative control: the unshifted low face area differs from the high face area (a cell's
// own two r-faces differ by exactly dr * dphi * dz), so the checker has real content.
#[test]
fn conservation_cyl_symbolic_detects_inconsistency() {
    let (area_lo, area_hi) = cyl_areas();
    assert!(
        !area_hi.equals(&area_lo),
        "area_hi_0 == area_lo_0 of the SAME cell would mean a degenerate (zero-width) cell"
    );
}
