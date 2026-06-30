// =============================================================================
// coords.rs
//
// the substrate coordinate-system + spacing TYPES: the two codegen-time enums
// every gv geometry builder takes. pure data (no logic) — the coordinate system
// selects which analytic finite-volume closed form `cell_geometry_gv` emits, and
// the per-axis spacing selects the index -> coordinate map branch.
//
// axis order is natural per system: Cartesian (x, y, z), Spherical (r, theta, phi),
// Cylindrical (r, phi, z). matching this to the field memory layout is an axis-role
// concern (see `gv::cell_geometry_gv`'s `axes` map).
// =============================================================================

/// per-axis grid spacing law — index -> coordinate. a codegen-time choice.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Spacing {
    /// `face(i) = start + i*dx`.
    Uniform,
    /// `face(i) = start * 10^(i*log_slope)` — radial zones in astrophysics.
    Log,
}

/// the coordinate system — selects the analytic `cell_volume`/`face_area`/
/// `centroid`/`scale_factor` closed forms in `gv::cell_geometry_gv`. defaults to
/// `Cartesian` (the uniform-flat kernel context), so existing kernels need no
/// geometry annotation.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
pub enum Coords {
    #[default]
    Cartesian,
    Spherical,
    Cylindrical,
}

/// the spacetime background — selects the GR lapse / sqrt(gamma) densitization in the gv stage
/// (`gv_lapse_weight`). ORTHOGONAL to `Coords` (spatial) and to the physics regime: GR is a
/// spacetime, not a regime, so any SR regime composes with any spacetime. defaults to `Minkowski`
/// (flat: the densitization is a no-op -> bit-identical), so existing kernels need no annotation.
/// the codegen-time mirror of `symbi_geometry::Spacetime` (like `Coords` mirrors `Geometry`).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
pub enum Spacetime {
    #[default]
    Minkowski,
    /// static spherically-symmetric vacuum: lapse alpha = sqrt(1 - 2M/r), shift = 0. the codegen
    /// TAG (M-agnostic — M rides as a kernel value, like `dx`). mirrors `symbi_geometry::Spacetime`.
    Schwarzschild,
    // Kerr lands here at B5.
}
