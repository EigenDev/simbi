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
/// spacetime axis independent of the physics regime, so any SR regime composes with any spacetime. defaults to `Minkowski`
/// (flat: the densitization is a no-op -> bit-identical), so existing kernels need no annotation.
/// the codegen-time mirror of `symbi_geometry::Spacetime` (like `Coords` mirrors `Geometry`).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
pub enum Spacetime {
    #[default]
    Minkowski,
    /// static spherically-symmetric vacuum: lapse alpha = sqrt(1 - 2M/r), shift = 0. the codegen
    /// schwarzschild in ingoing kerr-schild coords: lapse alpha = 1/sqrt(1 + 2M/r), radial shift
    /// beta^r = 2M/(r + 2M), gamma_rr = 1 + 2M/r. horizon-penetrating codegen TAG selecting the
    /// shift-advection flux + KS densitization path. mirrors `symbi_geometry::Spacetime::SchwarzschildKS`.
    SchwarzschildKS,
    /// spinning kerr in ingoing kerr-schild coords: Sigma = r^2 + a^2 cos^2(theta), b = 2Mr/Sigma,
    /// lapse alpha = 1/sqrt(1 + b) (THETA-dependent), radial shift beta^r = b/(1 + b),
    /// NON-DIAGONAL gamma_{r phi} (frame dragging). requires the covariant valencia storage and
    /// the azimuthal momentum DOF (the `_sph_swirl` family). the mass M and spin a ride as the
    /// `schwarzschild_mass` / `kerr_spin` kernel scalars. mirrors `symbi_geometry::Spacetime::KerrKS`.
    KerrKS,
}

impl Coords {
    /// project the codegen-time mirror onto the runtime `symbi_geometry::Geometry` — so the ONE
    /// shared kernel-name suffix protocol (`kernel_slug`) takes a single enum family for both the
    /// AOT bake (which speaks this mirror) and the runtime dispatch (which speaks `Geometry`).
    pub fn to_geometry(self) -> symbi_geometry::Geometry {
        match self {
            Coords::Cartesian => symbi_geometry::Geometry::Cartesian,
            Coords::Spherical => symbi_geometry::Geometry::Spherical,
            Coords::Cylindrical => symbi_geometry::Geometry::Cylindrical,
        }
    }
}

impl Spacetime {
    /// project the codegen-time mirror onto the runtime `symbi_geometry::Spacetime` (see
    /// [`Coords::to_geometry`]).
    pub fn to_spacetime(self) -> symbi_geometry::Spacetime {
        match self {
            Spacetime::Minkowski => symbi_geometry::Spacetime::Minkowski,
            Spacetime::SchwarzschildKS => symbi_geometry::Spacetime::SchwarzschildKS,
            Spacetime::KerrKS => symbi_geometry::Spacetime::KerrKS,
        }
    }
}
