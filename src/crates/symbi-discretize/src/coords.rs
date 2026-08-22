// =============================================================================
// coords.rs
//
// the substrate coordinate-system + spacing types: the two codegen-time enums
// every gv geometry builder takes. pure data (no logic) — the coordinate system
// selects which analytic finite-volume closed form `cell_geometry_gv` emits, and
// the per-axis spacing selects the index -> coordinate map branch.
//
// axis order is natural per system: cartesian (x, y, z), spherical (r, theta, phi),
// cylindrical (r, phi, z). matching this to the field memory layout is an axis-role
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
/// (`gv_lapse_weight`). orthogonal to `Coords` (spatial) and to the physics regime: GR is a
/// spacetime axis independent of the physics regime, so any SR regime composes with any spacetime. defaults to `Minkowski`
/// (flat: the densitization is a no-op -> bit-identical), so existing kernels need no annotation.
/// the codegen-time mirror of `symbi_geometry::Spacetime` (like `Coords` mirrors `Geometry`).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
pub enum Spacetime {
    #[default]
    Minkowski,
    /// static spherically-symmetric vacuum: lapse alpha = sqrt(1 - 2M/r), shift = 0. the codegen
    /// schwarzschild in ingoing kerr-schild coords: lapse alpha = 1/sqrt(1 + 2M/r), radial shift
    /// beta^r = 2M/(r + 2M), gamma_rr = 1 + 2M/r. horizon-penetrating codegen tag selecting the
    /// shift-advection flux + KS densitization path. mirrors `symbi_geometry::Spacetime::SchwarzschildKS`.
    SchwarzschildKS,
    /// spinning kerr in ingoing kerr-schild coords: Sigma = r^2 + a^2 cos^2(theta), b = 2Mr/Sigma,
    /// lapse alpha = 1/sqrt(1 + b) (theta-dependent), radial shift beta^r = b/(1 + b),
    /// non-diagonal gamma_{r phi} (frame dragging). requires the covariant valencia storage and
    /// the azimuthal momentum DOF (the `_sph_swirl` family). the mass M and spin a ride as the
    /// `schwarzschild_mass` / `kerr_spin` kernel scalars. mirrors `symbi_geometry::Spacetime::KerrKS`.
    KerrKS,
}

/// the evolution face reconstruction — a codegen-time choice. pcm and plm share one
/// baked kernel (the runtime `theta` scalar selects: theta = 0 collapses the limited
/// slope to piecewise-constant), so only stencil-width changes appear here: ppm widens
/// the load stencil from -2..+1 to -3..+2 and is therefore a distinct baked kernel.
/// defaults to `Plm` (the theta-parameterized kernel), so existing builders need no
/// annotation and their kernel names are unchanged.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
pub enum Recon {
    /// piecewise linear, runtime-theta limiter family (theta-MC / van leer / pcm at 0).
    #[default]
    Plm,
    /// piecewise parabolic (colella & woodward 1984 monotonized interfaces) supplying
    /// method-of-lines face states; carries its own monotonicity constraint, no theta.
    Ppm,
}

impl Recon {
    /// the balance-fade footprint in cells: the farthest offset from its anchor cell at
    /// which this reconstruction evaluates the local equilibrium (plm reads departures
    /// two cells out; the ppm parabola's six-point window reads three). the balanced
    /// flux and the equilibrium-pressure body source measure the segment's spend over
    /// this same reach, so the KM pair fades together on one weight per cell.
    pub fn balance_reach(self) -> i64 {
        match self {
            Recon::Plm => 2,
            Recon::Ppm => 3,
        }
    }
}

/// whether a reconstruction limits the state or its departure from local hydrostatic
/// equilibrium.
///
/// an axis of its own, orthogonal to `Recon` and to the riemann solver. the transform is
/// independent of which limiter consumes it -- plm, ppm and pcm all inherit it -- and
/// independent of which solver consumes the face states. tying it to a solver, as an earlier
/// arrangement did, denies it to the first-order FOFC redo (which runs HLLE) precisely in the
/// stagnant stratified cells most likely to trip that redo.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum Balance {
    /// limit the state itself.
    #[default]
    Plain,
    /// limit each cell's departure from the isentropic hydrostatic profile through it, and add
    /// that profile back at the face. exact on a discretely balanced column; reduces to `Plain`
    /// bit-for-bit under plm when there is no gravity.
    Hydrostatic,
}

impl Balance {
    pub fn suffix(self) -> &'static str {
        match self {
            Balance::Plain => "",
            Balance::Hydrostatic => "_wb",
        }
    }
}

impl Recon {
    /// kernel-name suffix. the plm family keeps its unsuffixed names, so every
    /// pre-existing baked kernel name is untouched by the reconstruction axis.
    pub fn suffix(self) -> &'static str {
        match self {
            Recon::Plm => "",
            Recon::Ppm => "_ppm",
        }
    }
}

/// the equation-of-state closure — a codegen-time choice for the relativistic
/// family. the gamma-law is a runtime scalar (one kernel serves every gamma),
/// so only closure-structure changes appear here: the taub-mathews synge-gas
/// approximation replaces the gamma-law algebra throughout the traced physics
/// and is therefore a distinct baked kernel. defaults to `IdealGamma`, so
/// existing builders need no annotation and their kernel names are unchanged.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
pub enum EosArm {
    /// gamma-law ideal gas, the runtime `gamma` scalar selecting the index.
    #[default]
    IdealGamma,
    /// taub-mathews approximation to the synge relativistic perfect gas:
    /// parameter-free, effective index 5/3 -> 4/3 across theta = p/rho.
    /// the `gamma` kernel scalar stays bound but unread.
    TaubMathews,
}

impl EosArm {
    /// kernel-name suffix. the gamma-law family keeps its unsuffixed names.
    pub fn suffix(self) -> &'static str {
        match self {
            EosArm::IdealGamma => "",
            EosArm::TaubMathews => "_tm",
        }
    }
}

impl Coords {
    /// project the codegen-time mirror onto the runtime `symbi_geometry::Geometry` — so the single
    /// shared kernel-name suffix protocol (`kernel_slug`) takes one enum family for both the
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
