// =============================================================================
// kernel_slug.rs
//
// the ONE kernel-name SUFFIX protocol, shared by the AOT bake (mints the kernel
// names, `symbi-aot/build.rs`) and the runtime dispatch (re-derives them,
// `symbi-substrate/regimes/substrate_kernels`). a kernel name is
// `{prefix}{...suffix...}_{ndim}d[...]`; every suffix segment below is a
// SEMANTIC AXIS of the discretization (spatial chart, momentum DOF lift,
// spacetime background) that must select the SAME baked kernel from both sides.
//
// the two sides speak different enum families (`coords::Coords` at bake time vs
// `symbi_geometry::Geometry` at runtime). a slug derived independently on each
// side can only be held in lockstep by convention, and the failure is silent:
// either a name that resolves to the wrong discretization, or a name no bake
// ever emitted. ONE definition, called from both sides, removes the channel.
//
// every fn takes the runtime `symbi_geometry` enums; the bake projects its
// codegen mirror via `Coords::to_geometry` / `Spacetime::to_spacetime`.
//
// usage:
//   let sfx = kernel_slug::geom_suffix(coords, dof, ndim);      // hydro / GR
//   let sfx = kernel_slug::mhd_geom_suffix(coords, &axes);      // MHD
//   let st  = kernel_slug::spacetime_slug(spacetime);
// =============================================================================

use symbi_geometry::{Geometry, Spacetime};

/// the plain coordinate-system suffix: cartesian is unsuffixed; spherical /
/// cylindrical select the `_sph` / `_cyl` instance (area-weighted divergence +
/// geometric source + per-cell CFL widths). used by the DOF-agnostic kernels
/// (immersed body source / feedback), which key on the chart alone.
pub fn coord_suffix(coords: Geometry) -> &'static str {
    match coords {
        Geometry::Cartesian => "",
        Geometry::Spherical => "_sph",
        Geometry::Cylindrical => "_cyl",
    }
}

/// the DOF-aware curvilinear suffix. a grid whose momentum DOF exceeds its axis
/// count carries an extra conserved law (the out-of-plane swirl), changing the
/// manifest, so it needs its own kernel instance: spherical `(r, theta)` grid
/// with azimuthal momentum -> `_sph_swirl`; cylindrical `(r, z)` axisymmetric
/// with azimuthal swirl -> `_cyl_rz`. `dof == ndim` collapses to the chart
/// suffix. drives the hydro / GR godunov, wave-speed, c2p, snapshot families.
pub fn geom_suffix(coords: Geometry, dof: usize, ndim: usize) -> &'static str {
    match coords {
        Geometry::Cartesian => "",
        Geometry::Spherical => {
            if dof > ndim {
                "_sph_swirl"
            } else {
                "_sph"
            }
        }
        Geometry::Cylindrical => {
            if dof > ndim {
                "_cyl_rz"
            } else {
                "_cyl"
            }
        }
    }
}

/// the immersed-boundary penalize kernel name for a chart: `{base}{chart}_{ndim}d`
/// (e.g. `penalize_drain_iso_cyl_2d`). the drain touches only the gridded momenta
/// (dof == ndim, no swirl slot), so the suffix is "" / "_sph" / "_cyl". cartesian
/// reproduces the `KernelId` name exactly, so lifting curvilinear support leaves
/// the cartesian bake and dispatch untouched.
pub fn penalize_name(base: &str, coords: Geometry, ndim: usize, axes: &[usize]) -> String {
    // the two cylindrical 2d planes carry different chart maps: the (r, phi)
    // disk keeps the plain `_cyl` suffix; the (r, z) axisymmetric section
    // (axes [0, 2]) is `_cyl_rz` (identity section frame, on-axis bodies).
    if coords == Geometry::Cylindrical && ndim == 2 && axes[..2] == [0, 2] {
        return format!("{base}_cyl_rz_{ndim}d");
    }
    format!("{base}{}_{ndim}d", geom_suffix(coords, ndim, ndim))
}

/// the general orthogonal viscous kernel name for a chart: `{base}{chart}_{ndim}d`
/// (`base` = `viscous_iso_ortho` for constant nu, `viscous_iso_alpha_ortho` for the
/// alpha law). one operator, one name per chart (cylindrical `_cyl`, spherical
/// `_sph`); cartesian keeps its own flat kernels.
pub fn viscous_ortho_name(base: &str, coords: Geometry, ndim: usize) -> String {
    format!("{base}{}_{ndim}d", geom_suffix(coords, ndim, ndim))
}

/// which grid property a family keys its chart segment on.
///
/// the two are NOT interchangeable on a curvilinear grid. hydro keys on the momentum-DOF
/// lift, so a 2-axis spherical grid carrying azimuthal momentum is `_sph_swirl`. MHD B is
/// always a 3-vector, so the lift cannot separate the two cylindrical planes and the chart
/// keys on the grid-axis set instead.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChartKeying {
    /// hydro: `geom_suffix(coords, dof, ndim)`.
    MomentumDof,
    /// MHD: `mhd_geom_suffix(coords, axes)`.
    GridAxes,
}

/// the chart segment of an admissible-boundary projection kernel name, derived from the
/// GRID rather than typed at the call site.
///
/// "the regime carries no DOF lift" does NOT imply an empty chart segment: an empty segment
/// is correct only on cartesian, and a curvilinear chart that short-circuits to it asks for a
/// name no bake emits — a dispatch panic on the first cell that needs the projection.
/// deriving the segment from the grid leaves no call site the opportunity to short-circuit.
pub fn fofc_project_chart(
    keying: ChartKeying,
    coords: Geometry,
    axes: &[usize],
    dof: usize,
    ndim: usize,
) -> &'static str {
    match keying {
        ChartKeying::MomentumDof => geom_suffix(coords, dof, ndim),
        ChartKeying::GridAxes => mhd_geom_suffix(coords, axes),
    }
}

/// the admissible-boundary projection kernel name: `{prefix}_fofc_project{chart}{st}_{ndim}d`.
///
/// the family is CURVED-SPACETIME ONLY, and the two regimes key their chart segment on
/// different axes — hydro on the DOF lift (`geom_suffix`), MHD on the grid-axis set
/// (`mhd_geom_suffix`, since B is always a 3-vector and the DOF lift cannot separate the
/// two cylindrical planes). the caller supplies the chart segment its regime uses; the
/// ORDER and the surrounding literals live here, once, so the bake and the dispatch cannot
/// spell the same kernel two ways. an empty chart segment resolves only on cartesian, whose
/// segment is empty anyway; on every curvilinear chart it names a kernel no bake emits.
pub fn fofc_project_name(
    prefix: &str,
    chart_suffix: &str,
    spacetime: Spacetime,
    ndim: usize,
) -> String {
    format!(
        "{prefix}_fofc_project{chart_suffix}{}_{ndim}d",
        spacetime_slug(spacetime)
    )
}

/// the FACE-FLUX kernel name:
/// `{prefix}_face_flux{solver}{recon}{chart}{eos}{geom}{spacetime}_{ndim}d_{dir}`.
///
/// six independent suffix axes, and the ORDER is the whole point of this function. spelled at
/// the call site it has been spelled three mutually incompatible ways -- the bake putting the
/// reconstruction before the chart, one runtime folding the chart into the solver segment, and
/// another emitting the chart FIRST in one branch and second in the branch below it. those agree
/// only while the segments they disagree about are empty, so the break arrives with the first
/// non-default reconstruction on a curvilinear grid: a name no bake emitted, and a dispatch panic
/// on the first cell.
///
/// the axes, in order:
///   - `solver`     the riemann solver arm (`kernel_suffix`), `""` for HLLE.
///   - `recon`      the face reconstruction (`Recon::suffix`), `""` for PLM.
///   - `chart`      the coordinate chart, present ONLY for a reconstruction that reads
///                  positions -- a well-balanced reconstruction evaluates the body potential at
///                  cartesian coordinates, so it is baked per chart while every chart-agnostic
///                  flux passes `""`. it sits beside `recon` because it is a property OF the
///                  reconstruction, not of the solver.
///   - `eos`        the equation-of-state arm (`EosArm::suffix`), `""` for gamma-law.
///   - `geom`       the DOF-lift / GR chart tag (`geom_suffix` / `gr_chart_dof_tag`), a
///                  different axis from `chart`: it keys on momentum DOF exceeding the grid
///                  dimension, not on where a position is evaluated.
///   - `spacetime`  the curved-background slug, empty on minkowski.
pub fn face_flux_name(
    prefix: &str,
    solver_suffix: &str,
    recon_suffix: &str,
    chart_suffix: &str,
    eos_suffix: &str,
    geom_suffix: &str,
    spacetime: Spacetime,
    ndim: usize,
    dir: usize,
) -> String {
    format!(
        "{prefix}_face_flux{solver_suffix}{recon_suffix}{chart_suffix}{eos_suffix}\
{geom_suffix}{}_{ndim}d_{dir}",
        spacetime_slug(spacetime)
    )
}

/// the MHD curvilinear suffix, keyed on the GRID-AXIS SET (not DOF-vs-ndim). MHD
/// B is ALWAYS a 3-vector, so both cylindrical 2D planes carry DOF = 3 and the
/// DOF lift cannot tell them apart: r-z axisymmetric = axes `[0, 2]` (out-of-plane
/// phi) -> `_cyl_rz`; r-phi disk = axes `[0, 1]` (out-of-plane z) -> `_cyl_rphi`.
/// a 2D spherical MHD grid is the normal `_sph` (NOT the hydro swirl lift).
/// cartesian / 1D-radial / 3D collapse to the chart suffix. drives the gas
/// godunov / wave-speed / CT curl / bcell MHD families.
pub fn mhd_geom_suffix(coords: Geometry, axes: &[usize]) -> &'static str {
    match coords {
        Geometry::Cartesian => "",
        Geometry::Spherical => "_sph",
        Geometry::Cylindrical => match axes {
            [0, 2] => "_cyl_rz",
            [0, 1] => "_cyl_rphi",
            _ => "_cyl", // 1D radial [0] / 3D [0,1,2]
        },
    }
}

/// the MHD normal-direction riemann FLUX suffix: geometry-INDEPENDENT wherever the
/// normal component `coord_n == dir` (every IDENTITY-axis grid — cart / sph / 3D-cyl
/// AND the r-phi disk `[0, 1]`, which is identity), so those share the suffix-free
/// flux family. ONLY r-z (`[0, 2]`) lifts grid axis 1 onto the z component
/// (`coord_n = 2 != 1`), so its normal flux differs and needs `_cyl_rz`.
pub fn mhd_flux_suffix(coords: Geometry, axes: &[usize]) -> &'static str {
    if matches!(coords, Geometry::Cylindrical) && axes == [0, 2] {
        "_cyl_rz"
    } else {
        ""
    }
}

/// the spacetime background tag: flat `Minkowski` is unsuffixed (the
/// densitization is a no-op -> bit-identical to the SR kernel); each curved
/// chart carries its slug. ORTHOGONAL to the spatial suffix and the physics
/// regime (GR is a spacetime).
pub fn spacetime_slug(spacetime: Spacetime) -> &'static str {
    match spacetime {
        Spacetime::Minkowski => "",
        Spacetime::SchwarzschildKS => "_ks",
        Spacetime::KerrKS => "_kerr",
    }
}

/// the chart/DOF tag for the GR hydro c2p + face-flux kernel names, which encode
/// the coordinate chart through the DOF lift plus an explicit tag for the
/// non-spherical GR charts. a spherical/cyl swirl (`dof > ndim`) rides the
/// [`geom_suffix`] lift (`_sph_swirl` / `_cyl_rz`); a non-swirl curved chart
/// tags cartesian `_cart` / cylindrical `_cyl` while spherical stays the
/// implicit untagged default (the validated spherical GR kernels keep their
/// names); a flat non-swirl grid is untagged.
pub fn gr_chart_dof_tag(
    coords: Geometry,
    spacetime: Spacetime,
    dof: usize,
    ndim: usize,
) -> &'static str {
    if dof > ndim {
        geom_suffix(coords, dof, ndim)
    } else if spacetime != Spacetime::Minkowski {
        match coords {
            Geometry::Cartesian => "_cart",
            Geometry::Cylindrical => "_cyl",
            Geometry::Spherical => "",
        }
    } else {
        ""
    }
}
