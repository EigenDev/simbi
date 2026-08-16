// =============================================================================
// kernel_slug.rs
//
// the kernel-name suffix protocol, defined once and shared by the AOT bake (mints
// the kernel names, `symbi-aot/build.rs`) and the runtime dispatch (re-derives them,
// `symbi-substrate/regimes/substrate_kernels`). a kernel name is
// `{prefix}{...suffix...}_{ndim}d[...]`; every suffix segment below is a
// semantic axis of the discretization (spatial chart, momentum DOF lift,
// spacetime background), and both sides land on one and the same baked kernel.
//
// the two sides speak different enum families (`coords::Coords` at bake time vs
// `symbi_geometry::Geometry` at runtime). a slug derived independently on each
// side can only be held in lockstep by convention, and the failure is silent:
// either a name that resolves to the wrong discretization, or a name no bake
// ever emitted. a single definition, called from both sides, removes the channel.
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

/// the suffix for a momentum DOF exceeding the axis count (the out-of-plane
/// swirl lift): `_sph_swirl` / `_cyl_rz`. every other grid gets the empty string,
/// including a curvilinear grid with `dof == ndim`, where `geom_suffix` would still
/// say `_sph`/`_cyl`. `geom_suffix` names the chart, this one names the lift;
/// families whose cartesian and curvilinear lift-free instances share one kernel
/// name key on this one.
pub fn dof_lift_suffix(coords: Geometry, dof: usize, ndim: usize) -> &'static str {
    if dof > ndim {
        geom_suffix(coords, dof, ndim)
    } else {
        ""
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
/// the two keyings diverge on a curvilinear grid. hydro keys on the momentum-DOF
/// lift, so a 2-axis spherical grid carrying azimuthal momentum is `_sph_swirl`. MHD B is
/// always a 3-vector, so both cylindrical planes carry DOF = 3 and the chart keys on the
/// grid-axis set, which separates them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChartKeying {
    /// hydro: `geom_suffix(coords, dof, ndim)`.
    MomentumDof,
    /// MHD: `mhd_geom_suffix(coords, axes)`.
    GridAxes,
}

/// the chart segment of an admissible-boundary projection kernel name, derived from the
/// grid itself.
///
/// the empty chart segment belongs to cartesian alone. a regime carrying no DOF lift on a
/// curvilinear chart still takes its `_sph`/`_cyl` segment; short-circuiting to the empty
/// one asks for a name outside the baked set — a dispatch panic on the first cell that
/// needs the projection. deriving the segment from the grid keeps that choice away from
/// the call site.
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
/// the family is baked for curved spacetimes, and the two regimes key their chart segment
/// on different axes — hydro on the DOF lift (`geom_suffix`), MHD on the grid-axis set
/// (`mhd_geom_suffix`, since B is always a 3-vector and both cylindrical planes carry
/// DOF = 3). the caller supplies the chart segment its regime uses; the segment order and
/// the surrounding literals live here, once, so the bake and the dispatch spell the kernel
/// one way. an empty chart segment resolves on cartesian, whose segment is empty anyway;
/// on every curvilinear chart it names a kernel outside the baked set.
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

/// the face-flux kernel name, built from named fields.
///
/// `{prefix}_face_flux{solver}{recon}{balance}{chart}{eos}{geom}{spacetime}_{ndim}d_{dir}`.
///
/// eight suffix axes, whose order is the whole point. spelled at the call site it has been
/// spelled three mutually incompatible ways -- the bake putting the reconstruction before the
/// chart, one runtime folding the chart into the solver segment, and another emitting the chart
/// first in one branch and second in the branch below it. those agree only while the segments
/// they disagree about are empty, so the break arrives with the first non-default reconstruction
/// on a curvilinear grid: a name outside the baked set, and a dispatch panic on the first cell.
///
/// the fields are named because seven adjacent `&str` parameters are a transposition waiting
/// to happen, and a transposed pair here fails the same way -- silently, on one
/// configuration. `..Default::default()` lets a call site mention only the axes it uses.
///
/// the axes, in order:
///   - `solver`     the riemann solver arm (`Solver::kernel_suffix`), `""` for HLLE.
///   - `recon`      the face reconstruction limiter (`Recon::suffix`), `""` for PLM.
///   - `balance`    whether the reconstruction limits the state or its departure from
///                  hydrostatic equilibrium (`Balance::suffix`). a property of the
///                  reconstruction, so it sits beside `recon`, and an axis of its own because
///                  any solver may be well-balanced: the first-order FOFC redo runs HLLE, and
///                  a piecewise-constant reconstruction of departures is exactly balanced, so
///                  the redo carries that property for free.
///   - `chart`      the coordinate chart, present for a reconstruction that reads
///                  positions -- a well-balanced reconstruction evaluates the body potential at
///                  cartesian coordinates, so it is baked per chart, while a chart-agnostic
///                  flux passes `""`.
///   - `eos`        the equation-of-state arm (`EosArm::suffix`), `""` for gamma-law.
///   - `geom`       the DOF-lift / GR chart tag, a different axis from `chart`: it keys on
///                  momentum DOF exceeding the grid dimension, while `chart` keys on where a
///                  position is read.
///   - `spacetime`  the curved-background slug, empty on minkowski.
#[derive(Clone, Copy)]
pub struct FaceFluxName<'a> {
    pub prefix: &'a str,
    pub solver: &'a str,
    pub recon: &'a str,
    pub balance: &'a str,
    pub chart: &'a str,
    pub eos: &'a str,
    pub geom: &'a str,
    pub spacetime: Spacetime,
    pub ndim: usize,
    pub dir: usize,
}

impl Default for FaceFluxName<'_> {
    fn default() -> Self {
        Self {
            prefix: "",
            solver: "",
            recon: "",
            balance: "",
            chart: "",
            eos: "",
            geom: "",
            spacetime: Spacetime::Minkowski,
            ndim: 1,
            dir: 0,
        }
    }
}

impl FaceFluxName<'_> {
    pub fn build(&self) -> String {
        format!(
            "{}_face_flux{}{}{}{}{}{}{}_{}d_{}",
            self.prefix,
            self.solver,
            self.recon,
            self.balance,
            self.chart,
            self.eos,
            self.geom,
            spacetime_slug(self.spacetime),
            self.ndim,
            self.dir
        )
    }
}


/// the MHD curvilinear suffix, keyed on the grid-axis set. MHD B is always a
/// 3-vector, so both cylindrical 2D planes carry DOF = 3 and the axis set is what
/// tells them apart: r-z axisymmetric = axes `[0, 2]` (out-of-plane
/// phi) -> `_cyl_rz`; r-phi disk = axes `[0, 1]` (out-of-plane z) -> `_cyl_rphi`.
/// a 2D spherical MHD grid takes the plain `_sph`, leaving the swirl lift to hydro.
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

/// the MHD normal-direction riemann flux suffix: one geometry-independent flux serves
/// every chart whose normal component satisfies `coord_n == dir` (the identity-axis
/// grids — cart / sph / 3D-cyl, plus the r-phi disk `[0, 1]`, which is identity), so those
/// share the suffix-free flux family. r-z (`[0, 2]`) lifts grid axis 1 onto the z component
/// (`coord_n = 2 != 1`), giving it a normal flux of its own under `_cyl_rz`.
pub fn mhd_flux_suffix(coords: Geometry, axes: &[usize]) -> &'static str {
    if matches!(coords, Geometry::Cylindrical) && axes == [0, 2] {
        "_cyl_rz"
    } else {
        ""
    }
}

/// the spacetime background tag: flat `Minkowski` is unsuffixed (the
/// densitization is a no-op -> bit-identical to the SR kernel); each curved
/// chart carries its slug. orthogonal to the spatial suffix and the physics
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
