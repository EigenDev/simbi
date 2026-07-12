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
// bake and dispatch previously each re-derived these slugs on their own enum
// family (`coords::Coords` vs `symbi_geometry::Geometry`), kept in lockstep by
// comment only. that convention has drifted twice (log-radial dispatched
// uniform-geometry kernels; a partial rename yielded kernel-not-found). one
// definition here — both sides call it — removes the drift channel by
// construction.
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

/// the plain coordinate-system suffix: Cartesian is unsuffixed; spherical /
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
            if dof > ndim { "_sph_swirl" } else { "_sph" }
        }
        Geometry::Cylindrical => {
            if dof > ndim { "_cyl_rz" } else { "_cyl" }
        }
    }
}

/// the immersed-boundary penalize kernel name for a chart: `{base}{chart}_{ndim}d`
/// (e.g. `penalize_drain_iso_cyl_2d`). the drain touches only the gridded momenta
/// (dof == ndim, no swirl slot), so the suffix is "" / "_sph" / "_cyl". Cartesian
/// reproduces the `KernelId` name exactly, so lifting curvilinear support leaves
/// the Cartesian bake and dispatch untouched.
pub fn penalize_name(base: &str, coords: Geometry, ndim: usize) -> String {
    format!("{base}{}_{ndim}d", geom_suffix(coords, ndim, ndim))
}

/// the general orthogonal viscous kernel name for a chart: `{base}{chart}_{ndim}d`
/// (`base` = `viscous_iso_ortho` for constant nu, `viscous_iso_alpha_ortho` for the
/// alpha law). one operator, one name per chart (cylindrical `_cyl`, spherical
/// `_sph`); Cartesian keeps its own flat kernels.
pub fn viscous_ortho_name(base: &str, coords: Geometry, ndim: usize) -> String {
    format!("{base}{}_{ndim}d", geom_suffix(coords, ndim, ndim))
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

/// the MHD normal-direction Riemann FLUX suffix: geometry-INDEPENDENT wherever the
/// normal component `coord_n == dir` (every IDENTITY-axis grid — cart / sph / 3D-cyl
/// AND the r-phi disk `[0, 1]`, which is identity), so those share the suffix-free
/// flux family. ONLY r-z (`[0, 2]`) lifts grid axis 1 onto the z component
/// (`coord_n = 2 != 1`), so its normal flux differs and needs `_cyl_rz`.
pub fn mhd_flux_suffix(coords: Geometry, axes: &[usize]) -> &'static str {
    if matches!(coords, Geometry::Cylindrical) && axes == [0, 2] { "_cyl_rz" } else { "" }
}

/// the spacetime background tag: flat `Minkowski` is unsuffixed (the
/// densitization is a no-op -> bit-identical to the SR kernel); each curved
/// chart carries its slug. ORTHOGONAL to the spatial suffix and the physics
/// regime (GR is a spacetime, not a regime).
pub fn spacetime_slug(spacetime: Spacetime) -> &'static str {
    match spacetime {
        Spacetime::Minkowski => "",
        Spacetime::Schwarzschild => "_schw",
        Spacetime::KerrSchild => "_ks",
        Spacetime::Kerr => "_kerr",
    }
}

/// the chart/DOF tag for the GR hydro c2p + face-flux kernel names, which encode
/// the coordinate chart through the DOF lift plus an explicit tag for the
/// non-spherical GR charts. a spherical/cyl swirl (`dof > ndim`) rides the
/// [`geom_suffix`] lift (`_sph_swirl` / `_cyl_rz`); a non-swirl curved chart
/// tags cartesian `_cart` / cylindrical `_cyl` while spherical stays the
/// implicit untagged default (the validated spherical GR kernels keep their
/// names); a flat non-swirl grid is untagged.
pub fn gr_chart_dof_tag(coords: Geometry, spacetime: Spacetime, dof: usize, ndim: usize) -> &'static str {
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
