// =============================================================================
// regimes/substrate_kernels/layout.rs
//
// the curvilinear kernel-name suffix helpers (coord/geom/mhd suffixes), the
// per-axis geometry-scalar push, and the `kernel_exists` coverage probe. pure
// index/string arithmetic — no dispatch. the executor-facing layouts
// (`alloc_layout`/`exec_layout`/`expect_kernel`) live in the `symbi-exec` crate
// (docs/design/40) and are re-exported below so the paths resolve.
// =============================================================================

use symbi_algebra::OrderedNumeric;
use symbi_ir::algebra::Scalar;
use symbi_geometry::Geometry;

use symbi_aot::kernel_by_name;

// the allocation/execution layouts + the AOT-registry lookup wrapper live in the
// `symbi-exec` crate (docs/design/40); re-exported here so the substrate's
// `super::layout::{alloc_layout, exec_layout, expect_kernel}` paths resolve.
pub use symbi_exec::layout::{alloc_layout, exec_layout, expect_kernel};

// the curvilinear kernel-name suffix for a coordinate system: Cartesian kernels are
// unsuffixed; spherical / cylindrical select the `_sph` / `_cyl` instance (which carries
// the area-weighted divergence + geometric source + per-cell physical CFL widths).
pub fn coord_suffix(coords: Geometry) -> &'static str {
    match coords {
        Geometry::Cartesian => "",
        Geometry::Spherical => "_sph",
        Geometry::Cylindrical => "_cyl",
    }
}

// the DOF-aware suffix: the cylindrical 2D plane is ambiguous — r-phi (the disk,
// DOF == ndim, in-plane v_r/v_phi) vs r-z (axisymmetric, DOF > ndim, the swirl v_phi
// out of the gridded r-z plane). DOF vs ndim discriminates them, matching build.rs's
// Geom::suffix (ncomp vs naxes): r-phi -> "_cyl", r-z -> "_cyl_rz". cartesian/spherical
// are DOF-independent. use this wherever a kernel's geometry depends on the cyl plane.
pub fn geom_suffix(coords: Geometry, dof: usize, ndim: usize) -> &'static str {
    match coords {
        Geometry::Cartesian => "",
        Geometry::Spherical => "_sph",
        Geometry::Cylindrical => {
            if dof > ndim { "_cyl_rz" } else { "_cyl" }
        }
    }
}

// the MHD kernel suffix keyed on the GRID-AXIS SET (not DOF-vs-ndim, which can't tell the two
// cylindrical 2D MHD planes apart since both carry a 3-vector B). r-z axisymmetric grids (r, z)
// = axes [0,2] (out-of-plane phi); r-phi disk grids (r, phi) = axes [0,1] (out-of-plane z).
// cartesian/spherical/1D-radial/3D collapse to the coord suffix. drives gas godunov / wave-speed
// / CT curl / bcell dispatch — anything whose metric depends on which cyl plane is gridded.
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

// the MHD FLUX kernel is the normal-direction Riemann flux: geometry-INDEPENDENT wherever the
// normal component coord_n == dir (every IDENTITY-axis grid — cart/sph/3d-cyl AND the r-phi disk
// [0,1], which is identity). those reuse the suffix-free flux family. ONLY r-z ([0,2]) lifts grid
// axis 1 onto the z component (coord_n=2 != 1), so its normal flux differs and needs "_cyl_rz".
pub fn mhd_flux_suffix(coords: Geometry, axes: &[usize]) -> &'static str {
    if matches!(coords, Geometry::Cylindrical) && axes == [0, 2] { "_cyl_rz" } else { "" }
}

// the per-axis geometry scalars a CURVILINEAR kernel expects, in the order cell_geometry
// interns them: interleaved (x_lo_ax, dx_ax) per axis. Cartesian kernels take dx (or
// inv_dx) instead; the caller pushes those directly.
pub fn push_curvilinear_geom<Sc: Scalar + OrderedNumeric, const D: usize>(scalars: &mut Vec<Sc>, x_lo: &[f64; D], dx: &[f64; D]) {
    for ax in 0..D {
        scalars.push(Sc::from_f64(x_lo[ax]));
        scalars.push(Sc::from_f64(dx[ax]));
    }
}

/// coverage introspection: does the AOT registry hold a kernel named `name`? (any geometry / solver
/// suffix is already baked into `name`.) wraps the f64 registry lookup — CPU + CUDA are emitted in
/// lockstep, so f64 presence implies the kernel exists for every carrier. the coverage gate uses
/// this to assert every CLI-reachable (regime, D, geometry, solver) flux name actually resolves,
/// catching the "valid flag, missing kernel" class before a user does.
pub fn kernel_exists(name: &str) -> bool {
    kernel_by_name::<f64>(name).is_some()
}
