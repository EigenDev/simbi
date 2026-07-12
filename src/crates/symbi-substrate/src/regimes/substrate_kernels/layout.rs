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

use symbi_aot::kernel_by_name;

// the allocation/execution layouts + the AOT-registry lookup wrapper live in the
// `symbi-exec` crate (docs/design/40); re-exported here so the substrate's
// `super::layout::{alloc_layout, exec_layout, expect_kernel}` paths resolve.
pub use symbi_exec::layout::{alloc_layout, exec_layout, expect_kernel};

// the kernel-name suffix protocol is defined ONCE in symbi-discretize::kernel_slug and shared with
// the AOT bake (build.rs), so bake and dispatch cannot drift. re-exported here so the substrate's
// `super::layout::{coord_suffix, geom_suffix, ...}` paths resolve unchanged.
pub use symbi_discretize::kernel_slug::{
    coord_suffix, geom_suffix, gr_chart_dof_tag, mhd_flux_suffix, mhd_geom_suffix, penalize_name,
    spacetime_slug,
};

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
