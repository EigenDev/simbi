// =============================================================================
// solver_coverage_gate.rs
//
// every CLI-reachable (MHD regime x dimension x geometry x --solver) must have its face-flux
// kernel emitted. the dispatch builds the name `{prefix}_face_flux{geom}{solver}_{D}d_{dir}`
// (substrate_{rmhd,newtonian_mhd,isothermal_mhd}.rs flux()); this asserts each such name resolves
// in the AOT registry, catching the "valid flag, missing kernel" class (2D RMHD HLLD, say)
// at test time, before a run requests the kernel. solver applicability mirrors the physics: iso MHD has no contact
// wave -> no HLLC.
// =============================================================================

use symbi::regimes::substrate_kernels::{Solver, kernel_exists};

// the MHD regimes and the solvers each one's KernelSet is meant to serve.
const MHD_REGIMES: &[(&str, &[Solver])] = &[
    ("nmhd", &[Solver::Hlle, Solver::Hllc, Solver::Hlld]),
    ("imhd", &[Solver::Hlle, Solver::Hlld]),
    ("rmhd", &[Solver::Hlle, Solver::Hllc, Solver::Hlld]),
];

// the hydro regimes (kernel prefix from each `flux()` dispatch) and their solvers. hydro flux is
// geometry-free except for the DOF-lifted cyl-swirl (DOF != D), so cartesian + spherical share
// these names. iso is HLLE-only (no contact wave); curvilinear-swirl HLLC is unsupported
// and is intentionally NOT asserted here.
const HYDRO_REGIMES: &[(&str, &[Solver])] = &[
    ("adiabatic", &[Solver::Hlle, Solver::Hllc]),
    ("rhd", &[Solver::Hlle, Solver::Hllc]),
    ("iso", &[Solver::Hlle]),
];

// the exact name the regime dispatch builds -- through the SAME composer the dispatch and the
// bake use, never a local `format!`. this gate spelled it independently with the chart segment
// BEFORE the solver, which is one of the three incompatible orders that let a curvilinear
// kernel name diverge from its bake; a gate that re-derives the protocol it is checking can
// only ever confirm its own copy of it.
fn flux_name(prefix: &str, geom: &str, solver: Solver, d: usize, dir: usize) -> String {
    symbi_discretize::kernel_slug::FaceFluxName {
        prefix,
        solver: solver.kernel_suffix(),
        geom,
        ndim: d,
        dir,
        ..Default::default()
    }
    .build()
}

#[test]
fn every_mhd_solver_flux_is_emitted() {
    let mut missing = Vec::new();
    for &(prefix, solvers) in MHD_REGIMES {
        for &solver in solvers {
            // cartesian: 1D / 2D / 3D, every direction.
            for d in 1..=3usize {
                for dir in 0..d {
                    let n = flux_name(prefix, "", solver, d, dir);
                    if !kernel_exists(&n) {
                        missing.push(n);
                    }
                }
            }
            // cyl r-z plane (axes [0,2]): 2D only. (r-phi reuses the cartesian flux via the "" suffix.)
            for dir in 0..2usize {
                let n = flux_name(prefix, "_cyl_rz", solver, 2, dir);
                if !kernel_exists(&n) {
                    missing.push(n);
                }
            }
        }
    }
    assert!(
        missing.is_empty(),
        "{} MHD solver flux kernel(s) the dispatch can request are NOT emitted:\n  {}",
        missing.len(),
        missing.join("\n  ")
    );
}

#[test]
fn every_hydro_solver_flux_is_emitted() {
    let mut missing = Vec::new();
    for &(prefix, solvers) in HYDRO_REGIMES {
        for &solver in solvers {
            for d in 1..=3usize {
                for dir in 0..d {
                    // geom-free flux (cartesian + spherical share the name; geom suffix only
                    // appears for the DOF-lifted cyl-swirl, excluded above).
                    let n = flux_name(prefix, "", solver, d, dir);
                    if !kernel_exists(&n) {
                        missing.push(n);
                    }
                }
            }
        }
    }
    assert!(
        missing.is_empty(),
        "{} hydro solver flux kernel(s) the dispatch can request are NOT emitted:\n  {}",
        missing.len(),
        missing.join("\n  ")
    );
}
