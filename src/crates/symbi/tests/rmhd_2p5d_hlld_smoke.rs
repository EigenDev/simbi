// =============================================================================
// rmhd_2p5d_hlld_smoke.rs
//
// regression: a 2.5D RMHD sim must run with `--solver hlld`. the RMHD HLLD face fluxes were
// emitted at 1D + 3D but the 2.5D block shipped HLLE-only, so a 2D RMHD run with `--solver hlld`
// panicked at dispatch ("no generated kernel rmhd_face_flux_hlld_2d_0"). this pins the now-emitted
// 2D RMHD HLLD fluxes (cartesian; r-phi reuses them; cyl r-z has its own "_cyl_rz" variants). also
// a compact showcase of the ergonomics surface.
//
// hllc is no longer a valid RMHD solver: the (solver, regime) matrix is enforced at bind time in
// `with_solver` (`Solver::valid_for`), and HLLC carries no magnetic wave structure, so it is
// MHD-invalid. the hllc case below pins that rejection (a `SolverRegimeMismatch` config error).
// =============================================================================

use symbi::prelude::*;

type Sim = SimCpuGeneric<Rmhd, 2, 3, Cartesian, IdealGas<f64>>;

/// build a 2.5D RMHD sim seeded with a uniform div-free in-plane B + gentle subluminal flow.
fn build_sim() -> Sim {
    let sim = Sim::build(Rmhd, IdealGas { gamma: 5.0 / 3.0 }, Cartesian)
        .cells([32, 32])
        .bounds([0.0, 0.0], [1.0, 1.0])
        .boundaries(BoundaryType::Periodic)
        .finish()
        .unwrap();
    sim.seed_face(0, 0.2);
    sim.seed_face(1, 0.2);
    sim.seed_cells(|_| MhdPrim {
        hydro: Prim { rho: 1.0, vel: Tensor::new([0.1, 0.0, 0.0]), pre: 1.0 },
        mag: Tensor::new([0.2, 0.2, 0.0]),
    });
    sim
}

#[test]
fn rmhd_2p5d_runs_with_hlld() {
    let mut sim = build_sim();
    let sub = sim.substrate().with_solver(Solver::Hlld).expect("hlld is valid for rmhd");
    evolve(&mut sim, &sub, 0.02).unwrap_or_else(|e| panic!("2D RMHD HLLD evolve failed: {e}"));
    assert!(sim.iteration >= 1, "no steps taken with HLLD");
    for c in sim.geom.interior.iter() {
        let p = sim.prim_at(c);
        assert!(p.rho.is_finite() && p.rho > 0.0, "HLLD cell {c:?}: rho={}", p.rho);
        assert!(p.pre.is_finite() && p.pre > 0.0, "HLLD cell {c:?}: p={}", p.pre);
    }
}

// the bind-time solver matrix for rmhd: hllc is VALID (hllc-mhd = the contact-resolving hllc
// flux + the hll edge emf — the contact carries no transverse field for B_x != 0, M&DZ p.11);
// hllc-lm is REJECTED (the low-mach correction is a non-relativistic gas closure).
#[test]
fn rmhd_2p5d_solver_matrix() {
    let sim = build_sim();
    assert!(sim.substrate().with_solver(Solver::Hllc).is_ok(), "hllc is valid for rmhd");
    // the substrate kernel set is not Debug, so match the Result directly rather than expect_err.
    match sim.substrate().with_solver(Solver::HllcLm) {
        Err(ConfigError::SolverRegimeMismatch { .. }) => {}
        Err(e) => panic!("expected SolverRegimeMismatch, got {e:?}"),
        Ok(_) => panic!("hllc-lm must be rejected for rmhd"),
    }
}
