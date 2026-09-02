// =============================================================================
// rmhd_2p5d_hlld_smoke.rs
//
// regression: a 2.5D RMHD sim must run with `--solver hlld`. an emission block covering only 1D
// and 3D leaves a 2D RMHD run with `--solver hlld` panicking at dispatch ("no generated kernel
// rmhd_face_flux_hlld_2d_0"). this pins the 2D RMHD HLLD fluxes (cartesian; r-phi reuses them;
// cyl r-z has its own "_cyl_rz" variants).
//
// hllc is valid for rmhd (hllc-mhd = the contact-resolving hllc flux + the hll edge emf); hllc-lm
// is rejected because the low-mach correction is a non-relativistic gas closure. the (solver,
// regime) matrix is enforced at bind time in `with_solver` (`Solver::valid_for`). the hllc-lm case
// below pins that rejection (a `SolverRegimeMismatch` config error).
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
    sim.seed_cells(|_| {
        MhdPrim::new(
            Prim::adiabatic(Density(1.0), Tensor::new([0.1, 0.0, 0.0]), Pressure(1.0)),
            Tensor::new([0.2, 0.2, 0.0]),
        )
    });
    sim
}

#[test]
fn rmhd_2p5d_runs_with_hlld() {
    let mut sim = build_sim();
    let sub = sim
        .substrate()
        .with_solver(Solver::Hlld)
        .expect("hlld is valid for rmhd");
    evolve(&mut sim, &sub, 0.02).unwrap_or_else(|e| panic!("2D RMHD HLLD evolve failed: {e}"));
    assert!(sim.iteration >= 1, "no steps taken with HLLD");
    for c in sim.geom.interior.iter() {
        let p = sim.expect_prim_at(c);
        assert!(
            p.rho().is_finite() && p.rho() > 0.0,
            "HLLD cell {c:?}: rho={}",
            p.rho()
        );
        assert!(
            p.pre().is_finite() && p.pre() > 0.0,
            "HLLD cell {c:?}: p={}",
            p.pre()
        );
    }
}

// the bind-time solver matrix for rmhd: hllc is valid (hllc-mhd = the contact-resolving hllc
// flux + the hll edge emf — the contact carries no transverse field for B_x != 0, M&DZ p.11);
// hllc-lm is rejected (the low-mach correction is a non-relativistic gas closure).
#[test]
fn rmhd_2p5d_solver_matrix() {
    let sim = build_sim();
    assert!(
        sim.substrate().with_solver(Solver::Hllc).is_ok(),
        "hllc is valid for rmhd"
    );
    // the substrate kernel set is not Debug, so match the Result directly.
    match sim.substrate().with_solver(Solver::HllcPlus) {
        Err(ConfigError::SolverRegimeMismatch { .. }) => {}
        Err(e) => panic!("expected SolverRegimeMismatch, got {e:?}"),
        Ok(_) => panic!("hllc-lm must be rejected for rmhd"),
    }
}
