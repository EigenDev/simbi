// =============================================================================
// problem_slice.rs
//
// two problems stated on the rust scientific surface and run end to end on
// the engine:
// - a Sod shock tube: regime, geometry, initial state and outflow walls, with
//   no source. marched to t = 0.1 at n = 128, the rarefaction, contact and
//   shock stay interior, so the edges hold the initial state, the middle
//   develops intermediate densities, and mass is conserved.
// - a uniform gas in a periodic box under uniform gravity: the source passes
//   the admission door at realization and then drives the step. a uniform
//   periodic box under a uniform force is a galilean boost, so density stays
//   uniform and the momentum grows as rho * g * t * L.
// =============================================================================

use symbi::problem::{
    Boundaries, Boundary, Cell, GasState, Geometry, Problem, Regime, Source, realize,
};

struct Sod {
    gamma: f64,
}

impl Problem for Sod {
    fn regime(&self) -> Regime {
        Regime::newtonian(self.gamma)
    }
    fn geometry(&self) -> Geometry {
        Geometry::cartesian_line(0.0, 1.0, 128)
    }
    fn boundaries(&self) -> Boundaries {
        Boundaries::uniform(Boundary::Outflow)
    }
    fn initial_state(&self, cell: Cell) -> GasState {
        if cell.center < 0.5 {
            GasState::at_rest(1.0, 1.0)
        } else {
            GasState::at_rest(0.125, 0.1)
        }
    }
}

struct FallingColumn {
    acceleration: f64,
}

impl Problem for FallingColumn {
    fn regime(&self) -> Regime {
        Regime::newtonian(1.4)
    }
    fn geometry(&self) -> Geometry {
        Geometry::cartesian_line(0.0, 1.0, 64)
    }
    fn boundaries(&self) -> Boundaries {
        Boundaries::uniform(Boundary::Periodic)
    }
    fn initial_state(&self, _cell: Cell) -> GasState {
        GasState::at_rest(1.0, 1.0)
    }
    fn sources(&self) -> Vec<Source> {
        vec![Source::UniformGravity {
            acceleration: self.acceleration,
        }]
    }
}

#[test]
fn sod_shock_tube_runs_from_the_scientific_surface() {
    let mut run = realize(&Sod { gamma: 1.4 }).expect("sod realizes");
    let mass0 = run.mass();
    run.advance_to(0.1).expect("sod advances");
    assert!(run.steps() > 0, "the engine took no step");
    assert!((run.time() - 0.1).abs() < 1e-12, "time = {}", run.time());

    let state = run.state();
    for (ii, gas) in state.iter().enumerate() {
        assert!(
            gas.density.is_finite() && gas.density > 0.0,
            "bad density {} at cell {ii}",
            gas.density
        );
        assert!(
            gas.pressure.is_finite() && gas.pressure > 0.0,
            "bad pressure {} at cell {ii}",
            gas.pressure
        );
    }
    let first = state.first().expect("interior");
    let last = state.last().expect("interior");
    assert!(
        (first.density - 1.0).abs() < 1e-9,
        "left edge disturbed: rho = {}",
        first.density
    );
    assert!(
        (last.density - 0.125).abs() < 1e-9,
        "right edge disturbed: rho = {}",
        last.density
    );
    assert!(
        state.iter().any(|g| g.density > 0.2 && g.density < 0.9),
        "no rarefaction or contact structure formed"
    );
    let max_vel = state.iter().map(|g| g.velocity.abs()).fold(0.0, f64::max);
    assert!(
        max_vel > 0.1,
        "gas did not accelerate (max |v| = {max_vel})"
    );
    let mass1 = run.mass();
    assert!(
        (mass1 - mass0).abs() < 1e-9 * mass0,
        "mass drift {:e}",
        mass1 - mass0
    );
}

#[test]
fn uniform_gravity_is_admitted_and_accelerates_the_gas() {
    let g = 0.5;
    let mut run = realize(&FallingColumn { acceleration: g }).expect("gravity passes admission");
    assert!(run.momentum().abs() < 1e-12, "momentum starts at zero");

    let t_final = 0.05;
    run.advance_to(t_final).expect("gravity advances");
    assert!(run.steps() > 0, "the engine took no step");

    // d(mom)/dt = rho * g over the unit line with rho = 1 uniform.
    let expected = g * t_final;
    let got = run.momentum();
    assert!(got > 0.0, "gas did not accelerate: momentum = {got}");
    assert!(
        (got - expected).abs() / expected < 0.02,
        "momentum = {got}, expected {expected}"
    );
    for (ii, gas) in run.state().iter().enumerate() {
        assert!(
            (gas.density - 1.0).abs() < 1e-6,
            "density drifted at cell {ii}: rho = {}",
            gas.density
        );
    }
}
