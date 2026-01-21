// =============================================================================
// sod_shock_tube.rs
//
// classic sod shock tube test case for 1d euler equations.
// validates solver accuracy and captures shocks, contact discontinuities,
// and rarefaction waves.
//
// initial condition:
//   left (x < 0.5):  rho = 1.0, vx = 0.0, p = 1.0
//   right (x > 0.5): rho = 0.125, vx = 0.0, p = 0.1
//
// run:
//   cargo run --release --example sod_shock_tube
// =============================================================================

use physics::hydro::{BoundaryCondition, Euler1DSolver, ExecutionMode, Primitive1D};

fn main() {
    println!("=== Sod Shock Tube Test ===\n");

    // simulation parameters
    let ncells = 200;
    let xmin = 0.0;
    let xmax = 1.0;
    let gamma = 1.4;
    let cfl = 0.5;
    let t_final = 0.2;

    // create solver (serial cpu by default)
    let mut solver = Euler1DSolver::new(
        ncells,
        xmin,
        xmax,
        gamma,
        cfl,
        BoundaryCondition::Outflow,
        3, // rk3
        ExecutionMode::Serial,
    );

    println!("Configuration:");
    println!("  ncells: {}", ncells);
    println!("  domain: [{}, {}]", xmin, xmax);
    println!("  gamma: {}", gamma);
    println!("  cfl: {}", cfl);
    println!("  t_final: {}", t_final);
    println!("  boundary: Outflow");
    println!("  time integrator: RK3");
    println!("  execution: Serial CPU");
    println!();

    // set initial conditions (sod shock tube)
    solver.set_initial_conditions(|x| {
        if x < 0.5 {
            Primitive1D::new(1.0, 0.0, 1.0) // left state
        } else {
            Primitive1D::new(0.125, 0.0, 0.1) // right state
        }
    });

    println!("Initial conditions set: Sod shock tube");
    println!("  Left (x < 0.5):  rho = 1.0, v = 0.0, p = 1.0");
    println!("  Right (x > 0.5): rho = 0.125, v = 0.0, p = 0.1");
    println!();

    // evolve to final time
    println!("Running simulation...");
    let mut step_count = 0;
    while solver.time() < t_final {
        solver.step();
        step_count += 1;

        if step_count % 100 == 0 {
            println!("  step {}: t = {:.6}", step_count, solver.time());
        }
    }

    println!("Simulation complete!");
    println!("  total steps: {}", step_count);
    println!("  final time: {:.6}", solver.time());
    println!();

    // get solution
    let solution = solver.get_solution();
    let x = solver.cell_centers();

    // print solution at selected points
    println!("Solution at t = {:.3}:", solver.time());
    println!("  x        rho      vx       p");
    println!("  -----------------------------------");

    let indices = [0, 50, 100, 150, 199];
    for &i in &indices {
        let prim = solution[i];
        println!(
            "  {:.3}    {:.4}   {:.4}   {:.4}",
            x[i], prim.rho, prim.vx, prim.p
        );
    }
    println!();

    // check for expected features
    println!("Validation checks:");

    // left state should be near initial
    let left_state = solution[10];
    println!("  Left state (x=0.05): rho={:.4}", left_state.rho);
    if (left_state.rho - 1.0).abs() < 0.1 {
        println!("    ✓ Left state preserved");
    } else {
        println!("    ✗ Left state error");
    }

    // contact discontinuity should show density jump
    let mid_left = solution[95];
    let mid_right = solution[105];
    let density_jump = (mid_left.rho - mid_right.rho).abs();
    println!("  Contact discontinuity density jump: {:.4}", density_jump);
    if density_jump > 0.1 {
        println!("    ✓ Contact discontinuity captured");
    } else {
        println!("    ✗ Contact discontinuity weak");
    }

    // right state should show expansion
    let right_state = solution[190];
    println!("  Right state (x=0.95): rho={:.4}", right_state.rho);
    if right_state.rho < 0.3 {
        println!("    ✓ Right state expanded");
    } else {
        println!("    ✗ Right state error");
    }

    println!();
    println!("=== Test Complete ===");
}
