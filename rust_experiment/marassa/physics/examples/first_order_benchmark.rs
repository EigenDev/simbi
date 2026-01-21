// =============================================================================
// first_order_benchmark.rs
//
// first-order benchmark: euler timestepping + pcm reconstruction
// apples-to-apples comparison with c++ baseline performance.
//
// configuration:
//   - euler time stepping (rk1, order=1)
//   - pcm reconstruction (no stencils)
//   - no slope limiters
//   - serial cpu execution
//
// this is the simplest possible scheme for pure performance measurement.
//
// run:
//   cargo run --release --example first_order_benchmark
// =============================================================================

use physics::hydro::{BoundaryCondition, Euler1DSolver, ExecutionMode, Primitive1D};

fn main() {
    println!("=============================================================================");
    println!("First-Order Benchmark: Euler + PCM");
    println!("=============================================================================\n");

    println!("Configuration:");
    println!("  Time integration: Euler (RK1, first-order)");
    println!("  Reconstruction: PCM (piecewise constant)");
    println!("  Riemann solver: HLLE");
    println!("  Test case: Sod shock tube");
    println!("  Boundary: Outflow");
    println!("  CFL: 0.5");
    println!();

    // test different grid sizes
    let ncells_list = [100, 1000, 10000, 100000, 1000000];
    let max_steps = 5000;
    let report_interval = 100;

    for &ncells in &ncells_list {
        println!("-----------------------------------------------------------------------------");
        println!("Grid Size: {} cells", ncells);
        println!("-----------------------------------------------------------------------------");

        // create first-order solver
        let mut solver = Euler1DSolver::new(
            ncells,
            0.0,
            1.0,
            1.4, // gamma
            0.5, // cfl
            BoundaryCondition::Outflow,
            1, // rk1 (euler timestepping)
            ExecutionMode::Serial,
        );

        // sod shock tube initial condition
        solver.set_initial_conditions(|x| {
            if x < 0.5 {
                Primitive1D::new(1.0, 0.0, 1.0) // left state
            } else {
                Primitive1D::new(0.125, 0.0, 0.1) // right state
            }
        });

        // run with profiling
        solver.evolve_with_profiling(0.2, report_interval, max_steps);

        // get final stats
        if let Some(stats) = solver.get_stats() {
            println!(
                "\n>>> RESULT: {} cells -> {:.2e} zone-cycles/sec <<<",
                ncells, stats.zone_cycles_per_second
            );
            println!("    Time per step: {:.6} ms", stats.time_per_step_ms);
            println!("    Total steps: {}", stats.steps);
            println!("    Total time: {:.3} s", stats.wall_time);
        }

        println!();
    }

    println!("=============================================================================");
    println!("Benchmark Complete");
    println!("=============================================================================");
    println!("\nNote: This is a first-order method for performance comparison only.");
    println!("      For production, use RK3 + PLM for accuracy.\n");
}
