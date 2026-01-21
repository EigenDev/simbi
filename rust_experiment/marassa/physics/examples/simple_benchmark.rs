// =============================================================================
// simple_benchmark.rs
//
// minimal first-order benchmark matching c++ baseline performance.
// euler timestepping + pcm reconstruction + zero allocations.
//
// this is the simplest possible solver for apples-to-apples comparison.
//
// run:
//   cargo run --release --example simple_benchmark
// =============================================================================

use physics::hydro::SimpleEuler1D;

fn main() {
    println!("=============================================================================");
    println!("Simple First-Order Benchmark (Matching C++ Baseline)");
    println!("=============================================================================\n");

    println!("Configuration:");
    println!("  Time integration: Euler (first-order)");
    println!("  Reconstruction: PCM (piecewise constant)");
    println!("  Riemann solver: HLLE");
    println!("  Allocations: Zero in hot path");
    println!("  Test case: Sod shock tube");
    println!("  Boundary: Outflow");
    println!("  CFL: 0.5\n");

    // test different grid sizes
    let sizes = [100, 1000, 10000, 100000, 1000000];

    println!("Running benchmarks...\n");

    for &ncells in &sizes {
        println!("-----------------------------------------------------------------------------");
        println!("Grid Size: {} cells", ncells);
        println!("-----------------------------------------------------------------------------");

        // create solver
        let mut solver = SimpleEuler1D::new(
            ncells, 0.0, 1.0, 1.4, // gamma
            0.5, // cfl
        );

        // sod shock tube
        solver.set_ic(|x| {
            if x < 0.5 {
                (1.0, 0.0, 1.0) // left: rho, vx, p
            } else {
                (0.125, 0.0, 0.1) // right
            }
        });

        // run 1000 steps
        let max_steps = 1000;
        for _ in 0..max_steps {
            solver.step();
        }

        // report performance
        if let Some((zone_cycles_per_sec, wall_time, steps)) = solver.stats() {
            println!("  Steps completed: {}", steps);
            println!("  Wall time: {:.3} s", wall_time);
            println!("  Zone-cycles/sec: {:.2e}", zone_cycles_per_sec);
            println!(
                "  Time per step: {:.6} ms",
                wall_time * 1000.0 / steps as f64
            );
            println!(
                "\n>>> {} cells -> {:.2e} zone-cycles/sec <<<\n",
                ncells, zone_cycles_per_sec
            );
        }
    }

    println!("=============================================================================");
    println!("Benchmark Complete");
    println!("=============================================================================\n");
}
