// =============================================================================
// performance_comparison.rs
//
// compares performance of 1d euler solver across different execution modes:
//   - serial cpu (baseline)
//   - parallel cpu (rayon)
//   - metal gpu (future)
//
// measures zone-cycles/second and reports every 100 iterations.
// runs for 5000 iterations or t=0.2, whichever comes first.
//
// run:
//   cargo run --release --example performance_comparison
// =============================================================================

use physics::hydro::{BoundaryCondition, Euler1DSolver, ExecutionMode, Primitive1D};

fn main() {
    println!("=============================================================================");
    println!("1D Euler Solver Performance Comparison");
    println!("=============================================================================\n");

    // test parameters
    let ncells_list = [100, 500, 1000, 5000, 10000, 50000, 100000, 500000];
    let max_steps = 5000;
    let report_interval = 100;
    let t_final = 0.2;

    println!("Configuration:");
    println!("  Max steps: {}", max_steps);
    println!("  Report interval: every {} steps", report_interval);
    println!("  Target time: t = {}", t_final);
    println!("  Boundary conditions: Outflow");
    println!("  Time integrator: RK3");
    println!("  Reconstruction: PLM with MinMod limiter");
    println!("  Test case: Sod shock tube\n");

    // test each grid size with different execution modes
    for &ncells in &ncells_list {
        println!("=============================================================================");
        println!("Grid Size: {} cells", ncells);
        println!("=============================================================================\n");

        // serial cpu
        let serial_stats = run_test(
            "Serial CPU",
            ncells,
            t_final,
            max_steps,
            report_interval,
            ExecutionMode::Serial,
        );

        // parallel cpu
        let parallel_stats = run_test(
            "Parallel CPU (Rayon)",
            ncells,
            t_final,
            max_steps,
            report_interval,
            ExecutionMode::ParallelCpu,
        );

        // compare
        if let (Some(serial), Some(parallel)) = (serial_stats, parallel_stats) {
            let speedup = parallel.zone_cycles_per_second / serial.zone_cycles_per_second;
            println!("\n>>> SPEEDUP: {:.2}x (Parallel vs Serial) <<<\n", speedup);
        }
    }

    println!("\n=============================================================================");
    println!("Performance Summary Complete");
    println!("=============================================================================\n");
    println!("Note: Metal GPU implementation coming soon!");
    println!("      XPU Metal backend integration in progress.\n");
}

fn run_test(
    name: &str,
    ncells: usize,
    t_final: f64,
    max_steps: usize,
    report_interval: usize,
    execution_mode: ExecutionMode,
) -> Option<physics::hydro::PerformanceStats> {
    println!("-----------------------------------------------------------------------------");
    println!("{}: {} cells", name, ncells);
    println!("-----------------------------------------------------------------------------");

    // create solver
    let mut solver = Euler1DSolver::new(
        ncells,
        0.0,
        1.0,
        1.4, // gamma
        0.5, // cfl
        BoundaryCondition::Outflow,
        3, // rk3
        execution_mode,
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
    solver.evolve_with_profiling(t_final, report_interval, max_steps);

    // get final stats
    let stats = solver.get_stats();
    if let Some(ref s) = stats {
        println!(
            "\n>>> RESULT [{}]: {} cells -> {:.2e} zone-cycles/sec <<<",
            name, ncells, s.zone_cycles_per_second
        );
    }

    println!();
    stats
}
