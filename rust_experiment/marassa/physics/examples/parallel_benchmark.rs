// =============================================================================
// parallel_benchmark.rs
//
// benchmark comparing serial vs parallel euler solvers.
//
// run:
//   cargo run --release --example parallel_benchmark
// =============================================================================

use physics::hydro::{ParallelEuler1D, SimpleEuler1D};

fn main() {
    println!("=============================================================================");
    println!("Serial vs Parallel Benchmark");
    println!("=============================================================================\n");

    let nthreads = std::thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(4);

    println!("Available threads: {}\n", nthreads);

    let sizes = [100_000, 1_000_000];
    let nsteps = 500;

    for &ncells in &sizes {
        println!("-----------------------------------------------------------------------------");
        println!("Grid Size: {} cells, {} steps", ncells, nsteps);
        println!("-----------------------------------------------------------------------------\n");

        // serial
        {
            let mut solver = SimpleEuler1D::new(ncells, 0.0, 1.0, 1.4, 0.5);
            solver.set_ic(|x| {
                if x < 0.5 {
                    (1.0, 0.0, 1.0)
                } else {
                    (0.125, 0.0, 0.1)
                }
            });

            for _ in 0..nsteps {
                solver.step();
            }

            if let Some((zps, wall, _)) = solver.stats() {
                println!("  Serial:     {:.2e} zones/sec  ({:.3}s)", zps, wall);
            }
        }

        // parallel
        {
            let mut solver = ParallelEuler1D::new(ncells, 0.0, 1.0, 1.4, 0.5, nthreads);
            solver.set_ic(|x| {
                if x < 0.5 {
                    (1.0, 0.0, 1.0)
                } else {
                    (0.125, 0.0, 0.1)
                }
            });

            for _ in 0..nsteps {
                solver.step();
            }

            if let Some((zps, wall, _)) = solver.stats() {
                println!(
                    "  Parallel:   {:.2e} zones/sec  ({:.3}s)  [{} threads]",
                    zps, wall, nthreads
                );
            }
        }

        println!();
    }

    println!("=============================================================================");
}
