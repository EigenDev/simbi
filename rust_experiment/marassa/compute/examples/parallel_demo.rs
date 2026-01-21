// =============================================================================
// parallel_demo.rs
//
// demonstrates parallel cpu evaluation using rayon.
// compares serial vs parallel performance on large domains.
//
// shows:
//   - parallel evaluation with ParCpuDevice
//   - performance comparison (serial vs parallel)
//   - runtime device selection
//   - parallel reduce operations
// =============================================================================

use compute::{constant, evaluate, from_fn, parallel_evaluate, Domain};
use std::time::Instant;
use xpu_core::Device;
use xpu_host::{CpuDevice, ParCpuDevice};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== parallel cpu evaluation demonstration ===\n");

    // example 1: basic parallel evaluation
    println!("example 1: basic parallel evaluation");
    {
        let par_cpu = ParCpuDevice::new_parallel(0)?;
        let domain = Domain::from_shape([100, 100]);

        let x = from_fn(domain, |coord| coord[0] as f64);
        let y = from_fn(domain, |coord| coord[1] as f64);
        let sum = x.add(y);

        let buf = parallel_evaluate(&par_cpu, sum)?;
        let mut data = vec![0.0; domain.size()];
        par_cpu.copy_to_host_par(&buf, &mut data)?;

        println!("  evaluated 100x100 domain in parallel");
        println!("  data[50,50] = {} (expected: 100)", data[5050]);
    }

    // example 2: performance comparison
    println!("\nexample 2: performance comparison (serial vs parallel)");
    {
        let sizes = vec![(50, 50), (100, 100), (200, 200), (400, 400), (800, 800)];

        println!(
            "\n  {:>10} {:>15} {:>15} {:>10}",
            "size", "serial (ms)", "parallel (ms)", "speedup"
        );
        println!("  {}", "-".repeat(55));

        for (nx, ny) in sizes {
            let domain = Domain::from_shape([nx, ny]);

            // complex computation: r^2 = x^2 + y^2
            let create_computation = || {
                let x = from_fn(domain, |coord| coord[0] as f64);
                let y = from_fn(domain, |coord| coord[1] as f64);
                let x_copy = from_fn(domain, |coord| coord[0] as f64);
                let y_copy = from_fn(domain, |coord| coord[1] as f64);
                let x_sq = x.mul(x_copy);
                let y_sq = y.mul(y_copy);
                x_sq.add(y_sq)
            };

            // serial timing
            let cpu = CpuDevice::new(0)?;
            let start = Instant::now();
            let _field = evaluate(&cpu, create_computation())?;
            let serial_time = start.elapsed();

            // parallel timing
            let par_cpu = ParCpuDevice::new_parallel(0)?;
            let start = Instant::now();
            let _buf = parallel_evaluate(&par_cpu, create_computation())?;
            let parallel_time = start.elapsed();

            let speedup = serial_time.as_secs_f64() / parallel_time.as_secs_f64();

            println!(
                "  {:>4}x{:<4} {:>12.2} {:>12.2} {:>9.2}x",
                nx,
                ny,
                serial_time.as_secs_f64() * 1000.0,
                parallel_time.as_secs_f64() * 1000.0,
                speedup
            );
        }
    }

    // example 3: parallel reduce operations
    println!("\nexample 3: parallel reduce operations");
    {
        use xpu_core::reduce::{Max, Min, Sum};

        let par_cpu = ParCpuDevice::new_parallel(0)?;
        let _domain = Domain::from_shape([1000]);

        let data: Vec<i32> = (1..=1000).collect();
        let mut buf = par_cpu.alloc_par::<i32>(1000)?;
        par_cpu.copy_to_device_par(&data, &mut buf)?;

        let total = par_cpu.reduce_par(&buf, Sum)?;
        let max_val = par_cpu.reduce_par(&buf, Max)?;
        let min_val = par_cpu.reduce_par(&buf, Min)?;

        println!("  sum of 1..=1000: {} (expected: 500500)", total);
        println!("  max: {} (expected: 1000)", max_val);
        println!("  min: {} (expected: 1)", min_val);
    }

    // example 4: runtime device selection
    println!("\nexample 4: runtime device selection");
    {
        let use_parallel = std::env::var("PARALLEL").is_ok();
        let domain = Domain::from_shape([200, 200]);

        let comp = constant(domain, 42.0);

        if use_parallel {
            println!("  using parallel cpu device (PARALLEL env set)");
            let par_cpu = ParCpuDevice::new_parallel(0)?;
            let buf = parallel_evaluate(&par_cpu, comp)?;
            let mut data = vec![0.0; domain.size()];
            par_cpu.copy_to_host_par(&buf, &mut data)?;
            println!("  result: {} elements, all = {}", data.len(), data[0]);
        } else {
            println!("  using serial cpu device (default)");
            let cpu = CpuDevice::new(0)?;
            let field = evaluate(&cpu, comp)?;
            let data = field.to_host()?;
            println!("  result: {} elements, all = {}", data.len(), data[0]);
            println!("  tip: run with `PARALLEL=1 cargo run --example parallel_demo`");
        }
    }

    // example 5: large-scale scientific computation
    println!("\nexample 5: large-scale domain (1024x1024)");
    {
        let par_cpu = ParCpuDevice::new_parallel(0)?;
        let domain = Domain::from_shape([1024, 1024]);

        println!("  computing gaussian: exp(-(x² + y²)/2σ²)");

        let sigma = 100.0;
        let x = from_fn(domain, |coord| (coord[0] - 512) as f64);
        let y = from_fn(domain, |coord| (coord[1] - 512) as f64);
        let x_copy = from_fn(domain, |coord| (coord[0] - 512) as f64);
        let y_copy = from_fn(domain, |coord| (coord[1] - 512) as f64);
        let x_sq = x.mul(x_copy);
        let y_sq = y.mul(y_copy);
        let r_sq = x_sq.add(y_sq);
        let gaussian = r_sq.map(|r2| (-r2 / (2.0 * sigma * sigma)).exp());

        let start = Instant::now();
        let buf = parallel_evaluate(&par_cpu, gaussian)?;
        let elapsed = start.elapsed();

        println!(
            "  evaluated {} elements in {:.2} ms",
            domain.size(),
            elapsed.as_secs_f64() * 1000.0
        );

        // get center value (at coordinate [512, 512])
        let center_idx = domain.coord_to_linear([512, 512]);
        let mut data = vec![0.0; domain.size()];
        par_cpu.copy_to_host_par(&buf, &mut data)?;
        println!("  center value: {:.6} (expected ≈ 1.0)", data[center_idx]);
    }

    println!("\n=== demonstration complete ===");
    println!("\nkey takeaways:");
    println!("  ✓ parallel cpu device uses rayon for multi-threaded execution");
    println!("  ✓ speedup scales with problem size (larger = better)");
    println!("  ✓ works with Send + Sync types");
    println!("  ✓ runtime choice: serial for debugging, parallel for performance");
    println!("  ✓ parallel reduce operations available");

    Ok(())
}
