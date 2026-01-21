// =============================================================================
// simple_1d.rs
//
// minimal demonstration of the new worldstate architecture.
// shows:
//   - pure functional state management
//   - soa field storage
//   - device-agnostic computation
//   - clean separation of concerns
//
// run:
//   cargo run --example simple_1d
// =============================================================================

use sim::{PhysicsConfig, WorldState};
use xpu_core::Device;
use xpu_host::CpuDevice;

// marker types for compile-time dispatch
struct Newtonian;
struct Hlle;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Simple 1D Euler Example ===\n");

    // initialize device
    let device = CpuDevice::new(0)?;
    println!("device: CPU (host)");

    // physics configuration
    let config = PhysicsConfig::new(
        [0.0], // x_min
        [1.0], // x_max
        [100], // n_cells
        1.4,   // gamma
    );

    println!("domain: [{}, {}]", config.x_min[0], config.x_max[0]);
    println!("resolution: {} cells", config.n_cells[0]);
    println!("gamma: {}", config.gamma);
    println!("dx: {}", config.dx[0]);
    println!();

    // create world state (the god object)
    let mut world = WorldState::<Newtonian, Hlle, _, 1>::single_device(&device, config)?;

    println!("world state initialized:");
    println!("  partitions: {}", world.num_partitions());
    println!("  total cells: {}", world.total_cells());
    println!("  time: {}", world.time);
    println!();

    // set initial conditions (sod shock tube)
    println!("setting initial conditions (sod shock tube)...");
    world.set_initial_conditions(|x| {
        if x[0] < 0.5 {
            // left state: high pressure
            (1.0, [0.0], 1.0) // (rho, vel, p)
        } else {
            // right state: low pressure
            (0.125, [0.0], 0.1)
        }
    })?;
    println!("  left (x < 0.5):  rho=1.0, v=0.0, p=1.0");
    println!("  right (x > 0.5): rho=0.125, v=0.0, p=0.1");
    println!();

    // time evolution parameters
    let cfl = 0.5;
    let t_final = 0.2;
    let mut step_count = 0;

    println!("evolving to t = {}...", t_final);
    println!("cfl = {}", cfl);
    println!();

    // main evolution loop
    while world.time < t_final {
        // compute timestep from cfl condition
        let dt = world.compute_dt(cfl)?;

        // advance one step (pure functional evolution)
        world.step(dt)?;

        step_count += 1;

        if step_count % 10 == 0 {
            println!(
                "  step {}: t = {:.6}, dt = {:.6}",
                step_count, world.time, dt
            );
        }
    }

    println!();
    println!("simulation complete!");
    println!("  total steps: {}", step_count);
    println!("  final time: {:.6}", world.time);
    println!();

    // extract solution to host for analysis
    println!("extracting solution...");
    let solution = world.to_host()?;

    // sample solution at a few points
    println!("solution samples:");
    println!("  i     x       rho      v        p");
    println!("  ---------------------------------------");

    let indices = [0, 25, 50, 75, 99];
    for &i in &indices {
        let x = config.cell_center([i as i64])[0];
        let (rho, vel, p) = solution.primitive_at(i, config.gamma);
        println!(
            "  {:3}  {:.3}   {:.4}   {:.4}   {:.4}",
            i, x, rho, vel[0], p
        );
    }

    println!();
    println!("=== Example Complete ===");
    println!();
    println!("architecture summary:");
    println!("  ✓ pure functional state (worldstate)");
    println!("  ✓ soa field storage (coalesced memory)");
    println!("  ✓ device-agnostic (works on cpu/gpu)");
    println!("  ✓ compile-time dispatch (zero overhead)");
    println!("  ✓ mathematical clarity (Φ: state → state)");

    Ok(())
}
