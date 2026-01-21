// =============================================================================
// godunov_workflow.rs
//
// comprehensive example demonstrating the lazy computation workflow
// for building godunov-type solvers.
//
// shows:
//   - domain creation and manipulation
//   - lazy expression building (no execution)
//   - field creation and initialization
//   - computation from fields
//   - evaluation (lazy -> eager)
//   - full workflow: field -> computation -> transform -> field
// =============================================================================

use compute::{evaluate, from_fn, Domain, Field};
use xpu_core::Device;
use xpu_host::CpuDevice;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== godunov workflow demonstration ===\n");

    // initialize device
    let device = CpuDevice::new(0)?;
    println!("device initialized: cpu");

    // example 1: domain operations
    println!("\nexample 1: domain topology");
    let domain = Domain::new([0, 0], [100, 100]);
    println!("  full domain: [{:?}, {:?})", domain.start, domain.end);
    println!("  shape: {:?}", domain.shape());
    println!("  size: {} elements", domain.size());

    let interior = domain.contract(2);
    println!(
        "  interior (contracted by 2): [{:?}, {:?})",
        interior.start, interior.end
    );

    // example 2: lazy computation building
    println!("\nexample 2: lazy expression graphs (no execution yet)");

    let small_domain = Domain::from_shape([10, 10]);

    // build expressions without executing
    let x_coord = from_fn(small_domain, |coord| coord[0] as f64);
    let y_coord = from_fn(small_domain, |coord| coord[1] as f64);

    println!("  created x and y coordinate functions");

    // compose: r^2 = x^2 + y^2
    let x_coord_dup = from_fn(small_domain, |coord| coord[0] as f64);
    let y_coord_dup = from_fn(small_domain, |coord| coord[1] as f64);
    let x_sq = x_coord.mul(x_coord_dup);
    let y_sq = y_coord.mul(y_coord_dup);
    let r_squared = x_sq.add(y_sq);

    println!("  composed: r² = x² + y²");
    println!("  still no execution - pure lazy graph");

    // can evaluate at single points without materializing
    let value_at_3_4 = r_squared.eval([3, 4]);
    println!("  lazy eval at [3,4]: {} (should be 25)", value_at_3_4);

    // example 3: field initialization
    println!("\nexample 3: field creation and initialization");

    let field_domain = Domain::from_shape([50, 50]);

    let rho = Field::<f64, _, 2>::filled(&device, field_domain, 1.0)?;
    println!("  created density field ρ, filled with 1.0");

    let _pressure = Field::<f64, _, 2>::zeros(&device, field_domain)?;
    println!("  created pressure field p, initialized to 0");

    // verify
    let rho_data = rho.to_host()?;
    println!("  verified: all ρ values = {}", rho_data[0]);

    // example 4: field -> computation -> transform -> field
    println!("\nexample 4: full lazy evaluation workflow");

    let workflow_domain = Domain::from_shape([20, 20]);

    // step 1: create initial field
    let mut initial_field = Field::<f64, _, 2>::zeros(&device, workflow_domain)?;
    let init_data: Vec<f64> = (0..400).map(|i| (i as f64) / 10.0).collect();
    initial_field.from_host(&init_data)?;
    println!("  step 1: created field with initial data");

    // step 2: convert field to lazy computation
    let view = initial_field.view();
    let comp = view.as_computation();
    println!("  step 2: field -> computation (lazy)");

    // step 3: transform lazily
    let transformed = comp.scale(2.0).offset(5.0); // 2*u + 5
    println!("  step 3: applied transform: 2*u + 5 (still lazy)");

    // step 4: materialize result
    let result_field = evaluate(&device, transformed)?;
    println!("  step 4: evaluated -> new field");

    // verify transformation
    let result_data = result_field.to_host()?;
    let original_first = init_data[0];
    let transformed_first = result_data[0];
    println!(
        "  verification: {} -> {} (should be {})",
        original_first,
        transformed_first,
        2.0 * original_first + 5.0
    );

    // example 5: godunov-like update pattern
    println!("\nexample 5: godunov-style time step");

    let nx = 10;
    let ny = 10;
    let godunov_domain = Domain::from_shape([nx, ny]);

    // initial conditions
    let mut u_n = Field::<f64, _, 2>::zeros(&device, godunov_domain)?;
    let u_data: Vec<f64> = (0..(nx * ny) as usize)
        .map(|i| ((i % nx as usize) as f64).sin())
        .collect();
    u_n.from_host(&u_data)?;
    println!("  initialized u^n with sine wave");

    // compute right-hand side (simplified)
    let u_view = u_n.view();
    let u_comp = u_view.as_computation();

    // du/dt = -u (decay for demo)
    let dt = 0.01;
    let u_comp_dup = u_view.as_computation();
    let dudt = u_comp_dup.scale(-1.0);
    let update = u_comp.add(dudt.scale(dt)); // u^{n+1} = u^n + dt * du/dt

    println!("  built update: u^{{n+1}} = u^n + dt*(-u)");

    // evaluate to get next timestep
    let u_next = evaluate(&device, update)?;
    println!("  evaluated update -> u^{{n+1}}");

    let u_next_data = u_next.to_host()?;
    println!(
        "  timestep complete: u[0] = {:.6} -> {:.6}",
        u_data[0], u_next_data[0]
    );

    // example 6: multi-field operations (density + energy)
    println!("\nexample 6: multi-field coupled system");

    let multi_domain = Domain::from_shape([8, 8]);

    let rho_field = Field::<f64, _, 2>::filled(&device, multi_domain, 1.0)?;
    let e_field = Field::<f64, _, 2>::filled(&device, multi_domain, 2.5)?;

    // p = (γ - 1) * ρ * e  (ideal gas)
    let gamma = 1.4;
    let rho_comp = rho_field.view().as_computation();
    let e_comp = e_field.view().as_computation();

    let pressure_comp = rho_comp.mul(e_comp).scale(gamma - 1.0);

    println!("  built: p = (γ-1)*ρ*e with γ={}", gamma);

    let pressure_field = evaluate(&device, pressure_comp)?;
    let p_data = pressure_field.to_host()?;

    println!(
        "  evaluated: p = {:.2} (expected: {:.2})",
        p_data[0],
        (gamma - 1.0) * 1.0 * 2.5
    );

    // example 7: stencil-like operations via remap
    println!("\nexample 7: coordinate remapping (stencil preparation)");

    let stencil_domain = Domain::from_shape([5]);
    let mut field_1d = Field::<f64, _, 1>::zeros(&device, stencil_domain)?;
    field_1d.from_host(&[1.0, 2.0, 3.0, 4.0, 5.0])?;

    let view_1d = field_1d.view();
    let u = view_1d.as_computation();
    let u_copy = view_1d.as_computation();

    // shift right by 1 (u[i] becomes u[i+1], with boundary handling needed)
    let shifted = u_copy.remap(|coord| {
        let i = coord[0];
        // clamp to domain
        let new_i = if i + 1 >= 5 { 4 } else { i + 1 };
        [new_i]
    });

    println!("  created right-shifted view");
    println!(
        "  original[0] = {}, shifted[0] = {} (should be original[1])",
        u.eval([0]),
        shifted.eval([0])
    );

    println!("\n=== workflow demonstration complete ===");
    println!("\nkey takeaways:");
    println!("  ✓ domains define pure topology");
    println!("  ✓ computations are lazy expression graphs");
    println!("  ✓ fields own data on devices");
    println!("  ✓ evaluate() materializes lazy -> eager");
    println!("  ✓ full separation: what (computation) vs where (device)");
    println!("\nready for godunov solver implementation!");

    Ok(())
}
