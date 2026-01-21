// =============================================================================
// godunov_2d_kh.rs
//
// production 2d godunov solver - kelvin-helmholtz instability.
// demonstrates full 2d capabilities with plm reconstruction.
//
// problem: shear layer instability
//   - domain: [0,1] x [0,1]
//   - shear layer at y=0.5
//   - velocity perturbation to seed instability
//   - periodic in x, reflecting in y
//
// physics:
//   - top half: rho=2, vx=0.5, vy=0, p=2.5
//   - bottom half: rho=1, vx=-0.5, vy=0, p=2.5
//   - perturbation: vy = 0.1 * sin(4πx) * exp(-((y-0.5)/0.05)²)
//
// expected:
//   - roll-up of vortices
//   - demonstrates 2d capability and shock capturing
//
// usage:
//   cargo run --package physics --example godunov_2d_kh --release
// =============================================================================

use compute::reconstruction::Limiter;
use compute::{Domain, Field};
use physics::hydro::{
    compute_dt, compute_fluxes_newtonian, cons2prim_newtonian, euler_step, Primitive,
};
use xpu_core::Device;
use xpu_host::CpuDevice;

const GAMMA: f64 = 1.4;
const CFL: f64 = 0.4;
const T_FINAL: f64 = 0.5;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== 2d godunov solver: kelvin-helmholtz instability ===\n");

    let device = CpuDevice::new(0)?;
    println!("device: cpu");

    // domain setup
    let nx: usize = 32;
    let ny: usize = 32;
    let nghosts: usize = 2; // plm needs 2 ghost zones
    let nx_total = nx + 2 * nghosts;
    let ny_total = ny + 2 * nghosts;

    let x_min = 0.0;
    let x_max = 1.0;
    let y_min = 0.0;
    let y_max = 1.0;
    let dx = (x_max - x_min) / nx as f64;
    let dy = (y_max - y_min) / ny as f64;

    let cell_domain = Domain::from_shape([nx_total as i64, ny_total as i64]);
    let face_domains = [
        Domain::new([0, 0], [nx_total as i64 + 1, ny_total as i64]),
        Domain::new([0, 0], [nx_total as i64, ny_total as i64 + 1]),
    ];

    // boundary specification
    use physics::hydro::{BoundarySpec, BoundaryType};
    let mut boundary_spec = BoundarySpec::<2>::outflow([nghosts, nghosts]);
    // periodic in x, reflecting in y
    boundary_spec.set(0, physics::hydro::Side::Left, BoundaryType::Periodic);
    boundary_spec.set(0, physics::hydro::Side::Right, BoundaryType::Periodic);
    boundary_spec.set(1, physics::hydro::Side::Left, BoundaryType::Reflect);
    boundary_spec.set(1, physics::hydro::Side::Right, BoundaryType::Reflect);

    println!("grid: {}x{}, dx={:.4}, dy={:.4}", nx, ny, dx, dy);
    println!("time: t_final = {}", T_FINAL);
    println!("reconstruction: plm with minmod limiter");
    println!("boundary: periodic (x), reflecting (y)");
    println!();

    // allocate fields
    let mut den = Field::<f64, _, 2>::zeros(&device, cell_domain)?;
    let mut mom = [
        Field::<f64, _, 2>::zeros(&device, cell_domain)?,
        Field::<f64, _, 2>::zeros(&device, cell_domain)?,
    ];
    let mut nrg = Field::<f64, _, 2>::zeros(&device, cell_domain)?;

    let mut rho = Field::<f64, _, 2>::zeros(&device, cell_domain)?;
    let mut vel = [
        Field::<f64, _, 2>::zeros(&device, cell_domain)?,
        Field::<f64, _, 2>::zeros(&device, cell_domain)?,
    ];
    let mut pre = Field::<f64, _, 2>::zeros(&device, cell_domain)?;

    let mut flux_den = [
        Field::<f64, _, 2>::zeros(&device, face_domains[0])?,
        Field::<f64, _, 2>::zeros(&device, face_domains[1])?,
    ];
    let mut flux_mom = [
        [
            Field::<f64, _, 2>::zeros(&device, face_domains[0])?,
            Field::<f64, _, 2>::zeros(&device, face_domains[0])?,
        ],
        [
            Field::<f64, _, 2>::zeros(&device, face_domains[1])?,
            Field::<f64, _, 2>::zeros(&device, face_domains[1])?,
        ],
    ];
    let mut flux_nrg = [
        Field::<f64, _, 2>::zeros(&device, face_domains[0])?,
        Field::<f64, _, 2>::zeros(&device, face_domains[1])?,
    ];

    println!("allocated 2d fields");

    // initial conditions: kelvin-helmholtz shear layer
    println!("initializing: shear layer with perturbation");
    let ncells = nx_total * ny_total;
    let mut den_data = vec![0.0; ncells];
    let mut mom_x_data = vec![0.0; ncells];
    let mut mom_y_data = vec![0.0; ncells];
    let mut nrg_data = vec![0.0; ncells];

    for j in 0..ny_total {
        for i in 0..nx_total {
            let idx = j * nx_total + i;
            // physical coordinates (ignore ghost zones for initial conditions)
            let ii = (i as i64 - nghosts as i64).max(0).min(nx as i64 - 1) as usize;
            let jj = (j as i64 - nghosts as i64).max(0).min(ny as i64 - 1) as usize;
            let x = x_min + (ii as f64 + 0.5) * dx;
            let y = y_min + (jj as f64 + 0.5) * dy;

            // shear layer
            let (rho_ic, vx_ic, p_ic) = if y > 0.5 {
                (2.0, 0.5, 2.5) // top
            } else {
                (1.0, -0.5, 2.5) // bottom
            };

            // velocity perturbation to seed instability
            let vy_ic =
                0.1 * (4.0 * std::f64::consts::PI * x).sin() * (-((y - 0.5) / 0.05).powi(2)).exp();

            // convert to conserved
            let prim = Primitive::<physics::hydro::Newtonian, 2>::new(rho_ic, [vx_ic, vy_ic], p_ic);
            let cons = prim.to_conserved(GAMMA);

            den_data[idx] = cons.den;
            mom_x_data[idx] = cons.mom[0];
            mom_y_data[idx] = cons.mom[1];
            nrg_data[idx] = cons.nrg;
        }
    }

    den.from_host(&den_data)?;
    mom[0].from_host(&mom_x_data)?;
    mom[1].from_host(&mom_y_data)?;
    nrg.from_host(&nrg_data)?;

    println!("  top:    rho=2.0, vx=0.5, vy=0, p=2.5");
    println!("  bottom: rho=1.0, vx=-0.5, vy=0, p=2.5");
    println!("  perturbation: vy ~ sin(4πx) * exp(-(y-0.5)²)");
    println!();

    // time integration loop
    let mut time: f64 = 0.0;
    let mut step: usize = 0;
    let output_interval = 50;

    println!("starting time integration...");
    println!("{:>6} {:>12} {:>12}", "step", "time", "dt");
    println!("{}", "-".repeat(36));

    while time < T_FINAL {
        // apply boundary conditions
        physics::hydro::apply_conserved_boundaries_2d(
            &mut den,
            &mut mom,
            &mut nrg,
            &boundary_spec,
            &device,
        )?;

        // cons2prim conversion
        cons2prim_newtonian(
            &den,
            &mom,
            &nrg,
            &mut rho,
            &mut vel,
            &mut pre,
            GAMMA,
            &device,
            cell_domain,
        )
        .map_err(|e| format!("cons2prim error: {:?}", e))?;

        // compute timestep
        let dt = compute_dt(&vel, &pre, &rho, GAMMA, [dx, dy], CFL, cell_domain)?;
        let dt = if time + dt > T_FINAL {
            T_FINAL - time
        } else {
            dt
        };

        if step % output_interval == 0 {
            println!("{:6} {:12.6e} {:12.6e}", step, time, dt);
        }

        // compute fluxes
        compute_fluxes_newtonian(
            &rho,
            &vel,
            &pre,
            &mut flux_den,
            &mut flux_mom,
            &mut flux_nrg,
            GAMMA,
            [dx, dy],
            Limiter::MinMod,
            &device,
            cell_domain,
            &face_domains,
        )?;

        // time integration
        euler_step(
            &mut den,
            &mut mom,
            &mut nrg,
            &flux_den,
            &flux_mom,
            &flux_nrg,
            dt,
            [dx, dy],
            &device,
            cell_domain,
        )?;

        time += dt;
        step += 1;

        if step > 100000 {
            println!("warning: exceeded max steps");
            break;
        }
    }

    println!("{}", "-".repeat(36));
    println!("integration complete:");
    println!("  final time: {:.6}", time);
    println!("  total steps: {}", step);
    println!();

    // final diagnostics
    cons2prim_newtonian(
        &den,
        &mom,
        &nrg,
        &mut rho,
        &mut vel,
        &mut pre,
        GAMMA,
        &device,
        cell_domain,
    )
    .map_err(|e| format!("cons2prim error: {:?}", e))?;

    let rho_final = rho.to_host()?;
    let vx_final = vel[0].to_host()?;
    let vy_final = vel[1].to_host()?;
    let pre_final = pre.to_host()?;

    let mut max_rho: f64 = 0.0;
    let mut min_rho: f64 = f64::MAX;
    let mut max_vel: f64 = 0.0;
    let mut max_pre: f64 = 0.0;
    let mut min_pre: f64 = f64::MAX;

    for i in 0..ncells {
        max_rho = max_rho.max(rho_final[i]);
        min_rho = min_rho.min(rho_final[i]);
        let vel_mag = (vx_final[i].powi(2) + vy_final[i].powi(2)).sqrt();
        max_vel = max_vel.max(vel_mag);
        max_pre = max_pre.max(pre_final[i]);
        min_pre = min_pre.min(pre_final[i]);
    }

    println!("final solution diagnostics:");
    println!("  density:  min = {:.6}, max = {:.6}", min_rho, max_rho);
    println!("  velocity: max = {:.6}", max_vel);
    println!("  pressure: min = {:.6}, max = {:.6}", min_pre, max_pre);
    println!();

    // sample at specific locations
    println!("solution samples (center line y=0.5):");
    println!(
        "{:>6} {:>12} {:>12} {:>12} {:>12} {:>12}",
        "i", "x", "rho", "vx", "vy", "pre"
    );
    println!("{}", "-".repeat(72));

    let j_mid = ny_total / 2;
    for &i in &[
        nx_total / 8,
        nx_total / 4,
        nx_total / 2,
        3 * nx_total / 4,
        7 * nx_total / 8,
    ] {
        let idx = j_mid * nx_total + i;
        let x = x_min + (i as f64 + 0.5) * dx;
        println!(
            "{:6} {:12.4} {:12.6} {:12.6} {:12.6} {:12.6}",
            i, x, rho_final[idx], vx_final[idx], vy_final[idx], pre_final[idx]
        );
    }

    println!();
    println!("expected physics:");
    println!("  - vortex roll-up from shear instability");
    println!("  - mixing between top and bottom fluids");
    println!("  - complex turbulent structures");
    println!("  - density should show rolled-up features");
    println!();

    // compute kinetic energy
    let mut total_ke: f64 = 0.0;
    for i in 0..ncells {
        let ke = 0.5 * rho_final[i] * (vx_final[i].powi(2) + vy_final[i].powi(2));
        total_ke += ke;
    }
    total_ke *= dx * dy;

    println!("global diagnostics:");
    println!("  total kinetic energy: {:.6e}", total_ke);
    println!();

    println!("=== 2d solver demonstration complete ===");

    Ok(())
}
