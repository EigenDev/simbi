// =============================================================================
// godunov_2d_simple.rs
//
// simple 2d godunov test - 1d riemann problem in 2d domain.
// validates 2d infrastructure with known solution.
//
// problem: sod shock tube along x-axis, uniform in y
//   - left state: rho=1, v=0, p=1
//   - right state: rho=0.125, v=0, p=0.1
//   - interface at x=0.5
//   - uniform in y direction
//
// expected:
//   - same as 1d sod shock tube
//   - validates 2d indexing and boundary conditions
//
// usage:
//   cargo run --package physics --example godunov_2d_simple --release
// =============================================================================

use compute::reconstruction::Limiter;
use compute::{Domain, Field};
use physics::hydro::{
    apply_conserved_boundaries_2d, compute_dt, compute_fluxes_newtonian, cons2prim_newtonian,
    euler_step, BoundarySpec, BoundaryType, Primitive,
};
use xpu_core::Device;
use xpu_host::CpuDevice;

const GAMMA: f64 = 1.4;
const CFL: f64 = 0.4;
const T_FINAL: f64 = 0.2;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== 2d godunov solver: simple riemann test ===\n");

    let device = CpuDevice::new(0)?;
    println!("device: cpu");

    // domain setup
    let nx: usize = 100;
    let ny: usize = 10;
    let nghosts: usize = 2;
    let nx_total = nx + 2 * nghosts;
    let ny_total = ny + 2 * nghosts;

    let x_min = 0.0;
    let x_max = 1.0;
    let y_min = 0.0;
    let y_max = 0.1;
    let dx = (x_max - x_min) / nx as f64;
    let dy = (y_max - y_min) / ny as f64;

    let cell_domain = Domain::from_shape([nx_total as i64, ny_total as i64]);
    let face_domains = [
        Domain::new([0, 0], [nx_total as i64 + 1, ny_total as i64]),
        Domain::new([0, 0], [nx_total as i64, ny_total as i64 + 1]),
    ];

    // boundary specification: outflow on all sides
    let boundary_spec = BoundarySpec::<2>::outflow([nghosts, nghosts]);

    println!("grid: {}x{}, dx={:.4}, dy={:.4}", nx, ny, dx, dy);
    println!("time: t_final = {}", T_FINAL);
    println!("reconstruction: plm with minmod limiter");
    println!("boundary: outflow on all sides");
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

    // initial conditions: sod shock tube along x-axis
    println!("initializing: sod shock tube (1d in 2d)");
    let ncells = nx_total * ny_total;
    let mut den_data = vec![0.0; ncells];
    let mut mom_x_data = vec![0.0; ncells];
    let mut mom_y_data = vec![0.0; ncells];
    let mut nrg_data = vec![0.0; ncells];

    for j in 0..ny_total {
        for i in 0..nx_total {
            let idx = j * nx_total + i;

            // physical x-coordinate (accounting for ghost zones)
            let i_phys = if i < nghosts {
                0
            } else if i >= nx + nghosts {
                nx - 1
            } else {
                i - nghosts
            };

            let x = x_min + (i_phys as f64 + 0.5) * dx;

            // sod shock tube initial condition
            let (rho_ic, vx_ic, vy_ic, p_ic) = if x < 0.5 {
                (1.0, 0.0, 0.0, 1.0) // left state
            } else {
                (0.125, 0.0, 0.0, 0.1) // right state
            };

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

    println!("  left:  rho=1.0, v=0, p=1.0 (x < 0.5)");
    println!("  right: rho=0.125, v=0, p=0.1 (x > 0.5)");
    println!("  uniform in y-direction");
    println!();

    // time integration loop
    let mut time: f64 = 0.0;
    let mut step: usize = 0;
    let output_interval = 10;

    println!("starting time integration...");
    println!("{:>6} {:>12} {:>12}", "step", "time", "dt");
    println!("{}", "-".repeat(36));

    while time < T_FINAL {
        // apply boundary conditions
        apply_conserved_boundaries_2d(&mut den, &mut mom, &mut nrg, &boundary_spec, &device)?;

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

        if step > 10000 {
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
    let mut max_vx: f64 = 0.0;
    let mut max_vy: f64 = 0.0;
    let mut max_pre: f64 = 0.0;
    let mut min_pre: f64 = f64::MAX;

    for i in 0..ncells {
        max_rho = max_rho.max(rho_final[i]);
        min_rho = min_rho.min(rho_final[i]);
        max_vx = max_vx.max(vx_final[i].abs());
        max_vy = max_vy.max(vy_final[i].abs());
        max_pre = max_pre.max(pre_final[i]);
        min_pre = min_pre.min(pre_final[i]);
    }

    println!("final solution diagnostics:");
    println!("  density:  min = {:.6}, max = {:.6}", min_rho, max_rho);
    println!("  velocity: max_vx = {:.6}, max_vy = {:.6}", max_vx, max_vy);
    println!("  pressure: min = {:.6}, max = {:.6}", min_pre, max_pre);
    println!();

    // sample along centerline (y = ny/2)
    println!("solution samples (centerline y ~ 0.05):");
    println!(
        "{:>6} {:>12} {:>12} {:>12} {:>12} {:>12}",
        "i", "x", "rho", "vx", "vy", "pre"
    );
    println!("{}", "-".repeat(72));

    let j_mid = ny_total / 2;
    for &i_rel in &[10, 25, 50, 75, 90] {
        let i = i_rel + nghosts;
        if i < nx_total {
            let idx = j_mid * nx_total + i;
            let x = x_min + ((i - nghosts) as f64 + 0.5) * dx;
            println!(
                "{:6} {:12.4} {:12.6} {:12.6} {:12.6} {:12.6}",
                i - nghosts,
                x,
                rho_final[idx],
                vx_final[idx],
                vy_final[idx],
                pre_final[idx]
            );
        }
    }

    println!();
    println!("expected features:");
    println!("  - shock at x ~ 0.85");
    println!("  - contact at x ~ 0.7");
    println!("  - expansion fan at x ~ 0.3-0.5");
    println!("  - vy should be near zero everywhere");
    println!();

    // detect shock
    let mut shock_idx: usize = 0;
    let mut max_jump: f64 = 0.0;
    for j in 0..ny_total {
        for i in nghosts..(nx_total - nghosts - 1) {
            let idx = j * nx_total + i;
            let idx_next = j * nx_total + (i + 1);
            let jump: f64 = (rho_final[idx_next] - rho_final[idx]).abs();
            if jump > max_jump {
                max_jump = jump;
                shock_idx = i;
            }
        }
    }

    let shock_x = x_min + ((shock_idx - nghosts) as f64 + 0.5) * dx;
    println!("detected shock:");
    println!(
        "  location: x ~ {:.4} (cell {})",
        shock_x,
        shock_idx - nghosts
    );
    println!("  density jump: {:.6}", max_jump);
    println!();

    println!("=== 2d solver validation complete ===");

    Ok(())
}
