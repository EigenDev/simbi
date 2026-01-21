// =============================================================================
// godunov_1d_complete.rs
//
// complete 1d godunov solver demonstration using the new kernel architecture.
// demonstrates full pipeline: cons2prim, flux computation, time integration.
//
// problem: 1d sod shock tube
//   - left state: rho=1, v=0, p=1
//   - right state: rho=0.125, v=0, p=0.1
//   - discontinuity at x=0.5
//
// algorithm:
//   1. initialize conserved fields
//   2. loop over timesteps:
//      a. cons2prim conversion
//      b. compute cfl timestep
//      c. compute fluxes at interfaces
//      d. forward euler update
//      e. repeat
//   3. output final state
//
// usage:
//   cargo run --package physics --example godunov_1d_complete
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
const T_FINAL: f64 = 0.2;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== 1d godunov solver with plm reconstruction ===\n");

    // initialize device
    let device = CpuDevice::new(0)?;
    println!("device: cpu");

    // domain setup
    let nx: usize = 100;
    let x_min = 0.0;
    let x_max = 1.0;
    let dx = (x_max - x_min) / nx as f64;

    let cell_domain = Domain::from_shape([nx as i64]);
    let face_domain = Domain::from_shape([nx as i64 + 1]);

    println!("grid: nx = {}, dx = {:.4}", nx, dx);
    println!("time: t_final = {}", T_FINAL);
    println!("reconstruction: plm with minmod limiter");
    println!();

    // allocate fields
    let mut den = Field::<f64, _, 1>::zeros(&device, cell_domain)?;
    let mut mom = [Field::<f64, _, 1>::zeros(&device, cell_domain)?];
    let mut nrg = Field::<f64, _, 1>::zeros(&device, cell_domain)?;

    let mut rho = Field::<f64, _, 1>::zeros(&device, cell_domain)?;
    let mut vel = [Field::<f64, _, 1>::zeros(&device, cell_domain)?];
    let mut pre = Field::<f64, _, 1>::zeros(&device, cell_domain)?;

    let mut flux_den = [Field::<f64, _, 1>::zeros(&device, face_domain)?];
    let mut flux_mom = [[Field::<f64, _, 1>::zeros(&device, face_domain)?]];
    let mut flux_nrg = [Field::<f64, _, 1>::zeros(&device, face_domain)?];

    println!("allocated fields:");
    println!("  conserved: den, mom, nrg");
    println!("  primitive: rho, vel, pre");
    println!("  fluxes: flux_den, flux_mom, flux_nrg");
    println!();

    // initial conditions: sod shock tube
    println!("initializing: sod shock tube");
    let mut den_data = vec![0.0; nx];
    let mut mom_data = vec![0.0; nx];
    let mut nrg_data = vec![0.0; nx];

    for i in 0..nx {
        let x = x_min + (i as f64 + 0.5) * dx;

        let (rho_ic, v_ic, p_ic) = if x < 0.5 {
            (1.0, 0.0, 1.0) // left state
        } else {
            (0.125, 0.0, 0.1) // right state
        };

        // convert to conserved
        let prim = Primitive::<physics::hydro::Newtonian, 1>::new(rho_ic, [v_ic], p_ic);
        let cons = prim.to_conserved(GAMMA);

        den_data[i] = cons.den;
        mom_data[i] = cons.mom[0];
        nrg_data[i] = cons.nrg;
    }

    den.from_host(&den_data)?;
    mom[0].from_host(&mom_data)?;
    nrg.from_host(&nrg_data)?;

    println!("  left:  rho=1.0, v=0.0, p=1.0");
    println!("  right: rho=0.125, v=0.0, p=0.1");
    println!("  interface: x=0.5");
    println!();

    // time integration loop
    let mut time = 0.0;
    let mut step = 0;

    println!("starting time integration...");
    println!("{:>6} {:>12} {:>12}", "step", "time", "dt");
    println!("{}", "-".repeat(36));

    while time < T_FINAL {
        // step 1: cons2prim conversion
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

        // step 2: compute timestep from cfl condition
        let dt = compute_dt(&vel, &pre, &rho, GAMMA, [dx], CFL, cell_domain)?;

        // clamp to final time
        let dt = if time + dt > T_FINAL {
            T_FINAL - time
        } else {
            dt
        };

        if step % 10 == 0 {
            println!("{:6} {:12.6e} {:12.6e}", step, time, dt);
        }

        // step 3: compute fluxes at all interfaces with plm reconstruction
        compute_fluxes_newtonian(
            &rho,
            &vel,
            &pre,
            &mut flux_den,
            &mut flux_mom,
            &mut flux_nrg,
            GAMMA,
            [dx],
            Limiter::MinMod,
            &device,
            cell_domain,
            &[face_domain],
        )?;

        // step 4: forward euler time integration
        euler_step(
            &mut den,
            &mut mom,
            &mut nrg,
            &flux_den,
            &flux_mom,
            &flux_nrg,
            dt,
            [dx],
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

    // final cons2prim for output
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

    // extract solution
    let rho_final = rho.to_host()?;
    let vel_final = vel[0].to_host()?;
    let pre_final = pre.to_host()?;

    // compute some diagnostics
    let mut max_rho: f64 = 0.0;
    let mut min_rho: f64 = f64::MAX;
    let mut max_vel: f64 = 0.0;
    let mut max_pre: f64 = 0.0;
    let mut min_pre: f64 = f64::MAX;

    for i in 0..nx {
        max_rho = max_rho.max(rho_final[i]);
        min_rho = min_rho.min(rho_final[i]);
        max_vel = max_vel.max(vel_final[i].abs());
        max_pre = max_pre.max(pre_final[i]);
        min_pre = min_pre.min(pre_final[i]);
    }

    println!("final solution diagnostics:");
    println!("  density:  min = {:.6}, max = {:.6}", min_rho, max_rho);
    println!("  velocity: max = {:.6}", max_vel);
    println!("  pressure: min = {:.6}, max = {:.6}", min_pre, max_pre);
    println!();

    // sample solution at specific points
    println!("solution samples:");
    println!(
        "{:>6} {:>12} {:>12} {:>12} {:>12}",
        "i", "x", "rho", "vel", "pre"
    );
    println!("{}", "-".repeat(60));

    for &i in &[10, 25, 50, 75, 90] {
        let x = x_min + (i as f64 + 0.5) * dx;
        println!(
            "{:6} {:12.4} {:12.6} {:12.6} {:12.6}",
            i, x, rho_final[i], vel_final[i], pre_final[i]
        );
    }

    println!();
    println!("expected features (plm + minmod):");
    println!("  - left state preserved at x ~ 0.1-0.3");
    println!("  - expansion fan at x ~ 0.3-0.5");
    println!("  - contact discontinuity at x ~ 0.6-0.7");
    println!("  - shock at x ~ 0.8-0.9");
    println!("  - sharper resolution than first-order");
    println!();

    // check if shock is present (density jump)
    let mut shock_idx: usize = 0;
    let mut max_jump: f64 = 0.0;
    for i in 1..nx - 1 {
        let jump: f64 = (rho_final[i + 1] - rho_final[i]).abs();
        if jump > max_jump {
            max_jump = jump;
            shock_idx = i;
        }
    }

    let shock_x = x_min + (shock_idx as f64 + 0.5) * dx;
    println!("detected shock:");
    println!("  location: x ~ {:.4} (cell {})", shock_x, shock_idx);
    println!("  density jump: {:.6}", max_jump);
    println!();

    println!("=== solver demonstration complete ===");

    Ok(())
}
