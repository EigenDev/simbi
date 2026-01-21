// =============================================================================
// euler_1d_hlle.rs
//
// 1d euler equations solver using hlle riemann solver.
// first-order godunov method for sod shock tube problem.
//
// benchmarks parallel cpu vs metal on 1 billion zones.
// reports performance in zone-cycles/second.
//
// physics:
//   - conservative variables: (ρ, ρu, E)
//   - primitive variables: (ρ, u, p)
//   - hlle riemann solver for interface fluxes
//   - ideal gas equation of state: p = (γ-1)(E - 0.5*ρ*u²)
//
// sod shock tube initial conditions:
//   - left state:  ρ=1.0, u=0.0, p=1.0
//   - right state: ρ=0.125, u=0.0, p=0.1
//   - discontinuity at x=0.5
// =============================================================================

use std::env;
use std::time::Instant;

// physics constants
const GAMMA: f64 = 1.4; // ratio of specific heats for ideal gas

// 1d euler state (conservative variables)
#[derive(Clone, Copy, Debug)]
struct State {
    rho: f64,   // density
    rho_u: f64, // momentum
    e: f64,     // total energy
}

impl State {
    fn new(rho: f64, u: f64, p: f64) -> Self {
        let rho_u = rho * u;
        let e = p / (GAMMA - 1.0) + 0.5 * rho * u * u;
        State { rho, rho_u, e }
    }

    fn density(&self) -> f64 {
        self.rho
    }

    fn velocity(&self) -> f64 {
        self.rho_u / self.rho
    }

    fn pressure(&self) -> f64 {
        let u = self.velocity();
        (GAMMA - 1.0) * (self.e - 0.5 * self.rho * u * u)
    }

    fn sound_speed(&self) -> f64 {
        let p = self.pressure();
        (GAMMA * p / self.rho).sqrt()
    }

    // convert to flux f(u)
    fn flux(&self) -> State {
        let u = self.velocity();
        let p = self.pressure();
        State {
            rho: self.rho_u,
            rho_u: self.rho_u * u + p,
            e: (self.e + p) * u,
        }
    }
}

// hlle riemann solver
fn hlle_flux(left: State, right: State) -> State {
    let u_l = left.velocity();
    let u_r = right.velocity();
    let c_l = left.sound_speed();
    let c_r = right.sound_speed();

    // wave speed estimates
    let s_l = (u_l - c_l).min(u_r - c_r);
    let s_r = (u_l + c_l).max(u_r + c_r);

    // hlle flux
    if s_l >= 0.0 {
        left.flux()
    } else if s_r <= 0.0 {
        right.flux()
    } else {
        let f_l = left.flux();
        let f_r = right.flux();
        let denom = s_r - s_l;
        State {
            rho: (s_r * f_l.rho - s_l * f_r.rho + s_l * s_r * (right.rho - left.rho)) / denom,
            rho_u: (s_r * f_l.rho_u - s_l * f_r.rho_u + s_l * s_r * (right.rho_u - left.rho_u))
                / denom,
            e: (s_r * f_l.e - s_l * f_r.e + s_l * s_r * (right.e - left.e)) / denom,
        }
    }
}

// 1d euler solver
struct EulerSolver1D {
    n_zones: usize,
    x_min: f64,
    x_max: f64,
    dx: f64,
    cfl: f64,
    // conserved variables
    rho: Vec<f64>,
    rho_u: Vec<f64>,
    e: Vec<f64>,
}

impl EulerSolver1D {
    fn new(n_zones: usize, x_min: f64, x_max: f64, cfl: f64) -> Self {
        let dx = (x_max - x_min) / (n_zones as f64);
        EulerSolver1D {
            n_zones,
            x_min,
            x_max,
            dx,
            cfl,
            rho: vec![0.0; n_zones],
            rho_u: vec![0.0; n_zones],
            e: vec![0.0; n_zones],
        }
    }

    // initialize sod shock tube
    fn init_sod(&mut self) {
        let x_disc = 0.5;
        for i in 0..self.n_zones {
            let x = self.x_min + (i as f64 + 0.5) * self.dx;
            let state = if x < x_disc {
                State::new(1.0, 0.0, 1.0) // left state
            } else {
                State::new(0.125, 0.0, 0.1) // right state
            };
            self.rho[i] = state.rho;
            self.rho_u[i] = state.rho_u;
            self.e[i] = state.e;
        }
    }

    // compute timestep from cfl condition
    fn compute_dt(&self) -> f64 {
        let mut max_speed: f64 = 0.0;
        for i in 0..self.n_zones {
            let state = State {
                rho: self.rho[i],
                rho_u: self.rho_u[i],
                e: self.e[i],
            };
            let u = state.velocity();
            let c = state.sound_speed();
            let speed = (u.abs() + c).abs();
            max_speed = max_speed.max(speed);
        }
        self.cfl * self.dx / max_speed
    }

    // single timestep using first-order godunov method
    fn step(&mut self, dt: f64) -> f64 {
        let mut rho_new = vec![0.0; self.n_zones];
        let mut rho_u_new = vec![0.0; self.n_zones];
        let mut e_new = vec![0.0; self.n_zones];

        // compute fluxes and update
        for i in 0..self.n_zones {
            let i_l = if i == 0 { 0 } else { i - 1 };
            let i_r = if i == self.n_zones - 1 {
                self.n_zones - 1
            } else {
                i + 1
            };

            // states at interfaces
            let state_i = State {
                rho: self.rho[i],
                rho_u: self.rho_u[i],
                e: self.e[i],
            };
            let state_l = State {
                rho: self.rho[i_l],
                rho_u: self.rho_u[i_l],
                e: self.e[i_l],
            };
            let state_r = State {
                rho: self.rho[i_r],
                rho_u: self.rho_u[i_r],
                e: self.e[i_r],
            };

            // hlle fluxes at left and right interfaces
            let flux_l = hlle_flux(state_l, state_i);
            let flux_r = hlle_flux(state_i, state_r);

            // conservative update
            let dtdx = dt / self.dx;
            rho_new[i] = self.rho[i] - dtdx * (flux_r.rho - flux_l.rho);
            rho_u_new[i] = self.rho_u[i] - dtdx * (flux_r.rho_u - flux_l.rho_u);
            e_new[i] = self.e[i] - dtdx * (flux_r.e - flux_l.e);
        }

        self.rho = rho_new;
        self.rho_u = rho_u_new;
        self.e = e_new;

        dt
    }

    // evolve for fixed number of iterations (for benchmarking)
    fn evolve(&mut self, max_iterations: usize, report_interval: usize) -> (usize, f64) {
        let mut t = 0.0;
        let mut n_steps = 0;
        let start_time = Instant::now();

        while n_steps < max_iterations {
            let dt = self.compute_dt();

            let _ = self.step(dt);
            t += dt;
            n_steps += 1;

            if n_steps % report_interval == 0 {
                let elapsed = start_time.elapsed().as_secs_f64();
                let zone_cycles = (self.n_zones * n_steps) as f64;
                let speed = zone_cycles / elapsed;
                println!(
                    "  iteration {:6} | dt={:.3e} | time={:.4} | speed={:.3e} zone-cycles/s",
                    n_steps, dt, t, speed
                );
            }
        }

        (n_steps, t)
    }

    // get state at index
    fn get_state(&self, i: usize) -> State {
        State {
            rho: self.rho[i],
            rho_u: self.rho_u[i],
            e: self.e[i],
        }
    }
}

// parallel cpu version using rayon
mod parallel {
    use super::*;
    use rayon::prelude::*;

    pub struct ParallelEulerSolver1D {
        n_zones: usize,
        x_min: f64,
        x_max: f64,
        dx: f64,
        cfl: f64,
        rho: Vec<f64>,
        rho_u: Vec<f64>,
        e: Vec<f64>,
    }

    impl ParallelEulerSolver1D {
        pub fn new(n_zones: usize, x_min: f64, x_max: f64, cfl: f64) -> Self {
            ParallelEulerSolver1D {
                n_zones,
                x_min,
                x_max,
                dx: (x_max - x_min) / (n_zones as f64),
                cfl,
                rho: vec![0.0; n_zones],
                rho_u: vec![0.0; n_zones],
                e: vec![0.0; n_zones],
            }
        }

        pub fn init_sod(&mut self) {
            let x_disc = 0.5;
            let dx = self.dx;
            let x_min = self.x_min;

            self.rho
                .par_iter_mut()
                .zip(self.rho_u.par_iter_mut())
                .zip(self.e.par_iter_mut())
                .enumerate()
                .for_each(|(i, ((rho, rho_u), e))| {
                    let x = x_min + (i as f64 + 0.5) * dx;
                    let state = if x < x_disc {
                        State::new(1.0, 0.0, 1.0)
                    } else {
                        State::new(0.125, 0.0, 0.1)
                    };
                    *rho = state.rho;
                    *rho_u = state.rho_u;
                    *e = state.e;
                });
        }

        pub fn compute_dt(&self) -> f64 {
            let max_speed = self
                .rho
                .par_iter()
                .zip(self.rho_u.par_iter())
                .zip(self.e.par_iter())
                .map(|((&rho, &rho_u), &e)| {
                    let state = State { rho, rho_u, e };
                    let u = state.velocity();
                    let c = state.sound_speed();
                    (u.abs() + c).abs()
                })
                .reduce(|| 0.0, |a, b| a.max(b));

            self.cfl * self.dx / max_speed
        }

        pub fn step(&mut self, dt: f64) -> f64 {
            let n_zones = self.n_zones;
            let dx = self.dx;
            let dtdx = dt / dx;

            // parallel flux computation and update
            let updates: Vec<(f64, f64, f64)> = (0..n_zones)
                .into_par_iter()
                .map(|i| {
                    let i_l = if i == 0 { 0 } else { i - 1 };
                    let i_r = if i == n_zones - 1 { n_zones - 1 } else { i + 1 };

                    let state_i = State {
                        rho: self.rho[i],
                        rho_u: self.rho_u[i],
                        e: self.e[i],
                    };
                    let state_l = State {
                        rho: self.rho[i_l],
                        rho_u: self.rho_u[i_l],
                        e: self.e[i_l],
                    };
                    let state_r = State {
                        rho: self.rho[i_r],
                        rho_u: self.rho_u[i_r],
                        e: self.e[i_r],
                    };

                    let flux_l = hlle_flux(state_l, state_i);
                    let flux_r = hlle_flux(state_i, state_r);

                    let rho_new = self.rho[i] - dtdx * (flux_r.rho - flux_l.rho);
                    let rho_u_new = self.rho_u[i] - dtdx * (flux_r.rho_u - flux_l.rho_u);
                    let e_new = self.e[i] - dtdx * (flux_r.e - flux_l.e);

                    (rho_new, rho_u_new, e_new)
                })
                .collect();

            // apply updates
            for (i, (rho, rho_u, e)) in updates.into_iter().enumerate() {
                self.rho[i] = rho;
                self.rho_u[i] = rho_u;
                self.e[i] = e;
            }

            dt
        }

        pub fn evolve(&mut self, max_iterations: usize, report_interval: usize) -> (usize, f64) {
            let mut t = 0.0;
            let mut n_steps = 0;
            let start_time = Instant::now();

            while n_steps < max_iterations {
                let dt = self.compute_dt();

                let _ = self.step(dt);
                t += dt;
                n_steps += 1;

                if n_steps % report_interval == 0 {
                    let elapsed = start_time.elapsed().as_secs_f64();
                    let zone_cycles = (self.n_zones * n_steps) as f64;
                    let speed = zone_cycles / elapsed;
                    println!(
                        "  iteration {:6} | dt={:.3e} | time={:.4} | speed={:.3e} zone-cycles/s",
                        n_steps, dt, t, speed
                    );
                }
            }

            (n_steps, t)
        }

        pub fn get_state(&self, i: usize) -> State {
            State {
                rho: self.rho[i],
                rho_u: self.rho_u[i],
                e: self.e[i],
            }
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== 1d euler equations with hlle riemann solver ===\n");
    println!("sod shock tube problem");
    println!("first-order godunov method\n");

    // problem setup - read from command line or default to 10M
    let args: Vec<String> = env::args().collect();
    let n_zones_high = if args.len() > 1 {
        args[1].parse::<usize>().unwrap_or_else(|_| {
            eprintln!("error: invalid number of zones, using default 10000000");
            10_000_000
        })
    } else {
        10_000_000
    };

    let n_zones_ref = 1000; // low-resolution reference
    let x_min = 0.0;
    let x_max = 1.0;
    let cfl = 0.4;
    let max_iterations = 1000; // cap at 1000 iterations for fast benchmarking
    let report_interval = 100;

    println!("benchmark configuration:");
    println!("  reference resolution: {} zones (serial cpu)", n_zones_ref);
    println!(
        "  high resolution: {} zones (parallel cpu + metal)",
        n_zones_high
    );
    println!("  domain: [{}, {}]", x_min, x_max);
    println!("  cfl number: {}", cfl);
    println!("  max iterations: {}", max_iterations);
    println!(
        "  high-res dx: {:.3e}",
        (x_max - x_min) / (n_zones_high as f64)
    );
    println!();

    // step 1: low-resolution reference (serial cpu)
    println!(
        "=== step 1: reference solution (serial cpu, {} zones) ===",
        n_zones_ref
    );
    let mut solver_ref = EulerSolver1D::new(n_zones_ref, x_min, x_max, cfl);
    solver_ref.init_sod();
    println!("running {} iterations...", max_iterations);
    let start = Instant::now();
    let (n_steps_ref, t_final_ref) = solver_ref.evolve(max_iterations, report_interval);
    let elapsed_ref = start.elapsed();

    let i_mid_ref = n_zones_ref / 2;
    let state_ref = solver_ref.get_state(i_mid_ref);

    println!("\nreference solution:");
    println!("  iterations: {}", n_steps_ref);
    println!("  final time: {:.4}", t_final_ref);
    println!("  wall time: {:.3} s", elapsed_ref.as_secs_f64());
    println!(
        "  state at x=0.5: ρ={:.6}, u={:.6}, p={:.6}",
        state_ref.density(),
        state_ref.velocity(),
        state_ref.pressure()
    );
    println!("  (this is the convergence target for high-res runs)");
    println!();

    // step 2: parallel cpu benchmark at high resolution
    println!(
        "=== step 2: parallel cpu benchmark ({} zones) ===",
        n_zones_high
    );
    let mut solver_par = parallel::ParallelEulerSolver1D::new(n_zones_high, x_min, x_max, cfl);

    println!("initializing...");
    let start_init = Instant::now();
    solver_par.init_sod();
    let init_time = start_init.elapsed();
    println!("  initialization: {:.3} s", init_time.as_secs_f64());

    println!("running {} iterations...", max_iterations);
    let start = Instant::now();
    let (n_steps_par, t_final_par) = solver_par.evolve(max_iterations, report_interval);
    let elapsed_par = start.elapsed();

    let total_zone_cycles_par = (n_zones_high as f64) * (n_steps_par as f64);
    let speed_par = total_zone_cycles_par / elapsed_par.as_secs_f64();

    let state_par = solver_par.get_state(n_zones_high / 2);

    println!("\nparallel cpu results:");
    println!("  iterations: {}", n_steps_par);
    println!("  final time: {:.4}", t_final_par);
    println!("  wall time: {:.2} s", elapsed_par.as_secs_f64());
    println!("  performance: {:.3e} zone-cycles/second", speed_par);
    println!("  performance: {:.2} Gzone-cycles/s", speed_par / 1e9);
    println!(
        "  state at x=0.5: ρ={:.6}, u={:.6}, p={:.6}",
        state_par.density(),
        state_par.velocity(),
        state_par.pressure()
    );

    // convergence check
    let rho_err = ((state_par.density() - state_ref.density()) / state_ref.density()).abs();
    let u_err = ((state_par.velocity() - state_ref.velocity()).abs()
        / (state_ref.velocity().abs() + 1e-10))
        .abs();
    let p_err = ((state_par.pressure() - state_ref.pressure()) / state_ref.pressure()).abs();

    println!("\nconvergence (vs reference):");
    println!("  ρ error: {:.2}%", rho_err * 100.0);
    println!("  u error: {:.2}%", u_err * 100.0);
    println!("  p error: {:.2}%", p_err * 100.0);
    println!();

    // step 3: metal gpu benchmark (placeholder)
    #[cfg(target_os = "macos")]
    {
        println!(
            "=== step 3: metal gpu benchmark ({} zones) ===",
            n_zones_high
        );
        println!("note: metal implementation requires custom gpu kernel");
        println!("      current metal device doesn't support riemann solver kernels");
        println!("      implementation would require:");
        println!("        1. metal compute shader for hlle flux calculation");
        println!("        2. metal kernel for conservative update");
        println!("        3. metal reduce for cfl timestep calculation");
        println!("\n      this is a future enhancement - parallel cpu shows the path forward");
        println!();
    }

    #[cfg(not(target_os = "macos"))]
    {
        println!("=== step 3: metal gpu benchmark ===");
        println!("skipped (not running on macos)");
        println!();
    }

    println!("=== benchmark summary ===");
    println!("\nreference solution (serial cpu, {} zones):", n_zones_ref);
    println!(
        "  state at x=0.5: ρ={:.6}, u={:.6}, p={:.6}",
        state_ref.density(),
        state_ref.velocity(),
        state_ref.pressure()
    );

    println!("\nparallel cpu ({} zones):", n_zones_high);
    println!("  performance: {:.3e} zone-cycles/second", speed_par);
    println!("  performance: {:.2} Gzone-cycles/s", speed_par / 1e9);
    println!(
        "  convergence: ρ={:.2}%, u={:.2}%, p={:.2}%",
        rho_err * 100.0,
        u_err * 100.0,
        p_err * 100.0
    );

    println!("\n=== benchmark complete ===");
    println!("\nusage: cargo run --release --example euler_1d_hlle [n_zones]");
    println!("  1000000 - 1M zones (quick test)");
    println!("  10000000 - 10M zones (default)");
    println!("  100000000 - 100M zones (requires ~2.4 GB RAM)");
    println!("  1000000000 - 1 billion zones (requires ~24 GB RAM)");
    println!(
        "\nperformance reported every {} iterations",
        report_interval
    );

    Ok(())
}
