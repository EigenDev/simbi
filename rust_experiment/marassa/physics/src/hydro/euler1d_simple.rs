// =============================================================================
// euler1d_simple.rs
//
// minimal first-order 1d euler solver for performance benchmarking.
// matches c++ baseline: euler timestepping + pcm reconstruction.
//
// design:
//   - zero allocations in time step loop
//   - pre-allocated scratch buffers
//   - direct array operations (no conversions)
//   - pcm reconstruction (no stencils)
//   - euler time stepping (first-order)
//
// usage:
//   let mut solver = SimpleEuler1D::new(ncells, xmin, xmax, gamma, cfl);
//   solver.set_ic(|x| if x < 0.5 { (1.0, 0.0, 1.0) } else { (0.125, 0.0, 0.1) });
//   while solver.time < t_final { solver.step(); }
// =============================================================================

use std::time::Instant;

/// minimal 1d euler solver with zero allocations in hot path
pub struct SimpleEuler1D {
    // grid
    ncells: usize,
    nghost: usize,
    dx: f64,
    xmin: f64,
    xmax: f64,

    // physics
    gamma: f64,
    cfl: f64,

    // state (including ghost zones)
    // soa layout: separate arrays for each variable
    rho: Vec<f64>,
    mom: Vec<f64>,
    energy: Vec<f64>,

    // scratch buffers (pre-allocated, reused every step)
    flux_rho: Vec<f64>,
    flux_mom: Vec<f64>,
    flux_energy: Vec<f64>,

    // time
    pub time: f64,
    pub step_count: usize,

    // profiling
    start_time: Option<Instant>,
}

impl SimpleEuler1D {
    /// creates a new solver
    pub fn new(ncells: usize, xmin: f64, xmax: f64, gamma: f64, cfl: f64) -> Self {
        let nghost = 1; // pcm only needs 1 ghost zone per side
        let ntotal = ncells + 2 * nghost;
        let dx = (xmax - xmin) / (ncells as f64);

        Self {
            ncells,
            nghost,
            dx,
            xmin,
            xmax,
            gamma,
            cfl,
            rho: vec![0.0; ntotal],
            mom: vec![0.0; ntotal],
            energy: vec![0.0; ntotal],
            flux_rho: vec![0.0; ntotal + 1],
            flux_mom: vec![0.0; ntotal + 1],
            flux_energy: vec![0.0; ntotal + 1],
            time: 0.0,
            step_count: 0,
            start_time: None,
        }
    }

    /// sets initial conditions: (rho, vx, p)
    pub fn set_ic<F>(&mut self, ic: F)
    where
        F: Fn(f64) -> (f64, f64, f64),
    {
        for i in 0..self.ncells {
            let x = self.xmin + (i as f64 + 0.5) * self.dx;
            let (rho, vx, p) = ic(x);

            let idx = i + self.nghost;
            self.rho[idx] = rho;
            self.mom[idx] = rho * vx;
            let ke = 0.5 * rho * vx * vx;
            let ie = p / (self.gamma - 1.0);
            self.energy[idx] = ke + ie;
        }

        self.apply_bc();
    }

    /// applies outflow boundary conditions
    #[inline]
    fn apply_bc(&mut self) {
        let ng = self.nghost;
        let nc = self.ncells;

        // left boundary (outflow)
        for i in 0..ng {
            self.rho[i] = self.rho[ng];
            self.mom[i] = self.mom[ng];
            self.energy[i] = self.energy[ng];
        }

        // right boundary (outflow)
        for i in 0..ng {
            self.rho[ng + nc + i] = self.rho[ng + nc - 1];
            self.mom[ng + nc + i] = self.mom[ng + nc - 1];
            self.energy[ng + nc + i] = self.energy[ng + nc - 1];
        }
    }

    /// computes time step from cfl condition
    #[inline]
    fn compute_dt(&self) -> f64 {
        let ng = self.nghost;
        let nc = self.ncells;
        let gamma = self.gamma;

        let mut max_speed = 0.0;

        for i in ng..ng + nc {
            let rho = self.rho[i];
            let vx = self.mom[i] / rho;
            let ke = 0.5 * self.mom[i] * self.mom[i] / rho;
            let ie = self.energy[i] - ke;
            let p = (gamma - 1.0) * ie;
            let cs = (gamma * p / rho).sqrt();
            let speed = vx.abs() + cs;

            if speed > max_speed {
                max_speed = speed;
            }
        }

        self.cfl * self.dx / max_speed
    }

    /// computes fluxes at all interfaces (pcm reconstruction + hlle solver)
    #[inline]
    fn compute_fluxes(&mut self) {
        let gamma = self.gamma;
        let ntotal = self.rho.len();

        // get raw pointers for unchecked access
        let rho_ptr = self.rho.as_ptr();
        let mom_ptr = self.mom.as_ptr();
        let energy_ptr = self.energy.as_ptr();
        let flux_rho_ptr = self.flux_rho.as_mut_ptr();
        let flux_mom_ptr = self.flux_mom.as_mut_ptr();
        let flux_energy_ptr = self.flux_energy.as_mut_ptr();

        // loop over interfaces
        for i in 1..ntotal {
            unsafe {
                // pcm reconstruction: left state = cell i-1, right state = cell i
                let rho_l = *rho_ptr.add(i - 1);
                let mom_l = *mom_ptr.add(i - 1);
                let energy_l = *energy_ptr.add(i - 1);

                let rho_r = *rho_ptr.add(i);
                let mom_r = *mom_ptr.add(i);
                let energy_r = *energy_ptr.add(i);

                // convert to primitives
                let vx_l = mom_l / rho_l;
                let ke_l = 0.5 * mom_l * mom_l / rho_l;
                let ie_l = energy_l - ke_l;
                let p_l = (gamma - 1.0) * ie_l;

                let vx_r = mom_r / rho_r;
                let ke_r = 0.5 * mom_r * mom_r / rho_r;
                let ie_r = energy_r - ke_r;
                let p_r = (gamma - 1.0) * ie_r;

                // compute fluxes for left and right states
                let f_rho_l = rho_l * vx_l;
                let f_mom_l = rho_l * vx_l * vx_l + p_l;
                let f_energy_l = (energy_l + p_l) * vx_l;

                let f_rho_r = rho_r * vx_r;
                let f_mom_r = rho_r * vx_r * vx_r + p_r;
                let f_energy_r = (energy_r + p_r) * vx_r;

                // wave speeds for hlle
                let cs_l = (gamma * p_l / rho_l).sqrt();
                let cs_r = (gamma * p_r / rho_r).sqrt();

                let s_l = (vx_l - cs_l).min(vx_r - cs_r);
                let s_r = (vx_l + cs_l).max(vx_r + cs_r);

                // hlle flux
                if s_l >= 0.0 {
                    // supersonic left
                    *flux_rho_ptr.add(i) = f_rho_l;
                    *flux_mom_ptr.add(i) = f_mom_l;
                    *flux_energy_ptr.add(i) = f_energy_l;
                } else if s_r <= 0.0 {
                    // supersonic right
                    *flux_rho_ptr.add(i) = f_rho_r;
                    *flux_mom_ptr.add(i) = f_mom_r;
                    *flux_energy_ptr.add(i) = f_energy_r;
                } else {
                    // subsonic: hll intermediate state
                    let denom = 1.0 / (s_r - s_l);
                    *flux_rho_ptr.add(i) =
                        (s_r * f_rho_l - s_l * f_rho_r + s_l * s_r * (rho_r - rho_l)) * denom;
                    *flux_mom_ptr.add(i) =
                        (s_r * f_mom_l - s_l * f_mom_r + s_l * s_r * (mom_r - mom_l)) * denom;
                    *flux_energy_ptr.add(i) = (s_r * f_energy_l - s_l * f_energy_r
                        + s_l * s_r * (energy_r - energy_l))
                        * denom;
                }
            }
        }
    }

    /// updates conserved variables: u^{n+1} = u^n - dt/dx * (f[i+1] - f[i])
    #[inline]
    fn update(&mut self, dt: f64) {
        let ng = self.nghost;
        let nc = self.ncells;
        let dtdx = dt / self.dx;

        // unchecked access for performance
        unsafe {
            let rho_ptr = self.rho.as_mut_ptr();
            let mom_ptr = self.mom.as_mut_ptr();
            let energy_ptr = self.energy.as_mut_ptr();
            let flux_rho_ptr = self.flux_rho.as_ptr();
            let flux_mom_ptr = self.flux_mom.as_ptr();
            let flux_energy_ptr = self.flux_energy.as_ptr();

            for i in ng..ng + nc {
                *rho_ptr.add(i) -= dtdx * (*flux_rho_ptr.add(i + 1) - *flux_rho_ptr.add(i));
                *mom_ptr.add(i) -= dtdx * (*flux_mom_ptr.add(i + 1) - *flux_mom_ptr.add(i));
                *energy_ptr.add(i) -=
                    dtdx * (*flux_energy_ptr.add(i + 1) - *flux_energy_ptr.add(i));
            }
        }
    }

    /// advances solution by one time step (euler forward)
    #[inline]
    pub fn step(&mut self) {
        // start profiling on first step
        if self.start_time.is_none() {
            self.start_time = Some(Instant::now());
        }

        let dt = self.compute_dt();
        self.compute_fluxes();
        self.update(dt);
        self.apply_bc();

        self.time += dt;
        self.step_count += 1;
    }

    /// returns performance stats
    pub fn stats(&self) -> Option<(f64, f64, usize)> {
        self.start_time.map(|start| {
            let elapsed = start.elapsed().as_secs_f64();
            let zone_cycles = (self.ncells * self.step_count) as f64;
            let zone_cycles_per_sec = zone_cycles / elapsed;
            (zone_cycles_per_sec, elapsed, self.step_count)
        })
    }

    /// returns solution (interior cells only)
    pub fn solution(&self) -> Vec<(f64, f64, f64)> {
        let ng = self.nghost;
        let nc = self.ncells;
        let gamma = self.gamma;

        (ng..ng + nc)
            .map(|i| {
                let rho = self.rho[i];
                let vx = self.mom[i] / rho;
                let ke = 0.5 * self.mom[i] * self.mom[i] / rho;
                let ie = self.energy[i] - ke;
                let p = (gamma - 1.0) * ie;
                (rho, vx, p)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_solver_creation() {
        let solver = SimpleEuler1D::new(100, 0.0, 1.0, 1.4, 0.5);
        assert_eq!(solver.ncells, 100);
        assert_eq!(solver.time, 0.0);
    }

    #[test]
    fn test_constant_state() {
        let mut solver = SimpleEuler1D::new(10, 0.0, 1.0, 1.4, 0.5);
        solver.set_ic(|_| (1.0, 0.0, 1.0));

        // constant state should remain constant
        for _ in 0..10 {
            solver.step();
        }

        let sol = solver.solution();
        for (rho, vx, p) in sol {
            assert!((rho - 1.0).abs() < 1e-10);
            assert!(vx.abs() < 1e-10);
            assert!((p - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_sod_shock_tube() {
        let mut solver = SimpleEuler1D::new(100, 0.0, 1.0, 1.4, 0.5);
        solver.set_ic(|x| {
            if x < 0.5 {
                (1.0, 0.0, 1.0)
            } else {
                (0.125, 0.0, 0.1)
            }
        });

        // run to t=0.2
        while solver.time < 0.2 {
            solver.step();
        }

        // check that shock propagated
        let sol = solver.solution();
        assert!(sol[0].0 > 0.9); // left state preserved
        assert!(sol[99].0 < 0.2); // right state expanded
    }
}
