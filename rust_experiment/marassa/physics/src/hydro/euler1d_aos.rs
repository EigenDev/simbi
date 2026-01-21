// =============================================================================
// euler1d_aos.rs
//
// aos (array-of-structs) layout matching typical c++ implementation.
// stores state as Vec<State> where State = {rho, mom, energy}.
// this matches typical c++ patterns and may have better cache locality.
//
// design:
//   - aos layout: [State, State, State, ...]
//   - contiguous memory for each cell's data
//   - zero allocations in time step
//   - pcm + euler + hlle
//
// usage:
//   let mut solver = AoSEuler1D::new(ncells, xmin, xmax, gamma, cfl);
//   solver.set_ic(|x| (rho, vx, p));
//   while solver.time < t_final { solver.step(); }
// =============================================================================

use std::time::Instant;

/// conserved state for a single cell
#[derive(Debug, Copy, Clone)]
#[repr(C)]
struct State {
    rho: f64,
    mom: f64,
    energy: f64,
}

impl State {
    #[inline]
    fn new(rho: f64, mom: f64, energy: f64) -> Self {
        Self { rho, mom, energy }
    }

    #[inline]
    fn zero() -> Self {
        Self {
            rho: 0.0,
            mom: 0.0,
            energy: 0.0,
        }
    }
}

/// flux vector
#[derive(Debug, Copy, Clone)]
#[repr(C)]
struct Flux {
    rho: f64,
    mom: f64,
    energy: f64,
}

impl Flux {
    #[inline]
    fn zero() -> Self {
        Self {
            rho: 0.0,
            mom: 0.0,
            energy: 0.0,
        }
    }
}

/// aos layout euler solver
pub struct AoSEuler1D {
    // grid
    ncells: usize,
    nghost: usize,
    dx: f64,
    xmin: f64,
    xmax: f64,

    // physics
    gamma: f64,
    cfl: f64,

    // state (aos layout)
    state: Vec<State>,

    // scratch buffers
    flux: Vec<Flux>,

    // time
    pub time: f64,
    pub step_count: usize,

    // profiling
    start_time: Option<Instant>,
}

impl AoSEuler1D {
    /// creates a new aos solver
    pub fn new(ncells: usize, xmin: f64, xmax: f64, gamma: f64, cfl: f64) -> Self {
        let nghost = 1;
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
            state: vec![State::zero(); ntotal],
            flux: vec![Flux::zero(); ntotal + 1],
            time: 0.0,
            step_count: 0,
            start_time: None,
        }
    }

    /// sets initial conditions
    pub fn set_ic<F>(&mut self, ic: F)
    where
        F: Fn(f64) -> (f64, f64, f64),
    {
        let ng = self.nghost;
        let gamma = self.gamma;

        for i in 0..self.ncells {
            let x = self.xmin + (i as f64 + 0.5) * self.dx;
            let (rho, vx, p) = ic(x);

            let mom = rho * vx;
            let ke = 0.5 * rho * vx * vx;
            let ie = p / (gamma - 1.0);
            let energy = ke + ie;

            self.state[ng + i] = State::new(rho, mom, energy);
        }

        self.apply_bc();
    }

    /// outflow boundary conditions
    #[inline]
    fn apply_bc(&mut self) {
        let ng = self.nghost;
        let nc = self.ncells;

        for i in 0..ng {
            self.state[i] = self.state[ng];
            self.state[ng + nc + i] = self.state[ng + nc - 1];
        }
    }

    /// computes dt from cfl
    #[inline]
    fn compute_dt(&self) -> f64 {
        let ng = self.nghost;
        let nc = self.ncells;
        let gamma = self.gamma;

        let mut max_speed = 0.0;

        for i in ng..ng + nc {
            let s = &self.state[i];
            let vx = s.mom / s.rho;
            let ke = 0.5 * s.mom * s.mom / s.rho;
            let ie = s.energy - ke;
            let p = (gamma - 1.0) * ie;
            let cs = (gamma * p / s.rho).sqrt();
            let speed = vx.abs() + cs;

            if speed > max_speed {
                max_speed = speed;
            }
        }

        self.cfl * self.dx / max_speed
    }

    /// computes fluxes using pcm + hlle
    #[inline]
    fn compute_fluxes(&mut self) {
        let gamma = self.gamma;
        let ntotal = self.state.len();

        // raw pointers for unchecked access
        let state_ptr = self.state.as_ptr();
        let flux_ptr = self.flux.as_mut_ptr();

        for i in 1..ntotal {
            unsafe {
                let left = &*state_ptr.add(i - 1);
                let right = &*state_ptr.add(i);

                // left primitives
                let vx_l = left.mom / left.rho;
                let ke_l = 0.5 * left.mom * left.mom / left.rho;
                let ie_l = left.energy - ke_l;
                let p_l = (gamma - 1.0) * ie_l;

                // right primitives
                let vx_r = right.mom / right.rho;
                let ke_r = 0.5 * right.mom * right.mom / right.rho;
                let ie_r = right.energy - ke_r;
                let p_r = (gamma - 1.0) * ie_r;

                // left flux
                let f_rho_l = left.rho * vx_l;
                let f_mom_l = left.rho * vx_l * vx_l + p_l;
                let f_energy_l = (left.energy + p_l) * vx_l;

                // right flux
                let f_rho_r = right.rho * vx_r;
                let f_mom_r = right.rho * vx_r * vx_r + p_r;
                let f_energy_r = (right.energy + p_r) * vx_r;

                // wave speeds
                let cs_l = (gamma * p_l / left.rho).sqrt();
                let cs_r = (gamma * p_r / right.rho).sqrt();
                let s_l = (vx_l - cs_l).min(vx_r - cs_r);
                let s_r = (vx_l + cs_l).max(vx_r + cs_r);

                // hlle
                let flux = &mut *flux_ptr.add(i);

                if s_l >= 0.0 {
                    flux.rho = f_rho_l;
                    flux.mom = f_mom_l;
                    flux.energy = f_energy_l;
                } else if s_r <= 0.0 {
                    flux.rho = f_rho_r;
                    flux.mom = f_mom_r;
                    flux.energy = f_energy_r;
                } else {
                    let denom = 1.0 / (s_r - s_l);
                    flux.rho = (s_r * f_rho_l - s_l * f_rho_r + s_l * s_r * (right.rho - left.rho))
                        * denom;
                    flux.mom = (s_r * f_mom_l - s_l * f_mom_r + s_l * s_r * (right.mom - left.mom))
                        * denom;
                    flux.energy = (s_r * f_energy_l - s_l * f_energy_r
                        + s_l * s_r * (right.energy - left.energy))
                        * denom;
                }
            }
        }
    }

    /// conservative update
    #[inline]
    fn update(&mut self, dt: f64) {
        let ng = self.nghost;
        let nc = self.ncells;
        let dtdx = dt / self.dx;

        let state_ptr = self.state.as_mut_ptr();
        let flux_ptr = self.flux.as_ptr();

        unsafe {
            for i in ng..ng + nc {
                let s = &mut *state_ptr.add(i);
                let f_left = &*flux_ptr.add(i);
                let f_right = &*flux_ptr.add(i + 1);

                s.rho -= dtdx * (f_right.rho - f_left.rho);
                s.mom -= dtdx * (f_right.mom - f_left.mom);
                s.energy -= dtdx * (f_right.energy - f_left.energy);
            }
        }
    }

    /// single euler step
    #[inline]
    pub fn step(&mut self) {
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

    /// returns stats
    pub fn stats(&self) -> Option<(f64, f64, usize)> {
        self.start_time.map(|start| {
            let elapsed = start.elapsed().as_secs_f64();
            let zone_cycles = (self.ncells * self.step_count) as f64;
            let zone_cycles_per_sec = zone_cycles / elapsed;
            (zone_cycles_per_sec, elapsed, self.step_count)
        })
    }

    /// returns solution
    pub fn solution(&self) -> Vec<(f64, f64, f64)> {
        let ng = self.nghost;
        let nc = self.ncells;
        let gamma = self.gamma;

        (ng..ng + nc)
            .map(|i| {
                let s = &self.state[i];
                let vx = s.mom / s.rho;
                let ke = 0.5 * s.mom * s.mom / s.rho;
                let ie = s.energy - ke;
                let p = (gamma - 1.0) * ie;
                (s.rho, vx, p)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aos_solver() {
        let mut solver = AoSEuler1D::new(100, 0.0, 1.0, 1.4, 0.5);
        solver.set_ic(|x| {
            if x < 0.5 {
                (1.0, 0.0, 1.0)
            } else {
                (0.125, 0.0, 0.1)
            }
        });

        for _ in 0..100 {
            solver.step();
        }

        assert!(solver.time > 0.0);
        assert_eq!(solver.step_count, 100);
    }
}
