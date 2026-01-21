// =============================================================================
// euler1d_parallel.rs
//
// parallel 1d euler solver using native rust threads.
// same algorithm as euler1d_simple.rs but with std::thread parallelism.
//
// design:
//   - static thread pool (spawned once, reused)
//   - chunk-based work distribution
//   - no external dependencies (no rayon)
//   - minimal synchronization overhead
//
// usage:
//   let mut solver = ParallelEuler1D::new(ncells, xmin, xmax, gamma, cfl, nthreads);
//   solver.set_ic(|x| if x < 0.5 { (1.0, 0.0, 1.0) } else { (0.125, 0.0, 0.1) });
//   while solver.time < t_final { solver.step(); }
// =============================================================================

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Barrier};
use std::thread::{self, JoinHandle};
use std::time::Instant;

// thread-safe raw pointer wrapper
struct SendPtr<T>(*mut T);
unsafe impl<T> Send for SendPtr<T> {}
unsafe impl<T> Sync for SendPtr<T> {}

impl<T> SendPtr<T> {
    fn new(ptr: *mut T) -> Self {
        Self(ptr)
    }
    fn as_ptr(&self) -> *mut T {
        self.0
    }
}

struct ConstPtr<T>(*const T);
unsafe impl<T> Send for ConstPtr<T> {}
unsafe impl<T> Sync for ConstPtr<T> {}

impl<T> ConstPtr<T> {
    fn new(ptr: *const T) -> Self {
        Self(ptr)
    }
    fn as_ptr(&self) -> *const T {
        self.0
    }
}

// shared state for worker threads
struct SharedState {
    // grid parameters
    ncells: usize,
    nghost: usize,
    ntotal: usize,
    gamma: f64,
    dx: f64,
    cfl: f64,

    // pointers to data arrays (owned by solver)
    rho: ConstPtr<f64>,
    mom: ConstPtr<f64>,
    energy: ConstPtr<f64>,
    flux_rho: SendPtr<f64>,
    flux_mom: SendPtr<f64>,
    flux_energy: SendPtr<f64>,

    // synchronization
    barrier: Barrier,
    max_speed: AtomicU64,
    dt: AtomicU64,

    // control
    running: std::sync::atomic::AtomicBool,
    phase: std::sync::atomic::AtomicU8,
}

// phases of computation
const PHASE_IDLE: u8 = 0;
const PHASE_COMPUTE_DT: u8 = 1;
const PHASE_COMPUTE_FLUX: u8 = 2;
const PHASE_UPDATE: u8 = 3;
const PHASE_SHUTDOWN: u8 = 255;

/// parallel 1d euler solver with native thread pool
pub struct ParallelEuler1D {
    // grid
    ncells: usize,
    nghost: usize,
    dx: f64,
    xmin: f64,

    // physics
    gamma: f64,
    cfl: f64,

    // state arrays
    rho: Vec<f64>,
    mom: Vec<f64>,
    energy: Vec<f64>,

    // flux buffers
    flux_rho: Vec<f64>,
    flux_mom: Vec<f64>,
    flux_energy: Vec<f64>,

    // time
    pub time: f64,
    pub step_count: usize,

    // parallelism
    nthreads: usize,
    shared: Arc<SharedState>,
    workers: Vec<JoinHandle<()>>,

    // profiling
    start_time: Option<Instant>,
}

impl ParallelEuler1D {
    /// creates a new parallel solver
    pub fn new(ncells: usize, xmin: f64, xmax: f64, gamma: f64, cfl: f64, nthreads: usize) -> Self {
        let nghost = 1;
        let ntotal = ncells + 2 * nghost;
        let dx = (xmax - xmin) / (ncells as f64);

        let rho = vec![0.0; ntotal];
        let mom = vec![0.0; ntotal];
        let energy = vec![0.0; ntotal];
        let mut flux_rho = vec![0.0; ntotal + 1];
        let mut flux_mom = vec![0.0; ntotal + 1];
        let mut flux_energy = vec![0.0; ntotal + 1];

        // create shared state with pointers
        let shared = Arc::new(SharedState {
            ncells,
            nghost,
            ntotal,
            gamma,
            dx,
            cfl,
            rho: ConstPtr::new(rho.as_ptr()),
            mom: ConstPtr::new(mom.as_ptr()),
            energy: ConstPtr::new(energy.as_ptr()),
            flux_rho: SendPtr::new(flux_rho.as_mut_ptr()),
            flux_mom: SendPtr::new(flux_mom.as_mut_ptr()),
            flux_energy: SendPtr::new(flux_energy.as_mut_ptr()),
            barrier: Barrier::new(nthreads + 1), // workers + main
            max_speed: AtomicU64::new(0),
            dt: AtomicU64::new(0),
            running: std::sync::atomic::AtomicBool::new(true),
            phase: std::sync::atomic::AtomicU8::new(PHASE_IDLE),
        });

        // spawn worker threads
        let workers: Vec<_> = (0..nthreads)
            .map(|tid| {
                let shared = Arc::clone(&shared);
                thread::spawn(move || worker_loop(tid, nthreads, shared))
            })
            .collect();

        Self {
            ncells,
            nghost,
            dx,
            xmin,
            gamma,
            cfl,
            rho,
            mom,
            energy,
            flux_rho,
            flux_mom,
            flux_energy,
            time: 0.0,
            step_count: 0,
            nthreads,
            shared,
            workers,
            start_time: None,
        }
    }

    /// sets initial conditions
    pub fn set_ic<F>(&mut self, ic: F)
    where
        F: Fn(f64) -> (f64, f64, f64),
    {
        for ii in 0..self.ncells {
            let x = self.xmin + (ii as f64 + 0.5) * self.dx;
            let (rho, vx, p) = ic(x);

            let idx = ii + self.nghost;
            self.rho[idx] = rho;
            self.mom[idx] = rho * vx;
            let ke = 0.5 * rho * vx * vx;
            let ie = p / (self.gamma - 1.0);
            self.energy[idx] = ke + ie;
        }

        self.apply_bc();
        self.update_pointers();
    }

    /// updates shared state pointers after reallocation
    fn update_pointers(&mut self) {
        // pointers are set at construction and arrays don't reallocate,
        // so this is a no-op. kept for safety if we ever resize.
    }

    /// applies outflow boundary conditions (serial, small work)
    #[inline]
    fn apply_bc(&mut self) {
        let ng = self.nghost;
        let nc = self.ncells;

        for ii in 0..ng {
            self.rho[ii] = self.rho[ng];
            self.mom[ii] = self.mom[ng];
            self.energy[ii] = self.energy[ng];
        }

        for ii in 0..ng {
            self.rho[ng + nc + ii] = self.rho[ng + nc - 1];
            self.mom[ng + nc + ii] = self.mom[ng + nc - 1];
            self.energy[ng + nc + ii] = self.energy[ng + nc - 1];
        }
    }

    /// advances solution by one time step
    pub fn step(&mut self) {
        if self.start_time.is_none() {
            self.start_time = Some(Instant::now());
        }

        // phase 1: compute dt (parallel reduction)
        self.shared.max_speed.store(0, Ordering::Relaxed);
        self.shared.phase.store(PHASE_COMPUTE_DT, Ordering::Release);
        self.shared.barrier.wait(); // start workers
        self.shared.barrier.wait(); // wait for completion

        let max_speed = f64::from_bits(self.shared.max_speed.load(Ordering::Acquire));
        let dt = self.cfl * self.dx / max_speed;
        self.shared.dt.store(dt.to_bits(), Ordering::Release);

        // phase 2: compute fluxes (parallel)
        self.shared
            .phase
            .store(PHASE_COMPUTE_FLUX, Ordering::Release);
        self.shared.barrier.wait();
        self.shared.barrier.wait();

        // phase 3: update conserved (parallel)
        self.shared.phase.store(PHASE_UPDATE, Ordering::Release);
        self.shared.barrier.wait();
        self.shared.barrier.wait();

        // boundary conditions (serial, small work)
        self.apply_bc();

        self.time += dt;
        self.step_count += 1;

        // return to idle
        self.shared.phase.store(PHASE_IDLE, Ordering::Release);
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

    /// returns solution
    pub fn solution(&self) -> Vec<(f64, f64, f64)> {
        let ng = self.nghost;
        let nc = self.ncells;
        let gamma = self.gamma;

        (ng..ng + nc)
            .map(|ii| {
                let rho = self.rho[ii];
                let vx = self.mom[ii] / rho;
                let ke = 0.5 * self.mom[ii] * self.mom[ii] / rho;
                let ie = self.energy[ii] - ke;
                let p = (gamma - 1.0) * ie;
                (rho, vx, p)
            })
            .collect()
    }
}

impl Drop for ParallelEuler1D {
    fn drop(&mut self) {
        // signal shutdown
        self.shared.phase.store(PHASE_SHUTDOWN, Ordering::Release);
        self.shared.barrier.wait();

        // join workers
        for worker in self.workers.drain(..) {
            let _ = worker.join();
        }
    }
}

// worker thread main loop
fn worker_loop(tid: usize, nthreads: usize, shared: Arc<SharedState>) {
    loop {
        // wait for work
        shared.barrier.wait();

        let phase = shared.phase.load(Ordering::Acquire);

        if phase == PHASE_SHUTDOWN {
            return;
        }

        match phase {
            PHASE_COMPUTE_DT => compute_dt_chunk(tid, nthreads, &shared),
            PHASE_COMPUTE_FLUX => compute_flux_chunk(tid, nthreads, &shared),
            PHASE_UPDATE => update_chunk(tid, nthreads, &shared),
            _ => {}
        }

        // signal completion
        shared.barrier.wait();
    }
}

// parallel dt computation with atomic max reduction
fn compute_dt_chunk(tid: usize, nthreads: usize, shared: &SharedState) {
    let ng = shared.nghost;
    let nc = shared.ncells;
    let gamma = shared.gamma;

    // chunk bounds
    let chunk_size = (nc + nthreads - 1) / nthreads;
    let start = ng + tid * chunk_size;
    let end = (start + chunk_size).min(ng + nc);

    if start >= end {
        return;
    }

    let rho_ptr = shared.rho.as_ptr();
    let mom_ptr = shared.mom.as_ptr();
    let energy_ptr = shared.energy.as_ptr();

    let mut local_max: f64 = 0.0;

    unsafe {
        for ii in start..end {
            let rho = *rho_ptr.add(ii);
            let mom = *mom_ptr.add(ii);
            let energy = *energy_ptr.add(ii);

            let vx = mom / rho;
            let ke = 0.5 * mom * mom / rho;
            let ie = energy - ke;
            let p = (gamma - 1.0) * ie;
            let cs = (gamma * p / rho).sqrt();
            let speed = vx.abs() + cs;

            if speed > local_max {
                local_max = speed;
            }
        }
    }

    // atomic max update
    let mut current = shared.max_speed.load(Ordering::Relaxed);
    loop {
        let current_f64 = f64::from_bits(current);
        if local_max <= current_f64 {
            break;
        }
        match shared.max_speed.compare_exchange_weak(
            current,
            local_max.to_bits(),
            Ordering::AcqRel,
            Ordering::Relaxed,
        ) {
            Ok(_) => break,
            Err(x) => current = x,
        }
    }
}

// parallel flux computation
fn compute_flux_chunk(tid: usize, nthreads: usize, shared: &SharedState) {
    let ntotal = shared.ntotal;
    let gamma = shared.gamma;

    // interfaces range from 1 to ntotal-1 (ntotal-1 interfaces)
    let n_interfaces = ntotal - 1;
    let chunk_size = (n_interfaces + nthreads - 1) / nthreads;
    let start = 1 + tid * chunk_size;
    let end = (start + chunk_size).min(ntotal);

    if start >= end {
        return;
    }

    let rho_ptr = shared.rho.as_ptr();
    let mom_ptr = shared.mom.as_ptr();
    let energy_ptr = shared.energy.as_ptr();
    let flux_rho_ptr = shared.flux_rho.as_ptr();
    let flux_mom_ptr = shared.flux_mom.as_ptr();
    let flux_energy_ptr = shared.flux_energy.as_ptr();

    unsafe {
        for ii in start..end {
            // pcm: left = cell ii-1, right = cell ii
            let rho_l = *rho_ptr.add(ii - 1);
            let mom_l = *mom_ptr.add(ii - 1);
            let energy_l = *energy_ptr.add(ii - 1);

            let rho_r = *rho_ptr.add(ii);
            let mom_r = *mom_ptr.add(ii);
            let energy_r = *energy_ptr.add(ii);

            // primitives
            let vx_l = mom_l / rho_l;
            let ke_l = 0.5 * mom_l * mom_l / rho_l;
            let ie_l = energy_l - ke_l;
            let p_l = (gamma - 1.0) * ie_l;

            let vx_r = mom_r / rho_r;
            let ke_r = 0.5 * mom_r * mom_r / rho_r;
            let ie_r = energy_r - ke_r;
            let p_r = (gamma - 1.0) * ie_r;

            // physical fluxes
            let f_rho_l = rho_l * vx_l;
            let f_mom_l = rho_l * vx_l * vx_l + p_l;
            let f_energy_l = (energy_l + p_l) * vx_l;

            let f_rho_r = rho_r * vx_r;
            let f_mom_r = rho_r * vx_r * vx_r + p_r;
            let f_energy_r = (energy_r + p_r) * vx_r;

            // hlle wave speeds
            let cs_l = (gamma * p_l / rho_l).sqrt();
            let cs_r = (gamma * p_r / rho_r).sqrt();

            let s_l = (vx_l - cs_l).min(vx_r - cs_r);
            let s_r = (vx_l + cs_l).max(vx_r + cs_r);

            // hlle flux
            if s_l >= 0.0 {
                *flux_rho_ptr.add(ii) = f_rho_l;
                *flux_mom_ptr.add(ii) = f_mom_l;
                *flux_energy_ptr.add(ii) = f_energy_l;
            } else if s_r <= 0.0 {
                *flux_rho_ptr.add(ii) = f_rho_r;
                *flux_mom_ptr.add(ii) = f_mom_r;
                *flux_energy_ptr.add(ii) = f_energy_r;
            } else {
                let denom = 1.0 / (s_r - s_l);
                *flux_rho_ptr.add(ii) =
                    (s_r * f_rho_l - s_l * f_rho_r + s_l * s_r * (rho_r - rho_l)) * denom;
                *flux_mom_ptr.add(ii) =
                    (s_r * f_mom_l - s_l * f_mom_r + s_l * s_r * (mom_r - mom_l)) * denom;
                *flux_energy_ptr.add(ii) = (s_r * f_energy_l - s_l * f_energy_r
                    + s_l * s_r * (energy_r - energy_l))
                    * denom;
            }
        }
    }
}

// parallel update
fn update_chunk(tid: usize, nthreads: usize, shared: &SharedState) {
    let ng = shared.nghost;
    let nc = shared.ncells;
    let dt = f64::from_bits(shared.dt.load(Ordering::Acquire));
    let dtdx = dt / shared.dx;

    // chunk bounds
    let chunk_size = (nc + nthreads - 1) / nthreads;
    let start = ng + tid * chunk_size;
    let end = (start + chunk_size).min(ng + nc);

    if start >= end {
        return;
    }

    // need mutable access to state arrays
    // safe because each thread writes to disjoint chunks
    let rho_ptr = shared.rho.as_ptr() as *mut f64;
    let mom_ptr = shared.mom.as_ptr() as *mut f64;
    let energy_ptr = shared.energy.as_ptr() as *mut f64;
    let flux_rho_ptr = shared.flux_rho.as_ptr() as *const f64;
    let flux_mom_ptr = shared.flux_mom.as_ptr() as *const f64;
    let flux_energy_ptr = shared.flux_energy.as_ptr() as *const f64;

    unsafe {
        for ii in start..end {
            *rho_ptr.add(ii) -= dtdx * (*flux_rho_ptr.add(ii + 1) - *flux_rho_ptr.add(ii));
            *mom_ptr.add(ii) -= dtdx * (*flux_mom_ptr.add(ii + 1) - *flux_mom_ptr.add(ii));
            *energy_ptr.add(ii) -= dtdx * (*flux_energy_ptr.add(ii + 1) - *flux_energy_ptr.add(ii));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parallel_solver_creation() {
        let solver = ParallelEuler1D::new(100, 0.0, 1.0, 1.4, 0.5, 4);
        assert_eq!(solver.ncells, 100);
        assert_eq!(solver.time, 0.0);
    }

    #[test]
    fn test_parallel_constant_state() {
        let mut solver = ParallelEuler1D::new(100, 0.0, 1.0, 1.4, 0.5, 4);
        solver.set_ic(|_| (1.0, 0.0, 1.0));

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
    fn test_parallel_matches_serial() {
        // run both solvers and compare results
        let mut serial = super::super::euler1d_simple::SimpleEuler1D::new(100, 0.0, 1.0, 1.4, 0.5);
        let mut parallel = ParallelEuler1D::new(100, 0.0, 1.0, 1.4, 0.5, 4);

        let ic = |x: f64| {
            if x < 0.5 {
                (1.0, 0.0, 1.0)
            } else {
                (0.125, 0.0, 0.1)
            }
        };

        serial.set_ic(ic);
        parallel.set_ic(ic);

        // run same number of steps
        for _ in 0..100 {
            serial.step();
            parallel.step();
        }

        let sol_s = serial.solution();
        let sol_p = parallel.solution();

        for ii in 0..sol_s.len() {
            assert!(
                (sol_s[ii].0 - sol_p[ii].0).abs() < 1e-10,
                "rho mismatch at {}",
                ii
            );
            assert!(
                (sol_s[ii].1 - sol_p[ii].1).abs() < 1e-10,
                "vx mismatch at {}",
                ii
            );
            assert!(
                (sol_s[ii].2 - sol_p[ii].2).abs() < 1e-10,
                "p mismatch at {}",
                ii
            );
        }
    }
}
