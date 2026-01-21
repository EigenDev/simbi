// =============================================================================
// timestepping.rs
//
// time integration schemes for hyperbolic pdes.
// implements explicit runge-kutta methods (rk2, rk3) with automatic cfl
// time step computation.
//
// design:
//   - explicit time stepping (no implicit solvers)
//   - tvd runge-kutta schemes (ssp-rk2, ssp-rk3)
//   - automatic dt from cfl condition
//   - boundary condition application
//
// usage:
//   let integrator = Rk3Integrator::new(cfl_number);
//   let dt = integrator.compute_dt(&fields, dx, gamma);
//   integrator.step(&mut fields, dt, dx, gamma, |u, f| riemann_solver(u, f));
// =============================================================================

use super::{Conserved1D, Primitive1D};

// =============================================================================
// cfl condition
// =============================================================================

/// computes maximum stable time step from cfl condition.
/// dt = cfl * dx / max(|v| + c)
///
/// # arguments
/// * `primitives` - array of primitive states
/// * `dx` - spatial resolution
/// * `gamma` - adiabatic index
/// * `cfl` - cfl number (typically 0.4-0.8)
///
/// # returns
/// maximum stable time step
pub fn compute_dt_cfl(primitives: &[Primitive1D], dx: f64, gamma: f64, cfl: f64) -> f64 {
    let max_speed = primitives
        .iter()
        .map(|p| p.max_wave_speed(gamma))
        .fold(0.0f64, f64::max);

    if max_speed > 0.0 {
        cfl * dx / max_speed
    } else {
        dx // fallback if no motion
    }
}

// =============================================================================
// boundary conditions
// =============================================================================

/// boundary condition types
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum BoundaryCondition {
    /// periodic boundaries (wrap around)
    Periodic,

    /// outflow boundaries (zero gradient)
    Outflow,

    /// reflecting boundaries (mirror velocity)
    Reflecting,

    /// fixed boundaries (constant state)
    Fixed,
}

/// applies boundary conditions to conserved state array.
///
/// # arguments
/// * `conserved` - array of conserved states (includes ghost zones)
/// * `bc` - boundary condition type
/// * `nghost` - number of ghost zones on each side
pub fn apply_boundary_conditions(
    conserved: &mut [Conserved1D],
    bc: BoundaryCondition,
    nghost: usize,
) {
    let n = conserved.len();
    let interior_start = nghost;
    let interior_end = n - nghost;

    match bc {
        BoundaryCondition::Periodic => {
            // left ghost = right interior
            for i in 0..nghost {
                conserved[i] = conserved[interior_end - nghost + i];
            }
            // right ghost = left interior
            for i in 0..nghost {
                conserved[interior_end + i] = conserved[interior_start + i];
            }
        }
        BoundaryCondition::Outflow => {
            // left ghost = leftmost interior
            for i in 0..nghost {
                conserved[i] = conserved[interior_start];
            }
            // right ghost = rightmost interior
            for i in 0..nghost {
                conserved[interior_end + i] = conserved[interior_end - 1];
            }
        }
        BoundaryCondition::Reflecting => {
            // left ghost = mirror with flipped velocity
            for i in 0..nghost {
                let source = conserved[interior_start + (nghost - 1 - i)];
                conserved[i] = Conserved1D {
                    rho: source.rho,
                    mom: -source.mom, // flip momentum
                    energy: source.energy,
                };
            }
            // right ghost = mirror with flipped velocity
            for i in 0..nghost {
                let source = conserved[interior_end - 1 - i];
                conserved[interior_end + i] = Conserved1D {
                    rho: source.rho,
                    mom: -source.mom,
                    energy: source.energy,
                };
            }
        }
        BoundaryCondition::Fixed => {
            // ghost zones remain unchanged (set externally)
        }
    }
}

// =============================================================================
// runge-kutta integrators
// =============================================================================

/// explicit euler (rk1) time integrator.
/// first-order accurate in time.
/// u^{n+1} = u^n + dt * L(u^n)
pub struct Rk1Integrator {
    pub cfl: f64,
}

impl Rk1Integrator {
    /// creates a new rk1 integrator with given cfl number.
    pub fn new(cfl: f64) -> Self {
        Self { cfl }
    }

    /// computes time step from cfl condition.
    pub fn compute_dt(&self, primitives: &[Primitive1D], dx: f64, gamma: f64) -> f64 {
        compute_dt_cfl(primitives, dx, gamma, self.cfl)
    }

    /// performs one time step: u^{n+1} = u^n + dt * L(u^n)
    ///
    /// # arguments
    /// * `conserved` - conserved state (modified in place)
    /// * `dt` - time step
    /// * `spatial_operator` - computes du/dt = L(u)
    pub fn step<F>(&self, conserved: &mut [Conserved1D], dt: f64, mut spatial_operator: F)
    where
        F: FnMut(&[Conserved1D]) -> Vec<Conserved1D>,
    {
        let dudt = spatial_operator(conserved);

        for (u, du) in conserved.iter_mut().zip(dudt.iter()) {
            *u = u.add(du.scale(dt));
        }
    }
}

/// ssp-rk2 (strong stability preserving runge-kutta 2) integrator.
/// second-order accurate in time.
///
/// u^* = u^n + dt * L(u^n)
/// u^{n+1} = 0.5 * u^n + 0.5 * u^* + 0.5 * dt * L(u^*)
pub struct Rk2Integrator {
    pub cfl: f64,
}

impl Rk2Integrator {
    /// creates a new rk2 integrator with given cfl number.
    pub fn new(cfl: f64) -> Self {
        Self { cfl }
    }

    /// computes time step from cfl condition.
    pub fn compute_dt(&self, primitives: &[Primitive1D], dx: f64, gamma: f64) -> f64 {
        compute_dt_cfl(primitives, dx, gamma, self.cfl)
    }

    /// performs one rk2 time step.
    pub fn step<F>(&self, conserved: &mut [Conserved1D], dt: f64, mut spatial_operator: F)
    where
        F: FnMut(&[Conserved1D]) -> Vec<Conserved1D>,
    {
        let n = conserved.len();
        let u_n: Vec<Conserved1D> = conserved.to_vec();

        // stage 1: u^* = u^n + dt * L(u^n)
        let dudt1 = spatial_operator(&u_n);
        let mut u_star = vec![Conserved1D::new(0.0, 0.0, 0.0); n];
        for i in 0..n {
            u_star[i] = u_n[i].add(dudt1[i].scale(dt));
        }

        // stage 2: u^{n+1} = 0.5*u^n + 0.5*u^* + 0.5*dt*L(u^*)
        let dudt2 = spatial_operator(&u_star);
        for i in 0..n {
            conserved[i] = u_n[i]
                .scale(0.5)
                .add(u_star[i].scale(0.5))
                .add(dudt2[i].scale(0.5 * dt));
        }
    }
}

/// ssp-rk3 (strong stability preserving runge-kutta 3) integrator.
/// third-order accurate in time.
///
/// u^(1) = u^n + dt * L(u^n)
/// u^(2) = 0.75*u^n + 0.25*u^(1) + 0.25*dt*L(u^(1))
/// u^{n+1} = 1/3*u^n + 2/3*u^(2) + 2/3*dt*L(u^(2))
pub struct Rk3Integrator {
    pub cfl: f64,
}

impl Rk3Integrator {
    /// creates a new rk3 integrator with given cfl number.
    pub fn new(cfl: f64) -> Self {
        Self { cfl }
    }

    /// computes time step from cfl condition.
    pub fn compute_dt(&self, primitives: &[Primitive1D], dx: f64, gamma: f64) -> f64 {
        compute_dt_cfl(primitives, dx, gamma, self.cfl)
    }

    /// performs one rk3 time step.
    pub fn step<F>(&self, conserved: &mut [Conserved1D], dt: f64, mut spatial_operator: F)
    where
        F: FnMut(&[Conserved1D]) -> Vec<Conserved1D>,
    {
        let n = conserved.len();
        let u_n: Vec<Conserved1D> = conserved.to_vec();

        // stage 1: u^(1) = u^n + dt * L(u^n)
        let dudt1 = spatial_operator(&u_n);
        let mut u1 = vec![Conserved1D::new(0.0, 0.0, 0.0); n];
        for i in 0..n {
            u1[i] = u_n[i].add(dudt1[i].scale(dt));
        }

        // stage 2: u^(2) = 0.75*u^n + 0.25*u^(1) + 0.25*dt*L(u^(1))
        let dudt2 = spatial_operator(&u1);
        let mut u2 = vec![Conserved1D::new(0.0, 0.0, 0.0); n];
        for i in 0..n {
            u2[i] = u_n[i]
                .scale(0.75)
                .add(u1[i].scale(0.25))
                .add(dudt2[i].scale(0.25 * dt));
        }

        // stage 3: u^{n+1} = 1/3*u^n + 2/3*u^(2) + 2/3*dt*L(u^(2))
        let dudt3 = spatial_operator(&u2);
        for i in 0..n {
            conserved[i] = u_n[i]
                .scale(1.0 / 3.0)
                .add(u2[i].scale(2.0 / 3.0))
                .add(dudt3[i].scale(2.0 / 3.0 * dt));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cfl_computation() {
        let gamma = 1.4;
        let dx = 0.01;
        let cfl = 0.5;

        // stationary fluid
        let prims = vec![Primitive1D::new(1.0, 0.0, 1.0)];
        let dt = compute_dt_cfl(&prims, dx, gamma, cfl);

        // dt = cfl * dx / (|v| + c) = 0.5 * 0.01 / 1.183... ≈ 0.00422
        assert!(dt > 0.004 && dt < 0.005);
    }

    #[test]
    fn test_periodic_boundaries() {
        let mut cons = vec![
            Conserved1D::new(0.0, 0.0, 0.0), // left ghost
            Conserved1D::new(1.0, 1.0, 1.0), // interior
            Conserved1D::new(2.0, 2.0, 2.0), // interior
            Conserved1D::new(0.0, 0.0, 0.0), // right ghost
        ];

        apply_boundary_conditions(&mut cons, BoundaryCondition::Periodic, 1);

        // left ghost should equal rightmost interior
        assert_eq!(cons[0].rho, 2.0);
        // right ghost should equal leftmost interior
        assert_eq!(cons[3].rho, 1.0);
    }

    #[test]
    fn test_outflow_boundaries() {
        let mut cons = vec![
            Conserved1D::new(0.0, 0.0, 0.0), // left ghost
            Conserved1D::new(1.0, 1.0, 1.0), // interior
            Conserved1D::new(2.0, 2.0, 2.0), // interior
            Conserved1D::new(0.0, 0.0, 0.0), // right ghost
        ];

        apply_boundary_conditions(&mut cons, BoundaryCondition::Outflow, 1);

        // left ghost = leftmost interior
        assert_eq!(cons[0].rho, 1.0);
        // right ghost = rightmost interior
        assert_eq!(cons[3].rho, 2.0);
    }

    #[test]
    fn test_reflecting_boundaries() {
        let mut cons = vec![
            Conserved1D::new(0.0, 0.0, 0.0), // left ghost
            Conserved1D::new(1.0, 2.0, 3.0), // interior (rho=1, mom=2, E=3)
            Conserved1D::new(2.0, 4.0, 5.0), // interior
            Conserved1D::new(0.0, 0.0, 0.0), // right ghost
        ];

        apply_boundary_conditions(&mut cons, BoundaryCondition::Reflecting, 1);

        // left ghost should mirror with flipped momentum
        assert_eq!(cons[0].rho, 1.0);
        assert_eq!(cons[0].mom, -2.0); // flipped
        assert_eq!(cons[0].energy, 3.0);

        // right ghost should mirror with flipped momentum
        assert_eq!(cons[3].rho, 2.0);
        assert_eq!(cons[3].mom, -4.0); // flipped
    }

    #[test]
    fn test_rk1_stability() {
        // simple test: constant state should remain constant
        let integrator = Rk1Integrator::new(0.5);

        let mut cons = vec![
            Conserved1D::new(1.0, 0.0, 2.5),
            Conserved1D::new(1.0, 0.0, 2.5),
        ];

        // spatial operator returns zero (no change)
        integrator.step(&mut cons, 0.01, |_| {
            vec![
                Conserved1D::new(0.0, 0.0, 0.0),
                Conserved1D::new(0.0, 0.0, 0.0),
            ]
        });

        // state should be unchanged
        assert!((cons[0].rho - 1.0).abs() < 1e-10);
        assert!((cons[1].rho - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_rk2_stages() {
        let integrator = Rk2Integrator::new(0.5);

        let mut cons = vec![Conserved1D::new(1.0, 0.0, 2.5)];
        let dt = 0.1;

        // linear growth: du/dt = 1
        integrator.step(&mut cons, dt, |_| vec![Conserved1D::new(1.0, 0.0, 0.0)]);

        // rk2 with constant derivative: u1 = u0 + dt, u_new = 0.5*u0 + 0.5*u1 + 0.5*dt
        // u_new = 0.5*1.0 + 0.5*(1.0+0.1) + 0.5*0.1 = 0.5 + 0.55 + 0.05 = 1.1
        assert!((cons[0].rho - 1.1).abs() < 1e-10);
    }

    #[test]
    fn test_rk3_stages() {
        let integrator = Rk3Integrator::new(0.5);

        let mut cons = vec![Conserved1D::new(1.0, 0.0, 2.5)];
        let dt = 0.1;

        // constant derivative
        integrator.step(&mut cons, dt, |_| vec![Conserved1D::new(1.0, 0.0, 0.0)]);

        // with constant du/dt = 1, rk3 should give u_new ≈ u0 + dt = 1.1
        assert!((cons[0].rho - 1.1).abs() < 1e-10);
    }
}
