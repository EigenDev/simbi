// =============================================================================
// metadata.rs
//
// simulation metadata: timing, physics parameters, runtime configuration.
// this is the global state that doesn't belong to any specific level.
// =============================================================================

// =============================================================================
// timestepping method
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Timestepping {
    Euler,
    Rk2,
    Rk3,
}

// =============================================================================
// boundary condition type
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoundaryCondition {
    Outflow,
    Reflecting,
    Periodic,
    Inflow,
}

// =============================================================================
// coordinate system
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CoordSystem {
    Cartesian,
    Spherical,
    Cylindrical,
}

// =============================================================================
// simulation metadata
// =============================================================================

#[derive(Debug, Clone)]
pub struct Metadata<const RANK: usize> {
    // timing
    pub time: f64,
    pub dt: f64,
    pub tend: f64,
    pub iteration: u64,

    // physics parameters
    pub gamma: f64,
    pub cfl: f64,

    // methods
    pub timestepping: Timestepping,
    pub coord_system: CoordSystem,

    // boundary conditions (per dimension, [lo, hi])
    pub boundary_conditions: [[BoundaryCondition; 2]; RANK],

    // checkpointing
    pub checkpoint_interval: f64,
    pub next_checkpoint_time: f64,
    pub checkpoint_index: u64,
}

impl<const RANK: usize> Metadata<RANK> {
    pub fn new(gamma: f64, tend: f64) -> Self {
        Self {
            time: 0.0,
            dt: 0.0,
            tend,
            iteration: 0,
            gamma,
            cfl: 0.4,
            timestepping: Timestepping::Rk2,
            coord_system: CoordSystem::Cartesian,
            boundary_conditions: [[BoundaryCondition::Outflow; 2]; RANK],
            checkpoint_interval: tend / 10.0,
            next_checkpoint_time: 0.0,
            checkpoint_index: 0,
        }
    }

    pub fn with_cfl(mut self, cfl: f64) -> Self {
        self.cfl = cfl;
        self
    }

    pub fn with_timestepping(mut self, ts: Timestepping) -> Self {
        self.timestepping = ts;
        self
    }

    pub fn with_coord_system(mut self, cs: CoordSystem) -> Self {
        self.coord_system = cs;
        self
    }

    pub fn with_boundary(
        mut self,
        dim: usize,
        lo: BoundaryCondition,
        hi: BoundaryCondition,
    ) -> Self {
        self.boundary_conditions[dim] = [lo, hi];
        self
    }

    pub fn with_checkpoint_interval(mut self, interval: f64) -> Self {
        self.checkpoint_interval = interval;
        self.next_checkpoint_time = interval;
        self
    }

    pub fn advance_time(&mut self, dt: f64) {
        self.time += dt;
        self.iteration += 1;
    }

    pub fn should_checkpoint(&self) -> bool {
        self.time >= self.next_checkpoint_time
    }

    pub fn mark_checkpoint(&mut self) {
        self.checkpoint_index += 1;
        self.next_checkpoint_time += self.checkpoint_interval;
    }

    pub fn is_finished(&self) -> bool {
        self.time >= self.tend
    }

    pub fn progress(&self) -> f64 {
        (self.time / self.tend).min(1.0)
    }
}

impl<const RANK: usize> Default for Metadata<RANK> {
    fn default() -> Self {
        Self::new(5.0 / 3.0, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metadata_creation() {
        let meta = Metadata::<1>::new(1.4, 0.5);

        assert_eq!(meta.gamma, 1.4);
        assert_eq!(meta.tend, 0.5);
        assert_eq!(meta.time, 0.0);
        assert_eq!(meta.iteration, 0);
    }

    #[test]
    fn metadata_builder_pattern() {
        let meta = Metadata::<2>::new(1.4, 1.0)
            .with_cfl(0.5)
            .with_timestepping(Timestepping::Rk3)
            .with_boundary(0, BoundaryCondition::Periodic, BoundaryCondition::Periodic)
            .with_checkpoint_interval(0.1);

        assert_eq!(meta.cfl, 0.5);
        assert_eq!(meta.timestepping, Timestepping::Rk3);
        assert_eq!(
            meta.boundary_conditions[0],
            [BoundaryCondition::Periodic; 2]
        );
        assert_eq!(meta.checkpoint_interval, 0.1);
    }

    #[test]
    fn metadata_time_advance() {
        let mut meta = Metadata::<1>::new(1.4, 1.0);

        meta.advance_time(0.1);
        assert!((meta.time - 0.1).abs() < 1e-10);
        assert_eq!(meta.iteration, 1);

        meta.advance_time(0.2);
        assert!((meta.time - 0.3).abs() < 1e-10);
        assert_eq!(meta.iteration, 2);
    }

    #[test]
    fn metadata_progress() {
        let mut meta = Metadata::<1>::new(1.4, 1.0);

        assert_eq!(meta.progress(), 0.0);

        meta.time = 0.5;
        assert!((meta.progress() - 0.5).abs() < 1e-10);

        meta.time = 1.0;
        assert!((meta.progress() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn metadata_checkpoint() {
        let mut meta = Metadata::<1>::new(1.4, 1.0).with_checkpoint_interval(0.25);

        assert!(!meta.should_checkpoint());

        meta.time = 0.25;
        assert!(meta.should_checkpoint());

        meta.mark_checkpoint();
        assert_eq!(meta.checkpoint_index, 1);
        assert!((meta.next_checkpoint_time - 0.5).abs() < 1e-10);
        assert!(!meta.should_checkpoint());
    }
}
