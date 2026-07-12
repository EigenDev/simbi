// =============================================================================
// body.rs
//
// core data structures for immersed boundary bodies. a body represents a
// discrete physical object (black hole, planet, rigid sphere) embedded in
// the fluid grid.
//
// uses a flat enum for capabilities. all state updates are immutable
// (copy-and-modify).
//
// usage:
//   let bh = Body::black_hole(0, pos, vel, 1.0, 0.1, 0.04, 10.0, 0.0, 0.2);
//   let moved = bh.at_position(new_pos);
// =============================================================================

use symbi_algebra::Tensor;
use symbi_ir::algebra::Scalar;

/// capability-specific data for a body.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum BodyKind<S: Scalar> {
    Passive,
    Gravitational {
        softening: S,
    },
    BlackHole {
        softening: S,
        sink_rate: S,
        accretion_radius: S,
        total_accreted_mass: S,
        accretion_rate: S,
        sink_delta: S, // 0 = torque-free, 1 = standard
    },
    Planet {
        softening: S,
        inertia: S,
        no_slip: bool,
    },
    RigidSphere {
        inertia: S,
        no_slip: bool,
    },
}

/// the penalization stack a body's surface runs (docs/design/50 property
/// algebra). config-static — parameters, never state, never checkpointed.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SurfaceSpec {
    /// the uniform-scaling drain: the validated accretor (p = 1).
    Drain,
    /// the porosity dial: `porosity` scales the drain channel, (1 - porosity)
    /// the wall channels; the wall rates are `k_eta_* c_s / dx`
    /// (multiplicative dials — zero is an exact off switch, so `k_eta_t = 0`
    /// is a free-slip surface).
    Porous { porosity: f64, k_eta_n: f64, k_eta_t: f64 },
    /// the torque-free accretor (docs/design/53): the drain plus a tangential
    /// anti-relaxation `lambda_t = -xi lambda_rho`, so the accreted mass carries
    /// no net angular momentum to the body (the Dittmann sink). `xi in [0, 1]`:
    /// `xi = 0` is the standard drain, `xi = 1` fully torque-free. isothermal.
    TorqueFree { xi: f64 },
}

/// a physical body embedded in the simulation grid.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Body<S: Scalar, const D: usize> {
    pub idx: usize,
    pub position: Tensor<S, D>,
    pub velocity: Tensor<S, D>,
    pub force: Tensor<S, D>,
    pub torque: Tensor<S, 3>,
    pub mass: S,
    pub radius: S,
    pub two_way_coupling: bool,
    pub kind: BodyKind<S>,
    /// the surface physics: which penalization stack acts at the boundary.
    /// kinematics stay on `kind`; this picks the baked kernel.
    pub surface: SurfaceSpec,
}

// -- factory functions --

impl<S: Scalar, const D: usize> Body<S, D> {
    fn base(idx: usize, position: Tensor<S, D>, velocity: Tensor<S, D>,
            mass: S, radius: S, kind: BodyKind<S>) -> Self {
        Self {
            idx, position, velocity,
            force: Tensor::zeros(),
            torque: Tensor::zeros(),
            mass, radius,
            two_way_coupling: false,
            kind,
            surface: SurfaceSpec::Drain,
        }
    }

    /// declare the surface stack (fluent; the default is the drain).
    pub fn with_surface(mut self, surface: SurfaceSpec) -> Self {
        self.surface = surface;
        self
    }

    pub fn passive(idx: usize, position: Tensor<S, D>, velocity: Tensor<S, D>,
                   mass: S, radius: S) -> Self {
        Self::base(idx, position, velocity, mass, radius, BodyKind::Passive)
    }

    pub fn gravitational(idx: usize, position: Tensor<S, D>, velocity: Tensor<S, D>,
                         mass: S, radius: S, softening: S) -> Self {
        Self::base(idx, position, velocity, mass, radius,
                   BodyKind::Gravitational { softening })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn black_hole(idx: usize, position: Tensor<S, D>, velocity: Tensor<S, D>,
                      mass: S, radius: S, softening: S,
                      sink_rate: S, sink_delta: S, accretion_radius: S) -> Self {
        Self::base(idx, position, velocity, mass, radius,
                   BodyKind::BlackHole {
                       softening,
                       sink_rate,
                       accretion_radius,
                       total_accreted_mass: S::ZERO,
                       accretion_rate: S::ZERO,
                       sink_delta,
                   })
    }

    pub fn planet(idx: usize, position: Tensor<S, D>, velocity: Tensor<S, D>,
                  mass: S, radius: S, inertia: S, no_slip: bool) -> Self {
        Self::base(idx, position, velocity, mass, radius,
                   BodyKind::Planet { softening: S::ZERO, inertia, no_slip })
    }

    pub fn rigid_sphere(idx: usize, position: Tensor<S, D>, velocity: Tensor<S, D>,
                        mass: S, radius: S, inertia: S, no_slip: bool) -> Self {
        Self::base(idx, position, velocity, mass, radius,
                   BodyKind::RigidSphere { inertia, no_slip })
    }
}

// -- immutable updates --

impl<S: Scalar, const D: usize> Body<S, D> {
    pub fn at_position(mut self, pos: Tensor<S, D>) -> Self {
        self.position = pos;
        self
    }

    pub fn with_velocity(mut self, vel: Tensor<S, D>) -> Self {
        self.velocity = vel;
        self
    }

    pub fn with_force(mut self, f: Tensor<S, D>) -> Self {
        self.force = f;
        self
    }

    pub fn with_torque(mut self, t: Tensor<S, 3>) -> Self {
        self.torque = t;
        self
    }

    pub fn with_mass(mut self, m: S) -> Self {
        self.mass = m;
        self
    }

    pub fn with_radius(mut self, r: S) -> Self {
        self.radius = r;
        self
    }

    pub fn with_two_way_coupling(mut self, flag: bool) -> Self {
        self.two_way_coupling = flag;
        self
    }
}

// -- capability queries --

impl<S: Scalar, const D: usize> Body<S, D> {
    pub fn has_gravity(&self) -> bool {
        matches!(self.kind,
            BodyKind::Gravitational { .. } |
            BodyKind::BlackHole { .. } |
            BodyKind::Planet { .. })
    }

    pub fn has_accretion(&self) -> bool {
        matches!(self.kind, BodyKind::BlackHole { .. })
    }

    pub fn has_rigid(&self) -> bool {
        matches!(self.kind,
            BodyKind::RigidSphere { .. } |
            BodyKind::Planet { .. })
    }

    /// softening length, or None if the body has no gravitational capability.
    pub fn softening(&self) -> Option<S> {
        match self.kind {
            BodyKind::Gravitational { softening, .. } => Some(softening),
            BodyKind::BlackHole { softening, .. } => Some(softening),
            BodyKind::Planet { softening, .. } => Some(softening),
            _ => None,
        }
    }

    /// accretion radius, or None if the body has no accretion capability.
    pub fn accretion_radius(&self) -> Option<S> {
        match self.kind {
            BodyKind::BlackHole { accretion_radius, .. } => Some(accretion_radius),
            _ => None,
        }
    }

    /// sink rate, or None if the body has no accretion capability.
    pub fn sink_rate(&self) -> Option<S> {
        match self.kind {
            BodyKind::BlackHole { sink_rate, .. } => Some(sink_rate),
            _ => None,
        }
    }

    /// sink delta (torque control), or None if the body has no accretion capability.
    pub fn sink_delta(&self) -> Option<S> {
        match self.kind {
            BodyKind::BlackHole { sink_delta, .. } => Some(sink_delta),
            _ => None,
        }
    }

    /// moment of inertia, or None if the body has no rigid capability.
    pub fn inertia(&self) -> Option<S> {
        match self.kind {
            BodyKind::RigidSphere { inertia, .. } => Some(inertia),
            BodyKind::Planet { inertia, .. } => Some(inertia),
            _ => None,
        }
    }

    /// no-slip flag, or None if the body has no rigid capability.
    pub fn no_slip(&self) -> Option<bool> {
        match self.kind {
            BodyKind::RigidSphere { no_slip, .. } => Some(no_slip),
            BodyKind::Planet { no_slip, .. } => Some(no_slip),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    type V2 = Tensor<f64, 2>;
    type V3 = Tensor<f64, 3>;

    #[test]
    fn passive_body() {
        let b = Body::passive(0, V2::new([1.0, 2.0]), V2::zeros(), 1.0, 0.1);
        assert!(!b.has_gravity());
        assert!(!b.has_accretion());
        assert!(!b.has_rigid());
    }

    #[test]
    fn gravitational_body() {
        let b = Body::gravitational(0, V2::new([0.0, 0.0]), V2::zeros(), 1.0, 0.1, 0.04);
        assert!(b.has_gravity());
        assert!(!b.has_accretion());
        assert_eq!(b.softening(), Some(0.04));
    }

    #[test]
    fn black_hole() {
        let b = Body::black_hole(
            0, V3::new([1.0, 0.0, 0.0]), V3::zeros(),
            1.0, 0.1, 0.04, 10.0, 0.0, 0.2,
        );
        assert!(b.has_gravity());
        assert!(b.has_accretion());
        assert_eq!(b.softening(), Some(0.04));
        assert_eq!(b.accretion_radius(), Some(0.2));
        assert_eq!(b.sink_rate(), Some(10.0));
        assert_eq!(b.sink_delta(), Some(0.0));
    }

    #[test]
    fn planet() {
        let b = Body::planet(0, V2::new([1.0, 0.0]), V2::zeros(), 0.001, 0.05, 0.1, true);
        assert!(b.has_gravity());
        assert!(b.has_rigid());
        assert!(!b.has_accretion());
        assert_eq!(b.inertia(), Some(0.1));
        assert_eq!(b.no_slip(), Some(true));
    }

    #[test]
    fn rigid_sphere() {
        let b = Body::rigid_sphere(0, V2::new([0.0, 0.0]), V2::zeros(), 1.0, 0.5, 0.3, false);
        assert!(!b.has_gravity());
        assert!(b.has_rigid());
        assert_eq!(b.inertia(), Some(0.3));
        assert_eq!(b.no_slip(), Some(false));
    }

    #[test]
    fn immutable_updates() {
        let b = Body::passive(0, V2::new([1.0, 2.0]), V2::zeros(), 1.0, 0.1);
        let moved = b.at_position(V2::new([3.0, 4.0]));
        assert_eq!(moved.position, V2::new([3.0, 4.0]));
        assert_eq!(b.position, V2::new([1.0, 2.0])); // original unchanged (copy)

        let fast = b.with_velocity(V2::new([1.0, 0.0]));
        assert_eq!(fast.velocity, V2::new([1.0, 0.0]));

        let heavy = b.with_mass(10.0);
        assert_eq!(heavy.mass, 10.0);
    }

    #[test]
    fn with_two_way_coupling() {
        let b = Body::passive(0, V2::zeros(), V2::zeros(), 1.0, 0.1)
            .with_two_way_coupling(true);
        assert!(b.two_way_coupling);
    }

    #[test]
    fn softening_none_on_passive() {
        let b = Body::passive(0, V2::zeros(), V2::zeros(), 1.0, 0.1);
        assert_eq!(b.softening(), None);
    }

    #[test]
    fn accretion_radius_none_on_grav() {
        let b = Body::gravitational(0, V2::zeros(), V2::zeros(), 1.0, 0.1, 0.04);
        assert_eq!(b.accretion_radius(), None);
    }

    #[test]
    fn inertia_none_on_passive() {
        let b = Body::passive(0, V2::zeros(), V2::zeros(), 1.0, 0.1);
        assert_eq!(b.inertia(), None);
    }
}
