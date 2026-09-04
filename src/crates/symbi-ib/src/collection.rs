// =============================================================================
// collection.rs
//
// heterogeneous body container with subcycle interpolation. the body list is
// partitioned [source bodies | fragments]: source bodies occupy the baked
// gravity/accretion kernel slots (at most MAX_SOURCE_BODIES), fragments are
// wall-only rigid bodies in unbounded number, reached through per-instance
// dispatch outside the baked source fan. provides capability-filtered
// iteration and supports
// snapshot/interpolate/finalize for AMR subcycling.
//
// usage:
//   let coll = BodyCollection::new()
//       .add(body1)
//       .add(body2)
//       .with_name("binary_system");
//   coll.visit_gravitational(|b| { ... });
// =============================================================================

use crate::body::{Body, BodyKind};
use symbi_algebra::{OrderedNumeric, Tensor};
use symbi_carrier::Scalar;

/// number of body slots statically unrolled into the baked gravity/accretion
/// source kernels. 2 covers binary systems. source bodies (gravity-on-gas or
/// sinks) are what occupy slots; wall-only fragments live beyond the slot prefix
/// and are reached through the per-instance penalization dispatch.
pub const MAX_SOURCE_BODIES: usize = 2;

/// orbital parameters for a binary system.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BinaryParams<S: Scalar> {
    pub total_mass: S,
    pub semi_major: S,
    pub eccentricity: S,
    pub mass_ratio: S,
    pub orbital_period: S,
    pub is_circular: bool,
    pub prescribed_motion: bool,
}

impl<S: Scalar + OrderedNumeric> BinaryParams<S> {
    pub fn new(total_mass: S, semi_major: S, eccentricity: S, mass_ratio: S) -> Self {
        let two_pi = S::from_f64(2.0 * std::f64::consts::PI);
        let a3 = semi_major * semi_major * semi_major;
        let orbital_period = two_pi * (a3 / total_mass).sqrt();
        let is_circular = eccentricity < S::from_f64(1e-10);
        Self {
            total_mass,
            semi_major,
            eccentricity,
            mass_ratio,
            orbital_period,
            is_circular,
            prescribed_motion: true,
        }
    }
}

/// reference frame for the body system.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReferenceFrame {
    Inertial,
    Corotating,
    Stationary,
}

/// collection of immersed bodies with subcycle interpolation.
#[derive(Clone, Debug)]
pub struct BodyCollection<S: Scalar, const D: usize> {
    bodies: Vec<Body<S, D>>,
    pub name: String,
    pub frame: ReferenceFrame,
    pub binary_params: Option<BinaryParams<S>>,

    // explicit binary-system capability flag: selects the dittmann & ryan (2020)
    // 2d accretion weight and enables prescribed orbital advance. set explicitly by
    // the builder, so renaming the display `name` leaves the physics untouched.
    binary: bool,

    // the source/fragment partition point: bodies[..n_sources] occupy the baked
    // source-kernel slots, bodies[n_sources..] are wall-only fragments. sources
    // are always added before fragments, so the prefix is contiguous and body
    // indices are stable for the lifetime of the collection.
    n_sources: usize,

    // subcycle snapshot
    bodies_n: Vec<Body<S, D>>,
    bodies_next: Vec<Body<S, D>>,
    has_snapshot: bool,
}

impl<S: Scalar, const D: usize> BodyCollection<S, D> {
    pub fn new() -> Self {
        Self {
            bodies: Vec::new(),
            name: "untitled".to_string(),
            frame: ReferenceFrame::Inertial,
            binary_params: None,
            binary: false,
            n_sources: 0,
            bodies_n: Vec::new(),
            bodies_next: Vec::new(),
            has_snapshot: false,
        }
    }

    // -- builder pattern --

    /// add a source body: it occupies the next baked source-kernel slot
    /// (gravity-on-gas, sink accretion). every source must be added before
    /// any fragment so the slot prefix stays contiguous.
    pub fn add(mut self, body: Body<S, D>) -> Self {
        assert!(
            self.n_sources < MAX_SOURCE_BODIES,
            "source-body slots are full ({MAX_SOURCE_BODIES}); wall-only bodies go through add_fragment"
        );
        assert!(
            self.n_sources == self.bodies.len(),
            "source bodies must be added before fragments"
        );
        assert!(
            !body.spec.magnetic.requires_material_drain()
                || body.spec.surface.provides_material_drain(),
            "a magnetic-slip body closes its slip coefficient on the drain time tau_rho, so its \
             surface must remove mass; pair the slip coupling with a draining surface (Drain, \
             TorqueFree, or Porous with porosity > 0)"
        );
        self.bodies.push(body);
        self.n_sources += 1;
        self
    }

    /// add a wall-only fragment: a rigid body that acts on the gas through its
    /// wall alone, so it lives outside the baked source-kernel slots and the
    /// fragment count is unbounded. the body's index is overwritten with its
    /// list position so per-body dispatch and delta ledgers stay aligned.
    pub fn add_fragment(mut self, mut body: Body<S, D>) -> Self {
        assert!(
            body.has_rigid() && !body.has_gravity() && !body.has_accretion(),
            "a fragment must be a wall-only rigid body (no gravity, no accretion)"
        );
        body.idx = self.bodies.len();
        self.bodies.push(body);
        self
    }

    pub fn with_name(mut self, name: &str) -> Self {
        self.name = name.to_string();
        self
    }

    pub fn with_frame(mut self, frame: ReferenceFrame) -> Self {
        self.frame = frame;
        self
    }

    pub fn with_binary_params(mut self, params: BinaryParams<S>) -> Self {
        self.binary_params = Some(params);
        self
    }

    /// flag this collection as a binary system (accretion-weight + orbital-advance
    /// capability). set explicitly by this builder call, independent of the display name.
    pub fn as_binary(mut self) -> Self {
        self.binary = true;
        self
    }

    // -- accessors --

    pub fn len(&self) -> usize {
        self.bodies.len()
    }
    pub fn is_empty(&self) -> bool {
        self.bodies.is_empty()
    }

    /// whether some body needs the backward feedback reduction: a two-way-coupled
    /// body (it moves in response to the gas) or a black-hole sink (its accreted
    /// mass + rate come from the reduction). for a one-way fixed gravitational mass
    /// the feedback would land in force/torque diagnostics alone, so the whole
    /// reduction pass is skipped for it (a large saving: kepler-like setups).
    pub fn needs_feedback(&self) -> bool {
        self.bodies
            .iter()
            .any(|b| b.two_way_coupling || matches!(b.kind, BodyKind::BlackHole { .. }))
    }

    pub fn get(&self, idx: usize) -> &Body<S, D> {
        &self.bodies[idx]
    }

    pub fn get_mut(&mut self, idx: usize) -> &mut Body<S, D> {
        &mut self.bodies[idx]
    }

    pub fn bodies(&self) -> &[Body<S, D>] {
        &self.bodies
    }

    /// the bodies occupying baked source-kernel slots (gravity + accretion fan).
    pub fn sources(&self) -> &[Body<S, D>] {
        &self.bodies[..self.n_sources]
    }

    /// the wall-only fragments beyond the source-slot prefix.
    pub fn fragments(&self) -> &[Body<S, D>] {
        &self.bodies[self.n_sources..]
    }

    pub fn source_count(&self) -> usize {
        self.n_sources
    }

    pub fn fragment_count(&self) -> usize {
        self.bodies.len() - self.n_sources
    }

    pub fn is_binary(&self) -> bool {
        self.binary
    }

    // -- capability-filtered visitors --

    pub fn visit_all(&self, mut visitor: impl FnMut(&Body<S, D>)) {
        for body in &self.bodies {
            visitor(body);
        }
    }

    pub fn visit_all_mut(&mut self, mut visitor: impl FnMut(&mut Body<S, D>)) {
        for body in &mut self.bodies {
            visitor(body);
        }
    }

    pub fn visit_gravitational(&self, mut visitor: impl FnMut(&Body<S, D>)) {
        for body in &self.bodies {
            if body.has_gravity() {
                visitor(body);
            }
        }
    }

    pub fn visit_accretion(&self, mut visitor: impl FnMut(&Body<S, D>)) {
        for body in &self.bodies {
            if body.has_accretion() {
                visitor(body);
            }
        }
    }

    pub fn visit_rigid(&self, mut visitor: impl FnMut(&Body<S, D>)) {
        for body in &self.bodies {
            if body.has_rigid() {
                visitor(body);
            }
        }
    }

    pub fn gravitational_count(&self) -> usize {
        self.bodies.iter().filter(|b| b.has_gravity()).count()
    }

    pub fn accretion_count(&self) -> usize {
        self.bodies.iter().filter(|b| b.has_accretion()).count()
    }

    pub fn rigid_count(&self) -> usize {
        self.bodies.iter().filter(|b| b.has_rigid()).count()
    }

    // -- subcycle interpolation --

    /// store current bodies as t^n, `advanced` as t^{n+1}.
    pub fn snapshot(&mut self, advanced: &[Body<S, D>]) {
        self.bodies_n = self.bodies.clone();
        self.bodies_next = advanced.to_vec();
        self.has_snapshot = true;
    }

    pub fn has_snapshot(&self) -> bool {
        self.has_snapshot
    }

    /// set bodies to lerp between t^n and t^{n+1}.
    /// alpha=0 -> t^n, alpha=1 -> t^{n+1}.
    pub fn interpolate_to(&mut self, alpha: S) {
        if !self.has_snapshot {
            return;
        }
        let one_minus = S::ONE - alpha;
        for ii in 0..self.bodies.len() {
            self.bodies[ii].position = self.bodies_n[ii].position.scale(one_minus)
                + self.bodies_next[ii].position.scale(alpha);
            self.bodies[ii].velocity = self.bodies_n[ii].velocity.scale(one_minus)
                + self.bodies_next[ii].velocity.scale(alpha);
        }
    }

    /// finalize to t^{n+1} state, clear snapshot.
    pub fn finalize_advance(&mut self) {
        if !self.has_snapshot {
            return;
        }
        self.bodies = self.bodies_next.clone();
        self.has_snapshot = false;
    }

    /// restore to t^n state.
    pub fn restore_from_snapshot(&mut self) {
        if !self.has_snapshot {
            return;
        }
        self.bodies = self.bodies_n.clone();
    }
}

impl<S: Scalar, const D: usize> Default for BodyCollection<S, D> {
    fn default() -> Self {
        Self::new()
    }
}

// -- binary system factory --

/// create a binary system from individual body parameters.
/// dispatches to black_hole or gravitational body based on sink rates.
#[allow(clippy::too_many_arguments)]
pub fn create_binary_system<S: Scalar + OrderedNumeric, const D: usize>(
    pos1: Tensor<S, D>,
    vel1: Tensor<S, D>,
    mass1: S,
    radius1: S,
    softening1: S,
    pos2: Tensor<S, D>,
    vel2: Tensor<S, D>,
    mass2: S,
    radius2: S,
    softening2: S,
    sink_rate1: S,
    sink_rate2: S,
    accr_radius1: S,
    accr_radius2: S,
    sink_delta1: S,
    sink_delta2: S,
) -> BodyCollection<S, D> {
    let make = |idx, pos, vel, mass, radius, softening, sr, sd, ar| {
        if sr > S::ZERO {
            Body::black_hole(idx, pos, vel, mass, radius, softening, sr, sd, ar)
        } else {
            Body::gravitational(idx, pos, vel, mass, radius, softening)
        }
    };

    let b1 = make(
        0,
        pos1,
        vel1,
        mass1,
        radius1,
        softening1,
        sink_rate1,
        sink_delta1,
        accr_radius1,
    );
    let b2 = make(
        1,
        pos2,
        vel2,
        mass2,
        radius2,
        softening2,
        sink_rate2,
        sink_delta2,
        accr_radius2,
    );

    BodyCollection::new()
        .add(b1)
        .add(b2)
        .with_name("binary_system")
        .as_binary()
}

#[cfg(test)]
mod tests {
    use super::*;

    type V2 = Tensor<f64, 2>;

    fn grav(idx: usize, x: f64, y: f64) -> Body<f64, 2> {
        Body::gravitational(idx, V2::new([x, y]), V2::zeros(), 1.0, 0.1, 0.04)
    }

    #[test]
    fn builder() {
        let coll = BodyCollection::new()
            .add(grav(0, 1.0, 0.0))
            .add(grav(1, -1.0, 0.0))
            .with_name("binary_system")
            .as_binary()
            .with_frame(ReferenceFrame::Inertial);
        assert_eq!(coll.len(), 2);
        assert!(coll.is_binary());
        assert_eq!(coll.frame, ReferenceFrame::Inertial);
    }

    #[test]
    #[should_panic(expected = "source-body slots are full")]
    fn add_overflow() {
        BodyCollection::new()
            .add(grav(0, 0.0, 0.0))
            .add(grav(1, 1.0, 0.0))
            .add(grav(2, 2.0, 0.0));
    }

    fn frag(idx: usize, x: f64, y: f64) -> Body<f64, 2> {
        Body::rigid_sphere(idx, V2::new([x, y]), V2::zeros(), 1.0, 0.1, 0.01, true)
    }

    #[test]
    fn fragments_are_unbounded_and_partitioned() {
        let mut coll = BodyCollection::new().add(grav(0, 0.0, 0.0));
        for ii in 0..8 {
            coll = coll.add_fragment(frag(0, ii as f64, 0.0));
        }
        assert_eq!(coll.len(), 9);
        assert_eq!(coll.source_count(), 1);
        assert_eq!(coll.fragment_count(), 8);
        assert_eq!(coll.sources().len(), 1);
        assert_eq!(coll.fragments().len(), 8);
        // fragment indices are their list positions, source prefix first
        for (ii, f) in coll.fragments().iter().enumerate() {
            assert_eq!(f.idx, 1 + ii);
        }
    }

    #[test]
    #[should_panic(expected = "source bodies must be added before fragments")]
    fn source_after_fragment_panics() {
        BodyCollection::new()
            .add_fragment(frag(0, 0.0, 0.0))
            .add(grav(1, 1.0, 0.0));
    }

    #[test]
    #[should_panic(expected = "wall-only rigid body")]
    fn gravitational_fragment_rejected() {
        BodyCollection::new().add_fragment(grav(0, 0.0, 0.0));
    }

    use crate::{MagneticSpec, SurfaceSpec};

    fn slip_accretor(surface: Option<SurfaceSpec>) -> Body<f64, 2> {
        let b = Body::<f64, 2>::black_hole(0, V2::zeros(), V2::zeros(), 1.0, 0.1, 0.04, 1.0, 1.0, 0.2);
        let b = match surface {
            Some(s) => b.with_surface(s),
            None => b, // the default surface is the drain
        };
        b.with_magnetic(MagneticSpec::Slip {
            diffusivity_ratio: 2.0,
            shell_width: 0.05,
            slip_length_ratio: 1.0,
            field_regularization: 0.1,
            placement: 0.0,
        })
    }

    // a slip coupling closes its coefficient on the drain time, so a body carrying it must drain.
    // the default (drain) surface is accepted.
    #[test]
    fn a_magnetic_slip_on_a_draining_accretor_is_accepted() {
        let coll = BodyCollection::new().add(slip_accretor(None));
        assert!(coll.get(0).spec.magnetic.requires_material_drain());
    }

    // a sealed porous wall supplies no drain time; pairing slip with it is refused at the door.
    #[test]
    #[should_panic(expected = "surface must remove mass")]
    fn a_magnetic_slip_without_a_draining_surface_is_refused() {
        let sealed = SurfaceSpec::Porous { porosity: 0.0, k_eta_n: 50.0, k_eta_t: 50.0 };
        BodyCollection::new().add(slip_accretor(Some(sealed)));
    }

    #[test]
    fn capability_counts() {
        let coll = BodyCollection::new()
            .add(grav(0, 1.0, 0.0))
            .add(Body::passive(1, V2::zeros(), V2::zeros(), 1.0, 0.1));
        assert_eq!(coll.gravitational_count(), 1);
        assert_eq!(coll.accretion_count(), 0);
    }

    #[test]
    fn visit_gravitational() {
        let coll = BodyCollection::new()
            .add(grav(0, 1.0, 0.0))
            .add(Body::passive(1, V2::zeros(), V2::zeros(), 1.0, 0.1));
        let mut count = 0;
        coll.visit_gravitational(|_| count += 1);
        assert_eq!(count, 1);
    }

    #[test]
    fn subcycle_interpolation() {
        let b0 = grav(0, 0.0, 0.0);
        let b1 = grav(1, 2.0, 0.0);
        let mut coll = BodyCollection::new().add(b0).add(b1);

        let advanced = vec![grav(0, 1.0, 0.0), grav(1, 3.0, 0.0)];
        coll.snapshot(&advanced);
        assert!(coll.has_snapshot());

        coll.interpolate_to(0.5);
        assert!((coll.get(0).position[0] - 0.5).abs() < 1e-14);
        assert!((coll.get(1).position[0] - 2.5).abs() < 1e-14);

        coll.finalize_advance();
        assert!(!coll.has_snapshot());
        assert!((coll.get(0).position[0] - 1.0).abs() < 1e-14);
    }

    #[test]
    fn restore_from_snapshot() {
        let mut coll = BodyCollection::new().add(grav(0, 0.0, 0.0));
        let advanced = vec![grav(0, 5.0, 0.0)];
        coll.snapshot(&advanced);
        coll.interpolate_to(1.0);
        coll.restore_from_snapshot();
        assert!((coll.get(0).position[0]).abs() < 1e-14);
    }

    #[test]
    fn binary_params_kepler() {
        let bp = BinaryParams::new(2.0_f64, 1.0, 0.0, 1.0);
        // T = 2 pi sqrt(a^3 / M) = 2 pi sqrt(1/2)
        let expected = 2.0 * std::f64::consts::PI * (0.5_f64).sqrt();
        assert!((bp.orbital_period - expected).abs() < 1e-13);
        assert!(bp.is_circular);
    }

    #[test]
    fn create_binary_system_mixed() {
        let coll = create_binary_system(
            V2::new([0.5, 0.0]),
            V2::new([0.0, 1.0]),
            0.5,
            0.1,
            0.04,
            V2::new([-0.5, 0.0]),
            V2::new([0.0, -1.0]),
            0.5,
            0.1,
            0.04,
            10.0,
            0.0, // body 0 accretes; body 1 is inert
            0.2,
            0.0,
            0.0,
            0.0,
        );
        assert_eq!(coll.len(), 2);
        assert!(coll.get(0).has_accretion());
        assert!(!coll.get(1).has_accretion());
        assert!(coll.get(1).has_gravity());
    }
}
