// =============================================================================
// motion.rs
//
// keplerian orbital evolution for binary systems.
// provides pure functions for rotating body positions and velocities
// by the orbital angle omega * dt.
//
// usage:
//   let advanced = advance_binary(&collection, dt);
//   collection.snapshot(&advanced);
// =============================================================================

use crate::body::{Body, BodyKind};
use crate::body_delta::BodyDelta;
use crate::collection::{BodyCollection, ReferenceFrame};
use symbi_algebra::Tensor;
use symbi_ir::algebra::Scalar;

/// rotate a 2D vector by angle theta (radians) in the xy-plane.
pub fn rotate_2d<S: Scalar>(v: Tensor<S, 2>, theta: S) -> Tensor<S, 2> {
    let c = theta.cos();
    let s = theta.sin();
    Tensor::new([v[0] * c - v[1] * s, v[0] * s + v[1] * c])
}

/// rotate a 3D vector by angle theta (radians) about the z-axis.
pub fn rotate_3d<S: Scalar>(v: Tensor<S, 3>, theta: S) -> Tensor<S, 3> {
    let c = theta.cos();
    let s = theta.sin();
    Tensor::new([v[0] * c - v[1] * s, v[0] * s + v[1] * c, v[2]])
}

/// compute advanced body positions for an inertial binary system.
///
/// rotates all bodies by omega * dt where omega = sqrt(M / a^3).
/// returns Some only for a 2D or 3D binary in an inertial frame; a corotating or
/// stationary frame, a 1D system, or any other body count yields None.
pub fn advance_binary<S: Scalar, const D: usize>(
    coll: &BodyCollection<S, D>,
    dt: S,
) -> Option<Vec<Body<S, D>>> {
    if D < 2 {
        return None;
    }
    if !coll.is_binary() {
        return None;
    }
    if coll.frame != ReferenceFrame::Inertial {
        return None;
    }

    let bp = coll.binary_params.as_ref()?;
    let omega = (bp.total_mass / (bp.semi_major * bp.semi_major * bp.semi_major)).sqrt();
    let d_theta = omega * dt;

    let mut advanced = Vec::with_capacity(coll.len());
    for body in coll.bodies() {
        let new_pos = rotate_nd(body.position, d_theta);
        let new_vel = rotate_nd(body.velocity, d_theta);
        advanced.push(body.at_position(new_pos).with_velocity(new_vel));
    }
    Some(advanced)
}

/// apply per-body diagnostic deltas (force / torque / would-be accreted mass) to a body collection,
/// then advance the prescribed binary orbit. shared by the single-grid `evolve_bodies` (which passes
/// its own consolidated per-step deltas) and the decomposed body step (which passes the cross-tile
/// sum of every tile's partials). the gravitating mass is held fixed (a fixed-potential sink):
/// force/torque are recorded for output, total_accreted_mass accumulates the would-be accretion, and
/// the binary motion is prescribed (Keplerian, independent of the feedback). applying identical deltas
/// to two identical collections yields identical state -- the property the decomposed loop relies on
/// to keep every tile's bodies in lockstep (same global delta + same prescribed advance on each tile).
/// the source prefix is what integrates here: fragments (bodies beyond
/// `source_count()`) get their ledger fields booked but their motion belongs to
/// the bonded-fragment subcycle, which consumes the same deltas as frozen
/// external loads.
pub fn apply_body_deltas<const D: usize>(
    bodies: &mut BodyCollection<f64, D>,
    deltas: &[BodyDelta<f64, D>],
    dt: f64,
) {
    let n_src = bodies.source_count();
    for delta in deltas {
        if delta.idx < bodies.len() {
            let body = bodies.get_mut(delta.idx);
            body.force = delta.force_delta;
            body.torque = delta.torque_delta;
            // translational reaction: force_delta is the drag force the gas exerts on the body
            // (momentum-transfer rate = absorbed_momentum * volume / dt). the body gains the impulse
            // over the step divided by its mass: dv = force_delta * dt / mass. gated on two_way +
            // mass > 0. the rotational reaction (torque_delta, likewise a torque = rate) is integrated
            // over dt in the orientation advance below via Euler's equations.
            if delta.idx < n_src && body.two_way_coupling && body.mass > 0.0 {
                let m = body.mass;
                for a in 0..D {
                    body.velocity[a] = body.velocity[a] + delta.force_delta[a] * dt / m;
                }
            }
            if let BodyKind::BlackHole {
                total_accreted_mass,
                accretion_rate,
                ..
            } = &mut body.kind
            {
                *total_accreted_mass += delta.mass_delta;
                *accretion_rate = if dt > 0.0 { delta.mass_delta / dt } else { 0.0 };
            }
            // the GR horizon books the accreted rest mass together with the covariant energy the shell-flux
            // reduction measured this step (the energy is exactly conserved, so edot is well-defined).
            if let BodyKind::Horizon {
                total_accreted_mass,
                total_accreted_energy,
                mdot,
                edot,
                ..
            } = &mut body.kind
            {
                *total_accreted_mass += delta.mass_delta;
                *total_accreted_energy += delta.energy_delta;
                *mdot = if dt > 0.0 { delta.mass_delta / dt } else { 0.0 };
                *edot = if dt > 0.0 {
                    delta.energy_delta / dt
                } else {
                    0.0
                };
            }
        }
    }
    // a prescribed binary orbit (fixed-potential) governs position/velocity when present and takes
    // precedence over the force-driven update below.
    let binary = advance_binary(bodies, dt);
    if let Some(advanced) = &binary {
        for ii in 0..advanced.len().min(bodies.len()) {
            bodies.get_mut(ii).position = advanced[ii].position;
            bodies.get_mut(ii).velocity = advanced[ii].velocity;
        }
    }
    for ii in 0..n_src {
        let body = bodies.get_mut(ii);
        // force-driven translation: a two-way body drifts under the velocity the gas reaction gave
        // it (position += velocity * dt). a prescribed binary orbit takes precedence where present.
        if binary.is_none() && body.two_way_coupling {
            for a in 0..D {
                body.position[a] = body.position[a] + body.velocity[a] * dt;
            }
        }
        // advance the full rigid-body rotation (Euler's equations + orientation roll). a two-way
        // body feels the reaction torque (world-frame per-step angular momentum on `body.torque`); a
        // prescribed body evolves torque-free, still precessing if its inertia is anisotropic.
        let torque = if body.two_way_coupling {
            body.torque
        } else {
            Tensor::zeros()
        };
        body.advance_rotation(torque, dt);
    }
}

/// rotate an N-dimensional vector by theta in the xy-plane.
/// D=1: identity. D=2: rotate_2d. D>=3: rotate about z-axis.
fn rotate_nd<S: Scalar, const D: usize>(v: Tensor<S, D>, theta: S) -> Tensor<S, D> {
    if D < 2 {
        return v;
    }

    let c = theta.cos();
    let s = theta.sin();
    let mut out = v;
    out[0] = v[0] * c - v[1] * s;
    out[1] = v[0] * s + v[1] * c;
    out
}

/// compute keplerian initial conditions for a binary system.
///
/// given total mass M, semi-major axis a, and mass ratio q = m2/m1:
///   m1 = M / (1 + q), m2 = M - m1
///   a1 = a / (1 + q), a2 = a - a1  (center of mass)
///   v1 = omega * a2,  v2 = -omega * a1
///
/// returns (pos1, vel1, mass1, pos2, vel2, mass2).
pub fn keplerian_binary<S: Scalar>(
    total_mass: S,
    semi_major: S,
    mass_ratio: S,
) -> (Tensor<S, 2>, Tensor<S, 2>, S, Tensor<S, 2>, Tensor<S, 2>, S) {
    let one = S::ONE;
    let one_plus_q = one + mass_ratio;

    let m1 = total_mass / one_plus_q;
    let m2 = total_mass - m1;

    // body 1 (heavier for q<1) is closer to com
    let a1 = semi_major * mass_ratio / one_plus_q;
    let a2 = semi_major - a1;

    let a3 = semi_major * semi_major * semi_major;
    let omega = (total_mass / a3).sqrt();

    let v1 = omega * a2;
    let v2 = -omega * a1;

    (
        Tensor::new([a1, S::ZERO]),
        Tensor::new([S::ZERO, v1]),
        m1,
        Tensor::new([-a2, S::ZERO]),
        Tensor::new([S::ZERO, v2]),
        m2,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::collection::BinaryParams;

    fn approx(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-10 * a.abs().max(b.abs()).max(1.0)
    }

    #[test]
    fn rotate_2d_quarter_turn() {
        let v = Tensor::new([1.0_f64, 0.0]);
        let r = rotate_2d(v, std::f64::consts::FRAC_PI_2);
        assert!(approx(r[0], 0.0));
        assert!(approx(r[1], 1.0));
    }

    #[test]
    fn rotate_2d_full_turn() {
        let v = Tensor::new([3.0_f64, 4.0]);
        let r = rotate_2d(v, 2.0 * std::f64::consts::PI);
        assert!(approx(r[0], 3.0));
        assert!(approx(r[1], 4.0));
    }

    #[test]
    fn advance_spin_integrates_the_orientation() {
        // spin about z at rate 1, advanced 90 deg: R becomes R_z(pi/2) = [[0,-1,0],[1,0,0],[0,0,1]].
        let mut b = crate::Body::<f64, 3>::rigid_sphere(
            0,
            Tensor::zeros(),
            Tensor::zeros(),
            1.0,
            0.1,
            1.0,
            true,
        )
        .with_angular_velocity(Tensor::new([0.0, 0.0, 1.0]));
        // isotropic + torque-free: omega is constant, so the orientation rolls by omega*dt about z.
        b.advance_rotation(Tensor::zeros(), std::f64::consts::FRAC_PI_2);
        let r = b.orientation;
        assert!(approx(r[0][0], 0.0) && approx(r[0][1], -1.0) && approx(r[0][2], 0.0));
        assert!(approx(r[1][0], 1.0) && approx(r[1][1], 0.0) && approx(r[1][2], 0.0));
        assert!(approx(r[2][0], 0.0) && approx(r[2][1], 0.0) && approx(r[2][2], 1.0));
    }

    #[test]
    fn two_way_off_axis_torque_tilts_omega_free_tumbling() {
        // a body spinning about z hit by a reaction torque about x: free 3D rotation integrates the
        // full torque vector, so omega tilts off z (gains an x component). integrating all three
        // components is what lets an asymmetric body tumble; a fixed-axis rotor holds omega on z.
        let mut coll = BodyCollection::new().add(
            crate::Body::<f64, 3>::rigid_sphere(
                0,
                Tensor::zeros(),
                Tensor::zeros(),
                1.0,
                0.1,
                1.0,
                true,
            )
            .with_angular_velocity(Tensor::new([0.0, 0.0, 5.0]))
            .with_two_way_coupling(true),
        );
        let mut delta = crate::BodyDelta::<f64, 3>::new(0);
        delta.torque_delta = Tensor::new([2.0, 0.0, 0.0]); // torque about x; I = 1, dt = 1e-3
        apply_body_deltas(&mut coll, &[delta], 1e-3);
        let w = coll.get(0).omega;
        // Euler integrates I domega = torque dt, so domega_x = 2 * 1e-3 = 2e-3; z spin untouched.
        assert!(
            approx(w[0], 2e-3),
            "omega should tilt toward x (free tumbling): {w:?}"
        );
        assert!(
            approx(w[2], 5.0),
            "the z spin is unchanged by the x torque: {w:?}"
        );
    }

    #[test]
    fn two_way_force_accelerates_by_the_impulse_over_mass() {
        // force_delta is the drag force; the body gains the impulse over the step, dv = F dt / mass.
        // a large force over a tiny step is a small velocity kick.
        let mut coll = BodyCollection::new().add(
            crate::Body::<f64, 3>::rigid_sphere(
                0,
                Tensor::zeros(),
                Tensor::zeros(),
                4.0,
                0.1,
                1.0,
                true,
            )
            .with_two_way_coupling(true),
        );
        let mut delta = crate::BodyDelta::<f64, 3>::new(0);
        delta.force_delta = Tensor::new([8.0, 0.0, 0.0]); // F = 8, mass = 4, dt = 1e-3
        apply_body_deltas(&mut coll, &[delta], 1e-3);
        // dv = F dt / m = 8 * 1e-3 / 4 = 2e-3.
        assert!(
            approx(coll.get(0).velocity[0], 2e-3),
            "{:?}",
            coll.get(0).velocity
        );
    }

    #[test]
    fn anisotropic_torque_free_body_precesses() {
        // torque-free motion of an anisotropic body spinning about a non-principal axis: Euler's
        // gyroscopic term `omega x (I omega)` makes omega precess — it develops a z component absent
        // from the initial state. isotropic (equal) moments zero the gyroscopic term, holding omega
        // fixed.
        let mut aniso = crate::Body::<f64, 3>::rigid_sphere(
            0,
            Tensor::zeros(),
            Tensor::zeros(),
            1.0,
            0.1,
            1.0,
            true,
        )
        .with_inertia_principal([1.0, 2.0, 3.0])
        .with_angular_velocity(Tensor::new([1.0, 1.0, 0.0]));
        aniso.advance_rotation(Tensor::zeros(), 0.05); // torque-free
        assert!(
            aniso.omega[2].abs() > 1e-6,
            "an anisotropic body should precess (omega gains a z component): {:?}",
            aniso.omega,
        );

        let mut iso = crate::Body::<f64, 3>::rigid_sphere(
            0,
            Tensor::zeros(),
            Tensor::zeros(),
            1.0,
            0.1,
            1.0,
            true,
        )
        .with_angular_velocity(Tensor::new([1.0, 1.0, 0.0])); // isotropic default (1,1,1)
        iso.advance_rotation(Tensor::zeros(), 0.05);
        assert!(
            iso.omega[2].abs() < 1e-15,
            "an isotropic body must not precess (zero gyroscopic term): {:?}",
            iso.omega,
        );
    }

    #[test]
    fn rotate_3d_preserves_z() {
        let v = Tensor::new([1.0_f64, 0.0, 5.0]);
        let r = rotate_3d(v, std::f64::consts::FRAC_PI_2);
        assert!(approx(r[2], 5.0));
    }

    #[test]
    fn keplerian_binary_equal_mass() {
        let (p1, v1, m1, p2, v2, m2) = keplerian_binary(2.0_f64, 1.0, 1.0);
        assert!(approx(m1, 1.0));
        assert!(approx(m2, 1.0));
        // equal mass -> equal separation from center
        assert!(approx(p1[0], 0.5));
        assert!(approx(p2[0], -0.5));
        // velocities opposite
        assert!(approx(v1[1], -v2[1]));
    }

    #[test]
    fn keplerian_binary_center_of_mass() {
        let (p1, _, m1, p2, _, m2) = keplerian_binary(3.0_f64, 1.0, 0.5);
        // m1*x1 + m2*x2 = 0
        let com = m1 * p1[0] + m2 * p2[0];
        assert!(approx(com, 0.0));
    }

    #[test]
    fn advance_binary_returns_none_for_non_binary() {
        let coll = BodyCollection::<f64, 2>::new().add(Body::passive(
            0,
            Tensor::zeros(),
            Tensor::zeros(),
            1.0,
            0.1,
        ));
        assert!(advance_binary(&coll, 0.01).is_none());
    }

    #[test]
    fn advance_binary_returns_none_for_corotating() {
        let coll = BodyCollection::<f64, 2>::new()
            .add(Body::gravitational(
                0,
                Tensor::new([0.5, 0.0]),
                Tensor::zeros(),
                1.0,
                0.1,
                0.04,
            ))
            .add(Body::gravitational(
                1,
                Tensor::new([-0.5, 0.0]),
                Tensor::zeros(),
                1.0,
                0.1,
                0.04,
            ))
            .with_name("binary_system")
            .as_binary()
            .with_frame(ReferenceFrame::Corotating)
            .with_binary_params(BinaryParams::new(2.0, 1.0, 0.0, 1.0));
        assert!(advance_binary(&coll, 0.01).is_none());
    }

    #[test]
    fn advance_binary_rotates() {
        let coll = BodyCollection::new()
            .add(Body::gravitational(
                0,
                Tensor::new([0.5, 0.0]),
                Tensor::new([0.0, 1.0]),
                1.0,
                0.1,
                0.04,
            ))
            .add(Body::gravitational(
                1,
                Tensor::new([-0.5, 0.0]),
                Tensor::new([0.0, -1.0]),
                1.0,
                0.1,
                0.04,
            ))
            .with_name("binary_system")
            .as_binary()
            .with_frame(ReferenceFrame::Inertial)
            .with_binary_params(BinaryParams::new(2.0, 1.0, 0.0, 1.0));

        let advanced = advance_binary(&coll, 0.01).unwrap();
        assert_eq!(advanced.len(), 2);

        // positions should have rotated slightly
        let r0 = advanced[0].position.norm();
        assert!(approx(r0, 0.5)); // radius preserved
        assert!(advanced[0].position[1] > 0.0); // rotated counterclockwise
    }
}
