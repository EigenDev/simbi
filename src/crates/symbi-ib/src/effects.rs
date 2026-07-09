// =============================================================================
// effects.rs
//
// per-cell source term functions for immersed body-fluid interaction.
// three effects: gravitational acceleration, mass accretion (Bondi-Hoyle),
// and rigid body penalization forcing. each returns a (Cons, BodyDelta) pair.
//
// all functions take explicit geometric data (cell position in cartesian,
// cell volume, etc.) and a Metric reference for coordinate transforms.
// no trait objects, no heap — GPU-portable by design.
//
// usage:
//   let (src, delta) = grav_source(&body, &prim, &metric, &cell, gamma);
// =============================================================================

use symbi_algebra::{Tensor, OrderedNumeric};
use symbi_ir::algebra::Scalar;
use symbi_geometry::{DiagonalMetric, Metric};
use symbi_algebra::{Physical, Embedded};
use symbi_hydro::state::{Prim, Cons};

use crate::body::Body;
use crate::body_delta::BodyDelta;

/// pre-computed cell geometry passed to effect functions.
#[derive(Clone, Copy, Debug)]
pub struct CellGeometry<S: Scalar, const D: usize> {
    /// cell center position in coordinate basis.
    pub position: Tensor<S, D>,
    /// cell volume (including metric determinant).
    pub volume: S,
    /// minimum cell width (from scale factors).
    pub min_width: S,
}

/// gravitational source terms. softened newtonian gravity (G=1).
///
/// returns: (conservative source, body delta from Newton's third law).
pub fn grav_source<S: Scalar, const D: usize>(
    body: &Body<S, D>,
    prim: &Prim<S, D>,
    metric: &(impl Metric<S, D> + DiagonalMetric<S, D>),
    cell: &CellGeometry<S, D>,
    _gamma: S,
) -> (Cons<S, D>, BodyDelta<S, D>) {
    let cell_cart = metric.to_cartesian(cell.position);
    let r_vec = cell_cart - body.position;
    let r_mag = r_vec.norm();

    let softening = body.softening().expect("grav_source requires a gravitational body");
    let r_eff = (r_mag * r_mag + softening * softening).sqrt();

    // gravitational acceleration in cartesian (G = 1)
    let inv_r3 = S::ONE / (r_eff * r_eff * r_eff);
    let g_cart = r_vec.scale(-body.mass * inv_r3);

    // convert to the PHYSICAL (orthonormal) frame the momentum lives in (Cart -> Ortho).
    // vector_from_cartesian yields the orthonormal frame, not the coordinate basis; the frame
    // types enforce this. `.into_raw()` unwraps at the cons.mom boundary (a Tensor until the
    // conserved fields are retyped as `Physical`).
    let g_phys = metric.vector_from_cartesian(cell.position, Embedded::new(g_cart));

    let density = prim.rho;
    let dp_dt = g_phys.into_raw().scale(density);

    // energy: v . F in cartesian. prim.vel is the physical (orthonormal) velocity -> lab frame.
    let vel_cart = metric.vector_to_cartesian(cell.position, Physical::new(prim.vel)).into_raw();
    let force_cart = g_cart.scale(density);
    let de_dt = vel_cart.dot(&force_cart);

    let source = Cons { den: S::ZERO, mom: dp_dt, nrg: de_dt };
    let delta = BodyDelta {
        idx: body.idx,
        force_delta: force_cart.scale(-cell.volume),
        torque_delta: Tensor::zeros(),
        mass_delta: S::ZERO,
        prev_mass_delta: S::ZERO,
        energy_delta: S::ZERO, // gravity is a force exchange, not absorbed accretion energy
    };
    (source, delta)
}

/// rigid body penalization forcing.
///
/// applies a volume-penalization method: strong forcing inside the body,
/// cubic falloff in the boundary layer, and a pre-emptive zone for
/// supersonic flows.
///
/// returns: (conservative source, body delta).
pub fn rigid_source<S: Scalar + OrderedNumeric, const D: usize>(
    body: &Body<S, D>,
    prim: &Prim<S, D>,
    metric: &(impl Metric<S, D> + DiagonalMetric<S, D>),
    cell: &CellGeometry<S, D>,
    gamma: S,
) -> (Cons<S, D>, BodyDelta<S, D>) {
    let zero_result = || (Cons::zero(), BodyDelta::new(body.idx));

    let cell_cart = metric.to_cartesian(cell.position);
    let r_vec = cell_cart - body.position;
    let distance = r_vec.norm();

    let safe_min = S::from_f64(1e-10);
    let r_norm = distance.max(safe_min);
    let r_hat = r_vec.scale(S::ONE / r_norm);
    let signed_distance = distance - body.radius;

    let cs = sound_speed(prim, gamma);
    let fluid_speed = prim.vel.norm();
    let mach = fluid_speed / cs.max(safe_min);

    let boundary_thickness = if mach > S::ONE {
        S::from_f64(0.5) * cell.min_width
    } else {
        cell.min_width
    };

    let extended_radius = body.radius + if mach > S::ONE {
        S::from_f64(2.0) * boundary_thickness
    } else {
        boundary_thickness
    };

    if distance > extended_radius + boundary_thickness {
        return zero_result();
    }

    let density = prim.rho;
    let rel_velocity = prim.vel - body.velocity;
    let normal_rel = r_hat.scale(rel_velocity.dot(&r_hat));

    let cs2 = cs * cs;
    let base_strength = if mach > S::ONE {
        S::from_f64(25.0) * density * cs2
    } else {
        S::from_f64(10.0) * density * cs2
    };

    let dp_dt;
    if signed_distance < S::ZERO {
        // inside body — strong forcing
        let depth_ratio = (-signed_distance) / body.radius;
        let interior_factor = S::ONE + S::from_f64(10.0) * depth_ratio * depth_ratio;
        dp_dt = rel_velocity.scale(-base_strength * interior_factor);
    } else if signed_distance < boundary_thickness {
        // boundary layer — cubic falloff
        let boundary_factor = S::ONE - signed_distance / boundary_thickness;
        let sharp_factor = boundary_factor * boundary_factor * boundary_factor;
        dp_dt = rel_velocity.scale(-base_strength * sharp_factor);
    } else if mach > S::ONE && signed_distance < S::from_f64(2.0) * boundary_thickness {
        // pre-emptive supersonic zone
        let pre_factor = S::ONE - (signed_distance - boundary_thickness) / boundary_thickness;
        let pre_strength = S::from_f64(0.5) * base_strength * pre_factor * pre_factor;

        let incoming = (-rel_velocity.dot(&r_hat)).max(S::ZERO);
        if incoming > S::from_f64(0.1) * cs {
            dp_dt = normal_rel.scale(-pre_strength);
        } else {
            return zero_result();
        }
    } else {
        return zero_result();
    }

    let de_dt = prim.vel.dot(&dp_dt);
    let dv = cell.volume;
    let r_3d = to_3d(r_vec);
    let dp_3d = to_3d(dp_dt);
    let torque = r_3d.cross(&dp_3d).scale(dv);

    let source = Cons { den: S::ZERO, mom: dp_dt, nrg: de_dt };
    let delta_body = BodyDelta {
        idx: body.idx,
        force_delta: dp_dt.scale(-S::ONE),
        torque_delta: torque,
        mass_delta: S::ZERO,
        prev_mass_delta: S::ZERO,
        energy_delta: S::ZERO, // rigid penalty is a momentum exchange; legacy path books no energy
    };
    (source, delta_body)
}

// -- helpers --

fn sound_speed<S: Scalar, const D: usize>(prim: &Prim<S, D>, gamma: S) -> S {
    (gamma * prim.pre / prim.rho).sqrt()
}

/// embed a D-dimensional vector into 3D (zero-pad).
fn to_3d<S: Scalar, const D: usize>(v: Tensor<S, D>) -> Tensor<S, 3> {
    let mut out = [S::ZERO; 3];
    for dd in 0..D.min(3) {
        out[dd] = v[dd];
    }
    Tensor::new(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use symbi_geometry::Cartesian;

    fn approx(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-10 * a.abs().max(b.abs()).max(1.0)
    }

    fn cell_at(x: f64, y: f64) -> CellGeometry<f64, 2> {
        CellGeometry {
            position: Tensor::new([x, y]),
            volume: 0.01,
            min_width: 0.1,
        }
    }

    #[test]
    fn grav_source_symmetry() {
        let body = Body::gravitational(
            0, Tensor::new([0.0, 0.0]), Tensor::zeros(), 1.0, 0.1, 0.04,
        );
        let prim = Prim { rho: 1.0, vel: Tensor::zeros(), pre: 1.0 };
        let metric = Cartesian;

        let (src_l, _) = grav_source(&body, &prim, &metric, &cell_at(-1.0, 0.0), 1.4);
        let (src_r, _) = grav_source(&body, &prim, &metric, &cell_at( 1.0, 0.0), 1.4);

        // momentum should be equal magnitude, opposite sign
        assert!(approx(src_l.mom[0], -src_r.mom[0]));
        assert!(approx(src_l.mom[1], src_r.mom[1])); // both zero
    }

    #[test]
    fn grav_source_attractive() {
        let body = Body::gravitational(
            0, Tensor::new([0.0, 0.0]), Tensor::zeros(), 1.0, 0.1, 0.04,
        );
        let prim = Prim { rho: 1.0, vel: Tensor::zeros(), pre: 1.0 };
        let metric = Cartesian;

        let (src, _) = grav_source(&body, &prim, &metric, &cell_at(1.0, 0.0), 1.4);
        // x-momentum should be negative (attracted toward body at origin)
        assert!(src.mom[0] < 0.0);
    }

    #[test]
    fn grav_source_newton_third_law() {
        let body = Body::gravitational(
            0, Tensor::new([0.0, 0.0]), Tensor::zeros(), 1.0, 0.1, 0.04,
        );
        let prim = Prim { rho: 1.0, vel: Tensor::zeros(), pre: 1.0 };
        let metric = Cartesian;
        let cell = cell_at(1.0, 0.0);

        let (src, delta) = grav_source(&body, &prim, &metric, &cell, 1.4);
        // body delta force should oppose fluid source * volume
        let fluid_force_x = src.mom[0] * cell.volume;
        assert!(approx(delta.force_delta[0], -fluid_force_x));
    }

    #[test]
    fn grav_zero_energy_at_rest() {
        let body = Body::gravitational(
            0, Tensor::new([0.0, 0.0]), Tensor::zeros(), 1.0, 0.1, 0.04,
        );
        let prim = Prim { rho: 1.0, vel: Tensor::zeros(), pre: 1.0 };
        let metric = Cartesian;

        let (src, _) = grav_source(&body, &prim, &metric, &cell_at(1.0, 0.0), 1.4);
        // v = 0 => dE/dt = v . F = 0
        assert!(approx(src.nrg, 0.0));
    }

    #[test]
    fn rigid_outside_body_is_zero() {
        let body = Body::rigid_sphere(
            0, Tensor::new([0.0, 0.0]), Tensor::zeros(), 1.0, 0.5, 0.3, false,
        );
        let prim = Prim { rho: 1.0, vel: Tensor::new([0.1, 0.0]), pre: 1.0 };
        let metric = Cartesian;

        let (src, _) = rigid_source(&body, &prim, &metric, &cell_at(5.0, 0.0), 1.4);
        assert!(approx(src.mom[0], 0.0));
        assert!(approx(src.mom[1], 0.0));
    }

    #[test]
    fn rigid_inside_body_opposes_motion() {
        let body = Body::rigid_sphere(
            0, Tensor::new([0.0, 0.0]), Tensor::zeros(), 1.0, 1.0, 0.3, false,
        );
        // fluid moving in +x direction
        let prim = Prim { rho: 1.0, vel: Tensor::new([1.0, 0.0]), pre: 1.0 };
        let metric = Cartesian;

        let (src, _) = rigid_source(
            &body, &prim, &metric,
            &CellGeometry { position: Tensor::new([0.1, 0.0]), volume: 0.01, min_width: 0.1 },
            1.4,
        );
        // forcing should oppose fluid velocity (negative x-momentum)
        assert!(src.mom[0] < 0.0);
    }

    #[test]
    fn sound_speed_ideal_gas() {
        let prim = Prim::<f64, 2> { rho: 1.0, vel: Tensor::zeros(), pre: 1.0 };
        let cs = super::sound_speed(&prim, 1.4);
        assert!(approx(cs, (1.4_f64).sqrt()));
    }

    #[test]
    fn to_3d_padding() {
        let v2: Tensor<f64, 2> = Tensor::new([1.0, 2.0]);
        let v3 = to_3d(v2);
        assert_eq!(v3[0], 1.0);
        assert_eq!(v3[1], 2.0);
        assert_eq!(v3[2], 0.0);
    }
}
