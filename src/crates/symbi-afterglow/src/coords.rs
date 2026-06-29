// =============================================================================
// coords.rs
//
// coordinate-system-agnostic transforms: map a hydro cell's coordinate position and
// PHYSICAL (orthonormal-frame) three-velocity into the global Cartesian frame the
// afterglow geometry (EATS arrival time, beaming toward an observer) works in.
//
// this matches the canonical transforms in `symbi_geometry::metric` — same coordinate
// conventions, same orthonormal-frame velocity rotation — so the afterglow ingests
// exactly what the hydro produces, for any geometry it runs (Cartesian / Spherical /
// Cylindrical). conventions (must match the hydro):
//   Cartesian   : (x, y, z),      v = (vx, vy, vz)
//   Spherical   : (r, theta, phi), v = (v_r, v_theta, v_phi)   [theta from +z, phi about +z]
//   Cylindrical : (r, phi, z),     v = (v_r, v_phi, v_z)
// the stored v1,v2,v3 are PHYSICAL components in units of c (the hydro's Lorentz factor
// is W = 1/sqrt(1 - v.v) with a plain dot product, i.e., v is the orthonormal velocity).
//
// usage:
//  let pos  = Coords::Spherical.position_to_cartesian([r, theta, phi]);
//  let beta = Coords::Spherical.velocity_to_cartesian([r, theta, phi], [vr, vt, vp]);
// =============================================================================

/// the coordinate system a hydro snapshot is stored in (matches `symbi_geometry::Geometry`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Coords {
    Cartesian,
    Spherical,
    Cylindrical,
}

impl Coords {
    /// map a coordinate position `(x1, x2, x3)` to global Cartesian `(x, y, z)`.
    pub fn position_to_cartesian(self, x: [f64; 3]) -> [f64; 3] {
        match self {
            Coords::Cartesian => x,
            Coords::Spherical => {
                let (r, theta, phi) = (x[0], x[1], x[2]);
                let st = theta.sin();
                [r * st * phi.cos(), r * st * phi.sin(), r * theta.cos()]
            }
            Coords::Cylindrical => {
                let (r, phi, z) = (x[0], x[1], x[2]);
                [r * phi.cos(), r * phi.sin(), z]
            }
        }
    }

    /// rotate a PHYSICAL (orthonormal-frame) three-velocity `v = (v1, v2, v3)` at coordinate
    /// position `x` into the global Cartesian frame. this is the orthonormal-basis rotation —
    /// identical to `symbi_geometry::Metric::vector_to_cartesian` for a `Physical` vector.
    pub fn velocity_to_cartesian(self, x: [f64; 3], v: [f64; 3]) -> [f64; 3] {
        match self {
            Coords::Cartesian => v,
            Coords::Spherical => {
                let (theta, phi) = (x[1], x[2]);
                let (st, ct) = (theta.sin(), theta.cos());
                let (sp, cp) = (phi.sin(), phi.cos());
                // v_r in r-hat, v_theta in theta-hat, v_phi in phi-hat, rotated to lab axes.
                [
                    v[0] * st * cp + v[1] * ct * cp - v[2] * sp,
                    v[0] * st * sp + v[1] * ct * sp + v[2] * cp,
                    v[0] * ct - v[1] * st,
                ]
            }
            Coords::Cylindrical => {
                let phi = x[1];
                let (sp, cp) = (phi.sin(), phi.cos());
                // v_r in r-hat, v_phi in phi-hat, v_z in z-hat.
                [v[0] * cp - v[1] * sp, v[0] * sp + v[1] * cp, v[2]]
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::{FRAC_PI_2, PI};

    fn close(a: [f64; 3], b: [f64; 3]) -> bool {
        (0..3).all(|i| (a[i] - b[i]).abs() < 1e-12)
    }

    // spherical position: known axes (matches symbi_geometry::metric tests).
    #[test]
    fn spherical_position_known_points() {
        // r=5, theta=0 (north pole) -> +z.
        assert!(close(
            Coords::Spherical.position_to_cartesian([5.0, 0.0, 0.0]),
            [0.0, 0.0, 5.0]
        ));
        // r=3, theta=pi/2, phi=0 -> +x.
        assert!(close(
            Coords::Spherical.position_to_cartesian([3.0, FRAC_PI_2, 0.0]),
            [3.0, 0.0, 0.0]
        ));
    }

    // cylindrical position: r=5, phi=pi/2, z=3 -> (0, 5, 3).
    #[test]
    fn cylindrical_position_known_point() {
        assert!(close(
            Coords::Cylindrical.position_to_cartesian([5.0, FRAC_PI_2, 3.0]),
            [0.0, 5.0, 3.0]
        ));
    }

    // a purely radial spherical velocity points along the position direction (r-hat).
    #[test]
    fn spherical_radial_velocity_is_along_rhat() {
        let x = [2.0, PI / 3.0, PI / 5.0];
        let pos = Coords::Spherical.position_to_cartesian(x);
        let rmag = (pos[0] * pos[0] + pos[1] * pos[1] + pos[2] * pos[2]).sqrt();
        let rhat = [pos[0] / rmag, pos[1] / rmag, pos[2] / rmag];
        let v = Coords::Spherical.velocity_to_cartesian(x, [0.7, 0.0, 0.0]); // v_r only
        assert!(close(v, [0.7 * rhat[0], 0.7 * rhat[1], 0.7 * rhat[2]]));
    }

    // a polar (theta-hat) velocity is perpendicular to r-hat — the spreading direction.
    #[test]
    fn spherical_polar_velocity_is_transverse() {
        let x = [2.0, PI / 3.0, PI / 5.0];
        let pos = Coords::Spherical.position_to_cartesian(x);
        let vlat = Coords::Spherical.velocity_to_cartesian(x, [0.0, 0.5, 0.0]); // v_theta only
        let dot = pos[0] * vlat[0] + pos[1] * vlat[1] + pos[2] * vlat[2];
        assert!(
            dot.abs() < 1e-12,
            "theta-hat must be perpendicular to r-hat, dot={dot}"
        );
        let mag = (vlat[0] * vlat[0] + vlat[1] * vlat[1] + vlat[2] * vlat[2]).sqrt();
        assert!(
            (mag - 0.5).abs() < 1e-12,
            "orthonormal rotation preserves magnitude"
        );
    }

    // cylindrical azimuthal (phi-hat) velocity is transverse and norm-preserving.
    #[test]
    fn cylindrical_azimuthal_velocity_is_transverse() {
        let x = [3.0, PI / 4.0, 1.0];
        let pos = Coords::Cylindrical.position_to_cartesian(x);
        let vphi = Coords::Cylindrical.velocity_to_cartesian(x, [0.0, 0.6, 0.0]);
        // phi-hat is perpendicular to the cylindrical radial direction (x, y) but z stays 0.
        let dot_xy = pos[0] * vphi[0] + pos[1] * vphi[1];
        assert!(
            dot_xy.abs() < 1e-12,
            "phi-hat perpendicular to cyl-radius, dot={dot_xy}"
        );
        assert!(vphi[2].abs() < 1e-12);
    }

    // cartesian is the identity for both position and velocity.
    #[test]
    fn cartesian_is_identity() {
        let x = [1.0, 2.0, 3.0];
        let v = [0.1, -0.2, 0.3];
        assert!(close(Coords::Cartesian.position_to_cartesian(x), x));
        assert!(close(Coords::Cartesian.velocity_to_cartesian(x, v), v));
    }
}
