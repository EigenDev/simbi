// =============================================================================
// motion.rs
//
// comoving-to-physical coordinate transformation for moving meshes.
// encapsulates the scale factor a(t) and its time derivative a_dot(t)
// for homologous expansion (cosmological-style) or uniform translation.
//
// the mesh stores comoving coordinates r_com (fixed throughout the run).
// physical coordinates are r_phys = a(t) * r_com, computed on-the-fly.
//
// grid velocity:
//   homologous: v_grid = a_dot * r_com  (equivalently H * r_phys)
//   uniform:    v_grid = a_dot
//
// usage:
//   let motion = MotionState::homologous(a, a_dot);
//   let r_phys = motion.to_physical(r_com);
//   let v_grid = motion.grid_velocity(r_com);
// =============================================================================

use symbi_algebra::Tensor;
use symbi_carrier::Scalar;

/// snapshot of the mesh expansion state at a particular time.
/// purely value-based — no mutation, no side effects.
#[derive(Debug, Clone, Copy)]
pub struct MotionState<S: Scalar> {
    /// scale factor a(t). a(0) = 1 typically.
    pub a: S,
    /// time derivative da/dt.
    pub a_dot: S,
    /// homologous (v = H*r) vs uniform translation (v = a_dot).
    pub homologous: bool,
}

impl<S: Scalar> MotionState<S> {
    /// static mesh: a = 1, a_dot = 0. all conversions are identity.
    pub fn static_mesh() -> Self {
        MotionState {
            a: S::ONE,
            a_dot: S::ZERO,
            homologous: false,
        }
    }

    /// homologous expansion with given scale factor and rate.
    pub fn homologous(a: S, a_dot: S) -> Self {
        MotionState {
            a,
            a_dot,
            homologous: true,
        }
    }

    /// uniform translation at velocity a_dot.
    pub fn uniform(a: S, a_dot: S) -> Self {
        MotionState {
            a,
            a_dot,
            homologous: false,
        }
    }

    /// hubble parameter H = a_dot / a.
    pub fn hubble(&self) -> S {
        self.a_dot / self.a
    }

    /// convert a comoving length to physical: L_phys = a * L_com.
    pub fn to_physical(&self, comoving: S) -> S {
        self.a * comoving
    }

    /// convert a physical length to comoving: L_com = L_phys / a.
    pub fn to_comoving(&self, physical: S) -> S {
        physical / self.a
    }

    /// convert a comoving coordinate vector to physical.
    pub fn to_physical_vec<const D: usize>(&self, x: Tensor<S, D>) -> Tensor<S, D> {
        Tensor::new(std::array::from_fn(|ii| self.a * x[ii]))
    }

    /// convert a physical coordinate vector to comoving.
    pub fn to_comoving_vec<const D: usize>(&self, x: Tensor<S, D>) -> Tensor<S, D> {
        Tensor::new(std::array::from_fn(|ii| x[ii] / self.a))
    }

    /// grid velocity at a comoving coordinate.
    ///   homologous: v = a_dot * r_com
    ///   uniform:    v = a_dot
    pub fn grid_velocity(&self, coord_comoving: S) -> S {
        if self.homologous {
            self.a_dot * coord_comoving
        } else {
            self.a_dot
        }
    }

    /// grid velocity vector at a comoving position.
    /// homologous: v_i = a_dot * x_i (radial expansion).
    /// uniform: v = (a_dot, 0, 0, ...) along the first axis.
    pub fn grid_velocity_vec<const D: usize>(&self, x: Tensor<S, D>) -> Tensor<S, D> {
        if self.homologous {
            Tensor::new(std::array::from_fn(|ii| self.a_dot * x[ii]))
        } else {
            let mut v = [S::ZERO; D];
            if D > 0 {
                v[0] = self.a_dot;
            }
            Tensor::new(v)
        }
    }

    /// scale a comoving scale factor array by a(t) to get physical scale factors.
    /// ds_physical = a(t) * h_comoving * dx.
    pub fn physical_scale_factors<const D: usize>(&self, h: Tensor<S, D>) -> Tensor<S, D> {
        Tensor::new(std::array::from_fn(|ii| self.a * h[ii]))
    }
}

// ============================================================
// tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;
    use symbi_algebra::{vec2, vec3};

    fn approx(a: f64, b: f64) -> bool {
        let diff = (a - b).abs();
        if diff < 1e-14 {
            return true;
        }
        diff / a.abs().max(b.abs()) < 1e-12
    }

    fn approx_vec<const D: usize>(a: Tensor<f64, D>, b: Tensor<f64, D>) -> bool {
        (0..D).all(|ii| approx(a[ii], b[ii]))
    }

    // ---- static mesh ----

    #[test]
    fn test_static_mesh_identity() {
        let m = MotionState::<f64>::static_mesh();
        assert!(approx(m.to_physical(5.0), 5.0));
        assert!(approx(m.to_comoving(5.0), 5.0));
        assert!(approx(m.grid_velocity(3.0), 0.0));
    }

    #[test]
    fn test_static_mesh_vec() {
        let m = MotionState::<f64>::static_mesh();
        let x = vec3(1.0, 2.0, 3.0);
        assert!(approx_vec(m.to_physical_vec(x), x));
        assert!(approx_vec(m.to_comoving_vec(x), x));
        assert!(approx_vec(m.grid_velocity_vec(x), Tensor::zeros()));
    }

    // ---- homologous expansion ----

    #[test]
    fn test_homologous_to_physical() {
        let m = MotionState::homologous(2.0, 0.5);
        assert!(approx(m.to_physical(3.0), 6.0));
        assert!(approx(m.to_comoving(6.0), 3.0));
    }

    #[test]
    fn test_homologous_roundtrip() {
        let m = MotionState::homologous(1.5, 0.3);
        let x = 7.0;
        assert!(approx(m.to_comoving(m.to_physical(x)), x));
        assert!(approx(m.to_physical(m.to_comoving(x)), x));
    }

    #[test]
    fn test_homologous_vec_roundtrip() {
        let m = MotionState::homologous(2.0, 0.5);
        let x = vec3(1.0, 2.0, 3.0);
        assert!(approx_vec(m.to_comoving_vec(m.to_physical_vec(x)), x));
    }

    #[test]
    fn test_homologous_grid_velocity() {
        // v = a_dot * r_com
        let m = MotionState::homologous(2.0, 0.5);
        assert!(approx(m.grid_velocity(4.0), 2.0));
        assert!(approx(m.grid_velocity(0.0), 0.0));
    }

    #[test]
    fn test_homologous_grid_velocity_vec() {
        let m = MotionState::homologous(2.0, 0.5);
        let x = vec2(3.0, 4.0);
        assert!(approx_vec(m.grid_velocity_vec(x), vec2(1.5, 2.0)));
    }

    #[test]
    fn test_homologous_hubble() {
        let m = MotionState::homologous(2.0, 0.6);
        assert!(approx(m.hubble(), 0.3));
    }

    // ---- uniform translation ----

    #[test]
    fn test_uniform_grid_velocity() {
        // v = a_dot regardless of position
        let m = MotionState::uniform(1.5, 0.7);
        assert!(approx(m.grid_velocity(0.0), 0.7));
        assert!(approx(m.grid_velocity(100.0), 0.7));
    }

    #[test]
    fn test_uniform_grid_velocity_vec() {
        let m = MotionState::uniform(1.5, 0.7);
        let x = vec3(1.0, 2.0, 3.0);
        let v = m.grid_velocity_vec(x);
        assert!(approx(v[0], 0.7));
        assert!(approx(v[1], 0.0));
        assert!(approx(v[2], 0.0));
    }

    #[test]
    fn test_uniform_to_physical() {
        let m = MotionState::uniform(3.0, 1.0);
        assert!(approx(m.to_physical(2.0), 6.0));
    }

    // ---- physical scale factors ----

    #[test]
    fn test_physical_scale_factors() {
        let m = MotionState::homologous(2.0, 0.5);
        let h = vec3(1.0, 3.0, 5.0);
        let h_phys = m.physical_scale_factors(h);
        assert!(approx(h_phys[0], 2.0));
        assert!(approx(h_phys[1], 6.0));
        assert!(approx(h_phys[2], 10.0));
    }

    #[test]
    fn test_static_scale_factors_unchanged() {
        let m = MotionState::<f64>::static_mesh();
        let h = vec2(3.0, 7.0);
        assert!(approx_vec(m.physical_scale_factors(h), h));
    }

    // ---- edge cases ----

    #[test]
    fn test_a_equals_one() {
        let m = MotionState::homologous(1.0, 0.1);
        assert!(approx(m.to_physical(5.0), 5.0));
        assert!(approx(m.grid_velocity(5.0), 0.5));
    }

    #[test]
    fn test_negative_a_dot_contraction() {
        let m = MotionState::homologous(2.0, -0.5);
        assert!(approx(m.grid_velocity(4.0), -2.0));
        assert!(approx(m.hubble(), -0.25));
    }
}
