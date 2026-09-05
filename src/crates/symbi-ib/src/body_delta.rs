// =============================================================================
// body_delta.rs
//
// accumulator for per-timestep changes to an immersed body from the fluid.
// two semantics: spatial reduction (operator +=) sums cell contributions
// within one timestep, and temporal update replaces instantaneous quantities
// while accumulating mass.
//
// usage:
//   let mut delta = BodyDelta::new(body_idx);
//   delta += cell_contribution;
//   total.update_for_new_timestep(&timestep_delta);
// =============================================================================

use std::ops::AddAssign;
use symbi_algebra::Tensor;
use symbi_carrier::Scalar;

/// accumulated changes to a body from fluid interaction.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BodyDelta<S: Scalar, const D: usize> {
    pub idx: usize,

    // instantaneous quantities (last timestep value)
    pub force_delta: Tensor<S, D>,
    /// the normal (form-drag / pressure) component of the surface force, projected onto the SDF
    /// outward normal: `force_normal = (F.n_hat) n_hat` summed over the body's cells. the tangential
    /// (skin-friction) part is `force_delta - force_normal_delta`. zero for a bare drain, which
    /// carries no wall normal. instantaneous, like force.
    pub force_normal_delta: Tensor<S, D>,
    pub torque_delta: Tensor<S, 3>,

    // accumulated quantities (summed across timesteps)
    pub mass_delta: S,
    pub prev_mass_delta: S,
    /// total (internal + kinetic) energy absorbed from the fluid -- the accretion-power
    /// budget `Edot = energy_delta / dt`. closes the gas+body energy ledger: the fluid
    /// loses exactly this (uniform-scaling drain), and the body books the matching gain.
    pub energy_delta: S,
    /// the magnetic-slip heat this body's shell released over the step, `sum_c dt qdot_c dV`:
    /// deposited in the gas on an adiabatic closure, exported to the cooling bath on an
    /// isothermal one. accumulated like the mass.
    pub slip_heat_delta: S,
}

impl<S: Scalar, const D: usize> BodyDelta<S, D> {
    pub fn new(idx: usize) -> Self {
        Self {
            idx,
            force_delta: Tensor::zeros(),
            force_normal_delta: Tensor::zeros(),
            torque_delta: Tensor::zeros(),
            mass_delta: S::ZERO,
            prev_mass_delta: S::ZERO,
            energy_delta: S::ZERO,
            slip_heat_delta: S::ZERO,
        }
    }

    /// temporal update: called after each timestep consolidation.
    /// preserves accumulated mass, replaces instantaneous force/torque.
    pub fn update_for_new_timestep(&mut self, timestep_totals: &Self) {
        debug_assert_eq!(self.idx, timestep_totals.idx);
        self.force_delta = timestep_totals.force_delta;
        self.force_normal_delta = timestep_totals.force_normal_delta;
        self.torque_delta = timestep_totals.torque_delta;
        self.mass_delta = self.mass_delta + timestep_totals.mass_delta;
        self.energy_delta = self.energy_delta + timestep_totals.energy_delta;
        self.slip_heat_delta = self.slip_heat_delta + timestep_totals.slip_heat_delta;
    }
}

/// spatial accumulation: sums all cell contributions within one timestep.
impl<S: Scalar, const D: usize> AddAssign for BodyDelta<S, D> {
    fn add_assign(&mut self, rhs: Self) {
        debug_assert_eq!(self.idx, rhs.idx);
        self.force_delta = self.force_delta + rhs.force_delta;
        self.force_normal_delta = self.force_normal_delta + rhs.force_normal_delta;
        self.torque_delta = self.torque_delta + rhs.torque_delta;
        self.mass_delta = self.mass_delta + rhs.mass_delta;
        self.energy_delta = self.energy_delta + rhs.energy_delta;
        self.slip_heat_delta = self.slip_heat_delta + rhs.slip_heat_delta;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    type V2 = Tensor<f64, 2>;
    type V3 = Tensor<f64, 3>;

    #[test]
    fn new_is_zeroed() {
        let d = BodyDelta::<f64, 2>::new(0);
        assert_eq!(d.force_delta, V2::zeros());
        assert_eq!(d.torque_delta, V3::zeros());
        assert_eq!(d.mass_delta, 0.0);
    }

    #[test]
    fn spatial_accumulation() {
        let mut a = BodyDelta::<f64, 2>::new(0);
        a.force_delta = V2::new([1.0, 2.0]);
        a.mass_delta = 0.1;

        let mut b = BodyDelta::<f64, 2>::new(0);
        b.force_delta = V2::new([3.0, 4.0]);
        b.mass_delta = 0.2;

        a += b;
        assert_eq!(a.force_delta, V2::new([4.0, 6.0]));
        assert!((a.mass_delta - 0.3).abs() < 1e-14);
    }

    #[test]
    fn temporal_update_replaces_force_accumulates_mass() {
        let mut total = BodyDelta::<f64, 2>::new(0);
        total.mass_delta = 1.0; // previous accumulated mass
        total.force_delta = V2::new([10.0, 20.0]); // old force

        let mut step = BodyDelta::<f64, 2>::new(0);
        step.force_delta = V2::new([5.0, 6.0]); // new force
        step.mass_delta = 0.3; // new mass gain

        total.update_for_new_timestep(&step);

        // force replaced
        assert_eq!(total.force_delta, V2::new([5.0, 6.0]));
        // mass accumulated
        assert!((total.mass_delta - 1.3).abs() < 1e-14);
    }
}
