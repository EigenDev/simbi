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
use symbi_algebra::{ExactSum, Tensor};
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

/// a step's body delta as exact sums of the cell contributions: every component an exact
/// accumulator, so the partials of any split of the grid (tiles, workers) merge into the whole
/// grid's integers and round once
#[derive(Clone, Debug)]
pub struct ExactBodyDelta<const D: usize> {
    pub idx: usize,
    pub force: [ExactSum; D],
    pub force_normal: [ExactSum; D],
    pub torque: [ExactSum; 3],
    pub mass: ExactSum,
    pub energy: ExactSum,
    pub slip_heat: ExactSum,
}

impl<const D: usize> ExactBodyDelta<D> {
    /// the number of exact accumulators a delta carries
    pub const COMPONENTS: usize = 2 * D + 6;
    /// the words a delta travels as: every accumulator's wire form
    pub const WORDS: usize = Self::COMPONENTS * ExactSum::WORDS;

    pub fn new(idx: usize) -> Self {
        Self {
            idx,
            force: std::array::from_fn(|_| ExactSum::new()),
            force_normal: std::array::from_fn(|_| ExactSum::new()),
            torque: std::array::from_fn(|_| ExactSum::new()),
            mass: ExactSum::new(),
            energy: ExactSum::new(),
            slip_heat: ExactSum::new(),
        }
    }

    fn components(&self) -> impl Iterator<Item = &ExactSum> {
        self.force.iter().chain(&self.force_normal).chain(&self.torque).chain([&self.mass, &self.energy, &self.slip_heat])
    }

    fn components_mut(&mut self) -> impl Iterator<Item = &mut ExactSum> {
        self.force
            .iter_mut()
            .chain(self.force_normal.iter_mut())
            .chain(self.torque.iter_mut())
            .chain([&mut self.mass, &mut self.energy, &mut self.slip_heat])
    }

    /// add every component of an already-rounded delta, exactly
    pub fn add(&mut self, delta: &BodyDelta<f64, D>) {
        debug_assert_eq!(self.idx, delta.idx);
        let values = (0..D)
            .map(|g| delta.force_delta[g])
            .chain((0..D).map(|g| delta.force_normal_delta[g]))
            .chain((0..3).map(|t| delta.torque_delta[t]))
            .chain([delta.mass_delta, delta.energy_delta, delta.slip_heat_delta]);
        for (sum, value) in self.components_mut().zip(values) {
            sum.add(value);
        }
    }

    /// add every component of `other`, exactly
    pub fn merge(&mut self, other: &ExactBodyDelta<D>) {
        debug_assert_eq!(self.idx, other.idx);
        for (sum, theirs) in self.components_mut().zip(other.components()) {
            sum.merge(theirs);
        }
    }

    /// the delta every component rounds to, once
    pub fn round(&self) -> BodyDelta<f64, D> {
        let mut delta = BodyDelta::new(self.idx);
        for g in 0..D {
            delta.force_delta[g] = self.force[g].value();
            delta.force_normal_delta[g] = self.force_normal[g].value();
        }
        for t in 0..3 {
            delta.torque_delta[t] = self.torque[t].value();
        }
        delta.mass_delta = self.mass.value();
        delta.energy_delta = self.energy.value();
        delta.slip_heat_delta = self.slip_heat.value();
        delta
    }

    /// the delta's wire form: every accumulator's words in component order
    pub fn to_words(&self) -> Vec<u64> {
        self.components().flat_map(|sum| sum.to_words()).collect()
    }

    /// read back the delta of body `idx` from its wire form
    pub fn from_words(idx: usize, words: &[u64]) -> Result<Self, String> {
        if words.len() != Self::WORDS {
            return Err(format!("a body delta travels as {} words, not {}", Self::WORDS, words.len()));
        }
        let mut delta = Self::new(idx);
        for (sum, chunk) in delta.components_mut().zip(words.chunks(ExactSum::WORDS)) {
            *sum = ExactSum::from_words(chunk)?;
        }
        Ok(delta)
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

    #[test]
    fn exact_deltas_merge_to_the_same_rounding_in_any_split() {
        // cancelling contributions whose float sum depends on the grouping: the exact merge of any
        // split rounds to the same delta, and the float sum of the split's rounded parts does not
        let cells = [1.0e16, 1.5, -1.0e16, 1.5];
        let delta_of = |values: &[f64]| {
            let mut exact = ExactBodyDelta::<2>::new(0);
            for v in values {
                let mut d = BodyDelta::<f64, 2>::new(0);
                d.force_delta = V2::new([*v, -*v]);
                d.mass_delta = *v;
                exact.add(&d);
            }
            exact
        };
        let whole = delta_of(&cells).round();
        let mut floats = Vec::new();
        for cut in 1..cells.len() {
            let mut merged = delta_of(&cells[..cut]);
            merged.merge(&delta_of(&cells[cut..]));
            let got = merged.round();
            assert_eq!(got.force_delta[0].to_bits(), whole.force_delta[0].to_bits(), "split at {cut}");
            assert_eq!(got.mass_delta.to_bits(), whole.mass_delta.to_bits(), "split at {cut}");
            floats.push(delta_of(&cells[..cut]).round().mass_delta + delta_of(&cells[cut..]).round().mass_delta);
        }
        assert!(floats.iter().any(|f| f.to_bits() != whole.mass_delta.to_bits()), "every float split agrees; the cells exert no pressure");
    }

    #[test]
    fn an_exact_delta_round_trips_its_wire_form() {
        let mut exact = ExactBodyDelta::<3>::new(4);
        let mut d = BodyDelta::<f64, 3>::new(4);
        d.force_delta = V3::new([1.5, -2.25e-300, 7.0e300]);
        d.torque_delta = V3::new([0.1, 0.2, -0.3]);
        d.mass_delta = -1.0e-12;
        d.slip_heat_delta = 3.0;
        exact.add(&d);
        let words = exact.to_words();
        assert_eq!(words.len(), ExactBodyDelta::<3>::WORDS);
        assert!(words.iter().all(|w| *w <= 0xffff_ffff), "a word takes the bits an absent value reserves");
        let back = ExactBodyDelta::<3>::from_words(4, &words).expect("decodes").round();
        assert_eq!(back, exact.round());
    }
}
