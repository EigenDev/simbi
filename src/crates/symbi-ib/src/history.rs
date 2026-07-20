// =============================================================================
// history.rs
//
// the per-step body diagnostic time series (docs/ideas/accretor.md §5): what
// each body exchanged with the gas on every step — the record the steady-state
// detector and the emergent-rate validation consume. Mdot(t) is
// mass_delta/dt, the accretion drag F_acc(t) is force; both are FUNCTIONALS of
// the solved flow (the drain's conserved-variable deltas reduced over the
// mask), never a prescribed rate.
//
// columnar storage so the checkpoint writer borrows each series as one flat
// dataset. the series restarts empty on checkpoint load — earlier segments
// live in the earlier checkpoint files.
//
// usage:
//   history.push(time, dt, &step_deltas);
//   // checkpoint: shape [len] for time/dt, [len, nb] per-body scalars,
//   // [len, nb, D] for force.
// =============================================================================

use crate::body_delta::BodyDelta;

/// the per-step series of body-gas exchanges, columnar. `nb` bodies fixed at
/// construction; one row appended per step.
#[derive(Debug)]
pub struct BodyHistory<const D: usize> {
    nb: usize,
    time: Vec<f64>,
    dt: Vec<f64>,
    /// mass removed from the gas this step, per body: shape [len, nb].
    mass_delta: Vec<f64>,
    /// energy removed from the gas this step, per body: shape [len, nb].
    energy_delta: Vec<f64>,
    /// force on the body (gravity reaction + accretion drag): [len, nb, D].
    force: Vec<f64>,
    /// torque on the body (the r x F moment that drives the Euler rotation / precession): the
    /// evolution consumes `torque_delta` but net force cannot reconstruct it, so record it. always
    /// 3-component (rotation is a 3-space rank-2 object even for a 2D flow): [len, nb, 3].
    torque: Vec<f64>,
}

impl<const D: usize> BodyHistory<D> {
    pub fn new(nb: usize) -> Self {
        Self {
            nb,
            time: Vec::new(),
            dt: Vec::new(),
            mass_delta: Vec::new(),
            energy_delta: Vec::new(),
            force: Vec::new(),
            torque: Vec::new(),
        }
    }

    /// append one step's consolidated per-body deltas.
    pub fn push(&mut self, time: f64, dt: f64, deltas: &[BodyDelta<f64, D>]) {
        self.time.push(time);
        self.dt.push(dt);
        for b in 0..self.nb {
            let d = deltas.get(b);
            self.mass_delta.push(d.map_or(0.0, |d| d.mass_delta));
            self.energy_delta.push(d.map_or(0.0, |d| d.energy_delta));
            for ax in 0..D {
                self.force.push(d.map_or(0.0, |d| d.force_delta[ax]));
            }
            for ax in 0..3 {
                self.torque.push(d.map_or(0.0, |d| d.torque_delta[ax]));
            }
        }
    }

    pub fn len(&self) -> usize {
        self.time.len()
    }

    pub fn is_empty(&self) -> bool {
        self.time.is_empty()
    }

    pub fn n_bodies(&self) -> usize {
        self.nb
    }

    pub fn time(&self) -> &[f64] {
        &self.time
    }

    pub fn dt(&self) -> &[f64] {
        &self.dt
    }

    pub fn mass_delta(&self) -> &[f64] {
        &self.mass_delta
    }

    pub fn energy_delta(&self) -> &[f64] {
        &self.energy_delta
    }

    pub fn force(&self) -> &[f64] {
        &self.force
    }

    pub fn torque(&self) -> &[f64] {
        &self.torque
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use symbi_algebra::Tensor;

    #[test]
    fn rows_are_columnar_and_body_major() {
        let mut h = BodyHistory::<2>::new(2);
        let mut d0 = BodyDelta::<f64, 2>::new(0);
        d0.mass_delta = 1.0;
        d0.force_delta = Tensor::new([3.0, 4.0]);
        d0.torque_delta = Tensor::new([0.0, 0.0, 5.0]);
        let mut d1 = BodyDelta::<f64, 2>::new(1);
        d1.mass_delta = 2.0;
        h.push(0.5, 0.1, &[d0, d1]);
        h.push(0.6, 0.1, &[d1, d0]);
        assert_eq!(h.len(), 2);
        assert_eq!(h.time(), &[0.5, 0.6]);
        assert_eq!(h.mass_delta(), &[1.0, 2.0, 2.0, 1.0]);
        assert_eq!(h.force(), &[3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 3.0, 4.0]);
        // torque is always 3-component + body-major: row0 = [d0, d1], row1 = [d1, d0].
        assert_eq!(h.torque(), &[0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0]);
    }

    #[test]
    fn missing_deltas_pad_with_zero() {
        let mut h = BodyHistory::<1>::new(2);
        let d0 = BodyDelta::<f64, 1>::new(0);
        h.push(0.0, 0.1, &[d0]);
        assert_eq!(h.mass_delta(), &[0.0, 0.0]);
        assert_eq!(h.force().len(), 2);
    }
}
