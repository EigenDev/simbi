// =============================================================================
// diagnostics.rs
//
// heterogeneous diagnostic accumulator for immersed body feedback.
// collects BodyDelta (force, torque, mass change) from all cells across
// a timestep, then consolidates into per-body totals.
//
// cpu: per-tile local accumulation in DomainForEach, merged via mutex.
//   each tile runs serially (cache-friendly), lock taken once per tile
//   merge. thread-local accumulation followed by a consolidate step.
//
// gpu: per-cell delta field, reduced after the step.
//   (CPU-only accumulation via unified memory sync)
//
// usage:
//   let mut acc = DiagnosticAccumulator::<2>::new(2);  // 2 bodies
//   acc.reset();
//   // ... inside DomainForEach per-cell loop:
//   acc.accumulate(delta);
//   // ... after timestep:
//   let totals = acc.consolidate();
// =============================================================================

use crate::body_delta::BodyDelta;
use std::sync::Mutex;
use symbi_carrier::Scalar;

/// diagnostic accumulator for body-fluid feedback.
/// thread-safe via interior mutex. the mutex is taken once per tile merge, so
/// contention stays low under DomainForEach tiled dispatch.
pub struct DiagnosticAccumulator<S: Scalar, const D: usize> {
    // per-body accumulated deltas. protected by mutex for cross-tile merging.
    totals: Mutex<Vec<BodyDelta<S, D>>>,
    n_bodies: usize,
}

impl<const D: usize> DiagnosticAccumulator<f64, D> {
    /// create an accumulator for the given number of bodies. sized by the
    /// full body count (sources + fragments), unbounded.
    pub fn new(n_bodies: usize) -> Self {
        let totals = (0..n_bodies).map(|ii| BodyDelta::new(ii)).collect();
        DiagnosticAccumulator {
            totals: Mutex::new(totals),
            n_bodies,
        }
    }

    /// reset all accumulators to zero. call at the start of each timestep.
    pub fn reset(&self) {
        let mut totals = self.totals.lock().unwrap();
        for ii in 0..self.n_bodies {
            totals[ii] = BodyDelta::new(ii);
        }
    }

    /// accumulate a single cell's contribution to a body.
    /// this acquires the mutex — call it once per serial tile inner loop.
    /// for per-tile batching from individual cell iterations, use
    /// accumulate_batch.
    pub fn accumulate(&self, delta: BodyDelta<f64, D>) {
        let mut totals = self.totals.lock().unwrap();
        if delta.idx < self.n_bodies {
            totals[delta.idx] += delta;
        }
    }

    /// accumulate a batch of deltas (one per body). designed for per-tile
    /// local accumulation: the caller sums cells within a tile into a
    /// per-body local buffer, then calls this once per tile.
    /// one mutex acquisition per tile.
    pub fn accumulate_batch(&self, deltas: &[BodyDelta<f64, D>]) {
        let mut totals = self.totals.lock().unwrap();
        for delta in deltas {
            if delta.idx < self.n_bodies {
                totals[delta.idx] += *delta;
            }
        }
    }

    /// return per-body totals for this timestep.
    /// call after all tiles have finished.
    pub fn consolidate(&self) -> Vec<BodyDelta<f64, D>> {
        self.totals.lock().unwrap().clone()
    }

    /// number of bodies this accumulator tracks.
    pub fn n_bodies(&self) -> usize {
        self.n_bodies
    }
}

// send + sync: the mutex handles interior synchronization.
unsafe impl<S: Scalar, const D: usize> Send for DiagnosticAccumulator<S, D> {}
unsafe impl<S: Scalar, const D: usize> Sync for DiagnosticAccumulator<S, D> {}

#[cfg(test)]
mod tests {
    use super::*;
    use symbi_algebra::Tensor;

    #[test]
    fn new_accumulator_is_zeroed() {
        let acc = DiagnosticAccumulator::<f64, 2>::new(2);
        let totals = acc.consolidate();
        assert_eq!(totals.len(), 2);
        assert_eq!(totals[0].force_delta, Tensor::zeros());
        assert_eq!(totals[1].mass_delta, 0.0);
    }

    #[test]
    fn accumulate_single() {
        let acc = DiagnosticAccumulator::<f64, 2>::new(1);
        let mut d = BodyDelta::new(0);
        d.force_delta = Tensor::new([1.0, 2.0]);
        d.mass_delta = 0.5;
        acc.accumulate(d);

        let mut d2 = BodyDelta::new(0);
        d2.force_delta = Tensor::new([3.0, 4.0]);
        d2.mass_delta = 0.3;
        acc.accumulate(d2);

        let totals = acc.consolidate();
        assert_eq!(totals[0].force_delta, Tensor::new([4.0, 6.0]));
        assert!((totals[0].mass_delta - 0.8).abs() < 1e-14);
    }

    #[test]
    fn accumulate_batch_two_bodies() {
        let acc = DiagnosticAccumulator::<f64, 2>::new(2);

        let mut d0 = BodyDelta::new(0);
        d0.force_delta = Tensor::new([1.0, 0.0]);

        let mut d1 = BodyDelta::new(1);
        d1.force_delta = Tensor::new([0.0, 2.0]);

        acc.accumulate_batch(&[d0, d1]);

        let totals = acc.consolidate();
        assert_eq!(totals[0].force_delta, Tensor::new([1.0, 0.0]));
        assert_eq!(totals[1].force_delta, Tensor::new([0.0, 2.0]));
    }

    #[test]
    fn reset_clears_all() {
        let acc = DiagnosticAccumulator::<f64, 2>::new(1);
        let mut d = BodyDelta::new(0);
        d.mass_delta = 1.0;
        acc.accumulate(d);

        acc.reset();
        let totals = acc.consolidate();
        assert_eq!(totals[0].mass_delta, 0.0);
    }

    #[test]
    fn concurrent_accumulation() {
        use std::sync::Arc;
        use std::thread;

        let acc = Arc::new(DiagnosticAccumulator::<f64, 2>::new(1));
        let n_threads = 8;
        let n_per_thread = 1000;

        let mut handles = Vec::new();
        for _ in 0..n_threads {
            let acc = acc.clone();
            handles.push(thread::spawn(move || {
                for _ in 0..n_per_thread {
                    let mut d = BodyDelta::new(0);
                    d.mass_delta = 1.0;
                    acc.accumulate(d);
                }
            }));
        }

        for handle in handles {
            handle.join().unwrap();
        }

        let totals = acc.consolidate();
        let expected = (n_threads * n_per_thread) as f64;
        assert!((totals[0].mass_delta - expected).abs() < 1e-10);
    }
}
