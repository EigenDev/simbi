// =============================================================================
// sink.rs
//
// Bondi-Hoyle accretion math primitives. provides the accretion coefficient
// lambda(gamma), sink weight functions, weighted sum accumulator, and the
// final mdot/r_bh computation from accumulated sums.
//
// the global reduction over cells is a runtime concern (deferred).
// this module provides the per-cell weight mapper and the final
// property computation as pure functions.
//
// usage:
//   let lambda = accretion_coefficient(gamma);
//   let weight = sink_weight(r_mag, r_acc, is_binary);
//   let props = compute_sink_properties(&body, &sums, gamma);
// =============================================================================

use symbi_algebra::{Tensor, OrderedNumeric};
use symbi_ir::algebra::Scalar;
use crate::body::Body;

/// bondi-hoyle accretion rate coefficient lambda(gamma).
/// isothermal (gamma=1): exp(1.5)/4, adiabatic (gamma=5/3): 0.25.
/// general: 0.25 * (2/(5-3*gamma))^((5-3*gamma)/(2*gamma-2)).
pub fn accretion_coefficient<S: Scalar + OrderedNumeric>(gamma: S) -> S {
    let one = S::ONE;
    let diff_iso = (gamma - one).abs();

    if diff_iso < S::from_f64(1e-5) {
        // isothermal
        return S::from_f64(std::f64::consts::E.powf(1.5) / 4.0);
    }

    let five_thirds = S::from_f64(5.0 / 3.0);
    let diff_adi = (gamma - five_thirds).abs();

    if diff_adi < S::from_f64(1e-5) {
        return S::from_f64(0.25);
    }

    // general case
    let five = S::from_f64(5.0);
    let three = S::from_f64(3.0);
    let two = S::from_f64(2.0);
    let quarter = S::from_f64(0.25);

    let num = five - three * gamma;
    let den = two * gamma - two;
    let base = two / num;
    let exponent = num / den;
    quarter * base.powf(exponent)
}

/// sink weight function for a cell at distance `r` from a body
/// with accretion radius `r_acc`.
///
/// binary mode (Dittmann & Ryan 2020): exp(-0.25 * (r/r_acc)^4)
/// standard (Krumholz et al. 2004): exp(-(r/r_k)^2) where r_k = 0.5 * r_acc
pub fn sink_weight<S: Scalar>(r: S, r_acc: S, is_binary: bool) -> S {
    if is_binary {
        let r_norm = r / r_acc;
        let r4 = r_norm * r_norm * r_norm * r_norm;
        (-S::from_f64(0.25) * r4).exp()
    } else {
        let r_kernel = S::from_f64(0.5) * r_acc;
        let r_norm = r / r_kernel;
        (-r_norm * r_norm).exp()
    }
}

/// accumulated weighted sums from all cells contributing to a sink.
#[derive(Clone, Copy, Debug)]
pub struct WeightedSums<S: Scalar, const D: usize> {
    pub weighted_density: S,
    pub weighted_cs: S,
    pub sum_weight: S,
    pub sum_mass: S,
    pub weighted_vel: Tensor<S, D>,
}

impl<S: Scalar, const D: usize> WeightedSums<S, D> {
    pub fn zero() -> Self {
        Self {
            weighted_density: S::ZERO,
            weighted_cs: S::ZERO,
            sum_weight: S::ZERO,
            sum_mass: S::ZERO,
            weighted_vel: Tensor::zeros(),
        }
    }

    /// accumulate a cell contribution.
    pub fn accumulate(&mut self, other: &Self) {
        self.weighted_density = self.weighted_density + other.weighted_density;
        self.weighted_cs = self.weighted_cs + other.weighted_cs;
        self.sum_weight = self.sum_weight + other.sum_weight;
        self.sum_mass = self.sum_mass + other.sum_mass;
        self.weighted_vel = self.weighted_vel + other.weighted_vel;
    }

    /// compute the per-cell contribution to the weighted sums.
    pub fn from_cell(weight: S, rho: S, cs: S, vel: Tensor<S, D>, cell_volume: S) -> Self {
        let mass = cell_volume * rho;
        Self {
            weighted_density: weight * rho,
            weighted_cs: weight * mass * cs,
            sum_weight: weight,
            sum_mass: weight * mass,
            weighted_vel: vel.scale(weight * mass),
        }
    }
}

/// computed sink properties for a single accreting body.
#[derive(Clone, Copy, Debug)]
pub struct SinkProperties<S: Scalar> {
    pub body_idx: usize,
    pub mdot: S,
    pub r_bh: S,
    pub total_weight: S,
}

/// compute Bondi-Hoyle sink properties from accumulated weighted sums.
///
/// returns None if the accumulated weight is too small (no contributing cells).
pub fn compute_sink_properties<S: Scalar + OrderedNumeric, const D: usize>(
    body: &Body<S, D>,
    sums: &WeightedSums<S, D>,
    gamma: S,
) -> Option<SinkProperties<S>> {
    if sums.sum_weight < S::from_f64(1e-10) {
        return None;
    }

    let rho_eff = sums.weighted_density / sums.sum_weight;
    let cs_eff = sums.weighted_cs / sums.sum_mass;

    let v_gas_avg = sums.weighted_vel.scale(S::ONE / sums.sum_mass);
    let v_rel = body.velocity - v_gas_avg;
    let v_eff = v_rel.norm();

    let v_sq = v_eff * v_eff;
    let cs_sq = cs_eff * cs_eff;

    let lambda = accretion_coefficient(gamma);
    let four_pi = S::from_f64(4.0 * std::f64::consts::PI);

    let r_bh = body.mass / (cs_sq + v_sq);
    let mdot = four_pi * r_bh * r_bh * rho_eff
        * (lambda * lambda * cs_sq + v_sq).sqrt();

    Some(SinkProperties {
        body_idx: body.idx,
        mdot,
        r_bh,
        total_weight: sums.sum_weight,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-10 * a.abs().max(b.abs()).max(1.0)
    }

    #[test]
    fn accretion_coefficient_isothermal() {
        let lambda = accretion_coefficient(1.0_f64);
        let expected = std::f64::consts::E.powf(1.5) / 4.0;
        assert!(approx(lambda, expected));
    }

    #[test]
    fn accretion_coefficient_adiabatic() {
        let lambda = accretion_coefficient(5.0_f64 / 3.0);
        assert!(approx(lambda, 0.25));
    }

    #[test]
    fn accretion_coefficient_gamma_1_4() {
        let gamma = 1.4_f64;
        let lambda = accretion_coefficient(gamma);
        // general formula
        let num = 5.0 - 3.0 * gamma; // 0.8
        let den = 2.0 * gamma - 2.0; // 0.8
        let expected = 0.25 * (2.0 / num).powf(num / den);
        assert!(approx(lambda, expected));
    }

    #[test]
    fn sink_weight_standard_at_zero() {
        let w = sink_weight(0.0_f64, 0.2, false);
        assert!(approx(w, 1.0)); // exp(0) = 1
    }

    #[test]
    fn sink_weight_decays() {
        let w_near = sink_weight(0.05_f64, 0.2, false);
        let w_far = sink_weight(0.15_f64, 0.2, false);
        assert!(w_near > w_far);
    }

    #[test]
    fn sink_weight_binary_at_zero() {
        let w = sink_weight(0.0_f64, 0.2, true);
        assert!(approx(w, 1.0));
    }

    #[test]
    fn sink_weight_binary_vs_standard() {
        // binary weight is broader (4th power vs 2nd power)
        let r = 0.15_f64;
        let r_acc = 0.2;
        let w_bin = sink_weight(r, r_acc, true);
        let w_std = sink_weight(r, r_acc, false);
        assert!(w_bin > w_std);
    }

    #[test]
    fn weighted_sums_accumulate() {
        let mut total = WeightedSums::<f64, 2>::zero();
        let a = WeightedSums::from_cell(0.5, 1.0, 1.0, Tensor::new([0.1, 0.0]), 0.01);
        let b = WeightedSums::from_cell(0.3, 2.0, 0.5, Tensor::new([0.0, 0.2]), 0.01);
        total.accumulate(&a);
        total.accumulate(&b);

        assert!(approx(total.weighted_density, 0.5 * 1.0 + 0.3 * 2.0));
        assert!(approx(total.sum_weight, 0.8));
    }

    #[test]
    fn compute_sink_properties_basic() {
        let body = Body::black_hole(
            0, Tensor::new([0.0, 0.0]), Tensor::zeros(),
            1.0, 0.1, 0.04, 10.0, 0.0, 0.2,
        );

        let sums = WeightedSums {
            weighted_density: 1.0,
            weighted_cs: 0.01,
            sum_weight: 1.0,
            sum_mass: 0.01,
            weighted_vel: Tensor::zeros(),
        };

        let props = compute_sink_properties(&body, &sums, 1.4);
        assert!(props.is_some());
        let p = props.unwrap();
        assert!(p.mdot > 0.0);
        assert!(p.r_bh > 0.0);
    }

    #[test]
    fn compute_sink_properties_zero_weight() {
        let body = Body::black_hole(
            0, Tensor::zeros(), Tensor::zeros(),
            1.0, 0.1, 0.04, 10.0, 0.0, 0.2,
        );
        let sums = WeightedSums::<f64, 2>::zero();
        assert!(compute_sink_properties(&body, &sums, 1.4).is_none());
    }
}
