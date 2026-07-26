// =============================================================================
// mass_transport.rs
//
// discrete mass-transport kernels and conservative low-variance sampling for
// passive tracer ownership. a kernel records non-negative mass transfers from
// one source container to its destinations. systematic resampling converts the
// normalized kernel into an exact-count, unbiased tracer assignment.
//
// usage:
//  let kernel = TransportKernel::new(source, mass, transfers)?;
//  let assignments = sample_systematic(&kernel, &tracer_ids, key);
// =============================================================================

use std::collections::BTreeMap;

/// stable identity for a fluid cell or material reservoir.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ContainerId(pub u64);

/// accepted mass transferred from the kernel source to one destination.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MassTransfer {
    pub destination: ContainerId,
    pub mass: f64,
}

/// one source container's normalized discrete mass-transport law.
#[derive(Clone, Debug, PartialEq)]
pub struct TransportKernel {
    source: ContainerId,
    source_mass: f64,
    destinations: Vec<(ContainerId, f64)>,
}

impl TransportKernel {
    /// validate, combine, and normalize accepted outgoing transfers.
    pub fn new(
        source: ContainerId,
        source_mass: f64,
        transfers: impl IntoIterator<Item = MassTransfer>,
    ) -> Result<Self, String> {
        if !source_mass.is_finite() || source_mass < 0.0 {
            return Err(format!(
                "invalid source mass {source_mass:?} for container {}",
                source.0
            ));
        }

        let mut combined = BTreeMap::<ContainerId, f64>::new();
        for transfer in transfers {
            if !transfer.mass.is_finite() || transfer.mass < 0.0 {
                return Err(format!(
                    "invalid transfer mass {:?} from container {} to {}",
                    transfer.mass, source.0, transfer.destination.0
                ));
            }
            if transfer.destination == source || transfer.mass == 0.0 {
                continue;
            }
            *combined.entry(transfer.destination).or_insert(0.0) += transfer.mass;
        }

        let outgoing: f64 = combined.values().sum();
        let tolerance = 32.0 * f64::EPSILON * source_mass.max(outgoing).max(1.0);
        if outgoing > source_mass + tolerance {
            return Err(format!(
                "outgoing mass {outgoing:?} exceeds source mass {source_mass:?} for container {}",
                source.0
            ));
        }
        if source_mass == 0.0 && outgoing > 0.0 {
            return Err(format!(
                "zero-mass container {} has outgoing mass {outgoing:?}",
                source.0
            ));
        }

        let scale = if source_mass > 0.0 {
            source_mass.recip()
        } else {
            0.0
        };
        let mut destinations: Vec<(ContainerId, f64)> = combined
            .into_iter()
            .map(|(destination, mass)| (destination, mass * scale))
            .collect();
        let outgoing_probability: f64 = destinations.iter().map(|entry| entry.1).sum();
        destinations.push((source, (1.0 - outgoing_probability).max(0.0)));
        destinations.sort_by_key(|entry| entry.0);

        Ok(Self {
            source,
            source_mass,
            destinations,
        })
    }

    pub fn source(&self) -> ContainerId {
        self.source
    }

    pub fn source_mass(&self) -> f64 {
        self.source_mass
    }

    pub fn destinations(&self) -> &[(ContainerId, f64)] {
        &self.destinations
    }
}

/// deterministic key for one source container in one accepted transport epoch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SamplingKey {
    pub run_seed: u64,
    pub epoch: u64,
}

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ItoOrder {
    Two = 2,
    Three = 3,
}

/// first three per-time displacement moments of one axis of an accepted jump law.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct JumpMomentRates {
    pub drift: f64,
    pub variance: f64,
    pub third: f64,
}

impl JumpMomentRates {
    pub fn from_probabilities(
        p_plus: f64,
        p_minus: f64,
        cell_width: f64,
        dt: f64,
    ) -> Result<Self, String> {
        if !p_plus.is_finite()
            || !p_minus.is_finite()
            || p_plus < 0.0
            || p_minus < 0.0
            || p_plus + p_minus > 1.0 + 32.0 * f64::EPSILON
        {
            return Err(
                "jump probabilities must be finite, non-negative, and sum to at most one"
                    .to_string(),
            );
        }
        if !cell_width.is_finite() || cell_width <= 0.0 {
            return Err("jump cell width must be positive and finite".to_string());
        }
        if !dt.is_finite() || dt <= 0.0 {
            return Err("jump timestep must be positive and finite".to_string());
        }
        let difference = p_plus - p_minus;
        let sum = p_plus + p_minus;
        let variance_factor = (sum - difference * difference).max(0.0);
        Ok(Self {
            drift: difference * cell_width / dt,
            variance: variance_factor * cell_width * cell_width / dt,
            third: difference
                * (1.0 - 3.0 * sum + 2.0 * difference * difference)
                * cell_width.powi(3)
                / dt,
        })
    }

    pub fn skewness(self, dt: f64) -> f64 {
        let variance = self.variance * dt;
        if variance == 0.0 {
            0.0
        } else {
            self.third * dt / variance.powf(1.5)
        }
    }
}

pub fn accepted_face_moment_rates(
    source_mass: f64,
    mass_to_minus: f64,
    mass_to_plus: f64,
    cell_width: f64,
    dt: f64,
) -> Result<JumpMomentRates, String> {
    if !source_mass.is_finite() || source_mass <= 0.0 {
        return Err("ito coefficient source mass must be positive and finite".to_string());
    }
    if !mass_to_minus.is_finite()
        || !mass_to_plus.is_finite()
        || mass_to_minus < 0.0
        || mass_to_plus < 0.0
    {
        return Err("accepted outward face masses must be finite and non-negative".to_string());
    }
    let outgoing = mass_to_minus + mass_to_plus;
    let tolerance = 32.0 * f64::EPSILON * source_mass.max(outgoing).max(1.0);
    if outgoing > source_mass + tolerance {
        return Err(format!(
            "accepted outward face mass {outgoing:?} exceeds source mass {source_mass:?}"
        ));
    }
    JumpMomentRates::from_probabilities(
        mass_to_plus / source_mass,
        mass_to_minus / source_mass,
        cell_width,
        dt,
    )
}

/// zero-mean, unit-variance piecewise-skew-uniform distribution.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PiecewiseSkewUniform {
    left_extent: f64,
    right_extent: f64,
    left_density: f64,
    right_density: f64,
}

impl PiecewiseSkewUniform {
    pub fn new(skewness: f64) -> Result<Self, String> {
        if !skewness.is_finite() {
            return Err("piecewise-skew-uniform skewness must be finite".to_string());
        }
        let root = (27.0 + 4.0 * skewness * skewness).sqrt();
        let left_extent = (root - 2.0 * skewness) / 3.0;
        let right_extent = (root + 2.0 * skewness) / 3.0;
        let extent_sum = left_extent + right_extent;
        Ok(Self {
            left_extent,
            right_extent,
            left_density: right_extent / (left_extent * extent_sum),
            right_density: left_extent / (right_extent * extent_sum),
        })
    }

    pub fn sample(self, unit: f64) -> Result<f64, String> {
        if !unit.is_finite() || !(0.0..1.0).contains(&unit) {
            return Err("piecewise-skew-uniform sample must lie in [0, 1)".to_string());
        }
        let left_mass = self.left_density * self.left_extent;
        Ok(if unit < left_mass {
            -self.left_extent + unit / self.left_density
        } else {
            (unit - left_mass) / self.right_density
        })
    }
}

pub fn ito2_displacement(rates: JumpMomentRates, dt: f64, unit: f64) -> Result<f64, String> {
    if !dt.is_finite() || dt <= 0.0 {
        return Err("ito tracer timestep must be positive and finite".to_string());
    }
    if !unit.is_finite() || !(0.0..1.0).contains(&unit) {
        return Err("ito tracer sample must lie in [0, 1)".to_string());
    }
    let standardized = 12.0_f64.sqrt() * (unit - 0.5);
    Ok(rates.drift * dt + (rates.variance * dt).sqrt() * standardized)
}

pub fn ito3_displacement(rates: JumpMomentRates, dt: f64, unit: f64) -> Result<f64, String> {
    if !dt.is_finite() || dt <= 0.0 {
        return Err("ito tracer timestep must be positive and finite".to_string());
    }
    let standardized = PiecewiseSkewUniform::new(rates.skewness(dt))?.sample(unit)?;
    Ok(rates.drift * dt + (rates.variance * dt).sqrt() * standardized)
}

/// deterministic counter-based sample for one particle axis and update.
pub fn ito_unit_sample(run_seed: u64, particle_id: u64, counter: u64, axis: usize) -> f64 {
    unit_f64(mix64(
        run_seed
            ^ mix64(particle_id)
            ^ mix64(counter)
            ^ mix64(axis as u64)
            ^ 0x243f_6a88_85a3_08d3,
    ))
}

/// assign every original tracer to exactly one kernel destination using a
/// randomly shifted systematic lattice and a keyed id permutation.
pub fn sample_systematic(
    kernel: &TransportKernel,
    tracer_ids: &[u64],
    key: SamplingKey,
) -> Vec<(u64, ContainerId)> {
    if tracer_ids.is_empty() {
        return Vec::new();
    }

    let source_key = mix64(key.run_seed ^ mix64(key.epoch) ^ mix64(kernel.source.0));
    let count = tracer_ids.len();
    let offset = unit_f64(mix64(source_key ^ 0x6a09_e667_f3bc_c909)) / count as f64;

    let mut ordered_ids = tracer_ids.to_vec();
    ordered_ids.sort_by_key(|id| (mix64(source_key ^ 0xbb67_ae85_84ca_a73b ^ mix64(*id)), *id));

    let mut assignments = Vec::with_capacity(count);
    let mut destination_index = 0usize;
    let mut cumulative = kernel.destinations[0].1;
    for (index, id) in ordered_ids.into_iter().enumerate() {
        let point = offset + index as f64 / count as f64;
        while destination_index + 1 < kernel.destinations.len() && point >= cumulative {
            destination_index += 1;
            cumulative += kernel.destinations[destination_index].1;
        }
        assignments.push((id, kernel.destinations[destination_index].0));
    }
    assignments.sort_by_key(|entry| entry.0);
    assignments
}

/// choose the candidate ancestry for an ssp convex blend. false selects the
/// step-entry snapshot and true selects the forward-euler candidate.
pub fn sample_convex_blend(
    tracer_ids: &[u64],
    candidate_weight: f64,
    key: SamplingKey,
) -> Result<Vec<(u64, bool)>, String> {
    if !candidate_weight.is_finite() || !(0.0..=1.0).contains(&candidate_weight) {
        return Err(format!("invalid candidate weight {candidate_weight:?}"));
    }
    if tracer_ids.is_empty() {
        return Ok(Vec::new());
    }

    let source_key = mix64(key.run_seed ^ mix64(key.epoch) ^ 0x3c6e_f372_fe94_f82b);
    let count = tracer_ids.len();
    let offset = unit_f64(mix64(source_key ^ 0xa54f_f53a_5f1d_36f1)) / count as f64;
    let snapshot_weight = 1.0 - candidate_weight;

    let mut ordered_ids = tracer_ids.to_vec();
    ordered_ids.sort_by_key(|id| (mix64(source_key ^ 0x510e_527f_ade6_82d1 ^ mix64(*id)), *id));
    let mut selections: Vec<(u64, bool)> = ordered_ids
        .into_iter()
        .enumerate()
        .map(|(index, id)| {
            let point = offset + index as f64 / count as f64;
            (id, point >= snapshot_weight)
        })
        .collect();
    selections.sort_by_key(|entry| entry.0);
    Ok(selections)
}

fn mix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

fn unit_f64(value: u64) -> f64 {
    const SCALE: f64 = 1.0 / ((1u64 << 53) as f64);
    ((value >> 11) as f64) * SCALE
}

#[cfg(test)]
mod tests {
    use super::{
        ContainerId, JumpMomentRates, MassTransfer, PiecewiseSkewUniform, SamplingKey,
        TransportKernel, accepted_face_moment_rates, ito2_displacement, ito3_displacement,
        sample_convex_blend, sample_systematic,
    };
    use std::collections::BTreeMap;

    fn kernel() -> TransportKernel {
        TransportKernel::new(
            ContainerId(7),
            10.0,
            [
                MassTransfer {
                    destination: ContainerId(9),
                    mass: 2.5,
                },
                MassTransfer {
                    destination: ContainerId(3),
                    mass: 1.25,
                },
            ],
        )
        .unwrap()
    }

    fn psu_moments(distribution: PiecewiseSkewUniform) -> (f64, f64, f64, f64) {
        let left = distribution.left_extent;
        let right = distribution.right_extent;
        let dl = distribution.left_density;
        let dr = distribution.right_density;
        let norm = dl * left + dr * right;
        let mean = (-dl * left.powi(2) + dr * right.powi(2)) / 2.0;
        let second = (dl * left.powi(3) + dr * right.powi(3)) / 3.0;
        let third = (-dl * left.powi(4) + dr * right.powi(4)) / 4.0;
        (norm, mean, second, third)
    }

    #[test]
    fn jump_moment_rates_match_the_discrete_transition() {
        let rates = JumpMomentRates::from_probabilities(0.3, 0.1, 2.0, 0.5).unwrap();
        assert!((rates.drift * 0.5 - 0.4).abs() < 1e-15);
        assert!((rates.variance * 0.5 - 1.44).abs() < 1e-15);
        assert!((rates.third * 0.5 + 0.192).abs() < 1e-15);
    }

    #[test]
    fn accepted_face_masses_define_the_same_jump_law() {
        let direct = JumpMomentRates::from_probabilities(0.3, 0.1, 2.0, 0.5).unwrap();
        let accepted = accepted_face_moment_rates(10.0, 1.0, 3.0, 2.0, 0.5).unwrap();
        assert_eq!(accepted, direct);
        assert!(
            accepted_face_moment_rates(1.0, 0.6, 0.5, 1.0, 0.1)
                .unwrap_err()
                .contains("exceeds source mass")
        );
    }

    #[test]
    fn piecewise_skew_uniform_has_requested_first_three_moments() {
        for skewness in [-4.0, -0.5, 0.0, 0.5, 4.0] {
            let distribution = PiecewiseSkewUniform::new(skewness).unwrap();
            let (norm, mean, variance, third) = psu_moments(distribution);
            assert!((norm - 1.0).abs() < 2e-14);
            assert!(mean.abs() < 2e-14);
            assert!((variance - 1.0).abs() < 2e-14);
            assert!((third - skewness).abs() < 2e-13);
        }
    }

    #[test]
    fn ito_displacements_match_the_requested_support_and_center() {
        let rates = JumpMomentRates::from_probabilities(0.3, 0.1, 2.0, 0.5).unwrap();
        let drift = rates.drift * 0.5;
        let ito2_center = ito2_displacement(rates, 0.5, 0.5).unwrap();
        let ito3 = PiecewiseSkewUniform::new(rates.skewness(0.5)).unwrap();
        let ito3_split = ito3.left_density * ito3.left_extent;
        let ito3_center = ito3_displacement(rates, 0.5, ito3_split).unwrap();
        assert!((ito2_center - drift).abs() < 1e-15);
        assert!((ito3_center - drift).abs() < 1e-15);
    }

    #[test]
    fn kernel_rejects_mass_creation() {
        let err = TransportKernel::new(
            ContainerId(1),
            1.0,
            [MassTransfer {
                destination: ContainerId(2),
                mass: 1.01,
            }],
        )
        .unwrap_err();
        assert!(err.contains("exceeds source mass"), "{err}");
    }

    #[test]
    fn kernel_probabilities_close() {
        let kernel = kernel();
        let total: f64 = kernel.destinations().iter().map(|entry| entry.1).sum();
        assert!(
            (total - 1.0).abs() <= 8.0 * f64::EPSILON,
            "probability sum is {total:e}"
        );
    }

    #[test]
    fn duplicate_destinations_are_combined_and_ordered() {
        let kernel = TransportKernel::new(
            ContainerId(4),
            8.0,
            [
                MassTransfer {
                    destination: ContainerId(9),
                    mass: 1.0,
                },
                MassTransfer {
                    destination: ContainerId(2),
                    mass: 2.0,
                },
                MassTransfer {
                    destination: ContainerId(9),
                    mass: 3.0,
                },
            ],
        )
        .unwrap();
        assert_eq!(
            kernel.destinations(),
            &[
                (ContainerId(2), 0.25),
                (ContainerId(4), 0.25),
                (ContainerId(9), 0.5)
            ]
        );
    }

    #[test]
    fn systematic_count_conservation_and_discrepancy_bound() {
        let kernel = kernel();
        let ids: Vec<u64> = (0..101).collect();
        let assignments = sample_systematic(
            &kernel,
            &ids,
            SamplingKey {
                run_seed: 11,
                epoch: 29,
            },
        );
        assert_eq!(assignments.len(), ids.len());

        let mut counts = BTreeMap::<ContainerId, usize>::new();
        for (_, destination) in assignments {
            *counts.entry(destination).or_insert(0) += 1;
        }
        for &(destination, probability) in kernel.destinations() {
            let actual = *counts.get(&destination).unwrap_or(&0) as f64;
            let quota = ids.len() as f64 * probability;
            assert!(
                (actual - quota).abs() < 1.0,
                "destination {} count {actual} differs from quota {quota}",
                destination.0
            );
        }
    }

    #[test]
    fn sampling_is_order_invariant_and_reproducible() {
        let first = TransportKernel::new(
            ContainerId(1),
            4.0,
            [
                MassTransfer {
                    destination: ContainerId(3),
                    mass: 1.0,
                },
                MassTransfer {
                    destination: ContainerId(2),
                    mass: 2.0,
                },
            ],
        )
        .unwrap();
        let second = TransportKernel::new(
            ContainerId(1),
            4.0,
            [
                MassTransfer {
                    destination: ContainerId(2),
                    mass: 2.0,
                },
                MassTransfer {
                    destination: ContainerId(3),
                    mass: 1.0,
                },
            ],
        )
        .unwrap();
        let key = SamplingKey {
            run_seed: 71,
            epoch: 5,
        };
        let forward: Vec<u64> = (0..257).collect();
        let mut reverse = forward.clone();
        reverse.reverse();
        assert_eq!(
            sample_systematic(&first, &forward, key),
            sample_systematic(&second, &reverse, key)
        );
    }

    #[test]
    fn sampling_is_unbiased_over_keys() {
        let kernel = TransportKernel::new(
            ContainerId(1),
            10.0,
            [MassTransfer {
                destination: ContainerId(2),
                mass: 3.7,
            }],
        )
        .unwrap();
        let ids: Vec<u64> = (0..13).collect();
        let trials = 20_000u64;
        let mut moved = 0usize;
        for epoch in 0..trials {
            moved += sample_systematic(
                &kernel,
                &ids,
                SamplingKey {
                    run_seed: 99,
                    epoch,
                },
            )
            .iter()
            .filter(|entry| entry.1 == ContainerId(2))
            .count();
        }
        let measured = moved as f64 / (trials as f64 * ids.len() as f64);
        assert!(
            (measured - 0.37).abs() < 5e-4,
            "measured probability {measured}"
        );

        for id in ids {
            let companions = [id, id + 100, id + 200, id + 300];
            let mut individual_moved = 0usize;
            for epoch in 0..trials {
                let destination = sample_systematic(
                    &kernel,
                    &companions,
                    SamplingKey {
                        run_seed: 101,
                        epoch,
                    },
                )
                .into_iter()
                .find(|entry| entry.0 == id)
                .unwrap()
                .1;
                individual_moved += (destination == ContainerId(2)) as usize;
            }
            let individual_probability = individual_moved as f64 / trials as f64;
            assert!(
                (individual_probability - 0.37).abs() < 0.015,
                "tracer {id} measured probability {individual_probability}"
            );
        }
    }

    #[test]
    fn ssp_ancestry_matches_coefficients() {
        let ids: Vec<u64> = (0..103).collect();
        for &(candidate_weight, expected) in &[
            (1.0, 103.0),
            (0.25, 25.75),
            (2.0 / 3.0, 68.666_666_666_666_67),
        ] {
            let selections = sample_convex_blend(
                &ids,
                candidate_weight,
                SamplingKey {
                    run_seed: 17,
                    epoch: 41,
                },
            )
            .unwrap();
            assert_eq!(selections.len(), ids.len());
            let selected = selections.iter().filter(|entry| entry.1).count() as f64;
            assert!(
                (selected - expected).abs() < 1.0,
                "candidate count {selected} differs from quota {expected}"
            );
        }
    }
}
