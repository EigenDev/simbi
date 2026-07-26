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
        ContainerId, MassTransfer, SamplingKey, TransportKernel, sample_convex_blend,
        sample_systematic,
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
