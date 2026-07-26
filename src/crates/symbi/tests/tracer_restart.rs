// =============================================================================
// tracer_restart.rs
//
// mass-transport restart continuity. checkpointing after runtime-source
// spawning and continuing from a fresh state must reproduce uninterrupted
// identities, owners, provenance, and fractional spawn state exactly.
// =============================================================================

use symbi::prelude::*;
use symbi::regimes::substrate_newton::AdiabaticSubstrateKernelSet;
use symbi::sim::checkpoint::{load_checkpoint, write_checkpoint};
use symbi_hydro::expr_bridge::build_user_source;
use symbi_hydro::{NEWTONIAN_SPEC, SourceConfig};
use symbi_io::Metadata;
use symbi_sim::mass_transport::ItoOrder;
use symbi_sim::tracers::{ContinuousTracerRecord, ContinuousTracerSet, seed_mass_weighted};

const GAMMA: f64 = 1.4;
const CFL: f64 = 0.4;
const T_MID: f64 = 0.03;
const T_FINAL: f64 = 0.06;

type Sim = SimState<Newtonian, 1, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;
type Kern = AdiabaticSubstrateKernelSet<HostMemory, f64, 1>;

fn make() -> (Sim, Kern) {
    let source = SourceConfig::from_json(
        r#"{
            "kind": "raw", "dim": 1, "outputs": [0], "params": [],
            "target": "den",
            "nodes": [ {"op": "CONSTANT", "value": 0.2} ]
        }"#,
    )
    .unwrap();
    let mut sim = Sim::build(Newtonian, IdealGas { gamma: GAMMA }, Cartesian)
        .cells([32])
        .bounds([0.0], [1.0])
        .boundaries(BoundaryType::Periodic)
        .cfl(CFL)
        .timestepping(Timestepping::Rk2)
        .allocate()
        .unwrap()
        .set_initial(|_| Prim {
            rho: 1.0,
            vel: Tensor::new([0.0]),
            pre: 1.0,
        })
        .build();
    let tracers = seed_mass_weighted(&sim, 1000);
    sim.continuous_tracers = Some(
        ContinuousTracerSet::from_discrete(&tracers, ItoOrder::Three).unwrap(),
    );
    sim.tracers = Some(tracers);
    let kernels = Kern::new(GAMMA, CFL, &sim.geom.allocated).with_runtime_source(
        build_user_source(&source, &NEWTONIAN_SPEC).unwrap(),
        source.params,
    );
    (sim, kernels)
}

fn continuous_records(
    tracers: &ContinuousTracerSet<1, HostMemory>,
) -> Vec<ContinuousTracerRecord<1>> {
    unsafe {
        (0..tracers.len)
            .map(|ii| ContinuousTracerRecord {
                x: [*tracers.x[0].as_ptr::<f64>().add(ii)],
                step_x: [*tracers.step_x[0].as_ptr::<f64>().add(ii)],
                id: *tracers.id.as_ptr::<u64>().add(ii),
                cohort: *tracers.cohort.as_ptr::<u16>().add(ii),
                owner: *tracers
                    .owner
                    .as_ptr::<symbi_sim::mass_transport::ContainerId>()
                    .add(ii),
                escaped: *tracers.escaped.as_ptr::<u8>().add(ii),
                crossed_sink: *tracers.crossed_sink.as_ptr::<u8>().add(ii),
                crossing_time: *tracers.crossing_time.as_ptr::<f64>().add(ii),
                random_counter: *tracers.random_counter.as_ptr::<u64>().add(ii),
            })
            .collect()
    }
}

#[test]
fn source_spawning_continues_identically_after_restart() {
    let (mut uninterrupted, uninterrupted_kernels) = make();
    evolve(&mut uninterrupted, &uninterrupted_kernels, T_MID).unwrap();
    let checkpoint = std::env::temp_dir().join(format!(
        "symbi_tracer_restart_{}.h5",
        std::process::id()
    ));
    write_checkpoint(
        &uninterrupted,
        checkpoint.to_str().unwrap(),
        &Metadata::new(),
    )
    .unwrap();

    let (mut restarted, restarted_kernels) = make();
    load_checkpoint(&mut restarted, checkpoint.to_str().unwrap()).unwrap();
    evolve(&mut uninterrupted, &uninterrupted_kernels, T_FINAL).unwrap();
    evolve(&mut restarted, &restarted_kernels, T_FINAL).unwrap();

    let expected = uninterrupted.tracers.as_ref().unwrap();
    let actual = restarted.tracers.as_ref().unwrap();
    assert_eq!(actual.id, expected.id);
    assert_eq!(actual.owner, expected.owner);
    assert_eq!(actual.flags, expected.flags);
    assert_eq!(actual.step_owner, expected.step_owner);
    assert_eq!(actual.step_flags, expected.step_flags);
    assert_eq!(actual.next_id, expected.next_id);
    assert_eq!(actual.run_seed, expected.run_seed);
    assert_eq!(actual.weight.to_bits(), expected.weight.to_bits());
    assert_eq!(
        actual.injection_remainder.to_bits(),
        expected.injection_remainder.to_bits()
    );
    assert!(
        actual.next_id > 1000,
        "restart gate never exercised source spawning"
    );
    let expected = uninterrupted.continuous_tracers.as_ref().unwrap();
    let actual = restarted.continuous_tracers.as_ref().unwrap();
    assert_eq!(continuous_records(actual), continuous_records(expected));
    assert_eq!(actual.order, expected.order);
    assert_eq!(actual.next_id, expected.next_id);
    assert_eq!(actual.run_seed, expected.run_seed);
    assert_eq!(actual.weight.to_bits(), expected.weight.to_bits());
    assert_eq!(
        actual.injection_remainder.to_bits(),
        expected.injection_remainder.to_bits()
    );
    assert!(
        actual.next_id > 1000,
        "continuous restart gate never exercised source spawning"
    );
}
