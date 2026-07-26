// =============================================================================
// tracer_provenance.rs
//
// accretion provenance from the same accepted density removal that feeds the
// immersed-body ledger. tracer ownership in the accretion reservoir must
// reproduce the removed mass within finite-population sampling error.
//
// run: cargo test -p symbi --test tracer_provenance
// =============================================================================

use symbi::regimes::substrate_newton::AdiabaticSubstrateKernelSet;
use symbi::sim::evolve::evolve;
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_geometry::Cartesian;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::newtonian::Newtonian;
use symbi_hydro::state::Prim;
use symbi_ib::{Body, BodyCollection};
use symbi_sim::tracers::{
    body_accretion_reservoir, is_accretion_reservoir, seed_mass_weighted,
};
use symbi_xpu::{CpuSpace, HostMemory};

const GAMMA: f64 = 1.4;
const N: usize = 64;
const L: f64 = 1.0;
const N_TRACERS: usize = 4000;

type Sim = SimState<Newtonian, 2, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;

#[test]
fn accretion_reservoir_matches_the_sink_ledger() {
    let dx = 2.0 * L / N as f64;
    let mut sim = Sim::build(Newtonian, IdealGas { gamma: GAMMA }, Cartesian)
        .cells([N, N])
        .origin([-L, -L])
        .spacing([dx, dx])
        .boundaries(Boundaries::uniform(BoundaryType::Outflow))
        .allocate()
        .expect("sim")
        .set_initial(|_| Prim {
            rho: 1.0,
            vel: Tensor::new([0.0, 0.0]),
            pre: 0.1,
        })
        .build()
        .with_bodies(BodyCollection::new().add(Body::black_hole(
            0,
            Tensor::new([0.0, 0.0]),
            Tensor::zeros(),
            0.6,
            0.1,
            0.05,
            1.0e3,
            1.0,
            0.15,
        )));
    sim.tracers = Some(seed_mass_weighted(&sim, N_TRACERS));

    let kernels =
        AdiabaticSubstrateKernelSet::<HostMemory, f64, 2>::new(GAMMA, 0.4, &sim.geom.allocated);
    evolve(&mut sim, &kernels, 0.5).expect("sink infall with tracers");

    let ledger = match sim.immersed.as_ref().unwrap().bodies.get(0).kind {
        symbi_ib::BodyKind::BlackHole {
            total_accreted_mass,
            ..
        } => total_accreted_mass,
        _ => unreachable!(),
    };
    let tracers = sim.tracers.as_ref().unwrap();
    let accreted = tracers
        .owner
        .iter()
        .filter(|&&owner| is_accretion_reservoir(owner))
        .count();
    let tracer_mass = accreted as f64 * tracers.weight;

    assert!(ledger > 1.0e-3, "the sink ledger is dead: {ledger}");
    assert!(
        accreted > 100,
        "too few reservoir transfers for a statistical comparison: {accreted}"
    );
    assert!(
        tracers
            .owner
            .iter()
            .filter(|&&owner| is_accretion_reservoir(owner))
            .all(|&owner| owner == body_accretion_reservoir(0)),
        "single-body accretion lost body-specific reservoir ownership"
    );
    let relative_error = (tracer_mass - ledger).abs() / ledger;
    assert!(
        relative_error < 0.15,
        "reservoir mass {tracer_mass:.4} vs ledger {ledger:.4}, relative error \
         {relative_error:.3} from {accreted} tracers"
    );
}
