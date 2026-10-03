// =============================================================================
// worker_refusals.rs
//
// the distributed loop's scope, enforced at entry, on the adiabatic newtonian
// fixture in host memory. each refusal gate builds a store or kernel set that
// leaves the scope by one property and shows the loop refuses it before any
// step, naming the property, on a single worker over the in-process link; the
// admitted properties (a one-dimensional grid, a constant viscosity, a dye)
// march to their end.
//
// run: cargo test -p symbi --test worker_refusals
// =============================================================================

use std::ops::ControlFlow;
use std::time::Duration;
use symbi::regimes::substrate_newton::AdiabaticSubstrateKernelSet;
use symbi::sim::decomp::{LocalCopy, Partition, Topology, plan_schema};
use symbi::sim::state::*;
use symbi::sim::substrate_seam::{RegimeKind, WithViscosity};
use symbi_algebra::Tensor;
use symbi_fabric::{Loopback, SessionId, WorkerId};
use symbi_geometry::{Cartesian, Spacetime};
use symbi_hydro::eos::IdealGas;
use symbi_hydro::newtonian::Newtonian;
use symbi_hydro::quantity::{Density, Pressure};
use symbi_hydro::state::Prim;
use symbi_sim::atlas::{ExchangePlan, Placement, TileId};
use symbi_sim::phase_table::PhaseTable;
use symbi_sim::plan_exchange::PlanExchange;
use symbi_sim::worker::{Injection, WorkerConfig, WorkerError, evolve_worker};
use symbi_xpu::{CpuSpace, HostMemory};

type Sim2 = SimState<Newtonian, 2, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;
type Kern2 = AdiabaticSubstrateKernelSet<HostMemory, f64, 2>;
type Sim1 = SimState<Newtonian, 1, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;
type Kern1 = AdiabaticSubstrateKernelSet<HostMemory, f64, 1>;

fn sim2() -> Sim2 {
    Sim2::build(Newtonian, IdealGas { gamma: 1.4 }, Cartesian)
        .cells([16, 16])
        .spacing([1.0 / 16.0; 2])
        .boundaries(Boundaries::uniform(BoundaryType::Outflow))
        .timestepping(Timestepping::Rk2)
        .allocate()
        .unwrap()
        .set_initial(|[x, y]| {
            Prim::adiabatic(
                Density(1.0 + 0.1 * x),
                Tensor::new([0.1 * y, 0.0]),
                Pressure(1.0),
            )
        })
        .build()
}

fn config(regime: RegimeKind) -> WorkerConfig {
    WorkerConfig {
        regime,
        timestepping: Timestepping::Rk2,
        start_time: 0.0,
        t_final: 1.0,
        max_steps: 1,
        deadline: Duration::from_secs(2),
        injection: Injection::default(),
        audit_every_point: false,
    }
}

/// run one worker over one tile with the in-process link; the outcome of entry.
fn run2(sim: &mut Sim2, kern: &Kern2, regime: RegimeKind) -> Result<(), WorkerError> {
    let partition = Partition::uniform([16, 16], [1, 1]).unwrap();
    let schema = plan_schema(&**sim);
    let plan = ExchangePlan::compile(&partition, &Topology::open(), &schema, sim.geom.ng).unwrap();
    let placement = Placement::new(1, vec![WorkerId(0)]).unwrap();
    let exchange = PlanExchange::new(&plan, &placement, WorkerId(0), PhaseTable::flat(2).unwrap());
    let mut fabric = exchange.fabric(Loopback::new(), SessionId(1));
    let mut stores = [&mut **sim];
    evolve_worker(
        &[TileId(0)],
        &mut stores,
        &[kern],
        &[0],
        &exchange,
        &LocalCopy,
        &mut fabric,
        &config(regime),
        |_, _, _, _| Ok(ControlFlow::Continue(())),
    )
    .and_then(|_| fabric.finish(Duration::from_secs(2)).map_err(WorkerError::from))
}

fn refused(result: Result<(), WorkerError>, expect: &str) {
    match result {
        Err(WorkerError::Refused(detail)) => assert!(
            detail.contains(expect),
            "refused for {detail:?}, expected {expect:?}"
        ),
        other => panic!("expected a refusal naming {expect:?}, got {other:?}"),
    }
}

/// a forced rejection needs the whole step entry kept: the flat magnetized set keeps no
/// step-entry snapshot, so a run that injects a rejection on it is refused at entry.
#[test]
fn a_forced_rejection_without_a_complete_rollback_snapshot_is_refused() {
    use symbi::prelude::SimSubstrate;
    use symbi_hydro::mhd_state::MhdPrim;
    use symbi_hydro::newtonian_mhd::NewtonianMhd;
    let mut sim = SimStateGeneric::<NewtonianMhd, 2, 3, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>::build(
        NewtonianMhd,
        IdealGas { gamma: 5.0 / 3.0 },
        Cartesian,
    )
    .cells([16, 16])
    .spacing([1.0 / 16.0; 2])
    .boundaries(Boundaries::uniform(BoundaryType::Periodic))
    .timestepping(Timestepping::Rk2)
    .allocate()
    .unwrap()
    .set_initial(|[_x, y]| {
        MhdPrim::new(
            Prim::adiabatic(Density(1.0), Tensor::new([0.1 * y, 0.0, 0.0]), Pressure(1.0)),
            Tensor::new([0.0, 0.5, 0.0]),
        )
    })
    .seed_faces_uniform([0.0, 0.5])
    .build();
    let kern = sim.substrate();
    let partition = Partition::uniform([16, 16], [1, 1]).unwrap();
    let schema = plan_schema(&*sim);
    let plan = ExchangePlan::compile(&partition, &Topology::wrapping([true; 2]), &schema, sim.geom.ng).unwrap();
    let placement = Placement::new(1, vec![WorkerId(0)]).unwrap();
    let exchange = PlanExchange::new(&plan, &placement, WorkerId(0), PhaseTable::flat(2).unwrap());
    let mut fabric = exchange.fabric(Loopback::new(), SessionId(1));
    let mut stores = [&mut *sim];
    let mut cfg = config(RegimeKind::NewtonianMhd);
    cfg.injection.reject_at = vec![(0, 0)];
    let result = evolve_worker(
        &[TileId(0)],
        &mut stores,
        &[&kern],
        &[0],
        &exchange,
        &LocalCopy,
        &mut fabric,
        &cfg,
        |_, _, _, _| Ok(std::ops::ControlFlow::Continue(())),
    )
    .map(|_| ());
    refused(result, "a forced rejection on a kernel set without a complete rollback snapshot");
}

#[test]
fn the_in_scope_fixture_is_admitted() {
    let mut sim = sim2();
    let kern = Kern2::new(1.4, 0.4, &sim.geom.allocated);
    run2(&mut sim, &kern, RegimeKind::of::<f64, 2, Newtonian>()).unwrap();
}

/// flat adiabatic newtonian MHD is inside the scope.
#[test]
fn the_newtonian_mhd_regime_is_admitted() {
    let mut sim = sim2();
    let kern = Kern2::new(1.4, 0.4, &sim.geom.allocated);
    run2(&mut sim, &kern, RegimeKind::NewtonianMhd).expect("newtonian MHD is admitted");
}

#[test]
fn a_curved_spacetime_is_refused() {
    let mut sim = sim2();
    let kern = Kern2::new(1.4, 0.4, &sim.geom.allocated);
    sim.geom.spacetime = Spacetime::SchwarzschildKS;
    refused(run2(&mut sim, &kern, RegimeKind::Newtonian), "spacetime");
}

/// the isothermal closure is admitted uniform and varying: the tile's builder seeds every
/// allocated cell, the cut ghosts included, from the whole grid's closure, and the worker marches
/// the field as given.
#[test]
fn a_locally_isothermal_closure_and_a_global_one_are_admitted() {
    use symbi::regimes::substrate::IsoSubstrateKernelSet;
    use symbi_hydro::eos::Isothermal;
    use symbi_hydro::isothermal::IsoNewtonian;
    use symbi_hydro::state::PrimG;
    type IsoSim = SimStateGeneric<IsoNewtonian, 2, 2, Cartesian, Isothermal<f64>, CpuSpace, HostMemory>;
    let build = || {
        IsoSim::build(IsoNewtonian, Isothermal { cs: 1.0 }, Cartesian)
        .cells([16, 16])
        .spacing([1.0 / 16.0; 2])
        .boundaries(Boundaries::uniform(BoundaryType::Outflow))
        .timestepping(Timestepping::Rk2)
        .allocate()
        .unwrap()
        .set_initial(|[x, y]| PrimG::isothermal(Density(1.0 + 0.1 * x), Tensor::new([0.1 * y, 0.0])))
        .build()
    };
    let mut sim = build();
    let kern = IsoSubstrateKernelSet::<HostMemory, f64, 2>::new(1.0, 0.4, &sim.geom.allocated);
    let plan_run = |sim: &mut IsoSim, kern: &IsoSubstrateKernelSet<HostMemory, f64, 2>| {
        let partition = Partition::uniform([16, 16], [1, 1]).unwrap();
        let schema = plan_schema(&**sim);
        let plan = ExchangePlan::compile(&partition, &Topology::open(), &schema, sim.geom.ng).unwrap();
        let placement = Placement::new(1, vec![WorkerId(0)]).unwrap();
        let exchange = PlanExchange::new(&plan, &placement, WorkerId(0), PhaseTable::flat(2).unwrap());
        let mut fabric = exchange.fabric(Loopback::new(), SessionId(1));
        let mut stores = [&mut **sim];
        evolve_worker(
            &[TileId(0)],
            &mut stores,
            &[kern],
            &[0],
            &exchange,
            &LocalCopy,
            &mut fabric,
            &config(RegimeKind::IsoNewtonian),
            |_, _, _, _| Ok(ControlFlow::Continue(())),
        )
        .map(|_| ())
    };
    plan_run(&mut sim, &kern).expect("a globally isothermal closure is admitted");
    let mut sim = build();
    let kern = IsoSubstrateKernelSet::<HostMemory, f64, 2>::new(1.0, 0.4, &sim.geom.allocated);
    let c = [sim.geom.interior.spaces[0].lo + 3, sim.geom.interior.spaces[1].lo + 4];
    kern.cs2.set(c, 0.25);
    plan_run(&mut sim, &kern).expect("a locally isothermal closure is admitted");
}

/// a dyed run is admitted: the exchange schema carries the concentration across the cuts, and
/// the worker marches the dyed state to its end
#[test]
fn a_passive_scalar_is_admitted() {
    let undyed = plan_schema(&*sim2()).len();
    let mut sim = sim2().with_passive_scalar().expect("dye allocation");
    assert_eq!(plan_schema(&*sim).len(), undyed + 1, "the exchange schema carries the dye");
    let kern = Kern2::new(1.4, 0.4, &sim.geom.allocated);
    run2(&mut sim, &kern, RegimeKind::Newtonian).expect("a passive scalar is admitted");
}

/// the viscous transport is a post-step operation that leaves the halos as current as the
/// interior, so the launched path admits a constant nu; the alpha prescription needs a central
/// body, which the path refuses ahead of it
#[test]
fn viscous_transport_is_admitted() {
    let mut sim = sim2();
    let kern = Kern2::new(1.4, 0.4, &sim.geom.allocated).with_viscosity(0.01);
    run2(&mut sim, &kern, RegimeKind::Newtonian).expect("a constant-nu viscosity is admitted");
}

#[test]
fn a_one_dimensional_grid_is_admitted() {
    let mut sim = Sim1::build(Newtonian, IdealGas { gamma: 1.4 }, Cartesian)
        .cells([32])
        .spacing([1.0 / 32.0])
        .boundaries(Boundaries::uniform(BoundaryType::Outflow))
        .timestepping(Timestepping::Rk2)
        .allocate()
        .unwrap()
        .set_initial(|[x]| {
            Prim::adiabatic(Density(1.0 + 0.1 * x), Tensor::new([0.0]), Pressure(1.0))
        })
        .build();
    let kern = Kern1::new(1.4, 0.4, &sim.geom.allocated);
    let partition = Partition::uniform([32], [1]).unwrap();
    let schema = plan_schema(&*sim);
    let plan = ExchangePlan::compile(&partition, &Topology::open(), &schema, sim.geom.ng).unwrap();
    let placement = Placement::new(1, vec![WorkerId(0)]).unwrap();
    let exchange = PlanExchange::new(&plan, &placement, WorkerId(0), PhaseTable::flat(2).unwrap());
    let mut fabric = exchange.fabric(Loopback::new(), SessionId(1));
    let mut stores = [&mut *sim];
    let result = evolve_worker(
        &[TileId(0)],
        &mut stores,
        &[&kern],
        &[0],
        &exchange,
        &LocalCopy,
        &mut fabric,
        &config(RegimeKind::of::<f64, 1, Newtonian>()),
        |_, _, _, _| Ok(ControlFlow::Continue(())),
    )
    .and_then(|_| fabric.finish(Duration::from_secs(2)).map_err(WorkerError::from));
    result.expect("a one-dimensional grid marches on a worker");
}
