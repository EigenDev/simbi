// =============================================================================
// worker_refusals.rs
//
// the distributed loop's first-release scope, enforced at entry: 2D cartesian
// adiabatic newtonian hydro in host memory on a static mesh, inviscid, without
// a passive scalar. each gate builds a store or kernel set that leaves the
// scope by one property and shows the loop refuses it before any step, naming
// the property, on a single worker over the in-process link.
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
use symbi_fabric::{Fabric, Loopback, SessionId, WorkerId};
use symbi_geometry::{Cartesian, Geometry, Spacetime};
use symbi_grid::Field;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::newtonian::Newtonian;
use symbi_hydro::quantity::{Density, Pressure};
use symbi_hydro::state::Prim;
use symbi_sim::atlas::{ExchangePlan, Placement, TileId};
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
    }
}

/// run one worker over one tile with the in-process link; the outcome of entry.
fn run2(sim: &mut Sim2, kern: &Kern2, regime: RegimeKind) -> Result<(), WorkerError> {
    let partition = Partition::uniform([16, 16], [1, 1]).unwrap();
    let schema = plan_schema(&**sim);
    let plan = ExchangePlan::compile(&partition, &Topology::open(), &schema, sim.geom.ng).unwrap();
    let placement = Placement::new(1, vec![WorkerId(0)]).unwrap();
    let exchange = PlanExchange::new(&plan, &placement, WorkerId(0));
    let mut fabric = Fabric::new(
        Loopback::new(),
        WorkerId(0),
        SessionId(1),
        1,
        2,
        &exchange.lens(),
        exchange.sends_per_peer_axis(),
    );
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
    .map(|_| ())
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

#[test]
fn the_in_scope_fixture_is_admitted() {
    let mut sim = sim2();
    let kern = Kern2::new(1.4, 0.4, &sim.geom.allocated);
    run2(&mut sim, &kern, RegimeKind::of::<f64, 2, Newtonian>()).unwrap();
}

#[test]
fn another_regime_is_refused() {
    let mut sim = sim2();
    let kern = Kern2::new(1.4, 0.4, &sim.geom.allocated);
    refused(run2(&mut sim, &kern, RegimeKind::Rhd), "Rhd regime");
}

#[test]
fn a_curvilinear_chart_is_refused() {
    let mut sim = sim2();
    let kern = Kern2::new(1.4, 0.4, &sim.geom.allocated);
    sim.geom.coords = Geometry::Spherical;
    refused(
        run2(&mut sim, &kern, RegimeKind::Newtonian),
        "Spherical chart",
    );
}

#[test]
fn a_curved_spacetime_is_refused() {
    let mut sim = sim2();
    let kern = Kern2::new(1.4, 0.4, &sim.geom.allocated);
    sim.geom.spacetime = Spacetime::SchwarzschildKS;
    refused(run2(&mut sim, &kern, RegimeKind::Newtonian), "spacetime");
}

#[test]
fn an_isothermal_closure_is_refused() {
    let mut sim = sim2();
    let kern = Kern2::new(1.4, 0.4, &sim.geom.allocated);
    sim.fields.prim.pre = None;
    refused(run2(&mut sim, &kern, RegimeKind::Newtonian), "isothermal");
}

#[test]
fn a_passive_scalar_is_refused() {
    let mut sim = sim2();
    let kern = Kern2::new(1.4, 0.4, &sim.geom.allocated);
    sim.fields.prim.chi = Some(Field::zeros(&sim.geom.allocated).unwrap());
    sim.fields.cons.chi = Some(Field::zeros(&sim.geom.allocated).unwrap());
    refused(
        run2(&mut sim, &kern, RegimeKind::Newtonian),
        "passive scalar",
    );
}

#[test]
fn viscous_transport_is_refused() {
    let mut sim = sim2();
    let kern = Kern2::new(1.4, 0.4, &sim.geom.allocated).with_viscosity(0.01);
    refused(run2(&mut sim, &kern, RegimeKind::Newtonian), "viscous");
    let kern = Kern2::new(1.4, 0.4, &sim.geom.allocated).with_alpha(0.1);
    refused(run2(&mut sim, &kern, RegimeKind::Newtonian), "viscous");
}

#[test]
fn a_one_dimensional_grid_is_refused() {
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
    let exchange = PlanExchange::new(&plan, &placement, WorkerId(0));
    let mut fabric = Fabric::new(
        Loopback::new(),
        WorkerId(0),
        SessionId(1),
        1,
        1,
        &exchange.lens(),
        exchange.sends_per_peer_axis(),
    );
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
    .map(|_| ());
    refused(result, "1-dimensional");
}
