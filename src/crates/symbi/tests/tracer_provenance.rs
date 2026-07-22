// =============================================================================
// tracer_provenance.rs
//
// the G-flux provenance gate: a mass-weighted tracer population and the sink's
// accretion ledger measure the SAME infall through two disjoint instruments —
// the ledger integrates the penalization drain's per-cell conserved removal,
// the tracers count crossings of the accretion radius times their per-tracer
// mass weight. on a cool collapsing ambient around a black-hole sink the two
// totals must agree within sampling error plus the drain's finite-time lag
// (gas entering the mask is removed over the drain timescale, while a tracer
// books its full weight at first crossing — the tracer total leads slightly).
//
// this is the tracer subsystem auditing the accretion certificate: an error in
// EITHER instrument (a dead ledger booking, a frozen population, a wrong
// weight) breaks the match.
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
use symbi_sim::tracers::seed_mass_weighted;
use symbi_xpu::{CpuSpace, HostMemory};

const GAMMA: f64 = 1.4;
const N: usize = 64;
const L: f64 = 1.0;
const R_ACC: f64 = 0.15;
const N_TRACERS: usize = 4000;

type Sim = SimState<Newtonian, 2, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;

#[test]
fn tracer_flux_matches_the_sink_ledger() {
    let dx = 2.0 * L / N as f64;
    let mut sim = Sim::build(Newtonian, IdealGas { gamma: GAMMA }, Cartesian)
        .cells([N, N])
        .origin([-L, -L])
        .spacing([dx, dx])
        .boundaries(Boundaries::uniform(BoundaryType::Outflow))
        .allocate()
        .expect("sim")
        .set_initial(|_| Prim { rho: 1.0, vel: Tensor::new([0.0, 0.0]), pre: 0.1 })
        .build()
        .with_bodies(BodyCollection::new().add(Body::black_hole(
            0,
            Tensor::new([0.0, 0.0]),
            Tensor::zeros(),
            0.6,   // gravitating mass
            0.1,   // body radius
            0.05,  // softening
            1.0e3, // sink rate (penalize owns the drain; this arms the kind)
            1.0,   // sink delta
            R_ACC,
        )));
    sim.tracers = Some(seed_mass_weighted(&sim, N_TRACERS));

    let sub =
        AdiabaticSubstrateKernelSet::<HostMemory, f64, 2>::new(GAMMA, 0.4, &sim.geom.allocated);
    evolve(&mut sim, &sub, 0.5).expect("sink infall with tracers");

    let ledger = match sim.immersed.as_ref().unwrap().bodies.get(0).kind {
        symbi_ib::BodyKind::BlackHole { total_accreted_mass, .. } => total_accreted_mass,
        _ => unreachable!(),
    };
    let tr = sim.tracers.as_ref().unwrap();
    let crossed = tr.flags.iter().filter(|f| f.crossed_sink).count();
    let tracer_mass = tr.crossed_mass();

    assert!(ledger > 1e-3, "the sink ledger is dead: {ledger}");
    assert!(
        crossed > 100,
        "too few crossings for a statistical comparison: {crossed}"
    );
    // two instruments, one infall: sampling error ~ 1/sqrt(crossed) plus the
    // drain-lag bias (tracers book at crossing, the drain removes over its
    // timescale), so the tracer total may LEAD the ledger by the mask's
    // undrained residue. demand agreement within 35% and the same order.
    let rel = (tracer_mass - ledger).abs() / ledger;
    assert!(
        rel < 0.35,
        "provenance mismatch: tracer mass {tracer_mass:.4} vs ledger {ledger:.4} \
         (rel {rel:.3}, {crossed} crossings of {N_TRACERS})"
    );
}
