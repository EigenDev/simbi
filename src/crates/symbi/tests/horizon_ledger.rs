// =============================================================================
// horizon_ledger.rs
//
// the GR horizon shell-flux accretion ledger, END TO END through BOTH drivers:
// ambient gas on the cartesian kerr-schild chart falls toward the excised
// hole, and the per-step boundary-flux reduction through the diagnostic shell
// books a NONZERO (mdot, edot) onto the Horizon body — on the uni-grid driver
// AND the single-level hierarchy the python frontend actually runs. the two
// drivers execute the identical phase sequence at the identical dt, so their
// ledgers must agree bitwise.
//
// the ledger booking once existed only in the uni-grid driver: every python
// GR-excised run wrote permanently-zero mdot/edot while the reducer itself
// validated perfectly against Michel. this gate pins the booking to both
// drivers forever.
//
// run: cargo test -p symbi --test horizon_ledger
// =============================================================================

use symbi::regimes::substrate_rhd::RhdSubstrateKernelSet;
use symbi::sim::evolve::evolve;
use symbi::sim::refinement::Hierarchy;
use symbi::sim::state::*;
use symbi::sim::substrate_seam::WithExcision;
use symbi_algebra::Tensor;
use symbi_geometry::SchwarzschildKSCartesian;
use symbi_hydro::Rhd;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::state::Prim;
use symbi_ib::{Body, BodyCollection};
use symbi_xpu::{CpuSpace, HostMemory};

const GAMMA: f64 = 4.0 / 3.0;
const N: usize = 24;
const L: f64 = 1.2;
const MASS: f64 = 0.3; // r_+ = 0.6 on the grid
const R_EXC: f64 = 0.35; // inside r_+, above the metric guard M/2
const R_DIAG: f64 = 0.9; // 3M, outside the horizon, on the grid
// sized for a DEBUG gate: the uniform ambient against the vacuum-floor rim
// drives heavy fofc traffic, so the run is small, hot (pressure-supported),
// and short — a handful of steps is enough for a nonzero shell flux.
const T_FINAL: f64 = 0.05;

type Sim = SimState<Rhd, 2, SchwarzschildKSCartesian<f64>, IdealGas<f64>, CpuSpace, HostMemory>;
type Kern = RhdSubstrateKernelSet<HostMemory, f64, 2>;

fn build() -> (Sim, Kern) {
    let dx = 2.0 * L / N as f64;
    let sim = Sim::build(
        Rhd,
        IdealGas { gamma: GAMMA },
        SchwarzschildKSCartesian { mass: MASS },
    )
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
    .with_bodies(BodyCollection::new().add(Body::horizon(0, R_EXC, R_DIAG)));
    let k = Kern::new(GAMMA, 0.3, &sim.geom.allocated).with_excision(R_EXC);
    (sim, k)
}

fn ledger(sim: &Sim) -> (f64, f64, f64, f64) {
    let im = sim.immersed.as_ref().expect("horizon body");
    match im.bodies.get(0).kind {
        symbi_ib::BodyKind::Horizon {
            total_accreted_mass,
            total_accreted_energy,
            mdot,
            edot,
            ..
        } => (total_accreted_mass, total_accreted_energy, mdot, edot),
        _ => panic!("body 0 is not the horizon"),
    }
}

#[test]
fn both_drivers_book_the_same_nonzero_accretion_ledger() {
    let (mut sim_a, k_a) = build();
    evolve(&mut sim_a, &k_a, T_FINAL).expect("uni-grid GR infall");
    let (ma, ea, mdot_a, edot_a) = ledger(&sim_a);

    let (sim_b, k_b) = build();
    let mut hier = Hierarchy::single(sim_b, k_b);
    hier.evolve(T_FINAL).expect("hierarchy GR infall");
    let (mb, eb, mdot_b, edot_b) = ledger(&hier.levels[0].state);

    // infalling ambient gas MUST book a nonzero ACCUMULATED ledger on both
    // drivers — a zero total is the silently-dead booking this gate exists to
    // catch. the instantaneous mdot is the LAST step's rate and may be zero
    // there (the final dt-clamped sliver), so only the integral is demanded.
    assert!(ma.abs() > 1e-12, "uni-grid ledger dead: total {ma}");
    assert!(mb.abs() > 1e-12, "hierarchy ledger dead: total {mb}");

    // identical phase sequence at identical dt: the ledgers agree bitwise.
    assert_eq!(
        ma.to_bits(),
        mb.to_bits(),
        "accreted-mass ledgers diverged: {ma} vs {mb}"
    );
    assert_eq!(
        ea.to_bits(),
        eb.to_bits(),
        "accreted-energy ledgers diverged: {ea} vs {eb}"
    );
    assert_eq!(
        mdot_a.to_bits(),
        mdot_b.to_bits(),
        "mdot diverged: {mdot_a} vs {mdot_b}"
    );
    assert_eq!(
        edot_a.to_bits(),
        edot_b.to_bits(),
        "edot diverged: {edot_a} vs {edot_b}"
    );
}
