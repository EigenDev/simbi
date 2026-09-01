// =============================================================================
// drain_rate_emergence.rs
//
// whether the accretion rate onto an immersed sink is emergent (set by how fast the flow can
// resupply the mask) or imposed (set by the penalization dial). the drain relaxes masked gas at
// `1/tau = k_drain sqrt(GM/r_acc^3)` — k_drain free-fall rates at the mask radius, a constant of
// the problem data; `k_drain` is declared a convergence-study parameter, which is a claim that
// the measured `Mdot` stops depending on it once the drain is fast enough. that is a falsifiable
// statement about the flow, and this is the measurement.
//
// the two hypotheses separate cleanly under a geometric sweep of the dial:
//   - imposed:  Mdot ~ 1/tau ~ k_drain, so each doubling of k_drain doubles Mdot and the
//               relative change per doubling stays ~1 forever.
//   - emergent: Mdot approaches a limit, so the relative change per doubling shrinks toward 0.
// the gate is that the successive relative changes decrease — a convergence statement, carrying
// no tuned tolerance.
//
// two preconditions decide whether the sweep means anything, and both are asserted, because
// either one failing makes the flat result vacuous rather than informative:
//   - the arms must actually differ: the dial has to reach the kernel, so at least one pair of
//     swept Mdots is distinct. four bitwise-identical measurements are a disconnected dial, and
//     a disconnected dial converges perfectly while measuring nothing.
//   - the mask reservoir must be in balance: more mass flows through the sink during the window
//     than the mask's own gas content changes by. a reservoir still filling or emptying reports a
//     transient, and a rate read off a transient says nothing about what sets it in steady state.
//
// the sink is a mask-radius body with `sink_rate = 0`, so the pointwise body-source drain is an
// exact no-op and the surface penalization is the only mass sink; with periodic walls it is the
// only mass sink in the problem, so `Mdot` is exactly the interior mass loss rate.
//
// run: cargo test -p symbi --test drain_rate_emergence
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
use symbi_xpu::{CpuSpace, HostMemory};

// gamma = 5/3 places the bondi sonic radius (5 - 3 gamma) R_B / 4 at the origin, so the sonic
// surface attaches to the accretor and no sonic point stands between the mask and the flow to set
// the rate on its own. that is the regime the dial's insensitivity claim is least protected in.
const GAMMA: f64 = 5.0 / 3.0;
const N: usize = 64;
const L: f64 = 1.0;
const CFL: f64 = 0.4;

const RHO_0: f64 = 1.0;
const PRE_0: f64 = 1.0;
const MASS: f64 = 0.5;
const R_ACC: f64 = 0.1;

// the spin-up carries the flow past the initial fill of the mask and into resupply; the window is
// the interval the rate is measured over. both are in units where the free-fall time from the
// bondi radius R_B = mass / c_s^2 = 0.3 is sqrt(R_B^3 / mass) = 0.23, so the spin-up is ~9
// free-fall times and the window ~2.
const T_SPINUP: f64 = 2.0;
const T_WINDOW: f64 = 0.5;

type Sim = SimState<Newtonian, 2, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;
type Kset = AdiabaticSubstrateKernelSet<HostMemory, f64, 2>;

fn dx() -> f64 {
    2.0 * L / N as f64
}

/// one free-fall rate at the mask radius: `sqrt(G M / r_acc^3)`. the drain runs at
/// `k_drain` multiples of this.
fn free_fall_rate() -> f64 {
    (MASS / (R_ACC * R_ACC * R_ACC)).sqrt()
}

/// total gas mass on the interior. with periodic walls the drain is the only sink, so the
/// difference of this across an interval is exactly the accreted mass.
fn interior_mass(sim: &Sim) -> f64 {
    let cell_volume: f64 = sim.geom.dx.iter().product();
    sim.geom
        .interior
        .iter()
        .map(|c| *sim.fields.cons.den.view().at(c) * cell_volume)
        .sum()
}

/// the gas mass held inside the mask — the reservoir the sink draws from. its change across the
/// window, measured against the mass that flowed through, is what separates a steady flux from a
/// reservoir still filling or emptying.
fn mask_mass(sim: &Sim) -> f64 {
    let h = dx();
    let cell_volume: f64 = sim.geom.dx.iter().product();
    let mut mass = 0.0;
    let mut n_masked = 0usize;
    for c in sim.geom.interior.iter() {
        let x = -L + (c[0] as f64 + 0.5) * h;
        let y = -L + (c[1] as f64 + 0.5) * h;
        if (x * x + y * y).sqrt() < R_ACC {
            mass += *sim.fields.cons.den.view().at(c) * cell_volume;
            n_masked += 1;
        }
    }
    assert!(
        n_masked > 0,
        "the mask covers no cell — the grid is too coarse to resolve r_acc"
    );
    mass
}

/// the sink: a mask-radius body carrying gravity, with the pointwise body-source drain switched
/// off (`sink_rate = 0` zeroes `min(sink, cs/dx)` exactly) so the surface penalization is the only
/// channel removing mass and `k_drain` is the only dial on the rate. softening equals the mask
/// radius, so the gravity the resupply flow falls through is the bare point mass everywhere
/// outside the sink.
fn sink() -> BodyCollection<f64, 2> {
    BodyCollection::new().add(Body::black_hole(
        0,
        Tensor::zeros(),
        Tensor::zeros(),
        MASS,
        R_ACC,
        R_ACC, // softening
        0.0,   // sink_rate: the body-source drain is an exact no-op
        1.0,   // sink_delta
        R_ACC, // accretion radius = the mask radius
    ))
}

fn build() -> (Sim, Kset) {
    let h = dx();
    let sim = Sim::build(Newtonian, IdealGas { gamma: GAMMA }, Cartesian)
        .cells([N, N])
        .origin([-L, -L])
        .spacing([h, h])
        .boundaries(Boundaries::uniform(BoundaryType::Periodic))
        .cfl(CFL)
        .allocate()
        .expect("sim allocation")
        .set_initial(|_x: [f64; 2]| Prim {
            rho: RHO_0,
            vel: Tensor::zeros(),
            pre: PRE_0,
        })
        .build()
        .with_bodies(sink());
    let kset = Kset::new(GAMMA, CFL, &sim.geom.allocated);
    (sim, kset)
}

struct Measurement {
    mdot: f64,
    /// the mask reservoir's change across the window, as a fraction of the mass that flowed
    /// through it. below one, throughput dominates storage and the measured rate is a flux.
    storage_fraction: f64,
}

/// evolve past the initial transient, then measure the mass loss rate over the window.
fn measure(k_drain: f64) -> Measurement {
    let (mut sim, mut kset) = build();
    kset.k_drain = k_drain;

    evolve(&mut sim, &kset, T_SPINUP).expect("spin-up");
    let mass_open = interior_mass(&sim);
    let mask_open = mask_mass(&sim);

    evolve(&mut sim, &kset, T_SPINUP + T_WINDOW).expect("measurement window");
    let mass_close = interior_mass(&sim);
    let mask_close = mask_mass(&sim);

    let drained = mass_open - mass_close;
    Measurement {
        mdot: drained / T_WINDOW,
        storage_fraction: (mask_close - mask_open).abs() / drained.abs().max(f64::MIN_POSITIVE),
    }
}

#[test]
fn the_accretion_rate_saturates_as_the_drain_dial_grows() {
    // each entry doubles the drain rate, so the imposed rate spans a factor of eight starting
    // from one free-fall crossing.
    let dials = [1.0, 2.0, 4.0, 8.0];
    let floor = free_fall_rate();

    let measured: Vec<Measurement> = dials.iter().map(|&c| measure(c)).collect();

    // the dial must reach the kernel: a sweep whose arms are bitwise identical converges
    // perfectly while measuring a disconnected dial.
    assert!(
        measured
            .windows(2)
            .any(|w| w[0].mdot.to_bits() != w[1].mdot.to_bits()),
        "every swept Mdot is bitwise identical; the dial does not reach the drain rate"
    );

    for (ii, m) in measured.iter().enumerate() {
        assert!(
            m.mdot > 0.0 && m.mdot.is_finite(),
            "k_drain = {}: no mass was accreted over the window (mdot {:e})",
            dials[ii],
            m.mdot
        );
        assert!(
            m.storage_fraction < 1.0,
            "k_drain = {}: the mask reservoir changed by {:.2} of the mass that flowed through it \
             during the window, so the sink is still filling or emptying and the number measured \
             is a transient rather than a steady flux",
            dials[ii],
            m.storage_fraction
        );
    }

    // the relative change in Mdot across each halving of the dial. a rate fixed by the drain holds
    // these near unity; a rate fixed by the flow drives them toward zero.
    let steps: Vec<f64> = measured
        .windows(2)
        .map(|w| (w[1].mdot - w[0].mdot).abs() / w[0].mdot)
        .collect();

    let report: Vec<String> = dials
        .iter()
        .zip(&measured)
        .map(|(c, m)| {
            format!(
                "k_drain {c} (imposed rate {:.1}) -> mdot {:.6e}, storage {:.3}",
                c * free_fall_rate(),
                m.mdot,
                m.storage_fraction
            )
        })
        .collect();
    println!("one free-fall rate {floor:.2}");
    for line in &report {
        println!("{line}");
    }
    for (ii, s) in steps.iter().enumerate() {
        println!(
            "doubling {} -> {}: relative change {s:.4}",
            dials[ii],
            dials[ii + 1]
        );
    }

    for ii in 1..steps.len() {
        assert!(
            steps[ii] < steps[ii - 1],
            "the accretion rate is not saturating: the relative change per halving of the dial \
             went {:.4} -> {:.4}, so a faster drain keeps buying more accretion and the rate is \
             IMPOSED by the penalization rather than emergent from the flow. measured: {}",
            steps[ii - 1],
            steps[ii],
            report.join(", ")
        );
    }
}
