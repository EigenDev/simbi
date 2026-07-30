// =============================================================================
// gravity_source_entropy.rs
//
// a gravitational source must not change the entropy of the gas it accelerates.
//
// gravity does work: it adds momentum `S_mom = rho a` and energy `S_nrg = rho a . v`.
// those two have to agree with each other to machine precision, because the internal
// energy is what is LEFT after the kinetic part is removed. if the energy term uses a
// velocity that is even slightly out of step with the one the momentum term produces --
// a stale stage value, say -- the residue lands in the internal energy as `rho a dv dt`,
// one-signed, every step. that is invisible in the conserved totals (both mass and total
// energy stay perfectly conserved) and shows up only as entropy drift.
//
// the setup makes the exact answer trivial. a UNIFORM medium at rest in a UNIFORM
// gravitational field is accelerated identically everywhere, so it never compresses: the
// exact solution is the same uniform state translating, with `K = p / rho^gamma` constant
// for all time. uniformity is obtained by putting the mass very far away, so the field
// varies by ~0.2 percent across the domain -- and even that residue only drives adiabatic
// compression, which preserves K exactly.
//
// so any drift here is the source term, isolated from reconstruction, the riemann solver,
// refinement transfer, sinks, and boundaries.
//
// the uniform case pins the source's SELF-CONSISTENCY but exercises it at only one field
// strength, and a residue proportional to the local acceleration is invisible there. the
// second setup below supplies the missing dynamic range: an isentropic atmosphere in
// HYDROSTATIC BALANCE against a point mass, whose exact solution is that nothing moves and
// `K = p / rho^gamma` stays at its initial value for all time, swept over four decades of
// `GM`. that is the configuration deep in an accretion flow -- strong gravity, near-zero
// velocity, held for a great many steps -- and it is where a per-step residue `rho a dv dt`
// would accumulate fastest while staying invisible in the conserved totals.
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
use symbi::regimes::substrate_kernels::FusedSourceBinding;
use symbi_xpu::{CpuSpace, HostMemory};

const GAMMA: f64 = 5.0 / 3.0;
const N: usize = 128;
const CFL: f64 = 0.4;
const RHO0: f64 = 1.0;
const P0: f64 = 0.6;
/// the mass sits at `-DIST` with `MASS` chosen so `g = MASS / DIST^2` is a gentle 0.1,
/// i.e. mach 0.1 after unit time against `cs = sqrt(gamma P0 / RHO0) = 1`. at this range
/// the field varies by about 0.2 percent across the unit domain.
const DIST: f64 = 1000.0;
const MASS: f64 = 1.0e5;
const T_END: f64 = 1.0;

type Sim = SimState<Newtonian, 1, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;
type Kset = AdiabaticSubstrateKernelSet<HostMemory, f64, 1>;

fn k0() -> f64 {
    P0 / RHO0.powf(GAMMA)
}

fn uniform_medium(with_gravity: bool) -> Sim {
    let sim = Sim::build(Newtonian, IdealGas { gamma: GAMMA }, Cartesian)
        .cells([N])
        .spacing([1.0 / N as f64])
        .boundaries(Boundaries::uniform(BoundaryType::Outflow))
        .cfl(CFL)
        .allocate()
        .expect("sim construction failed")
        .set_initial(|_| Prim {
            rho: RHO0,
            vel: Tensor::new([0.0]),
            pre: P0,
        })
        .build();
    if with_gravity {
        sim.with_bodies(BodyCollection::new().add(Body::gravitational(
            0,
            Tensor::new([-DIST]),
            Tensor::zeros(),
            MASS,
            0.01,
            0.1,
        )))
    } else {
        sim
    }
}

/// the worst `K / K0` over the interior, skipping the two cells nearest each edge so an
/// outflow boundary's own transient is not mistaken for a source-term defect.
fn worst_entropy_ratio(sim: &Sim) -> f64 {
    let rho = sim.fields.prim.rho.view();
    let pre = sim
        .fields
        .prim
        .pre
        .as_ref()
        .expect("adiabatic carries pressure")
        .view();
    let k0 = k0();
    let cells: Vec<_> = sim.geom.interior.iter().collect();
    let mut worst = f64::INFINITY;
    for c in cells.iter().skip(2).take(cells.len().saturating_sub(4)) {
        let r = *rho.at(*c);
        if r > 0.0 {
            worst = worst.min(*pre.at(*c) / r.powf(GAMMA) / k0);
        }
    }
    worst
}

fn run(with_gravity: bool) -> (f64, u64, f64) {
    let mut sim = uniform_medium(with_gravity);
    let kernels = Kset::new(GAMMA, CFL, &sim.geom.allocated);
    evolve(&mut sim, &kernels, T_END).expect("evolve failed");
    let vel = sim.fields.prim.vel[0].view();
    let mut vmax = 0.0_f64;
    for c in sim.geom.interior.iter() {
        vmax = vmax.max(vel.at(c).abs());
    }
    (worst_entropy_ratio(&sim), sim.iteration, vmax)
}

// =============================================================================
// hydrostatic atmosphere in a point-mass potential
// =============================================================================

/// the mass sits one domain-width to the left of `x = 0`, so the gas at `x` feels the bare
/// point mass at radius `x + 1` and the domain covers `r` in `[1, 2]` with no singularity.
const R_OFFSET: f64 = 1.0;
/// bare point particle: with the mass off the domain the potential needs no regularizing,
/// and `phi = -GM / r` exactly is what the analytic profile below is built against.
const SOFTENING: f64 = 0.0;
/// long enough that a per-step residue has many thousands of steps to accumulate over,
/// short enough that the sweep stays a unit test.
const T_HYDRO: f64 = 2.0;

/// density of the isentropic atmosphere in hydrostatic balance against `GM`.
///
/// `dp/dr = -rho GM / r^2` with `p = K0 rho^gamma` integrates to the bernoulli invariant
/// `gamma K0 / (gamma - 1) rho^(gamma-1) - GM / r = const`, so
///
///     rho(r) = [ (gamma - 1) / (gamma K0) * (GM / r + C) ]^(1/(gamma-1)),
///
/// with `C` fixed by `rho = RHO0` at the outer edge `r = 1 + R_OFFSET`.
fn hydrostatic_density(r: f64, gm: f64) -> f64 {
    let k0 = k0();
    let a = (GAMMA - 1.0) / (GAMMA * k0);
    let c = 1.0 / a * RHO0.powf(GAMMA - 1.0) - gm / (1.0 + R_OFFSET);
    (a * (gm / r + c)).powf(1.0 / (GAMMA - 1.0))
}

fn hydrostatic_atmosphere_full(gm: f64, cells: usize, cfl: f64, ts: Timestepping, with_body: bool) -> Sim {
    let ic = move |x: [f64; 1]| {
        let rho = hydrostatic_density(x[0] + R_OFFSET, gm);
        Prim {
            rho,
            vel: Tensor::new([0.0]),
            pre: k0() * rho.powf(GAMMA),
        }
    };
    let sim = Sim::build(Newtonian, IdealGas { gamma: GAMMA }, Cartesian)
        .cells([cells])
        .spacing([1.0 / cells as f64])
        // a reflecting wall exerts no work on gas at rest, so the exact hydrostatic state is
        // a fixed point of the boundary as well as of the interior. an outflow edge would
        // instead drain the stratified column and manufacture a real flow.
        .boundaries(Boundaries::uniform(BoundaryType::Reflect))
        .cfl(cfl)
        .timestepping(ts)
        .allocate()
        .expect("sim construction failed")
        .set_initial(ic)
        .build();
    if with_body {
        sim.with_bodies(BodyCollection::new().add(Body::gravitational(
            0,
            Tensor::new([-R_OFFSET]),
            Tensor::zeros(),
            gm,
            1.0e-3,
            SOFTENING,
        )))
    } else {
        sim
    }
}

fn hydrostatic_atmosphere_ts(gm: f64, cells: usize, cfl: f64, ts: Timestepping) -> Sim {
    hydrostatic_atmosphere_full(gm, cells, cfl, ts, true)
}

#[test]
fn a_hydrostatic_atmosphere_holds_its_entropy_at_every_field_strength() {
    // the exact solution is static at every `GM`: the pressure gradient balances gravity by
    // construction and the state is isentropic, so `K` is `K0` everywhere for all time. the
    // ONLY thing that varies across the sweep is the field strength, so a deficit that grows
    // with `GM` is a source residue proportional to the local acceleration -- the term that a
    // single weak-field case cannot see.
    let mut worsts = Vec::new();
    for gm in [0.1, 1.0, 10.0, 100.0] {
        let mut sim = hydrostatic_atmosphere_ts(gm, N, CFL, Timestepping::Rk2);
        let kernels = Kset::new(GAMMA, CFL, &sim.geom.allocated);
        evolve(&mut sim, &kernels, T_HYDRO).expect("evolve failed");

        let vel = sim.fields.prim.vel[0].view();
        let mut vmax = 0.0_f64;
        for c in sim.geom.interior.iter() {
            vmax = vmax.max(vel.at(c).abs());
        }
        let worst = worst_entropy_ratio(&sim);
        let contrast = hydrostatic_density(1.0, gm) / hydrostatic_density(1.0 + R_OFFSET, gm);
        println!(
            "GM = {gm:>6}: min K/K0 = {worst:.12}  (deficit {:.2e}, {} steps, \
             rho contrast {contrast:.3}, max|v| = {vmax:.3e})",
            (1.0 - worst).max(0.0),
            sim.iteration
        );
        worsts.push((gm, worst, contrast, sim.iteration));
    }

    // NON-VACUITY: the strong-field cases have to be genuinely stratified and genuinely
    // long-running. a flat atmosphere would make the balance trivial, and a run that barely
    // stepped could not accumulate a per-step residue whatever its size.
    let (gm_max, _, contrast_max, steps_max) = *worsts.last().unwrap();
    assert!(
        contrast_max > 10.0,
        "at GM = {gm_max} the atmosphere is nearly uniform (density contrast \
         {contrast_max:.3}); gravity is doing no compressive work and the law is vacuous"
    );
    assert!(
        steps_max > 1000,
        "the strongest field ran only {steps_max} steps, too few for a per-step residue to \
         show above round-off"
    );

    for (gm, worst, _, steps) in &worsts {
        assert!(
            *worst > 1.0 - 1.0e-9,
            "the gravitational source destroyed entropy at GM = {gm}: min K/K0 = \
             {worst:.12} after {steps} steps, on an isentropic atmosphere in exact \
             hydrostatic balance whose exact evolution is to stand still. mass and total \
             energy can be perfectly conserved while this drifts -- the residue of an \
             inconsistent `S_nrg = rho a . v` against the momentum kick lands in the \
             internal energy, one-signed, every step, and scales with the local acceleration"
        );
    }
}

#[test]
fn gravity_free_gas_holds_its_entropy_exactly() {
    // the yardstick. with no source at all a uniform state is a fixed point of the whole
    // scheme, so anything other than an exact K here would mean the measurement itself is
    // suspect and the gravity result below could not be attributed.
    let (worst, steps, vmax) = run(false);
    println!("no gravity: min K/K0 = {worst:.12}  ({steps} steps, max|v| = {vmax:.3e})");
    assert!(steps > 10, "the control barely stepped ({steps})");
    assert!(
        (worst - 1.0).abs() < 1.0e-12,
        "a uniform state with no sources drifted: min K/K0 = {worst:.12}. the measurement \
         is unreliable, so the gravity comparison below means nothing"
    );
}

#[test]
fn a_gravitational_source_does_not_destroy_entropy() {
    let (worst, steps, vmax) = run(true);
    println!("gravity:    min K/K0 = {worst:.12}  ({steps} steps, max|v| = {vmax:.3e})");

    // NON-VACUITY: gravity has to have actually done work, or a preserved K says nothing.
    // the field is gentle by construction, so this only asserts the gas really moved.
    assert!(
        vmax > 1.0e-2,
        "gravity accelerated the gas to only max|v| = {vmax:.3e}; the source did almost no \
         work and this law is vacuous"
    );

    assert!(
        worst > 1.0 - 1.0e-9,
        "the gravitational source destroyed entropy: min K/K0 = {worst:.12} after {steps} \
         steps, on a uniform medium in a near-uniform field whose exact evolution is a \
         translating uniform state. mass and total energy can still be perfectly conserved \
         while this drifts -- the residue of an inconsistent `S_nrg = rho a . v` against \
         the momentum kick lands in the internal energy, one-signed, every step"
    );
}
