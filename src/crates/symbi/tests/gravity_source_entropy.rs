// =============================================================================
// gravity_source_entropy.rs
//
// a gravitational source must not change the entropy of the gas it accelerates.
//
// gravity does work: it adds momentum `S_mom = rho a` and energy `S_nrg = rho a . v`.
// those two have to agree with each other to machine precision, because the internal
// energy is what is left after the kinetic part is removed. if the energy term uses a
// velocity that is even slightly out of step with the one the momentum term produces --
// a stale stage value, say -- the residue lands in the internal energy as `rho a dv dt`,
// one-signed, every step. that is invisible in the conserved totals (both mass and total
// energy stay perfectly conserved) and shows up only as entropy drift.
//
// the setup makes the exact answer trivial. a uniform medium at rest in a uniform
// gravitational field is accelerated identically everywhere, so it never compresses: the
// exact solution is the same uniform state translating, with `K = p / rho^gamma` constant
// for all time. uniformity is obtained by putting the mass very far away, so the field
// varies by ~0.2 percent across the domain -- and even that residue only drives adiabatic
// compression, which preserves K exactly.
//
// so any drift here is the source term, isolated from reconstruction, the riemann solver,
// refinement transfer, sinks, and boundaries.
//
// the uniform case pins the source's self-consistency but exercises it at only one field
// strength, and a residue proportional to the local acceleration is invisible there. the
// second setup below supplies the missing dynamic range: an isentropic atmosphere in
// hydrostatic balance against a point mass, whose exact solution is that nothing moves and
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
use symbi::regimes::substrate_kernels::FusedSourceBinding;
use symbi_ib::{Body, BodyCollection};
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

/// the worst `K / K0` over the interior with a fraction `skip` of the cells excluded at each
/// wall.
///
/// a reflecting boundary mirrors the conserved state but does not mirror the gravitational
/// source, so it cannot hold the hydrostatic equilibrium the interior holds: the wall drives a
/// slow flow the exact solution does not have, and that flow is a genuine feature of the
/// boundary condition rather than of the scheme under test. measuring across it would attribute
/// the wall's defect to the source term.
///
/// the exclusion is not a tolerance. the entropy error after one step is exactly `K = K0` to
/// every digit (`the_discrete_balance_is_exact_on_the_equilibrium_profile`), and over a long run
/// the deficit localizes against the outer wall and recovers monotonically as the wall is
/// excluded -- at `GM = 100`, `N = 512`: 0.994686 at 0.4 percent excluded, 0.995456 at 5
/// percent, 1.000690 at 10 percent and beyond. a defect of the interior discretization would
/// not clear like that.
fn worst_entropy_ratio_away_from_walls(sim: &Sim, skip_frac: f64) -> f64 {
    let rho = sim.fields.prim.rho.view();
    let pre = sim
        .fields
        .prim
        .pre
        .as_ref()
        .expect("adiabatic carries pressure")
        .view();
    let cells: Vec<_> = sim.geom.interior.iter().collect();
    let skip = (skip_frac * cells.len() as f64) as usize;
    let mut worst = f64::INFINITY;
    for c in cells.iter().skip(skip).take(cells.len() - 2 * skip) {
        let r = *rho.at(*c);
        if r > 0.0 {
            worst = worst.min(*pre.at(*c) / r.powf(GAMMA) / k0());
        }
    }
    worst
}

/// the fraction of the domain excluded at each wall. one fifth is well outside the boundary
/// layer measured above and still leaves three fifths of the atmosphere, spanning the great
/// majority of its density contrast, under test.
const WALL_SKIP: f64 = 0.2;

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
    // only thing that varies across the sweep is the field strength, so a deficit that grows
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
        let worst = worst_entropy_ratio_away_from_walls(&sim, WALL_SKIP);
        let contrast = hydrostatic_density(1.0, gm) / hydrostatic_density(1.0 + R_OFFSET, gm);
        println!(
            "GM = {gm:>6}: min K/K0 = {worst:.12}  (deficit {:.2e}, {} steps, \
             rho contrast {contrast:.3}, max|v| = {vmax:.3e})",
            (1.0 - worst).max(0.0),
            sim.iteration
        );
        worsts.push((gm, worst, contrast, sim.iteration));
    }

    // non-vacuity: the strong-field cases have to be genuinely stratified and genuinely
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
             energy can be perfectly conserved while this drifts, so the conserved totals \
             will not show it. what remains here is TIMESTEP-INDEPENDENT and identical \
             through the immersed-body and the additive source paths (see \
             `the_body_and_additive_sources_agree_and_carry_no_timestep_bias`), which \
             excludes the source-composition bias and points at the discrete hydrostatic \
             balance: the finite-volume pressure gradient and the cell-centered gravity \
             source cancel only to truncation order, and the residual drives a flow that \
             the exact solution does not have"
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

    // non-vacuity: gravity has to have actually done work, or a preserved K says nothing.
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

/// the immersed-body source and the additive source deliver the same gravity, so they must
/// produce the same state — and neither may carry a timestep-dependent entropy bias.
///
/// a source operator applied on top of an already-flux-advanced state composes sequentially with
/// the flux, which is first order in `dt` however accurately either half is integrated, and leaks
/// internal energy one-signed every stage. raising the Runge-Kutta order does not help, because
/// every stage repeats the same composition — so the signature is a deficit that tracks `dt` at
/// fixed cell width and is common to every integrator. evaluating both the flux and the source at
/// the stage input removes it.
///
/// the two clauses are independent. equality alone would pass if both paths shared the bias;
/// timestep-independence alone would pass if they were independently clean but disagreed.
#[test]
fn the_body_and_additive_sources_agree_and_carry_no_timestep_bias() {
    const CFLS: [f64; 3] = [0.4, 0.2, 0.1];
    for gm in [1.0, 10.0, 100.0] {
        let mut body_k = Vec::new();
        for cfl in CFLS {
            let mut body = hydrostatic_atmosphere_ts(gm, N, cfl, Timestepping::Rk2);
            let kb = Kset::new(GAMMA, cfl, &body.geom.allocated);
            evolve(&mut body, &kb, T_HYDRO).expect("body evolve failed");

            let mut add = hydrostatic_atmosphere_full(gm, N, cfl, Timestepping::Rk2, false);
            let ka = Kset::new(GAMMA, cfl, &add.geom.allocated).with_additive_source(
                FusedSourceBinding::new(
                    "point_mass_grav",
                    &[("gm", gm), ("xm_0", -R_OFFSET), ("eps", SOFTENING)],
                ),
            );
            evolve(&mut add, &ka, T_HYDRO).expect("additive evolve failed");

            let (kb_min, ka_min) = (worst_entropy_ratio(&body), worst_entropy_ratio(&add));
            println!("GM = {gm:>6}  cfl = {cfl:>5}:  body {kb_min:.9}  additive {ka_min:.9}");

            // non-vacuity: gravity has to be doing enough work for either clause to bite. at
            // GM = 1 the atmosphere is already 1.5x denser at the inner edge than the outer.
            let contrast = hydrostatic_density(1.0, gm) / hydrostatic_density(1.0 + R_OFFSET, gm);
            assert!(
                contrast > 1.4,
                "at GM = {gm} the atmosphere is nearly uniform (contrast {contrast:.3}); \
                 gravity does no compressive work and both clauses are vacuous"
            );
            assert!(
                (kb_min - ka_min).abs() < 1.0e-12,
                "the same gravity gave different answers through the two source paths at \
                 GM = {gm}, cfl = {cfl}: body {kb_min:.12}, additive {ka_min:.12}. both \
                 evaluate the body contribution at the stage input and apply it to the \
                 flux-advanced state, so they are the same discrete operator"
            );
            body_k.push(kb_min);
        }
        // the timestep clause. a sequential composition leaks in proportion to `dt`, so a 4x
        // change in `dt` would move this by a comparable factor; a purely spatial residual does
        // not move at all. the bound is far tighter than the bias it excludes -- before the
        // stage-input evaluation this spread was 8.8e-2 to 1.1e-2 at GM = 100.
        let spread = body_k
            .iter()
            .fold(0.0_f64, |a, k| a.max((k - body_k[0]).abs()));
        assert!(
            spread < 1.0e-5,
            "the entropy floor moved by {spread:.3e} across a 4x timestep change at GM = {gm} \
             ({body_k:?}). the body source is composing with the flux update rather than being \
             evaluated alongside it, and the residue accumulates one-signed every stage"
        );
    }
}

#[test]
fn the_hydrostatic_residue_converges_at_third_order() {
    // how well the discrete flux gradient cancels the discrete gravity source, measured on the
    // analytic equilibrium whose exact evolution is to stand still.
    //
    // this is not exactness. a genuinely well-balanced scheme leaves the discrete equilibrium at
    // round-off whatever the resolution; this one leaves a residue that shrinks with `dx`, so the
    // balance is a convergence property rather than an identity. holding the step count fixed and
    // varying `dx` is what separates the two: flat-at-round-off would mean balanced, and a clean
    // power means merely consistent.
    //
    // a `tend` at or below one CFL step is useless here — `evolve` clamps `dt` to the remaining
    // time, so it takes one degenerate step of that size and measures nothing. the end time below
    // scales with `dx` precisely so every resolution takes the same number of real steps.
    const STEPS: f64 = 20.0;
    for gm in [1.0, 100.0] {
        let mut residue = Vec::new();
        for n in [128usize, 256, 512] {
            let mut sim = hydrostatic_atmosphere_ts(gm, n, CFL, Timestepping::Rk2);
            let k = Kset::new(GAMMA, CFL, &sim.geom.allocated);
            evolve(&mut sim, &k, STEPS * CFL / (n as f64 * 6.0)).expect("evolve failed");

            let rho = sim.fields.prim.rho.view();
            let pre = sim.fields.prim.pre.as_ref().expect("adiabatic").view();
            let cells: Vec<_> = sim.geom.interior.iter().collect();
            let skip = (WALL_SKIP * cells.len() as f64) as usize;
            let mut worst = 0.0_f64;
            for c in cells.iter().skip(skip).take(cells.len() - 2 * skip) {
                let r = *rho.at(*c);
                worst = worst.max((*pre.at(*c) / r.powf(GAMMA) / k0() - 1.0).abs());
            }
            println!("GM = {gm:>6}  N = {n:>4}: {:>3} steps  |K/K0 - 1| = {worst:.4e}", sim.iteration);
            // non-vacuity: a clamped `dt` would make this measure nothing at all.
            assert!(
                sim.iteration >= 4,
                "only {} step(s) at N = {n}; the end time collapsed onto the timestep and this \
                 measures nothing",
                sim.iteration
            );
            assert!(
                worst > 1.0e-13,
                "the residue at N = {n} is at round-off ({worst:.3e}); the ratio below carries \
                 no information"
            );
            residue.push(worst);
        }
        let orders: Vec<f64> = residue
            .windows(2)
            .map(|w| (w[0] / w[1]).log2())
            .collect();
        println!("  observed orders: {orders:?}");
        for p in &orders {
            assert!(
                *p > 2.5,
                "the hydrostatic residue converges at order {p:.3} at GM = {gm} \
                 ({residue:?}); the balance between the flux gradient and the gravity source is \
                 degrading faster than the scheme's own accuracy"
            );
        }
    }
}


/// where in the column the entropy residue actually sits.
///
/// the sweep above measures a single worst value over the interior with the outer fifth of each
/// side removed, so a residue confined to the wall band is invisible to it. this prints the whole
/// profile: a residue at the reflecting edge is a boundary-operator property, one spread through
/// the interior is the source or the flux, and the two call for different fixes.
///
/// run: cargo test -p symbi --test gravity_source_entropy -- --ignored profile --nocapture
#[test]
#[ignore = "diagnostic: the radial location of the hydrostatic entropy residue"]
fn diagnose_entropy_residue_profile() {
    for gm in [1.0, 10.0, 100.0] {
        let mut sim = hydrostatic_atmosphere_ts(gm, N, CFL, Timestepping::Rk2);
        let kernels = Kset::new(GAMMA, CFL, &sim.geom.allocated);
        evolve(&mut sim, &kernels, T_HYDRO).expect("evolve failed");

        let rho = sim.fields.prim.rho.view();
        let pre = sim.fields.prim.pre.as_ref().expect("adiabatic").view();
        let k0 = k0();
        let cells: Vec<_> = sim.geom.interior.iter().collect();
        let ratio: Vec<f64> = cells
            .iter()
            .map(|c| (pre.at(*c) / rho.at(*c).powf(GAMMA)) / k0)
            .collect();

        let n = ratio.len();
        let worst_at = (0..n).min_by(|a, b| ratio[*a].total_cmp(&ratio[*b])).unwrap();
        println!(
            "\nGM = {gm}: {} cells, {} steps.  worst K/K0 = {:.9} at cell {} of {} \
             ({:.0}% of the way in from the near wall)",
            n, sim.iteration, ratio[worst_at], worst_at, n,
            100.0 * worst_at as f64 / n as f64
        );
        let below: Vec<usize> = (0..n).filter(|i| ratio[*i] < 1.0 - 1.0e-9).collect();
        println!(
            "   cells below K0: {} of {}   indices {:?}",
            below.len(),
            n,
            &below[..below.len().min(12)]
        );
        // the first and last twelve cells, then a coarse sample of the interior, so a wall
        // residue and an interior one are told apart at a glance.
        let show: Vec<usize> = (0..12)
            .chain((12..n - 12).step_by((n / 8).max(1)))
            .chain(n - 12..n)
            .collect();
        for i in show {
            let mark = if ratio[i] < 1.0 - 1.0e-9 { "  <== below K0" } else { "" };
            println!("     cell {i:4}  K/K0 = {:.9}{mark}", ratio[i]);
        }
    }
}


/// whether the residue accumulates or oscillates.
///
/// a source or flux defect deposits the same sign every step, so its worst value falls
/// monotonically and its location stays put. an acoustic transient launched by the initial
/// discretization instead moves, changes sign, and does not deepen: the column rings at its own
/// sound-crossing rate. the two are distinguished by sampling the same run at several times.
///
/// run: cargo test -p symbi --test gravity_source_entropy -- --ignored accumul --nocapture
#[test]
#[ignore = "diagnostic: does the hydrostatic entropy residue accumulate or ring"]
fn diagnose_entropy_residue_accumulates_or_rings() {
    for gm in [1.0, 10.0] {
        println!("\nGM = {gm}");
        let mut sim = hydrostatic_atmosphere_ts(gm, N, CFL, Timestepping::Rk2);
        let kernels = Kset::new(GAMMA, CFL, &sim.geom.allocated);
        let k0 = k0();
        for frac in 1..=8 {
            let t_end = T_HYDRO * frac as f64 / 4.0;
            evolve(&mut sim, &kernels, t_end).expect("evolve failed");
            let rho = sim.fields.prim.rho.view();
            let pre = sim.fields.prim.pre.as_ref().expect("adiabatic").view();
            let cells: Vec<_> = sim.geom.interior.iter().collect();
            let ratio: Vec<f64> = cells
                .iter()
                .map(|c| (pre.at(*c) / rho.at(*c).powf(GAMMA)) / k0)
                .collect();
            let n = ratio.len();
            let at = (0..n).min_by(|a, b| ratio[*a].total_cmp(&ratio[*b])).unwrap();
            let below = (0..n).filter(|i| ratio[*i] < 1.0 - 1.0e-9).count();
            println!(
                "   t = {:6.3}  steps {:6}  worst K/K0 = {:.9} at cell {:3}   cells below K0: {}",
                t_end, sim.iteration, ratio[at], at, below
            );
        }
    }
}

// =============================================================================
// solver-family entropy probe on the sealed column
// =============================================================================

/// the same sealed stratified column, swept over the riemann-solver family. the
/// stagnant strong-contrast column is the low-mach central-differencing limit of
/// the HLLC-LM (fleischmann) anti-diffusive scaling — the regime a sealed
/// accretor wall holds its masked cells in permanently. an entropy floor
/// violation that appears only on the low-mach arm localizes the production
/// K < K_0 deficit to the flux, not the source or the wall ledger (both hold
/// the floor on this exact setup under HLLE). production knobs: cfl 0.3, rk2.
///
/// the balanced low-mach arm's residual on this column is seeded by the last bits of the
/// equilibrium profile and grows in proportion to that seed: 4982 steps leave a deficit of
/// 1.9e-11 and max|v| of 1.1e-9, a factor 50 inside the 1e-9 floor the sweep asserts. a
/// change to the arithmetic that evaluates the isentrope moves both numbers by its own
/// factor in the last digits and leaves the bound untouched; a change that breaks the
/// balance moves them by orders.
#[test]
fn the_solver_family_holds_the_entropy_floor_on_the_sealed_column() {
    use symbi::prelude::Solver;
    let gm = 100.0;
    let mut rows = Vec::new();
    // the low-mach arm pairs with the balanced reconstruction: since the clamp's
    // retirement (2026-08-15) the published ramp leaves the hydrostatic residual undamped
    // by design, and the balancing removes it instead -- unpaired, this exact column loses
    // 4.2e-4 of the floor, which is the failure that once motivated the clamp.
    for (name, solver, balanced) in [
        ("hlle", Solver::Hlle, false),
        ("hllc", Solver::Hllc, false),
        ("hllc_lm", Solver::HllcPlus, true),
    ] {
        let mut sim = hydrostatic_atmosphere_full(gm, N, 0.3, Timestepping::Rk2, true);
        let kernels = Kset::new(GAMMA, 0.3, &sim.geom.allocated)
            .with_solver(solver)
            .expect("solver/regime mismatch")
            .well_balanced_reconstruction(balanced);
        evolve(&mut sim, &kernels, T_HYDRO).expect("evolve failed");
        let worst = worst_entropy_ratio_away_from_walls(&sim, WALL_SKIP);
        let vel = sim.fields.prim.vel[0].view();
        let mut vmax = 0.0_f64;
        for c in sim.geom.interior.iter() {
            vmax = vmax.max(vel.at(c).abs());
        }
        println!(
            "{name:>8}: min K/K0 = {worst:.12} (deficit {:.3e}), {} steps, max|v| = {vmax:.3e}",
            (1.0 - worst).max(0.0),
            sim.iteration
        );
        assert!(sim.iteration > 1000, "{name}: too few steps; the probe is vacuous");
        rows.push((name, worst));
    }
    for (name, worst) in &rows {
        assert!(
            *worst > 1.0 - 1.0e-9,
            "{name} destroyed entropy on the sealed stratified column: min K/K0 = {worst:.12} \
             — the stagnant low-mach column is the regime a sealed (porosity 0) accretor wall \
             holds its masked cells in, so a floor violation here reproduces the production \
             mask-interior deficit at 1/10^5 the cost"
        );
    }
}

/// the low-mach arm's floor law with its precondition made explicit: the sealed
/// stratified column holds `K >= K_0` under hllc_lm while the fleischmann ramp is
/// genuinely engaged — the residual velocities stay below the mach limit, so a pass
/// cannot come from the ramp being inactive. unpaired, this exact configuration loses
/// 4.2e-4 of the floor at t = 2 (5.4e-4 at half the timestep — spatial
/// anti-dissipation, not a source bias): the ramp cuts the acoustic dissipation that
/// damps the hydrostatic residual and the ringing undershoots the adiabat. the
/// balanced reconstruction removes that residual at its source — each cell's
/// departure from the isentrope through it is what gets limited, so the balanced
/// column presents no face jump and there is nothing to ring.
#[test]
fn the_low_mach_arm_holds_the_floor_with_the_correction_active() {
    use symbi::prelude::Solver;
    /// the residual mach number below which the correction has removed at least nine tenths
    /// of the classical velocity-jump dissipation: the rescaling factor is the local mach
    /// number itself, so this is a statement about how much of the damping is gone.
    const CORRECTION_ACTIVE_BELOW: f64 = 0.1;

    let gm = 100.0;
    let mut sim = hydrostatic_atmosphere_full(gm, N, 0.3, Timestepping::Rk2, true);
    let kernels = Kset::new(GAMMA, 0.3, &sim.geom.allocated)
        .with_solver(Solver::HllcPlus)
        .expect("solver/regime mismatch")
        .well_balanced_reconstruction(true);
    evolve(&mut sim, &kernels, 2.0).expect("evolve failed");

    // the precondition: the column's residual flow sits deep in the regime where the
    // correction acts, so the floor below is a statement about a scheme running with most of
    // its velocity-jump dissipation removed rather than about a correction that never engaged.
    let rho = sim.fields.prim.rho.view();
    let pre = sim.fields.prim.pre_field().expect("adiabatic pre").view();
    let vel = sim.fields.prim.vel[0].view();
    let mut max_mach = 0.0_f64;
    for c in sim.geom.interior.iter() {
        let cs = (GAMMA * *pre.at(c) / *rho.at(c)).sqrt();
        max_mach = max_mach.max(vel.at(c).abs() / cs);
    }
    assert!(
        max_mach < CORRECTION_ACTIVE_BELOW,
        "the column's residual flow reached mach {max_mach:.3}; the correction is barely \
         engaged and the floor law is vacuous"
    );

    let worst = worst_entropy_ratio_away_from_walls(&sim, WALL_SKIP);
    assert!(
        worst > 1.0 - 1.0e-9,
        "the low-mach arm destroyed entropy on the sealed column: min K/K0 = \
         {worst:.12} with max residual mach {max_mach:.4}"
    );
}
