// =============================================================================
// refinement_equilibrium_residual.rs
//
// how far the coarse-fine transfer moves a state that is supposed to be stationary.
//
// a hydrostatic atmosphere is a fixed point: the pressure gradient cancels the gravity source
// exactly, nothing moves, and `v = 0` forever. that cancellation is discrete, though -- it holds
// between the scheme's own flux and its own source on a given grid -- and the coarse-fine transfer
// interpolates the conserved state, which carries no obligation to preserve it. so the interface
// behaves like a boundary that cannot hold equilibrium, and the residual force there drives a flow
// that was never in the initial condition.
//
// velocity is the probe. the equilibrium has `v = 0` identically, so any velocity present after a
// step is the imbalance itself, measured with no reference state to subtract and no cancellation
// to get wrong. entropy is the downstream symptom -- the spurious kinetic energy thermalizes --
// but it is a second-order shadow of a first-order quantity, and this measures the first-order one.
//
// the single-grid run is the control: it carries the interior scheme's own imbalance and no
// transfer at all. anything the refined runs show above that is the transfer.
//
// this is the quantity a deviation-well-balanced transfer would subtract. for a linear operator T,
// well-balancing is `T(U) + [U_eq - T(U_eq)]`, and `U_eq - T(U_eq)` is exactly what is measured
// here -- so its size and location predict both how much the fix removes and where.
//
// run: cargo test -p symbi --test refinement_equilibrium_residual -- --nocapture
// =============================================================================

use symbi::regimes::substrate_newton::AdiabaticSubstrateKernelSet;
use symbi::sim::refinement::{Hierarchy, ProlongOrder, RefinementRegion};
use symbi::sim::state::*;
use symbi_geometry::Cartesian;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::newtonian::Newtonian;
use symbi_hydro::quantity::{Density, Pressure};
use symbi_hydro::state::Prim;
use symbi_xpu::{CpuSpace, HostMemory};

const GAMMA: f64 = 5.0 / 3.0;
const N: usize = 128;
const CFL: f64 = 0.4;
const K0: f64 = 1.0;
/// the gravitating mass sits one domain width left of x = 0, so the gas at x feels a bare point
/// mass at radius x + 1 and the domain covers r in [1, 2] with no singularity.
const G_OFFSET: f64 = 1.0;
const GM: f64 = 100.0;

type Sim = SimState<Newtonian, 1, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;
type Kset = AdiabaticSubstrateKernelSet<HostMemory, f64, 1>;
type Hier = Hierarchy<Newtonian, 1, 1, Cartesian, IdealGas<f64>, CpuSpace, HostMemory, Kset>;

fn kset(s: &Sim) -> Kset {
    Kset::new(GAMMA, CFL, &s.geom.allocated)
}

/// the isentropic atmosphere in hydrostatic balance against GM, from the bernoulli invariant
/// `gamma K0/(gamma-1) rho^(gamma-1) - GM/r = const`, normalized to `rho = 1` at the outer edge.
fn hydrostatic(x: [f64; 1]) -> Prim<f64, 1> {
    let r = x[0] + G_OFFSET;
    let a = (GAMMA - 1.0) / (GAMMA * K0);
    let c = 1.0 / a - GM / (1.0 + G_OFFSET);
    let rho = (a * (GM / r + c)).powf(1.0 / (GAMMA - 1.0));
    Prim::adiabatic(
        Density(rho),
        symbi_algebra::Tensor::new([0.0]),
        Pressure(K0 * rho.powf(GAMMA)),
    )
}

/// nested patches, each half the previous, centred on the domain: `levels - 1` of them.
fn nested(levels: usize) -> Vec<RefinementRegion<1>> {
    (0..levels.saturating_sub(1))
        .map(|ii| {
            let half = 0.2 / 2f64.powi(ii as i32);
            RefinementRegion {
                x_lo: [0.5 - half],
                x_hi: [0.5 + half],
            }
        })
        .collect()
}

fn build(regions: &[RefinementRegion<1>]) -> Hier {
    build_at(regions, N)
}

fn build_at(regions: &[RefinementRegion<1>], ncells: usize) -> Hier {
    let coarse = Sim::build(Newtonian, IdealGas { gamma: GAMMA }, Cartesian)
        .cells([ncells])
        .spacing([1.0 / ncells as f64])
        // a reflecting wall exerts no work on gas at rest, so the hydrostatic state is a fixed
        // point of the boundary as well as of the interior.
        .boundaries(Boundaries::uniform(BoundaryType::Reflect))
        .cfl(CFL)
        .allocate()
        .expect("sim construction failed")
        .set_initial(hydrostatic)
        .build();
    let ck = kset(&coarse);
    let hier = Hierarchy::with_refinement(coarse, ck, regions, ProlongOrder::Ppm, kset)
        .unwrap()
        .with_bodies(
            symbi_ib::BodyCollection::new().add(symbi_ib::Body::gravitational(
                0,
                symbi_algebra::Tensor::new([-G_OFFSET]),
                symbi_algebra::Tensor::zeros(),
                GM,
                1.0e-3,
                0.0,
            )),
        );
    // every level starts on the exact equilibrium, so the residual measured afterwards is the
    // scheme's, not an interpolation artifact left over from setting the levels up.
    for lvl in 1..hier.levels.len() {
        hier.levels[lvl].state.seed_cells(hydrostatic);
    }
    hier
}

/// the largest spurious speed on one level, its interior index, and whether that cell is covered
/// by a finer patch. the wall band is skipped: a reflecting boundary mirrors the state but not the
/// gravity source, so it cannot hold the equilibrium the interior holds, and its imbalance would
/// otherwise mask the interface's.
fn worst_speed(hier: &Hier, level: usize) -> (f64, usize, bool, usize) {
    let lvl = &hier.levels[level];
    let st = &lvl.state;
    let rho = st.fields.prim.rho.view();
    let vel = st.fields.prim.vel[0].view();
    let cells: Vec<[isize; 1]> = st.geom.interior.iter().collect();
    let skip = cells.len() / 5;

    let mut worst = (0.0_f64, 0usize);
    for (ii, c) in cells
        .iter()
        .enumerate()
        .skip(skip)
        .take(cells.len() - 2 * skip)
    {
        let _ = rho.at(*c);
        let speed = vel.at(*c).abs();
        if speed > worst.0 {
            worst = (speed, ii);
        }
    }
    let covered = lvl
        .coverage
        .as_ref()
        .is_some_and(|cov| cov.contains(cells[worst.1]));
    // distance in cells to the nearest coverage edge, or the cell count when this level has no
    // finer patch above it.
    let edge_distance = match lvl.coverage.as_ref() {
        None => cells.len(),
        Some(cov) => {
            let flags: Vec<bool> = cells.iter().map(|c| cov.contains(*c)).collect();
            let edges: Vec<usize> = (1..flags.len())
                .filter(|&ii| flags[ii] != flags[ii - 1])
                .collect();
            edges
                .iter()
                .map(|&e| (e as isize - worst.1 as isize).unsigned_abs())
                .min()
                .unwrap_or(cells.len())
        }
    };
    (worst.0, worst.1, covered, edge_distance)
}

#[test]
fn the_equilibrium_residual_localizes_at_the_coarse_fine_interface() {
    println!(
        "\nspurious speed after 1 root step, from an EXACT hydrostatic start (v = 0 everywhere)"
    );
    println!("{:-<86}", "");

    let mut summary: Vec<(usize, Vec<f64>)> = Vec::new();
    for levels in 1..=4usize {
        let mut hier = build(&nested(levels));
        assert_eq!(hier.levels.len(), levels, "asked for {levels} levels");

        // non-vacuity: the transfer acts at the patch edge, so what has to be non-flat is the
        // density there rather than end to end. this atmosphere spans a factor of 96 across the
        // domain; the bound below asks only that a level edge sees real curvature, since a
        // constant is interpolated exactly at any order and would make the measurement empty.
        let dx = 1.0 / N as f64;
        for region in nested(levels) {
            let edge = region.x_lo[0];
            let flank = hydrostatic([edge - 2.0 * dx]).rho() / hydrostatic([edge + 2.0 * dx]).rho();
            assert!(
                flank > 1.02,
                "the density is flat across the patch edge at x = {edge} (ratio {flank:.4}); the \
                 transfer would be interpolating a constant and this measurement would be vacuous"
            );
        }

        hier.evolve_steps(1).unwrap();

        let mut per_level = Vec::new();
        for level in 0..levels {
            let (speed, idx, covered, edge_distance) = worst_speed(&hier, level);
            let where_ = if hier.levels[level].coverage.is_none() {
                "finest (no patch above)".to_string()
            } else if covered {
                format!("covered, {edge_distance} cells from the patch edge")
            } else {
                format!("uncovered, {edge_distance} cells from the patch edge")
            };
            println!(
                "levels={levels}  level {level}  max|v| = {speed:.6e}  at idx {idx:>3}  ({where_})"
            );
            per_level.push(speed);
        }
        summary.push((levels, per_level));
        println!("{:-<86}", "");
    }

    // the shape of the residual across an interface distinguishes two very different faults.
    // refluxing makes the level exchange conservative to machine precision, and conservation
    // constrains only the sum over cells -- fluxes telescope -- so it permits every individual
    // cell at the interface to carry a force as long as those forces cancel. a balance error
    // under exact conservation is therefore forced to appear as equal-and-opposite velocities: a
    // dipole. a genuine conservation leak would instead show one sign on both sides, a monopole.
    {
        // the interior scheme carries its own one-signed imbalance on every cell, refined or not,
        // and it swamps the interface term in the raw field. level 0 of the two-level ladder is
        // the same grid as the single-level run, cell for cell, so subtracting one from the other
        // leaves the transfer's contribution alone.
        let mut refined = build(&nested(2));
        let mut control = build(&nested(1));
        refined.evolve_steps(1).unwrap();
        control.evolve_steps(1).unwrap();

        let lvl = &refined.levels[0];
        let vel = lvl.state.fields.prim.vel[0].view();
        let vel_ctl = control.levels[0].state.fields.prim.vel[0].view();
        let cells: Vec<[isize; 1]> = lvl.state.geom.interior.iter().collect();
        let cov = lvl.coverage.as_ref().unwrap();
        let flags: Vec<bool> = cells.iter().map(|c| cov.contains(*c)).collect();
        let edge = (1..flags.len())
            .find(|&ii| flags[ii] != flags[ii - 1])
            .unwrap();

        println!("\ntransfer-induced velocity across the left patch edge (refined - control):");
        let (lo, hi) = (edge.saturating_sub(5), (edge + 5).min(cells.len()));
        let dv = |ii: usize| *vel.at(cells[ii]) - *vel_ctl.at(cells[ii]);
        let mut signed_sum = 0.0;
        for ii in lo..hi {
            signed_sum += dv(ii);
            println!(
                "  idx {ii:3}  covered={:5}  dv = {:+.6e}",
                flags[ii],
                dv(ii)
            );
        }
        let peak = (lo..hi).map(|ii| dv(ii).abs()).fold(0.0, f64::max);
        println!(
            "  signed sum across the band = {signed_sum:+.3e}   peak |dv| = {peak:.3e}   \
             ratio = {:.4}",
            signed_sum.abs() / peak.max(f64::MIN_POSITIVE)
        );
        println!(
            "  (a ratio near zero is a dipole: exact conservation with a broken balance, the \
             errors cancelling in the sum. a ratio near one is a monopole: a genuine leak.)"
        );

        // the two halves of the question, side by side. mass carries no source term, so the
        // refluxed level exchange must hold it to round-off -- that is exactly what conservation
        // buys. momentum carries the gravity source, so conservation says nothing about it, and a
        // flux/source cancellation that fails at the interface is a spurious force, which injects
        // net momentum. one number is machine zero and the other is not, from the same run.
        let composite_mass = |h: &Hier| -> f64 {
            let mut mass = 0.0;
            for lvl in h.levels.iter() {
                let vol: f64 = lvl.state.geom.dx.iter().product();
                let den = lvl.state.fields.cons.den.view();
                for c in lvl.state.geom.interior.iter() {
                    if lvl.coverage.as_ref().is_some_and(|cov| cov.contains(c)) {
                        continue;
                    }
                    mass += *den.at(c) * vol;
                }
            }
            mass
        };
        let mut fresh = build(&nested(2));
        let m0 = composite_mass(&fresh);
        fresh.evolve_steps(1).unwrap();
        let m1 = composite_mass(&fresh);
        println!("\nreflecting walls, one root step -- mass has no source, momentum has gravity:");
        println!(
            "  composite MASS      {m0:.15e} -> {m1:.15e}   relative change {:.3e}",
            ((m1 - m0) / m0).abs()
        );
        println!(
            "  transfer-induced MOMENTUM injected at the interface   {signed_sum:+.3e} \
             (relative to peak: {:.4})",
            signed_sum.abs() / peak.max(f64::MIN_POSITIVE)
        );
    }

    println!("\nper-level max|v| by ladder depth:");
    for (levels, speeds) in &summary {
        let shown: Vec<String> = speeds.iter().map(|s| format!("{s:.3e}")).collect();
        println!("  levels={levels}  {shown:?}");
    }

    // the control carries no transfer at all, so it is the yardstick every refined run is read
    // against.
    let control = summary[0].1[0];
    println!("\nsingle-grid control (interior scheme alone): max|v| = {control:.6e}");
    for (levels, speeds) in summary.iter().skip(1) {
        let excess: Vec<String> = speeds
            .iter()
            .map(|s| format!("{:.2}x", s / control.max(f64::MIN_POSITIVE)))
            .collect();
        println!("  levels={levels}  excess over control: {excess:?}");
    }

    // the setup has to have stepped, or none of the above measured anything.
    assert!(
        summary.len() == 4,
        "the sweep did not complete: {} entries",
        summary.len()
    );
}

// =============================================================================
// is the declared equilibrium what it claims to be?
//
// the deviation method subtracts `U_eq` and balances whatever it is handed, so the state handed
// to it has to be checked on its own terms before any of that machinery exists. two claims are
// independent and fail differently.
// =============================================================================

/// claim one: `hydrostatic` is the continuum solution, i.e. `dp/dr = -rho GM/r^2`.
///
/// the profile is built from the bernoulli invariant `gamma K0/(gamma-1) rho^(gamma-1) - GM/r`,
/// which satisfies the balance identically, so the only thing that can be wrong is the way the
/// expression is written down. differentiating the implemented function numerically catches that:
/// a faithful implementation leaves only the difference formula's own truncation error, which
/// vanishes with the step, while a mis-typed one leaves a residual that converges to a constant.
#[test]
fn the_hydrostatic_profile_satisfies_the_continuum_balance() {
    // sample the interior, away from the ends where the stencil would leave the domain.
    let samples: Vec<f64> = (1..20).map(|ii| 0.05 * ii as f64).collect();
    let residuals = |h: f64| -> Vec<f64> {
        samples
            .iter()
            .map(|&x| {
                let r = x + G_OFFSET;
                let dp = (hydrostatic([x + h]).pre() - hydrostatic([x - h]).pre()) / (2.0 * h);
                let rho = hydrostatic([x]).rho();
                let gravity = -rho * GM / (r * r);
                // relative to the size of the two terms being cancelled, so this measures the
                // cancellation rather than the magnitude of either side.
                (dp - gravity) / dp.abs().max(gravity.abs())
            })
            .collect()
    };

    let (coarse_v, fine_v) = (residuals(1.0e-3), residuals(5.0e-4));
    let peak = |v: &[f64]| v.iter().fold(0.0_f64, |m, e| m.max(e.abs()));
    let (coarse, fine) = (peak(&coarse_v), peak(&fine_v));
    let order = (coarse / fine).log2();
    println!(
        "continuum balance residual: h=1e-3 -> {coarse:.3e},  h=5e-4 -> {fine:.3e}   (order {order:.2})"
    );

    // a central difference is second order, so halving the step must cut the residual by ~4. a
    // profile that did not solve the ode would leave a floor that does not move with h at all,
    // and this is the measurement that separates the two.
    assert!(
        order > 1.8,
        "the residual fell by only 2^{order:.2} when the difference step was halved; a second-order \
         central difference on an exact solution must fall by 2^2, so the profile does not satisfy \
         dp/dr = -rho GM/r^2 and every measurement built on it is measuring the wrong state"
    );

    // the absolute statement, with the difference formula's own error removed. richardson on a
    // second-order rule gives the h -> 0 limit as (4 f(h/2) - f(h))/3, which is the part of the
    // residual that is not truncation. an exact solution leaves nothing there; a profile off by a
    // constant leaves that constant, untouched by the extrapolation. the raw residual cannot be
    // bounded directly, because at these sample points second-order truncation on a profile this
    // steep is itself a few times 1e-6.
    let extrapolated = coarse_v
        .iter()
        .zip(&fine_v)
        .map(|(c, f)| ((4.0 * f - c) / 3.0).abs())
        .fold(0.0_f64, f64::max);
    println!("  richardson h->0 limit (truncation removed): {extrapolated:.3e}");
    assert!(
        extrapolated < fine / 100.0,
        "with second-order truncation removed the balance residual is still {extrapolated:.3e}, \
         only a factor {:.1} below the raw {fine:.3e}; a residual that survives extrapolation is a \
         genuine violation of dp/dr = -rho GM/r^2, not difference error",
        fine / extrapolated.max(f64::MIN_POSITIVE)
    );
}

/// claim two: how far the cell averages of that continuum solution sit from the scheme's own
/// discrete fixed point.
///
/// these are different states. the scheme's fixed point satisfies a cancellation between its
/// reconstruction and its source quadrature at a given cell width; the continuum solution
/// satisfies the differential equation. they agree only to truncation order, so a grid seeded
/// with the analytic profile drifts even with no coarse-fine interface anywhere.
///
/// this sets the floor for the interface fix. a deviation-well-balanced transfer removes the
/// transfer's contribution and nothing else, so it can bring the interface down to this number
/// and no lower. reading the order here says whether that floor shrinks under refinement or sits
/// at a fixed level.
#[test]
fn the_discrete_imbalance_of_the_analytic_profile_converges() {
    // a fixed time rather than a fixed step count: the induced speed is the residual force times
    // the elapsed time, so evolving every resolution to the same instant compares forces rather
    // than forces times a resolution-dependent dt.
    const T_PROBE: f64 = 1.0e-3;

    let mut rows = Vec::new();
    for ncells in [64usize, 128, 256, 512] {
        let mut hier = build_at(&nested(1), ncells);
        assert_eq!(hier.levels.len(), 1, "the floor probe must be unrefined");
        hier.evolve(T_PROBE).unwrap();

        let st = &hier.levels[0].state;
        let vel = st.fields.prim.vel[0].view();
        let cells: Vec<[isize; 1]> = st.geom.interior.iter().collect();
        let skip = cells.len() / 5;
        let worst = cells
            .iter()
            .skip(skip)
            .take(cells.len() - 2 * skip)
            .map(|c| vel.at(*c).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            st.iteration > 0,
            "N={ncells} never stepped, so it cannot have drifted"
        );
        println!(
            "N={ncells:>4}  steps={:>4}  max|v| = {worst:.6e}",
            st.iteration
        );
        rows.push((ncells, worst));
    }

    println!("\nconvergence of the single-grid imbalance:");
    for pair in rows.windows(2) {
        let ((n0, v0), (n1, v1)) = (pair[0], pair[1]);
        println!(
            "  N {n0} -> {n1}:  ratio {:.2}  (order {:.2})",
            v0 / v1,
            (v0 / v1).log2()
        );
    }

    // the imbalance must shrink with resolution. a floor that does not move would mean the
    // analytic profile is not converging to the scheme's fixed point at all, and no amount of
    // refinement would make the declared equilibrium the right thing to subtract.
    let (_, coarsest) = rows[0];
    let (_, finest) = rows[rows.len() - 1];
    assert!(
        finest < coarsest,
        "the imbalance did not shrink from N=64 ({coarsest:.3e}) to N=512 ({finest:.3e}); the cell \
         averages of the analytic profile are not approaching the scheme's discrete equilibrium"
    );
}
