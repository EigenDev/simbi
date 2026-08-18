// =============================================================================
// ppm_reconstruction.rs
//
// the ppm (piecewise parabolic, colella & woodward 1984 monotonized interfaces)
// evolution reconstruction: end-to-end evolution through the `_ppm` kernel twins,
// and the allocation guard. the parabola loads -3..+2 along the sweep, so a ppm
// sim allocates ng = 3 (`.ghosts(3)`); dispatch refuses an ng = 2 allocation
// before a ghost read could return garbage.
//
// run: cargo test -p symbi --test ppm_reconstruction
// =============================================================================

use symbi::prelude::Solver;
use symbi::regimes::substrate_newton::AdiabaticSubstrateKernelSet;
use symbi::sim::evolve::evolve;
use symbi::sim::refinement::{Hierarchy, ProlongOrder, RefinementRegion};
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_discretize::Recon;
use symbi_geometry::Cartesian;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::newtonian::Newtonian;
use symbi_hydro::state::Prim;
use symbi_xpu::{CpuSpace, HostMemory};

const GAMMA: f64 = 1.4;
const CFL: f64 = 0.4;

type Sim1 = SimState<Newtonian, 1, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;

fn sod_sim(ng: usize) -> Sim1 {
    const N: usize = 128;
    let dx = 1.0 / N as f64;
    Sim1::build(Newtonian, IdealGas { gamma: GAMMA }, Cartesian)
        .cells([N])
        .spacing([dx])
        .ghosts(ng)
        .boundaries(Boundaries::uniform(BoundaryType::Outflow))
        .cfl(CFL)
        .timestepping(Timestepping::Rk2)
        .allocate()
        .expect("sim construction failed")
        .set_initial(|[x]| {
            if x < 0.5 {
                Prim {
                    rho: 1.0,
                    vel: Tensor::new([0.0]),
                    pre: 1.0,
                }
            } else {
                Prim {
                    rho: 0.125,
                    vel: Tensor::new([0.0]),
                    pre: 0.1,
                }
            }
        })
        .build()
}

fn sod_density(recon: Recon, ng: usize) -> Vec<f64> {
    let mut sim = sod_sim(ng);
    let sub =
        AdiabaticSubstrateKernelSet::<HostMemory, f64, 1>::new(GAMMA, CFL, &sim.geom.allocated)
            .with_solver(Solver::Hllc)
            .expect("solver/regime mismatch")
            .theta(1.5)
            .reconstruction(recon);
    evolve(&mut sim, &sub, 0.1).expect("evolve failed");
    sim.geom
        .interior
        .iter()
        .map(|c| *sim.fields.prim.rho.view().at(c))
        .collect()
}

/// the ppm twins evolve a shocked state to a finite, physical result that is
/// genuinely a different scheme from plm (identical output would mean the ppm
/// dispatch silently fell through to the plm kernel).
#[test]
fn ppm_sod_evolves_and_differs_from_plm() {
    let rho_ppm = sod_density(Recon::Ppm, 3);
    assert!(
        rho_ppm.iter().all(|r| r.is_finite() && *r > 0.0),
        "ppm sod produced a non-finite or non-positive density"
    );
    let rho_plm = sod_density(Recon::Plm, 3);
    let max_diff = rho_ppm
        .iter()
        .zip(&rho_plm)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        max_diff > 1e-8,
        "ppm and plm sod are identical (max |drho| = {max_diff:e}); the ppm \
         dispatch fell through to the plm kernel"
    );
}

/// the allocation guard: dispatching ppm into the default ng = 2 halo must
/// refuse before any kernel runs — the -3 load would read unfilled memory.
#[test]
#[should_panic(expected = "allocated ghost width")]
fn ppm_refuses_the_default_two_ghost_allocation() {
    let _ = sod_density(Recon::Ppm, 2);
}

/// the refinement refusal: a hierarchy with more than one level has coarse-fine
/// boundaries, and the widest baked prolongation covers plm evolution only —
/// ppm across a level boundary would silently lose an order inside the domain,
/// so the first step refuses. a single-level hierarchy carries ppm freely (the
/// python driver wraps every uniform run in one).
#[test]
#[should_panic(expected = "reconstruction reach 3")]
fn ppm_refuses_a_refined_hierarchy() {
    let sim = sod_sim(3);
    let kern = |s: &Sim1| {
        AdiabaticSubstrateKernelSet::<HostMemory, f64, 1>::new(GAMMA, CFL, &s.geom.allocated)
            .with_solver(Solver::Hllc)
            .expect("solver/regime mismatch")
            .reconstruction(Recon::Ppm)
    };
    let ck = kern(&sim);
    let regions = [RefinementRegion {
        x_lo: [0.4],
        x_hi: [0.6],
    }];
    let mut hier =
        Hierarchy::with_refinement(sim, ck, &regions, ProlongOrder::Ppm, kern).unwrap();
    hier.levels[1].state.seed_cells(|[x]| {
        if x < 0.5 {
            Prim {
                rho: 1.0,
                vel: Tensor::new([0.0]),
                pre: 1.0,
            }
        } else {
            Prim {
                rho: 0.125,
                vel: Tensor::new([0.0]),
                pre: 0.1,
            }
        }
    });
    hier.evolve(0.01).unwrap();
}

/// monotonicity on a square wave: an advected discontinuity admits no new
/// extrema — the monotonized parabola clamps its interfaces to the neighbor
/// range and flattens at extrema, so any value outside the initial [1, 2]
/// band beyond roundoff accumulation is an oscillation. an unlimited parabola
/// rings at the 1e-1 scale here; the 1e-6 band separates bug from roundoff
/// without tolerating either.
#[test]
fn ppm_square_wave_admits_no_new_extrema() {
    const N: usize = 128;
    let dx = 1.0 / N as f64;
    let mut sim = Sim1::build(Newtonian, IdealGas { gamma: GAMMA }, Cartesian)
        .cells([N])
        .spacing([dx])
        .ghosts(3)
        .boundaries(Boundaries::uniform(BoundaryType::Periodic))
        .cfl(CFL)
        .timestepping(Timestepping::Rk2)
        .allocate()
        .expect("sim construction failed")
        .set_initial(|[x]| Prim {
            rho: if (0.25..0.75).contains(&x) { 2.0 } else { 1.0 },
            vel: Tensor::new([1.0]),
            pre: 1.0,
        })
        .build();
    let sub =
        AdiabaticSubstrateKernelSet::<HostMemory, f64, 1>::new(GAMMA, CFL, &sim.geom.allocated)
            .with_solver(Solver::Hllc)
            .expect("solver/regime mismatch")
            .reconstruction(Recon::Ppm);
    evolve(&mut sim, &sub, 1.0).expect("evolve failed");
    let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
    for c in sim.geom.interior.iter() {
        let r = *sim.fields.prim.rho.view().at(c);
        lo = lo.min(r);
        hi = hi.max(r);
    }
    assert!(
        lo >= 1.0 - 1e-6 && hi <= 2.0 + 1e-6,
        "square wave grew new extrema under ppm: rho in [{lo:.9}, {hi:.9}], initial band [1, 2]"
    );
}

/// monotonicity on a strong shock tube (pressure ratio 1e4): the exact solution's
/// density lies inside the initial [0.125, 1] band — the gamma = 1.4 shock
/// compression tops out at 6x the right state (0.75) and the rarefaction tail
/// stays above the right state — so any density outside the band is ringing at
/// the shock or contact. pressure obeys the same band argument on [1e-2, 1e2].
#[test]
fn ppm_strong_shock_stays_inside_the_wave_fan_band() {
    const N: usize = 256;
    let dx = 1.0 / N as f64;
    let mut sim = Sim1::build(Newtonian, IdealGas { gamma: GAMMA }, Cartesian)
        .cells([N])
        .spacing([dx])
        .ghosts(3)
        .boundaries(Boundaries::uniform(BoundaryType::Outflow))
        .cfl(CFL)
        .timestepping(Timestepping::Rk2)
        .allocate()
        .expect("sim construction failed")
        .set_initial(|[x]| {
            if x < 0.5 {
                Prim {
                    rho: 1.0,
                    vel: Tensor::new([0.0]),
                    pre: 100.0,
                }
            } else {
                Prim {
                    rho: 0.125,
                    vel: Tensor::new([0.0]),
                    pre: 0.01,
                }
            }
        })
        .build();
    let sub =
        AdiabaticSubstrateKernelSet::<HostMemory, f64, 1>::new(GAMMA, CFL, &sim.geom.allocated)
            .with_solver(Solver::Hllc)
            .expect("solver/regime mismatch")
            .reconstruction(Recon::Ppm);
    evolve(&mut sim, &sub, 0.012).expect("evolve failed");
    let (mut rho_lo, mut rho_hi) = (f64::INFINITY, f64::NEG_INFINITY);
    let (mut pre_lo, mut pre_hi) = (f64::INFINITY, f64::NEG_INFINITY);
    for c in sim.geom.interior.iter() {
        let r = *sim.fields.prim.rho.view().at(c);
        let p = *sim.fields.prim.pre_field().expect("adiabatic pre").view().at(c);
        rho_lo = rho_lo.min(r);
        rho_hi = rho_hi.max(r);
        pre_lo = pre_lo.min(p);
        pre_hi = pre_hi.max(p);
    }
    assert!(
        rho_lo >= 0.125 - 1e-6 && rho_hi <= 1.0 + 1e-6,
        "strong shock tube density left the wave-fan band under ppm: [{rho_lo:.9}, {rho_hi:.9}]"
    );
    assert!(
        pre_lo >= 0.01 - 1e-8 && pre_hi <= 100.0 + 1e-4,
        "strong shock tube pressure left the wave-fan band under ppm: [{pre_lo:.9}, {pre_hi:.9}]"
    );
}

/// the sink-regime flatten dials: onset above the subsonic accretor
/// turbulence's compressive scale (c ~ gamma Ma^2 < 1e-2 at mach 0.06), full by
/// the sealed-wall standing layer (c ~ 0.05, where a mid-ramp coefficient still
/// vents and the dip grows with resolution). the default dials are (0, 0) — the
/// pure parabola — because trans-sonic turbulence lives at c ~ 0.05-0.3 and an
/// active flatten there degrades ppm to first order in every eddy collision.
const SINK_FLATTEN: (f64, f64) = (0.015, 0.05);

/// gravitational infall onto a 4-cell body, optionally sealed by a porosity-0
/// penalized wall — the accretor geometry where the parabola's -3..+2 stencil
/// reads across the mask every step. returns (min K/K0 outside the mask + ppm
/// halo, radius of that minimum, max interior rho) after evolving to `t_end`;
/// panics on any non-finite or non-positive state.
#[cfg(test)]
fn sealed_wall_infall_probe(
    recon: Recon,
    solver: Solver,
    walled: bool,
    n: usize,
    ts: Timestepping,
    cfl: f64,
    theta: f64,
    flatten: (f64, f64),
    t_end: f64,
) -> (f64, f64, f64) {
    use symbi_ib::{Body, BodyCollection, SurfaceSpec};
    type Sim3 = SimState<Newtonian, 3, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;
    // the body radius is fixed in physical units (4 dx at n = 32) so a
    // resolution sweep refines the same physical problem.
    const R_BODY: f64 = 0.125;
    let dx = 1.0 / n as f64;
    let mut sim = Sim3::build(Newtonian, IdealGas { gamma: 5.0 / 3.0 }, Cartesian)
        .cells([n, n, n])
        .origin([-0.5, -0.5, -0.5])
        .spacing([dx, dx, dx])
        .ghosts(3)
        .boundaries(Boundaries::uniform(BoundaryType::Outflow))
        .cfl(cfl)
        .timestepping(ts)
        .allocate()
        .expect("sim construction failed")
        .set_initial(|_| Prim {
            rho: 1.0,
            vel: Tensor::new([0.0; 3]),
            pre: 0.6, // cs = 1 at gamma = 5/3
        })
        .build()
        .with_bodies(BodyCollection::new().add(if walled {
            // a black-hole-kind body carries the mask radius the penalize step keys on;
            // porosity 0 with the drain off makes it a pure sealed wall.
            Body::black_hole(
                0,
                Tensor::new([0.0; 3]),
                Tensor::zeros(),
                1.0,
                R_BODY,
                R_BODY,
                0.0,
                0.0,
                R_BODY,
            )
            .with_surface(SurfaceSpec::Porous {
                porosity: 0.0,
                k_eta_n: 50.0,
                k_eta_t: 0.0,
            })
        } else {
            Body::gravitational(0, Tensor::new([0.0; 3]), Tensor::zeros(), 1.0, R_BODY, R_BODY)
        }));
    let sub = AdiabaticSubstrateKernelSet::<HostMemory, f64, 3>::new(
        5.0 / 3.0,
        cfl,
        &sim.geom.allocated,
    )
    .with_solver(solver)
    .expect("solver/regime mismatch")
    .theta(theta)
    .ppm_flatten(flatten.0, flatten.1)
    .reconstruction(recon);
    evolve(&mut sim, &sub, t_end).expect("sealed wall evolve failed");
    assert!(sim.iteration > 10, "barely stepped; the probe is vacuous");

    let k0 = 0.6;
    let (mut worst_k, mut worst_r, mut rho_max) = (f64::INFINITY, 0.0_f64, 0.0_f64);
    let mut worst_c = [0isize; 3];
    for c in sim.geom.interior.iter() {
        let rho = *sim.fields.prim.rho.view().at(c);
        let pre = *sim.fields.prim.pre_field().expect("adiabatic pre").view().at(c);
        assert!(
            rho.is_finite() && pre.is_finite() && rho > 0.0 && pre > 0.0,
            "non-finite or non-positive state at {c:?} after {} steps",
            sim.iteration
        );
        rho_max = rho_max.max(rho);
        // K measured outside the mask plus its reconstruction halo (the ppm reach).
        let r: f64 = (0..3)
            .map(|a| {
                let x = (c[a] as f64 + 0.5) * dx - 0.5;
                x * x
            })
            .sum::<f64>()
            .sqrt();
        if r > R_BODY + 3.0 * dx {
            let kk = pre / rho.powf(5.0 / 3.0) / k0;
            if kk < worst_k {
                worst_k = kk;
                worst_r = r;
                worst_c = c;
            }
        }
    }
    // the character of the worst cell: a large pressure jump with converging
    // velocity marks a compression front (the classical ppm entropy-glitch
    // habitat); small jumps in diverging flow mark a smooth-region defect.
    let pre_v = sim.fields.prim.pre_field().expect("adiabatic pre").view();
    let (mut jump, mut divv) = (0.0_f64, 0.0_f64);
    for a in 0..3 {
        let (mut lo, mut hi) = (worst_c, worst_c);
        lo[a] -= 1;
        hi[a] += 1;
        let (pl, pc, pr) = (*pre_v.at(lo), *pre_v.at(worst_c), *pre_v.at(hi));
        jump = jump.max(((pr - pc).abs().max((pc - pl).abs())) / pc);
        let vl = *sim.fields.prim.vel[a].view().at(lo);
        let vr = *sim.fields.prim.vel[a].view().at(hi);
        divv += (vr - vl) / (2.0 * dx);
    }
    println!(
        "recon={recon:?} solver={solver:?} walled={walled} n={n} ts={ts:?} cfl={cfl}: \
         min K/K0 = {worst_k:.6} at r = {:.2} r_body ({:.2} dx past the halo), \
         max rho = {rho_max:.4}, worst cell |dp|/p = {jump:.3}, div v = {divv:.2}",
        worst_r / R_BODY,
        (worst_r - R_BODY - 3.0 * dx) / dx
    );
    (worst_k, worst_r, rho_max)
}

/// the attribution sweep behind the entropy gate below: rk order (time
/// coupling), cfl at fixed grid (dt scaling), and resolution at fixed cfl
/// (dx scaling), each printing the worst cell's shock character. diagnostic
/// only — run explicitly with --ignored.
#[test]
#[ignore]
fn diagnose_ppm_entropy_dip_scaling() {
    let p = |n, ts, cfl| {
        sealed_wall_infall_probe(
            Recon::Ppm,
            Solver::HllcPlus,
            false,
            n,
            ts,
            cfl,
            1.0,
            SINK_FLATTEN,
            0.08,
        )
    };
    p(32, Timestepping::Rk2, 0.3);
    p(32, Timestepping::Rk3, 0.3);
    p(32, Timestepping::Rk2, 0.15);
    p(48, Timestepping::Rk2, 0.3);
    p(64, Timestepping::Rk2, 0.3);
    // the plm limiter-compression axis: theta 1 (minmod) through 2 (full mc) —
    // the sharper members trade dissipation away exactly where the parabola did.
    for theta in [1.0, 1.5, 2.0] {
        for n in [32usize, 64] {
            sealed_wall_infall_probe(
                Recon::Plm,
                Solver::HllcPlus,
                false,
                n,
                Timestepping::Rk3,
                0.3,
                theta,
                (0.0, 0.0),
                0.08,
            );
        }
    }
}

/// the ppm entropy floor on gravitational infall: the adiabat violation must be
/// small and non-growing under refinement. the unflattened parabola vented
/// K = p/rho^gamma anti-diffusively — the dip grew with resolution (1.3e-3 at
/// n = 32 to 2.4e-3 at n = 64 open, 3.4e-5 to 1.7e-4 walled at a mid-ramp
/// flatten, worst cell tracking the steepening compression inward) — because
/// its dispersive truncation beats the riemann dissipation its own small face
/// jumps starve. the convergence-gated flatten restores the dissipation there;
/// what remains is truncation in the sub-onset band: measured 1.3e-5 at n = 32
/// falling to 9.7e-6 at n = 64 (walled), 3.4e-5 to 9.1e-6 (open). the bounds:
/// 5e-5 absolute sits 1.5-4x above the measured floor and 26x below the
/// unflattened vent; the 1.5x growth cap sits above extreme-value noise in a
/// min-over-cells statistic (healthy runs measure 0.23-0.75x) and below every
/// measured defective regime (1.85x, 5x). a fixed-n comparison against plm
/// (which holds K/K0 = 1.0 exactly here) is not the law: plm holds by
/// dissipation-dominance at second order, which a higher-order scheme cannot
/// match at fixed n and converges past instead.
/// the entropy dip's trend with resolution, per solver, over three refinements. two points
/// cannot separate a converging truncation error from a vent that saturates, and the flatten's
/// guarantee is a statement about the trend rather than about any single number.
///
/// run: cargo test --release -p symbi --test ppm_reconstruction -- --ignored dip_trend --nocapture
#[test]
#[ignore = "diagnostic: entropy-dip trend with resolution, per solver"]
fn diagnose_dip_trend_per_solver() {
    println!("\nsealed-wall infall, ppm, rk3, cfl 0.3");
    println!("{:>10} {:>6} {:>12} {:>9}", "solver", "n", "dip", "ratio");
    for solver in [Solver::Hllc, Solver::HllcPlus] {
        let mut prev: Option<f64> = None;
        for n in [32usize, 64, 128] {
            let (k, _, _) = sealed_wall_infall_probe(
                Recon::Ppm,
                solver,
                true,
                n,
                Timestepping::Rk3,
                0.3,
                1.0,
                SINK_FLATTEN,
                0.08,
            );
            let dip = (1.0 - k).max(0.0);
            let ratio = prev.map_or(f64::NAN, |p: f64| dip / p);
            println!("{:>10} {n:>6} {dip:>12.3e} {ratio:>9.3}", format!("{solver:?}"));
            prev = Some(dip);
        }
    }
    // the unflattened vent at each resolution: the scale the flatten is measured against.
    for n in [32usize, 64, 128] {
        let (k, _, _) = sealed_wall_infall_probe(
            Recon::Ppm,
            Solver::HllcPlus,
            true,
            n,
            Timestepping::Rk3,
            0.3,
            1.0,
            (0.0, 0.0),
            0.08,
        );
        println!("{:>10} {n:>6} {:>12.3e}", "unflattened", (1.0 - k).max(0.0));
    }
}

/// the convergence-gated flatten removes the ppm anti-diffusive entropy vent in a cell-scale
/// compression, and keeps removing more of it as the grid refines.
///
/// the measured quantity is the FRACTION of the vent that survives the flatten, read against
/// the unflattened parabola on the same grid, rather than the flattened dip on its own. the
/// vent is not a fixed target: on this sealed-wall infall the pure parabola dips 9.1e-4 at
/// n = 32, 9.5e-3 at n = 64 and 2.5e-2 at n = 128, sharpening by a factor of 28 across the
/// sweep as the compression it feeds on concentrates (div v runs -1.3 -> -31.9 over the same
/// grids). an absolute bound on the flattened dip therefore mixes two independent things —
/// how well the flatten works, and how large the defect it is working on has grown — and a
/// bound calibrated on one solver's dissipation transfers to no other.
///
/// the ratio separates them, and it is the statistic that behaves. measured over
/// n = 32, 64, 128 the surviving fraction falls monotonically for both solvers — classical
/// HLLC 3.0e-3 -> 9.3e-4 -> 1.8e-4, HLLC+ 7.2e-2 -> 9.0e-3 -> 4.1e-4 — while the flattened
/// dip itself is non-monotone for both (2.7e-6 -> 8.8e-6 -> 4.6e-6 and 6.5e-5 -> 8.5e-5 ->
/// 1.0e-5), because it is a min over cells and the minimum lands at a different radius on
/// each grid. the two resolutions here are what a debug build affords; the n = 128 point
/// lives in `diagnose_dip_trend_per_solver`.
///
/// the two bounds state the test's name. the fraction must be small on the finer grid, and it
/// must fall between the grids: a flatten that stopped tracking the vent would hold its
/// fraction fixed as the vent grew, and one that had died would let the fraction rise toward
/// one. HLLC+ carries the wider fraction of the two solvers because its low-mach correction
/// removes the velocity-jump damping that a stagnating compression would otherwise receive,
/// and it converges away all the same.
#[test]
fn the_ppm_entropy_dip_on_infall_is_small_and_converges_away() {
    /// the flatten dials under test; `(0, 0)` is the pure parabola they gate.
    const OFF: (f64, f64) = (0.0, 0.0);
    let dip = |flatten: (f64, f64), n: usize| -> (f64, f64) {
        let (k, _, rho_max) = sealed_wall_infall_probe(
            Recon::Ppm,
            Solver::HllcPlus,
            true,
            n,
            Timestepping::Rk3,
            0.3,
            1.0,
            flatten,
            0.08,
        );
        ((1.0 - k).max(0.0), rho_max)
    };

    let (vent_32, rho_max) = dip(OFF, 32);
    let (vent_64, _) = dip(OFF, 64);
    let (dip_32, _) = dip(SINK_FLATTEN, 32);
    let (dip_64, _) = dip(SINK_FLATTEN, 64);
    let (frac_32, frac_64) = (dip_32 / vent_32, dip_64 / vent_64);
    println!(
        "unflattened vent: n=32 {vent_32:.3e}, n=64 {vent_64:.3e}\n\
         flattened dip:    n=32 {dip_32:.3e}, n=64 {dip_64:.3e}\n\
         surviving fraction: n=32 {frac_32:.3e}, n=64 {frac_64:.3e}"
    );

    // the premise, in three parts. the wall has to load, the pure parabola has to vent, and
    // the vent has to sharpen with resolution -- the last is what makes the ratio the law
    // rather than the dip.
    assert!(
        rho_max > 1.05,
        "no pile-up developed (max rho = {rho_max:.4}); the wall never loaded and the probe \
         is vacuous"
    );
    assert!(
        vent_32 > 5.0e-4,
        "the pure parabola dips only {vent_32:.3e} on the infall probe (measured 9.1e-4 \
         walled at n = 32); the setup no longer exercises the vent the flatten dials close, \
         and the law below is vacuous"
    );
    assert!(
        vent_64 > 2.0 * vent_32,
        "the unflattened vent moved from {vent_32:.3e} to {vent_64:.3e} under refinement \
         (measured 9.1e-4 -> 9.5e-3); a vent that no longer sharpens with resolution is not \
         the anti-diffusive defect this gate is about, and the ratio below loses its meaning"
    );

    // small: the flatten leaves under two percent of the vent standing on the finer grid.
    // measured 9.0e-3 under HLLC+ and 9.3e-4 under classical HLLC.
    assert!(
        frac_64 < 2.0e-2,
        "the flatten leaves {frac_64:.3e} of the unflattened vent standing at n = 64 (dip \
         {dip_64:.3e} against vent {vent_64:.3e}); the convergence gate is no longer \
         restoring dissipation in cell-scale compressions"
    );
    // and converges away: the surviving fraction at least halves across the refinement.
    // measured 0.125 under HLLC+ and 0.31 under classical HLLC.
    assert!(
        frac_64 < 0.5 * frac_32,
        "the surviving fraction went {frac_32:.3e} -> {frac_64:.3e} across the refinement; \
         the flatten is no longer tracking a vent that grows with resolution, which is the \
         regime it exists for"
    );
}

