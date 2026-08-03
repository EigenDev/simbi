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
/// so the first step refuses. a SINGLE-level hierarchy carries ppm freely (the
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

/// monotonicity on a square wave: an advected discontinuity admits NO new
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
