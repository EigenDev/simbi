// =============================================================================
// substrate_well_balanced.rs
//
// the well-balanced (hydrostatic-equilibrium) regression gate for the
// curvilinear geometric pressure source. the source in
// symbi-discretize/src/gv/sources.rs (~:130) is written in the divergence's
// FLUX-DIFFERENCE form `(ptot*A_hi - ptot*A_lo)*inv_V` SPECIFICALLY so a static
// v=0, uniform-total-pressure state cancels the area-weighted flux divergence
// bit-for-bit: a uniform atmosphere must stay put.
//
// it is well-balanced BY CONSTRUCTION but nothing PINS it: the spherical sod
// tests assert the OPPOSITE (max |v| > 0.05, motion). a refactor of the
// area-weighting / source could silently break HSE with zero test failures.
//
// this gate seeds the minimal well-balanced state — UNIFORM rho, UNIFORM p, v=0
// — on a spherical-polar (strongest source) and a cylindrical annular shell, runs
// N evolve steps through the real evolve() loop, and asserts the radial velocity
// stays at machine zero (|v_r| < EPS). it directly exercises the sources.rs
// cancellation and needs no gravity: if the geometric source ever stops matching
// the geometric flux divergence, v grows and this FAILS.
// =============================================================================

use symbi::regimes::substrate::IsoSubstrateKernelSet;
use symbi::regimes::substrate_newton::AdiabaticSubstrateKernelSet;
use symbi::regimes::substrate_rhd::RhdSubstrateKernelSet;
use symbi::sim::evolve::evolve;
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_geometry::{Cylindrical, Spherical};
use symbi_hydro::eos::{IdealGas, Isothermal};
use symbi_hydro::isothermal::IsoNewtonian;
use symbi_hydro::newtonian::Newtonian;
use symbi_hydro::rhd::Rhd;
use symbi_hydro::state::{Prim, PrimG};
use symbi_xpu::{CpuSpace, HostMemory};

const GAMMA: f64 = 1.4;
// the uniform-state cancellation is grid-independent: it either holds bit-for-bit or it
// does not. a small grid pins it just as tightly as a large one and keeps the test fast.
const N: usize = 16;
const R_LO: f64 = 0.5; // annular shell r in [0.5, 1.5] — away from the r=0 singularity.
const DR: f64 = 1.0 / N as f64;
const T_FINAL: f64 = 0.05; // the uniform state's only signal is the sound speed; dt is finite.

// the machine-ish well-balancedness bound: a TRUE algebraic cancellation gives bit-zero, but the
// area-weighted divergence and the geometric source are evaluated as two separate floating-point
// expressions whose difference is a few rounding ulps per step, accumulated over the step count.
// 1e-12 is machine-ish (>> the ~1e-3..1e-1 a broken/absent source produces, << any physical flow).
const EPS: f64 = 1e-12;

// =============================================================================
// uniform-state seeders (rho, p constant, v=0) per regime.
// =============================================================================

fn seed_uniform_newton<M, const D: usize>(
    sim: &mut SimState<Newtonian, D, M, IdealGas<f64>, CpuSpace, HostMemory>,
    rho: f64,
    pre: f64,
) where
    M: symbi_geometry::Metric<f64, D> + Copy,
{
    let cnrg = sim.fields.cons.nrg_field().expect("Newtonian cons.nrg");
    for c in sim.geom.interior.iter() {
        sim.fields.cons.den.view_mut().set(c, rho);
        for k in 0..D {
            sim.fields.cons.mom[k].view_mut().set(c, 0.0);
        }
        cnrg.view_mut().set(c, pre / (GAMMA - 1.0));
    }
}

fn seed_uniform_iso<M, const D: usize>(
    sim: &mut SimState<IsoNewtonian, D, M, Isothermal<f64>, CpuSpace, HostMemory>,
    rho: f64,
) where
    M: symbi_geometry::Metric<f64, D> + Copy,
{
    for c in sim.geom.interior.iter() {
        sim.fields.cons.den.view_mut().set(c, rho);
        for k in 0..D {
            sim.fields.cons.mom[k].view_mut().set(c, 0.0);
        }
    }
}

fn seed_uniform_rhd<M, const D: usize>(
    sim: &mut SimState<Rhd, D, M, IdealGas<f64>, CpuSpace, HostMemory>,
    rho: f64,
    pre: f64,
) where
    M: symbi_geometry::Metric<f64, D> + Copy,
{
    // v=0 => W=1: D=rho, S=0, tau=p/(gamma-1).
    let cnrg = sim.fields.cons.nrg_field().expect("Rhd cons.nrg");
    for c in sim.geom.interior.iter() {
        sim.fields.cons.den.view_mut().set(c, rho);
        for k in 0..D {
            sim.fields.cons.mom[k].view_mut().set(c, 0.0);
        }
        cnrg.view_mut().set(c, pre / (GAMMA - 1.0));
    }
}

// the max |v_r| over the interior of a D-dim sim (radial component = vel[0]).
fn max_radial_vel<R, const D: usize, M, E>(sim: &SimState<R, D, M, E, CpuSpace, HostMemory>) -> f64
where
    R: symbi_hydro::Regime<f64, D>,
    M: symbi_geometry::Metric<f64, D> + Copy,
    E: symbi_hydro::eos::Eos<f64>,
{
    sim.geom
        .interior
        .iter()
        .map(|c| sim.fields.prim.vel[0].view().at(c).abs())
        .fold(0.0_f64, f64::max)
}

// =============================================================================
// spherical-polar (r): the geometric pressure source is the strongest here
// (radial face area ~ r^2; the 2p/r continuum source). uniform rho=1, p=1, v=0.
// =============================================================================

#[test]
fn well_balanced_spherical_1d_adiabatic() {
    // uniform rho=1, p=1, v=0 on a spherical shell: the area-weighted flux divergence and the
    // geometric pressure source must cancel, so the gas never moves. this is the canonical
    // sources.rs:130 cancellation pin.
    let mut sph = SimState::<Newtonian, 1, Spherical, IdealGas<f64>, CpuSpace, HostMemory>::build(
        Newtonian,
        IdealGas { gamma: GAMMA },
        Spherical,
    )
    .cells([N])
    .origin([R_LO])
    .spacing([DR])
    .boundaries(Boundaries::uniform(BoundaryType::Reflect))
    .allocate()
    .expect("spherical sim construction failed")
    .set_initial(|_| Prim {
        rho: 1.0,
        vel: Tensor::new([0.0]),
        pre: 1.0,
    })
    .build();
    seed_uniform_newton(&mut sph, 1.0, 1.0);
    assert_eq!(
        sph.geom.coords,
        symbi_geometry::Geometry::Spherical,
        "coords must be Spherical"
    );

    let sub =
        AdiabaticSubstrateKernelSet::<HostMemory, f64, 1>::new(GAMMA, 0.4, &sph.geom.allocated);
    evolve(&mut sph, &sub, T_FINAL).expect("spherical well-balanced evolution failed");

    // the state must stay finite, positive, and STATIC — the well-balanced cancellation pinned.
    for c in sph.geom.interior.iter() {
        let rho = *sph.fields.prim.rho.view().at(c);
        let p = *sph.fields.prim.pre_field().expect("prim.pre").view().at(c);
        assert!(rho.is_finite() && rho > 0.0, "bad density {rho} at {c:?}");
        assert!(p.is_finite() && p > 0.0, "bad pressure {p} at {c:?}");
    }
    let mv = max_radial_vel(&sph);
    assert!(
        mv < EPS,
        "spherical adiabatic NOT well-balanced: uniform HSE drifted, max |v_r| = {mv:e} (>= {EPS:e}) \
         over {} steps — the geometric pressure source no longer cancels the area-weighted divergence",
        sph.iteration,
    );
    println!(
        "WELL-BALANCED SPHERICAL ADIABATIC: {} steps to t={:.3}, max |v_r| {:e}",
        sph.iteration, sph.time, mv
    );
}

#[test]
fn well_balanced_spherical_1d_iso() {
    // iso: uniform rho=1 => uniform p = cs^2 rho; the geometric pressure source cs^2 rho *
    // (A_hi-A_lo)/V still must cancel the area-weighted divergence at v=0.
    let cs = 1.0_f64;
    let mut sph =
        SimState::<IsoNewtonian, 1, Spherical, Isothermal<f64>, CpuSpace, HostMemory>::build(
            IsoNewtonian,
            Isothermal { cs },
            Spherical,
        )
        .cells([N])
        .origin([R_LO])
        .spacing([DR])
        .boundaries(Boundaries::uniform(BoundaryType::Reflect))
        .allocate()
        .expect("iso spherical sim")
        .set_initial(|_| PrimG {
            rho: 1.0,
            vel: Tensor::new([0.0]),
            pre: Default::default(),
        })
        .build();
    seed_uniform_iso(&mut sph, 1.0);

    let sub = IsoSubstrateKernelSet::<HostMemory, f64, 1>::new(cs, 0.4, &sph.geom.allocated);
    evolve(&mut sph, &sub, T_FINAL).expect("iso spherical well-balanced evolution failed");

    for c in sph.geom.interior.iter() {
        let rho = *sph.fields.prim.rho.view().at(c);
        assert!(
            rho.is_finite() && rho > 0.0,
            "iso: bad density {rho} at {c:?}"
        );
    }
    let mv = max_radial_vel(&sph);
    assert!(
        mv < EPS,
        "iso spherical NOT well-balanced: max |v_r| = {mv:e} over {} steps",
        sph.iteration
    );
    println!(
        "WELL-BALANCED SPHERICAL ISO: {} steps, max |v_r| {:e}",
        sph.iteration, mv
    );
}

#[test]
fn well_balanced_spherical_1d_rhd() {
    // rhd at v=0 (W=1): tau = p/(gamma-1), S=0. the curvilinear source reads prim.pre; the
    // relativistic momentum density (cons.mom = rho h W^2 v = 0) makes the inertial vanish.
    let mut sph = SimState::<Rhd, 1, Spherical, IdealGas<f64>, CpuSpace, HostMemory>::build(
        Rhd,
        IdealGas { gamma: GAMMA },
        Spherical,
    )
    .cells([N])
    .origin([R_LO])
    .spacing([DR])
    .boundaries(Boundaries::uniform(BoundaryType::Reflect))
    .allocate()
    .expect("rhd spherical sim")
    .set_initial(|_| Prim {
        rho: 1.0,
        vel: Tensor::new([0.0]),
        pre: 1.0,
    })
    .build();
    seed_uniform_rhd(&mut sph, 1.0, 1.0);

    let sub = RhdSubstrateKernelSet::<HostMemory, f64, 1>::new(GAMMA, 0.4, &sph.geom.allocated);
    evolve(&mut sph, &sub, T_FINAL).expect("rhd spherical well-balanced evolution failed");

    for c in sph.geom.interior.iter() {
        let rho = *sph.fields.prim.rho.view().at(c);
        let p = *sph.fields.prim.pre_field().expect("prim.pre").view().at(c);
        assert!(
            rho.is_finite() && rho > 0.0,
            "rhd: bad density {rho} at {c:?}"
        );
        assert!(p.is_finite() && p > 0.0, "rhd: bad pressure {p} at {c:?}");
    }
    let mv = max_radial_vel(&sph);
    assert!(
        mv < EPS,
        "rhd spherical NOT well-balanced: max |v_r| = {mv:e} over {} steps",
        sph.iteration
    );
    println!(
        "WELL-BALANCED SPHERICAL RHD: {} steps, max |v_r| {:e}",
        sph.iteration, mv
    );
}

// =============================================================================
// cylindrical (r): 1D radial cylindrical shell. radial face area ~ r, volume ~ r dr;
// the geometric pressure source p*(A_hi-A_lo)/V must still cancel at v=0.
// =============================================================================

#[test]
fn well_balanced_cylindrical_1d_adiabatic() {
    let mut cyl =
        SimState::<Newtonian, 1, Cylindrical, IdealGas<f64>, CpuSpace, HostMemory>::build(
            Newtonian,
            IdealGas { gamma: GAMMA },
            Cylindrical,
        )
        .cells([N])
        .origin([R_LO])
        .spacing([DR])
        .boundaries(Boundaries::uniform(BoundaryType::Reflect))
        .allocate()
        .expect("cylindrical sim construction failed")
        .set_initial(|_| Prim {
            rho: 1.0,
            vel: Tensor::new([0.0]),
            pre: 1.0,
        })
        .build();
    seed_uniform_newton(&mut cyl, 1.0, 1.0);
    assert_eq!(
        cyl.geom.coords,
        symbi_geometry::Geometry::Cylindrical,
        "coords must be Cylindrical"
    );

    let sub =
        AdiabaticSubstrateKernelSet::<HostMemory, f64, 1>::new(GAMMA, 0.4, &cyl.geom.allocated);
    evolve(&mut cyl, &sub, T_FINAL).expect("cylindrical well-balanced evolution failed");

    for c in cyl.geom.interior.iter() {
        let rho = *cyl.fields.prim.rho.view().at(c);
        let p = *cyl.fields.prim.pre_field().expect("prim.pre").view().at(c);
        assert!(
            rho.is_finite() && rho > 0.0,
            "cyl: bad density {rho} at {c:?}"
        );
        assert!(p.is_finite() && p > 0.0, "cyl: bad pressure {p} at {c:?}");
    }
    let mv = max_radial_vel(&cyl);
    assert!(
        mv < EPS,
        "cylindrical adiabatic NOT well-balanced: max |v_r| = {mv:e} (>= {EPS:e}) over {} steps",
        cyl.iteration,
    );
    println!(
        "WELL-BALANCED CYLINDRICAL ADIABATIC: {} steps to t={:.3}, max |v_r| {:e}",
        cyl.iteration, cyl.time, mv
    );
}
