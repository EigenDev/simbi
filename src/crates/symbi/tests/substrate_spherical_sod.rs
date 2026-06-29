// =============================================================================
// substrate_spherical_sod.rs
//
// the curvilinear-integration capstone: a full SPHERICAL adiabatic Euler Sod
// evolution where the M-generic AdiabaticSubstrateKernelSet runs on a
// SimState<Newtonian, 1, Spherical, ..> through the real evolve() loop. the
// metric M=Spherical flows to PartitionGeometry.coords, which makes the kernelset
// select the curvilinear kernel instances (adiabatic_godunov_euler_sph_1d,
// iso_wave_speed_map_sph_1d) — the area-weighted divergence + the well-balanced
// geometric pressure source (2p/r in the radial continuum limit) + per-cell CFL.
//
// proof the geometry is ACTIVE (not silently Cartesian): run the IDENTICAL grid +
// initial state under M=Spherical and M=Cartesian; the spherical geometric source
// changes the solution, so the radial density profiles must DIFFER measurably. a
// missing _sph kernel would panic in kernel_by_name, so a clean run also proves the
// curvilinear instances are registered and dispatched.
// =============================================================================

use symbi::regimes::substrate::IsoSubstrateKernelSet;
use symbi::regimes::substrate_newton::AdiabaticSubstrateKernelSet;
use symbi::regimes::substrate_srhd::SrhdSubstrateKernelSet;
use symbi::sim::evolve::evolve;
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_geometry::{Cartesian, Spherical};
use symbi_hydro::eos::{IdealGas, Isothermal};
use symbi_hydro::isothermal::IsoNewtonian;
use symbi_hydro::newtonian::Newtonian;
use symbi_hydro::srhd::Srhd;
use symbi_hydro::state::{Prim, PrimG};
use symbi_xpu::{CpuSpace, HostMemory};

type SimSph = SimState<Newtonian, 1, Spherical, IdealGas<f64>, CpuSpace, HostMemory>;
type SimCart = SimState<Newtonian, 1, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;

const GAMMA: f64 = 1.4;
const N: usize = 128;
const R_LO: f64 = 0.5; // a shell r in [0.5, 1.5] — away from the r=0 singularity.
const DR: f64 = 1.0 / N as f64;
const T_FINAL: f64 = 0.05; // short: the waves stay interior to the shell.

// Sod across r = 1.0: (rho, p) = (1, 1) | (0.125, 0.1), v = 0.
fn set_sod_ic<M>(sim: &mut SimState<Newtonian, 1, M, IdealGas<f64>, CpuSpace, HostMemory>)
where
    M: symbi_geometry::Metric<f64, 1> + Copy,
{
    let cnrg = sim.fields.cons.nrg_field().expect("Newtonian cons.nrg");
    for c in sim.geom.interior.iter() {
        let r = R_LO + (c[0] as f64 + 0.5) * DR;
        let (rho, pre) = if r < 1.0 { (1.0, 1.0) } else { (0.125, 0.1) };
        sim.fields.cons.den.view_mut().set(c, rho);
        sim.fields.cons.mom[0].view_mut().set(c, 0.0);
        cnrg.view_mut().set(c, pre / (GAMMA - 1.0));
    }
}

// iso Sod: a density jump at r=1, v=0 (p = cs^2 rho is recovered in c2p). no energy.
fn set_iso_ic<M>(sim: &mut SimState<IsoNewtonian, 1, M, Isothermal<f64>, CpuSpace, HostMemory>)
where
    M: symbi_geometry::Metric<f64, 1> + Copy,
{
    for c in sim.geom.interior.iter() {
        let r = R_LO + (c[0] as f64 + 0.5) * DR;
        let rho = if r < 1.0 { 1.0 } else { 0.125 };
        sim.fields.cons.den.view_mut().set(c, rho);
        sim.fields.cons.mom[0].view_mut().set(c, 0.0);
    }
}

// srhd sharp Sod (rho,p) = (1,1) | (0.125,0.1), v=0: D=rho, S=0, tau=p/(gamma-1) (W=1).
// the relativistic HLLE keeps the flow subluminal; with the CORRECT per-cell physical CFL
// width (srhd_wave_speed_map's curvilinear dispatch — the earlier NaN was a wildly wrong
// dt from misreading x_lo as inv_dx, NOT a geometric-source or c2p flaw) the spherical
// shock tube runs clean.
fn set_srhd_ic<M>(sim: &mut SimState<Srhd, 1, M, IdealGas<f64>, CpuSpace, HostMemory>)
where
    M: symbi_geometry::Metric<f64, 1> + Copy,
{
    let cnrg = sim.fields.cons.nrg_field().expect("Srhd cons.nrg");
    for c in sim.geom.interior.iter() {
        let r = R_LO + (c[0] as f64 + 0.5) * DR;
        let (rho, pre) = if r < 1.0 { (1.0, 1.0) } else { (0.125, 0.1) };
        sim.fields.cons.den.view_mut().set(c, rho);
        sim.fields.cons.mom[0].view_mut().set(c, 0.0);
        cnrg.view_mut().set(c, pre / (GAMMA - 1.0));
    }
}

#[test]
fn full_substrate_spherical_adiabatic_sod() {
    // the SAME M-generic kernelset serves both metrics; the SimState's M drives the
    // curvilinear kernel selection (sim.geom.coords).
    let sub = AdiabaticSubstrateKernelSet::<HostMemory, f64, 1>::new(GAMMA, 0.4, &{
        // a throwaway allocated domain only to size the cfl scratch (N + 2*ng).
        let tmp = SimSph::build(Newtonian, IdealGas { gamma: GAMMA }, Spherical)
            .cells([N])
            .origin([R_LO])
            .spacing([DR])
            .boundaries(Boundaries::uniform(BoundaryType::Outflow))
            .allocate()
            .expect("sizing sim")
            .set_initial(|_| Prim { rho: 1.0, vel: Tensor::new([0.0]), pre: 1.0 })
            .build();
        tmp.geom.allocated.clone()
    });

    // construct + reach Ready with a trivial seed; the radial Sod IC is applied via set_sod_ic.
    let mut sph = SimSph::build(Newtonian, IdealGas { gamma: GAMMA }, Spherical)
        .cells([N])
        .origin([R_LO])
        .spacing([DR])
        .boundaries(Boundaries::uniform(BoundaryType::Outflow))
        .allocate()
        .expect("spherical sim construction failed")
        .set_initial(|_| Prim { rho: 1.0, vel: Tensor::new([0.0]), pre: 1.0 })
        .build();
    set_sod_ic(&mut sph);
    // the spherical SimState reports its coordinate system through the metric.
    assert_eq!(sph.geom.coords, symbi_geometry::Geometry::Spherical, "coords must be Spherical");

    evolve(&mut sph, &sub, T_FINAL).expect("spherical evolution failed");

    // finite + positive density and pressure everywhere (a real, stable curvilinear solve).
    let cells: Vec<[isize; 1]> = sph.geom.interior.iter().collect();
    let pre = sph.fields.prim.pre_field().expect("prim.pre");
    let mut max_vel = 0.0_f64;
    for c in &cells {
        let rho = *sph.fields.prim.rho.view().at(*c);
        let p = *pre.view().at(*c);
        let v = *sph.fields.prim.vel[0].view().at(*c);
        assert!(rho.is_finite() && rho > 0.0, "bad density {rho} at {c:?}");
        assert!(p.is_finite() && p > 0.0, "bad pressure {p} at {c:?}");
        max_vel = max_vel.max(v.abs());
    }
    assert!(max_vel > 0.05, "gas did not move (max |v| = {max_vel})");

    // geometry is ACTIVE: the IDENTICAL grid + IC under Cartesian gives a DIFFERENT
    // profile (no 2p/r geometric source, flat divergence). compare radial density.
    let mut cart = SimCart::build(Newtonian, IdealGas { gamma: GAMMA }, Cartesian)
        .cells([N])
        .origin([R_LO])
        .spacing([DR])
        .boundaries(Boundaries::uniform(BoundaryType::Outflow))
        .allocate()
        .expect("cartesian sim construction failed")
        .set_initial(|_| Prim { rho: 1.0, vel: Tensor::new([0.0]), pre: 1.0 })
        .build();
    set_sod_ic(&mut cart);
    evolve(&mut cart, &sub, T_FINAL).expect("cartesian evolution failed");

    let max_diff = cells
        .iter()
        .map(|c| {
            (*sph.fields.prim.rho.view().at(*c) - *cart.fields.prim.rho.view().at(*c)).abs()
        })
        .fold(0.0_f64, f64::max);
    assert!(
        max_diff > 1e-3,
        "spherical and cartesian profiles barely differ (max |drho| = {max_diff:e}) — \
         the geometric source did not engage",
    );

    println!(
        "SPHERICAL SOD: {} steps to t={:.3}, max |v| {:.3}, max |drho vs cartesian| {:.4}",
        sph.iteration, sph.time, max_vel, max_diff,
    );
}

// the same M-generic IsoSubstrateKernelSet on Spherical vs Cartesian. iso has no energy;
// the pressure p = cs^2 rho is substrate-owned (self.pre), so the geometric pressure
// source cs^2 rho * (A_hi-A_lo)/V still drives radial motion the flat divergence lacks.
#[test]
fn full_substrate_spherical_iso() {
    type SimSph = SimState<IsoNewtonian, 1, Spherical, Isothermal<f64>, CpuSpace, HostMemory>;
    type SimCart = SimState<IsoNewtonian, 1, Cartesian, Isothermal<f64>, CpuSpace, HostMemory>;
    let cs = 1.0_f64;

    let mut sph = SimSph::build(IsoNewtonian, Isothermal { cs }, Spherical)
        .cells([N])
        .origin([R_LO])
        .spacing([DR])
        .boundaries(Boundaries::uniform(BoundaryType::Outflow))
        .allocate()
        .expect("iso spherical sim")
        .set_initial(|_| PrimG { rho: 1.0, vel: Tensor::new([0.0]), pre: Default::default() })
        .build();
    set_iso_ic(&mut sph);
    let sub = IsoSubstrateKernelSet::<HostMemory, f64, 1>::new(cs, 0.4, &sph.geom.allocated);
    evolve(&mut sph, &sub, T_FINAL).expect("iso spherical evolution failed");

    let cells: Vec<[isize; 1]> = sph.geom.interior.iter().collect();
    for c in &cells {
        let rho = *sph.fields.prim.rho.view().at(*c);
        assert!(rho.is_finite() && rho > 0.0, "iso: bad density {rho} at {c:?}");
    }
    let mut cart = SimCart::build(IsoNewtonian, Isothermal { cs }, Cartesian)
        .cells([N])
        .origin([R_LO])
        .spacing([DR])
        .boundaries(Boundaries::uniform(BoundaryType::Outflow))
        .allocate()
        .expect("iso cartesian sim")
        .set_initial(|_| PrimG { rho: 1.0, vel: Tensor::new([0.0]), pre: Default::default() })
        .build();
    set_iso_ic(&mut cart);
    evolve(&mut cart, &sub, T_FINAL).expect("iso cartesian evolution failed");
    let max_diff = cells
        .iter()
        .map(|c| (*sph.fields.prim.rho.view().at(*c) - *cart.fields.prim.rho.view().at(*c)).abs())
        .fold(0.0_f64, f64::max);
    assert!(max_diff > 1e-3, "iso: spherical == cartesian (geometric source idle): {max_diff:e}");
    println!("SPHERICAL ISO: {} steps, max |drho vs cartesian| {:.4}", sph.iteration, max_diff);
}

// the same M-generic SrhdSubstrateKernelSet on Spherical vs Cartesian. for v=0 the
// relativistic conserved tau = p/(gamma-1) (W=1); the curvilinear source uses prim.pre
// (gas p) + the relativistic momentum density (cons.mom = rho h W^2 v) for the inertial.
#[test]
fn full_substrate_spherical_srhd() {
    type SimSph = SimState<Srhd, 1, Spherical, IdealGas<f64>, CpuSpace, HostMemory>;
    type SimCart = SimState<Srhd, 1, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;

    let mut sph = SimSph::build(Srhd, IdealGas { gamma: GAMMA }, Spherical)
        .cells([N])
        .origin([R_LO])
        .spacing([DR])
        .boundaries(Boundaries::uniform(BoundaryType::Outflow))
        .allocate()
        .expect("srhd spherical sim")
        .set_initial(|_| Prim { rho: 1.0, vel: Tensor::new([0.0]), pre: 1.0 })
        .build();
    set_srhd_ic(&mut sph);
    let sub = SrhdSubstrateKernelSet::<HostMemory, f64, 1>::new(GAMMA, 0.4, &sph.geom.allocated);
    evolve(&mut sph, &sub, T_FINAL).expect("srhd spherical evolution failed");

    let cells: Vec<[isize; 1]> = sph.geom.interior.iter().collect();
    let pre = sph.fields.prim.pre_field().expect("prim.pre");
    for c in &cells {
        let rho = *sph.fields.prim.rho.view().at(*c);
        let p = *pre.view().at(*c);
        assert!(rho.is_finite() && rho > 0.0, "srhd: bad density {rho} at {c:?}");
        assert!(p.is_finite() && p > 0.0, "srhd: bad pressure {p} at {c:?}");
    }
    let mut cart = SimCart::build(Srhd, IdealGas { gamma: GAMMA }, Cartesian)
        .cells([N])
        .origin([R_LO])
        .spacing([DR])
        .boundaries(Boundaries::uniform(BoundaryType::Outflow))
        .allocate()
        .expect("srhd cartesian sim")
        .set_initial(|_| Prim { rho: 1.0, vel: Tensor::new([0.0]), pre: 1.0 })
        .build();
    set_srhd_ic(&mut cart);
    evolve(&mut cart, &sub, T_FINAL).expect("srhd cartesian evolution failed");
    let max_diff = cells
        .iter()
        .map(|c| (*sph.fields.prim.rho.view().at(*c) - *cart.fields.prim.rho.view().at(*c)).abs())
        .fold(0.0_f64, f64::max);
    assert!(max_diff > 1e-3, "srhd: spherical == cartesian (geometric source idle): {max_diff:e}");
    println!("SPHERICAL SRHD: {} steps, max |drho vs cartesian| {:.4}", sph.iteration, max_diff);
}

// D-generic radial-gradient ICs for the ndim>=2 curvilinear smokes (rho or p decreasing in
// r, v=0): the radial gradient breaks the well-balanced HSE so the area-weighted divergence
// + geometric pressure source drive flow the flat Cartesian divergence lacks. radially
// symmetric (transverse v = 0), so the dominant geometry terms are exercised; the inertial
// centrifugal/coriolis source is validated analytically in geometry_algebra.rs. annular
// wedge away from r=0 and the poles; outflow walls. only c[0] (radial index) matters.
fn set_grad_newton<M, const D: usize>(
    sim: &mut SimState<Newtonian, D, M, IdealGas<f64>, CpuSpace, HostMemory>,
    r_lo: f64,
    dr: f64,
) where
    M: symbi_geometry::Metric<f64, D> + Copy,
{
    let cnrg = sim.fields.cons.nrg_field().expect("cons.nrg");
    for c in sim.geom.interior.iter() {
        let r = r_lo + (c[0] as f64 + 0.5) * dr;
        let pre = 1.0 - 0.3 * (r - r_lo);
        sim.fields.cons.den.view_mut().set(c, 1.0);
        for k in 0..D {
            sim.fields.cons.mom[k].view_mut().set(c, 0.0);
        }
        cnrg.view_mut().set(c, pre / (GAMMA - 1.0));
    }
}

// srhd is identical to newton at v=0 (W=1 => tau = p/(gamma-1), S = 0).
fn set_grad_srhd<M, const D: usize>(
    sim: &mut SimState<Srhd, D, M, IdealGas<f64>, CpuSpace, HostMemory>,
    r_lo: f64,
    dr: f64,
) where
    M: symbi_geometry::Metric<f64, D> + Copy,
{
    let cnrg = sim.fields.cons.nrg_field().expect("Srhd cons.nrg");
    for c in sim.geom.interior.iter() {
        let r = r_lo + (c[0] as f64 + 0.5) * dr;
        let pre = 1.0 - 0.3 * (r - r_lo);
        sim.fields.cons.den.view_mut().set(c, 1.0);
        for k in 0..D {
            sim.fields.cons.mom[k].view_mut().set(c, 0.0);
        }
        cnrg.view_mut().set(c, pre / (GAMMA - 1.0));
    }
}

// iso has no energy: a radial DENSITY gradient (p = cs^2 rho recovered in c2p); the
// geometric pressure source cs^2 rho * (A_hi-A_lo)/V drives the flow.
fn set_grad_iso<M, const D: usize>(
    sim: &mut SimState<IsoNewtonian, D, M, Isothermal<f64>, CpuSpace, HostMemory>,
    r_lo: f64,
    dr: f64,
) where
    M: symbi_geometry::Metric<f64, D> + Copy,
{
    for c in sim.geom.interior.iter() {
        let r = r_lo + (c[0] as f64 + 0.5) * dr;
        let rho = 1.0 - 0.3 * (r - r_lo);
        sim.fields.cons.den.view_mut().set(c, rho);
        for k in 0..D {
            sim.fields.cons.mom[k].view_mut().set(c, 0.0);
        }
    }
}

#[test]
fn full_substrate_spherical_2d_adiabatic() {
    type Sph = SimState<Newtonian, 2, Spherical, IdealGas<f64>, CpuSpace, HostMemory>;
    type Cart = SimState<Newtonian, 2, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;
    let (nr, nt) = (48usize, 12usize);
    let (r_lo, dr, t_lo, dt_) = (0.5_f64, 1.0 / nr as f64, 0.6_f64, 0.4 / nt as f64);
    let mut sph = Sph::build(Newtonian, IdealGas { gamma: GAMMA }, Spherical)
        .cells([nr, nt])
        .origin([r_lo, t_lo])
        .spacing([dr, dt_])
        .boundaries(Boundaries::uniform(BoundaryType::Outflow))
        .allocate()
        .expect("sph 2d")
        .set_initial(|_| Prim { rho: 1.0, vel: Tensor::new([0.0, 0.0]), pre: 1.0 })
        .build();
    set_grad_newton(&mut sph, r_lo, dr);
    let sub = AdiabaticSubstrateKernelSet::<HostMemory, f64, 2>::new(GAMMA, 0.4, &sph.geom.allocated);
    evolve(&mut sph, &sub, 0.03).expect("2d spherical evolve failed");

    let cells: Vec<[isize; 2]> = sph.geom.interior.iter().collect();
    let pre = sph.fields.prim.pre_field().expect("prim.pre");
    let mut max_vel = 0.0_f64;
    for c in &cells {
        let rho = *sph.fields.prim.rho.view().at(*c);
        let p = *pre.view().at(*c);
        assert!(rho.is_finite() && rho > 0.0, "2d: bad density {rho} at {c:?}");
        assert!(p.is_finite() && p > 0.0, "2d: bad pressure {p} at {c:?}");
        let v = ((*sph.fields.prim.vel[0].view().at(*c)).powi(2)
            + (*sph.fields.prim.vel[1].view().at(*c)).powi(2)).sqrt();
        max_vel = max_vel.max(v);
    }
    assert!(max_vel > 0.005, "2d: no flow developed (max |v| = {max_vel})");

    let mut cart = Cart::build(Newtonian, IdealGas { gamma: GAMMA }, Cartesian)
        .cells([nr, nt])
        .origin([r_lo, t_lo])
        .spacing([dr, dt_])
        .boundaries(Boundaries::uniform(BoundaryType::Outflow))
        .allocate()
        .expect("cart 2d")
        .set_initial(|_| Prim { rho: 1.0, vel: Tensor::new([0.0, 0.0]), pre: 1.0 })
        .build();
    set_grad_newton(&mut cart, r_lo, dr);
    evolve(&mut cart, &sub, 0.03).expect("2d cartesian evolve failed");
    let max_diff = cells
        .iter()
        .map(|c| (*sph.fields.prim.rho.view().at(*c) - *cart.fields.prim.rho.view().at(*c)).abs())
        .fold(0.0_f64, f64::max);
    // 2e-4 >> machine noise (a Cartesian-vs-Cartesian run would be bit-identical, diff 0):
    // the area-weighted divergence + geometric source are active in the spherical run.
    assert!(max_diff > 1e-4, "2d: spherical == cartesian (geometric source idle): {max_diff:e}");
    println!("SPHERICAL 2D ADIABATIC: {} steps, max |v| {:.3}, max |drho vs cart| {:.4}", sph.iteration, max_vel, max_diff);
}

// ---- ndim>=2 spherical smokes: close the (regime x dim x spherical) matrix ----
// each runs the M-generic _sph kernel through evolve() on an annular wedge and asserts a
// finite/positive/(subluminal) state + geometry ACTIVE (the radial profile differs from
// the IDENTICAL Cartesian run; a Cartesian-vs-Cartesian run is bit-identical, diff 0). a
// missing _sph kernel would panic in kernel_by_name, so a clean run proves dispatch too.
// these cover the emitted-but-untested cells: adiabatic 3D, iso 2D/3D, srhd 2D/3D.

const TR: usize = 24; // 3D wedge radial cells
const TT: usize = 8; // theta cells
const TP: usize = 8; // phi cells

#[test]
fn full_substrate_spherical_3d_adiabatic() {
    type Sph = SimState<Newtonian, 3, Spherical, IdealGas<f64>, CpuSpace, HostMemory>;
    type Cart = SimState<Newtonian, 3, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;
    let (r_lo, dr) = (0.5_f64, 1.0 / TR as f64);
    let dims = [TR, TT, TP];
    let xlo = [r_lo, 0.6, 0.2];
    let dx = [dr, 0.4 / TT as f64, 0.5 / TP as f64];
    let bcs = Boundaries::uniform(BoundaryType::Outflow);
    let mut sph = Sph::build(Newtonian, IdealGas { gamma: GAMMA }, Spherical)
        .cells(dims)
        .origin(xlo)
        .spacing(dx)
        .boundaries(bcs)
        .allocate()
        .expect("sph 3d")
        .set_initial(|_| Prim { rho: 1.0, vel: Tensor::new([0.0; 3]), pre: 1.0 })
        .build();
    set_grad_newton(&mut sph, r_lo, dr);
    let sub = AdiabaticSubstrateKernelSet::<HostMemory, f64, 3>::new(GAMMA, 0.4, &sph.geom.allocated);
    evolve(&mut sph, &sub, 0.02).expect("3d sph adiabatic evolve failed");
    let cells: Vec<[isize; 3]> = sph.geom.interior.iter().collect();
    let pre = sph.fields.prim.pre_field().expect("prim.pre");
    for c in &cells {
        let (rho, p) = (*sph.fields.prim.rho.view().at(*c), *pre.view().at(*c));
        assert!(rho.is_finite() && rho > 0.0, "3d adiabatic: bad density {rho} at {c:?}");
        assert!(p.is_finite() && p > 0.0, "3d adiabatic: bad pressure {p} at {c:?}");
    }
    let mut cart = Cart::build(Newtonian, IdealGas { gamma: GAMMA }, Cartesian)
        .cells(dims)
        .origin(xlo)
        .spacing(dx)
        .boundaries(bcs)
        .allocate()
        .expect("cart 3d")
        .set_initial(|_| Prim { rho: 1.0, vel: Tensor::new([0.0; 3]), pre: 1.0 })
        .build();
    set_grad_newton(&mut cart, r_lo, dr);
    evolve(&mut cart, &sub, 0.02).expect("3d cart adiabatic evolve failed");
    let max_diff = cells.iter().map(|c| (*sph.fields.prim.rho.view().at(*c) - *cart.fields.prim.rho.view().at(*c)).abs()).fold(0.0_f64, f64::max);
    assert!(max_diff > 1e-6, "3d adiabatic: spherical == cartesian (geometry idle): {max_diff:e}");
    println!("SPHERICAL 3D ADIABATIC: {} steps, max |drho vs cart| {:.4}", sph.iteration, max_diff);
}

#[test]
fn full_substrate_spherical_2d_iso() {
    type Sph = SimState<IsoNewtonian, 2, Spherical, Isothermal<f64>, CpuSpace, HostMemory>;
    type Cart = SimState<IsoNewtonian, 2, Cartesian, Isothermal<f64>, CpuSpace, HostMemory>;
    let (nr, nt, cs) = (32usize, 12usize, 1.0_f64);
    let (r_lo, dr) = (0.5_f64, 1.0 / nr as f64);
    let dims = [nr, nt];
    let xlo = [r_lo, 0.6];
    let dx = [dr, 0.4 / nt as f64];
    let bcs = Boundaries::uniform(BoundaryType::Outflow);
    let mut sph = Sph::build(IsoNewtonian, Isothermal { cs }, Spherical)
        .cells(dims)
        .origin(xlo)
        .spacing(dx)
        .boundaries(bcs)
        .allocate()
        .expect("sph 2d iso")
        .set_initial(|_| PrimG { rho: 1.0, vel: Tensor::new([0.0, 0.0]), pre: Default::default() })
        .build();
    set_grad_iso(&mut sph, r_lo, dr);
    let sub = IsoSubstrateKernelSet::<HostMemory, f64, 2>::new(cs, 0.4, &sph.geom.allocated);
    evolve(&mut sph, &sub, 0.03).expect("2d sph iso evolve failed");
    let cells: Vec<[isize; 2]> = sph.geom.interior.iter().collect();
    for c in &cells {
        let rho = *sph.fields.prim.rho.view().at(*c);
        assert!(rho.is_finite() && rho > 0.0, "2d iso: bad density {rho} at {c:?}");
    }
    let mut cart = Cart::build(IsoNewtonian, Isothermal { cs }, Cartesian)
        .cells(dims)
        .origin(xlo)
        .spacing(dx)
        .boundaries(bcs)
        .allocate()
        .expect("cart 2d iso")
        .set_initial(|_| PrimG { rho: 1.0, vel: Tensor::new([0.0, 0.0]), pre: Default::default() })
        .build();
    set_grad_iso(&mut cart, r_lo, dr);
    evolve(&mut cart, &sub, 0.03).expect("2d cart iso evolve failed");
    let max_diff = cells.iter().map(|c| (*sph.fields.prim.rho.view().at(*c) - *cart.fields.prim.rho.view().at(*c)).abs()).fold(0.0_f64, f64::max);
    assert!(max_diff > 1e-6, "2d iso: spherical == cartesian (geometry idle): {max_diff:e}");
    println!("SPHERICAL 2D ISO: {} steps, max |drho vs cart| {:.4}", sph.iteration, max_diff);
}

#[test]
fn full_substrate_spherical_3d_iso() {
    type Sph = SimState<IsoNewtonian, 3, Spherical, Isothermal<f64>, CpuSpace, HostMemory>;
    type Cart = SimState<IsoNewtonian, 3, Cartesian, Isothermal<f64>, CpuSpace, HostMemory>;
    let cs = 1.0_f64;
    let (r_lo, dr) = (0.5_f64, 1.0 / TR as f64);
    let dims = [TR, TT, TP];
    let xlo = [r_lo, 0.6, 0.2];
    let dx = [dr, 0.4 / TT as f64, 0.5 / TP as f64];
    let bcs = Boundaries::uniform(BoundaryType::Outflow);
    let mut sph = Sph::build(IsoNewtonian, Isothermal { cs }, Spherical)
        .cells(dims)
        .origin(xlo)
        .spacing(dx)
        .boundaries(bcs)
        .allocate()
        .expect("sph 3d iso")
        .set_initial(|_| PrimG { rho: 1.0, vel: Tensor::new([0.0; 3]), pre: Default::default() })
        .build();
    set_grad_iso(&mut sph, r_lo, dr);
    let sub = IsoSubstrateKernelSet::<HostMemory, f64, 3>::new(cs, 0.4, &sph.geom.allocated);
    evolve(&mut sph, &sub, 0.02).expect("3d sph iso evolve failed");
    let cells: Vec<[isize; 3]> = sph.geom.interior.iter().collect();
    for c in &cells {
        let rho = *sph.fields.prim.rho.view().at(*c);
        assert!(rho.is_finite() && rho > 0.0, "3d iso: bad density {rho} at {c:?}");
    }
    let mut cart = Cart::build(IsoNewtonian, Isothermal { cs }, Cartesian)
        .cells(dims)
        .origin(xlo)
        .spacing(dx)
        .boundaries(bcs)
        .allocate()
        .expect("cart 3d iso")
        .set_initial(|_| PrimG { rho: 1.0, vel: Tensor::new([0.0; 3]), pre: Default::default() })
        .build();
    set_grad_iso(&mut cart, r_lo, dr);
    evolve(&mut cart, &sub, 0.02).expect("3d cart iso evolve failed");
    let max_diff = cells.iter().map(|c| (*sph.fields.prim.rho.view().at(*c) - *cart.fields.prim.rho.view().at(*c)).abs()).fold(0.0_f64, f64::max);
    assert!(max_diff > 1e-6, "3d iso: spherical == cartesian (geometry idle): {max_diff:e}");
    println!("SPHERICAL 3D ISO: {} steps, max |drho vs cart| {:.4}", sph.iteration, max_diff);
}

#[test]
fn full_substrate_spherical_2d_srhd() {
    type Sph = SimState<Srhd, 2, Spherical, IdealGas<f64>, CpuSpace, HostMemory>;
    type Cart = SimState<Srhd, 2, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;
    let (nr, nt) = (32usize, 12usize);
    let (r_lo, dr) = (0.5_f64, 1.0 / nr as f64);
    let dims = [nr, nt];
    let xlo = [r_lo, 0.6];
    let dx = [dr, 0.4 / nt as f64];
    let bcs = Boundaries::uniform(BoundaryType::Outflow);
    let mut sph = Sph::build(Srhd, IdealGas { gamma: GAMMA }, Spherical)
        .cells(dims)
        .origin(xlo)
        .spacing(dx)
        .boundaries(bcs)
        .allocate()
        .expect("sph 2d srhd")
        .set_initial(|_| Prim { rho: 1.0, vel: Tensor::new([0.0, 0.0]), pre: 1.0 })
        .build();
    set_grad_srhd(&mut sph, r_lo, dr);
    let sub = SrhdSubstrateKernelSet::<HostMemory, f64, 2>::new(GAMMA, 0.4, &sph.geom.allocated);
    evolve(&mut sph, &sub, 0.03).expect("2d sph srhd evolve failed");
    let cells: Vec<[isize; 2]> = sph.geom.interior.iter().collect();
    let pre = sph.fields.prim.pre_field().expect("prim.pre");
    for c in &cells {
        let (rho, p) = (*sph.fields.prim.rho.view().at(*c), *pre.view().at(*c));
        let v = ((*sph.fields.prim.vel[0].view().at(*c)).powi(2) + (*sph.fields.prim.vel[1].view().at(*c)).powi(2)).sqrt();
        assert!(rho.is_finite() && rho > 0.0, "2d srhd: bad density {rho} at {c:?}");
        assert!(p.is_finite() && p > 0.0, "2d srhd: bad pressure {p} at {c:?}");
        assert!(v < 1.0, "2d srhd: superluminal |v| = {v} at {c:?}");
    }
    let mut cart = Cart::build(Srhd, IdealGas { gamma: GAMMA }, Cartesian)
        .cells(dims)
        .origin(xlo)
        .spacing(dx)
        .boundaries(bcs)
        .allocate()
        .expect("cart 2d srhd")
        .set_initial(|_| Prim { rho: 1.0, vel: Tensor::new([0.0, 0.0]), pre: 1.0 })
        .build();
    set_grad_srhd(&mut cart, r_lo, dr);
    evolve(&mut cart, &sub, 0.03).expect("2d cart srhd evolve failed");
    let max_diff = cells.iter().map(|c| (*sph.fields.prim.rho.view().at(*c) - *cart.fields.prim.rho.view().at(*c)).abs()).fold(0.0_f64, f64::max);
    assert!(max_diff > 1e-6, "2d srhd: spherical == cartesian (geometry idle): {max_diff:e}");
    println!("SPHERICAL 2D SRHD: {} steps, max |drho vs cart| {:.4}", sph.iteration, max_diff);
}

#[test]
fn full_substrate_spherical_3d_srhd() {
    type Sph = SimState<Srhd, 3, Spherical, IdealGas<f64>, CpuSpace, HostMemory>;
    type Cart = SimState<Srhd, 3, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;
    let (r_lo, dr) = (0.5_f64, 1.0 / TR as f64);
    let dims = [TR, TT, TP];
    let xlo = [r_lo, 0.6, 0.2];
    let dx = [dr, 0.4 / TT as f64, 0.5 / TP as f64];
    let bcs = Boundaries::uniform(BoundaryType::Outflow);
    let mut sph = Sph::build(Srhd, IdealGas { gamma: GAMMA }, Spherical)
        .cells(dims)
        .origin(xlo)
        .spacing(dx)
        .boundaries(bcs)
        .allocate()
        .expect("sph 3d srhd")
        .set_initial(|_| Prim { rho: 1.0, vel: Tensor::new([0.0; 3]), pre: 1.0 })
        .build();
    set_grad_srhd(&mut sph, r_lo, dr);
    let sub = SrhdSubstrateKernelSet::<HostMemory, f64, 3>::new(GAMMA, 0.4, &sph.geom.allocated);
    evolve(&mut sph, &sub, 0.02).expect("3d sph srhd evolve failed");
    let cells: Vec<[isize; 3]> = sph.geom.interior.iter().collect();
    let pre = sph.fields.prim.pre_field().expect("prim.pre");
    for c in &cells {
        let (rho, p) = (*sph.fields.prim.rho.view().at(*c), *pre.view().at(*c));
        let v = ((*sph.fields.prim.vel[0].view().at(*c)).powi(2) + (*sph.fields.prim.vel[1].view().at(*c)).powi(2) + (*sph.fields.prim.vel[2].view().at(*c)).powi(2)).sqrt();
        assert!(rho.is_finite() && rho > 0.0, "3d srhd: bad density {rho} at {c:?}");
        assert!(p.is_finite() && p > 0.0, "3d srhd: bad pressure {p} at {c:?}");
        assert!(v < 1.0, "3d srhd: superluminal |v| = {v} at {c:?}");
    }
    let mut cart = Cart::build(Srhd, IdealGas { gamma: GAMMA }, Cartesian)
        .cells(dims)
        .origin(xlo)
        .spacing(dx)
        .boundaries(bcs)
        .allocate()
        .expect("cart 3d srhd")
        .set_initial(|_| Prim { rho: 1.0, vel: Tensor::new([0.0; 3]), pre: 1.0 })
        .build();
    set_grad_srhd(&mut cart, r_lo, dr);
    evolve(&mut cart, &sub, 0.02).expect("3d cart srhd evolve failed");
    let max_diff = cells.iter().map(|c| (*sph.fields.prim.rho.view().at(*c) - *cart.fields.prim.rho.view().at(*c)).abs()).fold(0.0_f64, f64::max);
    assert!(max_diff > 1e-6, "3d srhd: spherical == cartesian (geometry idle): {max_diff:e}");
    println!("SPHERICAL 3D SRHD: {} steps, max |drho vs cart| {:.4}", sph.iteration, max_diff);
}
