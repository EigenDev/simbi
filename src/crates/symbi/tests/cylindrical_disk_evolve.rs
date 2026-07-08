// =============================================================================
// cylindrical_disk_evolve.rs
//
// the cylindrical r-phi DISK plane on the substrate (docs/design/20): a 2-axis
// (r, phi) grid carrying a 2-component velocity (v_r, v_phi) — DOF == NDIM == 2,
// the NATURAL cylindrical plane. distinct from the r-z axisymmetric family
// (cylindrical_swirl_evolve.rs, DOF=3 > NDIM=2): r-phi -> "_cyl" kernels, r-z ->
// "_cyl_rz", discriminated by DOF-vs-ndim at dispatch (geom_suffix).
//
// because DOF == NDIM the flux / c2p / snapshot / ghost reuse the cartesian ncomp=2
// instances; only the godunov (area-weighted r-phi divergence + centrifugal/coriolis
// inertial source) and the CFL wave-speed map (physical r\cdot dphi widths) are the "_cyl"
// instances. this is the disk-evolve hydro the immersed bodies ride.
//
// three checks, increasing in what they exercise:
//   1. the r-phi GEOMETRIC SOURCE (no body): a uniform swirl develops the 1/r
//      centrifugal radial velocity — proves the godunov's curvature source is active
//      on the r-phi plane (mirrors the r-z swirl test, but phi is now gridded).
//   2. a KEPLERIAN DISK HOLDS around a central point mass (the kepler core): with
//      v_phi balanced against the softened gravity, the disk stays radially steady
//      (v_r small, rho/v_phi bounded) over a fraction of an orbit.
//   3. an ORBITING OVERDENSITY advects in phi at the local Omega(r) without radial
//      drift — exercises the phi-flux + area-weighted phi-divergence (real azimuthal
//      transport, the thing a disk does).
// =============================================================================

use symbi::regimes::substrate_newton::AdiabaticSubstrateKernelSet;
use symbi::sim::evolve::evolve;
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_geometry::Cylindrical;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::newtonian::Newtonian;
use symbi_ib::{Body, BodyCollection};
use symbi_xpu::{CpuSpace, HostMemory};

const GAMMA: f64 = 1.4;
const TWO_PI: f64 = std::f64::consts::TAU;

// a 2-component velocity (v_r, v_phi) on a 2-axis (r, phi) grid: DOF == NDIM == 2.
type DiskSim = SimStateGeneric<Newtonian, 2, 2, Cylindrical, IdealGas<f64>, CpuSpace, HostMemory>;

// softened-Keplerian rotation: v_phi balancing |g| = mass\cdot r/(r^2+soft^2)^1.5 against the
// centrifugal v_phi^2/r -> v_phi = r\cdot sqrt(mass)/(r^2+soft^2)^0.75. soft=0 gives sqrt(mass/r).
fn v_kepler(r: f64, mass: f64, soft: f64) -> f64 {
    r * mass.sqrt() / (r * r + soft * soft).powf(0.75)
}

// build an r-phi annulus [r_lo, r_hi] x full 2*pi (phi periodic), nr x nphi cells.
fn disk_sim(nr: usize, nphi: usize, r_lo: f64, r_hi: f64) -> (DiskSim, f64, f64) {
    let dr = (r_hi - r_lo) / nr as f64;
    let dphi = TWO_PI / nphi as f64;
    // unseeded sim; each test seeds its own IC (uniform swirl / keplerian / orbiting blob),
    // some after chaining `.with_bodies`. `.finish()` is the fluent unseeded ctor.
    let sim = DiskSim::build(Newtonian, IdealGas { gamma: GAMMA }, Cylindrical)
        .cells([nr, nphi]).origin([r_lo, 0.0]).spacing([dr, dphi])
        // r: outflow (disk edge); phi: periodic (full 2*pi disk wraps).
        .boundaries(Boundaries::per_axis([
            [BoundaryType::Outflow, BoundaryType::Outflow],
            [BoundaryType::Periodic, BoundaryType::Periodic],
        ]))
        .cfl(0.3)
        .finish()
        .expect("cylindrical r-phi disk sim construction failed");
    (sim, dr, dphi)
}

#[test]
fn rphi_centrifugal_source_holds_1_over_r() {
    // uniform swirl: rho=1, p=1, v_phi=v0 const, v_r=0. axisymmetric (phi-uniform), so the
    // phi-fluxes vanish and the only radial force is centrifugal: d v_r/dt = v_phi^2/r ->
    // v_r\cdot r = v0^2\cdot t constant across r. a cartesian (no-source) scheme leaves v_r=0.
    let (nr, nphi) = (48usize, 16usize);
    let (r_lo, r_hi) = (1.0_f64, 2.0_f64);
    let (mut sim, dr, _dphi) = disk_sim(nr, nphi, r_lo, r_hi);
    let v0 = 1.0_f64;

    let cnrg = sim.fields.cons.nrg_field().expect("Newtonian cons.nrg");
    for c in sim.geom.interior.iter() {
        sim.fields.cons.den.view_mut().set(c, 1.0);
        sim.fields.cons.mom[0].view_mut().set(c, 0.0); // rho * v_r
        sim.fields.cons.mom[1].view_mut().set(c, v0);  // rho * v_phi
        cnrg.view_mut().set(c, 1.0 / (GAMMA - 1.0) + 0.5 * v0 * v0);
    }

    let sub = AdiabaticSubstrateKernelSet::<HostMemory, f64, 2>::new(GAMMA, 0.3, &sim.geom.allocated);
    let t_final = 0.02;
    evolve(&mut sim, &sub, t_final).expect("r-phi centrifugal evolution failed");

    let vr = &sim.fields.prim.vel[0];
    let vphi = &sim.fields.prim.vel[1];
    let rho = &sim.fields.prim.rho;

    let mut vr_times_r: Vec<f64> = Vec::new();
    for c in sim.geom.interior.iter() {
        if c[0] < 4 || c[0] >= nr as isize - 4 {
            continue; // away from the r-walls
        }
        let r = r_lo + (c[0] as f64 + 0.5) * dr;
        let vr_c = *vr.view().at(c);
        let vphi_c = *vphi.view().at(c);
        let rho_c = *rho.view().at(c);
        assert!(vr_c.is_finite() && vphi_c.is_finite() && rho_c.is_finite(),
            "non-finite at {c:?}: vr={vr_c} vphi={vphi_c} rho={rho_c}");
        assert!(vr_c > 0.0, "v_r must be outward (centrifugal) at r={r:.3}, got {vr_c:.3e}");
        assert!((vphi_c - v0).abs() < 0.1 * v0, "v_phi drifted at r={r:.3}: {vphi_c:.4}");
        vr_times_r.push(vr_c * r);
    }
    assert!(vr_times_r.len() > 8, "too few interior samples");

    // the 1/r signature: v_r\cdot r is radius-constant.
    let mean = vr_times_r.iter().sum::<f64>() / vr_times_r.len() as f64;
    let (lo, hi) = vr_times_r.iter().fold((f64::MAX, f64::MIN), |(l, h), &x| (l.min(x), h.max(x)));
    assert!((hi - lo) / mean < 0.15,
        "v_r·r not radius-constant (1/r centrifugal signature): spread {:.1}% (lo={lo:.4e} hi={hi:.4e})",
        100.0 * (hi - lo) / mean);
    // magnitude v_r\cdot r ~ v0^2\cdot t.
    let expected = v0 * v0 * t_final;
    assert!(mean > 0.5 * expected && mean < 1.4 * expected,
        "centrifugal magnitude off: v_r·r mean = {mean:.4e}, expected ~ v0²·t = {expected:.4e}");
}

#[test]
fn keplerian_disk_holds_around_central_mass() {
    // the kepler core: a centrifugally-supported disk around a fixed central point mass.
    // v_phi = v_kepler balances the softened gravity exactly, p uniform (no pressure force),
    // so the disk is in radial equilibrium and STAYS PUT (v_r ~ 0) as the gas orbits.
    let (nr, nphi) = (40usize, 32usize);
    let (r_lo, r_hi) = (0.5_f64, 1.5_f64);
    let (mass, soft) = (1.0_f64, 0.1_f64);
    let (sim, dr, _dphi) = disk_sim(nr, nphi, r_lo, r_hi);
    // a fixed GRAVITATIONAL central mass at the disk origin (no accretion -> disk not drained).
    // body position is ndim-D cartesian in the grid plane (r-phi -> x-y); origin -> [0, 0].
    let mut sim = sim.with_bodies(
        BodyCollection::new().add(Body::gravitational(
            0,
            Tensor::new([0.0, 0.0]),
            Tensor::zeros(),
            mass,
            0.05, // radius (unused by gravity)
            soft,
        )),
    );
    assert!(sim.has_bodies(), "with_bodies must register the central mass");

    let p0 = 1.0_f64; // uniform p (thick disk): no radial pressure force, robust c2p under keplerian shear
    let cnrg = sim.fields.cons.nrg_field().expect("Newtonian cons.nrg");
    for c in sim.geom.interior.iter() {
        let r = r_lo + (c[0] as f64 + 0.5) * dr;
        let vphi = v_kepler(r, mass, soft);
        sim.fields.cons.den.view_mut().set(c, 1.0);
        sim.fields.cons.mom[0].view_mut().set(c, 0.0); // v_r = 0
        sim.fields.cons.mom[1].view_mut().set(c, vphi); // rho * v_phi (rho=1)
        cnrg.view_mut().set(c, p0 / (GAMMA - 1.0) + 0.5 * vphi * vphi);
    }

    let sub = AdiabaticSubstrateKernelSet::<HostMemory, f64, 2>::new(GAMMA, 0.3, &sim.geom.allocated);
    // a fraction of an orbit (T ~ 2*pi/Omega ~ 6.3 at r=1); long enough that an unbalanced
    // disk would visibly drift radially or blow up.
    evolve(&mut sim, &sub, 0.5).expect("keplerian disk evolution failed");

    let vr = &sim.fields.prim.vel[0];
    let vphi = &sim.fields.prim.vel[1];
    let rho = &sim.fields.prim.rho;
    let mut checked = 0usize;
    for c in sim.geom.interior.iter() {
        if c[0] < 4 || c[0] >= nr as isize - 4 {
            continue; // away from the r-walls
        }
        let r = r_lo + (c[0] as f64 + 0.5) * dr;
        let vr_c = *vr.view().at(c);
        let vphi_c = *vphi.view().at(c);
        let rho_c = *rho.view().at(c);
        let vk = v_kepler(r, mass, soft);
        assert!(vr_c.is_finite() && vphi_c.is_finite() && rho_c.is_finite(),
            "non-finite at {c:?}: vr={vr_c} vphi={vphi_c} rho={rho_c}");
        assert!(rho_c > 0.3 && rho_c < 3.0, "density ran away at r={r:.3}: {rho_c:.4}");
        // the disk holds radially: v_r stays small vs the orbital speed.
        assert!(vr_c.abs() < 0.15 * vk, "disk not radially steady at r={r:.3}: v_r={vr_c:.3e} vs v_k={vk:.3e}");
        // v_phi stays near keplerian.
        assert!((vphi_c - vk).abs() < 0.15 * vk, "v_phi drifted from keplerian at r={r:.3}: {vphi_c:.4} vs {vk:.4}");
        checked += 1;
    }
    assert!(checked > 50, "too few interior samples checked: {checked}");
}

#[test]
fn orbiting_overdensity_advects_in_phi() {
    // azimuthal transport: a density blob at (r~1, phi~pi) with keplerian v_phi orbits at the
    // local Omega(r) = v_phi/r WITHOUT radial drift (p uniform -> centrifugal-gravity balance is
    // rho-independent). after t the blob's phi-centroid shifts by ~Omega*t; r-centroid stays put.
    // this exercises the phi-flux + area-weighted phi-divergence — real disk dynamics.
    let (nr, nphi) = (24usize, 64usize);
    let (r_lo, r_hi) = (0.6_f64, 1.4_f64);
    let (mass, soft) = (1.0_f64, 0.1_f64);
    let (sim, dr, dphi) = disk_sim(nr, nphi, r_lo, r_hi);
    let mut sim = sim.with_bodies(
        BodyCollection::new().add(Body::gravitational(
            0, Tensor::new([0.0, 0.0]), Tensor::zeros(), mass, 0.05, soft,
        )),
    );

    let p0 = 1.0_f64;
    let (r_blob, phi_blob) = (1.0_f64, std::f64::consts::PI);
    let (sig_r, sig_phi) = (0.12_f64, 0.35_f64);
    let cnrg = sim.fields.cons.nrg_field().expect("Newtonian cons.nrg");
    for c in sim.geom.interior.iter() {
        let r = r_lo + (c[0] as f64 + 0.5) * dr;
        let phi = (c[1] as f64 + 0.5) * dphi;
        let vphi = v_kepler(r, mass, soft);
        // gaussian overdensity centred at (r_blob, phi_blob); base density 1.
        let dr_b = (r - r_blob) / sig_r;
        let dphi_b = (phi - phi_blob) / sig_phi;
        let den = 1.0 + 1.5 * (-0.5 * (dr_b * dr_b + dphi_b * dphi_b)).exp();
        sim.fields.cons.den.view_mut().set(c, den);
        sim.fields.cons.mom[0].view_mut().set(c, 0.0);
        sim.fields.cons.mom[1].view_mut().set(c, den * vphi); // rho * v_phi
        cnrg.view_mut().set(c, p0 / (GAMMA - 1.0) + 0.5 * den * vphi * vphi);
    }

    // the analytic angular advance at the blob radius.
    let omega = v_kepler(r_blob, mass, soft) / r_blob;
    let t_final = 0.5_f64;
    let expected_dphi = omega * t_final;

    let sub = AdiabaticSubstrateKernelSet::<HostMemory, f64, 2>::new(GAMMA, 0.3, &sim.geom.allocated);
    evolve(&mut sim, &sub, t_final).expect("orbiting overdensity evolution failed");

    // density-weighted centroid of the OVERDENSITY (den - 1, the perturbation) in (r, phi).
    let rho = &sim.fields.prim.rho;
    let (mut wsum, mut r_cen, mut sin_acc, mut cos_acc) = (0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64);
    for c in sim.geom.interior.iter() {
        let r = r_lo + (c[0] as f64 + 0.5) * dr;
        let phi = (c[1] as f64 + 0.5) * dphi;
        let rho_c = *rho.view().at(c);
        assert!(rho_c.is_finite() && rho_c > 0.0, "non-finite/neg density at {c:?}: {rho_c}");
        let w = (rho_c - 1.0).max(0.0); // perturbation weight
        wsum += w;
        r_cen += w * r;
        // circular-mean accumulators for phi (handles the 2*pi wrap).
        sin_acc += w * phi.sin();
        cos_acc += w * phi.cos();
    }
    assert!(wsum > 0.0, "overdensity vanished");
    r_cen /= wsum;
    let phi_cen = sin_acc.atan2(cos_acc).rem_euclid(TWO_PI);

    // the blob advected azimuthally by ~Omega*t.
    let got_dphi = (phi_cen - phi_blob).rem_euclid(TWO_PI);
    assert!((got_dphi - expected_dphi).abs() < 0.3 * expected_dphi,
        "phi-advection off: blob moved {got_dphi:.4} rad, expected ~Omega*t = {expected_dphi:.4}");
    // no radial drift: the blob stays near its launch radius.
    assert!((r_cen - r_blob).abs() < 0.08,
        "blob drifted radially: r_centroid={r_cen:.4} vs launch r={r_blob:.4}");
}
