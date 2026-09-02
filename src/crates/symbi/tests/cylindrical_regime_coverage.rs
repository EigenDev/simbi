// =============================================================================
// cylindrical_regime_coverage.rs
//
// fills the cylindrical hydro matrix: 1D (radial), 2D (r-phi disk), 3D (r,phi,z) for
// all three EOS regimes (iso / newton-adiabatic / rhd) — the natural DOF == NDIM
// plane, dispatched through geom_suffix to the "_cyl" godunov + wave-speed (the
// c2p/flux/snapshot/ghost reuse the cartesian ncomp==NDIM instances). the geometric
// source falls out of the Geom and is shared across the three regimes
// (`GeoSource::Hydro`), so two newton physics checks prove the cyl 1D + 3D source is
// active + correctly signed, and the iso/rhd cells are NaN-free-smoke validated
// (their flux/c2p closures are already validated in cartesian/spherical).
//
//   newton: cyl-1D uniform radial outflow rarefies as rho_dot = -rho*v_r/r (the 1/r
//           area-weighted divergence); cyl-3D uniform swirl drives v_r*r ~ v0^2*t
//           (the centrifugal source, with z gridded but inert).
//   iso/rhd: each runs NaN-free + positive over a short evolve on cyl 1D/2D/3D.
// =============================================================================

use symbi::regimes::substrate::IsoSubstrateKernelSet;
use symbi::regimes::substrate_newton::AdiabaticSubstrateKernelSet;
use symbi::regimes::substrate_rhd::RhdSubstrateKernelSet;
use symbi::sim::evolve::evolve;
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_geometry::Cylindrical;
use symbi_hydro::energy::IsoModel;
use symbi_hydro::eos::{IdealGas, Isothermal};
use symbi_hydro::isothermal::IsoNewtonian;
use symbi_hydro::newtonian::Newtonian;
use symbi_hydro::quantity::{Density, Pressure};
use symbi_hydro::rhd::Rhd;
use symbi_hydro::state::{Prim, PrimG};
use symbi_xpu::{CpuSpace, HostMemory};

const GAMMA: f64 = 1.4;
const TWO_PI: f64 = std::f64::consts::TAU;

// the outflow-r, periodic-phi / periodic-z boundary set for a cyl grid: outflow on
// the radial axis (0), periodic on every transverse axis (phi, z).
fn cyl_bc<const D: usize>() -> Boundaries<D> {
    Boundaries::per_axis(std::array::from_fn(|ax| {
        if ax == 0 {
            [BoundaryType::Outflow, BoundaryType::Outflow]
        } else {
            [BoundaryType::Periodic, BoundaryType::Periodic]
        }
    }))
}

// ----- newton physics: cyl 1D radial divergence -----------------------------------

#[test]
fn cyl_1d_radial_outflow_rarefies_1_over_r() {
    // uniform rho/p with a uniform outward v_r=v0: cylindrical continuity gives
    // rho_dot = -(1/r) d(r rho v_r)/dr = -rho v_r / r (uniform). so (rho0-rho)*r ~ rho0*v0*t,
    // constant across radius — the 1/r area-weighting. a cartesian scheme leaves rho uniform.
    type Sim = SimStateGeneric<Newtonian, 1, 1, Cylindrical, IdealGas<f64>, CpuSpace, HostMemory>;
    let (nr, r_lo, r_hi) = (64usize, 1.0_f64, 2.0_f64);
    let dr = (r_hi - r_lo) / nr as f64;
    let (rho0, v0) = (1.0_f64, 0.5_f64);
    // uniform rho0/p=1 with a uniform outward v_r=v0.
    let mut sim = Sim::build(Newtonian, IdealGas { gamma: GAMMA }, Cylindrical)
        .cells([nr])
        .origin([r_lo])
        .spacing([dr])
        .boundaries(cyl_bc())
        .cfl(0.3)
        .allocate()
        .expect("cyl 1D sim construction failed")
        .set_initial(|_x| Prim::adiabatic(Density(rho0), Tensor::new([v0]), Pressure(1.0)))
        .build();

    let sub =
        AdiabaticSubstrateKernelSet::<HostMemory, f64, 1>::new(GAMMA, 0.3, &sim.geom.allocated);
    let t = 0.02;
    evolve(&mut sim, &sub, t).expect("cyl 1D evolution failed");

    let rho = &sim.fields.prim.rho;
    let mut drop_r: Vec<f64> = Vec::new();
    for c in sim.geom.interior.iter() {
        if c[0] < 6 || c[0] >= nr as isize - 6 {
            continue;
        }
        let r = r_lo + (c[0] as f64 + 0.5) * dr;
        let rho_c = *rho.view().at(c);
        assert!(
            rho_c.is_finite() && rho_c > 0.0,
            "bad rho at {c:?}: {rho_c}"
        );
        assert!(
            rho_c < rho0,
            "rho must rarefy under outflow at r={r:.3}: {rho_c:.5}"
        );
        drop_r.push((rho0 - rho_c) * r);
    }
    assert!(drop_r.len() > 8, "too few samples");
    let mean = drop_r.iter().sum::<f64>() / drop_r.len() as f64;
    let (lo, hi) = drop_r
        .iter()
        .fold((f64::MAX, f64::MIN), |(l, h), &x| (l.min(x), h.max(x)));
    assert!(
        (hi - lo) / mean < 0.15,
        "(rho0-rho)*r not radius-constant (1/r divergence): spread {:.1}%",
        100.0 * (hi - lo) / mean
    );
    let expected = rho0 * v0 * t;
    assert!(
        mean > 0.5 * expected && mean < 1.4 * expected,
        "divergence magnitude off: (rho0-rho)*r mean={mean:.4e}, expected ~rho0*v0*t={expected:.4e}"
    );
}

// ----- newton physics: cyl 3D centrifugal source ----------------------------------

#[test]
fn cyl_3d_swirl_centrifugal_outflow() {
    // uniform rho/p, uniform swirl v_phi=v0, v_r=v_z=0 on a full (r,phi,z) grid: the
    // centrifugal source drives v_r outward as v_r*r ~ v0^2*t (z gridded but inert).
    type Sim = SimStateGeneric<Newtonian, 3, 3, Cylindrical, IdealGas<f64>, CpuSpace, HostMemory>;
    let (nr, nphi, nz) = (32usize, 16usize, 4usize);
    let (r_lo, r_hi) = (1.0_f64, 2.0_f64);
    let dr = (r_hi - r_lo) / nr as f64;
    let dphi = TWO_PI / nphi as f64;
    let dz = 0.5 / nz as f64;
    let v0 = 1.0_f64;
    // uniform rho/p=1, uniform swirl v_phi=v0, v_r=v_z=0.
    let mut sim = Sim::build(Newtonian, IdealGas { gamma: GAMMA }, Cylindrical)
        .cells([nr, nphi, nz])
        .origin([r_lo, 0.0, 0.0])
        .spacing([dr, dphi, dz])
        .boundaries(cyl_bc())
        .cfl(0.3)
        .allocate()
        .expect("cyl 3D sim construction failed")
        .set_initial(|_x| Prim::adiabatic(Density(1.0), Tensor::new([0.0, v0, 0.0]), Pressure(1.0)))
        .build();

    let sub =
        AdiabaticSubstrateKernelSet::<HostMemory, f64, 3>::new(GAMMA, 0.3, &sim.geom.allocated);
    let t = 0.02;
    evolve(&mut sim, &sub, t).expect("cyl 3D evolution failed");

    let vr = &sim.fields.prim.vel[0];
    let vz = &sim.fields.prim.vel[2];
    let mut vr_r: Vec<f64> = Vec::new();
    for c in sim.geom.interior.iter() {
        if c[0] < 4 || c[0] >= nr as isize - 4 {
            continue;
        }
        let r = r_lo + (c[0] as f64 + 0.5) * dr;
        let vr_c = *vr.view().at(c);
        let vz_c = *vz.view().at(c);
        assert!(vr_c.is_finite(), "non-finite v_r at {c:?}");
        assert!(
            vr_c > 0.0,
            "v_r must be centrifugally outward at r={r:.3}: {vr_c:.3e}"
        );
        assert!(vz_c.abs() < 1e-3, "v_z must stay ~0 at {c:?}: {vz_c:.3e}");
        vr_r.push(vr_c * r);
    }
    assert!(vr_r.len() > 8, "too few samples");
    let mean = vr_r.iter().sum::<f64>() / vr_r.len() as f64;
    let (lo, hi) = vr_r
        .iter()
        .fold((f64::MAX, f64::MIN), |(l, h), &x| (l.min(x), h.max(x)));
    assert!(
        (hi - lo) / mean < 0.2,
        "v_r*r not radius-constant (3D centrifugal): spread {:.1}%",
        100.0 * (hi - lo) / mean
    );
    let expected = v0 * v0 * t;
    assert!(
        mean > 0.5 * expected && mean < 1.4 * expected,
        "3D centrifugal magnitude off: v_r*r mean={mean:.4e}, expected ~v0^2*t={expected:.4e}"
    );
}

// ----- iso + rhd: NaN-free smokes across cyl 1D/2D/3D ----------------------------

#[test]
fn cyl_iso_runs_all_dims() {
    let cs = 1.0_f64;
    // 1D
    {
        type Sim =
            SimStateGeneric<IsoNewtonian, 1, 1, Cylindrical, Isothermal<f64>, CpuSpace, HostMemory>;
        let (nr, r_lo, dr) = (48usize, 1.0_f64, 1.0 / 48.0);
        let mut sim = Sim::build(IsoNewtonian, Isothermal { cs }, Cylindrical)
            .cells([nr])
            .origin([r_lo])
            .spacing([dr])
            .boundaries(cyl_bc())
            .cfl(0.3)
            .allocate()
            .expect("iso cyl 1D ctor")
            .set_initial(|[r]| {
                let rho = 1.0 + 0.3 * (-((r - 1.5) / 0.2).powi(2)).exp();
                PrimG::<f64, 1, IsoModel>::isothermal(Density(rho), Tensor::new([0.05]))
            })
            .build();
        let sub = IsoSubstrateKernelSet::<HostMemory, f64, 1>::new(cs, 0.3, &sim.geom.allocated);
        evolve(&mut sim, &sub, 0.05).expect("iso cyl 1D evolve");
        assert_positive_finite_rho(&sim.fields.prim.rho, &sim.geom.interior, "iso cyl 1D");
    }
    // 2D r-phi
    {
        type Sim =
            SimStateGeneric<IsoNewtonian, 2, 2, Cylindrical, Isothermal<f64>, CpuSpace, HostMemory>;
        let (nr, nphi, r_lo, dr) = (24usize, 24usize, 0.6_f64, 0.8 / 24.0);
        let dphi = TWO_PI / nphi as f64;
        let mut sim = Sim::build(IsoNewtonian, Isothermal { cs }, Cylindrical)
            .cells([nr, nphi])
            .origin([r_lo, 0.0])
            .spacing([dr, dphi])
            .boundaries(cyl_bc())
            .cfl(0.3)
            .allocate()
            .expect("iso cyl 2D ctor")
            .set_initial(|[r, _phi]| {
                let rho = 1.0 + 0.2 * (-((r - 1.0) / 0.2).powi(2)).exp();
                // mild rotation v_phi = cs / sqrt(r).
                PrimG::<f64, 2, IsoModel>::isothermal(
                    Density(rho),
                    Tensor::new([0.0, cs / r.sqrt()]),
                )
            })
            .build();
        let sub = IsoSubstrateKernelSet::<HostMemory, f64, 2>::new(cs, 0.3, &sim.geom.allocated);
        evolve(&mut sim, &sub, 0.05).expect("iso cyl 2D evolve");
        assert_positive_finite_rho(&sim.fields.prim.rho, &sim.geom.interior, "iso cyl 2D");
    }
    // 3D
    {
        type Sim =
            SimStateGeneric<IsoNewtonian, 3, 3, Cylindrical, Isothermal<f64>, CpuSpace, HostMemory>;
        let (nr, nphi, nz, r_lo, dr) = (16usize, 12usize, 4usize, 1.0_f64, 1.0 / 16.0);
        let (dphi, dz) = (TWO_PI / nphi as f64, 0.5 / nz as f64);
        let mut sim = Sim::build(IsoNewtonian, Isothermal { cs }, Cylindrical)
            .cells([nr, nphi, nz])
            .origin([r_lo, 0.0, 0.0])
            .spacing([dr, dphi, dz])
            .boundaries(cyl_bc())
            .cfl(0.3)
            .allocate()
            .expect("iso cyl 3D ctor")
            .set_initial(|[r, _phi, _z]| {
                let rho = 1.0 + 0.2 * (-((r - 1.5) / 0.3).powi(2)).exp();
                PrimG::<f64, 3, IsoModel>::isothermal(Density(rho), Tensor::new([0.0, 0.3, 0.0]))
            })
            .build();
        let sub = IsoSubstrateKernelSet::<HostMemory, f64, 3>::new(cs, 0.3, &sim.geom.allocated);
        evolve(&mut sim, &sub, 0.03).expect("iso cyl 3D evolve");
        assert_positive_finite_rho(&sim.fields.prim.rho, &sim.geom.interior, "iso cyl 3D");
    }
}

#[test]
fn cyl_rhd_runs_all_dims() {
    // rest-frame rhd state with a mild density bump (v=0 -> W=1, D=rho, S=0,
    // tau=p/(gamma-1)); evolve a few steps NaN-free on cyl 1D/2D/3D.
    // 1D
    {
        type Sim = SimStateGeneric<Rhd, 1, 1, Cylindrical, IdealGas<f64>, CpuSpace, HostMemory>;
        let (nr, r_lo, dr) = (48usize, 1.0_f64, 1.0 / 48.0);
        // rest-frame (v=0, W=1): the prim -> cons map yields D=rho, S=0, tau=p/(gamma-1).
        let mut sim = Sim::build(Rhd, IdealGas { gamma: GAMMA }, Cylindrical)
            .cells([nr])
            .origin([r_lo])
            .spacing([dr])
            .boundaries(cyl_bc())
            .cfl(0.3)
            .allocate()
            .expect("rhd cyl 1D ctor")
            .set_initial(|[r]| {
                let rho = 1.0 + 0.3 * (-((r - 1.5) / 0.2).powi(2)).exp();
                Prim::adiabatic(Density(rho), Tensor::new([0.0]), Pressure(1.0))
            })
            .build();
        let sub = RhdSubstrateKernelSet::<HostMemory, f64, 1>::new(GAMMA, 0.3, &sim.geom.allocated);
        evolve(&mut sim, &sub, 0.03).expect("rhd cyl 1D evolve");
        assert_positive_finite_rho(&sim.fields.prim.rho, &sim.geom.interior, "rhd cyl 1D");
    }
    // 2D r-phi
    {
        type Sim = SimStateGeneric<Rhd, 2, 2, Cylindrical, IdealGas<f64>, CpuSpace, HostMemory>;
        let (nr, nphi, r_lo, dr) = (24usize, 24usize, 0.6_f64, 0.8 / 24.0);
        let dphi = TWO_PI / nphi as f64;
        // rest frame: p=1 (cnrg=p/(gamma-1)=1/(gamma-1)), v=0.
        let mut sim = Sim::build(Rhd, IdealGas { gamma: GAMMA }, Cylindrical)
            .cells([nr, nphi])
            .origin([r_lo, 0.0])
            .spacing([dr, dphi])
            .boundaries(cyl_bc())
            .cfl(0.3)
            .allocate()
            .expect("rhd cyl 2D ctor")
            .set_initial(|[r, _phi]| {
                let rho = 1.0 + 0.2 * (-((r - 1.0) / 0.2).powi(2)).exp();
                Prim::adiabatic(Density(rho), Tensor::new([0.0, 0.0]), Pressure(1.0))
            })
            .build();
        let sub = RhdSubstrateKernelSet::<HostMemory, f64, 2>::new(GAMMA, 0.3, &sim.geom.allocated);
        evolve(&mut sim, &sub, 0.03).expect("rhd cyl 2D evolve");
        assert_positive_finite_rho(&sim.fields.prim.rho, &sim.geom.interior, "rhd cyl 2D");
    }
    // 3D
    {
        type Sim = SimStateGeneric<Rhd, 3, 3, Cylindrical, IdealGas<f64>, CpuSpace, HostMemory>;
        let (nr, nphi, nz, r_lo, dr) = (16usize, 12usize, 4usize, 1.0_f64, 1.0 / 16.0);
        let (dphi, dz) = (TWO_PI / nphi as f64, 0.5 / nz as f64);
        // rest frame: p=1 (cnrg=1/(gamma-1)), v=0.
        let mut sim = Sim::build(Rhd, IdealGas { gamma: GAMMA }, Cylindrical)
            .cells([nr, nphi, nz])
            .origin([r_lo, 0.0, 0.0])
            .spacing([dr, dphi, dz])
            .boundaries(cyl_bc())
            .cfl(0.3)
            .allocate()
            .expect("rhd cyl 3D ctor")
            .set_initial(|[r, _phi, _z]| {
                let rho = 1.0 + 0.2 * (-((r - 1.5) / 0.3).powi(2)).exp();
                Prim::adiabatic(Density(rho), Tensor::new([0.0, 0.0, 0.0]), Pressure(1.0))
            })
            .build();
        let sub = RhdSubstrateKernelSet::<HostMemory, f64, 3>::new(GAMMA, 0.3, &sim.geom.allocated);
        evolve(&mut sim, &sub, 0.02).expect("rhd cyl 3D evolve");
        assert_positive_finite_rho(&sim.fields.prim.rho, &sim.geom.interior, "rhd cyl 3D");
    }
}

fn assert_positive_finite_rho<const D: usize>(
    rho: &symbi_grid::Field<f64, D, HostMemory>,
    interior: &symbi_algebra::Domain<D>,
    what: &str,
) {
    for c in interior.iter() {
        let v = *rho.view().at(c);
        assert!(v.is_finite() && v > 0.0, "{what}: bad rho at {c:?}: {v}");
    }
}
