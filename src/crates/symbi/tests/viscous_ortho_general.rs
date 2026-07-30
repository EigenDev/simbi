// =============================================================================
// viscous_ortho_general.rs
//
// the 2.5D and 3D orthogonal viscous family through the production dispatch:
// - the (r, z) axisymmetric swirl (cylindrical D=2 DOF=3, out-of-plane phi
//   with h3 = r): rigid rotation v_phi = Omega r is a discrete null, a sheared
//   profile acts and heats;
// - the spherical (r, theta) meridian swirl (out-of-plane phi, h3 = r sin
//   theta): rigid rotation v_phi = Omega r sin(theta) is a null;
// - full 3D cylindrical: rigid rotation nulls through the 3D chart operator;
// - MHD on the cylindrical (r, phi) disk: viscosity diffuses the gas momentum
//   and books heating while every staggered B face stays bit-untouched.
// prims are point samples at arithmetic centers vs the operator's volumetric
// centroids, so the production nulls hold to truncation order (the carrier
// nulls are exact and gated in symbi-hydro); thresholds sit orders below the
// physical stress scale and orders above the centroid-sampling floor.
// =============================================================================

use std::f64::consts::PI;

use symbi::regimes::substrate_newton::AdiabaticSubstrateKernelSet;
use symbi::regimes::substrate_newtonian_mhd::NewtonianMhdSubstrateKernelSet;
use symbi::sim::state::*;
use symbi::sim::substrate_seam::{KernelSet, WithViscosity};
use symbi_algebra::Tensor;
use symbi_geometry::{Cylindrical, Spherical};
use symbi_hydro::eos::IdealGas;
use symbi_hydro::mhd_state::MhdPrim;
use symbi_hydro::newtonian::Newtonian;
use symbi_hydro::newtonian_mhd::NewtonianMhd;
use symbi_hydro::state::Prim;
use symbi_xpu::{CpuSpace, HostMemory};

const GAMMA: f64 = 1.4;
const CFL: f64 = 0.3;
const NU: f64 = 0.05;
const DT: f64 = 1.0e-4;

#[test]
fn rz_swirl_rigid_rotation_nulls_and_shear_acts() {
    type Sim = SimStateGeneric<Newtonian, 2, 3, Cylindrical, IdealGas<f64>, CpuSpace, HostMemory>;
    type Kern = AdiabaticSubstrateKernelSet<HostMemory, f64, 2>;
    let (nr, nz) = (32usize, 16usize);
    let (r_lo, dr) = (1.0f64, 2.0 / nr as f64);
    let dz = 0.5 / nz as f64;
    let run = |vphi: &dyn Fn(f64) -> f64| -> (f64, f64) {
        let sim = Sim::build(Newtonian, IdealGas { gamma: GAMMA }, Cylindrical)
            .cells([nr, nz])
            .origin([r_lo, 0.0])
            .spacing([dr, dz])
            .boundaries(Boundaries::per_axis([
                [BoundaryType::Outflow, BoundaryType::Outflow],
                [BoundaryType::Periodic, BoundaryType::Periodic],
            ]))
            .cfl(CFL)
            .allocate()
            .expect("rz sim")
            .set_initial(|x| Prim {
                rho: 1.0,
                vel: Tensor::new([0.0, vphi(x[0]), 0.0]),
                pre: 1.0,
            })
            .build();
        let k = Kern::new(GAMMA, CFL, &sim.geom.allocated).with_viscosity(NU);
        k.c2p(&sim);
        k.ghost_fill(&sim);
        let grab = |c: usize| -> Vec<f64> {
            sim.geom
                .interior
                .iter()
                .map(|cc| *sim.fields.cons.mom[c].view().at(cc))
                .collect()
        };
        let nrg_b: Vec<f64> = sim
            .geom
            .interior
            .iter()
            .map(|c| *sim.fields.cons.nrg_field().unwrap().view().at(c))
            .collect();
        let before = [grab(0), grab(1), grab(2)];
        k.viscous(&sim, DT);
        let after = [grab(0), grab(1), grab(2)];
        let nrg_a: Vec<f64> = sim
            .geom
            .interior
            .iter()
            .map(|c| *sim.fields.cons.nrg_field().unwrap().view().at(c))
            .collect();
        // z is periodic and the profile is z-uniform, so only the radial edges see
        // outflow-ghost contamination: trim two r-rings, no z-rows. layout is
        // last-axis-fastest (z fastest), so ring i occupies [i*nz, (i+1)*nz).
        let trim = |v: &[f64], w: &[f64]| -> f64 {
            v.iter()
                .zip(w)
                .enumerate()
                .filter(|(idx, _)| {
                    let ring = idx / nz;
                    (2..nr - 2).contains(&ring)
                })
                .map(|(_, (x, y))| (x - y).abs())
                .fold(0.0_f64, f64::max)
        };
        let dm = before
            .iter()
            .zip(&after)
            .map(|(b, a)| trim(b, a))
            .fold(0.0_f64, f64::max);
        let de = trim(&nrg_b, &nrg_a);
        (dm, de)
    };
    let omega = 0.4;
    let (dm_rigid, _) = run(&|r| omega * r);
    assert!(
        dm_rigid < 1e-9,
        "rz rigid rotation produced a viscous force: {dm_rigid:e}"
    );
    let (dm_shear, de_shear) = run(&|r| 0.5 / r);
    assert!(dm_shear > 1e-12, "rz shear never acted");
    assert!(de_shear > 1e-14, "rz shear booked no heating");
}

#[test]
fn spherical_meridian_rigid_rotation_nulls() {
    type Sim = SimStateGeneric<Newtonian, 2, 3, Spherical, IdealGas<f64>, CpuSpace, HostMemory>;
    type Kern = AdiabaticSubstrateKernelSet<HostMemory, f64, 2>;
    let (nrr, nt) = (32usize, 24usize);
    let (r_lo, dr) = (1.0f64, 2.0 / nrr as f64);
    let (t_lo, dth) = (0.5f64, (PI - 1.0) / nt as f64);
    let omega = 0.4;
    let sim = Sim::build(Newtonian, IdealGas { gamma: GAMMA }, Spherical)
        .cells([nrr, nt])
        .origin([r_lo, t_lo])
        .spacing([dr, dth])
        .boundaries(Boundaries::uniform(BoundaryType::Outflow))
        .cfl(CFL)
        .allocate()
        .expect("sph sim")
        .set_initial(|x| Prim {
            rho: 1.0,
            vel: Tensor::new([0.0, 0.0, omega * x[0] * x[1].sin()]),
            pre: 1.0,
        })
        .build();
    let k = Kern::new(GAMMA, CFL, &sim.geom.allocated).with_viscosity(NU);
    // no c2p / ghost_fill: the adiabatic spherical-swirl gas pipeline is not
    // baked (this gate exercises the VISCOUS kernel only); set_initial seeds
    // the prims, and the zeroed ghost bands are excluded by the 2-cell trim.
    let grab = |c: usize| -> Vec<f64> {
        sim.geom
            .interior
            .iter()
            .map(|cc| *sim.fields.cons.mom[c].view().at(cc))
            .collect()
    };
    let before = [grab(0), grab(1), grab(2)];
    k.viscous(&sim, DT);
    let after = [grab(0), grab(1), grab(2)];
    // both axes are outflow: trim two cells from each edge (theta fastest).
    let trim = |v: &[f64], w: &[f64]| -> f64 {
        v.iter()
            .zip(w)
            .enumerate()
            .filter(|(idx, _)| {
                let (ring, th) = (idx / nt, idx % nt);
                (2..nrr - 2).contains(&ring) && (2..nt - 2).contains(&th)
            })
            .map(|(_, (x, y))| (x - y).abs())
            .fold(0.0_f64, f64::max)
    };
    let dm = before
        .iter()
        .zip(&after)
        .map(|(b, a)| trim(b, a))
        .fold(0.0_f64, f64::max);
    assert!(
        dm < 1e-9,
        "spherical-meridian rigid rotation produced a force: {dm:e}"
    );
}

#[test]
fn cylindrical_3d_rigid_rotation_nulls() {
    type Sim = SimState<Newtonian, 3, Cylindrical, IdealGas<f64>, CpuSpace, HostMemory>;
    type Kern = AdiabaticSubstrateKernelSet<HostMemory, f64, 3>;
    let (nr, np, nz) = (24usize, 12usize, 8usize);
    let (r_lo, dr) = (1.0f64, 2.0 / nr as f64);
    let (dp, dz) = (2.0 * PI / np as f64, 0.5 / nz as f64);
    let omega = 0.4;
    let sim = Sim::build(Newtonian, IdealGas { gamma: GAMMA }, Cylindrical)
        .cells([nr, np, nz])
        .origin([r_lo, 0.0, 0.0])
        .spacing([dr, dp, dz])
        .boundaries(Boundaries::per_axis([
            [BoundaryType::Outflow, BoundaryType::Outflow],
            [BoundaryType::Periodic, BoundaryType::Periodic],
            [BoundaryType::Periodic, BoundaryType::Periodic],
        ]))
        .cfl(CFL)
        .allocate()
        .expect("cyl3 sim")
        .set_initial(|x| Prim {
            rho: 1.0,
            vel: Tensor::new([0.0, omega * x[0], 0.0]),
            pre: 1.0,
        })
        .build();
    let k = Kern::new(GAMMA, CFL, &sim.geom.allocated).with_viscosity(NU);
    k.c2p(&sim);
    k.ghost_fill(&sim);
    let grab = |c: usize| -> Vec<f64> {
        sim.geom
            .interior
            .iter()
            .map(|cc| *sim.fields.cons.mom[c].view().at(cc))
            .collect()
    };
    let before = [grab(0), grab(1), grab(2)];
    k.viscous(&sim, DT);
    let after = [grab(0), grab(1), grab(2)];
    // only r is outflow; phi and z are periodic and profile-exact. layout is
    // last-axis-fastest: ring i spans [i*np*nz, (i+1)*np*nz).
    let slab = np * nz;
    let trim = |v: &[f64], w: &[f64]| -> f64 {
        v.iter()
            .zip(w)
            .enumerate()
            .filter(|(idx, _)| (2..nr - 2).contains(&(idx / slab)))
            .map(|(_, (x, y))| (x - y).abs())
            .fold(0.0_f64, f64::max)
    };
    let dm = before
        .iter()
        .zip(&after)
        .map(|(b, a)| trim(b, a))
        .fold(0.0_f64, f64::max);
    assert!(
        dm < 1e-9,
        "cylindrical 3d rigid rotation produced a force: {dm:e}"
    );
}

#[test]
fn mhd_rphi_disk_viscosity_diffuses_gas_and_leaves_b_untouched() {
    type Sim =
        SimStateGeneric<NewtonianMhd, 2, 3, Cylindrical, IdealGas<f64>, CpuSpace, HostMemory>;
    let (nr, np) = (24usize, 24usize);
    let (r_lo, dr) = (1.0f64, 2.0 / nr as f64);
    let dp = 2.0 * PI / np as f64;
    let mut sim = Sim::build(NewtonianMhd, IdealGas { gamma: GAMMA }, Cylindrical)
        .cells([nr, np])
        .origin([r_lo, 0.0])
        .spacing([dr, dp])
        .cyl_plane(CylPlane::RPhi)
        .boundaries(Boundaries::per_axis([
            [BoundaryType::Outflow, BoundaryType::Outflow],
            [BoundaryType::Periodic, BoundaryType::Periodic],
        ]))
        .cfl(CFL)
        .allocate()
        .expect("mhd disk")
        .set_initial(|x| MhdPrim {
            hydro: Prim {
                rho: 1.0,
                vel: Tensor::new([0.0, 0.5 / x[0], 0.0]),
                pre: 1.0,
            },
            mag: Tensor::new([0.0, 0.0, 0.1]),
        })
        .seed_faces(|axis, _| if axis == 2 { 0.1 } else { 0.0 })
        .build();
    let _ = &mut sim;
    let k = NewtonianMhdSubstrateKernelSet::<HostMemory, f64, 2>::new(
        GAMMA,
        CFL,
        1.0,
        &sim.geom.allocated,
    )
    .with_viscosity(NU);
    k.c2p(&sim);
    k.ghost_fill(&sim);
    let mhd = sim.fields.mhd.as_ref().expect("mhd");
    // 2.5D CT: two staggered in-plane faces + the cell-centered out-of-plane B.
    let b_before: Vec<u64> = (0..2)
        .flat_map(|a| {
            sim.geom
                .interior
                .iter()
                .map(move |c| mhd.bface[a].view().at(c).to_bits())
                .collect::<Vec<_>>()
        })
        .chain(
            sim.geom
                .interior
                .iter()
                .map(|c| mhd.bcell[2].view().at(c).to_bits()),
        )
        .collect();
    let m1_before: Vec<f64> = sim
        .geom
        .interior
        .iter()
        .map(|c| *sim.fields.cons.mom[1].view().at(c))
        .collect();
    let nrg_before: Vec<f64> = sim
        .geom
        .interior
        .iter()
        .map(|c| *sim.fields.cons.nrg_field().unwrap().view().at(c))
        .collect();
    k.viscous(&sim, DT);
    let b_after: Vec<u64> = (0..2)
        .flat_map(|a| {
            sim.geom
                .interior
                .iter()
                .map(move |c| mhd.bface[a].view().at(c).to_bits())
                .collect::<Vec<_>>()
        })
        .chain(
            sim.geom
                .interior
                .iter()
                .map(|c| mhd.bcell[2].view().at(c).to_bits()),
        )
        .collect();
    assert_eq!(b_before, b_after, "viscosity touched the staggered B field");
    let m1_after: Vec<f64> = sim
        .geom
        .interior
        .iter()
        .map(|c| *sim.fields.cons.mom[1].view().at(c))
        .collect();
    let nrg_after: Vec<f64> = sim
        .geom
        .interior
        .iter()
        .map(|c| *sim.fields.cons.nrg_field().unwrap().view().at(c))
        .collect();
    let dm = m1_before
        .iter()
        .zip(&m1_after)
        .map(|(b, a)| (b - a).abs())
        .fold(0.0_f64, f64::max);
    let de = nrg_before
        .iter()
        .zip(&nrg_after)
        .map(|(b, a)| (b - a).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        dm > 1e-12,
        "mhd disk viscosity never touched the sheared momentum"
    );
    assert!(de > 1e-14, "mhd disk viscosity booked no heating");
}
