// =============================================================================
// substrate_imhd_spherical.rs
//
// the curvilinear isothermal-mhd wiring proof: IsothermalMhdSubstrateKernelSet3D on
// a spherical shell through the real evolve() loop, exercising the iso _sph kernels
// — imhd_godunov_stage_sph_3d (area-weighted divergence + the isothermal geometric
// momentum source: cs^2*rho + 1/2|B|^2 pressure + inertial + lab-frame-B tension,
// GeoSource::IsothermalMhd), imhd_wave_speed_map_sph_3d, and the shared spherical CT.
//
// IC: the div-free split-monopole radial field B_r = B0/r^2, rho = 1, v = 0. the run
// must stay finite + positive; geometry is confirmed active by the spherical CFL dt
// differing from the Cartesian one.
// =============================================================================

use std::sync::atomic::Ordering;

use symbi::regimes::substrate_isothermal_mhd::IsothermalMhdSubstrateKernelSet3D;
use symbi::sim::evolve::{KernelSet, evolve};
use symbi::sim::state::*;
use symbi_geometry::{Cartesian, Spherical};
use symbi_hydro::eos::Isothermal;
use symbi_hydro::isothermal_mhd::IsothermalMhd;
use symbi_xpu::{CpuSpace, HostMemory};

type SimSph = SimState<IsothermalMhd, 3, Spherical, Isothermal<f64>, CpuSpace, HostMemory>;
type SimCart = SimState<IsothermalMhd, 3, Cartesian, Isothermal<f64>, CpuSpace, HostMemory>;

const N: usize = 8;
const R_LO: f64 = 1.0;
const DR: f64 = 0.1;
const T_LO: f64 = 0.6;
const DTH: f64 = 0.06;
const P_LO: f64 = 0.2;
const DPH: f64 = 0.07;
const CS: f64 = 1.0;
const CFL: f64 = 0.3;
const B0: f64 = 0.1;
const T_FINAL: f64 = 0.03;

fn set_ic<M>(sim: &mut SimState<IsothermalMhd, 3, M, Isothermal<f64>, CpuSpace, HostMemory>)
where
    M: symbi_geometry::Metric<f64, 3> + Copy,
{
    let mhd = sim
        .fields
        .mhd
        .as_ref()
        .expect("iso-MHD requires mhd fields");
    // face_coord(c, 0)[0] = the r-face radius — split-monopole B_r = B0/r^2 (div-free).
    for c in &sim.geom.interior.extend(0, 0, 1) {
        let rf = sim.geom.face_coord(c, 0)[0];
        mhd.bface[0].view_mut().set(c, B0 / (rf * rf));
    }
    mhd.bface_initialized.store(true, Ordering::Relaxed);
    for c in sim.geom.allocated.iter() {
        let rc = R_LO + (c[0] as f64 + 0.5) * DR;
        mhd.bcell[0].view_mut().set(c, B0 / (rc * rc));
        mhd.bcell[1].view_mut().set(c, 0.0);
        mhd.bcell[2].view_mut().set(c, 0.0);
    }
    // no energy slot: rho = 1, v = 0 -> set cons den/mom directly.
    for c in sim.geom.interior.iter() {
        sim.fields.cons.den.view_mut().set(c, 1.0);
        sim.fields.cons.mom[0].view_mut().set(c, 0.0);
        sim.fields.cons.mom[1].view_mut().set(c, 0.0);
        sim.fields.cons.mom[2].view_mut().set(c, 0.0);
    }
}

fn make_sph() -> SimSph {
    // unseeded; set_ic seeds the staggered B + bcell (full allocated domain) post-construction.
    SimSph::build(IsothermalMhd, Isothermal { cs: CS }, Spherical)
        .cells([N, N, N])
        .origin([R_LO, T_LO, P_LO])
        .spacing([DR, DTH, DPH])
        .boundaries(Boundaries::uniform(BoundaryType::Outflow))
        .cfl(CFL)
        .finish()
        .expect("spherical iso-MHD sim construction failed")
}
fn make_cart() -> SimCart {
    SimCart::build(IsothermalMhd, Isothermal { cs: CS }, Cartesian)
        .cells([N, N, N])
        .origin([R_LO, T_LO, P_LO])
        .spacing([DR, DTH, DPH])
        .boundaries(Boundaries::uniform(BoundaryType::Outflow))
        .cfl(CFL)
        .finish()
        .expect("cartesian iso-MHD sim construction failed")
}

#[test]
fn full_substrate_spherical_imhd_smoke() {
    let sub = IsothermalMhdSubstrateKernelSet3D::<HostMemory>::new(
        CS,
        CFL,
        1.0,
        &make_sph().geom.allocated,
    );

    let mut sph = make_sph();
    set_ic(&mut sph);
    assert_eq!(
        sph.geom.coords,
        symbi_geometry::Geometry::Spherical,
        "coords must be Spherical"
    );

    // geometry active in the wave-speed map: the spherical CFL dt differs from Cartesian.
    let mut cart = make_cart();
    set_ic(&mut cart);
    sub.c2p(&sph);
    sub.c2p(&cart);
    let dt_sph = sub.cfl(&sph);
    let dt_cart = sub.cfl(&cart);
    assert!(
        dt_sph.is_finite() && dt_sph > 0.0,
        "bad spherical dt {dt_sph}"
    );
    assert!(
        (dt_sph - dt_cart).abs() > 1e-9,
        "spherical and cartesian CFL dt identical ({dt_sph} vs {dt_cart}) — wave_speed_map_sph did not engage",
    );

    // the full curvilinear evolve exercises imhd_godunov_stage_sph_3d (the isothermal
    // geometric source) + the spherical CT.
    evolve(&mut sph, &sub, T_FINAL).expect("spherical iso-MHD evolution failed");

    let mhd = sph.fields.mhd.as_ref().expect("mhd fields");
    for c in sph.geom.interior.iter() {
        let rho = *sph.fields.prim.rho.view().at(c);
        assert!(rho.is_finite() && rho > 0.0, "bad density {rho} at {c:?}");
        for d in 0..3 {
            assert!(
                sph.fields.prim.vel[d].view().at(c).is_finite(),
                "non-finite vel[{d}] at {c:?}"
            );
            assert!(
                mhd.bcell[d].view().at(c).is_finite(),
                "non-finite bcell[{d}] at {c:?}"
            );
        }
    }
    assert!(sph.iteration > 0, "no steps taken");
    println!(
        "SPHERICAL iso-MHD smoke: {} steps to t={:.4}, dt_sph {:.5} vs dt_cart {:.5}",
        sph.iteration, sph.time, dt_sph, dt_cart,
    );
}
