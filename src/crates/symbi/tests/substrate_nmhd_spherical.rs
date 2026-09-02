// =============================================================================
// substrate_nmhd_spherical.rs
//
// the curvilinear Newtonian-MHD wiring proof: NewtonianMhdSubstrateKernelSet3D on
// a spherical shell through the real evolve() loop, exercising the NMHD _sph
// kernels — nmhd_godunov_stage_sph_3d (the area-weighted divergence + the newtonian
// geometric momentum source: pressure + inertial + magnetic tension via the
// lab-frame B, GeoSource::NewtonianMhd), nmhd_wave_speed_map_sph_3d (per-cell
// physical CFL widths), and the spherical CT curl (rmhd_ct_curl_3d_*_sph, shared).
//
// IC: the div-free split-monopole radial field B_r = B0/r^2 (area-weighted div
// telescopes to 0) over a smooth radial pressure bump, v = 0. the run must stay
// finite + positive — a wrong source buffer/scalar order NaNs. geometry is
// confirmed active by the spherical CFL dt differing from the Cartesian one.
// =============================================================================

use std::sync::atomic::Ordering;
use symbi_hydro::quantity::{Density, EnergyDensity, Pressure};

use symbi::regimes::substrate_newtonian_mhd::NewtonianMhdSubstrateKernelSet3D;
use symbi::sim::evolve::{KernelSet, evolve};
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_geometry::{Cartesian, Spherical};
use symbi_hydro::eos::IdealGas;
use symbi_hydro::mhd_state::MhdPrim;
use symbi_hydro::newtonian_mhd::NewtonianMhd;
use symbi_hydro::regime::Regime;
use symbi_hydro::state::{Cons, Prim};
use symbi_xpu::{CpuSpace, HostMemory};

type SimSph = SimState<NewtonianMhd, 3, Spherical, IdealGas<f64>, CpuSpace, HostMemory>;
type SimCart = SimState<NewtonianMhd, 3, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;

const N: usize = 8;
const R_LO: f64 = 1.0;
const DR: f64 = 0.1;
const T_LO: f64 = 0.6;
const DTH: f64 = 0.06;
const P_LO: f64 = 0.2;
const DPH: f64 = 0.07;
const GAMMA: f64 = 5.0 / 3.0;
const CFL: f64 = 0.3;
const B0: f64 = 0.1;
const T_FINAL: f64 = 0.03;

fn set_ic<M>(sim: &mut SimState<NewtonianMhd, 3, M, IdealGas<f64>, CpuSpace, HostMemory>)
where
    M: symbi_geometry::Metric<f64, 3> + Copy,
{
    let mhd = sim.fields.mhd.as_ref().expect("NMHD requires mhd fields");
    // face_coord(c, 0)[0] = the r-face radius — the one staggered-coordinate accessor.
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
    for c in sim.geom.interior.iter() {
        let r = R_LO + (c[0] as f64 + 0.5) * DR;
        let bc = B0 / (r * r);
        let pre = 1.0 + 0.3 * (-((r - 1.4) / 0.2).powi(2)).exp();
        let prim = MhdPrim::new(
            Prim::adiabatic(Density(1.0), Tensor::new([0.0, 0.0, 0.0]), Pressure(pre)),
            Tensor::new([bc, 0.0, 0.0]),
        );
        let cons = sim.physics.regime.to_conserved(&sim.physics.eos, &prim);
        sim.fields.cons.scatter(
            c,
            Cons::adiabatic(Density(cons.den()), *cons.mom(), EnergyDensity(cons.nrg())),
        );
    }
}

fn make_sph() -> SimSph {
    // unseeded; set_ic seeds the staggered B + bcell (full allocated domain) post-construction.
    SimSph::build(NewtonianMhd, IdealGas { gamma: GAMMA }, Spherical)
        .cells([N, N, N])
        .origin([R_LO, T_LO, P_LO])
        .spacing([DR, DTH, DPH])
        .boundaries(Boundaries::uniform(BoundaryType::Outflow))
        .cfl(CFL)
        .finish()
        .expect("spherical NMHD sim construction failed")
}
fn make_cart() -> SimCart {
    SimCart::build(NewtonianMhd, IdealGas { gamma: GAMMA }, Cartesian)
        .cells([N, N, N])
        .origin([R_LO, T_LO, P_LO])
        .spacing([DR, DTH, DPH])
        .boundaries(Boundaries::uniform(BoundaryType::Outflow))
        .cfl(CFL)
        .finish()
        .expect("cartesian NMHD sim construction failed")
}

#[test]
fn full_substrate_spherical_nmhd_smoke() {
    let sub = NewtonianMhdSubstrateKernelSet3D::<HostMemory>::new(
        GAMMA,
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

    // geometry is active in the wave-speed map: the spherical CFL dt (r-weighted angular
    // widths) differs from the Cartesian dt on the same state -> nmhd_wave_speed_map_sph engaged.
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

    // the full curvilinear evolve exercises nmhd_godunov_stage_sph_3d (the Newtonian
    // geometric source) + the spherical CT.
    evolve(&mut sph, &sub, T_FINAL).expect("spherical NMHD evolution failed");

    let pre = sph.fields.prim.pre_field().expect("prim.pre");
    let mhd = sph.fields.mhd.as_ref().expect("mhd fields");
    for c in sph.geom.interior.iter() {
        let rho = *sph.fields.prim.rho.view().at(c);
        let p = *pre.view().at(c);
        assert!(rho.is_finite() && rho > 0.0, "bad density {rho} at {c:?}");
        assert!(p.is_finite() && p > 0.0, "bad pressure {p} at {c:?}");
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
        "SPHERICAL NMHD smoke: {} steps to t={:.4}, dt_sph {:.5} vs dt_cart {:.5}",
        sph.iteration, sph.time, dt_sph, dt_cart,
    );
}
