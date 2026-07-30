// =============================================================================
// substrate_rmhd_spherical.rs
//
// the curvilinear RMHD wiring smoke: the M-generic RmhdSubstrateKernelSet3D
// on a SPHERICAL shell through the real evolve() loop, exercising the _sph
// kernels the godunov/cfl methods dispatch — rmhd_godunov_euler_sph_3d (the
// area-weighted divergence + the RMHD geometric momentum source: pressure +
// inertial + magnetic tension), rmhd_wave_speed_map_sph_3d (per-cell physical
// CFL widths), and the spherical CT curl (rmhd_ct_curl_3d_*_sph). flux/c2p/
// edge_emf/bcell_from_bface are geometry-agnostic (Cartesian kernels on a shell).
//
// the IC is a div-free RADIAL field B_r = B0/r^2 (the split-monopole: A_r B_r =
// r^2 sin(th) dth dph * B0/r^2 = const in r, so the area-weighted div telescopes
// to 0) over a smooth radial pressure bump. the run must stay finite, positive,
// and subluminal — the wiring proof (a wrong buffer/scalar order NaNs). geometry
// is confirmed active by the spherical CFL dt differing from the Cartesian one
// (the wave-speed map's r-weighted angular widths).
// =============================================================================

use std::sync::atomic::Ordering;

use symbi::regimes::substrate_rmhd::RmhdSubstrateKernelSet3D;
use symbi::sim::evolve::{KernelSet, evolve};
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_geometry::{Cartesian, Spherical};
use symbi_hydro::eos::IdealGas;
use symbi_hydro::mhd_state::MhdPrim;
use symbi_hydro::regime::Regime;
use symbi_hydro::rmhd::Rmhd;
use symbi_hydro::state::{Cons, Prim};
use symbi_xpu::{CpuSpace, HostMemory};

type SimSph = SimState<Rmhd, 3, Spherical, IdealGas<f64>, CpuSpace, HostMemory>;
type SimCart = SimState<Rmhd, 3, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;

const N: usize = 8;
const R_LO: f64 = 1.0; // shell away from r=0
const DR: f64 = 0.1;
const T_LO: f64 = 0.6; // theta away from the poles 0, pi
const DTH: f64 = 0.06;
const P_LO: f64 = 0.2;
const DPH: f64 = 0.07;
const GAMMA: f64 = 5.0 / 3.0;
const CFL: f64 = 0.3;
const B0: f64 = 0.1;
const T_FINAL: f64 = 0.03;

// div-free radial B_r = B0/r^2 + a smooth radial pressure bump, v = 0. generic over M
// so the SAME field values seed both the spherical and the Cartesian sim.
fn set_ic<M>(sim: &mut SimState<Rmhd, 3, M, IdealGas<f64>, CpuSpace, HostMemory>)
where
    M: symbi_geometry::Metric<f64, 3> + Copy,
{
    let mhd = sim.fields.mhd.as_ref().expect("Rmhd requires mhd fields");
    // staggered B_r on the r-faces (B_theta = B_phi = 0); the area weighting makes this
    // exactly div-free. face_coord(c, 0)[0] is the r-face radius (R_LO + i*DR for uniform DR,
    // or the log-map face for non-uniform) — the one staggered-coordinate accessor.
    for c in &sim.geom.interior.extend(0, 0, 1) {
        let rf = sim.geom.face_coord(c, 0)[0];
        mhd.bface[0].view_mut().set(c, B0 / (rf * rf));
    }
    mhd.bface_initialized.store(true, Ordering::Relaxed);
    // cell-centered B_r = B0/r_c^2 on the full allocated domain (r_c the cell center).
    for c in sim.geom.allocated.iter() {
        let rc = R_LO + (c[0] as f64 + 0.5) * DR;
        mhd.bcell[0].view_mut().set(c, B0 / (rc * rc));
        mhd.bcell[1].view_mut().set(c, 0.0);
        mhd.bcell[2].view_mut().set(c, 0.0);
    }
    // hydro: a smooth radial pressure bump drives a mild flow; conserved via the
    // production forward map (W=1 since v=0).
    for c in sim.geom.interior.iter() {
        let r = R_LO + (c[0] as f64 + 0.5) * DR;
        let bc = B0 / (r * r);
        let pre = 1.0 + 0.3 * (-((r - 1.4) / 0.2).powi(2)).exp();
        let prim = MhdPrim {
            hydro: Prim {
                rho: 1.0,
                vel: Tensor::new([0.0, 0.0, 0.0]),
                pre,
            },
            mag: Tensor::new([bc, 0.0, 0.0]),
        };
        let cons = sim.physics.regime.to_conserved(&sim.physics.eos, &prim);
        sim.fields.cons.scatter(
            c,
            Cons {
                chi: Default::default(),
                den: cons.den,
                mom: cons.mom,
                nrg: cons.nrg,
            },
        );
    }
}

fn make_sph() -> SimSph {
    // unseeded; set_ic seeds the staggered B + bcell (full allocated domain) post-construction.
    SimSph::build(Rmhd, IdealGas { gamma: GAMMA }, Spherical)
        .cells([N, N, N])
        .origin([R_LO, T_LO, P_LO])
        .spacing([DR, DTH, DPH])
        .boundaries(Boundaries::uniform(BoundaryType::Outflow))
        .cfl(CFL)
        .finish()
        .expect("spherical RMHD sim construction failed")
}
fn make_cart() -> SimCart {
    SimCart::build(Rmhd, IdealGas { gamma: GAMMA }, Cartesian)
        .cells([N, N, N])
        .origin([R_LO, T_LO, P_LO])
        .spacing([DR, DTH, DPH])
        .boundaries(Boundaries::uniform(BoundaryType::Outflow))
        .cfl(CFL)
        .finish()
        .expect("cartesian RMHD sim construction failed")
}

#[test]
fn full_substrate_spherical_rmhd_smoke() {
    let sub =
        RmhdSubstrateKernelSet3D::<HostMemory>::new(GAMMA, CFL, 1.0, &make_sph().geom.allocated);

    let mut sph = make_sph();
    set_ic(&mut sph);
    assert_eq!(
        sph.geom.coords,
        symbi_geometry::Geometry::Spherical,
        "coords must be Spherical"
    );

    // geometry is ACTIVE in the wave-speed map: the spherical CFL dt (r-weighted angular
    // widths h_theta=r, h_phi=r sin(theta)) differs from the Cartesian dt on the same state.
    // c2p first (cfl reads prim; the evolve loop also leads with c2p).
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

    evolve(&mut sph, &sub, T_FINAL).expect("spherical RMHD evolution failed");

    // finite + positive + subluminal everywhere — the _sph godunov/cfl/CT ran correctly.
    let pre = sph.fields.prim.pre_field().expect("prim.pre");
    let mhd = sph.fields.mhd.as_ref().expect("mhd fields");
    let mut max_v = 0.0_f64;
    for c in sph.geom.interior.iter() {
        let rho = *sph.fields.prim.rho.view().at(c);
        let p = *pre.view().at(c);
        let v2: f64 = (0..3)
            .map(|d| sph.fields.prim.vel[d].view().at(c).powi(2))
            .sum();
        let v = v2.sqrt();
        assert!(rho.is_finite() && rho > 0.0, "bad density {rho} at {c:?}");
        assert!(p.is_finite() && p > 0.0, "bad pressure {p} at {c:?}");
        assert!(v.is_finite() && v < 1.0, "superluminal |v| = {v} at {c:?}");
        for d in 0..3 {
            assert!(
                mhd.bcell[d].view().at(c).is_finite(),
                "non-finite bcell[{d}] at {c:?}"
            );
        }
        max_v = max_v.max(v);
    }
    assert!(sph.iteration > 0, "no steps taken");
    println!(
        "SPHERICAL RMHD smoke: {} steps to t={:.4}, max |v| {:.5}, dt_sph {:.5} vs dt_cart {:.5}",
        sph.iteration, sph.time, max_v, dt_sph, dt_cart,
    );
}
