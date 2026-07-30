// =============================================================================
// mhd_cyl_rz_logr.rs
//
// end-to-end validation of the LOG-RADIAL 2.5D cylindrical (r-z) Newtonian-MHD kernels — the
// `_cyl_rz_logr` gas godunov, wave-speed map, out-of-plane bcell predictor, and the metric CT
// curl `rmhd_ct_curl_2d_{dir}_cyl_rz_logr`. the spacing slug is what selects the geometric-mean
// cell geometry; omitting it silently resolves the uniform-geometry kernel on a log grid, which
// reads the wrong face radii. this runs
// the whole CT stack over a radial axis spanning nearly a decade (R in [1, 8], log-spaced) and
// asserts:
//   (a) the discrete area-weighted div(B) of the staggered in-plane field
//       div(B) = 2 (r_hi B_r_hi - r_lo B_r_lo)/(r_hi^2 - r_lo^2) + (B_z_hi - B_z_lo)/dz
//       stays at machine zero (the corner E_phi CT holds the constraint on the geometric-mean grid),
//   (b) the gas stays PHYSICAL (rho>0, p>0, finite) as a radial pressure bump drives a flow through
//       the log-spaced cells (a wrong log geometry -> wrong CFL widths / fluxes -> NaN or negative p).
// div(B) preservation is structural (the single-valued corner EMF telescopes on any grid); the
// load-bearing check for the log GEOMETRY is the physicality under a real flow.
// =============================================================================

use symbi::regimes::substrate_newtonian_mhd::NewtonianMhdSubstrateKernelSet;
use symbi::sim::evolve::evolve_with_callback;
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_geometry::{AxisMap, Cylindrical};
use symbi_hydro::eos::IdealGas;
use symbi_hydro::mhd_state::{MhdCons, MhdPrim};
use symbi_hydro::newtonian_mhd::{NewtonianMhd, nmhd_recover};
use symbi_hydro::state::{Cons, Prim};
use symbi_xpu::{CpuSpace, HostMemory};

type Sim = SimStateGeneric<NewtonianMhd, 2, 3, Cylindrical, IdealGas<f64>, CpuSpace, HostMemory>;

const NR: usize = 48;
const NZ: usize = 6;
const R_LO: f64 = 1.0;
const R_HI: f64 = 8.0; // ~0.9 decades, log-spaced
const Z_LO: f64 = 0.0;
const Z_HI: f64 = 1.0;
const GAMMA: f64 = 5.0 / 3.0;
const CFL: f64 = 0.3;
const B0: f64 = 0.3; // uniform vertical B_z (div-free with B_r = 0)
const T_FINAL: f64 = 0.15;
const DIVB_TOL: f64 = 1e-11;

fn make_sim() -> Sim {
    let dz = (Z_HI - Z_LO) / NZ as f64;
    // face(i) = R_LO * 10^(i * log_slope); log_slope spans [R_LO, R_HI] over NR cells.
    let log_slope = (R_HI / R_LO).log10() / NR as f64;
    let maps = [
        AxisMap::Log {
            start: R_LO,
            log_slope,
        },
        AxisMap::Uniform {
            start: Z_LO,
            dx: dz,
        },
    ];
    // nominal linear dr for the builder; the log map overrides the radial axis geometry.
    let dr = (R_HI - R_LO) / NR as f64;
    Sim::build(NewtonianMhd, IdealGas { gamma: GAMMA }, Cylindrical)
        .cells([NR, NZ])
        .origin([R_LO, Z_LO])
        .spacing([dr, dz])
        .coord_maps(Some(maps))
        .cfl(CFL)
        .boundaries(Boundaries::uniform(BoundaryType::Outflow))
        .allocate()
        .expect("cyl r-z logr sim construction failed")
        .set_initial(|[r, _z]| {
            // v = 0; a smooth radial pressure bump drives a radial flow through the log cells.
            // B = (B_r, B_phi, B_z) = (0, 0, B0): uniform vertical field, div-free.
            let pre = 1.0 + 0.5 * (-((r - 3.0) / 0.8).powi(2)).exp();
            MhdPrim {
                hydro: Prim {
                    rho: 1.0,
                    vel: Tensor::new([0.0, 0.0, 0.0]),
                    pre,
                },
                mag: Tensor::new([0.0, 0.0, B0]),
            }
        })
        // staggered faces: B_r = 0 on the r-faces (bface[0]), B_z = B0 on the z-faces (bface[1]).
        .seed_faces_uniform([0.0, B0])
        .build()
}

// the area-weighted staggered div(B) for the cylindrical r-z cell (out-of-plane phi carries no
// face): r-face annulus areas ~ r, z-face areas ~ (r_hi^2 - r_lo^2)/2.
fn max_divb_and_b(sim: &Sim, dz: f64) -> (f64, f64) {
    let mhd = sim.fields.mhd.as_ref().expect("mhd");
    let mut max_div = 0.0_f64;
    let mut max_b = 0.0_f64;
    for c in sim.geom.interior.iter() {
        let r_lo = sim.geom.face_coord(c, 0)[0];
        let r_hi = sim.geom.face_coord([c[0] + 1, c[1]], 0)[0];
        let br_lo = *mhd.bface[0].view().at(c);
        let br_hi = *mhd.bface[0].view().at([c[0] + 1, c[1]]);
        let bz_lo = *mhd.bface[1].view().at(c);
        let bz_hi = *mhd.bface[1].view().at([c[0], c[1] + 1]);
        let div = 2.0 * (r_hi * br_hi - r_lo * br_lo) / (r_hi * r_hi - r_lo * r_lo)
            + (bz_hi - bz_lo) / dz;
        max_div = max_div.max(div.abs());
        let bz = *mhd.bcell[2].view().at(c);
        max_b = max_b.max((br_lo * br_lo + bz * bz).sqrt());
    }
    (max_div, max_b)
}

fn recover(sim: &Sim, c: [isize; 2]) -> (f64, f64) {
    let mhd = sim.fields.mhd.as_ref().unwrap();
    let cnrg = sim.fields.cons.nrg_field().unwrap();
    let cons = MhdCons::<f64, 3> {
        hydro: Cons {
            den: *sim.fields.cons.den.view().at(c),
            mom: Tensor::new([
                *sim.fields.cons.mom[0].view().at(c),
                *sim.fields.cons.mom[1].view().at(c),
                *sim.fields.cons.mom[2].view().at(c),
            ]),
            nrg: *cnrg.view().at(c),
            chi: Default::default(),
        },
        mag: Tensor::new([
            *mhd.bcell[0].view().at(c),
            *mhd.bcell[1].view().at(c),
            *mhd.bcell[2].view().at(c),
        ]),
    };
    let prim = nmhd_recover(&IdealGas { gamma: GAMMA }, &cons);
    (prim.rho, prim.pre)
}

#[test]
fn nmhd_cyl_rz_logr_preserves_divb_and_stays_physical() {
    let mut sim = make_sim();
    let dz = (Z_HI - Z_LO) / NZ as f64;

    let (div0, b0) = max_divb_and_b(&sim, dz);
    assert!(
        div0 / b0.max(1.0) < DIVB_TOL,
        "log-radial cyl r-z IC not divergence-free: max|divB|={div0:e} (rel {:e})",
        div0 / b0.max(1.0),
    );

    let sub = NewtonianMhdSubstrateKernelSet::<HostMemory, f64, 2>::new(
        GAMMA,
        CFL,
        /* theta */ 1.5,
        &sim.geom.allocated,
    );

    let mut steps = 0u64;
    let mut max_rel_div = 0.0_f64;
    evolve_with_callback(&mut sim, &sub, T_FINAL, 1, |s| {
        let (max_div, max_b) = max_divb_and_b(s, dz);
        let rel = max_div / max_b.max(1.0);
        assert!(
            rel < DIVB_TOL,
            "log-radial cyl r-z div(B) grew under evolve at iter {} t={:.4e}: max|divB|={:e} rel={:e} \
             (tol {:e}) — the _cyl_rz_logr CT curl is broken",
            s.iteration, s.time, max_div, rel, DIVB_TOL,
        );
        max_rel_div = max_rel_div.max(rel);
        steps = s.iteration;
    })
    .expect("cyl r-z logr evolve failed");

    assert!(steps >= 5, "only {steps} steps — gate barely exercised");

    // physicality across the log-spaced shell (a wrong log geometry NaNs or drives p < 0).
    for c in sim.geom.interior.iter() {
        let (rho, p) = recover(&sim, c);
        assert!(rho.is_finite() && rho > 0.0, "cell {c:?}: rho={rho}");
        assert!(p.is_finite() && p > 0.0, "cell {c:?}: p={p}");
    }

    eprintln!(
        "[cyl_rz_logr] DONE iter={} t={:.4e} max rel divB = {:e} (tol {:e})",
        sim.iteration, sim.time, max_rel_div, DIVB_TOL,
    );
}
