// =============================================================================
// nmhd_rotor_2p5d.rs
//
// the magnetized rotor (Toth 2000 test 1) as a 2.5D NMHD gate — a strong-field
// robustness + div(B) stress test distinct from Orszag-Tang: a dense disk spins in
// a uniform Bx, winding the field into torsional Alfven waves. asserts:
//   (a) the in-plane staggered div(B) = dBx/dx + dBy/dy stays at machine zero,
//   (b) the state stays physical (rho>0, p>0, finite) through the low-beta core,
//   (c) the field actually winds — By (zero in the IC) develops from the rotation,
//       proving the CT genuinely evolves the in-plane field under shear.
// =============================================================================

use symbi::regimes::substrate_kernels::Solver;
use symbi::regimes::substrate_newtonian_mhd::NewtonianMhdSubstrateKernelSet;
use symbi::sim::evolve::evolve_with_callback;
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_geometry::Cartesian;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::mhd_state::{MhdCons, MhdPrim};
use symbi_hydro::newtonian_mhd::{NewtonianMhd, nmhd_recover};
use symbi_hydro::state::{Cons, Prim};
use symbi_xpu::{CpuSpace, HostMemory};

type Sim = SimStateGeneric<NewtonianMhd, 2, 3, Cartesian, IdealGas<f64>, CpuSpace, HostMemory, f64>;

const N: usize = 128;
const GAMMA: f64 = 1.4;
const CFL: f64 = 0.4;
const XC: f64 = 0.5;
const R0: f64 = 0.1;
const R1: f64 = 0.115;
const V0: f64 = 2.0;
const T_FINAL: f64 = 0.05;

fn b0() -> f64 {
    5.0 / (4.0 * std::f64::consts::PI).sqrt()
}

fn rotor_state(x: f64, y: f64) -> (f64, f64, f64) {
    let (dx, dy) = (x - XC, y - XC);
    let r = (dx * dx + dy * dy).sqrt();
    if r < R0 {
        (10.0, -V0 * dy / R0, V0 * dx / R0)
    } else if r < R1 {
        let f = (R1 - r) / (R1 - R0);
        (1.0 + 9.0 * f, -f * V0 * dy / r, f * V0 * dx / r)
    } else {
        (1.0, 0.0, 0.0)
    }
}

fn make_sim() -> Sim {
    let dx = 1.0 / N as f64;
    let bx = b0();
    // uniform Bx threading the domain; By face stays zero.
    Sim::build(NewtonianMhd, IdealGas { gamma: GAMMA }, Cartesian)
        .cells([N, N])
        .spacing([dx, dx])
        .boundaries(Boundaries::uniform(BoundaryType::Outflow))
        .allocate()
        .expect("rotor sim")
        .set_initial(|[x, y]| {
            let (rho, vx, vy) = rotor_state(x, y);
            MhdPrim {
                hydro: Prim {
                    rho,
                    vel: Tensor::new([vx, vy, 0.0]),
                    pre: 1.0,
                },
                mag: Tensor::new([bx, 0.0, 0.0]),
            }
        })
        .seed_faces_uniform([bx, 0.0])
        .build()
}

fn rel_divb(s: &Sim) -> f64 {
    let mhd = s.fields.mhd.as_ref().unwrap();
    let idx = N as f64;
    let mut md = 0.0_f64;
    let mut mb = 1.0_f64;
    for c in s.geom.interior.iter() {
        let bx_lo = *mhd.bface[0].view().at(c);
        let bx_hi = *mhd.bface[0].view().at([c[0] + 1, c[1]]);
        let by_lo = *mhd.bface[1].view().at(c);
        let by_hi = *mhd.bface[1].view().at([c[0], c[1] + 1]);
        md = md.max(((bx_hi - bx_lo) * idx + (by_hi - by_lo) * idx).abs());
        mb = mb.max((bx_lo * bx_lo + by_lo * by_lo).sqrt());
    }
    md / mb
}

#[test]
fn nmhd_rotor_2p5d_preserves_divb_winds_field_stays_physical() {
    let mut sim = make_sim();
    let sub = NewtonianMhdSubstrateKernelSet::<HostMemory, f64, 2>::new(
        GAMMA,
        CFL,
        1.5,
        &sim.geom.allocated,
    )
    .with_solver(Solver::Hlld)
    .expect("valid solver/regime pair");

    let mut steps = 0u64;
    evolve_with_callback(&mut sim, &sub, T_FINAL, 1, |s| {
        let rel = rel_divb(s);
        assert!(
            rel < 1e-12,
            "rotor div(B) grew to rel={rel:e} at iter {}",
            s.iteration
        );
        steps = s.iteration;
    })
    .expect("rotor evolve failed");
    assert!(
        steps >= 10,
        "rotor produced only {steps} steps — gate barely exercised"
    );

    // physicality through the low-beta core + the field has wound (By developed from 0).
    let eos = IdealGas { gamma: GAMMA };
    let mhd = sim.fields.mhd.as_ref().unwrap();
    let cnrg = sim.fields.cons.nrg_field().unwrap();
    let mut max_by = 0.0_f64;
    for c in sim.geom.interior.iter() {
        let cons = MhdCons::<f64, 3> {
            hydro: Cons {
                chi: Default::default(),
                den: *sim.fields.cons.den.view().at(c),
                mom: Tensor::new([
                    *sim.fields.cons.mom[0].view().at(c),
                    *sim.fields.cons.mom[1].view().at(c),
                    *sim.fields.cons.mom[2].view().at(c),
                ]),
                nrg: *cnrg.view().at(c),
            },
            mag: Tensor::new([
                *mhd.bcell[0].view().at(c),
                *mhd.bcell[1].view().at(c),
                *mhd.bcell[2].view().at(c),
            ]),
        };
        let prim = nmhd_recover(&eos, &cons);
        assert!(
            prim.rho.is_finite() && prim.rho > 0.0,
            "cell {c:?}: rho={}",
            prim.rho
        );
        assert!(
            prim.pre.is_finite() && prim.pre > 0.0,
            "cell {c:?}: p={}",
            prim.pre
        );
        max_by = max_by.max(mhd.bcell[1].view().at(c).abs());
    }
    assert!(
        max_by > 0.05,
        "field did not wind: max|By|={max_by:e} (rotation should generate By)"
    );

    eprintln!(
        "[nmhd_rotor 2.5d] DONE iter={} t={:.4e} max|By|={:.4}",
        sim.iteration, sim.time, max_by
    );
}
