// =============================================================================
// resistive_cyl_decay_assessment.rs
//
// the decay record behind the cylindrical r-z resistivity threshold: the
// poloidal seed B_z = B0 sin(k (r - r_min)) on [1, 3] x [0, 1] evolved ideal and
// resistive through the production evolve path, reported per step as
// - max |B_z| over the interior (the quantity the threshold reads);
// - the r-weighted projection of B_z onto the seed profile;
// against a fine 1D radial solve of dB_z/dt = eta (1/r) d_r (r d_r B_z) with
// Neumann (even) and Dirichlet (odd) walls, which bound the continuum decay.
// the resolution and the CFL number come from the environment so one binary
// sweeps both.
//
// run:
//   CYL_N=24 CYL_CFL=0.3 cargo test -p symbi --test resistive_cyl_decay_assessment \
//     -- --ignored --nocapture
// =============================================================================

use std::f64::consts::PI;
use symbi::regimes::substrate_newtonian_mhd::NewtonianMhdSubstrateKernelSet;
use symbi::sim::evolve::evolve_with_callback;
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_geometry::Cylindrical;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::mhd_state::MhdPrim;
use symbi_hydro::newtonian_mhd::NewtonianMhd;
use symbi_hydro::quantity::{Density, Pressure};
use symbi_hydro::state::Prim;
use symbi_xpu::{CpuSpace, HostMemory};

type Sim =
    SimStateGeneric<NewtonianMhd, 2, 3, Cylindrical, IdealGas<f64>, CpuSpace, HostMemory, f64>;

const GAMMA: f64 = 5.0 / 3.0;
const R_MIN: f64 = 1.0;
const R_MAX: f64 = 3.0;
const Z_MAX: f64 = 1.0;
const B0: f64 = 1e-2;
const T_FINAL: f64 = 0.3;
const ETA: f64 = 0.05;
const K: f64 = 4.0 * PI / (R_MAX - R_MIN);

fn env_or<T: std::str::FromStr>(key: &str, default: T) -> T {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn make_sim(n: usize, cfl: f64) -> Sim {
    Sim::build(NewtonianMhd, IdealGas { gamma: GAMMA }, Cylindrical)
        .cells([n, n])
        .bounds([R_MIN, 0.0], [R_MAX, Z_MAX])
        .boundaries(Boundaries::per_axis([
            [BoundaryType::Reflect, BoundaryType::Reflect],
            [BoundaryType::Periodic, BoundaryType::Periodic],
        ]))
        .cfl(cfl)
        .allocate()
        .expect("cyl r-z decay sim")
        .set_initial(|[r, _z]| {
            MhdPrim::new(
                Prim::adiabatic(Density(1.0), Tensor::new([0.0, 0.0, 0.0]), Pressure(1.0)),
                Tensor::new([0.0, 0.0, B0 * (K * (r - R_MIN)).sin()]),
            )
        })
        .seed_faces(|axis, [r, _z]| {
            if axis == 1 {
                B0 * (K * (r - R_MIN)).sin()
            } else {
                0.0
            }
        })
        .build()
}

/// max |B_z| and the r-weighted projection of B_z onto the seed profile, both relative to B0
fn measures(s: &Sim) -> (f64, f64) {
    let m = s.fields.mhd.as_ref().unwrap();
    let dr = s.geom.dx[0];
    let lo = s.geom.interior.spaces[0].lo;
    let (mut peak, mut num, mut den) = (0.0f64, 0.0f64, 0.0f64);
    for c in s.geom.interior.iter() {
        let bz = *m.bcell[2].view().at(c);
        let r = R_MIN + ((c[0] - lo) as f64 + 0.5) * dr;
        let shape = (K * (r - R_MIN)).sin();
        peak = peak.max(bz.abs());
        num += bz * shape * r;
        den += shape * shape * r;
    }
    (peak / B0, num / den / B0)
}

/// max |B| and the seed-profile projection of a fine 1D radial diffusion solve at T_FINAL
fn radial_reference(eta: f64, neumann: bool) -> (f64, f64) {
    let n = 4000usize;
    let dr = (R_MAX - R_MIN) / n as f64;
    let r: Vec<f64> = (0..n).map(|ii| R_MIN + (ii as f64 + 0.5) * dr).collect();
    let mut b: Vec<f64> = r.iter().map(|&x| (K * (x - R_MIN)).sin()).collect();
    let dt = 0.2 * dr * dr / eta.max(1e-300);
    let steps = if eta > 0.0 {
        (T_FINAL / dt).ceil() as usize
    } else {
        0
    };
    let dt = T_FINAL / steps.max(1) as f64;
    let sign = if neumann { 1.0 } else { -1.0 };
    for _ in 0..steps {
        let ghost_lo = sign * b[0];
        let ghost_hi = sign * b[n - 1];
        let mut next = b.clone();
        for ii in 0..n {
            let bl = if ii == 0 { ghost_lo } else { b[ii - 1] };
            let br = if ii == n - 1 { ghost_hi } else { b[ii + 1] };
            let (rl, rr) = (r[ii] - 0.5 * dr, r[ii] + 0.5 * dr);
            next[ii] =
                b[ii] + dt * eta * (rr * (br - b[ii]) - rl * (b[ii] - bl)) / (r[ii] * dr * dr);
        }
        b = next;
    }
    let peak = b.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    let (mut num, mut den) = (0.0, 0.0);
    for (x, v) in r.iter().zip(&b) {
        let shape = (K * (x - R_MIN)).sin();
        num += v * shape * x;
        den += shape * shape * x;
    }
    (peak, num / den)
}

/// the interior cell holding max |B_z|, as (r index, z index, B_z / B0, |v| there)
fn peak_cell(s: &Sim) -> (isize, isize, f64, f64) {
    let m = s.fields.mhd.as_ref().unwrap();
    let lo = [s.geom.interior.spaces[0].lo, s.geom.interior.spaces[1].lo];
    let mut best = (0, 0, 0.0f64, 0.0f64);
    for c in s.geom.interior.iter() {
        let bz = *m.bcell[2].view().at(c);
        if bz.abs() > best.2.abs() {
            let vr = *s.fields.prim.vel[0].view().at(c);
            let vz = *s.fields.prim.vel[2].view().at(c);
            best = (c[0] - lo[0], c[1] - lo[1], bz, (vr * vr + vz * vz).sqrt());
        }
    }
    (best.0, best.1, best.2 / B0, best.3)
}

fn run(n: usize, cfl: f64, eta: f64) -> Vec<(f64, f64, f64)> {
    let mut sim = make_sim(n, cfl);
    let (p0, q0) = measures(&sim);
    let sub = NewtonianMhdSubstrateKernelSet::<HostMemory, f64, 2>::new(
        GAMMA,
        cfl,
        1.0,
        &sim.geom.allocated,
    )
    .with_resistivity(eta);
    let series = std::cell::RefCell::new(vec![(0.0, p0, q0)]);
    let every: usize = env_or("CYL_WHERE", 0);
    let step = std::cell::Cell::new(0usize);
    evolve_with_callback(&mut sim, &sub, T_FINAL, 1, |s| {
        let (p, q) = measures(s);
        series.borrow_mut().push((s.time, p, q));
        step.set(step.get() + 1);
        if every > 0 && step.get() % every == 0 {
            let (ir, iz, bz, v) = peak_cell(s);
            println!("W eta={eta} step {} t {:.5} peak cell (ir {ir}, iz {iz}) of {n} B_z/B0 {bz:.4} |v_pol| {v:.3e}", step.get(), s.time);
        }
    })
    .expect("cyl decay evolve");
    if let Ok(ir) = std::env::var("CYL_COLUMN").map(|v| v.parse::<isize>().unwrap()) {
        let m = sim.fields.mhd.as_ref().unwrap();
        let lo = [
            sim.geom.interior.spaces[0].lo,
            sim.geom.interior.spaces[1].lo,
        ];
        for iz in 0..n as isize {
            let c = [lo[0] + ir, lo[1] + iz];
            println!(
                "Z eta={eta} ir {ir} iz {iz:>3} B_r/B0 {:+.4e} B_z/B0 {:+.4e} v_r {:+.3e} v_z {:+.3e} p-1 {:+.3e}",
                *m.bcell[0].view().at(c) / B0,
                *m.bcell[2].view().at(c) / B0,
                *sim.fields.prim.vel[0].view().at(c),
                *sim.fields.prim.vel[2].view().at(c),
                *sim.fields.prim.pre_field().unwrap().view().at(c) - 1.0
            );
        }
    }
    series.into_inner()
}

#[test]
#[ignore = "diagnostic record; prints the cylindrical decay curves"]
fn cylindrical_decay_record() {
    let n: usize = env_or("CYL_N", 24);
    let cfl: f64 = env_or("CYL_CFL", 0.3);
    let stride: usize = env_or("CYL_STRIDE", 1);
    let arms: String = env_or("CYL_ARMS", "both".to_string());
    let ideal = if arms == "resistive" {
        vec![(0.0, 1.0, 1.0)]
    } else {
        run(n, cfl, 0.0)
    };
    let resistive = if arms == "ideal" {
        vec![(0.0, 1.0, 1.0)]
    } else {
        run(n, cfl, ETA)
    };
    println!(
        "# N={n} cfl={cfl} eta={ETA} steps ideal={} resistive={}",
        ideal.len() - 1,
        resistive.len() - 1
    );
    println!("# series: t peak_ideal proj_ideal | t peak_res proj_res");
    let rows = ideal.len().max(resistive.len());
    for ii in (0..rows).step_by(stride.max(1)) {
        let a = ideal
            .get(ii)
            .map(|x| format!("{:.5} {:.6} {:.6}", x.0, x.1, x.2))
            .unwrap_or_default();
        let b = resistive
            .get(ii)
            .map(|x| format!("{:.5} {:.6} {:.6}", x.0, x.1, x.2))
            .unwrap_or_default();
        println!("S {a:<32} | {b}");
    }
    let (fi, fr) = (ideal.last().unwrap(), resistive.last().unwrap());
    let (ref_nm, ref_dr) = (radial_reference(ETA, true), radial_reference(ETA, false));
    println!(
        "END N={n} cfl={cfl} t={:.4}/{:.4} ideal peak {:.6} proj {:.6} | resistive peak {:.6} proj {:.6} | \
         loss ratio (peak) {:.3} | ref neumann peak {:.6} proj {:.6} | ref dirichlet peak {:.6} proj {:.6}",
        fi.0,
        fr.0,
        fi.1,
        fi.2,
        fr.1,
        fr.2,
        (1.0 - fr.1) / (1.0 - fi.1),
        ref_nm.0,
        ref_nm.1,
        ref_dr.0,
        ref_dr.1
    );
}
