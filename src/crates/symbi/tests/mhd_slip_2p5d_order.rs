// =============================================================================
// mhd_slip_2p5d_order.rs
//
// temporal order of the 2.5D ideal-MHD step H alone, body-free, swept over resolution, per storage
// complex: the in-plane faces, the cell B_z, the total energy, the pressure. the staggered field's
// ratio on a coarse grid reads the grid's own spatial structure, so the order is taken from the
// finest grid; the cell B_z, evolved in flux form, has no such structure and reads clean at every
// resolution.
// =============================================================================

use symbi::regimes::substrate_newtonian_mhd::NewtonianMhdSubstrateKernelSet;
use symbi::sim::refinement::Hierarchy;
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_geometry::Cartesian;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::mhd_state::MhdPrim;
use symbi_hydro::newtonian_mhd::NewtonianMhd;
use symbi_hydro::quantity::{Density, Pressure};
use symbi_hydro::state::Prim;
use symbi::regimes::substrate_kernels::Solver;
use symbi_sim::state::CtMethod;
use symbi_xpu::{CpuSpace, HostMemory};

type Sim = SimStateGeneric<NewtonianMhd, 2, 3, Cartesian, IdealGas<f64>, CpuSpace, HostMemory, f64>;
type Kernels = NewtonianMhdSubstrateKernelSet<HostMemory, f64, 2>;
const GAMMA: f64 = 5.0 / 3.0;

// under the UCT edge EMF with the two-wave HLLE solver, on the regime the 2.5D vertical-field
// model runs in: a strong in-plane field, a shear flow, and an out-of-plane component. the default
// Contact EMF upwinds its corner derivative terms by a soft sign of the mass flux, which switches
// branches wherever that flux crosses zero; on a flow with such crossings the in-plane field reads
// first order at every resolution while the flux-evolved B_z reads second order. under UCT the
// HLLD fan's degenerate branches switch where the normal field crosses zero under a rotated
// tangential field and the in-plane reading turns irregular once B_z is present; HLLE carries
// neither switch. the ladder below records every other reading.
#[test]
fn the_2p5d_ideal_mhd_step_is_second_order_per_field_under_uct_hlle() {
    let k = 2.0 * std::f64::consts::PI;
    let flow_prim = move |[x, _]: [f64; 2]| (1.0, [0.0, 0.1 * (k * x).sin(), 0.0], 1.0);
    let flowz_cell = move |[x, y]: [f64; 2]| [1.0 + 0.1 * (k * y).sin(), 0.1 * (k * x).sin(), 0.2 * (k * x).cos() * (k * y).cos()];
    let face = move |a: usize, [x, y]: [f64; 2]| if a == 0 { 1.0 + 0.1 * (k * y).sin() } else { 0.1 * (k * x).sin() };
    let dt = 2.5e-4;
    for n in [32usize, 64] {
        let r = ratios_2p5d_with(&move || build_2p5d(n, flow_prim, flowz_cell, face), dt, Solver::Hlle, CtMethod::Uct);
        for (i, nm) in ["bface", "bz", "energy", "pressure"].iter().enumerate() {
            println!("2.5D H (UCT+HLLE) N={n:<3} {nm:>8}: ratios {:.2} {:.2}", r[i].0, r[i].1);
            if n == 64 {
                assert!(
                    r[i].0 > 3.5 && r[i].1 > 3.5,
                    "the 2.5D ideal-MHD step under UCT+HLLE is not second order in {nm} at N={n}: ratios {:.2} {:.2}",
                    r[i].0,
                    r[i].1
                );
            }
        }
    }
}

// ---- ownership ladder: which flow regime carries the first-order in-plane reading ------------------

type Sim3 = SimStateGeneric<NewtonianMhd, 3, 3, Cartesian, IdealGas<f64>, CpuSpace, HostMemory, f64>;
type Kernels3 = NewtonianMhdSubstrateKernelSet<HostMemory, f64, 3>;

// a 2.5D box from primitive and face closures of (x, y); the cell field is the closure's value
// (each face component varies only across its own normal, so that is the exact face average).
fn build_2p5d(
    n: usize,
    prim: impl Fn([f64; 2]) -> (f64, [f64; 3], f64) + 'static,
    cell_b: impl Fn([f64; 2]) -> [f64; 3] + 'static,
    face: impl Fn(usize, [f64; 2]) -> f64 + 'static,
) -> Sim {
    let dx = 1.0 / n as f64;
    SimStateGeneric::<NewtonianMhd, 2, 3, Cartesian, IdealGas<f64>, CpuSpace, HostMemory, f64>::build(
        NewtonianMhd,
        IdealGas { gamma: GAMMA },
        Cartesian,
    )
    .cells([n, n])
    .origin([0.0, 0.0])
    .spacing([dx, dx])
    .boundaries(Boundaries::uniform(BoundaryType::Periodic))
    .cfl(0.3)
    .allocate()
    .expect("2.5D sim construction")
    .set_initial(move |x| {
        let (rho, v, p) = prim(x);
        MhdPrim::new(Prim::adiabatic(Density(rho), Tensor::new(v), Pressure(p)), Tensor::new(cell_b(x)))
    })
    .seed_faces(move |axis, x| face(axis, x))
    .build()
}

// the same fixture extruded along z on a 3D box (nz cells), z-invariant.
fn build_3d(
    n: usize,
    nz: usize,
    prim: impl Fn([f64; 2]) -> (f64, [f64; 3], f64) + 'static,
    cell_b: impl Fn([f64; 2]) -> [f64; 3] + Copy + 'static,
    face: impl Fn(usize, [f64; 2]) -> f64 + 'static,
) -> Sim3 {
    let dx = 1.0 / n as f64;
    SimStateGeneric::<NewtonianMhd, 3, 3, Cartesian, IdealGas<f64>, CpuSpace, HostMemory, f64>::build(
        NewtonianMhd,
        IdealGas { gamma: GAMMA },
        Cartesian,
    )
    .cells([n, n, nz])
    .origin([0.0, 0.0, 0.0])
    .spacing([dx, dx, dx])
    .boundaries(Boundaries::uniform(BoundaryType::Periodic))
    .cfl(0.3)
    .allocate()
    .expect("3D sim construction")
    .set_initial(move |[x, y, _z]| {
        let (rho, v, p) = prim([x, y]);
        MhdPrim::new(Prim::adiabatic(Density(rho), Tensor::new(v), Pressure(p)), Tensor::new(cell_b([x, y])))
    })
    // the z-faces of a z-invariant state carry the cell B_z, so every stored representation agrees.
    .seed_faces(move |axis, [x, y, _z]| if axis == 2 { cell_b([x, y])[2] } else { face(axis, [x, y]) })
    .build()
}

fn ratios_2p5d(sim_of: &dyn Fn() -> Sim, dt: f64) -> [(f64, f64); 4] {
    let l2 = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum::<f64>().sqrt();
    let run = |dt: f64, nsteps: usize| -> [Vec<f64>; 4] {
        let sim = sim_of();
        let sub = Kernels::new(GAMMA, 0.3, 1.0, &sim.geom.allocated);
        let mut hier = Hierarchy::single(sim, sub);
        hier.prime();
        for _ in 0..nsteps {
            assert!(!hier.hydro_map(0, dt), "H stage retry");
        }
        let (bf, bc, nrg, pre) = hier.slip_state_snapshots(0);
        let ncells = nrg.len();
        [bf, bc[2 * ncells..3 * ncells].to_vec(), nrg, pre]
    };
    let (u1, u2, u3, u4) = (run(dt, 8), run(dt / 2.0, 16), run(dt / 4.0, 32), run(dt / 8.0, 64));
    std::array::from_fn(|i| {
        let (e1, e2, e3) = (l2(&u1[i], &u2[i]), l2(&u2[i], &u3[i]), l2(&u3[i], &u4[i]));
        (e1 / e2.max(1e-300), e2 / e3.max(1e-300))
    })
}

fn ratios_3d(sim_of: &dyn Fn() -> Sim3, dt: f64) -> [(f64, f64); 4] {
    let l2 = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum::<f64>().sqrt();
    let run = |dt: f64, nsteps: usize| -> [Vec<f64>; 4] {
        let sim = sim_of();
        let sub = Kernels3::new(GAMMA, 0.3, 1.0, &sim.geom.allocated);
        let mut hier = Hierarchy::single(sim, sub);
        hier.prime();
        for _ in 0..nsteps {
            assert!(!hier.hydro_map(0, dt), "H stage retry");
        }
        let (bf, bc, nrg, pre) = hier.slip_state_snapshots(0);
        let ncells = nrg.len();
        [bf, bc[2 * ncells..3 * ncells].to_vec(), nrg, pre]
    };
    let (u1, u2, u3, u4) = (run(dt, 8), run(dt / 2.0, 16), run(dt / 4.0, 32), run(dt / 8.0, 64));
    std::array::from_fn(|i| {
        let (e1, e2, e3) = (l2(&u1[i], &u2[i]), l2(&u2[i], &u3[i]), l2(&u3[i], &u4[i]));
        (e1 / e2.max(1e-300), e2 / e3.max(1e-300))
    })
}

#[test]
fn diag_h_ownership_ladder_2p5d_and_3d() {
    let k = 2.0 * std::f64::consts::PI;
    let names = ["bface", "bz", "energy", "pressure"];
    let show = |label: &str, r: [(f64, f64); 4]| {
        for (i, nm) in names.iter().enumerate() {
            println!("LADDER {label:>14} {nm:>8}: ratios {:.2} {:.2}", r[i].0, r[i].1);
        }
    };
    // static strong field: zero velocity, the EMF vanishes at t = 0.
    let static_prim = move |_: [f64; 2]| (1.0, [0.0, 0.0, 0.0], 1.0);
    let static_cell = move |[x, y]: [f64; 2]| [1.0 + 0.1 * (k * y).sin(), 0.1 * (k * x).sin(), 0.0];
    let static_face = move |a: usize, [x, y]: [f64; 2]| if a == 0 { 1.0 + 0.1 * (k * y).sin() } else { 0.1 * (k * x).sin() };
    // passive advection: uniform flow, weak transverse field.
    let pass_prim = move |_: [f64; 2]| (1.0, [0.5, 0.0, 0.0], 1.0);
    let pass_cell = move |[x, _]: [f64; 2]| [0.0, 0.01 * (k * x).sin(), 0.0];
    let pass_face = move |a: usize, [x, _]: [f64; 2]| if a == 1 { 0.01 * (k * x).sin() } else { 0.0 };
    // strong field with a velocity perturbation, no out-of-plane component.
    let flow_prim = move |[x, _]: [f64; 2]| (1.0, [0.0, 0.1 * (k * x).sin(), 0.0], 1.0);
    // the same with an out-of-plane component.
    let flowz_cell = move |[x, y]: [f64; 2]| [1.0 + 0.1 * (k * y).sin(), 0.1 * (k * x).sin(), 0.2 * (k * x).cos() * (k * y).cos()];
    // the same flow with a weak velocity perturbation, and with a weak field.
    let weakv_prim = move |[x, _]: [f64; 2]| (1.0, [0.0, 0.01 * (k * x).sin(), 0.0], 1.0);
    let weakb_cell = move |[x, y]: [f64; 2]| [0.1 + 0.01 * (k * y).sin(), 0.01 * (k * x).sin(), 0.0];
    let weakb_face = move |a: usize, [x, y]: [f64; 2]| if a == 0 { 0.1 + 0.01 * (k * y).sin() } else { 0.01 * (k * x).sin() };
    let dt = 2.5e-4;
    let n = 64;
    show("2.5D static", ratios_2p5d(&move || build_2p5d(n, static_prim, static_cell, static_face), dt));
    show("2.5D passive", ratios_2p5d(&move || build_2p5d(n, pass_prim, pass_cell, pass_face), dt));
    show("2.5D flow", ratios_2p5d(&move || build_2p5d(n, flow_prim, static_cell, static_face), dt));
    show("2.5D flow+bz", ratios_2p5d(&move || build_2p5d(n, flow_prim, flowz_cell, static_face), dt));
    show("2.5D weak-v", ratios_2p5d(&move || build_2p5d(n, weakv_prim, static_cell, static_face), dt));
    show("2.5D weak-B", ratios_2p5d(&move || build_2p5d(n, flow_prim, weakb_cell, weakb_face), dt));
    // guard census on the strong flow: a fallback or floor firing would make the map irregular in dt.
    {
        let sim = build_2p5d(n, flow_prim, static_cell, static_face);
        let sub = Kernels::new(GAMMA, 0.3, 1.0, &sim.geom.allocated);
        let mut hier = Hierarchy::single(sim, sub);
        hier.prime();
        symbi_sim::guard_ledger::reset();
        let scope = symbi_sim::guard_ledger::open_scope();
        for _ in 0..32 {
            assert!(!hier.hydro_map(0, dt / 4.0), "H stage retry");
        }
        let (attempted, _) = symbi_sim::guard_ledger::report();
        drop(scope);
        symbi_sim::guard_ledger::reset();
        println!("LADDER 2.5D flow guards: troubled={} frozen={}", attempted.troubled_cells.total, attempted.frozen_cells.total);
    }
    let n3 = 32;
    show("3D static", ratios_3d(&move || build_3d(n3, 4, static_prim, static_cell, static_face), dt));
    show("3D flow", ratios_3d(&move || build_3d(n3, 4, flow_prim, static_cell, static_face), dt));
    show("3D flow+bz", ratios_3d(&move || build_3d(n3, 4, flow_prim, flowz_cell, static_face), dt));
    show("3D flow UCT", ratios_3d_uct(&move || build_3d(n3, 4, flow_prim, static_cell, static_face), dt));
    show("2.5D flow+bz UCT", ratios_2p5d_uct(&move || build_2p5d(n, flow_prim, flowz_cell, static_face), dt));
}

fn ratios_2p5d_uct(sim_of: &dyn Fn() -> Sim, dt: f64) -> [(f64, f64); 4] {
    let l2 = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum::<f64>().sqrt();
    let run = |dt: f64, nsteps: usize| -> [Vec<f64>; 4] {
        let sim = sim_of();
        let sub = Kernels::new(GAMMA, 0.3, 1.0, &sim.geom.allocated)
            .with_solver(Solver::Hlld)
            .expect("hlld")
            .ct_method(CtMethod::Uct);
        let mut hier = Hierarchy::single(sim, sub);
        hier.prime();
        for _ in 0..nsteps {
            assert!(!hier.hydro_map(0, dt), "H stage retry");
        }
        let (bf, bc, nrg, pre) = hier.slip_state_snapshots(0);
        let ncells = nrg.len();
        [bf, bc[2 * ncells..3 * ncells].to_vec(), nrg, pre]
    };
    let (u1, u2, u3, u4) = (run(dt, 8), run(dt / 2.0, 16), run(dt / 4.0, 32), run(dt / 8.0, 64));
    std::array::from_fn(|i| {
        let (e1, e2, e3) = (l2(&u1[i], &u2[i]), l2(&u2[i], &u3[i]), l2(&u3[i], &u4[i]));
        (e1 / e2.max(1e-300), e2 / e3.max(1e-300))
    })
}

fn ratios_3d_uct(sim_of: &dyn Fn() -> Sim3, dt: f64) -> [(f64, f64); 4] {
    let l2 = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum::<f64>().sqrt();
    let run = |dt: f64, nsteps: usize| -> [Vec<f64>; 4] {
        let sim = sim_of();
        let sub = Kernels3::new(GAMMA, 0.3, 1.0, &sim.geom.allocated)
            .with_solver(Solver::Hlld)
            .expect("hlld")
            .ct_method(CtMethod::Uct);
        let mut hier = Hierarchy::single(sim, sub);
        hier.prime();
        for _ in 0..nsteps {
            assert!(!hier.hydro_map(0, dt), "H stage retry");
        }
        let (bf, bc, nrg, pre) = hier.slip_state_snapshots(0);
        let ncells = nrg.len();
        [bf, bc[2 * ncells..3 * ncells].to_vec(), nrg, pre]
    };
    let (u1, u2, u3, u4) = (run(dt, 8), run(dt / 2.0, 16), run(dt / 4.0, 32), run(dt / 8.0, 64));
    std::array::from_fn(|i| {
        let (e1, e2, e3) = (l2(&u1[i], &u2[i]), l2(&u2[i], &u3[i]), l2(&u3[i], &u4[i]));
        (e1 / e2.max(1e-300), e2 / e3.max(1e-300))
    })
}

// ---- the corrector's share: the Alfvenic fixture under RK1, RK2, and RK3 -----------------------------

fn build_2p5d_ts(n: usize, ts: Timestepping) -> Sim {
    let k = 2.0 * std::f64::consts::PI;
    let dx = 1.0 / n as f64;
    SimStateGeneric::<NewtonianMhd, 2, 3, Cartesian, IdealGas<f64>, CpuSpace, HostMemory, f64>::build(
        NewtonianMhd,
        IdealGas { gamma: GAMMA },
        Cartesian,
    )
    .cells([n, n])
    .origin([0.0, 0.0])
    .spacing([dx, dx])
    .boundaries(Boundaries::uniform(BoundaryType::Periodic))
    .cfl(0.3)
    .timestepping(ts)
    .allocate()
    .expect("2.5D sim construction")
    .set_initial(move |[x, y]| {
        MhdPrim::new(
            Prim::adiabatic(Density(1.0), Tensor::new([0.0, 0.1 * (k * x).sin(), 0.0]), Pressure(1.0)),
            Tensor::new([1.0 + 0.1 * (k * y).sin(), 0.1 * (k * x).sin(), 0.0]),
        )
    })
    .seed_faces(move |axis, [x, y]| if axis == 0 { 1.0 + 0.1 * (k * y).sin() } else { 0.1 * (k * x).sin() })
    .build()
}

#[test]
fn diag_alfvenic_h_step_under_rk1_rk2_rk3() {
    let l2 = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum::<f64>().sqrt();
    let n = 64usize;
    let dt = 2.5e-4;
    let run = |ts: Timestepping, dt: f64, nsteps: usize| -> [Vec<f64>; 4] {
        let sim = build_2p5d_ts(n, ts);
        let sub = Kernels::new(GAMMA, 0.3, 1.0, &sim.geom.allocated);
        let mut hier = Hierarchy::single(sim, sub);
        hier.prime();
        for _ in 0..nsteps {
            assert!(!hier.hydro_map(0, dt), "H stage retry");
        }
        let (bf, _bc, nrg, pre) = hier.slip_state_snapshots(0);
        [bf, hier.density_snapshot(0), nrg, pre]
    };
    let names = ["bface", "density", "energy", "pressure"];
    let mut at_dt: Vec<[Vec<f64>; 4]> = Vec::new();
    for (label, ts) in [("RK1", Timestepping::Euler), ("RK2", Timestepping::Rk2)] {
        let (u1, u2, u3, u4) = (run(ts, dt, 8), run(ts, dt / 2.0, 16), run(ts, dt / 4.0, 32), run(ts, dt / 8.0, 64));
        for (i, nm) in names.iter().enumerate() {
            let (e1, e2, e3) = (l2(&u1[i], &u2[i]), l2(&u2[i], &u3[i]), l2(&u3[i], &u4[i]));
            println!("TS {label} {nm:>8}: diffs {e1:.3e} {e2:.3e} {e3:.3e}  ratios {:.2} {:.2}", e1 / e2.max(1e-300), e2 / e3.max(1e-300));
        }
        at_dt.push(u1);
    }
    for (i, nm) in names.iter().enumerate() {
        println!("TS |RK2 - RK1|(dt) {nm:>8}: {:.3e}", l2(&at_dt[1][i], &at_dt[0][i]));
    }
}

// the induction alone under a non-uniform flow: a gas heavy enough that the Lorentz force and the
// (absent) pressure gradient leave v_y(x) effectively prescribed, so the in-plane field evolves by
// E_z = -v_y B_x with a fixed velocity structure.
#[test]
fn diag_near_kinematic_induction_with_shear() {
    let k = 2.0 * std::f64::consts::PI;
    let n = 64usize;
    let dt = 2.5e-4;
    for (label, rho) in [("rho=1", 1.0), ("rho=1e4", 1.0e4)] {
        let prim = move |[x, _]: [f64; 2]| (rho, [0.0, 0.1 * (k * x).sin(), 0.0], 1.0);
        let cell = move |[x, y]: [f64; 2]| [1.0 + 0.1 * (k * y).sin(), 0.1 * (k * x).sin(), 0.0];
        let face = move |a: usize, [x, y]: [f64; 2]| if a == 0 { 1.0 + 0.1 * (k * y).sin() } else { 0.1 * (k * x).sin() };
        let r = ratios_2p5d(&move || build_2p5d(n, prim, cell, face), dt);
        for (i, nm) in ["bface", "bz", "energy", "pressure"].iter().enumerate() {
            if i == 1 { continue; }
            println!("KINEMATIC {label:>8} {nm:>8}: ratios {:.2} {:.2}", r[i].0, r[i].1);
        }
    }
}

// the same Alfvenic fixture under the UCT EMF (no mass-flux soft-sign blend), HLLD.
#[test]
fn diag_alfvenic_h_step_under_uct() {
    let l2 = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum::<f64>().sqrt();
    let n = 64usize;
    let dt = 2.5e-4;
    let k = 2.0 * std::f64::consts::PI;
    for (label, rho) in [("rho=1", 1.0), ("rho=1e4", 1.0e4)] {
        let prim = move |[x, _]: [f64; 2]| (rho, [0.0, 0.1 * (k * x).sin(), 0.0], 1.0);
        let cell = move |[x, y]: [f64; 2]| [1.0 + 0.1 * (k * y).sin(), 0.1 * (k * x).sin(), 0.0];
        let face = move |a: usize, [x, y]: [f64; 2]| if a == 0 { 1.0 + 0.1 * (k * y).sin() } else { 0.1 * (k * x).sin() };
        let run = |dt: f64, nsteps: usize| -> [Vec<f64>; 3] {
            let sim = build_2p5d(n, prim, cell, face);
            let sub = Kernels::new(GAMMA, 0.3, 1.0, &sim.geom.allocated)
                .with_solver(Solver::Hlld)
                .expect("hlld")
                .ct_method(CtMethod::Uct);
            let mut hier = Hierarchy::single(sim, sub);
            hier.prime();
            for _ in 0..nsteps {
                assert!(!hier.hydro_map(0, dt), "H stage retry");
            }
            let (bf, _bc, nrg, pre) = hier.slip_state_snapshots(0);
            [bf, nrg, pre]
        };
        let (u1, u2, u3, u4) = (run(dt, 8), run(dt / 2.0, 16), run(dt / 4.0, 32), run(dt / 8.0, 64));
        for (i, nm) in ["bface", "energy", "pressure"].iter().enumerate() {
            let (e1, e2, e3) = (l2(&u1[i], &u2[i]), l2(&u2[i], &u3[i]), l2(&u3[i], &u4[i]));
            println!("UCT {label:>8} {nm:>8}: ratios {:.2} {:.2}", e1 / e2.max(1e-300), e2 / e3.max(1e-300));
        }
    }
}

// the shear flow with an out-of-plane component under the UCT EMF and the two-wave solvers, which
// carry neither the Contact sign switch nor the HLLD degenerate fan.
fn ratios_2p5d_with(sim_of: &dyn Fn() -> Sim, dt: f64, solver: Solver, ct: CtMethod) -> [(f64, f64); 4] {
    let l2 = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum::<f64>().sqrt();
    let run = |dt: f64, nsteps: usize| -> [Vec<f64>; 4] {
        let sim = sim_of();
        let sub = Kernels::new(GAMMA, 0.3, 1.0, &sim.geom.allocated)
            .with_solver(solver)
            .expect("solver")
            .ct_method(ct);
        let mut hier = Hierarchy::single(sim, sub);
        hier.prime();
        for _ in 0..nsteps {
            assert!(!hier.hydro_map(0, dt), "H stage retry");
        }
        let (bf, bc, nrg, pre) = hier.slip_state_snapshots(0);
        let ncells = nrg.len();
        [bf, bc[2 * ncells..3 * ncells].to_vec(), nrg, pre]
    };
    let (u1, u2, u3, u4) = (run(dt, 8), run(dt / 2.0, 16), run(dt / 4.0, 32), run(dt / 8.0, 64));
    std::array::from_fn(|i| {
        let (e1, e2, e3) = (l2(&u1[i], &u2[i]), l2(&u2[i], &u3[i]), l2(&u3[i], &u4[i]));
        (e1 / e2.max(1e-300), e2 / e3.max(1e-300))
    })
}

#[test]
fn diag_shear_with_bz_under_uct_two_wave_solvers() {
    let k = 2.0 * std::f64::consts::PI;
    let flow_prim = move |[x, _]: [f64; 2]| (1.0, [0.0, 0.1 * (k * x).sin(), 0.0], 1.0);
    let flowz_cell = move |[x, y]: [f64; 2]| [1.0 + 0.1 * (k * y).sin(), 0.1 * (k * x).sin(), 0.2 * (k * x).cos() * (k * y).cos()];
    let face = move |a: usize, [x, y]: [f64; 2]| if a == 0 { 1.0 + 0.1 * (k * y).sin() } else { 0.1 * (k * x).sin() };
    let dt = 2.5e-4;
    for (label, solver) in [("HLLE", Solver::Hlle), ("HLLC", Solver::Hllc), ("HLLD", Solver::Hlld)] {
        for n in [32usize, 64] {
            let r = ratios_2p5d_with(&move || build_2p5d(n, flow_prim, flowz_cell, face), dt, solver, CtMethod::Uct);
            for (i, nm) in ["bface", "bz", "energy", "pressure"].iter().enumerate() {
                println!("SHEAR+BZ UCT+{label:<4} N={n:<3} {nm:>8}: ratios {:.2} {:.2}", r[i].0, r[i].1);
            }
        }
    }
}
