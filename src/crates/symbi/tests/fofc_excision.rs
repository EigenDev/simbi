// =============================================================================
// fofc_excision.rs
//
// the first-order flux correction and horizon excision on a magnetized
// cartesian kerr-schild box: uniform gas at rest in a vertical field
// B^z = B0 / sqrt(gamma), falling onto a hole of mass 1 excised at
// r_ks < 1.4. the recovery writes its candidate primitives on a rejected
// cell, so these gates pin how the excised region meets the correction:
// - a rejected recovery inside the excised region starts no correction: the
//   cell is dropped from the troubled flag before the count, and no redo
//   touches it;
// - a rejected recovery on a live cell beside the excision surface still
//   corrects, its constrained-transport edges included, while faces away
//   from it keep the high-order curl bit for bit;
// - every flux sweep reads valid excised cells: the first-order redo's
//   stage input and the next stage's high-order input, with an exterior
//   correction running while the core holds a rejected candidate, and with
//   no correction at all;
// - tiles cut through the excision boundary evolve bit for bit as the single
//   grid,
//   through steps whose recovery rejects excised cells;
// - with no excision configured, the same rejected core cell is flagged.
// the rejections are made by writing a negative energy into chosen cells just
// before the recovery, so the trouble enters through the recovery's own
// status channel.
//
// run: cargo test -p symbi --test fofc_excision
// =============================================================================

use symbi::regimes::substrate_kernels::Solver;
use symbi::regimes::substrate_rmhd::RmhdSubstrateKernelSet3D;
use symbi::sim::decomp::{LocalCopy, evolve_decomposed, unflatten};
use symbi::sim::evolve::{KernelSet, evolve};
use symbi::sim::state::*;
use symbi::sim::substrate_seam::WithExcision;
use symbi_algebra::Tensor;
use symbi_geometry::{Metric, SchwarzschildKSCartesian};
use symbi_hydro::eos::IdealGas;
use symbi_hydro::mhd_state::MhdPrim;
use symbi_hydro::quantity::{Density, Pressure};
use symbi_hydro::rmhd::Rmhd;
use symbi_hydro::state::Prim;
use symbi_sim::stage::{
    HookPoint, SingleGridStep, StageArgs, StageOutcome, fold_stage_excision, fold_stage_from_correction,
    fold_stage_through_mark,
};
use symbi_xpu::{CpuSpace, HostMemory};

type Sim = SimState<Rmhd, 3, SchwarzschildKSCartesian<f64>, IdealGas<f64>, CpuSpace, HostMemory>;
type Kern = RmhdSubstrateKernelSet3D<HostMemory, f64>;

const N: usize = 16;
const L: f64 = 5.0;
const DX: f64 = 2.0 * L / N as f64;
const MASS: f64 = 1.0;
const R_EXC: f64 = 1.4;
const GAMMA: f64 = 4.0 / 3.0;
const CFL: f64 = 0.3;
const RHO0: f64 = 1.0;
const PRE0: f64 = 0.1;
const BZ0: f64 = 0.05;
/// the time the box falls before a hand-driven stage, so the gas moves
const T_PRIME: f64 = 0.3;
/// the energy written into a cell to make its recovery reject it
const POISON: f64 = -1.0;
/// the cell whose centroid is nearest the hole, inside the excised region
const CORE: [usize; 3] = [N / 2, N / 2, N / 2];
/// a live cell beside the excision surface: its neighbor toward the hole is excised
const RIM: [usize; 3] = [N / 2 + 2, N / 2, N / 2];

fn metric() -> SchwarzschildKSCartesian<f64> {
    SchwarzschildKSCartesian { mass: MASS }
}

/// a grid of `cells` placed as production places a tile: the whole box's origin and the index
/// offset of its first cell
fn build(cells: [usize; 3], dx: f64, offset: [usize; 3], bnd: Boundaries<3>, r_exc: f64) -> (Sim, Kern) {
    let m = metric();
    let bz = move |x: [f64; 3]| BZ0 / m.sqrt_det_gamma(Tensor::new(x));
    let sim = Sim::build(Rmhd, IdealGas { gamma: GAMMA }, metric())
        .cells(cells)
        .spacing([dx; 3])
        .origin([-L; 3])
        .index_offset(std::array::from_fn(|a| offset[a] as isize))
        .boundaries(bnd)
        .cfl(CFL)
        .timestepping(Timestepping::Rk2)
        .allocate()
        .expect("sim")
        .set_initial(|x| {
            MhdPrim::new(
                Prim::adiabatic(Density(RHO0), Tensor::new([0.0; 3]), Pressure(PRE0)),
                Tensor::new([0.0, 0.0, bz(x)]),
            )
        })
        .seed_faces(|axis, x| if axis == 2 { bz(x) } else { 0.0 })
        .build();
    let k = Kern::new(GAMMA, CFL, 1.0, &sim.geom.allocated)
        .with_solver(Solver::Hlle)
        .expect("solver")
        .ct_method(CtMethod::Uct)
        .with_excision(r_exc, 1.0, 1.0);
    (sim, k)
}

fn whole(r_exc: f64) -> (Sim, Kern) {
    build([N; 3], DX, [0; 3], Boundaries::uniform(BoundaryType::Outflow), r_exc)
}

/// the storage index of global interior cell `g` on a store holding the whole grid
fn at(sim: &Sim, g: [usize; 3]) -> [isize; 3] {
    std::array::from_fn(|a| sim.geom.interior.spaces[a].lo + g[a] as isize)
}

/// whether a store cell lies in the excised region, by the fill's own test at its centroid
fn excised(sim: &Sim, origin: [f64; 3], c: [isize; 3]) -> bool {
    let x: [f64; 3] = std::array::from_fn(|a| origin[a] + ((c[a] - sim.geom.interior.spaces[a].lo) as f64 + 0.5) * DX);
    symbi_ib::excise::ks_excised(&x, 0.0, R_EXC)
}

fn valid(sim: &Sim, c: [isize; 3]) -> bool {
    let rho = *sim.fields.prim.rho.view().at(c);
    let pre = *sim.fields.prim.pre_field().unwrap().view().at(c);
    let v2: f64 = (0..3).map(|k| sim.fields.prim.vel[k].view().at(c).powi(2)).sum();
    rho.is_finite() && pre.is_finite() && v2.is_finite() && rho > 0.0 && pre > 0.0
}

fn args(sim: &Sim, k: &Kern) -> StageArgs {
    let dt = k.cfl(sim).min(0.1);
    assert!(dt.is_finite() && dt > 0.0, "no finite timestep ({dt:e})");
    StageArgs {
        dt,
        a0: 0.0,
        ac: 1.0,
        stage: 0,
        n_stages: 2,
        injection_weight: symbi_sim::driver::downstream_injection_weight(Timestepping::Rk2.stages(), 0),
        allow_elision: false,
        exit_fraction: symbi_sim::driver::stage_exit_fraction(Timestepping::Rk2.stages(), 0),
    }
}

/// the first span of an rk2 predictor stage with `poisoned` cells given a negative energy just
/// before the recovery; returns the stage arguments and the marked troubled count
fn poisoned_mark(sim: &Sim, k: &Kern, poisoned: &[[isize; 3]]) -> (StageArgs, u64) {
    let args = args(sim, k);
    // the step entry, as the driver captures it before the first stage: the rollback state and
    // the step-entry conserved state and faces the redo re-curls from
    k.snapshot_retry(sim);
    k.snapshot(sim);
    let nrg = sim.fields.cons.nrg_field().expect("nrg");
    let mut hook = |hp: HookPoint| {
        if matches!(hp, HookPoint::BeforeC2p) {
            for c in poisoned {
                *nrg.view_mut().at_mut(*c) = POISON;
            }
        }
    };
    let marked = fold_stage_through_mark(sim, k, args, &mut hook);
    for c in poisoned {
        assert_ne!(*sim.fields.c2p_error.view().at(*c), 0.0, "the recovery accepted poisoned cell {c:?}; the gate is vacuous");
    }
    (args, marked)
}

fn correct(sim: &Sim, k: &Kern, args: StageArgs) {
    let outcome = fold_stage_from_correction(sim, k, args, false, &mut |_| {});
    assert!(outcome == StageOutcome::Accepted, "the stage was rejected");
}

fn primed(r_exc: f64) -> (Sim, Kern) {
    let (mut sim, k) = whole(r_exc);
    evolve(&mut sim, &k, T_PRIME).expect("prime");
    (sim, k)
}

/// the interior faces the constrained transport owns, one list per axis; the ghost faces are
/// refilled by the stage's ghost fill after the correction
fn faces(sim: &Sim) -> Vec<Vec<f64>> {
    let mhd = sim.fields.mhd.as_ref().unwrap();
    (0..3).map(|d| sim.geom.interior.extend(d, 0, 1).iter().map(|c| *mhd.bface[d].view().at(c)).collect()).collect()
}

#[test]
fn core_only_trouble_starts_no_correction() {
    let (sim, k) = primed(R_EXC);
    let core = at(&sim, CORE);
    assert!(excised(&sim, [-L; 3], core), "the core cell is live; the gate is vacuous");
    let (args, marked) = poisoned_mark(&sim, &k, &[core]);
    assert_eq!(marked, 0, "a rejected excised cell was flagged");
    correct(&sim, &k, args);
    // a redo restores the stage input on every cell; the poisoned energy surviving the
    // correction span shows none ran
    assert_eq!(*sim.fields.cons.nrg_field().unwrap().view().at(core), POISON, "a correction ran on core-only trouble");
}

#[test]
fn trouble_beside_the_excision_surface_corrects_its_ct_edges() {
    let (sim, k) = primed(R_EXC);
    let rim = at(&sim, RIM);
    let inward = [rim[0] - 1, rim[1], rim[2]];
    assert!(!excised(&sim, [-L; 3], rim) && excised(&sim, [-L; 3], inward), "the rim cell is not beside the excision surface");
    let (args, marked) = poisoned_mark(&sim, &k, &[rim]);
    assert_eq!(marked, 1, "the rejected live cell beside the surface was not flagged");
    let high_order = faces(&sim);
    correct(&sim, &k, args);
    assert_ne!(*sim.fields.cons.nrg_field().unwrap().view().at(rim), POISON, "the flagged cell was not corrected");
    let corrected = faces(&sim);
    let (mut near_moved, mut far_moved) = (0usize, 0usize);
    for d in 0..3 {
        for (ii, c) in sim.geom.interior.extend(d, 0, 1).iter().enumerate() {
            if corrected[d][ii].to_bits() == high_order[d][ii].to_bits() {
                continue;
            }
            // a face of axis d at index c separates cells c - e_d and c: it touches an edge of
            // the rim cell when both cells lie within one cell of it
            let near = (0..3).all(|a| {
                let lo = if a == d { c[a] - 1 } else { c[a] };
                lo >= rim[a] - 1 && c[a] <= rim[a] + 1
            });
            if near {
                near_moved += 1;
            } else {
                far_moved += 1;
            }
        }
    }
    assert!(near_moved > 0, "no face on the flagged cell's edges took the first-order EMF");
    assert_eq!(far_moved, 0, "{far_moved} faces away from the flagged cell left the high-order curl");
}

/// the first-order redo's flux-entry state, reproduced on a twin: the correction restores the
/// stage-input conserved state and the stage-input cell field onto the live fields and recovers
/// the primitives, then sweeps the first-order fluxes from them
fn redo_entry(sim: &Sim, k: &Kern) {
    let cons = &sim.fields.cons;
    let input = sim.stage_input();
    let copy = |dst: &symbi_grid::Field<f64, 3, HostMemory>, src: &symbi_grid::Field<f64, 3, HostMemory>| {
        let (mut d, s) = (dst.view_mut(), src.view());
        for c in dst.domain().iter() {
            *d.at_mut(c) = *s.at(c);
        }
    };
    copy(&cons.den, &input.den);
    for m in 0..3 {
        copy(&cons.mom[m], &input.mom[m]);
    }
    copy(cons.nrg_field().unwrap(), input.nrg_field().unwrap());
    let mhd = sim.fields.mhd.as_ref().unwrap();
    for m in 0..3 {
        copy(&mhd.bcell[m], &mhd.bcell_stage[m]);
    }
    k.c2p(sim);
}

#[test]
fn every_flux_sweep_reads_valid_excised_cells() {
    // an exterior correction runs while the core holds a rejected candidate
    let (sim, k) = primed(R_EXC);
    let (twin, twin_k) = primed(R_EXC);
    let (core, rim) = (at(&sim, CORE), at(&sim, RIM));
    let (args, marked) = poisoned_mark(&sim, &k, &[core, rim]);
    assert_eq!(marked, 1, "the correction was not triggered by the exterior cell alone");
    let (_, twin_marked) = poisoned_mark(&twin, &twin_k, &[core, rim]);
    assert_eq!(twin_marked, marked);
    redo_entry(&twin, &twin_k);
    for c in twin.geom.interior.iter() {
        if excised(&twin, [-L; 3], c) {
            assert_eq!(*twin.fields.c2p_error.view().at(c), 0.0, "the redo's recovery rejected excised cell {c:?}");
            assert!(valid(&twin, c), "the redo's first-order sweep reads an invalid excised cell {c:?}");
        }
    }
    correct(&sim, &k, args);
    fold_stage_excision(&[&sim], &[&k], &mut SingleGridStep).expect("stage excision");
    // the next stage's high-order sweep reads every allocated cell
    for c in sim.geom.allocated.iter() {
        assert!(valid(&sim, c), "the next high-order sweep reads an invalid cell {c:?}");
    }

    // no correction runs, so only the stage-end fill stands between the candidate and the sweep
    let (sim, k) = primed(R_EXC);
    let core = at(&sim, CORE);
    let (args, marked) = poisoned_mark(&sim, &k, &[core]);
    assert_eq!(marked, 0);
    correct(&sim, &k, args);
    assert!(!valid(&sim, core), "the rejected candidate is already valid; the fill timing is untested");
    fold_stage_excision(&[&sim], &[&k], &mut SingleGridStep).expect("stage excision");
    let deep = at(&sim, [N / 2 - 1, N / 2 - 1, N / 2 - 1]);
    assert!(excised(&sim, [-L; 3], deep));
    for c in sim.geom.allocated.iter() {
        assert!(valid(&sim, c), "the next high-order sweep reads an invalid cell {c:?}");
    }
    assert_eq!(
        sim.fields.prim.rho.view().at(core).to_bits(),
        sim.fields.prim.rho.view().at(deep).to_bits(),
        "the rejected core cell does not hold the floor the rest of the excised region holds"
    );
}

#[test]
fn an_unexcised_run_flags_every_rejected_cell() {
    let (sim, k) = primed(0.0);
    let core = at(&sim, CORE);
    let (_, marked) = poisoned_mark(&sim, &k, &[core]);
    assert_eq!(marked, 1, "with no excision configured the rejected core cell went unflagged");
}

/// the tiled box's resolution: the recovery rejects excised cells from the second step on
const NT: usize = 32;
const DXT: f64 = 2.0 * L / NT as f64;

/// the tile grid over the origin-centered box; every cut passes through the hole
fn tiles(counts: [usize; 3]) -> Vec<(Sim, Kern)> {
    let m: [usize; 3] = std::array::from_fn(|a| NT / counts[a]);
    (0..counts.iter().product())
        .map(|flat| {
            let tc = unflatten(flat, counts);
            let offset: [usize; 3] = std::array::from_fn(|a| tc[a] * m[a]);
            let bnd = Boundaries(std::array::from_fn(|a| {
                let face = |edge: bool| if edge { BoundaryType::Outflow } else { BoundaryType::CoarseFine };
                [face(tc[a] == 0), face(tc[a] == counts[a] - 1)]
            }));
            build(m, DXT, offset, bnd, R_EXC)
        })
        .collect()
}

/// the excised interior cells of a tile whose recovery was rejected
fn rejected_excised(s: &Sim, origin: [f64; 3]) -> u64 {
    let status = s.fields.c2p_error.view();
    s.geom
        .interior
        .iter()
        .filter(|&c| {
            let x: [f64; 3] = std::array::from_fn(|a| origin[a] + ((c[a] - s.geom.interior.spaces[a].lo) as f64 + 0.5) * DXT);
            symbi_ib::excise::ks_excised(&x, 0.0, R_EXC) && *status.at(c) != 0.0
        })
        .count() as u64
}

/// every tile's interior conserved state, cell field and staggered faces keyed by global index
fn record(ts: &[(Sim, Kern)], counts: [usize; 3]) -> Vec<(usize, [usize; 3], u64)> {
    let m: [usize; 3] = std::array::from_fn(|a| NT / counts[a]);
    let mut out = Vec::new();
    for (flat, (s, _)) in ts.iter().enumerate() {
        let tc = unflatten(flat, counts);
        let lo: [isize; 3] = std::array::from_fn(|a| s.geom.interior.spaces[a].lo);
        let global = |c: [isize; 3]| -> [usize; 3] { std::array::from_fn(|a| tc[a] * m[a] + (c[a] - lo[a]) as usize) };
        let cons = &s.fields.cons;
        let mut fields = vec![&cons.den, &cons.mom[0], &cons.mom[1], &cons.mom[2], cons.nrg_field().unwrap()];
        let mhd = s.fields.mhd.as_ref().unwrap();
        fields.extend(mhd.bcell.b.iter());
        for (f, field) in fields.iter().enumerate() {
            for c in s.geom.interior.iter() {
                out.push((f, global(c), field.view().at(c).to_bits()));
            }
        }
        for d in 0..3 {
            for c in s.geom.interior.extend(d, 0, 1).iter() {
                out.push((10 + d, global(c), mhd.bface[d].view().at(c).to_bits()));
            }
        }
    }
    // a face on a cut is recorded by both of its tiles; equal copies merge, a disagreement stays
    out.sort();
    out.dedup();
    out
}

/// the whole box through the uni-grid driver, with the number of excised cells whose recovery
/// was rejected, summed over the steps
fn run_single(t_final: f64) -> (Vec<(usize, [usize; 3], u64)>, u64) {
    let (mut sim, k) = build([NT; 3], DXT, [0; 3], Boundaries::uniform(BoundaryType::Outflow), R_EXC);
    let mut rejected = 0u64;
    symbi::sim::evolve::evolve_with_callback(&mut sim, &k, t_final, 1, |s| rejected += rejected_excised(s, [-L; 3]))
        .expect("single-grid run");
    (record(&[(sim, k)], [1, 1, 1]), rejected)
}

/// the tiles through the decomposed driver
fn run_tiled(counts: [usize; 3], t_final: f64) -> Vec<(usize, [usize; 3], u64)> {
    let mut ts = tiles(counts);
    let devices = vec![0i32; ts.len()];
    {
        let mut stores = Vec::new();
        let mut kernels = Vec::new();
        for (s, k) in ts.iter_mut() {
            stores.push(&mut **s);
            kernels.push(&*k);
        }
        evolve_decomposed(&mut stores, &kernels, counts, &devices, Timestepping::Rk2, 0.0, t_final, u64::MAX, &LocalCopy, |_, _, _| {
            std::ops::ControlFlow::Continue(())
        });
    }
    record(&ts, counts)
}

#[test]
fn a_cut_through_the_excision_boundary_matches_the_single_grid() {
    let t_final = 0.3;
    let (single, rejected) = run_single(t_final);
    assert!(rejected > 0, "no excised cell's recovery was rejected; the gate is vacuous");
    let tiled = run_tiled([2, 2, 1], t_final);
    assert_eq!(single.len(), tiled.len(), "the tiled run records a different set of values");
    let differ = single.iter().zip(&tiled).filter(|(a, b)| a != b).count();
    assert_eq!(differ, 0, "{differ} values differ between the tiled and the single-grid run");
}

