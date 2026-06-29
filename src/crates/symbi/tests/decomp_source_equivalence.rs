// =============================================================================
// decomp_source_equivalence.rs
//
// the SOURCE correctness contract for multi-gpu domain decomposition: a domain split into tiles,
// each carrying the SAME runtime user source, evolved with the additive (two-pass) source pass in
// `evolve_decomposed`, must reproduce the monolithic run to round-off.
//
// the source is a POSITION-DEPENDENT force `a = [x, 0]` (the expr DAG's VARIABLE_X1 -> the cell's
// global x). this is the decisive cross-tile test: a tile that evaluated the source at its LOCAL
// coordinate instead of its global one would diverge from the monolithic run at every cut. a
// constant force could not catch that bug. the source also exercises the energy work overlay
// (S_nrg = mom . a) since Newtonian carries energy, so both overlays go through `source_apply`.
//
// the additive (NOT fused) path is what the decomposed loop drives: `evolve_decomposed` calls a
// plain `godunov_stage` and then `source_apply(ac*dt)` after `snapshot_stage` captured the stage
// input -- the same two-pass protocol the single-grid `step()` runs (evolve.rs STAGE_PIPELINE),
// proven equal to the fused stage by `jit_fused_equals_two_pass`. cpu-only + 2d: same exchange
// index math as the gpu path, the fast iteration loop for the source pass.
// =============================================================================

use symbi::prelude::SimSubstrate;
use symbi::regimes::substrate_newton::AdiabaticSubstrateKernelSet;
use symbi::sim::decomp::{evolve_decomposed, flatten, gather_interiors, unflatten, LocalCopy};
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_geometry::Cartesian;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::expr_bridge::build_user_source;
use symbi_hydro::newtonian::Newtonian;
use symbi_hydro::state::Prim;
use symbi_hydro::{SourceConfig, NEWTONIAN_SPEC};
use symbi_xpu::{CpuSpace, HostMemory};

const GAMMA: f64 = 1.4;
const CFL: f64 = 0.4;
const N: usize = 32; // cells per axis
const DX: f64 = 1.0 / N as f64;
const T_FINAL: f64 = 0.04;

type Sim = SimState<Newtonian, 2, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;
type Kern = AdiabaticSubstrateKernelSet<HostMemory, f64, 2>;

// a position-dependent force `a_x = x_0` (VARIABLE_X1, the cell's GLOBAL x), `a_y = 0`. the energy
// regime lowers this into BOTH a momentum overlay (S_mom = rho*a) and an nrg work overlay; both
// flow through the additive `source_apply` pass in `evolve_decomposed`.
const SOURCE_JSON: &str = r#"{
    "kind": "force", "dim": 2, "outputs": [0, 1], "params": [],
    "nodes": [ {"op": "VARIABLE_X1"}, {"op": "CONSTANT", "value": 0.0} ]
}"#;

// a smooth centered density+pressure bump -> sound waves cross the cut while the source drives a
// position-dependent acceleration; outflow boundaries + a short time keep waves interior so mono ==
// decomposed is exact.
fn bump(x: f64) -> f64 {
    0.2 * (-(((x - 0.5) / 0.1).powi(2))).exp()
}

fn make(cells: [usize; 2], origin: [f64; 2], bnd: Boundaries<2>, ts: Timestepping) -> (Sim, Kern) {
    let sim = Sim::build(Newtonian, IdealGas { gamma: GAMMA }, Cartesian)
        .cells(cells)
        .spacing([DX; 2])
        .origin(origin)
        .boundaries(bnd)
        .cfl(CFL)
        .timestepping(ts)
        .allocate()
        .expect("sim construction failed")
        .set_initial(|x| {
            let b = bump(x[0]);
            Prim { rho: 1.0 + b, vel: Tensor::new([0.0; 2]), pre: 1.0 + b }
        })
        .build();
    let cfg = SourceConfig::from_json(SOURCE_JSON).expect("parse source config");
    let built = build_user_source(&cfg, &NEWTONIAN_SPEC).expect("lower source");
    // two-pass (NOT fused): the decomposed loop drives plain godunov + the additive source_apply.
    // `with_runtime_source` sets has_additive_source -> evolve_decomposed runs snapshot_stage +
    // source_apply; the non-vacuous momentum check in `assert_source_matches` confirms it fires.
    let k = sim.substrate().with_runtime_source(built, cfg.params.clone());
    (sim, k)
}

// build the tile grid: each tile an equal slice, CoarseFine on internal faces, Outflow on the outer
// domain boundary. each tile gets its own origin, so VARIABLE_X1 binds the correct global x.
fn grid_tiles(counts: [usize; 2], ts: Timestepping) -> Vec<(Sim, Kern)> {
    let m: [usize; 2] = std::array::from_fn(|a| {
        assert!(N % counts[a] == 0, "N must split evenly into counts[{a}]");
        N / counts[a]
    });
    let total: usize = counts.iter().product();
    (0..total)
        .map(|flat| {
            let tc = unflatten(flat, counts);
            let origin = std::array::from_fn(|a| tc[a] as f64 * m[a] as f64 * DX);
            let bnd = Boundaries(std::array::from_fn(|a| {
                let lo = if tc[a] == 0 { BoundaryType::Outflow } else { BoundaryType::CoarseFine };
                let hi = if tc[a] == counts[a] - 1 { BoundaryType::Outflow } else { BoundaryType::CoarseFine };
                [lo, hi]
            }));
            make(m, origin, bnd, ts)
        })
        .collect()
}

fn run(tiles: &mut [(Sim, Kern)], counts: [usize; 2], ts: Timestepping) {
    let devices: Vec<i32> = vec![0; tiles.len()];
    // evolve_decomposed takes the tiles by &mut (per-step body bookkeeping); build the &mut store
    // handles + & kernels from the same tiles.
    let mut stores = Vec::new();
    let mut kernels = Vec::new();
    for (s, k) in tiles.iter_mut() {
        stores.push(&mut **s);
        kernels.push(&*k);
    }
    evolve_decomposed(
        &mut stores,
        &kernels,
        counts,
        &devices,
        ts,
        0.0,
        T_FINAL,
        u64::MAX,
        &LocalCopy,
        |_, _, _| std::ops::ControlFlow::Continue(()),
    );
}

// scatter a per-cell field (selected by `pick`) from every tile interior into one global N^2 grid.
fn global_field(tiles: &[(Sim, Kern)], counts: [usize; 2], pick: impl Fn(&Sim, [isize; 2]) -> f64) -> Vec<f64> {
    let m: [usize; 2] = std::array::from_fn(|a| N / counts[a]);
    let mut out = vec![f64::NAN; N * N];
    for (flat_tile, (sim, _)) in tiles.iter().enumerate() {
        let tc = unflatten(flat_tile, counts);
        let ilo: [isize; 2] = std::array::from_fn(|a| sim.geom.interior.spaces[a].lo);
        for c in sim.geom.interior.iter() {
            let g: [usize; 2] = std::array::from_fn(|a| tc[a] * m[a] + (c[a] - ilo[a]) as usize);
            out[flatten(g, [N; 2])] = pick(sim, c);
        }
    }
    out
}

fn max_err(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0_f64, f64::max)
}

fn assert_source_matches(counts: [usize; 2], ts: Timestepping) {
    let den = |s: &Sim, c| *s.fields.cons.den.view().at(c);
    let momx = |s: &Sim, c| *s.fields.cons.mom[0].view().at(c);

    let mut mono = grid_tiles([1, 1], ts);
    run(&mut mono, [1, 1], ts);
    let mono_den = global_field(&mono, [1, 1], den);
    let mono_momx = global_field(&mono, [1, 1], momx);

    let mut dec = grid_tiles(counts, ts);
    run(&mut dec, counts, ts);
    let dec_den = global_field(&dec, counts, den);
    let dec_momx = global_field(&dec, counts, momx);

    assert!(
        mono_den.iter().all(|v| v.is_finite()) && dec_den.iter().all(|v| v.is_finite()),
        "some global cells were never written"
    );
    // the source must have actually moved gas (else the test proves nothing): the position force
    // accelerates every cell, so total |mom_x| is well above zero.
    let total_momx: f64 = mono_momx.iter().map(|v| v.abs()).sum();
    assert!(total_momx > 1e-3, "source produced no momentum ({total_momx:e}); test is vacuous");

    let de = max_err(&mono_den, &dec_den);
    let me = max_err(&mono_momx, &dec_momx);
    assert!(de < 1e-12, "{counts:?} {ts:?} density err {de:e} under runtime source");
    assert!(me < 1e-12, "{counts:?} {ts:?} mom_x err {me:e} under runtime source");

    // ALSO the production gather path (the python checkpoint output).
    let bnd = Boundaries(std::array::from_fn(|_| [BoundaryType::Outflow; 2]));
    let global = make([N, N], [0.0, 0.0], bnd, ts);
    let stores: Vec<_> = dec.iter().map(|(s, _)| &**s).collect();
    gather_interiors(&*global.0, &stores, counts);
    let gathered = global_field(std::slice::from_ref(&global), [1, 1], den);
    let ge = max_err(&gathered, &dec_den);
    assert!(ge < 1e-12, "{counts:?} {ts:?} gather density err {ge:e}");
}

#[test]
fn source_euler_two_tile_x_cut() {
    assert_source_matches([2, 1], Timestepping::Euler);
}

#[test]
fn source_euler_two_tile_y_cut() {
    assert_source_matches([1, 2], Timestepping::Euler);
}

// rk2 is the real test of the additive pass: snapshot_stage must capture the stage input for BOTH
// stages, and source_apply uses the per-stage weight ac*dt -- a wrong weight or a stale snapshot
// diverges here even when euler passes.
#[test]
fn source_rk2_two_tile_x_cut() {
    assert_source_matches([2, 1], Timestepping::Rk2);
}

#[test]
fn source_rk2_two_tile_y_cut() {
    assert_source_matches([1, 2], Timestepping::Rk2);
}

// the 2x2 corner under rk2 -- both cuts AND the corrector source pass.
#[test]
fn source_rk2_quad_tile_2d_grid() {
    assert_source_matches([2, 2], Timestepping::Rk2);
}
