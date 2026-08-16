// =============================================================================
// decomp_source_equivalence.rs
//
// the source correctness contract for multi-gpu domain decomposition: a domain split into tiles,
// each carrying the same runtime source collection, evolved with the additive source pass in
// `evolve_decomposed`, must reproduce the monolithic run to round-off.
//
// two independently parameterized forces sum to `a = [x, 0]`; both use local parameter index zero,
// so the collection must isolate their values before composing the shared momentum and energy
// targets. the position leaf uses the cell's global x. this is the decisive cross-tile test: a tile
// that evaluated the source at its tile-local
// coordinate would diverge from the monolithic run at every cut. a
// only a position-dependent force exposes that bug. the source also exercises the energy work overlay
// (S_nrg = mom . a) since Newtonian carries energy, so both overlays go through `source_apply`.
//
// the additive path is what the decomposed loop drives, with fusion left off: `evolve_decomposed` calls a
// plain `godunov_stage` and then `source_apply(ac*dt)` after `snapshot_stage` captured the stage
// input -- the same two-pass protocol the single-grid `step()` runs (evolve.rs STAGE_PIPELINE),
// proven equal to the fused stage by `jit_fused_equals_two_pass`. cpu-only + 2d: same exchange
// index math as the gpu path, the fast iteration loop for the source pass.
// =============================================================================

use symbi::prelude::SimSubstrate;
use symbi::regimes::substrate_newton::AdiabaticSubstrateKernelSet;
use symbi::sim::decomp::{LocalCopy, evolve_decomposed, flatten, gather_interiors, unflatten};
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_geometry::Cartesian;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::expr_bridge::build_user_sources;
use symbi_hydro::newtonian::Newtonian;
use symbi_hydro::state::Prim;
use symbi_hydro::{NEWTONIAN_SPEC, SourceConfig};
use symbi_xpu::{CpuSpace, HostMemory};

const GAMMA: f64 = 1.4;
const CFL: f64 = 0.4;
const N: usize = 32; // cells per axis
const DX: f64 = 1.0 / N as f64;
const T_FINAL: f64 = 0.04;

type Sim = SimState<Newtonian, 2, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;
type Kern = AdiabaticSubstrateKernelSet<HostMemory, f64, 2>;

// a position-dependent force `a_x = x_0` (VARIABLE_X1, the cell's global x), `a_y = 0`. the energy
// regime lowers this into a momentum overlay (S_mom = rho*a) together with an nrg work overlay; both
// flow through the additive `source_apply` pass in `evolve_decomposed`.
const SOURCE_JSON_1: &str = r#"{
    "kind": "force", "dim": 2, "outputs": [2, 3], "params": [0.25],
    "nodes": [ {"op": "PARAMETER", "param_idx": 0}, {"op": "VARIABLE_X1"},
               {"op": "MULTIPLY", "left": 0, "right": 1},
               {"op": "CONSTANT", "value": 0.0} ]
}"#;

const SOURCE_JSON_2: &str = r#"{
    "kind": "force", "dim": 2, "outputs": [2, 3], "params": [0.75],
    "nodes": [ {"op": "PARAMETER", "param_idx": 0}, {"op": "VARIABLE_X1"},
               {"op": "MULTIPLY", "left": 0, "right": 1},
               {"op": "CONSTANT", "value": 0.0} ]
}"#;

const SOURCE_JSON_TABLE_2D: &str = r#"{
    "kind":"force","dim":2,"outputs":[44,45],"params":[],"nodes":[
        {"op":"VARIABLE_X1"},{"op":"VARIABLE_X2"},
        {"op":"CONSTANT","value":0.0},{"op":"CONSTANT","value":1.0},
        {"op":"CONSTANT","value":0.0},{"op":"SUBTRACT","left":0,"right":4},
        {"op":"CONSTANT","value":1.0},{"op":"DIVIDE","left":5,"right":6},
        {"op":"SUBTRACT","left":3,"right":2},{"op":"MULTIPLY","left":7,"right":8},
        {"op":"ADD","left":2,"right":9},{"op":"CONSTANT","value":0.0},
        {"op":"LT","left":0,"right":11},{"op":"CONSTANT","value":1.0},
        {"op":"GT","left":0,"right":13},
        {"op":"IF_THEN_ELSE","condition":14,"true_case":3,"false_case":10},
        {"op":"IF_THEN_ELSE","condition":12,"true_case":2,"false_case":15},
        {"op":"CONSTANT","value":1.0},{"op":"CONSTANT","value":2.0},
        {"op":"CONSTANT","value":0.0},{"op":"SUBTRACT","left":0,"right":19},
        {"op":"CONSTANT","value":1.0},{"op":"DIVIDE","left":20,"right":21},
        {"op":"SUBTRACT","left":18,"right":17},{"op":"MULTIPLY","left":22,"right":23},
        {"op":"ADD","left":17,"right":24},{"op":"CONSTANT","value":0.0},
        {"op":"LT","left":0,"right":26},{"op":"CONSTANT","value":1.0},
        {"op":"GT","left":0,"right":28},
        {"op":"IF_THEN_ELSE","condition":29,"true_case":18,"false_case":25},
        {"op":"IF_THEN_ELSE","condition":27,"true_case":17,"false_case":30},
        {"op":"CONSTANT","value":0.0},{"op":"SUBTRACT","left":1,"right":32},
        {"op":"CONSTANT","value":1.0},{"op":"DIVIDE","left":33,"right":34},
        {"op":"SUBTRACT","left":31,"right":16},{"op":"MULTIPLY","left":35,"right":36},
        {"op":"ADD","left":16,"right":37},{"op":"CONSTANT","value":0.0},
        {"op":"LT","left":1,"right":39},{"op":"CONSTANT","value":1.0},
        {"op":"GT","left":1,"right":41},
        {"op":"IF_THEN_ELSE","condition":42,"true_case":31,"false_case":38},
        {"op":"IF_THEN_ELSE","condition":40,"true_case":16,"false_case":43},
        {"op":"CONSTANT","value":0.0}
    ]
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
            Prim {
                rho: 1.0 + b,
                vel: Tensor::new([0.0; 2]),
                pre: 1.0 + b,
            }
        })
        .build();
    let table = r#"{
        "kind":"force", "dim":2, "outputs":[23,24], "params":[],
        "nodes":[
            {"op":"VARIABLE_X1"},{"op":"CONSTANT","value":0.0},
            {"op":"CONSTANT","value":0.0},{"op":"CONSTANT","value":1.0},
            {"op":"SUBTRACT","left":0,"right":1},
            {"op":"MULTIPLY","left":3,"right":4},{"op":"ADD","left":2,"right":5},
            {"op":"CONSTANT","value":0.5},{"op":"CONSTANT","value":0.5},
            {"op":"CONSTANT","value":-1.0},{"op":"SUBTRACT","left":0,"right":7},
            {"op":"MULTIPLY","left":9,"right":10},{"op":"ADD","left":8,"right":11},
            {"op":"CONSTANT","value":0.5},{"op":"LT","left":0,"right":13},
            {"op":"IF_THEN_ELSE","condition":14,"true_case":6,"false_case":12},
            {"op":"CONSTANT","value":0.0},{"op":"CONSTANT","value":0.0},
            {"op":"CONSTANT","value":0.0},{"op":"LT","left":0,"right":18},
            {"op":"CONSTANT","value":1.0},{"op":"GT","left":0,"right":20},
            {"op":"IF_THEN_ELSE","condition":21,"true_case":17,"false_case":15},
            {"op":"IF_THEN_ELSE","condition":19,"true_case":16,"false_case":22},
            {"op":"CONSTANT","value":0.0}
        ]
    }"#;
    let configs = [SOURCE_JSON_1, SOURCE_JSON_2, table, SOURCE_JSON_TABLE_2D]
        .map(|json| SourceConfig::from_json(json).expect("parse source config"));
    let (built, params) =
        build_user_sources(&configs, &NEWTONIAN_SPEC).expect("lower source collection");
    // two-pass, with fusion off: the decomposed loop drives plain godunov + the additive source_apply.
    // `with_runtime_source` sets has_additive_source -> evolve_decomposed runs snapshot_stage +
    // source_apply; the non-vacuous momentum check in `assert_source_matches` confirms it fires.
    let k = sim.substrate().with_runtime_source(built, params);
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
                let lo = if tc[a] == 0 {
                    BoundaryType::Outflow
                } else {
                    BoundaryType::CoarseFine
                };
                let hi = if tc[a] == counts[a] - 1 {
                    BoundaryType::Outflow
                } else {
                    BoundaryType::CoarseFine
                };
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
fn global_field(
    tiles: &[(Sim, Kern)],
    counts: [usize; 2],
    pick: impl Fn(&Sim, [isize; 2]) -> f64,
) -> Vec<f64> {
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
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f64, f64::max)
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
    assert!(
        total_momx > 1e-3,
        "source produced no momentum ({total_momx:e}); test is vacuous"
    );

    let de = max_err(&mono_den, &dec_den);
    let me = max_err(&mono_momx, &dec_momx);
    assert!(
        de < 1e-12,
        "{counts:?} {ts:?} density err {de:e} under runtime source"
    );
    assert!(
        me < 1e-12,
        "{counts:?} {ts:?} mom_x err {me:e} under runtime source"
    );

    // also the production gather path (the python checkpoint output).
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

// rk2 is the real test of the additive pass: snapshot_stage captures the stage input for each of
// the two stages, and source_apply uses the per-stage weight ac*dt -- a wrong weight or a stale
// snapshot diverges here even when euler passes.
#[test]
fn source_rk2_two_tile_x_cut() {
    assert_source_matches([2, 1], Timestepping::Rk2);
}

#[test]
fn source_rk2_two_tile_y_cut() {
    assert_source_matches([1, 2], Timestepping::Rk2);
}

// the 2x2 corner under rk2 -- both cuts together with the corrector source pass.
#[test]
fn source_rk2_quad_tile_2d_grid() {
    assert_source_matches([2, 2], Timestepping::Rk2);
}
