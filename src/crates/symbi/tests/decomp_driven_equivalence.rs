// =============================================================================
// decomp_driven_equivalence.rs
//
// the driven-boundary correctness contract for multi-gpu domain decomposition: a domain split
// into tiles, each registering the same boundary DAG, must reproduce the monolithic run to
// round-off. only edge tiles carry a Driven face (interior cuts are CoarseFine, halo-exchanged);
// each tile evaluates the coordinate prescription at its own global coordinates.
//
// the prescription is a position-dependent inflow on x_lo: rho = 2 + 0.2*y (VARIABLE_X2 -> the
// cell's global y). this is the decisive cross-tile test: with the domain cut along the driven
// face ([1, N] tiling), a tile that evaluated the prescription at its tile-local y would inject
// the wrong density on every non-origin tile. a constant inflow could not catch that bug. the
// [N, 1] tiling drives the inflow across the cut, exercising the halo exchange downstream of the
// driven fill.
// =============================================================================

use symbi::prelude::SimSubstrate;
use symbi::regimes::substrate_newton::AdiabaticSubstrateKernelSet;
use symbi::sim::decomp::{LocalCopy, evolve_decomposed, flatten, unflatten};
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_geometry::Cartesian;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::expr_bridge::build_boundary_dag;
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

fn boundary_json() -> String {
    // rho = 2 + 0.2*y as node 2' = add(node 3 (=2.0), node 2 (=0.2*y)); assembled here so the
    // node list stays readable: outputs are [rho, v1, v2, p].
    r#"{
        "kind": "dirichlet", "dim": 2, "outputs": [4, 5, 6, 7], "params": [],
        "nodes": [ {"op": "VARIABLE_X2"}, {"op": "CONSTANT", "value": 0.2},
                   {"op": "MULTIPLY", "left": 0, "right": 1}, {"op": "CONSTANT", "value": 2.0},
                   {"op": "ADD", "left": 3, "right": 2},
                   {"op": "CONSTANT", "value": 0.5}, {"op": "CONSTANT", "value": 0.0},
                   {"op": "CONSTANT", "value": 3.0} ]
    }"#
    .to_string()
}

// a quiescent interior, distinct from the inflow so the fill provably comes from the DAG.
fn make(cells: [usize; 2], origin: [f64; 2], bnd: Boundaries<2>) -> (Sim, Kern) {
    let sim = Sim::build(Newtonian, IdealGas { gamma: GAMMA }, Cartesian)
        .cells(cells)
        .spacing([DX; 2])
        .origin(origin)
        .boundaries(bnd)
        .cfl(CFL)
        .timestepping(Timestepping::Rk2)
        .allocate()
        .expect("sim construction failed")
        .set_initial(|_| Prim {
            rho: 1.0,
            vel: Tensor::new([0.0; 2]),
            pre: 1.0,
        })
        .build();
    let cfg = SourceConfig::from_json(&boundary_json()).expect("parse boundary config");
    let built = build_boundary_dag(&cfg, &NEWTONIAN_SPEC).expect("lower boundary dag");
    let (k, id) = sim
        .substrate()
        .with_driven_boundary(built, cfg.params.clone());
    assert_eq!(id, 0, "first registration is id 0 (matches Driven(0))");
    (sim, k)
}

// the tile grid: x_lo driven on the tiles that own the physical x_lo face, outflow on the other
// physical faces, CoarseFine on interior cuts — the same per-face copy the production decomposed
// build performs.
fn grid_tiles(counts: [usize; 2]) -> Vec<(Sim, Kern)> {
    let m: [usize; 2] = std::array::from_fn(|a| {
        assert!(N % counts[a] == 0, "N must split evenly into counts[{a}]");
        N / counts[a]
    });
    let total: usize = counts.iter().product();
    (0..total)
        .map(|flat| {
            let tc = unflatten(flat, counts);
            let origin = std::array::from_fn(|a| tc[a] as f64 * m[a] as f64 * DX);
            let phys_lo = [BoundaryType::Driven(0), BoundaryType::Outflow];
            let bnd = Boundaries(std::array::from_fn(|a| {
                let lo = if tc[a] == 0 {
                    phys_lo[a]
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
            make(m, origin, bnd)
        })
        .collect()
}

fn run(tiles: &mut [(Sim, Kern)], counts: [usize; 2]) {
    let devices: Vec<i32> = vec![0; tiles.len()];
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
        Timestepping::Rk2,
        0.0,
        T_FINAL,
        u64::MAX,
        &LocalCopy,
        |_, _, _| std::ops::ControlFlow::Continue(()),
    );
}

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

fn assert_driven_matches(counts: [usize; 2]) {
    let den = |s: &Sim, c| *s.fields.cons.den.view().at(c);
    let momx = |s: &Sim, c| *s.fields.cons.mom[0].view().at(c);

    let mut mono = grid_tiles([1, 1]);
    run(&mut mono, [1, 1]);
    let mono_den = global_field(&mono, [1, 1], den);
    let mono_momx = global_field(&mono, [1, 1], momx);

    let mut dec = grid_tiles(counts);
    run(&mut dec, counts);
    let dec_den = global_field(&dec, counts, den);
    let dec_momx = global_field(&dec, counts, momx);

    assert!(
        mono_den.iter().all(|v| v.is_finite()) && dec_den.iter().all(|v| v.is_finite()),
        "some global cells were never written"
    );
    // the inflow must have actually entered (else the test is vacuous): the driven face pushes
    // rho ~ 2 gas at v_x = 0.5 into a rho = 1 box, so total mom_x is well above zero.
    let total_momx: f64 = mono_momx.iter().map(|v| v.abs()).sum();
    assert!(
        total_momx > 1e-3,
        "driven inflow produced no momentum ({total_momx:e}); test is vacuous"
    );

    let de = max_err(&mono_den, &dec_den);
    let me = max_err(&mono_momx, &dec_momx);
    assert!(
        de < 1e-12,
        "{counts:?} density err {de:e} under driven boundary"
    );
    assert!(
        me < 1e-12,
        "{counts:?} mom_x err {me:e} under driven boundary"
    );
}

#[test]
fn driven_inflow_cut_perpendicular_to_the_face() {
    // [2, 1]: the cut is perpendicular to the driven face; only tile 0 owns it, and the injected
    // gas crosses the cut via the halo exchange.
    assert_driven_matches([2, 1]);
}

#[test]
fn driven_inflow_cut_along_the_face() {
    // [1, 2]: both tiles own a piece of the driven face; the y-dependent prescription is evaluated
    // at each tile's global y, so a local-coordinate bug diverges here.
    assert_driven_matches([1, 2]);
}

#[test]
fn driven_inflow_four_tiles() {
    assert_driven_matches([2, 2]);
}
