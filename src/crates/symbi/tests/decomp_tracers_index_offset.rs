// =============================================================================
// decomp_tracers_index_offset.rs
//
// mass-transport tracers on tiles placed the way the launched workers place
// them: every tile carries the whole grid's origin and coordinate maps and the
// whole-grid index of its first cell, so each position it evaluates is the
// whole grid's bit for bit. the decomposed run reproduces the undivided grid
// exactly: every tracer lands on the same cell and at the same position.
// - on a geometrically graded x axis the cell volumes, face areas and centroids
//   differ from tile to tile, so a tile evaluating its local indices would
//   transport the wrong masses and re-home tracers wrongly;
// - on a uniform axis the volumes agree and the positions alone tell.
// the population is seeded once from the undivided grid and partitioned by the
// owner addresses it carries. the gate asserts its own premises: the second tile
// sits at a nonzero offset, holds tracers at the end, at least one tracer
// crossed the cut, and on the graded axis the widths differ across the cut.
// =============================================================================

use symbi::regimes::substrate_newton::AdiabaticSubstrateKernelSet;
use symbi::sim::decomp::{LocalCopy, Partition, evolve_decomposed};
use symbi::sim::state::*;
use symbi::sim::tracers::{TracerSet, seed_and_partition, seed_mass_weighted};
use symbi_algebra::Tensor;
use symbi_geometry::{AxisMap, Cartesian};
use symbi_hydro::eos::IdealGas;
use symbi_hydro::newtonian::Newtonian;
use symbi_hydro::quantity::{Density, Pressure};
use symbi_hydro::state::Prim;
use symbi_xpu::{CpuSpace, HostMemory};

const GAMMA: f64 = 1.4;
const CFL: f64 = 0.4;
const N: usize = 32;
const DRIFT: f64 = 0.4;
const T_FINAL: f64 = 0.4;
const N_TRACERS: usize = 240;

type Sim = SimState<Newtonian, 2, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;
type Kern = AdiabaticSubstrateKernelSet<HostMemory, f64, 2>;

fn blob(x: [f64; 2]) -> f64 {
    2.0 * (-(((x[0] - 0.4).powi(2) + (x[1] - 0.4).powi(2)) / 0.02)).exp()
}

/// the whole grid's x map: graded with ratio 1.08, or uniform, over [0, 1)
fn x_map(graded: bool) -> AxisMap {
    if graded {
        let ratio: f64 = 1.08;
        let width = (ratio - 1.0) / (ratio.powi(N as i32) - 1.0);
        AxisMap::Geometric { start: 0.0, width, ratio }
    } else {
        AxisMap::Uniform { start: 0.0, dx: 1.0 / N as f64 }
    }
}

/// a tile of `cells` whose first cell is whole-grid cell `lo`, carrying the whole grid's origin
/// and maps
fn tile(graded: bool, cells: [usize; 2], lo: [usize; 2], bnd: Boundaries<2>) -> (Sim, Kern) {
    let maps = [x_map(graded), AxisMap::Uniform { start: 0.0, dx: 1.0 / N as f64 }];
    let sim = Sim::build(Newtonian, IdealGas { gamma: GAMMA }, Cartesian)
        .cells(cells)
        .spacing([1.0 / N as f64; 2])
        .origin([0.0, 0.0])
        .coord_maps(Some(maps))
        .index_offset(std::array::from_fn(|a| lo[a] as isize))
        .boundaries(bnd)
        .timestepping(Timestepping::Rk2)
        .allocate()
        .expect("sim construction failed")
        .set_initial(|x| Prim::adiabatic(Density(1.0 + blob(x)), Tensor::new([DRIFT, DRIFT]), Pressure(1.0 + blob(x))))
        .build();
    let k = Kern::new(GAMMA, CFL, &sim.geom.allocated);
    (sim, k)
}

fn run(tiles: &mut [(Sim, Kern)], counts: [usize; 2]) {
    let devices: Vec<i32> = vec![0; tiles.len()];
    let mut stores = Vec::new();
    let mut kernels = Vec::new();
    for (s, k) in tiles.iter_mut() {
        stores.push(&mut **s);
        kernels.push(&*k);
    }
    evolve_decomposed(&mut stores, &kernels, counts, &devices, Timestepping::Rk2, 0.0, T_FINAL, u64::MAX, &LocalCopy, |_, _, _| {
        std::ops::ControlFlow::Continue(())
    });
}

/// (owner, position bits, holding tile) per tracer id, the seeded population and every tracer
/// the inflow spawned
fn gather(tiles: &[(Sim, Kern)]) -> std::collections::BTreeMap<u64, (u64, [u64; 2], usize)> {
    let mut out = std::collections::BTreeMap::new();
    for (t, (sim, _)) in tiles.iter().enumerate() {
        if let Some(tr) = sim.tracers.as_ref() {
            for i in 0..tr.len() {
                let previous = out.insert(tr.id[i], (tr.owner[i].0, tr.x[i].map(f64::to_bits), t));
                assert!(previous.is_none(), "tracer {} is held twice", tr.id[i]);
            }
        }
    }
    out
}

fn assert_offset_tiles_match_the_whole_grid(graded: bool) {
    let label = if graded { "graded x" } else { "uniform" };
    let whole_bnd = Boundaries::uniform(BoundaryType::Outflow);
    let (reference, _) = tile(graded, [N, N], [0, 0], whole_bnd);
    let seed: TracerSet<2> = seed_mass_weighted(&reference, N_TRACERS);
    let partition = Partition::explicit([N, N], [vec![N / 2], vec![]]).expect("partition");
    let per_tile = seed_and_partition(&reference, N_TRACERS, &partition);

    let (mut whole_sim, whole_k) = tile(graded, [N, N], [0, 0], whole_bnd);
    whole_sim.tracers = Some(seed);
    let mut whole = vec![(whole_sim, whole_k)];
    run(&mut whole, [1, 1]);

    let cut = |side: usize| {
        Boundaries::per_axis([
            if side == 0 { [BoundaryType::Outflow, BoundaryType::CoarseFine] } else { [BoundaryType::CoarseFine, BoundaryType::Outflow] },
            [BoundaryType::Outflow; 2],
        ])
    };
    let mut tiles = vec![tile(graded, [N / 2, N], [0, 0], cut(0)), tile(graded, [N / 2, N], [N / 2, 0], cut(1))];
    assert_eq!(tiles[1].0.geom.index_offset, [(N / 2) as isize, 0], "{label}: the second tile sits at a whole-grid offset");
    if graded {
        let w = |sim: &Sim, i: isize| sim.geom.cell_width([i, 0], 0);
        assert!(w(&tiles[1].0, 0) != w(&tiles[0].0, 0), "{label}: the widths agree across the cut; the graded arm is vacuous");
    }
    for ((t, _), set) in tiles.iter_mut().zip(per_tile) {
        t.tracers = Some(set);
    }
    run(&mut tiles, [2, 1]);

    let want = gather(&whole);
    let got = gather(&tiles);
    assert_eq!(
        got.keys().collect::<Vec<_>>(),
        want.keys().collect::<Vec<_>>(),
        "{label}: the decomposed run holds a different population"
    );
    let on_second = got.values().filter(|g| g.2 == 1).count();
    assert!(on_second > 0, "{label}: no tracer ended on the offset tile; the gate is vacuous");
    let seeded_half = |owner: u64| usize::from((owner as usize % N) >= N / 2);
    let reseeded: Vec<u64> = seed_mass_weighted(&reference, N_TRACERS).owner.iter().map(|o| o.0).collect();
    let crossed = (0..N_TRACERS as u64).filter(|id| seeded_half(want[id].0) != seeded_half(reseeded[*id as usize])).count();
    assert!(crossed > 0, "{label}: no tracer crossed the cut; migration was never exercised");
    let owners = want.iter().filter(|(id, w)| got[id].0 != w.0).count();
    let positions = want.iter().filter(|(id, w)| got[id].1 != w.1).count();
    assert_eq!(owners, 0, "{label}: {owners} of {} tracers landed on a different cell than on the whole grid", want.len());
    assert_eq!(positions, 0, "{label}: {positions} of {} tracers sit at a different position than on the whole grid", want.len());
}

#[test]
fn offset_tiles_on_a_graded_axis_match_the_whole_grid() {
    assert_offset_tiles_match_the_whole_grid(true);
}

#[test]
fn offset_tiles_on_a_uniform_axis_match_the_whole_grid() {
    assert_offset_tiles_match_the_whole_grid(false);
}
