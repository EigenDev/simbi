// =============================================================================
// decomp_periodic_equivalence.rs
//
// a periodic axis carrying more than one tile: the first and last tiles are
// neighbors across the domain seam, so the halo schedule carries an end-to-end
// leg there. a tile's own periodic ghost fill reads that tile's opposite
// interior, which is the whole domain only when the axis is uncut -- across a
// cut it hands the end tiles their own far side and the seam carries the wrong
// state.
//
// a smooth density bump sits ON the seam and advects through it at uniform
// velocity, so the seam is the one place the answer is decided. the decomposed
// run must reproduce the single-tile periodic run.
//
// run: cargo test -p symbi --test decomp_periodic_equivalence
// =============================================================================

use symbi::regimes::substrate_newton::AdiabaticSubstrateKernelSet;
use symbi::sim::decomp::{
    LocalCopy, Partition, Schedule, Topology, evolve_scheduled, flatten, unflatten,
};
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_geometry::Cartesian;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::newtonian::Newtonian;
use symbi_hydro::quantity::{Density, Pressure};
use symbi_hydro::state::Prim;
use symbi_xpu::{CpuSpace, HostMemory};

const GAMMA: f64 = 1.4;
const CFL: f64 = 0.4;
const N: usize = 64;
const DX: f64 = 1.0 / N as f64;
// long enough for the bump to cross the seam several times at |v| = 1.
const T_FINAL: f64 = 0.35;
const VX: f64 = 1.0;
const VY: f64 = 0.5;

type Sim = SimState<Newtonian, 2, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;
type Kern = AdiabaticSubstrateKernelSet<HostMemory, f64, 2>;

// the shortest distance to the bump center along a periodic axis of unit length.
fn wrapped(x: f64, center: f64) -> f64 {
    let d = (x - center).abs();
    d.min(1.0 - d)
}

// a smooth bump straddling the domain seam at x = 0: half of it starts in the first
// tile and half in the last, and it advects straight through the seam.
fn bump(x: f64, y: f64) -> f64 {
    let r2 = (wrapped(x, 0.0) / 0.12).powi(2) + (wrapped(y, 0.0) / 0.12).powi(2);
    0.5 * (-r2).exp()
}

fn make(cells: [usize; 2], origin: [f64; 2], bnd: Boundaries<2>) -> (Sim, Kern) {
    let sim = Sim::build(Newtonian, IdealGas { gamma: GAMMA }, Cartesian)
        .cells(cells)
        .spacing([DX; 2])
        .origin(origin)
        .boundaries(bnd)
        .timestepping(Timestepping::Rk2)
        .allocate()
        .expect("sim construction failed")
        .set_initial(|[x, y]| {
            Prim::adiabatic(
                Density(1.0 + bump(x, y)),
                Tensor::new([VX, VY]),
                Pressure(1.0),
            )
        })
        .build();
    let k = Kern::new(GAMMA, CFL, &sim.geom.allocated);
    (sim, k)
}

// on a cut wrapping axis every face is a cut, the two at the domain seam included: those
// are carried by the schedule's wrap legs. an uncut axis keeps its periodic faces, which
// wrap the single tile onto itself over the whole domain.
fn partition_tiles(partition: &Partition<2>) -> Vec<(Sim, Kern)> {
    let counts = partition.counts();
    (0..partition.n_tiles())
        .map(|flat| {
            let tc = unflatten(flat, counts);
            let ext = partition.tile_extents(tc);
            let bnd = Boundaries(std::array::from_fn(|a| {
                let seam = if counts[a] > 1 {
                    BoundaryType::CoarseFine
                } else {
                    BoundaryType::Periodic
                };
                let lo = if tc[a] == 0 {
                    seam
                } else {
                    BoundaryType::CoarseFine
                };
                let hi = if tc[a] == counts[a] - 1 {
                    seam
                } else {
                    BoundaryType::CoarseFine
                };
                [lo, hi]
            }));
            make(
                [ext[0].1, ext[1].1],
                [ext[0].0 as f64 * DX, ext[1].0 as f64 * DX],
                bnd,
            )
        })
        .collect()
}

fn run(tiles: &mut [(Sim, Kern)], counts: [usize; 2]) {
    let devices = vec![0i32; tiles.len()];
    let mut stores = Vec::new();
    let mut kernels = Vec::new();
    for (s, k) in tiles.iter_mut() {
        stores.push(&mut **s);
        kernels.push(&*k);
    }
    let schedule = Schedule::derive(counts, stores[0].geom.ng, &Topology::wrapping([true; 2]));
    evolve_scheduled(
        &mut stores,
        &kernels,
        &schedule,
        &devices,
        Timestepping::Rk2,
        0.0,
        T_FINAL,
        u64::MAX,
        &LocalCopy,
        |_, _, _| std::ops::ControlFlow::Continue(()),
    );
}

fn global_den(tiles: &[(Sim, Kern)], partition: &Partition<2>) -> Vec<f64> {
    let counts = partition.counts();
    let mut out = vec![f64::NAN; N * N];
    for (flat_tile, (sim, _)) in tiles.iter().enumerate() {
        let ext = partition.tile_extents(unflatten(flat_tile, counts));
        let ilo: [isize; 2] = std::array::from_fn(|a| sim.geom.interior.spaces[a].lo);
        for c in sim.geom.interior.iter() {
            let g: [usize; 2] = std::array::from_fn(|a| ext[a].0 + (c[a] - ilo[a]) as usize);
            out[flatten(g, [N; 2])] = *sim.fields.cons.den.view().at(c);
        }
    }
    out
}

fn assert_matches(cuts: [Vec<usize>; 2]) {
    let split = Partition::explicit([N, N], cuts).expect("interior cuts lie inside the grid");
    let whole = Partition::explicit([N, N], [Vec::new(), Vec::new()])
        .expect("the uncut partition is one tile");

    let mut one = partition_tiles(&whole);
    let ic = global_den(&one, &whole);
    run(&mut one, whole.counts());
    let mono = global_den(&one, &whole);

    let mut dec = partition_tiles(&split);
    run(&mut dec, split.counts());
    let decomposed = global_den(&dec, &split);

    // the bump has to have moved through the seam, or a broken wrap would match a
    // frozen field and the gate would prove nothing.
    let moved = mono
        .iter()
        .zip(&ic)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        moved > 1e-2,
        "the bump never advected (max {moved:e}); the evolution is vacuous"
    );

    let mut worst = 0.0f64;
    for (i, (a, b)) in decomposed.iter().zip(mono.iter()).enumerate() {
        assert!(
            a.is_finite(),
            "uncovered global cell {i}: the scatter missed it"
        );
        worst = worst.max((a - b).abs());
    }
    assert!(
        worst < 1e-12,
        "decomposed {:?} vs single-tile periodic density max err {worst:e} (bump moved {moved:e})",
        split.counts()
    );
}

/// an even cut on each periodic axis: the four tiles meet at two interior seams and two
/// wrap seams, and the bump crosses the wrap seams.
#[test]
fn a_periodic_axis_wraps_across_its_cut() {
    assert_matches([vec![32], vec![32]]);
}

/// the same with unequal cuts, so the wrap pairs tiles of different sizes.
#[test]
fn a_periodic_axis_wraps_across_ragged_cuts() {
    assert_matches([vec![19, 37], vec![27]]);
}
