// =============================================================================
// decomp_message_seam.rs
//
// the one-sided halo seam: a transfer whose two tiles sit in different processes
// travels as a packed message, sent by the process holding the source and taken
// by the process holding the destination. this is the shape a wire transport
// needs, since a direct region copy requires both fields at once.
//
// the proof runs both halves in one process: the tiles are split into two owner
// groups, each group's post phase runs, then each group's complete phase, and
// the resulting ghosts must match cell for cell what the whole-ownership
// exchange writes with direct copies. every cut is crossed by messages in one
// arm and by copies in the other, so a packing, ordering or tag error shows up
// as a mismatch.
//
// run: cargo test -p symbi --test decomp_message_seam
// =============================================================================

use symbi::regimes::substrate_newton::AdiabaticSubstrateKernelSet;
use symbi::sim::decomp::{
    LocalCopy, MessageQueue, Ownership, Partition, Phase, Schedule, Topology, exchange_grid,
    exchange_grid_phase, unflatten,
};
use symbi::sim::state::*;
use symbi::sim::substrate_seam::KernelSet;
use symbi_algebra::Tensor;
use symbi_geometry::Cartesian;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::newtonian::Newtonian;
use symbi_hydro::state::Prim;
use symbi_xpu::{CpuSpace, HostMemory};

const GAMMA: f64 = 1.4;
const CFL: f64 = 0.4;
const N: usize = 64;
const DX: f64 = 1.0 / N as f64;

type Sim = SimState<Newtonian, 2, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;
type Kern = AdiabaticSubstrateKernelSet<HostMemory, f64, 2>;

// a field with structure along both axes, so a ghost strip filled from the wrong
// neighbor, in the wrong order, or under the wrong tag carries visibly wrong cells.
fn seed(x: f64, y: f64) -> f64 {
    1.0 + 0.3 * (7.0 * x).sin() * (5.0 * y).cos() + 0.1 * x * y
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
        .set_initial(|[x, y]| Prim {
            rho: seed(x, y),
            vel: Tensor::new([0.3 * seed(x, y), -0.2 * seed(y, x)]),
            pre: 0.5 + 0.25 * seed(y, x),
        })
        .build();
    let k = Kern::new(GAMMA, CFL, &sim.geom.allocated);
    (sim, k)
}

fn partition_tiles(partition: &Partition<2>) -> Vec<(Sim, Kern)> {
    let counts = partition.counts();
    (0..partition.n_tiles())
        .map(|flat| {
            let tc = unflatten(flat, counts);
            let ext = partition.tile_extents(tc);
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
            make(
                [ext[0].1, ext[1].1],
                [ext[0].0 as f64 * DX, ext[1].0 as f64 * DX],
                bnd,
            )
        })
        .collect()
}

// prime prim from cons and fill the physical ghosts: the state an exchange starts from.
fn prime(tiles: &[(Sim, Kern)]) {
    for (s, k) in tiles {
        k.c2p(s);
        k.ghost_fill(s);
    }
}

// every cell of every tile, ghosts included -- the exchange writes ghosts, so an
// interior-only comparison would miss exactly what is under test.
fn all_cells(tiles: &[(Sim, Kern)]) -> Vec<f64> {
    let mut out = Vec::new();
    for (sim, _) in tiles {
        let pre = sim
            .fields
            .prim
            .pre
            .as_ref()
            .expect("an adiabatic run carries a pressure field");
        for f in [
            &sim.fields.prim.rho,
            pre,
            &sim.fields.prim.vel[0],
            &sim.fields.prim.vel[1],
        ] {
            let view = f.view();
            for c in sim.geom.allocated.iter() {
                out.push(*view.at(c));
            }
        }
    }
    out
}

fn stores(tiles: &[(Sim, Kern)]) -> Vec<&FieldStore<2, 2, HostMemory, f64>> {
    tiles.iter().map(|(s, _)| &**s).collect()
}

/// the same schedule, exchanged two ways: whole ownership with direct copies, and split
/// across two owner groups where every cross-group transfer becomes a packed message.
fn assert_message_seam_matches(cuts: [Vec<usize>; 2], owner: &[usize]) {
    let partition = Partition::explicit([N, N], cuts).expect("interior cuts lie inside the grid");
    let counts = partition.counts();
    let devices = vec![0i32; partition.n_tiles()];

    let fused = partition_tiles(&partition);
    prime(&fused);
    let schedule = Schedule::derive(counts, fused[0].0.geom.ng, &Topology::open());
    let before = all_cells(&fused);
    exchange_grid(&stores(&fused), &schedule, &devices, &LocalCopy);
    let want = all_cells(&fused);

    // the exchange has to have written something, or both arms agree on an untouched
    // field and the comparison proves nothing.
    let touched = want
        .iter()
        .zip(&before)
        .filter(|(a, b)| a != b)
        .count();
    assert!(
        touched > 0,
        "the exchange wrote no cells; the seam is not under test"
    );

    let split = partition_tiles(&partition);
    prime(&split);
    let sh = stores(&split);
    let ranks: Vec<usize> = {
        let mut r: Vec<usize> = owner.to_vec();
        r.sort_unstable();
        r.dedup();
        r
    };
    assert!(
        ranks.len() > 1,
        "the owner map puts every tile in one process; no transfer becomes a message"
    );
    let queue = MessageQueue::new();
    // one axis at a time, every process posting what it sources before any takes what it
    // receives. the next axis's legs read the ghosts this one fills, so an axis has to
    // close before the next opens.
    for axis in 0..2 {
        for phase in [Phase::Post, Phase::Complete] {
            for &me in &ranks {
                exchange_grid_phase(
                    &sh,
                    &schedule,
                    &devices,
                    Ownership::Ranked { owner, me },
                    &LocalCopy,
                    &queue,
                    axis,
                    phase,
                );
            }
        }
        assert_eq!(
            queue.outstanding(),
            0,
            "axis {axis} left messages no receive took"
        );
    }
    assert_eq!(
        queue.outstanding(),
        0,
        "messages were posted that no receive took"
    );
    let got = all_cells(&split);

    for (i, (a, b)) in got.iter().zip(want.iter()).enumerate() {
        assert!(
            a.to_bits() == b.to_bits(),
            "cell {i}: message exchange {a:e} != direct-copy exchange {b:e}"
        );
    }
}

/// a 2x2 grid split down the middle: the two axis-0 legs cross the process boundary and
/// travel as messages, while the axis-1 legs stay inside a process and are copied.
#[test]
fn a_cut_between_processes_moves_as_messages() {
    assert_message_seam_matches([vec![32], vec![32]], &[0, 0, 1, 1]);
}

/// one process per tile: every transfer in the schedule is a message, so nothing is
/// carried by the direct-copy path.
#[test]
fn every_transfer_can_travel_as_a_message() {
    assert_message_seam_matches([vec![32], vec![32]], &[0, 1, 2, 3]);
}

/// unequal tiles, and an owner map that splits the grid diagonally so a tile's neighbors
/// sit in different processes on the two axes.
#[test]
fn ragged_tiles_cross_processes_on_both_axes() {
    assert_message_seam_matches([vec![19, 37], vec![27]], &[0, 1, 1, 0, 0, 1]);
}
