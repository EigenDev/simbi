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
// the same proof runs against every message transport. the host queue packs into
// a vector; the device transports gather the strip into a contiguous device
// buffer and, when the two halves sit on different devices, move it over the
// fabric before scattering. all three must write identical ghosts, so the device
// specialization is held to the host implementation's answer.
//
// run: cargo test -p symbi --test decomp_message_seam
//      cargo test -p symbi --test decomp_message_seam --features hip   (needs a device)
// =============================================================================

use symbi::regimes::substrate_gpu::device_sync;
use symbi::regimes::substrate_newton::AdiabaticSubstrateKernelSet;
#[cfg(feature = "gpu")]
use symbi::sim::decomp::{PeerCopy, StagedCopy};
use symbi::sim::decomp::{
    LocalCopy, MessageQueue, MessageTransport, Ownership, Partition, Phase, Schedule, Topology,
    exchange_grid, exchange_grid_phase, unflatten,
};
use symbi::sim::state::*;
use symbi::sim::substrate_seam::KernelSet;
use symbi_algebra::Tensor;
use symbi_geometry::Cartesian;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::newtonian::Newtonian;
use symbi_hydro::state::Prim;
#[cfg(feature = "gpu")]
use symbi_xpu::{DeviceMemory, DeviceSpace};
use symbi_xpu::{CpuSpace, HostMemory, with_device};

const GAMMA: f64 = 1.4;
const CFL: f64 = 0.4;
const N: usize = 64;
const DX: f64 = 1.0 / N as f64;

// a field with structure along both axes, so a ghost strip filled from the wrong
// neighbor, in the wrong order, or under the wrong tag carries visibly wrong cells.
fn seed(x: f64, y: f64) -> f64 {
    1.0 + 0.3 * (7.0 * x).sin() * (5.0 * y).cos() + 0.1 * x * y
}

// one harness per (memory space, halo transport, message transport). `$ndev` logical devices
// take the tiles round-robin: one device keeps every transfer local, while two put the halves
// of a cross-group transfer on different devices, which is the move a device message makes.
macro_rules! message_seam_harness {
    ($modname:ident, $space:ty, $mem:ty, $halo:expr, $messages:expr, $ndev:literal) => {
        mod $modname {
            use super::*;

            type Sim = SimState<Newtonian, 2, Cartesian, IdealGas<f64>, $space, $mem>;
            type Kern = AdiabaticSubstrateKernelSet<$mem, f64, 2>;

            fn tile_device(flat: usize) -> i32 {
                (flat as i32) % $ndev
            }

            // drain every logical context so device writes are visible to the next reader.
            fn sync_devices() {
                for dd in 0..$ndev {
                    with_device(dd, || device_sync::<$mem>());
                }
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
                        with_device(tile_device(flat), || {
                            make(
                                [ext[0].1, ext[1].1],
                                [ext[0].0 as f64 * DX, ext[1].0 as f64 * DX],
                                bnd,
                            )
                        })
                    })
                    .collect()
            }

            // prime prim from cons and fill the physical ghosts: the state an exchange
            // starts from.
            fn prime(tiles: &[(Sim, Kern)]) {
                for (flat, (s, k)) in tiles.iter().enumerate() {
                    with_device(tile_device(flat), || {
                        k.c2p(s);
                        k.ghost_fill(s);
                    });
                }
                sync_devices();
            }

            // every cell of every tile, ghosts included -- the exchange writes ghosts, so an
            // interior-only comparison would miss exactly what is under test.
            fn all_cells(tiles: &[(Sim, Kern)]) -> Vec<f64> {
                sync_devices();
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

            fn stores(tiles: &[(Sim, Kern)]) -> Vec<&FieldStore<2, 2, $mem, f64>> {
                tiles.iter().map(|(s, _)| &**s).collect()
            }

            /// the same schedule, exchanged two ways: whole ownership with direct copies, and
            /// split across owner groups where every cross-group transfer becomes a message.
            pub fn assert_matches(cuts: [Vec<usize>; 2], owner: &[usize]) {
                let partition =
                    Partition::explicit([N, N], cuts).expect("interior cuts lie inside the grid");
                let counts = partition.counts();
                let devices: Vec<i32> = (0..partition.n_tiles()).map(tile_device).collect();

                let fused = partition_tiles(&partition);
                prime(&fused);
                let schedule = Schedule::derive(counts, fused[0].0.geom.ng, &Topology::open());
                let before = all_cells(&fused);
                exchange_grid(&stores(&fused), &schedule, &devices, &$halo);
                let want = all_cells(&fused);

                // the exchange has to have written something, or both arms agree on an
                // untouched field and the comparison proves nothing.
                let touched = want.iter().zip(&before).filter(|(a, b)| a != b).count();
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
                let messages = $messages;
                assert_eq!(
                    messages.outstanding(),
                    0,
                    "the transport starts with messages already posted"
                );
                // one axis at a time, every process posting what it sources before any takes
                // what it receives. the next axis's legs read the ghosts this one fills, so an
                // axis has to close before the next opens.
                for axis in 0..2 {
                    for phase in [Phase::Post, Phase::Complete] {
                        for &me in &ranks {
                            exchange_grid_phase(
                                &sh,
                                &schedule,
                                &devices,
                                Ownership::Ranked { owner, me },
                                &$halo,
                                &messages,
                                axis,
                                phase,
                            );
                        }
                    }
                    assert_eq!(
                        messages.outstanding(),
                        0,
                        "axis {axis} left messages no receive took"
                    );
                }
                let got = all_cells(&split);

                for (i, (a, b)) in got.iter().zip(want.iter()).enumerate() {
                    assert!(
                        a.to_bits() == b.to_bits(),
                        "cell {i}: message exchange {a:e} != direct-copy exchange {b:e}"
                    );
                }
            }
        }
    };
}

message_seam_harness!(host, CpuSpace, HostMemory, LocalCopy, MessageQueue::new(), 1);

// the same host seam with the tiles spread over two logical contexts, so the device-binding
// path around every pack, launch and drain is walked on a backend where the answer is known.
message_seam_harness!(host_two_contexts, CpuSpace, HostMemory, LocalCopy, MessageQueue::new(), 2);

// the staged device transport as a message transport: the send gathers the strip into a
// contiguous device buffer and the receive scatters it, both halves on one device.
#[cfg(feature = "gpu")]
message_seam_harness!(staged, DeviceSpace, DeviceMemory, StagedCopy, StagedCopy, 1);

// the peer transport with the tiles spread over two logical devices, so a cross-group
// transfer's send and receive sit on different devices and the strip crosses the fabric.
// folds onto one physical card when the node has a single gpu, where the pair cannot peer
// and the move stages instead -- the ghosts must match either way.
#[cfg(feature = "gpu")]
message_seam_harness!(peer, DeviceSpace, DeviceMemory, PeerCopy, PeerCopy, 2);

/// a 2x2 grid split down the middle: the two axis-0 legs cross the process boundary and
/// travel as messages, while the axis-1 legs stay inside a process and are copied.
#[test]
fn a_cut_between_processes_moves_as_messages() {
    host::assert_matches([vec![32], vec![32]], &[0, 0, 1, 1]);
}

/// every tile in its own process: no transfer is a local copy, so the whole schedule is
/// carried by the message seam.
#[test]
fn every_transfer_can_travel_as_a_message() {
    host::assert_matches([vec![32], vec![32]], &[0, 1, 2, 3]);
}

/// unequal tiles on both axes, so a message's two regions have differing extents and a
/// packing that assumed a uniform strip length would mismatch.
#[test]
fn ragged_tiles_cross_processes_on_both_axes() {
    host::assert_matches([vec![19, 37], vec![27]], &[0, 1, 0, 1, 1, 0]);
}

/// the seam with tiles bound to two logical contexts: the binding, drain and message order
/// are the ones the device arms take, on a backend whose answer is already established.
#[test]
fn two_contexts_carry_the_same_messages() {
    host_two_contexts::assert_matches([vec![32], vec![32]], &[0, 1, 2, 3]);
    host_two_contexts::assert_matches([vec![19, 37], vec![27]], &[0, 1, 0, 1, 1, 0]);
}

/// the device gather/scatter halves, held to the host queue's answer.
#[cfg(feature = "gpu")]
#[test]
fn staged_device_strips_match_the_direct_copy() {
    staged::assert_matches([vec![32], vec![32]], &[0, 1, 2, 3]);
    staged::assert_matches([vec![19, 37], vec![27]], &[0, 1, 0, 1, 1, 0]);
}

/// the same with the two halves of a transfer on different devices.
#[cfg(feature = "gpu")]
#[test]
fn peer_device_strips_match_the_direct_copy() {
    peer::assert_matches([vec![32], vec![32]], &[0, 1, 2, 3]);
    peer::assert_matches([vec![19, 37], vec![27]], &[0, 1, 0, 1, 1, 0]);
}
