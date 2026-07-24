// =============================================================================
// decomp_equivalence.rs
//
// the correctness contract for multi-gpu domain decomposition, validated IN-PROCESS
// on the cpu -- no second device, no peer copy, no mpi.
//
// the one thing that must be right before any transport work: a domain split into a
// grid of tiles, with same-level halo exchange each step, must reproduce the
// monolithic run to round-off. this isolates the decomposition + halo math from all
// hardware and transport concerns. once green, peer-copy (2 gpus) and mpi (multi-node)
// are transport substitutions under a proven-correct decomposition.
//
// a decomposition is a per-axis tile count `counts: [usize; D]`. the monolithic run is
// just `counts = [1; D]` (one tile, no cuts), so the same code path validates both.
// the harness drives the stage loop itself (not `step_once`), so it works for both
// forward euler (one stage) and rk2 (two stages, with a halo exchange BETWEEN stages).
//
// coverage (dimension x integrator x topology):
//   - 1d euler 2-tile, 1d rk2 4-tile (interior tile fed from both neighbors)
//   - 2d euler single-axis, 2d rk2 2x2 grid (the corner case under rk2)
//   - 3d euler 2x2x2 grid (edges + corners in 3d)
//
// cut faces are `BoundaryType::CoarseFine` (ghost_fill skips them, so the exchange owns
// them). the exchange is a TWO-PASS scheme (`exchange_grid`): process axes in order; a
// cut face's transverse extent is INTERIOR for cut axes not yet exchanged, FULL
// otherwise. that carries corner ghosts to the diagonal neighbor without explicit
// diagonal communication. only the PRIM components the flux reconstructs from are
// copied; cons ghosts are never read.
//
// the `decomp_harness!` macro emits a concrete harness per dimension: a generic-over-D
// harness drowns in `Cartesian: Metric<f64,D>` / `Regime` / KernelSet bounds.
//
// device binding: each tile is bound to a LOGICAL device, round-robin
// over `NDEV`. on the one physical card those logical ordinals fold onto distinct cuda
// contexts (the modulo map in cuda.rs), so a tile's allocation + every physics kernel run
// in their own context while the host-orchestrated exchange runs on device 0 over the
// managed-global field memory (visible to every context after a per-context drain). this
// exercises the per-device dispatcher + context-bound modules on a single gpu; the
// decomposed run still has to reproduce the monolithic run (one tile, device 0) to
// round-off. on a host build `with_device` is a no-op, so the wrapping is uniform.
// =============================================================================

use symbi::regimes::substrate_gpu::device_sync;
use symbi::regimes::substrate_newton::AdiabaticSubstrateKernelSet;
use symbi::sim::decomp::{evolve_decomposed, flatten, unflatten, LocalCopy};
#[cfg(feature = "gpu")]
use symbi::sim::decomp::{DeviceCopy, PeerCopy, StagedCopy};
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_geometry::Cartesian;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::newtonian::Newtonian;
use symbi_hydro::state::Prim;
use symbi_xpu::{with_device, CpuSpace, HostMemory};
#[cfg(feature = "gpu")]
use symbi_xpu::{DeviceSpace, DeviceMemory};

const GAMMA: f64 = 1.4;
const CFL: f64 = 0.4;
const N: usize = 64; // cells per axis (kept modest so the 3d run is quick)
const DX: f64 = 1.0 / N as f64;
const T_FINAL: f64 = 0.05; // waves from the central bump stay well inside the domain.

// a smooth, centered pressure+density bump along axis 0 -> sound waves that cross the
// cut, so the halo is genuinely exercised; away from the physical ends so the outflow
// boundaries never activate and mono == decomposed is exact.
fn bump(x: f64) -> f64 {
    0.2 * (-((x - 0.5) / 0.1).powi(2)).exp()
}

macro_rules! decomp_harness {
    ($modname:ident, $d:literal, $space:ty, $mem:ty, $transport:expr) => {
        mod $modname {
            use super::*;

            type Sim = SimState<Newtonian, $d, Cartesian, IdealGas<f64>, $space, $mem>;
            type Kern = AdiabaticSubstrateKernelSet<$mem, f64, $d>;

            // spread decomposed tiles across this many LOGICAL devices (round-robin by tile
            // index). on the single physical card they fold onto NDEV distinct contexts, so
            // the device-binding path is exercised without a second gpu. the monolithic run
            // is one tile -> device 0, so it is unaffected.
            const NDEV: i32 = 2;
            fn tile_device(flat: usize) -> i32 {
                (flat as i32) % NDEV
            }

            // drain every logical context so its async device writes are visible to a
            // consumer running in another context (no-op on the host backend).
            fn sync_devices() {
                for dd in 0..NDEV {
                    with_device(dd, || device_sync::<$mem>());
                }
            }

            // the integrator is set on the sim so the builder allocates the buffers it
            // needs (rk2's u_n snapshot); the harness drives the matching stage table.
            fn make(
                cells: [usize; $d],
                origin: [f64; $d],
                bnd: Boundaries<$d>,
                ts: Timestepping,
            ) -> (Sim, Kern) {
                let sim = Sim::build(Newtonian, IdealGas { gamma: GAMMA }, Cartesian)
                    .cells(cells)
                    .spacing([DX; $d])
                    .origin(origin)
                    .boundaries(bnd)
                    .timestepping(ts)
                    .allocate()
                    .expect("sim construction failed")
                    .set_initial(|x| {
                        let b = bump(x[0]);
                        Prim { rho: 1.0 + b, vel: Tensor::new([0.0; $d]), pre: 1.0 + b }
                    })
                    .build();
                let k = Kern::new(GAMMA, CFL, &sim.geom.allocated);
                (sim, k)
            }

            // build the tile grid for `counts` tiles-per-axis. each tile is an equal slice
            // of the domain; it gets a CoarseFine face wherever it borders a neighbor and
            // physical outflow on the outer domain boundary. tiles are stored row-major.
            fn grid_tiles(counts: [usize; $d], ts: Timestepping) -> Vec<(Sim, Kern)> {
                let m: [usize; $d] = std::array::from_fn(|a| {
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
                        // allocate this tile's fields in its bound device's context, so the
                        // managed memory is hinted resident there (mirrors a real per-gpu tile).
                        with_device(tile_device(flat), || make(m, origin, bnd, ts))
                    })
                    .collect()
            }

            // drive the PRODUCTION decomposed evolve loop (symbi-sim::decomp) over this
            // harness's tiles. exercises the same function the multi-gpu python entry runs --
            // the hand-rolled loop is gone, so a divergence between test and production is
            // impossible. interval = u64::MAX: no mid-run callback, the equivalence check
            // reads the final state via `global_den`.
            fn run(tiles: &mut [(Sim, Kern)], counts: [usize; $d], ts: Timestepping) {
                let devices: Vec<i32> = (0..tiles.len()).map(tile_device).collect();
                // evolve_decomposed takes the tiles by &mut (the per-step body bookkeeping mutates
                // the bodies); build the &mut store handles + & kernels from the same tiles.
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
                    &$transport,
                    |_, _, _| std::ops::ControlFlow::Continue(()),
                );
            }

            // scatter every tile's interior density into one global N^D grid, indexed by
            // global cell coordinate. works for any tile topology (a 1d concat does not).
            fn global_den(tiles: &[(Sim, Kern)], counts: [usize; $d]) -> Vec<f64> {
                // drain every tile's context before reading cons back to host. no-op on cpu.
                sync_devices();
                let m: [usize; $d] = std::array::from_fn(|a| N / counts[a]);
                let mut out = vec![f64::NAN; N.pow($d as u32)];
                for (flat_tile, (sim, _)) in tiles.iter().enumerate() {
                    let tc = unflatten(flat_tile, counts);
                    let ilo: [isize; $d] = std::array::from_fn(|a| sim.geom.interior.spaces[a].lo);
                    for c in sim.geom.interior.iter() {
                        let g: [usize; $d] =
                            std::array::from_fn(|a| tc[a] * m[a] + (c[a] - ilo[a]) as usize);
                        out[flatten(g, [N; $d])] = *sim.fields.cons.den.view().at(c);
                    }
                }
                out
            }

            // the dye rides the mass flux, so a cut only exercises chi exchange if mass actually
            // crosses it AND the dye has a gradient there. the bump is centered on the domain
            // (x = 0.5), so at a cut sitting on it the flux is ~zero by symmetry -- which makes a
            // cut-centered dye gate vacuous. the chi runs therefore add a uniform DIAGONAL drift
            // so every cut is flux-bearing, and paint the dye as a diagonal ramp so every cut
            // carries a gradient; then dropping the exchange on ANY axis breaks the match.
            const CHI_DRIFT: f64 = 0.4; // subsonic (cs ~ 1.18 here), same on every axis

            // the dye concentration at a GLOBAL point: a ramp along the space diagonal, so it has
            // a gradient across every axis's cut. evaluated at the cell center so mono and every
            // tile seed the identical value.
            fn chi_ic(x: [f64; $d]) -> f64 {
                0.25 + 0.1 * x.iter().sum::<f64>()
            }

            // like `make`, plus a uniform diagonal drift (so cuts carry mass flux) and the dye
            // slot allocated + seeded from `chi_ic`. the kernel set is unchanged -- chi advection
            // is a runtime-gated phase in the substrate step (has_passive_scalar), dispatching
            // kernels baked unconditionally.
            fn make_chi(
                cells: [usize; $d],
                origin: [f64; $d],
                bnd: Boundaries<$d>,
                ts: Timestepping,
            ) -> (Sim, Kern) {
                let (mut sim, k) = make(cells, origin, bnd, ts);
                sim = sim.with_passive_scalar().expect("dye allocation failed");
                let ilo: [isize; $d] = std::array::from_fn(|a| sim.geom.interior.spaces[a].lo);
                let cons_chi = sim.fields.cons.chi_field().expect("cons chi");
                let prim_chi = sim.fields.prim.chi_field().expect("prim chi");
                for c in sim.geom.interior.iter() {
                    // the drift is uniform, so cons.mom = rho * CHI_DRIFT and prim.vel = CHI_DRIFT
                    // on every axis; both must be set since the evolve reads cons and the flux
                    // reconstructs prim.
                    let rho = *sim.fields.cons.den.view().at(c);
                    for dd in 0..$d {
                        sim.fields.cons.mom[dd].view_mut().set(c, rho * CHI_DRIFT);
                        sim.fields.prim.vel[dd].view_mut().set(c, CHI_DRIFT);
                    }
                    // global position of this cell center (tile origin + local offset).
                    let x: [f64; $d] =
                        std::array::from_fn(|a| origin[a] + ((c[a] - ilo[a]) as f64 + 0.5) * DX);
                    let chi = chi_ic(x);
                    cons_chi.view_mut().set(c, rho * chi);
                    prim_chi.view_mut().set(c, chi);
                }
                (sim, k)
            }

            fn grid_tiles_chi(counts: [usize; $d], ts: Timestepping) -> Vec<(Sim, Kern)> {
                let m: [usize; $d] = std::array::from_fn(|a| N / counts[a]);
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
                        with_device(tile_device(flat), || make_chi(m, origin, bnd, ts))
                    })
                    .collect()
            }

            // scatter every tile's interior primitive dye into one global N^D grid.
            fn global_chi(tiles: &[(Sim, Kern)], counts: [usize; $d]) -> Vec<f64> {
                sync_devices();
                let m: [usize; $d] = std::array::from_fn(|a| N / counts[a]);
                let mut out = vec![f64::NAN; N.pow($d as u32)];
                for (flat_tile, (sim, _)) in tiles.iter().enumerate() {
                    let tc = unflatten(flat_tile, counts);
                    let ilo: [isize; $d] = std::array::from_fn(|a| sim.geom.interior.spaces[a].lo);
                    let chi = sim.fields.prim.chi_field().expect("prim chi");
                    for c in sim.geom.interior.iter() {
                        let g: [usize; $d] =
                            std::array::from_fn(|a| tc[a] * m[a] + (c[a] - ilo[a]) as usize);
                        out[flatten(g, [N; $d])] = *chi.view().at(c);
                    }
                }
                out
            }

            // run mono (counts = [1; D]) and the requested decomposition with integrator
            // `ts`, then assert the global density grids agree to round-off.
            pub fn assert_matches(counts: [usize; $d], ts: Timestepping) {
                let mut mono = grid_tiles([1; $d], ts);
                run(&mut mono, [1; $d], ts);
                let mono_vals = global_den(&mono, [1; $d]);

                let mut dec = grid_tiles(counts, ts);
                run(&mut dec, counts, ts);
                let dec_vals = global_den(&dec, counts);

                assert!(
                    mono_vals.iter().all(|v| v.is_finite()) && dec_vals.iter().all(|v| v.is_finite()),
                    "some global cells were never written (gather bug)"
                );
                let max_err = mono_vals
                    .iter()
                    .zip(&dec_vals)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0_f64, f64::max);
                assert!(
                    max_err < 1e-12,
                    "decomposition {counts:?} (D={}, {ts:?}) vs monolithic density max err {max_err:e}",
                    $d
                );

                // ALSO exercise `gather_interiors` -- the production checkpoint gather the python
                // multi-gpu path runs. reassemble the decomposed tiles into a
                // full-size global sim and confirm its density equals the direct read. this covers
                // the gather index arithmetic (mirror of the IC scatter) across every topology
                // here -- the one piece the python path uses that `evolve_decomposed` does not.
                let bnd = Boundaries(std::array::from_fn(|_| [BoundaryType::Outflow; 2]));
                let global = make([N; $d], [0.0; $d], bnd, ts);
                let stores: Vec<_> = dec.iter().map(|(s, _)| &**s).collect();
                device_sync::<$mem>();
                symbi::sim::decomp::gather_interiors(&*global.0, &stores, counts);
                let gathered = global_den(std::slice::from_ref(&global), [1; $d]);
                let gather_err = gathered
                    .iter()
                    .zip(&dec_vals)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0_f64, f64::max);
                assert!(
                    gather_err < 1e-12,
                    "gather_interiors {counts:?} (D={}, {ts:?}) vs direct read max err {gather_err:e}",
                    $d
                );
            }

            // the ATLAS gate: with a passive scalar allocated, a decomposition must reproduce
            // the monolithic DYE field to round-off -- which it can only do if prim.chi is
            // exchanged across every cut. the dye is derived into the transport set from the
            // store, so this is what proves the derivation actually carries it.
            pub fn assert_chi_matches(counts: [usize; $d], ts: Timestepping) {
                let mut mono = grid_tiles_chi([1; $d], ts);
                run(&mut mono, [1; $d], ts);
                let mono_chi = global_chi(&mono, [1; $d]);

                let mut dec = grid_tiles_chi(counts, ts);
                run(&mut dec, counts, ts);
                let dec_chi = global_chi(&dec, counts);

                assert!(
                    mono_chi.iter().all(|v| v.is_finite()) && dec_chi.iter().all(|v| v.is_finite()),
                    "some dye cells were never written (gather bug)"
                );

                // NON-VACUITY: the flow must have actually MOVED the dye, or a decomposition that
                // never exchanges chi would match the monolithic run trivially and the gate would
                // test nothing. compare the evolved monolithic dye to its analytic IC and require a
                // real change -- the dye rides the bump-driven mass flux across the cut.
                let mut max_advect = 0.0_f64;
                for (flat, &chi_now) in mono_chi.iter().enumerate() {
                    let g = unflatten(flat, [N; $d]);
                    let x: [f64; $d] = std::array::from_fn(|a| (g[a] as f64 + 0.5) * DX);
                    max_advect = max_advect.max((chi_now - chi_ic(x)).abs());
                }
                assert!(
                    max_advect > 1e-6,
                    "dye never advected (max move {max_advect:e}); the chi phase did not fire, so \
                     the decomposition equivalence is vacuous"
                );

                let max_err = mono_chi
                    .iter()
                    .zip(&dec_chi)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0_f64, f64::max);
                assert!(
                    max_err < 1e-12,
                    "decomposition {counts:?} (D={}, {ts:?}) vs monolithic DYE max err {max_err:e} \
                     (advected {max_advect:e})",
                    $d
                );
            }
        }
    };
}

decomp_harness!(d1, 1, CpuSpace, HostMemory, LocalCopy);
decomp_harness!(d2, 2, CpuSpace, HostMemory, LocalCopy);
decomp_harness!(d3, 3, CpuSpace, HostMemory, LocalCopy);

// the same harness on the gpu memory space: every kernel routes through the production
// run_gpu path (NVRTC -> launch), fields live in unified memory, and the exchange's host
// LocalCopy reads them after a device drain. one device with several subdomains -- no
// speedup, but it proves the decomposition works against device fields before a second
// gpu exists. needs `--features cuda` and a cuda device.
#[cfg(feature = "gpu")]
decomp_harness!(gpu_d1, 1, DeviceSpace, DeviceMemory, DeviceCopy);
// the 2x2 grid exercises StagedCopy: the gather/scatter pack/unpack that peer-copy reuses.
#[cfg(feature = "gpu")]
decomp_harness!(gpu_d2, 2, DeviceSpace, DeviceMemory, StagedCopy);
// the peer-copy transport: tiles round-robin onto NDEV LOGICAL devices, so on a 2+ gpu node
// the 2x2 grid drives real cross-device `cuMemcpyPeer` halos. self-skips on a single card (a
// device cannot peer with itself); the equivalence check still applies, now across real devices.
#[cfg(feature = "gpu")]
decomp_harness!(gpu_peer_d2, 2, DeviceSpace, DeviceMemory, PeerCopy);

#[test]
fn euler_two_tile_1d() {
    d1::assert_matches([2], Timestepping::Euler);
}

// rk2 + an interior tile fed from both neighbors. rk2 exercises the BETWEEN-stage halo
// exchange (the cut ghosts must be refreshed from each neighbor's stage-1 interior).
#[test]
fn rk2_four_tile_1d() {
    d1::assert_matches([4], Timestepping::Rk2);
}

#[test]
fn euler_two_tile_2d_single_axis() {
    d2::assert_matches([2, 1], Timestepping::Euler);
}

// the passive scalar (dye) must survive decomposition: exchanged across the cut like any
// primitive the flux reconstructs from. 1d 4-tile puts an interior tile between two cuts;
// the 2x2 rk2 grid adds the per-stage exchange and diagonal corners.
#[test]
fn rk2_four_tile_1d_dye() {
    d1::assert_chi_matches([4], Timestepping::Rk2);
}

#[test]
fn rk2_quad_tile_2d_grid_dye() {
    d2::assert_chi_matches([2, 2], Timestepping::Rk2);
}

// 3d 2x2x2: the dye must cross faces, edges, and corners of the tile grid.
#[test]
fn euler_octo_tile_3d_grid_dye() {
    d3::assert_chi_matches([2, 2, 2], Timestepping::Euler);
}

// the hard combined case: a 2x2 grid (diagonal-neighbor corners) under rk2 (per-stage
// exchange). this is the topology and integrator a real 2d multi-gpu run uses.
#[test]
fn rk2_quad_tile_2d_grid() {
    d2::assert_matches([2, 2], Timestepping::Rk2);
}

// a 3d 2x2x2 grid: faces, edges, and corners. the two-pass exchange is already
// D-generic, so this confirms it carries over to 3d.
#[test]
fn euler_octo_tile_3d_grid() {
    d3::assert_matches([2, 2, 2], Timestepping::Euler);
}

// the gpu validation: the decomposition on device fields (unified memory), every kernel
// through run_gpu. gpu-mono vs gpu-decomposed must still agree to round-off -- the only
// difference is the exchange, run identically on both. proves the device-memory path
// before any multi-device hardware.
#[cfg(feature = "gpu")]
#[test]
fn gpu_rk2_four_tile_1d() {
    gpu_d1::assert_matches([4], Timestepping::Rk2);
}

#[cfg(feature = "gpu")]
#[test]
fn gpu_rk2_quad_tile_2d_grid() {
    gpu_d2::assert_matches([2, 2], Timestepping::Rk2);
}

// the multi-gpu peer transport, now the UNIVERSAL transport. `PeerCopy` is adaptive: on this
// single card the two logical devices fold onto the same physical gpu, can't peer, and it stages
// over managed memory; on a 2+ gpu node the SAME code moves halos with `cuMemcpyPeer` over
// nvlink. so this runs EVERYWHERE -- no self-skip -- and proves the one transport that ships to
// the cluster is correct here too. `enable_peer_mesh` is a no-op when no pair can peer.
#[cfg(feature = "gpu")]
#[test]
fn gpu_peer_rk2_quad_tile_2d_grid() {
    symbi::sim::decomp::enable_peer_mesh(&[0, 1]);
    gpu_peer_d2::assert_matches([2, 2], Timestepping::Rk2);
}
