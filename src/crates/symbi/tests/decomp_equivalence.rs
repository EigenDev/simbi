// =============================================================================
// decomp_equivalence.rs
//
// the correctness contract for multi-gpu domain decomposition (docs/design/36),
// validated IN-PROCESS on the cpu -- no second device, no peer copy, no mpi.
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
// device binding (docs/design/37 M2): each tile is bound to a LOGICAL device, round-robin
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
use symbi::sim::decomp::{exchange_grid, flatten, unflatten, LocalCopy};
#[cfg(feature = "cuda")]
use symbi::sim::decomp::{DeviceCopy, PeerCopy, StagedCopy};
use symbi::sim::evolve::KernelSet;
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_geometry::Cartesian;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::newtonian::Newtonian;
use symbi_hydro::state::Prim;
use symbi_xpu::{with_device, CpuSpace, HostMemory};
#[cfg(feature = "cuda")]
use symbi_xpu::cuda::{CudaSpace, UnifiedMemory};

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

            // drive the reusable module exchange over this harness's `(Sim, Kern)` tiles.
            // the device drain (no-op on cpu) makes the prior async gpu writes visible to the
            // transport; the sims deref to their `FieldStore` for the module's signature.
            fn exchange_all(tiles: &[(Sim, Kern)], counts: [usize; $d]) {
                // every tile may have run its physics in a different context; drain them all
                // so the transport (on device 0, over managed-global memory) sees current
                // interiors. the transport drains its own launches before returning.
                sync_devices();
                let tile_refs: Vec<_> = tiles.iter().map(|(s, _)| &**s).collect();
                // the tile->device map: device identity lives in the decomposition, not on
                // the field. parallel to `tile_refs` (both indexed by flat tile order).
                let devices: Vec<i32> = (0..tiles.len()).map(tile_device).collect();
                exchange_grid(&tile_refs, counts, &devices, &$transport);
            }

            // advance every tile one step at a shared dt, driving the SSP stage table
            // ourselves so the halo exchange can be interleaved BETWEEN stages (rk2 needs
            // the cut ghosts refreshed from each neighbor's stage-updated interior, not just
            // once per step). for adiabatic hydro the per-stage sequence is flux(all dirs)
            // -> godunov_stage -> c2p -> ghost_fill; the mhd / source / body hooks are no-ops.
            fn run(tiles: &mut [(Sim, Kern)], counts: [usize; $d], ts: Timestepping) {
                let stages = ts.stages();
                let multistage = stages.len() > 1;

                // prime prim + ghosts (the stage's entry contract), then seed the cut halos.
                for (flat, (sim, k)) in tiles.iter_mut().enumerate() {
                    with_device(tile_device(flat), || {
                        k.c2p(sim);
                        k.ghost_fill(sim);
                    });
                }
                exchange_all(tiles, counts);

                let mut t = 0.0;
                while t < T_FINAL {
                    // the global dt is the min over all tiles -- identical to the monolithic
                    // cfl since the union of the tile interiors is the monolithic interior.
                    let mut dt = T_FINAL - t;
                    for (flat, (sim, k)) in tiles.iter().enumerate() {
                        dt = dt.min(with_device(tile_device(flat), || k.cfl(sim)));
                    }
                    // snapshot u_n once before the stages for multi-stage schemes (the
                    // corrector reads it via a0 > 0).
                    if multistage {
                        for (flat, (sim, k)) in tiles.iter_mut().enumerate() {
                            with_device(tile_device(flat), || k.snapshot(sim));
                        }
                    }
                    for &(a0, ac) in stages {
                        for (flat, (sim, k)) in tiles.iter_mut().enumerate() {
                            with_device(tile_device(flat), || {
                                for dd in 0..$d {
                                    k.flux(sim, dd);
                                }
                                k.godunov_stage(sim, dt, a0, ac);
                                k.c2p(sim);
                                k.ghost_fill(sim);
                            });
                        }
                        // refresh the cut halos from each neighbor's stage-updated interior,
                        // so the next stage (or step) reconstructs from current neighbor data.
                        exchange_all(tiles, counts);
                    }
                    t += dt;
                }
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
#[cfg(feature = "cuda")]
decomp_harness!(gpu_d1, 1, CudaSpace, UnifiedMemory, DeviceCopy);
// the 2x2 grid exercises StagedCopy: the gather/scatter pack/unpack that peer-copy reuses.
#[cfg(feature = "cuda")]
decomp_harness!(gpu_d2, 2, CudaSpace, UnifiedMemory, StagedCopy);
// the peer-copy transport: tiles round-robin onto NDEV LOGICAL devices, so on a 2+ gpu node
// the 2x2 grid drives real cross-device `cuMemcpyPeer` halos. self-skips on a single card (a
// device cannot peer with itself); the math oracle still applies, now across real devices.
#[cfg(feature = "cuda")]
decomp_harness!(gpu_peer_d2, 2, CudaSpace, UnifiedMemory, PeerCopy);

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
#[cfg(feature = "cuda")]
#[test]
fn gpu_rk2_four_tile_1d() {
    gpu_d1::assert_matches([4], Timestepping::Rk2);
}

#[cfg(feature = "cuda")]
#[test]
fn gpu_rk2_quad_tile_2d_grid() {
    gpu_d2::assert_matches([2, 2], Timestepping::Rk2);
}

// the multi-gpu peer transport. tiles are split across logical devices 0 and 1; on a 2+ gpu
// node the 2x2 exchange moves halos with `cuMemcpyPeer` over nvlink and must still reproduce
// the monolithic run. SELF-SKIPS on a single card -- the peer move needs two physical devices
// (a context cannot peer with itself), so this is the one milestone validated on the cluster,
// not locally. peer access is best-effort enabled (cuMemcpyPeer falls back through host if the
// link is absent, so the correctness assertion holds either way).
#[cfg(feature = "cuda")]
#[test]
fn gpu_peer_rk2_quad_tile_2d_grid() {
    if symbi_xpu::cuda::device_count().unwrap_or(0) < 2 {
        eprintln!("skipping gpu_peer_rk2_quad_tile_2d_grid: needs >= 2 gpus");
        return;
    }
    if symbi_xpu::cuda::can_access_peer(0, 1).unwrap_or(false) {
        symbi_xpu::cuda::enable_peer_access(0, 1).ok();
        symbi_xpu::cuda::enable_peer_access(1, 0).ok();
    }
    gpu_peer_d2::assert_matches([2, 2], Timestepping::Rk2);
}
