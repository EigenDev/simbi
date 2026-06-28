# handoff: multi-gpu decomposition and scalability

audience: a fresh claude (or human) picking up the scalability work with no prior
context. read this top to bottom once, then start at "your first task".

orientation note: line numbers in this doc drift as the code changes. always
re-locate a symbol by grepping for its name, not by jumping to a line number.

## tl;dr

the codebase (a rust cargo workspace under `src/`, driven from python) is getting
multi-gpu, and eventually multi-node, support. the strategy is already decided and
written down in `docs/design/36_scalability_multi_gpu.md` (the adr). read that adr
first, it is short and it is the source of truth for the WHY.

the work is staged so the hardest question (is the decomposition math correct?) is
answered before any hardware or transport complexity. THAT QUESTION IS NOW FULLY
ANSWERED YES: an in-process, cpu-only equivalence test proves a domain split into a
grid of tiles, with same-level halo exchange each step, reproduces the monolithic run
to round-off across dimension x integrator x topology -- 1d/2d/3d, forward-euler AND
rk2 (with a halo exchange between stages), chains and grids including the 2x2 corner
case. the decomposition + halo math is DONE. what remains is purely transport: lift
the proven in-process exchange behind a seam whose impls are local copy (today) ->
gpu peer copy (2 devices) -> mpi (multi-node).

## strategic context (the decided direction)

from the adr, do not relitigate these:

- model: SPMD, one rank (process) per gpu. each rank owns one device, reuses the
  existing single-device code unchanged, holds one subdomain, and exchanges halos
  with neighbor ranks over a transport (local copy now, peer copy for 2 gpus, mpi
  for multi-node). this one model spans 1 gpu -> N gpus -> N nodes.
- we did NOT choose the "single process drives many devices with an explicit
  device_id threaded everywhere" model. it is invasive and does not reach
  multi-node. do not start threading device_id through the xpu layer.
- the asset: the AMR system is structurally a domain-decomposition engine already.
  each `LevelData` owns a self-contained `SimStateGeneric` (fields + geometry +
  boundaries + ghosts); coarse-fine halos are explicitly typed and exchanged with
  time interpolation (`prolong_cf`); conservation is handled by flux registers.
  multi-gpu reuses this machinery, it does not reinvent it.
- the one genuinely-missing primitive: SAME-LEVEL neighbor halo exchange. AMR only
  does coarse<->fine. splitting a uniform grid across ranks creates same-resolution
  neighbor boundaries. building that exchange is the first real piece of work, and
  it doubles as multi-patch-amr infrastructure.

## what exists right now (the starting state)

all on the `experimental` branch (the user treats it as a sandbox, so no feature
branch is needed).

1. `pub fn step_once` in `src/crates/symbi/src/sim/evolve.rs`.
   advances a sim by ONE step at a caller-supplied dt. `evolve` hides the per-step
   loop; the decomposition driver needs per-step control so a shared dt and a halo
   exchange can be interleaved between steps. it wraps the private `step`. prim +
   cons must be current at entry (prime with `c2p` + `ghost_fill` once before the
   first call).

2. `src/crates/symbi/tests/decomp_equivalence.rs`. the proven correctness contract,
   validated IN PROCESS on the cpu (no second gpu, no peer copy, no mpi). the
   `decomp_harness!` macro emits a concrete harness per dimension (a generic-over-D
   harness drowns in `Cartesian: Metric<f64,D>` / `Regime` / KernelSet bounds). FIVE
   tests PASS: 1d euler 2-tile, 1d rk2 4-tile, 2d euler single-axis, 2d rk2 2x2 grid,
   3d euler 2x2x2. each runs a monolithic sim and a decomposed sim and asserts the
   global density grids agree to < 1e-12.

key design choices in the test (understand them before extending):

- a decomposition is a per-axis tile count `counts: [usize; D]`. the monolithic run
  is `counts = [1; D]` (one tile, no cuts), so the same code path validates both.
- cut faces between tiles are `BoundaryType::CoarseFine`; `ghost_fill` treats that as
  `BcType::Skip` (see `src/crates/symbi-substrate/src/kernels/support.rs`, grep
  `CoarseFine`), so it leaves those ghosts untouched and the exchange owns them. a
  dedicated `BoundaryType::Neighbor` variant is a later cleanup (the ~49 match sites),
  not a prerequisite.
- the exchange is a TWO-PASS scheme (`exchange_grid`): process axes in order; a cut
  face's transverse extent is INTERIOR for cut axes not yet exchanged, FULL otherwise.
  this carries corner ghosts to the diagonal neighbor without explicit diagonal
  communication (the reason the 2x2 case passes). it copies only the PRIM components
  (rho, vel[..], pre) the flux stage reconstructs from; cons exchange is not needed
  for forward euler.
- the harness drives the SSP stage table itself (not `step_once`), so it serves both
  euler (one stage) and rk2 (two stages). for rk2 the cut halos are exchanged BETWEEN
  stages, not just per step (the corrector must reconstruct from each neighbor's
  stage-1-updated interior). the integrator is set on the sim in `make` so the builder
  allocates rk2's u_n snapshot buffer; the loop drives the matching `ts.stages()`.
- all runs share one `run` loop with a global dt = min over tiles' cfl, so the halo
  exchange is the only variable between mono and decomposed.

## where to go next (pick up here)

DONE: rk2 (per-stage exchange) and 3d are proven. the decomposition math is complete.
DONE: the transport seam is extracted -- `src/crates/symbi-sim/src/decomp.rs` holds the
`HaloTransport` trait, the `LocalCopy` impl, and the D-generic `exchange_faces` (built on
`Domain::boundary`/`slab`/`iter`). the equivalence test routes its per-face exchange
through `decomp::exchange_faces` with `LocalCopy` and stays green -- it is now the
permanent regression oracle for any future transport.

DONE: single-gpu validation. the harness macro is parameterized over (exec space, memory
space); `gpu_d1`/`gpu_d2` instances run on `CudaSpace`/`UnifiedMemory` through the
production run_gpu path (NVRTC -> launch), behind `#[cfg(feature = "cuda")]`. tests
`gpu_rk2_four_tile_1d` and `gpu_rk2_quad_tile_2d_grid` PASS on a single device
(`cargo test -p symbi --features cuda --test decomp_equivalence --release`). a
`device_sync::<Mem>()` (no-op on cpu) before every host read drains the async device
queue so the host `LocalCopy` reads coherent unified memory. this proves the
decomposition works against device fields; the host-roundtrip `LocalCopy` is correct but
slow (page migration) and is NOT the multi-device path.

DONE: the device-side transport. `decomp::DeviceCopy` (cuda-gated) JITs a D-independent
gather/scatter kernel (`dst[didx[i]] = src[sidx[i]]`) and runs the strip copy on the gpu
-- the field data never round-trips to host (only the small precomputed flat-index arrays
do). the gpu instances route through `DeviceCopy`; `gpu_rk2_four_tile_1d` and
`gpu_rk2_quad_tile_2d_grid` pass on the single device. this gather/scatter IS the
pack/unpack a peer or mpi transport reuses -- only the move in the middle changes.
(optimization left: `DeviceCopy` allocates the index buffers per call; pool/cache them,
since the strip geometry is fixed.)

DONE: the grid orchestration is extracted. `decomp::exchange_grid(tiles: &[&FieldStore],
counts, transport)` + `decomp::flatten`/`unflatten` now live in the module; the test is a
thin oracle harness that builds sims, drives the SSP stages, and calls `exchange_grid`
through a one-line `exchange_all` wrapper (which adds the `device_sync` drain). the module
is now the complete reusable decomposition layer; a cluster driver reuses it directly.

direction (chosen 2026-06-28): intra-node multi-gpu via nvlink peer-copy FIRST; mpi only
when going multi-node. so no mpi dependency yet. the pack/unpack half (`StagedCopy`) is
DONE and locally validated -- it gathers each strip into a contiguous buffer and scatters
it back, exactly what a peer/mpi move transfers between ranks.

remaining (NEEDS 2+ gpus / a node; the structure is locally developable, the cross-gpu
behavior is not):

1. peer transport: a `HaloTransport` impl = `StagedCopy`'s gather, then `cuMemcpyPeer`
   (nvlink) of the contiguous buffer to the neighbor device's buffer, then `StagedCopy`'s
   scatter. only the middle move is new; the gather/scatter are proven. (later, multi-node:
   swap the move for an mpi send/recv.)
2. device binding (scoped in docs/design/37; one process drives many gpus intra-node).
   M1 DONE: the xpu globals are now per-device -- `symbi-xpu/src/cuda.rs` has a per-device
   context registry (`CUDA_CTX: [OnceLock; MAX_GPUS]`), `current_device()`, `with_device(ord,
   f)` (ambient current-device model, no device_id threaded through signatures), and
   `runtime.rs` has a per-device dispatcher registry + `current_dispatcher()` (cuda modules
   are context-bound, so this is mandatory). all defaults to device 0 -> behaviorally a
   no-op, whole suite green.
   M2 DONE: tiles are bound to LOGICAL devices and validated on the single card. `cuda.rs
   ensure_init_device` round-robins logical ordinals onto physical devices (`ord % count`),
   so logical device 1+ become distinct contexts on the one gpu; `symbi-xpu/src/lib.rs`
   exposes a uniform `with_device` (cuda -> `cuda::with_device`, host -> `f()`). there is NO
   `Field.device_id` -- the ambient current-device model + managed-global memory make it
   unnecessary (yagni). two validations PASS: `symbi-xpu/tests/multi_device.rs` runs a kernel
   in two contexts on the one card through the per-device dispatcher (proves the context-bound
   -module landmine is handled), and `decomp_equivalence.rs` now binds each tile's allocation
   + every physics kernel to its logical device (round-robin over `NDEV=2`), drains every
   context at the exchange/read seams (`sync_devices`), runs the host-orchestrated exchange on
   device 0 over managed-global memory, and still reproduces the monolithic run (one tile,
   device 0) to < 1e-12 on the gpu harnesses. run: `cargo test -p symbi-xpu --features cuda
   --test multi_device --release` and `cargo test -p symbi --features cuda --test
   decomp_equivalence --release`.
   remaining: M3 peer-access bindings + the peer `HaloTransport` (`StagedCopy` gather +
   `cuMemcpyPeer` + scatter); M4 (needs the node) real multi-gpu + distributed cfl.
3. distributed cfl via a cross-rank reduce; distributed checkpoint i/o. under SPMD bind each
   rank with `CUDA_VISIBLE_DEVICES` so the single-device code stays valid per rank.

the in-process + single-gpu tests stay the oracle; a new transport must keep them green.

the iron rule: never advance a transport without its equivalence test green first. the
test is the contract.

verify the current suite with (from `src/`):

```
cargo test -p symbi --test decomp_equivalence --release
```

## hard-won gotchas (these will waste your time if you do not know them)

- TESTS MUST RUN IN RELEASE. `cargo test` in debug SIGSEGVs rustc itself (an llvm
  dwarf stack overflow on the heavily-generic `symbi-substrate` crate). always pass
  `--release`. `cargo check` (metadata only, no codegen) is fine in debug and is
  fast.
- the pre-push hook is now a fast build check (`cargo check --workspace
  --all-targets`), not a test run. it lives in `.git/hooks/pre-push` and
  `.githooks/pre-push`. it catches "does it compile", not "do tests pass". run the
  release test suite yourself.
- run long builds in the BACKGROUND. a foreground shell call has a ~2 minute
  timeout and will kill a multi-minute build (you will see exit 143), then a retry
  looks like a hang. a cold release build of the affected crates is several minutes.
- the user runs gpu builds and prefers to drive builds; default to editing + handing
  off the build, or run builds in the background and report results. do not invoke
  the old c++ build (meson/ninja); the project is maturin/cargo now.
- gpu/cuda specifics for later: the gpu path is NVRTC runtime jit (no nvcc, no arch
  flag). a cuda build of the python extension exports `PyInit_gpu_ext` and
  `dev.py install --gpu` renames the dylib to `gpu_ext` so cpu and gpu coexist.

## the roadmap after the test passes (from the adr)

do these in order. each is a transport or scope substitution under the
proven-correct decomposition, not a rewrite.

1. generalize the in-process exchange behind a small transport seam (a trait or
   enum: local-copy / peer-copy / mpi). milestone-1 target: 2 gpus, uniform grid,
   no amr, no mpi, halos via `cuMemcpyPeerAsync`, distributed cfl via a 2-rank
   reduce. prove strong/weak scaling.
2. extend the equivalence test to 2d, then to rk2 (which needs a per-stage
   exchange). these expand the contract; keep them green.
3. promote the CoarseFine-skip trick to a dedicated `BoundaryType::Neighbor`
   variant once the concept is proven. this is the ~49-site enum change; do it only
   after the math is trusted, as a clarity cleanup.
4. multi-node: swap the transport for mpi. one rank per device, across nodes. under
   SPMD, bind each rank with `CUDA_VISIBLE_DEVICES` so the existing single-device
   code (the global cuda context, `cuDeviceGet(0)`, the global dispatcher) stays
   valid verbatim, per rank. add a distributed cfl allreduce and distributed
   checkpoint i/o.
5. compose decomposition WITH amr (load-balance refined patches across ranks). this
   is the genuinely hard part; defer until 1-4 are solid.

explicitly do NOT do now: build an mpi layer, a device scheduler, a load balancer,
or thread `device_id` through the xpu layer. all premature. yagni.

## key file map (so you do not re-explore)

- the adr (decided strategy): `docs/design/36_scalability_multi_gpu.md`
- the complete reusable decomposition layer (D-generic, Domain-based at the cell level):
  `src/crates/symbi-sim/src/decomp.rs` -- `HaloTransport` trait; three impls: `LocalCopy`
  (host), `DeviceCopy` (pooled direct strided device kernel), `StagedCopy` (pooled
  gather -> contiguous buffer -> scatter, the pack/unpack a peer/mpi move reuses);
  `exchange_faces` (per-face); `exchange_grid` (two-pass over the tile grid);
  `flatten`/`unflatten`. re-exported as `symbi::sim::decomp`. gpu validation:
  `cargo test -p symbi --features cuda --test decomp_equivalence --release` (gpu_d1 routes
  through DeviceCopy, gpu_d2 through StagedCopy).
- per-step primitive: `step_once` in `src/crates/symbi/src/sim/evolve.rs`
- the equivalence test (5 passing oracle; grid orchestration + tile setup still here):
  `src/crates/symbi/tests/decomp_equivalence.rs`
- slab + copy primitives: `src/crates/symbi-amr/src/refinement/transfer.rs`
  (`cf_ghost_slabs`, `copy_field`, plus `prolong_*`/`restrict_*` for reference)
- the decomposition substrate (study how amr does halos): `src/crates/symbi-amr/
  src/refinement/hierarchy.rs` (`LevelData`, `prolong_cf`, `advance_level`,
  `step_root`, flux registers)
- boundary types: enum + `Boundaries` in `src/crates/symbi-sim/src/state.rs` (grep
  `BoundaryType`, `Boundaries`, `fn axis`); the Skip mapping in
  `src/crates/symbi-substrate/src/kernels/support.rs` (grep `CoarseFine`)
- single-device globals to make per-rank-aware much later (NOT now):
  `src/crates/symbi-xpu/src/cuda.rs` (the `CUDA_CTX` once-lock, `cuDeviceGet(.., 0)`,
  `ctx_sync`, `create_stream` ignoring its device id, `UnifiedMemory::allocate`);
  `src/crates/symbi-xpu/src/runtime.rs` (the global `DISPATCHER`);
  `src/crates/symbi-exec/src/engine.rs` (`run_gpu`, `field_reduce_device`);
  `src/crates/symbi-grid/src/field.rs` (`Field` carries `Locality`, no device id)

## how to know you are done with each step

- done so far: `decomp_equivalence` passes in release (4 tests: 1d 2/4-tile, 2d
  single-axis, 2d 2x2), all < 1e-12.
- each next step adds a test (rk2 equivalence, 3d equivalence, 2-gpu scaling numbers)
  and keeps all prior tests green. never advance a transport or integrator without its
  equivalence test green first. the test is the contract.
