# 37 - device binding: one process, many gpus (intra-node nvlink)

status: scoping (no code yet)
date: 2026-06-28
revises: 36 (scalability) -- the transport-model choice, see "decision" below
related: the `decomp` module (`symbi-sim/src/decomp.rs`), the gpu dispatch path
         (`symbi-exec/src/engine.rs`), the cuda layer (`symbi-xpu/src/cuda.rs`)

## why this doc

the in-process decomposition + halo transport are done and gpu-validated on one card
(adr 36): `LocalCopy`, `DeviceCopy`, `StagedCopy`, all green against the oracle. the next
step is to actually spread a run across multiple gpus on a node, over nvlink. that needs
two things the single-device code does not have: per-device CONTEXTS (so kernels can run on
each gpu) and a cross-device TRANSPORT (peer copy). this doc scopes the first; the transport
is mostly done (`StagedCopy` is the pack/unpack; only the peer `cuMemcpyPeer` move is new).

## decision (revises adr 36)

adr 36 chose PURE SPMD: one rank (process) per gpu, mpi everywhere, specifically to AVOID a
device-binding refactor (each rank owns one device, so the single-device globals stay valid
per rank). the new direction -- "nvlink intra-node, mpi only when multi-node" -- means ONE
process driving MULTIPLE gpus, which is the single-process-multi-device model adr 36 set
aside. that is a deliberate revision:

- intra-node: ONE process per node, device-binding within the node, nvlink peer copy. NO mpi.
- inter-node (later): mpi between nodes (one rank per node, or per gpu). the transport's
  "move" step swaps `cuMemcpyPeer` for an mpi send/recv; everything else is unchanged.

this hybrid (device-binding intra-node + mpi inter-node) is the standard mpi+cuda layout and
is what the nvlink-first goal requires. the cost vs pure spmd: we now OWN the device-binding
refactor adr 36 avoided. accepted.

## design: ambient "current device", not threaded device_id

cuda's driver api is built around a CURRENT context per host thread (`cuCtxSetCurrent`).
every existing launch/alloc/sync already targets "whatever context is current". so the least
invasive model is: keep that, and make the RIGHT context current before a tile's work --
rather than threading a `device_id` parameter through every launch signature (the hot path).

the model:
- a per-device context registry (lazily-created contexts, keyed by gpu ordinal).
- `with_device(ordinal, || { ... })` sets that device's context current for the closure,
  restoring the prior on exit. a tile's physics kernels run inside `with_device(tile.device)`.
- because contexts are per-thread-current, a worker thread bound to one gpu is the natural
  unit; the simple first cut is single-threaded (one tile's kernels at a time, switching the
  current device between tiles). real overlap (a thread/stream per gpu) is a later optimization.

this keeps the launch/alloc/sync signatures UNCHANGED. the only code that learns about
devices is: the context registry, the dispatcher (see below), `Field` (gains a device id),
and the decomp transport (sets the right device around its kernels + the peer copy).

## the one landmine: modules are context-bound

a cuda module (and the kernel handles from it) belongs to the context it was loaded into. a
kernel JIT'd in device A's context CANNOT launch in device B's. so the single global
`DISPATCHER` (one module cache, `runtime.rs:195`) BREAKS across devices: the second device
would get "invalid resource handle". the dispatcher must become PER-DEVICE -- a registry of
dispatchers keyed by ordinal (each with its own context + module cache), or one dispatcher
whose cache key includes the device. the per-device-dispatcher shape is cleaner and matches
the per-device-context registry.

## the minimal change set (from the current-code map)

| area | today (file:line) | change |
|------|-------------------|--------|
| context | `CUDA_CTX: OnceLock<SyncCtx>` (cuda.rs:210); `ensure_init` hardcodes `cuDeviceGet(0)` (cuda.rs:224) | per-device context registry; `ensure_init_device(ord)`; `with_device(ord, f)` |
| sync | `ctx_sync()` syncs current ctx (cuda.rs:182) | unchanged (it syncs whatever `with_device` made current); add a device-count + peer-enable helper |
| streams | `create_stream(device_id)` IGNORES the id (cuda.rs:282); launch uses null stream (runtime.rs:189) | make `create_stream` honor the id (set ctx current first); per-device stream is a later overlap optimization |
| dispatcher | single global `DISPATCHER` (runtime.rs:195) | per-device dispatcher registry (modules are context-bound -- see landmine) |
| launch | `B::dispatcher().runtime().launch(...)` (engine.rs:664) | resolve dispatcher for the CURRENT device; signatures unchanged |
| field | `Locality` enum, no device id (field.rs:52) | add `device_id: i64`; set at construction from the current device |
| alloc | `cuMemAllocManaged(GLOBAL)` (cuda.rs:442) | unchanged for now (managed is global). NOTE: managed across many gpus page-thrashes; a device-local memory space is a later perf step |
| peer access | none | add `cuDeviceCanAccessPeer`, `cuCtxEnablePeerAccess`, `cuMemcpyPeer(Async)` bindings + a one-time enable for exchanging device pairs |
| decomp transports | `DISPATCHER` + `ctx_sync` global (decomp.rs DeviceCopy/StagedCopy) | device-aware: gather on src device, `cuMemcpyPeer` to dst device buffer, scatter on dst device, sync the right device |

## staged plan (each step validated as far as the one 2070 allows)

M1 (DONE) -- per-device context + dispatcher registries; `with_device`. behaviorally a no-op
for a single device (the registry holds one entry; current = device 0). VALIDATED: the entire
existing suite (cpu + single-gpu decomp tests) stayed green. fully local.

M2 (DONE) -- tiles bound to LOGICAL devices, wrapped in `with_device`. NO `Field.device_id`:
the ambient current-device model + managed-global memory make per-field ids unnecessary
(yagni). `ensure_init_device` round-robins logical ordinals onto physical devices
(`ord % count`), so logical device 1+ are distinct contexts on the one card; `lib.rs` exposes
a uniform `with_device` (host build = `f()`). VALIDATED LOCALLY on two contexts on the one
2070: `symbi-xpu/tests/multi_device.rs` launches a kernel in each context through its
per-device dispatcher (proves the context-bound-module landmine is handled), and
`decomp_equivalence.rs` binds each tile's allocation + every physics kernel to its logical
device (round-robin over `NDEV=2`), drains all contexts at the exchange/read seams, runs the
host-orchestrated exchange on device 0 over managed-global memory, and still matches the
monolithic run to < 1e-12. NOT validated locally: real parallelism (the contexts share one
gpu) and true cross-gpu access.

M3 -- peer access bindings + the peer `HaloTransport`: `StagedCopy` gather on src device,
`cuMemcpyPeer` of the contiguous buffer to the dst device's buffer, scatter on dst device.
LOCALLY: the gather/scatter are already proven; the peer move between two contexts on one
device is a same-device copy (validates wiring, not nvlink). the real cross-gpu peer copy +
nvlink is the cluster-tested piece -- one well-defined call.

M4 (NEEDS the node) -- real multi-gpu run: tiles on distinct gpus, nvlink peer copy,
distributed cfl reduce across devices, perf/scaling. the oracle (mono vs decomposed) still
applies, now across real devices.

## risks

- HOT-PATH CHURN: minimized by the ambient model (launch/alloc/sync signatures unchanged).
  the churn is concentrated in the context + dispatcher registries.
- MODULE CONTEXT-BINDING: the per-device dispatcher is mandatory, not optional; getting it
  wrong yields "invalid resource handle" only on the 2nd device (invisible on one gpu). the
  logical-devices-on-one-card validation (M2) is what surfaces it locally.
- MANAGED MEMORY THRASH: `cuMemAllocManaged(GLOBAL)` across many gpus migrates pages over the
  bus and can dominate. fine for first correctness; a device-local memory space + explicit
  halo copy (which `StagedCopy` already is) is the performant follow-up. do not over-invest in
  managed-everywhere.
- THREAD/CURRENT-DEVICE SAFETY: `cuCtxSetCurrent` is per-thread. the single-threaded first cut
  is safe; a thread-per-gpu overlap model must keep each thread's current device consistent.

## open questions

- registry shape: `OnceLock<Mutex<HashMap<i64, ...>>>` vs a fixed `[Option<...>; MAX_GPUS]`.
  leaning on a small fixed array (gpu counts are tiny, avoids locking on the hot path).
- one process per NODE managing all its gpus, vs one process per gpu intra-node with cuda IPC.
  per-node-process + device-binding is simpler and is assumed here; revisit if IPC is needed.
- where `with_device` lives: `symbi-xpu` (next to the context registry) and is re-exported.
- does the substrate's per-stage dispatch need any change, or does wrapping the whole tile
  step in `with_device` suffice? RESOLVED (M2): the wrap suffices. no substrate change -- the
  decomp oracle wraps each tile's per-stage kernel group (flux/godunov/c2p/ghost_fill, plus
  cfl + snapshot) in `with_device` and matches the monolithic run, because every kernel routes
  through `current_dispatcher()`/`current_device()` and reads the current context.
