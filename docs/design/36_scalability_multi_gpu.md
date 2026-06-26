# 36 - scalability: multi-gpu and multi-node

status: accepted (direction), unimplemented
date: 2026-06-26
supersedes: none
related: 21 (amr), 35 (gpu backend seam), 15 (runtime kernel compilation)

## context

multi-gpu is on the near horizon; multi-node is the eventual target. the decision
that determines whether we shoot ourselves in the foot is the PARALLELISM MODEL,
because it dictates how much of the existing single-device code survives and
whether the multi-gpu work is reusable for multi-node.

two facts from the 2026-06-26 architecture review drive this document:

1. the single-device assumption is real but localized. one process == one cuda
   context (`symbi-xpu/src/cuda.rs:176` `CUDA_CTX: OnceLock`), device ordinal
   hardcoded (`cuda.rs:189` `cuDeviceGet(.., 0)`), one global dispatcher
   (`symbi-xpu/src/runtime.rs:195` `DISPATCHER`), one global ctx_sync
   (`cuda.rs:148`), `Field` carries no device id (`symbi-grid/src/field.rs`).

2. we already own a domain-decomposition engine: amr. each `LevelData` owns a
   self-contained `SimStateGeneric` (fields + geometry + boundaries + ghosts)
   (`symbi-amr/src/refinement/hierarchy.rs:80`); halo regions are explicitly
   typed (`BoundaryType::CoarseFine`, `symbi-sim/src/state.rs:1125`) and computed
   (`cf_ghost_slabs`, `symbi-amr/src/refinement/transfer.rs:84`); inter-subdomain
   halo exchange exists with time interpolation (`prolong_cf`, `hierarchy.rs:758`);
   conservative coupling exists (flux registers + restriction/reflux); per-level
   cfl and subcycling exist (`hierarchy.rs:497`).

decomposition is therefore not a green-field build. it is "generalize the
subdomain we already have so a neighbor can live on another device/rank."

## decision

adopt SPMD with one rank per gpu. each rank is a process that owns exactly one
device, uses the existing ambient-context single-device code unchanged, holds one
subdomain, and exchanges halos with neighbor ranks over a transport
(peer-copy / nvlink intra-node, mpi inter-node). distributed reductions (cfl)
become an allreduce.

the unit of decomposition is the existing self-contained subdomain
(`SimStateGeneric`-owning structure, today `LevelData`). it gains a rank/device
owner and the ability to exchange SAME-LEVEL neighbor halos, not only
coarse-fine halos.

## why SPMD and not single-process-multi-device

the obvious alternative is to thread an explicit `device_id` through context,
stream, allocation, dispatch, and sync (the ~5-tier refactor enumerated in the
review). rejected:

- it is invasive on the hot launch path and adds parameters that are always 0
  until a consumer exists -- speculative generality, untestable single-gpu.
- it does NOT extend to multi-node. we would build it, then redo it for the
  cluster. two systems for one goal.

SPMD wins on the axis that matters here -- one architecture spans the whole
roadmap:

- single-device code STAYS. `CUDA_CTX`, `DISPATCHER`, `cuDeviceGet(0)`,
  `ctx_sync()` are all correct when each rank genuinely owns one device. the
  tier-1..5 device-threading refactor largely evaporates.
- 1 gpu -> N gpus on a node -> N nodes is the SAME code with more ranks.
  multi-node is not a second system; it is more of the first.
- it is the model every production finite-volume astro code uses (athena++,
  flash, gamer) for exactly this reason.

cost (accepted): the python frontend must launch ranks (`mpirun -n N python ...`),
each rank constructs its own subdomain, and cfl becomes a distributed allreduce.
this is a real orchestration lift on the python side, isolated to the launch and
reduction paths.

## the one structural gap

inter-subdomain exchange today is coarse<->fine ONLY. physical bcs and cf
prolongation exist; SAME-LEVEL neighbor halo exchange does not
(`transfer.rs` has no same-resolution neighbor copy). splitting a uniform grid
across ranks creates same-resolution neighbor boundaries, so this exchange path
is the single genuinely-new primitive. it is also what multi-patch amr needs, so
it pays double. it must be abstracted over a transport so the same code does
local copy, peer copy, and mpi.

## consequences

- the memory model needs attention before scale. `cuMemAllocManaged` with
  `CU_MEM_ATTACH_GLOBAL` (`cuda.rs:357`) thrashes across many devices. under
  SPMD each rank should allocate on ITS device; global-attached managed memory
  stops being the only model. this is a smaller change than the device-threading
  refactor and is deferred to the first multi-gpu milestone, not now.
- the subdomain abstraction must NOT fork. "device/rank subdomain" and "amr
  patch" are the same concept (owns state+geometry+boundaries+ghosts, has typed
  neighbors). keeping them unified avoids maintaining two halo systems.
- reductions: `field_reduce_device` (`symbi-exec/src/engine.rs:250`) stays
  per-rank; only the final host fold gains a cross-rank allreduce. the global
  partials buffers (`engine.rs:208`) remain per-process and are correct under
  SPMD.

## staged roadmap

now (cheap, low-regret, while the code is small):
- do NOT deepen two assumptions: (a) new code always goes through
  `Domain`/`Boundaries`/`SimStateGeneric`, never "the domain is the whole
  problem"; (b) do not let more code assume `cuMemAllocManaged(GLOBAL)` "all
  memory is everywhere."
- this document is the guardrail. every feature is built knowing decomposition
  is coming and that the subdomain is the unit.

milestone 1 -- first multi-gpu (the near horizon):
- build same-level neighbor halo exchange as an abstraction over a transport
  (local copy -> peer copy -> mpi). generalize `cf_ghost_slabs` / `prolong_cf`
  to a same-level `Neighbor` boundary type.
- target: 2 gpus, uniform grid, NO amr, NO mpi. split the domain, exchange halos
  via `cuMemcpyPeer`, distributed cfl via a 2-rank reduce. prove strong/weak
  scaling. smallest possible consumer that validates the abstraction.

milestone 2 -- multi-node:
- swap the transport for mpi. rank-per-device across nodes. distributed
  checkpoint i/o.

milestone 3 -- decomposition composed with amr (far future):
- load-balance refined patches across ranks. this is the genuinely hard part
  (amr + spmd load balancing); deferred until 1 and 2 are proven.

explicitly NOT now: mpi layer, device scheduler, load balancer, the tier-1..5
device-threading refactor. all premature.

## open questions

- intra-node transport: cuda peer (`cuMemcpyPeerAsync`) vs mpi-everywhere
  (cuda-aware mpi). leaning peer intra-node, mpi inter-node, behind one transport
  trait.
- python orchestration: mpi4py vs a launcher that forks ranks. affects the
  `simbi run` entry path and checkpoint naming.
- per-rank device binding: `CUDA_VISIBLE_DEVICES` per rank (keeps ordinal 0 valid
  per process) vs explicit `cuDeviceGet(local_rank)`. the former preserves the
  current single-device code verbatim and is preferred.
