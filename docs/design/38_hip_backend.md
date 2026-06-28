# 38 — amd hip backend (rocm)

status: implemented (rust compiles under `--features hip`); runtime + multi-gpu validation
is cluster-only (no amd hardware locally, same constraint as docs/design/37 M3/M4).

## why this doc

simbi runs on nvidia gpus through a runtime-jit path (nvrtc -> ptx -> driver). amd clusters
are common, and the abstraction was built for a second backend from day one (`GpuRuntime`:
"adding a new backend (HIP, Metal, SYCL) = implement GpuRuntime"). this doc records the
decision to add hip as a first-class, co-equal backend and the feature/type structure that
keeps it (and any third backend) pluggable rather than bolted on.

## decision

- hip is a SIBLING backend to cuda behind the existing `ExecutionSpace` / `MemorySpace` /
  `GpuRuntime` / `GpuBackend` traits. no trait changes.
- the device gate is split: a `gpu` umbrella feature gates all BACKEND-AGNOSTIC device code
  (transports, run_gpu, the reduce path, the view ABI, gpu tests); `cuda` and `hip` select the
  CONCRETE backend and each imply `gpu`. backend-agnostic code reads `#[cfg(feature = "gpu")]`;
  only the concrete backend modules + their dispatchers read `cuda` / `hip`.
- exactly one backend at a time. enabling both is a `compile_error!`; `cuda` wins if both are
  somehow set (the `hip` arms are `cfg(all(hip, not(cuda)))`), so the tree never double-defines.
- downstream code names NEUTRAL types -- `symbi_xpu::DeviceSpace` / `DeviceMemory`,
  `symbi_xpu::runtime::DeviceRuntime` / `current_dispatcher`, and the neutral device api
  (`ctx_sync`, `with_device`, `device_count`, `memcpy_peer`, ...). the concrete `CudaSpace` /
  `UnifiedMemory` names stay exported under `cuda` for the cuda-specific backend tests only.

this is the forward-thinking payoff: a third backend (metal/sycl) is "add `backendX.rs` + a
runtime module + three Cargo arms + a `DeviceSpace` alias arm" -- no edits to the ~40 device
call sites, because they already name neutral types and the `gpu` gate.

## what was already in place (the groundwork paid off)

- `symbi_ir::emit::Target::Hip` exists; `header(Hip)` emits `#include <hip/hip_runtime.h>` and
  `global_qualifier` treats `Cuda | Hip` identically. the rendered kernel source is the SAME
  cuda-c++ dialect -- hip's headers define `__global__`, `blockIdx`, `threadIdx`, etc. so NO
  ir/codegen change was needed; `HipBackend::TARGET = Target::Hip`.
- `GpuRuntime` + `KernelDispatcher<R>` are backend-generic. hip is one impl.
- `GpuBackend` (engine.rs) already funnels every cuda-named leak (render target, dispatcher,
  the field-view launch ABI) through one trait. hip is a second unit-struct impl.
- no warp-shuffle reductions exist (block reductions use shared memory + `__syncthreads`), so
  amd's 64-wide wavefront vs nvidia's 32 needs no kernel changes.

## hip vs cuda: the binding differences that matter

1. NO context api. cuda needs `cuCtxCreate` + `cuCtxSetCurrent` (modules bind to a context).
   hip binds per-DEVICE: `hipSetDevice(ord)` is the per-thread current device, and the primary
   context is implicit. so `hip.rs` has no context registry -- `with_device` is just save /
   `hipSetDevice` / restore, and the per-device dispatcher registry keys on the ordinal exactly
   as before. simpler than cuda.
2. peer copy takes ORDINALS, not contexts: `hipMemcpyPeer(dst, dstDev, src, srcDev, bytes)`.
   the neutral `memcpy_peer(dst, dst_ord, src, src_ord, bytes)` maps cleanly; hip just forwards
   the ordinals (modulo physical count), cuda looks up the two contexts. peer enable is
   `hipDeviceEnablePeerAccess(peer, 0)` with the source device current.
3. jit output is a final code OBJECT for a specific arch, not re-jitted ptx. so the target arch
   matters: hiprtc takes `--offload-arch=gfx<NNN>`. we make it OPTIONAL via `SYMBI_HIP_ARCH`
   (e.g. `gfx90a` for MI200, `gfx942` for MI300); unset = let hiprtc default to the attached
   device. this is the ONE thing to set on the cluster if auto-detect is insufficient.
4. managed memory: `hipMallocManaged(ptr, size, hipMemAttachGlobal)`. same global-attach
   semantics; the same managed-thrash caveat as cuda (docs/design/37) applies, more so on amd
   where managed support depends on xnack -- a device-local memory space is the perf follow-up.

## the local-validation limit (same as 37 M3/M4)

no amd gpu here, so hip is CODE-AND-CLUSTER-VALIDATE. what is verified locally:
- `cargo check --features hip` type-checks the entire hip path (raw `extern "C"` bindings need
  no rocm headers/libs; check does not link).
- `cargo {check,test} --features cuda` stays green -- the refactor must not regress nvidia.
- `cargo check` (no gpu feature) stays green.
what is NOT verifiable locally: the hiprtc compile, any kernel launch, peer copy, scaling. those
run on the rocm cluster. the equivalence oracle (`decomp_equivalence.rs`, mono vs decomposed)
becomes the hip correctness gate there, unchanged.

## file/▸change map

new:
- `symbi-xpu/src/hip.rs` -- `HipSpace` (ExecutionSpace), `HipManaged` (MemorySpace), raw
  `extern "C"` driver bindings, per-device binding (`with_device`/`current_device`/
  `device_count`), peer api (`can_access_peer`/`enable_peer_access`/`memcpy_peer`), `ctx_sync`,
  `device_info`. mirrors `cuda.rs` minus the context registry.
- `symbi-xpu/src/hiprtc.rs` -- `compile_code`: hiprtc source -> code object, optional
  `--offload-arch` from `SYMBI_HIP_ARCH`. mirrors `nvrtc.rs`.

changed (mechanical):
- `symbi-xpu/src/lib.rs` -- `gpu_backend` alias to the active backend; neutral re-exports
  (`DeviceSpace`/`DeviceMemory` + device api); `DefaultSpace`/`DefaultMemory` keyed on `gpu`;
  `with_device` keyed on `gpu`; mutual-exclusion `compile_error!`.
- `symbi-xpu/src/runtime.rs` -- `hip_runtime` module (mirrors `cuda_runtime`); neutral
  `DeviceRuntime` + `current_dispatcher` re-exports keyed on the active backend.
- `symbi-xpu/build.rs` -- hip arm: link `amdhip64` (+ `hiprtc` if separate), search
  `ROCM_PATH/lib`.
- `symbi-exec/src/engine.rs` -- `HipBackend` impl; `DefaultGpuBackend` keyed on backend; the
  view ABI + `GpuBackend` trait + `run_gpu` + reduce path move from `cuda` to `gpu`; concrete
  cuda types -> neutral (`DeviceMemory`, `runtime::DeviceRuntime`, `symbi_xpu::ctx_sync`).
- `symbi-sim/src/decomp.rs`, `symbi-hydro/src/gpu_launcher.rs` -- `cuda` -> `gpu` on the device
  transports; cuda paths -> neutral types.
- `symbi/src/prelude.rs`, `symbi/benches`, the backend-agnostic `*_gpu` tests -- `cuda` -> `gpu`,
  `CudaSpace`/`UnifiedMemory` -> `DeviceSpace`/`DeviceMemory`. the xpu-internal cuda backend
  tests (`cuda_smoke`, `gpu_regimes`, `gpu_srhd_c2p`, `cpu_gpu_minmax_oracle`, `multi_device`,
  `device_view_abi`) stay `cuda`-gated -- they probe nvidia specifics.
- every crate `Cargo.toml`: `gpu` feature + `hip` feature cascading like `cuda`.

## staged plan

H1 (DONE) -- feature plumbing + neutral type surface in symbi-xpu (no behavior change; `cuda`
stays green, `gpu`-gated code now compiles under either backend selector).
H2 (DONE) -- `hip.rs` + `hiprtc.rs` + `hip_runtime` + `HipBackend`.
H3 (DONE) -- workspace sweep of device-shared sites to neutral types + `gpu` gate. ALL THREE
CONFIGS GREEN LOCALLY: `cargo check -p symbi -p symbi-hydro -p symbi-xpu --features cuda --tests`,
`... --features hip --tests` (linker needs the amd libs, so a local hip check points `ROCM_PATH`
at empty stub `.so`s -- `cargo check` only links build scripts, not the lib/test crates, so the
stubs suffice to type-check the whole hip path), and the no-gpu `cargo check -p symbi --tests`.
the on-device cuda decomp oracle still passes (the neutral-type refactor did not regress nvidia).
H4 (NEEDS the rocm cluster) -- build with `--features hip` against real rocm, run the
equivalence oracle on amd, then the multi-gpu peer path (37 M3) and scaling (37 M4) on amd.

## how to build/run on the rocm cluster

```
# one-time, if hiprtc auto-detect of the gpu arch is insufficient:
export SYMBI_HIP_ARCH=gfx90a        # MI200; gfx942 for MI300, etc.
export ROCM_PATH=/opt/rocm          # if rocm is not at the default

# the python extension (mirrors the cuda flow; dev.py renames it to gpu_ext):
./dev.py install --hip              # if dev.py grows a --hip flag; else:
maturin develop --features hip

# the correctness oracle on amd (the cluster gate, same as cuda):
cargo test -p symbi --features hip --release --test decomp_equivalence
```
