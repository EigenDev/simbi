// =============================================================================
// symbi-exec/src/lib.rs
//
// the execution layer crate: the backend-neutral dispatch engine
// + the CPU parallelism policy, lifted out of the `symbi` orchestration crate so the
// layering is compiler-enforced. it consumes only neutral artifacts
// (a kernel name resolved through the symbi-aot registry, `&[&Field]` buffers over a
// Domain, the packed ints/scalars tails, an ExecPolicy); a physics string or a
// SimState stays on the other side of the boundary. it depends only on lower crates
// (algebra/grid/aot/ir/xpu), so there is no cycle back to `symbi`.
//
// usage:
//  symbi_exec::policy::dispatch_fields::<Sc, Mem, D>(name, allocated, exec, ins, outs, ints, scalars);
//  symbi_exec::engine::dispatch::<Sc, Mem, _>(inv, ir, name, cpu);
// =============================================================================

pub mod engine;
pub mod layout;
pub mod policy;
