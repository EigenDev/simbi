// =============================================================================
// symbi-exec/src/lib.rs
//
// the EXECUTION LAYER crate (docs/design/40): the backend-neutral dispatch engine
// + the CPU parallelism policy, lifted out of the `symbi` orchestration crate so the
// layering is compiler-enforced, not convention. it consumes ONLY neutral artifacts
// (a kernel NAME resolved through the symbi-aot registry, `&[&Field]` buffers over a
// Domain, the packed ints/scalars tails, an ExecPolicy) — never a physics string or a
// SimState. it depends only on LOWER crates (algebra/grid/aot/ir/xpu), so there is no
// cycle back to `symbi`.
//
// usage:
//  symbi_exec::policy::dispatch_fields::<Sc, Mem, D>(name, allocated, exec, ins, outs, ints, scalars);
//  symbi_exec::engine::dispatch::<Sc, Mem, _>(inv, ir, name, cpu);
// =============================================================================

pub mod engine;
pub mod policy;
pub mod layout;
