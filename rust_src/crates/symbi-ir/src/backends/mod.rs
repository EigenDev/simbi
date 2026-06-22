// =============================================================================
// backends — IR-to-target source emitters (the homomorphism endpoints).
//
// each backend is the homomorphism Free<Op> -> Free<TargetOps> for one target.
// totality (every Op variant lowers in every backend) is enforced by task A5.
//
//   cpu        — Rust source emit (rank-0 expressions, used by kernel_cpu).
//   cuda       — CUDA C++ source emit (rank-0 expressions, used by kernel).
//   kernel     — kernel-level emit driver (the production CUDA path).
//   kernel_cpu — kernel-level emit driver (CPU Rust path).
//   render     — RenderPolicy + Prepared (neutral IR blob; NVRTC JIT input).
//   interp     — CPU runtime interpreter (test path, exercises the IR directly).
// =============================================================================

pub mod cpu;
pub mod cuda;
pub mod kernel;
pub mod kernel_cpu;
pub mod render;
pub mod interp;
