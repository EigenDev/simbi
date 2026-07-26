// =============================================================================
// passes — IR-to-IR transformations (the rewrite category).
//
// each pass is a structure-preserving map Free<Op> -> Free<Op>; rewrites that
// claim to be semantics-preserving must be equational laws every Carrier
// honors. these are not typed as
// `Rewrite` trait objects — that abstraction lands when consumers force it.
//
//   scalarize — tensor IR -> LoweredFn (rank-N to scalar). the production
//               kernel-mode lowering: scalarize_kernel -> prepare -> render.
//   splice    — Graph -> Graph composition primitive.
//   cse       — common subexpression elimination on the scalarized form.
//   pressure  — peak-register-pressure analysis;
//               powers `assert_peak_pressure!` for per-kernel bounds.
//   mask_form — float bool/if -> cmp_*/select spelling (branch-free bodies
//               for the Rust CPU backend), arm-cost gated: a select whose arm
//               divides or calls out keeps bool/if form.
//   unswitch  — param-invariant selects (the limiter pick) partially
//               evaluated both ways; the emitter renders two specialized
//               loop nests behind one per-call branch.
//   lazy_select — expensive select arms rescheduled as real branches with
//               arm-exclusive lets sunk in (runs in prepare).
//   stencil_reach — per-field, per-axis halo reach read off FieldLoadAt index
//               expressions; powers the ghost-width law.
// =============================================================================

pub mod cse;
pub mod lazy_select;
pub mod mask_form;
pub mod pressure;
pub mod scalarize;
pub mod splice;
pub mod stencil_reach;
pub mod unswitch;
