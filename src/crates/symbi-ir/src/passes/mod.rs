// =============================================================================
// passes — IR-to-IR transformations (the rewrite category).
//
// each pass is a structure-preserving map Free<Op> -> Free<Op>; rewrites that
// claim to be semantics-preserving must be equational laws every Carrier
// honors (A1 in docs/design/00_axioms.md). these are not typed as
// `Rewrite` trait objects — that abstraction lands when consumers force it.
//
//   scalarize — tensor IR -> LoweredFn (rank-N to scalar). the production
//               kernel-mode lowering: scalarize_kernel -> prepare -> render.
//   splice    — Graph -> Graph composition primitive.
//   cse       — common subexpression elimination on the scalarized form.
//   pressure  — peak-register-pressure analysis (docs/design/23);
//               powers `assert_peak_pressure!` for per-kernel bounds.
//   mask_form — float bool/if -> cmp_*/select spelling (branch-free bodies
//               for the Rust CPU backend), arm-cost gated: a select whose arm
//               divides or calls out keeps bool/if form (docs/design/47).
//   unswitch  — param-invariant selects (the limiter pick) partially
//               evaluated both ways; the emitter renders two specialized
//               loop nests behind one per-call branch (docs/design/47).
// =============================================================================

pub mod scalarize;
pub mod splice;
pub mod cse;
pub mod pressure;
pub mod mask_form;
pub mod unswitch;
