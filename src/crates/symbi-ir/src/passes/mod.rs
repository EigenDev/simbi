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
// =============================================================================

pub mod scalarize;
pub mod splice;
pub mod cse;
pub mod pressure;
