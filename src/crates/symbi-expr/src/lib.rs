// =============================================================================
// symbi-expr
//
// expression DAG and register-based VM evaluator. constructs mathematical
// expressions as directed acyclic graphs, linearizes them into a flat
// instruction stream via topological sort + register allocation, and
// evaluates them with a stack-free register machine.
//
// pipeline: Dag (build) -> linearize (compile) -> evaluate (run)
//
// designed for user-defined initial conditions, source terms, and boundary
// conditions. the linearized instruction stream is GPU-compatible (flat
// data, fixed register bank, no recursion).
//
// usage:
//   let mut dag = Dag::new();
//   let x = dag.var_x1();
//   let c = dag.constant(2.0);
//   let r = dag.mul(c, x);
//   let expr = dag.compile(&[r]);
//   assert_eq!(expr.eval(3.0, 0.0, 0.0, 0.0)[0], 6.0);
// =============================================================================

pub mod op;
pub mod dag;
pub mod linearize;
pub mod eval;
pub mod load;
pub mod strength;

pub use op::Op;
pub use dag::{Dag, Node, Payload};
pub use linearize::{Instr, linearize, max_register};
pub use eval::{Expression, evaluate};
pub use load::{NodeDesc, LoadError, SourceConfig, load_expression, nodes_from_descs};
pub use strength::strength_reduce;
