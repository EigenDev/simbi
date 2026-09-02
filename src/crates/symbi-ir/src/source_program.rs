// =============================================================================
// source_program.rs
//
// the opaque source-program artifact: a traced expression graph, its scalar
// parameter names, and its output roots. physics builders construct one by
// tracing carrier-generic code (`SourceProgram::trace`) and compose programs
// inside a later trace via `TraceCx::splice_source`; the graph representation
// stays private to the compiler layer, which reads it through the accessor
// views and the graph-level `splice_into`.
//
// usage:
//   let prog = SourceProgram::trace(|cx| vec![cx.scalar("rho") * cx.lit(2.0)]);
//   let outs = trace(|cx| {
//       let s = cx.splice_source_as_scalars(&prog);
//       ...
//   });
// =============================================================================

use std::collections::HashMap;

use crate::graph::{Graph, NodeId};
use crate::gv::{Gv, TraceCx, trace};
use crate::passes::splice::splice_graph;
use crate::symbol::Symbol;

/// a traced source expression: graph + named scalar params + output roots.
///
/// `params` lists every scalar the trace touched, in first-seen order. this is
/// a superset of what the outputs consume: a builder that evaluates a whole
/// conserved vector and publishes one component traces the other components
/// too, and their scalars stay listed. the surplus is deliberate — a caller
/// supplies values positionally against a `LoweredFn`, and `scalarize` lowers
/// every node in the graph rather than only the live ones, so the two lists
/// line up entry for entry precisely because neither is pruned.
#[derive(Debug, Clone)]
pub struct SourceProgram {
    graph: Graph,
    params: Vec<String>,
    outputs: Vec<NodeId>,
}

impl SourceProgram {
    /// trace a carrier-generic build into a standalone program. the closure
    /// runs inside a fresh isolated trace: it declares its scalar leaves via
    /// `cx.scalar` (deduped, first-seen order — that ordering is the `params`
    /// contract) and returns the output values. an enclosing trace, if any,
    /// is saved and restored, so a program can be built partway through a
    /// kernel trace.
    pub fn trace(build: impl for<'t> FnOnce(TraceCx<'t>) -> Vec<Gv<'t>>) -> Self {
        let (kernel, outputs) =
            trace(|cx| build(cx).iter().map(|g| g.node()).collect::<Vec<NodeId>>());
        SourceProgram {
            params: kernel
                .scalar_params()
                .iter()
                .map(|param| param.as_str().to_string())
                .collect(),
            graph: kernel.graph().clone(),
            outputs,
        }
    }

    /// compiler view of the traced graph. lowering, validation, and backends
    /// read this; physics composes programs through the trace instead.
    pub fn graph(&self) -> &Graph {
        &self.graph
    }
    pub fn params(&self) -> &[String] {
        &self.params
    }
    pub fn outputs(&self) -> &[NodeId] {
        &self.outputs
    }

    /// splice this program's operations into `dest`, substituting its param
    /// leaves via `name_to_node`. returns the dest NodeIds for each output in
    /// order. graph-level door for kernel builders that assemble their own
    /// substitution environments from raw nodes.
    ///
    /// **panics** if a param leaf has a name absent from `name_to_node`
    /// (programmer error — every declared param needs a substitute), or if
    /// the graph contains an Op variant outside the splice's algebraic subset
    /// (Const / Param / ElementWise / Transcendental / Select — the whole
    /// vocabulary source builders emit).
    pub fn splice_into(
        &self,
        dest: &mut Graph,
        name_to_node: &HashMap<String, NodeId>,
    ) -> Vec<NodeId> {
        let subst: HashMap<Symbol, NodeId> = name_to_node
            .iter()
            .map(|(name, &nid)| (Symbol::intern(name), nid))
            .collect();
        splice_graph(dest, &self.graph, &self.outputs, &subst)
            .unwrap_or_else(|e| panic!("SourceProgram::splice_into: {e}"))
    }

    /// render this program as a standalone CUDA `__global__` kernel with the
    /// given entry name: one input pointer per param, one output pointer per
    /// output, elementwise over `n_cells`.
    pub fn cuda_source_kernel(&self, entry_name: &str) -> String {
        crate::backends::cuda::emit_source_kernel(&self.graph, &self.params, &self.outputs, entry_name)
    }
}

impl<'t> TraceCx<'t> {
    /// splice a program into the active trace, binding each param leaf to the
    /// carrier value named for it in `env`. returns the program's outputs as
    /// branded values. params absent from `env` panic — every declared param
    /// needs a caller-side substitute.
    pub fn splice_source(
        self,
        program: &SourceProgram,
        env: &HashMap<String, Gv<'t>>,
    ) -> Vec<Gv<'t>> {
        let nodes: HashMap<String, NodeId> = env
            .iter()
            .map(|(name, gv)| (name.clone(), gv.node()))
            .collect();
        let outs =
            self.with_trace(|t| program.splice_into(t.graph(), &nodes));
        outs.into_iter().map(|n| self.gv(n)).collect()
    }

    /// splice a program into the active trace, binding every param to a
    /// same-named scalar leaf of this trace (deduped by the trace, so a param
    /// shared with the surrounding kernel lands on one runtime scalar).
    pub fn splice_source_as_scalars(self, program: &SourceProgram) -> Vec<Gv<'t>> {
        let env: HashMap<String, Gv<'t>> = program
            .params()
            .iter()
            .map(|name| (name.clone(), self.scalar(name)))
            .collect();
        self.splice_source(program, &env)
    }
}
