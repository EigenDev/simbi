// =============================================================================
// linearize.rs
//
// converts an expression DAG into a flat instruction stream for the register
// VM. pipeline: reachability analysis -> evaluation order -> register
// allocation -> instruction emission.
//
// variable registers are fixed: x1=0, x2=1, x3=2, t=3. computed nodes draw
// from a recycling pool starting at register 4, so the bank has to hold the
// values simultaneously LIVE rather than one per node.
//
// usage:
//   let (instrs, output_regs) = linearize(dag.nodes(), &[root_idx]);
// =============================================================================

use crate::dag::{Node, Payload};
use crate::eval::MAX_REGISTERS;
use crate::op::Op;

/// a single VM instruction. the evaluator executes these sequentially.
#[derive(Clone, Copy, Debug)]
pub struct Instr {
    /// operation to perform.
    pub op: Op,
    /// destination register.
    pub dst: usize,
    /// source register operands. unused slots are 0.
    pub src: [usize; 3],
    /// immediate f64 value (for Op::Constant).
    pub imm_f64: f64,
    /// immediate index (for Op::Parameter).
    pub imm_idx: usize,
}

/// first register available for computed nodes (after the 4 variable registers).
const FIRST_COMPUTED_REG: usize = 4;

/// find all nodes reachable from the given output indices.
fn find_reachable(nodes: &[Node], outputs: &[usize]) -> Vec<bool> {
    let mut reachable = vec![false; nodes.len()];
    let mut stack: Vec<usize> = outputs.to_vec();

    while let Some(idx) = stack.pop() {
        if reachable[idx] {
            continue;
        }
        reachable[idx] = true;
        nodes[idx].for_each_child(|child| {
            if !reachable[child] {
                stack.push(child);
            }
        });
    }
    reachable
}

/// evaluation order (leaves first): the reachable nodes in INDEX order.
///
/// the dag is append-only and a node can only name children that already exist, so a
/// child's index is always below its parent's and index order is a valid topological
/// order by construction. it is also the order the expression was BUILT in, which is
/// what keeps register pressure low: a value is created near the point it is consumed,
/// so its live range is short.
///
/// visiting each output's subtree in turn instead — the natural depth-first reading —
/// is equally valid and can be far worse. any subexpression shared between two outputs
/// then stays live across the whole of the first output's evaluation, so a field whose
/// components share per-term temporaries holds one live value per term rather than a
/// handful in total.
fn topo_sort(nodes: &[Node], _outputs: &[usize], reachable: &[bool]) -> Vec<usize> {
    let mut result = Vec::with_capacity(nodes.len());
    for (idx, node) in nodes.iter().enumerate() {
        if !reachable[idx] {
            continue;
        }
        node.for_each_child(|child| {
            assert!(
                child < idx,
                "expression dag node {idx} names child {child} at or above its own \
                 index; the graph is no longer append-only and index order is not a \
                 topological order"
            );
        });
        result.push(idx);
    }
    result
}

/// linearize an expression DAG into a flat instruction stream.
///
/// returns (instructions, output_registers) where output_registers[ii] is
/// the register holding the value of outputs[ii] after evaluation.
pub fn linearize(nodes: &[Node], outputs: &[usize]) -> (Vec<Instr>, Vec<usize>) {
    let reachable = find_reachable(nodes, outputs);
    let order = topo_sort(nodes, outputs, &reachable);

    // register allocation. variables occupy their fixed registers; every other node
    // takes one from a free pool, and a register returns to the pool once the value it
    // holds has been read for the last time.
    //
    // RECYCLING IS WHAT MAKES LARGE EXPRESSIONS REPRESENTABLE. allocating a fresh
    // register per node makes the register count equal the NODE count, so the fixed
    // bank caps an expression at a couple of hundred nodes — while the number of values
    // simultaneously LIVE is a property of the expression's shape, not its size, and
    // stays small for the wide sums and products configs actually build (a sum of a
    // thousand terms holds one accumulator plus the term under construction).
    let mut reg_map: Vec<Option<usize>> = vec![None; nodes.len()];
    let mut next_reg = FIRST_COMPUTED_REG;
    let mut free: Vec<usize> = Vec::new();

    // the last step of `order` at which each node's value is read. outputs are read by
    // the caller after the stream ends, so they never expire.
    let mut last_use: Vec<usize> = vec![0; nodes.len()];
    for (step, &idx) in order.iter().enumerate() {
        let mut mark = |child: usize| last_use[child] = last_use[child].max(step);
        match nodes[idx].payload {
            Payload::Unary(a) => mark(a),
            Payload::Binary(a, b) => {
                mark(a);
                mark(b);
            }
            Payload::Ternary(a, b, c) => {
                mark(a);
                mark(b);
                mark(c);
            }
            Payload::None | Payload::Value(_) | Payload::ParamIdx(_) => {}
        }
    }
    for &out in outputs {
        last_use[out] = order.len();
    }

    // nodes whose register becomes reusable after each step.
    let mut expiring: Vec<Vec<usize>> = vec![Vec::new(); order.len() + 1];
    for &idx in &order {
        if !nodes[idx].op.is_variable() {
            expiring[last_use[idx]].push(idx);
        }
    }

    for (step, &idx) in order.iter().enumerate() {
        // release everything whose final read was strictly before this instruction. a
        // source of THIS instruction has last_use >= step, so it is never recycled into
        // the destination it is about to be read for.
        if step > 0 {
            for &done in &expiring[step - 1] {
                if let Some(reg) = reg_map[done] {
                    free.push(reg);
                }
            }
        }
        let node = &nodes[idx];
        if node.op.is_variable() {
            reg_map[idx] = Some(node.op.variable_register());
            continue;
        }
        let reg = free.pop().unwrap_or_else(|| {
            let reg = next_reg;
            next_reg += 1;
            reg
        });
        assert!(
            reg < MAX_REGISTERS,
            "the expression needs more than {} simultaneously live values; the register \
             bank holds {MAX_REGISTERS}",
            MAX_REGISTERS - FIRST_COMPUTED_REG
        );
        reg_map[idx] = Some(reg);
    }

    // emit instructions. variable nodes skip (pre-loaded by VM).
    let mut instrs = Vec::with_capacity(order.len());

    for &idx in &order {
        let node = &nodes[idx];
        if node.op.is_variable() {
            continue;
        }

        let dst = reg_map[idx].unwrap();

        let instr = match &node.payload {
            Payload::Value(val) => Instr {
                op: Op::Constant,
                dst,
                src: [0, 0, 0],
                imm_f64: *val,
                imm_idx: 0,
            },
            Payload::ParamIdx(pidx) => Instr {
                op: Op::Parameter,
                dst,
                src: [0, 0, 0],
                imm_f64: 0.0,
                imm_idx: *pidx,
            },
            Payload::Unary(child) => Instr {
                op: node.op,
                dst,
                src: [reg_map[*child].unwrap(), 0, 0],
                imm_f64: 0.0,
                imm_idx: 0,
            },
            Payload::Binary(left, right) => Instr {
                op: node.op,
                dst,
                src: [reg_map[*left].unwrap(), reg_map[*right].unwrap(), 0],
                imm_f64: 0.0,
                imm_idx: 0,
            },
            Payload::Ternary(cond, then_, else_) => Instr {
                op: node.op,
                dst,
                src: [
                    reg_map[*cond].unwrap(),
                    reg_map[*then_].unwrap(),
                    reg_map[*else_].unwrap(),
                ],
                imm_f64: 0.0,
                imm_idx: 0,
            },
            Payload::None => unreachable!("non-variable node with no payload"),
        };
        instrs.push(instr);
    }

    let output_regs = outputs.iter().map(|&idx| reg_map[idx].unwrap()).collect();
    (instrs, output_regs)
}

/// max register index used by the instruction stream (for sizing the register bank).
pub fn max_register(instrs: &[Instr]) -> usize {
    instrs.iter().map(|ii| ii.dst).max().unwrap_or(3)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dag::Dag;

    #[test]
    fn linearize_constant() {
        let mut dag = Dag::new();
        let c = dag.constant(42.0);
        let (instrs, out_regs) = linearize(dag.nodes(), &[c]);
        assert_eq!(instrs.len(), 1);
        assert_eq!(instrs[0].op, Op::Constant);
        assert_eq!(instrs[0].imm_f64, 42.0);
        assert_eq!(out_regs, &[instrs[0].dst]);
    }

    #[test]
    fn linearize_variable() {
        let mut dag = Dag::new();
        let x = dag.var_x1();
        let (instrs, out_regs) = linearize(dag.nodes(), &[x]);
        // variable emits no instruction — it's pre-loaded.
        assert_eq!(instrs.len(), 0);
        assert_eq!(out_regs, &[0]); // x1 = register 0
    }

    #[test]
    fn linearize_add() {
        let mut dag = Dag::new();
        let x = dag.var_x1();
        let y = dag.var_x2();
        let s = dag.add(x, y);
        let (instrs, out_regs) = linearize(dag.nodes(), &[s]);
        assert_eq!(instrs.len(), 1);
        assert_eq!(instrs[0].op, Op::Add);
        assert_eq!(instrs[0].src[0], 0); // x1
        assert_eq!(instrs[0].src[1], 1); // x2
        assert_eq!(out_regs, &[instrs[0].dst]);
    }

    #[test]
    fn linearize_nested() {
        // 2 * x1^2
        let mut dag = Dag::new();
        let x = dag.var_x1();
        let two = dag.constant(2.0);
        let x_sq = dag.mul(x, x);
        let result = dag.mul(two, x_sq);
        let (instrs, out_regs) = linearize(dag.nodes(), &[result]);
        // instructions: constant(2.0), mul(x1,x1), mul(2, x1^2)
        assert_eq!(instrs.len(), 3);
        // verify dependency order: constant and x*x before final mul
        let dst_result = out_regs[0];
        let last = &instrs[instrs.len() - 1];
        assert_eq!(last.dst, dst_result);
        assert_eq!(last.op, Op::Mul);
    }

    #[test]
    fn linearize_multi_output() {
        // two outputs from shared subexpression
        let mut dag = Dag::new();
        let x = dag.var_x1();
        let y = dag.var_x2();
        let a = dag.add(x, y);
        let b = dag.mul(a, a);
        let one = dag.constant(1.0);
        let c = dag.add(a, one);
        let (instrs, out_regs) = linearize(dag.nodes(), &[b, c]);
        assert_eq!(out_regs.len(), 2);
        // 'a' should be computed once (shared node), then b and c reference its register
        // instructions: add(x,y), mul(a,a), constant(1), add(a,1)
        assert_eq!(instrs.len(), 4);
    }

    #[test]
    fn linearize_if_then_else() {
        let mut dag = Dag::new();
        let x = dag.var_x1();
        let zero = dag.constant(0.0);
        let cond = dag.gt(x, zero);
        let neg_x = dag.neg(x);
        let result = dag.if_then_else(cond, x, neg_x);
        let (instrs, out_regs) = linearize(dag.nodes(), &[result]);
        assert_eq!(out_regs.len(), 1);
        // find the ITE instruction
        let ite = instrs.iter().find(|ii| ii.op == Op::IfThenElse).unwrap();
        assert_eq!(ite.dst, out_regs[0]);
    }

    #[test]
    fn linearize_parameter() {
        let mut dag = Dag::new();
        let x = dag.var_x1();
        let p = dag.param(0);
        let r = dag.mul(p, x);
        let (instrs, _) = linearize(dag.nodes(), &[r]);
        let param_instr = instrs.iter().find(|ii| ii.op == Op::Parameter).unwrap();
        assert_eq!(param_instr.imm_idx, 0);
    }

    #[test]
    fn reachability_prunes() {
        // build a DAG with an unreachable node
        let mut dag = Dag::new();
        let x = dag.var_x1();
        let _dead = dag.constant(999.0); // not referenced by output
        let two = dag.constant(2.0);
        let result = dag.mul(two, x);
        let (instrs, _) = linearize(dag.nodes(), &[result]);
        // should not include the dead constant
        assert!(instrs.iter().all(|ii| ii.imm_f64 != 999.0));
    }

    /// a wide sum is the shape configs actually build (a mode sum, a multi-term
    /// source), and its NODE count is unbounded while only a couple of values are
    /// live at once. without register recycling the allocator hands out one register
    /// per node and the stream indexes past the bank — which panicked as an opaque
    /// out-of-bounds rather than reporting an expression too large to represent.
    #[test]
    fn a_wide_sum_far_exceeds_the_register_bank() {
        let terms = 2000;
        let mut dag = Dag::new();
        let x = dag.var_x1();
        let mut acc = dag.constant(0.0);
        for ii in 0..terms {
            let c = dag.constant(ii as f64);
            let term = dag.mul(c, x);
            acc = dag.add(acc, term);
        }
        let (instrs, out_regs) = linearize(dag.nodes(), &[acc]);
        assert!(
            instrs.len() > 3 * terms - 10,
            "the stream dropped instructions: {} for {terms} terms",
            instrs.len()
        );
        // live values stay O(1) regardless of size, so the bank is never approached.
        assert!(
            max_register(&instrs) < 32,
            "a wide sum held {} registers live; recycling is not happening",
            max_register(&instrs)
        );
        // and the value is right: sum_i (i * x) = x * terms (terms - 1) / 2.
        let mut outputs = [0.0f64];
        crate::eval::evaluate(&instrs, &out_regs, 2.0, 0.0, 0.0, 0.0, &[], &mut outputs);
        let expected = 2.0 * (terms as f64) * (terms as f64 - 1.0) / 2.0;
        assert!(
            (outputs[0] - expected).abs() < 1.0e-6,
            "wide sum evaluated to {} against {expected}",
            outputs[0]
        );
    }

    /// recycling must never hand an instruction's destination a register still
    /// holding one of its own sources. a deep chain where every value feeds the next
    /// is where that would show up as a silently wrong result rather than a crash.
    #[test]
    fn a_deep_chain_reuses_registers_without_clobbering_its_own_inputs() {
        let depth = 500;
        let mut dag = Dag::new();
        let mut acc = dag.var_x1();
        for _ in 0..depth {
            let one = dag.constant(1.0);
            acc = dag.add(acc, one);
        }
        let (instrs, out_regs) = linearize(dag.nodes(), &[acc]);
        let mut outputs = [0.0f64];
        crate::eval::evaluate(&instrs, &out_regs, 7.0, 0.0, 0.0, 0.0, &[], &mut outputs);
        assert_eq!(outputs[0], 7.0 + depth as f64);
    }

    /// a value read by several later instructions has to survive until the LAST of
    /// them; freeing at the first read would corrupt every consumer after it.
    #[test]
    fn a_shared_subexpression_survives_until_its_final_read() {
        let mut dag = Dag::new();
        let x = dag.var_x1();
        let shared = dag.sin(x);
        // pad the stream so the shared value would be recycled early if its last use
        // were mistaken for its first.
        let mut filler = dag.constant(0.0);
        for ii in 0..300 {
            let c = dag.constant(ii as f64);
            filler = dag.add(filler, c);
        }
        let late = dag.mul(shared, filler);
        let result = dag.add(shared, late);
        let (instrs, out_regs) = linearize(dag.nodes(), &[result]);
        let mut outputs = [0.0f64];
        crate::eval::evaluate(&instrs, &out_regs, 0.5, 0.0, 0.0, 0.0, &[], &mut outputs);
        let s = 0.5f64.sin();
        let expected = s + s * (0..300).map(|ii| ii as f64).sum::<f64>();
        assert!(
            (outputs[0] - expected).abs() < 1.0e-9,
            "shared subexpression evaluated to {} against {expected}",
            outputs[0]
        );
    }

    /// a vector field whose components share per-term temporaries — the shape of any
    /// mode sum — is where the EVALUATION ORDER decides whether the expression is
    /// representable at all. finishing one output before starting the next holds every
    /// shared temporary live across the whole first component, giving pressure that
    /// grows with the term count; building in index order retires each term's
    /// temporaries as soon as all three components have consumed them.
    #[test]
    fn multi_output_shared_terms_keep_pressure_bounded() {
        let terms = 400;
        let mut dag = Dag::new();
        let x = dag.var_x1();
        let mut acc = [dag.constant(0.0), dag.constant(0.0), dag.constant(0.0)];
        for ii in 0..terms {
            // one shared temporary per term, read by ALL THREE outputs
            let c = dag.constant(ii as f64);
            let scaled = dag.mul(c, x);
            let shared = dag.sin(scaled);
            for (ax, slot) in acc.iter_mut().enumerate() {
                let w = dag.constant((ax + 1) as f64);
                let term = dag.mul(w, shared);
                *slot = dag.add(*slot, term);
            }
        }
        let (instrs, out_regs) = linearize(dag.nodes(), &acc);
        assert!(
            max_register(&instrs) < 64,
            "a {terms}-term 3-component field held {} registers live; the schedule is \
             keeping shared temporaries alive across outputs",
            max_register(&instrs)
        );
        let mut outputs = [0.0f64; 3];
        crate::eval::evaluate(&instrs, &out_regs, 0.25, 0.0, 0.0, 0.0, &[], &mut outputs);
        let base: f64 = (0..terms).map(|ii| (ii as f64 * 0.25).sin()).sum();
        for (ax, got) in outputs.iter().enumerate() {
            let expected = (ax as f64 + 1.0) * base;
            assert!(
                (got - expected).abs() < 1.0e-9,
                "component {ax} evaluated to {got} against {expected}"
            );
        }
    }

    #[test]
    fn max_register_simple() {
        let mut dag = Dag::new();
        let x = dag.var_x1();
        let c = dag.constant(3.0);
        let r = dag.add(x, c);
        let (instrs, _) = linearize(dag.nodes(), &[r]);
        let max_reg = max_register(&instrs);
        // constant gets reg 4, add gets reg 5
        assert_eq!(max_reg, 5);
    }
}
