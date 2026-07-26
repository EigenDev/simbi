// =============================================================================
// linearize.rs
//
// converts an expression DAG into a flat instruction stream for the register
// VM. pipeline: reachability analysis -> topological sort (DFS) -> register
// allocation -> instruction emission.
//
// variable registers are fixed: x1=0, x2=1, x3=2, t=3. computed nodes
// get sequential registers starting at 4.
//
// usage:
//   let (instrs, output_regs) = linearize(dag.nodes(), &[root_idx]);
// =============================================================================

use crate::dag::{Node, Payload};
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

/// topological sort via DFS. returns nodes in evaluation order (leaves first).
fn topo_sort(nodes: &[Node], outputs: &[usize], reachable: &[bool]) -> Vec<usize> {
    let nn = nodes.len();
    let mut color = vec![0u8; nn]; // 0=white, 1=gray, 2=black
    let mut result = Vec::with_capacity(nn);

    for &root in outputs {
        dfs_visit(nodes, root, reachable, &mut color, &mut result);
    }
    result
}

fn dfs_visit(
    nodes: &[Node],
    idx: usize,
    reachable: &[bool],
    color: &mut [u8],
    result: &mut Vec<usize>,
) {
    if !reachable[idx] || color[idx] == 2 {
        return;
    }
    assert!(
        color[idx] != 1,
        "cycle detected in expression DAG at node {}",
        idx
    );

    color[idx] = 1;

    // collect children first to avoid borrow conflict with recursive call.
    let mut children = [0usize; 3];
    let mut num_children = 0;
    nodes[idx].for_each_child(|c| {
        children[num_children] = c;
        num_children += 1;
    });
    for &child in &children[..num_children] {
        dfs_visit(nodes, child, reachable, color, result);
    }

    color[idx] = 2;
    result.push(idx);
}

/// linearize an expression DAG into a flat instruction stream.
///
/// returns (instructions, output_registers) where output_registers[ii] is
/// the register holding the value of outputs[ii] after evaluation.
pub fn linearize(nodes: &[Node], outputs: &[usize]) -> (Vec<Instr>, Vec<usize>) {
    let reachable = find_reachable(nodes, outputs);
    let order = topo_sort(nodes, outputs, &reachable);

    // register allocation: variables get fixed regs, others get sequential.
    let mut reg_map: Vec<Option<usize>> = vec![None; nodes.len()];
    let mut next_reg = FIRST_COMPUTED_REG;

    for &idx in &order {
        let node = &nodes[idx];
        if node.op.is_variable() {
            reg_map[idx] = Some(node.op.variable_register());
        } else {
            reg_map[idx] = Some(next_reg);
            next_reg += 1;
        }
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
