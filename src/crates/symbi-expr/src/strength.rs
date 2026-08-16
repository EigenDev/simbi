// =============================================================================
// strength.rs
//
// constant-power strength reduction on the expression DAG: `pow(x, c)` for a
// small integer or half-integer constant `c` becomes a multiply / divide /
// sqrt chain. a user config writing `r ** (-2.0)` otherwise lowers to a libm
// `powf` call at every cell of every stage — ~50-100 cycles for what is
// algebraically two operations.
//
// runs on the loaded node array before either consumer (the f64 coordinate VM
// and the IR bridge), so both evaluate the same reduced form and the
// kernel-vs-reference oracle stays exact. the reduced chain is not bit-equal
// to libm `pow` (the exact product is at least as accurate), which is why the
// reduction must sit at this single seam and not in one consumer.
//
// usage:
//   let (nodes, outputs) = strength_reduce(nodes, &outputs);
// =============================================================================

use crate::dag::{Node, Payload};
use crate::op::Op;

// exponents with a cheap exact chain. anything else keeps `pow` (a runtime
// exponent, a large or non-representable power).
fn reducible(c: f64) -> bool {
    matches!(
        c,
        0.0 | 1.0 | 2.0 | 3.0 | 4.0 | -1.0 | -2.0 | -3.0 | 0.5 | -0.5
    )
}

/// rewrite reducible constant powers; returns the new node array and the
/// remapped output indices. non-pow nodes survive in order (children stay
/// before parents, the layout the evaluators require).
pub fn strength_reduce(nodes: &[Node], outputs: &[usize]) -> (Vec<Node>, Vec<usize>) {
    let mut out: Vec<Node> = Vec::with_capacity(nodes.len());
    let mut remap: Vec<usize> = Vec::with_capacity(nodes.len());

    for node in nodes {
        // remap children through earlier rewrites.
        let mapped = {
            let mut n = *node;
            n.payload = match node.payload {
                Payload::Unary(a) => Payload::Unary(remap[a]),
                Payload::Binary(a, b) => Payload::Binary(remap[a], remap[b]),
                Payload::Ternary(a, b, c) => Payload::Ternary(remap[a], remap[b], remap[c]),
                p => p,
            };
            n
        };
        let reduced = match mapped.payload {
            Payload::Binary(base, expo)
                if mapped.op == Op::Pow
                    && matches!(out[expo], Node { op: Op::Constant, payload: Payload::Value(c) } if reducible(c)) =>
            {
                let Node {
                    payload: Payload::Value(c),
                    ..
                } = out[expo]
                else {
                    unreachable!()
                };
                Some(emit_chain(&mut out, base, c))
            }
            _ => None,
        };
        match reduced {
            Some(idx) => remap.push(idx),
            None => {
                out.push(mapped);
                remap.push(out.len() - 1);
            }
        }
    }
    let outputs = outputs.iter().map(|&o| remap[o]).collect();
    (out, outputs)
}

fn push(out: &mut Vec<Node>, op: Op, payload: Payload) -> usize {
    out.push(Node { op, payload });
    out.len() - 1
}
fn mul(out: &mut Vec<Node>, a: usize, b: usize) -> usize {
    push(out, Op::Mul, Payload::Binary(a, b))
}
fn one(out: &mut Vec<Node>) -> usize {
    push(out, Op::Constant, Payload::Value(1.0))
}
fn div(out: &mut Vec<Node>, a: usize, b: usize) -> usize {
    push(out, Op::Div, Payload::Binary(a, b))
}

fn emit_chain(out: &mut Vec<Node>, base: usize, c: f64) -> usize {
    match c {
        // x^0 = 1 for every x including nan (ieee pow(x, 0) = 1).
        0.0 => one(out),
        1.0 => base,
        2.0 => mul(out, base, base),
        3.0 => {
            let sq = mul(out, base, base);
            mul(out, sq, base)
        }
        4.0 => {
            let sq = mul(out, base, base);
            mul(out, sq, sq)
        }
        -1.0 => {
            let unit = one(out);
            div(out, unit, base)
        }
        -2.0 => {
            let sq = mul(out, base, base);
            let unit = one(out);
            div(out, unit, sq)
        }
        -3.0 => {
            let sq = mul(out, base, base);
            let cb = mul(out, sq, base);
            let unit = one(out);
            div(out, unit, cb)
        }
        0.5 => push(out, Op::Sqrt, Payload::Unary(base)),
        -0.5 => {
            let rt = push(out, Op::Sqrt, Payload::Unary(base));
            let unit = one(out);
            div(out, unit, rt)
        }
        _ => unreachable!("reducible() admitted {c}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dag::Dag;
    use crate::eval::Expression;

    fn eval1(nodes: &[Node], outputs: &[usize], x: f64) -> f64 {
        let expr = Expression::from_nodes(nodes, outputs);
        expr.eval(x, 0.0, 0.0, 0.0)[0]
    }

    fn build_pow(c: f64) -> (Vec<Node>, Vec<usize>) {
        let mut d = Dag::new();
        let x = d.var_x1();
        let e = d.constant(c);
        let p = d.pow(x, e);
        (d.into_nodes(), vec![p])
    }

    #[test]
    fn reducible_powers_lose_their_pow_node() {
        for c in [0.0, 1.0, 2.0, 3.0, 4.0, -1.0, -2.0, -3.0, 0.5, -0.5] {
            let (nodes, outs) = build_pow(c);
            let (red, routs) = strength_reduce(&nodes, &outs);
            assert!(
                !red.iter().any(|n| n.op == Op::Pow),
                "pow survived for c = {c}"
            );
            // the chain equals the exact algebraic form (and matches libm pow
            // to a few ulps — the chain is the more accurate of the two).
            for x in [0.37_f64, 1.0, 2.5, 117.3] {
                let got = eval1(&red, &routs, x);
                let want = x.powf(c);
                let tol = 4.0 * f64::EPSILON * want.abs().max(1.0);
                assert!(
                    (got - want).abs() <= tol,
                    "c={c} x={x}: chain {got} vs powf {want}"
                );
            }
        }
    }

    #[test]
    fn runtime_and_irreducible_exponents_keep_pow() {
        // non-constant exponent
        let mut d = Dag::new();
        let x = d.var_x1();
        let y = d.var_x2();
        let p = d.pow(x, y);
        let (red, _) = strength_reduce(&d.into_nodes(), &[p]);
        assert!(red.iter().any(|n| n.op == Op::Pow));
        // irreducible constant
        let (nodes, outs) = build_pow(2.7);
        let (red, routs) = strength_reduce(&nodes, &outs);
        assert!(red.iter().any(|n| n.op == Op::Pow));
        assert!((eval1(&red, &routs, 1.9) - 1.9_f64.powf(2.7)).abs() < 1e-12);
    }

    #[test]
    fn shared_subgraphs_and_outputs_remap() {
        // out0 = x^-2, out1 = x + x^-2 : the chain is shared, outputs remap.
        let mut d = Dag::new();
        let x = d.var_x1();
        let e = d.constant(-2.0);
        let p = d.pow(x, e);
        let s = d.add(x, p);
        let (red, routs) = strength_reduce(&d.into_nodes(), &[p, s]);
        assert!(!red.iter().any(|n| n.op == Op::Pow));
        let expr = Expression::from_nodes(&red, &routs);
        let v = expr.eval(2.0, 0.0, 0.0, 0.0);
        assert!((v[0] - 0.25).abs() < 1e-15);
        assert!((v[1] - 2.25).abs() < 1e-15);
    }
}
