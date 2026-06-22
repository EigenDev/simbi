// =============================================================================
// dag.rs
//
// expression DAG: nodes, payloads, and a builder API. nodes reference children
// by index into a flat array. the Dag builder provides convenience methods
// for constructing expressions programmatically.
//
// usage:
//   let mut dag = Dag::new();
//   let x = dag.var_x1();
//   let two = dag.constant(2.0);
//   let result = dag.mul(two, x);
//   let expr = dag.compile(&[result]);
// =============================================================================

use crate::op::Op;

/// payload of a DAG node. discriminated by Op.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Payload {
    /// no data (variables).
    None,
    /// constant literal value.
    Value(f64),
    /// parameter index into runtime parameter array.
    ParamIdx(usize),
    /// unary op: single child index.
    Unary(usize),
    /// binary op: (left, right) child indices.
    Binary(usize, usize),
    /// ternary op: (condition, then, else) child indices.
    Ternary(usize, usize, usize),
}

/// a single node in the expression DAG.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Node {
    pub op: Op,
    pub payload: Payload,
}

impl Node {
    /// visit child indices referenced by this node.
    pub fn for_each_child(&self, mut f: impl FnMut(usize)) {
        match &self.payload {
            Payload::None | Payload::Value(_) | Payload::ParamIdx(_) => {}
            Payload::Unary(c) => f(*c),
            Payload::Binary(l, r) => { f(*l); f(*r); }
            Payload::Ternary(a, b, c) => { f(*a); f(*b); f(*c); }
        }
    }
}

/// expression DAG builder. nodes are stored in a flat vec, referenced by index.
pub struct Dag {
    nodes: Vec<Node>,
}

impl Dag {
    pub fn new() -> Self {
        Dag { nodes: Vec::new() }
    }

    /// access the underlying node array.
    pub fn nodes(&self) -> &[Node] {
        &self.nodes
    }

    /// consume the dag and return the node array.
    pub fn into_nodes(self) -> Vec<Node> {
        self.nodes
    }

    fn push(&mut self, node: Node) -> usize {
        let idx = self.nodes.len();
        self.nodes.push(node);
        idx
    }

    // ---- leaf constructors ----

    pub fn constant(&mut self, val: f64) -> usize {
        self.push(Node { op: Op::Constant, payload: Payload::Value(val) })
    }

    pub fn var_x1(&mut self) -> usize {
        self.push(Node { op: Op::VariableX1, payload: Payload::None })
    }

    pub fn var_x2(&mut self) -> usize {
        self.push(Node { op: Op::VariableX2, payload: Payload::None })
    }

    pub fn var_x3(&mut self) -> usize {
        self.push(Node { op: Op::VariableX3, payload: Payload::None })
    }

    pub fn var_t(&mut self) -> usize {
        self.push(Node { op: Op::VariableT, payload: Payload::None })
    }

    pub fn param(&mut self, idx: usize) -> usize {
        self.push(Node { op: Op::Parameter, payload: Payload::ParamIdx(idx) })
    }

    // ---- generic constructors ----

    pub fn unary(&mut self, op: Op, child: usize) -> usize {
        debug_assert_eq!(op.arity(), 1, "{:?} is not unary", op);
        self.push(Node { op, payload: Payload::Unary(child) })
    }

    pub fn binary(&mut self, op: Op, left: usize, right: usize) -> usize {
        debug_assert_eq!(op.arity(), 2, "{:?} is not binary", op);
        self.push(Node { op, payload: Payload::Binary(left, right) })
    }

    pub fn if_then_else(&mut self, cond: usize, then_: usize, else_: usize) -> usize {
        self.push(Node { op: Op::IfThenElse, payload: Payload::Ternary(cond, then_, else_) })
    }

    // ---- arithmetic convenience ----

    pub fn add(&mut self, a: usize, b: usize) -> usize { self.binary(Op::Add, a, b) }
    pub fn sub(&mut self, a: usize, b: usize) -> usize { self.binary(Op::Sub, a, b) }
    pub fn mul(&mut self, a: usize, b: usize) -> usize { self.binary(Op::Mul, a, b) }
    pub fn div(&mut self, a: usize, b: usize) -> usize { self.binary(Op::Div, a, b) }
    pub fn pow(&mut self, a: usize, b: usize) -> usize { self.binary(Op::Pow, a, b) }
    pub fn neg(&mut self, a: usize) -> usize { self.unary(Op::Neg, a) }
    pub fn abs(&mut self, a: usize) -> usize { self.unary(Op::Abs, a) }
    pub fn sqrt(&mut self, a: usize) -> usize { self.unary(Op::Sqrt, a) }
    pub fn sin(&mut self, a: usize) -> usize { self.unary(Op::Sin, a) }
    pub fn cos(&mut self, a: usize) -> usize { self.unary(Op::Cos, a) }
    pub fn exp(&mut self, a: usize) -> usize { self.unary(Op::Exp, a) }
    pub fn log(&mut self, a: usize) -> usize { self.unary(Op::Log, a) }
    pub fn min(&mut self, a: usize, b: usize) -> usize { self.binary(Op::Min, a, b) }
    pub fn max(&mut self, a: usize, b: usize) -> usize { self.binary(Op::Max, a, b) }

    // ---- comparison convenience ----

    pub fn lt(&mut self, a: usize, b: usize) -> usize { self.binary(Op::Lt, a, b) }
    pub fn gt(&mut self, a: usize, b: usize) -> usize { self.binary(Op::Gt, a, b) }
    pub fn le(&mut self, a: usize, b: usize) -> usize { self.binary(Op::Le, a, b) }
    pub fn ge(&mut self, a: usize, b: usize) -> usize { self.binary(Op::Ge, a, b) }

    /// compile this DAG into a ready-to-evaluate Expression.
    pub fn compile(self, outputs: &[usize]) -> crate::eval::Expression {
        crate::eval::Expression::from_dag(self, outputs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dag_constant() {
        let mut dag = Dag::new();
        let c = dag.constant(42.0);
        assert_eq!(c, 0);
        assert_eq!(dag.nodes()[0].op, Op::Constant);
        assert_eq!(dag.nodes()[0].payload, Payload::Value(42.0));
    }

    #[test]
    fn dag_variables() {
        let mut dag = Dag::new();
        let x1 = dag.var_x1();
        let x2 = dag.var_x2();
        let x3 = dag.var_x3();
        let t = dag.var_t();
        assert_eq!(x1, 0);
        assert_eq!(x2, 1);
        assert_eq!(x3, 2);
        assert_eq!(t, 3);
    }

    #[test]
    fn dag_arithmetic() {
        let mut dag = Dag::new();
        let x = dag.var_x1();
        let two = dag.constant(2.0);
        let r = dag.mul(two, x);
        assert_eq!(r, 2);
        assert_eq!(dag.nodes()[2].op, Op::Mul);
        assert_eq!(dag.nodes()[2].payload, Payload::Binary(1, 0));
    }

    #[test]
    fn dag_children() {
        let mut dag = Dag::new();
        let x = dag.var_x1();
        let y = dag.var_x2();
        let s = dag.add(x, y);
        let _n = dag.neg(s);

        let mut ch = Vec::new();
        dag.nodes()[0].for_each_child(|c| ch.push(c));
        assert!(ch.is_empty());

        ch.clear();
        dag.nodes()[2].for_each_child(|c| ch.push(c));
        assert_eq!(ch, &[0, 1]);

        ch.clear();
        dag.nodes()[3].for_each_child(|c| ch.push(c));
        assert_eq!(ch, &[2]);
    }

    #[test]
    fn dag_if_then_else() {
        let mut dag = Dag::new();
        let x = dag.var_x1();
        let zero = dag.constant(0.0);
        let cond = dag.gt(x, zero);
        let neg_x = dag.neg(x);
        let result = dag.if_then_else(cond, x, neg_x);
        assert_eq!(dag.nodes()[result].op, Op::IfThenElse);
        assert_eq!(dag.nodes()[result].payload, Payload::Ternary(2, 0, 3));
    }

    #[test]
    fn dag_parameter() {
        let mut dag = Dag::new();
        let p = dag.param(0);
        assert_eq!(dag.nodes()[p].payload, Payload::ParamIdx(0));
    }
}
