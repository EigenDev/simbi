// =============================================================================
// eval.rs
//
// register-based VM evaluator for linearized expression instructions.
// 256-register bank, registers 0-3 are (x1, x2, x3, t), computed values
// start at register 4. sequential execution, no branching in the VM itself.
//
// the Expression type bundles compiled instructions + output register map
// and provides a clean evaluation API.
//
// usage:
//   let expr = dag.compile(&[root]);
//   let vals = expr.eval(1.0, 2.0, 3.0, 0.0);
// =============================================================================

use crate::dag::{Dag, Node};
use crate::linearize::{Instr, linearize};
use crate::op::Op;

pub(crate) const MAX_REGISTERS: usize = 256;

/// execute a linearized instruction stream on the register VM.
///
/// registers 0-3 are pre-loaded with (x1, x2, x3, t).
/// results are copied from output_regs into the outputs slice.
pub fn evaluate(
    instrs: &[Instr],
    output_regs: &[usize],
    x1: f64,
    x2: f64,
    x3: f64,
    t: f64,
    params: &[f64],
    outputs: &mut [f64],
) {
    let mut regs = [0.0f64; MAX_REGISTERS];
    regs[0] = x1;
    regs[1] = x2;
    regs[2] = x3;
    regs[3] = t;

    for instr in instrs {
        let dd = instr.dst;
        let s0 = regs[instr.src[0]];
        let s1 = regs[instr.src[1]];
        let s2 = regs[instr.src[2]];

        regs[dd] = match instr.op {
            // leaf
            Op::Constant => instr.imm_f64,
            Op::Parameter => params[instr.imm_idx],
            Op::VariableX1 | Op::VariableX2 | Op::VariableX3 | Op::VariableT => {
                unreachable!("variable ops should not appear in instruction stream")
            }
            // per-cell state variables are resolved by the source bridge into field reads, never
            // linearized into this coordinate VM (which has no fluid-state registers).
            Op::VariableRho
            | Op::VariableVel1
            | Op::VariableVel2
            | Op::VariableVel3
            | Op::VariablePressure
            | Op::VariableCellVolume => {
                unreachable!("state-variable ops are bridge-only; not valid in the coordinate VM")
            }

            // arithmetic
            Op::Add => s0 + s1,
            Op::Sub => s0 - s1,
            Op::Mul => s0 * s1,
            Op::Div => s0 / s1,
            Op::Pow => s0.powf(s1),
            Op::Neg => -s0,

            // comparison (1.0 = true, 0.0 = false)
            Op::Lt => {
                if s0 < s1 {
                    1.0
                } else {
                    0.0
                }
            }
            Op::Gt => {
                if s0 > s1 {
                    1.0
                } else {
                    0.0
                }
            }
            Op::Eq => {
                if (s0 - s1).abs() < 1e-14 {
                    1.0
                } else {
                    0.0
                }
            }
            Op::Le => {
                if s0 <= s1 {
                    1.0
                } else {
                    0.0
                }
            }
            Op::Ge => {
                if s0 >= s1 {
                    1.0
                } else {
                    0.0
                }
            }

            // logical (nonzero = true)
            Op::And => {
                if s0 != 0.0 && s1 != 0.0 {
                    1.0
                } else {
                    0.0
                }
            }
            Op::Or => {
                if s0 != 0.0 || s1 != 0.0 {
                    1.0
                } else {
                    0.0
                }
            }
            Op::Not => {
                if s0 == 0.0 {
                    1.0
                } else {
                    0.0
                }
            }

            // math (unary)
            Op::Log => s0.ln(),
            Op::Log10 => s0.log10(),
            Op::Abs => s0.abs(),
            Op::Sin => s0.sin(),
            Op::Cos => s0.cos(),
            Op::Tan => s0.tan(),
            Op::Asin => s0.asin(),
            Op::Acos => s0.acos(),
            Op::Atan => s0.atan(),
            Op::Exp => s0.exp(),
            Op::Sqrt => s0.sqrt(),
            Op::Sinh => s0.sinh(),
            Op::Cosh => s0.cosh(),
            Op::Tanh => s0.tanh(),
            Op::Asinh => s0.asinh(),
            Op::Acosh => s0.acosh(),
            Op::Atanh => s0.atanh(),
            Op::Sgn => {
                if s0 > 0.0 {
                    1.0
                } else if s0 < 0.0 {
                    -1.0
                } else {
                    0.0
                }
            }
            Op::Ceil => s0.ceil(),
            Op::Floor => s0.floor(),

            // math (binary)
            Op::Min => s0.min(s1),
            Op::Max => s0.max(s1),
            Op::Mod => s0 % s1,
            Op::Atan2 => s0.atan2(s1),

            // ternary
            Op::IfThenElse => {
                if s0 != 0.0 {
                    s1
                } else {
                    s2
                }
            }
        };
    }

    for (ii, &reg) in output_regs.iter().enumerate() {
        outputs[ii] = regs[reg];
    }
}

/// a compiled, ready-to-evaluate expression.
#[derive(Clone, Debug)]
pub struct Expression {
    instrs: Vec<Instr>,
    output_regs: Vec<usize>,
    params: Vec<f64>,
}

impl Expression {
    /// compile a DAG into an expression. consumes the Dag.
    pub fn from_dag(dag: Dag, outputs: &[usize]) -> Self {
        Self::from_nodes(dag.nodes(), outputs)
    }

    /// compile from a raw node slice (used by config loading).
    pub fn from_nodes(nodes: &[Node], outputs: &[usize]) -> Self {
        let (instrs, output_regs) = linearize(nodes, outputs);
        Expression {
            instrs,
            output_regs,
            params: Vec::new(),
        }
    }

    /// set runtime parameters (referenced by Op::Parameter nodes).
    pub fn set_params(&mut self, params: &[f64]) {
        self.params = params.to_vec();
    }

    /// number of output values this expression produces.
    pub fn num_outputs(&self) -> usize {
        self.output_regs.len()
    }

    /// number of instructions in the compiled instruction stream.
    pub fn num_instructions(&self) -> usize {
        self.instrs.len()
    }

    /// evaluate at a point, returning all outputs as a Vec.
    pub fn eval(&self, x1: f64, x2: f64, x3: f64, t: f64) -> Vec<f64> {
        let mut out = vec![0.0; self.output_regs.len()];
        evaluate(
            &self.instrs,
            &self.output_regs,
            x1,
            x2,
            x3,
            t,
            &self.params,
            &mut out,
        );
        out
    }

    /// evaluate at a point, writing outputs into the provided slice.
    pub fn eval_into(&self, x1: f64, x2: f64, x3: f64, t: f64, out: &mut [f64]) {
        evaluate(
            &self.instrs,
            &self.output_regs,
            x1,
            x2,
            x3,
            t,
            &self.params,
            out,
        );
    }

    /// access the compiled instruction stream.
    pub fn instructions(&self) -> &[Instr] {
        &self.instrs
    }

    /// access the output register map.
    pub fn output_registers(&self) -> &[usize] {
        &self.output_regs
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dag::Dag;

    fn approx(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-12 * a.abs().max(b.abs()).max(1.0)
    }

    #[test]
    fn eval_constant() {
        let mut dag = Dag::new();
        let c = dag.constant(42.0);
        let expr = dag.compile(&[c]);
        let out = expr.eval(0.0, 0.0, 0.0, 0.0);
        assert!(approx(out[0], 42.0));
    }

    #[test]
    fn eval_variables() {
        let mut dag = Dag::new();
        let x1 = dag.var_x1();
        let x2 = dag.var_x2();
        let x3 = dag.var_x3();
        let t = dag.var_t();
        let expr = dag.compile(&[x1, x2, x3, t]);
        let out = expr.eval(1.0, 2.0, 3.0, 4.0);
        assert!(approx(out[0], 1.0));
        assert!(approx(out[1], 2.0));
        assert!(approx(out[2], 3.0));
        assert!(approx(out[3], 4.0));
    }

    #[test]
    fn eval_arithmetic() {
        // f(x) = 2*x + 3
        let mut dag = Dag::new();
        let x = dag.var_x1();
        let two = dag.constant(2.0);
        let three = dag.constant(3.0);
        let tx = dag.mul(two, x);
        let r = dag.add(tx, three);
        let expr = dag.compile(&[r]);
        assert!(approx(expr.eval(5.0, 0.0, 0.0, 0.0)[0], 13.0));
        assert!(approx(expr.eval(0.0, 0.0, 0.0, 0.0)[0], 3.0));
        assert!(approx(expr.eval(-1.0, 0.0, 0.0, 0.0)[0], 1.0));
    }

    #[test]
    fn eval_subtraction_division() {
        // f(x, y) = (x - y) / 2
        let mut dag = Dag::new();
        let x = dag.var_x1();
        let y = dag.var_x2();
        let two = dag.constant(2.0);
        let diff = dag.sub(x, y);
        let r = dag.div(diff, two);
        let expr = dag.compile(&[r]);
        assert!(approx(expr.eval(10.0, 4.0, 0.0, 0.0)[0], 3.0));
    }

    #[test]
    fn eval_power() {
        // f(x) = x^3
        let mut dag = Dag::new();
        let x = dag.var_x1();
        let three = dag.constant(3.0);
        let r = dag.pow(x, three);
        let expr = dag.compile(&[r]);
        assert!(approx(expr.eval(2.0, 0.0, 0.0, 0.0)[0], 8.0));
        assert!(approx(expr.eval(3.0, 0.0, 0.0, 0.0)[0], 27.0));
    }

    #[test]
    fn eval_neg() {
        let mut dag = Dag::new();
        let x = dag.var_x1();
        let r = dag.neg(x);
        let expr = dag.compile(&[r]);
        assert!(approx(expr.eval(7.0, 0.0, 0.0, 0.0)[0], -7.0));
    }

    #[test]
    fn eval_trig() {
        let mut dag = Dag::new();
        let x = dag.var_x1();
        let s = dag.sin(x);
        let c = dag.cos(x);
        let expr = dag.compile(&[s, c]);

        let pi_half = std::f64::consts::FRAC_PI_2;
        let out = expr.eval(pi_half, 0.0, 0.0, 0.0);
        assert!(approx(out[0], 1.0)); // sin(pi/2)
        assert!(approx(out[1], 0.0)); // cos(pi/2)
    }

    #[test]
    fn eval_exp_log() {
        let mut dag = Dag::new();
        let x = dag.var_x1();
        let e = dag.exp(x);
        let l = dag.log(e);
        let expr = dag.compile(&[l]);
        // log(exp(x)) = x
        assert!(approx(expr.eval(3.0, 0.0, 0.0, 0.0)[0], 3.0));
    }

    #[test]
    fn eval_sqrt_abs() {
        let mut dag = Dag::new();
        let x = dag.var_x1();
        let a = dag.abs(x);
        let s = dag.sqrt(a);
        let expr = dag.compile(&[s]);
        assert!(approx(expr.eval(-9.0, 0.0, 0.0, 0.0)[0], 3.0));
    }

    #[test]
    fn eval_comparison() {
        // f(x) = x > 0
        let mut dag = Dag::new();
        let x = dag.var_x1();
        let zero = dag.constant(0.0);
        let r = dag.gt(x, zero);
        let expr = dag.compile(&[r]);
        assert!(approx(expr.eval(1.0, 0.0, 0.0, 0.0)[0], 1.0));
        assert!(approx(expr.eval(-1.0, 0.0, 0.0, 0.0)[0], 0.0));
    }

    #[test]
    fn eval_if_then_else() {
        // f(x) = if x > 0 then x else -x  (manual abs)
        let mut dag = Dag::new();
        let x = dag.var_x1();
        let zero = dag.constant(0.0);
        let cond = dag.gt(x, zero);
        let neg_x = dag.neg(x);
        let r = dag.if_then_else(cond, x, neg_x);
        let expr = dag.compile(&[r]);
        assert!(approx(expr.eval(5.0, 0.0, 0.0, 0.0)[0], 5.0));
        assert!(approx(expr.eval(-3.0, 0.0, 0.0, 0.0)[0], 3.0));
    }

    #[test]
    fn eval_parameters() {
        // f(x) = p0 * x + p1
        let mut dag = Dag::new();
        let x = dag.var_x1();
        let p0 = dag.param(0);
        let p1 = dag.param(1);
        let px = dag.mul(p0, x);
        let r = dag.add(px, p1);
        let mut expr = dag.compile(&[r]);

        expr.set_params(&[2.0, 5.0]);
        assert!(approx(expr.eval(3.0, 0.0, 0.0, 0.0)[0], 11.0)); // 2*3 + 5

        expr.set_params(&[-1.0, 10.0]);
        assert!(approx(expr.eval(3.0, 0.0, 0.0, 0.0)[0], 7.0)); // -1*3 + 10
    }

    #[test]
    fn eval_multi_output() {
        // f(x, y) = (x + y, x * y)
        let mut dag = Dag::new();
        let x = dag.var_x1();
        let y = dag.var_x2();
        let s = dag.add(x, y);
        let p = dag.mul(x, y);
        let expr = dag.compile(&[s, p]);
        let out = expr.eval(3.0, 4.0, 0.0, 0.0);
        assert!(approx(out[0], 7.0));
        assert!(approx(out[1], 12.0));
    }

    #[test]
    fn eval_dag_sharing() {
        // a = x + y; outputs: (a * a, a + 1)
        // node 'a' is shared — should be computed once.
        let mut dag = Dag::new();
        let x = dag.var_x1();
        let y = dag.var_x2();
        let a = dag.add(x, y);
        let a_sq = dag.mul(a, a);
        let one = dag.constant(1.0);
        let a_plus_1 = dag.add(a, one);
        let expr = dag.compile(&[a_sq, a_plus_1]);
        let out = expr.eval(2.0, 3.0, 0.0, 0.0);
        // a = 5, a^2 = 25, a+1 = 6
        assert!(approx(out[0], 25.0));
        assert!(approx(out[1], 6.0));
    }

    #[test]
    fn eval_min_max() {
        let mut dag = Dag::new();
        let x = dag.var_x1();
        let y = dag.var_x2();
        let mn = dag.min(x, y);
        let mx = dag.max(x, y);
        let expr = dag.compile(&[mn, mx]);
        let out = expr.eval(3.0, 7.0, 0.0, 0.0);
        assert!(approx(out[0], 3.0));
        assert!(approx(out[1], 7.0));
    }

    #[test]
    fn eval_sgn() {
        let mut dag = Dag::new();
        let x = dag.var_x1();
        let s = dag.unary(Op::Sgn, x);
        let expr = dag.compile(&[s]);
        assert!(approx(expr.eval(5.0, 0.0, 0.0, 0.0)[0], 1.0));
        assert!(approx(expr.eval(-3.0, 0.0, 0.0, 0.0)[0], -1.0));
        assert!(approx(expr.eval(0.0, 0.0, 0.0, 0.0)[0], 0.0));
    }

    #[test]
    fn eval_floor_ceil() {
        let mut dag = Dag::new();
        let x = dag.var_x1();
        let fl = dag.unary(Op::Floor, x);
        let ce = dag.unary(Op::Ceil, x);
        let expr = dag.compile(&[fl, ce]);
        let out = expr.eval(2.7, 0.0, 0.0, 0.0);
        assert!(approx(out[0], 2.0));
        assert!(approx(out[1], 3.0));
    }

    #[test]
    fn eval_logical() {
        let mut dag = Dag::new();
        let x = dag.var_x1();
        let y = dag.var_x2();
        let zero = dag.constant(0.0);
        let xgt = dag.gt(x, zero);
        let ygt = dag.gt(y, zero);
        let a = dag.binary(Op::And, xgt, ygt);
        let o = dag.binary(Op::Or, xgt, ygt);
        let expr = dag.compile(&[a, o]);

        // x=1, y=1: both positive
        let out = expr.eval(1.0, 1.0, 0.0, 0.0);
        assert!(approx(out[0], 1.0));
        assert!(approx(out[1], 1.0));

        // x=1, y=-1: only x positive
        let out = expr.eval(1.0, -1.0, 0.0, 0.0);
        assert!(approx(out[0], 0.0));
        assert!(approx(out[1], 1.0));
    }

    #[test]
    fn eval_sod_initial_condition() {
        // sod tube: if x < 0.5 then 1.0 else 0.125
        let mut dag = Dag::new();
        let x = dag.var_x1();
        let half = dag.constant(0.5);
        let cond = dag.lt(x, half);
        let rho_l = dag.constant(1.0);
        let rho_r = dag.constant(0.125);
        let rho = dag.if_then_else(cond, rho_l, rho_r);
        let expr = dag.compile(&[rho]);
        assert!(approx(expr.eval(0.1, 0.0, 0.0, 0.0)[0], 1.0));
        assert!(approx(expr.eval(0.9, 0.0, 0.0, 0.0)[0], 0.125));
    }

    #[test]
    fn eval_time_dependent() {
        // f(x, t) = sin(x - t)  (traveling wave)
        let mut dag = Dag::new();
        let x = dag.var_x1();
        let t = dag.var_t();
        let diff = dag.sub(x, t);
        let s = dag.sin(diff);
        let expr = dag.compile(&[s]);
        // at t=0, f(pi/2) = sin(pi/2) = 1
        let pi_half = std::f64::consts::FRAC_PI_2;
        assert!(approx(expr.eval(pi_half, 0.0, 0.0, 0.0)[0], 1.0));
        // at t=pi/2, f(pi) = sin(pi - pi/2) = sin(pi/2) = 1
        assert!(approx(
            expr.eval(std::f64::consts::PI, 0.0, 0.0, pi_half)[0],
            1.0
        ));
    }

    #[test]
    fn eval_complex_gravity() {
        // point-mass gravity: g_x = -G * M * x / r^3
        // where r = sqrt(x^2 + y^2), G*M = 1.0
        let mut dag = Dag::new();
        let x = dag.var_x1();
        let y = dag.var_x2();
        let x2 = dag.mul(x, x);
        let y2 = dag.mul(y, y);
        let r2 = dag.add(x2, y2);
        let r = dag.sqrt(r2);
        let r_sq = dag.mul(r, r);
        let r3 = dag.mul(r, r_sq);
        let x_over_r3 = dag.div(x, r3);
        let gx = dag.neg(x_over_r3);
        let y_over_r3 = dag.div(y, r3);
        let gy = dag.neg(y_over_r3);
        let expr = dag.compile(&[gx, gy]);

        // at (1, 0): g_x = -1, g_y = 0
        let out = expr.eval(1.0, 0.0, 0.0, 0.0);
        assert!(approx(out[0], -1.0));
        assert!(approx(out[1], 0.0));

        // at (0, 2): r=2, r^3=8, g_x=0, g_y = -2/8 = -0.25
        let out = expr.eval(0.0, 2.0, 0.0, 0.0);
        assert!(approx(out[0], 0.0));
        assert!(approx(out[1], -0.25));
    }

    #[test]
    fn eval_into_slice() {
        let mut dag = Dag::new();
        let x = dag.var_x1();
        let r = dag.mul(x, x);
        let expr = dag.compile(&[r]);
        let mut out = [0.0; 1];
        expr.eval_into(3.0, 0.0, 0.0, 0.0, &mut out);
        assert!(approx(out[0], 9.0));
    }

    #[test]
    fn num_outputs_and_instructions() {
        let mut dag = Dag::new();
        let x = dag.var_x1();
        let y = dag.var_x2();
        let s = dag.add(x, y);
        let p = dag.mul(x, y);
        let expr = dag.compile(&[s, p]);
        assert_eq!(expr.num_outputs(), 2);
        assert_eq!(expr.num_instructions(), 2); // just add and mul
    }
}
