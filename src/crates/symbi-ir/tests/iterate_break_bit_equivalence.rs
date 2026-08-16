// =============================================================================
// iterate_break_bit_equivalence.rs
//
// the IterateInline `break_when` predicate is a perf shortcut, not a semantic
// change: running an iterate with a break-predicate produces bit-exact the same
// f64 result as running the same iterate without one, provided convergence fires
// before the iteration cap. the freeze pattern
// `step = select(conv, x, x_new)` already pins the converged value.
//
// physics: Newton iteration for sqrt(2), f(x) = x^2 - 2, f'(x) = 2x, step
// x_new = x - f(x)/f'(x). converges quadratically; the freeze predicate is
// |x_new - x| < 1e-10. 20 deterministic lcg initial guesses in [0.5, 10); for
// each:
//
//   ir-a:  Graph::iterate_inline_scalar(..., break_when = Some(conv))
//   ir-b:  Graph::iterate_inline_scalar(..., break_when = None)
//
//   same body, same predicate, same init, same max_steps. both lowered by
//   scalarize_kernel (which handles IterateInline's cone partitioning), both
//   run through the CPU interpreter. assert identical bits.
//
// the IR is built through the low-level Graph API (`with_trace |t| t.graph()...`)
// because `Gv::iterate` always passes
// Some(conv) — only the direct builder can flip the knob. lowering uses
// scalarize_kernel (not plain scalarize) because IterAcc only resolves inside
// the loop body context that scalarize_kernel sets up via lower_iterate_inline.
// =============================================================================

use symbi_ir::graph::{ConstValue, ElementWiseOp, NodeId};
use symbi_ir::passes::scalarize::{LoweredFn, scalarize_kernel};
use symbi_ir::{Backend, Cpu, Gv, begin_trace, end_trace, with_trace};

// build a Newton-for-sqrt(2) iterate via raw Graph ops, returning a LoweredFn
// the CPU interpreter can evaluate. `with_break = true` passes Some(conv);
// `false` passes None. all other IR is bit-identical so only the break knob
// is under test.
fn build_newton_sqrt2(with_break: bool, max_steps: usize) -> LoweredFn {
    begin_trace();

    let x0 = Gv::param("x0").node();

    // direct graph ops: the body is a Newton step rendered as raw NodeIds, so
    // the predicate's NodeId is in scope for both `select` and `break_when`.
    let (acc_id, two_id, eps_id) = with_trace(|t| {
        let g = t.graph();
        let acc = g.iter_acc(0, None);
        let two = g.add_const(ConstValue::F64(2.0), None);
        let eps = g.add_const(ConstValue::F64(1.0e-10), None);
        (acc, two, eps)
    });

    let x_new_id = with_trace(|t| {
        let g = t.graph();
        // body: x - (x*x - 2) / (x + x)
        let xx = g.element_wise(ElementWiseOp::Mul, vec![acc_id, acc_id], None);
        let f = g.element_wise(ElementWiseOp::Sub, vec![xx, two_id], None);
        let fp = g.element_wise(ElementWiseOp::Add, vec![acc_id, acc_id], None);
        let q = g.element_wise(ElementWiseOp::Div, vec![f, fp], None);
        g.element_wise(ElementWiseOp::Sub, vec![acc_id, q], None)
    });

    // freeze predicate: conv = |x_new - acc| < eps. one node, used in both
    // select(conv, acc, x_new) and break_when (Some/None).
    let conv_id = with_trace(|t| {
        let g = t.graph();
        let delta = g.element_wise(ElementWiseOp::Sub, vec![x_new_id, acc_id], None);
        let absd = g.element_wise(ElementWiseOp::Abs, vec![delta], None);
        g.element_wise(ElementWiseOp::Lt, vec![absd, eps_id], None)
    });

    // step = select(conv, acc, x_new) — the freeze law.
    let step_id = with_trace(|t| t.graph().select(conv_id, acc_id, x_new_id, None));

    let break_when: Option<NodeId> = if with_break { Some(conv_id) } else { None };
    let root = with_trace(|t| {
        t.graph()
            .iterate_inline_scalar(acc_id, x0, step_id, max_steps, break_when, None)
    });

    let kernel = end_trace();
    // scalarize_kernel handles IterateInline (cone-partitioned lowering into
    // a `for` with the body inside); plain `scalarize` doesn't, so the
    // kernel-scalarized output is lifted to a LoweredFn here.
    let ks = scalarize_kernel(&kernel.graph, &[root]);
    LoweredFn {
        name: "newton_sqrt2".to_string(),
        params: ks.params,
        body: ks.body,
        results: ks.outputs,
        result_element: symbi_ir::ElementTy::F64,
        result_shape: Vec::new(),
    }
}

#[test]
fn iterate_break_is_bit_equivalent_to_no_break() {
    let f_with_break = build_newton_sqrt2(true, 64);
    let f_no_break = build_newton_sqrt2(false, 64);

    // deterministic lcg over [0.5, 10.0]; both sides of sqrt2 at varying distances.
    // Newton converges in ~6-8 steps from this range, well inside max=64.
    let mut state: u64 = 0xA5;
    let mut mismatches: Vec<(f64, f64, f64)> = Vec::new();
    for i in 0..20 {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let u = ((state >> 33) as f64) / (1u64 << 31) as f64; // in [0,1)
        let x0_val = 0.5 + 9.5 * u;

        let a = Cpu.eval_elemental(&f_with_break, &[x0_val])[0];
        let b = Cpu.eval_elemental(&f_no_break, &[x0_val])[0];

        let sqrt2 = std::f64::consts::SQRT_2;
        assert!(
            (a - sqrt2).abs() < 1e-10,
            "case {i}: with-break newton did not converge from x0={x0_val}: got {a}, expected ~={sqrt2}",
        );
        assert!(
            (b - sqrt2).abs() < 1e-10,
            "case {i}: no-break newton did not converge from x0={x0_val}: got {b}, expected ~={sqrt2}",
        );

        if a.to_bits() != b.to_bits() {
            mismatches.push((x0_val, a, b));
        }
    }

    if !mismatches.is_empty() {
        let mut msg = String::from(
            "BREAK-EQUIVALENCE BROKEN: IterateInline with break_when=Some(conv) returned \
             DIFFERENT bits than with break_when=None on identical IR + input.\n",
        );
        for (x0, a, b) in &mismatches {
            msg.push_str(&format!(
                "  x0 = {:.17e}  with-break = {:.17e} (bits 0x{:016x})  \
                 no-break = {:.17e} (bits 0x{:016x})  xor = 0x{:016x}\n",
                x0,
                a,
                a.to_bits(),
                b,
                b.to_bits(),
                a.to_bits() ^ b.to_bits(),
            ));
        }
        panic!("{msg}");
    }
}

// guards against a wrong break_when accidentally gating something other than
// early exit. shallow cap (= 2 Newton steps) so conv never fires from x0=100.
// even with break_when never firing, both paths must produce identical bits.
#[test]
fn iterate_break_equivalent_when_cap_hit_before_convergence() {
    let f_break = build_newton_sqrt2(true, 2);
    let f_no_break = build_newton_sqrt2(false, 2);

    let a = Cpu.eval_elemental(&f_break, &[100.0])[0];
    let b = Cpu.eval_elemental(&f_no_break, &[100.0])[0];
    assert_eq!(
        a.to_bits(),
        b.to_bits(),
        "non-converging case: with-break {:e} != no-break {:e} (xor 0x{:016x}) — \
         break_when changed the value despite never firing",
        a,
        b,
        a.to_bits() ^ b.to_bits(),
    );
    // verify the test really is hitting the cap path (value != sqrt2).
    assert!(
        (a - std::f64::consts::SQRT_2).abs() > 1e-3,
        "shallow Newton converged unexpectedly — bump x0 further or shrink the cap",
    );
}
