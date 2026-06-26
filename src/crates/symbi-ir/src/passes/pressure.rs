// =============================================================================
// pressure.rs
//
// **docs/design/23 step 5**: peak-register-pressure analysis over the lowered
// scope tree. counts the maximum number of simultaneously-live let bindings
// along any program point. Scope's `{ body; result }` is a hoisting barrier
// — bindings declared inside die at the closing brace; only `name` (the
// scope's result) survives into the outer scope.
//
// the metric is an UPPER BOUND on true register pressure: a let bound on
// line 1 and last-used on line 2 still occupies a slot here for the rest of
// its enclosing block. but it's the metric the *scope structure* directly
// bounds — and the one the design's algebra (`seq`/`par`/`scope`/`share`)
// composes over without any liveness inference.
//
// usage:
//   - `peak_pressure(body)` on any `&[ScalarStmt]` returns a `PressureReport`
//     with the peak count and the scope path where it was observed.
//   - `peak_pressure_kernel(k)` / `peak_pressure_fn(f)` wrappers for the
//     two scalarizer outputs.
//   - `assert_peak_pressure!(body, max)` macro for kernel-author tests.
//
// usage:
//   #[test]
//   fn rmhd_wave_speed_under_bound() {
//       let mut k = trace_and_scalarize(rmhd::wave_speed_map);
//       cse::cse_kernel(&mut k);
//       assert_peak_pressure!(&k.body, 80);
//   }
// =============================================================================

use crate::passes::scalarize::{LoweredFn, ScalarStmt};
use crate::KernelScalarized;

/// the outcome of a peak-pressure analysis.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PressureReport {
    /// peak number of simultaneously-live `Let` / `LetMut` / Scope-result
    /// bindings across the function body. the design's invariant
    /// `#[kernel(peak_pressure ≤ N)]` asserts `peak ≤ N`.
    pub peak: usize,
    /// the scope path where the peak was observed. empty = function root
    /// (top-level body). a single entry like `["phase"]` = inside the `phase`
    /// scope. used by error diagnostics to point at the offending region.
    pub at_scope_path: Vec<String>,
}

/// run peak-pressure analysis over a top-level body. ENTRY POINT.
pub fn peak_pressure(body: &[ScalarStmt]) -> PressureReport {
    let mut state = PressureState {
        peak: 0,
        at_scope_path: Vec::new(),
        current_path: Vec::new(),
    };
    walk(body, 0, &mut state);
    PressureReport {
        peak: state.peak,
        at_scope_path: state.at_scope_path,
    }
}

/// convenience: run over a scalarized kernel.
pub fn peak_pressure_kernel(k: &KernelScalarized) -> PressureReport {
    peak_pressure(&k.body)
}

/// convenience: run over a lowered (elemental-form) function.
pub fn peak_pressure_fn(f: &LoweredFn) -> PressureReport {
    peak_pressure(&f.body)
}

// ----- internals -----

struct PressureState {
    peak:          usize,
    at_scope_path: Vec<String>,
    current_path:  Vec<String>,
}

impl PressureState {
    #[inline]
    fn observe(&mut self, live: usize) {
        if live > self.peak {
            self.peak = live;
            self.at_scope_path = self.current_path.clone();
        }
    }
}

/// walk one block. `live_in` is the count of bindings already live in the
/// enclosing scope (Scope body inherits its parent's live count, since
/// outer bindings remain visible inside).
fn walk(body: &[ScalarStmt], live_in: usize, state: &mut PressureState) {
    let mut live = live_in;
    // a fresh block boundary still counts: the very first stmt may peak.
    state.observe(live);
    for stmt in body {
        match stmt {
            ScalarStmt::Let { .. } | ScalarStmt::LetMut { .. } => {
                live += 1;
                state.observe(live);
            }
            ScalarStmt::Assign { .. } | ScalarStmt::CompoundAssign { .. }
              | ScalarStmt::Break => {
                // no new binding; assignments mutate existing slots.
            }
            ScalarStmt::For { iter, body, .. } => {
                // the iter variable is a fresh binding visible inside the body.
                // model it as +1 inside the loop, restored on exit.
                state.current_path.push(format!("for {iter}"));
                walk(body, live + 1, state);
                state.current_path.pop();
            }
            ScalarStmt::If { then_body, .. } => {
                // no new binding in the predicate path itself; recurse into the
                // branch with the same live count. the branch terminates at `}`.
                state.current_path.push("if".to_string());
                walk(then_body, live, state);
                state.current_path.pop();
            }
            ScalarStmt::Scope { name, body: scope_body, .. } => {
                // *** the load-bearing case for the design. ***
                //
                // inside the scope BODY, peak = max(scope_internal_peak)
                //   — the body inherits `live` (outer bindings still visible)
                //   — and the body's own `Let`s add on top of that.
                // outer bindings that were live BEFORE the scope opened
                // remain live (they may be referenced inside). that's the
                // baseline.
                //
                // after the scope CLOSES, the body's internal lets DIE (the
                // `}` reaps them). only the scope's named result survives,
                // adding +1 to the outer's live count.
                state.current_path.push(name.clone());
                walk(scope_body, live, state);
                state.current_path.pop();
                live += 1;
                state.observe(live);
            }
            ScalarStmt::IfElse { outs, then_body, else_body, .. } => {
                // each arm inherits the outer `live` (outer bindings stay
                // visible); the arm's own lets die at its `}`. the two arms are
                // mutually exclusive, so peak = max over the two arm walks, not
                // their sum. the branch's N named results survive afterward.
                let label = outs.first().map(|(n, _)| n.as_str()).unwrap_or("if");
                state.current_path.push(format!("{label} (then)"));
                walk(then_body, live, state);
                state.current_path.pop();
                state.current_path.push(format!("{label} (else)"));
                walk(else_body, live, state);
                state.current_path.pop();
                live += outs.len();
                state.observe(live);
            }
        }
    }
}

/// **the kernel-author surface**. asserts that the body's peak pressure does
/// not exceed `max`; panics with the offending scope path on violation. use
/// from `#[test]` to enforce a per-kernel pressure bound at `cargo test` time.
///
/// example:
/// ```ignore
/// #[test]
/// fn pressure_bound_rmhd_wave_speed() {
///     let mut k = trace_and_scalarize(rmhd::wave_speed_map);
///     symbi_ir::passes::cse::cse_kernel(&mut k);
///     assert_peak_pressure!(&k.body, 80);
/// }
/// ```
#[macro_export]
macro_rules! assert_peak_pressure {
    ($body:expr, $max:expr) => {{
        let __report = $crate::passes::pressure::peak_pressure($body);
        assert!(
            __report.peak <= $max,
            "peak register pressure {} exceeds bound {} at scope {:?}\n\
             hint: wrap a natural phase in `S::scope(|| ...)` to give nvcc \
             explicit lifetime information",
            __report.peak,
            $max,
            __report.at_scope_path,
        );
    }};
}

// ----- tests -----

#[cfg(test)]
mod tests {
    use super::*;
    use crate::passes::scalarize::{ScalarExpr, ScalarStmt};
    use crate::{ConstValue, ElementTy};

    fn let_f64(name: &str, value: ScalarExpr) -> ScalarStmt {
        ScalarStmt::Let { name: name.to_string(), element: ElementTy::F64, value }
    }

    fn v(name: &str) -> ScalarExpr { ScalarExpr::Var(name.to_string()) }

    fn lit(x: f64) -> ScalarExpr { ScalarExpr::Const(ConstValue::F64(x)) }

    fn scope(name: &str, body: Vec<ScalarStmt>, result: ScalarExpr) -> ScalarStmt {
        ScalarStmt::Scope { name: name.to_string(), element: ElementTy::F64, body, result }
    }

    /// empty body: peak = 0.
    #[test]
    fn empty_body_has_zero_peak() {
        let r = peak_pressure(&[]);
        assert_eq!(r.peak, 0);
        assert!(r.at_scope_path.is_empty());
    }

    /// N flat lets in a row: peak = N. THIS IS THE FLAT-KERNEL PATHOLOGY
    /// the design exists to fix.
    #[test]
    fn flat_lets_count_linearly() {
        let body: Vec<ScalarStmt> = (0..10)
            .map(|i| let_f64(&format!("t{i}"), lit(i as f64)))
            .collect();
        let r = peak_pressure(&body);
        assert_eq!(r.peak, 10, "10 flat lets should peak at 10");
        assert!(r.at_scope_path.is_empty(), "peak should be at function root");
    }

    /// the seq law: `peak(seq(K1, K2)) = max(peak(K1), peak(K2))` ONLY when
    /// neither leaks. for FLAT seq (no scope between), they DO leak (K1's
    /// bindings remain live during K2). this test pins the flat case.
    #[test]
    fn flat_seq_accumulates() {
        let mut body = Vec::new();
        for i in 0..5 { body.push(let_f64(&format!("a{i}"), lit(i as f64))); }
        for i in 0..5 { body.push(let_f64(&format!("b{i}"), lit(i as f64))); }
        let r = peak_pressure(&body);
        // flat seq: 5 a's + 5 b's all live = 10.
        assert_eq!(r.peak, 10);
    }

    /// **the scope law: `peak(scope(K)) = peak(K)`**. the internals die at the
    /// brace — only the result (`+1`) leaks. THIS is the design property.
    #[test]
    fn scope_drops_internals_on_close() {
        // outer: { scope phase { 5 lets, result }; 0 more outer lets }
        // peak: max(scope_internal = 5, after_scope = 1) = 5.
        let scope_body: Vec<ScalarStmt> = (0..5)
            .map(|i| let_f64(&format!("t{i}"), lit(i as f64)))
            .collect();
        let outer = vec![scope("phase", scope_body, v("t4"))];
        let r = peak_pressure(&outer);
        assert_eq!(r.peak, 5, "scope peak = body's internal peak");
        assert_eq!(r.at_scope_path, vec!["phase".to_string()],
            "peak observed INSIDE phase");
    }

    /// the share law: `peak(share(v, K1, K2)) = peak(K1) + size(v) + peak(K2)`
    /// — represented here as a let `v` declared in the outer, alive across two
    /// scopes that each peak at K.
    #[test]
    fn share_law_outer_binding_lives_across_scopes() {
        // outer:
        //   let v = ...;          // share
        //   scope k1 { 3 lets, _ } // inside: 1 (v) + 3 = 4
        //   scope k2 { 3 lets, _ } // inside: 1 (v) + 1 (k1.result) + 3 = 5
        // peak = 5, observed inside k2.
        let make_phase = |prefix: &str| -> Vec<ScalarStmt> {
            (0..3)
                .map(|i| let_f64(&format!("{prefix}{i}"), lit(i as f64)))
                .collect()
        };
        let body = vec![
            let_f64("v", lit(0.0)),
            scope("k1", make_phase("a"), v("a2")),
            scope("k2", make_phase("b"), v("b2")),
        ];
        let r = peak_pressure(&body);
        // v alive throughout, k1.result alive across k2, k2's body adds 3
        // → 1 + 1 + 3 = 5
        assert_eq!(r.peak, 5);
        assert_eq!(r.at_scope_path, vec!["k2".to_string()]);
    }

    /// nested scopes: inner peak is INSIDE the inner; outer-after-inner
    /// retains only the inner's named result.
    #[test]
    fn nested_scopes_isolate_inner_peak() {
        // outer:
        //   let outer_pre = ...;       // 1
        //   scope outer_name {
        //       let outer_local = ...;  // 1 (visible inside outer scope)
        //       scope inner_name {
        //           5 lets               // 1 (outer_pre) + 1 (outer_local) + 5 = 7
        //       }
        //       // inner closes; +1 for inner_name's result.
        //       // live = 1 + 1 + 1 = 3
        //   }
        //   // outer closes; +1 for outer_name. live = 1 + 1 = 2.
        // peak = 7, INSIDE inner_name within outer_name.
        let inner_body: Vec<ScalarStmt> = (0..5)
            .map(|i| let_f64(&format!("i{i}"), lit(i as f64)))
            .collect();
        let outer_body = vec![
            let_f64("outer_local", lit(0.0)),
            scope("inner_name", inner_body, v("i4")),
        ];
        let body = vec![
            let_f64("outer_pre", lit(0.0)),
            scope("outer_name", outer_body, v("outer_local")),
        ];
        let r = peak_pressure(&body);
        assert_eq!(r.peak, 7, "peak inside nested inner scope");
        assert_eq!(r.at_scope_path,
            vec!["outer_name".to_string(), "inner_name".to_string()],
            "path leads into both scopes");
    }

    /// the design pathology: **flat = 200 lets peaks at 200, scoped = same
    /// physics peaks at 30**. this is the wave_speed_map projection in
    /// miniature.
    #[test]
    fn flat_200_temps_peaks_at_200() {
        let body: Vec<ScalarStmt> = (0..200)
            .map(|i| let_f64(&format!("t{i}"), lit(i as f64)))
            .collect();
        let r = peak_pressure(&body);
        assert_eq!(r.peak, 200);
    }

    /// SAME computation phrased as 4 scopes of 50 lets each peaks at 50
    /// internal + 4 surviving phase results = 52 → still under 60. THE WIN.
    #[test]
    fn scoped_200_temps_peaks_at_phase_size() {
        let phase = |prefix: &str| -> Vec<ScalarStmt> {
            (0..50)
                .map(|i| let_f64(&format!("{prefix}{i}"), lit(i as f64)))
                .collect()
        };
        let body = vec![
            scope("phase0", phase("a"), v("a49")),
            scope("phase1", phase("b"), v("b49")),
            scope("phase2", phase("c"), v("c49")),
            scope("phase3", phase("d"), v("d49")),
        ];
        let r = peak_pressure(&body);
        // inside phase3: 3 prior phase-results live + 50 internals = 53.
        // (each earlier scope leaves +1; phase3's internal is 50.)
        assert_eq!(r.peak, 53,
            "scoped form peaks at phase-internal + prior surviving results");
        // and the peak is observed INSIDE phase3 — the last scope's body.
        assert_eq!(r.at_scope_path, vec!["phase3".to_string()]);
    }

    /// the assert_peak_pressure! macro passes for under-bound.
    #[test]
    fn macro_passes_when_under_bound() {
        let body = vec![let_f64("a", lit(0.0)), let_f64("b", lit(0.0))];
        assert_peak_pressure!(&body, 5);
    }

    /// the assert_peak_pressure! macro panics on violation.
    #[test]
    #[should_panic(expected = "exceeds bound")]
    fn macro_panics_when_over_bound() {
        let body: Vec<ScalarStmt> = (0..200)
            .map(|i| let_f64(&format!("t{i}"), lit(i as f64)))
            .collect();
        assert_peak_pressure!(&body, 50);
    }

    /// the panic message names the scope path so the kernel author can
    /// locate the offending phase.
    #[test]
    #[should_panic(expected = "hot_phase")]
    fn macro_diagnostic_names_offending_scope() {
        let scope_body: Vec<ScalarStmt> = (0..50)
            .map(|i| let_f64(&format!("t{i}"), lit(i as f64)))
            .collect();
        let body = vec![scope("hot_phase", scope_body, v("t49"))];
        assert_peak_pressure!(&body, 10);
    }

    /// For-loop adds its iter variable but the body's lets die per iteration.
    #[test]
    fn for_loop_adds_iter_var_but_body_dies() {
        use crate::{DimExpr, Symbol};
        let body = vec![
            let_f64("outer", lit(0.0)),
            ScalarStmt::For {
                iter: "ii".to_string(),
                bound: DimExpr::Generic(Symbol::intern("D")),
                body: vec![let_f64("inner", lit(0.0))],
            },
            let_f64("after", lit(0.0)),
        ];
        let r = peak_pressure(&body);
        // inside the loop: outer + ii + inner = 3.
        // after the loop: outer + after = 2 (loop body's `inner` dies).
        assert_eq!(r.peak, 3);
    }

    /// assignments don't grow live count.
    #[test]
    fn assignments_do_not_grow_pressure() {
        let body = vec![
            ScalarStmt::LetMut {
                name: "acc".to_string(),
                element: ElementTy::F64,
                init: lit(0.0),
            },
            ScalarStmt::Assign {
                name: "acc".to_string(),
                value: lit(1.0),
            },
            ScalarStmt::CompoundAssign {
                name: "acc".to_string(),
                op: crate::passes::scalarize::BinaryKind::Add,
                value: lit(2.0),
            },
        ];
        let r = peak_pressure(&body);
        assert_eq!(r.peak, 1, "LetMut + assigns = 1 live binding");
    }
}
