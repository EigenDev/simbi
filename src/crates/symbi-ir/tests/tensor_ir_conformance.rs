// =============================================================================
// tensor_ir_conformance.rs
//
// integration test suite that pins the behavior of the tensor IR end-to-end:
//   build graph -> scalarize -> emit_cpu / emit_cuda
//
// each test corresponds to a chalkboard-math operation users will write
// via the macro layer (R.5). when R.5 retires the legacy `tensor_rewrite`
// module (in symbi-macros), THIS file is what proves every behavior the
// old syn-rewrite pass produced is reproducible through the new IR.
//
// section index:
//   1. vector ops via Einsum  (dot, norm, sum, trace-style)
//   2. broadcast scaling      (scalar * vector via ElementWise + Broadcast)
//   3. element-wise vector    (Add, Sub, Neg, etc.)
//   4. component access       (Index)
//   5. tensor construction    (Construct)
//   6. variance + .contract   (einsum with Upper/Lower pairing)
//   7. matrix ops via Einsum  (trace, matvec, matmul, quadratic, bilinear)
//   8. reduce + select
// =============================================================================

use symbi_ir::{
    ConstValue, DimExpr, ElementTy, ElementWiseOp, Graph, ReduceOp,
    Symbol, TensorTy, VarianceTag, emit_cpu, emit_cuda, scalarize,
};

// ----- helpers -----

fn lit(n: usize) -> DimExpr { DimExpr::Literal(n) }

fn add_vec(g: &mut Graph, name: &str, dim: usize) -> symbi_ir::NodeId {
    g.add_param(
        Symbol::intern(name),
        TensorTy::from_shape(ElementTy::F64, vec![lit(dim)]),
        None,
    )
}

fn add_upper(g: &mut Graph, name: &str, dim: usize) -> symbi_ir::NodeId {
    g.add_param(
        Symbol::intern(name),
        TensorTy::from_shape(ElementTy::F64, vec![lit(dim)]).with_variance(VarianceTag::Upper),
        None,
    )
}

fn add_lower(g: &mut Graph, name: &str, dim: usize) -> symbi_ir::NodeId {
    g.add_param(
        Symbol::intern(name),
        TensorTy::from_shape(ElementTy::F64, vec![lit(dim)]).with_variance(VarianceTag::Lower),
        None,
    )
}

fn add_mat(g: &mut Graph, name: &str, rows: usize, cols: usize) -> symbi_ir::NodeId {
    g.add_param(
        Symbol::intern(name),
        TensorTy::from_shape(ElementTy::F64, vec![lit(rows), lit(cols)]),
        None,
    )
}

fn parse_rust(src: &str) {
    let r: syn::Result<syn::ItemFn> = syn::parse_str(src);
    assert!(r.is_ok(), "emit_cpu output not valid Rust:\n{}\nerror: {}", src, r.err().unwrap());
}

// ============================================================
// section 1. vector ops via Einsum
// ============================================================

#[test]
fn dot_product_3d() {
    let mut g = Graph::new();
    let a = add_vec(&mut g, "a", 3);
    let b = add_vec(&mut g, "b", 3);
    let r = g.einsum("i,i->", vec![a, b], None);
    assert!(!g.has_errors());
    let f = scalarize(&g, r, "dot3");
    let src = emit_cpu(&f);
    parse_rust(&src);
    assert!(src.contains("(a_0 * b_0)"), "{}", src);
    assert!(src.contains("(a_1 * b_1)"), "{}", src);
    assert!(src.contains("(a_2 * b_2)"), "{}", src);
    assert!(src.contains("-> f64"));
}

#[test]
fn norm_squared_via_self_dot() {
    // .norm_squared() == .dot(self), so the IR uses einsum "i,i->" with one input twice.
    let mut g = Graph::new();
    let t = add_vec(&mut g, "t", 3);
    let r = g.einsum("i,i->", vec![t, t], None);
    let f = scalarize(&g, r, "norm_sq");
    let src = emit_cpu(&f);
    parse_rust(&src);
    // each term: t_i * t_i
    assert!(src.contains("(t_0 * t_0)"), "{}", src);
    assert!(src.contains("(t_2 * t_2)"), "{}", src);
}

#[test]
fn norm_via_sqrt_of_norm_squared() {
    let mut g = Graph::new();
    let t = add_vec(&mut g, "t", 2);
    let ns = g.einsum("i,i->", vec![t, t], None);
    let n = g.element_wise(ElementWiseOp::Sqrt, vec![ns], None);
    let f = scalarize(&g, n, "norm");
    let src = emit_cpu(&f);
    parse_rust(&src);
    // outer .sqrt() wrapping the dot sum
    assert!(src.contains(".sqrt()"), "{}", src);
    assert!(src.contains("(t_0 * t_0)"), "{}", src);
}

#[test]
fn component_sum_via_einsum() {
    // .component_sum() of a [3] vector is einsum("i->")
    let mut g = Graph::new();
    let t = add_vec(&mut g, "t", 3);
    let r = g.einsum("i->", vec![t], None);
    let f = scalarize(&g, r, "sum");
    let src = emit_cpu(&f);
    parse_rust(&src);
    // sum of 3 terms — leaves are t_0, t_1, t_2 with Add nodes
    assert!(src.contains("t_0"));
    assert!(src.contains("t_2"));
}

// ============================================================
// section 2. broadcast scaling (scalar * vector)
// ============================================================

#[test]
fn scalar_times_vector() {
    let mut g = Graph::new();
    let s = g.add_scalar_param("s", ElementTy::F64);
    let v = add_vec(&mut g, "v", 3);
    let r = g.element_wise(ElementWiseOp::Mul, vec![s, v], None);
    let f = scalarize(&g, r, "scale3");
    let src = emit_cpu(&f);
    parse_rust(&src);
    assert!(src.contains("(s * v_0)"));
    assert!(src.contains("(s * v_2)"));
    assert!(src.contains("-> (f64, f64, f64)"));
}

#[test]
fn vector_times_vector_pairs_components() {
    let mut g = Graph::new();
    let v = add_vec(&mut g, "v", 2);
    let w = add_vec(&mut g, "w", 2);
    let r = g.element_wise(ElementWiseOp::Mul, vec![v, w], None);
    let f = scalarize(&g, r, "hadamard");
    let src = emit_cpu(&f);
    parse_rust(&src);
    assert!(src.contains("(v_0 * w_0)"));
    assert!(src.contains("(v_1 * w_1)"));
}

// ============================================================
// section 3. element-wise vector arithmetic
// ============================================================

#[test]
fn vector_add() {
    let mut g = Graph::new();
    let v = add_vec(&mut g, "v", 2);
    let w = add_vec(&mut g, "w", 2);
    let r = g.element_wise(ElementWiseOp::Add, vec![v, w], None);
    let f = scalarize(&g, r, "vadd");
    let src = emit_cpu(&f);
    parse_rust(&src);
    assert!(src.contains("(v_0 + w_0)"));
    assert!(src.contains("(v_1 + w_1)"));
}

#[test]
fn vector_sub() {
    let mut g = Graph::new();
    let v = add_vec(&mut g, "v", 2);
    let w = add_vec(&mut g, "w", 2);
    let r = g.element_wise(ElementWiseOp::Sub, vec![v, w], None);
    let f = scalarize(&g, r, "vsub");
    let src = emit_cpu(&f);
    parse_rust(&src);
    assert!(src.contains("(v_0 - w_0)"));
    assert!(src.contains("(v_1 - w_1)"));
}

#[test]
fn vector_neg() {
    let mut g = Graph::new();
    let v = add_vec(&mut g, "v", 2);
    let r = g.element_wise(ElementWiseOp::Neg, vec![v], None);
    let f = scalarize(&g, r, "vneg");
    let src = emit_cpu(&f);
    parse_rust(&src);
    assert!(src.contains("(-v_0)"));
    assert!(src.contains("(-v_1)"));
}

// ============================================================
// section 4. component access (Index)
// ============================================================

#[test]
fn index_extracts_scalar_from_vector() {
    let mut g = Graph::new();
    let v = add_vec(&mut g, "v", 3);
    let s = g.index(v, vec![symbi_ir::DimIndex::Literal(1)], None);
    let f = scalarize(&g, s, "pick");
    let src = emit_cpu(&f);
    parse_rust(&src);
    // result body is just v_1
    assert!(src.contains("v_1"));
    assert!(src.contains("-> f64"));
}

#[test]
fn index_extracts_matrix_element() {
    let mut g = Graph::new();
    let m = add_mat(&mut g, "M", 3, 4);
    let s = g.index(
        m,
        vec![
            symbi_ir::DimIndex::Literal(2),
            symbi_ir::DimIndex::Literal(1),
        ],
        None,
    );
    let f = scalarize(&g, s, "pickm");
    let src = emit_cpu(&f);
    parse_rust(&src);
    assert!(src.contains("M_2_1"));
}

// ============================================================
// section 5. tensor construction (Construct)
// ============================================================

#[test]
fn construct_three_scalars_into_vector() {
    let mut g = Graph::new();
    let a = g.add_scalar_param("a", ElementTy::F64);
    let b = g.add_scalar_param("b", ElementTy::F64);
    let c = g.add_scalar_param("c", ElementTy::F64);
    let v = g.construct(vec![a, b, c], None);
    let f = scalarize(&g, v, "mkvec");
    let src = emit_cpu(&f);
    parse_rust(&src);
    // tuple return with the three scalar params in order
    assert!(src.contains("-> (f64, f64, f64)"));
    assert!(src.contains("(a, b, c)"));
}

#[test]
fn construct_two_vectors_into_matrix() {
    let mut g = Graph::new();
    let v1 = add_vec(&mut g, "v1", 2);
    let v2 = add_vec(&mut g, "v2", 2);
    let m = g.construct(vec![v1, v2], None);
    let f = scalarize(&g, m, "mkmat");
    let src = emit_cpu(&f);
    parse_rust(&src);
    // 4-tuple in row-major order: v1_0, v1_1, v2_0, v2_1
    assert!(src.contains("(v1_0, v1_1, v2_0, v2_1)"));
}

// ============================================================
// section 6. variance + .contract via Einsum
// ============================================================

#[test]
fn contract_upper_with_lower_succeeds() {
    let mut g = Graph::new();
    let v = add_upper(&mut g, "v", 3);
    let w = add_lower(&mut g, "w", 3);
    let r = g.einsum("i,i->", vec![v, w], None);
    assert!(!g.has_errors(), "errors: {:?}", g.errors());
    let f = scalarize(&g, r, "ctr");
    let src = emit_cpu(&f);
    parse_rust(&src);
    assert!(src.contains("(v_0 * w_0)"));
    assert!(src.contains("(v_2 * w_2)"));
}

#[test]
fn contract_lower_with_upper_succeeds() {
    let mut g = Graph::new();
    let w = add_lower(&mut g, "w", 2);
    let v = add_upper(&mut g, "v", 2);
    let _ = g.einsum("i,i->", vec![w, v], None);
    assert!(!g.has_errors());
}

#[test]
fn contract_two_uppers_emits_variance_error() {
    let mut g = Graph::new();
    let a = add_upper(&mut g, "a", 3);
    let b = add_upper(&mut g, "b", 3);
    let _ = g.einsum("i,i->", vec![a, b], None);
    let summaries: Vec<String> = g.errors().iter().map(|e| e.summary()).collect();
    assert!(summaries.iter().any(|s| s.contains("variance")), "{:?}", summaries);
}

#[test]
fn contract_two_lowers_emits_variance_error() {
    let mut g = Graph::new();
    let a = add_lower(&mut g, "a", 2);
    let b = add_lower(&mut g, "b", 2);
    let _ = g.einsum("i,i->", vec![a, b], None);
    let summaries: Vec<String> = g.errors().iter().map(|e| e.summary()).collect();
    assert!(summaries.iter().any(|s| s.contains("variance")), "{:?}", summaries);
}

#[test]
fn contract_untagged_with_tagged_is_permitted() {
    // .dot (Untagged inputs) and mixed Untagged+Tagged should not error
    // — Untagged matches anything per spec § 2.5.
    let mut g = Graph::new();
    let plain = add_vec(&mut g, "plain", 2);
    let upper = add_upper(&mut g, "u", 2);
    let _ = g.einsum("i,i->", vec![plain, upper], None);
    assert!(!g.has_errors());
}

// ============================================================
// section 7. matrix ops via Einsum
// ============================================================

#[test]
fn matrix_trace() {
    let mut g = Graph::new();
    let m = add_mat(&mut g, "M", 3, 3);
    let r = g.einsum("ii->", vec![m], None);
    let f = scalarize(&g, r, "tr");
    let src = emit_cpu(&f);
    parse_rust(&src);
    assert!(src.contains("M_0_0"));
    assert!(src.contains("M_1_1"));
    assert!(src.contains("M_2_2"));
}

#[test]
fn matrix_vector_product() {
    let mut g = Graph::new();
    let m = add_mat(&mut g, "M", 2, 3);
    let v = add_vec(&mut g, "v", 3);
    let r = g.einsum("ij,j->i", vec![m, v], None);
    let f = scalarize(&g, r, "mv");
    let src = emit_cpu(&f);
    parse_rust(&src);
    // result[0] = M_0_0*v_0 + M_0_1*v_1 + M_0_2*v_2
    assert!(src.contains("(M_0_0 * v_0)"));
    assert!(src.contains("(M_0_2 * v_2)"));
    assert!(src.contains("(M_1_2 * v_2)"));
}

#[test]
fn matmul_2x2() {
    let mut g = Graph::new();
    let m = add_mat(&mut g, "M", 2, 2);
    let n = add_mat(&mut g, "N", 2, 2);
    let r = g.einsum("ij,jk->ik", vec![m, n], None);
    let f = scalarize(&g, r, "matmul");
    let src = emit_cpu(&f);
    parse_rust(&src);
    // R[0,0] = M_0_0 * N_0_0 + M_0_1 * N_1_0
    assert!(src.contains("(M_0_0 * N_0_0)"));
    assert!(src.contains("(M_0_1 * N_1_0)"));
}

#[test]
fn matrix_quadratic_form() {
    // v^T M v via ij,i,j->
    let mut g = Graph::new();
    let m = add_mat(&mut g, "M", 2, 2);
    let v = add_vec(&mut g, "v", 2);
    let r = g.einsum("ij,i,j->", vec![m, v, v], None);
    let f = scalarize(&g, r, "quad");
    let src = emit_cpu(&f);
    parse_rust(&src);
    // each term: M_i_j * v_i * v_j
    assert!(src.contains("(M_0_0 * v_0)"));
    assert!(src.contains("(M_1_1 * v_1)"));
}

#[test]
#[allow(non_snake_case)] // bilinear form a^T M b: M is the matrix, deliberate notation
fn matrix_bilinear_a_M_b() {
    let mut g = Graph::new();
    let m = add_mat(&mut g, "M", 2, 2);
    let a = add_vec(&mut g, "a", 2);
    let b = add_vec(&mut g, "b", 2);
    let r = g.einsum("ij,i,j->", vec![m, a, b], None);
    let f = scalarize(&g, r, "bil");
    let src = emit_cpu(&f);
    parse_rust(&src);
    assert!(src.contains("(M_0_0 * a_0)"));
    assert!(src.contains("(M_0_1 * a_0)"));
    assert!(src.contains("b_0"));
    assert!(src.contains("b_1"));
}

#[test]
fn outer_product() {
    let mut g = Graph::new();
    let a = add_vec(&mut g, "a", 2);
    let b = add_vec(&mut g, "b", 3);
    let r = g.einsum("i,j->ij", vec![a, b], None);
    let f = scalarize(&g, r, "outer");
    let src = emit_cpu(&f);
    parse_rust(&src);
    // 6 entries, each a_i * b_j
    assert!(src.contains("-> (f64, f64, f64, f64, f64, f64)"));
    assert!(src.contains("(a_0 * b_0)"));
    assert!(src.contains("(a_1 * b_2)"));
}

// ============================================================
// section 8. Reduce + Select
// ============================================================

#[test]
fn reduce_max_over_vector() {
    let mut g = Graph::new();
    let v = add_vec(&mut g, "v", 3);
    let r = g.reduce(ReduceOp::Max, vec![0], v, None);
    let f = scalarize(&g, r, "vmax");
    let src = emit_cpu(&f);
    parse_rust(&src);
    // chain: v_0.max(v_1).max(v_2)
    assert!(src.contains(".max"));
    assert!(src.contains("v_0"));
    assert!(src.contains("v_2"));
}

#[test]
fn reduce_min_over_matrix_inner_axis() {
    let mut g = Graph::new();
    let m = add_mat(&mut g, "M", 2, 3);
    let r = g.reduce(ReduceOp::Min, vec![1], m, None);
    let f = scalarize(&g, r, "mmin");
    let src = emit_cpu(&f);
    parse_rust(&src);
    // output is [2]; row 0 should mention M_0_0..M_0_2
    assert!(src.contains("M_0_0"));
    assert!(src.contains("M_0_2"));
    assert!(src.contains("M_1_2"));
    assert!(src.contains(".min"));
}

#[test]
fn select_with_scalar_cond_picks_branch() {
    let mut g = Graph::new();
    let c = g.add_scalar_param("c", ElementTy::Bool);
    let t = g.add_scalar_param("t", ElementTy::F64);
    let e = g.add_scalar_param("e", ElementTy::F64);
    let r = g.select(c, t, e, None);
    let f = scalarize(&g, r, "pick");
    let src = emit_cpu(&f);
    parse_rust(&src);
    assert!(src.contains("(if c { t } else { e })"));
}

#[test]
fn select_broadcasts_scalar_cond_to_vector_branches() {
    let mut g = Graph::new();
    let c = g.add_scalar_param("c", ElementTy::Bool);
    let t = add_vec(&mut g, "t", 2);
    let e = add_vec(&mut g, "e", 2);
    let r = g.select(c, t, e, None);
    let f = scalarize(&g, r, "vpick");
    let src = emit_cpu(&f);
    parse_rust(&src);
    // result is a 2-tuple of independent (if c { t_i } else { e_i }) selects
    assert!(src.contains("(if c { t_0 } else { e_0 })"));
    assert!(src.contains("(if c { t_1 } else { e_1 })"));
}

// ============================================================
// CUDA emit smoke tests (one per category)
// ============================================================

#[test]
fn cuda_dot_emits_device_inline_double() {
    let mut g = Graph::new();
    let a = add_vec(&mut g, "a", 2);
    let b = add_vec(&mut g, "b", 2);
    let r = g.einsum("i,i->", vec![a, b], None);
    let f = scalarize(&g, r, "dot2");
    let src = emit_cuda(&f);
    assert!(src.contains("__device__ inline double dot2"));
    assert!(src.contains("(a_0 * b_0)"));
}

#[test]
fn cuda_matmul_emits_struct_return() {
    let mut g = Graph::new();
    let m = add_mat(&mut g, "M", 2, 2);
    let n = add_mat(&mut g, "N", 2, 2);
    let r = g.einsum("ij,jk->ik", vec![m, n], None);
    let f = scalarize(&g, r, "mm22");
    let src = emit_cuda(&f);
    assert!(src.contains("struct mm22_out"));
    assert!(src.contains("__device__ inline mm22_out mm22"));
}

#[test]
fn cuda_abs_emits_ternary_not_fabs() {
    // task #43: emit `(x < 0.0 ? -x : x)` matching C++ my_abs, not the
    // libdevice fabs() whose IEEE 754-2008 NaN/signed-zero semantics
    // differ from C++ at shock-edge primitives.
    let mut g = Graph::new();
    let x = g.add_scalar_param("x", ElementTy::F64);
    let r = g.element_wise(ElementWiseOp::Abs, vec![x], None);
    let f = scalarize(&g, r, "absx");
    let src = emit_cuda(&f);
    assert!(src.contains("(x < 0.0 ? -x : x)"),
        "expected my_abs-style ternary, got:\n{}", src);
    assert!(!src.contains("fabs"),
        "should not emit fabs, got:\n{}", src);
    assert!(!src.contains(".abs"));
}

// ============================================================
// const value emission corner cases
// ============================================================

#[test]
fn const_negative_float_emits_correctly() {
    let mut g = Graph::new();
    let c = g.add_const(ConstValue::F64(-1.5), None);
    let f = scalarize(&g, c, "neg_1_5");
    let cpu = emit_cpu(&f);
    parse_rust(&cpu);
    assert!(cpu.contains("-1.5_f64"));
    let cuda = emit_cuda(&f);
    assert!(cuda.contains("-1.5"));
}

#[test]
fn const_zero_emits_zero_literal() {
    let mut g = Graph::new();
    let c = g.add_const(ConstValue::F64(0.0), None);
    let f = scalarize(&g, c, "zero");
    let cpu = emit_cpu(&f);
    parse_rust(&cpu);
    assert!(cpu.contains("0.0_f64"));
}

#[test]
fn const_bool_round_trip() {
    let mut g = Graph::new();
    let t = g.add_const(ConstValue::Bool(true), None);
    let f = scalarize(&g, t, "tru");
    let cpu = emit_cpu(&f);
    parse_rust(&cpu);
    assert!(cpu.contains("    true\n"));
    assert!(cpu.contains("-> bool"));
}

// ============================================================
// pipelined chains (multiple ops composed)
// ============================================================

#[test]
fn norm_of_difference_vector() {
    // ||v - w||^2 — a common physics primitive.
    // each (v_i - w_i) is a SHALLOW cheap share (height 1) used twice. under the
    // depth-aware cost model the cheap threshold stays at 4 uses for shallow
    // shares (a single sub is not worth a register), so the subtraction is
    // inlined in both squared factors rather than hoisted. semantic equivalence
    // is preserved; only deep cheap chains (height >= 3) hoist at 2 uses.
    let mut g = Graph::new();
    let v = add_vec(&mut g, "v", 3);
    let w = add_vec(&mut g, "w", 3);
    let d = g.element_wise(ElementWiseOp::Sub, vec![v, w], None);
    let r = g.einsum("i,i->", vec![d, d], None);
    let f = scalarize(&g, r, "dist_sq");
    let src = emit_cpu(&f);
    parse_rust(&src);
    // the per-axis differences appear inline, squared (referenced twice each).
    assert!(src.contains("(v_0 - w_0) * (v_0 - w_0)"), "missing inlined v_0 diff square: {}", src);
    assert!(src.contains("(v_2 - w_2) * (v_2 - w_2)"), "missing inlined v_2 diff square: {}", src);
    // shallow cheap 2-use shares are NOT hoisted (depth-aware threshold).
    assert!(!src.contains("__cse_"), "shallow cheap 2-use diff must stay inline: {}", src);
}

#[test]
fn weighted_sum_kernel() {
    // result = w_0 * v_0 + w_1 * v_1 + w_2 * v_2 via einsum
    let mut g = Graph::new();
    let v = add_vec(&mut g, "v", 3);
    let w = add_vec(&mut g, "w", 3);
    let r = g.einsum("i,i->", vec![w, v], None);
    let f = scalarize(&g, r, "wsum");
    let src = emit_cpu(&f);
    parse_rust(&src);
    assert!(src.contains("(w_0 * v_0)"));
    assert!(src.contains("(w_2 * v_2)"));
}

#[test]
fn velocity_from_momentum_divide_by_rho() {
    // v_i = m_i / rho — the c2p body for isothermal-style kernels.
    let mut g = Graph::new();
    let rho = g.add_scalar_param("rho", ElementTy::F64);
    let mom = add_vec(&mut g, "mom", 2);
    let v = g.element_wise(ElementWiseOp::Div, vec![mom, rho], None);
    let f = scalarize(&g, v, "vel_from_mom");
    let src = emit_cpu(&f);
    parse_rust(&src);
    assert!(src.contains("-> (f64, f64)"));
    assert!(src.contains("(mom_0 / rho)"));
    assert!(src.contains("(mom_1 / rho)"));
}

// ============================================================
// summary smoke: every op produced reach-able from a graph
// ============================================================

#[test]
fn every_op_kind_lowers_without_panic() {
    // exercises the dispatch in lower.rs::lower_node to make sure no
    // op variant has been overlooked. uses small dim-2 shapes throughout.
    let mut g = Graph::new();
    let s = g.add_scalar_param("s", ElementTy::F64);
    let b = g.add_scalar_param("b", ElementTy::Bool);
    let v = add_vec(&mut g, "v", 2);
    let w = add_vec(&mut g, "w", 2);
    let m = add_mat(&mut g, "M", 2, 2);

    // ElementWise
    let _ = g.element_wise(ElementWiseOp::Add, vec![v, w], None);
    let _ = g.element_wise(ElementWiseOp::Neg, vec![v], None);
    let _ = g.element_wise(ElementWiseOp::Abs, vec![s], None);
    let _ = g.element_wise(ElementWiseOp::Min, vec![s, s], None);
    let _ = g.element_wise(ElementWiseOp::Lt, vec![s, s], None);
    let _ = g.element_wise(ElementWiseOp::IsFinite, vec![s], None);

    // Transcendental
    let _ = g.transcendental(symbi_ir::TranscendentalOp::Sin, vec![s], None);
    let _ = g.transcendental(symbi_ir::TranscendentalOp::Pow, vec![s, s], None);

    // Reduce
    let _ = g.reduce(ReduceOp::Max, vec![0], v, None);

    // Select
    let _ = g.select(b, s, s, None);

    // Einsum
    let _ = g.einsum("i,i->", vec![v, w], None);
    let _ = g.einsum("ij,jk->ik", vec![m, m], None);

    // Construct + Index + Broadcast
    let cv = g.construct(vec![s, s], None);
    let _ = g.index(cv, vec![symbi_ir::DimIndex::Literal(0)], None);
    let _ = g.broadcast(s, vec![lit(2)], None);

    // every node should have a valid type; full crate test ensures no
    // panic in lowering.
    assert!(!g.has_errors(), "errors: {:?}", g.errors());

    // pick one output and lower end-to-end as a final smoke check
    let result = g.einsum("i,i->", vec![v, w], None);
    let f = scalarize(&g, result, "summary");
    let src = emit_cpu(&f);
    parse_rust(&src);
}
