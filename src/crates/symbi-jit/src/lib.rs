// =============================================================================
// symbi-jit/src/lib.rs
//
// the CPU's NVRTC. `compile(&LoweredFn)` translates a scalarized IR function (the
// SAME `LoweredFn` the interpreter `Cpu::eval_elemental` runs) into native machine
// code via Cranelift, returning a `CompiledFn` callable as
// `fn(inputs: &[f64], out: &mut [f64])`.
//
// the v1 subset is exactly what a SOURCE dag emits (docs/design/36): `ScalarStmt::Let`
// + `Const`/`Var`/`BinOp`/`UnaryOp`/`MethodCall`/`Select`/`Cast`. anything else
// (stencils, generic-dim loops, reductions) is REJECTED (`JitError::Unsupported`) so the
// caller falls back to the interpreter — never a miscompile.
//
// bit-identity with the interpreter (the oracle gate `f64-interp == cranelift`):
//   - arithmetic is plain IEEE `fadd/fmul/...`; Cranelift does NOT auto-contract to FMA,
//     so `a*b + c` matches the interpreter's separate mul+add;
//   - every `MethodCall` is routed through a Rust shim (`extern "C" fn` wrapping `x.sin()`
//     etc.), so the JIT calls the EXACT std functions the interpreter does — no `libm`
//     dependency, no platform-libm divergence to reason about.
// =============================================================================

use std::collections::HashMap;

use cranelift_codegen::ir::condcodes::{FloatCC, IntCC};
use cranelift_codegen::ir::{AbiParam, Block, InstBuilder, MemFlags, Signature, Value, types};
use cranelift_codegen::settings::{self, Configurable};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext, Variable};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{FuncId, Linkage, Module};

use symbi_ir::ElementTy;
use symbi_ir::dim::DimExpr;
use symbi_ir::graph::ConstValue;
use symbi_ir::passes::scalarize::{BinaryKind, LoweredFn, ScalarExpr, ScalarStmt, UnaryKind};

/// a named local: an immutable SSA `Value` (a `Let` binding) or a Cranelift `Variable` (a
/// mutable accumulator — `LetMut` — that crosses loop iterations / blocks; Cranelift inserts
/// the phi). the pointwise path uses only `Val`.
#[derive(Clone, Copy)]
enum LocalSlot {
    Val(Value),
    Var(Variable),
}

/// did a statement sequence fall through, or terminate the block (a `Break` jump)?
#[derive(Clone, Copy, PartialEq)]
enum Flow {
    Fallthrough,
    Terminated,
}

/// the JIT cannot compile this `LoweredFn` (an out-of-subset node). the caller falls
/// back to the interpreter — this is a clean refusal, never a wrong kernel.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum JitError {
    /// a node outside the v1 source subset (stencil/loop/reduction/exotic cast).
    Unsupported(String),
    /// a `Var` with no binding (malformed `LoweredFn`).
    UnboundVar(String),
    /// Cranelift codegen failed (should not happen for the validated subset).
    Codegen(String),
}

impl std::fmt::Display for JitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            JitError::Unsupported(s) => write!(f, "jit: unsupported node: {s}"),
            JitError::UnboundVar(s) => write!(f, "jit: unbound var '{s}'"),
            JitError::Codegen(s) => write!(f, "jit: codegen error: {s}"),
        }
    }
}
impl std::error::Error for JitError {}

// ---- the method shims: extern "C" wrappers over std f64, so the JIT calls the EXACT
//      functions the interpreter (`eval_method`) does. one shim per non-native method. ----

macro_rules! shim1 {
    ($name:ident, $m:ident) => {
        extern "C" fn $name(x: f64) -> f64 {
            x.$m()
        }
    };
}
macro_rules! shim2 {
    ($name:ident, $m:ident) => {
        extern "C" fn $name(a: f64, b: f64) -> f64 {
            a.$m(b)
        }
    };
}
shim1!(sh_sin, sin);
shim1!(sh_cos, cos);
shim1!(sh_tan, tan);
shim1!(sh_asin, asin);
shim1!(sh_acos, acos);
shim1!(sh_atan, atan);
shim1!(sh_sinh, sinh);
shim1!(sh_cosh, cosh);
shim1!(sh_tanh, tanh);
shim1!(sh_asinh, asinh);
shim1!(sh_acosh, acosh);
shim1!(sh_atanh, atanh);
shim1!(sh_exp, exp);
shim1!(sh_exp2, exp2);
shim1!(sh_ln, ln);
shim1!(sh_log2, log2);
shim1!(sh_log10, log10);
shim1!(sh_round, round);
// no min/max/abs shims: they are emitted inline as fcmp + select (a ternary)
// in translate_expr, matching cuda / interp / the Numeric carrier.
shim2!(sh_atan2, atan2);
shim2!(sh_powf, powf);
shim2!(sh_hypot, hypot);
extern "C" fn sh_powi(a: f64, b: f64) -> f64 {
    a.powi(b as i32)
}
extern "C" fn sh_div_euclid(a: f64, b: f64) -> f64 {
    (a / b).floor()
}

/// `(symbol, arity-incl-receiver, fn-ptr)` for every shimmed method. `arity` is the total
/// f64 args the CLIF call passes (receiver + method args).
fn shim_table() -> &'static [(&'static str, usize, *const u8)] {
    &[
        ("sin", 1, sh_sin as *const u8),
        ("cos", 1, sh_cos as *const u8),
        ("tan", 1, sh_tan as *const u8),
        ("asin", 1, sh_asin as *const u8),
        ("acos", 1, sh_acos as *const u8),
        ("atan", 1, sh_atan as *const u8),
        ("sinh", 1, sh_sinh as *const u8),
        ("cosh", 1, sh_cosh as *const u8),
        ("tanh", 1, sh_tanh as *const u8),
        ("asinh", 1, sh_asinh as *const u8),
        ("acosh", 1, sh_acosh as *const u8),
        ("atanh", 1, sh_atanh as *const u8),
        ("exp", 1, sh_exp as *const u8),
        ("exp2", 1, sh_exp2 as *const u8),
        ("ln", 1, sh_ln as *const u8),
        ("log2", 1, sh_log2 as *const u8),
        ("log10", 1, sh_log10 as *const u8),
        ("round", 1, sh_round as *const u8),
        ("atan2", 2, sh_atan2 as *const u8),
        ("powf", 2, sh_powf as *const u8),
        ("hypot", 2, sh_hypot as *const u8),
        ("powi", 2, sh_powi as *const u8),
        ("div_euclid", 2, sh_div_euclid as *const u8),
    ]
}

/// a JIT-compiled `LoweredFn`. owns the `JITModule` (keeps the code mapped) + the native
/// entry point `fn(inputs: *const f64, outputs: *mut f64)`.
pub struct CompiledFn {
    // SAFETY: the module is kept ONLY to keep the finalized code memory mapped; after
    // `finalize_definitions` it is read-only and never touched again. the entry point is a
    // bare code pointer (no state), so it is sound to call from many threads concurrently.
    _module: JITModule,
    func: unsafe extern "C" fn(*const f64, *mut f64),
    n_in: usize,
    n_out: usize,
}

// SAFETY: see the field comment — the only thing shared is an immutable code pointer.
unsafe impl Send for CompiledFn {}
unsafe impl Sync for CompiledFn {}

impl CompiledFn {
    /// number of inputs (params) the function expects, in `LoweredFn::params` order.
    pub fn n_inputs(&self) -> usize {
        self.n_in
    }
    /// number of outputs (result components).
    pub fn n_outputs(&self) -> usize {
        self.n_out
    }

    /// evaluate: `inputs` in param order, results written into `out`.
    #[inline]
    pub fn call(&self, inputs: &[f64], out: &mut [f64]) {
        assert_eq!(
            inputs.len(),
            self.n_in,
            "CompiledFn::call: input arity mismatch"
        );
        assert_eq!(
            out.len(),
            self.n_out,
            "CompiledFn::call: output arity mismatch"
        );
        // SAFETY: the function reads exactly `n_in` f64s from `inputs` and writes exactly
        // `n_out` f64s to `out`; the asserts above guarantee both slices are large enough.
        unsafe { (self.func)(inputs.as_ptr(), out.as_mut_ptr()) }
    }
}

/// compile a scalarized `LoweredFn` to native code. params with `array_len` (rank-1 array
/// inputs) are rejected — v1 is the scalar-param source subset.
pub fn compile(lowered: &LoweredFn) -> Result<CompiledFn, JitError> {
    for p in &lowered.params {
        if p.array_len.is_some() {
            return Err(JitError::Unsupported(format!("array param '{}'", p.name)));
        }
    }
    let n_in = lowered.params.len();
    let n_out = lowered.results.len();

    // ---- ISA + JIT module, with every shim symbol registered ----
    let mut flags = settings::builder();
    flags.set("use_colocated_libcalls", "false").unwrap();
    flags.set("is_pic", "false").unwrap();
    // optimize: the JIT'd kernel runs the WHOLE per-cell body (godunov + source) per cell, so the
    // codegen quality matters — Cranelift defaults to `opt_level=none`, which leaves it well behind
    // the AOT rustc `-O` kernels. "speed" enables GVN / LICM / redundant-load elimination / better
    // regalloc WITHOUT FP reassociation or auto-FMA (Cranelift never contracts FMA), so the
    // bit-identity oracle still holds (interp == cranelift).
    flags.set("opt_level", "speed").unwrap();
    let isa = cranelift_native::builder()
        .map_err(|e| JitError::Codegen(format!("native isa: {e}")))?
        .finish(settings::Flags::new(flags))
        .map_err(|e| JitError::Codegen(format!("isa finish: {e}")))?;
    let mut jb = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());
    for (name, _, ptr) in shim_table() {
        jb.symbol(*name, *ptr);
    }
    let mut module = JITModule::new(jb);
    let ptr_ty = module.target_config().pointer_type();

    // declare every shim as an imported function; cache FuncId by method name.
    let mut shim_ids: HashMap<&'static str, (FuncId, usize)> = HashMap::new();
    for (name, arity, _) in shim_table() {
        let mut sig = module.make_signature();
        for _ in 0..*arity {
            sig.params.push(AbiParam::new(types::F64));
        }
        sig.returns.push(AbiParam::new(types::F64));
        let id = module
            .declare_function(name, Linkage::Import, &sig)
            .map_err(|e| JitError::Codegen(format!("declare shim '{name}': {e}")))?;
        shim_ids.insert(name, (id, *arity));
    }

    // ---- the entry function: fn(inputs: *const f64, outputs: *mut f64) ----
    let mut ctx = module.make_context();
    ctx.func.signature = Signature::new(module.target_config().default_call_conv);
    ctx.func.signature.params.push(AbiParam::new(ptr_ty)); // inputs
    ctx.func.signature.params.push(AbiParam::new(ptr_ty)); // outputs

    let mut fctx = FunctionBuilderContext::new();
    let mut b = FunctionBuilder::new(&mut ctx.func, &mut fctx);
    let entry = b.create_block();
    b.append_block_params_for_function_params(entry);
    b.switch_to_block(entry);
    b.seal_block(entry);
    let inputs_ptr = b.block_params(entry)[0];
    let outputs_ptr = b.block_params(entry)[1];

    // import the shims into THIS function -> FuncRef by name.
    let mut shim_refs: HashMap<&'static str, (cranelift_codegen::ir::FuncRef, usize)> =
        HashMap::new();
    for (name, (id, arity)) in &shim_ids {
        let fref = module.declare_func_in_func(*id, b.func);
        shim_refs.insert(name, (fref, *arity));
    }

    // load params into the name->slot map.
    let mut vars: HashMap<String, LocalSlot> = HashMap::new();
    for (i, p) in lowered.params.iter().enumerate() {
        let off = (i * 8) as i32;
        let v = b
            .ins()
            .load(types::F64, MemFlags::trusted(), inputs_ptr, off);
        vars.insert(p.name.clone(), LocalSlot::Val(v));
    }

    // translate the body (Let-bindings only in the pointwise path).
    for stmt in &lowered.body {
        match stmt {
            ScalarStmt::Let { name, value, .. } => {
                let v = translate_expr(&mut b, value, &vars, &shim_refs, None)?;
                vars.insert(name.clone(), LocalSlot::Val(v));
            }
            other => {
                return Err(JitError::Unsupported(format!("statement {other:?}")));
            }
        }
    }

    // translate + store each result component.
    for (k, res) in lowered.results.iter().enumerate() {
        let v = translate_expr(&mut b, res, &vars, &shim_refs, None)?;
        let off = (k * 8) as i32;
        b.ins().store(MemFlags::trusted(), v, outputs_ptr, off);
    }

    b.ins().return_(&[]);
    b.finalize();

    // ---- define + finalize + extract the code pointer ----
    let func_id = module
        .declare_function("symbi_source", Linkage::Export, &ctx.func.signature)
        .map_err(|e| JitError::Codegen(format!("declare entry: {e}")))?;
    module
        .define_function(func_id, &mut ctx)
        .map_err(|e| JitError::Codegen(format!("define entry: {e:?}")))?;
    module.clear_context(&mut ctx);
    module
        .finalize_definitions()
        .map_err(|e| JitError::Codegen(format!("finalize: {e}")))?;
    let code = module.get_finalized_function(func_id);
    // SAFETY: `code` is the finalized entry whose signature is `(*const f64, *mut f64) -> ()`.
    let func: unsafe extern "C" fn(*const f64, *mut f64) = unsafe { std::mem::transmute(code) };

    Ok(CompiledFn {
        _module: module,
        func,
        n_in,
        n_out,
    })
}

/// the kernel-stencil context for `FieldLoadAt`: the input buffer base array + per-buffer
/// index, the shared `(lo, extent)` layout as i64 registers, the integer `_coord_N` registers,
/// and the pointer type. `None` in the pointwise path (where `FieldLoadAt` cannot appear).
struct StencilCtx<'a> {
    /// `*const *const f64` — the input buffer bases, by `field_idx`.
    in_bufs: Value,
    /// `field_key -> input buffer index`.
    field_idx: &'a HashMap<String, usize>,
    /// shared layout, per axis (i64 registers): `flat_index` matches `interp::flat_index`.
    lo: &'a [Value],
    extent: &'a [Value],
    /// integer `_coord_N` registers, by name (for the coord arithmetic in load components).
    coord_vars: &'a HashMap<String, Value>,
    /// body `Let`/`LetMut` definitions by name — so a load index that references a CSE'd integer
    /// offset var (e.g., `__cse_1`, which CSE hoists to function scope) resolves by recursing into
    /// its defining expression. index exprs are pure coord/const integer arithmetic.
    let_defs: &'a HashMap<String, &'a ScalarExpr>,
    ndim: usize,
    ptr_ty: cranelift_codegen::ir::Type,
}

/// collect every `Let`/`LetMut` binding (name -> defining expr) reachable in a kernel body, for the
/// load-index translator to resolve a CSE'd integer offset var by recursion. recurses into
/// `For`/`If`/`Scope` bodies; a non-integer def reached via an index var rejects naturally downstream.
fn collect_let_defs<'a>(stmts: &'a [ScalarStmt], out: &mut HashMap<String, &'a ScalarExpr>) {
    for s in stmts {
        match s {
            ScalarStmt::Let { name, value, .. } => {
                out.insert(name.clone(), value);
            }
            ScalarStmt::LetMut { name, init, .. } => {
                out.insert(name.clone(), init);
            }
            ScalarStmt::For { body, .. } => collect_let_defs(body, out),
            ScalarStmt::If { then_body, .. } => collect_let_defs(then_body, out),
            ScalarStmt::Scope { body, .. } => collect_let_defs(body, out),
            _ => {}
        }
    }
}

/// the flat index, axis-0-fastest ("physical-x-fastest") to EXACTLY match `interp::flat_index`
/// AND the real `symbi_grid` `Field`/`View` storage (`symbi_algebra::strides_from_extent`):
/// `stride[0]=1, stride[ax]=stride[ax-1]*extent[ax-1]`. axis 0 is fastest-varying in memory. this
/// is the convention the AOT kernels read real fields with — last-axis-fastest would transpose
/// axes for D>=2 and mis-read every off-diagonal stencil neighbour.
fn emit_flat_index(b: &mut FunctionBuilder, coord: &[Value], ctx: &StencilCtx) -> Value {
    let mut idx = b.ins().iconst(types::I64, 0);
    let mut stride = b.ins().iconst(types::I64, 1);
    for ax in 0..ctx.ndim {
        let d = b.ins().isub(coord[ax], ctx.lo[ax]);
        let term = b.ins().imul(d, stride);
        idx = b.ins().iadd(idx, term);
        stride = b.ins().imul(stride, ctx.extent[ax]);
    }
    idx
}

/// translate a `FieldLoadAt` COMPONENT (integer coord arithmetic) to an i64 `Value`. the
/// component is `_coord_N (+/-/* const offset)` — exact integers, so this matches the
/// interpreter's `f64.round() as i64` exactly while staying in integer registers.
fn translate_index_expr(
    b: &mut FunctionBuilder,
    expr: &ScalarExpr,
    ctx: &StencilCtx,
) -> Result<Value, JitError> {
    Ok(match expr {
        ScalarExpr::Const(ConstValue::I32(v)) => b.ins().iconst(types::I64, *v as i64),
        ScalarExpr::Const(ConstValue::U32(v)) => b.ins().iconst(types::I64, *v as i64),
        ScalarExpr::Var(name) => {
            if let Some(v) = ctx.coord_vars.get(name) {
                *v
            } else if let Some(def) = ctx.let_defs.get(name) {
                // a CSE'd integer offset (e.g., `__cse_1`): resolve by translating its definition.
                translate_index_expr(b, def, ctx)?
            } else {
                return Err(JitError::Unsupported(format!(
                    "non-coord var '{name}' in load index"
                )));
            }
        }
        ScalarExpr::BinOp(op, l, r) => {
            let lv = translate_index_expr(b, l, ctx)?;
            let rv = translate_index_expr(b, r, ctx)?;
            match op {
                BinaryKind::Add => b.ins().iadd(lv, rv),
                BinaryKind::Sub => b.ins().isub(lv, rv),
                BinaryKind::Mul => b.ins().imul(lv, rv),
                _ => return Err(JitError::Unsupported(format!("op {op:?} in load index"))),
            }
        }
        other => {
            return Err(JitError::Unsupported(format!(
                "expr {other:?} in load index"
            )));
        }
    })
}

/// translate one `ScalarExpr` to a CLIF `Value` (f64, or an i8 bool for comparisons /
/// logical ops, consumed only by `Select` / bitwise / `Not`). `stencil` is `Some` in the
/// kernel path (enables `FieldLoadAt`).
fn translate_expr(
    b: &mut FunctionBuilder,
    expr: &ScalarExpr,
    vars: &HashMap<String, LocalSlot>,
    shims: &HashMap<&'static str, (cranelift_codegen::ir::FuncRef, usize)>,
    stencil: Option<&StencilCtx>,
) -> Result<Value, JitError> {
    Ok(match expr {
        ScalarExpr::Const(c) => match c {
            ConstValue::F64(v) => b.ins().f64const(*v),
            ConstValue::F32(v) => b.ins().f64const(*v as f64),
            ConstValue::I32(v) => b.ins().iconst(types::I32, *v as i64),
            ConstValue::U32(v) => b.ins().iconst(types::I32, *v as i64),
            ConstValue::Bool(v) => b.ins().iconst(types::I8, *v as i64),
        },
        ScalarExpr::Var(name) => match vars
            .get(name)
            .ok_or_else(|| JitError::UnboundVar(name.clone()))?
        {
            LocalSlot::Val(v) => *v,
            LocalSlot::Var(var) => b.use_var(*var),
        },
        ScalarExpr::UnaryOp(op, x) => {
            let xv = translate_expr(b, x, vars, shims, stencil)?;
            match op {
                UnaryKind::Neg => b.ins().fneg(xv),
                // logical NOT on an i8 bool: xor with 1.
                UnaryKind::Not => {
                    let one = b.ins().iconst(types::I8, 1);
                    b.ins().bxor(xv, one)
                }
            }
        }
        ScalarExpr::BinOp(op, l, r) => {
            let lv = translate_expr(b, l, vars, shims, stencil)?;
            let rv = translate_expr(b, r, vars, shims, stencil)?;
            match op {
                BinaryKind::Add => b.ins().fadd(lv, rv),
                BinaryKind::Sub => b.ins().fsub(lv, rv),
                BinaryKind::Mul => b.ins().fmul(lv, rv),
                BinaryKind::Div => b.ins().fdiv(lv, rv),
                BinaryKind::Lt => b.ins().fcmp(FloatCC::LessThan, lv, rv),
                BinaryKind::Le => b.ins().fcmp(FloatCC::LessThanOrEqual, lv, rv),
                BinaryKind::Gt => b.ins().fcmp(FloatCC::GreaterThan, lv, rv),
                BinaryKind::Ge => b.ins().fcmp(FloatCC::GreaterThanOrEqual, lv, rv),
                BinaryKind::Eq => b.ins().fcmp(FloatCC::Equal, lv, rv),
                BinaryKind::Ne => b.ins().fcmp(FloatCC::NotEqual, lv, rv),
                BinaryKind::BitAnd => b.ins().band(lv, rv),
                BinaryKind::BitOr => b.ins().bor(lv, rv),
                BinaryKind::BitXor => b.ins().bxor(lv, rv),
            }
        }
        ScalarExpr::Select { cond, then, else_ } => {
            let c = translate_expr(b, cond, vars, shims, stencil)?;
            let t = translate_expr(b, then, vars, shims, stencil)?;
            let e = translate_expr(b, else_, vars, shims, stencil)?;
            b.ins().select(c, t, e)
        }
        ScalarExpr::MethodCall {
            receiver,
            method,
            args,
        } => {
            let recv = translate_expr(b, receiver, vars, shims, stencil)?;
            // native, IEEE-exact, bit-identical to std: sqrt/floor/ceil/trunc.
            match method.as_str() {
                "sqrt" => return Ok(b.ins().sqrt(recv)),
                "floor" => return Ok(b.ins().floor(recv)),
                "ceil" => return Ok(b.ins().ceil(recv)),
                "trunc" => return Ok(b.ins().trunc(recv)),
                // abs/min/max as a TERNARY (fcmp + select), NOT
                // libdevice fabs/fmin/fmax — bit-matches the cuda emit, the interp,
                // and the f64/f32 `Numeric` carrier at NaN / signed-zero (tier-1
                // #2b). CLIF select is a value, not a lexical scope, so there is no
                // debuginfo blow-up (the reason the CPU emit keeps these as method
                // calls rather than lowering to scoped `if`-selects in scalarize).
                "abs" => {
                    let zero = b.ins().f64const(0.0);
                    let neg = b.ins().fneg(recv);
                    let c = b.ins().fcmp(FloatCC::LessThan, recv, zero);
                    return Ok(b.ins().select(c, neg, recv));
                }
                "min" => {
                    let arg = translate_expr(b, &args[0], vars, shims, stencil)?;
                    let c = b.ins().fcmp(FloatCC::LessThan, recv, arg);
                    return Ok(b.ins().select(c, recv, arg));
                }
                "max" => {
                    let arg = translate_expr(b, &args[0], vars, shims, stencil)?;
                    let c = b.ins().fcmp(FloatCC::GreaterThan, recv, arg);
                    return Ok(b.ins().select(c, recv, arg));
                }
                // bool-returning predicates, inline (consumed by Select):
                "is_nan" => return Ok(b.ins().fcmp(FloatCC::NotEqual, recv, recv)),
                "is_finite" => {
                    let a = b.ins().fabs(recv);
                    let inf = b.ins().f64const(f64::INFINITY);
                    return Ok(b.ins().fcmp(FloatCC::LessThan, a, inf));
                }
                _ => {}
            }
            // everything else goes through the std shim (bit-identical to the interpreter).
            let (fref, arity) = *shims
                .get(method.as_str())
                .ok_or_else(|| JitError::Unsupported(format!("method '.{method}()'")))?;
            let mut call_args = Vec::with_capacity(arity);
            call_args.push(recv);
            for a in args {
                call_args.push(translate_expr(b, a, vars, shims, stencil)?);
            }
            if call_args.len() != arity {
                return Err(JitError::Unsupported(format!(
                    "method '.{method}()' arity {} != {arity}",
                    call_args.len()
                )));
            }
            let call = b.ins().call(fref, &call_args);
            b.inst_results(call)[0]
        }
        ScalarExpr::Cast { to, value } => {
            let v = translate_expr(b, value, vars, shims, stencil)?;
            let vty = b.func.dfg.value_type(v);
            match to {
                // no-op when the source already carries the target type: `_coord_N` is pre-converted
                // to f64 in the kernel var seed, so an int->f64 cast of it would re-convert an f64.
                ElementTy::F64 | ElementTy::F32 => {
                    if vty == types::F64 { v } else { b.ins().fcvt_from_sint(types::F64, v) }
                }
                ElementTy::I32 | ElementTy::U32 => {
                    if vty.is_int() { v } else { b.ins().fcvt_to_sint(types::I32, v) }
                }
                ElementTy::Bool => return Err(JitError::Unsupported("cast to bool".into())),
            }
        }
        ScalarExpr::IndexInto { container, .. } => {
            return Err(JitError::Unsupported(format!("index into '{container}'")));
        }
        ScalarExpr::FieldLoadAt {
            field_key,
            components,
        } => {
            let ctx = stencil
                .ok_or_else(|| JitError::Unsupported("FieldLoadAt outside a kernel".into()))?;
            // integer stencil coord -> matching flat index -> load from the field's buffer.
            let coords: Vec<Value> = components
                .iter()
                .map(|c| translate_index_expr(b, c, ctx))
                .collect::<Result<_, _>>()?;
            if coords.len() != ctx.ndim {
                return Err(JitError::Unsupported(format!(
                    "FieldLoadAt '{field_key}': {} components != ndim {}",
                    coords.len(),
                    ctx.ndim,
                )));
            }
            let idx = emit_flat_index(b, &coords, ctx);
            let fi = *ctx.field_idx.get(field_key).ok_or_else(|| {
                JitError::Unsupported(format!("FieldLoadAt unknown field '{field_key}'"))
            })?;
            let psz = ctx.ptr_ty.bytes() as i32;
            let base = b.ins().load(
                ctx.ptr_ty,
                MemFlags::trusted(),
                ctx.in_bufs,
                fi as i32 * psz,
            );
            let byte_off = b.ins().imul_imm(idx, 8);
            let addr = b.ins().iadd(base, byte_off);
            b.ins().load(types::F64, MemFlags::trusted(), addr, 0)
        }
        ScalarExpr::FreeCall { name, .. } => {
            return Err(JitError::Unsupported(format!("free call '{name}'")));
        }
    })
}

/// the CLIF type for a lowered element type.
fn clif_ty(e: ElementTy) -> cranelift_codegen::ir::Type {
    match e {
        ElementTy::F64 | ElementTy::F32 => types::F64,
        ElementTy::I32 | ElementTy::U32 => types::I64,
        ElementTy::Bool => types::I8,
    }
}

/// resolve a name that MUST be a mutable `Variable` (target of `Assign`/`CompoundAssign`).
fn expect_var(vars: &HashMap<String, LocalSlot>, name: &str) -> Result<Variable, JitError> {
    match vars.get(name) {
        Some(LocalSlot::Var(v)) => Ok(*v),
        Some(LocalSlot::Val(_)) => Err(JitError::Unsupported(format!(
            "assign to immutable '{name}'"
        ))),
        None => Err(JitError::UnboundVar(name.to_string())),
    }
}

/// translate a statement sequence into the current block — control flow (`For`/`If`/`Break`)
/// via CLIF blocks, mutable accumulators (`LetMut`) via Cranelift `Variable`s (Cranelift inserts
/// the loop phi on `seal`). `loop_exit` = the enclosing loop's exit block (for `Break`);
/// `next_var` allocates fresh `Variable` indices. returns whether the sequence fell through.
#[allow(clippy::too_many_arguments)]
fn translate_stmts(
    b: &mut FunctionBuilder,
    stmts: &[ScalarStmt],
    vars: &mut HashMap<String, LocalSlot>,
    shims: &HashMap<&'static str, (cranelift_codegen::ir::FuncRef, usize)>,
    stencil: &StencilCtx,
    next_var: &mut u32,
    loop_exit: Option<Block>,
) -> Result<Flow, JitError> {
    for stmt in stmts {
        match stmt {
            ScalarStmt::Let {
                name,
                element,
                value,
            } => {
                // integer-typed lets are CSE'd stencil INDEX offsets (e.g., `__cse_1 = _coord_0 + 1`),
                // used ONLY inside `FieldLoadAt` components — which the index translator resolves
                // separately via `let_defs` in the integer domain. translating them here as f64 body
                // statements would emit `fadd(f64_coord, i32_const)` (a verifier type error). skip
                // them; a (hypothetical) float use elsewhere hits `UnboundVar` -> a clean reject, not
                // a miscompile.
                if matches!(element, ElementTy::I32 | ElementTy::U32) {
                    continue;
                }
                let v = translate_expr(b, value, vars, shims, Some(stencil))?;
                vars.insert(name.clone(), LocalSlot::Val(v));
            }
            ScalarStmt::LetMut {
                name,
                element,
                init,
            } => {
                let var = Variable::from_u32(*next_var);
                *next_var += 1;
                b.declare_var(var, clif_ty(*element));
                let v = translate_expr(b, init, vars, shims, Some(stencil))?;
                b.def_var(var, v);
                vars.insert(name.clone(), LocalSlot::Var(var));
            }
            ScalarStmt::Assign { name, value } => {
                let var = expect_var(vars, name)?;
                let v = translate_expr(b, value, vars, shims, Some(stencil))?;
                b.def_var(var, v);
            }
            ScalarStmt::CompoundAssign { name, op, value } => {
                let var = expect_var(vars, name)?;
                let cur = b.use_var(var);
                let rhs = translate_expr(b, value, vars, shims, Some(stencil))?;
                let v = match op {
                    BinaryKind::Add => b.ins().fadd(cur, rhs),
                    BinaryKind::Mul => b.ins().fmul(cur, rhs),
                    BinaryKind::BitOr => b.ins().bor(cur, rhs),
                    BinaryKind::BitAnd => b.ins().band(cur, rhs),
                    BinaryKind::BitXor => b.ins().bxor(cur, rhs),
                    other => {
                        return Err(JitError::Unsupported(format!(
                            "compound-assign op {other:?}"
                        )));
                    }
                };
                b.def_var(var, v);
            }
            ScalarStmt::For { iter, bound, body } => {
                let DimExpr::Literal(n) = bound;
                let n = *n as i64;
                let iter_var = Variable::from_u32(*next_var);
                *next_var += 1;
                b.declare_var(iter_var, types::I64);
                let zero = b.ins().iconst(types::I64, 0);
                b.def_var(iter_var, zero);

                let header = b.create_block();
                let body_blk = b.create_block();
                let exit = b.create_block();
                b.ins().jump(header, &[]);
                b.switch_to_block(header);
                let i = b.use_var(iter_var);
                let nc = b.ins().iconst(types::I64, n);
                let cond = b.ins().icmp(IntCC::SignedLessThan, i, nc);
                b.ins().brif(cond, body_blk, &[], exit, &[]);
                b.seal_block(body_blk);
                b.switch_to_block(body_blk);

                // expose the loop index as an f64 local (for the rare body that reads `iter`).
                let iv = b.use_var(iter_var);
                let i_f = b.ins().fcvt_from_sint(types::F64, iv);
                let shadowed = vars.insert(iter.clone(), LocalSlot::Val(i_f));
                let flow = translate_stmts(b, body, vars, shims, stencil, next_var, Some(exit))?;
                if flow == Flow::Fallthrough {
                    let i2 = b.use_var(iter_var);
                    let inc = b.ins().iadd_imm(i2, 1);
                    b.def_var(iter_var, inc);
                    b.ins().jump(header, &[]);
                }
                match shadowed {
                    Some(s) => {
                        vars.insert(iter.clone(), s);
                    }
                    None => {
                        vars.remove(iter);
                    }
                }
                b.seal_block(header);
                b.seal_block(exit);
                b.switch_to_block(exit);
            }
            ScalarStmt::If { cond, then_body } => {
                let c = translate_expr(b, cond, vars, shims, Some(stencil))?;
                let then_blk = b.create_block();
                let merge = b.create_block();
                b.ins().brif(c, then_blk, &[], merge, &[]);
                b.seal_block(then_blk);
                b.switch_to_block(then_blk);
                let flow =
                    translate_stmts(b, then_body, vars, shims, stencil, next_var, loop_exit)?;
                if flow == Flow::Fallthrough {
                    b.ins().jump(merge, &[]);
                }
                b.seal_block(merge);
                b.switch_to_block(merge);
            }
            ScalarStmt::Break => {
                let exit = loop_exit
                    .ok_or_else(|| JitError::Unsupported("Break outside a loop".into()))?;
                b.ins().jump(exit, &[]);
                return Ok(Flow::Terminated);
            }
            ScalarStmt::Scope {
                name, body, result, ..
            } => {
                // a bounded-pressure scope is a renderer register-pressure HINT; the JIT does its
                // own regalloc, so flatten it: run the body inline (its lets are SSA + dominate),
                // then bind `name = result`.
                let flow = translate_stmts(b, body, vars, shims, stencil, next_var, loop_exit)?;
                if flow == Flow::Terminated {
                    return Ok(Flow::Terminated);
                }
                let v = translate_expr(b, result, vars, shims, Some(stencil))?;
                vars.insert(name.clone(), LocalSlot::Val(v));
            }
            ScalarStmt::IfElse { outs, cond, then_body, else_body } => {
                // a data-dependent branch where ONLY the taken arm runs. each result slot is a mutable
                // `Variable` declared BEFORE the branch; every arm ends with `Assign { outs[j], .. }`, so
                // the slot is defined on both paths and live at the merge (Cranelift inserts the phi on
                // seal). arm-internal lets die at the arm's block — only the taken arm's ops execute.
                for (name, element) in outs {
                    let var = Variable::from_u32(*next_var);
                    *next_var += 1;
                    b.declare_var(var, clif_ty(*element));
                    vars.insert(name.clone(), LocalSlot::Var(var));
                }
                let c = translate_expr(b, cond, vars, shims, Some(stencil))?;
                let then_blk = b.create_block();
                let else_blk = b.create_block();
                let merge = b.create_block();
                b.ins().brif(c, then_blk, &[], else_blk, &[]);

                b.seal_block(then_blk);
                b.switch_to_block(then_blk);
                let tflow = translate_stmts(b, then_body, vars, shims, stencil, next_var, loop_exit)?;
                if tflow == Flow::Fallthrough {
                    b.ins().jump(merge, &[]);
                }

                b.seal_block(else_blk);
                b.switch_to_block(else_blk);
                let eflow = translate_stmts(b, else_body, vars, shims, stencil, next_var, loop_exit)?;
                if eflow == Flow::Fallthrough {
                    b.ins().jump(merge, &[]);
                }

                b.seal_block(merge);
                b.switch_to_block(merge);
            }
        }
    }
    Ok(Flow::Fallthrough)
}

// =============================================================================
// stencil KERNELS: JIT a scalarized kernel — cell loads + `FieldLoadAt`
// stencil reads + scalars -> multi-output, mapped over a domain. the v1-kernel subset is
// `Let` bodies (no `For`/reductions yet). all buffers share one `(lo, extent)` layout.
// =============================================================================

/// a JIT-compiled stencil kernel: a per-cell `extern "C"` fn over field buffers + the metadata
/// to drive it over a domain. `Send + Sync` (a bare code pointer + counts).
pub struct CompiledKernel {
    // SAFETY: same as `CompiledFn` — the module is kept only to keep the code mapped; the entry
    // is a stateless code pointer, sound to call concurrently for disjoint cells.
    _module: JITModule,
    cell: unsafe extern "C" fn(
        *const i64,        // coord [ndim]
        *const i64,        // lo [ndim]   (shared layout)
        *const i64,        // extent [ndim]
        *const *const f64, // input buffer bases [n_in]
        *const f64,        // scalars [n_scalar]
        *mut *mut f64,     // output buffer bases [n_out]
    ),
    ndim: usize,
    n_in: usize,
    n_out: usize,
    n_scalar: usize,
}

unsafe impl Send for CompiledKernel {}
unsafe impl Sync for CompiledKernel {}

/// raw input/output buffer bases shared across rayon threads in [`CompiledKernel::run_parallel`].
/// SAFETY of the `Send + Sync` impls: inputs are read-only; each cell writes ONLY its own (disjoint)
/// flat output index, so concurrent access through these bases is race-free by construction.
struct SharedBufs {
    in_ptrs: Vec<*const f64>,
    out_ptrs: Vec<*mut f64>,
}
unsafe impl Send for SharedBufs {}
unsafe impl Sync for SharedBufs {}

impl CompiledKernel {
    /// map the per-cell kernel over `[dom_lo, dom_lo + grid)`, in the SAME order as
    /// `Cpu::run_kernel`. all input/output buffers share the `(lo, extent)` layout (v1-kernel).
    #[allow(clippy::too_many_arguments)]
    pub fn run(
        &self,
        grid_sizes: &[u32],
        dom_los: &[i32],
        lo: &[i32],
        extent: &[u32],
        in_bufs: &[&[f64]],
        scalars: &[f64],
        out_bufs: &mut [&mut [f64]],
    ) {
        assert_eq!(grid_sizes.len(), self.ndim);
        assert_eq!(in_bufs.len(), self.n_in);
        assert_eq!(out_bufs.len(), self.n_out);
        assert_eq!(scalars.len(), self.n_scalar);
        let lo_i: Vec<i64> = lo.iter().map(|&x| x as i64).collect();
        let ext_i: Vec<i64> = extent.iter().map(|&x| x as i64).collect();
        let in_ptrs: Vec<*const f64> = in_bufs.iter().map(|b| b.as_ptr()).collect();
        let mut out_ptrs: Vec<*mut f64> = out_bufs.iter_mut().map(|b| b.as_mut_ptr()).collect();
        let ndim = self.ndim;
        let total: usize = grid_sizes.iter().map(|&g| g as usize).product();
        for flat in 0..total {
            let mut coord = vec![0i64; ndim];
            let mut rem = flat;
            for ax in (0..ndim).rev() {
                let g = grid_sizes[ax] as usize;
                coord[ax] = dom_los[ax] as i64 + (rem % g) as i64;
                rem /= g;
            }
            // SAFETY: the kernel reads n_in bases + n_scalar scalars and writes n_out bases at the
            // flat index of `coord` within the shared (lo, extent) layout; the asserts size them.
            unsafe {
                (self.cell)(
                    coord.as_ptr(),
                    lo_i.as_ptr(),
                    ext_i.as_ptr(),
                    in_ptrs.as_ptr(),
                    scalars.as_ptr(),
                    out_ptrs.as_mut_ptr(),
                );
            }
        }
    }

    /// the parallel twin of [`Self::run`]: maps the per-cell kernel over the domain with rayon.
    /// each flat cell index reconstructs a UNIQUE coord, and each cell writes ONLY its own flat
    /// output index (bijective coord->index), so the writes are DISJOINT and the shared `*mut`
    /// output bases are race-free BY CONSTRUCTION. the result is bit-identical to [`Self::run`]:
    /// reordering independent per-cell ops changes no single cell's value. this is the production
    /// driver for the fused godunov+source kernel; `run` stays the serial oracle.
    #[allow(clippy::too_many_arguments)]
    pub fn run_parallel(
        &self,
        grid_sizes: &[u32],
        dom_los: &[i32],
        lo: &[i32],
        extent: &[u32],
        in_bufs: &[&[f64]],
        scalars: &[f64],
        out_bufs: &mut [&mut [f64]],
    ) {
        assert_eq!(in_bufs.len(), self.n_in);
        assert_eq!(out_bufs.len(), self.n_out);
        let in_bases: Vec<*const f64> = in_bufs.iter().map(|b| b.as_ptr()).collect();
        let out_bases: Vec<*mut f64> = out_bufs.iter_mut().map(|b| b.as_mut_ptr()).collect();
        // SAFETY: `run_parallel` is given DISTINCT (non-aliasing) input + output slices by the
        // borrow checker, so the raw bases below alias nothing. (the in-place dispatch path uses
        // `run_parallel_raw` directly, where aliasing a read+write buffer is intentional + sound.)
        unsafe {
            self.run_parallel_raw(
                grid_sizes, dom_los, lo, extent, &in_bases, scalars, &out_bases,
            );
        }
    }

    /// the raw-base parallel driver — the dispatch primitive. takes input/output buffer BASES as
    /// pointers so an IN-PLACE field (one the kernel both reads and writes, e.g., `cons.den` in the
    /// fused godunov) can be bound as the SAME pointer in both `in_bases` and `out_bases`. that
    /// aliasing is SOUND: `compile_kernel` loads every input at the cell's flat index at function
    /// ENTRY, then stores every output at the same index at EXIT (read-before-write per cell), and
    /// distinct cells write distinct indices on distinct threads. this mirrors how the AOT
    /// `dispatch_named` binds in-place `cons.*` once and lets the kernel read+write one pointer.
    ///
    /// SAFETY (caller contract): every base must point to a buffer at least as large as the
    /// `(lo, extent)` layout addresses; `in_bases.len() == n_in`, `out_bases.len() == n_out`,
    /// `scalars.len() == n_scalar`; output buffers (modulo intentional in-place aliasing with their
    /// own input) must not alias EACH OTHER or any read-only input.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn run_parallel_raw(
        &self,
        grid_sizes: &[u32],
        dom_los: &[i32],
        lo: &[i32],
        extent: &[u32],
        in_bases: &[*const f64],
        scalars: &[f64],
        out_bases: &[*mut f64],
    ) {
        use rayon::prelude::*;
        assert_eq!(grid_sizes.len(), self.ndim);
        assert_eq!(in_bases.len(), self.n_in);
        assert_eq!(out_bases.len(), self.n_out);
        assert_eq!(scalars.len(), self.n_scalar);
        let lo_i: Vec<i64> = lo.iter().map(|&x| x as i64).collect();
        let ext_i: Vec<i64> = extent.iter().map(|&x| x as i64).collect();
        let shared = SharedBufs {
            in_ptrs: in_bases.to_vec(),
            out_ptrs: out_bases.to_vec(),
        };
        // small stack coord (ndim <= 3 in practice; 8 is ample) — no per-cell heap alloc on the
        // hot path, unlike the serial `run`'s `vec![]`.
        const MAX_NDIM: usize = 8;
        assert!(
            self.ndim <= MAX_NDIM,
            "run_parallel: ndim {} exceeds {MAX_NDIM}",
            self.ndim
        );
        let ndim = self.ndim;
        let total: usize = grid_sizes.iter().map(|&g| g as usize).product();
        (0..total).into_par_iter().for_each(|flat| {
            let shared = &shared;
            let mut coord = [0i64; MAX_NDIM];
            let mut rem = flat;
            for ax in (0..ndim).rev() {
                let g = grid_sizes[ax] as usize;
                coord[ax] = dom_los[ax] as i64 + (rem % g) as i64;
                rem /= g;
            }
            // SAFETY: per the function contract — the kernel reads n_in bases + n_scalar scalars
            // and writes n_out bases at THIS coord's flat index; cell-disjoint write indices make
            // the shared bases sound to share across threads (in-place aliasing read-before-write).
            unsafe {
                (self.cell)(
                    coord.as_ptr(),
                    lo_i.as_ptr(),
                    ext_i.as_ptr(),
                    shared.in_ptrs.as_ptr(),
                    scalars.as_ptr(),
                    shared.out_ptrs.as_ptr() as *mut *mut f64,
                );
            }
        });
    }
}

/// compile a scalarized stencil kernel. `field_inputs` = the cell-load / stencil-read field keys
/// (input buffer order); `scalar_params` = kernel scalars; `field_writes` = `(key, RHS node)`.
/// the body may carry control flow (`For`/`If`/`Break`, e.g., an `IterateInline` root-find);
/// only generic-dim `For` bounds reject -> caller falls back to the interpreter.
pub fn compile_kernel(
    graph: &symbi_ir::graph::Graph,
    field_inputs: &[String],
    scalar_params: &[String],
    field_writes: &[(String, symbi_ir::graph::NodeId)],
    ndim: usize,
) -> Result<CompiledKernel, JitError> {
    let write_nodes: Vec<symbi_ir::graph::NodeId> = field_writes.iter().map(|(_, n)| *n).collect();
    let sc = symbi_ir::passes::scalarize::scalarize_kernel(graph, &write_nodes);
    let (n_in, n_out, n_scalar) = (field_inputs.len(), field_writes.len(), scalar_params.len());
    let field_idx: HashMap<String, usize> = field_inputs
        .iter()
        .enumerate()
        .map(|(i, k)| (k.clone(), i))
        .collect();

    // ---- module + shims (same setup as `compile`) ----
    let mut flags = settings::builder();
    flags.set("use_colocated_libcalls", "false").unwrap();
    flags.set("is_pic", "false").unwrap();
    // optimize: the JIT'd kernel runs the WHOLE per-cell body (godunov + source) per cell, so the
    // codegen quality matters — Cranelift defaults to `opt_level=none`, which leaves it well behind
    // the AOT rustc `-O` kernels. "speed" enables GVN / LICM / redundant-load elimination / better
    // regalloc WITHOUT FP reassociation or auto-FMA (Cranelift never contracts FMA), so the
    // bit-identity oracle still holds (interp == cranelift).
    flags.set("opt_level", "speed").unwrap();
    let isa = cranelift_native::builder()
        .map_err(|e| JitError::Codegen(format!("native isa: {e}")))?
        .finish(settings::Flags::new(flags))
        .map_err(|e| JitError::Codegen(format!("isa finish: {e}")))?;
    let mut jb = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());
    for (name, _, ptr) in shim_table() {
        jb.symbol(*name, *ptr);
    }
    let mut module = JITModule::new(jb);
    let ptr_ty = module.target_config().pointer_type();
    let psz = ptr_ty.bytes() as i32;

    let mut shim_ids: HashMap<&'static str, (FuncId, usize)> = HashMap::new();
    for (name, arity, _) in shim_table() {
        let mut sig = module.make_signature();
        for _ in 0..*arity {
            sig.params.push(AbiParam::new(types::F64));
        }
        sig.returns.push(AbiParam::new(types::F64));
        let id = module
            .declare_function(name, Linkage::Import, &sig)
            .map_err(|e| JitError::Codegen(format!("declare shim '{name}': {e}")))?;
        shim_ids.insert(name, (id, *arity));
    }

    // ---- entry: fn(coord, lo, extent, in_bufs, scalars, out_bufs) — 6 pointers ----
    let mut ctx = module.make_context();
    ctx.func.signature = Signature::new(module.target_config().default_call_conv);
    for _ in 0..6 {
        ctx.func.signature.params.push(AbiParam::new(ptr_ty));
    }

    let mut fctx = FunctionBuilderContext::new();
    let mut b = FunctionBuilder::new(&mut ctx.func, &mut fctx);
    let entry = b.create_block();
    b.append_block_params_for_function_params(entry);
    b.switch_to_block(entry);
    b.seal_block(entry);
    let pr = b.block_params(entry);
    let (coord_ptr, lo_ptr, ext_ptr, in_bufs, scalars_ptr, out_bufs) =
        (pr[0], pr[1], pr[2], pr[3], pr[4], pr[5]);

    let mut shim_refs: HashMap<&'static str, (cranelift_codegen::ir::FuncRef, usize)> =
        HashMap::new();
    for (name, (id, arity)) in &shim_ids {
        let fref = module.declare_func_in_func(*id, b.func);
        shim_refs.insert(name, (fref, *arity));
    }

    // load coord / lo / extent into i64 registers.
    let mut coord_i = Vec::with_capacity(ndim);
    let mut lo_v = Vec::with_capacity(ndim);
    let mut ext_v = Vec::with_capacity(ndim);
    for ax in 0..ndim {
        let o = (ax * 8) as i32;
        coord_i.push(b.ins().load(types::I64, MemFlags::trusted(), coord_ptr, o));
        lo_v.push(b.ins().load(types::I64, MemFlags::trusted(), lo_ptr, o));
        ext_v.push(b.ins().load(types::I64, MemFlags::trusted(), ext_ptr, o));
    }
    let mut coord_vars: HashMap<String, Value> = HashMap::new();
    for ax in 0..ndim {
        coord_vars.insert(format!("_coord_{ax}"), coord_i[ax]);
    }
    // body let-bindings, so a load index referencing a CSE'd integer offset var resolves by recursion.
    let mut let_defs: HashMap<String, &ScalarExpr> = HashMap::new();
    collect_let_defs(&sc.body, &mut let_defs);
    let sctx = StencilCtx {
        in_bufs,
        field_idx: &field_idx,
        lo: &lo_v,
        extent: &ext_v,
        coord_vars: &coord_vars,
        let_defs: &let_defs,
        ndim,
        ptr_ty,
    };

    // current-cell flat index, reused for cell loads + output stores.
    let idx0 = emit_flat_index(&mut b, &coord_i, &sctx);
    let idx0_off = b.ins().imul_imm(idx0, 8);

    // seed the var map: cell loads, scalars, and `_coord_N` as f64 (for float coord use). these
    // dominate everything (defined in the entry block) so they stay immutable `Val` slots.
    let mut vars: HashMap<String, LocalSlot> = HashMap::new();
    for (i, key) in field_inputs.iter().enumerate() {
        let base = b
            .ins()
            .load(ptr_ty, MemFlags::trusted(), in_bufs, i as i32 * psz);
        let addr = b.ins().iadd(base, idx0_off);
        let v = b.ins().load(types::F64, MemFlags::trusted(), addr, 0);
        vars.insert(key.clone(), LocalSlot::Val(v));
    }
    for (i, name) in scalar_params.iter().enumerate() {
        let v = b
            .ins()
            .load(types::F64, MemFlags::trusted(), scalars_ptr, (i * 8) as i32);
        vars.insert(name.clone(), LocalSlot::Val(v));
    }
    for ax in 0..ndim {
        let cf = b.ins().fcvt_from_sint(types::F64, coord_i[ax]);
        vars.insert(format!("_coord_{ax}"), LocalSlot::Val(cf));
    }

    // body (Let/LetMut/Assign/For/If/Break) + outputs.
    let mut next_var: u32 = 0;
    translate_stmts(
        &mut b,
        &sc.body,
        &mut vars,
        &shim_refs,
        &sctx,
        &mut next_var,
        None,
    )?;
    for (k, expr) in sc.outputs.iter().enumerate() {
        let v = translate_expr(&mut b, expr, &vars, &shim_refs, Some(&sctx))?;
        let base = b
            .ins()
            .load(ptr_ty, MemFlags::trusted(), out_bufs, k as i32 * psz);
        let addr = b.ins().iadd(base, idx0_off);
        b.ins().store(MemFlags::trusted(), v, addr, 0);
    }

    b.ins().return_(&[]);
    b.finalize();

    let func_id = module
        .declare_function("symbi_kernel", Linkage::Export, &ctx.func.signature)
        .map_err(|e| JitError::Codegen(format!("declare entry: {e}")))?;
    module
        .define_function(func_id, &mut ctx)
        .map_err(|e| JitError::Codegen(format!("define entry: {e:?}")))?;
    module.clear_context(&mut ctx);
    module
        .finalize_definitions()
        .map_err(|e| JitError::Codegen(format!("finalize: {e}")))?;
    let code = module.get_finalized_function(func_id);
    // SAFETY: the finalized entry has the 6-pointer signature declared above.
    let cell = unsafe {
        std::mem::transmute::<
            *const u8,
            unsafe extern "C" fn(
                *const i64,
                *const i64,
                *const i64,
                *const *const f64,
                *const f64,
                *mut *mut f64,
            ),
        >(code)
    };

    Ok(CompiledKernel {
        _module: module,
        cell,
        ndim,
        n_in,
        n_out,
        n_scalar,
    })
}

/// JIT a traced `GvKernel` (e.g., the combined godunov+source stage) via `compile_kernel`, mapping
/// the kernel's ABI manifest. `writes` are the trace's `(key, runtime, node)` outputs. THE BRIDGE
/// for v2 fusion: build the godunov+source `GvKernel` (`splice_fused_sources_to_contribs` /
/// `godunov_stage_gv_with_fused_sources`), JIT it here, dispatch it instead of the two-pass.
pub fn compile_gv_kernel(
    kernel: &symbi_ir::GvKernel,
    writes: &[(String, symbi_ir::FieldBind, symbi_ir::graph::NodeId)],
    ndim: usize,
) -> Result<CompiledKernel, JitError> {
    let field_inputs: Vec<String> = kernel.field_inputs.iter().map(|(k, _)| k.clone()).collect();
    let field_writes: Vec<(String, symbi_ir::graph::NodeId)> =
        writes.iter().map(|(k, _, n)| (k.clone(), *n)).collect();
    compile_kernel(
        &kernel.graph,
        &field_inputs,
        &kernel.scalar_params,
        &field_writes,
        ndim,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use symbi_ir::backends::interp::{Backend, Cpu};
    use symbi_ir::graph::{ElementWiseOp, Graph, NodeId, TranscendentalOp};
    use symbi_ir::passes::scalarize::{LoweredParam, scalarize};

    /// build a graph (the closure adds its own params + returns the output node), scalarize,
    /// and assert the Cranelift-compiled fn is BIT-IDENTICAL to the interpreter over many
    /// random input vectors — the oracle gate. inputs are passed in `LoweredFn::params` order
    /// to BOTH paths, so the exact param order is irrelevant.
    fn assert_jit_matches_interp(build: impl Fn(&mut Graph) -> NodeId) {
        let mut g = Graph::new();
        let out = build(&mut g);
        let lowered = scalarize(&g, out, "jit_test");
        let compiled = compile(&lowered).expect("jit compile");
        let n = lowered.params.len();
        assert_eq!(compiled.n_inputs(), n);
        assert_eq!(compiled.n_outputs(), 1);

        // deterministic xorshift inputs (no rng dep). the domain is DELIBERATELY HARSH — negatives,
        // the zero-crossing, signed zeros, and magnitude extremes — to stress the native CLIF ops
        // (fcmp/select/div/sqrt/neg) at the edges a narrow positive range never reaches. interp ==
        // cranelift must hold on NaN/Inf too: every native op is IEEE-754 on both sides, every
        // MethodCall routes through the SAME std shim. (the carrier oracle's whole job is to catch a
        // codegen divergence; a narrow fuzz domain is exactly how a min/max-style NaN bug hides.)
        let mut state = 0x2545F4914F6CDD1Du64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let u = (state >> 11) as f64 / (1u64 << 53) as f64; // [0, 1)
            match (state >> 5) & 0x7 {
                0 => 0.0,
                1 => -0.0,
                2 => 1e15 * (u - 0.5),  // huge magnitude, both signs
                3 => 1e-15 * (u - 0.5), // tiny magnitude (denormal-ish), both signs
                _ => 4.0 * u - 2.0,     // broad [-2, 2): negatives + zero-crossing
            }
        };

        for _ in 0..4000 {
            let inputs: Vec<f64> = (0..n).map(|_| next()).collect();
            let want = Cpu.eval_elemental(&lowered, &inputs)[0];
            let mut got = [0.0f64];
            compiled.call(&inputs, &mut got);
            // bit-equal, OR both NaN (a NaN output is correct on either side; the payload may differ).
            let ok = want.to_bits() == got[0].to_bits() || (want.is_nan() && got[0].is_nan());
            assert!(
                ok,
                "JIT != interp: inputs={inputs:?} interp={} ({:#018x}) jit={} ({:#018x})",
                want,
                want.to_bits(),
                got[0],
                got[0].to_bits(),
            );
        }
    }

    fn p(g: &mut Graph, name: &str) -> NodeId {
        g.add_scalar_param(name, ElementTy::F64)
    }

    #[test]
    fn arithmetic_matches() {
        // (a*b + c) / (a - b) — fadd/fmul/fsub/fdiv, no FMA contraction.
        assert_jit_matches_interp(|g| {
            let (a, b, c) = (p(g, "a"), p(g, "b"), p(g, "c"));
            let ab = g.element_wise(ElementWiseOp::Mul, vec![a, b], None);
            let num = g.element_wise(ElementWiseOp::Add, vec![ab, c], None);
            let den = g.element_wise(ElementWiseOp::Sub, vec![a, b], None);
            g.element_wise(ElementWiseOp::Div, vec![num, den], None)
        });
    }

    #[test]
    fn sqrt_and_abs_match() {
        // abs(sqrt(a*a + b*b)) — sqrt native CLIF; abs is fcmp+select (r<0?-r:r).
        assert_jit_matches_interp(|g| {
            let (a, b) = (p(g, "a"), p(g, "b"));
            let aa = g.element_wise(ElementWiseOp::Mul, vec![a, a], None);
            let bb = g.element_wise(ElementWiseOp::Mul, vec![b, b], None);
            let s = g.element_wise(ElementWiseOp::Add, vec![aa, bb], None);
            let r = g.element_wise(ElementWiseOp::Sqrt, vec![s], None);
            g.element_wise(ElementWiseOp::Abs, vec![r], None)
        });
    }

    #[test]
    fn transcendentals_match() {
        // p0 * sin(a) + exp(-b) — shimmed transcendentals, bit-identical to std.
        assert_jit_matches_interp(|g| {
            let (p0, a, b) = (p(g, "p0"), p(g, "a"), p(g, "b"));
            let sin_a = g.transcendental(TranscendentalOp::Sin, vec![a], None);
            let term = g.element_wise(ElementWiseOp::Mul, vec![p0, sin_a], None);
            let nb = g.element_wise(ElementWiseOp::Neg, vec![b], None);
            let exp_nb = g.transcendental(TranscendentalOp::Exp, vec![nb], None);
            g.element_wise(ElementWiseOp::Add, vec![term, exp_nb], None)
        });
    }

    #[test]
    fn select_matches() {
        // if a > b { a*2 } else { b } — fcmp -> select.
        assert_jit_matches_interp(|g| {
            let (a, b) = (p(g, "a"), p(g, "b"));
            let two = g.add_const(ConstValue::F64(2.0), None);
            let a2 = g.element_wise(ElementWiseOp::Mul, vec![a, two], None);
            let cond = g.element_wise(ElementWiseOp::Gt, vec![a, b], None);
            g.select(cond, a2, b, None)
        });
    }

    #[test]
    fn min_max_match() {
        // max(a, 0) * min(b, a) — fcmp+select ternary; the widened oracle
        // exercises NaN / signed-zero, so jit == interp under the my_min/my_max form.
        assert_jit_matches_interp(|g| {
            let (a, b) = (p(g, "a"), p(g, "b"));
            let zero = g.add_const(ConstValue::F64(0.0), None);
            let mx = g.element_wise(ElementWiseOp::Max, vec![a, zero], None);
            let mn = g.element_wise(ElementWiseOp::Min, vec![b, a], None);
            g.element_wise(ElementWiseOp::Mul, vec![mx, mn], None)
        });
    }

    #[test]
    fn unsupported_node_is_rejected_not_miscompiled() {
        // an out-of-subset node (FieldLoadAt) -> clean JitError, never a wrong kernel.
        let lowered = LoweredFn {
            name: "reject".into(),
            params: vec![LoweredParam::scalar("a".into(), ElementTy::F64)],
            body: vec![],
            results: vec![ScalarExpr::FieldLoadAt {
                field_key: "f".into(),
                components: vec![],
            }],
            result_element: ElementTy::F64,
            result_shape: vec![],
        };
        assert!(matches!(compile(&lowered), Err(JitError::Unsupported(_))));
    }

    // ---- stencil-kernel JIT, gated against `Cpu::run_kernel` ----

    #[test]
    fn kernel_stencil_matches_interp() {
        use symbi_ir::Symbol;
        use symbi_ir::backends::interp::{CpuField, CpuFieldMut};
        use symbi_ir::backends::kernel::KernelEmitInputs;
        use symbi_ir::emit::{Precision, Target, TargetConfig};

        // a 1D stencil kernel: out[c] = in[c] + 2 * in[c+1] - sqrt(in[c]).
        // exercises the cell load (`Var("in")`), the stencil read (`FieldLoadAt` at `_coord_0 + 1`),
        // a const, arithmetic, and a native method — the whole stencil-kernel lowering path.
        let mut g = Graph::new();
        let in_cell = g.add_scalar_param("in", ElementTy::F64); // cell load at the current coord
        let c0 = g.add_scalar_param("_coord_0", ElementTy::I32);
        let one = g.add_const(ConstValue::I32(1), None);
        let c0p = g.element_wise(ElementWiseOp::Add, vec![c0, one], None);
        let nbr = g.load_at(Symbol::intern("in"), vec![c0p], None); // in[c+1]
        let two = g.add_const(ConstValue::F64(2.0), None);
        let two_nbr = g.element_wise(ElementWiseOp::Mul, vec![two, nbr], None);
        let sum = g.element_wise(ElementWiseOp::Add, vec![in_cell, two_nbr], None);
        let sq = g.element_wise(ElementWiseOp::Sqrt, vec![in_cell], None);
        let out = g.element_wise(ElementWiseOp::Sub, vec![sum, sq], None);

        let spec = KernelEmitInputs {
            kernel_name: "stencil_test",
            coalesce_layout: false,
            ndim: 1,
            target: TargetConfig {
                target: Target::Cuda,
                precision: Precision::F64,
            },
            field_inputs: &[("in".into(), "in".into())],
            scalar_params: &[],
            field_writes: &[("out".into(), "out".into(), out)],
            coord_components: &[0],
            device_preamble: &[],
            tile_spec: None,
        };

        // domain [0, N); buffer [0, N+1) so the c+1 read at c=N-1 stays in bounds.
        let n = 64usize;
        let ext = (n + 1) as u32;
        let in_data: Vec<f64> = (0..ext)
            .map(|i| 0.3 + 1.7 * (i as f64 * 0.123).fract())
            .collect();

        let mut out_interp = vec![0.0f64; ext as usize];
        Cpu.run_kernel(
            &g,
            &spec,
            &[CpuField {
                data: &in_data,
                lo: &[0],
                extent: &[ext],
            }],
            &mut [CpuFieldMut {
                data: &mut out_interp,
                lo: &[0],
                extent: &[ext],
            }],
            &[],
            &[n as u32],
            &[0],
        );

        let kernel = compile_kernel(&g, &["in".into()], &[], &[("out".into(), out)], 1)
            .expect("jit compile_kernel");
        let mut out_jit = vec![0.0f64; ext as usize];
        kernel.run(
            &[n as u32],
            &[0],
            &[0],
            &[ext],
            &[&in_data],
            &[],
            &mut [&mut out_jit],
        );

        for c in 0..n {
            assert_eq!(
                out_interp[c].to_bits(),
                out_jit[c].to_bits(),
                "stencil JIT != interp at cell {c}: interp={} jit={}",
                out_interp[c],
                out_jit[c],
            );
        }
    }

    #[test]
    fn kernel_iterate_loop_matches_interp() {
        // control flow. an IterateInline Newton-sqrt — 8 iterations of
        // acc = 0.5*(acc + N/acc) per cell, N = in[c]. exercises LetMut/For/Assign (and the
        // CLIF loop + Variable phi) against the interpreter, bit-for-bit.
        use symbi_ir::backends::interp::{CpuField, CpuFieldMut};
        use symbi_ir::backends::kernel::KernelEmitInputs;
        use symbi_ir::emit::{Precision, Target, TargetConfig};

        let mut g = Graph::new();
        let in_cell = g.add_scalar_param("in", ElementTy::F64); // N = in[c]
        let acc = g.iter_acc(0, None); // current accumulator
        let half = g.add_const(ConstValue::F64(0.5), None);
        let div = g.element_wise(ElementWiseOp::Div, vec![in_cell, acc], None); // N/acc
        let sum = g.element_wise(ElementWiseOp::Add, vec![acc, div], None); // acc + N/acc
        let step = g.element_wise(ElementWiseOp::Mul, vec![half, sum], None); // 0.5*(...)
        let it = g.iterate_inline_scalar(acc, in_cell, step, 8, None, None);
        assert!(!g.has_errors(), "graph errors: {:?}", g.errors());

        let spec = KernelEmitInputs {
            kernel_name: "newton_sqrt",
            coalesce_layout: false,
            ndim: 1,
            target: TargetConfig {
                target: Target::Cuda,
                precision: Precision::F64,
            },
            field_inputs: &[("in".into(), "in".into())],
            scalar_params: &[],
            field_writes: &[("out".into(), "out".into(), it)],
            coord_components: &[],
            device_preamble: &[],
            tile_spec: None,
        };

        let n = 64usize;
        let in_data: Vec<f64> = (0..n)
            .map(|i| 0.5 + 4.0 * (i as f64 * 0.137).fract() + i as f64)
            .collect();
        let mut out_interp = vec![0.0f64; n];
        Cpu.run_kernel(
            &g,
            &spec,
            &[CpuField {
                data: &in_data,
                lo: &[0],
                extent: &[n as u32],
            }],
            &mut [CpuFieldMut {
                data: &mut out_interp,
                lo: &[0],
                extent: &[n as u32],
            }],
            &[],
            &[n as u32],
            &[0],
        );

        let kernel = compile_kernel(&g, &["in".into()], &[], &[("out".into(), it)], 1)
            .expect("jit compile_kernel (iterate)");
        let mut out_jit = vec![0.0f64; n];
        kernel.run(
            &[n as u32],
            &[0],
            &[0],
            &[n as u32],
            &[&in_data],
            &[],
            &mut [&mut out_jit],
        );

        for c in 0..n {
            assert_eq!(
                out_interp[c].to_bits(),
                out_jit[c].to_bits(),
                "iterate JIT != interp at cell {c}: interp={} jit={} (target sqrt {})",
                out_interp[c],
                out_jit[c],
                in_data[c].sqrt(),
            );
        }
    }

    // ---- the parallel driver, gated `run_parallel == run == interp` ----

    #[test]
    fn kernel_run_parallel_matches_serial_and_interp() {
        use symbi_ir::Symbol;
        use symbi_ir::backends::interp::{CpuField, CpuFieldMut};
        use symbi_ir::backends::kernel::KernelEmitInputs;
        use symbi_ir::emit::{Precision, Target, TargetConfig};

        // 2D multi-output stencil to exercise the bijective coord->index mapping across threads:
        //   out0[c] = in[c] + 2 * in[c + e_x] - sqrt(in[c]),  out1[c] = in[c] * in[c + e_y].
        // shared-base disjoint writes must reproduce serial `run` (hence interp) bit-for-bit.
        let mut g = Graph::new();
        let in_cell = g.add_scalar_param("in", ElementTy::F64);
        let c0 = g.add_scalar_param("_coord_0", ElementTy::I32);
        let c1 = g.add_scalar_param("_coord_1", ElementTy::I32);
        let one = g.add_const(ConstValue::I32(1), None);
        let c0p = g.element_wise(ElementWiseOp::Add, vec![c0, one], None);
        let c1p = g.element_wise(ElementWiseOp::Add, vec![c1, one], None);
        let nbr_x = g.load_at(Symbol::intern("in"), vec![c0p, c1], None);
        let nbr_y = g.load_at(Symbol::intern("in"), vec![c0, c1p], None);
        let two = g.add_const(ConstValue::F64(2.0), None);
        let two_nx = g.element_wise(ElementWiseOp::Mul, vec![two, nbr_x], None);
        let sum = g.element_wise(ElementWiseOp::Add, vec![in_cell, two_nx], None);
        let sq = g.element_wise(ElementWiseOp::Sqrt, vec![in_cell], None);
        let out0 = g.element_wise(ElementWiseOp::Sub, vec![sum, sq], None);
        let out1 = g.element_wise(ElementWiseOp::Mul, vec![in_cell, nbr_y], None);
        assert!(!g.has_errors(), "graph errors: {:?}", g.errors());

        // domain [0, nx) x [0, ny); buffer one cell larger per axis so c+1 reads stay in bounds.
        let (nx, ny) = (24usize, 17usize);
        let (ex, ey) = ((nx + 1) as u32, (ny + 1) as u32);
        let buf_len = (ex * ey) as usize;
        let in_data: Vec<f64> = (0..buf_len)
            .map(|i| 0.3 + 1.7 * (i as f64 * 0.0916).fract())
            .collect();

        let spec = KernelEmitInputs {
            kernel_name: "par_stencil_test",
            coalesce_layout: false,
            ndim: 2,
            target: TargetConfig {
                target: Target::Cuda,
                precision: Precision::F64,
            },
            field_inputs: &[("in".into(), "in".into())],
            scalar_params: &[],
            field_writes: &[
                ("out0".into(), "out0".into(), out0),
                ("out1".into(), "out1".into(), out1),
            ],
            coord_components: &[0, 1],
            device_preamble: &[],
            tile_spec: None,
        };

        let mut out0_interp = vec![0.0f64; buf_len];
        let mut out1_interp = vec![0.0f64; buf_len];
        Cpu.run_kernel(
            &g,
            &spec,
            &[CpuField {
                data: &in_data,
                lo: &[0, 0],
                extent: &[ex, ey],
            }],
            &mut [
                CpuFieldMut {
                    data: &mut out0_interp,
                    lo: &[0, 0],
                    extent: &[ex, ey],
                },
                CpuFieldMut {
                    data: &mut out1_interp,
                    lo: &[0, 0],
                    extent: &[ex, ey],
                },
            ],
            &[],
            &[nx as u32, ny as u32],
            &[0, 0],
        );

        let kernel = compile_kernel(
            &g,
            &["in".into()],
            &[],
            &[("out0".into(), out0), ("out1".into(), out1)],
            2,
        )
        .expect("jit compile_kernel (2d)");

        let mut out0_serial = vec![0.0f64; buf_len];
        let mut out1_serial = vec![0.0f64; buf_len];
        kernel.run(
            &[nx as u32, ny as u32],
            &[0, 0],
            &[0, 0],
            &[ex, ey],
            &[&in_data],
            &[],
            &mut [&mut out0_serial, &mut out1_serial],
        );

        let mut out0_par = vec![0.0f64; buf_len];
        let mut out1_par = vec![0.0f64; buf_len];
        kernel.run_parallel(
            &[nx as u32, ny as u32],
            &[0, 0],
            &[0, 0],
            &[ex, ey],
            &[&in_data],
            &[],
            &mut [&mut out0_par, &mut out1_par],
        );

        for jj in 0..ny {
            for ii in 0..nx {
                let c = jj * ex as usize + ii; // C row-major, last axis fastest
                for (lbl, interp, serial, par) in [
                    ("out0", &out0_interp, &out0_serial, &out0_par),
                    ("out1", &out1_interp, &out1_serial, &out1_par),
                ] {
                    assert_eq!(
                        serial[c].to_bits(),
                        par[c].to_bits(),
                        "{lbl} run_parallel != run at ({ii},{jj})",
                    );
                    assert_eq!(
                        interp[c].to_bits(),
                        par[c].to_bits(),
                        "{lbl} run_parallel != interp at ({ii},{jj})",
                    );
                }
            }
        }
    }

    #[test]
    fn kernel_run_parallel_ghost_offset_matches_interp() {
        // the production layout: the iteration window (interior) starts at allocated index `g`, and
        // the buffer's `lo` is NEGATIVE (ghost cells), e.g., alo=[-2,-2], dom_lo=[0,0]. the godunov
        // dispatch runs exactly this (alo=[-2,-2], dlo=[0,0]). a stencil read `f[c+e]` + flat_index
        // must subtract the (negative) `lo` correctly. gated run_parallel == interp, bit-for-bit.
        use symbi_ir::Symbol;
        use symbi_ir::backends::interp::{CpuField, CpuFieldMut};
        use symbi_ir::backends::kernel::KernelEmitInputs;
        use symbi_ir::emit::{Precision, Target, TargetConfig};

        // out[c] = in[c] + 2*in[c + e_x] - in[c + e_y].
        let mut g = Graph::new();
        let in_cell = g.add_scalar_param("in", ElementTy::F64);
        let c0 = g.add_scalar_param("_coord_0", ElementTy::I32);
        let c1 = g.add_scalar_param("_coord_1", ElementTy::I32);
        let one = g.add_const(ConstValue::I32(1), None);
        let c0p = g.element_wise(ElementWiseOp::Add, vec![c0, one], None);
        let c1p = g.element_wise(ElementWiseOp::Add, vec![c1, one], None);
        let nx_ = g.load_at(Symbol::intern("in"), vec![c0p, c1], None);
        let ny_ = g.load_at(Symbol::intern("in"), vec![c0, c1p], None);
        let two = g.add_const(ConstValue::F64(2.0), None);
        let two_nx = g.element_wise(ElementWiseOp::Mul, vec![two, nx_], None);
        let sum = g.element_wise(ElementWiseOp::Add, vec![in_cell, two_nx], None);
        let out = g.element_wise(ElementWiseOp::Sub, vec![sum, ny_], None);
        assert!(!g.has_errors(), "graph errors: {:?}", g.errors());

        // interior [0, n) x [0, n); allocated [-g, n+g) per axis (alo=[-g,-g], aext=n+2g).
        let (n, gh) = (16usize, 2i32);
        let ext = (n as i32 + 2 * gh) as u32;
        let alo = [-gh, -gh];
        let buf_len = (ext * ext) as usize;
        let in_data: Vec<f64> = (0..buf_len)
            .map(|i| 0.3 + 1.7 * (i as f64 * 0.0731).fract())
            .collect();

        let spec = KernelEmitInputs {
            kernel_name: "ghost_test",
            coalesce_layout: false,
            ndim: 2,
            target: TargetConfig {
                target: Target::Cuda,
                precision: Precision::F64,
            },
            field_inputs: &[("in".into(), "in".into())],
            scalar_params: &[],
            field_writes: &[("out".into(), "out".into(), out)],
            coord_components: &[0, 1],
            device_preamble: &[],
            tile_spec: None,
        };
        let mut out_interp = vec![0.0f64; buf_len];
        Cpu.run_kernel(
            &g,
            &spec,
            &[CpuField {
                data: &in_data,
                lo: &alo,
                extent: &[ext, ext],
            }],
            &mut [CpuFieldMut {
                data: &mut out_interp,
                lo: &alo,
                extent: &[ext, ext],
            }],
            &[],
            &[n as u32, n as u32],
            &[0, 0],
        );

        let kernel = compile_kernel(&g, &["in".into()], &[], &[("out".into(), out)], 2)
            .expect("jit compile_kernel (ghost)");
        let mut out_par = vec![0.0f64; buf_len];
        kernel.run_parallel(
            &[n as u32, n as u32],
            &[0, 0],
            &alo,
            &[ext, ext],
            &[&in_data],
            &[],
            &mut [&mut out_par],
        );

        for c in 0..buf_len {
            assert_eq!(
                out_interp[c].to_bits(),
                out_par[c].to_bits(),
                "ghost-offset run_parallel != interp at flat {c}: interp={} jit={}",
                out_interp[c],
                out_par[c],
            );
        }
    }

    #[test]
    fn run_parallel_raw_in_place_alias_matches_interp() {
        // the IN-PLACE dispatch primitive, in the EXACT shape the fused godunov produces:
        //   x[c] = x[c] + 2*f[c+1] - f[c]
        // `x` is read at its OWN cell AND written (the in-place `cons.*` field), while the only
        // NEIGHBOUR read (`f[c+1]`) is of a SEPARATE read-only field (the godunov's fluxes). this
        // is the soundness boundary of the aliasing: an in-place field is read only at its own
        // index (read-before-write per cell), never at a neighbour (which would be a cross-cell
        // read-after-write race). dispatched via `run_parallel_raw` with `x`'s buffer aliased as
        // both an input base and the output base; `f` distinct + read-only. must equal the interp
        // run (separate buffers), bit-for-bit.
        use symbi_ir::Symbol;
        use symbi_ir::backends::interp::{CpuField, CpuFieldMut};
        use symbi_ir::backends::kernel::KernelEmitInputs;
        use symbi_ir::emit::{Precision, Target, TargetConfig};

        let mut g = Graph::new();
        let x_cell = g.add_scalar_param("x", ElementTy::F64); // in-place field, own-cell read
        let f_cell = g.add_scalar_param("f", ElementTy::F64); // read-only field, own-cell read
        let c0 = g.add_scalar_param("_coord_0", ElementTy::I32);
        let one = g.add_const(ConstValue::I32(1), None);
        let c0p = g.element_wise(ElementWiseOp::Add, vec![c0, one], None);
        let f_nbr = g.load_at(Symbol::intern("f"), vec![c0p], None); // f[c+1] — read-only neighbour
        let two = g.add_const(ConstValue::F64(2.0), None);
        let two_fnbr = g.element_wise(ElementWiseOp::Mul, vec![two, f_nbr], None);
        let sum = g.element_wise(ElementWiseOp::Add, vec![x_cell, two_fnbr], None); // x[c] + 2*f[c+1]
        let out = g.element_wise(ElementWiseOp::Sub, vec![sum, f_cell], None); // - f[c]
        assert!(!g.has_errors(), "graph errors: {:?}", g.errors());

        let n = 48usize;
        let ext = (n + 1) as u32; // +1 so the f[c+1] neighbour read stays in bounds
        let x0: Vec<f64> = (0..ext)
            .map(|i| 0.5 + 1.3 * (i as f64 * 0.071).fract())
            .collect();
        let f0: Vec<f64> = (0..ext)
            .map(|i| 0.2 + 0.9 * (i as f64 * 0.053).fract())
            .collect();

        // interp: separate input (x0) and output buffers — the reference.
        let spec = KernelEmitInputs {
            kernel_name: "inplace_test",
            coalesce_layout: false,
            ndim: 1,
            target: TargetConfig {
                target: Target::Cuda,
                precision: Precision::F64,
            },
            field_inputs: &[("x".into(), "x".into()), ("f".into(), "f".into())],
            scalar_params: &[],
            field_writes: &[("x".into(), "x".into(), out)],
            coord_components: &[0],
            device_preamble: &[],
            tile_spec: None,
        };
        let mut out_interp = x0.clone();
        Cpu.run_kernel(
            &g,
            &spec,
            &[
                CpuField {
                    data: &x0,
                    lo: &[0],
                    extent: &[ext],
                },
                CpuField {
                    data: &f0,
                    lo: &[0],
                    extent: &[ext],
                },
            ],
            &mut [CpuFieldMut {
                data: &mut out_interp,
                lo: &[0],
                extent: &[ext],
            }],
            &[],
            &[n as u32],
            &[0],
        );

        // jit: `x`'s ONE buffer aliased as both an input base and the output base; `f` read-only.
        let kernel = compile_kernel(&g, &["x".into(), "f".into()], &[], &[("x".into(), out)], 1)
            .expect("jit compile_kernel (in-place)");
        let mut buf = x0.clone();
        let base = buf.as_mut_ptr();
        // SAFETY: `base` is the live `buf` allocation, sized to the (lo=0, extent) layout; aliasing
        // it as `x`'s read input AND the write output is the intended in-place dispatch pattern;
        // `f0` is a distinct read-only buffer.
        unsafe {
            kernel.run_parallel_raw(
                &[n as u32],
                &[0],
                &[0],
                &[ext],
                &[base as *const f64, f0.as_ptr()],
                &[],
                &[base],
            );
        }

        for c in 0..n {
            assert_eq!(
                out_interp[c].to_bits(),
                buf[c].to_bits(),
                "in-place run_parallel_raw != interp at cell {c}: interp={} jit={}",
                out_interp[c],
                buf[c],
            );
        }
    }
}
