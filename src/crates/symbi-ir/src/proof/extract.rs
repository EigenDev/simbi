// =============================================================================
// extract.rs
//
// the IR-DAG -> symbolic-form extraction: recursively interpret a curl(-of-edge-
// emf) DAG node into a `Value` (integer-poly path) or `RValue` (rational-function
// curvilinear path), then expose the entry points `LinForm::extract` /
// `LinFormR::extract_rat`. depends on the coefficient ring (poly) and the
// telescoping space (linform), plus `crate::graph`.
//
// usage:
//  let lf = LinForm::extract(&kernel.graph, root, &fields, &scalars);
//  let lr = LinFormR::extract_rat(&kernel.graph, root, &fields, &scalars);
// =============================================================================

use crate::graph::{ConstValue, ElementWiseOp, Graph, NodeId, Op};

use super::linform::{LinForm, LinFormR};
use super::poly::{FieldTerm, Poly, RatFun};

impl LinFormR {
    /// extract the symbolic rational-function linear form of the CURVILINEAR curl
    /// DAG rooted at `root`. `field_keys` are the edge-emf reads; `scalar_keys` are
    /// the geometry/time scalars (dt, x_lo_N, dx_N). the `_coord_N` params become
    /// the polynomial vars `c_N`; `Div` -> RatFun divide; `Sin` -> an opaque
    /// `sin_th@<2m>` symbol keyed by the resolved theta offset.
    pub fn extract_rat(
        graph: &Graph,
        root: NodeId,
        field_keys: &[&str],
        scalar_keys: &[&str],
    ) -> LinFormR {
        match eval_rat(graph, root, field_keys, scalar_keys) {
            RValue::Lin(lf) => lf,
            RValue::Scalar(_) => LinFormR::empty(),
        }
    }
}

/// extract a field-FREE geometry node as a pure rational-function scalar. a geometry
/// factor (face area, inverse volume, centroid) reads no fields, so `eval_rat`
/// returns `RValue::Scalar`; this returns that RatFun. panics if the node turns out
/// to be field-dependent (a `Lin`) — that would mean the wrong root was passed. the
/// public entry for the curvilinear conservation proof (area_hi == area_lo-shifted).
pub fn extract_scalar(graph: &Graph, root: NodeId, scalars: &[&str]) -> RatFun {
    match eval_rat(graph, root, &[], scalars) {
        RValue::Scalar(r) => r,
        RValue::Lin(_) => panic!("proof(rat): extract_scalar on a field-dependent node"),
    }
}

/// the rational-eval analog of `Value`: a pure rational scalar (a coefficient) or
/// a full rational linear form.
enum RValue {
    Scalar(RatFun),
    Lin(LinFormR),
}

/// recursively interpret the curvilinear curl DAG node as an `RValue`. handles the
/// op set the spherical curl actually produces (confirmed by tracing): Const,
/// Param (field / scalar / `_coord_N`), LoadAt, ElementWise(Add/Sub/Neg/Mul/Div/
/// Sin/Cast). panics loudly on anything else — an unexpected op would mean the
/// geometry changed shape and the representation may be too weak to prove it.
fn eval_rat(graph: &Graph, id: NodeId, fields: &[&str], scalars: &[&str]) -> RValue {
    let node = graph.node(id);
    match &node.op {
        Op::Const(ConstValue::I32(v)) => RValue::Scalar(RatFun::from_poly(Poly::constant(*v as i64))),
        Op::Const(ConstValue::F64(v)) => {
            // the curvilinear curl uses 0.5 (cell-center average) and integer
            // literals. carry a rational-valued constant exactly: 0.5 = 1/2.
            let two = Poly::constant(2);
            let r = v.round();
            if (v - r).abs() < 1e-12 {
                RValue::Scalar(RatFun::from_poly(Poly::constant(r as i64)))
            } else if (v - 0.5).abs() < 1e-12 {
                RValue::Scalar(RatFun::new(Poly::constant(1), two))
            } else {
                panic!("proof(rat): unexpected non-integer/half float literal {v} in curl DAG")
            }
        }
        Op::Param(sym) => {
            let key = sym.as_str();
            if fields.contains(&key) {
                let ndim = infer_ndim(graph);
                RValue::Lin(lin_single((key.to_string(), vec![0; ndim]), RatFun::from_poly(Poly::constant(1))))
            } else if scalars.contains(&key) {
                RValue::Scalar(RatFun::from_poly(Poly::var(key)))
            } else if let Some(ax) = coord_axis(key) {
                // the cell coord index becomes the polynomial var c_<ax>.
                RValue::Scalar(RatFun::from_poly(Poly::var(&format!("c_{ax}"))))
            } else {
                panic!("proof(rat): unknown param `{key}` used as a value");
            }
        }
        Op::LoadAt(sym, comps) => {
            let key = sym.as_str();
            assert!(fields.contains(&key), "proof(rat): LoadAt of non-field key `{key}`");
            let off: Vec<i32> = comps.iter().map(|&c| resolve_offset(graph, c)).collect();
            RValue::Lin(lin_single((key.to_string(), off), RatFun::from_poly(Poly::constant(1))))
        }
        Op::ElementWise(op, ins) => eval_rat_elementwise(graph, *op, ins, fields, scalars),
        // the spacing map's runtime `map_kind` cond (`map_kind > 0.5 ? log-face : uniform-face`) — a
        // LEAF reparametrization of the cell/face position. the discrete curl telescopes to
        // div(curl) = 0 INDEPENDENTLY of the face-position map: the two cells sharing a face use the
        // SAME position (whichever arm), so the cancellation is structural. both arms
        // carry the identical curl stencil; extract the uniform (else) arm — the canonical
        // instantiation, and the exact DAG this proof verified before spacing became a runtime scalar.
        Op::IfElse { else_results, .. } => eval_rat(graph, else_results[0], fields, scalars),
        // a metric COEFFICIENT guarded by a position predicate — the kerr-schild radius floor
        // selects |l|^2 = 1 outside it and |x|^2 / r_floor^2 within. like the spacing map above,
        // this is a leaf reparametrization the curl telescopes through: two cells sharing a face
        // evaluate the predicate at the SAME position and so take the SAME arm, making
        // div(curl) = 0 structural in either. extract the TRUE arm — the exterior, where the
        // kerr-schild null condition holds and the DAG is the canonical unit-l instantiation.
        Op::Select(_cond, then_id, _else_id) => eval_rat(graph, *then_id, fields, scalars),
        other => panic!("proof(rat): unsupported op in curvilinear curl DAG: {other:?}"),
    }
}

fn lin_single(term: FieldTerm, coeff: RatFun) -> LinFormR {
    let mut lf = LinFormR::empty();
    lf.insert(term, coeff);
    lf
}

/// parse `_coord_N` -> Some(N).
fn coord_axis(key: &str) -> Option<usize> {
    key.strip_prefix("_coord_").and_then(|s| s.parse().ok())
}

fn eval_rat_elementwise(
    graph: &Graph,
    op: ElementWiseOp,
    ins: &[NodeId],
    fields: &[&str],
    scalars: &[&str],
) -> RValue {
    match op {
        ElementWiseOp::Add | ElementWiseOp::Sub => {
            let a = eval_rat(graph, ins[0], fields, scalars);
            let b = eval_rat(graph, ins[1], fields, scalars);
            combine_rat_add(a, b, matches!(op, ElementWiseOp::Sub))
        }
        ElementWiseOp::Neg => match eval_rat(graph, ins[0], fields, scalars) {
            RValue::Scalar(p) => RValue::Scalar(p.neg()),
            RValue::Lin(lf) => RValue::Lin(lf.neg_form()),
        },
        ElementWiseOp::Abs => {
            // the ONLY abs in a metric volume factor is |sin(theta)| (sqrt_det_gamma's angular
            // measure); the CT domain theta in (0, pi) has sin >= 0, so |sin| = sin. the opaque
            // sin_th@ symbol already denotes that non-negative quantity, hence abs is the identity
            // here. (a field-linear argument would be a nonlinear |B|, which no curl weight contains.)
            match eval_rat(graph, ins[0], fields, scalars) {
                RValue::Scalar(r) => RValue::Scalar(r),
                RValue::Lin(_) => panic!("proof(rat): abs of a field-dependent argument — nonlinear"),
            }
        }
        ElementWiseOp::Cast(_) => {
            // the I32->F64 promotion of a `_coord_N` (the only Cast the curl emits);
            // a no-op for the symbolic value (c_N is already a poly var).
            eval_rat(graph, ins[0], fields, scalars)
        }
        ElementWiseOp::Mul => {
            let a = eval_rat(graph, ins[0], fields, scalars);
            let b = eval_rat(graph, ins[1], fields, scalars);
            combine_rat_mul(a, b)
        }
        ElementWiseOp::Div => {
            let a = eval_rat(graph, ins[0], fields, scalars);
            let b = eval_rat(graph, ins[1], fields, scalars);
            combine_rat_div(a, b)
        }
        ElementWiseOp::Sin => {
            // sin of an AFFINE theta argument x_lo_1 + (c_1 + m)*dx_1 -> the opaque
            // symbol sin_th@<2m>. resolving the offset m off the argument poly is
            // the heart of the curvilinear representation.
            let arg = match eval_rat(graph, ins[0], fields, scalars) {
                RValue::Scalar(r) => r,
                RValue::Lin(_) => panic!("proof(rat): sin of a field-dependent argument — nonlinear"),
            };
            RValue::Scalar(RatFun::from_poly(sin_symbol(&arg)))
        }
        ElementWiseOp::Cos => {
            // cos of the SAME affine theta argument -> the opaque cos_th@<2m> symbol,
            // keyed identically to sin so the same global theta-edge shares the symbol.
            // the spherical r-face solid angle Omega = (cos(tl) - cos(th)) dphi uses this.
            let arg = match eval_rat(graph, ins[0], fields, scalars) {
                RValue::Scalar(r) => r,
                RValue::Lin(_) => panic!("proof(rat): cos of a field-dependent argument — nonlinear"),
            };
            RValue::Scalar(RatFun::from_poly(cos_symbol(&arg)))
        }
        ElementWiseOp::Sqrt => {
            // sqrt of a radial metric factor f(r) (Schwarzschild f = 1 - 2M/r, Kerr-Schild h =
            // 1 + 2M/r) -> the opaque `sqrt_f@<2m>` symbol keyed by the radial offset the argument's
            // r = x_lo_0 + (c_0 + m) dx_0 resolves to. sqrt of a field-linear form would be nonlinear
            // (a curl weight is never sqrt of a field).
            let arg = match eval_rat(graph, ins[0], fields, scalars) {
                RValue::Scalar(r) => r,
                RValue::Lin(_) => panic!("proof(rat): sqrt of a field-dependent argument — nonlinear"),
            };
            // an AFFINE radial argument keys by its radial offset (`sqrt_f@<2m>`, shift-remappable —
            // the spherical Schwarzschild/Kerr-Schild path). a NON-affine one (nested sqrt(R^2+z^2),
            // Kerr Sigma) keys by its exact canonical form: two occurrences of the same argument at
            // the same face share the atom, so the div-weight's `1/sqrt` cancels the curl's `sqrt`
            // LOCALLY (num/den) with no shift-remap needed (the caller cancels the metric before any
            // divergence shift).
            let atom = match radial_offset_two_m(&arg) {
                Some(two_m) => Poly::sqrt_f_sym(two_m),
                None => Poly::var(&format!("sqrt[{}]", arg.canonical())),
            };
            RValue::Scalar(RatFun::from_poly(atom))
        }
        ElementWiseOp::Max => {
            // max of two field-independent metric factors (the r >= M/2 singular-core
            // clamp: max(sqrt(x^2 + ...), M/2)) -> ONE opaque atom keyed by BOTH
            // operands' canonical forms. the div(curl) telescope is STRUCTURAL — each
            // face weight must merely be THE SAME expression in the two adjacent cell
            // divergences — so the proof needs max only as a shared atom and can
            // ignore its semantics. atomization is conservative: it can only fail to prove a
            // true zero, never prove a false one. max of a field-dependent argument
            // would be nonlinear in the fields and stays rejected.
            let a = match eval_rat(graph, ins[0], fields, scalars) {
                RValue::Scalar(r) => r,
                RValue::Lin(_) => panic!("proof(rat): max of a field-dependent argument — nonlinear"),
            };
            let b = match eval_rat(graph, ins[1], fields, scalars) {
                RValue::Scalar(r) => r,
                RValue::Lin(_) => panic!("proof(rat): max of a field-dependent argument — nonlinear"),
            };
            let atom = Poly::var(&format!("max[{}|{}]", a.canonical(), b.canonical()));
            RValue::Scalar(RatFun::from_poly(atom))
        }
        other => panic!("proof(rat): unsupported element-wise op in curvilinear curl DAG: {other:?}"),
    }
}

/// turn an affine theta argument into the opaque `sin_th@<2m>` symbol. the
/// argument is `x_lo_1 + (c_1 + m)*dx_1` for a CONSTANT offset `m` (in units of
/// dx_1) — so the symbol depends only on the GLOBAL theta offset m (the shared-edge
/// property the telescope relies on). m may be half-integral (the (face0+face1)/2
/// cell-center average makes the argument a 1/2-denominator rational), so the key
/// is 2m as an integer.
///
/// the denominator must be a nonzero integer constant D (the center average's /2,
/// or 1 for an edge face). reading the offset: in D*arg the x_lo_1 and c_1*dx_1
/// coefficients are both D, and the bare-dx_1 coefficient is D*m, so 2m =
/// 2*(D*m)/D.
fn sin_symbol(arg: &RatFun) -> Poly {
    Poly::sin_sym(theta_offset_two_m(arg, "sin"))
}

/// cos of the SAME affine theta argument -> `cos_th@<2m>`, keyed identically to sin
/// (the global theta edge maps to the same half-unit offset 2m), so the spherical
/// solid-angle cos(tl)-cos(th) shares the edge symbol across adjacent cells.
fn cos_symbol(arg: &RatFun) -> Poly {
    Poly::cos_sym(theta_offset_two_m(arg, "cos"))
}

/// resolve the half-unit theta offset 2m off an affine sin/cos argument
/// `x_lo_1 + (c_1 + m)*dx_1` (denominator D from the cell-center /2). in D*arg the
/// x_lo_1 and c_1*dx_1 coeffs are both D and the bare-dx_1 coeff is D*m, so
/// 2m = 2*(D*m)/D. `trig` names the caller for the panic messages.
fn theta_offset_two_m(arg: &RatFun, trig: &str) -> i64 {
    let d = const_value(&arg.den)
        .unwrap_or_else(|| panic!("proof(rat): {trig} of a theta argument with a non-constant denominator"));
    assert!(d != 0, "proof(rat): {trig} theta argument has a zero denominator");
    let xlo1 = arg.num.coefficient_of(&["x_lo_1"]);
    let c1dx1 = arg.num.coefficient_of(&["c_1", "dx_1"]);
    let dx1_offset = arg.num.coefficient_of(&["dx_1"]); // the bare dx_1*(D*m) term
    assert!(xlo1 == d, "proof(rat): theta arg x_lo_1 coeff {xlo1} != denominator {d}");
    assert!(c1dx1 == d, "proof(rat): theta arg c_1*dx_1 coeff {c1dx1} != denominator {d}");
    // 2m = 2 * (D*m) / D, must be integral.
    let twice = 2 * dx1_offset;
    assert!(twice % d == 0, "proof(rat): theta offset 2m = {twice}/{d} is not a half-integer multiple of dx_1");
    twice / d
}

/// resolve the RADIAL half-unit offset 2m off a `sqrt(f(r))` argument. the argument is a radial
/// metric factor f(r) = (r +- 2M)/r, whose DENOMINATOR is the radius r = x_lo_0 + (c_0 + m)*dx_0
/// (affine in c_0; a cell center clears to a /2 denominator, so the coefficients scale by D). read
/// the offset off the denominator exactly as `theta_offset_two_m` reads it off the trig argument:
/// D is the x_lo_0 coefficient (1 for a face, 2 for a cleared /2 center), the c_0*dx_0 coefficient
/// must also be D, and the bare dx_0 coefficient is D*m, so 2m = 2*(D*m)/D. (M rides only in the
/// numerator, so the denominator is a clean radius — the offset is unambiguous.)
fn radial_offset_two_m(arg: &RatFun) -> Option<i64> {
    // affine radial factor f(r) = (r +- 2M)/r with r = x_lo_0 + (c_0 + m) dx_0: the denominator is
    // that AFFINE radius, so `x_lo_0` and `c_0*dx_0` both appear with the same nonzero coefficient D.
    // a NON-affine argument (a nested sqrt(R^2+z^2), Kerr's Sigma, ...) has no linear x_lo_0 term in
    // its denominator -> D = 0 -> None (the caller falls back to a canonical-argument atom).
    let d = arg.den.coefficient_of(&["x_lo_0"]);
    if d == 0 {
        return None;
    }
    let c0dx0 = arg.den.coefficient_of(&["c_0", "dx_0"]);
    if c0dx0 != d {
        return None;
    }
    let twice = 2 * arg.den.coefficient_of(&["dx_0"]);
    if twice % d != 0 {
        return None;
    }
    Some(twice / d)
}

/// the integer value of a constant polynomial (the empty monomial's coefficient if
/// it is the ONLY term), else None.
fn const_value(p: &Poly) -> Option<i64> {
    if p.terms.is_empty() {
        return Some(0);
    }
    if p.terms.len() == 1 {
        if let Some((mono, &c)) = p.terms.iter().next() {
            if mono.is_empty() {
                return Some(c);
            }
        }
    }
    None
}

fn combine_rat_add(a: RValue, b: RValue, neg_b: bool) -> RValue {
    match (a, b) {
        (RValue::Scalar(pa), RValue::Scalar(pb)) => {
            RValue::Scalar(if neg_b { pa.sub(&pb) } else { pa.add(&pb) })
        }
        (RValue::Lin(la), RValue::Lin(lb)) => {
            let mut lf = la;
            lf.add(&if neg_b { lb.neg_form() } else { lb });
            RValue::Lin(lf)
        }
        (RValue::Scalar(_), RValue::Lin(lb)) => RValue::Lin(if neg_b { lb.neg_form() } else { lb }),
        (RValue::Lin(la), RValue::Scalar(_)) => RValue::Lin(la),
    }
}

fn combine_rat_mul(a: RValue, b: RValue) -> RValue {
    match (a, b) {
        (RValue::Scalar(pa), RValue::Scalar(pb)) => RValue::Scalar(pa.mul(&pb)),
        (RValue::Scalar(p), RValue::Lin(lf)) | (RValue::Lin(lf), RValue::Scalar(p)) => {
            RValue::Lin(lf.scaled(&p))
        }
        (RValue::Lin(_), RValue::Lin(_)) => {
            panic!("proof(rat): field*field product in curl DAG — nonlinear")
        }
    }
}

fn combine_rat_div(a: RValue, b: RValue) -> RValue {
    match (a, b) {
        (RValue::Scalar(pa), RValue::Scalar(pb)) => RValue::Scalar(pa.div(&pb)),
        (RValue::Lin(lf), RValue::Scalar(p)) => {
            // dividing a field form by a scalar = scaling by its reciprocal.
            let recip = RatFun::from_poly(Poly::constant(1)).div(&p);
            RValue::Lin(lf.scaled(&recip))
        }
        (RValue::Scalar(_), RValue::Lin(_)) | (RValue::Lin(_), RValue::Lin(_)) => {
            panic!("proof(rat): division by a field-dependent value — nonlinear")
        }
    }
}

impl LinForm {
    /// extract the symbolic linear form of the DAG rooted at `root`. `field_keys`
    /// are the keys treated as per-cell field reads (the staggered emf/B reads
    /// the divergence operates on); `scalar_keys` are uniform coefficient params
    /// (dt, inverse widths). any other `Param` is index-only (a `_coord_N`) and
    /// must only appear inside a `LoadAt` offset (never as a value) — the
    /// cartesian curl satisfies this.
    pub fn extract(
        graph: &Graph,
        root: NodeId,
        field_keys: &[&str],
        scalar_keys: &[&str],
    ) -> LinForm {
        match eval(graph, root, field_keys, scalar_keys) {
            Value::Lin(lf) => lf,
            // a pure-scalar root has no field reads — an empty linear form.
            Value::Scalar(_) => LinForm::empty(),
        }
    }
}

/// the result of interpreting a DAG subtree: either a pure scalar polynomial (no
/// field reads — a coefficient) or a full linear form (field reads scaled by
/// scalar polynomials). keeping the two apart lets `Mul` require exactly one PURE
/// side (a field*field product would be nonlinear and is rejected — the cartesian
/// curl never produces one).
enum Value {
    Scalar(Poly),
    Lin(LinForm),
}

/// recursively interpret the DAG node as a `Value`. panics on any op the
/// cartesian curl does not produce (field*field, transcendental on a field,
/// etc.) — a curl that hit one would be NONLINEAR and the symbolic proof would
/// not apply; loudly failing is correct.
fn eval(graph: &Graph, id: NodeId, fields: &[&str], scalars: &[&str]) -> Value {
    let node = graph.node(id);
    match &node.op {
        Op::Const(ConstValue::I32(_)) => {
            // an integer coord-offset constant only ever appears inside a LoadAt
            // component (resolved there); as a value it is degenerate. treat as 0.
            Value::Scalar(Poly::zero())
        }
        Op::Const(ConstValue::F64(v)) => {
            // a float literal coefficient (0.5, 1.0, etc.). it must be integral
            // for the integer-polynomial representation; the cartesian curl only
            // uses 0/1 literals (the curvilinear inv_pref path uses Gv::ONE only).
            let r = v.round();
            assert!(
                (v - r).abs() < 1e-12,
                "proof: non-integer float literal {v} in curl DAG — needs rational coeffs"
            );
            Value::Scalar(Poly::constant(r as i64))
        }
        Op::Param(sym) => {
            let key = sym.as_str();
            if fields.contains(&key) {
                // a direct cell read (offset 0) of a field.
                let ndim = infer_ndim(graph);
                Value::Lin(LinForm::from_term((key.to_string(), vec![0; ndim]), Poly::constant(1)))
            } else if scalars.contains(&key) {
                Value::Scalar(Poly::var(key))
            } else {
                // a coord param used as a value is unexpected in a curl DAG.
                panic!("proof: bare coord/unknown param `{key}` used as a value");
            }
        }
        Op::LoadAt(sym, comps) => {
            let key = sym.as_str();
            assert!(fields.contains(&key), "proof: LoadAt of non-field key `{key}`");
            let off: Vec<i32> = comps.iter().map(|&c| resolve_offset(graph, c)).collect();
            Value::Lin(LinForm::from_term((key.to_string(), off), Poly::constant(1)))
        }
        Op::ElementWise(op, ins) => eval_elementwise(graph, *op, ins, fields, scalars),
        // the spacing `map_kind` cond is a leaf face-position reparametrization; div(curl) = 0
        // telescopes independently of it (see the eval_rat arm), so extract the uniform (else) arm.
        Op::IfElse { else_results, .. } => eval(graph, else_results[0], fields, scalars),
        other => panic!("proof: unsupported op in curl DAG: {other:?}"),
    }
}

fn eval_elementwise(
    graph: &Graph,
    op: ElementWiseOp,
    ins: &[NodeId],
    fields: &[&str],
    scalars: &[&str],
) -> Value {
    match op {
        ElementWiseOp::Add | ElementWiseOp::Sub => {
            let a = eval(graph, ins[0], fields, scalars);
            let b = eval(graph, ins[1], fields, scalars);
            combine_add(a, b, matches!(op, ElementWiseOp::Sub))
        }
        ElementWiseOp::Neg => match eval(graph, ins[0], fields, scalars) {
            Value::Scalar(p) => Value::Scalar(p.neg()),
            Value::Lin(lf) => Value::Lin(lf.neg()),
        },
        ElementWiseOp::Mul => {
            let a = eval(graph, ins[0], fields, scalars);
            let b = eval(graph, ins[1], fields, scalars);
            combine_mul(a, b)
        }
        other => panic!("proof: unsupported element-wise op in curl DAG: {other:?}"),
    }
}

fn combine_add(a: Value, b: Value, neg_b: bool) -> Value {
    match (a, b) {
        (Value::Scalar(pa), Value::Scalar(pb)) => {
            let mut p = pa;
            p.add_assign(&if neg_b { pb.neg() } else { pb });
            Value::Scalar(p)
        }
        (Value::Lin(la), Value::Lin(lb)) => {
            let mut lf = la;
            lf.add_assign(&if neg_b { lb.neg() } else { lb });
            Value::Lin(lf)
        }
        // a scalar added to a linear form contributes no field read — it only
        // shifts the field-free constant part, irrelevant to div(B). drop it.
        (Value::Scalar(_), Value::Lin(lb)) => Value::Lin(if neg_b { lb.neg() } else { lb }),
        (Value::Lin(la), Value::Scalar(_)) => Value::Lin(la),
    }
}

fn combine_mul(a: Value, b: Value) -> Value {
    match (a, b) {
        (Value::Scalar(pa), Value::Scalar(pb)) => Value::Scalar(pa.mul(&pb)),
        (Value::Scalar(p), Value::Lin(lf)) | (Value::Lin(lf), Value::Scalar(p)) => {
            Value::Lin(lf.scaled(&p))
        }
        (Value::Lin(_), Value::Lin(_)) => {
            panic!("proof: field*field product in curl DAG — nonlinear, symbolic proof N/A")
        }
    }
}

/// resolve a LoadAt component NodeId to its integer offset relative to the cell
/// coord. the component is either the bare `_coord_N` param (offset 0) or
/// `Add(_coord_N, Const(I32(off)))`.
fn resolve_offset(graph: &Graph, id: NodeId) -> i32 {
    match &graph.node(id).op {
        Op::Param(_) => 0,
        Op::ElementWise(ElementWiseOp::Add, ins) => {
            // exactly one operand is the coord param, the other the const offset.
            let mut off = 0;
            for &c in ins {
                if let Op::Const(ConstValue::I32(v)) = &graph.node(c).op {
                    off += *v;
                }
            }
            off
        }
        other => panic!("proof: unexpected LoadAt component op: {other:?}"),
    }
}

/// infer the stencil ndim from the widest LoadAt component count in the graph —
/// the direct `Param` field reads (offset 0) need a zero offset of this length so
/// they line up with the shifted reads. cartesian 3D curl => 3.
fn infer_ndim(graph: &Graph) -> usize {
    let mut ndim = 0;
    for (_, node, _) in graph.iter() {
        if let Op::LoadAt(_, comps) = &node.op {
            ndim = ndim.max(comps.len());
        }
    }
    ndim
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sin_symbol_keys_offset() {
        // theta argument x_lo_1 + c_1*dx_1 (offset 0) -> sin_th@0.
        let mut arg = Poly::var("x_lo_1");
        arg.add_assign(&Poly::var("c_1").mul(&Poly::var("dx_1")));
        let s0 = sin_symbol(&RatFun::from_poly(arg.clone()));
        assert!(s0.terms.keys().next().unwrap().contains_key("sin_th@0"));
        // offset +1 (edge): + dx_1 -> sin_th@2.
        let mut arg1 = arg.clone();
        arg1.add_assign(&Poly::var("dx_1"));
        let s1 = sin_symbol(&RatFun::from_poly(arg1));
        assert!(s1.terms.keys().next().unwrap().contains_key("sin_th@2"));
        // cell-center half offset via /2 rational: (2 x_lo_1 + (2 c_1 + 1) dx_1)/2 -> sin_th@1.
        let mut cnum = Poly::var("x_lo_1").mul(&Poly::constant(2));
        cnum.add_assign(&Poly::var("c_1").mul(&Poly::var("dx_1")).mul(&Poly::constant(2)));
        cnum.add_assign(&Poly::var("dx_1"));
        let sc = sin_symbol(&RatFun::new(cnum, Poly::constant(2)));
        assert!(sc.terms.keys().next().unwrap().contains_key("sin_th@1"));
    }

    #[test]
    fn sqrt_f_symbol_keys_radial_offset_and_shifts() {
        // f(r) = (r +- 2M)/r; only the DENOMINATOR r = x_lo_0 + (c_0 + m) dx_0 carries the radial
        // offset, so the numerator is immaterial. face offset 0 -> sqrt_f@0.
        let mut r0 = Poly::var("x_lo_0");
        r0.add_assign(&Poly::var("c_0").mul(&Poly::var("dx_0")));
        assert_eq!(radial_offset_two_m(&RatFun::new(Poly::constant(1), r0.clone())), Some(0));
        // face offset +1 -> sqrt_f@2.
        let mut r1 = r0.clone();
        r1.add_assign(&Poly::var("dx_0"));
        assert_eq!(radial_offset_two_m(&RatFun::new(Poly::constant(1), r1)), Some(2));
        // cell center (2 x_lo_0 + (2 c_0 + 1) dx_0)/2 -> sqrt_f@1.
        let mut rc = Poly::var("x_lo_0").mul(&Poly::constant(2));
        rc.add_assign(&Poly::var("c_0").mul(&Poly::var("dx_0")).mul(&Poly::constant(2)));
        rc.add_assign(&Poly::var("dx_0"));
        assert_eq!(radial_offset_two_m(&RatFun::new(Poly::constant(1), rc)), Some(1));
        // a NON-affine denominator (an opaque atom, no linear x_lo_0) -> None (canonical-key path).
        assert_eq!(radial_offset_two_m(&RatFun::new(Poly::constant(1), Poly::var("sqrt[inner]"))), None);
        // the atom remaps under a RADIAL (axis 0) shift — sqrt_f@0 -> sqrt_f@2 (one cell = +2
        // half-units) — and is UNTOUCHED by a theta (axis 1) shift, so it telescopes only radially.
        let atom = RatFun::from_poly(Poly::sqrt_f_sym(0));
        assert!(atom.shift_coords(&[1, 0]).num.terms.keys().next().unwrap().contains_key("sqrt_f@2"));
        assert!(atom.shift_coords(&[0, 1]).num.terms.keys().next().unwrap().contains_key("sqrt_f@0"));
    }
}
