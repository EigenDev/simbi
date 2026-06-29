// =============================================================================
// proof.rs
//
// symbolic div(curl B) = 0 checker over the traced IR DAG. instead of running a
// curl kernel and asserting div(B) is ~1e-12, this EXTRACTS the exact symbolic
// linear combination of edge-emf field reads the curl produces and PROVES the
// constraint by polynomial-coefficient cancellation to the zero polynomial.
//
// the curl-of-an-edge-emf is LINEAR in the staggered field reads: the only
// nonlinearity is multiplication by uniform scalar params (dt, inverse widths).
// so a curl node lowers to a `LinForm` = a map from a field read `(key, offset)`
// to a `Poly` (a multivariate polynomial in the scalar-param names with integer
// coefficients). the divergence stencil is applied symbolically by SHIFTING the
// offsets; the contribution must vanish as the zero LinForm — that is the proof.
//
// usage:
//  let (kernel, writes) = some_cartesian_curl_builder();
//  let lf = LinForm::extract(&kernel.graph, writes[0].2, &fields, &scalars);
//  // shift / combine LinForms per the divergence stencil, then:
//  assert!(combined.is_zero());
// =============================================================================

use std::collections::BTreeMap;

use crate::graph::{ConstValue, ElementWiseOp, Graph, NodeId, Op};

/// a multivariate polynomial in scalar-param names with i64 coefficients. a
/// monomial is a sorted map `name -> power` (the empty map = the constant
/// monomial); the poly is `monomial -> coefficient`. zero coefficients are
/// pruned on every operation so `is_zero` is just "no terms".
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Poly {
    terms: BTreeMap<BTreeMap<String, u32>, i64>,
}

impl Poly {
    /// the zero polynomial.
    pub fn zero() -> Self {
        Poly { terms: BTreeMap::new() }
    }

    /// an integer constant polynomial.
    pub fn constant(c: i64) -> Self {
        let mut p = Poly::zero();
        if c != 0 {
            p.terms.insert(BTreeMap::new(), c);
        }
        p
    }

    /// the degree-1 monomial for a single scalar-param variable.
    pub fn var(name: &str) -> Self {
        let mut mono = BTreeMap::new();
        mono.insert(name.to_string(), 1u32);
        let mut p = Poly::zero();
        p.terms.insert(mono, 1);
        p
    }

    /// true iff every coefficient is zero (the canonical form prunes zeros, so
    /// this is just an emptiness check).
    pub fn is_zero(&self) -> bool {
        self.terms.is_empty()
    }

    /// the integer coefficient of the squarefree monomial that is the product of
    /// `vars` (each to the first power); 0 if that monomial is absent. lets a
    /// caller assert a specific advective pairing survives in an emf coefficient,
    /// e.g. `coefficient_of(&["vbar_x", "al_x"])` is the weight on the al*by_w
    /// product — the upwind-pairing proof.
    pub fn coefficient_of(&self, vars: &[&str]) -> i64 {
        let mut mono = BTreeMap::new();
        for v in vars {
            *mono.entry((*v).to_string()).or_insert(0u32) += 1;
        }
        self.terms.get(&mono).copied().unwrap_or(0)
    }

    fn add_assign(&mut self, other: &Poly) {
        for (mono, &c) in &other.terms {
            let e = self.terms.entry(mono.clone()).or_insert(0);
            *e += c;
            if *e == 0 {
                self.terms.remove(mono);
            }
        }
    }

    fn neg(&self) -> Poly {
        let mut p = Poly::zero();
        for (mono, &c) in &self.terms {
            p.terms.insert(mono.clone(), -c);
        }
        p
    }

    /// rewrite variable names through `rename` (`old -> new`); names absent from
    /// the map pass through. powers are preserved (a renamed var keeps its
    /// exponent). used to canonicalize the per-dir generic scalar params
    /// (id_p1/id_p2) to their physical-axis identity before telescoping.
    fn rename_vars(&self, rename: &std::collections::HashMap<String, String>) -> Poly {
        let mut out = Poly::zero();
        for (mono, &c) in &self.terms {
            let mut new_mono = BTreeMap::new();
            for (name, &pow) in mono {
                let nn = rename.get(name).cloned().unwrap_or_else(|| name.clone());
                *new_mono.entry(nn).or_insert(0) += pow;
            }
            let e = out.terms.entry(new_mono).or_insert(0);
            *e += c;
        }
        out.terms.retain(|_, &mut c| c != 0);
        out
    }

    fn mul(&self, other: &Poly) -> Poly {
        let mut out = Poly::zero();
        for (ma, &ca) in &self.terms {
            for (mb, &cb) in &other.terms {
                let mut mono = ma.clone();
                for (name, &pow) in mb {
                    *mono.entry(name.clone()).or_insert(0) += pow;
                }
                let e = out.terms.entry(mono).or_insert(0);
                *e += ca * cb;
            }
        }
        out.terms.retain(|_, &mut c| c != 0);
        out
    }

    /// substitute the variable `name -> name + delta` (integer shift), expanding
    /// `name^k -> (name + delta)^k` exactly by the binomial theorem (repeated
    /// multiply). this is the CURVILINEAR shift's effect on a coefficient that
    /// depends on the cell coord `c_N`: translating the cell by `delta` along axis
    /// N shifts the geometry r/sin arguments built from `c_N`. the cartesian proof
    /// did not need this (its coefficients are translation-invariant constants).
    fn subst_shift(&self, name: &str, delta: i64) -> Poly {
        if delta == 0 {
            return self.clone();
        }
        // (name + delta) as a degree-1 poly.
        let mut lin = Poly::var(name);
        lin.add_assign(&Poly::constant(delta));
        let mut out = Poly::zero();
        for (mono, &c) in &self.terms {
            let pow = mono.get(name).copied().unwrap_or(0);
            // the part of the monomial WITHOUT `name`.
            let mut rest_mono = mono.clone();
            rest_mono.remove(name);
            let mut term = Poly::zero();
            term.terms.insert(rest_mono, c);
            for _ in 0..pow {
                term = term.mul(&lin);
            }
            out.add_assign(&term);
        }
        out
    }

    /// public polynomial sum `self + other` (for tests assembling area weights).
    pub fn plus(&self, other: &Poly) -> Poly {
        let mut out = self.clone();
        out.add_assign(other);
        out
    }

    /// public polynomial product `self * other` (for tests assembling area weights).
    pub fn times(&self, other: &Poly) -> Poly {
        self.mul(other)
    }

    /// the opaque `sin(theta at half-unit offset `two_m`)` symbol as a degree-1
    /// monomial — the public constructor for tests building area weights by hand.
    /// two_m is 2m: an integer face offset has two_m = 2*offset; the cell-center
    /// has two_m = 2*c + 1 reduced to its half-unit (e.g. center of cell 0 = 1).
    pub fn sin_sym(two_m: i64) -> Poly {
        let mut p = Poly::zero();
        let mut mono = BTreeMap::new();
        mono.insert(format!("sin_th@{two_m}"), 1u32);
        p.terms.insert(mono, 1);
        p
    }

    /// rename one variable to another in-place across all monomials (used to remap
    /// an opaque `sin@m` symbol to `sin@(m + delta)` under a theta shift).
    fn rename_one(&self, from: &str, to: &str) -> Poly {
        let mut map = std::collections::HashMap::new();
        map.insert(from.to_string(), to.to_string());
        self.rename_vars(&map)
    }
}

// =============================================================================
// RATIONAL-FUNCTION coefficient layer — the CURVILINEAR extension.
//
// the spherical CT curl multiplies edge EMFs by scale-factor weights h_p (= r,
// r*sin(theta)) and divides by the face-center prefactor 1/(h_p1c h_p2c) and the
// transverse widths. so its coefficients are RATIONAL FUNCTIONS, not integer
// polynomials. the variable set is:
//   - x_lo_0, dx_0, c_0   : `r` at any offset is AFFINE x_lo_0 + (c_0 + off)*dx_0,
//                           a real polynomial (r^2 in an area equals r*r in an
//                           h-product — r must be a true symbol, not opaque).
//   - x_lo_1, dx_1, c_1   : the theta argument is likewise affine, but enters ONLY
//                           through sin(.) — see below.
//   - x_lo_2, dx_2, c_2   : phi; appears only in widths (sin has no phi dep).
//   - dt
//   - sin_th@<2m>         : sin(theta at offset m*dx_1 from the cell) is an OPAQUE
//                           variable keyed by 2m (half-units, so cell-center 1/2
//                           offsets are integral keys). there is NO polynomial
//                           relation between sin at distinct offsets; the SAME
//                           global theta-edge maps to the SAME symbol in adjacent
//                           cells — that shared-symbol property makes the
//                           divergence telescope.
// the coord-shift is COVARIANT: shifting the cell by e_dir shifts the field reads
// AND the c_N / sin@m the coefficients depend on (geometry at cell c+e_dir differs
// from geometry at cell c). this mirrors the numerical test's absolute-index area
// weights area_r/area_th/area_ph.
// =============================================================================

/// a rational function num/den over `Poly`. INVARIANT: `den` is never the zero
/// polynomial — denominators are products of nonzero affine r-polys, opaque sin
/// symbols, and nonzero integer constants, so cross-multiplication never divides
/// by zero. `is_zero` is `num.is_zero()` (the denominator is structurally
/// nonzero); equality is by cross-multiplied numerator difference. no gcd
/// reduction is performed — common-denominator + numerator-zero suffices to PROVE
/// cancellation, which is all the div(curl B)=0 checker needs (KISS).
#[derive(Clone, Debug)]
pub struct RatFun {
    num: Poly,
    den: Poly,
}

impl RatFun {
    fn from_poly(p: Poly) -> Self {
        RatFun { num: p, den: Poly::constant(1) }
    }

    /// public num/den constructor (for tests building the area weights). `den` must
    /// be structurally nonzero (a product of r-polys / sin symbols / constants).
    pub fn new(num: Poly, den: Poly) -> Self {
        assert!(!den.is_zero(), "proof(rat): RatFun with a zero denominator");
        RatFun { num, den }
    }

    /// the reciprocal 1/p of a (nonzero) polynomial.
    pub fn recip(p: Poly) -> Self {
        RatFun::new(Poly::constant(1), p)
    }

    fn zero() -> Self {
        RatFun::from_poly(Poly::zero())
    }

    fn is_zero(&self) -> bool {
        self.num.is_zero()
    }

    fn neg(&self) -> RatFun {
        RatFun { num: self.num.neg(), den: self.den.clone() }
    }

    fn add(&self, other: &RatFun) -> RatFun {
        // a/b + c/d = (a*d + c*b) / (b*d).
        let mut num = self.num.mul(&other.den);
        num.add_assign(&other.num.mul(&self.den));
        let den = self.den.mul(&other.den);
        RatFun { num, den }
    }

    fn sub(&self, other: &RatFun) -> RatFun {
        self.add(&other.neg())
    }

    /// the product of two rational functions (public for tests assembling the area
    /// weight from r and sin factors).
    pub fn mul(&self, other: &RatFun) -> RatFun {
        RatFun { num: self.num.mul(&other.num), den: self.den.mul(&other.den) }
    }

    fn div(&self, other: &RatFun) -> RatFun {
        // a/b / (c/d) = (a*d) / (b*c). other.num is structurally nonzero (it is a
        // product of r-polys / sin symbols / constants), so b*c stays nonzero.
        assert!(!other.num.is_zero(), "proof: division by a zero rational function");
        RatFun { num: self.num.mul(&other.den), den: self.den.mul(&other.num) }
    }

    /// apply the covariant coord shift to BOTH num and den: c_N -> c_N + delta_N
    /// and sin_th@<2m> -> sin_th@<2m + 2*delta_theta>.
    fn shift_coords(&self, delta: &[i64]) -> RatFun {
        RatFun { num: shift_poly_coords(&self.num, delta), den: shift_poly_coords(&self.den, delta) }
    }
}

/// shift a coefficient polynomial under a cell translation by `delta` (per axis):
/// substitute each `c_N -> c_N + delta[N]` and remap every opaque sin symbol
/// `sin_th@<2m>` to `sin_th@<2m + 2*delta[1]>` (theta is axis 1; the sin argument
/// translates by delta[1] cells).
fn shift_poly_coords(p: &Poly, delta: &[i64]) -> Poly {
    let mut out = p.clone();
    for (ax, &d) in delta.iter().enumerate() {
        if d != 0 {
            out = out.subst_shift(&format!("c_{ax}"), d);
        }
    }
    // remap sin symbols by the theta shift (axis 1).
    let dth = *delta.get(1).unwrap_or(&0);
    if dth != 0 {
        // collect the sin symbols present, then rename each by +2*dth half-units.
        let mut syms: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
        for mono in out.terms.keys() {
            for name in mono.keys() {
                if name.starts_with("sin_th@") {
                    syms.insert(name.clone());
                }
            }
        }
        for s in syms {
            let half_units: i64 = s["sin_th@".len()..].parse().expect("malformed sin symbol");
            let to = format!("sin_th@{}", half_units + 2 * dth);
            out = out.rename_one(&s, &to);
        }
    }
    out
}

/// a linear combination of field reads with RATIONAL-FUNCTION coefficients — the
/// curvilinear analog of `LinForm`. cancellation to all-zero numerators is the
/// curvilinear div(curl B)=0 proof.
#[derive(Clone, Debug, Default)]
pub struct LinFormR {
    pub terms: BTreeMap<FieldTerm, RatFun>,
}

impl LinFormR {
    fn empty() -> Self {
        LinFormR { terms: BTreeMap::new() }
    }

    fn insert(&mut self, term: FieldTerm, coeff: RatFun) {
        if coeff.is_zero() {
            return;
        }
        let e = self.terms.entry(term).or_insert_with(RatFun::zero);
        *e = e.add(&coeff);
    }

    /// accumulate `other` into self.
    pub fn add(&mut self, other: &LinFormR) {
        for (term, coeff) in &other.terms {
            self.insert(term.clone(), coeff.clone());
        }
        self.prune();
    }

    fn prune(&mut self) {
        self.terms.retain(|_, c| !c.is_zero());
    }

    fn scaled(&self, c: &RatFun) -> LinFormR {
        let mut out = LinFormR::empty();
        for (term, coeff) in &self.terms {
            out.insert(term.clone(), coeff.mul(c));
        }
        out
    }

    fn neg(&self) -> LinFormR {
        let mut out = LinFormR::empty();
        for (term, coeff) in &self.terms {
            out.terms.insert(term.clone(), coeff.neg());
        }
        out
    }

    /// the additive inverse (mirror of `LinForm::neg_form`).
    pub fn neg_form(&self) -> LinFormR {
        self.neg()
    }

    /// scale the whole form by a rational function = the single scalar-param
    /// variable `name` (e.g. an inverse width). kept for API symmetry; the
    /// curvilinear divergence weights are area RatFuns, applied via `scale_rat`.
    pub fn scale_var(&self, name: &str) -> LinFormR {
        self.scaled(&RatFun::from_poly(Poly::var(name)))
    }

    /// scale by an arbitrary rational function (the area weight is a RatFun).
    pub fn scale_rat(&self, c: &RatFun) -> LinFormR {
        self.scaled(c)
    }

    /// shift every field read's offset by `delta` (per-axis) AND apply the same
    /// shift COVARIANTLY to each coefficient's coord/sin dependence. THIS is the
    /// curvilinear divergence stencil: the curl at cell c+e_dir uses the geometry
    /// at c+e_dir, so the coefficients must translate with the read.
    pub fn shifted(&self, delta: &[i32]) -> LinFormR {
        let dl: Vec<i64> = delta.iter().map(|&d| d as i64).collect();
        let mut out = LinFormR::empty();
        for ((key, off), coeff) in &self.terms {
            let new_off: Vec<i32> = off.iter().zip(delta).map(|(a, b)| a + b).collect();
            out.terms.insert((key.clone(), new_off), coeff.shift_coords(&dl));
        }
        out
    }

    /// rename the form's field keys through `rename` (the per-dir generic emf keys
    /// e_p1/e_p2 -> physical e_<axis>). UNLIKE the cartesian `canonicalize`, the
    /// per-dir generic SCALAR widths do not appear in the curvilinear coefficients
    /// (the geometry is built from the absolute axis scalars x_lo_N/dx_N, NOT
    /// per-dir id_pN), so only the field keys are renamed here.
    pub fn canonicalize_keys(&self, rename: &std::collections::HashMap<String, String>) -> LinFormR {
        let mut out = LinFormR::empty();
        for ((key, off), coeff) in &self.terms {
            let new_key = rename.get(key).cloned().unwrap_or_else(|| key.clone());
            out.insert((new_key, off.clone()), coeff.clone());
        }
        out
    }

    /// true iff every coefficient's numerator cancels — the curvilinear div(B)=0
    /// proof condition.
    pub fn is_zero(&self) -> bool {
        self.terms.values().all(|c| c.is_zero())
    }

    /// the non-cancelling terms (field read + residual numerator), for diagnostics
    /// when the proof FAILS.
    pub fn residual(&self) -> Vec<(FieldTerm, Poly)> {
        self.terms
            .iter()
            .filter(|(_, c)| !c.is_zero())
            .map(|(t, c)| (t.clone(), c.num.clone()))
            .collect()
    }

    /// a single-term form `coeff * read(term)` — public constructor for the
    /// negative-control test (coeff is a bare scalar-var rational).
    pub fn single_var(term: FieldTerm, var: &str) -> Self {
        let mut out = LinFormR::empty();
        out.insert(term, RatFun::from_poly(Poly::var(var)));
        out
    }

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
                RValue::Scalar(RatFun { num: Poly::constant(1), den: two })
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
            RValue::Lin(lf) => RValue::Lin(lf.neg()),
        },
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
    let d = const_value(&arg.den)
        .expect("proof(rat): sin of a theta argument with a non-constant denominator");
    assert!(d != 0, "proof(rat): sin theta argument has a zero denominator");
    let xlo1 = arg.num.coefficient_of(&["x_lo_1"]);
    let c1dx1 = arg.num.coefficient_of(&["c_1", "dx_1"]);
    let dx1_offset = arg.num.coefficient_of(&["dx_1"]); // the bare dx_1*(D*m) term
    assert!(xlo1 == d, "proof(rat): theta arg x_lo_1 coeff {xlo1} != denominator {d}");
    assert!(c1dx1 == d, "proof(rat): theta arg c_1*dx_1 coeff {c1dx1} != denominator {d}");
    // 2m = 2 * (D*m) / D, must be integral.
    let twice = 2 * dx1_offset;
    assert!(twice % d == 0, "proof(rat): theta offset 2m = {twice}/{d} is not a half-integer multiple of dx_1");
    let two_m = twice / d;
    let mut p = Poly::zero();
    let mut mono = BTreeMap::new();
    mono.insert(format!("sin_th@{two_m}"), 1u32);
    p.terms.insert(mono, 1);
    p
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
            lf.add(&if neg_b { lb.neg() } else { lb });
            RValue::Lin(lf)
        }
        (RValue::Scalar(_), RValue::Lin(lb)) => RValue::Lin(if neg_b { lb.neg() } else { lb }),
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

/// a field read at a known integer stencil offset — the LEAF of a curl DAG. the
/// key is the IR field-load name (e.g. `e_p1`); the offset is the per-axis
/// integer shift the `LoadAt` (or direct `Param`) read resolves to.
pub type FieldTerm = (String, Vec<i32>);

/// a linear combination of field reads with polynomial (in scalar params)
/// coefficients: `sum_t poly_t * read(t)`. this is the symbolic value of any
/// curl-of-edge-emf expression. cancellation to the empty map is the div(B)=0
/// proof.
#[derive(Clone, Debug, Default)]
pub struct LinForm {
    pub terms: BTreeMap<FieldTerm, Poly>,
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

impl LinForm {
    fn empty() -> Self {
        LinForm { terms: BTreeMap::new() }
    }

    fn from_term(term: FieldTerm, coeff: Poly) -> Self {
        let mut lf = LinForm::empty();
        if !coeff.is_zero() {
            lf.terms.insert(term, coeff);
        }
        lf
    }

    /// a single-term linear form `coeff * read(term)` — the public constructor
    /// for tests building reference forms by hand.
    pub fn single(term: FieldTerm, coeff: Poly) -> Self {
        Self::from_term(term, coeff)
    }

    /// the additive inverse of the whole form (public mirror of the internal
    /// `neg`, for the divergence's `- curl[+0]` term).
    pub fn neg_form(&self) -> LinForm {
        self.neg()
    }

    /// rewrite the form's field keys and scalar-param variable names through
    /// `rename` (`old -> new`). the per-dir curl builder reuses the GENERIC keys
    /// `e_p1`/`e_p2` and scalars `id_p1`/`id_p2` for all three face axes, but
    /// they denote DIFFERENT physical emf components / inverse widths per dir
    /// (the runtime dispatch binds the real buffers positionally). to telescope
    /// the three dirs symbolically they must first be canonicalized to their
    /// physical-axis identity. keys not in `rename` pass through unchanged.
    pub fn canonicalize(&self, rename: &std::collections::HashMap<String, String>) -> LinForm {
        let mut out = LinForm::empty();
        for ((key, off), poly) in &self.terms {
            let new_key = rename.get(key).cloned().unwrap_or_else(|| key.clone());
            let new_poly = poly.rename_vars(rename);
            let e = out.terms.entry((new_key, off.clone())).or_insert_with(Poly::zero);
            e.add_assign(&new_poly);
            if e.is_zero() {
                // leave for the final retain below.
            }
        }
        out.terms.retain(|_, p| !p.is_zero());
        out
    }

    fn add_assign(&mut self, other: &LinForm) {
        for (term, poly) in &other.terms {
            let e = self.terms.entry(term.clone()).or_insert_with(Poly::zero);
            e.add_assign(poly);
            if e.is_zero() {
                self.terms.remove(term);
            }
        }
    }

    fn scaled(&self, p: &Poly) -> LinForm {
        let mut out = LinForm::empty();
        for (term, poly) in &self.terms {
            let c = poly.mul(p);
            if !c.is_zero() {
                out.terms.insert(term.clone(), c);
            }
        }
        out
    }

    fn neg(&self) -> LinForm {
        let mut out = LinForm::empty();
        for (term, poly) in &self.terms {
            out.terms.insert(term.clone(), poly.neg());
        }
        out
    }

    /// shift every field read's offset by `delta` (per-axis). this is how the
    /// divergence stencil `q[+e_dir] - q[+0]` is applied symbolically: shift the
    /// curl LinForm by `+e_dir`, subtract the unshifted, accumulate.
    pub fn shifted(&self, delta: &[i32]) -> LinForm {
        let mut out = LinForm::empty();
        for ((key, off), poly) in &self.terms {
            let new_off: Vec<i32> = off.iter().zip(delta).map(|(a, b)| a + b).collect();
            out.terms.insert((key.clone(), new_off), poly.clone());
        }
        out
    }

    /// accumulate `other` into self (used to sum the per-dir divergence
    /// contributions).
    pub fn add(&mut self, other: &LinForm) {
        self.add_assign(other);
    }

    /// scale the whole linear form by a single scalar-param variable (the
    /// divergence's per-axis inverse width `id_dir`).
    pub fn scale_var(&self, name: &str) -> LinForm {
        self.scaled(&Poly::var(name))
    }

    /// true iff the linear form is identically zero — every coefficient
    /// polynomial cancelled. THIS is the div(B)=0 proof condition.
    pub fn is_zero(&self) -> bool {
        self.terms.values().all(|p| p.is_zero())
    }

    /// the non-cancelling terms, for diagnostics when the proof FAILS — a real
    /// bug-discovery report (which field read at which offset survived, with its
    /// residual coefficient).
    pub fn residual(&self) -> Vec<(FieldTerm, Poly)> {
        self.terms
            .iter()
            .filter(|(_, p)| !p.is_zero())
            .map(|(t, p)| (t.clone(), p.clone()))
            .collect()
    }

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
    fn poly_arithmetic_cancels() {
        // id_p2 * 1 - id_p2 * 1 = 0.
        let a = Poly::var("id_p2");
        let mut s = a.clone();
        s.add_assign(&a.neg());
        assert!(s.is_zero());
        // (dt * id_p1) is a degree-2 monomial, nonzero.
        let prod = Poly::var("dt").mul(&Poly::var("id_p1"));
        assert!(!prod.is_zero());
    }

    #[test]
    fn ratfun_arithmetic_and_zero() {
        // 1/r - 1/r = 0; 1/r + 1/r = 2/r (nonzero numerator).
        let r = Poly::var("x_lo_0"); // a stand-in nonzero denominator poly.
        let inv = RatFun { num: Poly::constant(1), den: r.clone() };
        assert!(inv.sub(&inv).is_zero());
        assert!(!inv.add(&inv).is_zero());
        // a/b == c/d cancellation by cross-multiply: (2)/(r) - (2*r)/(r*r) == 0.
        let two_over_r = RatFun { num: Poly::constant(2), den: r.clone() };
        let two_over_r2 = RatFun { num: Poly::constant(2).mul(&r), den: r.mul(&r) };
        assert!(two_over_r.sub(&two_over_r2).is_zero());
    }

    #[test]
    fn poly_subst_shift_binomial() {
        // (c_0 + 1)^2 = c_0^2 + 2 c_0 + 1 under c_0 -> c_0 + 1.
        let c0sq = Poly::var("c_0").mul(&Poly::var("c_0"));
        let shifted = c0sq.subst_shift("c_0", 1);
        // expected explicitly.
        let mut expect = Poly::var("c_0").mul(&Poly::var("c_0"));
        let mut two_c0 = Poly::var("c_0");
        two_c0 = two_c0.mul(&Poly::constant(2));
        expect.add_assign(&two_c0);
        expect.add_assign(&Poly::constant(1));
        let mut diff = shifted.clone();
        diff.add_assign(&expect.neg());
        assert!(diff.is_zero());
    }

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
        let sc = sin_symbol(&RatFun { num: cnum, den: Poly::constant(2) });
        assert!(sc.terms.keys().next().unwrap().contains_key("sin_th@1"));
    }

    #[test]
    fn shift_remaps_sin_symbol() {
        // a coefficient sin_th@1 under a theta shift +1 -> sin_th@3 (2*delta=2 half-units).
        let mut p = Poly::zero();
        let mut mono = BTreeMap::new();
        mono.insert("sin_th@1".to_string(), 1u32);
        p.terms.insert(mono, 1);
        let shifted = shift_poly_coords(&p, &[0, 1, 0]);
        assert!(shifted.terms.keys().next().unwrap().contains_key("sin_th@3"));
    }

    #[test]
    fn linform_shift_and_cancel() {
        // a single read e@[0,0,0] minus the same read shifted by [0,0,0] = 0;
        // shifted by [1,0,0] does NOT cancel (distinct offset key).
        let lf = LinForm::from_term(("e".into(), vec![0, 0, 0]), Poly::constant(1));
        let mut same = lf.clone();
        same.add(&lf.shifted(&[0, 0, 0]).neg());
        assert!(same.is_zero());
        let mut diff = lf.clone();
        diff.add(&lf.shifted(&[1, 0, 0]).neg());
        assert!(!diff.is_zero());
        assert_eq!(diff.residual().len(), 2);
    }
}
