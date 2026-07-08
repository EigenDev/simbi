// =============================================================================
// poly.rs
//
// the COEFFICIENT RING of the div(curl B)=0 proof: an integer multivariate
// polynomial `Poly` (in scalar-param names) and a rational function `RatFun`
// (num/den over `Poly`) for the curvilinear scale-factor weights. plus the
// `FieldTerm` leaf key (field name + integer stencil offset) and the covariant
// coord-shift `shift_poly_coords`. pure algebra — no IR/graph dependency.
//
// usage:
//  let p = Poly::var("dt").mul(&Poly::var("id_p1")); // a degree-2 monomial
//  let r = RatFun::recip(Poly::var("x_lo_0"));       // 1/r
// =============================================================================

use std::collections::BTreeMap;

/// a multivariate polynomial in scalar-param names with i64 coefficients. a
/// monomial is a sorted map `name -> power` (the empty map = the constant
/// monomial); the poly is `monomial -> coefficient`. zero coefficients are
/// pruned on every operation so `is_zero` is just "no terms".
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Poly {
    pub(crate) terms: BTreeMap<BTreeMap<String, u32>, i64>,
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

    pub(crate) fn add_assign(&mut self, other: &Poly) {
        for (mono, &c) in &other.terms {
            let e = self.terms.entry(mono.clone()).or_insert(0);
            *e += c;
            if *e == 0 {
                self.terms.remove(mono);
            }
        }
    }

    pub(crate) fn neg(&self) -> Poly {
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
    pub(crate) fn rename_vars(&self, rename: &std::collections::HashMap<String, String>) -> Poly {
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

    pub(crate) fn mul(&self, other: &Poly) -> Poly {
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
    pub(crate) fn subst_shift(&self, name: &str, delta: i64) -> Poly {
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

    /// the opaque `cos(theta at half-unit offset `two_m`)` symbol as a degree-1
    /// monomial — the cos analog of `sin_sym`, keyed the SAME way (`cos_th@<2m>`).
    /// the spherical r-face solid angle Omega = (cos(theta_lo) - cos(theta_hi)) dphi
    /// uses these; keyed by the global theta edge so adjacent cells share the symbol.
    pub fn cos_sym(two_m: i64) -> Poly {
        let mut p = Poly::zero();
        let mut mono = BTreeMap::new();
        mono.insert(format!("cos_th@{two_m}"), 1u32);
        p.terms.insert(mono, 1);
        p
    }

    /// the opaque `sqrt(f(r))` metric-lapse symbol at RADIAL half-unit offset `two_m` — the GR
    /// analog of `sin_sym`, keyed by the radial (axis 0) offset the sqrt argument resolves to (a
    /// face has two_m = 2*offset, a cell center two_m = 2*c + 1). the GR curl's `sqrt(gamma)` weight
    /// (Schwarzschild r/sqrt(f), Kerr-Schild r*sqrt(h)) carries this factor at each radial face;
    /// keying by the global radial edge lets the div-weight `sqrt(f)@<2m>` cancel the curl's
    /// `1/sqrt(f)@<2m>` at the SAME face (a rational num/den cancellation), and lets `shift_poly_coords`
    /// remap it across the divergence stencil so adjacent faces stay distinct. distinct offsets have
    /// no polynomial relation (sqrt(f) at r vs r+dr do not simplify), so an opaque symbol is exact.
    pub fn sqrt_f_sym(two_m: i64) -> Poly {
        let mut p = Poly::zero();
        let mut mono = BTreeMap::new();
        mono.insert(format!("sqrt_f@{two_m}"), 1u32);
        p.terms.insert(mono, 1);
        p
    }

    /// a STABLE canonical string of this polynomial — the `terms` map is a `BTreeMap` (sorted by
    /// monomial then var), so the debug form is deterministic. used to key an OPAQUE metric symbol by
    /// its argument's exact structure when the argument is NOT the affine radius the offset reader
    /// handles (a nested `sqrt(R^2 + z^2)`, Kerr's `Sigma`, ...): two occurrences of the same argument
    /// expression at the same face produce the SAME key, so they cancel locally (num/den).
    pub fn canonical(&self) -> String {
        format!("{:?}", self.terms)
    }

    /// rename one variable to another in-place across all monomials (used to remap
    /// an opaque `sin@m` symbol to `sin@(m + delta)` under a theta shift).
    pub(crate) fn rename_one(&self, from: &str, to: &str) -> Poly {
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
    pub(crate) num: Poly,
    pub(crate) den: Poly,
}

impl RatFun {
    pub(crate) fn from_poly(p: Poly) -> Self {
        RatFun { num: p, den: Poly::constant(1) }
    }

    /// a STABLE canonical string of this rational function (num + den), for keying an opaque metric
    /// symbol by its exact argument. see [`Poly::canonical`].
    pub fn canonical(&self) -> String {
        format!("{}//{}", self.num.canonical(), self.den.canonical())
    }

    /// the reciprocal `den/num` (swap). lets a proof recover the FACE AREA `w` from a flux-form GR
    /// curl whose edge-emf coefficient is `dt/w`: `w = dt * coeff.reciprocal()`. panics on a zero
    /// numerator (the reciprocal would divide by zero).
    pub fn reciprocal(&self) -> RatFun {
        assert!(!self.num.is_zero(), "proof: reciprocal of a zero rational function");
        RatFun { num: self.den.clone(), den: self.num.clone() }
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

    pub(crate) fn zero() -> Self {
        RatFun::from_poly(Poly::zero())
    }

    pub(crate) fn is_zero(&self) -> bool {
        self.num.is_zero()
    }

    pub(crate) fn neg(&self) -> RatFun {
        RatFun { num: self.num.neg(), den: self.den.clone() }
    }

    pub(crate) fn add(&self, other: &RatFun) -> RatFun {
        // a/b + c/d = (a*d + c*b) / (b*d).
        let mut num = self.num.mul(&other.den);
        num.add_assign(&other.num.mul(&self.den));
        let den = self.den.mul(&other.den);
        RatFun { num, den }
    }

    pub(crate) fn sub(&self, other: &RatFun) -> RatFun {
        self.add(&other.neg())
    }

    /// the product of two rational functions (public for tests assembling the area
    /// weight from r and sin factors).
    pub fn mul(&self, other: &RatFun) -> RatFun {
        RatFun { num: self.num.mul(&other.num), den: self.den.mul(&other.den) }
    }

    pub(crate) fn div(&self, other: &RatFun) -> RatFun {
        // a/b / (c/d) = (a*d) / (b*c). other.num is structurally nonzero (it is a
        // product of r-polys / sin symbols / constants), so b*c stays nonzero.
        assert!(!other.num.is_zero(), "proof: division by a zero rational function");
        RatFun { num: self.num.mul(&other.den), den: self.den.mul(&other.num) }
    }

    /// apply the covariant coord shift to BOTH num and den: c_N -> c_N + delta_N
    /// and sin_th@<2m>/cos_th@<2m> -> @<2m + 2*delta_theta>. public so a
    /// conservation test can form `area_lo.shift_coords(&e_r)` directly.
    pub fn shift_coords(&self, delta: &[i64]) -> RatFun {
        RatFun { num: shift_poly_coords(&self.num, delta), den: shift_poly_coords(&self.den, delta) }
    }

    /// EXACT symbolic equality: `self - other` is the zero rational function (the
    /// cross-multiplied numerator difference cancels to no terms). the public
    /// conservation-proof primitive (area_hi == area_lo-shifted).
    pub fn equals(&self, other: &RatFun) -> bool {
        self.sub(other).is_zero()
    }
}

/// shift a coefficient polynomial under a cell translation by `delta` (per axis):
/// substitute each `c_N -> c_N + delta[N]` and remap every opaque sin/cos symbol
/// `@<2m>` to `@<2m + 2*delta[1]>` (theta is axis 1; the trig argument translates
/// by delta[1] cells). a shift with delta[1] == 0 (e.g. the r-direction step) leaves
/// every theta-keyed symbol UNTOUCHED — the remap is purely along axis 1.
pub(crate) fn shift_poly_coords(p: &Poly, delta: &[i64]) -> Poly {
    let mut out = p.clone();
    for (ax, &d) in delta.iter().enumerate() {
        if d != 0 {
            out = out.subst_shift(&format!("c_{ax}"), d);
        }
    }
    // remap the opaque metric symbols by the shift along their axis: the trig sin_th@/cos_th@ track
    // theta (axis 1); the GR lapse sqrt_f@ tracks the radius (axis 0). each symbol at half-unit
    // offset <2m> moves to <2m + 2*delta[ax]> so a shifted contribution's edge symbol aligns with
    // the neighbor it telescopes against.
    remap_edge_symbols(&mut out, &["sin_th@", "cos_th@"], *delta.get(1).unwrap_or(&0));
    remap_edge_symbols(&mut out, &["sqrt_f@"], *delta.first().unwrap_or(&0));
    out
}

/// rename every opaque edge symbol with one of `prefixes` from `@<2m>` to `@<2m + 2*shift>` (a
/// half-unit-keyed translation along one axis). a zero shift leaves them untouched.
fn remap_edge_symbols(p: &mut Poly, prefixes: &[&str], shift: i64) {
    if shift == 0 {
        return;
    }
    for prefix in prefixes {
        let mut syms: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
        for mono in p.terms.keys() {
            for name in mono.keys() {
                if name.starts_with(prefix) {
                    syms.insert(name.clone());
                }
            }
        }
        for s in syms {
            let half_units: i64 = s[prefix.len()..].parse().expect("malformed edge symbol");
            let to = format!("{}{}", prefix, half_units + 2 * shift);
            *p = p.rename_one(&s, &to);
        }
    }
}

/// a field read at a known integer stencil offset — the LEAF of a curl DAG. the
/// key is the IR field-load name (e.g. `e_p1`); the offset is the per-axis
/// integer shift the `LoadAt` (or direct `Param`) read resolves to.
pub type FieldTerm = (String, Vec<i32>);
