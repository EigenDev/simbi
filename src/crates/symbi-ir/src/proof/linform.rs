// =============================================================================
// linform.rs
//
// the telescoping vector space: `LinForm` (integer-poly coefficients) and
// `LinFormR` (rational-function coefficients) — linear combinations of field
// reads `sum_t coeff_t * read(t)`. cancellation to the empty/zero map is the
// div(curl B)=0 proof condition. pure transforms only (add/scaled/shifted/
// canonicalize/neg/residual/...); the ir-dag extraction lives in extract.rs.
//
// usage:
//  let mut acc = curl_lf.shifted(&[1, 0, 0]);
//  acc.add(&curl_lf.neg_form());
//  assert!(acc.is_zero());
// =============================================================================

use std::collections::BTreeMap;

use super::poly::{FieldTerm, Poly, RatFun};

/// a linear combination of field reads with rational-function coefficients — the
/// curvilinear analog of `LinForm`. cancellation to all-zero numerators is the
/// curvilinear div(curl B)=0 proof.
#[derive(Clone, Debug, Default)]
pub struct LinFormR {
    pub terms: BTreeMap<FieldTerm, RatFun>,
}

impl LinFormR {
    pub(crate) fn empty() -> Self {
        LinFormR {
            terms: BTreeMap::new(),
        }
    }

    pub(crate) fn insert(&mut self, term: FieldTerm, coeff: RatFun) {
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

    pub(crate) fn scaled(&self, c: &RatFun) -> LinFormR {
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

    /// shift every field read's offset by `delta` (per-axis) and apply the same
    /// shift covariantly to each coefficient's coord/sin dependence. this is the
    /// curvilinear divergence stencil: the curl at cell c+e_dir uses the geometry
    /// at c+e_dir, so the coefficients must translate with the read.
    pub fn shifted(&self, delta: &[i32]) -> LinFormR {
        let dl: Vec<i64> = delta.iter().map(|&d| d as i64).collect();
        let mut out = LinFormR::empty();
        for ((key, off), coeff) in &self.terms {
            let new_off: Vec<i32> = off.iter().zip(delta).map(|(a, b)| a + b).collect();
            out.terms
                .insert((key.clone(), new_off), coeff.shift_coords(&dl));
        }
        out
    }

    /// rename the form's field keys through `rename` (the per-dir generic emf keys
    /// e_p1/e_p2 -> physical e_<axis>). unlike the cartesian `canonicalize`, the
    /// per-dir generic scalar widths do not appear in the curvilinear coefficients
    /// (the geometry is built from the absolute axis scalars x_lo_N/dx_N, not
    /// per-dir id_pN), so only the field keys are renamed here.
    pub fn canonicalize_keys(
        &self,
        rename: &std::collections::HashMap<String, String>,
    ) -> LinFormR {
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
    /// when the proof fails.
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
}

/// a linear combination of field reads with polynomial (in scalar params)
/// coefficients: `sum_t poly_t * read(t)`. this is the symbolic value of any
/// curl-of-edge-emf expression. cancellation to the empty map is the div(B)=0
/// proof.
#[derive(Clone, Debug, Default)]
pub struct LinForm {
    pub terms: BTreeMap<FieldTerm, Poly>,
}

impl LinForm {
    pub(crate) fn empty() -> Self {
        LinForm {
            terms: BTreeMap::new(),
        }
    }

    pub(crate) fn from_term(term: FieldTerm, coeff: Poly) -> Self {
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
    /// `rename` (`old -> new`). the per-dir curl builder reuses the generic keys
    /// `e_p1`/`e_p2` and scalars `id_p1`/`id_p2` for all three face axes, but
    /// they denote different physical emf components / inverse widths per dir
    /// (the runtime dispatch binds the real buffers positionally). to telescope
    /// the three dirs symbolically they must first be canonicalized to their
    /// physical-axis identity. keys not in `rename` pass through unchanged.
    pub fn canonicalize(&self, rename: &std::collections::HashMap<String, String>) -> LinForm {
        let mut out = LinForm::empty();
        for ((key, off), poly) in &self.terms {
            let new_key = rename.get(key).cloned().unwrap_or_else(|| key.clone());
            let new_poly = poly.rename_vars(rename);
            let e = out
                .terms
                .entry((new_key, off.clone()))
                .or_insert_with(Poly::zero);
            e.add_assign(&new_poly);
            if e.is_zero() {
                // leave for the final retain below.
            }
        }
        out.terms.retain(|_, p| !p.is_zero());
        out
    }

    pub(crate) fn add_assign(&mut self, other: &LinForm) {
        for (term, poly) in &other.terms {
            let e = self.terms.entry(term.clone()).or_insert_with(Poly::zero);
            e.add_assign(poly);
            if e.is_zero() {
                self.terms.remove(term);
            }
        }
    }

    pub(crate) fn scaled(&self, p: &Poly) -> LinForm {
        let mut out = LinForm::empty();
        for (term, poly) in &self.terms {
            let c = poly.mul(p);
            if !c.is_zero() {
                out.terms.insert(term.clone(), c);
            }
        }
        out
    }

    pub(crate) fn neg(&self) -> LinForm {
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
    /// polynomial cancelled. this is the div(B)=0 proof condition.
    pub fn is_zero(&self) -> bool {
        self.terms.values().all(|p| p.is_zero())
    }

    /// the non-cancelling terms, for diagnostics when the proof fails — a real
    /// bug-discovery report (which field read at which offset survived, with its
    /// residual coefficient).
    pub fn residual(&self) -> Vec<(FieldTerm, Poly)> {
        self.terms
            .iter()
            .filter(|(_, p)| !p.is_zero())
            .map(|(t, p)| (t.clone(), p.clone()))
            .collect()
    }
}
