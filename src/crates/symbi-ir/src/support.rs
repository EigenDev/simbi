// =============================================================================
// support.rs
//
// the SUPPORT of a kernel output: a region outside which its value is exactly
// zero in f64 — not just mathematically. the minimal
// lattice:
//   Everywhere — no bound is known (always sound)
//   Ball       — zero outside a coordinate ball; center/radius are expressions
//                over the kernel's own scalar params, evaluated at dispatch
//                time with the same values the kernel receives
//   Empty      — the output is identically zero
// declared by the kernel builder (where the saturation constants live),
// serialized into the neutral IR blob, and consumed by the dispatch layer:
// a reduction over a Ball-supported output only needs the cells inside the
// ball. the declaration is validated against the compiled kernel by sampling
// (outputs exactly zero outside the ball for arbitrary field values).
//
// usage:
//   let r = ParamExpr::param("body_0_racc")
//       + ParamExpr::constant(20.0) * ParamExpr::min_of(vec![...dx params...]);
//   kernel.with_output_support(Support::ball(centers, r));
//   // dispatch:
//   let radius = r.eval(&|name| scalar_value(name));
// =============================================================================

/// a scalar expression over a kernel's named scalar params — the language of
/// support geometry (a ball center coordinate, a radius). deliberately tiny:
/// grow it only when a declared support needs a form it cannot spell.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum ParamExpr {
    /// a kernel scalar param by its manifest name (e.g. "body_0_racc", "dx_0").
    Param(String),
    Const(f64),
    Add(Box<ParamExpr>, Box<ParamExpr>),
    Mul(Box<ParamExpr>, Box<ParamExpr>),
    /// the minimum over one or more sub-expressions (e.g. the smallest cell width).
    Min(Vec<ParamExpr>),
}

impl ParamExpr {
    pub fn param(name: &str) -> ParamExpr {
        ParamExpr::Param(name.to_string())
    }

    pub fn constant(v: f64) -> ParamExpr {
        ParamExpr::Const(v)
    }

    pub fn min_of(items: Vec<ParamExpr>) -> ParamExpr {
        assert!(!items.is_empty(), "ParamExpr::Min of nothing");
        ParamExpr::Min(items)
    }

    /// evaluate against a name -> value resolver (the dispatch scalar table).
    pub fn eval(&self, resolve: &impl Fn(&str) -> f64) -> f64 {
        match self {
            ParamExpr::Param(name) => resolve(name),
            ParamExpr::Const(v) => *v,
            ParamExpr::Add(a, b) => a.eval(resolve) + b.eval(resolve),
            ParamExpr::Mul(a, b) => a.eval(resolve) * b.eval(resolve),
            ParamExpr::Min(items) => items
                .iter()
                .map(|e| e.eval(resolve))
                .fold(f64::INFINITY, f64::min),
        }
    }
}

impl std::ops::Add for ParamExpr {
    type Output = ParamExpr;
    fn add(self, rhs: ParamExpr) -> ParamExpr {
        ParamExpr::Add(Box::new(self), Box::new(rhs))
    }
}

impl std::ops::Mul for ParamExpr {
    type Output = ParamExpr;
    fn mul(self, rhs: ParamExpr) -> ParamExpr {
        ParamExpr::Mul(Box::new(self), Box::new(rhs))
    }
}

/// the support of a kernel output. `Everywhere` is the sound default for
/// anything undeclared; `Ball` is the only bounded shape until a consumer
/// demands another.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum Support {
    Everywhere,
    /// exactly zero outside |x - center| > radius, in the grid's coordinate
    /// space, for EVERY field input value. one center component per grid axis.
    Ball {
        center: Vec<ParamExpr>,
        radius: ParamExpr,
    },
    Empty,
}

impl Support {
    pub fn ball(center: Vec<ParamExpr>, radius: ParamExpr) -> Support {
        Support::Ball { center, radius }
    }

    /// evaluate a Ball's geometry against the dispatch scalar table:
    /// (center, radius). `None` for Everywhere/Empty — no ball to evaluate.
    pub fn eval_ball(&self, resolve: &impl Fn(&str) -> f64) -> Option<(Vec<f64>, f64)> {
        match self {
            Support::Ball { center, radius } => Some((
                center.iter().map(|c| c.eval(resolve)).collect(),
                radius.eval(resolve),
            )),
            Support::Everywhere | Support::Empty => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn resolver<'a>(pairs: &'a [(&'a str, f64)]) -> impl Fn(&str) -> f64 + 'a {
        move |name: &str| {
            pairs
                .iter()
                .find(|(n, _)| *n == name)
                .map(|(_, v)| *v)
                .unwrap_or_else(|| panic!("unresolved param '{name}'"))
        }
    }

    #[test]
    fn param_expr_evaluates_the_radius_form() {
        // the drain support radius: racc + 20 * min(dx_0, dx_1).
        let r = ParamExpr::param("racc")
            + ParamExpr::constant(20.0)
                * ParamExpr::min_of(vec![ParamExpr::param("dx_0"), ParamExpr::param("dx_1")]);
        let v = r.eval(&resolver(&[
            ("racc", 0.15),
            ("dx_0", 0.0625),
            ("dx_1", 0.125),
        ]));
        assert_eq!(v, 0.15 + 20.0 * 0.0625);
    }

    #[test]
    fn ball_evaluates_center_and_radius() {
        let s = Support::ball(
            vec![ParamExpr::param("px"), ParamExpr::param("py")],
            ParamExpr::constant(2.0),
        );
        let (c, r) = s
            .eval_ball(&resolver(&[("px", -0.5), ("py", 0.25)]))
            .unwrap();
        assert_eq!(c, vec![-0.5, 0.25]);
        assert_eq!(r, 2.0);
        assert_eq!(Support::Everywhere.eval_ball(&resolver(&[])), None);
    }

    #[test]
    fn support_round_trips_through_serde() {
        let s = Support::ball(
            vec![ParamExpr::param("body_0_pos_0")],
            ParamExpr::param("body_0_racc")
                + ParamExpr::constant(20.0) * ParamExpr::min_of(vec![ParamExpr::param("dx_0")]),
        );
        let json = serde_json::to_string(&s).unwrap();
        let back: Support = serde_json::from_str(&json).unwrap();
        assert_eq!(back, s);
    }
}
