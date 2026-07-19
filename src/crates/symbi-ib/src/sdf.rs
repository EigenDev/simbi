// =============================================================================
// sdf.rs
//
// signed-distance geometry as carrier-generic CSG:
// negative inside the body, positive outside. every operation is min/max/
// affine arithmetic plus one sqrt, so the SAME expression evaluates at f64
// (host reference + tests), at Gv (the traced penalization kernel), and at
// Dual<S> — which is how normals are computed: the exact gradient of the
// exact expression, one seeded evaluation per axis, no finite-difference
// step size. after CSG the gradient magnitude drifts from 1 (min/max kinks,
// nested offsets), so `normal` renormalizes; the DIRECTION survives.
//
// the f64 bounding ball encloses every point with dist <= 0 — the geometric
// input to a kernel's declared support (the consumer pads it by the chi
// saturation width). `Complement` is unbounded and yields no ball.
//
// usage:
//   let s = SdfExpr::sphere([0.0; 3], 1.0).union(SdfExpr::cuboid([2.0, 0.0, 0.0], [0.5; 3]));
//   let d = s.dist([x, y, z]);
//   let n = s.normal([x, y, z]);
//   let ball = s.bounding_ball();
// =============================================================================

use symbi_ir::algebra::Scalar;
use symbi_ir::dual::Dual;


/// the mollified indicator of a signed distance: 1 well inside the body, 0
/// well outside, a tanh ramp of width `w` across the surface. spelled
/// identically to `drain::drain_mask` (same subtraction folded into phi, same
/// division, same tanh), so a sphere SDF's chi is bit-equal to the mask the
/// validated drain uses.
pub fn chi<S: Scalar>(phi: S, w: S) -> S {
    S::from_f64(0.5) * (S::ONE - (phi / w).tanh())
}

/// a signed-distance expression over carrier `S` in `D` dimensions. shape
/// parameters are carrier values: f64 constants on the host, scalar-param
/// nodes in a trace — one structure serves both.
#[derive(Clone, Debug, PartialEq)]
pub enum SdfExpr<S, const D: usize> {
    Sphere { center: [S; D], radius: S },
    /// an axis-aligned box (exact euclidean distance, inside and out).
    Cuboid { center: [S; D], half_extents: [S; D] },
    /// min of distances: inside either body.
    Union(Box<SdfExpr<S, D>>, Box<SdfExpr<S, D>>),
    /// max of distances: inside both bodies.
    Intersect(Box<SdfExpr<S, D>>, Box<SdfExpr<S, D>>),
    /// sign flip: inside becomes outside. unbounded — no bounding ball.
    Complement(Box<SdfExpr<S, D>>),
    Translated { inner: Box<SdfExpr<S, D>>, offset: [S; D] },
    /// a rigid rotation of the inner shape about the body-local origin. `rot` is the body's
    /// orientation matrix R (row-major); a world point maps into the inner's frame as `R^T x`,
    /// so `dist(x) = inner.dist(R^T x)`. rotation is an isometry, so the signed distance is exact
    /// and the bounding-ball radius is unchanged (its center rotates by R).
    Rotated { inner: Box<SdfExpr<S, D>>, rot: [[S; D]; D] },
}

impl<S: Scalar, const D: usize> SdfExpr<S, D> {
    pub fn sphere(center: [S; D], radius: S) -> Self {
        SdfExpr::Sphere { center, radius }
    }

    pub fn cuboid(center: [S; D], half_extents: [S; D]) -> Self {
        SdfExpr::Cuboid { center, half_extents }
    }

    pub fn union(self, other: Self) -> Self {
        SdfExpr::Union(Box::new(self), Box::new(other))
    }

    pub fn intersect(self, other: Self) -> Self {
        SdfExpr::Intersect(Box::new(self), Box::new(other))
    }

    pub fn complement(self) -> Self {
        SdfExpr::Complement(Box::new(self))
    }

    pub fn translated(self, offset: [S; D]) -> Self {
        SdfExpr::Translated { inner: Box::new(self), offset }
    }

    /// rotate the shape by the orientation matrix `rot` (row-major) about the body-local origin.
    pub fn rotated(self, rot: [[S; D]; D]) -> Self {
        SdfExpr::Rotated { inner: Box::new(self), rot }
    }

    /// the signed distance at `x`: negative inside, positive outside, zero on
    /// the surface. exact for the primitives; a CSG min/max is exact outside
    /// the blend locus and conservative (never overstates the distance) at it.
    pub fn dist(&self, x: [S; D]) -> S {
        match self {
            SdfExpr::Sphere { center, radius } => {
                let mut sq = S::ZERO;
                for a in 0..D {
                    let d = x[a] - center[a];
                    sq = sq + d * d;
                }
                sq.sqrt() - *radius
            }
            SdfExpr::Cuboid { center, half_extents } => {
                // q_a = |x_a - c_a| - h_a: the per-axis excursion beyond the
                // face. outside = |max(q, 0)|; inside = min(max_a q_a, 0).
                let mut out_sq = S::ZERO;
                let mut q_max = (x[0] - center[0]).abs() - half_extents[0];
                for a in 0..D {
                    let q = (x[a] - center[a]).abs() - half_extents[a];
                    let q_pos = q.max(S::ZERO);
                    out_sq = out_sq + q_pos * q_pos;
                    q_max = q_max.max(q);
                }
                // strictly inside, out_sq is identically zero in a
                // neighbourhood — the true derivative of the sqrt term is 0,
                // but the dual chain rule at sqrt(0) evaluates 0 * inf = nan.
                // the subnormal floor keeps the tangent finite (and zero) at a
                // distance bias of sqrt(min_positive) ~ 1e-154.
                out_sq.max(S::from_f64(f64::MIN_POSITIVE)).sqrt() + q_max.min(S::ZERO)
            }
            SdfExpr::Union(a, b) => a.dist(x).min(b.dist(x)),
            SdfExpr::Intersect(a, b) => a.dist(x).max(b.dist(x)),
            SdfExpr::Complement(a) => S::ZERO - a.dist(x),
            SdfExpr::Translated { inner, offset } => {
                inner.dist(std::array::from_fn(|a| x[a] - offset[a]))
            }
            SdfExpr::Rotated { inner, rot } => {
                // map the world point into the inner frame: (R^T x)_i = sum_j rot[j][i] * x[j].
                let xr: [S; D] = std::array::from_fn(|i| {
                    let mut s = S::ZERO;
                    for j in 0..D {
                        s = s + rot[j][i] * x[j];
                    }
                    s
                });
                inner.dist(xr)
            }
        }
    }

    /// map every shape parameter through `f` — the carrier lift (f64 -> Dual
    /// constants for normals; f64 -> Gv params at trace time).
    pub fn lift<T: Scalar>(&self, f: &impl Fn(S) -> T) -> SdfExpr<T, D> {
        match self {
            SdfExpr::Sphere { center, radius } => SdfExpr::Sphere {
                center: center.map(|c| f(c)),
                radius: f(*radius),
            },
            SdfExpr::Cuboid { center, half_extents } => SdfExpr::Cuboid {
                center: center.map(|c| f(c)),
                half_extents: half_extents.map(|h| f(h)),
            },
            SdfExpr::Union(a, b) => SdfExpr::Union(Box::new(a.lift(f)), Box::new(b.lift(f))),
            SdfExpr::Intersect(a, b) => {
                SdfExpr::Intersect(Box::new(a.lift(f)), Box::new(b.lift(f)))
            }
            SdfExpr::Complement(a) => SdfExpr::Complement(Box::new(a.lift(f))),
            SdfExpr::Translated { inner, offset } => SdfExpr::Translated {
                inner: Box::new(inner.lift(f)),
                offset: offset.map(|o| f(o)),
            },
            SdfExpr::Rotated { inner, rot } => SdfExpr::Rotated {
                inner: Box::new(inner.lift(f)),
                rot: rot.map(|row| row.map(|e| f(e))),
            },
        }
    }

    /// the outward unit normal at `x`: the exact gradient of `dist` via the
    /// Dual carrier (one seeded evaluation per axis), renormalized — after
    /// CSG the gradient magnitude drifts from 1 but its direction is the
    /// normal wherever `dist` is differentiable. at a kink (equidistant CSG
    /// locus, cuboid edge) Dual takes one branch of the min/max — a valid
    /// one-sided normal.
    pub fn normal(&self, x: [S; D]) -> [S; D] {
        let lifted: SdfExpr<Dual<S>, D> = self.lift(&Dual::constant);
        let mut g = [S::ZERO; D];
        for a in 0..D {
            let xd: [Dual<S>; D] = std::array::from_fn(|i| {
                if i == a { Dual::variable(x[i]) } else { Dual::constant(x[i]) }
            });
            g[a] = lifted.dist(xd).tangent;
        }
        let mut sq = S::ZERO;
        for a in 0..D {
            sq = sq + g[a] * g[a];
        }
        // a degenerate gradient (the exact center) divides by the guard, not zero.
        let inv = S::ONE / sq.sqrt().max(S::from_f64(1e-300));
        std::array::from_fn(|a| g[a] * inv)
    }
}

impl SdfExpr<f64, 3> {
    /// parse a config-authored shape from its json wire form (python's `Shape.to_wire`
    /// serialized with `json.dumps`). the tree is CSG over 3-space primitives:
    ///   {"kind":"sphere","center":[x,y,z],"radius":r}
    ///   {"kind":"box","center":[x,y,z],"half_extents":[hx,hy,hz]}
    ///   {"kind":"union"|"intersect","a":<shape>,"b":<shape>}
    ///   {"kind":"complement","inner":<shape>}
    ///   {"kind":"translated","inner":<shape>,"offset":[x,y,z]}
    /// coordinates are the body-LOCAL frame; the penalization kernel translates the whole
    /// tree to the runtime body position. an unknown kind or a malformed field fails loud.
    pub fn from_json(s: &str) -> Result<Self, String> {
        let v: serde_json::Value = serde_json::from_str(s).map_err(|e| format!("shape json: {e}"))?;
        Self::from_value(&v)
    }

    /// serialize the CSG tree to its json wire form (the inverse of `from_json`): the string a
    /// checkpoint persists so a reader can reconstruct + draw the body silhouette.
    pub fn to_json(&self) -> String {
        self.to_value().to_string()
    }

    fn to_value(&self) -> serde_json::Value {
        use serde_json::json;
        match self {
            SdfExpr::Sphere { center, radius } => {
                json!({"kind": "sphere", "center": center, "radius": radius})
            }
            SdfExpr::Cuboid { center, half_extents } => {
                json!({"kind": "box", "center": center, "half_extents": half_extents})
            }
            SdfExpr::Union(a, b) => json!({"kind": "union", "a": a.to_value(), "b": b.to_value()}),
            SdfExpr::Intersect(a, b) => {
                json!({"kind": "intersect", "a": a.to_value(), "b": b.to_value()})
            }
            SdfExpr::Complement(inner) => json!({"kind": "complement", "inner": inner.to_value()}),
            SdfExpr::Translated { inner, offset } => {
                json!({"kind": "translated", "inner": inner.to_value(), "offset": offset})
            }
            SdfExpr::Rotated { inner, rot } => {
                json!({"kind": "rotated", "inner": inner.to_value(), "rot": rot})
            }
        }
    }

    fn from_value(v: &serde_json::Value) -> Result<Self, String> {
        let kind = v.get("kind").and_then(|k| k.as_str()).ok_or("shape: missing string 'kind'")?;
        let vec3 = |key: &str| -> Result<[f64; 3], String> {
            let arr = v
                .get(key)
                .and_then(|a| a.as_array())
                .ok_or_else(|| format!("shape '{kind}': missing array '{key}'"))?;
            if arr.len() != 3 {
                return Err(format!("shape '{kind}': '{key}' must have 3 components, got {}", arr.len()));
            }
            let mut out = [0.0; 3];
            for (i, e) in arr.iter().enumerate() {
                out[i] = e.as_f64().ok_or_else(|| format!("shape '{kind}': '{key}[{i}]' not a number"))?;
            }
            Ok(out)
        };
        let scalar = |key: &str| -> Result<f64, String> {
            v.get(key).and_then(|s| s.as_f64()).ok_or_else(|| format!("shape '{kind}': missing number '{key}'"))
        };
        let child = |key: &str| -> Result<Self, String> {
            Self::from_value(v.get(key).ok_or_else(|| format!("shape '{kind}': missing sub-shape '{key}'"))?)
        };
        let mat3 = |key: &str| -> Result<[[f64; 3]; 3], String> {
            let rows = v
                .get(key)
                .and_then(|a| a.as_array())
                .ok_or_else(|| format!("shape '{kind}': missing 3x3 matrix '{key}'"))?;
            if rows.len() != 3 {
                return Err(format!("shape '{kind}': '{key}' must have 3 rows, got {}", rows.len()));
            }
            let mut out = [[0.0; 3]; 3];
            for (i, row) in rows.iter().enumerate() {
                let cols = row.as_array().ok_or_else(|| format!("shape '{kind}': '{key}[{i}]' not a row"))?;
                if cols.len() != 3 {
                    return Err(format!("shape '{kind}': '{key}[{i}]' must have 3 columns, got {}", cols.len()));
                }
                for (j, e) in cols.iter().enumerate() {
                    out[i][j] = e.as_f64().ok_or_else(|| format!("shape '{kind}': '{key}[{i}][{j}]' not a number"))?;
                }
            }
            Ok(out)
        };
        match kind {
            "sphere" => Ok(SdfExpr::Sphere { center: vec3("center")?, radius: scalar("radius")? }),
            "box" => Ok(SdfExpr::Cuboid { center: vec3("center")?, half_extents: vec3("half_extents")? }),
            "union" => Ok(SdfExpr::Union(Box::new(child("a")?), Box::new(child("b")?))),
            "intersect" => Ok(SdfExpr::Intersect(Box::new(child("a")?), Box::new(child("b")?))),
            "complement" => Ok(SdfExpr::Complement(Box::new(child("inner")?))),
            "translated" => Ok(SdfExpr::Translated {
                inner: Box::new(child("inner")?),
                offset: vec3("offset")?,
            }),
            "rotated" => Ok(SdfExpr::Rotated {
                inner: Box::new(child("inner")?),
                rot: mat3("rot")?,
            }),
            other => Err(format!("shape: unknown kind '{other}' (sphere | box | union | intersect | complement | translated)")),
        }
    }
}

impl<const D: usize> SdfExpr<f64, D> {
    /// the enclosing ball of the body: every point with `dist(x) <= 0` lies
    /// within `radius` of `center`. `None` when the body is unbounded (any
    /// complement). the ball is the GEOMETRIC bound — a kernel's declared
    /// support pads it by the chi saturation width.
    pub fn bounding_ball(&self) -> Option<([f64; D], f64)> {
        match self {
            SdfExpr::Sphere { center, radius } => Some((*center, *radius)),
            SdfExpr::Cuboid { center, half_extents } => {
                let r = half_extents.iter().map(|h| h * h).sum::<f64>().sqrt();
                Some((*center, r))
            }
            SdfExpr::Union(a, b) => match (a.bounding_ball(), b.bounding_ball()) {
                (Some(ba), Some(bb)) => Some(enclosing_ball(ba, bb)),
                _ => None,
            },
            // the intersection lies inside EITHER operand, so either ball is
            // sound; take the smaller, and tolerate one unbounded side (an
            // annulus: sphere minus its core).
            SdfExpr::Intersect(a, b) => match (a.bounding_ball(), b.bounding_ball()) {
                (Some(ba), Some(bb)) => Some(if ba.1 <= bb.1 { ba } else { bb }),
                (Some(ba), None) => Some(ba),
                (None, Some(bb)) => Some(bb),
                (None, None) => None,
            },
            SdfExpr::Complement(_) => None,
            SdfExpr::Translated { inner, offset } => {
                let (c, r) = inner.bounding_ball()?;
                Some((std::array::from_fn(|a| c[a] + offset[a]), r))
            }
            SdfExpr::Rotated { inner, rot } => {
                // rotation is an isometry: the radius is unchanged, the center rotates by R.
                let (c, r) = inner.bounding_ball()?;
                let rc = std::array::from_fn(|i| (0..D).map(|j| rot[i][j] * c[j]).sum());
                Some((rc, r))
            }
        }
    }
}

/// the smallest ball enclosing two balls: one contains the other, or the new
/// diameter spans the far sides of both.
fn enclosing_ball<const D: usize>(
    (ca, ra): ([f64; D], f64),
    (cb, rb): ([f64; D], f64),
) -> ([f64; D], f64) {
    let d = (0..D).map(|a| (cb[a] - ca[a]).powi(2)).sum::<f64>().sqrt();
    if d + rb <= ra {
        return (ca, ra);
    }
    if d + ra <= rb {
        return (cb, rb);
    }
    let r = 0.5 * (d + ra + rb);
    let t = (r - ra) / d;
    (std::array::from_fn(|a| ca[a] + t * (cb[a] - ca[a])), r)
}

#[cfg(test)]
mod tests {
    use super::*;

    type Sdf3 = SdfExpr<f64, 3>;

    fn approx(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol * a.abs().max(b.abs()).max(1.0)
    }

    #[test]
    fn sphere_distance_is_analytic() {
        let s = Sdf3::sphere([1.0, 2.0, 3.0], 0.5);
        assert_eq!(s.dist([1.0, 2.0, 3.0]), -0.5); // center
        assert_eq!(s.dist([1.5, 2.0, 3.0]), 0.0); // surface
        assert!(approx(s.dist([1.0, 2.0, 5.0]), 1.5, 1e-15)); // outside
    }

    #[test]
    fn cuboid_distance_is_analytic_inside_face_edge_and_corner() {
        let c = Sdf3::cuboid([0.0; 3], [1.0, 2.0, 3.0]);
        // inside: minus the distance to the nearest face.
        assert!(approx(c.dist([0.5, 0.0, 0.0]), -0.5, 1e-15));
        // outside a face: the perpendicular excess.
        assert!(approx(c.dist([2.0, 0.0, 0.0]), 1.0, 1e-15));
        // outside an edge: the 2d hypotenuse of the excesses.
        assert!(approx(c.dist([2.0, 3.0, 0.0]), (1.0f64 + 1.0).sqrt(), 1e-15));
        // outside a corner: the 3d hypotenuse.
        assert!(approx(c.dist([2.0, 3.0, 4.0]), 3.0f64.sqrt(), 1e-15));
    }

    #[test]
    fn dual_normals_match_the_analytic_normals() {
        let s = Sdf3::sphere([1.0, 0.0, 0.0], 0.5);
        let x = [1.0 + 0.3, 0.4, 0.0];
        let n = s.normal(x);
        let r = (0.3f64 * 0.3 + 0.4 * 0.4).sqrt();
        assert!(approx(n[0], 0.3 / r, 1e-12));
        assert!(approx(n[1], 0.4 / r, 1e-12));
        assert!(approx(n[2], 0.0, 1e-12));

        // a cuboid face point: the normal is the face axis, exactly.
        let c = Sdf3::cuboid([0.0; 3], [1.0; 3]);
        let n = c.normal([2.0, 0.2, -0.3]);
        assert!(approx(n[0], 1.0, 1e-12));
        assert!(approx(n[1], 0.0, 1e-12));
        assert!(approx(n[2], 0.0, 1e-12));
        // an INSIDE point near the +x face still points outward along +x.
        let n = c.normal([0.9, 0.1, 0.0]);
        assert!(approx(n[0], 1.0, 1e-12));
    }

    #[test]
    fn csg_identities_hold_bit_exactly() {
        let a = Sdf3::sphere([0.0; 3], 1.0);
        let b = Sdf3::cuboid([1.5, 0.0, 0.0], [0.5; 3]);
        let pts = [[0.0, 0.0, 0.0], [1.2, 0.3, -0.4], [3.0, 3.0, 3.0], [-0.9, 0.0, 0.1]];
        for x in pts {
            // union commutes.
            let u1 = a.clone().union(b.clone()).dist(x);
            let u2 = b.clone().union(a.clone()).dist(x);
            assert_eq!(u1.to_bits(), u2.to_bits());
            // complement . complement = identity.
            let cc = a.clone().complement().complement().dist(x);
            assert_eq!(cc.to_bits(), a.dist(x).to_bits());
            // intersect(a, a) = a.
            let ii = a.clone().intersect(a.clone()).dist(x);
            assert_eq!(ii.to_bits(), a.dist(x).to_bits());
            // translate . untranslate = identity.
            let tt = a.clone().translated([0.7, -0.2, 0.4]).translated([-0.7, 0.2, -0.4]).dist(x);
            assert_eq!(tt.to_bits(), a.dist(x).to_bits());
        }
    }

    #[test]
    fn union_normal_is_the_owning_primitives_normal() {
        // two overlapping spheres: at a surface point of A far from the blend
        // locus, the union's (renormalized) normal is exactly A's.
        let a = Sdf3::sphere([0.0; 3], 1.0);
        let b = Sdf3::sphere([1.5, 0.0, 0.0], 1.0);
        let u = a.clone().union(b);
        let x = [-1.0, 0.0, 0.0]; // A's surface, opposite side from B
        let (nu, na) = (u.normal(x), a.normal(x));
        for ax in 0..3 {
            assert!(approx(nu[ax], na[ax], 1e-12));
        }
    }

    #[test]
    fn a_wrong_half_extent_fails_the_distance_law() {
        // bug injection: the analytic-comparison law has teeth — perturbing
        // one half-extent breaks the corner distance.
        let good = Sdf3::cuboid([0.0; 3], [1.0, 2.0, 3.0]);
        let bad = Sdf3::cuboid([0.0; 3], [1.0, 2.0 + 1e-3, 3.0]);
        let corner = [2.0, 3.0, 4.0];
        assert!(approx(good.dist(corner), 3.0f64.sqrt(), 1e-12));
        assert!(!approx(bad.dist(corner), 3.0f64.sqrt(), 1e-12));
    }

    #[test]
    fn bounding_ball_contains_every_interior_point() {
        // randomized containment: any sampled point with dist <= 0 lies
        // within the ball, for a compound translated CSG body.
        let body = Sdf3::sphere([0.4, 0.0, 0.0], 0.8)
            .union(Sdf3::cuboid([-1.0, 0.5, 0.0], [0.3, 0.6, 0.2]))
            .translated([0.1, -0.2, 0.3]);
        let (c, r) = body.bounding_ball().unwrap();
        let mut state = 0x9e3779b97f4a7c15u64;
        let mut rand = || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (state >> 11) as f64 / (1u64 << 53) as f64 * 6.0 - 3.0
        };
        let mut inside = 0usize;
        for _ in 0..20000 {
            let x = [rand(), rand(), rand()];
            if body.dist(x) <= 0.0 {
                inside += 1;
                let d = (0..3).map(|a| (x[a] - c[a]).powi(2)).sum::<f64>().sqrt();
                assert!(d <= r + 1e-12, "interior point {x:?} outside the ball");
            }
        }
        assert!(inside > 100, "sample never landed inside — vacuous containment");

        // complement is unbounded; an annulus (intersect with a complement)
        // inherits the bounded side's ball.
        assert!(Sdf3::sphere([0.0; 3], 1.0).complement().bounding_ball().is_none());
        let annulus = Sdf3::sphere([0.0; 3], 1.0)
            .intersect(Sdf3::sphere([0.0; 3], 0.5).complement());
        let (_, r) = annulus.bounding_ball().unwrap();
        assert_eq!(r, 1.0);
    }

    #[test]
    fn the_sdf_traces_at_gv() {
        // the trace-native claim: the SAME expression that
        // ran at f64 above evaluates at the Gv carrier inside a trace — shape
        // parameters as scalar params, the distance as a graph node. the
        // penalization kernel builder consumes exactly this path.
        use symbi_algebra::algebra::Numeric as _;
        use symbi_ir::{begin_trace, end_trace, Gv};
        begin_trace();
        let body: SdfExpr<Gv, 3> = SdfExpr::Sphere {
            center: [Gv::scalar("body_0_pos_0"), Gv::scalar("body_0_pos_1"), Gv::scalar("body_0_pos_2")],
            radius: Gv::scalar("body_0_radius"),
        }
        .union(SdfExpr::Cuboid {
            center: [Gv::from_f64(1.0), Gv::from_f64(0.0), Gv::from_f64(0.0)],
            half_extents: [Gv::from_f64(0.5); 3],
        });
        let x = [Gv::scalar("x0"), Gv::scalar("x1"), Gv::scalar("x2")];
        let d = body.dist(x);
        let kernel = end_trace();
        assert!(!kernel.graph.has_errors(), "sdf trace produced graph errors");
        assert!(
            kernel.scalar_params.iter().any(|p| p == "body_0_radius"),
            "shape params must land in the scalar manifest",
        );
        let _ = d.node();
    }

    #[test]
    fn enclosing_ball_handles_containment_and_overlap() {
        // disjoint balls: the enclosing radius spans the far sides.
        let (c, r) = enclosing_ball(([0.0, 0.0], 1.0), ([4.0, 0.0], 1.0));
        assert!(approx(r, 3.0, 1e-15));
        assert!(approx(c[0], 2.0, 1e-15));
        // one contains the other: the big ball wins, exactly.
        let (c, r) = enclosing_ball(([0.0, 0.0], 2.0), ([0.5, 0.0], 0.1));
        assert_eq!(r, 2.0);
        assert_eq!(c, [0.0, 0.0]);
    }

    #[test]
    fn from_json_parses_csg_and_equals_native() {
        // the EXACT wire python's Shape.sphere(...).union(Shape.box(...)) emits.
        let wire = r#"{"kind":"union",
            "a":{"kind":"sphere","center":[0.0,0.0,0.0],"radius":1.0},
            "b":{"kind":"box","center":[2.0,0.0,0.0],"half_extents":[0.5,0.5,0.5]}}"#;
        let s = SdfExpr::<f64, 3>::from_json(wire).expect("parse csg wire");
        let native = SdfExpr::<f64, 3>::sphere([0.0; 3], 1.0)
            .union(SdfExpr::cuboid([2.0, 0.0, 0.0], [0.5; 3]));
        assert_eq!(s, native, "wire must reconstruct the native tree");
        // inside the sphere and inside the box are both interior (dist < 0); the gap between is out.
        assert!(s.dist([0.0, 0.0, 0.0]) < 0.0);
        assert!(s.dist([2.0, 0.0, 0.0]) < 0.0);
        assert!(s.dist([1.3, 0.0, 0.0]) > 0.0);
        // the bounding ball encloses every interior point.
        let (c, r) = s.bounding_ball().expect("csg is bounded");
        for x in [[0.0, 0.0, 0.0], [2.4, 0.0, 0.0]] {
            let d: f64 = (0..3).map(|a| (x[a] - c[a]).powi(2)).sum::<f64>().sqrt();
            assert!(d <= r + 1e-12, "interior point {x:?} escapes the bounding ball");
        }
    }

    #[test]
    fn rotated_shape_is_an_isometry() {
        // 90 deg about z: R = [[0,-1,0],[1,0,0],[0,0,1]] (row-major).
        let r = [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]];
        let cube = SdfExpr::<f64, 3>::cuboid([0.0, 0.0, 0.0], [0.5, 0.2, 0.3]);
        let rot = cube.clone().rotated(r);
        // rotating the query point by R leaves the distance unchanged: rot.dist(R p) == box.dist(p).
        for p in [[0.3, 0.0, 0.0], [0.0, 0.4, 0.1], [0.6, 0.6, 0.0]] {
            let rp = [-p[1], p[0], p[2]]; // R p for 90 deg about z
            assert!(
                (rot.dist(rp) - cube.dist(p)).abs() < 1e-12,
                "rotation is not an isometry: {} vs {}",
                rot.dist(rp),
                cube.dist(p),
            );
        }
        // a centered shape: the bounding ball is unchanged (radius fixed, center at the origin).
        let (c, rr) = rot.bounding_ball().unwrap();
        let (_, r0) = cube.bounding_ball().unwrap();
        assert!((rr - r0).abs() < 1e-12);
        assert!(c.iter().all(|&x| x.abs() < 1e-12));
        // the 0.5-extent rotates onto +y; just outside that face the normal is +y, unit length.
        let n = rot.normal([0.0, 0.55, 0.0]);
        let mag = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
        assert!((mag - 1.0).abs() < 1e-9, "normal not unit: {mag}");
        assert!(n[1] > 0.9, "outward normal of the rotated +y face should be +y: {n:?}");
    }

    #[test]
    fn from_json_parses_a_rotated_shape() {
        let wire = r#"{"kind":"rotated",
            "inner":{"kind":"box","center":[0.0,0.0,0.0],"half_extents":[0.5,0.2,0.3]},
            "rot":[[0.0,-1.0,0.0],[1.0,0.0,0.0],[0.0,0.0,1.0]]}"#;
        let s = SdfExpr::<f64, 3>::from_json(wire).expect("parse rotated");
        let native = SdfExpr::<f64, 3>::cuboid([0.0, 0.0, 0.0], [0.5, 0.2, 0.3])
            .rotated([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]);
        assert_eq!(s, native);
    }

    #[test]
    fn to_json_round_trips_through_from_json() {
        // the persisted shape wire reconstructs the exact tree (checkpoint -> viz).
        let s = SdfExpr::<f64, 3>::cuboid([0.1, 0.2, 0.3], [0.5, 0.4, 0.3])
            .union(SdfExpr::sphere([1.0, 0.0, 0.0], 0.6))
            .rotated([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
            .translated([2.0, 0.0, 0.0]);
        assert_eq!(SdfExpr::<f64, 3>::from_json(&s.to_json()).expect("round trip"), s);
    }

    #[test]
    fn from_json_rejects_malformed_shapes() {
        // unknown primitive.
        assert!(SdfExpr::<f64, 3>::from_json(r#"{"kind":"torus","center":[0,0,0]}"#).is_err());
        // wrong arity on a vector.
        assert!(SdfExpr::<f64, 3>::from_json(r#"{"kind":"sphere","center":[0,0],"radius":1.0}"#).is_err());
        // missing scalar field.
        assert!(SdfExpr::<f64, 3>::from_json(r#"{"kind":"sphere","center":[0,0,0]}"#).is_err());
        // missing sub-shape.
        assert!(SdfExpr::<f64, 3>::from_json(r#"{"kind":"complement"}"#).is_err());
    }
}
