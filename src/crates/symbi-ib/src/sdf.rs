// =============================================================================
// sdf.rs
//
// signed-distance geometry as carrier-generic CSG (docs/design/50 layer 1):
// negative inside the body, positive outside. every operation is min/max/
// affine arithmetic plus one sqrt, so the SAME expression evaluates at f64
// (host oracle + tests), at Gv (the traced penalization kernel), and at
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
        // the trace-native claim (docs/design/50): the SAME expression that
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
}
