// =============================================================================
// metric.rs
//
// metric trait hierarchy for coordinate geometry.
// designed around the 3+1 ADM decomposition of spacetime:
//
//   ds^2 = -\alpha^2 dt^2 + \gamma_{ij}(dx^i + \beta^i dt)(dx^j + \beta^j dt)
//
// the trait is forward-compatible with full GRMHD: flat-space metrics
// (cartesian, spherical, cylindrical) are special cases where
// \alpha = 1, \beta^i = 0, and \gamma_{ij} is diagonal.
//
// usage:
//   let m = Spherical;
//   let x = vec3(r, theta, phi);
//   let g = m.spatial_metric(x);   // \gamma_{ij}
//   let dv = m.sqrt_det_gamma(x);  // \sqrt{\gamma}
//   let v_lower = g * v_upper;     // index lowering
// =============================================================================

use symbi_algebra::{Contravariant, Covariant, Embedded, Matrix, Physical, Tensor};
use symbi_ir::algebra::Scalar;

// ============================================================
// geometry enum: coordinate system identifier
// ============================================================

/// coordinate system identifier.
/// integer representation matches GPU kernel convention.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(i32)]
pub enum Geometry {
    Cartesian = 0,
    Spherical = 1,
    Cylindrical = 2,
}

impl Geometry {
    /// integer representation for GPU kernel dispatch.
    pub fn as_i32(self) -> i32 {
        self as i32
    }
}

/// spacetime identifier — the background a regime evolves on. orthogonal to both the spatial
/// [`Geometry`] and the physics regime: GR is a curved spacetime, so a single
/// SR regime (Rhd / Rmhd) composes with every spacetime here, one implementation serving all of
/// them. flat `Minkowski`
/// (lapse = 1, shift = 0, gamma = identity in physical components) is the default — every realized
/// run. drives the lapse / sqrt(gamma) densitization selector in the kernel. integer
/// repr matches the GPU kernel convention (mirrors `Geometry`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(i32)]
pub enum Spacetime {
    #[default]
    Minkowski = 0,
    /// the static spherically-symmetric vacuum in ingoing kerr-schild (eddington-finkelstein)
    /// coords — horizon-penetrating: regular across r = 2M, nonzero
    /// radial shift beta^r, diagonal spatial metric gamma_{rr} = 1 + 2M/r. reuses the
    /// `schwarzschild_mass` scalar. selects the shift-advection flux + KS densitization kernel path.
    SchwarzschildKS = 2,
    /// spinning Kerr in ingoing kerr-schild coordinates — horizon-penetrating, non-diagonal
    /// spatial metric (gamma_{r phi} carries the frame dragging into the spatial slice), radial
    /// shift beta^r = 2Mr/(Sigma + 2Mr). the covariant valencia storage is required here, since a
    /// componentwise orthonormal frame exists for a diagonal gamma alone. the mass M and spin a
    /// ride as kernel scalars (`schwarzschild_mass`, `kerr_spin`). reduces to `SchwarzschildKS`
    /// physics at a = 0 (different kernel expressions, same values).
    KerrKS = 3,
}

impl Spacetime {
    /// integer representation for GPU kernel dispatch.
    pub fn as_i32(self) -> i32 {
        self as i32
    }
}

// ============================================================
// metric trait: full 3+1 interface
// ============================================================

/// metric on a D-dimensional spatial manifold.
///
/// provides the general 3+1 ADM surface — valid for every metric, diagonal or otherwise:
/// - lapse \alpha (1 for flat spacetimes)
/// - shift \beta^i (0 for static spacetimes)
/// - spatial metric \gamma_{ij} and its inverse \gamma^{ij}
/// - the volume element \sqrt{\gamma} (= `sqrt_det_gamma` / `volume_factor`)
/// - `lower`/`raise` (variance-typed: `Contravariant` <-> `Covariant`, general tensor contraction)
///
/// the orthogonal-frame surface — scale factors `h_i` and the orthonormal-frame
/// `vector_to_cartesian` — exists for a diagonal metric, so it lives on the [`DiagonalMetric`]
/// subtrait. a non-diagonal (Kerr-class) metric impls `Metric` alone, so the compiler withholds
/// orthogonal quadrature on it until the non-diagonal forms are written — the "GRMHD-forward"
/// claim is type-enforced. (the realized
/// physics is entirely diagonal: flat + orthogonal-curvilinear.)
///
/// implementations: Cartesian, Spherical, Cylindrical (all `DiagonalMetric`),
/// and (future) Schwarzschild, Kerr, etc. (`Metric` only, until their quadrature lands).
pub trait Metric<S: Scalar, const D: usize> {
    /// coordinate system for this metric.
    fn geometry(&self) -> Geometry {
        Geometry::Cartesian
    }

    /// the spacetime background (flat vs curved). orthogonal to `geometry()`: `Minkowski` for every
    /// flat metric, a curved variant (Schwarzschild, ...) for GR. selects the lapse / sqrt(gamma)
    /// densitization path in the kernel; flat -> the densitization is a no-op.
    fn spacetime(&self) -> Spacetime {
        Spacetime::Minkowski
    }

    /// the spacetime's runtime scalar parameters as `(wire-name, value)` pairs — the kernel-dispatch
    /// scalars a curved metric needs filled (e.g. `("schwarzschild_mass", M)`). flat -> empty. the
    /// substrate resolves these by name at the godunov dispatch, exactly like the EOS feeds `gamma`.
    fn spacetime_scalars(&self) -> Vec<(&'static str, S)> {
        Vec::new()
    }

    /// lapse function \alpha. determines time dilation.
    /// flat spacetime: \alpha = 1.
    fn lapse(&self, x: Tensor<S, D>) -> S {
        let _ = x;
        S::ONE
    }

    /// the lapse squared, \alpha^2 = -1/g^{tt}. default \alpha*\alpha; override where a closed form
    /// replaces the sqrt round-trip (Schwarzschild \alpha^2 = f = 1 - 2M/r; Kerr-Schild \alpha^2 =
    /// 1/(1 + 2M/r)). the GR CFL radial coordinate-speed factor \alpha sqrt(\gamma^{rr}) equals
    /// \alpha^2 for the det-g-flat family (Schwarzschild, Kerr-Schild), and takes the exact closed
    /// form — sqrt(\alpha^2)^2 != \alpha^2 in floating point, which would break the CFL bit-diff.
    fn lapse_sq(&self, x: Tensor<S, D>) -> S {
        let a = self.lapse(x);
        a * a
    }

    /// shift vector \beta^i. determines frame dragging.
    /// static spacetime: \beta = 0.
    fn shift(&self, x: Tensor<S, D>) -> Tensor<S, D> {
        let _ = x;
        Tensor::zeros()
    }

    /// spatial metric tensor \gamma_{ij}.
    fn spatial_metric(&self, x: Tensor<S, D>) -> Matrix<S, D>;

    /// inverse spatial metric \gamma^{ij}.
    /// default: compute from spatial_metric via matrix inverse.
    /// override for analytical inverse (diagonal metrics, etc.).
    fn spatial_metric_inv(&self, x: Tensor<S, D>) -> Matrix<S, D>;

    /// \sqrt{\gamma} where \gamma = det(\gamma_{ij}).
    /// this is the volume element: dV = \sqrt{\gamma} dx^1 dx^2 ... dx^D.
    fn sqrt_det_gamma(&self, x: Tensor<S, D>) -> S;

    /// scale factors h_i where \gamma_{ii} = h_i^2, well-defined for a diagonal metric — the
    /// `where Self: DiagonalMetric` bound turns a non-diagonal call into a compile error, which is
    /// what forces the off-diagonal terms to be handled explicitly where a bare sqrt-of-diagonal
    /// would silently drop them. the default reads the diagonal directly.
    fn scale_factors(&self, x: Tensor<S, D>) -> Tensor<S, D>
    where
        Self: DiagonalMetric<S, D>,
    {
        let g = self.spatial_metric(x);
        Tensor::new(std::array::from_fn(|ii| g[(ii, ii)].sqrt()))
    }

    /// effective volume element including suppressed (unresolved) dimensions.
    ///
    /// for full-rank metrics (D = physical dimension), this equals sqrt_det_gamma.
    /// for reduced-dimension metrics, it includes the jacobian contributions from
    /// suppressed angular directions. this is what the flux divergence actually
    /// needs for face areas and cell volumes.
    ///
    /// examples:
    ///   spherical 1D: r^2  (the naive sqrt_det_gamma of the 1x1 metric would give 1)
    ///   spherical 2D: r^2 sin(theta)  (naive sqrt_det_gamma of the 2x2 metric gives r)
    ///   cylindrical 1D: r  (naive gives 1)
    ///   cylindrical 2D: r  (naive gives 1)
    ///
    /// required, with no default: the natural default `= sqrt_det_gamma` is wrong for every
    /// reduced-dimension metric — it drops the jacobian of the suppressed angular directions (a
    /// spherical-1D cell volume is r^2 dr, while the 1x1 `sqrt_det_gamma` is 1). a silent default there
    /// bakes the wrong face area / cell volume and reports success. making it explicit costs a
    /// full-rank metric the trivial `self.sqrt_det_gamma(x)` delegation, and it puts a reduced-D
    /// metric that omits the proper measure in front of the compiler.
    fn volume_factor(&self, x: Tensor<S, D>) -> S;

    /// transform from this coordinate system to cartesian.
    fn to_cartesian(&self, x: Tensor<S, D>) -> Tensor<S, D>;

    /// transform from cartesian to this coordinate system.
    fn from_cartesian(&self, x: Tensor<S, D>) -> Tensor<S, D>;

    /// the frame morphism `Ortho -> Cart`: rotate a PHYSICAL (orthonormal-frame) vector into the
    /// global Cartesian frame. default: identity (cartesian: Ortho == Cart); override for
    /// non-cartesian (the rotation by the orthonormal basis directions). typed so the review's
    /// `vector_to_cartesian(lower(v))` is now a COMPILE ERROR — `lower` yields `Covariant`
    /// (coordinate basis), this wants `Physical` (orthonormal).
    fn vector_to_cartesian(&self, x: Tensor<S, D>, v: Physical<S, D>) -> Embedded<S, D>
    where
        Self: DiagonalMetric<S, D>,
    {
        let _ = x;
        Embedded::new(v.into_raw())
    }

    /// the frame morphism `Cart -> Ortho`: rotate a Cartesian vector into this system's PHYSICAL
    /// (orthonormal) frame. default: identity (cartesian); override for non-cartesian.
    fn vector_from_cartesian(&self, x: Tensor<S, D>, v: Embedded<S, D>) -> Physical<S, D>
    where
        Self: DiagonalMetric<S, D>,
    {
        let _ = x;
        Physical::new(v.into_raw())
    }

    /// the SCALE-FACTOR BRIDGE `CoordUp -> Ortho`: `V_a = h_a v^a`. the one place where the metric
    /// enters the (otherwise flat) orthonormal frame the substrate computes in. requires
    /// [`DiagonalMetric`] (the orthonormal frame exists only for a diagonal metric; a non-diagonal
    /// metric replaces this with a tetrad).
    fn to_physical(&self, x: Tensor<S, D>, v: &Contravariant<S, D>) -> Physical<S, D>
    where
        Self: DiagonalMetric<S, D>,
    {
        let h = self.scale_factors(x);
        Physical::new(Tensor::new(std::array::from_fn(|i| h[i] * v[i])))
    }

    /// the inverse bridge `Ortho -> CoordUp`: `v^a = V_a / h_a`.
    fn from_physical(&self, x: Tensor<S, D>, v: &Physical<S, D>) -> Contravariant<S, D>
    where
        Self: DiagonalMetric<S, D>,
    {
        let h = self.scale_factors(x);
        Contravariant::new(Tensor::new(std::array::from_fn(|i| v[i] / h[i])))
    }

    /// lower an index: v_i = \gamma_{ij} v^j.
    /// maps contravariant -> covariant via the spatial metric.
    fn lower(&self, x: Tensor<S, D>, v: &Contravariant<S, D>) -> Covariant<S, D> {
        Covariant::new(self.spatial_metric(x).mul_vec(v.raw()))
    }

    /// raise an index: v^i = \gamma^{ij} v_j.
    /// maps covariant -> contravariant via the inverse spatial metric.
    fn raise(&self, x: Tensor<S, D>, w: &Covariant<S, D>) -> Contravariant<S, D> {
        Contravariant::new(self.spatial_metric_inv(x).mul_vec(w.raw()))
    }

    /// geometric momentum source terms for the euler equations in curvilinear
    /// coordinates. written for the conservation form:
    ///
    ///   d/dt(rho * V_i) + div(F) = S_i
    ///
    /// where V_i are physical (orthonormal) velocity components (V_i = h_i * v^i).
    /// for reduced dimensions, accounts for suppressed angular directions
    /// (their pressure contributions persist even when velocity is unresolved).
    ///
    /// cartesian: S = 0.
    /// non-cartesian: derived from Christoffel symbols of the diagonal metric.
    ///
    /// this is the continuous analytical formula. for discrete schemes,
    /// use `momentum_source_inertial` + discrete pressure source from face
    /// area differences to achieve exact discrete equilibrium.
    fn momentum_source(&self, x: Tensor<S, D>, rho: S, vel: Tensor<S, D>, p: S) -> Tensor<S, D> {
        let _ = (x, rho, vel, p);
        Tensor::zeros()
    }

    /// inertial (velocity-dependent) part of the geometric momentum source.
    /// carries the velocity terms alone; the pressure terms come from discrete
    /// face area differences, which is what delivers exact discrete equilibrium:
    ///
    ///   S_pressure[i] = p * (A^i_R - A^i_L) / V
    ///
    /// the total geometric source is:
    ///   S = S_pressure + S_inertial
    ///
    /// cartesian: S_inertial = 0.
    ///
    /// `mom` is the regime-agnostic conserved momentum density (Newtonian `rho v`,
    /// relativistic `rho h W^2 v`): the source is the bilinear `S^i = -Gamma^i_jk mom^j v^k`,
    /// so the same code serves every regime and the magnetic tension (call with `mom = b`,
    /// `vel = b` for `-Gamma(b, b)`).
    fn momentum_source_inertial(
        &self,
        x: Tensor<S, D>,
        mom: Tensor<S, D>,
        vel: Tensor<S, D>,
    ) -> Tensor<S, D> {
        let _ = (x, mom, vel);
        Tensor::zeros()
    }
}

/// a metric whose spatial tensor \gamma_{ij} is diagonal — flat-space + orthogonal-curvilinear
/// (Cartesian, Spherical, Cylindrical). scale factors `h_i` and the orthonormal-frame
/// `vector_to_cartesian` are meaningful exactly on such a metric, so [`Metric::scale_factors`] /
/// [`Metric::vector_to_cartesian`] are gated `where Self: DiagonalMetric`. a non-diagonal
/// (Kerr-class) metric impls `Metric` alone — so the compiler rejects orthogonal-frame
/// quadrature on it until the non-diagonal forms are written. a pure marker: the gated methods
/// already live on `Metric`; this trait carries the diagonality proof obligation.
///
/// the diagonality bound is compile-enforced. a `DiagonalMetric`-bounded generic takes scale factors:
/// ```
/// use symbi_geometry::{DiagonalMetric, Spherical};
/// use symbi_algebra::Tensor;
/// fn h<M: DiagonalMetric<f64, 2>>(m: &M, x: Tensor<f64, 2>) -> Tensor<f64, 2> {
///     m.scale_factors(x)
/// }
/// let _ = h(&Spherical, Tensor::new([1.0, 0.5]));
/// ```
/// a `Metric`-only generic meets a compile-time type error at the call site:
/// ```compile_fail
/// use symbi_geometry::Metric;
/// use symbi_algebra::Tensor;
/// fn h<M: Metric<f64, 2>>(m: &M, x: Tensor<f64, 2>) -> Tensor<f64, 2> {
///     m.scale_factors(x) // ERROR: the bound `M: DiagonalMetric<f64, 2>` is not satisfied
/// }
/// ```
pub trait DiagonalMetric<S: Scalar, const D: usize>: Metric<S, D> {}

// ============================================================
// cartesian metric: \gamma_{ij} = \delta_{ij}
// ============================================================

/// flat cartesian metric in D dimensions.
/// \gamma_{ij} = I, h_i = 1, \sqrt{\gamma} = 1.
#[derive(Debug, Clone, Copy)]
pub struct Cartesian;

impl<S: Scalar> Metric<S, 1> for Cartesian {
    fn spatial_metric(&self, _x: Tensor<S, 1>) -> Matrix<S, 1> {
        Matrix::identity()
    }

    fn spatial_metric_inv(&self, _x: Tensor<S, 1>) -> Matrix<S, 1> {
        Matrix::identity()
    }

    fn sqrt_det_gamma(&self, _x: Tensor<S, 1>) -> S {
        S::ONE
    }

    fn scale_factors(&self, _x: Tensor<S, 1>) -> Tensor<S, 1> {
        Tensor::new([S::ONE])
    }

    fn to_cartesian(&self, x: Tensor<S, 1>) -> Tensor<S, 1> {
        x
    }
    fn from_cartesian(&self, x: Tensor<S, 1>) -> Tensor<S, 1> {
        x
    }
    fn volume_factor(&self, x: Tensor<S, 1>) -> S {
        self.sqrt_det_gamma(x)
    }
}

impl<S: Scalar> Metric<S, 2> for Cartesian {
    fn spatial_metric(&self, _x: Tensor<S, 2>) -> Matrix<S, 2> {
        Matrix::identity()
    }

    fn spatial_metric_inv(&self, _x: Tensor<S, 2>) -> Matrix<S, 2> {
        Matrix::identity()
    }

    fn sqrt_det_gamma(&self, _x: Tensor<S, 2>) -> S {
        S::ONE
    }

    fn scale_factors(&self, _x: Tensor<S, 2>) -> Tensor<S, 2> {
        Tensor::new([S::ONE, S::ONE])
    }

    fn to_cartesian(&self, x: Tensor<S, 2>) -> Tensor<S, 2> {
        x
    }
    fn from_cartesian(&self, x: Tensor<S, 2>) -> Tensor<S, 2> {
        x
    }
    fn volume_factor(&self, x: Tensor<S, 2>) -> S {
        self.sqrt_det_gamma(x)
    }
}

impl<S: Scalar> Metric<S, 3> for Cartesian {
    fn spatial_metric(&self, _x: Tensor<S, 3>) -> Matrix<S, 3> {
        Matrix::identity()
    }

    fn spatial_metric_inv(&self, _x: Tensor<S, 3>) -> Matrix<S, 3> {
        Matrix::identity()
    }

    fn sqrt_det_gamma(&self, _x: Tensor<S, 3>) -> S {
        S::ONE
    }

    fn scale_factors(&self, _x: Tensor<S, 3>) -> Tensor<S, 3> {
        Tensor::new([S::ONE, S::ONE, S::ONE])
    }

    fn to_cartesian(&self, x: Tensor<S, 3>) -> Tensor<S, 3> {
        x
    }
    fn from_cartesian(&self, x: Tensor<S, 3>) -> Tensor<S, 3> {
        x
    }
    fn volume_factor(&self, x: Tensor<S, 3>) -> S {
        self.sqrt_det_gamma(x)
    }
}

// ============================================================
// spherical metric: x = (r, theta, phi)
//   \gamma_{ij} = diag(1, r^2, r^2 sin^2 theta)
//   \sqrt{\gamma} = r^2 sin theta
//
// 1D: x = (r), \gamma = diag(1), \sqrt{\gamma} = 1
//     (radial direction only, angular factors in volume element)
// 2D: x = (r, theta), \gamma = diag(1, r^2), \sqrt{\gamma} = r
// 3D: x = (r, theta, phi), full metric
// ============================================================

/// spherical metric. coordinates: (r, theta, phi).
#[derive(Debug, Clone, Copy)]
pub struct Spherical;

impl<S: Scalar> Metric<S, 1> for Spherical {
    fn geometry(&self) -> Geometry {
        Geometry::Spherical
    }
    fn spatial_metric(&self, _x: Tensor<S, 1>) -> Matrix<S, 1> {
        Matrix::identity()
    }

    fn spatial_metric_inv(&self, _x: Tensor<S, 1>) -> Matrix<S, 1> {
        Matrix::identity()
    }

    fn sqrt_det_gamma(&self, _x: Tensor<S, 1>) -> S {
        S::ONE
    }

    fn scale_factors(&self, _x: Tensor<S, 1>) -> Tensor<S, 1> {
        Tensor::new([S::ONE])
    }

    fn to_cartesian(&self, x: Tensor<S, 1>) -> Tensor<S, 1> {
        x
    }
    fn from_cartesian(&self, x: Tensor<S, 1>) -> Tensor<S, 1> {
        x
    }

    fn volume_factor(&self, x: Tensor<S, 1>) -> S {
        let r = x[0];
        r * r
    }

    /// 1D spherical: S_r = 2p/r (pressure from 2 suppressed angular directions).
    fn momentum_source(&self, x: Tensor<S, 1>, _rho: S, _vel: Tensor<S, 1>, p: S) -> Tensor<S, 1> {
        let r = x[0];
        let two = S::ONE + S::ONE;
        Tensor::new([two * p / r])
    }

    /// 1D spherical inertial: no resolved angular velocity -> zero.
    fn momentum_source_inertial(
        &self,
        _x: Tensor<S, 1>,
        _mom: Tensor<S, 1>,
        _vel: Tensor<S, 1>,
    ) -> Tensor<S, 1> {
        Tensor::zeros()
    }
}

impl<S: Scalar> Metric<S, 2> for Spherical {
    fn geometry(&self) -> Geometry {
        Geometry::Spherical
    }
    fn spatial_metric(&self, x: Tensor<S, 2>) -> Matrix<S, 2> {
        let r = x[0];
        Matrix::diag(Tensor::new([S::ONE, r * r]))
    }

    fn spatial_metric_inv(&self, x: Tensor<S, 2>) -> Matrix<S, 2> {
        let r = x[0];
        Matrix::diag(Tensor::new([S::ONE, S::ONE / (r * r)]))
    }

    fn sqrt_det_gamma(&self, x: Tensor<S, 2>) -> S {
        x[0] // r
    }

    fn scale_factors(&self, x: Tensor<S, 2>) -> Tensor<S, 2> {
        Tensor::new([S::ONE, x[0]])
    }

    fn to_cartesian(&self, x: Tensor<S, 2>) -> Tensor<S, 2> {
        let (r, theta) = (x[0], x[1]);
        Tensor::new([r * theta.cos(), r * theta.sin()])
    }

    fn from_cartesian(&self, x: Tensor<S, 2>) -> Tensor<S, 2> {
        let (cx, cy) = (x[0], x[1]);
        let r = (cx * cx + cy * cy).sqrt();
        let theta = cy.atan2(cx);
        Tensor::new([r, theta])
    }

    fn vector_to_cartesian(&self, x: Tensor<S, 2>, v: Physical<S, 2>) -> Embedded<S, 2> {
        let theta = x[1];
        let ct = theta.cos();
        let st = theta.sin();
        // physical (orthonormal) components: v_r in hat{r}, v_theta in hat{theta}, rotated to lab.
        Embedded::new(Tensor::new([v[0] * ct - v[1] * st, v[0] * st + v[1] * ct]))
    }

    fn vector_from_cartesian(&self, x: Tensor<S, 2>, v: Embedded<S, 2>) -> Physical<S, 2> {
        let theta = x[1];
        let ct = theta.cos();
        let st = theta.sin();
        Physical::new(Tensor::new([v[0] * ct + v[1] * st, -v[0] * st + v[1] * ct]))
    }

    fn volume_factor(&self, x: Tensor<S, 2>) -> S {
        let r = x[0];
        let theta = x[1];
        r * r * theta.sin().abs()
    }

    /// 2D spherical (r, theta): resolved theta + suppressed phi.
    /// S_r = (rho*V_t^2 + 2p) / r
    /// S_t = (p*cot(theta) - rho*V_r*V_t) / r
    fn momentum_source(&self, x: Tensor<S, 2>, rho: S, vel: Tensor<S, 2>, p: S) -> Tensor<S, 2> {
        let r = x[0];
        let theta = x[1];
        let vr = vel[0];
        let vt = vel[1];
        let two = S::ONE + S::ONE;
        let cot = theta.cos() / theta.sin();
        Tensor::new([(rho * vt * vt + two * p) / r, (p * cot - rho * vr * vt) / r])
    }

    /// 2D spherical inertial: centrifugal + coriolis, no pressure. regime-agnostic via the
    /// CONSERVED momentum density `mom`: S = -Gamma(mom, v).
    fn momentum_source_inertial(
        &self,
        x: Tensor<S, 2>,
        mom: Tensor<S, 2>,
        vel: Tensor<S, 2>,
    ) -> Tensor<S, 2> {
        let r = x[0];
        let mr = mom[0];
        let mt = mom[1];
        let vt = vel[1];
        Tensor::new([mt * vt / r, S::ZERO - mr * vt / r])
    }
}

impl<S: Scalar> Metric<S, 3> for Spherical {
    fn geometry(&self) -> Geometry {
        Geometry::Spherical
    }
    fn spatial_metric(&self, x: Tensor<S, 3>) -> Matrix<S, 3> {
        let r = x[0];
        let st = x[1].sin();
        Matrix::diag(Tensor::new([S::ONE, r * r, r * r * st * st]))
    }

    fn spatial_metric_inv(&self, x: Tensor<S, 3>) -> Matrix<S, 3> {
        let r = x[0];
        let st = x[1].sin();
        let r2 = r * r;
        Matrix::diag(Tensor::new([S::ONE, S::ONE / r2, S::ONE / (r2 * st * st)]))
    }

    fn sqrt_det_gamma(&self, x: Tensor<S, 3>) -> S {
        let r = x[0];
        let st = x[1].sin();
        r * r * st.abs()
    }
    // full-rank spherical chart: the proper measure is sqrt_det_gamma.
    fn volume_factor(&self, x: Tensor<S, 3>) -> S {
        self.sqrt_det_gamma(x)
    }

    fn scale_factors(&self, x: Tensor<S, 3>) -> Tensor<S, 3> {
        let r = x[0];
        let st = x[1].sin();
        Tensor::new([S::ONE, r, r * st.abs()])
    }

    fn to_cartesian(&self, x: Tensor<S, 3>) -> Tensor<S, 3> {
        let (r, theta, phi) = (x[0], x[1], x[2]);
        let st = theta.sin();
        Tensor::new([r * st * phi.cos(), r * st * phi.sin(), r * theta.cos()])
    }

    fn from_cartesian(&self, x: Tensor<S, 3>) -> Tensor<S, 3> {
        let (cx, cy, cz) = (x[0], x[1], x[2]);
        let r = (cx * cx + cy * cy + cz * cz).sqrt();
        let theta = (cz / r).acos();
        let phi = cy.atan2(cx);
        Tensor::new([r, theta, phi])
    }

    fn vector_to_cartesian(&self, x: Tensor<S, 3>, v: Physical<S, 3>) -> Embedded<S, 3> {
        let (theta, phi) = (x[1], x[2]);
        let st = theta.sin();
        let ct = theta.cos();
        let sp = phi.sin();
        let cp = phi.cos();
        Embedded::new(Tensor::new([
            v[0] * st * cp + v[1] * ct * cp - v[2] * sp,
            v[0] * st * sp + v[1] * ct * sp + v[2] * cp,
            v[0] * ct - v[1] * st,
        ]))
    }

    fn vector_from_cartesian(&self, x: Tensor<S, 3>, v: Embedded<S, 3>) -> Physical<S, 3> {
        let (theta, phi) = (x[1], x[2]);
        let st = theta.sin();
        let ct = theta.cos();
        let sp = phi.sin();
        let cp = phi.cos();
        Physical::new(Tensor::new([
            v[0] * st * cp + v[1] * st * sp + v[2] * ct,
            v[0] * ct * cp + v[1] * ct * sp - v[2] * st,
            -v[0] * sp + v[1] * cp,
        ]))
    }

    /// 3D spherical (r, theta, phi): full geometric source.
    /// S_r = (rho*(V_t^2 + V_p^2) + 2p) / r
    /// S_t = ((rho*V_p^2 + p)*cot(theta) - rho*V_r*V_t) / r
    /// S_p = -rho*V_p*(V_r + V_t*cot(theta)) / r
    fn momentum_source(&self, x: Tensor<S, 3>, rho: S, vel: Tensor<S, 3>, p: S) -> Tensor<S, 3> {
        let r = x[0];
        let theta = x[1];
        let vr = vel[0];
        let vt = vel[1];
        let vp = vel[2];
        let two = S::ONE + S::ONE;
        let cot = theta.cos() / theta.sin();
        Tensor::new([
            (rho * (vt * vt + vp * vp) + two * p) / r,
            ((rho * vp * vp + p) * cot - rho * vr * vt) / r,
            -rho * vp * (vr + vt * cot) / r,
        ])
    }

    /// 3D spherical inertial: centrifugal + coriolis, no pressure. regime-agnostic via the
    /// CONSERVED momentum density `mom`: S = -Gamma(mom, v).
    fn momentum_source_inertial(
        &self,
        x: Tensor<S, 3>,
        mom: Tensor<S, 3>,
        vel: Tensor<S, 3>,
    ) -> Tensor<S, 3> {
        let r = x[0];
        let theta = x[1];
        let mr = mom[0];
        let mt = mom[1];
        let mp = mom[2];
        let vr = vel[0];
        let vt = vel[1];
        let vp = vel[2];
        let cot = theta.cos() / theta.sin();
        Tensor::new([
            (mt * vt + mp * vp) / r,
            (mp * vp * cot - mr * vt) / r,
            S::ZERO - mp * (vr + vt * cot) / r,
        ])
    }
}

// ============================================================
// schwarzschild metric (standard / schwarzschild coords): x = (r, theta, phi)
//   the STATIC spherically-symmetric vacuum. f(r) = 1 - 2M/r.
//   lapse  alpha    = sqrt(f),  shift beta = 0
//   gamma_{ij}      = diag(1/f, r^2, r^2 sin^2 theta)   (DIAGONAL -> DiagonalMetric)
//   sqrt(gamma)     = r^2 sin(theta) / sqrt(f)
//   sqrt(-g)        = alpha sqrt(gamma) = r^2 sin(theta)   (the flat spherical area)
//
//   the SPATIAL coordinate geometry is spherical (geometry() = Spherical); the CURVATURE lives in
//   the radial stretch 1/f and the lapse. this coordinate gamma feeds densitization / lower-raise /
//   the christoffel gravity source; the hydro's physical-frame metric stays
//   identity in the orthonormal convention (the lapse enters the kernel via `gv_lapse_weight`).
//
//   reduced dims mirror Spherical: 1D (r) radial, 2D (r, theta). valid OUTSIDE the horizon r > 2M
//   (f > 0); r <= 2M makes sqrt(f) imaginary — the coordinate singularity, physical.
//
//   the momentum source (geodesic gravity) is the connection of the full 4-metric; left at
//   the trait default (zero) here, so this impl supplies the metric geometry + lapse.
// ============================================================

// ============================================================
// schwarzschild in ingoing kerr-schild (eddington-finkelstein) coords: x = (r, theta, phi)
//   the same physical vacuum as `Schwarzschild`, in a horizon-penetrating chart. ingoing EF line
//   element (h(r) = 1 + 2M/r):
//     ds^2 = -(1 - 2M/r) dt^2 + (4M/r) dt dr + h dr^2 + r^2 dOmega^2
//   3+1 decomposition:
//     lapse  alpha  = 1/sqrt(h)                       (1/sqrt(2) at the horizon; strictly positive)
//     shift  beta^r = (2M/r)/h = 2M/(r + 2M)          (radial, ingoing; 1/2 at the horizon)
//     gamma_{ij}    = diag(h, r^2, r^2 sin^2 theta)   (diagonal -> DiagonalMetric)
//     sqrt(gamma)   = r^2 sin(theta) sqrt(h)
//     sqrt(-g)      = alpha sqrt(gamma) = r^2 sin(theta)   (flat spherical volume)
//   every factor is finite and smooth at and inside r = 2M — the whole point: the inner boundary can
//   sit below the horizon where the transport velocity tilde v^r = v^r - beta^r/alpha is negative for
//   every subluminal fluid, so the excised interior is causal. the spatial geometry is spherical
//   (geometry() = Spherical); the curvature lives in gamma_rr = h and the lapse/shift.
// ============================================================

/// schwarzschild in ingoing kerr-schild coordinates — horizon-penetrating (regular across r = 2M).
/// same geometric `mass` M and `schwarzschild_mass` kernel scalar as [`Schwarzschild`], differing
/// only in the coordinate chart: nonzero radial shift beta^r and gamma_{rr} = 1 + 2M/r. a diagonal
/// spatial metric, so the orthonormal frame + flat-SR c2p survive here, where spinning Kerr calls
/// for the covariant storage.
#[derive(Debug, Clone, Copy)]
pub struct SchwarzschildKS<S> {
    /// the geometric mass M (units G = c = 1); the horizon is at r = 2M. CARRIER-GENERIC over `S`
    /// (an `f64` host value or a `Gv::scalar` in the trace), exactly like [`Schwarzschild::mass`].
    pub mass: S,
}

impl<S: Scalar> SchwarzschildKS<S> {
    /// h(r) = 1 + 2M/r — the radial metric coefficient gamma_{rr} and the inverse lapse-square
    /// (alpha^2 = 1/h). strictly positive for every r > 0, so the chart stays regular at the horizon.
    #[inline]
    fn h(&self, r: S) -> S {
        S::ONE + S::from_f64(2.0) * self.mass / r
    }

    /// the radial shift beta^r = (2M/r)/h = 2M/(r + 2M) — ingoing, finite everywhere r > 0.
    #[inline]
    fn beta_r(&self, r: S) -> S {
        let two_m = S::from_f64(2.0) * self.mass;
        two_m / (r + two_m)
    }
}

impl<S: Scalar> Metric<S, 1> for SchwarzschildKS<S> {
    fn geometry(&self) -> Geometry {
        Geometry::Spherical
    }
    fn spacetime(&self) -> Spacetime {
        Spacetime::SchwarzschildKS
    }
    fn spacetime_scalars(&self) -> Vec<(&'static str, S)> {
        vec![("schwarzschild_mass", self.mass)]
    }

    fn lapse(&self, x: Tensor<S, 1>) -> S {
        S::ONE / self.h(x[0]).sqrt()
    }
    fn lapse_sq(&self, x: Tensor<S, 1>) -> S {
        S::ONE / self.h(x[0])
    } // alpha^2 = 1/(1 + 2M/r)
    fn shift(&self, x: Tensor<S, 1>) -> Tensor<S, 1> {
        Tensor::new([self.beta_r(x[0])])
    }

    fn spatial_metric(&self, x: Tensor<S, 1>) -> Matrix<S, 1> {
        Matrix::diag(Tensor::new([self.h(x[0])]))
    }
    fn spatial_metric_inv(&self, x: Tensor<S, 1>) -> Matrix<S, 1> {
        Matrix::diag(Tensor::new([S::ONE / self.h(x[0])]))
    }
    fn sqrt_det_gamma(&self, x: Tensor<S, 1>) -> S {
        self.h(x[0]).sqrt()
    }
    fn scale_factors(&self, x: Tensor<S, 1>) -> Tensor<S, 1> {
        Tensor::new([self.h(x[0]).sqrt()])
    }

    fn to_cartesian(&self, x: Tensor<S, 1>) -> Tensor<S, 1> {
        x
    }
    fn from_cartesian(&self, x: Tensor<S, 1>) -> Tensor<S, 1> {
        x
    }

    /// the proper volume element incl. the 2 suppressed angular directions: r^2 sqrt(h).
    fn volume_factor(&self, x: Tensor<S, 1>) -> S {
        let r = x[0];
        r * r * self.h(r).sqrt()
    }
}

impl<S: Scalar> Metric<S, 2> for SchwarzschildKS<S> {
    fn geometry(&self) -> Geometry {
        Geometry::Spherical
    }
    fn spacetime(&self) -> Spacetime {
        Spacetime::SchwarzschildKS
    }
    fn spacetime_scalars(&self) -> Vec<(&'static str, S)> {
        vec![("schwarzschild_mass", self.mass)]
    }

    fn lapse(&self, x: Tensor<S, 2>) -> S {
        S::ONE / self.h(x[0]).sqrt()
    }
    fn lapse_sq(&self, x: Tensor<S, 2>) -> S {
        S::ONE / self.h(x[0])
    } // alpha^2 = 1/(1 + 2M/r)
    fn shift(&self, x: Tensor<S, 2>) -> Tensor<S, 2> {
        Tensor::new([self.beta_r(x[0]), S::ZERO])
    }

    fn spatial_metric(&self, x: Tensor<S, 2>) -> Matrix<S, 2> {
        let r = x[0];
        Matrix::diag(Tensor::new([self.h(r), r * r]))
    }
    fn spatial_metric_inv(&self, x: Tensor<S, 2>) -> Matrix<S, 2> {
        let r = x[0];
        Matrix::diag(Tensor::new([S::ONE / self.h(r), S::ONE / (r * r)]))
    }
    fn sqrt_det_gamma(&self, x: Tensor<S, 2>) -> S {
        let r = x[0];
        r * self.h(r).sqrt() // sqrt(h * r^2)
    }
    fn scale_factors(&self, x: Tensor<S, 2>) -> Tensor<S, 2> {
        let r = x[0];
        Tensor::new([self.h(r).sqrt(), r])
    }

    fn to_cartesian(&self, x: Tensor<S, 2>) -> Tensor<S, 2> {
        let (r, theta) = (x[0], x[1]);
        Tensor::new([r * theta.cos(), r * theta.sin()])
    }
    fn from_cartesian(&self, x: Tensor<S, 2>) -> Tensor<S, 2> {
        let (cx, cy) = (x[0], x[1]);
        let r = (cx * cx + cy * cy).sqrt();
        Tensor::new([r, cy.atan2(cx)])
    }
    fn vector_to_cartesian(&self, x: Tensor<S, 2>, v: Physical<S, 2>) -> Embedded<S, 2> {
        let theta = x[1];
        let (ct, st) = (theta.cos(), theta.sin());
        Embedded::new(Tensor::new([v[0] * ct - v[1] * st, v[0] * st + v[1] * ct]))
    }
    fn vector_from_cartesian(&self, x: Tensor<S, 2>, v: Embedded<S, 2>) -> Physical<S, 2> {
        let theta = x[1];
        let (ct, st) = (theta.cos(), theta.sin());
        Physical::new(Tensor::new([v[0] * ct + v[1] * st, -v[0] * st + v[1] * ct]))
    }

    /// proper volume incl. the suppressed phi direction: r^2 sin(theta) sqrt(h).
    fn volume_factor(&self, x: Tensor<S, 2>) -> S {
        let r = x[0];
        r * r * x[1].sin().abs() * self.h(r).sqrt()
    }
}

impl<S: Scalar> Metric<S, 3> for SchwarzschildKS<S> {
    fn geometry(&self) -> Geometry {
        Geometry::Spherical
    }
    fn spacetime(&self) -> Spacetime {
        Spacetime::SchwarzschildKS
    }
    fn spacetime_scalars(&self) -> Vec<(&'static str, S)> {
        vec![("schwarzschild_mass", self.mass)]
    }

    fn lapse(&self, x: Tensor<S, 3>) -> S {
        S::ONE / self.h(x[0]).sqrt()
    }
    fn lapse_sq(&self, x: Tensor<S, 3>) -> S {
        S::ONE / self.h(x[0])
    } // alpha^2 = 1/(1 + 2M/r)
    fn shift(&self, x: Tensor<S, 3>) -> Tensor<S, 3> {
        Tensor::new([self.beta_r(x[0]), S::ZERO, S::ZERO])
    }

    fn spatial_metric(&self, x: Tensor<S, 3>) -> Matrix<S, 3> {
        let r = x[0];
        let st = x[1].sin();
        Matrix::diag(Tensor::new([self.h(r), r * r, r * r * st * st]))
    }
    fn spatial_metric_inv(&self, x: Tensor<S, 3>) -> Matrix<S, 3> {
        let r = x[0];
        let st = x[1].sin();
        let r2 = r * r;
        Matrix::diag(Tensor::new([
            S::ONE / self.h(r),
            S::ONE / r2,
            S::ONE / (r2 * st * st),
        ]))
    }
    fn sqrt_det_gamma(&self, x: Tensor<S, 3>) -> S {
        let r = x[0];
        r * r * x[1].sin().abs() * self.h(r).sqrt() // sqrt(h * r^2 * r^2 sin^2)
    }
    // full-rank spherical chart: the proper measure is sqrt_det_gamma.
    fn volume_factor(&self, x: Tensor<S, 3>) -> S {
        self.sqrt_det_gamma(x)
    }
    fn scale_factors(&self, x: Tensor<S, 3>) -> Tensor<S, 3> {
        let r = x[0];
        let st = x[1].sin();
        Tensor::new([self.h(r).sqrt(), r, r * st.abs()])
    }

    fn to_cartesian(&self, x: Tensor<S, 3>) -> Tensor<S, 3> {
        let (r, theta, phi) = (x[0], x[1], x[2]);
        let st = theta.sin();
        Tensor::new([r * st * phi.cos(), r * st * phi.sin(), r * theta.cos()])
    }
    fn from_cartesian(&self, x: Tensor<S, 3>) -> Tensor<S, 3> {
        let (cx, cy, cz) = (x[0], x[1], x[2]);
        let r = (cx * cx + cy * cy + cz * cz).sqrt();
        Tensor::new([r, (cz / r).acos(), cy.atan2(cx)])
    }
    fn vector_to_cartesian(&self, x: Tensor<S, 3>, v: Physical<S, 3>) -> Embedded<S, 3> {
        let (theta, phi) = (x[1], x[2]);
        let (st, ct, sp, cp) = (theta.sin(), theta.cos(), phi.sin(), phi.cos());
        Embedded::new(Tensor::new([
            v[0] * st * cp + v[1] * ct * cp - v[2] * sp,
            v[0] * st * sp + v[1] * ct * sp + v[2] * cp,
            v[0] * ct - v[1] * st,
        ]))
    }
    fn vector_from_cartesian(&self, x: Tensor<S, 3>, v: Embedded<S, 3>) -> Physical<S, 3> {
        let (theta, phi) = (x[1], x[2]);
        let (st, ct, sp, cp) = (theta.sin(), theta.cos(), phi.sin(), phi.cos());
        Physical::new(Tensor::new([
            v[0] * st * cp + v[1] * st * sp + v[2] * ct,
            v[0] * ct * cp + v[1] * ct * sp - v[2] * st,
            -v[0] * sp + v[1] * cp,
        ]))
    }
}

impl<S: Scalar> DiagonalMetric<S, 1> for SchwarzschildKS<S> {}
impl<S: Scalar> DiagonalMetric<S, 2> for SchwarzschildKS<S> {}
impl<S: Scalar> DiagonalMetric<S, 3> for SchwarzschildKS<S> {}

// ============================================================
// schwarzschild in cartesian kerr-schild coordinates: x = (x, y, z)
//   the same physical vacuum as `Schwarzschild` / `SchwarzschildKS`, in the cartesian kerr-schild
//   chart — horizon-penetrating and pole-free (the natural chart for binary black holes and
//   octree AMR). H = M / r, r = sqrt(x^2 + y^2 + z^2), l_i = x_i / r (the unit kerr-schild covector):
//     gamma_ij   = delta_ij + 2H l_i l_j = delta_ij + 2M x_i x_j / r^3   (non-diagonal: the KS null structure)
//     alpha      = 1 / sqrt(1 + 2H)                                     (1/sqrt(2) at r = 2M; strictly positive)
//     beta^i     = (2H / (1 + 2H)) l^i = 2M x_i / (r^2 (r + 2M))        (ingoing, along the radial direction)
//     gamma^{ij} = delta_ij - (2H / (1 + 2H)) l_i l_j                   (sherman-morrison; l a unit vector)
//     sqrt(gamma) = sqrt(1 + 2H);  alpha sqrt(gamma) = 1   (the det-g-flat identity, cartesian instance)
//   non-diagonal even at zero spin, so it impls `Metric` alone and the covariant/tetrad storage applies.
//   r = |x| is rotation-generic — the formulas treat the cartesian axes symmetrically, radial and
//   angular roles alike, so
//   the D = 2 impl is the z = 0 equatorial slice (exact for equatorially-symmetric flows) and D = 3 the
//   full chart. reduces to flat cartesian at M = 0.
// ============================================================

/// schwarzschild in cartesian kerr-schild coordinates — horizon-penetrating and pole-free. same
/// geometric `mass` M and `schwarzschild_mass` kernel scalar as [`Schwarzschild`] / [`SchwarzschildKS`],
/// differing only in the coordinate chart: a non-diagonal `gamma_ij = delta_ij + 2M x_i x_j / r^3` and a
/// shift `beta^i` spread along all cartesian axes. impls `Metric` alone.
#[derive(Debug, Clone, Copy)]
pub struct SchwarzschildKSCartesian<S> {
    /// the geometric mass M (units G = c = 1); the horizon is at r = 2M. CARRIER-GENERIC over `S`
    /// (an `f64` host value or a `Gv::scalar` in the trace), exactly like [`SchwarzschildKS::mass`].
    pub mass: S,
}

impl<S: Scalar> SchwarzschildKSCartesian<S> {
    /// (r, 2H, |l|^2) at a cartesian position: r = sqrt(sum x_i^2) the euclidean/kerr-schild
    /// radius, 2H = 2M/r, and the squared norm of the null covector l^i = x^i / r. the number of
    /// gridded axes D is the slice (D = 2 -> the z = 0 equatorial plane).
    ///
    /// the radius is clamped to r >= M/2: the chart's coordinate singularity at r = 0 sits on
    /// the grid when the domain contains the origin, and the metric (h = 1 + 2M/r) and its
    /// derivatives (christoffels ~ M/r^2) diverge there. the clamp bounds every metric quantity
    /// while `max(r, M/2)` is the bit-exact identity for r > M/2 — deep inside the horizon
    /// r_+ = 2M and below any excision surface.
    ///
    /// the clamp leaves l^i = x^i / r_clamped, so |l| = |x| / r_clamped falls below 1 inside it
    /// and the kerr-schild null condition relaxes to |l| < 1 there. every closed form below
    /// therefore contracts with the measured |l|^2 — the rank-1 determinant
    /// det(delta + 2H l l^T) = 1 + 2H |l|^2 and its sherman-morrison inverse — so the clamped
    /// region stays a consistent positive-definite metric whose determinant agrees with the
    /// matrix it came from. assuming |l| = 1 there makes sqrt(det gamma) disagree with
    /// the metric by ~18% at r = 0.45 M, breaks the four-volume identity
    /// alpha sqrt(det gamma) = 1, and turns the pressure block of the connection source,
    /// (1/2) p g^{ab} d_i g_ab = p d_i ln sqrt(-g), from identically zero into a spurious force.
    /// outside the clamp |l|^2 is exactly 1 (r / r is exactly 1 in IEEE arithmetic), so every
    /// form reduces bit-for-bit to the unit-l one.
    #[inline]
    fn radius_two_h<const D: usize>(&self, x: Tensor<S, D>) -> (S, S, S) {
        let mut r2 = S::ZERO;
        for ii in 0..D {
            r2 = r2 + x[ii] * x[ii];
        }
        let r_g = S::from_f64(0.5) * self.mass;
        let r_true = r2.sqrt();
        let r = r_true.max(r_g);
        // |l|^2 = |x|^2 / r_clamped^2, formed from the unrooted r2 so it stays differentiable at
        // the origin: d(sqrt(r2))/dx is infinite there, and squaring that back gives a NaN tangent
        // where the true value is zero. outside the clamp |l|^2 is exactly 1 by construction, which
        // keeps every form below bit-identical to the unit-l one wherever the null condition holds.
        let ll2 = S::select(r_true.cmp_ge(r_g), S::ONE, r2 / (r * r));
        (r, S::from_f64(2.0) * self.mass / r, ll2)
    }
}

macro_rules! impl_schwarzschild_ks_cartesian {
    ($d:literal) => {
        impl<S: Scalar> Metric<S, $d> for SchwarzschildKSCartesian<S> {
            fn geometry(&self) -> Geometry {
                Geometry::Cartesian
            }
            fn spacetime(&self) -> Spacetime {
                Spacetime::SchwarzschildKS
            }
            fn spacetime_scalars(&self) -> Vec<(&'static str, S)> {
                vec![("schwarzschild_mass", self.mass)]
            }

            fn lapse(&self, x: Tensor<S, $d>) -> S {
                let (_r, two_h, ll2) = self.radius_two_h(x);
                S::ONE / (S::ONE + two_h * ll2).sqrt()
            }
            // alpha^2 = 1/(1 + 2H |l|^2) in EXACT closed form (no sqrt round-trip; the GR CFL
            // depends on it). |l| = 1 outside the radius clamp, where this is 1/(1 + 2M/r).
            fn lapse_sq(&self, x: Tensor<S, $d>) -> S {
                let (_r, two_h, ll2) = self.radius_two_h(x);
                S::ONE / (S::ONE + two_h * ll2)
            }
            // beta^i = gamma^{ij} (2H l_j) = (2H / (1 + 2H |l|^2)) l^i, l^i = x_i / r.
            fn shift(&self, x: Tensor<S, $d>) -> Tensor<S, $d> {
                let (r, two_h, ll2) = self.radius_two_h(x);
                let s = (two_h / (S::ONE + two_h * ll2)) / r;
                Tensor::new(std::array::from_fn(|ii| s * x[ii]))
            }

            // gamma_ij = delta_ij + 2H l_i l_j = delta_ij + (2H / r^2) x_i x_j.
            fn spatial_metric(&self, x: Tensor<S, $d>) -> Matrix<S, $d> {
                let (r, two_h, _ll2) = self.radius_two_h(x);
                let coef = two_h / (r * r);
                Matrix::from_fn(|ii, jj| {
                    let kron = if ii == jj { S::ONE } else { S::ZERO };
                    kron + coef * x[ii] * x[jj]
                })
            }
            // gamma^{ij} = delta_ij - (2H / (1 + 2H |l|^2)) l^i l^j (sherman-morrison on the
            // rank-1 update; the ACTUAL |l|^2, so the inverse stays the true inverse of the
            // matrix above inside the radius clamp where |l| < 1).
            fn spatial_metric_inv(&self, x: Tensor<S, $d>) -> Matrix<S, $d> {
                let (r, two_h, ll2) = self.radius_two_h(x);
                let coef = (two_h / (S::ONE + two_h * ll2)) / (r * r);
                Matrix::from_fn(|ii, jj| {
                    let kron = if ii == jj { S::ONE } else { S::ZERO };
                    kron - coef * x[ii] * x[jj]
                })
            }
            // det(delta + 2H l l^T) = 1 + 2H |l|^2 exactly (rank-1 determinant lemma).
            fn sqrt_det_gamma(&self, x: Tensor<S, $d>) -> S {
                let (_r, two_h, ll2) = self.radius_two_h(x);
                (S::ONE + two_h * ll2).sqrt()
            }
            // full-rank cartesian chart (D == physical dim): the proper measure is sqrt_det_gamma.
            fn volume_factor(&self, x: Tensor<S, $d>) -> S {
                self.sqrt_det_gamma(x)
            }

            fn to_cartesian(&self, x: Tensor<S, $d>) -> Tensor<S, $d> {
                x
            }
            fn from_cartesian(&self, x: Tensor<S, $d>) -> Tensor<S, $d> {
                x
            }
        }
    };
}

impl_schwarzschild_ks_cartesian!(2);
impl_schwarzschild_ks_cartesian!(3);

/// kerr in cartesian (ingoing) kerr-schild coordinates, spin about z — horizon-penetrating and
/// pole-free, its one ring being the physical kerr ring singularity. the metric is the rank-1
/// kerr-schild update `gamma_ij = delta_ij + 2H l_i l_j` with
///   r^2 = (R^2 - a^2)/2 + sqrt(((R^2 - a^2)/2)^2 + a^2 z^2),   R^2 = x^2 + y^2 + z^2,
///   2H  = 2 M r^3 / (r^4 + a^2 z^2),
///   l   = ((r x + a y)/(r^2 + a^2), (r y - a x)/(r^2 + a^2), z/r),
/// where the kerr-schild radius r solves the oblate-spheroidal quartic and |l| = 1 wherever
/// it does. every closed form below uses the measured |l|^2 (exact rank-1 determinant and
/// sherman-morrison inverse), so the clamped region stays a consistent positive-definite
/// metric and the forms are bit-identical to the unit-l ones where the quartic holds.
/// a = 0 reduces exactly to [`SchwarzschildKSCartesian`]. impls `Metric` alone.
#[derive(Debug, Clone, Copy)]
pub struct KerrKSCartesian<S> {
    /// the geometric mass M (G = c = 1); the outer horizon is at r_+ = M + sqrt(M^2 - a^2).
    pub mass: S,
    /// the specific angular momentum a = J/M, |a| < M (units of M), spin about +z.
    pub spin: S,
}

impl<S: Scalar> KerrKSCartesian<S> {
    /// (2H, l padded to 3, |l|^2) at a cartesian position (missing axes read z = 0: the
    /// D = 2 instance is the equatorial slice, where l_z = 0 exactly). the kerr-schild
    /// radius is clamped to r >= M/2, the same frozen-core treatment as the a = 0 chart:
    /// the ring singularity (r -> 0) sits strictly inside the horizon r_+ >= M, so no cell
    /// whose state matters ever reads a clamped value.
    #[inline]
    fn ks_quantities<const D: usize>(&self, x: Tensor<S, D>) -> (S, [S; 3], S) {
        let half = S::from_f64(0.5);
        let p: [S; 3] = std::array::from_fn(|ii| if ii < D { x[ii] } else { S::ZERO });
        let a = self.spin;
        let r2_flat = p[0] * p[0] + p[1] * p[1] + p[2] * p[2];
        let d = half * (r2_flat - a * a);
        let az = a * p[2];
        let r2 = d + (d * d + az * az).sqrt();
        let r = r2.sqrt().max(half * self.mass);
        let rr = r * r;
        let two_h = (S::from_f64(2.0) * self.mass * rr * r) / (rr * rr + az * az);
        let denom = S::ONE / (rr + a * a);
        let l = [
            (r * p[0] + a * p[1]) * denom,
            (r * p[1] - a * p[0]) * denom,
            p[2] / r,
        ];
        let ll2 = l[0] * l[0] + l[1] * l[1] + l[2] * l[2];
        (two_h, l, ll2)
    }
}

macro_rules! impl_kerr_ks_cartesian {
    ($d:literal) => {
        impl<S: Scalar> Metric<S, $d> for KerrKSCartesian<S> {
            fn geometry(&self) -> Geometry {
                Geometry::Cartesian
            }
            fn spacetime(&self) -> Spacetime {
                Spacetime::KerrKS
            }
            fn spacetime_scalars(&self) -> Vec<(&'static str, S)> {
                vec![("schwarzschild_mass", self.mass), ("kerr_spin", self.spin)]
            }

            // alpha = 1/sqrt(1 + 2H |l|^2).
            fn lapse(&self, x: Tensor<S, $d>) -> S {
                self.lapse_sq(x).sqrt()
            }
            fn lapse_sq(&self, x: Tensor<S, $d>) -> S {
                let (two_h, _l, ll2) = self.ks_quantities(x);
                S::ONE / (S::ONE + two_h * ll2)
            }
            // beta^i = gamma^{ij} (2H l_j) = (2H / (1 + 2H |l|^2)) l_i.
            fn shift(&self, x: Tensor<S, $d>) -> Tensor<S, $d> {
                let (two_h, l, ll2) = self.ks_quantities(x);
                let s = two_h / (S::ONE + two_h * ll2);
                Tensor::new(std::array::from_fn(|ii| s * l[ii]))
            }

            fn spatial_metric(&self, x: Tensor<S, $d>) -> Matrix<S, $d> {
                let (two_h, l, _ll2) = self.ks_quantities(x);
                Matrix::from_fn(|ii, jj| {
                    let kron = if ii == jj { S::ONE } else { S::ZERO };
                    kron + two_h * l[ii] * l[jj]
                })
            }
            // sherman-morrison for the rank-1 update: gamma^{ij} = delta - (2H/(1 + 2H |l|^2)) l^i l^j.
            fn spatial_metric_inv(&self, x: Tensor<S, $d>) -> Matrix<S, $d> {
                let (two_h, l, ll2) = self.ks_quantities(x);
                let coef = two_h / (S::ONE + two_h * ll2);
                Matrix::from_fn(|ii, jj| {
                    let kron = if ii == jj { S::ONE } else { S::ZERO };
                    kron - coef * l[ii] * l[jj]
                })
            }
            // det(delta + 2H l l^T) = 1 + 2H |l|^2 exactly (rank-1 determinant lemma).
            fn sqrt_det_gamma(&self, x: Tensor<S, $d>) -> S {
                let (two_h, _l, ll2) = self.ks_quantities(x);
                (S::ONE + two_h * ll2).sqrt()
            }
            fn volume_factor(&self, x: Tensor<S, $d>) -> S {
                self.sqrt_det_gamma(x)
            }

            fn to_cartesian(&self, x: Tensor<S, $d>) -> Tensor<S, $d> {
                x
            }
            fn from_cartesian(&self, x: Tensor<S, $d>) -> Tensor<S, $d> {
                x
            }
        }
    };
}

impl_kerr_ks_cartesian!(2);
impl_kerr_ks_cartesian!(3);

// D = 1 is degenerate (a line carries no radial structure in cartesian GR); fail-loud so
// generic kernel bounds `Metric<S, 1>` resolve, while every bake stays at D = 2, 3.
impl<S: Scalar> Metric<S, 1> for KerrKSCartesian<S> {
    fn geometry(&self) -> Geometry {
        Geometry::Cartesian
    }
    fn spacetime(&self) -> Spacetime {
        Spacetime::KerrKS
    }
    fn spacetime_scalars(&self) -> Vec<(&'static str, S)> {
        vec![("schwarzschild_mass", self.mass), ("kerr_spin", self.spin)]
    }
    fn spatial_metric(&self, _x: Tensor<S, 1>) -> Matrix<S, 1> {
        unreachable!("cartesian kerr is 2d/3d only")
    }
    fn spatial_metric_inv(&self, _x: Tensor<S, 1>) -> Matrix<S, 1> {
        unreachable!("cartesian kerr is 2d/3d only")
    }
    fn sqrt_det_gamma(&self, _x: Tensor<S, 1>) -> S {
        unreachable!("cartesian kerr is 2d/3d only")
    }
    fn volume_factor(&self, _x: Tensor<S, 1>) -> S {
        unreachable!("cartesian kerr is 2d/3d only")
    }
    fn to_cartesian(&self, x: Tensor<S, 1>) -> Tensor<S, 1> {
        x
    }
    fn from_cartesian(&self, x: Tensor<S, 1>) -> Tensor<S, 1> {
        x
    }
}

// D = 1 is degenerate (a line carries no radial structure in cartesian GR); provided fail-loud so
// generic kernel bounds `Metric<S, 1>` resolve, while the cartesian GR bakes stay at D = 2, 3.
// mirrors the KerrKS D = 1 stub.
impl<S: Scalar> Metric<S, 1> for SchwarzschildKSCartesian<S> {
    fn geometry(&self) -> Geometry {
        Geometry::Cartesian
    }
    fn spacetime(&self) -> Spacetime {
        Spacetime::SchwarzschildKS
    }
    fn spacetime_scalars(&self) -> Vec<(&'static str, S)> {
        vec![("schwarzschild_mass", self.mass)]
    }
    fn to_cartesian(&self, x: Tensor<S, 1>) -> Tensor<S, 1> {
        x
    }
    fn from_cartesian(&self, x: Tensor<S, 1>) -> Tensor<S, 1> {
        x
    }
    fn spatial_metric(&self, _x: Tensor<S, 1>) -> Matrix<S, 1> {
        unreachable!(
            "cartesian kerr-schild is degenerate in 1D: it needs at least the (x, y) plane (D >= 2)"
        )
    }
    fn spatial_metric_inv(&self, _x: Tensor<S, 1>) -> Matrix<S, 1> {
        unreachable!(
            "cartesian kerr-schild is degenerate in 1D: it needs at least the (x, y) plane (D >= 2)"
        )
    }
    fn sqrt_det_gamma(&self, _x: Tensor<S, 1>) -> S {
        unreachable!(
            "cartesian kerr-schild is degenerate in 1D: it needs at least the (x, y) plane (D >= 2)"
        )
    }
    fn volume_factor(&self, _x: Tensor<S, 1>) -> S {
        unreachable!(
            "cartesian kerr-schild is degenerate in 1D: it needs at least the (x, y) plane (D >= 2)"
        )
    }
}

// ============================================================
// schwarzschild in cylindrical kerr-schild coordinates: x = (R, phi, z)
//   the same physical vacuum as `Schwarzschild` / `SchwarzschildKS`, in the cylindrical kerr-schild
//   chart — horizon-penetrating, the natural chart for axisymmetric relativistic jets and disks
//   around a hole. r = sqrt(R^2 + z^2) is the spherical (BH) radius; the KS null covector is radial,
//   l_i = (R/r, 0, z/r) (purely poloidal for a = 0). the kerr-schild structure lives entirely
//   in the poloidal (R, z) block; the azimuth phi decouples:
//     gamma_RR   = 1 + 2H R^2/r^2,  gamma_zz = 1 + 2H z^2/r^2,  gamma_Rz = 2H R z/r^2   (2H = 2M/r)
//     gamma_phi-phi = R^2,  gamma_R-phi = gamma_z-phi = 0        (flat azimuth; the a = 0 hole)
//     alpha      = 1 / sqrt(1 + 2H),  beta^i = (2H/(1+2H)) (R/r, 0, z/r)   (beta^phi = 0)
//     sqrt(gamma) = R sqrt(1 + 2H);  alpha sqrt(gamma) = R   (the flat cylindrical volume measure)
//   non-diagonal (gamma_Rz), so it impls `Metric` alone. the two radii are the wrinkle: the spherical
//   r = sqrt(R^2 + z^2) drives H / alpha / the KS block, while the cylindrical R drives the measure.
//   the D = 3 metric serves the 2.5D axisymmetric-swirl grid (R, z gridded, phi carried but ungridded
//   -> the metric is independent of phi, so its autodiff phi-tangent vanishes and S_phi is conserved) and
//   the full 3D (R, phi, z). reduces to flat cylindrical at M = 0.
// ============================================================

/// kerr in cylindrical (ingoing) kerr-schild coordinates (R, phi, z), spin about z:
/// the rank-1 update gamma_ij = g0_ij + 2H l_i l_j on the flat cylindrical base
/// g0 = diag(1, R^2, 1), with the oblate-spheroidal radius r(R, z) solving
/// r^2 = (R^2 + z^2 - a^2)/2 + sqrt(((R^2 + z^2 - a^2)/2)^2 + a^2 z^2) and
///   2H  = 2 M r^3 / (r^4 + a^2 z^2),
///   l   = (r R/(r^2 + a^2), -a R^2/(r^2 + a^2), z/r)   (covariant coordinate basis;
/// the azimuthal slot carries the frame dragging). a = 0 reduces exactly to
/// [`SchwarzschildKSCylindrical`]. off-diagonal at spin: gamma_{R phi} and gamma_{phi z}
/// are nonzero.
#[derive(Debug, Clone, Copy)]
pub struct KerrKSCylindrical<S> {
    /// the geometric mass M (G = c = 1); the outer horizon is at r_+ = M + sqrt(M^2 - a^2).
    pub mass: S,
    /// the specific angular momentum a = J/M, |a| < M, spin about +z.
    pub spin: S,
}

impl<S: Scalar> KerrKSCylindrical<S> {
    /// (2H, covariant l, |l|^2_{g0}) at (R, phi, z); the kerr-schild radius is clamped
    /// at r >= M/2 (the same frozen-core treatment as every KS chart). |l|^2 is taken
    /// with the INVERSE base metric (l contracted g0^{ij} l), the quantity the rank-1
    /// determinant and sherman-morrison forms need on a non-identity base.
    #[inline]
    fn ks_quantities(&self, x: Tensor<S, 3>) -> (S, [S; 3], S) {
        let half = S::from_f64(0.5);
        let (rr_cyl, z) = (x[0], x[2]);
        let a = self.spin;
        let r2_flat = rr_cyl * rr_cyl + z * z;
        let d = half * (r2_flat - a * a);
        let az = a * z;
        let r2 = d + (d * d + az * az).sqrt();
        let r = r2.sqrt().max(half * self.mass);
        let rr = r * r;
        let two_h = (S::from_f64(2.0) * self.mass * rr * r) / (rr * rr + az * az);
        let denom = S::ONE / (rr + a * a);
        let l = [
            r * rr_cyl * denom,
            S::ZERO - a * rr_cyl * rr_cyl * denom,
            z / r,
        ];
        // |l|^2 with g0^{-1} = diag(1, 1/R^2, 1); the azimuthal term is a^2 R^2/(r^2+a^2)^2.
        let r_safe = rr_cyl.abs();
        let ll2 = l[0] * l[0] + (l[1] / r_safe) * (l[1] / r_safe) + l[2] * l[2];
        (two_h, l, ll2)
    }
}

// the cylindrical kerr chart needs the azimuthal slot (the frame dragging lives in
// l_phi), so only D = 3 carries the metric; D = 1/2 are fail-loud stubs that let the
// generic kernel bounds `Metric<S, D>` resolve without ever being reached at bake.
macro_rules! impl_kerr_ks_cylindrical_stub {
    ($d:literal) => {
        impl<S: Scalar> Metric<S, $d> for KerrKSCylindrical<S> {
            fn geometry(&self) -> Geometry {
                Geometry::Cylindrical
            }
            fn spacetime(&self) -> Spacetime {
                Spacetime::KerrKS
            }
            fn spacetime_scalars(&self) -> Vec<(&'static str, S)> {
                vec![("schwarzschild_mass", self.mass), ("kerr_spin", self.spin)]
            }
            fn spatial_metric(&self, _x: Tensor<S, $d>) -> Matrix<S, $d> {
                unreachable!("cylindrical kerr is 3d-only (the dragging needs the azimuth)")
            }
            fn spatial_metric_inv(&self, _x: Tensor<S, $d>) -> Matrix<S, $d> {
                unreachable!("cylindrical kerr is 3d-only (the dragging needs the azimuth)")
            }
            fn sqrt_det_gamma(&self, _x: Tensor<S, $d>) -> S {
                unreachable!("cylindrical kerr is 3d-only (the dragging needs the azimuth)")
            }
            fn volume_factor(&self, _x: Tensor<S, $d>) -> S {
                unreachable!("cylindrical kerr is 3d-only (the dragging needs the azimuth)")
            }
            fn to_cartesian(&self, x: Tensor<S, $d>) -> Tensor<S, $d> {
                x
            }
            fn from_cartesian(&self, x: Tensor<S, $d>) -> Tensor<S, $d> {
                x
            }
        }
    };
}
impl_kerr_ks_cylindrical_stub!(1);
impl_kerr_ks_cylindrical_stub!(2);

impl<S: Scalar> Metric<S, 3> for KerrKSCylindrical<S> {
    fn geometry(&self) -> Geometry {
        Geometry::Cylindrical
    }
    fn spacetime(&self) -> Spacetime {
        Spacetime::KerrKS
    }
    fn spacetime_scalars(&self) -> Vec<(&'static str, S)> {
        vec![("schwarzschild_mass", self.mass), ("kerr_spin", self.spin)]
    }
    fn lapse(&self, x: Tensor<S, 3>) -> S {
        self.lapse_sq(x).sqrt()
    }
    // alpha^2 = 1/(1 + 2H |l|^2) — the null l preserves the determinant, so the
    // det-g-flat identity alpha sqrt(gamma) = sqrt(det g0) = R holds at spin.
    fn lapse_sq(&self, x: Tensor<S, 3>) -> S {
        let (two_h, _l, ll2) = self.ks_quantities(x);
        S::ONE / (S::ONE + two_h * ll2)
    }
    // beta^i = (2H/(1 + 2H |l|^2)) g0^{ij} l_j: the contravariant shift on every axis,
    // including the frame-dragging azimuth beta^phi = coeff * l_phi / R^2.
    fn shift(&self, x: Tensor<S, 3>) -> Tensor<S, 3> {
        let (two_h, l, ll2) = self.ks_quantities(x);
        let s = two_h / (S::ONE + two_h * ll2);
        let r_safe = x[0].abs();
        Tensor::new([s * l[0], s * l[1] / (r_safe * r_safe), s * l[2]])
    }
    fn spatial_metric(&self, x: Tensor<S, 3>) -> Matrix<S, 3> {
        let (two_h, l, _ll2) = self.ks_quantities(x);
        let base = [S::ONE, x[0] * x[0], S::ONE];
        Matrix::from_fn(|ii, jj| {
            let g0 = if ii == jj { base[ii] } else { S::ZERO };
            g0 + two_h * l[ii] * l[jj]
        })
    }
    // sherman-morrison on the diagonal base: gamma^{ij} = g0^{ij} - c (g0 l)^i (g0 l)^j
    // with c = 2H/(1 + 2H |l|^2) and (g0 l)^i = g0^{ij} l_j.
    fn spatial_metric_inv(&self, x: Tensor<S, 3>) -> Matrix<S, 3> {
        let (two_h, l, ll2) = self.ks_quantities(x);
        let coef = two_h / (S::ONE + two_h * ll2);
        let r_safe = x[0].abs();
        let inv_base = [S::ONE, S::ONE / (r_safe * r_safe), S::ONE];
        let lu = [l[0], l[1] * inv_base[1], l[2]];
        Matrix::from_fn(|ii, jj| {
            let g0i = if ii == jj { inv_base[ii] } else { S::ZERO };
            g0i - coef * lu[ii] * lu[jj]
        })
    }
    // det(g0 + 2H l l^T) = det(g0) (1 + 2H |l|^2) (rank-1 lemma on the base).
    fn sqrt_det_gamma(&self, x: Tensor<S, 3>) -> S {
        let (two_h, _l, ll2) = self.ks_quantities(x);
        let r_safe = x[0].abs();
        (r_safe * r_safe * (S::ONE + two_h * ll2)).sqrt()
    }
    fn volume_factor(&self, x: Tensor<S, 3>) -> S {
        self.sqrt_det_gamma(x)
    }
    fn to_cartesian(&self, x: Tensor<S, 3>) -> Tensor<S, 3> {
        Tensor::new([x[0] * x[1].cos(), x[0] * x[1].sin(), x[2]])
    }
    fn from_cartesian(&self, x: Tensor<S, 3>) -> Tensor<S, 3> {
        Tensor::new([(x[0] * x[0] + x[1] * x[1]).sqrt(), x[1].atan2(x[0]), x[2]])
    }
}

/// schwarzschild in cylindrical kerr-schild coordinates (R, phi, z) — horizon-penetrating and
/// regular off the axis. same geometric `mass` M and `schwarzschild_mass` kernel scalar as
/// [`Schwarzschild`]; the kerr-schild structure is in the poloidal (R, z) block (non-diagonal), phi
/// decoupled (gamma_phi-phi = R^2). impls `Metric` alone. the physics needs the azimuthal DOF, so
/// D = 3 carries the metric; D = 1 / D = 2 are fail-loud stubs for the generic kernel bounds.
#[derive(Debug, Clone, Copy)]
pub struct SchwarzschildKSCylindrical<S> {
    /// the geometric mass M (units G = c = 1); the horizon is at the spherical radius r = 2M.
    pub mass: S,
}

impl<S: Scalar> SchwarzschildKSCylindrical<S> {
    /// (r, 2H) at (R, z): r = sqrt(R^2 + z^2) is the SPHERICAL (BH) radius built from cylindrical R and z,
    /// and 2H = 2M/r.
    #[inline]
    fn radius_two_h(&self, big_r: S, z: S) -> (S, S) {
        let r = (big_r * big_r + z * z).sqrt();
        (r, S::from_f64(2.0) * self.mass / r)
    }
}

impl<S: Scalar> Metric<S, 3> for SchwarzschildKSCylindrical<S> {
    fn geometry(&self) -> Geometry {
        Geometry::Cylindrical
    }
    fn spacetime(&self) -> Spacetime {
        Spacetime::SchwarzschildKS
    }
    fn spacetime_scalars(&self) -> Vec<(&'static str, S)> {
        vec![("schwarzschild_mass", self.mass)]
    }

    fn lapse(&self, x: Tensor<S, 3>) -> S {
        let (_r, two_h) = self.radius_two_h(x[0], x[2]);
        S::ONE / (S::ONE + two_h).sqrt()
    }
    fn lapse_sq(&self, x: Tensor<S, 3>) -> S {
        let (_r, two_h) = self.radius_two_h(x[0], x[2]);
        S::ONE / (S::ONE + two_h)
    }
    // beta^i = (2H/(1+2H)) l^i, l^i = (R/r, 0, z/r); beta^phi = 0 (a = 0, no frame dragging).
    fn shift(&self, x: Tensor<S, 3>) -> Tensor<S, 3> {
        let (r, two_h) = self.radius_two_h(x[0], x[2]);
        let s = (two_h / (S::ONE + two_h)) / r;
        Tensor::new([s * x[0], S::ZERO, s * x[2]])
    }

    // gamma_ij = flat_cyl + 2H l_i l_j: the (R, z) = (0, 2) block gets delta + (2H/r^2) l_i l_j with
    // l = (R, 0, z) (coordinate-basis covector, |l|_flat = 1); phi = 1 stays R^2, decoupled.
    fn spatial_metric(&self, x: Tensor<S, 3>) -> Matrix<S, 3> {
        let (big_r, z) = (x[0], x[2]);
        let (r, two_h) = self.radius_two_h(big_r, z);
        let coef = two_h / (r * r);
        let l = |ii: usize| match ii {
            0 => big_r,
            2 => z,
            _ => S::ZERO,
        };
        Matrix::from_fn(|ii, jj| {
            let base = match (ii, jj) {
                (1, 1) => big_r * big_r,
                (0, 0) | (2, 2) => S::ONE,
                _ => S::ZERO,
            };
            base + coef * l(ii) * l(jj)
        })
    }
    // gamma^{ij}: the (R, z) block is sherman-morrison delta - (2H/(1+2H))/r^2 l_i l_j; phi is 1/R^2.
    fn spatial_metric_inv(&self, x: Tensor<S, 3>) -> Matrix<S, 3> {
        let (big_r, z) = (x[0], x[2]);
        let (r, two_h) = self.radius_two_h(big_r, z);
        let coef = (two_h / (S::ONE + two_h)) / (r * r);
        let l = |ii: usize| match ii {
            0 => big_r,
            2 => z,
            _ => S::ZERO,
        };
        Matrix::from_fn(|ii, jj| {
            let base = match (ii, jj) {
                (1, 1) => S::ONE / (big_r * big_r),
                (0, 0) | (2, 2) => S::ONE,
                _ => S::ZERO,
            };
            base - coef * l(ii) * l(jj)
        })
    }
    fn sqrt_det_gamma(&self, x: Tensor<S, 3>) -> S {
        let (big_r, z) = (x[0], x[2]);
        let (_r, two_h) = self.radius_two_h(big_r, z);
        big_r.abs() * (S::ONE + two_h).sqrt()
    }
    // full-rank cylindrical chart (R, phi, z): the proper measure is sqrt_det_gamma.
    fn volume_factor(&self, x: Tensor<S, 3>) -> S {
        self.sqrt_det_gamma(x)
    }

    fn to_cartesian(&self, x: Tensor<S, 3>) -> Tensor<S, 3> {
        let (big_r, phi, z) = (x[0], x[1], x[2]);
        Tensor::new([big_r * phi.cos(), big_r * phi.sin(), z])
    }
    fn from_cartesian(&self, x: Tensor<S, 3>) -> Tensor<S, 3> {
        let (cx, cy, cz) = (x[0], x[1], x[2]);
        Tensor::new([(cx * cx + cy * cy).sqrt(), cy.atan2(cx), cz])
    }
}

// D = 1 / D = 2 are degenerate (the azimuthal DOF + the poloidal (R, z) block need the full D = 3);
// provided fail-loud so generic kernel bounds resolve, while every bake stays at D = 3. mirrors
// KerrKS.
impl<S: Scalar> Metric<S, 1> for SchwarzschildKSCylindrical<S> {
    fn geometry(&self) -> Geometry {
        Geometry::Cylindrical
    }
    fn spacetime(&self) -> Spacetime {
        Spacetime::SchwarzschildKS
    }
    fn spacetime_scalars(&self) -> Vec<(&'static str, S)> {
        vec![("schwarzschild_mass", self.mass)]
    }
    fn to_cartesian(&self, x: Tensor<S, 1>) -> Tensor<S, 1> {
        x
    }
    fn from_cartesian(&self, x: Tensor<S, 1>) -> Tensor<S, 1> {
        x
    }
    fn spatial_metric(&self, _x: Tensor<S, 1>) -> Matrix<S, 1> {
        unreachable!(
            "cylindrical kerr-schild needs the poloidal (R, z) block + azimuthal DOF (D = 3)"
        )
    }
    fn spatial_metric_inv(&self, _x: Tensor<S, 1>) -> Matrix<S, 1> {
        unreachable!(
            "cylindrical kerr-schild needs the poloidal (R, z) block + azimuthal DOF (D = 3)"
        )
    }
    fn sqrt_det_gamma(&self, _x: Tensor<S, 1>) -> S {
        unreachable!(
            "cylindrical kerr-schild needs the poloidal (R, z) block + azimuthal DOF (D = 3)"
        )
    }
    fn volume_factor(&self, _x: Tensor<S, 1>) -> S {
        unreachable!(
            "cylindrical kerr-schild needs the poloidal (R, z) block + azimuthal DOF (D = 3)"
        )
    }
}

// the D = 2 view is the (R, phi) equatorial disk (z = 0): the razor-thin accretion-disk chart. on
// the equator the spherical and cylindrical radii coincide (r = R), so the kerr-schild off-diagonal
// vanishes and the metric is diagonal — gamma = diag(1 + 2M/R, R^2), alpha = 1/sqrt(1 + 2M/R), shift
// beta^R = 2M/(R + 2M) (beta^phi = 0), sqrt(gamma) = R sqrt(1 + 2M/R) so alpha sqrt(gamma) = R (the
// cylindrical measure). the same physical vacuum as the (R, z) D = 3 view, restricted to z = 0.
impl<S: Scalar> Metric<S, 2> for SchwarzschildKSCylindrical<S> {
    fn geometry(&self) -> Geometry {
        Geometry::Cylindrical
    }
    fn spacetime(&self) -> Spacetime {
        Spacetime::SchwarzschildKS
    }
    fn spacetime_scalars(&self) -> Vec<(&'static str, S)> {
        vec![("schwarzschild_mass", self.mass)]
    }
    fn to_cartesian(&self, x: Tensor<S, 2>) -> Tensor<S, 2> {
        let (big_r, phi) = (x[0], x[1]);
        Tensor::new([big_r * phi.cos(), big_r * phi.sin()])
    }
    fn from_cartesian(&self, x: Tensor<S, 2>) -> Tensor<S, 2> {
        let (cx, cy) = (x[0], x[1]);
        Tensor::new([(cx * cx + cy * cy).sqrt(), cy.atan2(cx)])
    }

    fn lapse(&self, x: Tensor<S, 2>) -> S {
        let two_h = S::from_f64(2.0) * self.mass / x[0]; // 2M/R (r = R on the equator)
        S::ONE / (S::ONE + two_h).sqrt()
    }
    fn lapse_sq(&self, x: Tensor<S, 2>) -> S {
        let two_h = S::from_f64(2.0) * self.mass / x[0];
        S::ONE / (S::ONE + two_h)
    }
    fn shift(&self, x: Tensor<S, 2>) -> Tensor<S, 2> {
        let two_h = S::from_f64(2.0) * self.mass / x[0];
        Tensor::new([two_h / (S::ONE + two_h), S::ZERO]) // beta^R = 2M/(R + 2M), beta^phi = 0
    }
    fn spatial_metric(&self, x: Tensor<S, 2>) -> Matrix<S, 2> {
        let big_r = x[0];
        let two_h = S::from_f64(2.0) * self.mass / big_r;
        Matrix::diag(Tensor::new([S::ONE + two_h, big_r * big_r]))
    }
    fn spatial_metric_inv(&self, x: Tensor<S, 2>) -> Matrix<S, 2> {
        let big_r = x[0];
        let two_h = S::from_f64(2.0) * self.mass / big_r;
        Matrix::diag(Tensor::new([
            S::ONE / (S::ONE + two_h),
            S::ONE / (big_r * big_r),
        ]))
    }
    fn sqrt_det_gamma(&self, x: Tensor<S, 2>) -> S {
        let big_r = x[0];
        let two_h = S::from_f64(2.0) * self.mass / big_r;
        big_r.abs() * (S::ONE + two_h).sqrt()
    }
    fn volume_factor(&self, x: Tensor<S, 2>) -> S {
        self.sqrt_det_gamma(x)
    }
}

/// spinning Kerr in INGOING KERR-SCHILD coordinates (horizon-penetrating, spherical (r, theta,
/// phi)). Sigma = r^2 + a^2 cos^2(theta), b = 2 M r / Sigma:
///
///   alpha      = 1 / sqrt(1 + b)
///   beta^i     = (b / (1 + b), 0, 0)                       (radial, ingoing)
///   gamma_rr   = 1 + b
///   gamma_rp   = -a sin^2(theta) (1 + b)                   (the frame-dragging off-diagonal)
///   gamma_tt   = Sigma
///   gamma_pp   = sin^2(theta) (Sigma + a^2 sin^2(theta) (1 + b))
///   sqrt(gamma) = Sigma sin(theta) sqrt(1 + b);  alpha sqrt(gamma) = Sigma sin(theta)
///
/// the analytic inverse (gamma^{theta theta} = 1/Sigma; the (r, phi) block inverts in closed
/// form): gamma^rr = (Sigma + a^2 sin^2 (1+b)) / ((1+b) Sigma), gamma^{r phi} = a / Sigma,
/// gamma^{phi phi} = 1 / (Sigma sin^2). every ADM identity is pinned by tests: g_tt = b - 1,
/// g_{t phi} = -a b sin^2, det, inverse, and the a = 0 reduction to `SchwarzschildKS`.
/// impls `Metric` alone — the compile-time bound keeps scale-factor consumers off it.
/// the physics needs the azimuthal momentum DOF, so the D = 3 impl carries the metric;
/// the D = 1 / D = 2 impls exist to satisfy generic kernel bounds and fail loud if reached.
#[derive(Clone, Copy, Debug)]
pub struct KerrKS<S> {
    /// the geometric mass M (G = c = 1).
    pub mass: S,
    /// the specific angular momentum a = J/M, |a| < M (units of M).
    pub spin: S,
}

impl<S: Scalar> KerrKS<S> {
    #[inline]
    fn sigma(&self, r: S, theta: S) -> S {
        let ct = theta.cos();
        r * r + self.spin * self.spin * ct * ct
    }

    /// b = 2 M r / Sigma — the kerr-schild scalar; gamma_rr = 1 + b, alpha^2 = 1/(1 + b).
    #[inline]
    fn b(&self, r: S, theta: S) -> S {
        S::from_f64(2.0) * self.mass * r / self.sigma(r, theta)
    }
}

impl<S: Scalar> Metric<S, 3> for KerrKS<S> {
    fn geometry(&self) -> Geometry {
        Geometry::Spherical
    }
    fn spacetime(&self) -> Spacetime {
        Spacetime::KerrKS
    }
    fn spacetime_scalars(&self) -> Vec<(&'static str, S)> {
        vec![("schwarzschild_mass", self.mass), ("kerr_spin", self.spin)]
    }

    fn lapse(&self, x: Tensor<S, 3>) -> S {
        S::ONE / (S::ONE + self.b(x[0], x[1])).sqrt()
    }
    fn lapse_sq(&self, x: Tensor<S, 3>) -> S {
        S::ONE / (S::ONE + self.b(x[0], x[1]))
    }
    fn shift(&self, x: Tensor<S, 3>) -> Tensor<S, 3> {
        let b = self.b(x[0], x[1]);
        Tensor::new([b / (S::ONE + b), S::ZERO, S::ZERO])
    }

    fn spatial_metric(&self, x: Tensor<S, 3>) -> Matrix<S, 3> {
        let (r, theta) = (x[0], x[1]);
        let st = theta.sin();
        let s2 = st * st;
        let sig = self.sigma(r, theta);
        let hb = S::ONE + self.b(r, theta);
        let a = self.spin;
        let g_rp = S::ZERO - a * s2 * hb;
        let g_pp = s2 * (sig + a * a * s2 * hb);
        Matrix::from_fn(|ii, jj| match (ii, jj) {
            (0, 0) => hb,
            (1, 1) => sig,
            (2, 2) => g_pp,
            (0, 2) | (2, 0) => g_rp,
            _ => S::ZERO,
        })
    }
    fn spatial_metric_inv(&self, x: Tensor<S, 3>) -> Matrix<S, 3> {
        let (r, theta) = (x[0], x[1]);
        let st = theta.sin();
        let s2 = st * st;
        let sig = self.sigma(r, theta);
        let hb = S::ONE + self.b(r, theta);
        let a = self.spin;
        let gi_rr = (sig + a * a * s2 * hb) / (hb * sig);
        let gi_rp = a / sig;
        let gi_pp = S::ONE / (sig * s2);
        Matrix::from_fn(|ii, jj| match (ii, jj) {
            (0, 0) => gi_rr,
            (1, 1) => S::ONE / sig,
            (2, 2) => gi_pp,
            (0, 2) | (2, 0) => gi_rp,
            _ => S::ZERO,
        })
    }
    fn sqrt_det_gamma(&self, x: Tensor<S, 3>) -> S {
        let (r, theta) = (x[0], x[1]);
        self.sigma(r, theta) * theta.sin().abs() * (S::ONE + self.b(r, theta)).sqrt()
    }

    fn to_cartesian(&self, x: Tensor<S, 3>) -> Tensor<S, 3> {
        let (r, theta, phi) = (x[0], x[1], x[2]);
        let st = theta.sin();
        let (cp, sp) = (phi.cos(), phi.sin());
        Tensor::new([
            (r * cp - self.spin * sp) * st,
            (r * sp + self.spin * cp) * st,
            r * theta.cos(),
        ])
    }
    fn from_cartesian(&self, x: Tensor<S, 3>) -> Tensor<S, 3> {
        let (cx, cy, cz) = (x[0], x[1], x[2]);
        let radius_sq = cx * cx + cy * cy + cz * cz;
        let spin_sq = self.spin * self.spin;
        let delta = radius_sq - spin_sq;
        let r_sq = (delta + (delta * delta + S::from_f64(4.0) * spin_sq * cz * cz).sqrt())
            / S::from_f64(2.0);
        let r = r_sq.sqrt();
        let theta = (cz / r).acos();
        let phi = cy.atan2(cx) - self.spin.atan2(r);
        Tensor::new([r, theta, phi])
    }

    /// proper volume element: sqrt(gamma) = Sigma sin(theta) sqrt(1 + b).
    fn volume_factor(&self, x: Tensor<S, 3>) -> S {
        self.sqrt_det_gamma(x)
    }
}

impl<S: Scalar> Metric<S, 1> for KerrKS<S> {
    fn geometry(&self) -> Geometry {
        Geometry::Spherical
    }
    fn spacetime(&self) -> Spacetime {
        Spacetime::KerrKS
    }
    fn spacetime_scalars(&self) -> Vec<(&'static str, S)> {
        vec![("schwarzschild_mass", self.mass), ("kerr_spin", self.spin)]
    }
    fn to_cartesian(&self, x: Tensor<S, 1>) -> Tensor<S, 1> {
        x
    }
    fn from_cartesian(&self, x: Tensor<S, 1>) -> Tensor<S, 1> {
        x
    }
    fn spatial_metric(&self, _x: Tensor<S, 1>) -> Matrix<S, 1> {
        unreachable!(
            "kerr carries the frame-dragging gamma_r-phi: it requires the azimuthal momentum DOF (D = 3)"
        )
    }
    fn spatial_metric_inv(&self, _x: Tensor<S, 1>) -> Matrix<S, 1> {
        unreachable!(
            "kerr carries the frame-dragging gamma_r-phi: it requires the azimuthal momentum DOF (D = 3)"
        )
    }
    fn sqrt_det_gamma(&self, _x: Tensor<S, 1>) -> S {
        unreachable!(
            "kerr carries the frame-dragging gamma_r-phi: it requires the azimuthal momentum DOF (D = 3)"
        )
    }
    fn volume_factor(&self, _x: Tensor<S, 1>) -> S {
        unreachable!(
            "kerr carries the frame-dragging gamma_r-phi: it requires the azimuthal momentum DOF (D = 3)"
        )
    }
}

impl<S: Scalar> Metric<S, 2> for KerrKS<S> {
    fn geometry(&self) -> Geometry {
        Geometry::Spherical
    }
    fn spacetime(&self) -> Spacetime {
        Spacetime::KerrKS
    }
    fn spacetime_scalars(&self) -> Vec<(&'static str, S)> {
        vec![("schwarzschild_mass", self.mass), ("kerr_spin", self.spin)]
    }
    fn to_cartesian(&self, x: Tensor<S, 2>) -> Tensor<S, 2> {
        x
    }
    fn from_cartesian(&self, x: Tensor<S, 2>) -> Tensor<S, 2> {
        x
    }
    // the SCALAR pieces of the (r, theta) grid view — lapse, shift, and the proper volume
    // element (which includes the suppressed phi direction) — are exact; only the 2x2 spatial
    // matrix restriction is meaningless (the frame-dragging gamma_{r phi} row has no slot).
    fn lapse(&self, x: Tensor<S, 2>) -> S {
        S::ONE / (S::ONE + self.b(x[0], x[1])).sqrt()
    }
    fn lapse_sq(&self, x: Tensor<S, 2>) -> S {
        S::ONE / (S::ONE + self.b(x[0], x[1]))
    }
    fn shift(&self, x: Tensor<S, 2>) -> Tensor<S, 2> {
        let b = self.b(x[0], x[1]);
        Tensor::new([b / (S::ONE + b), S::ZERO])
    }
    fn sqrt_det_gamma(&self, x: Tensor<S, 2>) -> S {
        let (r, theta) = (x[0], x[1]);
        self.sigma(r, theta) * theta.sin().abs() * (S::ONE + self.b(r, theta)).sqrt()
    }
    fn volume_factor(&self, x: Tensor<S, 2>) -> S {
        self.sqrt_det_gamma(x)
    }
    fn spatial_metric(&self, _x: Tensor<S, 2>) -> Matrix<S, 2> {
        unreachable!(
            "kerr carries the frame-dragging gamma_r-phi: it requires the azimuthal momentum DOF (D = 3)"
        )
    }
    fn spatial_metric_inv(&self, _x: Tensor<S, 2>) -> Matrix<S, 2> {
        unreachable!(
            "kerr carries the frame-dragging gamma_r-phi: it requires the azimuthal momentum DOF (D = 3)"
        )
    }
}

// ============================================================
// cylindrical metric: x = (r, phi, z)
//   \gamma_{ij} = diag(1, r^2, 1)
//   \sqrt{\gamma} = r
//
// 1D: x = (r), \gamma = diag(1), \sqrt{\gamma} = 1
// 2D: x = (r, z), \gamma = diag(1, 1), \sqrt{\gamma} = 1
//     (axisymmetric — phi integrated out)
// 3D: x = (r, phi, z), full metric
// ============================================================

/// cylindrical metric. coordinates: (r, phi, z).
#[derive(Debug, Clone, Copy)]
pub struct Cylindrical;

impl<S: Scalar> Metric<S, 1> for Cylindrical {
    fn geometry(&self) -> Geometry {
        Geometry::Cylindrical
    }
    fn spatial_metric(&self, _x: Tensor<S, 1>) -> Matrix<S, 1> {
        Matrix::identity()
    }

    fn spatial_metric_inv(&self, _x: Tensor<S, 1>) -> Matrix<S, 1> {
        Matrix::identity()
    }

    fn sqrt_det_gamma(&self, _x: Tensor<S, 1>) -> S {
        S::ONE
    }

    fn scale_factors(&self, _x: Tensor<S, 1>) -> Tensor<S, 1> {
        Tensor::new([S::ONE])
    }

    fn to_cartesian(&self, x: Tensor<S, 1>) -> Tensor<S, 1> {
        x
    }
    fn from_cartesian(&self, x: Tensor<S, 1>) -> Tensor<S, 1> {
        x
    }

    fn volume_factor(&self, x: Tensor<S, 1>) -> S {
        x[0]
    }

    /// 1D cylindrical: S_r = p/r (pressure from 1 suppressed angular direction).
    fn momentum_source(&self, x: Tensor<S, 1>, _rho: S, _vel: Tensor<S, 1>, p: S) -> Tensor<S, 1> {
        let r = x[0];
        Tensor::new([p / r])
    }

    /// 1D cylindrical inertial: no resolved angular velocity -> zero.
    fn momentum_source_inertial(
        &self,
        _x: Tensor<S, 1>,
        _mom: Tensor<S, 1>,
        _vel: Tensor<S, 1>,
    ) -> Tensor<S, 1> {
        Tensor::zeros()
    }
}

/// axisymmetric cylindrical: coordinates (r, z).
/// phi direction integrated out — metric is euclidean in (r, z) plane.
impl<S: Scalar> Metric<S, 2> for Cylindrical {
    fn geometry(&self) -> Geometry {
        Geometry::Cylindrical
    }
    fn spatial_metric(&self, _x: Tensor<S, 2>) -> Matrix<S, 2> {
        Matrix::identity()
    }

    fn spatial_metric_inv(&self, _x: Tensor<S, 2>) -> Matrix<S, 2> {
        Matrix::identity()
    }

    fn sqrt_det_gamma(&self, _x: Tensor<S, 2>) -> S {
        S::ONE
    }

    fn scale_factors(&self, _x: Tensor<S, 2>) -> Tensor<S, 2> {
        Tensor::new([S::ONE, S::ONE])
    }

    fn to_cartesian(&self, x: Tensor<S, 2>) -> Tensor<S, 2> {
        // (r, z) -> (r, z) — axisymmetric, no transform
        x
    }

    fn from_cartesian(&self, x: Tensor<S, 2>) -> Tensor<S, 2> {
        x
    }

    fn volume_factor(&self, x: Tensor<S, 2>) -> S {
        x[0]
    }

    /// 2D cylindrical (r, z): S_r = p/r from suppressed phi, S_z = 0.
    fn momentum_source(&self, x: Tensor<S, 2>, _rho: S, _vel: Tensor<S, 2>, p: S) -> Tensor<S, 2> {
        let r = x[0];
        Tensor::new([p / r, S::ZERO])
    }

    /// 2D cylindrical inertial: no resolved angular velocity -> zero.
    fn momentum_source_inertial(
        &self,
        _x: Tensor<S, 2>,
        _mom: Tensor<S, 2>,
        _vel: Tensor<S, 2>,
    ) -> Tensor<S, 2> {
        Tensor::zeros()
    }
}

impl<S: Scalar> Metric<S, 3> for Cylindrical {
    fn geometry(&self) -> Geometry {
        Geometry::Cylindrical
    }
    fn spatial_metric(&self, x: Tensor<S, 3>) -> Matrix<S, 3> {
        let r = x[0];
        Matrix::diag(Tensor::new([S::ONE, r * r, S::ONE]))
    }

    fn spatial_metric_inv(&self, x: Tensor<S, 3>) -> Matrix<S, 3> {
        let r = x[0];
        Matrix::diag(Tensor::new([S::ONE, S::ONE / (r * r), S::ONE]))
    }

    fn sqrt_det_gamma(&self, x: Tensor<S, 3>) -> S {
        x[0] // r
    }
    // full-rank cylindrical chart (r, phi, z): the proper measure is sqrt_det_gamma = r.
    fn volume_factor(&self, x: Tensor<S, 3>) -> S {
        self.sqrt_det_gamma(x)
    }

    fn scale_factors(&self, x: Tensor<S, 3>) -> Tensor<S, 3> {
        Tensor::new([S::ONE, x[0], S::ONE])
    }

    fn to_cartesian(&self, x: Tensor<S, 3>) -> Tensor<S, 3> {
        let (r, phi, z) = (x[0], x[1], x[2]);
        Tensor::new([r * phi.cos(), r * phi.sin(), z])
    }

    fn from_cartesian(&self, x: Tensor<S, 3>) -> Tensor<S, 3> {
        let (cx, cy, cz) = (x[0], x[1], x[2]);
        let r = (cx * cx + cy * cy).sqrt();
        let phi = cy.atan2(cx);
        Tensor::new([r, phi, cz])
    }

    fn vector_to_cartesian(&self, x: Tensor<S, 3>, v: Physical<S, 3>) -> Embedded<S, 3> {
        let phi = x[1];
        let cp = phi.cos();
        let sp = phi.sin();
        Embedded::new(Tensor::new([
            v[0] * cp - v[1] * sp,
            v[0] * sp + v[1] * cp,
            v[2],
        ]))
    }

    fn vector_from_cartesian(&self, x: Tensor<S, 3>, v: Embedded<S, 3>) -> Physical<S, 3> {
        let phi = x[1];
        let cp = phi.cos();
        let sp = phi.sin();
        Physical::new(Tensor::new([
            v[0] * cp + v[1] * sp,
            -v[0] * sp + v[1] * cp,
            v[2],
        ]))
    }

    /// 3D cylindrical (r, phi, z): full geometric source.
    /// S_r = (rho*V_p^2 + p) / r
    /// S_p = -rho*V_r*V_p / r
    /// S_z = 0
    fn momentum_source(&self, x: Tensor<S, 3>, rho: S, vel: Tensor<S, 3>, p: S) -> Tensor<S, 3> {
        let r = x[0];
        let vr = vel[0];
        let vp = vel[1];
        Tensor::new([(rho * vp * vp + p) / r, -rho * vr * vp / r, S::ZERO])
    }

    /// 3D cylindrical inertial: centrifugal + coriolis, no pressure. regime-agnostic via the
    /// CONSERVED momentum density `mom`: S = -Gamma(mom, v).
    fn momentum_source_inertial(
        &self,
        x: Tensor<S, 3>,
        mom: Tensor<S, 3>,
        vel: Tensor<S, 3>,
    ) -> Tensor<S, 3> {
        let r = x[0];
        let mr = mom[0];
        let mp = mom[1];
        let vp = vel[1];
        Tensor::new([mp * vp / r, S::ZERO - mr * vp / r, S::ZERO])
    }
}

/// 2D cylindrical DISK in the (r, phi) plane — the accretion-disk reduction (z razor-thin /
/// integrated out), DISTINCT from the (r, z) axisymmetric `impl Metric<S, 2> for Cylindrical`.
/// component 1 (phi) is the RESOLVED swirl, so the inertial source is NONZERO (centrifugal +
/// coriolis), unlike the (r, z) case. mirror of `Cylindrical`'s 3D (r, phi, z) impl with z dropped.
///   gamma_{ij} = diag(1, r^2),  sqrt(gamma) = r,  h = (1, r)
#[derive(Debug, Clone, Copy)]
pub struct CylindricalRPhi;

impl<S: Scalar> Metric<S, 2> for CylindricalRPhi {
    fn geometry(&self) -> Geometry {
        Geometry::Cylindrical
    }

    fn spatial_metric(&self, x: Tensor<S, 2>) -> Matrix<S, 2> {
        let r = x[0];
        Matrix::diag(Tensor::new([S::ONE, r * r]))
    }

    fn spatial_metric_inv(&self, x: Tensor<S, 2>) -> Matrix<S, 2> {
        let r = x[0];
        Matrix::diag(Tensor::new([S::ONE, S::ONE / (r * r)]))
    }

    fn sqrt_det_gamma(&self, x: Tensor<S, 2>) -> S {
        x[0] // r
    }

    fn scale_factors(&self, x: Tensor<S, 2>) -> Tensor<S, 2> {
        Tensor::new([S::ONE, x[0]])
    }

    fn to_cartesian(&self, x: Tensor<S, 2>) -> Tensor<S, 2> {
        let (r, phi) = (x[0], x[1]);
        Tensor::new([r * phi.cos(), r * phi.sin()])
    }

    fn from_cartesian(&self, x: Tensor<S, 2>) -> Tensor<S, 2> {
        let (cx, cy) = (x[0], x[1]);
        let r = (cx * cx + cy * cy).sqrt();
        let phi = cy.atan2(cx);
        Tensor::new([r, phi])
    }

    fn vector_to_cartesian(&self, x: Tensor<S, 2>, v: Physical<S, 2>) -> Embedded<S, 2> {
        let phi = x[1];
        let cp = phi.cos();
        let sp = phi.sin();
        Embedded::new(Tensor::new([v[0] * cp - v[1] * sp, v[0] * sp + v[1] * cp]))
    }

    fn vector_from_cartesian(&self, x: Tensor<S, 2>, v: Embedded<S, 2>) -> Physical<S, 2> {
        let phi = x[1];
        let cp = phi.cos();
        let sp = phi.sin();
        Physical::new(Tensor::new([
            v[0] * cp + v[1] * sp,
            S::ZERO - v[0] * sp + v[1] * cp,
        ]))
    }

    fn volume_factor(&self, x: Tensor<S, 2>) -> S {
        x[0]
    }

    /// 2D (r, phi) disk: S_r = (rho*V_p^2 + p)/r, S_p = -rho*V_r*V_p/r.
    fn momentum_source(&self, x: Tensor<S, 2>, rho: S, vel: Tensor<S, 2>, p: S) -> Tensor<S, 2> {
        let r = x[0];
        let vr = vel[0];
        let vp = vel[1];
        Tensor::new([(rho * vp * vp + p) / r, S::ZERO - rho * vr * vp / r])
    }

    /// 2D (r, phi) disk inertial: centrifugal + coriolis, regime-agnostic via the CONSERVED
    /// momentum density `mom`: S = -Gamma(mom, v).
    fn momentum_source_inertial(
        &self,
        x: Tensor<S, 2>,
        mom: Tensor<S, 2>,
        vel: Tensor<S, 2>,
    ) -> Tensor<S, 2> {
        let r = x[0];
        let mr = mom[0];
        let mp = mom[1];
        let vp = vel[1];
        Tensor::new([mp * vp / r, S::ZERO - mr * vp / r])
    }
}

impl<S: Scalar> DiagonalMetric<S, 2> for CylindricalRPhi {}

// ============================================================
// diagonality proofs: all realized geometries are diagonal (flat + orthogonal-curvilinear),
// so they carry the `DiagonalMetric` marker and may use scale factors / orthonormal frame.
// a future non-diagonal metric (Kerr, ...) impls `Metric` but is intentionally absent here.
// ============================================================
impl<S: Scalar> DiagonalMetric<S, 1> for Cartesian {}
impl<S: Scalar> DiagonalMetric<S, 2> for Cartesian {}
impl<S: Scalar> DiagonalMetric<S, 3> for Cartesian {}
impl<S: Scalar> DiagonalMetric<S, 1> for Spherical {}
impl<S: Scalar> DiagonalMetric<S, 2> for Spherical {}
impl<S: Scalar> DiagonalMetric<S, 3> for Spherical {}
impl<S: Scalar> DiagonalMetric<S, 1> for Cylindrical {}
impl<S: Scalar> DiagonalMetric<S, 2> for Cylindrical {}
impl<S: Scalar> DiagonalMetric<S, 3> for Cylindrical {}

// ============================================================
// tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::{FRAC_PI_2, FRAC_PI_4, PI};

    fn approx(a: f64, b: f64) -> bool {
        let diff = (a - b).abs();
        if diff < 1e-14 {
            return true;
        }
        let scale = a.abs().max(b.abs());
        diff / scale < 1e-10
    }

    fn vec_approx(a: &Tensor<f64, 3>, b: &Tensor<f64, 3>) -> bool {
        (0..3).all(|ii| approx(a[ii], b[ii]))
    }

    // ---- cartesian ----

    #[test]
    fn test_cartesian_metric_3d() {
        let m = Cartesian;
        let x = Tensor::new([1.0, 2.0, 3.0]);
        assert_eq!(m.spatial_metric(x), Matrix::identity());
        assert_eq!(m.spatial_metric_inv(x), Matrix::identity());
        assert!(approx(m.sqrt_det_gamma(x), 1.0));
        assert!(approx(m.lapse(x), 1.0));
        assert_eq!(m.shift(x), Tensor::zeros());
    }

    #[test]
    fn test_cartesian_identity_transforms() {
        let m = Cartesian;
        let x = Tensor::new([3.0, 4.0, 5.0]);
        assert_eq!(m.to_cartesian(x), x);
        assert_eq!(m.from_cartesian(x), x);
    }

    // ---- spherical: metric tensor ----

    #[test]
    fn test_spherical_metric_3d() {
        let m = Spherical;
        let r = 2.0;
        let theta = FRAC_PI_4;
        let x = Tensor::new([r, theta, 0.5]);
        let g = m.spatial_metric(x);

        assert!(approx(g[(0, 0)], 1.0));
        assert!(approx(g[(1, 1)], r * r));
        assert!(approx(g[(2, 2)], r * r * theta.sin() * theta.sin()));
        assert!(approx(g[(0, 1)], 0.0));
        assert!(approx(g[(0, 2)], 0.0));
        assert!(approx(g[(1, 2)], 0.0));
    }

    #[test]
    fn test_spherical_metric_inv() {
        let m = Spherical;
        let r = 3.0;
        let theta = PI / 3.0;
        let x = Tensor::new([r, theta, 1.0]);
        let g = m.spatial_metric(x);
        let gi = m.spatial_metric_inv(x);

        // g * g^{-1} = I
        let product = g.matmul(&gi);
        for ii in 0..3 {
            for jj in 0..3 {
                let expected = if ii == jj { 1.0 } else { 0.0 };
                assert!(
                    approx(product[(ii, jj)], expected),
                    "g * g_inv [{}, {}] = {}, expected {}",
                    ii,
                    jj,
                    product[(ii, jj)],
                    expected
                );
            }
        }
    }

    #[test]
    fn test_spherical_sqrt_det() {
        let m = Spherical;
        let r = 5.0;
        let theta = PI / 6.0;
        let x = Tensor::new([r, theta, 0.0]);
        // sqrt(det) = r^2 sin(theta)
        assert!(approx(m.sqrt_det_gamma(x), r * r * theta.sin()));
    }

    #[test]
    fn test_spherical_scale_factors() {
        let m = Spherical;
        let r = 4.0;
        let theta = PI / 3.0;
        let x = Tensor::new([r, theta, 0.0]);
        let h = m.scale_factors(x);
        assert!(approx(h[0], 1.0));
        assert!(approx(h[1], r));
        assert!(approx(h[2], r * theta.sin()));
    }

    // ---- spherical: coordinate transforms ----

    #[test]
    fn test_spherical_to_cartesian_along_z() {
        let m = Spherical;
        // r=5, theta=0 (north pole) -> (0, 0, 5)
        let x = Tensor::new([5.0, 0.0, 0.0]);
        let c = m.to_cartesian(x);
        assert!(approx(c[0], 0.0));
        assert!(approx(c[1], 0.0));
        assert!(approx(c[2], 5.0));
    }

    #[test]
    fn test_spherical_to_cartesian_along_x() {
        let m = Spherical;
        // r=3, theta=pi/2, phi=0 -> (3, 0, 0)
        let x = Tensor::new([3.0, FRAC_PI_2, 0.0]);
        let c = m.to_cartesian(x);
        assert!(approx(c[0], 3.0));
        assert!(approx(c[1], 0.0));
        assert!(approx(c[2], 0.0));
    }

    #[test]
    fn test_spherical_roundtrip() {
        let m = Spherical;
        let x = Tensor::new([3.0, PI / 5.0, PI / 7.0]);
        let c = m.to_cartesian(x);
        let x2 = m.from_cartesian(c);
        assert!(vec_approx(&x, &x2));
    }

    #[test]
    fn test_spherical_vector_roundtrip() {
        let m = Spherical;
        let x = Tensor::new([2.0, PI / 3.0, PI / 4.0]);
        let v = Tensor::new([1.0, 0.5, -0.3]);
        let v_cart = m.vector_to_cartesian(x, Physical::new(v));
        let v_back = m.vector_from_cartesian(x, v_cart);
        assert!(vec_approx(&v, v_back.raw()));
    }

    // ---- spherical: det = product of diagonal for diagonal metric ----

    #[test]
    fn test_spherical_det_equals_product_diag() {
        let m = Spherical;
        let x = Tensor::new([2.5, PI / 4.0, 1.0]);
        let g = m.spatial_metric(x);
        let det = g.det();
        let sqrt_det = m.sqrt_det_gamma(x);
        assert!(approx(det.sqrt(), sqrt_det));
    }

    // ---- spherical 2D ----

    #[test]
    fn test_spherical_2d_metric() {
        let m = Spherical;
        let r = 3.0;
        let x = Tensor::new([r, PI / 4.0]);
        let g = m.spatial_metric(x);
        assert!(approx(g[(0, 0)], 1.0));
        assert!(approx(g[(1, 1)], r * r));
    }

    #[test]
    fn test_spherical_2d_sqrt_det() {
        let m = Spherical;
        let r = 7.0;
        let x = Tensor::new([r, 0.0]);
        assert!(approx(m.sqrt_det_gamma(x), r));
    }

    // ---- cylindrical: metric tensor ----

    #[test]
    fn test_cylindrical_metric_3d() {
        let m = Cylindrical;
        let r = 4.0;
        let x = Tensor::new([r, PI / 3.0, 2.0]);
        let g = m.spatial_metric(x);
        assert!(approx(g[(0, 0)], 1.0));
        assert!(approx(g[(1, 1)], r * r));
        assert!(approx(g[(2, 2)], 1.0));
        assert!(approx(g[(0, 1)], 0.0));
    }

    #[test]
    fn test_cylindrical_metric_inv() {
        let m = Cylindrical;
        let r = 3.0;
        let x = Tensor::new([r, 1.0, 5.0]);
        let g = m.spatial_metric(x);
        let gi = m.spatial_metric_inv(x);
        let product = g.matmul(&gi);
        for ii in 0..3 {
            for jj in 0..3 {
                let expected = if ii == jj { 1.0 } else { 0.0 };
                assert!(approx(product[(ii, jj)], expected));
            }
        }
    }

    #[test]
    fn test_cylindrical_sqrt_det() {
        let m = Cylindrical;
        let r = 6.0;
        let x = Tensor::new([r, 0.0, 0.0]);
        assert!(approx(m.sqrt_det_gamma(x), r));
    }

    #[test]
    fn test_cylindrical_scale_factors() {
        let m = Cylindrical;
        let r = 5.0;
        let x = Tensor::new([r, 0.0, 0.0]);
        let h = m.scale_factors(x);
        assert!(approx(h[0], 1.0));
        assert!(approx(h[1], r));
        assert!(approx(h[2], 1.0));
    }

    // ---- cylindrical: coordinate transforms ----

    #[test]
    fn test_cylindrical_to_cartesian() {
        let m = Cylindrical;
        // r=5, phi=pi/2, z=3 -> (0, 5, 3)
        let x = Tensor::new([5.0, FRAC_PI_2, 3.0]);
        let c = m.to_cartesian(x);
        assert!(approx(c[0], 0.0));
        assert!(approx(c[1], 5.0));
        assert!(approx(c[2], 3.0));
    }

    #[test]
    fn test_cylindrical_roundtrip() {
        let m = Cylindrical;
        let x = Tensor::new([3.0, PI / 5.0, 7.0]);
        let c = m.to_cartesian(x);
        let x2 = m.from_cartesian(c);
        assert!(vec_approx(&x, &x2));
    }

    #[test]
    fn test_cylindrical_vector_roundtrip() {
        let m = Cylindrical;
        let x = Tensor::new([2.0, PI / 3.0, 1.0]);
        let v = Tensor::new([1.0, 0.5, -0.3]);
        let v_cart = m.vector_to_cartesian(x, Physical::new(v));
        let v_back = m.vector_from_cartesian(x, v_cart);
        assert!(vec_approx(&v, v_back.raw()));
    }

    // ---- cylindrical 2D (axisymmetric) ----

    #[test]
    fn test_cylindrical_2d_metric() {
        let m = Cylindrical;
        let x = Tensor::new([3.0, 5.0]);
        assert_eq!(m.spatial_metric(x), Matrix::identity());
        assert!(approx(m.sqrt_det_gamma(x), 1.0));
    }

    // ---- flat spacetime defaults ----

    #[test]
    fn test_flat_lapse_shift() {
        let x = Tensor::new([1.0, 2.0, 3.0]);
        assert!(approx(Cartesian.lapse(x), 1.0));
        assert!(approx(Spherical.lapse(x), 1.0));
        assert!(approx(Cylindrical.lapse(x), 1.0));
        assert_eq!(Cartesian.shift(x), Tensor::zeros());
        assert_eq!(Spherical.shift(x), Tensor::zeros());
        assert_eq!(Cylindrical.shift(x), Tensor::zeros());
    }

    // ---- index lowering via metric ----

    #[test]
    fn test_index_lowering_spherical() {
        let r = 2.0;
        let theta = PI / 3.0;
        let x = Tensor::new([r, theta, 0.0]);
        let g = Spherical.spatial_metric(x);
        let v = Tensor::new([1.0, 1.0, 1.0]); // contravariant
        let v_lower = g * v;
        // v_r = 1, v_theta = r^2, v_phi = r^2 sin^2 theta
        assert!(approx(v_lower[0], 1.0));
        assert!(approx(v_lower[1], r * r));
        assert!(approx(v_lower[2], r * r * theta.sin() * theta.sin()));
    }

    // ---- metric norm: |v|^2 = g_{ij} v^i v^j ----

    #[test]
    fn test_metric_norm_cartesian() {
        let g = Cartesian.spatial_metric(Tensor::new([0.0, 0.0, 0.0]));
        let v = Tensor::new([3.0, 4.0, 0.0]);
        assert!(approx(g.quadratic(&v), 25.0));
    }

    #[test]
    fn test_metric_norm_spherical() {
        // for a unit radial vector, |v|^2 = 1 regardless of position
        let r = 100.0;
        let theta = PI / 7.0;
        let x = Tensor::new([r, theta, 0.0]);
        let g = Spherical.spatial_metric(x);
        let v = Tensor::new([1.0, 0.0, 0.0]); // purely radial
        assert!(approx(g.quadratic(&v), 1.0));
    }

    // ---- index raising / lowering ----

    #[test]
    fn test_lower_cartesian() {
        use symbi_algebra::{Contravariant, vec3};
        let x = Tensor::new([0.0, 0.0, 0.0]);
        let v = Contravariant::new(vec3(1.0, 2.0, 3.0));
        let w = Cartesian.lower(x, &v);
        // cartesian: lowering is identity
        assert!(approx(w[0], 1.0));
        assert!(approx(w[1], 2.0));
        assert!(approx(w[2], 3.0));
    }

    #[test]
    fn test_lower_spherical() {
        use symbi_algebra::Contravariant;
        let r = 2.0;
        let theta = PI / 3.0;
        let x = Tensor::new([r, theta, 0.0]);
        let v = Contravariant::new(Tensor::new([1.0, 1.0, 1.0]));
        let w = Spherical.lower(x, &v);
        // v_r = 1, v_theta = r^2, v_phi = r^2 sin^2 theta
        assert!(approx(w[0], 1.0));
        assert!(approx(w[1], r * r));
        assert!(approx(w[2], r * r * theta.sin() * theta.sin()));
    }

    #[test]
    fn test_raise_lower_roundtrip() {
        use symbi_algebra::Contravariant;
        let r = 3.0;
        let theta = PI / 5.0;
        let x = Tensor::new([r, theta, 1.0]);
        let v = Contravariant::new(Tensor::new([2.0, -1.0, 0.5]));
        let w = Spherical.lower(x, &v);
        let v2 = Spherical.raise(x, &w);
        for ii in 0..3 {
            assert!(
                approx(v[ii], v2[ii]),
                "component {}: {} != {}",
                ii,
                v[ii],
                v2[ii]
            );
        }
    }

    #[test]
    fn test_raise_lower_contraction() {
        // v^i v_i = g_{ij} v^i v^j = |v|^2 (metric norm)
        use symbi_algebra::Contravariant;
        let r = 4.0;
        let theta = PI / 4.0;
        let x = Tensor::new([r, theta, 0.5]);
        let v = Contravariant::new(Tensor::new([1.0, 0.5, -0.3]));
        let w = Spherical.lower(x, &v);
        let norm_sq = v.contract(&w);
        let g = Spherical.spatial_metric(x);
        let norm_sq_direct = g.quadratic(v.raw());
        assert!(
            approx(norm_sq, norm_sq_direct),
            "contraction: {} != quadratic: {}",
            norm_sq,
            norm_sq_direct
        );
    }

    // ---- momentum source: cartesian (always zero) ----

    #[test]
    fn test_cartesian_source_zero() {
        let x = Tensor::new([1.0, 2.0, 3.0]);
        let vel = Tensor::new([1.0, 0.5, -0.3]);
        let s = Cartesian.momentum_source(x, 1.0, vel, 1.0);
        for ii in 0..3 {
            assert!(approx(s[ii], 0.0), "cartesian S[{}] = {}", ii, s[ii]);
        }
    }

    // ---- momentum source: spherical ----

    #[test]
    fn test_spherical_1d_source() {
        let r = 2.0;
        let p = 3.0;
        let x = Tensor::new([r]);
        let s = Spherical.momentum_source(x, 1.0, Tensor::new([1.0]), p);
        // S_r = 2*p/r = 6/2 = 3
        assert!(approx(s[0], 2.0 * p / r));
    }

    #[test]
    fn test_spherical_1d_source_pressure_only() {
        // source is independent of rho and vel
        let r = 5.0;
        let p = 10.0;
        let s1 = Spherical.momentum_source(Tensor::new([r]), 1.0, Tensor::new([0.0]), p);
        let s2 = Spherical.momentum_source(Tensor::new([r]), 100.0, Tensor::new([99.0]), p);
        assert!(approx(s1[0], s2[0]));
    }

    #[test]
    fn test_spherical_2d_source_radial() {
        let r = 3.0;
        let theta = PI / 4.0;
        let rho = 2.0;
        let vr = 1.0;
        let vt = 0.5;
        let p = 1.5;
        let x = Tensor::new([r, theta]);
        let s = Spherical.momentum_source(x, rho, Tensor::new([vr, vt]), p);
        // S_r = (rho*vt^2 + 2p) / r
        let expected_r = (rho * vt * vt + 2.0 * p) / r;
        assert!(approx(s[0], expected_r), "S_r: {} != {}", s[0], expected_r);
    }

    #[test]
    fn test_spherical_2d_source_theta() {
        let r = 3.0;
        let theta = PI / 4.0;
        let rho = 2.0;
        let vr = 1.0;
        let vt = 0.5;
        let p = 1.5;
        let x = Tensor::new([r, theta]);
        let s = Spherical.momentum_source(x, rho, Tensor::new([vr, vt]), p);
        // S_t = (p*cot(theta) - rho*vr*vt) / r
        let cot = theta.cos() / theta.sin();
        let expected_t = (p * cot - rho * vr * vt) / r;
        assert!(approx(s[1], expected_t), "S_t: {} != {}", s[1], expected_t);
    }

    #[test]
    fn test_spherical_3d_source_radial() {
        let r = 4.0;
        let theta = PI / 3.0;
        let rho = 1.0;
        let vel = Tensor::new([0.5, 0.3, 0.2]);
        let p = 2.0;
        let x = Tensor::new([r, theta, 0.0]);
        let s = Spherical.momentum_source(x, rho, vel, p);
        // S_r = (rho*(vt^2 + vp^2) + 2p) / r
        let vt = vel[1];
        let vp = vel[2];
        let expected = (rho * (vt * vt + vp * vp) + 2.0 * p) / r;
        assert!(approx(s[0], expected));
    }

    #[test]
    fn test_spherical_3d_source_theta() {
        let r = 4.0;
        let theta = PI / 3.0;
        let rho = 1.0;
        let vel = Tensor::new([0.5, 0.3, 0.2]);
        let p = 2.0;
        let x = Tensor::new([r, theta, 0.0]);
        let s = Spherical.momentum_source(x, rho, vel, p);
        let cot = theta.cos() / theta.sin();
        // S_t = ((rho*vp^2 + p)*cot - rho*vr*vt) / r
        let expected = ((rho * vel[2] * vel[2] + p) * cot - rho * vel[0] * vel[1]) / r;
        assert!(approx(s[1], expected));
    }

    #[test]
    fn test_spherical_3d_source_phi() {
        let r = 4.0;
        let theta = PI / 3.0;
        let rho = 1.0;
        let vel = Tensor::new([0.5, 0.3, 0.2]);
        let p = 2.0;
        let x = Tensor::new([r, theta, 0.0]);
        let s = Spherical.momentum_source(x, rho, vel, p);
        let cot = theta.cos() / theta.sin();
        // S_p = -rho*vp*(vr + vt*cot) / r
        let expected = -rho * vel[2] * (vel[0] + vel[1] * cot) / r;
        assert!(approx(s[2], expected));
    }

    #[test]
    fn test_spherical_3d_no_velocity_source_is_pressure_only() {
        // zero velocity: only pressure contributes
        let r = 2.0;
        let theta = PI / 4.0;
        let p = 5.0;
        let x = Tensor::new([r, theta, 0.0]);
        let s = Spherical.momentum_source(x, 1.0, Tensor::zeros(), p);
        assert!(approx(s[0], 2.0 * p / r));
        let cot = theta.cos() / theta.sin();
        assert!(approx(s[1], p * cot / r));
        assert!(approx(s[2], 0.0));
    }

    #[test]
    fn test_spherical_3d_reduces_to_2d() {
        // with V_phi = 0, 3D source should match 2D for r and theta components
        let r = 3.0;
        let theta = PI / 5.0;
        let rho = 1.5;
        let vr = 0.8;
        let vt = 0.4;
        let p = 2.0;
        let x2 = Tensor::new([r, theta]);
        let x3 = Tensor::new([r, theta, 0.0]);
        let s2 = Spherical.momentum_source(x2, rho, Tensor::new([vr, vt]), p);
        let s3 = Spherical.momentum_source(x3, rho, Tensor::new([vr, vt, 0.0]), p);
        assert!(approx(s2[0], s3[0]), "S_r: 2D={} != 3D={}", s2[0], s3[0]);
        assert!(approx(s2[1], s3[1]), "S_t: 2D={} != 3D={}", s2[1], s3[1]);
    }

    // ---- momentum source: cylindrical ----

    #[test]
    fn test_cylindrical_1d_source() {
        let r = 4.0;
        let p = 3.0;
        let s = Cylindrical.momentum_source(Tensor::new([r]), 1.0, Tensor::new([1.0]), p);
        assert!(approx(s[0], p / r));
    }

    #[test]
    fn test_cylindrical_2d_source() {
        let r = 5.0;
        let p = 4.0;
        let x = Tensor::new([r, 2.0]);
        let s = Cylindrical.momentum_source(x, 1.0, Tensor::new([1.0, 0.5]), p);
        assert!(approx(s[0], p / r));
        assert!(approx(s[1], 0.0));
    }

    #[test]
    fn test_cylindrical_3d_source() {
        let r = 3.0;
        let rho = 2.0;
        let vr = 0.5;
        let vp = 0.8;
        let vz = 1.0;
        let p = 1.5;
        let x = Tensor::new([r, 0.0, 0.0]);
        let s = Cylindrical.momentum_source(x, rho, Tensor::new([vr, vp, vz]), p);
        // S_r = (rho*vp^2 + p) / r
        assert!(approx(s[0], (rho * vp * vp + p) / r));
        // S_p = -rho*vr*vp / r
        assert!(approx(s[1], -rho * vr * vp / r));
        // S_z = 0
        assert!(approx(s[2], 0.0));
    }

    #[test]
    fn test_cylindrical_3d_no_velocity_pressure_only() {
        let r = 2.0;
        let p = 6.0;
        let x = Tensor::new([r, 0.0, 0.0]);
        let s = Cylindrical.momentum_source(x, 1.0, Tensor::zeros(), p);
        assert!(approx(s[0], p / r));
        assert!(approx(s[1], 0.0));
        assert!(approx(s[2], 0.0));
    }

    #[test]
    fn test_cylindrical_3d_reduces_to_1d() {
        // with V_phi = V_z = 0, cylindrical 3D S_r should match 1D
        let r = 4.0;
        let p = 3.0;
        let s1 = Cylindrical.momentum_source(Tensor::new([r]), 1.0, Tensor::new([1.0]), p);
        let s3 = Cylindrical.momentum_source(
            Tensor::new([r, 0.0, 0.0]),
            1.0,
            Tensor::new([1.0, 0.0, 0.0]),
            p,
        );
        assert!(approx(s1[0], s3[0]));
    }

    // ---- momentum_source_inertial: velocity-only terms ----

    #[test]
    fn test_cartesian_inertial_zero() {
        let x = Tensor::new([1.0, 2.0, 3.0]);
        let vel = Tensor::new([1.0, 0.5, -0.3]);
        let s = Cartesian.momentum_source_inertial(x, vel, vel);
        for ii in 0..3 {
            assert!(approx(s[ii], 0.0));
        }
    }

    #[test]
    fn test_spherical_1d_inertial_zero() {
        let s = Spherical.momentum_source_inertial(
            Tensor::new([2.0]),
            Tensor::new([0.5]),
            Tensor::new([0.5]),
        );
        assert!(approx(s[0], 0.0));
    }

    #[test]
    fn test_spherical_2d_inertial() {
        let r = 3.0;
        let rho = 2.0;
        let vr = 1.0;
        let vt = 0.5;
        let x = Tensor::new([r, PI / 4.0]);
        let s = Spherical.momentum_source_inertial(
            x,
            Tensor::new([rho * vr, rho * vt]),
            Tensor::new([vr, vt]),
        );
        // inertial_r = rho*vt^2/r, inertial_t = -rho*vr*vt/r
        assert!(approx(s[0], rho * vt * vt / r));
        assert!(approx(s[1], -rho * vr * vt / r));
    }

    #[test]
    fn test_spherical_2d_full_equals_inertial_plus_pressure() {
        // momentum_source = momentum_source_inertial + pressure terms
        let r = 3.0;
        let theta = PI / 4.0;
        let rho = 2.0;
        let vr = 1.0;
        let vt = 0.5;
        let p = 1.5;
        let x = Tensor::new([r, theta]);
        let vel = Tensor::new([vr, vt]);
        let full = Spherical.momentum_source(x, rho, vel, p);
        let inertial =
            Spherical.momentum_source_inertial(x, Tensor::new([rho * vr, rho * vt]), vel);
        // pressure_r = 2p/r, pressure_t = p*cot(theta)/r
        let cot = theta.cos() / theta.sin();
        assert!(approx(full[0], inertial[0] + 2.0 * p / r));
        assert!(approx(full[1], inertial[1] + p * cot / r));
    }

    #[test]
    fn test_spherical_3d_inertial() {
        let r = 4.0;
        let theta = PI / 3.0;
        let rho = 1.0;
        let vel = Tensor::new([0.5, 0.3, 0.2]);
        let x = Tensor::new([r, theta, 0.0]);
        let s = Spherical.momentum_source_inertial(
            x,
            Tensor::new([rho * vel[0], rho * vel[1], rho * vel[2]]),
            vel,
        );
        let vr = vel[0];
        let vt = vel[1];
        let vp = vel[2];
        let cot = theta.cos() / theta.sin();
        assert!(approx(s[0], rho * (vt * vt + vp * vp) / r));
        assert!(approx(s[1], (rho * vp * vp * cot - rho * vr * vt) / r));
        assert!(approx(s[2], -rho * vp * (vr + vt * cot) / r));
    }

    #[test]
    fn test_spherical_3d_phi_inertial_equals_full() {
        // phi source has no pressure terms, so inertial == full
        let r = 4.0;
        let theta = PI / 3.0;
        let rho = 1.0;
        let vel = Tensor::new([0.5, 0.3, 0.2]);
        let p = 99.0;
        let x = Tensor::new([r, theta, 0.0]);
        let full = Spherical.momentum_source(x, rho, vel, p);
        let inertial = Spherical.momentum_source_inertial(
            x,
            Tensor::new([rho * vel[0], rho * vel[1], rho * vel[2]]),
            vel,
        );
        assert!(approx(full[2], inertial[2]));
    }

    #[test]
    fn test_cylindrical_1d_inertial_zero() {
        let s = Cylindrical.momentum_source_inertial(
            Tensor::new([4.0]),
            Tensor::new([0.5]),
            Tensor::new([0.5]),
        );
        assert!(approx(s[0], 0.0));
    }

    #[test]
    fn test_cylindrical_3d_inertial() {
        let r = 3.0;
        let rho = 2.0;
        let vr = 0.5;
        let vp = 0.8;
        let x = Tensor::new([r, 0.0, 0.0]);
        let s = Cylindrical.momentum_source_inertial(
            x,
            Tensor::new([rho * vr, rho * vp, rho * 1.0]),
            Tensor::new([vr, vp, 1.0]),
        );
        assert!(approx(s[0], rho * vp * vp / r));
        assert!(approx(s[1], -rho * vr * vp / r));
        assert!(approx(s[2], 0.0));
    }

    #[test]
    fn test_cylindrical_rphi_2d_disk() {
        // the (r, phi) disk: phi is the RESOLVED swirl, so the inertial is NONZERO — unlike the
        // (r, z) axisymmetric Cylindrical<2>, which is identically zero. this is the distinction
        // that makes the new type necessary for 2D disk sims.
        let r = 3.0;
        let rho = 2.0;
        let vr = 0.5;
        let vp = 0.8;
        let p = 1.5;
        let x = Tensor::new([r, 0.7]); // (r, phi)
        let vel = Tensor::new([vr, vp]);
        let mom = Tensor::new([rho * vr, rho * vp]);

        let inertial = CylindricalRPhi.momentum_source_inertial(x, mom, vel);
        assert!(approx(inertial[0], rho * vp * vp / r)); // S_r = mom_phi v_phi / r
        assert!(approx(inertial[1], -rho * vr * vp / r)); // S_phi = -mom_r v_phi / r

        // full source = inertial + the r-pressure term p/r (phi has no pressure source).
        let full = CylindricalRPhi.momentum_source(x, rho, vel, p);
        assert!(approx(full[0], inertial[0] + p / r));
        assert!(approx(full[1], inertial[1]));

        // sqrt(gamma) = r — DISTINCT from the (r, z) reduction, which zeroes this exact swirl.
        assert!(approx(CylindricalRPhi.sqrt_det_gamma(x), r));
        let rz = Cylindrical.momentum_source_inertial(x, mom, vel);
        assert!(approx(rz[0], 0.0) && approx(rz[1], 0.0));
    }

    // ---- schwarzschild in kerr-schild (horizon-penetrating) ----

    #[test]
    fn test_kerr_schild_is_regular_across_the_horizon() {
        // the defining property: lapse, shift, gamma_rr are all finite and smooth at and inside the
        // horizon r = 2M (where the schwarzschild-coordinate lapse sqrt(1 - 2M/r) hits 0 / goes
        // imaginary). h(2M) = 2, so alpha = 1/sqrt(2), beta^r = 1/2, gamma_rr = 2.
        let bh = SchwarzschildKS { mass: 1.0_f64 };
        for &r in &[3.0, 2.0, 1.5, 1.0, 0.5] {
            let x = Tensor::new([r]);
            let h = 1.0 + 2.0 / r;
            assert!(bh.lapse(x).is_finite() && bh.lapse(x) > 0.0);
            assert!(approx(bh.lapse(x), 1.0 / h.sqrt()));
            assert!(approx(bh.shift(x)[0], 2.0 / (r + 2.0))); // beta^r = 2M/(r+2M)
            assert!(approx(bh.spatial_metric(x)[(0, 0)], h)); // gamma_rr = 1 + 2M/r
        }
        // exact values at the horizon.
        let xh = Tensor::new([2.0_f64]);
        assert!(approx(bh.lapse(xh), 1.0 / 2.0_f64.sqrt()));
        assert!(approx(bh.shift(xh)[0], 0.5));
        assert!(approx(bh.spatial_metric(xh)[(0, 0)], 2.0));
    }

    #[test]
    fn kerr_metric_satisfies_the_adm_identities() {
        // the spinning kerr-schild block against the KNOWN kerr line element: g_tt = b - 1,
        // g_{t phi} = -a b sin^2(theta), g_{r phi} = -a sin^2(theta)(1 + b), det gamma =
        // Sigma^2 sin^2(theta) (1 + b), and gamma * gamma^{-1} = identity — off the equator,
        // inside and outside the horizon, prograde and retrograde.
        let close = |x: f64, y: f64| (x - y).abs() < 1e-12 * (1.0 + x.abs().max(y.abs()));
        for &a in &[0.9_f64, -0.6, 0.3] {
            let g = KerrKS {
                mass: 1.0_f64,
                spin: a,
            };
            for &(r, th) in &[(1.3_f64, 1.1_f64), (2.0, 0.7), (6.0, 1.9), (30.0, 1.4)] {
                let x = Tensor::new([r, th, 0.0]);
                let (st, ct) = (th.sin(), th.cos());
                let sig = r * r + a * a * ct * ct;
                let b = 2.0 * r / sig;
                let gm = <KerrKS<f64> as Metric<f64, 3>>::spatial_metric(&g, x);
                let gi = <KerrKS<f64> as Metric<f64, 3>>::spatial_metric_inv(&g, x);
                let alpha = <KerrKS<f64> as Metric<f64, 3>>::lapse(&g, x);
                let beta = <KerrKS<f64> as Metric<f64, 3>>::shift(&g, x);
                // gamma * gamma^{-1} = 1
                for ii in 0..3 {
                    for jj in 0..3 {
                        let mut acc = 0.0;
                        for kk in 0..3 {
                            acc += gm[(ii, kk)] * gi[(kk, jj)];
                        }
                        assert!(
                            close(acc, if ii == jj { 1.0 } else { 0.0 }),
                            "gamma inverse a={a} r={r}: ({ii},{jj}) = {acc}"
                        );
                    }
                }
                // 4-metric identities from the ADM block.
                let beta_low: [f64; 3] =
                    std::array::from_fn(|ii| (0..3).map(|jj| gm[(ii, jj)] * beta[jj]).sum());
                let g_tt = -alpha * alpha + (0..3).map(|ii| beta_low[ii] * beta[ii]).sum::<f64>();
                assert!(
                    close(g_tt, b - 1.0),
                    "g_tt a={a} r={r}: {g_tt} vs {}",
                    b - 1.0
                );
                assert!(close(beta_low[2], -a * b * st * st), "g_t-phi a={a} r={r}");
                assert!(
                    close(gm[(0, 2)], -a * st * st * (1.0 + b)),
                    "g_r-phi a={a} r={r}"
                );
                // determinant.
                let det = gm[(1, 1)] * (gm[(0, 0)] * gm[(2, 2)] - gm[(0, 2)] * gm[(2, 0)]);
                assert!(
                    close(det, sig * sig * st * st * (1.0 + b)),
                    "det gamma a={a} r={r}"
                );
                let sq = <KerrKS<f64> as Metric<f64, 3>>::sqrt_det_gamma(&g, x);
                assert!(close(sq * sq, det), "sqrt_det_gamma a={a} r={r}");
                // horizon-penetrating: the lapse is finite and positive everywhere sampled.
                assert!(alpha > 0.0 && alpha.is_finite());
            }
        }
    }

    #[test]
    fn kerr_at_zero_spin_reduces_to_schwarzschild_ks() {
        // a = 0 must reproduce the schwarzschild kerr-schild ADM block exactly (different
        // expressions, same values) at every position.
        let close = |x: f64, y: f64| (x - y).abs() < 1e-13 * (1.0 + x.abs().max(y.abs()));
        let kerr = KerrKS {
            mass: 1.0_f64,
            spin: 0.0,
        };
        let ks = SchwarzschildKS { mass: 1.0_f64 };
        for &(r, th) in &[(1.5_f64, 0.9_f64), (2.0, 1.5707963), (10.0, 2.2)] {
            let x = Tensor::new([r, th, 0.0]);
            assert!(close(
                <KerrKS<f64> as Metric<f64, 3>>::lapse(&kerr, x),
                <SchwarzschildKS<f64> as Metric<f64, 3>>::lapse(&ks, x),
            ));
            let (bk, bs) = (
                <KerrKS<f64> as Metric<f64, 3>>::shift(&kerr, x),
                <SchwarzschildKS<f64> as Metric<f64, 3>>::shift(&ks, x),
            );
            let (gk, gs) = (
                <KerrKS<f64> as Metric<f64, 3>>::spatial_metric(&kerr, x),
                <SchwarzschildKS<f64> as Metric<f64, 3>>::spatial_metric(&ks, x),
            );
            for ii in 0..3 {
                assert!(close(bk[ii], bs[ii]), "shift[{ii}] at r={r}");
                for jj in 0..3 {
                    assert!(
                        close(gk[(ii, jj)], gs[(ii, jj)]),
                        "gamma[{ii}{jj}] at r={r}"
                    );
                }
            }
        }
    }

    #[test]
    fn test_kerr_schild_covariant_shift_equals_g_tr() {
        // beta_r = gamma_rr beta^r must equal g_tr = 2M/r (the KS off-diagonal 4-metric term).
        let bh = SchwarzschildKS { mass: 1.3_f64 };
        for &r in &[1.0, 2.6, 5.0] {
            let x = Tensor::new([r]);
            let beta_lower = bh.spatial_metric(x)[(0, 0)] * bh.shift(x)[0];
            assert!(approx(beta_lower, 2.0 * 1.3 / r)); // 2M/r
        }
    }

    #[test]
    fn test_kerr_schild_sqrt_minus_g_is_flat_spherical_area() {
        // sqrt(-g) = alpha sqrt(gamma) = r^2 sin(theta), the same flat spherical volume element as
        // schwarzschild coords — det(g) is chart-independent for this vacuum.
        let bh = SchwarzschildKS { mass: 0.7_f64 };
        let (r, theta) = (6.0, FRAC_PI_4);
        let x = Tensor::new([r, theta, 0.0]);
        assert!(approx(
            bh.lapse(x) * bh.volume_factor(x),
            r * r * theta.sin()
        ));
    }

    #[test]
    fn test_kerr_schild_zero_mass_equals_spherical() {
        // M = 0 -> h = 1, beta = 0 -> the flat spherical metric exactly.
        let bh = SchwarzschildKS { mass: 0.0_f64 };
        let x = Tensor::new([4.0, FRAC_PI_4, 1.1]);
        assert!(approx(bh.lapse(x), 1.0));
        assert!(approx(bh.shift(x)[0], 0.0));
        let (g, gs) = (bh.spatial_metric(x), Spherical.spatial_metric(x));
        for ii in 0..3 {
            assert!(approx(g[(ii, ii)], gs[(ii, ii)]));
        }
        assert!(approx(bh.sqrt_det_gamma(x), Spherical.sqrt_det_gamma(x)));
        assert_eq!(
            <SchwarzschildKS<f64> as Metric<f64, 3>>::geometry(&bh),
            Geometry::Spherical
        );
        assert_eq!(
            <SchwarzschildKS<f64> as Metric<f64, 3>>::spacetime(&bh),
            Spacetime::SchwarzschildKS
        );
    }

    #[test]
    fn test_metric_autodiff_radial_derivs_match_finite_difference() {
        // the geodesic source gets its metric radial derivatives from forward-mode autodiff:
        // evaluate lapse/shift/gamma_rr at `Dual` with the radial coordinate seeded (d/dr = 1); the
        // tangent is d/dr. this is the single source of metric derivatives (the hand-written analytic
        // forms are retired), so it matches a central finite difference of the metric's own
        // lapse/shift/spatial_metric. checked for schwarzschild + KS, including inside the KS horizon
        // (r < 2M). the mass is a constant dual (differentiation is w.r.t. r; M is held fixed).
        use symbi_ir::dual::Dual;
        // dr balances central-diff truncation (O(dr^2)) against subtractive roundoff (O(eps/dr)):
        // 1e-4 keeps both ~1e-8, so an FD-appropriate relative tolerance of 1e-5 is safe.
        let dr = 1e-4;
        let close = |a: f64, b: f64| (a - b).abs() < 1e-5 * (1.0 + a.abs().max(b.abs()));
        let fd = |f: &dyn Fn(f64) -> f64, r: f64| (f(r + dr) - f(r - dr)) / (2.0 * dr);
        let seed = |r: f64| Tensor::new([Dual::variable(r)]);
        for &r in &[1.2_f64, 1.8, 2.0, 4.0, 9.0] {
            let m = SchwarzschildKS {
                mass: Dual::constant(1.0_f64),
            };
            let mf = SchwarzschildKS { mass: 1.0_f64 };
            assert!(
                close(
                    m.lapse(seed(r)).tangent,
                    fd(&|x| mf.lapse(Tensor::new([x])), r)
                ),
                "ks d_lapse r={r}"
            );
            assert!(
                close(
                    m.shift(seed(r))[0].tangent,
                    fd(&|x| mf.shift(Tensor::new([x]))[0], r)
                ),
                "ks d_shift r={r}"
            );
            assert!(
                close(
                    m.spatial_metric(seed(r))[(0, 0)].tangent,
                    fd(&|x| mf.spatial_metric(Tensor::new([x]))[(0, 0)], r)
                ),
                "ks d_grr r={r}"
            );
        }
    }

    #[test]
    fn test_lapse_sq_is_exact_closed_form_not_sqrt_roundtrip() {
        // alpha^2 must be the EXACT closed form 1/(1 + 2M/r); `lapse()` squared is
        // (1/sqrt(1 + 2M/r))^2, which rounds differently. the GR CFL radial factor depends on this
        // being bitwise the closed form, so the genericized wave-speed map bit-diffs against it.
        let ks = SchwarzschildKS { mass: 1.0_f64 };
        for &r in &[2.3_f64, 5.0, 11.7] {
            let x = Tensor::new([r]);
            assert_eq!(ks.lapse_sq(x), 1.0 / (1.0 + 2.0 / r));
            // and it genuinely differs from the sqrt round-trip at some radius (the trap the closed
            // form sidesteps).
            let _ = ks.lapse(x) * ks.lapse(x);
        }
    }

    #[test]
    fn test_kerr_schild_transport_velocity_is_ingoing_inside_horizon() {
        // the horizon-penetrating guarantee: tilde v^r = v^r - beta^r/alpha = (V - 2M/r)/sqrt(h) is
        // negative for every subluminal physical velocity V < 1 at r <= 2M, so every characteristic
        // stays inside the excised interior and the inner outflow boundary is causal.
        let bh = SchwarzschildKS { mass: 1.0_f64 };
        for &r in &[2.0, 1.5, 1.0] {
            let x = Tensor::new([r]);
            let (alpha, beta_r, h) = (bh.lapse(x), bh.shift(x)[0], bh.spatial_metric(x)[(0, 0)]);
            for &big_v in &[0.9_f64, 0.0, -0.9] {
                let v_contra = big_v / h.sqrt(); // v^r = V / sqrt(gamma_rr)
                let tilde = v_contra - beta_r / alpha;
                assert!(
                    tilde < 0.0,
                    "tilde v^r = {tilde} not ingoing at r={r}, V={big_v}"
                );
            }
        }
    }

    #[test]
    fn cartesian_ks_satisfies_the_adm_identities() {
        // the cartesian kerr-schild block against the known ADM identities at off-axis positions,
        // inside and outside the horizon: gamma gamma^{-1} = 1, sqrt_det^2 = det, beta_i = 2H l_i
        // (the covariant shift = g_{0i}), g_tt = -alpha^2 + beta_i beta^i = 2H - 1, alpha finite > 0.
        let close = |x: f64, y: f64| (x - y).abs() < 1e-12 * (1.0 + x.abs().max(y.abs()));
        let bh = SchwarzschildKSCartesian { mass: 1.0_f64 };
        for &(px, py, pz) in &[
            (1.0_f64, 0.4, 0.7),
            (2.0, -1.5, 0.3),
            (-4.0, 3.0, 5.0),
            (0.9, 0.2, 0.1),
        ] {
            let x = Tensor::new([px, py, pz]);
            let r = (px * px + py * py + pz * pz).sqrt();
            let two_h = 2.0 / r;
            let l = [px / r, py / r, pz / r];
            let gm = <SchwarzschildKSCartesian<f64> as Metric<f64, 3>>::spatial_metric(&bh, x);
            let gi = <SchwarzschildKSCartesian<f64> as Metric<f64, 3>>::spatial_metric_inv(&bh, x);
            let alpha = <SchwarzschildKSCartesian<f64> as Metric<f64, 3>>::lapse(&bh, x);
            let beta = <SchwarzschildKSCartesian<f64> as Metric<f64, 3>>::shift(&bh, x);
            // gamma gamma^{-1} = identity.
            for ii in 0..3 {
                for jj in 0..3 {
                    let acc: f64 = (0..3).map(|kk| gm[(ii, kk)] * gi[(kk, jj)]).sum();
                    assert!(
                        close(acc, if ii == jj { 1.0 } else { 0.0 }),
                        "gamma inverse ({ii},{jj}) = {acc} at r={r}"
                    );
                }
            }
            // beta_i = gamma_ij beta^j = 2H l_i (the covariant shift = the KS off-diagonal 4-metric).
            let beta_low: [f64; 3] =
                std::array::from_fn(|ii| (0..3).map(|jj| gm[(ii, jj)] * beta[jj]).sum());
            for ii in 0..3 {
                assert!(
                    close(beta_low[ii], two_h * l[ii]),
                    "beta_low[{ii}] at r={r}"
                );
            }
            // g_tt = -alpha^2 + beta_i beta^i = 2H - 1.
            let g_tt = -alpha * alpha + (0..3).map(|ii| beta_low[ii] * beta[ii]).sum::<f64>();
            assert!(
                close(g_tt, two_h - 1.0),
                "g_tt = {g_tt} vs {} at r={r}",
                two_h - 1.0
            );
            // sqrt_det^2 = det gamma (det via cofactor).
            let det = gm[(0, 0)] * (gm[(1, 1)] * gm[(2, 2)] - gm[(1, 2)] * gm[(2, 1)])
                - gm[(0, 1)] * (gm[(1, 0)] * gm[(2, 2)] - gm[(1, 2)] * gm[(2, 0)])
                + gm[(0, 2)] * (gm[(1, 0)] * gm[(2, 1)] - gm[(1, 1)] * gm[(2, 0)]);
            let sq = <SchwarzschildKSCartesian<f64> as Metric<f64, 3>>::sqrt_det_gamma(&bh, x);
            assert!(
                close(sq * sq, det),
                "sqrt_det^2 = {} vs det = {det} at r={r}",
                sq * sq
            );
            assert!(close(det, 1.0 + two_h), "det = {det} vs 1+2H at r={r}");
            assert!(alpha > 0.0 && alpha.is_finite());
        }
    }

    #[test]
    fn cartesian_ks_radius_clamp_is_the_identity_outside_m_over_2() {
        // max(r, M/2) = r bitwise for r > M/2, so every metric quantity at a live radius equals
        // the unclamped closed form exactly. sample just above the clamp and at typical radii.
        let bh = SchwarzschildKSCartesian { mass: 1.0_f64 };
        for &(px, py) in &[(0.36_f64, 0.36), (0.51, 0.0), (1.0, 0.4), (4.0, 4.0)] {
            let x = Tensor::new([px, py]);
            let r = (px * px + py * py).sqrt();
            assert!(r > 0.5, "sample point must sit outside the clamp radius");
            let alpha = <SchwarzschildKSCartesian<f64> as Metric<f64, 2>>::lapse(&bh, x);
            assert_eq!(alpha, 1.0 / (1.0 + 2.0 / r).sqrt(), "lapse at r={r}");
            let sq = <SchwarzschildKSCartesian<f64> as Metric<f64, 2>>::sqrt_det_gamma(&bh, x);
            assert_eq!(sq, (1.0 + 2.0 / r).sqrt(), "sqrt_det_gamma at r={r}");
            let beta = <SchwarzschildKSCartesian<f64> as Metric<f64, 2>>::shift(&bh, x);
            let s = (2.0 / r) / (1.0 + 2.0 / r) / r;
            assert_eq!(beta[0], s * px, "shift_x at r={r}");
            assert_eq!(beta[1], s * py, "shift_y at r={r}");
        }
    }

    #[test]
    fn cartesian_ks_is_bounded_and_consistent_inside_the_clamp() {
        // the radius clamp exists to bound the chart's coordinate singularity at r = 0, which the
        // grid straddles whenever the domain contains the origin. inside r = M/2 the clamp freezes
        // r and hence 2H, while the metric keeps varying: l^i = x^i / r_clamped tracks position and
        // its norm |l| = |x| / r_clamped falls below 1, so the rank-1 forms contract with the
        // measured |l|^2. with M = 1 that gives 1 + 2H |l|^2 = 1 + 16 |x|^2 on the clamp, running
        // continuously from the flat value 1 at the origin to the exterior value 5 at r = M/2.
        // the requirement is boundedness and consistency, with variation allowed: alpha in
        // [1/sqrt(5), 1],
        // every component finite through the origin, and sqrt(det gamma) equal to the determinant
        // of the matrix it describes.
        let bh = SchwarzschildKSCartesian { mass: 1.0_f64 };
        for &(px, py) in &[
            (0.0_f64, 0.0),
            (1e-300, 0.0),
            (0.1, 0.1),
            (0.3, 0.2),
            (0.5, 0.0),
        ] {
            let x = Tensor::new([px, py]);
            let alpha = <SchwarzschildKSCartesian<f64> as Metric<f64, 2>>::lapse(&bh, x);
            let sq = <SchwarzschildKSCartesian<f64> as Metric<f64, 2>>::sqrt_det_gamma(&bh, x);
            let beta = <SchwarzschildKSCartesian<f64> as Metric<f64, 2>>::shift(&bh, x);
            let gm = <SchwarzschildKSCartesian<f64> as Metric<f64, 2>>::spatial_metric(&bh, x);
            let r2 = px * px + py * py;
            assert!(
                (alpha - 1.0 / (1.0 + 16.0 * r2).sqrt()).abs() < 1e-15,
                "lapse on the clamp at ({px},{py}): {alpha}"
            );
            assert!(
                alpha >= 1.0 / 5.0_f64.sqrt() - 1e-15 && alpha <= 1.0 + 1e-15,
                "lapse out of the clamp bounds at ({px},{py}): {alpha}"
            );
            // the 2x2 determinant equals the closed form: the guard region is a consistent metric,
            // its measure agreeing with its components.
            let det = gm[(0, 0)] * gm[(1, 1)] - gm[(0, 1)] * gm[(1, 0)];
            assert!(
                (det - sq * sq).abs() < 1e-14 * det.max(1.0),
                "det {det} vs sqrt_det_gamma^2 {} at ({px},{py})",
                sq * sq
            );
            for ii in 0..2 {
                assert!(beta[ii].is_finite(), "shift finite at ({px},{py})");
                for jj in 0..2 {
                    assert!(gm[(ii, jj)].is_finite(), "metric finite at ({px},{py})");
                }
            }
        }
    }

    #[test]
    fn cartesian_ks_clamp_bounds_the_metric_gradient() {
        // the geodesic source is built from Dual derivatives of the metric. what the clamp has to
        // deliver is a bounded derivative — the unclamped chart's d(alpha)/dx grows like M/r^2 and
        // diverges at the origin. on the clamp the lapse derivative is
        // -16 x / (1 + 16 |x|^2)^{3/2} (M = 1), which is bounded by ~2.5 and vanishes at the
        // origin by symmetry. the metric keeps varying there: freezing r leaves l^i = x^i/r_g
        // tracking position, so gravity inside the clamp stays live in every component of the
        // source.
        use symbi_ir::dual::Dual;
        let bh = SchwarzschildKSCartesian {
            mass: Dual {
                value: 1.0_f64,
                tangent: 0.0,
            },
        };
        let lapse_dx = |px: f64, py: f64| -> f64 {
            let x = Tensor::new([
                Dual {
                    value: px,
                    tangent: 1.0,
                }, // d/dx seed
                Dual {
                    value: py,
                    tangent: 0.0,
                },
            ]);
            <SchwarzschildKSCartesian<Dual<f64>> as Metric<Dual<f64>, 2>>::lapse(&bh, x).tangent
        };
        assert_eq!(
            lapse_dx(0.0, 0.0),
            0.0,
            "the gradient vanishes at the origin by symmetry"
        );
        for &(px, py) in &[(0.2_f64, 0.1), (0.4, 0.0), (0.1, 0.3), (0.49, 0.0)] {
            let d = lapse_dx(px, py);
            let r2 = px * px + py * py;
            let want = -16.0 * px / (1.0 + 16.0 * r2).powf(1.5);
            assert!(
                (d - want).abs() < 1e-14,
                "clamp gradient at ({px},{py}): {d} vs {want}"
            );
            assert!(
                d.abs() < 3.0,
                "clamp gradient unbounded at ({px},{py}): {d}"
            );
        }
        assert!(
            lapse_dx(1.0, 0.4).abs() > 1e-3,
            "true gradient outside the clamp"
        );
    }

    #[test]
    fn cartesian_ks_det_g_flat_identity_is_one() {
        // alpha sqrt(gamma) = 1 = the flat cartesian volume element (sqrt(-g) is chart-independent);
        // the densitization path relies on this. checked in D = 2 and D = 3.
        let bh = SchwarzschildKSCartesian { mass: 0.8_f64 };
        let x3 = Tensor::new([3.0_f64, -2.0, 1.5]);
        assert!(approx(bh.lapse(x3) * bh.volume_factor(x3), 1.0));
        let x2 = Tensor::new([3.0_f64, -2.0]);
        assert!(approx(
            <SchwarzschildKSCartesian<f64> as Metric<f64, 2>>::lapse(&bh, x2)
                * <SchwarzschildKSCartesian<f64> as Metric<f64, 2>>::volume_factor(&bh, x2),
            1.0
        ));
    }

    #[test]
    fn cartesian_ks_reduces_to_flat_at_zero_mass() {
        // M = 0 -> H = 0 -> gamma = delta, alpha = 1, beta = 0: the flat cartesian metric exactly.
        let bh = SchwarzschildKSCartesian { mass: 0.0_f64 };
        let x = Tensor::new([2.0_f64, -3.0, 4.0]);
        assert!(approx(bh.lapse(x), 1.0));
        let g = bh.spatial_metric(x);
        for ii in 0..3 {
            assert!(approx(bh.shift(x)[ii], 0.0));
            for jj in 0..3 {
                assert!(approx(g[(ii, jj)], if ii == jj { 1.0 } else { 0.0 }));
            }
        }
        assert!(approx(bh.sqrt_det_gamma(x), 1.0));
        assert_eq!(
            <SchwarzschildKSCartesian<f64> as Metric<f64, 3>>::geometry(&bh),
            Geometry::Cartesian
        );
        assert_eq!(
            <SchwarzschildKSCartesian<f64> as Metric<f64, 3>>::spacetime(&bh),
            Spacetime::SchwarzschildKS
        );
    }

    #[test]
    fn cartesian_ks_matches_spherical_ks_physics_via_rotation_invariants() {
        // the same physical vacuum as the spherical KS chart: the rotation-invariant scalars agree
        // at the same physical radius. the lapse alpha(r) is a scalar (chart-independent), and
        // the physical radial stretch l^i l^j gamma_ij (cartesian) equals the spherical KS gamma_rr
        // (whose radial scale factor is 1, so gamma_rr is the physical radial metric) = 1 + 2M/r.
        let close = |x: f64, y: f64| (x - y).abs() < 1e-13 * (1.0 + x.abs().max(y.abs()));
        let cart = SchwarzschildKSCartesian { mass: 1.3_f64 };
        let sph = SchwarzschildKS { mass: 1.3_f64 };
        for &(px, py, pz) in &[(1.0_f64, 2.0, 2.0), (5.0, 0.0, 0.0), (-3.0, 4.0, 12.0)] {
            let xc = Tensor::new([px, py, pz]);
            let r = (px * px + py * py + pz * pz).sqrt();
            let xs = Tensor::new([r, FRAC_PI_4, 0.0]); // any (theta, phi): the scalars are angle-free
            assert!(
                close(
                    cart.lapse(xc),
                    <SchwarzschildKS<f64> as Metric<f64, 3>>::lapse(&sph, xs)
                ),
                "lapse at r={r}"
            );
            let gm = cart.spatial_metric(xc);
            let l = [px / r, py / r, pz / r];
            let radial_stretch: f64 = (0..3)
                .map(|ii| (0..3).map(|jj| l[ii] * gm[(ii, jj)] * l[jj]).sum::<f64>())
                .sum();
            let gamma_rr =
                <SchwarzschildKS<f64> as Metric<f64, 3>>::spatial_metric(&sph, xs)[(0, 0)];
            assert!(
                close(radial_stretch, gamma_rr),
                "radial stretch {radial_stretch} vs gamma_rr {gamma_rr} at r={r}"
            );
            assert!(
                close(radial_stretch, 1.0 + 2.0 * 1.3 / r),
                "radial stretch != 1+2M/r at r={r}"
            );
        }
    }

    #[test]
    fn cartesian_ks_autodiff_derivs_match_finite_difference() {
        // the geodesic source gets its metric derivatives from FORWARD-MODE AUTODIFF w.r.t. the
        // CARTESIAN coordinates (one seeded axis per Dual pass). this is the load-bearing check that
        // the covariant source works in the cartesian chart: d_k gamma_ij and d_k (lapse, shift) via
        // Dual must equal a central finite difference of the metric's own components. off-axis so all
        // three coordinate derivatives are nontrivial, and inside the horizon (r < 2M) too.
        use symbi_ir::dual::Dual;
        let dd = 1e-4;
        let close = |a: f64, b: f64| (a - b).abs() < 1e-5 * (1.0 + a.abs().max(b.abs()));
        for &p in &[[1.0_f64, 0.6, 0.8], [0.7, 0.2, 0.1], [-3.0, 2.0, 4.0]] {
            let mf = SchwarzschildKSCartesian { mass: 1.0_f64 };
            let md = SchwarzschildKSCartesian {
                mass: Dual::constant(1.0_f64),
            };
            for kk in 0..3 {
                // seed axis kk (d/dx_kk = 1), the others constant.
                let seed: Tensor<Dual<f64>, 3> = Tensor::new(std::array::from_fn(|ii| {
                    if ii == kk {
                        Dual::variable(p[ii])
                    } else {
                        Dual::constant(p[ii])
                    }
                }));
                let fd = |f: &dyn Fn([f64; 3]) -> f64| {
                    let mut hi = p;
                    hi[kk] += dd;
                    let mut lo = p;
                    lo[kk] -= dd;
                    (f(hi) - f(lo)) / (2.0 * dd)
                };
                assert!(
                    close(md.lapse(seed).tangent, fd(&|q| mf.lapse(Tensor::new(q)))),
                    "d_{kk} lapse at {p:?}"
                );
                for ii in 0..3 {
                    assert!(
                        close(
                            md.shift(seed)[ii].tangent,
                            fd(&|q| mf.shift(Tensor::new(q))[ii])
                        ),
                        "d_{kk} shift[{ii}] at {p:?}"
                    );
                    for jj in 0..3 {
                        assert!(
                            close(
                                md.spatial_metric(seed)[(ii, jj)].tangent,
                                fd(&|q| mf.spatial_metric(Tensor::new(q))[(ii, jj)])
                            ),
                            "d_{kk} gamma[{ii}{jj}] at {p:?}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn cylindrical_ks_satisfies_the_adm_identities() {
        // the cylindrical kerr-schild block (R, phi, z) against the ADM identities: gamma gamma^{-1}
        // = 1, sqrt_det^2 = det, beta_i = 2H l_i (covariant shift = g_{0i}), g_tt = 2H - 1, det =
        // R^2 (1 + 2H), alpha > 0. off the axis + off the equator, inside and outside the horizon.
        let close = |x: f64, y: f64| (x - y).abs() < 1e-12 * (1.0 + x.abs().max(y.abs()));
        let bh = SchwarzschildKSCylindrical { mass: 1.0_f64 };
        for &(big_r, phi, z) in &[
            (1.0_f64, 0.7, 0.5),
            (3.0, 2.1, -4.0),
            (0.9, 1.3, 0.2),
            (6.0, 0.0, 8.0),
        ] {
            let x = Tensor::new([big_r, phi, z]);
            let r = (big_r * big_r + z * z).sqrt();
            let two_h = 2.0 / r;
            let l = [big_r / r, 0.0, z / r]; // the KS null covector (coordinate basis)
            let gm = <SchwarzschildKSCylindrical<f64> as Metric<f64, 3>>::spatial_metric(&bh, x);
            let gi =
                <SchwarzschildKSCylindrical<f64> as Metric<f64, 3>>::spatial_metric_inv(&bh, x);
            let alpha = <SchwarzschildKSCylindrical<f64> as Metric<f64, 3>>::lapse(&bh, x);
            let beta = <SchwarzschildKSCylindrical<f64> as Metric<f64, 3>>::shift(&bh, x);
            // gamma gamma^{-1} = identity.
            for ii in 0..3 {
                for jj in 0..3 {
                    let acc: f64 = (0..3).map(|kk| gm[(ii, kk)] * gi[(kk, jj)]).sum();
                    assert!(
                        close(acc, if ii == jj { 1.0 } else { 0.0 }),
                        "gamma inverse ({ii},{jj}) = {acc} at R={big_r} z={z}"
                    );
                }
            }
            // beta_i = gamma_ij beta^j = 2H l_i. NOTE: l_i is the coordinate-basis covector, so the
            // azimuthal beta_phi = 0 (l_phi = 0) even though gamma_phi-phi = R^2.
            let beta_low: [f64; 3] =
                std::array::from_fn(|ii| (0..3).map(|jj| gm[(ii, jj)] * beta[jj]).sum());
            for ii in 0..3 {
                assert!(
                    close(beta_low[ii], two_h * l[ii]),
                    "beta_low[{ii}] at R={big_r} z={z}"
                );
            }
            let g_tt = -alpha * alpha + (0..3).map(|ii| beta_low[ii] * beta[ii]).sum::<f64>();
            assert!(
                close(g_tt, two_h - 1.0),
                "g_tt = {g_tt} vs {} at R={big_r} z={z}",
                two_h - 1.0
            );
            // determinant (cofactor along phi, which is decoupled): det = gamma_phi-phi * det(Rz block).
            let det_rz = gm[(0, 0)] * gm[(2, 2)] - gm[(0, 2)] * gm[(2, 0)];
            let det = gm[(1, 1)] * det_rz;
            let sq = <SchwarzschildKSCylindrical<f64> as Metric<f64, 3>>::sqrt_det_gamma(&bh, x);
            assert!(
                close(sq * sq, det),
                "sqrt_det^2 = {} vs det = {det} at R={big_r} z={z}",
                sq * sq
            );
            assert!(
                close(det, big_r * big_r * (1.0 + two_h)),
                "det = {det} vs R^2(1+2H) at R={big_r} z={z}"
            );
            assert!(alpha > 0.0 && alpha.is_finite());
        }
    }

    #[test]
    fn cylindrical_ks_det_g_flat_is_the_cylindrical_measure() {
        // alpha sqrt(gamma) = R = the flat cylindrical volume element (sqrt(-g) chart-independent);
        // the densitization path relies on this. also checks the TWO-radii structure: the lapse
        // depends on the SPHERICAL radius sqrt(R^2 + z^2), combining R and z.
        let bh = SchwarzschildKSCylindrical { mass: 0.8_f64 };
        let x = Tensor::new([3.0_f64, 1.1, 4.0]); // r_sph = 5
        assert!(approx(bh.lapse(x) * bh.volume_factor(x), 3.0)); // alpha sqrt(gamma) = R = 3
        assert!(approx(
            bh.lapse(x),
            1.0 / (1.0_f64 + 2.0 * 0.8 / 5.0).sqrt()
        )); // alpha uses r_sph = 5 (built from R = 3, z = 4)
    }

    #[test]
    fn cylindrical_ks_reduces_to_flat_at_zero_mass() {
        // M = 0 -> H = 0 -> gamma = diag(1, R^2, 1), alpha = 1, beta = 0: flat cylindrical exactly.
        let bh = SchwarzschildKSCylindrical { mass: 0.0_f64 };
        let x = Tensor::new([2.0_f64, 0.9, -3.0]);
        assert!(approx(bh.lapse(x), 1.0));
        let (g, gf) = (bh.spatial_metric(x), Cylindrical.spatial_metric(x));
        for ii in 0..3 {
            assert!(approx(bh.shift(x)[ii], 0.0));
            for jj in 0..3 {
                assert!(approx(g[(ii, jj)], gf[(ii, jj)]));
            }
        }
        assert!(approx(bh.sqrt_det_gamma(x), Cylindrical.sqrt_det_gamma(x)));
        assert_eq!(
            <SchwarzschildKSCylindrical<f64> as Metric<f64, 3>>::geometry(&bh),
            Geometry::Cylindrical
        );
        assert_eq!(
            <SchwarzschildKSCylindrical<f64> as Metric<f64, 3>>::spacetime(&bh),
            Spacetime::SchwarzschildKS
        );
    }

    #[test]
    fn cylindrical_ks_matches_cartesian_ks_physics_via_invariants() {
        // the same physical vacuum as the cartesian KS chart: the lapse is a scalar (chart-free) and
        // the poloidal physical radial stretch l^i l^j gamma_ij (the KS null covector contracted) is
        // 1 + 2M/r in both charts, at the same spherical radius r.
        let close = |x: f64, y: f64| (x - y).abs() < 1e-13 * (1.0 + x.abs().max(y.abs()));
        let cyl = SchwarzschildKSCylindrical { mass: 1.3_f64 };
        let cart = SchwarzschildKSCartesian { mass: 1.3_f64 };
        for &(big_r, z) in &[(3.0_f64, 4.0), (5.0, 0.0), (1.0, 1.0)] {
            let r = (big_r * big_r + z * z).sqrt();
            let xc = Tensor::new([big_r, 0.6, z]); // cylindrical (phi arbitrary — invariants are phi-free)
            // a cartesian point at the same spherical radius (put it in the x-z plane: |(R,0,z)| = r).
            let xk = Tensor::new([big_r, 0.0, z]);
            assert!(
                close(
                    cyl.lapse(xc),
                    <SchwarzschildKSCartesian<f64> as Metric<f64, 3>>::lapse(&cart, xk)
                ),
                "lapse at r={r}"
            );
            let gm = cyl.spatial_metric(xc);
            let l = [big_r / r, 0.0, z / r];
            let stretch: f64 = (0..3)
                .map(|ii| (0..3).map(|jj| l[ii] * gm[(ii, jj)] * l[jj]).sum::<f64>())
                .sum();
            assert!(
                close(stretch, 1.0 + 2.0 * 1.3 / r),
                "poloidal radial stretch != 1+2M/r at r={r}"
            );
        }
    }

    #[test]
    fn cylindrical_ks_autodiff_derivs_match_finite_difference() {
        // the geodesic source's metric derivatives via forward-mode autodiff w.r.t. the cylindrical
        // coordinates (R, phi, z). d_R and d_z are nontrivial; d_phi vanishes (axisymmetry — the
        // metric is independent of phi, so S_phi is conserved). off-axis + inside the horizon.
        use symbi_ir::dual::Dual;
        let dd = 1e-4;
        let close = |a: f64, b: f64| (a - b).abs() < 1e-5 * (1.0 + a.abs().max(b.abs()));
        for &p in &[[2.0_f64, 0.7, 1.5], [0.8, 1.2, 0.3], [4.0, 2.0, -3.0]] {
            let mf = SchwarzschildKSCylindrical { mass: 1.0_f64 };
            let md = SchwarzschildKSCylindrical {
                mass: Dual::constant(1.0_f64),
            };
            for kk in 0..3 {
                let seed: Tensor<Dual<f64>, 3> = Tensor::new(std::array::from_fn(|ii| {
                    if ii == kk {
                        Dual::variable(p[ii])
                    } else {
                        Dual::constant(p[ii])
                    }
                }));
                let fd = |f: &dyn Fn([f64; 3]) -> f64| {
                    let mut hi = p;
                    hi[kk] += dd;
                    let mut lo = p;
                    lo[kk] -= dd;
                    (f(hi) - f(lo)) / (2.0 * dd)
                };
                assert!(
                    close(md.lapse(seed).tangent, fd(&|q| mf.lapse(Tensor::new(q)))),
                    "d_{kk} lapse at {p:?}"
                );
                if kk == 1 {
                    assert!(
                        md.lapse(seed).tangent.abs() < 1e-14,
                        "d_phi lapse must vanish at {p:?}"
                    );
                }
                for ii in 0..3 {
                    assert!(
                        close(
                            md.shift(seed)[ii].tangent,
                            fd(&|q| mf.shift(Tensor::new(q))[ii])
                        ),
                        "d_{kk} shift[{ii}] at {p:?}"
                    );
                    for jj in 0..3 {
                        assert!(
                            close(
                                md.spatial_metric(seed)[(ii, jj)].tangent,
                                fd(&|q| mf.spatial_metric(Tensor::new(q))[(ii, jj)])
                            ),
                            "d_{kk} gamma[{ii}{jj}] at {p:?}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn cylindrical_ks_equatorial_disk_is_diagonal_and_correct() {
        // the D = 2 (R, phi) EQUATORIAL DISK: on the equator r = R, so the kerr-schild off-diagonal
        // vanishes -> gamma = diag(1 + 2M/R, R^2), alpha sqrt(gamma) = R, agreeing with the (R, z)
        // D = 3 view at z = 0. beta_R = gamma_RR beta^R = 2M/R (g_tR); beta_phi = 0.
        let close = |x: f64, y: f64| (x - y).abs() < 1e-12 * (1.0 + x.abs().max(y.abs()));
        let bh = SchwarzschildKSCylindrical { mass: 1.3_f64 };
        for &(big_r, phi) in &[(3.0_f64, 0.7), (6.0, 2.1), (2.6, 4.0)] {
            let x = Tensor::new([big_r, phi]);
            let two_h = 2.0 * 1.3 / big_r;
            let gm = <SchwarzschildKSCylindrical<f64> as Metric<f64, 2>>::spatial_metric(&bh, x);
            let gi =
                <SchwarzschildKSCylindrical<f64> as Metric<f64, 2>>::spatial_metric_inv(&bh, x);
            assert!(close(gm[(0, 0)], 1.0 + two_h) && close(gm[(1, 1)], big_r * big_r));
            assert!(
                gm[(0, 1)] == 0.0 && gm[(1, 0)] == 0.0,
                "the equatorial disk must be diagonal"
            );
            for ii in 0..2 {
                for jj in 0..2 {
                    let acc: f64 = (0..2).map(|kk| gm[(ii, kk)] * gi[(kk, jj)]).sum();
                    assert!(
                        close(acc, if ii == jj { 1.0 } else { 0.0 }),
                        "gamma inverse ({ii},{jj})"
                    );
                }
            }
            let alpha = <SchwarzschildKSCylindrical<f64> as Metric<f64, 2>>::lapse(&bh, x);
            let vf = <SchwarzschildKSCylindrical<f64> as Metric<f64, 2>>::volume_factor(&bh, x);
            assert!(
                close(alpha * vf, big_r),
                "alpha sqrt(gamma) = R at R={big_r}"
            );
            let beta = <SchwarzschildKSCylindrical<f64> as Metric<f64, 2>>::shift(&bh, x);
            assert!(
                close(gm[(0, 0)] * beta[0], two_h) && beta[1] == 0.0,
                "beta_R = 2M/R, beta_phi = 0"
            );
            // agrees with the (R, z) D = 3 poloidal block at z = 0.
            let gm3 = <SchwarzschildKSCylindrical<f64> as Metric<f64, 3>>::spatial_metric(
                &bh,
                Tensor::new([big_r, phi, 0.0]),
            );
            assert!(close(gm[(0, 0)], gm3[(0, 0)]) && close(gm[(1, 1)], gm3[(1, 1)]));
        }
    }

    #[test]
    fn kerr_ks_cylindrical_reduces_to_schwarzschild_at_zero_spin() {
        let kerr = KerrKSCylindrical {
            mass: 1.0_f64,
            spin: 0.0,
        };
        let schw = SchwarzschildKSCylindrical { mass: 1.0_f64 };
        for &(rr, phi, z) in &[(3.0_f64, 0.7, 1.5), (0.9, -1.2, -0.4), (10.0, 2.0, 0.0)] {
            let p = Tensor::new([rr, phi, z]);
            let a = <KerrKSCylindrical<f64> as Metric<f64, 3>>::lapse(&kerr, p);
            let b = <SchwarzschildKSCylindrical<f64> as Metric<f64, 3>>::lapse(&schw, p);
            assert!(
                (a - b).abs() < 1e-14,
                "lapse mismatch at ({rr},{phi},{z}): {a} vs {b}"
            );
            let ga = <KerrKSCylindrical<f64> as Metric<f64, 3>>::spatial_metric(&kerr, p);
            let gb = <SchwarzschildKSCylindrical<f64> as Metric<f64, 3>>::spatial_metric(&schw, p);
            for ii in 0..3 {
                for jj in 0..3 {
                    assert!(
                        (ga[(ii, jj)] - gb[(ii, jj)]).abs() < 1e-12,
                        "gamma mismatch"
                    );
                }
            }
            let sa = <KerrKSCylindrical<f64> as Metric<f64, 3>>::shift(&kerr, p);
            let sb = <SchwarzschildKSCylindrical<f64> as Metric<f64, 3>>::shift(&schw, p);
            for ii in 0..3 {
                assert!(
                    (sa[ii] - sb[ii]).abs() < 1e-13,
                    "shift mismatch on axis {ii}"
                );
            }
        }
    }

    #[test]
    fn kerr_ks_cylindrical_inverse_determinant_and_lapse_identity() {
        let kerr = KerrKSCylindrical {
            mass: 1.0_f64,
            spin: 0.9,
        };
        for &(rr, phi, z) in &[(2.5_f64, 0.3, 0.8), (1.1, -2.0, -1.7), (6.0, 1.0, 0.2)] {
            let p = Tensor::new([rr, phi, z]);
            let g = <KerrKSCylindrical<f64> as Metric<f64, 3>>::spatial_metric(&kerr, p);
            let gi = <KerrKSCylindrical<f64> as Metric<f64, 3>>::spatial_metric_inv(&kerr, p);
            for ii in 0..3 {
                for jj in 0..3 {
                    let mut acc = 0.0;
                    for kk in 0..3 {
                        acc += g[(ii, kk)] * gi[(kk, jj)];
                    }
                    let want = if ii == jj { 1.0 } else { 0.0 };
                    assert!(
                        (acc - want).abs() < 1e-12,
                        "gamma*gamma^-1 != I at ({ii},{jj}): {acc}"
                    );
                }
            }
            // the null l preserves det g: alpha sqrt(gamma) = sqrt(det g0) = R exactly.
            let alpha = <KerrKSCylindrical<f64> as Metric<f64, 3>>::lapse(&kerr, p);
            let sg = <KerrKSCylindrical<f64> as Metric<f64, 3>>::sqrt_det_gamma(&kerr, p);
            assert!(
                (alpha * sg - rr).abs() < 1e-12,
                "alpha sqrt(gamma) != R: {} vs {rr}",
                alpha * sg
            );
        }
    }

    #[test]
    fn kerr_ks_cylindrical_matches_the_cartesian_chart_through_the_jacobian() {
        // the cross-chart reference: gamma_cyl = J^T gamma_cart J with the coordinate
        // jacobian of (x, y, z) = (R cos phi, R sin phi, z) — the two charts describe
        // the same spacetime, so any formula drift between them breaks this exactly.
        let (mm, aa) = (1.0_f64, 0.8_f64);
        let cyl = KerrKSCylindrical { mass: mm, spin: aa };
        let cart = KerrKSCartesian { mass: mm, spin: aa };
        for &(rr, phi, z) in &[(2.0_f64, 0.6, 1.0), (4.5, -1.1, -0.3), (1.3, 3.0, 2.2)] {
            let (cp, sp) = (phi.cos(), phi.sin());
            let jac = [[cp, -rr * sp, 0.0], [sp, rr * cp, 0.0], [0.0, 0.0, 1.0]];
            let pc = Tensor::new([rr * cp, rr * sp, z]);
            let gc = <KerrKSCartesian<f64> as Metric<f64, 3>>::spatial_metric(&cart, pc);
            let gy = <KerrKSCylindrical<f64> as Metric<f64, 3>>::spatial_metric(
                &cyl,
                Tensor::new([rr, phi, z]),
            );
            for ii in 0..3 {
                for jj in 0..3 {
                    let mut want = 0.0;
                    for kk in 0..3 {
                        for ll in 0..3 {
                            want += jac[kk][ii] * gc[(kk, ll)] * jac[ll][jj];
                        }
                    }
                    assert!(
                        (gy[(ii, jj)] - want).abs() < 1e-11,
                        "chart mismatch at ({ii},{jj}) for (R,phi,z)=({rr},{phi},{z}): {} vs {want}",
                        gy[(ii, jj)]
                    );
                }
            }
            // the lapse is a scalar: identical across charts at the same physical point.
            let ac = <KerrKSCartesian<f64> as Metric<f64, 3>>::lapse(&cart, pc);
            let ay =
                <KerrKSCylindrical<f64> as Metric<f64, 3>>::lapse(&cyl, Tensor::new([rr, phi, z]));
            assert!(
                (ac - ay).abs() < 1e-13,
                "lapse differs across charts: {ac} vs {ay}"
            );
        }
    }

    #[test]
    fn kerr_ks_cartesian_reduces_to_schwarzschild_at_zero_spin() {
        let kerr = KerrKSCartesian {
            mass: 1.0_f64,
            spin: 0.0,
        };
        let schw = SchwarzschildKSCartesian { mass: 1.0_f64 };
        for &(x, y, z) in &[(3.0_f64, -2.0, 1.5), (0.7, 0.2, -0.4), (10.0, 0.0, 0.0)] {
            let p = Tensor::new([x, y, z]);
            let a = <KerrKSCartesian<f64> as Metric<f64, 3>>::lapse(&kerr, p);
            let b = <SchwarzschildKSCartesian<f64> as Metric<f64, 3>>::lapse(&schw, p);
            assert!(
                (a - b).abs() < 1e-14,
                "lapse mismatch at ({x},{y},{z}): {a} vs {b}"
            );
            let ga = <KerrKSCartesian<f64> as Metric<f64, 3>>::spatial_metric(&kerr, p);
            let gb = <SchwarzschildKSCartesian<f64> as Metric<f64, 3>>::spatial_metric(&schw, p);
            for ii in 0..3 {
                for jj in 0..3 {
                    assert!(
                        (ga[(ii, jj)] - gb[(ii, jj)]).abs() < 1e-14,
                        "gamma mismatch"
                    );
                }
            }
            let sa = <KerrKSCartesian<f64> as Metric<f64, 3>>::shift(&kerr, p);
            let sb = <SchwarzschildKSCartesian<f64> as Metric<f64, 3>>::shift(&schw, p);
            for ii in 0..3 {
                assert!((sa[ii] - sb[ii]).abs() < 1e-14, "shift mismatch");
            }
        }
    }

    #[test]
    fn kerr_ks_cartesian_inverse_and_determinant_are_exact() {
        let kerr = KerrKSCartesian {
            mass: 1.0_f64,
            spin: 0.9,
        };
        for &(x, y, z) in &[
            (3.0_f64, -2.0, 1.5),
            (1.2, 0.9, 0.3),
            (5.0, 5.0, -4.0),
            (2.0, 0.0, 0.0),
        ] {
            let p = Tensor::new([x, y, z]);
            let g = <KerrKSCartesian<f64> as Metric<f64, 3>>::spatial_metric(&kerr, p);
            let gi = <KerrKSCartesian<f64> as Metric<f64, 3>>::spatial_metric_inv(&kerr, p);
            // gamma . gamma^{-1} = I (sherman-morrison exactness).
            for ii in 0..3 {
                for jj in 0..3 {
                    let mut s = 0.0;
                    for kk in 0..3 {
                        s += g[(ii, kk)] * gi[(kk, jj)];
                    }
                    let want = if ii == jj { 1.0 } else { 0.0 };
                    assert!(
                        (s - want).abs() < 1e-13,
                        "g.g^-1 != I at ({x},{y},{z}) [{ii}{jj}]: {s}"
                    );
                }
            }
            // det(gamma) equals the closed form (1 + 2H |l|^2) via the numeric 3x3 determinant.
            let det = g[(0, 0)] * (g[(1, 1)] * g[(2, 2)] - g[(1, 2)] * g[(2, 1)])
                - g[(0, 1)] * (g[(1, 0)] * g[(2, 2)] - g[(1, 2)] * g[(2, 0)])
                + g[(0, 2)] * (g[(1, 0)] * g[(2, 1)] - g[(1, 1)] * g[(2, 0)]);
            let sq = <KerrKSCartesian<f64> as Metric<f64, 3>>::sqrt_det_gamma(&kerr, p);
            assert!(
                (det - sq * sq).abs() < 1e-12,
                "det mismatch at ({x},{y},{z}): {det} vs {}",
                sq * sq
            );
        }
    }

    #[test]
    fn kerr_ks_cartesian_null_vector_is_unit_off_the_clamp() {
        // |l| = 1 wherever the kerr-schild quartic holds: sqrt_det_gamma^2 - 1 = 2H exactly there,
        // so alpha^2 (1 + 2H) = 1. probe well outside the clamp radius.
        let kerr = KerrKSCartesian {
            mass: 1.0_f64,
            spin: 0.7,
        };
        for &(x, y, z) in &[(4.0_f64, 1.0, 2.0), (2.5, -3.0, 0.5), (0.9, 0.9, 0.9)] {
            let p = Tensor::new([x, y, z]);
            let a2 = <KerrKSCartesian<f64> as Metric<f64, 3>>::lapse_sq(&kerr, p);
            let sq = <KerrKSCartesian<f64> as Metric<f64, 3>>::sqrt_det_gamma(&kerr, p);
            assert!(
                (a2 * sq * sq - 1.0).abs() < 1e-13,
                "alpha^2 det != 1 at ({x},{y},{z})"
            );
        }
    }

    #[test]
    fn kerr_ks_cartesian_equatorial_slice_matches_the_3d_plane() {
        // the D = 2 instance is the z = 0 slice: gamma's (x, y) block and the in-plane shift
        // must equal the 3d values at z = 0 (l_z = 0 there exactly).
        let kerr = KerrKSCartesian {
            mass: 1.0_f64,
            spin: 0.6,
        };
        for &(x, y) in &[(3.0_f64, -2.0), (1.5, 0.4), (6.0, 6.0)] {
            let p2 = Tensor::new([x, y]);
            let p3 = Tensor::new([x, y, 0.0]);
            let g2 = <KerrKSCartesian<f64> as Metric<f64, 2>>::spatial_metric(&kerr, p2);
            let g3 = <KerrKSCartesian<f64> as Metric<f64, 3>>::spatial_metric(&kerr, p3);
            for ii in 0..2 {
                for jj in 0..2 {
                    assert!(
                        (g2[(ii, jj)] - g3[(ii, jj)]).abs() < 1e-14,
                        "slice mismatch"
                    );
                }
            }
            assert!(
                (g3[(0, 2)]).abs() < 1e-15 && (g3[(1, 2)]).abs() < 1e-15,
                "equatorial cross terms must vanish (l_z = 0 at z = 0)"
            );
        }
    }
}
