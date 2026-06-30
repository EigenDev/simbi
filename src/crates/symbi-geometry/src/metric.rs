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

use symbi_algebra::{Tensor, Matrix, Contravariant, Covariant, Physical, Embedded};
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
    pub fn as_i32(self) -> i32 { self as i32 }
}

/// spacetime identifier — the background a regime evolves on. ORTHOGONAL to BOTH the spatial
/// [`Geometry`] and the physics regime: GR is not a regime, it is a curved spacetime, so a single
/// SR regime (Srhd / Rmhd) composes with any spacetime here without duplication. flat `Minkowski`
/// (lapse = 1, shift = 0, gamma = identity in physical components) is the default — every realized
/// run today. drives the lapse / sqrt(gamma) densitization selector in the kernel (B3). integer
/// repr matches the GPU kernel convention (mirrors `Geometry`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(i32)]
pub enum Spacetime {
    #[default]
    Minkowski = 0,
    /// static spherically-symmetric vacuum (standard coords). DIAGONAL spatial metric, shift = 0,
    /// non-trivial lapse alpha = sqrt(1 - 2M/r). the mass M is a kernel parameter, NOT part of the
    /// tag — the tag selects the kernel STRUCTURE, M rides as a value (the `Schwarzschild` Metric
    /// impl's field at host, a scalar in the trace).
    Schwarzschild = 1,
    // Kerr (non-diagonal, frame-dragging beta != 0) lands here at B5.
}

impl Spacetime {
    /// integer representation for GPU kernel dispatch.
    pub fn as_i32(self) -> i32 { self as i32 }
}

// ============================================================
// metric trait: full 3+1 interface
// ============================================================

/// metric on a D-dimensional spatial manifold.
///
/// provides the GENERAL 3+1 ADM surface — valid for ANY metric, diagonal or not:
/// - lapse \alpha (1 for flat spacetimes)
/// - shift \beta^i (0 for static spacetimes)
/// - spatial metric \gamma_{ij} and its inverse \gamma^{ij}
/// - the volume element \sqrt{\gamma} (= `sqrt_det_gamma` / `volume_factor`)
/// - `lower`/`raise` (variance-typed: `Contravariant` <-> `Covariant`, general tensor contraction)
///
/// the ORTHOGONAL-FRAME surface — scale factors `h_i` and the orthonormal-frame
/// `vector_to_cartesian` — is NOT here: it only exists for a DIAGONAL metric, so it lives on the
/// [`DiagonalMetric`] subtrait. a non-diagonal (Kerr-class) metric impls `Metric` but NOT
/// `DiagonalMetric`, so the compiler forbids orthogonal quadrature on it until the non-diagonal
/// forms are written — the "GRMHD-forward" claim is type-enforced, not asserted. (the realized
/// physics today is all diagonal: flat + orthogonal-curvilinear.)
///
/// implementations: Cartesian, Spherical, Cylindrical (all `DiagonalMetric`),
/// and (future) Schwarzschild, Kerr, etc. (`Metric` only, until their quadrature lands).
pub trait Metric<S: Scalar, const D: usize> {
    /// coordinate system for this metric.
    fn geometry(&self) -> Geometry { Geometry::Cartesian }

    /// the spacetime background (flat vs curved). ORTHOGONAL to `geometry()`: `Minkowski` for every
    /// flat metric, a curved variant (Schwarzschild, ...) for GR. selects the lapse / sqrt(gamma)
    /// densitization path in the kernel (B3); flat -> the densitization is a no-op.
    fn spacetime(&self) -> Spacetime { Spacetime::Minkowski }

    /// lapse function \alpha. determines time dilation.
    /// flat spacetime: \alpha = 1.
    fn lapse(&self, x: Tensor<S, D>) -> S {
        let _ = x;
        S::ONE
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

    /// scale factors h_i where \gamma_{ii} = h_i^2. ONLY well-defined for a diagonal metric — the
    /// `where Self: DiagonalMetric` bound makes this a COMPILE ERROR on a non-diagonal metric
    /// rather than silently dropping the off-diagonal terms (the former "panics for non-diagonal"
    /// behavior was a silent sqrt-of-diagonal). the default reads the diagonal directly.
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
    ///   spherical 1D: r^2  (not 1 = sqrt_det_gamma of the 1x1 metric)
    ///   spherical 2D: r^2 sin(theta)  (not r = sqrt_det_gamma of 2x2 metric)
    ///   cylindrical 1D: r  (not 1)
    ///   cylindrical 2D: r  (not 1)
    fn volume_factor(&self, x: Tensor<S, D>) -> S {
        self.sqrt_det_gamma(x)
    }

    /// transform from this coordinate system to cartesian.
    fn to_cartesian(&self, x: Tensor<S, D>) -> Tensor<S, D>;

    /// transform from cartesian to this coordinate system.
    fn from_cartesian(&self, x: Tensor<S, D>) -> Tensor<S, D>;

    /// the frame morphism `Ortho -> Cart`: rotate a PHYSICAL (orthonormal-frame) vector into the
    /// global Cartesian frame. default: identity (cartesian: Ortho == Cart); override for
    /// non-cartesian (the rotation by the orthonormal basis directions). typed so the review's
    /// `vector_to_cartesian(lower(v))` is now a COMPILE ERROR — `lower` yields `Covariant`
    /// (coordinate basis), this wants `Physical` (orthonormal). docs/design/31.
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

    /// the SCALE-FACTOR BRIDGE `CoordUp -> Ortho`: `V_a = h_a v^a`. the one seam where the metric
    /// enters the (otherwise flat) orthonormal frame the substrate computes in. requires
    /// [`DiagonalMetric`] (the orthonormal frame exists only for a diagonal metric; a non-diagonal
    /// metric replaces this with a tetrad). docs/design/31 §2.
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
    /// NOTE: this is the CONTINUOUS analytical formula. for discrete schemes,
    /// use `momentum_source_inertial` + discrete pressure source from face
    /// area differences to achieve exact discrete equilibrium.
    fn momentum_source(
        &self, x: Tensor<S, D>, rho: S, vel: Tensor<S, D>, p: S,
    ) -> Tensor<S, D> {
        let _ = (x, rho, vel, p);
        Tensor::zeros()
    }

    /// inertial (velocity-dependent) part of the geometric momentum source.
    /// excludes ALL pressure terms — those must be computed from discrete
    /// face area differences for exact discrete equilibrium:
    ///
    ///   S_pressure[i] = p * (A^i_R - A^i_L) / V
    ///
    /// the total geometric source is:
    ///   S = S_pressure + S_inertial
    ///
    /// cartesian: S_inertial = 0.
    ///
    /// `mom` is the REGIME-AGNOSTIC conserved momentum density (Newtonian `rho v`,
    /// relativistic `rho h W^2 v`): the source is the bilinear `S^i = -Gamma^i_jk mom^j v^k`,
    /// so the same code serves every regime AND the magnetic tension (call with `mom = b`,
    /// `vel = b` for `-Gamma(b, b)`).
    fn momentum_source_inertial(
        &self, x: Tensor<S, D>, mom: Tensor<S, D>, vel: Tensor<S, D>,
    ) -> Tensor<S, D> {
        let _ = (x, mom, vel);
        Tensor::zeros()
    }
}

/// a metric whose spatial tensor \gamma_{ij} is DIAGONAL — flat-space + orthogonal-curvilinear
/// (Cartesian, Spherical, Cylindrical). only on such a metric do scale factors `h_i` and the
/// orthonormal-frame `vector_to_cartesian` make sense, so [`Metric::scale_factors`] /
/// [`Metric::vector_to_cartesian`] are gated `where Self: DiagonalMetric`. a non-diagonal
/// (Kerr-class) metric impls `Metric` but NOT this — so the compiler rejects orthogonal-frame
/// quadrature on it until the non-diagonal forms are written. a pure marker: the gated methods
/// already live on `Metric`; this trait only carries the diagonality proof obligation.
///
/// the gate, executable. a `DiagonalMetric`-bounded generic MAY take scale factors:
/// ```
/// use symbi_geometry::{DiagonalMetric, Spherical};
/// use symbi_algebra::Tensor;
/// fn h<M: DiagonalMetric<f64, 2>>(m: &M, x: Tensor<f64, 2>) -> Tensor<f64, 2> {
///     m.scale_factors(x)
/// }
/// let _ = h(&Spherical, Tensor::new([1.0, 0.5]));
/// ```
/// a `Metric`-only generic MAY NOT — it is a type error, not a silent wrong answer:
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

    fn to_cartesian(&self, x: Tensor<S, 1>) -> Tensor<S, 1> { x }
    fn from_cartesian(&self, x: Tensor<S, 1>) -> Tensor<S, 1> { x }
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

    fn to_cartesian(&self, x: Tensor<S, 2>) -> Tensor<S, 2> { x }
    fn from_cartesian(&self, x: Tensor<S, 2>) -> Tensor<S, 2> { x }
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

    fn to_cartesian(&self, x: Tensor<S, 3>) -> Tensor<S, 3> { x }
    fn from_cartesian(&self, x: Tensor<S, 3>) -> Tensor<S, 3> { x }
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
    fn geometry(&self) -> Geometry { Geometry::Spherical }
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

    fn to_cartesian(&self, x: Tensor<S, 1>) -> Tensor<S, 1> { x }
    fn from_cartesian(&self, x: Tensor<S, 1>) -> Tensor<S, 1> { x }

    fn volume_factor(&self, x: Tensor<S, 1>) -> S {
        let r = x[0];
        r * r
    }

    /// 1D spherical: S_r = 2p/r (pressure from 2 suppressed angular directions).
    fn momentum_source(
        &self, x: Tensor<S, 1>, _rho: S, _vel: Tensor<S, 1>, p: S,
    ) -> Tensor<S, 1> {
        let r = x[0];
        let two = S::ONE + S::ONE;
        Tensor::new([two * p / r])
    }

    /// 1D spherical inertial: no resolved angular velocity -> zero.
    fn momentum_source_inertial(
        &self, _x: Tensor<S, 1>, _mom: Tensor<S, 1>, _vel: Tensor<S, 1>,
    ) -> Tensor<S, 1> {
        Tensor::zeros()
    }
}

impl<S: Scalar> Metric<S, 2> for Spherical {
    fn geometry(&self) -> Geometry { Geometry::Spherical }
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
        Embedded::new(Tensor::new([
            v[0] * ct - v[1] * st,
            v[0] * st + v[1] * ct,
        ]))
    }

    fn vector_from_cartesian(&self, x: Tensor<S, 2>, v: Embedded<S, 2>) -> Physical<S, 2> {
        let theta = x[1];
        let ct = theta.cos();
        let st = theta.sin();
        Physical::new(Tensor::new([
            v[0] * ct + v[1] * st,
            -v[0] * st + v[1] * ct,
        ]))
    }

    fn volume_factor(&self, x: Tensor<S, 2>) -> S {
        let r = x[0];
        let theta = x[1];
        r * r * theta.sin().abs()
    }

    /// 2D spherical (r, theta): resolved theta + suppressed phi.
    /// S_r = (rho*V_t^2 + 2p) / r
    /// S_t = (p*cot(theta) - rho*V_r*V_t) / r
    fn momentum_source(
        &self, x: Tensor<S, 2>, rho: S, vel: Tensor<S, 2>, p: S,
    ) -> Tensor<S, 2> {
        let r = x[0];
        let theta = x[1];
        let vr = vel[0];
        let vt = vel[1];
        let two = S::ONE + S::ONE;
        let cot = theta.cos() / theta.sin();
        Tensor::new([
            (rho * vt * vt + two * p) / r,
            (p * cot - rho * vr * vt) / r,
        ])
    }

    /// 2D spherical inertial: centrifugal + coriolis, no pressure. regime-agnostic via the
    /// CONSERVED momentum density `mom`: S = -Gamma(mom, v).
    fn momentum_source_inertial(
        &self, x: Tensor<S, 2>, mom: Tensor<S, 2>, vel: Tensor<S, 2>,
    ) -> Tensor<S, 2> {
        let r = x[0];
        let mr = mom[0];
        let mt = mom[1];
        let vt = vel[1];
        Tensor::new([
            mt * vt / r,
            S::ZERO - mr * vt / r,
        ])
    }
}

impl<S: Scalar> Metric<S, 3> for Spherical {
    fn geometry(&self) -> Geometry { Geometry::Spherical }
    fn spatial_metric(&self, x: Tensor<S, 3>) -> Matrix<S, 3> {
        let r = x[0];
        let st = x[1].sin();
        Matrix::diag(Tensor::new([S::ONE, r * r, r * r * st * st]))
    }

    fn spatial_metric_inv(&self, x: Tensor<S, 3>) -> Matrix<S, 3> {
        let r = x[0];
        let st = x[1].sin();
        let r2 = r * r;
        Matrix::diag(Tensor::new([
            S::ONE,
            S::ONE / r2,
            S::ONE / (r2 * st * st),
        ]))
    }

    fn sqrt_det_gamma(&self, x: Tensor<S, 3>) -> S {
        let r = x[0];
        let st = x[1].sin();
        r * r * st.abs()
    }

    fn scale_factors(&self, x: Tensor<S, 3>) -> Tensor<S, 3> {
        let r = x[0];
        let st = x[1].sin();
        Tensor::new([S::ONE, r, r * st.abs()])
    }

    fn to_cartesian(&self, x: Tensor<S, 3>) -> Tensor<S, 3> {
        let (r, theta, phi) = (x[0], x[1], x[2]);
        let st = theta.sin();
        Tensor::new([
            r * st * phi.cos(),
            r * st * phi.sin(),
            r * theta.cos(),
        ])
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
            v[0] * ct       - v[1] * st,
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
            -v[0] * sp      + v[1] * cp,
        ]))
    }

    /// 3D spherical (r, theta, phi): full geometric source.
    /// S_r = (rho*(V_t^2 + V_p^2) + 2p) / r
    /// S_t = ((rho*V_p^2 + p)*cot(theta) - rho*V_r*V_t) / r
    /// S_p = -rho*V_p*(V_r + V_t*cot(theta)) / r
    fn momentum_source(
        &self, x: Tensor<S, 3>, rho: S, vel: Tensor<S, 3>, p: S,
    ) -> Tensor<S, 3> {
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
        &self, x: Tensor<S, 3>, mom: Tensor<S, 3>, vel: Tensor<S, 3>,
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
//   sqrt(-g)        = alpha sqrt(gamma) = r^2 sin(theta)   (the flat spherical area: the B3 gift)
//
//   the SPATIAL coordinate geometry is spherical (geometry() = Spherical); the CURVATURE lives in
//   the radial stretch 1/f and the lapse. this coordinate gamma feeds densitization / lower-raise /
//   (B4) the christoffel gravity source — NOT the hydro's physical-frame metric, which stays
//   identity in the orthonormal convention (the lapse enters the kernel via `gv_lapse_weight`).
//
//   reduced dims mirror Spherical: 1D (r) radial, 2D (r, theta). valid OUTSIDE the horizon r > 2M
//   (f > 0); r <= 2M makes sqrt(f) imaginary — the coordinate singularity, physical.
//
//   the momentum source (geodesic gravity) is the connection of the FULL 4-metric -> B4; left at
//   the trait default (zero) here. step A is the metric geometry + lapse only.
// ============================================================

/// the Schwarzschild (static spherically-symmetric vacuum) metric in standard coordinates. `mass`
/// is the geometric mass M (G = c = 1). a DIAGONAL metric (impls [`DiagonalMetric`]); the curvature
/// is the radial stretch f(r) = 1 - 2M/r and the lapse sqrt(f).
#[derive(Debug, Clone, Copy)]
pub struct Schwarzschild {
    /// the geometric mass M (units G = c = 1). the horizon is at r = 2M.
    pub mass: f64,
}

impl Schwarzschild {
    /// f(r) = 1 - 2M/r — the lapse-squared `alpha^2` AND the inverse radial metric coefficient
    /// `gamma^{rr}` (so `gamma_{rr} = 1/f`). positive outside the horizon (r > 2M).
    #[inline]
    fn f<S: Scalar>(&self, r: S) -> S {
        S::ONE - S::from_f64(2.0 * self.mass) / r
    }
}

impl<S: Scalar> Metric<S, 1> for Schwarzschild {
    fn geometry(&self) -> Geometry { Geometry::Spherical }
    fn spacetime(&self) -> Spacetime { Spacetime::Schwarzschild }

    fn lapse(&self, x: Tensor<S, 1>) -> S { self.f(x[0]).sqrt() }

    fn spatial_metric(&self, x: Tensor<S, 1>) -> Matrix<S, 1> {
        Matrix::diag(Tensor::new([S::ONE / self.f(x[0])]))
    }
    fn spatial_metric_inv(&self, x: Tensor<S, 1>) -> Matrix<S, 1> {
        Matrix::diag(Tensor::new([self.f(x[0])]))
    }
    fn sqrt_det_gamma(&self, x: Tensor<S, 1>) -> S { S::ONE / self.f(x[0]).sqrt() }
    fn scale_factors(&self, x: Tensor<S, 1>) -> Tensor<S, 1> {
        Tensor::new([S::ONE / self.f(x[0]).sqrt()])
    }

    fn to_cartesian(&self, x: Tensor<S, 1>) -> Tensor<S, 1> { x }
    fn from_cartesian(&self, x: Tensor<S, 1>) -> Tensor<S, 1> { x }

    /// the full proper volume element including the 2 suppressed angular directions: r^2 / sqrt(f).
    fn volume_factor(&self, x: Tensor<S, 1>) -> S {
        let r = x[0];
        r * r / self.f(r).sqrt()
    }
}

impl<S: Scalar> Metric<S, 2> for Schwarzschild {
    fn geometry(&self) -> Geometry { Geometry::Spherical }
    fn spacetime(&self) -> Spacetime { Spacetime::Schwarzschild }

    fn lapse(&self, x: Tensor<S, 2>) -> S { self.f(x[0]).sqrt() }

    fn spatial_metric(&self, x: Tensor<S, 2>) -> Matrix<S, 2> {
        let r = x[0];
        Matrix::diag(Tensor::new([S::ONE / self.f(r), r * r]))
    }
    fn spatial_metric_inv(&self, x: Tensor<S, 2>) -> Matrix<S, 2> {
        let r = x[0];
        Matrix::diag(Tensor::new([self.f(r), S::ONE / (r * r)]))
    }
    fn sqrt_det_gamma(&self, x: Tensor<S, 2>) -> S {
        let r = x[0];
        r / self.f(r).sqrt() // sqrt((1/f) * r^2)
    }
    fn scale_factors(&self, x: Tensor<S, 2>) -> Tensor<S, 2> {
        let r = x[0];
        Tensor::new([S::ONE / self.f(r).sqrt(), r])
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

    /// proper volume incl. the suppressed phi direction: r^2 sin(theta) / sqrt(f).
    fn volume_factor(&self, x: Tensor<S, 2>) -> S {
        let r = x[0];
        r * r * x[1].sin().abs() / self.f(r).sqrt()
    }
}

impl<S: Scalar> Metric<S, 3> for Schwarzschild {
    fn geometry(&self) -> Geometry { Geometry::Spherical }
    fn spacetime(&self) -> Spacetime { Spacetime::Schwarzschild }

    fn lapse(&self, x: Tensor<S, 3>) -> S { self.f(x[0]).sqrt() }

    fn spatial_metric(&self, x: Tensor<S, 3>) -> Matrix<S, 3> {
        let r = x[0];
        let st = x[1].sin();
        Matrix::diag(Tensor::new([S::ONE / self.f(r), r * r, r * r * st * st]))
    }
    fn spatial_metric_inv(&self, x: Tensor<S, 3>) -> Matrix<S, 3> {
        let r = x[0];
        let st = x[1].sin();
        let r2 = r * r;
        Matrix::diag(Tensor::new([self.f(r), S::ONE / r2, S::ONE / (r2 * st * st)]))
    }
    fn sqrt_det_gamma(&self, x: Tensor<S, 3>) -> S {
        let r = x[0];
        r * r * x[1].sin().abs() / self.f(r).sqrt() // sqrt((1/f) r^2 r^2 sin^2)
    }
    fn scale_factors(&self, x: Tensor<S, 3>) -> Tensor<S, 3> {
        let r = x[0];
        let st = x[1].sin();
        Tensor::new([S::ONE / self.f(r).sqrt(), r, r * st.abs()])
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

impl<S: Scalar> DiagonalMetric<S, 1> for Schwarzschild {}
impl<S: Scalar> DiagonalMetric<S, 2> for Schwarzschild {}
impl<S: Scalar> DiagonalMetric<S, 3> for Schwarzschild {}

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
    fn geometry(&self) -> Geometry { Geometry::Cylindrical }
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

    fn to_cartesian(&self, x: Tensor<S, 1>) -> Tensor<S, 1> { x }
    fn from_cartesian(&self, x: Tensor<S, 1>) -> Tensor<S, 1> { x }

    fn volume_factor(&self, x: Tensor<S, 1>) -> S {
        x[0]
    }

    /// 1D cylindrical: S_r = p/r (pressure from 1 suppressed angular direction).
    fn momentum_source(
        &self, x: Tensor<S, 1>, _rho: S, _vel: Tensor<S, 1>, p: S,
    ) -> Tensor<S, 1> {
        let r = x[0];
        Tensor::new([p / r])
    }

    /// 1D cylindrical inertial: no resolved angular velocity -> zero.
    fn momentum_source_inertial(
        &self, _x: Tensor<S, 1>, _mom: Tensor<S, 1>, _vel: Tensor<S, 1>,
    ) -> Tensor<S, 1> {
        Tensor::zeros()
    }
}

/// axisymmetric cylindrical: coordinates (r, z).
/// phi direction integrated out — metric is euclidean in (r, z) plane.
impl<S: Scalar> Metric<S, 2> for Cylindrical {
    fn geometry(&self) -> Geometry { Geometry::Cylindrical }
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
    fn momentum_source(
        &self, x: Tensor<S, 2>, _rho: S, _vel: Tensor<S, 2>, p: S,
    ) -> Tensor<S, 2> {
        let r = x[0];
        Tensor::new([p / r, S::ZERO])
    }

    /// 2D cylindrical inertial: no resolved angular velocity -> zero.
    fn momentum_source_inertial(
        &self, _x: Tensor<S, 2>, _mom: Tensor<S, 2>, _vel: Tensor<S, 2>,
    ) -> Tensor<S, 2> {
        Tensor::zeros()
    }
}

impl<S: Scalar> Metric<S, 3> for Cylindrical {
    fn geometry(&self) -> Geometry { Geometry::Cylindrical }
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
    fn momentum_source(
        &self, x: Tensor<S, 3>, rho: S, vel: Tensor<S, 3>, p: S,
    ) -> Tensor<S, 3> {
        let r = x[0];
        let vr = vel[0];
        let vp = vel[1];
        Tensor::new([
            (rho * vp * vp + p) / r,
            -rho * vr * vp / r,
            S::ZERO,
        ])
    }

    /// 3D cylindrical inertial: centrifugal + coriolis, no pressure. regime-agnostic via the
    /// CONSERVED momentum density `mom`: S = -Gamma(mom, v).
    fn momentum_source_inertial(
        &self, x: Tensor<S, 3>, mom: Tensor<S, 3>, vel: Tensor<S, 3>,
    ) -> Tensor<S, 3> {
        let r = x[0];
        let mr = mom[0];
        let mp = mom[1];
        let vp = vel[1];
        Tensor::new([
            mp * vp / r,
            S::ZERO - mr * vp / r,
            S::ZERO,
        ])
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
    fn geometry(&self) -> Geometry { Geometry::Cylindrical }

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
        Embedded::new(Tensor::new([
            v[0] * cp - v[1] * sp,
            v[0] * sp + v[1] * cp,
        ]))
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
    fn momentum_source(
        &self, x: Tensor<S, 2>, rho: S, vel: Tensor<S, 2>, p: S,
    ) -> Tensor<S, 2> {
        let r = x[0];
        let vr = vel[0];
        let vp = vel[1];
        Tensor::new([
            (rho * vp * vp + p) / r,
            S::ZERO - rho * vr * vp / r,
        ])
    }

    /// 2D (r, phi) disk inertial: centrifugal + coriolis, regime-agnostic via the CONSERVED
    /// momentum density `mom`: S = -Gamma(mom, v).
    fn momentum_source_inertial(
        &self, x: Tensor<S, 2>, mom: Tensor<S, 2>, vel: Tensor<S, 2>,
    ) -> Tensor<S, 2> {
        let r = x[0];
        let mr = mom[0];
        let mp = mom[1];
        let vp = vel[1];
        Tensor::new([
            mp * vp / r,
            S::ZERO - mr * vp / r,
        ])
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
    use std::f64::consts::{PI, FRAC_PI_2, FRAC_PI_4};

    fn approx(a: f64, b: f64) -> bool {
        let diff = (a - b).abs();
        if diff < 1e-14 { return true; }
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
                assert!(approx(product[(ii, jj)], expected),
                    "g * g_inv [{}, {}] = {}, expected {}",
                    ii, jj, product[(ii, jj)], expected);
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
            assert!(approx(v[ii], v2[ii]),
                "component {}: {} != {}", ii, v[ii], v2[ii]);
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
        assert!(approx(norm_sq, norm_sq_direct),
            "contraction: {} != quadratic: {}", norm_sq, norm_sq_direct);
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
        assert!(approx(s[0], expected_r),
            "S_r: {} != {}", s[0], expected_r);
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
        assert!(approx(s[1], expected_t),
            "S_t: {} != {}", s[1], expected_t);
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
        let s1 = Cylindrical.momentum_source(
            Tensor::new([r]), 1.0, Tensor::new([1.0]), p);
        let s3 = Cylindrical.momentum_source(
            Tensor::new([r, 0.0, 0.0]), 1.0, Tensor::new([1.0, 0.0, 0.0]), p);
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
        let s = Spherical.momentum_source_inertial(Tensor::new([2.0]), Tensor::new([0.5]), Tensor::new([0.5]));
        assert!(approx(s[0], 0.0));
    }

    #[test]
    fn test_spherical_2d_inertial() {
        let r = 3.0;
        let rho = 2.0;
        let vr = 1.0;
        let vt = 0.5;
        let x = Tensor::new([r, PI / 4.0]);
        let s = Spherical.momentum_source_inertial(x, Tensor::new([rho * vr, rho * vt]), Tensor::new([vr, vt]));
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
        let inertial = Spherical.momentum_source_inertial(x, Tensor::new([rho * vr, rho * vt]), vel);
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
        let s = Spherical.momentum_source_inertial(x, Tensor::new([rho * vel[0], rho * vel[1], rho * vel[2]]), vel);
        let vr = vel[0]; let vt = vel[1]; let vp = vel[2];
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
        let inertial = Spherical.momentum_source_inertial(x, Tensor::new([rho * vel[0], rho * vel[1], rho * vel[2]]), vel);
        assert!(approx(full[2], inertial[2]));
    }

    #[test]
    fn test_cylindrical_1d_inertial_zero() {
        let s = Cylindrical.momentum_source_inertial(Tensor::new([4.0]), Tensor::new([0.5]), Tensor::new([0.5]));
        assert!(approx(s[0], 0.0));
    }

    #[test]
    fn test_cylindrical_3d_inertial() {
        let r = 3.0;
        let rho = 2.0;
        let vr = 0.5;
        let vp = 0.8;
        let x = Tensor::new([r, 0.0, 0.0]);
        let s = Cylindrical.momentum_source_inertial(x, Tensor::new([rho * vr, rho * vp, rho * 1.0]), Tensor::new([vr, vp, 1.0]));
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

    #[test]
    fn test_schwarzschild_lapse_and_metric_3d() {
        // M = 1, r = 5 -> f = 1 - 2/5 = 0.6 (outside the horizon r = 2M = 2).
        let bh = Schwarzschild { mass: 1.0 };
        let (r, theta) = (5.0_f64, FRAC_PI_4);
        let f = 1.0 - 2.0 / r; // 0.6
        let st = theta.sin();
        let x = Tensor::new([r, theta, 0.3]);

        // lapse alpha = sqrt(f).
        assert!(approx(bh.lapse(x), f.sqrt()));
        // diagonal gamma = (1/f, r^2, r^2 sin^2 theta).
        let g = bh.spatial_metric(x);
        assert!(approx(g[(0, 0)], 1.0 / f));
        assert!(approx(g[(1, 1)], r * r));
        assert!(approx(g[(2, 2)], r * r * st * st));
        // gamma^{ij} gamma_{jk} = delta (diagonal).
        let gi = bh.spatial_metric_inv(x);
        for ii in 0..3 {
            assert!(approx(g[(ii, ii)] * gi[(ii, ii)], 1.0));
        }
        // sqrt(gamma) = r^2 sin(theta) / sqrt(f).
        assert!(approx(bh.sqrt_det_gamma(x), r * r * st / f.sqrt()));
    }

    #[test]
    fn test_schwarzschild_sqrt_minus_g_is_flat_spherical_area() {
        // the B3 densitization GIFT: sqrt(-g) = alpha * sqrt(gamma) = r^2 sin(theta), the FLAT
        // spherical area element — so GR flux face areas are unchanged; only the time-volume
        // (1/sqrt(f)) and the source pick up the lapse.
        let bh = Schwarzschild { mass: 0.7 };
        let (r, theta) = (6.0, FRAC_PI_4);
        let x = Tensor::new([r, theta, 0.0]);
        assert!(approx(bh.lapse(x) * bh.volume_factor(x), r * r * theta.sin()));
    }

    #[test]
    fn test_schwarzschild_zero_mass_equals_spherical() {
        // M = 0 -> f = 1 -> the flat spherical metric exactly (lapse 1, gamma = spherical gamma).
        let bh = Schwarzschild { mass: 0.0 };
        let x = Tensor::new([4.0, FRAC_PI_4, 1.1]);
        assert!(approx(bh.lapse(x), 1.0));
        let (g, gs) = (bh.spatial_metric(x), Spherical.spatial_metric(x));
        for ii in 0..3 {
            assert!(approx(g[(ii, ii)], gs[(ii, ii)]));
        }
        assert!(approx(bh.sqrt_det_gamma(x), Spherical.sqrt_det_gamma(x)));
        // the orthogonal axes: spatial geometry is spherical, spacetime is the curved tag.
        assert_eq!(<Schwarzschild as Metric<f64, 3>>::geometry(&bh), Geometry::Spherical);
        assert_eq!(<Schwarzschild as Metric<f64, 3>>::spacetime(&bh), Spacetime::Schwarzschild);
    }

    #[test]
    fn test_schwarzschild_1d_radial() {
        // the radial reduction (the first GR target): M = 1, r = 10 -> f = 0.8.
        let bh = Schwarzschild { mass: 1.0 };
        let x = Tensor::new([10.0_f64]);
        let f = 0.8_f64;
        assert!(approx(bh.lapse(x), f.sqrt()));
        assert!(approx(bh.spatial_metric(x)[(0, 0)], 1.0 / f)); // gamma_rr = 1/f
        // volume_factor incl. the 2 suppressed angular dirs: r^2 / sqrt(f).
        assert!(approx(bh.volume_factor(x), 100.0 / f.sqrt()));
    }
}
