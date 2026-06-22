// =============================================================================
// block.rs
//
// block geometry: the solver-facing interface to curvilinear coordinates.
// wraps a metric + coordinate maps into functions that operate on grid indices.
//
// the solver never touches the metric directly. it calls:
//   geo.volume(coord)        — cell volume
//   geo.face_area(coord, d)  — face area in direction d
//   geo.centroid(coord)      — cell center in physical coordinates
//   geo.scale_factors(coord) — for gradient reconstruction
//   geo.source(coord, prim)  — geometric momentum source
//
// these compose naturally with the FP toolkit:
//   let source = computation(domain, |c| geo.source(c, &prim.at(c)));
//   let rhs = flux_div.zip(source, |a, b| a + b);
//
// usage:
//   let geo = BlockGeometry::uniform(Cartesian,(&domain, dx);
//   let geo = BlockGeometry::new(Spherical, &maps, &domain);
// =============================================================================

use symbi_algebra::Tensor;
use symbi_ir::algebra::Scalar;
use crate::metric::{DiagonalMetric, Metric};

/// per-axis coordinate map. Copy + GPU-friendly enum.
/// covers uniform and logarithmic spacing.
#[derive(Clone, Copy, Debug)]
pub enum AxisMap {
    /// x(i) = start + i * dx
    Uniform { start: f64, dx: f64 },
    /// x(i) = start * 10^(i * log_slope)
    Log { start: f64, log_slope: f64 },
}

impl AxisMap {
    /// left face position of cell ii.
    #[inline]
    pub fn face(&self, ii: isize) -> f64 {
        match self {
            AxisMap::Uniform { start, dx } => start + ii as f64 * dx,
            AxisMap::Log { start, log_slope } => start * 10.0_f64.powf(ii as f64 * log_slope),
        }
    }

    /// cell center position.
    #[inline]
    pub fn center(&self, ii: isize) -> f64 {
        match self {
            AxisMap::Uniform { start, dx } => start + (ii as f64 + 0.5) * dx,
            AxisMap::Log { .. } => {
                let lo = self.face(ii);
                let hi = self.face(ii + 1);
                (lo * hi).sqrt() // geometric mean
            }
        }
    }

    /// cell width.
    #[inline]
    pub fn width(&self, ii: isize) -> f64 {
        self.face(ii + 1) - self.face(ii)
    }

    /// is this a uniform map?
    #[inline]
    pub fn is_uniform(&self) -> bool {
        matches!(self, AxisMap::Uniform { .. })
    }
}

impl From<&crate::coord_map::UniformMap<f64>> for AxisMap {
    fn from(m: &crate::coord_map::UniformMap<f64>) -> Self {
        AxisMap::Uniform { start: m.start(), dx: m.dx() }
    }
}

impl From<&crate::coord_map::LogMap<f64>> for AxisMap {
    fn from(m: &crate::coord_map::LogMap<f64>) -> Self {
        use crate::coord_map::CoordMap;
        AxisMap::Log {
            start: m.face(0),
            log_slope: (m.face(1) / m.face(0)).log10(),
        }
    }
}

/// block geometry: metric + coordinate maps bound to a grid.
/// converts grid indices to physical quantities (volumes, areas, sources).
///
/// `M` is the metric (Cartesian, Spherical, Cylindrical).
/// `S` is the scalar type (f64, f32).
/// `D` is the spatial dimension.
pub struct BlockGeometry<M, S: Scalar, const D: usize> {
    pub metric: M,
    /// cell widths per axis (uniform grid shortcut, ignored when maps are set)
    pub dx: [S; D],
    /// domain lower bounds in physical coordinates (ignored when maps are set)
    pub x_lo: [S; D],
    /// per-axis coordinate maps. when Some, overrides dx/x_lo.
    pub maps: Option<[AxisMap; D]>,
}

impl<M, S: Scalar, const D: usize> BlockGeometry<M, S, D>
where
    M: Metric<S, D> + Copy,
{
    /// create a block geometry from a metric and uniform grid spacing.
    pub fn uniform(metric: M, x_lo: [S; D], dx: [S; D]) -> Self {
        BlockGeometry { metric, dx, x_lo, maps: None }
    }

    /// create a block geometry from a metric and per-axis coordinate maps.
    pub fn with_maps(metric: M, maps: [AxisMap; D]) -> Self {
        // derive dx and x_lo from maps (for backward compatibility)
        let dx = std::array::from_fn(|ax| S::from_f64(maps[ax].width(0)));
        let x_lo = std::array::from_fn(|ax| S::from_f64(maps[ax].face(0)));
        BlockGeometry { metric, dx, x_lo, maps: Some(maps) }
    }

    /// cell width along axis ax at grid index.
    #[inline]
    pub fn cell_width(&self, idx: [isize; D], ax: usize) -> S {
        if let Some(ref maps) = self.maps {
            S::from_f64(maps[ax].width(idx[ax]))
        } else {
            self.dx[ax]
        }
    }

    /// physical coordinate of cell center at grid index.
    #[inline]
    pub fn centroid(&self, idx: [isize; D]) -> Tensor<S, D> {
        if let Some(ref maps) = self.maps {
            Tensor::new(std::array::from_fn(|ax| S::from_f64(maps[ax].center(idx[ax]))))
        } else {
            Tensor::new(std::array::from_fn(|ax| {
                self.x_lo[ax] + S::from_f64(idx[ax] as f64 + 0.5) * self.dx[ax]
            }))
        }
    }

    /// physical coordinate of face at grid index in direction `dir`.
    #[inline]
    pub fn face_position(&self, idx: [isize; D], dir: usize) -> Tensor<S, D> {
        if let Some(ref maps) = self.maps {
            let mut x = self.centroid(idx);
            x[dir] = S::from_f64(maps[dir].face(idx[dir]));
            x
        } else {
            let mut x = self.centroid(idx);
            let half = S::from_f64(0.5);
            x[dir] = x[dir] - half * self.dx[dir];
            x
        }
    }

    /// left face position on axis ax at grid index.
    #[inline]
    fn axis_face(&self, idx: [isize; D], ax: usize) -> S {
        if let Some(ref maps) = self.maps {
            S::from_f64(maps[ax].face(idx[ax]))
        } else {
            self.x_lo[ax] + S::from_f64(idx[ax] as f64) * self.dx[ax]
        }
    }

    /// cell volume via 2-point Gauss quadrature of the metric volume factor.
    /// V = integral of volume_factor(x) * dx^1 * ... * dx^D over the cell.
    ///
    /// uses 2-point Gauss-Legendre per axis (exact for polynomials up to degree 3).
    /// for flat spherical (volume_factor = r^2 sin(theta)):
    ///   - r^2 is degree 2 → exact with 2-point Gauss in r
    ///   - sin(theta) is transcendental → O(dx^4) accurate (excellent)
    /// for Kerr-Schild (volume_factor includes sqrt(1+2M/r)):
    ///   - O(dx^4) accurate, consistent with the second-order PDE solver
    ///
    /// this replaces the previous hardcoded per-geometry formulas with a single
    /// general path that works for ANY metric.
    #[inline]
    pub fn volume(&self, idx: [isize; D]) -> S {
        // 2-point Gauss-Legendre: points at +/- 1/sqrt(3), weights = 1
        let offset = S::from_f64(0.5 / 3.0_f64.sqrt());

        // for D dimensions, we have 2^D quadrature points.
        // iterate over all combinations via bit mask.
        let n_quad = 1_usize << D;
        let mut vol = S::ZERO;

        for qq in 0..n_quad {
            let mut x = Tensor::<S, D>::zeros();
            for ax in 0..D {
                let x_center = self.axis_face(idx, ax) + self.cell_width(idx, ax) * S::from_f64(0.5);
                let dx_half = self.cell_width(idx, ax) * offset;
                let sign = if (qq >> ax) & 1 == 0 { S::ONE } else { -S::ONE };
                x[ax] = x_center + sign * dx_half;
            }
            vol = vol + self.metric.volume_factor(x);
        }
        // average over 2^D points, multiply by cell coordinate volume
        vol = vol / S::from_f64(n_quad as f64);
        for ax in 0..D {
            vol = vol * self.cell_width(idx, ax);
        }
        vol
    }

    /// face area in direction `dir`.
    /// area = volume_factor_on_face * product of dx_perp.
    /// uses the full volume factor (not divided by scale_factor[dir]) to
    /// maintain discrete compatibility with the CT curl formulas.
    #[inline]
    pub fn face_area(&self, idx: [isize; D], dir: usize) -> S {
        let x = self.face_position(idx, dir);
        let vf = self.metric.volume_factor(x);
        let mut area = vf;
        for ax in 0..D {
            if ax != dir {
                area = area * self.cell_width(idx, ax);
            }
        }
        area
    }

    /// lab-frame (physical) volume for ALE moving meshes.
    /// `vol_phys = vol_com * a^n` where n = number of scaling dimensions.
    /// spherical: a^3 (only r scales, but volume ~ r^3).
    /// cylindrical: a^2 (only r scales, volume ~ r^2).
    /// cartesian: a^D (all axes scale).
    /// for static meshes (a=1), returns comoving volume.
    #[inline]
    pub fn labframe_volume(&self, idx: [isize; D], a: S) -> S {
        use crate::metric::Geometry;
        let v = self.volume(idx);
        match self.metric.geometry() {
            Geometry::Spherical => v * a * a * a,
            Geometry::Cylindrical => v * a * a,
            Geometry::Cartesian => {
                let mut scale = S::ONE;
                for _ in 0..D { scale = scale * a; }
                v * scale
            }
        }
    }

    /// lab-frame (physical) face area for ALE moving meshes.
    /// spherical: all faces scale as a^2 (face area ~ r^2).
    /// cylindrical (r, phi, z): r-face and z-face scale as a, phi-face doesn't.
    /// cartesian: a^(D-1).
    #[inline]
    pub fn labframe_face_area(&self, idx: [isize; D], dir: usize, a: S) -> S {
        use crate::metric::Geometry;
        let area = self.face_area(idx, dir);
        match self.metric.geometry() {
            Geometry::Spherical => area * a * a,
            Geometry::Cylindrical => {
                // dir=0 (r): face area ~ r → scale by a
                // dir=1 (phi): no radial dependence → no scaling
                // dir=2 (z): face area ~ r → scale by a
                if dir == 1 { area } else { area * a }
            }
            Geometry::Cartesian => {
                let mut scale = S::ONE;
                for _ in 0..D.saturating_sub(1) { scale = scale * a; }
                area * scale
            }
        }
    }

    /// scale factors at cell center. requires a [`DiagonalMetric`] — `h_i` is only defined for a
    /// diagonal metric (the method-level bound keeps this localized: only callers of `scale_factors`
    /// inherit it, not all of `BlockGeometry`).
    #[inline]
    pub fn scale_factors(&self, idx: [isize; D]) -> Tensor<S, D>
    where
        M: DiagonalMetric<S, D>,
    {
        let x = self.centroid(idx);
        self.metric.scale_factors(x)
    }

    /// geometric momentum source term at a cell (continuous analytical formula).
    /// for discrete schemes, prefer `momentum_source_inertial` + discrete
    /// pressure source from face area differences.
    #[inline]
    pub fn momentum_source(
        &self, idx: [isize; D], rho: S, vel: Tensor<S, D>, pre: S,
    ) -> Tensor<S, D> {
        let x = self.centroid(idx);
        self.metric.momentum_source(x, rho, vel, pre)
    }

    /// inertial (velocity-dependent) part of geometric momentum source.
    /// pressure part must be computed separately from discrete face areas:
    ///   S_pressure[i] = p * (A^i_R - A^i_L) / V
    #[inline]
    pub fn momentum_source_inertial(
        &self, idx: [isize; D], rho: S, vel: Tensor<S, D>,
    ) -> Tensor<S, D> {
        let x = self.centroid(idx);
        self.metric.momentum_source_inertial(x, rho, vel)
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::metric::{Cartesian, Spherical, Cylindrical};

    fn approx(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-12 * a.abs().max(b.abs()).max(1.0)
    }

    #[test]
    fn cartesian_volume_1d() {
        let geo = BlockGeometry::uniform(Cartesian, [0.0], [0.1]);
        let vol = geo.volume([5]);
        assert!(approx(vol, 0.1));
    }

    #[test]
    fn cartesian_volume_2d() {
        let geo = BlockGeometry::uniform(Cartesian, [0.0, 0.0], [0.1, 0.2]);
        let vol = geo.volume([5, 3]);
        assert!(approx(vol, 0.1 * 0.2));
    }

    #[test]
    fn cartesian_volume_3d() {
        let geo = BlockGeometry::uniform(Cartesian, [0.0, 0.0, 0.0], [0.1, 0.2, 0.3]);
        let vol = geo.volume([0, 0, 0]);
        assert!(approx(vol, 0.1 * 0.2 * 0.3));
    }

    #[test]
    fn cartesian_face_area_2d() {
        let geo = BlockGeometry::uniform(Cartesian, [0.0, 0.0], [0.1, 0.2]);
        // face in x-direction: area = dy
        let ax = geo.face_area([5, 3], 0);
        assert!(approx(ax, 0.2));
        // face in y-direction: area = dx
        let ay = geo.face_area([5, 3], 1);
        assert!(approx(ay, 0.1));
    }

    #[test]
    fn cartesian_centroid_2d() {
        let geo = BlockGeometry::uniform(Cartesian, [1.0, 2.0], [0.5, 0.5]);
        let c = geo.centroid([0, 0]);
        assert!(approx(c[0], 1.25));
        assert!(approx(c[1], 2.25));
    }

    #[test]
    fn cartesian_source_is_zero() {
        let geo = BlockGeometry::uniform(Cartesian, [0.0, 0.0], [0.1, 0.1]);
        let src = geo.momentum_source([5, 5], 1.0, Tensor::new([0.5, -0.3]), 2.5);
        assert!(approx(src[0], 0.0));
        assert!(approx(src[1], 0.0));
    }

    #[test]
    fn spherical_volume_1d() {
        // 1D spherical: exact integral (r_R^3 - r_L^3)/3
        let geo = BlockGeometry::uniform(Spherical, [1.0], [0.1]);
        let vol = geo.volume([0]); // r in [1.0, 1.1]
        let expected = (1.1_f64.powi(3) - 1.0_f64.powi(3)) / 3.0;
        assert!(approx(vol, expected));
    }

    #[test]
    fn spherical_source_1d() {
        // 1D spherical: S_r = 2p/r
        let geo = BlockGeometry::uniform(Spherical, [1.0], [0.1]);
        let src = geo.momentum_source([0], 1.0, Tensor::new([0.0]), 2.5);
        let r = 1.05;
        let expected = 2.0 * 2.5 / r;
        assert!(approx(src[0], expected));
    }

    #[test]
    fn cylindrical_volume_1d() {
        // 1D cylindrical: exact integral (r_R^2 - r_L^2)/2
        let geo = BlockGeometry::uniform(Cylindrical, [1.0], [0.1]);
        let vol = geo.volume([0]); // r in [1.0, 1.1]
        let expected = (1.1_f64.powi(2) - 1.0_f64.powi(2)) / 2.0;
        assert!(approx(vol, expected));
    }

    #[test]
    fn cylindrical_source_1d() {
        // 1D cylindrical: S_r = p/r
        let geo = BlockGeometry::uniform(Cylindrical, [1.0], [0.1]);
        let src = geo.momentum_source([0], 1.0, Tensor::new([0.0]), 2.5);
        let r = 1.05;
        assert!(approx(src[0], 2.5 / r));
    }

    #[test]
    fn geometry_composes_with_flux_divergence() {
        // conceptual test: show that geometry integrates with the RHS pattern
        let geo = BlockGeometry::uniform(Cartesian, [0.0, 0.0], [0.1, 0.1]);

        // the RHS pattern: for each cell, compute flux_div + source
        let idx = [5, 5];
        let vol = geo.volume(idx);
        let area_x = geo.face_area(idx, 0);
        let area_y = geo.face_area(idx, 1);
        let src = geo.momentum_source(idx, 1.0, Tensor::new([0.0, 0.0]), 1.0);

        // cartesian: vol = dx*dy, area = dy (or dx), source = 0
        assert!(approx(vol, 0.01));
        assert!(approx(area_x, 0.1));
        assert!(approx(area_y, 0.1));
        assert!(approx(src[0], 0.0));
    }
}
