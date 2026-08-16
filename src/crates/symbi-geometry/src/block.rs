// =============================================================================
// block.rs
//
// block geometry: the solver-facing interface to curvilinear coordinates.
// wraps a metric + coordinate maps into functions that operate on grid indices.
//
// the solver reaches the metric through this interface alone. it calls:
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

use crate::metric::{DiagonalMetric, Metric};
use symbi_algebra::Tensor;
use symbi_ir::algebra::Scalar;

/// per-axis coordinate map. Copy + GPU-friendly enum.
/// covers uniform, logarithmic, and geometrically graded spacing.
#[derive(Clone, Copy, Debug)]
pub enum AxisMap {
    /// x(i) = start + i * dx
    Uniform { start: f64, dx: f64 },
    /// x(i) = start * 10^(i * log_slope)
    Log { start: f64, log_slope: f64 },
    /// cell widths follow `width(i) = width * ratio^i`.
    Geometric { start: f64, width: f64, ratio: f64 },
}

impl AxisMap {
    /// left face position of cell ii.
    #[inline]
    pub fn face(&self, ii: isize) -> f64 {
        match self {
            AxisMap::Uniform { start, dx } => start + ii as f64 * dx,
            AxisMap::Log { start, log_slope } => start * 10.0_f64.powf(ii as f64 * log_slope),
            AxisMap::Geometric {
                start,
                width,
                ratio,
            } => {
                if (*ratio - 1.0).abs() < 1.0e-12 {
                    start + ii as f64 * width
                } else {
                    start + width * (ratio.powf(ii as f64) - 1.0) / (ratio - 1.0)
                }
            }
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
            AxisMap::Geometric { .. } => 0.5 * (self.face(ii) + self.face(ii + 1)),
        }
    }

    /// cell width.
    #[inline]
    pub fn width(&self, ii: isize) -> f64 {
        self.face(ii + 1) - self.face(ii)
    }

    /// inverse map: physical position to the containing cell index.
    #[inline]
    pub fn index_at(&self, x: f64) -> isize {
        match self {
            AxisMap::Uniform { start, dx } => ((x - start) / dx).floor() as isize,
            AxisMap::Log { start, log_slope } => ((x / start).log10() / log_slope).floor() as isize,
            AxisMap::Geometric {
                start,
                width,
                ratio,
            } => {
                if (*ratio - 1.0).abs() < 1.0e-12 {
                    ((x - start) / width).floor() as isize
                } else {
                    (1.0 + (x - start) * (ratio - 1.0) / width)
                        .log(*ratio)
                        .floor() as isize
                }
            }
        }
    }

    /// is this a uniform map?
    #[inline]
    pub fn is_uniform(&self) -> bool {
        matches!(self, AxisMap::Uniform { .. })
    }
}

impl From<&crate::coord_map::UniformMap<f64>> for AxisMap {
    fn from(m: &crate::coord_map::UniformMap<f64>) -> Self {
        AxisMap::Uniform {
            start: m.start(),
            dx: m.dx(),
        }
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

#[cfg(test)]
mod axis_map_tests {
    use super::AxisMap;

    #[test]
    fn geometric_faces_reach_extent_and_follow_width_ratio() {
        let ratio = 0.8_f64;
        let cells = 8_i32;
        let extent = 3.0_f64;
        let width = extent * (ratio - 1.0) / (ratio.powi(cells) - 1.0);
        let map = AxisMap::Geometric {
            start: 2.0,
            width,
            ratio,
        };

        assert!((map.face(cells as isize) - 5.0).abs() < 1.0e-12);
        for ii in 0..cells - 1 {
            assert!((map.width(ii as isize + 1) / map.width(ii as isize) - ratio).abs() < 1.0e-12);
        }
        for ii in 0..cells {
            assert_eq!(map.index_at(map.center(ii as isize)), ii as isize);
        }
    }

    #[test]
    fn shifted_geometric_map_matches_global_faces() {
        let global = AxisMap::Geometric {
            start: -1.0,
            width: 0.3,
            ratio: 1.1,
        };
        let offset = 7_isize;
        let local = AxisMap::Geometric {
            start: global.face(offset),
            width: global.width(offset),
            ratio: 1.1,
        };

        for ii in -2..6 {
            assert!((local.face(ii) - global.face(offset + ii)).abs() < 1.0e-12);
        }
    }

    #[test]
    fn homologous_scaling_preserves_geometric_grading() {
        let map = AxisMap::Geometric {
            start: 1.5,
            width: 0.2,
            ratio: 0.93,
        };
        let scale = 2.4;
        let physical = AxisMap::Geometric {
            start: 1.5 * scale,
            width: 0.2 * scale,
            ratio: 0.93,
        };

        for ii in -2..12 {
            assert!((physical.face(ii) - scale * map.face(ii)).abs() < 1.0e-12);
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
    /// which coordinate slot each grid axis resolves — `[0, 2]` for a cylindrical (R, z) plane,
    /// identity otherwise. carried because the cell volume depends on which coordinates the grid
    /// leaves unresolved, and the chart alone leaves that open: cylindrical (R, phi) and (R, z) are
    /// both 2d cylindrical and leave different coordinates ungridded.
    pub axes: [usize; D],
}

impl<M, S: Scalar, const D: usize> BlockGeometry<M, S, D>
where
    M: Metric<S, D> + Copy,
{
    /// create a block geometry from a metric and uniform grid spacing.
    pub fn uniform(metric: M, x_lo: [S; D], dx: [S; D], axes: [usize; D]) -> Self {
        BlockGeometry {
            axes,
            metric,
            dx,
            x_lo,
            maps: None,
        }
    }

    /// create a block geometry from a metric and per-axis coordinate maps.
    pub fn with_maps(metric: M, maps: [AxisMap; D], axes: [usize; D]) -> Self {
        // derive dx and x_lo from maps (for backward compatibility)
        let dx = std::array::from_fn(|ax| S::from_f64(maps[ax].width(0)));
        let x_lo = std::array::from_fn(|ax| S::from_f64(maps[ax].face(0)));
        BlockGeometry {
            metric,
            dx,
            x_lo,
            maps: Some(maps),
            axes,
        }
    }

    /// the measure of the coordinates the grid leaves unresolved.
    ///
    /// an ungridded axis contributes its full physical extent exactly when that extent is fixed by
    /// the symmetry the reduction assumes — which holds iff the coordinate is an angle, and fails
    /// iff it is a length. an angle has a compact range the symmetry pins down; a length is open,
    /// since "the" transverse extent of a slab is undefined.
    ///
    ///   spherical theta   int sin(theta) dtheta over [0, pi] = 2
    ///   spherical phi     2 pi
    ///   cylindrical phi   2 pi
    ///   cartesian x/y/z   1     (the result is per unit length / area)
    ///   cylindrical z     1     (per unit length)
    ///
    /// so a 1d radial spherical cell is the whole shell (2 * 2 pi = 4 pi), a 2d (r, theta) cell is
    /// the revolved annulus (2 pi), a 2d cylindrical (R, z) cell is the revolved ring (2 pi) — all
    /// exact physical volumes — while cylindrical (R, phi) and every reduced cartesian grid stay
    /// per-unit-length, their thickness left open by the problem.
    pub fn ungridded_measure(&self) -> S {
        use crate::metric::Geometry;
        let two_pi = S::from_f64(std::f64::consts::TAU);
        let mut measure = S::ONE;
        for slot in 0..3usize {
            if self.axes.contains(&slot) {
                continue;
            }
            measure = measure
                * match (self.metric.geometry(), slot) {
                    (Geometry::Spherical, 1) => S::from_f64(2.0),
                    (Geometry::Spherical, 2) => two_pi,
                    (Geometry::Cylindrical, 1) => two_pi,
                    _ => S::ONE,
                };
        }
        measure
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
            Tensor::new(std::array::from_fn(|ax| {
                S::from_f64(maps[ax].center(idx[ax]))
            }))
        } else {
            Tensor::new(std::array::from_fn(|ax| {
                self.x_lo[ax] + S::from_f64(idx[ax] as f64 + 0.5) * self.dx[ax]
            }))
        }
    }

    /// physical coordinate of the CT face center in direction `dir`: the `dir` axis on the face,
    /// each transverse axis at the arithmetic midpoint of its cell. this is the point the CT curl
    /// evaluates the metric at (`a_c = (a_lo + a_hi)/2` in the ct_emf curl) and the point `volume`'s
    /// quadrature centers on, so `face_area` telescopes the area-weighted div(B) to machine zero.
    /// on a log axis the arithmetic midpoint departs from the cell `centroid`
    /// (whose centroid is the geometric mean sqrt(r_lo r_hi)); using the geometric mean here would
    /// evaluate the metric a distance O((dr/r)^2) off the curl's point, injecting a spurious ~1e-6
    /// divergence into the diagnostic for a field that is exactly div-free in the scheme.
    #[inline]
    pub fn face_position(&self, idx: [isize; D], dir: usize) -> Tensor<S, D> {
        let half = S::from_f64(0.5);
        Tensor::new(std::array::from_fn(|ax| {
            if ax == dir {
                self.axis_face(idx, dir)
            } else {
                self.axis_face(idx, ax) + half * self.cell_width(idx, ax)
            }
        }))
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
    ///   - r^2 is degree 2 -> exact with 2-point Gauss in r
    ///   - sin(theta) is transcendental -> O(dx^4) accurate (excellent)
    /// for Kerr-Schild (volume_factor includes sqrt(1+2M/r)):
    ///   - O(dx^4) accurate, consistent with the second-order pde solver
    ///
    /// a single general path that works for every metric, the quadrature standing in for
    /// per-geometry formulas.
    #[inline]
    pub fn volume(&self, idx: [isize; D]) -> S {
        // 2-point Gauss-Legendre: points at +/- 1/sqrt(3), weights = 1
        let offset = S::from_f64(0.5 / 3.0_f64.sqrt());

        // for D dimensions there are 2^D quadrature points.
        // iterate over all combinations via bit mask.
        let n_quad = 1_usize << D;
        let mut vol = S::ZERO;

        for qq in 0..n_quad {
            let mut x = Tensor::<S, D>::zeros();
            for ax in 0..D {
                let x_center =
                    self.axis_face(idx, ax) + self.cell_width(idx, ax) * S::from_f64(0.5);
                let dx_half = self.cell_width(idx, ax) * offset;
                let sign = if (qq >> ax) & 1 == 0 { S::ONE } else { -S::ONE };
                x[ax] = x_center + sign * dx_half;
            }
            vol = vol + self.metric.volume_factor(x);
        }
        // average over 2^D points, multiply by cell coordinate volume
        vol = vol / S::from_f64(n_quad as f64);
        // the coordinates the grid leaves unresolved: an angular one contributes its full range, so
        // a reduced-dimension curvilinear cell carries its true physical volume in place of a
        // per-steradian one.
        vol = vol * self.ungridded_measure();
        for ax in 0..D {
            vol = vol * self.cell_width(idx, ax);
        }
        vol
    }

    /// face area in direction `dir`.
    /// area = volume_factor_on_face * product of dx_perp.
    /// uses the full volume factor, undivided by scale_factor[dir], to
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
                for _ in 0..D {
                    scale = scale * a;
                }
                v * scale
            }
        }
    }

    /// lab-frame (physical) face area for ALE moving meshes.
    /// spherical: all faces scale as a^2 (face area ~ r^2).
    /// cylindrical (r, phi, z): r-face and z-face scale as a, phi-face stays fixed.
    /// cartesian: a^(D-1).
    #[inline]
    pub fn labframe_face_area(&self, idx: [isize; D], dir: usize, a: S) -> S {
        use crate::metric::Geometry;
        let area = self.face_area(idx, dir);
        match self.metric.geometry() {
            Geometry::Spherical => area * a * a,
            Geometry::Cylindrical => {
                // dir=0 (r): face area ~ r -> scale by a
                // dir=1 (phi): no radial dependence -> no scaling
                // dir=2 (z): face area ~ r -> scale by a
                if dir == 1 { area } else { area * a }
            }
            Geometry::Cartesian => {
                let mut scale = S::ONE;
                for _ in 0..D.saturating_sub(1) {
                    scale = scale * a;
                }
                area * scale
            }
        }
    }

    /// scale factors at cell center. requires a [`DiagonalMetric`] — `h_i` is only defined for a
    /// diagonal metric (the method-level bound keeps this localized: only callers of `scale_factors`
    /// inherit it).
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
        &self,
        idx: [isize; D],
        rho: S,
        vel: Tensor<S, D>,
        pre: S,
    ) -> Tensor<S, D> {
        let x = self.centroid(idx);
        self.metric.momentum_source(x, rho, vel, pre)
    }

    /// inertial (velocity-dependent) part of geometric momentum source.
    /// pressure part must be computed separately from discrete face areas:
    ///   S_pressure[i] = p * (A^i_R - A^i_L) / V
    #[inline]
    pub fn momentum_source_inertial(
        &self,
        idx: [isize; D],
        mom: Tensor<S, D>,
        vel: Tensor<S, D>,
    ) -> Tensor<S, D> {
        let x = self.centroid(idx);
        self.metric.momentum_source_inertial(x, mom, vel)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metric::{Cartesian, Cylindrical, Spherical};

    fn approx(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-12 * a.abs().max(b.abs()).max(1.0)
    }

    #[test]
    fn cartesian_volume_1d() {
        let geo = BlockGeometry::uniform(Cartesian, [0.0], [0.1], std::array::from_fn(|d| d));
        let vol = geo.volume([5]);
        assert!(approx(vol, 0.1));
    }

    #[test]
    fn cartesian_volume_2d() {
        let geo = BlockGeometry::uniform(
            Cartesian,
            [0.0, 0.0],
            [0.1, 0.2],
            std::array::from_fn(|d| d),
        );
        let vol = geo.volume([5, 3]);
        assert!(approx(vol, 0.1 * 0.2));
    }

    #[test]
    fn cartesian_volume_3d() {
        let geo = BlockGeometry::uniform(
            Cartesian,
            [0.0, 0.0, 0.0],
            [0.1, 0.2, 0.3],
            std::array::from_fn(|d| d),
        );
        let vol = geo.volume([0, 0, 0]);
        assert!(approx(vol, 0.1 * 0.2 * 0.3));
    }

    #[test]
    fn cartesian_face_area_2d() {
        let geo = BlockGeometry::uniform(
            Cartesian,
            [0.0, 0.0],
            [0.1, 0.2],
            std::array::from_fn(|d| d),
        );
        // face in x-direction: area = dy
        let ax = geo.face_area([5, 3], 0);
        assert!(approx(ax, 0.2));
        // face in y-direction: area = dx
        let ay = geo.face_area([5, 3], 1);
        assert!(approx(ay, 0.1));
    }

    #[test]
    fn cartesian_centroid_2d() {
        let geo = BlockGeometry::uniform(
            Cartesian,
            [1.0, 2.0],
            [0.5, 0.5],
            std::array::from_fn(|d| d),
        );
        let c = geo.centroid([0, 0]);
        assert!(approx(c[0], 1.25));
        assert!(approx(c[1], 2.25));
    }

    #[test]
    fn cartesian_source_is_zero() {
        let geo = BlockGeometry::uniform(
            Cartesian,
            [0.0, 0.0],
            [0.1, 0.1],
            std::array::from_fn(|d| d),
        );
        let src = geo.momentum_source([5, 5], 1.0, Tensor::new([0.5, -0.3]), 2.5);
        assert!(approx(src[0], 0.0));
        assert!(approx(src[1], 0.0));
    }

    #[test]
    fn spherical_volume_1d_is_the_whole_shell() {
        // a 1d radial grid leaves both angles unresolved, and the spherical symmetry fixes the full
        // range of each — so the cell is the whole shell, 4 pi (r_R^3 - r_L^3)/3, in place of a
        // per-steradian slice of one. this number is the physical volume of the region the run
        // represents, which is what makes an extensive total over it the actual mass.
        let geo = BlockGeometry::uniform(Spherical, [1.0], [0.1], std::array::from_fn(|d| d));
        let vol = geo.volume([0]); // r in [1.0, 1.1]
        let expected = std::f64::consts::TAU * 2.0 * (1.1_f64.powi(3) - 1.0_f64.powi(3)) / 3.0;
        assert!(approx(vol, expected), "got {vol}, want {expected}");
    }

    #[test]
    fn spherical_source_1d() {
        // 1D spherical: S_r = 2p/r
        let geo = BlockGeometry::uniform(Spherical, [1.0], [0.1], std::array::from_fn(|d| d));
        let src = geo.momentum_source([0], 1.0, Tensor::new([0.0]), 2.5);
        let r = 1.05;
        let expected = 2.0 * 2.5 / r;
        assert!(approx(src[0], expected));
    }

    #[test]
    fn cylindrical_volume_1d_revolves_but_stays_per_unit_height() {
        // a 1d radial cylindrical grid leaves phi and z unresolved, and the two differ in kind: phi
        // is an angle whose full 2 pi the axisymmetry fixes, z is an unbounded length. so the cell
        // revolves (x 2 pi) and stays per unit height — the honest answer while the column's height
        // is left open by the problem.
        let geo = BlockGeometry::uniform(Cylindrical, [1.0], [0.1], std::array::from_fn(|d| d));
        let vol = geo.volume([0]); // r in [1.0, 1.1]
        let expected = std::f64::consts::TAU * (1.1_f64.powi(2) - 1.0_f64.powi(2)) / 2.0;
        assert!(approx(vol, expected), "got {vol}, want {expected}");
    }

    /// the rule, per chart and dimension, so the convention holds at every call site at once.
    /// an ungridded coordinate contributes its full extent iff it is an angle.
    #[test]
    fn the_ungridded_measure_follows_the_angle_length_rule() {
        let tau = std::f64::consts::TAU;
        let id1: [usize; 1] = [0];
        let id2: [usize; 2] = [0, 1];
        let id3: [usize; 3] = [0, 1, 2];

        // 3d resolves every coordinate, so the measure is a bare 1.
        assert_eq!(
            BlockGeometry::<_, f64, 3>::uniform(Spherical, [1.0; 3], [0.1; 3], id3)
                .ungridded_measure(),
            1.0
        );
        assert_eq!(
            BlockGeometry::<_, f64, 3>::uniform(Cartesian, [0.0; 3], [0.1; 3], id3)
                .ungridded_measure(),
            1.0
        );

        // spherical: theta contributes int sin = 2 over its full range, phi contributes 2 pi.
        assert_eq!(
            BlockGeometry::<_, f64, 2>::uniform(Spherical, [1.0; 2], [0.1; 2], id2)
                .ungridded_measure(),
            tau
        );
        assert_eq!(
            BlockGeometry::<_, f64, 1>::uniform(Spherical, [1.0], [0.1], id1).ungridded_measure(),
            2.0 * tau
        );

        // cylindrical: the (R, z) plane leaves the angle unresolved and revolves; the (R, phi)
        // disk leaves a length unresolved and stays per unit thickness. same chart, same dimension,
        // different answer — which is why the axis roles are carried explicitly.
        assert_eq!(
            BlockGeometry::<_, f64, 2>::uniform(Cylindrical, [1.0; 2], [0.1; 2], [0, 2])
                .ungridded_measure(),
            tau
        );
        assert_eq!(
            BlockGeometry::<_, f64, 2>::uniform(Cylindrical, [1.0; 2], [0.1; 2], [0, 1])
                .ungridded_measure(),
            1.0
        );

        // cartesian leaves every extent open: each unresolved coordinate is a length.
        for m in [
            BlockGeometry::<_, f64, 1>::uniform(Cartesian, [0.0], [0.1], id1).ungridded_measure(),
            BlockGeometry::<_, f64, 2>::uniform(Cartesian, [0.0; 2], [0.1; 2], id2)
                .ungridded_measure(),
        ] {
            assert_eq!(m, 1.0);
        }
    }

    #[test]
    fn cylindrical_source_1d() {
        // 1D cylindrical: S_r = p/r
        let geo = BlockGeometry::uniform(Cylindrical, [1.0], [0.1], std::array::from_fn(|d| d));
        let src = geo.momentum_source([0], 1.0, Tensor::new([0.0]), 2.5);
        let r = 1.05;
        assert!(approx(src[0], 2.5 / r));
    }

    #[test]
    fn logradial_divb_telescopes_to_machine_zero() {
        // the div(B) diagnostic (area-weighted, sum_d A_d^+ B_d^+ - A_d^- B_d^-) must read machine
        // zero for a field built div-free by the CT curl. the CT curl evaluates the metric at the
        // arithmetic face center (a_c = (a_lo + a_hi)/2); face_area must do the same. on a log-radial
        // grid the cell centroid is the geometric mean sqrt(r_lo r_hi) != arithmetic, so a face_area
        // that evaluated the metric at the centroid injected a spurious ~1e-6 divergence (the fm
        // torus symptom). regression: face_position uses the arithmetic transverse center.
        let rmap = AxisMap::Log {
            start: 2.0,
            log_slope: 0.05,
        }; // strongly non-uniform radial zones
        let tmap = AxisMap::Uniform {
            start: 0.3,
            dx: 0.15,
        };
        let geo: BlockGeometry<Spherical, f64, 2> =
            BlockGeometry::with_maps(Spherical, [rmap, tmap], [0, 1]);
        let (nr, nth) = (8isize, 6isize);

        // flat-spherical volume factor r^2 sin(theta) at the metric-evaluation point.
        let vf = |r: f64, th: f64| r * r * th.sin();
        // a smooth corner potential A(r, theta); B = curl(A) is metric-div-free by construction.
        let aphi = |i: isize, j: isize| {
            let (r, th) = (rmap.face(i), tmap.face(j));
            (0.4 * r).sin() * (1.3 * th).cos()
        };
        let thc = |j: isize| 0.5 * (tmap.face(j) + tmap.face(j + 1));
        let rc = |i: isize| 0.5 * (rmap.face(i) + rmap.face(i + 1)); // arithmetic (the curl's point)

        // B on the staggered faces, from the CT arithmetic-center curl weights (independent of
        // face_area — this is what the IC / curl produce).
        let br = |i: isize, j: isize| {
            (aphi(i, j + 1) - aphi(i, j)) / (vf(rmap.face(i), thc(j)) * tmap.width(j))
        };
        let bth = |i: isize, j: isize| {
            -(aphi(i + 1, j) - aphi(i, j)) / (vf(rc(i), tmap.face(j)) * rmap.width(i))
        };

        // on a log-spaced radial grid the arithmetic and geometric cell centers differ at O(1e-3), so a
        // center-choice error in the curl shows up at a measurable size.
        let r_arith = rc(3);
        let r_geom = (rmap.face(3) * rmap.face(4)).sqrt();
        assert!(
            (r_arith - r_geom).abs() > 1e-3,
            "log grid must separate arithmetic/geometric centers"
        );

        let mut max_div = 0.0_f64;
        for i in 1..(nr - 1) {
            for j in 1..(nth - 1) {
                let vol = geo.volume([i, j]);
                let flux = geo.face_area([i + 1, j], 0) * br(i + 1, j)
                    - geo.face_area([i, j], 0) * br(i, j)
                    + geo.face_area([i, j + 1], 1) * bth(i, j + 1)
                    - geo.face_area([i, j], 1) * bth(i, j);
                max_div = max_div.max((flux / vol).abs());
            }
        }
        assert!(
            max_div < 1e-12,
            "log-radial area-weighted div(B) = {max_div:e}, expected machine zero"
        );
    }

    #[test]
    fn geometry_composes_with_flux_divergence() {
        // conceptual test: show that geometry integrates with the RHS pattern
        let geo = BlockGeometry::uniform(
            Cartesian,
            [0.0, 0.0],
            [0.1, 0.1],
            std::array::from_fn(|d| d),
        );

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
