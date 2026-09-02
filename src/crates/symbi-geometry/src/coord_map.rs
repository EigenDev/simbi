// =============================================================================
// coord_map.rs
//
// one-dimensional coordinate maps: index -> physical coordinate.
// maps are invertible and define the cell structure along one axis.
//
// - UniformMap: linear spacing, x(i) = start + i * dx
// - LogMap: logarithmic spacing, x(i) = start * 10^(i * slope)
//
// usage:
//   let map = UniformMap::new(0.0, 1.0, 100);  // [0, 1] with 100 cells
//   let cell = map.cell(50);                    // cell interval at index 50
//   let jj = map.index_at(0.5);                 // inverse map
// =============================================================================

use symbi_carrier::Scalar;

/// cell interval: physical extent of a single cell along one axis.
#[derive(Debug, Clone, Copy)]
pub struct CellInterval<S> {
    pub lo: S,
    pub hi: S,
    pub center: S,
    pub width: S,
}

/// coordinate map trait: index -> physical coordinate for one axis.
pub trait CoordMap<S: Scalar>: Clone + Send + Sync {
    /// full cell interval at index ii.
    fn cell(&self, ii: isize) -> CellInterval<S>;

    /// left face position of cell ii.
    fn face(&self, ii: isize) -> S;

    /// cell center position.
    fn center(&self, ii: isize) -> S;

    /// cell width (hi - lo).
    fn width(&self, ii: isize) -> S;

    /// inverse map: physical position -> cell index (floor).
    fn index_at(&self, x: S) -> isize;

    /// number of cells.
    fn num_cells(&self) -> usize;
}

// ============================================================
// uniform map: x(i) = start + i * dx
// ============================================================

#[derive(Debug, Clone, Copy)]
pub struct UniformMap<S> {
    start: S,
    dx: S,
    n: usize,
}

impl<S: Scalar> UniformMap<S> {
    pub fn new(start: S, end: S, n: usize) -> Self {
        let dx = (end - start) / S::from_f64(n as f64);
        Self { start, dx, n }
    }

    pub fn dx(&self) -> S {
        self.dx
    }

    pub fn start(&self) -> S {
        self.start
    }

    pub fn end(&self) -> S {
        self.start + self.dx * S::from_f64(self.n as f64)
    }
}

impl<S: Scalar> CoordMap<S> for UniformMap<S> {
    fn cell(&self, ii: isize) -> CellInterval<S> {
        let lo = self.face(ii);
        let hi = self.face(ii + 1);
        let half = S::HALF;
        CellInterval {
            lo,
            hi,
            center: (lo + hi) * half,
            width: hi - lo,
        }
    }

    fn face(&self, ii: isize) -> S {
        self.start + self.dx * S::from_f64(ii as f64)
    }

    fn center(&self, ii: isize) -> S {
        let half = S::HALF;
        self.start + self.dx * (S::from_f64(ii as f64) + half)
    }

    fn width(&self, _ii: isize) -> S {
        self.dx
    }

    fn index_at(&self, x: S) -> isize {
        ((x - self.start) / self.dx).floor().to_f64() as isize
    }

    fn num_cells(&self) -> usize {
        self.n
    }
}

// ============================================================
// logarithmic map: x(i) = start * 10^(i * slope)
// centroid = geometric mean: sqrt(x_lo * x_hi)
// ============================================================

#[derive(Debug, Clone, Copy)]
pub struct LogMap<S> {
    start: S,
    log_slope: S,
    n: usize,
}

impl<S: Scalar> LogMap<S> {
    /// logarithmic spacing from start to end over n cells.
    /// requires start > 0 and end > start.
    pub fn new(start: S, end: S, n: usize) -> Self {
        let log_slope = (end / start).log10() / S::from_f64(n as f64);
        Self {
            start,
            log_slope,
            n,
        }
    }
}

impl<S: Scalar> CoordMap<S> for LogMap<S> {
    fn cell(&self, ii: isize) -> CellInterval<S> {
        let lo = self.face(ii);
        let hi = self.face(ii + 1);
        CellInterval {
            lo,
            hi,
            center: (lo * hi).sqrt(), // geometric mean
            width: hi - lo,
        }
    }

    fn face(&self, ii: isize) -> S {
        let ten = S::from_f64(10.0);
        self.start * ten.powf(self.log_slope * S::from_f64(ii as f64))
    }

    fn center(&self, ii: isize) -> S {
        // geometric mean of lo and hi faces
        let lo = self.face(ii);
        let hi = self.face(ii + 1);
        (lo * hi).sqrt()
    }

    fn width(&self, ii: isize) -> S {
        self.face(ii + 1) - self.face(ii)
    }

    fn index_at(&self, x: S) -> isize {
        ((x / self.start).log10() / self.log_slope).floor().to_f64() as isize
    }

    fn num_cells(&self) -> usize {
        self.n
    }
}

// ============================================================
// tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64) -> bool {
        if a == b {
            return true;
        }
        let scale = a.abs().max(b.abs()).max(1e-30);
        (a - b).abs() / scale < 1e-12
    }

    // ---- uniform map ----

    #[test]
    fn test_uniform_basic() {
        let map = UniformMap::new(0.0, 1.0, 10);
        assert!(approx(map.dx(), 0.1));
        assert_eq!(map.num_cells(), 10);
    }

    #[test]
    fn test_uniform_face() {
        let map = UniformMap::new(0.0, 1.0, 10);
        assert!(approx(map.face(0), 0.0));
        assert!(approx(map.face(5), 0.5));
        assert!(approx(map.face(10), 1.0));
    }

    #[test]
    fn test_uniform_center() {
        let map = UniformMap::new(0.0, 1.0, 10);
        assert!(approx(map.center(0), 0.05));
        assert!(approx(map.center(5), 0.55));
        assert!(approx(map.center(9), 0.95));
    }

    #[test]
    fn test_uniform_cell() {
        let map = UniformMap::new(0.0, 1.0, 10);
        let c = map.cell(3);
        assert!(approx(c.lo, 0.3));
        assert!(approx(c.hi, 0.4));
        assert!(approx(c.center, 0.35));
        assert!(approx(c.width, 0.1));
    }

    #[test]
    fn test_uniform_width_constant() {
        let map = UniformMap::new(0.0, 1.0, 100);
        for ii in 0..100 {
            assert!(approx(map.width(ii), 0.01));
        }
    }

    #[test]
    fn test_uniform_index_at() {
        let map = UniformMap::new(0.0, 1.0, 10);
        assert_eq!(map.index_at(0.0), 0);
        assert_eq!(map.index_at(0.05), 0);
        assert_eq!(map.index_at(0.15), 1);
        assert_eq!(map.index_at(0.99), 9);
    }

    #[test]
    fn test_uniform_roundtrip() {
        // center -> index_at should recover the original index
        let map = UniformMap::new(0.0, 1.0, 100);
        for ii in 0..100 {
            let x = map.center(ii);
            assert_eq!(map.index_at(x), ii as isize);
        }
    }

    #[test]
    fn test_uniform_nonzero_start() {
        let map = UniformMap::new(2.0, 5.0, 30);
        assert!(approx(map.face(0), 2.0));
        assert!(approx(map.face(30), 5.0));
        assert!(approx(map.dx(), 0.1));
    }

    #[test]
    fn test_uniform_start_end() {
        let map = UniformMap::new(1.0, 3.0, 20);
        assert!(approx(map.start(), 1.0));
        assert!(approx(map.end(), 3.0));
    }

    // ---- log map ----

    #[test]
    fn test_log_faces_boundaries() {
        let map = LogMap::new(1.0, 100.0, 20);
        assert!(approx(map.face(0), 1.0));
        assert!(approx(map.face(20), 100.0));
    }

    #[test]
    fn test_log_geometric_mean_center() {
        let map = LogMap::new(1.0, 1000.0, 30);
        for ii in 0..30 {
            let c = map.cell(ii);
            let prod: f64 = c.lo * c.hi;
            let geometric_mean = prod.sqrt();
            assert!(approx(c.center, geometric_mean));
        }
    }

    #[test]
    fn test_log_width_increases() {
        // cells get wider as index increases
        let map = LogMap::new(1.0, 100.0, 10);
        for ii in 0..9 {
            assert!(map.width(ii) < map.width(ii + 1));
        }
    }

    #[test]
    fn test_log_index_at() {
        let map = LogMap::new(1.0, 100.0, 20);
        assert_eq!(map.index_at(1.0), 0);
        assert_eq!(map.index_at(100.0), 20);
    }

    #[test]
    fn test_log_roundtrip() {
        let map = LogMap::new(0.1, 1000.0, 50);
        for ii in 0..50 {
            let x = map.center(ii);
            assert_eq!(map.index_at(x), ii as isize);
        }
    }

    #[test]
    fn test_log_num_cells() {
        let map = LogMap::new(1.0, 100.0, 42);
        assert_eq!(map.num_cells(), 42);
    }

    #[test]
    fn test_log_face_spacing_is_geometric() {
        // ratio of consecutive face positions should be constant
        let map = LogMap::new(1.0, 100.0, 10);
        let ratio = map.face(1) / map.face(0);
        for ii in 1..10 {
            let r = map.face(ii + 1) / map.face(ii);
            assert!(approx(r, ratio));
        }
    }
}
