// =============================================================================
// symbi-geometry
//
// coordinate geometry for computational physics. provides coordinate maps,
// metric tensors, and the mathematical machinery for finite volume methods
// in curvilinear and curved spacetimes.
//
// designed around the 3+1 ADM decomposition: flat-space metrics (cartesian,
// spherical, cylindrical) are special cases. extending to full GRMHD
// (schwarzschild, kerr) requires only new Metric trait implementations.
//
// usage:
//   use symbi_geometry::{UniformMap, Spherical, Metric, CoordMap};
// =============================================================================

pub mod coord_map;
pub mod metric;
pub mod motion;
pub mod block;

pub use coord_map::{CellInterval, CoordMap, UniformMap, LogMap};
pub use metric::{DiagonalMetric, Metric, Geometry, Cartesian, Spherical, Cylindrical};
pub use motion::MotionState;
pub use block::{BlockGeometry, AxisMap};
