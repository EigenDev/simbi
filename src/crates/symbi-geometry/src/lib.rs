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

pub mod block;
pub mod centroid;
pub mod coord_map;
pub mod grhd_source;
pub mod metric;
pub mod motion;

pub use block::{AxisMap, BlockGeometry};
pub use centroid::volume_weighted_centroid;
pub use coord_map::{CellInterval, CoordMap, LogMap, UniformMap};
pub use metric::{
    Cartesian, Cylindrical, CylindricalRPhi, DiagonalMetric, Geometry, KerrKS, KerrKSCartesian,
    KerrKSCylindrical, Metric, SchwarzschildKS, SchwarzschildKSCartesian,
    SchwarzschildKSCylindrical, Spacetime, Spherical,
};
pub use motion::MotionState;
