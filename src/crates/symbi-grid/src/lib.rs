// =============================================================================
// symbi-grid
//
// field and view types for grid-based computation. a field owns memory
// (via symbi-xpu MemoryBlock) and is bound to a domain (from symbi-algebra).
// reads/writes go through views or coord-indexed at/set; the substrate
// kernels (symbi-discretize -> symbi-aot) operate on this storage directly.
//
// usage:
//   use symbi_grid::Field;
//   let f = Field::<f64, 2>::zeros(&domain)?;
//   let v = f.view();
// =============================================================================

pub mod view;
pub mod field;
pub mod ghost;
pub mod centering;

pub use view::{View, ViewMut};
pub use field::Field;
pub use centering::{Centering, Cell, Face, Edge};
pub use ghost::{
    BcType, FaceSide, GhostType, GhostRegion,
    analyze_ghost_regions, ghost_fill_field, ghost_fill_all,
    periodic_remap, clamp_remap, mirror_remap, build_bc_map,
};
