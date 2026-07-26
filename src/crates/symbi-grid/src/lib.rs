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

pub mod centering;
pub mod field;
pub mod ghost;
pub mod view;

pub use centering::{Cell, Centering, Edge, Face};
pub use field::Field;
pub use ghost::{
    BcType, FaceSide, GhostRegion, GhostType, analyze_ghost_regions, build_bc_map, clamp_remap,
    ghost_fill_all, ghost_fill_field, mirror_remap, periodic_remap,
};
pub use view::{View, ViewMut};
