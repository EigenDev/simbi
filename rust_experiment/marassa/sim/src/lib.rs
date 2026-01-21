// =============================================================================
// sim - simulation state
//
// production-grade state management matching c++ ecs exactly.
// complete field structure with conserved, primitive, fluxes, mhd, workspace.
//
// architecture:
//   worldstate = amr level hierarchy
//   levelstate = collection of partition states
//   partitionstate = soa fields on device
//   halo graph = communication topology
//
// mathematical structure:
//   state = ⊕ᵢ partition(Dᵢ)  (direct sum over devices)
//   Φ = boundary ∘ update ∘ flux ∘ reconstruct
//
// usage:
//   let level = LevelState::single_partition(&device, mesh_config, 0, false)?;
//   level.partitions[0].conserved.den // access fields
// =============================================================================

pub mod level;
pub mod partition;

// legacy modules (deprecated, will be removed)
pub mod entity;
pub mod metadata;
pub mod registry;
pub mod simulation;
pub mod world;

// primary exports
pub use level::{HaloGraph, HaloLink, LevelState, MeshConfig, Side};
pub use partition::{
    BoundaryInfo, BoundaryType, ConservedFields, DomainSet, FaceConnection, FluxFields, MhdFields,
    PartitionState, PrimitiveFields, RkWorkspace,
};

// legacy exports (deprecated)
pub use entity::Entity;
pub use metadata::Metadata;
pub use registry::Registry;
