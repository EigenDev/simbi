// =============================================================================
// backend.rs
//
// `IoBackend` — the abstract sink/source. one trait, many implementations:
//   - `Hdf5Backend` (production checkpoints)
//   - `JsonBackend` (introspection — dump the tree as JSON, no field data)
//   - `DisplayBackend` (ASCII / symbi-display table renderer)
//   - future: `ZarrBackend`, `VtkBackend`
//
// the contract: `write(path, tree)` walks the borrowed Tree and emits;
// `read(path)` materializes a TreeBuf. by design, every backend uses the
// SAME Tree representation, so adding a backend is a leaf concern, never
// a refactor of the schema or the call sites.
// =============================================================================

use std::path::Path;

use crate::error::Result;
use crate::tree::{Tree, TreeBuf};

/// the abstract serialization backend.
pub trait IoBackend {
    /// emit a Tree to `path`. data is borrowed from the Tree's `DataRef`s;
    /// the backend may write field-by-field without ever materializing the
    /// whole file in memory.
    fn write(&self, path: &Path, tree: &Tree<'_>) -> Result<()>;

    /// read the full file into a TreeBuf. for HDF5 this is straightforward;
    /// for backends that support lazy reads, this materializes everything
    /// (fine for our checkpoint sizes; revisit if/when we ship multi-GB
    /// snapshots).
    fn read(&self, path: &Path) -> Result<TreeBuf>;
}
