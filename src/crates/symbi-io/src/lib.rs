// =============================================================================
// symbi-io
//
// **schema-driven serialization for symbi**. one `Tree` (the schema graph)
// feeds every output channel:
//
//   • `Hdf5Backend`  — production checkpoint files
//   • `JsonBackend`  — schema introspection (`schema_json`)
//   • `TreeDisplay`  — terminal pretty-print
//   • `symbi-display`-side table renderer (via `Tree::iter_attrs`)
//
// the writer and reader walk the SAME Tree — no parallel mirror code, no
// hand-spelled field names in two places. typed `Attr` + `Metadata::with(..)`
// fluent builder kills the `to_string()` boilerplate that used to litter
// every example. `field_layout` exposes the canonical on-disk naming so it
// derives from `RegimeSpec.fields` instead of being duplicated in the writer.
//
// design references:
//   - error.rs       : `IoError` (proper enum, replaces `Result<(), String>`)
//   - attr.rs        : `Attr` typed scalar + `Metadata` fluent builder
//   - tree.rs        : `Tree<'a>` / `TreeBuf` / `Dataset` / `DType` / `DataRef`
//   - backend.rs     : `IoBackend` trait
//   - hdf5.rs        : `Hdf5Backend` (production)
//   - json.rs        : `schema_json` (introspection)
//   - display.rs     : `TreeDisplay` (terminal ASCII tree)
//   - field_layout.rs: `RegimeSpec`-driven canonical naming
// =============================================================================

pub mod error;
pub mod attr;
pub mod tree;
pub mod backend;
pub mod hdf5;
pub mod json;
pub mod display;
pub mod field_layout;

pub use error::{IoError, Result};
pub use attr::{Attr, Metadata};
pub use tree::{DataBuf, DataRef, DType, Dataset, DatasetBuf, Tree, TreeBuf};
pub use backend::IoBackend;
pub use hdf5::Hdf5Backend;
pub use json::schema_json;
pub use display::{display_tree_buf, TreeBufDisplay, TreeDisplay};
pub use field_layout::{component_count, dataset_name, iter_components};
