// =============================================================================
// symbi-fabric/src/lib.rs
//
// the process-to-process runtime beneath a decomposed run: wire identities,
// framing, the receiver-grant table, the per-worker endpoint with its send and
// receive state machines, and the error model. the crate moves byte buffers
// and f64 bit patterns between workers and names no field, region, or mesh;
// the simulation layer packs and unpacks.
//
// dependency floor: std only. every type that crosses the wire is defined
// here and imported by the simulation layer, never the reverse.
//
// usage:
//  use symbi_fabric::{Fabric, Loopback, PhaseSpec, Progress};
//  use symbi_fabric::{Digest, Epoch, Fnv, TransferId, WorkerId};
// =============================================================================

pub mod endpoint;
pub mod error;
pub mod frame;
pub mod grant;
pub mod ident;

pub use endpoint::{Fabric, Link, Loopback, PhaseSpec, Progress, RecvState, SendState};
pub use error::{AbortReason, Deadline, FabricError};
pub use frame::{HEADER_LEN, Header, Kind, MAGIC, VERSION};
pub use grant::GrantTable;
pub use ident::{Digest, Epoch, Fnv, OpId, SessionId, TransferId, WorkerId};
