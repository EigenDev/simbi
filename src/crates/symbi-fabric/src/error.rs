// =============================================================================
// error.rs
//
// the fabric's error model. every variant names the peer and, where one
// exists, the epoch, so a report locates the failure in the step loop. a
// protocol error is a contract violation by a peer or by this worker's own
// sequencing; both terminate the session.
//
// usage:
//  fn close(&mut self) -> Result<(), FabricError>
// =============================================================================

use crate::ident::{Epoch, WorkerId};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Deadline {
    Startup,
    Transfer,
    Checkpoint,
    Shutdown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AbortReason {
    LocalNumerics,
    LocalIo,
    Protocol,
    Deadline,
    Panic,
}

#[derive(Debug)]
pub enum FabricError {
    Protocol {
        peer: WorkerId,
        detail: String,
    },
    Disconnected {
        peer: WorkerId,
        epoch: Option<Epoch>,
    },
    Deadline {
        phase: Deadline,
        peer: Option<WorkerId>,
        epoch: Option<Epoch>,
        pending: usize,
    },
    PeerAborted {
        peer: WorkerId,
        reason: AbortReason,
        detail: String,
    },
    Rendezvous {
        detail: String,
    },
    Io(std::io::Error),
}

impl FabricError {
    pub fn protocol(peer: WorkerId, detail: impl Into<String>) -> Self {
        Self::Protocol {
            peer,
            detail: detail.into(),
        }
    }

    pub fn is_protocol(&self) -> bool {
        matches!(self, Self::Protocol { .. })
    }
}

impl std::fmt::Display for FabricError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Protocol { peer, detail } => {
                write!(f, "protocol violation with {peer:?}: {detail}")
            }
            Self::Disconnected { peer, epoch } => write!(f, "{peer:?} disconnected at {epoch:?}"),
            Self::Deadline {
                phase,
                peer,
                epoch,
                pending,
            } => write!(
                f,
                "{phase:?} deadline expired waiting on {peer:?} at {epoch:?} with {pending} pending"
            ),
            Self::PeerAborted {
                peer,
                reason,
                detail,
            } => {
                write!(f, "{peer:?} aborted ({reason:?}): {detail}")
            }
            Self::Rendezvous { detail } => write!(f, "rendezvous: {detail}"),
            Self::Io(e) => write!(f, "io: {e}"),
        }
    }
}

impl std::error::Error for FabricError {}

impl From<std::io::Error> for FabricError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}
