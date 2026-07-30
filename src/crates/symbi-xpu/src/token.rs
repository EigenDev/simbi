// =============================================================================
// token.rs
//
// async operation handle. records an event on a stream after work is
// submitted. used for cross-stream dependencies and completion queries.
// =============================================================================

use crate::error::Result;
use crate::space::ExecutionSpace;

/// async completion handle. wraps a stream event.
pub struct Token<S: ExecutionSpace> {
    event: Option<S::Event>,
}

impl<S: ExecutionSpace> Token<S> {
    /// create a token with a fresh event.
    pub fn create() -> Result<Self> {
        let event = S::create_event()?;
        Ok(Token { event: Some(event) })
    }

    /// record this token's event on the given stream.
    pub fn record(&mut self, stream: &S::Stream) -> Result<()> {
        if let Some(ref event) = self.event {
            S::record_event(event, stream)?;
        }
        Ok(())
    }

    /// block until the recorded event completes.
    pub fn wait(&self) -> Result<()> {
        if let Some(ref event) = self.event {
            S::sync_event(event)?;
        }
        Ok(())
    }

    /// non-blocking query: has the event completed?
    pub fn ready(&self) -> Result<bool> {
        match self.event {
            Some(ref event) => S::event_ready(event),
            None => Ok(true),
        }
    }

    /// make another executor's stream wait on this token.
    pub fn wait_on(&self, other: &crate::executor::Executor<S>) -> Result<()> {
        if let Some(ref event) = self.event {
            S::stream_wait_event(other.stream(), event)?;
        }
        Ok(())
    }
}

impl<S: ExecutionSpace> Drop for Token<S> {
    fn drop(&mut self) {
        if let Some(event) = self.event.take() {
            S::destroy_event(event);
        }
    }
}
