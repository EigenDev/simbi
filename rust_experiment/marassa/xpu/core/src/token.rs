// =============================================================================
// token.rs
//
// async execution tokens for tracking device operations.
// provides type-safe handles to async work (kernel launches, transfers).
//
// design:
//   - generic over device type (compile-time dispatch)
//   - wraps device-specific event handles
//   - move-only semantics (no cloning events)
//   - immediate tokens for already-completed work
//
// usage:
//   let token = device.record_event()?;
//   // ... do other work ...
//   token.wait()?;  // block until complete
//
//   if token.ready()? {
//       // work is done
//   }
// =============================================================================

use crate::Device;
use core::marker::PhantomData;

/// async execution token for tracking device operations.
/// generic over device type D for compile-time dispatch.
pub struct Token<D: Device> {
    state: TokenState<D>,
    _phantom: PhantomData<D>,
}

/// internal state of a token.
enum TokenState<D: Device> {
    /// token owns an event that tracks async work
    Pending { event: D::Event },
    /// token represents already-completed work (no event)
    Immediate,
    /// token has been consumed/invalidated
    Consumed,
}

impl<D: Device> Token<D> {
    /// creates a token from a device event.
    pub fn from_event(event: D::Event) -> Self {
        Self {
            state: TokenState::Pending { event },
            _phantom: PhantomData,
        }
    }

    /// creates an immediate token (already complete).
    pub fn immediate() -> Self {
        Self {
            state: TokenState::Immediate,
            _phantom: PhantomData,
        }
    }

    /// checks if the async operation is complete.
    /// non-blocking poll.
    pub fn ready(&self) -> Result<bool, D::Error> {
        match &self.state {
            TokenState::Pending { event } => D::event_query(event),
            TokenState::Immediate => Ok(true),
            TokenState::Consumed => Ok(true), // consumed tokens are "done"
        }
    }

    /// waits for the async operation to complete.
    /// blocks the calling thread.
    pub fn wait(mut self) -> Result<(), D::Error> {
        match &self.state {
            TokenState::Pending { event } => {
                D::event_synchronize(event)?;
                self.state = TokenState::Consumed;
                Ok(())
            }
            TokenState::Immediate => Ok(()),
            TokenState::Consumed => Ok(()),
        }
    }

    /// waits without consuming the token.
    /// allows reusing the token for multiple checks.
    pub fn wait_ref(&self) -> Result<(), D::Error> {
        match &self.state {
            TokenState::Pending { event } => D::event_synchronize(event),
            TokenState::Immediate => Ok(()),
            TokenState::Consumed => Ok(()),
        }
    }

    /// returns true if this is an immediate token.
    pub fn is_immediate(&self) -> bool {
        matches!(self.state, TokenState::Immediate)
    }

    /// returns true if token has been consumed.
    pub fn is_consumed(&self) -> bool {
        matches!(self.state, TokenState::Consumed)
    }
}

impl<D: Device> Drop for Token<D> {
    fn drop(&mut self) {
        // if token is dropped while pending, we should synchronize
        // to ensure work completes before event is destroyed.
        // this prevents use-after-free of device resources.
        if let TokenState::Pending { event } = &self.state {
            // best effort synchronization on drop
            // errors are ignored because we're in drop
            let _ = D::event_synchronize(event);
        }
    }
}

// tokens are send/sync if device events are
unsafe impl<D: Device> Send for Token<D> where D::Event: Send {}
unsafe impl<D: Device> Sync for Token<D> where D::Event: Sync {}

// =============================================================================
// free functions for token operations
// =============================================================================

/// waits for all tokens to complete.
pub fn wait_all<D: Device>(tokens: alloc::vec::Vec<Token<D>>) -> Result<(), D::Error> {
    for token in tokens {
        token.wait()?;
    }
    Ok(())
}

/// checks if all tokens are ready.
/// non-blocking poll of all tokens.
pub fn all_ready<D: Device>(tokens: &[Token<D>]) -> Result<bool, D::Error> {
    for token in tokens {
        if !token.ready()? {
            return Ok(false);
        }
    }
    Ok(true)
}

/// waits for any token to complete.
/// returns the index of the first completed token.
/// blocking operation.
pub fn wait_any<D: Device>(tokens: &[Token<D>]) -> Result<usize, D::Error> {
    // simple polling implementation
    // could be optimized with device-specific multi-wait primitives
    loop {
        for (idx, token) in tokens.iter().enumerate() {
            if token.ready()? {
                return Ok(idx);
            }
        }
        // brief sleep to avoid busy-waiting
        // todo: use device-specific wait primitives
        core::hint::spin_loop();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // mock device for testing
    struct MockDevice;
    struct MockEvent {
        ready: core::cell::Cell<bool>,
    }

    // mock buffer type for testing
    struct MockBuffer<T> {
        _phantom: core::marker::PhantomData<T>,
    }

    impl<T> crate::DeviceBuffer<T> for MockBuffer<T> {
        type Device = MockDevice;

        fn len(&self) -> usize {
            0
        }

        fn as_ptr(&self) -> *const T {
            core::ptr::null()
        }

        fn as_mut_ptr(&mut self) -> *mut T {
            core::ptr::null_mut()
        }
    }

    impl Device for MockDevice {
        type Buffer<T> = MockBuffer<T>;
        type Error = &'static str;
        type Event = MockEvent;

        fn id(&self) -> usize {
            0
        }
        fn device_count() -> usize {
            1
        }
        fn new(_id: usize) -> Result<Self, Self::Error> {
            Ok(MockDevice)
        }
        fn alloc<T>(&self, _n: usize) -> Result<Self::Buffer<T>, Self::Error>
        where
            T: Default + Clone,
        {
            Ok(MockBuffer {
                _phantom: core::marker::PhantomData,
            })
        }
        fn alloc_init<T: Clone>(
            &self,
            _n: usize,
            _value: T,
        ) -> Result<Self::Buffer<T>, Self::Error> {
            Ok(MockBuffer {
                _phantom: core::marker::PhantomData,
            })
        }
        fn copy_to_device<T>(
            &self,
            _host_data: &[T],
            _device_buf: &mut Self::Buffer<T>,
        ) -> Result<(), Self::Error>
        where
            T: Clone,
        {
            Ok(())
        }
        fn copy_to_host<T>(
            &self,
            _device_buf: &Self::Buffer<T>,
            _host_data: &mut [T],
        ) -> Result<(), Self::Error>
        where
            T: Clone,
        {
            Ok(())
        }
        fn launch<K, Args>(
            &self,
            _kernel: K,
            _config: crate::LaunchConfig,
            _args: Args,
        ) -> Result<(), Self::Error>
        where
            K: crate::Kernel<Args>,
        {
            Ok(())
        }
        fn synchronize(&self) -> Result<(), Self::Error> {
            Ok(())
        }
        fn fill<T: Clone>(&self, _buf: &mut Self::Buffer<T>, _value: T) -> Result<(), Self::Error> {
            Ok(())
        }
        fn copy_buffer<T: Clone>(
            &self,
            _src: &Self::Buffer<T>,
            _dst: &mut Self::Buffer<T>,
        ) -> Result<(), Self::Error> {
            Ok(())
        }

        fn event_query(event: &Self::Event) -> Result<bool, Self::Error> {
            Ok(event.ready.get())
        }

        fn event_synchronize(event: &Self::Event) -> Result<(), Self::Error> {
            event.ready.set(true);
            Ok(())
        }

        fn record_event(&self) -> Result<Token<Self>, Self::Error> {
            let event = MockEvent {
                ready: core::cell::Cell::new(false),
            };
            Ok(Token::from_event(event))
        }

        fn reduce<T, R>(&self, _buf: &Self::Buffer<T>, _op: R) -> Result<T, Self::Error>
        where
            T: Clone,
            R: crate::reduce::Reduce<T>,
        {
            Ok(R::identity())
        }
    }

    #[test]
    fn test_immediate_token() {
        let token = Token::<MockDevice>::immediate();
        assert!(token.is_immediate());
        assert!(token.ready().unwrap());
    }

    #[test]
    fn test_pending_token() {
        let event = MockEvent {
            ready: core::cell::Cell::new(false),
        };
        let token = Token::<MockDevice>::from_event(event);

        assert!(!token.is_immediate());
        assert!(!token.ready().unwrap());

        token.wait().unwrap();
        // after wait, token is consumed
    }

    #[test]
    fn test_wait_all() {
        use alloc::vec;
        let tokens = vec![
            Token::<MockDevice>::immediate(),
            Token::<MockDevice>::immediate(),
        ];

        wait_all(tokens).unwrap();
    }

    #[test]
    fn test_all_ready() {
        use alloc::vec;
        let tokens = vec![
            Token::<MockDevice>::immediate(),
            Token::<MockDevice>::immediate(),
        ];

        assert!(all_ready(&tokens).unwrap());
    }
}
