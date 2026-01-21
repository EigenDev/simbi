// =============================================================================
// entity.rs
//
// simple entity id. just a unique identifier, nothing more.
// =============================================================================

use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_ID: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Entity(u64);

impl Entity {
    pub fn new() -> Self {
        Self(NEXT_ID.fetch_add(1, Ordering::Relaxed))
    }

    pub fn id(&self) -> u64 {
        self.0
    }
}

impl Default for Entity {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn entities_are_unique() {
        let e1 = Entity::new();
        let e2 = Entity::new();
        let e3 = Entity::new();

        assert_ne!(e1, e2);
        assert_ne!(e2, e3);
        assert_ne!(e1, e3);
    }
}
