// =============================================================================
// ident.rs
//
// the identities that name things on the wire: workers, sessions, transfers,
// collective operations, and the epoch a transfer belongs to, plus the digest
// that two workers compare to establish they compiled the same plan.
//
// the digest is FNV-1a over a canonical little-endian encoding. it is a
// same-session consistency check between identical builds, so the encoding
// rules here are the contract: every producer pushes the same fields in the
// same order and width.
//
// usage:
//  let mut h = Fnv::new();
//  h.u64(reach as u64);
//  h.str("rho");
//  let digest = h.finish();
// =============================================================================

/// a worker process within a session. worker 0 is the coordinator.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct WorkerId(pub u32);

/// fresh for every process launch, including restart.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SessionId(pub u64);

/// one region move within a compiled exchange plan, numbered in plan order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TransferId(pub u32);

/// one collective operation, numbered in the order the step loop issues them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct OpId(pub u32);

/// the version a transfer belongs to. `point` is the exchange point's wire
/// byte (0 prime, 1 + k stage k, 0xF0 rollback, 0xFF none) and `axis` the
/// closing axis phase (0xFF for frames outside an exchange). the derived
/// ordering is the total order of phases: step, then attempt, then point,
/// then axis, which the wire bytes preserve by construction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Epoch {
    pub step: u64,
    pub attempt: u16,
    pub point: u8,
    pub axis: u8,
}

impl Epoch {
    pub const NO_POINT: u8 = 0xFF;
    pub const NO_AXIS: u8 = 0xFF;
}

/// a 64-bit FNV-1a digest of a canonical encoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Digest(pub u64);

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

/// an FNV-1a accumulator with fixed-width little-endian pushes.
#[derive(Debug, Clone)]
pub struct Fnv(u64);

impl Default for Fnv {
    fn default() -> Self {
        Self::new()
    }
}

impl Fnv {
    pub fn new() -> Self {
        Self(FNV_OFFSET)
    }

    pub fn bytes(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 ^= u64::from(b);
            self.0 = self.0.wrapping_mul(FNV_PRIME);
        }
    }

    pub fn u8(&mut self, v: u8) {
        self.bytes(&[v]);
    }

    pub fn u16(&mut self, v: u16) {
        self.bytes(&v.to_le_bytes());
    }

    pub fn u32(&mut self, v: u32) {
        self.bytes(&v.to_le_bytes());
    }

    pub fn u64(&mut self, v: u64) {
        self.bytes(&v.to_le_bytes());
    }

    pub fn i64(&mut self, v: i64) {
        self.bytes(&v.to_le_bytes());
    }

    /// a length-prefixed string, so two adjacent strings cannot alias.
    pub fn str(&mut self, s: &str) {
        self.u32(s.len() as u32);
        self.bytes(s.as_bytes());
    }

    pub fn finish(self) -> Digest {
        Digest(self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fnv1a_matches_the_published_vectors() {
        assert_eq!(Fnv::new().finish(), Digest(0xcbf2_9ce4_8422_2325));
        let mut h = Fnv::new();
        h.bytes(b"a");
        assert_eq!(h.finish(), Digest(0xaf63_dc4c_8601_ec8c));
        let mut h = Fnv::new();
        h.bytes(b"foobar");
        assert_eq!(h.finish(), Digest(0x8594_4171_f73967e8));
    }

    #[test]
    fn length_prefixed_strings_do_not_alias() {
        let mut a = Fnv::new();
        a.str("ab");
        a.str("c");
        let mut b = Fnv::new();
        b.str("a");
        b.str("bc");
        assert_ne!(a.finish(), b.finish());
    }

    #[test]
    fn epochs_order_by_step_attempt_point_axis() {
        let e = |step, attempt, point, axis| Epoch {
            step,
            attempt,
            point,
            axis,
        };
        assert!(e(0, 0, 0, 1) < e(0, 0, 1, 0));
        assert!(e(0, 0, 2, 1) < e(0, 0, 0xF0, 0));
        assert!(e(0, 0, 0xF0, 1) < e(0, 1, 1, 0));
        assert!(e(0, 1, 2, 1) < e(1, 0, 0, 0));
    }
}
