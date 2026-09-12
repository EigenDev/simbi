// =============================================================================
// frame.rs
//
// the wire frame: a fixed 36-byte little-endian header followed by a payload
// whose length the header states. the header carries the session, the epoch,
// and a kind-specific id, so every frame is validated against the receiver's
// expectations before its payload is touched.
//
// layout (byte offsets):
//   0  magic        b"SFAB"
//   4  version      u16
//   6  kind         u8
//   7  flags        u8 (zero)
//   8  session      u64
//  16  step         u64
//  24  attempt      u16
//  26  point        u8
//  27  axis         u8
//  28  id           u32
//  32  payload_len  u32
//
// usage:
//  let mut bytes = [0u8; HEADER_LEN];
//  header.encode(&mut bytes);
//  let back = Header::decode(&bytes)?;
// =============================================================================

use crate::ident::{Epoch, SessionId};

pub const MAGIC: [u8; 4] = *b"SFAB";
pub const VERSION: u16 = 1;
pub const HEADER_LEN: usize = 36;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Kind {
    Hello = 1,
    Welcome = 2,
    Grant = 3,
    Halo = 4,
    Credit = 5,
    Contribute = 6,
    Result = 7,
    Block = 8,
    Abort = 9,
    Done = 10,
}

impl Kind {
    pub fn from_byte(b: u8) -> Option<Self> {
        Some(match b {
            1 => Kind::Hello,
            2 => Kind::Welcome,
            3 => Kind::Grant,
            4 => Kind::Halo,
            5 => Kind::Credit,
            6 => Kind::Contribute,
            7 => Kind::Result,
            8 => Kind::Block,
            9 => Kind::Abort,
            10 => Kind::Done,
            _ => return None,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Header {
    pub kind: Kind,
    pub session: SessionId,
    pub epoch: Epoch,
    pub id: u32,
    pub payload_len: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FrameError {
    Magic([u8; 4]),
    Version(u16),
    Kind(u8),
    Flags(u8),
}

impl std::fmt::Display for FrameError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FrameError::Magic(m) => write!(f, "bad magic {m:?}"),
            FrameError::Version(v) => write!(f, "unsupported version {v}"),
            FrameError::Kind(k) => write!(f, "unknown frame kind {k}"),
            FrameError::Flags(x) => write!(f, "nonzero flags {x:#x}"),
        }
    }
}

impl Header {
    pub fn encode(&self, out: &mut [u8; HEADER_LEN]) {
        out[0..4].copy_from_slice(&MAGIC);
        out[4..6].copy_from_slice(&VERSION.to_le_bytes());
        out[6] = self.kind as u8;
        out[7] = 0;
        out[8..16].copy_from_slice(&self.session.0.to_le_bytes());
        out[16..24].copy_from_slice(&self.epoch.step.to_le_bytes());
        out[24..26].copy_from_slice(&self.epoch.attempt.to_le_bytes());
        out[26] = self.epoch.point;
        out[27] = self.epoch.axis;
        out[28..32].copy_from_slice(&self.id.to_le_bytes());
        out[32..36].copy_from_slice(&self.payload_len.to_le_bytes());
    }

    pub fn decode(bytes: &[u8; HEADER_LEN]) -> Result<Self, FrameError> {
        let magic: [u8; 4] = bytes[0..4].try_into().expect("four bytes");
        if magic != MAGIC {
            return Err(FrameError::Magic(magic));
        }
        let version = u16::from_le_bytes([bytes[4], bytes[5]]);
        if version != VERSION {
            return Err(FrameError::Version(version));
        }
        let kind = Kind::from_byte(bytes[6]).ok_or(FrameError::Kind(bytes[6]))?;
        if bytes[7] != 0 {
            return Err(FrameError::Flags(bytes[7]));
        }
        let u64_at =
            |o: usize| u64::from_le_bytes(bytes[o..o + 8].try_into().expect("eight bytes"));
        let u32_at = |o: usize| u32::from_le_bytes(bytes[o..o + 4].try_into().expect("four bytes"));
        Ok(Self {
            kind,
            session: SessionId(u64_at(8)),
            epoch: Epoch {
                step: u64_at(16),
                attempt: u16::from_le_bytes([bytes[24], bytes[25]]),
                point: bytes[26],
                axis: bytes[27],
            },
            id: u32_at(28),
            payload_len: u32_at(32),
        })
    }
}

/// f64 payloads travel as little-endian bit patterns.
pub fn encode_f64s(values: &[f64], out: &mut Vec<u8>) {
    out.clear();
    out.reserve(values.len() * 8);
    for v in values {
        out.extend_from_slice(&v.to_bits().to_le_bytes());
    }
}

pub fn decode_f64s(bytes: &[u8], out: &mut [f64]) {
    debug_assert_eq!(bytes.len(), out.len() * 8);
    for (i, chunk) in bytes.chunks_exact(8).enumerate() {
        out[i] = f64::from_bits(u64::from_le_bytes(chunk.try_into().expect("eight bytes")));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> Header {
        Header {
            kind: Kind::Halo,
            session: SessionId(0x1122_3344_5566_7788),
            epoch: Epoch {
                step: 7,
                attempt: 2,
                point: 1,
                axis: 0,
            },
            id: 42,
            payload_len: 8 * 5,
        }
    }

    #[test]
    fn magic_is_the_literal_bytes_sfab() {
        let mut bytes = [0u8; HEADER_LEN];
        sample().encode(&mut bytes);
        assert_eq!(&bytes[0..4], b"SFAB");
        assert_eq!(bytes[0], b'S');
    }

    #[test]
    fn header_round_trips() {
        let mut bytes = [0u8; HEADER_LEN];
        sample().encode(&mut bytes);
        assert_eq!(Header::decode(&bytes).unwrap(), sample());
    }

    #[test]
    fn header_rejects_bad_magic_version_kind_flags() {
        let mut bytes = [0u8; HEADER_LEN];
        sample().encode(&mut bytes);
        let mut m = bytes;
        m[0] = b'X';
        assert!(matches!(Header::decode(&m), Err(FrameError::Magic(_))));
        let mut v = bytes;
        v[4] = 9;
        assert!(matches!(Header::decode(&v), Err(FrameError::Version(_))));
        let mut k = bytes;
        k[6] = 200;
        assert!(matches!(Header::decode(&k), Err(FrameError::Kind(200))));
        let mut f = bytes;
        f[7] = 1;
        assert!(matches!(Header::decode(&f), Err(FrameError::Flags(1))));
    }

    #[test]
    fn f64_payloads_round_trip_bitwise() {
        let values = [0.0, -0.0, 1.5, f64::NAN, f64::INFINITY, 1e-310];
        let mut bytes = Vec::new();
        encode_f64s(&values, &mut bytes);
        let mut back = [0.0; 6];
        decode_f64s(&bytes, &mut back);
        for (a, b) in values.iter().zip(&back) {
            assert_eq!(a.to_bits(), b.to_bits());
        }
    }
}
