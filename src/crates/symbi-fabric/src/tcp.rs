// =============================================================================
// tcp.rs
//
// the socket mesh link: one nonblocking TCP stream per peer, driven by the
// endpoint's progress loop. every read and write may be partial; a frame is
// assembled from as many reads as the stream delivers and written in as many
// writes as the socket accepts.
//
// each peer has one outbox and one inbox, both allocated at attach and never
// resized. the outbox is a single ordered queue: frames leave in the order
// they were pushed, the frame at the front owns the socket until its last byte
// is written, and one payload-bearing frame at a time may be queued. the inbox
// holds one frame of at most `max_payload` bytes; a header stating more is a
// protocol error raised before any payload byte is read. end of file and
// resets are a disconnect reported with the peer's id.
//
// the outbox and inbox are generic over the stream so their sequencing is
// tested against deterministic short writers and readers.
//
// usage:
//  let link = TcpLink::new(me, workers, max_payload);
//  link.attach(peer, stream)?;
//  // then hand the link to Fabric::new
// =============================================================================

use crate::endpoint::Link;
use crate::error::FabricError;
use crate::frame::{HEADER_LEN, Header};
use crate::ident::WorkerId;
use std::cell::RefCell;
use std::collections::VecDeque;
use std::io::{self, ErrorKind, Read, Write};
use std::net::TcpStream;

/// payloads at or below this size travel as control frames in the small slots.
pub const CONTROL_PAYLOAD: usize = 64;
/// control frames held per peer before a push is refused; the protocol issues
/// at most one grant per axis plus Done and Abort between drains.
pub const CONTROL_QUEUE: usize = 16;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Slot {
    Large,
    Control(usize),
    Raw,
}

#[derive(Debug, Clone, Copy)]
struct Queued {
    slot: Slot,
    sent: usize,
}

/// the ordered outbound queue of one peer with its preallocated storage.
pub struct Outbox {
    large: Vec<u8>,
    control: Vec<Vec<u8>>,
    free_control: Vec<usize>,
    raw: Vec<u8>,
    queue: VecDeque<Queued>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PushError {
    ControlQueueFull,
    LargeInFlight,
    PayloadTooLarge,
    HeaderMisstatesPayload,
}

impl Outbox {
    pub fn new(max_payload: usize) -> Self {
        Self {
            large: Vec::with_capacity(HEADER_LEN + max_payload),
            control: (0..CONTROL_QUEUE)
                .map(|_| Vec::with_capacity(HEADER_LEN + CONTROL_PAYLOAD))
                .collect(),
            free_control: (0..CONTROL_QUEUE).rev().collect(),
            raw: Vec::new(),
            queue: VecDeque::with_capacity(CONTROL_QUEUE + 2),
        }
    }

    /// whether a payload-bearing frame is queued or in flight.
    pub fn has_large(&self) -> bool {
        self.queue.iter().any(|q| q.slot == Slot::Large)
    }

    pub fn pending(&self) -> bool {
        !self.queue.is_empty()
    }

    /// the bytes this outbox holds, fixed at construction.
    pub fn capacity_bytes(&self) -> usize {
        self.large.capacity() + self.control.iter().map(Vec::capacity).sum::<usize>()
    }

    /// queue a frame behind everything already queued. storage is reused; nothing is
    /// allocated here.
    pub fn push(&mut self, header: Header, payload: &[u8]) -> Result<(), PushError> {
        if payload.len() != header.payload_len as usize {
            return Err(PushError::HeaderMisstatesPayload);
        }
        let mut head = [0u8; HEADER_LEN];
        header.encode(&mut head);
        let slot = if payload.len() <= CONTROL_PAYLOAD {
            let Some(i) = self.free_control.pop() else {
                return Err(PushError::ControlQueueFull);
            };
            let buf = &mut self.control[i];
            buf.clear();
            buf.extend_from_slice(&head);
            buf.extend_from_slice(payload);
            Slot::Control(i)
        } else {
            if self.has_large() {
                return Err(PushError::LargeInFlight);
            }
            if HEADER_LEN + payload.len() > self.large.capacity() {
                return Err(PushError::PayloadTooLarge);
            }
            self.large.clear();
            self.large.extend_from_slice(&head);
            self.large.extend_from_slice(payload);
            Slot::Large
        };
        self.queue.push_back(Queued { slot, sent: 0 });
        Ok(())
    }

    /// queue raw bytes: the hostile-stream test hook. this is the one path that allocates.
    pub fn push_raw(&mut self, bytes: &[u8]) {
        self.raw = bytes.to_vec();
        self.queue.push_back(Queued {
            slot: Slot::Raw,
            sent: 0,
        });
    }

    fn bytes_of(&self, slot: Slot) -> &[u8] {
        match slot {
            Slot::Large => &self.large,
            Slot::Control(i) => &self.control[i],
            Slot::Raw => &self.raw,
        }
    }

    /// write as much of the queue as the stream accepts, front frame first and whole.
    /// `Ok(true)` when the queue is empty afterwards.
    pub fn write_to(&mut self, stream: &mut impl Write) -> io::Result<bool> {
        while let Some(front) = self.queue.front().copied() {
            let bytes = self.bytes_of(front.slot);
            let mut sent = front.sent;
            while sent < bytes.len() {
                match stream.write(&bytes[sent..]) {
                    Ok(0) => return Err(io::Error::from(ErrorKind::WriteZero)),
                    Ok(n) => sent += n,
                    Err(e) if e.kind() == ErrorKind::WouldBlock => {
                        self.queue.front_mut().expect("front exists").sent = sent;
                        return Ok(false);
                    }
                    Err(e) if e.kind() == ErrorKind::Interrupted => continue,
                    Err(e) => return Err(e),
                }
            }
            self.queue.pop_front();
            if let Slot::Control(i) = front.slot {
                self.free_control.push(i);
            }
        }
        Ok(true)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReadOutcome {
    /// no complete frame yet; the stream has nothing more right now
    Blocked,
    /// a frame is complete and waits to be taken
    Complete,
    /// the peer closed the stream
    Eof,
}

/// the inbound assembler of one peer: one frame at a time into preallocated storage.
pub struct Inbox {
    header_bytes: [u8; HEADER_LEN],
    header_have: usize,
    header: Option<Header>,
    payload: Vec<u8>,
    payload_have: usize,
    complete: bool,
    max_payload: usize,
}

#[derive(Debug)]
pub enum ReadError {
    Io(io::Error),
    /// a header that failed to decode or states a payload over the bound
    Protocol(String),
}

impl From<io::Error> for ReadError {
    fn from(e: io::Error) -> Self {
        ReadError::Io(e)
    }
}

impl Inbox {
    pub fn new(max_payload: usize) -> Self {
        Self {
            header_bytes: [0; HEADER_LEN],
            header_have: 0,
            header: None,
            payload: vec![0; max_payload],
            payload_have: 0,
            complete: false,
            max_payload,
        }
    }

    pub fn is_complete(&self) -> bool {
        self.complete
    }

    /// whether a frame is partly read and waits for more bytes.
    pub fn mid_frame(&self) -> bool {
        !self.complete && (self.header_have > 0 || self.header.is_some())
    }

    pub fn capacity_bytes(&self) -> usize {
        self.payload.capacity()
    }

    /// read toward one complete frame. a complete frame is held until `take`, and the
    /// stream is left unread meanwhile, which is the backpressure toward the sender.
    pub fn read_from(&mut self, stream: &mut impl Read) -> Result<ReadOutcome, ReadError> {
        if self.complete {
            return Ok(ReadOutcome::Complete);
        }
        loop {
            if self.header.is_none() {
                match stream.read(&mut self.header_bytes[self.header_have..]) {
                    Ok(0) => return Ok(ReadOutcome::Eof),
                    Ok(n) => self.header_have += n,
                    Err(e) if e.kind() == ErrorKind::WouldBlock => return Ok(ReadOutcome::Blocked),
                    Err(e) if e.kind() == ErrorKind::Interrupted => continue,
                    Err(e) => return Err(e.into()),
                }
                if self.header_have < HEADER_LEN {
                    continue;
                }
                let header = Header::decode(&self.header_bytes)
                    .map_err(|e| ReadError::Protocol(format!("frame header: {e}")))?;
                if header.payload_len as usize > self.max_payload {
                    return Err(ReadError::Protocol(format!(
                        "payload of {} bytes exceeds the {}-byte bound",
                        header.payload_len, self.max_payload
                    )));
                }
                self.header = Some(header);
                self.payload_have = 0;
            }
            let want = self.header.expect("set above").payload_len as usize;
            while self.payload_have < want {
                match stream.read(&mut self.payload[self.payload_have..want]) {
                    Ok(0) => return Ok(ReadOutcome::Eof),
                    Ok(n) => self.payload_have += n,
                    Err(e) if e.kind() == ErrorKind::WouldBlock => return Ok(ReadOutcome::Blocked),
                    Err(e) if e.kind() == ErrorKind::Interrupted => continue,
                    Err(e) => return Err(e.into()),
                }
            }
            self.complete = true;
            return Ok(ReadOutcome::Complete);
        }
    }

    /// hand over the complete frame: the header, with the payload copied into `out`.
    pub fn take(&mut self, out: &mut Vec<u8>) -> Option<Header> {
        if !self.complete {
            return None;
        }
        let header = self.header.take().expect("a complete frame has a header");
        out.clear();
        out.extend_from_slice(&self.payload[..header.payload_len as usize]);
        self.header_have = 0;
        self.payload_have = 0;
        self.complete = false;
        Some(header)
    }
}

/// counters a gate reads to confirm the stream was split where it claims.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct LinkStats {
    /// drive passes that left a frame partially written
    pub partial_writes: u64,
    /// drive passes that left a frame partially read
    pub partial_reads: u64,
}

struct Peer {
    stream: TcpStream,
    inbox: Inbox,
    outbox: Outbox,
}

pub struct TcpLink {
    me: WorkerId,
    max_payload: usize,
    peers: RefCell<Vec<Option<Peer>>>,
    next_poll: RefCell<usize>,
    stats: RefCell<LinkStats>,
}

fn disconnected(peer: WorkerId) -> FabricError {
    FabricError::Disconnected { peer, epoch: None }
}

fn io_to_fabric(e: io::Error, peer: WorkerId) -> FabricError {
    match e.kind() {
        ErrorKind::BrokenPipe
        | ErrorKind::ConnectionReset
        | ErrorKind::ConnectionAborted
        | ErrorKind::WriteZero
        | ErrorKind::UnexpectedEof => disconnected(peer),
        _ => FabricError::Io(e),
    }
}

impl TcpLink {
    pub fn new(me: WorkerId, workers: usize, max_payload: usize) -> Self {
        Self {
            me,
            max_payload,
            peers: RefCell::new((0..workers).map(|_| None).collect()),
            next_poll: RefCell::new(0),
            stats: RefCell::new(LinkStats::default()),
        }
    }

    pub fn stats(&self) -> LinkStats {
        *self.stats.borrow()
    }

    pub fn me(&self) -> WorkerId {
        self.me
    }

    pub fn max_payload(&self) -> usize {
        self.max_payload
    }

    /// bind a peer's stream; the stream becomes nonblocking here and the peer's inbox and
    /// outbox are allocated once.
    pub fn attach(&self, peer: WorkerId, stream: TcpStream) -> Result<(), FabricError> {
        stream.set_nonblocking(true)?;
        stream.set_nodelay(true)?;
        let slot = &mut self.peers.borrow_mut()[peer.0 as usize];
        if slot.is_some() {
            return Err(FabricError::Rendezvous {
                detail: format!("{peer:?} attached twice"),
            });
        }
        *slot = Some(Peer {
            stream,
            inbox: Inbox::new(self.max_payload),
            outbox: Outbox::new(self.max_payload),
        });
        Ok(())
    }

    pub fn attached(&self) -> usize {
        self.peers.borrow().iter().flatten().count()
    }

    /// queue raw bytes to `to` behind whatever is queued: the test hook for a malformed
    /// stream.
    pub fn inject_raw(&self, to: WorkerId, bytes: &[u8]) {
        let mut peers = self.peers.borrow_mut();
        let peer = peers[to.0 as usize].as_mut().expect("peer attached");
        peer.outbox.push_raw(bytes);
    }

    /// the bytes held in every peer's inbox and outbox, fixed for the session.
    pub fn peer_buffer_bytes(&self) -> usize {
        self.peers
            .borrow()
            .iter()
            .flatten()
            .map(|p| p.inbox.capacity_bytes() + p.outbox.capacity_bytes())
            .sum()
    }

    fn drive_peer(
        peer_id: WorkerId,
        peer: &mut Peer,
        stats: &mut LinkStats,
    ) -> Result<(), FabricError> {
        match peer.outbox.write_to(&mut peer.stream) {
            Ok(true) => {}
            Ok(false) => stats.partial_writes += 1,
            Err(e) => return Err(io_to_fabric(e, peer_id)),
        }
        match peer.inbox.read_from(&mut peer.stream) {
            Ok(ReadOutcome::Complete) => Ok(()),
            Ok(ReadOutcome::Blocked) => {
                if peer.inbox.mid_frame() {
                    stats.partial_reads += 1;
                }
                Ok(())
            }
            Ok(ReadOutcome::Eof) => Err(disconnected(peer_id)),
            Err(ReadError::Io(e)) => Err(io_to_fabric(e, peer_id)),
            Err(ReadError::Protocol(detail)) => Err(FabricError::protocol(peer_id, detail)),
        }
    }
}

impl Link for TcpLink {
    fn send(
        &self,
        _from: WorkerId,
        to: WorkerId,
        header: Header,
        payload: &[u8],
    ) -> Result<(), FabricError> {
        let mut peers = self.peers.borrow_mut();
        let Some(peer) = peers[to.0 as usize].as_mut() else {
            return Err(FabricError::protocol(
                self.me,
                format!("send to unattached {to:?}"),
            ));
        };
        peer.outbox
            .push(header, payload)
            .map_err(|e| FabricError::protocol(self.me, format!("send to {to:?}: {e:?}")))
    }

    fn poll(
        &self,
        _me: WorkerId,
        payload: &mut Vec<u8>,
    ) -> Result<Option<(WorkerId, Header)>, FabricError> {
        let mut peers = self.peers.borrow_mut();
        let n = peers.len();
        let start = *self.next_poll.borrow();
        for k in 0..n {
            let idx = (start + k) % n;
            let Some(peer) = peers[idx].as_mut() else {
                continue;
            };
            if let Some(header) = peer.inbox.take(payload) {
                *self.next_poll.borrow_mut() = (idx + 1) % n;
                return Ok(Some((WorkerId(idx as u32), header)));
            }
        }
        Ok(None)
    }

    fn ready(&self, to: WorkerId) -> bool {
        self.peers.borrow()[to.0 as usize]
            .as_ref()
            .is_some_and(|p| !p.outbox.has_large())
    }

    fn drive(&self, _me: WorkerId) -> Result<(), FabricError> {
        let mut peers = self.peers.borrow_mut();
        let mut stats = self.stats.borrow_mut();
        for (idx, peer) in peers.iter_mut().enumerate() {
            if let Some(peer) = peer.as_mut() {
                Self::drive_peer(WorkerId(idx as u32), peer, &mut stats)?;
            }
        }
        Ok(())
    }

    fn detach(&self, peer: WorkerId) {
        self.peers.borrow_mut()[peer.0 as usize] = None;
    }

    fn outbound_pending(&self) -> bool {
        self.peers
            .borrow()
            .iter()
            .flatten()
            .any(|p| p.outbox.pending())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame::Kind;
    use crate::ident::{Epoch, SessionId};

    /// accepts at most `limit` bytes per write and then blocks once.
    struct ShortWriter {
        out: Vec<u8>,
        limit: usize,
        block_next: bool,
    }

    impl Write for ShortWriter {
        fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
            if self.block_next {
                self.block_next = false;
                return Err(io::Error::from(ErrorKind::WouldBlock));
            }
            let n = buf.len().min(self.limit);
            self.out.extend_from_slice(&buf[..n]);
            self.block_next = true;
            Ok(n)
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    /// delivers one byte per read and blocks between bytes.
    struct TrickleReader {
        data: Vec<u8>,
        pos: usize,
        block_next: bool,
    }

    impl Read for TrickleReader {
        fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
            if self.block_next {
                self.block_next = false;
                return Err(io::Error::from(ErrorKind::WouldBlock));
            }
            if self.pos >= self.data.len() {
                return Ok(0);
            }
            buf[0] = self.data[self.pos];
            self.pos += 1;
            self.block_next = true;
            Ok(1)
        }
    }

    fn header(kind: Kind, id: u32, payload_len: u32) -> Header {
        Header {
            kind,
            session: SessionId(5),
            epoch: Epoch {
                step: 1,
                attempt: 0,
                point: 1,
                axis: 0,
            },
            id,
            payload_len,
        }
    }

    fn encoded(h: Header, payload: &[u8]) -> Vec<u8> {
        let mut bytes = [0u8; HEADER_LEN];
        h.encode(&mut bytes);
        let mut v = bytes.to_vec();
        v.extend_from_slice(payload);
        v
    }

    /// a control frame that stalled mid-write keeps the stream until it is complete; a halo
    /// queued behind it follows whole. the byte stream is the two frames in push order.
    #[test]
    fn a_halo_never_splices_into_a_partially_written_control_frame() {
        let mut outbox = Outbox::new(1024);
        let grant = header(Kind::Grant, 1, 0);
        outbox.push(grant, &[]).unwrap();
        let mut w = ShortWriter {
            out: Vec::new(),
            limit: 5,
            block_next: false,
        };
        assert!(!outbox.write_to(&mut w).unwrap());
        assert_eq!(w.out.len(), 5, "the control frame is five bytes in");
        let payload: Vec<u8> = (0..200u8).collect();
        let halo = header(Kind::Halo, 7, 200);
        outbox.push(halo, &payload).unwrap();
        let mut rounds = 0;
        while !outbox.write_to(&mut w).unwrap() {
            rounds += 1;
            assert!(rounds < 1000);
        }
        let mut want = encoded(grant, &[]);
        want.extend(encoded(halo, &payload));
        assert_eq!(w.out, want);
    }

    #[test]
    fn frames_leave_in_push_order_and_control_slots_recycle() {
        let mut outbox = Outbox::new(64);
        let mut want = Vec::new();
        let mut w = ShortWriter {
            out: Vec::new(),
            limit: 7,
            block_next: false,
        };
        for round in 0..3 {
            for i in 0..CONTROL_QUEUE as u32 {
                let h = header(Kind::Grant, round * 100 + i, 0);
                outbox.push(h, &[]).unwrap();
                want.extend(encoded(h, &[]));
            }
            assert_eq!(
                outbox.push(header(Kind::Grant, 999, 0), &[]),
                Err(PushError::ControlQueueFull)
            );
            while !outbox.write_to(&mut w).unwrap() {}
        }
        assert_eq!(w.out, want);
        assert!(!outbox.pending());
    }

    #[test]
    fn one_payload_frame_at_a_time_and_the_bound_holds() {
        let mut outbox = Outbox::new(100);
        let payload = vec![1u8; 100];
        outbox.push(header(Kind::Halo, 1, 100), &payload).unwrap();
        assert_eq!(
            outbox.push(header(Kind::Halo, 2, 100), &payload),
            Err(PushError::LargeInFlight)
        );
        let mut w = ShortWriter {
            out: Vec::new(),
            limit: 1000,
            block_next: false,
        };
        while !outbox.write_to(&mut w).unwrap() {}
        let big = vec![1u8; 101];
        assert_eq!(
            outbox.push(header(Kind::Halo, 3, 101), &big),
            Err(PushError::PayloadTooLarge)
        );
        assert_eq!(
            outbox.push(header(Kind::Halo, 4, 99), &payload),
            Err(PushError::HeaderMisstatesPayload)
        );
    }

    #[test]
    fn outbox_capacity_is_fixed_across_pushes() {
        let mut outbox = Outbox::new(256);
        let before = outbox.capacity_bytes();
        let payload = vec![9u8; 256];
        let mut w = ShortWriter {
            out: Vec::new(),
            limit: 4096,
            block_next: false,
        };
        for i in 0..50 {
            outbox.push(header(Kind::Halo, i, 256), &payload).unwrap();
            outbox.push(header(Kind::Grant, i, 0), &[]).unwrap();
            while !outbox.write_to(&mut w).unwrap() {}
        }
        assert_eq!(outbox.capacity_bytes(), before);
    }

    /// a frame arriving one byte per read, the header split at every offset.
    #[test]
    fn inbox_assembles_a_trickled_frame() {
        let payload: Vec<u8> = (0..37u8).collect();
        let h = header(Kind::Halo, 3, 37);
        let mut r = TrickleReader {
            data: encoded(h, &payload),
            pos: 0,
            block_next: false,
        };
        let mut inbox = Inbox::new(64);
        let mut blocked = 0;
        loop {
            match inbox.read_from(&mut r).unwrap() {
                ReadOutcome::Complete => break,
                ReadOutcome::Blocked => blocked += 1,
                ReadOutcome::Eof => panic!("eof before the frame completed"),
            }
        }
        assert!(
            blocked >= HEADER_LEN + 36,
            "the stream blocked between bytes"
        );
        let mut out = Vec::new();
        assert_eq!(inbox.take(&mut out), Some(h));
        assert_eq!(out, payload);
        r.block_next = false;
        assert_eq!(inbox.read_from(&mut r).unwrap(), ReadOutcome::Eof);
    }

    #[test]
    fn inbox_rejects_an_oversize_header_before_reading_the_payload() {
        let h = header(Kind::Halo, 3, 65);
        let mut r = TrickleReader {
            data: encoded(h, &[0u8; 65]),
            pos: 0,
            block_next: false,
        };
        let mut inbox = Inbox::new(64);
        let err = loop {
            match inbox.read_from(&mut r) {
                Ok(ReadOutcome::Blocked) => continue,
                Ok(other) => panic!("{other:?}"),
                Err(e) => break e,
            }
        };
        assert!(matches!(err, ReadError::Protocol(_)));
        assert_eq!(r.pos, HEADER_LEN, "no payload byte was read");
    }
}
