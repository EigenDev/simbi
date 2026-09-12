// =============================================================================
// endpoint.rs
//
// one worker's end of the fabric: the per-transfer buffers and their send and
// receive states, the grant table, and the phase protocol that moves halos
// between workers. a phase opens by posting receives and granting the peers
// that feed them, packs its sends, and then makes progress until every send
// has left and every receive has landed; the caller scatters received payloads
// and closes the phase, which frees every buffer for the next epoch. the
// session ends with a Done exchange: a worker leaves once every peer has said
// Done and its own outbound queue has drained.
//
// the endpoint moves f64 bit patterns under transfer ids and never sees a
// field or a region. `Link` is the byte-frame carrier beneath it: the in-process
// `Loopback` here, the socket mesh in `tcp`. every buffer the endpoint owns is
// allocated at construction and holds its size for the session; the phase
// bookkeeping reuses its vectors, so open, progress, close, and finish allocate
// nothing.
//
// usage:
//  let mut fabric = Fabric::new(link, me, session, workers, axes, &lens, sends_per_peer_axis);
//  fabric.pack(id, |buf| ...)?;
//  fabric.open(PhaseSpec { epoch, sends: &[(id, to)], receives: &[(id, from)] })?;
//  fabric.wait(deadline)?;
//  let values = fabric.payload(id)?; ... fabric.mark_unpacked(id)?;
//  fabric.close()?;
//  fabric.finish(deadline)?;
// =============================================================================

use crate::error::{AbortReason, Deadline, FabricError};
use crate::frame::{HEADER_LEN, Header, Kind, decode_f64s, encode_f64s};
use crate::grant::GrantTable;
use crate::ident::{Epoch, SessionId, TransferId, WorkerId};
use std::cell::RefCell;
use std::collections::VecDeque;
use std::rc::Rc;
use std::time::{Duration, Instant};

/// a frame carrier between workers. delivery order between one ordered pair of
/// workers is preserved. control frames (no payload or a few bytes) are always
/// accepted; a payload-bearing frame is accepted when `ready` says so, and the
/// carrier holds at most one such frame per peer, so its memory is fixed.
pub trait Link {
    fn send(
        &self,
        from: WorkerId,
        to: WorkerId,
        header: Header,
        payload: &[u8],
    ) -> Result<(), FabricError>;

    /// the oldest complete frame addressed to `me`, its payload copied into `payload`.
    fn poll(
        &self,
        me: WorkerId,
        payload: &mut Vec<u8>,
    ) -> Result<Option<(WorkerId, Header)>, FabricError>;

    /// whether a payload-bearing frame to `to` can be accepted now.
    fn ready(&self, _to: WorkerId) -> bool {
        true
    }

    /// push pending writes and pull pending reads.
    fn drive(&self, _me: WorkerId) -> Result<(), FabricError> {
        Ok(())
    }

    /// stop carrying frames to or from `peer`: it has finished the session.
    fn detach(&self, _peer: WorkerId) {}

    /// whether any frame this worker sent has bytes still to be written.
    fn outbound_pending(&self) -> bool {
        false
    }
}

type Queued = (WorkerId, WorkerId, Header, Vec<u8>);

/// the in-process reference link: every frame goes through one queue and a
/// worker takes the oldest frame addressed to it. frames are encoded to and
/// decoded from bytes on the way, so the codec is on the path. this link
/// allocates per frame; it is the reference, not the production carrier.
#[derive(Clone, Default)]
pub struct Loopback(Rc<RefCell<VecDeque<Queued>>>);

impl Loopback {
    pub fn new() -> Self {
        Self::default()
    }

    /// place a frame as if `from` had sent it: the test hook for hostile frames.
    pub fn inject(&self, from: WorkerId, to: WorkerId, header: Header, payload: &[u8]) {
        self.send(from, to, header, payload)
            .expect("the loopback accepts every frame");
    }

    pub fn queued(&self) -> usize {
        self.0.borrow().len()
    }

    pub fn queued_of_kind(&self, kind: Kind) -> usize {
        self.0
            .borrow()
            .iter()
            .filter(|(_, _, h, _)| h.kind == kind)
            .count()
    }
}

impl Link for Loopback {
    fn send(
        &self,
        from: WorkerId,
        to: WorkerId,
        header: Header,
        payload: &[u8],
    ) -> Result<(), FabricError> {
        let mut bytes = [0u8; HEADER_LEN];
        header.encode(&mut bytes);
        let decoded = Header::decode(&bytes).expect("an encoded header decodes");
        self.0
            .borrow_mut()
            .push_back((from, to, decoded, payload.to_vec()));
        Ok(())
    }

    fn poll(
        &self,
        me: WorkerId,
        payload: &mut Vec<u8>,
    ) -> Result<Option<(WorkerId, Header)>, FabricError> {
        let mut q = self.0.borrow_mut();
        let Some(pos) = q.iter().position(|(_, to, _, _)| *to == me) else {
            return Ok(None);
        };
        let (from, _, header, bytes) = q.remove(pos).expect("position found above");
        payload.clear();
        payload.extend_from_slice(&bytes);
        Ok(Some((from, header)))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SendState {
    Idle,
    Packed,
    Sent,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecvState {
    Idle,
    Expected,
    Received,
    Unpacked,
}

#[derive(Debug)]
struct Slot {
    buf: Vec<f64>,
    send: SendState,
    recv: RecvState,
    peer: Option<WorkerId>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Progress {
    Pending,
    Done,
}

/// counters a gate reads to confirm the pressure it set up was real.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Stats {
    /// progress passes in which a send waited for its peer's grant
    pub grant_waits: u64,
    /// progress passes in which a send waited for the link to accept a frame
    pub link_waits: u64,
    pub passes: u64,
}

/// a global operation. every worker issues the same sequence of operations, so
/// the operation id is the position in that sequence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum OpKind {
    /// the minimum of f64 bit patterns; every contribution is finite and positive
    Min = 1,
    /// whether any contribution is nonzero
    Any = 2,
    /// whether every contribution is nonzero: the checkpoint may open
    CheckpointOpen = 3,
    /// whether every contribution is nonzero: the checkpoint is complete
    CheckpointClose = 4,
}

impl OpKind {
    fn from_byte(b: u8) -> Option<Self> {
        Some(match b {
            1 => OpKind::Min,
            2 => OpKind::Any,
            3 => OpKind::CheckpointOpen,
            4 => OpKind::CheckpointClose,
            _ => return None,
        })
    }
}

/// the payload of a Contribute or Result frame: the kind byte and the value bits.
const OP_PAYLOAD: usize = 9;

fn encode_op(kind: OpKind, value: u64) -> [u8; OP_PAYLOAD] {
    let mut out = [0u8; OP_PAYLOAD];
    out[0] = kind as u8;
    out[1..].copy_from_slice(&value.to_le_bytes());
    out
}

fn decode_op(payload: &[u8]) -> Option<(OpKind, u64)> {
    if payload.len() != OP_PAYLOAD {
        return None;
    }
    let kind = OpKind::from_byte(payload[0])?;
    let value = u64::from_le_bytes(payload[1..].try_into().expect("eight bytes"));
    Some((kind, value))
}

/// the coordinator's table for one operation: one contribution per worker.
#[derive(Debug)]
struct Gather {
    op: u32,
    kind: Option<OpKind>,
    values: Vec<Option<u64>>,
    count: usize,
}

impl Gather {
    fn fresh(op: u32, workers: usize) -> Self {
        Self {
            op,
            kind: None,
            values: vec![None; workers],
            count: 0,
        }
    }

    fn reset(&mut self, op: u32) {
        self.op = op;
        self.kind = None;
        self.values.iter_mut().for_each(|v| *v = None);
        self.count = 0;
    }
}

/// the collective state of one worker. the coordinator keeps two gathers, for
/// the current operation and the next, since a worker that holds the current
/// result may contribute to the next before every worker has the current one.
#[derive(Debug)]
struct Collective {
    next_op: u32,
    pending: Option<(u32, OpKind)>,
    result: Option<u64>,
    gathers: [Gather; 2],
    /// results the coordinator holds back for a peer: the race-gate hook
    delay: Option<(WorkerId, Duration)>,
    held: Vec<(WorkerId, Header, [u8; OP_PAYLOAD], Instant)>,
}

/// the transfers of one phase from this worker's point of view.
pub struct PhaseSpec<'a> {
    pub epoch: Epoch,
    /// (transfer, destination worker)
    pub sends: &'a [(TransferId, WorkerId)],
    /// (transfer, source worker)
    pub receives: &'a [(TransferId, WorkerId)],
}

/// the open phase's bookkeeping. the vectors are allocated once with room for
/// every transfer and reused by every phase.
#[derive(Debug)]
struct Phase {
    open: bool,
    epoch: Epoch,
    sends: Vec<(TransferId, WorkerId)>,
    receives: Vec<TransferId>,
    /// sends per destination peer in this phase
    send_counts: Vec<u32>,
    /// peers whose grant for this phase has been consumed
    granted: Vec<bool>,
}

pub struct Fabric<L: Link> {
    me: WorkerId,
    session: SessionId,
    workers: usize,
    axes: usize,
    link: L,
    grants: GrantTable,
    slots: Vec<Slot>,
    /// the number of transfers this worker sends to `[peer][axis]` in any exchange point
    sends_per_peer_axis: Vec<Vec<u32>>,
    phase: Phase,
    scratch: Vec<u8>,
    inbox: Vec<u8>,
    stats: Stats,
    /// peers that have sent Done: their sockets may close from here on
    done_from: Vec<bool>,
    finishing: bool,
    collective: Collective,
    blocks: Blocks,
}

/// the checkpoint block path. a worker sends blocks to the coordinator under a byte
/// credit; the coordinator stages one block at a time and returns the credit once the
/// block is written.
#[derive(Debug)]
struct Blocks {
    /// bytes this worker may have in flight toward the coordinator
    credit: u32,
    outstanding: u32,
    next_seq: u32,
    /// the coordinator's staged block: sender, sequence, payload
    staged: Option<(WorkerId, u32)>,
    staged_payload: Vec<u8>,
    /// the coordinator's credit granted to each worker, for validating incoming blocks
    granted: Vec<u32>,
}

/// the pause between progress passes while waiting on the link.
const POLL_PAUSE: Duration = Duration::from_micros(50);

fn none_epoch() -> Epoch {
    Epoch {
        step: 0,
        attempt: 0,
        point: Epoch::NO_POINT,
        axis: Epoch::NO_AXIS,
    }
}

impl<L: Link> Fabric<L> {
    /// `lens` is every transfer's value count indexed by `TransferId`, sent or not;
    /// `sends_per_peer_axis[peer][axis]` the transfers this worker sends to `peer` in an
    /// axis phase, which is the count a valid grant from that peer must carry.
    pub fn new(
        link: L,
        me: WorkerId,
        session: SessionId,
        workers: usize,
        axes: usize,
        lens: &[usize],
        sends_per_peer_axis: Vec<Vec<u32>>,
    ) -> Self {
        assert_eq!(sends_per_peer_axis.len(), workers);
        assert!(sends_per_peer_axis.iter().all(|p| p.len() == axes));
        let largest = lens.iter().copied().max().unwrap_or(0) * 8;
        Self {
            me,
            session,
            workers,
            axes,
            link,
            grants: GrantTable::new(workers, axes),
            slots: lens
                .iter()
                .map(|&len| Slot {
                    buf: vec![0.0; len],
                    send: SendState::Idle,
                    recv: RecvState::Idle,
                    peer: None,
                })
                .collect(),
            sends_per_peer_axis,
            phase: Phase {
                open: false,
                epoch: none_epoch(),
                sends: Vec::with_capacity(lens.len()),
                receives: Vec::with_capacity(lens.len()),
                send_counts: vec![0; workers],
                granted: vec![false; workers],
            },
            scratch: Vec::with_capacity(largest),
            inbox: Vec::with_capacity(largest.max(HEADER_LEN)),
            stats: Stats::default(),
            done_from: vec![false; workers],
            finishing: false,
            collective: Collective {
                next_op: 0,
                pending: None,
                result: None,
                gathers: [Gather::fresh(0, workers), Gather::fresh(1, workers)],
                delay: None,
                held: Vec::new(),
            },
            blocks: Blocks {
                credit: 0,
                outstanding: 0,
                next_seq: 0,
                staged: None,
                staged_payload: Vec::new(),
                granted: vec![0; workers],
            },
        }
    }

    /// set the block credit agreed at the handshake: the bytes a worker may have in flight
    /// toward the coordinator. the inbox and the coordinator's staging grow to hold one
    /// block of that size, once, here.
    pub fn with_block_credit(mut self, credit: u32) -> Self {
        self.blocks.credit = credit;
        self.blocks.granted.iter_mut().for_each(|g| *g = credit);
        let need = credit as usize;
        if self.inbox.capacity() < need {
            self.inbox.reserve(need);
        }
        if self.is_coordinator() && self.blocks.staged_payload.capacity() < need {
            self.blocks.staged_payload.reserve(need);
        }
        self
    }

    pub fn block_credit(&self) -> u32 {
        self.blocks.credit
    }

    pub fn is_coordinator(&self) -> bool {
        self.me == WorkerId(0)
    }

    /// hold every Result frame for `peer` back by `duration`: the hook that makes a
    /// peer's grant reach a worker before its own result. coordinator only.
    pub fn delay_result_to(&mut self, peer: WorkerId, duration: Duration) {
        self.collective.delay = Some((peer, duration));
    }

    pub fn me(&self) -> WorkerId {
        self.me
    }

    pub fn session(&self) -> SessionId {
        self.session
    }

    pub fn link(&self) -> &L {
        &self.link
    }

    pub fn grants(&self) -> &GrantTable {
        &self.grants
    }

    pub fn stats(&self) -> Stats {
        self.stats
    }

    /// the bytes this endpoint holds in buffers: fixed for the session.
    pub fn buffer_bytes(&self) -> usize {
        self.slots
            .iter()
            .map(|s| s.buf.capacity() * 8)
            .sum::<usize>()
            + self.scratch.capacity()
            + self.inbox.capacity()
            + self.phase.sends.capacity() * std::mem::size_of::<(TransferId, WorkerId)>()
            + self.phase.receives.capacity() * std::mem::size_of::<TransferId>()
            + self.blocks.staged_payload.capacity()
    }

    fn local(&self, detail: impl Into<String>) -> FabricError {
        FabricError::protocol(self.me, detail)
    }

    fn slot(&self, id: TransferId) -> Result<&Slot, FabricError> {
        self.slots
            .get(id.0 as usize)
            .ok_or_else(|| self.local(format!("unknown transfer {id:?}")))
    }

    /// fill a send buffer. the buffer is leased from here until the phase closes, so a
    /// second pack before then is refused.
    pub fn pack(
        &mut self,
        id: TransferId,
        fill: impl FnOnce(&mut [f64]),
    ) -> Result<(), FabricError> {
        let state = self.slot(id)?.send;
        if state != SendState::Idle {
            return Err(self.local(format!("pack of {id:?} while its buffer is {state:?}")));
        }
        let slot = &mut self.slots[id.0 as usize];
        fill(&mut slot.buf);
        slot.send = SendState::Packed;
        Ok(())
    }

    /// open a phase: post its receives, grant every peer that feeds them, and register the
    /// packed sends. a send that was never packed, a transfer already in flight, a peer that
    /// has finished, or a phase already open is refused.
    pub fn open(&mut self, spec: PhaseSpec<'_>) -> Result<(), FabricError> {
        if self.phase.open {
            return Err(self.local(format!(
                "open of {:?} while {:?} is open",
                spec.epoch, self.phase.epoch
            )));
        }
        if self.finishing {
            return Err(self.local(format!("open of {:?} after finish began", spec.epoch)));
        }
        if spec.epoch.axis as usize >= self.axes {
            return Err(self.local(format!(
                "axis {} outside {} axes",
                spec.epoch.axis, self.axes
            )));
        }
        self.grants.open(spec.epoch)?;
        if let Some(&(_, peer)) = spec
            .receives
            .iter()
            .chain(spec.sends.iter())
            .find(|(_, peer)| self.done_from[peer.0 as usize])
        {
            return Err(FabricError::protocol(
                peer,
                format!("{:?} exchanges with a peer that has finished", spec.epoch),
            ));
        }
        for &(id, _) in spec.receives {
            let state = self.slot(id)?.recv;
            if state != RecvState::Idle {
                return Err(self.local(format!("receive of {id:?} posted while {state:?}")));
            }
        }
        for &(id, _) in spec.sends {
            let state = self.slot(id)?.send;
            if state != SendState::Packed {
                return Err(self.local(format!("send of {id:?} opened while {state:?}")));
            }
        }
        // every check passed: commit the phase into the reused bookkeeping.
        self.phase.send_counts.iter_mut().for_each(|c| *c = 0);
        self.phase.granted.iter_mut().for_each(|g| *g = false);
        self.phase.sends.clear();
        self.phase.receives.clear();
        for &(id, from) in spec.receives {
            let slot = &mut self.slots[id.0 as usize];
            slot.recv = RecvState::Expected;
            slot.peer = Some(from);
            self.phase.receives.push(id);
        }
        for &(id, to) in spec.sends {
            self.phase.send_counts[to.0 as usize] += 1;
            self.phase.sends.push((id, to));
        }
        for peer in 0..self.workers {
            let count = spec
                .receives
                .iter()
                .filter(|(_, from)| from.0 as usize == peer)
                .count() as u32;
            if count > 0 {
                let header = Header {
                    kind: Kind::Grant,
                    session: self.session,
                    epoch: spec.epoch,
                    id: count,
                    payload_len: 0,
                };
                self.link
                    .send(self.me, WorkerId(peer as u32), header, &[])?;
            }
        }
        self.phase.epoch = spec.epoch;
        self.phase.open = true;
        Ok(())
    }

    /// take every frame the link holds for this worker. runs inside `progress` and stands
    /// alone while the worker waits between phases, so grants from faster peers are tabled.
    pub fn service(&mut self) -> Result<(), FabricError> {
        self.release_held_results()?;
        self.tolerating_done(|link| link.drive(self.me))?;
        if self.blocks.staged.is_some() {
            return Ok(());
        }
        loop {
            let mut inbox = std::mem::take(&mut self.inbox);
            let polled = self.link.poll(self.me, &mut inbox);
            let result = match polled {
                Ok(Some((from, header))) => self.handle(from, header, &inbox).map(|()| true),
                Ok(None) => Ok(false),
                Err(e) => Err(e),
            };
            self.inbox = inbox;
            match result {
                Ok(true) => {
                    if self.blocks.staged.is_some() {
                        return Ok(());
                    }
                }
                Ok(false) => return Ok(()),
                Err(FabricError::Disconnected { peer, .. }) if self.done_from[peer.0 as usize] => {
                    self.link.detach(peer);
                }
                Err(e) => return Err(e),
            }
        }
    }

    /// run a link operation; a disconnect from a peer that has already sent Done is the
    /// peer leaving the session, so its slot is detached and the operation counts as done.
    fn tolerating_done(
        &self,
        op: impl Fn(&L) -> Result<(), FabricError>,
    ) -> Result<(), FabricError> {
        match op(&self.link) {
            Err(FabricError::Disconnected { peer, .. }) if self.done_from[peer.0 as usize] => {
                self.link.detach(peer);
                Ok(())
            }
            other => other,
        }
    }

    /// service the link for `duration`: what a worker does while it waits for a collective
    /// result, so a faster peer's grant is tabled rather than left on the wire.
    pub fn service_for(&mut self, duration: Duration) -> Result<(), FabricError> {
        let start = Instant::now();
        while start.elapsed() < duration {
            self.service()?;
            std::thread::sleep(POLL_PAUSE);
        }
        self.service()
    }

    fn handle(
        &mut self,
        from: WorkerId,
        header: Header,
        payload: &[u8],
    ) -> Result<(), FabricError> {
        if header.session != self.session {
            return Err(FabricError::protocol(
                from,
                format!("frame from session {:?}", header.session),
            ));
        }
        if payload.len() != header.payload_len as usize {
            return Err(FabricError::protocol(
                from,
                format!(
                    "payload of {} bytes under a header stating {}",
                    payload.len(),
                    header.payload_len
                ),
            ));
        }
        match header.kind {
            Kind::Grant => {
                let axis = header.epoch.axis as usize;
                let expected = self
                    .sends_per_peer_axis
                    .get(from.0 as usize)
                    .and_then(|p| p.get(axis))
                    .copied()
                    .unwrap_or(0);
                if expected == 0 {
                    return Err(FabricError::protocol(
                        from,
                        format!(
                            "grant for {:?}, a phase in which this worker sends it nothing",
                            header.epoch
                        ),
                    ));
                }
                if header.id != expected {
                    return Err(FabricError::protocol(
                        from,
                        format!(
                            "grant for {:?} expects {} transfers; this worker sends {expected}",
                            header.epoch, header.id
                        ),
                    ));
                }
                self.grants.table(from, header.epoch, header.id)
            }
            Kind::Halo => {
                if !self.phase.open {
                    return Err(FabricError::protocol(
                        from,
                        format!(
                            "halo {} for {:?} with no phase open",
                            header.id, header.epoch
                        ),
                    ));
                }
                if header.epoch != self.phase.epoch {
                    return Err(FabricError::protocol(
                        from,
                        format!(
                            "halo {} for {:?} while {:?} is open",
                            header.id, header.epoch, self.phase.epoch
                        ),
                    ));
                }
                let id = TransferId(header.id);
                let Some(slot) = self.slots.get(id.0 as usize) else {
                    return Err(FabricError::protocol(
                        from,
                        format!("halo for unknown transfer {id:?}"),
                    ));
                };
                if slot.recv != RecvState::Expected {
                    return Err(FabricError::protocol(
                        from,
                        format!("halo for {id:?} while its receive is {:?}", slot.recv),
                    ));
                }
                if slot.peer != Some(from) {
                    return Err(FabricError::protocol(
                        from,
                        format!("halo for {id:?} expected from {:?}", slot.peer),
                    ));
                }
                if header.payload_len as usize != 8 * slot.buf.len() {
                    return Err(FabricError::protocol(
                        from,
                        format!(
                            "halo for {id:?} carries {} bytes for {} values",
                            header.payload_len,
                            slot.buf.len()
                        ),
                    ));
                }
                let slot = &mut self.slots[id.0 as usize];
                decode_f64s(payload, &mut slot.buf);
                slot.recv = RecvState::Received;
                Ok(())
            }
            Kind::Contribute => {
                if !self.is_coordinator() {
                    return Err(FabricError::protocol(from, "Contribute sent to a worker"));
                }
                let Some((kind, value)) = decode_op(payload) else {
                    return Err(FabricError::protocol(from, "malformed Contribute payload"));
                };
                self.table_contribution(from, header.id, kind, value)
            }
            Kind::Result => {
                if self.is_coordinator() {
                    return Err(FabricError::protocol(
                        from,
                        "Result sent to the coordinator",
                    ));
                }
                if from != WorkerId(0) {
                    return Err(FabricError::protocol(from, "Result from a worker"));
                }
                let Some((kind, value)) = decode_op(payload) else {
                    return Err(FabricError::protocol(from, "malformed Result payload"));
                };
                match self.collective.pending {
                    Some((op, k)) if op == header.id && k == kind => {
                        self.collective.result = Some(value);
                        Ok(())
                    }
                    other => Err(FabricError::protocol(
                        from,
                        format!(
                            "Result for op {} ({kind:?}) while waiting on {other:?}",
                            header.id
                        ),
                    )),
                }
            }
            Kind::Block => {
                if !self.is_coordinator() {
                    return Err(FabricError::protocol(from, "Block sent to a worker"));
                }
                if self.blocks.staged.is_some() {
                    return Err(FabricError::protocol(from, "Block while one is staged"));
                }
                let len = header.payload_len;
                if len > self.blocks.granted[from.0 as usize] {
                    return Err(FabricError::protocol(
                        from,
                        format!(
                            "block of {len} bytes exceeds its credit of {}",
                            self.blocks.granted[from.0 as usize]
                        ),
                    ));
                }
                self.blocks.granted[from.0 as usize] -= len;
                self.blocks.staged_payload.clear();
                self.blocks.staged_payload.extend_from_slice(payload);
                self.blocks.staged = Some((from, header.id));
                Ok(())
            }
            Kind::Credit => {
                if self.is_coordinator() || from != WorkerId(0) {
                    return Err(FabricError::protocol(from, "Credit from a worker"));
                }
                if header.id > self.blocks.outstanding {
                    return Err(FabricError::protocol(
                        from,
                        format!(
                            "credit of {} returned against {} outstanding",
                            header.id, self.blocks.outstanding
                        ),
                    ));
                }
                self.blocks.outstanding -= header.id;
                Ok(())
            }
            Kind::Abort => Err(FabricError::PeerAborted {
                peer: from,
                reason: AbortReason::Protocol,
                detail: String::from_utf8_lossy(payload).into_owned(),
            }),
            Kind::Done => {
                if self.done_from[from.0 as usize] {
                    return Err(FabricError::protocol(from, "duplicate Done"));
                }
                // a peer may finish while this worker still has a phase open, as long as
                // nothing in that phase depends on it.
                if self.phase.open {
                    let owed = self.phase.receives.iter().any(|id| {
                        let s = &self.slots[id.0 as usize];
                        s.recv == RecvState::Expected && s.peer == Some(from)
                    }) || self.phase.sends.iter().any(|(id, to)| {
                        *to == from && self.slots[id.0 as usize].send != SendState::Sent
                    });
                    if owed {
                        return Err(FabricError::protocol(
                            from,
                            format!("Done while {:?} still exchanges with it", self.phase.epoch),
                        ));
                    }
                }
                self.done_from[from.0 as usize] = true;
                Ok(())
            }
            other => Err(FabricError::protocol(
                from,
                format!("unexpected {other:?} frame"),
            )),
        }
    }

    /// the coordinator records one worker's contribution to `op`; the operation completes
    /// once every worker has contributed, and its result goes to every peer and to the
    /// coordinator's own pending slot. a contribution beyond the next operation, a duplicate,
    /// a kind that differs from the others', or a Min value outside the finite positive
    /// range is a protocol error.
    fn table_contribution(
        &mut self,
        from: WorkerId,
        op: u32,
        kind: OpKind,
        value: u64,
    ) -> Result<(), FabricError> {
        let workers = self.workers;
        let current = self.collective.gathers[0].op;
        let slot = if op == current {
            0
        } else if op == current + 1 {
            1
        } else {
            return Err(FabricError::protocol(
                from,
                format!("contribution to op {op} while the coordinator gathers {current}"),
            ));
        };
        if kind == OpKind::Min {
            let v = f64::from_bits(value);
            if !v.is_finite() || v <= 0.0 {
                return Err(FabricError::protocol(
                    from,
                    format!("Min contribution {v:e}"),
                ));
            }
        }
        let gather = &mut self.collective.gathers[slot];
        match gather.kind {
            None => gather.kind = Some(kind),
            Some(k) if k == kind => {}
            Some(k) => {
                return Err(FabricError::protocol(
                    from,
                    format!("op {op} contributed as {kind:?} against {k:?}"),
                ));
            }
        }
        if gather.values[from.0 as usize].is_some() {
            return Err(FabricError::protocol(
                from,
                format!("duplicate contribution to op {op}"),
            ));
        }
        gather.values[from.0 as usize] = Some(value);
        gather.count += 1;
        if slot == 0 && gather.count == workers {
            let kind = gather.kind.expect("set with the first contribution");
            let values = gather
                .values
                .iter()
                .map(|v| v.expect("every worker contributed"));
            let result = match kind {
                OpKind::Min => values
                    .map(f64::from_bits)
                    .fold(f64::INFINITY, f64::min)
                    .to_bits(),
                OpKind::Any => u64::from(values.into_iter().any(|v| v != 0)),
                OpKind::CheckpointOpen | OpKind::CheckpointClose => {
                    u64::from(values.into_iter().all(|v| v != 0))
                }
            };
            let header = Header {
                kind: Kind::Result,
                session: self.session,
                epoch: none_epoch(),
                id: op,
                payload_len: OP_PAYLOAD as u32,
            };
            let payload = encode_op(kind, result);
            for peer in 1..workers {
                let peer = WorkerId(peer as u32);
                match self.collective.delay {
                    Some((held, duration)) if held == peer => {
                        self.collective.held.push((
                            peer,
                            header,
                            payload,
                            Instant::now() + duration,
                        ));
                    }
                    _ => self.link.send(self.me, peer, header, &payload)?,
                }
            }
            if let Some((pending, _)) = self.collective.pending {
                if pending == op {
                    self.collective.result = Some(result);
                }
            }
            self.collective.gathers.swap(0, 1);
            self.collective.gathers[1].reset(op + 2);
        }
        Ok(())
    }

    fn release_held_results(&mut self) -> Result<(), FabricError> {
        if self.collective.held.is_empty() {
            return Ok(());
        }
        let now = Instant::now();
        let mut i = 0;
        while i < self.collective.held.len() {
            if self.collective.held[i].3 <= now {
                let (peer, header, payload, _) = self.collective.held.remove(i);
                self.link.send(self.me, peer, header, &payload)?;
            } else {
                i += 1;
            }
        }
        Ok(())
    }

    /// issue the next collective operation with this worker's contribution. the result
    /// arrives through `collective_progress`.
    pub fn begin_collective(&mut self, kind: OpKind, value: u64) -> Result<u32, FabricError> {
        if let Some((op, k)) = self.collective.pending {
            return Err(self.local(format!(
                "collective {kind:?} begun while op {op} ({k:?}) is pending"
            )));
        }
        if kind == OpKind::Min {
            let v = f64::from_bits(value);
            if !v.is_finite() || v <= 0.0 {
                return Err(self.local(format!("Min contribution {v:e}")));
            }
        }
        let op = self.collective.next_op;
        self.collective.next_op += 1;
        self.collective.pending = Some((op, kind));
        self.collective.result = None;
        if self.is_coordinator() {
            self.table_contribution(self.me, op, kind, value)?;
        } else {
            let header = Header {
                kind: Kind::Contribute,
                session: self.session,
                epoch: none_epoch(),
                id: op,
                payload_len: OP_PAYLOAD as u32,
            };
            self.link
                .send(self.me, WorkerId(0), header, &encode_op(kind, value))?;
        }
        Ok(op)
    }

    /// service the link; the result bits once the pending operation has completed.
    pub fn collective_progress(&mut self) -> Result<Option<u64>, FabricError> {
        if self.collective.pending.is_none() {
            return Err(self.local("collective progress with no operation pending"));
        }
        self.service()?;
        if let Some(result) = self.collective.result.take() {
            self.collective.pending = None;
            return Ok(Some(result));
        }
        Ok(None)
    }

    /// a complete collective: contribute, then service until the result arrives or
    /// `deadline` passes. the result is the coordinator's bits, consumed as sent.
    pub fn collective(
        &mut self,
        kind: OpKind,
        value: u64,
        deadline: Duration,
    ) -> Result<u64, FabricError> {
        let op = self.begin_collective(kind, value)?;
        let start = Instant::now();
        loop {
            if let Some(result) = self.collective_progress()? {
                return Ok(result);
            }
            if start.elapsed() > deadline {
                return Err(FabricError::Deadline {
                    phase: Deadline::Transfer,
                    peer: None,
                    epoch: None,
                    pending: 1 + usize::from(op == 0),
                });
            }
            std::thread::sleep(POLL_PAUSE);
        }
    }

    /// send one checkpoint block to the coordinator, waiting for credit up to `deadline`.
    /// the coordinator writes its own blocks directly and never calls this.
    pub fn send_block(&mut self, payload: &[u8], deadline: Duration) -> Result<(), FabricError> {
        if self.is_coordinator() {
            return Err(self.local("send_block on the coordinator"));
        }
        let len = payload.len() as u32;
        if len > self.blocks.credit {
            return Err(self.local(format!(
                "block of {len} bytes exceeds the credit of {}",
                self.blocks.credit
            )));
        }
        // both the byte credit and the link's payload slot must be free: two blocks can fit
        // the credit while the first still occupies the outbox
        let start = Instant::now();
        while self.blocks.outstanding + len > self.blocks.credit || !self.link.ready(WorkerId(0)) {
            self.service()?;
            if start.elapsed() > deadline {
                return Err(FabricError::Deadline {
                    phase: Deadline::Checkpoint,
                    peer: Some(WorkerId(0)),
                    epoch: None,
                    pending: 1,
                });
            }
            std::thread::sleep(POLL_PAUSE);
        }
        let header = Header {
            kind: Kind::Block,
            session: self.session,
            epoch: none_epoch(),
            id: self.blocks.next_seq,
            payload_len: len,
        };
        self.blocks.next_seq += 1;
        self.blocks.outstanding += len;
        self.link.send(self.me, WorkerId(0), header, payload)?;
        self.link.drive(self.me)
    }

    /// the coordinator's staged block, if one arrived: the sender, its sequence number, and
    /// the payload. the block stays staged, and no further frame is taken, until
    /// `release_block` returns the credit.
    pub fn staged_block(&self) -> Option<(WorkerId, u32, &[u8])> {
        self.blocks
            .staged
            .map(|(from, seq)| (from, seq, self.blocks.staged_payload.as_slice()))
    }

    /// the coordinator has written the staged block: return its credit to the sender.
    pub fn release_block(&mut self) -> Result<(), FabricError> {
        let Some((from, _)) = self.blocks.staged.take() else {
            return Err(self.local("release with no block staged"));
        };
        let len = self.blocks.staged_payload.len() as u32;
        self.blocks.granted[from.0 as usize] += len;
        let header = Header {
            kind: Kind::Credit,
            session: self.session,
            epoch: none_epoch(),
            id: len,
            payload_len: 0,
        };
        self.link.send(self.me, from, header, &[])
    }

    /// the coordinator's count of contributions received for the current operation, for a
    /// coordinator that contributes last.
    pub fn contributions_to_current(&self) -> usize {
        self.collective.gathers[0].count
    }

    /// whether every contribution received so far for the current operation is nonzero:
    /// the workers' verdict, read by a coordinator before it acts and votes last.
    pub fn current_contributions_all_nonzero(&self) -> bool {
        self.collective.gathers[0]
            .values
            .iter()
            .flatten()
            .all(|&v| v != 0)
    }

    /// wait until every worker's block traffic and this worker's own sends have drained,
    /// up to `deadline`: the coordinator calls this before verifying the coverage ledger.
    pub fn drain_blocks(&mut self, deadline: Duration) -> Result<(), FabricError> {
        let start = Instant::now();
        while self.blocks.outstanding > 0 || self.link.outbound_pending() {
            self.service()?;
            if start.elapsed() > deadline {
                return Err(FabricError::Deadline {
                    phase: Deadline::Checkpoint,
                    peer: None,
                    epoch: None,
                    pending: self.blocks.outstanding as usize,
                });
            }
            std::thread::sleep(POLL_PAUSE);
        }
        Ok(())
    }

    /// service the link and push the open phase forward: sends leave once their peer's
    /// grant is in hand and the link can take the frame. `Done` when every send has left
    /// and every receive has landed.
    pub fn progress(&mut self) -> Result<Progress, FabricError> {
        self.stats.passes += 1;
        self.service()?;
        if !self.phase.open {
            return Err(self.local("progress with no phase open"));
        }
        let epoch = self.phase.epoch;
        let mut all_sent = true;
        for k in 0..self.phase.sends.len() {
            let (id, to) = self.phase.sends[k];
            let p = to.0 as usize;
            if self.slots[id.0 as usize].send == SendState::Sent {
                continue;
            }
            if !self.phase.granted[p] {
                match self.grants.consume(to, epoch) {
                    Some(count) => {
                        if count != self.phase.send_counts[p] {
                            let sends = self.phase.send_counts[p];
                            return Err(FabricError::protocol(
                                to,
                                format!(
                                    "grant carries {count} transfers; this phase sends it {sends}"
                                ),
                            ));
                        }
                        self.phase.granted[p] = true;
                    }
                    None => {
                        self.stats.grant_waits += 1;
                        all_sent = false;
                        continue;
                    }
                }
            }
            if !self.link.ready(to) {
                self.stats.link_waits += 1;
                all_sent = false;
                continue;
            }
            let slot = &mut self.slots[id.0 as usize];
            encode_f64s(&slot.buf, &mut self.scratch);
            let header = Header {
                kind: Kind::Halo,
                session: self.session,
                epoch,
                id: id.0,
                payload_len: self.scratch.len() as u32,
            };
            self.link.send(self.me, to, header, &self.scratch)?;
            slot.send = SendState::Sent;
        }
        let all_received = self.phase.receives.iter().all(|id| {
            matches!(
                self.slots[id.0 as usize].recv,
                RecvState::Received | RecvState::Unpacked
            )
        });
        self.tolerating_done(|link| link.drive(self.me))?;
        Ok(if all_sent && all_received {
            Progress::Done
        } else {
            Progress::Pending
        })
    }

    /// progress until the open phase is done or `deadline` passes.
    pub fn wait(&mut self, deadline: Duration) -> Result<(), FabricError> {
        let start = Instant::now();
        loop {
            if self.progress()? == Progress::Done {
                return Ok(());
            }
            if start.elapsed() > deadline {
                let pending = self
                    .phase
                    .sends
                    .iter()
                    .filter(|(id, _)| self.slots[id.0 as usize].send != SendState::Sent)
                    .count()
                    + self
                        .phase
                        .receives
                        .iter()
                        .filter(|id| self.slots[id.0 as usize].recv == RecvState::Expected)
                        .count();
                return Err(FabricError::Deadline {
                    phase: Deadline::Transfer,
                    peer: None,
                    epoch: Some(self.phase.epoch),
                    pending,
                });
            }
            std::thread::sleep(POLL_PAUSE);
        }
    }

    /// a received payload, for the caller to scatter.
    pub fn payload(&self, id: TransferId) -> Result<&[f64], FabricError> {
        let slot = self.slot(id)?;
        if slot.recv != RecvState::Received {
            return Err(self.local(format!("payload of {id:?} read while {:?}", slot.recv)));
        }
        Ok(&slot.buf)
    }

    pub fn mark_unpacked(&mut self, id: TransferId) -> Result<(), FabricError> {
        let state = self.slot(id)?.recv;
        if state != RecvState::Received {
            return Err(self.local(format!("unpack of {id:?} while {state:?}")));
        }
        self.slots[id.0 as usize].recv = RecvState::Unpacked;
        Ok(())
    }

    /// close the open phase: every send has left and every receive has been unpacked; every
    /// buffer returns to idle.
    pub fn close(&mut self) -> Result<(), FabricError> {
        if !self.phase.open {
            return Err(self.local("close with no phase open"));
        }
        for &(id, _) in &self.phase.sends {
            let s = self.slots[id.0 as usize].send;
            if s != SendState::Sent {
                return Err(self.local(format!("close with send {id:?} {s:?}")));
            }
        }
        for &id in &self.phase.receives {
            let r = self.slots[id.0 as usize].recv;
            if r != RecvState::Unpacked {
                return Err(self.local(format!("close with receive {id:?} {r:?}")));
            }
        }
        for &(id, _) in &self.phase.sends {
            self.slots[id.0 as usize].send = SendState::Idle;
        }
        for &id in &self.phase.receives {
            let slot = &mut self.slots[id.0 as usize];
            slot.recv = RecvState::Idle;
            slot.peer = None;
        }
        self.phase.sends.clear();
        self.phase.receives.clear();
        self.phase.open = false;
        Ok(())
    }

    pub fn is_open(&self) -> bool {
        self.phase.open
    }

    /// tell every peer this worker is terminating. best effort: a peer already gone is not
    /// an error here, since the session is ending either way.
    pub fn abort(&mut self, detail: &str) {
        let header = Header {
            kind: Kind::Abort,
            session: self.session,
            epoch: none_epoch(),
            id: 0,
            payload_len: detail.len() as u32,
        };
        for peer in 0..self.workers {
            let peer = WorkerId(peer as u32);
            if peer != self.me {
                let _ = self.link.send(self.me, peer, header, detail.as_bytes());
            }
        }
        let start = Instant::now();
        while start.elapsed() < Duration::from_millis(200) {
            if self.link.drive(self.me).is_err() || !self.link.outbound_pending() {
                break;
            }
            std::thread::sleep(POLL_PAUSE);
        }
    }

    /// the local end-of-session invariants: no phase open and no grant left tabled.
    pub fn shutdown(&self) -> Result<(), FabricError> {
        if self.phase.open {
            return Err(self.local("shutdown with a phase open"));
        }
        self.grants.assert_drained()
    }

    /// start leaving the session: the local invariants of `shutdown` hold and every peer is
    /// told Done. a socket may close only after both sides have said Done and this worker's
    /// own frames have all left, so a peer that is still driving the link never reads an
    /// unexpected end of file.
    pub fn begin_finish(&mut self) -> Result<(), FabricError> {
        self.shutdown()?;
        if self.collective.pending.is_some() {
            return Err(self.local("finish with a collective pending"));
        }
        if self.finishing {
            return Err(self.local("finish begun twice"));
        }
        self.finishing = true;
        let header = Header {
            kind: Kind::Done,
            session: self.session,
            epoch: none_epoch(),
            id: 0,
            payload_len: 0,
        };
        for peer in 0..self.workers {
            let peer = WorkerId(peer as u32);
            if peer != self.me {
                self.link.send(self.me, peer, header, &[])?;
            }
        }
        Ok(())
    }

    /// service the link; `Done` once every peer has said Done and nothing this worker sent
    /// is still queued or partially written.
    pub fn finish_progress(&mut self) -> Result<Progress, FabricError> {
        if !self.finishing {
            return Err(self.local("finish progress before begin_finish"));
        }
        self.service()?;
        let all_in = (0..self.workers).all(|p| p == self.me.0 as usize || self.done_from[p]);
        Ok(if all_in && !self.link.outbound_pending() {
            Progress::Done
        } else {
            Progress::Pending
        })
    }

    /// leave the session: Done to every peer, Done from every peer, own frames drained,
    /// within `deadline`.
    pub fn finish(&mut self, deadline: Duration) -> Result<(), FabricError> {
        self.begin_finish()?;
        let start = Instant::now();
        loop {
            if self.finish_progress()? == Progress::Done {
                return Ok(());
            }
            if start.elapsed() > deadline {
                let pending = (0..self.workers)
                    .filter(|&p| p != self.me.0 as usize && !self.done_from[p])
                    .count()
                    + usize::from(self.link.outbound_pending());
                return Err(FabricError::Deadline {
                    phase: Deadline::Shutdown,
                    peer: None,
                    epoch: None,
                    pending,
                });
            }
            std::thread::sleep(POLL_PAUSE);
        }
    }
}
