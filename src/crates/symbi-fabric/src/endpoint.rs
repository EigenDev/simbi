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
        }
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
        self.tolerating_done(|link| link.drive(self.me))?;
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
                Ok(true) => {}
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
