// =============================================================================
// endpoint.rs
//
// one worker's end of the fabric: the per-transfer buffers and their send and
// receive states, the grant table, and the phase protocol that moves halos
// between workers. a phase opens by posting receives and granting the peers
// that feed them, packs its sends, and then makes progress until every send
// has left and every receive has landed; the caller scatters received payloads
// and closes the phase, which frees every buffer for the next epoch.
//
// the endpoint moves f64 bit patterns under transfer ids and never sees a
// field or a region. `Link` is the byte-frame carrier beneath it: the in-process
// `Loopback` here, a socket in the network adapter.
//
// usage:
//  let mut fabric = Fabric::new(link, me, session, workers, axes, &lens, sends_per_peer_axis);
//  fabric.pack(id, |buf| ...)?;
//  fabric.open(PhaseSpec { epoch, sends: &[(id, to)], receives: &[(id, from)] })?;
//  while fabric.progress()? == Progress::Pending {}
//  let values = fabric.payload(id)?; ... fabric.mark_unpacked(id)?;
//  fabric.close()?;
// =============================================================================

use crate::error::{AbortReason, FabricError};
use crate::frame::{HEADER_LEN, Header, Kind, decode_f64s, encode_f64s};
use crate::grant::GrantTable;
use crate::ident::{Epoch, SessionId, TransferId, WorkerId};
use std::cell::RefCell;
use std::collections::VecDeque;
use std::rc::Rc;

/// a frame carrier between workers. delivery order between one ordered pair of
/// workers is preserved.
pub trait Link {
    fn send(&self, from: WorkerId, to: WorkerId, header: Header, payload: &[u8]);
    fn poll(&self, me: WorkerId) -> Option<(WorkerId, Header, Vec<u8>)>;
}

type Queued = (WorkerId, WorkerId, Header, Vec<u8>);

/// the in-process reference link: every frame goes through one queue and a
/// worker takes the oldest frame addressed to it. frames are encoded to and
/// decoded from bytes on the way, so the codec is on the path.
#[derive(Clone, Default)]
pub struct Loopback(Rc<RefCell<VecDeque<Queued>>>);

impl Loopback {
    pub fn new() -> Self {
        Self::default()
    }

    /// place a frame as if `from` had sent it: the test hook for hostile frames.
    pub fn inject(&self, from: WorkerId, to: WorkerId, header: Header, payload: &[u8]) {
        self.send(from, to, header, payload);
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
    fn send(&self, from: WorkerId, to: WorkerId, header: Header, payload: &[u8]) {
        let mut bytes = [0u8; HEADER_LEN];
        header.encode(&mut bytes);
        let decoded = Header::decode(&bytes).expect("an encoded header decodes");
        self.0
            .borrow_mut()
            .push_back((from, to, decoded, payload.to_vec()));
    }

    fn poll(&self, me: WorkerId) -> Option<(WorkerId, Header, Vec<u8>)> {
        let mut q = self.0.borrow_mut();
        let pos = q.iter().position(|(_, to, _, _)| *to == me)?;
        q.remove(pos).map(|(from, _, h, p)| (from, h, p))
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

/// the transfers of one phase from this worker's point of view.
pub struct PhaseSpec<'a> {
    pub epoch: Epoch,
    /// (transfer, destination worker)
    pub sends: &'a [(TransferId, WorkerId)],
    /// (transfer, source worker)
    pub receives: &'a [(TransferId, WorkerId)],
}

#[derive(Debug)]
struct OpenPhase {
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
    phase: Option<OpenPhase>,
    scratch: Vec<u8>,
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
            phase: None,
            scratch: Vec::new(),
        }
    }

    pub fn me(&self) -> WorkerId {
        self.me
    }

    pub fn link(&self) -> &L {
        &self.link
    }

    pub fn grants(&self) -> &GrantTable {
        &self.grants
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
    /// packed sends. a send that was never packed, a transfer already in flight, or a phase
    /// already open is refused.
    pub fn open(&mut self, spec: PhaseSpec<'_>) -> Result<(), FabricError> {
        if let Some(p) = &self.phase {
            return Err(self.local(format!(
                "open of {:?} while {:?} is open",
                spec.epoch, p.epoch
            )));
        }
        if spec.epoch.axis as usize >= self.axes {
            return Err(self.local(format!(
                "axis {} outside {} axes",
                spec.epoch.axis, self.axes
            )));
        }
        self.grants.open(spec.epoch)?;
        let mut expect_counts = vec![0u32; self.workers];
        for &(id, from) in spec.receives {
            let state = self.slot(id)?.recv;
            if state != RecvState::Idle {
                return Err(self.local(format!("receive of {id:?} posted while {state:?}")));
            }
            let slot = &mut self.slots[id.0 as usize];
            slot.recv = RecvState::Expected;
            slot.peer = Some(from);
            expect_counts[from.0 as usize] += 1;
        }
        let mut send_counts = vec![0u32; self.workers];
        for &(id, to) in spec.sends {
            let state = self.slot(id)?.send;
            if state != SendState::Packed {
                return Err(self.local(format!("send of {id:?} opened while {state:?}")));
            }
            send_counts[to.0 as usize] += 1;
        }
        for (peer, &count) in expect_counts.iter().enumerate() {
            if count > 0 {
                let header = Header {
                    kind: Kind::Grant,
                    session: self.session,
                    epoch: spec.epoch,
                    id: count,
                    payload_len: 0,
                };
                self.link.send(self.me, WorkerId(peer as u32), header, &[]);
            }
        }
        self.phase = Some(OpenPhase {
            epoch: spec.epoch,
            sends: spec.sends.to_vec(),
            receives: spec.receives.iter().map(|&(id, _)| id).collect(),
            send_counts,
            granted: vec![false; self.workers],
        });
        Ok(())
    }

    /// take every frame the link holds for this worker. runs inside `progress` and stands
    /// alone while the worker waits between phases, so grants from faster peers are tabled.
    pub fn service(&mut self) -> Result<(), FabricError> {
        while let Some((from, header, payload)) = self.link.poll(self.me) {
            self.handle(from, header, &payload)?;
        }
        Ok(())
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
                let Some(phase) = self.phase.as_ref() else {
                    return Err(FabricError::protocol(
                        from,
                        format!(
                            "halo {} for {:?} with no phase open",
                            header.id, header.epoch
                        ),
                    ));
                };
                if header.epoch != phase.epoch {
                    return Err(FabricError::protocol(
                        from,
                        format!(
                            "halo {} for {:?} while {:?} is open",
                            header.id, header.epoch, phase.epoch
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
            other => Err(FabricError::protocol(
                from,
                format!("unexpected {other:?} frame"),
            )),
        }
    }

    /// service the link and push the open phase forward: sends leave once their peer's
    /// grant is in hand. `Done` when every send has left and every receive has landed.
    pub fn progress(&mut self) -> Result<Progress, FabricError> {
        self.service()?;
        let Some(phase) = self.phase.take() else {
            return Err(self.local("progress with no phase open"));
        };
        let mut phase = phase;
        let mut all_sent = true;
        for &(id, to) in &phase.sends {
            let p = to.0 as usize;
            if self.slots[id.0 as usize].send == SendState::Sent {
                continue;
            }
            if !phase.granted[p] {
                match self.grants.consume(to, phase.epoch) {
                    Some(count) => {
                        if count != phase.send_counts[p] {
                            self.phase = Some(phase);
                            return Err(FabricError::protocol(
                                to,
                                format!(
                                    "grant carries {count} transfers; this phase sends it {}",
                                    self.phase.as_ref().map_or(0, |ph| ph.send_counts[p])
                                ),
                            ));
                        }
                        phase.granted[p] = true;
                    }
                    None => {
                        all_sent = false;
                        continue;
                    }
                }
            }
            let slot = &mut self.slots[id.0 as usize];
            encode_f64s(&slot.buf, &mut self.scratch);
            let header = Header {
                kind: Kind::Halo,
                session: self.session,
                epoch: phase.epoch,
                id: id.0,
                payload_len: self.scratch.len() as u32,
            };
            self.link.send(self.me, to, header, &self.scratch);
            slot.send = SendState::Sent;
        }
        let all_received = phase.receives.iter().all(|id| {
            matches!(
                self.slots[id.0 as usize].recv,
                RecvState::Received | RecvState::Unpacked
            )
        });
        self.phase = Some(phase);
        Ok(if all_sent && all_received {
            Progress::Done
        } else {
            Progress::Pending
        })
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
        let Some(phase) = self.phase.as_ref() else {
            return Err(self.local("close with no phase open"));
        };
        for &(id, _) in &phase.sends {
            let s = self.slots[id.0 as usize].send;
            if s != SendState::Sent {
                return Err(self.local(format!("close with send {id:?} {s:?}")));
            }
        }
        for &id in &phase.receives {
            let r = self.slots[id.0 as usize].recv;
            if r != RecvState::Unpacked {
                return Err(self.local(format!("close with receive {id:?} {r:?}")));
            }
        }
        let phase = self.phase.take().expect("checked above");
        for (id, _) in phase.sends {
            self.slots[id.0 as usize].send = SendState::Idle;
        }
        for id in phase.receives {
            let slot = &mut self.slots[id.0 as usize];
            slot.recv = RecvState::Idle;
            slot.peer = None;
        }
        Ok(())
    }

    pub fn is_open(&self) -> bool {
        self.phase.is_some()
    }

    /// the end of the session: no phase open and no grant left tabled.
    pub fn shutdown(&self) -> Result<(), FabricError> {
        if self.phase.is_some() {
            return Err(self.local("shutdown with a phase open"));
        }
        self.grants.assert_drained()
    }
}
