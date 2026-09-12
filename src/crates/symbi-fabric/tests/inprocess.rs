// =============================================================================
// inprocess.rs
//
// the phase protocol between two in-process endpoints over the loopback link,
// and the frames it refuses: a halo without a grant, a halo for another epoch
// or of the wrong length, a duplicate halo, a grant with the wrong count or for
// a phase this worker never sends in, a buffer packed twice. every refusal is a
// protocol error; every valid sequence, including a grant that arrives before
// the receiver has opened the phase, completes with no error.
//
// run: cargo test -p symbi-fabric --test inprocess
// =============================================================================

use symbi_fabric::frame::encode_f64s;
use symbi_fabric::{
    Epoch, Fabric, FabricError, Header, Kind, Loopback, OpKind, PhaseSpec, Progress, SessionId,
    TransferId, WorkerId,
};

const S: SessionId = SessionId(99);
const A: WorkerId = WorkerId(0);
const B: WorkerId = WorkerId(1);
const C: WorkerId = WorkerId(2);
// transfer 0: A -> B on axis 0; 1: B -> A on axis 0; 2: A -> B on axis 1; 3: B -> A on axis 1.
const LENS: [usize; 4] = [3, 3, 2, 2];
const T0: TransferId = TransferId(0);
const T1: TransferId = TransferId(1);
const T2: TransferId = TransferId(2);
const T3: TransferId = TransferId(3);

fn epoch(attempt: u16, point: u8, axis: u8) -> Epoch {
    Epoch {
        step: 3,
        attempt,
        point,
        axis,
    }
}

fn pair(link: &Loopback) -> (Fabric<Loopback>, Fabric<Loopback>) {
    let a = Fabric::new(
        link.clone(),
        A,
        S,
        3,
        2,
        &LENS,
        vec![vec![0, 0], vec![1, 1], vec![0, 0]],
    );
    let b = Fabric::new(
        link.clone(),
        B,
        S,
        3,
        2,
        &LENS,
        vec![vec![1, 1], vec![0, 0], vec![0, 0]],
    );
    (a, b)
}

fn halo(epoch: Epoch, id: TransferId, values: &[f64]) -> (Header, Vec<u8>) {
    let mut bytes = Vec::new();
    encode_f64s(values, &mut bytes);
    (
        Header {
            kind: Kind::Halo,
            session: S,
            epoch,
            id: id.0,
            payload_len: bytes.len() as u32,
        },
        bytes,
    )
}

fn grant(epoch: Epoch, count: u32) -> Header {
    Header {
        kind: Kind::Grant,
        session: S,
        epoch,
        id: count,
        payload_len: 0,
    }
}

/// interleave the endpoints until every one reports done; a stall is a failure.
fn drive(fabrics: &mut [&mut Fabric<Loopback>]) -> Result<(), FabricError> {
    for _ in 0..16 {
        let mut done = true;
        for f in fabrics.iter_mut() {
            if f.progress()? == Progress::Pending {
                done = false;
            }
        }
        if done {
            return Ok(());
        }
    }
    panic!("the phase made no progress in 16 rounds");
}

fn open_axis0(a: &mut Fabric<Loopback>, b: &mut Fabric<Loopback>, e: Epoch) {
    a.pack(T0, |buf| buf.copy_from_slice(&[1.0, 2.0, 3.0]))
        .unwrap();
    b.pack(T1, |buf| buf.copy_from_slice(&[-1.0, -2.0, -3.0]))
        .unwrap();
    a.open(PhaseSpec {
        epoch: e,
        sends: &[(T0, B)],
        receives: &[(T1, B)],
    })
    .unwrap();
    b.open(PhaseSpec {
        epoch: e,
        sends: &[(T1, A)],
        receives: &[(T0, A)],
    })
    .unwrap();
}

fn finish_axis0(a: &mut Fabric<Loopback>, b: &mut Fabric<Loopback>) {
    assert_eq!(a.payload(T1).unwrap(), &[-1.0, -2.0, -3.0]);
    assert_eq!(b.payload(T0).unwrap(), &[1.0, 2.0, 3.0]);
    a.mark_unpacked(T1).unwrap();
    b.mark_unpacked(T0).unwrap();
    a.close().unwrap();
    b.close().unwrap();
}

#[test]
fn two_workers_exchange_a_phase_and_drain() {
    let link = Loopback::new();
    let (mut a, mut b) = pair(&link);
    let e = epoch(0, 1, 0);
    open_axis0(&mut a, &mut b, e);
    drive(&mut [&mut a, &mut b]).unwrap();
    finish_axis0(&mut a, &mut b);
    assert_eq!(link.queued(), 0);
    a.shutdown().unwrap();
    b.shutdown().unwrap();
}

#[test]
fn halo_without_grant_rejected() {
    let link = Loopback::new();
    let (_a, mut b) = pair(&link);
    let (h, p) = halo(epoch(0, 1, 0), T0, &[1.0, 2.0, 3.0]);
    link.inject(A, B, h, &p);
    let err = b.service().unwrap_err();
    assert!(err.is_protocol(), "{err}");
}

#[test]
fn wrong_epoch_rejected() {
    let link = Loopback::new();
    let (mut a, mut b) = pair(&link);
    open_axis0(&mut a, &mut b, epoch(0, 1, 0));
    let (h, p) = halo(epoch(1, 1, 0), T0, &[1.0, 2.0, 3.0]);
    link.inject(A, B, h, &p);
    assert!(b.service().unwrap_err().is_protocol());
}

#[test]
fn wrong_length_rejected() {
    let link = Loopback::new();
    let (mut a, mut b) = pair(&link);
    open_axis0(&mut a, &mut b, epoch(0, 1, 0));
    let (h, p) = halo(epoch(0, 1, 0), T0, &[1.0, 2.0]);
    link.inject(A, B, h, &p);
    assert!(b.service().unwrap_err().is_protocol());
}

#[test]
fn header_and_payload_length_disagreement_rejected() {
    let link = Loopback::new();
    let (mut a, mut b) = pair(&link);
    open_axis0(&mut a, &mut b, epoch(0, 1, 0));
    let (mut h, p) = halo(epoch(0, 1, 0), T0, &[1.0, 2.0, 3.0]);
    h.payload_len = 8;
    link.inject(A, B, h, &p);
    assert!(b.service().unwrap_err().is_protocol());
}

#[test]
fn duplicate_halo_rejected() {
    let link = Loopback::new();
    let (mut a, mut b) = pair(&link);
    let e = epoch(0, 1, 0);
    open_axis0(&mut a, &mut b, e);
    drive(&mut [&mut a, &mut b]).unwrap();
    let (h, p) = halo(e, T0, &[1.0, 2.0, 3.0]);
    link.inject(A, B, h, &p);
    assert!(b.service().unwrap_err().is_protocol());
}

#[test]
fn halo_from_the_wrong_peer_rejected() {
    let link = Loopback::new();
    let (mut a, mut b) = pair(&link);
    let e = epoch(0, 1, 0);
    open_axis0(&mut a, &mut b, e);
    let (h, p) = halo(e, T0, &[1.0, 2.0, 3.0]);
    link.inject(C, B, h, &p);
    assert!(b.service().unwrap_err().is_protocol());
}

#[test]
fn grant_count_mismatch_rejected() {
    let link = Loopback::new();
    let (mut a, _b) = pair(&link);
    link.inject(B, A, grant(epoch(0, 1, 0), 2), &[]);
    assert!(a.service().unwrap_err().is_protocol());
}

#[test]
fn grant_for_a_phase_this_worker_never_sends_in_rejected() {
    let link = Loopback::new();
    let (mut a, _b) = pair(&link);
    link.inject(C, A, grant(epoch(0, 1, 0), 1), &[]);
    assert!(a.service().unwrap_err().is_protocol());
}

#[test]
fn grant_tabled_while_result_pending() {
    let link = Loopback::new();
    let (mut a, mut b) = pair(&link);
    let e = epoch(0, 2, 0);
    // a holds its collective result and opens stage 1; b is still waiting for the same result
    // and only services the link.
    a.pack(T0, |buf| buf.copy_from_slice(&[1.0, 2.0, 3.0]))
        .unwrap();
    a.open(PhaseSpec {
        epoch: e,
        sends: &[(T0, B)],
        receives: &[(T1, B)],
    })
    .unwrap();
    assert_eq!(a.progress().unwrap(), Progress::Pending);
    b.service().unwrap();
    assert!(b.grants().is_tabled(A, e));
    // b's result arrives and it opens the same phase: the tabled grant is consumed.
    b.pack(T1, |buf| buf.copy_from_slice(&[-1.0, -2.0, -3.0]))
        .unwrap();
    b.open(PhaseSpec {
        epoch: e,
        sends: &[(T1, A)],
        receives: &[(T0, A)],
    })
    .unwrap();
    drive(&mut [&mut a, &mut b]).unwrap();
    finish_axis0(&mut a, &mut b);
    assert_eq!(b.grants().tabled(A), 0);
    a.shutdown().unwrap();
    b.shutdown().unwrap();
}

#[test]
fn grant_for_the_other_branch_rejected_at_open() {
    let link = Loopback::new();
    let (mut a, mut b) = pair(&link);
    // a took the accept branch into stage 1
    a.pack(T0, |buf| buf.copy_from_slice(&[1.0, 2.0, 3.0]))
        .unwrap();
    a.open(PhaseSpec {
        epoch: epoch(0, 2, 0),
        sends: &[(T0, B)],
        receives: &[(T1, B)],
    })
    .unwrap();
    b.service().unwrap();
    // b's result said rollback
    b.pack(T1, |buf| buf.copy_from_slice(&[0.0; 3])).unwrap();
    let err = b
        .open(PhaseSpec {
            epoch: epoch(0, 0xF0, 0),
            sends: &[(T1, A)],
            receives: &[(T0, A)],
        })
        .unwrap_err();
    assert!(err.is_protocol(), "{err}");
}

#[test]
fn axis_one_grant_survives_axis_zero() {
    let link = Loopback::new();
    let (mut a, mut b) = pair(&link);
    let e0 = epoch(0, 1, 0);
    let e1 = epoch(0, 1, 1);
    // b tables a's grants for both axes before opening either
    link.inject(A, B, grant(e0, 1), &[]);
    link.inject(A, B, grant(e1, 1), &[]);
    b.service().unwrap();
    assert_eq!(b.grants().tabled(A), 2);
    // a opens axis 0 for real; b opens, sends under the tabled axis-0 grant
    a.pack(T0, |buf| buf.copy_from_slice(&[1.0, 2.0, 3.0]))
        .unwrap();
    a.open(PhaseSpec {
        epoch: e0,
        sends: &[(T0, B)],
        receives: &[(T1, B)],
    })
    .unwrap();
    // the real grant duplicates the injected one
    assert!(b.service().unwrap_err().is_protocol());
}

#[test]
fn axis_one_grant_survives_axis_zero_in_a_valid_sequence() {
    let link = Loopback::new();
    let (mut a, mut b) = pair(&link);
    let e0 = epoch(0, 1, 0);
    let e1 = epoch(0, 1, 1);
    open_axis0(&mut a, &mut b, e0);
    drive(&mut [&mut a, &mut b]).unwrap();
    finish_axis0(&mut a, &mut b);
    // a closes axis 0 and opens axis 1 at once; b has not opened axis 1
    a.pack(T2, |buf| buf.copy_from_slice(&[5.0, 6.0])).unwrap();
    a.open(PhaseSpec {
        epoch: e1,
        sends: &[(T2, B)],
        receives: &[(T3, B)],
    })
    .unwrap();
    b.service().unwrap();
    assert!(b.grants().is_tabled(A, e1));
    b.pack(T3, |buf| buf.copy_from_slice(&[-5.0, -6.0]))
        .unwrap();
    b.open(PhaseSpec {
        epoch: e1,
        sends: &[(T3, A)],
        receives: &[(T2, A)],
    })
    .unwrap();
    drive(&mut [&mut a, &mut b]).unwrap();
    assert_eq!(a.payload(T3).unwrap(), &[-5.0, -6.0]);
    assert_eq!(b.payload(T2).unwrap(), &[5.0, 6.0]);
    a.mark_unpacked(T3).unwrap();
    b.mark_unpacked(T2).unwrap();
    a.close().unwrap();
    b.close().unwrap();
    a.shutdown().unwrap();
    b.shutdown().unwrap();
}

#[test]
fn local_only_axis_issues_no_grant() {
    let link = Loopback::new();
    let (mut a, _b) = pair(&link);
    a.open(PhaseSpec {
        epoch: epoch(0, 1, 1),
        sends: &[],
        receives: &[],
    })
    .unwrap();
    assert_eq!(link.queued_of_kind(Kind::Grant), 0);
    assert_eq!(a.progress().unwrap(), Progress::Done);
    a.close().unwrap();
}

#[test]
fn posted_buffer_is_leased() {
    let link = Loopback::new();
    let (mut a, _b) = pair(&link);
    a.pack(T0, |buf| buf.fill(1.0)).unwrap();
    assert!(a.pack(T0, |buf| buf.fill(2.0)).unwrap_err().is_protocol());
    // an unpacked send cannot open
    assert!(
        a.open(PhaseSpec {
            epoch: epoch(0, 1, 1),
            sends: &[(T2, B)],
            receives: &[],
        })
        .unwrap_err()
        .is_protocol()
    );
}

#[test]
fn an_abort_frame_surfaces_as_peer_aborted() {
    let link = Loopback::new();
    let (mut a, _b) = pair(&link);
    link.inject(
        B,
        A,
        Header {
            kind: Kind::Abort,
            session: S,
            epoch: Epoch {
                step: 0,
                attempt: 0,
                point: Epoch::NO_POINT,
                axis: Epoch::NO_AXIS,
            },
            id: 0,
            payload_len: 3,
        },
        b"nan",
    );
    assert!(matches!(
        a.service().unwrap_err(),
        FabricError::PeerAborted { peer: B, .. }
    ));
}

#[test]
fn a_frame_from_another_session_is_rejected() {
    let link = Loopback::new();
    let (mut a, _b) = pair(&link);
    let mut h = grant(epoch(0, 1, 0), 1);
    h.session = SessionId(1);
    link.inject(B, A, h, &[]);
    assert!(a.service().unwrap_err().is_protocol());
}

/// the session-end barrier over the loopback: both sides say Done and each leaves once it
/// has heard the other.
#[test]
fn finish_exchanges_done_both_ways() {
    let link = Loopback::new();
    let mut a = Fabric::new(
        link.clone(),
        A,
        S,
        2,
        2,
        &LENS,
        vec![vec![0, 0], vec![1, 1]],
    );
    let mut b = Fabric::new(
        link.clone(),
        B,
        S,
        2,
        2,
        &LENS,
        vec![vec![1, 1], vec![0, 0]],
    );
    open_axis0(&mut a, &mut b, epoch(0, 1, 0));
    drive(&mut [&mut a, &mut b]).unwrap();
    finish_axis0(&mut a, &mut b);
    a.begin_finish().unwrap();
    assert_eq!(a.finish_progress().unwrap(), Progress::Pending);
    b.begin_finish().unwrap();
    assert_eq!(b.finish_progress().unwrap(), Progress::Done);
    assert_eq!(a.finish_progress().unwrap(), Progress::Done);
    assert!(a.begin_finish().unwrap_err().is_protocol());
}

/// a Done from a peer this worker still expects a halo from is a protocol error.
#[test]
fn done_from_a_peer_still_owed_a_halo_rejected() {
    let link = Loopback::new();
    let (mut a, mut b) = pair(&link);
    open_axis0(&mut a, &mut b, epoch(0, 1, 0));
    link.inject(
        A,
        B,
        Header {
            kind: Kind::Done,
            session: S,
            epoch: Epoch {
                step: 0,
                attempt: 0,
                point: Epoch::NO_POINT,
                axis: Epoch::NO_AXIS,
            },
            id: 0,
            payload_len: 0,
        },
        &[],
    );
    assert!(b.service().unwrap_err().is_protocol());
}

/// a Done from a peer with nothing owed in the open phase is tabled; opening a later phase
/// that exchanges with that peer is refused.
#[test]
fn done_from_an_uninvolved_peer_is_tabled_and_bars_later_phases() {
    let link = Loopback::new();
    let (mut a, _b) = pair(&link);
    a.open(PhaseSpec {
        epoch: epoch(0, 1, 1),
        sends: &[],
        receives: &[],
    })
    .unwrap();
    link.inject(
        C,
        A,
        Header {
            kind: Kind::Done,
            session: S,
            epoch: Epoch {
                step: 0,
                attempt: 0,
                point: Epoch::NO_POINT,
                axis: Epoch::NO_AXIS,
            },
            id: 0,
            payload_len: 0,
        },
        &[],
    );
    a.service().unwrap();
    a.close().unwrap();
    // c has finished; a phase that would receive from c is refused
    assert!(
        a.open(PhaseSpec {
            epoch: epoch(0, 2, 0),
            sends: &[],
            receives: &[(T1, C)],
        })
        .unwrap_err()
        .is_protocol()
    );
}

/// a link that holds every frame until released, so this worker's own Done stays queued
/// while every peer's Done has already arrived.
#[derive(Clone)]
struct Held {
    inner: Loopback,
    open: std::rc::Rc<std::cell::Cell<bool>>,
    queue: std::rc::Rc<std::cell::RefCell<Vec<(WorkerId, WorkerId, Header, Vec<u8>)>>>,
}

impl symbi_fabric::Link for Held {
    fn send(
        &self,
        from: WorkerId,
        to: WorkerId,
        header: Header,
        payload: &[u8],
    ) -> Result<(), FabricError> {
        self.queue
            .borrow_mut()
            .push((from, to, header, payload.to_vec()));
        Ok(())
    }
    fn poll(
        &self,
        me: WorkerId,
        payload: &mut Vec<u8>,
    ) -> Result<Option<(WorkerId, Header)>, FabricError> {
        self.inner.poll(me, payload)
    }
    fn drive(&self, _me: WorkerId) -> Result<(), FabricError> {
        if self.open.get() {
            for (from, to, header, payload) in self.queue.borrow_mut().drain(..) {
                self.inner.inject(from, to, header, &payload);
            }
        }
        Ok(())
    }
    fn outbound_pending(&self) -> bool {
        !self.queue.borrow().is_empty()
    }
}

/// every peer's Done is in hand but this worker's own Done has not left the link: finish
/// stays pending until the outbound queue drains.
#[test]
fn finish_waits_for_outbound_drain() {
    let link = Loopback::new();
    let held = Held {
        inner: link.clone(),
        open: std::rc::Rc::new(std::cell::Cell::new(false)),
        queue: std::rc::Rc::new(std::cell::RefCell::new(Vec::new())),
    };
    let mut a = Fabric::new(
        held.clone(),
        A,
        S,
        2,
        2,
        &LENS,
        vec![vec![0, 0], vec![1, 1]],
    );
    link.inject(
        B,
        A,
        Header {
            kind: Kind::Done,
            session: S,
            epoch: Epoch {
                step: 0,
                attempt: 0,
                point: Epoch::NO_POINT,
                axis: Epoch::NO_AXIS,
            },
            id: 0,
            payload_len: 0,
        },
        &[],
    );
    a.begin_finish().unwrap();
    assert_eq!(
        a.finish_progress().unwrap(),
        Progress::Pending,
        "own Done still queued"
    );
    assert_eq!(a.finish_progress().unwrap(), Progress::Pending);
    held.open.set(true);
    assert_eq!(a.finish_progress().unwrap(), Progress::Done);
    assert_eq!(
        link.queued_of_kind(Kind::Done),
        1,
        "the Done reached the link"
    );
}

fn pair2(link: &Loopback) -> (Fabric<Loopback>, Fabric<Loopback>) {
    let a = Fabric::new(
        link.clone(),
        A,
        S,
        2,
        2,
        &LENS,
        vec![vec![0, 0], vec![1, 1]],
    );
    let b = Fabric::new(
        link.clone(),
        B,
        S,
        2,
        2,
        &LENS,
        vec![vec![1, 1], vec![0, 0]],
    );
    (a, b)
}

/// drive a collective on both endpoints until both hold the result.
fn run_collective(
    a: &mut Fabric<Loopback>,
    b: &mut Fabric<Loopback>,
    kind: OpKind,
    va: u64,
    vb: u64,
) -> (u64, u64) {
    a.begin_collective(kind, va).unwrap();
    b.begin_collective(kind, vb).unwrap();
    let (mut ra, mut rb) = (None, None);
    for _ in 0..16 {
        if ra.is_none() {
            ra = a.collective_progress().unwrap();
        }
        if rb.is_none() {
            rb = b.collective_progress().unwrap();
        }
        if ra.is_some() && rb.is_some() {
            return (ra.unwrap(), rb.unwrap());
        }
    }
    panic!("the collective made no progress");
}

#[test]
fn min_broadcasts_the_smaller_candidate_as_bits() {
    let link = Loopback::new();
    let (mut a, mut b) = pair2(&link);
    let (ra, rb) = run_collective(
        &mut a,
        &mut b,
        OpKind::Min,
        0.25f64.to_bits(),
        0.125f64.to_bits(),
    );
    assert_eq!(f64::from_bits(ra), 0.125);
    assert_eq!(ra, rb);
    let (ra, rb) = run_collective(&mut a, &mut b, OpKind::Any, 0, 1);
    assert_eq!((ra, rb), (1, 1));
    let (ra, rb) = run_collective(&mut a, &mut b, OpKind::Any, 0, 0);
    assert_eq!((ra, rb), (0, 0));
    let (ra, _) = run_collective(&mut a, &mut b, OpKind::CheckpointOpen, 1, 0);
    assert_eq!(ra, 0);
}

#[test]
fn an_invalid_min_candidate_is_refused_locally() {
    let link = Loopback::new();
    let (mut a, _b) = pair2(&link);
    assert!(
        a.begin_collective(OpKind::Min, f64::NAN.to_bits())
            .unwrap_err()
            .is_protocol()
    );
    assert!(
        a.begin_collective(OpKind::Min, 0.0f64.to_bits())
            .unwrap_err()
            .is_protocol()
    );
}

#[test]
fn a_result_for_an_unexpected_op_is_rejected() {
    let link = Loopback::new();
    let (_a, mut b) = pair2(&link);
    let mut payload = [0u8; 9];
    payload[0] = OpKind::Any as u8;
    link.inject(
        A,
        B,
        Header {
            kind: Kind::Result,
            session: S,
            epoch: Epoch {
                step: 0,
                attempt: 0,
                point: Epoch::NO_POINT,
                axis: Epoch::NO_AXIS,
            },
            id: 0,
            payload_len: 9,
        },
        &payload,
    );
    assert!(b.service().unwrap_err().is_protocol());
}

#[test]
fn a_contribution_two_ops_ahead_is_rejected() {
    let link = Loopback::new();
    let (mut a, _b) = pair2(&link);
    let mut payload = [0u8; 9];
    payload[0] = OpKind::Any as u8;
    let contribute = |op| Header {
        kind: Kind::Contribute,
        session: S,
        epoch: Epoch {
            step: 0,
            attempt: 0,
            point: Epoch::NO_POINT,
            axis: Epoch::NO_AXIS,
        },
        id: op,
        payload_len: 9,
    };
    // one ahead is tabled
    link.inject(B, A, contribute(1), &payload);
    a.service().unwrap();
    // two ahead is refused
    link.inject(B, A, contribute(2), &payload);
    assert!(a.service().unwrap_err().is_protocol());
}

/// a grant arriving while this worker's collective is pending is tabled, and consumed once
/// the result lets the worker open the phase.
#[test]
fn a_grant_during_a_pending_collective_is_tabled() {
    let link = Loopback::new();
    let (mut a, mut b) = pair2(&link);
    b.begin_collective(OpKind::Any, 0).unwrap();
    assert_eq!(b.collective_progress().unwrap(), None);
    let e = epoch(0, 1, 0);
    link.inject(A, B, grant(e, 1), &[]);
    assert_eq!(b.collective_progress().unwrap(), None);
    assert!(b.grants().is_tabled(A, e));
    a.begin_collective(OpKind::Any, 0).unwrap();
    assert_eq!(a.collective_progress().unwrap(), Some(0));
    assert_eq!(b.collective_progress().unwrap(), Some(0));
    b.pack(T1, |buf| buf.fill(1.0)).unwrap();
    b.open(PhaseSpec {
        epoch: e,
        sends: &[(T1, A)],
        receives: &[(T0, A)],
    })
    .unwrap();
    assert_eq!(b.progress().unwrap(), Progress::Pending);
    assert!(
        !b.grants().is_tabled(A, e),
        "the grant was consumed at open"
    );
}

#[test]
fn finish_refuses_with_a_collective_pending() {
    let link = Loopback::new();
    let (mut a, _b) = pair2(&link);
    a.begin_collective(OpKind::Any, 0).unwrap();
    assert!(a.begin_finish().unwrap_err().is_protocol());
}
