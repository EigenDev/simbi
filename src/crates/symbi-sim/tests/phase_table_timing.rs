// =============================================================================
// phase_table_timing.rs
//
// the phase table under adversarial timing, on three in-process endpoints over
// the loopback link walking a refined table (two root stages, two fine
// substeps of two stages): a worker without fine tiles runs ahead through its
// empty fine phases while the others exchange; a participant that services
// the link late; a clean stage whose troubled-flag traffic every worker skips;
// a rollback taken on one attempt and declined on the next; and the refusals:
// a peer whose grants pass a phase it owes, a grant beyond the table, a prime
// after the first attempt, and a worker passing a phase at its own open. every
// accepted sequence ends with every grant table drained, which is the bound
// the grant storage holds for this table.
//
// run: cargo test -p symbi-sim --test phase_table_timing
// =============================================================================
use std::collections::VecDeque;

use symbi_fabric::{
    Epoch, Fabric, FabricError, Header, Kind, Loopback, PhaseSpec, Progress, SessionId,
    TransferId, WorkerId,
};
use symbi_sim::atlas::{AuditAt, ExchangePoint, FinePoint};
use symbi_sim::phase_table::PhaseTable;

const S: SessionId = SessionId(21);
const A: WorkerId = WorkerId(0);
const B: WorkerId = WorkerId(1);
const C: WorkerId = WorkerId(2);
const WORKERS: usize = 3;
const AXES: usize = 1;
// every transfer lies on axis 0: 0 A->B and 1 B->A at the root state points, 2 B->C and
// 3 C->B at the root state points, 4 A->B and 5 B->A at the fine points, 6 A->B and 7 B->A
// at the troubled points (the flag band). c holds no fine tile and no band.
const LENS: [usize; 8] = [2; 8];

fn t(id: u32) -> TransferId {
    TransferId(id)
}

/// the transfers that move at `point`, as (id, from, to)
fn moving(point: ExchangePoint) -> Vec<(TransferId, WorkerId, WorkerId)> {
    match point {
        ExchangePoint::Prime
        | ExchangePoint::EquilibriumPrime
        | ExchangePoint::Stage(_)
        | ExchangePoint::PostStep
        | ExchangePoint::ExciseSweep(_)
        | ExchangePoint::ExciseRebuild(_)
        | ExchangePoint::PostBody
        | ExchangePoint::Rollback
        | ExchangePoint::Restrict => vec![(t(0), A, B), (t(1), B, A), (t(2), B, C), (t(3), C, B)],
        // the primed root and the primed fine level, before the prime audit
        ExchangePoint::Primed => vec![(t(0), A, B), (t(1), B, A), (t(2), B, C), (t(3), C, B), (t(4), A, B), (t(5), B, A)],
        ExchangePoint::Troubled(_) | ExchangePoint::EquilibriumTroubled => vec![(t(6), A, B), (t(7), B, A)],
        ExchangePoint::Fine { point: FinePoint::Prolong | FinePoint::Stage(_) | FinePoint::PostTail, .. } => {
            vec![(t(4), A, B), (t(5), B, A)]
        }
        ExchangePoint::Audit(_)
        | ExchangePoint::Fine { .. }
        | ExchangePoint::CorrectedFaces
        | ExchangePoint::Nested { .. }
        | ExchangePoint::NestedRestrict { .. }
        | ExchangePoint::Seed { .. }
        | ExchangePoint::TracerCoefficients
        | ExchangePoint::EquilibriumLevel { .. }
        | ExchangePoint::RootCoefficients => vec![],
    }
}

/// `[byte][peer][axis]`: what `me` sends each peer at every phase of the table
fn counts_by_byte(table: &PhaseTable, me: WorkerId) -> Vec<Vec<Vec<u32>>> {
    table
        .phases()
        .iter()
        .map(|phase| {
            let mut per = vec![vec![0u32; AXES]; WORKERS];
            for (_, from, to) in moving(phase.point) {
                if from == me {
                    per[to.0 as usize][0] += 1;
                }
            }
            per
        })
        .collect()
}

fn table() -> PhaseTable {
    PhaseTable::refined(2, 2, 2).expect("the table fits the byte")
}

fn endpoint(link: &Loopback, me: WorkerId, table: &PhaseTable) -> Fabric<Loopback> {
    let counts = counts_by_byte(table, me);
    let default = counts[table.byte_of(ExchangePoint::Prime).unwrap() as usize].clone();
    Fabric::new(link.clone(), me, S, WORKERS, AXES, &LENS, default)
        .with_phase_counts(counts)
        .with_phase_rules(table.rules())
}

/// the phases of one attempt in table order, with the branches this script takes: the
/// troubled exchanges of the stages in `troubled`, the rollback when `reject` names a root
/// stage (the attempt ends there), the fine phases and the restriction otherwise
fn attempt(table: &PhaseTable, step: u64, attempt: u16, troubled: &[u8], reject: Option<u8>) -> Vec<Epoch> {
    let mut out = Vec::new();
    let mut push = |point: ExchangePoint| {
        out.push(Epoch { step, attempt, point: table.byte_of(point).unwrap(), axis: 0 });
    };
    if (step, attempt) == (0, 0) {
        push(ExchangePoint::Prime);
    }
    for k in 0..table.stages() {
        if troubled.contains(&k) {
            push(ExchangePoint::Troubled(k));
        }
        if reject == Some(k) {
            push(ExchangePoint::Rollback);
            return out;
        }
        push(ExchangePoint::Stage(k));
    }
    for s in 0..table.substeps() {
        push(ExchangePoint::Fine { substep: s, point: FinePoint::Prolong });
        for k in 0..table.fine_stages() {
            push(ExchangePoint::Fine { substep: s, point: FinePoint::Stage(k) });
        }
        push(ExchangePoint::Fine { substep: s, point: FinePoint::PostTail });
    }
    push(ExchangePoint::Restrict);
    push(ExchangePoint::PostStep);
    out
}

/// two accepted steps with no troubled cell and no rejection
fn two_clean_steps(table: &PhaseTable) -> Vec<Epoch> {
    let mut v = attempt(table, 0, 0, &[], None);
    v.extend(attempt(table, 1, 0, &[], None));
    v
}

/// one endpoint walking its script: a phase opens with the transfers it moves and closes
/// once the fabric reports it done
struct Walker {
    me: WorkerId,
    fabric: Fabric<Loopback>,
    table: PhaseTable,
    queue: VecDeque<Epoch>,
    open: Option<Vec<TransferId>>,
    done: bool,
}

impl Walker {
    fn new(link: &Loopback, me: WorkerId, table: &PhaseTable, script: Vec<Epoch>) -> Self {
        Self {
            me,
            fabric: endpoint(link, me, table),
            table: table.clone(),
            queue: script.into(),
            open: None,
            done: false,
        }
    }

    /// one unit of work: progress the open phase and close it when done, else open the next
    /// phase; false once the script is exhausted
    fn advance(&mut self) -> Result<bool, FabricError> {
        if let Some(receives) = self.open.as_ref() {
            if self.fabric.progress()? == Progress::Done {
                for id in receives.clone() {
                    let _ = self.fabric.payload(id)?;
                    self.fabric.mark_unpacked(id)?;
                }
                self.fabric.close()?;
                self.open = None;
            } else {
                return Ok(true);
            }
        }
        let Some(epoch) = self.queue.pop_front() else {
            if !self.done {
                self.fabric.service()?;
            }
            return Ok(false);
        };
        let point = self.table.point_of(epoch.point).expect("a scripted point");
        let mut sends = Vec::new();
        let mut receives = Vec::new();
        for (id, from, to) in moving(point) {
            if from == self.me {
                let value = f64::from(id.0) + f64::from(epoch.point) * 0.01;
                self.fabric.pack(id, |buf| buf.fill(value))?;
                sends.push((id, to));
            } else if to == self.me {
                receives.push((id, from));
            }
        }
        self.fabric.open(PhaseSpec { epoch, sends: &sends, receives: &receives })?;
        self.open = Some(receives.iter().map(|r| r.0).collect());
        Ok(true)
    }

    fn finished(&self) -> bool {
        self.queue.is_empty() && self.open.is_none()
    }
}

/// drive the walkers in the order `schedule(round)` returns until every script is walked;
/// a round with no progress on a stalled set of walkers is a failure after the bound
fn run(walkers: &mut [Walker], mut schedule: impl FnMut(usize) -> Vec<usize>) -> Result<(), FabricError> {
    let mut idle_rounds = 0;
    for round in 0..4000 {
        let mut busy = false;
        for w in schedule(round) {
            busy |= walkers[w].advance()?;
        }
        if walkers.iter().all(Walker::finished) {
            return Ok(());
        }
        idle_rounds = if busy { 0 } else { idle_rounds + 1 };
        if idle_rounds > 64 {
            break;
        }
    }
    panic!(
        "the walkers stalled: {:?}",
        walkers.iter().map(|w| (w.me, w.queue.len(), w.open.is_some())).collect::<Vec<_>>()
    );
}

fn drained(walkers: &mut [Walker]) {
    for w in walkers.iter_mut() {
        w.fabric.grants().assert_drained().unwrap_or_else(|e| panic!("{:?}: {e}", w.me));
        w.done = true;
    }
}

fn three(link: &Loopback, table: &PhaseTable, scripts: [Vec<Epoch>; 3]) -> Vec<Walker> {
    let [a, b, c] = scripts;
    vec![Walker::new(link, A, table, a), Walker::new(link, B, table, b), Walker::new(link, C, table, c)]
}

/// c holds no fine tile: it opens every fine phase empty and runs ahead to the restriction,
/// whose grant reaches b while b is still inside the fine subcycle; b tables it, the fine
/// phases close, and every table drains
#[test]
fn a_worker_without_fine_tiles_runs_ahead_through_its_empty_phases() {
    let link = Loopback::new();
    let table = table();
    let script = two_clean_steps(&table);
    let mut walkers = three(&link, &table, [script.clone(), script.clone(), script]);
    let restrict = table.byte_of(ExchangePoint::Restrict).unwrap();
    let first_fine = table.byte_of(ExchangePoint::Fine { substep: 0, point: FinePoint::Prolong }).unwrap();
    // the premise: at some round b, inside the fine subcycle, already holds c's grant for
    // the restriction
    let mut c_ran_ahead = false;
    let mut rounds = 0;
    while !walkers.iter().all(Walker::finished) {
        for w in [2usize, 2, 2, 2, 1, 0] {
            walkers[w].advance().unwrap_or_else(|e| panic!("{e}"));
        }
        let b_open = walkers[1].fabric.phase_epoch();
        let held = walkers[1]
            .fabric
            .grants()
            .is_tabled(C, Epoch { step: 0, attempt: 0, point: restrict, axis: 0 });
        if held && b_open.is_some_and(|e| e.step == 0 && e.point >= first_fine && e.point < restrict) {
            c_ran_ahead = true;
        }
        rounds += 1;
        assert!(rounds < 4000, "the walk stalled");
    }
    assert!(c_ran_ahead, "c never held the restriction ahead of b's fine subcycle; the gate is vacuous");
    drained(&mut walkers);
}

/// a participant that services the link one round in four: its peers' grants wait in its
/// table and its own arrive late, within the storage bound
#[test]
fn a_delayed_participant_keeps_the_sequence_within_the_grant_bound() {
    let link = Loopback::new();
    let table = table();
    let script = two_clean_steps(&table);
    let mut walkers = three(&link, &table, [script.clone(), script.clone(), script]);
    run(&mut walkers, |round| if round % 4 == 0 { vec![0, 1, 2] } else { vec![1, 2] }).unwrap();
    drained(&mut walkers);
}

/// the troubled-flag traffic of stage 0 runs on the first step and every worker skips it
/// on the second: the skip is a property of the collective, and the sequence closes
#[test]
fn a_clean_stage_skips_the_troubled_traffic_on_every_worker() {
    let link = Loopback::new();
    let table = table();
    let mut script = attempt(&table, 0, 0, &[0], None);
    script.extend(attempt(&table, 1, 0, &[], None));
    script.extend(attempt(&table, 2, 0, &[1], None));
    let mut walkers = three(&link, &table, [script.clone(), script.clone(), script]);
    run(&mut walkers, |_| vec![0, 1, 2]).unwrap();
    drained(&mut walkers);
}

/// a rejection at root stage 1 of the first attempt branches every worker to the rollback,
/// the second attempt runs the whole table, and the next step follows
#[test]
fn a_rollback_taken_then_declined_walks_the_table_to_the_end() {
    let link = Loopback::new();
    let table = table();
    let mut script = attempt(&table, 0, 0, &[], Some(1));
    script.extend(attempt(&table, 0, 1, &[], None));
    script.extend(attempt(&table, 1, 0, &[], None));
    let mut walkers = three(&link, &table, [script.clone(), script.clone(), script]);
    run(&mut walkers, |round| if round % 3 == 0 { vec![2, 1, 0] } else { vec![0, 1, 2] }).unwrap();
    drained(&mut walkers);
}

/// a grant from b that passes root stage 1, which b owes a, is refused by a on receipt; the
/// premise: an endpoint without the sequence rules tables the same grant
#[test]
fn a_peer_that_passes_an_owed_phase_is_refused_on_receipt() {
    let link = Loopback::new();
    let table = table();
    let grant = |epoch: Epoch| Header { kind: Kind::Grant, session: S, epoch, id: 1, payload_len: 0 };
    let stage0 = table.byte_of(ExchangePoint::Stage(0)).unwrap();
    let prolong = table.byte_of(ExchangePoint::Fine { substep: 0, point: FinePoint::Prolong }).unwrap();
    // b's grants for the prime and stage 0 arrive and are consumed as a sends to b in those
    // phases; then b's grant for the first fine prolongation arrives, passing stage 1
    let walk = |a: &mut Fabric<Loopback>| -> Result<(), FabricError> {
        for byte in [table.byte_of(ExchangePoint::Prime).unwrap(), stage0] {
            let epoch = Epoch { step: 0, attempt: 0, point: byte, axis: 0 };
            link.inject(B, A, grant(epoch), &[]);
            a.pack(t(0), |buf| buf.fill(1.0))?;
            a.open(PhaseSpec { epoch, sends: &[(t(0), B)], receives: &[] })?;
            while a.progress()? == Progress::Pending {}
            a.close()?;
        }
        link.inject(B, A, grant(Epoch { step: 0, attempt: 0, point: prolong, axis: 0 }), &[]);
        a.service()
    };
    let mut ruled = endpoint(&link, A, &table);
    let err = walk(&mut ruled).unwrap_err().to_string();
    assert!(err.contains("grant for") && err.contains("passes the phase"), "another refusal: {err}");
    let counts = counts_by_byte(&table, A);
    let default = counts[table.byte_of(ExchangePoint::Prime).unwrap() as usize].clone();
    let mut unruled = Fabric::new(link.clone(), A, S, WORKERS, AXES, &LENS, default).with_phase_counts(counts);
    walk(&mut unruled).expect("without the sequence rules the grant is tabled; the premise is vacuous");
}

/// a worker's own opens are judged strictly on a two-axis endpoint: an axis repeated,
/// reversed or omitted, a phase left before its last axis or opened past its first, a
/// phase moved backward, step zero skipped from the prime, a checkpoint audit inside the
/// subcycle, a rollback audit without a rollback
#[test]
fn a_worker_is_refused_at_its_own_open_for_every_illegal_transition() {
    let table = table();
    let byte = |p| table.byte_of(p).unwrap();
    let e = |attempt: u16, point: u8, axis: u8| Epoch { step: 0, attempt, point, axis };
    let prolong = byte(ExchangePoint::Fine { substep: 0, point: FinePoint::Prolong });
    let stage = |k, axis| e(0, byte(ExchangePoint::Stage(k)), axis);
    let cases: [(&str, Vec<Epoch>, &str); 8] = [
        ("a repeated axis", vec![e(0, byte(ExchangePoint::Prime), 0), e(0, byte(ExchangePoint::Prime), 0)], "skips an axis"),
        ("a reversed axis", vec![stage(0, 0), stage(0, 1), stage(0, 0)], "skips an axis"),
        ("a backward phase", vec![stage(0, 0), stage(0, 1), stage(1, 0), stage(1, 1), stage(0, 0)], "moves backward"),
        ("a phase left before its last axis", vec![stage(0, 0), stage(1, 0)], "leaves"),
        ("a phase opened past its first axis", vec![stage(0, 0), stage(0, 1), stage(1, 1)], "past its first axis"),
        ("step zero skipped from the prime", vec![e(0, byte(ExchangePoint::Prime), 0), e(0, byte(ExchangePoint::Prime), 1), Epoch { step: 1, attempt: 0, point: byte(ExchangePoint::Stage(0)), axis: 0 }], "leaves the primed state"),
        ("a checkpoint audit inside the subcycle", vec![stage(0, 0), stage(0, 1), stage(1, 0), stage(1, 1), e(0, prolong, 0), e(0, prolong, 1), e(symbi_fabric::CHECKPOINT_ATTEMPT, byte(ExchangePoint::Audit(AuditAt::Checkpoint)), 0)], "passes the phase"),
        ("a rollback audit without a rollback", vec![stage(0, 0), stage(0, 1), e(0, byte(ExchangePoint::Audit(AuditAt::Rollback)), 0)], "audits a rollback"),
    ];
    for (label, script, marker) in cases {
        let link = Loopback::new();
        // two axes, so an axis can reverse
        let counts: Vec<Vec<Vec<u32>>> = counts_by_byte(&table, A)
            .into_iter()
            .map(|per_peer| per_peer.into_iter().map(|per_axis| vec![per_axis[0], 0]).collect())
            .collect();
        let default = counts[table.byte_of(ExchangePoint::Prime).unwrap() as usize].clone();
        let mut a = Fabric::new(link.clone(), A, S, WORKERS, 2, &LENS, default)
            .with_phase_counts(counts)
            .with_phase_rules(table.rules());
        let mut result = Ok(());
        for epoch in script {
            if a.is_open() {
                a.close().unwrap();
            }
            result = a.open(PhaseSpec { epoch, sends: &[], receives: &[] });
            if result.is_err() {
                break;
            }
        }
        let err = result.expect_err(label).to_string();
        assert!(err.contains("open of") && err.contains(marker), "{label}: another refusal: {err}");
    }
}

/// a grant for a byte the table lacks and a prime after the first attempt are refused
#[test]
fn a_grant_beyond_the_table_and_a_late_prime_are_refused() {
    let link = Loopback::new();
    let table = table();
    let mut a = endpoint(&link, A, &table);
    let grant = |epoch: Epoch| Header { kind: Kind::Grant, session: S, epoch, id: 1, payload_len: 0 };
    link.inject(B, A, grant(Epoch { step: 0, attempt: 0, point: table.len() as u8, axis: 0 }), &[]);
    let err = a.service().unwrap_err().to_string();
    assert!(err.contains("outside the table"), "another refusal: {err}");
    let mut a = endpoint(&link, A, &table);
    let prime = table.byte_of(ExchangePoint::Prime).unwrap();
    link.inject(B, A, grant(Epoch { step: 1, attempt: 0, point: prime, axis: 0 }), &[]);
    let err = a.service().unwrap_err().to_string();
    assert!(err.contains("first attempt alone"), "another refusal: {err}");
}

/// a worker whose own script passes a phase every worker opens is refused at its open
#[test]
fn a_worker_that_passes_a_phase_at_its_own_open_is_refused() {
    let link = Loopback::new();
    let table = table();
    let full = two_clean_steps(&table);
    let stage0 = table.byte_of(ExchangePoint::Stage(0)).unwrap();
    let short: Vec<Epoch> = full.iter().copied().filter(|e| !(e.step == 0 && e.point == stage0)).collect();
    let mut walkers = three(&link, &table, [short, full.clone(), full]);
    let err = run(&mut walkers, |_| vec![0, 1, 2]).unwrap_err();
    let text = err.to_string();
    assert!(text.contains("open of") && text.contains("passes the phase"), "another refusal: {text}");
    let _ = AuditAt::Prime;
}
