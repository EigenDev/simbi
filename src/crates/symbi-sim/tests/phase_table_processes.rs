// =============================================================================
// phase_table_processes.rs
//
// the phase table across three real processes over TCP on the loopback
// interface, walking a refined table (two root stages, two fine substeps of
// two stages) with the collectives that decide its branches: two axes with
// different participants (a and b exchange on axis 0, b and c on axis 1), a
// worker without fine tiles (c), a troubled stage decided by `Any(troubled)`
// with the coordinator holding c's result back, a rejection in the second fine
// substep decided by `Any(retry)` and rolled back on every worker, one-way
// audit traffic after every root stage and before every checkpoint, and at the
// end every grant table drained with its high-water mark reported. the parent
// spawns this binary three times as workers, reads exit codes, stderr and the
// report lines, and is the outer watchdog.
//
// run: cargo test -p symbi-sim --test phase_table_processes
// =============================================================================
use std::io::Read;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use symbi_fabric::rendezvous::{Identity, Rendezvous, connect};
use symbi_fabric::{
    CHECKPOINT_ATTEMPT, Digest, Epoch, Fabric, FabricError, OpKind, PhaseSpec, TcpLink,
    TransferId, WorkerId,
};
use symbi_sim::atlas::{AuditAt, ExchangePoint, FinePoint};
use symbi_sim::phase_table::PhaseTable;

const ROLE: &str = "SYMBI_PHASE_TABLE_TEST_ROLE";
const DEADLINE: Duration = Duration::from_secs(6);
const STARTUP: Duration = Duration::from_secs(8);
const WATCHDOG: Duration = Duration::from_secs(40);
const A: WorkerId = WorkerId(0);
const B: WorkerId = WorkerId(1);
const C: WorkerId = WorkerId(2);
const WORKERS: usize = 3;
const AXES: usize = 2;
const STEPS: u64 = 3;
// transfers: 0 a->b and 1 b->a on axis 0 at the root state points; 2 b->c and 3 c->b on
// axis 1 at the root state points; 4 a->b and 5 b->a on axis 0 at the fine points; 6 a->b
// and 7 b->a on axis 0 at the troubled points; 8 a->b on axis 0 at the audits, one way
const LENS: [usize; 9] = [4; 9];

fn t(id: u32) -> TransferId {
    TransferId(id)
}

fn moving(point: ExchangePoint) -> Vec<(TransferId, WorkerId, WorkerId, u8)> {
    match point {
        ExchangePoint::Prime
        | ExchangePoint::EquilibriumPrime
        | ExchangePoint::Stage(_)
        | ExchangePoint::PostStep
        | ExchangePoint::ExciseSweep(_)
        | ExchangePoint::ExciseRebuild(_)
        | ExchangePoint::PostBody
        | ExchangePoint::Rollback
        | ExchangePoint::Restrict => vec![(t(0), A, B, 0), (t(1), B, A, 0), (t(2), B, C, 1), (t(3), C, B, 1)],
        // the primed root and the primed fine level, before the prime audit
        ExchangePoint::Primed => vec![(t(0), A, B, 0), (t(1), B, A, 0), (t(2), B, C, 1), (t(3), C, B, 1), (t(4), A, B, 0), (t(5), B, A, 0)],
        ExchangePoint::Troubled(_) | ExchangePoint::EquilibriumTroubled => vec![(t(6), A, B, 0), (t(7), B, A, 0)],
        ExchangePoint::Fine { point: FinePoint::Prolong | FinePoint::Stage(_) | FinePoint::PostTail, .. } => {
            vec![(t(4), A, B, 0), (t(5), B, A, 0)]
        }
        ExchangePoint::Audit(AuditAt::Stage(_)) | ExchangePoint::Audit(AuditAt::Checkpoint) => {
            vec![(t(8), A, B, 0)]
        }
        ExchangePoint::Audit(_)
        | ExchangePoint::Fine { .. }
        | ExchangePoint::CorrectedFaces
        | ExchangePoint::Nested { .. }
        | ExchangePoint::NestedRestrict { .. }
        | ExchangePoint::Seed { .. }
        | ExchangePoint::TracerCoefficients
        | ExchangePoint::EquilibriumLevel { .. }
        | ExchangePoint::RootCoefficients
        | ExchangePoint::BandSeam { .. }
        | ExchangePoint::SlipDrain { .. }
        | ExchangePoint::SlipSolve { .. }
        | ExchangePoint::SlipCommit { .. }
        | ExchangePoint::SlipRebuild { .. }
        | ExchangePoint::SlipSync { .. } => vec![],
    }
}

fn counts_by_byte(table: &PhaseTable, me: WorkerId) -> Vec<Vec<Vec<u32>>> {
    table
        .phases()
        .iter()
        .map(|phase| {
            let mut per = vec![vec![0u32; AXES]; WORKERS];
            for (_, from, to, axis) in moving(phase.point) {
                if from == me {
                    per[to.0 as usize][axis as usize] += 1;
                }
            }
            per
        })
        .collect()
}

fn table() -> PhaseTable {
    PhaseTable::refined(2, 2, 2).expect("the table fits the byte")
}

fn value(from: WorkerId, id: TransferId, epoch: Epoch) -> f64 {
    1.0 + f64::from(from.0) * 0.5
        + f64::from(id.0) * 0.01
        + epoch.step as f64 * 1e-3
        + f64::from(epoch.attempt.min(9)) * 1e-4
        + f64::from(epoch.point) * 1e-6
}

/// the march of one worker: every phase of the table in order, the branches from the
/// collectives, the receives checked value for value
struct March {
    me: WorkerId,
    table: PhaseTable,
    fabric: Fabric<TcpLink>,
    retries: u64,
    troubled_phases: u64,
    audits: u64,
    /// the fine stage exchanges opened in the attempt that rejects
    fine_stage_opens_in_rejected_attempt: u64,
    /// on c: b's grant for step 1's first stage was tabled while c awaited the held
    /// troubled result
    early_grant_seen: bool,
}

impl March {
    fn open_phase(&mut self, point: ExchangePoint, step: u64, attempt: u16) -> Result<(), FabricError> {
        let byte = self.table.byte_of(point).expect("a point of the table");
        for axis in 0..AXES as u8 {
            let epoch = Epoch { round: 0, op: 0, step, attempt, point: byte, axis };
            let mut sends = Vec::new();
            let mut receives = Vec::new();
            for (id, from, to, ax) in moving(point) {
                if ax != axis {
                    continue;
                }
                if from == self.me {
                    let v = value(from, id, epoch);
                    self.fabric.pack(id, |buf| buf.fill(v))?;
                    sends.push((id, to));
                } else if to == self.me {
                    receives.push((id, from));
                }
            }
            self.fabric.open(PhaseSpec { epoch, sends: &sends, receives: &receives })?;
            self.fabric.wait(DEADLINE)?;
            for (id, from) in receives {
                let expected = value(from, id, epoch);
                let got = self.fabric.payload(id)?;
                if got.iter().any(|&x| x != expected) {
                    return Err(FabricError::protocol(from, format!("payload of {id:?} at {epoch:?} differs")));
                }
                self.fabric.mark_unpacked(id)?;
            }
            self.fabric.close()?;
        }
        Ok(())
    }

    fn any(&mut self, local: bool) -> Result<bool, FabricError> {
        Ok(self.fabric.collective(OpKind::Any, u64::from(local), DEADLINE)? != 0)
    }

    fn walk(&mut self) -> Result<(), FabricError> {
        let stages = self.table.stages();
        let substeps = self.table.substeps();
        let fine_stages = self.table.fine_stages();
        self.open_phase(ExchangePoint::Prime, 0, 0)?;
        self.open_phase(ExchangePoint::Audit(AuditAt::Prime), 0, 0)?;
        self.audits += 1;
        for step in 0..STEPS {
            let mut attempt: u16 = 0;
            'attempt: loop {
                for k in 0..stages {
                    // a troubled cell on a at the first stage of step 1
                    let troubled_here = self.me == A && step == 1 && k == 0 && attempt == 0;
                    if self.any(troubled_here)? {
                        self.troubled_phases += 1;
                        self.open_phase(ExchangePoint::Troubled(k), step, attempt)?;
                    }
                    // the rejection is decided before the stage's halos move. at step 1's
                    // first stage the coordinator holds c's result back, so a and b open the
                    // stage while c still awaits the vote, and b's grant for it (b receives
                    // from c there) reaches c before its own result
                    let held = step == 1 && k == 0 && attempt == 0;
                    if held && self.fabric.is_coordinator() {
                        self.fabric.delay_result_to(C, Duration::from_millis(150));
                    }
                    if self.any(false)? {
                        unreachable!("no root stage rejects");
                    }
                    if held && self.me == C {
                        let stage = self.table.byte_of(ExchangePoint::Stage(0)).unwrap();
                        self.early_grant_seen = self
                            .fabric
                            .grants()
                            .is_tabled(B, Epoch { round: 0, op: 0, step, attempt, point: stage, axis: 1 });
                    }
                    self.open_phase(ExchangePoint::Stage(k), step, attempt)?;
                    self.open_phase(ExchangePoint::Audit(AuditAt::Stage(k)), step, attempt)?;
                    self.audits += 1;
                }
                for s in 0..substeps {
                    self.open_phase(ExchangePoint::Fine { substep: s, point: FinePoint::Prolong }, step, attempt)?;
                    for k in 0..fine_stages {
                        if self.any(false)? {
                            unreachable!("no fine stage is troubled");
                        }
                        // b rejects the last stage of the second substep of step 2's first
                        // attempt; the rejection is decided before the stage's halos move, so
                        // the rejected stage sends no state
                        let reject_here = self.me == B && step == 2 && attempt == 0 && s == 1 && k == 1;
                        if self.any(reject_here)? {
                            self.open_phase(ExchangePoint::Rollback, step, attempt)?;
                            self.open_phase(ExchangePoint::Audit(AuditAt::Rollback), step, attempt)?;
                            self.retries += 1;
                            attempt += 1;
                            continue 'attempt;
                        }
                        if step == 2 && attempt == 0 {
                            self.fine_stage_opens_in_rejected_attempt += 1;
                        }
                        self.open_phase(ExchangePoint::Fine { substep: s, point: FinePoint::Stage(k) }, step, attempt)?;
                    }
                    self.open_phase(ExchangePoint::Fine { substep: s, point: FinePoint::PostTail }, step, attempt)?;
                }
                self.open_phase(ExchangePoint::Restrict, step, attempt)?;
                break;
            }
            self.open_phase(ExchangePoint::Audit(AuditAt::Checkpoint), step, CHECKPOINT_ATTEMPT)?;
            self.audits += 1;
        }
        Ok(())
    }
}

fn worker_main() -> ! {
    let me = WorkerId(std::env::var("W_ME").expect("W_ME").parse().expect("worker id"));
    let table = table();
    let r = Rendezvous {
        me,
        coordinator: "127.0.0.1:0".parse().expect("socket address"),
        announce: Some(std::path::PathBuf::from(std::env::var("W_COORD").expect("rendezvous file"))),
        bind: None,
        advertise: None,
        identity: Identity {
            credential: 0xC0DE,
            build_id: "phase_table_processes".into(),
            config_digest: Digest(1),
            plan_digest: Digest(2),
            placement_digest: Digest(3),
            phase_table_digest: table.digest(),
            ng: 2,
            workers: WORKERS as u32,
            block_credit: 4096,
        },
        startup: STARTUP,
    };
    let run = || -> Result<String, FabricError> {
        let (link, session) = connect(&r, LENS.iter().max().copied().unwrap() * 8)?;
        let counts = counts_by_byte(&table, me);
        let default = counts[table.byte_of(ExchangePoint::Prime).unwrap() as usize].clone();
        let fabric = Fabric::new(link, me, session, WORKERS, AXES, &LENS, default)
            .with_phase_counts(counts)
            .with_phase_rules(table.rules());
        let mut march = March {
            me,
            table: table.clone(),
            fabric,
            retries: 0,
            troubled_phases: 0,
            audits: 0,
            fine_stage_opens_in_rejected_attempt: 0,
            early_grant_seen: false,
        };
        let walked = march.walk();
        if let Err(e) = &walked {
            march.fabric.abort(&e.to_string());
            return Err(walked.unwrap_err());
        }
        march.fabric.grants().assert_drained()?;
        let high: Vec<usize> = (0..WORKERS as u32).map(|p| march.fabric.grants().high_water(WorkerId(p))).collect();
        march.fabric.finish(DEADLINE)?;
        Ok(format!(
            "REPORT worker={} retries={} troubled_phases={} audits={} rejected_attempt_fine_stage_opens={} early_grant={} high_water={:?}",
            me.0,
            march.retries,
            march.troubled_phases,
            march.audits,
            march.fine_stage_opens_in_rejected_attempt,
            u8::from(march.early_grant_seen),
            high
        ))
    };
    match run() {
        Ok(report) => {
            println!("{report}");
            std::process::exit(0)
        }
        Err(e) => {
            eprintln!("worker {me:?}: {e}");
            std::process::exit(2)
        }
    }
}

struct Outcome {
    code: Option<i32>,
    stdout: String,
    stderr: String,
}

fn spawn(test: &str, me: u32, coord: &std::path::Path) -> Child {
    let exe = std::env::current_exe().unwrap();
    Command::new(exe)
        .args([test, "--exact", "--nocapture", "--test-threads=1"])
        .env(ROLE, "worker")
        .env("W_ME", me.to_string())
        .env("W_COORD", coord)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap()
}

fn collect(mut children: Vec<Child>) -> Vec<Outcome> {
    let start = Instant::now();
    let mut out: Vec<Option<Outcome>> = (0..children.len()).map(|_| None).collect();
    loop {
        let mut all = true;
        for (i, child) in children.iter_mut().enumerate() {
            if out[i].is_some() {
                continue;
            }
            match child.try_wait().unwrap() {
                Some(status) => {
                    let mut stdout = String::new();
                    child.stdout.take().unwrap().read_to_string(&mut stdout).unwrap();
                    let mut stderr = String::new();
                    child.stderr.take().unwrap().read_to_string(&mut stderr).unwrap();
                    out[i] = Some(Outcome { code: status.code(), stdout, stderr });
                }
                None => all = false,
            }
        }
        if all {
            break;
        }
        if start.elapsed() > WATCHDOG {
            for child in children.iter_mut() {
                let _ = child.kill();
            }
            panic!("the watchdog had to intervene");
        }
        std::thread::sleep(Duration::from_millis(5));
    }
    out.into_iter().map(|o| o.expect("collected")).collect()
}

#[test]
fn three_workers_walk_the_refined_table_over_tcp_with_collectives() {
    if std::env::var(ROLE).is_ok() {
        worker_main();
    }
    let test = "three_workers_walk_the_refined_table_over_tcp_with_collectives";
    let coord = std::env::temp_dir().join(format!("symbi_phase_table_rv_{}", std::process::id()));
    let _ = std::fs::remove_file(&coord);
    let children = (0..WORKERS as u32).map(|me| spawn(test, me, &coord)).collect();
    let outcomes = collect(children);
    let report: String = outcomes
        .iter()
        .enumerate()
        .map(|(i, o)| format!("worker {i}: exit {:?}\n{}\n{}", o.code, o.stdout, o.stderr))
        .collect();
    for o in &outcomes {
        assert_eq!(o.code, Some(0), "{report}");
    }
    let mut seen = 0;
    for o in &outcomes {
        // libtest prints the test name without a newline, so the report may end that line
        let line = o
            .stdout
            .lines()
            .find_map(|l| l.find("REPORT").map(|at| &l[at..]))
            .unwrap_or_else(|| panic!("no report:\n{report}"));
        let field = |name: &str| -> String {
            line.split_whitespace()
                .find_map(|kv| kv.strip_prefix(&format!("{name}=")).map(str::to_string))
                .unwrap_or_else(|| panic!("{name} missing from {line}"))
        };
        assert_eq!(field("retries"), "1", "{line}");
        assert_eq!(field("troubled_phases"), "1", "{line}");
        // the rejected attempt exchanged the first substep's two stages and the second
        // substep's first; the rejected stage moved no state
        assert_eq!(field("rejected_attempt_fine_stage_opens"), "3", "{line}");
        if field("worker") == "2" {
            assert_eq!(field("early_grant"), "1", "c never held b's grant while awaiting the held result; the skew premise is vacuous: {line}");
        }
        // the prime audit, one per root stage per attempt (3 steps of 2 stages, one retried
        // step adds 2), one per checkpoint
        assert_eq!(field("audits"), (1 + 2 * 3 + 2 + 3).to_string(), "{line}");
        let high = line.split("high_water=").nth(1).unwrap();
        let held: Vec<usize> = high.trim_matches(|c| c == '[' || c == ']').split(", ").map(|v| v.parse().unwrap()).collect();
        assert!(held.iter().all(|&h| h <= AXES), "a peer's lead exceeded one grant per axis: {line}");
        eprintln!("{line}");
        seen += 1;
    }
    assert_eq!(seen, WORKERS);
}
