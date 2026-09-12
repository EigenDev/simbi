// =============================================================================
// two_process.rs
//
// the fabric across real processes. every test spawns this test binary again
// as worker processes that rendezvous over TCP on the loopback interface and
// run one scenario; the parent is the outer watchdog and reads exit codes and
// stderr. a scenario passes when every worker exits zero within the deadline
// and no watchdog kill was needed; a refusal scenario passes when every worker
// exits nonzero within the deadline for the stated reason.
//
// the payload of transfer 0 is sized by the scenario so that one frame
// exceeds any socket buffer, which makes partial writes and partial reads the
// ordinary path rather than an edge.
//
// run: cargo test -p symbi-fabric --test two_process
// =============================================================================

use std::alloc::{GlobalAlloc, Layout, System};
use std::io::{BufRead, BufReader, Read};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::{Duration, Instant};
use symbi_fabric::rendezvous::{Identity, Rendezvous, connect};
use symbi_fabric::{Digest, Epoch, Fabric, FabricError, OpKind, PhaseSpec, TransferId, WorkerId};

/// every allocation in this binary is counted, so a worker can show that a phase and the
/// session end allocate nothing and never raise the high-water mark.
struct Counting;

static ALLOCS: AtomicU64 = AtomicU64::new(0);
static LIVE: AtomicUsize = AtomicUsize::new(0);
static HIGH: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        let live = LIVE.fetch_add(layout.size(), Ordering::Relaxed) + layout.size();
        HIGH.fetch_max(live, Ordering::Relaxed);
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        LIVE.fetch_sub(layout.size(), Ordering::Relaxed);
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

const ROLE: &str = "SYMBI_FABRIC_TEST_ROLE";
const TRANSFER_DEADLINE: Duration = Duration::from_secs(4);
const STARTUP_DEADLINE: Duration = Duration::from_secs(6);
const WATCHDOG: Duration = Duration::from_secs(25);
const EXIT_FABRIC: i32 = 2;

fn env_u64(name: &str, default: u64) -> u64 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn value(worker: u32, transfer: u32, i: usize) -> f64 {
    1.0 + f64::from(worker) * 0.5 + f64::from(transfer) * 0.01 + i as f64 * 1e-6
}

fn epoch(point: u8, axis: u8) -> Epoch {
    Epoch {
        step: 1,
        attempt: 0,
        point,
        axis,
    }
}

/// the scenario a worker process runs, from its environment.
struct Scenario {
    me: WorkerId,
    workers: u32,
    len0: usize,
    /// service the link this long before opening axis 0 (a pending collective result)
    wait_before_open_ms: u64,
    /// service the link this long between axis 0 and axis 1
    delay_axis1_ms: u64,
    /// sleep without servicing this long before the first progress (a slow receiver)
    slow_receiver_ms: u64,
    /// worker 1 writes garbage to worker 0 before opening
    garbage: bool,
    /// print READY after the handshake and then stall forever (the kill target)
    stall_after_ready: bool,
    expect_grant_waits: bool,
    /// the phases and the session end must allocate nothing and hold the high-water mark
    expect_no_alloc: bool,
    expect_partial_writes: bool,
    expect_partial_reads: bool,
    expect_early_grant: bool,
    /// three workers: 1 and 2 exchange after a collective whose result the coordinator
    /// delays to worker 2, so worker 1's grant reaches worker 2 before its own result
    race: bool,
    /// worker 1 streams blocks of varying size to a slow coordinator under a credit that
    /// admits two small blocks while a large one still occupies the outbox
    blocks: bool,
}

impl Scenario {
    fn from_env() -> Self {
        Self {
            me: WorkerId(env_u64("W_ME", 0) as u32),
            workers: env_u64("W_N", 2) as u32,
            len0: env_u64("W_LEN0", 1000) as usize,
            wait_before_open_ms: env_u64("W_WAIT_OPEN_MS", 0),
            delay_axis1_ms: env_u64("W_DELAY_AXIS1_MS", 0),
            slow_receiver_ms: env_u64("W_SLOW_RECV_MS", 0),
            garbage: env_u64("W_GARBAGE", 0) == 1,
            stall_after_ready: env_u64("W_STALL", 0) == 1,
            expect_grant_waits: env_u64("W_EXPECT_GRANT_WAITS", 0) == 1,
            expect_no_alloc: env_u64("W_EXPECT_NO_ALLOC", 0) == 1,
            expect_partial_writes: env_u64("W_EXPECT_PARTIAL_WRITES", 0) == 1,
            expect_partial_reads: env_u64("W_EXPECT_PARTIAL_READS", 0) == 1,
            expect_early_grant: env_u64("W_EXPECT_EARLY_GRANT", 0) == 1,
            race: env_u64("W_RACE", 0) == 1,
            blocks: env_u64("W_BLOCKS", 0) == 1,
        }
    }
}

fn identity(workers: u32) -> Identity {
    Identity {
        credential: env_u64("W_CRED", 0xC0FFEE),
        build_id: "two_process".into(),
        config_digest: Digest(env_u64("W_CONFIG", 11)),
        plan_digest: Digest(env_u64("W_PLAN", 22)),
        placement_digest: Digest(env_u64("W_PLACEMENT", 33)),
        ng: 2,
        workers,
        block_credit: BLOCK_CREDIT,
    }
}

/// transfers: 0 is A -> B on axis 0 (len0), 1 is B -> A on axis 0 (len 3), 2 is A -> B on
/// axis 1 (len 2), 3 is B -> A on axis 1 (len 2). a third worker, when present, exchanges
/// nothing and only takes part in the handshake.
fn run_worker(s: &Scenario) -> Result<(), FabricError> {
    let r = Rendezvous {
        me: s.me,
        coordinator: "127.0.0.1:0".parse().expect("socket address"),
        announce: Some(std::path::PathBuf::from(
            std::env::var("W_COORD").expect("rendezvous file"),
        )),
        identity: identity(s.workers),
        startup: STARTUP_DEADLINE,
    };
    let lens = [s.len0, 3, 2, 2];
    let max_payload = (lens.iter().max().copied().unwrap() * 8).max(BLOCK_CREDIT as usize);
    let (link, session) = connect(&r, max_payload)?;
    println!("READY");
    if s.stall_after_ready {
        std::thread::sleep(Duration::from_secs(60));
    }
    let a = WorkerId(0);
    let b = WorkerId(1);
    let n = s.workers as usize;
    let mut sends = vec![vec![0u32; 2]; n];
    if s.race {
        if s.me == WorkerId(1) {
            sends[2] = vec![1, 0];
        } else if s.me == WorkerId(2) {
            sends[1] = vec![1, 0];
        }
    } else if s.me == a {
        sends[1] = vec![1, 1];
    } else if s.me == b {
        sends[0] = vec![1, 1];
    }
    let peer_bytes_before = link.peer_buffer_bytes();
    let mut fabric =
        Fabric::new(link, s.me, session, n, 2, &lens, sends).with_block_credit(BLOCK_CREDIT);
    // a local failure tells every peer before this worker leaves
    let result = run_phases(s, &mut fabric, peer_bytes_before);
    if let Err(e) = &result {
        fabric.abort(&e.to_string());
    }
    result
}

const BLOCK_MAX: usize = 2 << 20;
const BLOCK_CREDIT: u32 = 2 * BLOCK_MAX as u32;
const BLOCK_COUNT: u32 = 9;

fn block_len(seq: u32) -> usize {
    [BLOCK_MAX, BLOCK_MAX / 3, BLOCK_MAX / 2][seq as usize % 3]
}

/// the block scenario: worker 1 sends nine blocks whose sizes cycle large, third, half;
/// the coordinator sleeps before releasing each one, so the socket fills and the sender
/// must wait for the outbox as well as for credit.
fn run_blocks(s: &Scenario, fabric: &mut Fabric<symbi_fabric::TcpLink>) -> Result<(), FabricError> {
    if s.me == WorkerId(0) {
        let mut received = 0u32;
        let start = Instant::now();
        while received < BLOCK_COUNT {
            fabric.service()?;
            if let Some((from, seq, payload)) = fabric.staged_block() {
                if from != WorkerId(1) || seq != received || payload.len() != block_len(seq) {
                    eprintln!(
                        "block {seq} from {from:?} of {} bytes; expected {} of {}",
                        payload.len(),
                        received,
                        block_len(received)
                    );
                    std::process::exit(4);
                }
                if payload.iter().any(|&b| b != seq as u8) {
                    eprintln!("block {seq} payload corrupted");
                    std::process::exit(4);
                }
                std::thread::sleep(Duration::from_millis(30));
                fabric.release_block()?;
                received += 1;
            }
            if start.elapsed() > TRANSFER_DEADLINE * 3 {
                eprintln!("blocks stalled at {received}");
                std::process::exit(4);
            }
            std::thread::sleep(Duration::from_micros(50));
        }
    } else {
        let mut payload = vec![0u8; BLOCK_MAX];
        for seq in 0..BLOCK_COUNT {
            let len = block_len(seq);
            payload[..len].fill(seq as u8);
            fabric.send_block(&payload[..len], TRANSFER_DEADLINE)?;
        }
        fabric.drain_blocks(TRANSFER_DEADLINE)?;
    }
    fabric.finish(TRANSFER_DEADLINE)
}

/// the race scenario: worker 0 coordinates and holds worker 2's result for 300 ms; workers
/// 1 and 2 contribute, then exchange transfers 0 (1 -> 2) and 1 (2 -> 1) on axis 0. worker 1
/// gets its result at once, opens, and grants worker 2, whose result is still on hold.
fn run_race(s: &Scenario, fabric: &mut Fabric<symbi_fabric::TcpLink>) -> Result<(), FabricError> {
    let e = epoch(1, 0);
    let one = WorkerId(1);
    let two = WorkerId(2);
    if s.me == WorkerId(0) {
        fabric.delay_result_to(two, Duration::from_millis(300));
    }
    let result = fabric.collective(OpKind::Any, u64::from(s.me == one), TRANSFER_DEADLINE)?;
    if result != 1 {
        eprintln!("Any over (0, 1, 0) gave {result}");
        std::process::exit(4);
    }
    if s.me == two && !fabric.grants().is_tabled(one, e) {
        eprintln!("worker 2's result arrived before worker 1's grant; the race was not exercised");
        std::process::exit(4);
    }
    let (send, recv) = match s.me {
        WorkerId(1) => (Some((TransferId(0), two)), Some((TransferId(1), two))),
        WorkerId(2) => (Some((TransferId(1), one)), Some((TransferId(0), one))),
        _ => (None, None),
    };
    if let Some((id, _)) = send {
        fabric.pack(id, |buf| {
            for (i, v) in buf.iter_mut().enumerate() {
                *v = value(s.me.0, id.0, i);
            }
        })?;
    }
    let sends: Vec<_> = send.into_iter().collect();
    let receives: Vec<_> = recv.into_iter().collect();
    fabric.open(PhaseSpec {
        epoch: e,
        sends: &sends,
        receives: &receives,
    })?;
    fabric.wait(TRANSFER_DEADLINE)?;
    if let Some((id, from)) = recv {
        for (i, v) in fabric.payload(id)?.iter().enumerate() {
            if v.to_bits() != value(from.0, id.0, i).to_bits() {
                eprintln!("transfer {id:?} value {i} wrong");
                std::process::exit(4);
            }
        }
        fabric.mark_unpacked(id)?;
    }
    fabric.close()?;
    // the second axis is local-only for everyone; a second collective closes the run
    fabric.open(PhaseSpec {
        epoch: epoch(1, 1),
        sends: &[],
        receives: &[],
    })?;
    fabric.wait(TRANSFER_DEADLINE)?;
    fabric.close()?;
    let dt = fabric.collective(
        OpKind::Min,
        (0.5 + f64::from(s.me.0)).to_bits(),
        TRANSFER_DEADLINE,
    )?;
    if f64::from_bits(dt) != 0.5 {
        eprintln!("Min gave {}", f64::from_bits(dt));
        std::process::exit(4);
    }
    fabric.finish(TRANSFER_DEADLINE)
}

fn run_phases(
    s: &Scenario,
    fabric: &mut Fabric<symbi_fabric::TcpLink>,
    peer_bytes_before: usize,
) -> Result<(), FabricError> {
    if s.race {
        return run_race(s, fabric);
    }
    if s.blocks {
        return run_blocks(s, fabric);
    }
    let a = WorkerId(0);
    let b = WorkerId(1);
    let bytes_before = fabric.buffer_bytes();
    let allocs_before = ALLOCS.load(Ordering::Relaxed);
    let high_before = HIGH.load(Ordering::Relaxed);
    if s.garbage && s.me == b {
        fabric.link().inject_raw(
            a,
            b"this is not a frame at all, and it is long enough to cover a header",
        );
    }
    let (my_sends, my_receives): ([(TransferId, WorkerId); 2], [(TransferId, WorkerId); 2]) =
        if s.me == a {
            (
                [(TransferId(0), b), (TransferId(2), b)],
                [(TransferId(1), b), (TransferId(3), b)],
            )
        } else if s.me == b {
            (
                [(TransferId(1), a), (TransferId(3), a)],
                [(TransferId(0), a), (TransferId(2), a)],
            )
        } else {
            // a bystander: local-only phases throughout
            for axis in 0..2u8 {
                fabric.open(PhaseSpec {
                    epoch: epoch(1, axis),
                    sends: &[],
                    receives: &[],
                })?;
                fabric.wait(TRANSFER_DEADLINE)?;
                fabric.close()?;
            }
            return fabric.finish(TRANSFER_DEADLINE);
        };
    let peer = if s.me == a { b } else { a };
    for axis in 0..2usize {
        let e = epoch(1, axis as u8);
        let (send, recv) = (my_sends[axis], my_receives[axis]);
        fabric.pack(send.0, |buf| {
            for (i, v) in buf.iter_mut().enumerate() {
                *v = value(s.me.0, send.0.0, i);
            }
        })?;
        if axis == 0 && s.wait_before_open_ms > 0 {
            fabric.service_for(Duration::from_millis(s.wait_before_open_ms))?;
            if s.expect_early_grant && !fabric.grants().is_tabled(peer, e) {
                eprintln!("the peer's grant for {e:?} was not tabled while waiting");
                std::process::exit(4);
            }
        }
        if axis == 1 && s.delay_axis1_ms > 0 {
            fabric.service_for(Duration::from_millis(s.delay_axis1_ms))?;
        }
        fabric.open(PhaseSpec {
            epoch: e,
            sends: &[send],
            receives: &[recv],
        })?;
        if axis == 0 && s.slow_receiver_ms > 0 {
            std::thread::sleep(Duration::from_millis(s.slow_receiver_ms));
        }
        fabric.wait(TRANSFER_DEADLINE)?;
        let got = fabric.payload(recv.0)?;
        for (i, v) in got.iter().enumerate() {
            let want = value(peer.0, recv.0.0, i);
            if v.to_bits() != want.to_bits() {
                eprintln!("transfer {:?} value {i}: {v} != {want}", recv.0);
                std::process::exit(4);
            }
        }
        fabric.mark_unpacked(recv.0)?;
        fabric.close()?;
        if fabric.grants().tabled(peer) > 2 {
            eprintln!("grant table exceeds the axis count");
            std::process::exit(4);
        }
    }
    fabric.finish(TRANSFER_DEADLINE)?;
    let allocs = ALLOCS.load(Ordering::Relaxed) - allocs_before;
    let high_after = HIGH.load(Ordering::Relaxed);
    let stats = fabric.stats();
    let link_stats = fabric.link().stats();
    if fabric.buffer_bytes() != bytes_before
        || fabric.link().peer_buffer_bytes() != peer_bytes_before
    {
        eprintln!("buffer bytes changed during the session");
        std::process::exit(4);
    }
    if s.expect_no_alloc && (allocs != 0 || high_after != high_before) {
        eprintln!("the phases allocated {allocs} times; high-water {high_before} -> {high_after}");
        std::process::exit(4);
    }
    if s.expect_grant_waits && stats.grant_waits == 0 {
        eprintln!("no send waited for a grant; the delay exerted no pressure");
        std::process::exit(4);
    }
    if s.expect_partial_writes && link_stats.partial_writes == 0 {
        eprintln!("no frame was written in parts; the payload never filled the socket");
        std::process::exit(4);
    }
    if s.expect_partial_reads && link_stats.partial_reads == 0 {
        eprintln!("no frame was read in parts");
        std::process::exit(4);
    }
    println!(
        "STATS grant_waits={} link_waits={} passes={} partial_writes={} partial_reads={} allocs={allocs}",
        stats.grant_waits,
        stats.link_waits,
        stats.passes,
        link_stats.partial_writes,
        link_stats.partial_reads
    );
    Ok(())
}

fn worker_main() -> ! {
    let s = Scenario::from_env();
    match run_worker(&s) {
        Ok(()) => std::process::exit(0),
        Err(e) => {
            eprintln!("worker {:?}: {e}", s.me);
            std::process::exit(EXIT_FABRIC);
        }
    }
}

struct Spawned {
    child: Child,
    stderr: std::process::ChildStderr,
    stdout: BufReader<std::process::ChildStdout>,
}

/// a fresh rendezvous file path; the coordinator writes its bound address there.
fn rendezvous_file() -> std::path::PathBuf {
    static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let n = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let path = std::env::temp_dir().join(format!("symbi_fabric_rv_{}_{n}", std::process::id()));
    let _ = std::fs::remove_file(&path);
    path
}

fn spawn(
    test: &str,
    me: u32,
    workers: u32,
    coord: &std::path::Path,
    extra: &[(&str, String)],
) -> Spawned {
    let exe = std::env::current_exe().unwrap();
    let mut cmd = Command::new(exe);
    cmd.args([test, "--exact", "--nocapture", "--test-threads=1"])
        .env(ROLE, "worker")
        .env("W_ME", me.to_string())
        .env("W_N", workers.to_string())
        .env("W_COORD", coord)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    for (k, v) in extra {
        cmd.env(k, v);
    }
    let mut child = cmd.spawn().unwrap();
    let stderr = child.stderr.take().unwrap();
    let stdout = BufReader::new(child.stdout.take().unwrap());
    Spawned {
        child,
        stderr,
        stdout,
    }
}

/// wait for READY on a worker's stdout.
fn wait_ready(s: &mut Spawned) {
    let start = Instant::now();
    let mut line = String::new();
    loop {
        line.clear();
        let n = s.stdout.read_line(&mut line).unwrap();
        // libtest prints the test name without a newline, so READY may end that same line
        if n == 0 || line.trim_end().ends_with("READY") {
            return;
        }
        assert!(start.elapsed() < WATCHDOG, "no READY before the watchdog");
    }
}

struct Outcome {
    code: Option<i32>,
    stderr: String,
    elapsed: Duration,
}

/// the outer watchdog: every worker must exit on its own before `WATCHDOG`; a kill here is
/// a test failure, never a recovery.
fn collect(mut spawned: Vec<Spawned>) -> Vec<Outcome> {
    let start = Instant::now();
    let mut out: Vec<Option<Outcome>> = (0..spawned.len()).map(|_| None).collect();
    loop {
        let mut all = true;
        for (i, s) in spawned.iter_mut().enumerate() {
            if out[i].is_some() {
                continue;
            }
            match s.child.try_wait().unwrap() {
                Some(status) => {
                    let mut stderr = String::new();
                    s.stderr.read_to_string(&mut stderr).unwrap();
                    let mut rest = String::new();
                    s.stdout.read_to_string(&mut rest).unwrap();
                    out[i] = Some(Outcome {
                        code: status.code(),
                        stderr,
                        elapsed: start.elapsed(),
                    });
                }
                None => all = false,
            }
        }
        if all {
            break;
        }
        if start.elapsed() > WATCHDOG {
            for s in spawned.iter_mut() {
                let _ = s.child.kill();
            }
            panic!("the watchdog had to intervene");
        }
        std::thread::sleep(Duration::from_millis(5));
    }
    out.into_iter().map(|o| o.expect("collected")).collect()
}

fn in_worker_role() -> bool {
    std::env::var(ROLE).is_ok()
}

fn run_pair(test: &str, extra_a: &[(&str, String)], extra_b: &[(&str, String)]) -> Vec<Outcome> {
    let coord = rendezvous_file();
    let a = spawn(test, 0, 2, &coord, extra_a);
    let b = spawn(test, 1, 2, &coord, extra_b);
    collect(vec![a, b])
}

fn assert_all_ok(outcomes: &[Outcome]) {
    let report: String = outcomes
        .iter()
        .enumerate()
        .map(|(i, o)| {
            format!(
                "worker {i}: exit {:?} after {:?}\n{}",
                o.code, o.elapsed, o.stderr
            )
        })
        .collect();
    for o in outcomes {
        assert_eq!(o.code, Some(0), "{report}");
    }
}

fn assert_all_refused(outcomes: &[Outcome], within: Duration) {
    for (i, o) in outcomes.iter().enumerate() {
        assert_ne!(o.code, Some(0), "worker {i} exited zero:\n{}", o.stderr);
        assert!(
            o.elapsed < within,
            "worker {i} took {:?} to terminate",
            o.elapsed
        );
    }
}

#[test]
fn seed_exchange_across_two_processes() {
    if in_worker_role() {
        worker_main();
    }
    let no_alloc = [("W_EXPECT_NO_ALLOC", "1".to_string())];
    assert_all_ok(&run_pair(
        "seed_exchange_across_two_processes",
        &no_alloc,
        &no_alloc,
    ));
}

/// transfer 0 is 32 MB: many partial writes on the sender and partial reads on the receiver.
#[test]
fn a_payload_larger_than_the_socket_buffer_completes() {
    if in_worker_role() {
        worker_main();
    }
    let len = [
        ("W_LEN0", (4usize << 20).to_string()),
        ("W_EXPECT_NO_ALLOC", "1".to_string()),
    ];
    assert_all_ok(&run_pair(
        "a_payload_larger_than_the_socket_buffer_completes",
        &len,
        &len,
    ));
}

/// the receiver sleeps without servicing while the sender's frame fills the socket: the
/// sender writes the frame in parts, the receiver reads it in parts, and neither side
/// allocates or raises its high-water mark while under that backpressure.
#[test]
fn a_slow_receiver_leaves_buffers_bounded() {
    if in_worker_role() {
        worker_main();
    }
    let len = ("W_LEN0", (2usize << 20).to_string());
    assert_all_ok(&run_pair(
        "a_slow_receiver_leaves_buffers_bounded",
        &[
            len.clone(),
            ("W_EXPECT_PARTIAL_WRITES", "1".into()),
            ("W_EXPECT_NO_ALLOC", "1".into()),
        ],
        &[
            len,
            ("W_SLOW_RECV_MS", "400".into()),
            ("W_EXPECT_PARTIAL_READS", "1".into()),
            ("W_EXPECT_NO_ALLOC", "1".into()),
        ],
    ));
}

/// worker 1 services the link for a while before opening axis 0, as it will while waiting
/// for a collective result; worker 0's grant arrives during that wait and is tabled. no
/// collective exists yet, so this establishes early-grant tabling alone; the ordering of a
/// grant against a real Result frame is a collective gate.
#[test]
fn early_grant_tabled_while_servicing() {
    if in_worker_role() {
        worker_main();
    }
    assert_all_ok(&run_pair(
        "early_grant_tabled_while_servicing",
        &[],
        &[
            ("W_WAIT_OPEN_MS", "500".into()),
            ("W_EXPECT_EARLY_GRANT", "1".into()),
        ],
    ));
}

/// worker 1 opens axis 1 late; worker 0's axis-1 send waits for the grant.
#[test]
fn peer_opens_next_axis_late() {
    if in_worker_role() {
        worker_main();
    }
    assert_all_ok(&run_pair(
        "peer_opens_next_axis_late",
        &[("W_EXPECT_GRANT_WAITS", "1".into())],
        &[("W_DELAY_AXIS1_MS", "500".into())],
    ));
}

/// a third worker that exchanges nothing: local-only phases on a live mesh.
#[test]
fn a_bystander_worker_runs_local_only_phases() {
    if in_worker_role() {
        worker_main();
    }
    let coord = rendezvous_file();
    let test = "a_bystander_worker_runs_local_only_phases";
    let outcomes = collect(vec![
        spawn(test, 0, 3, &coord, &[]),
        spawn(test, 1, 3, &coord, &[]),
        spawn(test, 2, 3, &coord, &[]),
    ]);
    assert_all_ok(&outcomes);
}

#[test]
fn garbage_frame_aborts_every_worker() {
    if in_worker_role() {
        worker_main();
    }
    let outcomes = run_pair(
        "garbage_frame_aborts_every_worker",
        &[],
        &[("W_GARBAGE", "1".into())],
    );
    assert_all_refused(&outcomes, TRANSFER_DEADLINE * 3);
    assert!(
        outcomes[0].stderr.contains("protocol violation"),
        "{}",
        outcomes[0].stderr
    );
    assert!(
        outcomes[1].stderr.contains("aborted"),
        "the peer must learn of the failure by Abort:\n{}",
        outcomes[1].stderr
    );
}

#[test]
fn killed_peer_aborts_the_survivor() {
    if in_worker_role() {
        worker_main();
    }
    let coord = rendezvous_file();
    let test = "killed_peer_aborts_the_survivor";
    let a = spawn(test, 0, 2, &coord, &[]);
    let mut b = spawn(test, 1, 2, &coord, &[("W_STALL", "1".into())]);
    wait_ready(&mut b);
    b.child.kill().unwrap();
    let outcomes = collect(vec![a, b]);
    assert_ne!(outcomes[0].code, Some(0));
    assert!(
        outcomes[0].elapsed < TRANSFER_DEADLINE * 3,
        "survivor took {:?}",
        outcomes[0].elapsed
    );
    assert!(
        outcomes[0].stderr.contains("disconnected"),
        "{}",
        outcomes[0].stderr
    );
}

#[test]
fn mismatched_plan_digest_refused() {
    if in_worker_role() {
        worker_main();
    }
    let outcomes = run_pair(
        "mismatched_plan_digest_refused",
        &[],
        &[("W_PLAN", "9999".into())],
    );
    assert_all_refused(&outcomes, STARTUP_DEADLINE * 2);
    assert!(
        outcomes[0].stderr.contains("numerical plan"),
        "{}",
        outcomes[0].stderr
    );
    assert!(
        outcomes[1].stderr.contains("numerical plan"),
        "{}",
        outcomes[1].stderr
    );
}

#[test]
fn mismatched_placement_digest_refused() {
    if in_worker_role() {
        worker_main();
    }
    let outcomes = run_pair(
        "mismatched_placement_digest_refused",
        &[],
        &[("W_PLACEMENT", "9999".into())],
    );
    assert_all_refused(&outcomes, STARTUP_DEADLINE * 2);
    assert!(
        outcomes[0].stderr.contains("launcher"),
        "{}",
        outcomes[0].stderr
    );
}

#[test]
fn mismatched_credential_refused() {
    if in_worker_role() {
        worker_main();
    }
    let outcomes = run_pair(
        "mismatched_credential_refused",
        &[],
        &[("W_CRED", "1".into())],
    );
    assert_all_refused(&outcomes, STARTUP_DEADLINE * 2);
    assert!(
        outcomes[0].stderr.contains("credential"),
        "{}",
        outcomes[0].stderr
    );
}

#[test]
fn a_missing_worker_trips_the_startup_deadline() {
    if in_worker_role() {
        worker_main();
    }
    let coord = rendezvous_file();
    let outcomes = collect(vec![spawn(
        "a_missing_worker_trips_the_startup_deadline",
        0,
        2,
        &coord,
        &[],
    )]);
    assert_ne!(outcomes[0].code, Some(0));
    assert!(
        outcomes[0].stderr.contains("Startup"),
        "{}",
        outcomes[0].stderr
    );
}

/// a real collective result racing a peer's grant: the coordinator holds worker 2's Result
/// while worker 1, already holding its own, opens the next phase and grants worker 2. the
/// grant is tabled during worker 2's wait and consumed after its result arrives.
#[test]
fn a_peer_grant_arrives_before_the_collective_result() {
    if in_worker_role() {
        worker_main();
    }
    let coord = rendezvous_file();
    let test = "a_peer_grant_arrives_before_the_collective_result";
    let race = [("W_RACE", "1".to_string())];
    let outcomes = collect(vec![
        spawn(test, 0, 3, &coord, &race),
        spawn(test, 1, 3, &coord, &race),
        spawn(test, 2, 3, &coord, &race),
    ]);
    assert_all_ok(&outcomes);
}

/// variable-sized blocks to a slow coordinator: credit for two small blocks does not make
/// the outbox free, so the sender waits for both before each send.
#[test]
fn variable_sized_blocks_complete_under_backpressure() {
    if in_worker_role() {
        worker_main();
    }
    let blocks = [("W_BLOCKS", "1".to_string())];
    assert_all_ok(&run_pair(
        "variable_sized_blocks_complete_under_backpressure",
        &blocks,
        &blocks,
    ));
}
