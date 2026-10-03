// =============================================================================
// worker_tracer_frames.rs
//
// the tracer populations launched workers move cross the fabric in frames of a
// bounded size. two workers on threads, each on a link with its own frame bound,
// gather a modest discrete and continuous population in many windows cut from
// the smaller bound, which the workers agree: every frame either sends fits
// both bounds, and every record arrives with its id and every field bit for
// bit, from an empty contributor, at an exact window boundary and through a
// partial final window. when the smaller bound carries no assembled word, every
// worker refuses by name before any window moves.
//
// run: cargo test -p symbi-sim --test worker_tracer_frames
// =============================================================================

use std::collections::VecDeque;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use symbi_fabric::frame::HEADER_LEN;
use symbi_fabric::{Fabric, FabricError, Header, Link, SessionId, WorkerId};
use symbi_sim::mass_transport::{ContainerId, ItoOrder};
use symbi_sim::tracers::{ContinuousTracerRecord, ContinuousTracerSet, TracerFlags, TracerSet};
use symbi_sim::worker_tracers::{gather_continuous, gather_discrete, record_words};
use symbi_xpu::HostMemory;

type Queued = (WorkerId, WorkerId, Header, Vec<u8>);

/// one worker's end of an in-process link shared across threads: its frames carry at most
/// `limit` payload bytes, sent or received, as a socket sized to that bound would. the shared
/// counter records the largest payload any worker sent
#[derive(Clone)]
struct Bounded {
    queue: Arc<Mutex<VecDeque<Queued>>>,
    limit: usize,
    largest: Arc<AtomicUsize>,
}

impl Bounded {
    fn shared(limits: [usize; 2]) -> [Bounded; 2] {
        let queue = Arc::new(Mutex::new(VecDeque::new()));
        let largest = Arc::new(AtomicUsize::new(0));
        limits.map(|limit| Bounded { queue: queue.clone(), limit, largest: largest.clone() })
    }
}

impl Link for Bounded {
    fn max_payload(&self) -> usize {
        self.limit
    }

    fn send(&self, from: WorkerId, to: WorkerId, header: Header, payload: &[u8]) -> Result<(), FabricError> {
        self.largest.fetch_max(payload.len(), Ordering::Relaxed);
        if payload.len() > self.limit {
            return Err(FabricError::protocol(to, format!("a frame of {} bytes exceeds the bound {}", payload.len(), self.limit)));
        }
        let mut bytes = [0u8; HEADER_LEN];
        header.encode(&mut bytes);
        let decoded = Header::decode(&bytes).expect("an encoded header decodes");
        self.queue.lock().unwrap().push_back((from, to, decoded, payload.to_vec()));
        Ok(())
    }

    fn poll(&self, me: WorkerId, payload: &mut Vec<u8>) -> Result<Option<(WorkerId, Header)>, FabricError> {
        let mut q = self.queue.lock().unwrap();
        let Some(pos) = q.iter().position(|(_, to, _, _)| *to == me) else {
            return Ok(None);
        };
        let (from, _, header, bytes) = q.remove(pos).expect("position found above");
        if bytes.len() > self.limit {
            return Err(FabricError::protocol(from, format!("a frame of {} bytes exceeds the bound {}", bytes.len(), self.limit)));
        }
        payload.clear();
        payload.extend_from_slice(&bytes);
        Ok(Some((from, header)))
    }
}

/// long enough for every window of these populations; a worker left waiting on a peer that
/// refused reports its deadline well inside the test's run
const DEADLINE: Duration = Duration::from_secs(10);

fn fabric(link: &Bounded, me: u32) -> Fabric<Bounded> {
    Fabric::new(link.clone(), WorkerId(me), SessionId(7), 2, 1, &[], vec![vec![0], vec![0]])
}

/// a discrete population of `n` tracers whose fields carry distinct bit patterns
fn discrete(n: usize, base: u64) -> TracerSet<2> {
    let mut set = TracerSet::<2> { weight: 0.125, run_seed: 41, next_id: 900, injection_remainder: 0.0625, ..Default::default() };
    for k in 0..n {
        let id = base + k as u64;
        set.x.push([0.1 + id as f64 * 1e-3, -0.0]);
        set.id.push(id);
        set.cohort.push((id % 3) as u16);
        set.owner.push(ContainerId(id * 7 + 1));
        set.step_owner.push(ContainerId(id * 7 + 2));
        set.flags.push(TracerFlags { escaped: k % 2 == 0, crossed_sink: k % 3 == 0, crossing_time: id as f64 * 0.5 });
        set.step_flags.push(TracerFlags { escaped: false, crossed_sink: k % 5 == 0, crossing_time: f64::from_bits(0x7ff0_0000_0000_0001) });
    }
    set
}

fn continuous(n: usize, base: u64) -> ContinuousTracerSet<2, HostMemory> {
    let mut set = ContinuousTracerSet::<2, HostMemory>::allocate(0, ItoOrder::Three).unwrap();
    set.weight = 0.125;
    set.run_seed = 41;
    set.next_id = 900;
    set.injection_remainder = 0.0625;
    for k in 0..n {
        let id = base + k as u64;
        set.push_host(ContinuousTracerRecord {
            x: [0.3 + id as f64 * 1e-3, -0.0],
            step_x: [0.2, f64::from_bits(0x7ff0_0000_0000_0001)],
            id,
            cohort: (id % 3) as u16,
            owner: ContainerId(id * 7 + 1),
            escaped: (k % 2) as u8,
            crossed_sink: (k % 3 == 0) as u8,
            crossing_time: id as f64 * 0.5,
            random_counter: id * 11,
        })
        .unwrap();
    }
    set
}

fn bits(x: &[f64]) -> Vec<u64> {
    x.iter().map(|v| v.to_bits()).collect()
}

fn same_discrete(got: &TracerSet<2>, want: &TracerSet<2>, label: &str) {
    assert_eq!(got.id, want.id, "{label}: ids");
    assert_eq!(got.x.iter().flat_map(|p| bits(p)).collect::<Vec<_>>(), want.x.iter().flat_map(|p| bits(p)).collect::<Vec<_>>(), "{label}: positions");
    assert_eq!(got.cohort, want.cohort, "{label}: cohorts");
    assert_eq!(got.owner, want.owner, "{label}: owners");
    assert_eq!(got.step_owner, want.step_owner, "{label}: step owners");
    let flags = |f: &[TracerFlags]| f.iter().map(|g| (g.escaped, g.crossed_sink, g.crossing_time.to_bits())).collect::<Vec<_>>();
    assert_eq!(flags(&got.flags), flags(&want.flags), "{label}: flags");
    assert_eq!(flags(&got.step_flags), flags(&want.step_flags), "{label}: step flags");
    assert_eq!(
        (got.weight.to_bits(), got.run_seed, got.next_id, got.injection_remainder.to_bits()),
        (want.weight.to_bits(), want.run_seed, want.next_id, want.injection_remainder.to_bits()),
        "{label}: population state"
    );
}

fn records(set: &ContinuousTracerSet<2, HostMemory>) -> Vec<[u64; 11]> {
    (0..set.len)
        .map(|ii| {
            let r = set.record_host(ii).unwrap();
            [
                r.x[0].to_bits(), r.x[1].to_bits(), r.step_x[0].to_bits(), r.step_x[1].to_bits(), r.id, u64::from(r.cohort),
                r.owner.0, u64::from(r.escaped), u64::from(r.crossed_sink), r.crossing_time.to_bits(), r.random_counter,
            ]
        })
        .collect()
}

/// gather both populations of tile 0 (worker 0) and tile 1 (worker 1), each worker on a link of
/// its own bound; the gathered sets every worker holds and the largest frame sent
#[allow(clippy::type_complexity)]
fn gather(
    limits: [usize; 2],
    tiles: [(TracerSet<2>, ContinuousTracerSet<2, HostMemory>); 2],
) -> (Vec<(Vec<TracerSet<2>>, Vec<ContinuousTracerSet<2, HostMemory>>)>, usize) {
    let links = Bounded::shared(limits);
    let largest = links[0].largest.clone();
    let handles: Vec<_> = tiles
        .into_iter()
        .zip(links)
        .enumerate()
        .map(|(me, ((d, c), link))| {
            std::thread::spawn(move || {
                let mut f = fabric(&link, me as u32);
                let discrete = gather_discrete(&mut f, 2, &[(me, &d)], DEADLINE).expect("discrete gather");
                let continuous = gather_continuous(&mut f, 2, &[(me, &c)], DEADLINE).expect("continuous gather");
                (discrete, continuous)
            })
        })
        .collect();
    let out = handles.into_iter().map(|h| h.join().expect("a worker thread panicked")).collect();
    (out, largest.load(Ordering::Relaxed))
}

#[test]
fn modest_populations_cross_small_frames_whole() {
    // worker 0's link carries 73 bytes, windows of 4 words (9 + 16 * 4); worker 1's carries
    // 201, which alone would cut windows of 12. the agreed bound is the smaller, every
    // frame fits it, and the windows land on the 4-word boundaries below
    let limits = [73, 201];
    let limit = 73;
    for (n0, n1, label) in [
        // an empty contributor
        (23, 0, "tile 1 empty"),
        // records ending on a window boundary: (6 + 6) records of 10 words, 120 words
        (6, 6, "exact boundary"),
        // a partial final window: 31 records of 10 words, 310 words
        (17, 14, "partial final window"),
    ] {
        let words = (n0 + n1) * record_words(2);
        assert!(words % 4 == 0 || label != "exact boundary", "{label}: the discrete words leave a partial window");
        let tiles = [(discrete(n0, 0), continuous(n0, 0)), (discrete(n1, 500), continuous(n1, 500))];
        let want: Vec<(TracerSet<2>, Vec<[u64; 11]>)> = tiles.iter().map(|(d, c)| (d.clone(), records(c))).collect();
        let (gathered, largest) = gather(limits, tiles);
        assert!(largest <= limit, "{label}: a frame of {largest} bytes exceeded the bound {limit}");
        assert!(words > 4 * 4, "{label}: the population fits a few windows; the arm is vacuous");
        for (worker, (d, c)) in gathered.iter().enumerate() {
            for tile in 0..2 {
                same_discrete(&d[tile], &want[tile].0, &format!("{label}, worker {worker}, tile {tile}"));
                assert_eq!(records(&c[tile]), want[tile].1, "{label}, worker {worker}, tile {tile}: continuous records");
                assert_eq!(c[tile].next_id, 900);
                assert_eq!(c[tile].injection_remainder.to_bits(), 0.0625f64.to_bits());
            }
        }
    }
}

#[test]
fn a_bound_that_carries_no_word_is_refused_on_every_worker() {
    // worker 0's 24-byte frames carry the agreement's scalar but no assembled word; worker 1's
    // carry 1000 bytes. both refuse at the agreed bound, so neither waits on the other
    let tiles = [(discrete(3, 0), continuous(3, 0)), (discrete(3, 500), continuous(3, 500))];
    let started = std::time::Instant::now();
    let handles: Vec<_> = tiles
        .into_iter()
        .zip(Bounded::shared([24, 1000]))
        .enumerate()
        .map(|(me, ((d, _), link))| {
            std::thread::spawn(move || {
                let mut f = fabric(&link, me as u32);
                gather_discrete(&mut f, 2, &[(me, &d)], DEADLINE).map(|_| ()).unwrap_err().to_string()
            })
        })
        .collect();
    for (worker, h) in handles.into_iter().enumerate() {
        let err = h.join().expect("a worker thread panicked");
        assert!(err.contains("carries no assembled word"), "worker {worker} ended otherwise: {err}");
    }
    assert!(started.elapsed() < DEADLINE, "a worker waited out its deadline before refusing");
}
