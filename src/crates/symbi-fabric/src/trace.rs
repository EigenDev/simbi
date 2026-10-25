// =============================================================================
// trace.rs
//
// a diagnostic record of one worker's waits, enabled by SIMBI_FABRIC_TRACE=<prefix>
// and written to <prefix>.w<worker>.tsv when the fabric is dropped.
// - every collective: its begin, the arrival of its result, and the idle pauses and
//   slept nanoseconds of its wait
// - on the coordinator: each contribution's arrival and the result's send
// - every exchange wait: its begin and end
// - every idle pause: its slept nanoseconds, in a histogram
// times are wall-clock nanoseconds, comparable across the processes of one host.
// a row holds an event name, an operation id, a time, a worker id or a count: no
// session credential and no payload. the rows are capped at MAX_ROWS (about 30 MB);
// later events are counted as dropped and the histogram keeps counting.
// with the variable unset the fabric holds no trace and records nothing.
// usage:
//   SIMBI_FABRIC_TRACE=/tmp/run simbi launch ...   # /tmp/run.w0.tsv, /tmp/run.w1.tsv
// =============================================================================

use std::io::Write;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// slept-time histogram bounds, microseconds: [0,75), [75,150), [150,300), [300,1000), [1000,inf)
const BOUNDS_US: [u128; 4] = [75, 150, 300, 1000];

/// the most rows one worker's trace holds
pub const MAX_ROWS: usize = 1 << 19;

pub struct Trace {
    path: std::path::PathBuf,
    rows: Vec<(&'static str, u64, u128, u64, u64)>,
    cap: usize,
    dropped: u64,
    /// the idle pauses and slept nanoseconds of the collective in progress
    pauses: u64,
    slept: u128,
    sleeps: u64,
    slept_total: u128,
    slept_max: u128,
    histogram: [u64; 5],
}

pub fn now_ns() -> u128 {
    SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_nanos())
}

impl Trace {
    /// the trace of worker `me` when SIMBI_FABRIC_TRACE names a prefix
    pub fn from_env(me: u32) -> Option<Box<Self>> {
        let prefix = std::env::var("SIMBI_FABRIC_TRACE").ok().filter(|p| !p.is_empty())?;
        Some(Box::new(Self::to_path(format!("{prefix}.w{me}.tsv").into(), MAX_ROWS)))
    }

    /// a trace written to `path` holding at most `cap` rows
    pub fn to_path(path: std::path::PathBuf, cap: usize) -> Self {
        Self {
            path,
            rows: Vec::with_capacity(cap.min(1 << 16)),
            cap,
            dropped: 0,
            pauses: 0,
            slept: 0,
            sleeps: 0,
            slept_total: 0,
            slept_max: 0,
            histogram: [0; 5],
        }
    }

    pub fn event(&mut self, name: &'static str, op: u64, a: u64, b: u64) {
        if self.rows.len() < self.cap {
            self.rows.push((name, op, now_ns(), a, b));
        } else {
            self.dropped += 1;
        }
    }

    pub fn rows(&self) -> usize {
        self.rows.len()
    }

    pub fn dropped(&self) -> u64 {
        self.dropped
    }

    pub fn begin_collective(&mut self, op: u32, kind: u8) {
        self.pauses = 0;
        self.slept = 0;
        self.event("begin", u64::from(op), u64::from(kind), 0);
    }

    pub fn end_collective(&mut self, op: u32) {
        let (pauses, slept) = (self.pauses, self.slept as u64);
        self.event("done", u64::from(op), pauses, slept);
    }

    pub fn slept(&mut self, actual: Duration) {
        let ns = actual.as_nanos();
        self.pauses += 1;
        self.slept += ns;
        self.sleeps += 1;
        self.slept_total += ns;
        self.slept_max = self.slept_max.max(ns);
        let us = ns / 1000;
        let bucket = BOUNDS_US.iter().position(|&b| us < b).unwrap_or(BOUNDS_US.len());
        self.histogram[bucket] += 1;
    }

    fn write(&self) -> std::io::Result<()> {
        let mut out = std::io::BufWriter::new(std::fs::File::create(&self.path)?);
        writeln!(
            out,
            "# rows={} dropped={} sleeps={} slept_ns={} slept_max_ns={} hist_us_lt75={} lt150={} lt300={} lt1000={} ge1000={}",
            self.rows.len(),
            self.dropped,
            self.sleeps, self.slept_total, self.slept_max, self.histogram[0], self.histogram[1], self.histogram[2],
            self.histogram[3], self.histogram[4]
        )?;
        writeln!(out, "event\top\tt_ns\ta\tb")?;
        for (name, op, t, a, b) in &self.rows {
            writeln!(out, "{name}\t{op}\t{t}\t{a}\t{b}")?;
        }
        out.flush()
    }
}

impl Drop for Trace {
    fn drop(&mut self) {
        if let Err(e) = self.write() {
            eprintln!("SIMBI fabric trace: writing {:?}: {e}", self.path);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rows_stop_at_the_cap_and_later_events_are_counted() {
        let path = std::env::temp_dir().join(format!("simbi-trace-test-{}.tsv", std::process::id()));
        let mut trace = Trace::to_path(path.clone(), 3);
        for op in 0..5 {
            trace.event("begin", op, 0, 0);
        }
        assert_eq!((trace.rows(), trace.dropped()), (3, 2));
        trace.slept(Duration::from_micros(80));
        assert_eq!(trace.histogram, [0, 1, 0, 0, 0], "an 80 us sleep falls in [75, 150)");
        drop(trace);
        let written = std::fs::read_to_string(&path).unwrap();
        std::fs::remove_file(&path).unwrap();
        assert!(written.starts_with("# rows=3 dropped=2 "), "the header reports the cap: {written}");
        assert_eq!(written.lines().count(), 2 + 3, "the header, the column names and three rows");
    }
}
