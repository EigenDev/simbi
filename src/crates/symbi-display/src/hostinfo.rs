// =============================================================================
// hostinfo.rs
//
// host machine + process resource sampling for the dashboard's machine card:
// hostname, logical cpu count, and this process's resident memory against the
// node's physical ram. sampled on the SOLVER side (the compute node), so an
// attach client sees the compute node running the job even when the ui renders elsewhere.
//
// stdlib + libc only; every value is best-effort (0 / "?" on an unsupported
// platform or a failed syscall) so a sample never fails a run.
//
// usage:
//   let ram = hostinfo::total_ram();   // sampled once (static)
//   let rss = hostinfo::process_rss(); // sampled each cadence (grows)
// =============================================================================

use std::ffi::CStr;

use serde::{Deserialize, Serialize};

/// a host + process resource sample for the dashboard's machine card. serialized
/// into the attach snapshot, so a client reads the compute node's stats.
#[derive(Clone, Serialize, Deserialize)]
pub struct HostStats {
    pub hostname: String,
    pub cpu_count: usize,
    pub threads: usize,
    /// this process's resident set size in bytes (the run's live footprint).
    pub mem_rss: u64,
    /// the node's total physical ram in bytes.
    pub mem_total: u64,
}

impl HostStats {
    /// sample all fields now. cheap (a handful of syscalls); called each cadence
    /// so `mem_rss` tracks the run's growing footprint.
    pub fn sample() -> HostStats {
        HostStats {
            hostname: hostname(),
            cpu_count: cpu_count(),
            threads: thread_count(),
            mem_rss: process_rss(),
            mem_total: total_ram(),
        }
    }
}

/// the compute host's name (`gethostname`), or "?" on failure.
pub fn hostname() -> String {
    let mut buf = [0u8; 256];
    let rc = unsafe { libc::gethostname(buf.as_mut_ptr() as *mut libc::c_char, buf.len()) };
    if rc != 0 {
        return "?".into();
    }
    // guarantee a nul terminator even if the name was truncated to fill the buffer.
    buf[255] = 0;
    unsafe { CStr::from_ptr(buf.as_ptr() as *const libc::c_char) }
        .to_string_lossy()
        .into_owned()
}

/// logical cpu count (falls back to 1).
pub fn cpu_count() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
}

/// threads the run is configured to use: OMP_NUM_THREADS / NTHREADS if set (the
/// dashboard driver honors these), else the logical cpu count.
pub fn thread_count() -> usize {
    for var in ["OMP_NUM_THREADS", "NTHREADS", "RAYON_NUM_THREADS"] {
        if let Ok(n) = std::env::var(var) {
            if let Ok(n) = n.trim().parse::<usize>() {
                if n > 0 {
                    return n;
                }
            }
        }
    }
    cpu_count()
}

/// total physical ram in bytes, or 0 on failure.
#[cfg(target_os = "macos")]
pub fn total_ram() -> u64 {
    let mut val: u64 = 0;
    let mut len = std::mem::size_of::<u64>();
    let name = b"hw.memsize\0";
    let rc = unsafe {
        libc::sysctlbyname(
            name.as_ptr() as *const libc::c_char,
            &mut val as *mut _ as *mut libc::c_void,
            &mut len,
            std::ptr::null_mut(),
            0,
        )
    };
    if rc == 0 { val } else { 0 }
}

/// total physical ram in bytes, or 0 on failure. reads `MemTotal` (kB) from
/// /proc/meminfo.
#[cfg(target_os = "linux")]
pub fn total_ram() -> u64 {
    let s = std::fs::read_to_string("/proc/meminfo").unwrap_or_default();
    for line in s.lines() {
        if let Some(rest) = line.strip_prefix("MemTotal:") {
            if let Some(kb) = rest.split_whitespace().next() {
                if let Ok(kb) = kb.parse::<u64>() {
                    return kb * 1024;
                }
            }
        }
    }
    0
}

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
pub fn total_ram() -> u64 {
    0
}

/// this process's resident set size in bytes, or 0 on failure. this is the run's
/// live memory footprint — the number that matters against `total_ram` for an
/// oom watch.
#[cfg(target_os = "macos")]
pub fn process_rss() -> u64 {
    // proc_pidinfo(PROC_PIDTASKINFO) -> proc_taskinfo.pti_resident_size.
    let mut info: libc::proc_taskinfo = unsafe { std::mem::zeroed() };
    let size = std::mem::size_of::<libc::proc_taskinfo>() as libc::c_int;
    let n = unsafe {
        libc::proc_pidinfo(
            std::process::id() as libc::c_int,
            libc::PROC_PIDTASKINFO,
            0,
            &mut info as *mut _ as *mut libc::c_void,
            size,
        )
    };
    if n == size { info.pti_resident_size } else { 0 }
}

/// this process's resident set size in bytes, or 0 on failure. the second field
/// of /proc/self/statm is the resident page count.
#[cfg(target_os = "linux")]
pub fn process_rss() -> u64 {
    let s = std::fs::read_to_string("/proc/self/statm").unwrap_or_default();
    let rss_pages: u64 = s
        .split_whitespace()
        .nth(1)
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    if page > 0 { rss_pages * page as u64 } else { 0 }
}

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
pub fn process_rss() -> u64 {
    0
}
