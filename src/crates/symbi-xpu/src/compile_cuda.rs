// =============================================================================
// compile_cuda.rs
//
// ahead-of-time CUDA compilation: take a CUDA C source string, invoke nvcc
// --ptx, return the PTX bytes. lives in symbi-xpu because this is execution-side
// orchestration (a sibling of nvrtc.rs's runtime JIT).
//
// AOT-compiles a kernel when a caller prefers nvcc PTX over NVRTC. if nvcc is
// unavailable or can't compile here, returns None and the caller falls back to
// NVRTC.
//
// the host-compiler probe (`which_first` / `ccbin_retry_candidate`) is shared
// with symbi/build.rs's stub-PTX compile so the `-ccbin` fallback policy lives
// in ONE place.
//
// usage:
//   let ptx = try_compile_cuda("__global__ void k(...) { ... }", "k");
//   match ptx { Some(bytes) => { /* load PTX */ } None => { /* NVRTC */ } }
// =============================================================================

use std::io::Write;
use std::process::Command;

/// attempt to compile CUDA C source to PTX via nvcc.
/// returns the PTX bytes on success, None if nvcc is unavailable or fails.
pub fn try_compile_cuda(cuda_source: &str, kernel_name: &str) -> Option<Vec<u8>> {
    try_compile_cuda_with_includes(cuda_source, kernel_name, &[])
}

/// compile CUDA C source to PTX via nvcc with include directories.
pub fn try_compile_cuda_with_includes(
    cuda_source: &str,
    kernel_name: &str,
    include_dirs: &[&str],
) -> Option<Vec<u8>> {
    // detect nvcc
    let nvcc = find_nvcc()?;

    // nvcc may be PRESENT but its default host compiler unusable (host gcc too
    // new for the toolkit, or nvcc hard-wired to a g++ version absent in a
    // distrobox). probe ONCE: determine the working host-compiler args (possibly
    // `-ccbin <PATH g++>`), or None if nvcc can't compile here at all — then warn
    // once and fall back to NVRTC quietly, so a single environment failure is not
    // re-reported per kernel. real per-kernel codegen errors still surface below.
    let host_args = nvcc_host_ccbin(&nvcc)?;

    // write source to temp file
    let tmp_dir = std::env::temp_dir().join("symbi_cuda_compile");
    std::fs::create_dir_all(&tmp_dir).ok()?;

    let cu_path = tmp_dir.join(format!("{}.cu", kernel_name));
    let ptx_path = tmp_dir.join(format!("{}.ptx", kernel_name));

    let mut f = std::fs::File::create(&cu_path).ok()?;
    f.write_all(cuda_source.as_bytes()).ok()?;
    drop(f);

    // FMA fusion: nvcc's default `a*b + c -> fma(a,b,c)` stays ON: trust the
    // compiler, accept ULP-bounded drift vs the CPU (which doesn't auto-fuse).
    // do NOT re-introduce `--fmad=false`.
    let mut args = vec![
        "-ptx".to_string(),
        "-O3".to_string(),
        cu_path.to_str()?.to_string(),
        "-o".to_string(),
        ptx_path.to_str()?.to_string(),
    ];
    for dir in include_dirs {
        args.push("-I".to_string());
        args.push(dir.to_string());
    }
    // the working host-compiler args determined by the probe (empty, or -ccbin g++).
    args.extend(host_args.iter().cloned());

    let output = Command::new(&nvcc).args(&args).output().ok()?;

    if !output.status.success() {
        // nvcc errors are real failures — always surface them.
        let stderr = String::from_utf8_lossy(&output.stderr);
        eprintln!(
            "symbi: nvcc compilation failed for '{}': {}",
            kernel_name, stderr
        );
        eprintln!("symbi: source kept at {}", cu_path.display());
        return None;
    }

    // read PTX bytes
    let ptx_bytes = std::fs::read(&ptx_path).ok()?;

    // per-kernel success chatter is gated behind SYMBI_VERBOSE — most builds
    // don't need to see one line per kernel.
    if std::env::var("SYMBI_VERBOSE").is_ok() {
        eprintln!(
            "symbi: compiled {} -> {} ({} bytes PTX)",
            cu_path.display(),
            ptx_path.display(),
            ptx_bytes.len()
        );
    }
    // don't clean up — keep for inspection.

    Some(ptx_bytes)
}

/// determine the host-compiler args nvcc needs to compile here, cached per
/// process (one rustc per crate, so at most one probe/warning per crate).
/// returns Some(extra args) — empty if the default host compiler
/// works, or `-ccbin <PATH g++>` if the default failed but a PATH g++ works
/// (the common distrobox case) — or None if nvcc can't compile at all (warns
/// once; caller falls back to NVRTC).
fn nvcc_host_ccbin(nvcc: &str) -> Option<Vec<String>> {
    static ARGS: std::sync::OnceLock<Option<Vec<String>>> = std::sync::OnceLock::new();
    ARGS.get_or_init(|| {
        if probe_nvcc(nvcc, &[]) {
            return Some(vec![]);
        }
        // the default host compiler failed; retry with a PATH g++ IF the configured
        // NVCC_CCBIN is untrustworthy (shared policy with symbi/build.rs).
        if let Some(retry) = ccbin_retry_candidate() {
            let args = vec!["-ccbin".to_string(), retry.gxx.clone()];
            if probe_nvcc(nvcc, &args) {
                eprintln!(
                    "symbi: {}; using -ccbin {} for PTX.",
                    retry.reason, retry.gxx
                );
                return Some(args);
            }
        }
        eprintln!(
            "symbi: nvcc is present but cannot compile in this environment \
             (missing/unsupported host compiler — use the symbi-cuda distrobox, \
             or set NVCC_CCBIN). skipping ahead-of-time PTX; falling back to runtime NVRTC."
        );
        None
    })
    .clone()
}

/// the g++ to retry nvcc with via `-ccbin`, and why. produced by
/// [`ccbin_retry_candidate`].
pub struct CcbinRetry {
    pub gxx: String,
    pub reason: String,
}

/// the candidate host compiler to retry nvcc with via `-ccbin` when its default
/// host compiler is unusable. returns Some ONLY when the configured NVCC_CCBIN is
/// untrustworthy — UNSET (nvcc fell back to an absent hard-wired g++), or SET to a
/// path that DOESN'T EXIST (a stale value leaked from another machine via a shared
/// env; env vars carry no provenance, so a missing target is the signal it's
/// stale) — AND a `g++`/`c++` is resolvable on PATH. a real-but-failing NVCC_CCBIN
/// is the user's deliberate choice and is left alone (None). the CALLER verifies
/// the candidate actually works in its own context (a trivial probe at runtime, the
/// real stub compile at build time). shared by both so the policy lives in ONE place.
pub fn ccbin_retry_candidate() -> Option<CcbinRetry> {
    let nvcc_ccbin = std::env::var("NVCC_CCBIN").ok();
    let ccbin_stale = nvcc_ccbin
        .as_deref()
        .is_some_and(|p| !std::path::Path::new(p).exists());
    if nvcc_ccbin.is_none() || ccbin_stale {
        if let Some(gxx) = which_first(&["g++", "c++"]) {
            let reason = if ccbin_stale {
                format!(
                    "NVCC_CCBIN={} does not exist here (stale env)",
                    nvcc_ccbin.as_deref().unwrap_or("")
                )
            } else {
                "nvcc's default host compiler was unusable".to_string()
            };
            return Some(CcbinRetry { gxx, reason });
        }
    }
    None
}

/// probe whether nvcc can compile a trivial kernel with the given extra args.
fn probe_nvcc(nvcc: &str, extra: &[String]) -> bool {
    let tmp = std::env::temp_dir().join(format!("symbi_nvcc_probe_{}.cu", std::process::id()));
    let ptx = std::env::temp_dir().join(format!("symbi_nvcc_probe_{}.ptx", std::process::id()));
    if std::fs::write(&tmp, "__global__ void __symbi_nvcc_probe() {}\n").is_err() {
        return false;
    }
    let ok = Command::new(nvcc)
        .args(["-ptx", "-o"])
        .arg(&ptx)
        .arg(&tmp)
        .args(extra)
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false);
    let _ = std::fs::remove_file(&tmp);
    let _ = std::fs::remove_file(&ptx);
    ok
}

/// first of `names` resolvable on PATH (absolute path), for nvcc -ccbin fallback.
pub fn which_first(names: &[&str]) -> Option<String> {
    for n in names {
        if let Ok(out) = Command::new("which").arg(n).output() {
            if out.status.success() {
                let p = String::from_utf8_lossy(&out.stdout).trim().to_string();
                if !p.is_empty() {
                    return Some(p);
                }
            }
        }
    }
    None
}

/// find nvcc on the system. checks CUDA_HOME / CUDA_PATH, then PATH.
fn find_nvcc() -> Option<String> {
    if let Ok(cuda_home) = std::env::var("CUDA_HOME") {
        let nvcc = format!("{}/bin/nvcc", cuda_home);
        if std::path::Path::new(&nvcc).exists() {
            return Some(nvcc);
        }
    }
    if let Ok(cuda_path) = std::env::var("CUDA_PATH") {
        let nvcc = format!("{}/bin/nvcc", cuda_path);
        if std::path::Path::new(&nvcc).exists() {
            return Some(nvcc);
        }
    }
    let output = Command::new("which").arg("nvcc").output().ok()?;
    if output.status.success() {
        let path = String::from_utf8(output.stdout).ok()?;
        return Some(path.trim().to_string());
    }
    None
}
