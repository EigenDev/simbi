// =============================================================================
// build.rs
//
// CUDA-only build scaffold (gated on the `cuda` feature; a no-op otherwise).
// nvcc-compiles a tiny stub PTX (src/stub_kernels.cu) so cuModuleLoadData has
// a valid module to load at startup.
//
// the live GPU path needs nothing more: kernels are JIT-compiled at runtime by
// NVRTC from backend-neutral IR (symbi_ir::render_from_ir), so no CUDA
// source is generated or nvcc-compiled here.
// =============================================================================

use std::path::PathBuf;
use std::process::Command;

fn main() {
    if std::env::var("CARGO_FEATURE_CUDA").is_err() {
        return;
    }

    let out_dir = std::env::var("OUT_DIR").expect("OUT_DIR not set");
    compile_stub_ptx(&out_dir);
}

fn compile_stub_ptx(out_dir: &str) {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR not set");
    let cu_path = PathBuf::from(&manifest_dir).join("src/stub_kernels.cu");
    let ptx_path = PathBuf::from(out_dir).join("stub_kernels.ptx");
    let cu = cu_path.to_str().unwrap();
    let ptx = ptx_path.to_str().unwrap();

    // nvcc -ptx; an explicit host compiler via -ccbin when given.
    let run = |ccbin: Option<&str>| {
        let mut cmd = Command::new("nvcc");
        cmd.args(["-ptx", "-o", ptx, cu, "--gpu-architecture=native", "-O0"]);
        if let Some(cc) = ccbin {
            cmd.args(["-ccbin", cc]);
        }
        cmd.output()
            .expect("nvcc not found (--features cuda needs the CUDA toolkit; build in the symbi-cuda distrobox)")
    };

    // nvcc auto-reads NVCC_CCBIN for its host compiler. on failure, retry once with a
    // PATH g++ IF the configured NVCC_CCBIN is untrustworthy (unset, or a stale absent
    // path) — the shared policy in symbi_xpu::compile_cuda, identical to the runtime AOT
    // path. a real-but-failing NVCC_CCBIN is the user's deliberate choice (no retry); the
    // panic below gives guidance instead.
    let mut output = run(None);
    if !output.status.success() {
        if let Some(retry) = symbi_xpu::compile_cuda::ccbin_retry_candidate() {
            let r = run(Some(&retry.gxx));
            if r.status.success() {
                println!("cargo:warning=symbi: {}; retried with -ccbin {}", retry.reason, retry.gxx);
                output = r;
            }
        }
    }
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        panic!(
            "nvcc failed to compile the CUDA stub PTX (required for --features cuda):\n{stderr}\n\
             nvcc cannot find/use a compatible host compiler here (the host gcc is likely too \
             new for the installed CUDA toolkit). either:\n  \
             - build inside the symbi-cuda distrobox: \
             `distrobox enter symbi-cuda -- cargo run --example mub09 -p symbi --release --features cuda -- ...`\n  \
             - or point nvcc at a supported host compiler: \
             `NVCC_CCBIN=/usr/bin/g++ cargo build --features cuda ...`"
        );
    }
    println!("cargo:rerun-if-changed={}", cu_path.display());
}
