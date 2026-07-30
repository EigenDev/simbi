/// the CUDA toolkit lib directories that EXIST on this machine, searched broadly so a cluster
/// module install (where the toolkit is not at /opt/cuda and the env var is named differently)
/// still resolves libnvrtc. roots come from the common toolkit env vars, an `nvcc` PATH probe,
/// and the two standard install prefixes; each root is tried against the layouts CUDA uses
/// (`lib64`, `lib`, and the `targets/<arch>/lib` split of recent toolkits).
fn cuda_libdirs() -> Vec<String> {
    let mut roots: Vec<String> = [
        "CUDA_PATH",
        "CUDA_HOME",
        "CUDA_ROOT",
        "CUDAToolkit_ROOT",
        "CUDA_INSTALL_PATH",
    ]
    .iter()
    .filter_map(|v| std::env::var(v).ok())
    .collect();

    // `nvcc` on PATH -> toolkit root is two levels up (<root>/bin/nvcc). resolves the common
    // cluster case where a `module load cuda` puts nvcc on PATH but sets a non-standard var.
    if let Ok(out) = std::process::Command::new("which").arg("nvcc").output()
        && out.status.success()
    {
        let p = String::from_utf8_lossy(&out.stdout).trim().to_string();
        if let Some(root) = std::path::Path::new(&p).parent().and_then(|b| b.parent()) {
            roots.push(root.display().to_string());
        }
    }
    roots.push("/opt/cuda".to_string());
    roots.push("/usr/local/cuda".to_string());

    let mut dirs: Vec<String> = Vec::new();
    for r in roots {
        for sub in ["lib64", "lib", "targets/x86_64-linux/lib"] {
            let d = format!("{r}/{sub}");
            if std::path::Path::new(&d).is_dir() && !dirs.contains(&d) {
                dirs.push(d);
            }
        }
    }
    dirs
}

fn main() {
    // re-run when the toolkit roots change, so a relinked build picks up a new lib path
    // (the link-search/rpath below are baked from these at build-script time).
    for v in [
        "CUDA_PATH",
        "CUDA_HOME",
        "CUDA_ROOT",
        "CUDAToolkit_ROOT",
        "CUDA_INSTALL_PATH",
        "ROCM_PATH",
        "ROCM_HOME",
    ] {
        println!("cargo:rerun-if-env-changed={v}");
    }

    if cfg!(feature = "cuda") {
        // link against the CUDA driver API library (libcuda.so / cuda.lib); the
        // runtime API libcudart is a distinct library exposing a different symbol
        // set. libcuda.so is always present on systems with NVIDIA GPU drivers
        // (default linker search path).
        println!("cargo:rustc-link-lib=cuda");

        // libnvrtc.so (the runtime CUDA compiler) lives in the toolkit libdir, outside the
        // default linker search path. add every existing toolkit libdir + bake an rpath so the
        // loader finds it at runtime without LD_LIBRARY_PATH.
        let dirs = cuda_libdirs();
        if dirs.is_empty() {
            println!(
                "cargo:warning=no CUDA toolkit libdir found (set CUDA_HOME or load a cuda \
                 module); -lnvrtc will fail to link"
            );
        }
        for d in &dirs {
            println!("cargo:rustc-link-search=native={d}");
            println!("cargo:rustc-link-arg=-Wl,-rpath,{d}");
        }
        println!("cargo:rustc-link-lib=nvrtc");
    }

    if cfg!(feature = "hip") {
        // link the amd hip runtime (libamdhip64.so) and the runtime compiler (libhiprtc.so).
        // both live under ROCM_PATH/lib (default /opt/rocm/lib); bake an rpath so the loader
        // finds them without LD_LIBRARY_PATH.
        let rocm_root = std::env::var("ROCM_PATH")
            .or_else(|_| std::env::var("ROCM_HOME"))
            .unwrap_or_else(|_| "/opt/rocm".to_string());
        let libdir = format!("{rocm_root}/lib");
        println!("cargo:rustc-link-search=native={libdir}");
        println!("cargo:rustc-link-arg=-Wl,-rpath,{libdir}");
        println!("cargo:rustc-link-lib=amdhip64");
        println!("cargo:rustc-link-lib=hiprtc");
    }
}
