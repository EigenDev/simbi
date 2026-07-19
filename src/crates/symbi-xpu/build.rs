fn main() {
    // re-run when the toolkit roots change, so a relinked build picks up a new lib path
    // (the link-search/rpath below are baked from these at build-script time).
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-env-changed=ROCM_PATH");
    println!("cargo:rerun-if-env-changed=ROCM_HOME");

    if cfg!(feature = "cuda") {
        // link against the CUDA driver API library (libcuda.so / cuda.lib); the
        // runtime API libcudart is a distinct library exposing a different symbol
        // set. libcuda.so is always present on systems with NVIDIA GPU drivers
        // (default linker search path).
        println!("cargo:rustc-link-lib=cuda");

        // libnvrtc.so (the runtime CUDA compiler) lives in the
        // toolkit libdir, outside the default linker search path — add it from CUDA_PATH /
        // CUDA_HOME (fallback /opt/cuda), and bake an rpath so the loader finds it at
        // runtime without requiring LD_LIBRARY_PATH.
        let cuda_root = std::env::var("CUDA_PATH")
            .or_else(|_| std::env::var("CUDA_HOME"))
            .unwrap_or_else(|_| "/opt/cuda".to_string());
        let libdir = format!("{cuda_root}/lib64");
        println!("cargo:rustc-link-search=native={libdir}");
        println!("cargo:rustc-link-arg=-Wl,-rpath,{libdir}");
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
