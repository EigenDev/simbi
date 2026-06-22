fn main() {
    if cfg!(feature = "cuda") {
        // link against the CUDA driver API library (libcuda.so / cuda.lib) — the
        // driver API, NOT the runtime API (libcudart). libcuda.so is always present
        // on systems with NVIDIA GPU drivers (default linker search path).
        println!("cargo:rustc-link-lib=cuda");

        // libnvrtc.so (the runtime CUDA compiler, docs/design/15 §1) lives in the
        // toolkit libdir, not the default search path — add it from CUDA_PATH /
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
}
