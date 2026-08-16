// =============================================================================
// build.rs
//
// on macos the pyo3 `extension-module` feature leaves libpython unlinked —
// the python symbols are resolved at import time by the host
// interpreter. the mach-o linker rejects the resulting undefined symbols
// unless told to defer them, so pass `-undefined dynamic_lookup` scoped
// narrowly to this crate's cdylib artifact.
// =============================================================================

fn main() {
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos") {
        println!("cargo:rustc-link-arg-cdylib=-undefined");
        println!("cargo:rustc-link-arg-cdylib=dynamic_lookup");
    }
}
