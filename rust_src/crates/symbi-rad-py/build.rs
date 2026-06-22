// =============================================================================
// build.rs
//
// pyo3 `extension-module` leaves libpython symbols undefined (resolved at import
// by the host interpreter); the mach-o linker rejects that without
// `-undefined dynamic_lookup`, scoped to this crate's cdylib only.
// =============================================================================

fn main() {
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos") {
        println!("cargo:rustc-link-arg-cdylib=-undefined");
        println!("cargo:rustc-link-arg-cdylib=dynamic_lookup");
    }
}
