// =============================================================================
// physics_independence.rs
//
// structural gate: physics crates depend on the carrier foundation, and their
// production dependency graphs are free of the compiler. geometry and the
// immersed-body crate compile against `symbi-carrier` alone; a compiler
// dependency may appear only under [dev-dependencies] (tests may inspect
// traced artifacts). the scan reads each crate's Cargo.toml production
// dependency section directly, so a reintroduced `symbi-ir` dependency fails
// here by name.
// =============================================================================

use std::path::Path;

/// the `[dependencies]` section of a manifest: the lines from that header to
/// the next section header.
fn production_dependencies(manifest: &str) -> String {
    let start = manifest
        .find("[dependencies]")
        .expect("manifest declares a [dependencies] section");
    let body = &manifest[start + "[dependencies]".len()..];
    let end = body.find("\n[").unwrap_or(body.len());
    body[..end].to_string()
}

#[test]
fn geometry_and_bodies_depend_on_the_foundation_alone() {
    let crates = Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
    for name in ["symbi-geometry", "symbi-ib"] {
        let manifest = std::fs::read_to_string(crates.join(name).join("Cargo.toml"))
            .expect("physics crate manifest must exist");
        let deps = production_dependencies(&manifest);
        assert!(
            deps.contains("symbi-carrier"),
            "{name} must depend on the carrier foundation"
        );
        assert!(
            !deps.contains("symbi-ir"),
            "{name} carries a production compiler dependency; the carrier \
             algebra is its whole compiler-facing surface"
        );
    }
}
