// =============================================================================
// physics_independence.rs
//
// structural gate: physics crates depend on the carrier foundation, and their
// production dependency graphs are free of the compiler stack. geometry, the
// immersed-body crate, and hydro compile against `symbi-carrier` alone
// (compiler dependencies may appear only under [dev-dependencies] — tests may
// inspect traced artifacts), and the carrier constitution is imported from
// this crate directly rather than through deleted compiler re-export paths.
// the scans read manifests and sources by spelling, so a reintroduced
// dependency or facade fails here by name.
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
fn physics_depends_on_the_foundation_alone() {
    let crates = Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
    for name in ["symbi-geometry", "symbi-ib", "symbi-hydro"] {
        let manifest = std::fs::read_to_string(crates.join(name).join("Cargo.toml"))
            .expect("physics crate manifest must exist");
        let deps = production_dependencies(&manifest);
        assert!(
            deps.contains("symbi-carrier"),
            "{name} must depend on the carrier foundation"
        );
        for compiler in ["symbi-ir", "symbi-source-compile", "symbi-jit", "symbi-expr"] {
            assert!(
                !deps.contains(compiler),
                "{name} carries a production dependency on {compiler}; the carrier \
                 algebra is its whole compiler-facing surface"
            );
        }
    }
}

/// the carrier constitution is imported from this crate directly: the
/// compiler re-export paths that once carried it are deleted, so a
/// reintroduced facade fails here by spelling.
#[test]
fn the_carrier_facade_stays_deleted() {
    fn rust_sources(dir: &Path, out: &mut Vec<std::path::PathBuf>) {
        for entry in std::fs::read_dir(dir).expect("workspace directory must be readable") {
            let path = entry.expect("directory entry must be readable").path();
            let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
            if path.is_dir() {
                if name == "target" || name.starts_with('.') {
                    continue;
                }
                rust_sources(&path, out);
            } else if name.ends_with(".rs") {
                out.push(path);
            }
        }
    }
    let crates = Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
    let mut sources = Vec::new();
    rust_sources(&crates, &mut sources);
    assert!(
        sources.len() > 100,
        "the scan found only {} files; the gate is not seeing the workspace",
        sources.len()
    );
    let forbidden = [concat!("symbi_ir::", "algebra"), concat!("symbi_ir::", "dual")];
    for path in sources {
        let text = std::fs::read_to_string(&path).expect("source file must be readable");
        for spelling in forbidden {
            assert!(
                !text.contains(spelling),
                "carrier facade path `{spelling}` found in {}",
                path.display()
            );
        }
    }
}
