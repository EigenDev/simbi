// =============================================================================
// dispatch_binding_invariants.rs
//
// structural closure gates for the dispatch/binding layer. three invariants the
// Phase 6 door migration established, held in place so a later change cannot
// regress them:
//   1. the baked manifest is the sole read/write classifier — `kernel_field_binds`
//      is read only inside `binding.rs`, so no site hand-walks a manifest;
//   2. resource identity is typed — the lossy `FieldBind::from_path` string
//      classifier stays at the ABI boundary and the AOT name resolver, never in
//      the dispatch layer;
//   3. a resolver decides a binding by the typed variant, never by comparing a
//      rendered `name()` to a string literal.
//
// usage:
//  cargo test -p symbi-substrate --test dispatch_binding_invariants
// =============================================================================

use std::fs;
use std::path::{Path, PathBuf};

fn rust_sources(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            rust_sources(&path, out);
        } else if path.extension().is_some_and(|e| e == "rs") {
            out.push(path);
        }
    }
}

fn crates_root() -> PathBuf {
    // CARGO_MANIFEST_DIR is crates/symbi-substrate; its parent is the crates root.
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crates root")
        .to_path_buf()
}

/// every `.rs` under `crates/<crate>/src`.
fn workspace_sources() -> Vec<PathBuf> {
    let mut files = Vec::new();
    for entry in fs::read_dir(crates_root())
        .expect("read crates root")
        .flatten()
    {
        let src = entry.path().join("src");
        if src.is_dir() {
            rust_sources(&src, &mut files);
        }
    }
    assert!(
        files.len() > 100,
        "expected to scan the whole workspace; found {}",
        files.len()
    );
    files
}

#[test]
fn the_baked_manifest_is_the_sole_binding_classifier() {
    // `kernel_field_binds` reads a kernel's `(FieldBind, is_output)` manifest. it is
    // the input to the one `bind_by_manifest` constructor; a call anywhere else is a
    // hand-rolled classification the migration removed.
    let substrate_src = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut offenders = Vec::new();
    let mut files = Vec::new();
    rust_sources(&substrate_src, &mut files);
    for file in &files {
        if file.ends_with("binding.rs") {
            continue;
        }
        let src = fs::read_to_string(file).expect("read source");
        if src.contains("kernel_field_binds(") {
            offenders.push(file.display().to_string());
        }
    }
    assert!(
        offenders.is_empty(),
        "kernel_field_binds must be read only in binding.rs (the manifest is the sole classifier); found in:\n{}",
        offenders.join("\n")
    );
}

#[test]
fn the_lossy_string_classifier_stays_at_the_boundary() {
    // `FieldBind::from_path` classifies an unrecognized spelling as `Scratch` with no
    // error — the one place a string becomes a resource identity. it belongs to the ABI
    // boundary (symbi-abi) and the AOT name resolver (named_call.rs); using it in the
    // dispatch layer would reintroduce stringly identity.
    let mut offenders = Vec::new();
    for file in workspace_sources() {
        let path = file.display().to_string();
        let allowed = path.contains("/symbi-abi/") || path.ends_with("named_call.rs");
        if allowed {
            continue;
        }
        let src = fs::read_to_string(&file).expect("read source");
        if src.contains("from_path(") {
            offenders.push(path);
        }
    }
    assert!(
        offenders.is_empty(),
        "FieldBind::from_path (stringly identity) must stay at the ABI boundary; found in:\n{}",
        offenders.join("\n")
    );
}

#[test]
fn a_resolver_decides_by_typed_variant_not_a_rendered_name() {
    // a dispatch resolver matches `FieldBind::Ref(..)` / `Scratch(ScratchKey::Free(..))`,
    // never `bind.name() == "literal"` — the spelling is not the identity (a Ct wire and
    // a Free scratch can share one).
    let substrate_src = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut files = Vec::new();
    rust_sources(&substrate_src, &mut files);
    let mut offenders = Vec::new();
    for file in &files {
        let src = fs::read_to_string(file).expect("read source");
        for (n, line) in src.lines().enumerate() {
            if line.contains(".name() == \"") {
                offenders.push(format!("{}:{}", file.display(), n + 1));
            }
        }
    }
    assert!(
        offenders.is_empty(),
        "resource identity must be typed, not a name()-string comparison; found at:\n{}",
        offenders.join("\n")
    );
}
