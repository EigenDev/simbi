// =============================================================================
// no_public_loose_pair.rs
//
// structural gate: an executable kernel travels as the `KernelProgram` owner, so
// no public function signature across the workspace hands out or accepts the
// bare `(GvKernel, KernelWrites)` pair. the private test fixtures and the
// `KernelWrites` type alias are internal and exempt; this scans `pub fn`
// signatures alone (the cross-crate surface).
//
// usage:
//  cargo test -p symbi-ir --test no_public_loose_pair
// =============================================================================

use std::fs;
use std::path::{Path, PathBuf};

const LOOSE_PAIR: &str = "(GvKernel, KernelWrites)";

/// collect every `.rs` file under a crate `src` directory.
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

/// every `pub fn` signature — the text from `pub fn` up to the body-opening
/// `{` or the `;` of a trait declaration. the body is excluded, so a loose pair
/// used inside a function is not mistaken for a public door.
fn public_signatures(src: &str) -> Vec<&str> {
    let mut sigs = Vec::new();
    let mut rest = src;
    while let Some(at) = rest.find("pub fn ") {
        let tail = &rest[at..];
        let end = tail.find('{').unwrap_or(tail.len());
        let semi = tail.find(';').unwrap_or(tail.len());
        sigs.push(&tail[..end.min(semi)]);
        rest = &tail[7..];
    }
    sigs
}

#[test]
fn no_public_signature_exposes_the_loose_kernel_writes_pair() {
    // CARGO_MANIFEST_DIR is crates/symbi-ir; its parent is the crates root.
    let crates_root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crates root")
        .to_path_buf();

    let mut files = Vec::new();
    for entry in fs::read_dir(&crates_root).expect("read crates root").flatten() {
        let src_dir = entry.path().join("src");
        if src_dir.is_dir() {
            rust_sources(&src_dir, &mut files);
        }
    }
    assert!(
        files.len() > 100,
        "expected to scan the whole workspace; found only {} files",
        files.len()
    );

    let mut offenders = Vec::new();
    for file in &files {
        let src = fs::read_to_string(file).expect("read source");
        for sig in public_signatures(&src) {
            if sig.contains(LOOSE_PAIR) {
                offenders.push(format!(
                    "{}: {}",
                    file.display(),
                    sig.split_whitespace().collect::<Vec<_>>().join(" ")
                ));
            }
        }
    }

    assert!(
        offenders.is_empty(),
        "public signatures must take/return KernelProgram, not the loose pair:\n{}",
        offenders.join("\n")
    );
}
