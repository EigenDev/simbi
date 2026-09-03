// =============================================================================
// no_public_boolean_compatibility.rs
//
// structural gate: read/write compatibility between kernel programs is decided
// once, by the effect algebra, and reported as evidence (`Dependence` /
// `ConflictSet` / `Result`). no public function signature across the workspace
// answers "can these run together" with a bare `bool`: a signature that returns
// `bool` and either names a compatibility predicate (`compatible`,
// `can_parallel`, `independent`, `conflicts`, ...) or takes an effect-algebra
// type (`Effects`, `Composition`, `KernelProgram`, `Dependence`) is an offender.
// this scans `pub fn` signatures alone (the cross-crate surface).
//
// usage:
//  cargo test -p symbi-ir --test no_public_boolean_compatibility
// =============================================================================

use std::fs;
use std::path::{Path, PathBuf};

/// name fragments that mark a compatibility predicate.
const PREDICATE_NAMES: &[&str] = &[
    "compatib",
    "can_parallel",
    "can_fuse",
    "can_compose",
    "independent",
    "conflict",
    "commute",
    "hazard",
];

/// effect-algebra types a boolean signature may not reason over.
const EFFECT_TYPES: &[&str] = &["Effects", "Composition", "KernelProgram", "Dependence"];

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
/// `{` or the `;` of a trait declaration.
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

/// the function name of a `pub fn` signature.
fn fn_name(sig: &str) -> &str {
    let after = &sig["pub fn ".len()..];
    let end = after
        .find(|c: char| c == '(' || c == '<' || c.is_whitespace())
        .unwrap_or(after.len());
    &after[..end]
}

/// the signature returns a bare `bool`.
fn returns_bool(sig: &str) -> bool {
    sig.rsplit_once("->")
        .is_some_and(|(_, ret)| ret.trim().trim_end_matches("where").trim() == "bool")
}

/// a boolean signature that names a compatibility predicate or takes an
/// effect-algebra type.
fn is_boolean_compatibility(sig: &str) -> bool {
    if !returns_bool(sig) {
        return false;
    }
    let name = fn_name(sig);
    let params = sig.rsplit_once("->").map(|(head, _)| head).unwrap_or(sig);
    PREDICATE_NAMES.iter().any(|frag| name.contains(frag))
        || EFFECT_TYPES.iter().any(|ty| params.contains(ty))
}

#[test]
fn the_matcher_recognizes_each_offender_shape() {
    // the gate is only as good as its matcher; each shape it exists to catch
    // must fire, and the lawful shapes must pass.
    assert!(is_boolean_compatibility(
        "pub fn compatible(&self, other: &Self) -> bool "
    ));
    assert!(is_boolean_compatibility(
        "pub fn can_parallel(a: &Foo, b: &Foo) -> bool "
    ));
    assert!(is_boolean_compatibility(
        "pub fn is_independent_of(&self, other: &Self) -> bool "
    ));
    assert!(is_boolean_compatibility(
        "pub fn conflicts_with(&self, other: &Self) -> bool "
    ));
    assert!(is_boolean_compatibility(
        "pub fn disjoint(a: &Effects, b: &Effects) -> bool "
    ));
    assert!(is_boolean_compatibility(
        "pub fn lawful(left: &Composition, right: &Composition) -> bool "
    ));
    assert!(!is_boolean_compatibility(
        "pub fn parallel(self, other: impl Into<Composition>) -> Result<Composition, ConflictSet> "
    ));
    assert!(!is_boolean_compatibility(
        "pub fn dependences_into(&self, later: &Effects) -> Vec<Dependence> "
    ));
    assert!(!is_boolean_compatibility(
        "pub fn has_no_outputs(&self) -> bool "
    ));
    assert!(!is_boolean_compatibility(
        "pub fn overlaps(&self, other: &Domain<R>) -> bool "
    ));
}

#[test]
fn no_public_signature_answers_compatibility_with_a_bool() {
    // CARGO_MANIFEST_DIR is crates/symbi-ir; its parent is the crates root.
    let crates_root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crates root")
        .to_path_buf();

    let mut files = Vec::new();
    for entry in fs::read_dir(&crates_root)
        .expect("read crates root")
        .flatten()
    {
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
            if is_boolean_compatibility(sig) {
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
        "compatibility is evidence from the effect algebra (Result / Dependence / ConflictSet), never a bool:\n{}",
        offenders.join("\n")
    );
}
