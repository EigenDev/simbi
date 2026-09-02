// =============================================================================
// zero_panic.rs
//
// the algebraic core must contain no `panic!`, `.unwrap()`, or `.expect(` in
// production code. admitted panics live only at I/O / driver boundaries (config parse,
// NVRTC compile, HDF5 write) — those happen in other files. these are the substrate
// constitution and they are zero-panic by invariant.
//
// membership is discovered rather than listed: a file joins the constitution by
// declaring
//   #![deny(clippy::panic, clippy::unwrap_used, clippy::expect_used)]
// which is the same contract `cargo clippy` enforces during development. this test is
// the ci-side complement — it runs under plain `cargo test` with no clippy required.
//
// deriving the set from the contract means a new core file is covered the moment it
// declares one, and a file cannot quietly leave the set: dropping the attribute empties
// the discovered set and trips the precondition below instead of silently reducing
// coverage to nothing.
// =============================================================================

use std::path::{Path, PathBuf};

/// the crate's rust sources, recursively. an unreadable directory contributes nothing
/// and is caught by the discovery precondition rather than by an early return.
fn rust_sources(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            rust_sources(&path, out);
        } else if path.extension().is_some_and(|ext| ext == "rs") {
            out.push(path);
        }
    }
}

/// whether a source declares the zero-panic contract as a file-level deny.
///
/// each `#![deny(..)]` span is examined on its own, so an item-level
/// `#[allow(clippy::unwrap_used)]` — a deliberate opt-out — cannot pull a file into the
/// constitution, and a `#![deny(..)]` of unrelated lints cannot either.
fn declares_the_zero_panic_contract(src: &str) -> bool {
    let mut rest = src;
    while let Some(at) = rest.find("#![deny(") {
        let span = &rest[at..];
        let end = span.find(")]").map_or(span.len(), |e| e + 2);
        let attr = &span[..end];
        if attr.contains("clippy::unwrap_used")
            || attr.contains("clippy::panic")
            || attr.contains("clippy::expect_used")
        {
            return true;
        }
        rest = &span[end..];
    }
    false
}

#[test]
fn zero_panic_in_constitution_files() {
    // integration tests run from the crate root (where Cargo.toml lives), so `src` is
    // relative to that.
    let mut sources = Vec::new();
    rust_sources(Path::new("src"), &mut sources);
    sources.sort();

    // the walk reaching the tree is a premise of everything below: a gate that discovers
    // nothing passes forever while checking nothing. the foundation crate carries the
    // constitution (lib.rs) plus the dual carrier, so fewer than two sources means the
    // walk found the wrong directory or none at all.
    assert!(
        sources.len() >= 2,
        "the src walk found only {} rust file(s) — discovery is broken, so this gate \
         checked nothing",
        sources.len()
    );

    let mut constitution: Vec<(PathBuf, String)> = Vec::new();
    for path in &sources {
        let Ok(content) = std::fs::read_to_string(path) else {
            continue;
        };
        if declares_the_zero_panic_contract(&content) {
            constitution.push((path.clone(), content));
        }
    }

    // the second premise: at least one file still declares the contract. an empty set
    // means the attribute was renamed or dropped everywhere, which is exactly the silent
    // coverage loss this discovery exists to prevent.
    assert!(
        !constitution.is_empty(),
        "no source declares #![deny(clippy::panic, clippy::unwrap_used, \
         clippy::expect_used)] — the constitution is empty and this gate checks nothing"
    );

    let forbidden_patterns = ["panic!", ".unwrap()", ".expect("];
    let mut violations: Vec<String> = Vec::new();

    for (path, content) in &constitution {
        let name = path.display();
        // production code is everything before the first `#[cfg(test)]` guard. the
        // in-file test module is allowed to use panic-style macros (it's test code;
        // asserts are normal). this is a coarse heuristic; if a production block ever
        // lands after a test module, this test misses it — but the file-level deny lints
        // catch that.
        let prod_code = content.split("#[cfg(test)]").next().unwrap_or("");

        for (line_idx, line) in prod_code.lines().enumerate() {
            // skip comments — doc comments and `//` lines.
            let trimmed = line.trim_start();
            if trimmed.starts_with("//") {
                continue;
            }
            for forbidden in &forbidden_patterns {
                if line.contains(forbidden) {
                    violations.push(format!(
                        "{name}:{}: forbidden '{forbidden}' in the algebraic core\n    {}",
                        line_idx + 1,
                        line.trim()
                    ));
                }
            }
        }
    }

    if !violations.is_empty() {
        let scanned: Vec<String> = constitution
            .iter()
            .map(|(p, _)| p.display().to_string())
            .collect();
        panic!(
            "zero-panic gate failed — {} violation(s) across {} constitution file(s) [{}]:\n\n{}",
            violations.len(),
            scanned.len(),
            scanned.join(", "),
            violations.join("\n"),
        );
    }
}
