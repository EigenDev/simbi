// =============================================================================
// ct_vocabulary_gate.rs
//
// structural gate for the typed CT scratch vocabulary. the reserved wire names
// (`bface_a`, `emf`, `ez`, ...) carry buffer identity across the
// discretize/substrate seam; identity lives in the typed `CtScratchKey` and the
// strings exist only at the ABI renderer. the gate scans the production sources
// of both crates and rejects:
// - any string literal the canonical `CtWireName::parse` recognizes (the
//   vocabulary itself drives the scan, so a grown vocabulary widens the ban);
// - any string-typed runtime binding left inside the CT builder file, so a
//   typo like `emff` cannot slip in as free scratch.
// test items are brace-skipped, so a gate fixture or unit test inside the
// scanned crates stays out of the scan.
// =============================================================================

use std::path::{Path, PathBuf};
use symbi_ir::CtWireName;

fn rust_sources(dir: &Path, out: &mut Vec<PathBuf>) {
    for entry in std::fs::read_dir(dir).expect("scanned source dir must exist") {
        let path = entry.expect("dir entry").path();
        if path.is_dir() {
            rust_sources(&path, out);
        } else if path.extension().is_some_and(|e| e == "rs") {
            out.push(path);
        }
    }
}

/// the file's production lines: every line outside `#[cfg(test)]` items.
/// a `#[cfg(test)]` attribute removes the item that follows it — a braced
/// module (skipped to its matching close brace) or a single semicolon item —
/// so production code positioned after a test module stays in the scan.
fn production_lines(text: &str) -> Vec<(usize, &str)> {
    let lines: Vec<&str> = text.lines().collect();
    let mut kept = Vec::new();
    let mut ii = 0;
    while ii < lines.len() {
        if lines[ii].trim_start().starts_with("#[cfg(test)]") {
            let mut depth: i64 = 0;
            let mut entered = false;
            while ii < lines.len() {
                let line = lines[ii];
                depth += line.matches('{').count() as i64;
                depth -= line.matches('}').count() as i64;
                entered |= line.contains('{');
                let terse_end = !entered && line.trim_end().ends_with(';');
                ii += 1;
                if (entered && depth <= 0) || terse_end {
                    break;
                }
            }
        } else {
            kept.push((ii + 1, lines[ii]));
            ii += 1;
        }
    }
    kept
}

/// every double-quoted literal on a line, ignoring escapes (the scanned
/// sources spell wire names as plain literals).
fn string_literals(line: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut rest = line;
    while let Some(start) = rest.find('"') {
        let tail = &rest[start + 1..];
        match tail.find('"') {
            Some(end) => {
                out.push(tail[..end].to_string());
                rest = &tail[end + 1..];
            }
            None => break,
        }
    }
    out
}

/// the two production trees whose seam the vocabulary crosses.
fn scanned_roots() -> Vec<PathBuf> {
    let here = Path::new(env!("CARGO_MANIFEST_DIR"));
    vec![
        here.join("src"),
        here.parent().unwrap().join("symbi-substrate").join("src"),
    ]
}

#[test]
fn reserved_wire_names_appear_only_at_the_renderer() {
    let mut files = Vec::new();
    for root in scanned_roots() {
        rust_sources(&root, &mut files);
    }
    assert!(
        files.len() > 40,
        "the scan found only {} files; the gate is not seeing both crates",
        files.len()
    );
    // spellings that legitimately live in other namespaces: trace-local SSA
    // keys of FieldRef-typed reads (`rho`/`pre`), the FOFC flag's write key and
    // the gas-FOFC binder (`flag`), and single-token names that double as
    // scalar params or write keys (`b`/`e`/`b0`/`b1`). their runtime identity
    // is protected by the typed doors and the pair scan below; the literal ban
    // covers the distinctive CT spellings.
    let key_namespace = ["rho", "pre", "flag", "b", "e", "b0", "b1"];
    let mut hits = Vec::new();
    for path in &files {
        let text = std::fs::read_to_string(path).expect("source file must be readable");
        for (number, line) in production_lines(&text) {
            for lit in string_literals(line) {
                if CtWireName::parse(&lit).is_some() && !key_namespace.contains(&lit.as_str()) {
                    hits.push(format!("{}:{number}: \"{lit}\"", path.display()));
                }
            }
        }
    }
    assert!(
        hits.is_empty(),
        "reserved CT wire names as raw literals in production sources (identity \
         lives in the typed CtScratchKey; the string exists only at the ABI \
         renderer):\n{}",
        hits.join("\n")
    );
}

#[test]
fn ct_builders_register_fields_through_typed_keys_only() {
    // the CT builder file: every runtime binding is a typed CtScratchKey (or a
    // FieldRef), so an unrecognized spelling is a compile error at the door
    // instead of a silently free scratch name. the pattern below matches a
    // string literal in the runtime-argument position of the field doors.
    let here = Path::new(env!("CARGO_MANIFEST_DIR"));
    let text = std::fs::read_to_string(here.join("src/gv/ct_emf.rs")).expect("ct_emf.rs");
    let doors = [
        "gv_register_field(",
        "gv_field_at(",
        ".field(",
        ".field_shifted(",
    ];
    let mut hits = Vec::new();
    for (number, line) in production_lines(&text) {
        if !doors.iter().any(|d| line.contains(d)) {
            continue;
        }
        // a stringly runtime binding spells two consecutive string-literal
        // arguments: `"key", "runtime"`. typed sites have a typed second slot.
        let lits = string_literals(line);
        if lits.len() >= 2 && line.contains("\", \"") {
            hits.push(format!("ct_emf.rs:{number}: {}", line.trim()));
        }
    }
    assert!(
        hits.is_empty(),
        "string-typed runtime bindings inside the CT builders:\n{}",
        hits.join("\n")
    );
}
