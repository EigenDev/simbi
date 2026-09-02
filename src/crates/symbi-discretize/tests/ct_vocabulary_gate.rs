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

/// resolution-syntax violations in a substrate binder source. comment text is
/// stripped and each line is whitespace-normalized (so `%3` and `bface[ 0]`
/// match), then a production line is flagged when it
/// - spells a cyclic modulo (`% 3`) outside the braced body of a validated
///   resolution function (`CtEdgeMap::try_new`, `CtEdgeMap::grid_ordered`,
///   `ct_face_curl`),
/// - subscripts a staggered field array with a literal index (`bface[0]`,
///   `efield[1]`, ...), or
/// - names a `Transverse::` role on a line that also carries a literal
///   subscript or a modulo (the single-line restatement).
/// the split-line evasion — the formula on one line, the role match on
/// another — is caught by the first two rules, which need no role mention.
/// the exemption tracks the function's actual braced extent: a `fn` line opens
/// a body, brace balance closes it, and the exemption ends with it (format
/// braces inside string literals come in balanced pairs, so the depth count
/// survives them).
fn ct_resolution_violations(text: &str) -> Vec<String> {
    let allowed_fns = ["try_new", "grid_ordered", "ct_face_curl"];
    let mut fn_name = String::new();
    let mut in_fn = false;
    let mut entered = false;
    let mut depth: i64 = 0;
    let mut hits = Vec::new();
    for (number, raw) in production_lines(text) {
        let line = raw.split("//").next().unwrap_or("");
        let flat: String = line.chars().filter(|c| !c.is_whitespace()).collect();
        if !in_fn {
            if let Some(pos) = line.find("fn ") {
                fn_name = line[pos + 3..]
                    .split(|c: char| !(c.is_alphanumeric() || c == '_'))
                    .next()
                    .unwrap_or("")
                    .to_string();
                in_fn = true;
                entered = false;
                depth = 0;
            }
        }
        if in_fn {
            depth += line.matches('{').count() as i64;
            depth -= line.matches('}').count() as i64;
            entered |= line.contains('{');
        }
        let in_allowed = in_fn && allowed_fns.contains(&fn_name.as_str());
        let literal_stagger = ["bface[", "efield["].iter().any(|f| {
            flat.split(f)
                .skip(1)
                .any(|rest| rest.starts_with(|c: char| c.is_ascii_digit()))
        });
        let stray_modulo = flat.contains("%3") && !in_allowed;
        let inline_role = flat.contains("Transverse::")
            && (flat.contains("%3") || ["[0", "[1", "[2"].iter().any(|p| flat.contains(p)));
        if literal_stagger || stray_modulo || inline_role {
            hits.push(format!("{number}: {}", raw.trim()));
        }
        if in_fn && entered && depth <= 0 {
            in_fn = false;
            fn_name.clear();
        }
    }
    hits
}

#[test]
fn transverse_roles_resolve_through_the_validated_edge_maps() {
    // the substrate binder file: role -> absolute-index resolution lives in the
    // validated maps (the edge descriptor's constructors and the incident-edge
    // accessors), so a binder arm reads a resolved local and the file carries
    // no stray cyclic modulo and no literal staggered-field subscript.
    let here = Path::new(env!("CARGO_MANIFEST_DIR"));
    let path = here
        .parent()
        .unwrap()
        .join("symbi-substrate/src/regimes/mhd_substrate.rs");
    let text = std::fs::read_to_string(&path).expect("mhd_substrate.rs");
    let seen = production_lines(&text)
        .iter()
        .filter(|(_, l)| l.contains("Transverse::"))
        .count();
    assert!(
        seen >= 10,
        "the scan saw only {seen} transverse-role lines; the gate is not seeing the binder seam"
    );
    let hits = ct_resolution_violations(&text);
    assert!(
        hits.is_empty(),
        "inline transverse-role resolution in mhd_substrate.rs (resolution belongs \
         to the validated edge maps):\n{}",
        hits.join("\n")
    );
}

#[test]
fn resolution_gate_flags_split_line_formulas() {
    // the formula on one line and the role match on another is still flagged:
    // the modulo rule fires without a role mention on the same line.
    let split = "fn binder() {\n    let p1 = (dir + 1) % 3;\n    match role {\n        \
                 Transverse::A => &mhd.bface[p1],\n        Transverse::B => &mhd.bface[p2],\n    }\n}\n";
    let hits = ct_resolution_violations(split);
    assert_eq!(hits.len(), 1, "split-line modulo is flagged: {hits:?}");

    let literal = "fn binder() { let f = &mhd.efield[0]; }\n";
    assert_eq!(ct_resolution_violations(literal).len(), 1);

    let lawful = "fn binder() {\n    match role {\n        Transverse::A => &mhd.bface[g1],\n        \
                  Transverse::B => &mhd.efield[slot],\n    }\n}\n";
    assert!(ct_resolution_violations(lawful).is_empty());

    let validated = "fn ct_face_curl() { let plane = [(c + 1) % 3, (c + 2) % 3]; }\n";
    assert!(ct_resolution_violations(validated).is_empty());

    // the exemption ends with the allowlisted body: a modulo in the next
    // function is flagged even with no recognized fn line in between.
    let after_closed = "fn ct_face_curl() {}\n\nfn unrelated() {\n    let p1 = (dir + 1) % 3;\n}\n";
    let hits = ct_resolution_violations(after_closed);
    assert_eq!(
        hits.len(),
        1,
        "modulo after a closed allowlisted body: {hits:?}"
    );

    // a multiline allowlisted signature still covers its body.
    let multiline = "fn ct_face_curl<const D: usize>(\n    dir: usize,\n) -> usize {\n    \
                     let plane = [(c + 1) % 3];\n    plane[0]\n}\n";
    assert!(ct_resolution_violations(multiline).is_empty());

    // whitespace variants of both rules are rejected.
    let tight_modulo = "fn binder() { let p1 = (dir + 1) %3; }\n";
    assert_eq!(ct_resolution_violations(tight_modulo).len(), 1);
    let spaced_subscript = "fn binder() { let f = &mhd.bface[ 0 ]; }\n";
    assert_eq!(ct_resolution_violations(spaced_subscript).len(), 1);
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
