// =============================================================================
// recovery_boundaries.rs
//
// structural gate for the recovery certification boundary: `Recovered` is
// minted by `judge` alone, `judge` consumes an opaque `RecoveryAudit`, and the
// audit is constructed only by the named recovery-interior predicates inside
// recovery.rs. crate visibility cannot express "only predicates certify", so
// this scan pins it at both levels: outside recovery.rs the mint and the
// audit constructor have no spelling at all, and inside recovery.rs each
// spelling is confined to its owning function's braced extent (mint in
// `judge`, the constructor in the named predicates).
// =============================================================================

use std::path::{Path, PathBuf};

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

/// the file's production lines: every line outside `#[cfg(test)]` items,
/// brace-skipped like the CT vocabulary gate.
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

/// the audit predicates allowed to construct a `RecoveryAudit`.
const AUDIT_FNS: [&str; 5] = [
    "newtonian_prim_audit",
    "newtonian_mhd_prim_audit",
    "isothermal_prim_audit",
    "isothermal_mhd_prim_audit",
    "relativistic_c2p_audit",
];

/// certification-syntax violations inside recovery.rs itself: comment text is
/// stripped, function extents are brace-tracked (a `fn` line opens a body,
/// brace balance closes it), and a line is flagged when it
/// - spells `RecoveryAudit(` outside the named audit predicates (the struct
///   declaration line is the one non-function occurrence), or
/// - spells `::mint(` outside `judge`.
fn certification_violations(text: &str) -> Vec<String> {
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
        let is_decl = flat.starts_with("pub(crate)structRecoveryAudit(");
        if flat.contains("RecoveryAudit(")
            && !is_decl
            && !(in_fn && AUDIT_FNS.contains(&fn_name.as_str()))
        {
            hits.push(format!("{number}: {}", raw.trim()));
        }
        if flat.contains("::mint(") && !(in_fn && fn_name == "judge") {
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
fn certification_is_spelled_only_by_the_named_predicates() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut files = Vec::new();
    rust_sources(&root, &mut files);
    assert!(
        files.len() > 20,
        "the scan found only {} files; the gate is not seeing the crate",
        files.len()
    );
    let mut hits = Vec::new();
    for path in &files {
        let text = std::fs::read_to_string(path).expect("source file must be readable");
        if path.file_name().is_some_and(|n| n == "recovery.rs") {
            // inside recovery.rs the spellings are confined to their owning
            // functions: mint in judge, the audit constructor in the named
            // predicates.
            for hit in certification_violations(&text) {
                hits.push(format!("{}:{hit}", path.display()));
            }
            continue;
        }
        for (number, line) in production_lines(&text) {
            let flat: String = line.chars().filter(|c| !c.is_whitespace()).collect();
            // the mint and the audit constructor live in recovery.rs alone;
            // `judge` calls elsewhere are lawful because the audit argument
            // is opaque and predicate-produced.
            if flat.contains("RecoveryAudit(") || flat.contains("::mint(") {
                hits.push(format!("{}:{number}: {}", path.display(), line.trim()));
            }
        }
    }
    assert!(
        hits.is_empty(),
        "recovery certification spelled outside its owning functions (only the \
         named predicates construct an audit; only judge mints):\n{}",
        hits.join("\n")
    );
}

#[test]
fn certification_gate_flags_rogue_construction() {
    // an audit constructed after an allowlisted predicate's closing brace is
    // caught: the exemption ends with the body.
    let after_closed = "fn relativistic_c2p_audit() {}\n\nfn rogue() {\n    \
                        let a = RecoveryAudit(None);\n}\n";
    let hits = certification_violations(after_closed);
    assert_eq!(
        hits.len(),
        1,
        "rogue audit construction is flagged: {hits:?}"
    );

    // a mint outside judge is caught even inside an audit predicate.
    let rogue_mint = "fn newtonian_prim_audit() {\n    let r = Recovered::mint(x);\n}\n";
    assert_eq!(certification_violations(rogue_mint).len(), 1);

    // the lawful shape passes: the declaration line, the predicates, and
    // judge's mint.
    let lawful = "pub(crate) struct RecoveryAudit(Option<RecoveryIssues>);\n\n\
                  fn newtonian_prim_audit() {\n    RecoveryAudit(audit)\n}\n\n\
                  fn judge() {\n    Ok(Recovered::mint(candidate))\n}\n";
    assert!(certification_violations(lawful).is_empty());
}
