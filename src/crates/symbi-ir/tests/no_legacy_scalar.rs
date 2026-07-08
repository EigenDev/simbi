// =============================================================================
// no_legacy_scalar.rs — CI gate (Tier 1 chunk 8).
//
// the legacy `symbi_algebra::Scalar` / `symbi_algebra::Selectable` traits were
// deleted in Tier 1 chunk 5 (2026-05-30). this test PERMANENTLY BANS their
// re-introduction by greping every `.rs` file in the workspace's `crates/`
// tree for references and failing the build if any survive.
//
// the new production carrier-generic surface lives in `symbi_ir::algebra`
// (`Scalar`, `Selectable`, `Mask`). every workspace site should `use
// symbi_ir::algebra::{...}` going forward.
//
// allowed exceptions:
//   - this file itself (it contains the literal strings in its grep patterns).
//   - comments (`//` lines, and lines inside `/* */` blocks at single-line resolution).
//   - the symbi-algebra crate's `algebra.rs` historical-note comment.
//
// rationale: every "second source that silently diverges" failure this project
// has paid for (the macro lowering, the discretize Expr DSL, the gen-2 codegen)
// shared one root cause — a legacy surface lingering after a cutover. the gate
// closes the door so it cannot re-open by accident.
// =============================================================================

use std::path::{Path, PathBuf};

const FORBIDDEN: &[&str] = &[
    "symbi_algebra::Scalar",
    "symbi_algebra::Selectable",
];

const SKIP_DIRS: &[&str] = &["target", ".git", "node_modules", "abandoned"];

/// integration tests run from the crate root (where Cargo.toml lives). the
/// workspace's `crates/` tree is one level up.
const CRATES_ROOT: &str = "../";

fn walk_rs(dir: &Path, out: &mut Vec<PathBuf>) {
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return,
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
            if SKIP_DIRS.contains(&name) {
                continue;
            }
            walk_rs(&path, out);
        } else if path.extension().and_then(|e| e.to_str()) == Some("rs") {
            out.push(path);
        }
    }
}

#[test]
fn no_legacy_scalar_imports_in_workspace() {
    let mut rs_files = Vec::new();
    walk_rs(Path::new(CRATES_ROOT), &mut rs_files);

    // the test file itself: skip by relative-path suffix (the walker produces
    // paths anchored at CRATES_ROOT). this match deliberately uses a string
    // CONTAINS check on the file name so the file's own grep patterns above
    // don't trigger a self-match.
    let self_filename = "no_legacy_scalar.rs";

    let mut violations: Vec<String> = Vec::new();
    for path in &rs_files {
        let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
        if name == self_filename {
            continue;
        }
        let content = match std::fs::read_to_string(path) {
            Ok(c) => c,
            Err(_) => continue,
        };
        for (line_idx, line) in content.lines().enumerate() {
            let trimmed = line.trim_start();
            // skip comments (the historical-note comment in symbi-algebra's
            // algebra.rs is the deliberate exclusion).
            if trimmed.starts_with("//") || trimmed.starts_with("///") {
                continue;
            }
            for forbidden in FORBIDDEN {
                if line.contains(forbidden) {
                    violations.push(format!(
                        "{}:{}: forbidden '{forbidden}' — legacy trait re-introduced\n    {}",
                        path.display(),
                        line_idx + 1,
                        line.trim()
                    ));
                }
            }
        }
    }

    if !violations.is_empty() {
        let count = violations.len();
        let report = violations.join("\n");
        assert!(
            false,
            "no-legacy-scalar gate failed — {count} violation(s). use \
             `symbi_ir::algebra::{{Scalar, Selectable}}` instead.\n\n{report}",
        );
    }
}

#[test]
fn gate_self_check_finds_workspace_rs_files() {
    // sanity: the walker DID find files. if the path resolution drifts the
    // gate would silently pass — this asserts it sees a plausible workspace.
    let mut rs_files = Vec::new();
    walk_rs(Path::new(CRATES_ROOT), &mut rs_files);
    assert!(
        rs_files.len() > 50,
        "no-legacy-scalar gate self-check: walker found only {} .rs files \
         under {CRATES_ROOT} — path resolution probably broken",
        rs_files.len()
    );
}
