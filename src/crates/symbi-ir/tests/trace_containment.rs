// =============================================================================
// trace_containment.rs
//
// structural gates for scoped tracing: kernel construction goes through the
// branded closure API alone. the compile-fail half of the contract (a Gv
// cannot escape its trace, cross-trace arithmetic is rejected) lives in the
// `compile_fail` doctests on `symbi_ir::trace` and `symbi_ir::Gv`; the gates
// here pin the workspace-wide absence of any ambient begin/end protocol.
// =============================================================================

use std::path::{Path, PathBuf};

const GATE_FILE: &str = "trace_containment.rs";

fn rust_sources(dir: &Path, out: &mut Vec<PathBuf>) {
    for entry in std::fs::read_dir(dir).expect("workspace directory must be readable") {
        let path = entry.expect("directory entry must be readable").path();
        let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
        if path.is_dir() {
            if name == "target" || name.starts_with('.') {
                continue;
            }
            rust_sources(&path, out);
        } else if name.ends_with(".rs") && name != GATE_FILE {
            out.push(path);
        }
    }
}

fn workspace_sources() -> Vec<PathBuf> {
    let crates = Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
    let mut sources = Vec::new();
    rust_sources(&crates, &mut sources);
    assert!(
        sources.len() > 100,
        "the scan found only {} files; the gate is not seeing the workspace",
        sources.len()
    );
    sources
}

/// the ambient trace protocol stays deleted: every kernel trace opens and
/// closes through the scoped `trace(|cx| ...)` family, so the identifiers of
/// the paired protocol appear nowhere in the workspace.
#[test]
fn manual_trace_protocol_is_absent_workspace_wide() {
    // spelled via concat so the gate's own source stays clean under its scan
    // discipline even if the self-exclusion is ever dropped.
    let forbidden = [
        concat!("begin", "_trace("),
        concat!("end", "_trace("),
        concat!("end", "_trace_for_domain("),
        concat!("end", "_trace_with("),
        concat!("in_isolated", "_trace("),
    ];
    for path in workspace_sources() {
        let text = std::fs::read_to_string(&path).expect("source file must be readable");
        for spelling in forbidden {
            assert!(
                !text.contains(spelling),
                "manual trace call `{spelling}...)` found in {}",
                path.display()
            );
        }
    }
}

/// the thread-local recording slot is an implementation detail: mutable trace
/// access is granted by the `TraceCx` capability token alone, so the free
/// accessor stays crate-private.
#[test]
fn raw_trace_access_requires_the_capability_token() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let gv = std::fs::read_to_string(root.join("crates/symbi-ir/src/gv.rs"))
        .expect("gv.rs must exist");
    assert!(
        gv.contains("pub(crate) fn with_trace"),
        "the free with_trace accessor must stay crate-private"
    );
    // the top-level (unindented) free-function form; the indented
    // `TraceCx::with_trace` method is the sanctioned door and stays public.
    assert!(
        !gv.contains("\npub fn with_trace"),
        "a public free with_trace would reopen the ambient protocol"
    );
    assert!(
        !gv.contains("pub fn begin_trace"),
        "the paired protocol must stay deleted"
    );
}
