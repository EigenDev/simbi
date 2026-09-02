// =============================================================================
// physics_graph_boundary.rs
//
// structural gate: physics contains no graph construction. production code in
// this crate speaks the carrier algebra (`S: Scalar`, `Gv`, `TraceCx`) and the
// opaque `SourceProgram`; the graph representation — `Graph`, `NodeId`, the IR
// operation vocabulary, splicing — belongs to the compiler layer. test modules
// may inspect compiler artifacts, so the scan covers each file's region ahead
// of its first `#[cfg(test)]`.
//
// `symbi_expr::op::Op` (the user-expression vocabulary) is a domain input
// language and stays legal; the banned `Op` spellings are the IR's.
// =============================================================================

use std::path::{Path, PathBuf};

fn rust_sources(dir: &Path, out: &mut Vec<PathBuf>) {
    for entry in std::fs::read_dir(dir).expect("src directory must be readable") {
        let path = entry.expect("directory entry must be readable").path();
        if path.is_dir() {
            rust_sources(&path, out);
        } else if path.extension().and_then(|e| e.to_str()) == Some("rs") {
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
            // skip attribute lines through the item head, then the item body:
            // brace-matched for a module/fn, through the semicolon for a
            // `use` or other terse item.
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

#[test]
fn physics_holds_no_graph_construction() {
    let forbidden = [
        "symbi_ir::graph",
        "symbi_ir::Graph",
        "symbi_ir::NodeId",
        "symbi_ir::Op",
        "symbi_ir::ElementWiseOp",
        "symbi_ir::ConstValue",
        "splice_graph",
        "import_subgraph",
        "add_scalar_param",
        "element_wise(",
        "add_const(",
        ": Graph",
        ": NodeId",
        "<NodeId>",
        "&Graph",
        "&mut Graph",
        "Graph::new",
        "ElementWiseOp::",
        "ConstValue::",
    ];
    let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut files = Vec::new();
    rust_sources(&src, &mut files);
    assert!(
        files.len() > 30,
        "the scan found only {} files; the gate is not seeing the crate",
        files.len()
    );
    for path in files {
        let text = std::fs::read_to_string(&path).expect("source file must be readable");
        for (number, line) in production_lines(&text) {
            if line.trim_start().starts_with("//") {
                continue;
            }
            for spelling in forbidden {
                assert!(
                    !line.contains(spelling),
                    "graph-representation reference `{spelling}` in production physics: {}:{}: {}",
                    path.display(),
                    number,
                    line.trim()
                );
            }
        }
    }
}

/// the skipper keeps production code that sits after a test module in view.
#[test]
fn scan_covers_production_after_a_test_module() {
    let fixture = "fn early() {}\n#[cfg(test)]\nmod tests {\n    fn inner() { let x = 1; }\n}\nfn late_production() { /* NodeId would live here */ }\n";
    let kept: Vec<&str> = production_lines(fixture).iter().map(|(_, l)| *l).collect();
    assert!(kept.iter().any(|l| l.contains("fn early")));
    assert!(
        kept.iter().any(|l| l.contains("fn late_production")),
        "production items after a test module must stay in the scan"
    );
    assert!(
        !kept.iter().any(|l| l.contains("fn inner")),
        "test-module bodies stay out of the scan"
    );
}
