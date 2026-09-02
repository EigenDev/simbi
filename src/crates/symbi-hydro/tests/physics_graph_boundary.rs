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

/// the file's production region: everything ahead of the first `#[cfg(test)]`
/// marker. test modules in this crate sit at file tails behind that marker.
fn production_region(text: &str) -> &str {
    match text.find("#[cfg(test)]") {
        Some(cut) => &text[..cut],
        None => text,
    }
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
        for (number, line) in production_region(&text).lines().enumerate() {
            if line.trim_start().starts_with("//") {
                continue;
            }
            for spelling in forbidden {
                assert!(
                    !line.contains(spelling),
                    "graph-representation reference `{spelling}` in production physics: {}:{}: {}",
                    path.display(),
                    number + 1,
                    line.trim()
                );
            }
        }
    }
}
