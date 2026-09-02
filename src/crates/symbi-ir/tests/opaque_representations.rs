use std::path::Path;

fn source(relative: &str) -> String {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    std::fs::read_to_string(root.join(relative)).expect("gate source must exist")
}

#[test]
fn kernel_and_source_representations_are_not_public_records() {
    let gv = source("crates/symbi-ir/src/gv.rs");
    assert!(!gv.contains("pub graph: Graph"));
    assert!(!gv.contains("pub field_inputs: Vec<(InputKey, FieldBind)>"));

    let sources = source("crates/symbi-ir/src/source_program.rs");
    assert!(!sources.contains("pub graph: Graph"));
    assert!(!sources.contains("pub params: Vec<String>"));
    assert!(!sources.contains("pub outputs: Vec<NodeId>"));
}

#[test]
fn catch_all_raw_field_namespace_cannot_return() {
    let abi = source("crates/symbi-abi/src/field_ref.rs");
    let forbidden = ["Raw(Box<str>)", "FieldBind::Raw"];
    for spelling in forbidden {
        assert!(
            !abi.contains(spelling),
            "forbidden catch-all field identity: {spelling}"
        );
    }
}
