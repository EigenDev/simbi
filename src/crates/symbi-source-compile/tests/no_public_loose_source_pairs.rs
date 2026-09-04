// =============================================================================
// no_public_loose_source_pairs.rs
//
// structural gate: a lowered source contribution travels as the `AdmittedSources`
// witness the checked contribution door returns (a driven-boundary prescription
// as `BoundaryPrescription`, a census map as `CensusProgram`), so no public
// function signature across the workspace accepts a bare `(String, SourceProgram)`
// or `(&str, &SourceProgram)` pair list, and no public function outside the
// composition crate accepts a raw `SourceSpec` list. accepting one is the only
// way a program could reach an evaluator, a substrate attach or a fused-kernel
// producer without passing a door. the scan covers `pub fn` parameter lists
// alone (the cross-crate surface); the witnesses' own accessors return the
// pairs and are exempt, since nothing public consumes what they return.
//
// the second half names every production door — the rust `user_defined_source`
// constructor, the config `build_user_source*` lowerings, the runtime
// `RuntimeSource` / `with_runtime_source` attach, the AOT fused-kernel producers
// — and asserts each signature carries the admission witness (or, for the rust
// constructor, the declaration the witness is checked against).
//
// usage:
//  cargo test -p symbi-source-compile --test no_public_loose_source_pairs
// =============================================================================

use std::fs;
use std::path::{Path, PathBuf};

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

/// the signature text of every `fn` whose name is `name` in `src`: from the `fn` keyword
/// up to the body-opening `{` or the `;` of a trait declaration, return type included.
fn signatures_named<'a>(src: &'a str, name: &str) -> Vec<&'a str> {
    let needle = format!("fn {name}(");
    let mut found = Vec::new();
    let mut rest = src;
    while let Some(at) = rest.find(&needle) {
        let tail = &rest[at..];
        let end = tail.find('{').unwrap_or(tail.len());
        let semi = tail.find(';').unwrap_or(tail.len());
        found.push(&tail[..end.min(semi)]);
        rest = &tail[needle.len()..];
    }
    found
}

/// every `pub fn` parameter list — the text from `pub fn` up to the return
/// arrow, or to the body-opening `{` / the `;` of a trait declaration when
/// the function returns unit. the return type and the body are excluded, so a
/// witness accessor returning the pairs is not mistaken for a door.
fn public_parameter_lists(src: &str) -> Vec<&str> {
    let mut lists = Vec::new();
    let mut rest = src;
    while let Some(at) = rest.find("pub fn ") {
        let tail = &rest[at..];
        let end = tail.find('{').unwrap_or(tail.len());
        let semi = tail.find(';').unwrap_or(tail.len());
        let signature = &tail[..end.min(semi)];
        let params = signature
            .find("->")
            .map_or(signature, |arrow| &signature[..arrow]);
        lists.push(params);
        rest = &tail[7..];
    }
    lists
}

/// true when `params` mentions a `(String, <path>SourceProgram)` or
/// `(&str, &<path>SourceProgram)` pair under any path spelling.
fn accepts_loose_pair(params: &str) -> bool {
    for head in ["(String, ", "(&str, &"] {
        let mut rest = params;
        while let Some(at) = rest.find(head) {
            let tail = &rest[at + head.len()..];
            let Some(close) = tail.find(')') else {
                break;
            };
            if tail[..close].trim().ends_with("SourceProgram") {
                return true;
            }
            rest = tail;
        }
    }
    false
}

/// true when `params` accepts a `SourceSpec` list under any path spelling: the raw
/// declarative form a producer would have to build and fuse without admitting.
fn accepts_spec_list(params: &str) -> bool {
    params.contains("SourceSpec]") || params.contains("SourceSpec>")
}

fn crates_root() -> PathBuf {
    // CARGO_MANIFEST_DIR is crates/symbi-source-compile; its parent is the crates root.
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crates root")
        .to_path_buf()
}

fn workspace_sources() -> Vec<PathBuf> {
    let mut files = Vec::new();
    for entry in fs::read_dir(crates_root())
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
    files
}

#[test]
fn no_public_signature_accepts_the_loose_source_pair_list() {
    let mut offenders = Vec::new();
    for file in &workspace_sources() {
        let src = fs::read_to_string(file).expect("read source");
        for params in public_parameter_lists(&src) {
            if accepts_loose_pair(params) {
                offenders.push(format!(
                    "{}: {}",
                    file.display(),
                    params.split_whitespace().collect::<Vec<_>>().join(" ")
                ));
            }
        }
    }

    assert!(
        offenders.is_empty(),
        "public signatures must accept AdmittedSources / BoundaryPrescription / CensusProgram, \
         not a loose (String, SourceProgram) or (&str, &SourceProgram) pair list:\n{}",
        offenders.join("\n")
    );
}

#[test]
fn no_public_signature_outside_the_composition_crate_accepts_raw_specs() {
    // the composition crate owns `SourceSpec`: its laws hold every spec to its signature
    // when they compose. a producer elsewhere taking specs would build and fuse them
    // unadmitted, so the raw list stays inside the crate that admits it.
    let own = crates_root().join("symbi-source-compile");
    let mut offenders = Vec::new();
    for file in &workspace_sources() {
        if file.starts_with(&own) {
            continue;
        }
        let src = fs::read_to_string(file).expect("read source");
        for params in public_parameter_lists(&src) {
            if accepts_spec_list(params) {
                offenders.push(format!(
                    "{}: {}",
                    file.display(),
                    params.split_whitespace().collect::<Vec<_>>().join(" ")
                ));
            }
        }
    }
    assert!(
        offenders.is_empty(),
        "producers outside symbi-source-compile must accept AdmittedSources, not raw \
         SourceSpec lists:\n{}",
        offenders.join("\n")
    );
}

#[test]
fn every_production_source_path_carries_the_admission_witness() {
    // (file under crates/, function name, the token its signature must carry). the rust
    // constructor carries the declaration; every lowering returns the witness; every
    // consumer accepts it.
    let doors: &[(&str, &str, &str)] = &[
        (
            "symbi-source-compile/src/source_spec.rs",
            "user_defined_source",
            "UserVocabulary",
        ),
        (
            "symbi-source-compile/src/expr_bridge.rs",
            "build_user_source",
            "AdmittedSources",
        ),
        (
            "symbi-source-compile/src/expr_bridge.rs",
            "build_user_source_with_law",
            "AdmittedSources",
        ),
        (
            "symbi-source-compile/src/expr_bridge.rs",
            "build_user_sources",
            "AdmittedSources",
        ),
        (
            "symbi-source-compile/src/expr_bridge.rs",
            "build_user_sources_with_law",
            "AdmittedSources",
        ),
        (
            "symbi-source-compile/src/source_evaluator.rs",
            "from_built",
            "AdmittedSources",
        ),
        (
            "symbi-substrate/src/regimes/substrate_kernels/runtime_source.rs",
            "new",
            "AdmittedSources",
        ),
        (
            "symbi-substrate/src/regimes/substrate_newton.rs",
            "with_runtime_source",
            "AdmittedSources",
        ),
        (
            "symbi-substrate/src/regimes/substrate_newton.rs",
            "with_fused_runtime_source",
            "AdmittedSources",
        ),
        (
            "symbi-substrate/src/regimes/substrate.rs",
            "with_runtime_source",
            "AdmittedSources",
        ),
        (
            "symbi-substrate/src/regimes/substrate.rs",
            "with_fused_runtime_source",
            "AdmittedSources",
        ),
        (
            "symbi-substrate/src/regimes/substrate_rhd.rs",
            "with_runtime_source",
            "AdmittedSources",
        ),
        (
            "symbi-substrate/src/regimes/substrate_rhd.rs",
            "with_fused_runtime_source",
            "AdmittedSources",
        ),
        (
            "symbi-substrate/src/regimes/substrate_mhd.rs",
            "with_runtime_source",
            "AdmittedSources",
        ),
        (
            "symbi-py/src/lib.rs",
            "attach_runtime_source",
            "AdmittedSources",
        ),
        (
            "symbi-discretize/src/gv/godunov.rs",
            "godunov_stage_gv_with_fused_sources",
            "AdmittedSources",
        ),
        (
            "symbi-discretize/src/gv/godunov.rs",
            "godunov_stage_gv_with_fused_bodies",
            "AdmittedSources",
        ),
        (
            "symbi-discretize/src/gv/godunov.rs",
            "godunov_stage_gv_with_fused_bodies_and_geo_weight",
            "AdmittedSources",
        ),
        (
            "symbi-discretize/src/gv/godunov.rs",
            "source_apply_gv",
            "AdmittedSources",
        ),
        (
            "symbi-discretize/src/gv/godunov.rs",
            "boundary_fill_from_prescription_gv",
            "BoundaryPrescription",
        ),
    ];
    let root = crates_root();
    let mut missing = Vec::new();
    for (file, name, token) in doors {
        let src = fs::read_to_string(root.join(file)).unwrap_or_else(|e| {
            panic!("door file {file} vanished: {e}; update the door table with its new home")
        });
        let signatures = signatures_named(&src, name);
        assert!(
            !signatures.is_empty(),
            "door `{name}` vanished from {file}; update the door table with its new home"
        );
        for signature in signatures {
            if !signature.contains(token) {
                missing.push(format!(
                    "{file}: {}",
                    signature.split_whitespace().collect::<Vec<_>>().join(" ")
                ));
            }
        }
    }
    assert!(
        missing.is_empty(),
        "every production source path carries the admission witness; these do not:\n{}",
        missing.join("\n")
    );
}

#[test]
fn the_scan_recognizes_every_path_spelling_of_the_pair() {
    assert!(accepts_loose_pair(
        "pub fn attach(self, built: Vec<(String, SourceProgram)>, params: Vec<f64>)"
    ));
    assert!(accepts_loose_pair(
        "pub fn from_built(sources: &[(String, crate::source_spec::SourceProgram)])"
    ));
    assert!(accepts_loose_pair(
        "pub fn new(built: Vec<(String, symbi_source_compile::source_spec::SourceProgram)>)"
    ));
    assert!(accepts_loose_pair(
        "pub fn fused(sources: &[(&str, &symbi_source_compile::source_spec::SourceProgram)])"
    ));
    assert!(!accepts_loose_pair(
        "pub fn from_built(sources: &crate::source_effects::AdmittedSources)"
    ));
    assert!(!accepts_loose_pair(
        "pub fn keys(pairs: &[(&str, &str)], fields: &[(&str, &Field<Sc, D, Mem>)])"
    ));
    assert!(accepts_spec_list(
        "pub fn bake(user_sources: &[&symbi_source_compile::source_spec::SourceSpec])"
    ));
    assert!(accepts_spec_list(
        "pub fn with_user(mut self, sources: Vec<SourceSpec>)"
    ));
    assert!(!accepts_spec_list("pub fn bake(sources: &AdmittedSources)"));
    // the return type is outside the scan: a witness accessor hands the pairs out.
    let [params] = public_parameter_lists(
        "pub fn pairs(&self) -> &[(String, SourceProgram)] {\n    &self.0\n}",
    )[..] else {
        panic!("one signature");
    };
    assert!(!accepts_loose_pair(params));
    // the door scan reads the whole signature, return type included.
    let [signature] = signatures_named(
        "pub fn build_user_source(cfg: &SourceConfig) -> Result<AdmittedSources, String> {}",
        "build_user_source",
    )[..] else {
        panic!("one signature");
    };
    assert!(signature.contains("AdmittedSources"));
}
