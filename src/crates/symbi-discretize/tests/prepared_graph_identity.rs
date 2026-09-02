// =============================================================================
// prepared_graph_identity.rs
//
// pins the prepared graph identity of a representative kernel. the serialized
// neutral IR blob is the kernel's complete DAG — ops, operand dependencies,
// select structure, writes, field and scalar bindings; the backend renderers
// consume it whole (`render_from_ir`) — so byte equality of blobs is full
// structural graph equality. two gates:
// - the golden comparison: the freshly traced blob equals the checked-in
//   fixture byte-for-byte, across independent test processes and builds, so a
//   graph change in the builder or the lowering (scalarize + cse + lazy-select
//   + buffer assignment) fails here. regenerate the fixture with
//   `SYMBI_BLESS=1 cargo test -p symbi-discretize --test prepared_graph_identity`
//   and account for the diff.
// - the double-trace check: two traces in one process serialize identically.
//
// the rendered CPU text sits downstream of this identity and is
// scheduling-dependent: loop unswitching picks its specialization condition
// among tied param-only booleans by hash-map iteration order, so
// `{name}_generated.rs` may differ between builds of one identical graph while
// every arm value stays bit-identical (specializing `select(c, t, f)` at
// c = true/false is the definition of select, and the dispatcher branches once
// per kernel call). the blob, not the rendered text, is the object to compare
// when proving two builds share a graph.
// =============================================================================

use std::path::Path;
use symbi_discretize::{Coords, kernel_coalesces_layout, wb_ghost_fill_gv};
use symbi_ir::emit::{Precision, Target, TargetConfig};
use symbi_ir::{KernelEmitInputs, prepare, prepared_to_ir};

/// trace the balance-aware ghost fill (a map-select-heavy kernel: per-axis
/// linear/log spacing selects plus boundary-kind selects) and serialize its
/// prepared IR.
fn traced_blob() -> String {
    let (k, writes) = wb_ghost_fill_gv(2, 2, &[0, 1], 2, Coords::Cartesian);
    let inputs = KernelEmitInputs {
        kernel_name: "wb_ghost_fill_2d",
        ndim: 2,
        target: TargetConfig {
            target: Target::Cuda,
            precision: Precision::F64,
        },
        coalesce_layout: kernel_coalesces_layout("wb_ghost_fill_2d"),
        field_inputs: k.field_inputs(),
        scalar_params: k.scalar_params(),
        field_writes: &writes,
        coord_components: k.coord_components(),
        device_preamble: &[],
        tile_spec: None,
    };
    prepared_to_ir(&prepare(k.graph(), &inputs))
}

#[test]
fn prepared_ir_matches_the_checked_in_golden() {
    let golden_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/wb_ghost_fill_2d_prepared_ir.json");
    let blob = traced_blob();
    if std::env::var_os("SYMBI_BLESS").is_some() {
        std::fs::write(&golden_path, &blob).expect("bless: write golden");
        return;
    }
    let golden = std::fs::read_to_string(&golden_path)
        .expect("golden fixture missing; bless with SYMBI_BLESS=1");
    if blob != golden {
        let at = blob
            .bytes()
            .zip(golden.bytes())
            .position(|(a, b)| a != b)
            .unwrap_or(blob.len().min(golden.len()));
        let lo = at.saturating_sub(60);
        panic!(
            "prepared IR diverges from the golden at byte {at} \
             (traced {} bytes, golden {} bytes):\n traced: ...{}\n golden: ...{}",
            blob.len(),
            golden.len(),
            &blob[lo..(at + 60).min(blob.len())],
            &golden[lo..(at + 60).min(golden.len())],
        );
    }
}

#[test]
fn prepared_ir_is_a_deterministic_function_of_the_builder() {
    let a = traced_blob();
    let b = traced_blob();
    assert!(
        a == b,
        "two traces of one builder must serialize to byte-identical prepared IR"
    );
    for key in ["\"bindings\"", "\"scalarized\"", "\"field_writes\""] {
        assert!(a.contains(key), "the blob carries {key}");
    }
}
