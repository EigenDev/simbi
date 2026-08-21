// =============================================================================
// emit_cf_both.rs
//
// the balance-aware coarse-fine transfer kernels rendered to CUDA and HIP from
// the one graph, so the two device texts can be read against each other. the
// two targets share one emitter, so any divergence localizes to the small
// per-target arms (qualifiers, headers, intrinsic spellings). the cpu path
// executes the same graph through the exec engine and produces no text; a
// cpu-device comparison is a numerical oracle gate, and lives in tests.
//
// the chain walk carries integer index arithmetic (a signed step, `Ge`
// comparisons, selects on I32, an integer negation), which is the part with the
// least shared surface across target arms and therefore the part worth reading
// first.
//
// usage:
//   cargo run -p symbi-discretize --example emit_cf_both -- <out_dir>
//   then, per kernel, the integer lines side by side:
//     diff <(grep -nE 'int|_i[0-9]|\?' <out_dir>/wb_cf_decode.cuda.cpp) \
//          <(grep -nE 'int|_i[0-9]|\?' <out_dir>/wb_cf_decode.hip.cpp)
// =============================================================================

use std::fs;

use symbi_discretize::{GvKernel, wb_cf_decode_gv, wb_cf_lerp_encode_gv};
use symbi_ir::emit::{Precision, Target, TargetConfig};
use symbi_ir::graph::NodeId;
use symbi_ir::{KernelEmitInputs, emit_kernel_from_lowering};

type Writes = Vec<(String, symbi_ir::FieldBind, NodeId)>;

/// render one kernel at one target and return its source.
fn render(name: &str, k: &GvKernel, writes: &Writes, target: Target) -> String {
    assert!(
        !k.graph.has_errors(),
        "{name} graph errors: {:?}",
        k.graph.errors()
    );
    emit_kernel_from_lowering(
        &k.graph,
        &KernelEmitInputs {
            kernel_name: name,
            coalesce_layout: symbi_discretize::kernel_coalesces_layout(name),
            ndim: 3,
            target: TargetConfig {
                target,
                precision: Precision::F64,
            },
            field_inputs: &k.field_inputs,
            scalar_params: &k.scalar_params,
            field_writes: writes,
            coord_components: &k.coord_components,
            device_preamble: &[],
            tile_spec: None,
        },
    )
    .source
}

/// a census of the integer-typed operations a rendering contains, so the two
/// backends can be compared on count before they are compared on text.
fn integer_census(src: &str) -> Vec<(String, usize)> {
    let probes = [
        ("int declarations", "int "),
        ("ternary selects", "?"),
        ("casts to double", "(double)"),
        ("floor division", "floor"),
        ("fmin/fmax", "fmin"),
    ];
    probes
        .iter()
        .map(|(label, needle)| ((*label).to_string(), src.matches(needle).count()))
        .collect()
}

fn main() {
    let out_dir = std::env::args().nth(1).unwrap_or_else(|| ".".to_string());
    fs::create_dir_all(&out_dir).expect("create out dir");

    let (enc, enc_w) = wb_cf_lerp_encode_gv(3, 5, 2);
    let (dec, dec_w) = wb_cf_decode_gv(3, 2);

    for (name, k, w) in [
        ("wb_cf_lerp_encode", &enc, &enc_w),
        ("wb_cf_decode", &dec, &dec_w),
    ] {
        let cuda = render(name, k, w, Target::Cuda);
        let hip = render(name, k, w, Target::Hip);
        let cuda_path = format!("{out_dir}/{name}.cuda.cpp");
        let hip_path = format!("{out_dir}/{name}.hip.cpp");
        fs::write(&cuda_path, &cuda).expect("write cuda");
        fs::write(&hip_path, &hip).expect("write hip");

        println!("\n=== {name}");
        println!("  cuda {} bytes, hip {} bytes", cuda.len(), hip.len());
        for ((label, a), (_, b)) in integer_census(&cuda).iter().zip(integer_census(&hip)) {
            let mark = if *a == b { " " } else { "  <== DIFFERS" };
            println!("  {label:<20} cuda {a:>5}   hip {b:>5}{mark}");
        }
        println!("  wrote {cuda_path}");
        println!("  wrote {hip_path}");
    }
}
