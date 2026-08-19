// =============================================================================
// emit_wb_hip.rs
//
// emit the plain and well-balanced hllc+ flux kernels (plm, 3d, cartesian) as
// standalone HIP source, for a register and occupancy census on the device
// host. the two graphs differ only in the Balance arm, so the difference in
// compiled resource usage is the well-balanced reconstruction's own footprint.
//
// an arithmetic cost model prices the operations a kernel performs; the
// device's other currency is the values it holds live, which set registers per
// lane and through them waves per SIMD. this census reads that currency
// directly from the compiler.
//
// usage:
//   cargo run -p symbi-discretize --example emit_wb_hip -- <out_dir>
//   then on the device host (MI250X = gfx90a):
//     hipcc -c --offload-arch=gfx90a \
//       -Rpass-analysis=kernel-resource-usage <out_dir>/*.hip.cpp
//   the remarks print VGPRs, ScratchSize [bytes/lane], and Occupancy
//   [waves/SIMD] per kernel; nonzero scratch marks register spills.
// =============================================================================

use std::fs;

use symbi_discretize::coords::{Balance, Coords};
use symbi_discretize::gv::adiabatic_hllc_plus_flux_gv;
use symbi_discretize::{GvKernel, Recon};
use symbi_ir::emit::{Precision, Target, TargetConfig};
use symbi_ir::graph::NodeId;
use symbi_ir::{KernelEmitInputs, emit_kernel_from_lowering};

type Writes = Vec<(String, symbi_ir::FieldBind, NodeId)>;

fn emit_gv(out_dir: &str, name: &str, k: GvKernel, writes: Writes) {
    assert!(
        !k.graph.has_errors(),
        "{name} graph errors: {:?}",
        k.graph.errors()
    );
    let desc = emit_kernel_from_lowering(
        &k.graph,
        &KernelEmitInputs {
            kernel_name: name,
            coalesce_layout: symbi_discretize::kernel_coalesces_layout(name),
            ndim: 3,
            target: TargetConfig {
                target: Target::Hip,
                precision: Precision::F64,
            },
            field_inputs: &k.field_inputs,
            scalar_params: &k.scalar_params,
            field_writes: &writes,
            coord_components: &k.coord_components,
            device_preamble: &[],
            tile_spec: None,
        },
    );
    let path = format!("{out_dir}/{name}.hip.cpp");
    fs::write(&path, &desc.source).unwrap_or_else(|e| panic!("write {path}: {e}"));
    println!("emitted {path}");
}

fn main() {
    let out = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "target/wb_hip".to_string());
    fs::create_dir_all(&out).expect("create out dir");

    for (name, balance) in [
        ("flux_hllc_plus_plm_3d_plain", Balance::Plain),
        ("flux_hllc_plus_plm_3d_wb", Balance::Hydrostatic),
    ] {
        let (k, w) =
            adiabatic_hllc_plus_flux_gv::<3>(0, Recon::Plm, balance, Coords::Cartesian, &[0, 1, 2]);
        emit_gv(&out, name, k, w);
    }
    println!("done -> {out}");
}
