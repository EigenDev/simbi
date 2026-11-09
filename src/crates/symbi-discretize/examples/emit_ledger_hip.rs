// =============================================================================
// emit_ledger_hip.rs
//
// the conservation receipt kernels of one wall on the spherical (r, theta) plane with the
// azimuthal velocity, rendered as HIP source from the same traces the bake compiles: each
// wall channel in its own kernel and the wall's batch, which evaluates the same channels into
// the slots lb_0..lb_4 in the order mass, inflow, energy, angular momentum, potential. the
// device preamble is the bake's, so the source is the source the device compiles.
//
// usage:
//  cargo run -p symbi-discretize --example emit_ledger_hip -- <out_dir> [axis]
// =============================================================================

use std::fs;

use symbi_discretize::coords::Coords;
use symbi_discretize::{LedgerBatch, LedgerChannel, kernel_device_preamble, ledger_batch_gv, ledger_gv};
use symbi_ir::emit::{Precision, Target, TargetConfig};
use symbi_ir::{KernelEmitInputs, KernelProgram, emit_kernel_from_lowering};

fn emit(out_dir: &str, name: &str, program: KernelProgram) {
    let k = program.kernel();
    let writes = program.writes();
    assert!(!k.graph().has_errors(), "{name} graph errors: {:?}", k.graph().errors());
    let preamble = kernel_device_preamble(name);
    let desc = emit_kernel_from_lowering(
        k.graph(),
        &KernelEmitInputs {
            kernel_name: name,
            coalesce_layout: symbi_discretize::kernel_coalesces_layout(name),
            ndim: 2,
            target: TargetConfig { target: Target::Hip, precision: Precision::F64 },
            field_inputs: k.field_inputs(),
            scalar_params: k.scalar_params(),
            field_writes: &writes,
            coord_components: k.coord_components(),
            device_preamble: &preamble,
            tile_spec: None,
        },
    );
    let path = format!("{out_dir}/{name}.hip.cpp");
    fs::write(&path, &desc.source).unwrap_or_else(|e| panic!("write {path}: {e}"));
    println!("emitted {path}");
}

fn main() {
    let out = std::env::args().nth(1).expect("usage: emit_ledger_hip <out_dir> [axis]");
    let axis: usize = std::env::args().nth(2).map_or(1, |a| a.parse().expect("axis 0 or 1"));
    fs::create_dir_all(&out).expect("the output directory");
    let axes = [0usize, 1];
    for channel in [
        LedgerChannel::WallMass { axis },
        LedgerChannel::WallMassIn { axis },
        LedgerChannel::WallEnergy { axis },
        LedgerChannel::WallAngular { axis },
        LedgerChannel::WallPotential { axis },
    ] {
        emit(&out, &format!("{}_sph_swirl_2d", channel.stem()), ledger_gv(Coords::Spherical, &axes, channel));
    }
    let batch = LedgerBatch::Wall { axis, body: true };
    emit(&out, &format!("{}_sph_swirl_2d", batch.stem()), ledger_batch_gv(Coords::Spherical, &axes, batch));
}
