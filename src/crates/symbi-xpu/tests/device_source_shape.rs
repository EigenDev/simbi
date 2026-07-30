// =============================================================================
// device_source_shape.rs
//
// shape invariants the runtime compilers impose on rendered device source.
// nvrtc and hiprtc compile from an in-memory buffer with no filesystem access and
// pre-define the device api themselves, so a rendered kernel must carry no
// `#include`: hiprtc reports the toolkit's own runtime header as "file not found"
// and aborts the compile.
//
// host-only -- renders the neutral IR blob and inspects the text; no gpu required.
//
// usage:
//  cargo test -p symbi-xpu --test device_source_shape
// =============================================================================

use symbi_aot::{ADIABATIC_C2P_1D_IR, ISO_FACE_FLUX_1D_0_IR};
use symbi_ir::emit::{Precision, Target};
use symbi_ir::render_from_ir;

#[test]
fn c_family_device_source_carries_no_include_directive() {
    let kernels = [
        ("adiabatic_c2p_1d", ADIABATIC_C2P_1D_IR),
        ("iso_face_flux_1d_0", ISO_FACE_FLUX_1D_0_IR),
    ];

    for (name, ir) in kernels {
        for target in [Target::Cuda, Target::Hip] {
            let source = render_from_ir(ir, target, Precision::F64).source;
            assert!(
                !source.contains("#include"),
                "{name} rendered for {target:?} carries an #include directive; the \
                 runtime compiler has no filesystem and rejects it:\n{source}"
            );
        }
    }
}
