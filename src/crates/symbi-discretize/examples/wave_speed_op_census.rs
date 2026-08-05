// =============================================================================
// wave_speed_op_census.rs
//
// empirical confirmation of the compute-all-paths carrier tax. renders each
// regime's CFL wave_speed_map kernel to CPU (Rust) source and counts the
// expensive ops every cell executes per launch (post-scalarize, post-CSE — i.e.
// exact ground truth). the rmhd quartic's select-blended cubic
// resolvent computes ALL transcendental branches; the count is the proof.
//
// usage: cargo run -p symbi-discretize --release --example wave_speed_op_census
// =============================================================================

use symbi_discretize::{
    Coords, EosArm, GvKernel, Spacetime, Spacing, iso_flux_gv, iso_wave_speed_map_gv, rhd_flux_gv,
    rhd_wave_speed_map_gv, rmhd_c2p_gv, rmhd_flux_gv, rmhd_hllc_flux_gv, rmhd_hlld_flux_gv,
    rmhd_wave_speed_map_gv,
};
use symbi_ir::emit::{Precision, Target, TargetConfig};
use symbi_ir::graph::NodeId;
use symbi_ir::{KernelEmitInputs, emit_kernel_from_lowering};

type Writes = Vec<(String, symbi_ir::FieldBind, NodeId)>;

// the expensive ops — each is a multi-cycle scalar instruction or a libm call.
// textual occurrences in the rendered (CUDA C) body are counted. CSE has already
// run, so every occurrence is a DISTINCT computation the cell performs. the
// op COUNT is backend-independent: same post-CSE scalarized graph drives CPU
// and CUDA alike — only the spelling differs (`asinh(x)` vs `x.asinh()`).
const TRANSCENDENTALS: &[&str] = &[
    "asinh(", "acosh(", "atanh(", "asin(", "acos(", "atan(", "sinh(", "cosh(", "tanh(", "sin(",
    "cos(", "exp(", "log(", "pow(",
];

fn render_rust(name: &str, ndim: u8, k: GvKernel, writes: Writes) -> String {
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
            ndim,
            target: TargetConfig {
                target: Target::Cuda,
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
    desc.source
}

// count occurrences of `needle` whose char immediately before is NOT ascii
// alphabetic — disambiguates `sinh(` inside `asinh(`, `cos(` inside `acos(`, etc.
fn count(src: &str, needle: &str) -> usize {
    let bytes = src.as_bytes();
    let mut n = 0;
    let mut from = 0;
    while let Some(rel) = src[from..].find(needle) {
        let at = from + rel;
        let prev_ok = at == 0 || !bytes[at - 1].is_ascii_alphabetic();
        if prev_ok {
            n += 1;
        }
        from = at + 1;
    }
    n
}

fn census(label: &str, src: &str) {
    let transc: usize = TRANSCENDENTALS.iter().map(|t| count(src, t)).sum();
    let sqrt = count(src, "sqrt(");
    // every divide (incl. Recip's `1.0 / x`) contains ` / ` exactly once.
    let div = src.matches(" / ").count();
    println!(
        "{label:<26} transcendental={transc:<4} sqrt={sqrt:<4} divide={div:<5} lines={}",
        src.lines().count()
    );
    // per-transcendental breakdown for the heavy kernel
    if transc > 12 {
        print!("    breakdown:");
        for t in TRANSCENDENTALS {
            let c = count(src, t);
            if c > 0 {
                print!(" {}={}", t.trim_end_matches('('), c);
            }
        }
        println!();
    }
}

fn main() {
    let cart1: [Spacing; 1] = [Spacing::Uniform];
    let ax1: [usize; 1] = [0];
    let cart3: [Spacing; 3] = [Spacing::Uniform; 3];
    let ax3: [usize; 3] = [0, 1, 2];

    println!("=== per-cell expensive-op census (rendered CPU kernel, post-CSE) ===\n");

    let (k, w) = iso_wave_speed_map_gv(Coords::Cartesian, &cart1, &ax1, 1);
    census("iso  (Newtonian) 1D", &render_rust("iso_ws", 1, k, w));

    let (k, w) = rhd_wave_speed_map_gv(Coords::Cartesian, Spacetime::Minkowski, &cart1, &ax1, 1, EosArm::IdealGamma);
    census("rhd 1D", &render_rust("rhd_ws", 1, k, w));

    let (k, w) = rmhd_wave_speed_map_gv(Coords::Cartesian, &cart1, &ax1, 1);
    census("rmhd 1D (1 axis)", &render_rust("rmhd_ws1", 1, k, w));

    let (k, w) = rmhd_wave_speed_map_gv(Coords::Cartesian, &cart3, &ax3, 3);
    census("rmhd 3D (orszag_tang)", &render_rust("rmhd_ws3", 3, k, w));

    println!("\n=== Riemann flux kernels (per face per axis — the quartic lives HERE) ===\n");

    let (k, w) = iso_flux_gv::<1>(0);
    census("iso  flux 1D", &render_rust("iso_flux", 1, k, w));

    let (k, w) = rhd_flux_gv::<1>(0, EosArm::IdealGamma);
    census("rhd flux 1D", &render_rust("rhd_flux", 1, k, w));

    let (k, w) = rmhd_flux_gv(1, 0, 0);
    census("rmhd flux 1D (HLLE)", &render_rust("rmhd_hlle", 1, k, w));

    let (k, w) = rmhd_hllc_flux_gv(1, 0, 0);
    census("rmhd flux 1D (HLLC)", &render_rust("rmhd_hllc", 1, k, w));

    let (k, w) = rmhd_hlld_flux_gv(1, 0, 0);
    census("rmhd flux 1D (HLLD)", &render_rust("rmhd_hlld", 1, k, w));

    println!("\n=== c2p (cons->prim) — the dominant kernel by time ===\n");

    let (k, w) = rmhd_c2p_gv(100);
    census("rmhd c2p (max_iter=100)", &render_rust("rmhd_c2p", 1, k, w));

    println!(
        "\nthe native single-branch quartic needs ~2 transcendentals per axis (ONE\n\
         resolvent-cubic case is physically taken). everything above that count is\n\
         the select tax: all 4 cubic cases + all 3 outer eq.56/57/58 paths computed\n\
         per face, then blended."
    );
}
