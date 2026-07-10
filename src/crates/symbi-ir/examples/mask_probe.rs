// =============================================================================
// mask_probe.rs
//
// diagnostic: deserialize a kernel's Prepared IR blob (the .ir.json emitted
// next to each generated kernel) and report what the render-time passes do
// with it — whether mask_form applies, and how many selects lazy_select
// converts. names the pass outcome for a kernel whose emitted spelling looks
// wrong.
//
// usage:
//   cargo run -p symbi-ir --example mask_probe <path/to/kernel.ir.json>
// =============================================================================

fn main() {
    let path = std::env::args().nth(1).expect("usage: mask_probe <ir.json>");
    let ir = std::fs::read_to_string(&path).expect("read ir json");
    let mut prepared: symbi_ir::Prepared = serde_json::from_str(&ir).expect("deserialize Prepared");
    println!(
        "kernel: {}  params: {:?}",
        prepared.kernel_name,
        prepared
            .scalarized
            .params
            .iter()
            .map(|p| format!("{}:{:?}", p.name, p.element))
            .collect::<Vec<_>>()
    );
    let lazy = symbi_ir::passes::lazy_select::apply(&mut prepared.scalarized);
    println!("lazy_select conversions: {lazy}");
    let masked = symbi_ir::passes::mask_form::apply(&mut prepared.scalarized);
    println!("mask_form applied: {masked}");
}
