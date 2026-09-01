// =============================================================================
// wb_cost_model.rs
//
// static cost model for the well-balanced flux kernel, evaluated from the IR
// rather than from a device run.
//
// the cost of a kernel is a property of its lowered graph, and tracing costs
// seconds because it is ordinary rust with no code generation. this histograms
// the lowered ops for the plain and balanced arms of the same kernel, weights
// them by a device instruction-cost table, and reports the predicted ratio.
//
// calibrate the table once against a measured throughput ratio, and every
// candidate optimization is then a re-trace rather than a re-bake.
//
// usage: cargo run -p symbi-discretize --example wb_cost_model
// =============================================================================

use std::collections::BTreeMap;

use symbi_discretize::Recon;
use symbi_discretize::coords::{Balance, Coords};
use symbi_discretize::gv::{adiabatic_hllc_flux_gv, adiabatic_hllc_plus_flux_gv};
use symbi_ir::graph::Op;

/// reciprocal-throughput weights for f64 ops on a CDNA-class device, relative to
/// a fused multiply-add. the transcendentals are the software-emulated
/// double-precision routines, which is where the asymmetry lives.
fn weight(kind: &str) -> f64 {
    match kind {
        // a library double-precision pow against the ln/exp composition: hydrostatic.rs
        // measures the composition at 0.67 of it on arm64 and notes the gpu margin is
        // wider, so the general power carries ~1.5x the pair's 60.
        "Pow" => 90.0,
        "Exp" | "Ln" | "Log" => 30.0,
        "Sqrt" | "Rsqrt" => 8.0,
        "Div" => 8.0,
        "Sin" | "Cos" | "Tan" | "Atan2" => 30.0,
        "Abs" | "Neg" | "Min" | "Max" | "Select" => 1.0,
        "Add" | "Sub" | "Mul" => 1.0,
        "Const" | "Param" | "Index" | "Construct" | "Broadcast" => 0.0,
        _ => 1.0,
    }
}

fn histogram(recon: Recon, balance: Balance) -> (BTreeMap<String, usize>, f64) {
    let (k, writes) =
        adiabatic_hllc_plus_flux_gv::<3>(0, recon, balance, Coords::Cartesian, &[0, 1, 2]);
    let outputs: Vec<_> = writes.iter().map(|write| write.value).collect();
    let live = k.graph().reachable_from(&outputs);
    let mut h: BTreeMap<String, usize> = BTreeMap::new();
    let mut cost = 0.0;
    for (id, node, _) in k.graph().iter() {
        if !live.contains(&id) {
            continue;
        }
        let kind = match &node.op {
            Op::ElementWise(op, _) => format!("{op:?}"),
            Op::Select(..) => "Select".to_string(),
            other => {
                let s = format!("{other:?}");
                s.split(['(', ' ', '{']).next().unwrap_or("?").to_string()
            }
        };
        cost += weight(&kind);
        *h.entry(kind).or_insert(0) += 1;
    }
    (h, cost)
}

fn ablate() {
    // attribute the baseline divisions: plain hllc against hllc+, both unbalanced.
    let count = |wb: bool, plus: bool| -> BTreeMap<String, usize> {
        let bal = if wb {
            Balance::Hydrostatic
        } else {
            Balance::Plain
        };
        let (k, w) = if plus {
            adiabatic_hllc_plus_flux_gv::<3>(0, Recon::Plm, bal, Coords::Cartesian, &[0, 1, 2])
        } else {
            {
                let _ = bal;
                adiabatic_hllc_flux_gv::<3>(0, Recon::Plm)
            }
        };
        let outs: Vec<_> = w.iter().map(|write| write.value).collect();
        let live = k.graph().reachable_from(&outs);
        let mut h = BTreeMap::new();
        for (id, node, _) in k.graph().iter() {
            if !live.contains(&id) {
                continue;
            }
            let kind = match &node.op {
                Op::ElementWise(op, _) => format!("{op:?}"),
                Op::Select(..) => "Select".to_string(),
                other => format!("{other:?}")
                    .split(['(', ' ', '{'])
                    .next()
                    .unwrap_or("?")
                    .to_string(),
            };
            *h.entry(kind).or_insert(0) += 1;
        }
        h
    };
    let hllc = count(false, false);
    let plus = count(false, true);
    println!("\n=== baseline division attribution (3d plm, unbalanced)");
    for op in ["Div", "Sqrt", "Pow", "Exp", "Log"] {
        let a = hllc.get(op).copied().unwrap_or(0);
        let b = plus.get(op).copied().unwrap_or(0);
        println!(
            "   {op:<5} plain hllc {a:>4}   hllc+ {b:>4}   the shear/APC terms add {:>4}",
            b as i64 - a as i64
        );
    }
}

fn main() {
    ablate();
    emit_probe();
    for recon in [Recon::Plm, Recon::Ppm] {
        let (plain, c_plain) = histogram(recon, Balance::Plain);
        let (wb, c_wb) = histogram(recon, Balance::Hydrostatic);
        let n_plain: usize = plain.values().sum();
        let n_wb: usize = wb.values().sum();
        println!("\n===== {recon:?}  (3d, cartesian, hllc+, dir 0)");
        println!(
            "  live nodes : plain {n_plain:6}   wb {n_wb:6}   ratio {:.2}x",
            n_wb as f64 / n_plain as f64
        );
        println!(
            "  model cost : plain {c_plain:6.0}   wb {c_wb:6.0}   ratio {:.2}x",
            c_wb / c_plain
        );
        println!(
            "  {:<12} {:>8} {:>8} {:>8}  {:>10}",
            "op", "plain", "wb", "delta", "cost delta"
        );
        let mut keys: Vec<_> = plain.keys().chain(wb.keys()).cloned().collect();
        keys.sort();
        keys.dedup();
        let mut rows: Vec<_> = keys
            .into_iter()
            .map(|k| {
                let a = *plain.get(&k).unwrap_or(&0);
                let b = *wb.get(&k).unwrap_or(&0);
                let dc = (b as f64 - a as f64) * weight(&k);
                (dc, k, a, b)
            })
            .collect();
        rows.sort_by(|x, y| y.0.abs().partial_cmp(&x.0.abs()).unwrap());
        // the baseline arm's own cost: what every run pays, wb or not.
        let mut plain_rows: Vec<_> = plain
            .iter()
            .map(|(k, &n)| (n as f64 * weight(k), k.clone(), n))
            .collect();
        plain_rows.sort_by(|x, y| y.0.partial_cmp(&x.0).unwrap());
        println!("  -- baseline arm, top cost centers --");
        for (c, k, n) in plain_rows.into_iter().take(6) {
            if c <= 0.0 {
                continue;
            }
            println!(
                "     {k:<12} n={n:<6} cost={c:>8.0}  ({:.0}% of plain)",
                100.0 * c / c_plain
            );
        }
        for (dc, k, a, b) in rows.into_iter().take(10) {
            if a == b {
                continue;
            }
            println!(
                "  {k:<12} {a:>8} {b:>8} {:>8} {dc:>10.0}",
                b as i64 - a as i64
            );
        }
    }
}

// ---- do the expensive arms survive as real branches, or get flattened? ----
// a node inside a branch that a launch does not take costs nothing at runtime, and the
// histogram above cannot see that. this reads the emitted source to tell the two apart.
#[allow(dead_code)]
fn emit_probe() {
    use symbi_ir::emit::{Precision, Target, TargetConfig};
    use symbi_ir::{KernelEmitInputs, emit_kernel_from_lowering};
    let (k, w) = adiabatic_hllc_flux_gv::<3>(0, Recon::Plm);
    let desc = emit_kernel_from_lowering(
        k.graph(),
        &KernelEmitInputs {
            kernel_name: "plain_probe",
            coalesce_layout: false,
            ndim: 3,
            target: TargetConfig {
                target: Target::Hip,
                precision: Precision::F64,
            },
            field_inputs: k.field_inputs(),
            scalar_params: k.scalar_params(),
            field_writes: &w,
            coord_components: k.coord_components(),
            device_preamble: &[],
            tile_spec: None,
        },
    );
    let src = desc.source;
    // divisions by a literal: a power-of-two divisor is an exact multiply by its
    // reciprocal, so folding it changes no bit and removes an 8-weight op.
    let mut pow2 = 0usize;
    let mut other_div = 0usize;
    for seg in src.split('/').skip(1) {
        let t = seg.trim_start();
        let lit: String = t
            .chars()
            .take_while(|c| c.is_ascii_digit() || *c == '.')
            .collect();
        if lit.is_empty() {
            continue;
        }
        match lit.parse::<f64>() {
            Ok(v) if v > 0.0 && (v.log2().fract() == 0.0) => pow2 += 1,
            Ok(_) => other_div += 1,
            _ => {}
        }
    }
    println!("   divisions by a power-of-two literal : {pow2}");
    println!("   divisions by another literal        : {other_div}");
    let pows = src.matches("pow(").count();
    let ifs = src.matches("if (").count();
    println!("\n=== emitted HIP for the plain HLLC PLM kernel");
    println!("   pow( occurrences : {pows}");
    println!("   if (  occurrences : {ifs}");
    for (i, line) in src.lines().enumerate() {
        if line.contains("pow(") {
            let ctx: Vec<&str> = src.lines().skip(i.saturating_sub(3)).take(4).collect();
            println!("   --- context around a pow:");
            for c in ctx {
                println!("     {}", &c[..c.len().min(110)]);
            }
            break;
        }
    }
}
