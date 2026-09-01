// =============================================================================
// kernel_survey.rs
//
// price the substrate's other production kernels the way `wb_cost_model` prices
// the flux: histogram the live lowered ops, weight them by the device
// instruction-cost table, and surface anything structural.
//
// the point is triage, not micro-optimization. a kernel whose cost sits in adds
// and multiplies is doing its job; one whose cost sits in transcendentals, or
// which carries far more nodes than its arithmetic warrants, is worth a look.
//
// usage: cargo run -p symbi-discretize --example kernel_survey
// =============================================================================

use std::collections::BTreeMap;

use symbi_discretize::GvKernel;
use symbi_discretize::coords::{Coords, Spacetime, Spacing};
use symbi_discretize::gv::{adiabatic_c2p_gv, godunov_stage_gv, neumann_ghost_fill_gv};
use symbi_ir::KernelWrites;
use symbi_ir::graph::{NodeId, Op};

fn weight(kind: &str) -> f64 {
    match kind {
        "Pow" => 90.0,
        "Exp" | "Ln" | "Log" => 30.0,
        "Sin" | "Cos" | "Tan" | "Atan2" | "Tanh" => 30.0,
        "Sqrt" => 8.0,
        "Div" => 8.0,
        "Const" | "Param" | "Index" | "Construct" | "Broadcast" => 0.0,
        _ => 1.0,
    }
}

fn price(name: &str, k: GvKernel, writes: KernelWrites) {
    let outs: Vec<NodeId> = writes.iter().map(|write| write.value).collect();
    let live = k.graph().reachable_from(&outs);
    let mut h: BTreeMap<String, usize> = BTreeMap::new();
    let mut cost = 0.0;
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
        cost += weight(&kind);
        *h.entry(kind).or_insert(0) += 1;
    }
    let n: usize = h.values().sum();
    let transcendental: f64 = h
        .iter()
        .filter(|(k, _)| matches!(k.as_str(), "Pow" | "Exp" | "Ln" | "Log" | "Sqrt" | "Div"))
        .map(|(k, &c)| c as f64 * weight(k))
        .sum();
    println!(
        "{name:<34} nodes {n:>5}   cost {cost:>7.0}   expensive-op share {:>4.0}%",
        100.0 * transcendental / cost.max(1.0)
    );
    let mut rows: Vec<_> = h
        .iter()
        .map(|(k, &c)| (c as f64 * weight(k), k.clone(), c))
        .collect();
    rows.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    let top: Vec<String> = rows
        .into_iter()
        .filter(|(c, _, _)| *c > 0.0)
        .take(4)
        .map(|(c, k, n)| format!("{k}x{n}={c:.0}"))
        .collect();
    println!("{:38}{}", "", top.join("  "));
}

fn main() {
    let sp = [Spacing::Uniform, Spacing::Uniform, Spacing::Uniform];
    let (k, w) = godunov_stage_gv(
        Coords::Cartesian,
        Spacetime::Minkowski,
        &sp,
        &[0, 1, 2],
        3,
        5,
        true,
        symbi_discretize::gv::GeoSource::Hydro { inertial: false },
    );
    price("godunov_stage 3d (5 comp)", k, w);
    let (k, w) = adiabatic_c2p_gv::<3>();
    price("adiabatic_c2p 3d", k, w);
    let (k, w) = neumann_ghost_fill_gv(3, 5, true, &sp);
    price("neumann_ghost_fill 3d", k, w);
}
