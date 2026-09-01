// =============================================================================
// one_text_three_carriers.rs
//
// a walkthrough of how the computation graph comes about. the physics below is
// written once, as an ordinary generic rust function over `S: Scalar`. nothing
// about it knows what a graph is.
//
// evaluating that same text at `S = f64` computes a number. evaluating it at
// `S = Gv` records a graph instead, because `Gv`'s arithmetic operators append
// nodes to an ambient trace rather than doing arithmetic. lowering the graph
// gives a flat statement list, and interpreting that list returns the number the
// f64 evaluation already produced.
//
// usage:
//   cargo run -p symbi-ir --example one_text_three_carriers
// =============================================================================

use symbi_ir::algebra::Scalar;
use symbi_ir::{Backend, Cpu, Gv, ScalarExpr, ScalarStmt, emit_cpu, emit_cuda, scalarize, trace};

/// the specific enthalpy of an ideal gas, `h = 1 + gamma/(gamma - 1) p/rho`.
///
/// this is the whole trick. it is a normal function. it has no macro on it, it
/// is not written in a domain-specific language, and it does not mention the IR.
fn specific_enthalpy<S: Scalar>(rho: S, pre: S, gamma: S) -> S {
    S::ONE + gamma / (gamma - S::ONE) * pre / rho
}

/// the sound speed that follows from it, `c^2 = gamma p / (rho h)`.
fn sound_speed<S: Scalar>(rho: S, pre: S, gamma: S) -> S {
    let h = specific_enthalpy(rho, pre, gamma);
    (gamma * pre / (rho * h)).sqrt()
}

/// the lorentz factor from the squared three-speed, `W = 1/sqrt(1 - v^2)`.
fn lorentz<S: Scalar>(vsq: S) -> S {
    (S::ONE / (S::ONE - vsq)).sqrt()
}

/// the conserved energy density a relativistic evolution stores,
/// `tau = rho h W^2 - p - rho W`.
///
/// `W` is consulted three times here. that repetition is the interesting part
/// below, since the lowering is what decides whether it gets computed once.
fn tau<S: Scalar>(rho: S, pre: S, gamma: S, vsq: S) -> S {
    let h = specific_enthalpy(rho, pre, gamma);
    let w = lorentz(vsq);
    rho * h * w * w - pre - rho * w
}

fn render(expr: &ScalarExpr) -> String {
    match expr {
        ScalarExpr::Const(v) => format!("{v:?}"),
        ScalarExpr::Var(name) => name.to_string(),
        ScalarExpr::BinOp(op, lhs, rhs) => {
            format!("{:?}({}, {})", op, render(lhs), render(rhs))
        }
        ScalarExpr::UnaryOp(op, arg) => format!("{:?}({})", op, render(arg)),
        ScalarExpr::MethodCall {
            receiver,
            method,
            args,
        } => {
            let rest: Vec<String> = args.iter().map(render).collect();
            format!("{}.{}({})", render(receiver), method, rest.join(", "))
        }
        other => format!("{other:?}"),
    }
}

fn main() {
    let (rho, pre, gamma) = (1.5_f64, 2.0_f64, 5.0 / 3.0);

    // ---- carrier one: f64. ordinary arithmetic, an ordinary answer. ----------
    let direct = sound_speed(rho, pre, gamma);
    println!("evaluated at S = f64:  cs = {direct}");

    // ---- carrier two: Gv. the same call records a graph. --------------------
    // `Gv::scalar(name)` is a leaf standing for a runtime input. every operator
    // the function applies to it appends a node to the trace opened here.
    let (kernel, out) = trace(|| {
        let rho = Gv::scalar("rho");
        let pre = Gv::scalar("pre");
        let gamma = Gv::scalar("gamma");
        sound_speed(rho, pre, gamma).node()
    });
    println!(
        "traced at  S = Gv:     {} nodes, params {:?}",
        kernel.graph.len(),
        kernel.scalar_params
    );

    // ---- lowering: the graph becomes a flat statement list. -----------------
    // this is where common subexpression elimination has already run, so a value
    // used twice appears as one binding rather than being recomputed.
    let lowered = scalarize(&kernel.graph, out, "sound_speed");
    println!("\nlowered to {} statements:", lowered.body.len());
    for stmt in &lowered.body {
        if let ScalarStmt::Let { name, value, .. } = stmt {
            println!("    let {name} = {}", render(value));
        }
    }
    for (ii, result) in lowered.results.iter().enumerate() {
        println!("    result[{ii}] = {}", render(result));
    }

    // ---- and the graph computes what the f64 path computed. -----------------
    // the interpreter walks the lowered form with the parameters supplied in
    // signature order. agreement here is the property the whole arrangement
    // rests on, since the GPU kernel is emitted from this same lowered form.
    let inputs: Vec<f64> = lowered
        .params
        .iter()
        .map(|p| match p.name.as_str() {
            "rho" => rho,
            "pre" => pre,
            "gamma" => gamma,
            other => panic!("unexpected parameter {other}"),
        })
        .collect();
    let interpreted = Cpu.eval_elemental(&lowered, &inputs)[0];
    println!("\ninterpreted from the graph: cs = {interpreted}");
    assert_eq!(direct, interpreted, "the two carriers disagree");
    println!("the two carriers agree bit for bit.");

    // ---- a second trace, where a shared subexpression has somewhere to go ----
    // nothing repeated in the sound speed, so its lowering folded into a single
    // expression. the relativistic energy density consults the lorentz factor
    // three times, and common subexpression elimination binds it once.
    let vsq = 0.36_f64;
    let (kernel, out) = trace(|| {
        tau(
            Gv::scalar("rho"),
            Gv::scalar("pre"),
            Gv::scalar("gamma"),
            Gv::scalar("vsq"),
        )
        .node()
    });
    let lowered = scalarize(&kernel.graph, out, "tau");
    println!(
        "\ntau traced to {} nodes, lowered to {} bound values:",
        kernel.graph.len(),
        lowered.body.len()
    );
    for stmt in &lowered.body {
        if let ScalarStmt::Let { name, value, .. } = stmt {
            println!("    let {name} = {}", render(value));
        }
    }
    for (ii, result) in lowered.results.iter().enumerate() {
        println!("    result[{ii}] = {}", render(result));
    }
    let inputs: Vec<f64> = lowered
        .params
        .iter()
        .map(|p| match p.name.as_str() {
            "rho" => rho,
            "pre" => pre,
            "gamma" => gamma,
            "vsq" => vsq,
            other => panic!("unexpected parameter {other}"),
        })
        .collect();
    let from_graph = Cpu.eval_elemental(&lowered, &inputs)[0];
    assert_eq!(
        tau(rho, pre, gamma, vsq),
        from_graph,
        "the two carriers disagree"
    );
    println!("\ntau = {from_graph}, again matching the f64 evaluation exactly.");

    // ---- and the same graph renders for either machine -----------------------
    // both emitters read the one lowered form, so the CPU and the GPU cannot
    // hold different opinions about what the physics is.
    let (kernel, out) = trace(|| {
        specific_enthalpy(Gv::scalar("rho"), Gv::scalar("pre"), Gv::scalar("gamma")).node()
    });
    let h = scalarize(&kernel.graph, out, "specific_enthalpy");
    println!("\n---- emitted as CPU rust ----\n{}", emit_cpu(&h));
    println!("---- emitted as CUDA ----\n{}", emit_cuda(&h));
}
