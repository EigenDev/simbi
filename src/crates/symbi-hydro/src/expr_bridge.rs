// =============================================================================
// expr_bridge.rs
//
// the bridge: interpret a `symbi-expr` DAG (a parsed user script) in the
// carrier algebra. tracing the interpretation produces a `SourceProgram`
// (traced expression + named params + outputs), structurally identical to a
// built-in `SourceSpec`'s output, so it rides the existing splice/fuse path
// unchanged: `TraceCx::splice_source` fuses it into a godunov / ghost-fill
// kernel, which then codegens (CPU + CUDA) or interprets. this collapses the
// two parallel expression engines (`symbi-expr`'s register-VM and the codegen
// substrate) into one path that codegens directly, so every cell runs
// compiled code.
//
// the leaf convention matches the source vocabulary so a user expression fuses like any
// other source: VariableX{1,2,3} -> `x_0/x_1/x_2` (the cell position, bound to the
// centroid at splice), VariableT -> `t` (time), Parameter(i) -> `p{i}` (runtime scalar).
//
// the carrier algebra enforces traceability: a user `IF_THEN_ELSE` becomes a
// mask-guided `select`, comparisons produce a mask consumed by conditionals
// and logicals, and ops outside the carrier set (`Sgn`, `Mod`) are rejected
// at bridge time, so every accepted expression compiles faithfully.
//
// usage:
//   let nodes = dag.nodes().to_vec();
//   let built = lower_dag_to_program(&nodes, &[root])?;   // -> SourceProgram
// =============================================================================

use symbi_expr::dag::{Node, Payload};
use symbi_expr::op::Op;
use symbi_ir::algebra::Scalar as _;
use symbi_ir::{Gv, GvMask, SourceProgram, TraceCx};
use symbi_algebra::algebra::Numeric as _;

/// a `symbi-expr` op falls outside the `symbi-ir` vocabulary, or the DAG is malformed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BridgeError {
    /// op lies outside the carrier-traceable `symbi-ir` primitives (`Sgn`, `Mod`). rejected
    /// at bridge time — the user gets a loud error, so every emitted kernel matches the script.
    UnsupportedOp(Op),
    /// a node references a child index outside the already-defined prefix (forward ref / out of
    /// range). well-formed DAGs are topologically ordered, so this means malformed input.
    BadChild { node: usize, child: usize },
    /// payload shape disagrees with the op's arity (malformed input).
    BadPayload { node: usize, op: Op },
    /// an operand's kind disagrees with the op: arithmetic consumes scalars,
    /// logicals consume masks, and a conditional selects scalars on a mask.
    TypeMismatch { node: usize, op: Op },
}

/// the leaf-param name for a `symbi-expr` variable op, matching the source vocabulary
/// (`source_params::x` + `t`). every other op returns `None`.
fn variable_name(op: Op) -> Option<&'static str> {
    match op {
        Op::VariableX1 => Some("x_0"),
        Op::VariableX2 => Some("x_1"),
        Op::VariableX3 => Some("x_2"),
        Op::VariableT => Some("t"),
        // per-cell fluid state: the carrier binds these to field reads (rho/vel from the SSP
        // stage snapshot `u_stage`, pre from `prim.pre`); the CPU `resolve_runtime_param` mirrors them.
        Op::VariableRho => Some("rho"),
        Op::VariableVel1 => Some("vel_0"),
        Op::VariableVel2 => Some("vel_1"),
        Op::VariableVel3 => Some("vel_2"),
        Op::VariablePressure => Some("pre"),
        // the cell's lab-frame volume measure. bound to the finite-volume cell volume the
        // update itself uses (the reciprocal of the in-kernel `inv_volume`), so an extensive
        // sum weighted by it is correct on curvilinear grids.
        Op::VariableCellVolume => Some("dv"),
        _ => None,
    }
}

/// a walked expression value: a scalar carrier value, or the mask a
/// comparison / logical produces. the split is what makes ill-typed user
/// expressions (arithmetic on a comparison, a logical on a scalar) loud
/// bridge-time errors instead of malformed kernels.
enum ExprValue<'t> {
    Scalar(Gv<'t>),
    Mask(GvMask<'t>),
}

impl<'t> ExprValue<'t> {
    fn scalar(&self, node: usize, op: Op) -> Result<Gv<'t>, BridgeError> {
        match self {
            ExprValue::Scalar(v) => Ok(*v),
            ExprValue::Mask(_) => Err(BridgeError::TypeMismatch { node, op }),
        }
    }
    fn mask(&self, node: usize, op: Op) -> Result<GvMask<'t>, BridgeError> {
        match self {
            ExprValue::Mask(m) => Ok(*m),
            ExprValue::Scalar(_) => Err(BridgeError::TypeMismatch { node, op }),
        }
    }
    /// the numeric reading of a value at an output slot: a scalar passes
    /// through; a mask materializes as its 0/1 indicator (the region-mask
    /// convention — chi multiplies the masked outputs).
    fn materialize(&self) -> Gv<'t> {
        match self {
            ExprValue::Scalar(v) => *v,
            ExprValue::Mask(m) => Gv::select(*m, Gv::from_f64(1.0), Gv::from_f64(0.0)),
        }
    }
}

/// fetch a child value by DAG index, rejecting forward refs.
fn child<'a, 't>(
    map: &'a [ExprValue<'t>],
    node: usize,
    c: usize,
) -> Result<&'a ExprValue<'t>, BridgeError> {
    map.get(c).ok_or(BridgeError::BadChild { node, child: c })
}

/// interpret the DAG in the carrier algebra, one value per node in order.
/// leaves become scalar-param leaves of the active trace (deduped by name,
/// first-seen order — the params contract).
fn eval_dag<'t>(cx: TraceCx<'t>, nodes: &[Node]) -> Result<Vec<ExprValue<'t>>, BridgeError> {
    use Op::*;
    let mut map: Vec<ExprValue<'t>> = Vec::with_capacity(nodes.len());
    for (i, node) in nodes.iter().enumerate() {
        let op = node.op;
        let bad = || BridgeError::BadPayload { node: i, op };
        let unary = |map: &[ExprValue<'t>]| -> Result<Gv<'t>, BridgeError> {
            match node.payload {
                Payload::Unary(c) => child(map, i, c)?.scalar(i, op),
                _ => Err(bad()),
            }
        };
        let binary = |map: &[ExprValue<'t>]| -> Result<(Gv<'t>, Gv<'t>), BridgeError> {
            match node.payload {
                Payload::Binary(l, r) => Ok((
                    child(map, i, l)?.scalar(i, op)?,
                    child(map, i, r)?.scalar(i, op)?,
                )),
                _ => Err(bad()),
            }
        };
        let value = match op {
            Constant => match node.payload {
                Payload::Value(v) => ExprValue::Scalar(cx.lit(v)),
                _ => return Err(bad()),
            },
            VariableX1 | VariableX2 | VariableX3 | VariableT | VariableRho | VariableVel1
            | VariableVel2 | VariableVel3 | VariablePressure | VariableCellVolume => {
                let name = variable_name(op).expect("variable op has a name");
                ExprValue::Scalar(cx.scalar(name))
            }
            Parameter => match node.payload {
                Payload::ParamIdx(idx) => ExprValue::Scalar(cx.scalar(&format!("p{idx}"))),
                _ => return Err(bad()),
            },
            IfThenElse => match node.payload {
                Payload::Ternary(c, t, e) => {
                    let m = child(&map, i, c)?.mask(i, op)?;
                    let t = child(&map, i, t)?.scalar(i, op)?;
                    let e = child(&map, i, e)?.scalar(i, op)?;
                    ExprValue::Scalar(Gv::select(m, t, e))
                }
                _ => return Err(bad()),
            },
            Add => { let (a, b) = binary(&map)?; ExprValue::Scalar(a + b) }
            Sub => { let (a, b) = binary(&map)?; ExprValue::Scalar(a - b) }
            Mul => { let (a, b) = binary(&map)?; ExprValue::Scalar(a * b) }
            Div => { let (a, b) = binary(&map)?; ExprValue::Scalar(a / b) }
            Pow => { let (a, b) = binary(&map)?; ExprValue::Scalar(a.powf(b)) }
            Min => { let (a, b) = binary(&map)?; ExprValue::Scalar(a.min(b)) }
            Max => { let (a, b) = binary(&map)?; ExprValue::Scalar(a.max(b)) }
            Atan2 => { let (a, b) = binary(&map)?; ExprValue::Scalar(a.atan2(b)) }
            Neg => ExprValue::Scalar(-unary(&map)?),
            Abs => ExprValue::Scalar(unary(&map)?.abs()),
            Sqrt => ExprValue::Scalar(unary(&map)?.sqrt()),
            Ceil => ExprValue::Scalar(unary(&map)?.ceil()),
            Floor => ExprValue::Scalar(unary(&map)?.floor()),
            Log => ExprValue::Scalar(unary(&map)?.ln()),
            Log10 => ExprValue::Scalar(unary(&map)?.log10()),
            Exp => ExprValue::Scalar(unary(&map)?.exp()),
            Sin => ExprValue::Scalar(unary(&map)?.sin()),
            Cos => ExprValue::Scalar(unary(&map)?.cos()),
            Tan => ExprValue::Scalar(unary(&map)?.tan()),
            Asin => ExprValue::Scalar(unary(&map)?.asin()),
            Acos => ExprValue::Scalar(unary(&map)?.acos()),
            Atan => ExprValue::Scalar(unary(&map)?.atan()),
            Sinh => ExprValue::Scalar(unary(&map)?.sinh()),
            Cosh => ExprValue::Scalar(unary(&map)?.cosh()),
            Tanh => ExprValue::Scalar(unary(&map)?.tanh()),
            Asinh => ExprValue::Scalar(unary(&map)?.asinh()),
            Acosh => ExprValue::Scalar(unary(&map)?.acosh()),
            Atanh => ExprValue::Scalar(unary(&map)?.atanh()),
            Lt => { let (a, b) = binary(&map)?; ExprValue::Mask(a.cmp_lt(b)) }
            Gt => { let (a, b) = binary(&map)?; ExprValue::Mask(a.cmp_gt(b)) }
            Le => { let (a, b) = binary(&map)?; ExprValue::Mask(a.cmp_le(b)) }
            Ge => { let (a, b) = binary(&map)?; ExprValue::Mask(a.cmp_ge(b)) }
            Eq => { let (a, b) = binary(&map)?; ExprValue::Mask(a.cmp_eq(b)) }
            And | Or => match node.payload {
                Payload::Binary(l, r) => {
                    let a = child(&map, i, l)?.mask(i, op)?;
                    let b = child(&map, i, r)?.mask(i, op)?;
                    ExprValue::Mask(if matches!(op, And) { a & b } else { a | b })
                }
                _ => return Err(bad()),
            },
            Not => match node.payload {
                Payload::Unary(c) => ExprValue::Mask(!child(&map, i, c)?.mask(i, op)?),
                _ => return Err(bad()),
            },
            Sgn | Mod => return Err(BridgeError::UnsupportedOp(op)),
        };
        map.push(value);
    }
    Ok(map)
}

/// trace a `symbi-expr` DAG (`nodes`, in topological order) with the given
/// `outputs` into a `SourceProgram`. params land in first-seen declaration
/// order (the splice/runtime binding order).
pub fn lower_dag_to_program(
    nodes: &[Node],
    outputs: &[usize],
) -> Result<SourceProgram, BridgeError> {
    lower_masked_dag_to_program(nodes, outputs, None, 0..0)
}

/// trace a DAG whose output list may end in a region mask `chi(x)`: `region`
/// names the chi node, and the outputs indexed by `mask` are multiplied by
/// chi inside the same trace (so chi shares the field's leaves). the
/// conservation lifts are linear in these outputs, so masking the output
/// masks the final conserved contribution.
fn lower_masked_dag_to_program(
    nodes: &[Node],
    outputs: &[usize],
    region: Option<usize>,
    mask: std::ops::Range<usize>,
) -> Result<SourceProgram, BridgeError> {
    // the trace closure cannot carry a Result out (its outputs are branded),
    // so a walk failure parks the error beside the trace and discards the
    // empty program.
    let mut error: Option<BridgeError> = None;
    let program = SourceProgram::trace(|cx| {
        let map = match eval_dag(cx, nodes) {
            Ok(map) => map,
            Err(e) => {
                error = Some(e);
                return Vec::new();
            }
        };
        let fetch = |o: usize| -> Result<&ExprValue, BridgeError> {
            map.get(o).ok_or(BridgeError::BadChild {
                node: usize::MAX,
                child: o,
            })
        };
        let chi = match region.map(fetch).transpose() {
            Ok(chi) => chi.map(|v| v.materialize()),
            Err(e) => {
                error = Some(e);
                return Vec::new();
            }
        };
        let mut outs = Vec::with_capacity(outputs.len());
        for &o in outputs {
            match fetch(o) {
                Ok(v) => outs.push(v.materialize()),
                Err(e) => {
                    error = Some(e);
                    return Vec::new();
                }
            }
        }
        if let Some(chi) = chi {
            for slot in outs[mask.clone()].iter_mut() {
                *slot = *slot * chi;
            }
        }
        outs
    });
    match error {
        Some(e) => Err(e),
        None => Ok(program),
    }
}

/// the front door: turn a serialized `SourceConfig` (python -> json -> `SourceConfig::from_json`)
/// into the axiomatic `SourceProgram`(s), wrapping the user's free field in the conservation law
/// per `kind`. returns `(target_field, SourceProgram)` pairs ready to splice into the godunov source
/// dispatch — `"force"` yields momentum (+ energy, if the regime has it), `"cooling"` yields a
/// lone energy sink, `"inject"` writes the full conserved vector [den, mom, (nrg)] additively from
/// one config (mass+momentum+energy deposition), `"raw"` writes the outputs straight to a single
/// `target` (the escape hatch). the framework authors the conservative components for
/// force/cooling, so the coupling holds by construction.
///
/// validation is regime-driven via `spec` (the static `RegimeSpec`), so an ill-posed config fails
/// at build time, ahead of attach and well ahead of any evolve step:
/// - `force`/`cooling`/`relax` carry newtonian conservation laws; they are rejected for a
///   relativistic regime (whose conserved momentum is `rho h W^2 v`, a different law). a
///   relativistic regime uses `raw`, supplying conserved components directly.
/// - `cooling` (and `force`/`relax`'s energy overlay) require `has_energy`; cooling on an
///   energy-free (isothermal) regime is rejected loudly at bridge time.
/// - `raw` targets must be a substrate conserved slot (`den | mom | nrg`), and `nrg` requires
///   `has_energy`.
/// - arity: `force` declares one acceleration per dim (`outputs.len() == dim`); `cooling` a single
///   rate; `relax` `[kappa, v_ref_0 .. v_ref_{dim-1}]` (`1 + dim`); `inject` `[den, mom_0 ..
///   mom_{dim-1}, nrg]` (`2 + dim` on energy regimes, `1 + dim` on iso). (the `dim == sim DOF`
///   cross-check is the const-generic apply path's, where DOF is known.)
///
/// the **`region`** axis: if `cfg.region` names a mask node `chi(x)`, the
/// contribution is multiplied by it. the conservation lifts are linear in the field, so masking the
/// field (for `relax`: the rate `kappa` alone) equals masking the conserved contribution — the
/// splice is unchanged, and CPU + GPU fall out of the existing path.
/// lower a census's bin-axis and value expressions into one `SourceProgram`, whose outputs are
/// the axis coordinates followed by the accumulator values (the order
/// `CensusConfig::output_nodes` declares).
///
/// sharing one graph is the point: a shell census bins on `log r` and accumulates several
/// moments that each reference `r`, so hash-consing the common subexpression evaluates the
/// radius and its logarithm once per cell, shared across every registered output. cost then
/// scales with the size of the dag, independent of the accumulator count.
///
/// `dv` is a legal leaf here, where a source term refuses it: the cell measure is the natural
/// weight for an extensive quantity, and it is what keeps a sum correct on a curvilinear grid.
pub fn build_census_expressions(cfg: &symbi_expr::CensusConfig) -> Result<SourceProgram, String> {
    if cfg.values.is_empty() {
        return Err(format!("census '{}': registers no values", cfg.name));
    }
    if cfg.values.len() != cfg.value_names.len() {
        return Err(format!(
            "census '{}': {} value expressions against {} labels",
            cfg.name,
            cfg.values.len(),
            cfg.value_names.len()
        ));
    }
    let nodes = symbi_expr::nodes_from_descs(&cfg.nodes)
        .map_err(|e| format!("census '{}': dag load: {e}", cfg.name))?;
    let outputs = cfg.output_nodes();
    for &o in &outputs {
        if o >= nodes.len() {
            return Err(format!(
                "census '{}': output node {o} is past the end of a {}-node dag",
                cfg.name,
                nodes.len()
            ));
        }
    }
    // constant-power strength reduction: `r ** (-2.0)` becomes a multiply/divide chain,
    // replacing a per-cell libm pow in a kernel that runs over every leaf cell.
    let (nodes, outputs) = symbi_expr::strength_reduce(&nodes, &outputs);
    lower_dag_to_program(&nodes, &outputs)
        .map_err(|e| format!("census '{}': bridge: {e:?}", cfg.name))
}

/// lower one user source without a state law. every kind but `sponge` is
/// law-free; a sponge lowered through this door is refused, since relaxing toward
/// a reference state means knowing which conserved state the regime stores.
pub fn build_user_source(
    cfg: &symbi_expr::SourceConfig,
    spec: &crate::regime_spec::RegimeSpec,
) -> Result<Vec<(String, SourceProgram)>, String> {
    build_user_source_with_law(cfg, spec, None)
}

/// lower one user source against `law`, the conserved state the regime builds from
/// primitives. `sponge` relaxes toward that state and so requires it; the other
/// kinds ignore it.
pub fn build_user_source_with_law(
    cfg: &symbi_expr::SourceConfig,
    spec: &crate::regime_spec::RegimeSpec,
    law: Option<&crate::state_law::StateLaw>,
) -> Result<Vec<(String, SourceProgram)>, String> {
    // a law and a spec describe the same regime from two directions, and a source that
    // relaxes toward a conserved state built under one while the evolution stores the
    // other would be wrong in a way no output reveals. the disagreement is a
    // construction error at the call site, so it is caught here rather than carried.
    if let Some(law) = law {
        if law.relativistic != spec.is_relativistic {
            return Err(format!(
                "state law describes a {} gas while regime '{}' is {}; the source would \
                 relax toward a conserved state the evolution does not store",
                if law.relativistic {
                    "relativistic"
                } else {
                    "newtonian"
                },
                spec.name,
                if spec.is_relativistic {
                    "relativistic"
                } else {
                    "newtonian"
                },
            ));
        }
    }
    let nodes = symbi_expr::nodes_from_descs(&cfg.nodes).map_err(|e| format!("dag load: {e}"))?;
    // lower the field once. if a region mask is declared, lower it as an extra output in the same
    // graph (so chi shares the field's leaves), then peel it off as the mask factor.
    let mut lower_outputs = cfg.outputs.clone();
    if let Some(r) = cfg.region {
        lower_outputs.push(r);
    }
    // constant-power strength reduction (symbi_expr::strength): `x ** (-2.0)` in a
    // user config becomes a multiply/divide chain, replacing a per-cell libm pow.
    let (nodes, lower_outputs) = symbi_expr::strength_reduce(&nodes, &lower_outputs);
    // lower once, with the region chi (when declared) as a trailing extra
    // output sharing the field's leaves; each kind arm below strips it back
    // off and multiplies its maskable output range by chi.
    let field =
        lower_dag_to_program(&nodes, &lower_outputs).map_err(|e| format!("bridge: {e:?}"))?;
    let has_region = cfg.region.is_some();
    let n_out = cfg.outputs.len();

    // the cell-volume leaf is a reduction weight: a source is a density (per unit volume)
    // added to a conserved density, so multiplying it by the cell measure would make the
    // deposited amount depend on the grid. rejecting it here turns what the per-cell source
    // param resolver (which binds no `dv`) would raise mid-evolve into a build-time error.
    if field.params().iter().any(|p| p == "dv") {
        return Err(
            "cell volume is not a source-term input: a source is a per-unit-volume density, so \
             weighting it by the cell measure makes the deposited amount resolution-dependent. \
             it is a binned-reduction weight only"
                .to_string(),
        );
    }

    let reject_relativistic = |law: &str| -> Result<(), String> {
        if spec.is_relativistic {
            return Err(format!(
                "'{law}' bakes a newtonian conservation law, invalid for the relativistic regime \
                 '{}'; use kind='raw' and supply conserved components",
                spec.name,
            ));
        }
        Ok(())
    };

    match cfg.kind.as_str() {
        "force" => {
            reject_relativistic("force")?;
            if n_out != cfg.dim {
                return Err(format!(
                    "'force' needs one acceleration component per dim: outputs.len() = {n_out}, dim = {}",
                    cfg.dim,
                ));
            }
            let field = strip_and_mask(&field, has_region, 0..cfg.dim); // mask the whole acceleration vector
            let mut out = vec![(
                "mom".to_string(),
                crate::source_spec::user_force_momentum_source(&field, cfg.dim),
            )];
            if spec.has_energy {
                out.push((
                    "nrg".to_string(),
                    crate::source_spec::user_force_energy_source(&field, cfg.dim),
                ));
            }
            Ok(out)
        }
        "rotating_frame" => {
            reject_relativistic("rotating_frame")?;
            if !matches!(cfg.dim, 2 | 3) {
                return Err(format!(
                    "'rotating_frame' requires a 2D or 3D cartesian state; dim = {}",
                    cfg.dim,
                ));
            }
            if n_out != 3 {
                return Err(format!(
                    "'rotating_frame' needs [omega, origin_x, origin_y]: outputs.len() = {n_out}, expected 3",
                ));
            }
            if has_region {
                return Err(
                    "'rotating_frame' does not accept a region mask; make omega zero outside the region"
                        .to_string(),
                );
            }
            let mut out = vec![(
                "mom".to_string(),
                crate::source_spec::user_rotating_frame_momentum_source(&field, cfg.dim),
            )];
            if spec.has_energy {
                out.push((
                    "nrg".to_string(),
                    crate::source_spec::user_rotating_frame_energy_source(&field, cfg.dim),
                ));
            }
            Ok(out)
        }
        "cooling" => {
            reject_relativistic("cooling")?;
            if !spec.has_energy {
                return Err(format!(
                    "'cooling' requires an energy equation; regime '{}' has none (isothermal)",
                    spec.name,
                ));
            }
            if n_out != 1 {
                return Err(format!(
                    "'cooling' is a single rate: outputs.len() = {n_out}, expected 1"
                ));
            }
            let field = strip_and_mask(&field, has_region, 0..1);
            Ok(vec![(
                "nrg".to_string(),
                crate::source_spec::user_cooling_source(&field, cfg.dim),
            )])
        }
        "relax" => {
            // velocity relaxation (sponge / buffer zone): S_mom = kappa*rho*(v_ref - v), kappa >= 0.
            reject_relativistic("relax")?;
            if n_out != 1 + cfg.dim {
                return Err(format!(
                    "'relax' needs [kappa, v_ref_0..v_ref_{}]: outputs.len() = {n_out}, expected {}",
                    cfg.dim.saturating_sub(1),
                    1 + cfg.dim,
                ));
            }
            // region masks the rate kappa (output 0) alone — masking v_ref would corrupt the target.
            let field = strip_and_mask(&field, has_region, 0..1);
            let mut out = vec![(
                "mom".to_string(),
                crate::source_spec::user_relax_momentum_source(&field, cfg.dim),
            )];
            if spec.has_energy {
                out.push((
                    "nrg".to_string(),
                    crate::source_spec::user_relax_energy_source(&field, cfg.dim),
                ));
            }
            Ok(out)
        }
        "sponge" => {
            // the buffer zone relaxes the whole conserved state toward a reference, and
            // which state that is belongs to the regime: `rho v` on a newtonian gas,
            // `rho h W^2 v` on a relativistic one, `sqrt(gamma) rho h W^2 v` on a curved
            // background. the reference is therefore supplied as primitives and converted
            // by the regime itself, which is why this kind needs a state law and why it is
            // no longer restricted to newtonian regimes.
            let law = law.ok_or_else(|| {
                "'sponge' relaxes toward a conserved reference state and needs the state \
                 law that builds it; lower it through `build_user_source_with_law`"
                    .to_string()
            })?;
            let want = if spec.has_energy {
                3 + cfg.dim
            } else {
                2 + cfg.dim
            };
            if n_out != want {
                return Err(format!(
                    "'sponge' needs [kappa, rho_ref, vel_ref_0..vel_ref_{}{}]: outputs.len() = {n_out}, expected {want}",
                    cfg.dim.saturating_sub(1),
                    if spec.has_energy { ", pre_ref" } else { "" },
                ));
            }
            // region masks the rate kappa (output 0) alone, which factors into every
            // channel; masking the reference state would corrupt the target the flow
            // relaxes toward rather than the region it relaxes in.
            let field = strip_and_mask(&field, has_region, 0..1);
            crate::source_spec::user_sponge_sources(&field, cfg.dim, law, spec.has_energy)
        }
        "inject" => {
            // additive deposition of the full conserved vector in one config: outputs =
            // [S_den, S_mom_0..S_mom_{D-1}, (S_nrg on energy regimes)], each written straight to
            // its conserved slot (identity, like `raw`, spanning every slot at once). this is the
            // mass+momentum+energy injection (a jet/wind depositing all three) that single-slot
            // `raw` reaches one slot at a time. relativistic-safe: the user supplies conserved
            // components directly, so the kind skips `reject_relativistic` — the components are
            // the regime's conserved rates (D=rho*W, S=rho*h*W^2*v, tau in rhd).
            let want = if spec.has_energy {
                2 + cfg.dim
            } else {
                1 + cfg.dim
            };
            if n_out != want {
                return Err(format!(
                    "'inject' needs [den, mom_0..mom_{}{}]: outputs.len() = {n_out}, expected {want}",
                    cfg.dim.saturating_sub(1),
                    if spec.has_energy { ", nrg" } else { "" },
                ));
            }
            let field = strip_and_mask(&field, has_region, 0..n_out); // region masks every conserved channel
            let mut out = vec![
                (
                    "den".to_string(),
                    crate::source_spec::user_inject_slot_source(&field, 0..1),
                ),
                (
                    "mom".to_string(),
                    crate::source_spec::user_inject_slot_source(&field, 1..1 + cfg.dim),
                ),
            ];
            if spec.has_energy {
                out.push((
                    "nrg".to_string(),
                    crate::source_spec::user_inject_slot_source(&field, 1 + cfg.dim..2 + cfg.dim),
                ));
            }
            Ok(out)
        }
        "raw" => {
            let target = cfg
                .target
                .clone()
                .ok_or_else(|| "raw source requires a `target` field".to_string())?;
            if !matches!(target.as_str(), "den" | "mom" | "nrg") {
                return Err(format!(
                    "raw target '{target}' is not a conserved slot (expected den | mom | nrg)"
                ));
            }
            if target == "nrg" && !spec.has_energy {
                return Err(format!(
                    "raw target 'nrg' requires an energy equation; regime '{}' has none",
                    spec.name,
                ));
            }
            let field = strip_and_mask(&field, has_region, 0..n_out); // mask every conserved component the user wrote
            Ok(vec![(target, field)])
        }
        other => Err(format!(
            "unknown source kind '{other}' (expected force | rotating_frame | cooling | relax | sponge | inject | raw)"
        )),
    }
}

/// lower an ordered collection of user sources into one runtime source set.
/// parameter indices are made global before lowering, and contributions that
/// target the same conserved field are summed component by component.
/// lower an ordered collection without a state law; see `build_user_source`.
pub fn build_user_sources(
    configs: &[symbi_expr::SourceConfig],
    spec: &crate::regime_spec::RegimeSpec,
) -> Result<(Vec<(String, SourceProgram)>, Vec<f64>), String> {
    build_user_sources_with_law(configs, spec, None)
}

/// lower an ordered collection against `law`; see `build_user_source_with_law`.
pub fn build_user_sources_with_law(
    configs: &[symbi_expr::SourceConfig],
    spec: &crate::regime_spec::RegimeSpec,
    law: Option<&crate::state_law::StateLaw>,
) -> Result<(Vec<(String, SourceProgram)>, Vec<f64>), String> {
    let mut parameter_offset = 0usize;
    let mut params = Vec::new();
    let mut lowered = Vec::new();

    for config in configs {
        let mut config = config.clone();
        for node in &mut config.nodes {
            if let Some(index) = &mut node.param_idx {
                *index += parameter_offset;
            }
        }
        parameter_offset += config.params.len();
        params.extend_from_slice(&config.params);
        lowered.extend(build_user_source_with_law(&config, spec, law)?);
    }

    let mut composed: Vec<(String, SourceProgram)> = Vec::new();
    for (target, built) in lowered {
        if let Some((_, existing)) = composed.iter_mut().find(|(name, _)| name == &target) {
            *existing = sum_source_programs(existing, &built, &target)?;
        } else {
            composed.push((target, built));
        }
    }
    Ok((composed, params))
}

fn sum_source_programs(
    left: &SourceProgram,
    right: &SourceProgram,
    target: &str,
) -> Result<SourceProgram, String> {
    if left.outputs().len() != right.outputs().len() {
        return Err(format!(
            "source target '{target}' has incompatible component counts: {} and {}",
            left.outputs().len(),
            right.outputs().len()
        ));
    }
    // splice both programs into one trace with params bound to same-named
    // scalar leaves (a shared param lands on one leaf), then sum
    // component-wise.
    Ok(SourceProgram::trace(|cx| {
        let l = cx.splice_source_as_scalars(left);
        let r = cx.splice_source_as_scalars(right);
        l.into_iter().zip(r).map(|(a, b)| a + b).collect()
    }))
}

/// splice `field` into a fresh trace, strip the trailing region chi (when
/// present), and multiply the outputs indexed by `idxs` by it. the lifts are
/// linear in these outputs, so this masks the final conserved contribution.
/// `idxs` selects which outputs carry the maskable quantity (e.g., relax
/// masks the rate `kappa` alone, leaving the reference velocity at full
/// strength). without a region the field passes through unchanged.
fn strip_and_mask(
    field: &SourceProgram,
    has_region: bool,
    idxs: std::ops::Range<usize>,
) -> SourceProgram {
    if !has_region {
        return field.clone();
    }
    SourceProgram::trace(|cx| {
        let mut outs = cx.splice_source_as_scalars(field);
        let chi = outs.pop().expect("region output");
        for slot in outs[idxs.clone()].iter_mut() {
            *slot = *slot * chi;
        }
        outs
    })
}

/// the boundary front door: compile a `SourceConfig` into a driven-boundary
/// prescription — a complete primitive state `[rho, vel_0..vel_{D-1}, pre]` the ghost cells are set
/// to (Dirichlet), `combine = overwrite`. returns `(slot, SourceProgram)` in the structural-slot
/// convention `den`/`mom`/`nrg` that [`symbi_discretize::boundary_fill_from_built_gv`] writes to
/// `prim.rho`/`prim.vel_k`/`prim.pre`. each slot is an independent lowering of the user DAG over its
/// output subset, so the velocity vector lands as the `ncomp`-output `mom` slot.
///
/// validation (regime-driven via `spec`, at attach time):
/// - **regime-agnostic across hydro and MHD.** a prim prescription assigns `prim` outright, which
///   keeps it valid for RHD too (`is_relativistic` allowed). MHD additionally prescribes the cell-B
///   vector (`bcell` slot -> `prim.mag`): the out-of-plane component (B_phi in a 2.5D axisymmetric
///   grid) is cell-centered + flux-evolved, so prescribing it is a plain Dirichlet; the CT
///   tangential-EMF sub-problem belongs to a prescribed poloidal/in-plane face field.
///   the in-plane cell-B components are the user's responsibility to keep div-compatible (=0 for a
///   purely toroidal field) — `raw`-style: garbage in, garbage out.
/// - **complete prim state.** the DAG must output exactly the regime's primitive components:
///   `1 (rho) + d (vel) + has_energy (pre) + is_mhd*d (cell B)` where `d = cfg.dim` is the vector
///   component count (= DOF). a partial prescription is rejected.
/// (the `dim == sim DOF` cross-check is the const-generic dispatch's, where DOF is known.)
/// the number of outputs a driven-boundary prescription carries for `spec` at vector-component
/// count `d`: the complete primitive state, `rho` + `d` velocities + pressure (energy regimes) +
/// `d` cell-B components (MHD). a run carrying the passive scalar appends exactly one more, the dye
/// concentration of the injected fluid, so its prescription has arity `n + 1`.
///
/// the single definition of that count: `build_boundary_dag` validates against it, and the config
/// layer sizes its dye requirement from it.
pub fn boundary_prim_arity(spec: &crate::regime_spec::RegimeSpec, d: usize) -> usize {
    1 + d + usize::from(spec.has_energy) + if spec.is_mhd { d } else { 0 }
}

pub fn build_boundary_dag(
    cfg: &symbi_expr::SourceConfig,
    spec: &crate::regime_spec::RegimeSpec,
) -> Result<Vec<(String, SourceProgram)>, String> {
    let nodes = symbi_expr::nodes_from_descs(&cfg.nodes).map_err(|e| format!("dag load: {e}"))?;
    let (nodes, reduced_outputs) = symbi_expr::strength_reduce(&nodes, &cfg.outputs);
    let d = cfg.dim;
    // MHD prescribes the cell-B vector too. the out-of-plane component (B_phi in a 2.5D
    // axisymmetric grid: cell-centered, flux-evolved) is the safe toroidal case — div-free
    // by axisymmetry. the in-plane components are the user's responsibility to keep
    // div-compatible (=0 for a purely toroidal field); they enter as cell-centered values,
    // so the tangential-EMF sub-problem stays with the CT face prescription, which is where
    // a poloidal (in-plane face) field would be set.
    let n_prim = boundary_prim_arity(spec, d);
    // a run carrying the passive scalar prescribes one more output, the dye concentration of the
    // injected fluid, appended after the prim state. optional here because the dye is a run-level
    // opt-in outside the regime spec's view; the config layer knows whether a dye is allocated
    // and is what rejects a driven face that omits it.
    let has_dye = cfg.outputs.len() == n_prim + 1;
    if cfg.outputs.len() != n_prim && !has_dye {
        return Err(format!(
            "driven boundary must prescribe the full prim state [rho, vel_0..vel_{}{}{}]: outputs.len() \
             = {}, expected {n_prim} (or {} with a trailing dye concentration)",
            d.saturating_sub(1),
            if spec.has_energy { ", pre" } else { "" },
            if spec.is_mhd {
                format!(", B_0..B_{}", d.saturating_sub(1))
            } else {
                String::new()
            },
            cfg.outputs.len(),
            n_prim + 1,
        ));
    }
    // split the user DAG into per-slot prescriptions: den <- rho, mom <- the d-vector vel,
    // nrg <- pre, bcell <- the d-vector cell B. each is an independent lowering over its
    // output subset, building its own graph.
    let lower = |outs: &[usize]| {
        lower_dag_to_program(&nodes, outs).map_err(|e| format!("bridge: {e:?}"))
    };
    let mut out = vec![
        ("den".to_string(), lower(&reduced_outputs[0..1])?),
        ("mom".to_string(), lower(&reduced_outputs[1..1 + d])?),
    ];
    let mut next = 1 + d;
    if spec.has_energy {
        out.push(("nrg".to_string(), lower(&reduced_outputs[next..next + 1])?));
        next += 1;
    }
    if spec.is_mhd {
        out.push((
            "bcell".to_string(),
            lower(&reduced_outputs[next..next + d])?,
        ));
        next += d;
    }
    if has_dye {
        out.push(("chi".to_string(), lower(&reduced_outputs[next..next + 1])?));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::regime_spec::{ISO_NEWTONIAN_SPEC, NEWTONIAN_SPEC, RHD_SPEC, RMHD_SPEC};
    use symbi_expr::dag::Dag;
    use symbi_ir::backends::interp::{Backend, Cpu};
    use symbi_ir::NodeId;
    use symbi_ir::passes::scalarize::scalarize;

    // ---- the axiomatic validation gate (build_user_source vs RegimeSpec) -------------------
    // each ill-posed config must fail here (pre-attach), driven entirely by the regime's spec,
    // ahead of any evolve step.

    fn cfg_from(json: &str) -> symbi_expr::SourceConfig {
        symbi_expr::SourceConfig::from_json(json).expect("parse")
    }

    // extract the error message (SourceProgram lacks Debug, so this stands in for `unwrap_err`).
    /// the ideal-gas law the fixtures below lower against. a sponge relaxes toward a
    /// reference state expressed in primitives, so lowering one needs the conversion its
    /// regime uses; every other source kind ignores the law and lowers identically with it.
    fn fixture_law() -> crate::state_law::StateLaw {
        crate::state_law::StateLaw::newtonian(1.4)
    }

    /// the law matching `spec`'s regime. lowering rejects a law and a spec that disagree
    /// about relativity, so a fixture that means to exercise some other rejection has to
    /// hand over the law belonging to the regime it names.
    fn law_for(spec: &crate::regime_spec::RegimeSpec) -> crate::state_law::StateLaw {
        if spec.is_relativistic {
            crate::state_law::StateLaw::relativistic(1.4, crate::state_law::Background::Minkowski)
        } else {
            fixture_law()
        }
    }

    fn lower(
        cfg: &symbi_expr::SourceConfig,
        spec: &crate::regime_spec::RegimeSpec,
    ) -> Result<Vec<(String, SourceProgram)>, String> {
        build_user_source_with_law(cfg, spec, Some(&law_for(spec)))
    }

    fn expect_err(cfg: &symbi_expr::SourceConfig, spec: &crate::regime_spec::RegimeSpec) -> String {
        match lower(cfg, spec) {
            Err(e) => e,
            Ok(_) => panic!("expected the config to be rejected"),
        }
    }

    #[test]
    fn cell_volume_is_rejected_as_a_source_input() {
        // a source term is a per-unit-volume density added to a conserved density. weighting
        // it by the cell measure would make the deposited amount scale with the resolution,
        // so the leaf is refused at build time; the per-cell source param resolver binds no
        // `dv`, so this check converts a mid-evolve panic into a build-time error.
        let cfg = cfg_from(
            r#"{ "kind":"raw", "dim":1, "outputs":[1], "params":[], "target":"den",
                 "nodes":[ {"op":"VARIABLE_DV"},
                           {"op":"MULTIPLY","left":0,"right":0} ] }"#,
        );
        let err = expect_err(&cfg, &NEWTONIAN_SPEC);
        assert!(
            err.contains("cell volume is not a source-term input"),
            "expected the cell-volume rejection, got: {err}"
        );
    }

    #[test]
    fn source_collection_isolates_params_and_sums_shared_targets() {
        let first = cfg_from(
            r#"{ "kind":"raw", "dim":1, "outputs":[0], "params":[2.0], "target":"nrg",
                 "nodes":[ {"op":"PARAMETER","param_idx":0} ] }"#,
        );
        let second = cfg_from(
            r#"{ "kind":"raw", "dim":1, "outputs":[0], "params":[5.0], "target":"nrg",
                 "nodes":[ {"op":"PARAMETER","param_idx":0} ] }"#,
        );

        let (built, params) =
            build_user_sources(&[first, second], &NEWTONIAN_SPEC).expect("compose sources");
        assert_eq!(params, [2.0, 5.0]);
        assert_eq!(built.len(), 1);
        assert_eq!(built[0].0, "nrg");
        let evaluator = crate::SourceEvaluator::from_built(&built);
        let value = evaluator
            .eval("nrg", &[("p0", params[0]), ("p1", params[1])])
            .expect("energy source");
        assert_eq!(value, [7.0]);
    }

    #[test]
    fn python_serialize_source_json_loads_and_lowers() {
        // pins the cross-language wire: this is the exact json the python
        // CompiledExpr.serialize_source('force', 2) emits for a = (0, -1) gravity.
        // if the python adapter and this loader ever drift, this fails.
        let cfg = cfg_from(
            r#"{"kind": "force", "dim": 2, "outputs": [0, 1], "params": [],
                "nodes": [{"op": "CONSTANT", "value": 0.0},
                          {"op": "CONSTANT", "value": -1.0}]}"#,
        );
        assert_eq!(cfg.kind, "force");
        assert_eq!(cfg.dim, 2);
        assert_eq!(cfg.outputs, vec![0, 1]);
        let built = lower(&cfg, &NEWTONIAN_SPEC).expect("python force config lowers");
        assert_eq!(
            built.iter().map(|(t, _)| t.as_str()).collect::<Vec<_>>(),
            ["mom", "nrg"]
        );
    }

    #[test]
    fn python_tabulated_field_lowers_and_interpolates_in_the_rust_evaluator() {
        let cfg = cfg_from(
            r#"{"kind":"raw","dim":1,"outputs":[23],"params":[],"target":"nrg","nodes":[
                {"op":"VARIABLE_X1"},{"op":"CONSTANT","value":1.0},
                {"op":"CONSTANT","value":10.0},{"op":"CONSTANT","value":10.0},
                {"op":"SUBTRACT","left":0,"right":1},
                {"op":"MULTIPLY","left":3,"right":4},{"op":"ADD","left":2,"right":5},
                {"op":"CONSTANT","value":2.0},{"op":"CONSTANT","value":20.0},
                {"op":"CONSTANT","value":-10.0},{"op":"SUBTRACT","left":0,"right":7},
                {"op":"MULTIPLY","left":9,"right":10},{"op":"ADD","left":8,"right":11},
                {"op":"CONSTANT","value":2.0},{"op":"LT","left":0,"right":13},
                {"op":"IF_THEN_ELSE","condition":14,"true_case":6,"false_case":12},
                {"op":"CONSTANT","value":10.0},{"op":"CONSTANT","value":0.0},
                {"op":"CONSTANT","value":1.0},{"op":"LT","left":0,"right":18},
                {"op":"CONSTANT","value":4.0},{"op":"GT","left":0,"right":20},
                {"op":"IF_THEN_ELSE","condition":21,"true_case":17,"false_case":15},
                {"op":"IF_THEN_ELSE","condition":19,"true_case":16,"false_case":22}
            ]}"#,
        );
        let built = lower(&cfg, &NEWTONIAN_SPEC).expect("lower table");
        let evaluator = crate::SourceEvaluator::from_built(&built);
        assert_eq!(
            evaluator.eval("nrg", &[("x_0", 1.5)]).expect("interior"),
            [15.0],
        );
        assert_eq!(
            evaluator
                .eval("nrg", &[("x_0", -1.0)])
                .expect("lower clamp"),
            [10.0],
        );
        assert_eq!(
            evaluator.eval("nrg", &[("x_0", 9.0)]).expect("upper clamp"),
            [0.0],
        );
    }

    #[test]
    fn rotating_frame_lowers_to_momentum_and_energy_and_rejects_relativity() {
        let cfg = cfg_from(
            r#"{"kind": "rotating_frame", "dim": 2, "outputs": [0, 1, 2], "params": [],
                "nodes": [{"op": "CONSTANT", "value": 2.0},
                          {"op": "CONSTANT", "value": 0.0},
                          {"op": "CONSTANT", "value": 0.0}]}"#,
        );
        let built = lower(&cfg, &NEWTONIAN_SPEC).expect("rotating frame config lowers");
        assert_eq!(
            built
                .iter()
                .map(|(target, _)| target.as_str())
                .collect::<Vec<_>>(),
            ["mom", "nrg"],
        );
        assert!(expect_err(&cfg, &RHD_SPEC).contains("invalid for the relativistic regime"));
    }

    #[test]
    fn rotating_frame_composes_with_sponge_in_one_momentum_plan() {
        let rotating = cfg_from(
            r#"{"kind": "rotating_frame", "dim": 2, "outputs": [0, 1, 2], "params": [],
                "nodes": [{"op": "CONSTANT", "value": 2.0},
                          {"op": "CONSTANT", "value": 0.0},
                          {"op": "CONSTANT", "value": 0.0}]}"#,
        );
        let sponge = cfg_from(
            r#"{"kind": "sponge", "dim": 2, "outputs": [0, 1, 2, 3], "params": [],
                "nodes": [{"op": "CONSTANT", "value": 1.0},
                          {"op": "CONSTANT", "value": 1.0},
                          {"op": "CONSTANT", "value": 0.0},
                          {"op": "CONSTANT", "value": 0.0}]}"#,
        );
        let (built, params) = build_user_sources_with_law(
            &[rotating, sponge],
            &ISO_NEWTONIAN_SPEC,
            Some(&fixture_law()),
        )
        .expect("compose");
        assert!(params.is_empty());
        assert_eq!(
            built.iter().filter(|(target, _)| target == "mom").count(),
            1
        );

        let evaluator = crate::SourceEvaluator::from_built(&built);
        let momentum = evaluator
            .eval(
                "mom",
                &[
                    ("rho", 1.0),
                    ("vel_0", 1.0),
                    ("vel_1", 0.0),
                    ("x_0", 1.0),
                    ("x_1", 0.0),
                ],
            )
            .expect("momentum source");
        assert_eq!(momentum, [3.0, -4.0]);
    }

    #[test]
    fn rotating_frame_centrifugal_force_balances_a_harmonic_force() {
        let rotating = cfg_from(
            r#"{"kind": "rotating_frame", "dim": 2, "outputs": [0, 1, 2], "params": [],
                "nodes": [{"op": "CONSTANT", "value": 2.0},
                          {"op": "CONSTANT", "value": 0.0},
                          {"op": "CONSTANT", "value": 0.0}]}"#,
        );
        let restoring = cfg_from(
            r#"{"kind": "force", "dim": 2, "outputs": [3, 4], "params": [],
                "nodes": [{"op": "VARIABLE_X1"}, {"op": "VARIABLE_X2"},
                          {"op": "CONSTANT", "value": -4.0},
                          {"op": "MULTIPLY", "left": 0, "right": 2},
                          {"op": "MULTIPLY", "left": 1, "right": 2}]}"#,
        );
        let (built, _) =
            build_user_sources(&[rotating, restoring], &ISO_NEWTONIAN_SPEC).expect("compose");
        let evaluator = crate::SourceEvaluator::from_built(&built);
        let momentum = evaluator
            .eval(
                "mom",
                &[
                    ("rho", 1.0),
                    ("vel_0", 0.0),
                    ("vel_1", 0.0),
                    ("x_0", 3.0),
                    ("x_1", -2.0),
                ],
            )
            .expect("balanced momentum source");
        assert_eq!(momentum, [0.0, 0.0]);
    }

    #[test]
    fn python_sponge_json_loads_and_lowers() {
        // the exact json python's serialize_source(SourceKind.sponge, dim=3) emits.
        // outputs = [kappa, rho_ref, vel_ref_0..2, pre_ref] mapped to node indices; here
        // kappa=2, rho_ref=1, vel_ref=(x_0,x_1,x_2) (reads position), pre_ref=10. the wire
        // carries PRIMITIVES, so the regime's own conversion supplies the conserved
        // reference and the adiabatic index never crosses the language boundary.
        let cfg = cfg_from(
            r#"{"kind": "sponge", "dim": 3, "outputs": [3, 4, 0, 1, 2, 5], "params": [],
                "nodes": [{"op": "VARIABLE_X1"}, {"op": "VARIABLE_X2"}, {"op": "VARIABLE_X3"},
                          {"op": "CONSTANT", "value": 2.0}, {"op": "CONSTANT", "value": 1.0},
                          {"op": "CONSTANT", "value": 10.0}]}"#,
        );
        let built = lower(&cfg, &NEWTONIAN_SPEC).expect("python sponge config lowers");
        assert_eq!(
            built.iter().map(|(t, _)| t.as_str()).collect::<Vec<_>>(),
            ["den", "mom", "nrg"]
        );
        // at x=(1,0,0) the reference is rho=1, v=(1,0,0), p=10; the state is rho=1.5,
        // v=(3,0,0), p=2, under gamma = 1.4.
        let state: &[(&str, f64)] = &[
            ("rho", 1.5),
            ("vel_0", 3.0),
            ("vel_1", 0.0),
            ("vel_2", 0.0),
            ("pre", 2.0),
            ("x_0", 1.0),
            ("x_1", 0.0),
            ("x_2", 0.0),
        ];
        // S_den = kappa*(rho_ref - rho) = 2*(1 - 1.5) = -1.
        let (_, den) = &built[0];
        let s_den = eval_lowered(den, den.outputs()[0], state);
        assert!(
            (s_den - (-1.0)).abs() < 1e-12,
            "python sponge den wrong: {s_den}"
        );
        // S_mom_0 = kappa*(rho_ref vel_ref_0 - rho vel_0) = 2*(1 - 4.5) = -7.
        let (_, mom) = &built[1];
        let s_mom0 = eval_lowered(mom, mom.outputs()[0], state);
        assert!(
            (s_mom0 - (-7.0)).abs() < 1e-12,
            "python sponge mom_0 wrong: {s_mom0}"
        );
        // the energy slot is where the conversion earns its keep: both sides are built by
        // the regime, so the reference kinetic term rho_ref |v_ref|^2/2 is carried rather
        // than dropped. S_nrg = 2*((10/0.4 + 0.5) - (2/0.4 + 6.75)) = 2*(25.5 - 11.75) = 27.5.
        let (_, nrg) = &built[2];
        let s_nrg = eval_lowered(nrg, nrg.outputs()[0], state);
        assert!(
            (s_nrg - 27.5).abs() < 1e-12,
            "python sponge nrg wrong: {s_nrg}"
        );
    }

    #[test]
    fn force_on_newtonian_is_accepted() {
        // mom + nrg overlays (newtonian has energy).
        let cfg = cfg_from(
            r#"{ "kind":"force", "dim":2, "outputs":[0,1], "params":[0.5],
                 "nodes":[ {"op":"PARAMETER","param_idx":0}, {"op":"CONSTANT","value":0.0} ] }"#,
        );
        let built = lower(&cfg, &NEWTONIAN_SPEC).expect("force ok on newtonian");
        assert_eq!(
            built.iter().map(|(t, _)| t.as_str()).collect::<Vec<_>>(),
            ["mom", "nrg"]
        );
    }

    #[test]
    fn force_on_iso_drops_energy_overlay() {
        // iso has no energy: the mom overlay is the whole emission.
        let cfg = cfg_from(
            r#"{ "kind":"force", "dim":2, "outputs":[0,1], "params":[0.5],
                 "nodes":[ {"op":"PARAMETER","param_idx":0}, {"op":"CONSTANT","value":0.0} ] }"#,
        );
        let built = lower(&cfg, &ISO_NEWTONIAN_SPEC).expect("force ok on iso");
        assert_eq!(
            built.iter().map(|(t, _)| t.as_str()).collect::<Vec<_>>(),
            ["mom"]
        );
    }

    #[test]
    fn force_on_relativistic_is_rejected() {
        // RHD momentum is rho*h*W^2*v — the newtonian force law is wrong; reject.
        let cfg = cfg_from(
            r#"{ "kind":"force", "dim":1, "outputs":[0], "params":[0.5],
                 "nodes":[ {"op":"PARAMETER","param_idx":0} ] }"#,
        );
        let err = expect_err(&cfg, &RHD_SPEC);
        assert!(
            err.contains("relativistic"),
            "expected relativistic rejection, got: {err}"
        );
    }

    #[test]
    fn inject_on_newtonian_writes_all_conserved_slots() {
        // one config depositing mass+momentum+energy: outputs = [S_den, S_mom_0, S_mom_1, S_nrg]
        // = [1, 2, 3, 4], each written identity to its conserved slot (like raw, spanning every
        // slot at once). the multi-channel deposition single-slot raw reaches one slot at a time.
        let cfg = cfg_from(
            r#"{ "kind":"inject", "dim":2, "outputs":[0,1,2,3], "params":[],
                 "nodes":[ {"op":"CONSTANT","value":1.0}, {"op":"CONSTANT","value":2.0},
                           {"op":"CONSTANT","value":3.0}, {"op":"CONSTANT","value":4.0} ] }"#,
        );
        let built = lower(&cfg, &NEWTONIAN_SPEC).expect("inject ok on newtonian");
        assert_eq!(
            built.iter().map(|(t, _)| t.as_str()).collect::<Vec<_>>(),
            ["den", "mom", "nrg"]
        );
        // den: single output = 1.
        let (_, den) = &built[0];
        assert_eq!(den.outputs().len(), 1);
        assert!((eval_lowered(den, den.outputs()[0], &[]) - 1.0).abs() < 1e-12);
        // mom: D=2 outputs = [2, 3], in order.
        let (_, mom) = &built[1];
        assert_eq!(mom.outputs().len(), 2);
        assert!((eval_lowered(mom, mom.outputs()[0], &[]) - 2.0).abs() < 1e-12);
        assert!((eval_lowered(mom, mom.outputs()[1], &[]) - 3.0).abs() < 1e-12);
        // nrg: single output = 4.
        let (_, nrg) = &built[2];
        assert_eq!(nrg.outputs().len(), 1);
        assert!((eval_lowered(nrg, nrg.outputs()[0], &[]) - 4.0).abs() < 1e-12);
    }

    #[test]
    fn inject_on_iso_drops_energy_channel() {
        // iso has no energy: outputs = [S_den, S_mom_0, S_mom_1] (1+dim); den + mom are the
        // whole emission.
        let cfg = cfg_from(
            r#"{ "kind":"inject", "dim":2, "outputs":[0,1,2], "params":[],
                 "nodes":[ {"op":"CONSTANT","value":1.0}, {"op":"CONSTANT","value":2.0},
                           {"op":"CONSTANT","value":3.0} ] }"#,
        );
        let built = lower(&cfg, &ISO_NEWTONIAN_SPEC).expect("inject ok on iso");
        assert_eq!(
            built.iter().map(|(t, _)| t.as_str()).collect::<Vec<_>>(),
            ["den", "mom"]
        );
    }

    #[test]
    fn inject_on_relativistic_is_accepted() {
        // inject supplies conserved components directly (like raw, with the law wrap left off), so
        // it is valid on a relativistic regime where force/cooling/relax are rejected. rhd has energy:
        // [D_dot, S_dot_0, tau_dot] at dim=1 -> 3 outputs.
        let cfg = cfg_from(
            r#"{ "kind":"inject", "dim":1, "outputs":[0,1,2], "params":[],
                 "nodes":[ {"op":"CONSTANT","value":1.0}, {"op":"CONSTANT","value":2.0},
                           {"op":"CONSTANT","value":3.0} ] }"#,
        );
        let built = build_user_source(&cfg, &RHD_SPEC).expect("inject ok on rhd (raw-like)");
        assert_eq!(
            built.iter().map(|(t, _)| t.as_str()).collect::<Vec<_>>(),
            ["den", "mom", "nrg"]
        );
    }

    #[test]
    fn inject_relativistic_2d_engine_channels() {
        // the Duffell & MacFadyen 2015 collimated engine on SRHD (dim=2): one nozzle power
        // S_0 drives three coupled conserved-rate channels — S_den = S_0/eta_0, S_mom_r =
        // S_0*sqrt(1-1/gamma_0^2), S_mom_theta = 0 (purely radial), S_nrg = S_0. here S_0=10,
        // eta_0=100 (node divide -> 0.1), S_mom_r=9.998, mirroring the axis_jet source shape.
        let cfg = cfg_from(
            r#"{ "kind":"inject", "dim":2, "outputs":[2,3,4,0], "params":[],
                 "nodes":[ {"op":"CONSTANT","value":10.0}, {"op":"CONSTANT","value":100.0},
                           {"op":"DIVIDE","left":0,"right":1}, {"op":"CONSTANT","value":9.998},
                           {"op":"CONSTANT","value":0.0} ] }"#,
        );
        let built = build_user_source(&cfg, &RHD_SPEC).expect("2d engine inject ok on rhd");
        assert_eq!(
            built.iter().map(|(t, _)| t.as_str()).collect::<Vec<_>>(),
            ["den", "mom", "nrg"]
        );
        // S_den = S_0/eta_0 = 10/100 = 0.1 (the divide channel).
        let (_, den) = &built[0];
        assert_eq!(den.outputs().len(), 1);
        assert!((eval_lowered(den, den.outputs()[0], &[]) - 0.1).abs() < 1e-12);
        // S_mom = [S_mom_r, S_mom_theta] = [9.998, 0]; the theta channel is exactly zero.
        let (_, mom) = &built[1];
        assert_eq!(mom.outputs().len(), 2);
        assert!((eval_lowered(mom, mom.outputs()[0], &[]) - 9.998).abs() < 1e-12);
        assert!(eval_lowered(mom, mom.outputs()[1], &[]).abs() < 1e-12);
        // S_nrg = S_0 = 10.
        let (_, nrg) = &built[2];
        assert!((eval_lowered(nrg, nrg.outputs()[0], &[]) - 10.0).abs() < 1e-12);
    }

    #[test]
    fn inject_relativistic_3d_engine_channels() {
        // the 3d (r, theta, phi) SRHD engine: 5 outputs [S_den, S_mom_r, S_mom_theta, S_mom_phi,
        // S_nrg]. the radial nozzle leaves both transverse momentum channels zero, so the mom slot
        // carries 3 components [S_mom_r, 0, 0]. mirrors the blade / threed_jet source shape.
        let cfg = cfg_from(
            r#"{ "kind":"inject", "dim":3, "outputs":[2,3,4,5,0], "params":[],
                 "nodes":[ {"op":"CONSTANT","value":10.0}, {"op":"CONSTANT","value":100.0},
                           {"op":"DIVIDE","left":0,"right":1}, {"op":"CONSTANT","value":9.998},
                           {"op":"CONSTANT","value":0.0}, {"op":"CONSTANT","value":0.0} ] }"#,
        );
        let built = build_user_source(&cfg, &RHD_SPEC).expect("3d engine inject ok on rhd");
        assert_eq!(
            built.iter().map(|(t, _)| t.as_str()).collect::<Vec<_>>(),
            ["den", "mom", "nrg"]
        );
        let (_, den) = &built[0];
        assert!((eval_lowered(den, den.outputs()[0], &[]) - 0.1).abs() < 1e-12);
        // mom carries all three spatial components; only the radial one is nonzero.
        let (_, mom) = &built[1];
        assert_eq!(mom.outputs().len(), 3);
        assert!((eval_lowered(mom, mom.outputs()[0], &[]) - 9.998).abs() < 1e-12);
        assert!(eval_lowered(mom, mom.outputs()[1], &[]).abs() < 1e-12);
        assert!(eval_lowered(mom, mom.outputs()[2], &[]).abs() < 1e-12);
        let (_, nrg) = &built[2];
        assert!((eval_lowered(nrg, nrg.outputs()[0], &[]) - 10.0).abs() < 1e-12);
    }

    #[test]
    fn inject_wrong_arity_is_rejected() {
        // energy regime at dim=2 needs [den, mom_0, mom_1, nrg] = 4 outputs; supplying 3 is
        // rejected pre-attach, ahead of any evolve step.
        let cfg = cfg_from(
            r#"{ "kind":"inject", "dim":2, "outputs":[0,1,2], "params":[],
                 "nodes":[ {"op":"CONSTANT","value":1.0}, {"op":"CONSTANT","value":2.0},
                           {"op":"CONSTANT","value":3.0} ] }"#,
        );
        let err = expect_err(&cfg, &NEWTONIAN_SPEC);
        assert!(
            err.contains("inject"),
            "expected inject arity rejection, got: {err}"
        );
    }

    #[test]
    fn cooling_on_iso_is_rejected() {
        // cooling targets nrg, which iso lacks -> reject up front.
        let cfg = cfg_from(
            r#"{ "kind":"cooling", "dim":1, "outputs":[0], "params":[1.0],
                 "nodes":[ {"op":"PARAMETER","param_idx":0} ] }"#,
        );
        let err = expect_err(&cfg, &ISO_NEWTONIAN_SPEC);
        assert!(
            err.contains("energy"),
            "expected energy-required rejection, got: {err}"
        );
    }

    #[test]
    fn force_wrong_arity_is_rejected() {
        // force declares one accel component per dim: outputs.len() must == dim.
        let cfg = cfg_from(
            r#"{ "kind":"force", "dim":2, "outputs":[0], "params":[0.5],
                 "nodes":[ {"op":"PARAMETER","param_idx":0} ] }"#,
        );
        let err = expect_err(&cfg, &NEWTONIAN_SPEC);
        assert!(
            err.contains("per dim"),
            "expected arity rejection, got: {err}"
        );
    }

    #[test]
    fn raw_bad_target_is_rejected() {
        let cfg = cfg_from(
            r#"{ "kind":"raw", "dim":1, "outputs":[0], "params":[1.0], "target":"pressure",
                 "nodes":[ {"op":"PARAMETER","param_idx":0} ] }"#,
        );
        let err = expect_err(&cfg, &NEWTONIAN_SPEC);
        assert!(
            err.contains("conserved slot"),
            "expected target rejection, got: {err}"
        );
    }

    #[test]
    fn raw_nrg_on_iso_is_rejected() {
        let cfg = cfg_from(
            r#"{ "kind":"raw", "dim":1, "outputs":[0], "params":[1.0], "target":"nrg",
                 "nodes":[ {"op":"PARAMETER","param_idx":0} ] }"#,
        );
        let err = expect_err(&cfg, &ISO_NEWTONIAN_SPEC);
        assert!(
            err.contains("energy"),
            "expected nrg-needs-energy rejection, got: {err}"
        );
    }

    // ---- region axis ----------------------------------------------

    #[test]
    fn region_masks_the_contribution() {
        // force a = [p0, 0], region chi = x_0 (a linear ramp). the lift is linear, so the masked
        // momentum source is S_mom_0 = rho * (chi * a_0) = rho * x_0 * p0.
        // nodes: 0=param p0, 1=const 0, 2=VARIABLE_X1 (chi). outputs=[0,1], region=2.
        let cfg = cfg_from(
            r#"{ "kind":"force", "dim":2, "outputs":[0,1], "region":2, "params":[0.5],
                 "nodes":[ {"op":"PARAMETER","param_idx":0}, {"op":"CONSTANT","value":0.0},
                           {"op":"VARIABLE_X1"} ] }"#,
        );
        let built = lower(&cfg, &NEWTONIAN_SPEC).expect("force+region");
        let (tgt, mom) = &built[0];
        assert_eq!(tgt, "mom");
        let s_at = |x0: f64| {
            eval_lowered(
                mom,
                mom.outputs()[0],
                &[("rho", 2.0), ("p0", 0.5), ("x_0", x0)],
            )
        };
        assert!(
            s_at(0.0).abs() < 1e-12,
            "region masks to zero where chi = 0: got {}",
            s_at(0.0)
        );
        assert!(
            (s_at(1.0) - 1.0).abs() < 1e-12,
            "full contribution where chi = 1: got {}",
            s_at(1.0)
        );
        assert!(
            (s_at(0.5) - 0.5).abs() < 1e-12,
            "linear in chi: got {}",
            s_at(0.5)
        );
    }

    // ---- relax combine --------------------------------------------

    #[test]
    fn relax_damps_toward_reference_velocity() {
        // relax: outputs = [kappa, v_ref_0, v_ref_1]. kappa=p0, v_ref=[p1, 0].
        // S_mom_0 = max(kappa,0) * rho * (v_ref_0 - vel_0).
        let cfg = cfg_from(
            r#"{ "kind":"relax", "dim":2, "outputs":[0,1,2], "params":[2.0, 0.0],
                 "nodes":[ {"op":"PARAMETER","param_idx":0}, {"op":"PARAMETER","param_idx":1},
                           {"op":"CONSTANT","value":0.0} ] }"#,
        );
        let built = lower(&cfg, &NEWTONIAN_SPEC).expect("relax newtonian");
        assert_eq!(
            built.iter().map(|(t, _)| t.as_str()).collect::<Vec<_>>(),
            ["mom", "nrg"]
        );
        let (_, mom) = &built[0];
        // kappa=2, rho=1, v_ref_0=0, vel_0=3 -> 2*1*(0-3) = -6: the drag opposes the velocity.
        let s_mom0 = eval_lowered(
            mom,
            mom.outputs()[0],
            &[
                ("rho", 1.0),
                ("vel_0", 3.0),
                ("vel_1", 0.0),
                ("p0", 2.0),
                ("p1", 0.0),
            ],
        );
        assert!(
            (s_mom0 - (-6.0)).abs() < 1e-12,
            "relax drag wrong: {s_mom0}"
        );
        // the energy overlay = work = sum vel_k * S_mom_k = 3*(-6) + 0 = -18 < 0: KE is removed.
        let (_, nrg) = &built[1];
        let s_nrg = eval_lowered(
            nrg,
            nrg.outputs()[0],
            &[
                ("rho", 1.0),
                ("vel_0", 3.0),
                ("vel_1", 0.0),
                ("p0", 2.0),
                ("p1", 0.0),
            ],
        );
        assert!(
            s_nrg < 0.0,
            "relaxation must remove kinetic energy, got S_nrg = {s_nrg}"
        );
    }

    #[test]
    fn relax_clamps_negative_rate_to_zero() {
        // the stability invariant: a negative kappa (anti-damping) is clamped to 0 -> no-op, so the
        // flow keeps its energy. the energy-injecting mode is unexpressible by construction.
        let cfg = cfg_from(
            r#"{ "kind":"relax", "dim":1, "outputs":[0,1], "params":[-5.0, 0.0],
                 "nodes":[ {"op":"PARAMETER","param_idx":0}, {"op":"PARAMETER","param_idx":1} ] }"#,
        );
        let built = lower(&cfg, &NEWTONIAN_SPEC).expect("relax");
        let (_, mom) = &built[0];
        // kappa = -5 -> clamped to 0 -> S_mom_0 = 0 regardless of the velocity overshoot.
        let s = eval_lowered(
            mom,
            mom.outputs()[0],
            &[("rho", 1.0), ("vel_0", 7.0), ("p0", -5.0), ("p1", 0.0)],
        );
        assert!(
            s.abs() < 1e-12,
            "negative kappa must clamp to a no-op, got {s}"
        );
    }

    // ---- sponge: full conserved-state relaxation (the buffer zone) -----------------

    #[test]
    fn sponge_relaxes_full_state_toward_reference() {
        // outputs = [kappa, rho_ref, vel_ref_0, vel_ref_1, pre_ref] as constant nodes. the
        // reference is primitive and the regime converts it, so the conserved target is
        //   den = rho_ref = 1, mom = rho_ref vel_ref = [0.5, 0],
        //   nrg = pre_ref/(gamma-1) + rho_ref |vel_ref|^2 / 2 = 3.95*2.5 + 0.125 = 10,
        // which is the same conserved reference this test asserted against when the wire
        // carried it directly. the expected sources below are therefore unchanged: the
        // meaning of the wire moved, the physics did not.
        //   kappa=2, rho_ref=1, vel_ref=[0.5,0], pre_ref=3.95, gamma=1.4.
        let cfg = cfg_from(
            r#"{ "kind":"sponge", "dim":2, "outputs":[0,1,2,3,4], "params":[],
                 "nodes":[ {"op":"CONSTANT","value":2.0}, {"op":"CONSTANT","value":1.0},
                           {"op":"CONSTANT","value":0.5}, {"op":"CONSTANT","value":0.0},
                           {"op":"CONSTANT","value":3.95} ] }"#,
        );
        let law = crate::state_law::StateLaw::newtonian(1.4);
        let built = build_user_source_with_law(&cfg, &NEWTONIAN_SPEC, Some(&law))
            .expect("sponge newtonian");
        assert_eq!(
            built.iter().map(|(t, _)| t.as_str()).collect::<Vec<_>>(),
            ["den", "mom", "nrg"]
        );

        // state: rho=1.5, vel=[3,0], pre=2 (each channel reads only what it needs from this).
        let state = [("rho", 1.5), ("vel_0", 3.0), ("vel_1", 0.0), ("pre", 2.0)];
        // S_den = kappa*(rho_ref - rho) = 2*(1 - 1.5) = -1.0 (density relaxes down toward the ref).
        let (_, den) = &built[0];
        let s_den = eval_lowered(den, den.outputs()[0], &state);
        assert!(
            (s_den - (-1.0)).abs() < 1e-12,
            "sponge density wrong: {s_den}"
        );
        // S_mom_0 = kappa*(rho_ref vel_ref_0 - rho vel_0) = 2*(0.5 - 4.5) = -8.0 (opposes it).
        let (_, mom) = &built[1];
        let s_mom0 = eval_lowered(mom, mom.outputs()[0], &state);
        assert!(
            (s_mom0 - (-8.0)).abs() < 1e-12,
            "sponge mom_0 wrong: {s_mom0}"
        );
        // S_nrg = kappa*(E_ref - E), E = pre/(gamma-1) + 0.5*rho*|v|^2 = 2*2.5 + 0.5*1.5*9 = 11.75;
        //   -> 2*(10 - 11.75) = -3.5 (total energy relaxes down toward the ref).
        let (_, nrg) = &built[2];
        let s_nrg = eval_lowered(nrg, nrg.outputs()[0], &state);
        assert!((s_nrg - (-3.5)).abs() < 1e-12, "sponge nrg wrong: {s_nrg}");
    }

    #[test]
    fn sponge_on_iso_drops_energy_channel() {
        // iso has no energy: the reference is [kappa, rho_ref, vel_ref_0] (2+D), and the den +
        // mom channels are the whole emission — the list stops before pre_ref. the mass and
        // momentum a cold gas carries are the newtonian ones, so the law converts them the
        // same way and only the energy slot, which iso does not evolve, is dropped.
        //
        // this arm also exercises the one-dimensional door: the curved charts start at two
        // spatial dimensions, so a dimension-generic conversion admitting them would take
        // this flat 1d sponge with them.
        let cfg = cfg_from(
            r#"{ "kind":"sponge", "dim":1, "outputs":[0,1,2], "params":[],
                 "nodes":[ {"op":"CONSTANT","value":1.0}, {"op":"CONSTANT","value":2.0},
                           {"op":"CONSTANT","value":0.0} ] }"#,
        );
        let law = crate::state_law::StateLaw::newtonian(5.0 / 3.0);
        let built =
            build_user_source_with_law(&cfg, &ISO_NEWTONIAN_SPEC, Some(&law)).expect("sponge iso");
        assert_eq!(
            built.iter().map(|(t, _)| t.as_str()).collect::<Vec<_>>(),
            ["den", "mom"]
        );
        // S_den = 1*(2 - rho); rho=0.5 -> 1.5 (density relaxes up toward the ref).
        // the conversion evaluates the whole conserved vector, so the velocity is in the
        // signature of every channel built from it; a large value here proves the density
        // channel emits the mass relaxation alone, with no momentum leaking into it.
        let (_, den) = &built[0];
        let s_den = eval_lowered(den, den.outputs()[0], &[("rho", 0.5), ("vel_0", 37.0)]);
        assert!(
            (s_den - 1.5).abs() < 1e-12,
            "iso sponge density wrong: {s_den}"
        );
    }

    #[test]
    fn sponge_relaxes_a_relativistic_state_through_the_relativistic_law() {
        // the seam the primitive wire exists for: the same six outputs a newtonian gas
        // sends, converted by the relativistic law instead. the density a relativistic
        // evolution stores is D = rho W, so a reference at rest and a moving state differ
        // by the lorentz factor -- a newtonian conversion would report the rest densities
        // and relax toward a state the evolution does not store.
        let cfg = cfg_from(
            r#"{ "kind":"sponge", "dim":2, "outputs":[0,1,2,2,3], "params":[],
                 "nodes":[ {"op":"CONSTANT","value":1.0}, {"op":"CONSTANT","value":2.0},
                           {"op":"CONSTANT","value":0.0}, {"op":"CONSTANT","value":1.0} ] }"#,
        );
        let law = crate::state_law::StateLaw::relativistic(
            4.0 / 3.0,
            crate::state_law::Background::Minkowski,
        );
        let built = build_user_source_with_law(&cfg, &RHD_SPEC, Some(&law))
            .expect("relativistic sponge lowers");
        // v = 0.6 gives W = 1.25, so D = 1.25 against a reference D_ref = 2 at rest.
        // S_den = kappa (D_ref - D) = 1*(2 - 1.25) = 0.75.
        let (_, den) = &built[0];
        let s_den = eval_lowered(
            den,
            den.outputs()[0],
            &[("rho", 1.0), ("vel_0", 0.6), ("vel_1", 0.0), ("pre", 0.1)],
        );
        assert!(
            (s_den - 0.75).abs() < 1e-12,
            "relativistic sponge density wrong: {s_den}; a newtonian conversion reports 1.0"
        );
    }

    #[test]
    fn a_massless_curved_sponge_matches_the_flat_one() {
        // the curved law multiplies the conserved state by sqrt(det gamma), which is unity
        // exactly when the hole has no mass. running the same reference and state through
        // both arms at M = 0 therefore has to agree to roundoff, which pins the
        // densitization to the metric rather than to a constant the arm carries.
        let cfg = cfg_from(
            r#"{ "kind":"sponge", "dim":2, "outputs":[0,1,2,2,3], "params":[],
                 "nodes":[ {"op":"CONSTANT","value":1.0}, {"op":"CONSTANT","value":2.0},
                           {"op":"CONSTANT","value":0.0}, {"op":"CONSTANT","value":1.0} ] }"#,
        );
        let state: &[(&str, f64)] = &[
            ("rho", 1.0),
            ("vel_0", 0.6),
            ("vel_1", 0.0),
            ("pre", 0.1),
            ("x_0", 3.0),
            ("x_1", 4.0),
            ("x_2", 0.0),
        ];
        let mut channels = Vec::new();
        for background in [
            crate::state_law::Background::Minkowski,
            crate::state_law::Background::SchwarzschildKsCartesian { mass: 0.0 },
        ] {
            let law = crate::state_law::StateLaw::relativistic(4.0 / 3.0, background);
            let built = build_user_source_with_law(&cfg, &RHD_SPEC, Some(&law))
                .expect("sponge lowers on both backgrounds");
            channels.push(
                built
                    .iter()
                    .map(|(target, source)| {
                        let values: Vec<f64> = source
                            .outputs()
                            .iter()
                            .map(|out| eval_lowered(source, *out, state))
                            .collect();
                        (target.clone(), values)
                    })
                    .collect::<Vec<_>>(),
            );
        }
        assert_eq!(
            channels[0].len(),
            channels[1].len(),
            "channel counts differ"
        );
        for ((target, flat), (_, curved)) in channels[0].iter().zip(channels[1].iter()) {
            for (ii, (a, b)) in flat.iter().zip(curved.iter()).enumerate() {
                assert!(
                    (a - b).abs() < 1e-14,
                    "massless curved sponge differs from flat on {target}[{ii}]: {a} vs {b}"
                );
            }
        }
        // and the massless case is not vacuously zero: the flat arm carries real values.
        assert!(
            channels[0]
                .iter()
                .any(|(_, v)| v.iter().any(|x| x.abs() > 1e-9)),
            "both arms emitted nothing, so the agreement tests nothing"
        );
        // giving the hole mass has to move the answer, or the agreement above would hold
        // for a curved arm that never reached the metric at all. at r = 5 the spatial
        // volume element is sqrt(1 + 2M/r) = 1.183 for M = 1.
        let massive = crate::state_law::StateLaw::relativistic(
            4.0 / 3.0,
            crate::state_law::Background::SchwarzschildKsCartesian { mass: 1.0 },
        );
        let built = build_user_source_with_law(&cfg, &RHD_SPEC, Some(&massive))
            .expect("sponge lowers on a massive hole");
        let (_, den) = &built[0];
        let s_den = eval_lowered(den, den.outputs()[0], state);
        let (_, flat_den) = &channels[0][0];
        assert!(
            (s_den - flat_den[0]).abs() > 1e-3,
            "the mass left the conserved state untouched: {s_den} against the flat {}",
            flat_den[0]
        );
    }

    #[test]
    fn sponge_wrong_arity_is_rejected() {
        // an energy regime needs 3+D outputs (kappa, rho_ref, D vel_ref, pre_ref); a short
        // list fails before anything is lowered, since a missing slot would otherwise be
        // read from whichever output happened to sit at that index.
        let cfg = cfg_from(
            r#"{ "kind":"sponge", "dim":2, "outputs":[0,1,2], "params":[2.5],
                 "nodes":[ {"op":"CONSTANT","value":1.0}, {"op":"CONSTANT","value":1.0},
                           {"op":"CONSTANT","value":0.0} ] }"#,
        );
        let err = expect_err(&cfg, &NEWTONIAN_SPEC);
        assert!(
            err.contains("pre_ref"),
            "expected sponge arity rejection, got: {err}"
        );
    }

    // ---- state variables: density + pressure in user source expressions -------------------

    #[test]
    fn raw_source_reads_density_and_pressure() {
        // a radiative-cooling-style rate S_nrg = -(C * rho * pre): the user expression reads the
        // per-cell state (density + pressure) — the capability that lets adiabatic cooling
        // Lambda(rho, T), T = pre/rho, be user-defined. nodes: 0=param C, 1=VARIABLE_RHO,
        // 2=VARIABLE_PRESSURE, 3=mul(C,rho), 4=mul(3,pre), 5=neg(4). outputs=[5], target=nrg.
        let cfg = cfg_from(
            r#"{ "kind":"raw", "dim":1, "outputs":[5], "params":[0.25], "target":"nrg",
                 "nodes":[ {"op":"PARAMETER","param_idx":0}, {"op":"VARIABLE_RHO"},
                           {"op":"VARIABLE_PRESSURE"}, {"op":"MULTIPLY","left":0,"right":1},
                           {"op":"MULTIPLY","left":3,"right":2}, {"op":"NEG","left":4} ] }"#,
        );
        let built = lower(&cfg, &NEWTONIAN_SPEC).expect("raw pressure-reading cooling");
        let (tgt, nrg) = &built[0];
        assert_eq!(tgt, "nrg");
        // S_nrg = -(C * rho * pre); C=0.25, rho=2, pre=3 -> -(0.25*2*3) = -1.5.
        let s = eval_lowered(
            nrg,
            nrg.outputs()[0],
            &[("p0", 0.25), ("rho", 2.0), ("pre", 3.0)],
        );
        assert!(
            (s - (-1.5)).abs() < 1e-12,
            "cooling rate must read rho*pre: got {s}"
        );
        // it genuinely depends on pressure: doubling pre doubles the rate.
        let s2 = eval_lowered(
            nrg,
            nrg.outputs()[0],
            &[("p0", 0.25), ("rho", 2.0), ("pre", 6.0)],
        );
        assert!(
            (s2 - (-3.0)).abs() < 1e-12,
            "rate must scale with pressure: got {s2}"
        );
    }

    #[test]
    fn raw_source_targets_density_slot() {
        // a density-only injection (pure mass loading) — a baryon source rate S_den = p0 * rho
        // written straight to the `den` slot. the single-slot path for mass loading, where
        // `inject` carries the full conserved vector.
        let cfg = cfg_from(
            r#"{ "kind":"raw", "dim":1, "outputs":[2], "params":[0.5], "target":"den",
                 "nodes":[ {"op":"PARAMETER","param_idx":0}, {"op":"VARIABLE_RHO"},
                           {"op":"MULTIPLY","left":0,"right":1} ] }"#,
        );
        let built = lower(&cfg, &NEWTONIAN_SPEC).expect("raw den ok");
        assert_eq!(
            built.iter().map(|(t, _)| t.as_str()).collect::<Vec<_>>(),
            ["den"]
        );
        let (tgt, den) = &built[0];
        assert_eq!(tgt, "den");
        // S_den = p0 * rho; p0=0.5, rho=2 -> 1.
        let s = eval_lowered(den, den.outputs()[0], &[("p0", 0.5), ("rho", 2.0)]);
        assert!(
            (s - 1.0).abs() < 1e-12,
            "raw den rate must be p0*rho: got {s}"
        );
    }

    #[test]
    fn relax_on_iso_drops_energy_overlay() {
        // iso has no energy: relax yields the momentum drag alone.
        let cfg = cfg_from(
            r#"{ "kind":"relax", "dim":1, "outputs":[0,1], "params":[1.0, 0.0],
                 "nodes":[ {"op":"PARAMETER","param_idx":0}, {"op":"PARAMETER","param_idx":1} ] }"#,
        );
        let built = lower(&cfg, &ISO_NEWTONIAN_SPEC).expect("relax iso");
        assert_eq!(
            built.iter().map(|(t, _)| t.as_str()).collect::<Vec<_>>(),
            ["mom"]
        );
    }

    #[test]
    fn relax_on_relativistic_is_rejected() {
        let cfg = cfg_from(
            r#"{ "kind":"relax", "dim":1, "outputs":[0,1], "params":[1.0, 0.0],
                 "nodes":[ {"op":"PARAMETER","param_idx":0}, {"op":"PARAMETER","param_idx":1} ] }"#,
        );
        let err = expect_err(&cfg, &RHD_SPEC);
        assert!(
            err.contains("relativistic"),
            "expected relativistic rejection, got: {err}"
        );
    }

    #[test]
    fn relax_wrong_arity_is_rejected() {
        // relax needs [kappa, v_ref_0..v_ref_{dim-1}] = 1 + dim outputs.
        let cfg = cfg_from(
            r#"{ "kind":"relax", "dim":2, "outputs":[0,1], "params":[1.0, 0.0],
                 "nodes":[ {"op":"PARAMETER","param_idx":0}, {"op":"PARAMETER","param_idx":1} ] }"#,
        );
        let err = expect_err(&cfg, &NEWTONIAN_SPEC);
        assert!(
            err.contains("v_ref"),
            "expected relax arity rejection, got: {err}"
        );
    }

    // ---- driven boundaries -----------------------------------------------

    fn expect_boundary_err(
        cfg: &symbi_expr::SourceConfig,
        spec: &crate::regime_spec::RegimeSpec,
    ) -> String {
        match build_boundary_dag(cfg, spec) {
            Err(e) => e,
            Ok(_) => panic!("expected build_boundary_dag to reject the config"),
        }
    }

    #[test]
    fn boundary_dirichlet_splits_into_prim_slots() {
        // newtonian 2D prescribes [rho, vel_0, vel_1, pre] -> slots den(1), mom(2 = the vel vector),
        // nrg(1). the structural-slot names map to prim.rho / prim.vel_k / prim.pre at assign.
        let cfg = cfg_from(
            r#"{ "kind":"dirichlet", "dim":2, "outputs":[0,1,2,3], "params":[],
                 "nodes":[ {"op":"CONSTANT","value":1.0}, {"op":"CONSTANT","value":0.5},
                           {"op":"CONSTANT","value":0.0}, {"op":"CONSTANT","value":2.0} ] }"#,
        );
        let built = build_boundary_dag(&cfg, &NEWTONIAN_SPEC).expect("driven boundary");
        let slots: Vec<(&str, usize)> = built
            .iter()
            .map(|(s, b)| (s.as_str(), b.outputs().len()))
            .collect();
        assert_eq!(slots, vec![("den", 1), ("mom", 2), ("nrg", 1)]);
    }

    #[test]
    fn boundary_on_iso_drops_pressure() {
        // iso has no energy: the prim state is exactly [rho, vel].
        let cfg = cfg_from(
            r#"{ "kind":"dirichlet", "dim":1, "outputs":[0,1], "params":[],
                 "nodes":[ {"op":"CONSTANT","value":1.0}, {"op":"CONSTANT","value":0.5} ] }"#,
        );
        let built = build_boundary_dag(&cfg, &ISO_NEWTONIAN_SPEC).expect("iso driven boundary");
        let slots: Vec<&str> = built.iter().map(|(s, _)| s.as_str()).collect();
        assert_eq!(slots, vec!["den", "mom"]);
    }

    #[test]
    fn boundary_on_rhd_is_allowed() {
        // a prim prescription is regime-agnostic across hydro, so rhd (relativistic) is accepted,
        // where force/cooling are rejected because their newtonian conservation law is wrong for rhd.
        let cfg = cfg_from(
            r#"{ "kind":"dirichlet", "dim":1, "outputs":[0,1,2], "params":[],
                 "nodes":[ {"op":"CONSTANT","value":1.0}, {"op":"CONSTANT","value":0.1},
                           {"op":"CONSTANT","value":1.0} ] }"#,
        );
        assert!(build_boundary_dag(&cfg, &RHD_SPEC).is_ok());
    }

    #[test]
    fn boundary_on_mhd_prescribes_cell_b() {
        // RMHD 2.5D (dim=3 vector components): a driven boundary prescribes the full prim
        // [rho, v_0..v_2, pre, B_0..B_2] -> slots den(1), mom(3), nrg(1), bcell(3). a purely
        // toroidal injection sets B_0=B_1=0 (in-plane), B_2=B_phi (out-of-plane, the safe case).
        let cfg = cfg_from(
            r#"{ "kind":"dirichlet", "dim":3,
                 "outputs":[0,1,2,3,4,5,6,7], "params":[],
                 "nodes":[ {"op":"CONSTANT","value":1.0},
                           {"op":"CONSTANT","value":0.1}, {"op":"CONSTANT","value":0.0},
                           {"op":"CONSTANT","value":0.0}, {"op":"CONSTANT","value":1.0},
                           {"op":"CONSTANT","value":0.0}, {"op":"CONSTANT","value":0.0},
                           {"op":"CONSTANT","value":0.5} ] }"#,
        );
        let built = build_boundary_dag(&cfg, &RMHD_SPEC).expect("toroidal driven boundary");
        let slots: Vec<(&str, usize)> = built
            .iter()
            .map(|(s, b)| (s.as_str(), b.outputs().len()))
            .collect();
        assert_eq!(
            slots,
            vec![("den", 1), ("mom", 3), ("nrg", 1), ("bcell", 3)]
        );
    }

    #[test]
    fn boundary_mhd_wrong_arity_is_rejected() {
        // RMHD dim=3 needs 8 prim outputs (rho + 3 vel + pre + 3 B); give 5 (the hydro count).
        let cfg = cfg_from(
            r#"{ "kind":"dirichlet", "dim":3, "outputs":[0,1,2,3,4], "params":[],
                 "nodes":[ {"op":"CONSTANT","value":1.0}, {"op":"CONSTANT","value":0.1},
                           {"op":"CONSTANT","value":0.0}, {"op":"CONSTANT","value":0.0},
                           {"op":"CONSTANT","value":1.0} ] }"#,
        );
        let err = expect_boundary_err(&cfg, &RMHD_SPEC);
        assert!(err.contains("full prim state"), "got: {err}");
    }

    #[test]
    fn boundary_wrong_arity_is_rejected() {
        // newtonian 2D needs 4 prim outputs (rho + 2 vel + pre); give 3.
        let cfg = cfg_from(
            r#"{ "kind":"dirichlet", "dim":2, "outputs":[0,1,2], "params":[],
                 "nodes":[ {"op":"CONSTANT","value":1.0}, {"op":"CONSTANT","value":0.5},
                           {"op":"CONSTANT","value":0.0} ] }"#,
        );
        let err = expect_boundary_err(&cfg, &NEWTONIAN_SPEC);
        assert!(err.contains("full prim state"), "got: {err}");
    }

    /// evaluate one output of a lowered SourceProgram on the CPU interpreter, binding params
    /// by the declared (name -> value) map in manifest order.
    fn eval_lowered(built: &SourceProgram, output: NodeId, values: &[(&str, f64)]) -> f64 {
        let lowered = scalarize(&built.graph(), output, "expr_bridge");
        let inputs: Vec<f64> = built
            .params()
            .iter()
            .map(|p| {
                values
                    .iter()
                    .find(|(n, _)| *n == p.as_str())
                    .map(|(_, v)| *v)
                    .unwrap_or_else(|| panic!("eval_lowered: missing param '{p}'"))
            })
            .collect();
        Cpu.eval_elemental(&lowered, &inputs)[0]
    }

    /// the load-bearing bridge proof: a mixed user expression, lowered to the symbi-ir
    /// Graph, evaluates identically to `symbi-expr`'s own register-VM interpreter — proving
    /// the two engines unify on one IR. exercises arithmetic, a transcendental, a parameter,
    /// variables, time, a comparison, and the IF_THEN_ELSE -> Select lowering.
    #[test]
    fn lowered_graph_matches_symbi_expr_vm() {
        // f = if x1 > p0 then sin(x1)*2 + x2 else exp(-t)
        let mut dag = Dag::new();
        let x1 = dag.var_x1();
        let x2 = dag.var_x2();
        let t = dag.var_t();
        let p0 = dag.param(0);
        let two = dag.constant(2.0);
        let cond = dag.gt(x1, p0);
        let sinx = dag.sin(x1);
        let term = dag.mul(sinx, two);
        let then_ = dag.add(term, x2);
        let negt = dag.neg(t);
        let else_ = dag.exp(negt);
        let root = dag.if_then_else(cond, then_, else_);

        let nodes = dag.nodes().to_vec();
        let built = lower_dag_to_program(&nodes, &[root]).expect("bridge lowers");

        let mut expr = dag.compile(&[root]);

        // two states: one taking the `then` branch (x1 > p0), one taking `else`.
        for (x1v, x2v, tv, p0v) in [(1.0, 0.7, 0.3, 0.5), (0.2, -0.4, 1.1, 0.5)] {
            expr.set_params(&[p0v]);
            let want = expr.eval(x1v, x2v, x3_unused(), tv)[0];
            let got = eval_lowered(
                &built,
                built.outputs()[0],
                &[("x_0", x1v), ("x_1", x2v), ("t", tv), ("p0", p0v)],
            );
            assert!(
                (want - got).abs() < 1e-12,
                "bridge != VM at (x1={x1v}, x2={x2v}, t={tv}, p0={p0v}): VM={want}, IR={got}",
            );
        }
    }

    fn x3_unused() -> f64 {
        0.0
    }

    #[test]
    fn unsupported_ops_are_rejected_not_miscompiled() {
        // `Sgn` / `Mod` fall outside the carrier-traceable primitives — the bridge must reject them.
        let mut dag = Dag::new();
        let x1 = dag.var_x1();
        let sgn = dag.unary(Op::Sgn, x1);
        let nodes = dag.nodes().to_vec();
        let result = lower_dag_to_program(&nodes, &[sgn]);
        assert!(
            matches!(result, Err(BridgeError::UnsupportedOp(Op::Sgn))),
            "Sgn must be rejected as unsupported",
        );
    }
}

#[cfg(test)]
mod state_law_seam_tests {
    use super::*;
    use crate::state_law::{Background, StateLaw};

    fn force_cfg() -> symbi_expr::SourceConfig {
        symbi_expr::SourceConfig::from_json(
            r#"{"kind":"force","dim":1,"outputs":[0],"params":[0.0],
                "nodes":[{"op":"PARAMETER","param_idx":0}]}"#,
        )
        .expect("parse")
    }

    #[test]
    fn a_law_disagreeing_with_the_regime_is_refused_at_build_time() {
        // the two describe the same regime from opposite ends. a relativistic law on a
        // newtonian regime would have a source relax toward `rho h W^2 v` while the
        // evolution stores `rho v` — a wrong answer no output reveals, so it is caught
        // where the mismatch is made rather than carried into the graph.
        let law = StateLaw::relativistic(4.0 / 3.0, Background::Minkowski);
        let err = match build_user_source_with_law(&force_cfg(), &crate::NEWTONIAN_SPEC, Some(&law))
        {
            Err(e) => e,
            Ok(_) => panic!("a relativistic law on a newtonian regime must be refused"),
        };
        assert!(err.contains("relativistic"), "unhelpful message: {err}");
        assert!(err.contains("newtonian"), "unhelpful message: {err}");
    }

    #[test]
    fn a_matching_law_lowers_exactly_as_no_law_does() {
        // threading the law changes nothing for the kinds that do not read it, so every
        // existing source keeps its graph. the sponge is what will consume it.
        let law = StateLaw::newtonian(5.0 / 3.0);
        let with = build_user_source_with_law(&force_cfg(), &crate::NEWTONIAN_SPEC, Some(&law))
            .expect("lowers with a law");
        let without =
            build_user_source(&force_cfg(), &crate::NEWTONIAN_SPEC).expect("lowers without one");
        assert_eq!(with.len(), without.len());
        for ((ta, a), (tb, b)) in with.iter().zip(&without) {
            assert_eq!(ta, tb, "target moved");
            assert_eq!(
                a.outputs().len(),
                b.outputs().len(),
                "output count moved for {ta}"
            );
        }
    }
}
