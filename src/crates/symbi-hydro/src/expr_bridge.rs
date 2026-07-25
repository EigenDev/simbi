// =============================================================================
// expr_bridge.rs
//
// the bridge: lower a `symbi-expr` DAG (a parsed user script) into the `symbi-ir`
// Graph — the ONE typed IR the physics is built on. a user expression becomes a
// `BuiltSource` (graph + named params + output NodeIds), structurally identical to a
// built-in `SourceSpec`'s output, so it rides the EXISTING splice/fuse path unchanged:
// `splice_built_source_into` fuses it into a godunov / ghost-fill kernel, which then
// codegens (CPU + CUDA) or interprets. this collapses the two parallel expression
// engines (`symbi-expr`'s register-VM and `symbi-ir`'s codegen substrate) into one IR
// that codegens directly, with no per-cell VM interpreter.
//
// the leaf convention matches the source vocabulary so a user expression fuses like any
// other source: VariableX{1,2,3} -> `x_0/x_1/x_2` (the cell position, bound to the
// centroid at splice), VariableT -> `t` (time), Parameter(i) -> `p{i}` (runtime scalar).
//
// the typed IR enforces carrier-traceability: a user `IF_THEN_ELSE` lowers to `Select` (no
// native branch — the carrier dialect), comparisons produce a Bool consumed only by a
// conditional, and ops with no carrier-traceable equivalent (`Sgn`, `Mod`) are REJECTED
// at bridge time, so they cannot silently miscompile.
//
// usage:
//   let nodes = dag.nodes().to_vec();
//   let built = lower_dag_to_builtsource(&nodes, &[root])?;   // -> BuiltSource
// =============================================================================

use std::collections::HashMap;

use symbi_expr::dag::{Node, Payload};
use symbi_expr::op::Op;
use symbi_ir::graph::{ConstValue, ElementWiseOp, Graph, NodeId, TranscendentalOp};
use symbi_ir::ElementTy;

use crate::source_spec::BuiltSource;

/// a `symbi-expr` op has no `symbi-ir` equivalent or the DAG is malformed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BridgeError {
    /// op carries no carrier-traceable `symbi-ir` primitive (`Sgn`, `Mod`). rejected
    /// at bridge time — the user gets a loud error, and a silently wrong kernel is never emitted.
    UnsupportedOp(Op),
    /// a node references a child index not yet defined (forward ref / out of range).
    /// well-formed DAGs are topologically ordered, so this means malformed input.
    BadChild { node: usize, child: usize },
    /// payload shape disagrees with the op's arity (malformed input).
    BadPayload { node: usize, op: Op },
}

/// how a `symbi-expr` op maps onto the `symbi-ir` algebra. leaves + `IfThenElse` are
/// handled directly in the walk; this covers the arithmetic / transcendental interior.
enum Mapped {
    Elem(ElementWiseOp),
    Trans(TranscendentalOp),
}

fn map_op(op: Op) -> Option<Mapped> {
    use Op::*;
    Some(match op {
        // arithmetic + comparison + logical + min/max/abs/sqrt/floor/ceil -> ElementWise.
        Add => Mapped::Elem(ElementWiseOp::Add),
        Sub => Mapped::Elem(ElementWiseOp::Sub),
        Mul => Mapped::Elem(ElementWiseOp::Mul),
        Div => Mapped::Elem(ElementWiseOp::Div),
        Pow => Mapped::Elem(ElementWiseOp::Pow),
        Neg => Mapped::Elem(ElementWiseOp::Neg),
        Abs => Mapped::Elem(ElementWiseOp::Abs),
        Sqrt => Mapped::Elem(ElementWiseOp::Sqrt),
        Ceil => Mapped::Elem(ElementWiseOp::Ceil),
        Floor => Mapped::Elem(ElementWiseOp::Floor),
        Min => Mapped::Elem(ElementWiseOp::Min),
        Max => Mapped::Elem(ElementWiseOp::Max),
        Lt => Mapped::Elem(ElementWiseOp::Lt),
        Gt => Mapped::Elem(ElementWiseOp::Gt),
        Eq => Mapped::Elem(ElementWiseOp::Eq),
        Le => Mapped::Elem(ElementWiseOp::Le),
        Ge => Mapped::Elem(ElementWiseOp::Ge),
        And => Mapped::Elem(ElementWiseOp::BitAnd),
        Or => Mapped::Elem(ElementWiseOp::BitOr),
        Not => Mapped::Elem(ElementWiseOp::BitNot),
        // math functions -> Transcendental (the complete set; ElementWise lacks Tan/Exp/Log/...).
        Log => Mapped::Trans(TranscendentalOp::Log),
        Log10 => Mapped::Trans(TranscendentalOp::Log10),
        Exp => Mapped::Trans(TranscendentalOp::Exp),
        Sin => Mapped::Trans(TranscendentalOp::Sin),
        Cos => Mapped::Trans(TranscendentalOp::Cos),
        Tan => Mapped::Trans(TranscendentalOp::Tan),
        Asin => Mapped::Trans(TranscendentalOp::Asin),
        Acos => Mapped::Trans(TranscendentalOp::Acos),
        Atan => Mapped::Trans(TranscendentalOp::Atan),
        Atan2 => Mapped::Trans(TranscendentalOp::Atan2),
        Sinh => Mapped::Trans(TranscendentalOp::Sinh),
        Cosh => Mapped::Trans(TranscendentalOp::Cosh),
        Tanh => Mapped::Trans(TranscendentalOp::Tanh),
        Asinh => Mapped::Trans(TranscendentalOp::Asinh),
        Acosh => Mapped::Trans(TranscendentalOp::Acosh),
        Atanh => Mapped::Trans(TranscendentalOp::Atanh),
        // leaves / ternary handled in the walk; Sgn + Mod have no carrier primitive.
        Constant | VariableX1 | VariableX2 | VariableX3 | VariableT | Parameter
        | VariableRho | VariableVel1 | VariableVel2 | VariableVel3 | VariablePressure
        | IfThenElse | Sgn | Mod => return None,
    })
}

/// the leaf-param name for a `symbi-expr` variable op, matching the source vocabulary
/// (`source_params::x` + `t`). non-variable / non-time ops return `None`.
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
        _ => None,
    }
}

/// get-or-declare a named scalar param leaf, recording first-seen order in `params` so
/// repeated uses (e.g., `x_0` twice) share ONE leaf — the runtime fills one scalar.
fn declare_param(
    g: &mut Graph,
    name: &str,
    cache: &mut HashMap<String, NodeId>,
    params: &mut Vec<String>,
) -> NodeId {
    if let Some(&id) = cache.get(name) {
        return id;
    }
    let id = g.add_scalar_param(name, ElementTy::F64);
    cache.insert(name.to_string(), id);
    params.push(name.to_string());
    id
}

/// translate a child index through the partially-built node map, rejecting forward refs.
fn child(map: &[NodeId], node: usize, c: usize) -> Result<NodeId, BridgeError> {
    map.get(c).copied().ok_or(BridgeError::BadChild { node, child: c })
}

/// lower a `symbi-expr` DAG (`nodes`, in topological order) with the given `outputs`
/// into a `BuiltSource` over the `symbi-ir` Graph. returns the params in first-seen
/// declaration order (the splice/runtime binding order).
pub fn lower_dag_to_builtsource(
    nodes: &[Node],
    outputs: &[usize],
) -> Result<BuiltSource, BridgeError> {
    let mut g = Graph::new();
    let mut node_map: Vec<NodeId> = Vec::with_capacity(nodes.len());
    let mut param_cache: HashMap<String, NodeId> = HashMap::new();
    let mut params: Vec<String> = Vec::new();

    for (i, node) in nodes.iter().enumerate() {
        let id = match node.op {
            Op::Constant => match node.payload {
                Payload::Value(v) => g.add_const(ConstValue::F64(v), None),
                _ => return Err(BridgeError::BadPayload { node: i, op: node.op }),
            },
            Op::VariableX1 | Op::VariableX2 | Op::VariableX3 | Op::VariableT
            | Op::VariableRho | Op::VariableVel1 | Op::VariableVel2 | Op::VariableVel3
            | Op::VariablePressure => {
                let name = variable_name(node.op).expect("variable op has a name");
                declare_param(&mut g, name, &mut param_cache, &mut params)
            }
            Op::Parameter => match node.payload {
                Payload::ParamIdx(idx) => {
                    declare_param(&mut g, &format!("p{idx}"), &mut param_cache, &mut params)
                }
                _ => return Err(BridgeError::BadPayload { node: i, op: node.op }),
            },
            Op::IfThenElse => match node.payload {
                Payload::Ternary(c, t, e) => g.select(
                    child(&node_map, i, c)?,
                    child(&node_map, i, t)?,
                    child(&node_map, i, e)?,
                    None,
                ),
                _ => return Err(BridgeError::BadPayload { node: i, op: node.op }),
            },
            other => {
                let mapped = map_op(other).ok_or(BridgeError::UnsupportedOp(other))?;
                let args: Vec<NodeId> = match node.payload {
                    Payload::Unary(c) => vec![child(&node_map, i, c)?],
                    Payload::Binary(l, r) => {
                        vec![child(&node_map, i, l)?, child(&node_map, i, r)?]
                    }
                    _ => return Err(BridgeError::BadPayload { node: i, op: node.op }),
                };
                match mapped {
                    Mapped::Elem(op) => g.element_wise(op, args, None),
                    Mapped::Trans(op) => g.transcendental(op, args, None),
                }
            }
        };
        node_map.push(id);
    }

    let out_nodes: Result<Vec<NodeId>, BridgeError> = outputs
        .iter()
        .map(|&o| node_map.get(o).copied().ok_or(BridgeError::BadChild { node: usize::MAX, child: o }))
        .collect();
    Ok(BuiltSource { graph: g, params, outputs: out_nodes? })
}

/// THE FRONT DOOR: turn a serialized `SourceConfig` (python -> json -> `SourceConfig::from_json`)
/// into the axiomatic `BuiltSource`(s), wrapping the user's free field in the conservation law
/// per `kind`. returns `(target_field, BuiltSource)` pairs ready to splice into the godunov source
/// dispatch — `"force"` yields momentum (+ energy, if the regime has it), `"cooling"` yields a
/// lone energy sink, `"inject"` writes the full conserved vector [den, mom, (nrg)] additively from
/// one config (mass+momentum+energy deposition), `"raw"` writes the outputs straight to a single
/// `target` (the escape hatch). the user never authors conservative components for force/cooling,
/// so the coupling is unbreakable.
///
/// VALIDATION is regime-driven via `spec` (the static `RegimeSpec`), so an ill-posed config fails
/// HERE — before attach, never as a mid-evolve panic:
/// - `force`/`cooling`/`relax` carry NEWTONIAN conservation laws; they are REJECTED for a
///   relativistic regime (whose conserved momentum is `rho h W^2 v`, a different law). a
///   relativistic regime must use `raw` (the user supplies conserved components).
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
/// contribution is multiplied by it. the conservation lifts are LINEAR in the field, so masking the
/// field (for `relax`: only the rate `kappa`) equals masking the conserved contribution — no splice
/// change, CPU + GPU fall out of the existing path.
pub fn build_user_source(
    cfg: &symbi_expr::SourceConfig,
    spec: &crate::regime_spec::RegimeSpec,
) -> Result<Vec<(String, BuiltSource)>, String> {
    let nodes = symbi_expr::nodes_from_descs(&cfg.nodes).map_err(|e| format!("dag load: {e}"))?;
    // lower the field once. if a region mask is declared, lower it as an EXTRA output in the SAME
    // graph (so chi shares the field's leaves), then peel it off to mask with below.
    let mut lower_outputs = cfg.outputs.clone();
    if let Some(r) = cfg.region {
        lower_outputs.push(r);
    }
    // constant-power strength reduction (symbi_expr::strength): `x ** (-2.0)` in a
    // user config becomes a multiply/divide chain, avoiding a per-cell libm pow.
    let (nodes, lower_outputs) = symbi_expr::strength_reduce(&nodes, &lower_outputs);
    let mut field =
        lower_dag_to_builtsource(&nodes, &lower_outputs).map_err(|e| format!("bridge: {e:?}"))?;
    let region: Option<NodeId> = cfg.region.map(|_| field.outputs.pop().expect("region output"));
    let n_out = cfg.outputs.len();

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
            mask_field(&mut field, region, 0..cfg.dim); // mask the whole acceleration vector
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
        "cooling" => {
            reject_relativistic("cooling")?;
            if !spec.has_energy {
                return Err(format!(
                    "'cooling' requires an energy equation; regime '{}' has none (isothermal)",
                    spec.name,
                ));
            }
            if n_out != 1 {
                return Err(format!("'cooling' is a single rate: outputs.len() = {n_out}, expected 1"));
            }
            mask_field(&mut field, region, 0..1);
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
            // region masks ONLY the rate kappa (output 0) — masking v_ref would corrupt the target.
            mask_field(&mut field, region, 0..1);
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
            // full conserved-state relaxation (buffer zone): S_U = kappa*(U_ref - U) for den, mom
            // (+ nrg on energy regimes). the reference conserved state U_ref is supplied per-cell.
            reject_relativistic("sponge")?;
            // adiabatic: [kappa, den_ref, mom_ref_0..mom_ref_{D-1}, nrg_ref] = 3+D; iso drops nrg_ref.
            let want = if spec.has_energy { 3 + cfg.dim } else { 2 + cfg.dim };
            if n_out != want {
                return Err(format!(
                    "'sponge' needs [kappa, den_ref, mom_ref_0..mom_ref_{}{}]: outputs.len() = {n_out}, expected {want}",
                    cfg.dim.saturating_sub(1),
                    if spec.has_energy { ", nrg_ref" } else { "" },
                ));
            }
            // region masks ONLY the rate kappa (output 0), which factors into all three channels;
            // masking the reference state would corrupt the target the flow relaxes toward.
            mask_field(&mut field, region, 0..1);
            let mut out = vec![
                (
                    "den".to_string(),
                    crate::source_spec::user_sponge_density_source(&field, cfg.dim),
                ),
                (
                    "mom".to_string(),
                    crate::source_spec::user_sponge_momentum_source(&field, cfg.dim),
                ),
            ];
            if spec.has_energy {
                // inv_gm1 = 1/(gamma-1): the ideal-gas internal-energy coefficient, folded as a
                // build-time constant so the energy channel reconstructs E from `pre` without a
                // runtime gamma binding.
                let inv_gm1 = *cfg.params.first().ok_or_else(|| {
                    "'sponge' on an energy regime needs params=[inv_gm1] = 1/(gamma-1)".to_string()
                })?;
                out.push((
                    "nrg".to_string(),
                    crate::source_spec::user_sponge_energy_source(&field, cfg.dim, inv_gm1),
                ));
            }
            Ok(out)
        }
        "inject" => {
            // additive deposition of the FULL conserved vector in one config: outputs =
            // [S_den, S_mom_0..S_mom_{D-1}, (S_nrg on energy regimes)], each written straight to
            // its conserved slot (identity, like `raw`, but spanning every slot at once). this is
            // the mass+momentum+energy injection (a jet/wind depositing all three) that a
            // single-slot `raw` cannot express. relativistic-safe: the user supplies conserved
            // components directly (no newtonian law wrap), so no `reject_relativistic` — the
            // components ARE the regime's conserved rates (D=rho*W, S=rho*h*W^2*v, tau in rhd).
            let want = if spec.has_energy { 2 + cfg.dim } else { 1 + cfg.dim };
            if n_out != want {
                return Err(format!(
                    "'inject' needs [den, mom_0..mom_{}{}]: outputs.len() = {n_out}, expected {want}",
                    cfg.dim.saturating_sub(1),
                    if spec.has_energy { ", nrg" } else { "" },
                ));
            }
            mask_field(&mut field, region, 0..n_out); // region masks every conserved channel
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
            mask_field(&mut field, region, 0..n_out); // mask every conserved component the user wrote
            Ok(vec![(target, field)])
        }
        other => {
            Err(format!("unknown source kind '{other}' (expected force | cooling | relax | sponge | inject | raw)"))
        }
    }
}

/// lower an ordered collection of user sources into one runtime source set.
/// parameter indices are made global before lowering, and contributions that
/// target the same conserved field are summed component by component.
pub fn build_user_sources(
    configs: &[symbi_expr::SourceConfig],
    spec: &crate::regime_spec::RegimeSpec,
) -> Result<(Vec<(String, BuiltSource)>, Vec<f64>), String> {
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
        lowered.extend(build_user_source(&config, spec)?);
    }

    let mut composed: Vec<(String, BuiltSource)> = Vec::new();
    for (target, built) in lowered {
        if let Some((_, existing)) = composed.iter_mut().find(|(name, _)| name == &target) {
            *existing = sum_built_sources(existing, &built, &target)?;
        } else {
            composed.push((target, built));
        }
    }
    Ok((composed, params))
}

fn sum_built_sources(
    left: &BuiltSource,
    right: &BuiltSource,
    target: &str,
) -> Result<BuiltSource, String> {
    if left.outputs.len() != right.outputs.len() {
        return Err(format!(
            "source target '{target}' has incompatible component counts: {} and {}",
            left.outputs.len(),
            right.outputs.len()
        ));
    }

    let mut graph = Graph::new();
    let mut params = left.params.clone();
    for name in &right.params {
        if !params.contains(name) {
            params.push(name.clone());
        }
    }
    let leaves: HashMap<String, NodeId> = params
        .iter()
        .map(|name| {
            let node = graph.add_scalar_param(name, ElementTy::F64);
            (name.clone(), node)
        })
        .collect();
    let resolve = |symbol: &symbi_ir::Symbol| leaves.get(symbol.as_str()).copied();
    let outputs = left
        .outputs
        .iter()
        .zip(&right.outputs)
        .map(|(&left_root, &right_root)| {
            let left_node = graph.import_subgraph(&left.graph, left_root, resolve);
            let right_node = graph.import_subgraph(&right.graph, right_root, resolve);
            graph.element_wise(ElementWiseOp::Add, vec![left_node, right_node], None)
        })
        .collect();

    Ok(BuiltSource { graph, params, outputs })
}

/// multiply the named `field` outputs by the region mask `chi` (in the field's own graph), if a
/// region is present. the lifts are linear in these outputs, so this masks the final conserved
/// contribution. `idxs` selects WHICH outputs carry the maskable quantity (e.g., relax masks only
/// the rate `kappa`, leaving the reference velocity unmasked).
fn mask_field(field: &mut BuiltSource, chi: Option<NodeId>, idxs: std::ops::Range<usize>) {
    let Some(chi) = chi else { return };
    for i in idxs {
        let masked = field.graph.element_wise(ElementWiseOp::Mul, vec![field.outputs[i], chi], None);
        field.outputs[i] = masked;
    }
}

/// THE BOUNDARY FRONT DOOR: compile a `SourceConfig` into a DRIVEN-BOUNDARY
/// prescription — a complete primitive state `[rho, vel_0..vel_{D-1}, pre]` the ghost cells are SET
/// to (Dirichlet), `combine = overwrite`. returns `(slot, BuiltSource)` in the structural-slot
/// convention `den`/`mom`/`nrg` that [`symbi_discretize::boundary_fill_from_built_gv`] writes to
/// `prim.rho`/`prim.vel_k`/`prim.pre`. each slot is an INDEPENDENT lowering of the user DAG over its
/// output subset, so the velocity vector lands as the `ncomp`-output `mom` slot.
///
/// VALIDATION (regime-driven via `spec`, at attach — never mid-evolve):
/// - **regime-agnostic across hydro AND mhd.** a prim prescription sets `prim` (no conservation
///   law), valid for RHD too (`is_relativistic` ALLOWED). MHD additionally prescribes the CELL-B
///   vector (`bcell` slot -> `prim.mag`): the OUT-OF-PLANE component (B_phi in a 2.5D axisymmetric
///   grid) is cell-centered + flux-evolved, so prescribing it is a plain Dirichlet; the CT
///   tangential-EMF sub-problem arises only for a prescribed POLOIDAL/in-plane FACE field.
///   the in-plane cell-B components are the user's responsibility to keep div-compatible (=0 for a
///   purely toroidal field) — `raw`-style: garbage in, garbage out.
/// - **complete prim state.** the DAG must output exactly the regime's primitive components:
///   `1 (rho) + d (vel) + has_energy (pre) + is_mhd*d (cell B)` where `d = cfg.dim` is the vector
///   component count (= DOF). a partial prescription is rejected.
/// (the `dim == sim DOF` cross-check is the const-generic dispatch's, where DOF is known.)
pub fn build_boundary_dag(
    cfg: &symbi_expr::SourceConfig,
    spec: &crate::regime_spec::RegimeSpec,
) -> Result<Vec<(String, BuiltSource)>, String> {
    let nodes = symbi_expr::nodes_from_descs(&cfg.nodes).map_err(|e| format!("dag load: {e}"))?;
    let (nodes, reduced_outputs) = symbi_expr::strength_reduce(&nodes, &cfg.outputs);
    let d = cfg.dim;
    // MHD prescribes the cell-B vector too. the OUT-OF-PLANE component (B_phi in a 2.5D
    // axisymmetric grid: cell-centered, flux-evolved) is the safe toroidal case — div-free
    // by axisymmetry. the IN-PLANE components are the user's responsibility to keep
    // div-compatible (=0 for a purely toroidal field); they are NOT a CT face prescription
    // here, so no tangential-EMF sub-problem — that constraint only applies to
    // a prescribed POLOIDAL (in-plane face) field, which this does not provide.
    let n_mag = if spec.is_mhd { d } else { 0 };
    let n_prim = 1 + d + usize::from(spec.has_energy) + n_mag;
    if cfg.outputs.len() != n_prim {
        return Err(format!(
            "driven boundary must prescribe the full prim state [rho, vel_0..vel_{}{}{}]: outputs.len() \
             = {}, expected {n_prim}",
            d.saturating_sub(1),
            if spec.has_energy { ", pre" } else { "" },
            if spec.is_mhd { format!(", B_0..B_{}", d.saturating_sub(1)) } else { String::new() },
            cfg.outputs.len(),
        ));
    }
    // split the user DAG into per-slot prescriptions: den <- rho, mom <- the d-vector vel,
    // nrg <- pre, bcell <- the d-vector cell B. each is an independent lowering over its
    // output subset (own graph) — no graph cloning.
    let lower = |outs: &[usize]| {
        lower_dag_to_builtsource(&nodes, outs).map_err(|e| format!("bridge: {e:?}"))
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
        out.push(("bcell".to_string(), lower(&reduced_outputs[next..next + d])?));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::regime_spec::{ISO_NEWTONIAN_SPEC, NEWTONIAN_SPEC, RMHD_SPEC, RHD_SPEC};
    use symbi_expr::dag::Dag;
    use symbi_ir::backends::interp::{Backend, Cpu};
    use symbi_ir::passes::scalarize::scalarize;

    // ---- the axiomatic VALIDATION gate (build_user_source vs RegimeSpec) -------------------
    // each ill-posed config must fail HERE (pre-attach), driven entirely by the regime's spec,
    // never as a mid-evolve panic.

    fn cfg_from(json: &str) -> symbi_expr::SourceConfig {
        symbi_expr::SourceConfig::from_json(json).expect("parse")
    }

    // extract the error message (BuiltSource isn't Debug, so `unwrap_err` won't compile).
    fn expect_err(cfg: &symbi_expr::SourceConfig, spec: &crate::regime_spec::RegimeSpec) -> String {
        match build_user_source(cfg, spec) {
            Err(e) => e,
            Ok(_) => panic!("expected build_user_source to reject the config"),
        }
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
        // pins the cross-language wire: this is the EXACT json the python
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
        let built = build_user_source(&cfg, &NEWTONIAN_SPEC).expect("python force config lowers");
        assert_eq!(built.iter().map(|(t, _)| t.as_str()).collect::<Vec<_>>(), ["mom", "nrg"]);
    }

    #[test]
    fn python_sponge_json_loads_and_lowers() {
        // the EXACT json python's serialize_source(SourceKind.SPONGE, dim=3, params=[inv_gm1]) emits.
        // outputs = [kappa, den_ref, mom_ref_0..2, nrg_ref] mapped to node indices; here kappa=2,
        // den_ref=1, mom_ref=(x_0,x_1,x_2) (reads position), nrg_ref=10, inv_gm1=2.5. pins the
        // cross-language wire for the buffer-zone sponge.
        let cfg = cfg_from(
            r#"{"kind": "sponge", "dim": 3, "outputs": [3, 4, 0, 1, 2, 5], "params": [2.5],
                "nodes": [{"op": "VARIABLE_X1"}, {"op": "VARIABLE_X2"}, {"op": "VARIABLE_X3"},
                          {"op": "CONSTANT", "value": 2.0}, {"op": "CONSTANT", "value": 1.0},
                          {"op": "CONSTANT", "value": 10.0}]}"#,
        );
        let built = build_user_source(&cfg, &NEWTONIAN_SPEC).expect("python sponge config lowers");
        assert_eq!(built.iter().map(|(t, _)| t.as_str()).collect::<Vec<_>>(), ["den", "mom", "nrg"]);
        // at x=(1,0,0), state rho=1.5, vel=(3,0,0), pre=2:
        // S_mom_0 = kappa*(mom_ref_0 - rho*vel_0) = 2*(x_0 - 4.5) = 2*(1 - 4.5) = -7.
        let (_, mom) = &built[1];
        let s_mom0 = eval_lowered(mom, mom.outputs[0], &[
            ("rho", 1.5), ("vel_0", 3.0), ("vel_1", 0.0), ("vel_2", 0.0),
            ("x_0", 1.0), ("x_1", 0.0), ("x_2", 0.0),
        ]);
        assert!((s_mom0 - (-7.0)).abs() < 1e-12, "python sponge mom_0 wrong: {s_mom0}");
        // S_nrg = kappa*(nrg_ref - (pre*inv_gm1 + 0.5*rho*|v|^2)) = 2*(10 - (5 + 6.75)) = -3.5.
        // (the x_k leaves ride along in the spliced field — unused by the const nrg_ref, but present.)
        let (_, nrg) = &built[2];
        let s_nrg = eval_lowered(nrg, nrg.outputs[0], &[
            ("rho", 1.5), ("vel_0", 3.0), ("vel_1", 0.0), ("vel_2", 0.0), ("pre", 2.0),
            ("x_0", 1.0), ("x_1", 0.0), ("x_2", 0.0),
        ]);
        assert!((s_nrg - (-3.5)).abs() < 1e-12, "python sponge nrg wrong: {s_nrg}");
    }

    #[test]
    fn force_on_newtonian_is_accepted() {
        // mom + nrg overlays (newtonian has energy).
        let cfg = cfg_from(
            r#"{ "kind":"force", "dim":2, "outputs":[0,1], "params":[0.5],
                 "nodes":[ {"op":"PARAMETER","param_idx":0}, {"op":"CONSTANT","value":0.0} ] }"#,
        );
        let built = build_user_source(&cfg, &NEWTONIAN_SPEC).expect("force ok on newtonian");
        assert_eq!(built.iter().map(|(t, _)| t.as_str()).collect::<Vec<_>>(), ["mom", "nrg"]);
    }

    #[test]
    fn force_on_iso_drops_energy_overlay() {
        // iso has no energy: the mom overlay survives, the nrg overlay is NOT emitted.
        let cfg = cfg_from(
            r#"{ "kind":"force", "dim":2, "outputs":[0,1], "params":[0.5],
                 "nodes":[ {"op":"PARAMETER","param_idx":0}, {"op":"CONSTANT","value":0.0} ] }"#,
        );
        let built = build_user_source(&cfg, &ISO_NEWTONIAN_SPEC).expect("force ok on iso");
        assert_eq!(built.iter().map(|(t, _)| t.as_str()).collect::<Vec<_>>(), ["mom"]);
    }

    #[test]
    fn force_on_relativistic_is_rejected() {
        // RHD momentum is rho*h*W^2*v — the newtonian force law is wrong; reject.
        let cfg = cfg_from(
            r#"{ "kind":"force", "dim":1, "outputs":[0], "params":[0.5],
                 "nodes":[ {"op":"PARAMETER","param_idx":0} ] }"#,
        );
        let err = expect_err(&cfg, &RHD_SPEC);
        assert!(err.contains("relativistic"), "expected relativistic rejection, got: {err}");
    }

    #[test]
    fn inject_on_newtonian_writes_all_conserved_slots() {
        // one config depositing mass+momentum+energy: outputs = [S_den, S_mom_0, S_mom_1, S_nrg]
        // = [1, 2, 3, 4], each written IDENTITY to its conserved slot (no law wrap, like raw but
        // spanning every slot at once). the multi-channel deposition a single-slot raw cannot do.
        let cfg = cfg_from(
            r#"{ "kind":"inject", "dim":2, "outputs":[0,1,2,3], "params":[],
                 "nodes":[ {"op":"CONSTANT","value":1.0}, {"op":"CONSTANT","value":2.0},
                           {"op":"CONSTANT","value":3.0}, {"op":"CONSTANT","value":4.0} ] }"#,
        );
        let built = build_user_source(&cfg, &NEWTONIAN_SPEC).expect("inject ok on newtonian");
        assert_eq!(built.iter().map(|(t, _)| t.as_str()).collect::<Vec<_>>(), ["den", "mom", "nrg"]);
        // den: single output = 1.
        let (_, den) = &built[0];
        assert_eq!(den.outputs.len(), 1);
        assert!((eval_lowered(den, den.outputs[0], &[]) - 1.0).abs() < 1e-12);
        // mom: D=2 outputs = [2, 3], in order.
        let (_, mom) = &built[1];
        assert_eq!(mom.outputs.len(), 2);
        assert!((eval_lowered(mom, mom.outputs[0], &[]) - 2.0).abs() < 1e-12);
        assert!((eval_lowered(mom, mom.outputs[1], &[]) - 3.0).abs() < 1e-12);
        // nrg: single output = 4.
        let (_, nrg) = &built[2];
        assert_eq!(nrg.outputs.len(), 1);
        assert!((eval_lowered(nrg, nrg.outputs[0], &[]) - 4.0).abs() < 1e-12);
    }

    #[test]
    fn inject_on_iso_drops_energy_channel() {
        // iso has no energy: outputs = [S_den, S_mom_0, S_mom_1] (1+dim); the nrg slot is NOT emitted.
        let cfg = cfg_from(
            r#"{ "kind":"inject", "dim":2, "outputs":[0,1,2], "params":[],
                 "nodes":[ {"op":"CONSTANT","value":1.0}, {"op":"CONSTANT","value":2.0},
                           {"op":"CONSTANT","value":3.0} ] }"#,
        );
        let built = build_user_source(&cfg, &ISO_NEWTONIAN_SPEC).expect("inject ok on iso");
        assert_eq!(built.iter().map(|(t, _)| t.as_str()).collect::<Vec<_>>(), ["den", "mom"]);
    }

    #[test]
    fn inject_on_relativistic_is_accepted() {
        // inject supplies CONSERVED components directly (like raw, no newtonian law wrap), so it is
        // valid on a relativistic regime where force/cooling/relax are rejected. rhd has energy:
        // [D_dot, S_dot_0, tau_dot] at dim=1 -> 3 outputs.
        let cfg = cfg_from(
            r#"{ "kind":"inject", "dim":1, "outputs":[0,1,2], "params":[],
                 "nodes":[ {"op":"CONSTANT","value":1.0}, {"op":"CONSTANT","value":2.0},
                           {"op":"CONSTANT","value":3.0} ] }"#,
        );
        let built = build_user_source(&cfg, &RHD_SPEC).expect("inject ok on rhd (raw-like)");
        assert_eq!(built.iter().map(|(t, _)| t.as_str()).collect::<Vec<_>>(), ["den", "mom", "nrg"]);
    }

    #[test]
    fn inject_relativistic_2d_engine_channels() {
        // the Duffell & MacFadyen 2015 collimated engine on SRHD (dim=2): one nozzle power
        // S_0 drives three coupled conserved-rate channels — S_den = S_0/eta_0, S_mom_r =
        // S_0*sqrt(1-1/gamma_0^2), S_mom_theta = 0 (purely radial), S_nrg = S_0. here S_0=10,
        // eta_0=100 (node DIVIDE -> 0.1), S_mom_r=9.998, mirroring the axis_jet source shape.
        let cfg = cfg_from(
            r#"{ "kind":"inject", "dim":2, "outputs":[2,3,4,0], "params":[],
                 "nodes":[ {"op":"CONSTANT","value":10.0}, {"op":"CONSTANT","value":100.0},
                           {"op":"DIVIDE","left":0,"right":1}, {"op":"CONSTANT","value":9.998},
                           {"op":"CONSTANT","value":0.0} ] }"#,
        );
        let built = build_user_source(&cfg, &RHD_SPEC).expect("2d engine inject ok on rhd");
        assert_eq!(built.iter().map(|(t, _)| t.as_str()).collect::<Vec<_>>(), ["den", "mom", "nrg"]);
        // S_den = S_0/eta_0 = 10/100 = 0.1 (the DIVIDE channel).
        let (_, den) = &built[0];
        assert_eq!(den.outputs.len(), 1);
        assert!((eval_lowered(den, den.outputs[0], &[]) - 0.1).abs() < 1e-12);
        // S_mom = [S_mom_r, S_mom_theta] = [9.998, 0]; the theta channel is exactly zero.
        let (_, mom) = &built[1];
        assert_eq!(mom.outputs.len(), 2);
        assert!((eval_lowered(mom, mom.outputs[0], &[]) - 9.998).abs() < 1e-12);
        assert!(eval_lowered(mom, mom.outputs[1], &[]).abs() < 1e-12);
        // S_nrg = S_0 = 10.
        let (_, nrg) = &built[2];
        assert!((eval_lowered(nrg, nrg.outputs[0], &[]) - 10.0).abs() < 1e-12);
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
        assert_eq!(built.iter().map(|(t, _)| t.as_str()).collect::<Vec<_>>(), ["den", "mom", "nrg"]);
        let (_, den) = &built[0];
        assert!((eval_lowered(den, den.outputs[0], &[]) - 0.1).abs() < 1e-12);
        // mom carries all three spatial components; only the radial one is nonzero.
        let (_, mom) = &built[1];
        assert_eq!(mom.outputs.len(), 3);
        assert!((eval_lowered(mom, mom.outputs[0], &[]) - 9.998).abs() < 1e-12);
        assert!(eval_lowered(mom, mom.outputs[1], &[]).abs() < 1e-12);
        assert!(eval_lowered(mom, mom.outputs[2], &[]).abs() < 1e-12);
        let (_, nrg) = &built[2];
        assert!((eval_lowered(nrg, nrg.outputs[0], &[]) - 10.0).abs() < 1e-12);
    }

    #[test]
    fn inject_wrong_arity_is_rejected() {
        // energy regime at dim=2 needs [den, mom_0, mom_1, nrg] = 4 outputs; supplying 3 is
        // rejected pre-attach, never a mid-evolve panic.
        let cfg = cfg_from(
            r#"{ "kind":"inject", "dim":2, "outputs":[0,1,2], "params":[],
                 "nodes":[ {"op":"CONSTANT","value":1.0}, {"op":"CONSTANT","value":2.0},
                           {"op":"CONSTANT","value":3.0} ] }"#,
        );
        let err = expect_err(&cfg, &NEWTONIAN_SPEC);
        assert!(err.contains("inject"), "expected inject arity rejection, got: {err}");
    }

    #[test]
    fn cooling_on_iso_is_rejected() {
        // cooling targets nrg, which iso lacks -> reject up front.
        let cfg = cfg_from(
            r#"{ "kind":"cooling", "dim":1, "outputs":[0], "params":[1.0],
                 "nodes":[ {"op":"PARAMETER","param_idx":0} ] }"#,
        );
        let err = expect_err(&cfg, &ISO_NEWTONIAN_SPEC);
        assert!(err.contains("energy"), "expected energy-required rejection, got: {err}");
    }

    #[test]
    fn force_wrong_arity_is_rejected() {
        // force declares one accel component per dim: outputs.len() must == dim.
        let cfg = cfg_from(
            r#"{ "kind":"force", "dim":2, "outputs":[0], "params":[0.5],
                 "nodes":[ {"op":"PARAMETER","param_idx":0} ] }"#,
        );
        let err = expect_err(&cfg, &NEWTONIAN_SPEC);
        assert!(err.contains("per dim"), "expected arity rejection, got: {err}");
    }

    #[test]
    fn raw_bad_target_is_rejected() {
        let cfg = cfg_from(
            r#"{ "kind":"raw", "dim":1, "outputs":[0], "params":[1.0], "target":"pressure",
                 "nodes":[ {"op":"PARAMETER","param_idx":0} ] }"#,
        );
        let err = expect_err(&cfg, &NEWTONIAN_SPEC);
        assert!(err.contains("conserved slot"), "expected target rejection, got: {err}");
    }

    #[test]
    fn raw_nrg_on_iso_is_rejected() {
        let cfg = cfg_from(
            r#"{ "kind":"raw", "dim":1, "outputs":[0], "params":[1.0], "target":"nrg",
                 "nodes":[ {"op":"PARAMETER","param_idx":0} ] }"#,
        );
        let err = expect_err(&cfg, &ISO_NEWTONIAN_SPEC);
        assert!(err.contains("energy"), "expected nrg-needs-energy rejection, got: {err}");
    }

    // ---- region axis ----------------------------------------------

    #[test]
    fn region_masks_the_contribution() {
        // force a = [p0, 0], region chi = x_0 (a linear ramp). the lift is linear, so the masked
        // momentum source is S_mom_0 = rho * (chi * a_0) = rho * x_0 * p0.
        // nodes: 0=PARAM p0, 1=CONST 0, 2=VARIABLE_X1 (chi). outputs=[0,1], region=2.
        let cfg = cfg_from(
            r#"{ "kind":"force", "dim":2, "outputs":[0,1], "region":2, "params":[0.5],
                 "nodes":[ {"op":"PARAMETER","param_idx":0}, {"op":"CONSTANT","value":0.0},
                           {"op":"VARIABLE_X1"} ] }"#,
        );
        let built = build_user_source(&cfg, &NEWTONIAN_SPEC).expect("force+region");
        let (tgt, mom) = &built[0];
        assert_eq!(tgt, "mom");
        let s_at = |x0: f64| eval_lowered(mom, mom.outputs[0], &[("rho", 2.0), ("p0", 0.5), ("x_0", x0)]);
        assert!(s_at(0.0).abs() < 1e-12, "region masks to zero where chi = 0: got {}", s_at(0.0));
        assert!((s_at(1.0) - 1.0).abs() < 1e-12, "full contribution where chi = 1: got {}", s_at(1.0));
        assert!((s_at(0.5) - 0.5).abs() < 1e-12, "linear in chi: got {}", s_at(0.5));
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
        let built = build_user_source(&cfg, &NEWTONIAN_SPEC).expect("relax newtonian");
        assert_eq!(built.iter().map(|(t, _)| t.as_str()).collect::<Vec<_>>(), ["mom", "nrg"]);
        let (_, mom) = &built[0];
        // kappa=2, rho=1, v_ref_0=0, vel_0=3 -> 2*1*(0-3) = -6: the drag OPPOSES the velocity.
        let s_mom0 = eval_lowered(mom, mom.outputs[0],
            &[("rho", 1.0), ("vel_0", 3.0), ("vel_1", 0.0), ("p0", 2.0), ("p1", 0.0)]);
        assert!((s_mom0 - (-6.0)).abs() < 1e-12, "relax drag wrong: {s_mom0}");
        // the energy overlay = work = sum vel_k * S_mom_k = 3*(-6) + 0 = -18 < 0: KE is REMOVED.
        let (_, nrg) = &built[1];
        let s_nrg = eval_lowered(nrg, nrg.outputs[0],
            &[("rho", 1.0), ("vel_0", 3.0), ("vel_1", 0.0), ("p0", 2.0), ("p1", 0.0)]);
        assert!(s_nrg < 0.0, "relaxation must remove kinetic energy, got S_nrg = {s_nrg}");
    }

    #[test]
    fn relax_clamps_negative_rate_to_zero() {
        // the stability invariant: a NEGATIVE kappa (anti-damping) is clamped to 0 -> no-op, so no
        // energy-injecting instability arises. unexpressible by construction.
        let cfg = cfg_from(
            r#"{ "kind":"relax", "dim":1, "outputs":[0,1], "params":[-5.0, 0.0],
                 "nodes":[ {"op":"PARAMETER","param_idx":0}, {"op":"PARAMETER","param_idx":1} ] }"#,
        );
        let built = build_user_source(&cfg, &NEWTONIAN_SPEC).expect("relax");
        let (_, mom) = &built[0];
        // kappa = -5 -> clamped to 0 -> S_mom_0 = 0 regardless of the velocity overshoot.
        let s = eval_lowered(mom, mom.outputs[0],
            &[("rho", 1.0), ("vel_0", 7.0), ("p0", -5.0), ("p1", 0.0)]);
        assert!(s.abs() < 1e-12, "negative kappa must clamp to a no-op, got {s}");
    }

    // ---- sponge: full conserved-state relaxation (the buffer zone) -----------------

    #[test]
    fn sponge_relaxes_full_state_toward_reference() {
        // outputs = [kappa, den_ref, mom_ref_0, mom_ref_1, nrg_ref] as CONSTANT nodes (the reference
        // is a pure function of position — params carries only inv_gm1). the
        // three channels each relax toward the reference conserved value.
        //   kappa=2, den_ref=1, mom_ref=[0.5,0], nrg_ref=10, inv_gm1=2.5 (gamma=1.4).
        let cfg = cfg_from(
            r#"{ "kind":"sponge", "dim":2, "outputs":[0,1,2,3,4], "params":[2.5],
                 "nodes":[ {"op":"CONSTANT","value":2.0}, {"op":"CONSTANT","value":1.0},
                           {"op":"CONSTANT","value":0.5}, {"op":"CONSTANT","value":0.0},
                           {"op":"CONSTANT","value":10.0} ] }"#,
        );
        let built = build_user_source(&cfg, &NEWTONIAN_SPEC).expect("sponge newtonian");
        assert_eq!(built.iter().map(|(t, _)| t.as_str()).collect::<Vec<_>>(), ["den", "mom", "nrg"]);

        // state: rho=1.5, vel=[3,0], pre=2 (each channel reads only what it needs from this).
        let state = [("rho", 1.5), ("vel_0", 3.0), ("vel_1", 0.0), ("pre", 2.0)];
        // S_den = kappa*(den_ref - rho) = 2*(1 - 1.5) = -1.0 (density relaxes DOWN toward the ref).
        let (_, den) = &built[0];
        let s_den = eval_lowered(den, den.outputs[0], &state);
        assert!((s_den - (-1.0)).abs() < 1e-12, "sponge density wrong: {s_den}");
        // S_mom_0 = kappa*(mom_ref_0 - rho*vel_0) = 2*(0.5 - 4.5) = -8.0 (opposes the momentum).
        let (_, mom) = &built[1];
        let s_mom0 = eval_lowered(mom, mom.outputs[0], &state);
        assert!((s_mom0 - (-8.0)).abs() < 1e-12, "sponge mom_0 wrong: {s_mom0}");
        // S_nrg = kappa*(nrg_ref - E), E = pre*inv_gm1 + 0.5*rho*|v|^2 = 2*2.5 + 0.5*1.5*9 = 11.75;
        //   -> 2*(10 - 11.75) = -3.5 (total energy relaxes DOWN toward the ref).
        let (_, nrg) = &built[2];
        let s_nrg = eval_lowered(nrg, nrg.outputs[0], &state);
        assert!((s_nrg - (-3.5)).abs() < 1e-12, "sponge nrg wrong: {s_nrg}");
    }

    #[test]
    fn sponge_on_iso_drops_energy_channel() {
        // iso has no energy: the reference is [kappa, den_ref, mom_ref_0] (2+D), and only den+mom
        // channels are emitted (no nrg_ref, no inv_gm1 needed).
        let cfg = cfg_from(
            r#"{ "kind":"sponge", "dim":1, "outputs":[0,1,2], "params":[],
                 "nodes":[ {"op":"CONSTANT","value":1.0}, {"op":"CONSTANT","value":2.0},
                           {"op":"CONSTANT","value":0.0} ] }"#,
        );
        let built = build_user_source(&cfg, &ISO_NEWTONIAN_SPEC).expect("sponge iso");
        assert_eq!(built.iter().map(|(t, _)| t.as_str()).collect::<Vec<_>>(), ["den", "mom"]);
        // S_den = 1*(2 - rho); rho=0.5 -> 1.5 (density relaxes UP toward the ref).
        let (_, den) = &built[0];
        let s_den = eval_lowered(den, den.outputs[0], &[("rho", 0.5)]);
        assert!((s_den - 1.5).abs() < 1e-12, "iso sponge density wrong: {s_den}");
    }

    #[test]
    fn sponge_wrong_arity_is_rejected() {
        // energy regime needs 3+D outputs (kappa, den_ref, D mom_ref, nrg_ref); a short list fails.
        let cfg = cfg_from(
            r#"{ "kind":"sponge", "dim":2, "outputs":[0,1,2], "params":[2.5],
                 "nodes":[ {"op":"CONSTANT","value":1.0}, {"op":"CONSTANT","value":1.0},
                           {"op":"CONSTANT","value":0.0} ] }"#,
        );
        let err = expect_err(&cfg, &NEWTONIAN_SPEC);
        assert!(err.contains("nrg_ref"), "expected sponge arity rejection, got: {err}");
    }

    // ---- state variables: density + pressure in user source expressions -------------------

    #[test]
    fn raw_source_reads_density_and_pressure() {
        // a radiative-cooling-style rate S_nrg = -(C * rho * pre): the user expression reads the
        // per-cell STATE (density + pressure) — the capability that lets adiabatic cooling
        // Lambda(rho, T), T = pre/rho, be user-defined. nodes: 0=PARAM C, 1=VARIABLE_RHO,
        // 2=VARIABLE_PRESSURE, 3=MUL(C,rho), 4=MUL(3,pre), 5=NEG(4). outputs=[5], target=nrg.
        let cfg = cfg_from(
            r#"{ "kind":"raw", "dim":1, "outputs":[5], "params":[0.25], "target":"nrg",
                 "nodes":[ {"op":"PARAMETER","param_idx":0}, {"op":"VARIABLE_RHO"},
                           {"op":"VARIABLE_PRESSURE"}, {"op":"MULTIPLY","left":0,"right":1},
                           {"op":"MULTIPLY","left":3,"right":2}, {"op":"NEG","left":4} ] }"#,
        );
        let built = build_user_source(&cfg, &NEWTONIAN_SPEC).expect("raw pressure-reading cooling");
        let (tgt, nrg) = &built[0];
        assert_eq!(tgt, "nrg");
        // S_nrg = -(C * rho * pre); C=0.25, rho=2, pre=3 -> -(0.25*2*3) = -1.5.
        let s = eval_lowered(nrg, nrg.outputs[0], &[("p0", 0.25), ("rho", 2.0), ("pre", 3.0)]);
        assert!((s - (-1.5)).abs() < 1e-12, "cooling rate must read rho*pre: got {s}");
        // it genuinely DEPENDS on pressure: doubling pre doubles the rate.
        let s2 = eval_lowered(nrg, nrg.outputs[0], &[("p0", 0.25), ("rho", 2.0), ("pre", 6.0)]);
        assert!((s2 - (-3.0)).abs() < 1e-12, "rate must scale with pressure: got {s2}");
    }

    #[test]
    fn raw_source_targets_density_slot() {
        // a density-only injection (pure mass loading, no momentum/energy) — a baryon source
        // rate S_den = p0 * rho written straight to the `den` slot. the single-slot path for
        // mass loading (the full-vector path is `inject`).
        let cfg = cfg_from(
            r#"{ "kind":"raw", "dim":1, "outputs":[2], "params":[0.5], "target":"den",
                 "nodes":[ {"op":"PARAMETER","param_idx":0}, {"op":"VARIABLE_RHO"},
                           {"op":"MULTIPLY","left":0,"right":1} ] }"#,
        );
        let built = build_user_source(&cfg, &NEWTONIAN_SPEC).expect("raw den ok");
        assert_eq!(built.iter().map(|(t, _)| t.as_str()).collect::<Vec<_>>(), ["den"]);
        let (tgt, den) = &built[0];
        assert_eq!(tgt, "den");
        // S_den = p0 * rho; p0=0.5, rho=2 -> 1.
        let s = eval_lowered(den, den.outputs[0], &[("p0", 0.5), ("rho", 2.0)]);
        assert!((s - 1.0).abs() < 1e-12, "raw den rate must be p0*rho: got {s}");
    }

    #[test]
    fn relax_on_iso_drops_energy_overlay() {
        // iso has no energy: relax yields ONLY the momentum drag (no work term).
        let cfg = cfg_from(
            r#"{ "kind":"relax", "dim":1, "outputs":[0,1], "params":[1.0, 0.0],
                 "nodes":[ {"op":"PARAMETER","param_idx":0}, {"op":"PARAMETER","param_idx":1} ] }"#,
        );
        let built = build_user_source(&cfg, &ISO_NEWTONIAN_SPEC).expect("relax iso");
        assert_eq!(built.iter().map(|(t, _)| t.as_str()).collect::<Vec<_>>(), ["mom"]);
    }

    #[test]
    fn relax_on_relativistic_is_rejected() {
        let cfg = cfg_from(
            r#"{ "kind":"relax", "dim":1, "outputs":[0,1], "params":[1.0, 0.0],
                 "nodes":[ {"op":"PARAMETER","param_idx":0}, {"op":"PARAMETER","param_idx":1} ] }"#,
        );
        let err = expect_err(&cfg, &RHD_SPEC);
        assert!(err.contains("relativistic"), "expected relativistic rejection, got: {err}");
    }

    #[test]
    fn relax_wrong_arity_is_rejected() {
        // relax needs [kappa, v_ref_0..v_ref_{dim-1}] = 1 + dim outputs.
        let cfg = cfg_from(
            r#"{ "kind":"relax", "dim":2, "outputs":[0,1], "params":[1.0, 0.0],
                 "nodes":[ {"op":"PARAMETER","param_idx":0}, {"op":"PARAMETER","param_idx":1} ] }"#,
        );
        let err = expect_err(&cfg, &NEWTONIAN_SPEC);
        assert!(err.contains("v_ref"), "expected relax arity rejection, got: {err}");
    }

    // ---- driven boundaries -----------------------------------------------

    fn expect_boundary_err(cfg: &symbi_expr::SourceConfig, spec: &crate::regime_spec::RegimeSpec) -> String {
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
        let slots: Vec<(&str, usize)> =
            built.iter().map(|(s, b)| (s.as_str(), b.outputs.len())).collect();
        assert_eq!(slots, vec![("den", 1), ("mom", 2), ("nrg", 1)]);
    }

    #[test]
    fn boundary_on_iso_drops_pressure() {
        // iso has no energy: the prim state is [rho, vel] only — no pre slot.
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
        // a prim prescription is regime-agnostic across hydro -> rhd (relativistic) is FINE,
        // unlike force/cooling (whose newtonian conservation law is wrong for rhd).
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
        let slots: Vec<(&str, usize)> =
            built.iter().map(|(s, b)| (s.as_str(), b.outputs.len())).collect();
        assert_eq!(slots, vec![("den", 1), ("mom", 3), ("nrg", 1), ("bcell", 3)]);
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

    /// evaluate one output of a lowered BuiltSource on the CPU interpreter, binding params
    /// by the declared (name -> value) map in manifest order.
    fn eval_lowered(built: &BuiltSource, output: NodeId, values: &[(&str, f64)]) -> f64 {
        let lowered = scalarize(&built.graph, output, "expr_bridge");
        let inputs: Vec<f64> = built
            .params
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
    /// Graph, evaluates IDENTICALLY to `symbi-expr`'s own register-VM interpreter — proving
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
        let built = lower_dag_to_builtsource(&nodes, &[root]).expect("bridge lowers");

        let mut expr = dag.compile(&[root]);

        // two states: one taking the `then` branch (x1 > p0), one taking `else`.
        for (x1v, x2v, tv, p0v) in [(1.0, 0.7, 0.3, 0.5), (0.2, -0.4, 1.1, 0.5)] {
            expr.set_params(&[p0v]);
            let want = expr.eval(x1v, x2v, x3_unused(), tv)[0];
            let got = eval_lowered(
                &built,
                built.outputs[0],
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
        // `Sgn` / `Mod` have no carrier-traceable primitive — the bridge must reject them.
        let mut dag = Dag::new();
        let x1 = dag.var_x1();
        let sgn = dag.unary(Op::Sgn, x1);
        let nodes = dag.nodes().to_vec();
        let result = lower_dag_to_builtsource(&nodes, &[sgn]);
        assert!(
            matches!(result, Err(BridgeError::UnsupportedOp(Op::Sgn))),
            "Sgn must be rejected as unsupported",
        );
    }
}
