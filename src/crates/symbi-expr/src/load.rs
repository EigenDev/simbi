// =============================================================================
// load.rs
//
// deserializes expression DAGs from the config format. takes flat arrays
// of node descriptors (matching the python serialization) and produces a
// compiled Expression ready for evaluation.
//
// usage:
//   let nodes = vec![
//       NodeDesc::constant(1.0),
//       NodeDesc::variable("VARIABLE_X1"),
//       NodeDesc::binary("MULTIPLY", 0, 1),
//   ];
//   let expr = load_expression(&nodes, &[2], &[])?;
//   assert_eq!(expr.eval(3.0, 0.0, 0.0, 0.0)[0], 3.0);
// =============================================================================

use crate::dag::{Node, Payload};
use crate::eval::Expression;
use crate::op::Op;

/// error type for expression loading.
#[derive(Debug)]
pub enum LoadError {
    UnknownOp(String),
    MissingField {
        node: usize,
        field: &'static str,
    },
    InvalidIndex {
        node: usize,
        field: &'static str,
        index: usize,
        num_nodes: usize,
    },
}

impl std::fmt::Display for LoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            LoadError::UnknownOp(op) => write!(f, "unknown op: {}", op),
            LoadError::MissingField { node, field } => {
                write!(f, "node {}: missing field '{}'", node, field)
            }
            LoadError::InvalidIndex {
                node,
                field,
                index,
                num_nodes,
            } => {
                write!(
                    f,
                    "node {}: field '{}' index {} >= num_nodes {}",
                    node, field, index, num_nodes
                )
            }
        }
    }
}

impl std::error::Error for LoadError {}

/// descriptor for a single DAG node in the serialized config format.
/// matches the python/json serialization: each node has an "op" string
/// and type-dependent optional fields. None fields are omitted on the wire
/// (and default to None on load) so the python emitter only writes what a
/// node uses.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct NodeDesc {
    pub op: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub value: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub param_idx: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub left: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub right: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub condition: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub true_case: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub false_case: Option<usize>,
}

impl NodeDesc {
    /// constant node.
    pub fn constant(val: f64) -> Self {
        NodeDesc {
            op: "CONSTANT".into(),
            value: Some(val),
            ..Self::empty()
        }
    }

    /// variable node (op must be "VARIABLE_X1", "VARIABLE_X2", etc.).
    pub fn variable(op: &str) -> Self {
        NodeDesc {
            op: op.into(),
            ..Self::empty()
        }
    }

    /// parameter node.
    pub fn parameter(idx: usize) -> Self {
        NodeDesc {
            op: "PARAMETER".into(),
            param_idx: Some(idx),
            ..Self::empty()
        }
    }

    /// unary operation.
    pub fn unary(op: &str, child: usize) -> Self {
        NodeDesc {
            op: op.into(),
            left: Some(child),
            ..Self::empty()
        }
    }

    /// binary operation.
    pub fn binary(op: &str, left: usize, right: usize) -> Self {
        NodeDesc {
            op: op.into(),
            left: Some(left),
            right: Some(right),
            ..Self::empty()
        }
    }

    /// if-then-else ternary.
    pub fn ternary(condition: usize, true_case: usize, false_case: usize) -> Self {
        NodeDesc {
            op: "IF_THEN_ELSE".into(),
            condition: Some(condition),
            true_case: Some(true_case),
            false_case: Some(false_case),
            ..Self::empty()
        }
    }

    fn empty() -> Self {
        NodeDesc {
            op: String::new(),
            value: None,
            param_idx: None,
            left: None,
            right: None,
            condition: None,
            true_case: None,
            false_case: None,
        }
    }
}

/// load an expression from serialized node descriptors.
///
/// each node in `node_descs` references other nodes by index. the
/// `output_indices` specify which nodes are the final outputs.
/// `params` are runtime parameter values (for Op::Parameter nodes).
pub fn load_expression(
    node_descs: &[NodeDesc],
    output_indices: &[usize],
    params: &[f64],
) -> Result<Expression, LoadError> {
    let nodes = nodes_from_descs(node_descs)?;
    // constant-power strength reduction — the SAME rewrite the ir bridge applies, so the
    // f64 reference vm and the lowered kernel evaluate identical arithmetic.
    let (nodes, output_indices) = crate::strength::strength_reduce(&nodes, output_indices);
    let mut expr = Expression::from_nodes(&nodes, &output_indices);
    if !params.is_empty() {
        expr.set_params(params);
    }
    Ok(expr)
}

/// build the topologically-ordered `Node` array from serialized descriptors — the shared
/// front half of `load_expression`. `symbi-hydro::expr_bridge` lowers this same `Vec<Node>`
/// into the IR Graph (for fused codegen), so json -> nodes -> {VM `Expression` | IR
/// `BuiltSource`} is ONE parse. validates op names, arity, and child indices.
pub fn nodes_from_descs(node_descs: &[NodeDesc]) -> Result<Vec<Node>, LoadError> {
    let nn = node_descs.len();
    let mut nodes = Vec::with_capacity(nn);

    for (ii, desc) in node_descs.iter().enumerate() {
        let op = Op::from_name(&desc.op).ok_or_else(|| LoadError::UnknownOp(desc.op.clone()))?;

        let payload = match op.arity() {
            0 if op == Op::Constant => {
                let val = desc.value.ok_or(LoadError::MissingField {
                    node: ii,
                    field: "value",
                })?;
                Payload::Value(val)
            }
            0 if op == Op::Parameter => {
                let idx = desc.param_idx.ok_or(LoadError::MissingField {
                    node: ii,
                    field: "param_idx",
                })?;
                Payload::ParamIdx(idx)
            }
            0 => Payload::None, // variable
            1 => {
                let child = desc.left.ok_or(LoadError::MissingField {
                    node: ii,
                    field: "left",
                })?;
                if child >= nn {
                    return Err(LoadError::InvalidIndex {
                        node: ii,
                        field: "left",
                        index: child,
                        num_nodes: nn,
                    });
                }
                Payload::Unary(child)
            }
            2 => {
                let left = desc.left.ok_or(LoadError::MissingField {
                    node: ii,
                    field: "left",
                })?;
                let right = desc.right.ok_or(LoadError::MissingField {
                    node: ii,
                    field: "right",
                })?;
                if left >= nn {
                    return Err(LoadError::InvalidIndex {
                        node: ii,
                        field: "left",
                        index: left,
                        num_nodes: nn,
                    });
                }
                if right >= nn {
                    return Err(LoadError::InvalidIndex {
                        node: ii,
                        field: "right",
                        index: right,
                        num_nodes: nn,
                    });
                }
                Payload::Binary(left, right)
            }
            3 => {
                let cond = desc.condition.ok_or(LoadError::MissingField {
                    node: ii,
                    field: "condition",
                })?;
                let then_ = desc.true_case.ok_or(LoadError::MissingField {
                    node: ii,
                    field: "true_case",
                })?;
                let else_ = desc.false_case.ok_or(LoadError::MissingField {
                    node: ii,
                    field: "false_case",
                })?;
                if cond >= nn {
                    return Err(LoadError::InvalidIndex {
                        node: ii,
                        field: "condition",
                        index: cond,
                        num_nodes: nn,
                    });
                }
                if then_ >= nn {
                    return Err(LoadError::InvalidIndex {
                        node: ii,
                        field: "true_case",
                        index: then_,
                        num_nodes: nn,
                    });
                }
                if else_ >= nn {
                    return Err(LoadError::InvalidIndex {
                        node: ii,
                        field: "false_case",
                        index: else_,
                        num_nodes: nn,
                    });
                }
                Payload::Ternary(cond, then_, else_)
            }
            _ => unreachable!(),
        };

        nodes.push(Node { op, payload });
    }

    Ok(nodes)
}

/// a serialized USER source — the python front door's wire format. a python builder emits this
/// as json (`SourceConfig::from_json`); the rust side turns it into a VM `Expression`
/// (`to_expression`) or hands the `nodes` + `outputs` to `symbi-hydro::expr_bridge` for a fused
/// IR `BuiltSource`, then wraps it in the conservation law per `kind`.
///
/// json shape (None node fields omitted):
/// ```json
/// { "kind": "force", "dim": 2, "outputs": [2, 3], "params": [],
///   "nodes": [ {"op":"VARIABLE_X1"}, {"op":"PARAMETER","param_idx":0},
///              {"op":"MULTIPLY","left":1,"right":0}, {"op":"CONSTANT","value":0.4} ] }
/// ```
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SourceConfig {
    /// the conservation LAW the framework wraps the field in (`kind <-> invariant`):
    /// `"force"` (acceleration -> S_mom=rho*a, S_nrg=rho*a.v), `"cooling"` (rate -> S_nrg=-Lambda),
    /// `"relax"` (velocity relaxation S_mom=kappa*rho*(v_ref-v), kappa>=0 -> stable damping; the
    /// `outputs` are `[kappa, v_ref_0..v_ref_{D-1}]`), or `"raw"` (outputs written directly to
    /// `target`). force/cooling/relax are the SAFE primitive-lifted constructors; raw is the hole.
    pub kind: String,
    /// spatial dimension D the source is built at.
    pub dim: usize,
    /// for `kind="raw"`: the conserved field the outputs target. derived for force/cooling/relax.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target: Option<String>,
    /// node indices that are the field outputs (D for a force acceleration, 1 for cooling,
    /// 1+D for relax).
    pub outputs: Vec<usize>,
    /// runtime parameter values, indexed by each `PARAMETER` node's `param_idx`.
    #[serde(default)]
    pub params: Vec<f64>,
    /// **`region` axis** — an OPTIONAL node index (into `nodes`) of a mask
    /// `chi(x) in [0,1]` restricting WHERE the source acts (sponge layers, jet nozzles). the
    /// contribution is multiplied by `chi` at build time (the lift is linear in the field, so
    /// masking the field == masking the conserved contribution). `None` => everywhere (`chi == 1`),
    /// byte-identical to the pre-region kernels.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub region: Option<usize>,
    /// the flat, topologically-ordered DAG.
    pub nodes: Vec<NodeDesc>,
}

impl SourceConfig {
    /// parse the python-emitted json.
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }

    /// serialize back to json (round-trip / golden tests).
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string(self)
    }

    /// build the VM `Expression` (the CPU-interpreter form). for the fused IR form, hand
    /// `self.nodes` + `self.outputs` to `symbi-hydro::expr_bridge::lower_dag_to_builtsource`.
    pub fn to_expression(&self) -> Result<Expression, LoadError> {
        load_expression(&self.nodes, &self.outputs, &self.params)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-12 * a.abs().max(b.abs()).max(1.0)
    }

    #[test]
    fn load_constant() {
        let nodes = vec![NodeDesc::constant(42.0)];
        let expr = load_expression(&nodes, &[0], &[]).unwrap();
        assert!(approx(expr.eval(0.0, 0.0, 0.0, 0.0)[0], 42.0));
    }

    #[test]
    fn source_config_json_round_trips_and_evals() {
        // the python front door's wire format: f(x1) = p0 * sin(x1), p0 = 2.5. None node
        // fields are omitted, exactly as the python emitter writes them.
        let json = r#"{
            "kind": "raw", "dim": 1, "target": "nrg", "outputs": [3], "params": [2.5],
            "nodes": [
                {"op": "VARIABLE_X1"},
                {"op": "SIN", "left": 0},
                {"op": "PARAMETER", "param_idx": 0},
                {"op": "MULTIPLY", "left": 2, "right": 1}
            ]
        }"#;
        let cfg = SourceConfig::from_json(json).expect("parse json");
        assert_eq!(cfg.kind, "raw");
        assert_eq!(cfg.dim, 1);
        assert_eq!(cfg.outputs, vec![3]);

        let want = 2.5 * 0.7_f64.sin();
        let got = cfg
            .to_expression()
            .expect("build expr")
            .eval(0.7, 0.0, 0.0, 0.0)[0];
        assert!(
            approx(got, want),
            "json -> expr -> eval: got {got}, want {want}"
        );

        // serialize -> parse -> same eval (the round-trip the python golden test pins).
        let rt = SourceConfig::from_json(&cfg.to_json().unwrap()).unwrap();
        assert!(approx(
            rt.to_expression().unwrap().eval(0.7, 0.0, 0.0, 0.0)[0],
            want
        ));
    }

    #[test]
    fn load_arithmetic() {
        // 2 * x1
        let nodes = vec![
            NodeDesc::constant(2.0),
            NodeDesc::variable("VARIABLE_X1"),
            NodeDesc::binary("MULTIPLY", 0, 1),
        ];
        let expr = load_expression(&nodes, &[2], &[]).unwrap();
        assert!(approx(expr.eval(3.0, 0.0, 0.0, 0.0)[0], 6.0));
    }

    #[test]
    fn load_sod_ic() {
        // if x < 0.5 then 1.0 else 0.125
        let nodes = vec![
            NodeDesc::variable("VARIABLE_X1"), // 0
            NodeDesc::constant(0.5),           // 1
            NodeDesc::binary("LT", 0, 1),      // 2: x < 0.5
            NodeDesc::constant(1.0),           // 3
            NodeDesc::constant(0.125),         // 4
            NodeDesc::ternary(2, 3, 4),        // 5: if-then-else
        ];
        let expr = load_expression(&nodes, &[5], &[]).unwrap();
        assert!(approx(expr.eval(0.1, 0.0, 0.0, 0.0)[0], 1.0));
        assert!(approx(expr.eval(0.9, 0.0, 0.0, 0.0)[0], 0.125));
    }

    #[test]
    fn load_with_parameters() {
        // p0 * x + p1
        let nodes = vec![
            NodeDesc::parameter(0),             // 0
            NodeDesc::variable("VARIABLE_X1"),  // 1
            NodeDesc::binary("MULTIPLY", 0, 1), // 2
            NodeDesc::parameter(1),             // 3
            NodeDesc::binary("ADD", 2, 3),      // 4
        ];
        let expr = load_expression(&nodes, &[4], &[2.0, 5.0]).unwrap();
        assert!(approx(expr.eval(3.0, 0.0, 0.0, 0.0)[0], 11.0));
    }

    #[test]
    fn load_multi_output() {
        // outputs: (x + y, x * y)
        let nodes = vec![
            NodeDesc::variable("VARIABLE_X1"),  // 0
            NodeDesc::variable("VARIABLE_X2"),  // 1
            NodeDesc::binary("ADD", 0, 1),      // 2
            NodeDesc::binary("MULTIPLY", 0, 1), // 3
        ];
        let expr = load_expression(&nodes, &[2, 3], &[]).unwrap();
        let out = expr.eval(3.0, 4.0, 0.0, 0.0);
        assert!(approx(out[0], 7.0));
        assert!(approx(out[1], 12.0));
    }

    #[test]
    fn load_unknown_op() {
        let nodes = vec![NodeDesc::variable("NONSENSE")];
        let err = load_expression(&nodes, &[0], &[]);
        assert!(err.is_err());
        assert!(err.unwrap_err().to_string().contains("unknown op"));
    }

    #[test]
    fn load_missing_field() {
        let nodes = vec![NodeDesc {
            op: "CONSTANT".into(),
            ..NodeDesc::empty()
        }];
        let err = load_expression(&nodes, &[0], &[]);
        assert!(err.is_err());
        assert!(err.unwrap_err().to_string().contains("missing field"));
    }

    #[test]
    fn load_invalid_index() {
        let nodes = vec![
            NodeDesc::variable("VARIABLE_X1"),
            NodeDesc::binary("ADD", 0, 99), // 99 is out of bounds
        ];
        let err = load_expression(&nodes, &[1], &[]);
        assert!(err.is_err());
        assert!(err.unwrap_err().to_string().contains("index 99"));
    }

    #[test]
    fn load_unary() {
        // sin(x)
        let nodes = vec![NodeDesc::variable("VARIABLE_X1"), NodeDesc::unary("SIN", 0)];
        let expr = load_expression(&nodes, &[1], &[]).unwrap();
        let pi_half = std::f64::consts::FRAC_PI_2;
        assert!(approx(expr.eval(pi_half, 0.0, 0.0, 0.0)[0], 1.0));
    }
}
