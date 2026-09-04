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
//       NodeDesc::binary("multiply", 0, 1),
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
    /// a node referencing itself or a later node. the wire format is a topologically
    /// ordered dag, so every operand must already exist when its parent is read. without
    /// this check the malformed graph loads and the failure surfaces much later, as an
    /// out-of-bounds inside a graph pass, with nothing pointing back at the config.
    ForwardReference {
        node: usize,
        field: &'static str,
        index: usize,
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
            LoadError::ForwardReference { node, field, index } => {
                write!(
                    f,
                    "node {node}: field '{field}' references node {index}, which is not \
                     earlier in the dag; nodes must be topologically ordered so every \
                     operand is defined before it is used"
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
    // constant-power strength reduction — the same rewrite the ir bridge applies, so the
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
/// `SourceProgram`} is one parse. validates op names, arity, and child indices.
pub fn nodes_from_descs(node_descs: &[NodeDesc]) -> Result<Vec<Node>, LoadError> {
    let nn = node_descs.len();
    let mut nodes = Vec::with_capacity(nn);

    // an operand must be earlier in the array, which is what topological order means and
    // what every downstream pass assumes. checking only against the node count would admit
    // a self-reference or a forward edge, and the graph would then fail deep inside a pass
    // rather than at the config that caused it.
    let child = |node: usize, field: &'static str, index: usize| -> Result<usize, LoadError> {
        if index >= nn {
            return Err(LoadError::InvalidIndex {
                node,
                field,
                index,
                num_nodes: nn,
            });
        }
        if index >= node {
            return Err(LoadError::ForwardReference { node, field, index });
        }
        Ok(index)
    };

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
                let c = desc.left.ok_or(LoadError::MissingField {
                    node: ii,
                    field: "left",
                })?;
                Payload::Unary(child(ii, "left", c)?)
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
                Payload::Binary(child(ii, "left", left)?, child(ii, "right", right)?)
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
                Payload::Ternary(
                    child(ii, "condition", cond)?,
                    child(ii, "true_case", then_)?,
                    child(ii, "false_case", else_)?,
                )
            }
            _ => unreachable!(),
        };

        nodes.push(Node { op, payload });
    }

    Ok(nodes)
}

/// a serialized user source — the python front door's wire format. a python builder emits this
/// as json (`SourceConfig::from_json`); the rust side turns it into a VM `Expression`
/// (`to_expression`) or hands the `nodes` + `outputs` to `symbi-hydro::expr_bridge` for a fused
/// IR `SourceProgram`, then wraps it in the conservation law per `kind`.
///
/// json shape (None node fields omitted):
/// ```json
/// { "kind": "force", "dim": 2, "outputs": [2, 3], "params": [0.4],
///   "vocabulary": { "reads": ["x_0"], "params": [0] },
///   "nodes": [ {"op":"VARIABLE_X1"}, {"op":"parameter","param_idx":0},
///              {"op":"multiply","left":1,"right":0}, {"op":"constant","value":0.4} ] }
/// ```
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SourceConfig {
    /// the conservation law the framework wraps the field in (`kind <-> invariant`):
    /// `"force"` (acceleration -> S_mom=rho*a, S_nrg=rho*a.v), `"cooling"` (rate -> S_nrg=-Lambda),
    /// `"relax"` (velocity relaxation S_mom=kappa*rho*(v_ref-v), kappa>=0 -> stable damping; the
    /// `outputs` are `[kappa, v_ref_0..v_ref_{D-1}]`), or `"raw"` (outputs written directly to
    /// `target`). force/cooling/relax are the safe primitive-lifted constructors; raw is the hole.
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
    /// the leaves the source's building context granted it, captured as the frontend
    /// minted them and carried apart from `nodes`. a source contribution is admitted only
    /// when every leaf its dag observes is in this declaration, so the field is required
    /// for the source kinds; a boundary prescription or a motion law (which add to no
    /// conserved slot) carries none.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vocabulary: Option<SourceVocabulary>,
    /// **`region` axis** — an optional node index (into `nodes`) of a mask
    /// `chi(x) in [0,1]` restricting where the source acts (sponge layers, jet nozzles). the
    /// contribution is multiplied by `chi` at build time (the lift is linear in the field, so
    /// masking the field == masking the conserved contribution). `None` => everywhere (`chi == 1`),
    /// byte-identical to the pre-region kernels.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub region: Option<usize>,
    /// the flat, topologically-ordered DAG.
    pub nodes: Vec<NodeDesc>,
}

/// the vocabulary a source's frontend context granted it: the state and coordinate
/// reads by wire symbol (`rho`, `pre`, `vel_<k>`, `x_<k>`, `t`) and the `param_idx`
/// values of its parameter leaves. the frontend records each as the corresponding leaf
/// constructor runs, so the set is a capability grant rather than a hand-written list,
/// and the dag in `nodes` is compared against it at admission.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct SourceVocabulary {
    pub reads: Vec<String>,
    pub params: Vec<usize>,
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

/// a stationary target state, as an expression of position: the primitive vector a run declares
/// its equilibrium to be, so that a well-balanced scheme can hold it exactly.
///
/// this is a state, not a source, so it carries no conservation law to be wrapped in and no
/// conserved field to target — `outputs` are the primitive components themselves, in the order
/// `[rho, v_0 .. v_{DOF-1}, p]`, with the pressure slot present exactly when the regime carries
/// energy.
///
/// the target crosses the wire as an expression rather than as sampled field data because it must
/// be re-derivable at any resolution: a restart that adds a refinement level needs the target
/// defined on cells that did not exist when the run began, and sampled data cannot supply them.
///
/// json shape:
/// ```json
/// { "dim": 1, "outputs": [4, 5, 6], "params": [100.0],
///   "nodes": [ {"op":"VARIABLE_X1"}, ... ] }
/// ```
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct EquilibriumConfig {
    /// spatial dimension of the grid the target is declared on.
    pub dim: usize,
    /// node indices of the primitive components: density, then one velocity component per momentum
    /// degree of freedom, then pressure when the regime carries energy.
    pub outputs: Vec<usize>,
    /// runtime parameter values, indexed by each `PARAMETER` node's `param_idx`.
    #[serde(default)]
    pub params: Vec<f64>,
    /// the flat, topologically-ordered DAG.
    pub nodes: Vec<NodeDesc>,
}

impl EquilibriumConfig {
    /// parse the python-emitted json.
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }

    /// serialize back to json (round-trip / golden tests, and the checkpoint record that lets a
    /// restart verify it is continuing the same target it started with).
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string(self)
    }

    /// build the VM `Expression` evaluated per cell centre at setup.
    pub fn to_expression(&self) -> Result<Expression, LoadError> {
        load_expression(&self.nodes, &self.outputs, &self.params)
    }
}

/// one bin axis of a census: the expression giving the coordinate to bin on, and the edges
/// that cut it. edges are explicit rather than a spacing rule, so log spacing, linear
/// spacing and hand-chosen edges all cross the wire the same way.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CensusAxisConfig {
    /// labels the axis's edges in the output.
    pub name: String,
    /// node index (into the census `nodes`) of the coordinate expression.
    pub expr: usize,
    /// `n + 1` strictly increasing edges cutting the coordinate into `n` bins.
    pub edges: Vec<f64>,
}

/// a serialized user census — a pointwise map followed by a segmented reduce, emitted by the
/// python front door as json alongside the source expressions.
///
/// the axis expressions and the value expressions share one dag, so a subexpression used by
/// both (a radius, its logarithm) is written once and evaluated once per cell.
///
/// json shape:
/// ```json
/// { "name": "shells", "op": "add", "params": [],
///   "axes": [ {"name": "r", "expr": 0, "edges": [1.0, 2.0, 4.0]} ],
///   "values": [1], "value_names": ["mass"],
///   "nodes": [ {"op":"VARIABLE_X1"},
///              {"op":"multiply","left":2,"right":3},
///              {"op":"VARIABLE_RHO"}, {"op":"VARIABLE_DV"} ] }
/// ```
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CensusConfig {
    /// names the census's output group. unique across a run's registrations.
    pub name: String,
    /// bin axes, in registration order. they take an outer product, and an empty list is a
    /// global reduction over the grid — the case total mass and energy occupy.
    #[serde(default)]
    pub axes: Vec<CensusAxisConfig>,
    /// node indices of the accumulator expressions.
    pub values: Vec<usize>,
    /// one label per accumulator, so a reader can name a column without re-deriving the
    /// registration order.
    pub value_names: Vec<String>,
    /// how accumulators combine: `"add"` for moments, histograms, mass budgets and fluxes;
    /// `"min"` / `"max"` for per-bin extrema. a product is refused — it overflows to zero or
    /// infinity at any realistic cell count and is not a census statistic. mean, variance and
    /// percentile are deliberately absent: they are not order-agnostic, so they cannot be
    /// reduced in parallel or combined across restart segments. a census accumulates `m*v`
    /// and `m`, and the reader divides.
    pub op: String,
    /// runtime parameter values, indexed by each `PARAMETER` node's `param_idx`.
    #[serde(default)]
    pub params: Vec<f64>,
    /// the shortest simulation-time interval between samples. `None` samples every step, which is
    /// the finest a run can offer and also the most expensive: a sample is a full extra sweep of
    /// the grid plus its reduction, measured at roughly a third of a hydro step on a small 1d
    /// problem, so sampling every step means paying that on every step.
    ///
    /// an interval in time rather than a step count because it is the time series that is being
    /// recorded: dt varies over a run — with the cfl, with refinement, with the state — so a
    /// fixed step stride produces a non-uniform sampling of the physics, and the spacing of the
    /// resulting series would be an artifact of the timestepper rather than a choice.
    #[serde(default)]
    pub sample_interval: Option<f64>,
    /// fold every sample into one running row rather than storing a row apiece, combining them
    /// with the census's own reduce op.
    ///
    /// what this trades away is the time series; what it buys is that a two-dimensional histogram
    /// costs one row for a whole run segment instead of order a hundred kilobytes per sample. the
    /// row travels with the number of samples folded into it and the times of the first and last,
    /// so a reader forms the time average, and two run segments combine as a sample-count-weighted
    /// sum without either having stored its samples.
    #[serde(default)]
    pub accumulate: bool,
    /// when a refinement hierarchy samples: `"root_step"` (every level reduced into one row at the
    /// boundary where their clocks meet) or `"per_level_step"` (each level on its own subcycle,
    /// tagged with its own time). ignored by a single-level run, where the two are the same
    /// instant.
    #[serde(default = "default_cadence")]
    pub cadence: String,
    /// the flat, topologically-ordered DAG shared by every axis and value expression.
    pub nodes: Vec<NodeDesc>,
}

fn default_cadence() -> String {
    "root_step".to_string()
}

impl CensusConfig {
    /// parse the python-emitted json.
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }

    /// serialize back to json (round-trip / golden tests).
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string(self)
    }

    /// the axis and value expression node indices, axes first — the output order a lowered
    /// census graph produces, and the order the per-cell evaluation unpacks.
    pub fn output_nodes(&self) -> Vec<usize> {
        self.axes
            .iter()
            .map(|a| a.expr)
            .chain(self.values.iter().copied())
            .collect()
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
            "vocabulary":{"reads":["x_0"],"params":[0]},
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
