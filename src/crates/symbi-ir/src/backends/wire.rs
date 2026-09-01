//! flat, versioned wire representation for backend-neutral prepared kernels.
//!
//! `ScalarExpr` is deliberately ergonomic in memory: it is an owned tree. json must
//! not inherit that recursion, however, because valid generated algebra can exceed a
//! generic document parser's nesting limit. this module stores expressions and
//! statements as postorder arenas whose edges are checked integer indices. json depth
//! is therefore independent of kernel algebra depth.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize, de::Error as _};

use super::render::Prepared;
use crate::emit::FieldBinding;
use crate::passes::scalarize::{
    BinaryKind, KernelScalarized, LoweredParam, ScalarExpr, ScalarStmt, UnaryKind,
};
use crate::{ConstValue, DimExpr, ElementTy};
use symbi_abi::{FieldBind, ScalarBind};

const WIRE_VERSION: u8 = 2;
type ExprId = u32;
type StmtId = u32;

#[derive(Serialize, Deserialize)]
struct PreparedWire {
    version: u8,
    kernel_name: String,
    ndim: u8,
    scalarized: FlatKernelScalarized,
    bindings: Vec<FieldBinding>,
    field_inputs: Vec<(crate::InputKey, FieldBind)>,
    field_writes: Vec<FieldBind>,
    scalar_params: Vec<ScalarBind>,
    coord_components: Vec<u8>,
    device_preamble: Vec<String>,
    param_elem: BTreeMap<String, ElementTy>,
    tile_spec: Option<crate::gv::TileSpec>,
    coalesce_layout: bool,
    output_support: Option<crate::support::Support>,
}

#[derive(Serialize, Deserialize)]
struct FlatKernelScalarized {
    params: Vec<LoweredParam>,
    exprs: Vec<FlatExpr>,
    stmts: Vec<FlatStmt>,
    body: Vec<StmtId>,
    outputs: Vec<ExprId>,
}

#[derive(Serialize, Deserialize)]
enum FlatExpr {
    Const(ConstValue),
    Var(String),
    BinOp(BinaryKind, ExprId, ExprId),
    UnaryOp(UnaryKind, ExprId),
    MethodCall {
        receiver: ExprId,
        method: String,
        args: Vec<ExprId>,
    },
    Select {
        cond: ExprId,
        then: ExprId,
        else_: ExprId,
    },
    IndexInto {
        container: String,
        index: ExprId,
    },
    FieldLoadAt {
        field_key: String,
        components: Vec<ExprId>,
    },
    FreeCall {
        name: String,
        args: Vec<ExprId>,
    },
    Cast {
        to: ElementTy,
        value: ExprId,
    },
}

#[derive(Serialize, Deserialize)]
enum FlatStmt {
    Let {
        name: String,
        element: ElementTy,
        value: ExprId,
    },
    LetMut {
        name: String,
        element: ElementTy,
        init: ExprId,
    },
    CompoundAssign {
        name: String,
        op: BinaryKind,
        value: ExprId,
    },
    Assign {
        name: String,
        value: ExprId,
    },
    For {
        iter: String,
        bound: DimExpr,
        body: Vec<StmtId>,
    },
    If {
        cond: ExprId,
        then_body: Vec<StmtId>,
    },
    Break,
    Scope {
        name: String,
        element: ElementTy,
        body: Vec<StmtId>,
        result: ExprId,
    },
    IfElse {
        outs: Vec<(String, ElementTy)>,
        cond: ExprId,
        then_body: Vec<StmtId>,
        else_body: Vec<StmtId>,
    },
}

struct Encoder {
    exprs: Vec<FlatExpr>,
    stmts: Vec<FlatStmt>,
}

impl Encoder {
    fn expr(&mut self, root: &ScalarExpr) -> ExprId {
        enum Frame<'a> {
            Visit(&'a ScalarExpr),
            Build(&'a ScalarExpr),
        }

        let mut frames = vec![Frame::Visit(root)];
        let mut values = Vec::<ExprId>::new();
        while let Some(frame) = frames.pop() {
            match frame {
                Frame::Visit(expr) => {
                    frames.push(Frame::Build(expr));
                    for child in expr.children().into_iter().rev() {
                        frames.push(Frame::Visit(child));
                    }
                }
                Frame::Build(expr) => {
                    let child_count = expr.children().len();
                    let children = values.split_off(values.len() - child_count);
                    let node = match expr {
                        ScalarExpr::Const(value) => FlatExpr::Const(value.clone()),
                        ScalarExpr::Var(name) => FlatExpr::Var(name.clone()),
                        ScalarExpr::BinOp(op, ..) => FlatExpr::BinOp(*op, children[0], children[1]),
                        ScalarExpr::UnaryOp(op, ..) => FlatExpr::UnaryOp(*op, children[0]),
                        ScalarExpr::MethodCall { method, .. } => FlatExpr::MethodCall {
                            receiver: children[0],
                            method: method.clone(),
                            args: children[1..].to_vec(),
                        },
                        ScalarExpr::Select { .. } => FlatExpr::Select {
                            cond: children[0],
                            then: children[1],
                            else_: children[2],
                        },
                        ScalarExpr::IndexInto { container, .. } => FlatExpr::IndexInto {
                            container: container.clone(),
                            index: children[0],
                        },
                        ScalarExpr::FieldLoadAt { field_key, .. } => FlatExpr::FieldLoadAt {
                            field_key: field_key.clone(),
                            components: children,
                        },
                        ScalarExpr::FreeCall { name, .. } => FlatExpr::FreeCall {
                            name: name.clone(),
                            args: children,
                        },
                        ScalarExpr::Cast { to, .. } => FlatExpr::Cast {
                            to: *to,
                            value: children[0],
                        },
                    };
                    let id = u32::try_from(self.exprs.len()).expect("too many scalar expressions");
                    self.exprs.push(node);
                    values.push(id);
                }
            }
        }
        debug_assert_eq!(values.len(), 1);
        values[0]
    }

    fn body(&mut self, body: &[ScalarStmt]) -> Vec<StmtId> {
        body.iter().map(|stmt| self.stmt(stmt)).collect()
    }

    fn stmt(&mut self, stmt: &ScalarStmt) -> StmtId {
        let flat = match stmt {
            ScalarStmt::Let {
                name,
                element,
                value,
            } => FlatStmt::Let {
                name: name.clone(),
                element: *element,
                value: self.expr(value),
            },
            ScalarStmt::LetMut {
                name,
                element,
                init,
            } => FlatStmt::LetMut {
                name: name.clone(),
                element: *element,
                init: self.expr(init),
            },
            ScalarStmt::CompoundAssign { name, op, value } => FlatStmt::CompoundAssign {
                name: name.clone(),
                op: *op,
                value: self.expr(value),
            },
            ScalarStmt::Assign { name, value } => FlatStmt::Assign {
                name: name.clone(),
                value: self.expr(value),
            },
            ScalarStmt::For { iter, bound, body } => FlatStmt::For {
                iter: iter.clone(),
                bound: bound.clone(),
                body: self.body(body),
            },
            ScalarStmt::If { cond, then_body } => FlatStmt::If {
                cond: self.expr(cond),
                then_body: self.body(then_body),
            },
            ScalarStmt::Break => FlatStmt::Break,
            ScalarStmt::Scope {
                name,
                element,
                body,
                result,
            } => FlatStmt::Scope {
                name: name.clone(),
                element: *element,
                body: self.body(body),
                result: self.expr(result),
            },
            ScalarStmt::IfElse {
                outs,
                cond,
                then_body,
                else_body,
            } => FlatStmt::IfElse {
                outs: outs.clone(),
                cond: self.expr(cond),
                then_body: self.body(then_body),
                else_body: self.body(else_body),
            },
        };
        let id = u32::try_from(self.stmts.len()).expect("too many scalar statements");
        self.stmts.push(flat);
        id
    }
}

fn invalid(message: impl std::fmt::Display) -> serde_json::Error {
    serde_json::Error::custom(message.to_string())
}

fn take_expr(
    expressions: &mut [Option<ScalarExpr>],
    id: ExprId,
) -> Result<ScalarExpr, serde_json::Error> {
    expressions
        .get_mut(id as usize)
        .and_then(Option::take)
        .ok_or_else(|| invalid(format!("invalid or forward scalar expression index {id}")))
}

fn decode_expressions(nodes: Vec<FlatExpr>) -> Result<Vec<Option<ScalarExpr>>, serde_json::Error> {
    let mut decoded = Vec::with_capacity(nodes.len());
    for node in nodes {
        let expr = match node {
            FlatExpr::Const(value) => ScalarExpr::Const(value),
            FlatExpr::Var(name) => ScalarExpr::Var(name),
            FlatExpr::BinOp(op, a, b) => ScalarExpr::BinOp(
                op,
                Box::new(take_expr(&mut decoded, a)?),
                Box::new(take_expr(&mut decoded, b)?),
            ),
            FlatExpr::UnaryOp(op, value) => {
                ScalarExpr::UnaryOp(op, Box::new(take_expr(&mut decoded, value)?))
            }
            FlatExpr::MethodCall {
                receiver,
                method,
                args,
            } => ScalarExpr::MethodCall {
                receiver: Box::new(take_expr(&mut decoded, receiver)?),
                method,
                args: args
                    .into_iter()
                    .map(|id| take_expr(&mut decoded, id))
                    .collect::<Result<_, _>>()?,
            },
            FlatExpr::Select { cond, then, else_ } => ScalarExpr::Select {
                cond: Box::new(take_expr(&mut decoded, cond)?),
                then: Box::new(take_expr(&mut decoded, then)?),
                else_: Box::new(take_expr(&mut decoded, else_)?),
            },
            FlatExpr::IndexInto { container, index } => ScalarExpr::IndexInto {
                container,
                index: Box::new(take_expr(&mut decoded, index)?),
            },
            FlatExpr::FieldLoadAt {
                field_key,
                components,
            } => ScalarExpr::FieldLoadAt {
                field_key,
                components: components
                    .into_iter()
                    .map(|id| take_expr(&mut decoded, id))
                    .collect::<Result<_, _>>()?,
            },
            FlatExpr::FreeCall { name, args } => ScalarExpr::FreeCall {
                name,
                args: args
                    .into_iter()
                    .map(|id| take_expr(&mut decoded, id))
                    .collect::<Result<_, _>>()?,
            },
            FlatExpr::Cast { to, value } => ScalarExpr::Cast {
                to,
                value: Box::new(take_expr(&mut decoded, value)?),
            },
        };
        decoded.push(Some(expr));
    }
    Ok(decoded)
}

fn expr(
    expressions: &mut [Option<ScalarExpr>],
    id: ExprId,
) -> Result<ScalarExpr, serde_json::Error> {
    take_expr(expressions, id)
}

fn stmt_body(
    statements: &mut [Option<ScalarStmt>],
    ids: Vec<StmtId>,
) -> Result<Vec<ScalarStmt>, serde_json::Error> {
    ids.into_iter()
        .map(|id| {
            statements
                .get_mut(id as usize)
                .and_then(Option::take)
                .ok_or_else(|| invalid(format!("invalid or forward scalar statement index {id}")))
        })
        .collect()
}

fn decode_statements(
    nodes: Vec<FlatStmt>,
    expressions: &mut [Option<ScalarExpr>],
) -> Result<Vec<Option<ScalarStmt>>, serde_json::Error> {
    let mut decoded = Vec::with_capacity(nodes.len());
    for node in nodes {
        let stmt = match node {
            FlatStmt::Let {
                name,
                element,
                value,
            } => ScalarStmt::Let {
                name,
                element,
                value: expr(expressions, value)?,
            },
            FlatStmt::LetMut {
                name,
                element,
                init,
            } => ScalarStmt::LetMut {
                name,
                element,
                init: expr(expressions, init)?,
            },
            FlatStmt::CompoundAssign { name, op, value } => ScalarStmt::CompoundAssign {
                name,
                op,
                value: expr(expressions, value)?,
            },
            FlatStmt::Assign { name, value } => ScalarStmt::Assign {
                name,
                value: expr(expressions, value)?,
            },
            FlatStmt::For { iter, bound, body } => ScalarStmt::For {
                iter,
                bound,
                body: stmt_body(&mut decoded, body)?,
            },
            FlatStmt::If { cond, then_body } => ScalarStmt::If {
                cond: expr(expressions, cond)?,
                then_body: stmt_body(&mut decoded, then_body)?,
            },
            FlatStmt::Break => ScalarStmt::Break,
            FlatStmt::Scope {
                name,
                element,
                body,
                result,
            } => ScalarStmt::Scope {
                name,
                element,
                body: stmt_body(&mut decoded, body)?,
                result: expr(expressions, result)?,
            },
            FlatStmt::IfElse {
                outs,
                cond,
                then_body,
                else_body,
            } => ScalarStmt::IfElse {
                outs,
                cond: expr(expressions, cond)?,
                then_body: stmt_body(&mut decoded, then_body)?,
                else_body: stmt_body(&mut decoded, else_body)?,
            },
        };
        decoded.push(Some(stmt));
    }
    Ok(decoded)
}

impl From<&Prepared> for PreparedWire {
    fn from(prepared: &Prepared) -> Self {
        let mut encoder = Encoder {
            exprs: vec![],
            stmts: vec![],
        };
        let body = encoder.body(&prepared.scalarized.body);
        let outputs = prepared
            .scalarized
            .outputs
            .iter()
            .map(|output| encoder.expr(output))
            .collect();
        Self {
            version: WIRE_VERSION,
            kernel_name: prepared.kernel_name.clone(),
            ndim: prepared.ndim,
            scalarized: FlatKernelScalarized {
                params: prepared.scalarized.params.clone(),
                exprs: encoder.exprs,
                stmts: encoder.stmts,
                body,
                outputs,
            },
            bindings: prepared.bindings.clone(),
            field_inputs: prepared.field_inputs.clone(),
            field_writes: prepared.field_writes.clone(),
            scalar_params: prepared.scalar_params.clone(),
            coord_components: prepared.coord_components.clone(),
            device_preamble: prepared.device_preamble.clone(),
            param_elem: prepared.param_elem.clone(),
            tile_spec: prepared.tile_spec.clone(),
            coalesce_layout: prepared.coalesce_layout,
            output_support: prepared.output_support.clone(),
        }
    }
}

impl TryFrom<PreparedWire> for Prepared {
    type Error = serde_json::Error;

    fn try_from(wire: PreparedWire) -> Result<Self, Self::Error> {
        if wire.version != WIRE_VERSION {
            return Err(invalid(format!(
                "unsupported Prepared IR wire version {} (expected {WIRE_VERSION})",
                wire.version
            )));
        }
        let mut expressions = decode_expressions(wire.scalarized.exprs)?;
        let mut statements = decode_statements(wire.scalarized.stmts, &mut expressions)?;
        let body = stmt_body(&mut statements, wire.scalarized.body)?;
        let outputs = wire
            .scalarized
            .outputs
            .into_iter()
            .map(|id| expr(&mut expressions, id))
            .collect::<Result<_, _>>()?;
        Ok(Prepared {
            kernel_name: wire.kernel_name,
            ndim: wire.ndim,
            scalarized: KernelScalarized {
                params: wire.scalarized.params,
                body,
                outputs,
            },
            bindings: wire.bindings,
            field_inputs: wire.field_inputs,
            field_writes: wire.field_writes,
            scalar_params: wire.scalar_params,
            coord_components: wire.coord_components,
            device_preamble: wire.device_preamble,
            param_elem: wire.param_elem,
            tile_spec: wire.tile_spec,
            coalesce_layout: wire.coalesce_layout,
            output_support: wire.output_support,
        })
    }
}

pub(super) fn serialize(prepared: &Prepared) -> Result<String, serde_json::Error> {
    serde_json::to_string(&PreparedWire::from(prepared))
}

pub(super) fn deserialize(ir: &str) -> Result<Prepared, serde_json::Error> {
    serde_json::from_str::<PreparedWire>(ir)?.try_into()
}
