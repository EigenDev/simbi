// =============================================================================
// op.rs
//
// operation codes for the expression DAG and VM instruction stream.
// single flat enum shared by dag nodes and linearized instructions.
// string parsing for config deserialization, arity for validation.
//
// usage:
//   let op = Op::from_name("ADD").unwrap();
//   assert_eq!(op.arity(), 2);
// =============================================================================

/// operation code for expression DAG nodes and VM instructions.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Op {
    // leaf (arity 0)
    Constant,
    VariableX1,
    VariableX2,
    VariableX3,
    VariableT,
    Parameter,

    // per-cell STATE variables (arity 0): the fluid state a source/expression reads at the
    // cell. unlike the coordinate variables above (which the standalone VM pre-loads), these
    // are resolved by the source bridge into carrier field reads (`rho`/`vel_k`/`pre`); they
    // are NOT VM registers, so they appear only on the source-lowering path, never the VM.
    VariableRho,
    VariableVel1,
    VariableVel2,
    VariableVel3,
    VariablePressure,

    // the cell's lab-frame volume measure dV (arity 0). the natural weight for every
    // extensive quantity, and what keeps a sum correct on a curvilinear grid, where the
    // measure is r^2 sin(theta) dr dtheta dphi rather than dx^3. resolved like the state
    // leaves above — a bridge-level per-cell read, never a VM register.
    VariableCellVolume,

    // arithmetic
    Add,
    Sub,
    Mul,
    Div,
    Pow,
    Neg,

    // comparison
    Lt,
    Gt,
    Eq,
    Le,
    Ge,

    // logical
    And,
    Or,
    Not,

    // math (unary)
    Log,
    Log10,
    Abs,
    Sin,
    Cos,
    Tan,
    Asin,
    Acos,
    Atan,
    Exp,
    Sqrt,
    Sinh,
    Cosh,
    Tanh,
    Asinh,
    Acosh,
    Atanh,
    Sgn,
    Ceil,
    Floor,

    // math (binary)
    Min,
    Max,
    Mod,
    Atan2,

    // ternary
    IfThenElse,
}

impl Op {
    /// number of child operands: 0 (leaf), 1 (unary), 2 (binary), 3 (ternary).
    pub fn arity(self) -> usize {
        match self {
            // leaf
            Op::Constant
            | Op::VariableX1
            | Op::VariableX2
            | Op::VariableX3
            | Op::VariableT
            | Op::Parameter
            | Op::VariableRho
            | Op::VariableVel1
            | Op::VariableVel2
            | Op::VariableVel3
            | Op::VariablePressure
            | Op::VariableCellVolume => 0,

            // unary
            Op::Neg
            | Op::Not
            | Op::Log
            | Op::Log10
            | Op::Abs
            | Op::Sin
            | Op::Cos
            | Op::Tan
            | Op::Asin
            | Op::Acos
            | Op::Atan
            | Op::Exp
            | Op::Sqrt
            | Op::Sinh
            | Op::Cosh
            | Op::Tanh
            | Op::Asinh
            | Op::Acosh
            | Op::Atanh
            | Op::Sgn
            | Op::Ceil
            | Op::Floor => 1,

            // binary
            Op::Add
            | Op::Sub
            | Op::Mul
            | Op::Div
            | Op::Pow
            | Op::Lt
            | Op::Gt
            | Op::Eq
            | Op::Le
            | Op::Ge
            | Op::And
            | Op::Or
            | Op::Min
            | Op::Max
            | Op::Mod
            | Op::Atan2 => 2,

            // ternary
            Op::IfThenElse => 3,
        }
    }

    /// parse from the config string format
    pub fn from_name(s: &str) -> Option<Op> {
        match s {
            "CONSTANT" => Some(Op::Constant),
            "VARIABLE_X1" => Some(Op::VariableX1),
            "VARIABLE_X2" => Some(Op::VariableX2),
            "VARIABLE_X3" => Some(Op::VariableX3),
            "VARIABLE_T" => Some(Op::VariableT),
            "PARAMETER" => Some(Op::Parameter),
            "VARIABLE_RHO" => Some(Op::VariableRho),
            "VARIABLE_VEL1" => Some(Op::VariableVel1),
            "VARIABLE_VEL2" => Some(Op::VariableVel2),
            "VARIABLE_VEL3" => Some(Op::VariableVel3),
            "VARIABLE_PRESSURE" => Some(Op::VariablePressure),
            "VARIABLE_DV" => Some(Op::VariableCellVolume),
            "ADD" => Some(Op::Add),
            "SUBTRACT" => Some(Op::Sub),
            "MULTIPLY" => Some(Op::Mul),
            "DIVIDE" => Some(Op::Div),
            "POW" => Some(Op::Pow),
            "NEG" => Some(Op::Neg),
            "LT" => Some(Op::Lt),
            "GT" => Some(Op::Gt),
            "EQ" => Some(Op::Eq),
            "LE" => Some(Op::Le),
            "GE" => Some(Op::Ge),
            "AND" => Some(Op::And),
            "OR" => Some(Op::Or),
            "NOT" => Some(Op::Not),
            "LOG" => Some(Op::Log),
            "LOG10" => Some(Op::Log10),
            "ABS" => Some(Op::Abs),
            "SIN" => Some(Op::Sin),
            "COS" => Some(Op::Cos),
            "TAN" => Some(Op::Tan),
            "ASIN" => Some(Op::Asin),
            "ACOS" => Some(Op::Acos),
            "ATAN" => Some(Op::Atan),
            "EXP" => Some(Op::Exp),
            "SQRT" => Some(Op::Sqrt),
            "SINH" => Some(Op::Sinh),
            "COSH" => Some(Op::Cosh),
            "TANH" => Some(Op::Tanh),
            "ASINH" => Some(Op::Asinh),
            "ACOSH" => Some(Op::Acosh),
            "ATANH" => Some(Op::Atanh),
            "SGN" => Some(Op::Sgn),
            "CEIL" => Some(Op::Ceil),
            "FLOOR" => Some(Op::Floor),
            "MIN" => Some(Op::Min),
            "MAX" => Some(Op::Max),
            "MOD" => Some(Op::Mod),
            "ATAN2" => Some(Op::Atan2),
            "IF_THEN_ELSE" => Some(Op::IfThenElse),
            _ => None,
        }
    }

    /// whether this op is a variable reference (x1, x2, x3, t).
    pub fn is_variable(self) -> bool {
        matches!(
            self,
            Op::VariableX1 | Op::VariableX2 | Op::VariableX3 | Op::VariableT
        )
    }

    /// fixed register index for variable ops. panics on non-variable ops.
    pub fn variable_register(self) -> usize {
        match self {
            Op::VariableX1 => 0,
            Op::VariableX2 => 1,
            Op::VariableX3 => 2,
            Op::VariableT => 3,
            _ => panic!("not a variable op: {:?}", self),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arity_leaves() {
        assert_eq!(Op::Constant.arity(), 0);
        assert_eq!(Op::VariableX1.arity(), 0);
        assert_eq!(Op::Parameter.arity(), 0);
    }

    #[test]
    fn arity_unary() {
        assert_eq!(Op::Neg.arity(), 1);
        assert_eq!(Op::Sin.arity(), 1);
        assert_eq!(Op::Not.arity(), 1);
    }

    #[test]
    fn arity_binary() {
        assert_eq!(Op::Add.arity(), 2);
        assert_eq!(Op::Mul.arity(), 2);
        assert_eq!(Op::Lt.arity(), 2);
        assert_eq!(Op::Atan2.arity(), 2);
    }

    #[test]
    fn arity_ternary() {
        assert_eq!(Op::IfThenElse.arity(), 3);
    }

    #[test]
    fn from_name_roundtrip() {
        assert_eq!(Op::from_name("ADD"), Some(Op::Add));
        assert_eq!(Op::from_name("MULTIPLY"), Some(Op::Mul));
        assert_eq!(Op::from_name("POW"), Some(Op::Pow));
        assert_eq!(Op::from_name("IF_THEN_ELSE"), Some(Op::IfThenElse));
        assert_eq!(Op::from_name("NONSENSE"), None);
    }

    #[test]
    fn variable_registers() {
        assert_eq!(Op::VariableX1.variable_register(), 0);
        assert_eq!(Op::VariableX2.variable_register(), 1);
        assert_eq!(Op::VariableX3.variable_register(), 2);
        assert_eq!(Op::VariableT.variable_register(), 3);
    }

    #[test]
    fn is_variable() {
        assert!(Op::VariableX1.is_variable());
        assert!(Op::VariableT.is_variable());
        assert!(!Op::Constant.is_variable());
        assert!(!Op::Add.is_variable());
    }
}
