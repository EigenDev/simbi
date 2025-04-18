#ifndef EXPRESSION_HPP
#define EXPRESSION_HPP

#include "build_options.hpp"

namespace simbi::expression {
    // expanded operation types to match python implementation :)
    enum class ExprOp : int {
        // constants and variables
        CONSTANT,      // represents a literal value
        VARIABLE_X1,   // represents x1 variable
        VARIABLE_X2,   // represents x2 variable (2D)
        VARIABLE_X3,   // represents x3 variable (3D)
        VARIABLE_T,    // represents time variable
        VARIABLE_DT,   // represents time step variable
        PARAMETER,     // runtime parameter reference

        // arithmetic operations
        ADD,        // binary addition
        SUBTRACT,   // binary subtraction (SUB in Python)
        MULTIPLY,   // binary multiplication (MUL in Python)
        DIVIDE,     // binary division (DIV in Python)
        POWER,      // binary power operation (POW in Python)
        NEG,        // unary negation

        // comparison operations
        LT,   // less than
        GT,   // greater than
        EQ,   // equal to
        LE,   // less than or equal to
        GE,   // greater than or equal to

        // logical operations
        AND,   // logical AND
        OR,    // logical OR
        NOT,   // logical NOT

        // math functions
        LOG,     // natural logarithm
        LOG10,   // base-10 logarithm
        ABS,     // absolute value
        SIN,     // sine
        COS,     // cosine
        TAN,     // tangent
        ASIN,    // arc sine
        ACOS,    // arc cosine
        ATAN,    // arc tangent
        EXP,     // exponential
        SQRT,    // square root
        MIN,     // minimum of two values
        MAX,     // maximum of two values

        // special operations
        IF_THEN_ELSE   // ternary conditional operation
    };

    // expression node to support all operations
    struct ExprNode {
        ExprOp op;   // operation type

        //  need a more flexible structure to handle:
        // i. Different parameter types (value, index)
        // ii. Variable number of children (0-3)
        union {
            // for CONSTANT
            real value;

            // for PARAMETER
            int param_idx;

            // for binary and unary operations
            struct {
                int left;
                int right;
            } children;

            // for ternary operations (IF_THEN_ELSE)
            struct {
                int condition;
                int then_expr;
                int else_expr;
            } ternary;
        };

        // constructor for CONSTANT
        static ExprNode constant(real val)
        {
            ExprNode node;
            node.op    = ExprOp::CONSTANT;
            node.value = val;
            return node;
        }

        // constructor for PARAMETER
        static ExprNode parameter(int idx)
        {
            ExprNode node;
            node.op        = ExprOp::PARAMETER;
            node.param_idx = idx;
            return node;
        }

        // constructor for unary operations
        static ExprNode unary(ExprOp op, int left)
        {
            ExprNode node;
            node.op             = op;
            node.children.left  = left;
            node.children.right = -1;   // Unused
            return node;
        }

        // constructor for binary operations
        static ExprNode binary(ExprOp op, int left, int right)
        {
            ExprNode node;
            node.op             = op;
            node.children.left  = left;
            node.children.right = right;
            return node;
        }

        // constructor for ternary operations
        static ExprNode from_ternary(int cond, int then_branch, int else_branch)
        {
            ExprNode node;
            node.op                = ExprOp::IF_THEN_ELSE;
            node.ternary.condition = cond;
            node.ternary.then_expr = then_branch;
            node.ternary.else_expr = else_branch;
            return node;
        }
    };

}   // namespace simbi::expression

#endif
