#include "evaluator.hpp"
#include <cmath>

// Error handling macro - customize behavior based on compile flags
#ifdef EXPR_STRICT_ERROR_CHECKING
#define HANDLE_ERROR(msg)                                                      \
    printf("[ExprError] %s\n", msg);                                           \
    return NAN;
#else
#define HANDLE_ERROR(msg) return 0.0;
#endif

namespace simbi::expression {

    DEV real evaluate_expr(
        const ExprNode* nodes,
        int node_idx,
        real x1,
        real x2,
        real x3,
        real t,
        real dt,
        const real* parameters
    )
    {
        // bounds check
        if (node_idx < 0) {
            HANDLE_ERROR("Invalid node index");
        }

        const ExprNode& node = nodes[node_idx];

        switch (node.op) {
            // constants and variables
            case ExprOp::CONSTANT: return node.value;

            case ExprOp::VARIABLE_X1: return x1;

            case ExprOp::VARIABLE_X2: return x2;

            case ExprOp::VARIABLE_X3: return x3;

            case ExprOp::VARIABLE_T: return t;

            case ExprOp::VARIABLE_DT: return dt;

            case ExprOp::PARAMETER:
                if (parameters == nullptr) {
                    HANDLE_ERROR("Parameter array is null");
                }
                return parameters[node.param_idx];

            // binary arithmetic operations
            case ExprOp::ADD: {
                real left = evaluate_expr(
                    nodes,
                    node.children.left,
                    x1,
                    x2,
                    x3,
                    t,
                    dt,
                    parameters
                );
                real right = evaluate_expr(
                    nodes,
                    node.children.right,
                    x1,
                    x2,
                    x3,
                    t,
                    dt,
                    parameters
                );
                return left + right;
            }

            case ExprOp::SUBTRACT: {
                real left = evaluate_expr(
                    nodes,
                    node.children.left,
                    x1,
                    x2,
                    x3,
                    t,
                    dt,
                    parameters
                );
                real right = evaluate_expr(
                    nodes,
                    node.children.right,
                    x1,
                    x2,
                    x3,
                    t,
                    dt,
                    parameters
                );
                return left - right;
            }

            case ExprOp::MULTIPLY: {
                real left = evaluate_expr(
                    nodes,
                    node.children.left,
                    x1,
                    x2,
                    x3,
                    t,
                    dt,
                    parameters
                );
                real right = evaluate_expr(
                    nodes,
                    node.children.right,
                    x1,
                    x2,
                    x3,
                    t,
                    dt,
                    parameters
                );
                return left * right;
            }

            case ExprOp::DIVIDE: {
                real left = evaluate_expr(
                    nodes,
                    node.children.left,
                    x1,
                    x2,
                    x3,
                    t,
                    dt,
                    parameters
                );
                real right = evaluate_expr(
                    nodes,
                    node.children.right,
                    x1,
                    x2,
                    x3,
                    t,
                    dt,
                    parameters
                );
                if (right == 0.0) {
                    HANDLE_ERROR("Division by zero");
                }
                return left / right;
            }

            case ExprOp::POWER: {
                real base = evaluate_expr(
                    nodes,
                    node.children.left,
                    x1,
                    x2,
                    x3,
                    t,
                    dt,
                    parameters
                );
                real exponent = evaluate_expr(
                    nodes,
                    node.children.right,
                    x1,
                    x2,
                    x3,
                    t,
                    dt,
                    parameters
                );
                // Handle potential domain errors
                if (base == 0.0 && exponent < 0.0) {
                    HANDLE_ERROR("Zero raised to negative power");
                }
                if (base < 0.0 && std::floor(exponent) != exponent) {
                    HANDLE_ERROR("Negative base with non-integer exponent");
                }
                return std::pow(base, exponent);
            }

            // unary operations
            case ExprOp::NEG: {
                real value = evaluate_expr(
                    nodes,
                    node.children.left,
                    x1,
                    x2,
                    x3,
                    t,
                    dt,
                    parameters
                );
                return -value;
            }

            // comparison operations - return 1.0 for true, 0.0 for false
            case ExprOp::LT: {
                real left = evaluate_expr(
                    nodes,
                    node.children.left,
                    x1,
                    x2,
                    x3,
                    t,
                    dt,
                    parameters
                );
                real right = evaluate_expr(
                    nodes,
                    node.children.right,
                    x1,
                    x2,
                    x3,
                    t,
                    dt,
                    parameters
                );
                return left < right ? 1.0 : 0.0;
            }

            case ExprOp::GT: {
                real left = evaluate_expr(
                    nodes,
                    node.children.left,
                    x1,
                    x2,
                    x3,
                    t,
                    dt,
                    parameters
                );
                real right = evaluate_expr(
                    nodes,
                    node.children.right,
                    x1,
                    x2,
                    x3,
                    t,
                    dt,
                    parameters
                );
                return left > right ? 1.0 : 0.0;
            }

            case ExprOp::EQ: {
                real left = evaluate_expr(
                    nodes,
                    node.children.left,
                    x1,
                    x2,
                    x3,
                    t,
                    dt,
                    parameters
                );
                real right = evaluate_expr(
                    nodes,
                    node.children.right,
                    x1,
                    x2,
                    x3,
                    t,
                    dt,
                    parameters
                );
                return left == right ? 1.0 : 0.0;
            }

            case ExprOp::LE: {
                real left = evaluate_expr(
                    nodes,
                    node.children.left,
                    x1,
                    x2,
                    x3,
                    t,
                    dt,
                    parameters
                );
                real right = evaluate_expr(
                    nodes,
                    node.children.right,
                    x1,
                    x2,
                    x3,
                    t,
                    dt,
                    parameters
                );
                return left <= right ? 1.0 : 0.0;
            }

            case ExprOp::GE: {
                real left = evaluate_expr(
                    nodes,
                    node.children.left,
                    x1,
                    x2,
                    x3,
                    t,
                    dt,
                    parameters
                );
                real right = evaluate_expr(
                    nodes,
                    node.children.right,
                    x1,
                    x2,
                    x3,
                    t,
                    dt,
                    parameters
                );
                return left >= right ? 1.0 : 0.0;
            }

            // logical operations
            case ExprOp::AND: {
                real left = evaluate_expr(
                    nodes,
                    node.children.left,
                    x1,
                    x2,
                    x3,
                    t,
                    dt,
                    parameters
                );
                // short-circuit evaluation
                if (left == 0.0) {
                    return 0.0;
                }
                real right = evaluate_expr(
                    nodes,
                    node.children.right,
                    x1,
                    x2,
                    x3,
                    t,
                    dt,
                    parameters
                );
                return (left != 0.0 && right != 0.0) ? 1.0 : 0.0;
            }

            case ExprOp::OR: {
                real left = evaluate_expr(
                    nodes,
                    node.children.left,
                    x1,
                    x2,
                    x3,
                    t,
                    dt,
                    parameters
                );
                // short-circuit evaluation
                if (left != 0.0) {
                    return 1.0;
                }
                real right = evaluate_expr(
                    nodes,
                    node.children.right,
                    x1,
                    x2,
                    x3,
                    t,
                    dt,
                    parameters
                );
                return (left != 0.0 || right != 0.0) ? 1.0 : 0.0;
            }

            case ExprOp::NOT: {
                real value = evaluate_expr(
                    nodes,
                    node.children.left,
                    x1,
                    x2,
                    x3,
                    t,
                    dt,
                    parameters
                );
                return value == 0.0 ? 1.0 : 0.0;
            }

            // min/max functions
            case ExprOp::MIN: {
                real left = evaluate_expr(
                    nodes,
                    node.children.left,
                    x1,
                    x2,
                    x3,
                    t,
                    dt,
                    parameters
                );
                real right = evaluate_expr(
                    nodes,
                    node.children.right,
                    x1,
                    x2,
                    x3,
                    t,
                    dt,
                    parameters
                );
                return left < right ? left : right;
            }

            case ExprOp::MAX: {
                real left = evaluate_expr(
                    nodes,
                    node.children.left,
                    x1,
                    x2,
                    x3,
                    t,
                    dt,
                    parameters
                );
                real right = evaluate_expr(
                    nodes,
                    node.children.right,
                    x1,
                    x2,
                    x3,
                    t,
                    dt,
                    parameters
                );
                return left > right ? left : right;
            }

            // math functions
            case ExprOp::SIN: {
                real value = evaluate_expr(
                    nodes,
                    node.children.left,
                    x1,
                    x2,
                    x3,
                    t,
                    dt,
                    parameters
                );
                return std::sin(value);
            }

            case ExprOp::COS: {
                real value = evaluate_expr(
                    nodes,
                    node.children.left,
                    x1,
                    x2,
                    x3,
                    t,
                    dt,
                    parameters
                );
                return std::cos(value);
            }

            case ExprOp::TAN: {
                real value = evaluate_expr(
                    nodes,
                    node.children.left,
                    x1,
                    x2,
                    x3,
                    t,
                    dt,
                    parameters
                );
                return std::tan(value);
            }

            case ExprOp::LOG: {
                real value = evaluate_expr(
                    nodes,
                    node.children.left,
                    x1,
                    x2,
                    x3,
                    t,
                    dt,
                    parameters
                );
                if (value <= 0.0) {
                    HANDLE_ERROR("Log of non-positive number");
                }
                return std::log(value);
            }

            case ExprOp::LOG10: {
                real value = evaluate_expr(
                    nodes,
                    node.children.left,
                    x1,
                    x2,
                    x3,
                    t,
                    dt,
                    parameters
                );
                if (value <= 0.0) {
                    HANDLE_ERROR("Log10 of non-positive number");
                }
                return std::log10(value);
            }

            case ExprOp::EXP: {
                real value = evaluate_expr(
                    nodes,
                    node.children.left,
                    x1,
                    x2,
                    x3,
                    t,
                    dt,
                    parameters
                );
                return std::exp(value);
            }

            case ExprOp::ABS: {
                real value = evaluate_expr(
                    nodes,
                    node.children.left,
                    x1,
                    x2,
                    x3,
                    t,
                    dt,
                    parameters
                );
                return std::abs(value);
            }

            case ExprOp::SQRT: {
                real value = evaluate_expr(
                    nodes,
                    node.children.left,
                    x1,
                    x2,
                    x3,
                    t,
                    dt,
                    parameters
                );
                if (value < 0.0) {
                    HANDLE_ERROR("Square root of negative number");
                }
                return std::sqrt(value);
            }

            case ExprOp::ASIN: {
                real value = evaluate_expr(
                    nodes,
                    node.children.left,
                    x1,
                    x2,
                    x3,
                    t,
                    dt,
                    parameters
                );
                if (value < -1.0 || value > 1.0) {
                    HANDLE_ERROR("Arc sine argument out of range");
                }
                return std::asin(value);
            }

            case ExprOp::ACOS: {
                real value = evaluate_expr(
                    nodes,
                    node.children.left,
                    x1,
                    x2,
                    x3,
                    t,
                    dt,
                    parameters
                );
                if (value < -1.0 || value > 1.0) {
                    HANDLE_ERROR("Arc cosine argument out of range");
                }
                return std::acos(value);
            }

            case ExprOp::ATAN: {
                real value = evaluate_expr(
                    nodes,
                    node.children.left,
                    x1,
                    x2,
                    x3,
                    t,
                    dt,
                    parameters
                );
                return std::atan(value);
            }

            // conditional operation
            case ExprOp::IF_THEN_ELSE: {
                real condition = evaluate_expr(
                    nodes,
                    node.ternary.condition,
                    x1,
                    x2,
                    x3,
                    t,
                    dt,
                    parameters
                );
                if (condition != 0.0) {
                    return evaluate_expr(
                        nodes,
                        node.ternary.then_expr,
                        x1,
                        x2,
                        x3,
                        t,
                        dt,
                        parameters
                    );
                }
                else {
                    return evaluate_expr(
                        nodes,
                        node.ternary.else_expr,
                        x1,
                        x2,
                        x3,
                        t,
                        dt,
                        parameters
                    );
                }
            }

            default: {
                // unknown operation
                HANDLE_ERROR("Unknown operation type");
            }
        }
    }

    // non-recursive implementation using a simple stack
    // this avoids stack overflow for deeply nested expressions
    DEV real evaluate_expr_nonrecursive(
        const ExprNode* nodes,
        int node_idx,
        real x1,
        real x2,
        real x3,
        real t,
        real dt,
        const real* parameters
    )
    {
        // based on expected expression depth (doubtful it will be more than 64)
        constexpr int MAX_STACK = 64;

        // stack to track nodes to evaluate
        struct StackEntry {
            int node_idx;
            real value;
            bool evaluated;
        };

        StackEntry stack[MAX_STACK];
        int stack_top = 0;

        // push root node
        stack[stack_top++] = {node_idx, 0.0, false};

        while (stack_top > 0) {
            // check current stack entry
            StackEntry& entry = stack[stack_top - 1];

            // if already evaluated, pop and use result
            if (entry.evaluated) {
                stack_top--;
                if (stack_top > 0) {
                    // Pass result to parent
                    stack[stack_top - 1].value     = entry.value;
                    stack[stack_top - 1].evaluated = true;
                }
                continue;
            }

            const ExprNode& node = nodes[entry.node_idx];

            // process based on operation type
            switch (node.op) {
                // leaf nodes that can be evaluated immediately
                case ExprOp::CONSTANT:
                    entry.value     = node.value;
                    entry.evaluated = true;
                    break;

                case ExprOp::VARIABLE_X1:
                    entry.value     = x1;
                    entry.evaluated = true;
                    break;

                case ExprOp::VARIABLE_X2:
                    entry.value     = x2;
                    entry.evaluated = true;
                    break;

                case ExprOp::VARIABLE_X3:
                    entry.value     = x3;
                    entry.evaluated = true;
                    break;

                case ExprOp::VARIABLE_T:
                    entry.value     = t;
                    entry.evaluated = true;
                    break;

                case ExprOp::VARIABLE_DT:
                    entry.value     = dt;
                    entry.evaluated = true;
                    break;

                case ExprOp::PARAMETER:
                    if (parameters == nullptr) {
                        entry.value = 0.0;
                    }
                    else {
                        entry.value = parameters[node.param_idx];
                    }
                    entry.evaluated = true;
                    break;

                // for binary operations, we need to evaluate both children
                case ExprOp::ADD:
                case ExprOp::SUBTRACT:
                case ExprOp::MULTIPLY:
                case ExprOp::DIVIDE:
                case ExprOp::POWER:
                case ExprOp::LT:
                case ExprOp::GT:
                case ExprOp::EQ:
                case ExprOp::LE:
                case ExprOp::GE:
                case ExprOp::AND:
                case ExprOp::OR:
                case ExprOp::MIN:
                case ExprOp::MAX: {
                    // push right child onto stack (will be evaluated first)
                    if (stack_top >= MAX_STACK - 1) {
                        // stack overflow (!)
                        return 0.0;
                    }
                    stack[stack_top++] = {node.children.right, 0.0, false};

                    // push left child onto stack
                    if (stack_top >= MAX_STACK - 1) {
                        // stack overflow (!)
                        return 0.0;
                    }
                    stack[stack_top++] = {node.children.left, 0.0, false};

                    // ,ark this entry as waiting for children
                    entry.evaluated = false;
                } break;

                // for unary operations, we need to evaluate the operand
                case ExprOp::NEG:
                case ExprOp::NOT:
                case ExprOp::SIN:
                case ExprOp::COS:
                case ExprOp::TAN:
                case ExprOp::LOG:
                case ExprOp::LOG10:
                case ExprOp::EXP:
                case ExprOp::ABS:
                case ExprOp::SQRT:
                case ExprOp::ASIN:
                case ExprOp::ACOS:
                case ExprOp::ATAN: {
                    // push operand onto stack
                    if (stack_top >= MAX_STACK - 1) {
                        // stack overflow
                        return 0.0;
                    }
                    stack[stack_top++] = {node.children.left, 0.0, false};

                    // mark this entry as waiting for operand
                    entry.evaluated = false;
                } break;

                // for ternary operations
                case ExprOp::IF_THEN_ELSE: {
                    // push condition onto stack
                    if (stack_top >= MAX_STACK - 1) {
                        // stack overflow
                        return 0.0;
                    }
                    stack[stack_top++] = {node.ternary.condition, 0.0, false};

                    // the rest will be handled after condition is evaluated
                    entry.evaluated = false;
                } break;

                default:
                    // unknown operation
                    entry.value     = 0.0;
                    entry.evaluated = true;
                    break;
            }

            // continue evaluation if this node has been marked as evaluated
            if (entry.evaluated) {
                continue;
            }

            // for nodes waiting on children/operands
            if (stack[stack_top - 1].evaluated) {
                // child/operand has been evaluated

                switch (node.op) {
                    // binary operations
                    case ExprOp::ADD: {
                        real right      = stack[--stack_top].value;
                        real left       = stack[--stack_top].value;
                        entry.value     = left + right;
                        entry.evaluated = true;
                    } break;

                    case ExprOp::SUBTRACT: {
                        real right      = stack[--stack_top].value;
                        real left       = stack[--stack_top].value;
                        entry.value     = left - right;
                        entry.evaluated = true;
                    } break;

                    case ExprOp::MULTIPLY: {
                        real right      = stack[--stack_top].value;
                        real left       = stack[--stack_top].value;
                        entry.value     = left * right;
                        entry.evaluated = true;
                    } break;

                    case ExprOp::DIVIDE: {
                        real right = stack[--stack_top].value;
                        real left  = stack[--stack_top].value;
                        if (right == 0.0) {
                            entry.value = 0.0;   // Handle division by zero
                        }
                        else {
                            entry.value = left / right;
                        }
                        entry.evaluated = true;
                    } break;

                    case ExprOp::POWER: {
                        real exponent = stack[--stack_top].value;
                        real base     = stack[--stack_top].value;
                        // Handle potential domain errors
                        if (base == 0.0 && exponent <= 0.0) {
                            entry.value =
                                0.0;   // Zero raised to zero or negative power
                        }
                        else if (base < 0.0 &&
                                 std::floor(exponent) != exponent) {
                            entry.value = 0.0;   // Negative base with
                                                 // non-integer exponent
                        }
                        else {
                            entry.value = std::pow(base, exponent);
                        }
                        entry.evaluated = true;
                    } break;

                    case ExprOp::LT: {
                        real right      = stack[--stack_top].value;
                        real left       = stack[--stack_top].value;
                        entry.value     = (left < right) ? 1.0 : 0.0;
                        entry.evaluated = true;
                    } break;

                    case ExprOp::GT: {
                        real right      = stack[--stack_top].value;
                        real left       = stack[--stack_top].value;
                        entry.value     = (left > right) ? 1.0 : 0.0;
                        entry.evaluated = true;
                    } break;

                    case ExprOp::EQ: {
                        real right      = stack[--stack_top].value;
                        real left       = stack[--stack_top].value;
                        entry.value     = (left == right) ? 1.0 : 0.0;
                        entry.evaluated = true;
                    } break;

                    case ExprOp::LE: {
                        real right      = stack[--stack_top].value;
                        real left       = stack[--stack_top].value;
                        entry.value     = (left <= right) ? 1.0 : 0.0;
                        entry.evaluated = true;
                    } break;

                    case ExprOp::GE: {
                        real right      = stack[--stack_top].value;
                        real left       = stack[--stack_top].value;
                        entry.value     = (left >= right) ? 1.0 : 0.0;
                        entry.evaluated = true;
                    } break;

                    case ExprOp::AND: {
                        real right  = stack[--stack_top].value;
                        real left   = stack[--stack_top].value;
                        entry.value = (left != 0.0 && right != 0.0) ? 1.0 : 0.0;
                        entry.evaluated = true;
                    } break;

                    case ExprOp::OR: {
                        real right  = stack[--stack_top].value;
                        real left   = stack[--stack_top].value;
                        entry.value = (left != 0.0 || right != 0.0) ? 1.0 : 0.0;
                        entry.evaluated = true;
                    } break;

                    case ExprOp::MIN: {
                        real right      = stack[--stack_top].value;
                        real left       = stack[--stack_top].value;
                        entry.value     = (left < right) ? left : right;
                        entry.evaluated = true;
                    } break;

                    case ExprOp::MAX: {
                        real right      = stack[--stack_top].value;
                        real left       = stack[--stack_top].value;
                        entry.value     = (left > right) ? left : right;
                        entry.evaluated = true;
                    } break;

                    // unary operations
                    case ExprOp::NEG: {
                        real operand    = stack[--stack_top].value;
                        entry.value     = -operand;
                        entry.evaluated = true;
                    } break;

                    case ExprOp::SQRT: {
                        real operand = stack[--stack_top].value;
                        if (operand < 0.0) {
                            entry.value = 0.0;   // Handle sqrt of negative
                        }
                        else {
                            entry.value = std::sqrt(operand);
                        }
                        entry.evaluated = true;
                    } break;

                    case ExprOp::ABS: {
                        real operand    = stack[--stack_top].value;
                        entry.value     = std::abs(operand);
                        entry.evaluated = true;
                    } break;

                    case ExprOp::SIN: {
                        real operand    = stack[--stack_top].value;
                        entry.value     = std::sin(operand);
                        entry.evaluated = true;
                    } break;

                    case ExprOp::COS: {
                        real operand    = stack[--stack_top].value;
                        entry.value     = std::cos(operand);
                        entry.evaluated = true;
                    } break;

                    case ExprOp::TAN: {
                        real operand    = stack[--stack_top].value;
                        entry.value     = std::tan(operand);
                        entry.evaluated = true;
                    } break;

                    case ExprOp::LOG: {
                        real operand = stack[--stack_top].value;
                        if (operand <= 0.0) {
                            entry.value = 0.0;   // Handle log of non-positive
                        }
                        else {
                            entry.value = std::log(operand);
                        }
                        entry.evaluated = true;
                    } break;

                    case ExprOp::LOG10: {
                        real operand = stack[--stack_top].value;
                        if (operand <= 0.0) {
                            entry.value = 0.0;   // Handle log10 of non-positive
                        }
                        else {
                            entry.value = std::log10(operand);
                        }
                        entry.evaluated = true;
                    } break;

                    case ExprOp::EXP: {
                        real operand    = stack[--stack_top].value;
                        entry.value     = std::exp(operand);
                        entry.evaluated = true;
                    } break;

                    case ExprOp::ASIN: {
                        real operand = stack[--stack_top].value;
                        if (operand < -1.0 || operand > 1.0) {
                            entry.value = 0.0;   // Handle asin out of range
                        }
                        else {
                            entry.value = std::asin(operand);
                        }
                        entry.evaluated = true;
                    } break;

                    case ExprOp::ACOS: {
                        real operand = stack[--stack_top].value;
                        if (operand < -1.0 || operand > 1.0) {
                            entry.value = 0.0;   // Handle acos out of range
                        }
                        else {
                            entry.value = std::acos(operand);
                        }
                        entry.evaluated = true;
                    } break;

                    case ExprOp::ATAN: {
                        real operand    = stack[--stack_top].value;
                        entry.value     = std::atan(operand);
                        entry.evaluated = true;
                    } break;

                    // ternary operation (if-then-else)
                    case ExprOp::IF_THEN_ELSE: {
                        real condition = stack[--stack_top].value;
                        if (condition != 0.0) {
                            // push 'then' branch
                            stack[stack_top++] =
                                {node.ternary.then_expr, 0.0, false};
                        }
                        else {
                            // push 'else' branch
                            stack[stack_top++] =
                                {node.ternary.else_expr, 0.0, false};
                        }
                    } break;

                    default:
                        // unknown op !
                        entry.value     = 0.0;
                        entry.evaluated = true;
                        break;
                }
            }
        }

        // return the final result
        return stack[0].value;
    }

    // batch evaluation for multiple points
    DEV void evaluate_expr_batch(
        const ExprNode* nodes,
        int root_idx,
        const real* x1_values,
        const real* x2_values,
        const real* x3_values,
        real t,
        real dt,
        const real* parameters,
        real* results,
        int count
    )
    {
        for (int i = 0; i < count; ++i) {
            results[i] = evaluate_expr(
                nodes,
                root_idx,
                x1_values[i],
                x2_values ? x2_values[i] : 0.0,
                x3_values ? x3_values[i] : 0.0,
                t,
                dt,
                parameters
            );
        }
    }

    // evaluate a vector of expressions
    DEV void evaluate_expr_vector(
        const ExprNode* nodes,
        const int* root_indices,
        int num_components,
        real x1,
        real x2,
        real x3,
        real t,
        const real* parameters,
        real* results,
        real dt
    )
    {
        for (int i = 0; i < num_components; ++i) {
            results[i] = evaluate_expr(
                nodes,
                root_indices[i],
                x1,
                x2,
                x3,
                t,
                dt,
                parameters
            );
        }
    }

}   // namespace simbi::expression
