#include "evaluator.hpp"
#include "build_options.hpp"
#include "util/math/expression.hpp"
#include "util/tools/helpers.hpp"
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
    enum class EvalError {
        NONE,
        INVALID_NODE_INDEX,
        NULL_PARAMETER_ARRAY,
        DIVISION_BY_ZERO,
        ZERO_TO_NEGATIVE_POWER,
        NEGATIVE_BASE_NON_INTEGER_EXPONENT,
        STACK_OVERFLOW,
        INFINITE_LOOP,
        UNKNOWN_OPERATION
    };

    int get_max_register(const ndarray<LinearExprInstr>& instructions)
    {
        int max_reg = 4;   // Start with input registers

        for (size_t i = 0; i < instructions.size(); i++) {
            const auto& instr = instructions[i];
            max_reg           = std::max(max_reg, instr.result_register);

            // Also check operands for completeness
            if (instr.op != ExprOp::CONSTANT && instr.op != ExprOp::PARAMETER) {
                max_reg = std::max(max_reg, instr.register_operands.operand1);
                max_reg = std::max(max_reg, instr.register_operands.operand2);
                max_reg = std::max(max_reg, instr.register_operands.operand3);
            }
        }

        return max_reg;
    }

    DEV void evaluate_linear_expr(
        const LinearExprInstr* instructions,
        size_type instruction_count,
        const int* mapped_output_indices,
        size_type output_count,
        int reg_count,
        real x1,
        real x2,
        real x3,
        real t,
        real dt,
        const real* parameters,
        real* outputs
    )
    {
        // Create register bank
        // Use a fixed size array for performance - adjust MAX_REGISTERS as
        // needed
        constexpr int MAX_REGISTERS   = 256;
        real registers[MAX_REGISTERS] = {0.0};

        // Initialize input registers
        registers[0] = x1;
        registers[1] = x2;
        registers[2] = x3;
        registers[3] = t;
        registers[4] = dt;   // if needed

        // Execute each instruction in reverse order
        for (size_type ii = 0; ii < instruction_count; ii++) {
            const auto& instr    = instructions[ii];
            const int result_reg = instr.result_register;

            // make sure reg index is valid
            if (result_reg < 0 || result_reg >= MAX_REGISTERS) {
                continue;   // this should never happen
            }

            switch (instr.op) {
                case ExprOp::CONSTANT:
                    registers[result_reg] = instr.constant_eval;
                    break;

                case ExprOp::VARIABLE_X1:
                case ExprOp::VARIABLE_X2:
                case ExprOp::VARIABLE_X3:
                case ExprOp::VARIABLE_T:
                case ExprOp::VARIABLE_DT:
                    // Already handled in initialization
                    break;

                case ExprOp::PARAMETER:
                    registers[result_reg] = parameters[instr.parameter_idx];
                    break;

                case ExprOp::ADD:
                    registers[result_reg] =
                        registers[instr.register_operands.operand1] +
                        registers[instr.register_operands.operand2];
                    break;

                case ExprOp::SUBTRACT:
                    registers[result_reg] =
                        registers[instr.register_operands.operand1] -
                        registers[instr.register_operands.operand2];
                    break;

                case ExprOp::MULTIPLY:
                    registers[result_reg] =
                        registers[instr.register_operands.operand1] *
                        registers[instr.register_operands.operand2];
                    break;
                case ExprOp::DIVIDE: {
                    real denominator =
                        registers[instr.register_operands.operand2];
                    if (denominator == 0.0) {
                        printf(
                            "[ExprError] Division by zero in instruction %zu\n",
                            ii
                        );
                        registers[result_reg] = 0.0;   // Handle gracefully
                        break;
                    }
                    registers[result_reg] =
                        registers[instr.register_operands.operand1] /
                        denominator;
                    break;
                }

                case ExprOp::POWER: {
                    real base     = registers[instr.register_operands.operand1];
                    real exponent = registers[instr.register_operands.operand2];
                    // Handle potential domain errors
                    if (base == 0.0 && exponent < 0.0) {
                        printf(
                            "[ExprError] Zero raised to negative power in "
                            "instruction %zu\n",
                            ii
                        );
                    }
                    if (base < 0.0 && std::floor(exponent) != exponent) {
                        printf(
                            "[ExprError] Negative base with non-integer "
                            "exponent in instruction %zu\n",
                            ii
                        );
                    }
                    registers[result_reg] = std::pow(base, exponent);
                    break;
                }

                case ExprOp::NEG:
                    registers[result_reg] =
                        -registers[instr.register_operands.operand1];
                    break;
                case ExprOp::LT:
                    registers[result_reg] =
                        registers[instr.register_operands.operand1] <
                                registers[instr.register_operands.operand2]
                            ? 1.0
                            : 0.0;
                    break;
                case ExprOp::GT:
                    registers[result_reg] =
                        registers[instr.register_operands.operand1] >
                                registers[instr.register_operands.operand2]
                            ? 1.0
                            : 0.0;
                    break;
                case ExprOp::EQ:
                    registers[result_reg] =
                        registers[instr.register_operands.operand1] ==
                                registers[instr.register_operands.operand2]
                            ? 1.0
                            : 0.0;
                    break;
                case ExprOp::LE:
                    registers[result_reg] =
                        registers[instr.register_operands.operand1] <=
                                registers[instr.register_operands.operand2]
                            ? 1.0
                            : 0.0;
                    break;
                case ExprOp::GE:
                    registers[result_reg] =
                        registers[instr.register_operands.operand1] >=
                                registers[instr.register_operands.operand2]
                            ? 1.0
                            : 0.0;
                    break;
                case ExprOp::AND:
                    registers[result_reg] =
                        static_cast<int>(
                            registers[instr.register_operands.operand1]
                        ) &&
                                static_cast<int>(
                                    registers[instr.register_operands.operand2]
                                )
                            ? 1.0
                            : 0.0;
                    break;
                case ExprOp::OR:
                    registers[result_reg] =
                        static_cast<int>(
                            registers[instr.register_operands.operand1]
                        ) ||
                                static_cast<int>(
                                    registers[instr.register_operands.operand2]
                                )
                            ? 1.0
                            : 0.0;
                    break;

                case ExprOp::NOT:
                    registers[result_reg] =
                        !static_cast<int>(
                            registers[instr.register_operands.operand1]
                        )
                            ? 1.0
                            : 0.0;
                    break;
                case ExprOp::LOG:
                    registers[result_reg] =
                        std::log(registers[instr.register_operands.operand1]);
                    break;
                case ExprOp::LOG10:
                    registers[result_reg] =
                        std::log10(registers[instr.register_operands.operand1]);
                    break;
                case ExprOp::ABS:
                    registers[result_reg] =
                        std::abs(registers[instr.register_operands.operand1]);
                    break;
                case ExprOp::SIN:
                    registers[result_reg] =
                        std::sin(registers[instr.register_operands.operand1]);
                    break;
                case ExprOp::COS:
                    registers[result_reg] =
                        std::cos(registers[instr.register_operands.operand1]);
                    break;
                case ExprOp::TAN:
                    registers[result_reg] =
                        std::tan(registers[instr.register_operands.operand1]);
                    break;
                case ExprOp::ASIN:
                    registers[result_reg] =
                        std::asin(registers[instr.register_operands.operand1]);
                    break;
                case ExprOp::ACOS:
                    registers[result_reg] =
                        std::acos(registers[instr.register_operands.operand1]);
                    break;
                case ExprOp::ATAN:
                    registers[result_reg] =
                        std::atan(registers[instr.register_operands.operand1]);
                    break;
                case ExprOp::ATAN2:
                    registers[result_reg] = std::atan2(
                        registers[instr.register_operands.operand1],
                        registers[instr.register_operands.operand2]
                    );
                    break;
                case ExprOp::EXP:
                    registers[result_reg] =
                        std::exp(registers[instr.register_operands.operand1]);
                    break;
                case ExprOp::SQRT:
                    registers[result_reg] =
                        std::sqrt(registers[instr.register_operands.operand1]);
                    break;
                case ExprOp::MIN:
                    registers[result_reg] = std::min(
                        registers[instr.register_operands.operand1],
                        registers[instr.register_operands.operand2]
                    );
                    break;
                case ExprOp::MAX:
                    registers[result_reg] = std::max(
                        registers[instr.register_operands.operand1],
                        registers[instr.register_operands.operand2]
                    );
                    break;
                case ExprOp::SINH:
                    registers[result_reg] =
                        std::sinh(registers[instr.register_operands.operand1]);
                    break;
                case ExprOp::COSH:
                    registers[result_reg] =
                        std::cosh(registers[instr.register_operands.operand1]);
                    break;
                case ExprOp::TANH:
                    registers[result_reg] =
                        std::tanh(registers[instr.register_operands.operand1]);
                    break;
                case ExprOp::ASINH:
                    registers[result_reg] =
                        std::asinh(registers[instr.register_operands.operand1]);
                    break;
                case ExprOp::ACOSH:
                    registers[result_reg] =
                        std::acosh(registers[instr.register_operands.operand1]);
                    break;
                case ExprOp::ATANH:
                    registers[result_reg] =
                        std::atanh(registers[instr.register_operands.operand1]);
                    break;
                case ExprOp::SGN:
                    registers[result_reg] =
                        registers[instr.register_operands.operand1] > 0.0
                            ? 1.0
                            : (registers[instr.register_operands.operand1] < 0.0
                                   ? -1.0
                                   : 0.0);
                    break;
                case ExprOp::BITWISE_AND:
                    registers[result_reg] =
                        static_cast<int>(
                            registers[instr.register_operands.operand1]
                        ) &
                        static_cast<int>(
                            registers[instr.register_operands.operand2]
                        );
                    break;
                case ExprOp::BITWISE_OR:
                    registers[result_reg] =
                        static_cast<int>(
                            registers[instr.register_operands.operand1]
                        ) |
                        static_cast<int>(
                            registers[instr.register_operands.operand2]
                        );
                    break;
                case ExprOp::BITWISE_XOR:
                    registers[result_reg] =
                        static_cast<int>(
                            registers[instr.register_operands.operand1]
                        ) ^
                        static_cast<int>(
                            registers[instr.register_operands.operand2]
                        );
                    break;
                case ExprOp::BITWISE_NOT:
                    registers[result_reg] = ~static_cast<int>(
                        registers[instr.register_operands.operand1]
                    );
                    break;
                case ExprOp::BITWISE_LEFT_SHIFT:
                    registers[result_reg] =
                        static_cast<int>(
                            registers[instr.register_operands.operand1]
                        )
                        << static_cast<int>(
                               registers[instr.register_operands.operand2]
                           );
                    break;
                case ExprOp::BITWISE_RIGHT_SHIFT:
                    registers[result_reg] =
                        static_cast<int>(
                            registers[instr.register_operands.operand1]
                        ) >>
                        static_cast<int>(
                            registers[instr.register_operands.operand2]
                        );
                    break;
                case ExprOp::LOGICAL_AND:
                    registers[result_reg] =
                        static_cast<int>(
                            registers[instr.register_operands.operand1]
                        ) &&
                        static_cast<int>(
                            registers[instr.register_operands.operand2]
                        );
                    break;
                case ExprOp::LOGICAL_OR:
                    registers[result_reg] =
                        static_cast<int>(
                            registers[instr.register_operands.operand1]
                        ) ||
                        static_cast<int>(
                            registers[instr.register_operands.operand2]
                        );
                    break;
                case ExprOp::LOGICAL_NOT:
                    registers[result_reg] = !static_cast<int>(
                        registers[instr.register_operands.operand1]
                    );
                    break;
                case ExprOp::LOGICAL_XOR:
                    registers[result_reg] =
                        static_cast<int>(
                            registers[instr.register_operands.operand1]
                        ) ^
                        static_cast<int>(
                            registers[instr.register_operands.operand2]
                        );
                    break;
                case ExprOp::LOGICAL_NAND:
                    registers[result_reg] =
                        !(static_cast<int>(
                              registers[instr.register_operands.operand1]
                          ) &&
                          static_cast<int>(
                              registers[instr.register_operands.operand2]
                          ));
                    break;
                case ExprOp::LOGICAL_NOR:
                    registers[result_reg] =
                        !(static_cast<int>(
                              registers[instr.register_operands.operand1]
                          ) ||
                          static_cast<int>(
                              registers[instr.register_operands.operand2]
                          ));
                    break;
                case ExprOp::LOGICAL_XNOR:
                    registers[result_reg] =
                        !(static_cast<int>(
                              registers[instr.register_operands.operand1]
                          ) ^
                          static_cast<int>(
                              registers[instr.register_operands.operand2]
                          ));
                    break;

                case ExprOp::IF_THEN_ELSE:
                    registers[result_reg] =
                        registers[instr.register_operands.operand1] != 0.0
                            ? registers[instr.register_operands.operand2]
                            : registers[instr.register_operands.operand3];
                    break;
            }
        }

        // Copy results to output array
        for (size_type i = 0; i < output_count; i++) {
            int reg_idx = mapped_output_indices[i];
            if (reg_idx >= 0 && reg_idx < MAX_REGISTERS) {
                outputs[i] = registers[reg_idx];
            }
            else {
                outputs[i] = 0.0;   // Fallback for invalid register
            }
        }
    }

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

            case ExprOp::SINH: {
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
                return std::sinh(value);
            }
            case ExprOp::COSH: {
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
                return std::cosh(value);
            }
            case ExprOp::TANH: {
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
                return std::tanh(value);
            }
            case ExprOp::ASINH: {
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
                return std::asinh(value);
            }
            case ExprOp::ACOSH: {
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
                if (value < 1.0) {
                    HANDLE_ERROR("Arc hyperbolic cosine argument out of range");
                }
                return std::acosh(value);
            }
            case ExprOp::ATANH: {
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
                if (value <= -1.0 || value >= 1.0) {
                    HANDLE_ERROR(
                        "Arc hyperbolic tangent argument out of range"
                    );
                }
                return std::atanh(value);
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

            case ExprOp::ATAN2: {
                real y = evaluate_expr(
                    nodes,
                    node.children.left,
                    x1,
                    x2,
                    x3,
                    t,
                    dt,
                    parameters
                );
                real x = evaluate_expr(
                    nodes,
                    node.children.right,
                    x1,
                    x2,
                    x3,
                    t,
                    dt,
                    parameters
                );
                return std::atan2(y, x);
            }

            case ExprOp::SGN: {
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
                return helpers::sgn(value);
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

            // bitwise operations
            case ExprOp::BITWISE_AND: {
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
                return static_cast<int>(left) & static_cast<int>(right);
            }

            case ExprOp::BITWISE_OR: {
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
                return static_cast<int>(left) | static_cast<int>(right);
            }

            case ExprOp::BITWISE_XOR: {
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
                return static_cast<int>(left) ^ static_cast<int>(right);
            }

            case ExprOp::BITWISE_NOT: {
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
                return ~static_cast<int>(value);
            }

            case ExprOp::BITWISE_LEFT_SHIFT: {
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
                return static_cast<int>(left) << static_cast<int>(right);
            }

            case ExprOp::BITWISE_RIGHT_SHIFT: {
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
                return static_cast<int>(left) >> static_cast<int>(right);
            }

            // logical operations
            case ExprOp::LOGICAL_AND: {
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

            case ExprOp::LOGICAL_OR: {
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

            case ExprOp::LOGICAL_NOT: {
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

            case ExprOp::LOGICAL_XOR: {
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
                return (left != 0.0) ^ (right != 0.0) ? 1.0 : 0.0;
            }

            case ExprOp::LOGICAL_NAND: {
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
                return !(left != 0.0 && right != 0.0) ? 1.0 : 0.0;
            }

            case ExprOp::LOGICAL_NOR: {
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
                return !(left != 0.0 || right != 0.0) ? 1.0 : 0.0;
            }

            case ExprOp::LOGICAL_XNOR: {
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
                return !(left != 0.0) == !(right != 0.0) ? 1.0 : 0.0;
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
        // define constants for memory limits
        // max number of nodes in expression tree
        constexpr int MAX_NODES = 1024;
        // max number of nodes in evaluation stack
        constexpr int MAX_STACK = 64;

        // node evaluation status tracking
        bool node_evaluated[MAX_NODES] = {false};
        real node_values[MAX_NODES]    = {0.0};

        // track nodes currently being evaluated to detect dependencies
        // (this avoids circular dependencies)
        bool in_progress[MAX_NODES] = {false};

        // stack entry structure with a clearer design
        struct StackEntry {
            int node_idx;   // Index of the node being evaluated
            /// 0. inital, 1. evaluating children, 2. computing result
            int phase;                // current phase of evaluation
            int children_needed;      // number of children required
            int children_evaluated;   // number of children already evaluated
            // values of evaluated children (max 3 for ternary ops)
            real child_values[3];
        };

        // initialize stack with root node
        StackEntry stack[MAX_STACK];
        int stack_size = 0;

        // check node index validity
        if (node_idx < 0 || node_idx >= MAX_NODES) {
            HANDLE_ERROR("Invalid node index");
        }

        // push the root node to start evaluation
        stack[stack_size++]   = {node_idx, 0, 0, 0, {0.0, 0.0, 0.0}};
        in_progress[node_idx] = true;

        // main evaluation loop
        while (stack_size > 0) {
            // guard against stack overflow
            if (stack_size >= MAX_STACK) {
                HANDLE_ERROR("Expression evaluation stack overflow");
            }

            // get current stack entry (without popping yet)
            StackEntry& entry = stack[stack_size - 1];

            // get the corresponding node
            const ExprNode& node = nodes[entry.node_idx];

            // check if this node was already evaluated previously
            if (node_evaluated[entry.node_idx]) {
                // skip processing, just use cached value
                real result = node_values[entry.node_idx];
                // no longer in progress, so we can pop it
                in_progress[entry.node_idx] = false;
                stack_size--;

                // pass result to parent if stack isn't empty
                if (stack_size > 0) {
                    StackEntry& parent = stack[stack_size - 1];
                    parent.child_values[parent.children_evaluated++] = result;
                    // parent is now in child evaluation phase
                    parent.phase = 1;
                }
                continue;
            }

            // process node based on its phase
            bool process_next = true;

            while (process_next) {
                process_next = false;

                switch (entry.phase) {
                    case 0: {   // initial processing phase
                        // simple cases - constants and variables
                        if (node.op == ExprOp::CONSTANT ||
                            node.op == ExprOp::VARIABLE_X1 ||
                            node.op == ExprOp::VARIABLE_X2 ||
                            node.op == ExprOp::VARIABLE_X3 ||
                            node.op == ExprOp::VARIABLE_T ||
                            node.op == ExprOp::VARIABLE_DT ||
                            node.op == ExprOp::PARAMETER) {

                            // determine value based on operation type
                            real value = 0.0;
                            switch (node.op) {
                                case ExprOp::CONSTANT:
                                    value = node.value;
                                    break;
                                case ExprOp::VARIABLE_X1: value = x1; break;
                                case ExprOp::VARIABLE_X2: value = x2; break;
                                case ExprOp::VARIABLE_X3: value = x3; break;
                                case ExprOp::VARIABLE_T: value = t; break;
                                case ExprOp::VARIABLE_DT: value = dt; break;
                                case ExprOp::PARAMETER:
                                    if (parameters == nullptr) {
                                        HANDLE_ERROR("Parameter array is null");
                                    }
                                    value = parameters[node.param_idx];
                                    break;
                                default: break;
                            }

                            // cache result and pop from stack
                            node_values[entry.node_idx]    = value;
                            node_evaluated[entry.node_idx] = true;
                            in_progress[entry.node_idx]    = false;
                            stack_size--;

                            // pass value to parent if stack isn't empty
                            if (stack_size > 0) {
                                StackEntry& parent = stack[stack_size - 1];
                                parent
                                    .child_values[parent.children_evaluated++] =
                                    value;
                            }
                        }
                        else {
                            // for operations requiring child evaluation
                            // move to child evaluation phase
                            entry.phase = 1;
                            // continue to next phase immediately
                            process_next = true;
                        }
                    } break;

                    case 1: {   // child evaluation phase
                        // determine children needed based on operation type
                        if (entry.children_needed == 0) {
                            // first time in this phase - figure out how many
                            // children we need
                            switch (node.op) {
                                // unary operations
                                case ExprOp::NEG:
                                case ExprOp::SQRT:
                                case ExprOp::SIN:
                                case ExprOp::COS:
                                case ExprOp::TAN:
                                case ExprOp::LOG:
                                case ExprOp::LOG10:
                                case ExprOp::EXP:
                                case ExprOp::ABS:
                                case ExprOp::ASIN:
                                case ExprOp::ACOS:
                                case ExprOp::ATAN:
                                case ExprOp::SGN:
                                case ExprOp::SINH:
                                case ExprOp::COSH:
                                case ExprOp::TANH:
                                case ExprOp::ASINH:
                                case ExprOp::ACOSH:
                                case ExprOp::ATANH:
                                case ExprOp::NOT:
                                case ExprOp::BITWISE_NOT:
                                    entry.children_needed = 1;
                                    break;

                                // binary ops
                                case ExprOp::ADD:
                                case ExprOp::SUBTRACT:
                                case ExprOp::MULTIPLY:
                                case ExprOp::DIVIDE:
                                case ExprOp::POWER:
                                case ExprOp::MIN:
                                case ExprOp::MAX:
                                case ExprOp::ATAN2:
                                case ExprOp::LT:
                                case ExprOp::LE:
                                case ExprOp::GT:
                                case ExprOp::GE:
                                case ExprOp::EQ:
                                case ExprOp::AND:
                                case ExprOp::OR:
                                case ExprOp::BITWISE_AND:
                                case ExprOp::BITWISE_OR:
                                case ExprOp::BITWISE_XOR:
                                case ExprOp::BITWISE_LEFT_SHIFT:
                                case ExprOp::BITWISE_RIGHT_SHIFT:
                                case ExprOp::LOGICAL_AND:
                                case ExprOp::LOGICAL_OR:
                                case ExprOp::LOGICAL_XOR:
                                case ExprOp::LOGICAL_NAND:
                                case ExprOp::LOGICAL_NOR:
                                case ExprOp::LOGICAL_XNOR:
                                    entry.children_needed = 2;
                                    break;

                                // ternary operation
                                case ExprOp::IF_THEN_ELSE:
                                    entry.children_needed = 3;
                                    break;

                                default:
                                    HANDLE_ERROR("Unknown operation type");
                                    break;
                            }
                        }

                        // evaluate required children
                        bool need_more_children = false;

                        if (entry.children_evaluated < entry.children_needed) {
                            // we still need to evaluate more children
                            int child_idx = -1;

                            // determine which child to evaluate next
                            if (entry.children_needed == 1) {
                                // unary op - we need the left child
                                child_idx = node.children.left;
                            }
                            else if (entry.children_needed == 2) {
                                // binary op - need both left and right
                                if (entry.children_evaluated == 0) {
                                    // Process left child first
                                    child_idx = node.children.left;
                                }
                                else {
                                    // Then process right child
                                    child_idx = node.children.right;
                                }
                            }
                            else if (entry.children_needed == 3) {
                                // ternary op (if-then-else)
                                if (entry.children_evaluated == 0) {
                                    // First evaluate condition
                                    child_idx = node.ternary.condition;
                                }
                                else if (entry.children_evaluated == 1) {
                                    // then evaluate either then or else branch
                                    // based on condition
                                    if (entry.child_values[0] != 0.0) {
                                        child_idx = node.ternary.then_expr;
                                    }
                                    else {
                                        child_idx = node.ternary.else_expr;
                                    }
                                }
                            }

                            // validate child index
                            if (child_idx < 0 || child_idx >= MAX_NODES) {
                                HANDLE_ERROR("Invalid child node index");
                            }

                            // check if child already evaluated
                            if (node_evaluated[child_idx]) {
                                // use cached value
                                entry.child_values[entry.children_evaluated++] =
                                    node_values[child_idx];
                                // continue in this phase
                                process_next = true;
                            }
                            // check if child is currently being evaluated
                            // (would cause circular dependency)
                            else if (in_progress[child_idx]) {
                                HANDLE_ERROR(
                                    "Circular dependency in expression"
                                );
                            }
                            else {
                                // push child onto stack
                                in_progress[child_idx] = true;
                                stack[stack_size++] =
                                    {child_idx, 0, 0, 0, {0.0, 0.0, 0.0}};
                                need_more_children = true;
                            }
                        }

                        // if all children are evaluated, move to result phase
                        if (!need_more_children &&
                            entry.children_evaluated >= entry.children_needed) {
                            // move to result computation phase
                            entry.phase  = 2;
                            process_next = true;
                        }
                    } break;

                    case 2: {   // result computation phase
                        // calculate result based on operation type
                        real result = 0.0;

                        switch (node.op) {
                            // process each operation type...
                            case ExprOp::ADD:
                                result = entry.child_values[0] +
                                         entry.child_values[1];
                                break;
                            case ExprOp::SUBTRACT:
                                result = entry.child_values[0] -
                                         entry.child_values[1];
                                break;
                            case ExprOp::MULTIPLY:
                                result = entry.child_values[0] *
                                         entry.child_values[1];
                                break;
                            case ExprOp::DIVIDE:
                                if (entry.child_values[1] == 0.0) {
                                    HANDLE_ERROR("Division by zero");
                                }
                                result = entry.child_values[0] /
                                         entry.child_values[1];
                                break;
                            case ExprOp::POWER:
                                if (entry.child_values[0] < 0.0 &&
                                    entry.child_values[1] !=
                                        static_cast<int>(entry.child_values[1]
                                        )) {
                                    HANDLE_ERROR(
                                        "Negative base with non-integer "
                                        "exponent"
                                    );
                                }
                                result = std::pow(
                                    entry.child_values[0],
                                    entry.child_values[1]
                                );
                                break;
                            case ExprOp::NEG:
                                result = -entry.child_values[0];
                                break;
                            case ExprOp::MIN:
                                result = std::min(
                                    entry.child_values[0],
                                    entry.child_values[1]
                                );
                                break;
                            case ExprOp::MAX:
                                result = std::max(
                                    entry.child_values[0],
                                    entry.child_values[1]
                                );
                                break;
                            case ExprOp::LT:
                                result = entry.child_values[0] <
                                                 entry.child_values[1]
                                             ? 1.0
                                             : 0.0;
                                break;
                            case ExprOp::LE:
                                result = entry.child_values[0] <=
                                                 entry.child_values[1]
                                             ? 1.0
                                             : 0.0;
                                break;
                            case ExprOp::GT:
                                result = entry.child_values[0] >
                                                 entry.child_values[1]
                                             ? 1.0
                                             : 0.0;
                                break;
                            case ExprOp::GE:
                                result = entry.child_values[0] >=
                                                 entry.child_values[1]
                                             ? 1.0
                                             : 0.0;
                                break;
                            case ExprOp::EQ:
                                result = entry.child_values[0] ==
                                                 entry.child_values[1]
                                             ? 1.0
                                             : 0.0;
                                break;
                            case ExprOp::AND:
                                result = entry.child_values[0] != 0.0 &&
                                                 entry.child_values[1] != 0.0
                                             ? 1.0
                                             : 0.0;
                                break;
                            case ExprOp::OR:
                                result = entry.child_values[0] != 0.0 ||
                                                 entry.child_values[1] != 0.0
                                             ? 1.0
                                             : 0.0;
                                break;
                            case ExprOp::NOT:
                                result =
                                    entry.child_values[0] == 0.0 ? 1.0 : 0.0;
                                break;
                            case ExprOp::BITWISE_AND:
                                result =
                                    static_cast<int>(entry.child_values[0]) &
                                    static_cast<int>(entry.child_values[1]);
                                break;
                            case ExprOp::BITWISE_OR:
                                result =
                                    static_cast<int>(entry.child_values[0]) |
                                    static_cast<int>(entry.child_values[1]);
                                break;
                            case ExprOp::BITWISE_XOR:
                                result =
                                    static_cast<int>(entry.child_values[0]) ^
                                    static_cast<int>(entry.child_values[1]);
                                break;
                            case ExprOp::BITWISE_NOT:
                                result =
                                    ~static_cast<int>(entry.child_values[0]);
                                break;
                            case ExprOp::BITWISE_LEFT_SHIFT:
                                result =
                                    static_cast<int>(entry.child_values[0])
                                    << static_cast<int>(entry.child_values[1]);
                                break;
                            case ExprOp::BITWISE_RIGHT_SHIFT:
                                result =
                                    static_cast<int>(entry.child_values[0]) >>
                                    static_cast<int>(entry.child_values[1]);
                                break;
                            case ExprOp::IF_THEN_ELSE:
                                if (entry.child_values[0] != 0.0) {
                                    result = entry.child_values[1];
                                }
                                else {
                                    result = entry.child_values[2];
                                }
                                break;
                            case ExprOp::SIN:
                                result = std::sin(entry.child_values[0]);
                                break;
                            case ExprOp::COS:
                                result = std::cos(entry.child_values[0]);
                                break;
                            case ExprOp::TAN:
                                result = std::tan(entry.child_values[0]);
                                break;
                            case ExprOp::SINH:
                                result = std::sinh(entry.child_values[0]);
                                break;
                            case ExprOp::COSH:
                                result = std::cosh(entry.child_values[0]);
                                break;
                            case ExprOp::TANH:
                                result = std::tanh(entry.child_values[0]);
                                break;
                            case ExprOp::ASIN:
                                if (entry.child_values[0] < -1.0 ||
                                    entry.child_values[0] > 1.0) {
                                    HANDLE_ERROR(
                                        "Arc sine argument out of range"
                                    );
                                }
                                result = std::asin(entry.child_values[0]);
                                break;
                            case ExprOp::ACOS:
                                if (entry.child_values[0] < -1.0 ||
                                    entry.child_values[0] > 1.0) {
                                    HANDLE_ERROR(
                                        "Arc cosine argument out of range"
                                    );
                                }
                                result = std::acos(entry.child_values[0]);
                                break;
                            case ExprOp::ATAN:
                                result = std::atan(entry.child_values[0]);
                                break;
                            case ExprOp::ATAN2:
                                result = std::atan2(
                                    entry.child_values[0],
                                    entry.child_values[1]
                                );
                                break;
                            case ExprOp::ASINH:
                                result = std::asinh(entry.child_values[0]);
                                break;
                            case ExprOp::ACOSH:
                                if (entry.child_values[0] < 1.0) {
                                    HANDLE_ERROR(
                                        "Arc hyperbolic cosine argument "
                                        "out of range"
                                    );
                                }
                                result = std::acosh(entry.child_values[0]);
                                break;
                            case ExprOp::ATANH:
                                if (entry.child_values[0] <= -1.0 ||
                                    entry.child_values[0] >= 1.0) {
                                    HANDLE_ERROR(
                                        "Arc hyperbolic tangent argument "
                                        "out of range"
                                    );
                                }
                                result = std::atanh(entry.child_values[0]);
                                break;
                            case ExprOp::ABS:
                                result = std::abs(entry.child_values[0]);
                                break;
                            case ExprOp::LOG:
                                if (entry.child_values[0] <= 0.0) {
                                    HANDLE_ERROR(
                                        "Logarithm of non-positive "
                                        "number"
                                    );
                                }
                                result = std::log(entry.child_values[0]);
                                break;
                            case ExprOp::LOG10:
                                if (entry.child_values[0] <= 0.0) {
                                    HANDLE_ERROR(
                                        "Logarithm of non-positive "
                                        "number"
                                    );
                                }
                                result = std::log10(entry.child_values[0]);
                                break;
                            case ExprOp::EXP:
                                result = std::exp(entry.child_values[0]);
                                break;
                            case ExprOp::SQRT:
                                if (entry.child_values[0] < 0.0) {
                                    HANDLE_ERROR(
                                        "Square root of negative "
                                        "number"
                                    );
                                }
                                result = std::sqrt(entry.child_values[0]);
                                break;
                            case ExprOp::SGN:
                                result = helpers::sgn(entry.child_values[0]);
                                break;
                            default:
                                HANDLE_ERROR("Unknown operation type");
                                break;
                        }

                        // cache the result
                        node_values[entry.node_idx]    = result;
                        node_evaluated[entry.node_idx] = true;
                        in_progress[entry.node_idx]    = false;

                        // pop from stack
                        stack_size--;

                        // pass result to parent if stack isn't empty
                        if (stack_size > 0) {
                            StackEntry& parent = stack[stack_size - 1];
                            parent.child_values[parent.children_evaluated++] =
                                result;
                        }
                    } break;
                }
            }
        }

        // return cached result for root node
        return node_evaluated[node_idx] ? node_values[node_idx] : 0.0;
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
            if constexpr (global::on_gpu) {
                // gpu version uses non-recursive evaluation
                // this avoids stack overflow issues
                results[i] = evaluate_expr_nonrecursive(
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
            else {
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
    }

}   // namespace simbi::expression
