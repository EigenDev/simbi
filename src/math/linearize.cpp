#include "containers/ndarray.hpp"
#include "expression.hpp"
#include "linearizer.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <stack>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace simbi::expression {
    using namespace nd;

    std::int64_t op2reg(ExprOp op)
    {
        switch (op) {
            case ExprOp::VARIABLE_X1: return 0;
            case ExprOp::VARIABLE_X2: return 1;
            case ExprOp::VARIABLE_X3: return 2;
            case ExprOp::VARIABLE_T: return 3;
            case ExprOp::VARIABLE_DT: return 4;
            default: return -1;   // Invalid operation
        }
    }

    std::pair<ndarray_t<LinearExprInstr>, ndarray_t<std::int64_t>>
    linearize_expression_tree(
        const ndarray_t<ExprNode>& nodes,
        const ndarray_t<std::int64_t>& output_indices
    )
    {
        // First identify all nodes needed for output expressions
        std::vector<std::int64_t> eval_order;
        topological_sort(nodes, output_indices, eval_order);

        // Map tree node indices to register numbers
        std::unordered_map<std::int64_t, std::int64_t> node_to_reg;

        // Fixed registers for inputs
        node_to_reg[-1] = 0;   // x1
        node_to_reg[-2] = 1;   // x2
        node_to_reg[-3] = 2;   // x3
        node_to_reg[-4] = 3;   // t
        node_to_reg[-5] = 4;   // dt

        // start regular registers after input registers
        std::int64_t next_reg = 5;
        // pre-allocate registers for all needed nodes
        // in topological order
        for (std::int64_t node_idx : eval_order) {
            if (node_to_reg.find(node_idx) == node_to_reg.end()) {
                node_to_reg[node_idx] = next_reg++;
            }
        }

        // create instructions in same order
        std::vector<LinearExprInstr> instructions;
        for (std::int64_t node_idx : eval_order) {
            const auto& node        = nodes[node_idx];
            std::int64_t result_reg = node_to_reg[node_idx];

            LinearExprInstr instr;
            instr.op              = node.op;
            instr.result_register = result_reg;

            switch (node.op) {
                case ExprOp::CONSTANT: instr.constant_eval = node.value; break;

                case ExprOp::VARIABLE_X1:
                case ExprOp::VARIABLE_X2:
                case ExprOp::VARIABLE_X3:
                case ExprOp::VARIABLE_T:
                case ExprOp::VARIABLE_DT:
                    // Map variable to its input register
                    instr.register_operands.operand1 = op2reg(node.op);
                    break;

                case ExprOp::PARAMETER:
                    instr.op            = ExprOp::PARAMETER;
                    instr.parameter_idx = node.param_idx;
                    break;

                case ExprOp::IF_THEN_ELSE:
                    instr.register_operands.operand1 =
                        node_to_reg[node.ternary.condition];
                    instr.register_operands.operand2 =
                        node_to_reg[node.ternary.then_expr];
                    instr.register_operands.operand3 =
                        node_to_reg[node.ternary.else_expr];
                    break;

                default:
                    // Binary and unary operations
                    if (node.children.left >= 0) {
                        instr.register_operands.operand1 =
                            node_to_reg[node.children.left];
                    }
                    else {
                        instr.register_operands.operand1 = -1;
                    }

                    if (node.children.right >= 0) {
                        instr.register_operands.operand2 =
                            node_to_reg[node.children.right];
                    }
                    else {
                        instr.register_operands.operand2 = -1;
                    }

                    instr.register_operands.operand3 = -1;
                    break;
            }
            instructions.push_back(instr);
        }
        // place leaf nodes at the front of the instruction list
        std::reverse(instructions.begin(), instructions.end());

        // Create mapped output indices to match the register numbers
        ndarray_t<std::int64_t> mapped_output_indices(output_indices.size());
        ndarray_t<LinearExprInstr> result_instrs(instructions.size());
        for (size_t ii = 0; ii < output_indices.size(); ii++) {
            mapped_output_indices[ii] = node_to_reg[output_indices[ii]];
        }
        for (size_t ii = 0; ii < instructions.size(); ii++) {
            result_instrs[ii] = instructions[ii];
        }

        return {std::move(result_instrs), std::move(mapped_output_indices)};
    }

    void topological_sort(
        const ndarray_t<ExprNode>& nodes,
        const ndarray_t<std::int64_t>& output_indices,
        std::vector<std::int64_t>& result
    )
    {
        result.clear();

        // Track visited status: 0 = unvisited, 1 = temporary mark, 2 =
        // permanently marked
        std::unordered_map<std::int64_t, std::int64_t> visited;

        // Set of all nodes needed for evaluation (to be populated)
        std::unordered_set<std::int64_t> needed_nodes;

        // First identify all nodes needed for evaluation
        std::stack<std::int64_t> to_process;
        for (size_t i = 0; i < output_indices.size(); i++) {
            to_process.push(output_indices[i]);
        }

        while (!to_process.empty()) {
            std::int64_t node_id = to_process.top();
            to_process.pop();

            if (needed_nodes.find(node_id) != needed_nodes.end()) {
                continue;   // Already processed
            }

            needed_nodes.insert(node_id);

            const auto& node = nodes[node_id];

            // Add dependencies based on operation type
            switch (node.op) {
                case ExprOp::IF_THEN_ELSE:
                    to_process.push(node.ternary.condition);
                    to_process.push(node.ternary.then_expr);
                    to_process.push(node.ternary.else_expr);
                    break;

                case ExprOp::CONSTANT:
                case ExprOp::VARIABLE_X1:
                case ExprOp::VARIABLE_X2:
                case ExprOp::VARIABLE_X3:
                case ExprOp::VARIABLE_T:
                case ExprOp::VARIABLE_DT:
                case ExprOp::PARAMETER:
                    // No dependencies
                    break;

                default:
                    // Binary and unary operations
                    if (node.children.left >= 0) {
                        to_process.push(node.children.left);
                    }
                    if (node.children.right >= 0) {
                        to_process.push(node.children.right);
                    }
                    break;
            }
        }

        // Now perform the actual topological sort using DFS
        std::function<void(int)> visit = [&](std::int64_t node_id) {
            // Check visited status
            auto it = visited.find(node_id);
            if (it != visited.end()) {
                if (it->second == 1) {
                    // Temporary mark means cycle
                    throw std::runtime_error(
                        "Cycle detected in expression graph"
                    );
                }
                else if (it->second == 2) {
                    // Already permanently marked
                    return;
                }
            }

            // Mark temporarily
            visited[node_id] = 1;

            const auto& node = nodes[node_id];

            // Visit dependencies based on operation type
            switch (node.op) {
                case ExprOp::IF_THEN_ELSE:
                    visit(node.ternary.condition);
                    visit(node.ternary.then_expr);
                    visit(node.ternary.else_expr);
                    break;

                case ExprOp::CONSTANT:
                case ExprOp::VARIABLE_X1:
                case ExprOp::VARIABLE_X2:
                case ExprOp::VARIABLE_X3:
                case ExprOp::VARIABLE_T:
                case ExprOp::VARIABLE_DT:
                case ExprOp::PARAMETER:
                    // No dependencies
                    break;

                default:
                    // Binary and unary operations
                    if (node.children.left >= 0) {
                        visit(node.children.left);
                    }
                    if (node.children.right >= 0) {
                        visit(node.children.right);
                    }
                    break;
            }

            // Mark permanently and add to result
            visited[node_id] = 2;
            result.push_back(node_id);
        };

        // Start the sort from each needed output node
        for (std::int64_t node_id : needed_nodes) {
            if (visited.find(node_id) == visited.end()) {
                visit(node_id);
            }
        }

        // Our DFS gives reverse topological order, so reverse it
        std::reverse(result.begin(), result.end());
    }
}   // namespace simbi::expression
