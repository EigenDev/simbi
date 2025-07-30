#include "exp_load.hpp"
#include "config.hpp"
#include "containers/ndarray.hpp"
#include "math/expression.hpp"
#include "utility/config_dict.hpp"
#include <iostream>
#include <list>
#include <string>
#include <unordered_map>
#include <vector>

namespace simbi::expression {
    using namespace nd;
    // convert a string operation to ExprOp enum

    ExprOp string_to_expr_op(const std::string& op_str)
    {
        // create a map of string to ExprOp
        static const std::unordered_map<std::string, ExprOp> op_map = {
          {"CONSTANT", ExprOp::CONSTANT},
          {"VARIABLE_X1", ExprOp::VARIABLE_X1},
          {"VARIABLE_X2", ExprOp::VARIABLE_X2},
          {"VARIABLE_X3", ExprOp::VARIABLE_X3},
          {"VARIABLE_T", ExprOp::VARIABLE_T},
          {"VARIABLE_DT", ExprOp::VARIABLE_DT},
          {"PARAMETER", ExprOp::PARAMETER},
          {"ADD", ExprOp::ADD},
          {"SUBTRACT", ExprOp::SUBTRACT},
          {"MULTIPLY", ExprOp::MULTIPLY},
          {"DIVIDE", ExprOp::DIVIDE},
          {"POW", ExprOp::POWER},
          {"NEG", ExprOp::NEG},
          {"LT", ExprOp::LT},
          {"GT", ExprOp::GT},
          {"EQ", ExprOp::EQ},
          {"LE", ExprOp::LE},
          {"GE", ExprOp::GE},
          {"AND", ExprOp::AND},
          {"OR", ExprOp::OR},
          {"NOT", ExprOp::NOT},
          {"LOG", ExprOp::LOG},
          {"LOG10", ExprOp::LOG10},
          {"ABS", ExprOp::ABS},
          {"SIN", ExprOp::SIN},
          {"COS", ExprOp::COS},
          {"TAN", ExprOp::TAN},
          {"ASIN", ExprOp::ASIN},
          {"ACOS", ExprOp::ACOS},
          {"ATAN", ExprOp::ATAN},
          {"EXP", ExprOp::EXP},
          {"SQRT", ExprOp::SQRT},
          {"MIN", ExprOp::MIN},
          {"MAX", ExprOp::MAX},
          {"ATAN2", ExprOp::ATAN2},
          {"IF_THEN_ELSE", ExprOp::IF_THEN_ELSE},
          {"FLOOR", ExprOp::FLOOR},
          {"CEIL", ExprOp::CEIL},
          {"MOD", ExprOp::MOD},
          {"SGN", ExprOp::SGN},
          {"BITWISE_AND", ExprOp::BITWISE_AND},
          {"BITWISE_OR", ExprOp::BITWISE_OR},
          {"BITWISE_XOR", ExprOp::BITWISE_XOR},
          {"BITWISE_NOT", ExprOp::BITWISE_NOT},
          {"BITWISE_LEFT_SHIFT", ExprOp::BITWISE_LEFT_SHIFT},
          {"BITWISE_RIGHT_SHIFT", ExprOp::BITWISE_RIGHT_SHIFT},
          {"LOGICAL_AND", ExprOp::LOGICAL_AND},
          {"LOGICAL_OR", ExprOp::LOGICAL_OR},
          {"LOGICAL_XOR", ExprOp::LOGICAL_XOR},
          {"LOGICAL_NAND", ExprOp::LOGICAL_NAND},
          {"LOGICAL_NOR", ExprOp::LOGICAL_NOR},
          {"LOGICAL_XNOR", ExprOp::LOGICAL_XNOR}
        };

        auto it = op_map.find(op_str);
        if (it != op_map.end()) {
            return it->second;
        }

        // default to constant for unknown operations
        std::cerr << "Unknown operation: " << op_str
                  << ". Defaulting to CONSTANT." << std::endl;
        return ExprOp::CONSTANT;
    }

    ndarray_t<ExprNode> load_expressions(const config_dict_t& expr_data)
    {
        ndarray_t<ExprNode> nodes;
        std::vector<ExprNode> nodes_vec;

        // get the expressions array
        if (!expr_data.contains("expressions") ||
            !expr_data.at("expressions").is_list()) {
            return nodes;
        }

        const auto& expressions_list =
            expr_data.at("expressions")
                .template get<std::list<config_dict_t>>();
        // We need to determine the actual type in the list
        // Since config_dict_t doesn't store lists of dictionaries directly,
        // we need to access each dict in the list individually

        // Reserve space assuming a reasonable size
        nodes.reserve(50);

        // For each expression in the list
        for (auto& expr_node : expressions_list) {
            ExprNode node;

            // Get the operation type
            if (expr_node.contains("op") && expr_node.at("op").is_string()) {
                std::string op_str =
                    expr_node.at("op").template get<std::string>();
                node.op = string_to_expr_op(op_str);

                // Handle different node types
                if (op_str == "CONSTANT") {
                    node.value = expr_node.at("value").template get<real>();
                }
                else if (op_str.find("VARIABLE_") == 0) {
                    // Variables don't need additional data
                }
                else if (op_str == "PARAMETER") {
                    node.param_idx =
                        expr_node.at("param_idx").template get<std::int64_t>();
                }
                else if (op_str == "IF_THEN_ELSE") {
                    node.ternary.condition =
                        expr_node.at("condition").template get<std::int64_t>();
                    node.ternary.then_expr =
                        expr_node.at("true_case").template get<std::int64_t>();
                    node.ternary.else_expr =
                        expr_node.at("false_case").template get<std::int64_t>();
                }
                else {
                    // Binary/unary operations
                    if (expr_node.contains("left")) {
                        node.children.left =
                            expr_node.at("left").template get<std::int64_t>();

                        if (expr_node.contains("right")) {
                            node.children.right =
                                expr_node.at("right")
                                    .template get<std::int64_t>();
                        }
                    }
                }
            }
            nodes.push_back_with_sync(node);
        }

        return nodes;
    }

    ndarray_t<std::int64_t> get_output_indices(const config_dict_t& expr_data)
    {
        if (!(expr_data.contains("output_indices") &&
              expr_data.at("output_indices").is_array_of_ints())) {
            return ndarray_t<std::int64_t>{};
        }

        ndarray_t res(expr_data.at("output_indices")
                          .template get<std::vector<std::int64_t>>());
        res.to_gpu();
        return res;
    }

    ndarray_t<real> get_parameters(const config_dict_t& expr_data)
    {
        if (!(expr_data.contains("parameters") &&
              expr_data.at("parameters").is_array_of_floats())) {
            return ndarray_t<real>{};
        }

        ndarray_t res(
            expr_data.at("parameters").template get<std::vector<real>>()
        );
        res.to_gpu();
        return res;
    }

    ndarray_t<real> get_parameter_range(const config_dict_t& expr_data)
    {
        if (!expr_data.contains("param_count")) {
            return ndarray_t<real>{};
        }

        auto res = ndarray_t<real>(
            expr_data.at("param_count").template get<std::int64_t>()
        );
        res.to_gpu();
        return res;
    }

    std::tuple<ndarray_t<ExprNode>, ndarray_t<std::int64_t>, ndarray_t<real>>
    load_expression_data(const config_dict_t& data)
    {
        return {
          load_expressions(data),
          get_output_indices(data),
          get_parameter_range(data)
        };
    }
}   // namespace simbi::expression
