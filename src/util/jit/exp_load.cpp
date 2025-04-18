#include "exp_load.hpp"
#include "core/types/containers/ndarray.hpp"
#include "core/types/utility/config_dict.hpp"
#include "core/types/utility/expression.hpp"
#include <iostream>
#include <unordered_map>

namespace simbi::expression {
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
          {"IF_THEN_ELSE", ExprOp::IF_THEN_ELSE}
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

    ndarray<ExprNode> load_expressions(const ConfigDict& expr_data)
    {
        ndarray<ExprNode> nodes;

        // get the expressions array
        if (!expr_data.contains("expressions") ||
            !expr_data.at("expressions").is_list()) {
            return nodes;
        }

        const auto& expressions_list =
            expr_data.at("expressions").get<std::list<ConfigDict>>();
        // We need to determine the actual type in the list
        // Since ConfigDict doesn't store lists of dictionaries directly,
        // we need to access each dict in the list individually

        // Reserve space assuming a reasonable size
        nodes.reserve(50);

        // For each expression in the list
        for (auto& expr_node : expressions_list) {
            ExprNode node;

            // Get the operation type
            if (expr_node.contains("op") && expr_node.at("op").is_string()) {
                std::string op_str = expr_node.at("op").get<std::string>();
                node.op            = string_to_expr_op(op_str);

                // Handle different node types
                if (op_str == "CONSTANT") {
                    node.value = expr_node.at("value").get<real>();
                }
                else if (op_str.find("VARIABLE_") == 0) {
                    // Variables don't need additional data
                }
                else if (op_str == "PARAMETER") {
                    node.param_idx = expr_node.at("param_idx").get<int>();
                }
                else if (op_str == "IF_THEN_ELSE") {
                    node.ternary.condition =
                        expr_node.at("condition").get<int>();
                    node.ternary.then_expr = expr_node.at("then").get<int>();
                    node.ternary.else_expr = expr_node.at("else").get<int>();
                }
                else {
                    // Binary/unary operations
                    if (expr_node.contains("left")) {
                        node.children.left = expr_node.at("left").get<int>();

                        if (expr_node.contains("right")) {
                            node.children.right =
                                expr_node.at("right").get<int>();
                        }
                    }
                }
            }

            nodes.push_back(node);
        }

        return nodes;
    }

    ndarray<int> get_output_indices(const ConfigDict& expr_data)
    {
        if (!(expr_data.contains("output_indices") &&
              expr_data.at("output_indices").is_array_of_ints())) {
            return ndarray<int>{};
        }

        ndarray res(expr_data.at("output_indices").get<std::vector<int>>());
        res.sync_to_device();
        return res;
    }

    ndarray<real> get_parameters(const ConfigDict& expr_data)
    {
        if (!(expr_data.contains("parameters") &&
              expr_data.at("parameters").is_array())) {
            return ndarray<real>{};
        }

        ndarray res(expr_data.at("parameters").get<std::vector<real>>());
        res.sync_to_device();
        return res;
    }

    ndarray<real> get_parameter_extent(const ConfigDict& expr_data)
    {
        if (!(expr_data.contains("parameter_extent") &&
              expr_data.at("parameter_extent").is_array())) {
            return ndarray<real>{};
        }

        ndarray res(expr_data.at("parameter_extent").get<std::vector<real>>());
        res.sync_to_device();
        return res;
    }

    std::tuple<ndarray<ExprNode>, ndarray<int>, ndarray<real>>
    load_expression_data(const ConfigDict& data)
    {
        return {
          load_expressions(data),
          get_output_indices(data),
          get_parameter_extent(data)
        };
    }
}   // namespace simbi::expression
