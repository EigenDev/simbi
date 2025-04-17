#ifndef EXP_LOAD_HPP
#define EXP_LOAD_HPP

#include "core/types/utility/config_dict.hpp"
#include "core/types/utility/expression.hpp"
#include <map>
#include <string>
#include <variant>
#include <vector>

using json_t =
    std::vector<std::map<std::string, std::variant<std::string, real, int>>>;
namespace simbi::expression {
    std::vector<ExprNode> load_expressions(const ConfigDict& expr_data);

    std::tuple<std::vector<ExprNode>, std::vector<int>, std::vector<real>>
    load_expression_data(const ConfigDict& json_data);

    // Convert a string operation to ExprOp enum
    ExprOp string_to_expr_op(const std::string& op);

    std::vector<int> get_output_indices(const ConfigDict& expr_data);
    std::vector<real> get_parameters(const ConfigDict& expr_data);
}   // namespace simbi::expression

#endif   // EXP_LOAD_HPP
