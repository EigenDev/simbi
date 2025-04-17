#ifndef EXP_LOAD_HPP
#define EXP_LOAD_HPP

#include "core/types/containers/ndarray.hpp"
#include "core/types/utility/config_dict.hpp"
#include "core/types/utility/expression.hpp"
#include <string>

namespace simbi::expression {
    ndarray<ExprNode> load_expressions(const ConfigDict& expr_data);

    std::tuple<ndarray<ExprNode>, ndarray<int>, ndarray<real>>
    load_expression_data(const ConfigDict& json_data);

    // Convert a string operation to ExprOp enum
    ExprOp string_to_expr_op(const std::string& op);

    ndarray<int> get_output_indices(const ConfigDict& expr_data);
    ndarray<real> get_parameters(const ConfigDict& expr_data);
}   // namespace simbi::expression

#endif   // EXP_LOAD_HPP
