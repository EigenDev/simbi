#ifndef EXP_LOAD_HPP
#define EXP_LOAD_HPP

#include "config.hpp"
#include "core/containers/ndarray.hpp"
#include "core/utility/config_dict.hpp"
#include "util/math/expression.hpp"
#include <string>
#include <tuple>

namespace simbi::expression {
    using namespace containers;
    ndarray_t<ExprNode> load_expressions(const config_dict_t& expr_data);

    std::tuple<ndarray_t<ExprNode>, ndarray_t<int>, ndarray_t<real>>
    load_expression_data(const config_dict_t& json_data);

    // convert a string operation to ExprOp enum
    ExprOp string_to_expr_op(const std::string& op);

    ndarray_t<int> get_output_indices(const config_dict_t& expr_data);
    ndarray_t<real> get_parameters(const config_dict_t& expr_data);
}   // namespace simbi::expression

#endif   // EXP_LOAD_HPP
