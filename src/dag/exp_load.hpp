#ifndef EXP_LOAD_HPP
#define EXP_LOAD_HPP

#include "build_config.hpp"
#include "containers/store.hpp"
#include "dag/expression.hpp"
#include "utility/config_dict.hpp"

#include <cstdint>
#include <string>
#include <tuple>

namespace simbi::expression {
    store_t<ExprNode> load_expressions(const config_dict_t& expr_data);

    std::tuple<store_t<ExprNode>, store_t<std::int64_t>, store_t<real>>
    load_expression_data(const config_dict_t& json_data);

    // convert a string operation to ExprOp enum
    ExprOp string_to_expr_op(const std::string& op);

    store_t<std::int64_t> get_output_indices(const config_dict_t& expr_data);
    store_t<real> get_parameters(const config_dict_t& expr_data);
}   // namespace simbi::expression

#endif   // EXP_LOAD_HPP
