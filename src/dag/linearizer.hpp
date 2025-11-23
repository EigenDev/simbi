#ifndef LINEARIZER_HPP
#define LINEARIZER_HPP

#include "containers/store.hpp"
#include "expression.hpp"

#include <cstdint>
#include <utility>
#include <vector>

namespace simbi::expression {
    void topological_sort(
        const store_t<ExprNode>& nodes,
        const store_t<std::int64_t>& output_indices,
        std::vector<std::int64_t>& result
    );

    std::pair<store_t<LinearExprInstr>, store_t<std::int64_t>>
    linearize_expression_tree(
        const store_t<ExprNode>& nodes,
        const store_t<std::int64_t>& output_indices
    );

    std::int64_t op2reg(ExprOp op);
}   // namespace simbi::expression

#endif   // LINEARIZER_HPP
