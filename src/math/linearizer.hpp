#ifndef LINEARIZER_HPP
#define LINEARIZER_HPP

#include "containers/ndarray.hpp"
#include "expression.hpp"
#include <utility>
#include <vector>

namespace simbi::expression {
    using namespace nd;
    void topological_sort(
        const ndarray_t<ExprNode>& nodes,
        const ndarray_t<std::int64_t>& output_indices,
        std::vector<std::int64_t>& result
    );

    std::pair<ndarray_t<LinearExprInstr>, ndarray_t<std::int64_t>>
    linearize_expression_tree(
        const ndarray_t<ExprNode>& nodes,
        const ndarray_t<std::int64_t>& output_indices
    );

    std::int64_t op2reg(ExprOp op);
}   // namespace simbi::expression

#endif   // LINEARIZER_HPP
