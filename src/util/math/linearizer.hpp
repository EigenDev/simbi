#ifndef LINEARIZER_HPP
#define LINEARIZER_HPP

#include "core/containers/ndarray.hpp"
#include "expression.hpp"
#include <utility>
#include <vector>

namespace simbi::expression {
    using namespace containers;
    void topological_sort(
        const ndarray_t<ExprNode>& nodes,
        const ndarray_t<int>& output_indices,
        std::vector<int>& result
    );

    std::pair<ndarray_t<LinearExprInstr>, ndarray_t<int>>
    linearize_expression_tree(
        const ndarray_t<ExprNode>& nodes,
        const ndarray_t<int>& output_indices
    );

    int op2reg(ExprOp op);
}   // namespace simbi::expression

#endif   // LINEARIZER_HPP
