#ifndef LINEARIZER_HPP
#define LINEARIZER_HPP

#include "config.hpp"
#include "core/types/containers/ndarray.hpp"
#include "expression.hpp"

namespace simbi::expression {
    void topological_sort(
        const ndarray<ExprNode>& nodes,
        const ndarray<int>& output_indices,
        std::vector<int>& result
    );

    std::pair<ndarray<LinearExprInstr>, ndarray<int>> linearize_expression_tree(
        const ndarray<ExprNode>& nodes,
        const ndarray<int>& output_indices
    );

    int op2reg(ExprOp op);
}   // namespace simbi::expression

#endif   // LINEARIZER_HPP
