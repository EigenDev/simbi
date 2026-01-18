// =============================================================================
// linearizer.hpp
//
// converts an expression dag into a linear instruction sequence.
// this file provides the functions necessary to transform the graph-based
// representation of an expression (`exprnode`) into a flat, array-based
// sequence of instructions (`linearexprinstr`) suitable for a simple
// virtual machine. this process involves a topological sort of the dag.
//
// usage:
//   auto [instrs, mapped_outs] = linearize_expression_tree(nodes, outputs);
// =============================================================================
#pragma once

#include "containers/store.hpp"
#include "expression.hpp"

#include <cstdint>
#include <utility>
#include <vector>

namespace simbi::expression {
    void topological_sort(
        const store_t<ExprNode>&     nodes,
        const store_t<std::int64_t>& output_indices,
        std::vector<std::int64_t>&   result
    );

    std::pair<store_t<LinearExprInstr>, store_t<std::int64_t>> linearize_expression_tree(
        const store_t<ExprNode>&     nodes,
        const store_t<std::int64_t>& output_indices
    );

    std::int64_t op2reg(ExprOp op);
} // namespace simbi::expression
