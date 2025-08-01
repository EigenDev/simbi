#ifndef EVALUATOR_HPP
#define EVALUATOR_HPP

#include "config.hpp"
#include "containers/ndarray.hpp"
#include "math/expression.hpp"

#include <cstdint>

namespace simbi::expression {
    using namespace nd;
    // main evaluation function with parameter support
    DEV real evaluate_expr(
        const ExprNode* nodes,
        std::int64_t node_idx,
        real x1,
        real x2                = 0.0,
        real x3                = 0.0,
        real t                 = 0.0,
        real dt                = 0.0,
        const real* parameters = nullptr
    );

    // non-recursive version for deeply nested expressions
    DEV real evaluate_expr_nonrecursive(
        const ExprNode* nodes,
        std::int64_t node_idx,
        real x1,
        real x2                = 0.0,
        real x3                = 0.0,
        real t                 = 0.0,
        real dt                = 0.0,
        const real* parameters = nullptr
    );

    // batch evaluation for multiple points
    DEV void evaluate_expr_batch(
        const ExprNode* nodes,
        std::int64_t root_idx,
        const real* x1_values,
        const real* x2_values,
        const real* x3_values,
        real t,
        real dt,
        const real* parameters,
        real* results,
        std::int64_t count
    );

    // evaluate a vector of expressions (e.g., for 3D vector results)
    DEV void evaluate_expr_vector(
        const ExprNode* nodes,
        const std::int64_t* root_indices,
        std::int64_t num_components,
        real x1,
        real x2,
        real x3,
        real t,
        const real* parameters,
        real* results,
        real dt = 0.0
    );

    DEV void evaluate_linear_expr(
        const LinearExprInstr* instructions,
        std::uint64_t instruction_count,
        const std::int64_t* mapped_output_indices,
        std::uint64_t output_count,
        // std::int64_treg_count,
        real x1,
        real x2,
        real x3,
        real t,
        real dt,
        const real* parameters,
        real* outputs
    );

    std::int64_t
    get_max_register(const nd::ndarray_t<LinearExprInstr>& instructions);

}   // namespace simbi::expression

#endif
