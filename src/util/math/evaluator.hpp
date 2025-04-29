#ifndef EVALUATOR_HPP
#define EVALUATOR_HPP

#include "build_options.hpp"
#include "core/types/containers/ndarray.hpp"
#include "util/math/expression.hpp"

namespace simbi::expression {
    // main evaluation function with parameter support
    DEV real evaluate_expr(
        const ExprNode* nodes,
        int node_idx,
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
        int node_idx,
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
        int root_idx,
        const real* x1_values,
        const real* x2_values,
        const real* x3_values,
        real t,
        real dt,
        const real* parameters,
        real* results,
        int count
    );

    // evaluate a vector of expressions (e.g., for 3D vector results)
    DEV void evaluate_expr_vector(
        const ExprNode* nodes,
        const int* root_indices,   // array of root node indices
        int num_components,        // number of components in the vector
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
        size_type instruction_count,
        const int* mapped_output_indices,
        size_type output_count,
        int reg_count,
        real x1,
        real x2,
        real x3,
        real t,
        real dt,
        const real* parameters,
        real* outputs
    );

    int get_max_register(const ndarray<LinearExprInstr>& instructions);

}   // namespace simbi::expression

#endif
