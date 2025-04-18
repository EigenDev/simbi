/**
 *  *=============================================================================
 *  *           SIMBI - Special Relativistic Magnetohydrodynamics Code
 *  *=============================================================================
 *  *
 *  * @file            io_manager.hpp
 *  * @brief           I/O manager for HDF5 file I/O and JIT compilation
 *  * @details
 *  *
 *  * @version         0.8.0
 *  * @date            2025-02-23
 *  * @author          Marcus DuPont
 *  * @email           marcus.dupont@princeton.edu
 *  *
 *  *==============================================================================
 *  * @build           Requirements & Dependencies
 *  *==============================================================================
 *  * @requires        C++20
 *  * @depends         CUDA >= 11.0, HDF5 >= 1.12, OpenMP >= 4.5
 *  * @platform        Linux, MacOS
 *  * @parallel        GPU (CUDA, HIP), CPU (OpenMP)
 *  *
 *  *==============================================================================
 *  * @documentation   Reference & Notes
 *  *==============================================================================
 *  * @usage
 *  * @note
 *  * @warning
 *  * @todo
 *  * @bug
 *  * @performance
 *  *
 *  *==============================================================================
 *  * @testing        Quality Assurance
 *  *==============================================================================
 *  * @test
 *  * @benchmark
 *  * @validation
 *  *
 *  *==============================================================================
 *  * @history        Version History
 *  *==============================================================================
 *  * 2025-02-23      v0.8.0      Initial implementation
 *  *
 *  *==============================================================================
 *  * @copyright (C) 2025 Marcus DuPont. All rights reserved.
 *  *==============================================================================
 */
#ifndef IO_MANAGER_HPP
#define IO_MANAGER_HPP

#include "build_options.hpp"
#include "core/managers/solver_manager.hpp"         // for SolverManager
#include "core/types/utility/init_conditions.hpp"   // for InitialConditions
#include "physics/hydro/types/generic_structs.hpp"
#include "util/jit/evaluator.hpp"
#include "util/jit/exp_load.hpp"
#include <string>

namespace simbi {

    template <size_type D>
    struct SourceParams {
        using type = std::conditional_t<
            D == 1,
            std::tuple<real, real, real*>,   // 1D: x1, t, res
            std::conditional_t<
                D == 2,
                std::tuple<real, real, real, real*>,   // 2D: x1, x2, t, res
                std::tuple<real, real, real, real, real*>>>;   // 3D: x1,
                                                               // x2, x3, t,
                                                               // res
    };

    template <size_type D>
    struct FunctionSignature {
        using type = std::conditional_t<
            D == 1,
            void(real, real, real*),   // 1D: x1, t, res
            std::conditional_t<
                D == 2,
                void(real, real, real, real*),           // 2D: x1, x2, t, res
                void(real, real, real, real, real*)>>;   // 3D: x1,
                                                         // x2, x3,
                                                         // t, res
    };

    template <size_type D>
    using function_signature_t = typename FunctionSignature<D>::type;

    template <size_type D, typename... Args>
    concept ValidSourceParams = requires(Args... args) {
        requires sizeof...(Args) == D + 2;
        requires(std::convertible_to<std::remove_reference_t<Args>, real> && ...
                ) ||
                    (std::is_pointer_v<std::tuple_element_t<
                         sizeof...(Args) - 1,
                         std::tuple<std::remove_reference_t<Args>...>>>);
    };

    template <size_type Dims>
    class IOManager
    {
      private:
        SolverManager& solver_manager_;
        std::string data_directory_;

        size_type current_iter_{0};
        size_type checkpoint_zones_;
        size_type checkpoint_idx_{0};

        // expressions for all boundaries
        bool using_bx1_inner_expressions_{false};
        ndarray<expression::ExprNode> bx1_inner_expr_nodes_;
        ndarray<int> bx1_inner_output_indices_;
        ndarray<real> bx1_inner_parameters_;

        bool using_bx1_outer_expressions_{false};
        ndarray<expression::ExprNode> bx1_outer_expr_nodes_;
        ndarray<int> bx1_outer_output_indices_;
        ndarray<real> bx1_outer_parameters_;

        bool using_bx2_inner_expressions_{false};
        ndarray<expression::ExprNode> bx2_inner_expr_nodes_;
        ndarray<int> bx2_inner_output_indices_;
        ndarray<real> bx2_inner_parameters_;

        bool using_bx2_outer_expressions_{false};
        ndarray<expression::ExprNode> bx2_outer_expr_nodes_;
        ndarray<int> bx2_outer_output_indices_;
        ndarray<real> bx2_outer_parameters_;

        bool using_bx3_inner_expressions_{false};
        ndarray<expression::ExprNode> bx3_inner_expr_nodes_;
        ndarray<int> bx3_inner_output_indices_;
        ndarray<real> bx3_inner_parameters_;

        bool using_bx3_outer_expressions_{false};
        ndarray<expression::ExprNode> bx3_outer_expr_nodes_;
        ndarray<int> bx3_outer_output_indices_;
        ndarray<real> bx3_outer_parameters_;

        // hydro source expressions
        ndarray<expression::ExprNode> hydro_source_expr_nodes_;
        ndarray<int> hydro_source_output_indices_;
        ndarray<real> hydro_source_parameters_;

        // gravity source expressions
        ndarray<expression::ExprNode> gravity_source_expr_nodes_;
        ndarray<int> gravity_source_output_indices_;
        ndarray<real> gravity_source_parameters_;

      public:
        // move constructor and assignment
        IOManager(IOManager&& other) noexcept            = default;
        IOManager& operator=(IOManager&& other) noexcept = default;

        // Delete copy operations to prevent double-free
        IOManager(const IOManager&)            = delete;
        IOManager& operator=(const IOManager&) = delete;

        IOManager(SolverManager& solver_manager, const InitialConditions& init)
            : solver_manager_(solver_manager),
              data_directory_(init.data_directory),
              checkpoint_zones_(determine_checkpoint_zones(init)),
              checkpoint_idx_(init.checkpoint_index)
        {
            setup_boundary_expressions(init);
            setup_hydro_source_expressions(init);
            setup_gravity_source_expressions(init);
        }

        // shared_ptr cleans up libraries
        ~IOManager() = default;

        void increment_iter() { current_iter_++; }
        void increment_checkpoint_idx() { checkpoint_idx_++; }

        // accessors
        auto& data_directory() const { return data_directory_; }
        auto& data_directory() { return data_directory_; }
        auto current_iter() const { return current_iter_; }
        auto checkpoint_zones() const { return checkpoint_zones_; }
        auto checkpoint_index() const { return checkpoint_idx_; }

        // call 'em
        template <typename... Args>
            requires(sizeof...(Args) == Dims + 2)
        DEV void call_hydro_source(Args&&... args) const
        {
            // extract the coordinates and results array from args
            real coords[3] = {0.0, 0.0, 0.0};
            real t         = 0.0;
            real* results  = nullptr;

            if constexpr (Dims == 1) {
                //  1D: args are (x1, t, results*)
                std::tie(coords[0], t, results) =
                    std::forward_as_tuple(std::forward<Args>(args)...);
            }
            else if constexpr (Dims == 2) {
                //  2D: args are (x1, x2, t, results*)
                std::tie(coords[0], coords[1], t, results) =
                    std::forward_as_tuple(std::forward<Args>(args)...);
            }
            else if constexpr (Dims == 3) {
                //  3D: args are (x1, x2, x3, t, results*)
                std::tie(coords[0], coords[1], coords[2], t, results) =
                    std::forward_as_tuple(std::forward<Args>(args)...);
            }

            expression::evaluate_expr_vector(
                hydro_source_expr_nodes_.data(),
                hydro_source_output_indices_.data(),
                hydro_source_output_indices_.size(),
                coords[0],
                coords[1],
                coords[2],
                t,
                hydro_source_parameters_.data(),
                results
            );
        }

        template <typename... Args>
            requires(sizeof...(Args) == Dims + 2)
        DEV void call_gravity_source(Args&&... args) const
        {
            // extract the coordinates and results array from args
            real coords[3] = {0.0, 0.0, 0.0};
            real t         = 0.0;
            real* results  = nullptr;

            if constexpr (Dims == 1) {
                //  1D: args are (x1, t, results*)
                std::tie(coords[0], t, results) =
                    std::forward_as_tuple(std::forward<Args>(args)...);
            }
            else if constexpr (Dims == 2) {
                //  2D: args are (x1, x2, t, results*)
                std::tie(coords[0], coords[1], t, results) =
                    std::forward_as_tuple(std::forward<Args>(args)...);
            }
            else if constexpr (Dims == 3) {
                //  3D: args are (x1, x2, x3, t, results*)
                std::tie(coords[0], coords[1], coords[2], t, results) =
                    std::forward_as_tuple(std::forward<Args>(args)...);
            }

            expression::evaluate_expr_vector(
                gravity_source_expr_nodes_.data(),
                gravity_source_output_indices_.data(),
                gravity_source_output_indices_.size(),
                coords[0],
                coords[1],
                coords[2],
                t,
                gravity_source_parameters_.data(),
                results
            );
        }

        template <typename... Args>
            requires(sizeof...(Args) == Dims + 3)
        DEV void call_boundary_source(BoundaryFace face, Args&&... args) const
        {
            // extract the coordinates and results array from args
            real coords[3] = {0.0, 0.0, 0.0};
            real t         = 0.0;
            real dt        = 0.0;
            real* results  = nullptr;

            if constexpr (Dims == 1) {
                //  1D: args are (x1, t, results*)
                std::tie(coords[0], t, dt, results) =
                    std::forward_as_tuple(std::forward<Args>(args)...);
            }
            else if constexpr (Dims == 2) {
                //  2D: args are (x1, x2, t, results*)
                std::tie(coords[0], coords[1], t, dt, results) =
                    std::forward_as_tuple(std::forward<Args>(args)...);
            }
            else if constexpr (Dims == 3) {
                //  3D: args are (x1, x2, x3, t, results*)
                std::tie(coords[0], coords[1], coords[2], t, dt, results) =
                    std::forward_as_tuple(std::forward<Args>(args)...);
            }

            const auto nvars = bx1_inner_parameters_.size();
            // Determine which boundary we're evaluating
            switch (face) {
                case BoundaryFace::X1_INNER:
                    if (using_bx1_inner_expressions_) {
                        for (size_type ii = 0; ii < nvars; ii++) {
                            bx1_inner_parameters_[ii] = results[ii];
                        }
                        expression::evaluate_expr_vector(
                            bx1_inner_expr_nodes_.data(),
                            bx1_inner_output_indices_.data(),
                            bx1_inner_output_indices_.size(),
                            coords[0],
                            coords[1],
                            coords[2],
                            t,
                            bx1_inner_parameters_.data(),
                            results,
                            dt
                        );
                    }
                    break;

                case BoundaryFace::X1_OUTER:
                    if (using_bx1_outer_expressions_) {
                        for (size_type ii = 0; ii < nvars; ii++) {
                            bx1_outer_parameters_[ii] = results[ii];
                        }
                        expression::evaluate_expr_vector(
                            bx1_outer_expr_nodes_.data(),
                            bx1_outer_output_indices_.data(),
                            bx1_outer_output_indices_.size(),
                            coords[0],
                            coords[1],
                            coords[2],
                            t,
                            bx1_outer_parameters_.data(),
                            results,
                            dt
                        );
                    }
                    break;

                case BoundaryFace::X2_INNER:
                    if (using_bx2_inner_expressions_) {
                        for (size_type ii = 0; ii < nvars; ii++) {
                            bx2_inner_parameters_[ii] = results[ii];
                        }
                        expression::evaluate_expr_vector(
                            bx2_inner_expr_nodes_.data(),
                            bx2_inner_output_indices_.data(),
                            bx2_inner_output_indices_.size(),
                            coords[0],
                            coords[1],
                            coords[2],
                            t,
                            bx2_inner_parameters_.data(),
                            results,
                            dt
                        );
                    }
                    break;
                case BoundaryFace::X2_OUTER:
                    if (using_bx2_outer_expressions_) {
                        for (size_type ii = 0; ii < nvars; ii++) {
                            bx2_outer_parameters_[ii] = results[ii];
                        }
                        expression::evaluate_expr_vector(
                            bx2_outer_expr_nodes_.data(),
                            bx2_outer_output_indices_.data(),
                            bx2_outer_output_indices_.size(),
                            coords[0],
                            coords[1],
                            coords[2],
                            t,
                            bx2_outer_parameters_.data(),
                            results,
                            dt
                        );
                    }
                    break;
                case BoundaryFace::X3_INNER:
                    if (using_bx3_inner_expressions_) {
                        for (size_type ii = 0; ii < nvars; ii++) {
                            bx3_inner_parameters_[ii] = results[ii];
                        }
                        expression::evaluate_expr_vector(
                            bx3_inner_expr_nodes_.data(),
                            bx3_inner_output_indices_.data(),
                            bx3_inner_output_indices_.size(),
                            coords[0],
                            coords[1],
                            coords[2],
                            t,
                            bx3_inner_parameters_.data(),
                            results,
                            dt
                        );
                    }
                    break;
                case BoundaryFace::X3_OUTER:
                    if (using_bx3_outer_expressions_) {
                        for (size_type ii = 0; ii < nvars; ii++) {
                            bx3_outer_parameters_[ii] = results[ii];
                        }
                        expression::evaluate_expr_vector(
                            bx3_outer_expr_nodes_.data(),
                            bx3_outer_output_indices_.data(),
                            bx3_outer_output_indices_.size(),
                            coords[0],
                            coords[1],
                            coords[2],
                            t,
                            bx3_outer_parameters_.data(),
                            results,
                            dt
                        );
                    }
            }
        }

      private:
        size_type determine_checkpoint_zones(const InitialConditions& init)
        {
            const auto [xag, yag, zag] = init.active_zones();
            return (zag > 1) ? zag : (yag > 1) ? yag : xag;
        }

        void setup_boundary_expressions(const InitialConditions& init)
        {
            if (init.bx1_inner_expressions.size() > 0) {
                // load the expressions :)))
                auto [node, indices, params] =
                    expression::load_expression_data(init.bx1_inner_expressions
                    );
                bx1_inner_expr_nodes_        = std::move(node);
                bx1_inner_output_indices_    = std::move(indices);
                bx1_inner_parameters_        = std::move(params);
                using_bx1_inner_expressions_ = true;
            }
            if (init.bx1_outer_expressions.size() > 0) {
                // load the expressions :)))
                auto [node, indices, params] =
                    expression::load_expression_data(init.bx1_outer_expressions
                    );
                bx1_outer_expr_nodes_        = std::move(node);
                bx1_outer_output_indices_    = std::move(indices);
                bx1_outer_parameters_        = std::move(params);
                using_bx1_outer_expressions_ = true;
            }

            if (init.bx2_inner_expressions.size() > 0) {
                // load the expressions :)))
                auto [node, indices, params] =
                    expression::load_expression_data(init.bx2_inner_expressions
                    );
                bx2_inner_expr_nodes_        = std::move(node);
                bx2_inner_output_indices_    = std::move(indices);
                bx2_inner_parameters_        = std::move(params);
                using_bx2_inner_expressions_ = true;
            }
            if (init.bx2_outer_expressions.size() > 0) {
                // load the expressions :)))
                auto [node, indices, params] =
                    expression::load_expression_data(init.bx2_outer_expressions
                    );
                bx2_outer_expr_nodes_        = std::move(node);
                bx2_outer_output_indices_    = std::move(indices);
                bx2_outer_parameters_        = std::move(params);
                using_bx2_outer_expressions_ = true;
            }

            if (init.bx3_inner_expressions.size() > 0) {
                // load the expressions :)))
                auto [node, indices, params] =
                    expression::load_expression_data(init.bx3_inner_expressions
                    );
                bx3_inner_expr_nodes_        = std::move(node);
                bx3_inner_output_indices_    = std::move(indices);
                bx3_inner_parameters_        = std::move(params);
                using_bx3_inner_expressions_ = true;
            }
            if (init.bx3_outer_expressions.size() > 0) {
                // load the expressions :)))
                auto [node, indices, params] =
                    expression::load_expression_data(init.bx3_outer_expressions
                    );
                bx3_outer_expr_nodes_        = std::move(node);
                bx3_outer_output_indices_    = std::move(indices);
                bx3_outer_parameters_        = std::move(params);
                using_bx3_outer_expressions_ = true;
            }
        }

        void setup_gravity_source_expressions(const InitialConditions& init)
        {
            if (init.gravity_source_expressions.size() > 0) {
                // load the expressions :)))
                auto [node, indices, params] = expression::load_expression_data(
                    init.gravity_source_expressions
                );
                gravity_source_expr_nodes_     = std::move(node);
                gravity_source_output_indices_ = std::move(indices);
                gravity_source_parameters_     = std::move(params);
                solver_manager_.set_null_gravity(false);
            }
            else {
                solver_manager_.set_null_gravity(true);
            }
        }

        void setup_hydro_source_expressions(const InitialConditions& init)
        {
            if (init.hydro_source_expressions.size() > 0) {
                // load the expressions :)))
                auto [node, indices, params] = expression::load_expression_data(
                    init.hydro_source_expressions
                );
                hydro_source_expr_nodes_     = std::move(node);
                hydro_source_output_indices_ = std::move(indices);
                hydro_source_parameters_     = std::move(params);
                solver_manager_.set_null_sources(false);
            }
            else {
                solver_manager_.set_null_sources(true);
            }
        }
    };   // namespace simbi
}   // namespace simbi

#endif
