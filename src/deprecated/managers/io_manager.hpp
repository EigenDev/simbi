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

#include "config.hpp"
#include "containers/vector.hpp"
#include "core/managers/solver_manager.hpp"   // for SolverManager
#include "core/utility/init_conditions.hpp"   // for InitialConditions
#include "core/utility/managed.hpp"
#include "util/math/evaluator.hpp"
#include "util/math/exp_load.hpp"
#include "util/math/expression.hpp"
#include "util/math/linearizer.hpp"
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
        requires(std::convertible_to<std::remove_reference_t<Args>, real> &&
                 ...) ||
                    (std::is_pointer_v<std::tuple_element_t<
                         sizeof...(Args) - 1,
                         std::tuple<std::remove_reference_t<Args>...>>>);
    };

    template <size_type Dims>
    class IOManager : public Managed<global::managed_memory>
    {
      private:
        SolverManager& solver_manager_;
        std::string data_directory_;

        size_type current_iter_{0};
        size_type checkpoint_zones_;
        size_type checkpoint_idx_{0};

        // expressions for all boundaries
        bool using_bx1_inner_expressions_{false};
        ndarray_t<expression::ExprNode> bx1_inner_expr_nodes_;
        ndarray_t<int> bx1_inner_output_indices_;
        ndarray_t<real> bx1_inner_parameters_;

        bool using_bx1_outer_expressions_{false};
        ndarray_t<expression::ExprNode> bx1_outer_expr_nodes_;
        ndarray_t<int> bx1_outer_output_indices_;
        ndarray_t<real> bx1_outer_parameters_;

        bool using_bx2_inner_expressions_{false};
        ndarray_t<expression::ExprNode> bx2_inner_expr_nodes_;
        ndarray_t<int> bx2_inner_output_indices_;
        ndarray_t<real> bx2_inner_parameters_;

        bool using_bx2_outer_expressions_{false};
        ndarray_t<expression::ExprNode> bx2_outer_expr_nodes_;
        ndarray_t<int> bx2_outer_output_indices_;
        ndarray_t<real> bx2_outer_parameters_;

        bool using_bx3_inner_expressions_{false};
        ndarray_t<expression::ExprNode> bx3_inner_expr_nodes_;
        ndarray_t<int> bx3_inner_output_indices_;
        ndarray_t<real> bx3_inner_parameters_;

        bool using_bx3_outer_expressions_{false};
        ndarray_t<expression::ExprNode> bx3_outer_expr_nodes_;
        ndarray_t<int> bx3_outer_output_indices_;
        ndarray_t<real> bx3_outer_parameters_;

        // hydro source expressions
        ndarray_t<expression::ExprNode> hydro_source_expr_nodes_;
        ndarray_t<int> hydro_source_output_indices_;
        ndarray_t<real> hydro_source_parameters_;

        // linearized expr logic
        int hydro_source_reg_count_;
        ndarray_t<int> hydro_source_linear_outputs_;
        ndarray_t<expression::LinearExprInstr> hydro_source_linear_instrs_;

        // gravity source expressions
        ndarray_t<expression::ExprNode> gravity_source_expr_nodes_;
        ndarray_t<int> gravity_source_output_indices_;
        ndarray_t<real> gravity_source_parameters_;

        // linearized expr logic
        int gravity_source_reg_count_;
        ndarray_t<int> gravity_source_linear_outputs_;
        ndarray_t<expression::LinearExprInstr> gravity_source_linear_instrs_;

        // local sound speed function
        ndarray_t<expression::ExprNode> sound_speed_expr_nodes_;
        ndarray_t<int> sound_speed_output_indices_;
        ndarray_t<real> sound_speed_parameters_;

        // linearized expr logic
        int sound_speed_reg_count_;
        ndarray_t<int> sound_speed_linear_outputs_;
        ndarray_t<expression::LinearExprInstr> sound_speed_linear_instrs_;

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
            setup_sound_speed_expressions(init);
        }

        // shared_ptr cleans up libraries
        ~IOManager() = default;

        void increment_iter() { current_iter_++; }
        void increment_checkpoint_idx() { checkpoint_idx_++; }

        // accessors
        auto& data_directory() const { return data_directory_; }
        // auto& data_directory() { return data_directory_; }
        auto current_iter() const { return current_iter_; }
        auto checkpoint_zones() const { return checkpoint_zones_; }
        auto checkpoint_index() const { return checkpoint_idx_; }

        // call 'em
        template <typename Conserved>
        DEV Conserved call_hydro_source(
            const vector_t<real, Dims> coords,
            const real time,
            const Conserved& cons
        ) const
        {
            auto local_cons = cons;
            vector_t<real, 3> local_coords{0.0, 0.0, 0.0};
            for (size_type ii = 0; ii < Dims; ++ii) {
                local_coords[ii] = coords[ii];
            }

            expression::evaluate_linear_expr(
                hydro_source_linear_instrs_.data(),
                hydro_source_linear_instrs_.size(),
                hydro_source_linear_outputs_.data(),
                hydro_source_output_indices_.size(),
                // hydro_source_reg_count_,
                local_coords[0],
                local_coords[1],
                local_coords[2],
                time,
                0.0,   // dt not used for hydro_source
                cons.data(),
                local_cons.data()
            );
            // expression::evaluate_expr_vector(
            //     hydro_source_expr_nodes_.data(),       // node data
            //     hydro_source_output_indices_.data(),   // node idx data
            //     hydro_source_output_indices_.size(),   // output element size
            //     local_coords[0],                       // x1
            //     local_coords[1],                       // x2
            //     local_coords[2],                       // x3
            //     time,                                  // time
            //     cons.data(),        // passed-in conservatives
            //     local_cons.data()   // local conservatives
            // );
            return local_cons;
        }

        DEV vector_t<real, Dims> call_gravity_source(
            const vector_t<real, Dims>& coords,
            const real time
        ) const
        {
            vector_t<real, Dims> local_vec;
            vector_t<real, 3> local_coords{0.0, 0.0, 0.0};
            for (size_type ii = 0; ii < Dims; ++ii) {
                local_coords[ii] = coords[ii];
            }

            expression::evaluate_linear_expr(
                gravity_source_linear_instrs_.data(),
                gravity_source_linear_instrs_.size(),
                gravity_source_linear_outputs_.data(),
                gravity_source_output_indices_.size(),
                // gravity_source_reg_count_,
                local_coords[0],
                local_coords[1],
                local_coords[2],
                time,
                0.0,   // dt not used for hydro_source
                nullptr,
                local_vec.data()
            );

            return local_vec;

            // expression::evaluate_expr_vector(
            //     gravity_source_expr_nodes_.data(),
            //     gravity_source_output_indices_.data(),
            //     gravity_source_output_indices_.size(),
            //     coords[0],
            //     coords[1],
            //     coords[2],
            //     t,
            //     gravity_source_parameters_.data(),
            //     results
            // );
        }

        template <typename Conserved>
        DEV Conserved call_boundary_source(
            BoundaryFace face,
            const vector_t<real, Dims>& coords,
            real t,
            real dt,
            const Conserved& conserved_data
        ) const
        {
            auto local_results = conserved_data;
            vector_t<real, 3> local_coords{0.0, 0.0, 0.0};
            for (size_type ii = 0; ii < Dims; ++ii) {
                local_coords[ii] = coords[ii];
            }
            // Determine which boundary we're evaluating
            switch (face) {
                case BoundaryFace::X1_INNER:
                    if (using_bx1_inner_expressions_) {
                        expression::evaluate_expr_vector(
                            bx1_inner_expr_nodes_.data(),
                            bx1_inner_output_indices_.data(),
                            bx1_inner_output_indices_.size(),
                            local_coords[0],
                            local_coords[1],
                            local_coords[2],
                            t,
                            conserved_data.data(),
                            local_results.data(),
                            dt
                        );
                    }
                    return local_results;

                case BoundaryFace::X1_OUTER:
                    if (using_bx1_outer_expressions_) {
                        expression::evaluate_expr_vector(
                            bx1_outer_expr_nodes_.data(),
                            bx1_outer_output_indices_.data(),
                            bx1_outer_output_indices_.size(),
                            local_coords[0],
                            local_coords[1],
                            local_coords[2],
                            t,
                            conserved_data.data(),
                            local_results.data(),
                            dt
                        );
                    }
                    return local_results;

                case BoundaryFace::X2_INNER:
                    if (using_bx2_inner_expressions_) {
                        expression::evaluate_expr_vector(
                            bx2_inner_expr_nodes_.data(),
                            bx2_inner_output_indices_.data(),
                            bx2_inner_output_indices_.size(),
                            local_coords[0],
                            local_coords[1],
                            local_coords[2],
                            t,
                            conserved_data.data(),
                            local_results.data(),
                            dt
                        );
                    }
                    return local_results;
                case BoundaryFace::X2_OUTER:
                    if (using_bx2_outer_expressions_) {
                        expression::evaluate_expr_vector(
                            bx2_outer_expr_nodes_.data(),
                            bx2_outer_output_indices_.data(),
                            bx2_outer_output_indices_.size(),
                            local_coords[0],
                            local_coords[1],
                            local_coords[2],
                            t,
                            conserved_data.data(),
                            local_results.data(),
                            dt
                        );
                    }
                    return local_results;
                case BoundaryFace::X3_INNER:
                    if (using_bx3_inner_expressions_) {
                        expression::evaluate_expr_vector(
                            bx3_inner_expr_nodes_.data(),
                            bx3_inner_output_indices_.data(),
                            bx3_inner_output_indices_.size(),
                            local_coords[0],
                            local_coords[1],
                            local_coords[2],
                            t,
                            conserved_data.data(),
                            local_results.data(),
                            dt
                        );
                    }
                    return local_results;
                default:
                    if (using_bx3_outer_expressions_) {
                        expression::evaluate_expr_vector(
                            bx3_outer_expr_nodes_.data(),
                            bx3_outer_output_indices_.data(),
                            bx3_outer_output_indices_.size(),
                            local_coords[0],
                            local_coords[1],
                            local_coords[2],
                            t,
                            conserved_data.data(),
                            local_results.data(),
                            dt
                        );
                    }
                    return local_results;
            }
        }

        DEV real
        call_local_sound_speed(const vector_t<real, Dims>& coords) const
        {
            vector_t<real, 3> local_coords{0.0, 0.0, 0.0};
            for (size_type ii = 0; ii < Dims; ++ii) {
                local_coords[ii] = coords[ii];
            }
            real local_sound_speed[1] = {0.0};
            expression::evaluate_linear_expr(
                sound_speed_linear_instrs_.data(),
                sound_speed_linear_instrs_.size(),
                sound_speed_linear_outputs_.data(),
                sound_speed_output_indices_.size(),
                // sound_speed_reg_count_,
                local_coords[0],
                local_coords[1],
                local_coords[2],
                0.0,   // time not used for sound speed
                0.0,   // dt not used for sound speed
                nullptr,
                local_sound_speed
            );

            return local_sound_speed[0];
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
                auto [node, indices, params] = expression::load_expression_data(
                    init.bx1_inner_expressions
                );
                bx1_inner_expr_nodes_        = std::move(node);
                bx1_inner_output_indices_    = std::move(indices);
                bx1_inner_parameters_        = std::move(params);
                using_bx1_inner_expressions_ = true;
            }
            if (init.bx1_outer_expressions.size() > 0) {
                // load the expressions :)))
                auto [node, indices, params] = expression::load_expression_data(
                    init.bx1_outer_expressions
                );
                bx1_outer_expr_nodes_        = std::move(node);
                bx1_outer_output_indices_    = std::move(indices);
                bx1_outer_parameters_        = std::move(params);
                using_bx1_outer_expressions_ = true;
            }

            if (init.bx2_inner_expressions.size() > 0) {
                // load the expressions :)))
                auto [node, indices, params] = expression::load_expression_data(
                    init.bx2_inner_expressions
                );
                bx2_inner_expr_nodes_        = std::move(node);
                bx2_inner_output_indices_    = std::move(indices);
                bx2_inner_parameters_        = std::move(params);
                using_bx2_inner_expressions_ = true;
            }
            if (init.bx2_outer_expressions.size() > 0) {
                // load the expressions :)))
                auto [node, indices, params] = expression::load_expression_data(
                    init.bx2_outer_expressions
                );
                bx2_outer_expr_nodes_        = std::move(node);
                bx2_outer_output_indices_    = std::move(indices);
                bx2_outer_parameters_        = std::move(params);
                using_bx2_outer_expressions_ = true;
            }

            if (init.bx3_inner_expressions.size() > 0) {
                // load the expressions :)))
                auto [node, indices, params] = expression::load_expression_data(
                    init.bx3_inner_expressions
                );
                bx3_inner_expr_nodes_        = std::move(node);
                bx3_inner_output_indices_    = std::move(indices);
                bx3_inner_parameters_        = std::move(params);
                using_bx3_inner_expressions_ = true;
            }
            if (init.bx3_outer_expressions.size() > 0) {
                // load the expressions :)))
                auto [node, indices, params] = expression::load_expression_data(
                    init.bx3_outer_expressions
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

                // now we linearize it
                auto [linear_instrs, mapped_outputs] =
                    expression::linearize_expression_tree(
                        gravity_source_expr_nodes_,
                        gravity_source_output_indices_
                    );
                gravity_source_linear_instrs_  = std::move(linear_instrs);
                gravity_source_linear_outputs_ = std::move(mapped_outputs);
                gravity_source_reg_count_      = expression::get_max_register(
                                                gravity_source_linear_instrs_
                                            ) +
                                            1;
                // sync everything to device
                gravity_source_expr_nodes_.sync_to_device();
                gravity_source_output_indices_.sync_to_device();
                gravity_source_parameters_.sync_to_device();
                gravity_source_linear_instrs_.sync_to_device();
                gravity_source_linear_outputs_.sync_to_device();

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

                // idem
                auto [linear_instrs, mapped_outputs] =
                    expression::linearize_expression_tree(
                        hydro_source_expr_nodes_,
                        hydro_source_output_indices_
                    );

                hydro_source_linear_instrs_  = std::move(linear_instrs);
                hydro_source_linear_outputs_ = std::move(mapped_outputs);
                hydro_source_reg_count_ =
                    expression::get_max_register(hydro_source_linear_instrs_) +
                    1;

                // sync everything to device
                hydro_source_expr_nodes_.sync_to_device();
                hydro_source_output_indices_.sync_to_device();
                hydro_source_parameters_.sync_to_device();
                hydro_source_linear_instrs_.sync_to_device();
                hydro_source_linear_outputs_.sync_to_device();

                solver_manager_.set_null_sources(false);
            }
            else {
                solver_manager_.set_null_sources(true);
            }
        }

        void setup_sound_speed_expressions(const InitialConditions& init)
        {
            if (init.local_sound_speed_expressions.size() > 0) {
                // load the expressions :)))
                auto [node, indices, params] = expression::load_expression_data(
                    init.local_sound_speed_expressions
                );

                sound_speed_expr_nodes_     = std::move(node);
                sound_speed_output_indices_ = std::move(indices);
                sound_speed_parameters_     = std::move(params);

                // idem
                auto [linear_instrs, mapped_outputs] =
                    expression::linearize_expression_tree(
                        sound_speed_expr_nodes_,
                        sound_speed_output_indices_
                    );
                sound_speed_linear_instrs_  = std::move(linear_instrs);
                sound_speed_linear_outputs_ = std::move(mapped_outputs);
                sound_speed_reg_count_ =
                    expression::get_max_register(sound_speed_linear_instrs_) +
                    1;

                // sync everything to device
                sound_speed_expr_nodes_.sync_to_device();
                sound_speed_output_indices_.sync_to_device();
                sound_speed_parameters_.sync_to_device();
                sound_speed_linear_instrs_.sync_to_device();
                sound_speed_linear_outputs_.sync_to_device();
            }
        }
    };   // namespace simbi
}   // namespace simbi

#endif
