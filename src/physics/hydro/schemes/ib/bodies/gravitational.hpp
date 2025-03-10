/**
 *  *=============================================================================
 *  *           SIMBI - Special Relativistic Magnetohydrodynamics Code
 *  *=============================================================================
 *  *
 *  * @file            gravitational.hpp
 *  * @brief           Gravitational Immersed Body Implementation
 *  * @details
 *  *
 *  * @version         0.8.0
 *  * @date            2025-02-16
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
 *  * 2025-02-16      v0.8.0      Initial implementation
 *  *
 *  *==============================================================================
 *  * @copyright (C) 2025 Marcus DuPont. All rights reserved.
 *  *==============================================================================
 */

#ifndef GRAVITATIONAL_HPP
#define GRAVITATIONAL_HPP

#include "build_options.hpp"
#include "immersed_boundary.hpp"

namespace simbi {
    namespace ib {

        // gravitational functionality
        template <typename T, size_type Dims, typename MeshType>
        class GravitationalBody : public ImmersedBody<T, Dims, MeshType>
        {
          protected:
            T softening_length_;   // softening length for gravity
            T grav_strength_;      // dimensionless gravitational coupling

          public:
            DUAL GravitationalBody(
                const MeshType& mesh,
                const spatial_vector_t<T, Dims>& position,
                const spatial_vector_t<T, Dims>& velocity,
                const T mass,
                const T radius,
                const T grav_strength,
                const T softening = 0.01   // Small fraction of radius
            )
                : ImmersedBody<T, Dims, MeshType>(
                      mesh,
                      position,
                      velocity,
                      mass,
                      radius
                  ),
                  softening_length_(softening * radius),
                  grav_strength_(grav_strength)
            {
                if (grav_strength_ < 0) {
                    throw std::invalid_argument(
                        "Gravitational strength must be non-negative."
                    );
                }
            }

            DUAL void compute_body_forces(
                const std::vector<
                    std::unique_ptr<ib::ImmersedBody<T, Dims, MeshType>>>&
                    bodies
            ) override
            {
                this->force_ = spatial_vector_t<T, Dims>();
                for (const auto& other : bodies) {
                    if (other.get() != this) {
                        const auto r = other->position() - this->position_;
                        const auto r2 =
                            r.dot(r) + softening_length_ * softening_length_;
                        this->force_ += -grav_strength_ * other->mass() * r /
                                        (r2 * std::sqrt(r2));
                    }
                }
            }

            DUAL void compute_surface_forces(
                const ImmersedBody<T, Dims, MeshType>::CellInfo& cell,
                const spatial_vector_t<T, Dims>& dA_normal
            ) override
            {
                // do nothing
            }

            template <typename StateArray, typename PrimitiveArray>
            DUAL void apply_body_forces_impl(
                StateArray& cons_states,
                const PrimitiveArray& prims,
                const std::vector<
                    std::unique_ptr<ImmersedBody<T, Dims, MeshType>>>& bodies,
                const T dt
            )
            {
                using conserved_t = typename StateArray::value_type;

                // Reset force accumulator
                this->force_ = spatial_vector_t<T, Dims>();

                // Apply N-body gravitational forces between bodies
                compute_body_forces(bodies);

                // Apply gravitational force to entire fluid domain
                cons_states.transform_with_indices(
                    [&](auto& state, size_type idx, auto& prim) {
                        const auto mesh_cell =
                            this->mesh_.get_cell_from_global(idx);
                        const auto r = mesh_cell.centroid() - this->position_;
                        const auto r2 =
                            r.dot(r) + softening_length_ * softening_length_;

                        // Gravitational force on fluid element
                        const auto force = -grav_strength_ * this->mass_ * r /
                                           (r2 * std::sqrt(r2));

                        // momentum and energy change
                        const auto dp = prim->rho() * force * dt;

                        const auto v_old = prim->velocity();
                        auto v_new = (state.momentum() + dp) / prim->rho();
                        const auto v_avg = 0.5 * (v_old + v_new);
                        const auto dE    = dp.dot(v_avg);
                        state += conserved_t{0.0, dp, dE};

                        // Store reaction force on body
                        this->force_ -=
                            force * state.dens() * mesh_cell.volume();

                        return state;
                    },
                    this->get_default_policy(),
                    prims
                );
            }

            // Override each regime-specific version
            DUAL void apply_body_forces(
                ndarray<anyConserved<Dims, Regime::NEWTONIAN>, Dims>&
                    cons_states,
                const ndarray<
                    Maybe<anyPrimitive<Dims, Regime::NEWTONIAN>>,
                    Dims>& prims,
                const std::vector<
                    std::unique_ptr<ImmersedBody<T, Dims, MeshType>>>& bodies,
                const T dt
            ) override
            {
                apply_body_forces_impl(cons_states, prims, bodies, dt);
            }

            DUAL void apply_body_forces(
                ndarray<anyConserved<Dims, Regime::SRHD>, Dims>& cons_states,
                const ndarray<Maybe<anyPrimitive<Dims, Regime::SRHD>>, Dims>&
                    prims,
                const std::vector<
                    std::unique_ptr<ImmersedBody<T, Dims, MeshType>>>& bodies,
                const T dt
            ) override
            {
                apply_body_forces_impl(cons_states, prims, bodies, dt);
            }

            DUAL void apply_body_forces(
                ndarray<anyConserved<Dims, Regime::RMHD>, Dims>& cons_states,
                const ndarray<Maybe<anyPrimitive<Dims, Regime::RMHD>>, Dims>&
                    prims,
                const std::vector<
                    std::unique_ptr<ImmersedBody<T, Dims, MeshType>>>& bodies,
                const T dt
            ) override
            {
                apply_body_forces_impl(cons_states, prims, bodies, dt);
            }
        };

    }   // namespace ib

}   // namespace simbi

#endif