/**
 *  *=============================================================================
 *  *           SIMBI - Special Relativistic Magnetohydrodynamics Code
 *  *=============================================================================
 *  *
 *  * @file            sink.hpp
 *  * @brief           Sink Immersed Body Implementation
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

#ifndef SINK_HPP
#define SINK_HPP

#include "build_options.hpp"
#include "gravitational.hpp"

namespace simbi {
    namespace ib {
        // concrete sink particle implementation
        template <typename T, size_type Dims, typename MeshType>
        class SinkParticle : public ImmersedBody<T, Dims, MeshType>
        {
          protected:
            T accretion_efficiency_;
            T accretion_radius_;

          public:
            DUAL SinkParticle(
                const MeshType& mesh,
                const spatial_vector_t<T, Dims>& position,
                const spatial_vector_t<T, Dims>& velocity,
                const T mass,
                const T radius,
                const T accretion_efficiency
            )
                : ImmersedBody<T, Dims, MeshType>(
                      mesh,
                      position,
                      velocity,
                      mass,
                      radius
                  ),
                  accretion_efficiency_(accretion_efficiency),
                  accretion_radius_(radius)
            {
                this->is_sink_ = true;
            }

            // Sink particle methods
            DUAL void accrete(auto& prim_states)
            {
                T total_mass = 0;
                spatial_vector_t<T, Dims> total_momentum;

                for (const auto& idx : this->cut_cell_indices()) {
                    const auto& cell = this->cell_info_[idx];
                    if (std::abs(cell.distance) <= accretion_radius_) {
                        const auto mesh_cell = this->mesh_.cell_geometry(idx);
                        const T dm =
                            accretion_efficiency_ * prim_states[idx].rho() *
                            cell.volume_fraction * mesh_cell.calculate_volume();

                        total_mass += dm;
                        total_momentum += prim_states[idx].velocity() * dm;

                        prim_states[idx].rho() *= (1.0 - cell.volume_fraction);
                    }
                }

                if (total_mass > 0) {
                    this->mass_ += total_mass;
                    this->velocity_ =
                        total_momentum / (this->mass_ * total_mass);
                }
            }
        };

        // gravitational sink functionality
        template <typename T, size_type Dims, typename MeshType>
        class GravitationalSinkParticle
            : public GravitationalBody<T, Dims, MeshType>
        {
          protected:
            struct AccretionStats {
                T total_mass_accreted{0};
                T total_energy_acrcreted{0};
                spatial_vector_t<T, Dims> total_momentum_accreted;
                spatial_vector_t<T, Dims> total_angular_momentum_accreted;
            };

            T accretion_efficiency_;
            T accretion_radius_;
            AccretionStats accretion_stats_;

          public:
            DUAL GravitationalSinkParticle(
                const MeshType& mesh,
                const spatial_vector_t<T, Dims>& position,
                const spatial_vector_t<T, Dims>& velocity,
                const T mass,
                const T radius,
                const T grav_strength,
                const T softening,
                const T accretion_efficiency
            )
                : GravitationalBody<T, Dims, MeshType>(
                      mesh,
                      position,
                      velocity,
                      mass,
                      radius,
                      grav_strength,
                      softening
                  ),
                  accretion_efficiency_(accretion_efficiency),
                  accretion_radius_(radius)
            {
            }

            DUAL bool
            should_accrete(const auto& cell, const auto& prim_state, const T cs)
                const
            {
                if (std::abs(cell.distance) > accretion_radius_) {
                    return false;
                }

                // bondi radius
                const auto r_bondi =
                    2.0 * this->grav_strength_ * this->mass_ / (cs * cs);
                if (std::abs(cell.distance) > r_bondi) {
                    return false;
                }

                // escape velocity check
                const auto v_esc = std::sqrt(
                    2.0 * this->grav_strength_ * this->mass_ /
                    std::abs(cell.distance)
                );
                const auto v_rel = (this->velocity_ - prim_state.velocity());
                if (v_rel.norm() > v_esc) {
                    return false;
                }

                return true;
            }

            DUAL void accrete(auto& prim_states, real gamma)
            {
                T total_mass = 0;
                spatial_vector_t<T, Dims> total_momentum;

                for (const auto& idx : this->cut_cell_indices()) {
                    const auto& cell = this->cell_info_[idx];

                    // Get local sound speed
                    const T cs = prim_states[idx].sound_speed(gamma);

                    if (should_accrete(cell, prim_states[idx], cs)) {
                        const auto mesh_cell = this->mesh_.cell_geometry(idx);
                        const T dm =
                            accretion_efficiency_ * prim_states[idx].rho() *
                            cell.volume_fraction * mesh_cell.volume()();

                        total_mass += dm;
                        total_momentum += prim_states[idx].velocity() * dm;

                        // remove mass from fluid
                        prim_states[idx].rho() *= (1.0 - cell.volume_fraction);

                        // update accretion stats
                        accretion_stats_.total_mass_accreted += dm;
                        accretion_stats_.total_energy_acrcreted +=
                            0.5 * dm *
                            prim_states[idx].velocity().norm_squared();
                        accretion_stats_.total_momentum_accreted +=
                            prim_states[idx].velocity() * dm;
                    }
                }

                // Update particle state
                if (total_mass > 0) {
                    const auto old_momentum = this->mass_ * this->velocity_;
                    this->mass_ += total_mass;
                    this->velocity_ =
                        (old_momentum + total_momentum) / this->mass_;

                    // Update accretion radius based on new mass
                    accretion_radius_ =
                        this->radius_ *
                        std::sqrt(
                            this->mass_ / accretion_stats_.total_mass_accreted
                        );
                }
            }

            DUAL auto accretion_stats() const { return accretion_stats_; }
        };

    }   // namespace ib

}   // namespace simbi

#endif