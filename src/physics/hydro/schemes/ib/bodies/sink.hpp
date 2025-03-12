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
                        const auto mesh_cell =
                            this->mesh_.get_cell_from_global(idx);
                        const T dm =
                            accretion_efficiency_ * prim_states[idx]->rho() *
                            cell.volume_fraction * mesh_cell.calculate_volume();

                        total_mass += dm;
                        total_momentum += prim_states[idx]->velocity() * dm;

                        prim_states[idx]->rho() *= (1.0 - cell.volume_fraction);
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
                T total_energy_accreted{0};
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
                const auto v_rel = (this->velocity_ - prim_state->velocity());
                if (v_rel.norm() > v_esc) {
                    return false;
                }

                return true;
            }

            DUAL void accrete(
                auto& cons_states,
                const auto& prim_states,
                const real gamma
            )
            {
                T total_mass = 0;
                spatial_vector_t<T, Dims> total_momentum;
                const auto cut_cells = this->cut_cell_indices();

                // floor values for stability
                const auto density_floor = global::epsilon;

                for (const auto idx : cut_cells) {
                    const auto& cell = this->cell_info_[idx];

                    // Get local sound speed
                    const T cs = prim_states[idx]->sound_speed(gamma);

                    if (should_accrete(cell, prim_states[idx], cs)) {
                        const auto mesh_cell =
                            this->mesh_.get_cell_from_global(idx);
                        const T cell_mass =
                            cons_states[idx].dens() * mesh_cell.volume();

                        // Calculate maximum mass fraction that can be removed
                        const T max_removable_fraction =
                            (cell_mass - density_floor * mesh_cell.volume()) /
                            cell_mass;

                        // Calculate desired removal fraction based on
                        // efficiency
                        const T desired_fraction =
                            accretion_efficiency_ * cell.volume_fraction;

                        // Take minimum to ensure we don't go below floor
                        const T removal_fraction =
                            std::min(max_removable_fraction, desired_fraction);

                        if (removal_fraction > 0) {
                            // Calculate actual mass removed
                            const T dm = removal_fraction * cell_mass;

                            // Remove mass and momentum proportionally
                            cons_states[idx].dens() *= (1.0 - removal_fraction);
                            cons_states[idx].momentum() *=
                                (1.0 - removal_fraction);
                            cons_states[idx].nrg() *= (1.0 - removal_fraction);

                            total_mass += dm;
                            total_momentum += prim_states[idx]->velocity() * dm;

                            // Update accretion stats
                            accretion_stats_.total_mass_accreted += dm;
                            accretion_stats_.total_energy_accreted +=
                                0.5 * dm *
                                prim_states[idx]->velocity().dot(
                                    prim_states[idx]->velocity()
                                );
                            accretion_stats_.total_momentum_accreted +=
                                prim_states[idx]->velocity() * dm;
                        }
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