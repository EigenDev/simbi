/**
 *  *=============================================================================
 *  *           SIMBI - Special Relativistic Magnetohydrodynamics Code
 *  *=============================================================================
 *  *
 *  * @file            orbital.hpp
 *  * @brief           Orbital Immersed Body System Implementation
 *  * @details
 *  *
 *  * @version         0.8.0
 *  * @date            2025-02-17
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
 *  * 2025-02-17      v0.8.0      Initial implementation
 *  *
 *  *==============================================================================
 *  * @copyright (C) 2025 Marcus DuPont. All rights reserved.
 *  *==============================================================================
 */
#ifndef ORBITAL_HPP
#define ORBITAL_HPP

#include "build_options.hpp"
#include "core/types/containers/ndarray.hpp"
#include "physics/hydro/schemes/ib/bodies/gravitational.hpp"
#include "physics/hydro/schemes/ib/bodies/sink.hpp"
#include "physics/hydro/schemes/ib/systems/body_system.hpp"

namespace simbi {
    namespace ibsystem {
        // concrete orbital n-body system implementation
        // that tracks orbital parameters like energy, angular momentum, etc.
        template <typename T, size_type Dims, typename MeshType>
        class OrbitalSystem : public BodySystem<T, Dims, MeshType>
        {
            // dims < 3D, we work with j-scalar. Otherwise, it's the full
            // vector.
            using ang_type =
                std::conditional_t < Dims<3, T, spatial_vector_t<T, Dims>>;

          private:
            spatial_vector_t<T, Dims> com_;
            ang_type angular_momentum_;
            T total_energy_;
            T total_mass_;
            T gamma_;

          protected:
            T initial_energy_;
            ang_type initial_angular_momentum_;

            DUAL void check_conservation_errors()
            {
                // Check relative energy error
                T energy_error = std::abs(
                    (total_energy_ - initial_energy_) / initial_energy_
                );

                // Check angular momentum conservation
                T ang_mom_error = [&]() {
                    if constexpr (Dims == 3) {
                        return (angular_momentum_ - initial_angular_momentum_)
                                   .norm() /
                               initial_angular_momentum_.norm();
                    }
                    else {
                        return (angular_momentum_ - initial_angular_momentum_) /
                               initial_angular_momentum_;
                    }
                }();

                if (energy_error > 1e-6 || ang_mom_error > 1e-6) {
                    // Log warning or take corrective action
                    // Could add a correction step here
                }
            }

          public:
            // using BodySystem<T, Dims, MeshType>::BodySystem;
            DUAL OrbitalSystem(const MeshType& mesh, const real gamma)
                : BodySystem<T, Dims, MeshType>(mesh), gamma_(gamma)
            {
            }

            // method to add a gravitational sink to the system
            DUAL void add_sink(
                const spatial_vector_t<T, Dims>& position,
                const spatial_vector_t<T, Dims>& velocity,
                const T mass,
                const T radius,
                const T grav_strength,
                const T softening = 0.01,   // Small fraction of radius,
                const T accretion_efficiency = 0.1
            )
            {
                auto sink = util::make_unique<
                    ib::GravitationalSinkParticle<T, Dims, MeshType>>(
                    this->mesh_,
                    position,
                    velocity,
                    mass,
                    radius,
                    grav_strength,
                    softening,
                    accretion_efficiency
                );
                this->bodies_.push_back(sink);
            }

            DUAL void update_system(auto& cons_state, const T dt)
            {
                // First half kick for all bodies
                for (auto& body : this->bodies_) {
                    body->advance_velocity(0.5 * dt);
                }

                // Full drift
                for (auto& body : this->bodies_) {
                    body->advance_position(dt);
                }

                // Recompute forces
                for (auto& body : this->bodies_) {
                    body->compute_body_forces(this->bodies_);
                }

                for (auto& body : this->bodies_) {
                    body->spread_boundary_forces(cons_state, dt);
                }

                for (auto& body : this->bodies_) {
                    body->apply_body_forces(cons_state, this->bodies_, dt);
                }

                // Second half kick
                for (auto& body : this->bodies_) {
                    body->advance_velocity(0.5 * dt);
                }

                // Handle accretion if needed
                // for (auto& body : this->bodies_) {
                //     if (auto* sink = dynamic_cast<
                //             ib::GravitationalSinkParticle<T, Dims,
                //             MeshType>*>( body.get()
                //         )) {
                //         sink->accrete(cons_state, gamma_);
                //     }
                // }

                // Update diagnostics
                compute_center_of_mass();
                update_conserved_quantities();
                check_conservation_errors();
            }

            DUAL void init_system()
            {
                // Initial force computation for all bodies
                for (auto& body : this->bodies_) {
                    body->compute_body_forces(this->bodies_);
                }

                // Initialize conserved quantities
                compute_center_of_mass();
                update_conserved_quantities();

                // Store initial values for energy tracking
                initial_energy_           = total_energy_;
                initial_angular_momentum_ = angular_momentum_;
            }

            DUAL void advance_velocities(const T dt)
            {
                for (auto& body : this->bodies_) {
                    body->advance_velocity(dt);
                }
            }
            DUAL void advance_positions(const T dt)
            {
                for (auto& body : this->bodies_) {
                    body->advance_position(dt);
                }
            }

            DUAL void compute_and_apply_forces(
                auto& cons_state,
                const auto& prim_state,
                const T dt
            )
            {
                for (auto& body : this->bodies_) {
                    body->compute_body_forces(this->bodies_);
                }

                for (auto& body : this->bodies_) {
                    body->apply_body_forces(
                        cons_state,
                        prim_state,
                        this->bodies_,
                        dt
                    );
                }
            }

          private:
            DUAL void compute_center_of_mass()
            {
                com_        = spatial_vector_t<T, Dims>();
                total_mass_ = 0;

                for (const auto& body : this->bodies_) {
                    com_ += body->position() * body->mass();
                    total_mass_ += body->mass();
                }
                com_ /= total_mass_;
            }

            DUAL void update_conserved_quantities()
            {
                total_energy_     = 0;
                angular_momentum_ = ang_type();

                for (const auto& body : this->bodies_) {
                    const auto v = body->velocity();
                    const auto r = body->position() - com_;

                    total_energy_ += 0.5 * body->mass() * v.dot(v);
                    angular_momentum_ += body->mass() * r.cross(v);
                }
            }
        };
    }   // namespace ibsystem

}   // namespace simbi

#endif