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

namespace simbi {
    namespace ibsystem {
        // concrete orbital n-body system implementation
        // that tracks orbital parameters like energy, angular momentum, etc.
        template <typename T, size_type Dims, typename MeshType>
        class OrbitalSystem
        {
          private:
            ndarray<
                util::smart_ptr<ib::GravitationalBody<T, Dims, MeshType>>,
                1>
                bodies_;
            spatial_vector_t<T, Dims> com_;
            spatial_vector_t<T, Dims> angular_momentum_;
            T total_energy_;
            T total_mass_;
            MeshType mesh_;

          public:
            DUAL OrbitalSystem(const MeshType& mesh) : mesh_(mesh) {}

            // method to add a gravitational body to the system
            DUAL void add_body(
                const spatial_vector_t<T, Dims>& position,
                const spatial_vector_t<T, Dims>& velocity,
                const T mass,
                const T radius,
                const T grav_strength,
                const T softening = 0.01   // Small fraction of radius
            )
            {
                auto body =
                    util::make_unique<ib::GravitationalBody<T, Dims, MeshType>>(
                        mesh_,
                        position,
                        velocity,
                        mass,
                        radius,
                        grav_strength,
                        softening
                    );
                bodies_.push_back(body);
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
                    mesh_,
                    position,
                    velocity,
                    mass,
                    radius,
                    grav_strength,
                    softening,
                    accretion_efficiency
                );
                bodies_.push_back(sink);
            }

            DUAL void update_system(auto& prim_states, const T dt)
            {
                compute_center_of_mass();
                // update bodies and sinks
                for (auto& body : bodies_) {
                    body->compute_body_forces(bodies_);
                    body->update_position(dt);

                    // Handle accretion if body is a sink
                    if (auto* sink = dynamic_cast<
                            ib::GravitationalSinkParticle<T, Dims, MeshType>*>(
                            body.get()
                        )) {
                        sink->accrete(prim_states);
                    }
                }

                update_conserved_quantities();
            }

          private:
            DUAL void compute_center_of_mass()
            {
                com_        = spatial_vector_t<T, Dims>();
                total_mass_ = 0;

                for (const auto& body : bodies_) {
                    com_ += body->position() * body->mass();
                    total_mass_ += body->mass();
                }
                com_ /= total_mass_;
            }

            DUAL void update_conserved_quantities()
            {
                total_energy_     = 0;
                angular_momentum_ = spatial_vector_t<T, Dims>();

                for (const auto& body : bodies_) {
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