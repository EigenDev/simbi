/**
 *  *=============================================================================
 *  *           SIMBI - Special Relativistic Magnetohydrodynamics Code
 *  *=============================================================================
 *  *
 *  * @file            gravitational.hpp
 *  * @brief
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
                  grav_strength_(grav_strength),
                  softening_length_(softening * radius)
            {
            }

            // N-body methods
            DUAL void update_position(const T dt)
            {
                // first half-kick
                this->velocity_ += 0.5 * dt * this->force_ / this->mass_;
                // drift
                this->position_ += dt * this->velocity_;
                // second half-kick after force computation
                compute_body_forces(this->mesh_.bodies_);
                this->velocity_ += 0.5 * dt * this->force_ / this->mass_;
            }

            DUAL auto compute_body_forces(const auto& bodies)
            {
                this->force_ = spatial_vector_t<T, Dims>();
                for (const auto& other : bodies) {
                    if (other.get() != this) {
                        const auto r = other->position_ - this->position_;
                        const auto r2 =
                            r.dot(r) + softening_length_ * softening_length_;
                        this->force_ += grav_strength_ * other->mass_ * r /
                                        (r2 * std::sqrt(r2));
                    }
                }
            }
        };

    }   // namespace ib

}   // namespace simbi

#endif