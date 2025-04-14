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

#ifndef FORCE_POLICIES_HPP
#define FORCE_POLICIES_HPP

#include "../body_traits.hpp"                 // for traits::Gravitational
#include "build_options.hpp"                  // for , size_type
#include "core/types/containers/vector.hpp"   // for spatial_vectir_t
#include <cmath>                              // for std::sqrt

namespace simbi::ib {
    // -----------------------------------------------------------------------
    // GravitationalForcePolicy
    // -----------------------------------------------------------------------
    template <typename T, size_type Dims>
    class GravitationalForcePolicy
    {
        using trait_t = traits::Gravitational<T>;

      public:
        using Params = typename trait_t::Params;

        GravitationalForcePolicy(const Params& params = {}) : trait_(params) {}

        DEV const trait_t& gravitational_trait() const { return trait_; }
        DEV trait_t& gravitational_trait() { return trait_; }
        DEV T softening_length() const { return trait_.softening_length(); }

        // Calculate gravitational forces
        template <typename Body>
        DEV void
        calculate_forces(Body& body, const auto& other_bodies, const T dt)
        {
            body.force_ = spatial_vector_t<T, Dims>();

            for (const auto& other_ref : other_bodies) {
                const auto& other = other_ref.get();
                if (&other != &body) {
                    // Check if other body has gravitational trait
                    if (other.has_gravitational_capability()) {

                        const auto r = other.position() - body.position();
                        const auto r2 =
                            r.dot(r) + trait_.softening_length() *
                                           trait_.softening_length();
                        body.force_ += -other.mass() * body.mass() * r /
                                       (r2 * std::sqrt(r2));
                    }
                }
            }
        }

      private:
        trait_t trait_;

        // trait detection helpers
        template <typename OtherBody>
        struct has_gravitational_trait {
            template <typename U>
            static std::true_type test(typename U::Gravitational*);

            template <typename U>
            static std::false_type test(...);

            static constexpr bool value = decltype(test<T>(0))::value;
        };

        template <typename Body>
        DEV spatial_vector_t<T, Dims> calculate_prescribed_force(
            const Body& body,
            const auto& other_bodies
        ) const
        {
            // Calculate center of mass
            spatial_vector_t<T, Dims> com;
            T total_mass = 0;

            for (const auto& other_ref : other_bodies) {
                const auto& b = other_ref.get();
                com += b.position() * b.mass();
                total_mass += b.mass();
            }

            if (total_mass > 0) {
                com /= total_mass;
            }

            // Calculate orbital forces
            const auto r     = body.position() - com;
            const auto r_mag = r.norm();
            if (r_mag > 0) {
                const auto v     = body.velocity();
                const auto v_mag = v.norm();

                // Centripetal force
                const auto a_centripetal = (v_mag * v_mag) / r_mag;
                return -body.mass() * a_centripetal * (r / r_mag);
            }

            return spatial_vector_t<T, Dims>();
        }
    };

    //-------------------------------------------------------------------------
    // NullForcePolicy
    // -----------------------------------------------------------------------
    template <typename T, size_type Dims>
    class NullForcePolicy
    {
      public:
        using trait_t = traits::Rigid<T>;
        NullForcePolicy(const trait_t& trait) : trait_(trait) {}

        template <typename Body>
        DEV void
        calculate_forces(Body& body, const auto& other_bodies, const T dt)
        {
            // do nothing
        }

      private:
        trait_t trait_;
    };

}   // namespace simbi::ib

#endif
