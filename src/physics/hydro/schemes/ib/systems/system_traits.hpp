#ifndef SYSTEM_TRAITS_HPP
#define SYSTEM_TRAITS_HPP

#include "build_options.hpp"
#include "core/types/containers/vector.hpp"   // for spatial_vector_t
#include "system_config.hpp"                  // for GravitationalConfig
#include <cmath>                              // for M_PI
#include <iostream>
#include <type_traits>   // for std::conditional_t
namespace simbi::ibsystem::traits {
    template <size_type Dims>
    concept AtLeastTwoDimensional = (Dims >= 2);

    template <typename T>
    class GravitationalTrait
    {
      private:
        config::GravitationalConfig<T> config_;

      public:
        GravitationalTrait(const config::GravitationalConfig<T>& config)
            : config_(config)
        {
        }

        DUAL bool use_prescribed_motion() const
        {
            return config_.prescribed_motion;
        }
        DUAL std::string reference_frame() const
        {
            return config_.reference_frame;
        }

        const auto& config() const { return config_; }
    };

    // Binary system behaviors
    template <typename T, size_type Dims>
        requires AtLeastTwoDimensional<Dims>
    class BinaryTrait
    {
      private:
        config::BinaryConfig<T> config_;

      public:
        BinaryTrait(const config::BinaryConfig<T>& config) : config_(config) {}

        // Calculate orbital period
        DUAL T orbital_period() const
        {
            return T(2.0) * M_PI *
                   std::sqrt(
                       std::pow(config_.semi_major, T(3.0)) /
                       (config_.total_mass)
                   );
        }

        // Generate initial positions/velocities
        DUAL std::pair<spatial_vector_t<T, Dims>, spatial_vector_t<T, Dims>>
        initial_positions() const
        {
            T a1 = config_.semi_major / (T(1) + config_.mass_ratio);
            T a2 = config_.semi_major - a1;

            if constexpr (Dims == 2) {
                spatial_vector_t<T, Dims> r1 = {a1, 0.0};
                spatial_vector_t<T, Dims> r2 = {-a2, 0.0};
                return {r1, r2};
            }
            else {
                spatial_vector_t<T, Dims> r1 = {a1, 0.0, 0.0};
                spatial_vector_t<T, Dims> r2 = {-a2, 0.0, 0.0};
                return {r1, r2};
            }
        }

        DUAL std::pair<spatial_vector_t<T, Dims>, spatial_vector_t<T, Dims>>
        initial_velocities() const
        {
            if (config_.circular) {
                const T separation = config_.semi_major;
                const T mu         = config_.total_mass;
                const T phi_dot    = std::sqrt(mu / std::pow(separation, T(3)));
                T a1               = separation / (T(1) + config_.mass_ratio);
                T a2               = separation - a1;

                if constexpr (Dims == 2) {
                    spatial_vector_t<T, Dims> v1 = {0.0, phi_dot * a2};
                    spatial_vector_t<T, Dims> v2 = {0.0, -phi_dot * a1};
                    return {v1, v2};
                }
                else {
                    spatial_vector_t<T, Dims> v1 = {0.0, phi_dot * a2, 0.0};
                    spatial_vector_t<T, Dims> v2 = {0.0, -phi_dot * a1, 0.0};
                    return {v1, v2};
                }
            }
            else {
                // TODO: implement eccentric orbits
                if constexpr (global::on_gpu) {
                    printf("Non-circular orbits not yet implemented\n");
                    return {
                      spatial_vector_t<T, Dims>(),
                      spatial_vector_t<T, Dims>()
                    };
                }
                else {
                    throw std::runtime_error(
                        "Non-circular orbits not yet implemented"
                    );
                }
            }
        }

        // Update positions and velocities at a given time
        DUAL void update_positions_and_velocities(
            T time,
            std::vector<spatial_vector_t<T, Dims>>& positions,
            std::vector<spatial_vector_t<T, Dims>>& velocities
        ) const
        {
            auto [r1, r2] = initial_positions();
            auto [v1, v2] = initial_velocities();

            T phi_dot = std::sqrt(
                config_.total_mass / std::pow(config_.semi_major, T(3))
            );
            T phi = phi_dot * time;

            positions[0]  = vecops::rotate(r1, phi);
            positions[1]  = vecops::rotate(r2, phi);
            velocities[0] = vecops::rotate(v1, phi);
            velocities[1] = vecops::rotate(v2, phi);
        }
    };
}   // namespace simbi::ibsystem::traits
#endif   // SYSTEM_TRAITS_HPP
