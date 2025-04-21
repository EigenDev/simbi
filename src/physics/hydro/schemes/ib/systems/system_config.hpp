#ifndef SYSTEM_CONFIG_HPP
#define SYSTEM_CONFIG_HPP

#include "build_options.hpp"
#include "core/types/utility/managed.hpp"

namespace simbi::ibsystem {
    struct SystemConfig : public Managed<global::managed_memory> {
        virtual ~SystemConfig() = default;
    };

    template <typename T>
    struct BinarySystemConfig : public SystemConfig {
        T semi_major;
        T mass_ratio;
        T eccentricity;
        T orbital_period;
        bool circular_orbit;
        bool prescribed_motion;
        std::pair<size_t, size_t> body_indices;

        BinarySystemConfig(
            T semi_major,
            T mass_ratio,
            T eccentricity,
            T orbital_period,
            bool circular_orbit,
            bool prescribed_motion,
            size_t body1_idx,
            size_t body2_idx
        )
            : semi_major(semi_major),
              mass_ratio(mass_ratio),
              eccentricity(eccentricity),
              orbital_period(orbital_period),
              circular_orbit(circular_orbit),
              prescribed_motion(prescribed_motion),
              body_indices(body1_idx, body2_idx)
        {
        }
    };
}   // namespace simbi::ibsystem

#endif   // SYSTEM_CONFIG_HPP
