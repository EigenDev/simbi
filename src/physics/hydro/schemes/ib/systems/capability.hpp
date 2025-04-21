#ifndef CAPABILITY_HPP
#define CAPABILITY_HPP

#include "build_options.hpp"
#include <cstdint>

namespace simbi::ibsystem {
    enum class BodyCapability : uint32_t {
        NONE          = 0,
        GRAVITATIONAL = 1 << 0,
        ACCRETION     = 1 << 1,
        ELASTIC       = 1 << 2,
        DEFORMABLE    = 1 << 3,
        RIGID         = 1 << 4,
        // TODO: add more capabilities as needed
    };

    DUAL inline BodyCapability operator|(BodyCapability lhs, BodyCapability rhs)
    {
        return static_cast<BodyCapability>(
            static_cast<uint32_t>(lhs) | static_cast<uint32_t>(rhs)
        );
    }

    DUAL inline BodyCapability&
    operator|=(BodyCapability& lhs, BodyCapability rhs)
    {
        lhs = lhs | rhs;
        return lhs;
    }

    DUAL inline bool has_capability(BodyCapability caps, BodyCapability query)
    {
        return (static_cast<uint32_t>(caps) & static_cast<uint32_t>(query)) !=
               0;
    }

    template <typename T>
    struct GravitationalComponent {
        T softening_length;
        bool two_way_coupling;
    };

    template <typename T>
    struct AccretionComponent {
        T accretion_efficiency;
        T accretion_radius;
        T total_accreted_mass;
        T accretion_rate;
    };

    template <typename T>
    struct ElasticComponent {
        T elastic_modulus;
        T poisson_ratio;
    };

    template <typename T>
    struct DeformableComponent {
        T yield_stress;
        T plastic_strain;
    };

    template <typename T>
    struct RigidComponent {
        T interia;
    };
}   // namespace simbi::ibsystem
#endif
