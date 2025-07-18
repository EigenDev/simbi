/**
 *=============================================================================
 *           SIMBI - Special Relativistic Magnetohydrodynamics Code
 *=============================================================================
 *
 * @file            capability.hpp
 * @brief           Body capabilities for the IB scheme
 * @details
 *
 * @version         0.8.0
 * @date            2025-05-11
 * @author          Marcus DuPont
 * @email           marcus.dupont@princeton.edu
 *
 *==============================================================================
 * @build           Requirements & Dependencies
 *==============================================================================
 * @requires        C++20
 * @depends         CUDA >= 11.0, HDF5 >= 1.12, OpenMP >= 4.5
 * @platform        Linux, MacOS
 * @parallel        GPU (CUDA, HIP), CPU (OpenMP)
 *
 *==============================================================================
 * @documentation   Reference & Notes
 *==============================================================================
 * @usage
 * @note
 * @warning
 * @todo
 * @bug
 * @performance
 *
 *==============================================================================
 * @testing        Quality Assurance
 *==============================================================================
 * @test
 * @benchmark
 * @validation
 *
 *==============================================================================
 * @history        Version History
 *==============================================================================
 * 2025-05-11      v0.8.0      Initial implementation
 *
 *==============================================================================
 * @copyright (C) 2025 Marcus DuPont. All rights reserved.
 *==============================================================================
 */
#ifndef CAPABILITY_HPP
#define CAPABILITY_HPP

#include "config.hpp"
#include "core/utility/enums.hpp"
#include <cstdint>

namespace simbi::ibsystem {
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

    struct grav_component_t {
        real softening_length;
    };

    struct accretion_component_t {
        real accretion_efficiency;
        real accretion_radius;
        real total_accreted_mass;
        real accretion_rate;
    };

    struct elastic_component_t {
        real elastic_modulus;
        real poisson_ratio;
    };

    struct deformable_component_t {
        real yield_stress;
        real plastic_strain;
    };

    struct rigid_component_t {
        real inertia;
        bool apply_no_slip;
    };
}   // namespace simbi::ibsystem
#endif
