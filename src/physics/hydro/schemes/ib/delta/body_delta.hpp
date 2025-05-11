/**
 *=============================================================================
 *           SIMBI - Special Relativistic Magnetohydrodynamics Code
 *=============================================================================
 *
 * @file            body_delta.hpp
 * @brief           BodyDelta class for representing changes in body properties
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
#ifndef BODY_DELTA_HPP
#define BODY_DELTA_HPP

#include "build_options.hpp"
#include "core/types/containers/vector.hpp"   // for spatial_vector_t

namespace simbi::ibsystem {
    template <typename T, size_type Dims>
    struct BodyDelta {
        size_t body_idx;
        spatial_vector_t<T, Dims> force_delta;
        T mass_delta;
        T accreted_mass_delta;
        T accretion_rate_delta;

        // combine two deltas affecting the same body
        DUAL static BodyDelta combine(const BodyDelta& a, const BodyDelta& b)
        {
            assert(a.body_idx == b.body_idx);
            return {
              a.body_idx,
              a.force_delta + b.force_delta,
              a.mass_delta + b.mass_delta,
              a.accreted_mass_delta + b.accreted_mass_delta,
              b.accretion_rate_delta
            };
        }
    };
}   // namespace simbi::ibsystem

#endif
