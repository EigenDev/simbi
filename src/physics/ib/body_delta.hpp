/**
 *=============================================================================
 *           SIMBI - Special Relativistic Magnetohydrodynamics Code
 *=============================================================================
 *
 * @file            body_delta.hpp
 * @brief           body_delta_t class for representing changes in body
 *properties
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

#include "config.hpp"
#include "containers/vector.hpp"   // for vector_t
#include <cassert>
#include <cstdint>

namespace simbi::body {
    template <std::uint64_t Dims>
    struct body_delta_t {
        std::uint64_t idx{0};
        vector_t<real, Dims> force_delta{0};
        vector_t<real, 3> torque_delta{0};
        real mass_delta{0};
        real accretion_rate_delta{0};

        // combine two deltas affecting the same body
        DUAL static body_delta_t
        combine(const body_delta_t& a, const body_delta_t& b)
        {
            assert(a.idx == b.idx);
            return {
              a.idx,
              a.force_delta + b.force_delta,
              a.mass_delta + b.mass_delta,
              b.accretion_rate_delta
            };
        }

        DEV body_delta_t& operator+=(const body_delta_t& other)
        {
            assert(idx == other.idx);
            force_delta += other.force_delta;
            torque_delta += other.torque_delta;
            mass_delta += other.mass_delta;
            accretion_rate_delta += other.accretion_rate_delta;
            return *this;
        }
    };
}   // namespace simbi::body

#endif
