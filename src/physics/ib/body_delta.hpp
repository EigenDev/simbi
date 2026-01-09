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

#include "compat.hpp"
#include "containers/vector.hpp" // for vector_t

#include <cassert>
#include <cstdint>

namespace simbi::body {
    template <std::uint64_t Rank>
    struct body_delta_t
    {
        std::uint64_t idx{0};

        // instantaneous quantities (last timestep value)
        vector_t<real, Rank> force_delta{0};  // [force] instantaneous total force
        vector_t<real, 3>    torque_delta{0}; // [torque] instantaneous total torque

        // accumulated quantities (summed across timesteps)
        real mass_delta{0}; // [mass] total accreted mass
        real prev_mass_delta{0};

        // spatial accumulation: sums all cell contributions within one timestep
        DUAL body_delta_t& operator+=(const body_delta_t& other)
        {
            assert(idx == other.idx);
            force_delta += other.force_delta;   // sum spatially
            torque_delta += other.torque_delta; // sum spatially
            mass_delta += other.mass_delta;     // sum spatially AND temporally
            return *this;
        }

        // temporal update: called after each timestep consolidation
        // preserves accumulated mass, replaces instantaneous force/torque
        DUAL void update_for_new_timestep(const body_delta_t& timestep_totals)
        {
            assert(idx == timestep_totals.idx);
            force_delta  = timestep_totals.force_delta;  // replace with latest
            torque_delta = timestep_totals.torque_delta; // replace with latest
            mass_delta += timestep_totals.mass_delta;    // accumulate
        }
    };
} // namespace simbi::body

#endif
