/**
 *=============================================================================
 *           SIMBI - Special Relativistic Magnetohydrodynamics Code
 *=============================================================================
 *
 * @file            motion_policies.hpp
 * @brief
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

#ifndef MOTION_POLICIES_HPP
#define MOTION_POLICIES_HPP

#include "build_options.hpp"                  // for , size_type
#include "core/types/containers/vector.hpp"   // for spatial_vector_t

namespace simbi::ib {
    //----------------------------------------------------------------------------
    // Dynamic Motion
    // --------------------------------------------------------------------------a-
    template <typename T, size_t Dims>
    class DynamicMotionPolicy
    {
      public:
        struct Params {
            bool live_motion = false;
        };

        DynamicMotionPolicy(const Params& params = {}) : params_(params) {}

        template <typename Body>
        DEV void advance_position(Body& body, const auto dt)
        {
            if (params_.live_motion) {
                body.position_ += body.velocity_ * dt;
            }
        }

        template <typename Body>
        DEV void advance_velocity(Body& body, const auto dt)
        {
            body.velocity_ += body.force_ / body.mass_ * dt;
        }

        const Params& params() const { return params_; }
        Params& params() { return params_; }

      private:
        Params params_;
    };

    //----------------------------------------------------------------------------
    // Static Motion
    // ---------------------------------------------------------------------------
    template <typename T, size_t Dims>
    class StaticMotionPolicy
    {
      public:
        struct Params {
        };

        StaticMotionPolicy(const Params& params = {}) : params_(params) {}

        template <typename Body>
        DEV void advance_position(Body& body, const auto dt)
        {
            // do nothing, it's static!
        }

        template <typename Body>
        DEV void advance_velocity(Body& body, const auto dt)
        {
            // do nothing, it's static!
        }

        DEV const Params& params() const { return params_; }
        DEV Params& params() { return params_; }

      private:
        Params params_;
    };

    template <typename T, size_type Dims>
    class PrescribedMotionPolicy
    {
      public:
        struct Params {
            spatial_vector_t<T, Dims> force = spatial_vector_t<T, Dims>();
        };

        PrescribedMotionPolicy(const Params& params = {}) : params_(params) {}

        template <typename Body>
        DEV void advance_position(Body& body, const auto dt)
        {
            body.position_ += body.velocity_ * dt;
        }

        template <typename Body>
        DEV void advance_velocity(Body& body, const auto dt)
        {
            body.velocity_ += params_.force / body.mass_ * dt;
        }

        DEV const Params& params() const { return params_; }
        DEV Params& params() { return params_; }

      private:
        Params params_;
    };

}   // namespace simbi::ib

#endif
