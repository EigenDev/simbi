#ifndef MOTION_POLICIES_HPP
#define MOTION_POLICIES_HPP

#include "build_options.hpp"                  // for DUAL, size_type
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
        DUAL void advance_position(Body& body, const auto dt)
        {
            if (params_.live_motion) {
                body.position_ += body.velocity_ * dt;
            }
        }

        template <typename Body>
        DUAL void advance_velocity(Body& body, const auto dt)
        {
            body.velocity_ += body.force_ / body.mass_ * dt;
        }

        DUAL const Params& params() const { return params_; }
        DUAL Params& params() { return params_; }

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
        DUAL void advance_position(Body& body, const auto dt)
        {
            // do nothing, it's static!
        }

        template <typename Body>
        DUAL void advance_velocity(Body& body, const auto dt)
        {
            // do nothing, it's static!
        }

        DUAL const Params& params() const { return params_; }
        DUAL Params& params() { return params_; }

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
        DUAL void advance_position(Body& body, const auto dt)
        {
            body.position_ += body.velocity_ * dt;
        }

        template <typename Body>
        DUAL void advance_velocity(Body& body, const auto dt)
        {
            body.velocity_ += params_.force / body.mass_ * dt;
        }

        DUAL const Params& params() const { return params_; }
        DUAL Params& params() { return params_; }

      private:
        Params params_;
    };

}   // namespace simbi::ib

#endif
