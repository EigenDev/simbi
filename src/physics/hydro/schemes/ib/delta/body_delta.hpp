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
