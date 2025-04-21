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
        DUAL BodyDelta<T, Dims> combine(const BodyDelta<T, Dims>& other) const
        {
            assert(body_idx == other.body_idx);
            BodyDelta<T, Dims> result = *this;
            result.force_delta += other.force_delta;
            result.mass_delta += other.mass_delta;
            result.accreted_mass_delta += other.accreted_mass_delta;
            result.accretion_rate_delta += other.accretion_rate_delta;
            return result;
        }
    };
}   // namespace simbi::ibsystem

#endif
