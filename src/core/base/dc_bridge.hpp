#ifndef SIMBI_CORE_BASE_DOMAIN_COORDINATE_BRIDGE_HPP
#define SIMBI_CORE_BASE_DOMAIN_COORDINATE_BRIDGE_HPP

#include "coordinate.hpp"
#include "domain.hpp"
#include <array>
#include <cstddef>

namespace simbi::base {
    // Extended domain with origin for grid operations
    template <std::uint64_t Dims>
    class domain_with_origin_t
    {
      private:
        base::domain_t<Dims> base_domain_;
        base::coordinate_t<Dims> origin_;

      public:
        // Constructor
        domain_with_origin_t(
            const base::domain_t<Dims>& domain,
            const base::coordinate_t<Dims>& origin
        )
            : base_domain_(domain), origin_(origin)
        {
        }

        // Convenience constructors
        domain_with_origin_t(
            const std::array<std::uint64_t, Dims>& extents,
            const base::coordinate_t<Dims>& origin
        )
            : base_domain_(extents), origin_(origin)
        {
        }

        // Check if global coordinate is within domain
        bool contains_global(const base::coordinate_t<Dims>& global_coord) const
        {
            for (std::uint64_t i = 0; i < Dims; ++i) {
                if (global_coord[i] < origin_[i] ||
                    global_coord[i] >= origin_[i] + static_cast<std::int64_t>(
                                                        base_domain_.extents[i]
                                                    )) {
                    return false;
                }
            }
            return true;
        }

        // Convert global coordinate to local index
        std::uint64_t
        global_to_index(const base::coordinate_t<Dims>& global_coord) const
        {
            auto local_coord = global_to_local(global_coord);
            return base_domain_.linear_index(local_coord.to_domain_point());
        }

        // Convert global to local coordinate
        base::coordinate_t<Dims>
        global_to_local(const base::coordinate_t<Dims>& global_coord) const
        {
            base::coordinate_t<Dims> local;
            for (std::uint64_t i = 0; i < Dims; ++i) {
                local[i] = global_coord[i] - origin_[i];
            }
            return local;
        }

        // Convert local to global coordinate
        base::coordinate_t<Dims>
        local_to_global(const base::coordinate_t<Dims>& local_coord) const
        {
            base::coordinate_t<Dims> global;
            for (std::uint64_t i = 0; i < Dims; ++i) {
                global[i] = local_coord[i] + origin_[i];
            }
            return global;
        }

        // Accessors
        const base::domain_t<Dims>& base_domain() const { return base_domain_; }
        const base::coordinate_t<Dims>& origin() const { return origin_; }

        // Delegate common operations to base domain
        std::uint64_t total_size() const { return base_domain_.total_size(); }
        std::uint64_t extent(std::uint64_t dim) const
        {
            return base_domain_.extent(dim);
        }
    };
}   // namespace simbi::base

#endif   // SIMBI_CORE_BASE_DOMAIN_COORDINATE_BRIDGE_HPP
