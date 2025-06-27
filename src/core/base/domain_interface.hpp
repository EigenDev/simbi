#ifndef SIMBI_CORE_GRAPH_DOMAIN_INTERFACE_HPP
#define SIMBI_CORE_GRAPH_DOMAIN_INTERFACE_HPP

#include "core/base/concepts.hpp"
#include "core/base/coordinate.hpp"
#include "core/base/stencil.hpp"
#include "core/utility/enums.hpp"
#include <cstddef>
#include <vector>

namespace simbi::base {
    using namespace simbi::concepts;
    // abstract interface for data access patterns
    template <typename T, std::uint64_t Dims, std::uint64_t Order>
        requires valid_dimension<Dims>
    class domain_interface_t
    {
      public:
        virtual ~domain_interface_t() = default;

        // core data access
        virtual T get(const coordinate_t<Dims>& coord) const              = 0;
        virtual void set(const coordinate_t<Dims>& coord, const T& value) = 0;

        // stencil-based access
        virtual vector_t<T, 2 * Order - 1)> gather_stencil(
            const coordinate_t<Dims>& center,
            const stencil_t<T, Order>& pattern
        ) const = 0;

        // functional operations
        template <typename func_t>
        auto
        map(const std::vector<coordinate_t<Dims>>& coords, func_t&& f) const
        {
            using result_type = decltype(f(get(coords[0])));
            std::vector<result_type> result;
            result.reserve(coords.size());

            for (const auto& coord : coords) {
                result.push_back(f(get(coord)));
            }

            return result;
        }

        template <typename predicate_t>
        std::vector<coordinate_t<Dims>> filter_coordinates(
            const std::vector<coordinate_t<Dims>>& coords,
            predicate_t&& pred
        ) const
        {
            std::vector<coordinate_t<Dims>> result;

            for (const auto& coord : coords) {
                if (pred(get(coord))) {
                    result.push_back(coord);
                }
            }

            return result;
        }
    };
}   // namespace simbi::base
#endif   // SIMBI_CORE_GRAPH_DOMAIN_INTERFACE_HPP
