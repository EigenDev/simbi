#ifndef FMR_HIERARCHY_HPP
#define FMR_HIERARCHY_HPP

#include "containers/vector.hpp"   // for vector_t
#include "level_descriptor.hpp"    // for level_descriptor_t

#include <cstdint>   //   for std::uint64_t

namespace simbi::mesh::fmr {

    template <std::uint64_t Dims>
    struct mesh_hierarchy_t {
        static constexpr std::uint64_t max_levels = 8;

        vector_t<level_descriptor_t<Dims>, max_levels> levels;
        std::uint64_t num_levels{0};

        // accessors
        const level_descriptor_t<Dims>& operator[](std::uint64_t id) const
        {
            return levels[id];
        }

        level_descriptor_t<Dims>& operator[](std::uint64_t id)
        {
            return levels[id];
        }

        // queries
        constexpr bool has_refinement() const { return num_levels > 1; }
        constexpr std::uint64_t finest_level() const { return num_levels - 1; }
    };

}   // namespace simbi::mesh::fmr

#endif
