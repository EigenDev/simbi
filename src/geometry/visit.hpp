#ifndef GRID_GEOMETRY_VISIT_HPP
#define GRID_GEOMETRY_VISIT_HPP

#include "grid/patch_id.hpp"

#include <cstdint>
#include <variant>

namespace simbi::geometry {

    // helper to unpack 1, 2, or 3 map variants and invoke a visitor
    // this generates the combinatorial explosion of kernels (uniform-uniform,
    // uniform-log, etc) so the compiler can optimize each one perfectly.
    template <std::uint64_t Rank, typename GeoService, typename Visitor>
    void
    visit_block_geometry(const GeoService& service, const grid::patch_id_t& id, Visitor&& visitor)
    {
        // create the variants for each dimension
        auto map0 = service.create_map(0, id.coords[0], id.level);

        if constexpr (Rank == 1) {
            std::visit([&](auto&& m0) { visitor(m0); }, map0);
        }
        else if constexpr (Rank == 2) {
            auto map1 = service.create_map(1, id.coords[1], id.level);
            std::visit([&](auto&& m0, auto&& m1) { visitor(m0, m1); }, map0, map1);
        }
        else {
            auto map1 = service.create_map(1, id.coords[1], id.level);
            auto map2 = service.create_map(2, id.coords[2], id.level);
            std::visit(
                [&](auto&& m0, auto&& m1, auto&& m2) { visitor(m0, m1, m2); },
                map0,
                map1,
                map2
            );
        }
    }

} // namespace simbi::geometry

#endif
