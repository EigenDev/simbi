#ifndef ECS_GEOMETRY_VISITOR_HPP
#define ECS_GEOMETRY_VISITOR_HPP

// =============================================================================
// geometry_visitor.hpp
//
// visitor pattern for building block geometry from mesh config.
// handles log/uniform coordinate map variants at runtime.
// =============================================================================

#include "build_config.hpp"
#include "geometry/api.hpp"
#include "geometry/block_geometry.hpp"
#include "geometry/coordinate_map.hpp"
#include "geometry/metrics.hpp"
#include "grid/mesh_config.hpp"
#include "utility/enums.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>

namespace simbi::ecs {

    // =========================================================================
    // with_block_geometry
    //
    // builds block geometry from mesh config and invokes callback.
    // G is known at compile time from the simulation template.
    // uses visitor pattern to handle log/uniform map variants.
    //
    // usage:
    //   with_block_geometry<G>(mesh_cfg, motion, [&](const auto& geo) {
    //       auto h = geo.scale_factors(coord);
    //   });
    // =========================================================================
    template <geometry_t G, std::uint64_t Rank, typename Func>
    decltype(auto) with_block_geometry(
        const grid::mesh_config_t<Rank>& mesh_cfg,
        const geometry::motion_state_t&  motion,
        Func&&                           func
    )
    {
        constexpr auto x1c     = Rank - 1;
        constexpr auto x2c     = Rank - 2;
        constexpr auto x3c     = Rank - 3;
        const auto&    geo_cfg = mesh_cfg.geometry;

        // helper to create uniform map for a dimension
        auto make_uniform = [&](std::size_t dim) {
            const auto& dcfg = geo_cfg.dims[dim];
            real        len  = dcfg.end - dcfg.start;
            real        dx   = len / static_cast<real>(mesh_cfg.global_cells[dim]);
            return geometry::uniform_map_t(dcfg.start, dx);
        };

        // helper to create log map for a dimension
        auto make_log = [&](std::size_t dim) {
            const auto& dcfg = geo_cfg.dims[dim];
            real        log_slope =
                std::log10(dcfg.end / dcfg.start) / static_cast<real>(mesh_cfg.global_cells[dim]);
            return geometry::log_map_t(dcfg.start, log_slope);
        };

        // check if radial uses log spacing
        const bool radial_log =
            !geo_cfg.dims.empty() && geo_cfg.dims[x1c].type == geometry::map_type_t::log;

        if constexpr (G == geometry_t::CARTESIAN) {
            // cartesian always uses uniform maps
            if constexpr (Rank == 1) {
                auto metric = geometry::cartesian_metric_t(make_uniform(x1c));
                return func(geometry::block_geometry(metric, motion));
            }
            else if constexpr (Rank == 2) {
                auto metric = geometry::cartesian_metric_t(make_uniform(x1c), make_uniform(x2c));
                return func(geometry::block_geometry(metric, motion));
            }
            else {
                auto metric = geometry::cartesian_metric_t(
                    make_uniform(x1c),
                    make_uniform(x2c),
                    make_uniform(x3c)
                );
                return func(geometry::block_geometry(metric, motion));
            }
        }
        else if constexpr (G == geometry_t::SPHERICAL) {
            // spherical: radial may be log, angular always uniform
            if constexpr (Rank == 1) {
                if (radial_log) {
                    auto metric = geometry::spherical_metric_t(make_log(x1c));
                    return func(geometry::block_geometry(metric, motion));
                }
                else {
                    auto metric = geometry::spherical_metric_t(make_uniform(x1c));
                    return func(geometry::block_geometry(metric, motion));
                }
            }
            else if constexpr (Rank == 2) {
                if (radial_log) {
                    auto metric = geometry::spherical_metric_t(make_log(x1c), make_uniform(x2c));
                    return func(geometry::block_geometry(metric, motion));
                }
                else {
                    auto metric =
                        geometry::spherical_metric_t(make_uniform(x1c), make_uniform(x2c));
                    return func(geometry::block_geometry(metric, motion));
                }
            }
            else {
                if (radial_log) {
                    auto metric = geometry::spherical_metric_t(
                        make_log(x1c),
                        make_uniform(x2c),
                        make_uniform(x3c)
                    );
                    return func(geometry::block_geometry(metric, motion));
                }
                else {
                    auto metric = geometry::spherical_metric_t(
                        make_uniform(x1c),
                        make_uniform(x2c),
                        make_uniform(x3c)
                    );
                    return func(geometry::block_geometry(metric, motion));
                }
            }
        }
        else {
            // cylindrical: radial may be log, angular/axial always uniform
            if constexpr (Rank == 1) {
                if (radial_log) {
                    auto metric = geometry::cylindrical_metric_t(make_log(x1c));
                    return func(geometry::block_geometry(metric, motion));
                }
                else {
                    auto metric = geometry::cylindrical_metric_t(make_uniform(x1c));
                    return func(geometry::block_geometry(metric, motion));
                }
            }
            else if constexpr (Rank == 2) {
                if (radial_log) {
                    auto metric = geometry::cylindrical_metric_t(make_log(x1c), make_uniform(x2c));
                    return func(geometry::block_geometry(metric, motion));
                }
                else {
                    auto metric =
                        geometry::cylindrical_metric_t(make_uniform(x1c), make_uniform(x2c));
                    return func(geometry::block_geometry(metric, motion));
                }
            }
            else {
                if (radial_log) {
                    auto metric = geometry::cylindrical_metric_t(
                        make_log(x1c),
                        make_uniform(x2c),
                        make_uniform(x3c)
                    );
                    return func(geometry::block_geometry(metric, motion));
                }
                else {
                    auto metric = geometry::cylindrical_metric_t(
                        make_uniform(x1c),
                        make_uniform(x2c),
                        make_uniform(x3c)
                    );
                    return func(geometry::block_geometry(metric, motion));
                }
            }
        }
    }

} // namespace simbi::ecs

#endif // ECS_GEOMETRY_VISITOR_HPP
