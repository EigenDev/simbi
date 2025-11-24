#ifndef GRID_BOUNDARY_DRIVER_HPP
#define GRID_BOUNDARY_DRIVER_HPP

#include "compat.hpp"
#include "containers/vector.hpp"
#include "geometry/boundary/index_map.hpp"
#include "geometry/visit.hpp"
#include "grid/boundary.hpp"
#include "grid/connectivity.hpp"
#include "grid/domain.hpp"
#include "grid/field.hpp"
#include "grid/mesh_config.hpp"
#include "grid/patch_id.hpp"
#include "grid/skeleton.hpp"
#include "index_map.hpp"

#include <cstdint>

namespace simbi::geometry {

    // for simple runs without dynamic boundaries
    struct simple_context_t {
        using metric_type                 = void;   // not needed
        static constexpr bool has_dynamic = false;
    };

    // helper tag for CTAD
    // usage: dynamic_context_t ctx(use_metric<my_metric>, geo, vm, time);
    template <typename Metric>
    struct use_metric_t {
    };

    template <typename Metric>
    constexpr use_metric_t<Metric> use_metric{};

    // for full physics runs
    template <typename Metric, typename GeoService, typename VM>
    struct dynamic_context_t {
        using metric_type = Metric;   // e.g., spherical_metric_t
        using vm_type     = VM;
        static constexpr bool has_dynamic = true;

        const GeoService& geo_service;
        const VM& vm;
        real time;

        // constructor required for CTAD to strip the tag
        constexpr dynamic_context_t(
            use_metric_t<Metric>,
            const GeoService& g,
            const VM& v,
            const real t
        )
            : geo_service(g), vm(v), time(t)
        {
        }
    };

    template <typename Metric, typename GeoService, typename VM>
    dynamic_context_t(use_metric_t<Metric>, const GeoService&, const VM&, real)
        -> dynamic_context_t<Metric, GeoService, VM>;

    template <typename T>
    concept is_dynamic_context_c = requires {
        typename T::metric_type;
        requires T::has_dynamic == true;
    };

    // -------------------------------------------------------------------------
    // dynamic boundary operator
    // bridges the gap between grid indices (i,j,k) and the physics VM (x,y,z,t)
    // -------------------------------------------------------------------------
    template <
        typename T,
        typename Blockgeometry_t,
        typename Expression,
        std::uint64_t Rank>
    struct dynamic_boundary_op_t {
        // geometry: converts index -> physical centroid
        Blockgeometry_t geometry;

        // vm: the compiled dag expression (handle to managed memory)
        Expression vm;

        // simulation time
        real time;

        // ---------------------------------------------------------------------
        // operator()
        // signature matches enum_map: (coordinate, value) -> new_value
        // ---------------------------------------------------------------------
        // idx: the ghost cell index (provided by enum_map)
        // interior_state: the state at the nearest active edge (provided by
        // remap)
        DUAL T
        operator()(const iarray<Rank>& idx, const T& interior_state) const
        {
            // 1. geometry: convert index to physical position
            auto phys_pos = geometry.centroid(idx);

            // 2. physics: execute the vm
            // apply(position, interior_state, time)
            return vm.apply(phys_pos, interior_state, time);
        }
    };

    struct boundary_driver_t {

        // ---------------------------------------------------------------------
        // apply_boundaries
        // ---------------------------------------------------------------------
        template <
            typename T,
            std::uint64_t Rank,
            typename Context,
            typename Policy,
            typename Exec>
        static void apply_boundaries(
            grid::field_t<T, Rank>& field,
            const grid::patch_id_t& id,
            const grid::skeleton_t<Rank>& skeleton,
            const grid::mesh_config_t<Rank>& config,
            const Policy& physics_policy,
            const Context& context,
            Exec& exec
        )
        {
            using namespace grid::domain_algebra;

            const auto* block_info = skeleton.get_block(id);
            if (!block_info) {
                return;
            }

            const auto& geometry    = block_info->geometry;
            const auto& global_dims = config.global_cells;

            // -----------------------------------------------------------------
            // dimensional cascade: x -> y -> z
            // -----------------------------------------------------------------
            for (std::uint64_t dd = 0; dd < Rank; ++dd) {

                // define working extent (cascade logic)
                grid::domain_t<Rank> transverse_domain = geometry;
                const auto& allocated_domain           = field.domain();

                for (std::uint64_t k = 0; k < dd; ++k) {
                    transverse_domain.start[k] = allocated_domain.start[k];
                    transverse_domain.fin[k]   = allocated_domain.fin[k];
                }

                // -------------------------------------------------------------
                // process sides
                // -------------------------------------------------------------
                auto process_side = [&](grid::side_t side) {
                    const auto& conn = block_info->get_face(dd, side);

                    if (!conn.is_physical()) {
                        return;
                    }

                    const auto bc_type = conn.boundary_type();

                    // define ghost box
                    grid::domain_t<Rank> ghost_box = transverse_domain;

                    if (side == grid::side_t::left) {
                        ghost_box.start[dd] = allocated_domain.start[dd];
                        ghost_box.fin[dd]   = geometry.start[dd];
                    }
                    else {
                        ghost_box.start[dd] = geometry.fin[dd];
                        ghost_box.fin[dd]   = allocated_domain.fin[dd];
                    }

                    if (ghost_box.empty()) {
                        return;
                    }

                    // helper to execute standard static boundaries
                    auto execute_static = [&](auto&& map_op) {
                        auto phys_op = [=] DUAL(const T& val) {
                            return physics_policy.apply(val, dd, side, bc_type);
                        };

                        field[ghost_box] =
                            field[ghost_box].remap(map_op).map(phys_op).with(
                                exec
                            );
                    };

                    // ---------------------------------------------------------
                    // thin dimension override (Quasi-(1D/2D) support)
                    // ---------------------------------------------------------
                    // if the global dimension is 1, we must force a direct copy
                    // (periodic) to populate ghosts for transverse flux
                    // calculations in CT. standard bcs (like reflect) fail for
                    // depth > 1.
                    if (global_dims[dd] == 1) {
                        execute_static(
                            periodic_map_t{dd, geometry.start[dd], 1}
                        );
                        return;
                    }

                    // dispatch
                    switch (bc_type) {
                        case grid::boundary_type_t::outflow: {
                            std::int64_t edge = (side == grid::side_t::left)
                                                    ? geometry.start[dd]
                                                    : geometry.fin[dd] - 1;
                            execute_static(clamp_map_t{dd, edge, edge + 1});
                            break;
                        }
                        case grid::boundary_type_t::reflect: {
                            std::int64_t pivot = (side == grid::side_t::left)
                                                     ? geometry.start[dd]
                                                     : geometry.fin[dd];
                            execute_static(mirror_map_t{dd, pivot});
                            break;
                        }
                        case grid::boundary_type_t::periodic: {
                            execute_static(
                                periodic_map_t{
                                  dd,
                                  geometry.start[dd],
                                  global_dims[dd]
                                }
                            );
                            break;
                        }
                        case grid::boundary_type_t::dynamic: {
                            if constexpr (is_dynamic_context_c<Context>) {
                                // visit geometry to resolve specific metric
                                // type (uniform vs log vs user-defined)
                                geometry::visit_block_geometry<Rank>(
                                    context.geo_service,
                                    id,
                                    [&](const auto&... maps) {
                                        auto block_geo =
                                            typename Context::metric_type(
                                                maps...
                                            );

                                        // construct the dynamic operator
                                        // captures geo, vm, and time
                                        auto dyn_op = dynamic_boundary_op_t<
                                            T,
                                            decltype(block_geo),
                                            typename Context::vm_type,
                                            Rank>{
                                          block_geo,
                                          context.vm,
                                          context.time
                                        };

                                        // clamp indices to edge to get interior
                                        // state
                                        std::int64_t edge =
                                            (side == grid::side_t::left)
                                                ? geometry.start[dd]
                                                : geometry.fin[dd] - 1;

                                        auto clamp =
                                            clamp_map_t{dd, edge, edge + 1};

                                        // Execute:
                                        // > remap(clamp) -> gets U_interior
                                        // > enum_map(dynOp) -> uses  GhostCoord
                                        // + U_interior -> U_new
                                        field[ghost_box] = field[ghost_box]
                                                               .remap(clamp)
                                                               .enum_map(dyn_op)
                                                               .with(exec);
                                    }
                                );
                                break;
                            }
                            else {
                                // default to outflow if no dynamic context
                                std::int64_t edge = (side == grid::side_t::left)
                                                        ? geometry.start[dd]
                                                        : geometry.fin[dd] - 1;
                                execute_static(clamp_map_t{dd, edge, edge + 1});
                                break;
                            }
                        }
                        default: break;
                    }
                };

                process_side(grid::side_t::left);
                process_side(grid::side_t::right);
            }
        }
    };

}   // namespace simbi::geometry

#endif   // GRID_BOUNDARY_DRIVER_HPP
