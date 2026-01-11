#ifndef GRID_BOUNDARY_DRIVER_HPP
#define GRID_BOUNDARY_DRIVER_HPP

#include "build_config.hpp"
#include "containers/vector.hpp"
#include "geometry/boundary/index_map.hpp"
#include "geometry/visit.hpp"
#include "grid/boundary.hpp"
#include "grid/connectivity.hpp"
#include "grid/field.hpp"
#include "grid/ghost.hpp"
#include "grid/mesh_config.hpp"
#include "grid/patch_id.hpp"
#include "grid/skeleton.hpp"
#include "index_map.hpp"

#include <cstddef>
#include <cstdint>

namespace simbi::geometry {

    // for simple runs without dynamic boundaries
    struct simple_context_t
    {
        using metric_type                 = void; // not needed
        static constexpr bool has_dynamic = false;
    };

    // helper tag for CTAD
    // usage: dynamic_context_t ctx(use_metric<my_metric>, geo, vm, time);
    template <typename Metric>
    struct use_metric_t
    {
    };

    template <typename Metric>
    constexpr use_metric_t<Metric> use_metric{};

    // for full physics runs
    template <typename Metric, typename GeoService, typename VM>
    struct dynamic_context_t
    {
        using metric_type                 = Metric; // e.g., spherical_metric_t
        using vm_type                     = VM;
        static constexpr bool has_dynamic = true;

        const GeoService& geo_service;
        const VM&         vm;
        real              time;

        // constructor required for CTAD to strip the tag
        constexpr dynamic_context_t(
            use_metric_t<Metric>,
            const GeoService& g,
            const VM&         v,
            const real        t
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
    template <typename T, typename Blockgeometry_t, typename Expression, std::uint64_t Rank>
    struct dynamic_boundary_op_t
    {
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
        // idx: the ghost cell index
        // interior_state: the state at the nearest active edge (provided by
        // remap)
        DEV T operator()(const iarray<Rank>& idx, const T& interior_state) const
        {
            // geometry: convert index to physical position
            auto phys_pos = geometry.centroid(idx);

            // physics: execute the vm
            // apply(position, interior_state, time)
            return vm.apply(phys_pos, interior_state, time);
        }
    };

    // -------------------------------------------------------------------------
    // physics boundary functor
    // applies physics policy to boundary values
    // -------------------------------------------------------------------------
    template <typename T, std::uint64_t Rank, typename Policy>
    struct physics_boundary_t
    {
        Policy                physics_policy;
        std::uint64_t         dim;
        grid::side_t          side;
        grid::boundary_type_t bc_type;

        DEV T operator()(const T& val) const
        {
            return physics_policy.apply(val, dim, side, bc_type);
        }
    };

    struct boundary_driver_t
    {

        // ---------------------------------------------------------------------
        // apply_boundaries
        // ---------------------------------------------------------------------
        template <typename T, std::uint64_t Rank, typename Context, typename Policy, typename Exec>
        static void apply_boundaries(
            grid::field_t<T, Rank>&          field,
            const grid::patch_id_t&          id,
            const grid::skeleton_t<Rank>&    skeleton,
            const grid::mesh_config_t<Rank>& config,
            const Policy&                    physics_policy,
            const Context&                   context,
            Exec&                            exec
        )
        {
            using namespace grid::domain_algebra;
            using namespace grid::boundary;

            const auto* block_info = skeleton.get_block(id);
            if (!block_info) {
                return;
            }

            const auto& geometry    = block_info->geometry;
            const auto& global_dims = config.global_cells;

            // -----------------------------------------------------------------
            // ghost region analysis: faces, edges, corners
            // -----------------------------------------------------------------
            auto ghost_regions = grid::boundary::analyze_ghost_regions(field.domain(), geometry);

            // -----------------------------------------------------------------
            // process each ghost region
            // -----------------------------------------------------------------
            for (const auto& ghost : ghost_regions) {
                // collect boundary info for all contact dimensions
                struct contact_info_t
                {
                    std::uint64_t         dim;
                    grid::side_t          side;
                    grid::boundary_type_t bc_type;
                };
                vector_t<contact_info_t, Rank> contacts;
                std::size_t                    num_contacts = 0;

                for (std::uint64_t dd = 0; dd < Rank; ++dd) {
                    using face_side = face_side_t;
                    if (ghost.directions[dd] == face_side::none) {
                        continue;
                    }

                    auto side = (ghost.directions[dd] == face_side::minus) ? grid::side_t::left
                                                                           : grid::side_t::right;
                    const auto& conn = block_info->get_face(dd, side);

                    // periodic boundaries are neighbor connections with
                    // wrapping
                    bool is_periodic_wrap = false;
                    if (conn.is_connected() && conn.is_conforming()) {
                        auto neighbor       = conn.single_neighbor();
                        auto my_coord       = id.coords[dd];
                        auto nb_coord       = neighbor.coords[dd];
                        auto blocks_per_dim = config.global_cells[dd] / config.block_size[dd];

                        // single block: wraps to itself
                        if (blocks_per_dim == 1) {
                            is_periodic_wrap = (my_coord == nb_coord);
                        }
                        // multi-block: wraps to opposite edge
                        else {
                            is_periodic_wrap = (side == grid::side_t::left && my_coord == 0 &&
                                                nb_coord == blocks_per_dim - 1) ||
                                               (side == grid::side_t::right &&
                                                my_coord == blocks_per_dim - 1 && nb_coord == 0);
                        }
                    }

                    // skip internal partition boundaries
                    if (!conn.is_physical() && !is_periodic_wrap) {
                        continue;
                    }

                    // determine BC type
                    auto bc_type =
                        is_periodic_wrap ? grid::boundary_type_t::periodic : conn.boundary_type();

                    contacts[num_contacts++] = {dd, side, bc_type};
                }

                if (num_contacts == 0) {
                    continue;
                }

                // use first contact for physics policy
                auto        primary_dd      = contacts[0].dim;
                auto        primary_side    = contacts[0].side;
                auto        primary_bc_type = contacts[0].bc_type;
                const auto& primary_conn    = block_info->get_face(primary_dd, primary_side);

                // -------------------------------------------------------------
                // special case: spherical pole (single contact only)
                // -------------------------------------------------------------
                if (num_contacts == 1 && primary_bc_type == grid::boundary_type_t::reflect) {

                    bool                    is_pole         = false;
                    constexpr std::uint64_t theta_array_dim = Rank - 2;

                    if constexpr (Rank >= 2) {
                        if constexpr (is_dynamic_context_c<Context>) {
                            if (primary_conn.has_metric_info() && primary_dd == theta_array_dim) {
                                is_pole = primary_conn.is_pole();
                            }
                        }
                    }

                    if (is_pole) {
                        std::int64_t phi_start = 0;
                        std::int64_t phi_len   = 0;

                        if constexpr (Rank >= 3) {
                            constexpr std::uint64_t phi_dim = Rank - 3;
                            phi_start                       = geometry.start[phi_dim];
                            phi_len                         = global_dims[phi_dim];
                        }

                        std::int64_t pivot = (primary_side == grid::side_t::left)
                                                 ? geometry.start[primary_dd]
                                                 : geometry.fin[primary_dd];

                        auto pole_map = spherical_pole_map_t<Rank>{pivot, phi_start, phi_len};

                        auto phys_op = physics_boundary_t<T, Rank, Policy>{
                            physics_policy,
                            primary_dd,
                            primary_side,
                            primary_bc_type
                        };

                        field[ghost.domain] =
                            field[ghost.domain].remap(pole_map).map(phys_op).with(exec);
                        continue;
                    }
                }

                // -------------------------------------------------------------
                // special case: thin dimension override
                // -------------------------------------------------------------
                if (num_contacts == 1 && global_dims[primary_dd] == 1) {
                    auto periodic_map = periodic_map_t<Rank>{
                        primary_dd,
                        geometry.start[primary_dd],
                        global_dims[primary_dd]
                    };

                    auto phys_op = physics_boundary_t<T, Rank, Policy>{
                        physics_policy,
                        primary_dd,
                        primary_side,
                        primary_bc_type
                    };

                    field[ghost.domain] =
                        field[ghost.domain].remap(periodic_map).map(phys_op).with(exec);
                    continue;
                }

                // -------------------------------------------------------------
                // special case: dynamic boundary (single contact only)
                // -------------------------------------------------------------
                if (num_contacts == 1 && primary_bc_type == grid::boundary_type_t::dynamic) {

                    if constexpr (is_dynamic_context_c<Context>) {
                        geometry::visit_block_geometry<Rank>(
                            context.geo_service,
                            id,
                            [&](const auto&... maps) {
                                auto block_geo = typename Context::metric_type(maps...);

                                auto dyn_op = dynamic_boundary_op_t<
                                    T,
                                    decltype(block_geo),
                                    typename Context::vm_type,
                                    Rank>{block_geo, context.vm, context.time};

                                std::int64_t edge = (primary_side == grid::side_t::left)
                                                        ? geometry.start[primary_dd]
                                                        : geometry.fin[primary_dd] - 1;

                                auto clamp = clamp_map_t<Rank>{primary_dd, edge, edge + 1};

                                field[ghost.domain] =
                                    field[ghost.domain].remap(clamp).enum_map(dyn_op).with(exec);
                            }
                        );
                    }
                    else {
                        // fallback to outflow
                        std::int64_t edge  = (primary_side == grid::side_t::left)
                                                 ? geometry.start[primary_dd]
                                                 : geometry.fin[primary_dd] - 1;
                        auto         clamp = clamp_map_t<Rank>{primary_dd, edge, edge + 1};

                        auto phys_op = physics_boundary_t<T, Rank, Policy>{
                            physics_policy,
                            primary_dd,
                            primary_side,
                            primary_bc_type
                        };

                        field[ghost.domain] =
                            field[ghost.domain].remap(clamp).map(phys_op).with(exec);
                    }
                    continue;
                }

                // -------------------------------------------------------------
                // general case: build multi-dimensional map
                // -------------------------------------------------------------
                multidim_map_t<Rank> map;
                map.active_dims_.fill(0);
                map.map_types_.fill(0);
                map.starts_.fill(0);
                map.lens_.fill(0);
                map.pivots_.fill(0);
                map.clamp_vals_.fill(0);

                for (std::size_t ii = 0; ii < num_contacts; ++ii) {
                    auto dd      = contacts[ii].dim;
                    auto side    = contacts[ii].side;
                    auto bc_type = contacts[ii].bc_type;

                    map.active_dims_[dd] = 1;

                    switch (bc_type) {
                        case grid::boundary_type_t::periodic:
                            map.map_types_[dd] = 1;
                            map.starts_[dd]    = geometry.start[dd];
                            map.lens_[dd]      = global_dims[dd];
                            break;

                        case grid::boundary_type_t::reflect:
                            map.map_types_[dd] = 2;
                            map.pivots_[dd] = 2 * (side == grid::side_t::left ? geometry.start[dd]
                                                                              : geometry.fin[dd]) -
                                              1;
                            break;

                        case grid::boundary_type_t::outflow:
                            map.map_types_[dd]  = 3;
                            map.clamp_vals_[dd] = (side == grid::side_t::left)
                                                      ? geometry.start[dd]
                                                      : geometry.fin[dd] - 1;
                            break;

                        default:
                            break;
                    }
                }

                auto phys_op = physics_boundary_t<T, Rank, Policy>{
                    physics_policy,
                    primary_dd,
                    primary_side,
                    primary_bc_type
                };

                field[ghost.domain] = field[ghost.domain].remap(map).map(phys_op).with(exec);
                // field[ghost.domain] = field[ghost.domain]
                //                           .enum_map([](auto coord, auto c) {
                //                               std::cout << "ghost coord: " << coord
                //                                         << " value: " << c << "\n";
                //                               return c;
                //                           })
                //                           .with(exec);
            }
        }
    };

} // namespace simbi::geometry

#endif // GRID_BOUNDARY_DRIVER_HPP
