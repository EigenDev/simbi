#ifndef BOUNDARY_CONDITIONS_HPP
#define BOUNDARY_CONDITIONS_HPP

#include "compat.hpp"
#include "compute/field.hpp"
#include "containers/vector.hpp"
#include "domain/domain.hpp"
#include "domain/ghost.hpp"
#include "mesh/mesh_config.hpp"
#include "mesh/mesh_ops.hpp"
#include "state/express_t.hpp"
#include "utility/bimap.hpp"
#include "utility/enums.hpp"
#include "utility/helpers.hpp"

#include <cstdint>
#include <numbers>
#include <utility>
#include <vector>

namespace simbi::boundary {

    // detect mhd capabilities via type traits
    template <typename T>
    struct bc_traits {
        static constexpr bool is_mhd = requires { typename T::bfield_type; };
    };

    // unified context for all bc operations
    template <typename FieldType, Geometry G, std::uint64_t Dims>
    struct bc_context_t {
        using value_type                     = typename FieldType::value_type;
        static constexpr Geometry geometry_t = G;
        const FieldType& field;
        domain_t<Dims> ghost_region;
        domain_t<Dims> active_region;
        mesh::mesh_config_t<Dims, G> mesh;
        std::uint64_t contact_dim{0};
        face_side_t contact_dir{face_side_t::none};
        std::uint64_t bc_index{0};
        BoundaryCondition bc_type{BoundaryCondition::OUTFLOW};
        real time{0};
        const state::expression_t<Dims>* bc_expr{nullptr};   // for dynamic BCs

        // validate context consistency
        constexpr bool is_valid() const
        {
            if (ghost_region.empty() || active_region.empty()) {
                return false;
            }
            // verify contact_dim is valid for Dims
            if (contact_dim >= Dims) {
                return false;
            }
            // verify regions actually contact each other
            return domain_algebra::adjacent(ghost_region, active_region);
        }
    };

    // base transform interface
    template <typename T>
    struct bc_transform_base_t {
        using value_type = T;

        template <std::uint64_t Dims, typename MeshConfig>
        DUAL value_type operator()(
            coordinate_t<Dims> ghost_coord,
            const bc_context_t<
                T,
                MeshConfig::geometry_t,
                MeshConfig::dimensions>& ctx
        ) const;
    };

    // coordinate reflection transform
    template <typename T>
    struct reflecting_transform_t : bc_transform_base_t<T> {
        template <std::uint64_t Dims, typename Context>
        DUAL auto
        operator()(const Context& ctx, coordinate_t<Dims> ghost_coord) const
        {
            // get reflected coordinate
            auto reflected = ghost_coord;
            if (ctx.contact_dir == face_side_t::minus) {
                reflected[ctx.contact_dim] =
                    ctx.active_region.start[ctx.contact_dim] +
                    (ctx.active_region.start[ctx.contact_dim] -
                     ghost_coord[ctx.contact_dim]) -
                    1;
            }
            else {
                reflected[ctx.contact_dim] =
                    ctx.active_region.fin[ctx.contact_dim] -
                    (ghost_coord[ctx.contact_dim] -
                     ctx.active_region.fin[ctx.contact_dim]) -
                    1;
            }

            // handle special spherical geometry case for mhd
            if constexpr (bc_traits<T>::is_mhd) {
                if constexpr (Context::geometry_t == Geometry::SPHERICAL) {
                    if constexpr (Dims > 1) {
                        if (ctx.contact_dim == Dims - 2) {
                            if (helpers::goes_to_zero(
                                    ctx.mesh.curren_bounds_max()[Dims - 2] -
                                    0.5 * std::numbers::pi
                                ) &&
                                ctx.contact_dir == face_side_t::plus) {
                                auto value          = ctx.field(reflected);
                                value.mom[Dims - 2] = -value.mom[Dims - 2];
                                return value;
                            }
                            return ctx.field(reflected);
                        }
                    }
                }
            }

            // apply regular momentum reflection
            auto value = ctx.field(reflected);
            if constexpr (requires { value.mom; }) {
                auto momentum_idx       = (Dims - 1) - ctx.contact_dim;
                value.mom[momentum_idx] = -value.mom[momentum_idx];
            }
            return value;
        }
    };

    // outflow (copy boundary) transform
    template <typename T>
    struct outflow_transform_t : bc_transform_base_t<T> {
        template <std::uint64_t Dims, typename Context>
        DUAL auto
        operator()(const Context& ctx, coordinate_t<Dims> ghost_coord) const
        {
            auto clamped = ghost_coord;
            if (ctx.contact_dir == face_side_t::minus) {
                clamped[ctx.contact_dim] =
                    ctx.active_region.start[ctx.contact_dim];
            }
            else {
                clamped[ctx.contact_dim] =
                    ctx.active_region.fin[ctx.contact_dim] - 1;
            }
            return ctx.field(clamped);
        }
    };

    // periodic boundary transform
    template <typename T>
    struct periodic_transform_t : bc_transform_base_t<T> {
        template <std::uint64_t Dims, typename Context>
        DUAL auto
        operator()(const Context& ctx, coordinate_t<Dims> ghost_coord) const
        {
            auto wrapped = ghost_coord;
            if (ctx.contact_dir == face_side_t::minus) {
                auto offset = ctx.active_region.start[ctx.contact_dim] -
                              ghost_coord[ctx.contact_dim];
                wrapped[ctx.contact_dim] =
                    ctx.active_region.fin[ctx.contact_dim] - offset;
            }
            else {
                auto offset = ghost_coord[ctx.contact_dim] -
                              ctx.active_region.fin[ctx.contact_dim] + 1;
                wrapped[ctx.contact_dim] =
                    ctx.active_region.start[ctx.contact_dim] + offset - 1;
            }
            return ctx.field(wrapped);
        }
    };

    // dynamic boundary condition transform
    template <typename T>
    struct dynamic_transform_t : bc_transform_base_t<T> {
        template <std::uint64_t Dims, typename Context>
        DUAL auto
        operator()(const Context& ctx, coordinate_t<Dims> ghost_coord) const
        {
            const auto& expr = ctx.bc_expr[ctx.bc_index];
            if (!expr.enabled) {
                return outflow_transform_t<T>{}(ctx, ghost_coord);
            }
            const auto position = mesh::centroid(ghost_coord, ctx.mesh);
            auto current_cons   = ctx.field(ghost_coord);
            return expr.apply(position, current_cons, ctx.time);
        }
    };

    // helper to identify thin dimensions
    template <typename MeshConfig>
    auto get_thin_dimensions(const MeshConfig& mesh)
    {
        std::vector<std::uint64_t> thin_dims;
        constexpr auto Dims = MeshConfig::dimensions;

        for (std::uint64_t dd = 0; dd < Dims; ++dd) {
            if (mesh.shape[dd] == 1) {
                thin_dims.push_back(dd);
            }
        }
        return thin_dims;
    }

    // helper for thin dimension contraction
    template <std::uint64_t Dims>
    auto contract_in_thin_dims(
        domain_t<Dims> domain,
        const std::vector<std::uint64_t>& thin_dims,
        std::uint64_t halo_radius
    )
    {
        auto contracted = domain;
        for (auto thin_dim : thin_dims) {
            contracted.start[thin_dim] += halo_radius;
            contracted.fin[thin_dim] -= halo_radius;
        }
        return contracted;
    }

    template <typename Context>
    auto make_bc_transform(const Context& ctx)
    {
        using T = Context::value_type;
        // create stateless transforms
        auto reflecting = reflecting_transform_t<T>{};
        auto outflow    = outflow_transform_t<T>{};
        auto periodic   = periodic_transform_t<T>{};
        auto dynamic    = dynamic_transform_t<T>{};

        // compose using fp toolkit
        return fp::select(
            [=]() { return ctx.bc_type == BoundaryCondition::REFLECTING; },
            fp::partial(reflecting, ctx),
            fp::select(
                [=]() { return ctx.bc_type == BoundaryCondition::PERIODIC; },
                fp::partial(periodic, ctx),
                fp::select(
                    [=]() { return ctx.bc_type == BoundaryCondition::DYNAMIC; },
                    fp::partial(dynamic, ctx),
                    fp::partial(outflow, ctx)
                )
            )
        );
    }

    // flux boundary transform
    template <typename FluxField, typename MeshConfig>
    void stagg_bc_transform(
        const ghost_region_t<MeshConfig::dimensions>& ghost,
        domain_t<MeshConfig::dimensions> active_staggered,
        std::uint64_t flux_dim,
        FluxField& flux,
        const MeshConfig& mesh,
        const auto& boundary_conditions
    )
    {
        auto [contact_dim, contact_dir] = find_contact_info(ghost.directions);
        const auto bc_index = get_bc_index(contact_dim, contact_dir);

        auto ctx = bc_context_t{
          .field         = flux[flux_dim],
          .ghost_region  = ghost.domain,
          .active_region = active_staggered,
          .mesh          = mesh,
          .contact_dim   = contact_dim,
          .contact_dir   = contact_dir,
          .bc_index      = bc_index,
          .bc_type       = boundary_conditions[bc_index],
        };

        auto transform = make_bc_transform(ctx);
        flux[flux_dim] = flux[flux_dim].insert(field(ghost.domain, transform));
    }

    // apply boundary conditions to staggered fields
    template <typename HydroState, typename MeshConfig>
    void apply_stagg_bcs(
        HydroState& state,
        const MeshConfig& mesh,
        const auto& boundary_conditions
    )
    {
        constexpr auto Dims = MeshConfig::dimensions;

        // handle each flux dimension
        for (std::uint64_t flux_dim = 0; flux_dim < Dims; ++flux_dim) {
            auto flux             = state.flux[flux_dim];
            auto staggered_domain = flux.domain();
            auto active_staggered = mesh.face_domain[flux_dim];

            // analyze ghost regions for this staggered domain
            auto ghost_info =
                analyze_ghost_regions(staggered_domain, active_staggered);

            // apply bcs to valid ghost regions
            for (auto ghost : ghost_info) {
                // ignore corners and edges for flux bcs
                if (ghost.type == ghost_type_t::edge ||
                    ghost.type == ghost_type_t::corner) {
                    continue;
                }

                stagg_bc_transform(
                    ghost,
                    active_staggered,
                    flux_dim,
                    state.flux,
                    mesh,
                    boundary_conditions
                );
            }
        }
    }

    // handle thin dimension boundary conditions
    template <typename HydroState, typename MeshConfig>
    void apply_thin_dimension_bcs(HydroState& state, const MeshConfig& mesh)
    {
        constexpr auto Dims = HydroState::dimensions;

        // get thin dimensions from mesh
        auto thin_dims = get_thin_dimensions(mesh);
        if (thin_dims.empty()) {
            return;
        }

        // contract domain in thin dimensions
        auto full_domain = state.cons.domain();
        auto interior_domain =
            contract_in_thin_dims(full_domain, thin_dims, mesh.halo_radius);

        vector_t<std::uint64_t, Dims> dev_thin_dims;
        for (std::uint64_t ii = 0; ii < thin_dims.size(); ++ii) {
            dev_thin_dims[ii] = thin_dims[ii];
        }

        auto cons      = state.cons;
        auto transform = [interior_domain,
                          dev_thin_dims,
                          cons] DEV(coordinate_t<Dims> coord) {
            if (interior_domain.contains(coord)) {
                return cons(coord);   // interior cell
            }

            // project to interior
            auto interior_coord = coord;
            for (auto thin_dim : dev_thin_dims) {
                interior_coord[thin_dim] = interior_domain.start[thin_dim];
            }

            return cons(interior_coord);
        };

        // apply transform
        state.cons = state.cons.insert(field(full_domain, transform));
    }

    template <typename SimState>
    void apply_thin_dimension_bcs(SimState& sim)
    {
        constexpr auto Dims = SimState::dimensions;
        const auto& mesh    = sim.mesh(0);
        auto& cons          = sim.hydro(0).cons;

        // get thin dimensions from mesh
        auto thin_dims = get_thin_dimensions(mesh);
        if (thin_dims.empty()) {
            return;
        }

        // contract domain in thin dimensions
        auto full_domain = cons.domain();
        auto interior_domain =
            contract_in_thin_dims(full_domain, thin_dims, mesh.halo_radius);

        vector_t<std::uint64_t, Dims> dev_thin_dims;
        for (std::uint64_t ii = 0; ii < thin_dims.size(); ++ii) {
            dev_thin_dims[ii] = thin_dims[ii];
        }

        auto transform = [interior_domain,
                          dev_thin_dims,
                          cons] DEV(coordinate_t<Dims> coord) {
            if (interior_domain.contains(coord)) {
                return cons(coord);   // interior cell
            }

            // project to interior
            auto interior_coord = coord;
            for (auto thin_dim : dev_thin_dims) {
                interior_coord[thin_dim] = interior_domain.start[thin_dim];
            }

            return cons(interior_coord);
        };

        // apply transform
        cons = cons.insert(field(full_domain, transform));
    }

    // helper to find contact dimension and direction
    template <std::uint64_t Dims>
    auto find_contact_info(const vector_t<face_side_t, Dims>& directions)
    {
        for (std::uint64_t dd = 0; dd < Dims; ++dd) {
            if (directions[dd] != face_side_t::none) {
                return std::make_pair(dd, directions[dd]);
            }
        }
        return std::make_pair(static_cast<std::uint64_t>(0), face_side_t::none);
    }

    inline auto get_bc_index(std::uint64_t contact_dim, face_side_t contact_dir)
    {
        return contact_dim * 2 + (contact_dir == face_side_t::plus);
    }

    template <typename HydroState, typename MeshConfig>
    void face_bc_transform(
        const ghost_region_t<HydroState::dimensions>& ghost,
        HydroState& state,
        const MeshConfig& mesh
    )
    {
        constexpr auto Dims             = HydroState::dimensions;
        auto [contact_dim, contact_dir] = find_contact_info(ghost.directions);
        auto bc_index = get_bc_index(contact_dim, contact_dir);

        auto ctx = bc_context_t{
          .field         = state.cons,
          .ghost_region  = ghost.domain,
          .active_region = mesh.domain,
          .mesh          = mesh,
          .contact_dim   = contact_dim,
          .contact_dir   = contact_dir,
          .bc_index      = bc_index,
          .bc_type       = state.metadata.boundary_conditions[bc_index],
          .time          = state.metadata.time,
          .bc_expr       = state.sources.bc_sources.data()
        };

        auto transform = make_bc_transform(ctx);

        state.cons = state.cons.insert(field(ghost.domain, transform));
    }

    template <typename SimState>
    void face_bc_transform(
        const ghost_region_t<SimState::dimensions>& ghost,
        SimState& sim
    )
    {
        const auto& mesh       = sim.mesh(0);
        const auto& meta       = sim.metadata();
        const auto& bc_sources = sim.sources().bc_sources;
        auto& cons             = sim.hydro(0).cons;

        auto [contact_dim, contact_dir] = find_contact_info(ghost.directions);
        auto bc_index = get_bc_index(contact_dim, contact_dir);

        auto ctx = bc_context_t{
          .field         = cons,
          .ghost_region  = ghost.domain,
          .active_region = mesh.domain,
          .mesh          = mesh,
          .contact_dim   = contact_dim,
          .contact_dir   = contact_dir,
          .bc_index      = bc_index,
          .bc_type       = meta.boundary_conditions[bc_index],
          .time          = meta.time,
          .bc_expr       = bc_sources.data()
        };

        auto transform = make_bc_transform(ctx);

        cons = cons.insert(field(ghost.domain, transform));
    }

    template <typename HydroState, typename MeshConfig>
    void corner_bc_transform(
        const ghost_region_t<HydroState::dimensions>& ghost,
        HydroState& state,
        const MeshConfig& mesh
    )
    {
        constexpr auto Dims = HydroState::dimensions;

        auto ctx = bc_context_t{
          .field         = state.cons,
          .ghost_region  = ghost.domain,
          .active_region = mesh.domain,
          .mesh          = mesh
        };

        auto& meta = state.metadata;
        auto cons  = state.cons;
        // handle multiple reflecting boundaries
        auto transform = [=] DEV(coordinate_t<Dims> coord) {
            auto interior_coord = coord;

            // first map to interior
            for (std::uint64_t dd = 0; dd < Dims; ++dd) {
                if (ghost.directions[dd] != face_side_t::none) {
                    interior_coord[dd] =
                        (ghost.directions[dd] == face_side_t::minus)
                            ? ctx.active_region.start[dd]
                            : ctx.active_region.fin[dd] - 1;
                }
            }

            // get base value from interior
            auto value = cons(interior_coord);

            //  apply any needed reflections
            for (std::uint64_t dd = 0; dd < Dims; ++dd) {
                if (ghost.directions[dd] != face_side_t::none) {
                    auto bc_index = get_bc_index(dd, ghost.directions[dd]);
                    if (meta.boundary_conditions[bc_index] ==
                        BoundaryCondition::REFLECTING) {
                        if constexpr (requires { value.mom; }) {
                            auto momentum_idx       = (Dims - 1) - dd;
                            value.mom[momentum_idx] = -value.mom[momentum_idx];
                        }
                    }
                }
            }
            return value;
        };

        state.cons = state.cons.insert(field(ghost.domain, transform));
    }

    template <typename SimState>
    void corner_bc_transform(
        const ghost_region_t<SimState::dimensions>& ghost,
        SimState& sim
    )
    {
        constexpr auto Dims = SimState::dimensions;

        const auto& mesh = sim.mesh(0);
        const auto& meta = sim.metadata();
        auto& cons       = sim.hydro(0).cons;

        auto ctx = bc_context_t{
          .field         = cons,
          .ghost_region  = ghost.domain,
          .active_region = mesh.domain,
          .mesh          = mesh
        };

        // handle multiple reflecting boundaries
        auto transform = [=] DEV(coordinate_t<Dims> coord) {
            auto interior_coord = coord;

            // first map to interior
            for (std::uint64_t dd = 0; dd < Dims; ++dd) {
                if (ghost.directions[dd] != face_side_t::none) {
                    interior_coord[dd] =
                        (ghost.directions[dd] == face_side_t::minus)
                            ? ctx.active_region.start[dd]
                            : ctx.active_region.fin[dd] - 1;
                }
            }

            // get base value from interior
            auto value = cons(interior_coord);

            //  apply any needed reflections
            for (std::uint64_t dd = 0; dd < Dims; ++dd) {
                if (ghost.directions[dd] != face_side_t::none) {
                    auto bc_index = get_bc_index(dd, ghost.directions[dd]);
                    if (meta.boundary_conditions[bc_index] ==
                        BoundaryCondition::REFLECTING) {
                        if constexpr (requires { value.mom; }) {
                            auto momentum_idx       = (Dims - 1) - dd;
                            value.mom[momentum_idx] = -value.mom[momentum_idx];
                        }
                    }
                }
            }
            return value;
        };

        cons = cons.insert(field(ghost.domain, transform));
    }

    // apply all boundary conditions
    template <typename HydroState, typename MeshConfig>
    void apply_boundary_conditions(HydroState& state, const MeshConfig& mesh)
    {
        auto full_domain   = mesh.full_domain;
        auto active_domain = mesh.domain;
        auto ghost_info    = analyze_ghost_regions(full_domain, active_domain);

        for (auto ghost : ghost_info) {
            if (ghost.type == ghost_type_t::face) {
                face_bc_transform(ghost, state, mesh);
            }
            if constexpr (HydroState::is_mhd) {
                if (ghost.type == ghost_type_t::corner) {
                    corner_bc_transform(ghost, state, mesh);
                }
            }
        }

        apply_thin_dimension_bcs(state, mesh);
    }

    template <typename SimState>
    void apply_boundary_conditions(SimState& sim)
    {
        // bcs only valid at level 0
        const auto& mesh   = sim.mesh(0);
        auto full_domain   = mesh.full_domain;
        auto active_domain = mesh.domain;
        auto ghost_info    = analyze_ghost_regions(full_domain, active_domain);

        for (auto ghost : ghost_info) {
            if (ghost.type == ghost_type_t::face) {
                face_bc_transform(ghost, sim);
            }
            if constexpr (SimState::is_mhd) {
                if (ghost.type == ghost_type_t::corner) {
                    corner_bc_transform(ghost, sim);
                }
            }
        }

        apply_thin_dimension_bcs(sim);
    }

}   // namespace simbi::boundary

#endif   // BOUNDARY_CONDITIONS_HPP
