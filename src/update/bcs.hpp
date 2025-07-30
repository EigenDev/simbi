#ifndef BOUNDARY_CONDITIONS_HPP
#define BOUNDARY_CONDITIONS_HPP

#include "config.hpp"
#include "containers/vector.hpp"
#include "core/utility/enums.hpp"
#include "core/utility/helpers.hpp"
#include "domain/domain.hpp"
#include "domain/ghost.hpp"
#include "mesh/mesh_ops.hpp"

#include <cstdint>
#include <numbers>
#include <utility>
#include <vector>

namespace simbi::boundary {
    // pure coordinate transformation logic
    template <std::uint64_t Dims>
    struct coordinate_transform_t {
        domain_t<Dims> ghost_region;
        domain_t<Dims> active_region;
        std::uint64_t contact_dim;
        face_side_t contact_dir;

        coordinate_t<Dims> reflect(coordinate_t<Dims> ghost_coord) const
        {
            auto reflected = ghost_coord;
            if (contact_dir == face_side_t::minus) {
                reflected[contact_dim] = active_region.start[contact_dim] +
                                         (active_region.start[contact_dim] -
                                          ghost_coord[contact_dim]) -
                                         1;
            }
            else {
                reflected[contact_dim] = active_region.end[contact_dim] -
                                         (ghost_coord[contact_dim] -
                                          active_region.end[contact_dim]) -
                                         1;
            }
            return reflected;
        }

        coordinate_t<Dims> clamp(coordinate_t<Dims> ghost_coord) const
        {
            auto clamped = ghost_coord;
            if (contact_dir == face_side_t::minus) {
                clamped[contact_dim] = active_region.start[contact_dim];
            }
            else {
                clamped[contact_dim] = active_region.end[contact_dim] - 1;
            }
            return clamped;
        }

        coordinate_t<Dims> wrap(coordinate_t<Dims> ghost_coord) const
        {
            auto wrapped = ghost_coord;
            if (contact_dir == face_side_t::minus) {
                auto offset =
                    active_region.start[contact_dim] - ghost_coord[contact_dim];
                wrapped[contact_dim] = active_region.end[contact_dim] - offset;
            }
            else {
                auto offset = ghost_coord[contact_dim] -
                              active_region.end[contact_dim] + 1;
                wrapped[contact_dim] =
                    active_region.start[contact_dim] + offset - 1;
            }
            return wrapped;
        }

        coordinate_t<Dims>
        apply(coordinate_t<Dims> ghost_coord, BoundaryCondition bc_type) const
        {
            switch (bc_type) {
                case BoundaryCondition::REFLECTING: return reflect(ghost_coord);
                case BoundaryCondition::OUTFLOW: return clamp(ghost_coord);
                case BoundaryCondition::PERIODIC: return wrap(ghost_coord);
                default: return clamp(ghost_coord);
            }
        }
    };

    // helper functions
    template <std::uint64_t Dims>
    auto find_contact_info(const vector_t<face_side_t, Dims>& directions)
    {
        for (std::uint64_t d = 0; d < Dims; ++d) {
            if (directions[d] != face_side_t::none) {
                return std::make_pair(d, directions[d]);
            }
        }
        return std::make_pair(static_cast<std::uint64_t>(0), face_side_t::none);
    }

    template <typename MeshConfig>
    auto get_thin_dimensions(const MeshConfig& mesh)
    {
        std::vector<std::uint64_t> thin_dims;
        constexpr auto Dims = MeshConfig::dimensions;
        for (std::uint64_t dim = 0; dim < Dims; ++dim) {
            if (mesh.shape[dim] == 1) {
                thin_dims.push_back(dim);
            }
        }
        return thin_dims;
    }

    // momentum transformation for reflecting BCs
    template <typename HydroState, typename MeshConfig>
    auto apply_reflecting_transform(
        typename HydroState::conserved_t value,
        std::uint64_t contact_dim,
        face_side_t contact_dir,
        const MeshConfig& mesh
    ) -> typename HydroState::conserved_t
    {
        if constexpr (requires { value.mom; }) {
            auto modified       = value;
            constexpr auto Dims = HydroState::dimensions;

            // handle special spherical geometry case
            if constexpr (MeshConfig::geometry == Geometry::SPHERICAL) {
                if (contact_dim == Dims - 2) {
                    const auto theta_max = mesh.current_bounds_max()[Dims - 2];
                    if (helpers::goes_to_zero(
                            theta_max - 0.5 * std::numbers::pi
                        ) &&
                        contact_dir == face_side_t::plus) {
                        modified.mom[Dims - 2] = -modified.mom[Dims - 2];
                        return modified;
                    }
                    else {
                        // theta momentum is continuous acrross the poles,
                        // so we do not flip it
                        return modified;
                    }
                }
            }

            // regular momentum flip
            auto momentum_idx          = (Dims - 1) - contact_dim;
            modified.mom[momentum_idx] = -modified.mom[momentum_idx];
            return modified;
        }
        return value;
    }

    // dynamic BC evaluation
    template <typename MeshType>
    auto evaluate_dynamic_bc(
        coordinate_t<MeshType::dimensions> ghost_coord,
        std::uint64_t contact_dim,
        domain_t<MeshType::dimensions> ghost_region,
        domain_t<MeshType::dimensions> active_region,
        const MeshType& mesh,
        const auto* bc_sources,
        real time,
        real dt,
        const auto& field
    )
    {
        auto directions     = ghost_direction(ghost_region, active_region);
        bool plus           = (directions[contact_dim] == face_side_t::plus);
        auto bc_index       = contact_dim * 2 + plus;
        const auto& bc_expr = bc_sources[bc_index];

        if (!bc_expr.enabled) {
            // fallback to outflow
            coordinate_transform_t<MeshType::dimensions> transform{
              ghost_region,
              active_region,
              contact_dim,
              directions[contact_dim]
            };
            auto src_coord = transform.clamp(ghost_coord);
            return field(src_coord);
        }

        // evaluate dynamic expression
        const auto position = mesh::centroid(ghost_coord, mesh);
        auto current_cons   = field(ghost_coord);
        return bc_expr.apply(position, current_cons, time, dt);
    }

    // BC transform creation for face ghosts
    template <typename HydroState, typename MeshConfig>
    auto face_bc_transform(
        const ghost_region_t<HydroState::dimensions>& ghost,
        const HydroState& state,
        const MeshConfig& mesh
    )
    {
        using conserved_t   = typename HydroState::conserved_t;
        constexpr auto Dims = HydroState::dimensions;

        auto ghost_domain               = ghost.domain;
        auto directions                 = ghost.directions;
        auto active_domain              = mesh.domain;
        auto [contact_dim, contact_dir] = find_contact_info(directions);

        auto bc_index = contact_dim * 2 + (contact_dir == face_side_t::plus);
        auto bc_type  = state.metadata.boundary_conditions[bc_index];

        coordinate_transform_t<Dims>
            transform{ghost_domain, active_domain, contact_dim, contact_dir};

        // capture needed state by value
        const auto time        = state.metadata.time;
        const auto dt          = state.metadata.dt;
        const auto* bc_sources = state.sources.bc_sources.data();
        const auto cons        = state.cons;

        auto bc_func = [=](coordinate_t<Dims> coord) -> conserved_t {
            if (bc_type == BoundaryCondition::DYNAMIC) {
                return evaluate_dynamic_bc(
                    coord,
                    contact_dim,
                    ghost_domain,
                    active_domain,
                    mesh,
                    bc_sources,
                    time,
                    dt,
                    cons
                );
            }

            auto source_coord = transform.apply(coord, bc_type);
            auto source_value = cons(source_coord);

            if (bc_type == BoundaryCondition::REFLECTING) {
                return apply_reflecting_transform<HydroState>(
                    source_value,
                    contact_dim,
                    contact_dir,
                    mesh
                );
            }

            return source_value;
        };

        return field(ghost_domain, bc_func);
    }

    // BC transform creation for corner ghosts
    template <typename HydroState, typename MeshConfig>
    auto corner_bc_transform(
        const ghost_region_t<HydroState::dimensions>& ghost,
        const HydroState& state,
        const MeshConfig& mesh
    )
    {
        using conserved_t   = typename HydroState::conserved_t;
        constexpr auto Dims = HydroState::dimensions;

        auto directions          = ghost.directions;
        auto active_domain       = mesh.domain;
        auto boundary_conditions = state.metadata.boundary_conditions;
        const auto cons          = state.cons;

        auto bc_func = [=](coordinate_t<Dims> coord) -> conserved_t {
            // map to interior cell
            auto interior_coord = coord;
            for (std::uint64_t d = 0; d < Dims; ++d) {
                if (directions[d] != face_side_t::none) {
                    interior_coord[d] = (directions[d] == face_side_t::minus)
                                            ? active_domain.start[d]
                                            : active_domain.end[d] - 1;
                }
            }

            auto base_value = cons(interior_coord);

            // apply momentum flips for reflecting boundaries
            auto final_value = base_value;
            for (std::uint64_t d = 0; d < Dims; ++d) {
                if (directions[d] != face_side_t::none) {
                    auto dir      = directions[d];
                    auto bc_index = d * 2 + (dir == face_side_t::plus);
                    auto bc_type  = boundary_conditions[bc_index];

                    if (bc_type == BoundaryCondition::REFLECTING) {
                        final_value = apply_reflecting_transform<HydroState>(
                            final_value,
                            d,
                            dir,
                            mesh
                        );
                    }
                }
            }

            return final_value;
        };

        return field(ghost.domain, bc_func);
    }

    // flux BC transform creation
    template <typename HydroState, typename MeshConfig>
    auto create_flux_bc_transform(
        const ghost_region_t<HydroState::dimensions>& ghost,
        domain_t<HydroState::dimensions> active_staggered,
        std::uint64_t flux_dim,
        const HydroState& state,
        const MeshConfig& mesh
    )
    {
        using conserved_t   = typename HydroState::conserved_t;
        constexpr auto Dims = HydroState::dimensions;

        auto ghost_domain = ghost.domain;
        auto directions   = ghost_direction(ghost_domain, active_staggered);
        auto [contact_dim, contact_dir] = find_contact_info(directions);

        auto bc_index = contact_dim * 2 + (contact_dir == face_side_t::plus);
        auto bc_type  = state.metadata.boundary_conditions[bc_index];

        coordinate_transform_t<Dims>
            transform{ghost_domain, active_staggered, contact_dim, contact_dir};

        // check for thin dimension handling
        auto thin_dims = get_thin_dimensions(mesh);

        auto flux    = state.flux[flux_dim];
        auto bc_func = [=](coordinate_t<Dims> coord) -> conserved_t {
            //  do regular BC handling first
            auto source_coord = transform.apply(coord, bc_type);

            //  apply symmetric copying for perpendicular thin dimensions
            auto final_coord = source_coord;
            for (auto thin_dim : thin_dims) {
                if (thin_dim == contact_dim) {
                    final_coord[thin_dim] = active_staggered.start[thin_dim];
                }
            }

            auto source_value = flux(final_coord);

            // apply momentum flip for reflecting BC
            if (bc_type == BoundaryCondition::REFLECTING) {
                if constexpr (requires { source_value.mom; }) {
                    auto modified              = source_value;
                    auto momentum_idx          = (Dims - 1) - contact_dim;
                    modified.mom[momentum_idx] = -modified.mom[momentum_idx];
                    source_value               = modified;
                }
            }

            return source_value;
        };
        return field(ghost.domain, bc_func);
    }

    // main BC application functions using cursor + map pattern
    template <typename HydroState, typename MeshConfig>
    void apply_cell_bcs(HydroState& state, const MeshConfig& mesh)
    {
        auto full_domain   = mesh.full_domain;
        auto active_domain = mesh.domain;
        auto ghost_info    = analyze_ghost_regions(full_domain, active_domain);

        for (auto ghost : ghost_info) {
            if (ghost.type == ghost_type_t::face) {
                auto bc_transform = face_bc_transform(ghost, state, mesh);
                state.cons        = state.cons.insert(bc_transform);
            }
            if constexpr (HydroState::is_mhd) {
                if (ghost.type == ghost_type_t::corner) {
                    auto bc_transform = corner_bc_transform(ghost, state, mesh);
                    state.cons        = state.cons.insert(bc_transform);
                }
            }
        }
    }

    template <typename HydroState, typename MeshConfig>
    void apply_flux_bcs(HydroState& state, const MeshConfig& mesh)
    {
        constexpr auto Dims = HydroState::dimensions;

        for (std::uint64_t flux_dim = 0; flux_dim < Dims; ++flux_dim) {
            auto flux             = state.flux[flux_dim];
            auto staggered_domain = flux.domain();
            auto active_staggered = mesh.face_domain[flux_dim];
            auto ghost_info =
                analyze_ghost_regions(staggered_domain, active_staggered);

            for (auto ghost : ghost_info) {
                // ignore corners and edges for flux BCs
                if (ghost.type == ghost_type_t::edge ||
                    ghost.type == ghost_type_t::corner) {
                    continue;
                }

                auto flux_bc_transform = create_flux_bc_transform(
                    ghost,
                    active_staggered,
                    flux_dim,
                    state,
                    mesh
                );
                flux = flux.insert(flux_bc_transform);
            }
        }
    }

    // thin dimension handling
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
            contracted.end[thin_dim] -= halo_radius;
        }
        return contracted;
    }

    template <typename HydroState, typename MeshConfig>
    void apply_thin_dimension_bcs(HydroState& state, const MeshConfig& mesh)
    {
        constexpr auto dims = HydroState::dimensions;
        auto thin_dims      = get_thin_dimensions(mesh);
        if (thin_dims.empty()) {
            return;
        }

        auto full_domain = state.cons.domain();
        auto interior_domain =
            contract_in_thin_dims(full_domain, thin_dims, mesh.halo_radius);
        auto cons = state.cons;
        auto thin_bc_transform =
            [interior_domain, thin_dims, cons, mesh](coordinate_t<dims> coord) {
                if (interior_domain.contains(coord)) {
                    return cons(coord);   // interior cell
                }

                // project to interior
                auto interior_coord = coord;
                for (auto thin_dim : thin_dims) {
                    interior_coord[thin_dim] = interior_domain.start[thin_dim];
                }

                return cons(interior_coord);
            };

        state.cons = state.cons.insert(field(full_domain, thin_bc_transform));
    }

    // main API
    template <typename HydroState, typename MeshConfig>
    void apply_boundary_conditions(HydroState& state, const MeshConfig& mesh)
    {
        // apply cell-centered boundary conditions
        apply_cell_bcs(state, mesh);

        // handle thin dimensions if needed
        apply_thin_dimension_bcs(state, mesh);
    }

}   // namespace simbi::boundary

#endif
