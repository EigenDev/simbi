#ifndef BOUNDARY_CONDTIONS_HPP
#define BOUNDARY_CONDTIONS_HPP

#include "compute/functional/fp.hpp"    // for fp::
#include "compute/math/domain.hpp"      // for domain_t
#include "compute/math/expr.hpp"        // for expr::expression_t
#include "core/utility/enums.hpp"       // for direction_t
#include "data/containers/vector.hpp"   // for vector_t, iarray
#include <cstddef>                      // for std::size_t
#include <cstdint>                      // for std::uint64_t
#include <numbers>                      // for std::numbers::pi
#include <utility>                      // for std::move
#include <vector>                       // for std::vector

namespace simbi::hydro {
    using namespace simbi::set_ops;
    // forward declarations
    template <std::uint64_t Dims>
    auto reflect_coordinate(
        const iarray<Dims>& ghost_coord,
        std::uint64_t contact_dim,
        domain_t<Dims> ghost_region,
        domain_t<Dims> active_region
    );

    template <std::uint64_t Dims>
    auto clamp_to_boundary(
        const iarray<Dims>& ghost_coord,
        std::uint64_t contact_dim,
        domain_t<Dims> ghost_region,
        domain_t<Dims> active_region
    );

    template <std::uint64_t Dims>
    auto wrap_coordinate(
        const iarray<Dims>& ghost_coord,
        std::uint64_t contact_dim,
        domain_t<Dims> ghost_region,
        domain_t<Dims> active_region
    );

    template <std::uint64_t Dims>
    auto make_coord_range(domain_t<Dims> domain);

    template <std::uint64_t Dims>
    auto find_contact_dimension(const vector_t<direction_t, Dims>& directions);

    template <typename HydroState>
    void apply_boundary_conditions(HydroState& state);

    template <typename HydroState>
    auto apply_face_boundary(
        auto ghost_region,
        auto active_domain,
        HydroState& state
    );

    template <typename HydroState>
    auto apply_corner_boundary(
        auto ghost_region,
        auto active_domain,
        HydroState& state
    );

    template <typename Field>
    void apply_thin_dimension_bc(
        Field& field,
        const std::vector<std::uint64_t>& thin_dims,
        std::uint64_t
    );

    template <typename HydroState>
    auto apply_flux_boundary(
        auto ghost_region,
        auto active_domain,
        std::uint64_t flux_dim,
        HydroState& state,
        const std::vector<std::uint64_t>& cell_thin_dims
    );

    // coordinate transformation functions - fixed parameter issues
    template <std::uint64_t Dims>
    auto reflect_coordinate(
        const iarray<Dims>& ghost_coord,
        std::uint64_t contact_dim,
        domain_t<Dims> ghost_region,
        domain_t<Dims> active_region
    )
    {
        auto directions = ghost_direction(ghost_region, active_region);
        auto reflected  = ghost_coord;

        if (directions[contact_dim] == direction_t::minus) {
            reflected[contact_dim] =
                active_region.start[contact_dim] +
                (active_region.start[contact_dim] - ghost_coord[contact_dim]) -
                1;
        }
        else {
            reflected[contact_dim] =
                active_region.fin[contact_dim] -
                (ghost_coord[contact_dim] - active_region.fin[contact_dim]) - 1;
        }
        return reflected;
    }

    template <std::uint64_t Dims>
    auto clamp_to_boundary(
        const iarray<Dims>& ghost_coord,
        std::uint64_t contact_dim,
        domain_t<Dims> ghost_region,
        domain_t<Dims> active_region
    )
    {
        auto directions = ghost_direction(ghost_region, active_region);
        auto clamped    = ghost_coord;

        if (directions[contact_dim] == direction_t::minus) {
            clamped[contact_dim] = active_region.start[contact_dim];
        }
        else {
            clamped[contact_dim] = active_region.fin[contact_dim] - 1;
        }
        return clamped;
    }

    template <std::uint64_t Dims>
    auto wrap_coordinate(
        const iarray<Dims>& ghost_coord,
        std::uint64_t contact_dim,
        domain_t<Dims> ghost_region,
        domain_t<Dims> active_region
    )
    {
        auto directions = ghost_direction(ghost_region, active_region);
        auto wrapped    = ghost_coord;

        if (directions[contact_dim] == direction_t::minus) {
            auto offset =
                active_region.start[contact_dim] - ghost_coord[contact_dim];
            wrapped[contact_dim] = active_region.fin[contact_dim] - offset;
        }
        else {
            auto offset =
                ghost_coord[contact_dim] - active_region.fin[contact_dim] + 1;
            wrapped[contact_dim] =
                active_region.start[contact_dim] + offset - 1;
        }
        return wrapped;
    }

    template <typename HydroState>
    auto evaluate_dynamic_bc(
        auto ghost_coord,
        std::uint64_t contact_dim,
        domain_t<HydroState::dimensions> ghost_region,
        domain_t<HydroState::dimensions> active_region,
        HydroState& state
    )
    {
        auto directions = ghost_direction(ghost_region, active_region);
        auto bc_index   = contact_dim * 2 +
                        (directions[contact_dim] == direction_t::plus ? 1 : 0);
        const auto& bc_expr = state.sources.bc_sources[bc_index];

        if (!bc_expr.enabled) {
            // fallback to outflow - find active domain boundary
            auto active_domain = state.mesh.domain;
            auto src_coords    = clamp_to_boundary(
                ghost_coord,
                contact_dim,
                ghost_region,
                active_domain
            );
            return state.cons[src_coords];
        }

        // evaluate dynamic expression
        const auto position = mesh::centroid(ghost_coord, state.mesh);
        auto current_cons   = state.cons[ghost_coord];
        auto bc_value       = bc_expr.apply(
            position,
            current_cons,
            state.metadata.time,
            state.metadata.dt
        );

        return bc_value;
    }

    // create coordinate range from domain
    template <std::uint64_t Dims>
    auto make_coord_range(domain_t<Dims> domain)
    {
        return fp::generate([domain](std::uint64_t linear_idx) {
            return domain.linear_to_coord(linear_idx);
        });
    }

    // contract domain only in thin dimensions
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

    // derive staggered active domain - extend by 1 in stagger direction
    template <std::uint64_t Dims>
    auto derive_active_staggered_domain(
        domain_t<Dims> active_domain,
        std::uint64_t flux_dim
    )
    {
        auto staggered = active_domain;
        // staggered domain is one larger in the flux direction
        staggered.fin[flux_dim] += 1;
        return staggered;
    }

    template <typename HydroState>
    auto get_thin_dimensions(const HydroState& state)
    {
        std::vector<std::uint64_t> thin_dims;
        for (std::uint64_t dim = 0; dim < HydroState::dimensions; ++dim) {
            if (state.mesh.shape[dim] == 1) {
                thin_dims.push_back(dim);
            }
        }
        return thin_dims;
    }

    // find contact dimension
    template <std::uint64_t Dims>
    auto find_contact_dimension(const vector_t<direction_t, Dims>& directions)
    {
        // create range for dimensions
        auto dim_range = fp::range(Dims);

        // find first non-none direction
        for (auto dim : dim_range) {
            if (directions[dim] != direction_t::none) {
                auto dir =
                    (directions[dim] == direction_t::plus) ? Dir::E : Dir::W;
                return std::make_pair(dim, dir);
            }
        }

        // fallback - shouldn't happen
        return std::make_pair(static_cast<std::uint64_t>(0), Dir::W);
    }

    // main boundary condition application using fixed coordinate ranges
    // staggered boundary conditions
    template <typename HydroState>
    void apply_staggered_boundary_conditions(HydroState& state)
    {
        const auto cell_thin_dims = get_thin_dimensions(state);

        fp::range(HydroState::dimensions) | fp::for_each([&](auto flux_dim) {
            auto staggered_domain = state.flux[flux_dim].domain();
            auto active_staggered =
                derive_active_staggered_domain(state.mesh.domain, flux_dim);
            active_staggered = center(active_staggered, staggered_domain);
            auto ghost_regions =
                identify_ghost_regions(staggered_domain, active_staggered);

            for (std::uint64_t ii = 0; ii < ghost_regions.count; ++ii) {
                apply_flux_boundary(
                    ghost_regions.regions[ii],
                    active_staggered,
                    flux_dim,
                    state,
                    cell_thin_dims
                );
            }
        });
    }

    template <typename HydroState>
    void apply_cell_centered_boundary_conditions(HydroState& state)
    {
        auto full_domain   = state.mesh.full_domain;
        auto active_domain = center(state.mesh.domain, full_domain);
        auto ghost_regions = identify_ghost_regions(full_domain, active_domain);

        // process face boundaries
        for (std::uint64_t ii = 0; ii < ghost_regions.count; ++ii) {
            auto ghost_region = ghost_regions.regions[ii];
            if (classify_ghost_contact(ghost_region, active_domain) ==
                ghost_type_t::face) {
                apply_face_boundary(ghost_region, active_domain, state);
            }
        }

        // process corner boundaries
        for (std::uint64_t ii = 0; ii < ghost_regions.count; ++ii) {
            auto ghost_region = ghost_regions.regions[ii];
            if (classify_ghost_contact(ghost_region, active_domain) ==
                ghost_type_t::corner) {
                apply_corner_boundary(ghost_region, active_domain, state);
            }
        }
    }

    // main boundary condition orchestrator
    template <typename HydroState>
    void apply_boundary_conditions(HydroState& state)
    {
        // apply cell-centered boundary conditions
        apply_cell_centered_boundary_conditions(state);

        const auto thin_dims = get_thin_dimensions(state);
        // handle thin dimensions last if any exist
        if (!thin_dims.empty()) {
            const auto& mesh     = state.mesh;
            const auto cell_halo = mesh.halo_radius;
            // apply to cell-centered fields
            apply_thin_dimension_bc(state.cons, thin_dims, cell_halo);
        }
    }

    template <typename HydroState>
    auto apply_face_boundary(
        auto ghost_region,
        auto active_domain,
        HydroState& state
    )
    {
        auto directions = ghost_direction(ghost_region, active_domain);
        auto [contact_dim, contact_dir] = find_contact_dimension(directions);

        // get runtime bc type
        auto bc_index = contact_dim * 2 + (contact_dir == Dir::E ? 1 : 0);
        auto bc_type  = state.metadata.boundary_conditions[bc_index];

        // process all coordinates in ghost region
        auto coord_count = ghost_region.size();
        auto coord_range = fp::range(coord_count);

        coord_range | fp::for_each([&](auto linear_idx) {
            auto ghost_coord = ghost_region.linear_to_coord(linear_idx);

            auto source_value = [&]() {
                switch (bc_type) {
                    case BoundaryCondition::REFLECTING: {
                        auto src_coord = reflect_coordinate(
                            ghost_coord,
                            contact_dim,
                            ghost_region,
                            active_domain
                        );
                        return state.cons[src_coord];
                    }
                    case BoundaryCondition::OUTFLOW: {
                        auto src_coord = clamp_to_boundary(
                            ghost_coord,
                            contact_dim,
                            ghost_region,
                            active_domain
                        );
                        return state.cons[src_coord];
                    }
                    case BoundaryCondition::PERIODIC: {
                        auto src_coord = wrap_coordinate(
                            ghost_coord,
                            contact_dim,
                            ghost_region,
                            active_domain
                        );
                        return state.cons[src_coord];
                    }
                    case BoundaryCondition::DYNAMIC: {
                        return evaluate_dynamic_bc(
                            ghost_coord,
                            contact_dim,
                            ghost_region,
                            active_domain,
                            state
                        );
                    }
                    default: return state.cons[ghost_coord];   // fallback
                }
            }();

            // apply value transformation
            auto final_value = [&]() {
                if (bc_type == BoundaryCondition::REFLECTING) {
                    if constexpr (HydroState::geometry_t ==
                                  Geometry::SPHERICAL) {
                        constexpr auto dims = HydroState::dimensions;
                        if (contact_dim == dims - 2) {
                            const auto theta_max =
                                state.mesh.current_bounds_max()[dims - 2];
                            if (helpers::goes_to_zero(
                                    theta_max - 0.5 * std::numbers::pi
                                ) &&
                                contact_dir == Dir::E) {
                                // if we're simulating a half sphere,
                                // them the theta momentum gets flipped
                                // at the equator
                                auto modified = source_value;
                                modified.mom[dims - 2] =
                                    -modified.mom[dims - 2];
                                return modified;
                            }
                            else {
                                return source_value;   // no flip
                            }
                        }
                        else {
                            // flip the momentum component
                            auto modified    = source_value;
                            auto mc          = (dims - 1) - contact_dim;
                            modified.mom[mc] = -modified.mom[mc];
                            return modified;
                        }
                    }
                    else {
                        auto modified = source_value;
                        auto mc = (HydroState::dimensions - 1) - contact_dim;
                        modified.mom[mc] = -modified.mom[mc];
                        return modified;
                    }
                }
                else {
                    return source_value;
                }
            }();

            state.cons[ghost_coord] = final_value;
        });
    }

    template <typename HydroState>
    auto apply_corner_boundary(
        auto corner_region,
        auto active_domain,
        HydroState& state
    )
    {
        auto directions = ghost_direction(corner_region, active_domain);

        // find all touching dimensions using your range utilities
        auto dim_range = fp::range(HydroState::dimensions);

        // collect touching dimensions into a compile-time container
        iarray<HydroState::dimensions> touching_dims_data;
        std::size_t touching_count = 0;

        for (auto dim : dim_range) {
            if (directions[dim] != direction_t::none) {
                touching_dims_data[touching_count++] = dim;
            }
        }

        auto coord_count = corner_region.size();
        auto coord_range = fp::range(coord_count);

        coord_range | fp::for_each([&](auto linear_idx) {
            auto corner_coord = corner_region.linear_to_coord(linear_idx);

            // map to interior cell
            auto interior_coord = corner_coord;
            for (std::size_t i = 0; i < touching_count; ++i) {
                auto dim = touching_dims_data[i];
                auto dir =
                    (directions[dim] == direction_t::plus) ? Dir::E : Dir::W;
                interior_coord[dim] = (dir == Dir::W)
                                          ? active_domain.start[dim]
                                          : active_domain.fin[dim] - 1;
            }

            auto base_cons = state.cons[interior_coord];

            // apply momentum flips for reflecting boundaries
            auto final_cons = base_cons;
            for (std::uint64_t ii = 0; ii < touching_count; ++ii) {
                auto dim = touching_dims_data[ii];
                auto dir =
                    (directions[dim] == direction_t::plus) ? Dir::E : Dir::W;
                auto bc_index = dim * 2 + (dir == Dir::E ? 1 : 0);
                auto bc_type  = state.metadata.boundary_conditions[bc_index];

                if (bc_type == BoundaryCondition::REFLECTING) {
                    auto mc            = (HydroState::dimensions - 1) - dim;
                    final_cons.mom[mc] = -final_cons.mom[mc];
                }
            }

            state.cons[corner_coord] = final_cons;
        });
    }

    // thin dimensions using your range infrastructure
    template <typename Field>
    void apply_thin_dimension_bc(
        Field& field,
        const std::vector<std::uint64_t>& thin_dims,
        std::uint64_t halo_radius
    )
    {
        auto full_domain = field.domain();
        auto interior_domain =
            contract_in_thin_dims(full_domain, thin_dims, halo_radius);

        auto coord_count = full_domain.size();
        auto coord_range = fp::range(coord_count);

        coord_range | fp::filter([&](auto linear_idx) {
            auto coord = full_domain.linear_to_coord(linear_idx);
            return !interior_domain.contains(coord);
        }) | fp::for_each([&](auto linear_idx) {
            auto ghost_coord = full_domain.linear_to_coord(linear_idx);

            // project to interior - check each dimension
            auto interior_coord = ghost_coord;
            for (auto thin_dim : thin_dims) {
                interior_coord[thin_dim] = interior_domain.start[thin_dim];
            }

            field[ghost_coord] = field[interior_coord];
        });
    }

    template <typename HydroState>
    auto apply_flux_boundary(
        auto ghost_region,
        auto active_domain,
        std::uint64_t flux_dim,
        HydroState& state,
        const std::vector<std::uint64_t>& cell_thin_dims
    )
    {
        auto directions = ghost_direction(ghost_region, active_domain);
        auto [contact_dim, contact_dir] = find_contact_dimension(directions);

        // note: symmetric copying vs real BCs
        bool use_symmetric_copying = std::find(
                                         cell_thin_dims.begin(),
                                         cell_thin_dims.end(),
                                         contact_dim
                                     ) != cell_thin_dims.end();

        auto coord_count = ghost_region.size();
        auto coord_range = fp::range(coord_count);

        coord_range | fp::for_each([&](auto linear_idx) {
            auto ghost_coord = ghost_region.linear_to_coord(linear_idx);

            if (use_symmetric_copying) {
                // symmetric copying for thin dimensions
                auto source_coord = ghost_coord;

                // clamp all thin dimensions to the center of the active
                // domain
                for (auto thin_dim : cell_thin_dims) {
                    source_coord[thin_dim] = active_domain.start[thin_dim];
                }
                state.flux[flux_dim][ghost_coord] =
                    state.flux[flux_dim][source_coord];
            }
            else {
                // real boundary conditions for physical boundaries
                auto bc_index =
                    contact_dim * 2 + (contact_dir == Dir::E ? 1 : 0);
                auto bc_type = state.metadata.boundary_conditions[bc_index];

                auto source_value = [&]() {
                    switch (bc_type) {
                        case BoundaryCondition::REFLECTING: {
                            auto src_coord = reflect_coordinate(
                                ghost_coord,
                                contact_dim,
                                ghost_region,
                                active_domain
                            );
                            return state.flux[flux_dim][src_coord];
                        }
                        case BoundaryCondition::OUTFLOW: {
                            auto src_coord = clamp_to_boundary(
                                ghost_coord,
                                contact_dim,
                                ghost_region,
                                active_domain
                            );
                            return state.flux[flux_dim][src_coord];
                        }
                        case BoundaryCondition::PERIODIC: {
                            auto src_coord = wrap_coordinate(
                                ghost_coord,
                                contact_dim,
                                ghost_region,
                                active_domain
                            );
                            return state.flux[flux_dim][src_coord];
                        }
                        default:
                            auto src_coord = clamp_to_boundary(
                                ghost_coord,
                                contact_dim,
                                ghost_region,
                                active_domain
                            );
                            return state.flux[flux_dim][src_coord];
                    }
                }();

                // apply momentum flip for reflecting BCs
                auto final_value = [&]() {
                    if (bc_type == BoundaryCondition::REFLECTING) {
                        auto modified = source_value;
                        // flip momentum component normal to the boundary
                        // (contact_dim)
                        auto momentum_index =
                            (HydroState::dimensions - 1) - contact_dim;
                        modified.mom[momentum_index] =
                            -modified.mom[momentum_index];
                        return modified;
                    }
                    return source_value;
                }();

                state.flux[flux_dim][ghost_coord] = final_value;
            }
        });
    }

}   // namespace simbi::hydro
#endif
