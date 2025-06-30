#ifndef SIMBI_BOUNDARY_CONDITIONS_HPP
#define SIMBI_BOUNDARY_CONDITIONS_HPP

#include "compute/math/index_space.hpp"
#include "core/utility/enums.hpp"
#include <cstdint>

namespace simbi::hydro {

    // ============================================================================
    // HELPER FUNCTIONS
    // ============================================================================

    template <typename HydroState>
    auto get_full_domain_with_ghosts(const HydroState& state)
    {
        // expand active domain to include ghost cells
        return state.domain().expand(state.metadata.halo_radius);
    }

    template <std::uint64_t Dims>
    auto get_boundary_region(
        const index_space_t<Dims>& full_domain,
        std::uint64_t dim,
        Dir side,
        std::uint64_t halo_radius
    )
    {

        // auto shape         = full_domain.shape();
        uarray<Dims> start = full_domain.start();
        uarray<Dims> end   = full_domain.end();

        if (side == Dir::W) {
            // inner boundary - first halo_radius slices in dimension dim
            end[dim] = start[dim] + halo_radius;
        }
        else {
            // outer boundary - last halo_radius slices in dimension dim
            start[dim] = end[dim] - halo_radius;
        }

        return index_space_t<Dims>{start, end};
    }

    // ============================================================================
    // REFLECTING BOUNDARY CONDITIONS
    // ============================================================================

    template <typename Field, std::uint64_t Dims, typename HydroState>
    void apply_reflecting_bc(
        Field& cons_field,
        const index_space_t<Dims>& boundary_region,
        std::uint64_t dim,
        Dir side,
        HydroState& state
    )
    {

        for (std::uint64_t ii = 0; ii < boundary_region.size(); ++ii) {
            auto boundary_coord = boundary_region.index_to_coord(ii);

            // find corresponding interior cell (mirror across boundary)
            auto interior_coord = boundary_coord;
            if (side == Dir::W) {
                // inner boundary - reflect across the left edge
                auto active_start   = state.domain().start()[dim];
                auto offset         = active_start - boundary_coord[dim];
                interior_coord[dim] = active_start + offset - 1;
            }
            else {
                // outer boundary - reflect across the right edge
                auto active_end     = state.domain().end()[dim];
                auto offset         = boundary_coord[dim] - active_end;
                interior_coord[dim] = active_end - offset - 1;
            }

            // copy conserved variables from interior cell
            auto reflected_cons = cons_field[interior_coord];

            // flip momentum component in the reflection direction
            reflected_cons.mom[dim] = -reflected_cons.mom[dim];

            // set boundary cell
            cons_field[boundary_coord] = reflected_cons;
        }
    }

    // ============================================================================
    // OUTFLOW BOUNDARY CONDITIONS
    // ============================================================================

    template <typename Field, std::uint64_t Dims, typename HydroState>
    void apply_outflow_bc(
        Field& cons_field,
        const index_space_t<Dims>& boundary_region,
        std::uint64_t dim,
        Dir side,
        HydroState& state
    )
    {

        for (std::uint64_t ii = 0; ii < boundary_region.size(); ++ii) {
            auto boundary_coord = boundary_region.index_to_coord(ii);

            // find nearest interior cell
            auto interior_coord = boundary_coord;
            if (side == Dir::W) {
                // inner boundary - copy from first active cell
                interior_coord[dim] = state.domain().start()[dim];
            }
            else {
                // outer boundary - copy from last active cell
                interior_coord[dim] = state.domain().end()[dim] - 1;
            }

            // simple copy (zero-gradient extrapolation)
            cons_field[boundary_coord] = cons_field[interior_coord];
        }
    }

    // ============================================================================
    // PERIODIC BOUNDARY CONDITIONS
    // ============================================================================

    template <typename Field, std::uint64_t Dims, typename HydroState>
    void apply_periodic_bc(
        Field& cons_field,
        const index_space_t<Dims>& boundary_region,
        std::uint64_t dim,
        Dir side,
        HydroState& state
    )
    {

        for (std::uint64_t ii = 0; ii < boundary_region.size(); ++ii) {
            auto boundary_coord = boundary_region.index_to_coord(ii);

            // find corresponding cell on opposite boundary
            auto source_coord = boundary_coord;
            if (side == Dir::W) {
                // inner boundary gets data from outer active cells
                auto offset = state.domain().start()[dim] - boundary_coord[dim];
                source_coord[dim] = state.domain().end()[dim] - offset;
            }
            else {
                // outer boundary gets data from inner active cells
                auto offset =
                    boundary_coord[dim] - state.domain().end()[dim] + 1;
                source_coord[dim] = state.domain().start()[dim] + offset - 1;
            }

            // copy from opposite boundary
            cons_field[boundary_coord] = cons_field[source_coord];
        }
    }

    // ============================================================================
    // DYNAMIC BOUNDARY CONDITIONS (using expression system)
    // ============================================================================

    template <typename Field, std::uint64_t Dims, typename HydroState>
    void apply_dynamic_bc(
        Field& cons_field,
        const index_space_t<Dims>& boundary_region,
        std::uint64_t dim,
        Dir side,
        HydroState& state
    )
    {
        // get the appropriate boundary condition expression
        auto bc_index       = 2 * dim + (side == Dir::E ? 1 : 0);
        const auto& bc_expr = state.sources.bc_sources[bc_index];

        if (!bc_expr.enabled) {
            // if no expression is provided, fall back to outflow
            apply_outflow_bc(cons_field, boundary_region, dim, side, state);
            return;
        }

        for (std::uint64_t ii = 0; ii < boundary_region.size(); ++ii) {
            auto boundary_coord = boundary_region.index_to_coord(ii);

            // get physical position of boundary cell
            auto position = state.geom_solver.centroid(boundary_coord);

            // evaluate boundary condition expression
            auto current_cons = cons_field[boundary_coord];
            auto bc_value     = bc_expr.apply(
                position,
                current_cons,
                state.metadata.time,
                state.metadata.dt
            );

            // set boundary value from expression result
            cons_field[boundary_coord] = bc_value;
        }
    }

    // ============================================================================
    // BOUNDARY CONDITION DISPATCH
    // ============================================================================

    template <typename Field, std::uint64_t Dims, typename HydroState>
    void apply_bc_to_region(
        Field& cons_field,
        const index_space_t<Dims>& boundary_region,
        BoundaryCondition bc_type,
        std::uint64_t dim,
        Dir side,
        HydroState& state
    )
    {

        switch (bc_type) {
            case BoundaryCondition::REFLECTING:
                apply_reflecting_bc(
                    cons_field,
                    boundary_region,
                    dim,
                    side,
                    state
                );
                break;
            case BoundaryCondition::OUTFLOW:
                apply_outflow_bc(cons_field, boundary_region, dim, side, state);
                break;
            case BoundaryCondition::PERIODIC:
                apply_periodic_bc(
                    cons_field,
                    boundary_region,
                    dim,
                    side,
                    state
                );
                break;
            case BoundaryCondition::DYNAMIC:
                apply_dynamic_bc(cons_field, boundary_region, dim, side, state);
                break;
        }
    }

    // ============================================================================
    // BOUNDARY CONDITION IMPLEMENTATION
    // ============================================================================

    template <typename HydroState>
    void apply_boundary_conditions(HydroState& state)
    {
        // get full domain including ghost cells
        auto full_domain = get_full_domain_with_ghosts(state);

        // apply boundary conditions for each dimension and direction
        for (std::uint64_t dim = 0; dim < HydroState::dimensions; ++dim) {
            // inner boundary (e.g., x=0 side)
            auto inner_bc     = state.metadata.boundary_conditions[2 * dim];
            auto inner_region = get_boundary_region(
                full_domain,
                dim,
                Dir::W,
                state.metadata.halo_radius
            );
            apply_bc_to_region(
                state.cons,
                inner_region,
                inner_bc,
                dim,
                Dir::W,
                state
            );

            // outer boundary (e.g., x=max side)
            auto outer_bc     = state.metadata.boundary_conditions[2 * dim + 1];
            auto outer_region = get_boundary_region(
                full_domain,
                dim,
                Dir::E,
                state.metadata.halo_radius
            );
            apply_bc_to_region(
                state.cons,
                outer_region,
                outer_bc,
                dim,
                Dir::E,
                state
            );
        }
    }

}   // namespace simbi::hydro

#endif   // SIMBI_BOUNDARY_CONDITIONS_HPP
