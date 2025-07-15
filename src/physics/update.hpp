#ifndef SIMBI_UPDATE_HPP
#define SIMBI_UPDATE_HPP

#include "bcs.hpp"
#include "compute/functional/monad/serializer.hpp"
#include "compute/math/domain.hpp"
#include "compute/math/expr.hpp"
#include "config.hpp"
#include "data/containers/vector.hpp"
#include "em/perm.hpp"
#include "physics/hydro/conversion.hpp"
#include "physics/hydro/wave_speeds.hpp"
#include "system/io/exceptions.hpp"
#include "system/mesh/mesh_ops.hpp"
#include <algorithm>
#include <cstdint>
#include <iostream>
#include <limits>

namespace simbi::hydro {
    using namespace simbi::expr;
    using namespace simbi::unit_vectors;
    using namespace simbi::set_ops;

    // forward declarations
    template <typename HydroState>
    void recover_primitives(HydroState& state);

    template <typename HydroState>
    void update_timestep(HydroState& state);

    template <typename HydroState>
    real compute_local_timestep(
        const iarray<HydroState::dimensions>& coord,
        const HydroState& state
    );

    template <typename HydroState>
    void compute_all_fluxes(HydroState& state);

    template <typename HydroState>
    void update_conservatives(HydroState& state);

    template <typename HydroState>
    void timestep(HydroState& state);

    template <typename HydroState>
    void print_failure_diagnostics(const HydroState& state);

    // ============================================================================
    // CONSERVATIVE TO PRIMITIVE CONVERSION
    // ============================================================================

    template <typename HydroState>
    void recover_primitives(HydroState& state)
    {
        error_budget_t error_budget{};
        error_budget.reset();
        for (std::uint64_t ii = 0; ii < state.mesh.full_domain.size(); ++ii) {
            const auto coord      = state.mesh.full_domain.linear_to_coord(ii);
            const auto& cons      = state.cons[coord];
            const auto maybe_prim = to_primitive(cons, state.metadata.gamma);

            // early exit if budget exhausted (other thread found error)
            if (error_budget.is_exhausted()) {
                continue;
            }

            state.prim[coord] = maybe_prim.unwrap_or_else([&]() {
                if (error_budget.try_consume()) {
                    // this thread wins - capture detailed error info
                    error_budget.first_error_code.store(
                        maybe_prim.error_code()
                    );
                    error_budget.first_error_index.store(ii);
                    error_budget.error_captured.store(true);
                }
                // all other threads: continue until they see budget exhausted
                return typename HydroState::primitive_t{};
            });
        }

        if (error_budget.error_captured.load()) {
            // construct error context and throw
            auto error_index = error_budget.first_error_index.load();
            auto error_coord =
                state.mesh.full_domain.linear_to_coord(error_index);
            auto error_code = error_budget.first_error_code.load();

            throw primitive_conversion_error_t{error_info_t{
              .coord_str = format_coord(error_coord),
              .position_str =
                  format_position(mesh::centroid(error_coord, state.mesh)),
              .error_code       = error_code,
              .time             = state.metadata.time,
              .iteration        = state.metadata.iteration,
              .conservative_str = format_conserved(state.cons[error_coord]),
              .message          = "Primitives in non-physical state."
            }};
        }
    }
    // ============================================================================
    // ADAPTIVE TIMESTEP CALCULATION
    // ============================================================================

    template <typename HydroState>
    void update_timestep(HydroState& state)
    {
        real min_dt = std::numeric_limits<real>::max();

        for (std::uint64_t ii = 0; ii < state.mesh.full_domain.size(); ++ii) {
            auto coord = state.mesh.full_domain.linear_to_coord(ii);

            // compute local timestep at this coordinate
            auto local_dt = compute_local_timestep(coord, state);

            // track minimum across all cells
            min_dt = std::min(min_dt, local_dt);
        }

        // update state metadata
        state.metadata.dt = min_dt;
    }

    template <typename HydroState>
    real compute_local_timestep(
        const iarray<HydroState::dimensions>& coord,
        const HydroState& state
    )
    {
        const auto& prim = state.prim[coord];
        auto cell_widths = mesh::cell_widths(coord, state.mesh);

        real min_dt = std::numeric_limits<real>::max();

        // check wave speeds in each spatial dimension
        for (std::uint64_t dim = 0; dim < HydroState::dimensions; ++dim) {
            // unit vector in this dimension
            auto nhat = ehat<HydroState::dimensions>(dim);

            // compute characteristic wave speeds
            auto speeds = wave_speeds(prim, nhat, state.metadata.gamma);

            // maximum wave speed in this direction
            auto max_speed =
                std::max(std::abs(speeds.left), std::abs(speeds.right));

            // CFL timestep constraint for this dimension
            if (max_speed > 0.0) {
                real dt_dim = state.metadata.cfl * cell_widths[dim] / max_speed;
                min_dt      = std::min(min_dt, dt_dim);
            }
        }

        return min_dt;
    }

    // ============================================================================
    // FLUX COMPUTATION AND CONSERVATIVE UPDATE
    // ============================================================================
    template <typename HydroState>
    void compute_all_fluxes(HydroState& state)
    {
        constexpr auto dims = HydroState::dimensions;
        const auto& mesh    = state.mesh;
        for (std::uint64_t dir = 0; dir < dims; ++dir) {
            const auto domain       = active_staggered_domain(mesh.domain, dir);
            state.flux[dir][domain] = compute_fluxes(state, dir);
        }

        if constexpr (HydroState::is_mhd) {
            // if we're doing MHD, then we need to make sure
            // that quantities that are staggered (i.e, the fluxes)
            // correctly have the perpendicular boundary conditions applied
            // since we do not save the edge-centered electric fields
            // but rather compute them on-the-fly,
            apply_staggered_boundary_conditions(state);
        }
    }

    template <typename HydroState>
    void update_conservatives(HydroState& state)
    {
        const auto& mesh         = state.mesh;
        const auto active_domain = mesh.domain;
        auto u                   = state.cons[active_domain];
        const real dt            = state.metadata.dt;
        // u^{n+1} = u^n + L(u^n)
        u += flux_divergence(state, dt) + gravity_sources(state, dt) +
             hydro_sources(state, dt) + geometric_sources(state, dt);

        if constexpr (HydroState::is_mhd) {
            // correct the energy density for MHD
            em::update_energy_density(state);
        }
    }

    // ============================================================================
    // COMPLETE TIMESTEP
    // ============================================================================
    template <typename HydroState>
    void timestep(HydroState& state)
    {
        compute_all_fluxes(state);
        if constexpr (HydroState::is_mhd) {
            // update magnetic fields using CT algorithm
            em::update_magnetic_fields(state);
        }
        update_conservatives(state);
        apply_boundary_conditions(state);
        recover_primitives(state);
        update_timestep(state);
        state.metadata.time += state.metadata.dt;
        state.metadata.iteration++;
    }

    // ============================================================================
    // SIMULATION LOOP
    // ============================================================================

    template <typename HydroState>
    void run_simulation(HydroState& state)
    {
        const auto t_final = state.metadata.tend;
        std::cout << "Starting simulation...\n";
        std::cout << "Initial time: " << state.metadata.time << "\n";
        std::cout << "Final time: " << t_final << "\n";

        // initialize timestep
        apply_boundary_conditions(state);
        recover_primitives(state);
        update_timestep(state);

        real tinterval = 0.0;
        // now we can start the simulation loop :D
        while (state.metadata.time < t_final && !state.in_failure_state) {
            timestep(state);
            state.mesh = mesh::update_mesh(
                state.mesh,
                state.metadata.time,
                state.metadata.dt
            );

            if (state.metadata.time >= tinterval) {
                tinterval += 0.01;
                std::cout << "Iteration " << state.metadata.iteration
                          << ", time = " << state.metadata.time
                          << ", dt = " << state.metadata.dt << "\n";
                io::serialize_hydro_state(state);
            }
        }

        std::cout << "Simulation completed!\n";
        std::cout << "Final time: " << state.metadata.time << "\n";
        std::cout << "Total iterations: " << state.metadata.iteration << "\n";
    }

}   // namespace simbi::hydro

#endif   // SIMBI_UPDATE_HPP
