#ifndef SIMBI_UPDATE_HPP
#define SIMBI_UPDATE_HPP

#include "bcs.hpp"
#include "compute/functional/monad/serializer.hpp"
#include "compute/math/cfd_expressions.hpp"
#include "compute/math/cfd_operations.hpp"
#include "data/containers/vector.hpp"
#include "physics/hydro/conversion.hpp"
#include "physics/hydro/wave_speeds.hpp"
#include "system/io/exceptions.hpp"
#include <algorithm>
#include <cstdint>
#include <iostream>
#include <limits>

namespace simbi::hydro {
    using namespace simbi::cfd;
    using namespace simbi::unit_vectors;

    // forward declarations
    template <typename HydroState>
    void recover_primitives(HydroState& state);

    template <typename HydroState>
    void update_timestep(HydroState& state);

    template <typename HydroState>
    double compute_local_timestep(
        const uarray<HydroState::dimensions>& coord,
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
        for (std::uint64_t ii = 0; ii < state.domain().size(); ++ii) {
            const auto coord      = state.domain().index_to_coord(ii);
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
            // construct rich error context and throw
            auto error_index = error_budget.first_error_index.load();
            auto error_coord = state.domain().index_to_coord(error_index);
            auto error_code  = error_budget.first_error_code.load();
            const auto& geo  = state.geom_solver;

            throw primitive_conversion_error_t{error_info_t{
              .coord_str        = format_coord(error_coord),
              .position_str     = format_position(geo.centroid(error_coord)),
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

        double min_dt = std::numeric_limits<double>::max();

        for (std::uint64_t ii = 0; ii < state.domain().size(); ++ii) {
            auto coord = state.domain().index_to_coord(ii);

            // compute local timestep at this coordinate
            auto local_dt = compute_local_timestep(coord, state);

            // track minimum across all cells
            min_dt = std::min(min_dt, local_dt);
        }

        // update state metadata
        state.metadata.dt = min_dt;
    }

    template <typename HydroState>
    double compute_local_timestep(
        const uarray<HydroState::dimensions>& coord,
        const HydroState& state
    )
    {

        const auto& prim = state.prim[coord];
        auto cell_widths = state.geom_solver.cell_widths(coord);

        double min_dt = std::numeric_limits<double>::max();

        // Check wave speeds in each spatial dimension
        for (std::uint64_t dim = 0; dim < HydroState::dimensions; ++dim) {

            // Unit vector in this dimension
            auto nhat = canonical_basis<HydroState::dimensions>(dim + 1);

            // Compute characteristic wave speeds
            auto speeds = wave_speeds(prim, nhat, state.metadata.gamma);

            // Maximum wave speed in this direction
            auto max_speed =
                std::max(std::abs(speeds.left), std::abs(speeds.right));

            // CFL timestep constraint for this dimension
            if (max_speed > 0.0) {
                double dt_dim =
                    state.metadata.cfl * cell_widths[dim] / max_speed;
                min_dt = std::min(min_dt, dt_dim);
            }
        }

        return min_dt;
    }

    // ============================================================================
    // FLUX COMPUTATION AND CONSERVATIVE UPDATE (using your expression system)
    // ============================================================================

    template <typename HydroState>
    void compute_all_fluxes(HydroState& state)
    {

        // Compute fluxes for each dimension using lazy expressions
        for (std::uint64_t dim = 0; dim < HydroState::dimensions; ++dim) {

            // Get flux domain (extends one cell in this dimension)
            uarray<HydroState::dimensions> expansion{};
            expansion[dim]   = 1;
            auto flux_domain = state.domain().expand_end(expansion);

            // Compute fluxes lazily and store in state.flux[dim]
            flux_domain |
                compute_fluxes<HydroState::reconstruct_t, HydroState::solver_t>(
                    state.prim[state.domain()],
                    state.geom_solver,
                    dim,
                    state.metadata.plm_theta,
                    state.metadata.gamma,
                    state.metadata.shock_smoother
                ) |
                realize_to(state.flux[dim]);
        }
    }

    template <typename HydroState>
    void update_conservatives(HydroState& state)
    {
        auto u = state.cons[state.domain()];
        state.domain() | flux_divergence(state, state.metadata.dt) |
            add(gravity_sources(state, state.metadata.dt)) |
            add(hydro_sources(state, state.metadata.dt)) |
            add(geometric_sources(state, state.metadata.dt)) |
            add_to(u);   // u^{n+1} = u^n + L(u^n)
    }

    // ============================================================================
    // COMPLETE LAZY TIMESTEP IMPLEMENTATION
    // ============================================================================
    template <typename HydroState>
    void timestep(HydroState& state)
    {
        compute_all_fluxes(state);
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
        update_timestep(state);
        apply_boundary_conditions(state);
        recover_primitives(state);

        // now we can start the simulation loop :D
        while (state.metadata.time < t_final && !state.in_failure_state) {

            timestep(state);
            state.geom_solver.expand_mesh(
                state.metadata.time,
                state.metadata.dt
            );

            if (state.metadata.iteration % 100 == 0) {
                static int idx = 0;
                std::cout << "Iteration " << state.metadata.iteration
                          << ", time = " << state.metadata.time
                          << ", dt = " << state.metadata.dt << "\n";
                io::serialize_hydro_state(
                    state,
                    std::string("sod" + std::to_string(idx) + ".h5")
                );
                idx++;
            }
        }

        std::cout << "Simulation completed!\n";
        std::cout << "Final time: " << state.metadata.time << "\n";
        std::cout << "Total iterations: " << state.metadata.iteration << "\n";
    }

    // ============================================================================
    // UTILITY FUNCTIONS
    // ============================================================================

    template <typename HydroState>
    bool is_simulation_stable(const HydroState& state)
    {
        // Check for NaN/inf values, negative densities, etc.
        for (std::uint64_t i = 0; i < state.domain().size(); ++i) {
            auto coord       = state.domain().index_to_coord(i);
            const auto& prim = state.prim[coord];

            if (prim.rho <= 0.0 || !std::isfinite(prim.rho)) {
                return false;
            }
            if (prim.pre <= 0.0 || !std::isfinite(prim.pre)) {
                return false;
            }
            // Could add more checks for velocity, magnetic field, etc.
        }
        return true;
    }

    template <typename HydroState>
    void print_simulation_summary(const HydroState& state)
    {
        std::cout << "\n=== Simulation Summary ===\n";
        std::cout << "Regime: " << static_cast<int>(HydroState::regime_t)
                  << "\n";
        std::cout << "Dimensions: " << HydroState::dimensions << "\n";
        std::cout << "Geometry: " << static_cast<int>(HydroState::geometry_t)
                  << "\n";
        std::cout << "Solver: " << static_cast<int>(HydroState::solver_t)
                  << "\n";
        std::cout << "Reconstruction: "
                  << static_cast<int>(HydroState::reconstruct_t) << "\n";
        std::cout << "Final time: " << state.metadata.time << "\n";
        std::cout << "Iterations: " << state.metadata.iteration << "\n";
        std::cout << "Final dt: " << state.metadata.dt << "\n";
        std::cout << "Stable: " << (is_simulation_stable(state) ? "Yes" : "No")
                  << "\n";
        std::cout << "========================\n\n";
    }

}   // namespace simbi::hydro

#endif   // SIMBI_UPDATE_HPP
