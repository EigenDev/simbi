#ifndef SIMBI_HYDRO_UPDATE_HPP
#define SIMBI_HYDRO_UPDATE_HPP

#include "config.hpp"
#include "core/containers/array.hpp"
#include "core/containers/vector.hpp"
#include "core/index/global_index.hpp"
#include "core/index/stagger_patterns.hpp"
#include "core/index/stencil_patterns.hpp"
#include "core/memory/values/state_value.hpp"
#include "core/memory/views/state_tile.hpp"
#include "core/memory/views/state_view.hpp"
#include "core/parallel/domain.hpp"
#include "core/parallel/executor.hpp"
#include "core/parallel/pattern.hpp"
#include "core/parallel/view.hpp"
#include "core/state/hydro_state.hpp"
#include "core/types/alias/alias.hpp"
#include "core/utility/enums.hpp"
#include "geometry/mesh/cell.hpp"
#include "physics/eos/ideal.hpp"
#include "physics/hydro/conversion.hpp"
#include "physics/hydro/solvers/hllc.hpp"
#include "physics/hydro/wave_speeds.hpp"
#include "util/tools/helpers.hpp"
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iostream>

namespace simbi::hydro {
    using namespace index;
    using namespace unit_vectors;
    using namespace parallel;

    // function type aliases for composable operations
    template <size_type Dims>
    using riemann_solver_t =
        std::function<conserved_value_t<Regime::NEWTONIAN, Dims>(
            const primitive_value_t<Regime::NEWTONIAN, Dims>&,
            const primitive_value_t<Regime::NEWTONIAN, Dims>&,
            const unit_vector_t<Dims>&,
            real,              // face velocity
            real,              // gamma
            ShockWaveLimiter   // shock smoother
        )>;

    // select the appropriate Riemann solver based on solver_type
    template <size_type Dims>
    riemann_solver_t<Dims> get_riemann_solver(Solver solver_type)
    {
        switch (solver_type) {
            case Solver::HLLC:
                return newtonian::hllc_flux<
                    primitive_value_t<Regime::NEWTONIAN, Dims>>;
            case Solver::HLLE:
                return hlle_flux<primitive_value_t<Regime::NEWTONIAN, Dims>>;
            // Add other solvers as needed
            default:
                return newtonian::hllc_flux<
                    primitive_value_t<Regime::NEWTONIAN, Dims>>;
        }
    }

    template <Regime R, size_type Dims, typename EoS = ideal_gas_eos_t<R>>
    void advance_state(state::hydro_state_t<R, Dims, EoS>& state, real dt)
    {
        // extract key state information
        auto& cons           = state.cons;
        auto& prim           = state.prim;
        auto& flux           = state.flux;
        const auto& mesh     = state.mesh;
        const auto& metadata = state.metadata;

        // get the appropriate Riemann solver
        auto riemann_solver = get_riemann_solver<Dims>(metadata.solver);

        // create computation domain
        auto interior_domain   = domain_t<Dims>(mesh.active_dimensions());
        auto entire_domain     = domain_t<Dims>(mesh.dimensions());
        auto xstaggered_domain = domain_t<Dims>(mesh.xstaggered_shape());
        auto ystaggered_domain = domain_t<Dims>(mesh.ystaggered_shape());
        auto zstaggered_domain = domain_t<Dims>(mesh.zstaggered_shape());
        array_t<domain_t<Dims>, 3> stagg_domains =
            {xstaggered_domain, ystaggered_domain, zstaggered_domain};

        // create executor with appropriate strategy
        auto exec = executor_t<Dims>();

        // create views for the state arrays
        auto cons_view  = make_view(cons, entire_domain);
        auto prim_view  = make_view(prim, entire_domain);
        auto flux_views = array_t<data_view_t<real, Dims>, Dims>{};
        for (size_type d = 0; d < Dims; ++d) {
            flux_views[d] = make_view(flux[d], stagg_domains[d]);
        }

        for (size_type d = 0; d < Dims; ++d) {
            exec.apply_to_container(
                // flux_views[d],
                state.prim,
                xstaggered_domain,
                [&](const auto& prim_tile, const auto& pos) {
                    std::cout << "Processing position: " << pos << std::endl;
                    // get cell index from position
                    cell_index_t idx{pos};
                    const auto mesh_cell = mesh.from_position(idx);

                    // loop over dimensions for flux calculation
                    helpers::compile_time_for<0, Dims>([&](auto dim_const) {
                        constexpr size_type id = decltype(dim_const)::value;
                        constexpr auto dir     = static_cast<direction_t>(id);
                        const auto [prim_L, prim_R] = views::
                            primitive_stencil<Regime::NEWTONIAN, Dims, dir>(
                                prim_tile,
                                idx,
                                state.mesh.size()
                            );

                        auto flux = riemann_solver(
                            prim_L,
                            prim_R,
                            unit_vectors::canonical_basis<Dims>(id),
                            mesh_cell.velocity(2 * d),
                            metadata.gamma,
                            metadata.shock_smoother
                        );

                        // std::cout << "Flux: " << flux.den << ", " << flux.mom
                        //           << ", " << flux.nrg << std::endl;
                    });
                },
                pattern_t<Dims>::von_neumann(state.metadata.halo_radius)
            );
        }

        // update metadata
        state.metadata.time += dt;
        state.metadata.dt = dt;
        state.metadata.iteration++;
    }
}   // namespace simbi::hydro

#endif   // SIMBI_HYDRO_UPDATE_HPP
