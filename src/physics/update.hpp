#ifndef SIMBI_PHYSICS_UPDATE_HPP
#define SIMBI_PHYSICS_UPDATE_HPP

#include "compute/functional/fp.hpp"
#include "compute/math/field.hpp"
#include "compute/math/index_space.hpp"
#include "compute/math/lazy_expr.hpp"
#include "config.hpp"
#include "core/base/concepts.hpp"
#include "core/base/coordinate.hpp"
#include "core/base/stencil.hpp"
#include "core/base/stencil_view.hpp"
#include "core/utility/enums.hpp"
#include "core/utility/helpers.hpp"
#include "data/containers/state_struct.hpp"
#include "data/containers/vector.hpp"
#include "data/state/hydro_state.hpp"
#include "data/state/hydro_state_types.hpp"
#include "physics/hydro/solvers/hllc.hpp"
#include "physics/hydro/solvers/hlld.hpp"
#include "system/mesh/solver.hpp"
#include <cstddef>
#include <cstdint>

namespace simbi::hydro {
    using namespace simbi::base;
    using namespace simbi::concepts;
    using namespace simbi::stencils;

    template <Regime R, Solver S, is_hydro_primitive_c prim_t>
    constexpr auto get_solver()
    {
        if constexpr (R == Regime::NEWTONIAN) {
            if constexpr (S == Solver::HLLC) {
                return newtonian::hllc_flux<prim_t>;
            }
            else if constexpr (S == Solver::HLLE) {
                return hlle_flux<prim_t>;
            }
            else {
                static_assert(
                    false,
                    "Unsupported solver for the specified regime and primitive "
                    "type"
                );
            }
        }
        else if constexpr (R == Regime::SRHD) {
            if constexpr (S == Solver::HLLC) {
                return srhd::hllc_flux<prim_t>;
            }
            else if constexpr (S == Solver::HLLE) {
                return hlle_flux<prim_t>;
            }
            else {
                static_assert(
                    false,
                    "Unsupported solver for the specified regime and primitive "
                    "type"
                );
            }
        }
        else if constexpr (R == Regime::RMHD) {
            if constexpr (S == Solver::HLLC) {
                return rmhd::hllc_flux<prim_t>;
            }
            else if constexpr (S == Solver::HLLE) {
                return hlle_flux<prim_t>;
            }
            else if constexpr (S == Solver::HLLD) {
                return rmhd::hlld_flux<prim_t>;
            }
            else {
                static_assert(
                    false,
                    "Unsupported solver for the specified regime and primitive "
                    "type"
                );
            }
        }
        else {
            static_assert(false, "Unsupported regime for hydro solver");
        }
    }

    template <typename HydroState>
    void compute_fluxes(HydroState& state)
    {
        constexpr auto rec_order = HydroState::reconstruct_t;
        constexpr auto dims      = HydroState::dimensions;
        constexpr auto nghosts   = reconstruction_to_ghosts(rec_order);

        const auto& mesh_config = state.geom_solver.config;
        const auto plm_theta    = state.metadata.plm_theta;
        const auto& prims       = state.prim;
        const auto domain       = make_space<dims>(mesh_config.shape);

        constexpr auto riemann_solver = get_solver<
            HydroState::regime_t,
            HydroState::solver_t,
            typename HydroState::primitive_t>();

        compile_time_for<0, dims>([&](auto dim) {
            constexpr auto dir     = decltype(dim)::value;
            const auto flux_domain = [&]() {
                uarray<dims> expansion{};
                expansion[dir] = 1;
                return domain.expand_end(expansion);
            }();
            const auto p        = prims.contract(nghosts);
            const auto gamma    = state.metadata.gamma;
            const auto nhat     = unit_vectors::canonical_basis<dims>(dir + 1);
            const auto smoother = state.metadata.shock_smoother;

            flux_domain
                .map([=] DEV(auto coord) {
                    const auto stencil = make_stencil<rec_order>(p, coord, dir);
                    const auto [left, right] = stencil.neighbor_values();
                    const auto pl =
                        reconstruct_left<rec_order>(left, plm_theta);
                    const auto pr =
                        reconstruct_right<rec_order>(right, plm_theta);
                    const auto vface =
                        mesh::face_velocity(coord, dir, mesh_config);

                    return riemann_solver(pl, pr, nhat, vface, gamma, smoother);
                })
                .realize_into(state.flux[dir]);
        });
    }

    template <typename HydroState>
    auto compute_flux_difference_for_dimension(
        const HydroState& state,
        std::uint64_t dim
    )
    {
        constexpr auto dims     = HydroState::dimensions;
        const auto& mesh_config = state.geom_solver.config;
        const auto domain       = make_space<dims>(mesh_config.shape);
        const auto dt           = 0.01;   // Get from somewhere appropriate

        return domain.map([=, &state] DEV(auto coord) {
            const auto dv  = state.geom_solver.volume(coord);
            const auto al  = state.geom_solver.face_area(coord, dim, Dir::W);
            const auto ar  = state.geom_solver.face_area(coord, dim, Dir::E);
            const auto off = unit_vectors::canonical_basis<dims>(dim + 1);
            const auto& fl = state.flux[dim][coord /***/];
            const auto& fr = state.flux[dim][coord + off];
            return -dt * (fr * ar - fl * al) / dv;
        });
    }

    template <typename HydroState>
    void update_state(HydroState& state)
    {
        constexpr auto dims     = HydroState::dimensions;
        const auto& geo         = state.geom_solver;
        const auto& mesh_config = geo.config;
        const auto domain       = make_space<dims>(mesh_config.shape);
        // const auto& srcs        = state.sources;
        auto u = state.cons[domain];
        // const auto& p           = state.prim[domain];

        const auto du =
            expr::make_flux_accumulator(domain, [&](std::uint64_t dim) {
                return compute_flux_difference_for_dimension(state, dim);
            });

        const auto dt = state.metadata.dt;
        // source contributions
        du += expr::make_source<gravity_source_tag>(state) * dt;
        du += expr::make_source<hydro_source_tag>(state) * dt;
        du += expr::make_source<ib_source_tag>(state);

        const auto unext = u + du;

        for (const auto& x : unext.realize()) {
            std::cout << x << std::endl;
        }

        // u = u + du + source;
        // u.realize();
    }
}   // namespace simbi::hydro
#endif
