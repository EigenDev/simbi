#ifndef SIMBI_CFD_OPERATIONS_HPP
#define SIMBI_CFD_OPERATIONS_HPP

#include "core/base/stencil_view.hpp"
#include "core/utility/enums.hpp"
#include "physics/hydro/solvers/hllc.hpp"
#include "physics/hydro/solvers/hlld.hpp"
#include "physics/hydro/solvers/hlle.hpp"
#include <cstdint>

namespace simbi::cfd {
    using namespace stencils;
    // ============================================================================
    // FLUX COMPUTATION
    // ============================================================================

    template <
        typename prim_field_t,
        typename geometry_solver_t,
        Reconstruction Rec,
        Solver solver_t>
    struct compute_fluxes_t {
        const prim_field_t& primitives_;
        const geometry_solver_t& geom_;
        std::uint64_t dimension_;
        double plm_theta_;
        double gamma_;
        ShockWaveLimiter shock_smoother_;
        static constexpr bool apply_elementwise_ = true;

        compute_fluxes_t(
            const prim_field_t& prim,
            const geometry_solver_t& geom,
            std::uint64_t dim,
            double theta,
            double gamma,
            ShockWaveLimiter shock_limiter
        )
            : primitives_(prim),
              geom_(geom),
              dimension_(dim),
              plm_theta_(theta),
              gamma_(gamma),
              shock_smoother_(shock_limiter)
        {
        }

        template <typename Coord, typename Input>
        auto apply(Coord coord, const Input& /*input*/) const
        {
            using primitive_t = typename prim_field_t::value_type;
            // using conserved_t = typename primitive_t::counterpart_t;

            // get riemann solver based on compile-time parameters
            constexpr auto solver = get_riemann_solver<solver_t, primitive_t>();

            // create stencil for reconstruction around this face
            auto stencil = make_stencil<Rec>(primitives_, coord, dimension_);

            auto [left_states, right_states] = stencil.neighbor_values();

            // reconstruct left and right states at face
            auto left_prim  = reconstruct_left<Rec>(left_states, plm_theta_);
            auto right_prim = reconstruct_right<Rec>(right_states, plm_theta_);

            // normal vector for this dimension
            auto nhat = unit_vectors::canonical_basis<prim_field_t::dimensions>(
                dimension_ + 1
            );

            // face velocity
            auto vface = geom_.face_velocity(coord, dimension_);

            // solve riemann problem
            return solver(
                left_prim,
                right_prim,
                nhat,
                vface,
                gamma_,
                shock_smoother_
            );
        }

      private:
        template <Solver S, typename PrimType>
        constexpr static auto get_riemann_solver()
        {
            if constexpr (S == Solver::HLLC) {
                return hydro::hllc_flux<PrimType>;
            }
            else if constexpr (S == Solver::HLLE) {
                return hydro::hlle_flux<PrimType>;
            }
            else if constexpr (S == Solver::HLLD) {
                return hydro::rmhd::hlld_flux<PrimType>;
            }
            else {
                static_assert(
                    false,
                    "Unsupported solver type for the given primitive regime"
                );
            }
        }
    };

    // ============================================================================
    // FLUX DIVERGENCE - operates on pre-computed flux arrays
    // ============================================================================

    template <typename HydroState>
    struct flux_divergence_t {
        const HydroState& state_;
        double dt_;

        flux_divergence_t(const HydroState& state, double dt)
            : state_(state), dt_(dt)
        {
        }

        template <typename Coord, typename Input>
        auto apply(Coord coord, const Input& /*input*/) const
        {
            using conserved_t   = typename HydroState::conserved_t;
            constexpr auto dims = HydroState::dimensions;

            conserved_t divergence{};
            const auto dv = state_.geom_solver.volume(coord);

            // compute divergence using pre-computed fluxes
            for (std::uint64_t dim = 0; dim < dims; ++dim) {
                auto offset     = unit_vectors::canonical_basis<dims>(dim + 1);
                auto coord_plus = coord + offset;

                // flux values at left and right faces
                auto fl = state_.flux[dim][coord];
                auto fr = state_.flux[dim][coord_plus];

                // geometric face areas
                auto al = state_.geom_solver.face_area(coord, dim, Dir::W);
                auto ar = state_.geom_solver.face_area(coord, dim, Dir::E);

                // add contribution to divergence
                divergence = divergence + (fr * ar - fl * al) / dv;
            }

            return divergence * (-dt_);   // negative for conservative form
        }
    };

    // ============================================================================
    // SOURCE TERMS
    // ============================================================================

    template <typename HydroState>
    struct gravity_sources_t {
        const HydroState& state_;
        double dt_;

        gravity_sources_t(const HydroState& state, double dt)
            : state_(state), dt_(dt)
        {
        }

        template <typename Coord, typename Input>
        auto apply(Coord coord, const Input& /*input*/) const
        {
            if (!state_.sources.gravity_source.enabled) {
                return typename HydroState::conserved_t{};
            }

            auto position     = state_.geom_solver.centroid(coord);
            auto conservative = state_.cons[coord];

            return state_.sources.gravity_source
                .apply(position, conservative, state_.metadata.time, dt_);
        }
    };

    template <typename HydroState>
    struct hydro_sources_t {
        const HydroState& state_;
        double dt_;

        hydro_sources_t(const HydroState& state, double dt)
            : state_(state), dt_(dt)
        {
        }

        template <typename Coord, typename Input>
        auto apply(Coord coord, const Input& /*input*/) const
        {
            if (!state_.sources.hydro_source.enabled) {
                return typename HydroState::conserved_t{};
            }

            auto position  = state_.geom_solver.centroid(coord);
            auto primitive = state_.prim[coord];

            return state_.sources.hydro_source.apply(
                position,
                primitive,
                state_.metadata.time,
                state_.metadata.gamma
            );
        }
    };

    template <typename HydroState>
    struct geometric_sources_t {
        const HydroState& state_;
        double dt_;

        geometric_sources_t(const HydroState& state, double dt)
            : state_(state), dt_(dt)
        {
        }

        template <typename Coord, typename Input>
        auto apply(Coord coord, const Input& /*input*/) const
        {
            // geometric sources only exist for non-Cartesian geometries
            if constexpr (HydroState::geometry_t == Geometry::CARTESIAN) {
                return typename HydroState::conserved_t{};
            }
            else {
                auto primitive = state_.prim[coord];
                return state_.geom_solver.geometric_sources(
                           coord,
                           primitive,
                           state_.metadata.gamma
                       ) *
                       dt_;
            }
        }
    };

    // ============================================================================
    // FACTORY FUNCTIONS
    // ============================================================================

    template <
        Reconstruction Rec,
        Solver solver_t,
        typename prim_field_t,
        typename geometry_solver_t>
    auto compute_fluxes(
        const prim_field_t& primitives,
        const geometry_solver_t& geom,
        std::uint64_t dimension,
        double plm_theta,
        double gamma,
        ShockWaveLimiter shock_smoother
    )
    {
        return compute_fluxes_t<prim_field_t, geometry_solver_t, Rec, solver_t>{
          primitives,
          geom,
          dimension,
          plm_theta,
          gamma,
          shock_smoother
        };
    }

    template <typename HydroState>
    auto flux_divergence(const HydroState& state, double dt)
    {
        return flux_divergence_t<HydroState>{state, dt};
    }

    template <typename HydroState>
    auto gravity_sources(const HydroState& state, double dt)
    {
        return gravity_sources_t<HydroState>{state, dt};
    }

    template <typename HydroState>
    auto hydro_sources(const HydroState& state, double dt)
    {
        return hydro_sources_t<HydroState>{state, dt};
    }

    template <typename HydroState>
    auto geometric_sources(const HydroState& state, double dt)
    {
        return geometric_sources_t<HydroState>{state, dt};
    }

}   // namespace simbi::cfd

#endif   // SIMBI_CFD_OPERATIONS_HPP
