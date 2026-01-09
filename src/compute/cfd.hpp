#ifndef CFD_OPS_HPP
#define CFD_OPS_HPP

// =============================================================================
// cfd.hpp
//
// cfd operations using block_geometry_t instead of mesh_config.
// all operations are lazy computations that compose with the field algebra.
// =============================================================================

#include "base/stencil_view.hpp"
#include "compat.hpp"
#include "compute/computation.hpp"
#include "containers/state_ops.hpp"
#include "containers/vector.hpp"
#include "grid/domain.hpp"
#include "io/exceptions.hpp"
#include "physics/ib/body.hpp"
#include "physics/ib/body_delta.hpp"
#include "physics/ib/diagnostics.hpp"
#include "physics/ib/effects.hpp"
#include "utility/enums.hpp"

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace simbi::cfd {
    using namespace base::stencils;
    using namespace body::expr;
    using namespace body;

    // =========================================================================
    // geometry concept
    // any type that provides the required geometric operations
    // =========================================================================
    template <typename G, std::uint64_t Rank>
    concept block_geometry_c = requires(const G& geo, const iarray<Rank>& idx, std::size_t dim) {
        { geo.volume(idx) } -> std::convertible_to<real>;
        { geo.face_area(idx, dim) } -> std::convertible_to<real>;
        { geo.centroid(idx) } -> std::convertible_to<vector_t<real, Rank>>;
        { geo.scale_factors(idx) } -> std::convertible_to<vector_t<real, Rank>>;
    };

    // =========================================================================
    // flux divergence
    // =========================================================================
    template <typename Fluxes, typename Geometry>
    struct flux_divergence_op_t
    {
        using flux_t                        = typename Fluxes::value_type;
        using conserved_t                   = std::remove_cvref_t<typename flux_t::value_type>;
        using value_type                    = conserved_t;
        using argument_type                 = iarray<Fluxes::rank>;
        static constexpr std::uint64_t rank = Fluxes::rank;

        Fluxes   fluxes;
        Geometry geometry;

        DEV constexpr auto operator()(iarray<rank> coord) const
        {
            conserved_t divergence{};
            const auto  dv = geometry.volume(coord);

            for (std::uint64_t dir = 0; dir < rank; ++dir) {
                const auto offset     = unit_vectors::array_offset<rank>(dir);
                const auto coord_plus = coord + offset;

                // flux at left and right faces
                const auto fl = fluxes[dir](coord);
                const auto fr = fluxes[dir](coord_plus);

                // face areas from geometry
                const auto al = geometry.face_area(coord, dir);
                const auto ar = geometry.face_area(coord_plus, dir);

                divergence = divergence + (fr * ar - fl * al) / dv;
            }

            return divergence * (-1.0);
        }
    };

    template <typename FluxField, typename Geometry, std::uint64_t Rank>
    auto flux_divergence(
        const FluxField&                            flux,
        const grid::domain_t<Rank>&                 active_domain,
        const vector_t<grid::domain_t<Rank>, Rank>& face_domains,
        const Geometry&                             geometry
    )
    {
        vector_t<decltype(flux[0][face_domains[0]]), Rank> flux_views;

        for (std::uint64_t dir = 0; dir < Rank; ++dir) {
            flux_views[dir] = flux[dir][face_domains[dir]];
        }

        return compute::computation_t{
            flux_divergence_op_t<decltype(flux_views), Geometry>{flux_views, geometry},
            active_domain
        };
    }

    // =========================================================================
    // gravity source terms
    // =========================================================================
    template <typename GravitySource, typename PrimField, typename Geometry>
    struct gravity_source_op_t
    {
        static constexpr std::uint64_t rank = PrimField::rank;
        using prim_t                        = std::remove_cvref_t<typename PrimField::value_type>;
        using conserved_t                   = typename prim_t::counterpart_t;
        using value_type                    = conserved_t;
        using argument_type                 = iarray<PrimField::rank>;

        const GravitySource* gravity_source;
        PrimField            prims;
        Geometry             geometry;
        real                 time;
        real                 gamma;

        DEV constexpr conserved_t operator()(iarray<rank> coord) const
        {
            if (!gravity_source || !gravity_source->enabled) {
                return conserved_t{};
            }

            const auto position  = geometry.centroid(coord);
            const auto primitive = prims(coord);

            return gravity_source->apply(position, primitive, time, gamma);
        }
    };

    template <typename PrimField, typename Geometry, typename GravSource>
    auto gravity_sources(
        const PrimField&                       prims,
        const grid::domain_t<PrimField::rank>& domain,
        const Geometry&                        geometry,
        const GravSource*                      gravity_source,
        real                                   time,
        real                                   gamma
    )
    {
        return compute::computation_t{
            gravity_source_op_t<GravSource, PrimField, Geometry>{
                gravity_source,
                prims,
                geometry,
                time,
                gamma
            },
            domain
        };
    }

    // =========================================================================
    // hydro source terms
    // =========================================================================
    template <typename HydroSource, typename ConsField, typename Geometry>
    struct hydro_source_op_t
    {
        static constexpr std::uint64_t rank = ConsField::rank;
        using conserved_t                   = std::remove_cvref_t<typename ConsField::value_type>;
        using value_type                    = conserved_t;
        using argument_type                 = iarray<ConsField::rank>;

        const HydroSource* hydro_source;
        ConsField          cons;
        Geometry           geometry;
        real               time;

        DEV constexpr conserved_t operator()(iarray<rank> coord) const
        {
            if (!hydro_source || !hydro_source->enabled) {
                return conserved_t{};
            }

            const auto position  = geometry.centroid(coord);
            const auto conserved = cons(coord);

            return hydro_source->apply(position, conserved, time);
        }
    };

    template <typename ConsField, typename Geometry, typename HydroSource>
    auto hydro_sources(
        const ConsField&                       cons,
        const grid::domain_t<ConsField::rank>& domain,
        const Geometry&                        geometry,
        const HydroSource*                     source,
        real                                   time
    )
    {
        return compute::computation_t{
            hydro_source_op_t<HydroSource, ConsField, Geometry>{source, cons, geometry, time},
            domain
        };
    }

    // =========================================================================
    // geometric source terms (curvilinear coordinates)
    // =========================================================================
    template <typename PrimField, typename Geometry>
    struct geometric_source_op_t
    {
        static constexpr std::uint64_t rank = PrimField::rank;
        using prim_t                        = std::remove_cvref_t<typename PrimField::value_type>;
        using conserved_t                   = typename prim_t::counterpart_t;
        using value_type                    = conserved_t;
        using argument_type                 = iarray<PrimField::rank>;

        PrimField prims;
        Geometry  geometry;
        real      gamma;

        DEV constexpr auto operator()(iarray<rank> coord) const
        {
            const auto primitive = prims(coord);

            // delegate to geometry's metric for source term computation
            return geometry.geomtric_source_factors(primitive, gamma, coord);
        }
    };

    template <typename PrimField, typename Geometry>
    auto geometric_sources(
        const PrimField&                       prims,
        const grid::domain_t<PrimField::rank>& domain,
        const Geometry&                        geometry,
        real                                   gamma
    )
    {
        return compute::computation_t{
            geometric_source_op_t<PrimField, Geometry>{prims, geometry, gamma},
            domain
        };
    }

    // =========================================================================
    // flux computation at interfaces
    // =========================================================================
    template <typename PrimField, typename Geometry, typename CfdOps>
    struct compute_fluxes_op_t
    {
        static constexpr std::uint64_t rank = PrimField::rank;
        using prim_t                        = std::remove_cvref_t<typename PrimField::value_type>;
        using conserved_t                   = typename prim_t::counterpart_t;
        using value_type                    = conserved_t;
        using argument_type                 = iarray<PrimField::rank>;

        PrimField           prims;
        Geometry            geometry;
        CfdOps              ops;
        real                gamma;
        real                plm_theta;
        real                viscosity;
        shockwave_limiter_t shock_smoother;
        std::uint64_t       dir;

        DEV auto operator()(iarray<rank> coord) const
        {
            // create stencil for reconstruction
            const auto stenc    = make_stencil<CfdOps::rec_t>(prims, coord, dir);
            const auto [pl, pr] = ops.reconstruct(stenc, plm_theta);

            // normal vector
            const auto nhat = unit_vectors::ehat<rank>(dir);

            // face grid velocity (moving mesh)
            const auto vface = geometry.face_grid_velocity(coord, dir);

            // solve riemann problem
            auto flux = ops.flux(pl, pr, nhat, vface, gamma, shock_smoother);

            // add viscous stress if enabled
            if (viscosity > 0) {
                const auto visc = compute_viscous_flux(coord, dir, pl.rho, pr.rho);
                flux.mom        = flux.mom - visc;
                flux.nrg        = flux.nrg + vecops::dot(visc, pl.vel);
            }

            return flux;
        }

      private:
        DEV auto
        compute_viscous_flux(iarray<rank> coord, std::uint64_t flux_dir, real rhoL, real rhoR) const
        {
            const auto offset     = unit_vectors::array_offset<rank>(flux_dir);
            const auto left_cell  = coord - offset;
            const auto right_cell = coord;

            auto stress_left  = compute_stress_tensor(left_cell, rhoL);
            auto stress_right = compute_stress_tensor(right_cell, rhoR);

            // average to interface
            vector_t<vector_t<real, rank>, rank> avg_stress;
            for (std::uint64_t ii = 0; ii < rank; ++ii) {
                for (std::uint64_t jj = 0; jj < rank; ++jj) {
                    avg_stress[ii][jj] = 0.5 * (stress_left[ii][jj] + stress_right[ii][jj]);
                }
            }

            // extract flux for this direction
            vector_t<real, rank> stress_flux{};
            const auto           ldd = rank - 1 - flux_dir;
            for (std::uint64_t ii = 0; ii < rank; ++ii) {
                stress_flux[ii] = avg_stress[ii][ldd];
            }

            return stress_flux;
        }

        DEV auto compute_stress_tensor(iarray<rank> coord, real rho) const
        {
            // velocity gradient tensor
            vector_t<vector_t<real, rank>, rank> dv_dx{};
            const auto                           h = geometry.scale_factors(coord);

            for (std::uint64_t dd = 0; dd < rank; ++dd) {
                const auto ldd    = rank - 1 - dd;
                const auto offset = unit_vectors::array_offset<rank>(ldd);
                const real dx     = h[ldd];

                const auto v_plus  = prims(coord + offset).vel;
                const auto v_minus = prims(coord - offset).vel;
                const auto dv      = (v_plus - v_minus) / (2.0 * dx);

                for (std::uint64_t ii = 0; ii < rank; ++ii) {
                    dv_dx[ii][dd] = dv[ii];
                }
            }

            // divergence
            real div_v = 0.0;
            for (std::uint64_t ii = 0; ii < rank; ++ii) {
                div_v += dv_dx[ii][ii];
            }

            // dynamic viscosity
            const auto mu = rho * viscosity;

            // stress tensor
            vector_t<vector_t<real, rank>, rank> sigma;
            for (std::uint64_t ii = 0; ii < rank; ++ii) {
                for (std::uint64_t jj = 0; jj < rank; ++jj) {
                    if (ii == jj) {
                        sigma[ii][jj] = 2.0 * mu * (dv_dx[ii][jj] - div_v / 3.0);
                    }
                    else {
                        sigma[ii][jj] = mu * (dv_dx[ii][jj] + dv_dx[jj][ii]);
                    }
                }
            }

            return sigma;
        }
    };

    template <typename PrimField, typename Geometry, typename CfdOps>
    auto compute_fluxes(
        const PrimField&                       prims,
        const grid::domain_t<PrimField::rank>& face_domain,
        const Geometry&                        geometry,
        const CfdOps&                          ops,
        real                                   gamma,
        real                                   plm_theta,
        real                                   viscosity,
        shockwave_limiter_t                    shock_smoother,
        std::uint64_t                          dir
    )
    {
        return compute::computation_t{
            compute_fluxes_op_t<PrimField, Geometry, CfdOps>{
                prims,
                geometry,
                ops,
                gamma,
                plm_theta,
                viscosity,
                shock_smoother,
                dir
            },
            face_domain
        };
    }

    // =========================================================================
    // godunov operator (complete RHS)
    // =========================================================================
    template <
        typename HydroState,
        typename Geometry,
        typename Sources,
        typename MetaData,
        std::uint64_t Rank>
    auto godunov_op(
        const HydroState&           state,
        const grid::domain_t<Rank>& active_domain,
        const Geometry&             geometry,
        const MetaData&             meta,
        const Sources&              sources
    )
    {
        constexpr std::uint64_t rank = Rank;

        // build face domain array
        vector_t<grid::domain_t<Rank>, Rank> active_face_domains;
        for (std::uint64_t dd = 0; dd < rank; ++dd) {
            active_face_domains[dd] = active_domain;
            active_face_domains[dd].fin[dd] += 1;
        }

        return flux_divergence(state.flux, active_domain, active_face_domains, geometry) +
               gravity_sources(
                   state.prim[active_domain],
                   active_domain,
                   geometry,
                   &sources.gravity_source,
                   meta.time,
                   meta.gamma
               ) +
               hydro_sources(
                   state.cons[active_domain],
                   active_domain,
                   geometry,
                   &sources.hydro_source,
                   meta.time
               ) +
               geometric_sources(state.prim[active_domain], active_domain, geometry, meta.gamma);
    }

    // =========================================================================
    // body effects operator
    // =========================================================================
    template <typename Bodies, typename PrimField, typename Geometry, typename Diagnostics>
    struct body_effects_op_t
    {
        using prim_t                        = std::remove_cvref_t<typename PrimField::value_type>;
        using conserved_t                   = prim_t::counterpart_t;
        using value_type                    = conserved_t;
        using argument_type                 = iarray<PrimField::rank>;
        static constexpr std::uint64_t rank = PrimField::rank;

        Bodies       bodies;
        PrimField    prims;
        Geometry     geometry;
        Diagnostics* diagnostics;
        real         gamma;
        real         dt;

        DEV constexpr auto operator()(iarray<rank> coord) const
        {
            conserved_t total_effect{};
            if (bodies.empty()) {
                return total_effect;
            }

            const auto prim       = prims(coord);
            const bool is_binary  = (bodies.size() == 2);
            const auto sink_cache = bodies.sink_cache;

            bodies.visit_all([&](const auto& body) {
                using body_type = std::decay_t<decltype(body)>;
                body_delta_t<rank> delta{
                    .idx          = body.idx,
                    .force_delta  = {},
                    .torque_delta = {},
                    .mass_delta   = 0.0
                };

                if constexpr (has_gravitational_capability_c<body_type>) {
                    auto grav_op           = make_grav_op(prim, geometry);
                    auto [effect, g_delta] = grav_op(body, coord);
                    total_effect           = total_effect | structs::add_gas(effect);
                    delta += g_delta;
                }

                if constexpr (has_accretion_capability_c<body_type>) {
                    auto accr_op     = make_accretion_op(prim, geometry, gamma, dt);
                    real mdot_target = 0.0;
                    real w_total     = 0.0;
                    real r_bh        = 0.0;

                    if (sink_cache && !sink_cache->empty()) {
                        const auto& props = (*sink_cache)[body.idx];
                        mdot_target       = props.mdot;
                        w_total           = props.total_weight;
                        r_bh              = props.r_bh;
                    }

                    auto [effect, a_delta] =
                        accr_op(body, coord, is_binary, mdot_target, w_total, r_bh);
                    total_effect = total_effect | structs::add_gas(effect);
                    delta += a_delta;
                }

                if constexpr (has_rigid_capability_c<body_type>) {
                    auto rigid_op          = make_rigid_op(prim, geometry, gamma);
                    auto [effect, r_delta] = rigid_op(body, coord);
                    total_effect           = total_effect | structs::add_gas(effect);
                    delta += r_delta;
                }

                if (diagnostics) {
                    diagnostics->accumulate_delta(delta);
                }
            });

            return total_effect;
        }
    };

    // =========================================================================
    // body effects computation
    // =========================================================================
    template <typename PrimField, typename Geometry, typename Bodies, typename Diagnostics>
    auto body_effects(
        const PrimField&                       prims,
        const grid::domain_t<PrimField::rank>& active_domain,
        const Geometry&                        geometry,
        const Bodies&                          bodies,
        Diagnostics*                           diagnostics,
        real                                   gamma,
        real                                   dt
    )
    {
        return compute::computation_t{
            body_effects_op_t{bodies, prims, geometry, diagnostics, gamma, dt},
            active_domain
        };
    }

} // namespace simbi::cfd

#endif // CFD_OPS_HPP
