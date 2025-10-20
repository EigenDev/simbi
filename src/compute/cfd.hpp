#ifndef CFD_OPS_HPP
#define CFD_OPS_HPP

#include "base/stencil_view.hpp"
#include "compat.hpp"
#include "compute/field.hpp"
#include "containers/state_ops.hpp"
#include "containers/vector.hpp"
#include "domain/domain.hpp"
#include "mesh/mesh_ops.hpp"
#include "physics/em/ct_updater.hpp"
#include "physics/ib/body.hpp"
#include "physics/ib/body_delta.hpp"
#include "physics/ib/collection.hpp"
#include "physics/ib/effects.hpp"
#include "update/adaptive_timestep.hpp"
#include "update/bcs.hpp"
#include "update/flux.hpp"
#include "update/prim_recovery.hpp"
#include "update/rk.hpp"
#include "utility/enums.hpp"

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace simbi::cfd {
    using namespace base::stencils;
    using namespace simbi::body::expr;
    using namespace simbi::body;

    // =================================================================
    // Pure CFD Operations - Return compute_field_t
    // =================================================================

    template <typename Fluxes, typename MeshConfig>
    struct flux_divergence_op_t {
        using flux_t      = Fluxes::value_type;
        using conserved_t = std::remove_cvref_t<typename flux_t::value_type>;
        static constexpr std::uint64_t dims = Fluxes::dimensions;

        Fluxes fluxes;
        MeshConfig mesh;

        DEV constexpr auto operator()(auto coord) const
        {
            conserved_t divergence{};
            const auto dv = mesh::volume(coord, mesh);

            // compute divergence using pre-computed fluxes
            for (std::uint64_t dir = 0; dir < dims; ++dir) {
                const auto offset     = unit_vectors::logical_offset<dims>(dir);
                const auto coord_plus = coord + offset;

                // flux values at left and right faces
                const auto fl = fluxes[dir][coord];
                const auto fr = fluxes[dir][coord_plus];

                // geometric face areas
                const auto al = mesh::face_area(coord, dir, Dir::W, mesh);
                const auto ar = mesh::face_area(coord, dir, Dir::E, mesh);

                // add contribution to divergence
                divergence = divergence + (fr * ar - fl * al) / dv;
            }

            return divergence * (-1.0);
        }
    };

    /**
     * flux divergence using pre-computed interface fluxes
     * returns: conservative update from flux divergence
     */
    template <typename HydroState, typename MeshConfig>
    auto flux_divergence(const HydroState& state, const MeshConfig& mesh)
    {
        vector_t<
            decltype(state.flux[0][mesh.face_domain[0]]),
            HydroState::dimensions>
            flux_views;

        for (std::uint64_t dir = 0; dir < HydroState::dimensions; ++dir) {
            flux_views[dir] = state.flux[dir][mesh.face_domain[dir]];
        }
        return compute_field_t{
          flux_divergence_op_t{flux_views, mesh},
          make_domain(mesh.domain.shape())
        };
    }

    template <typename GravitySource, typename PrimField, typename MeshConfig>
    struct gravity_source_op_t {
        constexpr static auto dims = MeshConfig::dimensions;
        using prim_t      = std::remove_cvref_t<typename PrimField::value_type>;
        using conserved_t = prim_t::counterpart_t;

        GravitySource* gravity_source;
        PrimField prims;
        MeshConfig mesh;
        real time;
        real gamma;

        DEV constexpr auto operator()(auto coord) const
        {
            if (!gravity_source->enabled) {
                return conserved_t{};
            }

            const auto position  = mesh::centroid(coord, mesh);
            const auto primitive = prims[coord];

            return gravity_source->apply(position, primitive, time, gamma);
        }
    };

    /**
     * gravity source terms
     * returns: conservative update from gravitational acceleration
     */
    template <typename HydroState, typename MeshConfig>
    auto gravity_sources(const HydroState& state, const MeshConfig& mesh)
    {
        return compute_field_t{
          gravity_source_op_t{
            &state.sources.gravity_source,
            state.prim[mesh.domain],
            mesh,
            state.metadata.time,
            state.metadata.gamma
          },
          make_domain(mesh.domain.shape())
        };
    }

    template <typename HydroSource, typename ConsField, typename MeshConfig>
    struct hydro_sources_op_t {
        using conserved_t = std::remove_cvref_t<typename ConsField::value_type>;

        HydroSource* hydro_source;
        ConsField cons;
        MeshConfig mesh;
        real time;

        DEV constexpr auto operator()(auto coord) const
        {
            if (!hydro_source->enabled) {
                return conserved_t{};
            }

            const auto position  = mesh::centroid(coord, mesh);
            const auto conserved = cons[coord];

            return hydro_source->apply(position, conserved, time);
        }
    };

    /**
     * hydro source terms (cooling, heating, etc.)
     * returns: conservative update from hydro sources
     */
    template <typename HydroState, typename MeshConfig>
    auto hydro_sources(const HydroState& state, const MeshConfig& mesh)
    {
        return compute_field_t{
          hydro_sources_op_t{
            &state.sources.hydro_source,
            state.cons[mesh.domain],
            mesh,
            state.metadata.time
          },
          make_domain(mesh.domain.shape())
        };
    }

    template <typename PrimField, typename MeshConfig>
    struct geometric_source_op_t {
        using prim_t      = std::remove_cvref_t<typename PrimField::value_type>;
        using conserved_t = prim_t::counterpart_t;

        PrimField prims;
        MeshConfig mesh;
        real gamma;

        DEV constexpr auto operator()(auto coord) const
        {
            // geometric sources only exist for non-Cartesian geometries
            if constexpr (MeshConfig::geometry == Geometry::CARTESIAN) {
                return conserved_t{};
            }
            else {
                const auto primitive = prims[coord];
                return mesh::geometric_source_terms(
                    primitive,
                    coord,
                    mesh,
                    gamma
                );
            }
        }
    };

    /**
     * geometric source terms for non-Cartesian coordinates
     * returns: conservative update from geometric effects
     */
    template <typename HydroState, typename MeshConfig>
    auto geometric_sources(const HydroState& state, const MeshConfig& mesh)
    {
        return compute_field_t{
          geometric_source_op_t{
            state.prim[mesh.domain],
            mesh,
            state.metadata.gamma
          },
          make_domain(mesh.domain.shape())
        };
    }

    template <
        typename Bodies,
        typename PrimField,
        typename MeshConfig,
        typename BodyDiagnostics>
    struct body_effects_op_t {
        using prim_t      = std::remove_cvref_t<typename PrimField::value_type>;
        using conserved_t = prim_t::counterpart_t;
        static constexpr std::uint64_t Dims = PrimField::dimensions;

        Bodies bodies;
        PrimField prims;
        MeshConfig mesh;
        BodyDiagnostics* diagnostics;
        real gamma;
        real dt;

        DEV constexpr auto operator()(auto coord) const
        {
            if (!bodies.has_value() || bodies->empty()) {
                return conserved_t{};
            }
            conserved_t total_effect{};

            const auto prim       = prims[coord];
            const bool is_binary  = (bodies->size() == 2);
            const auto sink_cache = bodies->sink_cache;
            bodies->visit_all([&](const auto& body) {
                using body_type = std::decay_t<decltype(body)>;
                body_delta_t<Dims> delta{
                  .idx          = body.idx,
                  .force_delta  = {},
                  .torque_delta = {},
                  .mass_delta   = 0.0
                };

                if constexpr (has_gravitational_capability_c<body_type>) {
                    auto grav_op           = grav_op_t{prim, mesh};
                    auto [effect, g_delta] = grav_op(body, coord);
                    total_effect = total_effect | structs::add_gas(effect);
                    delta += g_delta;
                }

                if constexpr (has_accretion_capability_c<body_type>) {
                    auto accr_op     = accretion_op_t{prim, mesh, gamma, dt};
                    real mdot_target = 0.0;
                    real w_total     = 0.0;
                    real r_bh        = 0.0;

                    if (sink_cache && !sink_cache->empty()) {
                        const auto& props = (*sink_cache)[body.idx];
                        mdot_target       = props.mdot;
                        w_total           = props.total_weight;
                        r_bh              = props.r_bh;
                    }

                    auto [effect, a_delta] = accr_op(
                        body,
                        coord,
                        is_binary,
                        mdot_target,
                        w_total,
                        r_bh
                    );
                    total_effect = total_effect | structs::add_gas(effect);
                    delta += a_delta;
                }

                if constexpr (has_rigid_capability_c<body_type>) {
                    auto rigid_op          = rigid_op_t{prim, mesh, gamma};
                    auto [effect, r_delta] = rigid_op(body, coord);
                    total_effect = total_effect | structs::add_gas(effect);
                    delta += r_delta;
                }

                if (diagnostics) {
                    diagnostics->accumulate_delta(delta);
                }
            });

            return total_effect;
        }
    };

    /**
     * immersed body effects
     * returns: conservative update from body forces/sources
     */
    template <typename HydroState, typename MeshConfig>
    auto body_effects(const HydroState& state, const MeshConfig& mesh)
    {
        return compute_field_t{
          body_effects_op_t{
            state.bodies,
            state.prim[mesh.domain],
            mesh,
            state.diagnostics.get(),
            state.metadata.gamma,
            state.metadata.dt
          },
          make_domain(mesh.domain.shape())
        };
    }

    // =================================================================
    // Flux Computation Operations
    // =================================================================
    // viscous stress computation

    // cylindrical/spherical coordinate gradients
    template <typename PrimField, typename MeshConfig>
    DEV auto compute_curvilinear_gradients(
        const PrimField& prims,
        const auto& coord,
        const MeshConfig& mesh
    )
    {
        constexpr auto dims = PrimField::dimensions;
        constexpr auto geom = MeshConfig::geometry;

        vector_t<vector_t<real, dims>, dims> dv_dx{};
        const auto widths = mesh::cell_widths(coord, mesh);
        const auto cent   = mesh::centroid(coord, mesh);

        for (std::uint64_t dd = 0; dd < dims; ++dd) {
            const auto ldd    = dims - 1 - dd;   // logical dimension
            const auto offset = unit_vectors::logical_offset<dims>(ldd);
            const real dx     = widths[ldd];

            const auto v_plus  = prims[coord + offset].vel;
            const auto v_minus = prims[coord - offset].vel;
            const auto dv      = (v_plus - v_minus) / (2.0 * dx);

            for (std::uint64_t ii = 0; ii < dims; ++ii) {
                if constexpr (geom == Geometry::CYLINDRICAL) {
                    // cylindrical metric corrections
                    if (dd == 0) {   // radial derivative
                        dv_dx[ii][dd] = dv[ii];
                    }
                    else if (dd == 1 && dims > 1) {   // azimuthal derivative
                        const real r  = cent[dims - 1];
                        dv_dx[ii][dd] = dv[ii] / r;
                    }
                    else {   // z derivative
                        dv_dx[ii][dd] = dv[ii];
                    }
                }
                else if constexpr (geom == Geometry::SPHERICAL) {
                    // spherical metric corrections
                    if (dd == 0) {   // radial derivative
                        dv_dx[ii][dd] = dv[ii];
                    }
                    else if (dd == 1 && dims > 1) {   // theta derivative
                        const real r  = cent[dims - 1];
                        dv_dx[ii][dd] = dv[ii] / r;
                    }
                    else if (dd == 2 && dims > 2) {   // phi derivative
                        const real r     = cent[dims - 1];
                        const real theta = cent[dims - 2];
                        dv_dx[ii][dd]    = dv[ii] / (r * std::sin(theta));
                    }
                }
            }
        }

        return dv_dx;
    }

    // generalized velocity gradient computation accounting for coordinate
    // system
    template <typename PrimField, typename MeshConfig>
    DEV auto compute_velocity_gradients(
        const PrimField& prims,
        const auto& coord,
        const MeshConfig& mesh
    )
    {
        constexpr auto dims = PrimField::dimensions;
        constexpr auto geom = MeshConfig::geometry;

        // velocity gradient tensor
        vector_t<vector_t<real, dims>, dims> dv_dx;

        if constexpr (geom == Geometry::CARTESIAN) {
            const auto widths = mesh::cell_widths(coord, mesh);

            for (std::uint64_t dd = 0; dd < dims; ++dd) {
                const auto ldd    = dims - 1 - dd;   // logical dimension
                const auto offset = unit_vectors::logical_offset<dims>(ldd);
                const real dxi    = widths[ldd];

                const auto v_plus  = prims[coord + offset].vel;
                const auto v_minus = prims[coord - offset].vel;
                const auto dv      = (v_plus - v_minus) / (2.0 * dxi);

                for (std::uint64_t ii = 0; ii < dims; ++ii) {
                    dv_dx[ii][dd] = dv[ii];
                }
            }
        }
        else {
            // need metric tensor corrections for curvilinear coordinates
            return compute_curvilinear_gradients(prims, coord, mesh);
        }

        return dv_dx;
    }

    // extract stress components for flux direction
    template <std::uint64_t dims>
    DEV auto extract_stress_flux(
        const vector_t<vector_t<real, dims>, dims>& sigma,
        std::uint64_t dir
    )
    {
        vector_t<real, dims> stress_flux{};
        // logical dimension
        const auto ldd = dims - 1 - dir;
        for (std::uint64_t ii = 0; ii < dims; ++ii) {
            stress_flux[ii] = sigma[ii][ldd];   // sigma column for direction j
        }

        return stress_flux;
    }

    // compute full stress tensor at cell center
    template <typename PrimField, typename MeshConfig>
    DEV auto stress_tensor(
        const PrimField& prims,
        const auto& coord,
        const MeshConfig& mesh,
        real nu,
        real rho_interface
    )
    {
        constexpr auto dims = PrimField::dimensions;

        const auto dv_dx = compute_velocity_gradients(prims, coord, mesh);

        real div_v = 0.0;
        for (std::uint64_t ii = 0; ii < dims; ++ii) {
            div_v += dv_dx[ii][ii];
        }

        // dynamic viscosity
        const auto mu = rho_interface * nu;

        // assemble stress tensor
        vector_t<vector_t<real, dims>, dims> sigma;
        for (std::uint64_t ii = 0; ii < dims; ++ii) {
            for (std::uint64_t jj = 0; jj < dims; ++jj) {
                if (ii == jj) {
                    // diagonal components
                    sigma[ii][jj] = 2.0 * mu * (dv_dx[ii][jj] - div_v / 3.0);
                }
                else {
                    // off-diagonal components
                    sigma[ii][jj] = mu * (dv_dx[ii][jj] + dv_dx[jj][ii]);
                }
            }
        }

        return sigma;
    }

    // viscous stress computation
    template <typename PrimField, typename MeshConfig>
    DEV auto viscous_stress_flux(
        const PrimField& prims,
        const auto& coord,
        std::uint64_t dir,
        const MeshConfig& mesh,
        real nu,
        real rhoL,
        real rhoR
    )
    {
        constexpr auto dims = PrimField::dimensions;

        // get cells on either side of interface
        const auto offset     = unit_vectors::logical_offset<dims>(dir);
        const auto left_cell  = coord - offset;
        const auto right_cell = coord;

        // compute stress tensor at both cells
        auto stress_left  = stress_tensor(prims, left_cell, mesh, nu, rhoL);
        auto stress_right = stress_tensor(prims, right_cell, mesh, nu, rhoR);

        // average stress tensor to interface
        vector_t<vector_t<real, dims>, dims> avg_stress;
        for (std::uint64_t ii = 0; ii < dims; ++ii) {
            for (std::uint64_t jj = 0; jj < dims; ++jj) {
                avg_stress[ii][jj] =
                    0.5 * (stress_left[ii][jj] + stress_right[ii][jj]);
            }
        }

        // extract stress flux for this direction
        return extract_stress_flux<dims>(avg_stress, dir);
    }

    template <
        typename PrimField,
        typename Metadata,
        typename CfdOps,
        typename MeshConfig>
    struct compute_fluxes_op_t {
        PrimField prims;
        Metadata meta;
        MeshConfig mesh;
        CfdOps ops;
        std::uint64_t dir;

        DEV auto operator()(auto coord) const
        {
            constexpr auto dims       = MeshConfig::dimensions;
            const auto gamma          = meta.gamma;
            const auto shock_smoother = meta.shock_smoother;
            const auto plm_theta      = meta.plm_theta;
            const auto nu             = meta.viscosity;

            // create stencil for reconstruction around this interface
            const auto stenc = make_stencil<CfdOps::rec_t>(prims, coord, dir);
            const auto [pl, pr] = ops.reconstruct(stenc, plm_theta);
            // normal vector for this dimension
            const auto nhat = unit_vectors::ehat<dims>(dir);
            // face velocity (for moving meshes)
            const auto vface = mesh::face_velocity(coord, dir, mesh);

            // solve Riemann problem
            auto flux = ops.flux(pl, pr, nhat, vface, gamma, shock_smoother);
            if (nu > 0) {
                const auto visc = viscous_stress_flux(
                    prims,
                    coord,
                    dir,
                    mesh,
                    nu,
                    pl.rho,
                    pr.rho
                );
                flux.mom = flux.mom - visc;
                flux.nrg = flux.nrg + vecops::dot(visc, pl.vel);
            }
            return flux;
        }
    };

    /**
     * compute interface fluxes using Riemann solvers
     * returns: flux field for a specific direction
     */
    template <typename HydroState, typename CfdOps, typename MeshConfig>
    auto compute_fluxes(
        const HydroState& state,
        const MeshConfig& mesh,
        const CfdOps& ops,
        std::uint64_t dir
    )
    {
        return compute_field_t{
          compute_fluxes_op_t{
            state.prim[mesh.domain],
            state.metadata,
            mesh,
            ops,
            dir
          },
          make_domain(mesh.face_domain[dir].shape())
        };
    }

    // =================================================================
    // Composite Operations - Automatic Fusion
    // =================================================================

    /**
     * complete RHS for conservative update
     * returns: fused field of all source terms
     */
    template <typename HydroState, typename MeshConfig>
    auto godunov_op(const HydroState& state, const MeshConfig& mesh)
    {
        return flux_divergence(state, mesh) + gravity_sources(state, mesh) +
               hydro_sources(state, mesh) + geometric_sources(state, mesh) +
               body_effects(state, mesh);
    }

    /**
     * time step update with CFL condition
     * returns: new conservative state
     */
    template <typename HydroState, typename MeshConfig, typename CfdOps>
    auto step(HydroState& state, const MeshConfig& mesh, const CfdOps& ops)
    {
        // u' = u + L(u)
        // where L(u) is the godunov operator
        if (state.metadata.timestepping == Timestepping::EULER) {
            const auto dt = state.metadata.dt;
            update_sink_cache(state, mesh);
            update_staggered_fields(state, ops, mesh);

            const auto ell = godunov_op(state, mesh) * dt;
            auto u_p       = state.cons[mesh.domain];
            u_p            = u_p.enum_map([=](auto coord, auto u) {
                return u | structs::add_gas(ell(coord));
            });

            if constexpr (HydroState::is_mhd) {
                // correct energy density from CT algorithm
                em::update_energy_density(state, mesh);
            }

            boundary::apply_boundary_conditions(state, mesh);
            hydro::recover_primitives(state);
            update_timestep(state, mesh);
            state.metadata.time += dt;
        }
        else if (state.metadata.timestepping == Timestepping::RK2) {
            rk::rk2_step(state, mesh, ops);
        }
        else {
            throw std::runtime_error(
                "Unsupported timestepping method: " +
                std::to_string(
                    static_cast<std::uint64_t>(state.metadata.timestepping)
                )
            );
        }
    }
}   // namespace simbi::cfd

#endif   // CFD_OPS_HPP
