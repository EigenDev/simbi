#ifndef CFD_OPS_HPP
#define CFD_OPS_HPP

#include "base/stencil_view.hpp"
#include "compute/field.hpp"
#include "config.hpp"
#include "containers/state_ops.hpp"
#include "containers/vector.hpp"
#include "domain/domain.hpp"
#include "mesh/mesh_ops.hpp"
#include "physics/em/perm.hpp"
#include "physics/hydro/ib/body.hpp"
#include "physics/hydro/ib/effects.hpp"
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

    template <typename HydroState, typename MeshConfig>
    struct flux_divergence_op_t {
        HydroState state;
        MeshConfig mesh;

        DEV auto operator()(auto coord) const
        {
            using conserved_t   = typename HydroState::conserved_t;
            constexpr auto dims = HydroState::dimensions;

            conserved_t divergence{};
            const auto dv    = mesh::volume(coord, mesh);
            const auto& flux = state.flux;

            // compute divergence using pre-computed fluxes
            for (std::uint64_t dim = 0; dim < dims; ++dim) {
                auto offset     = unit_vectors::logical_offset<dims>(dim);
                auto coord_plus = coord + offset;

                // flux values at left and right faces
                auto fd = mesh.face_domain[dim];
                auto fl = flux[dim][fd][coord];
                auto fr = flux[dim][fd][coord_plus];

                // geometric face areas
                auto al = mesh::face_area(coord, dim, Dir::W, mesh);
                auto ar = mesh::face_area(coord, dim, Dir::E, mesh);

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
        return compute_field_t{
          flux_divergence_op_t{state, mesh},
          make_domain(mesh.domain.shape())
        };
    }

    template <typename HydroState, typename MeshConfig>
    struct gravity_source_op_t {
        HydroState state;
        MeshConfig mesh;
        constexpr static auto dims = HydroState::dimensions;

        DEV auto operator()(auto coord) const
        {
            using conserved_t          = typename HydroState::conserved_t;
            const auto* gravity_source = &state.sources.gravity_source;

            if (!gravity_source->enabled) {
                return conserved_t{};
            }

            const auto position  = mesh::centroid(coord, mesh);
            const auto primitive = state.prim[mesh.domain][coord];

            return gravity_source->apply(
                position,
                primitive,
                state.metadata.time,
                state.metadata.dt
            );
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
          gravity_source_op_t<HydroState, MeshConfig>{state, mesh},
          make_domain(mesh.domain.shape())
        };
    }

    template <typename HydroState, typename MeshConfig>
    struct hydro_sources_op_t {
        HydroState state;
        MeshConfig mesh;

        DEV auto operator()(auto coord) const
        {
            using conserved_t         = typename HydroState::conserved_t;
            const auto* hydro_sources = &state.sources.hydro_source;

            if (!hydro_sources->enabled) {
                return conserved_t{};
            }

            const auto position  = mesh::centroid(coord, mesh);
            const auto conserved = state.cons[mesh.domain][coord];

            return hydro_sources->apply(
                position,
                conserved,
                state.metadata.time,
                state.metadata.gamma
            );
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
          hydro_sources_op_t<HydroState, MeshConfig>{state, mesh},
          make_domain(mesh.domain.shape())
        };
    }

    template <typename HydroState, typename MeshConfig>
    struct geometric_source_op_t {
        HydroState state;
        MeshConfig mesh;

        DEV auto operator()(auto coord) const
        {
            using conserved_t   = typename HydroState::conserved_t;
            constexpr auto dims = HydroState::dimensions;

            // geometric sources only exist for non-Cartesian geometries
            if constexpr (MeshConfig::geometry == Geometry::CARTESIAN) {
                return conserved_t{};
            }
            else {
                const auto primitive = state.prim[mesh.domain][coord];
                return mesh::geometric_source_terms(
                    primitive,
                    coord,
                    mesh,
                    state.metadata.gamma
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
          geometric_source_op_t<HydroState, MeshConfig>{state, mesh},
          make_domain(mesh.domain.shape())
        };
    }

    template <typename HydroState, typename MeshConfig>
    struct body_effects_op_t {
        HydroState state;
        MeshConfig mesh;
        constexpr static auto dims = HydroState::dimensions;

        DEV auto apply_to_body(const auto& body, const auto& coord) const
        {
            using conserved_t = typename HydroState::conserved_t;
            return body.apply_effects(state, coord);
        }
    };

    template <typename HydroState, typename MeshConfig>
    struct body_effects_op {
        HydroState state;
        MeshConfig mesh;

        DEV auto operator()(auto coord) const
        {
            using conserved_t   = typename HydroState::conserved_t;
            constexpr auto dims = HydroState::dimensions;
            const auto bodies   = state.bodies;

            // check if we have bodies
            if (!bodies.has_value() || bodies->empty()) {
                return conserved_t{};
            }

            conserved_t total_effect{};

            // visit all bodies and accumulate effects
            bodies->visit_all([&](const auto& body) {
                using body_type = std::decay_t<decltype(body)>;

                // For the constexpr-if issue, move the operators inside the
                // lambda or use if constexpr with local variables
                if constexpr (has_gravitational_capability_c<body_type>) {
                    auto local_grav_op =
                        gravitational_effect_op_t<HydroState, MeshConfig, dims>{
                          state,
                          mesh
                        };
                    total_effect += local_grav_op.apply_to_body(body, coord);
                }

                if constexpr (has_accretion_capability_c<body_type>) {
                    auto local_accr_op =
                        accretion_effect_op_t<HydroState, MeshConfig, dims>{
                          state,
                          mesh
                        };
                    total_effect += local_accr_op.apply_to_body(body, coord);
                }

                if constexpr (has_rigid_capability_c<body_type>) {
                    auto local_rigid_op =
                        rigid_effect_op_t<HydroState, MeshConfig, dims>{
                          state,
                          mesh
                        };
                    total_effect += local_rigid_op.apply_to_body(body, coord);
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
          body_effects_op<HydroState, MeshConfig>{state, mesh},
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

        for (std::uint64_t d = 0; d < dims; ++d) {
            const auto offset = unit_vectors::logical_offset<dims>(d);
            const real dx     = widths[d];

            const auto v_plus  = prims[coord + offset].vel;
            const auto v_minus = prims[coord - offset].vel;
            const auto dv      = (v_plus - v_minus) / (2.0 * dx);

            for (std::uint64_t i = 0; i < dims; ++i) {
                if constexpr (geom == Geometry::CYLINDRICAL) {
                    // cylindrical metric corrections
                    if (d == dims - 1) {   // radial derivative
                        dv_dx[i][d] = dv[i];
                    }
                    else if (d == dims - 2 &&
                             dims > 1) {   // azimuthal derivative
                        const real r = cent[dims - 1];
                        dv_dx[i][d]  = dv[i] / r;   // 1/r ∂/∂φ
                    }
                    else {   // z derivative
                        dv_dx[i][d] = dv[i];
                    }
                }
                else if constexpr (geom == Geometry::SPHERICAL) {
                    // spherical metric corrections
                    if (d == dims - 1) {   // radial derivative
                        dv_dx[i][d] = dv[i];
                    }
                    else if (d == dims - 2 && dims > 1) {   // theta derivative
                        const real r = cent[dims - 1];
                        dv_dx[i][d]  = dv[i] / r;
                    }
                    else if (d == dims - 3 && dims > 2) {   // phi derivative
                        const real r     = cent[dims - 1];
                        const real theta = cent[dims - 2];
                        dv_dx[i][d]      = dv[i] / (r * std::sin(theta));
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
            // standard cartesian gradients
            const auto widths = mesh::cell_widths(coord, mesh);

            for (std::uint64_t d = 0; d < dims; ++d) {
                const auto offset = unit_vectors::logical_offset<dims>(d);
                const real dx     = widths[d];

                const auto v_plus  = prims[coord + offset].vel;
                const auto v_minus = prims[coord - offset].vel;
                const auto dv      = (v_plus - v_minus) / (2.0 * dx);

                for (std::uint64_t i = 0; i < dims; ++i) {
                    dv_dx[i][d] = dv[i];
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
    DEV auto
    extract_stress_flux(const real (&sigma)[dims][dims], std::uint64_t dir)
    {
        using vec_t = vector_t<real, dims>;
        vec_t stress_flux{};

        for (std::uint64_t i = 0; i < dims; ++i) {
            stress_flux[i] = sigma[i][dir];   // sigma column for direction j
        }

        return stress_flux;
    }

    // compute full stress tensor at cell center
    template <typename PrimField, typename MeshConfig>
    DEV auto compute_stress_tensor(
        const PrimField& prims,
        const auto& coord,
        const MeshConfig& mesh,
        real nu
    )
    {
        constexpr auto dims = PrimField::dimensions;

        // get velocity gradients
        auto dv_dx = compute_velocity_gradients(prims, coord, mesh);

        // compute divergence
        real div_v = 0.0;
        for (std::uint64_t i = 0; i < dims; ++i) {
            div_v += dv_dx[i][i];
        }

        // dynamic viscosity
        const auto rho = prims[coord].rho;
        const auto mu  = rho * nu;

        // assemble stress tensor
        vector_t<vector_t<real, dims>, dims> sigma;
        for (std::uint64_t i = 0; i < dims; ++i) {
            for (std::uint64_t j = 0; j < dims; ++j) {
                if (i == j) {
                    // diagonal components
                    sigma[i][j] = 2.0 * mu * (dv_dx[i][j] - div_v / 3.0);
                }
                else {
                    // off-diagonal components
                    sigma[i][j] = mu * (dv_dx[i][j] + dv_dx[j][i]);
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
        real nu
    )
    {
        constexpr auto dims = PrimField::dimensions;

        // get cells on either side of interface
        const auto offset     = unit_vectors::logical_offset<dims>(dir);
        const auto left_cell  = coord - offset;
        const auto right_cell = coord;

        // compute stress tensor at both cells
        auto stress_left  = compute_stress_tensor(prims, left_cell, mesh, nu);
        auto stress_right = compute_stress_tensor(prims, right_cell, mesh, nu);

        // average stress tensor to interface
        real avg_stress[dims][dims] = {};
        for (std::uint64_t i = 0; i < dims; ++i) {
            for (std::uint64_t j = 0; j < dims; ++j) {
                avg_stress[i][j] =
                    0.5 * (stress_left[i][j] + stress_right[i][j]);
            }
        }

        // extract stress flux for this direction
        return extract_stress_flux<dims>(avg_stress, dir);
    }

    template <typename HydroState, typename CfdOps, typename MeshConfig>
    struct compute_fluxes_op_t {
        HydroState state;
        MeshConfig mesh;
        CfdOps ops;
        std::uint64_t dir;

        DEV auto operator()(auto coord) const
        {
            constexpr auto dims       = HydroState::dimensions;
            const auto gamma          = state.metadata.gamma;
            const auto shock_smoother = state.metadata.shock_smoother;
            const auto plm_theta      = state.metadata.plm_theta;
            const auto prims          = state.prim[mesh.domain];
            const auto nu             = state.metadata.viscosity;

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
                auto visc = viscous_stress_flux(prims, coord, dir, mesh, nu);
                flux.mom  = flux.mom + visc;
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
          compute_fluxes_op_t<HydroState, CfdOps, MeshConfig>{
            state,
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
            update_staggered_fields(state, ops, mesh);

            auto u_p       = state.cons[mesh.domain];
            const auto ell = godunov_op(state, mesh) * dt;
            u_p            = u_p.map([=](const auto coord, const auto u) {
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
