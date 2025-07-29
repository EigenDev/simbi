#ifndef SIMBI_CFD_OPS_HPP
#define SIMBI_CFD_OPS_HPP

#include "compute/field.hpp"
#include "compute/math/domain.hpp"
#include "config.hpp"
#include "containers/vector.hpp"
#include "core/base/stencil_view.hpp"
#include "core/utility/enums.hpp"
#include "mesh/mesh_ops.hpp"
#include "physics/em/perm.hpp"
#include "physics/hydro/ib/body.hpp"
#include "physics/hydro/ib/effects.hpp"
#include "update/adaptive_timestep.hpp"
#include "update/bcs.hpp"
#include "update/flux.hpp"
#include "update/prim_recovery.hpp"

#include <cstdint>
#include <type_traits>

namespace simbi::cfd {
    using namespace base::stencils;
    using namespace simbi::body::expr;
    using namespace simbi::body;

    // =================================================================
    // Pure CFD Operations - Return compute_field_t
    // =================================================================

    /**
     * flux divergence using pre-computed interface fluxes
     * returns: conservative update from flux divergence
     */
    template <typename HydroState, typename MeshConfig>
    auto flux_divergence(const HydroState& state, const MeshConfig& mesh)
    {
        const auto flux = state.flux;
        return compute_field_t{
          [=] DEV(auto coord) {
              using conserved_t   = typename HydroState::conserved_t;
              constexpr auto dims = HydroState::dimensions;

              conserved_t divergence{};
              const auto dv = mesh::volume(coord, mesh);

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
          },
          make_domain(mesh.domain.shape())
        };
    }

    /**
     * gravity source terms
     * returns: conservative update from gravitational acceleration
     */
    template <typename HydroState, typename MeshConfig>
    auto gravity_sources(const HydroState& state, const MeshConfig& mesh)
    {
        const auto time            = state.metadata.time;
        const auto dt              = state.metadata.dt;
        const auto* gravity_source = &state.sources.gravity_source;
        const auto prim            = state.prim[mesh.domain];
        return compute_field_t{
          [=](auto coord) {
              using conserved_t = typename HydroState::conserved_t;
              if (!gravity_source->enabled) {
                  return conserved_t{};
              }

              const auto position  = mesh::centroid(coord, mesh);
              const auto primitive = prim[coord];

              return gravity_source->apply(position, primitive, time, dt);
          },
          make_domain(mesh.domain.shape())
        };
    }

    /**
     * hydro source terms (cooling, heating, etc.)
     * returns: conservative update from hydro sources
     */
    template <typename HydroState, typename MeshConfig>
    auto hydro_sources(const HydroState& state, const MeshConfig& mesh)
    {
        const auto time           = state.metadata.time;
        const auto gamma          = state.metadata.gamma;
        const auto* hydro_sources = &state.sources.hydro_source;
        const auto cons           = state.cons[mesh.domain];
        return compute_field_t{
          [=] DEV(auto coord) {
              using conserved_t = typename HydroState::conserved_t;

              if (!hydro_sources->enabled) {
                  return conserved_t{};
              }

              const auto position  = mesh::centroid(coord, mesh);
              const auto conserved = cons[coord];

              return hydro_sources->apply(position, conserved, time, gamma);
          },
          make_domain(mesh.domain.shape())
        };
    }

    /**
     * geometric source terms for non-Cartesian coordinates
     * returns: conservative update from geometric effects
     */
    template <typename HydroState, typename MeshConfig>
    auto geometric_sources(const HydroState& state, const MeshConfig& mesh)
    {
        const auto domain = mesh.domain;
        const auto gamma  = state.metadata.gamma;
        const auto prims  = state.prim;
        return compute_field_t{
          [=] DEV(auto coord) {
              using conserved_t = typename HydroState::conserved_t;

              // geometric sources only exist for
              // non-Cartesian geometries
              if constexpr (MeshConfig::geometry == Geometry::CARTESIAN) {
                  return conserved_t{};
              }
              else {
                  const auto primitive = prims[domain][coord];
                  auto val             = mesh::geometric_source_terms(
                      primitive,
                      coord,
                      mesh,
                      gamma
                  );
                  return val;
              }
          },
          make_domain(mesh.domain.shape())
        };
    }

    /**
     * immersed body effects
     * returns: conservative update from body forces/sources
     */
    template <typename HydroState, typename MeshConfig>
    auto body_effects(const HydroState& state, const MeshConfig& mesh)
    {
        constexpr auto dims = HydroState::dimensions;
        // effect operators
        auto grav_op = gravitational_effect_op_t<HydroState, MeshConfig, dims>{
          state,
          mesh
        };
        auto accr_op =
            accretion_effect_op_t<HydroState, MeshConfig, dims>{state, mesh};
        auto rigid_op =
            rigid_effect_op_t<HydroState, MeshConfig, dims>{state, mesh};

        const auto bodies = state.bodies;
        return compute_field_t{
          [=] DEV(auto coord) {
              using conserved_t = typename HydroState::conserved_t;

              // check if we have bodies
              if (!bodies.has_value() || bodies->empty()) {
                  return conserved_t{};
              }

              conserved_t total_effect{};

              // visit all bodies and accumulate effects
              bodies->visit_all([&](const auto& body) {
                  using body_type = std::decay_t<decltype(body)>;

                  // gravitational effects
                  if constexpr (has_gravitational_capability_c<body_type>) {
                      total_effect += grav_op.apply_to_body(body, coord);
                  }

                  // accretion effects
                  if constexpr (has_accretion_capability_c<body_type>) {
                      total_effect += accr_op.apply_to_body(body, coord);
                  }

                  // rigid body effects
                  if constexpr (has_rigid_capability_c<body_type>) {
                      total_effect += rigid_op.apply_to_body(body, coord);
                  }
              });

              return total_effect;
          },
          make_domain(mesh.domain.shape())
        };
    }

    // =================================================================
    // Flux Computation Operations
    // =================================================================

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
        constexpr auto dims       = HydroState::dimensions;
        const auto face_domain    = mesh.face_domain[dir];
        const auto gamma          = state.metadata.gamma;
        const auto shock_smoother = state.metadata.shock_smoother;
        const auto plm_theta      = state.metadata.plm_theta;
        const auto prims          = state.prim[mesh.domain];

        return compute_field_t{
          [=] DEV(auto coord) {
              // create stencil for reconstruction around this interface
              const auto stenc = make_stencil<CfdOps::rec_t>(prims, coord, dir);
              const auto [pl, pr] = ops.reconstruct(stenc, plm_theta);
              // normal vector for this dimension
              const auto nhat = unit_vectors::ehat<dims>(dir);
              // face velocity (for moving meshes)
              const auto vface = mesh::face_velocity(coord, dir, mesh);

              if (dir == 1) {
                  auto f = ops.flux(pl, pr, nhat, vface, gamma, shock_smoother);
                  std::cout << "coord: " << coord << std::endl;
                  std::cout << "pl: " << pl << std::endl;
                  std::cout << "pr: " << pr << std::endl;
                  std::cout << "f: " << f << std::endl;
              }

              // solve Riemann problem
              return ops.flux(pl, pr, nhat, vface, gamma, shock_smoother);
          },
          make_domain(face_domain.shape())
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
    auto rk_step(HydroState& state, const MeshConfig& mesh, const CfdOps& ops)
    {
        const auto dt = state.metadata.dt;
        auto u_p      = state.cons[mesh.domain];

        // u' = u + L(u)
        // where L(u) is the godunov operator
        if (state.metadata.timestepping == Timestepping::EULER) {
            update_staggered_fields(state, ops, mesh);

            const auto ell = godunov_op(state, mesh) * dt;
            u_p = u_p.map([=](auto coord, auto u) { return u += ell(coord); });

            boundary::apply_boundary_conditions(state, mesh);
            hydro::recover_primitives(state);
            update_timestep(state, mesh);
        }
        else if (state.metadata.timestepping == Timestepping::RK2) {
            update_staggered_fields(state, ops, mesh);

            //  k1 = L(u^n)
            const auto k1 = godunov_op(state, mesh);

            // u* = u^n + dt*k1
            u_p = u_p.map([k1, dt](auto coord, auto u) {
                return u += k1(coord) * dt;
            });
            boundary::apply_boundary_conditions(state, mesh);
            hydro::recover_primitives(state);
            update_timestep(state, mesh);

            update_staggered_fields(state, ops, mesh);

            // k2 = L(u*)
            auto k2 = godunov_op(state, mesh);

            // u^{n+1} = u* - 0.5*dt*k1 + 0.5*dt*k2 = u* + 0.5*dt*(k2 - k1)
            u_p = u_p.map([k1, k2, dt](auto coord, auto u) {
                return u += 0.5 * dt * (k2(coord) - k1(coord));
            });
            boundary::apply_boundary_conditions(state, mesh);
            hydro::recover_primitives(state);
            update_timestep(state, mesh);
        }
        else {
            throw std::runtime_error(
                "Unsupported timestepping method: " +
                std::to_string(
                    static_cast<std::uint64_t>(state.metadata.timestepping)
                )
            );
        }
        if constexpr (HydroState::is_mhd) {
            // correct energy density from CT algorithm
            em::update_energy_density(state, mesh);
        }
    }

}   // namespace simbi::cfd

#endif   // SIMBI_CFD_OPS_HPP
