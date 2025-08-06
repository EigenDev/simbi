#ifndef BODY_EXPR_EFFECTS_HPP
#define BODY_EXPR_EFFECTS_HPP

#include "base/concepts.hpp"
#include "body.hpp"
#include "body_delta.hpp"
#include "config.hpp"
#include "containers/vector.hpp"
#include "mesh/mesh_ops.hpp"
#include "physics/hydro/physics.hpp"

#include <cmath>
#include <cstdint>

namespace simbi::body::expr {
    using namespace simbi::hydro;

    // ========================================================================
    // individual body effect operations
    // ========================================================================

    // gravitational effect operation
    template <typename HydroState, typename MeshConfig, std::uint64_t Dims>
    struct gravitational_effect_op_t {
        const HydroState& state_;
        const MeshConfig& mesh_;

        template <typename Body, typename Coord>
        auto apply_to_body(const Body& body, Coord coord) const
        {
            using conserved_t = typename HydroState::conserved_t;

            // get physics context
            const auto& ctx = state_.metadata;
            auto cell_pos   = mesh::to_cartesian(coord, mesh_);

            const auto prim = state_.prim[coord];

            // gravitational physics
            const auto r_vec = cell_pos - body.position;
            const auto r_mag = r_vec.norm();

            const auto softening    = softening_length(body);
            const auto softening_sq = softening * softening;
            const auto r_eff        = std::sqrt(r_mag * r_mag + softening_sq);

            // gravitational acceleration (G = 1)
            const auto g_accel = -body.mass * r_vec / (r_eff * r_eff * r_eff);

            // fluid changes
            const auto dt      = state_.metadata.dt;
            const auto density = labframe_density(prim);
            const auto dp_dt   = density * g_accel;
            const auto v_old   = prim.vel;
            const auto v_new =
                (linear_momentum(prim, ctx.gamma) + dp_dt * dt) / density;
            const auto v_avg = 0.5 * (v_old + v_new);
            const auto dE_dt = vecops::dot(v_avg, dp_dt);

            // [TODO]: impl torque from gravity?
            return std::make_pair(
                conserved_t{0.0, dp_dt, dE_dt},
                body_delta_t<Dims>{
                  .idx                  = body.idx,
                  .force_delta          = -dp_dt * mesh::volume(coord, mesh_),
                  .torque_delta         = {},
                  .mass_delta           = 0.0,
                  .accretion_rate_delta = 0.0
                }
            );
        }
    };

    // accretion effect operation
    template <typename HydroState, typename MeshConfig, std::uint64_t Dims>
    struct accretion_effect_op_t {
        const HydroState& state_;
        const MeshConfig& mesh_;

        template <typename Body, typename Coord>
        auto apply_to_body(const Body& body, Coord coord) const
        {
            using conserved_t   = typename HydroState::conserved_t;
            const auto& ctx     = state_.metadata;
            const auto cell_pos = mesh::to_cartesian(coord, mesh_);

            const auto prim  = state_.prim[coord];
            const auto r_vec = cell_pos - body.position;
            const auto r_mag = r_vec.norm();

            const auto accr_radius = accretion_radius(body);

            // skip if too far away
            if (r_mag > 2.5 * accr_radius) {
                return std::make_pair(conserved_t{}, body_delta_t<Dims>{});
            }

            // accretion physics
            const auto accr_eff            = accretion_efficiency(body);
            const auto cell_size           = mesh::max_cell_width(coord, mesh_);
            const auto local_cs            = sound_speed(prim, ctx.gamma);
            const auto sound_crossing_time = cell_size / local_cs;
            const auto stability_limit     = ctx.dt / sound_crossing_time;
            const auto eps = std::min({accr_eff, 0.5, stability_limit});
            // for now, I will set the sink rate to the inverse of sound
            // crossing time [TODO]: make this configurable
            const auto sr = 1.0 / sound_crossing_time;

            // calculate accreted quantities
            const auto den_dot = eps * labframe_density(prim) * sr;
            const auto mom_dot = eps * linear_momentum(prim, ctx.gamma) * sr;
            const auto power   = eps * energy_density(prim, ctx.gamma) * sr;

            const auto dv          = mesh::volume(coord, mesh_);
            const auto force_delta = -mom_dot * dv * ctx.dt;
            auto torque_delta      = [&]() -> vector_t<real, 3> {
                if constexpr (Dims == 3) {
                    return vecops::cross(r_vec, force_delta) * dv;
                }
                else if constexpr (Dims == 2) {
                    return vector_t<real, 3>{
                      0,
                      0,
                      r_vec[0] * force_delta[1] - r_vec[1] * force_delta[0]
                    };
                }
                else {
                    return vector_t<real, 3>{};
                }
            }();

            return std::make_pair(
                conserved_t{-den_dot, -mom_dot, -power},
                body_delta_t<Dims>{
                  .idx                  = body.idx,
                  .force_delta          = force_delta,
                  .torque_delta         = std::move(torque_delta),
                  .mass_delta           = den_dot * dv * ctx.dt,
                  .accretion_rate_delta = den_dot * dv
                }
            );
        }
    };

    // rigid body effect operation
    template <typename HydroState, typename MeshConfig, std::uint64_t Dims>
    struct rigid_effect_op_t {
        const HydroState& state_;
        const MeshConfig& mesh_;

        template <typename Body, typename Coord>
        auto apply_to_body(const Body& body, Coord coord) const
        {
            using conserved_t         = typename HydroState::conserved_t;
            const auto& ctx           = state_.metadata;
            const auto cell_pos       = mesh::to_cartesian(coord, mesh_);
            const auto min_cell_width = mesh::min_cell_width(coord, mesh_);

            const auto prim     = state_.prim[coord];
            const auto r_vec    = cell_pos - body.position;
            const auto distance = r_vec.norm();

            // early exit if too far from body
            constexpr real SAFE_MINIMUM = 1e-10;
            const auto r_norm           = std::max(SAFE_MINIMUM, distance);
            const auto r_hat            = r_vec / r_norm;
            const auto signed_distance  = distance - body.radius;

            // get fluid properties
            const auto density         = labframe_density(prim);
            const auto sound_speed_val = sound_speed(prim, ctx.gamma);
            const auto fluid_velocity  = prim.vel;
            const auto mach_number =
                fluid_velocity.norm() / std::max(sound_speed_val, SAFE_MINIMUM);

            // calculate boundary thickness
            real boundary_thickness =
                (mach_number > 1.0) ? 0.5 * min_cell_width : min_cell_width;

            const real extended_radius =
                body.radius + ((mach_number > 1.0) ? 2.0 * boundary_thickness
                                                   : boundary_thickness);

            // skip if outside influence region
            if (distance > extended_radius + boundary_thickness) {
                return std::make_pair(conserved_t{}, body_delta_t<Dims>{});
            }

            // rigid body forcing physics
            const auto body_velocity = body.velocity;
            const auto rel_velocity  = fluid_velocity - body_velocity;
            const auto normal_rel_velocity =
                vecops::dot(rel_velocity, r_hat) * r_hat;

            // calculate forcing strength
            real base_strength =
                (mach_number > 1.0)
                    ? 25.0 * density * sound_speed_val * sound_speed_val
                    : 10.0 * density * sound_speed_val * sound_speed_val;

            vector_t<real, Dims> total_force{};

            if (signed_distance < 0) {
                // inside body - strong forcing
                const real depth_ratio =
                    std::abs(signed_distance) / body.radius;
                const real interior_factor =
                    1.0 + 10.0 * depth_ratio * depth_ratio;
                total_force = -rel_velocity * base_strength * interior_factor;
            }
            else if (signed_distance < boundary_thickness) {
                // boundary region
                const real boundary_factor =
                    1.0 - signed_distance / boundary_thickness;
                const real sharp_factor = std::pow(boundary_factor, 3);

                // check if body has no-slip (would need to access rigid
                // component) for now, assume no-slip
                total_force = -rel_velocity * base_strength * sharp_factor;
            }
            else if (mach_number > 1.0 &&
                     signed_distance < 2.0 * boundary_thickness) {
                // pre-emptive zone for supersonic flows
                const real pre_factor =
                    1.0 -
                    (signed_distance - boundary_thickness) / boundary_thickness;
                const real pre_strength =
                    0.5 * base_strength * std::pow(pre_factor, 2);

                const real incoming_velocity =
                    -std::min(real{0}, vecops::dot(rel_velocity, r_hat));
                if (incoming_velocity > 0.1 * sound_speed_val) {
                    total_force = -normal_rel_velocity * pre_strength;
                }
            }

            // convert force to momentum change
            const auto dp = total_force * ctx.dt;

            // calculate energy change
            const auto invd  = 1.0 / density;
            const auto v_new = (linear_momentum(prim, ctx.gamma) + dp) * invd;
            const auto v_avg = 0.5 * (fluid_velocity + v_new);
            const auto dE    = vecops::dot(v_avg, dp);
            const auto dv    = mesh::volume(coord, mesh_);
            auto torque      = [&]() -> vector_t<real, 3> {
                if constexpr (Dims == 3) {
                    return vecops::cross(r_vec, total_force) * dv;
                }
                else if constexpr (Dims == 2) {
                    return vector_t<real, 3>{
                      0,
                      0,
                      r_vec[0] * total_force[1] - r_vec[1] * total_force[0]
                    };
                }
                else {
                    return vector_t<real, 3>{};
                }
            }();

            return std::make_pair(
                conserved_t{0.0, dp, dE},
                body_delta_t<Dims>{
                  .idx                  = body.idx,
                  .force_delta          = -total_force * dv,
                  .torque_delta         = std::move(torque),
                  .mass_delta           = 0.0,
                  .accretion_rate_delta = 0.0
                }
            );
        }
    };
}   // namespace simbi::body::expr

#endif   // BODY_EXPR_EFFECTS_HPP
