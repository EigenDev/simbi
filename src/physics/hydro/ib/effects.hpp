#ifndef SIMBI_BODY_EXPR_EFFECTS_HPP
#define SIMBI_BODY_EXPR_EFFECTS_HPP

#include "body.hpp"
#include "config.hpp"
#include "core/base/concepts.hpp"
#include "data/containers/vector.hpp"
#include "physics/hydro/physics.hpp"
#include "system/mesh/mesh_ops.hpp"
#include <cmath>
#include <cstdint>
#include <type_traits>

namespace simbi::body::expr {
    using namespace simbi::hydro;

    // ========================================================================
    // physics context for body interactions
    // ========================================================================

    template <std::uint64_t Dims>
    struct physics_context_t {
        real gamma;
        real dt;
        vector_t<real, Dims> cell_pos;
        real cell_volume;
        real min_cell_width;
        real max_cell_width;
    };

    // ========================================================================
    // individual body effect operations
    // ========================================================================

    // gravitational effect operation
    template <typename HydroState, std::uint64_t Dims>
    struct gravitational_effect_op_t {
        const HydroState& state_;

        template <typename Body, typename Coord>
        auto apply_to_body(const Body& body, Coord coord) const
        {
            using conserved_t = typename HydroState::conserved_t;
            // using primitive_t = typename HydroState::primitive_t;

            // get physics context
            auto ctx = physics_context_t<Dims>{
              .gamma          = state_.metadata.gamma,
              .dt             = state_.metadata.dt,
              .cell_pos       = mesh::centroid(coord, state_.mesh),
              .cell_volume    = mesh::volume(coord, state_.mesh),
              .min_cell_width = mesh::min_cell_width(coord, state_.mesh),
              .max_cell_width = mesh::max_cell_width(coord, state_.mesh)
            };

            const auto prim = state_.prim[coord];

            // gravitational physics
            const auto r_vec = ctx.cell_pos - body.position;
            const auto r_mag = r_vec.norm();

            // get softening length using your accessor
            const auto softening    = softening_length(body);
            const auto softening_sq = softening * softening;
            const auto r_eff        = std::sqrt(r_mag * r_mag + softening_sq);

            // gravitational acceleration (G = 1)
            const auto g_accel = -body.mass * r_vec / (r_eff * r_eff * r_eff);

            // fluid changes
            const auto density         = labframe_density(prim);
            const auto momentum_change = density * g_accel * ctx.dt;
            const auto energy_change   = vecops::dot(prim.vel, momentum_change);

            return conserved_t{0.0, momentum_change, energy_change};
        }
    };

    // accretion effect operation
    template <typename HydroState, std::uint64_t Dims>
    struct accretion_effect_op_t {
        const HydroState& state_;

        template <typename Body, typename Coord>
        auto apply_to_body(const Body& body, Coord coord) const
        {
            using conserved_t = typename HydroState::conserved_t;

            // get physics context
            auto ctx = physics_context_t<Dims>{
              .gamma          = state_.metadata.gamma,
              .dt             = state_.metadata.dt,
              .cell_pos       = mesh::centroid(coord, state_.mesh),
              .cell_volume    = mesh::volume(coord, state_.mesh),
              .min_cell_width = mesh::min_cell_width(coord, state_.mesh),
              .max_cell_width = mesh::max_cell_width(coord, state_.mesh)
            };

            const auto prim  = state_.prim[coord];
            const auto r_vec = ctx.cell_pos - body.position;
            const auto r_mag = r_vec.norm();

            // get accretion properties using your accessors
            const auto accr_radius = accretion_radius(body);

            // skip if too far away
            if (r_mag > 2.5 * accr_radius) {
                return conserved_t{};
            }

            // accretion physics
            const auto accr_eff            = accretion_efficiency(body);
            const auto cell_size           = ctx.max_cell_width;
            const auto local_cs            = sound_speed(prim, ctx.gamma);
            const auto sound_crossing_time = cell_size / local_cs;
            const auto stability_limit     = ctx.dt / sound_crossing_time;
            const auto eps =
                std::min(accr_eff, std::min(real{0.5}, stability_limit));

            // calculate accreted quantities
            const auto accreted_density = eps * labframe_density(prim);
            const auto accreted_momentum =
                eps * linear_momentum(prim, ctx.gamma);
            const auto accreted_energy = eps * energy_density(prim, ctx.gamma);

            return conserved_t{
              -accreted_density,
              -accreted_momentum,
              -accreted_energy
            };
        }
    };

    // rigid body effect operation
    template <typename HydroState, std::uint64_t Dims>
    struct rigid_effect_op_t {
        const HydroState& state_;

        template <typename Body, typename Coord>
        auto apply_to_body(const Body& body, Coord coord) const
        {
            using conserved_t = typename HydroState::conserved_t;

            // get physics context
            auto ctx = physics_context_t<Dims>{
              .gamma          = state_.metadata.gamma,
              .dt             = state_.metadata.dt,
              .cell_pos       = mesh::centroid(coord, state_.mesh),
              .cell_volume    = mesh::volume(coord, state_.mesh),
              .min_cell_width = mesh::min_cell_width(coord, state_.mesh),
              .max_cell_width = mesh::max_cell_width(coord, state_.mesh)
            };

            const auto prim     = state_.prim[coord];
            const auto r_vec    = ctx.cell_pos - body.position;
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
            real boundary_thickness = (mach_number > 1.0)
                                          ? 0.5 * ctx.min_cell_width
                                          : ctx.min_cell_width;

            const real extended_radius =
                body.radius + ((mach_number > 1.0) ? 2.0 * boundary_thickness
                                                   : boundary_thickness);

            // skip if outside influence region
            if (distance > extended_radius + boundary_thickness) {
                return conserved_t{};
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

            return conserved_t{0.0, dp, dE};
        }
    };

    // ========================================================================
    // unified body effects operation
    // ========================================================================

    template <typename HydroState>
    struct body_effects_op_t {
        const HydroState& state_;

        template <typename Coord>
        auto operator()(Coord coord) const
        {
            using conserved_t   = typename HydroState::conserved_t;
            constexpr auto dims = HydroState::dimensions;

            // check if we have bodies
            if (!state_.bodies.has_value() || state_.bodies->empty()) {
                return conserved_t{};
            }

            conserved_t total_effect{};

            // create effect operators
            auto grav_op  = gravitational_effect_op_t<HydroState, dims>{state_};
            auto accr_op  = accretion_effect_op_t<HydroState, dims>{state_};
            auto rigid_op = rigid_effect_op_t<HydroState, dims>{state_};

            // visit all bodies and accumulate effects
            state_.bodies->visit_all([&](const auto& body) {
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
        }
    };
}   // namespace simbi::body::expr

#endif   // SIMBI_BODY_EXPR_EFFECTS_HPP
