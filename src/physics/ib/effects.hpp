#ifndef BODY_EXPR_EFFECTS_HPP
#define BODY_EXPR_EFFECTS_HPP

#include "base/concepts.hpp"
#include "body.hpp"
#include "body_delta.hpp"
#include "compat.hpp"
#include "containers/vector.hpp"
#include "mesh/mesh_ops.hpp"
#include "physics/hydro/physics.hpp"
#include "utility/helpers.hpp"

#include <cmath>
#include <cstdint>

namespace simbi::body::expr {
    using namespace simbi::hydro;

    // ========================================================================
    // individual body effect operations
    // ========================================================================

    // gravitational effect operation
    template <typename Primitive, typename MeshConfig>
    struct grav_op_t {
        using conserved_t                   = Primitive::counterpart_t;
        static constexpr std::uint64_t Dims = Primitive::dimensions;

        Primitive prim;
        MeshConfig mesh_;

        template <typename Body, typename Coord>
        constexpr DEV auto operator()(const Body& body, Coord coord) const
        {
            const auto cell_pos = mesh::to_cartesian(coord, mesh_);
            // gravitational physics
            const auto r_vec = cell_pos - body.position;
            const auto r_mag = r_vec.norm();

            const auto softening    = softening_length(body);
            const auto softening_sq = softening * softening;
            const auto r_eff        = std::sqrt(r_mag * r_mag + softening_sq);

            // gravitational acceleration (G = 1)
            const auto g_cart  = body.mass * r_vec / (r_eff * r_eff * r_eff);
            const auto g_accel = -vecops::centralize(g_cart, mesh_.geometry);

            // fluid changes
            const auto density = labframe_density(prim);
            const auto dp_dt   = density * g_accel;
            const auto dE_dt   = vecops::dot(prim.vel, dp_dt);

            return std::make_pair(
                conserved_t{0.0, dp_dt, dE_dt},
                body_delta_t<Dims>{
                  .idx          = body.idx,
                  .force_delta  = -dp_dt * mesh::volume(coord, mesh_),
                  .torque_delta = {},
                  .mass_delta   = 0.0
                }
            );
        }
    };

    // accretion effect operation
    template <typename Primitive, typename MeshConfig>
    struct accretion_op_t {
        using conserved_t                   = Primitive::counterpart_t;
        static constexpr std::uint64_t Dims = Primitive::dimensions;

        Primitive prim;
        MeshConfig mesh_;
        real gamma;
        real dt;

        /**
         * torque control described in Section 2.1 of Dittmann & Ryan (2021)
         * https://ui.adsabs.harvard.edu/abs/2021ApJ...921...71D/abstract
         * delta = 0 (torque-free)
         * delta = 1 (standatd sink)
         */
        constexpr DEV auto apply_torque_control(
            const vector_t<real, Dims>& r_hat,
            const vector_t<real, Dims>& v_sink,
            const vector_t<real, Dims>& v_parc,
            real delta = 0.0
        ) const
        {
            const auto gas_vel_cart = mesh::to_cartesian(v_parc, mesh_);
            const auto v_rel_cart   = gas_vel_cart - v_sink;
            const auto v_rad_comp   = vecops::dot(v_rel_cart, r_hat);
            const auto v_rad_cart   = v_rad_comp * r_hat;
            const auto v_angular    = v_rel_cart - v_rad_cart;
            const auto v_star_cart  = v_rad_cart + delta * v_angular + v_sink;
            return mesh::from_cartesian(v_star_cart, mesh_);
        }

        template <typename Body, typename Coord>
        constexpr DEV auto
        operator()(const Body& body, Coord coord, bool is_binary) const
        {
            using namespace simbi::helpers;
            const auto cell_pos    = mesh::to_cartesian(coord, mesh_);
            const auto r_vec       = cell_pos - body.position;
            const auto r_mag       = r_vec.norm();
            const auto accr_radius = accretion_radius(body);

            // physical timescales
            const auto cell_size           = mesh::min_cell_width(coord, mesh_);
            const auto local_cs            = sound_speed(prim, gamma);
            const auto sound_crossing_time = cell_size / local_cs;

            // free-fall time to sink center (gravitational timescale)
            const auto t_ff =
                (r_mag > 1e-10)
                    ? std::sqrt(r_mag * r_mag * r_mag / (2.0 * body.mass))
                    : sound_crossing_time;   // fallback for r→0

            // use the shorter of the two natural timescales
            const auto t_natural = std::min(sound_crossing_time, t_ff);

            // stability limits (maybe? TODO: revisit...)
            const auto nat_rate = 1.0 / t_natural;   // [1/time] - physical rate
            const auto sr_param = sink_rate(body);   // [1/time] - desired rate
            const auto stability_rate = 1.0 / dt;    // [1/time] - max safe rate
            const auto sr_base = my_min3(sr_param, nat_rate, stability_rate);

            // torque-controlled sink prescription from Dittmann & Ryan(2021)
            // https://ui.adsabs.harvard.edu/abs/2021ApJ...921...71D/abstract
            const auto r_norm         = r_mag / accr_radius;
            const auto radial_profile = [is_binary, r_norm]() {
                if (is_binary) {
                    return std::exp(-0.25 * std::pow(r_norm, 4));
                }
                return std::exp(-0.5 * r_norm * r_norm);
            }();
            const auto sr = sr_base * radial_profile;

            const auto v_star = apply_torque_control(
                r_vec / r_mag,
                body.velocity,
                prim.vel,
                sink_delta(body)
            );
            const auto den_dot = labframe_density(prim) * sr;
            const auto mom_dot = den_dot * v_star;
            const auto ke_dot  = 0.5 * den_dot * vecops::dot(v_star, v_star);
            const auto ie_dot = den_dot * specific_internal_energy(prim, gamma);
            const auto nrg_dot = ke_dot + ie_dot;

            // force and torque from momentum removal
            const auto dv           = mesh::volume(coord, mesh_);
            const auto force_delta  = -mom_dot * dv;
            const auto torque_delta = [&]() -> vector_t<real, 3> {
                if constexpr (Dims > 2) {
                    return vecops::cross(r_vec, force_delta);
                }
                else {
                    return vector_t<real, 3>{};
                }
            }();

            return std::make_pair(
                conserved_t{-den_dot, -mom_dot, -nrg_dot},
                body_delta_t<Dims>{
                  .idx          = body.idx,
                  .force_delta  = std::move(force_delta),
                  .torque_delta = std::move(torque_delta),
                  .mass_delta   = den_dot * dv * dt
                }
            );
        }
    };

    // rigid body effect operation
    template <typename Primitive, typename MeshConfig>
    struct rigid_op_t {
        using conserved_t                   = Primitive::counterpart_t;
        static constexpr std::uint64_t Dims = Primitive::dimensions;

        Primitive prim;
        MeshConfig mesh_;
        real gamma;

        template <typename Body, typename Coord>
        constexpr DEV auto operator()(const Body& body, Coord coord) const
        {
            using namespace simbi::helpers;
            const auto cell_pos       = mesh::to_cartesian(coord, mesh_);
            const auto min_cell_width = mesh::min_cell_width(coord, mesh_);

            const auto r_vec    = cell_pos - body.position;
            const auto distance = r_vec.norm();

            // early exit if too far from body
            constexpr real SAFE_MINIMUM = 1e-10;
            const auto r_norm           = my_max(SAFE_MINIMUM, distance);
            const auto r_hat            = r_vec / r_norm;
            const auto signed_distance  = distance - body.radius;

            // get fluid properties
            const auto density         = labframe_density(prim);
            const auto sound_speed_val = sound_speed(prim, gamma);
            const auto fluid_velocity  = prim.vel;
            const auto mach_number =
                fluid_velocity.norm() / my_max(sound_speed_val, SAFE_MINIMUM);

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

            vector_t<real, Dims> dp_dt{};

            if (signed_distance < 0) {
                // inside body - strong forcing
                const real depth_ratio =
                    std::abs(signed_distance) / body.radius;
                const real interior_factor =
                    1.0 + 10.0 * depth_ratio * depth_ratio;
                dp_dt = -rel_velocity * base_strength * interior_factor;
            }
            else if (signed_distance < boundary_thickness) {
                // boundary region
                const real boundary_factor =
                    1.0 - signed_distance / boundary_thickness;
                const real sharp_factor = std::pow(boundary_factor, 3);

                // check if body has no-slip (would need to access rigid
                // component) for now, assume no-slip
                dp_dt = -rel_velocity * base_strength * sharp_factor;
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
                    -my_min(real{0}, vecops::dot(rel_velocity, r_hat));
                if (incoming_velocity > 0.1 * sound_speed_val) {
                    dp_dt = -normal_rel_velocity * pre_strength;
                }
            }

            // calculate energy change
            const auto dE_dt = vecops::dot(prim.vel, dp_dt);
            const auto dv    = mesh::volume(coord, mesh_);
            auto torque      = [&]() -> vector_t<real, 3> {
                if constexpr (Dims == 3) {
                    return vecops::cross(r_vec, dp_dt) * dv;
                }
                else if constexpr (Dims == 2) {
                    return vector_t<real, 3>{
                      0,
                      0,
                      r_vec[0] * dp_dt[1] - r_vec[1] * dp_dt[0]
                    };
                }
                else {
                    return vector_t<real, 3>{};
                }
            }();

            return std::make_pair(
                conserved_t{0.0, dp_dt, dE_dt},
                body_delta_t<Dims>{
                  .idx          = body.idx,
                  .force_delta  = -dp_dt,
                  .torque_delta = std::move(torque),
                  .mass_delta   = 0.0
                }
            );
        }
    };
}   // namespace simbi::body::expr

#endif   // BODY_EXPR_EFFECTS_HPP
