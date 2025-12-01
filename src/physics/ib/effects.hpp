#ifndef BODY_EXPR_NEFFECTS_HPP
#define BODY_EXPR_NEFFECTS_HPP

#include "base/concepts.hpp"
#include "body.hpp"
#include "body_delta.hpp"
#include "compat.hpp"
#include "containers/vector.hpp"
#include "physics/hydro/physics.hpp"
#include "utility/helpers.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>

namespace simbi::body::expr {
    using namespace simbi::hydro;

    // ========================================================================
    // gravitational effect operation (geometry-based)
    // ========================================================================
    template <typename Primitive, typename Geometry>
    struct grav_op_t {
        using conserved_t                   = Primitive::counterpart_t;
        static constexpr std::uint64_t Rank = Primitive::rank;

        Primitive prim;
        Geometry geometry;

        template <typename Body, typename Coord>
        constexpr DEV auto operator()(const Body& body, Coord coord) const
        {
            const auto cell_pos =
                geometry.metric.to_cartesian(geometry.centroid(coord));

            // gravitational physics
            const auto r_vec = cell_pos - body.position;
            const auto r_mag = r_vec.norm();

            const auto softening    = softening_length(body);
            const auto softening_sq = softening * softening;
            const auto r_eff        = std::sqrt(r_mag * r_mag + softening_sq);

            // gravitational acceleration (G = 1)
            const auto g_cart  = body.mass * r_vec / (r_eff * r_eff * r_eff);
            const auto g_accel = -geometry.metric.from_cartesian(g_cart);

            // fluid changes
            const auto density = labframe_density(prim);
            const auto dp_dt   = density * g_accel;
            const auto dE_dt   = vecops::dot(prim.vel, dp_dt);

            return std::make_pair(
                conserved_t{0.0, dp_dt, dE_dt},
                body_delta_t<Rank>{
                  .idx          = body.idx,
                  .force_delta  = -dp_dt * geometry.volume(coord),
                  .torque_delta = {},
                  .mass_delta   = 0.0
                }
            );
        }
    };

    // ========================================================================
    // accretion effect operation (geometry-based)
    // ========================================================================
    template <typename Primitive, typename Geometry>
    struct accretion_op_t {
        using conserved_t                   = Primitive::counterpart_t;
        static constexpr std::uint64_t Rank = Primitive::rank;

        Primitive prim;
        Geometry geometry;
        real gamma;
        real dt;

        // torque control from Dittmann & Ryan (2021)
        // delta = 0 (torque-free), delta = 1 (standard sink)
        constexpr DEV auto apply_torque_control(
            const vector_t<real, Rank>& r_hat,
            const vector_t<real, Rank>& v_sink,
            const vector_t<real, Rank>& v_parc,
            real delta = 0.0
        ) const
        {
            const auto gas_vel_cart = geometry.metric.to_cartesian(v_parc);
            const auto v_rel_cart   = gas_vel_cart - v_sink;
            const auto v_rad_comp   = vecops::dot(v_rel_cart, r_hat);
            const auto v_rad_cart   = v_rad_comp * r_hat;
            const auto v_angular    = v_rel_cart - v_rad_cart;
            const auto v_star_cart  = v_rad_cart + delta * v_angular + v_sink;
            return geometry.metric.from_cartesian(v_star_cart);
        }

        template <typename Coord>
        constexpr DEV auto min_cell_width(Coord coord) const
        {
            const auto h = geometry.scale_factors(coord);
            // approximate cell width from scale factors and unit spacing
            real min_width = h[0];
            for (std::size_t dd = 1; dd < Rank; ++dd) {
                if (h[dd] < min_width) {
                    min_width = h[dd];
                }
            }
            return min_width;
        }

        template <typename Body, typename Coord>
        constexpr DEV auto operator()(
            const Body& body,
            Coord coord,
            bool is_binary,
            real mdot_target = 0.0,
            real w_total     = 0.0,
            real /*r_bh*/    = 0.0
        ) const
        {
            using namespace simbi::helpers;
            const auto cell_pos =
                geometry.metric.to_cartesian(geometry.centroid(coord));
            const auto r_vec    = cell_pos - body.position;
            const auto r_mag    = r_vec.norm();
            const auto r_acc    = accretion_radius(body);
            const auto sr_param = sink_rate(body);
            const auto dv       = geometry.volume(coord);

            // quick exit if outside accretion radius
            if (r_mag > 4.0 * r_acc) {
                return std::make_pair(
                    conserved_t{},
                    body_delta_t<Rank>{
                      .idx          = body.idx,
                      .force_delta  = {},
                      .torque_delta = {},
                      .mass_delta   = 0.0
                    }
                );
            }

            real den_dot;
            if (std::abs(sr_param) != 0.0) {
                const auto weight = [is_binary, r_mag, r_acc]() {
                    if (is_binary) {
                        const auto r_norm = r_mag / r_acc;
                        return std::exp(-0.25 * std::pow(r_norm, 4));
                    }
                    const auto r_kernel = 0.5 * r_acc;
                    const auto r_norm   = r_mag / r_kernel;
                    return std::exp(-r_norm * r_norm);
                }();

                // physical timescales
                const auto cell_size           = min_cell_width(coord);
                const auto local_cs            = sound_speed(prim, gamma);
                const auto sound_crossing_time = cell_size / local_cs;

                // free-fall time to sink center
                const auto t_ff =
                    (r_mag > 1e-10)
                        ? std::sqrt(r_mag * r_mag * r_mag / (2.0 * body.mass))
                        : sound_crossing_time;

                const auto t_natural = std::min(sound_crossing_time, t_ff);
                const auto nat_rate  = 1.0 / t_natural;
                const auto stab_rate = 1.0 / dt;
                const auto sr_base   = my_min3(sr_param, nat_rate, stab_rate);
                const auto sr        = sr_base * weight;
                den_dot              = labframe_density(prim) * sr;
            }
            else {
                const auto r_kernel    = 0.5 * r_acc;
                const auto r_norm      = r_mag / r_kernel;
                const auto weight      = std::exp(-r_norm * r_norm);
                const auto norm_weight = weight / w_total;
                den_dot                = mdot_target * (norm_weight / dv);

                // no mass reduced by more than 25% per timestep
                if (den_dot > 0.0) {
                    const auto rho_n         = labframe_density(prim);
                    const auto delta_rho     = den_dot * dt;
                    const auto delta_rho_lim = my_min(delta_rho, 0.25 * rho_n);
                    const auto eta           = delta_rho_lim / delta_rho;
                    den_dot                  = den_dot * eta;
                }
            }

            // torque-controlled sink prescription
            const auto v_star = apply_torque_control(
                r_vec / r_mag,
                body.velocity,
                prim.vel,
                sink_delta(body)
            );
            const auto mom_dot = den_dot * v_star;
            const auto ke_dot  = 0.5 * den_dot * vecops::dot(v_star, v_star);
            const auto ie_dot = den_dot * specific_internal_energy(prim, gamma);
            const auto nrg_dot = ke_dot + ie_dot;

            // force and torque from momentum removal
            const auto force_delta  = -mom_dot * dv;
            const auto torque_delta = [&]() -> vector_t<real, 3> {
                if constexpr (Rank > 2) {
                    return vecops::cross(r_vec, force_delta);
                }
                else {
                    return vector_t<real, 3>{};
                }
            }();

            return std::make_pair(
                conserved_t{-den_dot, -mom_dot, -nrg_dot},
                body_delta_t<Rank>{
                  .idx          = body.idx,
                  .force_delta  = std::move(force_delta),
                  .torque_delta = std::move(torque_delta),
                  .mass_delta   = den_dot * dv * dt
                }
            );
        }
    };

    // ========================================================================
    // rigid body effect operation (geometry-based)
    // ========================================================================
    template <typename Primitive, typename Geometry>
    struct rigid_op_t {
        using conserved_t                   = Primitive::counterpart_t;
        static constexpr std::uint64_t Rank = Primitive::rank;

        Primitive prim;
        Geometry geometry;
        real gamma;

        template <typename Coord>
        constexpr DEV auto min_cell_width(Coord coord) const
        {
            const auto h   = geometry.scale_factors(coord);
            real min_width = h[0];
            for (std::size_t dd = 1; dd < Rank; ++dd) {
                if (h[dd] < min_width) {
                    min_width = h[dd];
                }
            }
            return min_width;
        }

        template <typename Body, typename Coord>
        constexpr DEV auto operator()(const Body& body, Coord coord) const
        {
            using namespace simbi::helpers;
            const auto cell_pos =
                geometry.metric.to_cartesian(geometry.centroid(coord));
            const auto min_cw = min_cell_width(coord);

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
                (mach_number > 1.0) ? 0.5 * min_cw : min_cw;

            const real extended_radius =
                body.radius + ((mach_number > 1.0) ? 2.0 * boundary_thickness
                                                   : boundary_thickness);

            // skip if outside influence region
            if (distance > extended_radius + boundary_thickness) {
                return std::make_pair(conserved_t{}, body_delta_t<Rank>{});
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

            vector_t<real, Rank> dp_dt{};

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
            const auto dv    = geometry.volume(coord);
            auto torque      = [&]() -> vector_t<real, 3> {
                if constexpr (Rank == 3) {
                    return vecops::cross(r_vec, dp_dt) * dv;
                }
                else if constexpr (Rank == 2) {
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
                body_delta_t<Rank>{
                  .idx          = body.idx,
                  .force_delta  = -dp_dt,
                  .torque_delta = std::move(torque),
                  .mass_delta   = 0.0
                }
            );
        }
    };

    // ========================================================================
    // factory helpers
    // ========================================================================
    template <typename Primitive, typename Geometry>
    DEV auto make_grav_op(const Primitive& prim, const Geometry& geo)
    {
        return grav_op_t<Primitive, Geometry>{prim, geo};
    }

    template <typename Primitive, typename Geometry>
    DEV auto make_accretion_op(
        const Primitive& prim,
        const Geometry& geo,
        real gamma,
        real dt
    )
    {
        return accretion_op_t<Primitive, Geometry>{prim, geo, gamma, dt};
    }

    template <typename Primitive, typename Geometry>
    DEV auto
    make_rigid_op(const Primitive& prim, const Geometry& geo, real gamma)
    {
        return rigid_op_t<Primitive, Geometry>{prim, geo, gamma};
    }

}   // namespace simbi::body::expr

#endif   // BODY_EXPR_NEFFECTS_HPP
