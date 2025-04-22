/**
 * @file fluid_interaction_functions.hpp
 * @brief Pure functions for fluid-body interactions in a functional style
 */
#ifndef INTERACTION_FUNCTIONS_HPP
#define INTERACTION_FUNCTIONS_HPP

#include "body_delta.hpp"
#include "build_options.hpp"   // for real, size_type, DEV
#include "core/types/containers/array.hpp"
#include "core/types/containers/vector.hpp"            // for spatial_vector_t
#include "geometry/mesh/cell.hpp"                      // for Cell
#include "physics/hydro/schemes/ib/systems/body.hpp"   // for Body
#include "physics/hydro/schemes/ib/systems/component_body_system.hpp"   // for ComponentBodySystem
#include "physics/hydro/types/context.hpp"   // for HydroContext
#include <cmath>                             // for std::sqrt
#include <tuple>                             // for std::tuple
using namespace simbi::vecops;
using namespace simbi::ibsystem;

namespace simbi::ibsystem::body_functions {
    namespace gravitational {

        template <typename T, size_type Dims, typename Primitive>
        DEV std::pair<typename Primitive::counterpart_t, BodyDelta<T, Dims>>
        apply_gravitational_force(
            size_type body_idx,
            const Body<T, Dims>& body,
            const Primitive& prim,
            const Cell<Dims>& mesh_cell,
            const HydroContext& context,
            T dt
        )
        {
            using conserved_t = Primitive::counterpart_t;
            // Calculate distance vector from body to cell
            const auto softening_length = body.softening_length();
            const auto r = mesh_cell.cartesian_centroid() - body.position;
            const auto r2 =
                vecops::dot(r, r) + softening_length * softening_length;
            // Gravitational force on fluid element (G = 1)
            const auto f_cart = body.mass * r / (r2 * std::sqrt(r2));

            // Centralize force based on geometry
            const auto ff_hat = -centralize(f_cart, mesh_cell.geometry());

            // Calculate momentum and energy change
            const auto dp = prim.labframe_density() * ff_hat * dt;

            const auto v_old = prim.velocity();
            const auto v_new = (prim.spatial_momentum(context.gamma) + dp) /
                               prim.labframe_density();
            const auto v_avg = 0.5 * (v_old + v_new);
            const auto dE    = vecops::dot(v_avg, dp);

            // Apply two-way coupling if enabled
            BodyDelta<T, Dims> delta{body_idx, {}, 0, 0, 0};
            if (body.two_way_coupling()) {
                // all vector quantities for the body
                // are in Cartesian coordinates
                delta.force_delta =
                    prim.labframe_density() * mesh_cell.volume() * f_cart;
            }

            return {conserved_t(0.0, dp, dE), delta};
        }

    }   // namespace gravitational

    namespace accretion {

        // Cubic spline smoothing kernel
        template <typename T>
        DEV T smoothing_kernel(const T r_norm)
        {
            if (r_norm >= 1.0) {
                return 0.0;
            }
            const T q = 1.0 - r_norm;
            return q * q * (1.0 + 2.0 * r_norm);
        }

        // Calculate radial accretion profile
        template <typename T, size_type Dims>
        DEV T calculate_radial_accretion_profile(
            const T distance,
            const T r_bondi,
            const T accretion_efficiency,
            const T gamma,
            const spatial_vector_t<T, Dims>& rel_position,
            const spatial_vector_t<T, Dims>& rel_velocity,
            const T local_cs,
            const T local_j_specific = 0.0,
            const bool isothermal    = true
        )
        {
            // ======== BONDI-HOYLE-LYTTLETON EFFECTS [:^)] ========
            // For bodies moving relative to gas
            // from: Edgar (2004) -
            // https://doi.org/10.1016/j.newar.2004.06.001

            const bool use_local_conditions = (local_cs > 0.0);
            const T vel_mag                 = rel_velocity.norm();
            T mach_number                   = 0.0;
            T effective_r_bondi             = r_bondi;
            T direction_factor              = 1.0;

            if (use_local_conditions && vel_mag > 0.0) {
                mach_number = vel_mag / local_cs;

                // Bondi-Hoyle radius modification
                // from: Ruffert & Arnett (1994) -
                // https://ui.adsabs.harvard.edu/abs/1994ApJ...427..351R/abstract
                // and: Ruffert (1994) Eq. (14)
                // https://ui.adsabs.harvard.edu/abs/1994ApJ...427..342R/abstract
                effective_r_bondi = r_bondi / (1.0 + mach_number * mach_number);

                // directional dependence - enhance upstream, reduce downstream
                // from: Blondin & Raymer (2012) -
                // https://doi.org/10.1088/0004-637X/752/1/30
                if (mach_number > 0.1 && rel_position.norm() > 0.0) {
                    // Cosine of angle between position and velocity vectors
                    T cos_angle = vecops::dot(rel_position, rel_velocity) /
                                  (vecops::norm(rel_position) * vel_mag);

                    // directional weighting (strongest upstream, weakest
                    // downstream) Factor ranges from ~0.5 (downstream) to ~1.5
                    // (upstream)
                    direction_factor =
                        1.0 + 0.5 * mach_number *
                                  std::min(1.0, std::max(-1.0, cos_angle));
                }
            }

            // ======== ANGULAR MOMENTUM BARRIER ========
            // From: Hubber et al. (2013) -
            // https://doi.org/10.1093/mnras/stt128

            T j_factor = 1.0;
            if (local_j_specific > 0.0) {
                // calculate circularization radius from specific angular
                // momentum r_circ = j * j  / M - radius where centrifugal force
                // balances gravity
                const T min_distance  = 1e-10 * r_bondi;
                const T safe_distance = std::max(distance, min_distance);
                const T r_circ =
                    local_j_specific * local_j_specific / safe_distance;

                // apply angular momentum barrier reduction factor
                // From: Bate et al. (1995) -
                // https://doi.org/10.1093/mnras/277.2.362
                if (r_circ > 0.0) {
                    // reduce accretion when circularization radius approaches
                    // distance (material would form a disk rather than accrete
                    // directly)
                    j_factor =
                        1.0 / (1.0 + (r_circ / (0.1 * effective_r_bondi)));
                }
            }

            // ======== STANDARD BONDI PROFILE COMPONENTS ========

            // density and velocity power indices based on gamma
            // from: Bondi (1952) - https://doi.org/10.1093/mnras/112.2.195
            T density_power = [=]() {
                if (isothermal) {
                    return 2.0;
                }
                else {
                    return (3.0 * gamma - 1.0) / (gamma + 1.0);
                }
            }();

            // velocity power index
            // from: Frank, King & Raine (2002) - ISBN: 9780521620536
            T velocity_power = (gamma - 1.0) / (gamma + 1.0);

            // combined mass flux power
            T mass_flux_power = density_power + velocity_power - 2.0;

            // calculate sonic radius from effective Bondi radius
            // from: Krumholz et al. (2004) - https://doi.org/10.1086/421935
            const T r_sonic = effective_r_bondi * 0.25 * (5.0 - 3.0 * gamma);

            // inner radii based on sonic radius
            const T inner_radius = 0.5 * r_sonic;
            const T core_radius  = 0.2 * r_sonic;

            auto apply_smoothing = [](const T base_value,
                                      const T transition_value,
                                      const T blend_param,
                                      const T floor   = 0.0,
                                      const T ceiling = 1.0) -> T {
                // Ensures blend_param is between 0 and 1
                const T safe_blend =
                    std::min(ceiling, std::max(floor, blend_param));

                // Linear blending between base_value and transition_value
                return base_value * (1.0 - safe_blend) +
                       transition_value * safe_blend;
            };

            // ======== BONDI-HOYLE FACTOR ========
            // standard reduction for moving sink
            // from: Ruffert (1996) -
            // https://ui.adsabs.harvard.edu/abs/1996A%26A...311..817R/abstract
            T hoyle_factor = 1.0;
            if (use_local_conditions && mach_number > 0.1) {
                // Formula approximated from numerical experiments
                hoyle_factor = std::pow(1.0 + mach_number * mach_number, -1.5);
            }

            // ======== CALCULATE ACCRETION FACTOR BY ZONE ========
            T accretion_factor;

            if (distance <= core_radius) {
                // Zone 1: core region - maximum accretion
                // Apply directional, Hoyle and angular momentum factors
                accretion_factor = accretion_efficiency * hoyle_factor *
                                   direction_factor * j_factor;
            }
            else if (distance <= inner_radius) {
                // Zone 2: transition region within supersonic flow
                // from: Federrath et al. (2010) -
                // https://doi.org/10.1088/0004-637X/713/1/269

                const T kernel_value = smoothing_kernel(
                    (distance - core_radius) / (inner_radius - core_radius)
                );

                // Calculate both extreme values
                const T core_value = accretion_efficiency * hoyle_factor *
                                     direction_factor * j_factor;
                const T bondi_value =
                    core_value *
                    std::pow(distance / effective_r_bondi, mass_flux_power);

                // Blend between them using the kernel
                accretion_factor =
                    apply_smoothing(bondi_value, core_value, kernel_value);
            }
            else if (distance <= r_sonic) {
                // Zone 3: near sonic transition
                // from: Krumholz et al. (2004) -
                // https://doi.org/10.1086/421935

                const T sonic_factor = distance / r_sonic;
                const T kernel_value = smoothing_kernel(sonic_factor);

                // Base value is the theoretical scaling
                const T base_value =
                    accretion_efficiency * hoyle_factor * direction_factor *
                    j_factor *
                    std::pow(distance / effective_r_bondi, mass_flux_power);

                // Enhance accretion near sonic point by blending with a higher
                // value You can keep your 0.2 minimum factor or make it
                // configurable
                const T min_blend_factor = 0.2;
                accretion_factor =
                    base_value *
                    apply_smoothing(1.0, 1.0, kernel_value, min_blend_factor);
            }
            else if (distance <= effective_r_bondi) {
                // Zone 4: subsonic region out to effective Bondi radius
                // from: Rosen et al. (2020) -
                // https://doi.org/10.1093/mnras/staa738

                T r_scaled = distance / effective_r_bondi;

                // Base theoretical scaling
                accretion_factor = accretion_efficiency * hoyle_factor *
                                   direction_factor * j_factor *
                                   std::pow(r_scaled, mass_flux_power);

                // Apply smooth cutoff near boundary with consistent approach
                if (r_scaled > 0.7) {
                    const T transition_blend =
                        smoothing_kernel((1.0 - r_scaled) / 0.3);
                    // Blend between base value and zero
                    accretion_factor = apply_smoothing(
                        0.0,
                        accretion_factor,
                        transition_blend
                    );
                }
            }
            else {
                // Zone 5: beyond effective Bondi radius - no accretion
                accretion_factor = 0.0;
            }

            return std::min(accretion_factor, accretion_efficiency);
        }

        // Apply accretion effect to fluid
        template <typename T, size_type Dims, typename Primitive>
        DEV std::pair<typename Primitive::counterpart_t, BodyDelta<T, Dims>>
        apply_accretion_effect(
            size_type body_idx,
            const Body<T, Dims>& body,
            const Primitive& prim,
            const Cell<Dims>& mesh_cell,
            const HydroContext& context,
            T dt
        )
        {
            using conserved_t = Primitive::counterpart_t;
            auto delta        = BodyDelta<T, Dims>{body_idx, {}, 0, 0, 0};

            // Get position vector from sink to cell center
            const auto r_vector =
                mesh_cell.cartesian_centroid() - body.position;
            const auto distance = r_vector.norm();

            // Calculate standard Bondi radius based on ambient sound speed
            const auto cs_ambient = context.ambient_sound_speed;
            const auto r_bondi    = 2.0 * body.mass / (cs_ambient * cs_ambient);

            // Skip if too far away
            if (distance > 2.5 * r_bondi) {
                return {conserved_t{}, delta};
            }

            // Get local fluid properties
            const auto local_cs     = prim.sound_speed(context.gamma);
            const auto rel_velocity = prim.velocity() - body.velocity;

            // Calculate specific angular momentum
            T specific_angular_momentum = [&]() {
                if constexpr (Dims < 3) {
                    // 2D case: use z-component of angular momentum
                    if constexpr (Dims == 1) {
                        return 0.0;
                    }
                    else {
                        return vecops::cross(r_vector, rel_velocity);
                    }
                }
                else {
                    return vecops::cross(r_vector, rel_velocity).norm();
                }
            }();

            // Calculate accretion factor based on profile
            T accretion_factor = calculate_radial_accretion_profile(
                distance,
                r_bondi,
                body.accretion_efficiency(),
                context.gamma,
                r_vector,
                rel_velocity,
                local_cs,
                specific_angular_momentum,
                context.is_isothermal
            );

            // Apply accretion if factor is positive
            if (accretion_factor > 0) {
                // Calculate stability-limited accretion rate
                const T cell_size           = mesh_cell.max_cell_width();
                const T sound_crossing_time = cell_size / local_cs;
                const T r_ratio             = distance / r_bondi;

                // Apply appropriate accretion limit
                T max_fraction;
                if (r_ratio < 0.5) {
                    max_fraction = 1.0;
                }
                else {
                    const T blend =
                        smoothing_kernel(std::max(0.0, (0.8 - r_ratio) / 0.2));
                    const T stability_limit = dt / (2.0 * sound_crossing_time);
                    max_fraction =
                        blend * 1.0 +
                        (1.0 - blend) * std::min(0.5, stability_limit);
                }

                // Apply minimum of theoretical and numerical stability limits
                const T max_accretion =
                    std::min(accretion_factor, max_fraction);

                // Calculate accreted quantities
                const T accreted_density =
                    max_accretion * prim.labframe_density();
                const auto accreted_momentum =
                    max_accretion * prim.spatial_momentum(context.gamma);
                const T accreted_energy =
                    max_accretion * prim.energy(context.gamma);

                // Create conserved state with removed material
                conserved_t result(
                    -accreted_density,
                    -accreted_momentum,
                    -accreted_energy
                );

                // Update body statistics
                const auto dV              = mesh_cell.volume();
                delta.accreted_mass_delta  = dV * accreted_density;
                delta.accretion_rate_delta = dV * accreted_density / dt;
                return {result, delta};
            }

            return {conserved_t{}, delta};
        }

    }   // namespace accretion

    namespace binary {
        // Binary system parameters
        template <size_type Dims, typename T>
        std::tuple<spatial_vector_t<T, Dims>, spatial_vector_t<T, Dims>>
        initial_positions(T semi_major, T mass_ratio)
            requires(Dims >= 2)
        {
            T a1 = semi_major / (T(1) + mass_ratio);
            T a2 = semi_major - a1;

            if constexpr (Dims == 2) {
                spatial_vector_t<T, Dims> r1{a1, 0.0};
                spatial_vector_t<T, Dims> r2{-a2, 0.0};
                return {r1, r2};
            }
            else {
                spatial_vector_t<T, Dims> r1{a1, 0.0, 0.0};
                spatial_vector_t<T, Dims> r2{-a2, 0.0, 0.0};
                return {r1, r2};
            }
        }

        template <size_type Dims, typename T>
        std::tuple<spatial_vector_t<T, Dims>, spatial_vector_t<T, Dims>>
        initial_velocities(
            T semi_major,
            T total_mass,
            T mass_ratio,
            bool circular_orbit
        )
            requires(Dims >= 2)
        {
            if (circular_orbit) {
                const T separation = semi_major;
                const T mu         = total_mass;
                const T phi_dot    = std::sqrt(mu / std::pow(separation, T(3)));
                T a1               = separation / (T(1) + mass_ratio);
                T a2               = separation - a1;

                if constexpr (Dims == 2) {
                    spatial_vector_t<T, Dims> v1{0.0, phi_dot * a2};
                    spatial_vector_t<T, Dims> v2{0.0, -phi_dot * a1};
                    return {v1, v2};
                }
                else {
                    spatial_vector_t<T, Dims> v1{0.0, phi_dot * a2, 0.0};
                    spatial_vector_t<T, Dims> v2{0.0, -phi_dot * a1, 0.0};
                    return {v1, v2};
                }
            }
            else {
                // TODO: implement eccentric orbits
                if constexpr (global::on_gpu) {
                    printf("Non-circular orbits not yet implemented\n");
                    return {
                      spatial_vector_t<T, Dims>(),
                      spatial_vector_t<T, Dims>()
                    };
                }
                else {
                    throw std::runtime_error(
                        "Non-circular orbits not yet implemented"
                    );
                }
            }
        }

        template <typename T, size_type Dims>
        array_t<Body<T, Dims>, 2> calculate_binary_motion(
            const ibsystem::ComponentBodySystem<T, Dims>& system,
            T time
        )
        {
            // get binary config
            auto config = system.template get_system_config<
                ibsystem::BinarySystemConfig<T>>();
            if (!config || !config->prescribed_motion) {
                return {Body<T, Dims>(), Body<T, Dims>()};
            }

            // Extract parameters
            const auto [body1_idx, body2_idx] = config->body_indices;
            // const T orbital_period            = config->orbital_period;
            const T semi_major = config->semi_major;
            // const T ecc                       = config->eccentricity;

            // Calculate masses
            const auto& bodies = system.bodies();
            const auto body1   = bodies[body1_idx];
            const auto body2   = bodies[body2_idx];
            const T m1         = body1.mass;
            const T m2         = body2.mass;
            const T total_mass = m1 + m2;
            const T mass_ratio = m2 / m1;

            // Calculate positions and velocities
            auto [pos1, pos2] = initial_positions<Dims>(semi_major, mass_ratio);
            auto [vel1, vel2] = initial_velocities<Dims>(
                semi_major,
                total_mass,
                mass_ratio,
                config->circular_orbit
            );

            T phi_dot = std::sqrt(total_mass / std::pow(semi_major, T(3)));
            T phi     = phi_dot * time;

            // Update positions and velocities
            return {
              body1.update_position(vecops::rotate(pos1, phi))
                  .update_velocity(vecops::rotate(vel1, phi)),
              body2.update_position(vecops::rotate(pos2, phi))
                  .update_velocity(vecops::rotate(vel2, phi))
            };
        }

        // Calculate orbital timestep
        template <typename T, size_type Dims>
        T calculate_binary_timestep(
            const ibsystem::ComponentBodySystem<T, Dims>& system,
            T cfl
        )
        {
            auto config = system.template get_system_config<
                ibsystem::BinarySystemConfig<T>>();
            if (!config) {
                return std::numeric_limits<T>::infinity();
            }

            // for binary systems, a sensible timestep is a fraction of the
            // orbital period
            return config->orbital_period * cfl / 100.0;
        }

    }   // namespace binary
}   // namespace simbi::ibsystem::body_functions
#endif
