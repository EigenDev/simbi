/**
 *=============================================================================
 *           SIMBI - Special Relativistic Magnetohydrodynamics Code
 *=============================================================================
 *
 * @file            accretion.hpp
 * @brief           accretion functions from physics and IBM (experimental)
 * @details
 *
 * @version         0.8.0
 * @date            2025-05-19
 * @author          Marcus DuPont
 * @email           marcus.dupont@princeton.edu
 *
 *==============================================================================
 * @build           Requirements & Dependencies
 *==============================================================================
 * @requires        C++20
 * @depends         CUDA >= 11.0, HDF5 >= 1.12, OpenMP >= 4.5
 * @platform        Linux, MacOS
 * @parallel        GPU (CUDA, HIP), CPU (OpenMP)
 *
 *==============================================================================
 * @documentation   Reference & Notes
 *==============================================================================
 * @usage
 * @note
 * @warning
 * @todo
 * @bug
 * @performance
 *
 *==============================================================================
 * @testing        Quality Assurance
 *==============================================================================
 * @test
 * @benchmark
 * @validation
 *
 *==============================================================================
 * @history        Version History
 *==============================================================================
 * 2025-05-19      v0.8.0      Initial implementation
 *
 *==============================================================================
 * @copyright (C) 2025 Marcus DuPont. All rights reserved.
 *==============================================================================
 */

#ifndef ACCRETION_HPP
#define ACCRETION_HPP

#include "config.hpp"
#include "geometry/mesh/cell.hpp"
#include "physics/hydro/schemes/ib/delta/body_delta.hpp"
#include "physics/hydro/schemes/ib/systems/body.hpp"
#include "physics/hydro/types/context.hpp"
#include "util/tools/helpers.hpp"

namespace simbi::ibsystem::body_functions {
    namespace accretion {

        // cubic spline smoothing kernel
        template <typename T>
        DEV T smoothing_kernel(const T r_norm)
        {
            if (r_norm >= 1.0) {
                return 0.0;
            }
            const T q = 1.0 - r_norm;
            return q * q * (1.0 + 2.0 * r_norm);
        }

        // helper function for smooth transitions
        template <typename T>
        DEV T smoothstep(T x)
        {
            // CLamp x to [0,1]
            x = std::min(std::max(x, T(0.0)), T(1.0));
            // smoothstep polynomial: 3x^2 - 2x^3
            return x * x * (3.0 - 2.0 * x);
        }

        // calculate radial accretion profile
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
                                  (rel_position.norm() * vel_mag);

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
                // value
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

        // apply accretion effect to fluid
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
            auto delta        = BodyDelta<T, Dims>{body_idx};

            // get position vector from sink to cell center
            const auto r_vector =
                mesh_cell.cartesian_centroid() - body.position;
            const auto distance = r_vector.norm();

            // calc standard Bondi radius based on ambient sound speed
            const auto cs_ambient    = context.ambient_sound_speed;
            const auto cs_ambient_sq = cs_ambient * cs_ambient;
            const auto canon_bondi   = 2.0 * body.mass / cs_ambient_sq;
            const auto r_bondi       = [=]() {
                if (!goes_to_zero(cs_ambient)) {
                    // sometimes, we are working with locally isothermal
                    // flows, so global ambient sound speed is meaningless
                    return std::min(canon_bondi, body.accretion_radius());
                }
                return body.accretion_radius();
            }();

            // skip if too far away
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

            // calc accretion factor based on profile
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

            // apply accretion if factor is positive
            if (accretion_factor > 0) {
                // calc stability-limited accretion rate
                const T cell_size           = mesh_cell.max_cell_width();
                const T sound_crossing_time = cell_size / local_cs;
                const T r_ratio             = distance / r_bondi;

                // apply appropriate accretion limit
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
                // apply minimum of theoretical and numerical stability limits
                const T max_accretion =
                    std::min(accretion_factor, max_fraction);

                // Calculate accreted quantities
                const T accreted_density =
                    max_accretion * prim.labframe_density();
                const auto accreted_momentum =
                    max_accretion * prim.spatial_momentum(context.gamma);
                const T accreted_energy =
                    max_accretion * prim.energy(context.gamma);

                // create conserved state with removed material
                conserved_t result(
                    -accreted_density,
                    -accreted_momentum,
                    -accreted_energy
                );

                // update body statistics
                const auto dV              = mesh_cell.volume();
                delta.accreted_mass_delta  = dV * accreted_density;
                delta.accretion_rate_delta = dV * accreted_density / dt;
                return {result, delta};
            }

            return {conserved_t{}, delta};
        }

        template <typename T, size_type Dims, typename Primitive>
        DEV std::pair<typename Primitive::counterpart_t, BodyDelta<T, Dims>>
        apply_simple_accretion(
            size_type body_idx,
            const Body<T, Dims>& body,
            const Primitive& prim,
            const Cell<Dims>& mesh_cell,
            const HydroContext& context,
            T dt
        )
        {
            using conserved_t = Primitive::counterpart_t;
            auto delta        = BodyDelta<T, Dims>{body_idx};

            // Get position vector from sink to cell center
            const auto r_vector =
                mesh_cell.cartesian_centroid() - body.position;
            const auto distance = r_vector.norm();

            // Simple radius check with constant accretion rate inside radius
            if (distance <= body.accretion_radius()) {
                // Accrete fixed fraction of available mass per timestep
                const T cell_size           = mesh_cell.max_cell_width();
                const T local_cs            = prim.sound_speed(context.gamma);
                const T sound_crossing_time = cell_size / local_cs;
                const T stability_limit     = dt / (sound_crossing_time);
                const T max_accretion       = std::min(
                    body.accretion_efficiency(),
                    std::min(0.5, stability_limit)
                );

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

}   // namespace simbi::ibsystem::body_functions

#endif
