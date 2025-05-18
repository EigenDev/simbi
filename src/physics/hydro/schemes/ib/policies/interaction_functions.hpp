/**
 *=============================================================================
 *           SIMBI - Special Relativistic Magnetohydrodynamics Code
 *=============================================================================
 *
 * @file            interaction_functions.hpp
 * @brief           Interaction functions for the IB scheme
 * @details
 *
 * @version         0.8.0
 * @date            2025-05-11
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
 * 2025-05-11      v0.8.0      Initial implementation
 *
 *==============================================================================
 * @copyright (C) 2025 Marcus DuPont. All rights reserved.
 *==============================================================================
 */
#ifndef INTERACTION_FUNCTIONS_HPP
#define INTERACTION_FUNCTIONS_HPP

#include "build_options.hpp"   // for real, size_type, DEV
#include "core/types/containers/array.hpp"
#include "core/types/containers/vector.hpp"   // for spatial_vector_t
#include "geometry/mesh/cell.hpp"             // for Cell
#include "physics/hydro/schemes/ib/delta/body_delta.hpp"   // for BodyDelta
#include "physics/hydro/schemes/ib/systems/body.hpp"       // for Body
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
            const auto softening_sq     = softening_length * softening_length;
            const auto r      = mesh_cell.cartesian_centroid() - body.position;
            const auto r2     = vecops::dot(r, r) + softening_sq;
            const auto r3_inv = 1.0 / (r2 * std::sqrt(r2));
            // Gravitational force on fluid element (G = 1)
            const auto f_cart = body.mass * r * r3_inv;

            // Centralize force based on geometry
            const auto ff_hat = -centralize(f_cart, mesh_cell.geometry());

            const auto density = prim.labframe_density();
            // Calculate momentum and energy change
            const auto dp = density * ff_hat * dt;

            const auto& v_old = prim.velocity();
            const auto invd   = 1.0 / density;
            const auto v_new =
                (prim.spatial_momentum(context.gamma) + dp) * invd;
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

            // get position vector from sink to cell center
            const auto r_vector =
                mesh_cell.cartesian_centroid() - body.position;
            const auto distance = std::sqrt(
                vecops::dot(r_vector, r_vector) +
                body.softening_length() * body.softening_length()
            );

            // calc standard Bondi radius based on ambient sound speed
            const auto cs_ambient    = context.ambient_sound_speed;
            const auto cs_ambient_sq = cs_ambient * cs_ambient;
            const auto canon_bondi   = 2.0 * body.mass / cs_ambient_sq;
            const auto r_bondi       = [=]() {
                if (cs_ambient > 0.0) {
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
                const T cell_size           = mesh_cell.min_cell_width();
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
        apply_ibm_interior_treatment(
            size_type body_idx,
            const Body<T, Dims>& body,
            const Primitive& prim,
            const Cell<Dims>& mesh_cell,
            const HydroContext& context,
            T dt
        )
        {
            using conserved_t = typename Primitive::counterpart_t;
            auto delta        = BodyDelta<T, Dims>{body_idx, {}, 0, 0, 0};

            // cell and fluid props
            const auto cell_center     = mesh_cell.cartesian_centroid();
            const auto cell_volume     = mesh_cell.volume();
            const auto cell_width      = mesh_cell.min_cell_width();
            const auto density         = prim.labframe_density();
            const auto fluid_velocity  = prim.velocity();
            const auto specific_energy = prim.specific_energy(context.gamma);
            const auto pressure        = prim.press();
            const auto local_cs        = prim.sound_speed(context.gamma);

            // calculate distance to black hole
            const auto r_vector = cell_center - body.position;
            const auto distance = std::sqrt(
                vecops::dot(r_vector, r_vector) +
                body.softening_length() * body.softening_length()
            );
            const auto accretion_radius = body.accretion_radius();
            const auto accr_eff         = body.accretion_efficiency();
            const auto r_scaled         = distance / accretion_radius;

            // =========== PHYSICAL PARAMETERS ===========

            // calculate local dynamical time
            const T local_dynamical_time =
                std::sqrt(std::pow(distance, 3) / body.mass);
            const T sound_crossing_time = cell_width / local_cs;
            // we define the safety sink rate to not exceed the local sound
            // speed and 0.5 is an aditional safety margin
            constexpr T safety_factor = 0.5;
            const T max_allowed_sink_rate =
                safety_factor * density / sound_crossing_time;

            // calculate free-fall velocity
            const T v_freefall = std::sqrt(2.0 * body.mass / distance);

            // calculate specific angular momentum and circularization radius
            T specific_angular_momentum;
            if constexpr (Dims < 3) {
                if constexpr (Dims == 1) {
                    specific_angular_momentum = 0.0;
                }
                else {
                    specific_angular_momentum = std::abs(
                        vecops::cross(r_vector, fluid_velocity - body.velocity)
                    );
                }
            }
            else {
                specific_angular_momentum =
                    vecops::cross(r_vector, fluid_velocity - body.velocity)
                        .norm();
            }
            const T r_circ = specific_angular_momentum *
                             specific_angular_momentum / body.mass;

            // =========== REGIME IDENTIFICATION ===========

            // ID physical regime for this cell
            enum class AccretionRegime {
                DEEP_INTERIOR,       // we are cery close to black hole
                TRANSITION_REGION,   // intermediate region
                DISK_LIKE   // material with significant angular momentum
            };

            AccretionRegime regime;

            // check for disk-like structure first
            constexpr T r_circ_threshold   = 0.2;
            constexpr T r_scale_threshold  = 0.3;
            const T angular_momentum_ratio = r_circ / distance;
            const bool high_angular_momentum =
                angular_momentum_ratio > r_circ_threshold;

            if (high_angular_momentum) {
                regime = AccretionRegime::DISK_LIKE;
            }
            else if (r_scaled < r_scale_threshold) {
                regime = AccretionRegime::DEEP_INTERIOR;
            }
            else {
                regime = AccretionRegime::TRANSITION_REGION;
            }

            // =========== SINK TIMESCALE CALCULATION ===========

            // base timescale dependent on regime
            T sink_timescale;

            switch (regime) {
                case AccretionRegime::DEEP_INTERIOR: {
                    // rapid removal in deep interior
                    constexpr T deep_interior_factor = 0.1;
                    sink_timescale =
                        deep_interior_factor * local_dynamical_time;
                } break;

                case AccretionRegime::TRANSITION_REGION:
                    // gradual transition to boundary
                    {
                        constexpr T blend_factor = 0.9;

                        // smooth transition from deep interior to boundary
                        const T transition_factor =
                            (r_scaled - r_scale_threshold) /
                            (1 - r_scale_threshold);
                        const T blend = smoothstep(transition_factor);
                        sink_timescale =
                            ((1 - blend_factor) + blend_factor * blend) *
                            local_dynamical_time;
                    }
                    break;

                case AccretionRegime::DISK_LIKE:
                    // use viscous-like timescale for disk material
                    {
                        // estimate alpha-disk viscous timescale
                        // TODO: make context dependent?
                        // Shakura-Sunyaev viscosity parameter
                        const T alpha = 0.01;
                        const T h_r =
                            local_cs / std::sqrt(body.mass / distance);
                        const T viscous_timescale =
                            local_dynamical_time / (alpha * h_r * h_r);

                        // scale with circularization radius - farther out means
                        // slower accretion
                        const T circ_factor = std::min(5.0, r_circ / distance);

                        sink_timescale = viscous_timescale * circ_factor;
                    }
                    break;
            }

            // =========== VELOCITY PROFILE ===========

            // calculate desired velocity field inside boundary
            spatial_vector_t<T, Dims> desired_velocity;

            switch (regime) {
                case AccretionRegime::DEEP_INTERIOR:
                    // pure radial infall for deep interior
                    desired_velocity = -v_freefall * (r_vector / distance);
                    break;

                case AccretionRegime::TRANSITION_REGION:
                    // transitional velocity field
                    {
                        // base velocity - radial component
                        const auto radial_dir = r_vector / distance;
                        const auto radial_vel = -v_freefall * radial_dir;

                        // keep some of the original velocity's non-radial
                        // component
                        const auto current_radial =
                            vecops::dot(fluid_velocity, radial_dir) *
                            radial_dir;
                        const auto current_nonradial =
                            fluid_velocity - current_radial;

                        // blend between pure infall and current tangential
                        // component
                        const T blend_factor = (r_scaled - r_scale_threshold) /
                                               (1 - r_scale_threshold);
                        const T tangential_retention = smoothstep(blend_factor);

                        desired_velocity = radial_vel + tangential_retention *
                                                            current_nonradial;
                    }
                    break;

                case AccretionRegime::DISK_LIKE:
                    // maintain rotational support but induce slow inspiral
                    {
                        // extract radial and tangential components
                        const auto radial_dir = r_vector / distance;
                        const auto current_radial =
                            vecops::dot(fluid_velocity, radial_dir) *
                            radial_dir;
                        const auto current_nonradial =
                            fluid_velocity - current_radial;

                        // modified radial velocity - slower infall
                        constexpr T radial_infall_factor = 0.1;
                        const auto modified_radial =
                            -radial_infall_factor * v_freefall * radial_dir;

                        // keep non-radial component for disk structure
                        desired_velocity =
                            modified_radial +
                            (1 - radial_infall_factor) * current_nonradial;
                    }
                    break;
            }

            // =========== PRESSURE TREATMENT ===========

            // calculate pressure profile consistent with velocity field
            T pressure_forcing = 0.0;

            // only apply pressure forcing in certain regimes
            if (regime != AccretionRegime::DISK_LIKE) {
                // calculate pressure gradient needed to support desired
                // velocity field
                const T gravity = body.mass / (distance * distance);

                T velocity_gradient;
                if (regime == AccretionRegime::DEEP_INTERIOR) {
                    // for v ~ r^(-1/2): dv/dr = -0.5v/r
                    velocity_gradient = -0.5 * v_freefall / distance;
                }
                else {
                    // approximate gradient for transition region
                    velocity_gradient = -0.3 * v_freefall / distance;
                }

                // pressure gradient to balance forces: dp/dr = \rho v(dv/dr) -
                // \rho g
                const T desired_pressure_gradient =
                    density * (v_freefall * velocity_gradient - gravity);

                // current pressure gradient (approximate)
                // simple approximation
                const T current_pressure_gradient = -pressure / distance;

                // pressure forcing to correct gradient
                pressure_forcing =
                    (desired_pressure_gradient - current_pressure_gradient) *
                    distance / 3.0;
            }

            // =========== SINK TERM CALCULATION ===========

            // calculate sink rate from timescale
            const T sink_rate = density / sink_timescale;

            // apply mathematically consistent sink for mass based on velocity
            // divergence
            T divergence_sink = 0.0;

            if (regime == AccretionRegime::DEEP_INTERIOR) {
                // for v ~ r^(-1/2), div(v) = -1.5v/r
                divergence_sink = density * 1.5 * v_freefall / distance;
            }
            else if (regime == AccretionRegime::TRANSITION_REGION) {
                // approximate divergence for transition region
                divergence_sink = density * 0.8 * v_freefall / distance;
            }

            // final sink rate is the maximum of timescale-based and
            // divergence-based sinks
            const T final_sink_rate = std::max(sink_rate, divergence_sink);
            const T mass_removed =
                std::min(final_sink_rate, max_allowed_sink_rate) * dt *
                accr_eff;

            // =========== ENERGY TREATMENT ===========

            // calculate energy removal consistent with physical processes
            T energy_removed;

            // different energy treatment based on regime
            if (regime == AccretionRegime::DISK_LIKE) {
                // for disk-like flow, account for energy release from viscous
                // dissipation
                // we use Virial theorem here [TODO: can we do better?]
                const T orbital_energy       = -0.5 * body.mass / distance;
                const T accretion_efficiency = body.accretion_efficiency();

                // energy removed minus energy released by accretion
                energy_removed =
                    mass_removed *
                    (specific_energy - orbital_energy * accretion_efficiency);
            }
            else {
                // for direct infall, standard energy removal
                energy_removed = mass_removed * specific_energy;

                // account for gravitational energy release
                // TODO: implement
                // if (context.include_energy_release) {
                //     const T gravitational_potential = -body.mass / distance;
                //     const T kinetic_energy =
                //         0.5 * vecops::dot(fluid_velocity, fluid_velocity);
                //     const T binding_energy =
                //         kinetic_energy + gravitational_potential;

                //     // Typical accretion efficiency
                //     const T accretion_efficiency = 0.1;

                //     // Reduce energy removal due to energy release
                //     energy_removed =
                //         mass_removed * (specific_energy -
                //                         binding_energy *
                //                         accretion_efficiency);
                // }
            }

            // =========== MOMENTUM AND VELOCITY FORCING ===========

            // calculate momentum removal
            const auto momentum_removed = mass_removed * fluid_velocity;

            // velocity forcing to maintain desired profile
            const auto velocity_difference = desired_velocity - fluid_velocity;
            const auto velocity_forcing    = density * velocity_difference / dt;

            // =========== FINAL UPDATE ===========

            // combine all contributions
            conserved_t result(
                -mass_removed,                               // mass update
                -momentum_removed + velocity_forcing * dt,   // momentum update
                -energy_removed +
                    vecops::dot(velocity_forcing, fluid_velocity) * dt +
                    pressure_forcing * dt   // energy update
            );

            // update accretion statistics
            delta.accreted_mass_delta  = mass_removed * cell_volume;
            delta.accretion_rate_delta = mass_removed * cell_volume / dt;

            return {result, delta};
        }

        template <typename T, size_type Dims, typename Primitive>
        DEV std::pair<typename Primitive::counterpart_t, BodyDelta<T, Dims>>
        apply_ibm_accretion(
            size_type body_idx,
            const Body<T, Dims>& body,
            const Primitive& prim,
            const Cell<Dims>& mesh_cell,
            const HydroContext& context,
            T dt
        )
        {

            using conserved_t = typename Primitive::counterpart_t;
            auto delta        = BodyDelta<T, Dims>{body_idx, {}, 0, 0, 0};

            // cell properties
            const auto center     = mesh_cell.cartesian_centroid();
            const auto volume     = mesh_cell.volume();
            const auto cell_width = mesh_cell.min_cell_width();

            // distance to sink particle surface
            const auto r_vector = center - body.position;
            const auto softening_sq =
                body.softening_length() * body.softening_length();
            const auto distance =
                std::sqrt(vecops::dot(r_vector, r_vector) + softening_sq);
            const auto accretion_radius = body.accretion_radius();

            // skip if too far away
            if (distance > 1.5 * accretion_radius) {
                return {conserved_t{}, delta};
            }

            // only apply to interior cells
            if (distance < accretion_radius - 0.5 * cell_width) {
                return apply_ibm_interior_treatment(
                    body_idx,
                    body,
                    prim,
                    mesh_cell,
                    context,
                    dt
                );
            }
            // determin cell type relative to boundary
            bool is_boundary_cell =
                std::abs(distance - accretion_radius) < cell_width;
            // const bool is_interior_cell =
            //     distance < accretion_radius - 0.5 * cell_width;

            if (!is_boundary_cell) {
                // TODO: handle interior cells with direct sink treament
                // although his is not part of IBM proper
                return {conserved_t{}, delta};
            }

            // get local fluid properties
            const auto density      = prim.labframe_density();
            const auto vfluid       = prim.velocity();
            const auto rel_velocity = prim.velocity() - body.velocity;

            // desired boundary velocity calulcation (inflow for accretion)
            const auto boundary_normal = -r_vector / distance;

            // calculate specific angular momentum
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

            //  circularization radius
            const T r_circ = specific_angular_momentum *
                             specific_angular_momentum / body.mass;

            //  timescales
            const T dynamical_time =
                std::sqrt(std::pow(distance, 3) / body.mass);
            const T alpha = 0.01;   // Effective viscosity parameter

            // I will define some "core radius"
            // constexpr T core_radius = 0.1;   // Core radius for accretion

            //  accretion timescale with angular momentum dependence
            const auto final_accretion_time = [=]() {
                if (r_circ < 0.1 * distance) {
                    // direct accretion case
                    return dynamical_time;
                }
                else {
                    // disk formation case
                    const T viscous_time = dynamical_time / alpha;
                    // smooth transition based on circularization radius
                    const T ratio = r_circ / (0.1 * distance);
                    const T blend = ratio / (1.0 + ratio);
                    return dynamical_time * (1.0 - blend) +
                           viscous_time * blend;
                }
            }();

            // set the boundary velocity based on free-fall speed
            const T v_ff          = std::sqrt(2.0 * body.mass / distance);
            const auto u_boundary = v_ff * boundary_normal;

            // now we invoke Huang & Sung (2007)

            // init mass sink term
            T q = 0.0;

            // check each face to see if it's crossed by the boundary
            for (size_type f = 0; f < 2 * Dims; ++f) {
                const auto normal     = mesh_cell.normal_vec(f);
                const auto face_coord = mesh_cell.normal(f);
                const auto face_vec   = normal * face_coord;

                // calculate distance from face center to sink
                const auto face_r_vector = face_vec - body.position;
                const auto face_distance = std::sqrt(
                    vecops::dot(face_r_vector, face_r_vector) + softening_sq
                );

                // check if this face is crossed by the boundary
                // a face is crossed if the accretion radius lies between this
                // face and the opposite face
                bool is_crossed_face = false;

                // for each direction (x, y, z), check the opposite face
                size_type opposite_face = f % 2 == 0 ? f + 1 : f - 1;
                const auto opp_norm     = mesh_cell.normal_vec(opposite_face);
                const auto opp_coord    = mesh_cell.normal(opposite_face);
                const auto opp_vec      = opp_norm * opp_coord;

                const auto opp_face_r_vector = opp_vec - body.position;
                const auto opp_face_distance = std::sqrt(
                    vecops::dot(opp_face_r_vector, opp_face_r_vector) +
                    softening_sq
                );

                // the boundary crosses between these faces if one is inside and
                // one is outside
                is_crossed_face = (face_distance < accretion_radius &&
                                   opp_face_distance > accretion_radius) ||
                                  (face_distance > accretion_radius &&
                                   opp_face_distance < accretion_radius);

                if (is_crossed_face) {
                    // calc \beta (Eq. 18) for this face
                    // beta = 1/2 - d/\delta x where d is distance from face to
                    // boundary
                    T beta_i =
                        0.5 - (accretion_radius - face_distance) / cell_width;

                    // get fluid velocity at face center (interpolate from cell
                    // center)
                    // simplified - TODO: ideally interpolate
                    // from neighboring cells
                    auto u_face = vfluid;

                    // calc face-centered velocity for virtual cell
                    auto u_c = [&]() -> spatial_vector_t<T, Dims> {
                        if (beta_i > 0) {
                            // face is outside boundary
                            return beta_i / (1.0 + 2.0 * beta_i) * u_face;
                        }
                        else {
                            // face is inside boundary
                            return ((3.0 / 2.0) + (beta_i / 2.0)) * u_boundary -
                                   ((1.0 / 2.0) + (beta_i / 2.0)) * u_face;
                        }
                    }();

                    // calc mass flux term for this face (Eq. 24)
                    // \omega * u \cdot n - \beta * u' \cdot n
                    // Where \omega is 1 for forcing points and 0 otherwise
                    T face_flux = 0.0;

                    // If face is inside boundary, apply forcing (\omega = 1)
                    if (face_distance < accretion_radius) {
                        face_flux += vecops::dot(u_face, normal);
                    }

                    // apply virtual cell correction
                    face_flux -= beta_i * vecops::dot(u_c, normal);

                    // weight by face area and add to sink term
                    q += face_flux * mesh_cell.area(f);
                }
            }

            // normalize by cell volume
            q /= volume;

            // we are studying accretion, so we only want to consider
            // negative fluxes
            q = std::max(q, 0.0);

            // scale by accretion timescale for angular momentum dependence
            // note to self: this modifies the IBM sink term to account for
            // angular momentum
            if (q < 0) {   // only for inflow (negative flux)
                q *= dynamical_time / final_accretion_time;
            }

            // apply sink term to create conserved update
            // note that q * dt is the fractional change in density
            // negative b/c q is positive for outflow
            const T density_removed = density * q * dt;

            // create conserved state update
            // the negative sign because we are ADDING
            // the source term to the conserved state
            conserved_t result(
                -density_removed,
                -density_removed * vfluid,
                -density_removed * prim.energy(context.gamma) / density
            );

            if (density_removed > 0.0) {
                // update body statistics
                delta.accreted_mass_delta  = density_removed * volume;
                delta.accretion_rate_delta = density_removed * volume / dt;
            }
            return {result, delta};
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
            T dt
        )
        {
            // get binary config
            auto config = system.template get_system_config<
                ibsystem::BinarySystemConfig<T>>();
            if (!config || !config->prescribed_motion) {
                const auto& bodies = system.bodies();
                return {bodies[0], bodies[1]};
            }

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
            // const T mass_ratio = m2 / m1;

            const auto& pos1 = body1.position;
            const auto& pos2 = body2.position;
            const auto& vel1 = body1.velocity;
            const auto& vel2 = body2.velocity;

            T phi_dot = std::sqrt(total_mass / std::pow(semi_major, T(3)));
            T phi     = phi_dot * dt;

            // Update positions and velocities
            return {
              body1.update_position(vecops::rotate(pos1, phi))
                  .update_velocity(vecops::rotate(vel1, phi)),
              body2.update_position(vecops::rotate(pos2, phi))
                  .update_velocity(vecops::rotate(vel2, phi))
            };
        }

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
            // orbital period. TODO: implement eccentric orbits
            return config->orbital_period * cfl / 100.0;
        }

    }   // namespace binary
}   // namespace simbi::ibsystem::body_functions
#endif
