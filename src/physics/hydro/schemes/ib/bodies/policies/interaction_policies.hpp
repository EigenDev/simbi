#ifndef INTERACTION_POLICIES_HPP
#define INTERACTION_POLICIES_HPP

#include "../body_traits.hpp"                 // for Accreting
#include "build_options.hpp"                  // for , size_type, real
#include "core/types/containers/vector.hpp"   // for spatial_vector_t
#include <cmath>                              // for M_PI

using namespace simbi::vecops;
namespace simbi::ib {
    //---------------------------------------------------------------------------
    // Gravitational fluid interaction policy
    // --------------------------------------------------------------------------
    template <typename T, size_type Dims>
    class GravitationalFluidInteractionPolicy
    {
        using trait_t = traits::Gravitational<T>;

      public:
        using Params = typename trait_t::Params;
        GravitationalFluidInteractionPolicy(const Params& params = {})
            : trait_(params)
        {
        }

        template <typename Body, typename Primitive>
        auto apply_forces_to_fluid(
            Body& body,
            const Primitive& prim,
            const auto& mesh_cell,
            const auto& coords,
            const auto& context,
            const T dt
        )
        {
            using conserved_t = typename Primitive::counterpart_t;
            // Reset force accumulator
            body.force_ref() = spatial_vector_t<T, Dims>();

            // load gravitational trait parameters
            const auto softening        = trait_.softening_length();
            const auto two_way_coupling = trait_.two_way_coupling();

            // Apply gravitational force to entire fluid domain
            const auto r  = mesh_cell.cartesian_centroid() - body.position();
            const auto r2 = r.dot(r) + softening * softening;

            // Gravitational force on fluid element (G = 1)
            // This is the force in cartesian coordinates
            const auto f_cart = body.mass() * r / (r2 * std::sqrt(r2));

            // this centralizes the force to |f|r_hat in spherical coordinates
            // or |f_r| rhat + |f_z| zhat in cylindrical coordinates. Unchanged
            // in cartesian coordinates
            const auto ff_hat = -centralize(f_cart, mesh_cell.geometry());

            // momentum and energy change
            const auto dp = prim.labframe_density() * ff_hat * dt;

            const auto v_old = prim.velocity();
            const auto v_new = (prim.spatial_momentum() + dp) / prim.rho();
            const auto v_avg = 0.5 * (v_old + v_new);
            const auto dE    = dp.dot(v_avg);
            const auto state = conserved_t{0.0, dp, dE};

            if (two_way_coupling) {
                // Store reaction force on body
                body.force_ref() -= ff_hat * state.dens() * mesh_cell.volume();
            }
            return state;
        }

        // fluid forces are nill here
        template <typename Body>
        void calculate_fluid_forces(Body& body, const auto& mesh, const T dt)
        {
            // do nada
        }

        const trait_t& trait() const { return trait_; }

      private:
        trait_t trait_;
    };

    //---------------------------------------------------------------------------
    // Accreting fluid interaction policy
    // --------------------------------------------------------------------------
    template <typename T, size_type Dims>
    class AccretingFluidInteractionPolicy
        : public GravitationalFluidInteractionPolicy<T, Dims>
    {
      private:
        using accr_trait_t = traits::Accreting<T>;
        using Base         = GravitationalFluidInteractionPolicy<T, Dims>;

      public:
        using accretionParams = typename accr_trait_t::Params;
        using gravParams      = typename Base::Params;
        struct Params {
            accretionParams accretion_params;
            gravParams grav_params;
        };

        AccretingFluidInteractionPolicy(const Params& params = {})
            : Base(params.grav_params), accr_trait_(params.accretion_params)
        {
        }

        // Access to the accreting trait
        const accr_trait_t& accreting_trait() const { return accr_trait_; }
        accr_trait_t& accreting_trait() { return accr_trait_; }

        // Forward trait methods
        T accretion_efficiency() const
        {
            return accr_trait_.accretion_efficiency();
        }
        T accretion_radius() const { return accr_trait_.accretion_radius(); }
        T total_accreted_mass() const
        {
            return accr_trait_.total_accreted_mass();
        }

        // calc fluid forces on the body
        template <typename Body>
        void calculate_fluid_forces(Body& body, const auto& mesh, const T dt)
        {
            // do nothing
        }

        // cubic spline smoothing kernel
        // from: Monaghan (1992) -
        // https://doi.org/10.1146/annurev.aa.30.090192.002551
        T smoothing_kernel(const T r_norm)
        {
            if (r_norm >= 1.0) {
                return 0.0;
            }
            const T q = 1.0 - r_norm;
            return q * q * (1.0 + 2.0 * r_norm);
        };

        auto calculate_radial_accretion_profile(
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
                    T cos_angle = rel_position.dot(rel_velocity) /
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

        template <typename Body, typename Primitive>
        auto accrete_from_cell(
            Body& body,
            const Primitive& prim,
            const auto& mesh_cell,
            const auto& coords,
            const auto& context,
            const T dt
        )
        {
            using conserved_t = typename Primitive::counterpart_t;
            conserved_t result{};

            // Get position vector from sink to cell center
            const auto r_vector =
                mesh_cell.cartesian_centroid() - body.position();
            const auto distance = r_vector.norm();

            // Calculate standard Bondi radius based on ambient sound speed
            const auto cs_ambient = context.ambient_sound_speed;
            const auto r_bondi = 2.0 * body.mass() / (cs_ambient * cs_ambient);

            // Improved cutoff condition (2-3 Bondi radii is standard)
            // from: Teyssier & Commerçon (2019) -
            // https://ui.adsabs.harvard.edu/abs/2019FrASS...6...51T/abstract
            if (distance > 2.5 * r_bondi) {
                return conserved_t{};
            }

            // Get local fluid properties
            const auto local_cs     = prim.sound_speed(context.gamma);
            const auto fluid_vel    = prim.velocity();
            const auto rel_velocity = fluid_vel - body.velocity();

            // include specific angular momentum for j-barrier
            // from: Hubber et al. (2013) -
            // https://doi.org/10.1093/mnras/stt128
            T specific_angular_momentum = [&]() {
                if constexpr (Dims < 3) {
                    return r_vector.cross(rel_velocity);
                }
                else {
                    return r_vector.cross(rel_velocity).norm();
                }
            }();

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

            if (accretion_factor > 0) {
                // limit accretion based on sound-crossing time for numerical
                // stability from: Federrath et al. (2010) -
                // https://doi.org/10.1088/0004-637X/713/1/269
                // Improved sound-crossing time stability limit with smooth
                // transition
                const T cell_size           = mesh_cell.max_cell_width();
                const T sound_crossing_time = cell_size / local_cs;
                const T r_ratio             = distance / r_bondi;

                // create a smooth transition function between inner
                // (aggressive) and outer regions

                // Controls width of transition zone
                const T transition_width = 0.2;
                // Where the transition is centered
                const T center_point = 0.8;

                // sigmoid-like function for smooth transition from 1.0 to
                // sound-crossing-limited
                T max_fraction;
                if (r_ratio < 0.5) {
                    // deep inside Bondi radius - aggressive accretion to create
                    // "holes"
                    max_fraction = 1.0;
                }
                else {
                    // smoothly transition from aggressive to stability-limited
                    // accretion
                    const T blend = smoothing_kernel(
                        std::max(
                            0.0,
                            (center_point - r_ratio) / transition_width
                        )
                    );
                    const T stability_limit = dt / (2.0 * sound_crossing_time);
                    max_fraction =
                        blend * 1.0 +
                        (1.0 - blend) * std::min(0.5, stability_limit);
                }

                // apply the minimum of theoretical and numerical stability
                // limits
                const T max_accretion =
                    std::min(accretion_factor, max_fraction);

                // Track accreted mass for reporting
                const T accreted_density =
                    max_accretion * prim.labframe_density();
                const auto accreted_momentum =
                    max_accretion * prim.spatial_momentum(context.gamma);
                const T accreted_energy =
                    max_accretion * prim.energy(context.gamma);

                // Create conserved state with accreted material
                result -= conserved_t{
                  accreted_density,
                  accreted_momentum,
                  accreted_energy
                };

                // Update accretion tracking statistics
                const T dV               = mesh_cell.volume();
                const auto accreted_mass = accreted_density * dV;
                accr_trait_.add_accreted_mass(accreted_density * dV);

                accr_trait_.add_accreted_momentum(
                    accreted_momentum.norm() * dV
                );

                accr_trait_.add_accreted_energy(accreted_energy * dV);

                accr_trait_.add_accreted_angular_momentum(
                    max_accretion * prim.labframe_density() *
                    specific_angular_momentum * mesh_cell.volume()
                );

                body.set_mass(body.mass() + accreted_mass);
                body.set_radius(2.0 * body.mass() / (cs_ambient * cs_ambient));
            }

            return result;
        }

        template <typename Body, typename Primitive>
        auto apply_forces_to_fluid(
            Body& body,
            const Primitive& prim,
            const auto& mesh_cell,
            const auto& coords,
            const auto& context,
            const T dt
        )
        {
            auto state = Base::apply_forces_to_fluid(
                body,
                prim,
                mesh_cell,
                coords,
                context,
                dt
            );
            return state;
        }

      private:
        accr_trait_t accr_trait_;
    };

}   // namespace simbi::ib
#endif   // INTERACTION_POLICIES_HPP
