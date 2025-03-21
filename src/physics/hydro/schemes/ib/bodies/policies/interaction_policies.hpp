#ifndef INTERACTION_POLICIES_HPP
#define INTERACTION_POLICIES_HPP

#include "../body_traits.hpp"                 // for Accreting
#include "build_options.hpp"                  // for DUAL, size_type, real
#include "core/types/containers/vector.hpp"   // for spatial_vector_t
#include "util/parallel/parallel_for.hpp"
#include <cmath>   // for M_PI

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
        DUAL GravitationalFluidInteractionPolicy(const Params& params = {})
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
            const auto r  = mesh_cell.centroid() - body.position();
            const auto r2 = r.dot(r) + softening * softening;

            // Gravitational force on fluid element (G = 1)
            const auto force = -body.mass() * r / (r2 * std::sqrt(r2));

            // momentum and energy change
            const auto dp = prim.labframe_density() * force * dt;

            const auto v_old = prim.velocity();
            const auto v_new = (prim.spatial_momentum() + dp) / prim.rho();
            const auto v_avg = 0.5 * (v_old + v_new);
            const auto dE    = dp.dot(v_avg);
            const auto state = conserved_t{0.0, dp, dE};

            if (two_way_coupling) {
                // Store reaction force on body
                body.force_ref() -= force * state.dens() * mesh_cell.volume();
            }
            return state;
        }

        // fluid forces are nill here
        template <typename Body>
        DUAL void
        calculate_fluid_forces(Body& body, const auto& mesh, const T dt)
        {
            // do nada
        }

        DUAL const trait_t& trait() const { return trait_; }

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

        DUAL AccretingFluidInteractionPolicy(const Params& params = {})
            : Base(params.grav_params), accr_trait_(params.accretion_params)
        {
        }

        // Access to the accreting trait
        DUAL const accr_trait_t& accreting_trait() const { return accr_trait_; }
        DUAL accr_trait_t& accreting_trait() { return accr_trait_; }

        // Forward trait methods
        DUAL T accretion_efficiency() const
        {
            return accr_trait_.accretion_efficiency();
        }
        DUAL T accretion_radius() const
        {
            return accr_trait_.accretion_radius();
        }
        DUAL T total_accreted_mass() const
        {
            return accr_trait_.total_accreted_mass();
        }

        // calc fluid forces on the body
        template <typename Body>
        DUAL void
        calculate_fluid_forces(Body& body, const auto& mesh, const T dt)
        {
            // do nothing
        }

        auto calculate_radial_accretion_profile(
            const T distance,
            const T r_bondi,
            const T accretion_efficiency,
            const T gamma
        )
        {
            // the bondi accretion rate is proportional to
            // r^{-2} for isothermal gas, and r^{-3/2} for adiabatic gas
            // of index 5/3.
            T power_index = (gamma == 1.0) ? 2.0 : 1.5;

            // let's pretend we have some ISCO
            const T inner_radius = 0.1 * r_bondi;

            // and the core
            const T core_radius = 0.2 * inner_radius;

            // cubic spline as described in Monaghan's 1992
            // "Smoothed Particle Hydrodynamics" paper
            auto smoothing_kernel = [](const T r_norm) -> T {
                if (r_norm >= 1.0) {
                    return 0.0;
                }
                const T q = 1.0 - r_norm;
                return q * q * (1.0 + 2.0 * r_norm);
            };

            T accretion_factor;

            if (distance <= core_radius) {
                // zone 1: core region - maximum accretion
                accretion_factor = accretion_efficiency;
            }
            else if (distance <= inner_radius) {
                // zone 2: transition region - cubic spline
                T kernel_value = smoothing_kernel(
                    (distance - core_radius) / (inner_radius - core_radius)
                );

                // add Bondi-like componenet that increases as kernel
                // decreases
                T bondi_factor = accretion_efficiency *
                                 std::pow(distance / r_bondi, -power_index);
                accretion_factor = accretion_efficiency * kernel_value +
                                   bondi_factor * (1.0 - kernel_value);
            }
            else if (distance <= r_bondi) {
                // zone 3: outer region - Bondi-like accretion
                constexpr T cutoff = 0.7;
                T r_scaled         = distance / r_bondi;
                accretion_factor =
                    accretion_efficiency * std::pow(r_scaled, -power_index);

                // smooth cutoff near Bondi radius
                if (r_scaled > cutoff) {
                    // Apply smooth transition to zero at Bondi radius
                    T transition =
                        smoothing_kernel((1.0 - r_scaled) / (1.0 - cutoff));
                    accretion_factor *= transition;
                }
            }
            else {
                // zone 4: outer region - no accretion
                accretion_factor = 0.0;
            }

            return std::min(accretion_factor, accretion_efficiency);
        }

        template <typename Body, typename Primitive>
        DUAL auto accrete_from_cell(
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
            // T total_accretion_fraction = 0.0;

            const auto r_vector = mesh_cell.centroid() - body.position();
            const auto distance = r_vector.norm();

            const auto cs_ambient = context.ambient_sound_speed;
            const auto r_bondi = 2.0 * body.mass() / (cs_ambient * cs_ambient);

            if (distance > 2.0 * r_bondi) {
                return conserved_t{};
            }

            T accretion_factor = calculate_radial_accretion_profile(
                distance,
                r_bondi,
                body.accretion_efficiency(),
                context.gamma
            );

            if (accretion_factor > 0) {
                const T max_accretion = std::min(accretion_factor, 0.5);
                result -= conserved_t{
                  max_accretion * prim.labframe_density(),
                  max_accretion * prim.spatial_momentum(context.gamma),
                  max_accretion * prim.energy(context.gamma)
                };
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
