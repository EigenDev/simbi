#ifndef INTERACTION_POLICIES_HPP
#define INTERACTION_POLICIES_HPP

#include "../body_traits.hpp"   // for Accreting
#include "build_options.hpp"    // for DUAL, size_type, real
#include <cmath>                // for M_PI

namespace simbi::ib {
    //---------------------------------------------------------------------------
    // Minimal fluid interaction policy
    // --------------------------------------------------------------------------
    template <typename T, size_type Dims>
    class MinimalFluidInteractionPolicy
    {
      public:
        struct Params {
            T drag_coefficient = T(0.0);   // No drag by default
        };

        DUAL MinimalFluidInteractionPolicy(const Params& params = {})
            : params_(params)
        {
        }

        // Calculate fluid forces on the body
        template <typename Body>
        DUAL void
        calculate_fluid_forces(Body& body, const auto& mesh, const T dt)
        {
            // Minimal implementation - could be empty or just add minimal drag
            if (params_.drag_coefficient > 0) {
                add_minimal_drag(body);
            }
        }

        // Apply forces from body to fluid
        template <typename Body, typename ConsArray, typename PrimArray>
        DUAL void apply_to_fluid(
            Body& body,
            ConsArray& cons_states,
            const PrimArray& prim_states,
            const T dt
        )
        {
            // Do nothing - body doesn't affect fluid
        }

        DUAL const Params& params() const { return params_; }
        DUAL Params& params() { return params_; }

      private:
        Params params_;

        template <typename Body>
        DUAL void add_minimal_drag(Body& body)
        {
            // Simple drag proportional to velocity squared
            const auto v      = body.velocity();
            const auto v_norm = v.norm();

            if (v_norm > 0) {
                // Simple drag force: F = -0.5 * rho * Cd * A * v^2
                // We'll use a simplified version based just on velocity
                const auto drag_force = -params_.drag_coefficient * v_norm * v;
                body.force_ += drag_force;
            }
        }
    };

    //---------------------------------------------------------------------------
    // Standard fluid interaction policy
    // --------------------------------------------------------------------------
    template <typename T, size_type Dims>
    class StandardFluidInteractionPolicy
    {
      public:
        struct Params {
            T drag_coefficient    = T(0.47);   // Default sphere drag
            bool two_way_coupling = true;   // Body affects fluid and vice versa
        };

        DUAL StandardFluidInteractionPolicy(const Params& params = {})
            : params_(params)
        {
        }

        // Calculate fluid forces on the body
        template <typename Body>
        DUAL void
        calculate_fluid_forces(Body& body, const auto& mesh, const T dt)
        {
            // TODO: Interpolate fluid velocity at body position
            // body.interpolate_fluid_velocity();

            // Calculate drag force
            const auto v_rel  = body.velocity() - body.fluid_velocity();
            const auto v_norm = v_rel.norm();

            if (v_norm > 0) {
                // Surface area - approximation for a sphere
                const auto area = T(M_PI) * body.radius() * body.radius();

                // Density - use average from cut cells
                T avg_density  = 0;
                T total_volume = 0;

                for (const auto& idx : body.cut_cell_indices()) {
                    const auto& cell     = body.cell_info()[idx];
                    const auto mesh_cell = mesh.get_cell_from_global(idx);

                    // Only consider cells that are partially fluid
                    if (cell.is_cut && cell.volume_fraction < 1.0) {
                        const T cell_volume =
                            mesh_cell.volume() * (1.0 - cell.volume_fraction);
                        // Note: we would need to access density from
                        // conservation states here For simplicity, we'll use a
                        // constant density of 1.0
                        const T cell_density = 1.0;

                        avg_density += cell_density * cell_volume;
                        total_volume += cell_volume;
                    }
                }

                // Avoid division by zero
                if (total_volume > 0) {
                    avg_density /= total_volume;
                }
                else {
                    avg_density = 1.0;   // Default density
                }

                // Full drag equation: F = -0.5 * rho * Cd * A * v^2 * (v/|v|)
                const auto drag_force = -0.5 * avg_density *
                                        params_.drag_coefficient * area *
                                        v_norm * v_rel;

                body.force_ += drag_force;
            }
        }

        // Apply forces from body to fluid
        template <typename Body, typename ConsArray, typename PrimArray>
        DUAL void apply_to_fluid(
            Body& body,
            ConsArray& cons_states,
            const PrimArray& prim_states,
            const T dt
        )
        {
            if (!params_.two_way_coupling) {
                return;
            }

            using conserved_t = typename ConsArray::value_type;

            // Apply reaction forces to fluid in cut cells
            for (const auto& idx : body.cut_cell_indices()) {
                const auto& cell     = body.cell_info()[idx];
                const auto mesh_cell = body.mesh().get_cell_from_global(idx);
                const auto dV        = mesh_cell.volume();

                // Skip completely solid cells
                if (cell.volume_fraction >= 1.0) {
                    continue;
                }

                // Force on fluid = -force on body
                const auto force_density =
                    -body.force_ * (1.0 - cell.volume_fraction) / dV;

                // Use force to update momentum and energy
                auto& state = cons_states[idx];

                // Momentum change
                const auto dp = force_density * dt;

                // Energy change - work done by force
                const auto work = dp.dot(body.velocity());

                // Update conserved state
                state += conserved_t{0.0, dp, work};
            }
        }

        DUAL const Params& params() const { return params_; }
        DUAL Params& params() { return params_; }

      private:
        Params params_;
    };

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

        template <typename Body, typename ConsArray, typename PrimArray>
        void apply_to_fluid(
            Body& body,
            ConsArray& cons_states,
            const PrimArray& prim_states,
            const auto dt
        )
        {
            using conserved_t = typename ConsArray::value_type;

            // Reset force accumulator
            body.force_ref() = spatial_vector_t<T, Dims>();

            // load gravitational trait parameters
            const auto softening        = trait_.softening_length();
            const auto mesh             = body.mesh();
            const auto two_way_coupling = trait_.two_way_coupling();

            // Apply gravitational force to entire fluid domain
            cons_states.transform_with_indices(
                [=,
                 body_ptr =
                     &body] DEV(auto& state, size_type idx, const auto& prim) {
                    const auto mesh_cell = mesh.get_cell_from_global(idx);
                    const auto r  = mesh_cell.centroid() - body_ptr->position();
                    const auto r2 = r.dot(r) + softening * softening;

                    // Gravitational force on fluid element
                    const auto force =
                        -body_ptr->mass() * r / (r2 * std::sqrt(r2));

                    // momentum and energy change
                    const auto dp = prim->rho() * force * dt;

                    const auto v_old = prim->velocity();
                    auto v_new       = (state.momentum() + dp) / prim->rho();
                    const auto v_avg = 0.5 * (v_old + v_new);
                    const auto dE    = dp.dot(v_avg);
                    state += conserved_t{0.0, dp, dE};

                    if (two_way_coupling) {
                        // Store reaction force on body
                        body_ptr->force_ref() -=
                            force * state.dens() * mesh_cell.volume();
                    }

                    return state;
                },
                body.get_default_policy(),
                prim_states
            );
        }

        // fluid forces are nill here
        template <typename Body>
        DUAL void
        calculate_fluid_forces(Body& body, const auto& mesh, const T dt)
        {
            // do nada
        }

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
        DUAL T accretion_radius_factor() const
        {
            return accr_trait_.accretion_radius_factor();
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

        // Accrete material from fluid
        template <typename Body, typename ConsArray, typename PrimArray>
        DUAL void accrete(
            Body& body,
            ConsArray& cons_states,
            const PrimArray& prim_states
        )
        {
            const auto accretion_radius =
                body.radius() * accr_trait_.accretion_radius_factor();
            T total_accreted_mass        = 0;
            auto total_accreted_momentum = spatial_vector_t<T, Dims>();
            T total_accreted_energy      = 0;

            for (const auto& idx : body.cut_cell_indices()) {
                const auto& cell = body.cell_info()[idx];
                // const auto cs           = prim_states[idx]->sound_speed();
                // const auto bondi_radius = 2.0 * body.mass() / (cs * cs);

                // Check if cell is within accretion radius
                if (std::abs(cell.distance) <= accretion_radius) {
                    const auto mesh_cell =
                        body.mesh().get_cell_from_global(idx);
                    auto& state = cons_states[idx];

                    // Skip cells with negligible mass
                    if (state.dens() <= global::epsilon) {
                        continue;
                    }

                    // Calculate maximum allowed accretion fraction
                    const T max_accretion_fraction =
                        accr_trait_.accretion_efficiency() *
                        cell.volume_fraction;

                    // Calculate mass to accrete
                    const T cell_volume   = mesh_cell.volume();
                    const T cell_mass     = state.dens() * cell_volume;
                    const T accreted_mass = max_accretion_fraction * cell_mass;

                    if (accreted_mass > 0) {
                        // Calculate momentum and energy accreted
                        const auto cell_momentum = state.momentum();
                        const auto accreted_momentum =
                            cell_momentum * (accreted_mass / cell_mass);
                        const auto cell_energy = state.nrg();
                        const auto accreted_energy =
                            cell_energy * (accreted_mass / cell_mass);

                        // Remove accreted material from fluid
                        state.dens() *= (1.0 - max_accretion_fraction);
                        state.momentum() *= (1.0 - max_accretion_fraction);
                        state.nrg() *= (1.0 - max_accretion_fraction);

                        // Accumulate totals
                        total_accreted_mass += accreted_mass;
                        total_accreted_momentum += accreted_momentum;
                        total_accreted_energy += accreted_energy;
                    }
                }
            }

            // Update body properties based on accreted material
            if (total_accreted_mass > 0) {
                // Update mass and momentum
                const auto old_momentum = body.mass() * body.velocity();
                body.mass_ += total_accreted_mass;
                body.velocity_ =
                    (old_momentum + total_accreted_momentum) / body.mass();

                // Update accretion stats in trait
                accr_trait_.add_accreted_mass(total_accreted_mass);
                accr_trait_.add_accreted_momentum(total_accreted_momentum.norm()
                );
                accr_trait_.add_accreted_energy(total_accreted_energy);
            }
        }

        // Apply local accretion interaction onto fluid
        template <typename Body, typename ConsArray, typename PrimArray>
        DUAL void apply_to_fluid(
            Body& body,
            ConsArray& cons_states,
            const PrimArray& prim_states,
            const T dt
        )
        {
            Base::apply_to_fluid(body, cons_states, prim_states, dt);
            // Perform accretion if conditions are met
            if (accr_trait_.accretion_efficiency() > 0) {
                accrete(body, cons_states, prim_states);
            }
        }

      private:
        accr_trait_t accr_trait_;
    };

}   // namespace simbi::ib
#endif   // INTERACTION_POLICIES_HPP
