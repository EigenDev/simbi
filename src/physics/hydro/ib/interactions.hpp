#ifndef SIMBI_IB_INTERACTIONS_HPP
#define SIMBI_IB_INTERACTIONS_HPP

#include "body.hpp"
#include "body_delta.hpp"
#include "collector.hpp"
#include "component_body_system.hpp"
#include "config.hpp"
#include "core/base/concepts.hpp"
#include "core/utility/enums.hpp"
#include "data/containers/ctx.hpp"
#include "data/containers/vector.hpp"
#include "physics/hydro/physics.hpp"
#include "system/mesh/solver.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <utility>

namespace simbi::ibsystem {
    // coordinate-natieve gravitational interaction
    template <
        is_hydro_primitive_c prim_t,
        std::uint64_t Dims = prim_t::dimensions>
    DEV auto apply_gravitational_interaction(
        const Body<real, Dims>& body,
        const prim_t& prim,
        const physics_context_t<Dims>& ctx
    )
    {
        using conserved_t = typename prim_t::counterpart_t;

        // clean cartesian physics - no coordinate system complexity
        const auto r_vec        = ctx.cell_pos - body.position;
        const auto r_mag        = r_vec.norm();
        const auto softening_sq = body.ctx.epsilon * body.ctx.epsilon;
        const auto r_eff        = std::sqrt(r_mag * r_mag + softening_sq);

        // gravitational acceleration (G = 1)
        const auto g_accel = -body.mass * r_vec / (r_eff * r_eff * r_eff);

        // fluid changes
        const auto density         = labframe_density(prim);
        const auto momentum_change = density * g_accel;
        const auto v_old           = prim.vel;
        const auto energy_change   = vecops::dot(prim.vel, momentum_change);

        // body delta for diagnostics
        BodyDelta<real, Dims> delta{body.index};
        if (body.two_way_coupling) {
            // Newton's third law
            delta.force_delta = -density * g_accel * ctx.cell_volume;
        }

        return std::make_pair(
            conserved_t{0.0, momentum_change, energy_change},
            delta
        );
    }

    // coordinate-native accretion interaction
    template <
        is_hydro_primitive_c prim_t,
        std::uint64_t Dims = prim_t::dimensions>
    DEV auto apply_accretion_interaction(
        const Body<real, Dims>& body,
        const prim_t& prim,
        const physics_context_t<Dims>& ctx
    )
    {
        using conserved_t = typename prim_t::counterpart_t;

        // get position vector from sink to cell center
        const auto r_vec = ctx.cell_pos - body.position;
        const auto r_mag = r_vec.norm();

        // skip if too far away
        if (r_mag > 2.5 * body.accretion_radius()) {
            return std::make_pair(
                conserved_t{},
                BodyDelta<real, Dims>{body.index}
            );
        }

        // Accrete fixed fraction of available mass per timestep
        const auto cell_size           = ctx.max_cell_width;
        const auto local_cs            = sound_speed(prim, ctx.gamma);
        const auto sound_crossing_time = cell_size / local_cs;
        const auto stability_limit     = ctx.dt / (sound_crossing_time);
        const auto eps                 = std::min(
            body.accretion_efficiency(),
            std::min(0.5, stability_limit)
        );

        // Calculate accreted quantities
        const auto accreted_density  = eps * labframe_density(prim, ctx.gamma);
        const auto accreted_momentum = eps * spatial_momentum(prim, ctx.gamma);
        const auto accreted_energy   = eps * energy_density(prim, ctx.gamma);

        // Create conserved state with removed material
        conserved_t result(
            -accreted_density,
            -accreted_momentum,
            -accreted_energy
        );

        auto delta = BodyDelta<real, Dims>{body.index};
        // update body statistics
        const auto& dV             = ctx.cell_volume;
        delta.accreted_mass_delta  = dV * accreted_density;
        delta.accretion_rate_delta = dV * accreted_density / ctx.dt;

        return std::make_pair(result, delta);
    }

    template <
        is_hydro_primitive_c prim_t,
        std::uint64_t Dims = prim_t::dimensions>
    DEV std::pair<typename prim_t::counterpart_t, BodyDelta<real, Dims>>
    apply_rigid_interaction(
        const Body<real, Dims>& body,
        const prim_t& prim,
        const physics_context_t<Dims>& ctx
    )
    {
        using conserved_t = typename prim_t::counterpart_t;

        // initialize the delta for tracking changes to the body
        auto delta = BodyDelta<real, Dims>{body.index};

        // get the rigid component settings
        const auto& rigid_comp   = body.rigid.value();
        const bool apply_no_slip = rigid_comp.apply_no_slip;

        // calculate distance vector from body center to cell center
        const auto cell_center = ctx.cell_pos;
        const auto r_vector    = cell_center - body.position;
        const auto distance    = r_vector.norm();

        // get normalized direction and signed distance to surface
        constexpr real SAFE_MINIMUM = 1e-10;
        const auto r_norm           = my_max(SAFE_MINIMUM, distance);
        const auto r_hat            = r_vector / r_norm;
        // positive outside, negative inside
        const auto signed_distance = distance - body.radius;

        // get fluid properties
        const auto density        = prim.labframe_density();
        const auto pressure       = prim.press();
        const auto sound_speed    = prim.sound_speed(ctx.gamma);
        const auto fluid_velocity = prim.velocity();
        const auto mach_number =
            fluid_velocity.norm() / my_max(sound_speed, SAFE_MINIMUM);
        const bool is_supersonic = mach_number > 1.0;

        // calculate thickness of boundary region based on flow speed
        // for supersonic flow, we need a thinner, sharper boundary layer
        real boundary_thickness;
        if (is_supersonic) {
            // narrower boundary for supersonic flows
            boundary_thickness = 0.5 * ctx.min_cell_width;
        }
        else {
            boundary_thickness = 1.0 * ctx.min_cell_width;
        }

        // calculate region of influence for pre-emptive forcing
        // extend further into fluid for supersonic flows to prevent
        // penetration
        const real extended_radius =
            body.radius +
            (is_supersonic ? 2.0 * boundary_thickness : boundary_thickness);

        // skip cells too far from boundary
        if (distance > extended_radius + boundary_thickness) {
            return {conserved_t(), delta};
        }

        // get body velocity at this location (including rotation)
        auto body_velocity = body.velocity;
        // if (body.angular_velocity.norm() > SAFE_MINIMUM) {
        //     const auto rotational_velocity =
        //         vecops::cross(body.angular_velocity, r_vector);
        //     body_velocity += rotational_velocity;
        // }

        // calculate relative velocity between fluid and body
        const auto rel_velocity = fluid_velocity - body_velocity;

        // calculate normal and tangential components
        const auto normal_rel_velocity =
            vecops::dot(rel_velocity, r_hat) * r_hat;
        const auto tangential_rel_velocity = rel_velocity - normal_rel_velocity;

        // initialize force components
        vector_t<real, Dims> total_force{};

        // scale strength based on flow characteristics
        // stronger forcing for supersonic flows
        real base_strength;
        if (mach_number > 1.0) {
            // much stronger forcing for supersonic
            base_strength = 25.0 * density * sound_speed * sound_speed;
        }
        else {
            base_strength = 10.0 * density * sound_speed * sound_speed;
        }

        // determine forcing based on cell location
        if (signed_distance < 0) {
            // inside body - extremely strong force to prevent penetration
            const real depth_ratio = std::abs(signed_distance) / body.radius;
            // exponential increase in strength deeper inside body
            const real interior_factor = 1.0 + 10.0 * depth_ratio * depth_ratio;

            // force fluid velocity to exactly match body velocity
            total_force = -rel_velocity * base_strength * interior_factor;

            // add extra pressure force pointing outward for supersonic
            if (mach_number > 1.0) {
                const real pressure_scale = 2.0 * interior_factor;
                total_force += r_hat * pressure * pressure_scale;
            }
        }
        else if (signed_distance < boundary_thickness) {
            // boundary region - transition forcing
            const real boundary_factor =
                1.0 - signed_distance / boundary_thickness;
            // use steeper, non-linear profile for sharper boundary
            const real sharp_factor = std::pow(boundary_factor, 3);

            // stronger reflection force for incoming flow
            const real reflection_scale =
                my_max(0.0, -vecops::dot(fluid_velocity, r_hat));
            const real incoming_factor =
                1.0 +
                2.0 * reflection_scale / my_max(sound_speed, SAFE_MINIMUM);

            // apply appropriate boundary conditions
            if (apply_no_slip) {
                // no-slip: match boundary velocity exactly
                total_force = -rel_velocity * base_strength * sharp_factor *
                              incoming_factor;
            }
            else {
                // free-slip: only constrain normal component
                total_force = -normal_rel_velocity * base_strength *
                              sharp_factor * incoming_factor;
            }

            // for supersonic, add shock-handling term
            if (mach_number > 1.0) {
                // add artificial viscosity term to stabilize shock
                const real shock_strength = 0.5 * base_strength * sharp_factor;
                total_force += -tangential_rel_velocity * shock_strength;

                // add pressure term for shock stabilization
                const real pressure_scale =
                    0.5 * sharp_factor * incoming_factor;
                total_force += r_hat * pressure * pressure_scale;
            }
        }
        else if (mach_number > 1.0 &&
                 signed_distance < 2.0 * boundary_thickness) {
            // pre-emptive zone for supersonic flows only
            // gradually slow down incoming flow before it hits the boundary
            const real pre_factor =
                1.0 -
                (signed_distance - boundary_thickness) / boundary_thickness;
            const real pre_strength =
                0.5 * base_strength * std::pow(pre_factor, 2);

            // only apply to incoming flow
            const real incoming_velocity =
                -my_min(0.0, vecops::dot(rel_velocity, r_hat));
            if (incoming_velocity > 0.1 * sound_speed) {
                total_force = -normal_rel_velocity * pre_strength;
            }
        }
        else {
            // outside all forcing regions
            return {conserved_t(), delta};
        }

        // calculate momentum change for the fluid
        const auto dp = total_force * ctx.dt;

        // calculate energy change (work done on fluid)
        const auto& v_old = fluid_velocity;
        const auto invd   = 1.0 / density;
        const auto v_new  = (prim.spatial_momentum(ctx.gamma) + dp) * invd;
        const auto v_avg  = 0.5 * (v_old + v_new);
        const auto dE     = vecops::dot(v_avg, dp);

        // conservative update for the fluid
        conserved_t result(0.0, dp, dE);

        // apply two-way coupling - reaction force on the body
        if (body.two_way_coupling) {
            delta.force_delta  = -dp * ctx.cell_volume;
            const auto r_lever = body.position - cell_center;
            if constexpr (Dims == 2) {
                // 2D case: only z-component of torque
                delta.torque_delta[1] =
                    vecops::cross(r_lever, delta.force_delta);
            }
            else if constexpr (Dims == 3) {
                // 3D case: full torque vector
                delta.torque_delta = vecops::cross(r_lever, delta.force_delta);
            }
        }

        return {result, delta};
    }

    // core body interaction - handles all capability types
    template <
        is_hydro_primitive_c prim_t,
        std::uint64_t Dims = prim_t::dimensions>
    DEV auto compute_body_interaction(
        const Body<real, Dims>& body,
        const prim_t& prim_state,
        const physics_context_t<Dims>& ctx
    )
    {
        using conserved_t = typename prim_t::counterpart_t;

        conserved_t fluid_change{};
        BodyDelta<real, Dims> body_delta{body.index};

        // check each capability and apply appropriate physics
        if (body.has_capability(BodyCapability::GRAVITATIONAL)) {
            auto [grav_fluid, grav_delta] =
                apply_gravitational_interaction(body, prim_state, ctx);
            fluid_change += grav_fluid;
            body_delta += grav_delta;
        }

        if (body.has_capability(BodyCapability::ACCRETION)) {
            auto [accr_fluid, accr_delta] =
                apply_accretion_interaction(body, prim_state, ctx);
            fluid_change += accr_fluid;
            body_delta += accr_delta;
        }

        if (body.has_capability(BodyCapability::RIGID)) {
            auto [rigid_fluid, rigid_delta] =
                apply_rigid_interaction(body, prim_state, ctx);
            fluid_change += rigid_fluid;
            body_delta += rigid_delta;
        }

        return std::make_pair(fluid_change, body_delta);
    }

    // coordinate-native computation - no cell objects needed
    template <std::uint64_t Dims, Geometry G, typename Primitive>
    DEV auto compute_ib_at_coordinate(
        const uarray<Dims>& coord,
        const mesh::geometry_solver_t<Dims, G>& geo,
        const ComponentBodySystem<real, Dims>& bodies,
        GridBodyDeltaCollector<real, Dims>& collector,
        const physics_context_t<Dims>& ctx
    )
    {
        using conserved_t = typename Primitive::counterpart_t;

        // geometry solver gives us everything we need
        const auto cartesian_pos = geo.cartesian_centroid(coord);
        const auto cell_volume   = geo.volume(coord);
        const auto prim_state =
            get_primitive_at(coord);   // from your existing infrastructure

        conserved_t total_effect{};

        // loop through all bodies - accumulate effects and deltas in one pass
        for (const auto& [body_idx, body] : bodies.bodies()) {
            auto [fluid_change, body_delta] =
                compute_body_interaction(body, prim_state, ctx);

            total_effect += fluid_change;
            // side effect (kinda)
            collector.record_delta(coord, body_idx, body_delta);
        }

        return total_effect;
    }

}   // namespace simbi::ibsystem

#endif
