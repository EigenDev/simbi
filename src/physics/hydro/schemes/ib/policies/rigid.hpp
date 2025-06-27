/**
 *=============================================================================
 *           SIMBI - Special Relativistic Magnetohydrodynamics Code
 *=============================================================================
 *
 * @file            rigid.hpp
 * @brief           sharp interface immersed boundary method for rigid bodies
 * @details         cell-by-cell forcing approach for compressible flows
 *
 * @version         1.0.0
 * @date            2025-05-20
 * @author          Marcus DuPont
 * @email           marcus.dupont@princeton.edu
 *
 *==============================================================================
 */

#ifndef RIGID_HPP
#define RIGID_HPP

#include "config.hpp"
#include "geometry/mesh/cell.hpp"
#include "physics/hydro/schemes/ib/delta/body_delta.hpp"
#include "physics/hydro/schemes/ib/systems/body.hpp"
#include "physics/hydro/types/context.hpp"

namespace simbi::ibsystem::body_functions {
    namespace rigid {

        template <typename T, std::uint64_t Dims, typename Primitive>
        DEV std::pair<typename Primitive::counterpart_t, BodyDelta<T, Dims>>
        apply_sharp_interface_ibm(
            const std::uint64_t body_idx,
            const Body<T, Dims>& body,
            const Primitive& prim,
            const auto& mesh_cell,
            const HydroContext& context,
            const T dt
        )
        {
            using conserved_t = typename Primitive::counterpart_t;

            // initialize the delta for tracking changes to the body
            auto delta = BodyDelta<T, Dims>{body_idx};

            // get the rigid component settings
            const auto& rigid_comp   = body.rigid.value();
            const bool apply_no_slip = rigid_comp.apply_no_slip;

            // calculate distance vector from body center to cell center
            const auto cell_center = mesh_cell.cartesian_centroid();
            const auto r_vector    = cell_center - body.position;
            const auto distance    = r_vector.norm();

            // get normalized direction and signed distance to surface
            constexpr T SAFE_MINIMUM = 1e-10;
            const auto r_norm        = my_max(SAFE_MINIMUM, distance);
            const auto r_hat         = r_vector / r_norm;
            // positive outside, negative inside
            const auto signed_distance = distance - body.radius;

            // get fluid properties
            const auto density        = prim.labframe_density();
            const auto pressure       = prim.press();
            const auto sound_speed    = prim.sound_speed(context.gamma);
            const auto fluid_velocity = prim.velocity();
            const auto mach_number =
                fluid_velocity.norm() / my_max(sound_speed, SAFE_MINIMUM);
            const bool is_supersonic = mach_number > 1.0;

            // calculate thickness of boundary region based on flow speed
            // for supersonic flow, we need a thinner, sharper boundary layer
            T boundary_thickness;
            if (is_supersonic) {
                // narrower boundary for supersonic flows
                boundary_thickness = 0.5 * mesh_cell.min_cell_width();
            }
            else {
                boundary_thickness = 1.0 * mesh_cell.min_cell_width();
            }

            // calculate region of influence for pre-emptive forcing
            // extend further into fluid for supersonic flows to prevent
            // penetration
            const T extended_radius =
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
            const auto tangential_rel_velocity =
                rel_velocity - normal_rel_velocity;

            // initialize force components
            vector_t<T, Dims> total_force{};

            // scale strength based on flow characteristics
            // stronger forcing for supersonic flows
            T base_strength;
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
                const T depth_ratio = std::abs(signed_distance) / body.radius;
                // exponential increase in strength deeper inside body
                const T interior_factor =
                    1.0 + 10.0 * depth_ratio * depth_ratio;

                // force fluid velocity to exactly match body velocity
                total_force = -rel_velocity * base_strength * interior_factor;

                // add extra pressure force pointing outward for supersonic
                if (mach_number > 1.0) {
                    const T pressure_scale = 2.0 * interior_factor;
                    total_force += r_hat * pressure * pressure_scale;
                }
            }
            else if (signed_distance < boundary_thickness) {
                // boundary region - transition forcing
                const T boundary_factor =
                    1.0 - signed_distance / boundary_thickness;
                // use steeper, non-linear profile for sharper boundary
                const T sharp_factor = std::pow(boundary_factor, 3);

                // stronger reflection force for incoming flow
                const T reflection_scale =
                    my_max(0.0, -vecops::dot(fluid_velocity, r_hat));
                const T incoming_factor =
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
                    const T shock_strength = 0.5 * base_strength * sharp_factor;
                    total_force += -tangential_rel_velocity * shock_strength;

                    // add pressure term for shock stabilization
                    const T pressure_scale =
                        0.5 * sharp_factor * incoming_factor;
                    total_force += r_hat * pressure * pressure_scale;
                }
            }
            else if (mach_number > 1.0 &&
                     signed_distance < 2.0 * boundary_thickness) {
                // pre-emptive zone for supersonic flows only
                // gradually slow down incoming flow before it hits the boundary
                const T pre_factor =
                    1.0 -
                    (signed_distance - boundary_thickness) / boundary_thickness;
                const T pre_strength =
                    0.5 * base_strength * std::pow(pre_factor, 2);

                // only apply to incoming flow
                const T incoming_velocity =
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
            const auto dp = total_force * dt;

            // calculate energy change (work done on fluid)
            const auto& v_old = fluid_velocity;
            const auto invd   = 1.0 / density;
            const auto v_new =
                (prim.spatial_momentum(context.gamma) + dp) * invd;
            const auto v_avg = 0.5 * (v_old + v_new);
            const auto dE    = vecops::dot(v_avg, dp);

            // conservative update for the fluid
            conserved_t result(0.0, dp, dE);

            // apply two-way coupling - reaction force on the body
            if (body.two_way_coupling) {
                delta.force_delta  = -dp * mesh_cell.volume();
                const auto r_lever = body.position - cell_center;
                if constexpr (Dims == 2) {
                    // 2D case: only z-component of torque
                    delta.torque_delta[1] =
                        vecops::cross(r_lever, delta.force_delta);
                }
                else if constexpr (Dims == 3) {
                    // 3D case: full torque vector
                    delta.torque_delta =
                        vecops::cross(r_lever, delta.force_delta);
                }
            }

            return {result, delta};
        }

        /**
         * @brief apply sharp interface immersed boundary forces to the
         * fluid
         *
         * this function uses a cell-by-cell approach to apply forcing from
         * a rigid body to each fluid cell, creating a sharp representation
         * of the boundary
         */
        // template <typename T, std::uint64_t Dims, typename Primitive>
        // DEV std::pair<typename Primitive::counterpart_t, BodyDelta<T,
        // Dims>> apply_sharp_interface_ibm(
        //     const std::uint64_t body_idx,
        //     const Body<T, Dims>& body,
        //     const Primitive& prim,
        //     const auto& mesh_cell,
        //     const HydroContext& context,
        //     const T dt
        // )
        // {
        //     using conserved_t = typename Primitive::counterpart_t;

        //     // initialize the delta for tracking changes to the body
        //     auto delta = BodyDelta<T, Dims>{body_idx, {}, {}, 0, 0, 0};

        //     // get the rigid component settings
        //     const auto& rigid_comp   = body.rigid.value();
        //     const bool apply_no_slip = rigid_comp.apply_no_slip;

        //     // calculate distance vector from body center to cell center
        //     const auto cell_center = mesh_cell.cartesian_centroid();
        //     const auto r_vector    = cell_center - body.position;
        //     const auto distance    = r_vector.norm();

        //     // get normalized direction and signed distance to surface
        //     constexpr T SAFE_MINIMUM = 1e-10;
        //     const auto r_norm         = my_max(SAFE_MINIMUM, distance);
        //     const auto r_hat          = r_vector / r_norm;
        //     // positive outside, negative inside
        //     const auto signed_distance = distance - body.radius;

        //     // constants for the different regions
        //     // narrow band around interface
        //     constexpr T BOUNDARY_THICKNESS = 0.05;
        //     // stronger forcing inside body
        //     constexpr T INTERIOR_STRENGTH = 10.0;
        //     // scale for transition region
        //     constexpr T TRANSITION_SCALE = 1.0;

        //     // skip cells too far from boundary
        //     if (signed_distance > 2.0 * mesh_cell.min_cell_width()) {
        //         return {conserved_t(), delta};
        //     }

        //     // get fluid properties
        //     const auto density        = prim.labframe_density();
        //     const auto pressure       = prim.press();
        //     const auto sound_speed    = prim.sound_speed(context.gamma);
        //     const auto fluid_velocity = prim.velocity();

        //     // get body velocity at this location (TODO: include
        //     rotation) auto body_velocity = body.velocity;
        //     // if (body.angular_velocity.norm() > SAFE_MINIMUM) {
        //     //     const auto rotational_velocity =
        //     //         vecops::cross(body.angular_velocity, r_vector);
        //     //     body_velocity += rotational_velocity;
        //     // }

        //     // calculate relative velocity between fluid and body
        //     const auto rel_velocity = fluid_velocity - body_velocity;

        //     // initialize force components
        //     vector_t<T, Dims> total_force{};

        //     // determine cell type based on signed distance
        //     if (signed_distance < 0) {
        //         // inside body - strong force to maintain body velocity
        //         const auto interior_strength =
        //             INTERIOR_STRENGTH * density * sound_speed *
        //             sound_speed;

        //         // penalize all velocity deviation inside body
        //         total_force = -rel_velocity * interior_strength;

        //         // stronger forcing deeper inside
        //         const T depth_factor =
        //             my_min(1.0, std::abs(signed_distance) / body.radius);
        //         total_force *= (1.0 + 2.0 * depth_factor);
        //     }
        //     else if (signed_distance <
        //              BOUNDARY_THICKNESS * mesh_cell.min_cell_width()) {
        //         // narrow band around interface - precise boundary
        //         imposition

        //         // calculate exact boundary intercept postd::int64_t along
        //         r_hat const auto boundary_postd::int64_t = body.position +
        //         r_hat * body.radius;

        //         // for cells very close to boundary, use targeted forcing
        //         const T boundary_factor =
        //             1.0 - signed_distance /
        //                       (BOUNDARY_THICKNESS *
        //                       mesh_cell.min_cell_width());
        //         const T boundary_strength =
        //             TRANSITION_SCALE * density * sound_speed *
        //             sound_speed;

        //         if (apply_no_slip) {
        //             // no-slip: force entire relative velocity to zero at
        //             // boundary
        //             total_force =
        //                 -rel_velocity * boundary_strength *
        //                 boundary_factor;
        //         }
        //         else {
        //             // free-slip: force only normal component
        //             const auto normal_vel_component =
        //                 vecops::dot(rel_velocity, r_hat) * r_hat;
        //             total_force = -normal_vel_component *
        //             boundary_strength *
        //                           boundary_factor;
        //         }

        //         // add compressibility correction for high mach number
        //         flows if (vecops::dot(rel_velocity, r_hat) / sound_speed
        //         > 0.1) {
        //             // higher forcing for compression, lower for
        //             expansion const T mach_factor =
        //                 1.0 + std::min(
        //                           1.0,
        //                           vecops::dot(rel_velocity, r_hat) /
        //                           sound_speed
        //                       );
        //             total_force *= mach_factor;

        //             // add pressure gradient force for high mach flows
        //             if (context.gamma > 1.0) {
        //                 // approximate pressure gradient along normal
        //                 direction
        //                 // helps prevent spurious acoustic reflections
        //                 const T pressure_scale = 0.2 * boundary_factor;
        //                 const auto pressure_force =
        //                     r_hat * pressure * pressure_scale;
        //                 total_force += pressure_force;
        //             }
        //         }
        //     }
        //     else {
        //         // outside the narrow band - no forcing
        //         return {conserved_t(), delta};
        //     }

        //     // calculate momentum change for the fluid
        //     const auto dp = total_force * dt;

        //     // calculate energy change (work done on fluid)
        //     const auto& v_old = fluid_velocity;
        //     const auto invd   = 1.0 / density;
        //     const auto v_new =
        //         (prim.spatial_momentum(context.gamma) + dp) * invd;
        //     const auto v_avg = 0.5 * (v_old + v_new);
        //     const auto dE    = vecops::dot(v_avg, dp);

        //     // conservative update for the fluid
        //     conserved_t result(0.0, dp, dE);
        //     // std::cout << "result: " << result << std::endl;

        //     // apply two-way coupling - reaction force on the body
        //     if (body.two_way_coupling) {
        //         delta.force_delta = -dp * mesh_cell.volume();

        //         // calculate torque contribution (r × F)
        //         const auto r_lever = cell_center - body.position;
        //         if constexpr (Dims == 2) {
        //             // 2D case: only z-component of torque
        //             delta.torque_delta[1] =
        //                 vecops::cross(r_lever, delta.force_delta);
        //         }
        //         else if constexpr (Dims == 3) {
        //             // 3D case: full torque vector
        //             delta.torque_delta =
        //                 vecops::cross(r_lever, delta.force_delta);
        //         }
        //     }

        //     return {result, delta};
        // }

        /**
         * @brief main entry postd::int64_t for rigid body ibm using sharp
         * interface approach
         *
         * wrapper that calls the sharp interface implementation
         */
        template <typename T, std::uint64_t Dims, typename Primitive>
        DEV std::pair<typename Primitive::counterpart_t, BodyDelta<T, Dims>>
        apply_rigid_body_ibm(
            std::uint64_t body_idx,
            const Body<T, Dims>& body,
            const Primitive& prim,
            const Cell<Dims>& mesh_cell,
            const HydroContext& context,
            T dt
        )
        {
            return apply_sharp_interface_ibm<T, Dims, Primitive>(
                body_idx,
                body,
                prim,
                mesh_cell,
                context,
                dt
            );
        }

    }   // namespace rigid
}   // namespace simbi::ibsystem::body_functions

#endif
