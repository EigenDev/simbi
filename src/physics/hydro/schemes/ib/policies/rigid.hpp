/**
 *=============================================================================
 *           SIMBI - Special Relativistic Magnetohydrodynamics Code
 *=============================================================================
 *
 * @file            rigid.hpp
 * @brief           where rigid body functions are defined
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

#ifndef RIGID_HPP
#define RIGID_HPP

#include "build_options.hpp"
#include "geometry/mesh/cell.hpp"
#include "physics/hydro/schemes/ib/delta/body_delta.hpp"
#include "physics/hydro/schemes/ib/systems/body.hpp"
#include "physics/hydro/types/context.hpp"

namespace simbi::ibsystem::body_functions {
    namespace rigid {
        // apply rigid body forces to the fluid state
        template <typename T, size_type Dims, typename Primitive>
        DEV std::pair<typename Primitive::counterpart_t, BodyDelta<T, Dims>>
        apply_rigid_body_force(
            const size_type body_idx,
            const Body<T, Dims>& body,
            const Primitive& prim,
            const auto& mesh_cell,
            const HydroContext& context,
            const T dt
        )
        {
            using conserved_t = typename Primitive::counterpart_t;

            // initialize the delta for tracking changes to the body
            auto delta = BodyDelta<T, Dims>{body_idx, {}, 0, 0, 0};

            // extend interaction beyond body radius
            constexpr T INTERACTION_RADIUS_FACTOR = 1.2;

            // get the rigid component settings
            const auto& rigid_comp   = body.rigid.value();
            const bool apply_no_slip = rigid_comp.apply_no_slip;
            const T interaction_radius =
                body.radius * INTERACTION_RADIUS_FACTOR;

            // calculate distance vector from body center to cell center
            const auto r_vector =
                mesh_cell.cartesian_centroid() - body.position;
            const auto distance = r_vector.norm();

            // if outside interaction radius, no effect
            if (distance > interaction_radius) {
                return {conserved_t(), delta};
            }

            // calculate normalized direction vector and penetration depth
            constexpr T SAFE_DISTANCE = 1e-10;
            const auto r_norm         = my_max(SAFE_DISTANCE, distance);
            const auto r_hat          = r_vector / r_norm;
            const auto penetration    = body.radius - distance;

            // get fluid properties
            const auto density        = prim.labframe_density();
            const auto pressure       = prim.press();
            const auto sound_speed    = prim.sound_speed(context.gamma);
            const auto fluid_velocity = prim.velocity();

            // calculate relative velocity between fluid and body
            const auto rel_velocity = fluid_velocity - body.velocity;

            // apply IBM force - using smoothed delta function for force
            // distribution
            // NOTE --- [this is the key part for handling compressible fluid
            // with IBM]
            T kernel_value;
            if (distance < body.radius) {
                // inside the body - strong repulsive force
                kernel_value = 1.0;
            }
            else {
                // outside but within interaction radius - smoothly decaying
                // force
                const T normalized_distance =
                    (distance - body.radius) /
                    (interaction_radius - body.radius);
                // smoothing kernel with compact support
                kernel_value = std::max(
                    0.0,
                    1.0 - 3.0 * std::pow(normalized_distance, 2) +
                        2.0 * std::pow(normalized_distance, 3)
                );
            }

            // calculate force components:

            // i) repulsion force based on penetration depth
            const auto penetration_factor =
                penetration > 0.0 ? penetration / body.radius : 0.0;
            auto repulsion_force =
                r_hat * penetration_factor * std::pow(kernel_value, 2);

            // 2. no-slip or free-slip condition
            auto viscous_force = spatial_vector_t<T, Dims>();
            if (apply_no_slip) {
                // for no-slip condition, counteract tangential velocity
                // component
                const auto normal_vel_component =
                    vecops::dot(rel_velocity, r_hat) * r_hat;
                const auto tangential_vel_component =
                    rel_velocity - normal_vel_component;
                viscous_force = -tangential_vel_component * kernel_value;
            }

            // iii) compression/rarefaction handling for compressible fluid
            // scale force based on local pressure and sound speed
            constexpr T MIN_SOUND_SPEED = 1e-10;
            const auto mach_scaling =
                1.0 + std::min(
                          1.0,
                          vecops::dot(rel_velocity, r_hat) /
                              (sound_speed + MIN_SOUND_SPEED)
                      );

            // combine forces with appropriate scaling factors
            // these factors should be tuned based on your specific simulation
            // needs
            const T repulsion_strength =
                5.0 * density * sound_speed * sound_speed;
            const T viscous_strength = 0.5 * density * sound_speed;

            const auto total_force =
                repulsion_force * repulsion_strength * mach_scaling +
                viscous_force * viscous_strength;

            // calculate momentum change for the fluid
            const auto dp = total_force * dt;

            // calculate energy change (work done on fluid)
            const auto& v_old = fluid_velocity;
            const auto invd   = 1.0 / density;
            const auto v_new =
                (prim.spatial_momentum(context.gamma) + dp) * invd;
            const auto v_avg = 0.5 * (v_old + v_new);
            const auto dE    = vecops::dot(v_avg, dp);

            // apply two-way coupling - reaction force on the body
            if (body.two_way_coupling) {
                delta.force_delta = -dp * mesh_cell.volume();
            }

            return {conserved_t(0.0, dp, dE), delta};
        }

        template <typename T, size_type Dims, typename Primitive>
        DEV std::pair<typename Primitive::counterpart_t, BodyDelta<T, Dims>>
        apply_rigid_body_ibm(
            size_type body_idx,
            const Body<T, Dims>& body,
            const Primitive& prim,
            const Cell<Dims>& mesh_cell,
            const HydroContext& context,
            T dt
        )
        {
            using conserved_t = typename Primitive::counterpart_t;

            // initialize the delta for tracking changes to the body
            auto delta = BodyDelta<T, Dims>{body_idx, {}, 0, 0, 0};

            // physical parameters and constants
            //
            //
            // extend interaction beyond body radius
            constexpr T INTERACTION_RADIUS_FACTOR = 1.2;
            // thickness of the boundary region in cell widths
            constexpr T BOUNDARY_THICKNESS_FACTOR = 1.0;
            // prevent division by zero
            constexpr T MIN_SAFE_DISTANCE = 1e-10;
            // strength of repulsion force
            constexpr T REPULSION_STRENGTH = 5.0;
            // strength of viscous (no-slip) force
            constexpr T VISCOUS_STRENGTH = 0.5;

            // get the rigid component settings
            const auto& rigid_comp   = body.rigid.value();
            const bool apply_no_slip = rigid_comp.apply_no_slip;
            const T interaction_radius =
                body.radius * INTERACTION_RADIUS_FACTOR;

            // cell properties
            const auto cell_center = mesh_cell.cartesian_centroid();
            const auto cell_volume = mesh_cell.volume();
            const auto cell_width  = mesh_cell.min_cell_width();

            // calculate distance vector from body center to cell center
            const auto r_vector = cell_center - body.position;
            const auto distance = r_vector.norm();

            // if outside interaction radius, no effect
            if (distance > interaction_radius) {
                return {conserved_t(), delta};
            }

            // calculate normalized direction vector
            const auto r_norm = std::max(MIN_SAFE_DISTANCE, distance);
            const auto r_hat  = r_vector / r_norm;

            // calculate penetration depth (positive means inside the body)
            const auto penetration = body.radius - distance;

            // get fluid properties
            const auto density        = prim.labframe_density();
            const auto pressure       = prim.press();
            const auto sound_speed    = prim.sound_speed(context.gamma);
            const auto fluid_velocity = prim.velocity();

            // calculate relative velocity between fluid and body
            const auto rel_velocity = fluid_velocity - body.velocity;

            // initialize forces and source terms
            spatial_vector_t<T, Dims> total_force{};
            T mass_source = 0.0;

            // check if cell is a boundary cell (near the solid-fluid interface)
            const bool is_boundary_cell =
                std::abs(distance - body.radius) <
                BOUNDARY_THICKNESS_FACTOR * cell_width;

            if (is_boundary_cell) {
                // === ibm treatment for cells intersecting the boundary ===

                // reset accumulated flux
                T accumulated_mass_flux = 0.0;
                int crossing_faces      = 0;

                // loop over cell faces to find those crossing the boundary
                for (size_type f = 0; f < 2 * Dims; ++f) {
                    const auto normal      = mesh_cell.normal_vec(f);
                    const auto face_coord  = mesh_cell.normal(f);
                    const auto face_center = normal * face_coord;

                    // calculate face distance to body center
                    const auto face_r_vector = face_center - body.position;
                    const auto face_distance = face_r_vector.norm();

                    // check opposite face
                    const size_type opposite_face = f % 2 == 0 ? f + 1 : f - 1;
                    const auto opp_normal = mesh_cell.normal_vec(opposite_face);
                    const auto opp_coord  = mesh_cell.normal(opposite_face);
                    const auto opp_center = opp_normal * opp_coord;

                    const auto opp_r_vector = opp_center - body.position;
                    const auto opp_distance = opp_r_vector.norm();

                    // check if boundary crosses between faces
                    const bool is_crossed_face = (face_distance < body.radius &&
                                                  opp_distance > body.radius) ||
                                                 (face_distance > body.radius &&
                                                  opp_distance < body.radius);

                    if (is_crossed_face) {
                        // count crossing faces for diagnostics
                        crossing_faces++;

                        // calculate boundary location on grid line
                        T distance_ratio;
                        if (face_distance < body.radius) {
                            distance_ratio = (body.radius - face_distance) /
                                             std::max(
                                                 opp_distance - face_distance,
                                                 MIN_SAFE_DISTANCE
                                             );
                        }
                        else {
                            distance_ratio = (body.radius - opp_distance) /
                                             std::max(
                                                 face_distance - opp_distance,
                                                 MIN_SAFE_DISTANCE
                                             );
                        }

                        // calculate β parameter for ibm (huang & sung 2007)
                        const T beta = 0.5 - distance_ratio;

                        // face velocity (ideally would interpolate from
                        // neighboring cells)
                        const auto u_face = fluid_velocity;

                        // set boundary velocity to rigid body velocity
                        const auto u_boundary = body.velocity;

                        // calculate virtual cell velocity for ibm (eqn. 19 & 20
                        // from Huang & Sung 2007)
                        auto u_virtual = [&]() -> spatial_vector_t<T, Dims> {
                            if (beta > 0) {
                                // face outside boundary (eqn. 19)
                                return beta / (1.0 + 2.0 * beta) * u_face;
                            }
                            else {
                                // face inside boundary - use boundary velocity
                                // (eqn. 20)
                                return ((3.0 / 2.0) + (beta / 2.0)) *
                                           u_boundary -
                                       ((1.0 / 2.0) + (beta / 2.0)) * u_face;
                            }
                        }();

                        // === compressible extension ===

                        // get local mach number for compressible effects
                        const T mach_number =
                            vecops::norm(u_face) / sound_speed;

                        // calculate density in virtual cell - use physical jump
                        // conditions
                        T rho_virtual;

                        if (beta > 0) {
                            // face outside boundary
                            rho_virtual = density;
                        }
                        else {
                            // face inside boundary - density can jump across
                            // boundary
                            if (mach_number > 1.0) {
                                // supersonic - use shock jump conditions
                                // (simplified)
                                const T gamma = context.gamma;
                                const T shock_ratio =
                                    ((gamma + 1.0) * mach_number * mach_number
                                    ) /
                                    ((gamma - 1.0) * mach_number * mach_number +
                                     2.0);
                                rho_virtual = density * shock_ratio;
                            }
                            else {
                                // subsonic - smoother transition
                                rho_virtual =
                                    (1.0 + beta) * density - beta * density;
                            }
                        }

                        // calculate mass flux through this face
                        T face_mass_flux = 0.0;

                        // contribution from fluid side
                        if (face_distance < body.radius) {
                            face_mass_flux +=
                                density * vecops::dot(u_face, normal);
                        }

                        // contribution from virtual cell
                        face_mass_flux -=
                            beta * rho_virtual * vecops::dot(u_virtual, normal);

                        // add to mass source term, scaled by face area
                        accumulated_mass_flux +=
                            face_mass_flux * mesh_cell.area(f);
                    }
                }

                // normalize by cell volume if we found any crossing faces
                if (crossing_faces > 0) {
                    mass_source = accumulated_mass_flux / cell_volume;
                }

                // calculate kernel value for force distribution
                T kernel_value;
                if (distance < body.radius) {
                    // inside the body - strong repulsive force
                    kernel_value = 1.0;
                }
                else {
                    // outside but within interaction radius - smoothly decaying
                    // force
                    const T normalized_distance =
                        (distance - body.radius) /
                        (interaction_radius - body.radius);
                    // smoothing kernel with compact support
                    kernel_value = std::max(
                        0.0,
                        1.0 - 3.0 * std::pow(normalized_distance, 2) +
                            2.0 * std::pow(normalized_distance, 3)
                    );
                }

                // calculate force components:

                // i) repulsion force based on penetration depth
                const auto penetration_factor =
                    penetration > 0.0 ? penetration / body.radius : 0.0;
                auto repulsion_force =
                    r_hat * penetration_factor * std::pow(kernel_value, 2);

                // ii) no-slip or free-slip condition
                auto viscous_force = spatial_vector_t<T, Dims>();
                if (apply_no_slip) {
                    // for no-slip condition, counteract tangential velocity
                    // component
                    const auto normal_vel_component =
                        vecops::dot(rel_velocity, r_hat) * r_hat;
                    const auto tangential_vel_component =
                        rel_velocity - normal_vel_component;
                    viscous_force = -tangential_vel_component * kernel_value;
                }

                // iii) compression/rarefaction handling for compressible fluid
                // scale force based on local pressure and sound speed
                const auto mach_scaling =
                    1.0 + std::min(
                              1.0,
                              vecops::dot(rel_velocity, r_hat) /
                                  (sound_speed + MIN_SAFE_DISTANCE)
                          );

                // combine forces with appropriate scaling factors
                const T repulsion_strength =
                    REPULSION_STRENGTH * density * sound_speed * sound_speed;
                const T viscous_strength =
                    VISCOUS_STRENGTH * density * sound_speed;

                total_force =
                    repulsion_force * repulsion_strength * mach_scaling +
                    viscous_force * viscous_strength;
            }
            else if (distance < body.radius) {
                // === interior cell treatment (fully inside the body) ===

                // strong repulsion force pointing outward
                const T interior_force_scale =
                    REPULSION_STRENGTH * (1.0 - distance / body.radius);
                const T repulsion_strength =
                    interior_force_scale * density * sound_speed * sound_speed;

                // repulsion force
                auto repulsion_force = r_hat * repulsion_strength;

                // viscous force if no-slip is applied
                auto viscous_force = spatial_vector_t<T, Dims>();
                if (apply_no_slip) {
                    viscous_force = -rel_velocity * VISCOUS_STRENGTH * density *
                                    sound_speed;
                }

                total_force = repulsion_force + viscous_force;
            }
            else {
                // === outside body but within interaction radius ===

                // calculate kernel value for force distribution
                const T normalized_distance =
                    (distance - body.radius) /
                    (interaction_radius - body.radius);
                const T kernel_value = std::max(
                    0.0,
                    1.0 - 3.0 * std::pow(normalized_distance, 2) +
                        2.0 * std::pow(normalized_distance, 3)
                );

                // no repulsion force (we're outside the body)
                auto repulsion_force = spatial_vector_t<T, Dims>();

                // viscous force if no-slip is applied (reduced by distance)
                auto viscous_force = spatial_vector_t<T, Dims>();
                if (apply_no_slip) {
                    // for no-slip condition, counteract tangential velocity
                    // component
                    const auto normal_vel_component =
                        vecops::dot(rel_velocity, r_hat) * r_hat;
                    const auto tangential_vel_component =
                        rel_velocity - normal_vel_component;
                    viscous_force = -tangential_vel_component * kernel_value;
                }

                // combine forces with appropriate scaling
                const T viscous_strength =
                    VISCOUS_STRENGTH * density * sound_speed;
                total_force = viscous_force * viscous_strength;
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

            // create conserved update (adding the mass source for boundary
            // cells)
            conserved_t result(
                mass_source * dt,
                dp + mass_source * dt * fluid_velocity,
                dE + mass_source * dt *
                         (0.5 * vecops::dot(fluid_velocity, fluid_velocity))
            );

            // apply two-way coupling - reaction force on the body
            if (body.two_way_coupling) {
                delta.force_delta = -dp * cell_volume;
            }

            return {result, delta};
        }

    }   // namespace rigid
}   // namespace simbi::ibsystem::body_functions

#endif
