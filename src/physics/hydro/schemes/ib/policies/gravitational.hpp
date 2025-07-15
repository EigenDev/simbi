/**
 *=============================================================================
 *           SIMBI - Special Relativistic Magnetohydrodynamics Code
 *=============================================================================
 *
 * @file            gravitational.hpp
 * @brief           gravitational interaction functions
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

#ifndef GRAVITATIONAL_HPP
#define GRAVITATIONAL_HPP

#include "config.hpp"
#include "data/containers/vector.hpp"
#include "geometry/mesh/cell.hpp"
#include "physics/hydro/schemes/ib/delta/body_delta.hpp"
#include "physics/hydro/schemes/ib/systems/body.hpp"
#include "physics/hydro/types/context.hpp"
#include <cmath>

namespace simbi::ibsystem::body_functions {
    namespace gravitational {

        template <typename T, std::uint64_t Dims, typename Primitive>
        DEV std::pair<typename Primitive::counterpart_t, BodyDelta<T, Dims>>
        apply_gravitational_force(
            std::uint64_t body_idx,
            const Body<T, Dims>& body,
            const Primitive& prim,
            const iarray<Dims>& coords,
            real gamma,
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
            const auto ff_hat =
                -vecops::centralize(f_cart, mesh_cell.geometry());

            const auto density = prim.labframe_density();
            // Calculate momentum and energy change
            const auto dp = density * ff_hat * dt;

            const auto& v_old = prim.velocity();
            const auto invd   = 1.0 / density;
            const auto v_new =
                (prim.linear_momentum(context.gamma) + dp) * invd;
            const auto v_avg = 0.5 * (v_old + v_new);
            const auto dE    = vecops::dot(v_avg, dp);

            // Apply two-way coupling if enabled
            BodyDelta<T, Dims> delta{body_idx};
            if (body.two_way_coupling) {
                // all vector quantities for the body
                // are in Cartesian coordinates
                delta.force_delta =
                    prim.labframe_density() * mesh_cell.volume() * f_cart;
            }

            return {conserved_t(0.0, dp, dE), delta};
        }

    }   // namespace gravitational

}   // namespace simbi::ibsystem::body_functions

#endif
