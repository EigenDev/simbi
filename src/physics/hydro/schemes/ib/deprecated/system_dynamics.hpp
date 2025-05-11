/**
 *=============================================================================
 *           SIMBI - Special Relativistic Magnetohydrodynamics Code
 *=============================================================================
 *
 * @file            system_dynamics.hpp
 * @brief
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

#ifndef SYSTEM_DYNAMICS_HPP
#define SYSTEM_DYNAMICS_HPP

#include "build_options.hpp"
#include "physics/hydro/schemes/ib/bodies/types/any_body.hpp"
#include "system_traits.hpp"

namespace simbi::ibsystem::dynamics {
    // binary system
    template <typename T, size_type Dims>
        requires(traits::AtLeastTwoDimensional<Dims>)
    class BinaryDynamics
    {
      public:
        BinaryDynamics(
            const traits::GravitationalTrait<T>& grav_trait,
            const traits::BinaryTrait<T, Dims>& binary_trait
        )
            : grav_trait_(grav_trait), binary_trait_(binary_trait)
        {
        }

        // update the system for prescribed trajectory
        void update_prescribed(ndarray<ib::AnyBody<T, Dims>*> bodies, T time)
        {
            // Use binary trait to calculate new positions and velocities
            ndarray<spatial_vector_t<T, Dims>> positions(bodies.size());
            ndarray<spatial_vector_t<T, Dims>> velocities(bodies.size());

            binary_trait_
                .update_positions_and_velocities(time, positions, velocities);

            // Update bodies
            for (size_t i = 0; i < bodies.size(); ++i) {
                bodies[i]->set_position(positions[i]);
                bodies[i]->set_velocity(velocities[i]);
            }
        }

        // update the system using numerical integration
        void update_numerical(ndarray<ib::AnyBody<T, Dims>*> bodies, T dt)
        {
            // TODO: use leapfrog, symplectic, etc here
            throw std::runtime_error(
                "Numerical integration not implemented yet"
            );
        }

      private:
        traits::GravitationalTrait<T> grav_trait_;
        traits::BinaryTrait<T, Dims> binary_trait_;
    };
}   // namespace simbi::ibsystem::dynamics
#endif   // SYSTEM_DYNAMICS_HPP
