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
