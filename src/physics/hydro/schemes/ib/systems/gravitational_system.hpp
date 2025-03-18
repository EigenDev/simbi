/**
 *  *=============================================================================
 *  *           SIMBI - Special Relativistic Magnetohydrodynamics Code
 *  *=============================================================================
 *  *
 *  * @file            gravitational_system.hpp
 *  * @brief           Gravitational System of Immersed Bodies
 *  * @details
 *  *
 *  * @version         0.8.0
 *  * @date            2025-02-17
 *  * @author          Marcus DuPont
 *  * @email           marcus.dupont@princeton.edu
 *  *
 *  *==============================================================================
 *  * @build           Requirements & Dependencies
 *  *==============================================================================
 *  * @requires        C++20
 *  * @depends         CUDA >= 11.0, HDF5 >= 1.12, OpenMP >= 4.5
 *  * @platform        Linux, MacOS
 *  * @parallel        GPU (CUDA, HIP), CPU (OpenMP)
 *  *
 *  *==============================================================================
 *  * @documentation   Reference & Notes
 *  *==============================================================================
 *  * @usage
 *  * @note
 *  * @warning
 *  * @todo
 *  * @bug
 *  * @performance
 *  *
 *  *==============================================================================
 *  * @testing        Quality Assurance
 *  *==============================================================================
 *  * @test
 *  * @benchmark
 *  * @validation
 *  *
 *  *==============================================================================
 *  * @history        Version History
 *  *==============================================================================
 *  * 2025-02-17      v0.8.0      Initial implementation
 *  *
 *  *==============================================================================
 *  * @copyright (C) 2025 Marcus DuPont. All rights reserved.
 *  *==============================================================================
 */
#ifndef GRAVITATIONAL_SYSTEM_HPP
#define GRAVITATIONAL_SYSTEM_HPP

#include "build_options.hpp"
#include "core/types/containers/ndarray.hpp"
#include "core/types/utility/enums.hpp"
#include "physics/hydro/schemes/ib/systems/body_system.hpp"
#include <type_traits>

namespace simbi::ibsystem {
    template <typename T, size_type Dims>
    class GravitationalSystem : public BodySystem<T, Dims>
    {
      private:
        struct orbit_params_t {
            T semi_major;
            T eccentricity;
            T inclination;
            T longitude_of_ascending_node;
            T argument_of_periapsis;
            T true_anomaly;
        };

        // Base class type alias
        using Base = BodySystem<T, Dims>;

        // Angular momentum type based on dimensionality
        using ang_type =
            std::conditional_t<Dims == 3, spatial_vector_t<T, Dims>, T>;

        // Use the state types from concepts
        using ConsArray = typename ib::concepts::StateType<Dims>::ConsArray;
        using PrimArray = typename ib::concepts::StateType<Dims>::PrimArray;

        // System state
        spatial_vector_t<T, Dims> com_;
        ang_type angular_momentum_;
        T total_energy_;
        T total_mass_;
        T gamma_;
        orbit_params_t orbital_params_;

        // Conservation tracking
        T initial_energy_;
        ang_type initial_angular_momentum_;

        // Integration strategies
        enum class IntegrationMethod {
            LEAPFROG,
            SYMPLECTIC_EULER
        };

        IntegrationMethod integration_method_ = IntegrationMethod::LEAPFROG;

      public:
        DUAL
        GravitationalSystem(const typename Base::MeshType& mesh, const T gamma)
            : Base(mesh), gamma_(gamma)
        {
        }

        DUAL void update_system(
            ConsArray& cons_states,
            const PrimArray& prim_states,
            const T dt
        )
        {
            switch (integration_method_) {
                case IntegrationMethod::LEAPFROG: integrate_leapfrog(dt); break;
                default: integrate_symplectic_euler(dt); break;
            }

            // Apply forces to fluid if needed
            this->apply_forces_to_fluid(cons_states, prim_states, dt);

            // Update diagnostics
            compute_center_of_mass();
            update_conserved_quantities();
        }

        // Calculate time step based on orbital dynamics
        DUAL T get_orbital_timestep(T cfl) const
        {
            T orbital_dt = std::numeric_limits<T>::infinity();

            // if there is only one body, we can't calculate a timestep
            // so we return infinity
            if (this->bodies().size() < 2) {
                return orbital_dt;
            }

            for (const auto& body_ptr : this->bodies()) {
                const auto pos   = body_ptr->position();
                const auto vel   = body_ptr->velocity();
                const auto force = body_ptr->force();
                const auto mass  = body_ptr->mass();

                // Skip if any values are invalid
                if (mass <= 0) {
                    continue;
                }

                const auto accel = force / mass;

                // Calculate relevant timescales
                const auto r_mag = pos.norm();
                const auto v_mag = vel.norm();
                const auto a_mag = accel.norm();

                if (v_mag > 0) {
                    orbital_dt = std::min(orbital_dt, r_mag / v_mag);
                }

                if (a_mag > 0) {
                    orbital_dt = std::min(orbital_dt, std::sqrt(r_mag / a_mag));
                    orbital_dt = std::min(orbital_dt, v_mag / a_mag);
                }

                // Approximate orbital period
                if (total_mass_ > 0 && r_mag > 0) {
                    const auto period =
                        2.0 * M_PI *
                        std::sqrt(std::pow(r_mag, 3) / (total_mass_ * gamma_));
                    orbital_dt = std::min(orbital_dt, period / 100.0);
                }
            }

            return orbital_dt * cfl;
        }

        // Initialize system state
        DUAL void init_system()
        {
            // Calculate initial forces
            this->calculate_forces(0.0);

            // Initialize conservation quantities
            compute_center_of_mass();
            update_conserved_quantities();

            // Store initial values
            initial_energy_           = total_energy_;
            initial_angular_momentum_ = angular_momentum_;
        }

        // Check conservation errors
        DUAL void check_conservation_errors() const
        {
            if (std::abs(initial_energy_) < 1e-10) {
                return;
            }

            // Check conservation of energy
            const auto energy_error =
                std::abs((total_energy_ - initial_energy_) / initial_energy_);
            if (energy_error > 1e-6) {
                std::cerr << "Energy conservation error: " << energy_error
                          << std::endl;
            }

            // Check conservation of angular momentum (implementation depends on
            // dimension)
            if constexpr (Dims == 3) {
                // For 3D, angular momentum is a vector
                if (initial_angular_momentum_.norm() > 1e-10) {
                    const auto ang_mom_error =
                        (angular_momentum_ - initial_angular_momentum_).norm() /
                        initial_angular_momentum_.norm();
                    if (ang_mom_error > 1e-6) {
                        std::cerr << "Angular momentum conservation error: "
                                  << ang_mom_error << std::endl;
                    }
                }
            }
            else {
                // For 2D, angular momentum is a scalar
                if (std::abs(initial_angular_momentum_) > 1e-10) {
                    const auto ang_mom_error = std::abs(
                        (angular_momentum_ - initial_angular_momentum_) /
                        initial_angular_momentum_
                    );
                    if (ang_mom_error > 1e-6) {
                        std::cerr << "Angular momentum conservation error: "
                                  << ang_mom_error << std::endl;
                    }
                }
            }
        }

      private:
        // Integration methods
        DUAL void integrate_leapfrog(const T dt)
        {
            // First half kick - update velocities
            this->calculate_forces(dt);
            this->advance_velocities(0.5 * dt);

            // Full drift - update positions
            this->advance_positions(dt);

            // Second half kick - update velocities again
            this->calculate_forces(dt);
            this->advance_velocities(0.5 * dt);
        }

        DUAL void integrate_symplectic_euler(const T dt)
        {
            // Calculate forces first
            this->calculate_forces(dt);

            // Update velocities based on forces
            this->advance_velocities(dt);

            // Update positions based on new velocities
            this->advance_positions(dt);
        }

        // Conservation tracking methods
        DUAL void compute_center_of_mass()
        {
            com_        = spatial_vector_t<T, Dims>();
            total_mass_ = 0;

            for (const auto& body_ptr : this->bodies()) {
                const auto pos  = body_ptr->position();
                const auto mass = body_ptr->mass();

                com_ += pos * mass;
                total_mass_ += mass;
            }

            if (total_mass_ > 0) {
                com_ /= total_mass_;
            }
        }

        DUAL void update_conserved_quantities()
        {
            // Reset energy
            total_energy_ = 0;

            // Compute potential energy from gravitational interactions
            const auto& bodies      = this->bodies();
            const size_t num_bodies = bodies.size();

            for (size_t i = 0; i < num_bodies; ++i) {
                for (size_t j = i + 1; j < num_bodies; ++j) {
                    const auto& body1 = bodies[i];
                    const auto& body2 = bodies[j];

                    const auto r =
                        (body2->position() - body1->position()).norm();
                    if (r > 0) {
                        total_energy_ -=
                            body1->G() * body1->mass() * body2->mass() / r;
                    }
                }
            }

            // Reset angular momentum
            if constexpr (Dims == 3) {
                angular_momentum_ = spatial_vector_t<T, Dims>();
            }
            else {
                angular_momentum_ = 0;
            }

            // Add kinetic energy and calculate angular momentum
            for (const auto& body_ptr : bodies) {
                const auto vel  = body_ptr->velocity();
                const auto pos  = body_ptr->position() - com_;
                const auto mass = body_ptr->mass();

                // Kinetic energy
                total_energy_ += 0.5 * mass * vel.dot(vel);

                // Angular momentum
                if constexpr (Dims == 3) {
                    angular_momentum_ += mass * pos.cross(vel);
                }
                else if constexpr (Dims == 2) {
                    angular_momentum_ +=
                        mass * (pos[0] * vel[1] - pos[1] * vel[0]);
                }
            }
        }
    };
}   // namespace simbi::ibsystem

#endif
