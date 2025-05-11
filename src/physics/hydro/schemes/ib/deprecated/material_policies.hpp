/**
 *=============================================================================
 *           SIMBI - Special Relativistic Magnetohydrodynamics Code
 *=============================================================================
 *
 * @file            material_policies.hpp
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

#ifndef MATERIAL_POLICIES_HPP
#define MATERIAL_POLICIES_HPP

#include "../body_traits.hpp"
#include "build_options.hpp"
#include "core/types/containers/ndarray.hpp"
#include "core/types/containers/vector.hpp"

namespace simbi::ib {
    //----------------------------------------------------------------------------
    // Rigid Material
    // ---------------------------------------------------------------------------
    template <typename T, size_type Dims>
    class RigidMaterialPolicy
    {
      private:
        using trait_t = traits::Rigid<T>;

      public:
        using Params = typename trait_t::Params;

        RigidMaterialPolicy(const Params& params = {}) : trait_(params) {}

        // Access to the trait
        DEV const trait_t& rigid_trait() const { return trait_; }
        DEV trait_t& rigid_trait() { return trait_; }

        // Forward trait methods
        DEV T density() const { return trait_.density(); }
        DEV T restitution_coefficient() const
        {
            return trait_.restitution_coefficient();
        }
        DEV bool infinitely_rigid() const { return trait_.infinitely_rigid(); }

        // Implementation for material response
        template <typename Body>
        DEV void update_material_state(Body& body, const T dt)
        {
            // For rigid bodies, we don't need to track internal deformation
            // This method can be empty or do minimal work

            // However, we might handle collision response here
            handle_collision_response(body);
        }

        // Implementation for collision handling
        template <typename Body>
        DEV void handle_collision_response(Body& body)
        {
            // Iterate through the cut cells to check for collisions
            for (const auto& idx : body.cut_cell_indices()) {
                const auto& cell = body.cell_info()[idx];

                // If the cell is inside a solid obstacle
                if (cell.distance < 0 &&
                    std::abs(cell.distance) > global::epsilon) {
                    // Calculate impulse response based on restitution
                    // coefficient
                    const auto normal = cell.normal.normalized();

                    // Get the component of velocity along normal direction
                    const auto v_normal = vecopss::body.velocity().dot(normal);

                    // If we're moving toward the obstacle
                    if (v_normal < 0) {
                        // Apply impulse to bounce away from obstacle
                        const auto impulse =
                            -v_normal *
                            (1.0 + trait_.restitution_coefficient());
                        body.velocity_ += impulse * normal;
                    }
                }
            }
        }

      private:
        trait_t trait_;
    };

    //----------------------------------------------------------------------------
    // Deformable Material
    // ---------------------------------------------------------------------------
    template <typename T, size_type Dims>
    class DeformableMaterialPolicy
    {
      private:
        using trait_t = traits::Deformable<T>;

        // Internal structure to track deformation of different body regions
        struct DeformationPoint {
            spatial_vector_t<T, Dims> position;
            spatial_vector_t<T, Dims> displacement;
            T strain;
            bool yielded;
        };

      public:
        using Params = typename trait_t::Params;

        DeformableMaterialPolicy(const Params& params = {}) : trait_(params)
        {
            // Initialize deformation points if needed
            if constexpr (Dims == 2) {
                // Create a simple 8-point deformation mesh for 2D
                init_deformation_mesh(8);
            }
            else if constexpr (Dims == 3) {
                // Create a 20-point deformation mesh for 3D
                init_deformation_mesh(20);
            }
        }

        // Access to the trait
        DEV const trait_t& deformable_trait() const { return trait_; }
        DEV trait_t& deformable_trait() { return trait_; }

        // Forward core trait methods
        DEV T youngs_modulus() const { return trait_.youngs_modulus(); }
        DEV T poisson_ratio() const { return trait_.poisson_ratio(); }
        DEV T yield_strength() const { return trait_.yield_strength(); }
        DEV T failure_strain() const { return trait_.failure_strain(); }
        DEV bool plastic_deformation() const
        {
            return trait_.plastic_deformation();
        }
        DEV T stored_elastic_energy() const
        {
            return trait_.stored_elastic_energy();
        }
        DEV bool is_permanently_deformed() const
        {
            return trait_.is_permanently_deformed();
        }
        DEV bool is_failed() const { return trait_.is_failed(); }

        // Implementation for updating material state
        template <typename Body>
        DEV void update_material_state(Body& body, const T dt)
        {
            T total_strain_energy = 0;
            T max_strain          = 0;

            // Calculate deformations at each point
            for (auto& point : deformation_points_) {
                // Transform point from body frame to world frame
                const auto world_position = body.position() + point.position;

                // For each point, we check all cut cells to find forces
                for (const auto& idx : body.cut_cell_indices()) {
                    const auto& cell = body.cell_info()[idx];
                    const auto mesh_cell =
                        body.mesh().get_cell_from_global(idx);

                    // Calculate distance from deformation point to cell
                    const auto r    = mesh_cell.centroid() - world_position;
                    const auto dist = r.norm();

                    if (dist <
                        body.radius() * 1.5) {   // Only consider nearby cells
                        // Apply simplified strain model - in a real
                        // implementation this would be a proper finite element
                        // calculation
                        const auto strain =
                            std::abs(cell.distance) / body.radius();
                        point.strain = std::max(point.strain, strain);

                        // Calculate elastic energy
                        const auto stress = trait_.youngs_modulus() * strain;
                        const auto strain_energy =
                            0.5 * stress * strain * mesh_cell.volume();

                        total_strain_energy += strain_energy;

                        // Update displacement
                        point.displacement = cell.normal * cell.distance;

                        // Track maximum strain for material yielding/failure
                        max_strain = std::max(max_strain, strain);

                        // Check for material yielding
                        if (stress > trait_.yield_strength()) {
                            point.yielded = true;
                        }
                    }
                }
            }

            // Update trait state with calculated values
            trait_.update_deformation(max_strain);
            trait_.set_stored_elastic_energy(total_strain_energy);

            // If we're plastically deforming, we would adjust the rest shape
            // here
            if (trait_.plastic_deformation() &&
                trait_.is_permanently_deformed()) {
                update_rest_shape(body);
            }
        }

        // Calculate elastic restoring force
        template <typename Body>
        DEV spatial_vector_t<T, Dims>
        calculate_elastic_force(const Body& body) const
        {
            spatial_vector_t<T, Dims> force;

            for (const auto& point : deformation_points_) {
                // Simple Hookean spring model
                if (point.displacement.norm() > 0) {
                    // Force is proportional to displacement
                    force -= trait_.youngs_modulus() * point.displacement;
                }
            }

            return force;
        }

      private:
        trait_t trait_;
        ndarray<DeformationPoint> deformation_points_;

        // Initialize a simplified deformation mesh
        void init_deformation_mesh(int num_points)
        {
            deformation_points_.resize(num_points);

            if constexpr (Dims == 2) {
                // Create points in a circle for 2D
                const T radius = 0.8;   // Slightly inside the body radius
                for (int i = 0; i < num_points; i++) {
                    const T angle = 2.0 * M_PI * i / num_points;
                    deformation_points_[i].position = spatial_vector_t<T, Dims>{
                      radius * std::cos(angle),
                      radius * std::sin(angle)
                    };
                    deformation_points_[i].displacement =
                        spatial_vector_t<T, Dims>{0, 0};
                    deformation_points_[i].strain  = 0;
                    deformation_points_[i].yielded = false;
                }
            }
            else if constexpr (Dims == 3) {
                // Create points based on a icosahedron vertices for 3D
                // This is a simplification - would use proper distribution in
                // practice [Code would place vertices approximately uniformly
                // on a sphere]
            }
        }

        // Update the rest shape for plastic deformation
        template <typename Body>
        void update_rest_shape(Body& body)
        {
            // For permanent deformation, we adjust the rest positions
            // of our deformation points

            for (auto& point : deformation_points_) {
                if (point.yielded) {
                    // Update the rest position toward the current deformed
                    // position This is a simplified model - real plasticity is
                    // more complex
                    const T relaxation_factor =
                        0.1;   // How quickly we "remember" deformation
                    point.position += point.displacement * relaxation_factor;

                    // Reset the displacement since we've incorporated it into
                    // the rest shape
                    point.displacement *= (1.0 - relaxation_factor);
                }
            }
        }
    };
}   // namespace simbi::ib
#endif
