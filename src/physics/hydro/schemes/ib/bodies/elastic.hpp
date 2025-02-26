/**
 *  *=============================================================================
 *  *           SIMBI - Special Relativistic Magnetohydrodynamics Code
 *  *=============================================================================
 *  *
 *  * @file            elastic.hpp
 *  * @brief           Elastic Immersed Body Implementation
 *  * @details
 *  *
 *  * @version         0.8.0
 *  * @date            2025-02-16
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
 *  * 2025-02-16      v0.8.0      Initial implementation
 *  *
 *  *==============================================================================
 *  * @copyright (C) 2025 Marcus DuPont. All rights reserved.
 *  *==============================================================================
 */

#ifndef ELASTIC_HPP
#define ELASTIC_HPP

#include "build_options.hpp"
#include "immersed_boundary.hpp"

namespace simbi {
    namespace ib {
        // Elastic body functionality
        template <typename T, size_type Dims, typename MeshType>
        class ElasticBody : public ImmersedBody<T, Dims, MeshType>
        {
          protected:
            T stiffness_;
            T damping_;
            T pressure_;

          public:
            // Elastic body methods
            DUAL auto
            compute_surface_forces(const auto& cell, const auto& dA_normal)
            {
                interpolate_fluid_velocity();
                // Compute relative velocity
                const auto v_rel = this->velocity_ - this->fluid_velocity_;

                // Compute elastic force (Hooke's Law)
                this->force_ = -dA_normal * stiffness_ * cell.distance;

                // Add damping
                this->force_ += -dA_normal * damping_ * v_rel.dot(cell.normal);
            }
        };

    }   // namespace ib

}   // namespace simbi

#endif