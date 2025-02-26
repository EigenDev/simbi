/**
 *  *=============================================================================
 *  *           SIMBI - Special Relativistic Magnetohydrodynamics Code
 *  *=============================================================================
 *  *
 *  * @file            viscuous.hpp
 *  * @brief
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

#ifndef VISCOUS_HPP
#define VISCOUS_HPP

#include "build_options.hpp"
#include "immersed_boundary.hpp"

namespace simbi {
    namespace ib {
        // Viscous body functionality
        template <typename T, size_type Dims, typename MeshType>
        class ViscousBody : public ImmersedBody<T, Dims, MeshType>
        {
          protected:
            T viscosity_;
            T drag_coeff_;

          public:
            // Viscous body methods
            DUAL auto
            compute_surface_forces(const auto& cell, const auto& dA_normal)
            {
                // Compute relative velocity
                const auto v_rel = this->velocity_ - this->fluid_velocity_;

                // Compute viscous force
                this->force_ = -viscosity_ * da_normal.norm() * v_rel;
            }
        };
    }   // namespace ib

}   // namespace simbi

#endif