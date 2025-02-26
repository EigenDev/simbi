/**
 *  *=============================================================================
 *  *           SIMBI - Special Relativistic Magnetohydrodynamics Code
 *  *=============================================================================
 *  *
 *  * @file            rigid.hpp
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

#ifndef RIGID_HPP
#define RIGID_HPP

#include "build_options.hpp"
#include "immersed_boundary.hpp"

namespace simbi {
    namespace ib {
        // Rigid body functionality
        template <typename T, size_type Dims, typename MeshType>
        class RigidBody : public ImmersedBody<T, Dims, MeshType>
        {
          protected:
            T density_;
            T drag_coeff_;

          public:
            // Rigid body methods
            DUAL void
            compute_surface_forces(const auto& cell, const auto& dA_normal)
            {
                // Pressure force
                force_ = -dA_normal * pressure_ * cell.volume_fraction;
            }
        };

    }   // namespace ib

}   // namespace simbi

#endif