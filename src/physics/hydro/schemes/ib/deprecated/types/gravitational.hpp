/**
 *=============================================================================
 *           SIMBI - Special Relativistic Magnetohydrodynamics Code
 *=============================================================================
 *
 * @file            gravitational.hpp
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

#ifndef GRAVITATIONAL_HPP
#define GRAVITATIONAL_HPP

#include "../immersed_boundary.hpp"
#include "physics/hydro/schemes/ib/policies/force_policies.hpp"   // for ElasticForcePolicy, GravitationalForcePolicy
#include "physics/hydro/schemes/ib/policies/interaction_policies.hpp"   // for AccretingFluidInteractionPolicy, MinimalFluidInteractionPolicy, StandardFluidInteractionPolicy
#include "physics/hydro/schemes/ib/policies/material_policies.hpp"   // for DeformableMaterialPolicy, RigidMaterialPolicy
#include "physics/hydro/schemes/ib/policies/motion_policies.hpp"   // for DynamicMotionPolicy

namespace simbi::ib {
    template <typename T, size_t Dims>
    using GravitationalBody = ImmersedBody<
        T,
        Dims,
        GravitationalForcePolicy,
        RigidMaterialPolicy,
        GravitationalFluidInteractionPolicy,
        DynamicMotionPolicy>;

    template <typename T, size_t Dims>
    using GravitationalSinkBody = ImmersedBody<
        T,
        Dims,
        GravitationalForcePolicy,
        RigidMaterialPolicy,
        AccretingFluidInteractionPolicy,
        DynamicMotionPolicy>;

}   // namespace simbi::ib

#endif
