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
