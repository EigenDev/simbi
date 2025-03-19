/**
 *  *=============================================================================
 *  *           SIMBI - Special Relativistic Magnetohydrodynamics Code
 *  *=============================================================================
 *  *
 *  * @file            body_factory.hpp
 *  * @brief           A generic factory system for creating and managing bodies
 *  * @details
 *  *
 *  * @version         0.8.0
 *  * @date            2025-03-07
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
 *  * 2025-03-07      v0.8.0      Initial implementation
 *  *
 *  *==============================================================================
 *  * @copyright (C) 2025 Marcus DuPont. All rights reserved.
 *  *==============================================================================
 */

#ifndef BODY_FACTORY_HPP
#define BODY_FACTORY_HPP

#include "../bodies/types/gravitational.hpp"   // for GravitationalBody, GravitationalSinkBody
#include "core/types/containers/vector.hpp"     // for spatial_vector_t
#include "core/types/utility/config_dict.hpp"   // for ConfigDict
#include "core/types/utility/enums.hpp"         // for BodyType
#include "physics/hydro/schemes/ib/bodies/immersed_boundary.hpp"   // for ImmersedBody
#include "physics/hydro/schemes/ib/bodies/policies/force_policies.hpp"
#include "physics/hydro/schemes/ib/bodies/policies/motion_policies.hpp"
#include "physics/hydro/schemes/ib/bodies/types/any_body.hpp"

namespace simbi {
    template <size_type Dims>
    class Mesh;
}

namespace simbi::ib {
    template <typename T, size_type Dims>
    class BodyFactory
    {
      public:
        using MeshType = Mesh<Dims>;
        // Main build method - constructs a body of the specific type
        static std::unique_ptr<AnyBody<T, Dims>> build(
            BodyType type,
            const MeshType& mesh,
            const spatial_vector_t<T, Dims>& position,
            const spatial_vector_t<T, Dims>& velocity,
            T mass,
            T radius,
            const ConfigDict& props
        )
        {
            switch (type) {
                case BodyType::GRAVITATIONAL:
                    return build_gravitational_body(
                        mesh,
                        position,
                        velocity,
                        mass,
                        radius,
                        props
                    );
                // case BodyType::ELASTIC:
                //     return build_elastic_body(
                //         mesh,
                //         position,
                //         velocity,
                //         mass,
                //         radius,
                //         props
                //     );
                // case BodyType::RIGID:
                //     return build_rigid_body(
                //         mesh,
                //         position,
                //         velocity,
                //         mass,
                //         radius,
                //         props
                //     );
                case BodyType::GRAVITATIONAL_SINK:
                    return build_gravitational_sink_body(
                        mesh,
                        position,
                        velocity,
                        mass,
                        radius,
                        props
                    );
                // Other body types...
                default:
                    throw std::runtime_error(
                        "Unknown body type: " +
                        std::to_string(static_cast<int>(type))
                    );
            }
        }

      private:
        // Extract a property with a default value
        template <typename V>
        static V extract_property(
            const ConfigDict& props,
            const std::string& name,
            V default_value
        )
        {
            // extract the property from the config dictionary
            auto it = props.find(name);
            if (it != props.end()) {
                return it->second.get<V>();
            }
            // if not found, return the default value
            return default_value;
        }

        // Build gravitational force policy parameters
        static GravitationalForcePolicy<T, Dims>::Params
        build_grav_force_params(const ConfigDict& props)
        {
            typename GravitationalForcePolicy<T, Dims>::Params params;
            params.softening_length =
                extract_property<T>(props, "softening_length", T(0.01));
            params.two_way_coupling =
                extract_property<bool>(props, "two_way_coupling", false);
            return params;
        }

        // Build standard fluid interaction policy parameters
        static GravitationalFluidInteractionPolicy<T, Dims>::Params
        build_grav_fluid_params(const ConfigDict& props)
        {
            typename GravitationalFluidInteractionPolicy<T, Dims>::Params
                params;
            // params.interaction_strength = extract_property<T>(
            //     props,
            //     "fluid_interaction_strength",
            //     T(1.0)
            // );
            return params;
        }

        // Build accreting fluid interaction policy parameters
        static AccretingFluidInteractionPolicy<T, Dims>::Params
        build_accretion_fluid_params(const ConfigDict& props)
        {
            typename AccretingFluidInteractionPolicy<T, Dims>::Params params;
            params.accretion_params.accretion_efficiency =
                extract_property<T>(props, "accretion_efficiency", T(0.01));
            params.accretion_params.accretion_radius_factor =
                extract_property<T>(props, "accretion_radius_factor", T(1.0));
            params.grav_params.softening_length =
                extract_property<T>(props, "softening_length", T(0.01));
            params.grav_params.two_way_coupling =
                extract_property<bool>(props, "two_way_coupling", false);
            return params;
        }

        // Build rigid material policy parameters
        static RigidMaterialPolicy<T, Dims>::Params
        build_rigid_material_params(const ConfigDict& props)
        {
            typename RigidMaterialPolicy<T, Dims>::Params params;
            params.density = extract_property<T>(props, "density", T(1.0));
            params.restitution_coefficient =
                extract_property<T>(props, "restitution", T(0.8));
            params.infinitely_rigid =
                extract_property<bool>(props, "infinitely_rigid", true);
            return params;
        }

        // Build motion policy parameters
        static DynamicMotionPolicy<T, Dims>::Params
        build_dynamic_motion_params(const ConfigDict& props)
        {
            typename DynamicMotionPolicy<T, Dims>::Params params;
            params.live_motion =
                extract_property<bool>(props, "live_motion", true);
            return params;
        }

        static StaticMotionPolicy<T, Dims>::Params
        build_static_motion_params(const ConfigDict& props)
        {
            typename StaticMotionPolicy<T, Dims>::Params params;
            return params;
        }

        // Build a gravitational body
        static std::unique_ptr<AnyBody<T, Dims>> build_gravitational_body(
            const MeshType& mesh,
            const spatial_vector_t<T, Dims>& position,
            const spatial_vector_t<T, Dims>& velocity,
            T mass,
            T radius,
            const ConfigDict& props
        )
        {
            auto grav_params     = build_grav_force_params(props);
            auto fluid_params    = build_grav_fluid_params(props);
            auto material_params = build_rigid_material_params(props);
            auto motion_params   = build_dynamic_motion_params(props);

            return std::make_unique<AnyBody<T, Dims>>(
                std::in_place_type<GravitationalBody<T, Dims>>,
                mesh,
                position,
                velocity,
                mass,
                radius,
                grav_params,
                material_params,
                fluid_params,
                motion_params
            );
        }

        // Build a gravitational sink body (combines gravitational and
        // accretion)
        static std::unique_ptr<AnyBody<T, Dims>> build_gravitational_sink_body(
            const MeshType& mesh,
            const spatial_vector_t<T, Dims>& position,
            const spatial_vector_t<T, Dims>& velocity,
            T mass,
            T radius,
            const ConfigDict& props
        )
        {
            auto grav_params     = build_grav_force_params(props);
            auto fluid_params    = build_accretion_fluid_params(props);
            auto material_params = build_rigid_material_params(props);
            auto motion_params   = build_dynamic_motion_params(props);

            return std::make_unique<AnyBody<T, Dims>>(
                std::in_place_type<GravitationalSinkBody<T, Dims>>,
                mesh,
                position,
                velocity,
                mass,
                radius,
                grav_params,
                material_params,
                fluid_params,
                motion_params
            );
        }

        static std::unique_ptr<AnyBody<T, Dims>> build_elastic_body(...)
        {
            // TODO: Implement elastic body construction
        }

        static std::unique_ptr<AnyBody<T, Dims>> build_rigid_body(...)
        {
            // TODO: Implement rigid body construction
        }
    };
}   // namespace simbi::ib
#endif
