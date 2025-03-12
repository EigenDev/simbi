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

#include "../bodies/elastic.hpp"                    // for ElasticBody
#include "../bodies/gravitational.hpp"              // for GravitationalBody
#include "../bodies/rigid.hpp"                      // for RigidBody
#include "../bodies/sink.hpp"                       // for SinkBody
#include "../bodies/source.hpp"                     // for SourceBody
#include "../bodies/viscous.hpp"                    // for ViscuousBody
#include "core/types/containers/vector.hpp"         // for spatial_vector_t
#include "core/types/utility/init_conditions.hpp"   // for InitialConditions
#include <string>                                   // for std::string
#include <unordered_map>                            // for std::unordered_map

namespace simbi::ib {
    template <typename T, size_t Dims, typename MeshType>
    class BodyFactory
    {
      public:
        static std::unique_ptr<ImmersedBody<T, Dims, MeshType>> create_body(
            BodyType type,
            const MeshType& mesh,
            const spatial_vector_t<T, Dims>& pos,
            const spatial_vector_t<T, Dims>& vel,
            const T mass,
            const T radius,
            const std::unordered_map<
                std::string,
                InitialConditions::PropertyValue>& properties
        )
        {
            // Helper to get scalar value
            auto get_scalar = [](const auto& prop_value) -> T {
                return std::get<T>(prop_value);
            };

            switch (type) {
                case BodyType::GRAVITATIONAL:
                    return std::unique_ptr<ImmersedBody<T, Dims, MeshType>>(
                        new GravitationalBody<T, Dims, MeshType>(
                            mesh,
                            pos,
                            vel,
                            mass,
                            radius,
                            get_scalar(properties.at("grav_strength")),
                            get_scalar(properties.at("softening"))
                        )
                    );

                case BodyType::GRAVITATIONAL_SINK:
                    return std::unique_ptr<ImmersedBody<T, Dims, MeshType>>(
                        new GravitationalSinkParticle<T, Dims, MeshType>(
                            mesh,
                            pos,
                            vel,
                            mass,
                            radius,
                            get_scalar(properties.at("grav_strength")),
                            get_scalar(properties.at("softening")),
                            get_scalar(properties.at("accretion_efficiency"))
                        )
                    );
                default: throw std::runtime_error("Invalid body type");
            }
        }
    };
}   // namespace simbi::ib
#endif