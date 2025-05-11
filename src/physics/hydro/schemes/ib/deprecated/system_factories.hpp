/**
 *=============================================================================
 *           SIMBI - Special Relativistic Magnetohydrodynamics Code
 *=============================================================================
 *
 * @file            system_factories.hpp
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

#ifndef SYSTEM_FACTORIES_HPP
#define SYSTEM_FACTORIES_HPP

#include "body_factory.hpp"         // for BodyFactory
#include "geometry/mesh/mesh.hpp"   // for Mesh
#include "physics/hydro/schemes/ib/bodies/types/any_body.hpp"   // for AnyBody
#include "physics/hydro/schemes/ib/systems/system_traits.hpp"
#include "system_config.hpp"   // for BinaryConfig
#include <memory>

namespace simbi::ibsystem::factory {
    // body pair factory
    template <typename T, size_type Dims>
        requires traits::AtLeastTwoDimensional<Dims>
    class BinaryFactory
    {
      public:
        static ndarray<util::smart_ptr<ib::AnyBody<T, Dims>>> create(
            const Mesh<Dims>& mesh,
            const config::GravitationalConfig<T> grav_config,
            const config::BinaryConfig<T>& binary_config
        )
        {
            ndarray<util::smart_ptr<ib::AnyBody<T, Dims>>> bodies;
            bodies.reserve(2);

            // components
            auto body1 = binary_config.binary_pair.first;
            auto body2 = binary_config.binary_pair.second;
            body1.configure();
            body2.configure();

            T m1      = body1.mass;
            T m2      = body2.mass;
            T radius1 = body1.radius;
            T radius2 = body2.radius;

            // if (m1 + m2 != binary_config.total_mass) {
            //     throw std::runtime_error(
            //         "Individual masses do not sum to total mass"
            //     );
            // }

            traits::BinaryTrait<T, Dims> trait(binary_config);
            auto [pos1, pos2] = trait.initial_positions();
            auto [vel1, vel2] = trait.initial_velocities();

            // create the bodies
            ConfigDict body1_props;
            ConfigDict body2_props;
            ConfigDict orbital_props;

            body1_props["softening_length"] =
                ConfigValue(body1.softening_length);
            body1_props["two_way_coupling"] =
                ConfigValue(body1.two_way_coupling);
            if (body1.is_an_accretor) {
                body1_props["accretion_efficiency"] =
                    ConfigValue(body1.accretion_efficiency);
                body1_props["accretion_radius"] =
                    ConfigValue(body1.accretion_radius);
            }

            body2_props["softening_length"] =
                ConfigValue(body2.softening_length);
            body2_props["two_way_coupling"] =
                ConfigValue(body2.two_way_coupling);
            if (body2.is_an_accretor) {
                body2_props["accretion_efficiency"] =
                    ConfigValue(body2.accretion_efficiency);
                body2_props["accretion_radius"] =
                    ConfigValue(body2.accretion_radius);
            }

            bodies.push_back(
                ib::BodyFactory<T, Dims>::build(
                    body1.body_type,
                    mesh,
                    pos1,
                    vel1,
                    m1,
                    radius1,
                    body1_props
                )
            );
            bodies.push_back(
                ib::BodyFactory<T, Dims>::build(
                    body2.body_type,
                    mesh,
                    pos2,
                    vel2,
                    m2,
                    radius2,
                    body2_props
                )
            );

            return bodies;
        }
    };
}   // namespace simbi::ibsystem::factory
#endif   // SYSTEM_FACTORIES_HPP
