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
        static std::vector<std::unique_ptr<ib::AnyBody<T, Dims>>> create(
            const Mesh<Dims>& mesh,
            const config::GravitationalConfig<T> grav_config,
            const config::BinaryConfig<T>& binary_config
        )
        {
            std::vector<std::unique_ptr<ib::AnyBody<T, Dims>>> bodies;
            bodies.reserve(2);

            // the masses
            T total_mass = binary_config.total_mass;
            T m1         = total_mass / (T(1) + binary_config.mass_ratio);
            T m2         = total_mass - m1;
            // we'll set some arbitrary defaults for now.
            // TODO: make these binary_configurable
            T radius1 = binary_config.semi_major * T(0.01);
            T radius2 = radius1;

            traits::BinaryTrait<T, Dims> trait(binary_config);
            auto [pos1, pos2] = trait.initial_positions();
            auto [vel1, vel2] = trait.initial_velocities();

            // create the bodies
            ConfigDict body1_props;
            ConfigDict body2_props;

            if (grav_config.prescribed_motion) {
                ConfigDict orbital_props;
                orbital_props["semi_major_axis"] =
                    ConfigValue(binary_config.semi_major);
                orbital_props["eccentricity"] =
                    ConfigValue(binary_config.eccentricity);
                orbital_props["period"] = ConfigValue(trait.orbital_period());
                orbital_props["argument_of_periapsis"] = ConfigValue(T(0));
                orbital_props["primary_mass"]          = ConfigValue(m2);
                orbital_props["system_mass"]    = ConfigValue(total_mass);
                body1_props["orbital_elements"] = ConfigValue(orbital_props);
                body2_props["orbital_elements"] = ConfigValue(orbital_props);
            }
            else {
                throw std::runtime_error("Live motion not implemented yet");
            }

            bodies.push_back(ib::BodyFactory<T, Dims>::build(
                BodyType::GRAVITATIONAL,
                mesh,
                pos1,
                vel1,
                m1,
                radius1,
                body1_props
            ));
            bodies.push_back(ib::BodyFactory<T, Dims>::build(
                BodyType::GRAVITATIONAL,
                mesh,
                pos2,
                vel2,
                m2,
                radius2,
                body2_props
            ));

            return bodies;
        }
    };
}   // namespace simbi::ibsystem::factory
#endif   // SYSTEM_FACTORIES_HPP
