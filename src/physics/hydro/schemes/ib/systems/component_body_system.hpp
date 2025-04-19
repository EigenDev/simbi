#ifndef COMPONENT_BODY_SYSTEM_HPP
#define COMPONENT_BODY_SYSTEM_HPP

#include "build_options.hpp"
#include "core/types/containers/array.hpp"
#include "core/types/containers/ndarray.hpp"
#include "core/types/containers/vector.hpp"
#include "core/types/utility/config_dict.hpp"
#include "core/types/utility/enums.hpp"
#include "core/types/utility/managed.hpp"
#include "geometry/mesh/mesh.hpp"
#include "physics/hydro/schemes/ib/serialization/body_serialization.hpp"
#include "physics/hydro/types/generic_structs.hpp"

constexpr size_type MAX_BODIES = 3;
namespace simbi::ibsystem {
    // body capabilities as bit flags for efficient querying
    enum class BodyCapability : uint32_t {
        NONE          = 0,
        GRAVITATIONAL = 1 << 0,
        ACCRETION     = 1 << 1,
        ELASTIC       = 1 << 2,
        DEFORMABLE    = 1 << 3,
        RIGID         = 1 << 4,
        // TODO: add more capabilities as needed
    };

    struct SystemConfig {
        virtual ~SystemConfig() = default;
    };

    template <typename T>
    struct BinarySystemConfig : public SystemConfig {
        T semi_major;
        T mass_ratio;
        T eccentricity;
        T orbital_period;
        bool circular_orbit;
        bool prescribed_motion;
        std::pair<size_t, size_t> body_indices;

        BinarySystemConfig(
            T semi_major,
            T mass_ratio,
            T eccentricity,
            T orbital_period,
            bool circular_orbit,
            bool prescribed_motion,
            size_t body1_idx,
            size_t body2_idx
        )
            : semi_major(semi_major),
              mass_ratio(mass_ratio),
              eccentricity(eccentricity),
              orbital_period(orbital_period),
              circular_orbit(circular_orbit),
              prescribed_motion(prescribed_motion),
              body_indices(body1_idx, body2_idx)
        {
        }
    };

    DUAL inline BodyCapability operator|(BodyCapability lhs, BodyCapability rhs)
    {
        return static_cast<BodyCapability>(
            static_cast<uint32_t>(lhs) | static_cast<uint32_t>(rhs)
        );
    }

    DUAL inline BodyCapability&
    operator|=(BodyCapability& lhs, BodyCapability rhs)
    {
        lhs = lhs | rhs;
        return lhs;
    }

    DUAL inline bool has_capability(BodyCapability caps, BodyCapability query)
    {
        return (static_cast<uint32_t>(caps) & static_cast<uint32_t>(query)) !=
               0;
    }

    template <typename T, size_type Dims>
    class ComponentBodySystem : public Managed<global::managed_memory>
    {
      private:
        util::smart_ptr<SystemConfig> system_config_;
        size_type num_bodies_{0};

      public:
        using MeshType    = Mesh<Dims>;
        using conserved_t = anyConserved<Dims, Regime::NEWTONIAN>;

        // ctor
        ComponentBodySystem(const MeshType& mesh) : mesh_(mesh) {}

        template <typename PropertyType>
        ndarray<PropertyDescriptor<PropertyType>>
        get_property_descriptors(BodyCapability capability) const
        {
            ndarray<PropertyDescriptor<PropertyType>> descriptors;
            if (has_capability(capability, BodyCapability::GRAVITATIONAL)) {
                descriptors.push_back(
                    PropertyDescriptor<PropertyType>{
                      "softening_length",
                      [this](size_t body_idx) {
                          return softening_length(body_idx);
                      },
                      {{"description", "Softening length for gravity"}}
                    }
                );
                descriptors.push_back(
                    PropertyDescriptor<PropertyType>{
                      "two_way_coupling",
                      [this](size_t body_idx) {
                          return two_way_coupling(body_idx);
                      },
                      {{"description", "Two-way coupling flag"}}
                    }
                );
            }
            if (has_capability(capability, BodyCapability::ACCRETION)) {
                descriptors.push_back(
                    PropertyDescriptor<PropertyType>{
                      "accretion_efficiency",
                      [this](size_t body_idx) {
                          return accretion_efficiency(body_idx);
                      },
                      {{"description", "Accretion efficiency"}}
                    }
                );
                descriptors.push_back(
                    PropertyDescriptor<PropertyType>{
                      "accretion_radius",
                      [this](size_t body_idx) {
                          return accretion_radius(body_idx);
                      },
                      {{"description", "Accretion radius"}}
                    }
                );
            }
            return descriptors;
        }

        // method to generate all serializable properties for a body
        ndarray<std::variant<
            PropertyDescriptor<T>,
            PropertyDescriptor<bool>,
            PropertyDescriptor<std::string>,
            PropertyDescriptor<spatial_vector_t<T, Dims>>>>
        get_serializable_properties(size_t body_idx) const
        {
            using PropertyVariant = std::variant<
                PropertyDescriptor<T>,
                PropertyDescriptor<bool>,
                PropertyDescriptor<std::string>,
                PropertyDescriptor<spatial_vector_t<T, Dims>>>;

            ndarray<PropertyVariant> properties;

            // Universal properties for all bodies
            properties.push_back(
                PropertyDescriptor<T>{
                  "mass",
                  [this, body_idx](size_t) { return masses_[body_idx]; }
                }
            );

            properties.push_back(
                PropertyDescriptor<T>{
                  "radius",
                  [this, body_idx](size_t) { return radii_[body_idx]; }
                }
            );

            properties.push_back(
                PropertyDescriptor<spatial_vector_t<T, Dims>>{
                  "position",
                  [this, body_idx](size_t) { return positions_[body_idx]; }
                }
            );

            properties.push_back(
                PropertyDescriptor<spatial_vector_t<T, Dims>>{
                  "velocity",
                  [this, body_idx](size_t) { return velocities_[body_idx]; }
                }
            );

            properties.push_back(
                PropertyDescriptor<spatial_vector_t<T, Dims>>{
                  "force",
                  [this, body_idx](size_t) { return forces_[body_idx]; }
                }
            );

            // Add capability-specific properties
            if (has_capability(body_idx, BodyCapability::GRAVITATIONAL)) {
                properties.push_back(
                    PropertyDescriptor<T>{
                      "softening_length",
                      [this, body_idx](size_t) {
                          return softening_length(body_idx);
                      }
                    }
                );

                properties.push_back(
                    PropertyDescriptor<bool>{
                      "two_way_coupling",
                      [this, body_idx](size_t) {
                          return two_way_coupling(body_idx);
                      }
                    }
                );
            }

            if (has_capability(body_idx, BodyCapability::ACCRETION)) {
                properties.push_back(
                    PropertyDescriptor<T>{
                      "accretion_efficiency",
                      [this, body_idx](size_t) {
                          return accretion_efficiency(body_idx);
                      }
                    }
                );

                properties.push_back(
                    PropertyDescriptor<T>{
                      "accretion_radius",
                      [this, body_idx](size_t) {
                          return accretion_radius(body_idx);
                      }
                    }
                );

                properties.push_back(
                    PropertyDescriptor<T>{
                      "total_accreted_mass",
                      [this, body_idx](size_t) {
                          return total_accreted_mass(body_idx);
                      }
                    }
                );
            }

            return properties;
        }

        // add a body to the system
        size_type add_body(
            const BodyType type,
            const spatial_vector_t<T, Dims>& position,
            const spatial_vector_t<T, Dims>& velocity,
            const T mass,
            const T radius
        )
        {
            // check if the body type is valid
            if (num_bodies_ > MAX_BODIES) {
                throw std::runtime_error("Maximum number of bodies exceeded");
            }
            positions_[num_bodies_]    = position;
            velocities_[num_bodies_]   = velocity;
            forces_[num_bodies_]       = spatial_vector_t<T, Dims>();
            masses_[num_bodies_]       = mass;
            radii_[num_bodies_]        = radius;
            body_types_[num_bodies_]   = type;
            capabilities_[num_bodies_] = BodyCapability::NONE;
            positions_[num_bodies_]    = position;

            num_bodies_++;

            // return the index of the new body
            return num_bodies_ - 1;
        }

        size_type add_body_from_config(
            BodyType type,
            const spatial_vector_t<T, Dims>& position,
            const spatial_vector_t<T, Dims>& velocity,
            const T mass,
            const T radius,
            const ConfigDict& config
        )
        {
            // create the bsic body first
            size_type body_idx =
                add_body(type, position, velocity, mass, radius);

            // check for gravitational properties
            if (config.contains("softening_length") ||
                config.contains("two_way_coupling")) {
                // extract gravitational properties with defaults
                T softening =
                    extract_property<T>(config, "softening_length", T(0.01));
                // bool two_way_coupling =
                // extract_property<bool>(config, "two_way_coupling", false);

                // add gravitational capability
                add_gravitational_capability(body_idx, softening);
            }

            // check for accretion properties
            if (config.contains("accretion_efficiency") ||
                config.contains("accretion_radius")) {
                // extract accretion properties with defaults
                T accretion_efficiency = extract_property<T>(
                    config,
                    "accretion_efficiency",
                    T(0.01)
                );
                T accretion_radius =
                    extract_property<T>(config, "accretion_radius", T(0.0));

                // add accretion capability
                add_accretion_capability(
                    body_idx,
                    accretion_efficiency,
                    accretion_radius
                );
            }

            // TODO: add more properties as needed
            return body_idx;
        }

        // add gravitational capability to a body
        void add_gravitational_capability(
            size_type body_idx,
            T softening_length,
            bool two_way_coupling = false
        )
        {
            if (body_idx >= num_bodies_) {
                throw std::out_of_range("Body index out of range");
            }

            capabilities_[body_idx] |= BodyCapability::GRAVITATIONAL;

            // Add to gravitational properties
            grav_body_indices_[grav_count_]      = body_idx;
            grav_softening_lengths_[grav_count_] = softening_length;
            grav_two_way_coupling_[grav_count_]  = two_way_coupling;

            // Store mapping from body index to property index
            grav_map_keys_[grav_count_]   = body_idx;
            grav_map_values_[grav_count_] = grav_count_;

            grav_count_++;
        }

        // add accretion capability to a body
        void add_accretion_capability(
            size_t body_idx,
            T accretion_efficiency = T(0.01),
            T accretion_radius     = T(0.0)   // Default to body radius if 0
        )
        {
            // Ensure body exists
            if (body_idx >= num_bodies_) {
                throw std::out_of_range("Body index out of range");
            }

            // Set capability flag
            capabilities_[body_idx] |= BodyCapability::ACCRETION;

            // If accretion_radius is 0, use body radius
            if (accretion_radius <= 0) {
                accretion_radius = radii_[body_idx];
            }

            // Add to accretion properties
            accr_body_indices_[accr_count_] = body_idx;
            accr_efficiencies_[accr_count_] = accretion_efficiency;
            accr_radii_[accr_count_]        = accretion_radius;
            accr_total_masses_[accr_count_] = T(0);

            // Store mapping from body index to property index
            accr_map_keys_[accr_count_]   = body_idx;
            accr_map_values_[accr_count_] = accr_count_;
            accr_count_++;
        }

        // Find property index for a body in gravitational maps
        DUAL size_t find_grav_property_index(size_t body_idx) const
        {
            for (size_t ii = 0; ii < grav_count_; ++ii) {
                if (grav_map_keys_[ii] == body_idx) {
                    return grav_map_values_[ii];
                }
            }
            return size_t(-1);   // Not found
        }

        // Find property index for a body in accretion maps
        DUAL size_t find_accr_property_index(size_t body_idx) const
        {
            for (size_t ii = 0; ii < accr_count_; ++ii) {
                if (accr_map_keys_[ii] == body_idx) {
                    return accr_map_values_[ii];
                }
            }
            return size_t(-1);   // Not found
        }

        // accesor functions for universal properties
        DUAL const auto& positions() const { return positions_; }
        DUAL const auto& velocities() const { return velocities_; }
        DUAL const auto& forces() const { return forces_; }
        DUAL const auto& masses() const { return masses_; }
        DUAL const auto& radii() const { return radii_; }
        DUAL const auto& body_types() const { return body_types_; }
        DUAL const auto& capabilities() const { return capabilities_; }

        // fine-grained access to properties
        DUAL const auto& position_at(size_t idx) const
        {
            return positions_[idx];
        }
        DUAL const auto& velocity_at(size_t idx) const
        {
            return velocities_[idx];
        }
        DUAL const auto& force_at(size_t idx) const { return forces_[idx]; }
        DUAL T mass_at(size_t idx) const { return masses_[idx]; }
        DUAL T radius_at(size_t idx) const { return radii_[idx]; }
        DUAL BodyType body_type_at(size_t idx) const
        {
            return body_types_[idx];
        }
        DUAL BodyCapability capability_at(size_t idx) const
        {
            return capabilities_[idx];
        }

        // Mutable access (for algorithms)
        DUAL auto& positions_mut() { return positions_; }
        DUAL auto& velocities_mut() { return velocities_; }
        DUAL auto& forces_mut() { return forces_; }
        DUAL auto& masses_mut() { return masses_; }

        // Size information
        DUAL size_t size() const { return num_bodies_; }

        // Capability checking
        DUAL bool has_capability(size_t body_idx, BodyCapability cap) const
        {
            if (body_idx >= num_bodies_) {
                return false;
            }
            return ibsystem::has_capability(capabilities_[body_idx], cap);
        }

        DUAL T total_mass() const
        {
            T total_mass = 0;
            for (size_t ii = 0; ii < num_bodies_; ++ii) {
                total_mass += masses_[ii];
            }
            return total_mass;
        }

        // specialized property access

        // grav properties
        DUAL T softening_length(size_t body_idx) const
        {
            size_t prop_idx = find_grav_property_index(body_idx);
            if (prop_idx == size_t(-1)) {
                return T(0);
            }
            return grav_softening_lengths_[prop_idx];
        }

        DUAL bool two_way_coupling(size_t body_idx) const
        {
            size_t prop_idx = find_grav_property_index(body_idx);
            if (prop_idx == size_t(-1)) {
                return false;
            }
            return grav_two_way_coupling_[prop_idx];
        }

        // accretion properties
        DUAL T accretion_efficiency(size_t body_idx) const
        {
            size_t prop_idx = find_accr_property_index(body_idx);
            if (prop_idx == size_t(-1)) {
                return T(0);
            }
            return accr_efficiencies_[prop_idx];
        }

        DUAL T accretion_radius(size_t body_idx) const
        {
            size_t prop_idx = find_accr_property_index(body_idx);
            if (prop_idx == size_t(-1)) {
                return radii_[body_idx];
            }
            return accr_radii_[prop_idx];
        }

        DUAL T total_accreted_mass(size_t body_idx) const
        {
            size_t prop_idx = find_accr_property_index(body_idx);
            if (prop_idx == size_t(-1)) {
                return T(0);
            }
            return accr_total_masses_[prop_idx];
        }

        void add_accreted_mass(size_t body_idx, T mass)
        {
            size_t prop_idx = find_accr_property_index(body_idx);
            if (prop_idx != size_t(-1)) {
                accr_total_masses_[prop_idx] += mass;
            }
        }

        // the mesh
        const MeshType& mesh() const { return mesh_; }

        // property extraction
        template <typename U>
        U extract_property(
            const ConfigDict& config,
            const std::string& key,
            const U& default_value
        ) const
        {
            auto it = config.find(key);
            if (it != config.end()) {
                return it->second.get<U>();
            }
            return default_value;
        }

        template <typename ConfigType, typename... Args>
        void set_system_config(Args&&... args)
        {
            system_config_ =
                util::make_unique<ConfigType>(std::forward<Args>(args)...);
        }

        template <typename ConfigType>
        const ConfigType* get_system_config() const
        {
            return dynamic_cast<const ConfigType*>(system_config_.get());
        }

        bool has_system_config() const { return system_config_ != nullptr; }

        bool is_binary() const
        {
            if (system_config_) {
                return dynamic_cast<BinarySystemConfig<T>*>(system_config_.get()
                       ) != nullptr;
            }
            return false;
        }

      private:
        // const reference to the mesh to prevent copying
        const MeshType& mesh_;

        // universal properties (all bodies have these)
        // TODO: replace with a more efficient data structure
        array_t<spatial_vector_t<T, Dims>, MAX_BODIES> positions_;
        array_t<spatial_vector_t<T, Dims>, MAX_BODIES> velocities_;
        array_t<spatial_vector_t<T, Dims>, MAX_BODIES> forces_;
        array_t<T, MAX_BODIES> masses_;
        array_t<T, MAX_BODIES> radii_;
        array_t<BodyType, MAX_BODIES> body_types_;
        array_t<BodyCapability, MAX_BODIES> capabilities_;

        // gravitational properties
        array_t<size_type, MAX_BODIES> grav_body_indices_;
        array_t<T, MAX_BODIES> grav_softening_lengths_;
        array_t<bool, MAX_BODIES> grav_two_way_coupling_;
        size_type grav_count_{0};

        // GPU-friendly map replacement for grav_property_map_
        array_t<size_type, MAX_BODIES> grav_map_keys_;     // body indices
        array_t<size_type, MAX_BODIES> grav_map_values_;   // property indices

        // accretion properties
        array_t<size_type, MAX_BODIES> accr_body_indices_;
        array_t<T, MAX_BODIES> accr_efficiencies_;
        array_t<T, MAX_BODIES> accr_radii_;
        array_t<T, MAX_BODIES> accr_total_masses_;
        size_type accr_count_{0};

        // GPU-friendly map replacement for accr_property_map_
        array_t<size_t, MAX_BODIES> accr_map_keys_;     // body indices
        array_t<size_t, MAX_BODIES> accr_map_values_;   // property indices

        // TODO: add more specialized properties can be added following this
        // pattern
    };

}   // namespace simbi::ibsystem
#endif
