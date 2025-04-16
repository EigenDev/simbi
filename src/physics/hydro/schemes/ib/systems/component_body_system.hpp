#ifndef COMPONENT_BODY_SYSTEM_HPP
#define COMPONENT_BODY_SYSTEM_HPP

#include "build_options.hpp"
#include "core/types/containers/ndarray.hpp"
#include "core/types/containers/vector.hpp"
#include "core/types/utility/config_dict.hpp"
#include "core/types/utility/enums.hpp"
#include "core/types/utility/managed.hpp"
#include "geometry/mesh/mesh.hpp"
#include "physics/hydro/types/generic_structs.hpp"

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

      public:
        using MeshType    = Mesh<Dims>;
        using conserved_t = anyConserved<Dims, Regime::NEWTONIAN>;

        // ctor
        ComponentBodySystem(const MeshType& mesh) : mesh_(mesh) {}

        // add a body to the system
        size_type add_body(
            const BodyType type,
            const spatial_vector_t<T, Dims>& position,
            const spatial_vector_t<T, Dims>& velocity,
            const T mass,
            const T radius
        )
        {
            positions_.push_back(position);
            velocities_.push_back(velocity);
            forces_.push_back(spatial_vector_t<T, Dims>());
            masses_.push_back(mass);
            radii_.push_back(radius);
            body_types_.push_back(type);
            capabilities_.push_back(BodyCapability::NONE);

            // return the index of the new body
            return positions_.size() - 1;
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
            if (body_idx >= positions_.size()) {
                throw std::out_of_range("Body index out of range");
            }

            capabilities_[body_idx] |= BodyCapability::GRAVITATIONAL;

            // Add to gravitational properties
            grav_body_indices_.push_back(body_idx);
            grav_softening_lengths_.push_back(softening_length);
            grav_two_way_coupling_.push_back(two_way_coupling);

            // Store mapping from body index to property index
            grav_map_keys_.push_back(body_idx);
            grav_map_values_.push_back(grav_softening_lengths_.size() - 1);
        }

        // add accretion capability to a body
        void add_accretion_capability(
            size_t body_idx,
            T accretion_efficiency = T(0.01),
            T accretion_radius     = T(0.0)   // Default to body radius if 0
        )
        {
            // Ensure body exists
            if (body_idx >= positions_.size()) {
                throw std::out_of_range("Body index out of range");
            }

            // Set capability flag
            capabilities_[body_idx] |= BodyCapability::ACCRETION;

            // If accretion_radius is 0, use body radius
            if (accretion_radius <= 0) {
                accretion_radius = radii_[body_idx];
            }

            // Add to accretion properties
            accr_body_indices_.push_back(body_idx);
            accr_efficiencies_.push_back(accretion_efficiency);
            accr_radii_.push_back(accretion_radius);
            accr_total_masses_.push_back(T(0)
            );   // Initialize accreted mass to 0

            // Store mapping from body index to property index
            accr_map_keys_.push_back(body_idx);
            accr_map_values_.push_back(accr_efficiencies_.size() - 1);
        }

        // Find property index for a body in gravitational maps
        DUAL size_t find_grav_property_index(size_t body_idx) const
        {
            for (size_t i = 0; i < grav_map_keys_.size(); ++i) {
                if (grav_map_keys_[i] == body_idx) {
                    return grav_map_values_[i];
                }
            }
            return size_t(-1);   // Not found
        }

        // Find property index for a body in accretion maps
        DUAL size_t find_accr_property_index(size_t body_idx) const
        {
            for (size_t i = 0; i < accr_map_keys_.size(); ++i) {
                if (accr_map_keys_[i] == body_idx) {
                    return accr_map_values_[i];
                }
            }
            return size_t(-1);   // Not found
        }

        // accesor functions for universal properties
        DUAL const ndarray<spatial_vector_t<T, Dims>>& positions() const
        {
            return positions_;
        }
        DUAL const ndarray<spatial_vector_t<T, Dims>>& velocities() const
        {
            return velocities_;
        }
        DUAL const ndarray<spatial_vector_t<T, Dims>>& forces() const
        {
            return forces_;
        }
        DUAL const ndarray<T>& masses() const { return masses_; }
        DUAL const ndarray<T>& radii() const { return radii_; }
        DUAL const ndarray<BodyType>& body_types() const { return body_types_; }
        DUAL const ndarray<BodyCapability>& capabilities() const
        {
            return capabilities_;
        }

        // Mutable access (for algorithms)
        DUAL ndarray<spatial_vector_t<T, Dims>>& positions_mut()
        {
            return positions_;
        }
        DUAL ndarray<spatial_vector_t<T, Dims>>& velocities_mut()
        {
            return velocities_;
        }
        DUAL ndarray<spatial_vector_t<T, Dims>>& forces_mut()
        {
            return forces_;
        }
        DUAL ndarray<T>& masses_mut() { return masses_; }

        // Size information
        DUAL size_t size() const { return positions_.size(); }

        // Capability checking
        DUAL bool has_capability(size_t body_idx, BodyCapability cap) const
        {
            if (body_idx >= capabilities_.size()) {
                return false;
            }
            return ibsystem::has_capability(capabilities_[body_idx], cap);
        }

        DUAL T total_mass() const
        {
            T total_mass = 0;
            for (size_t i = 0; i < masses_.size(); ++i) {
                total_mass += masses_[i];
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
        ndarray<spatial_vector_t<T, Dims>> positions_;
        ndarray<spatial_vector_t<T, Dims>> velocities_;
        ndarray<spatial_vector_t<T, Dims>> forces_;
        ndarray<T> masses_;
        ndarray<T> radii_;
        ndarray<BodyType> body_types_;
        ndarray<BodyCapability> capabilities_;

        // gravitational properties
        ndarray<size_t> grav_body_indices_;
        ndarray<T> grav_softening_lengths_;
        ndarray<bool> grav_two_way_coupling_;

        // GPU-friendly map replacement for grav_property_map_
        ndarray<size_t> grav_map_keys_;     // body indices
        ndarray<size_t> grav_map_values_;   // property indices

        // accretion properties
        ndarray<size_t> accr_body_indices_;
        ndarray<T> accr_efficiencies_;
        ndarray<T> accr_radii_;
        ndarray<T> accr_total_masses_;

        // GPU-friendly map replacement for accr_property_map_
        ndarray<size_t> accr_map_keys_;     // body indices
        ndarray<size_t> accr_map_values_;   // property indices

        // TODO: add more specialized properties can be added following this
        // pattern
    };

}   // namespace simbi::ibsystem
#endif
