#ifndef COMPONENT_BODY_SYSTEM_HPP
#define COMPONENT_BODY_SYSTEM_HPP

#include "body.hpp"
#include "build_options.hpp"
#include "core/types/containers/ndarray.hpp"
#include "core/types/containers/vector.hpp"
#include "core/types/utility/config_dict.hpp"
#include "core/types/utility/enums.hpp"
#include "core/types/utility/managed.hpp"
#include "geometry/mesh/mesh.hpp"
#include "physics/hydro/schemes/ib/serialization/body_serialization.hpp"
#include "physics/hydro/types/generic_structs.hpp"
#include "system_config.hpp"

namespace simbi::ibsystem {
    template <typename T, size_type Dims>
    class ComponentBodySystem : public Managed<global::managed_memory>
    {
      public:
        using MeshType    = Mesh<Dims>;
        using conserved_t = anyConserved<Dims, Regime::NEWTONIAN>;

        // ctor
        ComponentBodySystem(const MeshType& mesh) : mesh_(mesh) {}

        // copy constructor
        ComponentBodySystem(const ComponentBodySystem& other)
            : mesh_(other.mesh_),
              bodies_(other.bodies_),
              grav_body_indices_(other.grav_body_indices_),
              accr_body_indices_(other.accr_body_indices_)
        {
            // deep copy the system config if it exists
            if (other.system_config_) {
                // we need a proper clone mechanism for SystemConfig.
                // for now, assume BinarySystemConfig is the only type and copy
                // it directly
                if (auto binary_config = dynamic_cast<BinarySystemConfig<T>*>(
                        other.system_config_.get()
                    )) {
                    system_config_ =
                        util::make_unique<BinarySystemConfig<T>>(*binary_config
                        );
                }
            }
        }

        // move constructor
        ComponentBodySystem(ComponentBodySystem&& other) noexcept
            : mesh_(other.mesh_),
              bodies_(std::move(other.bodies_)),
              grav_body_indices_(std::move(other.grav_body_indices_)),
              accr_body_indices_(std::move(other.accr_body_indices_)),
              system_config_(std::move(other.system_config_))
        {
        }

        // copy assignment operator
        ComponentBodySystem& operator=(const ComponentBodySystem& other)
        {
            if (this != &other) {
                // we can't change the mesh reference, so we'll assert it's the
                // same.
                assert(
                    &mesh_ == &other.mesh_ &&
                    "Cannot change mesh reference in assignment"
                );

                bodies_            = other.bodies_;
                grav_body_indices_ = other.grav_body_indices_;
                accr_body_indices_ = other.accr_body_indices_;

                // deep copy the system config if it exists
                if (other.system_config_) {
                    if (auto binary_config =
                            dynamic_cast<BinarySystemConfig<T>*>(
                                other.system_config_.get()
                            )) {
                        system_config_ =
                            util::make_unique<BinarySystemConfig<T>>(
                                *binary_config
                            );
                    }
                }
                else {
                    system_config_.reset();
                }
            }
            return *this;
        }

        // move assignment operator
        ComponentBodySystem& operator=(ComponentBodySystem&& other) noexcept
        {
            if (this != &other) {
                // we can't change the mesh reference, so we'll assert it's the
                // same
                assert(
                    &mesh_ == &other.mesh_ &&
                    "Cannot change mesh reference in assignment"
                );

                bodies_            = std::move(other.bodies_);
                grav_body_indices_ = std::move(other.grav_body_indices_);
                accr_body_indices_ = std::move(other.accr_body_indices_);
                system_config_     = std::move(other.system_config_);
            }
            return *this;
        }

        ComponentBodySystem<T, Dims> add_body(const Body<T, Dims>& body) const
        {
            ComponentBodySystem<T, Dims> new_system(*this);

            // add to parallel arrays for GPU compatibility
            new_system.bodies_.push_back_with_sync(body);

            // update specialized arrays for GPU access patterns
            if (body.gravitational.has_value()) {
                new_system.grav_body_indices_.push_back_with_sync(
                    new_system.bodies_.size() - 1
                );
            }

            if (body.accretion.has_value()) {
                new_system.accr_body_indices_.push_back_with_sync(
                    new_system.bodies_.size() - 1
                );
            }

            return new_system;
        }

        // factory method using user config
        ComponentBodySystem<T, Dims> add_body_from_config(
            BodyType type,
            const spatial_vector_t<T, Dims>& position,
            const spatial_vector_t<T, Dims>& velocity,
            const T mass,
            const T radius,
            const ConfigDict& config
        ) const
        {
            // create basic body
            Body<T, Dims> body(type, position, velocity, mass, radius);

            // add capabilities based on config
            if (config.contains("softening_length") ||
                config.contains("two_way_coupling")) {
                T softening =
                    extract_property<T>(config, "softening_length", T(0.01));
                bool two_way =
                    extract_property<bool>(config, "two_way_coupling", false);
                body = body.with_gravitational(softening, two_way);
            }

            if (config.contains("accretion_efficiency") ||
                config.contains("accretion_radius")) {
                T accr_efficiency = extract_property<T>(
                    config,
                    "accretion_efficiency",
                    T(0.01)
                );
                T accr_radius =
                    extract_property<T>(config, "accretion_radius", T(0.0));
                body = body.with_accretion(accr_efficiency, accr_radius);
            }

            // add the body to system
            return add_body(body);
        }

        // immutable access functions
        DUAL const ndarray<Body<T, Dims>>& bodies() const { return bodies_; }
        DUAL size_t size() const { return bodies_.size(); }
        DUAL const MeshType& mesh() const { return mesh_; }

        // query functions
        DUAL Maybe<Body<T, Dims>> get_body(size_t index) const
        {
            if (index >= bodies_.size()) {
                return Nothing;
            }
            return bodies_[index];
        }

        ComponentBodySystem<T, Dims>
        update_body(size_t index, const Body<T, Dims>& updated_body) const
        {
            // return unchanged system if index is out of bounds
            if (index >= bodies_.size()) {
                return *this;
            }

            ComponentBodySystem<T, Dims> new_system(*this);
            new_system.bodies_[index] = updated_body;

            // update specialized arrays if capabilities changed

            return new_system;
        }

        ComponentBodySystem<T, Dims>
        update_body(size_t index, Body<T, Dims>&& updated_body) const
        {
            if (index >= bodies_.size()) {
                return *this;
            }

            ComponentBodySystem<T, Dims> new_system(*this);
            new_system.bodies_[index] = std::move(updated_body);
            return new_system;
        }

        static ComponentBodySystem<T, Dims> update_body_in(
            ComponentBodySystem<T, Dims>&& system,
            size_type index,
            Body<T, Dims>&& updated_body
        )
        {
            if (index >= system.bodies_.size()) {
                return std::move(system);
            }

            system.bodies_[index] = std::move(updated_body);
            return std::move(system);
        }

        // allows for chaining multiple updates
        template <typename UpdateFunc>
        ComponentBodySystem<T, Dims> update_with(UpdateFunc&& func) const
        {
            return std::forward<UpdateFunc>(func)(*this);
        }

        template <typename UpdateFunc>
        static ComponentBodySystem<T, Dims>
        update_with_in(ComponentBodySystem<T, Dims>&& system, UpdateFunc&& func)
        {
            return std::forward<UpdateFunc>(func)(std::move(system));
        }

        // calculate derived properties
        DUAL T total_mass() const
        {
            T total = T(0);
            for (const auto& body : bodies_) {
                total += body.mass;
            }
            return total;
        }

        // system config support
        template <typename ConfigType, typename... Args>
        ComponentBodySystem<T, Dims> with_system_config(Args&&... args) const
        {
            ComponentBodySystem<T, Dims> new_system(*this);
            new_system.system_config_ =
                util::make_unique<ConfigType>(std::forward<Args>(args)...);
            return new_system;
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

        bool inertial() const
        {
            if (system_config_) {
                return dynamic_cast<BinarySystemConfig<T>*>(system_config_.get()
                )
                    ->prescribed_motion;
            }
            return false;
        }

        bool invokes_gravity() const { return !grav_body_indices_.empty(); }
        bool invokes_accretion() const { return !accr_body_indices_.empty(); }

        // property extraction helper
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

        // generate serializable properties for a body
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
            if (body_idx >= bodies_.size()) {
                return properties;
            }

            const auto& body = bodies_[body_idx];

            // Add core properties
            properties.push_back(PropertyDescriptor<T>{"mass", [body](size_t) {
                                                           return body.mass;
                                                       }});

            properties
                .push_back(PropertyDescriptor<T>{"radius", [body](size_t) {
                                                     return body.radius;
                                                 }});

            properties.push_back(
                PropertyDescriptor<spatial_vector_t<T, Dims>>{
                  "position",
                  [body](size_t) { return body.position; }
                }
            );

            properties.push_back(
                PropertyDescriptor<spatial_vector_t<T, Dims>>{
                  "velocity",
                  [body](size_t) { return body.velocity; }
                }
            );

            properties.push_back(
                PropertyDescriptor<spatial_vector_t<T, Dims>>{
                  "force",
                  [body](size_t) { return body.force; }
                }
            );

            // Add component-specific properties
            if (body.gravitational.has_value()) {
                properties.push_back(
                    PropertyDescriptor<T>{
                      "softening_length",
                      [body](size_t) { return body.softening_length(); }
                    }
                );

                properties.push_back(
                    PropertyDescriptor<bool>{
                      "two_way_coupling",
                      [body](size_t) { return body.two_way_coupling(); }
                    }
                );
            }

            if (body.accretion.has_value()) {
                properties.push_back(
                    PropertyDescriptor<T>{
                      "accretion_efficiency",
                      [body](size_t) { return body.accretion_efficiency(); }
                    }
                );

                properties.push_back(
                    PropertyDescriptor<T>{
                      "accretion_radius",
                      [body](size_t) { return body.accretion_radius(); }
                    }
                );

                properties.push_back(
                    PropertyDescriptor<T>{
                      "total_accreted_mass",
                      [body](size_t) { return body.total_accreted_mass(); }
                    }
                );

                properties.push_back(
                    PropertyDescriptor<T>{
                      "accretion_rate",
                      [body](size_t) { return body.accretion_rate(); }
                    }
                );
            }

            return properties;
        }

      private:
        const MeshType& mesh_;

        // primary storage - immutable by default
        ndarray<Body<T, Dims>> bodies_;

        // GPU-optimized lookup arrays
        ndarray<size_t> grav_body_indices_;
        ndarray<size_t> accr_body_indices_;

        // system config - optional
        util::smart_ptr<SystemConfig> system_config_;
    };
};   // namespace simbi::ibsystem
#endif
