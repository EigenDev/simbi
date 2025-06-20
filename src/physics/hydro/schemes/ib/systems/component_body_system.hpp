/**
 *=============================================================================
 *           SIMBI - Special Relativistic Magnetohydrodynamics Code
 *=============================================================================
 *
 * @file            component_body_system.hpp
 * @brief           ComponentBodySystem class for the IB scheme
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
#ifndef COMPONENT_BODY_SYSTEM_HPP
#define COMPONENT_BODY_SYSTEM_HPP

#include "body.hpp"
#include "config.hpp"
#include "core/containers/ndarray.hpp"
#include "core/containers/vector.hpp"
#include "core/utility/config_dict.hpp"
#include "core/utility/enums.hpp"
#include "core/utility/managed.hpp"
#include "geometry/mesh/mesh.hpp"
#include "physics/hydro/schemes/ib/serialization/body_serialization.hpp"
#include "physics/hydro/types/generic_structs.hpp"
#include "system_config.hpp"

namespace simbi::ibsystem {
    using namespace containers;
    template <typename T, size_type Dims>
    class ComponentBodySystem : public Managed<global::managed_memory>
    {
      public:
        using MeshType    = Mesh<Dims>;
        using conserved_t = anyConserved<Dims, Regime::NEWTONIAN>;

        // ctor
        ComponentBodySystem(
            const MeshType& mesh,
            std::string reference_frame = "inertial"
        )
            : mesh_(mesh), reference_frame_(reference_frame)
        {
        }

        // copy constructor
        ComponentBodySystem(const ComponentBodySystem& other)
            : mesh_(other.mesh_),
              bodies_(other.bodies_),
              grav_body_indices_(other.grav_body_indices_),
              accr_body_indices_(other.accr_body_indices_),
              reference_frame_(other.reference_frame_)
        {
            // deep copy the system config if it exists
            if (other.system_config_) {
                // we need a proper clone mechanism for SystemConfig.
                // for now, assume BinarySystemConfig is the only type and copy
                // it directly
                if (auto binary_config = dynamic_cast<BinarySystemConfig<T>*>(
                        other.system_config_.get()
                    )) {
                    system_config_ = util::make_unique<BinarySystemConfig<T>>(
                        *binary_config
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
              system_config_(std::move(other.system_config_)),
              reference_frame_(std::move(other.reference_frame_))
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
                reference_frame_ = other.reference_frame_;
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
                reference_frame_   = std::move(other.reference_frame_);
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

        // immutable access functions
        DUAL const ndarray_t<Body<T, Dims>>& bodies() const { return bodies_; }
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
        auto system_config() const { return system_config_; }

        bool is_binary() const
        {
            if (system_config_) {
                return dynamic_cast<BinarySystemConfig<T>*>(
                           system_config_.get()
                       ) != nullptr;
            }
            return false;
        }

        bool inertial() const { return reference_frame_ == "inertial"; }

        bool invokes_gravity() const { return !grav_body_indices_.empty(); }
        bool invokes_accretion() const { return !accr_body_indices_.empty(); }

        std::string reference_frame() const { return reference_frame_; }

        // generate serializable properties for a body
        ndarray_t<std::variant<
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

            ndarray_t<PropertyVariant> properties;
            if (body_idx >= bodies_.size()) {
                return properties;
            }

            const auto& body = bodies_[body_idx];

            // Add core properties
            properties.push_back(PropertyDescriptor<T>{"mass", [body](size_t) {
                                                           return body.mass;
                                                       }});

            properties.push_back(
                PropertyDescriptor<T>{"radius", [body](size_t) {
                                          return body.radius;
                                      }}
            );

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
                    PropertyDescriptor<T>{"softening_length", [body](size_t) {
                                              return body.softening_length();
                                          }}
                );

                properties.push_back(
                    PropertyDescriptor<bool>{
                      "two_way_coupling",
                      [body](size_t) { return body.two_way_coupling; }
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
                    PropertyDescriptor<T>{"accretion_radius", [body](size_t) {
                                              return body.accretion_radius();
                                          }}
                );

                properties.push_back(
                    PropertyDescriptor<T>{
                      "total_accreted_mass",
                      [body](size_t) { return body.total_accreted_mass(); }
                    }
                );

                properties.push_back(
                    PropertyDescriptor<T>{"accretion_rate", [body](size_t) {
                                              return body.accretion_rate();
                                          }}
                );
            }

            if (body.rigid.has_value()) {
                properties.push_back(
                    PropertyDescriptor<T>{"inertia", [body](size_t) {
                                              return body.inertia();
                                          }}
                );
                properties.push_back(
                    PropertyDescriptor<bool>{"apply_no_slip", [body](size_t) {
                                                 return body.apply_no_slip();
                                             }}
                );
            }

            if (body.elastic.has_value()) {
                properties.push_back(
                    PropertyDescriptor<T>{"elastic_modulus", [body](size_t) {
                                              return body.elastic_modulus();
                                          }}
                );

                properties.push_back(
                    PropertyDescriptor<T>{"poisson_ratio", [body](size_t) {
                                              return body.poisson_ratio();
                                          }}
                );
            }

            if (body.deformable.has_value()) {
                properties.push_back(
                    PropertyDescriptor<T>{"yield_stress", [body](size_t) {
                                              return body.yield_stress();
                                          }}
                );

                properties.push_back(
                    PropertyDescriptor<T>{"plastic_strain", [body](size_t) {
                                              return body.plastic_strain();
                                          }}
                );
            }

            return properties;
        }

        void sync_to_device()
        {
            bodies_.sync_to_device();
            grav_body_indices_.sync_to_device();
            accr_body_indices_.sync_to_device();
        }

      private:
        const MeshType& mesh_;

        // primary storage - immutable by default
        ndarray_t<Body<T, Dims>> bodies_;

        // GPU-optimized lookup arrays
        ndarray_t<size_t> grav_body_indices_;
        ndarray_t<size_t> accr_body_indices_;

        // system config - optional
        util::smart_ptr<SystemConfig> system_config_;

        // reference frame for the system
        std::string reference_frame_;
    };
};   // namespace simbi::ibsystem
#endif
