#ifndef ANY_BODY_HPP
#define ANY_BODY_HPP

#include "body_concepts.hpp"
#include "build_options.hpp"
#include "core/types/containers/vector.hpp"
#include "core/types/utility/managed.hpp"
#include "geometry/mesh/cell.hpp"
#include "physics/hydro/types/context.hpp"
#include "physics/hydro/types/generic_structs.hpp"

namespace simbi::ib {
    // Forward declarations
    template <typename T, size_type Dims>
    class AnyBody;

    template <typename T, size_type Dims>
    class BodyReference : public Managed<global::managed_memory>
    {
      private:
        AnyBody<T, Dims>* body_;

      public:
        using Conserved = anyConserved<Dims, Regime::NEWTONIAN>;
        using Primitive = anyPrimitive<Dims, Regime::NEWTONIAN>;
        using CellType  = Cell<Dims>;

        BodyReference() : body_(nullptr) {}
        BodyReference(AnyBody<T, Dims>& body) : body_(&body) {}

        spatial_vector_t<T, Dims> position() const { return body_->position(); }
        spatial_vector_t<T, Dims> velocity() const { return body_->velocity(); }
        spatial_vector_t<T, Dims> force() const { return body_->force(); }
        T mass() const { return body_->mass(); }
        T radius() const { return body_->radius(); }

        void advance_position(T dt) { body_->advance_position(dt); }
        void advance_velocity(T dt) { body_->advance_velocity(dt); }
        void
        calculate_forces(const ndarray<BodyReference<T, Dims>>& others, T dt)
        {
            body_->calculate_forces(others, dt);
        }
        void update_material_state(T dt) { body_->update_material_state(dt); }
        Conserved apply_forces_to_fluid(
            const Primitive& prim,
            const CellType& mesh_cell,
            const std::tuple<size_type, size_type, size_type>& coords,
            const HydroContext& context,
            T dt
        )
        {
            return body_
                ->apply_forces_to_fluid(prim, mesh_cell, coords, context, dt);
        }
    };

    // Main type erasure class
    template <typename T, size_type Dims>
    class AnyBody : public Managed<global::managed_memory>
    {
      private:
        using BodyRef   = BodyReference<T, Dims>;
        using Conserved = anyConserved<Dims, Regime::NEWTONIAN>;
        using Primitive = anyPrimitive<Dims, Regime::NEWTONIAN>;
        using CellType  = Cell<Dims>;

        // Interface for the body
        struct Concept {
            virtual ~Concept() = default;

            // Basic properties
            virtual spatial_vector_t<T, Dims> position() const = 0;
            virtual spatial_vector_t<T, Dims> velocity() const = 0;
            virtual spatial_vector_t<T, Dims> force() const    = 0;
            virtual T mass() const                             = 0;
            virtual T radius() const                           = 0;

            // Dynamics
            virtual void advance_position(T dt) = 0;
            virtual void advance_velocity(T dt) = 0;
            virtual void
            calculate_forces(const ndarray<BodyRef>& others, T dt) = 0;
            virtual void update_material_state(T dt)               = 0;
            virtual void set_position(const spatial_vector_t<T, Dims>& pos) {}
            virtual void set_velocity(const spatial_vector_t<T, Dims>& vel) {}
            virtual void set_mass(const T mass) {}
            virtual void set_radius(const T radius) {}

            // Fluid interaction
            virtual Conserved apply_forces_to_fluid(
                const Primitive& prim,
                const CellType& mesh_cell,
                const std::tuple<size_type, size_type, size_type>& coords,
                const HydroContext& context,
                T dt
            )                                                        = 0;
            virtual spatial_vector_t<T, Dims> fluid_velocity() const = 0;
            virtual Conserved accrete_from_cell(
                const Primitive& prim,
                const CellType& mesh_cell,
                const std::tuple<size_type, size_type, size_type>& coords,
                const HydroContext& context,
                T dt
            )
            {
                return Conserved{};
            }

            // Capability testing (returns true if capability exists)
            virtual bool has_gravitational_capability() const { return false; }
            virtual bool has_elastic_capability() const { return false; }
            virtual bool has_accretion_capability() const { return false; }
            virtual bool has_deformable_capability() const { return false; }

            // Optional gravitational properties
            virtual T softening_length() const { return T(0); }
            virtual bool two_way_coupling() const { return false; }

            // Optional elastic properties
            virtual T stiffness() const { return T(0); }
            virtual T damping() const { return T(0); }
            virtual T rest_length() const { return T(0); }

            // Optional deformable properties
            virtual T youngs_modulus() const { return T(0); }
            virtual T poisson_ratio() const { return T(0); }
            virtual T yield_strength() const { return T(0); }
            virtual bool is_permanently_deformed() const { return false; }
            virtual T stored_elastic_energy() const { return T(0); }

            // Optional accretion properties
            virtual T accretion_efficiency() const { return T(0); }
            virtual T accretion_radius() const { return T(0); }
            virtual T total_accreted_mass() const { return T(0); }
        };

        // Concrete implementation that wraps a specific body type
        template <typename BodyType>
        struct Model : Concept {
            BodyType body_;

            template <typename... Args>
            Model(Args&&... args) : body_(std::forward<Args>(args)...)
            {
            }

            // Implement basic property methods
            spatial_vector_t<T, Dims> position() const override
            {
                return body_.position();
            }
            spatial_vector_t<T, Dims> velocity() const override
            {
                return body_.velocity();
            }
            spatial_vector_t<T, Dims> force() const override
            {
                return body_.force();
            }
            T mass() const override { return body_.mass(); }
            T radius() const override { return body_.radius(); }

            // Implement dynamics methods
            void advance_position(T dt) override { body_.advance_position(dt); }
            void advance_velocity(T dt) override { body_.advance_velocity(dt); }
            void set_position(const spatial_vector_t<T, Dims>& pos) override
            {
                body_.set_position(pos);
            }
            void set_velocity(const spatial_vector_t<T, Dims>& vel) override
            {
                body_.set_velocity(vel);
            }

            void set_mass(const T mass) override { body_.set_mass(mass); }
            void set_radius(const T radius) override
            {
                body_.set_radius(radius);
            }

            void calculate_forces(const ndarray<BodyRef>& others, T dt) override
            {
                // Convert references to actual body references
                ndarray<std::reference_wrapper<BodyType>> body_refs;

                // convert the references to the actual body references
                // this is necessary because the calculate_forces method
                // expects a vector of references to the actual body type
                // and not the type-erased AnyBody type
                // for (const auto& ref : others) {
                //     body_refs.push_back(
                //         std::ref(
                //             static_cast<Model<BodyType>*>(ref.body_)->body_
                //         )
                //     );
                // }

                body_.calculate_forces(body_refs, dt);
            }

            void update_material_state(T dt) override
            {
                body_.update_material_state(dt);
            }

            Conserved apply_forces_to_fluid(
                const Primitive& prim,
                const CellType& mesh_cell,
                const std::tuple<size_type, size_type, size_type>& coords,
                const HydroContext& context,
                T dt
            ) override
            {
                return body_.apply_forces_to_fluid(
                    prim,
                    mesh_cell,
                    coords,
                    context,
                    dt
                );
            }

            spatial_vector_t<T, Dims> fluid_velocity() const override
            {
                return body_.fluid_velocity();
            }

            // Implement capability testing
            bool has_gravitational_capability() const override
            {
                return concepts::HasGravitationalProperties<BodyType, T>;
            }

            bool has_elastic_capability() const override
            {
                return concepts::HasElasticProperties<BodyType, T>;
            }

            bool has_accretion_capability() const override
            {
                return concepts::HasAccretionProperties<BodyType, T>;
            }

            bool has_deformable_capability() const override
            {
                return concepts::HasDeformableProperties<BodyType, T>;
            }

            // Implement optional capability methods if they exist

            T softening_length() const override
            {
                if constexpr (concepts::
                                  HasGravitationalProperties<BodyType, T>) {
                    return body_.softening_length();
                }
                return Concept::softening_length();
            }

            bool two_way_coupling() const override
            {
                if constexpr (concepts::
                                  HasGravitationalProperties<BodyType, T>) {
                    return body_.two_way_coupling();
                }
                return Concept::two_way_coupling();
            }

            Conserved accrete_from_cell(
                const Primitive& prim,
                const CellType& mesh_cell,
                const std::tuple<size_type, size_type, size_type>& coords,
                const HydroContext& context,
                T dt
            ) override
            {
                if constexpr (concepts::HasAccretionProperties<BodyType, T>) {
                    return body_.accrete_from_cell(
                        prim,
                        mesh_cell,
                        coords,
                        context,
                        dt
                    );
                }
                return Conserved{};
            }

            T stiffness() const override
            {
                if constexpr (concepts::HasElasticProperties<BodyType, T>) {
                    return body_.stiffness();
                }
                return Concept::stiffness();
            }

            T damping() const override
            {
                if constexpr (concepts::HasElasticProperties<BodyType, T>) {
                    return body_.damping();
                }
                return Concept::damping();
            }

            T rest_length() const override
            {
                if constexpr (concepts::HasElasticProperties<BodyType, T>) {
                    return body_.rest_length();
                }
                return Concept::rest_length();
            }

            T youngs_modulus() const override
            {
                if constexpr (concepts::HasDeformableProperties<BodyType, T>) {
                    return body_.youngs_modulus();
                }
                return Concept::youngs_modulus();
            }

            T poisson_ratio() const override
            {
                if constexpr (concepts::HasDeformableProperties<BodyType, T>) {
                    return body_.poisson_ratio();
                }
                return Concept::poisson_ratio();
            }

            T yield_strength() const override
            {
                if constexpr (concepts::HasDeformableProperties<BodyType, T>) {
                    return body_.yield_strength();
                }
                return Concept::yield_strength();
            }

            bool is_permanently_deformed() const override
            {
                if constexpr (concepts::HasDeformableProperties<BodyType, T>) {
                    return body_.is_permanently_deformed();
                }
                return Concept::is_permanently_deformed();
            }

            T stored_elastic_energy() const override
            {
                if constexpr (concepts::HasDeformableProperties<BodyType, T>) {
                    return body_.stored_elastic_energy();
                }
                return Concept::stored_elastic_energy();
            }

            T accretion_efficiency() const override
            {
                if constexpr (concepts::HasAccretionProperties<BodyType, T>) {
                    return body_.accretion_efficiency();
                }
                return Concept::accretion_efficiency();
            }

            T accretion_radius() const override
            {
                if constexpr (concepts::HasAccretionProperties<BodyType, T>) {
                    return body_.accretion_radius();
                }
                return Concept::accretion_radius();
            }

            T total_accreted_mass() const override
            {
                if constexpr (concepts::HasAccretionProperties<BodyType, T>) {
                    return body_.total_accreted_mass();
                }
                return Concept::total_accreted_mass();
            }
        };

        // The actual implementation
        util::smart_ptr<Concept> concept_;

      public:
        // Constructor from any type that satisfies the ImmersedBody concept
        template <concepts::ImmersedBody<T, Dims> BodyType, typename... Args>
        AnyBody(std::in_place_type_t<BodyType>, Args&&... args)
            : concept_(
                  util::make_unique<Model<BodyType>>(std::forward<Args>(args)...
                  )
              )
        {
        }

        // Forwarding methods to access the body
        spatial_vector_t<T, Dims> position() const
        {
            return concept_->position();
        }
        spatial_vector_t<T, Dims> velocity() const
        {
            return concept_->velocity();
        }

        spatial_vector_t<T, Dims> force() const { return concept_->force(); }

        T mass() const { return concept_->mass(); }
        T radius() const { return concept_->radius(); }

        void advance_position(T dt) { concept_->advance_position(dt); }
        void advance_velocity(T dt) { concept_->advance_velocity(dt); }
        void calculate_forces(const ndarray<BodyRef>& others, T dt)
        {
            concept_->calculate_forces(others, dt);
        }

        void update_material_state(T dt)
        {
            concept_->update_material_state(dt);
        }

        Conserved apply_forces_to_fluid(
            const Primitive& prim,
            const CellType& mesh_cell,
            const std::tuple<size_type, size_type, size_type>& coords,
            const HydroContext& context,
            T dt
        )
        {
            return concept_
                ->apply_forces_to_fluid(prim, mesh_cell, coords, context, dt);
        }

        Conserved accrete_from_cell(
            const Primitive& prim,
            const CellType& mesh_cell,
            const std::tuple<size_type, size_type, size_type>& coords,
            const HydroContext& context,
            T dt
        )
        {
            return concept_
                ->accrete_from_cell(prim, mesh_cell, coords, context, dt);
        }

        spatial_vector_t<T, Dims> fluid_velocity() const
        {
            return concept_->fluid_velocity();
        }

        // Capability testing
        bool has_gravitational_capability() const
        {
            return concept_->has_gravitational_capability();
        }

        bool has_elastic_capability() const
        {
            return concept_->has_elastic_capability();
        }

        bool has_accretion_capability() const
        {
            return concept_->has_accretion_capability();
        }

        bool has_deformable_capability() const
        {
            return concept_->has_deformable_capability();
        }

        // Optional properties
        T softening_length() const { return concept_->softening_length(); }

        bool two_way_coupling() const { return concept_->two_way_coupling(); }

        T stiffness() const { return concept_->stiffness(); }

        T damping() const { return concept_->damping(); }

        T rest_length() const { return concept_->rest_length(); }

        T youngs_modulus() const { return concept_->youngs_modulus(); }

        T poisson_ratio() const { return concept_->poisson_ratio(); }

        T yield_strength() const { return concept_->yield_strength(); }

        bool is_permanently_deformed() const
        {
            return concept_->is_permanently_deformed();
        }

        T stored_elastic_energy() const
        {
            return concept_->stored_elastic_energy();
        }

        T accretion_efficiency() const
        {
            return concept_->accretion_efficiency();
        }

        T accretion_radius() const { return concept_->accretion_radius(); }

        T total_accreted_mass() const
        {
            return concept_->total_accreted_mass();
        }

        // Setters for position and velocity

        void set_position(const spatial_vector_t<T, Dims>& pos)
        {
            concept_->set_position(pos);
        }

        void set_velocity(const spatial_vector_t<T, Dims>& vel)
        {
            concept_->set_velocity(vel);
        }

        void set_mass(const T mass) { concept_->set_mass(mass); }
        void set_radius(const T radius) { concept_->set_radius(radius); }
    };   // class AnyBody
}   // namespace simbi::ib
#endif   // ANY_HPP
