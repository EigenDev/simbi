#ifndef ANY_BODY_HPP
#define ANY_BODY_HPP

#include "body_concepts.hpp"
#include "core/types/containers/vector.hpp"

namespace simbi::ib {
    // Forward declarations
    template <typename T, size_type Dims>
    class AnyBody;

    template <typename T, size_type Dims>
    class BodyReference
    {
      private:
        AnyBody<T, Dims>* body_;

      public:
        using ConsArray = concepts::StateType<Dims>::ConsArray;
        using PrimArray = concepts::StateType<Dims>::PrimArray;

        BodyReference(AnyBody<T, Dims>& body) : body_(&body) {}

        spatial_vector_t<T, Dims> position() const { return body_->position(); }
        spatial_vector_t<T, Dims> velocity() const { return body_->velocity(); }
        spatial_vector_t<T, Dims> force() const { return body_->force(); }
        T mass() const { return body_->mass(); }
        T radius() const { return body_->radius(); }

        void advance_position(T dt) { body_->advance_position(dt); }
        void advance_velocity(T dt) { body_->advance_velocity(dt); }
        void calculate_forces(
            const std::vector<BodyReference<T, Dims>>& others,
            T dt
        )
        {
            body_->calculate_forces(others, dt);
        }
        void update_material_state(T dt) { body_->update_material_state(dt); }
        void apply_to_fluid(
            ConsArray& cons_states,
            const PrimArray& prim_states,
            T dt
        )
        {
            body_->apply_to_fluid(cons_states, prim_states, dt);
        }
    };

    // Main type erasure class
    template <typename T, size_type Dims>
    class AnyBody
    {
      private:
        using BodyRef   = BodyReference<T, Dims>;
        using ConsArray = concepts::StateType<Dims>::ConsArray;
        using PrimArray = concepts::StateType<Dims>::PrimArray;

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
            calculate_forces(const std::vector<BodyRef>& others, T dt) = 0;
            virtual void update_material_state(T dt)                   = 0;
            virtual void set_position(const spatial_vector_t<T, Dims>& pos) {}
            virtual void set_velocity(const spatial_vector_t<T, Dims>& vel) {}

            // Fluid interaction
            virtual void apply_to_fluid(
                ConsArray& cons_states,
                const PrimArray& prim_states,
                T dt
            )                                                        = 0;
            virtual spatial_vector_t<T, Dims> fluid_velocity() const = 0;

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
            virtual T accretion_radius_factor() const { return T(0); }
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

            void
            calculate_forces(const std::vector<BodyRef>& others, T dt) override
            {
                // Convert references to actual body references
                std::vector<std::reference_wrapper<BodyType>> body_refs;

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

            // Implement fluid interaction methods
            void apply_to_fluid(
                ConsArray& cons_states,
                const PrimArray& prim_states,
                T dt
            ) override
            {
                body_.apply_to_fluid(cons_states, prim_states, dt);
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

            // Similarly implement other optional properties...
        };

        // The actual implementation
        std::unique_ptr<Concept> concept_;

      public:
        // Constructor from any type that satisfies the ImmersedBody concept
        template <concepts::ImmersedBody<T, Dims> BodyType, typename... Args>
        AnyBody(std::in_place_type_t<BodyType>, Args&&... args)
            : concept_(std::make_unique<Model<BodyType>>(std::forward<Args>(args
              )...))
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
        void calculate_forces(const std::vector<BodyRef>& others, T dt)
        {
            concept_->calculate_forces(others, dt);
        }

        void update_material_state(T dt)
        {
            concept_->update_material_state(dt);
        }

        void apply_to_fluid(
            ConsArray& cons_states,
            const PrimArray& prim_states,
            T dt
        )
        {
            concept_->apply_to_fluid(cons_states, prim_states, dt);
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

        void set_position(const spatial_vector_t<T, Dims>& pos)
        {
            concept_->set_position(pos);
        }

        void set_velocity(const spatial_vector_t<T, Dims>& vel)
        {
            concept_->set_velocity(vel);
        }
    };
}   // namespace simbi::ib
#endif   // ANY_HPP
