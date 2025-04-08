#ifndef BODY_SYSTEM_HPP
#define BODY_SYSTEM_HPP

#include "../bodies/types/any_body.hpp"        // for AnyBody
#include "../bodies/types/body_concepts.hpp"   // for concepts
#include "body_factory.hpp"                    // for BodyFactory
#include "build_options.hpp"                   // for DUAL
#include "core/types/containers/vector.hpp"    // for spatial_vector_t
#include "core/types/utility/config_dict.hpp"
#include "physics/hydro/types/generic_structs.hpp"

using namespace simbi::ib::concepts;

namespace simbi {
    template <size_type Dims>
    class Mesh;
}

namespace simbi::ibsystem {
    template <typename T, size_type Dims>
    class BodySystem
    {
      protected:
        using MeshType    = Mesh<Dims>;
        using BodyRef     = ib::BodyReference<T, Dims>;
        using ConsArray   = typename ib::concepts::StateType<Dims>::ConsArray;
        using PrimArray   = typename ib::concepts::StateType<Dims>::PrimArray;
        using conserved_t = anyConserved<Dims, Regime::NEWTONIAN>;

        std::vector<std::unique_ptr<ib::AnyBody<T, Dims>>> bodies_;
        MeshType mesh_;

      public:
        BodySystem() = default;
        DUAL BodySystem(const MeshType& mesh) : mesh_(mesh) {}

        DUAL void add_body(
            BodyType type,
            const spatial_vector_t<T, Dims>& position,
            const spatial_vector_t<T, Dims>& velocity,
            T mass,
            T radius,
            const ConfigDict& props
        )
        {
            auto body = ib::BodyFactory<T, Dims>::build(
                type,
                mesh_,
                position,
                velocity,
                mass,
                radius,
                props
            );
            bodies_.push_back(std::move(body));
        }

        // Get a reference to a specific body
        DUAL BodyRef body_at(size_t index)
        {
            if (index >= bodies_.size()) {
                throw std::out_of_range("Body index out of range");
            }
            return BodyRef(*bodies_[index]);
        }

        // Get a vector of body references
        DUAL std::vector<BodyRef> get_body_references()
        {
            std::vector<BodyRef> refs;
            refs.reserve(bodies_.size());
            for (auto& body : bodies_) {
                refs.emplace_back(*body);
            }
            return refs;
        }

        // Apply forces to all bodies
        DUAL void calculate_forces(T dt)
        {
            auto refs = get_body_references();
            for (auto& body : bodies_) {
                body->calculate_forces(refs, dt);
            }
        }

        // Apply forces to fluid
        DUAL auto apply_forces_to_fluid(
            const auto& prim,
            const auto& mesh_cell,
            const auto& coords,
            const auto& context,
            const T dt
        )
        {
            auto state = conserved_t{};
            for (auto& body : bodies_) {
                state += body->apply_forces_to_fluid(
                    prim,
                    mesh_cell,
                    coords,
                    context,
                    dt
                );
            }

            // apply accretion only if the body has accretion traits
            for (auto& body : bodies_) {
                state += body->accrete_from_cell(
                    prim,
                    mesh_cell,
                    coords,
                    context,
                    dt
                );
            }
            return state;
        }

        // Update positions (advance positions for all bodies)
        DUAL void advance_positions(T dt)
        {
            for (auto& body : bodies_) {
                body->advance_position(dt);
            }
        }

        // Update velocities
        DUAL void advance_velocities(T dt)
        {
            for (auto& body : bodies_) {
                body->advance_velocity(dt);
            }
        }

        // Update material states
        DUAL void update_material_states(T dt)
        {
            for (auto& body : bodies_) {
                body->update_material_state(dt);
            }
        }

        // Size accessor
        DUAL size_t size() const { return bodies_.size(); }

        // Access to bodies (const)
        DUAL const auto& bodies() const { return bodies_; }

        DUAL const auto& mesh() const { return mesh_; }
    };
}   // namespace simbi::ibsystem

#endif
