#ifndef BODY_CONCEPTS_HPP
#define BODY_CONCEPTS_HPP

#include "build_options.hpp"
#include "core/types/containers/ndarray.hpp"
#include "core/types/containers/vector.hpp"
#include "core/types/utility/enums.hpp"
#include "physics/hydro/types/generic_structs.hpp"
#include <functional>
#include <vector>

namespace simbi::ib::concepts {
    template <size_type Dims>
    struct StateType {
        using ConsArray = ndarray<anyConserved<Dims, Regime::NEWTONIAN>, Dims>;
        using PrimArray =
            ndarray<Maybe<anyPrimitive<Dims, Regime::NEWTONIAN>>, Dims>;
    };

    // Core body capabilities
    template <typename Body, typename T, size_type Dims>
    concept HasBasicProperties = requires(const Body& b) {
        { b.position() } -> std::convertible_to<spatial_vector_t<T, Dims>>;
        { b.velocity() } -> std::convertible_to<spatial_vector_t<T, Dims>>;
        { b.force() } -> std::convertible_to<spatial_vector_t<T, Dims>>;
        { b.mass() } -> std::convertible_to<T>;
        { b.radius() } -> std::convertible_to<T>;
    };

    template <typename Body, typename T, size_type Dims>
    concept HasFluidInteraction = requires(
        Body& b,
        typename StateType<Dims>::ConsArray& cons_states,
        const typename StateType<Dims>::PrimArray& prim_states,
        T dt
    ) {
        {
            b.apply_to_fluid(cons_states, prim_states, dt)
        } -> std::same_as<void>;
        { b.interpolate_fluid_velocity(prim_states) } -> std::same_as<void>;
        {
            b.fluid_velocity()
        } -> std::convertible_to<spatial_vector_t<T, Dims>>;
    };

    template <typename Body, typename T, size_type Dims>
    concept HasDynamics = requires(Body& b, T dt) {
        { b.advance_position(dt) } -> std::same_as<void>;
        { b.advance_velocity(dt) } -> std::same_as<void>;
        {
            b.calculate_forces(
                std::declval<const std::vector<std::reference_wrapper<Body>>&>(
                ),
                dt
            )
        } -> std::same_as<void>;
        { b.update_material_state(dt) } -> std::same_as<void>;
    };

    // Specific capability concepts
    template <typename Body, typename T>
    concept HasGravitationalProperties = requires(const Body& b) {
        { b.softening_length() } -> std::convertible_to<T>;
        { b.two_way_coupling() } -> std::convertible_to<bool>;
    };

    template <typename Body, typename T>
    concept HasElasticProperties = requires(const Body& b) {
        { b.stiffness() } -> std::convertible_to<T>;
        { b.damping() } -> std::convertible_to<T>;
        { b.rest_length() } -> std::convertible_to<T>;
    };

    template <typename Body, typename T>
    concept HasDeformableProperties = requires(const Body& b) {
        { b.youngs_modulus() } -> std::convertible_to<T>;
        { b.poisson_ratio() } -> std::convertible_to<T>;
        { b.yield_strength() } -> std::convertible_to<T>;
        { b.is_permanently_deformed() } -> std::convertible_to<bool>;
        { b.stored_elastic_energy() } -> std::convertible_to<T>;
    };

    template <typename Body, typename T>
    concept HasAccretionProperties = requires(const Body& b) {
        { b.accretion_efficiency() } -> std::convertible_to<T>;
        { b.accretion_radius_factor() } -> std::convertible_to<T>;
        { b.total_accreted_mass() } -> std::convertible_to<T>;
    };

    // Combined concept for a complete immersed body
    template <typename Body, typename T, size_type Dims>
    concept ImmersedBody =
        HasBasicProperties<Body, T, Dims> &&
        HasFluidInteraction<Body, T, Dims> && HasDynamics<Body, T, Dims>;

    // Specialized body types
    template <typename Body, typename T, size_type Dims>
    concept GravitationalBody =
        ImmersedBody<Body, T, Dims> && HasGravitationalProperties<Body, T>;

    template <typename Body, typename T, size_type Dims>
    concept ElasticBody =
        ImmersedBody<Body, T, Dims> && HasElasticProperties<Body, T>;

    template <typename Body, typename T, size_type Dims>
    concept AccretingBody =
        ImmersedBody<Body, T, Dims> && HasAccretionProperties<Body, T>;
}   // namespace simbi::ib::concepts

#endif   // BODY_CONCEPTS_HPP
