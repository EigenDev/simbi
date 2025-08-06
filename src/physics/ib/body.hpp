#ifndef BODY_HPP
#define BODY_HPP

#include "config.hpp"
#include "containers/vector.hpp"
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <tuple>
#include <unordered_map>

namespace simbi::body::capabilities {
    struct gravitational_tag {
    };
    struct accretion_tag {
    };
    struct elastic_tag {
    };
    struct rigid_tag {
    };
    struct deformable_tag {
    };
}   // namespace simbi::body::capabilities

namespace simbi::body {
    // primary template - not found case (will cause compile error if used)
    template <typename Tag, typename... Caps>
    struct find_capability_index {
        static constexpr std::size_t value =
            std::numeric_limits<std::size_t>::max();
    };

    // specialized template for when the first type has matching tag
    template <typename Tag, typename First, typename... Rest>
    struct find_capability_index<Tag, First, Rest...> {
      private:
        static constexpr bool is_match =
            std::is_same_v<Tag, typename First::tag_type>;

        // recursively search the rest if not a match
        static constexpr std::size_t next_index =
            find_capability_index<Tag, Rest...>::value;

      public:
        static constexpr std::size_t value =
            is_match ? 0
                     : (next_index == std::numeric_limits<std::size_t>::max()
                            ? next_index
                            : 1 + next_index);
    };

    // base case for recursion
    template <typename Tag>
    struct find_capability_index<Tag> {
        static constexpr std::size_t value =
            std::numeric_limits<std::size_t>::max();
    };

    template <typename T>
    struct body_properties_t {
        std::unordered_map<std::string, T> scalars;
        std::unordered_map<std::string, bool> flags;

        bool has_capability(const std::string& cap) const
        {
            auto it = flags.find(cap);
            return it != flags.end() && it->second;
        }

        T get_scalar(const std::string& key, T default_val = T{0}) const
        {
            auto it = scalars.find(key);
            return it != scalars.end() ? it->second : default_val;
        }
    };

}   // namespace simbi::body

namespace simbi::body {
    template <std::uint64_t Dims, typename... Caps>
    struct body_t;

    struct grav_component_t {
        using tag_type = capabilities::gravitational_tag;
        real softening_length;
    };

    struct accretion_component_t {
        using tag_type = capabilities::accretion_tag;
        real accretion_efficiency;
        real accretion_radius;
        real total_accreted_mass;
        real accretion_rate;
    };

    struct elastic_component_t {
        using tag_type = capabilities::elastic_tag;
        real elastic_modulus;
        real poisson_ratio;
    };

    struct rigid_component_t {
        using tag_type = capabilities::rigid_tag;
        real inertia;
        bool apply_no_slip;
    };

    struct deformable_component_t {
        using tag_type = capabilities::deformable_tag;
        real yield_stress;
        real plastic_strain;
    };

    // type aliases for common body types
    template <std::uint64_t Dims>
    using rigid_sphere_t = body_t<Dims, rigid_component_t>;

    template <std::uint64_t Dims>
    using gravitational_body_t = body_t<Dims, grav_component_t>;

    template <std::uint64_t Dims>
    using black_hole_t = body_t<Dims, grav_component_t, accretion_component_t>;

    template <std::uint64_t Dims>
    using planet_t = body_t<Dims, grav_component_t, rigid_component_t>;

    template <std::uint64_t Dims, typename... Caps>
    struct body_t {
        // expose the types for easier access
        using caps_tuple                     = std::tuple<Caps...>;
        static constexpr std::uint64_t ncaps = sizeof...(Caps);

        std::uint64_t idx;
        vector_t<real, Dims> position;
        vector_t<real, Dims> velocity;
        vector_t<real, Dims> force;
        vector_t<real, Dims> torque;
        real mass;
        real radius;
        bool two_way_coupling;

        std::tuple<Caps...> capabilities;

        template <typename Tag>
        static constexpr bool has_capability_v =
            (std::is_same_v<Tag, typename Caps::tag_type> || ...);
    };

    // concepts for capabilities b/c c++20 is amazing :D
    template <typename T>
    concept has_gravitational_capability_c = requires {
        requires T::template has_capability_v<
                     capabilities::gravitational_tag> == true;
    };

    template <typename T>
    concept has_accretion_capability_c = requires {
        requires T::template has_capability_v<capabilities::accretion_tag> ==
                     true;
    };

    template <typename T>
    concept has_elastic_capability_c = requires {
        requires T::template has_capability_v<capabilities::elastic_tag> ==
                     true;
    };

    template <typename T>
    concept has_rigid_capability_c = requires {
        requires T::template has_capability_v<capabilities::rigid_tag> == true;
    };

    template <typename T>
    concept has_deformable_capability_c = requires {
        requires T::template has_capability_v<capabilities::deformable_tag> ==
                     true;
    };

    // immutable update functions
    template <typename Tag, std::uint64_t Dims, typename... Caps>
    DUAL constexpr auto get_capabilities(const body_t<Dims, Caps...>& body)
    {
        constexpr auto index = find_capability_index<Tag, Caps...>::value;
        return std::get<index>(body.capabilities);
    }

    template <std::uint64_t Dims, typename... Caps>
    DUAL constexpr auto with_force(
        const body_t<Dims, Caps...>& body,
        const vector_t<real, Dims>& new_force
    )
    {
        auto result  = body;
        result.force = new_force;
        return result;
    }

    template <std::uint64_t Dims, typename... Caps>
    DUAL constexpr auto with_torque(
        const body_t<Dims, Caps...>& body,
        const vector_t<real, Dims>& new_torque
    )
    {
        auto result   = body;
        result.torque = new_torque;
        return result;
    }

    template <std::uint64_t Dims, typename... Caps>
    DUAL constexpr auto with_velocity(
        const body_t<Dims, Caps...>& body,
        const vector_t<real, Dims>& new_velocity
    )
    {
        auto result     = body;
        result.velocity = new_velocity;
        return result;
    }

    template <std::uint64_t Dims, typename... Caps>
    DUAL constexpr auto
    with_mass(const body_t<Dims, Caps...>& body, real new_mass)
    {
        auto result = body;
        result.mass = new_mass;
        return result;
    }

    template <std::uint64_t Dims, typename... Caps>
    DUAL constexpr auto
    with_radius(const body_t<Dims, Caps...>& body, real new_radius)
    {
        auto result   = body;
        result.radius = new_radius;
        return result;
    }

    template <std::uint64_t Dims, typename... Caps>
    DUAL constexpr auto at_position(
        const body_t<Dims, Caps...>& body,
        const vector_t<real, Dims>& new_position
    )
    {
        auto result     = body;
        result.position = new_position;
        return result;
    }

    // factory functions for common body types
    template <std::uint64_t Dims>
    DUAL constexpr auto make_basic_body(
        std::uint64_t idx,
        const vector_t<real, Dims>& position,
        const vector_t<real, Dims>& velocity,
        real mass,
        real radius,
        bool two_way_coupling = false
    )
    {
        return body_t<Dims>{
          idx,
          position,
          velocity,
          vector_t<real, Dims>{},
          vector_t<real, Dims>{},
          mass,
          radius,
          two_way_coupling,
          std::tuple<>()   // no capabilities
        };
    }

    template <std::uint64_t Dims>
    DUAL constexpr auto make_gravitational_body(
        std::uint64_t idx,
        const vector_t<real, Dims>& position,
        const vector_t<real, Dims>& velocity,
        real mass,
        real radius,
        real softening_length,
        bool two_way_coupling = false
    )
    {
        return body_t<Dims, grav_component_t>{
          idx,
          position,
          velocity,
          vector_t<real, Dims>{},
          vector_t<real, Dims>{},
          mass,
          radius,
          two_way_coupling,
          std::make_tuple(grav_component_t{softening_length})
        };
    }

    template <std::uint64_t Dims>
    DUAL constexpr auto make_black_hole(
        std::uint64_t idx,
        const vector_t<real, Dims>& position,
        const vector_t<real, Dims>& velocity,
        real mass,
        real radius,
        real softening_length,
        real accretion_efficiency,
        real accretion_radius,
        real accretion_rate      = 0.0,
        real total_accreted_mass = 0.0,
        bool two_way_coupling    = false
    )
    {
        return body_t<Dims, grav_component_t, accretion_component_t>{
          idx,
          position,
          velocity,
          vector_t<real, Dims>{},
          vector_t<real, Dims>{},
          mass,
          radius,
          two_way_coupling,
          std::make_tuple(
              grav_component_t{softening_length},
              accretion_component_t{
                accretion_efficiency,
                accretion_radius,
                total_accreted_mass,
                accretion_rate
              }
          )
        };
    }

    template <std::uint64_t Dims>
    DUAL constexpr auto make_planet(
        std::uint64_t idx,
        const vector_t<real, Dims>& position,
        const vector_t<real, Dims>& velocity,
        real mass,
        real radius,
        real inertia,
        bool apply_no_slip    = true,
        bool two_way_coupling = false
    )
    {
        return body_t<Dims, grav_component_t, rigid_component_t>{
          idx,
          position,
          velocity,
          vector_t<real, Dims>{},
          vector_t<real, Dims>{},
          mass,
          radius,
          two_way_coupling,
          std::make_tuple(
              grav_component_t{
                0.0
              },   // no softening for planets (can be set later)
              rigid_component_t{inertia, apply_no_slip}
          )
        };
    }

    template <std::uint64_t Dims>
    DUAL constexpr auto make_rigid_sphere(
        std::uint64_t idx,
        const vector_t<real, Dims>& position,
        const vector_t<real, Dims>& velocity,
        real mass,
        real radius,
        real inertia,
        bool apply_no_slip    = true,
        bool two_way_coupling = false
    )
    {
        return body_t<Dims, rigid_component_t>{
          idx,
          position,
          velocity,
          vector_t<real, Dims>{},
          vector_t<real, Dims>{},
          mass,
          radius,
          two_way_coupling,
          std::make_tuple(rigid_component_t{inertia, apply_no_slip})
        };
    }

    // convenient property accessors
    // gravitational properties
    template <has_gravitational_capability_c Body>
    DUAL constexpr auto softening_length(const Body& body) -> real
    {
        auto grav_cap = get_capabilities<capabilities::gravitational_tag>(body);
        return grav_cap.softening_length;
    }

    // accretion properties
    template <has_accretion_capability_c Body>
    DUAL constexpr auto accretion_efficiency(const Body& body) -> real
    {
        auto accr_cap = get_capabilities<capabilities::accretion_tag>(body);
        return accr_cap.accretion_efficiency;
    }

    template <has_accretion_capability_c Body>
    DUAL constexpr auto accretion_radius(const Body& body) -> real
    {
        auto accr_cap = get_capabilities<capabilities::accretion_tag>(body);
        return accr_cap.accretion_radius;
    }

    template <has_accretion_capability_c Body>
    DUAL constexpr auto total_accreted_mass(const Body& body) -> real
    {
        auto accr_cap = get_capabilities<capabilities::accretion_tag>(body);
        return accr_cap.total_accreted_mass;
    }

    template <has_accretion_capability_c Body>
    DUAL constexpr auto accretion_rate(const Body& body) -> real
    {
        auto accr_cap = get_capabilities<capabilities::accretion_tag>(body);
        return accr_cap.accretion_rate;
    }

    template <has_accretion_capability_c Body>
    DUAL constexpr auto sinking_rate(const Body& /*body*/) -> real
    {
        // placeholder for sinking rate calculation
        return 1e-3;
        // auto accr_cap = get_capabilities<capabilities::accretion_tag>(body);
        // return accr_cap.accretion_rate / body.mass;
    }

    // rigid body properties
    template <has_rigid_capability_c Body>
    DUAL constexpr auto inertia(const Body& body) -> real
    {
        auto rigid_cap = get_capabilities<capabilities::rigid_tag>(body);
        return rigid_cap.inertia;
    }

    // [TODO] add more properties as needed

}   // namespace simbi::body

#endif
