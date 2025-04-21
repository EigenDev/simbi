#ifndef BODY_HPP
#define BODY_HPP

#include "build_options.hpp"
#include "capability.hpp"
#include "core/types/containers/vector.hpp"
#include "core/types/monad/maybe.hpp"

namespace simbi::ibsystem {
    template <typename T, size_type Dims>
    struct Body {
        // core properties (always present)
        BodyType type;
        spatial_vector_t<T, Dims> position;
        spatial_vector_t<T, Dims> velocity;
        spatial_vector_t<T, Dims> force;
        T mass;
        T radius;

        // optional components using Maybe monad
        Maybe<GravitationalComponent<T>> gravitational;
        Maybe<AccretionComponent<T>> accretion;
        Maybe<ElasticComponent<T>> elastic;
        Maybe<RigidComponent<T>> rigid;
        Maybe<DeformableComponent<T>> deformable;

        // ctors
        DUAL Body()
            : type(BodyType::GRAVITATIONAL),
              position(spatial_vector_t<T, Dims>()),
              velocity(spatial_vector_t<T, Dims>()),
              force(spatial_vector_t<T, Dims>()),
              mass(T(0)),
              radius(T(0)),
              gravitational(Nothing),
              accretion(Nothing)
        {
        }
        DUAL Body(
            BodyType type,
            const spatial_vector_t<T, Dims>& position,
            const spatial_vector_t<T, Dims>& velocity,
            T mass,
            T radius
        )
            : type(type),
              position(position),
              velocity(velocity),
              force(spatial_vector_t<T, Dims>()),
              mass(mass),
              radius(radius),
              gravitational(Nothing),
              accretion(Nothing)
        {
        }

        // copy ctor
        DUAL constexpr Body(const Body& other)
            : type(other.type),
              position(other.position),
              velocity(other.velocity),
              force(other.force),
              mass(other.mass),
              radius(other.radius),
              gravitational(other.gravitational),
              accretion(other.accretion)
        {
        }
        // move ctor
        DUAL constexpr Body(Body&& other) noexcept
            : type(other.type),
              position(std::move(other.position)),
              velocity(std::move(other.velocity)),
              force(std::move(other.force)),
              mass(other.mass),
              radius(other.radius),
              gravitational(std::move(other.gravitational)),
              accretion(std::move(other.accretion))
        {
        }
        // copy assignment
        DUAL constexpr Body& operator=(const Body& other)
        {
            if (this != &other) {
                type          = other.type;
                position      = other.position;
                velocity      = other.velocity;
                force         = other.force;
                mass          = other.mass;
                radius        = other.radius;
                gravitational = other.gravitational;
                accretion     = other.accretion;
            }
            return *this;
        }
        // move assignment
        DUAL constexpr Body& operator=(Body&& other) noexcept
        {
            if (this != &other) {
                type          = other.type;
                position      = std::move(other.position);
                velocity      = std::move(other.velocity);
                force         = std::move(other.force);
                mass          = other.mass;
                radius        = other.radius;
                gravitational = std::move(other.gravitational);
                accretion     = std::move(other.accretion);
            }
            return *this;
        }

        DUAL Body<T, Dims>
        with_gravitational(T softening, bool two_way = false) const
        {
            Body<T, Dims> new_body = *this;
            new_body.gravitational =
                GravitationalComponent<T>{softening, two_way};
            return new_body;
        }

        DUAL Body<T, Dims> with_accretion(T efficiency, T accr_radius = 0) const
        {
            Body<T, Dims> new_body = *this;
            new_body.accretion     = AccretionComponent<T>{
              efficiency,
              accr_radius <= 0 ? radius : accr_radius,
              T(0)   // Initial accreted mass
            };
            return new_body;
        }

        DUAL Body<T, Dims>
        with_force(const spatial_vector_t<T, Dims>& new_force) const
        {
            Body<T, Dims> new_body = *this;
            new_body.force         = new_force;
            return new_body;
        }

        DUAL Body<T, Dims>
        update_position(const spatial_vector_t<T, Dims>& new_position) const
        {
            Body<T, Dims> new_body = *this;
            new_body.position      = new_position;
            return new_body;
        }

        DUAL Body<T, Dims>
        update_velocity(const spatial_vector_t<T, Dims>& new_velocity) const
        {
            Body<T, Dims> new_body = *this;
            new_body.velocity      = new_velocity;
            return new_body;
        }

        DUAL Body<T, Dims> add_accreted_mass(T added_mass) const
        {
            Body<T, Dims> new_body = *this;
            if (new_body.accretion.has_value()) {
                auto component = new_body.accretion.value();
                component.total_accreted_mass += added_mass;
                new_body.accretion = component;
            }
            return new_body;
        }

        // query functions
        DUAL bool has_capability(BodyCapability cap) const
        {
            switch (cap) {
                case BodyCapability::GRAVITATIONAL:
                    return gravitational.has_value();
                case BodyCapability::ACCRETION: return accretion.has_value();
                // Add more capabilities as needed
                default: return false;
            }
        }

        // get the capabilities of the body
        DUAL BodyCapability capabilities() const
        {
            BodyCapability caps = BodyCapability::NONE;
            if (gravitational.has_value()) {
                caps = static_cast<BodyCapability>(
                    static_cast<int>(caps) |
                    static_cast<int>(BodyCapability::GRAVITATIONAL)
                );
            }
            if (accretion.has_value()) {
                caps = static_cast<BodyCapability>(
                    static_cast<int>(caps) |
                    static_cast<int>(BodyCapability::ACCRETION)
                );
            }
            return caps;
        }

        // access helper functions
        // gravitational
        DUAL T softening_length() const
        {
            return gravitational
                .map([](const auto& g) { return g.softening_length; })
                .unwrap_or(T(0));
        }

        DUAL bool two_way_coupling() const
        {
            return gravitational
                .map([](const auto& g) { return g.two_way_coupling; })
                .unwrap_or(false);
        }

        // accretion
        DUAL T accretion_efficiency() const
        {
            return accretion
                .map([](const auto& a) { return a.accretion_efficiency; })
                .unwrap_or(T(0));
        }

        DUAL T accretion_radius() const
        {
            return accretion
                .map([](const auto& a) { return a.accretion_radius; })
                .unwrap_or(radius);
        }

        DUAL T total_accreted_mass() const
        {
            return accretion
                .map([](const auto& a) { return a.total_accreted_mass; })
                .unwrap_or(T(0));
        }

        DUAL T accretion_rate() const
        {
            return accretion.map([](const auto& a) { return a.accretion_rate; }
            ).unwrap_or(T(0));
        }

        // elastic
        DUAL T elastic_modulus() const
        {
            return elastic.map([](const auto& e) { return e.elastic_modulus; }
            ).unwrap_or(T(0));
        }

        DUAL T poisson_ratio() const
        {
            return elastic.map([](const auto& e) { return e.poisson_ratio; }
            ).unwrap_or(T(0));
        }

        // deformable
        DUAL T yield_stress() const
        {
            return deformable.map([](const auto& d) { return d.yield_stress; }
            ).unwrap_or(T(0));
        }

        DUAL T plastic_strain() const
        {
            return deformable
                .map([](const auto& d) { return d.plastic_strain; })
                .unwrap_or(T(0));
        }

        // rigid
        DUAL T inertia() const
        {
            return rigid.map([](const auto& r) { return r.inertia; }
            ).unwrap_or(T(0));
        }
    };
}   // namespace simbi::ibsystem

#endif
