/**
 *=============================================================================
 *           SIMBI - Special Relativistic Magnetohydrodynamics Code
 *=============================================================================
 *
 * @file            body.hpp
 * @brief           Body class for representing bodies in the IB scheme
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
#ifndef BODY_HPP
#define BODY_HPP

#include "capability.hpp"
#include "compute/functional/monad/maybe.hpp"
#include "config.hpp"
#include "containers/vector.hpp"
#include "utility/enums.hpp"
#include <cstdint>

namespace simbi::ibsystem {
    template <typename T, std::uint64_t Dims>
    struct Body {
        // core properties (always present)
        vector_t<T, Dims> position;
        vector_t<T, Dims> velocity;
        vector_t<T, Dims> force;
        T mass;
        T radius;
        bool two_way_coupling;

        // optional components using maybe_t monad
        maybe_t<GravitationalComponent<T>> gravitational;
        maybe_t<AccretionComponent<T>> accretion;
        maybe_t<ElasticComponent<T>> elastic;
        maybe_t<RigidComponent<T>> rigid;
        maybe_t<DeformableComponent<T>> deformable;

        // ctors
        DUAL Body()
            : position(vector_t<T, Dims>()),
              velocity(vector_t<T, Dims>()),
              force(vector_t<T, Dims>()),
              mass(T(0)),
              radius(T(0)),
              two_way_coupling(false),
              gravitational(Nothing),
              accretion(Nothing),
              elastic(Nothing),
              rigid(Nothing),
              deformable(Nothing)
        {
        }
        DUAL Body(
            const vector_t<T, Dims>& position,
            const vector_t<T, Dims>& velocity,
            T mass,
            T radius,
            bool two_way_coupling = false
        )
            : position(position),
              velocity(velocity),
              force(vector_t<T, Dims>()),
              mass(mass),
              radius(radius),
              two_way_coupling(two_way_coupling),
              gravitational(Nothing),
              accretion(Nothing),
              elastic(Nothing),
              rigid(Nothing),
              deformable(Nothing)
        {
        }

        // copy ctor
        DUAL constexpr Body(const Body& other)
            : position(other.position),
              velocity(other.velocity),
              force(other.force),
              mass(other.mass),
              radius(other.radius),
              two_way_coupling(other.two_way_coupling),
              gravitational(other.gravitational),
              accretion(other.accretion),
              elastic(other.elastic),
              rigid(other.rigid),
              deformable(other.deformable)
        {
        }
        // move ctor
        DUAL constexpr Body(Body&& other) noexcept
            : position(std::move(other.position)),
              velocity(std::move(other.velocity)),
              force(std::move(other.force)),
              mass(other.mass),
              radius(other.radius),
              two_way_coupling(other.two_way_coupling),
              gravitational(std::move(other.gravitational)),
              accretion(std::move(other.accretion)),
              elastic(std::move(other.elastic)),
              rigid(std::move(other.rigid)),
              deformable(std::move(other.deformable))
        {
        }
        // copy assignment
        DUAL constexpr Body& operator=(const Body& other)
        {
            if (this != &other) {
                position         = other.position;
                velocity         = other.velocity;
                force            = other.force;
                mass             = other.mass;
                radius           = other.radius;
                two_way_coupling = other.two_way_coupling;
                gravitational    = other.gravitational;
                accretion        = other.accretion;
                elastic          = other.elastic;
                rigid            = other.rigid;
                deformable       = other.deformable;
            }
            return *this;
        }
        // move assignment
        DUAL constexpr Body& operator=(Body&& other) noexcept
        {
            if (this != &other) {
                position         = std::move(other.position);
                velocity         = std::move(other.velocity);
                force            = std::move(other.force);
                mass             = other.mass;
                radius           = other.radius;
                two_way_coupling = other.two_way_coupling;
                gravitational    = std::move(other.gravitational);
                accretion        = std::move(other.accretion);
                elastic          = std::move(other.elastic);
                rigid            = std::move(other.rigid);
                deformable       = std::move(other.deformable);
            }
            return *this;
        }

        DUAL Body<T, Dims> with_gravitational(T softening) const
        {
            Body<T, Dims> new_body = *this;
            new_body.gravitational = GravitationalComponent<T>{softening};
            return new_body;
        }

        DUAL Body<T, Dims> with_rigid(T inertia, bool apply_no_slip) const
        {
            Body<T, Dims> new_body = *this;
            new_body.rigid         = RigidComponent<T>{inertia, apply_no_slip};
            return new_body;
        }

        DUAL Body<T, Dims> with_accretion(
            T efficiency,
            T accr_radius         = 0,
            T total_accreted_mass = 0,
            T accr_rate           = 0
        ) const
        {
            Body<T, Dims> new_body = *this;
            new_body.accretion     = AccretionComponent<T>{
              efficiency,
              accr_radius <= 0 ? radius : accr_radius,
              total_accreted_mass,
              accr_rate
            };
            return new_body;
        }

        DUAL Body<T, Dims> with_force(const vector_t<T, Dims>& new_force) const
        {
            Body<T, Dims> new_body = *this;
            new_body.force         = new_force;
            return new_body;
        }

        DUAL Body<T, Dims>
        update_position(const vector_t<T, Dims>& new_position) const
        {
            Body<T, Dims> new_body = *this;
            new_body.position      = new_position;
            return new_body;
        }

        DUAL Body<T, Dims>
        update_velocity(const vector_t<T, Dims>& new_velocity) const
        {
            Body<T, Dims> new_body = *this;
            new_body.velocity      = new_velocity;
            return new_body;
        }

        DUAL Body<T, Dims> add_mass(T added_mass) const
        {
            Body<T, Dims> new_body = *this;
            new_body.mass          = mass + added_mass;
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

        DUAL Body<T, Dims> with_accretion_rate(T accr_rate) const
        {
            Body<T, Dims> new_body = *this;
            if (new_body.accretion.has_value()) {
                auto component           = new_body.accretion.value();
                component.accretion_rate = accr_rate;
                new_body.accretion       = component;
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
                case BodyCapability::RIGID: return rigid.has_value();
                case BodyCapability::ELASTIC: return elastic.has_value();
                case BodyCapability::DEFORMABLE: return deformable.has_value();
                // add more capabilities as needed
                default: return false;
            }
        }

        // get the capabilities of the body
        DUAL BodyCapability capabilities() const
        {
            BodyCapability caps = BodyCapability::NONE;
            if (gravitational.has_value()) {
                caps = static_cast<BodyCapability>(
                    static_cast<std::int64_t>(caps) |
                    static_cast<std::int64_t>(BodyCapability::GRAVITATIONAL)
                );
            }
            if (accretion.has_value()) {
                caps = static_cast<BodyCapability>(
                    static_cast<std::int64_t>(caps) |
                    static_cast<std::int64_t>(BodyCapability::ACCRETION)
                );
            }
            if (rigid.has_value()) {
                caps = static_cast<BodyCapability>(
                    static_cast<std::int64_t>(caps) |
                    static_cast<std::int64_t>(BodyCapability::RIGID)
                );
            }
            if (elastic.has_value()) {
                caps = static_cast<BodyCapability>(
                    static_cast<std::int64_t>(caps) |
                    static_cast<std::int64_t>(BodyCapability::ELASTIC)
                );
            }
            if (deformable.has_value()) {
                caps = static_cast<BodyCapability>(
                    static_cast<std::int64_t>(caps) |
                    static_cast<std::int64_t>(BodyCapability::DEFORMABLE)
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
            return accretion.map(
                                [](const auto& a) { return a.accretion_rate; }
            ).unwrap_or(T(0));
        }

        // elastic
        DUAL T elastic_modulus() const
        {
            return elastic.map(
                              [](const auto& e) { return e.elastic_modulus; }
            ).unwrap_or(T(0));
        }

        DUAL T poisson_ratio() const
        {
            return elastic.map(
                              [](const auto& e) { return e.poisson_ratio; }
            ).unwrap_or(T(0));
        }

        // deformable
        DUAL T yield_stress() const
        {
            return deformable.map(
                                 [](const auto& d) { return d.yield_stress; }
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
            return rigid.map(
                            [](const auto& r) { return r.inertia; }
            ).unwrap_or(T(0));
        }

        DUAL bool apply_no_slip() const
        {
            return rigid.map(
                            [](const auto& r) { return r.apply_no_slip; }
            ).unwrap_or(false);
        }
    };
}   // namespace simbi::ibsystem

#endif
