/**
 *=============================================================================
 *           SIMBI - Special Relativistic Magnetohydrodynamics Code
 *=============================================================================
 *
 * @file            body.hpp
 * @brief           body_t class for representing bodies in the IB scheme
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
#include "core/utility/enums.hpp"
#include <cstdint>
#include <type_traits>
#include <utility>

namespace simbi::ibsystem {
    template <std::uint64_t Dims, typename... Caps>
    struct body_t {
        // core properties (always present)
        vector_t<real, Dims> position;
        vector_t<real, Dims> velocity;
        vector_t<real, Dims> force;
        real mass;
        real radius;
        bool two_way_coupling;
        std::uint64_t index;

        // capabilities stored as tuple
        std::tuple<Caps...> capabilities;

        // optional components using maybe_t monad
        maybe_t<grav_component_t> gravitational;
        maybe_t<accretion_component_t> accretion;
        maybe_t<elastic_component_t> elastic;
        maybe_t<rigid_component_t> rigid;
        maybe_t<deformable_component_t> deformable;

        // ctors
        DUAL body_t()
            : position(vector_t<real, Dims>()),
              velocity(vector_t<real, Dims>()),
              force(vector_t<real, Dims>()),
              mass(0),
              radius(0),
              two_way_coupling(false),
              gravitational(Nothing),
              accretion(Nothing),
              elastic(Nothing),
              rigid(Nothing),
              deformable(Nothing),
              index(0)
        {
        }
        DUAL body_t(
            const vector_t<real, Dims>& position,
            const vector_t<real, Dims>& velocity,
            real mass,
            real radius,
            bool two_way_coupling = false
        )
            : position(position),
              velocity(velocity),
              force(vector_t<real, Dims>()),
              mass(mass),
              radius(radius),
              two_way_coupling(two_way_coupling),
              gravitational(Nothing),
              accretion(Nothing),
              elastic(Nothing),
              rigid(Nothing),
              deformable(Nothing),
              index(0)
        {
        }

        // copy ctor
        DUAL constexpr body_t(const body_t& other)
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
              deformable(other.deformable),
              index(other.index)
        {
        }
        // move ctor
        DUAL constexpr body_t(body_t&& other) noexcept
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
              deformable(std::move(other.deformable)),
              index(other.index)
        {
        }
        // copy assignment
        DUAL constexpr body_t& operator=(const body_t& other)
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
                index            = other.index;
            }
            return *this;
        }
        // move assignment
        DUAL constexpr body_t& operator=(body_t&& other) noexcept
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
                index            = other.index;
            }
            return *this;
        }

        DUAL constexpr auto with_gravitational(real softening) const
        {
            body_t<Dims> new_body  = *this;
            new_body.gravitational = grav_component_t{softening};
            return new_body;
        }

        DUAL constexpr auto with_rigid(real inertia, bool apply_no_slip) const
        {
            body_t<Dims> new_body = *this;
            new_body.rigid        = rigid_component_t{inertia, apply_no_slip};
            return new_body;
        }

        DUAL constexpr auto with_accretion(
            real efficiency,
            real accr_radius         = 0.0,
            real total_accreted_mass = 0.0,
            real accr_rate           = 0.0
        ) const
        {
            body_t<Dims> new_body = *this;
            new_body.accretion    = accretion_component_t{
              efficiency,
              accr_radius <= 0 ? radius : accr_radius,
              total_accreted_mass,
              accr_rate
            };
            return new_body;
        }

        DUAL constexpr auto
        with_force(const vector_t<real, Dims>& new_force) const
        {
            body_t<Dims> new_body = *this;
            new_body.force        = new_force;
            return new_body;
        }

        DUAL constexpr auto
        update_position(const vector_t<real, Dims>& new_position) const
        {
            body_t<Dims> new_body = *this;
            new_body.position     = new_position;
            return new_body;
        }

        DUAL constexpr auto
        update_velocity(const vector_t<real, Dims>& new_velocity) const
        {
            body_t<Dims> new_body = *this;
            new_body.velocity     = new_velocity;
            return new_body;
        }

        DUAL constexpr auto add_mass(real added_mass) const
        {
            body_t<Dims> new_body = *this;
            new_body.mass         = mass + added_mass;
            return new_body;
        }

        DUAL constexpr auto add_accreted_mass(real added_mass) const
        {
            body_t<Dims> new_body = *this;
            if (new_body.accretion.has_value()) {
                auto component = new_body.accretion.value();
                component.total_accreted_mass += added_mass;
                new_body.accretion = component;
            }
            return new_body;
        }

        DUAL constexpr auto with_accretion_rate(real accr_rate) const
        {
            body_t<Dims> new_body = *this;
            if (new_body.accretion.has_value()) {
                auto component           = new_body.accretion.value();
                component.accretion_rate = accr_rate;
                new_body.accretion       = component;
            }
            return new_body;
        }

        // query functions
        template <typename Cap>
        static constexpr bool has_capability_v =
            (std::is_same_v<Cap, Caps> || ...);

        template <typename Cap>
        constexpr auto get_capability() const
            -> std::enable_if_t<has_capability_v<Cap>, Cap>
        {
            return std::get<Cap>(capabilities);
        }

        template <typename NewCap, typename... Args>
        DUAL constexpr auto with_capability(Args&&... args) const
        {
            // creates a new body_t type with the additional capability
            return body_t<Dims, Caps..., NewCap>{
              position,
              velocity,
              force,
              mass,
              radius,
              two_way_coupling,
              index,
              std::tuple_cat(
                  capabilities,
                  std::make_tuple(NewCap{std::forward<Args>(args)...})
              )
            };
        }

        // get the capabilities of the body
        // DUAL BodyCapability capabilities() const
        // {
        //     BodyCapability caps = BodyCapability::NONE;
        //     if (gravitational.has_value()) {
        //         caps = static_cast<BodyCapability>(
        //             static_cast<std::int64_t>(caps) |
        //             static_cast<std::int64_t>(BodyCapability::GRAVITATIONAL)
        //         );
        //     }
        //     if (accretion.has_value()) {
        //         caps = static_cast<BodyCapability>(
        //             static_cast<std::int64_t>(caps) |
        //             static_cast<std::int64_t>(BodyCapability::ACCRETION)
        //         );
        //     }
        //     if (rigid.has_value()) {
        //         caps = static_cast<BodyCapability>(
        //             static_cast<std::int64_t>(caps) |
        //             static_cast<std::int64_t>(BodyCapability::RIGID)
        //         );
        //     }
        //     if (elastic.has_value()) {
        //         caps = static_cast<BodyCapability>(
        //             static_cast<std::int64_t>(caps) |
        //             static_cast<std::int64_t>(BodyCapability::ELASTIC)
        //         );
        //     }
        //     if (deformable.has_value()) {
        //         caps = static_cast<BodyCapability>(
        //             static_cast<std::int64_t>(caps) |
        //             static_cast<std::int64_t>(BodyCapability::DEFORMABLE)
        //         );
        //     }
        //     return caps;
        // }

        // access helper functions
        // gravitational
        DUAL auto softening_length() const
        {
            return gravitational
                .map([](const auto& g) { return g.softening_length; })
                .unwrap_or(real(0));
        }

        // accretion
        DUAL real accretion_efficiency() const
        {
            return accretion
                .map([](const auto& a) { return a.accretion_efficiency; })
                .unwrap_or(real(0));
        }

        DUAL real accretion_radius() const
        {
            return accretion
                .map([](const auto& a) { return a.accretion_radius; })
                .unwrap_or(radius);
        }

        DUAL real total_accreted_mass() const
        {
            return accretion
                .map([](const auto& a) { return a.total_accreted_mass; })
                .unwrap_or(real(0));
        }

        DUAL real accretion_rate() const
        {
            return accretion.map(
                                [](const auto& a) { return a.accretion_rate; }
            ).unwrap_or(real(0));
        }

        // elastic
        DUAL real elastic_modulus() const
        {
            return elastic.map(
                              [](const auto& e) { return e.elastic_modulus; }
            ).unwrap_or(real(0));
        }

        DUAL real poisson_ratio() const
        {
            return elastic.map(
                              [](const auto& e) { return e.poisson_ratio; }
            ).unwrap_or(real(0));
        }

        // deformable
        DUAL real yield_stress() const
        {
            return deformable.map(
                                 [](const auto& d) { return d.yield_stress; }
            ).unwrap_or(real(0));
        }

        DUAL real plastic_strain() const
        {
            return deformable
                .map([](const auto& d) { return d.plastic_strain; })
                .unwrap_or(real(0));
        }

        // rigid
        DUAL real inertia() const
        {
            return rigid.map(
                            [](const auto& r) { return r.inertia; }
            ).unwrap_or(real(0));
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
