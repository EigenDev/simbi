#ifndef IO_SERIAL_BODY_HPP
#define IO_SERIAL_BODY_HPP

#include "compat.hpp"
#include "io/h5_serializable.hpp"
#include "io/write_policy.hpp"
#include "physics/ib/body.hpp"
#include "physics/ib/collection.hpp"
#include "utility/enums.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace simbi::io {

    // =========================================================================
    // h5_serializable specialization for body_collection_t
    // =========================================================================
    template <std::uint64_t Rank, std::uint64_t MaxBodies>
    struct h5_serializable<body::body_collection_t<Rank, MaxBodies>> {
        using collection_t = body::body_collection_t<Rank, MaxBodies>;
        static constexpr std::string_view group_name = "bodies";

        static void write(
            H5::Group& parent,
            const collection_t& bodies,
            const write_policy_t& policy
        )
        {
            auto g = parent.createGroup(std::string(group_name));

            // collection metadata
            write_attribute(g, "count", bodies.size_);
            write_attribute(g, "system_name", bodies.system_name_);
            write_attribute(g, "reference_frame", bodies.reference_frame_);

            // binary parameters if present
            if (bodies.binary_params_) {
                write_binary_params(g, *bodies.binary_params_, policy);
            }

            // individual bodies
            for (std::size_t ii = 0; ii < bodies.size_; ++ii) {
                auto body_group = g.createGroup("body_" + std::to_string(ii));
                std::visit(
                    [&](const auto& body) {
                        write_body(body_group, body, policy);
                    },
                    bodies.bodies_[ii]
                );
            }
        }

        static collection_t read(const H5::Group& parent)
        {
            auto g = parent.openGroup(std::string(group_name));

            collection_t bodies;
            bodies.size_        = read_attribute<std::size_t>(g, "count");
            bodies.system_name_ = read_attribute<std::string>(g, "system_name");
            bodies.reference_frame_ =
                read_attribute<std::string>(g, "reference_frame");

            // binary parameters
            if (group_exists(g, "binary_params")) {
                bodies.binary_params_ = read_binary_params(g);
            }

            // individual bodies
            for (std::size_t ii = 0; ii < bodies.size_; ++ii) {
                auto body_group    = g.openGroup("body_" + std::to_string(ii));
                bodies.bodies_[ii] = read_body_variant(body_group);
            }

            return bodies;
        }

      private:
        // ---------------------------------------------------------------------
        // binary parameters
        // ---------------------------------------------------------------------
        static void write_binary_params(
            H5::Group& parent,
            const body::binary_parameters_t& params,
            const write_policy_t& /*policy*/
        )
        {
            auto g = parent.createGroup("binary_params");

            write_attribute(g, "total_mass", params.total_mass);
            write_attribute(g, "semi_major", params.semi_major);
            write_attribute(g, "eccentricity", params.eccentricity);
            write_attribute(g, "mass_ratio", params.mass_ratio);
            write_attribute(g, "orbital_period", params.orbital_period);
            write_attribute(g, "is_circular_orbit", params.is_circular_orbit);
            write_attribute(g, "prescribed_motion", params.prescribed_motion);
        }

        static body::binary_parameters_t
        read_binary_params(const H5::Group& parent)
        {
            auto g = parent.openGroup("binary_params");

            body::binary_parameters_t params;
            params.total_mass     = read_attribute<real>(g, "total_mass");
            params.semi_major     = read_attribute<real>(g, "semi_major");
            params.eccentricity   = read_attribute<real>(g, "eccentricity");
            params.mass_ratio     = read_attribute<real>(g, "mass_ratio");
            params.orbital_period = read_attribute<real>(g, "orbital_period");
            params.is_circular_orbit =
                read_attribute<bool>(g, "is_circular_orbit");
            params.prescribed_motion =
                read_attribute<bool>(g, "prescribed_motion");

            return params;
        }

        // ---------------------------------------------------------------------
        // generic body serialization
        // ---------------------------------------------------------------------
        template <typename Body>
        static void
        write_body(H5::Group& g, const Body& body, const write_policy_t& policy)
        {
            // core properties
            write_attribute(g, "idx", body.idx);
            write_attribute(g, "mass", body.mass);
            write_attribute(g, "radius", body.radius);
            write_attribute(g, "two_way_coupling", body.two_way_coupling);
            write_attribute(g, "capabilities", body.caps());

            // position/velocity
            std::vector<real> pos(body.position.begin(), body.position.end());
            std::vector<real> vel(body.velocity.begin(), body.velocity.end());
            std::vector<real> force(body.force.begin(), body.force.end());
            std::vector<real> torque(body.torque.begin(), body.torque.end());

            std::vector<hsize_t> dims_vec{Rank};
            std::vector<hsize_t> dims_3{3};

            write_dataset(g, "position", pos, dims_vec, policy);
            write_dataset(g, "velocity", vel, dims_vec, policy);
            write_dataset(g, "force", force, dims_vec, policy);
            write_dataset(g, "torque", torque, dims_3, policy);

            // capability-specific data
            write_capability_data(g, body, policy);
        }

        template <typename Body>
        static void write_capability_data(
            H5::Group& g,
            const Body& body,
            const write_policy_t& /*policy*/
        )
        {
            using namespace body::capabilities;

            if constexpr (Body::template has_capability_v<gravitational_tag>) {
                auto grav = body::get_capabilities<gravitational_tag>(body);
                auto cg   = g.createGroup("gravitational");
                write_attribute(cg, "softening_length", grav.softening_length);
            }

            if constexpr (Body::template has_capability_v<accretion_tag>) {
                auto accr = body::get_capabilities<accretion_tag>(body);
                auto ca   = g.createGroup("accretion");
                write_attribute(ca, "sink_rate", accr.sink_rate);
                write_attribute(ca, "accretion_radius", accr.accretion_radius);
                write_attribute(
                    ca,
                    "total_accreted_mass",
                    accr.total_accreted_mass
                );
                write_attribute(ca, "accretion_rate", accr.accretion_rate);
                write_attribute(ca, "sink_delta", accr.sink_delta);
            }

            if constexpr (Body::template has_capability_v<rigid_tag>) {
                auto rigid = body::get_capabilities<rigid_tag>(body);
                auto cr    = g.createGroup("rigid");
                write_attribute(cr, "inertia", rigid.inertia);
                write_attribute(cr, "apply_no_slip", rigid.apply_no_slip);
            }

            if constexpr (Body::template has_capability_v<elastic_tag>) {
                auto elastic = body::get_capabilities<elastic_tag>(body);
                auto ce      = g.createGroup("elastic");
                write_attribute(ce, "elastic_modulus", elastic.elastic_modulus);
                write_attribute(ce, "poisson_ratio", elastic.poisson_ratio);
            }

            if constexpr (Body::template has_capability_v<deformable_tag>) {
                auto deform = body::get_capabilities<deformable_tag>(body);
                auto cd     = g.createGroup("deformable");
                write_attribute(cd, "yield_stress", deform.yield_stress);
                write_attribute(cd, "plastic_strain", deform.plastic_strain);
            }
        }

        // ---------------------------------------------------------------------
        // body deserialization
        // ---------------------------------------------------------------------
        static body::body_variant_t<Rank> read_body_variant(const H5::Group& g)
        {
            auto caps = read_attribute<std::uint32_t>(g, "capabilities");

            // determine body type from capabilities
            bool has_grav = (caps & static_cast<std::uint32_t>(
                                        body_capability_t::GRAVITATIONAL
                                    )) != 0;
            bool has_accr =
                (caps &
                 static_cast<std::uint32_t>(body_capability_t::ACCRETION)) != 0;
            bool has_rigid =
                (caps & static_cast<std::uint32_t>(body_capability_t::RIGID)) !=
                0;

            // match to known body types
            if (has_grav && has_accr) {
                return read_body<body::black_hole_t<Rank>>(g);
            }
            else if (has_grav && has_rigid) {
                return read_body<body::planet_t<Rank>>(g);
            }
            else if (has_grav) {
                return read_body<body::gravitational_body_t<Rank>>(g);
            }
            else if (has_rigid) {
                return read_body<body::rigid_sphere_t<Rank>>(g);
            }
            else {
                return read_body<body::passive_body_t<Rank>>(g);
            }
        }

        template <typename Body>
        static Body read_body(const H5::Group& g)
        {
            Body body;

            // core properties
            body.idx              = read_attribute<std::uint64_t>(g, "idx");
            body.mass             = read_attribute<real>(g, "mass");
            body.radius           = read_attribute<real>(g, "radius");
            body.two_way_coupling = read_attribute<bool>(g, "two_way_coupling");

            // position/velocity
            auto pos    = read_dataset<real>(g, "position");
            auto vel    = read_dataset<real>(g, "velocity");
            auto force  = read_dataset<real>(g, "force");
            auto torque = read_dataset<real>(g, "torque");

            for (std::size_t dd = 0; dd < Rank; ++dd) {
                body.position[dd] = pos[dd];
                body.velocity[dd] = vel[dd];
                body.force[dd]    = force[dd];
            }
            for (std::size_t dd = 0; dd < 3; ++dd) {
                body.torque[dd] = torque[dd];
            }

            // capability-specific data
            read_capability_data(g, body);

            return body;
        }

        template <typename Body>
        static void read_capability_data(const H5::Group& g, Body& body)
        {
            using namespace body::capabilities;

            if constexpr (Body::template has_capability_v<gravitational_tag>) {
                if (group_exists(g, "gravitational")) {
                    auto cg = g.openGroup("gravitational");
                    auto& grav =
                        std::get<body::grav_component_t>(body.capabilities);
                    grav.softening_length =
                        read_attribute<real>(cg, "softening_length");
                }
            }

            if constexpr (Body::template has_capability_v<accretion_tag>) {
                if (group_exists(g, "accretion")) {
                    auto ca    = g.openGroup("accretion");
                    auto& accr = std::get<body::accretion_component_t>(
                        body.capabilities
                    );
                    accr.sink_rate = read_attribute<real>(ca, "sink_rate");
                    accr.accretion_radius =
                        read_attribute<real>(ca, "accretion_radius");
                    accr.total_accreted_mass =
                        read_attribute<real>(ca, "total_accreted_mass");
                    accr.accretion_rate =
                        read_attribute<real>(ca, "accretion_rate");
                    accr.sink_delta = read_attribute<real>(ca, "sink_delta");
                }
            }

            if constexpr (Body::template has_capability_v<rigid_tag>) {
                if (group_exists(g, "rigid")) {
                    auto cr = g.openGroup("rigid");
                    auto& rigid =
                        std::get<body::rigid_component_t>(body.capabilities);
                    rigid.inertia = read_attribute<real>(cr, "inertia");
                    rigid.apply_no_slip =
                        read_attribute<bool>(cr, "apply_no_slip");
                }
            }

            if constexpr (Body::template has_capability_v<elastic_tag>) {
                if (group_exists(g, "elastic")) {
                    auto ce = g.openGroup("elastic");
                    auto& elastic =
                        std::get<body::elastic_component_t>(body.capabilities);
                    elastic.elastic_modulus =
                        read_attribute<real>(ce, "elastic_modulus");
                    elastic.poisson_ratio =
                        read_attribute<real>(ce, "poisson_ratio");
                }
            }

            if constexpr (Body::template has_capability_v<deformable_tag>) {
                if (group_exists(g, "deformable")) {
                    auto cd      = g.openGroup("deformable");
                    auto& deform = std::get<body::deformable_component_t>(
                        body.capabilities
                    );
                    deform.yield_stress =
                        read_attribute<real>(cd, "yield_stress");
                    deform.plastic_strain =
                        read_attribute<real>(cd, "plastic_strain");
                }
            }
        }
    };

}   // namespace simbi::io

#endif   // IO_SERIAL_BODY_HPP
