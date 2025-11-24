#ifndef IO_SERIAL_HYDRO_HPP
#define IO_SERIAL_HYDRO_HPP

#include "compat.hpp"
#include "containers/vector.hpp"
#include "ecs/components.hpp"
#include "field_serial.hpp"
#include "grid/field.hpp"
#include "io/h5_serializable.hpp"
#include "io/write_policy.hpp"

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string>
#include <string_view>

namespace simbi::io {

    // =========================================================================
    // h5_serializable specialization for partition_fields_t
    // =========================================================================
    template <typename Conserved, typename Primitive, std::uint64_t Rank>
    struct h5_serializable<
        ecs::partition_fields_t<Conserved, Primitive, Rank>> {
        using fields_t = ecs::partition_fields_t<Conserved, Primitive, Rank>;
        static constexpr bool is_mhd = requires(Primitive p) { p.mag; };
        static constexpr std::string_view group_name = "hydro";

        static void write(
            H5::Group& parent,
            const fields_t& fields,
            const write_policy_t& policy
        )
        {
            auto g = parent.createGroup(std::string(group_name));

            // write primitive field with domain
            write_primitives(g, fields.prim, policy);

            // write conserved field with domain
            // write_conserved(g, fields.cons, policy);

            // write magnetic fields if present (mhd)
            if constexpr (is_mhd) {
                write_magnetic_fields(g, fields.bfield, policy);
            }
        }

        static fields_t read(const H5::Group& parent)
        {
            auto g = parent.openGroup(std::string(group_name));

            fields_t fields;

            // read primitives with domain
            fields.prim = read_primitives(g);

            // read conserved with domain
            fields.cons = read_conserved(g);

            // read magnetic fields if present
            if (group_exists(g, "magnetic")) {
                fields.bfield = read_magnetic_fields(g);
            }

            return fields;
        }

      private:
        // ---------------------------------------------------------------------
        // primitive field serialization
        // ---------------------------------------------------------------------
        static void write_primitives(
            H5::Group& parent,
            const grid::field_t<Primitive, Rank>& prim,
            const write_policy_t& policy
        )
        {
            write_struct_field(parent, "primitives", prim, policy)
                .component("rho", [](const auto& p) { return p.rho; })
                .component("pre", [](const auto& p) { return p.pre; })
                .component("chi", [](const auto& p) { return p.chi; });

            // velocity components
            auto g = parent.openGroup("primitives");
            for (std::uint64_t dd = 0; dd < Rank; ++dd) {
                write_field_component(
                    g,
                    "v" + std::to_string(dd + 1),
                    prim,
                    [dd](const auto& p) { return p.vel[dd]; },
                    policy
                );
            }

            // magnetic field components for mhd primitives
            if constexpr (requires(Primitive p) { p.mag; }) {
                for (std::uint64_t dd = 0; dd < Rank; ++dd) {
                    const auto lidx = Rank - 1 - dd;
                    write_field_component(
                        g,
                        "b" + std::to_string(dd + 1) + "_mean",
                        prim,
                        [lidx](const auto& p) { return p.mag[lidx]; },
                        policy
                    );
                }
            }
        }

        static grid::field_t<Primitive, Rank>
        read_primitives(const H5::Group& parent)
        {
            return read_struct_field_with<Primitive, Rank>(
                parent,
                "primitives",
                [](const H5::Group& g, grid::field_t<Primitive, Rank>& field) {
                    read_field_component(
                        g,
                        "density",
                        field,
                        [](auto& p, real v) { p.rho = v; }
                    );
                    read_field_component(
                        g,
                        "pressure",
                        field,
                        [](auto& p, real v) { p.pre = v; }
                    );

                    // chi if present
                    if (group_exists(g, "chi")) {
                        read_field_component(
                            g,
                            "chi",
                            field,
                            [](auto& p, real v) { p.chi = v; }
                        );
                    }

                    // velocity components
                    for (std::uint64_t dd = 0; dd < Rank; ++dd) {
                        read_field_component(
                            g,
                            "v" + std::to_string(dd + 1),
                            field,
                            [dd](auto& p, real v) { p.vel[dd] = v; }
                        );
                    }

                    // magnetic field components for mhd primitives
                    if constexpr (requires(Primitive p) { p.mag; }) {
                        for (std::uint64_t dd = 0; dd < Rank; ++dd) {
                            const auto lidx = Rank - 1 - dd;
                            std::string name =
                                "b" + std::to_string(dd + 1) + "_mean";
                            if (group_exists(g, name)) {
                                read_field_component(
                                    g,
                                    name,
                                    field,
                                    [lidx](auto& p, real v) { p.mag[lidx] = v; }
                                );
                            }
                        }
                    }
                }
            );
        }

        // ---------------------------------------------------------------------
        // conserved field serialization
        // ---------------------------------------------------------------------
        static void write_conserved(
            H5::Group& parent,
            const grid::field_t<Conserved, Rank>& cons,
            const write_policy_t& policy
        )
        {
            write_struct_field(parent, "conserved", cons, policy)
                .component("D", [](const auto& u) { return u.den; })
                .component("tau", [](const auto& u) { return u.nrg; })
                .component("D_chi", [](const auto& u) { return u.chi; });

            // momentum components
            auto g = parent.openGroup("conserved");
            for (std::uint64_t dd = 0; dd < Rank; ++dd) {
                write_field_component(
                    g,
                    "S" + std::to_string(dd + 1),
                    cons,
                    [dd](const auto& u) { return u.mom[dd]; },
                    policy
                );
            }

            // magnetic field components for mhd conserved
            if constexpr (requires(Conserved u) { u.mag; }) {
                for (std::uint64_t dd = 0; dd < Rank; ++dd) {
                    const auto lidx = Rank - 1 - dd;
                    write_field_component(
                        g,
                        "B" + std::to_string(dd + 1) + "_mean",
                        cons,
                        [lidx](const auto& u) { return u.mag[lidx]; },
                        policy
                    );
                }
            }
        }

        static grid::field_t<Conserved, Rank>
        read_conserved(const H5::Group& parent)
        {
            return read_struct_field_with<Conserved, Rank>(
                parent,
                "conserved",
                [](const H5::Group& g, grid::field_t<Conserved, Rank>& field) {
                    read_field_component(g, "D", field, [](auto& u, real v) {
                        u.den = v;
                    });
                    read_field_component(g, "tau", field, [](auto& u, real v) {
                        u.nrg = v;
                    });

                    // chi if present
                    if (group_exists(g, "D_chi")) {
                        read_field_component(
                            g,
                            "D_chi",
                            field,
                            [](auto& u, real v) { u.chi = v; }
                        );
                    }

                    // momentum components
                    for (std::uint64_t dd = 0; dd < Rank; ++dd) {
                        read_field_component(
                            g,
                            "S" + std::to_string(dd + 1),
                            field,
                            [dd](auto& u, real v) { u.mom[dd] = v; }
                        );
                    }

                    // magnetic field components for mhd conserved
                    if constexpr (requires(Conserved u) { u.mag; }) {
                        for (std::uint64_t dd = 0; dd < Rank; ++dd) {
                            std::string name =
                                "B" + std::to_string(dd + 1) + "_mean";
                            if (group_exists(g, name)) {
                                const auto lidx = Rank - 1 - dd;
                                read_field_component(
                                    g,
                                    name,
                                    field,
                                    [lidx](auto& u, real v) { u.mag[lidx] = v; }
                                );
                            }
                        }
                    }
                }
            );
        }

        // ---------------------------------------------------------------------
        // magnetic field serialization (staggered face-centered fields)
        // ---------------------------------------------------------------------
        static void write_magnetic_fields(
            H5::Group& parent,
            const vector_t<grid::field_t<real, Rank>, Rank>& bfield,
            const write_policy_t& policy
        )
        {
            // check if any magnetic field data exists
            bool has_bfield = false;
            for (std::uint64_t dd = 0; dd < Rank; ++dd) {
                if (bfield[dd].domain().size() > 0) {
                    has_bfield = true;
                    break;
                }
            }

            if (!has_bfield) {
                return;
            }

            auto g = parent.createGroup("magnetic");

            for (std::uint64_t dd = 0; dd < Rank; ++dd) {
                const auto lidx = Rank - 1 - dd;
                if (bfield[lidx].domain().size() == 0) {
                    continue;
                }

                write_scalar_field(
                    g,
                    "B" + std::to_string(dd + 1),
                    bfield[lidx],
                    policy
                );
            }
        }

        static vector_t<grid::field_t<real, Rank>, Rank>
        read_magnetic_fields(const H5::Group& parent)
        {
            auto g = parent.openGroup("magnetic");

            vector_t<grid::field_t<real, Rank>, Rank> bfield;

            for (std::uint64_t dd = 0; dd < Rank; ++dd) {
                std::string name = "B" + std::to_string(dd + 1);
                if (!group_exists(g, name)) {
                    continue;
                }
                const auto lidx = Rank - 1 - dd;

                bfield[lidx] = read_scalar_field<Rank>(g, name);
            }

            return bfield;
        }
    };

}   // namespace simbi::io

#endif   // IO_SERIAL_HYDRO_HPP
