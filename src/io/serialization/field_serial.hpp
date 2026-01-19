// =============================================================================
// field_serial.hpp
//
// hdf5 serialization helpers for grid fields.
// provides utilities for serializing `grid::field_t` objects. it includes
// functions to write scalar fields and component-wise writers/readers for
// struct-based fields, which are used to build the full serialization logic
// for primitive and conserved state variables.
//
// usage:
//   write_scalar_field(group, "my_field", field, policy);
//   write_struct_field(group, "my_struct_field", field, policy)
//       .component("rho", [](const auto& s) { return s.rho; });
// =============================================================================
#pragma once

#include "build_config.hpp"
#include "grid/domain.hpp"
#include "grid/field.hpp"
#include "io/h5_serializable.hpp"
#include "io/write_policy.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace simbi::io {

    // -------------------------------------------------------------------------
    // write a scalar field (field_t<real, Rank>)
    // -------------------------------------------------------------------------
    template <std::uint64_t Rank>
    void write_scalar_field(
        H5::Group&                       parent,
        const std::string&               name,
        const grid::field_t<real, Rank>& field,
        const write_policy_t&            policy
    )
    {
        auto g = parent.createGroup(name);

        // serialize domain
        const auto& domain = field.domain();
        h5_serializable<grid::domain_t<Rank>>::write_named(g, "domain", domain, policy);

        // extract data in row-major order
        auto              shape = domain.shape();
        std::vector<real> data;
        data.reserve(static_cast<std::size_t>(domain.size()));

        // iterate domain in row-major order (matches hdf5 default)
        for (std::uint64_t linear = 0; linear < domain.size(); ++linear) {
            auto coord = domain.linear_to_coord(linear);
            data.push_back(field(coord));
        }

        // write data with shape from domain
        std::vector<hsize_t> dims(shape.begin(), shape.end());
        write_dataset(g, "data", data, dims, policy);
    }

    // -------------------------------------------------------------------------
    // read a scalar field
    // -------------------------------------------------------------------------
    template <std::uint64_t Rank>
    grid::field_t<real, Rank> read_scalar_field(const H5::Group& parent, const std::string& name)
    {
        auto g = parent.openGroup(name);

        // read domain
        auto domain = h5_serializable<grid::domain_t<Rank>>::read_named(g, "domain");

        // read data
        auto data = read_dataset<real>(g, "data");

        // construct field with correct domain
        grid::field_t<real, Rank> field(domain);

        // populate field
        for (std::uint64_t linear = 0; linear < domain.size(); ++linear) {
            auto coord                      = domain.linear_to_coord(linear);
            const_cast<real&>(field(coord)) = data[linear];
        }

        return field;
    }

    // -------------------------------------------------------------------------
    // write a struct field (field_t<Primitive, Rank> or field_t<Conserved,
    // Rank>) extracts each component as a separate dataset
    // -------------------------------------------------------------------------
    template <typename T, std::uint64_t Rank, typename Extractor>
    void write_field_component(
        H5::Group&                    parent,
        const std::string&            name,
        const grid::field_t<T, Rank>& field,
        Extractor&&                   extractor,
        const write_policy_t&         policy
    )
    {
        const auto& domain = field.domain();
        auto        shape  = domain.shape();

        std::vector<real> data;
        data.reserve(static_cast<std::size_t>(domain.size()));

        for (std::uint64_t linear = 0; linear < domain.size(); ++linear) {
            auto coord = domain.linear_to_coord(linear);
            data.push_back(extractor(field(coord)));
        }

        std::vector<hsize_t> dims(shape.begin(), shape.end());
        write_dataset(parent, name, data, dims, policy);
    }

    // -------------------------------------------------------------------------
    // read a field component into an existing field
    // -------------------------------------------------------------------------
    template <typename T, std::uint64_t Rank, typename Inserter>
    void read_field_component(
        const H5::Group&        parent,
        const std::string&      name,
        grid::field_t<T, Rank>& field,
        Inserter&&              inserter
    )
    {
        auto        data   = read_dataset<real>(parent, name);
        const auto& domain = field.domain();

        for (std::uint64_t linear = 0; linear < domain.size(); ++linear) {
            auto coord = domain.linear_to_coord(linear);
            inserter(const_cast<T&>(field(coord)), data[linear]);
        }
    }

    // -------------------------------------------------------------------------
    // write a struct-valued field with all components
    // domain is serialized once, components serialized individually
    // -------------------------------------------------------------------------
    template <typename T, std::uint64_t Rank>
    struct struct_field_writer_t
    {
        H5::Group                     group;
        const grid::field_t<T, Rank>& field;
        const write_policy_t&         policy;

        struct_field_writer_t(H5::Group g, const grid::field_t<T, Rank>& f, const write_policy_t& p)
            : group(std::move(g)), field(f), policy(p)
        {
        }

        template <typename Extractor>
        struct_field_writer_t& component(const std::string& name, Extractor&& extractor)
        {
            write_field_component(group, name, field, std::forward<Extractor>(extractor), policy);
            return *this;
        }
    };

    template <typename T, std::uint64_t Rank>
    struct_field_writer_t<T, Rank> write_struct_field(
        H5::Group&                    parent,
        const std::string&            name,
        const grid::field_t<T, Rank>& field,
        const write_policy_t&         policy
    )
    {
        auto g = parent.createGroup(name);

        // serialize domain once
        h5_serializable<grid::domain_t<Rank>>::write_named(g, "domain", field.domain(), policy);

        return struct_field_writer_t<T, Rank>(g, field, policy);
    }

    // -------------------------------------------------------------------------
    // read a struct-valued field
    // -------------------------------------------------------------------------
    template <typename T, std::uint64_t Rank>
    struct struct_field_reader_t
    {
        H5::Group              group;
        grid::field_t<T, Rank> field;

        struct_field_reader_t(H5::Group g, grid::field_t<T, Rank> f)
            : group(std::move(g)), field(std::move(f))
        {
        }

        template <typename Inserter>
        struct_field_reader_t& component(const std::string& name, Inserter&& inserter)
        {
            read_field_component(group, name, field, std::forward<Inserter>(inserter));
            return *this;
        }

        // extract the field after all components are read
        grid::field_t<T, Rank> take()
        {
            return std::move(field);
        }
    };

    template <typename T, std::uint64_t Rank>
    struct_field_reader_t<T, Rank>
    read_struct_field(const H5::Group& parent, const std::string& name)
    {
        auto g = parent.openGroup(name);

        // read domain
        auto domain = h5_serializable<grid::domain_t<Rank>>::read_named(g, "domain");

        // allocate field with correct domain
        grid::field_t<T, Rank> field(domain);

        return struct_field_reader_t<T, Rank>(std::move(g), std::move(field));
    }

    // simpler version that returns the field directly after building
    template <typename T, std::uint64_t Rank>
    grid::field_t<T, Rank>
    read_struct_field_with(const H5::Group& parent, const std::string& name, auto&& builder)
    {
        auto g = parent.openGroup(name);

        // read domain
        auto domain = h5_serializable<grid::domain_t<Rank>>::read_named(g, "domain");

        // allocate field with correct domain
        grid::field_t<T, Rank> field(domain);

        // let the builder populate it
        builder(g, field);

        return field;
    }

} // namespace simbi::io
