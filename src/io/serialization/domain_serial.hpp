// =============================================================================
// domain_serial.hpp
//
// hdf5 serialization for grid domains.
// provides the `h5_serializable` specialization for `grid::domain_t`,
// allowing grid domains to be written to and read from hdf5 files by storing
// their `start` and `fin` coordinates as datasets.
//
// usage:
//   h5_serializable<domain_t<3>>::write(group, domain, policy);
//   auto domain = h5_serializable<domain_t<3>>::read(group);
// =============================================================================
#pragma once

#include "grid/domain.hpp"
#include "io/h5_serializable.hpp"
#include "io/write_policy.hpp"

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace simbi::io {

    template <std::uint64_t Rank>
    struct h5_serializable<grid::domain_t<Rank>>
    {
        static constexpr std::string_view group_name = "domain";

        static void
        write(H5::Group& parent, const grid::domain_t<Rank>& domain, const write_policy_t& policy)
        {
            auto g = parent.createGroup(std::string(group_name));

            std::vector<std::int64_t> start_vec(domain.start.begin(), domain.start.end());
            std::vector<std::int64_t> fin_vec(domain.fin.begin(), domain.fin.end());

            std::vector<hsize_t> dims{Rank};
            write_dataset(g, "start", start_vec, dims, policy);
            write_dataset(g, "fin", fin_vec, dims, policy);
        }

        static grid::domain_t<Rank> read(const H5::Group& parent)
        {
            auto g = parent.openGroup(std::string(group_name));

            auto start_vec = read_dataset<std::int64_t>(g, "start");
            auto fin_vec   = read_dataset<std::int64_t>(g, "fin");

            grid::domain_t<Rank> domain;
            for (std::uint64_t ii = 0; ii < Rank; ++ii) {
                domain.start[ii] = start_vec[ii];
                domain.fin[ii]   = fin_vec[ii];
            }

            return domain;
        }

        // named variants for embedding in other groups
        static void write_named(
            H5::Group&                  parent,
            const std::string&          name,
            const grid::domain_t<Rank>& domain,
            const write_policy_t&       policy
        )
        {
            auto g = parent.createGroup(name);

            std::vector<std::int64_t> start_vec(domain.start.begin(), domain.start.end());
            std::vector<std::int64_t> fin_vec(domain.fin.begin(), domain.fin.end());

            std::vector<hsize_t> dims{Rank};
            write_dataset(g, "start", start_vec, dims, policy);
            write_dataset(g, "fin", fin_vec, dims, policy);
        }

        static grid::domain_t<Rank> read_named(const H5::Group& parent, const std::string& name)
        {
            auto g = parent.openGroup(name);

            auto start_vec = read_dataset<std::int64_t>(g, "start");
            auto fin_vec   = read_dataset<std::int64_t>(g, "fin");

            grid::domain_t<Rank> domain;
            for (std::uint64_t ii = 0; ii < Rank; ++ii) {
                domain.start[ii] = start_vec[ii];
                domain.fin[ii]   = fin_vec[ii];
            }

            return domain;
        }
    };

} // namespace simbi::io
