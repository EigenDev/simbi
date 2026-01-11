#ifndef IO_SERIAL_SKELETON_HPP
#define IO_SERIAL_SKELETON_HPP

// =============================================================================
// skeleton_serial.hpp
//
// serialization for grid skeleton and related types.
// preserves all boundary metadata including metric info for poles.
//
// types serialized:
//   - patch_id_t: block identifier (level, coords)
//   - connection_t: face connectivity with metric info
//   - block_info_t<Rank>: complete block descriptor
//   - skeleton_t<Rank>: full topology map
// =============================================================================

#include "build_config.hpp"
#include "grid/block_info.hpp"
#include "grid/connectivity.hpp"
#include "grid/patch_id.hpp"
#include "grid/skeleton.hpp"
#include "io/h5_serializable.hpp"
#include "io/write_policy.hpp"
#include "utility/bimap.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace simbi::io {

    // =========================================================================
    // h5_serializable specialization for patch_id_t
    // =========================================================================
    template <>
    struct h5_serializable<grid::patch_id_t>
    {
        static constexpr std::string_view group_name = "patch_id";

        static void
        write(H5::Group& parent, const grid::patch_id_t& id, const write_policy_t& policy)
        {
            auto g = parent.createGroup(std::string(group_name));

            write_attribute(g, "level", id.level);

            std::vector<std::int64_t> coords(id.coords.begin(), id.coords.end());
            std::vector<hsize_t>      dims{3};
            write_dataset(g, "coords", coords, dims, policy);
        }

        static grid::patch_id_t read(const H5::Group& parent)
        {
            auto g = parent.openGroup(std::string(group_name));

            grid::patch_id_t id;
            id.level = read_attribute<std::int64_t>(g, "level");

            auto coords = read_dataset<std::int64_t>(g, "coords");
            for (std::size_t ii = 0; ii < 3; ++ii) {
                id.coords[ii] = coords[ii];
            }

            return id;
        }
    };

    // =========================================================================
    // h5_serializable specialization for connection_t
    // =========================================================================
    template <>
    struct h5_serializable<grid::connection_t>
    {
        static constexpr std::string_view group_name = "connection";

        static void
        write(H5::Group& parent, const grid::connection_t& conn, const write_policy_t& policy)
        {
            auto g = parent.createGroup(std::string(group_name));

            write_attribute(g, "type", serialize(conn.type));
            write_attribute(g, "num_neighbors", conn.neighbors.size());

            // write neighbors
            for (std::size_t ii = 0; ii < conn.neighbors.size(); ++ii) {
                auto ng = g.createGroup("neighbor_" + std::to_string(ii));
                h5_serializable<grid::patch_id_t>::write(ng, conn.neighbors[ii], policy);
            }

            // write metric info
            write_attribute(g, "has_metric_info", conn.has_metric_info_);
            if (conn.has_metric_info_) {
                write_attribute(g, "theta_min", conn.theta_min_);
                write_attribute(g, "theta_max", conn.theta_max_);
            }
        }

        static grid::connection_t read(const H5::Group& parent)
        {
            auto g = parent.openGroup(std::string(group_name));

            grid::connection_t conn;

            auto type_str = read_attribute<std::string>(g, "type");
            conn.type     = deserialize<grid::boundary_type_t>(type_str);

            auto num_neighbors = read_attribute<std::size_t>(g, "num_neighbors");
            conn.neighbors.resize(num_neighbors);

            for (std::size_t ii = 0; ii < num_neighbors; ++ii) {
                auto ng            = g.openGroup("neighbor_" + std::to_string(ii));
                conn.neighbors[ii] = h5_serializable<grid::patch_id_t>::read(ng);
            }

            // read metric info
            conn.has_metric_info_ = read_attribute<bool>(g, "has_metric_info");
            if (conn.has_metric_info_) {
                conn.theta_min_ = read_attribute<real>(g, "theta_min");
                conn.theta_max_ = read_attribute<real>(g, "theta_max");
            }

            return conn;
        }
    };

    // =========================================================================
    // h5_serializable specialization for block_info_t<Rank>
    // =========================================================================
    template <std::uint64_t Rank>
    struct h5_serializable<grid::block_info_t<Rank>>
    {
        static constexpr std::string_view group_name = "block_info";

        static void write(
            H5::Group&                      parent,
            const grid::block_info_t<Rank>& block,
            const write_policy_t&           policy
        )
        {
            auto g = parent.createGroup(std::string(group_name));

            // write patch id
            h5_serializable<grid::patch_id_t>::write(g, block.id, policy);

            // write geometry
            auto                      geom_group = g.createGroup("geometry");
            std::vector<std::int64_t> starts(
                block.geometry.start.begin(),
                block.geometry.start.end()
            );
            std::vector<std::int64_t> fins(block.geometry.fin.begin(), block.geometry.fin.end());
            std::vector<hsize_t>      dims{Rank};

            write_dataset(geom_group, "start", starts, dims, policy);
            write_dataset(geom_group, "fin", fins, dims, policy);

            // write faces
            write_attribute(g, "num_faces", block.faces.size());
            for (std::size_t ii = 0; ii < block.faces.size(); ++ii) {
                auto face_group = g.createGroup("face_" + std::to_string(ii));
                h5_serializable<grid::connection_t>::write(face_group, block.faces[ii], policy);
            }
        }

        static grid::block_info_t<Rank> read(const H5::Group& parent)
        {
            auto g = parent.openGroup(std::string(group_name));

            grid::block_info_t<Rank> block;

            // read patch id
            block.id = h5_serializable<grid::patch_id_t>::read(g);

            // read geometry
            auto geom_group = g.openGroup("geometry");
            auto starts     = read_dataset<std::int64_t>(geom_group, "start");
            auto fins       = read_dataset<std::int64_t>(geom_group, "fin");

            for (std::size_t ii = 0; ii < Rank; ++ii) {
                block.geometry.start[ii] = starts[ii];
                block.geometry.fin[ii]   = fins[ii];
            }

            // read faces (faces is vector_t<connection_t, 2*Rank>, fixed size)
            auto num_faces = read_attribute<std::size_t>(g, "num_faces");

            for (std::size_t ii = 0; ii < num_faces && ii < block.faces.size(); ++ii) {
                auto face_group = g.openGroup("face_" + std::to_string(ii));
                block.faces[ii] = h5_serializable<grid::connection_t>::read(face_group);
            }

            return block;
        }
    };

    // =========================================================================
    // h5_serializable specialization for skeleton_t<Rank>
    // =========================================================================
    template <std::uint64_t Rank>
    struct h5_serializable<grid::skeleton_t<Rank>>
    {
        static constexpr std::string_view group_name = "skeleton";

        static void write(
            H5::Group&                    parent,
            const grid::skeleton_t<Rank>& skeleton,
            const write_policy_t&         policy
        )
        {
            auto g = parent.createGroup(std::string(group_name));

            write_attribute(g, "num_blocks", skeleton.size());

            std::size_t block_idx = 0;
            for (const auto& [id, block] : skeleton) {
                auto bg = g.createGroup("block_" + std::to_string(block_idx++));
                h5_serializable<grid::block_info_t<Rank>>::write(bg, block, policy);
            }
        }

        static grid::skeleton_t<Rank> read(const H5::Group& parent)
        {
            auto g = parent.openGroup(std::string(group_name));

            auto num_blocks = read_attribute<std::size_t>(g, "num_blocks");

            grid::skeleton_t<Rank> skeleton;

            for (std::size_t ii = 0; ii < num_blocks; ++ii) {
                auto bg            = g.openGroup("block_" + std::to_string(ii));
                auto block         = h5_serializable<grid::block_info_t<Rank>>::read(bg);
                skeleton[block.id] = block;
            }

            return skeleton;
        }
    };

} // namespace simbi::io

#endif // IO_SERIAL_SKELETON_HPP
