#ifndef IO_SERIAL_MESH_HPP
#define IO_SERIAL_MESH_HPP

#include "compat.hpp"
#include "geometry/api.hpp"
#include "grid/boundary.hpp"
#include "grid/mesh_config.hpp"
#include "io/h5_serializable.hpp"
#include "io/write_policy.hpp"
#include "utility/bimap.hpp"

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

namespace simbi::io {

    // =========================================================================
    // h5_serializable specialization for mesh_config_t
    // =========================================================================
    template <std::uint64_t Rank>
    struct h5_serializable<grid::mesh_config_t<Rank>>
    {
        static constexpr std::string_view group_name = "mesh";

        static void write(
            H5::Group&                       parent,
            const grid::mesh_config_t<Rank>& mesh,
            const write_policy_t&            policy
        )
        {
            auto g = parent.createGroup(std::string(group_name));

            // topology
            std::vector<std::int64_t> global_cells(
                mesh.global_cells.begin(),
                mesh.global_cells.end()
            );
            std::vector<std::int64_t> block_size(mesh.block_size.begin(), mesh.block_size.end());
            std::vector<hsize_t>      dims{Rank};
            write_dataset(g, "global_cells", global_cells, dims, policy);
            write_dataset(g, "block_size", block_size, dims, policy);

            // ghost width
            write_attribute(g, "halo_width", mesh.halo_width);

            // geometry config
            write_geometry_config(g, mesh.geometry, policy);

            // motion config
            write_motion_config(g, mesh.motion, policy);

            // boundaries
            write_boundaries(g, mesh.boundaries, policy);
        }

        static grid::mesh_config_t<Rank> read(const H5::Group& parent)
        {
            auto g = parent.openGroup(std::string(group_name));

            grid::mesh_config_t<Rank> mesh;

            // topology
            auto global_cells = read_dataset<std::int64_t>(g, "global_cells");
            auto block_size   = read_dataset<std::int64_t>(g, "block_size");
            for (std::size_t ii = 0; ii < Rank; ++ii) {
                mesh.global_cells[ii] = global_cells[ii];
                mesh.block_size[ii]   = block_size[ii];
            }

            // ghost width
            mesh.halo_width = read_attribute<std::int64_t>(g, "halo_width");

            // geometry config
            mesh.geometry = read_geometry_config(g);

            // motion config
            mesh.motion = read_motion_config(g);

            // boundaries
            mesh.boundaries = read_boundaries(g);

            return mesh;
        }

      private:
        static void write_geometry_config(
            H5::Group&                               parent,
            const geometry::geometry_config_t<Rank>& geo,
            const write_policy_t&                    policy
        )
        {
            auto g = parent.createGroup("geometry");

            // use string serialization for metric type
            write_attribute(g, "metric", serialize(geo.metric));

            // dimension configs
            for (std::size_t dd = 0; dd < geo.dims.size(); ++dd) {
                auto dg = g.createGroup("dim_" + std::to_string(dd));
                write_attribute(dg, "type", serialize(geo.dims[dd].type));
                write_attribute(dg, "start", geo.dims[dd].start);
                write_attribute(dg, "end", geo.dims[dd].end);
            }

            // block size cells
            std::vector<std::int64_t> block_cells(
                geo.block_size_cells.begin(),
                geo.block_size_cells.end()
            );
            std::vector<hsize_t> dims{Rank};
            write_dataset(g, "block_size_cells", block_cells, dims, policy);
        }

        static geometry::geometry_config_t<Rank> read_geometry_config(const H5::Group& parent)
        {
            auto g = parent.openGroup("geometry");

            geometry::geometry_config_t<Rank> geo;

            // try string first, fall back to int for backward compatibility
            try {
                auto metric_str = read_attribute<std::string>(g, "metric");
                geo.metric      = deserialize<geometry::metric_type_t>(metric_str);
            }
            catch (...) {
                geo.metric = static_cast<geometry::metric_type_t>(read_attribute<int>(g, "metric"));
            }

            // dimension configs
            geo.dims.resize(Rank);
            for (std::size_t dd = 0; dd < Rank; ++dd) {
                auto dg = g.openGroup("dim_" + std::to_string(dd));

                // try string first, fall back to int
                try {
                    auto type_str     = read_attribute<std::string>(dg, "type");
                    geo.dims[dd].type = deserialize<geometry::map_type_t>(type_str);
                }
                catch (...) {
                    geo.dims[dd].type =
                        static_cast<geometry::map_type_t>(read_attribute<int>(dg, "type"));
                }

                geo.dims[dd].start = read_attribute<real>(dg, "start");
                geo.dims[dd].end   = read_attribute<real>(dg, "end");
            }

            // block size cells
            auto block_cells = read_dataset<std::int64_t>(g, "block_size_cells");
            for (std::size_t ii = 0; ii < Rank; ++ii) {
                geo.block_size_cells[ii] = block_cells[ii];
            }

            return geo;
        }

        static void write_motion_config(
            H5::Group&                   parent,
            const grid::motion_config_t& motion,
            const write_policy_t& /*policy*/
        )
        {
            auto g = parent.createGroup("motion");

            write_attribute(g, "enabled", motion.enabled);
            write_attribute(g, "homologous", motion.homologous);
            write_attribute(g, "scale_factor_0", motion.scale_factor_0);
        }

        static grid::motion_config_t read_motion_config(const H5::Group& parent)
        {
            auto g = parent.openGroup("motion");

            grid::motion_config_t motion;
            motion.enabled        = read_attribute<bool>(g, "enabled");
            motion.homologous     = read_attribute<bool>(g, "homologous");
            motion.scale_factor_0 = read_attribute<real>(g, "scale_factor_0");

            return motion;
        }

        static void write_boundaries(
            H5::Group&                        parent,
            const grid::boundary_set_t<Rank>& boundaries,
            const write_policy_t& /*policy*/
        )
        {
            auto g = parent.createGroup("boundaries");

            // write boundary rules as string pairs per dimension
            for (std::size_t dd = 0; dd < Rank; ++dd) {
                auto dim_group = g.createGroup("dim_" + std::to_string(dd));
                write_attribute(dim_group, "left", serialize(boundaries.rules[dd].first));
                write_attribute(dim_group, "right", serialize(boundaries.rules[dd].second));
            }
        }

        static grid::boundary_set_t<Rank> read_boundaries(const H5::Group& parent)
        {
            auto g = parent.openGroup("boundaries");

            grid::boundary_set_t<Rank> boundaries;

            // try new format first (string pairs per dimension)
            for (std::size_t dd = 0; dd < Rank; ++dd) {
                auto dim_group = g.openGroup("dim_" + std::to_string(dd));
                auto left_str  = read_attribute<std::string>(dim_group, "left");
                auto right_str = read_attribute<std::string>(dim_group, "right");

                boundaries.rules[dd].first  = deserialize<grid::boundary_type_t>(left_str);
                boundaries.rules[dd].second = deserialize<grid::boundary_type_t>(right_str);
            }

            return boundaries;
        }
    };

} // namespace simbi::io

#endif // IO_SERIAL_MESH_HPP
