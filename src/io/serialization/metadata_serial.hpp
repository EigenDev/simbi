#ifndef IO_SERIAL_METADATA_HPP
#define IO_SERIAL_METADATA_HPP

#include "build_config.hpp"
#include "ecs/components.hpp"
#include "grid/boundary.hpp"
#include "io/h5_serializable.hpp"
#include "io/write_policy.hpp"
#include "utility/bimap.hpp"
#include "utility/enums.hpp"

#include <H5Cpp.h>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace simbi::io {

    // =========================================================================
    // helper for reading enums with backward compatibility
    // =========================================================================
    template <typename EnumType>
    EnumType read_enum_attribute(const H5::Group& g, const std::string& name)
    {
        // try string first (new format)
        try {
            auto str = read_attribute<std::string>(g, name);
            return deserialize<EnumType>(str);
        }
        catch (...) {
            // fall back to int (old format)
            return static_cast<EnumType>(read_attribute<int>(g, name));
        }
    }

    // =========================================================================
    // h5_serializable specialization for simulation_metadata_t
    // =========================================================================
    template <std::uint64_t Rank>
    struct h5_serializable<ecs::simulation_metadata_t<Rank>>
    {
        static constexpr std::string_view group_name = "metadata";

        static void write(
            H5::Group&                              parent,
            const ecs::simulation_metadata_t<Rank>& meta,
            const write_policy_t&                   policy
        )
        {
            auto g = parent.createGroup(std::string(group_name));

            // numerics
            write_attribute(g, "gamma", meta.gamma);
            write_attribute(g, "plm_theta", meta.plm_theta);
            write_attribute(g, "viscosity", meta.viscosity);
            write_attribute(g, "cfl", meta.cfl);
            write_attribute(g, "time", meta.time);
            write_attribute(g, "tend", meta.tend);
            write_attribute(g, "dt", meta.global_dt);
            write_attribute(g, "dlogt", meta.dlogt);
            write_attribute(g, "checkpoint_interval", meta.checkpoint_interval);
            write_attribute(g, "checkpoint_time", meta.checkpoint_time);
            write_attribute(g, "prev_checkpoint_time", meta.prev_checkpoint_time);
            write_attribute(g, "ambient_sound_speed", meta.ambient_sound_speed);

            // int tracking
            write_attribute(g, "iteration", meta.iteration);
            write_attribute(g, "halo_radius", meta.halo_radius);
            write_attribute(g, "checkpoint_index", meta.checkpoint_index);
            write_attribute(g, "checkpoint_zones", meta.checkpoint_zones);
            write_attribute(g, "dimensions", meta.dimensions);

            // enums (as strings for robustness)
            write_attribute(g, "regime", serialize(meta.regime));
            write_attribute(g, "solver", serialize(meta.solver));
            write_attribute(g, "coord_system", serialize(meta.coord_system));
            write_attribute(g, "reconstruction", serialize(meta.reconstruction));
            write_attribute(g, "timestepping", serialize(meta.timestepping));
            write_attribute(g, "shock_smoother", serialize(meta.shock_smoother));
            write_attribute(g, "subcycling_mode", serialize(meta.subcycling_mode));

            // cell spacing enums
            write_attribute(g, "x1_spacing", serialize(meta.x1_spacing));
            write_attribute(g, "x2_spacing", serialize(meta.x2_spacing));
            write_attribute(g, "x3_spacing", serialize(meta.x3_spacing));

            // flags
            write_attribute(g, "is_mhd", meta.is_mhd);
            write_attribute(g, "is_relativistic", meta.is_relativistic);

            // strings
            write_attribute(g, "data_dir", meta.data_dir);

            // boundary conditions (as string array)
            write_boundary_conditions(g, meta.boundary_conditions, policy);

            // resolution
            std::vector<std::int64_t> res_data(meta.resolution.begin(), meta.resolution.end());
            std::vector<hsize_t>      res_dims{3};
            write_dataset(g, "resolution", res_data, res_dims, policy);

            // level timesteps if present
            if (!meta.level_dts.empty()) {
                std::vector<hsize_t> lvl_dims{meta.level_dts.size()};
                write_dataset(g, "level_dts", meta.level_dts, lvl_dims, policy);
            }

            // level substeps if present
            if (!meta.level_substeps.empty()) {
                std::vector<std::uint64_t> substeps(
                    meta.level_substeps.begin(),
                    meta.level_substeps.end()
                );
                std::vector<hsize_t> dims{substeps.size()};
                write_dataset(g, "level_substeps", substeps, dims, policy);
            }
        }

        static ecs::simulation_metadata_t<Rank> read(const H5::Group& parent)
        {
            auto g = parent.openGroup(std::string(group_name));

            ecs::simulation_metadata_t<Rank> meta;

            // numerics
            meta.gamma                = read_attribute<real>(g, "gamma");
            meta.plm_theta            = read_attribute<real>(g, "plm_theta");
            meta.viscosity            = read_attribute<real>(g, "viscosity");
            meta.cfl                  = read_attribute<real>(g, "cfl");
            meta.time                 = read_attribute<real>(g, "time");
            meta.tend                 = read_attribute<real>(g, "tend");
            meta.global_dt            = read_attribute<real>(g, "dt");
            meta.dlogt                = read_attribute<real>(g, "dlogt");
            meta.checkpoint_interval  = read_attribute<real>(g, "checkpoint_interval");
            meta.checkpoint_time      = read_attribute<real>(g, "checkpoint_time");
            meta.prev_checkpoint_time = read_attribute<real>(g, "prev_checkpoint_time");
            meta.ambient_sound_speed  = read_attribute<real>(g, "ambient_sound_speed");

            // int tracking
            meta.iteration        = read_attribute<std::uint64_t>(g, "iteration");
            meta.halo_radius      = read_attribute<std::uint64_t>(g, "halo_radius");
            meta.checkpoint_index = read_attribute<std::uint64_t>(g, "checkpoint_index");

            meta.checkpoint_zones = read_attribute<std::uint64_t>(g, "checkpoint_zones");
            meta.dimensions       = read_attribute<std::uint64_t>(g, "dimensions");

            // enums
            meta.regime          = read_enum_attribute<regime_t>(g, "regime");
            meta.solver          = read_enum_attribute<solver_t>(g, "solver");
            meta.coord_system    = read_enum_attribute<geometry_t>(g, "coord_system");
            meta.reconstruction  = read_enum_attribute<reconstruction_t>(g, "reconstruction");
            meta.timestepping    = read_enum_attribute<timestepping_t>(g, "timestepping");
            meta.shock_smoother  = read_enum_attribute<shockwave_limiter_t>(g, "shock_smoother");
            meta.subcycling_mode = read_enum_attribute<subcycling_mode_t>(g, "subcycling_mode");

            // cell spacing
            meta.x1_spacing = read_enum_attribute<cellspacing_t>(g, "x1_spacing");
            meta.x2_spacing = read_enum_attribute<cellspacing_t>(g, "x2_spacing");
            meta.x3_spacing = read_enum_attribute<cellspacing_t>(g, "x3_spacing");

            // flags
            meta.is_mhd          = read_attribute<bool>(g, "is_mhd");
            meta.is_relativistic = read_attribute<bool>(g, "is_relativistic");

            // strings
            meta.data_dir = read_attribute<std::string>(g, "data_dir");

            // boundary conditions
            meta.boundary_conditions = read_boundary_conditions(g);

            // resolution
            auto res_data = read_dataset<std::int64_t>(g, "resolution");
            for (std::size_t ii = 0; ii < 3 && ii < res_data.size(); ++ii) {
                meta.resolution[ii] = res_data[ii];
            }

            // level timesteps
            if (dataset_exists(g, "level_dts")) {
                meta.level_dts = read_dataset<real>(g, "level_dts");
            }

            // level substeps
            if (dataset_exists(g, "level_substeps")) {
                auto substeps       = read_dataset<std::uint64_t>(g, "level_substeps");
                meta.level_substeps = std::vector<std::uint64_t>(substeps.begin(), substeps.end());
            }

            return meta;
        }

      private:
        static void write_boundary_conditions(
            H5::Group&                                       parent,
            const vector_t<grid::boundary_type_t, 2 * Rank>& bcs,
            const write_policy_t& /*policy*/
        )
        {
            auto g = parent.createGroup("boundary_conditions");
            for (std::size_t ii = 0; ii < 2 * Rank; ++ii) {
                write_attribute(g, "bc_" + std::to_string(ii), serialize(bcs[ii]));
            }
        }

        static vector_t<grid::boundary_type_t, 2 * Rank>
        read_boundary_conditions(const H5::Group& parent)
        {
            vector_t<grid::boundary_type_t, 2 * Rank> bcs;

            // try new format first (individual string attributes)
            if (group_exists(parent, "boundary_conditions")) {
                auto g = parent.openGroup("boundary_conditions");
                for (std::size_t ii = 0; ii < 2 * Rank; ++ii) {
                    std::string name = "bc_" + std::to_string(ii);
                    if (attribute_exists(g, name)) {
                        auto str = read_attribute<std::string>(g, name);
                        bcs[ii]  = deserialize<grid::boundary_type_t>(str);
                    }
                }
            }
            else {
                // backward compatibility: old format with int dataset
                auto bc_data = read_dataset<int>(parent, "boundary_conditions");
                for (std::size_t ii = 0; ii < 2 * Rank && ii < bc_data.size(); ++ii) {
                    bcs[ii] = static_cast<grid::boundary_type_t>(bc_data[ii]);
                }
            }

            return bcs;
        }

        // helper to check if an attribute exists
        static bool attribute_exists(const H5::Group& g, const std::string& name)
        {
            return g.attrExists(name);
        }

        // helper to check if a dataset exists
        static bool dataset_exists(const H5::Group& g, const std::string& name)
        {
            return g.nameExists(name);
        }
    };

} // namespace simbi::io

#endif // IO_SERIAL_METADATA_HPP
