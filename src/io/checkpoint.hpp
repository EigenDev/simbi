#ifndef IO_CHECKPOINT_HPP
#define IO_CHECKPOINT_HPP

#include "compat.hpp"
#include "ecs/components.hpp"
#include "grid/mesh_config.hpp"
#include "h5_serializable.hpp"
#include "physics/ib/collection.hpp"
#include "utility/helpers.hpp"
#include "write_policy.hpp"

#include <H5Cpp.h>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace simbi::io {

    // =========================================================================
    // checkpoint filename utilities
    // =========================================================================
    inline std::string compute_checkpoint_filename(
        const std::string& data_dir,
        real               checkpoint_id,
        std::uint64_t      checkpoint_index,
        std::uint64_t      checkpoint_zones,
        real               time,
        real               dlogt,
        bool               was_interrupted,
        bool               in_failure_state
    )
    {
        using namespace helpers;
        static std::int64_t   tchunk_order_of_mag = 2;
        std::filesystem::path dir(data_dir);

        const auto data_directory      = data_dir;
        const auto step                = checkpoint_index;
        const auto timestepping_of_mag = std::floor(std::log10(time));

        if (timestepping_of_mag > tchunk_order_of_mag) {
            tchunk_order_of_mag += 1;
        }

        std::string tnow;
        if (was_interrupted) {
            tnow = "interrupted";
        }
        else if (in_failure_state) {
            tnow = "crashed";
        }
        else if (dlogt != 0) {
            const auto timestepping_of_mag = std::floor(std::log10(step));
            if (timestepping_of_mag > tchunk_order_of_mag) {
                tchunk_order_of_mag += 1;
            }
            tnow = format_real(step);
        }
        else {
            tnow = format_real(checkpoint_id);
        }

        return data_directory + string_format("%d.chkpt." + tnow + ".h5", checkpoint_zones);
    }

    // =========================================================================
    // checkpoint writer
    // =========================================================================
    template <typename Sim>
    struct checkpoint_writer_t
    {
        const Sim&     sim;
        write_policy_t policy;

        void write(const std::string& filename) const
        {
            // ensure directory exists
            std::filesystem::path path(filename);
            if (path.has_parent_path()) {
                std::filesystem::create_directories(path.parent_path());
            }

            H5::H5File file(filename, H5F_ACC_TRUNC);

            // write file format version
            write_attribute(file, "format_version", std::string("2.0"));
            write_attribute(file, "simbi_version", std::string("0.8.0"));

            // metadata
            write_metadata(file);

            // hierarchy info
            write_hierarchy_info(file);

            // per-level data
            for (std::uint64_t lvl = 0; lvl < sim.num_levels(); ++lvl) {
                write_level(file, lvl);
            }

            // optional: bodies
            write_bodies(file);
        }

      private:
        void write_metadata(H5::H5File& file) const
        {
            h5_serializable<ecs::simulation_metadata_t<Sim::rank>>::write(
                file,
                sim.metadata(),
                policy
            );
        }

        void write_hierarchy_info(H5::H5File& file) const
        {
            auto g = file.createGroup("hierarchy");

            write_attribute(g, "num_levels", sim.num_levels());

            // per-level info
            for (std::uint64_t lvl = 0; lvl < sim.num_levels(); ++lvl) {
                auto lg = g.createGroup("level_" + std::to_string(lvl));
                write_attribute(lg, "num_partitions", sim.num_partitions(lvl));

                if (lvl > 0) {
                    write_attribute(lg, "refinement_ratio", sim.level_info(lvl).refinement_ratio);
                }
            }
        }

        void write_level(H5::H5File& file, std::uint64_t lvl) const
        {
            auto level_group = file.createGroup("level_" + std::to_string(lvl));

            // write mesh config with global coordinate system
            // global_cells is the hypothetical full-domain resolution
            // used for coordinate mapping, not the actual patch size
            auto mesh_cfg = sim.mesh(lvl);

            h5_serializable<grid::mesh_config_t<Sim::rank>>::write(level_group, mesh_cfg, policy);

            // per-partition hydro data
            for (std::uint64_t pp = 0; pp < sim.num_partitions(lvl); ++pp) {
                auto part_group = level_group.createGroup("partition_" + std::to_string(pp));

                // partition topology
                write_partition_info(part_group, lvl, pp);

                // hydro fields
                using fields_t = typename Sim::fields_t;
                h5_serializable<fields_t>::write(part_group, sim.partition_hydro(lvl, pp), policy);
            }
        }

        void write_partition_info(H5::Group& g, std::uint64_t lvl, std::uint64_t pp) const
        {
            const auto& part = sim.partition(lvl, pp);

            write_attribute(g, "device_id", part.device_id);

            // owned domain bounds
            std::vector<std::int64_t> owned_start(
                part.owned_domain.start.begin(),
                part.owned_domain.start.end()
            );
            std::vector<std::int64_t> owned_fin(
                part.owned_domain.fin.begin(),
                part.owned_domain.fin.end()
            );
            std::vector<hsize_t> dims{Sim::rank};

            write_dataset(g, "owned_start", owned_start, dims, policy);
            write_dataset(g, "owned_fin", owned_fin, dims, policy);
        }

        void write_bodies(H5::H5File& file) const
        {
            if constexpr (requires { sim.has_bodies(); }) {
                if (sim.has_bodies()) {
                    // consolidate diagnostics and merge into bodies
                    auto bodies_copy = sim.bodies();

                    if constexpr (requires { sim.diagnostics(); }) {
                        auto& diag   = sim.diagnostics();
                        auto  deltas = diag->consolidate();

                        // compute dt since last checkpoint
                        const auto& meta          = sim.metadata();
                        const real  dt_checkpoint = meta.time - meta.prev_checkpoint_time;

                        // merge accumulated deltas into body state
                        for (std::size_t ii = 0; ii < bodies_copy.size(); ++ii) {
                            std::visit(
                                [&](auto& body) {
                                    const auto& delta = deltas[ii];
                                    body.force        = delta.force_delta;
                                    body.torque       = delta.torque_delta;

                                    // update accretion properties if body has
                                    // them
                                    using body_type = std::decay_t<decltype(body)>;
                                    if constexpr (body_type::template has_capability_v<
                                                      body::capabilities::accretion_tag>) {
                                        auto& accr = std::get<body::accretion_component_t>(
                                            body.capabilities
                                        );
                                        accr.total_accreted_mass += delta.mass_delta;
                                        accr.accretion_rate = delta.mass_delta / dt_checkpoint;
                                    }
                                },
                                bodies_copy.bodies_[ii]
                            );
                        }
                    }

                    using collection_t = body::body_collection_t<Sim::rank>;
                    h5_serializable<collection_t>::write(file, bodies_copy, policy);
                }
            }
        }
    };

    // =========================================================================
    // checkpoint reader
    // =========================================================================
    template <typename Sim>
    struct checkpoint_reader_t
    {
        static Sim read(const std::string& filename)
        {
            H5::H5File file(filename, H5F_ACC_RDONLY);

            // check format version
            auto version = read_attribute<std::string>(file, "format_version");

            // read metadata
            auto meta = h5_serializable<ecs::simulation_metadata_t<Sim::rank>>::read(file);

            // read hierarchy info
            auto hierarchy_group = file.openGroup("hierarchy");
            auto num_levels      = read_attribute<std::uint64_t>(hierarchy_group, "num_levels");

            // construct simulation (implementation-specific)
            // this is a simplified version - actual implementation
            // would need to reconstruct the full simulation state
            Sim sim;
            sim.set_metadata(meta);

            // read per-level data
            for (std::uint64_t lvl = 0; lvl < num_levels; ++lvl) {
                read_level(file, sim, lvl);
            }

            // read bodies if present
            if (group_exists(file, "bodies")) {
                using collection_t = body::body_collection_t<Sim::rank>;
                auto bodies        = h5_serializable<collection_t>::read(file);
                sim.set_bodies(std::move(bodies));
            }

            return sim;
        }

      private:
        static void read_level(const H5::H5File& file, Sim& sim, std::uint64_t lvl)
        {
            auto level_group = file.openGroup("level_" + std::to_string(lvl));

            // mesh config
            auto mesh_cfg = h5_serializable<grid::mesh_config_t<Sim::rank>>::read(level_group);

            // determine partition count
            auto hierarchy_group = file.openGroup("hierarchy");
            auto lg              = hierarchy_group.openGroup("level_" + std::to_string(lvl));
            auto num_partitions  = read_attribute<std::uint64_t>(lg, "num_partitions");

            // read per-partition data
            for (std::uint64_t pp = 0; pp < num_partitions; ++pp) {
                auto part_group = level_group.openGroup("partition_" + std::to_string(pp));

                using fields_t = typename Sim::fields_t;
                auto fields    = h5_serializable<fields_t>::read(part_group);

                sim.set_partition_hydro(lvl, pp, std::move(fields));
            }
        }
    };

    // =========================================================================
    // convenience functions
    // =========================================================================

    template <typename Sim>
    void
    write_checkpoint(const Sim& sim, const std::string& filename, const write_policy_t& policy = {})
    {
        checkpoint_writer_t<Sim>{sim, policy}.write(filename);
    }

    template <typename Sim>
    void write_checkpoint(const Sim& sim, const write_policy_t& policy = {})
    {
        const auto& meta = sim.metadata();
        auto        filename =
            compute_checkpoint_filename(meta.data_dir, meta.checkpoint_identifier(), meta.dlogt);
        write_checkpoint(sim, filename, policy);
    }

    template <typename Sim>
    Sim load_checkpoint(const std::string& filename)
    {
        return checkpoint_reader_t<Sim>::read(filename);
    }

} // namespace simbi::io

#endif // IO_CHECKPOINT_HPP
