#ifndef IO_CHECKPOINT_HPP
#define IO_CHECKPOINT_HPP

#include "build_config.hpp"
#include "ecs/components.hpp"
#include "geometry/block_geometry.hpp"
#include "grid/mesh_config.hpp"
#include "grid/skeleton.hpp"
#include "h5_serializable.hpp"
#include "physics/ib/collection.hpp"
#include "serialization/skeleton_serial.hpp"
#include "utility/helpers.hpp"
#include "write_policy.hpp"
#include "xpu/xpu.hpp"

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

            // force a device syncronization before writing
            xpu::synchronize();

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

            // write motion state snapshot if mesh is moving
            if (sim.registry.template has<ecs::mesh_motion_config_t>(sim.global)) {
                const auto& motion_cfg =
                    sim.registry.template get<ecs::mesh_motion_config_t>(sim.global);
                auto motion_snapshot = motion_cfg.snapshot(sim.metadata().time);

                auto motion_group = file.createGroup("motion_state");
                write_attribute(motion_group, "enabled", motion_snapshot.enabled);
                write_attribute(motion_group, "is_homologous", motion_snapshot.is_homologous);
                write_attribute(motion_group, "a", motion_snapshot.a);
                write_attribute(motion_group, "a_dot", motion_snapshot.a_dot);
            }
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

            // annotate mesh interpretation for moving meshes
            if (sim.registry.template has<ecs::mesh_motion_config_t>(sim.global)) {
                const auto& motion_cfg =
                    sim.registry.template get<ecs::mesh_motion_config_t>(sim.global);
                auto motion_snapshot = motion_cfg.snapshot(sim.metadata().time);

                // add motion metadata to level group
                write_attribute(level_group, "coordinate_system", std::string("comoving"));
                write_attribute(level_group, "scale_factor_a", motion_snapshot.a);
                write_attribute(level_group, "scale_factor_adot", motion_snapshot.a_dot);
                write_attribute(
                    level_group,
                    "mesh_interpretation",
                    std::string("mesh bounds are comoving; physical = a(t) * comoving")
                );
            }
            else {
                write_attribute(level_group, "coordinate_system", std::string("physical"));
                write_attribute(
                    level_group,
                    "mesh_interpretation",
                    std::string("mesh bounds are physical (static mesh)")
                );
            }

            // write skeleton (topology with boundary metadata)
            const auto& skeleton = sim.decomposition(lvl).skeleton;
            h5_serializable<grid::skeleton_t<Sim::rank>>::write(level_group, skeleton, policy);

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

            write_attribute(g, "device_id", part.executor.device_id());

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

                    // preserve diagnostic deltas for checkpoint continuity
                    std::optional<std::vector<body::body_delta_t<Sim::rank>>> saved_deltas;

                    if constexpr (requires { sim.diagnostics(); }) {
                        auto& diag   = sim.diagnostics();
                        auto  deltas = diag->consolidate();

                        // save deltas for serialization
                        saved_deltas = std::vector<body::body_delta_t<Sim::rank>>(
                            deltas.begin(),
                            deltas.begin() + bodies_copy.size()
                        );

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
                                        if (dt_checkpoint > 0) {
                                            accr.accretion_rate = delta.mass_delta / dt_checkpoint;
                                        }
                                        else {
                                            accr.accretion_rate = 0.0;
                                        }
                                    }
                                },
                                bodies_copy.bodies_[ii]
                            );
                        }
                    }

                    using collection_t = body::body_collection_t<Sim::rank>;
                    h5_serializable<collection_t>::write(file, bodies_copy, policy);

                    // save diagnostic deltas for restart continuity
                    if (saved_deltas) {
                        write_diagnostic_deltas(file, *saved_deltas);
                    }
                }
            }
        }

        void write_diagnostic_deltas(
            H5::H5File&                                       file,
            const std::vector<body::body_delta_t<Sim::rank>>& deltas
        ) const
        {
            auto g = file.createGroup("diagnostic_deltas");
            write_attribute(g, "count", static_cast<std::uint64_t>(deltas.size()));

            for (std::size_t ii = 0; ii < deltas.size(); ++ii) {
                const auto& delta = deltas[ii];
                auto        dg    = g.createGroup("delta_" + std::to_string(ii));

                write_attribute(dg, "idx", delta.idx);
                write_attribute(dg, "mass_delta", delta.mass_delta);

                std::vector<real> force(delta.force_delta.begin(), delta.force_delta.end());
                std::vector<real> torque(delta.torque_delta.begin(), delta.torque_delta.end());

                std::vector<hsize_t> dims_rank{Sim::rank};
                std::vector<hsize_t> dims_3{3};

                write_dataset(dg, "force_delta", force, dims_rank, policy);
                write_dataset(dg, "torque_delta", torque, dims_3, policy);
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

            // read motion state snapshot if present (user must re-provide callbacks)
            // this is informational - the actual motion_config_t with callbacks
            // will be set by the user when constructing the restarted simulation
            if (group_exists(file, "motion_state")) {
                auto motion_group = file.openGroup("motion_state");
                // optionally validate motion state matches expected initial conditions
                // but actual motion_config_t callbacks must come from user config
            }

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

                // restore diagnostic deltas for continuity
                if (group_exists(file, "diagnostic_deltas")) {
                    auto deltas = read_diagnostic_deltas_helper(file);
                    if constexpr (requires { sim.diagnostics(); }) {
                        auto& diag = sim.diagnostics();
                        diag->restore_deltas(deltas);
                    }
                }
            }

            return sim;
        }

      private:
        static std::vector<body::body_delta_t<Sim::rank>>
        read_diagnostic_deltas_helper(const H5::H5File& file)
        {
            auto g     = file.openGroup("diagnostic_deltas");
            auto count = read_attribute<std::uint64_t>(g, "count");

            std::vector<body::body_delta_t<Sim::rank>> deltas(count);

            for (std::size_t ii = 0; ii < count; ++ii) {
                auto dg = g.openGroup("delta_" + std::to_string(ii));

                deltas[ii].idx        = read_attribute<std::uint64_t>(dg, "idx");
                deltas[ii].mass_delta = read_attribute<real>(dg, "mass_delta");

                auto force  = read_dataset<real>(dg, "force_delta");
                auto torque = read_dataset<real>(dg, "torque_delta");

                for (std::size_t dd = 0; dd < Sim::rank; ++dd) {
                    deltas[ii].force_delta[dd] = force[dd];
                }
                for (std::size_t dd = 0; dd < 3; ++dd) {
                    deltas[ii].torque_delta[dd] = torque[dd];
                }
            }

            return deltas;
        }

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
    Sim load_checkpoint(const std::string& filename)
    {
        return checkpoint_reader_t<Sim>::read(filename);
    }

} // namespace simbi::io

#endif // IO_CHECKPOINT_HPP
