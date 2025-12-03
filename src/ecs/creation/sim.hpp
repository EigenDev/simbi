#ifndef ECS_BUILDERS_SIMULATION_BUILDER_HPP
#define ECS_BUILDERS_SIMULATION_BUILDER_HPP

// =============================================================================
// sim.hpp
//
// simulation builder - assembles a complete simulation_t from blueprints.
//
// build phases:
//   1. global entity + metadata
//   2. topology hierarchy (AMR levels)
//   3. decomposition (partition domains across devices)
//   4. geometry (coordinate maps)
//   5. sources (expression compilation)
//
// usage:
//   auto sim = simulation_builder_t<R, Rank, G, EoS>{}
//       .configure_mesh(mesh_bp)
//       .configure_physics(phys_bp)
//       .configure_execution(exec_bp)
//       .configure_amr(amr_bp)
//       .configure_decomposition(decomp_bp)  // optional, defaults to single
//       .build();
// =============================================================================

#include "compat.hpp"
#include "containers/vector.hpp"
#include "decomposition.hpp"
#include "ecs/blueprints.hpp"
#include "ecs/components.hpp"
#include "ecs/entity.hpp"
#include "ecs/simulation.hpp"
#include "geometry/api.hpp"
#include "grid/block_info.hpp"

#include "grid/boundary.hpp"
#include "grid/creation/topology.hpp"
#include "grid/domain.hpp"
#include "grid/mesh_config.hpp"
#include "grid/patch_id.hpp"
#include "grid/skeleton.hpp"
#include "hesi/adapter.hpp"
#include "hesi/core/types.hpp"
#include "io/h5_serializable.hpp"
#include "io/serialization/all.hpp"
#include "physics/ib/collection.hpp"
#include "physics/ib/diagnostics.hpp"
#include "physics/ib/factory.hpp"
#include "utility/bimap.hpp"
#include "utility/enums.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>

namespace simbi::ecs::builders {

    template <regime_t R, std::uint64_t Rank, geometry_t G, typename EoS>
    struct simulation_builder_t {
        using sim_t       = simulation_t<R, Rank, G, EoS>;
        using conserved_t = typename sim_t::conserved_t;
        using primitive_t = typename sim_t::primitive_t;

        // =========================================================================
        // blueprint storage
        // =========================================================================

        mesh_blueprint_t<Rank> mesh_bp_;
        physics_blueprint_t phys_bp_;
        execution_blueprint_t exec_bp_;
        amr_blueprint_t amr_bp_;
        numerics_blueprint_t num_bp_;

        // optional decomposition config (defaults to single-device)
        std::optional<decomposition_blueprint_t<Rank>> decomp_bp_;

        // optional expressions config (source terms, boundary injectors)
        expressions_blueprint_t expr_bp_;

        // optional bodies config (immersed boundary objects)
        bodies_blueprint_t bodies_bp_;

        // optional locality override:
        // default base locality is chosen at runtime based on detected devices.
        // prefer the compiled backend (cuda/hip) when devices are available,
        // otherwise fall back to host. callers may override via
        // configure_locality().
        het::locality_t base_locality_ = []() {
            if constexpr (platform::is_cuda) {
                int n = het::info::device_count();
                if (n > 0) {
                    return het::locality_t{het::backend_type_t::cuda, 0};
                }
                return het::locality_t::host();
            }
            else if constexpr (platform::is_hip) {
                int n = het::info::device_count();
                if (n > 0) {
                    return het::locality_t{het::backend_type_t::hip, 0};
                }
                return het::locality_t::host();
            }
            else {
                return het::locality_t::host();
            }
        }();

        // =========================================================================
        // configuration methods (fluent interface)
        // =========================================================================

        auto& configure_mesh(const mesh_blueprint_t<Rank>& bp)
        {
            mesh_bp_ = bp;
            return *this;
        }

        auto& configure_physics(const physics_blueprint_t& bp)
        {
            phys_bp_ = bp;
            return *this;
        }

        auto& configure_execution(const execution_blueprint_t& bp)
        {
            exec_bp_ = bp;
            return *this;
        }

        auto& configure_amr(const amr_blueprint_t& bp)
        {
            amr_bp_ = bp;
            return *this;
        }

        auto& configure_numerics(const numerics_blueprint_t& bp)
        {
            num_bp_ = bp;
            return *this;
        }

        // -------------------------------------------------------------------------
        // configure_decomposition
        //
        // sets the multi-device decomposition config.
        // if not called, build() uses single-device mode.
        // -------------------------------------------------------------------------
        auto& configure_decomposition(const decomposition_blueprint_t<Rank>& bp)
        {
            decomp_bp_ = bp;
            return *this;
        }

        // -------------------------------------------------------------------------
        // configure_expressions
        //
        // sets user-defined source terms and boundary expressions.
        // -------------------------------------------------------------------------
        auto& configure_expressions(const expressions_blueprint_t& bp)
        {
            expr_bp_ = bp;
            return *this;
        }

        // -------------------------------------------------------------------------
        // configure_bodies
        //
        // sets immersed boundary object configuration.
        // -------------------------------------------------------------------------
        auto& configure_bodies(const bodies_blueprint_t& bp)
        {
            bodies_bp_ = bp;
            return *this;
        }

        // -------------------------------------------------------------------------
        // configure_locality
        //
        // sets the base locality for field allocation.
        // use this to allocate on gpu instead of host.
        // -------------------------------------------------------------------------
        auto& configure_locality(het::locality_t loc)
        {
            base_locality_ = loc;
            return *this;
        }

        // =========================================================================
        // build
        // =========================================================================

        sim_t build()
        {
            // check if we're restarting from a checkpoint
            if (!exec_bp_.restart_file.empty() &&
                std::filesystem::exists(exec_bp_.restart_file)) {
                return build_from_checkpoint();
            }

            return build_from_scratch();
        }

      private:
        // -------------------------------------------------------------------------
        // build_from_scratch
        //
        // creates a fresh simulation from blueprints.
        // -------------------------------------------------------------------------
        sim_t build_from_scratch()
        {
            sim_t sim;

            // phase 1: global entity + metadata
            sim.global = sim.registry.create();
            build_metadata(sim);

            // phase 2: topology hierarchy
            auto hierarchy =
                grid::creation::topology_builder_t<Rank>::build_hierarchy(
                    mesh_bp_,
                    amr_bp_
                );

            // phase 3: build levels with decomposition
            for (std::uint64_t lvl = 0; lvl < hierarchy.size(); ++lvl) {
                build_level(sim, hierarchy[lvl], lvl);
            }

            // phase 4: compile source expressions
            build_sources(sim);

            // phase 5: create immersed bodies
            build_bodies(sim);

            return sim;
        }

        // -------------------------------------------------------------------------
        // build_from_checkpoint
        //
        // loads simulation state from hdf5 checkpoint file.
        // uses blueprints for configuration that isn't stored in checkpoint.
        // -------------------------------------------------------------------------
        sim_t build_from_checkpoint()
        {
            H5::H5File file(exec_bp_.restart_file, H5F_ACC_RDONLY);

            sim_t sim;
            sim.global = sim.registry.create();

            // load metadata from checkpoint, but override some values from
            // blueprints
            auto meta =
                io::h5_serializable<simulation_metadata_t<Rank>>::read(file);

            // override execution params from current blueprints
            // (user may want different end time, output dir, etc.)
            meta.tend                = exec_bp_.end_time;
            meta.checkpoint_interval = exec_bp_.checkpoint_interval;
            meta.dlogt               = exec_bp_.dlogt;
            meta.data_dir            = exec_bp_.data_directory;

            sim.registry.add(sim.global, std::move(meta));

            // read hierarchy info
            auto hierarchy_group = file.openGroup("hierarchy");
            auto num_levels      = io::read_attribute<std::uint64_t>(
                hierarchy_group,
                "num_levels"
            );

            // build each level from checkpoint
            for (std::uint64_t lvl = 0; lvl < num_levels; ++lvl) {
                build_level_from_checkpoint(sim, file, lvl);
            }

            // load bodies if present
            if (io::group_exists(file, "bodies")) {
                using collection_t = body::body_collection_t<Rank>;
                auto bodies = io::h5_serializable<collection_t>::read(file);
                sim.registry.add(
                    sim.global,
                    immersed_bodies_t<Rank>{std::move(bodies)}
                );
            }

            // sources are not stored in checkpoint; recompile from blueprints
            build_sources(sim);

            return sim;
        }

        // -------------------------------------------------------------------------
        // build_level_from_checkpoint
        //
        // loads a single level's data from checkpoint.
        // -------------------------------------------------------------------------
        void build_level_from_checkpoint(
            sim_t& sim,
            const H5::H5File& file,
            std::uint64_t lvl
        )
        {
            auto level_group = file.openGroup("level_" + std::to_string(lvl));

            // create level entity
            entity_t level_entity = sim.registry.create();
            sim.levels.push_back(level_entity);

            // read mesh config
            auto mesh_cfg =
                io::h5_serializable<grid::mesh_config_t<Rank>>::read(
                    level_group
                );

            // get partition count from hierarchy
            auto hierarchy_group = file.openGroup("hierarchy");
            auto lg = hierarchy_group.openGroup("level_" + std::to_string(lvl));
            auto num_partitions =
                io::read_attribute<std::uint64_t>(lg, "num_partitions");

            // rebuild skeleton from mesh config
            // for multi-block AMR, skeleton would need to be serialized to
            // checkpoint and loaded here. for now, single-block is sufficient.
            grid::skeleton_t<Rank> skeleton;
            grid::block_info_t<Rank> block;
            block.id           = grid::patch_id_t{0, {}};
            block.geometry     = grid::extents(mesh_cfg.global_cells);
            skeleton[block.id] = block;

            level_decomposition_t<Rank> decomp;
            if (decomp_bp_) {
                decomp = creation::decomposition_builder_t<Rank>::
                    template build<conserved_t, primitive_t>(
                        skeleton,
                        *decomp_bp_,
                        mesh_bp_,
                        phys_bp_,
                        sim.registry,
                        base_locality_
                    );
            }
            else {
                decomp = creation::decomposition_builder_t<Rank>::
                    template build_single_device<conserved_t, primitive_t>(
                        skeleton,
                        mesh_bp_,
                        phys_bp_,
                        sim.registry,
                        base_locality_
                    );
            }

            // load hydro fields from checkpoint into partitions
            for (std::uint64_t pp = 0; pp < num_partitions; ++pp) {
                if (pp >= decomp.partitions.size()) {
                    break;
                }

                auto part_group =
                    level_group.openGroup("partition_" + std::to_string(pp));

                using fields_t =
                    partition_fields_t<conserved_t, primitive_t, Rank>;
                auto fields = io::h5_serializable<fields_t>::read(part_group);

                // update the partition's field entity with loaded data
                auto field_entity = decomp.partition_entities[pp];
                sim.registry.template get<fields_t>(field_entity) =
                    std::move(fields);
            }

            sim.registry.add(level_entity, std::move(decomp));

            // level info
            std::uint64_t ratio = 1;
            if (lvl > 0) {
                ratio =
                    io::read_attribute<std::uint64_t>(lg, "refinement_ratio");
            }

            sim.registry.add(
                level_entity,
                level_info_t{.level_id = lvl, .refinement_ratio = ratio}
            );

            // mesh config for geometry queries (use loaded config from
            // checkpoint)
            sim.registry.add(
                level_entity,
                level_mesh_t<Rank>{.config = mesh_cfg}
            );

            // refinement linkage for child levels
            if (lvl > 0) {
                entity_t parent_entity = sim.levels[lvl - 1];
                grid::domain_t<Rank> parent_coverage;
                // compute parent coverage from mesh bounds
                for (std::uint64_t dd = 0; dd < Rank; ++dd) {
                    parent_coverage.start[dd] = 0;
                    parent_coverage.fin[dd] = mesh_cfg.global_cells[dd] / ratio;
                }

                sim.registry.add(
                    level_entity,
                    refinement_child_t<Rank>{
                      .parent          = parent_entity,
                      .parent_coverage = parent_coverage
                    }
                );
            }
        }

      private:
        // -------------------------------------------------------------------------
        // build_metadata
        //
        // creates simulation_metadata_t from blueprints and attaches to global.
        // -------------------------------------------------------------------------
        void build_metadata(sim_t& sim)
        {
            simulation_metadata_t<Rank> meta;

            // physics
            meta.gamma               = phys_bp_.gamma;
            meta.cfl                 = phys_bp_.cfl;
            meta.plm_theta           = phys_bp_.plm_theta;
            meta.viscosity           = phys_bp_.viscosity;
            meta.is_mhd              = phys_bp_.is_mhd;
            meta.is_relativistic     = phys_bp_.is_relativistic;
            meta.ambient_sound_speed = phys_bp_.ambient_sound_speed;
            meta.regime              = phys_bp_.regime;

            // execution
            meta.time                 = exec_bp_.start_time;
            meta.tend                 = exec_bp_.end_time;
            meta.checkpoint_interval  = exec_bp_.checkpoint_interval;
            meta.dlogt                = exec_bp_.dlogt;
            meta.data_dir             = exec_bp_.data_directory;
            meta.checkpoint_index     = exec_bp_.start_index;
            meta.checkpoint_zones     = exec_bp_.checkpoint_zones;
            meta.global_dt            = 0.0;
            meta.checkpoint_time      = exec_bp_.start_time;
            meta.prev_checkpoint_time = 0.0;
            meta.iteration            = 0;

            // mesh
            meta.resolution = to_3d_resolution(mesh_bp_.active_resolution);
            meta.halo_radius =
                decomp_bp_ ? decomp_bp_->halo_width : mesh_bp_.halo_width;
            meta.coord_system = G;
            meta.dimensions   = Rank;

            // cell spacing per dimension
            auto to_spacing = [](const std::string& s) {
                return (s == "log") ? cellspacing_t::LOG
                                    : cellspacing_t::LINEAR;
            };
            meta.x1_spacing = to_spacing(mesh_bp_.spacing[Rank - 1]);
            meta.x2_spacing = (Rank > 1)
                                  ? to_spacing(mesh_bp_.spacing[Rank - 2])
                                  : cellspacing_t::LINEAR;
            meta.x3_spacing = (Rank > 2)
                                  ? to_spacing(mesh_bp_.spacing[Rank - 3])
                                  : cellspacing_t::LINEAR;

            // boundary conditions
            for (std::uint64_t ii = 0; ii < 2 * Rank; ++ii) {
                meta.boundary_conditions[ii] =
                    deserialize<grid::boundary_type_t>(
                        mesh_bp_.boundary_conditions[ii]
                    );
            }

            // numerics
            meta.reconstruction =
                deserialize<reconstruction_t>(phys_bp_.reconstruction);
            meta.solver = deserialize<solver_t>(phys_bp_.solver);
            meta.timestepping =
                deserialize<timestepping_t>(phys_bp_.timestepping);

            // shock limiters
            if (num_bp_.use_quirk_smoothing) {
                meta.shock_smoother = shockwave_limiter_t::QUIRK;
            }
            else if (num_bp_.use_fleischmann_limiter) {
                meta.shock_smoother = shockwave_limiter_t::FLEISCHMANN;
            }
            else {
                meta.shock_smoother = shockwave_limiter_t::NONE;
            }

            // subcycling
            if (amr_bp_.enabled) {
                meta.subcycling_mode = amr_bp_.subcycling_mode;
                meta.level_substeps  = amr_bp_.manual_substeps;
                meta.level_dts.resize(amr_bp_.max_levels, 0.0);
            }
            else {
                meta.level_dts.resize(1, 0.0);
            }

            sim.registry.add(sim.global, std::move(meta));
        }

        // -------------------------------------------------------------------------
        // build_sources
        //
        // compiles user-defined source expressions and attaches to global.
        // -------------------------------------------------------------------------
        void build_sources(sim_t& sim)
        {
            sources_t<Rank> sources;

            // hydro source (momentum/energy injection)
            sources.hydro_source =
                state::expression_t<Rank>::from_config(expr_bp_.hydro_source);

            // gravity source (acceleration field)
            sources.gravity_source =
                state::expression_t<Rank>::from_config(expr_bp_.gravity_source);

            // boundary condition sources
            for (std::uint64_t ii = 0; ii < 2 * Rank; ++ii) {
                if (ii < expr_bp_.boundary_sources.size()) {
                    sources.bc_sources[ii] =
                        state::expression_t<Rank>::from_config(
                            expr_bp_.boundary_sources[ii]
                        );
                }
                else {
                    // empty config produces disabled expression
                    sources.bc_sources[ii] =
                        state::expression_t<Rank>::from_config({});
                }
            }

            sim.registry.add(sim.global, std::move(sources));
        }

        // -------------------------------------------------------------------------
        // build_bodies
        //
        // creates immersed boundary objects from blueprint.
        // -------------------------------------------------------------------------
        void build_bodies(sim_t& sim)
        {
            auto collection =
                body::factory::create_body_collection<Rank>(bodies_bp_);

            if (collection.has_value()) {
                sim.registry.add(
                    sim.global,
                    immersed_bodies_t<Rank>{std::move(*collection)}
                );
                sim.registry.add(
                    sim.global,
                    body_info_t<Rank>{
                      .diagnostics =
                          body::create_diagnostics_accumulator<Rank>()
                    }
                );
            }
        }

        // -------------------------------------------------------------------------
        // build_level
        //
        // creates a level entity with decomposition, fields, and level_info.
        // -------------------------------------------------------------------------
        void build_level(
            sim_t& sim,
            const grid::skeleton_t<Rank>& skeleton,
            std::uint64_t lvl
        )
        {
            // create level entity
            entity_t level_entity = sim.registry.create();
            sim.levels.push_back(level_entity);

            // build decomposition (handles partitioning + field allocation)
            level_decomposition_t<Rank> decomp;

            if (decomp_bp_) {
                // multi-device path
                decomp = creation::decomposition_builder_t<Rank>::
                    template build<conserved_t, primitive_t>(
                        skeleton,
                        *decomp_bp_,
                        mesh_bp_,
                        phys_bp_,
                        sim.registry,
                        base_locality_
                    );
            }
            else {
                // single-device path (backward compatible)
                decomp = creation::decomposition_builder_t<Rank>::
                    template build_single_device<conserved_t, primitive_t>(
                        skeleton,
                        mesh_bp_,
                        phys_bp_,
                        sim.registry,
                        base_locality_
                    );
            }

            // register decomposition on level entity
            sim.registry.add(level_entity, std::move(decomp));

            // level info
            std::uint64_t ratio = 1;
            if (lvl > 0 && lvl - 1 < amr_bp_.refinement_ratios.size()) {
                ratio = amr_bp_.refinement_ratios[lvl - 1];
            }

            sim.registry.add(
                level_entity,
                level_info_t{.level_id = lvl, .refinement_ratio = ratio}
            );

            // mesh config for geometry queries
            sim.registry.add(
                level_entity,
                level_mesh_t<Rank>{.config = build_mesh_config(lvl)}
            );

            // refinement linkage
            if (lvl > 0) {
                entity_t parent_entity = sim.levels[lvl - 1];
                auto child_domain      = skeleton.begin()->second.geometry;

                // map child domain to parent space
                grid::domain_t<Rank> parent_coverage;
                for (std::uint64_t dd = 0; dd < Rank; ++dd) {
                    parent_coverage.start[dd] = child_domain.start[dd] / ratio;
                    parent_coverage.fin[dd]   = child_domain.fin[dd] / ratio;
                }

                sim.registry.add(
                    level_entity,
                    refinement_child_t<Rank>{
                      .parent          = parent_entity,
                      .parent_coverage = parent_coverage
                    }
                );
            }
        }

        // -------------------------------------------------------------------------
        // helper: convert Rank resolution to 3d array
        // -------------------------------------------------------------------------
        static iarray<3> to_3d_resolution(const iarray<Rank>& res)
        {
            iarray<3> result{1, 1, 1};
            for (std::uint64_t ii = 0; ii < Rank && ii < 3; ++ii) {
                // map to (nz, ny, nx) ordering
                result[3 - Rank + ii] = res[ii];
            }
            return result;
        }

        // -------------------------------------------------------------------------
        // helper: build mesh_config_t from blueprint
        // -------------------------------------------------------------------------
        grid::mesh_config_t<Rank> build_mesh_config(std::uint64_t lvl)
        {
            // some things are given to use in logical order (ni, nj, nz)
            // and we manually convert them into array order (nz, ny, nx)
            grid::mesh_config_t<Rank> cfg;

            // compute hypothetical full-domain resolution at this level
            // for global coordinate system
            cfg.global_cells = mesh_bp_.active_resolution;
            for (std::uint64_t ll = 0; ll < lvl; ++ll) {
                std::uint64_t ratio = (ll < amr_bp_.refinement_ratios.size())
                                          ? amr_bp_.refinement_ratios[ll]
                                          : 2;
                for (std::uint64_t dd = 0; dd < Rank; ++dd) {
                    cfg.global_cells[dd] *= ratio;
                }
            }

            cfg.block_size = cfg.global_cells;
            cfg.halo_width = mesh_bp_.halo_width;

            // boundaries
            // blueprint provides strings packed as [left, right, left,
            // right...] for dimensions in reverse order (highest dim first)
            const auto& bc_strs = mesh_bp_.boundary_conditions;
            for (std::uint64_t ii = 0; ii < Rank; ++ii) {
                std::size_t vec_offset = ii * 2;
                if (vec_offset + 1 < bc_strs.size()) {
                    auto left_type =
                        deserialize<grid::boundary_type_t>(bc_strs[vec_offset]);
                    auto right_type = deserialize<grid::boundary_type_t>(
                        bc_strs[vec_offset + 1]
                    );

                    cfg.boundaries.set_left(ii, left_type);
                    cfg.boundaries.set_right(ii, right_type);
                }
            }

            // geometry config
            cfg.geometry.metric = geometry::metric_type_t::cartesian;
            if (mesh_bp_.coord_system == "spherical") {
                cfg.geometry.metric = geometry::metric_type_t::spherical;
            }
            else if (mesh_bp_.coord_system == "cylindrical") {
                cfg.geometry.metric = geometry::metric_type_t::cylindrical;
            }

            // dimension configs (bounds and spacing type)
            // all levels use root domain bounds for global coordinate system
            for (std::uint64_t dd = 0; dd < Rank; ++dd) {
                geometry::dimension_config_t dcfg;
                dcfg.type = geometry::map_type_t::uniform;

                if (mesh_bp_.spacing[dd] == "log") {
                    dcfg.type = geometry::map_type_t::log;
                }

                // all levels use root bounds (global coordinates)
                dcfg.start = mesh_bp_.bounds[dd].first;
                dcfg.end   = mesh_bp_.bounds[dd].second;

                cfg.geometry.dims.push_back(dcfg);
            }

            cfg.geometry.block_size_cells = cfg.block_size;

            return cfg;
        }
    };

}   // namespace simbi::ecs::builders

#endif
