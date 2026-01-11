#ifndef ECS_SIMULATION_HPP
#define ECS_SIMULATION_HPP

// =============================================================================
// nsimulation.hpp
//
// multi-device simulation state container.
//
// architecture:
//   simulation_t owns:
//     - global entity (metadata, sources, bodies)
//     - level entities (level_info, refinement_child)
//     - level_decomposition (partitions, halo_graph)
//     - communicator (inter-partition data movement)
//
// access patterns:
//   single-device (backward compat):
//     sim.hydro(level)        -> fields for level (partition 0)
//     sim.mesh(level)         -> mesh config for level (partition 0)
//
//   multi-device:
//     sim.decomposition(level)              -> level_decomposition_t
//     sim.partition(level, part_id)         -> partition_t
//     sim.partition_hydro(level, part_id)   -> partition_fields_t
//     sim.mesh(level)                       -> mesh_config_t
//     sim.exchange_halos(level)             -> trigger all halo transfers
//
// the multi-device api exposes partitions explicitly. kernels are launched
// per-partition, each on its own stream, enabling concurrent execution.
// =============================================================================

#include "build_config.hpp"
#include "components.hpp"
#include "entity.hpp"
#include "geometry/block_geometry.hpp"
#include "geometry_visitor.hpp"
#include "grid/field.hpp"
#include "hydro_state_types.hpp"
#include "utility/enums.hpp"
#include "xpu/xpu.hpp"

#include <cstdint>
#include <vector>

namespace simbi::ecs {

    template <regime_t R, std::uint64_t Rank, geometry_t G, typename EoS>
    struct simulation_t
    {
        // =========================================================================
        // type aliases
        // =========================================================================

        using conserved_t = typename vtraits<R, Rank, EoS>::conserved_type;
        using primitive_t = typename vtraits<R, Rank, EoS>::primitive_type;
        using eos_t       = EoS;

        using fields_t    = partition_fields_t<conserved_t, primitive_t, Rank>;
        using workspace_t = partition_workspace_t<conserved_t, Rank>;
        using decomp_t    = level_decomposition_t<Rank>;

        // =========================================================================
        // compile-time constants
        // =========================================================================

        static constexpr std::uint64_t     rank         = Rank;
        static constexpr regime_t          regime       = R;
        static constexpr simbi::geometry_t coord_system = G;
        static constexpr bool              is_mhd = (R == regime_t::MHD || R == regime_t::RMHD);
        static constexpr auto              nvars  = is_mhd ? 9 : Rank + 3;

        // =========================================================================
        // core state
        // =========================================================================

        // ecs registry holding all components
        registry_t registry;

        // entity handles for each amr level
        std::vector<entity_t> levels;

        // global entity (metadata, sources, bodies)
        entity_t global;

        // communication backbone for halo exchange
        // todo: implement xpu communicator wrapper
        // het::comm::communicator_t communicator;

        // error/interrupt flags
        bool in_failure_state{false};
        bool was_interrupted{false};

        // =========================================================================
        // level queries
        // =========================================================================

        std::uint64_t num_levels() const
        {
            return levels.size();
        }

        bool has_refinement() const
        {
            return num_levels() > 1;
        }

        // =========================================================================
        // global component accessors
        // =========================================================================

        auto& metadata()
        {
            return registry.get<simulation_metadata_t<Rank>>(global);
        }

        const auto& metadata() const
        {
            return registry.get<simulation_metadata_t<Rank>>(global);
        }

        auto& sources()
        {
            return registry.get<sources_t<Rank>>(global);
        }

        const auto& sources() const
        {
            return registry.get<sources_t<Rank>>(global);
        }

        // =========================================================================
        // level component accessors
        // =========================================================================

        auto& level_info(std::uint64_t lvl)
        {
            return registry.get<level_info_t>(levels[lvl]);
        }

        const auto& level_info(std::uint64_t lvl) const
        {
            return registry.get<level_info_t>(levels[lvl]);
        }

        entity_t level_entity(std::uint64_t lvl) const
        {
            return levels[lvl];
        }

        auto& refinement(std::uint64_t lvl)
        {
            return registry.get<refinement_child_t<Rank>>(levels[lvl]);
        }

        const auto& refinement(std::uint64_t lvl) const
        {
            return registry.get<refinement_child_t<Rank>>(levels[lvl]);
        }

        auto& level_mesh(std::uint64_t lvl)
        {
            return registry.get<level_mesh_t<Rank>>(levels[lvl]);
        }

        const auto& level_mesh(std::uint64_t lvl) const
        {
            return registry.get<level_mesh_t<Rank>>(levels[lvl]);
        }

        // =========================================================================
        // decomposition accessors (multi-device api)
        // =========================================================================

        // -------------------------------------------------------------------------
        // decomposition
        //
        // returns the level_decomposition_t for a level, which contains:
        //   - skeleton (block metadata)
        //   - partitions (device assignments)
        //   - halo_graph (transfer descriptors)
        //   - partition_entities (ecs handles for fields)
        // -------------------------------------------------------------------------
        auto& decomposition(std::uint64_t lvl)
        {
            return registry.get<decomp_t>(levels[lvl]);
        }

        const auto& decomposition(std::uint64_t lvl) const
        {
            return registry.get<decomp_t>(levels[lvl]);
        }

        // -------------------------------------------------------------------------
        // num_partitions
        //
        // returns how many partitions exist for a level.
        // for single-device, this is 1.
        // for multi-device, this equals the number of devices used.
        // -------------------------------------------------------------------------
        std::uint64_t num_partitions(std::uint64_t lvl) const
        {
            return decomposition(lvl).num_partitions();
        }

        // -------------------------------------------------------------------------
        // partition
        //
        // returns the partition_t for a specific partition of a level.
        // contains device_id, stream, owned/allocated domains, block info.
        // -------------------------------------------------------------------------
        auto& partition(std::uint64_t lvl, std::uint64_t part_id)
        {
            return decomposition(lvl).partitions[part_id];
        }

        const auto& partition(std::uint64_t lvl, std::uint64_t part_id) const
        {
            return decomposition(lvl).partitions[part_id];
        }

        // -------------------------------------------------------------------------
        // partition_hydro
        //
        // returns the partition_fields_t for a specific partition.
        // this is where the actual cons/prim/flux data lives.
        // -------------------------------------------------------------------------
        auto& partition_hydro(std::uint64_t lvl, std::uint64_t part_id)
        {
            auto& decomp = decomposition(lvl);
            return registry.get<fields_t>(decomp.partition_entities[part_id]);
        }

        const auto& partition_hydro(std::uint64_t lvl, std::uint64_t part_id) const
        {
            const auto& decomp = decomposition(lvl);
            return registry.get<fields_t>(decomp.partition_entities[part_id]);
        }

        // -------------------------------------------------------------------------
        // partition_executor
        //
        // returns reference to the partition's executor.
        // use this to launch kernels on the partition's device.
        // -------------------------------------------------------------------------
        auto& partition_executor(std::uint64_t lvl, std::uint64_t part_id)
        {
            return partition(lvl, part_id).executor;
        }

        const auto& partition_executor(std::uint64_t lvl, std::uint64_t part_id) const
        {
            return partition(lvl, part_id).executor;
        }

        // -------------------------------------------------------------------------
        // partition flux register
        // -------------------------------------------------------------------------
        const auto& flux_register(std::uint64_t lvl) const
        {
            return registry.get<flux_register_component_t<conserved_t, Rank>>(levels[lvl]);
        }

        auto& flux_register(std::uint64_t lvl)
        {
            return registry.get<flux_register_component_t<conserved_t, Rank>>(levels[lvl]);
        }

        bool has_flux_register(std::uint64_t lvl) const
        {
            return registry.has<flux_register_component_t<conserved_t, Rank>>(levels[lvl]);
        }

        // -------------------------------------------------------------------------
        // with_geometry
        //
        // invokes a callback with the block geometry for a level.
        // uses visitor pattern to handle log/uniform coordinate maps.
        //
        // usage:
        //   sim.with_geometry(lvl, motion, [&](const auto& geo) {
        //       // geo is block_geometry_t<Metric>
        //       auto h = geo.scale_factors(coord);
        //   });
        // -------------------------------------------------------------------------
        template <typename Func>
        decltype(auto)
        with_geometry(std::uint64_t lvl, const geometry::motion_state_t& motion, Func&& func)
        {
            return ecs::with_block_geometry<coord_system>(
                level_mesh(lvl).config,
                motion,
                std::forward<Func>(func)
            );
        }

        template <typename Func>
        decltype(auto) with_geometry(std::uint64_t lvl, Func&& func)
        {
            return with_geometry(
                lvl,
                mesh_motion_config_t::static_mesh(),
                std::forward<Func>(func)
            );
        }

        // -------------------------------------------------------------------------
        // mesh
        //
        // returns mesh config for a level.
        // -------------------------------------------------------------------------
        auto& mesh(std::uint64_t lvl)
        {
            return level_mesh(lvl).config;
        }

        const auto& mesh(std::uint64_t lvl) const
        {
            return level_mesh(lvl).config;
        }

        // =========================================================================
        // halo exchange
        // =========================================================================

        // -------------------------------------------------------------------------
        // exchange_halos
        //
        // performs all halo exchanges for a level.
        // iterates the halo_graph and issues transfers via communicator.
        //
        // call this after all partitions have computed their interior,
        // before the next stage that requires neighbor data.
        // -------------------------------------------------------------------------
        void exchange_halos(std::uint64_t lvl)
        {
            auto& decomp = decomposition(lvl);

            for (const auto& link : decomp.halo_graph) {
                // find local partition indices
                auto src_idx = decomp.find_partition(link.src_patch);
                auto dst_idx = decomp.find_partition(link.dst_patch);

                // skip if neither endpoint is local
                if (src_idx < 0 && dst_idx < 0) {
                    continue;
                }

                // get field views
                // for non-local endpoints, communicator handles staging
                if (src_idx >= 0 && dst_idx >= 0) {
                    // both local: direct device-to-device copy
                    auto& src_fields = partition_hydro(lvl, src_idx);
                    auto& dst_fields = partition_hydro(lvl, dst_idx);

                    // copy cons field halo region
                    xpu::comm::transfer_region_sync(
                        dst_fields.cons.view(),
                        link.dst_region,
                        src_fields.cons.view(),
                        link.src_region
                    );
                }
                else {
                    // one endpoint is remote: requires MPI
                    // distributed memory support not yet implemented
                }
            }
        }

        // -------------------------------------------------------------------------
        // synchronize_partitions
        //
        // waits for all partition streams to complete.
        // call after launching kernels, before operations that need results.
        // -------------------------------------------------------------------------
        void synchronize_partitions(std::uint64_t lvl)
        {
            auto& decomp = decomposition(lvl);
            for (auto& part : decomp.partitions) {
                part.stream.synchronize();
            }
        }

        // =========================================================================
        // workspace management
        // =========================================================================

        // -------------------------------------------------------------------------
        // has_workspace
        //
        // checks if a partition has rk workspace allocated.
        // -------------------------------------------------------------------------
        bool has_workspace(std::uint64_t lvl, std::uint64_t part_id) const
        {
            const auto& decomp = decomposition(lvl);
            return registry.has<workspace_t>(decomp.partition_entities[part_id]);
        }

        // -------------------------------------------------------------------------
        // create_workspace
        //
        // allocates rk workspace for a partition.
        // call during initialization or lazily on first rk step.
        // -------------------------------------------------------------------------
        void create_workspace(std::uint64_t lvl, std::uint64_t part_id)
        {
            if (has_workspace(lvl, part_id)) {
                return;
            }

            auto& decomp = decomposition(lvl);
            auto& part   = partition(lvl, part_id);

            // allocate workspace on same device as fields
            workspace_t ws;
            ws.u_n    = grid::field_t<conserved_t, Rank>(part.allocated_domain);
            ws.u_star = grid::field_t<conserved_t, Rank>(part.allocated_domain);

            if constexpr (is_mhd) {
                for (std::uint64_t dd = 0; dd < Rank; ++dd) {
                    ws.e_n[dd] = grid::field_t<real, Rank>(part.edge_domains[dd]);
                }
            }

            registry.add<workspace_t>(decomp.partition_entities[part_id], ws);
        }

        // -------------------------------------------------------------------------
        // workspace
        //
        // returns the rk workspace for a partition.
        // -------------------------------------------------------------------------
        auto& workspace(std::uint64_t lvl, std::uint64_t part_id)
        {
            auto& decomp = decomposition(lvl);
            return registry.get<workspace_t>(decomp.partition_entities[part_id]);
        }

        const auto& workspace(std::uint64_t lvl, std::uint64_t part_id) const
        {
            const auto& decomp = decomposition(lvl);
            return registry.get<workspace_t>(decomp.partition_entities[part_id]);
        }

        // =========================================================================
        // immersed bodies (optional)
        // =========================================================================

        bool has_bodies() const
        {
            return registry.has<immersed_bodies_t<Rank>>(global);
        }

        auto& bodies()
        {
            return registry.get<immersed_bodies_t<Rank>>(global).bodies;
        }

        const auto& bodies() const
        {
            return registry.get<immersed_bodies_t<Rank>>(global).bodies;
        }

        auto& diagnostics()
        {
            return registry.get<body_info_t<Rank>>(global).diagnostics;
        }

        const auto& diagnostics() const
        {
            return registry.get<body_info_t<Rank>>(global).diagnostics;
        }
    };

} // namespace simbi::ecs

#endif
