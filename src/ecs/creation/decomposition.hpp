#ifndef ECS_CREATION_DECOMPOSITION_HPP
#define ECS_CREATION_DECOMPOSITION_HPP

// =============================================================================
// decomposition.hpp
//
// builds level_decomposition_t from blueprints.
//
// responsibilities:
//   1. partition the domain across devices using topology_t
//   2. create partition_t for each device (block info, stream, domains)
//   3. build halo_graph from block connectivity
//   4. allocate partition fields via field_allocator
//
// usage:
//   auto decomp = decomposition_builder_t<Rank>::build(
//       skeleton,           // block layout from topology_builder
//       decomp_bp,          // device topology config
//       mesh_bp,            // for ghost width, boundaries
//       phys_bp,            // for mhd flag
//       amr_bp,             // for flux averaging
//       locality            // where to allocate (host or device)
//   );
// =============================================================================

#include "base/concepts.hpp"
#include "compat.hpp"
#include "containers/vector.hpp"
#include "ecs/blueprints.hpp"
#include "ecs/components.hpp"
#include "ecs/entity.hpp"
#include "grid/block_info.hpp"
#include "grid/boundary.hpp"
#include "grid/cartesian_builder.hpp"
#include "grid/connectivity.hpp"
#include "grid/domain.hpp"
#include "grid/field.hpp"
#include "grid/skeleton.hpp"
#include "hesi/core/types.hpp"
#include "hesi/exec/stream.hpp"

#include <cstddef>
#include <cstdint>
#include <iostream>

namespace simbi::ecs::creation {
    // -----------------------------------------------------------------------------
    // decomposition_builder_t
    //
    // builds level_decomposition_t from skeleton and blueprints.
    // handles:
    //   - domain partitioning via grid::decomposer_t
    //   - partition creation with streams and device assignment
    //   - halo graph construction from block connectivity
    // -----------------------------------------------------------------------------
    template <std::uint64_t Rank>
    struct decomposition_builder_t {

        // -------------------------------------------------------------------------
        // build
        //
        // main entry point. creates a complete level_decomposition_t.
        //
        // for single-device: pass topology_dims = {1, 1, 1}
        // for multi-device:  pass topology_dims matching device count
        // -------------------------------------------------------------------------
        template <typename Conserved, typename Primitive>
        static level_decomposition_t<Rank> build(
            const grid::skeleton_t<Rank>& base_skeleton,
            const decomposition_blueprint_t<Rank>& decomp_bp,
            const mesh_blueprint_t<Rank>& mesh_bp,
            const physics_blueprint_t& phys_bp,
            const amr_blueprint_t& amr_bp,
            registry_t& registry,
            het::locality_t base_locality = het::locality_t::host()
        )
        {
            level_decomposition_t<Rank> decomp;

            // store topology for later queries
            decomp.topology.dims = to_3d(decomp_bp.topology_dims);

            // get the global domain from the base skeleton
            // assumes single-block input skeleton (root level)
            auto global_domain = base_skeleton.begin()->second.geometry;

            // get boundary conditions from mesh blueprint
            auto boundaries = extract_boundaries(mesh_bp);

            // phase 1: build decomposed skeleton using cartesian_builder
            // this creates block_info_t for each partition with correct
            // connectivity
            build_partitioned_skeleton(decomp, global_domain, boundaries);

            // phase 2: create partition runtime state (streams, domains)
            build_partitions(decomp, decomp_bp, base_locality);

            // phase 3: build halo graph from connectivity
            build_halo_graph(decomp, decomp_bp.halo_width);

            // phase 4: allocate fields for each partition
            allocate_partition_fields<Conserved, Primitive>(
                decomp,
                phys_bp,
                amr_bp,
                registry,
                base_locality
            );

            return decomp;
        }

        // -------------------------------------------------------------------------
        // build_single_device
        //
        // convenience wrapper for single-device case.
        // equivalent to build() with topology = {1, 1, ...}
        // -------------------------------------------------------------------------
        template <typename Conserved, typename Primitive>
        static level_decomposition_t<Rank> build_single_device(
            const grid::skeleton_t<Rank>& skeleton,
            const mesh_blueprint_t<Rank>& mesh_bp,
            const physics_blueprint_t& phys_bp,
            const amr_blueprint_t& amr_bp,
            registry_t& registry,
            het::locality_t base_locality = het::locality_t::host()
        )
        {
            decomposition_blueprint_t<Rank> decomp_bp;
            decomp_bp.topology_dims.fill(1);
            decomp_bp.halo_width = mesh_bp.halo_width;

            return build<Conserved, Primitive>(
                skeleton,
                decomp_bp,
                mesh_bp,
                phys_bp,
                amr_bp,
                registry,
                base_locality
            );
        }

      private:
        // -------------------------------------------------------------------------
        // build_partitioned_skeleton
        //
        // uses cartesian_builder to create block_info for each partition.
        // handles neighbor connectivity and periodic boundaries.
        // -------------------------------------------------------------------------
        static void build_partitioned_skeleton(
            level_decomposition_t<Rank>& decomp,
            const grid::domain_t<Rank>& global_domain,
            const grid::boundary_set_t<Rank>& boundaries
        )
        {
            std::int64_t num_partitions = decomp.topology.size();

            // build skeleton for each rank in the topology
            for (std::int64_t rank = 0; rank < num_partitions; ++rank) {
                grid::cartesian_builder_t::build(
                    decomp.skeleton,
                    global_domain,
                    decomp.topology,
                    rank,
                    boundaries
                );
            }
        }

        // -------------------------------------------------------------------------
        // build_partitions
        //
        // creates partition_t for each block in the skeleton.
        // assigns devices, creates streams, computes owned/allocated domains.
        // -------------------------------------------------------------------------
        static void build_partitions(
            level_decomposition_t<Rank>& decomp,
            const decomposition_blueprint_t<Rank>& decomp_bp,
            het::locality_t base_locality
        )
        {
            std::uint64_t part_idx = 0;

            for (const auto& [patch_id, block] : decomp.skeleton) {
                partition_t<Rank> part;

                // device assignment
                // round-robin if fewer device_ids than partitions
                if (!decomp_bp.device_ids.empty()) {
                    part.device_id =
                        decomp_bp
                            .device_ids[part_idx % decomp_bp.device_ids.size()];
                }
                else {
                    part.device_id = base_locality.device_id;
                }

                // copy block info
                part.block = block;

                // owned domain is the block's geometry (no ghosts)
                part.owned_domain = block.geometry;

                // allocated domain includes ghost padding
                part.allocated_domain =
                    compute_allocated_domain(block, decomp_bp.halo_width);

                // face-centered domains (for mhd ct updates)
                // each face_domain[d] has one extra cell in dimension d
                for (std::uint64_t dd = 0; dd < Rank; ++dd) {
                    auto face_dom = part.owned_domain;
                    face_dom.fin[dd] += 1;
                    part.face_domains[dd] = face_dom;
                }

                // edge-centered domains (for mhd constrained transport)
                // each edge_domain[d] has one extra cell in both transverse
                // dims
                for (std::uint64_t dd = 0; dd < Rank; ++dd) {
                    auto edge_dom = part.owned_domain;
                    for (std::uint64_t tt = 0; tt < Rank; ++tt) {
                        if (tt != dd) {
                            edge_dom.fin[tt] += 1;
                        }
                    }
                    part.edge_domains[dd] = edge_dom;
                }

                // create stream for this partition's device
                het::locality_t loc{base_locality.backend, part.device_id};
                part.stream = het::exec::stream_t{loc};

                // mpi rank info
                part.rank_id.node   = decomp_bp.mpi_rank;
                part.rank_id.device = part.device_id;

                // std::cout << "Created partition " << part_idx << " on device
                // "
                //           << part.device_id << " with "
                //           << "owned domain " << part.owned_domain << " and "
                //           << "allocated domain " << part.allocated_domain
                //           << "edge domain " << part.edge_domains
                //           << "face domain " << part.face_domains << "\n";
                // std::cin.get();

                decomp.partitions.push_back(std::move(part));
                ++part_idx;
            }
        }

        // -------------------------------------------------------------------------
        // compute_allocated_domain
        //
        // expands owned domain by halo_width, but only on faces that have
        // neighbors (partition boundaries). physical boundaries don't need
        // ghosts (boundary conditions are applied directly).
        // -------------------------------------------------------------------------
        static grid::domain_t<Rank> compute_allocated_domain(
            const grid::block_info_t<Rank>& block,
            std::int64_t halo_width
        )
        {
            auto domain = block.geometry;

            for (std::uint64_t dd = 0; dd < Rank; ++dd) {
                // left face
                // const auto& left_conn = block.get_face(dd,
                // grid::side_t::left); if (left_conn.is_connected()) {
                //     domain.start[dd] -= halo_width;
                // }
                domain.start[dd] -= halo_width;

                // right face
                // const auto& right_conn =
                //     block.get_face(dd, grid::side_t::right);
                // if (right_conn.is_connected()) {
                //     domain.fin[dd] += halo_width;
                // }
                domain.fin[dd] += halo_width;
            }

            return domain;
        }

        // -------------------------------------------------------------------------
        // build_halo_graph
        //
        // iterates all partition faces and creates halo_link_t for each
        // inter-partition boundary. each link describes:
        //   - source: interior boundary region of sender
        //   - dest: ghost region of receiver
        // -------------------------------------------------------------------------
        static void build_halo_graph(
            level_decomposition_t<Rank>& decomp,
            std::int64_t halo_width
        )
        {
            for (const auto& part : decomp.partitions) {
                // const auto& block = part.block;

                for (std::uint64_t dd = 0; dd < Rank; ++dd) {
                    // check left face
                    create_halo_link_if_connected(
                        decomp,
                        part,
                        dd,
                        grid::side_t::left,
                        halo_width
                    );

                    // check right face
                    create_halo_link_if_connected(
                        decomp,
                        part,
                        dd,
                        grid::side_t::right,
                        halo_width
                    );
                }
            }
        }

        // -------------------------------------------------------------------------
        // create_halo_link_if_connected
        //
        // if the face has a neighbor, creates a halo_link_t describing the
        // transfer from neighbor's interior to this partition's ghost zone.
        // -------------------------------------------------------------------------
        static void create_halo_link_if_connected(
            level_decomposition_t<Rank>& decomp,
            const partition_t<Rank>& recv_part,
            std::uint64_t dim,
            grid::side_t side,
            std::int64_t halo_width
        )
        {
            const auto& conn = recv_part.block.get_face(dim, side);

            if (!conn.is_connected()) {
                return;   // physical boundary, no halo needed
            }

            // find the sending partition
            const auto& sender_id = conn.single_neighbor();
            auto sender_idx       = decomp.find_partition(sender_id);

            if (sender_idx < 0) {
                // sender is on a different mpi rank
                // still create the link for mpi exchange
                // sender_idx will be resolved during exchange
            }

            const auto& send_part =
                (sender_idx >= 0)
                    ? decomp.partitions[sender_idx]
                    : recv_part;   // placeholder, will use rank_id lookup

            // compute regions
            // receiver's ghost zone
            grid::domain_t<Rank> recv_region = compute_ghost_region(
                recv_part.owned_domain,
                dim,
                side,
                halo_width
            );

            // sender's interior boundary (the data to copy)
            // this is the opposite side of the sender's domain
            auto opposite_side               = (side == grid::side_t::left)
                                                   ? grid::side_t::right
                                                   : grid::side_t::left;
            grid::domain_t<Rank> send_region = compute_interior_boundary(
                send_part.owned_domain,
                dim,
                opposite_side,
                halo_width
            );

            // create link
            halo_link_t<Rank> link;
            link.src_patch  = sender_id;
            link.src_rank   = send_part.rank_id;
            link.src_region = send_region;

            link.dst_patch  = recv_part.block.id;
            link.dst_rank   = recv_part.rank_id;
            link.dst_region = recv_region;

            link.dimension = dim;
            link.direction = side;

            decomp.halo_graph.push_back(link);
        }

        // -------------------------------------------------------------------------
        // compute_ghost_region
        //
        // returns the ghost zone domain for a given face.
        // ghost zone is outside the owned domain, adjacent to the specified
        // face.
        // -------------------------------------------------------------------------
        static grid::domain_t<Rank> compute_ghost_region(
            const grid::domain_t<Rank>& owned,
            std::uint64_t dim,
            grid::side_t side,
            std::int64_t width
        )
        {
            auto region = owned;

            if (side == grid::side_t::left) {
                // ghost is before owned in this dimension
                region.fin[dim]   = owned.start[dim];
                region.start[dim] = owned.start[dim] - width;
            }
            else {
                // ghost is after owned in this dimension
                region.start[dim] = owned.fin[dim];
                region.fin[dim]   = owned.fin[dim] + width;
            }

            return region;
        }

        // -------------------------------------------------------------------------
        // compute_interior_boundary
        //
        // returns the interior boundary region (the halo_width cells adjacent
        // to a face, but inside the owned domain).
        // -------------------------------------------------------------------------
        static grid::domain_t<Rank> compute_interior_boundary(
            const grid::domain_t<Rank>& owned,
            std::uint64_t dim,
            grid::side_t side,
            std::int64_t width
        )
        {
            auto region = owned;

            if (side == grid::side_t::left) {
                // interior boundary at left face
                region.fin[dim] = owned.start[dim] + width;
            }
            else {
                // interior boundary at right face
                region.start[dim] = owned.fin[dim] - width;
            }

            return region;
        }

        // -------------------------------------------------------------------------
        // allocate_partition_fields
        //
        // allocates partition_fields_t for each partition and registers
        // them in the ecs registry.
        // -------------------------------------------------------------------------
        template <typename Conserved, typename Primitive>
        static void allocate_partition_fields(
            level_decomposition_t<Rank>& decomp,
            const physics_blueprint_t& phys_bp,
            const amr_blueprint_t& amr_bp,
            registry_t& registry,
            het::locality_t base_locality
        )
        {
            using fields_t = partition_fields_t<Conserved, Primitive, Rank>;

            for (auto& part : decomp.partitions) {
                // create entity for this partition's components
                entity_t part_entity = registry.create();
                decomp.partition_entities.push_back(part_entity);

                // determine locality for allocation
                het::locality_t loc{base_locality.backend, part.device_id};

                // allocate fields on the allocated domain (includes ghosts)
                fields_t fields = allocate_partition<Conserved, Primitive>(
                    part.allocated_domain,
                    part.owned_domain,
                    phys_bp,
                    amr_bp,
                    loc
                );

                // register in ecs
                registry.add(part_entity, std::move(fields));
            }
        }

        // -------------------------------------------------------------------------
        // allocate_partition
        //
        // allocates a single partition_fields_t on the given domain and device.
        // -------------------------------------------------------------------------
        template <typename Conserved, typename Primitive>
        static partition_fields_t<Conserved, Primitive, Rank>
        allocate_partition(
            const grid::domain_t<Rank>& allocated_domain,
            const grid::domain_t<Rank>& active_domain,
            const physics_blueprint_t& phys_bp,
            const amr_blueprint_t& amr_bp,
            het::locality_t loc
        )
        {
            partition_fields_t<Conserved, Primitive, Rank> fields;

            // primary state fields
            fields.cons = grid::field_t<Conserved, Rank>(allocated_domain, loc);
            fields.prim = grid::field_t<Primitive, Rank>(allocated_domain, loc);

            // flux fields (face-centered, one extra cell in each direction)
            for (std::uint64_t dd = 0; dd < Rank; ++dd) {
                auto flux_domain = active_domain;
                flux_domain.fin[dd] += 1;   // n+1 faces for n cells
                if constexpr (is_mhd_conserved_c<Conserved>) {
                    // for MHD, extend faces in transverse directions
                    for (std::uint64_t tt = 0; tt < Rank; ++tt) {
                        if (tt != dd) {
                            flux_domain.start[tt] -= 1;
                            flux_domain.fin[tt] += 1;
                        }
                    }
                }
                fields.flux[dd] =
                    grid::field_t<Conserved, Rank>(flux_domain, loc);
            }

            // flux averaging for amr subcycling
            if (amr_bp.enabled) {
                for (std::uint64_t dd = 0; dd < Rank; ++dd) {
                    auto flux_domain = active_domain;
                    flux_domain.fin[dd] += 1;
                    for (std::uint64_t tt = 0; tt < Rank; ++tt) {
                        if (tt != dd) {
                            flux_domain.start[tt] -= 1;
                            flux_domain.fin[tt] += 1;
                        }
                    }
                    fields.flux_avg[dd] =
                        grid::field_t<Conserved, Rank>(flux_domain, loc);
                }
            }

            // magnetic field (face-centered, mhd only)
            if (phys_bp.is_mhd) {
                for (std::uint64_t dd = 0; dd < Rank; ++dd) {
                    auto bfield_domain = active_domain;
                    bfield_domain.fin[dd] += 1;
                    fields.bfield[dd] =
                        grid::field_t<real, Rank>(bfield_domain, loc);
                }

                // electric field (edge-centered, for constrained transport)
                // efield[d] has +1 in both transverse dimensions
                for (std::uint64_t dd = 0; dd < Rank; ++dd) {
                    auto efield_domain = active_domain;
                    for (std::uint64_t tt = 0; tt < Rank; ++tt) {
                        if (tt != dd) {
                            efield_domain.fin[tt] += 1;
                        }
                    }
                    fields.efield[dd] =
                        grid::field_t<real, Rank>(efield_domain, loc);
                }

                // efield averaging for amr subcycling
                if (amr_bp.enabled) {
                    for (std::uint64_t dd = 0; dd < Rank; ++dd) {
                        auto efield_domain = active_domain;
                        for (std::uint64_t tt = 0; tt < Rank; ++tt) {
                            if (tt != dd) {
                                efield_domain.fin[tt] += 1;
                            }
                        }
                        fields.efield_avg[dd] =
                            grid::field_t<real, Rank>(efield_domain, loc);
                    }
                }
            }

            return fields;
        }

        // -------------------------------------------------------------------------
        // helper: extract boundary_set from mesh blueprint
        // -------------------------------------------------------------------------
        static grid::boundary_set_t<Rank>
        extract_boundaries(const mesh_blueprint_t<Rank>& mesh_bp)
        {
            grid::boundary_set_t<Rank> boundaries;

            const auto& bc_strs = mesh_bp.boundary_conditions;
            for (std::uint64_t ii = 0; ii < Rank; ++ii) {
                std::size_t vec_offset = ii * 2;
                if (vec_offset + 1 < bc_strs.size()) {
                    auto left =
                        deserialize<grid::boundary_type_t>(bc_strs[vec_offset]);
                    auto right = deserialize<grid::boundary_type_t>(
                        bc_strs[vec_offset + 1]
                    );
                    boundaries.set_left(ii, left);
                    boundaries.set_right(ii, right);
                }
            }

            return boundaries;
        }

        // -------------------------------------------------------------------------
        // helper: convert Rank-dimensional array to 3d for topology_t
        // -------------------------------------------------------------------------
        static vector_t<std::int64_t, 3> to_3d(const iarray<Rank>& arr)
        {
            vector_t<std::int64_t, 3> result{1, 1, 1};
            for (std::uint64_t ii = 0; ii < Rank && ii < 3; ++ii) {
                result[ii] = arr[ii];
            }
            return result;
        }
    };

}   // namespace simbi::ecs::creation

#endif   // ECS_CREATION_DECOMPOSITION_HPP
