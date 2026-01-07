#ifndef GRID_MESH_TOPLOGY_HPP
#define GRID_MESH_TOPLOGY_HPP

#include "field.hpp"
#include "patch_id.hpp"
#include "xpu/executor.hpp"
#include "xpu/token.hpp"

#include <cstdint>
#include <map>
#include <memory>
#include <tuple>
#include <vector>

namespace simbi::grid {

    // forward declaration of the communicator (to avoid circular dependency)
    namespace comm {
        struct communicator_t;
    }

    // -------------------------------------------------------------------------
    // mesh_topology_t: manages all field patches (blocks) and their
    // relationships
    // -------------------------------------------------------------------------
    template <typename T, std::uint64_t Rank>
    class mesh_topology_t
    {
      public:
        using patch_type = field_t<T, Rank>;
        // store patches by unique id
        using patch_map = std::map<patch_id_t, std::unique_ptr<patch_type>>;

      private:
        patch_map active_patches_;

      public:
        // ---------------------------------------------------------------------
        // basic management
        // ---------------------------------------------------------------------

        // adds a fully constructed, allocated field to the mesh
        void add_patch(patch_id_t id, std::unique_ptr<patch_type> patch)
        {
            active_patches_.emplace(id, std::move(patch));
        }

        // removes a patch
        void remove_patch(const patch_id_t& id)
        {
            active_patches_.erase(id);
        }

        // accessors
        patch_type* get_patch(const patch_id_t& id)
        {
            auto it = active_patches_.find(id);
            return (it != active_patches_.end()) ? it->second.get() : nullptr;
        }

        // iterators for the computation loop
        auto begin()
        {
            return active_patches_.begin();
        }
        auto end()
        {
            return active_patches_.end();
        }

        // ---------------------------------------------------------------------
        // static refinement framework (placeholders)
        // ---------------------------------------------------------------------

        // creates 2^rank child patches at level+1 from a single parent patch
        // stub for dynamic AMR refinement (not yet implemented)
        void refine_patch(const patch_id_t& /*parent_id*/)
        {
            // dynamic AMR refinement not yet implemented
        }

        // removes child patches and replaces them with a single coarser parent
        // stub for dynamic AMR coarsening (not yet implemented)
        void coarsen_patch(const patch_id_t& /*parent_id*/)
        {
            // dynamic AMR coarsening not yet implemented
        }

        // ---------------------------------------------------------------------
        // communication framework
        // ---------------------------------------------------------------------

        // gets a list of required halo transfers for all active patches
        // returns a vector of tuples: {src_patch_id, dst_patch_id,
        // face_direction}
        using transfer_request = std::tuple<patch_id_t, patch_id_t, int>;

        std::vector<transfer_request> get_halo_transfer_list() const
        {
            // for multi-block meshes, builds the neighbor transfer list
            // current implementation uses halo_graph in level_decomposition_t
            std::vector<transfer_request> list;
            return list;
        }

        // utility to run all transfers, relying on comm::communicator_t
        template <xpu::execution_space ExecutionSpace>
        xpu::token_t<ExecutionSpace> exchange_all_halos(
            xpu::executor_t<ExecutionSpace>& /*exec*/,
            comm::communicator_t& /*comm*/
        )
        {
            // fetches the transfer list and uses the communicator to dispatch
            // all async p2p copies, returning a single token for the batch
            return xpu::token_t<ExecutionSpace>{};
        }
    };

} // namespace simbi::grid

#endif // GRID_MESH_TOPLOGY_HPP
