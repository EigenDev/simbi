// =============================================================================
// skeleton.hpp
//
// [TODO: Add description of what this file does]
//
// usage:
//   [TODO: Add usage example]
// =============================================================================
#pragma once

#include "block_info.hpp"
#include "domain.hpp"
#include "patch_id.hpp"

#include <cstddef>
#include <cstdint>
#include <map>

namespace simbi::grid {

    template <std::uint64_t Rank>
    class skeleton_t
    {
        // owns the metadata for the local partition's blocks
        // We use std::map to keep them ordered by ID (useful for deterministic
        // iteration)
        std::map<patch_id_t, block_info_t<Rank>> blocks_;

        // we might also store ghost/halo metadata here in the future
        // but let's keep it simple.

      public:
        // ---------------------------------------------------------------------
        // construction / mutation
        // ---------------------------------------------------------------------

        void add_block(const block_info_t<Rank>& info)
        {
            blocks_.insert_or_assign(info.id, info);
        }

        void remove_block(const patch_id_t& id) { blocks_.erase(id); }

        // ---------------------------------------------------------------------
        // access / query
        // ---------------------------------------------------------------------

        const block_info_t<Rank>* get_block(const patch_id_t& id) const
        {
            auto it = blocks_.find(id);
            if (it == blocks_.end()) {
                return nullptr;
            }
            return &it->second;
        }

        // subscript operator for convenient access
        block_info_t<Rank>& operator[](const patch_id_t& id)
        {
            return blocks_[id];
        }

        const block_info_t<Rank>& operator[](const patch_id_t& id) const
        {
            return blocks_.at(id);
        }

        // iterators for "for each block" loops
        auto begin() const { return blocks_.begin(); }
        auto end() const { return blocks_.end(); }
        std::size_t size() const { return blocks_.size(); }
        bool empty() const { return blocks_.empty(); }

        // ---------------------------------------------------------------------
        // global properties
        // ---------------------------------------------------------------------

        // calculate the union of all local domains (bounding box)
        // useful for load balancing checks
        domain_t<Rank> bounding_box() const
        {
            if (blocks_.empty()) {
                return {};
            }

            domain_t<Rank> box = blocks_.begin()->second.geometry;
            for (const auto& [id, block] : blocks_) {
                for (std::uint64_t dd = 0; dd < Rank; ++dd) {
                    box.start[dd] =
                        std::min(box.start[dd], block.geometry.start[dd]);
                    box.fin[dd] = std::max(box.fin[dd], block.geometry.fin[dd]);
                }
            }
            return box;
        }
    };

}   // namespace simbi::grid


