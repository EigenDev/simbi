// =============================================================================
// communicator.hpp
//
// [TODO: Add description of what this file does]
//
// usage:
//   [TODO: Add usage example]
// =============================================================================
#pragma once

#include "compute/computation.hpp"
#include "containers/vector.hpp"
#include "decorators.hpp"
#include "grid/amr/prolongation.hpp"
#include "grid/amr/restriction.hpp"
#include "grid/domain.hpp"
#include "grid/exchange_pattern.hpp"
#include "grid/field.hpp"
#include "grid/patch_id.hpp"
#include "xpu/xpu.hpp"

#include <array>
#include <cstdint>
#include <functional>
#include <map>
#include <vector>

namespace simbi::grid::comm {

    struct communicator_t
    {

        template <xpu::execution_space_c ExecutionSpace>
        using executor_lookup_f = std::function<xpu::executor_t<ExecutionSpace>&(int)>;

        template <typename T, std::uint64_t Rank, xpu::execution_space_c ExecutionSpace>
        static void exchange_halos(
            const std::vector<transfer_op_t<Rank>>&       pattern,
            const std::map<patch_id_t, field_t<T, Rank>>& patches,
            executor_lookup_f<ExecutionSpace>             get_exec
        )
        {
            for (const auto& op : pattern) {
                auto src_it = patches.find(op.src_id);
                auto dst_it = patches.find(op.dst_id);

                if (src_it == patches.end() || dst_it == patches.end()) {
                    continue;
                }

                const auto& src_field = src_it->second;
                auto&       dst_field = dst_it->second;

                auto& executor = get_exec(src_field.device_id());

                // check levels
                std::int64_t src_level = op.src_id.level;
                std::int64_t dst_level = op.dst_id.level;

                // create views
                auto src_view = src_field[op.send_box]; // global coords
                auto dst_view = dst_field[op.recv_box]; // global coords

                if (src_level == dst_level) {
                    // === case a: peer-to-peer copy ===
                    // unified memory: simple copy via dispatch
                    auto copy_domain =
                        grid::extents<1>({static_cast<std::int64_t>(src_view.size())});
                    executor.dispatch(
                        copy_domain,
                        [src = src_view.data(),
                         dst = dst_view.data()] DUAL(const std::array<std::int64_t, 1>& idx) {
                            dst[idx[0]] = src[idx[0]];
                        }
                    );
                }
                else if (src_level < dst_level) {
                    // === case b: coarse -> fine (prolongation) ===
                    // we need to interpolate.
                    // "fill dst_view by sampling src_view"

                    // refinement ratio (assuming 2 for now)
                    simbi::iarray<Rank> ratio;
                    ratio.fill(1 << (dst_level - src_level));

                    // select interpolator (e.g., linear or parabolic)
                    // in a real engine, this is a policy configuration.
                    auto interpolator = amr::prolong<2>(src_view, ratio);

                    // functional assignment via field commitment
                    dst_view = compute::computation(op.recv_box, interpolator).with(executor);
                }
                else {
                    // === case c: fine -> coarse (restriction) ===
                    // usually handled in a separate 'sync' step,
                    // but sometimes needed for ghosts if the grid is wild.

                    iarray<Rank> ratio;
                    ratio.fill(1 << (src_level - dst_level));

                    auto restrictor = amr::restrict(src_view, ratio);

                    // functional assignment via field commitment
                    dst_view = compute::computation(op.recv_box, restrictor).with(executor);
                }
            }
        }
    };

} // namespace simbi::grid::comm


