// =============================================================================
// cpu_dispatch.hpp
//
// cpu execution space dispatch implementation for parallel domain iteration.
// supports both serial and openmp modes with cache-friendly tiling.
//
// design principles:
//   - tiled iteration for cache locality (mimics cuda block structure)
//   - serial mode for debugging and testing
//   - openmp parallel mode with configurable tile sizes
//   - consistent coordinate access: coord[0]=z, coord[1]=y, coord[2]=x for 3d
//
// usage:
//   cpu_dispatch(domain, tile_size, use_serial, [](auto idx) { /* work */ });
// =============================================================================
#pragma once

#include "grid/domain.hpp"
#include "xpu/core/types.hpp"

#include <algorithm>
#include <cstdint>

namespace simbi::xpu::exec::detail {

    // =============================================================================
    // serial dispatch (no threading, for debugging)
    // =============================================================================

    template <typename Func>
    void cpu_dispatch_serial_1d(const grid::domain_t<1>& domain, Func&& func)
    {
        auto coord = domain.start;

        for (coord[0] = domain.start[0]; coord[0] < domain.fin[0]; ++coord[0]) {
            func(coord);
        }
    }

    template <typename Func>
    void cpu_dispatch_serial_2d(const grid::domain_t<2>& domain, Func&& func)
    {
        auto coord = domain.start;

        for (coord[0] = domain.start[0]; coord[0] < domain.fin[0]; ++coord[0]) {
            for (coord[1] = domain.start[1]; coord[1] < domain.fin[1]; ++coord[1]) {
                func(coord);
            }
        }
    }

    template <typename Func>
    void cpu_dispatch_serial_3d(const grid::domain_t<3>& domain, Func&& func)
    {
        auto coord = domain.start;

        for (coord[0] = domain.start[0]; coord[0] < domain.fin[0]; ++coord[0]) {
            for (coord[1] = domain.start[1]; coord[1] < domain.fin[1]; ++coord[1]) {
                for (coord[2] = domain.start[2]; coord[2] < domain.fin[2]; ++coord[2]) {
                    func(coord);
                }
            }
        }
    }

    // =============================================================================
    // tiled openmp dispatch (cache-friendly parallel execution)
    // =============================================================================

    template <typename Func>
    void cpu_dispatch_tiled_1d(
        const grid::domain_t<1>& domain,
        const core::dim3_t&      tile_size,
        Func&&                   func
    )
    {
        const auto shape       = domain.shape();
        const auto tx          = std::max<std::uint64_t>(1, tile_size.x);
        const auto num_tiles_x = (shape[0] + tx - 1) / tx;

#pragma omp parallel for schedule(static)
        for (std::uint64_t tile_x = 0; tile_x < num_tiles_x; ++tile_x) {
            grid::domain_t<1> tile = domain;
            tile.start[0]          = domain.start[0] + tile_x * tx;
            tile.fin[0]            = std::min<std::int64_t>(tile.start[0] + tx, domain.fin[0]);

            cpu_dispatch_serial_1d(tile, func);
        }
    }

    template <typename Func>
    void cpu_dispatch_tiled_2d(
        const grid::domain_t<2>& domain,
        const core::dim3_t&      tile_size,
        Func&&                   func
    )
    {
        const auto shape       = domain.shape();
        const auto tx          = std::max<std::uint64_t>(1, tile_size.x);
        const auto ty          = std::max<std::uint64_t>(1, tile_size.y);
        const auto num_tiles_x = (shape[1] + tx - 1) / tx;
        const auto num_tiles_y = (shape[0] + ty - 1) / ty;

#pragma omp parallel for collapse(2) schedule(static)
        for (std::uint64_t tile_y = 0; tile_y < num_tiles_y; ++tile_y) {
            for (std::uint64_t tile_x = 0; tile_x < num_tiles_x; ++tile_x) {
                grid::domain_t<2> tile = domain;
                tile.start[0]          = domain.start[0] + tile_y * ty;
                tile.start[1]          = domain.start[1] + tile_x * tx;
                tile.fin[0]            = std::min<std::int64_t>(tile.start[0] + ty, domain.fin[0]);
                tile.fin[1]            = std::min<std::int64_t>(tile.start[1] + tx, domain.fin[1]);

                cpu_dispatch_serial_2d(tile, func);
            }
        }
    }

    template <typename Func>
    void cpu_dispatch_tiled_3d(
        const grid::domain_t<3>& domain,
        const core::dim3_t&      tile_size,
        Func&&                   func
    )
    {
        const auto shape       = domain.shape();
        const auto tx          = std::max<std::uint64_t>(1, tile_size.x);
        const auto ty          = std::max<std::uint64_t>(1, tile_size.y);
        const auto tz          = std::max<std::uint64_t>(1, tile_size.z);
        const auto num_tiles_x = (shape[2] + tx - 1) / tx;
        const auto num_tiles_y = (shape[1] + ty - 1) / ty;
        const auto num_tiles_z = (shape[0] + tz - 1) / tz;

#pragma omp parallel for collapse(3) schedule(static)
        for (std::uint64_t tile_z = 0; tile_z < num_tiles_z; ++tile_z) {
            for (std::uint64_t tile_y = 0; tile_y < num_tiles_y; ++tile_y) {
                for (std::uint64_t tile_x = 0; tile_x < num_tiles_x; ++tile_x) {
                    grid::domain_t<3> tile = domain;
                    tile.start[0]          = domain.start[0] + tile_z * tz;
                    tile.start[1]          = domain.start[1] + tile_y * ty;
                    tile.start[2]          = domain.start[2] + tile_x * tx;
                    tile.fin[0] = std::min<std::int64_t>(tile.start[0] + tz, domain.fin[0]);
                    tile.fin[1] = std::min<std::int64_t>(tile.start[1] + ty, domain.fin[1]);
                    tile.fin[2] = std::min<std::int64_t>(tile.start[2] + tx, domain.fin[2]);

                    cpu_dispatch_serial_3d(tile, func);
                }
            }
        }
    }

    // =============================================================================
    // generic dispatch with mode selection
    // =============================================================================

    template <std::uint64_t Rank, typename Func>
    void cpu_dispatch(
        const grid::domain_t<Rank>& domain,
        const core::dim3_t&         tile_size,
        bool                        use_serial,
        Func&&                      func
    )
    {
        if (use_serial) {
            if constexpr (Rank == 1) {
                cpu_dispatch_serial_1d(domain, std::forward<Func>(func));
            }
            else if constexpr (Rank == 2) {
                cpu_dispatch_serial_2d(domain, std::forward<Func>(func));
            }
            else if constexpr (Rank == 3) {
                cpu_dispatch_serial_3d(domain, std::forward<Func>(func));
            }
            else {
                const auto total_size = domain.size();
                for (std::uint64_t linear = 0; linear < total_size; ++linear) {
                    auto coord = domain.linear_to_coord(linear);
                    func(coord);
                }
            }
        }
        else {
            if constexpr (Rank == 1) {
                cpu_dispatch_tiled_1d(domain, tile_size, std::forward<Func>(func));
            }
            else if constexpr (Rank == 2) {
                cpu_dispatch_tiled_2d(domain, tile_size, std::forward<Func>(func));
            }
            else if constexpr (Rank == 3) {
                cpu_dispatch_tiled_3d(domain, tile_size, std::forward<Func>(func));
            }
            else {
                const auto total_size = domain.size();
#pragma omp parallel for schedule(static)
                for (std::uint64_t linear = 0; linear < total_size; ++linear) {
                    auto coord = domain.linear_to_coord(linear);
                    func(coord);
                }
            }
        }
    }

    // =============================================================================
    // cpu reduction with tiling support
    // =============================================================================

    template <std::uint64_t Rank, typename T, typename MapFunc, typename ReduceOp>
    T cpu_reduce_serial(
        const grid::domain_t<Rank>& domain,
        T                           init_value,
        MapFunc&&                   map_func,
        ReduceOp&&                  reduce_op
    )
    {
        const auto total_size = domain.size();
        T          result     = init_value;

        for (std::uint64_t linear = 0; linear < total_size; ++linear) {
            auto coord        = domain.linear_to_coord(linear);
            T    mapped_value = map_func(coord);
            result            = reduce_op(result, mapped_value);
        }

        return result;
    }

    template <std::uint64_t Rank, typename T, typename MapFunc, typename ReduceOp>
    T cpu_reduce_parallel(
        const grid::domain_t<Rank>& domain,
        T                           init_value,
        MapFunc&&                   map_func,
        ReduceOp&&                  reduce_op
    )
    {
        const auto total_size = domain.size();
        T          result     = init_value;

#pragma omp parallel
        {
            T thread_local_result = init_value;

#pragma omp for schedule(static) nowait
            for (std::uint64_t linear = 0; linear < total_size; ++linear) {
                auto coord          = domain.linear_to_coord(linear);
                T    mapped_value   = map_func(coord);
                thread_local_result = reduce_op(thread_local_result, mapped_value);
            }

#pragma omp critical
            {
                result = reduce_op(result, thread_local_result);
            }
        }

        return result;
    }

    template <std::uint64_t Rank, typename T, typename MapFunc, typename ReduceOp>
    T cpu_reduce(
        const grid::domain_t<Rank>& domain,
        T                           init_value,
        MapFunc&&                   map_func,
        ReduceOp&&                  reduce_op,
        bool                        use_serial
    )
    {
        if (use_serial) {
            return cpu_reduce_serial(
                domain,
                init_value,
                std::forward<MapFunc>(map_func),
                std::forward<ReduceOp>(reduce_op)
            );
        }
        else {
            return cpu_reduce_parallel(
                domain,
                init_value,
                std::forward<MapFunc>(map_func),
                std::forward<ReduceOp>(reduce_op)
            );
        }
    }

} // namespace simbi::xpu::exec::detail
