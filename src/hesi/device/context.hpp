#ifndef HET_DEVICE_CONTEXT_HPP
#define HET_DEVICE_CONTEXT_HPP

#include "compat.hpp"
#include "hesi/core/types.hpp"

#include <atomic>
#include <cstddef>
#include <cstdint>

namespace simbi::het {
    struct execution_coord_t {
        dim3_t block_idx;
        dim3_t thread_idx;
        dim3_t block_dim;
        dim3_t grid_dim;

        // thread rank within its block (0 to block_size - 1)
        DUAL std::uint64_t rank() const
        {
            return thread_idx.z * block_dim.x * block_dim.y +
                   thread_idx.y * block_dim.x + thread_idx.x;
        }

        // block rank within grid (0 to num_blocks - 1)
        DUAL std::uint64_t block_rank() const
        {
            return block_idx.z * grid_dim.x * grid_dim.y +
                   block_idx.y * grid_dim.x + block_idx.x;
        }

        // global linear thread id across entire grid
        DUAL std::uint64_t global_linear_id() const
        {
            std::uint64_t block_size = static_cast<std::uint64_t>(block_dim.x) *
                                       block_dim.y * block_dim.z;
            return block_rank() * block_size + rank();
        }

        // total threads launched in entire grid
        DUAL std::uint64_t total_threads() const
        {
            return static_cast<std::uint64_t>(block_dim.x) * block_dim.y *
                   block_dim.z * grid_dim.x * grid_dim.y * grid_dim.z;
        }

        // convenience accessors
        DUAL std::uint64_t thread_x() const { return thread_idx.x; }
        DUAL std::uint64_t thread_y() const { return thread_idx.y; }
        DUAL std::uint64_t thread_z() const { return thread_idx.z; }

        DUAL std::uint64_t block_x() const { return block_idx.x; }
        DUAL std::uint64_t block_y() const { return block_idx.y; }
        DUAL std::uint64_t block_z() const { return block_idx.z; }

        DUAL std::uint64_t block_dim_x() const { return block_dim.x; }
        DUAL std::uint64_t block_dim_y() const { return block_dim.y; }
        DUAL std::uint64_t block_dim_z() const { return block_dim.z; }

        DUAL std::uint64_t grid_dim_x() const { return grid_dim.x; }
        DUAL std::uint64_t grid_dim_y() const { return grid_dim.y; }
        DUAL std::uint64_t grid_dim_z() const { return grid_dim.z; }

        // query helpers
        DUAL bool is_first_in_block() const { return rank() == 0; }

        DUAL bool is_last_in_block() const
        {
            std::uint64_t block_size = block_dim.x * block_dim.y * block_dim.z;
            return rank() == block_size - 1;
        }

        DEV std::uint64_t global_thread_id() const
        {
            const auto block_size   = threads_per_block();
            const auto block_idx_1d = block_z() * grid_dim_x() * grid_dim_y() +
                                      block_y() * grid_dim_x() + block_x();
            const auto thread_idx_1d =
                thread_z() * block_dim_x() * block_dim_y() +
                thread_y() * block_dim_x() + thread_x();

            return block_idx_1d * block_size + thread_idx_1d;
        }

        DEV std::uint64_t thread_id() const
        {
            return thread_z() * block_dim_x() * block_dim_y() +
                   thread_y() * block_dim_x() + thread_x();
        }

        DEV std::uint64_t block_id() const
        {
            return block_z() * grid_dim_x() * grid_dim_y() +
                   block_y() * grid_dim_x() + block_x();
        }

        DEV std::uint64_t threads_per_block() const
        {
            return block_dim_x() * block_dim_y() * block_dim_z();
        }
    };

    // factory function: get current execution coordinate
    DEV inline execution_coord_t current()
    {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
        // gpu: read from hardware registers
        return execution_coord_t{
          {blockIdx.x, blockIdx.y, blockIdx.z},
          {threadIdx.x, threadIdx.y, threadIdx.z},
          {blockDim.x, blockDim.y, blockDim.z},
          {gridDim.x, gridDim.y, gridDim.z}
        };
#else
        // cpu: return dummy single-thread context
        return execution_coord_t{{0, 0, 0}, {0, 0, 0}, {1, 1, 1}, {1, 1, 1}};
#endif
    }
    namespace dgrid {
        class launch_config_t
        {
            dim3_t grid_;
            dim3_t block_;
            std::size_t shared_memory_;

          public:
            launch_config_t(
                dim3_t grid               = {1, 1, 1},
                dim3_t block              = {1, 1, 1},
                std::size_t shared_memory = 0
            )
                : grid_(grid), block_(block), shared_memory_(shared_memory)
            {
            }

            const dim3_t& grid() const { return grid_; }
            const dim3_t& block() const { return block_; }
            std::size_t shared_memory() const { return shared_memory_; }
        };

        DEV inline execution_coord_t ctx() { return current(); }

        inline launch_config_t
        config(std::uint32_t blocks, std::uint32_t threads)
        {
            return launch_config_t({blocks, 1, 1}, {threads, 1, 1});
        }

        inline launch_config_t
        config(dim3_t grid, dim3_t block, std::size_t shared_memory = 0)
        {
            return launch_config_t(grid, block, shared_memory);
        }

        inline dim3_t
        calculate_grid(std::size_t total_elements, dim3_t block_dims)
        {
            std::uint32_t blocks_x =
                (total_elements + block_dims.x - 1) / block_dims.x;
            return {blocks_x, 1, 1};
        }

        inline dim3_t
        calculate_grid_2d(std::uint32_t nx, std::uint32_t ny, dim3_t block_dims)
        {
            std::uint32_t blocks_x = (nx + block_dims.x - 1) / block_dims.x;
            std::uint32_t blocks_y = (ny + block_dims.y - 1) / block_dims.y;
            return {blocks_x, blocks_y, 1};
        }

        inline dim3_t calculate_grid_3d(
            std::uint32_t nx,
            std::uint32_t ny,
            std::uint32_t nz,
            dim3_t block_dims
        )
        {
            std::uint32_t blocks_x = (nx + block_dims.x - 1) / block_dims.x;
            std::uint32_t blocks_y = (ny + block_dims.y - 1) / block_dims.y;
            std::uint32_t blocks_z = (nz + block_dims.z - 1) / block_dims.z;
            return {blocks_x, blocks_y, blocks_z};
        }
    }   // namespace dgrid

    namespace api {
        DEV inline void sync_threads()
        {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
            __syncthreads();
#endif
        }

        template <typename T>
        DEV inline T atomic_add(T* addr, T val)
        {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
            if constexpr (std::is_same_v<T, float>) {
                return atomicAdd(addr, val);
            }
            else if constexpr (std::is_same_v<T, double>) {
                unsigned long long int* address_as_ull =
                    (unsigned long long int*) addr;
                unsigned long long int old = *address_as_ull, assumed;
                do {
                    assumed = old;
                    old     = atomicCAS(
                        address_as_ull,
                        assumed,
                        __double_as_longlong(
                            val + __longlong_as_double(assumed)
                        )
                    );
                } while (assumed != old);
                return __longlong_as_double(old);
            }
            else {
                return atomicAdd(addr, val);
            }
#else
            std::atomic_ref<T> ref(*addr);
            return ref.fetch_add(val);
#endif
        }

        template <typename T>
        DEV inline T atomic_min(T* addr, T val)
        {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
            return atomicMin(addr, val);
#else
            std::atomic_ref<T> ref(*addr);
            T old = ref.load();
            while (val < old && !ref.compare_exchange_weak(old, val))
                ;
            return old;
#endif
        }
    }   // namespace api

}   // namespace simbi::het

#endif
