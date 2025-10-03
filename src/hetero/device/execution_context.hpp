#ifndef HETERO_DEVICE_EXECUTION_CONTEXT_HPP
#define HETERO_DEVICE_EXECUTION_CONTEXT_HPP

#include "../config.hpp"
#include "../core/backend_traits.hpp"
#include "../core/common_types.hpp"
#include "compat.hpp"

#include <cstddef>
#include <cstdint>
#include <utility>

namespace simbi::hetero {

    template <typename backend_t>
    class execution_index_t
    {
        dim3_t block_idx_;
        dim3_t thread_idx_;
        dim3_t block_dim_;
        dim3_t grid_dim_;

      public:
        static DEV execution_index_t current();

        DEV std::uint32_t block_x() const { return block_idx_.x; }
        DEV std::uint32_t block_y() const { return block_idx_.y; }
        DEV std::uint32_t block_z() const { return block_idx_.z; }

        DEV std::uint32_t thread_x() const { return thread_idx_.x; }
        DEV std::uint32_t thread_y() const { return thread_idx_.y; }
        DEV std::uint32_t thread_z() const { return thread_idx_.z; }

        DEV std::uint32_t block_dim_x() const { return block_dim_.x; }
        DEV std::uint32_t block_dim_y() const { return block_dim_.y; }
        DEV std::uint32_t block_dim_z() const { return block_dim_.z; }

        DEV std::uint32_t grid_dim_x() const { return grid_dim_.x; }
        DEV std::uint32_t grid_dim_y() const { return grid_dim_.y; }
        DEV std::uint32_t grid_dim_z() const { return grid_dim_.z; }

        DEV const dim3_t& block() const { return block_idx_; }
        DEV const dim3_t& thread() const { return thread_idx_; }
        DEV const dim3_t& block_dims() const { return block_dim_; }
        DEV const dim3_t& grid_dims() const { return grid_dim_; }

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

        DEV std::uint64_t total_threads() const
        {
            return block_dim_x() * grid_dim_x() * block_dim_y() * grid_dim_y() *
                   block_dim_z() * grid_dim_z();
        }

      private:
        DEV execution_index_t(
            dim3_t block_idx,
            dim3_t thread_idx,
            dim3_t block_dim,
            dim3_t grid_dim
        )
            : block_idx_(block_idx),
              thread_idx_(thread_idx),
              block_dim_(block_dim),
              grid_dim_(grid_dim)
        {
        }

        template <typename T>
        friend class execution_index_t;
    };

    template <>
    DEV inline execution_index_t<cpu_backend_t>
    execution_index_t<cpu_backend_t>::current()
    {
        return execution_index_t<cpu_backend_t>(
            {0, 0, 0},
            {0, 0, 0},
            {1, 1, 1},
            {1, 1, 1}
        );
    }

    template <>
    DEV inline execution_index_t<cuda_backend_t>
    execution_index_t<cuda_backend_t>::current()
    {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
        return execution_index_t<cuda_backend_t>(
            {blockIdx.x, blockIdx.y, blockIdx.z},
            {threadIdx.x, threadIdx.y, threadIdx.z},
            {blockDim.x, blockDim.y, blockDim.z},
            {gridDim.x, gridDim.y, gridDim.z}
        );
#else
        return execution_index_t<cuda_backend_t>(
            {0, 0, 0},
            {0, 0, 0},
            {1, 1, 1},
            {1, 1, 1}
        );
#endif
    }

    namespace grid {

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

        DEV inline execution_index_t<default_backend_t> idx()
        {
            return execution_index_t<default_backend_t>::current();
        }

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
    }   // namespace grid

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
            T old = *addr;
            *addr += val;
            return old;
#endif
        }

        template <typename T>
        DEV inline T atomic_min(T* addr, T val)
        {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
            return atomicMin(addr, val);
#else
            T old = *addr;
            *addr = (*addr < val) ? *addr : val;
            return old;
#endif
        }
    }   // namespace api
}   // namespace simbi::hetero

#endif
