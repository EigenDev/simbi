/**
 *  *=============================================================================
 *  *           SIMBI - Special Relativistic Magnetohydrodynamics Code
 *  *=============================================================================
 *  *
 *  * @file            exec_policy.hpp
 *  * @brief           Parrallelization configuration for GPU and CPU
 *  * @details
 *  *
 *  * @version         0.8.0
 *  * @date            2025-02-26
 *  * @author          Marcus DuPont
 *  * @email           marcus.dupont@princeton.edu
 *  *
 *  *==============================================================================
 *  * @build           Requirements & Dependencies
 *  *==============================================================================
 *  * @requires        C++20
 *  * @depends         CUDA >= 11.0, HDF5 >= 1.12, OpenMP >= 4.5
 *  * @platform        Linux, MacOS
 *  * @parallel        GPU (CUDA, HIP), CPU (OpenMP)
 *  *
 *  *==============================================================================
 *  * @documentation   Reference & Notes
 *  *==============================================================================
 *  * @usage
 *  * @note
 *  * @warning
 *  * @todo
 *  * @bug
 *  * @performance
 *  *
 *  *==============================================================================
 *  * @testing        Quality Assurance
 *  *==============================================================================
 *  * @test
 *  * @benchmark
 *  * @validation
 *  *
 *  *==============================================================================
 *  * @history        Version History
 *  *==============================================================================
 *  * 2025-02-26      v0.8.0      Initial implementation
 *  *
 *  *==============================================================================
 *  * @copyright (C) 2025 Marcus DuPont. All rights reserved.
 *  *==============================================================================
 */
#ifndef EXEC_POLICY_HPP
#define EXEC_POLICY_HPP

#include "adapter/device_adapter_api.hpp"   // for api::set_device
#include "adapter/device_types.hpp"
#include "config.hpp"              // std::uint64_t, global::col_maj,
#include "containers/vector.hpp"   // for array
#include <exception>               // for exception
#include <vector>                  // for vector

struct ExecutionException : public std::exception {
    const char* what() const throw() { return "Invalid constructor args"; }
};

using namespace simbi::adapter;
namespace simbi {

    enum class MemoryType {
        DEVICE,
        MANAGED,
        PINNED
    };

    enum class HaloExchangeMode {
        SYNC,
        ASYNC,
    };

    struct ExecutionPolicyConfig {
        std::uint64_t shared_mem_bytes           = 0;
        std::vector<adapter::stream_t<>> streams = {};
        std::vector<std::int64_t> devices        = {0};
        std::uint64_t batch_size                 = 1024;
        std::uint64_t min_elements_per_thread    = 1;
        bool enable_peer_access                  = true;
        std::uint64_t halo_radius                = 2;
        MemoryType memory_type                   = MemoryType::DEVICE;
        HaloExchangeMode halo_mode               = HaloExchangeMode::ASYNC;
    };

    template <typename T = std::uint64_t, typename U = std::uint64_t>
    struct ExecutionPolicy {
        T nzones;
        adapter::types::dim3 grid_size;
        adapter::types::dim3 block_size;
        std::uint64_t shared_mem_bytes;
        std::vector<adapter::stream_t<>> streams;
        std::vector<std::int64_t> devices;
        T nzones_per_device;
        std::vector<adapter::types::dim3> device_grid_sizes;
        std::uint64_t batch_size;
        std::uint64_t min_elements_per_thread;
        ExecutionPolicyConfig config;

        ~ExecutionPolicy() = default;
        ExecutionPolicy()  = default;

        template <std::uint64_t Dims>
        ExecutionPolicy(
            const simbi::vector_t<T, Dims>& grid_sizes,
            const simbi::vector_t<U, Dims>& block_sizes,
            const ExecutionPolicyConfig& config = {}
        )
            : shared_mem_bytes(config.shared_mem_bytes),
              streams(config.streams),
              devices(config.devices),
              batch_size(config.batch_size),
              min_elements_per_thread(config.min_elements_per_thread),
              config(config)
        {
            if (grid_sizes.size() != block_sizes.size()) {
                throw ExecutionException();
            }
            init_devices();
            build_grid(grid_sizes, block_sizes);
        }

        ExecutionPolicy(
            std::initializer_list<T> grid_sizes_list,
            std::initializer_list<U> block_sizes_list,
            const ExecutionPolicyConfig& config = {}
        )
            : shared_mem_bytes(config.shared_mem_bytes),
              streams(config.streams),
              devices(config.devices),
              batch_size(config.batch_size),
              min_elements_per_thread(config.min_elements_per_thread),
              config(config)
        {
            if (grid_sizes_list.size() != block_sizes_list.size() ||
                grid_sizes_list.size() > 3) {
                throw ExecutionException();
            }

            simbi::vector_t<T, 3> grid_sizes;
            simbi::vector_t<U, 3> block_sizes;

            // Copy the provided dimensions
            std::uint64_t dim = grid_sizes_list.size();
            for (std::uint64_t i = 0; i < dim; ++i) {
                grid_sizes[i]  = *(grid_sizes_list.begin() + i);
                block_sizes[i] = *(block_sizes_list.begin() + i);
            }

            // Set remaining dimensions to 1
            for (std::uint64_t i = dim; i < 3; ++i) {
                grid_sizes[i]  = 1;
                block_sizes[i] = 1;
            }

            init_devices();
            build_grid(grid_sizes, block_sizes);
        }

        T compute_blocks(const T nzones, const std::uint64_t nThreads) const
        {
            return (nzones + nThreads - 1) / nThreads;
        }

        constexpr auto get_xextent() const
        {
            if constexpr (global::col_major) {
                return block_size.y;
            }
            return block_size.x;
        }

        constexpr auto get_yextent() const
        {
            if constexpr (global::col_major) {
                return block_size.x;
            }
            return block_size.y;
        }

        constexpr auto get_full_extent() const { return nzones; }

        constexpr auto get_active_extent() const { return nzones; }

        constexpr auto set_shared_mem_bytes(std::uint64_t bytes)
        {
            shared_mem_bytes = bytes;
        }

        // build grid to handle multiple devices
        template <std::uint64_t Dims>
        void build_grid(
            const simbi::vector_t<T, Dims> grid_list,
            const simbi::vector_t<U, Dims> block_list
        )
        {
            // store original grid dimensions
            for (size_t ii = 0; ii < grid_list.size(); ii++) {
                this->grid_list[ii]  = grid_list[ii];
                this->block_list[ii] = block_list[ii];
            }

            // Calculate total zoness
            if (grid_list.size() == 1) {
                nzones = grid_list[0];
            }
            else if (grid_list.size() == 2) {
                nzones = grid_list[0] * grid_list[1];
            }
            else if (grid_list.size() == 3) {
                nzones = grid_list[0] * grid_list[1] * grid_list[2];
            }

            const std::int64_t numDevices = devices.size();

            // Calculate number of zones per device
            nzones_per_device = (nzones + numDevices - 1) / numDevices;

            // Set block size based on input
            if (block_list.size() == 1) {
                block_size = adapter::types::dim3(block_list[0]);
            }
            else if (block_list.size() == 2) {
                if constexpr (global::col_major) {
                    block_size =
                        adapter::types::dim3(block_list[1], block_list[0]);
                }
                else {
                    block_size =
                        adapter::types::dim3(block_list[0], block_list[1]);
                }
            }
            else if (block_list.size() == 3) {
                block_size = adapter::types::dim3(
                    block_list[0],
                    block_list[1],
                    block_list[2]
                );
            }

            // Calculate grid size per device
            for (std::int64_t dev = 0; dev < numDevices; dev++) {
                T dev_zones = (dev == numDevices - 1)
                                  ? (nzones - dev * nzones_per_device)
                                  : nzones_per_device;

                if (grid_list.size() == 1) {
                    device_grid_sizes[dev] = adapter::types::dim3(
                        (dev_zones + block_size.x - 1) / block_size.x
                    );
                }
                else if (grid_list.size() == 2) {
                    // Handle 2D case considering column/row major ordering
                    std::uint64_t nxBlocks =
                        (grid_list[0] + block_size.x - 1) / block_size.x;
                    std::uint64_t nyBlocks =
                        (grid_list[1] + block_size.y - 1) / block_size.y;
                    if constexpr (global::col_major) {
                        device_grid_sizes[dev] =
                            adapter::types::dim3(nyBlocks, nxBlocks);
                    }
                    else {
                        device_grid_sizes[dev] =
                            adapter::types::dim3(nxBlocks, nyBlocks);
                    }
                }
                else if (grid_list.size() == 3) {
                    // Handle 3D case
                    std::uint64_t nxBlocks =
                        (grid_list[0] + block_size.x - 1) / block_size.x;
                    std::uint64_t nyBlocks =
                        (grid_list[1] + block_size.y - 1) / block_size.y;
                    std::uint64_t nzBlocks =
                        (grid_list[2] + block_size.z - 1) / block_size.z;
                    device_grid_sizes[dev] =
                        adapter::types::dim3(nxBlocks, nyBlocks, nzBlocks);
                }
            }

            // Set default grid size to first device's grid size
            this->grid_size = device_grid_sizes[0];
        }

        adapter::types::dim3 get_device_grid_size(std::int64_t device) const
        {
            if (device < device_grid_sizes.size()) {
                return device_grid_sizes[device];
            }
            return adapter::types::dim3(0);
        }

        bool switch_to_device(std::int64_t device) const
        {
            if (device < devices.size()) {
                gpu::api::set_device(device);
                // this->grid_size = device_grid_sizes[device];
                return true;
            }
            return false;
        }

        void set_device(std::int64_t device)
        {
            this->devices = {device};
            gpu::api::set_device(device);
        }

        void set_devices(const std::vector<std::int64_t>& devices)
        {
            this->devices = devices;
        }

        void add_stream(adapter::stream_t<> stream)
        {
            streams.push_back(stream);
        }

        void synchronize() const
        {
            for (const auto& stream : streams) {
                gpu::api::stream_synchronize(stream);
            }
        }

        void optimize_batch_size()
        {
            if constexpr (platform::is_gpu) {
                const size_t threads_per_block =
                    block_size.x * block_size.y * block_size.z;
                batch_size = threads_per_block * min_elements_per_thread;
            }
        }

        std::uint64_t get_num_batches(std::uint64_t sz) const
        {
            return (sz + batch_size - 1) / batch_size;
        }

        ExecutionPolicy contract(std::uint64_t radius) const
        {
            simbi::vector_t<T, 3> new_grid_sizes;
            for (size_t ii = 0; ii < grid_list.size(); ii++) {
                new_grid_sizes[ii] =
                    grid_list[ii] - 2 * radius * (grid_list[ii] > 1);
            }

            // keep same config
            ExecutionPolicyConfig config{
              shared_mem_bytes,
              streams,
              devices,
              batch_size,
              min_elements_per_thread
            };

            return ExecutionPolicy(new_grid_sizes, block_list, config);
        }

        // similar contraction as before, but now we denote how much we want
        // to conrtact each dimension
        ExecutionPolicy contract(const std::vector<T>& radii) const
        {
            simbi::vector_t<T, 3> new_grid_sizes;
            for (size_t ii = 0; ii < grid_list.size(); ii++) {
                new_grid_sizes[ii] =
                    grid_list[ii] - 2 * radii[ii] * (grid_list[ii] > 1);
            }

            // keep same config
            ExecutionPolicyConfig config{
              shared_mem_bytes,
              streams,
              devices,
              batch_size,
              min_elements_per_thread
            };

            return ExecutionPolicy(new_grid_sizes, block_list, config);
        }

      private:
        simbi::vector_t<T, 3> grid_list  = {1, 1, 1};
        simbi::vector_t<U, 3> block_list = {1, 1, 1};

        void init_devices()
        {
            if (devices.empty()) {
                devices = {0};
            }
            // Initialize per-device resources
            device_grid_sizes.resize(devices.size());
            nzones_per_device = (nzones + devices.size() - 1) / devices.size();

            // Set initial device
            set_device(devices[0]);
        }
    };

}   // namespace simbi

#endif
