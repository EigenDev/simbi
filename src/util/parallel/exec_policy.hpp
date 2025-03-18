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

#include "build_options.hpp"   // for dim3, luint, global::col_maj, simbiStream_t
#include "core/types/containers/array.hpp"   // for array
#include "util/tools/device_api.hpp"         // for api::setDevice
#include <exception>                         // for exception
#include <iostream>   // for operator<<, char_traits, basic_ostream
#include <ranges>     // for ranges::views::transform
#include <vector>     // for vector

struct ExecutionException : public std::exception {
    const char* what() const throw() { return "Invalid constructor args"; }
};

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
        size_type shared_mem_bytes         = 0;
        std::vector<simbiStream_t> streams = {};
        std::vector<int> devices           = {0};
        size_type batch_size               = 1024;
        size_type min_elements_per_thread  = 1;
        bool enable_peer_access            = true;
        size_type halo_radius              = 2;
        MemoryType memory_type             = MemoryType::DEVICE;
        HaloExchangeMode halo_mode         = HaloExchangeMode::ASYNC;
    };

    template <typename T = luint, typename U = luint>
    struct ExecutionPolicy {
        T nzones;
        dim3 grid_size;
        dim3 block_size;
        size_type shared_mem_bytes;
        std::vector<simbiStream_t> streams;
        std::vector<int> devices;
        T nzones_per_device;
        std::vector<dim3> device_grid_sizes;
        size_type batch_size;
        size_type min_elements_per_thread;
        ExecutionPolicyConfig config;

        ~ExecutionPolicy() = default;
        ExecutionPolicy()  = default;

        // Vector constructor with config
        ExecutionPolicy(
            const simbi::array_t<T, 3>& grid_sizes,
            const simbi::array_t<U, 3>& block_sizes,
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

        T compute_blocks(const T nzones, const luint nThreads) const
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

        constexpr auto get_full_extent() const
        {
            if constexpr (global::on_gpu) {
                return block_size.z * grid_size.z * block_size.x *
                       block_size.y * grid_size.x * grid_size.y;
            }
            else {
                return nzones;
            }
        }

        constexpr auto get_real_extent() const { return nzones; }

        // build grid to handle multiple devices
        void build_grid(
            const simbi::array_t<T, 3> grid_list,
            const simbi::array_t<U, 3> block_list
        )
        {
            // store original grid dimensions
            this->grid_list  = grid_list;
            this->block_list = block_list;

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

            const int numDevices = devices.size();

            // Calculate number of zones per device
            nzones_per_device = (nzones + numDevices - 1) / numDevices;

            // Set block size based on input
            if (block_list.size() == 1) {
                block_size = dim3(block_list[0]);
            }
            else if (block_list.size() == 2) {
                if constexpr (global::col_major) {
                    block_size = dim3(block_list[1], block_list[0]);
                }
                else {
                    block_size = dim3(block_list[0], block_list[1]);
                }
            }
            else if (block_list.size() == 3) {
                block_size = dim3(block_list[0], block_list[1], block_list[2]);
            }

            // Calculate grid size per device
            for (int dev = 0; dev < numDevices; dev++) {
                T dev_zones = (dev == numDevices - 1)
                                  ? (nzones - dev * nzones_per_device)
                                  : nzones_per_device;

                if (grid_list.size() == 1) {
                    device_grid_sizes[dev] =
                        dim3((dev_zones + block_size.x - 1) / block_size.x);
                }
                else if (grid_list.size() == 2) {
                    // Handle 2D case considering column/row major ordering
                    luint nxBlocks =
                        (grid_list[0] + block_size.x - 1) / block_size.x;
                    luint nyBlocks =
                        (grid_list[1] + block_size.y - 1) / block_size.y;
                    if constexpr (global::col_major) {
                        device_grid_sizes[dev] = dim3(nyBlocks, nxBlocks);
                    }
                    else {
                        device_grid_sizes[dev] = dim3(nxBlocks, nyBlocks);
                    }
                }
                else if (grid_list.size() == 3) {
                    // Handle 3D case
                    luint nxBlocks =
                        (grid_list[0] + block_size.x - 1) / block_size.x;
                    luint nyBlocks =
                        (grid_list[1] + block_size.y - 1) / block_size.y;
                    luint nzBlocks =
                        (grid_list[2] + block_size.z - 1) / block_size.z;
                    device_grid_sizes[dev] = dim3(nxBlocks, nyBlocks, nzBlocks);
                }
            }

            // Set default grid size to first device's grid size
            this->grid_size = device_grid_sizes[0];
        }

        dim3 get_device_grid_size(int device) const
        {
            if (device < device_grid_sizes.size()) {
                return device_grid_sizes[device];
            }
            return dim3(0);
        }

        bool switch_to_device(int device) const
        {
            if (device < devices.size()) {
                gpu::api::setDevice(device);
                // this->grid_size = device_grid_sizes[device];
                return true;
            }
            return false;
        }

        void set_device(int device)
        {
            this->devices = {device};
            gpu::api::setDevice(device);
        }

        void set_devices(const std::vector<int>& devices)
        {
            this->devices = devices;
        }

        void add_stream(simbiStream_t stream) { streams.push_back(stream); }

        void synchronize() const
        {
            for (const auto& stream : streams) {
                gpu::api::streamSynchronize(stream);
            }
        }

        void optimize_batch_size()
        {
            if constexpr (global::on_gpu) {
                const size_t threads_per_block =
                    block_size.x * block_size.y * block_size.z;
                batch_size = threads_per_block * min_elements_per_thread;
            }
        }

        size_type get_num_batches(size_type sz) const
        {
            return (sz + batch_size - 1) / batch_size;
        }

        ExecutionPolicy contract(size_type radius) const
        {
            simbi::array_t<T, 3> new_grid_sizes;
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
            simbi::array_t<T, 3> new_grid_sizes;
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
        simbi::array_t<T, 3> grid_list;
        simbi::array_t<U, 3> block_list;

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
