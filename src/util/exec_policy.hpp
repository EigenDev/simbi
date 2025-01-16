#ifndef EXEC_POLICY_HPP
#define EXEC_POLICY_HPP

#include "build_options.hpp"   // for dim3, luint, global::col_maj, simbiStream_t
#include "device_api.hpp"      // for api::setDevice
#include <exception>           // for exception
#include <iostream>            // for operator<<, char_traits, basic_ostream
#include <vector>              // for vector

struct ExecutionException : public std::exception {
    const char* what() const throw() { return "Invalid constructor args"; }
};

namespace simbi {

    struct ExecutionPolicyConfig {
        size_t sharedMemBytes              = 0;
        std::vector<simbiStream_t> streams = {};
        std::vector<int> devices           = {0};   // Default to device 0
        size_t batch_size                  = 1024;
        size_t min_elements_per_thread     = 1;
    };

    template <typename T = luint, typename U = luint>
    struct ExecutionPolicy {
        T nzones;
        dim3 gridSize;
        dim3 blockSize;
        dim3 stride;
        size_t sharedMemBytes;
        std::vector<simbiStream_t> streams;
        std::vector<int> devices;
        T nzones_per_device;
        std::vector<dim3> device_gridSizes;
        size_t batch_size;
        size_t min_elements_per_thread;

        ~ExecutionPolicy() = default;
        ExecutionPolicy()  = default;

        // Vector constructor with config
        ExecutionPolicy(
            const std::vector<T>& gridSizes,
            const std::vector<U>& blockSizes,
            const std::vector<U>& strides       = {},
            const ExecutionPolicyConfig& config = {}
        )
            : sharedMemBytes(config.sharedMemBytes),
              streams(config.streams),
              devices(config.devices)
        {
            if (gridSizes.size() != blockSizes.size()) {
                throw ExecutionException();
            }
            init_devices();
            build_grid(gridSizes, blockSizes, strides);
        }

        T compute_blocks(const T nzones, const luint nThreads) const
        {
            return (nzones + nThreads - 1) / nThreads;
        }

        constexpr auto get_xextent() const
        {
            if constexpr (global::col_maj) {
                return blockSize.y;
            }
            return blockSize.x;
        }

        constexpr auto get_yextent() const
        {
            if constexpr (global::col_maj) {
                return blockSize.x;
            }
            return blockSize.y;
        }

        constexpr auto get_full_extent() const
        {
            if constexpr (global::on_gpu) {
                return blockSize.z * gridSize.z * blockSize.x * blockSize.y *
                       gridSize.x * gridSize.y;
            }
            else {
                return nzones;
            }
        }

        // build grid to handle multiple devices
        void build_grid(
            const std::vector<T> gridList,
            const std::vector<U> blockList,
            const std::vector<U> strideList
        )
        {
            // Calculate total zoness
            if (gridList.size() == 1) {
                nzones = gridList[0];
            }
            else if (gridList.size() == 2) {
                nzones = gridList[0] * gridList[1];
            }
            else if (gridList.size() == 3) {
                nzones = gridList[0] * gridList[1] * gridList[2];
            }

            const int numDevices = devices.size();

            // Calculate number of zones per device
            nzones_per_device = (nzones + numDevices - 1) / numDevices;

            // Set block size based on input
            if (blockList.size() == 1) {
                blockSize = dim3(blockList[0]);
                stride    = dim3(strideList[0]);
            }
            else if (blockList.size() == 2) {
                if constexpr (global::col_maj) {
                    blockSize = dim3(blockList[1], blockList[0]);
                    stride    = dim3(strideList[1], strideList[0]);
                }
                else {
                    blockSize = dim3(blockList[0], blockList[1]);
                    stride    = dim3(strideList[0], strideList[1]);
                }
            }
            else if (blockList.size() == 3) {
                blockSize = dim3(blockList[0], blockList[1], blockList[2]);
                stride    = dim3(strideList[0], strideList[1], strideList[2]);
            }

            // Calculate grid size per device
            for (int dev = 0; dev < numDevices; dev++) {
                T dev_zones = (dev == numDevices - 1)
                                  ? (nzones - dev * nzones_per_device)
                                  : nzones_per_device;

                if (gridList.size() == 1) {
                    device_gridSizes[dev] =
                        dim3((dev_zones + blockSize.x - 1) / blockSize.x);
                }
                else if (gridList.size() == 2) {
                    // Handle 2D case considering column/row major ordering
                    luint nxBlocks =
                        (gridList[0] + blockSize.x - 1) / blockSize.x;
                    luint nyBlocks =
                        (gridList[1] + blockSize.y - 1) / blockSize.y;
                    if constexpr (global::col_maj) {
                        device_gridSizes[dev] = dim3(nyBlocks, nxBlocks);
                    }
                    else {
                        device_gridSizes[dev] = dim3(nxBlocks, nyBlocks);
                    }
                }
                else if (gridList.size() == 3) {
                    // Handle 3D case
                    luint nxBlocks =
                        (gridList[0] + blockSize.x - 1) / blockSize.x;
                    luint nyBlocks =
                        (gridList[1] + blockSize.y - 1) / blockSize.y;
                    luint nzBlocks =
                        (gridList[2] + blockSize.z - 1) / blockSize.z;
                    device_gridSizes[dev] = dim3(nxBlocks, nyBlocks, nzBlocks);
                }
            }

            // Set default grid size to first device's grid size
            this->gridSize = device_gridSizes[0];
        }

        dim3 get_device_gridSize(int device) const
        {
            if (device < device_gridSizes.size()) {
                return device_gridSizes[device];
            }
            return dim3(0);
        }

        bool switch_to_device(int device) const
        {
            if (device < devices.size()) {
                gpu::api::setDevice(device);
                // this->gridSize = device_gridSizes[device];
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
                    blockSize.x * blockSize.y * blockSize.z;
                batch_size = threads_per_block * min_elements_per_thread;
            }
        }

        size_t get_num_batches(size_t sz) const
        {
            return (sz + batch_size - 1) / batch_size;
        }

      private:
        void init_devices()
        {
            if (devices.empty()) {
                devices = {0};
            }
            // Initialize per-device resources
            device_gridSizes.resize(devices.size());
            nzones_per_device = (nzones + devices.size() - 1) / devices.size();

            // Set initial device
            set_device(devices[0]);
        }
    };

}   // namespace simbi

#endif