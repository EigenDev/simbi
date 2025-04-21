/**
 *  *=============================================================================
 *  *           SIMBI - Special Relativistic Magnetohydrodynamics Code
 *  *=============================================================================
 *  *
 *  * @file            exec_policy_manager.hpp
 *  * @brief           a helper struct to manage execution policies for sim
 *  * @details
 *  *
 *  * @version         0.8.0
 *  * @date            2025-02-21
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
 *  * 2025-02-21      v0.8.0      Initial implementation
 *  *
 *  *==============================================================================
 *  * @copyright (C) 2025 Marcus DuPont. All rights reserved.
 *  *==============================================================================
 */

#ifndef EXEC_POLICY_MANAGER_HPP
#define EXEC_POLICY_MANAGER_HPP

#include "core/types/utility/init_conditions.hpp"   // for InitialConditions
#include "geometry/mesh/grid_manager.hpp"           // for GridManager
#include "util/parallel/exec_policy.hpp"            // for ExecutionPolicy
#include <numeric>

namespace simbi {
    template <size_type Dims>
    class ExecutionPolicyManager
    {
      private:
        // Move GPU-related members here
        size_type gpu_block_dimx_, gpu_block_dimy_, gpu_block_dimz_;
        ExecutionPolicy<> full_policy_, interior_policy_;
        ExecutionPolicy<> xvertex_policy_, yvertex_policy_, zvertex_policy_;
        ExecutionPolicy<> fullxvertex_policy_, fullyvertex_policy_,
            fullzvertex_policy_;
        std::vector<simbiStream_t> streams_;
        std::vector<int> devices_;

        // GPU configuration methods
        luint get_block_dims(const std::string& key) const
        {
            if (const char* val = std::getenv(key.c_str())) {
                return static_cast<luint>(std::stoi(val));
            }
            return 1;
        }

        std::vector<int> parse_device_list(const char* dev_list) const
        {
            std::vector<int> devices;
            std::string dev_str(dev_list);
            std::istringstream ss(dev_str);
            std::string token;
            while (std::getline(ss, token, ',')) {
                devices.push_back(std::stoi(token));
            }
            return devices;
        }

        void setup_multi_gpu(const InitialConditions& init)
        {
            if constexpr (global::on_gpu) {
                int device_count;
                gpu::api::getDeviceCount(&device_count);

                // grab the requested devices from environment or use all
                // available
                if (const char* dev_list = std::getenv("GPU_DEVICES")) {
                    devices_ = parse_device_list(dev_list);
                }
                else {
                    devices_.resize(device_count);
                    std::iota(devices_.begin(), devices_.end(), 0);
                }

                // create streams per device
                streams_.resize(devices_.size());
                for (size_t i = 0; i < devices_.size(); i++) {
                    gpu::api::setDevice(devices_[i]);
                    gpu::api::streamCreate(&streams_[i]);

                    // enable peer access if configured
                    if (init.enable_peer_access) {
                        for (int j = 0; j < devices_.size(); j++) {
                            if (i != j) {
                                gpu::api::enablePeerAccess(devices_[j]);
                            }
                        }
                    }
                }
            }
        }

      public:
        ExecutionPolicyManager(
            const GridManager& grid,
            const InitialConditions& init
        )
            : gpu_block_dimx_(get_block_dims("GPU_BLOCK_X")),
              gpu_block_dimy_(get_block_dims("GPU_BLOCK_Y")),
              gpu_block_dimz_(get_block_dims("GPU_BLOCK_Z"))
        {
            setup_multi_gpu(init);
            compute_execution_policies(grid, init);
        }

        // Access methods
        const auto& full_policy() const { return full_policy_; }
        const auto& interior_policy() const { return interior_policy_; }
        const auto& xvertex_policy() const { return xvertex_policy_; }
        const auto& yvertex_policy() const { return yvertex_policy_; }
        const auto& zvertex_policy() const { return zvertex_policy_; }
        const auto& full_xvertex_policy() const { return fullxvertex_policy_; }
        const auto& full_yvertex_policy() const { return fullyvertex_policy_; }
        const auto& full_zvertex_policy() const { return fullzvertex_policy_; }

        void compute_execution_policies(
            const GridManager& grid,
            const InitialConditions& init
        )
        {
            ExecutionPolicyConfig config{
              .shared_mem_bytes        = 0,
              .streams                 = streams_,
              .devices                 = devices_,
              .batch_size              = 1024,
              .min_elements_per_thread = 1,
              .enable_peer_access      = init.enable_peer_access,
              .halo_radius             = grid.halo_radius(),
              .memory_type = init.managed_memory ? MemoryType::MANAGED
                                                 : MemoryType::DEVICE,
              .halo_mode   = HaloExchangeMode::ASYNC
            };

            auto xblockdim = std::min(grid.active_gridsize(0), gpu_block_dimx_);
            auto yblockdim = std::min(grid.active_gridsize(1), gpu_block_dimy_);
            auto zblockdim = std::min(grid.active_gridsize(2), gpu_block_dimz_);

            if constexpr (global::on_gpu) {
                if (xblockdim * yblockdim * zblockdim < global::WARP_SIZE) {
                    if (grid.total_gridsize(2) > 1) {
                        xblockdim = yblockdim = zblockdim = 4;
                    }
                    else if (grid.total_gridsize(1) > 1) {
                        xblockdim = yblockdim = 16;
                        zblockdim             = 1;
                    }
                    else {
                        xblockdim = 128;
                        yblockdim = zblockdim = 1;
                    }
                }
            }

            full_policy_ = ExecutionPolicy(
                grid.dimensions(),
                {xblockdim, yblockdim, zblockdim},
                config
            );
            fullxvertex_policy_ = ExecutionPolicy(
                grid.flux_shape(0),
                {xblockdim, yblockdim, zblockdim},
                config
            );
            fullyvertex_policy_ = ExecutionPolicy(
                grid.flux_shape(1),
                {xblockdim, yblockdim, zblockdim},
                config
            );
            fullzvertex_policy_ = ExecutionPolicy(
                grid.flux_shape(2),
                {xblockdim, yblockdim, zblockdim},
                config
            );
            interior_policy_ = full_policy_.contract(grid.halo_radius());
            xvertex_policy_  = fullxvertex_policy_.contract({0, 1, 1});
            yvertex_policy_  = fullyvertex_policy_.contract({1, 0, 1});
            zvertex_policy_  = fullzvertex_policy_.contract({1, 1, 0});
        }
    };

}   // namespace simbi
#endif
