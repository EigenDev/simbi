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

        // GPU configuration methods
        DUAL luint get_block_dims(const std::string& key) const
        {
            if (const char* val = std::getenv(key.c_str())) {
                return static_cast<luint>(std::stoi(val));
            }
            return 1;
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
            compute_execution_policies(grid, init);
        }

        // Access methods
        DUAL const auto& full_policy() const { return full_policy_; }
        DUAL const auto& interior_policy() const { return interior_policy_; }
        DUAL const auto& xvertex_policy() const { return xvertex_policy_; }
        DUAL const auto& yvertex_policy() const { return yvertex_policy_; }
        DUAL const auto& zvertex_policy() const { return zvertex_policy_; }
        DUAL const auto& full_xvertex_policy() const
        {
            return fullxvertex_policy_;
        }
        DUAL const auto& full_yvertex_policy() const
        {
            return fullyvertex_policy_;
        }
        DUAL const auto& full_zvertex_policy() const
        {
            return fullzvertex_policy_;
        }

        constexpr void compute_execution_policies(
            const GridManager& grid,
            const InitialConditions& init
        )
        {
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

            const simbiStream_t stream = nullptr;

            full_policy_ = ExecutionPolicy(
                grid.dimensions(),
                {xblockdim, yblockdim, zblockdim},
                {.shared_mem_bytes = 0, .streams = {stream}, .devices = {0}}
            );
            fullxvertex_policy_ = ExecutionPolicy(
                grid.flux_shape(0),
                {xblockdim, yblockdim, zblockdim},
                {.shared_mem_bytes = 0, .streams = {stream}, .devices = {0}}
            );
            fullyvertex_policy_ = ExecutionPolicy(
                grid.flux_shape(1),
                {xblockdim, yblockdim, zblockdim},
                {.shared_mem_bytes = 0, .streams = {stream}, .devices = {0}}
            );
            fullzvertex_policy_ = ExecutionPolicy(
                grid.flux_shape(2),
                {xblockdim, yblockdim, zblockdim},
                {.shared_mem_bytes = 0, .streams = {stream}, .devices = {0}}
            );
            interior_policy_ = full_policy_.contract(grid.halo_radius());
            xvertex_policy_  = fullxvertex_policy_.contract({0, 1, 1});
            yvertex_policy_  = fullyvertex_policy_.contract({1, 0, 1});
            zvertex_policy_  = fullzvertex_policy_.contract({1, 1, 0});
        }
    };

}   // namespace simbi
#endif