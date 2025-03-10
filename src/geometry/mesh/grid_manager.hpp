/**
 *  *=============================================================================
 *  *           SIMBI - Special Relativistic Magnetohydrodynamics Code
 *  *=============================================================================
 *  *
 *  * @file            grid_manager.hpp
 *  * @brief           provides generic grid information for mesh
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
#ifndef GRID_MANAGER_HPP
#define GRID_MANAGER_HPP

#include "build_options.hpp"                        // for DUAL
#include "core/types/containers/array.hpp"          // for array
#include "core/types/utility/enums.hpp"             // for Geometry, Cellspacing
#include "core/types/utility/init_conditions.hpp"   // for InitialConditions
#include <cmath>

namespace simbi {
    class GridManager
    {
      private:
        size_type nx_, ny_, nz_;      // Total grid dimensions
        size_type halo_radius_;       // Number of ghost cells per side
        size_type nhalos_;            // Total ghost cells (2 * radius)
        size_type xag_, yag_, zag_;   // Active grid dimensions
        size_type active_zones_;      // Total active cells

      public:
        // Constructors
        constexpr GridManager() = default;

        DUAL constexpr GridManager(const InitialConditions& init)
            : nx_(init.nx),
              ny_(init.ny),
              nz_(init.nz),
              halo_radius_(1 + (init.spatial_order == "plm"))
        {
            calculate_derived_quantities();
        }

        DUAL constexpr void calculate_derived_quantities()
        {
            nhalos_       = 2 * halo_radius_;
            xag_          = nx_ - nhalos_;
            yag_          = std::max<int>(1, ny_ - nhalos_);
            zag_          = std::max<int>(1, nz_ - nhalos_);
            active_zones_ = xag_ * yag_ * zag_;
        }

        // Grid  accessors
        DUAL auto dimensions() const
        {
            return array_t<size_type, 3>{nx_, ny_, nz_};
        }

        DUAL auto active_dimensions() const
        {
            return array_t<size_type, 3>{xag_, yag_, zag_};
        }

        DUAL auto flux_shape(const size_type ii) const
        {
            if (ii == 0) {
                return array_t<size_type, 3>{xag_ + 1, yag_ + 2, zag_ + 2};
            }
            else if (ii == 1) {
                return array_t<size_type, 3>{xag_ + 2, yag_ + 1, zag_ + 2};
            }
            else {
                return array_t<size_type, 3>{xag_ + 2, yag_ + 2, zag_ + 1};
            }
        }
        DUAL constexpr auto active_gridsize(int ii) const
        {
            return ii == 0 ? xag_ : ii == 1 ? yag_ : zag_;
        }
        DUAL constexpr auto total_gridsize(int ii) const
        {
            return ii == 0 ? nx_ : ii == 1 ? ny_ : nz_;
        }
        DUAL constexpr auto halo_radius() const { return halo_radius_; }
        DUAL constexpr auto nhalos() const { return nhalos_; }
        DUAL constexpr auto active_zones() const { return active_zones_; }
        DUAL constexpr auto total_zones() const { return nx_ * ny_ * nz_; }
    };
}   // namespace simbi

#endif