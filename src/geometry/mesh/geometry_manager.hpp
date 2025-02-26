/**
 *  *=============================================================================
 *  *           SIMBI - Special Relativistic Magnetohydrodynamics Code
 *  *=============================================================================
 *  *
 *  * @file            geometry_manager.hpp
 *  * @brief           manages the grid geometry and mesh data and mesh motion
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

#ifndef GEOMETRY_MANAGER_HPP
#define GEOMETRY_MANAGER_HPP

#include "build_options.hpp"
#include "core/types/containers/array.hpp"
#include "core/types/utility/enums.hpp"
#include "core/types/utility/init_conditions.hpp"
#include <map>

namespace simbi {
    // map geometry string to simbi::Geometry enum class
    const std::unordered_map<std::string, simbi::Geometry> geometry_map = {
      {"spherical", simbi::Geometry::SPHERICAL},
      {"cartesian", simbi::Geometry::CARTESIAN},
      {"planar_cylindrical", simbi::Geometry::PLANAR_CYLINDRICAL},
      {"axis_cylindrical", simbi::Geometry::AXIS_CYLINDRICAL},
      {"cylindrical", simbi::Geometry::CYLINDRICAL}
    };

    std::unordered_map<std::string, simbi::Cellspacing> const str2cell = {
      {"log", simbi::Cellspacing::LOG},
      {"linear", simbi::Cellspacing::LINEAR}
      // {"log-linear",Cellspacing},
      // {"linear-log",Cellspacing},
    };
    class GeometryManager
    {
      private:
        Geometry geometry_;
        bool is_half_sphere_{false};

        // coordinate system details
        array_t<Cellspacing, 3> spacing_types_;
        array_t<real, 3> min_bounds_;
        array_t<real, 3> max_bounds_;

        // is the grid expanding homologously?
        bool homologous_{false};
        bool mesh_motion_{false};
        real expansion_term_{0.0};

      public:
        DUAL GeometryManager(const InitialConditions& init)
            : geometry_(geometry_map.at(init.coord_system)),
              spacing_types_{
                str2cell.at(init.x1_cell_spacing),
                str2cell.at(init.x2_cell_spacing),
                str2cell.at(init.x3_cell_spacing)
              },
              min_bounds_{
                init.x1bounds.first,
                init.x2bounds.first,
                init.x3bounds.first
              },
              max_bounds_{
                init.x1bounds.second,
                init.x2bounds.second,
                init.x3bounds.second
              },
              homologous_(init.homologous),
              mesh_motion_(init.mesh_motion)
        {
            if (geometry_ == Geometry::SPHERICAL) {
                is_half_sphere_ = max_bounds_[1] == 0.5 * M_PI;
            }
        }

        void move_grid(
            std::function<real(real)> const& a,
            std::function<real(real)> const& adot,
            const real t,
            const real time_step
        )
        {
            if (!mesh_motion_) {
                return;
            }

            auto update = [=, this](real x, real h) {
                return x + time_step * (homologous_ ? x * h : h);
            };

            expansion_term_ = adot(t) / a(t);
            min_bounds_[0]  = update(min_bounds_[0], expansion_term_);
            max_bounds_[1]  = update(max_bounds_[1], expansion_term_);
        }

        auto geometry_to_c_str() const
        {
            switch (geometry_) {
                case Geometry::CARTESIAN: return "cartesian";
                case Geometry::SPHERICAL: return "spherical";
                case Geometry::CYLINDRICAL: return "cylindrical";
                case Geometry::AXIS_CYLINDRICAL: return "axis_cylindrical";
                case Geometry::PLANAR_CYLINDRICAL: return "planar_cylindrical";
            }
        }

        // accessors
        DUAL constexpr Geometry geometry() const { return geometry_; }
        DUAL constexpr bool is_half_sphere() const { return is_half_sphere_; }
        DUAL constexpr Cellspacing spacing_type(int i) const
        {
            return spacing_types_[i];
        }
        DUAL constexpr real min_bound(int i) const { return min_bounds_[i]; }
        DUAL constexpr real max_bound(int i) const { return max_bounds_[i]; }
        DUAL constexpr auto min_bounds() const { return min_bounds_; }
        DUAL constexpr auto max_bounds() const { return max_bounds_; }
        DUAL constexpr bool homologous() const { return homologous_; }
        DUAL constexpr real expansion_term() const { return expansion_term_; }
        DUAL constexpr bool mesh_is_moving() const { return mesh_motion_; }
    };
}   // namespace simbi

#endif