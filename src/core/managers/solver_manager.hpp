/**
 *  *=============================================================================
 *  *           SIMBI - Special Relativistic Magnetohydrodynamics Code
 *  *=============================================================================
 *  *
 *  * @file            solver_manager.hpp
 *  * @brief
 *  * @details
 *  *
 *  * @version         0.8.0
 *  * @date            2025-02-23
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
 *  * 2025-02-23      v0.8.0      Initial implementation
 *  *
 *  *==============================================================================
 *  * @copyright (C) 2025 Marcus DuPont. All rights reserved.
 *  *==============================================================================
 */

#ifndef SOLVER_MANAGER_HPP
#define SOLVER_MANAGER_HPP

#include "build_options.hpp"
#include "core/types/containers/ndarray.hpp"
#include "core/types/utility/enums.hpp"
#include "core/types/utility/init_conditions.hpp"
#include <utility>

namespace simbi {
    // bi-directional map for bcs
    template <typename T1, typename T2, size_t N>
    class BiMap
    {
      private:
        std::array<std::pair<T1, T2>, N> forward_map_;

      public:
        constexpr BiMap(std::array<std::pair<T1, T2>, N> init)
            : forward_map_(init)
        {
        }

        // Add initializer list constructor
        constexpr BiMap(std::initializer_list<std::pair<T1, T2>> init)
        {
            if (init.size() != N) {
                throw std::length_error(
                    "Initializer list size must match template parameter N"
                );
            }
            std::copy(init.begin(), init.end(), forward_map_.begin());
        }

        constexpr T2 forward(const T1& key) const
        {
            auto it = std::find_if(
                forward_map_.begin(),
                forward_map_.end(),
                [&key](const auto& pair) { return pair.first == key; }
            );
            if (it == forward_map_.end()) {
                throw std::runtime_error("Key not found in forward map");
            }
            return it->second;
        }

        constexpr T1 reverse(const T2& key) const
        {
            auto it = std::find_if(
                forward_map_.begin(),
                forward_map_.end(),
                [&key](const auto& pair) { return pair.second == key; }
            );
            if (it == forward_map_.end()) {
                throw std::runtime_error("Key not found in reverse map");
            }
            return it->first;
        }
    };

    class SolverManager
    {
      private:
        // Move maps inside class as static members
        static constexpr std::array<std::pair<std::string_view, Solver>, 3>
            solver_map_data = {
              {{"hllc", Solver::HLLC},
               {"hlle", Solver::HLLE},
               {"hlld", Solver::HLLD}}
        };

        static constexpr BiMap<std::string_view, BoundaryCondition, 4>
            boundary_map{
              {{"dynamic", BoundaryCondition::DYNAMIC},
               {"outflow", BoundaryCondition::OUTFLOW},
               {"reflecting", BoundaryCondition::REFLECTING},
               {"periodic", BoundaryCondition::PERIODIC}}
            };

        static Solver get_solver(std::string_view name)
        {
            auto it = std::find_if(
                solver_map_data.begin(),
                solver_map_data.end(),
                [name](const auto& pair) { return pair.first == name; }
            );
            if (it == solver_map_data.end()) {
                throw std::runtime_error(
                    "Invalid solver type: " + std::string(name)
                );
            }
            return it->second;
        }

        static BoundaryCondition get_boundary_condition(std::string_view name)
        {
            return boundary_map.forward(name);
        }

        static std::string_view get_boundary_name(BoundaryCondition bc)
        {
            return boundary_map.reverse(bc);
        }

        // Solver configuration
        Solver solver_type_;
        std::string spatial_order_;
        std::string temporal_order_;
        bool use_pcm_;
        bool using_rk1_;
        bool quirk_smoothing_;
        real plm_theta_, step_, viscosity_;

        // Physics flags
        bool null_gravity_{true};
        bool null_sources_{true};

        // Boundary conditions
        ndarray<BoundaryCondition> bcs_;

      public:
        SolverManager(const InitialConditions& init)
            : solver_type_(get_solver(init.solver)),
              spatial_order_(init.spatial_order),
              temporal_order_(init.temporal_order),
              use_pcm_(spatial_order_ == "pcm"),
              using_rk1_(temporal_order_ == "rk1"),
              quirk_smoothing_(init.quirk_smoothing),
              plm_theta_(init.plm_theta),
              step_((temporal_order_ == "rk1") ? 1.0 : 0.5),
              viscosity_(init.viscosity)
        {
            set_boundary_conditions(init.boundary_conditions);
        }

        // turn boundary conditions into a container of c strings
        // for use in the C API (e.g. HDF5)
        auto boundary_conditions_c_str() const
        {
            std::vector<const char*> c_strs;
            for (size_type ii = 0; ii < bcs_.size(); ++ii) {
                c_strs.push_back(get_boundary_name(bcs_[ii]).data());
            }
            return c_strs;
        }

        void set_boundary_conditions(const std::vector<std::string>& bcs)
        {
            for (auto&& bc : bcs) {
                bcs_.push_back_with_sync(get_boundary_condition(bc));
            }
            bcs_.sync_to_device();
        }

        // Accessors
        DUAL auto solver_type() const { return solver_type_; }
        DUAL auto spatial_order() const { return spatial_order_; }
        DUAL auto temporal_order() const { return temporal_order_; }
        DUAL bool is_pcm() const { return use_pcm_; }
        DUAL bool is_rk1() const { return using_rk1_; }
        DUAL bool is_quirk() const { return quirk_smoothing_; }
        DUAL auto plm_theta() const { return plm_theta_; }
        DUAL bool null_gravity() const { return null_gravity_; }
        DUAL bool null_sources() const { return null_sources_; }
        DUAL auto& boundary_conditions() const { return bcs_; }
        DUAL auto& boundary_conditions() { return bcs_; }
        DUAL auto step() const { return step_; }
        DUAL auto set_null_sources(bool state) { null_sources_ = state; }
        DUAL auto set_null_gravity(bool state) { null_gravity_ = state; }
        DUAL auto viscosity() const { return viscosity_; }
    };
}   // namespace simbi

#endif
