/**
 *  *=============================================================================
 *  *           SIMBI - Special Relativistic Magnetohydrodynamics Code
 *  *=============================================================================
 *  *
 *  * @file            mesh.hpp
 *  * @brief           provides mesh information for the simulation
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
#ifndef MESH_HPP
#define MESH_HPP
#include "build_options.hpp"
#include "cell.hpp"
#include "geometry_manager.hpp"
#include "grid_manager.hpp"
#include "util/tools/helpers.hpp"

namespace simbi {
    // Mesh class
    template <size_type Dims>
    class Mesh
    {
      private:
        // Static type checks
        static_assert(Dims >= 1 && Dims <= 3, "Invalid dimension");
        GeometryManager geometry_;   // handles coordinates and metric
        GridManager grid_;           // handles indexing and dimensions

      public:
        // Constructors
        Mesh() = default;

        Mesh(const InitialConditions& init) : geometry_(init), grid_(init) {}

        DUAL auto get_cell_from_indices(
            const luint ii,
            const luint jj = 0,
            const luint kk = 0
        ) const
        {
            return Cell<Dims>(geometry_, grid_, ii, jj, kk);
        }

        DUAL auto get_cell_from_global(const luint global_idx) const
        {
            auto coords =
                helpers::unravel_idx<Dims>(global_idx, grid_.dimensions());
            if constexpr (Dims == 1) {
                return get_cell_from_indices(coords[0]);
            }
            else if constexpr (Dims == 2) {
                auto [ii, jj] = coords;
                return get_cell_from_indices(ii, jj);
            }
            else {
                auto [ii, jj, kk] = coords;
                return get_cell_from_indices(ii, jj, kk);
            }
        }

        // retrieve cell from array_t
        DUAL auto retrieve_cell(const array_t<size_type, Dims>& coords) const
        {
            if constexpr (Dims == 1) {
                return get_cell_from_indices(coords[0]);
            }
            else if constexpr (Dims == 2) {
                return get_cell_from_indices(coords[0], coords[1]);
            }
            else {
                return get_cell_from_indices(coords[0], coords[1], coords[2]);
            }
        }

        DUAL Maybe<size_type> get_neighbor_cell_index(
            size_type center_idx,
            int offset_i,
            int offset_j
        ) const
        {
            // get current cell coordinates
            auto cell     = get_cell_from_global(center_idx);
            auto centroid = cell.centroid();

            // calculate neighbor centroid position
            spatial_vector_t<real, Dims> neighbor_pos = centroid;
            neighbor_pos[0] += offset_i * cell.width(0);
            if constexpr (Dims > 1) {
                neighbor_pos[1] += offset_j * cell.width(1);
            }

            // Find cell closest to this position
            // TODO: make more sophisticated if needed later
            real min_distance            = std::numeric_limits<real>::max();
            Maybe<size_type> closest_idx = Nothing;

            // Check cells in a small radius around the original cell
            for (int di = -1; di <= 1; di++) {
                for (int dj = -1; dj <= 1; dj++) {
                    // Skip the current cell
                    if (di == 0 && dj == 0) {
                        continue;
                    }

                    // Calculate potential neighbor index
                    auto idx = center_idx + di + dj * grid().active_gridsize(0);
                    if (idx >= 0 && idx < size()) {
                        auto candidate_cell = get_cell_from_global(idx);
                        auto candidate_pos  = candidate_cell.centroid();

                        // Calculate distance
                        real distance = (candidate_pos - neighbor_pos).norm();
                        if (distance < min_distance) {
                            min_distance = distance;
                            closest_idx  = idx;
                        }
                    }
                }
            }

            return closest_idx;
        }

        // member accessors
        DUAL const auto& grid() const { return grid_; }
        DUAL const auto& geometry_state() const { return geometry_; }
        // non-const member accessors
        DUAL auto& grid() { return grid_; }
        DUAL auto& geometry_state() { return geometry_; }

        // accessors
        DUAL auto dimensions() const { return grid_.dimensions(); }
        DUAL auto active_dimensions() const
        {
            return grid_.active_dimensions();
        }
        DUAL auto halo_radius() const { return grid_.halo_radius(); }
        DUAL auto nhalos() const { return grid_.nhalos(); }
        DUAL auto geometry() const { return geometry_.geometry(); }
        DUAL auto is_half_sphere() const { return geometry_.is_half_sphere(); }
        DUAL auto spacing_type(int ii) const
        {
            return geometry_.spacing_type(ii);
        }
        DUAL auto min_bound(int ii) const { return geometry_.min_bound(ii); }
        DUAL auto max_bound(int ii) const { return geometry_.max_bound(ii); }
        DUAL auto homologous() const { return geometry_.homologous(); }
        DUAL auto expansion_term() const { return geometry_.expansion_term(); }
        DUAL auto mesh_is_moving() const { return geometry_.mesh_is_moving(); }
        auto geometry_to_c_str() const { return geometry_.geometry_to_c_str(); }

        DUAL auto size() const { return grid_.active_zones(); }
    };
}   // namespace simbi
#endif   // MESH_HPP
