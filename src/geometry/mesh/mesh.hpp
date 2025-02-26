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
        DUAL Mesh(const InitialConditions& init) : geometry_(init), grid_(init)
        {
        }

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
        DUAL auto active_cells() const { return grid_.active_cells(); }
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
        DUAL auto geometry_to_c_str() const
        {
            return geometry_.geometry_to_c_str();
        }
    };
}   // namespace simbi
#endif   // MESH_HPP