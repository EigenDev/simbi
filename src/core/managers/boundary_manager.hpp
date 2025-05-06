/**
 *  *=============================================================================
 *  *           SIMBI - Special Relativistic Magnetohydrodynamics Code
 *  *=============================================================================
 *  *
 *  * @file            boundary_manager.hpp
 *  * @brief           a helper struct to manage boundary conditions for ndarray
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

#ifndef BOUNDARY_MANAGER_HPP
#define BOUNDARY_MANAGER_HPP

#include "build_options.hpp"
#include "core/managers/io_manager.hpp"
#include "core/traits.hpp"
#include "core/types/containers/array_view.hpp"
#include "core/types/containers/ndarray.hpp"
#include "core/types/monad/maybe.hpp"
#include "core/types/utility/enums.hpp"
#include "geometry/mesh/mesh.hpp"
#include "util/parallel/exec_policy.hpp"
#include "util/tools/algorithms.hpp"
#include <cstddef>

namespace simbi {
    struct conserved_tag;
    struct flux_tag;
    template <typename T, size_type Dims>
    class boundary_manager
    {
      public:
        template <typename TagType = conserved_tag>
        void sync_boundaries(
            const ExecutionPolicy<>& policy,
            ndarray<T, Dims>& full_array,
            const array_view<T, Dims>& interior_view,
            const ndarray<BoundaryCondition>& conditions,
            const Mesh<Dims>& mesh,
            const Maybe<const IOManager<Dims>*> io_manager = Nothing,
            const real time                                = 0.0,
            const real time_step                           = 0.0,
            const bool need_corners                        = false
        ) const
        {
            // Sync faces
            sync_faces<TagType>(
                policy,
                full_array,
                interior_view,
                conditions,
                io_manager.unwrap_or(nullptr),
                &mesh,
                time,
                time_step
            );

            // Sync corners if needed
            if constexpr (comp_ct_type != CTTYPE::MdZ) {
                if constexpr (is_conserved_v<T>) {
                    if (need_corners) {
                        sync_corners(
                            policy,
                            full_array,
                            interior_view,
                            conditions
                        );
                    }
                }
            }
        }

        template <typename TagType = conserved_tag>
        void sync_corners(
            const ExecutionPolicy<>& policy,
            ndarray<T, Dims>& full_array,
            const array_view<T, Dims>& interior_view,
            const ndarray<BoundaryCondition>& conditions
        ) const
        {
            auto* data = full_array.data();
            auto radii = [&]() {
                uarray<Dims> rad;
                for (size_type ii = 0; ii < Dims; ++ii) {
                    rad[ii] =
                        (full_array.shape()[ii] - interior_view.shape()[ii]) /
                        2;
                }
                return rad;
            }();

            parallel_for(
                policy,
                [data,
                 radii,
                 full_shape     = full_array.shape(),
                 full_strides   = full_array.strides(),
                 interior_shape = interior_view.shape(),
                 conditions_ptr = conditions.data(),
                 this] DEV(size_type idx) {
                    auto coords = unravel_idx(idx, full_shape);
                    auto rshape = interior_shape;
                    // reverse shape if row major
                    if constexpr (!global::col_major) {
                        std::reverse(rshape.begin(), rshape.end());
                    }

                    // Only process corner points
                    if (!is_corner_point(coords, rshape, radii)) {
                        return;
                    }

                    // Find which dimensions are at boundaries
                    simbi::array_t<std::pair<size_type, bool>, Dims>
                        boundary_dims;
                    size_type num_boundaries = 0;

                    for (size_type dim = 0; dim < Dims; ++dim) {
                        if (coords[dim] < radii[dim]) {
                            boundary_dims[num_boundaries++] = {
                              dim,
                              true
                            };   // true = lower bound
                        }
                        else if (coords[dim] >= rshape[dim] + radii[dim]) {
                            boundary_dims[num_boundaries++] = {
                              dim,
                              false
                            };   // false = upper bound
                        }
                    }

                    // Get interior indices for each boundary dimension
                    size_type interior_idx = 0;
                    auto int_coords        = coords;

                    for (size_type ii = 0; ii < num_boundaries; ++ii) {
                        auto [dim, is_lower] = boundary_dims[ii];
                        const auto bc_idx    = 2 * dim + (is_lower ? 0 : 1);
                        const auto bc        = conditions_ptr[bc_idx];

                        // Update coordinate based on boundary condition
                        switch (bc) {
                            case BoundaryCondition::REFLECTING:
                                int_coords[dim] =
                                    (is_lower)
                                        ? 2 * radii[dim] - coords[dim] - 1
                                        : 2 * (rshape[dim] + radii[dim]) -
                                              coords[dim] - 1;
                                break;
                            case BoundaryCondition::PERIODIC:
                                int_coords[dim] =
                                    (is_lower) ? rshape[dim] + coords[dim]
                                               : coords[dim] - rshape[dim];
                                break;
                            default:   // OUTFLOW
                                int_coords[dim] =
                                    (is_lower) ? radii[dim]
                                               : rshape[dim] + radii[dim] - 1;
                        }
                    }

                    // Calculate interior linear index
                    for (size_type d = 0; d < Dims; d++) {
                        interior_idx += int_coords[d] * full_strides[d];
                    }
                    // std::cout << coords << " -> " << int_coords << std::endl;

                    // Apply boundary conditions
                    data[idx] = data[interior_idx];

                    // Handle reflecting conditions for momentum/magnetic
                    // components
                    if constexpr (is_conserved_v<T>) {
                        for (size_type ii = 0; ii < num_boundaries; ++ii) {
                            auto [dim, is_lower] = boundary_dims[ii];
                            const auto bc_idx    = 2 * dim + (is_lower ? 0 : 1);
                            if (conditions_ptr[bc_idx] ==
                                BoundaryCondition::REFLECTING) {
                                data[idx].mcomponent(dim + 1) *= -1.0;
                                if constexpr (is_relativistic_mhd<T>::value) {
                                    data[idx].bcomponent(dim + 1) *= -1.0;
                                }
                            }
                        }
                    }
                }
            );
        }

        template <typename TagType = conserved_tag>
        void sync_faces(
            const ExecutionPolicy<>& policy,
            ndarray<T, Dims>& full_array,
            const array_view<T, Dims>& interior_view,
            const ndarray<BoundaryCondition>& conditions,
            const IOManager<Dims>* io_manager,
            const Mesh<Dims>* mesh,
            const real time,
            const real time_step
        ) const
        {
            auto* data       = full_array.data();
            const auto radii = [fshape = full_array.shape(),
                                ishape = interior_view.shape()]() {
                uarray<Dims> rad;
                for (size_type ii = 0; ii < Dims; ++ii) {
                    if constexpr (global::col_major) {
                        rad[ii] = (fshape[ii] - ishape[ii]) / 2;
                    }
                    else {
                        rad[ii] = (fshape[Dims - (ii + 1)] -
                                   ishape[Dims - (ii + 1)]) /
                                  2;
                    }
                }
                return rad;
            }();

            // validate io_manager before kernel launch
            if constexpr (is_conserved_v<T> &&
                          std::is_same_v<TagType, conserved_tag>) {
                if (!io_manager &&
                    std::any_of(
                        conditions.begin(),
                        conditions.end(),
                        [](auto bc) { return bc == BoundaryCondition::DYNAMIC; }
                    )) {
                    throw std::runtime_error(
                        "IO Manager required for dynamic boundary conditions"
                    );
                }
            }

            // copy necessary data to avoid pointer issues
            auto handle_dynamic_bc = [io_manager, time, time_step]() {
                if constexpr (is_conserved_v<T> &&
                              std::is_same_v<TagType, conserved_tag>) {
                    return [io_manager, time, time_step] DEV(
                               const auto& ghost_coords,
                               const BoundaryFace face,
                               const T& result
                           ) {
                        return io_manager->call_boundary_source(
                            face,
                            ghost_coords,
                            time,
                            time_step,
                            result
                        );
                    };
                }
                else {
                    // suppress compiler warning about unused captures
                    (void) io_manager;
                    (void) time;
                    (void) time_step;
                    // Return a no-op lambda when T is not conserved
                    return [] DEV(
                               const auto& coords,
                               const BoundaryFace fc,
                               const auto& val
                           ) { return val; };
                }
            }();

            parallel_for(
                policy,
                [data,
                 handle_dynamic_bc,
                 conditions_ptr = conditions.data(),
                 full_shape     = full_array.shape(),
                 full_strides   = full_array.strides(),
                 interior_shape = interior_view.shape(),
                 radii,
                 mesh,
                 this] DEV(size_type idx) {
                    auto coordinates = unravel_idx(idx, full_shape);
                    auto rshape      = interior_shape;
                    // reverse shape if row major
                    if constexpr (!global::col_major) {
                        std::reverse(rshape.begin(), rshape.end());
                    }

                    // Only process boundary points (automatically excludes
                    // corners)
                    if (!is_boundary_point(coordinates, rshape, radii)) {
                        return;
                    }

                    // Find which dimension's boundary we're on
                    int boundary_dim = -1;
                    bool is_lower    = false;
                    for (size_type dim = 0; dim < Dims; ++dim) {
                        if (coordinates[dim] < radii[dim]) {
                            boundary_dim = dim;
                            is_lower     = true;
                            break;
                        }
                        if (coordinates[dim] >= rshape[dim] + radii[dim]) {
                            boundary_dim = dim;
                            is_lower     = false;
                            break;
                        }
                    }

                    // Process boundary point
                    if (boundary_dim >= 0) {
                        size_t bc_idx = 2 * boundary_dim + (is_lower ? 0 : 1);
                        const auto bc = conditions_ptr[bc_idx];
                        const auto interior_idx = get_interior_idx(
                            coordinates,
                            boundary_dim,
                            interior_shape,
                            full_strides,
                            radii,
                            bc
                        );

                        // Apply boundary condition
                        switch (bc) {
                            case BoundaryCondition::DYNAMIC: {
                                if constexpr (is_conserved_v<T> &&
                                              std::is_same_v<
                                                  TagType,
                                                  conserved_tag>) {
                                    const auto int_coords =
                                        unravel_idx(interior_idx, full_shape);
                                    // get interior cell reference
                                    const auto interior_cell =
                                        mesh->retrieve_cell(int_coords);

                                    // calc the ghost layer depth based on
                                    // coordinates Determine how many layers
                                    // deep this ghost cell is
                                    size_type ghost_layer = 1;

                                    if (is_lower) {
                                        // for lower boundary, calculate
                                        // distance from the boundary in cell
                                        // units
                                        ghost_layer = radii[boundary_dim] -
                                                      coordinates[boundary_dim];
                                    }
                                    else {
                                        // for upper boundary, calculate
                                        // distance from the boundary in cell
                                        // units
                                        ghost_layer =
                                            coordinates[boundary_dim] -
                                            (rshape[boundary_dim] +
                                             radii[boundary_dim] - 1);
                                    }

                                    // calc the physical position of this
                                    // ghost cell
                                    auto ghost_coords =
                                        interior_cell.calculate_ghost_position(
                                            boundary_dim,
                                            is_lower,
                                            ghost_layer
                                        );

                                    T dynamic_conserved = handle_dynamic_bc(
                                        ghost_coords,
                                        static_cast<BoundaryFace>(bc_idx),
                                        data[interior_idx]
                                    );
                                    data[idx] = dynamic_conserved;
                                }
                                else {
                                    // default to outflow or maybe reflecting ?
                                    // TODO: add support for non-conserved types
                                    data[idx] = data[interior_idx];
                                    // data[idx] = apply_reflecting(
                                    //     data[interior_idx],
                                    //     boundary_dim + 1
                                    // );
                                }
                                break;
                            }
                            case BoundaryCondition::REFLECTING:
                                (void) handle_dynamic_bc;
                                data[idx] = apply_reflecting<TagType>(
                                    data[interior_idx],
                                    boundary_dim + 1,
                                    mesh->get_cell_from_global(interior_idx),
                                    mesh->geometry()
                                );
                                break;
                            case BoundaryCondition::PERIODIC:
                                (void) handle_dynamic_bc;
                                data[idx] = apply_periodic(data[interior_idx]);
                                break;
                            default:   // OUTFLOW
                                (void) handle_dynamic_bc;
                                data[idx] = data[interior_idx];
                        }
                    }
                }
            );
        }

      private:
        DEV uarray<Dims>
        unravel_idx(size_type idx, const uarray<Dims>& shape) const
        {
            return memory_layout_coordinates<Dims>(idx, shape);
        }

        DEV static bool is_boundary_point(
            const uarray<Dims>& coordinates,
            const uarray<Dims>& shape,
            const uarray<Dims>& radii
        )
        {
            int boundary_count = 0;
            for (size_type ii = 0; ii < Dims; ++ii) {
                if (coordinates[ii] < radii[ii] ||
                    coordinates[ii] >= shape[ii] + radii[ii]) {
                    boundary_count++;
                }
            }

            // True only if exactly one dimension is at a boundary
            return boundary_count == 1;
        }

        DEV static bool is_corner_point(
            const uarray<Dims>& coordinates,
            const uarray<Dims>& shape,
            const uarray<Dims>& radii
        )
        {
            size_type boundary_count = 0;

            // Count how many dimensions are at boundaries
            for (size_type ii = 0; ii < Dims; ++ii) {
                if (coordinates[ii] < radii[ii] ||
                    coordinates[ii] >= shape[ii] + radii[ii]) {
                    boundary_count++;
                }
                if (boundary_count >= 2) {
                    return true;   // We're at a corner when 2+ dimensions
                                   // are at boundaries
                }
            }
            return false;
        }

        DEV size_type get_interior_idx(
            const uarray<Dims>& coords,
            size_type dim,
            const uarray<Dims>& shape,
            const uarray<Dims>& strides,
            const uarray<Dims>& radii,
            BoundaryCondition bc
        ) const
        {
            // Copy coordinates
            auto int_coords = coords;

            auto tshape = shape;
            // get inverted copy of shape if we are in row major
            if constexpr (!global::col_major) {
                std::reverse(tshape.begin(), tshape.end());
            }

            // Adjust coordinate based on boundary condition
            switch (bc) {
                case BoundaryCondition::REFLECTING:
                    int_coords[dim] =
                        (coords[dim] < radii[dim])
                            ? 2 * radii[dim] - coords[dim] - 1
                            : 2 * (tshape[dim] + radii[dim]) - coords[dim] - 1;
                    break;

                case BoundaryCondition::PERIODIC:
                    int_coords[dim] = (coords[dim] < radii[dim])
                                          ? tshape[dim] + coords[dim]
                                          : coords[dim] - tshape[dim];
                    break;

                default:   // OUTFLOW or DYNAMIC
                    int_coords[dim] = (coords[dim] < radii[dim])
                                          ? radii[dim]
                                          : tshape[dim] + radii[dim] - 1;
            }

            // // Calculate linear index directly
            size_type idx = 0;
            for (size_type i = 0; i < Dims; i++) {
                idx += int_coords[i] * strides[i];
            }
            return idx;
        }

        template <typename TagType, typename U>
        DEV static U apply_reflecting(
            const U& val,
            int momentum_idx,
            const auto& cell,
            const Geometry geom
        )
        {
            auto result = val;
            if constexpr (is_conserved_v<T>) {
                // m_\theta only changes sign at equator
                if constexpr (std::is_same_v<TagType, conserved_tag>) {
                    if (geom == Geometry::SPHERICAL && !cell.at_pole()) {
                        result.mcomponent(momentum_idx) *= -1.0;
                        if constexpr (is_mhd<U>::value) {
                            result.bcomponent(momentum_idx) *= -1.0;
                        }
                    }
                    else {
                        result.mcomponent(momentum_idx) *= -1.0;
                    }
                }
                else {
                    result.mcomponent(momentum_idx) *= -1.0;
                }
            }
            return result;
        }

        template <typename U>
        DEV static U apply_periodic(const U& val)
        {
            return val;
        }

        template <typename U>
        DEV static U apply_outflow(const U& val)
        {
            return val;
        }
    };
}   // namespace simbi

#endif
