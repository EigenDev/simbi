/**
 *  *=============================================================================
 *  *           SIMBI - Special Relativistic Magnetohydrodynamics Code
 *  *=============================================================================
 *  *
 *  * @file            stencil.hpp
 *  * @brief           Stencil View for CT Schemes
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
#ifndef STENCIL_HPP
#define STENCIL_HPP

#include "build_options.hpp"              // for global::on_gpu
#include "core/types/utility/enums.hpp"   // for Dir, Plane

namespace simbi {
    // map the corners to strings
    // static const std::map<Corner, std::string> corner_map{
    //   {Corner::SW, "SW"},
    //   {Corner::SE, "SE"},
    //   {Corner::NW, "NW"},
    //   {Corner::NE, "NE"}
    // };

    // // map the block axes to strings
    // static const std::map<BlockAx, std::string> blk_ax_map{
    //   {BlockAx::I, "I"},
    //   {BlockAx::J, "J"},
    //   {BlockAx::K, "K"}
    // };

    // // map the direction to strings
    // static const std::map<Dir, std::string> dir_map{
    //   {Dir::N, "N"},
    //   {Dir::S, "S"},
    //   {Dir::E, "E"},
    //   {Dir::W, "W"},
    //   {Dir::NE, "NE"},
    //   {Dir::SE, "SE"},
    //   {Dir::SW, "SW"},
    //   {Dir::NW, "NW"}
    // };

    namespace ct {
        struct DirectionPattern {
            int di;
            int dj;
            int dk;

            constexpr DirectionPattern(int i, int j, int k)
                : di(i), dj(j), dk(k)
            {
            }
        };

        template <
            BlockAx B,
            Plane P,
            Corner C,
            typename Flux,
            typename Primitive>
        class StencilView
        {
          public:
            // Constructor for creating stencil around a point
            StencilView(
                const Flux& vertical_flux,
                const Flux& horizontal_flux,
                const Primitive& primitives
            )
                : v_flux_(vertical_flux),
                  h_flux_(horizontal_flux),
                  prims_(primitives),
                  coordinates(primitives.position())
            {
            }

            // Get flux at stencil point based on plane-aware directions
            DUAL constexpr auto vertical_flux(Dir dir) const
            {
                auto [di, dj, dk] = get_vertical_offsets(dir, coordinates);
                return v_flux_.at(di, dj, dk);
            }

            DUAL constexpr auto horizontal_flux(Dir dir) const
            {
                auto [di, dj, dk] = get_horizontal_offsets(dir, coordinates);
                return h_flux_.at(di, dj, dk);
            }

            // Get primitive at stencil point
            DUAL constexpr auto prim(Dir dir) const
            {
                auto [di, dj, dk] = get_plane_offsets(dir);
                if constexpr (P == Plane::IJ) {
                    return prims_.at(di, dj, 0);
                }
                else if constexpr (P == Plane::JK) {
                    return prims_.at(0, dj, dk);
                }
                else {   // IK plane
                    return prims_.at(di, 0, dk);
                }
            }

          private:
            // static constexpr auto PLANE_PATTERNS = []() {
            //     std::array<DirectionPattern, 4> patterns;
            //     if constexpr (P == Plane::IJ) {
            //         patterns = {
            //           DirectionPattern{1, 0, 0},    // E
            //           DirectionPattern{-1, 0, 0},   // W
            //           DirectionPattern{0, 1, 0},    // N
            //           DirectionPattern{0, -1, 0}    // S
            //         };
            //     }
            //     else if constexpr (P == Plane::JK) {
            //         patterns = {
            //           DirectionPattern{0, 0, 1},    // N
            //           DirectionPattern{0, 0, -1},   // S
            //           DirectionPattern{0, 1, 0},    // E
            //           DirectionPattern{0, -1, 0}    // W
            //         };
            //     }
            //     else {   // IK plane
            //         patterns = {
            //           DirectionPattern{1, 0, 0},    // E
            //           DirectionPattern{-1, 0, 0},   // W
            //           DirectionPattern{0, 0, 1},    // N
            //           DirectionPattern{0, 0, -1}    // S
            //         };
            //     }
            // }();

            // Get offsets for each direction based on plane
            static constexpr auto get_plane_offsets(Dir dir)
            {
                // Base indices for cell-centers relative to corner
                constexpr int base_i = (is_east ? 0 : -1);
                constexpr int base_j =
                    ((P == Plane::JK) ? (is_east ? 0 : -1) : (is_north ? 0 : -1)
                    );
                constexpr int base_k = (is_north ? 0 : -1);

                if constexpr (P == Plane::IJ) {
                    // Handle cell-centered quantities
                    switch (dir) {
                        case Dir::NE:
                            return std::make_tuple(base_i + 1, base_j + 1, 0);
                        case Dir::SE:
                            return std::make_tuple(base_i + 1, base_j, 0);
                        case Dir::SW: return std::make_tuple(base_i, base_j, 0);
                        default:   // NW
                            return std::make_tuple(base_i, base_j + 1, 0);
                    }
                }
                else if constexpr (P == Plane::JK) {
                    // Handle cell-centered quantities
                    switch (dir) {
                        case Dir::NE:
                            return std::make_tuple(0, base_j + 1, base_k + 1);
                        case Dir::SE:
                            return std::make_tuple(0, base_j + 1, base_k);
                        case Dir::SW: return std::make_tuple(0, base_j, base_k);
                        default:   // NW
                            return std::make_tuple(0, base_j, base_k + 1);
                    }
                }
                else {   // IK plane
                    // Handle cell-centered quantities
                    switch (dir) {
                        case Dir::NE:
                            return std::make_tuple(base_i + 1, 0, base_k + 1);
                        case Dir::SE:
                            return std::make_tuple(base_i + 1, 0, base_k);
                        case Dir::SW: return std::make_tuple(base_i, 0, base_k);
                        default:   // NW
                            return std::make_tuple(base_i, 0, base_k + 1);
                    }
                }
            }

            static constexpr auto
            get_vertical_offsets(Dir dir, const auto& coordinates)
            {
                constexpr int ghost_offset = 1;

                // Face indices (-1/2 face = 0, +1/2 face = 1)
                constexpr int face_0 = 0;
                constexpr int face_1 = 1;

                // for eastern corners, the horizontal shift changes depending
                // on whether we want to go east of west
                // - For east corners: +1 if going east, 0 if going west
                // - For west corners: 0 if going east, -1 if going west
                const auto horizontal_shift =
                    (is_east == (dir == Dir::E)) ? (is_east ? 1 : -1) : 0;

                if constexpr (P == Plane::IJ) {
                    return std::make_tuple(
                        coordinates[0] + ghost_offset + horizontal_shift,
                        coordinates[1] + (is_north ? face_1 : face_0),
                        coordinates[2] + ghost_offset
                    );
                }
                else if constexpr (P == Plane::JK) {
                    // Handle face-centered quantities
                    return std::make_tuple(
                        coordinates[0] + ghost_offset,
                        coordinates[1] + ghost_offset + horizontal_shift,
                        coordinates[2] + (is_north ? face_1 : face_0)
                    );
                }
                else {   // IK plane
                    // Handle face-centered quantities
                    return std::make_tuple(
                        coordinates[0] + ghost_offset + horizontal_shift,
                        coordinates[1] + ghost_offset,
                        coordinates[2] + (is_north ? face_1 : face_0)
                    );
                }
            }

            static constexpr auto
            get_horizontal_offsets(Dir dir, const auto& coordinates)
            {
                constexpr int ghost_offset = 1;

                // Face indices (-1/2 face = 0, +1/2 face = 1)
                constexpr int face_0 = 0;
                constexpr int face_1 = 1;

                // for a given northern or southern corner, the vertical shift
                // changes depending on whether we want to go south of north
                // north of north, south of south or north of south
                // - For north corners: +1 if going north, 0 if going south
                // - For south corners: 0 if going north, -1 if going south
                const auto vertical_shift =
                    (is_north == (dir == Dir::N)) ? (is_north ? 1 : -1) : 0;

                if constexpr (P == Plane::IJ) {
                    return std::make_tuple(
                        coordinates[0] + (is_east ? face_1 : face_0),
                        coordinates[1] + ghost_offset + vertical_shift,
                        coordinates[2] + ghost_offset
                    );
                }
                else if constexpr (P == Plane::JK) {
                    // Handle face-centered quantities
                    return std::make_tuple(
                        coordinates[0] + ghost_offset,
                        coordinates[1] + (is_east ? face_1 : face_0),
                        coordinates[2] + ghost_offset + vertical_shift
                    );
                }
                else {   // IK plane
                    // Handle face-centered quantities
                    return std::make_tuple(
                        coordinates[0] + (is_east ? face_1 : face_0),
                        coordinates[1] + ghost_offset,
                        coordinates[2] + ghost_offset + vertical_shift
                    );
                }
            }

            alignas(64) const Flux& v_flux_;
            alignas(64) const Flux& h_flux_;
            alignas(64) const Primitive& prims_;
            alignas(64) const uarray<3> coordinates;

            static constexpr bool is_north =
                (C == Corner::NW || C == Corner::NE);
            static constexpr bool is_east =
                (C == Corner::NE || C == Corner::SE);
        };

        // type deduction guide
        // template <Plane P, Corner C, typename Flux, typename Primitive>
        // StencilView(const Flux&, const Flux&, const Primitive&)
        //     -> StencilView<P, C, Flux, Primitive>;

    }   // namespace ct
}   // namespace simbi

#endif