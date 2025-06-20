/**
 *  *=============================================================================
 *  *           SIMBI - Special Relativistic Magnetohydrodynamics Code
 *  *=============================================================================
 *  *
 *  * @file            emf_field.hpp
 *  * @brief           Electromagnetic Field at Cell Edges
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
#ifndef EMF_FIELD_HPP
#define EMF_FIELD_HPP

#include "config.hpp"
#include "core/containers/vector_field.hpp"
#include "core/utility/enums.hpp"
#include "physics/hydro/schemes/ct/ct_calculator.hpp"

namespace simbi {
    namespace ct {
        template <typename CTScheme>
        class EMField : public vector_field::VectorField<real, 3>
        {
          public:
            DUAL EMField(
                const general_vector_t<real, 3>& left,
                const general_vector_t<real, 3>& right
            )
                : vector_field::VectorField<real, 3>(left, right)
            {
            }

            DUAL EMField()
                : vector_field::VectorField<real, 3>(
                      general_vector_t<real, 3>{0.0, 0.0, 0.0},
                      general_vector_t<real, 3>{0.0, 0.0, 0.0}
                  )
            {
            }

            ~EMField() = default;

            template <int l_dir, int m_dir>
            DUAL void compute_edge_components(
                const auto& fri,
                const auto& gri,
                const auto& hri,
                const auto& bstag1,
                const auto& bstag2,
                const auto& bstag3,
                const auto& prims
            )
            {
                using ct_algo = scheme::EMFCalculator<CTScheme>;
                // Compute EMF based on CT scheme
                if constexpr (l_dir == 1 && m_dir == 2) {   // updating B3
                    // E_{1, i, j-1/2,k-1/2}
                    left_field[l_dir - 1] = ct_algo::template calc_edge_emf<
                        BlockAx::K,
                        Plane::JK,
                        Corner::SW>(hri, gri, bstag3, bstag2, prims, l_dir);

                    // E_{2, i-1/2, j ,k-1/2}
                    left_field[m_dir - 1] = ct_algo::template calc_edge_emf<
                        BlockAx::K,
                        Plane::IK,
                        Corner::SW>(hri, fri, bstag3, bstag1, prims, m_dir);

                    // E_{1, i, j+1/2,k-1/2}
                    right_field[l_dir - 1] = ct_algo::template calc_edge_emf<
                        BlockAx::K,
                        Plane::JK,
                        Corner::SE>(hri, gri, bstag3, bstag2, prims, l_dir);

                    // E_{2, i+1/2, j ,k-1/2}
                    right_field[m_dir - 1] = ct_algo::template calc_edge_emf<
                        BlockAx::K,
                        Plane::IK,
                        Corner::SE>(hri, fri, bstag3, bstag1, prims, m_dir);
                }
                else if constexpr (l_dir == 2 && m_dir == 3) {   // updating B1
                    // E_{2, i-1/2, j,k-1/2}
                    left_field[l_dir - 1] = ct_algo::template calc_edge_emf<
                        BlockAx::I,
                        Plane::IK,
                        Corner::SW>(hri, fri, bstag3, bstag1, prims, l_dir);

                    // E_{3, i-1/2, j-1/2, k}
                    left_field[m_dir - 1] = ct_algo::template calc_edge_emf<
                        BlockAx::I,
                        Plane::IJ,
                        Corner::SW>(gri, fri, bstag2, bstag1, prims, m_dir);

                    // E_{2, i-1/2, j,k+1/2}
                    right_field[l_dir - 1] = ct_algo::template calc_edge_emf<
                        BlockAx::I,
                        Plane::IK,
                        Corner::NW>(hri, fri, bstag3, bstag1, prims, l_dir);

                    // E_{3, i-1/2, j+1/2, k}
                    right_field[m_dir - 1] = ct_algo::template calc_edge_emf<
                        BlockAx::I,
                        Plane::IJ,
                        Corner::NW>(gri, fri, bstag2, bstag1, prims, m_dir);
                }
                else {   // updating B2
                    // E_{3, i-1/2, j-1/2, k}
                    left_field[l_dir - 1] = ct_algo::template calc_edge_emf<
                        BlockAx::J,
                        Plane::IJ,
                        Corner::SW>(gri, fri, bstag2, bstag1, prims, l_dir);

                    // E_{1, i, j-1/2, k-1/2}
                    left_field[m_dir - 1] = ct_algo::template calc_edge_emf<
                        BlockAx::J,
                        Plane::JK,
                        Corner::SW>(hri, gri, bstag3, bstag2, prims, m_dir);

                    // E_{3, i+1/2, j-1/2, k}
                    right_field[l_dir - 1] = ct_algo::template calc_edge_emf<
                        BlockAx::J,
                        Plane::IJ,
                        Corner::SE>(gri, fri, bstag2, bstag1, prims, l_dir);

                    // E_{1, i, j-1/2, k+1/2}
                    right_field[m_dir - 1] = ct_algo::template calc_edge_emf<
                        BlockAx::J,
                        Plane::JK,
                        Corner::NW>(hri, gri, bstag3, bstag2, prims, m_dir);
                }
            }
        };
    }   // namespace ct

}   // namespace simbi
#endif
