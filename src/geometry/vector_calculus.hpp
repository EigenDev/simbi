/**
 *  *=============================================================================
 *  *           SIMBI - Special Relativistic Magnetohydrodynamics Code
 *  *=============================================================================
 *  *
 *  * @file            vector_calculus.hpp
 *  * @brief           general vector calculus operations
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
#ifndef VECTOR_CALCULUS_HPP
#define VECTOR_CALCULUS_HPP

#include "build_options.hpp"                        // for real, DUAL
#include "core/types/containers/vector_field.hpp"   // for Vector, VectorField

using namespace simbi::vector_field;

namespace simbi {
    namespace vector_calculus {

        enum class CurlComponent {
            X1 = 0,
            X2 = 1,
            X3 = 2
        };

        DUAL general_vector_t<real, 3>
        curl_cartesian(const auto& cell, const VectorField<real, 3>& vec_field)
        {
            return general_vector_t<real, 3>{
              (1.0 / cell.width(1)) *
                      (vec_field.right()[2] - vec_field.left()[2]) -
                  (1.0 / cell.width(2)) *
                      (vec_field.right()[1] - vec_field.left()[1]),
              (1.0 / cell.width(2)) *
                      (vec_field.right()[0] - vec_field.left()[0]) -
                  (1.0 / cell.width(0)) *
                      (vec_field.right()[2] - vec_field.left()[2]),
              (1.0 / cell.width(0)) *
                      (vec_field.right()[1] - vec_field.left()[1]) -
                  (1.0 / cell.width(1)) *
                      (vec_field.right()[0] - vec_field.left()[0]),
            };
        }

        DUAL general_vector_t<real, 3> curl_cylindrical(
            const auto& cell,
            const VectorField<real, 3>& vec_field
        )
        {
            const auto x1mean = cell.centroid_coordinate(0);
            const auto dr     = cell.x1R() - cell.x1L();
            const auto dphi   = cell.x2R() - cell.x2L();
            const auto dz     = cell.x3R() - cell.x3L();
            return general_vector_t<real, 3>{
              (1.0 / x1mean / dphi) *
                      (vec_field.right()[2] - vec_field.left()[2]) -
                  (1.0 / dz) * (vec_field.right()[1] - vec_field.left()[1]),
              (1.0 / dz) * (vec_field.right()[0] - vec_field.left()[0]) -
                  (1.0 / dr) * (vec_field.right()[2] - vec_field.left()[2]),
              (1.0 / x1mean) *
                  ((vec_field.right()[1] * cell.x1R() -
                    vec_field.left()[1] * cell.x1L()) /
                       dr -
                   (vec_field.right()[0] - vec_field.left()[0]) / dphi)
            };
        }

        DUAL general_vector_t<real, 3>
        curl_spherical(const auto& cell, const VectorField<real, 3>& vec_field)
        {
            const auto r      = cell.centroid_coordinate(0);
            const auto sint   = std::sin(cell.centroid_coordinate(1));
            const auto dr     = cell.x1R() - cell.x1L();
            const auto dtheta = cell.x2R() - cell.x2L();
            const auto dphi   = cell.x3R() - cell.x3L();
            const auto rcomp =
                (1.0 / (r * sint)) *
                ((1.0 / dtheta) * (vec_field.right()[2] * std::sin(cell.x2R()) -
                                   vec_field.left()[2] * std::sin(cell.x2L())) -
                 (1.0 / dphi) * (vec_field.right()[1] - vec_field.left()[1]));

            const auto tcomp =
                (1.0 / r) * ((1.0 / (dphi * sint)) *
                                 (vec_field.right()[0] - vec_field.left()[0]) -
                             (1.0 / dr) * (vec_field.right()[2] * cell.x1R() -
                                           vec_field.left()[2] * cell.x1L()));
            const auto pcomp =
                (1.0 / r) *
                ((1.0 / dr) * (vec_field.right()[1] * cell.x1R() -
                               vec_field.left()[1] * cell.x1L()) -
                 (1.0 / dtheta) * (vec_field.right()[0] - vec_field.left()[0]));

            return general_vector_t<real, 3>{rcomp, tcomp, pcomp};
        }

        DUAL general_vector_t<real, 3>
        curl(auto& cell, VectorField<real, 3>& vec_field)
        {
            switch (cell.geometry()) {
                case Geometry::CARTESIAN:
                    return curl_cartesian(cell, vec_field);
                case Geometry::CYLINDRICAL:
                    return curl_cylindrical(cell, vec_field);
                case Geometry::SPHERICAL:
                    return curl_spherical(cell, vec_field);
                default: return curl_cartesian(cell, vec_field);
            }
        }

        // Component-wise curl functions
        template <CurlComponent C>
        DUAL real curl_cartesian_component(
            const auto& cell,
            const VectorField<real, 3>& vec_field
        )
        {
            if constexpr (C == CurlComponent::X1) {
                return (1.0 / cell.width(1)) *
                           (vec_field.right()[2] - vec_field.left()[2]) -
                       (1.0 / cell.width(2)) *
                           (vec_field.right()[1] - vec_field.left()[1]);
            }
            else if constexpr (C == CurlComponent::X2) {
                return (1.0 / cell.width(2)) *
                           (vec_field.right()[0] - vec_field.left()[0]) -
                       (1.0 / cell.width(0)) *
                           (vec_field.right()[2] - vec_field.left()[2]);
            }
            else {
                return (1.0 / cell.width(0)) *
                           (vec_field.right()[1] - vec_field.left()[1]) -
                       (1.0 / cell.width(1)) *
                           (vec_field.right()[0] - vec_field.left()[0]);
            }
        }

        template <CurlComponent C>
        DUAL real curl_cylindrical_component(
            const auto& cell,
            const VectorField<real, 3>& vec_field
        )
        {
            const auto r    = cell.centroid_coordinate(0);
            const auto dr   = cell.x1R() - cell.x1L();
            const auto dphi = cell.x2R() - cell.x2L();
            const auto dz   = cell.x3R() - cell.x3L();
            if constexpr (C == CurlComponent::X1) {
                return (1.0 / r / dphi) *
                           (vec_field.right()[2] - vec_field.left()[2]) -
                       (1.0 / dz) *
                           (vec_field.right()[1] - vec_field.left()[1]);
            }
            else if constexpr (C == CurlComponent::X2) {
                return (1.0 / dz) *
                           (vec_field.right()[0] - vec_field.left()[0]) -
                       (1.0 / dr) *
                           (vec_field.right()[2] - vec_field.left()[2]);
            }
            else {
                return (1.0 / r) *
                       ((vec_field.right()[1] * cell.x1R() -
                         vec_field.left()[1] * cell.x1L()) /
                            dr -
                        (vec_field.right()[0] - vec_field.left()[0]) / dphi);
            }
        }

        template <CurlComponent C>
        DUAL real curl_spherical_component(
            const auto& cell,
            const VectorField<real, 3>& vec_field
        )
        {
            const auto r      = cell.centroid_coordinate(0);
            const auto sint   = std::sin(cell.centroid_coordinate(1));
            const auto dr     = cell.x1R() - cell.x1L();
            const auto dtheta = cell.x2R() - cell.x2L();
            const auto dphi   = cell.x3R() - cell.x3L();
            if constexpr (C == CurlComponent::X1) {
                return (1.0 / (r * sint)) *
                       ((1.0 / dtheta) *
                            (vec_field.right()[2] * std::sin(cell.x2R()) -
                             vec_field.left()[2] * std::sin(cell.x2L())) -
                        (1.0 / dphi) *
                            (vec_field.right()[1] - vec_field.left()[1]));
            }
            else if constexpr (C == CurlComponent::X2) {
                return (1.0 / r) *
                       ((1.0 / (dphi * sint)) *
                            (vec_field.right()[0] - vec_field.left()[0]) -
                        (1.0 / dr) * (vec_field.right()[2] * cell.x1R() -
                                      vec_field.left()[2] * cell.x1L()));
            }
            else {
                return (1.0 / r) *
                       ((1.0 / dr) * (vec_field.right()[1] * cell.x1R() -
                                      vec_field.left()[1] * cell.x1L()) -
                        (1.0 / dtheta) *
                            (vec_field.right()[0] - vec_field.left()[0]));
            }
        }

        template <int nhat>
        DUAL real
        curl_component(const auto& cell, VectorField<real, 3>& vec_field)
        {
            switch (cell.geometry()) {
                case Geometry::CARTESIAN:
                    return curl_cartesian_component<
                        static_cast<CurlComponent>(nhat - 1)>(cell, vec_field);
                case Geometry::CYLINDRICAL:
                    return curl_cylindrical_component<
                        static_cast<CurlComponent>(nhat - 1)>(cell, vec_field);
                case Geometry::SPHERICAL:
                    return curl_spherical_component<
                        static_cast<CurlComponent>(nhat - 1)>(cell, vec_field);
                default:
                    return curl_cartesian_component<
                        static_cast<CurlComponent>(nhat - 1)>(cell, vec_field);
            }
        }
    }   // namespace vector_calculus

}   // namespace simbi

#endif