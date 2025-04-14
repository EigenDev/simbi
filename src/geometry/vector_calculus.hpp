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

#include "build_options.hpp"                        // for real, DEV
#include "core/types/containers/vector_field.hpp"   // for Vector, VectorField
#include <iostream>                                 // for cout, cin
#include <vector>                                   // for vector
using namespace simbi::vector_field;

namespace simbi {
    namespace vector_calculus {

        enum class CurlComponent {
            X1 = 0,
            X2 = 1,
            X3 = 2
        };

        DEV general_vector_t<real, 3>
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

        DEV general_vector_t<real, 3> curl_cylindrical(
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

        DEV general_vector_t<real, 3>
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

        DEV general_vector_t<real, 3>
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
        DEV real curl_cartesian_component(
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
        DEV real curl_cylindrical_component(
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
        DEV real curl_spherical_component(
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
                if (cell.at_pole()) {
                    // At exact pole, use limiting formula
                    const auto at_north_pole = cell.at_north_pole();
                    const real pole_sign     = at_north_pole ? 1.0 : -1.0;

                    // Use one-sided difference for B_phi derivative
                    //
                    // Should be 0
                    const real B_phi_pole = vec_field.left()[2];
                    const real B_phi_next = at_north_pole ? vec_field.right()[2]
                                                          : vec_field.left()[2];

                    // Calculate proper derivative at pole
                    return pole_sign * 2.0 / r * (B_phi_next - B_phi_pole) /
                           dtheta;
                }
                return (1.0 / (cell.x1L() * sint)) *
                       ((1.0 / dtheta) *
                            (vec_field.right()[2] * std::sin(cell.x2R()) -
                             vec_field.left()[2] * std::sin(cell.x2L())) -
                        (1.0 / dphi) *
                            (vec_field.right()[1] - vec_field.left()[1]));
            }
            else if constexpr (C == CurlComponent::X2) {
                if (cell.at_pole()) {
                    return 0.0;
                }

                return (1.0 / r) *
                       ((1.0 / (dphi * std::sin(cell.x2L()))) *
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
                // if (res != 0) {
                //     const auto theta_part =
                //         (1.0 / dr) * (vec_field.right()[1] * cell.x1R() -
                //                       vec_field.left()[1] * cell.x1L());
                //     const auto r_part = (1.0 / dtheta) *
                //     (vec_field.right()[0] -
                //                                           vec_field.left()[0]);

                // printf(
                //     "[E_theta] R: %f, L: %f\n",
                //     vec_field.right()[1],
                //     vec_field.left()[1]
                // );
                // printf("X1R: %f, X1L: %f\n", cell.x1R(), cell.x1L());
                // printf(
                //     "[E_r] R: %f, L: %f\n",
                //     vec_field.right()[0],
                //     vec_field.left()[0]
                // );
                // printf("X2R: %f, X2L: %f\n", cell.x2R(), cell.x2L());
                // printf("r_part: %f, theta_part: %f\n", r_part,
                // theta_part); std::cin.get();
                // }

                // return res;
            }
        }

        template <int nhat>
        DEV real
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

        template <typename... Args>
        DEV real
        divergence_cartesian(const auto& cell, const Args... components)
        {
            auto to_real = [](auto val) { return static_cast<real>(val); };
            real res     = 0.0;
            std::vector<real> vals = {to_real(components)...};
            // first two components L/R for the x1 direction
            res += (1.0 / cell.width(0)) * (vals[1] - vals[0]);
            // next two components L/R for the x2 direction
            res += (1.0 / cell.width(1)) * (vals[3] - vals[2]);
            // last two components L/R for the x3 direction
            res += (1.0 / cell.width(2)) * (vals[5] - vals[4]);
            return res;
        }

        template <typename... Args>
        DEV real
        divergence_spherical(const auto& cell, const Args... components)
        {
            auto to_real = [](auto val) { return static_cast<real>(val); };
            real res     = 0.0;
            std::vector<real> vals = {to_real(components)...};
            const auto r           = cell.centroid_coordinate(0);
            const auto sint        = std::sin(cell.centroid_coordinate(1));
            const auto dr          = cell.x1R() - cell.x1L();
            const auto dtheta      = cell.x2R() - cell.x2L();
            const auto dphi        = cell.x3R() - cell.x3L();
            // first two components L/R for the r direction
            res += (1.0 / (r * r)) * (1.0 / dr) *
                   (vals[1] * cell.x1R() * cell.x1R() -
                    vals[0] * cell.x1L() * cell.x1L());
            // next two components L/R for the theta direction
            res += (1.0 / (r * sint)) * (1.0 / dtheta) *
                   (vals[3] * std::sin(cell.x2R()) -
                    vals[2] * std::sin(cell.x2L()));
            // last two components L/R for the phi direction
            res += (1.0 / (r * sint)) * (1.0 / dphi) * (vals[5] - vals[4]);
            return res;
        }

        template <typename... Args>
        DEV real
        divergence_cylindrical(const auto& cell, const Args... components)
        {
            auto to_real = [](auto val) { return static_cast<real>(val); };
            real res     = 0.0;
            std::vector<real> vals = {to_real(components)...};
            const auto x1mean      = cell.centroid_coordinate(0);
            const auto dr          = cell.x1R() - cell.x1L();
            const auto dphi        = cell.x2R() - cell.x2L();
            const auto dz          = cell.x3R() - cell.x3L();
            // first two components L/R for the r direction
            res += (1.0 / x1mean) * (1.0 / dr) *
                   (vals[1] * cell.x1R() - vals[0] * cell.x1L());
            // next two components L/R for the phi direction
            res += (1.0 / x1mean) * (1.0 / dphi) * (vals[3] - vals[2]);
            // last two components L/R for the z direction
            res += (1.0 / dz) * (vals[5] - vals[4]);
            return res;
        }

        template <typename... Args>
        DEV real divergence(const auto& cell, const Args... components)
        {
            switch (cell.geometry()) {
                case Geometry::CARTESIAN:
                    return divergence_cartesian(cell, components...);
                case Geometry::CYLINDRICAL:
                    return divergence_cylindrical(cell, components...);
                default: return divergence_spherical(cell, components...);
            }
        }
    }   // namespace vector_calculus

}   // namespace simbi

#endif
