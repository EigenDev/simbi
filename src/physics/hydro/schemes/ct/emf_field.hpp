#ifndef EMF_FIELD_HPP
#define EMF_FIELD_HPP

#include "build_options.hpp"
#include "core/types/enums.hpp"
#include "core/types/vector_field.hpp"
#include "geometry/mesh.hpp"
#include "physics/hydro/schemes/ct/ct_calculator.hpp"

namespace simbi {
    namespace ct {
        template <typename CTScheme>
        class EMField : public vector_field::VectorField<real, 3>
        {
          public:
            EMField(
                const general_vector_t<real, 3>& left,
                const general_vector_t<real, 3>& right
            )
                : vector_field::VectorField<real, 3>(left, right)
            {
            }

            EMField()
                : vector_field::VectorField<real, 3>(
                      general_vector_t<real, 3>{0.0, 0.0, 0.0},
                      general_vector_t<real, 3>{0.0, 0.0, 0.0}
                  )
            {
            }

            ~EMField() = default;

            template <int l_dir, int m_dir>
            void compute_edge_components(
                const auto& fri,
                const auto& gri,
                const auto& hri,
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
                        Corner::SW>(hri, gri, prims, l_dir);

                    // E_{2, i-1/2, j ,k-1/2}
                    left_field[m_dir - 1] = ct_algo::template calc_edge_emf<
                        BlockAx::K,
                        Plane::IK,
                        Corner::SW>(hri, fri, prims, m_dir);

                    // E_{1, i, j+1/2,k-1/2}
                    right_field[l_dir - 1] = ct_algo::template calc_edge_emf<
                        BlockAx::K,
                        Plane::JK,
                        Corner::SE>(hri, gri, prims, l_dir);

                    // E_{2, i+1/2, j ,k-1/2}
                    right_field[m_dir - 1] = ct_algo::template calc_edge_emf<
                        BlockAx::K,
                        Plane::IK,
                        Corner::SE>(hri, fri, prims, m_dir);
                }
                else if constexpr (l_dir == 2 && m_dir == 3) {   // updating B1
                    // E_{2, i-1/2, j,k-1/2}
                    left_field[l_dir - 1] = ct_algo::template calc_edge_emf<
                        BlockAx::I,
                        Plane::IK,
                        Corner::SW>(hri, fri, prims, l_dir);

                    // E_{3, i-1/2, j-1/2, k}
                    left_field[m_dir - 1] = ct_algo::template calc_edge_emf<
                        BlockAx::I,
                        Plane::IJ,
                        Corner::SW>(gri, fri, prims, m_dir);

                    // E_{2, i-1/2, j,k+1/2}
                    right_field[l_dir - 1] = ct_algo::template calc_edge_emf<
                        BlockAx::I,
                        Plane::IK,
                        Corner::NW>(hri, fri, prims, l_dir);

                    // E_{3, i-1/2, j+1/2, k}
                    right_field[m_dir - 1] = ct_algo::template calc_edge_emf<
                        BlockAx::I,
                        Plane::IJ,
                        Corner::NW>(gri, fri, prims, m_dir);
                }
                else {   // updating B2
                    // E_{3, i-1/2, j-1/2, k}
                    left_field[l_dir - 1] = ct_algo::template calc_edge_emf<
                        BlockAx::J,
                        Plane::IJ,
                        Corner::SW>(gri, fri, prims, l_dir);

                    // E_{1, i, j-1/2, k-1/2}
                    left_field[m_dir - 1] = ct_algo::template calc_edge_emf<
                        BlockAx::J,
                        Plane::JK,
                        Corner::SW>(hri, gri, prims, m_dir);

                    // E_{3, i+1/2, j-1/2, k}
                    right_field[l_dir - 1] = ct_algo::template calc_edge_emf<
                        BlockAx::J,
                        Plane::IJ,
                        Corner::SE>(gri, fri, prims, l_dir);

                    // E_{1, i, j-1/2, k+1/2}
                    right_field[m_dir - 1] = ct_algo::template calc_edge_emf<
                        BlockAx::J,
                        Plane::JK,
                        Corner::NW>(hri, gri, prims, m_dir);
                }
            }
        };
    }   // namespace ct

}   // namespace simbi
#endif
