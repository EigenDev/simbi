/**
 *  *=============================================================================
 *  *           SIMBI - Special Relativistic Magnetohydrodynamics Code
 *  *=============================================================================
 *  *
 *  * @file            source.hpp
 *  * @brief           Source Immersed Body Implementation
 *  * @details
 *  *
 *  * @version         0.8.0
 *  * @date            2025-02-16
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
 *  * 2025-02-16      v0.8.0      Initial implementation
 *  *
 *  *==============================================================================
 *  * @copyright (C) 2025 Marcus DuPont. All rights reserved.
 *  *==============================================================================
 */

#ifndef SOURCE_HPP
#define SOURCE_HPP

#include "build_options.hpp"
#include "immersed_boundary.hpp"

namespace simbi {
    namespace ib {
        // source (matter injection) particle functionality
        template <typename T, size_type Dims, typename MeshType>
        class SourceParticle : public ImmersedBody<T, Dims, MeshType>
        {
          protected:
            T injection_rate_;       // mass injection rate
            T injection_energy_;     // energy injection rate
            T injection_momentum_;   // momentum injection rate

          public:
            DUAL SourceParticle(
                const MeshType& mesh,
                const spatial_vector_t<T, Dims>& position,
                const spatial_vector_t<T, Dims>& velocity,
                const T mass,
                const T radius,
                const T injection_rate,
                const T injection_energy,
                const T injection_momentum
            )
                : ImmersedBody<T, Dims, MeshType>(
                      mesh,
                      position,
                      velocity,
                      mass,
                      radius
                  ),
                  injection_rate_(injection_rate),
                  injection_energy_(injection_energy),
                  injection_momentum_(injection_momentum)
            {
            }

            // Source particle methods
            DUAL void inject(auto& cons_state)
            {
                for (const auto& idx : cut_cell_indices()) {
                    auto& cell = cons_state[idx];
                    cell.mass() += injection_rate_ * cell.volume_fraction;
                    cell.momentum() +=
                        injection_momentum_ * cell.volume_fraction;
                    cell.nrg() += injection_energy_ * cell.volume_fraction;
                }
            }
        }
    }   // namespace ib

}   // namespace simbi

#endif