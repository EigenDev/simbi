/**
 *  *=============================================================================
 *  *           SIMBI - Special Relativistic Magnetohydrodynamics Code
 *  *=============================================================================
 *  *
 *  * @file            body_system.hpp
 *  * @brief
 *  * @details
 *  *
 *  * @version         0.8.0
 *  * @date            2025-03-07
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
 *  * 2025-03-07      v0.8.0      Initial implementation
 *  *
 *  *==============================================================================
 *  * @copyright (C) 2025 Marcus DuPont. All rights reserved.
 *  *==============================================================================
 */

#ifndef BODY_SYSTEM_HPP
#define BODY_SYSTEM_HPP

#include "body_factory.hpp"                    // for BodyFactory
#include "build_options.hpp"                   // for DUAL
#include "core/types/containers/ndarray.hpp"   // for ndarray
#include "core/types/containers/vector.hpp"    // for spatial_vector_t
#include "core/types/utility/smart_ptr.hpp"    // for std::shared_ptr
#include <unordered_map>                       // for std::unordered_map

namespace simbi::ibsystem {
    template <typename T, size_type Dims, typename MeshType>
    class BodySystem
    {
      protected:
        std::vector<std::unique_ptr<ib::ImmersedBody<T, Dims, MeshType>>>
            bodies_;
        MeshType mesh_;

      public:
        BodySystem() = default;
        DUAL BodySystem(const MeshType& mesh) : mesh_(mesh) {}

        DUAL void add_body(
            ib::BodyType type,
            const spatial_vector_t<T, Dims>& position,
            const spatial_vector_t<T, Dims>& velocity,
            T mass,
            T radius,
            const auto& params = {}
        )
        {
            auto body = ib::BodyFactory<T, Dims, MeshType>::create_body(
                type,
                mesh_,
                position,
                velocity,
                mass,
                radius,
                params
            );
            bodies_.push_back(std::move(body));
        }

        // DUAL void update_system(auto& prim_states, const T dt)
        // {
        //     for (auto& body : bodies_) {
        //         body->update_conserved_state(prim_states, dt);
        //     }
        // }
    };
}   // namespace simbi::ibsystem

#endif