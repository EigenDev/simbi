/**
 *=============================================================================
 *           SIMBI - Special Relativistic Magnetohydrodynamics Code
 *=============================================================================
 *
 * @file            system_config.hpp
 * @brief
 * @details
 *
 * @version         0.8.0
 * @date            2025-05-11
 * @author          Marcus DuPont
 * @email           marcus.dupont@princeton.edu
 *
 *==============================================================================
 * @build           Requirements & Dependencies
 *==============================================================================
 * @requires        C++20
 * @depends         CUDA >= 11.0, HDF5 >= 1.12, OpenMP >= 4.5
 * @platform        Linux, MacOS
 * @parallel        GPU (CUDA, HIP), CPU (OpenMP)
 *
 *==============================================================================
 * @documentation   Reference & Notes
 *==============================================================================
 * @usage
 * @note
 * @warning
 * @todo
 * @bug
 * @performance
 *
 *==============================================================================
 * @testing        Quality Assurance
 *==============================================================================
 * @test
 * @benchmark
 * @validation
 *
 *==============================================================================
 * @history        Version History
 *==============================================================================
 * 2025-05-11      v0.8.0      Initial implementation
 *
 *==============================================================================
 * @copyright (C) 2025 Marcus DuPont. All rights reserved.
 *==============================================================================
 */

#ifndef SYSTEM_CONFIG_HPP
#define SYSTEM_CONFIG_HPP

#include "core/types/containers/ndarray.hpp"
#include "core/types/utility/enums.hpp"
#include <string>
#include <utility>

// houses pure data structures for a system
namespace simbi::ibsystem::config {
    // basic gravitational config
    template <typename T>
    struct GravitationalConfig {
        bool prescribed_motion      = true;
        std::string reference_frame = "center_of_mass";
    };

    template <typename T>
    struct GravitationalComponent {
        T mass;
        T radius;
        T softening_length;
        T accretion_efficiency = 0.01;
        T accretion_radius     = 0.01;
        bool two_way_coupling  = false;
        bool is_an_accretor    = false;
        BodyType body_type     = BodyType::GRAVITATIONAL;

        void configure()
        {
            if (is_an_accretor) {
                body_type = BodyType::GRAVITATIONAL_SINK;
            }
        }
    };

    template <typename T>
    using binary_pair_t =
        std::pair<GravitationalComponent<T>, GravitationalComponent<T>>;

    template <typename T>
    struct BinaryConfig {
        binary_pair_t<T> binary_pair;
        T semi_major                  = 1.0;
        T eccentricity                = 0.0;
        T mass_ratio                  = 1.0;    // q = m2/m1
        bool circular                 = true;   // shorthand for e = 0
        bool equal_mass               = true;   // shorthand for q = 1
        T total_mass                  = 1.0;
        T inclination                 = 0.0;
        T longitude_of_ascending_node = 0.0;
        T argument_of_periapsis       = 0.0;
        T true_anomaly                = 0.0;
    };

    template <typename T>
    struct PlanetaryConfig {
        T central_mass = T(1.0);
        ndarray<T> planet_masses;
        ndarray<T> planet_semi_majors;
        ndarray<T> planet_eccentricities;
    };
}   // namespace simbi::ibsystem::config
#endif   // SYSTEM_CONFIG_HPP
