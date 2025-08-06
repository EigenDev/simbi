/**
 *=============================================================================
 *           SIMBI - Special Relativistic Magnetohydrodynamics Code
 *=============================================================================
 *
 * @file            system_serialization.hpp
 * @brief           System serialization for the IB scheme
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
#ifndef SYSTEM_CONFIG_SERIALIZATION_HPP
#define SYSTEM_CONFIG_SERIALIZATION_HPP

#include <H5Cpp.h>
#include <string>

namespace simbi::ibsystem {

    // system config property descriptor
    template <typename T>
    struct SystemPropertyDescriptor {
        std::string name;
        T value;
    };

    // function to create H5 attributes for various property types
    template <typename T>
    void write_property_to_h5(
        H5::Group& group,
        const std::string& name,
        const T& value
    );

}   // namespace simbi::ibsystem

#endif   // SYSTEM_CONFIG_SERIALIZATION_HPP
