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

#include "system_config.hpp"
#include <H5Cpp.h>
#include <string>

namespace simbi::ibsystem {

    // system config property descriptor
    template <typename T>
    struct SystemPropertyDescriptor {
        std::string name;
        T value;
    };

    // Function to create H5 attributes for various property types
    template <typename T>
    inline void write_property_to_h5(
        H5::Group& group,
        const std::string& name,
        const T& value
    );

    // Specialization for double
    template <>
    inline void write_property_to_h5<double>(
        H5::Group& group,
        const std::string& name,
        const double& value
    )
    {
        H5::DataSpace scalar_space(H5S_SCALAR);
        auto attr = group.createAttribute(
            name,
            H5::PredType::NATIVE_DOUBLE,
            scalar_space
        );
        attr.write(H5::PredType::NATIVE_DOUBLE, &value);
    }

    // Specialization for bool
    template <>
    inline void write_property_to_h5<bool>(
        H5::Group& group,
        const std::string& name,
        const bool& value
    )
    {
        H5::DataSpace scalar_space(H5S_SCALAR);
        auto attr = group.createAttribute(
            name,
            H5::PredType::NATIVE_HBOOL,
            scalar_space
        );
        attr.write(H5::PredType::NATIVE_HBOOL, &value);
    }

    // Specialization for std::string
    template <>
    inline void write_property_to_h5<std::string>(
        H5::Group& group,
        const std::string& name,
        const std::string& value
    )
    {
        H5::DataSpace scalar_space(H5S_SCALAR);
        H5::StrType str_type(H5::PredType::C_S1, H5T_VARIABLE);
        auto attr         = group.createAttribute(name, str_type, scalar_space);
        const char* c_str = value.c_str();
        attr.write(str_type, &c_str);
    }

    // Specialization for integer pairs (like body indices)
    template <>
    inline void write_property_to_h5<std::pair<size_t, size_t>>(
        H5::Group& group,
        const std::string& name,
        const std::pair<size_t, size_t>& value
    )
    {
        hsize_t dims[1] = {2};
        H5::DataSpace pair_space(1, dims);
        auto dataset =
            group.createDataSet(name, H5::PredType::NATIVE_UINT64, pair_space);
        size_t indices[2] = {value.first, value.second};
        dataset.write(indices, H5::PredType::NATIVE_UINT64);
    }

    // Method to serialize any SystemConfig to an HDF5 group
    inline void serialize_system_config(
        const SystemConfig* config,
        H5::Group& parent_group,
        const std::string& group_name = "system_config"
    )
    {
        if (!config) {
            return;
        }

        // Create a group for system configuration
        H5::Group config_group = parent_group.createGroup(group_name);

        // Let the config write its specific properties
        config->write_to_h5(config_group);

        // Close the config group
        config_group.close();
    }

}   // namespace simbi::ibsystem

#endif   // SYSTEM_CONFIG_SERIALIZATION_HPP
