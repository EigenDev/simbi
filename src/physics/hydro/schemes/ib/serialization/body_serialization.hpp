/**
 *=============================================================================
 *           SIMBI - Special Relativistic Magnetohydrodynamics Code
 *=============================================================================
 *
 * @file            body_serialization.hpp
 * @brief           Body serialization for the IB scheme
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

#ifndef BODY_SERIALIZATION_HPP
#define BODY_SERIALIZATION_HPP

#include "core/utility/config_dict.hpp"
#include <H5Cpp.h>
#include <functional>
#include <string>
#include <type_traits>

namespace simbi::ibsystem {

    // Serializable property descriptor
    template <typename T>
    struct PropertyDescriptor {
        // Property name
        std::string name;

        // Function to extract property value from a body system
        std::function<T(size_t)> extractor;

        // Metadata (optional)
        // config_dict_t metadata;
    };

    // Property serialization trait for different property types
    template <typename T>
    struct PropertySerializationTrait {
        static H5::DataType h5_type();
        static void
        write_to_h5(H5::Group& group, const std::string& name, const T& value);
    };

    // Specialization for floating point values
    template <>
    struct PropertySerializationTrait<double> {
        static H5::DataType h5_type() { return H5::PredType::NATIVE_DOUBLE; }

        static void write_to_h5(
            H5::Group& group,
            const std::string& name,
            const double& value
        )
        {
            H5::DataSpace scalar_space(H5S_SCALAR);
            auto dataset = group.createDataSet(name, h5_type(), scalar_space);
            dataset.write(&value, h5_type());
        }
    };

    // Specialization for boolean values
    template <>
    struct PropertySerializationTrait<bool> {
        static H5::DataType h5_type() { return H5::PredType::NATIVE_HBOOL; }

        static void write_to_h5(
            H5::Group& group,
            const std::string& name,
            const bool& value
        )
        {
            H5::DataSpace scalar_space(H5S_SCALAR);
            auto dataset = group.createDataSet(name, h5_type(), scalar_space);
            dataset.write(&value, h5_type());
        }
    };

    // Specialization for string values
    template <>
    struct PropertySerializationTrait<std::string> {
        static H5::DataType h5_type()
        {
            return H5::StrType(H5::PredType::C_S1, H5T_VARIABLE);
        }

        static void write_to_h5(
            H5::Group& group,
            const std::string& name,
            const std::string& value
        )
        {
            H5::DataSpace scalar_space(H5S_SCALAR);
            auto dataset = group.createDataSet(name, h5_type(), scalar_space);
            const char* c_str = value.c_str();
            dataset.write(&c_str, h5_type());
        }
    };

    // Specialization for vector data (e.g., position, velocity)
    template <typename T, size_t N>
    struct PropertySerializationTrait<spatial_vector_t<T, N>> {
        static H5::DataType h5_type()
        {
            return PropertySerializationTrait<T>::h5_type();
        }

        static void write_to_h5(
            H5::Group& group,
            const std::string& name,
            const spatial_vector_t<T, N>& value
        )
        {
            hsize_t dims[1] = {N};
            H5::DataSpace vec_space(1, dims);
            auto dataset = group.createDataSet(name, h5_type(), vec_space);
            dataset.write(value.data(), h5_type());
        }
    };
};   // namespace simbi::ibsystem

#endif
