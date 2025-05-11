/**
 *=============================================================================
 *           SIMBI - Special Relativistic Magnetohydrodynamics Code
 *=============================================================================
 *
 * @file            system_config.hpp
 * @brief           System configuration for the IB scheme
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

#include "build_options.hpp"
#include "core/types/utility/managed.hpp"
#include <utility>

namespace H5 {
    class Group;
}

namespace simbi::ibsystem {
    struct SystemConfig : public Managed<global::managed_memory> {
        virtual ~SystemConfig() = default;

        // Serialization method that each derived class must implement
        virtual void write_to_h5(H5::Group& group) const
        {
            // Base class implementation - just writes the type
            H5::DataSpace scalar_space(H5S_SCALAR);
            H5::StrType str_type(H5::PredType::C_S1, H5T_VARIABLE);
            auto type_attr =
                group.createAttribute("config_type", str_type, scalar_space);
            const char* type_str = "base";
            type_attr.write(str_type, &type_str);
        }
    };

    template <typename T>
    struct BinarySystemConfig : public SystemConfig {
        T semi_major;
        T mass_ratio;
        T eccentricity;
        T orbital_period;
        bool circular_orbit;
        bool prescribed_motion;
        std::pair<size_t, size_t> body_indices;

        BinarySystemConfig(
            T semi_major,
            T mass_ratio,
            T eccentricity,
            T orbital_period,
            bool circular_orbit,
            bool prescribed_motion,
            size_t body1_idx,
            size_t body2_idx
        )
            : semi_major(semi_major),
              mass_ratio(mass_ratio),
              eccentricity(eccentricity),
              orbital_period(orbital_period),
              circular_orbit(circular_orbit),
              prescribed_motion(prescribed_motion),
              body_indices(body1_idx, body2_idx)
        {
        }

        // Implementation of serialization method for binary systems
        void write_to_h5(H5::Group& group) const override;
    };

    // Implementation of the serialization method for binary systems
    template <typename T>
    void BinarySystemConfig<T>::write_to_h5(H5::Group& group) const
    {
        // Write config type
        H5::DataSpace scalar_space(H5S_SCALAR);
        H5::StrType str_type(H5::PredType::C_S1, H5T_VARIABLE);
        auto type_attr =
            group.createAttribute("config_type", str_type, scalar_space);
        const char* type_str = "binary";
        type_attr.write(str_type, &type_str);

        // Write all binary system properties as attributes
        auto write_scalar = [&group](const std::string& name, auto value) {
            H5::DataSpace scalar_space(H5S_SCALAR);
            using ValueType = decltype(value);

            if constexpr (std::is_same_v<ValueType, bool>) {
                auto attr = group.createAttribute(
                    name,
                    H5::PredType::NATIVE_HBOOL,
                    scalar_space
                );
                attr.write(H5::PredType::NATIVE_HBOOL, &value);
            }
            else if constexpr (std::is_floating_point_v<ValueType>) {
                auto attr = group.createAttribute(
                    name,
                    H5::PredType::NATIVE_DOUBLE,
                    scalar_space
                );
                attr.write(H5::PredType::NATIVE_DOUBLE, &value);
            }
        };

        // Write scalar properties
        write_scalar("semi_major", this->semi_major);
        write_scalar("mass_ratio", this->mass_ratio);
        write_scalar("eccentricity", this->eccentricity);
        write_scalar("orbital_period", this->orbital_period);
        write_scalar("circular_orbit", this->circular_orbit);
        write_scalar("prescribed_motion", this->prescribed_motion);

        // Write body indices as a dataset
        hsize_t dims[1] = {2};
        H5::DataSpace pair_space(1, dims);
        auto dataset = group.createDataSet(
            "body_indices",
            H5::PredType::NATIVE_UINT64,
            pair_space
        );
        size_t indices[2] = {
          this->body_indices.first,
          this->body_indices.second
        };
        dataset.write(indices, H5::PredType::NATIVE_UINT64);
    }
}   // namespace simbi::ibsystem

#endif   // SYSTEM_CONFIG_HPP
