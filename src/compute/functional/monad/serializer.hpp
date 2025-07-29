#ifndef SIMBI_SERIALIZATION_HPP
#define SIMBI_SERIALIZATION_HPP

#include "compute/field.hpp"   // for field_t<T, Dims>
#include "config.hpp"          // for real, DEV, etc
#include "core/utility/enums.hpp"
#include "core/utility/helpers.hpp"
#include "mesh/mesh_config.hpp"   // for mesh::mesh_config_t
#include "result.hpp"             // for result_t<T> monad
#include <H5Cpp.h>                // for HDF5 C++ API
#include <algorithm>
#include <concepts>        // for concepts
#include <cstdint>         // for std::uint64_t
#include <functional>      // for std::function
#include <iostream>        // for std::cout, std::cerr
#include <string>          // for std::string
#include <type_traits>     // for std::is_arithmetic_v, std::same_as
#include <unordered_map>   // for std::unordered_map
#include <utility>         // for std::move
#include <vector>          // for std::vector
namespace simbi::io {
    using namespace simbi::helpers;
    // serialization context - accumulates state through pipeline
    struct serialization_context_t {
        H5::H5File file;
        std::string filename;
        std::vector<std::string> written_datasets;
        std::unordered_map<std::string, hsize_t> dimensions;

        // builder-style methods for chaining
        serialization_context_t with_dataset(const std::string& name) const
        {
            auto ctx = *this;
            ctx.written_datasets.push_back(name);
            return ctx;
        }

        serialization_context_t
        with_dimension(const std::string& name, hsize_t size) const
        {
            auto ctx             = *this;
            ctx.dimensions[name] = size;
            return ctx;
        }
    };

    // concepts for what can be serialized
    template <typename T>
    concept field_serializable_c = requires {
        typename T::value_type;
        T::dimensions;
        requires std::
            same_as<T, field_t<typename T::value_type, T::dimensions>>;
    };

    template <typename T>
    concept hydro_state_serializable_c = requires {
        T::dimensions;
        T::regime_t;
        T::is_mhd;
        typename T::primitive_t;
        typename T::conserved_t;
    };

    // core serialization traits - specialize for different types
    template <typename T>
    struct serialization_trait_t {
        // default: not serializable
        static constexpr bool serializable = false;
    };

    // trait specialization for scalar field_t (real, int, etc.)
    template <typename T, std::uint64_t Dims>
        requires std::is_arithmetic_v<T>
    struct serialization_trait_t<field_t<T, Dims>> {
        static constexpr bool serializable = true;

        static result_t<serialization_context_t> serialize(
            const field_t<T, Dims>& field,
            const std::string& dataset_name,
            serialization_context_t ctx
        )
        {
            // ensure data is on cpu for serialization
            // field.memory()->ensure_cpu_synced();

            // create dataspace from field domain
            auto shape = field.domain().shape();
            std::vector<hsize_t> dims(Dims);
            for (std::uint64_t ii = 0; ii < Dims; ++ii) {
                dims[ii] = shape[ii];
            }

            H5::DataSpace dataspace(Dims, dims.data());

            // determine hdf5 data type
            H5::DataType h5_type;
            if constexpr (std::same_as<T, double>) {
                h5_type = H5::PredType::NATIVE_DOUBLE;
            }
            else if constexpr (std::same_as<T, int>) {
                h5_type = H5::PredType::NATIVE_INT;
            }
            else if constexpr (std::same_as<T, float>) {
                h5_type = H5::PredType::NATIVE_FLOAT;
            }

            try {
                auto dataset =
                    ctx.file.createDataSet(dataset_name, h5_type, dataspace);
                dataset.write(field.data(), h5_type);
                dataset.close();

                return result_t<serialization_context_t>::ok(
                    ctx.with_dataset(dataset_name)
                );
            }
            catch (const H5::Exception& e) {
                return result_t<serialization_context_t>::error(
                    "hdf5 error writing " + dataset_name + ": " +
                    e.getDetailMsg()
                );
            }
        }
    };

    // trait specialization for hydro state field_t (AoS layout)
    template <typename StateType, std::uint64_t Dims>
        requires requires {
            StateType::nmem;
            StateType::dimensions;
            typename StateType::eos_t;
        }
    struct serialization_trait_t<field_t<StateType, Dims>> {
        static constexpr bool serializable = true;

        static result_t<serialization_context_t> serialize_component(
            const field_t<StateType, Dims>& field,
            std::uint64_t component_idx,
            const std::string& component_name,
            serialization_context_t ctx
        )
        {
            // ensure data is on cpu for serialization
            // field.memory()->ensure_cpu_synced();

            // create temporary array for this component
            auto total_size = field.domain().size();
            std::vector<real> component_data(total_size);

            // extract component from aos layout
            for (std::uint64_t ii = 0; ii < total_size; ++ii) {
                component_data[ii] = field[ii][component_idx];
            }

            // create dataspace from field domain
            const auto shape = field.domain().shape();
            std::vector<hsize_t> dims(Dims);
            for (std::uint64_t ii = 0; ii < Dims; ++ii) {
                dims[ii] = shape[ii];
            }

            H5::DataSpace dataspace(Dims, dims.data());
            H5::DataType h5_type = H5::PredType::NATIVE_DOUBLE;

            try {
                auto dataset =
                    ctx.file.createDataSet(component_name, h5_type, dataspace);
                dataset.write(component_data.data(), h5_type);
                dataset.close();

                return result_t<serialization_context_t>::ok(
                    ctx.with_dataset(component_name)
                );
            }
            catch (const H5::Exception& e) {
                return result_t<serialization_context_t>::error(
                    "hdf5 error writing " + component_name + ": " +
                    e.getDetailMsg()
                );
            }
        }

        static result_t<serialization_context_t> serialize(
            const field_t<StateType, Dims>& field,
            const std::string& /*base_name*/,
            serialization_context_t ctx
        )
        {
            // component names based on state type
            std::vector<std::string> component_names;

            // common components: rho, vel, pre
            component_names.push_back("rho");
            for (std::uint64_t d = 0; d < StateType::dimensions; ++d) {
                component_names.push_back("v" + std::to_string(d + 1));
            }
            component_names.push_back("p");

            // mhd components: mag
            if constexpr (StateType::nmem == 2 * StateType::dimensions + 3) {
                for (std::uint64_t d = 0; d < StateType::dimensions; ++d) {
                    component_names.push_back(
                        "b" + std::to_string(d + 1) + "_mean"
                    );
                }
            }

            // final component: chi
            component_names.push_back("chi");

            // serialize each component in sequence using and_then
            auto result = result_t<serialization_context_t>::ok(ctx);
            for (std::uint64_t ii = 0; ii < component_names.size(); ++ii) {
                result = result.and_then([&, ii](auto current_ctx) {
                    return serialize_component(
                        field,
                        ii,
                        component_names[ii],
                        current_ctx
                    );
                });
            }

            return result;
        }
    };

    // metadata serialization trait - handles structured metadata
    template <typename MetaData>
    struct metadata_serialization_trait_t {
        static result_t<serialization_context_t> serialize_attributes(
            const MetaData& metadata,
            const std::string& group_name,
            serialization_context_t ctx
        )
        {
            try {
                // create empty dataspace for attributes
                H5::DataSpace attr_dataspace(H5S_NULL);
                H5::DataSpace scalar_dataspace(H5S_SCALAR);

                // data types
                H5::DataType real_type   = H5::PredType::NATIVE_DOUBLE;
                H5::DataType int_type    = H5::PredType::NATIVE_INT;
                H5::DataType uint64_type = H5::PredType::NATIVE_UINT64;
                H5::DataType bool_type   = H5::PredType::NATIVE_HBOOL;
                H5::StrType str_type(H5::PredType::C_S1, 256);

                // create group/dataset for metadata
                H5::DataSet sim_info = ctx.file.createDataSet(
                    group_name,
                    int_type,
                    attr_dataspace
                );

                // helper lambda for writing attributes
                auto write_attr =
                    [&](const std::string& name,
                        const auto& value,
                        const H5::DataType& type) -> result_t<void> {
                    try {
                        auto attr = sim_info.createAttribute(
                            name,
                            type,
                            scalar_dataspace
                        );
                        attr.write(type, &value);
                        attr.close();
                        return result_t<void>::ok();
                    }
                    catch (const H5::Exception& e) {
                        return result_t<void>::error(
                            "failed to write attribute " + name + ": " +
                            e.getDetailMsg()
                        );
                    }
                };

                // helper for string attributes
                auto write_str_attr =
                    [&](const std::string& name,
                        const std::string& value) -> result_t<void> {
                    try {
                        auto attr = sim_info.createAttribute(
                            name,
                            str_type,
                            scalar_dataspace
                        );
                        attr.write(str_type, value.c_str());
                        attr.close();
                        return result_t<void>::ok();
                    }
                    catch (const H5::Exception& e) {
                        return result_t<void>::error(
                            "failed to write string attribute " + name + ": " +
                            e.getDetailMsg()
                        );
                    }
                };

                // serialize metadata fields
                if constexpr (requires { metadata.gamma; }) {
                    auto result =
                        write_attr("gamma", metadata.gamma, real_type);
                    if (!result.is_ok()) {
                        return result_t<serialization_context_t>::error(
                            result.error()
                        );
                    }
                }

                if constexpr (requires { metadata.plm_theta; }) {
                    auto result =
                        write_attr("plm_theta", metadata.plm_theta, real_type);
                    if (!result.is_ok()) {
                        return result_t<serialization_context_t>::error(
                            result.error()
                        );
                    }
                }

                if constexpr (requires { metadata.cfl; }) {
                    auto result = write_attr("cfl", metadata.cfl, real_type);
                    if (!result.is_ok()) {
                        return result_t<serialization_context_t>::error(
                            result.error()
                        );
                    }
                }

                if constexpr (requires { metadata.time; }) {
                    auto result = write_attr("time", metadata.time, real_type);
                    if (!result.is_ok()) {
                        return result_t<serialization_context_t>::error(
                            result.error()
                        );
                    }
                }

                if constexpr (requires { metadata.tend; }) {
                    auto result = write_attr("tend", metadata.tend, real_type);
                    if (!result.is_ok()) {
                        return result_t<serialization_context_t>::error(
                            result.error()
                        );
                    }
                }

                if constexpr (requires { metadata.dt; }) {
                    auto result = write_attr("dt", metadata.dt, real_type);
                    if (!result.is_ok()) {
                        return result_t<serialization_context_t>::error(
                            result.error()
                        );
                    }
                }

                if constexpr (requires { metadata.iteration; }) {
                    auto result = write_attr(
                        "iteration",
                        metadata.iteration,
                        uint64_type
                    );
                    if (!result.is_ok()) {
                        return result_t<serialization_context_t>::error(
                            result.error()
                        );
                    }
                }

                if constexpr (requires { metadata.halo_radius; }) {
                    auto result = write_attr(
                        "halo_radius",
                        metadata.halo_radius,
                        uint64_type
                    );
                    if (!result.is_ok()) {
                        return result_t<serialization_context_t>::error(
                            result.error()
                        );
                    }
                }

                if constexpr (requires { metadata.is_mhd; }) {
                    auto result =
                        write_attr("is_mhd", metadata.is_mhd, bool_type);
                    if (!result.is_ok()) {
                        return result_t<serialization_context_t>::error(
                            result.error()
                        );
                    }
                }

                if constexpr (requires { metadata.is_relativistic; }) {
                    auto result = write_attr(
                        "is_relativistic",
                        metadata.is_relativistic,
                        bool_type
                    );
                    if (!result.is_ok()) {
                        return result_t<serialization_context_t>::error(
                            result.error()
                        );
                    }
                }

                // enum serialization
                if constexpr (requires {
                                  metadata.regime;
                                  serialize(metadata.regime);
                              }) {
                    auto regime_str = std::string(serialize(metadata.regime));
                    auto result     = write_str_attr("regime", regime_str);
                    if (!result.is_ok()) {
                        return result_t<serialization_context_t>::error(
                            result.error()
                        );
                    }
                }

                if constexpr (requires {
                                  metadata.solver;
                                  serialize(metadata.solver);
                              }) {
                    auto solver_str = std::string(serialize(metadata.solver));
                    auto result     = write_str_attr("solver", solver_str);
                    if (!result.is_ok()) {
                        return result_t<serialization_context_t>::error(
                            result.error()
                        );
                    }
                }

                if constexpr (requires {
                                  metadata.coord_system;
                                  serialize(metadata.coord_system);
                              }) {
                    auto coord_str =
                        std::string(serialize(metadata.coord_system));
                    auto result = write_str_attr("coord_system", coord_str);
                    if (!result.is_ok()) {
                        return result_t<serialization_context_t>::error(
                            result.error()
                        );
                    }
                }

                if constexpr (requires {
                                  metadata.reconstruction;
                                  serialize(metadata.reconstruction);
                              }) {
                    auto recon_str =
                        std::string(serialize(metadata.reconstruction));
                    auto result = write_str_attr("reconstruction", recon_str);
                    if (!result.is_ok()) {
                        return result_t<serialization_context_t>::error(
                            result.error()
                        );
                    }
                }

                if constexpr (requires {
                                  metadata.timestepping;
                                  serialize(metadata.timestepping);
                              }) {
                    auto timestep_str =
                        std::string(serialize(metadata.timestepping));
                    auto result = write_str_attr("timestepping", timestep_str);
                    if (!result.is_ok()) {
                        return result_t<serialization_context_t>::error(
                            result.error()
                        );
                    }
                }

                if constexpr (requires {
                                  metadata.shock_smoother;
                                  serialize(metadata.shock_smoother);
                              }) {
                    auto shock_smoother_str =
                        std::string(serialize(metadata.shock_smoother));
                    auto result =
                        write_str_attr("shock_smoother", shock_smoother_str);
                    if (!result.is_ok()) {
                        return result_t<serialization_context_t>::error(
                            result.error()
                        );
                    }
                }

                if constexpr (requires {
                                  metadata.x1_spacing;
                                  serialize(metadata.x1_spacing);
                              }) {
                    auto x1_spacing_str =
                        std::string(serialize(metadata.x1_spacing));
                    auto result = write_str_attr("x1_spacing", x1_spacing_str);
                    if (!result.is_ok()) {
                        return result_t<serialization_context_t>::error(
                            result.error()
                        );
                    }
                }

                if constexpr (requires {
                                  metadata.x2_spacing;
                                  serialize(metadata.x2_spacing);
                              }) {
                    auto x2_spacing_str =
                        std::string(serialize(metadata.x2_spacing));
                    auto result = write_str_attr("x2_spacing", x2_spacing_str);
                    if (!result.is_ok()) {
                        return result_t<serialization_context_t>::error(
                            result.error()
                        );
                    }
                }

                if constexpr (requires {
                                  metadata.x3_spacing;
                                  serialize(metadata.x3_spacing);
                              }) {
                    auto x3_spacing_str =
                        std::string(serialize(metadata.x3_spacing));
                    auto result = write_str_attr("x3_spacing", x3_spacing_str);
                    if (!result.is_ok()) {
                        return result_t<serialization_context_t>::error(
                            result.error()
                        );
                    }
                }

                // boundary conditions - serialize as comma-separated string
                if constexpr (requires {
                                  metadata.boundary_conditions;
                                  metadata.boundary_conditions.size();
                              }) {
                    std::string bc_str = "";
                    for (std::uint64_t i = 0;
                         i < metadata.boundary_conditions.size();
                         ++i) {
                        if (i > 0) {
                            bc_str += ",";
                        }
                        bc_str += serialize(metadata.boundary_conditions[i]);
                    }
                    auto result = write_str_attr("boundary_conditions", bc_str);
                    if (!result.is_ok()) {
                        return result_t<serialization_context_t>::error(
                            result.error()
                        );
                    }
                }
                if constexpr (requires {
                                  metadata.resolution;
                                  metadata.resolution.size();
                              }) {
                    std::string res_str = "";
                    for (std::uint64_t i = 0; i < metadata.resolution.size();
                         ++i) {
                        if (i > 0) {
                            res_str += ",";
                        }
                        res_str += std::to_string(metadata.resolution[i]);
                    }
                    auto result = write_str_attr("resolution", res_str);
                    if (!result.is_ok()) {
                        return result_t<serialization_context_t>::error(
                            result.error()
                        );
                    }
                }

                if constexpr (requires { metadata.dimensions; }) {
                    auto result =
                        write_attr("dimensions", metadata.dimensions, int_type);
                    if (!result.is_ok()) {
                        return result_t<serialization_context_t>::error(
                            result.error()
                        );
                    }
                }

                sim_info.close();

                return result_t<serialization_context_t>::ok(
                    ctx.with_dataset(group_name)
                );
            }
            catch (const H5::Exception& e) {
                return result_t<serialization_context_t>::error(
                    "hdf5 error writing metadata: " + e.getDetailMsg()
                );
            }
        }
    };

    // specialization for mesh_config_t
    template <std::uint64_t Dims, Geometry G>
    struct metadata_serialization_trait_t<mesh::mesh_config_t<Dims, G>> {
        static result_t<serialization_context_t> serialize_attributes(
            const mesh::mesh_config_t<Dims, G>& mesh_config,
            const std::string& group_name,
            serialization_context_t ctx
        )
        {
            try {
                // create dataspace and data types
                H5::DataSpace scalar_dataspace(H5S_SCALAR);
                H5::DataType real_type = H5::PredType::NATIVE_DOUBLE;
                H5::DataType size_type = H5::PredType::NATIVE_UINT64;
                H5::DataType bool_type = H5::PredType::NATIVE_HBOOL;
                H5::StrType str_type(H5::PredType::C_S1, 256);

                // create group for mesh config
                H5::Group mesh_group = ctx.file.createGroup(group_name);

                // helper lambda for writing attributes
                auto write_attr =
                    [&](const std::string& name,
                        const auto& value,
                        const H5::DataType& type) -> result_t<void> {
                    try {
                        auto attr = mesh_group.createAttribute(
                            name,
                            type,
                            scalar_dataspace
                        );
                        attr.write(type, &value);
                        attr.close();
                        return result_t<void>::ok();
                    }
                    catch (const H5::Exception& e) {
                        return result_t<void>::error(
                            "failed to write mesh attribute " + name + ": " +
                            e.getDetailMsg()
                        );
                    }
                };

                // helper for array attributes
                auto write_array_attr =
                    [&](const std::string& name,
                        const auto& array,
                        const H5::DataType& type) -> result_t<void> {
                    try {
                        hsize_t dims[1] = {Dims};
                        H5::DataSpace array_space(1, dims);
                        auto attr =
                            mesh_group.createAttribute(name, type, array_space);
                        attr.write(type, array.data());
                        attr.close();
                        return result_t<void>::ok();
                    }
                    catch (const H5::Exception& e) {
                        return result_t<void>::error(
                            "failed to write mesh array attribute " + name +
                            ": " + e.getDetailMsg()
                        );
                    }
                };

                // helper for string attributes
                auto write_str_attr =
                    [&](const std::string& name,
                        const std::string& value) -> result_t<void> {
                    try {
                        auto attr = mesh_group.createAttribute(
                            name,
                            str_type,
                            scalar_dataspace
                        );
                        attr.write(str_type, value.c_str());
                        attr.close();
                        return result_t<void>::ok();
                    }
                    catch (const H5::Exception& e) {
                        return result_t<void>::error(
                            "failed to write mesh string attribute " + name +
                            ": " + e.getDetailMsg()
                        );
                    }
                };

                // serialize grid shape
                auto result =
                    write_array_attr("shape", mesh_config.shape, size_type);
                if (!result.is_ok()) {
                    return result_t<serialization_context_t>::error(
                        result.error()
                    );
                }

                // serialize ghost radius
                result = write_attr(
                    "halo_radius",
                    mesh_config.halo_radius,
                    size_type
                );
                if (!result.is_ok()) {
                    return result_t<serialization_context_t>::error(
                        result.error()
                    );
                }

                // serialize bounds
                result = write_array_attr(
                    "bounds_min",
                    mesh_config.bounds_min,
                    real_type
                );
                if (!result.is_ok()) {
                    return result_t<serialization_context_t>::error(
                        result.error()
                    );
                }

                result = write_array_attr(
                    "bounds_max",
                    mesh_config.bounds_max,
                    real_type
                );
                if (!result.is_ok()) {
                    return result_t<serialization_context_t>::error(
                        result.error()
                    );
                }

                // serialize spacing types as strings
                std::vector<std::string> spacing_strs(Dims);
                for (std::uint64_t i = 0; i < Dims; ++i) {
                    spacing_strs[i] =
                        std::string(serialize(mesh_config.spacing_types[i]));
                }

                // write spacing types as comma-separated string
                std::string spacing_combined = "";
                for (std::uint64_t i = 0; i < Dims; ++i) {
                    if (i > 0) {
                        spacing_combined += ",";
                    }
                    spacing_combined += spacing_strs[i];
                }
                result = write_str_attr("spacing_types", spacing_combined);
                if (!result.is_ok()) {
                    return result_t<serialization_context_t>::error(
                        result.error()
                    );
                }

                // serialize boolean flags
                result =
                    write_attr("homologous", mesh_config.homologous, bool_type);
                if (!result.is_ok()) {
                    return result_t<serialization_context_t>::error(
                        result.error()
                    );
                }

                result = write_attr(
                    "mesh_motion",
                    mesh_config.mesh_motion,
                    bool_type
                );
                if (!result.is_ok()) {
                    return result_t<serialization_context_t>::error(
                        result.error()
                    );
                }

                // serialize expansion state
                result = write_attr(
                    "expansion_factor",
                    mesh_config.expansion_factor,
                    real_type
                );
                if (!result.is_ok()) {
                    return result_t<serialization_context_t>::error(
                        result.error()
                    );
                }

                result = write_attr(
                    "expansion_rate",
                    mesh_config.expansion_rate,
                    real_type
                );
                if (!result.is_ok()) {
                    return result_t<serialization_context_t>::error(
                        result.error()
                    );
                }

                // add dimensionality for reference
                auto dims_val = static_cast<std::uint64_t>(Dims);
                result        = write_attr("dimensions", dims_val, size_type);
                if (!result.is_ok()) {
                    return result_t<serialization_context_t>::error(
                        result.error()
                    );
                }

                // std::cout << "Serialized mesh config to group: " <<
                // group_name
                //           << std::endl;

                mesh_group.close();
                return result_t<serialization_context_t>::ok(
                    ctx.with_dataset(group_name)
                );
            }
            catch (const H5::Exception& e) {
                return result_t<serialization_context_t>::error(
                    "hdf5 error writing mesh config: " + e.getDetailMsg()
                );
            }
        }
    };

    // pipeline operations
    template <field_serializable_c FieldType>
    auto serialize_field(const FieldType& field, const std::string& name)
    {
        return [&field, name](
                   serialization_context_t ctx
               ) -> result_t<serialization_context_t> {
            return serialization_trait_t<FieldType>::serialize(
                field,
                name,
                ctx
            );
        };
    }

    template <typename T>
    auto serialize_metadata(
        const T& metadata,
        const std::string& group_name = "sim_info"
    )
    {
        return [&metadata, group_name](
                   serialization_context_t ctx
               ) -> result_t<serialization_context_t> {
            return metadata_serialization_trait_t<T>::serialize_attributes(
                metadata,
                group_name,
                ctx
            );
        };
    }

    auto create_file(const std::string& filename)
        -> result_t<serialization_context_t>;

    // pipeline helper functions - return callables for and_then chaining
    template <typename FieldType>
    auto serialize_field_components(
        const FieldType& field,
        const std::string& base_name
    )
    {
        return [&field, base_name](
                   serialization_context_t ctx
               ) -> result_t<serialization_context_t> {
            return serialization_trait_t<FieldType>::serialize(
                field,
                base_name,
                ctx
            );
        };
    }

    template <typename T, std::uint64_t Dims>
    auto serialize_scalar_field(
        const field_t<T, Dims>& field,
        const std::string& name
    )
    {
        return [&field, name](
                   serialization_context_t ctx
               ) -> result_t<serialization_context_t> {
            return serialization_trait_t<field_t<T, Dims>>::serialize(
                field,
                name,
                ctx
            );
        };
    }

    template <typename MetaData>
    auto serialize_attributes(
        const MetaData& metadata,
        const std::string& group_name = "sim_info"
    )
    {
        return [&metadata, group_name](
                   serialization_context_t ctx
               ) -> result_t<serialization_context_t> {
            return metadata_serialization_trait_t<
                MetaData>::serialize_attributes(metadata, group_name, ctx);
        };
    }

    template <typename HydroState>
    auto serialize_magnetic_fields(const HydroState& state)
    {
        return [&state](
                   serialization_context_t ctx
               ) -> result_t<serialization_context_t> {
            if constexpr (HydroState::is_mhd) {
                return serialize_scalar_field(state.bstaggs[2], "b1")(ctx)
                    .and_then(serialize_scalar_field(state.bstaggs[1], "b2"))
                    .and_then(serialize_scalar_field(state.bstaggs[0], "b3"));
            }
            else {
                (void) state;   // suppress unused warning
                return result_t<serialization_context_t>::ok(ctx);
            }
        };
    }

    auto close_file()
        -> std::function<result_t<std::string>(serialization_context_t)>;

    // main serialization function for hydro_state_t
    template <hydro_state_serializable_c HydroState, typename MeshConfig>
    result_t<std::string>
    serialize_hydro_state(HydroState& state, const MeshConfig& mesh)
    {
        // ensure all data is synced to cpu
        // state.to_cpu();

        auto& geo     = mesh;
        auto max_iter = std::max_element(geo.shape.begin(), geo.shape.end());
        auto chkpt    = *max_iter;
        auto tnow     = format_real(state.metadata.time);
        auto filename = state.metadata.data_dir +
                        string_format("%d.chkpt." + tnow + ".h5", chkpt);
        std::cout << "Serializing hydro state to: " << filename << std::endl;

        //  monadic pipeline
        return create_file(filename)
            .and_then(serialize_field_components(state.prim, "primitives"))
            .and_then(serialize_magnetic_fields(state))
            .and_then(serialize_attributes(mesh, "mesh_config"))
            .and_then(serialize_attributes(state.metadata))
            .and_then(close_file());
    }

    // operator overloading for pipeline style (as backup)
    template <typename F>
    auto operator|(result_t<serialization_context_t> result, F&& func)
    {
        return result.and_then(std::forward<F>(func));
    }

    // convenience functions for common patterns
    template <typename T, std::uint64_t Dims>
    result_t<std::string> quick_serialize_field(
        const field_t<T, Dims>& field,
        const std::string& filename,
        const std::string& dataset_name = "data"
    )
    {
        return create_file(filename)
            .and_then([&field, &dataset_name](auto ctx) {
                return serialization_trait_t<field_t<T, Dims>>::serialize(
                    field,
                    dataset_name,
                    ctx
                );
            })
            .and_then(close_file());
    }

    // error handling helpers for better error messages
    namespace error_handling {
        template <typename T>
        void handle_serialization_result(
            const result_t<T>& result,
            const std::string& operation
        )
        {
            if (result.is_ok()) {
                std::cout << "✓ " << operation
                          << " succeeded: " << result.value() << std::endl;
            }
            else {
                std::cerr << "✗ " << operation << " failed: " << result.error()
                          << std::endl;
            }
        }

        template <typename T>
        bool
        check_and_log(const result_t<T>& result, const std::string& operation)
        {
            if (!result.is_ok()) {
                std::cerr << "[serialization error] " << operation << ": "
                          << result.error() << std::endl;
                return false;
            }
            return true;
        }
    }   // namespace error_handling

}   // namespace simbi::io

#endif   // SIMBI_SERIALIZATION_HPP
