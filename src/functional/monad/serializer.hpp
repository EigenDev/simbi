#ifndef SERIALIZATION_HPP
#define SERIALIZATION_HPP

#include "compat.hpp"              // for real, DEV, etc
#include "compute/field.hpp"       // for field_t<T, Dims>
#include "containers/vector.hpp"   // for simbi::vector_t
#include "functional/fp.hpp"
#include "io/tabulate/table.hpp"   // for tabulate::table_t
#include "mesh/mesh_config.hpp"    // for mesh::mesh_config_t
#include "physics/ib/body.hpp"   // for softening_length, accretion_efficienct, etc
#include "physics/ib/body_delta.hpp"   // for body_delta_t
#include "physics/ib/collection.hpp"   // for ib::body_collection_t
#include "result.hpp"                  // for result_t<T> monad
#include "utility/enums.hpp"           // for Regime, Geometry, etc
#include "utility/helpers.hpp"         // for simbi::helpers::serialize

#include <H5Cpp.h>         // for HDF5 C++ API
#include <concepts>        // for concepts
#include <cstddef>         // for std::size_t
#include <cstdint>         // for std::uint64_t
#include <functional>      // for std::function
#include <optional>        // for std::optional
#include <string>          // for std::string
#include <type_traits>     // for std::is_arithmetic_v, std::same_as
#include <unordered_map>   // for std::unordered_map
#include <utility>         // for std::move
#include <vector>          // for std::vector

namespace simbi::io {
    using namespace simbi::helpers;

    template <typename T>
    concept body_collection_serializable_c = requires(const T& collection) {
        collection.size();
        collection.name();
        collection.begin();
        collection.end();
    };

    template <typename T>
    concept body_diagnostics_serializable_c = requires(const T& diagnostics) {
        diagnostics.force_1;
        diagnostics.total_mass;
        diagnostics.accretion_rate;
    };

    template <typename T, std::uint64_t N>
    void serialize_vector_component(
        H5::Group& group,
        const std::string& name,
        const vector_t<T, N>& vec
    )
    {
        hsize_t dims[] = {N};
        auto dataspace = H5::DataSpace(1, dims);
        auto dataset   = group.createDataSet(
            name,
            H5::PredType::NATIVE_DOUBLE,   // assuming T is real/double
            dataspace
        );
        dataset.write(vec.data(), H5::PredType::NATIVE_DOUBLE);
    }

    template <typename T>
    void
    serialize_scalar(H5::Group& group, const std::string& name, const T& value)
    {
        auto scalar_space = H5::DataSpace(H5S_SCALAR);
        auto attr         = group.createAttribute(
            name,
            H5::PredType::NATIVE_DOUBLE,
            scalar_space
        );
        attr.write(H5::PredType::NATIVE_DOUBLE, &value);
    }

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
                    auto result = write_attr(
                        "adiabatic_index",
                        metadata.gamma,
                        real_type
                    );
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
                    auto result =
                        write_attr("cfl_number", metadata.cfl, real_type);
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
                    auto result =
                        write_attr("end_time", metadata.tend, real_type);
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
                if constexpr (requires { metadata.checkpoint_index; }) {
                    auto result = write_attr(
                        "checkpoint_index",
                        metadata.checkpoint_index,
                        uint64_type
                    );
                    if (!result.is_ok()) {
                        return result_t<serialization_context_t>::error(
                            result.error()
                        );
                    }
                }
                if constexpr (requires { metadata.checkpoint_zones; }) {
                    auto result = write_attr(
                        "checkpoint_zones",
                        metadata.checkpoint_zones,
                        uint64_type
                    );
                    if (!result.is_ok()) {
                        return result_t<serialization_context_t>::error(
                            result.error()
                        );
                    }
                }
                if constexpr (requires { metadata.data_dir; }) {
                    auto result = write_str_attr("data_dir", metadata.data_dir);
                    if (!result.is_ok()) {
                        return result_t<serialization_context_t>::error(
                            result.error()
                        );
                    }
                }
                if constexpr (requires { metadata.dlogt; }) {
                    auto result = write_attr("dlogt", metadata.dlogt, int_type);
                    if (!result.is_ok()) {
                        return result_t<serialization_context_t>::error(
                            result.error()
                        );
                    }
                }
                if constexpr (requires { metadata.checkpoint_interval; }) {
                    auto result = write_attr(
                        "checkpoint_interval",
                        metadata.checkpoint_interval,
                        real_type
                    );
                    if (!result.is_ok()) {
                        return result_t<serialization_context_t>::error(
                            result.error()
                        );
                    }
                }
                if constexpr (requires { metadata.checkpoint_time; }) {
                    auto result = write_attr(
                        "checkpoint_time",
                        metadata.checkpoint_time,
                        real_type
                    );
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

    // body collection serialization trait
    template <std::uint64_t Dims, std::uint64_t MaxBodies>
    struct serialization_trait_t<
        std::optional<body::body_collection_t<Dims, MaxBodies>>> {
        constexpr static bool serializable = true;

        static result_t<serialization_context_t> serialize(
            const std::optional<body::body_collection_t<Dims, MaxBodies>>&
                optional_collection,
            serialization_context_t ctx
        )
        {
            if (!optional_collection.has_value()) {
                return result_t<serialization_context_t>::error(
                    "no body collection to serialize"
                );
            }

            const auto& collection = optional_collection.value();
            if (collection.empty()) {
                return result_t<serialization_context_t>::error(
                    "body collection is empty, nothing to serialize"
                );
            }

            const auto dataset_name = "bodies/" + collection.name();

            // serialize body count and metadata
            auto scalar_space = H5::DataSpace(H5S_SCALAR);
            auto group        = ctx.file.createGroup("bodies");

            // basic collection info
            auto size_attr = group.createAttribute(
                "body_count",
                H5::PredType::NATIVE_UINT64,
                scalar_space
            );
            const auto size = collection.size();
            size_attr.write(H5::PredType::NATIVE_UINT64, &size);

            auto name_type =
                H5::StrType(H5::PredType::C_S1, collection.name().size());
            auto name_attr =
                group.createAttribute("system_name", name_type, scalar_space);
            name_attr.write(name_type, collection.name().c_str());

            auto ref_frame_type = H5::StrType(
                H5::PredType::C_S1,
                collection.reference_frame().size()
            );
            auto ref_frame_attr = group.createAttribute(
                "reference_frame",
                ref_frame_type,
                scalar_space
            );
            ref_frame_attr.write(
                ref_frame_type,
                collection.reference_frame().c_str()
            );

            // serialize binary parameters if present
            if (collection.binary_params_) {
                auto binary_group  = group.createGroup("binary_params");
                const auto& params = collection.binary_params();

                auto write_param = [&](const char* name, real value) {
                    auto attr = binary_group.createAttribute(
                        name,
                        H5::PredType::NATIVE_DOUBLE,
                        scalar_space
                    );
                    attr.write(H5::PredType::NATIVE_DOUBLE, &value);
                };

                write_param("total_mass", params.total_mass);
                write_param("semi_major", params.semi_major);
                write_param("eccentricity", params.eccentricity);
                write_param("mass_ratio", params.mass_ratio);
                write_param("orbital_period", params.orbital_period);

                auto bool_attr = binary_group.createAttribute(
                    "is_circular_orbit",
                    H5::PredType::NATIVE_HBOOL,
                    scalar_space
                );
                bool_attr.write(
                    H5::PredType::NATIVE_HBOOL,
                    &params.is_circular_orbit
                );
            }

            // serialize individual bodies
            collection.visit_all([&](const auto& body) {
                auto bg = group.createGroup("body_" + std::to_string(body.idx));
                serialize_body(body, bg);
            });

            return result_t<serialization_context_t>::ok(
                ctx.with_dataset(dataset_name)
            );
        }

      private:
        template <typename Body>
        static void serialize_body(const Body& body, H5::Group& group)
        {
            using namespace simbi::body;
            auto scalar_space = H5::DataSpace(H5S_SCALAR);

            // common properties
            auto mass_attr = group.createAttribute(
                "mass",
                H5::PredType::NATIVE_DOUBLE,
                scalar_space
            );
            mass_attr.write(H5::PredType::NATIVE_DOUBLE, &body.mass);

            auto radius_attr = group.createAttribute(
                "radius",
                H5::PredType::NATIVE_DOUBLE,
                scalar_space
            );
            radius_attr.write(H5::PredType::NATIVE_DOUBLE, &body.radius);

            // position and velocity arrays
            hsize_t vec_dims[] = {Dims};
            auto vec_space     = H5::DataSpace(1, vec_dims);

            auto pos_dataset = group.createDataSet(
                "position",
                H5::PredType::NATIVE_DOUBLE,
                vec_space
            );
            pos_dataset.write(
                body.position.data(),
                H5::PredType::NATIVE_DOUBLE
            );

            auto vel_dataset = group.createDataSet(
                "velocity",
                H5::PredType::NATIVE_DOUBLE,
                vec_space
            );
            vel_dataset.write(
                body.velocity.data(),
                H5::PredType::NATIVE_DOUBLE
            );

            // type-specific properties
            if constexpr (has_gravitational_capability_c<Body>) {
                auto soft_attr = group.createAttribute(
                    "softening_length",
                    H5::PredType::NATIVE_DOUBLE,
                    scalar_space
                );

                const auto soft = softening_length(body);
                soft_attr.write(H5::PredType::NATIVE_DOUBLE, &soft);
            }

            if constexpr (has_accretion_capability_c<Body>) {
                auto sink_rate_attr = group.createAttribute(
                    "sink_rate",
                    H5::PredType::NATIVE_DOUBLE,
                    scalar_space
                );
                const auto eff = sink_rate(body);
                sink_rate_attr.write(H5::PredType::NATIVE_DOUBLE, &eff);

                auto sink_delta_attr = group.createAttribute(
                    "sink_delta",
                    H5::PredType::NATIVE_DOUBLE,
                    scalar_space
                );
                const auto sdelta = sink_delta(body);
                sink_delta_attr.write(H5::PredType::NATIVE_DOUBLE, &sdelta);

                auto accr_rad_attr = group.createAttribute(
                    "accretion_radius",
                    H5::PredType::NATIVE_DOUBLE,
                    scalar_space
                );
                const auto rad = accretion_radius(body);
                accr_rad_attr.write(H5::PredType::NATIVE_DOUBLE, &rad);
            }

            if constexpr (has_rigid_capability_c<Body>) {
                auto inertia_attr = group.createAttribute(
                    "inertia",
                    H5::PredType::NATIVE_DOUBLE,
                    scalar_space
                );
                const auto inert = inertia(body);
                inertia_attr.write(H5::PredType::NATIVE_DOUBLE, &inert);

                auto apply_no_slip_attr = group.createAttribute(
                    "apply_no_slip",
                    H5::PredType::NATIVE_HBOOL,
                    scalar_space
                );
                const auto ans = apply_no_slip(body);
                apply_no_slip_attr.write(H5::PredType::NATIVE_HBOOL, &ans);
            }

            // if constexpr (has_elastic_capability_c<Body>) {
            //     auto elastic_mod_attr = group.createAttribute(
            //         "elastic_modulus",
            //         H5::PredType::NATIVE_DOUBLE,
            //         scalar_space
            //     );
            //     const auto elastic_mod = body.elastic_modulus();
            //     elastic_mod_attr.write(
            //         H5::PredType::NATIVE_DOUBLE,
            //         &elastic_mod
            //     );

            //     auto poisson_ratio_attr = group.createAttribute(
            //         "poisson_ratio",
            //         H5::PredType::NATIVE_DOUBLE,
            //         scalar_space
            //     );
            //     const auto poisson_ratio = poisson_ratio(body);
            //     poisson_ratio_attr.write(
            //         H5::PredType::NATIVE_DOUBLE,
            //         &poisson_ratio
            //     );
            // }

            // if constexpr (has_deformable_capability_c<Body>) {
            //     auto yield_stress_attr = group.createAttribute(
            //         "yield_stress",
            //         H5::PredType::NATIVE_DOUBLE,
            //         scalar_space
            //     );
            //     const auto yield_stress = body.yield_stress();
            //     yield_stress_attr.write(
            //         H5::PredType::NATIVE_DOUBLE,
            //         &yield_stress
            //     );

            //     auto plastic_strain_attr = group.createAttribute(
            //         "plastic_strain",
            //         H5::PredType::NATIVE_DOUBLE,
            //         scalar_space
            //     );
            //     const auto plastic_strain = plastic_strain(body);
            //     plastic_strain_attr.write(
            //         H5::PredType::NATIVE_DOUBLE,
            //         &plastic_strain
            //     );
            // }

            auto cap_attr = group.createAttribute(
                "capabilities",
                H5::PredType::NATIVE_INT,
                scalar_space
            );
            auto caps = body.caps();
            cap_attr.write(H5::PredType::NATIVE_INT, &caps);
        }
    };

    // body diagnostics serialization trait
    template <std::uint64_t Dims, std::uint64_t MaxBodies>
    struct serialization_trait_t<
        vector_t<body::body_delta_t<Dims>, MaxBodies>> {
        constexpr static bool serializable = true;

        static result_t<serialization_context_t> serialize(
            const vector_t<body::body_delta_t<Dims>, MaxBodies>& diagnostics,
            serialization_context_t ctx,
            real dt
        )
        {
            const auto dataset_name = "diagnostics/body_diagnostics";

            try {
                static auto prev_masses =
                    diagnostics | fp::map([](const auto& body) -> real {
                        return body.prev_mass_delta;
                    }) |
                    fp::collect<vector_t<real, MaxBodies>>;

                auto diag_group = ctx.file.createGroup("diagnostics");
                auto body_diag_group =
                    diag_group.createGroup("body_diagnostics");

                // extract and serialize force components
                for (std::uint64_t body_idx = 0; body_idx < MaxBodies;
                     ++body_idx) {
                    const auto& delta = diagnostics[body_idx];
                    const auto& pmass = prev_masses[body_idx];

                    // serialize force components for this body
                    serialize_vector_component(
                        body_diag_group,
                        "force_" + std::to_string(body_idx),
                        delta.force_delta
                    );

                    // serialize torque
                    serialize_vector_component(
                        body_diag_group,
                        "torque_" + std::to_string(body_idx),
                        delta.torque_delta
                    );

                    // serialize mass delta
                    serialize_scalar(
                        body_diag_group,
                        "cumulative_mass_delta_" + std::to_string(body_idx),
                        delta.mass_delta
                    );

                    const auto dm   = delta.mass_delta - pmass;
                    const auto mdot = (dt > 0) ? dm / dt : 0.0;

                    serialize_scalar(
                        body_diag_group,
                        "accretion_rate_" + std::to_string(body_idx),
                        mdot
                    );
                    prev_masses[body_idx] = delta.mass_delta;
                }

                return result_t<serialization_context_t>::ok(
                    ctx.with_dataset("diagnostics")
                );
            }
            catch (const H5::Exception& e) {
                return result_t<serialization_context_t>::error(
                    "hdf5 error writing body diagnostics: " + e.getDetailMsg()
                );
            }

            return result_t<serialization_context_t>::ok(
                ctx.with_dataset(dataset_name)
            );
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

    // extend existing pipeline functions
    auto serialize_body_collection(const auto& collection)
    {
        return [collection](
                   serialization_context_t ctx
               ) -> result_t<serialization_context_t> {
            using collection_type = std::decay_t<decltype(collection)>;
            return serialization_trait_t<collection_type>::serialize(
                collection,
                ctx
            );
        };
    }

    auto serialize_body_diagnostics(const auto& diagnostics, real time_since)
    {
        return [&diagnostics, time_since](
                   serialization_context_t ctx
               ) -> result_t<serialization_context_t> {
            using diagnostics_type = std::decay_t<decltype(diagnostics)>;
            return serialization_trait_t<diagnostics_type>::serialize(
                diagnostics,
                ctx,
                time_since
            );
        };
    }

    // convenience function for complete body system serialization
    auto serialize_body_system(
        const auto& collection,
        const auto& diagnostics,
        real time_since
    )
    {
        return [collection, &diagnostics, time_since](
                   serialization_context_t ctx
               ) -> result_t<serialization_context_t> {
            return serialize_body_collection(collection)(ctx).and_then(
                serialize_body_diagnostics(diagnostics, time_since)
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

    template <typename HydroState>
    auto compute_filename(const HydroState& state)
    {
        static std::int64_t tchunk_order_of_mag = 2;

        const auto meta                = state.metadata;
        const auto data_directory      = meta.data_dir;
        const auto step                = meta.checkpoint_index;
        const auto timestepping_of_mag = std::floor(std::log10(meta.time));

        if (timestepping_of_mag > tchunk_order_of_mag) {
            tchunk_order_of_mag += 1;
        }

        std::string tnow;
        if (meta.dlogt != 0) {
            const auto timestepping_of_mag = std::floor(std::log10(step));
            if (timestepping_of_mag > tchunk_order_of_mag) {
                tchunk_order_of_mag += 1;
            }
            tnow = format_real(step);
        }
        else if (!state.in_failure_state) {
            tnow = format_real(meta.checkpoint_identifier());
        }
        else {
            if (state.was_interrupted) {
                tnow = "interrupted";
            }
            else {
                tnow = "crashed";
            }
        }

        return data_directory +
               string_format("%d.chkpt." + tnow + ".h5", meta.checkpoint_zones);
    }

    // main serialization function for hydro_state_t
    template <hydro_state_serializable_c HydroState, typename MeshConfig>
    void serialize_hydro_state(
        HydroState& state,
        const MeshConfig& mesh,
        Table& table
    )
    {
        auto& meta                  = state.metadata;
        static auto last_chkpt_time = meta.prev_checkpoint_time;
        const auto filename         = compute_filename(state);
        const auto delta_time       = meta.checkpoint_time - last_chkpt_time;
        const auto diagnostics      = state.diagnostics->consolidate();

        table.post_info("[Writing checkpoint to path: " + filename + "]");
        table.refresh();
        //  monadic pipeline
        create_file(filename)
            .and_then(serialize_field_components(state.prim, "primitives"))
            .and_then(serialize_magnetic_fields(state))
            .and_then(serialize_attributes(mesh, "mesh_config"))
            .and_then(serialize_attributes(meta))
            .and_then(
                serialize_body_system(state.bodies, diagnostics, delta_time)
            )
            .and_then(close_file());

        last_chkpt_time = meta.checkpoint_time;
    }

    // operator overloading for pipeline style (as backup)
    template <typename F>
    auto operator|(result_t<serialization_context_t> result, F&& func)
    {
        return result.and_then(std::forward<F>(func));
    }

}   // namespace simbi::io

#endif   // SERIALIZATION_HPP
