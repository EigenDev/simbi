#ifndef IO_H5_SERIALIZABLE_HPP
#define IO_H5_SERIALIZABLE_HPP

#include "compat.hpp"
#include "write_policy.hpp"
#include "write_traits.hpp"

#include <H5Cpp.h>
#include <array>
#include <concepts>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

namespace simbi::io {

    // =========================================================================
    // primary template - types are not serializable by default
    // =========================================================================
    template <typename T, typename = void>
    struct h5_serializable : std::false_type {
    };

    // =========================================================================
    // concept for checking if a type is h5-serializable
    // =========================================================================
    template <typename T>
    concept serializable_c =
        requires(H5::Group& g, const T& val, const write_policy_t& policy) {
            { h5_serializable<T>::write(g, val, policy) };
            { h5_serializable<T>::read(g) } -> std::same_as<T>;
            {
                h5_serializable<T>::group_name
            } -> std::convertible_to<std::string_view>;
        };

    // =========================================================================
    // dataset helpers
    // =========================================================================

    // write a vector as a dataset with policy-controlled precision/compression
    template <typename T>
    void write_dataset(
        H5::Group& group,
        const std::string& name,
        const std::vector<T>& data,
        const std::vector<hsize_t>& dims,
        const write_policy_t& policy
    )
    {
        H5::DataSpace space(static_cast<int>(dims.size()), dims.data());
        auto plist = policy.creation_props(dims);
        auto dtype = policy.data_type();

        auto dataset = group.createDataSet(name, dtype, space, plist);

        // handle precision downsampling
        if (policy.precision == precision_t::float32 &&
            std::is_same_v<T, double>) {
            std::vector<float> temp(data.begin(), data.end());
            dataset.write(temp.data(), H5::PredType::NATIVE_FLOAT);
        }
        else {
            dataset.write(data.data(), h5_pred_type<T>::value());
        }
    }

    // write a scalar as a dataset
    template <typename T>
    void write_scalar_dataset(
        H5::Group& group,
        const std::string& name,
        T value,
        const write_policy_t& policy
    )
    {
        std::vector<hsize_t> dims{1};
        std::vector<T> data{value};
        write_dataset(group, name, data, dims, policy);
    }

    // read a dataset into a vector
    template <typename T>
    std::vector<T> read_dataset(const H5::Group& group, const std::string& name)
    {
        auto dataset = group.openDataSet(name);
        auto space   = dataset.getSpace();

        int ndims = space.getSimpleExtentNdims();
        std::vector<hsize_t> dims(ndims);
        space.getSimpleExtentDims(dims.data());

        std::size_t total = 1;
        for (auto d : dims) {
            total *= d;
        }

        std::vector<T> data(total);
        dataset.read(data.data(), h5_pred_type<T>::value());
        return data;
    }

    // read a scalar dataset
    template <typename T>
    T read_scalar_dataset(const H5::Group& group, const std::string& name)
    {
        auto data = read_dataset<T>(group, name);
        return data.empty() ? T{} : data[0];
    }

    // =========================================================================
    // attribute helpers
    // =========================================================================

    template <typename T>
    void write_attribute(H5::Group& group, const std::string& name, T value)
    {
        H5::DataSpace scalar_space(H5S_SCALAR);
        auto attr =
            group.createAttribute(name, h5_pred_type<T>::value(), scalar_space);
        attr.write(h5_pred_type<T>::value(), &value);
    }

    // specialization for strings
    inline void write_attribute(
        H5::Group& group,
        const std::string& name,
        const std::string& value
    )
    {
        H5::StrType str_type(H5::PredType::C_S1, value.size() + 1);
        H5::DataSpace scalar_space(H5S_SCALAR);
        auto attr = group.createAttribute(name, str_type, scalar_space);
        attr.write(str_type, value.c_str());
    }

    template <typename T>
    T read_attribute(const H5::Group& group, const std::string& name)
    {
        auto attr = group.openAttribute(name);
        T value;
        attr.read(h5_pred_type<T>::value(), &value);
        return value;
    }

    // specialization for strings
    template <>
    inline std::string
    read_attribute<std::string>(const H5::Group& group, const std::string& name)
    {
        auto attr       = group.openAttribute(name);
        auto str_type   = attr.getStrType();
        std::size_t len = str_type.getSize();

        std::string value;
        value.resize(len);
        attr.read(str_type, value.data());

        // trim null terminator if present
        if (!value.empty() && value.back() == '\0') {
            value.pop_back();
        }
        return value;
    }

    // =========================================================================
    // array helpers (fixed-size vectors)
    // =========================================================================

    template <typename T, std::size_t N>
    void write_array(
        H5::Group& group,
        const std::string& name,
        const T (&arr)[N],
        const write_policy_t& policy
    )
    {
        std::vector<T> data(arr, arr + N);
        std::vector<hsize_t> dims{N};
        write_dataset(group, name, data, dims, policy);
    }

    template <typename T, std::size_t N>
    void write_array(
        H5::Group& group,
        const std::string& name,
        const std::array<T, N>& arr,
        const write_policy_t& policy
    )
    {
        std::vector<T> data(arr.begin(), arr.end());
        std::vector<hsize_t> dims{N};
        write_dataset(group, name, data, dims, policy);
    }

    template <typename T, std::size_t N>
    std::array<T, N> read_array(const H5::Group& group, const std::string& name)
    {
        auto data = read_dataset<T>(group, name);
        std::array<T, N> arr{};
        for (std::size_t ii = 0; ii < N && ii < data.size(); ++ii) {
            arr[ii] = data[ii];
        }
        return arr;
    }

    // =========================================================================
    // group helpers
    // =========================================================================

    inline H5::Group
    create_or_open_group(H5::Group& parent, const std::string& name)
    {
        if (parent.nameExists(name)) {
            return parent.openGroup(name);
        }
        return parent.createGroup(name);
    }

    inline bool group_exists(const H5::Group& parent, const std::string& name)
    {
        return parent.nameExists(name);
    }

}   // namespace simbi::io

#endif   // IO_H5_SERIALIZABLE_HPP
