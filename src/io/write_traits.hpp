// =============================================================================
// write_traits.hpp
//
// type traits for mapping c++ types to hdf5 native types.
// provides `h5_pred_type`, a type trait that is specialized for various c++
// fundamental types (double, float, int, etc.) to map them to the
// corresponding hdf5 native predicate types (`h5::predtype`) used for i/o
// operations.
//
// usage:
//   auto h5_type = h5_pred_type<double>::value();
//   dataset.write(h5_type, &my_double);
// =============================================================================
#pragma once

#include <H5Cpp.h>
#include <cstddef>
#include <cstdint>

namespace simbi::io {
    template <typename T>
    struct h5_pred_type;

    // specialization for double
    template <>
    struct h5_pred_type<double>
    {
        static const H5::PredType& value()
        {
            return H5::PredType::NATIVE_DOUBLE;
        }
    };

    // spec for float
    template <>
    struct h5_pred_type<float>
    {
        static const H5::PredType& value()
        {
            return H5::PredType::NATIVE_FLOAT;
        }
    };

    // spec for uint64_t
    template <>
    struct h5_pred_type<std::uint64_t>
    {
        static const H5::PredType& value()
        {
            return H5::PredType::NATIVE_UINT64;
        }
    };

    // spec for int64_t
    template <>
    struct h5_pred_type<std::int64_t>
    {
        static const H5::PredType& value()
        {
            return H5::PredType::NATIVE_INT64;
        }
    };

    // spec for int
    template <>
    struct h5_pred_type<int>
    {
        static const H5::PredType& value()
        {
            return H5::PredType::NATIVE_INT;
        }
    };

    // spec for bool
    template <>
    struct h5_pred_type<bool>
    {
        static const H5::PredType& value()
        {
            return H5::PredType::NATIVE_HBOOL;
        }
    };

#ifdef __APPLE__
    template <>
    struct h5_pred_type<std::size_t>
    {
        static const H5::PredType& value()
        {
            return H5::PredType::NATIVE_UINT64;
        }
    };
#endif

    template <>
    struct h5_pred_type<std::uint32_t>
    {
        static const H5::PredType& value()
        {
            return H5::PredType::NATIVE_UINT32;
        }
    };
} // namespace simbi::io
