#ifndef IO_TYPE_TRAITS_HPP
#define IO_TYPE_TRAITS_HPP

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

    // #ifdef __clang__
    //     template <>
    //     struct h5_pred_type<std::size_t> {
    //         static const H5::PredType& value()
    //         {
    //             return H5::PredType::NATIVE_UINT64;
    //         }
    //     };
    // #endif

    template <>
    struct h5_pred_type<std::uint32_t>
    {
        static const H5::PredType& value()
        {
            return H5::PredType::NATIVE_UINT32;
        }
    };
} // namespace simbi::io

#endif
