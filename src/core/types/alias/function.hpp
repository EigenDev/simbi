#ifndef FUNCTION_HPP
#define FUNCTION_HPP

#include "build_options.hpp"

namespace simbi {
    // Forward declarations for dimension-specific function types
    template <size_type D>
    struct user_function;

    // specializations for 1D
    template <>
    struct user_function<1> {
        using type    = void(real, real, real*);
        using pointer = void (*)(real, real, real*);
        static constexpr const char* args_description = "x, t, result_array";
    };

    // specializations for 2D
    template <>
    struct user_function<2> {
        using type    = void(real, real, real, real*);
        using pointer = void (*)(real, real, real, real*);
        static constexpr const char* args_description = "x, y, t, result_array";
    };

    // specializations for 3D
    template <>
    struct user_function<3> {
        using type    = void(real, real, real, real, real*);
        using pointer = void (*)(real, real, real, real, real*);
        static constexpr const char* args_description =
            "x, y, z, t, result_array";
    };

    // helper aliases
    template <size_type D>
    using user_function_t = typename user_function<D>::type;

    template <size_type D>
    using user_function_ptr_t = typename user_function<D>::pointer;
}   // namespace simbi

#endif   // FUNCTION_HPP
