#ifndef VECTOR_HPP
#define VECTOR_HPP

#include "base/concepts.hpp"
#include "compute/functional/fp.hpp"
#include "compute/functional/monad/maybe.hpp"
#include "config.hpp"
#include "utility/enums.hpp"
#include <array>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iterator>
#include <ostream>
#include <type_traits>

namespace simbi {
    using namespace simbi::concepts;

    // forward declarations
    template <typename T, std::uint64_t Dims>
    struct vector_t;

    namespace detail {
        // type promotion helper
        template <typename T, typename U>
        using promote_t = std::common_type_t<T, U>;
    }   // namespace detail

    namespace vecops {
        using namespace simbi::concepts;
        // dot product
        template <VectorLike Vec1, VectorLike Vec2>
        DUAL constexpr auto dot(const Vec1& a, const Vec2& b)
        {
            using T        = typename Vec1::value_type;
            using U        = typename Vec2::value_type;
            using result_t = detail::promote_t<T, U>;
            return fp::zip(a, b) |
                   fp::unpack_map([](auto l, auto m) -> result_t {
                       return m * l;
                   }) |
                   fp::sum;
        }

        // norm
        template <VectorLike Vec>
        DUAL constexpr auto norm(const Vec& vec)
        {
            return std::sqrt(dot(vec, vec));
        }

        // normalize
        template <VectorLike Vec>
        DUAL constexpr auto normalize(const Vec& vec)
        {
            using result_t = detail::promote_t<typename Vec::value_type, real>;
            const auto n   = norm(vec);
            return n > 0 ? vec | fp::map([n](const auto& x) -> result_t {
                               return x / n;
                           }) | fp::collect<vector_t<result_t, Vec::dimensions>>
                         : vec;
        }

        // cross product
        template <VectorLike Vec1, VectorLike Vec2>
        DUAL constexpr auto cross(const Vec1& a, const Vec2& b)
            requires(Vec1::dimensions == 3 && Vec2::dimensions == 3)
        {
            using T = decltype(a[0] * b[0]);
            return vector_t<T, 3>{
              a[1] * b[2] - a[2] * b[1],
              a[2] * b[0] - a[0] * b[2],
              a[0] * b[1] - b[0] * a[1]
            };
        }

        // cross product magnitude for Dim = 2
        template <VectorLike Vec1, VectorLike Vec2>
        DUAL constexpr auto cross(const Vec1& a, const Vec2& b)
            requires(Vec1::dimensions == 2 && Vec2::dimensions == 2)
        {
            return a[0] * b[1] - a[1] * b[0];
        }

        // cross product component
        template <VectorLike Vec1, VectorLike Vec2>
        DUAL constexpr auto
        cross_component(const Vec1& a, const Vec2& b, std::uint64_t ehat)
        {
            using T = decltype(a[0] * b[0]);
            if (ehat == 1) {
                return a[1] * b[2] - a[2] * b[1];
            }
            if (ehat == 2) {
                return a[2] * b[0] - a[0] * b[2];
            }
            if (ehat == 3) {
                return a[0] * b[1] - a[1] * b[0];
            }
            return static_cast<T>(0.0);
        }

        // helpers to rotate vectors by some angle
        template <VectorLike Vec, typename T>
        DEV static constexpr auto rotate_2D(const Vec& vec, const T& angle)
        {
            return vector_t<T, 2>{
              vec[0] * std::cos(angle) - vec[1] * std::sin(angle),
              vec[0] * std::sin(angle) + vec[1] * std::cos(angle)
            };
        }

        template <VectorLike Vec, typename T>
        DEV static constexpr auto rotate_3D(const Vec& vec, const T& angle)
        {
            return Vec{
              vec[0] * std::cos(angle) - vec[1] * std::sin(angle),
              vec[0] * std::sin(angle) + vec[1] * std::cos(angle),
              vec[2]
            };
        }

        // general rotation function that checks the dimension at compile-time
        template <VectorLike Vec, typename T>
        DEV static constexpr auto rotate(const Vec& vec, const T& angle)
        {
            if constexpr (Vec::dimensions == 2) {
                return rotate_2D(vec, angle);
            }
            else {
                return rotate_3D(vec, angle);
            }
        }

        template <VectorLike Vec>
        DEV auto constexpr spherical_to_cartesian(const Vec& vec)
        {
            if constexpr (Vec::dimensions == 1) {
                return vec;
            }
            else if constexpr (Vec::dimensions == 2) {   // r-theta, not r-phi
                return Vec{
                  vec[0] * std::sin(vec[1]),
                  vec[0] * std::cos(vec[1])
                };
            }
            else {
                return Vec{
                  vec[0] * std::sin(vec[1]) * std::cos(vec[2]),
                  vec[0] * std::sin(vec[1]) * std::sin(vec[2]),
                  vec[0] * std::cos(vec[1])
                };
            }
        }

        template <VectorLike Vec>
        DEV auto constexpr cylindrical_to_cartesian(const Vec& vec)
        {
            if constexpr (Vec::dimensions == 1) {
                return vec;
            }
            else if constexpr (Vec::dimensions == 2) {
                return Vec{
                  vec[0] * std::cos(vec[1]),
                  vec[0] * std::sin(vec[1])
                };
            }
            else {
                return Vec{
                  vec[0] * std::cos(vec[1]),
                  vec[0] * std::sin(vec[1]),
                  vec[2]
                };
            }
        }

        template <VectorLike Vec>
        DEV auto constexpr cartesian_to_spherical(const Vec& vec)
        {
            if constexpr (Vec::dimensions == 1) {
                return vec;
            }
            else if constexpr (Vec::dimensions == 2) {
                return Vec{vec.norm(), std::atan2(vec[1], vec[0])};
            }
            else {
                return Vec{
                  vec.norm(),
                  std::acos(vec[2] / vec.norm()),
                  std::atan2(vec[1], vec[0])
                };
            }
        }

        template <VectorLike Vec>
        DEV auto constexpr centralize_cartesian_to_spherical(const Vec& vec)
        {
            if constexpr (Vec::dimensions == 1) {
                return vec;
            }
            else if constexpr (Vec::dimensions == 2) {
                return Vec{
                  vec.norm(),
                  0.0,
                };
            }
            else {
                return Vec{vec.norm(), 0.0, 0.0};
            }
        }

        template <VectorLike Vec>
        DEV auto constexpr centralize_cartesian_to_cylindrical(const Vec& vec)
        {
            if constexpr (Vec::dimensions == 1) {
                return vec;
            }
            else if constexpr (Vec::dimensions == 2) {
                return Vec{
                  vec.norm(),
                  0.0,
                };
            }
            else {
                return Vec{vec.norm(), 0.0, 0.0};
            }
        }

        template <VectorLike Vec>
        DEV auto constexpr cartesian_to_cylindrical(const Vec& vec)
        {
            if constexpr (Vec::dimensions == 1) {
                return vec;
            }
            else if constexpr (Vec::dimensions == 2) {
                return Vec{vec.norm(), std::atan2(vec[1], vec[0])};
            }
            else {
                return Vec{vec.norm(), std::atan2(vec[1], vec[0]), vec[2]};
            }
        }

        // convert a cartesian vector to a curvlinear
        // coordinate system
        template <VectorLike Vec>
        DEV auto to_geometry(const Vec& vec, Geometry geometry)
        {
            if (geometry == Geometry::SPHERICAL) {
                return cartesian_to_spherical(vec);
            }
            else if (geometry == Geometry::CYLINDRICAL) {
                return cartesian_to_cylindrical(vec);
            }
            else {
                return vec;
            }
        }

        // project some vector onto the mesh
        // using the mesh basis vectors
        template <VectorLike Vec>
        DEV auto centralize(const Vec& vec, Geometry mesh_geometry)
        {
            switch (mesh_geometry) {
                case Geometry::SPHERICAL:
                    return centralize_cartesian_to_spherical(vec);
                case Geometry::CYLINDRICAL:
                    return centralize_cartesian_to_cylindrical(vec);
                default: return vec;
            }
        }
    }   // namespace vecops

    // -------------------------------------------------------------
    // Vector: pure value-based immutable vector with direct storage
    // -------------------------------------------------------------
    template <typename T, std::uint64_t Dims>
    struct vector_t {
        T storage[Dims];   // direct storage of elements
        static inline T zero_value{};

        // type definitions for type traits and functional interfaces
        using value_type                          = T;
        using reference                           = T&;
        using const_reference                     = const T&;
        static constexpr std::uint64_t dimensions = Dims;

        // element access
        DUAL constexpr reference operator[](std::uint64_t idx)
        {
            return storage[idx];
        }

        DUAL constexpr const_reference operator[](std::uint64_t idx) const
        {
            return storage[idx];
        }

        DUAL constexpr bool is_zero() const
        {
            bool result = true;
            for (std::uint64_t ii = 0; ii < Dims; ++ii) {
                result &= (storage[ii] == 0.0);
            }
            return result;
        }

        // safe access with bounds checking
        DUAL constexpr maybe_t<reference> at(std::uint64_t idx)
        {
            if (idx < Dims) {
                return maybe_t<reference>(storage[idx]);
            }
            return Nothing;
        }

        DUAL constexpr maybe_t<const_reference> at(std::uint64_t idx) const
        {
            if (idx < Dims) {
                return maybe_t<const_reference>(storage[idx]);
            }
            return Nothing;
        }

        DUAL auto reverse() const
        {
            // traditional for loop version
            // vector_t<T, Dims> result;
            // for (std::uint64_t ii = 0; ii < Dims; ++ii) {
            //     result[ii] = storage[Dims - ii - 1];
            // }
            // return result;
            return *this | fp::map([this](const auto& x) -> T {
                return storage[Dims - 1 - (&x - &storage[0])];
            }) | fp::collect<vector_t<T, Dims>>;
        }

        // data access for algorithms
        DUAL constexpr T* data() { return &storage[0]; }
        DUAL constexpr const T* data() const { return &storage[0]; }

        // size and capacity
        DUAL constexpr std::uint64_t size() const { return Dims; }

        // iterators for standard algorithms (forward)
        DUAL constexpr T* begin() { return &storage[0]; }
        DUAL constexpr T* end() { return &storage[0] + Dims; }
        DUAL constexpr const T* begin() const { return &storage[0]; }
        DUAL constexpr const T* end() const { return &storage[0] + Dims; }
        DUAL constexpr const T* cbegin() const { return &storage[0]; }
        DUAL constexpr const T* cend() const { return &storage[0] + Dims; }

        // reverse iterators
        DUAL constexpr std::reverse_iterator<T*> rbegin()
        {
            return std::reverse_iterator<T*>(end());
        }
        DUAL constexpr std::reverse_iterator<T*> rend()
        {
            return std::reverse_iterator<T*>(begin());
        }
        DUAL constexpr std::reverse_iterator<const T*> rbegin() const
        {
            return std::reverse_iterator<const T*>(end());
        }
        DUAL constexpr std::reverse_iterator<const T*> rend() const
        {
            return std::reverse_iterator<const T*>(begin());
        }
        DUAL constexpr std::reverse_iterator<const T*> crbegin() const
        {
            return std::reverse_iterator<const T*>(cend());
        }
        DUAL constexpr std::reverse_iterator<const T*> crend() const
        {
            return std::reverse_iterator<const T*>(cbegin());
        }

        // norm using dot product
        DUAL constexpr auto norm_squared() const
        {
            return vecops::dot(*this, *this);
        }

        DUAL constexpr auto norm() const { return std::sqrt(norm_squared()); }

        // normalize returning new vector
        DUAL constexpr auto normalize() const
        {
            // traditional for loop version
            // const auto n = norm();
            // for (std::uint64_t ii = 0; ii < Dims; ++ii) {
            //     storage[ii] /= n;
            // }
            // return *this;

            const auto n   = norm();
            using result_t = detail::promote_t<T, decltype(n)>;
            if (n > T{0}) {
                return *this | fp::map([n](const auto& x) -> result_t {
                    return x / n;
                }) | fp::collect<vector_t<result_t, Dims>>;
            }
            return *this;
        }

        // unary negation
        DUAL constexpr auto operator-() const
        {
            // tradiational for loop version
            // vector_t<T, Dims> result;
            // for (std::uint64_t ii = 0; ii < Dims; ++ii) {
            //     result[ii] = -storage[ii];
            // }
            // return result;

            return *this | fp::map([](const auto& x) { return -x; }) |
                   fp::collect<vector_t<T, Dims>>;
        }

        // comparison operators
        DUAL constexpr bool operator==(const vector_t& other) const
        {
            for (std::uint64_t ii = 0; ii < Dims; ++ii) {
                if (storage[ii] != other[ii]) {
                    return false;
                }
            }
            return true;
        }

        DUAL constexpr bool operator!=(const vector_t& other) const
        {
            return !(*this == other);
        }

        // compound assignment (+=, -=, *=, /=)
        template <typename U, std::uint64_t OtherDims>
        DUAL constexpr auto operator+=(const vector_t<U, OtherDims>& other)
        {
            // traditional for loop version
            // for (std::uint64_t ii = 0; ii < Dims; ++ii) {
            //     storage[ii] += static_cast<T>(other[ii]);
            // }
            // return *this;

            return *this = *this + other;
        }

        template <typename U, std::uint64_t OtherDims>
        DUAL constexpr auto operator-=(const vector_t<U, OtherDims>& other)
        {
            // traditional for loop version
            // for (std::uint64_t ii = 0; ii < Dims; ++ii) {
            //     storage[ii] -= static_cast<T>(other[ii]);
            // }
            // return *this;

            return *this = *this - other;
        }

        // specialized operations for RMHD Magnetic four-vectors
        // get a view of the spatial part of the magnetic four-vector
        DUAL constexpr auto spatial_part() const
        {
            return vector_t<T, 3>{storage + 1};
        }

        template <VectorLike OtherVec>
        DUAL constexpr auto inner_product(const OtherVec& other) const
            requires(Dims == OtherVec::dimensions)
        {
            if constexpr (Dims == 4) {
                // special case for 4-vectors (spacetime vectors)
                return -storage[0] * other[0] + storage[1] * other[1] +
                       storage[2] * other[2] + storage[3] * other[3];
            }
            else {
                // general case for lower dimensions
                T result = 0;
                for (std::uint64_t ii = 0; ii < Dims; ++ii) {
                    result += storage[ii] * other[ii];
                }
                return result;
            }
        }

        DUAL constexpr auto spatial_dot(const auto& other) const
        {
            return storage[1] * other[0] + storage[2] * other[1] +
                   storage[3] * other[2];
        }

        // structured binding support
        template <size_t I>
        DUAL constexpr T& get() &
        {
            if constexpr (I >= Dims) {
                return zero_value;
            }
            else {
                return storage[I];
            }
        }

        template <size_t I>
        DUAL constexpr const T& get() const&
        {
            if constexpr (I >= Dims) {
                return zero_value;
            }
            else {
                return storage[I];
            }
        }

        template <size_t I>
        DUAL constexpr T&& get() &&
        {
            if constexpr (I >= Dims) {
                return std::move(zero_value);
            }
            else {
                return std::move(storage[I]);
            }
        }

        template <size_t I>
        friend DUAL constexpr decltype(auto) get(vector_t& v)
        {
            return v.template get<I>();
        }

        template <size_t I>
        friend DUAL constexpr decltype(auto) get(const vector_t& v)
        {
            return v.template get<I>();
        }

        template <size_t I>
        friend DUAL constexpr decltype(auto) get(vector_t&& v)
        {
            return std::move(v).template get<I>();
        }
    };

    template <typename... Args>
    vector_t(Args...) -> vector_t<std::common_type_t<Args...>, sizeof...(Args)>;

    // vector-like scalar multiplication
    template <VectorLike Vec, typename U>
    DUAL constexpr auto operator*(const Vec& vec, U scalar)
        requires(std::is_arithmetic_v<U>)
    {
        using result_t = detail::promote_t<typename Vec::value_type, U>;
        return vec | fp::map([scalar](const auto& x) -> result_t {
                   return static_cast<result_t>(x) *
                          static_cast<result_t>(scalar);
               }) |
               fp::collect<vector_t<result_t, Vec::dimensions>>;
    }

    template <VectorLike Vec, typename U>
    DUAL constexpr auto operator*(U scalar, const Vec& vec)
        requires(std::is_arithmetic_v<U>)
    {
        using result_t = detail::promote_t<typename Vec::value_type, U>;
        return vec | fp::map([scalar](const auto& x) -> result_t {
                   return static_cast<result_t>(x) *
                          static_cast<result_t>(scalar);
               }) |
               fp::collect<vector_t<result_t, Vec::dimensions>>;
    }

    // vector-like scalar division
    template <VectorLike Vec, typename U>
    DUAL constexpr auto operator/(const Vec& vec, U scalar)
        requires(std::is_arithmetic_v<U>)
    {
        using result_t = detail::promote_t<typename Vec::value_type, U>;
        return vec | fp::map([scalar](const auto& x) -> result_t {
                   return static_cast<result_t>(x) /
                          static_cast<result_t>(scalar);
               }) |
               fp::collect<vector_t<result_t, Vec::dimensions>>;
    }

    // vector-like scalar multiply assignment
    template <VectorLike Vec, typename U>
    DUAL constexpr auto& operator*=(Vec& vec, U scalar)
        requires(std::is_arithmetic_v<U>)
    {
        using result_t = detail::promote_t<typename Vec::value_type, U>;
        for (size_t i = 0; i < Vec::dimensions; ++i) {
            vec[i] *= static_cast<result_t>(scalar);
        }
        return vec;
    }

    // vector-like scalar divide assignment
    template <VectorLike Vec, typename U>
    DUAL constexpr auto& operator/=(Vec& vec, U scalar)
        requires(std::is_arithmetic_v<U>)
    {
        using result_t = detail::promote_t<typename Vec::value_type, U>;
        for (size_t i = 0; i < Vec::dimensions; ++i) {
            vec[i] /= static_cast<result_t>(scalar);
        }
        return vec;
    }

    template <VectorLike Vec1, VectorLike Vec2>
    DUAL constexpr auto operator+(const Vec1& lhs, const Vec2& rhs)
        requires(Vec1::dimensions == Vec2::dimensions)
    {
        using T                      = typename Vec1::value_type;
        using U                      = typename Vec2::value_type;
        using result_t               = detail::promote_t<T, U>;
        constexpr std::uint64_t Dims = Vec1::dimensions;

        vector_t<result_t, Dims> result;
        for (size_t i = 0; i < Dims; ++i) {
            result[i] =
                static_cast<result_t>(lhs[i]) + static_cast<result_t>(rhs[i]);
        }
        return result;
    }

    template <VectorLike Vec1, VectorLike Vec2>
    DUAL constexpr auto operator-(const Vec1& lhs, const Vec2& rhs)
        requires(Vec1::dimensions == Vec2::dimensions)
    {
        using T                      = typename Vec1::value_type;
        using U                      = typename Vec2::value_type;
        using result_t               = detail::promote_t<T, U>;
        constexpr std::uint64_t Dims = Vec1::dimensions;

        vector_t<result_t, Dims> result;
        for (size_t i = 0; i < Dims; ++i) {
            result[i] =
                static_cast<result_t>(lhs[i]) - static_cast<result_t>(rhs[i]);
        }
        return result;
    }

    // utility for pipeline-style composition with operator|
    template <typename T, std::uint64_t Dims, typename F>
    DUAL constexpr auto operator|(const vector_t<T, Dims>& v, F&& f)
        -> decltype(std::invoke(std::forward<F>(f), v))
    {
        return std::invoke(std::forward<F>(f), v);
    }

    namespace unit_vectors {

        template <std::uint64_t Dims>
        DEV constexpr auto canonical_basis(std::uint64_t i)
        {
            vector_t<std::uint64_t, Dims> basis{0};
            if (i > 0 && i <= Dims) {
                basis[i - 1] = 1;   // 1-indexed to 0-indexed
            }
            return basis;
        }

        template <std::uint64_t Dims>
        DEV constexpr std::uint64_t
        index(const vector_t<std::uint64_t, Dims>& comp)
        {
            // return the index of the first non-zero element
            for (std::uint64_t ii = 0; ii < Dims; ++ii) {
                if (comp[ii] != 0) {
                    return ii;
                }
            }
            return 0;
        }

        // generate offset vector for array dimension (0=z, 1=y, 2=x)
        template <std::uint64_t Dims>
        DEV constexpr auto array_offset(std::uint64_t array_dim)
        {
            vector_t<std::int64_t, Dims> offset{};
            if (array_dim < Dims) {
                offset[array_dim] = 1;
            }
            return offset;
        }

        // generate offset vector for logical dimension (0=x, 1=y, 2=z)
        template <std::uint64_t Dims>
        DEV constexpr auto logical_offset(std::uint64_t logical_dim)
        {
            return array_offset<Dims>(logical_dim);
        }

        template <std::uint64_t Dims>
        DEV constexpr auto ehat(std::uint64_t array_dim)
        {
            return canonical_basis<Dims>(Dims - array_dim);
        }
    }   // namespace unit_vectors

    // overload ostream operator for printing vectors
    template <typename T, std::uint64_t Dims>
    DUAL std::ostream& operator<<(std::ostream& os, const vector_t<T, Dims>& v)
    {
        os << "[";
        for (std::uint64_t i = 0; i < Dims; ++i) {
            os << v[i];
            if (i < Dims - 1) {
                os << ", ";
            }
        }
        os << "]";
        return os;
    }

    template <std::uint64_t Dims, typename T = std::uint64_t>
    constexpr auto ones()
    {
        return fp::range(Dims) | fp::map([](auto) { return T{1}; }) |
               fp::collect<vector_t<T, Dims>>;
    }

    // -------------------------------------------------------------
    // type aliases for common vector types
    // -------------------------------------------------------------
    template <typename T>
    using magnetic_four_vector_t = vector_t<T, 4>;
    template <typename T>
    using spacetime_vector_t = vector_t<T, 4>;
    // unit vector type
    template <std::uint64_t Dims>
    using unit_vector_t = vector_t<std::uint64_t, Dims>;

    template <std::integral T, std::uint64_t Dims>
    using ivec = vector_t<T, Dims>;

    // domain-specific aliases
    template <std::uint64_t Dims>
    using coordinate_t = ivec<std::int64_t, Dims>;

    template <std::uint64_t Dims>
    using shape_t = ivec<std::uint64_t, Dims>;

    template <std::uint64_t Dims>
    using compact_coord_t = ivec<std::int32_t, Dims>;

    template <std::uint64_t Dims>
    using iarray = ivec<std::int64_t, Dims>;

    template <std::uint64_t Dims>
    using uarray = ivec<std::uint64_t, Dims>;

    template <std::uint64_t Dims>
    using iarray32 = ivec<std::int32_t, Dims>;
}   // namespace simbi

// structured binding support
namespace std {
    template <typename T, std::uint64_t Dims>
    struct tuple_size<simbi::vector_t<T, Dims>>
        : integral_constant<size_t, Dims> {
    };

    template <size_t I, typename T, std::uint64_t Dims>
    struct tuple_element<I, simbi::vector_t<T, Dims>> {
        using type = T;
    };

    template <size_t I, typename T, std::uint64_t Dims>
    DUAL constexpr T& get(simbi::vector_t<T, Dims>& v)
    {
        static_assert(I < Dims, "index out of bounds");
        return v[I];
    }

    template <size_t I, typename T, std::uint64_t Dims>
    DUAL constexpr const T& get(const simbi::vector_t<T, Dims>& v)
    {
        static_assert(I < Dims, "index out of bounds");
        return v[I];
    }
}   // namespace std
#endif
