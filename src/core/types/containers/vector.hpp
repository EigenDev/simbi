#ifndef VECTOR_HPP
#define VECTOR_HPP

#include "array.hpp"
#include "build_options.hpp"
#include "core/functional/fp.hpp"
#include "core/types/monad/maybe.hpp"
#include "core/types/utility/enums.hpp"
#include "util/tools/algorithms.hpp"
#include <array>
#include <cmath>
#include <functional>
#include <type_traits>

namespace simbi {
    // vector types remain as enums for strong typing
    enum class VectorType {
        SPATIAL,
        MAGNETIC,
        MAGNETIC_FOUR,
        SPACETIME,
        GENERAL,
    };

    // vector traits
    template <VectorType Type>
    struct vector_traits {
        static constexpr size_t dimensions = 3;   // default
    };

    template <>
    struct vector_traits<VectorType::SPACETIME> {
        static constexpr size_t dimensions    = 4;
        static constexpr bool is_relativistic = true;
    };

    template <>
    struct vector_traits<VectorType::MAGNETIC> {
        static constexpr size_t dimensions      = 3;
        static constexpr bool requires_div_free = true;
    };

    template <>
    struct vector_traits<VectorType::MAGNETIC_FOUR> {
        static constexpr size_t dimensions    = 4;
        static constexpr bool is_relativistic = true;
        static constexpr bool is_magnetic     = true;
    };

    template <VectorType T>
    concept Magnetic = T == VectorType::MAGNETIC;

    template <VectorType T>
    concept Relativistic = vector_traits<T>::is_relativistic;

    template <VectorType T>
    concept MagneticFour = T == VectorType::MAGNETIC_FOUR;

    template <VectorType T>
    concept Spatial = T == VectorType::SPATIAL || T == VectorType::MAGNETIC;

    template <VectorType T>
    concept FourVector =
        T == VectorType::SPACETIME || T == VectorType::MAGNETIC_FOUR;

    // forward declarations
    template <typename T, size_type Dims, VectorType Type>
    class Vector;

    template <typename T, size_type Dims, VectorType Type>
    class VectorView;

    namespace detail {
        // type promotion helper
        template <typename T, typename U>
        using promote_t = std::common_type_t<T, U>;

        template <VectorType T, VectorType U>
        constexpr auto promote_vector_t = T == U ? T : VectorType::GENERAL;
    }   // namespace detail

    namespace vecops {
        // functitonal-style zip function that has nothing
        // to do with vectors

        // dot product
        template <typename Vec1, typename Vec2>
        DUAL constexpr auto dot(const Vec1& a, const Vec2& b)
        {
            using T  = decltype(a[0] * b[0]);
            T result = T{0};
            for (size_type ii = 0; ii < Vec1::dimensions; ++ii) {
                result += a[ii] * b[ii];
            }

            return result;
        }

        // norm
        template <typename T, size_type Dims, VectorType Type>
        DUAL constexpr auto norm(const Vector<T, Dims, Type>& vec)
        {
            return std::sqrt(dot(vec, vec));
        }

        // normalize
        template <typename T, size_type Dims, VectorType Type>
        DUAL constexpr auto normalize(const Vector<T, Dims, Type>& vec)
        {
            const auto n = norm(vec);
            if (n > T{0}) {
                return vec * (1.0 / n);
            }
            return vec;
        }

        // cross product
        template <typename Vec1, typename Vec2>
        DUAL constexpr auto cross(const Vec1& a, const Vec2& b)
            requires(Vec1::dimensions == 3 && Vec2::dimensions == 3)
        {
            using T = decltype(a[0] * b[0]);
            return Vector<T, 3, VectorType::GENERAL>(
                a[1] * b[2] - a[2] * b[1],
                a[2] * b[0] - a[0] * b[2],
                a[0] * b[1] - b[0] * a[1]
            );
        }

        // cross product magnitude for Dim = 2
        template <typename Vec1, typename Vec2>
        DUAL constexpr auto cross(const Vec1& a, const Vec2& b)
            requires(Vec1::dimensions == 2 && Vec2::dimensions == 2)
        {
            return a[0] * b[1] - a[1] * b[0];
        }

        // specialized RMHD operations
        template <
            template <typename, size_type, VectorType> typename Vec,
            typename T,
            size_type Dims,
            VectorType Type>
        DUAL constexpr auto as_fourvec(
            const Vec<T, Dims, Type>& bfield,
            const auto& vel,
            const auto lorentz
        )
            requires Magnetic<Type>
        {
            const auto vdB = dot(vel, bfield);
            const auto b0  = lorentz * vdB;
            const auto bs  = bfield * (1.0 / lorentz) + vel * lorentz * vdB;

            return Vector<T, 4, VectorType::MAGNETIC_FOUR>{
              b0,
              bs[0],
              bs[1],
              bs[2]
            };
        }

        // helpers to rotate vectors by some angle
        template <typename Vec, typename T>
        DEV static constexpr auto rotate_2D(const Vec& vec, const T& angle)
        {
            return Vector<T, 2, VectorType::SPATIAL>{
              vec[0] * std::cos(angle) - vec[1] * std::sin(angle),
              vec[0] * std::sin(angle) + vec[1] * std::cos(angle)
            };
        }

        template <typename Vec, typename T>
        DEV static constexpr auto rotate_3D(const Vec& vec, const T& angle)
        {
            return Vec{
              vec[0] * std::cos(angle) - vec[1] * std::sin(angle),
              vec[0] * std::sin(angle) + vec[1] * std::cos(angle),
              vec[2]
            };
        }

        // general rotation function that checks the dimension at compile-time
        template <typename Vec, typename T>
        DEV static constexpr auto rotate(const Vec& vec, const T& angle)
        {
            if constexpr (Vec::dimensions == 2) {
                return rotate_2D(vec, angle);
            }
            else {
                return rotate_3D(vec, angle);
            }
        }

        template <typename Vec>
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

        template <typename Vec>
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

        template <typename Vec>
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

        template <typename Vec>
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

        template <typename Vec>
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

        template <typename Vec>
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
        template <typename Vec>
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
        template <typename Vec>
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
    template <typename T, size_type Dims, VectorType Type = VectorType::GENERAL>
    class Vector
    {
      private:
        // direct storage - no indirection
        array_t<T, Dims> storage_{};
        // Static zero value for out-of-bounds access in
        // structured binding access
        static inline T zero_value_{};

      public:
        // type definitions for type traits and functional interfaces
        using value_type      = T;
        using reference       = T&;
        using const_reference = const T&;
        using iterator        = typename array_t<T, Dims>::iterator;
        using const_iterator  = typename array_t<T, Dims>::const_iterator;

        static constexpr size_type dimensions = Dims;
        static constexpr VectorType vec_type  = Type;

        // default constructor creates zero vector
        constexpr Vector() = default;

        // variadic constructor with perfect forwarding
        template <typename... Args>
        DUAL constexpr explicit Vector(Args&&... args)
            requires(
                sizeof...(Args) == Dims &&
                (std::is_convertible_v<Args, T> && ...)
            )
            : storage_{static_cast<T>(std::forward<Args>(args))...}
        {
        }

        // explicit construction from array_t
        DUAL constexpr explicit Vector(const array_t<T, Dims>& arr)
            : storage_(arr)
        {
        }

        // construction from raw pointer
        DUAL constexpr explicit Vector(const T* data)
        {
            algorithms::copy_n(data, Dims, storage_.begin());
        }

        // element access - direct, no indirection
        DUAL constexpr reference operator[](size_type idx)
        {
            return storage_[idx];
        }

        DUAL constexpr const_reference operator[](size_type idx) const
        {
            return storage_[idx];
        }

        // safe access with bounds checking
        DUAL constexpr Maybe<reference> at(size_type idx)
        {
            if (idx < Dims) {
                return Maybe<reference>(storage_[idx]);
            }
            return Nothing;
        }

        DUAL constexpr Maybe<const_reference> at(size_type idx) const
        {
            if (idx < Dims) {
                return Maybe<const_reference>(storage_[idx]);
            }
            return Nothing;
        }

        // data access for algorithms
        DUAL constexpr T* data() { return storage_.data(); }
        DUAL constexpr const T* data() const { return storage_.data(); }

        // size and capacity
        DUAL constexpr size_type size() const { return Dims; }

        // iterators for standard algorithms
        DUAL constexpr iterator begin() { return storage_.begin(); }
        DUAL constexpr iterator end() { return storage_.end(); }
        DUAL constexpr const_iterator begin() const { return storage_.begin(); }
        DUAL constexpr const_iterator end() const { return storage_.end(); }
        DUAL constexpr const_iterator cbegin() const
        {
            return storage_.cbegin();
        }
        DUAL constexpr const_iterator cend() const { return storage_.cend(); }
        DUAL constexpr const_iterator rbegin() const
        {
            return storage_.rbegin();
        }
        DUAL constexpr const_iterator rend() const { return storage_.rend(); }

        // functional operations

        // map - apply function to each element, returning new vector
        template <typename F>
        DUAL constexpr auto map(F&& f) const
        {
            using result_t = std::invoke_result_t<F, T>;
            Vector<result_t, Dims, Type> result;

            for (size_type i = 0; i < Dims; ++i) {
                result[i] = std::invoke(std::forward<F>(f), storage_[i]);
            }

            return result;
        }

        // vector operations built on functional primitives
        // so that I can practice functional programming

        // dot product using fold
        template <typename U, VectorType OtherType>
        DUAL constexpr detail::promote_t<T, U>
        dot(const Vector<U, Dims, OtherType>& other) const
        {
            using result_t = detail::promote_t<T, U>;
            result_t sum{0};
            for (size_type i = 0; i < Dims; ++i) {
                sum += storage_[i] * other[i];
            }

            return sum;
        }

        // allow dot product of vectors with different dims,
        // but we only use the first Dims elements
        template <typename U, size_type OtherDims, VectorType OtherType>
        DUAL constexpr detail::promote_t<T, U>
        dot(const Vector<U, OtherDims, OtherType>& other) const
        {
            using result_t = detail::promote_t<T, U>;
            result_t sum{0};
            for (size_type i = 0; i < Dims; ++i) {
                sum += storage_[i] * other[i];
            }
            return sum;
        }

        // norm using dot product
        DUAL constexpr auto norm_squared() const { return dot(*this); }

        DUAL constexpr auto norm() const { return std::sqrt(norm_squared()); }

        // normalize returning new vector
        DUAL constexpr auto normalize() const
        {
            const auto n = norm();
            if (n > T{0}) {
                return map([n](const T& x) { return x / n; });
            }
            return *this;
        }

        // unary negation
        DUAL constexpr auto operator-() const
        {
            return map([](const T& x) { return -x; });
        }

        // arithmetic operations

        // element-wise addition
        template <typename U, size_type OtherDims, VectorType OtherType>
        DUAL constexpr Vector<
            detail::promote_t<T, U>,
            Dims,
            detail::promote_vector_t<Type, OtherType>>
        operator+(const Vector<U, OtherDims, OtherType>& other) const
        {
            using result_type = detail::promote_t<T, U>;
            constexpr auto result_vectype =
                Type == OtherType ? Type : VectorType::GENERAL;
            Vector<result_type, Dims, result_vectype> result;
            for (size_type ii = 0; ii < Dims; ++ii) {
                result[ii] = (*this)[ii] + other[ii];
            }
            return result;
        }

        // element-wise subtraction
        template <typename U, size_type OtherDims, VectorType OtherType>
        DUAL constexpr Vector<
            detail::promote_t<T, U>,
            Dims,
            detail::promote_vector_t<Type, OtherType>>
        operator-(const Vector<U, OtherDims, OtherType>& other) const
        {
            using result_type = detail::promote_t<T, U>;
            constexpr auto result_vectype =
                Type == OtherType ? Type : VectorType::GENERAL;
            Vector<result_type, Dims, result_vectype> result;
            for (size_type ii = 0; ii < Dims; ++ii) {
                result[ii] = (*this)[ii] - other[ii];
            }
            return result;
        }

        // scalar multiplication
        template <typename U>
        DUAL constexpr auto operator*(U scalar) const
        {
            using result_t = detail::promote_t<T, U>;
            Vector<result_t, Dims, Type> result;
            for (size_type i = 0; i < Dims; ++i) {
                result[i] = static_cast<result_t>(storage_[i]) * scalar;
            }
            return result;
        }

        // scalar division
        template <typename U>
        DUAL constexpr auto operator/(U scalar) const
        {
            using result_t = detail::promote_t<T, U>;
            Vector<result_t, Dims, Type> result;
            for (size_type i = 0; i < Dims; ++i) {
                result[i] = static_cast<result_t>(storage_[i]) / scalar;
            }
            return result;
        }

        // create a view to this vector
        DUAL constexpr VectorView<T, Dims, Type> view()
        {
            return VectorView<T, Dims, Type>(storage_.data());
        }

        DUAL constexpr VectorView<const T, Dims, Type> view() const
        {
            return VectorView<const T, Dims, Type>(storage_.data());
        }

        // specialized operations for RMHD Magnetic four-vectors
        // get a view of the spatial part of the magnetic four-vector
        DUAL constexpr auto spatial_part() const
            requires MagneticFour<Type>
        {
            return Vector<T, 3, VectorType::MAGNETIC>{storage_.data() + 1};
        }

        DUAL constexpr auto inner_product(const auto& other) const
            requires MagneticFour<Type>
        {
            return -storage_[0] * other[0] + storage_[1] * other[1] +
                   storage_[2] * other[2] + storage_[3] * other[3];
        }

        DUAL constexpr auto spatial_dot(const auto& other) const
            requires MagneticFour<Type>
        {
            return storage_[1] * other[1] + storage_[2] * other[2] +
                   storage_[3] * other[3];
        }

        // conversion to other vector types
        template <VectorType NewType>
        DUAL constexpr auto as_type() const
        {
            return Vector<T, Dims, NewType>(storage_);
        }

        // structured binding support
        template <size_t I>
        DUAL constexpr T& get() &
        {
            if constexpr (I >= Dims) {
                return zero_value_;
            }
            else {
                return storage_[I];
            }
        }

        template <size_t I>
        DUAL constexpr const T& get() const&
        {
            if constexpr (I >= Dims) {
                return zero_value_;
            }
            else {
                return storage_[I];
            }
        }
    };

    // scalar multiplication from the left
    template <typename U, typename T, size_type Dims, VectorType Type>
    DUAL constexpr auto operator*(U scalar, const Vector<T, Dims, Type>& vec)
    {
        return vec * scalar;
    }

    // -------------------------------------------------------------
    // VectorView: non-owning view of vector data
    // -------------------------------------------------------------
    template <typename T, size_type Dims, VectorType Type = VectorType::GENERAL>
    class VectorView
    {
      private:
        T* data_;   // pointer to external data

      public:
        // type definitions
        using value_type      = std::remove_const_t<T>;
        using reference       = T&;
        using const_reference = const T&;
        using iterator        = T*;
        using const_iterator  = const T*;

        static constexpr size_type dimensions = Dims;
        static constexpr VectorType vec_type  = Type;

        // constructor from pointer
        DUAL constexpr explicit VectorView(T* data = nullptr) : data_(data) {}

        // element access
        DUAL constexpr reference operator[](size_type idx)
        {
            return data_[idx];
        }
        DUAL constexpr const_reference operator[](size_type idx) const
        {
            return data_[idx];
        }

        // bounds-checked access returning Maybe
        DUAL constexpr Maybe<reference> at(size_type idx)
        {
            if (idx < Dims) {
                return Maybe<reference>(data_[idx]);
            }
            return Nothing;
        }

        DUAL constexpr Maybe<const_reference> at(size_type idx) const
        {
            if (idx < Dims) {
                return Maybe<const_reference>(data_[idx]);
            }
            return Nothing;
        }

        // raw data access
        DUAL constexpr T* data() { return data_; }
        DUAL constexpr const T* data() const { return data_; }
        DUAL constexpr size_type size() const { return Dims; }

        // convert to owning Vector
        DUAL constexpr Vector<std::remove_const_t<T>, Dims, Type>
        to_vector() const
        {
            Vector<std::remove_const_t<T>, Dims, Type> result;
            for (size_type i = 0; i < Dims; ++i) {
                result[i] = data_[i];
            }
            return result;
        }

        template <typename U, VectorType OtherType>
        DUAL constexpr auto dot(const Vector<U, Dims, OtherType>& other) const
        {
            return to_vector().dot(other);
        }

        template <typename U, VectorType OtherType>
        DUAL constexpr auto
        dot(const VectorView<U, Dims, OtherType>& other) const
        {
            return to_vector().dot(other.to_vector());
        }

        DUAL constexpr auto norm() const { return to_vector().norm(); }

        DUAL constexpr auto normalize() const
        {
            return to_vector().normalize();
        }

        // fix pointer after device synchronization
        DUAL constexpr void fix_device_pointer(T* new_data)
        {
            data_ = new_data;
        }

        // increment vector by another vector
        template <typename Vec>
        DUAL constexpr auto& operator+=(const Vec& other)
            requires(!std::is_const_v<T>)
        {
            for (size_type i = 0; i < Dims; ++i) {
                data_[i] += static_cast<T>(other[i]);
            }
            return *this;
        }

        // iterator logic for stl algorithms
        DUAL constexpr iterator begin() { return data_; }
        DUAL constexpr iterator end() { return data_ + Dims; }
        DUAL constexpr const_iterator begin() const { return data_; }
        DUAL constexpr const_iterator end() const { return data_ + Dims; }
        DUAL constexpr const_iterator cbegin() const { return data_; }
        DUAL constexpr const_iterator cend() const { return data_ + Dims; }
        DUAL constexpr const_iterator rbegin() const { return data_ + Dims; }
        DUAL constexpr const_iterator rend() const { return data_; }
    };

    //---------------------------------------------------------------
    // ConstVectorView: const version of VectorView
    // ---------------------------------------------------------------
    template <typename T, size_type Dims, VectorType Type = VectorType::GENERAL>
    class ConstVectorView
    {
      private:
        const T* data_;   // pointer to external data

      public:
        // type definitions
        using value_type      = T;
        using reference       = const T&;
        using const_reference = const T&;
        using iterator        = const T*;
        using const_iterator  = const T*;

        static constexpr size_type dimensions = Dims;
        static constexpr VectorType vec_type  = Type;

        // constructor from pointer
        DUAL constexpr explicit ConstVectorView(const T* data) : data_(data) {}

        // brace-enclosed initializer list constructor
        DUAL constexpr explicit ConstVectorView(std::initializer_list<T> data)
            : data_(data.begin())
        {
        }

        // element access
        DUAL constexpr reference operator[](size_type idx) const
        {
            return data_[idx];
        }

        // bounds-checked access returning Maybe
        DUAL constexpr Maybe<reference> at(size_type idx) const
        {
            if (idx < Dims) {
                return Maybe<reference>(data_[idx]);
            }
            return Nothing;
        }

        // raw data access
        DUAL constexpr const T* data() const { return data_; }
        DUAL constexpr size_type size() const { return Dims; }

        // convert to owning Vector
        DUAL constexpr Vector<T, Dims, Type> to_vector() const
        {
            Vector<T, Dims, Type> result;
            for (size_type i = 0; i < Dims; ++i) {
                result[i] = data_[i];
            }
            return result;
        }

        // arithmetic operations
        // scalar multiplication
        template <typename U>
        DUAL constexpr auto operator*(U scalar) const
        {
            using result_t = detail::promote_t<T, U>;
            Vector<result_t, Dims, Type> result;
            for (size_type i = 0; i < Dims; ++i) {
                result[i] = static_cast<result_t>(data_[i]) * scalar;
            }
            return result;
        }

        // scalar division
        template <typename U>
        DUAL constexpr auto operator/(U scalar) const
        {
            using result_t = detail::promote_t<T, U>;
            Vector<result_t, Dims, Type> result;
            for (size_type i = 0; i < Dims; ++i) {
                result[i] = static_cast<result_t>(data_[i]) / scalar;
            }
            return result;
        }

        // addition
        template <typename U, VectorType OtherType>
        DUAL constexpr auto operator+(const Vector<U, Dims, OtherType>& other
        ) const
        {
            using result_t = detail::promote_t<T, U>;
            constexpr auto result_type =
                detail::promote_vector_t<Type, OtherType>;
            Vector<result_t, Dims, result_type> result;
            for (size_type i = 0; i < Dims; ++i) {
                result[i] = static_cast<result_t>(data_[i]) +
                            static_cast<result_t>(other[i]);
            }
            return result;
        }

        // subtraction
        template <typename U, VectorType OtherType>
        DUAL constexpr auto operator-(const Vector<U, Dims, OtherType>& other
        ) const
        {
            using result_t = detail::promote_t<T, U>;
            constexpr auto result_type =
                detail::promote_vector_t<Type, OtherType>;

            Vector<result_t, Dims, result_type> result;
            for (size_type i = 0; i < Dims; ++i) {
                result[i] = static_cast<result_t>(data_[i]) -
                            static_cast<result_t>(other[i]);
            }
            return result;
        }

        // norm
        DUAL constexpr auto norm() const { return to_vector().norm(); }

        // iterator logic for stl algorithms
        DUAL constexpr iterator begin() { return data_; }
        DUAL constexpr iterator end() { return data_ + Dims; }
        DUAL constexpr const_iterator begin() const { return data_; }
        DUAL constexpr const_iterator end() const { return data_ + Dims; }
        DUAL constexpr const_iterator cbegin() const { return data_; }
        DUAL constexpr const_iterator cend() const { return data_ + Dims; }
        DUAL constexpr const_iterator rbegin() const { return data_ + Dims; }
        DUAL constexpr const_iterator rend() const { return data_; }
    };

    // -------------------------------------------------------------
    // type aliases for common vector types
    // -------------------------------------------------------------
    template <typename T, size_type Dims>
    using spatial_vector_t = Vector<T, Dims, VectorType::SPATIAL>;
    template <typename T, size_type Dims>
    using magnetic_vector_t = Vector<T, Dims, VectorType::MAGNETIC>;
    template <typename T, size_type Dims>
    using general_vector_t = Vector<T, Dims, VectorType::GENERAL>;

    template <typename T>
    using magnetic_four_vector_t = Vector<T, 4, VectorType::MAGNETIC_FOUR>;
    template <typename T>
    using spacetime_vector_t = Vector<T, 4, VectorType::SPACETIME>;
    // unit vector type
    template <size_type Dims>
    using unit_vector_t = Vector<luint, Dims, VectorType::SPATIAL>;

    // view aliases
    template <typename T, size_type Dims>
    using spatial_vector_view_t = VectorView<T, Dims, VectorType::SPATIAL>;
    template <typename T, size_type Dims>
    using magnetic_vector_view_t = VectorView<T, Dims, VectorType::MAGNETIC>;
    template <typename T, size_type Dims>
    using general_vector_view_t = VectorView<T, Dims, VectorType::GENERAL>;

    template <typename T>
    using spacetime_vector_view_t = VectorView<T, 4, VectorType::SPACETIME>;

    // const aliases
    template <typename T, size_type Dims>
    using const_spatial_vector_view_t =
        ConstVectorView<T, Dims, VectorType::SPATIAL>;
    template <typename T, size_type Dims>
    using const_magnetic_vector_view_t =
        ConstVectorView<T, Dims, VectorType::MAGNETIC>;
    template <typename T>
    using const_magnetic_four_vector_view_t =
        ConstVectorView<T, 4, VectorType::MAGNETIC_FOUR>;

    // zero vector view as a global singleton
    template <typename T, size_type Dims, VectorType Type>
    class ZeroVectorView
    {
      private:
        static inline array_t<T, Dims> zero_storage_{};

      public:
        DUAL constexpr VectorView<const T, Dims, Type> view() const
        {
            return VectorView<const T, Dims, Type>(zero_storage_.data());
        }
    };

    class ZeroMagneticVectorView : public const_magnetic_vector_view_t<real, 3>
    {
      private:
        const real zero_storage[3] = {0.0, 0.0, 0.0};

      public:
        DUAL ZeroMagneticVectorView()
            : const_magnetic_vector_view_t<real, 3>(zero_storage)
        {
        }
    };

    class ZeroMagneticFourVectorView
        : public const_magnetic_four_vector_view_t<real>
    {
      private:
        const real zero_storage[4] = {0.0, 0.0, 0.0, 0.0};

      public:
        DUAL ZeroMagneticFourVectorView()
            : const_magnetic_four_vector_view_t<real>(zero_storage)
        {
        }
    };

    // utility for pipeline-style composition with operator|
    template <typename T, size_type Dims, VectorType Type, typename F>
    DUAL constexpr auto operator|(const Vector<T, Dims, Type>& v, F&& f)
        -> decltype(std::invoke(std::forward<F>(f), v))
    {
        return std::invoke(std::forward<F>(f), v);
    }

    namespace unit_vectors {
        template <size_type Dims>
        DUAL constexpr auto get(size_type i)
        {
            if constexpr (Dims == 1) {
                // Return a vector directly instead of referencing static data
                return (i == 1) ? Vector<luint, 1, VectorType::SPATIAL>{1}
                                : Vector<luint, 1, VectorType::SPATIAL>{0};
            }
            else if constexpr (Dims == 2) {
                if (i == 1) {
                    return Vector<luint, 2, VectorType::SPATIAL>{1, 0};
                }
                else if (i == 2) {
                    return Vector<luint, 2, VectorType::SPATIAL>{0, 1};
                }
                else {
                    return Vector<luint, 2, VectorType::SPATIAL>{0, 0};
                }
            }
            else {
                if (i == 1) {
                    return Vector<luint, 3, VectorType::SPATIAL>{1, 0, 0};
                }
                else if (i == 2) {
                    return Vector<luint, 3, VectorType::SPATIAL>{0, 1, 0};
                }
                else if (i == 3) {
                    return Vector<luint, 3, VectorType::SPATIAL>{0, 0, 1};
                }
                else {
                    return Vector<luint, 3, VectorType::SPATIAL>{0, 0, 0};
                }
            }
        }
    }   // namespace unit_vectors

}   // namespace simbi

// structured binding support
namespace std {
    template <typename T, size_type Dims, simbi::VectorType Type>
    struct tuple_size<simbi::Vector<T, Dims, Type>>
        : integral_constant<size_t, Dims> {
    };

    template <size_t I, typename T, size_type Dims, simbi::VectorType Type>
    struct tuple_element<I, simbi::Vector<T, Dims, Type>> {
        using type = T;
    };

    template <size_t I, typename T, size_type Dims, simbi::VectorType Type>
    DUAL constexpr T& get(simbi::Vector<T, Dims, Type>& v)
    {
        static_assert(I < Dims, "index out of bounds");
        return v[I];
    }

    template <size_t I, typename T, size_type Dims, simbi::VectorType Type>
    DUAL constexpr const T& get(const simbi::Vector<T, Dims, Type>& v)
    {
        static_assert(I < Dims, "index out of bounds");
        return v[I];
    }
}   // namespace std
#endif
