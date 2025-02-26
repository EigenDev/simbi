/**
 *  *=============================================================================
 *  *           SIMBI - Special Relativistic Magnetohydrodynamics Code
 *  *=============================================================================
 *  *
 *  * @file            vector.hpp
 *  * @brief           custom vector types for regime-specific operations
 *  * @details
 *  *
 *  * @version         0.8.0
 *  * @date            2025-02-26
 *  * @author          Marcus DuPont
 *  * @email           marcus.dupont@princeton.edu
 *  *
 *  *==============================================================================
 *  * @build           Requirements & Dependencies
 *  *==============================================================================
 *  * @requires        C++20
 *  * @depends         CUDA >= 11.0, HDF5 >= 1.12, OpenMP >= 4.5
 *  * @platform        Linux, MacOS
 *  * @parallel        GPU (CUDA, HIP), CPU (OpenMP)
 *  *
 *  *==============================================================================
 *  * @documentation   Reference & Notes
 *  *==============================================================================
 *  * @usage
 *  * @note
 *  * @warning
 *  * @todo
 *  * @bug
 *  * @performance
 *  *
 *  *==============================================================================
 *  * @testing        Quality Assurance
 *  *==============================================================================
 *  * @test
 *  * @benchmark
 *  * @validation
 *  *
 *  *==============================================================================
 *  * @history        Version History
 *  *==============================================================================
 *  * 2025-02-26      v0.8.0      Initial implementation
 *  *
 *  *==============================================================================
 *  * @copyright (C) 2025 Marcus DuPont. All rights reserved.
 *  *==============================================================================
 */
#ifndef VECTOR_HPP
#define VECTOR_HPP

#include "build_options.hpp"   // for real, lint, luint
#include <type_traits>

namespace simbi {
    enum class VectorType {
        SPATIAL,         // 3D spatial vectors
        MAGNETIC,        // Magnetic field vectors
        MAGNETIC_FOUR,   // 4D magnetic four-vectors
        SPACETIME,       // 4D spacetime vectors
        GENERAL,         // Generic vectors
    };

    // Vector traits
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

    // type promotion helper
    template <typename T, typename U>
    using promote_t = std::common_type_t<T, U>;

    template <VectorType A, VectorType B>
    struct promote_vector_type {
        static constexpr VectorType value =
            (A == VectorType::MAGNETIC_FOUR || B == VectorType::MAGNETIC_FOUR)
                ? VectorType::MAGNETIC_FOUR
            : (A == VectorType::SPACETIME || B == VectorType::SPACETIME)
                ? VectorType::SPACETIME
            : (A == VectorType::MAGNETIC || B == VectorType::MAGNETIC)
                ? VectorType::MAGNETIC
                : VectorType::SPATIAL;
    };

    // forward declarations
    template <typename T, size_type Dims, VectorType Type>
    class Vector;

    template <typename T, size_type Dims, VectorType Type>
    class VectorView;

    //=====================================================
    // Base Storage Type for Vectors
    //===================================================
    template <typename T, size_type Dims>
    class BaseStorage
    {
      protected:
        T* data_;
        bool owns_memory_;

        DUAL constexpr BaseStorage(T* data, bool owns = false)
            : data_{data}, owns_memory_{owns}
        {
        }

      public:
        using value_type                      = T;
        static constexpr size_type dimensions = Dims;

        DUAL constexpr size_type size() const { return Dims; }

        DUAL constexpr T* data() { return data_; }

        DUAL constexpr const T* data() const { return data_; }

        DUAL constexpr bool owns_memory() const { return owns_memory_; }

        DUAL constexpr T& operator[](size_type ii) { return data_[ii]; }

        DUAL constexpr const T& operator[](size_type ii) const
        {
            return data_[ii];
        }

        // ostream operator
        friend std::ostream& operator<<(std::ostream& os, const BaseStorage& v)
        {
            os << "(";
            for (size_type ii = 0; ii < Dims; ++ii) {
                os << v[ii] << (ii != Dims - 1 ? " " : "");
            }
            os << ")";
            return os;
        }

        // implicitly convert to underlying data
        DUAL constexpr operator T*() { return data_; }
    };

    // Base vector type for generic vector operations
    template <typename T, size_type Dims, VectorType Type>
    class VectorOps : public BaseStorage<T, Dims>
    {
      private:
        using base = BaseStorage<T, Dims>;

      protected:
        static constexpr auto zero = T{0};

      public:
        using base::base;
        using typename base::value_type;
        static constexpr VectorType vec_type = Type;

        // Basic operations
        DUAL constexpr auto norm() const
        {
            auto mag = zero;
            for (size_type ii = 0; ii < Dims; ++ii) {
                mag += (*this)[ii] * (*this)[ii];
            }
            return std::sqrt(mag);
        }

        DUAL constexpr auto norm_squared() const
        {
            auto mag = zero;
            for (size_type ii = 0; ii < Dims; ++ii) {
                mag += (*this)[ii] * (*this)[ii];
            }
            return mag;
        }

        DUAL constexpr auto normalize() const
        {
            auto mag = norm();
            if (mag > zero) {
                for (size_type ii = 0; ii < Dims; ++ii) {
                    (*this)[ii] /= mag;
                }
            }
            return *this;
        }

        DUAL constexpr auto unit() const
        {
            auto mag = norm();
            if (mag > zero) {
                for (size_type ii = 0; ii < Dims; ++ii) {
                    (*this)[ii] /= mag;
                }
            }
            return *this;
        }

        // vector-to-vector operations w/ type promotions
        template <typename U, size_type OtherDims, VectorType OtherType>
        DUAL constexpr auto dot(const VectorOps<U, OtherDims, OtherType>& other
        ) const
        {
            using result_type = promote_t<T, U>;
            result_type sum{0};
            for (size_type ii = 0; ii < Dims; ++ii) {
                sum += (*this)[ii] * other[ii];
            }
            return sum;
        }

        // binary operations w/ type promotions
        template <typename U, size_type OtherDims, VectorType OtherType>
        DUAL constexpr auto
        operator+(const VectorOps<U, OtherDims, OtherType>& other) const
        {
            using result_type = promote_t<T, U>;
            constexpr auto result_vectype =
                Type == OtherType ? Type : VectorType::GENERAL;
            Vector<result_type, Dims, result_vectype> result;
            for (size_type ii = 0; ii < Dims; ++ii) {
                result[ii] = (*this)[ii] + other[ii];
            }
            return result;
        }

        template <typename U, size_type OtherDims, VectorType OtherType>
        DUAL constexpr auto
        operator-(const VectorOps<U, OtherDims, OtherType>& other) const
        {
            using result_type = promote_t<T, U>;
            constexpr auto result_vectype =
                Type == OtherType ? Type : VectorType::GENERAL;
            Vector<result_type, Dims, result_vectype> result;
            for (size_type ii = 0; ii < Dims; ++ii) {
                result[ii] = (*this)[ii] - other[ii];
            }
            return result;
        }

        template <typename U, VectorType OtherType>
        DUAL constexpr auto&
        operator+=(const VectorOps<U, Dims, OtherType>& other)
            requires(!std::is_const_v<T>)
        {
            for (size_type i = 0; i < Dims; ++i) {
                (*this)[i] += static_cast<T>(other[i]);
            }
            return *this;
        }

        template <typename U, VectorType OtherType>
        DUAL constexpr auto&
        operator-=(const VectorOps<U, Dims, OtherType>& other)
            requires(!std::is_const_v<T>)
        {
            for (size_type i = 0; i < Dims; ++i) {
                (*this)[i] -= static_cast<T>(other[i]);
            }
            return *this;
        }

        template <typename U>
        DUAL constexpr auto operator*(const U scalar) const
        {
            using result_type = promote_t<T, U>;
            Vector<result_type, Dims, Type> result;
            for (size_type ii = 0; ii < Dims; ++ii) {
                result[ii] = (*this)[ii] * scalar;
            }
            return result;
        }

        template <typename U>
        DUAL constexpr auto operator/(const U scalar) const
        {
            using result_type = promote_t<T, U>;
            Vector<result_type, Dims, Type> result;
            for (size_type ii = 0; ii < Dims; ++ii) {
                result[ii] = (*this)[ii] / scalar;
            }
            return result;
        }

        // unary operations
        DUAL constexpr auto operator-() const
        {
            Vector<T, Dims, Type> result;
            for (size_type ii = 0; ii < Dims; ++ii) {
                result[ii] = -(*this)[ii];
            }
            return result;
        }

        // cross product
        template <
            template <typename, size_type, VectorType> class Vec_t,
            typename U,
            VectorType OtherType>
        DUAL constexpr auto cross(const Vec_t<U, Dims, OtherType>& other) const
            requires(Dims == 3)
        {
            using result_type = promote_t<T, U>;
            constexpr auto result_vectype =
                Type == OtherType ? Type : VectorType::GENERAL;
            return Vector<result_type, Dims, result_vectype>{
              (*this)[1] * other[2] - (*this)[2] * other[1],
              (*this)[2] * other[0] - (*this)[0] * other[2],
              (*this)[0] * other[1] - (*this)[1] * other[0]
            };
        }

        // component-specific cross product
        template <
            template <typename, size_type, VectorType> class Vec_t,
            typename U,
            VectorType OtherType>
        DUAL constexpr auto
        cross(size_type ii, const Vec_t<U, Dims, OtherType>& other) const
            requires(Dims == 3)
        {
            if (ii == 0) {
                return (*this)[1] * other[2] - (*this)[2] * other[1];
            }
            else if (ii == 1) {
                return (*this)[2] * other[0] - (*this)[0] * other[2];
            }
            else {
                return (*this)[0] * other[1] - (*this)[1] * other[0];
            }
        }

        // Implicit conversion to general vector type
        DUAL constexpr operator Vector<T, Dims, VectorType::GENERAL>() const
        {
            Vector<T, Dims, VectorType::GENERAL> result;
            for (size_type ii = 0; ii < Dims; ++ii) {
                result[ii] = (*this)[ii];
            }
            return result;
        }

        // Type conversion operations
        DUAL constexpr auto as_magnetic() const
        {
            return Vector<T, 3, VectorType::MAGNETIC>{this->data_};
        }

        DUAL constexpr auto as_spatial() const
        {
            return Vector<T, 3, VectorType::SPATIAL>{this->data_};
        }

        DUAL constexpr auto to_general() const
        {
            return Vector<real, 3, VectorType::GENERAL>(*this);
        }

        DUAL constexpr auto as_spatial()
            requires FourVector<Type>
        {
            return Vector<T, 3, VectorType::SPATIAL>{this->data_};
        }
    };

    // Main vector class, memory owned
    template <typename T, size_type Dims, VectorType Type>
    class Vector : public VectorOps<T, Dims, Type>
    {
      private:
        alignas(alignof(T)) T storage_[Dims];
        static constexpr bool owned = true;
        // Static zero value for out-of-bounds access in
        // structured binding access
        static inline T zero_value_{};

      public:
        DUAL constexpr Vector()
            : VectorOps<T, Dims, Type>{storage_, owned}, storage_{}
        {
            if constexpr (std::is_same_v<T, const T>) {
                return;
            }
            else {
                for (size_type ii = 0; ii < Dims; ++ii) {
                    this->data_[ii] = T{};
                }
            }
        }

        // DUAL constexpr Vector(std::initializer_list<T> values)
        //     : VectorOps<T, Dims, Type>{storage_, owned},
        //       storage_{values.begin(), values.end()}
        // {
        //     //  if const T is passed, skip this
        //     if constexpr (std::is_same_v<T, const T>) {
        //         this->data_ = storage_;
        //     }
        //     else {
        //         size_type ii = 0;
        //         for (const auto& value : values) {
        //             this->data_[ii++] = value;
        //         }
        //     }
        // }

        // Variadic constructor for direct element initialization
        template <typename... Args>
        DUAL constexpr Vector(Args... args)
            requires(sizeof...(Args) == Dims)
            : VectorOps<T, Dims, Type>{storage_, owned},
              storage_{static_cast<T>(args)...}
        {
            this->data_ = storage_;
        }

        // construct from raw memory
        DUAL constexpr Vector(T* data)
            : VectorOps<T, Dims, Type>{data, !owned}, storage_{}
        {
        }

        template <VectorType OtherType>
        DUAL constexpr Vector(const Vector<T, Dims, OtherType>& other)
            : VectorOps<T, Dims, Type>{storage_, true}
        {
            std::copy_n(other.data(), Dims, storage_);
        }

        // copy and move constructors
        DUAL constexpr Vector(const Vector& other)
            : VectorOps<T, Dims, Type>{storage_, owned}
        {
            for (size_type ii = 0; ii < Dims; ++ii) {
                this->data_[ii] = other[ii];
            }
        }

        DUAL constexpr Vector(Vector&& other) noexcept
            : VectorOps<T, Dims, Type>{storage_, owned}, storage_{}
        {
            std::move(other.data_, other.data_ + Dims, this->data_);
            other.data_ = nullptr;   // Null out moved-from object
        }

        // structured binding support
        template <size_t I>
        DUAL constexpr T& get() &
        {
            if constexpr (I >= Dims) {
                return zero_value_;
            }
            else {
                return this->data_[I];
            }
        }

        template <size_t I>
        DUAL constexpr const T& get() const&
        {
            if constexpr (I >= Dims) {
                return zero_value_;
            }
            else {
                return this->data_[I];
            }
        }

        template <size_t I>
        DUAL constexpr T&& get() &&
        {
            if constexpr (I >= Dims) {
                return std::move(zero_value_);
            }
            else {
                return std::move(this->data_[I]);
            }
        }

        template <size_t I>
        friend DUAL constexpr decltype(auto) get(Vector& v)
        {
            return v.template get<I>();
        }

        template <size_t I>
        friend DUAL constexpr decltype(auto) get(const Vector& v)
        {
            return v.template get<I>();
        }

        template <size_t I>
        friend DUAL constexpr decltype(auto) get(Vector&& v)
        {
            return std::move(v).template get<I>();
        }

        // assignment operators
        DUAL constexpr Vector& operator=(const Vector& other)
        {
            if (this != &other) {
                for (size_type ii = 0; ii < Dims; ++ii) {
                    this->data_[ii] = other[ii];
                }
            }
            return *this;
        }

        // compound assignment
        template <typename U, VectorType OtherType>
        DUAL constexpr Vector&
        operator+=(const Vector<U, Dims, OtherType>& other)
        {
            for (size_type ii = 0; ii < Dims; ++ii) {
                this->data_[ii] += other[ii];
            }
            return *this;
        }

        DUAL constexpr auto
        as_fourvec(const auto& vel, const auto lorentz) const
            requires Magnetic<Type>
        {
            const auto vdB = vel.dot(*this);
            const auto b0  = lorentz * vdB;
            const auto bs  = (*this) * (1.0 / lorentz) + vel * lorentz * vdB;
            return Vector<T, 4, VectorType::MAGNETIC_FOUR>{
              b0,
              bs[0],
              bs[1],
              bs[2]
            };
        }

        // Specialized operations for RMHD Magnetic four-vectors
        // get a view of the spatial part of the magnetic four-vector
        DUAL constexpr auto spatial_part() const
            requires MagneticFour<Type>
        {
            return Vector<T, 3, VectorType::MAGNETIC>{this->data_ + 1};
        }

        DUAL constexpr auto inner_product(const auto& other) const
            requires MagneticFour<Type>
        {
            return -this->data_[0] * other[0] + this->data_[1] * other[1] +
                   this->data_[2] * other[2] + this->data_[3] * other[3];
        }

        DUAL constexpr auto spatial_dot(const auto& other) const
            requires MagneticFour<Type>
        {
            return this->data_[1] * other[0] + this->data_[2] * other[1] +
                   this->data_[3] * other[2];
        }
    };

    // Vector view class, memory not owned
    template <typename T, size_type Dims, VectorType Type>
    class VectorView : public VectorOps<T, Dims, Type>
    {
      private:
        bool valid_ = false;

      public:
        DUAL constexpr VectorView(T* data = nullptr)
            : VectorOps<T, Dims, Type>{data}, valid_{data != nullptr}
        {
        }

        DUAL constexpr bool valid() const { return valid_; }

        // cache the data into memory-owning vector
        DUAL constexpr auto cache() const
        {
            return Vector<T, Dims, Type>{this->data_};
        }
    };

    // const vector view
    template <typename T, size_type Dims, VectorType Type>
    class ConstVectorView : public VectorOps<const T, Dims, Type>
    {
      public:
        DUAL constexpr ConstVectorView(const T* data = nullptr)
            : VectorOps<const T, Dims, Type>{data}
        {
        }

        // cache the data into memory-owning vector
        DUAL constexpr auto cache() const
        {
            return Vector<const T, Dims, Type>{this->data_};
        }

        DUAL constexpr auto
        as_fourvec(const auto& vel, const auto lorentz) const
            requires Magnetic<Type>
        {
            const auto vdB = vel.dot(*this);
            const auto b0  = lorentz * vdB;
            const auto bs  = (*this) * (1.0 / lorentz) + vel * lorentz * vdB;

            return Vector<const T, 4, VectorType::MAGNETIC_FOUR>{
              b0,
              bs[0],
              bs[1],
              bs[2]
            };
        }
    };

    // Specialization helpers
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

    class ZeroMagneticVectorView : public const_magnetic_vector_view_t<real, 3>
    {
      private:
        static constexpr real zero_storage[3] = {0.0, 0.0, 0.0};

      public:
        ZeroMagneticVectorView()
            : const_magnetic_vector_view_t<real, 3>(zero_storage)
        {
        }
    };

    class ZeroMagneticFourVectorView
        : public const_magnetic_four_vector_view_t<real>
    {
      private:
        static constexpr real zero_storage[4] = {0.0, 0.0, 0.0, 0.0};

      public:
        ZeroMagneticFourVectorView()
            : const_magnetic_four_vector_view_t<real>(zero_storage)
        {
        }
    };

    // set of unit vectors in the x1, x2, x3 directions
    namespace unit_vectors {
        static constexpr unit_vector_t<1> x1_1D{1};
        static constexpr unit_vector_t<1> x2_1D{0};
        static constexpr unit_vector_t<1> x3_1D{0};

        static constexpr unit_vector_t<2> x1_2D{1, 0};
        static constexpr unit_vector_t<2> x2_2D{0, 1};
        static constexpr unit_vector_t<2> x3_2D{0, 0};

        static constexpr unit_vector_t<3> x1_3D{1, 0, 0};
        static constexpr unit_vector_t<3> x2_3D{0, 1, 0};
        static constexpr unit_vector_t<3> x3_3D{0, 0, 1};

        // Helper to get unit vector by index
        template <size_type Dims>
        static constexpr const auto& get(size_type i)
        {
            if constexpr (Dims == 1) {
                return x1_1D;
            }
            else if constexpr (Dims == 2) {
                return (i == 1) ? x1_2D : x2_2D;
            }
            else {
                return (i == 1) ? x1_3D : (i == 2) ? x2_3D : x3_3D;
            }
        }
    }   // namespace unit_vectors

}   // namespace simbi

namespace std {
    template <typename T, size_type Dims, simbi::VectorType Type>
    struct tuple_size<simbi::Vector<T, Dims, Type>>
        : integral_constant<size_type, Dims> {
    };

    template <size_type I, typename T, size_type Dims, simbi::VectorType Type>
    struct tuple_element<I, simbi::Vector<T, Dims, Type>> {
        using type = T;
    };
}   // namespace std

#endif