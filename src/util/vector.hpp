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

    template <typename T, size_type Dims, VectorType Type = VectorType::GENERAL>
    class Vector;

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

    template <typename T, size_type Dims, VectorType Type>
    class Vector
    {
        static_assert(
            Dims >= vector_traits<Type>::dimensions,
            "Vector dimensions must match or exceed required dimensions for "
            "type"
        );

        T data[Dims];

      public:
        static constexpr VectorType vector_type = Type;

        Vector() = default;

        DUAL Vector(std::initializer_list<T> vals)
        {
            size_type ii = 0;
            for (const auto& val : vals) {
                data[ii++] = val;
            }
        }

        // Constructors
        template <typename... Args>
        DUAL Vector(Args... args) : data{static_cast<T>(args)...}
        {
        }

        // Vector operations
        DUAL Vector<T, Dims, Type> operator+(const Vector<T, Dims, Type>& other)
        {
            Vector<T, Dims, Type> result;
            for (size_type i = 0; i < Dims; ++i) {
                result[i] = data[i] + other[i];
            }
            return result;
        }

        DUAL Vector<T, Dims, Type> operator-(const Vector<T, Dims, Type>& other)
        {
            Vector<T, Dims, Type> result;
            for (size_type i = 0; i < Dims; ++i) {
                result[i] = data[i] - other[i];
            }
            return result;
        }

        DUAL Vector<T, Dims, Type> operator*(const T scalar) const
        {
            Vector<T, Dims, Type> result;
            for (size_type i = 0; i < Dims; ++i) {
                result[i] = data[i] * scalar;
            }
            return result;
        }

        DUAL T magnitude() const
        {
            T sum = 0;
            for (size_type i = 0; i < Dims; ++i) {
                sum += data[i] * data[i];
            }
            return std::sqrt(sum);
        }

        // normalization
        DUAL Vector<T, Dims, Type> normalize() const
        {
            T mag = magnitude();
            if (mag == 0) {
                throw std::runtime_error("Cannot normalize zero vector");
            }
            return *this * (1.0 / mag);
        }

        // LHS friend multiply operators with scalar
        friend DUAL Vector<T, Dims, Type>
        operator*(const T scalar, const Vector<T, Dims, Type>& vec)
        {
            return vec * scalar;
        }

        // Type-specific operations
        template <typename = std::enable_if_t<Type == VectorType::MAGNETIC>>
        DUAL T divergence() const
        {
            static_assert(Dims >= 3, "Divergence requires 3D vector");
            // Implement magnetic field divergence
            return data[0] + data[1] + data[2];
        }

        template <typename = std::enable_if_t<Type == VectorType::MAGNETIC>>
        DUAL Vector<T, 3, VectorType::MAGNETIC> curl() const
        {
            static_assert(Dims >= 3, "Curl requires 3D vector");
            // Implement magnetic field curl
            return Vector<T, 3, VectorType::MAGNETIC>{/*...*/};
        }

        // type conversion
        template <VectorType NewType>
        DUAL Vector<T, vector_traits<NewType>::dimensions, NewType> as() const
        {
            Vector<T, vector_traits<NewType>::dimensions, NewType> result;
            for (size_t i = 0;
                 i < std::min(Dims, vector_traits<NewType>::dimensions);
                 ++i) {
                result[i] = data[i];
            }
            return result;
        }

        DUAL T& operator[](size_type i) { return data[i]; }

        DUAL const T& operator[](size_type i) const { return data[i]; }

        // coordinate transformations
        DUAL Vector<T, Dims, Type> to_cartesian() const
        {
            // TODO: Implement conversion to Cartesian coordinates
            return *this;
        }

        DUAL Vector<T, Dims, Type> to_spherical() const
        {
            // TODO: Implement conversion to spherical coordinates
            return *this;
        }

        DUAL Vector<T, Dims, Type> to_cylindrical() const
        {
            // TODO: Implement conversion to cylindrical coordinates
            return *this;
        }

        //============================================
        // Vector calculus
        //============================================
        // dot product
        DUAL T dot(const Vector<T, Dims, Type>& other) const
        {
            T result = 0;
            for (size_type i = 0; i < Dims; ++i) {
                result += data[i] * other[i];
            }
            return result;
        }

        // inner product for spacetime vectors, assuming signature (-+++)
        template <typename = std::enable_if_t<Type == VectorType::SPACETIME>>
        DUAL T inner_product(const Vector<T, 4, VectorType::SPACETIME>& other
        ) const
        {
            static_assert(Dims == 4, "Inner product requires 4D vector");
            return -data[0] * other[0] + data[1] * other[1] +
                   data[2] * other[2] + data[3] * other[3];
        }

        // projection onto another vector
        DUAL Vector<T, Dims, Type> projection(const Vector<T, Dims, Type>& other
        ) const
        {
            T dot_product     = dot(other);
            T other_magnitude = other.magnitude();
            return (dot_product / (other_magnitude * other_magnitude)) * other;
        }

        // cross product
        template <typename = std::enable_if_t<Type == VectorType::SPATIAL>>
        DUAL Vector<T, 3, VectorType::SPATIAL>
        cross(const Vector<T, 3, VectorType::SPATIAL>& other) const
        {
            static_assert(Dims >= 3, "Cross product requires 3D vector");
            return Vector<T, 3, VectorType::SPATIAL>{
              data[1] * other[2] - data[2] * other[1],
              data[2] * other[0] - data[0] * other[2],
              data[0] * other[1] - data[1] * other[0]
            };
        }

        //==============================================
        // RELATIVISTIC OPERATIONS
        //==============================================
        // proper velocity magnitude
        DUAL T proper_velocity_magnitude() const
            requires(vector_traits<Type>::is_relativistic)
        {
            static_assert(
                Dims == 4,
                "Proper velocity magnitude requires 4D vector"
            );
            return std::sqrt(
                data[1] * data[1] + data[2] * data[2] + data[3] * data[3]
            );
        }

        DUAL Vector<T, 3, VectorType::SPATIAL> spatial_part() const
            requires(vector_traits<Type>::is_relativistic)
        {
            static_assert(Dims == 4, "Spatial part requires 4D vector");
            return Vector<T, 3, VectorType::SPATIAL>{data[1], data[2], data[3]};
        }

        //==============================================
        // RELATIVISTIC MHD OPERATIONS
        //==============================================

        // Magnetic four-vector composition
        template <typename = std::enable_if_t<Type == VectorType::MAGNETIC>>
        DUAL Vector<T, 4, VectorType::MAGNETIC_FOUR>
        compose_magnetic_fourvector(
            const Vector<T, 3, VectorType::SPATIAL>& velocity,
            T lorentz_f
        ) const
        {
            static_assert(Dims == 3, "Magnetic field must be 3D");

            const T vdotb = velocity.dot(*this);

            // Time component
            const T b0 = lorentz_f * vdotb;

            // Spatial components
            const auto spatial = (*this) * (1.0 / lorentz_f) + velocity * vdotb;

            return Vector<T, 4, VectorType::MAGNETIC_FOUR>{
              b0,
              spatial[0],
              spatial[1],
              spatial[2]
            };
        }

        // Helper to get magnetic field in fluid rest frame
        template <
            typename = std::enable_if_t<Type == VectorType::MAGNETIC_FOUR>>
        DUAL Vector<T, 3, VectorType::MAGNETIC> get_rest_field() const
        {
            static_assert(Dims == 4, "Must be 4-vector");
            return Vector<T, 3, VectorType::MAGNETIC>{
              data[1],
              data[2],
              data[3]
            };
        }

        // Get field strength scalar F_{\mu\nu}F^{\mu\nu}
        template <
            typename = std::enable_if_t<Type == VectorType::MAGNETIC_FOUR>>
        DUAL T field_strength_scalar() const
        {
            static_assert(Dims == 4, "Must be 4-vector");
            const T b0   = data[0];
            const auto b = get_rest_field();
            return 2 * (b.dot(b) - b0 * b0);
        }
    };

    // Specialization helpers
    template <typename T>
    using spatial_vector_t = Vector<T, 3, VectorType::SPATIAL>;

    template <typename T>
    using magnetic_vector_t = Vector<T, 3, VectorType::MAGNETIC>;

    template <typename T>
    using magnetic_four_vector_t = Vector<T, 4, VectorType::MAGNETIC_FOUR>;

    template <typename T>
    using spacetime_vector_t = Vector<T, 4, VectorType::SPACETIME>;

    // unit vector type
    using unit_vector_t = Vector<luint, 3, VectorType::GENERAL>;

}   // namespace simbi

#endif