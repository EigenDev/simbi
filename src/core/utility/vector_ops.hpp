#ifndef VECTOR_OPS_HPP
#define VECTOR_OPS_HPP

#include "config.hpp"
#include "core/containers/array.hpp"

namespace simbi::vector {
    template <typename VectorView>
    auto norm_squared(const VectorView& v)
    {
        real sum = 0.0;
        for (size_type ii = 0; ii < v.size(); ++ii) {
            sum += v[ii] * v[ii];
        }
        return sum;
    }

    template <typename VectorView>
    auto norm(const VectorView& v)
    {
        return std::sqrt(norm_squared(v));
    }

    template <typename VectorView1, typename VectorView2>
    auto dot(const VectorView1& a, const VectorView2& b)
    {
        static_assert(a.size() == b.size());
        real sum = 0.0;
        for (size_type ii = 0; ii < a.size(); ++ii) {
            sum += a[ii] * b[ii];
        }
        return sum;
    }

    // 3D cross product
    template <typename VectorView1, typename VectorView2>
    auto cross(const VectorView1& a, const VectorView2& b)
        requires(a.size() == 3 && b.size() == 3)
    {
        return array_t<real, 3>{
          a[1] * b[2] - a[2] * b[1],
          a[2] * b[0] - a[0] * b[2],
          a[0] * b[1] - a[1] * b[0]
        };
    }

    // cross products in less than 3D simply return the magntiude
    // for 2D, and zero for 1D
    template <typename VectorView1, typename VectorView2>
    auto cross(const VectorView1& a, const VectorView2& b)
        requires(a.size() == 2 && b.size() == 2)
    {
        return a[0] * b[1] - a[1] * b[0];   // scalar result
    }

    template <typename VectorView1, typename VectorView2>
    auto cross(const VectorView1& a, const VectorView2& b)
        requires(a.size() == 1 && b.size() == 1)
    {
        return 0.0;   // scalar result
    }

    // scale the vectors
    template <typename VectorView>
    auto operator*(VectorView& vec, real factor)
    {
        for (size_type ii = 0; ii < vec.size(); ++ii) {
            vec[ii] *= factor;
        }
        return vec;
    }

    template <typename VectorView>
    auto operator*(real factor, VectorView& vec)
    {
        return vec * factor;   // resue the operator* defined above
    }

    template <typename VectorView>
    auto operator/(VectorView& vec, real factor)
    {
        for (size_type ii = 0; ii < vec.size(); ++ii) {
            vec[ii] /= factor;
        }
        return vec;
    }

    // allow vector views to be added, subtracted, and compared
    template <typename VectorView1, typename VectorView2>
    auto operator+(const VectorView1& a, const VectorView2& b)
    {
        static_assert(a.size() == b.size());
        array_t<real, a.size()> result;
        for (size_type ii = 0; ii < a.size(); ++ii) {
            result[ii] = a[ii] + b[ii];
        }
        return result;
    }

    template <typename VectorView1, typename VectorView2>
    auto operator-(const VectorView1& a, const VectorView2& b)
    {
        static_assert(a.size() == b.size());
        array_t<real, a.size()> result;
        for (size_type ii = 0; ii < a.size(); ++ii) {
            result[ii] = a[ii] - b[ii];
        }
        return result;
    }

    template <typename VectorView1, typename VectorView2>
    auto operator==(const VectorView1& a, const VectorView2& b)
    {
        static_assert(a.size() == b.size());
        for (size_type ii = 0; ii < a.size(); ++ii) {
            if (a[ii] != b[ii]) {
                return false;
            }
        }
        return true;
    }

    template <typename VectorView1, typename VectorView2>
    auto operator!=(const VectorView1& a, const VectorView2& b)
    {
        static_assert(a.size() == b.size());
        for (size_type ii = 0; ii < a.size(); ++ii) {
            if (a[ii] != b[ii]) {
                return true;
            }
        }
        return false;
    }

    // unary minus operator for vectors
    template <typename VectorView>
    auto operator-(const VectorView& v)
    {
        array_t<real, v.size()> result;
        for (size_type ii = 0; ii < v.size(); ++ii) {
            result[ii] = -v[ii];
        }
        return result;
    }

}   // namespace simbi::vector

#endif   // VECTOR_OPS_HPP
