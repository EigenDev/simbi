#ifndef SIMBI_FIELD_HPP
#define SIMBI_FIELD_HPP

#include "compute/functional/fp.hpp"   // for fp::<etc>
#include "compute/math/domain.hpp"
#include "core/base/buffer.hpp"         // for buffer_t
#include "data/containers/vector.hpp"   // for iarray
#include <concepts>                     // for std::integral
#include <cstdint>                      // for std::uint64_t
#include <type_traits>                  // for std::conditional_t
#include <utility>                      // for std::move

namespace simbi::expr {
    template <typename Field, typename Expr>
    void assign(Field& field, const Expr& expr);

    template <typename Derived>
    struct expression_t;
}   // namespace simbi::expr

namespace simbi {
    // forward declarations
    template <std::uint64_t Dims>
    struct domain_t;

    template <typename T>
    concept expression_type = requires(const T& expr) {
        // Must inherit from expression_t CRTP base
        requires std::derived_from<T, expr::expression_t<T>>;

        // Must have domain() method
        { expr.domain() } -> std::same_as<decltype(expr.domain())>;
    };

    template <typename T, std::uint64_t Dims, bool IsConst = false>
    struct field_view_t {
        using value_type     = T;
        using pointer_type   = std::conditional_t<IsConst, const T*, T*>;
        using reference_type = std::conditional_t<IsConst, const T&, T&>;
        static constexpr std::uint64_t dimensions = Dims;

        pointer_type data_;
        domain_t<Dims> global_domain_;
        domain_t<Dims> local_domain_;
        iarray<Dims> strides_;

        // constructor
        field_view_t(
            pointer_type data,
            domain_t<Dims> global_domain,
            domain_t<Dims> local_domain,
            iarray<Dims> strides
        )
            : data_(data),
              global_domain_(std::move(global_domain)),
              local_domain_(std::move(local_domain)),
              strides_(std::move(strides))
        {
        }

        // copy constructor and assignment (cheap copies)
        field_view_t(const field_view_t&)            = default;
        field_view_t& operator=(const field_view_t&) = default;
        field_view_t(field_view_t&&)                 = default;
        field_view_t& operator=(field_view_t&&)      = default;

        // implicit conversion from non-const to const view
        template <bool OtherConst>
        field_view_t(const field_view_t<T, Dims, OtherConst>& other)
            requires(IsConst && !OtherConst)
            : data_(other.data_),
              global_domain_(other.global_domain_),
              local_domain_(other.local_domain_),
              strides_(other.strides_)
        {
        }

        // basic accessors
        pointer_type data() const noexcept { return data_; }
        auto size() const noexcept { return local_domain_.size(); }
        auto shape() const noexcept { return local_domain_.shape(); }

        // coordinate access
        template <std::integral coord_t>
        reference_type operator[](const ivec<coord_t, Dims>& coord) const
        {
            return data_[memory_offset(coord)];
        }

        // linear access
        reference_type operator[](std::int64_t linear_idx) const
        {
            auto coord = local_domain_.linear_to_coord(linear_idx);
            return (*this)[coord];
        }

        // create sub-views (might need this, dunno when)
        template <std::integral coord_t>
        field_view_t contract(const ivec<coord_t, Dims>& amount) const
        {
            auto new_local = set_ops::contract(local_domain_, amount);
            return create_view(new_local);
        }

        field_view_t contract(std::int64_t amount) const
        {
            iarray<Dims> uniform_amount;
            for (std::uint64_t i = 0; i < Dims; ++i) {
                uniform_amount[i] = amount;
            }
            return contract(uniform_amount);
        }

        // arbitrary subdomain view
        field_view_t operator[](const domain_t<Dims>& subdomain) const
        {
            return create_view(subdomain);
        }

        // pipeline syntax
        template <typename Op>
        auto operator|(Op&& op) const
        {
            return std::forward<Op>(op)(*this);
        }

        // evaluation interface
        template <typename coord_t>
        auto operator()(const ivec<coord_t, Dims>& coord) const
        {
            return (*this)[coord];
        }

        // domain queries
        const auto& domain() const noexcept { return local_domain_; }
        const auto& global_domain() const noexcept { return global_domain_; }

        // create sub-sub-view with same constness
        field_view_t create_view(const domain_t<Dims>& new_local) const
        {
            return field_view_t{data_, global_domain_, new_local, strides_};
        }

        // assignmet operators from expressions
        template <expression_type Expr>
        field_view_t& operator=(const Expr& expr)
        {
            expr::assign(*this, expr);
            return *this;
        }

        template <expression_type Expr>
        field_view_t& operator+=(const Expr& expr)
        {
            auto domain = make_domain(shape());
            for (auto coord : domain) {
                (*this)[coord] += expr.eval(coord);
            }
            return *this;
        }

      private:
        // coordinate to memory offset mapping
        template <std::integral coord_t>
        auto memory_offset(const ivec<coord_t, Dims>& local_coord) const
        {
            // convert local coord to global buffer coordinate
            auto global_coord = local_coord + local_domain_.start;

            // use strides to get memory offset
            return fp::zip(global_coord, strides_) |
                   fp::unpack_map([](auto x, auto y) { return x * y; }) |
                   fp::sum;
        }
    };

    template <typename T, std::uint64_t Dims = 1>
    struct field_t {
        using value_type                          = T;
        static constexpr std::uint64_t dimensions = Dims;

        buffer_t<T> buffer_;             // owns the data
        domain_t<Dims> global_domain_;   // full buffer coordinate space
        domain_t<Dims> local_domain_;    // this view's coordinate space
        iarray<Dims> strides_;           // memory layout for global domain

        field_t() : buffer_{}, global_domain_{}, local_domain_{}, strides_{} {}

        // move only - no copying massive data
        field_t(const field_t&)            = delete;
        field_t& operator=(const field_t&) = delete;
        field_t(field_t&&)                 = default;
        field_t& operator=(field_t&&)      = default;

        // factory functions
        static field_t make_field(const iarray<Dims>& shape)
        {
            auto domain      = domain_t<Dims>{{}, shape};
            auto domain_copy = domain;
            auto buffer      = buffer_t<T>(domain.size());
            auto strides     = compute_strides(shape);

            return field_t{
              std::move(buffer),
              std::move(domain),
              std::move(domain_copy),
              std::move(strides)
            };
        }

        static field_t
        copy_from_numpy(void* numpy_data, const iarray<Dims>& shape)
        {
            auto domain = domain_t<Dims>{{}, shape};
            auto buffer =
                buffer_t<T>::copy_from_numpy(numpy_data, domain.size());
            auto strides = compute_strides(shape);

            return field_t{std::move(buffer), domain, domain, strides};
        }

        static field_t wrap_numpy(void* numpy_data, const iarray<Dims>& shape)
        {
            auto domain  = domain_t<Dims>{{}, shape};
            auto buffer  = buffer_t<T>::wrap_numpy(numpy_data, domain.size());
            auto strides = compute_strides(shape);

            return field_t{std::move(buffer), domain, domain, strides};
        }

        static field_t zeros(const iarray<Dims>& shape)
        {
            auto field = make_field(shape);
            if constexpr (std::is_floating_point_v<T>) {
                std::fill(field.data(), field.data() + field.size(), T{0});
            }
            else {
                std::fill(field.data(), field.data() + field.size(), T{});
            }
            return field;
        }

        // basic accessors
        T* data() { return buffer_.data(); }
        const T* data() const { return buffer_.data(); }
        auto size() const { return local_domain_.size(); }
        auto shape() const { return local_domain_.shape(); }
        auto device() const { return buffer_.device(); }

        // coordinate access
        template <std::integral coord_t>
        T& operator[](const ivec<coord_t, Dims>& coord)
        {
            return buffer_.data()[memory_offset(coord)];
        }
        template <std::integral coord_t>
        const T& operator[](const ivec<coord_t, Dims>& coord) const
        {
            return buffer_.data()[memory_offset(coord)];
        }

        // linear access
        T& operator[](std::int64_t linear_idx)
        {
            auto coord = local_domain_.linear_to_coord(linear_idx);
            return (*this)[coord];
        }
        const T& operator[](std::int64_t linear_idx) const
        {
            auto coord = local_domain_.linear_to_coord(linear_idx);
            return (*this)[coord];
        }

        // domain operations that return views
        template <std::integral coord_t>
        auto contract(const ivec<coord_t, Dims>& amount) &
        {
            auto new_local = set_ops::contract(local_domain_, amount);
            return create_view(new_local);
        }

        template <std::integral coord_t>
        auto contract(const ivec<coord_t, Dims>& amount) const&
        {
            auto new_local = set_ops::contract(local_domain_, amount);
            return create_view(new_local);
        }

        auto contract(std::int64_t amount) &
        {
            iarray<Dims> uniform_amount;
            for (std::uint64_t i = 0; i < Dims; ++i) {
                uniform_amount[i] = amount;
            }
            return contract(uniform_amount);
        }

        auto contract(std::int64_t amount) const&
        {
            iarray<Dims> uniform_amount;
            for (std::uint64_t i = 0; i < Dims; ++i) {
                uniform_amount[i] = amount;
            }
            return contract(uniform_amount);
        }

        // arbitrary subdomain view
        auto operator[](const domain_t<Dims>& subdomain) &
        {
            return create_view(subdomain);
        }

        auto operator[](const domain_t<Dims>& subdomain) const&
        {
            return create_view(subdomain);
        }

        // pipeline syntax
        template <typename Op>
        auto operator|(Op&& op) const
        {
            return std::forward<Op>(op)(*this);
        }

        // evaluation interface
        template <typename coord_t>
        auto operator()(const ivec<coord_t, Dims>& coord) const
        {
            return (*this)[coord];
        }

        // domain queries
        const auto& domain() const { return local_domain_; }
        const auto& global_domain() const { return global_domain_; }

        // assignment operator for expressions
        template <expression_type Expr>
        field_t& operator=(const Expr& expr)
        {
            expr::assign(*this, expr);
            return *this;
        }

        // compound assignment for expressions
        template <expression_type Expr>
        field_t& operator+=(const Expr& expr)
        {
            auto domain = this->domain();
            for (auto coord : domain) {
                (*this)[coord] += expr.eval(coord);
            }
            return *this;
        }

        // similar compound operators for other operations
        template <expression_type Expr>
        field_t& operator-=(const Expr& expr)
        {
            auto domain = this->domain();
            for (auto coord : domain) {
                (*this)[coord] -= expr.eval(coord);
            }
            return *this;
        }

        template <expression_type Expr>
        field_t& operator*=(const Expr& expr)
        {
            auto domain = this->domain();
            for (auto coord : domain) {
                (*this)[coord] *= expr.eval(coord);
            }
            return *this;
        }

        template <expression_type Expr>
        field_t& operator/=(const Expr& expr)
        {
            auto domain = this->domain();
            for (auto coord : domain) {
                (*this)[coord] /= expr.eval(coord);
            }
            return *this;
        }

      private:
        // private constructor for internal use
        field_t(
            buffer_t<T> buffer,
            domain_t<Dims> global_domain,
            domain_t<Dims> local_domain,
            iarray<Dims> strides
        )
            : buffer_(std::move(buffer)),
              global_domain_(std::move(global_domain)),
              local_domain_(std::move(local_domain)),
              strides_(std::move(strides))
        {
        }

        // coordinate to memory offset mapping
        template <std::integral coord_t>
        auto memory_offset(const ivec<coord_t, Dims>& local_coord) const
        {
            // convert local coord to global buffer coordinate
            auto global_coord =
                local_coord + (global_domain_.start - local_domain_.start);

            // use strides to get memory offset using your FP toolkit
            return fp::zip(global_coord, strides_) | fp::map([](auto pair) {
                       auto [c, s] = pair;
                       return c * s;
                   }) |
                   fp::sum;
        }

        // stride computation for row-major layout
        static iarray<Dims> compute_strides(const iarray<Dims>& shape)
        {
            iarray<Dims> strides;
            strides[Dims - 1] = 1;   // rightmost dimension has stride 1

            for (std::int64_t dim = Dims - 2; dim >= 0; --dim) {
                strides[dim] = strides[dim + 1] * shape[dim + 1];
            }

            return strides;
        }

        // domain operations (return views)
        auto create_view(const domain_t<Dims>& new_local) &
        {
            return field_view_t<T, Dims, false>{
              buffer_.data(),
              global_domain_,
              set_ops::center(new_local, local_domain_),
              strides_
            };
        }

        // create const view from const field
        auto create_view(const domain_t<Dims>& new_local) const&
        {
            return field_view_t<T, Dims, true>{
              buffer_.data(),
              global_domain_,
              set_ops::center(new_local, local_domain_),
              strides_
            };
        }

        // prevent creation of views from temporaries
        auto create_view(const domain_t<Dims>& new_local) &&      = delete;
        auto create_view(const domain_t<Dims>& new_local) const&& = delete;
    };
}   // namespace simbi

#endif   // SIMBI_FIELD_HPP
