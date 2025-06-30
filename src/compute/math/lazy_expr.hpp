#ifndef SIMBI_LAZY_EXPR_HPP
#define SIMBI_LAZY_EXPR_HPP

#include "config.hpp"
#include "core/base/concepts.hpp"
#include "core/base/memory.hpp"
#include "core/utility/enums.hpp"
#include "index_space.hpp"
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <iterator>
#include <memory>
#include <ostream>
#include <type_traits>

namespace simbi {
    template <std::uint64_t Dims>
    struct physics_context_t;

    namespace vecops {
        template <concepts::VectorLike Vec1, concepts::VectorLike Vec2>
        DUAL constexpr auto dot(const Vec1& a, const Vec2& b);
    }

    namespace mesh {
        template <std::uint64_t Dims>
        struct mesh_config_t;

        template <std::uint64_t Dims, Geometry G>
        struct geometry_solver_t;
    }   // namespace mesh

    namespace ibsystem {
        template <typename T, std::uint64_t Dims>
        class GridBodyDeltaCollector;

        template <typename T, std::uint64_t Dims>
        class ComponentBodySystem;

        template <typename T, std::uint64_t Dims>
        struct BodyDelta;

        template <typename T, std::uint64_t Dims>
        struct Body;

        template <std::uint64_t Dims, Geometry G, typename Primitive>
        DEV auto compute_ib_at_coordinate(
            const uarray<Dims>& coord,
            const mesh::geometry_solver_t<Dims, G>& geo,
            const ComponentBodySystem<real, Dims>& bodies,
            GridBodyDeltaCollector<real, Dims>& collector,
            const physics_context_t<Dims>& ctx
        );
    }   // namespace ibsystem

    namespace state {
        template <std::uint64_t Dims>
        struct expression_t;

        struct hydro_source_tag;
        struct gravity_source_tag;
        struct ib_source_tag;
    }   // namespace state

    using namespace state;
    using namespace ibsystem;
    // forward declaration
    template <typename T, std::uint64_t Dims>
    struct field_t;

    template <typename T, std::uint64_t Dims>
    struct vector_t;

    template <std::uint64_t Dims>
    using uarray = vector_t<std::uint64_t, Dims>;
}   // namespace simbi

namespace simbi::expr {
    using namespace simbi::concepts;
    // forward declaration of expression types
    template <typename Left, typename Right>
    class add_expr_t;

    template <typename Expr>
    class scale_expr_t;

    template <typename T, std::uint64_t Dims>
    class zero_expr_t;

    template <typename T>
    class integer_range_t;

    template <typename Generator>
    class generator_range_t;

    // helper to deduce the element type from an expression
    template <Expression E>
    struct expression_element {
        using type = typename decltype(std::declval<E>().realize())::value_type;
    };

    // pecializations
    template <typename T, std::uint64_t Dims>
    struct expression_element<zero_expr_t<T, Dims>> {
        using type = T;
    };

    template <typename T>
    struct expression_element<integer_range_t<T>> {
        using type = T;
    };

    template <typename Generator>
    struct expression_element<generator_range_t<Generator>> {
        using type = typename generator_range_t<Generator>::value_type;
    };

    template <Expression E>
    using expression_element_t = typename expression_element<E>::type;

    // unified ops that work with any combination of expressions
    template <Expression E1, Expression E2>
    auto operator+(const E1& lhs, const E2& rhs) -> add_expr_t<E1, E2>
    {
        return add_expr_t<E1, E2>{lhs, rhs};
    }

    template <Expression E1, Expression E2>
    auto operator-(const E1& lhs, const E2& rhs)
        -> add_expr_t<E1, scale_expr_t<E2>>
    {
        return add_expr_t<E1, scale_expr_t<E2>>{
          lhs,
          scale_expr_t<E2>{rhs, -1.0}
        };
    }

    template <Expression E>
    auto operator*(const E& expr, double factor) -> scale_expr_t<E>
    {
        return scale_expr_t<E>{expr, factor};
    }

    template <Expression E>
    auto operator*(double factor, const E& expr) -> scale_expr_t<E>
    {
        return scale_expr_t<E>{expr, factor};
    }

    template <Expression E>
    auto operator/(const E& expr, double factor) -> scale_expr_t<E>
    {
        return scale_expr_t<E>{expr, 1.0 / factor};
    }

    // unary minus
    template <Expression E>
    auto operator-(const E& expr) -> scale_expr_t<E>
    {
        return scale_expr_t<E>{expr, -1.0};
    }

    // assignment-style operators for expressions that support it
    template <Expression E1, Expression E2>
        requires requires(E1& e1, const E2& e2) { e1.add(e2); }
    auto operator+=(E1& lhs, const E2& rhs) -> E1&
    {
        lhs.add(rhs);
        return lhs;
    }

    // comparison operators
    template <Expression E1, Expression E2>
        requires std::equality_comparable_with<
            expression_element_t<E1>,
            expression_element_t<E2>>
    auto operator==(const E1& lhs, const E2& rhs) -> bool
    {
        if (lhs.size() != rhs.size()) {
            return false;
        }

        auto lhs_result = lhs.realize();
        auto rhs_result = rhs.realize();

        for (std::uint64_t i = 0; i < lhs.size(); ++i) {
            if (lhs_result.data()[i] != rhs_result.data()[i]) {
                return false;
            }
        }
        return true;
    }

    // stream output for debugging
    template <Expression E>
    auto operator<<(std::ostream& os, const E& expr) -> std::ostream&
    {
        auto result = expr.realize();
        os << "[";
        for (std::uint64_t i = 0; i < std::min(expr.size(), 10ULL); ++i) {
            if (i > 0) {
                os << ", ";
            }
            os << result.data()[i];
        }
        if (expr.size() > 10) {
            os << ", ...";
        }
        os << "]";
        return os;
    }

}   // namespace simbi::expr

namespace simbi::expr {
    using namespace simbi::base;

    template <typename Source, typename Transform>
    class lazy_expr_t;

    // helper to deduce result type of transformation
    template <typename Transform, typename Input>
    using transform_result_t = std::invoke_result_t<Transform, Input>;

    // arithmetic expression types for composition
    template <typename Left, typename Right>
    class add_expr_t
    {
      private:
        const Left& left_;
        const Right& right_;

      public:
        add_expr_t(const Left& left, const Right& right)
            : left_(left), right_(right)
        {
        }

        std::uint64_t size() const { return left_.size(); }

        template <typename F>
        auto map(F&& func) -> lazy_expr_t<add_expr_t, F>
        {
            return lazy_expr_t<add_expr_t, F>{*this, std::forward<F>(func)};
        }

        template <typename Other>
        auto add(const Other& other) const -> add_expr_t<decltype(*this), Other>
        {
            return add_expr_t<decltype(*this), Other>{*this, other};
        }

        // materialize addition - returns new field
        // default, should be deduced
        // properly
        auto realize() const
        {
            // auto get_result = [](const auto& operand) {
            //     if constexpr (requires { operand.realize(); }) {
            //         // it's an expression, realize it
            //         return operand.realize();
            //     }
            //     else {
            //         return operand;   // it's already realized
            //     }
            // };
            auto left_result  = left_.realize();    // get_result(left_);
            auto right_result = right_.realize();   // get_result(right_);

            constexpr auto Dims = decltype(left_result)::value_type::dimensions;

            // create result field with same domain as left
            auto result =
                field_t<typename decltype(left_result)::value_type, Dims>{
                  left_result.domain(),
                  std::make_shared<unified_memory_t<
                      typename decltype(left_result)::value_type>>(
                      left_result.size()
                  )
                };

            // element-wise addition
            for (std::uint64_t ii = 0; ii < result.size(); ++ii) {
                result.data()[ii] =
                    left_result.data()[ii] + right_result.data()[ii];
            }

            return result;
        }
    };

    template <typename Expr>
    class scale_expr_t
    {
      private:
        const Expr& expr_;
        double factor_;

      public:
        scale_expr_t(const Expr& expr, double factor)
            : expr_(expr), factor_(factor)
        {
        }

        std::uint64_t size() const { return expr_.size(); }

        template <typename F>
        auto map(F&& func) -> lazy_expr_t<scale_expr_t, F>
        {
            return lazy_expr_t<scale_expr_t, F>{*this, std::forward<F>(func)};
        }

        template <typename Other>
        auto add(const Other& other) const -> add_expr_t<decltype(*this), Other>
        {
            return add_expr_t<decltype(*this), Other>{*this, other};
        }

        // materialize scaling - returns new field
        auto realize() const
        {
            auto expr_result = expr_.realize();
            using value_type = typename decltype(expr_result)::value_type;

            // create result field with same domain
            auto result =
                field_t<value_type, decltype(expr_result)::dimensions>{
                  expr_result.domain_,
                  std::make_shared<unified_memory_t<value_type>>(
                      expr_result.size()
                  )
                };

            // wlement-wise scaling
            for (std::uint64_t ii = 0; ii < result.size(); ++ii) {
                result.data()[ii] = expr_result.data()[ii] * factor_;
            }

            return result;
        }
    };

    // core lazy expression class
    template <typename Source, typename Transform>
    class lazy_expr_t
    {
      private:
        const Source& source_;
        Transform transform_;

      public:
        lazy_expr_t(const Source& source, Transform transform)
            : source_(source), transform_(std::move(transform))
        {
        }

        template <typename F>
        auto map(F&& func) const -> lazy_expr_t<lazy_expr_t, F>
        {
            return lazy_expr_t<lazy_expr_t, F>{*this, std::forward<F>(func)};
        }

        template <typename Other>
        auto add(const Other& other) const -> add_expr_t<decltype(*this), Other>
        {
            return add_expr_t<decltype(*this), Other>{*this, other};
        }

        // execute the computation chain - returns new field
        auto realize() const
        {
            // deduce result type
            using source_element_t = typename Source::value_type;
            using result_t = transform_result_t<Transform, source_element_t>;

            // for field sources, preserve domain structure
            // if constexpr (requires { source_.domain(); }) {
            auto result = field_t<
                result_t,
                std::decay_t<decltype(source_.domain())>::Dims>{
              source_.domain(),
              std::make_shared<unified_memory_t<result_t>>(source_.size())
            };

            // apply transformation to each element
            auto* input_ptr  = source_.data();
            auto* output_ptr = result.data();

            for (std::uint64_t ii = 0; ii < source_.size(); ++ii) {
                output_ptr[ii] = apply_transform(input_ptr[ii]);
            }

            return result;
            // }
            // else {
            //     // fallback for non-field sources
            //     unified_memory_t<result_t> result(source_.size());
            //     auto* input_ptr  = source_.data();
            //     auto* output_ptr = result.data();

            //     for (std::uint64_t ii = 0; ii < source_.size(); ++ii) {
            //         output_ptr[ii] = apply_transform(input_ptr[ii]);
            //     }

            //     return result;
            // }
        }

        std::uint64_t size() const { return source_.size(); }

      private:
        template <typename Input>
        auto apply_transform(const Input& input) const
        {
            if constexpr (std::is_same_v<Source, unified_memory_t<Input>>) {
                return transform_(input);
            }
            else {
                auto intermediate = source_.apply_transform(input);
                return transform_(intermediate);
            }
        }
    };

    // specialization for field_t sources
    template <typename T, std::uint64_t Dims, typename Transform>
    class lazy_expr_t<field_t<T, Dims>, Transform>
    {
      private:
        const field_t<T, Dims>& source_;
        Transform transform_;

      public:
        using value_type = transform_result_t<Transform, T>;

        lazy_expr_t(const field_t<T, Dims>& source, Transform transform)
            : source_(source), transform_(std::move(transform))
        {
        }

        template <typename F>
        auto map(F&& func) const -> lazy_expr_t<lazy_expr_t, F>
        {
            return lazy_expr_t<lazy_expr_t, F>{*this, std::forward<F>(func)};
        }

        template <typename Other>
        auto add(const Other& other) const -> add_expr_t<decltype(*this), Other>
        {
            return add_expr_t<decltype(*this), Other>{*this, other};
        }

        auto realize() const
        {
            auto result = field_t<value_type, Dims>{
              source_.domain_,
              std::make_shared<unified_memory_t<value_type>>(source_.size())
            };

            // Apply transformation
            auto* input_ptr  = source_.data();
            auto* output_ptr = result.data();

            for (std::uint64_t i = 0; i < source_.size(); ++i) {
                output_ptr[i] = transform_(input_ptr[i]);
            }

            return result;
        }

        std::uint64_t size() const { return source_.size(); }
    };

    // coordinate space specialization - direct iteration over coordinates
    template <std::uint64_t Dims, typename Transform>
    class lazy_expr_t<index_space_t<Dims>, Transform>
    {
      private:
        index_space_t<Dims> source_;
        Transform transform_;

        auto compute_at(std::uint64_t linear_idx) const
        {
            // convert linear index to coordinate
            auto coord = source_.index_to_coord(linear_idx);

            // apply the transformation function
            return transform_(coord);
        }

      public:
        lazy_expr_t(const index_space_t<Dims>& source, Transform transform)
            : source_(source), transform_(std::move(transform))
        {
        }

        template <typename F>
        auto map(F&& func) -> lazy_expr_t<lazy_expr_t, F>
        {
            return lazy_expr_t<lazy_expr_t, F>{*this, std::forward<F>(func)};
        }

        template <typename Other>
        auto add(const Other& other) const -> add_expr_t<decltype(*this), Other>
        {
            return add_expr_t<decltype(*this), Other>{*this, other};
        }

        auto realize() const
        {
            // deduce result type from transform applied to coordinate
            using coord_t          = uarray<Dims>;
            using result_element_t = std::invoke_result_t<Transform, coord_t>;

            // create result memory with same size as coordinate space
            unified_memory_t<result_element_t> result(source_.size());

            // apply transform to each coordinate in the space
            for (std::uint64_t i = 0; i < source_.size(); ++i) {
                auto coord       = source_.index_to_coord(i);
                result.data()[i] = transform_(coord);
            }

            return result;
        }

        template <typename OutputField>
        void realize_into(OutputField& target) const
        {
            // write computed results directly into target's memory
            auto* output_ptr = target.data();
            for (std::uint64_t ii = 0; ii < this->size(); ++ii) {
                // compute without storing
                output_ptr[ii] = this->compute_at(ii);
            }
        }

        std::uint64_t size() const { return source_.size(); }
    };

    // void specialization for non-returning expressions
    template <typename Source, typename Transform>
        requires std::is_void_v<
            transform_result_t<Transform, typename Source::value_type>>
    class lazy_expr_t<Source, Transform>
    {
      private:
        const Source& source_;
        Transform transform_;

      public:
        lazy_expr_t(const Source& source, Transform transform)
            : source_(source), transform_(std::move(transform))
        {
        }

        template <typename F>
        auto map(F&& func) -> lazy_expr_t<lazy_expr_t, F>
        {
            return lazy_expr_t<lazy_expr_t, F>{*this, std::forward<F>(func)};
        }

        template <typename Other>
        auto add(const Other& other) const -> add_expr_t<decltype(*this), Other>
        {
            return add_expr_t<decltype(*this), Other>{*this, other};
        }

        void realize() const
        {
            // execute the transformation without returning a value
            for (std::uint64_t i = 0; i < source_.size(); ++i) {
                auto coord = source_.index_to_coord(i);
                transform_(coord);
            }
        }

        std::uint64_t size() const { return source_.size(); }
    };

    template <typename T, std::uint64_t Dims>
    class zero_expr_t
    {
        index_space_t<Dims> domain_;

      public:
        zero_expr_t(const index_space_t<Dims>& domain) : domain_(domain) {}

        template <typename Other>
        auto add(const Other& other) const -> add_expr_t<decltype(*this), Other>
        {
            return add_expr_t<decltype(*this), Other>{*this, other};
        }

        auto realize() const
        {
            return field_t<T, Dims>{
              domain_,
              std::make_shared<unified_memory_t<T>>(domain_.size())
            };
        }

        template <typename Other>
        auto operator+(const Other& other) const
        {
            return add_expr_t<zero_expr_t, Other>{*this, other};
        }
    };

    template <std::uint64_t Dims, typename FluxComputeFunction>
    class dimension_fold_expr_t
    {
        index_space_t<Dims> domain_;
        FluxComputeFunction flux_compute_;

      public:
        // deduce the result type from what the flux function produces
        using flux_expr_t =
            std::invoke_result_t<FluxComputeFunction, std::uint64_t>;
        using result_t   = decltype(std::declval<flux_expr_t>().realize());
        using value_type = typename result_t::value_type;

        // ctor takes the domain and a function that computes flux
        // difference for a given dimension
        dimension_fold_expr_t(
            const index_space_t<Dims>& domain,
            FluxComputeFunction func
        )
            : domain_(domain), flux_compute_(std::move(func))
        {
        }

        auto realize() const
        {
            auto result = flux_compute_(0).realize();

            for (std::uint64_t dim = 1; dim < Dims; ++dim) {
                auto flux_diff = flux_compute_(dim).realize();

                // add to accumulated result
                for (std::uint64_t i = 0; i < result.size(); ++i) {
                    result.data()[i] = result.data()[i] + flux_diff.data()[i];
                }
            }
            return result;
        }

        // chainable add method
        template <typename Other>
        auto add(const Other& other) const
            -> add_expr_t<dimension_fold_expr_t, Other>
        {
            return add_expr_t<dimension_fold_expr_t, Other>{*this, other};
        }

        std::uint64_t size() const { return domain_.size(); }
    };

    // integer range generator
    template <typename T = std::uint64_t>
    class integer_range_t
    {
      public:
        using value_type = T;

      private:
        T start_, end_, step_;

      public:
        constexpr integer_range_t() : start_(0), end_(0), step_(1) {}

        constexpr integer_range_t(T end) : start_(0), end_(end), step_(1) {}

        constexpr integer_range_t(T start, T end, T step = 1)
            : start_(start), end_(end), step_(step)
        {
        }

        constexpr T operator[](std::uint64_t idx) const
        {
            return start_ + static_cast<T>(idx) * step_;
        }

        constexpr std::uint64_t size() const
        {
            return static_cast<std::uint64_t>(
                (end_ - start_ + step_ - 1) / step_
            );
        }

        // iterator support for range-based loops
        class iterator
        {
            T current_;
            T step_;

          public:
            using iterator_category = std::forward_iterator_tag;
            using value_type        = T;
            using difference_type   = std::ptrdiff_t;
            using pointer           = T*;
            using reference         = T&;

            constexpr iterator() : current_(0), step_(1) {}

            constexpr iterator(T current, T step)
                : current_(current), step_(step)
            {
            }

            constexpr T operator*() const { return current_; }

            constexpr iterator& operator++()
            {
                current_ += step_;
                return *this;
            }

            constexpr iterator operator++(int)
            {
                iterator temp = *this;
                ++(*this);
                return temp;
            }

            constexpr bool operator==(const iterator& other) const
            {
                // Note: >= for end comparison
                return current_ >= other.current_;
            }

            constexpr bool operator!=(const iterator& other) const
            {
                return !(*this == other);
            }
        };

        constexpr iterator begin() const { return iterator{start_, step_}; }
        constexpr iterator end() const { return iterator{end_, step_}; }
    };

    template <std::uint64_t Dims, typename HydroState, typename Tag>
    class source_expr_t
    {
        const expression_t<Dims>& expr_;
        index_space_t<Dims> domain_;
        const HydroState& state_;
        bool enabled_;

      public:
        using conserved_t = typename HydroState::conserved_t;

        source_expr_t(
            const expression_t<Dims>& expr,
            const index_space_t<Dims>& domain,
            const HydroState& state,
            bool enabled
        )
            : expr_(expr), domain_(domain), state_(state), enabled_(enabled)
        {
        }

        auto realize() const
        {
            // create result field
            auto result = field_t<conserved_t, Dims>::zeros(domain_.shape());

            if (!enabled_) {
                return result;
            }

            // evaluate source at each point in the domain
            for (std::uint64_t i = 0; i < domain_.size(); ++i) {
                auto coord = domain_.index_to_coord(i);

                // convert to spatial coordinates
                auto spatial_coord = state_.geom_solver.the_centroid(coord);

                if constexpr (std::same_as<Tag, hydro_source_tag>) {
                    // get conserved state at this point
                    const auto& cons = state_.cons[coord];
                    // compute source term
                    auto source_contrib = expr_.apply(
                        spatial_coord,
                        cons,
                        state_.metadata.time,   // current time
                        state_.metadata.dt      // time step
                    );

                    result.data()[i] = source_contrib;
                }
                else {
                    // get primitive state at this point
                    const auto& prim = state_.prim[coord];
                    // compute source term
                    auto source_contrib = expr_.apply(
                        spatial_coord,
                        prim,
                        state_.metadata.time,   // current time
                        state_.metadata.gamma   // adiabatic index
                    );

                    result.data()[i] = source_contrib;
                }
            }

            return result;
        }

        template <typename Other>
        auto add(const Other& other) const -> add_expr_t<source_expr_t, Other>
        {
            return add_expr_t<source_expr_t, Other>{*this, other};
        }

        std::uint64_t size() const { return domain_.size(); }
    };

    // lightweight ib source expression
    template <typename HydroState>
    class ib_source_expr_t
    {
        static constexpr auto Dims = HydroState::dimensions;
        index_space_t<Dims> domain_;
        const HydroState& state_;

      public:
        ib_source_expr_t(index_space_t<Dims> domain, const HydroState& state)
            : domain_{domain}, state_{state}
        {
        }

        // single pass: compute fluid effects + accumulate deltas
        auto realize() const
        {
            return domain_.map([&state = state_] DEV(auto coord) {
                std::cout << "Computing IB at coordinate: " << coord
                          << std::endl;
                return compute_ib_at_coordinate(coord, state);
            });
        }

        auto add(const auto& other) { return *this + other; }
        std::uint64_t size() const { return domain_.size(); }
    };

    // for when there are no sources
    template <typename T, std::uint64_t Dims>
    class zero_source_expr_t
    {
        index_space_t<Dims> domain_;

      public:
        zero_source_expr_t(const index_space_t<Dims>& domain) : domain_(domain)
        {
        }

        auto realize() const
        {
            auto result = field_t<T, Dims>{
              domain_,
              std::make_shared<unified_memory_t<T>>(domain_.size())
            };

            // zero-fill
            std::fill(result.data(), result.data() + result.size(), T{});
            return result;
        }

        template <typename Other>
        auto add(const Other& other) const
            -> add_expr_t<zero_source_expr_t, Other>
        {
            return add_expr_t<zero_source_expr_t, Other>{*this, other};
        }

        std::uint64_t size() const { return domain_.size(); }
    };

    // function-based generator
    template <typename Generator>
    class generator_range_t
    {
        Generator gen_;
        std::uint64_t max_size_;

      public:
        using value_type = std::invoke_result_t<Generator, std::uint64_t>;

        constexpr generator_range_t(
            Generator gen,
            std::uint64_t max_size = ~0ULL
        )
            : gen_(std::move(gen)), max_size_(max_size)
        {
        }

        constexpr auto operator[](std::uint64_t idx) const { return gen_(idx); }

        constexpr std::uint64_t size() const { return max_size_; }
    };

    // identity expr_t for no-op transformations
    template <typename T>
    class identity_expr_t
    {
      private:
        const T& input_;

      public:
        using value_type = T;

        explicit identity_expr_t(const T& input) : input_(input) {}

        std::uint64_t size() const { return 1; }   // single element

        const T& data() const { return input_; }

        template <typename F>
        auto map(F&& func) const -> lazy_expr_t<identity_expr_t, F>
        {
            return lazy_expr_t<identity_expr_t, F>{
              *this,
              std::forward<F>(func)
            };
        }

        template <typename Input>
        const Input& apply_transform(const Input& input) const
        {
            return input;   // no transformation
        }
    };

    // convenience helper for starting expression chains from memory
    template <typename T>
    class memory_wrapper_t
    {
      private:
        const unified_memory_t<T>& memory_;

      public:
        using value_type = T;

        explicit memory_wrapper_t(const unified_memory_t<T>& mem) : memory_(mem)
        {
        }

        std::uint64_t size() const { return memory_.size(); }
        const T* data() const { return memory_.data(); }

        template <typename F>
        auto map(F&& func) -> lazy_expr_t<memory_wrapper_t, F>
        {
            return lazy_expr_t<memory_wrapper_t, F>{
              *this,
              std::forward<F>(func)
            };
        }

        template <typename Input>
        const Input& apply_transform(const Input& input) const
        {
            return input;
        }
    };

    template <typename T>
    auto make_lazy(const unified_memory_t<T>& memory)
    {
        return memory_wrapper_t<T>{memory};
    }

    template <std::uint64_t Dims, typename FluxFunction>
    auto
    make_flux_accumulator(const index_space_t<Dims>& domain, FluxFunction func)
    {
        return dimension_fold_expr_t<Dims, FluxFunction>{
          domain,
          std::move(func)
        };
    }

    template <typename HydroState>
    auto make_flux_accumulator(const HydroState& state)
    {
        constexpr auto dims     = HydroState::dimensions;
        const auto& mesh_config = state.geom_solver.config;
        const auto domain       = make_space<dims>(mesh_config.shape);

        return dimension_fold_expr_t<
            dims,
            decltype([&](std::uint64_t dim) {
                return compute_flux_difference_for_dimension(state, dim);
            })>{domain, [&](std::uint64_t dim) {
                    return compute_flux_difference_for_dimension(state, dim);
                }};
    }

    template <typename T, std::uint64_t Dims>
    auto zeros_like(const index_space_t<Dims>& domain)
    {
        return zero_expr_t<T, Dims>{domain};
    }

    template <typename T, std::uint64_t Dims>
    auto make_zero_source(const index_space_t<Dims>& domain)
    {
        return zero_source_expr_t<T, Dims>{domain};
    }

    template <typename Tag, typename HydroState>
    auto make_source(const HydroState& state)
    {
        const auto& mesh_config = state.geom_solver.config;
        constexpr auto dims     = HydroState::dimensions;
        const auto domain       = make_space<dims>(mesh_config.shape);
        if constexpr (std::same_as<Tag, gravity_source_tag>) {
            return source_expr_t<dims, HydroState, gravity_source_tag>{
              state.sources.gravity_source,
              domain,
              state,
              state.sources.gravity_source.enabled
            };
        }
        else if constexpr (std::same_as<Tag, hydro_source_tag>) {
            return source_expr_t<dims, HydroState, gravity_source_tag>{
              state.sources.gravity_source,
              domain,
              state,
              state.sources.gravity_source.enabled
            };
        }
        else if constexpr (std::is_same_v<Tag, ib_source_tag>) {
            // new ib source case
            return ib_source_expr_t<HydroState>{
              domain,
              state,
            };
        }
        else {
            return make_zero_source<HydroState::dimensions>(domain);
        }
    }

    template <typename HydroState>
    auto with_gravity(const HydroState& state)
    {
        return make_source<gravity_source_tag>(state);
    }

    template <typename HydroState>
    auto with_hydro(const HydroState& state)
    {
        return make_source<hydro_source_tag>(state);
    }

    template <typename HydroState>
    auto with_ib(const HydroState& state)
    {
        return make_source<ib_source_tag>(state);
    }

}   // namespace simbi::expr

#endif   // SIMBI_LAZY_EXPR_HPP
