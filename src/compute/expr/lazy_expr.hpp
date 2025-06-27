#ifndef LAZY_EXPR_HPP
#define LAZY_EXPR_HPP

#include "containers/index_iterator.hpp"   // for coordinate_iterator_t
#include "core/base/memory.hpp"            // for unified_memory_t
#include "data/containers/vector.hpp"      // for vector_t
#include <cstddef>                         // for std::uint64_t
#include <cstdint>                         // for std::uint64_t
#include <type_traits>                     // for std::invoke_result_t

namespace simbi::expr {
    using namespace simbi::base;

    // forward declaration for chaining
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

        // materialize addition
        auto realize()
        {
            auto left_result  = left_.realize();
            auto right_result = right_.realize();

            // element-wise addition
            for (std::uint64_t ii = 0; ii < left_result.size(); ++ii) {
                left_result.data()[ii] += right_result.data()[ii];
            }

            return left_result;
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

        auto realize()
        {
            auto result = expr_.realize();

            // element-wise scaling
            for (std::uint64_t ii = 0; ii < result.size(); ++ii) {
                result.data()[ii] *= factor_;
            }

            return result;
        }
    };

    // my core lazy expression class
    template <typename Source, typename Transform>
    class lazy_expr_t
    {
      private:
        const Source& source_;   // reference to source data (zero copy! :] )
        Transform transform_;    // stored transformation function

      public:
        // construct with source + transformation
        lazy_expr_t(const Source& source, Transform transform)
            : source_(source), transform_(std::move(transform))
        {
        }

        // chain another transformation (returns new expression)
        template <typename F>
        auto map(F&& func) -> lazy_expr_t<lazy_expr_t, F>
        {
            return lazy_expr_t<lazy_expr_t, F>{*this, std::forward<F>(func)};
        }

        // execute the entire computation chain
        template <typename T = void>
        auto realize()
        {
            // deduce result type from applying transform to source element type
            using source_element_t = typename Source::value_type;
            using result_t = transform_result_t<Transform, source_element_t>;

            // create output memory with same size as source
            unified_memory_t<result_t> result(source_.size());

            // apply transformation to each element
            // (executor will handle cpu/gpu dispatch later)
            auto* input_ptr  = source_.data();
            auto* output_ptr = result.data();

            for (std::uint64_t i = 0; i < source_.size(); ++i) {
                output_ptr[i] = apply_transform(input_ptr[i]);
            }

            return result;
        }

      private:
        // apply the transformation chain recursively
        template <typename Input>
        auto apply_transform(const Input& input)
        {
            if constexpr (std::is_same_v<Source, unified_memory_t<Input>>) {
                // base case: source is raw data, apply transform directly
                return transform_(input);
            }
            else {
                // recursive case: source is another expression
                auto intermediate = source_.apply_transform(input);
                return transform_(intermediate);
            }
        }
    };

    // helper function to create expressions from unified_memory_t
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

        // start expression chain
        template <typename F>
        auto map(F&& func) -> lazy_expr_t<memory_wrapper_t, F>
        {
            return lazy_expr_t<memory_wrapper_t, F>{
              *this,
              std::forward<F>(func)
            };
        }

        // identity transform for base case
        template <typename Input>
        const Input& apply_transform(const Input& input) const
        {
            return input;
        }
    };

    // enhanced lazy_expr_t materialization for coordinate operations
    template <typename T, std::uint64_t Dims, typename Transform>
    class lazy_expr_t<coordinate_iterator_t<T, Dims>, Transform>
    {
      private:
        coordinate_iterator_t<T, Dims> source_;
        Transform transform_;

      public:
        lazy_expr_t(
            const coordinate_iterator_t<T, Dims>& source,
            Transform transform
        )
            : source_(source), transform_(std::move(transform))
        {
        }

        template <typename F>
        auto map(F&& func) -> lazy_expr_t<lazy_expr_t, F>
        {
            return lazy_expr_t<lazy_expr_t, F>{*this, std::forward<F>(func)};
        }

        // coordinate-specific materialization
        auto realize()
        {
            // deduce result type from transform applied to coordinate
            using coord_t          = vector_t<std::uint64_t, Dims>;
            using result_element_t = std::invoke_result_t<Transform, coord_t>;

            // create result collection with same dimensions as source
            auto result =
                field_t<result_element_t, Dims>{source_.collection().shape()};

            // apply transform to each coordinate in the space
            for (std::uint64_t i = 0; i < source_.size(); ++i) {
                auto coord       = source_.coordinate_at(i);
                auto transformed = transform_(coord);

                // store result at the coordinate position
                result.data()[source_.collection().linear_index(coord)] =
                    transformed;
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

        auto realize()
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

        std::uint64_t size() const { return source_.size(); }
    };

    // convenience function to start expression chains
    template <typename T>
    auto make_lazy(const unified_memory_t<T>& memory)
    {
        return memory_wrapper_t<T>{memory};
    }

}   // namespace simbi::expr

#endif   // LAZY_EXPR_HPP
