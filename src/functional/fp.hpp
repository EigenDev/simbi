#ifndef FP_TOOKKIT_HPP
#define FP_TOOKKIT_HPP

#include "config.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iterator>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace simbi {
    template <typename T, std::uint64_t Dims>
    struct vector_t;

    template <std::uint64_t Dims>
    using coordinate_t = vector_t<std::int64_t, Dims>;

    template <std::uint64_t Dims>
    struct domain_t;

    // allow std::vector to be piped through custom FP toolkit
    // funcs
    template <typename T, typename Op>
    constexpr auto operator|(const std::vector<T>& vec, Op&& op)
    {
        return std::forward<Op>(op)(vec);
    }
}   // namespace simbi

namespace simbi::fp {
    // ========================================================================
    // core concepts
    // ========================================================================

    template <typename T>
    concept iterable = requires(T& t) {
        { std::begin(t) } -> std::input_iterator;
        { std::end(t) } -> std::sentinel_for<decltype(std::begin(t))>;
    };

    // ========================================================================
    // core utilities
    // ========================================================================
    // ========================================================================
    // composition: (f (of) g)(x) = f(g(x))
    // ========================================================================

    template <typename F, typename G>
    struct compose_t {
        F f;
        G g;

        template <typename Arg>
        constexpr DUAL auto operator()(Arg&& arg) const -> decltype(auto)
        {
            return f(g(std::forward<Arg>(arg)));
        }
    };

    template <typename F, typename G>
    constexpr auto compose(F f, G g)
    {
        return compose_t<F, G>{std::move(f), std::move(g)};
    }

    // ========================================================================
    // partial application: curry(f, args...)(x) = f(args..., x)
    // ========================================================================

    template <typename F, typename... BoundArgs>
    struct partial_t {
        F f;
        std::tuple<BoundArgs...> bound_args;

        template <typename... Args>
        constexpr DUAL auto operator()(Args&&... args) const -> decltype(auto)
        {
            return std::apply(
                [&](auto&&... bound) -> decltype(auto) {
                    return f(bound..., std::forward<Args>(args)...);
                },
                bound_args
            );
        }
    };

    template <typename F, typename... Args>
    constexpr auto partial(F f, Args... args)
    {
        return partial_t<F, Args...>{
          std::move(f),
          std::make_tuple(std::move(args)...)
        };
    }

    // ========================================================================
    // selection: pred(x) ? a(x) : b(x)
    // ========================================================================

    template <typename Pred, typename A, typename B>
    struct select_t {
        Pred pred;
        A a;
        B b;

        template <typename Arg>
        constexpr DUAL auto operator()(Arg&& arg) const
        {
            if (pred(arg)) {
                return a(std::forward<Arg>(arg));
            }
            else {
                return b(std::forward<Arg>(arg));
            }
        }
    };

    template <typename Pred, typename A, typename B>
    constexpr auto select(Pred pred, A a, B b)
    {
        return select_t<Pred, A, B>{
          std::move(pred),
          std::move(a),
          std::move(b)
        };
    }

    // ========================================================================
    // product/zip: combine(f, g)(x) = binary_op(f(x), g(x))
    // ========================================================================

    template <typename F, typename G, typename BinaryOp>
    struct zip_t {
        F f;
        G g;
        BinaryOp op;

        template <typename Arg>
        constexpr DUAL auto operator()(Arg&& arg) const -> decltype(auto)
        {
            return op(f(arg), g(arg));
        }
    };

    template <typename F, typename G, typename BinaryOp>
    constexpr auto zip(F f, G g, BinaryOp op)
    {
        return zip_t<F, G, BinaryOp>{std::move(f), std::move(g), std::move(op)};
    }

    // ========================================================================
    // coordinate transformation: transform(f, t)(x) = f(t(x))
    // ========================================================================

    template <typename F, typename Transform>
    struct transform_t {
        F f;
        Transform t;

        template <typename Arg>
        constexpr DUAL auto operator()(Arg&& arg) const -> decltype(auto)
        {
            return f(t(std::forward<Arg>(arg)));
        }

        template <typename Arg>
        constexpr DUAL auto operator()(Arg&& arg) -> decltype(auto)
        {
            return f(t(std::forward<Arg>(arg)));
        }
    };

    template <typename F, typename Transform>
    constexpr auto transform(F f, Transform t)
    {
        return transform_t<F, Transform>{std::move(f), std::move(t)};
    }

    // ========================================================================
    // mathematical operators as function objects
    // ========================================================================

    struct add_op_t {
        template <typename A, typename B>
        constexpr DUAL auto operator()(A&& a, B&& b) const -> decltype(auto)
        {
            return std::forward<A>(a) + std::forward<B>(b);
        }
    };

    struct subtract_op_t {
        template <typename A, typename B>
        constexpr DUAL auto operator()(A&& a, B&& b) const -> decltype(auto)
        {
            return std::forward<A>(a) - std::forward<B>(b);
        }
    };

    struct multiply_op_t {
        template <typename A, typename B>
        constexpr DUAL auto operator()(A&& a, B&& b) const -> decltype(auto)
        {
            return std::forward<A>(a) * std::forward<B>(b);
        }
    };

    struct divide_op_t {
        template <typename A, typename B>
        constexpr DUAL auto operator()(A&& a, B&& b) const -> decltype(auto)
        {
            return std::forward<A>(a) / std::forward<B>(b);
        }
    };

    constexpr auto add_op      = add_op_t{};
    constexpr auto subtract_op = subtract_op_t{};
    constexpr auto multiply_op = multiply_op_t{};
    constexpr auto divide_op   = divide_op_t{};

    // ========================================================================
    // coordinate utilities
    // ========================================================================

    template <std::uint64_t Dims>
    struct offset_transform_t {
        coordinate_t<Dims> offset;

        constexpr DUAL auto operator()(coordinate_t<Dims> coord) const
        {
            return offset + coord;
        }
    };

    template <std::uint64_t Dims>
    constexpr auto offset_transform(coordinate_t<Dims> offset)
    {
        return offset_transform_t<Dims>{offset};
    }

    template <std::uint64_t Dims>
    struct domain_predicate_t {
        domain_t<Dims> domain;

        constexpr DUAL bool operator()(coordinate_t<Dims> coord) const
        {
            return domain.contains(coord);
        }
    };

    template <std::uint64_t Dims>
    constexpr auto domain_predicate(domain_t<Dims> domain)
    {
        return domain_predicate_t<Dims>{domain};
    }

    // ========================================================================
    // identity and constants
    // ========================================================================

    struct identity_t {
        template <typename T>
        constexpr DUAL auto operator()(T&& t) const -> decltype(auto)
        {
            return std::forward<T>(t);
        }
    };

    template <typename T>
    struct constant_t {
        T value;

        template <typename Arg>
        constexpr DUAL auto operator()(Arg&&) const -> const T&
        {
            return value;
        }
    };

    constexpr auto identity = identity_t{};

    template <typename T>
    constexpr auto constant(T value)
    {
        return constant_t<T>{std::move(value)};
    }

    template <typename T>
    struct default_t {
        template <typename Arg>
        constexpr DUAL T operator()(Arg&&) const
        {
            return T{};
        }
    };

    // domain operations
    template <std::uint64_t Dims>
    struct contains_op_t {
        domain_t<Dims> domain;

        constexpr DEV bool operator()(coordinate_t<Dims> coord) const
        {
            return domain.contains(coord);
        }
    };

    template <std::uint64_t Dims>
    constexpr auto contains_op(domain_t<Dims> domain)
    {
        return contains_op_t<Dims>{domain};
    }

    // ========================================================================
    // integer range generator
    // ========================================================================

    template <typename T = std::uint64_t>
    struct integer_range_t {
        T start_, end_, step_;

        class iterator
        {
            T current_, end_, step_;

          public:
            using iterator_category = std::forward_iterator_tag;
            using value_type        = T;
            using difference_type   = std::ptrdiff_t;
            using pointer           = T*;
            using reference         = T;

            constexpr iterator() : current_(0), end_(0), step_(1) {}
            constexpr iterator(T current, T end, T step)
                : current_(current), end_(end), step_(step)
            {
            }

            constexpr T operator*() const noexcept { return current_; }

            constexpr iterator& operator++() noexcept
            {
                current_ += step_;
                return *this;
            }

            constexpr iterator operator++(int) noexcept
            {
                auto temp = *this;
                ++(*this);
                return temp;
            }

            constexpr bool operator==(const iterator& other) const noexcept
            {
                return current_ >= end_ || current_ == other.current_;
            }

            constexpr bool operator!=(const iterator& other) const noexcept
            {
                return !(*this == other);
            }
        };

        constexpr iterator begin() const
        {
            return iterator{start_, end_, step_};
        }
        constexpr iterator end() const { return iterator{end_, end_, step_}; }

        template <typename Op>
        constexpr auto operator|(Op&& op) const
        {
            return std::forward<Op>(op)(*this);
        }
    };

    // ========================================================================
    // generator for infinite sequences
    // ========================================================================

    template <typename Generator>
    struct generator_view_t {
        Generator gen_;

        constexpr generator_view_t(Generator gen) : gen_(std::move(gen)) {}

        class iterator
        {
            const Generator* gen_;
            std::uint64_t index_;

          public:
            using iterator_category = std::forward_iterator_tag;
            using value_type = std::invoke_result_t<Generator, std::uint64_t>;
            using difference_type = std::ptrdiff_t;
            using pointer         = void;
            using reference       = value_type;

            constexpr iterator() : gen_(nullptr), index_(0) {}
            constexpr iterator(const Generator* gen, std::uint64_t index)
                : gen_(gen), index_(index)
            {
            }

            constexpr value_type operator*() const noexcept
            {
                return (*gen_)(index_);
            }

            constexpr iterator& operator++()
            {
                ++index_;
                return *this;
            }

            constexpr iterator operator++(int)
            {
                auto temp = *this;
                ++(*this);
                return temp;
            }

            // infinite sequence - never equal to end
            constexpr bool operator==(const iterator&) const { return false; }
            constexpr bool operator!=(const iterator&) const { return true; }
        };

        constexpr iterator begin() const { return iterator{&gen_, 0}; }
        constexpr iterator end() const { return iterator{&gen_, ~0ULL}; }

        template <typename Op>
        constexpr auto operator|(Op&& op) const
        {
            return std::forward<Op>(op)(*this);
        }
    };

    // ========================================================================
    // view implementations
    // ========================================================================

    template <iterable Source, typename Func>
    class map_view_t
    {
        Source source_;
        Func func_;

      public:
        template <typename S>
        constexpr map_view_t(S&& source, Func func)
            : source_(std::forward<S>(source)), func_(std::move(func))
        {
        }

        template <typename SourceIter>
        class iterator_t
        {
            SourceIter it_;
            const Func* func_;

          public:
            using iterator_category =
                typename std::iterator_traits<SourceIter>::iterator_category;
            using difference_type =
                typename std::iterator_traits<SourceIter>::difference_type;
            using value_type = std::invoke_result_t<
                Func,
                typename std::iterator_traits<SourceIter>::reference>;
            using reference = value_type;

            constexpr iterator_t() : it_{}, func_{nullptr} {}
            constexpr iterator_t(SourceIter it, const Func* func)
                : it_(std::move(it)), func_(func)
            {
            }

            constexpr reference operator*() const noexcept
            {
                return std::invoke(*func_, *it_);
            }
            constexpr iterator_t& operator++()
            {
                ++it_;
                return *this;
            }
            constexpr iterator_t operator++(int)
            {
                auto tmp = *this;
                ++(*this);
                return tmp;
            }
            constexpr bool operator==(const iterator_t& other) const
            {
                return it_ == other.it_;
            }
            constexpr bool operator!=(const iterator_t& other) const
            {
                return !(*this == other);
            }
        };

        constexpr auto begin() const
        {
            return iterator_t<decltype(std::begin(source_))>(
                std::begin(source_),
                &func_
            );
        }
        constexpr auto end() const
        {
            return iterator_t<decltype(std::end(source_))>(
                std::end(source_),
                &func_
            );
        }

        template <typename Op>
        constexpr auto operator|(Op&& op) const
        {
            return std::forward<Op>(op)(*this);
        }
    };

    template <iterable Source, typename Pred>
    class filter_view_t
    {
        Source source_;
        Pred pred_;

      public:
        template <typename S>
        constexpr filter_view_t(S&& source, Pred pred)
            : source_(std::forward<S>(source)), pred_(std::move(pred))
        {
        }

        template <typename SourceIter>
        class iterator_t
        {
            SourceIter it_, end_;
            const Pred* pred_;

            constexpr void skip()
            {
                while (it_ != end_ && !std::invoke(*pred_, *it_)) {
                    ++it_;
                }
            }

          public:
            using iterator_category = std::input_iterator_tag;
            using difference_type =
                typename std::iterator_traits<SourceIter>::difference_type;
            using value_type =
                typename std::iterator_traits<SourceIter>::value_type;
            using reference =
                typename std::iterator_traits<SourceIter>::reference;

            constexpr iterator_t() : it_{}, end_{}, pred_{nullptr} {}
            constexpr iterator_t(
                SourceIter it,
                SourceIter end,
                const Pred* pred
            )
                : it_(std::move(it)), end_(std::move(end)), pred_(pred)
            {
                if (pred_) {
                    skip();
                }
            }

            constexpr reference operator*() const noexcept { return *it_; }
            constexpr iterator_t& operator++()
            {
                ++it_;
                skip();
                return *this;
            }
            constexpr iterator_t operator++(int)
            {
                auto tmp = *this;
                ++(*this);
                return tmp;
            }
            constexpr bool operator==(const iterator_t& other) const
            {
                return it_ == other.it_;
            }
            constexpr bool operator!=(const iterator_t& other) const
            {
                return !(*this == other);
            }
        };

        constexpr auto begin() const
        {
            return iterator_t<decltype(std::begin(source_))>(
                std::begin(source_),
                std::end(source_),
                &pred_
            );
        }
        constexpr auto end() const
        {
            return iterator_t<decltype(std::end(source_))>(
                std::end(source_),
                std::end(source_),
                &pred_
            );
        }

        template <typename Op>
        constexpr auto operator|(Op&& op) const
        {
            return std::forward<Op>(op)(*this);
        }
    };

    template <iterable First, iterable Second>
    class zip_view_t
    {
        First first_;
        Second second_;

      public:
        template <typename F, typename S>
        constexpr zip_view_t(F&& first, S&& second)
            : first_(std::forward<F>(first)), second_(std::forward<S>(second))
        {
        }

        template <typename FirstIter, typename SecondIter>
        class iterator_t
        {
            FirstIter first_it_;
            SecondIter second_it_;

          public:
            using iterator_category = std::input_iterator_tag;
            using difference_type   = std::ptrdiff_t;
            using value_type        = std::pair<
                       typename std::iterator_traits<FirstIter>::reference,
                       typename std::iterator_traits<SecondIter>::reference>;
            using reference = value_type;

            constexpr iterator_t() : first_it_{}, second_it_{} {}
            constexpr iterator_t(FirstIter first_it, SecondIter second_it)
                : first_it_(std::move(first_it)),
                  second_it_(std::move(second_it))
            {
            }

            constexpr reference operator*() const noexcept
            {
                return {*first_it_, *second_it_};
            }
            constexpr iterator_t& operator++()
            {
                ++first_it_;
                ++second_it_;
                return *this;
            }
            constexpr iterator_t operator++(int)
            {
                auto tmp = *this;
                ++(*this);
                return tmp;
            }
            constexpr bool operator==(const iterator_t& other) const
            {
                return first_it_ == other.first_it_ &&
                       second_it_ == other.second_it_;
            }
            constexpr bool operator!=(const iterator_t& other) const
            {
                return !(*this == other);
            }
        };

        constexpr auto begin() const
        {
            return iterator_t<
                decltype(std::begin(first_)),
                decltype(std::begin(second_))>(
                std::begin(first_),
                std::begin(second_)
            );
        }
        constexpr auto end() const
        {
            return iterator_t<
                decltype(std::end(first_)),
                decltype(std::end(second_))>(
                std::end(first_),
                std::end(second_)
            );
        }

        template <typename Op>
        constexpr auto operator|(Op&& op) const
        {
            return std::forward<Op>(op)(*this);
        }
    };

    template <iterable Source>
    class take_view_t
    {
        Source source_;
        std::size_t count_;

      public:
        template <typename S>
        constexpr take_view_t(S&& source, std::size_t count)
            : source_(std::forward<S>(source)), count_(count)
        {
        }

        template <typename SourceIter>
        class iterator_t
        {
            SourceIter it_;
            std::size_t remaining_;

          public:
            using iterator_category =
                typename std::iterator_traits<SourceIter>::iterator_category;
            using difference_type =
                typename std::iterator_traits<SourceIter>::difference_type;
            using value_type =
                typename std::iterator_traits<SourceIter>::value_type;
            using reference =
                typename std::iterator_traits<SourceIter>::reference;

            constexpr iterator_t() : it_{}, remaining_{0} {}
            constexpr iterator_t(SourceIter it, std::size_t remaining)
                : it_(std::move(it)), remaining_(remaining)
            {
            }

            constexpr reference operator*() const noexcept { return *it_; }
            constexpr iterator_t& operator++()
            {
                if (remaining_ > 0) {
                    ++it_;
                    --remaining_;
                }
                return *this;
            }
            constexpr iterator_t operator++(int)
            {
                auto tmp = *this;
                ++(*this);
                return tmp;
            }
            constexpr bool operator==(const iterator_t& other) const
            {
                return remaining_ == 0 || it_ == other.it_;
            }
            constexpr bool operator!=(const iterator_t& other) const
            {
                return !(*this == other);
            }
        };

        constexpr auto begin() const
        {
            return iterator_t<decltype(std::begin(source_))>(
                std::begin(source_),
                count_
            );
        }
        constexpr auto end() const
        {
            return iterator_t<decltype(std::end(source_))>(
                std::end(source_),
                0
            );
        }

        template <typename Op>
        constexpr auto operator|(Op&& op) const
        {
            return std::forward<Op>(op)(*this);
        }
    };

    // ========================================================================
    // collection terminal
    // ========================================================================

    template <typename Container>
    struct collect_t {
        template <iterable Source>
        constexpr auto operator()(Source&& source) const
        {
            using source_value_type =
                std::decay_t<decltype(*std::begin(source))>;

            if constexpr (std::is_same_v<Container, void>) {
                // auto-deduce container type
                std::vector<source_value_type> result;
                for (auto&& item : source) {
                    result.push_back(item);
                }
                return result;
            }
            else {
                Container result{};

                if constexpr (requires {
                                  result.push_back(source_value_type{});
                              }) {
                    // dynamic containers (std::vector, std::deque, etc.)
                    if constexpr (requires { result.reserve(1); }) {
                        // reserve space if possible
                        if constexpr (requires { std::size(source); }) {
                            result.reserve(std::size(source));
                        }
                    }
                    for (auto&& item : source) {
                        result.push_back(item);
                    }
                }
                // else if constexpr (requires {
                //                        result.insert(
                //                            result.end(),
                //                            source_value_type{}
                //                        );
                //                    }) {
                //     // associative containers
                //     for (auto&& item : source) {
                //         result.insert(result.end(), item);
                //     }
                // }
                else if constexpr (requires {
                                       result[0] = source_value_type{};
                                       result.size();
                                   }) {
                    // fixed-size indexable containers
                    std::size_t idx = 0;
                    for (auto&& item : source) {
                        if (idx >= result.size()) {
                            break;   // prevent overflow
                        }
                        result[idx++] = item;
                    }
                }
                else {
                    []<bool success = false>() {
                        static_assert(
                            success,
                            "Container must support push_back, insert, or "
                            "indexing "
                            "with size()"
                        );
                    }();
                }
                return result;
            }
        }
    };

    // ========================================================================
    // function adapters
    // ========================================================================

    template <typename F>
    struct map_fn_t {
        F func_;
        constexpr explicit map_fn_t(F func) : func_(std::move(func)) {}

        template <iterable Source>
        constexpr auto operator()(Source&& source) const
        {
            return map_view_t<std::decay_t<Source>, F>(
                std::forward<Source>(source),
                func_
            );
        }
    };

    template <typename Pred>
    struct filter_fn_t {
        Pred pred_;
        constexpr explicit filter_fn_t(Pred pred) : pred_(std::move(pred)) {}

        template <iterable Source>
        constexpr auto operator()(Source&& source) const
        {
            return filter_view_t<std::decay_t<Source>, Pred>(
                std::forward<Source>(source),
                pred_
            );
        }
    };

    template <iterable Second>
    struct zip_fn_t {
        Second second_;
        constexpr explicit zip_fn_t(Second second) : second_(std::move(second))
        {
        }

        template <iterable First>
        constexpr auto operator()(First&& first) const
        {
            return zip_view_t<std::decay_t<First>, Second>(
                std::forward<First>(first),
                second_
            );
        }
    };

    struct take_fn_t {
        std::size_t count_;
        constexpr explicit take_fn_t(std::size_t count) : count_(count) {}

        template <iterable Source>
        constexpr auto operator()(Source&& source) const
        {
            return take_view_t<std::decay_t<Source>>(
                std::forward<Source>(source),
                count_
            );
        }
    };

    template <typename F>
    struct for_each_fn_t {
        F func_;
        constexpr explicit for_each_fn_t(F func) : func_(std::move(func)) {}

        template <iterable Source>
        constexpr void operator()(Source&& source) const
        {
            for (auto&& item : source) {
                std::invoke(func_, item);
            }
        }
    };

    // any_of, all_of, none_of
    template <typename Pred>
    struct any_of_fn_t {
        Pred pred_;
        constexpr explicit any_of_fn_t(Pred pred) : pred_(std::move(pred)) {}

        template <iterable Source>
        constexpr bool operator()(Source&& source) const
        {
            for (auto&& item : source) {
                if (std::invoke(pred_, item)) {
                    return true;
                }
            }
            return false;
        }
    };

    template <typename Pred>
    struct all_of_fn_t {
        Pred pred_;
        constexpr explicit all_of_fn_t(Pred pred) : pred_(std::move(pred)) {}

        template <iterable Source>
        constexpr bool operator()(Source&& source) const
        {
            for (auto&& item : source) {
                if (!std::invoke(pred_, item)) {
                    return false;
                }
            }
            return true;
        }
    };

    template <typename Pred>
    struct none_of_fn_t {
        Pred pred_;
        constexpr explicit none_of_fn_t(Pred pred) : pred_(std::move(pred)) {}

        template <iterable Source>
        constexpr bool operator()(Source&& source) const
        {
            for (auto&& item : source) {
                if (std::invoke(pred_, item)) {
                    return false;
                }
            }
            return true;
        }
    };

    // ========================================================================
    // terminals
    // ========================================================================

    struct sum_fn_t {
        template <iterable Range>
        constexpr auto operator()(Range&& range) const noexcept
        {
            auto begin = std::begin(std::forward<Range>(range));
            auto end   = std::end(std::forward<Range>(range));

            if (begin == end) {
                using value_type =
                    typename std::iterator_traits<decltype(begin)>::value_type;
                return value_type{0};
            }

            auto result = *begin;
            ++begin;
            for (; begin != end; ++begin) {
                result = result + *begin;
            }
            return result;
        }
    };

    struct product_fn_t {
        template <iterable Range>
        constexpr auto operator()(Range&& range) const noexcept
        {
            auto begin = std::begin(std::forward<Range>(range));
            auto end   = std::end(std::forward<Range>(range));

            if (begin == end) {
                using value_type =
                    typename std::iterator_traits<decltype(begin)>::value_type;
                return value_type{1};
            }

            auto result = *begin;
            ++begin;
            for (; begin != end; ++begin) {
                result = result * *begin;
            }
            return result;
        }
    };

    // ========================================================================
    // factory functions
    // ========================================================================
    constexpr auto range(std::uint64_t end)
    {
        return integer_range_t<std::uint64_t>{0, end, 1};
    }

    constexpr auto range(std::uint64_t start, std::uint64_t end)
    {
        return integer_range_t<std::uint64_t>{start, end, 1};
    }

    constexpr auto
    range(std::uint64_t start, std::uint64_t end, std::uint64_t step)
    {
        return integer_range_t<std::uint64_t>{start, end, step};
    }

    template <typename Generator>
    constexpr auto generate(Generator&& gen)
    {
        return generator_view_t<Generator>{std::forward<Generator>(gen)};
    }

    template <typename F>
    constexpr auto map(F&& func)
    {
        return map_fn_t<std::decay_t<F>>(std::forward<F>(func));
    }

    template <typename Pred>
    constexpr auto filter(Pred&& pred)
    {
        return filter_fn_t<std::decay_t<Pred>>(std::forward<Pred>(pred));
    }

    template <iterable Second>
    constexpr auto zip(Second&& second)
    {
        return zip_fn_t<std::decay_t<Second>>(std::forward<Second>(second));
    }

    constexpr auto take(std::size_t count) { return take_fn_t{count}; }

    template <typename F>
    constexpr auto for_each(F&& func)
    {
        return for_each_fn_t<std::decay_t<F>>(std::forward<F>(func));
    }

    template <typename Container = void>
    constexpr DUAL auto collect = collect_t<Container>{};

    constexpr DUAL auto sum     = sum_fn_t{};
    constexpr DUAL auto product = product_fn_t{};

    template <typename Pred>
    constexpr DUAL auto any_of(Pred&& pred)
    {
        return any_of_fn_t<std::decay_t<Pred>>(std::forward<Pred>(pred));
    }

    template <typename Pred>
    constexpr DUAL auto all_of(Pred&& pred)
    {
        return all_of_fn_t<std::decay_t<Pred>>(std::forward<Pred>(pred));
    }

    template <typename Pred>
    constexpr DUAL auto none_of(Pred&& pred)
    {
        return none_of_fn_t<std::decay_t<Pred>>(std::forward<Pred>(pred));
    }

    // ========================================================================
    // reduction operations
    // ========================================================================

    template <typename BinaryOp>
    struct reduce_fn_t {
        BinaryOp op_;
        constexpr explicit reduce_fn_t(BinaryOp op) : op_(std::move(op)) {}

        template <iterable Source>
        constexpr auto operator()(Source&& source) const
        {
            auto begin = std::begin(std::forward<Source>(source));
            auto end   = std::end(std::forward<Source>(source));

            if (begin == end) {
                using value_type =
                    typename std::iterator_traits<decltype(begin)>::value_type;
                // for empty range, return default-constructed value
                return value_type{};
            }

            auto result = *begin;
            ++begin;
            for (; begin != end; ++begin) {
                result = op_(result, *begin);
            }
            return result;
        }
    };

    // ========================================================================
    // async reduction - combines reduce + execute_async
    // ========================================================================
    template <typename Executor, typename BinaryOp>
    struct async_reduce_fn_t {
        Executor executor_;
        BinaryOp op_;

        constexpr async_reduce_fn_t(Executor executor, BinaryOp op)
            : executor_(std::move(executor)), op_(std::move(op))
        {
        }

        template <iterable Source>
        constexpr auto operator()(Source&& source) const
        {
            return executor_.async([source = std::forward<Source>(source),
                                    op     = op_]() {
                auto begin = std::begin(source);
                auto end   = std::end(source);

                if (begin == end) {
                    using value_type = typename std::iterator_traits<
                        decltype(begin)>::value_type;
                    return value_type{};
                }

                auto result = *begin;
                ++begin;
                for (; begin != end; ++begin) {
                    result = op(result, *begin);
                }
                return result;
            });
        }
    };

    template <typename Executor, typename BinaryOp>
    constexpr auto async_reduce(Executor&& executor, BinaryOp&& op)
    {
        return async_reduce_fn_t<
            std::decay_t<Executor>,
            std::decay_t<BinaryOp>>(
            std::forward<Executor>(executor),
            std::forward<BinaryOp>(op)
        );
    }

    // ========================================================================
    // factory functions
    // ========================================================================

    template <typename BinaryOp>
    constexpr DUAL auto reduce(BinaryOp&& op)
    {
        return reduce_fn_t<std::decay_t<BinaryOp>>(std::forward<BinaryOp>(op));
    }

    template <typename Executor>
    constexpr DUAL auto execute_async(Executor&& executor)
    {
        return execute_async_fn_t<std::decay_t<Executor>>(
            std::forward<Executor>(executor)
        );
    }

    // ========================================================================
    // convenience helpers
    // ========================================================================

    // unpack_map for tuples/pairs
    template <typename F>
    constexpr DUAL auto unpack_map(F&& func)
    {
        return map([func = std::forward<F>(func)](const auto& tuple) {
            return std::apply(func, tuple);
        });
    }

    // binary zip for convenience
    template <iterable First, iterable Second>
    constexpr DUAL auto zip(First&& first, Second&& second)
    {
        return zip_view_t<std::decay_t<First>, std::decay_t<Second>>(
            std::forward<First>(first),
            std::forward<Second>(second)
        );
    }
}   // namespace simbi::fp

#endif   // FP_MINIMAL_HPP
