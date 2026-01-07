#ifndef FP_TOOLKIT_HPP
#define FP_TOOLKIT_HPP

// =============================================================================
// fp.hpp
//
// functional programming toolkit for simbi
// designed to work with nvcc's limited template deduction
//
// key design principles:
//   1. computable protocol: explicit type metadata to avoid invoke_result_t
//   2. single operator() overloads: no constraint-based overload sets
//   3. pure functions: no mutable state unless absolutely necessary
//   4. nvcc-safe: avoid deep template nesting, heavy STL in device code
//
// computable protocol:
//   types that provide explicit metadata for composition:
//     - argument_type: input type
//     - value_type: output type
//     - rank: spatial dimensionality (for grid operations)
//
// usage:
//   auto pipeline = compose(f, g, h);  // variadic compose
//   auto combined = zip(f, g, add_op); // element-wise combination
//   auto selected = select(pred, a, b); // conditional application
// =============================================================================

#include "base/concepts.hpp"
#include "compat.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iterator>
#include <tuple>
#include <type_traits>
#include <utility>

// cpu-only features (iterators, views)
#ifndef __CUDA_ARCH__
#include <vector>
#endif

namespace simbi {
    template <typename T, std::uint64_t Rank>
    struct vector_t;

    template <std::uint64_t Rank>
    using coordinate_t = vector_t<std::int64_t, Rank>;

    template <std::uint64_t Rank>
    struct domain_t;
} // namespace simbi

namespace simbi::fp {

    // =========================================================================
    // concepts
    // =========================================================================

    template <typename T>
    concept iterable = requires(T& t) {
        { std::begin(t) } -> std::input_iterator;
        { std::end(t) } -> std::sentinel_for<decltype(std::begin(t))>;
    };

    // =========================================================================
    // composition: compose(f, g, h)(x) = f(g(h(x)))
    // =========================================================================

    // SFINAE-friendly type extraction helpers
    template <typename T>
    struct extract_argument_type
    {
        using type = void;
    };

    template <concepts::computable T>
    struct extract_argument_type<T>
    {
        using type = typename T::argument_type;
    };

    template <typename T>
    struct extract_value_type
    {
        using type = void;
    };

    template <concepts::computable T>
    struct extract_value_type<T>
    {
        using type = typename T::value_type;
    };

    template <typename T>
    struct extract_rank
    {
        static constexpr std::uint64_t value = 0;
    };

    template <concepts::computable T>
    struct extract_rank<T>
    {
        static constexpr std::uint64_t value = T::rank;
    };

    // binary compose: computable if both F and G are computable with matching types
    template <typename F, typename G>
    struct compose_t
    {
        F f;
        G g;

        // check if this composition is computable
        static constexpr bool is_computable = [] {
            if constexpr (concepts::computable<F> && concepts::computable<G>) {
                return std::same_as<typename F::argument_type, typename G::value_type>;
            }
            else {
                return false;
            }
        }();

        // provide protocol types only if computable (SFINAE-safe)
        using argument_type                 = typename extract_argument_type<G>::type;
        using value_type                    = typename extract_value_type<F>::type;
        static constexpr std::uint64_t rank = extract_rank<G>::value;

        // single operator() - no overload ambiguity
        template <typename Arg>
        constexpr DUAL auto operator()(Arg&& arg) const
        {
            return f(g(std::forward<Arg>(arg)));
        }
    };

    // variadic compose: base case
    template <typename F>
    constexpr auto compose(F f)
    {
        return f;
    }

    // variadic compose: recursive case
    template <typename F, typename G, typename... Rest>
    constexpr auto compose(F f, G g, Rest... rest)
    {
        if constexpr (sizeof...(Rest) == 0) {
            return compose_t<F, G>{f, g};
        }
        else {
            return compose_t<F, decltype(compose(g, rest...))>{f, compose(g, rest...)};
        }
    }

    // =========================================================================
    // partial application: partial(f, args...)(x) = f(args..., x)
    // =========================================================================

    template <typename F, typename... BoundArgs>
    struct partial_t
    {
        F                        f;
        std::tuple<BoundArgs...> bound_args;

        template <typename... Args>
        constexpr DUAL auto operator()(Args&&... args) const
        {
            return std::apply(
                [&](const auto&... bound) { return f(bound..., std::forward<Args>(args)...); },
                bound_args
            );
        }
    };

    template <typename F, typename... Args>
    constexpr auto partial(F f, Args... args)
    {
        return partial_t<F, Args...>{f, std::make_tuple(args...)};
    }

    // =========================================================================
    // selection: select(pred, a, b)(x) = pred(x) ? a(x) : b(x)
    // =========================================================================

    template <typename Pred, typename A, typename B>
        requires concepts::computable<A> && concepts::computable<B>
    struct select_t
    {
        using argument_type                 = typename A::argument_type;
        using value_type                    = typename A::value_type;
        static constexpr std::uint64_t rank = A::rank;

        Pred pred;
        A    a;
        B    b;

        constexpr DUAL value_type operator()(argument_type arg) const
        {
            return pred(arg) ? a(arg) : b(arg);
        }
    };

    template <typename Pred, typename A, typename B>
    constexpr auto select(Pred pred, A a, B b)
    {
        return select_t<Pred, A, B>{pred, a, b};
    }

    // =========================================================================
    // zip: zip(f, g, op)(x) = op(f(x), g(x))
    // =========================================================================

    // both F and G are computable
    template <typename F, typename G, typename BinaryOp>
        requires concepts::computable<F> && concepts::computable<G>
    struct zip_t
    {
        using argument_type                 = typename F::argument_type;
        static constexpr std::uint64_t rank = F::rank;

        // value_type: must use invoke_result_t here (unavoidable for BinaryOp)
        // but at least F and G are explicitly typed
        using value_type =
            std::invoke_result_t<BinaryOp, typename F::value_type, typename G::value_type>;

        F        f;
        G        g;
        BinaryOp op;

        constexpr DUAL value_type operator()(argument_type arg) const
        {
            return op(f(arg), g(arg));
        }
    };

    // F is not computable but G is (adapter for polymorphic functions)
    template <typename F, typename G, typename BinaryOp>
        requires(!concepts::computable<F>) && concepts::computable<G>
    struct zip_adapter_t
    {
        using argument_type                 = typename G::argument_type;
        static constexpr std::uint64_t rank = G::rank;

        using value_type = std::invoke_result_t<
            BinaryOp,
            std::invoke_result_t<F, argument_type>,
            typename G::value_type>;

        F        f;
        G        g;
        BinaryOp op;

        constexpr DUAL value_type operator()(argument_type arg) const
        {
            return op(f(arg), g(arg));
        }
    };

    // factory: dispatch to correct type
    template <typename F, typename G, typename BinaryOp>
    constexpr auto zip(F f, G g, BinaryOp op)
    {
        if constexpr (concepts::computable<F> && concepts::computable<G>) {
            return zip_t<F, G, BinaryOp>{f, g, op};
        }
        else if constexpr (concepts::computable<G>) {
            return zip_adapter_t<F, G, BinaryOp>{f, g, op};
        }
        else {
            static_assert(concepts::computable<G>, "at least G must be computable");
        }
    }

    // =========================================================================
    // mathematical operators
    // =========================================================================

    struct add_op_t
    {
        template <typename A, typename B>
        constexpr DUAL auto operator()(A&& a, B&& b) const
        {
            return std::forward<A>(a) + std::forward<B>(b);
        }
    };

    struct subtract_op_t
    {
        template <typename A, typename B>
        constexpr DUAL auto operator()(A&& a, B&& b) const
        {
            return std::forward<A>(a) - std::forward<B>(b);
        }
    };

    struct multiply_op_t
    {
        template <typename A, typename B>
        constexpr DUAL auto operator()(A&& a, B&& b) const
        {
            return std::forward<A>(a) * std::forward<B>(b);
        }
    };

    struct divide_op_t
    {
        template <typename A, typename B>
        constexpr DUAL auto operator()(A&& a, B&& b) const
        {
            return std::forward<A>(a) / std::forward<B>(b);
        }
    };

    inline constexpr add_op_t      add_op{};
    inline constexpr subtract_op_t subtract_op{};
    inline constexpr multiply_op_t multiply_op{};
    inline constexpr divide_op_t   divide_op{};

    // =========================================================================
    // coordinate utilities (computable protocol)
    // =========================================================================

    template <std::uint64_t Rank>
    struct offset_transform_t
    {
        using value_type                    = coordinate_t<Rank>;
        using argument_type                 = coordinate_t<Rank>;
        static constexpr std::uint64_t rank = Rank;

        coordinate_t<Rank> offset;

        constexpr DUAL value_type operator()(argument_type coord) const
        {
            return offset + coord;
        }
    };

    template <std::uint64_t Rank>
    constexpr auto offset_transform(coordinate_t<Rank> offset)
    {
        return offset_transform_t<Rank>{offset};
    }

    template <std::uint64_t Rank>
    struct domain_predicate_t
    {
        using value_type                    = bool;
        using argument_type                 = coordinate_t<Rank>;
        static constexpr std::uint64_t rank = Rank;

        domain_t<Rank> domain;

        constexpr DUAL value_type operator()(argument_type coord) const
        {
            return domain.contains(coord);
        }
    };

    template <std::uint64_t Rank>
    constexpr auto domain_predicate(domain_t<Rank> domain)
    {
        return domain_predicate_t<Rank>{domain};
    }

    template <std::uint64_t Rank>
    struct contains_op_t
    {
        using value_type                    = bool;
        using argument_type                 = coordinate_t<Rank>;
        static constexpr std::uint64_t rank = Rank;

        domain_t<Rank> domain;

        constexpr DUAL value_type operator()(argument_type coord) const
        {
            return domain.contains(coord);
        }
    };

    template <std::uint64_t Rank>
    constexpr auto contains_op(domain_t<Rank> domain)
    {
        return contains_op_t<Rank>{domain};
    }

    // =========================================================================
    // identity and constants
    // =========================================================================

    struct identity_t
    {
        template <typename T>
        constexpr DUAL auto operator()(T&& t) const -> decltype(auto)
        {
            return std::forward<T>(t);
        }
    };

    template <typename T>
    struct constant_t
    {
        T value;

        template <typename Arg>
        constexpr DUAL const T& operator()(Arg&&) const
        {
            return value;
        }
    };

    inline constexpr identity_t identity{};

    template <typename T>
    constexpr auto constant(T value)
    {
        return constant_t<T>{value};
    }

    // =========================================================================
    // device-compatible fold/reduce (requires known types)
    // =========================================================================

    // fold with explicit init value
    template <typename BinaryOp, typename Init, iterable Source>
    constexpr DUAL auto fold(BinaryOp op, Init init, Source&& source)
    {
        auto result = init;
        for (auto&& item : source) {
            result = op(result, std::forward<decltype(item)>(item));
        }
        return result;
    }

    // reduce without init (uses first element)
    template <typename BinaryOp, iterable Source>
    constexpr DUAL auto reduce(BinaryOp op, Source&& source)
    {
        auto begin = std::begin(source);
        auto end   = std::end(source);

        using value_type = std::decay_t<decltype(*begin)>;

        if (begin == end) {
            return value_type{};
        }

        auto result = *begin;
        ++begin;

        for (; begin != end; ++begin) {
            result = op(result, *begin);
        }

        return result;
    }

    // =========================================================================
    // cpu-only iterator-based views
    // these use std::iterator concepts and won't work in device code
    // =========================================================================

#ifndef __CUDA_ARCH__

    // -------------------------------------------------------------------------
    // integer_range_t
    // -------------------------------------------------------------------------

    template <typename T = std::uint64_t>
    struct integer_range_t
    {
        T start_;
        T end_;
        T step_;

        struct iterator
        {
            T current_;
            T end_;
            T step_;

            using difference_type   = std::ptrdiff_t;
            using value_type        = T;
            using pointer           = const T*;
            using reference         = T;
            using iterator_category = std::input_iterator_tag;

            constexpr iterator() : current_{}, end_{}, step_{1} {}

            constexpr iterator(T current, T end, T step) : current_{current}, end_{end}, step_{step}
            {
            }

            constexpr T operator*() const
            {
                return current_;
            }

            constexpr iterator& operator++()
            {
                current_ += step_;
                return *this;
            }

            constexpr iterator operator++(int)
            {
                auto tmp = *this;
                ++(*this);
                return tmp;
            }

            constexpr bool operator==(const iterator& other) const
            {
                return current_ >= end_ && other.current_ >= other.end_;
            }

            constexpr bool operator!=(const iterator& other) const
            {
                return !(*this == other);
            }
        };

        constexpr iterator begin() const
        {
            return iterator{start_, end_, step_};
        }
        constexpr iterator end() const
        {
            return iterator{end_, end_, step_};
        }

        template <typename F>
        constexpr auto operator|(F&& fn) const
        {
            return fn(*this);
        }
    };

    // -------------------------------------------------------------------------
    // generator_view_t
    // -------------------------------------------------------------------------

    template <typename Generator>
    struct generator_view_t
    {
        Generator gen_;

        constexpr generator_view_t(Generator gen) : gen_{gen} {}

        struct iterator
        {
            const Generator* gen_;
            std::uint64_t    index_;

            using difference_type   = std::ptrdiff_t;
            using value_type        = std::invoke_result_t<Generator, std::uint64_t>;
            using pointer           = const value_type*;
            using reference         = value_type;
            using iterator_category = std::input_iterator_tag;

            constexpr iterator() : gen_{nullptr}, index_{0} {}

            constexpr iterator(const Generator* gen, std::uint64_t index) : gen_{gen}, index_{index}
            {
            }

            constexpr value_type operator*() const
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
                auto tmp = *this;
                ++(*this);
                return tmp;
            }

            constexpr bool operator==(const iterator&) const
            {
                return false;
            }
            constexpr bool operator!=(const iterator&) const
            {
                return true;
            }
        };

        constexpr iterator begin() const
        {
            return iterator{&gen_, 0};
        }
        constexpr iterator end() const
        {
            return iterator{&gen_, 0};
        }

        template <typename F>
        constexpr auto operator|(F&& fn) const
        {
            return fn(*this);
        }
    };

#endif // __CUDA_ARCH__

#ifndef __CUDA_ARCH__
    // -------------------------------------------------------------------------
    // iterator-based views (cpu-only)
    // -------------------------------------------------------------------------

    // -------------------------------------------------------------------------
    // map_view_t
    // -------------------------------------------------------------------------

    template <typename Source, typename Func>
    class map_view_t
    {
        Source source_;
        Func   func_;

      public:
        constexpr map_view_t(Source source, Func func) : source_{source}, func_{func} {}

        template <typename SourceIter>
        class iterator_t
        {
            SourceIter  it_;
            const Func* func_;

          public:
            using difference_type = typename std::iterator_traits<SourceIter>::difference_type;
            using value_type =
                std::invoke_result_t<Func, typename std::iterator_traits<SourceIter>::reference>;
            using pointer           = value_type*;
            using reference         = value_type;
            using iterator_category = std::input_iterator_tag;

            constexpr iterator_t() : it_{}, func_{nullptr} {}

            constexpr iterator_t(SourceIter it, const Func* func) : it_{it}, func_{func} {}

            constexpr reference operator*() const
            {
                return (*func_)(*it_);
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
                return it_ != other.it_;
            }
        };

        constexpr auto begin()
        {
            return iterator_t<decltype(std::begin(source_))>{std::begin(source_), &func_};
        }

        constexpr auto end()
        {
            return iterator_t<decltype(std::end(source_))>{std::end(source_), &func_};
        }

        template <typename F>
        constexpr auto operator|(F&& fn)
        {
            return fn(*this);
        }
    };

    // -------------------------------------------------------------------------
    // filter_view_t
    // -------------------------------------------------------------------------

    template <typename Source, typename Pred>
    class filter_view_t
    {
        Source source_;
        Pred   pred_;

      public:
        constexpr filter_view_t(Source source, Pred pred) : source_{source}, pred_{pred} {}

        template <typename SourceIter>
        class iterator_t
        {
            SourceIter  it_;
            SourceIter  end_;
            const Pred* pred_;

            constexpr void skip()
            {
                while (it_ != end_ && !(*pred_)(*it_)) {
                    ++it_;
                }
            }

          public:
            using difference_type   = typename std::iterator_traits<SourceIter>::difference_type;
            using value_type        = typename std::iterator_traits<SourceIter>::value_type;
            using pointer           = typename std::iterator_traits<SourceIter>::pointer;
            using reference         = typename std::iterator_traits<SourceIter>::reference;
            using iterator_category = std::input_iterator_tag;

            constexpr iterator_t() : it_{}, end_{}, pred_{nullptr} {}

            constexpr iterator_t(SourceIter it, SourceIter end, const Pred* pred)
                : it_{it}, end_{end}, pred_{pred}
            {
                skip();
            }

            constexpr reference operator*() const
            {
                return *it_;
            }

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
                return it_ != other.it_;
            }
        };

        constexpr auto begin()
        {
            return iterator_t<decltype(std::begin(source_))>{
                std::begin(source_),
                std::end(source_),
                &pred_
            };
        }

        constexpr auto end()
        {
            return iterator_t<decltype(std::end(source_))>{
                std::end(source_),
                std::end(source_),
                &pred_
            };
        }

        template <typename F>
        constexpr auto operator|(F&& fn)
        {
            return fn(*this);
        }
    };

    // -------------------------------------------------------------------------
    // zip_view_t
    // -------------------------------------------------------------------------

    template <typename First, typename Second>
    class zip_view_t
    {
        First  first_;
        Second second_;

      public:
        constexpr zip_view_t(First first, Second second) : first_{first}, second_{second} {}

        template <typename FirstIter, typename SecondIter>
        class iterator_t
        {
            FirstIter  first_it_;
            SecondIter second_it_;

          public:
            using difference_type = std::ptrdiff_t;
            using value_type      = std::pair<
                     typename std::iterator_traits<FirstIter>::value_type,
                     typename std::iterator_traits<SecondIter>::value_type>;
            using pointer           = value_type*;
            using reference         = value_type;
            using iterator_category = std::input_iterator_tag;

            constexpr iterator_t() : first_it_{}, second_it_{} {}

            constexpr iterator_t(FirstIter first_it, SecondIter second_it)
                : first_it_{first_it}, second_it_{second_it}
            {
            }

            constexpr reference operator*() const
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
                return first_it_ == other.first_it_;
            }

            constexpr bool operator!=(const iterator_t& other) const
            {
                return first_it_ != other.first_it_;
            }
        };

        constexpr auto begin()
        {
            return iterator_t<decltype(std::begin(first_)), decltype(std::begin(second_))>{
                std::begin(first_),
                std::begin(second_)
            };
        }

        constexpr auto end()
        {
            return iterator_t<decltype(std::end(first_)), decltype(std::end(second_))>{
                std::end(first_),
                std::end(second_)
            };
        }

        template <typename F>
        constexpr auto operator|(F&& fn)
        {
            return fn(*this);
        }
    };

    // -------------------------------------------------------------------------
    // take_view_t
    // -------------------------------------------------------------------------

    template <typename Source>
    class take_view_t
    {
        Source      source_;
        std::size_t count_;

      public:
        constexpr take_view_t(Source source, std::size_t count) : source_{source}, count_{count} {}

        template <typename SourceIter>
        class iterator_t
        {
            SourceIter  it_;
            std::size_t remaining_;

          public:
            using difference_type   = typename std::iterator_traits<SourceIter>::difference_type;
            using value_type        = typename std::iterator_traits<SourceIter>::value_type;
            using pointer           = typename std::iterator_traits<SourceIter>::pointer;
            using reference         = typename std::iterator_traits<SourceIter>::reference;
            using iterator_category = std::input_iterator_tag;

            constexpr iterator_t() : it_{}, remaining_{0} {}

            constexpr iterator_t(SourceIter it, std::size_t remaining)
                : it_{it}, remaining_{remaining}
            {
            }

            constexpr reference operator*() const
            {
                return *it_;
            }

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

        constexpr auto begin()
        {
            return iterator_t<decltype(std::begin(source_))>{std::begin(source_), count_};
        }

        constexpr auto end()
        {
            return iterator_t<decltype(std::end(source_))>{std::end(source_), 0};
        }

        template <typename F>
        constexpr auto operator|(F&& fn)
        {
            return fn(*this);
        }
    };

    // -------------------------------------------------------------------------
    // collect_t (terminal operation)
    // -------------------------------------------------------------------------

    template <typename Container = void>
    struct collect_t
    {
        template <iterable Source>
        constexpr auto operator()(Source&& source) const
        {
            using value_type = std::decay_t<decltype(*std::begin(source))>;

            if constexpr (std::is_same_v<Container, void>) {
                // auto-deduce container type
                std::vector<value_type> result;
                for (auto&& item : source) {
                    result.push_back(std::forward<decltype(item)>(item));
                }
                return result;
            }
            else {
                // use explicit container type
                Container   result{};
                std::size_t idx = 0;
                for (auto&& item : source) {
                    if constexpr (requires { result.push_back(value_type{}); }) {
                        result.push_back(std::forward<decltype(item)>(item));
                    }
                    else if constexpr (requires { result[0] = value_type{}; }) {
                        // aggregate type with subscript operator (like vector_t)
                        result[idx++] = std::forward<decltype(item)>(item);
                    }
                    else {
                        static_assert(
                            sizeof(Container) == 0,
                            "Container must support push_back or subscript operator"
                        );
                    }
                }
                return result;
            }
        }
    };

    // -------------------------------------------------------------------------
    // pipeline adaptors
    // -------------------------------------------------------------------------

    template <typename F>
    struct map_fn_t
    {
        F func_;

        constexpr map_fn_t(F func) : func_{func} {}

        template <iterable Source>
        constexpr auto operator()(Source&& source) const
        {
            return map_view_t<Source, F>{std::forward<Source>(source), func_};
        }
    };

    template <typename Pred>
    struct filter_fn_t
    {
        Pred pred_;

        constexpr filter_fn_t(Pred pred) : pred_{pred} {}

        template <iterable Source>
        constexpr auto operator()(Source&& source) const
        {
            return filter_view_t<Source, Pred>{std::forward<Source>(source), pred_};
        }
    };

    template <typename Second>
    struct zip_fn_t
    {
        Second second_;

        constexpr zip_fn_t(Second second) : second_{second} {}

        template <iterable First>
        constexpr auto operator()(First&& first) const
        {
            return zip_view_t<First, Second>{std::forward<First>(first), second_};
        }
    };

    struct take_fn_t
    {
        std::size_t count_;

        constexpr take_fn_t(std::size_t count) : count_{count} {}

        template <iterable Source>
        constexpr auto operator()(Source&& source) const
        {
            return take_view_t<Source>{std::forward<Source>(source), count_};
        }
    };

    template <typename F>
    struct for_each_fn_t
    {
        F func_;

        constexpr for_each_fn_t(F func) : func_{func} {}

        template <iterable Source>
        constexpr void operator()(Source&& source) const
        {
            for (auto&& item : source) {
                func_(std::forward<decltype(item)>(item));
            }
        }
    };

    template <typename Pred>
    struct any_of_fn_t
    {
        Pred pred_;

        constexpr any_of_fn_t(Pred pred) : pred_{pred} {}

        template <iterable Source>
        constexpr bool operator()(Source&& source) const
        {
            for (auto&& item : source) {
                if (pred_(item)) {
                    return true;
                }
            }
            return false;
        }
    };

    template <typename Pred>
    struct all_of_fn_t
    {
        Pred pred_;

        constexpr all_of_fn_t(Pred pred) : pred_{pred} {}

        template <iterable Source>
        constexpr bool operator()(Source&& source) const
        {
            for (auto&& item : source) {
                if (!pred_(item)) {
                    return false;
                }
            }
            return true;
        }
    };

    template <typename Pred>
    struct none_of_fn_t
    {
        Pred pred_;

        constexpr none_of_fn_t(Pred pred) : pred_{pred} {}

        template <iterable Source>
        constexpr bool operator()(Source&& source) const
        {
            for (auto&& item : source) {
                if (pred_(item)) {
                    return false;
                }
            }
            return true;
        }
    };

    struct sum_fn_t
    {
        template <iterable Source>
        constexpr auto operator()(Source&& source) const
        {
            using value_type = std::decay_t<decltype(*std::begin(source))>;
            value_type sum{};

            for (auto&& item : source) {
                sum += item;
            }

            return sum;
        }
    };

    struct product_fn_t
    {
        template <iterable Source>
        constexpr auto operator()(Source&& source) const
        {
            using value_type = std::decay_t<decltype(*std::begin(source))>;
            value_type product{1};

            for (auto&& item : source) {
                product *= item;
            }

            return product;
        }
    };

    // -------------------------------------------------------------------------
    // range generators
    // -------------------------------------------------------------------------

    template <typename T = std::uint64_t>
    constexpr auto range(T start, T end, T step = 1)
    {
        return integer_range_t<T>{start, end, step};
    }

    template <typename T = std::uint64_t>
    constexpr auto range(T end)
    {
        return integer_range_t<T>{T{0}, end, T{1}};
    }

    template <typename Generator>
    constexpr auto generate(Generator gen)
    {
        return generator_view_t<Generator>{gen};
    }

    // -------------------------------------------------------------------------
    // pipeline factory functions
    // -------------------------------------------------------------------------

    template <typename F>
    constexpr auto map(F func)
    {
        return map_fn_t<F>{func};
    }

    template <typename Pred>
    constexpr auto filter(Pred pred)
    {
        return filter_fn_t<Pred>{pred};
    }

    template <typename Second>
    constexpr auto zip(Second second)
    {
        return zip_fn_t<Second>{second};
    }

    constexpr auto take(std::size_t count)
    {
        return take_fn_t{count};
    }

    template <typename F>
    constexpr auto for_each(F func)
    {
        return for_each_fn_t<F>{func};
    }

    template <typename Pred>
    constexpr auto any_of(Pred pred)
    {
        return any_of_fn_t<Pred>{pred};
    }

    template <typename Pred>
    constexpr auto all_of(Pred pred)
    {
        return all_of_fn_t<Pred>{pred};
    }

    template <typename Pred>
    constexpr auto none_of(Pred pred)
    {
        return none_of_fn_t<Pred>{pred};
    }

    // collect as template function (for fp::collect<T> syntax)
    template <typename Container = void>
    constexpr auto collect = collect_t<Container>{};

    inline constexpr sum_fn_t     sum{};
    inline constexpr product_fn_t product{};

    // -------------------------------------------------------------------------
    // convenience helpers
    // -------------------------------------------------------------------------

    // unpack tuples/pairs into function arguments
    template <typename F>
    constexpr auto unpack_map(F func)
    {
        return map([func](const auto& tuple) { return std::apply(func, tuple); });
    }

    // binary zip (non-pipeline version)
    template <typename First, typename Second>
    constexpr auto zip(First first, Second second)
    {
        return zip_view_t<First, Second>{first, second};
    }

#endif // __CUDA_ARCH__

} // namespace simbi::fp

#endif // FP_TOOLKIT_HPP
