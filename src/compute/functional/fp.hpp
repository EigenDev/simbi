#ifndef SIMBI_FP_MINIMAL_HPP
#define SIMBI_FP_MINIMAL_HPP

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iterator>
#include <type_traits>
#include <utility>
#include <vector>

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

            constexpr T operator*() const { return current_; }

            constexpr iterator& operator++()
            {
                current_ += step_;
                return *this;
            }

            constexpr iterator operator++(int)
            {
                auto temp = *this;
                ++(*this);
                return temp;
            }

            constexpr bool operator==(const iterator& other) const
            {
                return current_ >= end_ || current_ == other.current_;
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

            constexpr value_type operator*() const { return (*gen_)(index_); }

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
        constexpr map_view_t(Source source, Func func)
            : source_(std::move(source)), func_(std::move(func))
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

            constexpr reference operator*() const
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
        constexpr filter_view_t(Source source, Pred pred)
            : source_(std::move(source)), pred_(std::move(pred))
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

            constexpr reference operator*() const { return *it_; }
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
        constexpr zip_view_t(First first, Second second)
            : first_(std::move(first)), second_(std::move(second))
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
                return first_it_ ==
                       other.first_it_ &&
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
        constexpr take_view_t(Source source, std::size_t count)
            : source_(std::move(source)), count_(count)
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

            constexpr reference operator*() const { return *it_; }
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
                else if constexpr (requires {
                                       result.insert(
                                           result.end(),
                                           source_value_type{}
                                       );
                                   }) {
                    // associative containers
                    for (auto&& item : source) {
                        result.insert(result.end(), item);
                    }
                }
                else if constexpr (requires {
                                       result[0] = source_value_type{};
                                       result.size();
                                   }) {
                    // fixed-size indexable containers (your vector_t,
                    // std::array)
                    std::size_t idx = 0;
                    for (auto&& item : source) {
                        if (idx >= result.size()) {
                            break;   // prevent overflow
                        }
                        result[idx++] = item;
                    }
                }
                else {
                    static_assert(
                        false,
                        "Container must support push_back, insert, or indexing "
                        "with size()"
                    );
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

    template <typename Domain, typename Func>
    struct transform_domain_view_t {
        Domain domain_;
        Func func_;

        class iterator
        {
            typename Domain::iterator domain_it_;
            const Domain* domain_;
            const Func* func_;

          public:
            using iterator_category = std::forward_iterator_tag;
            using value_type        = std::pair<
                       typename Domain::iterator::value_type,
                       std::invoke_result_t<
                           Func,
                           typename Domain::iterator::value_type,
                           Domain>>;
            using difference_type = std::ptrdiff_t;
            using reference       = value_type;

            iterator() : domain_it_{}, domain_{nullptr}, func_{nullptr} {}
            iterator(
                typename Domain::iterator it,
                const Domain* dom,
                const Func* f
            )
                : domain_it_(it), domain_{dom}, func_(f)
            {
            }

            value_type operator*() const
            {
                auto coord = *domain_it_;
                return {coord, func_->apply(coord, *domain_)};
            }

            constexpr iterator& operator++()
            {
                ++domain_it_;
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
                return domain_it_ == other.domain_it_;
            }
            constexpr bool operator!=(const iterator& other) const
            {
                return !(*this == other);
            }
        };

        auto begin() const
        {
            return iterator{domain_.begin(), &domain_, &func_};
        }
        auto end() const { return iterator{domain_.end(), &domain_, &func_}; }

        template <typename Op>
        auto operator|(Op&& op) const
        {
            return std::forward<Op>(op)(*this);
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
        constexpr auto operator()(Range&& range) const
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
        constexpr auto operator()(Range&& range) const
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

    template <typename Func>
    auto transform_domain(Func&& func)
    {
        return [func = std::forward<Func>(func)](auto&& domain) {
            return transform_domain_view_t<
                std::decay_t<decltype(domain)>,
                Func>(std::forward<decltype(domain)>(domain), func);
        };
    }

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
    constexpr auto collect = collect_t<Container>{};

    constexpr auto sum     = sum_fn_t{};
    constexpr auto product = product_fn_t{};

    template <typename Pred>
    constexpr auto any_of(Pred&& pred)
    {
        return any_of_fn_t<std::decay_t<Pred>>(std::forward<Pred>(pred));
    }

    template <typename Pred>
    constexpr auto all_of(Pred&& pred)
    {
        return all_of_fn_t<std::decay_t<Pred>>(std::forward<Pred>(pred));
    }

    template <typename Pred>
    constexpr auto none_of(Pred&& pred)
    {
        return none_of_fn_t<std::decay_t<Pred>>(std::forward<Pred>(pred));
    }

    // ========================================================================
    // convenience helpers
    // ========================================================================

    // unpack_map for tuples/pairs
    template <typename F>
    constexpr auto unpack_map(F&& func)
    {
        return map([func = std::forward<F>(func)](const auto& tuple) {
            return std::apply(func, tuple);
        });
    }

    // binary zip for convenience
    template <iterable First, iterable Second>
    constexpr auto zip(First&& first, Second&& second)
    {
        return zip_view_t<std::decay_t<First>, std::decay_t<Second>>(
            std::forward<First>(first),
            std::forward<Second>(second)
        );
    }
}   // namespace simbi::fp

#endif   // SIMBI_FP_MINIMAL_HPP
