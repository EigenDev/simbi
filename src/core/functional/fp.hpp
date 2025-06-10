
/**
 *  *=============================================================================
 *  *           SIMBI - Special Relativistic Magnetohydrodynamics Code
 *  *=============================================================================
 *  *
 *  * @file            functional.hpp
 *  * @brief           functional programming primitives for iterable types
 *  * @details
 *  *
 *  * @version         0.8.0
 *  * @date            2025-03-01
 *  * @author          Marcus DuPont
 *  * @email           marcus.dupont@princeton.edu
 *  *
 *  *==============================================================================
 *  */

#ifndef FUNCTIONAL_PROGRAMMING_HPP
#define FUNCTIONAL_PROGRAMMING_HPP

#include "adapter/device_adapter_api.hpp"
#include "config.hpp"
#include "core/types/containers/ndarray.hpp"
#include "core/types/utility/enums.hpp"
#include "util/parallel/exec_policy.hpp"
#include <concepts>
#include <functional>
#include <type_traits>
#include <utility>

namespace simbi {
    // forward declarations
    template <typename T, size_type Dims, VectorType Type>
    class Vector;
}   // namespace simbi
namespace simbi::detail {
    // Primary template (fallback)
    template <typename Container, typename NewValueType>
    struct rebind_container {
        // assuming some standard container type
        using type = Container;
    };

    // specialization for Vectors lass
    template <
        template <typename, size_type, VectorType> typename Vec,
        typename T,
        size_type Dims,
        VectorType Type,
        typename NewValueType>
    struct rebind_container<Vec<T, Dims, Type>, NewValueType> {
        using type = Vector<NewValueType, Dims, Type>;
    };

    // specialization for Vectors with const T. We make it non-const to allow
    // modification of the elements
    template <
        template <typename, size_type, VectorType> typename Vec,
        typename T,
        size_type Dims,
        VectorType Type,
        typename NewValueType>
    struct rebind_container<Vec<const T, Dims, Type>, NewValueType> {
        using type = Vector<NewValueType, Dims, Type>;
    };

    // helper alias for rebind_container
    template <typename Container, typename NewValueType>
    using rebind_container_t = typename rebind_container<
        std::remove_cvref_t<Container>,
        NewValueType>::type;

    // TODO: add more specializations later
}   // namespace simbi::detail

namespace simbi::fp {

    // concepts to constrain function inputs

    // concept for iterable types that support begin/end
    template <typename T>
    concept Iterable = requires(T t) {
        { std::begin(t) } -> std::input_iterator;
        { std::end(t) } -> std::sentinel_for<decltype(std::begin(t))>;
    };

    // concept for containers that support indexing and size
    template <typename T>
    concept Indexable = requires(T t, size_type i) {
        { t[i] } -> std::convertible_to<typename T::value_type>;
        { t.size() } -> std::convertible_to<size_type>;
    };

    // concept for containers that support both iterating and indexing
    template <typename T>
    concept Container = Iterable<T> && Indexable<T>;

    // -------------------------------------------------------------
    // core functional operations
    // -------------------------------------------------------------

    // invoke: a cpu/gpu compatible implementation that works like
    // std::invoke
    // template <typename F, typename... Args>
    // DUAL constexpr auto invoke(F&& f, Args&&... args)
    // {
    //     if constexpr (std::is_member_function_pointer_v<
    //                       std::remove_cvref_t<F>>) {
    //         return (std::forward<F>(f)->*(std::forward<Args>(args)...));
    //     }
    //     else {
    //         return std::forward<F>(f)(std::forward<Args>(args)...);
    //     }
    // }

    // map: transform each element with a function, returning a new
    // container of the same type
    template <Container C, typename F>
    DUAL constexpr auto map(const C& container, F&& f)
    {
        using T                = typename C::value_type;
        using base_container_t = std::remove_cvref_t<C>;
        using result_t         = std::invoke_result_t<F, T>;

        // rebind the container type to the result type
        // but allow for the elements to be modified
        using result_container_t =
            typename detail::rebind_container<base_container_t, result_t>::type;

        result_container_t result{};

        for (size_type ii = 0; ii < container.size(); ++ii) {
            result[ii] = std::invoke(std::forward<F>(f), container[ii]);
        }

        return result;
    }

    // reduce: combine elements using a binary function, with optional
    // initial value
    template <Container C, typename F>
    DUAL constexpr auto
    reduce(const C& container, F&& f, typename C::value_type init = {})
    {
        using value_t  = typename C::value_type;
        value_t result = init;

        for (size_type ii = 0; ii < container.size(); ++ii) {
            result = std::invoke(std::forward<F>(f), result, container[ii]);
        }

        return result;
    }

    // fold: like reduce but with explicit initial value of possibly
    // different type
    template <Container C, typename F, typename U>
    DUAL constexpr auto fold(const C& container, F&& f, U init)
    {
        U result = init;

        for (size_type ii = 0; ii < container.size(); ++ii) {
            result = std::invoke(std::forward<F>(f), result, container[ii]);
        }

        return result;
    }

    // zip: combine two containers element-wise with a binary function
    template <Container C1, Container C2, typename F>
    DUAL constexpr auto zip(const C1& container1, const C2& container2, F&& f)
    {
        using T1       = typename C1::value_type;
        using T2       = typename C2::value_type;
        using result_t = std::invoke_result_t<F, T1, T2>;

        // if the container is const, then we need to remove the const
        // and rebind the container type to the result type
        // but allow for the elements to be modified
        using base_container_t = std::remove_cvref_t<C1>;
        using result_container_t =
            detail::rebind_container_t<base_container_t, result_t>;

        result_container_t result{};
        const size_type min_size =
            std::min(container1.size(), container2.size());

        for (size_type ii = 0; ii < min_size; ++ii) {
            result[ii] =
                std::invoke(std::forward<F>(f), container1[ii], container2[ii]);
        }

        return result;
    }

    // -------------------------------------------------------------
    // higher-order functions
    // -------------------------------------------------------------

    // compose: create a function that applies g after f
    template <typename F, typename G>
    DUAL constexpr auto compose(F&& f, G&& g)
    {
        return [f = std::forward<F>(f),
                g = std::forward<G>(g)]<typename... Args>(Args&&... args) {
            return g(f(std::forward<Args>(args)...));
        };
    }

    // curry: transform a function that takes multiple arguments into a
    // sequence of functions that each take a single argument
    template <typename F>
    DUAL constexpr auto curry(F&& f)
    {
        return [f = std::forward<F>(f)]<typename T>(T&& t) {
            return [f = std::forward<F>(f),
                    t = std::forward<T>(t)]<typename... Args>(Args&&... args) {
                return f(t, std::forward<Args>(args)...);
            };
        };
    }

    // -------------------------------------------------------------
    // common functional operations
    // -------------------------------------------------------------

    // sum: add all elements in a container
    template <Container C>
    DUAL constexpr auto sum(const C& container)
    {
        return reduce(container, std::plus<>{});
    }

    // product: multiply all elements in a container
    template <Container C>
    DUAL constexpr auto product(const C& container)
    {
        using value_t = typename C::value_type;
        return reduce(container, std::multiplies<>{}, value_t{1});
    }

    // any: check if any element satisfies a predicate
    template <Container C, typename F>
    DUAL constexpr bool any_of(const C& container, F&& pred)
    {
        for (size_type i = 0; i < container.size(); ++i) {
            if (std::invoke(std::forward<F>(pred), container[i])) {
                return true;
            }
        }
        return false;
    }

    // general any_of predicate combinator
    template <typename T, typename... Predicates>
    DUAL constexpr bool any_of(const T& value, Predicates&&... preds)
    {
        return (... || (std::invoke(std::forward<Predicates>(preds), value)));
    }

    // all: check if all elements satisfy a predicate
    template <Container C, typename F>
    DUAL constexpr bool all_of(const C& container, F&& pred)
    {
        for (size_type i = 0; i < container.size(); ++i) {
            if (!std::invoke(std::forward<F>(pred), container[i])) {
                return false;
            }
        }
        return true;
    }

    // all: general predicate combinator
    template <typename T, typename... Predicates>
    DUAL constexpr bool all_of(const T& value, Predicates&&... preds)
    {
        return (... && (std::invoke(std::forward<Predicates>(preds), value)));
    }

    // filter: create a new container with elements that satisfy a predicate
    template <Container C, typename F>
    DUAL auto filter(const C& container, F&& pred)
    {
        C result{};
        size_type result_idx = 0;

        for (size_type i = 0; i < container.size(); ++i) {
            if (std::invoke(std::forward<F>(pred), container[i])) {
                result[result_idx++] = container[i];
            }
        }

        // Note: For fixed-size containers like Vector, this assumes
        // we're only using the first result_idx elements
        return result;
    }

    // pipeline operator for functional composition
    template <typename T, typename F>
    DUAL constexpr auto operator|(T&& value, F&& f)
        -> decltype(std::invoke(std::forward<F>(f), std::forward<T>(value)))
    {
        return std::invoke(std::forward<F>(f), std::forward<T>(value));
    }

    // parallel reduce: reduce a range of indices using an execution policy
    template <typename T, typename F>
    T reduce(const ExecutionPolicy<>& policy, T init, F&& reduce_op)
    {
        if constexpr (platform::is_gpu) {
            ndarray<T> result(1, init);
            result.sync_to_device();
            auto result_ptr      = result.data();
            const size_type size = policy.get_full_extent();

            // define a kernel function that processes each element and
            // atomically updates the result
            parallel_for(policy, [=] DEV(size_type idx) {
                extern SHARED T shared_data[];
                T thread_val = idx < size ? reduce_op(init, idx) : init;

                // block-level reduction
                size_type tid    = get_thread_id();
                shared_data[tid] = thread_val;
                gpu::api::synchronize();

                // reduce within block
                for (size_type s = get_threads_per_block() / 2; s > 0;
                     s >>= 1) {
                    if (tid < s && idx + s < size) {
                        shared_data[tid] =
                            my_min(shared_data[tid], shared_data[tid + s]);
                    }
                    gpu::api::synchronize();
                }

                // write block result to global memory
                if (tid == 0) {
                    gpu::api::atomic_min(&result_ptr[0], shared_data[0]);
                }
            });

            // sync to ensure all threads have completed
            policy.synchronize();

            // copy result back to host
            result.sync_to_host();
            return result[0];
        }
        else {
            // cpu implementation using atomic for thread safety
            std::atomic<T> result(init);
            const size_type batch_size = policy.batch_size;
            const size_type num_batches =
                policy.get_num_batches(policy.get_full_extent());

            parallel_for(policy, num_batches, [&](size_type bid) {
                const size_type start = bid * batch_size;
                const size_type end =
                    std::min(start + batch_size, policy.get_full_extent());

                // process each element in this batch
                T local_min = init;
                for (size_type idx = start; idx < end; idx++) {
                    local_min = std::min(local_min, reduce_op(init, idx));
                }

                // atomically update the global minimum
                T current_min = result.load(std::memory_order_relaxed);
                while (local_min < current_min && !result.compare_exchange_weak(
                                                      current_min,
                                                      local_min,
                                                      std::memory_order_release,
                                                      std::memory_order_relaxed
                                                  )) {
                    // keep trying if the comparison failed but our value is
                    // still smaller
                }
            });

            return result.load(std::memory_order_acquire);
        }
    }

}   // namespace simbi::fp

#endif
