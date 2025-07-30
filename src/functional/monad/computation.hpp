#ifndef COMPUTATION_HPP
#define COMPUTATION_HPP

#include <cstddef>

namespace simbi {
    template <typename T>
    struct computation_t {
        T& state_;

        static computation_t from_ref(T& state) { return {state}; }

        template <typename F>
        auto then(F&& f) &&
        {
            f(state_);
            return computation_t{state_};
        }

        template <typename Pred, typename F>
        auto when(Pred&& pred, F&& f) &&
        {
            if (pred(state_)) {
                f(state_);
            }
            return computation_t{state_};
        }

        void run() && { /* */ }
    };

    template <typename T>
    struct io_computation_t {
        const T& state_;   // observe only, never mutate

        static io_computation_t from_ref(const T& state) { return {state}; }

        template <typename F>
        auto then(F&& f) &&
        {
            f(state_);
            return io_computation_t{state_};
        }

        template <typename Pred, typename F>
        auto when(Pred&& pred, F&& f) &&
        {
            if (pred(state_)) {
                f(state_);
            }
            return io_computation_t{state_};
        }

        void run() && { /* nothing to return - pure side effects */ }
    };

    template <typename T>
    auto compute(T& state)
    {
        return computation_t<T>::from_ref(state);
    }

    template <typename T>
    auto observe(const T& state)
    {
        return io_computation_t<T>::from_ref(state);
    }
}   // namespace simbi

#endif
