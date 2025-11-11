#ifndef EXECUTOR_HPP
#define EXECUTOR_HPP

#include "compat.hpp"
#include "containers/vector.hpp"
#include "domain/domain.hpp"
#include "execution/completion.hpp"
#include "execution/future.hpp"
#include "functional/fp.hpp"
#include "hetero/adapter.hpp"
#include "hetero/core/primitives.hpp"
#include "hetero/device/execution_context.hpp"
#include "memory/device.hpp"
#include "thread_pool.hpp"
#include "tiling.hpp"

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <type_traits>
#include <utility>
#include <vector>

namespace simbi::exec {
    template <typename Derived>
    class executor_base_t
    {
      protected:
        Derived& derived() { return static_cast<Derived&>(*this); }
        const Derived& derived() const

        {
            return static_cast<const Derived&>(*this);
        }

      public:
        template <typename Func, typename... Args>
        auto async(Func&& func, Args&&... args) const
            -> future_t<decltype(func(args...))>
        {
            return derived().async_impl(
                std::forward<Func>(func),
                std::forward<Args>(args)...
            );
        }

        template <typename Func, typename... Args>
        auto sync(Func&& func, Args&&... args) const -> decltype(func(args...))
        {
            return async(std::forward<Func>(func), std::forward<Args>(args)...)
                .wait();
        }

        template <std::uint64_t Dims, typename Func>
        auto for_each(
            const domain_t<Dims>& domain,
            Func&& func,
            const iarray<Dims>& tile_size = {-1}
        ) const -> future_t<void>
        {
            return derived().for_each_impl(
                domain,
                std::forward<Func>(func),
                tile_size[0] == -1 ? tiling::default_tile_size<Dims>()
                                   : tile_size
            );
        }

        template <
            std::uint64_t Dims,
            typename U,
            typename Mapper,
            typename Reducer>
        auto reduce(
            const domain_t<Dims>& domain,
            U init,
            Mapper&& mapper,
            Reducer&& reducer,
            const iarray<Dims>& tile_size = {-1}
        ) const -> future_t<U>
        {
            return derived().reduce_impl(
                domain,
                init,
                std::forward<Mapper>(mapper),
                std::forward<Reducer>(reducer),
                tile_size[0] == -1 ? tiling::default_tile_size<Dims>()
                                   : tile_size
            );
        }
    };

    class cpu_executor_t : public executor_base_t<cpu_executor_t>
    {
      public:
        cpu_executor_t() = default;

        template <typename Func, typename... Args>
        auto async_impl(Func&& func, Args&&... args) const
            -> future_t<decltype(func(args...))>
        {
            using result_t = decltype(func(args...));

            auto state =
                std::make_shared<typename future_t<result_t>::future_state_t>();
            state->completion_context = completion_context_t::direct();

            try {
                if constexpr (std::is_void_v<result_t>) {
                    func(args...);
                    state->ready.store(true);
                }
                else {
                    auto result = func(args...);
                    state->construct_result(std::move(result));
                    state->ready.store(true);
                }
            }
            catch (...) {
                state->exception = std::current_exception();
                state->has_error.store(true);
                state->ready.store(true);
            }

            return future_t<result_t>{std::move(state)};
        }

        template <std::uint64_t Dims, typename Func>
        auto for_each_impl(
            const domain_t<Dims>& domain,
            Func&& func,
            const iarray<Dims>& tile_size
        ) const -> future_t<void>
        {
            return async_impl([=, this]() {
                tiling::for_each_tile(
                    domain,
                    [&](const auto& tile) {
                        iterate_domain(tile.domain, func);
                    },
                    tile_size
                );
            });
        }

        template <
            std::uint64_t Dims,
            typename T,
            typename Mapper,
            typename Reducer>
        auto reduce_impl(
            const domain_t<Dims>& domain,
            T init,
            Mapper&& mapper,
            Reducer&& reducer,
            const iarray<Dims>& tile_size
        ) const -> future_t<T>
        {
            return async_impl([=, this]() {
                T accumulator = init;
                tiling::for_each_tile(
                    domain,
                    [&](const auto& tile) {
                        iterate_domain(tile.domain, [&](auto coord) {
                            accumulator = reducer(accumulator, mapper(coord));
                        });
                    },
                    tile_size
                );
                return accumulator;
            });
        }

      private:
        template <std::uint64_t Dims, typename Func>
        void iterate_domain(const domain_t<Dims>& domain, Func func) const
        {
            if constexpr (Dims == 1) {
                for (auto ii = domain.start[0]; ii < domain.fin[0]; ++ii) {
                    func(iarray<1>{ii});
                }
            }
            else if constexpr (Dims == 2) {
                for (auto ii = domain.start[0]; ii < domain.fin[0]; ++ii) {
                    for (auto jj = domain.start[1]; jj < domain.fin[1]; ++jj) {
                        func(iarray<2>{ii, jj});
                    }
                }
            }
            else if constexpr (Dims == 3) {
                for (auto ii = domain.start[0]; ii < domain.fin[0]; ++ii) {
                    for (auto jj = domain.start[1]; jj < domain.fin[1]; ++jj) {
                        for (auto kk = domain.start[2]; kk < domain.fin[2];
                             ++kk) {
                            func(iarray<3>{ii, jj, kk});
                        }
                    }
                }
            }
        }
    };

    class par_cpu_executor_t : public executor_base_t<par_cpu_executor_t>
    {
      private:
        thread_pool_t* pool_;
        std::size_t nthreads_;

      public:
        explicit par_cpu_executor_t()
            : pool_(&thread_pool_manager_t::get_pool()),
              nthreads_(get_nthreads())
        {
        }

        template <typename Func, typename... Args>
        auto async_impl(Func&& func, Args&&... args) const
            -> future_t<decltype(func(args...))>
        {
            using result_t = decltype(func(args...));

            auto state =
                std::make_shared<typename future_t<result_t>::future_state_t>();
            state->completion_context = completion_context_t::work_stealing();

            pool_->submit(
                [state, func = std::forward<Func>(func), args...]() mutable {
                    try {
                        if constexpr (std::is_void_v<result_t>) {
                            func(args...);
                            {
                                std::lock_guard<std::mutex> lock(state->mutex);
                                state->ready.store(true);
                            }
                            state->cv.notify_one();
                        }
                        else {
                            auto result = func(args...);
                            {
                                std::lock_guard<std::mutex> lock(state->mutex);
                                state->construct_result(std::move(result));
                                state->ready.store(true);
                            }
                            state->cv.notify_one();
                        }
                    }
                    catch (...) {
                        state->exception = std::current_exception();
                        state->has_error.store(true);
                        state->ready.store(true);
                    }
                }
            );

            return future_t<result_t>{std::move(state)};
        }

        template <std::uint64_t Dims, typename Func>
        auto for_each_impl(
            const domain_t<Dims>& domain,
            Func&& func,
            const iarray<Dims>& tile_size
        ) const -> future_t<void>
        {
            return async_impl([=, this]() {
                const auto tiles = tiling::make_tiles(domain, tile_size);

                std::vector<future_t<void>> tile_futures;
                tile_futures.reserve(tiles.size());

                for (const auto& tile : tiles) {
                    tile_futures.push_back(async_impl([=, this]() {
                        iterate_domain(tile.domain, func);
                    }));
                }

                for (auto& future : tile_futures) {
                    future.wait();
                }
            });
        }

        template <
            std::uint64_t Dims,
            typename T,
            typename Mapper,
            typename Reducer>
        auto reduce_impl(
            const domain_t<Dims>& domain,
            T init,
            Mapper&& mapper,
            Reducer&& reducer,
            const iarray<Dims>& tile_size
        ) const -> future_t<T>
        {
            return async_impl([=, this]() {
                const auto tiles = tiling::make_tiles(domain, tile_size);

                std::vector<future_t<T>> tile_futures;
                tile_futures.reserve(tiles.size());

                for (const auto& tile : tiles) {
                    tile_futures.push_back(async_impl([=, this]() {
                        T tile_result = init;
                        iterate_domain(tile.domain, [&](auto coord) {
                            tile_result = reducer(tile_result, mapper(coord));
                        });
                        return tile_result;
                    }));
                }

                T final_result = init;
                for (auto& future : tile_futures) {
                    final_result = reducer(final_result, future.wait());
                }
                return final_result;
            });
        }

      private:
        template <std::uint64_t Dims, typename Func>
        void iterate_domain(const domain_t<Dims>& domain, Func func) const
        {
            if constexpr (Dims == 1) {
                for (auto ii = domain.start[0]; ii < domain.fin[0]; ++ii) {
                    func(iarray<1>{ii});
                }
            }
            else if constexpr (Dims == 2) {
                for (auto ii = domain.start[0]; ii < domain.fin[0]; ++ii) {
                    for (auto jj = domain.start[1]; jj < domain.fin[1]; ++jj) {
                        func(iarray<2>{ii, jj});
                    }
                }
            }
            else if constexpr (Dims == 3) {
                for (auto ii = domain.start[0]; ii < domain.fin[0]; ++ii) {
                    for (auto jj = domain.start[1]; jj < domain.fin[1]; ++jj) {
                        for (auto kk = domain.start[2]; kk < domain.fin[2];
                             ++kk) {
                            func(iarray<3>{ii, jj, kk});
                        }
                    }
                }
            }
        }
    };

    class omp_executor_t : public executor_base_t<omp_executor_t>
    {
      public:
        omp_executor_t() = default;

        template <typename Func, typename... Args>
        auto async_impl(Func&& func, Args&&... args) const
            -> future_t<decltype(func(args...))>
        {
            using result_t = decltype(func(args...));

            auto state =
                std::make_shared<typename future_t<result_t>::future_state_t>();
            state->completion_context = completion_context_t::direct();

            try {
                if constexpr (std::is_void_v<result_t>) {
                    func(args...);
                    state->ready.store(true);
                }
                else {
                    auto result = func(args...);
                    state->construct_result(std::move(result));
                    state->ready.store(true);
                }
            }
            catch (...) {
                state->exception = std::current_exception();
                state->has_error.store(true);
                state->ready.store(true);
            }

            return future_t<result_t>{std::move(state)};
        }

        template <std::uint64_t Dims, typename Func>
        auto for_each_impl(
            const domain_t<Dims>& domain,
            Func&& func,
            const iarray<Dims>& tile_size
        ) const -> future_t<void>
        {
            return async_impl([=, this]() {
                // iterate_domain_parallel(domain, func);
                const auto tiles = tiling::make_tiles(domain, tile_size);

#pragma omp parallel for schedule(static)
                for (std::size_t tile_idx = 0; tile_idx < tiles.size();
                     ++tile_idx) {
                    const auto& tile = tiles[tile_idx];
                    iterate_domain_serial(tile.domain, func);
                }
            });
        }

        template <
            std::uint64_t Dims,
            typename T,
            typename Mapper,
            typename Reducer>
        auto reduce_impl(
            const domain_t<Dims>& domain,
            T init,
            Mapper&& mapper,
            Reducer&& reducer,
            const iarray<Dims>& tile_size
        ) const -> future_t<T>
        {
            return async_impl([=, this]() {
                const auto tiles = tiling::make_tiles(domain, tile_size);

                if constexpr (std::is_same_v<
                                  std::decay_t<Reducer>,
                                  std::plus<T>>) {
                    T accumulator = init;
#pragma omp parallel for schedule(static) reduction(+ : accumulator)
                    for (std::size_t tile_idx = 0; tile_idx < tiles.size();
                         ++tile_idx) {
                        const auto& tile = tiles[tile_idx];
                        iterate_domain_serial(tile.domain, [&](auto coord) {
                            accumulator += mapper(coord);
                        });
                    }
                    return accumulator;
                }
                else {
                    // Generic fallback - manual reduction
                    std::vector<T> tile_results(tiles.size(), init);

#pragma omp parallel for schedule(static)
                    for (std::size_t tile_idx = 0; tile_idx < tiles.size();
                         ++tile_idx) {
                        const auto& tile = tiles[tile_idx];
                        iterate_domain_serial(tile.domain, [&](auto coord) {
                            tile_results[tile_idx] =
                                reducer(tile_results[tile_idx], mapper(coord));
                        });
                    }

                    T final_result = init;
                    for (const auto& tile_result : tile_results) {
                        final_result = reducer(final_result, tile_result);
                    }
                    return final_result;
                }
            });
        }

      private:
        template <std::uint64_t Dims, typename Func>
        void
        iterate_domain_serial(const domain_t<Dims>& domain, Func func) const
        {
            if constexpr (Dims == 1) {
                for (auto ii = domain.start[0]; ii < domain.fin[0]; ++ii) {
                    func(iarray<1>{ii});
                }
            }
            else if constexpr (Dims == 2) {
                for (auto ii = domain.start[0]; ii < domain.fin[0]; ++ii) {
                    for (auto jj = domain.start[1]; jj < domain.fin[1]; ++jj) {
                        func(iarray<2>{ii, jj});
                    }
                }
            }
            else if constexpr (Dims == 3) {
                for (auto ii = domain.start[0]; ii < domain.fin[0]; ++ii) {
                    for (auto jj = domain.start[1]; jj < domain.fin[1]; ++jj) {
                        for (auto kk = domain.start[2]; kk < domain.fin[2];
                             ++kk) {
                            func(iarray<3>{ii, jj, kk});
                        }
                    }
                }
            }
        }

        template <std::uint64_t Dims, typename Func>
        void
        iterate_domain_parallel(const domain_t<Dims>& domain, Func func) const
        {
            if constexpr (Dims == 1) {
#pragma omp parallel for schedule(static)
                for (auto ii = domain.start[0]; ii < domain.fin[0]; ++ii) {
                    func(iarray<1>{ii});
                }
            }
            else if constexpr (Dims == 2) {
#pragma omp parallel for collapse(2) schedule(static)
                for (auto ii = domain.start[0]; ii < domain.fin[0]; ++ii) {
                    for (auto jj = domain.start[1]; jj < domain.fin[1]; ++jj) {
                        func(iarray<2>{ii, jj});
                    }
                }
            }
            else if constexpr (Dims == 3) {
#pragma omp parallel for collapse(3) schedule(static)
                for (auto ii = domain.start[0]; ii < domain.fin[0]; ++ii) {
                    for (auto jj = domain.start[1]; jj < domain.fin[1]; ++jj) {
                        for (auto kk = domain.start[2]; kk < domain.fin[2];
                             ++kk) {
                            func(iarray<3>{ii, jj, kk});
                        }
                    }
                }
            }
            else {
                static_assert(Dims <= 3, "Dims must be <= 3");
            }
        }
    };

    namespace detail {
        // --- reduction kernel definitions ---

        // this first kernel performs a partial reduction on each block
        template <
            std::uint64_t Dims,
            typename U,
            typename Mapper,
            typename Reducer>
        KERNEL void reduce_kernel_part1(
            domain_t<Dims> domain,
            Mapper mapper,
            Reducer reducer,
            U init,
            std::size_t n,
            U* partial_results   // output buffer [size = grid_size]
        )
        {
            // "rehydrate" the type-safe shared memory wrapper
            extern SHARED hetero::shared_memory_t<U> shared_mem_proxy[];
            auto& shared_mem = *shared_mem_proxy;

            const auto block = hetero::this_block();
            const auto warp  = block.get_sub_group();
            const auto grid  = hetero::grid::idx();

            U thread_sum = init;

            // --- per-thread reduction (grid-stride loop) ---
            const auto global_idx    = grid.global_thread_id();
            const auto total_threads = grid.total_threads();

            for (auto ii = global_idx; ii < n; ii += total_threads) {
                thread_sum =
                    reducer(thread_sum, mapper(domain.linear_to_coord(ii)));
            }

            // --- per-block reduction (two-level: warp + shared mem) ---

            // level 1: fast warp-level reduction using shuffles
            U warp_sum = hetero::reduce(warp, thread_sum, reducer);

            // level 2: block-level reduction
            // one thread from each warp writes its partial sum to shared memory
            if (warp.is_leader()) {
                shared_mem[warp.id()] = warp_sum;
            }

            block.sync();   // wait for all warps to write

            // the first warp (warp 0) now reduces the shared memory results
            U block_sum = init;
            if (warp.id() == 0) {
                // read from shared mem, guarding against non-full warps
                U val     = (block.rank() < block.num_sub_groups())
                                ? shared_mem[block.rank()]
                                : init;
                block_sum = hetero::reduce(warp, val, reducer);
            }

            // the leader of the entire block writes the final block sum
            if (block.is_leader()) {
                partial_results[grid.block_id()] = block_sum;
            }
        }

        // this second kernel (run as one block) reduces the partial sums
        template <typename U, typename Reducer>
        KERNEL void reduce_kernel_part2(
            U* partial_results,
            U* final_result,
            std::size_t num_partials,
            Reducer reducer,
            U init
        )
        {
            extern SHARED hetero::shared_memory_t<U> shared_mem_proxy[];
            auto& shared_mem = *shared_mem_proxy;

            const auto block = hetero::this_block();
            const auto warp  = block.get_sub_group();

            U thread_sum = init;

            // each thread sums a portion of the partial results
            for (auto ii = block.rank(); ii < num_partials;
                 ii += block.size()) {
                thread_sum = reducer(thread_sum, partial_results[ii]);
            }

            // perform one final block-level reduction
            U warp_sum = hetero::reduce(warp, thread_sum, reducer);

            if (warp.is_leader()) {
                shared_mem[warp.id()] = warp_sum;
            }

            block.sync();

            U block_sum = init;
            if (warp.id() == 0) {
                U val     = (block.rank() < block.num_sub_groups())
                                ? shared_mem[block.rank()]
                                : init;
                block_sum = hetero::reduce(warp, val, reducer);
            }

            // the block leader writes the single, final answer
            if (block.is_leader()) {
                *final_result = block_sum;
            }
        }
    }   // namespace detail

    /**
     * @brief a "worker" executor that launches kernels on a *single* gpu.
     */
    class gpu_executor_t : public executor_base_t<gpu_executor_t>
    {
      private:
        mem::device_t device_;
        mutable hetero::stream stream_;

        void set_device_for_thread() const
        {
            hetero::device::set_device(device_.device_id);
        }

        template <typename SizeType>
        auto default_grid_size(SizeType n) const
        {
            constexpr std::int64_t block_size = 256;
            std::int64_t grid_size = (n + block_size - 1) / block_size;
            constexpr std::int64_t max_blocks = 65535;
            grid_size                         = std::min(grid_size, max_blocks);
            return hetero::grid::config(grid_size, block_size);
        }

      public:
        // constructs a worker for a *single* device
        explicit gpu_executor_t(std::int64_t device_id = 0)
            : device_(mem::device_t::gpu(device_id))
        {
            set_device_for_thread();
            stream_ = hetero::device::create_stream();
        }

        ~gpu_executor_t() {}

        gpu_executor_t(const gpu_executor_t&)            = delete;
        gpu_executor_t& operator=(const gpu_executor_t&) = delete;
        gpu_executor_t(gpu_executor_t&&)                 = default;
        gpu_executor_t& operator=(gpu_executor_t&&)      = default;

        mem::device_t device() const { return device_; }
        hetero::stream& stream() { return stream_; }

        void synchronize() const { stream_.synchronize(); }

        template <typename Func, typename... Args>
        auto async_impl(Func&&, Args&&...) const -> future_t<void>
        {
            throw std::runtime_error(
                "gpu_executor_t::async_impl not supported"
            );
        }

        /**
         * @brief   asynchronously executes a void function on the domain.
         * returns a future<void> that can be waited upon.
         */
        template <std::uint64_t Dims, typename Func>
        auto for_each_impl(
            const domain_t<Dims>& domain,
            Func&& func,
            const iarray<Dims>& /*tile_size*/
        ) const -> future_t<void>
        {
            set_device_for_thread();

            auto state =
                std::make_shared<typename future_t<void>::future_state_t>();

            const auto n = domain.size();
            if (n == 0) {
                state->ready.store(true);
                return future_t<void>{std::move(state)};
            }

            auto launch_config = default_grid_size(n);

            auto kernel = [domain, f = std::forward<Func>(func)] DEV() {
                const auto idx           = hetero::grid::idx();
                const auto global_idx    = idx.global_thread_id();
                const auto total_threads = idx.total_threads();

                for (auto ii = global_idx; ii < domain.size();
                     ii += total_threads) {
                    const auto coord = domain.linear_to_coord(ii);
                    f(coord);
                }
            };

            try {
                hetero::device::launch_async(kernel, launch_config, stream_);

                auto event = hetero::device::create_event();
                event.record(stream_);

                state->completion_events.push_back(std::move(event));
            }
            catch (...) {
                state->exception = std::current_exception();
                state->has_error.store(true);
                state->ready.store(true);
            }

            return future_t<void>{std::move(state)};
        }

        /**
         * @brief   synchronously executes a reduction on the domain.
         * this is *blocking* because future_t<T> has no
         * mechanism to receive an async Gpu-to-Host value.
         */
        template <
            std::uint64_t Dims,
            typename U,
            typename Mapper,
            typename Reducer>
        auto reduce_impl(
            const domain_t<Dims>& domain,
            U init,
            Mapper&& mapper,
            Reducer&& reducer,
            const iarray<Dims>& /*tile_size*/
        ) const -> future_t<U>
        {
            set_device_for_thread();

            auto state =
                std::make_shared<typename future_t<U>::future_state_t>();
            // this is a synchronous operation, so it's "direct"
            state->completion_context = completion_context_t::direct();

            const std::size_t n = domain.size();
            if (n == 0) {
                state->construct_result(init);
                state->ready.store(true);
                return future_t<U>{std::move(state)};
            }

            try {
                constexpr std::uint32_t block_size = 256;
                std::uint32_t grid_size = (n + block_size - 1) / block_size;
                grid_size = std::min(grid_size, (std::uint32_t) 1024);

                auto shared_mem_size =
                    (block_size / hetero::sub_group_t{}.size()) * sizeof(U);

                auto partial_results_dev =
                    hetero::device::allocate_vector<U>(grid_size);
                auto final_result_dev = hetero::device::allocate_vector<U>(1);

                auto kernel1_config = hetero::grid::config(
                    grid_size,
                    block_size,
                    shared_mem_size
                );
                hetero::device::launch_async(
                    detail::reduce_kernel_part1<Dims, U, Mapper, Reducer>,
                    kernel1_config,
                    stream_,
                    domain,
                    std::forward<Mapper>(mapper),
                    reducer,
                    init,
                    n,
                    partial_results_dev.data()
                );

                auto kernel2_config =
                    hetero::grid::config(1, block_size, shared_mem_size);
                hetero::device::launch_async(
                    detail::reduce_kernel_part2<U, Reducer>,
                    kernel2_config,
                    stream_,
                    partial_results_dev.data(),
                    final_result_dev.data(),
                    grid_size,
                    reducer,
                    init
                );

                // this is the *only* way to guarantee the result is
                // ready to be copied back.
                stream_.synchronize();

                // copy final result and populate state
                U final_result;
                hetero::device::copy_vector_to_host(
                    &final_result,
                    final_result_dev,
                    1
                );

                state->construct_result(std::move(final_result));
                state->ready.store(true);
            }
            catch (...) {
                state->exception = std::current_exception();
                state->has_error.store(true);
                state->ready.store(true);
            }

            return future_t<U>{std::move(state)};
        }
    };

    template <bool OnGPU = global::on_gpu>
    auto& default_executor()
    {
        if constexpr (OnGPU) {
            static gpu_executor_t executor{};
            return executor;
        }
        else {
            static omp_executor_t executor{};
            return executor;
        }
    }

    using default_executor_t =
        std::conditional_t<global::on_gpu, gpu_executor_t, omp_executor_t>;

}   // namespace simbi::exec

#endif   // EXECUTOR_HPP
