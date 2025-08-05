#ifndef EXECUTOR_HPP
#define EXECUTOR_HPP

#include "adapter/device_adapter_api.hpp"
#include "adapter/device_types.hpp"
#include "config.hpp"
#include "containers/vector.hpp"
#include "domain/domain.hpp"
#include "execution/future.hpp"
#include "functional/fp.hpp"
#include "memory/device.hpp"
#include "tiling.hpp"

#include <algorithm>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

namespace simbi::async {
    inline auto get_nthreads() -> std::uint64_t
    {
        if (const char* thread_env = std::getenv("NTHREADS")) {
            return static_cast<std::uint64_t>(
                std::stoul(std::string(thread_env))
            );
        }

        if (const char* thread_env = std::getenv("OMP_NUM_THREADS")) {
            return static_cast<std::uint64_t>(
                std::stoul(std::string(thread_env))
            );
        }

        return std::thread::hardware_concurrency();
    };

    // minimal thread pool implementation
    class thread_pool_t
    {
      private:
        std::vector<std::thread> workers_;
        std::queue<std::function<void()>> tasks_;
        std::mutex queue_mutex_;
        std::condition_variable condition_;
        bool stop_;

      public:
        explicit thread_pool_t(std::size_t threads) : stop_(false)
        {
            for (std::size_t i = 0; i < threads; ++i) {
                workers_.emplace_back([this] {
                    for (;;) {
                        std::function<void()> task;
                        {
                            std::unique_lock<std::mutex> lock(queue_mutex_);
                            condition_.wait(lock, [this] {
                                return stop_ || !tasks_.empty();
                            });
                            if (stop_ && tasks_.empty()) {
                                return;
                            }
                            task = std::move(tasks_.front());
                            tasks_.pop();
                        }
                        task();
                    }
                });
            }
        }

        template <typename Func>
        void submit(Func&& func)
        {
            {
                std::unique_lock<std::mutex> lock(queue_mutex_);
                if (stop_) {
                    return;
                }
                tasks_.emplace(std::forward<Func>(func));
            }
            condition_.notify_one();
        }

        ~thread_pool_t()
        {
            {
                std::unique_lock<std::mutex> lock(queue_mutex_);
                stop_ = true;
            }
            condition_.notify_all();
            for (std::thread& worker : workers_) {
                worker.join();
            }
        }
    };

    // singleton thread pool manager - lazy initialization
    class thread_pool_manager_t
    {
      public:
        static thread_pool_t& get_pool()
        {
            static thread_pool_t singleton(get_nthreads());
            return singleton;
        }

        static std::size_t get_nthreads()
        {
            return ::simbi::async::get_nthreads();
        }
    };

    // executor base
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
        // async interface
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

        // domain-aware parallel iteration
        template <std::uint64_t Dims, typename Func>
        auto for_each(const domain_t<Dims>& domain, Func&& func) const
            -> future_t<void>
        {
            return derived().for_each_impl(domain, std::forward<Func>(func));
        }

        // generic reduction operation
        template <
            std::uint64_t Dims,
            typename T,
            typename Mapper,
            typename Reducer>
        auto reduce(
            const domain_t<Dims>& domain,
            T init,
            Mapper&& mapper,
            Reducer&& reducer
        ) const -> future_t<T>
        {
            return derived().reduce_impl(
                domain,
                init,
                std::forward<Mapper>(mapper),
                std::forward<Reducer>(reducer)
            );
        }

        // tiled domain-aware parallel iteration
        template <std::uint64_t Dims, typename T, typename Func>
        auto for_each_tiled(
            const domain_t<Dims>& domain,
            Func&& func,
            const iarray<Dims>& tile_size = {-1}
        ) const -> future_t<void>
        {
            return derived().for_each_tiled_impl(
                domain,
                std::forward<Func>(func),
                tile_size[0] == -1 ? tiling::optimal_tile_size<Dims, T>()
                                   : tile_size
            );
        }

        template <
            std::uint64_t Dims,
            typename T,   // type of the elements being processed
            typename U,
            typename Mapper,
            typename Reducer>
        auto reduce_tiled(
            const domain_t<Dims>& domain,
            U init,
            Mapper&& mapper,
            Reducer&& reducer,
            const iarray<Dims>& tile_size = {-1}
        ) const -> future_t<T>
        {
            return derived().reduce_tiled_impl(
                domain,
                init,
                std::forward<Mapper>(mapper),
                std::forward<Reducer>(reducer),
                tile_size[0] == -1 ? tiling::optimal_tile_size<Dims, T>()
                                   : tile_size
            );
        }
    };

    // cpu executor
    class cpu_executor_t : public executor_base_t<cpu_executor_t>
    {
      public:
        cpu_executor_t() = default;

        //  async implementation
        template <typename Func, typename... Args>
        auto async_impl(Func&& func, Args&&... args) const
            -> future_t<decltype(func(args...))>
        {
            using result_t = decltype(func(args...));

            auto state =
                std::make_shared<typename future_t<result_t>::future_state_t>();

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

        // domain-aware for_each implementation
        template <std::uint64_t Dims, typename Func>
        auto for_each_impl(const domain_t<Dims>& domain, Func&& func) const
            -> future_t<void>
        {
            return async_impl([=, this]() { iterate_domain(domain, func); });
        }

        // reduction implementation
        template <
            std::uint64_t Dims,
            typename T,
            typename Mapper,
            typename Reducer>
        auto reduce_impl(
            const domain_t<Dims>& domain,
            T init,
            Mapper&& mapper,
            Reducer&& reducer
        ) const -> future_t<T>
        {
            return async_impl([=, this]() {
                T accumulator = init;
                iterate_domain(domain, [&](auto coord) {
                    accumulator = reducer(accumulator, mapper(coord));
                });
                return accumulator;
            });
        }

        template <std::uint64_t Dims, typename Func>
        auto for_each_tiled_impl(
            const domain_t<Dims>& domain,
            Func&& func,
            const iarray<Dims>& tile_size
        ) const -> future_t<void>
        {
            return async_impl([=, this]() {
                tiling::tile_range(domain, tile_size) |
                    fp::for_each([&](const auto& tile) {
                        iterate_domain(tile.domain, func);
                    });
            });
        }

        template <
            std::uint64_t Dims,
            typename T,
            typename Mapper,
            typename Reducer>
        auto reduce_tiled_impl(
            const domain_t<Dims>& domain,
            T init,
            Mapper&& mapper,
            Reducer&& reducer,
            const iarray<Dims>& tile_size
        ) const -> future_t<T>
        {
            return async_impl([=, this]() {
                return tiling::tile_range(domain, tile_size) |
                       fp::map([&](const auto& tile) {
                           T tile_result = init;
                           iterate_domain(tile.domain, [&](auto coord) {
                               tile_result =
                                   reducer(tile_result, mapper(coord));
                           });
                           return tile_result;
                       }) |
                       fp::reduce(reducer);
            });
        }

      private:
        template <std::uint64_t Dims, typename Func>
        void iterate_domain(const domain_t<Dims>& domain, Func func) const
        {
            if constexpr (Dims == 1) {
                for (auto ii = domain.start[0]; ii < domain.end[0]; ++ii) {
                    func(iarray<1>{ii});
                }
            }
            else if constexpr (Dims == 2) {
                for (auto ii = domain.start[0]; ii < domain.end[0]; ++ii) {
                    for (auto jj = domain.start[1]; jj < domain.end[1]; ++jj) {
                        func(iarray<2>{ii, jj});
                    }
                }
            }
            else if constexpr (Dims == 3) {
                for (auto ii = domain.start[0]; ii < domain.end[0]; ++ii) {
                    for (auto jj = domain.start[1]; jj < domain.end[1]; ++jj) {
                        for (auto kk = domain.start[2]; kk < domain.end[2];
                             ++kk) {
                            func(iarray<3>{ii, jj, kk});
                        }
                    }
                }
            }
        }
    };

    // parallel cpu executor
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

        //  async implementation
        template <typename Func, typename... Args>
        auto async_impl(Func&& func, Args&&... args) const
            -> future_t<decltype(func(args...))>
        {
            using result_t = decltype(func(args...));

            auto state =
                std::make_shared<typename future_t<result_t>::future_state_t>();

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

        // parallel for_each implementation
        template <std::uint64_t Dims, typename Func>
        auto for_each_impl(const domain_t<Dims>& domain, Func&& func) const
            -> future_t<void>
        {
            return async_impl([=, this]() { iterate_domain(domain, func); });
        }

        // parallel reduction implementation
        template <
            std::uint64_t Dims,
            typename T,
            typename Mapper,
            typename Reducer>
        auto reduce_impl(
            const domain_t<Dims>& domain,
            T init,
            Mapper&& mapper,
            Reducer&& reducer
        ) const -> future_t<T>
        {
            return async_impl([=, this]() {
                auto total_size = domain.size();
                auto chunk_size = (total_size + nthreads_ - 1) / nthreads_;

                std::vector<T> partial_results(nthreads_, init);
                std::vector<std::thread> threads;
                std::mutex results_mutex;

                for (std::size_t t = 0; t < nthreads_; ++t) {
                    auto start_idx = t * chunk_size;
                    auto end_idx = std::min(start_idx + chunk_size, total_size);

                    if (start_idx >= end_idx) {
                        break;
                    }

                    threads.emplace_back([=, &partial_results, &domain]() {
                        T local_accumulator = init;
                        for (auto idx = start_idx; idx < end_idx; ++idx) {
                            auto coord = domain.linear_to_coord(idx);
                            local_accumulator =
                                reducer(local_accumulator, mapper(coord));
                        }
                        partial_results[t] = local_accumulator;
                    });
                }

                for (auto& thread : threads) {
                    thread.join();
                }

                // combine partial results
                T final_result = init;
                for (const auto& partial : partial_results) {
                    final_result = reducer(final_result, partial);
                }

                return final_result;
            });
        }

        template <std::uint64_t Dims, typename Func>
        auto for_each_tiled_impl(
            const domain_t<Dims>& domain,
            Func&& func,
            const iarray<Dims>& tile_size
        ) const -> future_t<void>
        {
            return async_impl([=, this]() {
                auto tiles =
                    tiling::tile_range(domain, tile_size) | fp::collect<>;

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
        auto reduce_tiled_impl(
            const domain_t<Dims>& domain,
            T init,
            Mapper&& mapper,
            Reducer&& reducer,
            const iarray<Dims>& tile_size
        ) const -> future_t<T>
        {
            return async_impl([=, this]() {
                auto tiles =
                    tiling::tile_range(domain, tile_size) | fp::collect<>;

                std::vector<future_t<T>> tile_futures;
                tile_futures.reserve(tiles.size());

                // process each tile in parallel
                for (const auto& tile : tiles) {
                    tile_futures.push_back(async_impl([=, this]() {
                        T tile_result = init;
                        iterate_domain(tile.domain, [&](auto coord) {
                            tile_result = reducer(tile_result, mapper(coord));
                        });
                        return tile_result;
                    }));
                }

                // collect and reduce tile results
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
            auto total_size = domain.size();
            auto chunk_size = (total_size + nthreads_ - 1) / nthreads_;

            std::vector<future_t<void>> futures;

            for (std::size_t tt = 0; tt < nthreads_; ++tt) {
                auto start_idx = tt * chunk_size;
                auto end_idx   = std::min(start_idx + chunk_size, total_size);

                if (start_idx >= end_idx) {
                    break;
                }

                futures.push_back(async_impl([=]() {
                    for (auto idx = start_idx; idx < end_idx; ++idx) {
                        auto coord = domain.linear_to_coord(idx);
                        func(coord);
                    }
                }));
            }

            for (auto& future : futures) {
                future.wait();
            }
        }
    };

    // openMP executor
    class omp_executor_t : public executor_base_t<omp_executor_t>
    {
      public:
        omp_executor_t() = default;

        //  async implementation
        template <typename Func, typename... Args>
        auto async_impl(Func&& func, Args&&... args) const
            -> future_t<decltype(func(args...))>
        {
            using result_t = decltype(func(args...));

            auto state =
                std::make_shared<typename future_t<result_t>::future_state_t>();

            try {
                if constexpr (std::is_void_v<result_t>) {
#pragma omp parallel
                    {
                        func(args...);
                    }
                    state->ready.store(true);
                }
                else {
                    result_t result;
#pragma omp parallel
                    {
                        result = func(args...);
                    }
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

        // domain-aware for_each implementation
        template <std::uint64_t Dims, typename Func>
        auto for_each_impl(const domain_t<Dims>& domain, Func&& func) const
            -> future_t<void>
        {
            return async_impl([=, this]() { iterate_domain(domain, func); });
        }

        // reduction implementation
        template <
            std::uint64_t Dims,
            typename T,
            typename Mapper,
            typename Reducer>
        auto reduce_impl(
            const domain_t<Dims>& domain,
            T init,
            Mapper&& mapper,
            Reducer&& reducer
        ) const -> future_t<T>
        {
            return async_impl([=, this]() {
                T accumulator = init;
                iterate_domain(domain, [&](auto coord) {
                    accumulator = reducer(accumulator, mapper(coord));
                });
                return accumulator;
            });
        }

      private:
        template <std::uint64_t Dims, typename Func>
        void iterate_domain(const domain_t<Dims>& domain, Func func) const
        {
            if constexpr (Dims == 1) {
#pragma omp parallel for
                for (auto ii = domain.start[0]; ii < domain.end[0]; ++ii) {
                    func(iarray<1>{ii});
                }
            }
            else if constexpr (Dims == 2) {
#pragma omp parallel for collapse(2)
                for (auto ii = domain.start[0]; ii < domain.end[0]; ++ii) {
                    for (auto jj = domain.start[1]; jj < domain.end[1]; ++jj) {
                        func(iarray<2>{ii, jj});
                    }
                }
            }
            else if constexpr (Dims == 3) {
#pragma omp parallel for collapse(3)
                for (auto ii = domain.start[0]; ii < domain.end[0]; ++ii) {
                    for (auto jj = domain.start[1]; jj < domain.end[1]; ++jj) {
                        for (auto kk = domain.start[2]; kk < domain.end[2];
                             ++kk) {
                            func(iarray<3>{ii, jj, kk});
                        }
                    }
                }
            }
        }
    };

    // gpu executor with dynamic grid sizing
    class gpu_executor_t : public executor_base_t<gpu_executor_t>
    {
      private:
        mem::device_id_t device_;
        adapter::stream_t<> stream_;

      public:
        explicit gpu_executor_t(
            mem::device_id_t device = mem::device_id_t::gpu_device(0)
        )
            : device_(device)
        {
            gpu::api::set_device(device.device_id);
            gpu::api::stream_create(&stream_);
        }

        ~gpu_executor_t()
        {
            if (stream_) {
                gpu::api::stream_destroy(stream_);
            }
        }

        gpu_executor_t(const gpu_executor_t&)            = delete;
        gpu_executor_t& operator=(const gpu_executor_t&) = delete;
        gpu_executor_t(gpu_executor_t&& other) noexcept
            : device_(other.device_), stream_(other.stream_)
        {
            other.stream_ = {};
        }

        //  async implementation
        template <typename Kernel, typename... Args>
        auto async_impl(Kernel&& kernel, Args&&... args) const -> future_t<void>
        {
            auto state =
                std::make_shared<typename future_t<void>::future_state_t>();
            state->stream = stream_;

            try {
                gpu::api::set_device(device_.device_id);

                // basic grid config - will be enhanced for domain-aware
                // sizing
                auto total_threads = 256;
                auto blocks        = 1;
                auto threads       = total_threads;

                auto launch_config = grid::config(blocks, threads);
                grid::launch(
                    std::forward<Kernel>(kernel),
                    launch_config,
                    std::forward<Args>(args)...
                );

                gpu::api::event_create(&state->event);
                gpu::api::event_record(state->event, stream_);
            }
            catch (...) {
                state->exception = std::current_exception();
                state->has_error.store(true);
                state->ready.store(true);
            }

            return future_t<void>{std::move(state)};
        }

        // gpu for_each implementation with dynamic grid sizing
        template <std::uint64_t Dims, typename Func>
        auto for_each_impl(const domain_t<Dims>& domain, Func&& func) const
            -> future_t<void>
        {
            auto state =
                std::make_shared<typename future_t<void>::future_state_t>();
            state->stream = stream_;

            try {
                gpu::api::set_device(device_.device_id);

                auto [blocks, threads] = optimal_grid_size(domain);

                auto kernel = [=] DEV(
                                  int thread_idx,
                                  int block_idx,
                                  int block_dim,
                                  int grid_dim
                              ) {
                    auto global_idx    = block_idx * block_dim + thread_idx;
                    auto total_threads = grid_dim * block_dim;
                    auto domain_size   = domain.size();

                    // stride loop for large domains
                    for (auto idx = global_idx; idx < domain_size;
                         idx += total_threads) {
                        auto coord = domain.linear_to_coord(idx);
                        func(coord);
                    }
                };

                auto launch_config = grid::config(blocks, threads);
                grid::launch(kernel, launch_config);

                gpu::api::event_create(&state->event);
                gpu::api::event_record(state->event, stream_);
            }
            catch (...) {
                state->exception = std::current_exception();
                state->has_error.store(true);
                state->ready.store(true);
            }

            return future_t<void>{std::move(state)};
        }

        // gpu reduction implementation
        template <
            std::uint64_t Dims,
            typename T,
            typename Mapper,
            typename Reducer>
        auto reduce_impl(
            const domain_t<Dims>& /*domain*/,
            T init,
            Mapper&& /*mapper*/,
            Reducer&& /*reducer*/
        ) const -> future_t<T>
        {
            // for now, fallback to cpu for reductions - gpu reductions
            // are complex could implement proper gpu reductions with
            // shared memory later
            return async_impl([=]() {
                // transfer to cpu, reduce, transfer back
                // this is a placeholder - proper gpu reduction would
                // use shared memory + atomics
                T accumulator = init;
                // sequential fallback for now
                return accumulator;
            });
        }

        template <std::uint64_t Dims, typename Func>
        auto for_each_tiled_impl(
            const domain_t<Dims>& domain,
            Func&& func,
            const iarray<Dims>& tile_size
        ) const -> future_t<void>
        {
            // For GPU, we can either:
            // 1. Launch one kernel per tile (good for large tiles)
            // 2. Use single kernel with tile-aware indexing (better for small
            // tiles)

            auto state =
                std::make_shared<typename future_t<void>::future_state_t>();
            state->stream = stream_;

            try {
                gpu::api::set_device(device_.device_id);

                // collect tiles upfront for GPU processing
                auto tiles =
                    tiling::tile_range(domain, tile_size) | fp::collect<>;

                if (tiles.size() == 1) {
                    // single tile - use existing implementation
                    return for_each_impl(domain, std::forward<Func>(func));
                }

                // multiple tiles - launch kernel per tile or batch them
                for (const auto& tile : tiles) {
                    auto [blocks, threads] = optimal_grid_size(tile.domain);

                    auto kernel = [=] DEV(
                                      int thread_idx,
                                      int block_idx,
                                      int block_dim,
                                      int grid_dim
                                  ) {
                        auto global_idx    = block_idx * block_dim + thread_idx;
                        auto total_threads = grid_dim * block_dim;
                        auto tile_size     = tile.domain.size();

                        for (auto idx = global_idx; idx < tile_size;
                             idx += total_threads) {
                            auto coord = tile.domain.linear_to_coord(idx);
                            func(coord);
                        }
                    };

                    auto launch_config = grid::config(blocks, threads);
                    grid::launch(kernel, launch_config);
                }

                gpu::api::event_create(&state->event);
                gpu::api::event_record(state->event, stream_);
            }
            catch (...) {
                state->exception = std::current_exception();
                state->has_error.store(true);
                state->ready.store(true);
            }

            return future_t<void>{std::move(state)};
        }

        template <
            std::uint64_t Dims,
            typename T,
            typename Mapper,
            typename Reducer>
        auto reduce_tiled_impl(
            const domain_t<Dims>& domain,
            T init,
            Mapper&& mapper,
            Reducer&& reducer,
            const iarray<Dims>& tile_size
        ) const -> future_t<T>
        {
            // GPU reductions with tiling are complex - for now, fallback to CPU
            // TODO: implement proper GPU tiled reduction with shared memory
            return async_impl([=]() {
                T accumulator = init;
                tiling::tile_range(domain, tile_size) |
                    fp::for_each([&](const auto& tile) {
                        // would need to transfer tile data to CPU, reduce, then
                        // accumulate this is a placeholder for proper GPU tiled
                        // reduction
                    });
                return accumulator;
            });
        }

        mem::device_id_t device() const { return device_; }
        adapter::stream_t<> stream() const { return stream_; }
        void synchronize() { gpu::api::stream_synchronize(stream_); }

      private:
        template <std::uint64_t Dims>
        auto optimal_grid_size(const domain_t<Dims>& domain) const
        {
            auto domain_size = domain.size();

            // optimal thread count based on domain size
            constexpr std::int64_t max_threads_per_block = 1024;
            constexpr std::int64_t preferred_threads     = 256;

            std::int64_t threads = std::min(
                static_cast<std::int64_t>(domain_size),
                std::min(max_threads_per_block, preferred_threads)
            );

            std::int64_t blocks = (domain_size + threads - 1) / threads;

            // limit blocks to reasonable number for memory bandwidth
            constexpr std::int64_t max_blocks = 65535;
            blocks                            = std::min(blocks, max_blocks);

            return std::pair{blocks, threads};
        }
    };

    // factory functions
    inline auto cpu_executor() -> cpu_executor_t { return cpu_executor_t{}; }

    inline auto par_cpu_executor() -> par_cpu_executor_t
    {
        return par_cpu_executor_t{};
    }

    inline auto omp_executor() -> omp_executor_t { return omp_executor_t{}; }

    inline auto gpu_executor(int device_id = 0) -> gpu_executor_t
    {
        return gpu_executor_t{mem::device_id_t::gpu_device(device_id)};
    }

    inline auto default_executor(std::size_t device_id = 0)
    {
        if constexpr (global::on_gpu) {
            return gpu_executor(device_id);
        }
        else {
            // if (global::use_omp) {
            //     return omp_executor();
            // }
            return par_cpu_executor();
        }
    }

    using default_executor_t =
        std::conditional_t<global::on_gpu, gpu_executor_t, par_cpu_executor_t>;

}   // namespace simbi::async

#endif   // EXECUTOR_HPP
