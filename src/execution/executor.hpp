#ifndef EXECUTOR_HPP
#define EXECUTOR_HPP

#include "adapter/device_adapter_api.hpp"
#include "adapter/device_types.hpp"
#include "config.hpp"
#include "containers/vector.hpp"
#include "domain/domain.hpp"
#include "execution/completion.hpp"
#include "execution/future.hpp"
#include "functional/fp.hpp"
#include "new/device.hpp"
#include "thread_pool.hpp"
#include "tiling.hpp"

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <memory>
#include <mutex>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

namespace simbi::exec {
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

    // OpenMP executor
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

    class gpu_executor_t : public executor_base_t<gpu_executor_t>
    {
      private:
        std::vector<mem::device_t> devices_;
        std::vector<adapter::stream_t<>> streams_;

      public:
        constexpr gpu_executor_t(
            const std::vector<std::int64_t>& device_ids = {}
        )
        {
            if (device_ids.empty()) {
                std::int64_t device_count = 0;
                gpu::api::get_device_count(&device_count);
                for (std::int64_t ii = 0; ii < device_count; ++ii) {
                    devices_.push_back(mem::device_t::gpu(ii));
                    adapter::stream_t<> stream;
                    gpu::api::stream_create(&stream);
                    streams_.push_back(stream);
                }
            }
            else {
                for (auto id : device_ids) {
                    devices_.push_back(mem::device_t::gpu(id));
                    adapter::stream_t<> stream;
                    gpu::api::stream_create(&stream);
                    streams_.push_back(stream);
                }
            }

            if (devices_.empty()) {
                devices_.push_back(mem::device_t::gpu(0));
                adapter::stream_t<> stream;
                gpu::api::stream_create(&stream);
                streams_.push_back(stream);
            }
        }

        ~gpu_executor_t()
        {
            for (auto& stream : streams_) {
                gpu::api::stream_destroy(stream);
            }
        }

        gpu_executor_t(const gpu_executor_t&)            = delete;
        gpu_executor_t& operator=(const gpu_executor_t&) = delete;
        gpu_executor_t(gpu_executor_t&&)                 = default;
        gpu_executor_t& operator=(gpu_executor_t&&)      = default;

        template <typename Func, typename... Args>
        auto async_impl(Func&& func, Args&&... args) const -> future_t<void>
        {
            auto state =
                std::make_shared<typename future_t<void>::future_state_t>();
            state->completion_context =
                completion_context_t::gpu_stream(streams_[0]);

            try {
                // create events for tracking completion on each device
                std::vector<adapter::event_t<>> events(devices_.size());
                for (std::size_t ii = 0; ii < devices_.size(); ++ii) {
                    gpu::api::event_create(&events[ii]);
                }

                for (std::size_t ii = 0; ii < devices_.size(); ++ii) {
                    set_current_device(devices_[ii]);

                    std::forward<Func>(func)(
                        ii,
                        devices_.size(),
                        std::forward<Args>(args)...
                    );

                    gpu::api::event_record(events[ii], streams_[ii]);
                }

                // create a callback to wait for all devices to complete
                state->set_ready_callback([events = std::move(events)]() {
                    for (auto& event : events) {
                        gpu::api::event_synchronize(event);
                        gpu::api::event_destroy(event);
                    }
                });
            }
            catch (...) {
                state->exception = std::current_exception();
                state->has_error.store(true);
                state->ready.store(true);
            }

            return future_t<void>{state};
        }

        template <std::uint64_t Dims, typename Func>
        auto for_each_impl(const domain_t<Dims>& domain, Func&& func) const
            -> future_t<void>
        {
            return async_impl([domain, f = std::forward<Func>(func), this](
                                  std::size_t device_idx,
                                  std::size_t device_count
                              ) {
                std::uint64_t partition_axis = 0;
                auto shape                   = domain.shape();
                for (std::uint64_t ii = 1; ii < Dims; ++ii) {
                    if (shape[ii] > shape[partition_axis]) {
                        partition_axis = ii;
                    }
                }

                auto subdomain =
                    domain.partition(device_count, device_idx, partition_axis);
                if (subdomain.empty()) {
                    return;
                }

                // calculate grid and block sizes for this partition
                auto [grid_size, block_size] =
                    optimal_grid_size(subdomain.size());

                // launch kernel to process this partition
                auto kernel = [subdomain, f] DEV() {
                    auto idx        = adapter::grid::execution_index::current();
                    auto global_idx = idx.global_thread_id();
                    auto total_threads = idx.total_threads();
                    auto domain_size   = subdomain.size();

                    // stride loop for large domains
                    for (auto ii = global_idx; ii < domain_size;
                         ii += total_threads) {
                        auto coord = subdomain.linear_to_coord(ii);
                        f(coord);
                    }
                };

                auto launch_config = grid::config(grid_size, block_size);
                grid::launch(kernel, launch_config);
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
            Reducer&& reducer
        ) const -> future_t<T>
        {
            // This would need to be expanded to a full GPU reduction algorithm
            // For now, we'll use a simple implementation that delegates to
            // async_impl

            return async_impl(
                [domain,
                 init,
                 mapper = std::forward<Mapper>(mapper),
                 reducer =
                     std::forward<Reducer>(reducer)](std::size_t, std::size_t) {
                    // Implementation would partition the domain and perform
                    // reduction For now, this is a placeholder
                }
            );
        }

        void synchronize() const
        {
            for (auto& stream : streams_) {
                gpu::api::stream_synchronize(stream);
            }
        }

        // calculate optimal grid size for kernel launch
        template <typename SizeType>
        auto optimal_grid_size(SizeType size) const
        {
            constexpr std::int64_t threads_per_block = 256;
            std::int64_t grid_size =
                (size + threads_per_block - 1) / threads_per_block;

            // limit blocks to a reasonable number
            constexpr std::int64_t max_blocks = 65535;
            grid_size                         = std::min(grid_size, max_blocks);

            return std::make_pair(grid_size, threads_per_block);
        }
    };

    // factory functions
    inline auto par_cpu_executor() -> par_cpu_executor_t
    {
        return par_cpu_executor_t{};
    }

    inline auto omp_executor() -> omp_executor_t { return omp_executor_t{}; }

    inline auto gpu_executor() -> gpu_executor_t { return gpu_executor_t{}; }

    template <bool OnGPU = global::on_gpu>
    constexpr auto default_executor()
    {
        if constexpr (OnGPU) {
            return gpu_executor();
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

}   // namespace simbi::exec

#endif   // EXECUTOR_HPP
