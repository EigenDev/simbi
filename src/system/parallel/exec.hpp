#ifndef SIMBI_EXECUTION_HPP
#define SIMBI_EXECUTION_HPP

#include "config.hpp"
#include <algorithm>
#include <array>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

namespace simbi::exec {

    // =================================================================
    // Core Concepts
    // =================================================================

    template <typename T>
    concept domain_like = requires(T domain) {
        { domain.size() } -> std::convertible_to<std::size_t>;
        { domain.begin() } -> std::input_iterator;
        { domain.end() } -> std::sentinel_for<decltype(domain.begin())>;
    };

    template <typename F, typename Coord>
    concept coordinate_function = requires(F f, Coord coord) {
        { f(coord) } -> std::same_as<void>;
    };

    // =================================================================
    // Execution Targets - Type-Safe Device Selection
    // =================================================================

    struct cpu_target_t {
        std::size_t num_threads = std::thread::hardware_concurrency();

        constexpr auto with_threads(std::size_t n) const
        {
            auto copy        = *this;
            copy.num_threads = n;
            return copy;
        }
    };

    struct gpu_target_t {
        std::int32_t device_id                = 0;
        std::array<std::size_t, 3> block_size = {256, 1, 1};
        std::size_t shared_memory             = 0;

        constexpr auto on_device(std::int32_t id) const
        {
            auto copy      = *this;
            copy.device_id = id;
            return copy;
        }

        constexpr auto with_block_size(
            std::size_t x,
            std::size_t y = 1,
            std::size_t z = 1
        ) const
        {
            auto copy       = *this;
            copy.block_size = {x, y, z};
            return copy;
        }

        constexpr auto with_shared_memory(std::size_t bytes) const
        {
            auto copy          = *this;
            copy.shared_memory = bytes;
            return copy;
        }
    };

    struct multi_gpu_target_t {
        std::vector<std::int32_t> device_ids  = {0};
        std::array<std::size_t, 3> block_size = {256, 1, 1};
        bool enable_peer_access               = true;

        constexpr auto on_devices(std::vector<std::int32_t> ids) const
        {
            auto copy       = *this;
            copy.device_ids = std::move(ids);
            return copy;
        }

        constexpr auto with_block_size(
            std::size_t x,
            std::size_t y = 1,
            std::size_t z = 1
        ) const
        {
            auto copy       = *this;
            copy.block_size = {x, y, z};
            return copy;
        }

        constexpr auto with_peer_access(bool enable = true) const
        {
            auto copy               = *this;
            copy.enable_peer_access = enable;
            return copy;
        }
    };

    // global instances
    // constexpr cpu_target_t cpu{};
    // constexpr gpu_target_t gpu{};
    // constexpr multi_gpu_target_t multi_gpu{};

    // =================================================================
    // Tiling Strategy - Core Building Block
    // =================================================================

    template <typename Domain>
    struct tile_t {
        Domain domain;
        std::size_t linear_start;
        std::size_t linear_end;

        auto size() const { return linear_end - linear_start; }

        // iterator over coordinates in this tile
        auto begin() const { return domain.begin() + linear_start; }

        auto end() const { return domain.begin() + linear_end; }
    };

    // Tiling strategies
    template <domain_like Domain>
    auto linear_tiles(const Domain& domain, std::size_t tile_size)
    {
        std::vector<tile_t<Domain>> tiles;
        const auto total_size = domain.size();

        for (std::size_t start = 0; start < total_size; start += tile_size) {
            const auto end = std::min(start + tile_size, total_size);
            tiles.push_back(tile_t<Domain>{domain, start, end});
        }

        return tiles;
    }

    // Cache-optimal tiling (for 2D/3D domains)
    template <domain_like Domain>
    auto
    cache_optimal_tiles(const Domain& domain, std::size_t cache_size_hint = 64)
    {
        // For CFD: prefer spatial locality over linear chunking
        // This would use domain.shape() to create spatial blocks
        return linear_tiles(domain, cache_size_hint);   // simplified for now
    }

    // Multi-device tiling
    template <domain_like Domain>
    auto device_tiles(const Domain& domain, std::size_t num_devices)
    {
        const auto total_size = domain.size();
        const auto base_size  = total_size / num_devices;
        const auto remainder  = total_size % num_devices;

        std::vector<tile_t<Domain>> tiles;
        std::size_t start = 0;

        for (std::size_t device = 0; device < num_devices; ++device) {
            const auto size = base_size + (device < remainder ? 1 : 0);
            tiles.push_back(tile_t<Domain>{domain, start, start + size});
            start += size;
        }

        return tiles;
    }

    // =================================================================
    // Execution Policies - Functional Composition
    // =================================================================

    template <typename Target>
    struct execution_policy_t {
        Target target;
        std::size_t tile_size      = 64;
        bool vectorize             = true;
        bool prefer_cache_locality = true;

        constexpr execution_policy_t(Target t) : target(t) {}

        constexpr auto with_tile_size(std::size_t size) const
        {
            auto copy      = *this;
            copy.tile_size = size;
            return copy;
        }

        constexpr auto disable_vectorization() const
        {
            auto copy      = *this;
            copy.vectorize = false;
            return copy;
        }

        constexpr auto linear_access() const
        {
            auto copy                  = *this;
            copy.prefer_cache_locality = false;
            return copy;
        }
    };

    // Factory functions for clean syntax
    template <typename Target>
    constexpr auto on(Target target)
    {
        return execution_policy_t<Target>{target};
    }

    // =================================================================
    // Execution Implementations
    // =================================================================

    // CPU execution - uses std::execution or thread pool
    template <
        domain_like Domain,
        coordinate_function<typename Domain::value_type> Func>
    void execute_cpu(
        const execution_policy_t<cpu_target_t>& policy,
        const Domain& domain,
        Func&& func
    )
    {

        const auto tiles = policy.prefer_cache_locality
                               ? cache_optimal_tiles(domain, policy.tile_size)
                               : linear_tiles(domain, policy.tile_size);

        // use std::execution for simplicity
        std::for_each(
            std::execution::par_unseq,
            tiles.begin(),
            tiles.end(),
            [&func](const auto& tile) {
                for (auto coord : tile) {
                    func(coord);
                }
            }
        );
    }

    // GPU execution - single device
    template <
        domain_like Domain,
        coordinate_function<typename Domain::value_type> Func>
    void execute_gpu(
        const execution_policy_t<gpu_target_t>& policy,
        const Domain& domain,
        Func&& func
    )
    {

#if GPU_ENABLED
        const auto total_size        = domain.size();
        const auto threads_per_block = policy.target.block_size[0] *
                                       policy.target.block_size[1] *
                                       policy.target.block_size[2];
        const auto num_blocks =
            (total_size + threads_per_block - 1) / threads_per_block;

        // Set device
        gpu::api::set_device(policy.target.device_id);

        // Launch kernel - simplified kernel that iterates over domain
        gpu_kernel<<<
            num_blocks,
            policy.target.block_size[0],
            policy.target.shared_memory>>>([=] __device__() {
            const auto tid    = blockIdx.x * blockDim.x + threadIdx.x;
            const auto stride = blockDim.x * gridDim.x;

            for (auto idx = tid; idx < total_size; idx += stride) {
                if (idx < total_size) {
                    auto coord = domain.linear_to_coord(idx);
                    func(coord);
                }
            }
        });

        gpu::api::device_synchronize();
#else
        // Fallback to CPU
        execute_cpu(on(cpu.with_threads(1)), domain, std::forward<Func>(func));
#endif
    }

    // Multi-GPU execution
    template <
        domain_like Domain,
        coordinate_function<typename Domain::value_type> Func>
    void execute_multi_gpu(
        const execution_policy_t<multi_gpu_target_t>& policy,
        const Domain& domain,
        Func&& func
    )
    {

#if GPU_ENABLED
        const auto& devices     = policy.target.device_ids;
        const auto device_tiles = device_tiles(domain, devices.size());

        // Launch on each device
        std::vector<std::thread> device_threads;

        for (std::size_t i = 0; i < devices.size(); ++i) {
            device_threads.emplace_back([&, i]() {
                auto single_gpu_policy = on(gpu.on_device(devices[i])
                                                .with_block_size(
                                                    policy.target.block_size[0],
                                                    policy.target.block_size[1],
                                                    policy.target.block_size[2]
                                                ));

                // Create sub-domain for this device
                auto sub_domain = domain.subrange(
                    device_tiles[i].linear_start,
                    device_tiles[i].linear_end
                );

                execute_gpu(single_gpu_policy, sub_domain, func);
            });
        }

        // Wait for all devices
        for (auto& thread : device_threads) {
            thread.join();
        }
#else
        // Fallback to CPU
        execute_cpu(on(cpu), domain, std::forward<Func>(func));
#endif
    }

    // =================================================================
    // Unified Execution Interface
    // =================================================================

    template <
        typename Policy,
        domain_like Domain,
        coordinate_function<typename Domain::value_type> Func>
    void execute(const Policy& policy, const Domain& domain, Func&& func)
    {
        if constexpr (std::is_same_v<
                          typename Policy::target_type,
                          cpu_target_t>) {
            execute_cpu(policy, domain, std::forward<Func>(func));
        }
        else if constexpr (std::is_same_v<
                               typename Policy::target_type,
                               gpu_target_t>) {
            execute_gpu(policy, domain, std::forward<Func>(func));
        }
        else if constexpr (std::is_same_v<
                               typename Policy::target_type,
                               multi_gpu_target_t>) {
            execute_multi_gpu(policy, domain, std::forward<Func>(func));
        }
        else {
            static_assert(false, "Unsupported execution target");
        }
    }

    // =================================================================
    // Integration with Expression System
    // =================================================================

    // Assignment with execution policy
    template <typename Field, typename OpBundle, typename Policy>
    void assign_with(Field& field, const OpBundle& ops, const Policy& policy)
    {
        auto domain = field.domain();

        execute(policy, domain, [&](auto coord) {
            field[coord] = ops.eval_at(coord);
        });
    }

    template <typename Field, typename OpBundle, typename Policy>
    void
    accumulate_with(Field& field, const OpBundle& ops, const Policy& policy)
    {
        auto domain = field.domain();

        execute(policy, domain, [&](auto coord) {
            field[coord] += ops.eval_at(coord);
        });
    }

    // =================================================================
    // Adaptive Execution - Automatically Choose Strategy
    // =================================================================

    template <domain_like Domain>
    auto adaptive_policy(const Domain& domain)
    {
        const auto size = domain.size();

        // Simple heuristics - could be much more sophisticated
        if (size < 1000) {
            return on(cpu.with_threads(1));   // Sequential for small problems
        }
        else if (size < 100000) {
            return on(cpu);   // Parallel CPU for medium problems
        }
        else {
#if GPU_ENABLED
            return on(gpu);   // GPU for large problems
#else
            return on(cpu);
#endif
        }
    }

    // Automatic execution - chooses strategy based on problem size
    template <typename Field, typename OpBundle>
    void auto_assign(Field& field, const OpBundle& ops)
    {
        auto policy = adaptive_policy(field.domain());
        assign_with(field, ops, policy);
    }

    // =================================================================
    // Performance Hints - Future Extension Points
    // =================================================================

    template <typename OpBundle>
    struct operation_hints_t {
        bool memory_bound            = false;
        bool compute_bound           = false;
        std::size_t memory_footprint = 0;
        bool vectorizable            = true;

        static constexpr auto infer_from()
        {
            // Could analyze OpBundle types to automatically set hints
            return operation_hints_t{};
        }
    };

}   // namespace simbi::exec

#endif   // SIMBI_EXECUTION_HPP
