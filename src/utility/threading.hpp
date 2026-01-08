// =============================================================================
// threading.hpp
//
// thread utilities using standard library only.
// provides portable thread count queries without requiring openmp headers.
// openmp pragmas and runtime functions still available where needed.
//
// usage:
//   auto max_threads = threading::hardware_concurrency();
//   threading::set_thread_affinity(cpu_id);
// =============================================================================

#ifndef UTILITY_THREADING_HPP
#define UTILITY_THREADING_HPP

#include <algorithm>
#include <cstddef>
#include <thread>

namespace simbi::threading {

    // =========================================================================
    // hardware queries using standard library
    // =========================================================================

    // get maximum available hardware threads
    inline std::size_t hardware_concurrency() noexcept
    {
        const auto hw_threads = std::thread::hardware_concurrency();
        return hw_threads > 0 ? hw_threads : 1; // fallback to 1 if unknown
    }

    // suggest good thread count for parallel work
    inline std::size_t suggested_thread_count() noexcept
    {
        return hardware_concurrency();
    }

    // clamp thread count to reasonable range
    inline std::size_t clamp_thread_count(std::size_t requested) noexcept
    {
        const auto max_threads = hardware_concurrency();
        return std::clamp(requested, std::size_t{1}, max_threads);
    }

    // =========================================================================
    // thread identification
    // =========================================================================

    // get current thread id as size_t (for indexing)
    inline std::size_t current_thread_index() noexcept
    {
        return std::hash<std::thread::id>{}(std::this_thread::get_id());
    }

    // check if running on main thread
    inline bool is_main_thread() noexcept
    {
        static const auto main_id = std::this_thread::get_id();
        return std::this_thread::get_id() == main_id;
    }

} // namespace simbi::threading

#endif // UTILITY_THREADING_HPP
