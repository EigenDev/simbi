#ifndef SPAN_HPP
#define SPAN_HPP

#include <cstddef>

namespace simbi::mem {
    /**
     * span_t - non-owning view of contiguous data
     *
     * Single responsibility (I've been hardcore on this kick lately): provide
     * safe access to pointer + size Zero ownership, zero allocation, zero
     * overhead
     */
    template <typename T>
    struct span_t {
        T* data;
        std::size_t size;

        // construction
        constexpr span_t() noexcept : data(nullptr), size(0) {}

        constexpr span_t(T* ptr, std::size_t count) noexcept
            : data(ptr), size(count)
        {
        }

        template <std::size_t N>
        constexpr span_t(T (&array)[N]) noexcept : data(array), size(N)
        {
        }

        // element access
        constexpr T& operator[](std::size_t idx) const noexcept
        {
            return data[idx];
        }

        constexpr T& front() const noexcept { return data[0]; }
        constexpr T& back() const noexcept { return data[size - 1]; }

        // iterators
        constexpr T* begin() const noexcept { return data; }
        constexpr T* end() const noexcept { return data + size; }

        // queries
        constexpr std::size_t size_bytes() const noexcept
        {
            return size * sizeof(T);
        }

        constexpr bool empty() const noexcept { return size == 0; }

        // subspans
        constexpr span_t subspan(
            std::size_t offset,
            std::size_t count = std::size_t(-1)
        ) const noexcept
        {
            if (count == std::size_t(-1)) {
                count = size - offset;
            }
            return span_t{data + offset, count};
        }

        constexpr span_t first(std::size_t count) const noexcept
        {
            return span_t{data, count};
        }

        constexpr span_t last(std::size_t count) const noexcept
        {
            return span_t{data + size - count, count};
        }
    };

    // deduction guides
    template <typename T>
    span_t(T*, std::size_t) -> span_t<T>;

    template <typename T, std::size_t N>
    span_t(T (&)[N]) -> span_t<T>;

    // utilities
    template <typename T>
    constexpr span_t<const std::byte> as_bytes(span_t<T> s) noexcept
    {
        return {reinterpret_cast<const std::byte*>(s.data), s.size_bytes()};
    }

    template <typename T>
    constexpr span_t<std::byte> as_writable_bytes(span_t<T> s) noexcept
    {
        return {reinterpret_cast<std::byte*>(s.data), s.size_bytes()};
    }

}   // namespace simbi::mem

#endif
