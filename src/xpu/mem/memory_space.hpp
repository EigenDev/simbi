// =============================================================================
// memory_space.hpp
//
// compile-time memory space concept and trait definitions.
// memory spaces define where data resides and how it can be accessed.
// each space provides:
//   - allocation and deallocation functions
//   - accessibility queries between spaces
//   - transfer capabilities and constraints
//
// usage:
//   auto buffer = allocate<host_memory>(size);
//   shared_buffer_t<float, unified_memory> data(n);
// =============================================================================

#pragma once

#include "xpu/core/memory_concepts.hpp"

#include <cstddef>
#include <new>
#include <string_view>

namespace simbi::xpu::mem {

    // =============================================================================
    // memory space concept - defined in core/memory_concepts.hpp
    // =============================================================================

    using core::memory_space_c;

    // =============================================================================
    // memory space traits
    // =============================================================================

    template <memory_space_c Space>
    struct memory_space_traits
    {
        using space_type = Space;

        static constexpr std::string_view name()
        {
            return Space::name();
        }
        static constexpr bool is_device_accessible = Space::is_device_accessible;
        static constexpr bool is_host_accessible   = Space::is_host_accessible;
        static constexpr bool is_unified = Space::is_device_accessible && Space::is_host_accessible;

        template <memory_space_c OtherSpace>
        static constexpr bool is_accessible_from()
        {
            return Space::template is_accessible_from<OtherSpace>();
        }
    };

    // =============================================================================
    // memory allocation block
    // =============================================================================

    template <memory_space_c Space>
    struct block_t
    {
        void*       data;
        std::size_t size;
        Space       space;

        block_t() : data(nullptr), size(0), space{} {}

        block_t(void* ptr, std::size_t sz) : data(ptr), size(sz), space{} {}

        block_t(const block_t&)            = delete;
        block_t& operator=(const block_t&) = delete;

        block_t(block_t&& other) noexcept
            : data(other.data), size(other.size), space(std::move(other.space))
        {
            other.data = nullptr;
            other.size = 0;
        }

        block_t& operator=(block_t&& other) noexcept
        {
            if (this != &other) {
                if (data) {
                    space.deallocate(data, size);
                }
                data       = other.data;
                size       = other.size;
                space      = std::move(other.space);
                other.data = nullptr;
                other.size = 0;
            }
            return *this;
        }

        ~block_t()
        {
            if (data) {
                space.deallocate(data, size);
            }
        }

        explicit operator bool() const
        {
            return data != nullptr;
        }

        template <typename T>
        T* as() const
        {
            return static_cast<T*>(data);
        }
    };

    // =============================================================================
    // space compatibility utilities
    // =============================================================================

    template <memory_space_c SourceSpace, memory_space_c DestSpace>
    constexpr bool can_access_directly()
    {
        return DestSpace::template is_accessible_from<SourceSpace>();
    }

    template <memory_space_c SourceSpace, memory_space_c DestSpace>
    constexpr bool requires_staging()
    {
        return !can_access_directly<SourceSpace, DestSpace>();
    }

    template <memory_space_c Space>
    constexpr bool is_unified_space()
    {
        return Space::is_device_accessible && Space::is_host_accessible;
    }

    template <memory_space_c Space>
    constexpr bool is_host_only_space()
    {
        return Space::is_host_accessible && !Space::is_device_accessible;
    }

    template <memory_space_c Space>
    constexpr bool is_device_only_space()
    {
        return Space::is_device_accessible && !Space::is_host_accessible;
    }

    // =============================================================================
    // allocation helpers
    // =============================================================================

    template <memory_space_c Space, typename T>
    block_t<Space> allocate(std::size_t count)
    {
        std::size_t size = count * sizeof(T);
        void*       ptr  = Space::allocate(size);
        if (!ptr) {
            throw std::bad_alloc{};
        }
        return block_t<Space>(ptr, size);
    }

    template <memory_space_c Space>
    block_t<Space> allocate_bytes(std::size_t size)
    {
        void* ptr = Space::allocate(size);
        if (!ptr) {
            throw std::bad_alloc{};
        }
        return block_t<Space>(ptr, size);
    }

    // =============================================================================
    // forward declarations for memory space implementations
    // =============================================================================

    struct host_memory;
    struct device_memory;
    struct unified_memory;

    // =============================================================================
    // default memory space selection
    // =============================================================================

    template <bool gpu_available = false>
    struct default_memory_space_selector;

} // namespace simbi::xpu::mem
