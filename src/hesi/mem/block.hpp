#ifndef HET_MEM_BLOCK_HPP
#define HET_MEM_BLOCK_HPP

#include "hesi/backend/memory.hpp"
#include "hesi/backend/transfer.hpp"
#include "hesi/core/types.hpp"
#include "hesi/exec/stream.hpp"

#include <cstddef>

namespace simbi::het::mem {

    // owning raii handle to contiguous memory region
    // strict ownership: assumes ownership on construction, frees on destruction
    struct block_t {
        void* ptr_         = nullptr;
        std::size_t bytes_ = 0;
        locality_t loc_;
        memory_type_t type_;

        // default constructor
        block_t() : loc_(locality_t::host()), type_(memory_type_t::host_visible)
        {
        }

        // primary constructor
        block_t(std::size_t bytes, locality_t loc, memory_type_t type)
            : bytes_(bytes), loc_(loc), type_(type)
        {
            if (bytes > 0) {
                ptr_ =
                    backend::allocate(loc.backend, bytes, type, loc.device_id);
            }
        }

        // destructor
        ~block_t() { reset(); }

        // disable copy (unique ownership)
        block_t(const block_t&)            = delete;
        block_t& operator=(const block_t&) = delete;

        // enable move
        block_t(block_t&& other) noexcept
            : ptr_(other.ptr_),
              bytes_(other.bytes_),
              loc_(other.loc_),
              type_(other.type_)
        {
            other.ptr_   = nullptr;
            other.bytes_ = 0;
        }

        block_t& operator=(block_t&& other) noexcept
        {
            if (this != &other) {
                reset();
                ptr_   = other.ptr_;
                bytes_ = other.bytes_;
                loc_   = other.loc_;
                type_  = other.type_;

                other.ptr_   = nullptr;
                other.bytes_ = 0;
            }
            return *this;
        }

        // accessors
        void* data() const noexcept { return ptr_; }
        std::size_t size() const noexcept { return bytes_; }
        locality_t locality() const noexcept { return loc_; }
        memory_type_t memory_type() const noexcept { return type_; }

        // typed access
        template <typename T>
        T* as() const noexcept
        {
            return static_cast<T*>(ptr_);
        }

        // prefetch hint for managed memory
        void prefetch_to(locality_t target, exec::stream_t& stream) const
        {
            if (type_ == memory_type_t::managed && ptr_ && bytes_ > 0) {
                backend::prefetch(
                    ptr_,
                    target,
                    bytes_,
                    stream.native(),
                    stream.backend()
                );
            }
        }

      private:
        void reset()
        {
            if (ptr_) {
                backend::deallocate(loc_.backend, ptr_, type_);
                ptr_ = nullptr;
            }
            bytes_ = 0;
        }
    };

}   // namespace simbi::het::mem

#endif
