#ifndef HET_EXEC_STREAM_HPP
#define HET_EXEC_STREAM_HPP

#include "hesi/backend/stream.hpp"
#include "hesi/core/types.hpp"

#include <cstdint>

namespace simbi::het::exec {

    struct stream_t {
        using native_handle_type = backend::stream_handle_t;

        native_handle_type handle_ = nullptr;
        bool owns_resource_        = false;
        backend_type_t backend_;
        locality_t locality_;

        // construction
        explicit stream_t(
            backend_type_t backend = backend_type_t::cpu,
            std::int32_t device_id = 0
        )
            : backend_(backend), locality_({backend, device_id})
        {
            if (backend_ == backend_type_t::cpu) {
                handle_        = nullptr;
                owns_resource_ = false;
            }
            else {
                handle_        = backend::create_stream(backend_, device_id);
                owns_resource_ = true;
            }
        }

        // construct from locality
        explicit stream_t(locality_t loc)
            : backend_(loc.backend), locality_(loc)
        {
            if (backend_ == backend_type_t::cpu) {
                handle_        = nullptr;
                owns_resource_ = false;
            }
            else {
                handle_ = backend::create_stream(loc.backend, loc.device_id);
                owns_resource_ = true;
            }
        }

        // destruction
        ~stream_t() { destroy(); }

        // move semantics
        stream_t(stream_t&& other) noexcept
            : handle_(other.handle_),
              owns_resource_(other.owns_resource_),
              backend_(other.backend_),
              locality_(other.locality_)
        {
            other.handle_        = nullptr;
            other.owns_resource_ = false;
        }

        stream_t& operator=(stream_t&& other) noexcept
        {
            if (this != &other) {
                destroy();
                handle_              = other.handle_;
                owns_resource_       = other.owns_resource_;
                backend_             = other.backend_;
                locality_            = other.locality_;
                other.handle_        = nullptr;
                other.owns_resource_ = false;
            }
            return *this;
        }

        // disable copy
        stream_t(const stream_t&)            = delete;
        stream_t& operator=(const stream_t&) = delete;

        // operations
        void synchronize() const
        {
            backend::synchronize_stream(backend_, handle_);
        }

        bool query_complete() const
        {
            return backend::query_stream(backend_, handle_);
        }

        // accessors
        native_handle_type native() const noexcept { return handle_; }
        backend_type_t backend() const noexcept { return backend_; }
        locality_t locality() const noexcept { return locality_; }

        explicit operator bool() const noexcept
        {
            return handle_ != nullptr || backend_ == backend_type_t::cpu;
        }

      private:
        void destroy()
        {
            if (owns_resource_ && handle_) {
                // synchronize before destroying to avoid use-after-free
                backend::synchronize_stream(backend_, handle_);
                backend::destroy_stream(backend_, handle_);
            }
            handle_        = nullptr;
            owns_resource_ = false;
        }
    };

    static inline stream_t make_a_default_stream()
    {
#if defined(CUDA_ENABLED)
        return stream_t(backend_type_t::cuda, 0);
#elif defined(HIP_ENABLED)
        return stream_t(backend_type_t::hip, 0);
#else
        return stream_t(backend_type_t::cpu, 0);
#endif
    }

}   // namespace simbi::het::exec

#endif
