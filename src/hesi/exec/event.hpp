#ifndef HET_EXEC_EVENT_HPP
#define HET_EXEC_EVENT_HPP

#include "hesi/backend/event.hpp"
#include "hesi/core/types.hpp"
#include "hesi/exec/stream.hpp"

namespace simbi::het::exec {

    struct event_t {
        using native_handle_type = backend::event_handle_t;

        native_handle_type handle_ = nullptr;
        bool owns_resource_        = false;
        backend_type_t backend_;

        // construction
        explicit event_t(backend_type_t backend = backend_type_t::cpu)
            : backend_(backend)
        {
            if (backend_ == backend_type_t::cpu) {
                handle_        = nullptr;
                owns_resource_ = false;
            }
            else {
                handle_        = backend::create_event(backend_);
                owns_resource_ = true;
            }
        }

        // destruction
        ~event_t() { destroy(); }

        // move semantics
        event_t(event_t&& other) noexcept
            : handle_(other.handle_),
              owns_resource_(other.owns_resource_),
              backend_(other.backend_)
        {
            other.handle_        = nullptr;
            other.owns_resource_ = false;
        }

        event_t& operator=(event_t&& other) noexcept
        {
            if (this != &other) {
                destroy();
                handle_              = other.handle_;
                owns_resource_       = other.owns_resource_;
                backend_             = other.backend_;
                other.handle_        = nullptr;
                other.owns_resource_ = false;
            }
            return *this;
        }

        // disable copy
        event_t(const event_t&)            = delete;
        event_t& operator=(const event_t&) = delete;

        // operations
        void record(const stream_t& stream)
        {
            backend::record_event(backend_, handle_, stream.native());
        }

        void wait(const stream_t& stream) const
        {
            backend::wait_event(backend_, stream.native(), handle_);
        }

        void synchronize() const
        {
            backend::synchronize_event(backend_, handle_);
        }

        bool query() const { return backend::query_event(backend_, handle_); }

        // accessors
        native_handle_type native() const noexcept { return handle_; }
        backend_type_t backend() const noexcept { return backend_; }

        explicit operator bool() const noexcept
        {
            return handle_ != nullptr || backend_ == backend_type_t::cpu;
        }

      private:
        void destroy()
        {
            if (owns_resource_ && handle_) {
                backend::destroy_event(backend_, handle_);
            }
            handle_        = nullptr;
            owns_resource_ = false;
        }
    };

}   // namespace simbi::het::exec

#endif
