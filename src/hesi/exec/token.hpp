#ifndef HET_EXEC_TOKEN_HPP
#define HET_EXEC_TOKEN_HPP

#include "hesi/core/types.hpp"
#include "hesi/exec/event.hpp"

#include <utility>

namespace simbi::het::exec {

    // forward declaration
    struct stream_t;

    // token_t: lightweight token representing async completion
    // stores a native backend event handle; can be owning or non-owning
    // move-only type; destruction only destroys owned native handle
    struct token_t {
        using native_handle_type = backend::event_handle_t;

        native_handle_type native_handle = nullptr;
        backend_type_t backend_          = backend_type_t::cpu;
        bool owns_resource_              = true;

        // empty token = no work
        token_t()
            : native_handle(nullptr),
              backend_(backend_type_t::cpu),
              owns_resource_(false)
        {
        }

        // construct an owning or non-owning token from a native handle
        // owns_resource indicates whether the token is responsible for
        // destroying the native handle on destruction
        explicit token_t(
            native_handle_type h,
            backend_type_t backend,
            bool owns_resource = true
        )
            : native_handle(h), backend_(backend), owns_resource_(owns_resource)
        {
        }

        // disable copy; support move semantics to transfer ownership flag
        token_t(const token_t&)            = delete;
        token_t& operator=(const token_t&) = delete;

        token_t(token_t&& other) noexcept
            : native_handle(other.native_handle),
              backend_(other.backend_),
              owns_resource_(other.owns_resource_)
        {
            other.native_handle  = nullptr;
            other.backend_       = backend_type_t::cpu;
            other.owns_resource_ = false;
        }

        token_t& operator=(token_t&& other) noexcept
        {
            if (this != &other) {
                if (native_handle && owns_resource_) {
                    backend::destroy_event(backend_, native_handle);
                }
                native_handle        = other.native_handle;
                backend_             = other.backend_;
                owns_resource_       = other.owns_resource_;
                other.native_handle  = nullptr;
                other.backend_       = backend_type_t::cpu;
                other.owns_resource_ = false;
            }
            return *this;
        }

        ~token_t()
        {
            if (native_handle && owns_resource_) {
                backend::destroy_event(backend_, native_handle);
                native_handle  = nullptr;
                owns_resource_ = false;
            }
        }

        // allocate a new native event for the backend and return owning token
        static token_t create(backend_type_t backend)
        {
            native_handle_type h = nullptr;
            if (backend != backend_type_t::cpu) {
                h = backend::create_event(backend);
            }
            return token_t(h, backend, true);
        }

        // immediate token: represents already-complete / no-op (non-owning)
        static token_t immediate(backend_type_t /*backend*/)
        {
            return token_t(nullptr, backend_type_t::cpu, false);
        }

        // record completion on an execution stream
        void record(const stream_t& stream) const
        {
            if (native_handle) {
                backend::record_event(backend_, native_handle, stream.native());
            }
        }

        // wait for the recorded event on the provided stream
        void wait(const stream_t& stream) const
        {
            if (native_handle) {
                backend::wait_event(backend_, stream.native(), native_handle);
            }
        }

        void synchronize() const
        {
            if (native_handle) {
                backend::synchronize_event(backend_, native_handle);
            }
        }

        bool query() const
        {
            if (!native_handle) {
                return true;
            }
            return backend::query_event(backend_, native_handle);
        }

        explicit operator bool() const { return native_handle != nullptr; }

        backend_type_t backend() const { return backend_; }

        // return the native handle (may be null). non-owning callers can create
        // a token wrapper with owns_resource=false if they need to return a
        // non-owning view of the same native event.
        native_handle_type native() const { return native_handle; }
    };

}   // namespace simbi::het::exec

#endif
