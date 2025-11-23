#ifndef HET_EXEC_TOKEN_HPP
#define HET_EXEC_TOKEN_HPP

#include "hesi/core/types.hpp"
#include "hesi/exec/event.hpp"
#include "hesi/mem/rc.hpp"

#include <utility>

namespace simbi::het::exec {

    // forward declaration
    struct stream_t;

    // shared handle to synchronization event
    // lightweight, copyable, passed by value
    // represents "when does this work complete?"
    struct token_t {
        using handle_type = mem::handle_t<event_t>;

        handle_type event_;

        // default constructor (empty token)
        token_t() = default;

        // construct from existing event handle
        explicit token_t(handle_type h) : event_(std::move(h)) {}

        // factory: create new event and return token
        static token_t create(backend_type_t backend)
        {
            return token_t(handle_type::make(backend));
        }

        // factory: create "already complete" token
        static token_t immediate(backend_type_t backend)
        {
            (void) backend;     // unused, but kept for API consistency
            return token_t{};   // empty token = already complete
        }

        // operations
        void wait(const stream_t& stream) const;   // defined after stream_t

        void synchronize() const
        {
            if (event_) {
                event_->synchronize();
            }
        }

        bool is_ready() const
        {
            if (!event_) {
                return true;   // empty token = always ready
            }
            return event_->query();
        }

        // check if token represents actual work
        explicit operator bool() const { return static_cast<bool>(event_); }

        // accessor
        backend_type_t backend() const
        {
            return event_ ? event_->backend() : backend_type_t::cpu;
        }
    };

    // implementation of wait (requires complete stream_t definition)
    inline void token_t::wait(const stream_t& stream) const
    {
        if (event_) {
            event_->wait(stream);
        }
    }

}   // namespace simbi::het::exec

#endif
