#ifndef ATOMIC_BOOL_HPP
#define ATOMIC_BOOL_HPP

#include "build_options.hpp"
#include "core/types/utility/managed.hpp"
#include "managed.hpp"
#include "smart_ptr.hpp"

namespace simbi::atomic {
    template <typename T>
    class simbi_atomic
    {
      public:
        simbi_atomic()                               = delete;
        simbi_atomic(const simbi_atomic&)            = delete;
        simbi_atomic(simbi_atomic&&)                 = delete;
        simbi_atomic& operator=(const simbi_atomic&) = delete;
        simbi_atomic& operator=(simbi_atomic&&)      = delete;
        ~simbi_atomic()                              = default;

        simbi_atomic(const T& value)
            : value_(new shared_atomic_bool<Managed<>>(value))
        {
        }

        auto* operator->() const { return value_.get(); }

        auto& operator*() const { return *value_; }

        bool operator==(const simbi_atomic& other) const
        {
            return value_->load() == other.value_->load();
        }

        bool operator!=(const simbi_atomic& other) const
        {
            return value_->load() != other.value_->load();
        }

        auto* get() const { return value_.get(); }
        T load() const
        {
            gpu::api::deviceSynch();
            return value_->load();
        }

      private:
        util::smart_ptr<shared_atomic_bool<Managed<>>> value_;
    };
}   // namespace simbi::atomic

#endif
