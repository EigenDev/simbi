#ifndef ATOMIC_BOOL_HPP
#define ATOMIC_BOOL_HPP

#include "adapter/device_types.hpp"
#include "core/types/utility/managed.hpp"
#include "managed.hpp"
#include "smart_ptr.hpp"

namespace simbi::atomic {
    using shared_atomic_bool_t = adapter::types::atomic_bool<>;
    template <typename T>
    class simbi_atomic : public Managed<global::managed_memory>
    {
      public:
        simbi_atomic()                               = delete;
        simbi_atomic(const simbi_atomic&)            = delete;
        simbi_atomic(simbi_atomic&&)                 = delete;
        simbi_atomic& operator=(const simbi_atomic&) = delete;
        simbi_atomic& operator=(simbi_atomic&&)      = delete;
        ~simbi_atomic()                              = default;

        simbi_atomic(const T& value) : value_(new shared_atomic_bool_t(value))
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
            gpu::api::device_synch();
            return value_->load();
        }

      private:
        util::smart_ptr<shared_atomic_bool_t> value_;
    };
}   // namespace simbi::atomic

#endif
