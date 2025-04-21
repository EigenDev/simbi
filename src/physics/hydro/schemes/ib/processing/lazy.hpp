
#ifndef LAZY_OPS_HPP
#define LAZY_OPS_HPP

#include "build_options.hpp"
#include "core/types/monad/maybe.hpp"
#include "physics/hydro/schemes/ib/systems/capability.hpp"
#include "physics/hydro/schemes/ib/systems/component_body_system.hpp"

namespace simbi::ibsystem {

    template <typename T, size_type Dims>
    class LazyCapabilityView
    {
      public:
        // body-index pair for iterator to return
        struct BodyWithIndex {
            size_t index;
            Body<T, Dims> body;
        };

        // ctor takes system and requested capability
        LazyCapabilityView(
            const ComponentBodySystem<T, Dims>& system,
            BodyCapability capability
        )
            : system_(system), capability_(capability)
        {
        }

        // iterator class that only yields bodies with matching capability
        class iterator
        {
          private:
            // pointer to original system
            const ComponentBodySystem<T, Dims>* system_;
            // capability to filter for
            BodyCapability capability_;
            // current body index
            size_t current_index_;
            // max body index (system size)
            size_t max_index_;
            // current body being pointed to
            std::optional<BodyWithIndex> current_value_;

            // find the next body with the required capability
            DUAL void find_next_valid()
            {
                current_value_.reset();   // reset current value
                // scan forward until we find a matching body or reach the end
                while (current_index_ < max_index_) {
                    auto maybe_body = system_->get_body(current_index_);
                    if (maybe_body.has_value()) {
                        const auto& body = maybe_body.value();

                        if (body.has_capability(capability_)) {
                            // found a valid body - store it
                            current_value_ =
                                BodyWithIndex{current_index_, body};
                            break;
                        }
                    }
                    ++current_index_;   // move to next index
                }
            }

          public:
            // iterator traits
            using difference_type   = std::ptrdiff_t;
            using value_type        = BodyWithIndex;
            using pointer           = const BodyWithIndex*;
            using reference         = const BodyWithIndex&;
            using iterator_category = std::input_iterator_tag;

            // constructor - finds first valid body
            DUAL iterator(
                const ComponentBodySystem<T, Dims>* system,
                BodyCapability capability,
                size_t start_index,
                size_t max_index
            )
                : system_(system),
                  capability_(capability),
                  current_index_(start_index),
                  max_index_(max_index)
            {
                find_next_valid();   // initialize to first valid body
            }

            // dereference operator
            DUAL reference operator*() const
            {
                if (!current_value_) {
                    throw std::runtime_error("Dereferencing invalid iterator");
                }
                return *current_value_;
            }

            // arrow operator
            DUAL pointer operator->() const
            {
                if (!current_value_) {
                    throw std::runtime_error("Dereferencing invalid iterator");
                }
                return &(*current_value_);
            }

            // pre-increment
            DUAL iterator& operator++()
            {
                if (current_index_ < max_index_) {
                    ++current_index_;
                    find_next_valid();
                }
                return *this;
            }

            // post-increment
            DUAL iterator operator++(int)
            {
                iterator tmp = *this;
                ++(*this);
                return tmp;
            }

            // equality comparison
            DUAL bool operator==(const iterator& other) const
            {
                return current_index_ == other.current_index_;
            }

            // inequality comparison
            DUAL bool operator!=(const iterator& other) const
            {
                return !(*this == other);
            }
        };

        // begin iterator - starts from first body
        DUAL iterator begin() const
        {
            return iterator(&system_, capability_, 0, system_.size());
        }

        // end iterator - positioned at end
        DUAL iterator end() const
        {
            return iterator(
                &system_,
                capability_,
                system_.size(),
                system_.size()
            );
        }

      private:
        // reference to original system
        const ComponentBodySystem<T, Dims>& system_;
        // capability to filter for
        BodyCapability capability_;
    };

}   // namespace simbi::ibsystem

#endif
