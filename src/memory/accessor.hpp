#ifndef UNIFIED_ACCESSOR_HPP
#define UNIFIED_ACCESSOR_HPP

#include "buffer_pool.hpp"
#include "config.hpp"
#include "containers/vector.hpp"
#include "domain/domain.hpp"
#include "io/exceptions.hpp"
#include "traits/traits.hpp"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <utility>

namespace simbi::mem {
    /**
     * accessor_t - host-device agnostic field storage
     *
     * srp: provide coordinate-based access to arena-backed memory
     * works on both cpu and gpu via hd annotations
     */
    template <typename T, std::uint64_t Dims>
    class accessor_t
    {
        buffer_id_t buffer_id_;
        std::shared_ptr<buffer_pool_t<T>> pool_;
        domain_t<Dims> domain_;
        iarray<Dims> strides_;

      public:
        using value_type                          = T;
        static constexpr std::uint64_t dimensions = Dims;

        // construction
        accessor_t() = default;

        accessor_t(
            domain_t<Dims> domain,
            std::shared_ptr<buffer_pool_t<T>> pool = global_buffer_pool<T>()
        )
            : pool_(std::move(pool)),
              domain_(domain),
              strides_(compute_strides(domain.shape()))
        {
            buffer_id_ = pool_->allocate(domain_.size());
        }

        // copy semantics - share buffer via shared_ptr
        accessor_t(const accessor_t&)            = default;
        accessor_t& operator=(const accessor_t&) = default;
        accessor_t(accessor_t&&)                 = default;
        accessor_t& operator=(accessor_t&&)      = default;

        // host-device element access
        DUAL const T& operator[](coordinate_t<Dims> coord) const
        {
            return pool_->get_data(buffer_id_)[compute_offset(coord)];
        }

        DUAL T& operator[](coordinate_t<Dims> coord)
        {
            return pool_->get_data(buffer_id_)[compute_offset(coord)];
        }

        DUAL const T& operator()(coordinate_t<Dims> coord) const
        {
            return (*this)[coord];
        }

        // direct data access
        DUAL const T* data() const { return pool_->get_data(buffer_id_); }

        DUAL T* data() { return pool_->get_data(buffer_id_); }

        // queries
        const domain_t<Dims>& domain() const { return domain_; }
        std::size_t size() const { return domain_.size(); }
        bool is_allocated() const { return pool_ && pool_->exists(buffer_id_); }

        // materialization interface for compute fields
        template <typename ComputeField, typename Executor>
        void commit(const ComputeField& computation, const Executor& executor)
        {
            // ensure we have a valid pool
            if (!pool_) {
                pool_ = global_buffer_pool<T>();
            }

            // ensure we have allocated memory
            if (!is_allocated()) {
                // allocate with same domain as computation
                buffer_id_ = pool_->allocate(computation.domain().size());
                domain_    = computation.domain();
                strides_   = compute_strides(domain_.shape());
            }

            using source_result_t = ComputeField::value_type;

            if constexpr (is_maybe_v<source_result_t>) {
                // maybe types - use parallel execution with error counting
                auto error_count =
                    executor
                        .reduce(
                            domain_,
                            std::size_t{0},
                            [this, computation] DUAL(coordinate_t<Dims> coord)
                                -> std::size_t {
                                auto maybe_result = computation(coord);
                                if (maybe_result.has_value()) {
                                    (*this)[coord] = maybe_result.value();
                                    return 0;   // no error
                                }
                                else {
                                    return 1;   // error occurred
                                }
                            },
                            std::plus{}
                        )
                        .wait();

                if (error_count > 0) {
                    throw exception::SimulationFailureException();
                }
            }
            else {
                // regular types - standard parallel execution
                executor
                    .for_each(
                        domain_,
                        [this, computation] DUAL(coordinate_t<Dims> coord) {
                            (*this)[coord] = computation.function(coord);
                        }
                    )
                    .wait();
            }
        }

        auto clone() const
        {
            auto new_accessor = accessor_t{domain_, pool_};
            std::copy_n(this->data(), size(), new_accessor.data());
            return new_accessor;
        }

        // factory methods
        static auto zeros(
            const iarray<Dims>& shape,
            std::shared_ptr<buffer_pool_t<T>> pool = global_buffer_pool<T>()
        )
        {
            auto domain   = make_domain(shape);
            auto accessor = accessor_t{domain, std::move(pool)};

            // zero-fill the buffer
            auto* data_ptr = accessor.data();
            std::fill_n(data_ptr, domain.size(), T{});

            return accessor;
        }

        static auto from_numpy(
            T* numpy_data,
            const iarray<Dims>& shape,
            std::shared_ptr<buffer_pool_t<T>> pool = global_buffer_pool<T>()
        )
        {
            auto domain   = make_domain(shape);
            auto accessor = accessor_t{domain, std::move(pool)};

            // copy from numpy
            std::copy_n(numpy_data, domain.size(), accessor.data());

            return accessor;
        }

      private:
        DUAL std::size_t compute_offset(coordinate_t<Dims> coord) const
        {
            return vecops::dot(coord - domain_.start, strides_);
        }

        static iarray<Dims> compute_strides(const iarray<Dims>& shape)
        {
            iarray<Dims> strides;
            strides[Dims - 1] = 1;
            for (std::int64_t i = Dims - 2; i >= 0; --i) {
                strides[i] = strides[i + 1] * shape[i + 1];
            }
            return strides;
        }
    };

}   // namespace simbi::mem

#endif
