#ifndef ACCESSOR_HPP
#define ACCESSOR_HPP

#include "arena.hpp"
#include "config.hpp"
#include "containers/vector.hpp"
#include "domain/domain.hpp"
#include "memory/smart_ptr.hpp"
#include "traits/traits.hpp"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <stdexcept>

namespace simbi::mem {
    template <typename T, std::uint64_t Dims>
    class accessor_t;

    template <
        typename T,
        std::uint64_t Dims,
        typename ComputeField,
        typename Executor>
    void commit(
        accessor_t<T, Dims>& accessor,
        ComputeField computation,
        Executor executor
    );
    template <
        typename T,
        std::uint64_t Dims,
        typename ComputeField,
        typename Executor>
    void direct_commit(
        accessor_t<T, Dims>& accessor,
        ComputeField computation,
        Executor executor
    );
    template <
        typename T,
        std::uint64_t Dims,
        typename ComputeField,
        typename Executor>
    void try_commit(
        accessor_t<T, Dims>& accessor,
        ComputeField computation,
        Executor executor
    );

    template <typename T, std::uint64_t Dims>
    class accessor_t
    {
      private:
        mem::shared_ptr<T> data_;
        std::shared_ptr<arena_t<T>> arena_;
        domain_t<Dims> domain_;
        iarray<Dims> strides_;

        static iarray<Dims> compute_strides(const iarray<Dims>& shape)
        {
            iarray<Dims> result;
            result[Dims - 1] = 1;
            for (std::int64_t ii = Dims - 2; ii >= 0; --ii) {
                result[ii] = result[ii + 1] * shape[ii + 1];
            }
            return result;
        }

        std::size_t compute_offset(const iarray<Dims>& coord) const
        {
            return vecops::dot(coord - domain_.start, strides_);
        }

      public:
        accessor_t() = default;

        accessor_t(
            const domain_t<Dims>& domain,
            std::shared_ptr<arena_t<T>> arena = cpu_arena<T>()
        )
            : arena_(arena),
              domain_(domain),
              strides_(compute_strides(domain.shape()))
        {
            if (arena_) {
                data_ = arena_->get(domain.size());
            }
        }

        accessor_t(const domain_t<Dims>& domain, mem::shared_ptr<T> data)
            : domain_(domain),
              data_(data),
              strides_(compute_strides(domain.shape()))
        {
        }

        DUAL T& operator()(const iarray<Dims>& coord)
        {
            return data_[compute_offset(coord)];
        }

        DUAL const T& operator()(const iarray<Dims>& coord) const
        {
            return data_[compute_offset(coord)];
        }

        DUAL T* data() { return data_.get(); }
        DUAL const T* data() const { return data_.get(); }

        const domain_t<Dims>& domain() const { return domain_; }
        std::size_t size() const { return domain_.size(); }
        bool is_allocated() const { return data_ != nullptr; }
        std::shared_ptr<arena_t<T>> arena() const { return arena_; }

        accessor_t<T, Dims>
        clone(std::shared_ptr<arena_t<T>> target_arena = nullptr) const
        {
            if (!target_arena && arena_) {
                target_arena = arena_;
            }

            if (!target_arena) {
                throw std::runtime_error(
                    "Cannot clone accessor without a target arena"
                );
            }

            accessor_t<T, Dims> result(domain_, target_arena);

            if (is_allocated() && result.is_allocated()) {
                std::copy_n(data_.get(), domain_.size(), result.data());
            }

            return result;
        }

        // allow commit to access private members
        template <typename U, std::uint64_t D, typename CF, typename E>
        friend void
        commit(accessor_t<U, D>& accessor, CF computation, E executor);
        template <typename U, std::uint64_t D, typename CF, typename E>
        friend void
        direct_commit(accessor_t<U, D>& accessor, CF computation, E executor);
        template <typename U, std::uint64_t D, typename CF, typename E>
        friend void
        try_commit(accessor_t<U, D>& accessor, CF computation, E executor);
    };

    template <
        typename T,
        std::uint64_t Dims,
        typename ComputeField,
        typename Executor>
    void direct_commit(
        accessor_t<T, Dims>& accessor,
        ComputeField computation,
        Executor executor
    )
    {
        if (!accessor.is_allocated() && !accessor.arena_) {
            throw std::runtime_error(
                "Cannot commit to accessor without memory"
            );
        }

        if (!accessor.is_allocated()) {
            accessor.domain_ = computation.domain();
            accessor.strides_ =
                accessor_t<T, Dims>::compute_strides(accessor.domain_.shape());
            accessor.data_ = accessor.arena_->get(accessor.domain_.size());
        }

        executor
            .for_each(
                accessor.domain_,
                [&accessor, &computation] DUAL(const auto& coord) {
                    accessor(coord) = computation(coord);
                }
            )
            .wait();
    }

    template <
        typename T,
        std::uint64_t Dims,
        typename ComputeField,
        typename Executor>
    void try_commit(
        accessor_t<T, Dims>& accessor,
        ComputeField computation,
        Executor executor
    )
    {
        if (!accessor.is_allocated() && !accessor.arena_) {
            throw std::runtime_error(
                "Cannot commit to accessor without memory"
            );
        }

        if (!accessor.is_allocated()) {
            accessor.domain_ = computation.domain();
            accessor.strides_ =
                accessor_t<T, Dims>::compute_strides(accessor.domain_.shape());
            accessor.data_ = accessor.arena_->get(accessor.domain_.size());
        }

        auto nerrors =
            executor
                .reduce(
                    accessor.domain_,
                    std::size_t{0},
                    [&accessor, &computation] DUAL(const auto& coord) {
                        auto value = computation(coord);
                        if (value.has_value()) {
                            accessor(coord) = value.value();
                            return std::size_t{0};
                        }
                        else {
                            return std::size_t{1};
                        }
                    },
                    std::plus<std::size_t>{}
                )
                .wait();

        if (nerrors > 0) {
            throw std::runtime_error("Computation failed during commit");
        }
    }

    template <
        typename T,
        std::uint64_t Dims,
        typename ComputeField,
        typename Executor>
    void commit(
        accessor_t<T, Dims>& accessor,
        ComputeField computation,
        Executor executor
    )
    {
        if constexpr (is_maybe_v<typename ComputeField::value_type>) {
            try_commit(accessor, computation, executor);
        }
        else {
            direct_commit(accessor, computation, executor);
        }
    }

    template <typename T, std::uint64_t Dims>
    accessor_t<T, Dims> from_data(
        const T* host_data,
        const iarray<Dims>& shape,
        std::shared_ptr<arena_t<T>> arena = cpu_arena<T>()
    )
    {
        auto domain = make_domain(shape);
        accessor_t<T, Dims> result(domain, arena);

        if (result.is_allocated()) {
            std::copy_n(host_data, domain.size(), result.data());
        }

        return result;
    }

}   // namespace simbi::mem

#endif   // ACCESSOR_HPP
