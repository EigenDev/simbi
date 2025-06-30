#ifndef SIMBI_FIELD_HPP
#define SIMBI_FIELD_HPP

#include "compute/math/cfd_expressions.hpp"
#include "core/base/concepts.hpp"
#include "core/base/memory.hpp"         // for unified_memory_t
#include "data/containers/vector.hpp"   // for vector_t
#include "index_space.hpp"              // for index_space_t
#include <cstddef>                      // for std::uint64_t
#include <cstdint>                      // for std::uint64_t
#include <initializer_list>             // for std::initializer_list
#include <iterator>                     // for std::rbegin
#include <memory>                       // for std::shared_ptr
#include <stdexcept>

namespace simbi {
    using namespace simbi::base;

    using ulist = std::initializer_list<std::uint64_t>;

    // pure functional field - immutable spatial function
    template <typename T, std::uint64_t Dims = 1>
    struct field_t {
        index_space_t<Dims> domain_;
        std::shared_ptr<unified_memory_t<T>> data_;
        using value_type                          = T;
        static constexpr std::uint64_t dimensions = Dims;

        // function evaluation
        const T& operator()(const uarray<Dims>& coord) const
        {
            return at(coord);
        }

        const T& at(const uarray<Dims>& coord) const
        {
            auto linear_idx = domain_to_memory_index(coord);
            return data_->data()[linear_idx];
        }

        const T& operator[](const uarray<Dims>& coord) const
        {
            return at(coord);
        }

        T& operator[](const uarray<Dims>& coord)
        {
            auto linear_idx = domain_to_memory_index(coord);
            return data_->data()[linear_idx];
        }

        T& operator[](std::uint64_t ii) { return data_->data()[ii]; }

        const T& operator[](std::uint64_t ii) const
        {
            return data_->data()[ii];
        }

        // spatial transformations - return new fields with shared memory
        field_t operator[](const index_space_t<Dims>& subdomain) const
        {
            // create new field with restricted domain but same data
            auto restricted_domain = domain_[subdomain];
            return field_t{restricted_domain, data_};
        }

        field_t contract(std::uint64_t radius) const
        {
            return field_t{domain_.contract(radius), data_};
        }

        field_t contract(const uarray<Dims>& radii) const
        {
            return field_t{domain_.contract(radii), data_};
        }

        field_t expand(std::uint64_t radius) const
        {
            return field_t{domain_.expand(radius), data_};
        }

        field_t expand(const uarray<Dims>& radii) const
        {
            return field_t{domain_.expand(radii), data_};
        }

        // explicit copying when needed
        field_t copy() const
        {
            auto new_data =
                std::make_shared<unified_memory_t<T>>(data_->size());
            std::copy(
                data_->data(),
                data_->data() + data_->size(),
                new_data->data()
            );
            return field_t{domain_, new_data};
        }

        // accessors for integration with expression system
        std::uint64_t size() const { return domain_.size(); }
        const T* data() const { return data_->data(); }
        T* data() { return data_->data(); }
        bool null() const { return size() == 0; }
        const index_space_t<Dims>& domain() const { return domain_; }
        const std::shared_ptr<unified_memory_t<T>>& memory() const
        {
            return data_;
        }

        auto concat(const T& val) const
        {
            // create a new field with an additional element
            auto new_domain = domain_.expand({1});
            auto new_data =
                std::make_shared<unified_memory_t<T>>(new_domain.size());
            std::copy(
                data_->data(),
                data_->data() + data_->size(),
                new_data->data()
            );
            new_data->data()[new_domain.size() - 1] = val;
            return field_t{new_domain, new_data};
        }

        // shape convenience method
        uarray<Dims> shape() const { return domain_.shape(); }

        // GPU/CPU management
        void to_gpu() { data_->to_gpu(); }
        void to_cpu() { data_->to_cpu(); }

        // factory functions for clean construction
        static field_t make_field(const uarray<Dims>& shape)
        {
            auto domain = make_space(shape);
            auto data   = std::make_shared<unified_memory_t<T>>(domain.size());
            return field_t<T, Dims>{domain, data};
        }

        static field_t
        make_field(const uarray<Dims>& start, const uarray<Dims>& end)
        {
            auto domain = make_space(start, end);
            auto data   = std::make_shared<unified_memory_t<T>>(domain.size());
            return field_t<T, Dims>{domain, data};
        }

        static field_t
        wrap_external(T* ptr, const ulist& lshape, bool take_ownership = false)
        {
            uarray<Dims> shape;
            for (std::uint64_t ii = 0; ii < Dims && ii < lshape.size(); ++ii) {
                shape[ii] = *(std::rbegin(lshape) + ii);
            }
            auto domain = make_space(shape);
            auto data   = std::make_shared<unified_memory_t<T>>();
            if (take_ownership) {
                // take ownership - the memory will be managed by
                // unified_memory_t
                data->set_data(ptr, domain.size(), true);
            }
            else {
                // wrap external memory without taking ownership
                data->wrap_external_memory(ptr, domain.size());
            }
            // std::cout << "domain from wrapped: " << domain << std::endl;
            return field_t<T, Dims>{domain, data};
        }

        static field_t zeros(const ulist& lshape)
        {
            uarray<Dims> shape;
            for (std::uint64_t ii = 0; ii < Dims && ii < lshape.size(); ++ii) {
                shape[ii] = *(std::rbegin(lshape) + ii);
            }
            auto field = field_t<T, Dims>::make_field(shape);
            if constexpr (is_any_state_variable_c<T>) {
                std::fill(field.data(), field.data() + field.size(), T{});
            }
            else {
                std::fill(field.data(), field.data() + field.size(), T{0});
            }
            return field;
        }

        static field_t zeros(const uarray<Dims>& shape)
        {
            auto field = field_t<T, Dims>::make_field(shape);
            if constexpr (is_any_state_variable_c<T>) {
                std::fill(field.data(), field.data() + field.size(), T{});
            }
            else {
                std::fill(field.data(), field.data() + field.size(), T{0});
            }
            return field;
        }

        static field_t ones(const ulist& lshape)
        {
            uarray<Dims> shape;
            for (std::uint64_t ii = 0; ii < Dims && ii < lshape.size(); ++ii) {
                shape[ii] = *(std::rbegin(lshape) + ii);
            }
            auto field = field_t<T, Dims>::make_field(shape);
            if constexpr (is_any_state_variable_c<T>) {
                std::fill(field.data(), field.data() + field.size(), T{});
            }
            else {
                std::fill(field.data(), field.data() + field.size(), T{1});
            }
            return field;
        }

        static field_t ones(const uarray<Dims>& shape)
        {
            auto field = field_t<T, Dims>::make_field(shape);
            if constexpr (is_any_state_variable_c<T>) {
                std::fill(field.data(), field.data() + field.size(), T{});
            }
            else {
                std::fill(field.data(), field.data() + field.size(), T{1});
            }
            return field;
        }

        // iterator support
        auto begin() const { return std::make_move_iterator(data_->data()); }

        auto end() const
        {
            return std::make_move_iterator(data_->data() + data_->size());
        }

        template <typename Op>
        auto operator|(Op&& op) const
        {
            return cfd::make_field_expr(*this) | std::forward<Op>(op);
        }

      private:
        // convert domain coordinate to linear memory index
        std::uint64_t domain_to_memory_index(const uarray<Dims>& coord) const
        {
            // delegate to index_space_t once it has coord_to_linear_index()
            return domain_.coord_to_linear_index(coord);
        }
    };

}   // namespace simbi

#endif   // SIMBI_FIELD_HPP
