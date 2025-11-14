#ifndef FIELD_HPP
#define FIELD_HPP

#include "base/concepts.hpp"
#include "compat.hpp"
#include "computation.hpp"
#include "containers/vector.hpp"
#include "domain/domain.hpp"
#include "exec_context.hpp"
#include "io/exceptions.hpp"
#include "memory/arena.hpp"
#include "memory/buffer.hpp"
#include "memory/device.hpp"
#include "traits/traits.hpp"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <pybind11/cast.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <stdexcept>
#include <type_traits>

namespace py = pybind11;
namespace simbi {

    // forward declaration
    template <typename T, std::uint64_t Dims>
    struct field_t;

    namespace detail {
        // check if type is a buffer
        template <typename>
        struct is_buffer : std::false_type {
        };

        template <typename T, std::uint64_t Dims>
        struct is_buffer<mem::buffer_t<T, Dims>> : std::true_type {
        };

        template <typename T>
        inline constexpr bool is_buffer_v = is_buffer<T>::value;
    }   // namespace detail

    // user-facing field type: buffer + lazy computation API
    // wraps buffer_t, provides computation graph building
    template <typename T, std::uint64_t Dims>
    struct field_t {
        using value_type                          = T;
        static constexpr std::uint64_t dimensions = Dims;

        mem::buffer_t<T, Dims> buffer;

        // default ctor
        field_t() = default;

        // construction from buffer
        explicit field_t(mem::buffer_t<T, Dims> buf) : buffer(std::move(buf)) {}

        // construction with allocation
        field_t(
            const domain_t<Dims>& domain,
            mem::device_t dev = mem::device_t::cpu()
        )
            : buffer(domain, dev)
        {
        }

        field_t(
            const domain_t<Dims>& domain,
            std::shared_ptr<mem::arena_t<T>> arena
        )
            : buffer(domain, arena)
        {
        }

        // queries - forward to buffer
        const domain_t<Dims>& domain() const { return buffer.domain(); }
        std::size_t size() const { return buffer.size(); }
        mem::device_t device() const { return buffer.device; }

        T* data() { return buffer.data(); }
        const T* data() const { return buffer.data(); }

        // get accessor for building computations
        auto accessor() const { return buffer.accessor(); }

        // create computation from this field's data
        auto as_computation() const
        {
            return computation(buffer.domain(), buffer.accessor());
        }

        DUAL T& operator()(const coordinate_t<Dims>& coord)
        {
            return buffer(coord);
        }

        DUAL const T& operator()(const coordinate_t<Dims>& coord) const
        {
            return buffer(coord);
        }

        DUAL T& operator[](std::uint64_t idx)
        {
            const auto coord = buffer.domain().linear_to_coord(idx);
            return buffer(coord);
        }

        DUAL const T& operator[](std::uint64_t idx) const
        {
            const auto coord = buffer.domain().linear_to_coord(idx);
            return buffer(coord);
        }

        // lazy operations - return computations, not fields
        template <typename F>
        auto map(F func) const
        {
            return as_computation().map(std::move(func));
        }

        template <typename F>
        auto enum_map(F func) const
        {
            return as_computation().enum_map(std::move(func));
        }

        template <typename F>
        auto coord_map(F func) const
        {
            return as_computation().coord_map(std::move(func));
        }

        template <typename G, typename BinaryOp>
        auto zip(const computation_t<Dims, G>& other, BinaryOp op) const
        {
            return as_computation().zip(other, std::move(op));
        }

        template <typename G, typename BinaryOp>
        auto zip(const field_t<T, Dims>& other, BinaryOp op) const
        {
            return as_computation().zip(other.as_computation(), std::move(op));
        }

        auto operator[](const domain_t<Dims>& subdomain) const
        {
            return slice(subdomain);
        }

        auto at(const domain_t<Dims>& subdomain) const
        {
            return slice(subdomain);
        }

        template <typename G>
        auto insert(const computation_t<Dims, G>& overlay) const
        {
            return as_computation().insert(overlay);
        }

        //  assignment materializes computation into buffer
        template <typename F>
        field_t& operator=(const computation_t<Dims, F>& comp)
        {
            auto& ctx      = current_context();
            auto& executor = ctx.get_executor(buffer.device);
            commit(comp, executor);
            return *this;
        }

        template <typename ComputeField, typename Executor>
        void direct_commit(ComputeField computation, const Executor& executor)
        {
            auto acc = buffer.accessor();

            executor
                .for_each(
                    buffer.domain(),
                    [acc, computation] DUAL(coordinate_t<Dims> coord) {
                        acc(coord) = computation(coord);
                    }
                )
                .wait();
        }

        template <typename Computation, typename Executor>
        void try_commit(Computation computation, const Executor& executor)
        {
            auto acc = accessor();

            auto nerrors =
                executor
                    .reduce(
                        domain(),
                        std::size_t{0},
                        [acc, computation] DUAL(coordinate_t<Dims> coord) {
                            auto value = computation(coord);
                            if (value.has_value()) {
                                acc(coord) = value.value();
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
                throw exception::SimulationFailureException();
            }
        }

        // commit with explicit executor
        template <typename Computation, typename Executor>
        void commit(const Computation& comp, const Executor& executor)
        {
            if constexpr (is_maybe_v<typename Computation::value_type>) {
                try_commit(comp, executor);
            }
            else {
                direct_commit(comp, executor);
            }
        }

        // clone to same or different device
        field_t<T, Dims> clone(mem::device_t target_device) const
        {
            return field_t<T, Dims>(buffer.clone(target_device));
        }

        field_t<T, Dims> clone() const
        {
            return field_t<T, Dims>(buffer.clone());
        }

        // slice: create field view into subdomain
        field_t<T, Dims> slice(const domain_t<Dims>& subdomain) const
        {
            return field_t<T, Dims>(buffer.slice(subdomain));
        }
    };

    // arithmetic operators - forward to computation
    template <typename T, std::uint64_t Dims>
    auto operator+(const field_t<T, Dims>& a, const field_t<T, Dims>& b)
    {
        return a.as_computation() + b.as_computation();
    }

    template <typename T, std::uint64_t Dims>
    auto operator-(const field_t<T, Dims>& a, const field_t<T, Dims>& b)
    {
        return a.as_computation() - b.as_computation();
    }

    template <typename T, std::uint64_t Dims>
    auto operator*(const field_t<T, Dims>& a, const field_t<T, Dims>& b)
    {
        return a.as_computation() * b.as_computation();
    }

    template <typename T, std::uint64_t Dims>
    auto operator/(const field_t<T, Dims>& a, const field_t<T, Dims>& b)
    {
        return a.as_computation() / b.as_computation();
    }

    // field-computation mixed operations
    template <typename T, std::uint64_t Dims, typename F>
    auto
    operator+(const field_t<T, Dims>& field, const computation_t<Dims, F>& comp)
    {
        return field.as_computation() + comp;
    }

    template <typename T, std::uint64_t Dims, typename F>
    auto
    operator+(const computation_t<Dims, F>& comp, const field_t<T, Dims>& field)
    {
        return comp + field.as_computation();
    }

    template <typename T, std::uint64_t Dims, typename F>
    auto
    operator-(const field_t<T, Dims>& field, const computation_t<Dims, F>& comp)
    {
        return field.as_computation() - comp;
    }

    template <typename T, std::uint64_t Dims, typename F>
    auto
    operator-(const computation_t<Dims, F>& comp, const field_t<T, Dims>& field)
    {
        return comp - field.as_computation();
    }

    template <typename T, std::uint64_t Dims, typename F>
    auto
    operator*(const field_t<T, Dims>& field, const computation_t<Dims, F>& comp)
    {
        return field.as_computation() * comp;
    }

    template <typename T, std::uint64_t Dims, typename F>
    auto
    operator*(const computation_t<Dims, F>& comp, const field_t<T, Dims>& field)
    {
        return comp * field.as_computation();
    }

    template <typename T, std::uint64_t Dims, typename F>
    auto
    operator/(const field_t<T, Dims>& field, const computation_t<Dims, F>& comp)
    {
        return field.as_computation() / comp;
    }

    template <typename T, std::uint64_t Dims, typename F>
    auto
    operator/(const computation_t<Dims, F>& comp, const field_t<T, Dims>& field)
    {
        return comp / field.as_computation();
    }

    // scalar operations
    template <typename T, std::uint64_t Dims, typename Scalar>
    auto operator*(const field_t<T, Dims>& field, Scalar scalar)
    {
        return field.as_computation() * scalar;
    }

    template <typename T, std::uint64_t Dims, typename Scalar>
    auto operator*(Scalar scalar, const field_t<T, Dims>& field)
    {
        return scalar * field.as_computation();
    }

    template <typename T, std::uint64_t Dims, typename Scalar>
    auto operator/(const field_t<T, Dims>& field, Scalar scalar)
    {
        return field.as_computation() / scalar;
    }

    template <typename T, std::uint64_t Dims, typename Scalar>
    auto operator+(const field_t<T, Dims>& field, Scalar scalar)
    {
        return field.as_computation() + scalar;
    }

    template <typename T, std::uint64_t Dims, typename Scalar>
    auto operator+(Scalar scalar, const field_t<T, Dims>& field)
    {
        return scalar + field.as_computation();
    }

    template <typename T, std::uint64_t Dims, typename Scalar>
    auto operator-(const field_t<T, Dims>& field, Scalar scalar)
    {
        return field.as_computation() - scalar;
    }

    template <typename T, std::uint64_t Dims, typename Scalar>
    auto operator-(Scalar scalar, const field_t<T, Dims>& field)
    {
        return scalar - field.as_computation();
    }

    // factory functions
    template <typename T, std::uint64_t Dims>
    field_t<T, Dims> field(
        const domain_t<Dims>& domain,
        mem::device_t dev = mem::device_t::cpu()
    )
    {
        return field_t<T, Dims>(domain, dev);
    }

    template <typename T, std::uint64_t Dims>
    field_t<T, Dims>
    field(const domain_t<Dims>& domain, std::shared_ptr<mem::arena_t<T>> arena)
    {
        return field_t<T, Dims>(domain, arena);
    }

    // fields from python generators
    struct cell_centered_tag {
    };
    struct face_centered_tag {
    };

    template <typename State>
    State py_tuple_to_state(py::handle obj)
    {
        State s;
        std::uint64_t chi_position = State::dimensions + 3;

        // check if it's a tuple/sequence
        if (py::isinstance<py::tuple>(obj)) {
            auto tuple = obj.cast<py::tuple>();
            for (std::uint64_t ii = 0; ii < tuple.size(); ++ii) {
                if constexpr (is_hydro_primitive_c<State>) {
                    if (chi_position == ii) {
                        s[ii + State::dimensions] = tuple[ii].cast<real>();
                    }
                    else {
                        s[ii] = tuple[ii].cast<real>();
                    }
                }
                else {
                    s[ii] = tuple[ii].cast<real>();
                }
            }
        }
        else if (py::isinstance<py::sequence>(obj)) {
            auto seq = obj.cast<py::list>();
            for (std::uint64_t ii = 0; ii < seq.size(); ++ii) {
                if constexpr (is_hydro_primitive_c<State>) {
                    if (chi_position == ii) {
                        s[ii + State::dimensions] = seq[ii].cast<real>();
                    }
                    else {
                        s[ii] = seq[ii].cast<real>();
                    }
                }
                else {
                    s[ii] = seq[ii].cast<real>();
                }
            }
        }
        else {
            throw std::runtime_error(
                "Expected tuple or sequence from generator"
            );
        }

        return s;
    }

    // template <
    //     typename element_type,
    //     std::uint64_t Dims,
    //     typename Tag = cell_centered_tag>
    // field_t<element_type, Dims> from_generator(
    //     py::iterator gen,
    //     const iarray<Dims>& full_shape,
    //     std::uint64_t halo_radius,
    //     std::uint64_t direction = 0
    // )
    // {
    //     auto full_domain   = make_domain(full_shape);
    //     auto active_domain = [halo_radius, direction, full_domain]() {
    //         if constexpr (std::is_same_v<Tag, cell_centered_tag>) {
    //             (void) direction;   // suppress unused warning
    //             auto width = static_cast<std::int64_t>(halo_radius);
    //             return domain_algebra::contract(
    //                 full_domain,
    //                 ones<Dims, std::int64_t>() * width
    //             );
    //         }
    //         else {
    //             auto amount = ones<Dims, std::int64_t>();
    //             amount[direction] -= halo_radius;
    //             return domain_algebra::contract(full_domain, amount);
    //         }
    //     }();

    //     // allocate full domain (with halos)
    //     auto accessor = accessor_t<element_type, Dims>{full_domain};
    //     auto field    = compute_field_t{std::move(accessor), full_domain};
    //     for (auto coord : active_domain) {
    //         // get current value
    //         auto value = *gen;

    //         if (gen == py::iterator::sentinel()) {
    //             throw std::runtime_error("Generator exhausted");
    //         }
    //         if constexpr (std::is_same_v<element_type, real>) {
    //             field[coord] = py::cast<real>(value);
    //         }
    //         else {
    //             field[coord] = py_tuple_to_state<element_type>(value);
    //         }
    //         // advance iterator for next call
    //         ++gen;
    //     }

    //     // halos remain uninitialized (will be filled by boundary conditions)
    //     return field;
    // }

    template <typename T, std::uint64_t Dims>
    field_t<T, Dims> from_generator(
        py::iterator gen,
        const domain_t<Dims>& full_domain,
        const domain_t<Dims>& active_domain,
        mem::device_t dev = mem::device_t::cpu()
    )
    {
        field_t<T, Dims> field(full_domain, dev);

        // fill active region from generator
        for (auto coord : active_domain) {
            auto value = *gen;
            if (gen == py::iterator::sentinel()) {
                throw std::runtime_error("Generator exhausted");
            }

            if constexpr (std::is_same_v<T, real>) {
                field(coord) = py::cast<real>(value);
            }
            else {
                field(coord) = py_tuple_to_state<T>(value);
            }
            ++gen;
        }

        return field;
    }
}   // namespace simbi

#endif   // FIELD_HPP
