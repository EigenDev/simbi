#ifndef FIELD_HPP
#define FIELD_HPP

#include "config.hpp"
#include "containers/vector.hpp"
#include "domain/algebra.hpp"
#include "domain/domain.hpp"
#include "execution/executor.hpp"
#include "functional/fp.hpp"
#include "memory/accessor.hpp"

#include <cstdint>
#include <type_traits>

namespace simbi {
    using namespace mem;

    namespace detail {
        // type trait to check if Computation wraps accessor_t
        template <typename Comp>
        struct is_accessor_computation : std::false_type {
        };

        template <typename T, std::uint64_t Dims>
        struct is_accessor_computation<accessor_t<T, Dims>> : std::true_type {
        };

        template <typename Comp>
        inline constexpr bool is_accessor_v =
            is_accessor_computation<Comp>::value;

        // helper to detect if computation returns a reference
        template <typename Comp, std::uint64_t Dims>
        struct returns_reference {
            template <typename C>
            static auto test(int)
                -> std::is_reference<decltype(std::declval<C>()(
                    std::declval<coordinate_t<Dims>>()
                ))>;

            template <typename>
            static std::false_type test(...);

            static constexpr bool value = decltype(test<Comp>(0))::value;
        };

        template <typename Comp, std::uint64_t Dims>
        inline constexpr bool returns_reference_v =
            returns_reference<Comp, Dims>::value;

        // get value type from computation
        template <typename Comp, std::uint64_t Dims>
        using computation_value_t =
            std::invoke_result_t<Comp, coordinate_t<Dims>>;
    }   // namespace detail

    // forward declarations
    template <std::uint64_t Dims, typename Computation>
    struct compute_field_t;

    // deduction guides
    template <std::uint64_t Dims, typename Computation>
    compute_field_t(Computation&&, domain_t<Dims>)
        -> compute_field_t<Dims, std::decay_t<Computation>>;

    template <typename T, std::uint64_t Dims>
    compute_field_t(accessor_t<T, Dims>, domain_t<Dims>)
        -> compute_field_t<Dims, accessor_t<T, Dims>>;

    template <typename T, std::uint64_t Dims>
    using field_t = compute_field_t<Dims, accessor_t<T, Dims>>;

    // unified field abstraction using mathematical function composition
    template <std::uint64_t Dims, typename Computation>
    struct compute_field_t {
        using value_type = detail::computation_value_t<Computation, Dims>;
        static constexpr std::uint64_t dimensions = Dims;

        Computation computation;
        domain_t<Dims> domain_;

        // basic queries
        constexpr auto domain() const { return domain_; }

        auto data() const
            requires detail::is_accessor_v<Computation>
        {
            return computation.data();
        }

        // assignment materialization
        template <typename OtherComputation>
        auto operator=(const compute_field_t<Dims, OtherComputation>& source)
        {
            if constexpr (detail::is_accessor_v<Computation>) {
                if (domain_.empty()) {
                    domain_     = source.domain_;
                    using cvt   = std::remove_cvref_t<value_type>;
                    computation = accessor_t<cvt, Dims>{domain_};
                }
                computation.commit(source, exec::default_executor());
            }
            else if constexpr (detail::returns_reference_v<Computation, Dims>) {
                exec::default_executor()
                    .for_each(
                        domain_,
                        [this, source](auto coord) {
                            computation(coord) = source(coord);
                        }
                    )
                    .wait();
            }
            else {
                computation = source.computation;
                domain_     = source.domain_;
            }
            return *this;
        }

        auto clone() const
            requires detail::is_accessor_v<Computation>
        {
            auto new_accessor = computation.clone();
            return compute_field_t<Dims, decltype(new_accessor)>{
              std::move(new_accessor),
              domain_
            };
        }

        // function evaluation
        constexpr DEV decltype(auto)
        operator()(const coordinate_t<Dims>& coord) const
        {
            return computation(coord);
        }

        constexpr DEV decltype(auto) operator()(const coordinate_t<Dims>& coord)
        {
            return computation(coord);
        }

        constexpr DEV decltype(auto) operator[](coordinate_t<Dims> coord) const
        {
            return computation(coord);
        }

        constexpr DEV decltype(auto) operator[](coordinate_t<Dims> coord)
        {
            return computation(coord);
        }

        constexpr decltype(auto) operator[](std::int64_t idx) const
        {
            return computation(domain_.linear_to_coord(idx));
        }

        // slicing using mathematical transformation
        constexpr auto operator[](domain_t<Dims> subdomain) const
        {
            auto sliced_computation = fp::transform(
                computation,
                fp::partial(fp::add_op, subdomain.start)
            );
            auto local_domain = make_domain(subdomain.shape());

            return compute_field_t<Dims, decltype(sliced_computation)>{
              std::move(sliced_computation),
              local_domain
            };
        }

        // map using function composition
        template <typename UnaryOp>
        auto map(UnaryOp op) const
        {
            auto mapped_computation = fp::compose(op, computation);
            return compute_field_t<Dims, decltype(mapped_computation)>{
              std::move(mapped_computation),
              domain_
            };
        }

        template <typename BinaryOp>
        auto enum_map(BinaryOp op) const
        {
            auto coord_func = fp::identity;
            auto value_func = computation;
            auto enum_func  = fp::zip(coord_func, value_func, op);
            return compute_field_t<Dims, decltype(enum_func)>{
              std::move(enum_func),
              domain_
            };
        }

        template <typename UnaryOp>
        auto coord_map(UnaryOp op) const
        {
            auto coord_func  = fp::identity;
            auto mapped_func = fp::compose(op, coord_func);
            return compute_field_t<Dims, decltype(mapped_func)>{
              std::move(mapped_func),
              domain_
            };
        }

        // zip using binary combination
        template <typename OtherComputation, typename BinaryOp>
        auto
        zip(const compute_field_t<Dims, OtherComputation>& other,
            BinaryOp op) const
        {
            using namespace domain_algebra;
            auto combined_domain = intersection(domain_, other.domain_);
            auto zipped_computation =
                fp::zip(computation, other.computation, op);

            return compute_field_t<Dims, decltype(zipped_computation)>{
              std::move(zipped_computation),
              combined_domain
            };
        }

        // at: restrict to subdomain
        auto at(const domain_t<Dims>& subdomain) const
        {
            using namespace domain_algebra;
            auto restricted_domain = intersection(domain_, subdomain);
            return compute_field_t{computation, restricted_domain};
        }

        // insert using conditional selection
        template <typename OtherComputation>
        auto
        insert(const compute_field_t<Dims, OtherComputation>& overlay) const
        {
            using namespace domain_algebra;
            auto union_domain      = union_of(domain_, overlay.domain_);
            auto overlay_predicate = fp::contains_op(overlay.domain_);
            auto insert_computation =
                fp::select(overlay_predicate, overlay.computation, computation);

            return compute_field_t<Dims, decltype(insert_computation)>{
              std::move(insert_computation),
              union_domain
            };
        }

        template <typename Executor = exec::default_executor_t>
        auto commit(const Executor& executor = exec::default_executor_t{}) const
        {
            if constexpr (detail::is_accessor_v<Computation>) {
                return *this;
            }
            else {
                auto acc = accessor_t<value_type, Dims>{domain_};
                acc.commit(*this, executor);
                return compute_field_t<Dims, accessor_t<value_type, Dims>>{
                  std::move(acc),
                  domain_
                };
            }
        }
    };

    template <std::uint64_t Dims, typename CompA, typename CompB>
    auto operator+(
        const compute_field_t<Dims, CompA>& a,
        const compute_field_t<Dims, CompB>& b
    )
    {
        return a.zip(b, fp::add_op);
    }

    template <std::uint64_t Dims, typename CompA, typename CompB>
    auto operator-(
        const compute_field_t<Dims, CompA>& a,
        const compute_field_t<Dims, CompB>& b
    )
    {
        return a.zip(b, fp::subtract_op);
    }

    template <std::uint64_t Dims, typename CompA, typename CompB>
    auto operator*(
        const compute_field_t<Dims, CompA>& a,
        const compute_field_t<Dims, CompB>& b
    )
    {
        return a.zip(b, fp::multiply_op);
    }

    template <std::uint64_t Dims, typename CompA, typename CompB>
    auto operator/(
        const compute_field_t<Dims, CompA>& a,
        const compute_field_t<Dims, CompB>& b
    )
    {
        return a.zip(b, fp::divide_op);
    }

    template <std::uint64_t Dims, typename Computation, typename Scalar>
    auto
    operator*(const compute_field_t<Dims, Computation>& field, Scalar scalar)
    {
        auto scalar_computation = fp::constant(scalar);
        auto result_computation =
            fp::zip(field.computation, scalar_computation, fp::multiply_op);

        return compute_field_t<Dims, decltype(result_computation)>{
          std::move(result_computation),
          field.domain_
        };
    }

    template <std::uint64_t Dims, typename Computation, typename Scalar>
    auto
    operator*(Scalar scalar, const compute_field_t<Dims, Computation>& field)
    {
        return field * scalar;
    }

    template <std::uint64_t Dims, typename Computation, typename Scalar>
    auto
    operator/(const compute_field_t<Dims, Computation>& field, Scalar scalar)
    {
        auto scalar_computation = fp::constant(scalar);
        auto result_computation =
            fp::zip(field.computation, scalar_computation, fp::divide_op);

        return compute_field_t<Dims, decltype(result_computation)>{
          std::move(result_computation),
          field.domain_
        };
    }

    template <std::uint64_t Dims, typename Computation, typename Scalar>
    auto
    operator+(const compute_field_t<Dims, Computation>& field, Scalar scalar)
    {
        auto scalar_computation = fp::constant(scalar);
        auto result_computation =
            fp::zip(field.computation, scalar_computation, fp::add_op);

        return compute_field_t<Dims, decltype(result_computation)>{
          std::move(result_computation),
          field.domain_
        };
    }

    template <std::uint64_t Dims, typename Computation, typename Scalar>
    auto
    operator+(Scalar scalar, const compute_field_t<Dims, Computation>& field)
    {
        return field + scalar;
    }

    template <std::uint64_t Dims, typename Computation, typename Scalar>
    auto
    operator-(const compute_field_t<Dims, Computation>& field, Scalar scalar)
    {
        auto scalar_computation = fp::constant(scalar);
        auto result_computation =
            fp::zip(field.computation, scalar_computation, fp::subtract_op);

        return compute_field_t<Dims, decltype(result_computation)>{
          std::move(result_computation),
          field.domain_
        };
    }

    template <std::uint64_t Dims, typename Computation, typename Scalar>
    auto
    operator-(Scalar scalar, const compute_field_t<Dims, Computation>& field)
    {
        auto scalar_computation = fp::constant(scalar);
        auto result_computation =
            fp::zip(scalar_computation, field.computation, fp::subtract_op);

        return compute_field_t<Dims, decltype(result_computation)>{
          std::move(result_computation),
          field.domain_
        };
    }

    template <std::uint64_t Dims, typename F>
    auto field(const domain_t<Dims>& domain, F&& fn)
    {
        return compute_field_t{std::forward<F>(fn), domain};
    }

    template <typename T, std::uint64_t Dims>
    auto from_data_field(T* data, const iarray<Dims>& shape)
    {
        auto accessor = from_data(data, shape);
        auto domain   = make_domain(shape);
        return compute_field_t{std::move(accessor), domain};
    }

    template <std::uint64_t Dims>
    auto identity(const domain_t<Dims>& domain)
    {
        return compute_field_t{fp::identity, domain};
    }

}   // namespace simbi

#endif   // FIELD_HPP
