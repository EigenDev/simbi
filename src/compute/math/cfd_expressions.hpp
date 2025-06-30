#ifndef SIMBI_CFD_EXPRESSIONS_HPP
#define SIMBI_CFD_EXPRESSIONS_HPP

#include <cstdint>
#include <type_traits>
#include <utility>

namespace simbi {
    // forward declarations
    template <typename T, std::uint64_t Dims>
    struct field_t;

    template <std::uint64_t Dims>
    struct index_space_t;
}   // namespace simbi

namespace simbi::cfd {

    // more forward declarations
    template <typename Target>
    struct realize_to_t;

    template <typename Target>
    struct add_to_t;

    template <typename Other>
    struct add_t;

    template <typename Input, typename Op>
    struct unary_expr_t;

    template <typename Left, typename Right, typename Op>
    struct binary_expr_t;

    // operation types
    struct add_op_t {
        template <typename Left, typename Right>
        auto apply(const Left& left, const Right& right) const
        {
            return left + right;
        }
    };

    // expression template base - crtp pattern
    template <typename Derived>
    struct expression_t {
        const Derived& derived() const
        {
            return static_cast<const Derived&>(*this);
        }

        Derived& derived() { return static_cast<Derived&>(*this); }

        // execution interface - derived classes must implement
        template <typename Target>
        void realize_to(Target& target) const
        {
            derived().realize_to(target);
        }

        template <typename Target>
        void add_to(Target& target) const
        {
            // default implementation for add_to
            // derived classes can override this if needed
            derived().add_to(target);
        }

        // get the domain for this expression
        auto domain() const { return derived().domain(); }

        // specialized version for realize_to (terminal operation)
        template <typename Target>
        void operator|(realize_to_t<Target>&& op) &&
        {
            std::move(derived()).realize_to(op.target_);
        }

        // specialized version for add (binary operation)
        template <typename Other>
        auto operator|(add_t<Other>&& op) &&
        {
            return binary_expr_t<Derived, Other, add_op_t>{
              std::move(derived()),
              std::move(op.other_),
              add_op_t{}
            };
        }

        template <typename Target>
        void operator|(add_to_t<Target>&& op) &&
        {
            std::move(derived()).add_to(op.target_);
        }

        // generic operator| for all other operations (compute_fluxes_t,
        // scale_t, map_t, etc.)
        template <typename Op>
        auto operator|(Op&& op) &&
        {
            return unary_expr_t<Derived, std::decay_t<Op>>{
              std::move(derived()),
              std::forward<Op>(op)
            };
        }

        template <typename Coord>
        auto evaluate_at(Coord coord) const
        {
            return derived().evaluate_at(coord);
        }
    };

    // domain expression - starting point
    template <std::uint64_t Dims>
    struct domain_expr_t : expression_t<domain_expr_t<Dims>> {
        index_space_t<Dims> domain_;

        explicit domain_expr_t(index_space_t<Dims> dom) : domain_(dom) {}

        auto domain() const { return domain_; }

        template <typename Target>
        void realize_to(Target& /*target*/) const
        {
            // domain expression by itself does nothing
            // this would be used for coordinate-based generation
        }

        template <typename Coord>
        auto evaluate_at(Coord coord) const
        {
            // domain expression does not evaluate to a value
            // but can be used to generate coordinates
            return domain_.coord_to_linear_index(coord);
        }

        template <typename Target>
        void add_to(Target& target) const
        {
            // domain expression does not add anything to target
            // but can be used to initialize a target field
            // with the domain size
            target.resize(domain_.size());
        }
    };

    // field expression
    template <typename Field>
    struct field_expr_t : expression_t<field_expr_t<Field>> {
        const Field& field_;

        explicit field_expr_t(const Field& f) : field_(f) {}

        auto domain() const { return field_.domain(); }

        template <typename Target>
        void realize_to(Target& target) const
        {
            // copy field data to target
            for (std::uint64_t ii = 0; ii < field_.size(); ++ii) {
                target[ii] = field_[ii];
            }
        }

        template <typename Target>
        void add_to(Target& target) const
        {
            // add field data to target
            for (std::uint64_t ii = 0; ii < field_.size(); ++ii) {
                target[ii] = target[ii] + field_[ii];
            }
        }

        template <typename Coord>
        auto evaluate_at(Coord coord) const
        {
            auto idx = field_.domain().coord_to_linear_index(coord);
            return field_[idx];
        }
    };

    // unary expression - single input with operation
    template <typename Input, typename Op>
    struct unary_expr_t : expression_t<unary_expr_t<Input, Op>> {
        Input input_;
        Op operation_;

        unary_expr_t(Input input, Op op)
            : input_(std::move(input)), operation_(std::move(op))
        {
        }

        auto domain() const { return input_.domain(); }

        template <typename Target>
        void realize_to(Target& target) const
        {
            auto dom = domain();

            // element-wise operation (map, scale)
            for (std::uint64_t ii = 0; ii < dom.size(); ++ii) {
                auto coord = dom.index_to_coord(ii);
                target[ii] = operation_.apply(coord, input_);
            }
        }

        template <typename Target>
        void add_to(Target& target) const
        {
            // add unary expression result to target
            auto dom = domain();
            // coordinate-based operation
            for (std::uint64_t ii = 0; ii < dom.size(); ++ii) {
                auto coord = dom.index_to_coord(ii);
                target[ii] = target[ii] + operation_.apply(coord, input_);
            }
        }

        template <typename Coord>
        auto evaluate_at(Coord coord) const
        {
            return operation_.apply(coord, input_);
        }
    };

    // binary expression - two inputs with operation
    template <typename Left, typename Right, typename Op>
    struct binary_expr_t : expression_t<binary_expr_t<Left, Right, Op>> {
        Left left_;
        Right right_;
        Op operation_;

        binary_expr_t(Left left, Right right, Op op)
            : left_(std::move(left)),
              right_(std::move(right)),
              operation_(std::move(op))
        {
        }

        auto domain() const { return left_.domain(); }

        template <typename Target>
        void realize_to(Target& target) const
        {
            auto dom = domain();

            for (std::uint64_t ii = 0; ii < dom.size(); ++ii) {
                auto coord     = dom.index_to_coord(ii);
                auto left_val  = evaluate_at(left_, coord, ii);
                auto right_val = evaluate_at(right_, coord, ii);
                target[ii]     = operation_.apply(left_val, right_val);
            }
        }

        template <typename Target>
        void add_to(Target& target) const
        {
            auto dom = domain();

            for (std::uint64_t ii = 0; ii < dom.size(); ++ii) {
                auto coord     = dom.index_to_coord(ii);
                auto left_val  = evaluate_at(left_, coord, ii);
                auto right_val = evaluate_at(right_, coord, ii);
                target[ii] = target[ii] + operation_.apply(left_val, right_val);
            }
        }

        template <typename Coord>
        auto evaluate_at(Coord coord) const
        {
            auto right_val = [&]() {
                if constexpr (requires { right_.evaluate_at(coord); }) {
                    return right_.evaluate_at(coord);   // it's an expression
                }
                else {
                    return right_.apply(
                        coord,
                        left_.domain()
                    );   // it's an operation
                }
            }();
            auto left_val = [&]() {
                if constexpr (requires { left_.evaluate_at(coord); }) {
                    return left_.evaluate_at(coord);   // it's an expression
                }
                else {
                    return left_.apply(
                        coord,
                        right_.domain()
                    );   // it's an operation
                }
            }();
            return operation_.apply(left_val, right_val);
        }

      private:
        template <typename Expr>
        auto evaluate_at(
            const Expr& expr,
            auto coord,
            std::uint64_t linear_idx
        ) const
        {
            if constexpr (requires { expr.field_; }) {
                return expr.field_[linear_idx];   // field expression
            }
            else if constexpr (requires { expr.evaluate_at(coord); }) {
                return expr.evaluate_at(coord);   // complex expression
            }
            else {
                return expr.apply(coord, left_.domain());   // operation
            }
        }
    };

    // operation types
    template <typename Func>
    struct map_t {
        Func func_;
        static constexpr bool apply_elementwise_ = true;

        template <typename Coord, typename Input>
        auto apply(Coord coord, const Input& input) const
        {
            if constexpr (requires { input.field_; }) {
                auto idx = input.domain().coord_to_linear_index(coord);
                return func_(input.field_[idx]);
            }
            else {
                // complex input expression - evaluate it
                return func_(input.evaluate_at(coord));
            }
        }
    };

    template <typename Scalar>
    struct scale_t {
        Scalar factor_;
        static constexpr bool apply_elementwise_ = true;

        template <typename Coord, typename Input>
        auto apply(Coord coord, const Input& input) const
        {
            if constexpr (requires { input.field_; }) {
                auto idx = input.domain().coord_to_linear_index(coord);
                return input.field_[idx] * factor_;
            }
            else {
                // complex input expression - evaluate it
                return input.evaluate_at(coord) * factor_;
            }
        }
    };

    template <typename Other>
    struct add_t {
        Other other_;
    };

    template <typename Target>
    struct realize_to_t {
        Target& target_;
    };

    template <typename Target>
    struct add_to_t {
        Target& target_;
    };

    // add expression - combines two expressions

    // factory functions
    template <std::uint64_t Dims>
    auto make_domain_expr(index_space_t<Dims> domain)
    {
        return domain_expr_t<Dims>{domain};
    }

    template <typename Field>
    auto make_field_expr(const Field& field)
    {
        return field_expr_t<Field>{field};
    }

    template <typename Func>
    auto map(Func&& func)
    {
        return map_t<std::decay_t<Func>>{std::forward<Func>(func)};
    }

    template <typename Scalar>
    auto scale(Scalar factor)
    {
        return scale_t<Scalar>{factor};
    }

    template <typename Other>
    auto add(Other&& other)
    {
        return add_t<std::decay_t<Other>>{std::forward<Other>(other)};
    }

    template <typename Target>
    auto realize_to(Target& target)
    {
        return realize_to_t<Target>{target};
    }

    template <typename Target>
    auto add_to(Target& target)
    {
        return add_to_t<Target&>{target};
    }

    // coordinate-based generation
    template <typename Func>
    struct coordinate_map_t {
        Func func_;

        template <typename Coord, typename Input>
        auto apply(Coord coord, const Input& /*input*/) const
        {
            return func_(coord);
        }
    };

    template <typename Func>
    auto coordinate_map(Func&& func)
    {
        return coordinate_map_t<std::decay_t<Func>>{std::forward<Func>(func)};
    }

}   // namespace simbi::cfd

#endif   // SIMBI_CFD_EXPRESSIONS_HPP
