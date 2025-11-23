#ifndef GRID_FIELD_HPP
#define GRID_FIELD_HPP

#include "compat.hpp"
#include "compute/computation.hpp"
#include "containers/vector.hpp"
#include "functional/fp.hpp"
#include "grid/algebra.hpp"
#include "grid/domain.hpp"
#include "hesi/adapter.hpp"
#include "hesi/core/types.hpp"
#include "hesi/exec/for_each.hpp"
#include "hesi/exec/reduce.hpp"
#include "io/exceptions.hpp"
#include "traits/traits.hpp"
#include <functional>

#include <cstddef>
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace simbi::grid {
    using namespace compute;

    // forward declaration of the execution driver
    template <typename T, std::uint64_t Rank, typename Expression>
    void commit(
        het::executor_t& exec,
        const domain_t<Rank>& domain,
        het::view_t<T, Rank> dest,
        const Expression& expr
    );

    // -------------------------------------------------------------------------
    // field view
    // a lightweight, non-owning reference to a field region
    // this is the proxy object that enables u = expr.with(exec)
    // -------------------------------------------------------------------------
    template <typename T, std::uint64_t Rank>
    struct field_view_t : public het::view_t<T, Rank> {
        using value_type                    = T;
        using base_type                     = het::view_t<T, Rank>;
        static constexpr std::uint64_t rank = Rank;

        // the topological domain this view represents
        domain_t<Rank> domain_;

        field_view_t() = default;

        field_view_t(
            T* ptr,
            iarray<Rank> shape,
            iarray<Rank> strides,
            domain_t<Rank> dom
        )
            : base_type(ptr, shape, dom.start, strides), domain_(dom)
        {
        }

        const T& operator()(const iarray<Rank>& coord) const
        {
            return base_type::operator()(coord);
        }
        T& operator[](const iarray<Rank>& coord)
        {
            return base_type::operator[](coord);
        }

        auto as_computation() const { return computation(*this); }

        const domain_t<Rank>& domain() const { return domain_; }

        // commitment operator (the execution trigger)
        // allows syntax: view = expr.with(exec)
        template <typename Comp, typename Exec>
        const field_view_t&
        operator=(const bound_computation_t<Rank, Comp, Exec>& package) const
        {
            using namespace domain_algebra;
            // strict domain intersection check
            // we verify that the computation defines values for the entire
            // target view
            auto intersect = intersection(domain_, package.comp.domain());

            if (intersect != domain_) {
                // the domain_t overload the stream operator, so we can use that
                // for error reporting. The to_string method will not work here.
                std::ostringstream oss;
                oss << "Domain mismatch in field commitment.\n"
                    << "  Target domain: " << domain_ << "\n"
                    << "  Computation domain: " << package.comp.domain() << "\n"
                    << "  Intersection: " << intersect << "\n";
                throw std::runtime_error(oss.str());
            }

            // trigger the kernel execution
            commit(package.exec, domain_, *this, package.comp);

            return *this;
        }

        // views also support mapping and zipping directly
        // map: view.map(f) -> computation
        template <typename UnaryOp>
        auto map(UnaryOp&& op) const
        {
            return computation(*this).map(std::forward<UnaryOp>(op));
        }

        // zip: view.zip(other, op) -> computation
        // zip: lift both, then zip
        template <typename Other, typename BinaryOp>
        auto zip(const Other& other, BinaryOp&& op) const
        {
            // helper to normalize inputs to computations
            auto ensure_comp = [](const auto& obj) {
                if constexpr (is_computation_v<decltype(obj)>) {
                    return obj;
                }
                else if constexpr (std::
                                       is_same_v<decltype(obj), field_view_t>) {
                    return computation(obj);
                }
                else {
                    // assume view-like or already liftable
                    return computation(obj);
                }
            };

            return ensure_comp(*this).zip(
                ensure_comp(other),
                std::forward<BinaryOp>(op)
            );
        }

        // coord map
        template <typename F>
        auto space_map(F&& op) const
        {
            return computation(*this).space_map(std::forward<F>(op));
        }

        // enum map
        template <typename G>
        auto enum_map(G&& op) const
        {
            return computation(*this).enum_map(std::forward<G>(op));
        }

        // remap : view.remap(f) -> computation
        template <typename UnaryOp>
        auto remap(UnaryOp&& op) const
        {
            return computation(*this).remap(std::forward<UnaryOp>(op));
        }
    };

    // -------------------------------------------------------------------------
    // field_t
    // the shared owner of physics data.
    // semantics:
    //   - copy: shallow copy (increments ref count)
    //   - slice: shallow copy with new logical domain
    //   - commit: pointer update
    // -------------------------------------------------------------------------
    template <typename T, std::uint64_t Rank>
    class field_t
    {
      public:
        using value_type                    = T;
        using view_type                     = field_view_t<T, Rank>;
        using coord_type                    = typename domain_t<Rank>::coord_t;
        static constexpr std::uint64_t rank = Rank;

      private:
        // shared ownership of gpu/cpu memory
        het::shared_handle_t<het::block_t> storage_;

        // the "logical" domain this field represents (active region)
        domain_t<Rank> domain_;

      public:
        // ---------------------------------------------------------------------
        // construction
        // ---------------------------------------------------------------------

        // default
        field_t() = default;
        // allocates memory for domain + ghost_width
        field_t(
            const domain_t<Rank>& domain,
            het::locality_t loc = het::locality_t::host()
        )
            : domain_(domain)
        {
            auto shape                = domain.shape();
            std::uint64_t total_elems = fp::product(shape);

            // allocate via shared handle factory
            storage_ = het::shared_handle_t<het::block_t>::make(
                total_elems * sizeof(T),
                loc,
                het::memory_type_t::host_visible
            );
        }

        // copy constructor (shallow)
        // creates a new reference to the same memory
        field_t(const field_t& other) = default;

        // copy commitment (shallow)
        // reseats this field to point to other's memory
        field_t& operator=(const field_t& other) = default;

        // move semantics
        field_t(field_t&&)            = default;
        field_t& operator=(field_t&&) = default;

        auto as_computation() const { return computation(view()); }

        // ---------------------------------------------------------------------
        // accessors (views)
        // ---------------------------------------------------------------------

        // returns a view of the interior (logical) domain
        view_type view() const { return make_view(domain_); }

        // returns a view of a specific subdomain
        // clips request to the physical allocation bounds
        view_type operator[](const domain_t<Rank>& request) const
        {
            auto valid = domain_.intersect(request);
            return make_view(valid);
        }

        // ---------------------------------------------------------------------
        // slicing (sub-fields)
        // ---------------------------------------------------------------------

        // returns a new field object sharing the same memory
        // but restricting the logical domain to 'subdomain'
        field_t slice(const domain_t<Rank>& subdomain) const
        {
            field_t sub = *this;   // increments ref count

            // restrict the logical window
            // note: we intersect with allocated_domain_ to ensure safety
            sub.domain_ = domain_.intersect(subdomain);

            return sub;
        }

        // ---------------------------------------------------------------------
        // combinators & commitment
        // ---------------------------------------------------------------------

        // map: u.map(f) -> computation
        template <typename UnaryOp>
        auto map(UnaryOp&& op) const
        {
            // lift view -> computation (identity)
            // apply map -> computation (transformed)
            return computation(view()).map(std::forward<UnaryOp>(op));
        }

        // zip: lift both, then zip
        // zip: u.zup(v, op) -> computation
        template <typename Other, typename BinaryOp>
        auto zip(const Other& other, BinaryOp&& op) const
        {

            // helper to normalize inputs to computations
            auto ensure_comp = [](const auto& obj) {
                if constexpr (is_computation_v<decltype(obj)>) {
                    return obj;
                }
                else if constexpr (std::is_same_v<decltype(obj), field_t>) {
                    return computation(obj.view());
                }
                else {
                    // assume view-like or already liftable
                    return computation(obj);
                }
            };

            return ensure_comp(*this).zip(
                ensure_comp(other),
                std::forward<BinaryOp>(op)
            );
        }

        template <typename F>
        auto space_map(F&& op) const
        {
            return computation(view()).space_map(std::forward<F>(op));
        }

        template <typename G>
        auto enum_map(G&& op) const
        {
            return computation(view()).enum_map(std::forward<G>(op));
        }

        template <typename UnaryOp>
        auto remap(UnaryOp&& op) const
        {
            return computation(view()).remap(std::forward<UnaryOp>(op));
        }

        // call operator
        // allows syntax: view(coord) to access elements
        // T& operator()(const iarray<Rank>& coord) { return view()(coord); }
        const T& operator()(const iarray<Rank>& coord) const
        {
            return view()(coord);
        }
        T& operator[](const iarray<Rank>& coord) { return view()[coord]; }

        // execution commitment shortcut
        // allows syntax: field = expr.with(exec)
        // delegates to the view's commitment operator
        template <typename Comp, typename Exec>
        field_t& operator=(const bound_computation_t<Rank, Comp, Exec>& package)
        {
            view() = package;
            return *this;
        }

        // metadata
        const domain_t<Rank>& domain() const { return domain_; }
        het::locality_t locality() const { return storage_->locality(); }

        // direct pointer access (use with caution)
        T* data() const { return static_cast<T*>(storage_->data()); }

      private:
        // constructs a view for the target domain pointing into storage
        view_type make_view(const domain_t<Rank>& target) const
        {
            // calculate offset from the absolute start of allocation
            iarray<Rank> offset_vec = target.start - domain_.start;

            iarray<Rank> alloc_shape = domain_.shape();
            iarray<Rank> strides;

            std::uint64_t linear_offset = 0;
            std::uint64_t stride_accum  = 1;

            // row-major stride calculation
            // strides are fixed based on the allocation shape, not the view
            // shape
            for (std::int64_t ii = Rank - 1; ii >= 0; --ii) {
                strides[ii] = stride_accum;
                linear_offset += offset_vec[ii] * stride_accum;
                stride_accum *= alloc_shape[ii];
            }

            T* ptr = static_cast<T*>(storage_->data()) + linear_offset;
            return view_type(ptr, target.shape(), strides, target);
        }
    };

    // -------------------------------------------------------------------------
    // commitment driver
    // -------------------------------------------------------------------------
    template <typename T, std::uint64_t Rank, typename Expression>
    void try_commit(
        het::executor_t& exec,
        const domain_t<Rank>& domain,
        het::view_t<T, Rank> dest,
        const Expression& expr
    )
    {
        auto nerrors = het::exec::reduce_sync(
            exec,
            domain,
            std::size_t{0},
            [=] DUAL(const iarray<Rank>& coord) -> std::size_t {
                auto maybe_value = expr(coord);
                if (maybe_value.has_value()) {
                    dest(coord) = maybe_value.value();
                    return std::size_t{0};
                }
                else {
                    return std::size_t{1};
                }
            },
            std::plus<std::size_t>{}
        );

        if (nerrors > 0) {
            throw exception::SimulationFailureException();
        };
    }

    template <typename T, std::uint64_t Rank, typename Expression>
    void direct_commit(
        het::executor_t& exec,
        const domain_t<Rank>& domain,
        het::view_t<T, Rank> dest,
        const Expression& expr
    )
    {
        // launch kernel using generic parallel_for
        // the compiler fuses the expression tree into the lambda
        het::exec::parallel_for(
            het::exec::default_t{},
            exec,
            domain,
            [=] DUAL(const iarray<Rank>& coord) -> void {
                dest(coord) = expr(coord);
            }
        );
    }

    template <typename T, std::uint64_t Rank, typename Expression>
    void commit(
        het::executor_t& exec,
        const domain_t<Rank>& domain,
        het::view_t<T, Rank> dest,
        const Expression& expr
    )
    {
        // check the expression's return type, not the destination field type
        using expr_result_t = std::invoke_result_t<Expression, iarray<Rank>>;

        if constexpr (is_maybe_v<expr_result_t>) {
            try_commit(exec, domain, dest, expr);
        }
        else {
            direct_commit(exec, domain, dest, expr);
        }
    }

}   // namespace simbi::grid

#endif   // GRID_FIELD_HPP
