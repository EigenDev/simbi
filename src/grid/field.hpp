#ifndef GRID_FIELD_HPP
#define GRID_FIELD_HPP

#include "compat.hpp"
#include "compute/computation.hpp"
#include "containers/vector.hpp"
#include "functional/fp.hpp"
#include "grid/algebra.hpp"
#include "grid/domain.hpp"
#include "io/exceptions.hpp"
#include "traits/traits.hpp"
#include "xpu/execution/execution_space.hpp"
#include "xpu/xpu.hpp"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace simbi::grid {
    using namespace compute;

    // forward declaration of the execution driver
    template <
        typename T,
        std::uint64_t Rank,
        typename Expression,
        xpu::execution_space ExecutionSpace>
    void commit(
        xpu::executor_t<ExecutionSpace>& exec,
        const domain_t<Rank>&            domain,
        xpu::view_t<T, Rank>             dest,
        const Expression&                expr
    );

    // -------------------------------------------------------------------------
    // field view
    // a lightweight, non-owning reference to a field region
    // this is the proxy object that enables u = expr.with(exec)
    // -------------------------------------------------------------------------
    template <typename T, std::uint64_t Rank>
    struct field_view_t : public xpu::view_t<T, Rank>
    {
        using value_type                    = T;
        using reference_type                = T&;
        using const_reference_type          = const T&;
        using argument_type                 = iarray<Rank>;
        using base_type                     = xpu::view_t<T, Rank>;
        static constexpr std::uint64_t rank = Rank;

        // the topological domain this view represents
        domain_t<Rank> domain_;

        field_view_t() = default;

        field_view_t(T* ptr, iarray<Rank> shape, iarray<Rank> strides, domain_t<Rank> dom)
            : base_type(ptr, shape, dom.start, strides), domain_(dom)
        {
        }

        DUAL const T& operator()(const iarray<Rank>& coord) const
        {
            return base_type::operator()(coord);
        }
        DUAL T& operator[](const iarray<Rank>& coord)
        {
            return base_type::operator[](coord);
        }

        auto as_computation() const
        {
            return computation(*this);
        }

        const domain_t<Rank>& domain() const
        {
            return domain_;
        }

        // commitment operator (the execution trigger)
        // allows syntax: view = expr.with(exec)
        template <typename Comp, typename Exec>
        const field_view_t& operator=(const bound_computation_t<Rank, Comp, Exec>& package) const
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
                using obj_t = std::decay_t<decltype(obj)>;

                if constexpr (is_computation_v<obj_t>) {
                    return obj;
                }
                else if constexpr (requires { obj.as_computation(); }) {
                    // has .as_computation() method - use it
                    return obj.as_computation();
                }
                else {
                    // assume view-like - call computation() to lift it
                    return computation(obj);
                }
            };

            return ensure_comp(*this).zip(ensure_comp(other), std::forward<BinaryOp>(op));
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
        using argument_type                 = iarray<Rank>;
        using view_type                     = field_view_t<T, Rank>;
        using coord_type                    = typename domain_t<Rank>::coord_t;
        static constexpr std::uint64_t rank = Rank;

      private:
        // shared ownership using new clean memory system
        xpu::shared_handle_t<xpu::sim_block_t> storage_;

        // the "logical" domain this field represents (active region)
        domain_t<Rank> domain_;

      public:
        // ---------------------------------------------------------------------
        // construction
        // ---------------------------------------------------------------------

        // default
        field_t() = default;
        // allocates memory for domain using configured memory space
        explicit field_t(const domain_t<Rank>& domain) : domain_(domain)
        {
            auto          shape       = domain.shape();
            std::uint64_t total_elems = 1;
            for (std::uint64_t ii = 0; ii < Rank; ++ii) {
                total_elems *= shape[ii];
            }

            // allocate using new clean memory system
            auto block = xpu::make_memory_block<T>(total_elems);
            storage_   = xpu::make_shared_handle<xpu::sim_block_t>(std::move(block));
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

        auto as_computation() const
        {
            return computation(view());
        }

        // ---------------------------------------------------------------------
        // accessors (views)
        // ---------------------------------------------------------------------

        // returns a view of the interior (logical) domain
        view_type view() const
        {
            return make_view(domain_);
        }

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
            field_t sub = *this; // increments ref count

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
                using obj_t = std::decay_t<decltype(obj)>;

                if constexpr (is_computation_v<obj_t>) {
                    return obj;
                }
                else if constexpr (requires { obj.as_computation(); }) {
                    // has .as_computation() method - use it
                    return obj.as_computation();
                }
                else {
                    // assume view-like - call computation() to lift it
                    return computation(obj);
                }
            };

            return ensure_comp(*this).zip(ensure_comp(other), std::forward<BinaryOp>(op));
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
        T& operator[](const iarray<Rank>& coord)
        {
            return view()[coord];
        }

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
        const domain_t<Rank>& domain() const
        {
            return domain_;
        }
        // unified memory is accessible from all devices
        constexpr std::int64_t device_id() const
        {
            return 0; // unified memory, no specific device
        }

        // direct pointer access (use with caution)
        T* data() const
        {
            return storage_->template as<T>();
        }

        // ---------------------------------------------------------------------
        // cloning helpers
        // - clone(target) performs a synchronous clone/move into the requested
        //   locality and returns a field_t usable on that locality.
        // - clone_async(target, stream, out) prepares `out` to receive the
        //   cloned storage and returns a token that completes when the copy
        //   finishes. the caller may schedule kernels on the same stream and
        //   rely on stream ordering.
        // ---------------------------------------------------------------------

        // synchronous clone (convenience)
        // deep copy: allocates new storage and copies data
        field_t clone() const
        {
            field_t out;
            out.domain_ = domain_;

            auto          shape       = domain_.shape();
            std::uint64_t total_elems = 1;
            for (std::uint64_t ii = 0; ii < Rank; ++ii) {
                total_elems *= shape[ii];
            }

            auto block   = xpu::make_memory_block<T>(total_elems);
            out.storage_ = xpu::make_shared_handle<xpu::sim_block_t>(std::move(block));

            // synchronous copy using new memory system
            std::copy(
                storage_->template as<T>(),
                storage_->template as<T>() + total_elems,
                out.storage_->template as<T>()
            );
            xpu::mark_host_dirty_if_needed(out.storage_);

            return out;
        }

        // async clone with executor
        template <xpu::execution_space ExecutionSpace>
        xpu::token_t<ExecutionSpace>
        clone_async(xpu::executor_t<ExecutionSpace>& exec, field_t& out) const
        {
            out.domain_ = domain_;

            auto          shape       = domain_.shape();
            std::uint64_t total_elems = 1;
            for (std::uint64_t ii = 0; ii < Rank; ++ii) {
                total_elems *= shape[ii];
            }

            auto block   = xpu::make_memory_block<T>(total_elems);
            out.storage_ = xpu::make_shared_handle<xpu::sim_block_t>(std::move(block));

            // create device-compatible copy functor using views
            struct copy_t
            {

                using argument_type = std::array<std::int64_t, 1>;
                enum {
                    rank = 1
                };

                xpu::view_t<T, 1> src_view;
                xpu::view_t<T, 1> dst_view;

                DEV void operator()(const argument_type& idx) const
                {
                    dst_view[idx[0]] = src_view[idx[0]];
                }
            };

            // create views and dispatch copy kernel
            auto          copy_shape = domain_.shape();
            std::uint64_t copy_elems = 1;
            for (std::uint64_t ii = 0; ii < Rank; ++ii) {
                copy_elems *= copy_shape[ii];
            }

            auto src_view  = xpu::view_t<T, 1>(storage_->template as<T>(), {copy_elems});
            auto dst_view  = xpu::view_t<T, 1>(out.storage_->template as<T>(), {copy_elems});
            auto domain_1d = grid::extents<1>({static_cast<std::int64_t>(copy_elems)});
            return exec.dispatch(domain_1d, copy_t{src_view, dst_view});
        }

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

            xpu::mark_host_dirty_if_needed(storage_); // mark as modified from host
            T* ptr = storage_->template as<T>() + linear_offset;
            return view_type(ptr, target.shape(), strides, target);
        }
    };

    // -------------------------------------------------------------------------
    // commitment driver
    // -------------------------------------------------------------------------
    template <
        typename T,
        std::uint64_t Rank,
        typename Expression,
        xpu::execution_space ExecutionSpace>
    void try_commit(
        xpu::executor_t<ExecutionSpace>& exec,
        const domain_t<Rank>&            domain,
        xpu::view_t<T, Rank>             dest,
        const Expression&                expr
    )
    {
        auto nerrors = exec.reduce(
            domain,
            std::size_t{0},
            [=] DEV(const typename grid::domain_t<Rank>::coord_t& coord) -> std::size_t {
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

    // function object for assignment operation
    template <typename T, std::uint64_t Rank, typename Expression>
    struct assign_t
    {
        using value_type                    = void;
        using argument_type                 = typename grid::domain_t<Rank>::coord_t;
        static constexpr std::uint64_t rank = Rank;

        xpu::view_t<T, Rank> dest;
        Expression           expr;

        constexpr assign_t(xpu::view_t<T, Rank> d, const Expression& e) : dest(d), expr(e) {}

        DEV void operator()(const argument_type& coord) const
        {
            dest(coord) = expr(coord);
        }
    };

    template <
        typename T,
        std::uint64_t Rank,
        typename Expression,
        xpu::execution_space ExecutionSpace>
    void direct_commit(
        xpu::executor_t<ExecutionSpace>& exec,
        const domain_t<Rank>&            domain,
        xpu::view_t<T, Rank>             dest,
        const Expression&                expr
    )
    {
        // dispatch kernel over domain
        auto functor = assign_t<T, Rank, Expression>{dest, expr};
        exec.dispatch(domain, functor);
    }

    template <
        typename T,
        std::uint64_t Rank,
        typename Expression,
        xpu::execution_space ExecutionSpace>
    void commit(
        xpu::executor_t<ExecutionSpace>& exec,
        const domain_t<Rank>&            domain,
        xpu::view_t<T, Rank>             dest,
        const Expression&                expr
    )
    {
        // check the expression's return type
        using expr_result_t = std::invoke_result_t<Expression, iarray<Rank>>;

        if constexpr (is_maybe_v<expr_result_t>) {
            try_commit(exec, domain, dest, expr);
        }
        else {
            direct_commit(exec, domain, dest, expr);
        }
    }

} // namespace simbi::grid

#endif // GRID_FIELD_HPP
