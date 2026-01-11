#ifndef SIMBI_TEST_HELPERS_HPP
#define SIMBI_TEST_HELPERS_HPP

// lightweight test helpers to adapt lambdas into computable functors
// these allow legacy tests to continue using lambda-style computations
// while satisfying the library's `computable` concept.
//
// usage:
//   auto comp = test_helpers::make_computation<2>(domain, [](iarray<2> c){ ...
//   }); field = comp.with(exec);
// =============================================================================

#include "build_config.hpp"
#include "compute/computation.hpp"
#include "grid/domain.hpp"

#include <type_traits>
#include <utility>

namespace test_helpers {

    // wrapper converting a lambda/functor into a type satisfying the
    // simbi::concepts::computable requirements used by computation_t.
    template <std::uint64_t Rank, typename Lambda>
    struct lambda_computable_t {
        using argument_type = simbi::iarray<Rank>;
        using value_type    = std::remove_cv_t<
               std::invoke_result_t<std::decay_t<Lambda>, argument_type>>;
        static constexpr std::uint64_t rank = Rank;

        std::decay_t<Lambda> f;

        lambda_computable_t() = default;
        explicit lambda_computable_t(Lambda&& fn) : f(std::forward<Lambda>(fn))
        {
        }
        explicit lambda_computable_t(const Lambda& fn) : f(fn) {}

        // evaluator used by computation_t
        DUAL value_type operator()(argument_type coord) const
        {
            return f(coord);
        }
    };

    // factory: build a computation_t<Rank, wrapper> from a domain + lambda
    template <std::uint64_t Rank, typename Lambda>
    auto make_computation(const simbi::grid::domain_t<Rank>& dom, Lambda&& lam)
    {
        using wrapper_t = lambda_computable_t<Rank, std::decay_t<Lambda>>;
        wrapper_t w{std::forward<Lambda>(lam)};
        return simbi::compute::computation_t<Rank, wrapper_t>(
            std::move(w),
            dom
        );
    }

}   // namespace test_helpers

#endif   // SIMBI_TEST_HELPERS_HPP
