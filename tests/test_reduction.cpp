#include "compat.hpp"
#include "compute/computation.hpp"
#include "containers/vector.hpp"
#include "grid/domain.hpp"
#include "grid/field.hpp"
#include "hesi/adapter.hpp"
#include "hesi/core/types.hpp"
#include "hesi/exec/reduce.hpp"
#include "hesi/mem/transfer.hpp"
#include "test_helpers.hpp"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>

using namespace simbi;
using namespace simbi::grid;

struct cfl_calculator_t {
    double gamma;
    double dx;

    DUAL double operator()(double rho, double press, double vel) const
    {
        double cs =
            (rho > 0.0 && press > 0.0) ? std::sqrt(gamma * press / rho) : 0.0;
        double signal = std::abs(vel) + cs;
        return (signal > 1e-12) ? (dx / signal) : 1e12;
    }
};

struct min_op_t {
    DUAL double operator()(double a, double b) const { return (a < b) ? a : b; }
};

int main()
{
    std::cout << "testing CFL reduction..." << std::endl;

    constexpr std::uint64_t N = 100;
    grid::domain_t<2> domain{{0, 0}, {N, N}};

    double gamma = 1.4;
    double dx    = 0.1;

    auto backend = het::info::is_gpu ? het::backend_type_t::cuda
                                     : het::backend_type_t::cpu;
    het::locality_t loc{backend, 0};
    het::stream_t stream(backend);
    het::executor_t exec(stream);

    auto init = [=] DUAL(auto coord) {
        double x  = coord[0] - (double) N / 2.0;
        double y  = coord[1] - (double) N / 2.0;
        double r2 = x * x + y * y;

        return (r2 < (N / 4.0) * (N / 4.0))
                   ? vector_t<double, 3>{1.0, 10.0, 5.0}
                   : vector_t<double, 3>{1.0, 0.1, 0.0};
    };

    auto physics = test_helpers::make_computation<2>(domain, init);
    auto dt_comp = physics.map([=](auto val) {
        return cfl_calculator_t{gamma, dx}(val[0], val[1], val[2]);
    });

    field_t<double, 1> result(extents(iarray<1>{1}), loc);
    double infinity = std::numeric_limits<double>::max();

    auto token = het::exec::reduce(
        exec,
        dt_comp,
        result,
        infinity,
        min_op_t{},
        infinity
    );
    token.wait(stream);

    field_t<double, 1> host_result(
        extents(iarray<1>{1}),
        het::locality_t::host()
    );
    het::mem::copy_async(
        host_result.data(),
        het::locality_t::host(),
        result.data(),
        result.locality(),
        sizeof(double),
        stream
    )
        .wait(stream);

    double computed = *host_result.data();
    double expected = dx / (5.0 + std::sqrt(gamma * 10.0 / 1.0));

    assert(std::abs(computed - expected) < 1e-9);
    std::cout << "  min dt: " << computed << " ✓" << std::endl;
    std::cout << "[PASS] CFL reduction verified" << std::endl;

    return 0;
}
