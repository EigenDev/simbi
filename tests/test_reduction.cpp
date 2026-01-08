#include "compat.hpp"
#include "compute/computation.hpp"
#include "containers/vector.hpp"
#include "grid/domain.hpp"
#include "grid/field.hpp"
#include "test_helpers.hpp"
#include "xpu/execution/cpu_space.hpp"
#include "xpu/execution/cuda_space.hpp"
#include "xpu/execution/executor.hpp"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>

using namespace simbi;
using namespace simbi::grid;

struct cfl_calculator_t
{
    double gamma;
    double dx;

    DUAL double operator()(double rho, double press, double vel) const
    {
        double cs     = (rho > 0.0 && press > 0.0) ? std::sqrt(gamma * press / rho) : 0.0;
        double signal = std::abs(vel) + cs;
        return (signal > 1e-12) ? (dx / signal) : 1e12;
    }
};

struct min_op_t
{
    DUAL double operator()(double a, double b) const
    {
        return (a < b) ? a : b;
    }
};

int main()
{
    std::cout << "testing CFL reduction..." << std::endl;

    constexpr std::uint64_t N = 100;
    grid::domain_t<2>       domain{{0, 0}, {N, N}};

    double gamma = 1.4;
    double dx    = 0.1;

#ifdef XPU_CUDA_AVAILABLE
    using execution_space = xpu::cuda_space;
#else
    using execution_space = xpu::cpu_space;
#endif

    xpu::executor_t<execution_space> exec(0);

    auto init = [=] DUAL(auto coord) {
        double x  = coord[0] - (double) N / 2.0;
        double y  = coord[1] - (double) N / 2.0;
        double r2 = x * x + y * y;

        return (r2 < (N / 4.0) * (N / 4.0)) ? vector_t<double, 3>{1.0, 10.0, 5.0}
                                            : vector_t<double, 3>{1.0, 0.1, 0.0};
    };

    auto physics = test_helpers::make_computation<2>(domain, init);
    auto dt_comp =
        physics.map([=](auto val) { return cfl_calculator_t{gamma, dx}(val[0], val[1], val[2]); });

    double infinity = std::numeric_limits<double>::max();
    double computed = exec.reduce(
        domain,
        infinity,
        [=] DUAL(auto coord) {
            auto val = dt_comp(coord);
            return val;
        },
        min_op_t{}
    );
    double expected = dx / (5.0 + std::sqrt(gamma * 10.0 / 1.0));

    assert(std::abs(computed - expected) < 1e-9);
    std::cout << "  min dt: " << computed << " ✓" << std::endl;
    std::cout << "[PASS] CFL reduction verified" << std::endl;

    return 0;
}
