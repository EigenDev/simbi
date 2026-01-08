#include "compat.hpp"
#include "compute/computation.hpp"
#include "grid/domain.hpp"
#include "grid/field.hpp"
#include "xpu/execution/cpu_space.hpp"
#include "xpu/execution/cuda_space.hpp"
#include "xpu/execution/executor.hpp"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <vector>

using namespace simbi;

int main()
{
    std::cout << "testing field map operations..." << std::endl;

#ifdef XPU_CUDA_AVAILABLE
    using execution_space = xpu::cuda_space;
#else
    using execution_space = xpu::cpu_space;
#endif

    xpu::executor_t<execution_space> exec(0);
    grid::domain_t<1>                alloc_domain({-1}, {11});
    grid::field_t<real, 1>           u(alloc_domain);

    std::int64_t halo_width = 1;
    auto         interior   = alloc_domain.contract(halo_width);

    u           = compute::constant(alloc_domain, 0.0).with(exec);
    u[interior] = compute::constant(interior, 1.0).with(exec);

    real scale  = 2.5;
    u[interior] = u[interior].map([=] DUAL(real v) { return v * scale; }).with(exec);

    exec.sync();

    std::vector<real> host_data(alloc_domain.size());
    std::copy(u.data(), u.data() + alloc_domain.size(), host_data.data());

    assert(host_data[0] == 0.0);
    assert(host_data[11] == 0.0);
    std::cout << "  ghosts preserved ✓" << std::endl;

    for (std::size_t i = 1; i <= 10; ++i) {
        assert(host_data[i] == 2.5);
    }
    std::cout << "  interior scaled ✓" << std::endl;

    std::cout << "[PASS] field map verified" << std::endl;
    return 0;
}
