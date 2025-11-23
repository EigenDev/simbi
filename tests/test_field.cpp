#include "compat.hpp"
#include "compute/computation.hpp"
#include "grid/domain.hpp"
#include "grid/field.hpp"
#include "hesi/adapter.hpp"
#include "hesi/core/types.hpp"
#include "hesi/mem/transfer.hpp"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <vector>

using namespace simbi;

int main()
{
    std::cout << "testing field map operations..." << std::endl;

    auto backend = het::info::is_gpu ? het::backend_type_t::cuda
                                     : het::backend_type_t::cpu;
    het::locality_t loc{backend, 0};
    het::stream_t stream(backend);
    het::executor_t exec(stream);

    grid::domain_t<1> alloc_domain({-1}, {11});
    grid::field_t<real, 1> u(alloc_domain, loc);

    std::int64_t ghost_width = 1;
    auto interior            = alloc_domain.contract(ghost_width);

    u           = compute::constant(alloc_domain, 0.0).with(exec);
    u[interior] = compute::constant(interior, 1.0).with(exec);

    real scale = 2.5;
    u[interior] =
        u[interior].map([=] DUAL(real v) { return v * scale; }).with(exec);

    exec.stream().synchronize();

    std::vector<real> host_data(alloc_domain.size());
    het::mem::copy(
        host_data.data(),
        het::locality_t::host(),
        u.view().data(),
        u.locality(),
        alloc_domain.size() * sizeof(real)
    );

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
