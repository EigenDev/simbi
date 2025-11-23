#include "compat.hpp"
#include "compute/computation.hpp"
#include "containers/vector.hpp"
#include "geometry/boundary/driver.hpp"
#include "geometry/coordinate_map.hpp"
#include "geometry/metrics.hpp"
#include "grid/block_info.hpp"
#include "grid/boundary.hpp"
#include "grid/connectivity.hpp"
#include "grid/domain.hpp"
#include "grid/field.hpp"
#include "grid/mesh_config.hpp"
#include "grid/patch_id.hpp"
#include "grid/skeleton.hpp"
#include "hesi/adapter.hpp"
#include "hesi/core/types.hpp"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <variant>

using namespace simbi::grid;
using namespace simbi::geometry;

struct mock_vm_t {
    DEV double apply(
        const simbi::vector_t<double, 1>& coords,
        const double& interior_state,
        double time
    ) const
    {
        return coords[0] * time + interior_state;
    }
};

struct mock_geo_service_t {
    auto create_map(std::uint64_t, std::int64_t, int) const
    {
        return std::variant<uniform_map_t>{uniform_map_t{0.0, 1.0}};
    }
};

struct test_policy_t {
    double apply(double val, std::uint64_t, side_t, boundary_type_t) const
    {
        return val;
    }
};

int main()
{
    using test_metric_t = cartesian_metric_t<uniform_map_t>;
    std::cout << "testing dynamic boundaries..." << std::endl;

    domain_t<1> alloc{{-1}, {5}};
    domain_t<1> active{{0}, {4}};

    auto backend = simbi::het::info::is_gpu ? simbi::het::backend_type_t::cuda
                                            : simbi::het::backend_type_t::cpu;
    simbi::het::locality_t loc{backend, 0};
    simbi::het::stream_t stream(backend);
    simbi::het::executor_t exec(stream);

    field_t<double, 1> u(alloc, loc);

    auto init = [](auto coord) {
        return (coord[0] >= 0 && coord[0] < 4) ? 100.0 : -999.0;
    };
    u = simbi::compute::computation(alloc, init).with(exec);

    block_info_t<1> block;
    block.id       = patch_id_t{0, {0, 0, 0}};
    block.geometry = active;
    block.set_boundary(0, side_t::left, boundary_type_t::dynamic);
    block.set_boundary(0, side_t::right, boundary_type_t::outflow);

    skeleton_t<1> skeleton;
    skeleton.add_block(block);

    mesh_config_t<1> config;
    config.global_cells = {4};

    double time = 2.0;
    dynamic_context_t
        ctx(use_metric<test_metric_t>, mock_geo_service_t{}, mock_vm_t{}, time);

    boundary_driver_t::apply_boundaries(
        u,
        block.id,
        skeleton,
        config,
        test_policy_t{},
        ctx,
        exec
    );

    double ghost = u.view()({-1});
    assert(std::abs(ghost - 99.0) < 1e-9);
    std::cout << "  ghost value: " << ghost << " ✓" << std::endl;
    std::cout << "[PASS] dynamic boundaries verified" << std::endl;

    return 0;
}
