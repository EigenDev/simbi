#include "compute/computation.hpp"
#include "containers/vector.hpp"
#include "decorators.hpp"
#include "geometry/block_geometry.hpp"
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
#include "test_helpers.hpp"
#include "xpu/xpu.hpp"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <variant>

using namespace simbi;
using namespace simbi::grid;
using namespace simbi::geometry;

struct mock_vm_t
{
    DEV double
    apply(const simbi::vector_t<double, 1>& coords, const double& interior_state, double time) const
    {
        return coords[0] * time + interior_state;
    }
};

struct mock_geo_service_t
{
    auto create_map(std::uint64_t, std::int64_t, int) const
    {
        return std::variant<uniform_map_t>{uniform_map_t{0.0, 1.0}};
    }
};

struct test_policy_t
{
    double apply(double val, std::uint64_t, side_t, boundary_type_t) const
    {
        return val;
    }
};

int main()
{
    std::cout << "testing dynamic boundaries..." << std::endl;

    domain_t<1> alloc(iarray<1>{-1}, iarray<1>{5});
    domain_t<1> active(iarray<1>{0}, iarray<1>{4});

#ifdef XPU_CUDA_AVAILABLE
    using execution_space = simbi::xpu::cuda_space_t;
#else
    using execution_space = simbi::xpu::cpu_space_t;
#endif

    simbi::xpu::executor_t<execution_space> exec(0);
    field_t<double, 1>                      u(alloc);

    auto init = [](auto coord) { return (coord[0] >= 0 && coord[0] < 4) ? 100.0 : -999.0; };
    u         = test_helpers::make_computation<1>(alloc, init).with(exec);

    block_info_t<1> block;
    block.id       = patch_id_t{0, {0, 0, 0}};
    block.geometry = active;
    block.set_boundary(0, side_t::left, boundary_type_t::dynamic);
    block.set_boundary(0, side_t::right, boundary_type_t::outflow);

    skeleton_t<1> skeleton;
    skeleton.add_block(block);

    mesh_config_t<1> config;
    config.global_cells = {4};

    // create static mesh geometry
    auto metric    = geometry::cartesian_metric_t{geometry::dummy_map_t{}};
    auto motion    = geometry::motion_state_t::static_mesh();
    auto block_geo = geometry::block_geometry(metric, motion);

    double            time = 2.0;
    dynamic_context_t ctx(metric_kind_t::cartesian, mock_geo_service_t{}, mock_vm_t{}, time);

    boundary_driver_t::apply_boundaries(
        u,
        block.id,
        skeleton,
        config,
        test_policy_t{},
        ctx,
        block_geo,
        exec
    );

    double ghost = u.view()({-1});
    assert(std::abs(ghost - 99.0) < 1e-9);
    std::cout << "  ghost value: " << ghost << " ✓" << std::endl;
    std::cout << "[PASS] dynamic boundaries verified" << std::endl;

    return 0;
}
