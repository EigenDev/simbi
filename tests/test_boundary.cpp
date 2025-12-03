#include "compute/computation.hpp"
#include "containers/vector.hpp"
#include "geometry/boundary/driver.hpp"
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
#include "test_helpers.hpp"

#include <cassert>
#include <cstdint>
#include <iostream>

using namespace simbi;
using namespace simbi::grid;

struct reflect_policy_t {
    double apply(
        double val,
        std::uint64_t /*dim*/,
        side_t /*side*/,
        boundary_type_t type
    ) const
    {
        return (type == boundary_type_t::reflect) ? -val : val;
    }
};

int main()
{
    std::cout << "testing boundary driver..." << std::endl;

    // setup: 4x4 active interior with 1-cell ghost ring
    grid::domain_t<2> alloc_domain{{-1, -1}, {5, 5}};
    std::uint64_t halo_width = 1;

    auto backend = het::info::is_gpu ? het::backend_type_t::cuda
                                     : het::backend_type_t::cpu;
    het::locality_t loc{backend, 0};
    het::stream_t stream(backend);
    het::executor_t exec(stream);

    field_t<double, 2> u(alloc_domain, loc);

    // initialize: interior = 10.0, ghosts = -999.0
    auto init = [](const iarray<2>& coord) {
        bool inside =
            coord[0] >= 0 && coord[0] < 4 && coord[1] >= 0 && coord[1] < 4;
        return inside ? 10.0 : -999.0;
    };

    u = test_helpers::make_computation<2>(u.domain(), init).with(exec);

    // configure block with mixed boundaries
    block_info_t<2> block;
    block.id       = patch_id_t{0, {0, 0, 0}};
    block.geometry = alloc_domain.contract(halo_width);
    block.set_boundary(0, side_t::left, boundary_type_t::reflect);
    block.set_boundary(0, side_t::right, boundary_type_t::outflow);
    block.set_boundary(1, side_t::left, boundary_type_t::periodic);
    block.set_boundary(1, side_t::right, boundary_type_t::periodic);

    skeleton_t<2> skeleton;
    skeleton.add_block(block);

    mesh_config_t<2> config;
    config.global_cells = {4, 4};

    // apply boundaries
    geometry::boundary_driver_t::apply_boundaries(
        u,
        block.id,
        skeleton,
        config,
        reflect_policy_t{},
        geometry::simple_context_t{},
        exec
    );

    // verify
    auto view = u.view();

    assert(view({0, 0}) == 10.0);
    std::cout << "  interior preserved ✓" << std::endl;

    double left_ghost = view({-1, 2});
    assert(left_ghost == -10.0);
    std::cout << "  reflect boundary ✓" << std::endl;

    double corner_ghost = view({-1, -1});
    assert(corner_ghost == -10.0);
    std::cout << "  corner cascade ✓" << std::endl;

    std::cout << "[PASS] boundary driver verified" << std::endl;
    return 0;
}
