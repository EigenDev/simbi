#include "compat.hpp"
#include "compute/computation.hpp"
#include "containers/vector.hpp"
#include "geometry/block_geometry.hpp"
#include "geometry/coordinate_map.hpp"
#include "geometry/metrics.hpp"
#include "grid/algebra.hpp"
#include "grid/amr/api.hpp"
#include "grid/amr/flux_correction.hpp"
#include "grid/connectivity.hpp"
#include "grid/domain.hpp"
#include "grid/field.hpp"
#include "test_helpers.hpp"
#include "xpu/execution/cpu_space.hpp"
#include "xpu/execution/cuda_space.hpp"
#include "xpu/execution/executor.hpp"

#include <cassert>
#include <cmath>
#include <iostream>

using namespace simbi;
using namespace simbi::grid;

struct linear_gradient_t
{
    DUAL double operator()(const iarray<2>& coord) const
    {
        return static_cast<double>((coord[0] + 0.5) + (coord[1] + 0.5));
    }
};

int main()
{
    std::cout << "testing SMR operations..." << std::endl;

    iarray<2>         ratio{2, 2};
    grid::domain_t<2> coarse_domain{{0, 0}, {8, 8}};
    grid::domain_t<2> fine_active{{4, 4}, {12, 12}};
    grid::domain_t<2> fine_alloc = domain_algebra::expand(fine_active, iarray<2>{2, 2});

#ifdef XPU_CUDA_AVAILABLE
    using execution_space = xpu::cuda_space;
#else
    using execution_space = xpu::cpu_space;
#endif

    xpu::executor_t<execution_space> exec(0);
    field_t<double, 2>               coarse(coarse_domain);
    field_t<double, 2>               fine(fine_alloc);

    // test restriction
    fine   = compute::constant(fine.domain(), 10.0).with(exec);
    coarse = compute::constant(coarse.domain(), 0.0).with(exec);

    amr::restrict_to_coarse(coarse, fine, ratio, exec);

    auto c_view = coarse.view();
    assert(std::abs(c_view({2, 2}) - 10.0) < 1e-9);
    assert(c_view({0, 0}) == 0.0);
    std::cout << "  restriction ✓" << std::endl;

    // test prolongation
    coarse = test_helpers::make_computation<2>(coarse_domain, linear_gradient_t{}).with(exec);
    fine   = compute::constant(fine.domain(), -999.0).with(exec);

    amr::fill_fine_ghosts(fine, coarse, fine_active, ratio, exec);

    auto f_view = fine.view();
    assert(std::abs(f_view({3, 4}) - 4.0) < 1e-9);
    assert(f_view({4, 4}) == -999.0);
    std::cout << "  prolongation ✓" << std::endl;

    // test flux correction
    grid::domain_t<2>               footprint{{2, 2}, {6, 6}};
    amr::flux_register_t<double, 2> flux_reg(footprint, ratio);
    flux_reg.initialize_face(0, side_t::left);

    // create uniform cartesian geometry for test
    auto x1_map = geometry::uniform_map_t(0.0, 1.0);
    auto x2_map = geometry::uniform_map_t(0.0, 1.0);
    auto metric = geometry::cartesian_metric_t(x1_map, x2_map);
    auto motion = geometry::motion_state_t::static_mesh();
    auto geo    = geometry::block_geometry(metric, motion);

    double dt = 0.1;

    field_t<double, 2> c_flux(grid::domain_t<2>{{2, 2}, {3, 6}});
    c_flux = compute::constant(c_flux.domain(), 1.0).with(exec);
    flux_reg.accumulate_coarse(exec, c_flux, geo, 0, side_t::left, dt);

    field_t<double, 2> f_flux(grid::domain_t<2>{{4, 4}, {6, 12}});
    f_flux = compute::constant(f_flux.domain(), 1.2).with(exec);
    flux_reg.accumulate_fine(exec, f_flux, geo, 0, side_t::left, dt);

    auto* reg = flux_reg.get_register(0, side_t::left);
    assert(std::abs(reg->view()({2, 2}) - 0.02) < 1e-9);
    std::cout << "  flux correction ✓" << std::endl;

    std::cout << "[PASS] SMR operations verified" << std::endl;
    return 0;
}
