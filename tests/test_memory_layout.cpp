// =============================================================================
// test_memory_layout.cpp
//
// benchmark aos vs soa memory layout for typical stencil operations
// measures bandwidth and kernel execution time on gpu
// =============================================================================

#include "build_config.hpp"
#include "containers/state_struct.hpp"
#include "containers/vector.hpp"
#include "decorators.hpp"
#include "grid/domain.hpp"
#include "grid/field.hpp"
#include "utility/enums.hpp"
#include "xpu/xpu.hpp"

#include <chrono>
#include <cstdint>
#include <iostream>

using namespace simbi;
using namespace simbi::structs;

constexpr std::uint64_t GRID_SIZE  = 8192;
constexpr std::uint64_t ITERATIONS = 1000;

// simple hydro state for testing
using state_t = primitive_t<regime_t::NEWTONIAN, 1, void>;

// aos: single field of structs
struct aos_layout_t
{
    grid::field_t<state_t, 1> state;

    aos_layout_t(const grid::domain_t<1>& domain) : state(domain)
    {
        state_t init_state{1.0, {0.5}, 1.0, 0.0};
        for (auto idx : domain) {
            state[idx] = init_state;
        }
    }

    template <typename Exec>
    void stencil_op(Exec& exec, const grid::domain_t<1>& interior)
    {
        auto view       = state[interior];
        state[interior] = view.map(
                                  [=] DUAL(const state_t& s) {
                                      state_t result;
                                      result.rho    = s.rho * 1.01;
                                      result.vel[0] = s.vel[0] * 0.99;
                                      result.pre    = s.pre * 1.005;
                                      result.chi    = s.chi;
                                      return result;
                                  }
        ).with(exec);
    }
};

// soa: separate fields for each component (unfused)
struct soa_layout_t
{
    grid::field_t<real, 1> rho;
    grid::field_t<real, 1> vel;
    grid::field_t<real, 1> pre;
    grid::field_t<real, 1> chi;

    soa_layout_t(const grid::domain_t<1>& domain)
        : rho(domain), vel(domain), pre(domain), chi(domain)
    {
        for (auto idx : domain) {
            rho[idx] = 1.0;
            vel[idx] = 0.5;
            pre[idx] = 1.0;
            chi[idx] = 0.0;
        }
    }

    template <typename Exec>
    void stencil_op(Exec& exec, const grid::domain_t<1>& interior)
    {
        rho[interior] = rho[interior].map([=] DUAL(real r) { return r * 1.01; }).with(exec);
        vel[interior] = vel[interior].map([=] DUAL(real v) { return v * 0.99; }).with(exec);
        pre[interior] = pre[interior].map([=] DUAL(real p) { return p * 1.005; }).with(exec);
    }
};

// soa_fused: separate fields with manual kernel fusion
struct soa_fused_layout_t
{
    grid::field_t<real, 1> rho;
    grid::field_t<real, 1> vel;
    grid::field_t<real, 1> pre;
    grid::field_t<real, 1> chi;

    soa_fused_layout_t(const grid::domain_t<1>& domain)
        : rho(domain), vel(domain), pre(domain), chi(domain)
    {
        for (auto idx : domain) {
            rho[idx] = 1.0;
            vel[idx] = 0.5;
            pre[idx] = 1.0;
            chi[idx] = 0.0;
        }
    }

    template <typename Exec>
    void stencil_op(Exec& exec, const grid::domain_t<1>& interior)
    {
        auto rho_view = rho[interior];
        auto vel_view = vel[interior];
        auto pre_view = pre[interior];

        exec.dispatch(interior, [=] DUAL(const iarray<1>& coord) mutable {
            rho_view[coord] = rho_view(coord) * 1.01;
            vel_view[coord] = vel_view(coord) * 0.99;
            pre_view[coord] = pre_view(coord) * 1.005;
        });
    }
};

template <typename Layout, typename Exec>
double benchmark(Exec& exec, const grid::domain_t<1>& domain)
{
    using clock = std::chrono::high_resolution_clock;

    auto alloc_domain =
        grid::domain_t<1>(iarray<1>{domain.start[0] - 1}, iarray<1>{domain.fin[0] + 1});

    Layout layout(alloc_domain);
    exec.sync();

    auto start = clock::now();
    for (std::uint64_t ii = 0; ii < ITERATIONS; ++ii) {
        layout.stencil_op(exec, domain);
    }
    exec.sync();
    auto end = clock::now();

    return std::chrono::duration<double, std::milli>(end - start).count();
}

int main()
{
    std::cout << "memory layout benchmark: aos vs soa\n";
    std::cout << "grid size: " << GRID_SIZE << "\n";
    std::cout << "iterations: " << ITERATIONS << "\n";
    std::cout << "state size: " << sizeof(state_t) << " bytes\n\n";

#ifdef XPU_CUDA_AVAILABLE
    using execution_space = xpu::cuda_space_t;
    std::cout << "execution: gpu (cuda)\n\n";
#elif defined(XPU_HIP_AVAILABLE)
    using execution_space = xpu::hip_space_t;
    std::cout << "execution: gpu (hip)\n\n";
#else
    using execution_space = xpu::cpu_space_t;
    std::cout << "execution: cpu (openmp)\n\n";
#endif

    xpu::executor_t<execution_space> exec(0);
    grid::domain_t<1>                domain(iarray<1>{0}, iarray<1>{GRID_SIZE});

    double aos_time       = benchmark<aos_layout_t>(exec, domain);
    double soa_time       = benchmark<soa_layout_t>(exec, domain);
    double soa_fused_time = benchmark<soa_fused_layout_t>(exec, domain);

    std::cout << "results:\n";
    std::cout << "  aos:        " << aos_time << " ms\n";
    std::cout << "  soa:        " << soa_time << " ms (3 kernel launches)\n";
    std::cout << "  soa_fused:  " << soa_fused_time << " ms (1 kernel launch)\n";
    std::cout << "\n";
    std::cout << "ratios:\n";
    std::cout << "  aos/soa:        " << (aos_time / soa_time) << "x\n";
    std::cout << "  aos/soa_fused:  " << (aos_time / soa_fused_time) << "x\n";
    std::cout << "  soa/soa_fused:  " << (soa_time / soa_fused_time)
              << "x (kernel launch overhead)\n";

    if (aos_time < soa_fused_time * 1.1) {
        std::cout << "\n[PASS] aos competitive with fused soa\n";
    }
    else if (soa_fused_time < aos_time * 0.9) {
        std::cout << "\n[INFO] fused soa faster than aos by >10%\n";
    }
    else {
        std::cout << "\n[PASS] aos and fused soa within 10%\n";
    }

    return 0;
}
