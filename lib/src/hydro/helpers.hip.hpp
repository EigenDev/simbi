#ifndef HELPERS_HIP_HPP
#define HELPERS_HIP_HPP

#include "common/config.hpp"
#include "common/hydro_structs.hpp"
#include "common/helpers.hpp"
#include "euler1D.hpp"
#include "euler2D.hpp"
#include "srhydro1D.hip.hpp"
#include "srhydro2D.hip.hpp"
#include "srhydro3D.hip.hpp"
namespace simbi
{
    //======================================
    //          GPU TEMPLATES
    //======================================
    template <unsigned int blockSize>
    GPU_DEV void warpReduceMin(volatile real* smem, unsigned int tid);

    template<typename T, typename N, unsigned int blockSize>
    GPU_LAUNCHABLE typename std::enable_if<is_1D_primitive<N>::value>::type 
    dtWarpReduce(T *s);

    template<typename T, typename N, unsigned int blockSize>
    GPU_LAUNCHABLE typename std::enable_if<is_2D_primitive<N>::value>::type
    dtWarpReduce(T *s, const simbi::Geometry geometry, 
        luint bytes,
        real dx, 
        real dy, 
        real rmin = 0, 
        real rmax = 1,
        real ymin = 0,
        real ymax = 1);

    template<typename T, typename N, unsigned int blockSize>
    GPU_LAUNCHABLE typename std::enable_if<is_3D_primitive<N>::value>::type 
    dtWarpReduce(T *s, const simbi::Geometry geometry);

    //======================================
    //              HELPER OVERLOADS
    //======================================
    GPU_CALLABLE_INLINE
    bool quirk_strong_shock(real pl, real pr)
    {
        return std::abs(pr - pl) / my_min(pl, pr) > QUIRK_THRESHOLD;
    }

    GPU_CALLABLE_INLINE
    constexpr unsigned int kronecker(luint i, luint j) { return (i == j ? 1 : 0); }

    void config_ghosts1DGPU(
        const ExecutionPolicy<> p,
        SRHD *dev_sim, 
        const int grid_size,
        const bool first_order, 
        const bool reflecting = true);

    void config_ghosts1DGPU(
        const ExecutionPolicy<> p,
        Newtonian1D *dev_sim, 
        const int grid_size,
        const bool first_order, 
        const bool reflecting = true);
        
    void config_ghosts2DGPU(
        const ExecutionPolicy<> p,
        SRHD2D *sim, 
        const int x1grid_size, 
        const int x2grid_size, 
        const bool first_order,
        const bool bipolar = true);

    void config_ghosts2DGPU(
        const ExecutionPolicy<> p,
        Newtonian2D *sim, 
        const int x1grid_size, 
        const int x2grid_size, 
        const bool first_order,
        const bool bipolar = true);

    void config_ghosts3DGPU(
        const ExecutionPolicy<> p,
        SRHD3D *sim, 
        const int x1grid_size, 
        const int x2grid_size,
        const int x3grid_size,  
        const bool first_order,
        const bool bipolar = true);

} // end simbi

#include "helpers.hip.tpp"
#endif