#ifndef HELPERS_HIP_HPP
#define HELPERS_HIP_HPP

#include "common/config.hpp"
#include "common/hydro_structs.hpp"
#include "common/helpers.hpp"
#include "srhydro1D.hip.hpp"
#include "srhydro2D.hip.hpp"
#include "srhydro3D.hip.hpp"
namespace simbi
{
    //======================================
    //          GPU TEMPLATES
    //======================================
    template <unsigned int blockSize>
    GPU_DEV void warpReduceMin(volatile real smem[BLOCK_SIZE], const int tid);

    template<unsigned int blockSize>
    GPU_DEV void warpReduceMin(volatile real smem[BLOCK_SIZE2D * BLOCK_SIZE2D], const int tx, const int ty);

    template<unsigned int blockSize>
    GPU_DEV void warpReduceMin(volatile real smem[BLOCK_SIZE2D * BLOCK_SIZE2D], const int tx, const int ty, const int tz);

    template<typename T, typename N, unsigned int blockSize>
    GPU_LAUNCHABLE typename std::enable_if<is_1D_primitive<N>::value>::type 
    dtWarpReduce(T *s);

    template<typename T, typename N, unsigned int blockSize>
    GPU_LAUNCHABLE typename std::enable_if<is_2D_primitive<N>::value>::type 
    dtWarpReduce(T *s, simbi::Geometry geometry);

    template<typename T, typename N, unsigned int blockSize>
    GPU_LAUNCHABLE typename std::enable_if<is_3D_primitive<N>::value>::type 
    dtWarpReduce(T *s, simbi::Geometry geometry);

    //======================================
    //              HELPER OVERLOADS
    //======================================
    GPU_CALLABLE_INLINE
    bool quirk_strong_shock(real pl, real pr)
    {
        return abs(pr - pl) / my_min(pl, pr) > QUIRK_THERSHOLD;
    }

    void config_ghosts1DGPU(
        const ExecutionPolicy<> p,
        SRHD *dev_sim, 
        const int, 
        const bool = true);
        
    void config_ghosts2DGPU(
        const ExecutionPolicy<> p,
        SRHD2D *sim, 
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