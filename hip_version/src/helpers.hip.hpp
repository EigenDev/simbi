#ifndef HELPERS_HIP_HPP
#define HELPERS_HIP_HPP

#include "config.hpp"
#include "hydro_structs.hpp"
#include "helpers.hpp"

namespace simbi
{
    //======================================
    //          GPU TEMPLATES
    //======================================
    template <unsigned int blockSize>
    __device__ void warpReduceMin(volatile real smem[BLOCK_SIZE], int tid);

    template<typename T, typename N, unsigned int blockSize>
    __global__ typename std::enable_if<is_1D_primitive<N>::value>::type dtWarpReduce(T *s);

    template <typename T>
    __global__ void config_ghosts1DGPU(T *dev_sim, int, bool = true);

    //======================================
    //              HELPER OVERLOADS
    //======================================

    __global__ void config_ghosts2DGPU(
    SRHD2D *d_sim, 
    int x1grid_size, 
    int x2grid_size, 
    bool first_order,
    bool bipolar = true);

    __global__ void config_ghosts3DGPU(
    SRHD3D *d_sim, 
    int x1grid_size, 
    int x2grid_size,
    int x3grid_size,  
    bool first_order,
    bool bipolar = true);

} // end simbi

#include "helpers.hip.tpp"
#endif