#include "helpers.hip.hpp"

//==================================
//              GPU HELPERS
//==================================
namespace simbi{
    __global__ void config_ghosts2DGPU(
    SRHD2D *dev_sim, 
    int x1grid_size, 
    int x2grid_size, 
    bool first_order,
    bool bipolar)
    {
        const int ii = blockDim.x * blockIdx.x + threadIdx.x;
        const int jj = blockDim.y * blockIdx.y + threadIdx.y;
        sr2d::Conserved *u_state = dev_sim->gpu_state2D;
        if (first_order){
            if ((jj < x2grid_size) && (ii < x1grid_size))
            {
                if (jj < 1){
                    u_state[ii + x1grid_size * jj].D   =   u_state[ii + x1grid_size * 1].D;
                    u_state[ii + x1grid_size * jj].S1  =   u_state[ii + x1grid_size * 1].S1;
                    u_state[ii + x1grid_size * jj].S2  = - u_state[ii + x1grid_size * 1].S2;
                    u_state[ii + x1grid_size * jj].tau =   u_state[ii + x1grid_size * 1].tau;
                    
                } else if (jj > x2grid_size - 2) {
                    u_state[ii + x1grid_size * jj].D    =   u_state[(x2grid_size - 2) * x1grid_size + ii].D;
                    u_state[ii + x1grid_size * jj].S1   =   u_state[(x2grid_size - 2) * x1grid_size + ii].S1;
                    u_state[ii + x1grid_size * jj].S2   = - u_state[(x2grid_size - 2) * x1grid_size + ii].S2;
                    u_state[ii + x1grid_size * jj].tau  =   u_state[(x2grid_size - 2) * x1grid_size + ii].tau;

                } else {
                    u_state[jj * x1grid_size].D    = u_state[jj * x1grid_size + 1].D;
                    u_state[jj * x1grid_size + x1grid_size - 1].D = u_state[jj*x1grid_size + x1grid_size - 2].D;

                    u_state[jj * x1grid_size + 0].S1               = - u_state[jj * x1grid_size + 1].S1;
                    u_state[jj * x1grid_size + x1grid_size - 1].S1 =   u_state[jj * x1grid_size + x1grid_size - 2].S1;

                    u_state[jj * x1grid_size + 0].S2                = u_state[jj * x1grid_size + 1].S2;
                    u_state[jj * x1grid_size + x1grid_size - 1].S2  = u_state[jj * x1grid_size + x1grid_size - 2].S2;

                    u_state[jj * x1grid_size + 0].tau               = u_state[jj * x1grid_size + 1].tau;
                    u_state[jj * x1grid_size + x1grid_size - 1].tau = u_state[jj * x1grid_size + x1grid_size - 2].tau;
                }
            }

        } else {
            if(jj < x2grid_size)
            {

                // Fix the ghost zones at the radial boundaries
                u_state[jj * x1grid_size +  0].D               = u_state[jj * x1grid_size +  3].D;
                u_state[jj * x1grid_size +  1].D               = u_state[jj * x1grid_size +  2].D;
                u_state[jj * x1grid_size +  x1grid_size - 1].D = u_state[jj * x1grid_size +  x1grid_size - 3].D;
                u_state[jj * x1grid_size +  x1grid_size - 2].D = u_state[jj * x1grid_size +  x1grid_size - 3].D;

                u_state[jj * x1grid_size + 0].S1               = - u_state[jj * x1grid_size + 3].S1;
                u_state[jj * x1grid_size + 1].S1               = - u_state[jj * x1grid_size + 2].S1;
                u_state[jj * x1grid_size + x1grid_size - 1].S1 =   u_state[jj * x1grid_size + x1grid_size - 3].S1;
                u_state[jj * x1grid_size + x1grid_size - 2].S1 =   u_state[jj * x1grid_size + x1grid_size - 3].S1;

                u_state[jj * x1grid_size + 0].S2               = u_state[jj * x1grid_size + 3].S2;
                u_state[jj * x1grid_size + 1].S2               = u_state[jj * x1grid_size + 2].S2;
                u_state[jj * x1grid_size + x1grid_size - 1].S2 = u_state[jj * x1grid_size + x1grid_size - 3].S2;
                u_state[jj * x1grid_size + x1grid_size - 2].S2 = u_state[jj * x1grid_size + x1grid_size - 3].S2;

                u_state[jj * x1grid_size + 0].tau                = u_state[jj * x1grid_size + 3].tau;
                u_state[jj * x1grid_size + 1].tau                = u_state[jj * x1grid_size + 2].tau;
                u_state[jj * x1grid_size + x1grid_size - 1].tau  = u_state[jj * x1grid_size + x1grid_size - 3].tau;
                u_state[jj * x1grid_size + x1grid_size - 2].tau  = u_state[jj * x1grid_size + x1grid_size - 3].tau;

                // Fix the ghost zones at the angular boundaries
                
                if (jj < 2){
                    if (ii < x1grid_size){
                        if (jj == 0){
                            u_state[jj * x1grid_size + ii].D   =   u_state[3 * x1grid_size + ii].D;
                            u_state[jj * x1grid_size + ii].S1  =   u_state[3 * x1grid_size + ii].S1;
                            u_state[jj * x1grid_size + ii].S2  =   u_state[3 * x1grid_size + ii].S2;
                            u_state[jj * x1grid_size + ii].tau =   u_state[3 * x1grid_size + ii].tau;
                        } else {
                            u_state[jj * x1grid_size + ii].D    =   u_state[2 * x1grid_size + ii].D;
                            u_state[jj * x1grid_size + ii].S1   =   u_state[2 * x1grid_size + ii].S1;
                            u_state[jj * x1grid_size + ii].S2   =   u_state[2 * x1grid_size + ii].S2;
                            u_state[jj * x1grid_size + ii].tau  =   u_state[2 * x1grid_size + ii].tau;
                        }
                    }
                } else if (jj > x2grid_size - 3) {
                    if (ii < x1grid_size){
                        if (jj == x2grid_size - 1){
                            u_state[jj * x1grid_size + ii].D   =   u_state[(x2grid_size - 4) * x1grid_size + ii].D;
                            u_state[jj * x1grid_size + ii].S1  =   u_state[(x2grid_size - 4) * x1grid_size + ii].S1;
                            u_state[jj * x1grid_size + ii].S2  =   u_state[(x2grid_size - 4) * x1grid_size + ii].S2;
                            u_state[jj * x1grid_size + ii].tau =   u_state[(x2grid_size - 4) * x1grid_size + ii].tau;
                        } else {
                            u_state[jj * x1grid_size + ii].D   =   u_state[(x2grid_size - 3) * x1grid_size + ii].D;
                            u_state[jj * x1grid_size + ii].S1  =   u_state[(x2grid_size - 3) * x1grid_size + ii].S1;
                            u_state[jj * x1grid_size + ii].S2  =   u_state[(x2grid_size - 3) * x1grid_size + ii].S2;
                            u_state[jj * x1grid_size + ii].tau =   u_state[(x2grid_size - 3) * x1grid_size + ii].tau;
                        }
                    }
                }
                
            }

        }
    };

    __global__ void config_ghosts3DGPU(
        SRHD3D *dev_sim, 
        int x1grid_size, 
        int x2grid_size,
        int x3grid_size, 
        bool first_order,
        bool bipolar)
    {
        const int ii = blockDim.x * blockIdx.x + threadIdx.x;
        const int jj = blockDim.y * blockIdx.y + threadIdx.y;
        const int kk = blockDim.z * blockIdx.z + threadIdx.z;
        sr3d::Conserved *u_state = dev_sim->gpu_state3D;
        if (first_order){
            if ((jj < x2grid_size) && (ii < x1grid_size) && (kk < x3grid_size))
            {
                if (jj < 1){
                    u_state[ii + x1grid_size * jj + x1grid_size * x2grid_size * kk].D   =   u_state[ii + x1grid_size * 1 + x1grid_size * x2grid_size * kk].D;
                    u_state[ii + x1grid_size * jj + x1grid_size * x2grid_size * kk].S1  =   u_state[ii + x1grid_size * 1 + x1grid_size * x2grid_size * kk].S1;
                    u_state[ii + x1grid_size * jj + x1grid_size * x2grid_size * kk].S2  =   u_state[ii + x1grid_size * 1 + x1grid_size * x2grid_size * kk].S2;
                    u_state[ii + x1grid_size * jj + x1grid_size * x2grid_size * kk].S3  =   u_state[ii + x1grid_size * 1 + x1grid_size * x2grid_size * kk].S3;
                    u_state[ii + x1grid_size * jj + x1grid_size * x2grid_size * kk].tau =   u_state[ii + x1grid_size * 1 + x1grid_size * x2grid_size * kk].tau;
                    
                } else if (jj > x2grid_size - 2) {
                    u_state[ii + x1grid_size * jj + x1grid_size * x2grid_size * kk].D    =   u_state[x1grid_size * x2grid_size * kk + (x2grid_size - 2) * x1grid_size + ii].D;
                    u_state[ii + x1grid_size * jj + x1grid_size * x2grid_size * kk].S1   =   u_state[x1grid_size * x2grid_size * kk + (x2grid_size - 2) * x1grid_size + ii].S1;
                    u_state[ii + x1grid_size * jj + x1grid_size * x2grid_size * kk].S2   =   u_state[x1grid_size * x2grid_size * kk + (x2grid_size - 2) * x1grid_size + ii].S2;
                    u_state[ii + x1grid_size * jj + x1grid_size * x2grid_size * kk].S3   =   u_state[x1grid_size * x2grid_size * kk + (x2grid_size - 2) * x1grid_size + ii].S3;
                    u_state[ii + x1grid_size * jj + x1grid_size * x2grid_size * kk].tau  =   u_state[x1grid_size * x2grid_size * kk + (x2grid_size - 2) * x1grid_size + ii].tau;

                } 
                if (kk < 1){
                    u_state[ii + x1grid_size * jj + x1grid_size * x2grid_size * kk].D   =   u_state[ii + x1grid_size * jj + x1grid_size * x2grid_size * 1].D;
                    u_state[ii + x1grid_size * jj + x1grid_size * x2grid_size * kk].S1  =   u_state[ii + x1grid_size * jj + x1grid_size * x2grid_size * 1].S1;
                    u_state[ii + x1grid_size * jj + x1grid_size * x2grid_size * kk].S2  =   u_state[ii + x1grid_size * jj + x1grid_size * x2grid_size * 1].S2;
                    u_state[ii + x1grid_size * jj + x1grid_size * x2grid_size * kk].S3  =   u_state[ii + x1grid_size * jj + x1grid_size * x2grid_size * 1].S3;
                    u_state[ii + x1grid_size * jj + x1grid_size * x2grid_size * kk].tau =   u_state[ii + x1grid_size * jj + x1grid_size * x2grid_size * 1].tau;
                    
                } else if (kk > x3grid_size - 2) {
                    u_state[ii + x1grid_size * jj + x1grid_size * x2grid_size * kk].D    =   u_state[x1grid_size * x2grid_size * (x3grid_size - 3) + jj * x1grid_size + ii].D;
                    u_state[ii + x1grid_size * jj + x1grid_size * x2grid_size * kk].S1   =   u_state[x1grid_size * x2grid_size * (x3grid_size - 3) + jj * x1grid_size + ii].S1;
                    u_state[ii + x1grid_size * jj + x1grid_size * x2grid_size * kk].S2   =   u_state[x1grid_size * x2grid_size * (x3grid_size - 3) + jj * x1grid_size + ii].S2;
                    u_state[ii + x1grid_size * jj + x1grid_size * x2grid_size * kk].S3   =   u_state[x1grid_size * x2grid_size * (x3grid_size - 3) + jj * x1grid_size + ii].S3;
                    u_state[ii + x1grid_size * jj + x1grid_size * x2grid_size * kk].tau  =   u_state[x1grid_size * x2grid_size * (x3grid_size - 3) + jj * x1grid_size + ii].tau;

                } else {
                    u_state[kk * x1grid_size * x2grid_size + jj * x1grid_size + 0].D               = u_state[kk * x1grid_size * x2grid_size + jj * x1grid_size + 1].D;
                    u_state[kk * x1grid_size * x2grid_size + jj * x1grid_size + x1grid_size - 1].D = u_state[kk * x1grid_size * x2grid_size + jj * x1grid_size + x1grid_size - 2].D;

                    u_state[kk * x1grid_size * x2grid_size + jj * x1grid_size + 0].S1               = - u_state[kk * x1grid_size * x2grid_size + jj * x1grid_size + 1].S1;
                    u_state[kk * x1grid_size * x2grid_size + jj * x1grid_size + x1grid_size - 1].S1 =   u_state[kk * x1grid_size * x2grid_size + jj * x1grid_size + x1grid_size - 2].S1;

                    u_state[kk * x1grid_size * x2grid_size + jj * x1grid_size + 0].S2                = u_state[kk * x1grid_size * x2grid_size + jj * x1grid_size + 1].S2;
                    u_state[kk * x1grid_size * x2grid_size + jj * x1grid_size + x1grid_size - 1].S2  = u_state[kk * x1grid_size * x2grid_size + jj * x1grid_size + x1grid_size - 2].S2;

                    u_state[kk * x1grid_size * x2grid_size + jj * x1grid_size + 0].S3                = u_state[kk * x1grid_size * x2grid_size + jj * x1grid_size + 1].S3;
                    u_state[kk * x1grid_size * x2grid_size + jj * x1grid_size + x1grid_size - 1].S3  = u_state[kk * x1grid_size * x2grid_size + jj * x1grid_size + x1grid_size - 2].S3;

                    u_state[kk * x1grid_size * x2grid_size + jj * x1grid_size + 0].tau               = u_state[kk * x1grid_size * x2grid_size + jj * x1grid_size + 1].tau;
                    u_state[kk * x1grid_size * x2grid_size + jj * x1grid_size + x1grid_size - 1].tau = u_state[kk * x1grid_size * x2grid_size + jj * x1grid_size + x1grid_size - 2].tau;
                }
            }

        } else {
            if((jj < x2grid_size) && (kk < x3grid_size))
            {

                // Fix the ghost zones at the radial boundaries
                u_state[kk * x1grid_size * x2grid_size + jj * x1grid_size +  0].D               = u_state[kk * x1grid_size * x2grid_size + jj * x1grid_size +  3].D;
                u_state[kk * x1grid_size * x2grid_size + jj * x1grid_size +  1].D               = u_state[kk * x1grid_size * x2grid_size + jj * x1grid_size +  2].D;
                u_state[kk * x1grid_size * x2grid_size + jj * x1grid_size +  x1grid_size - 1].D = u_state[kk * x1grid_size * x2grid_size + jj * x1grid_size +  x1grid_size - 3].D;
                u_state[kk * x1grid_size * x2grid_size + jj * x1grid_size +  x1grid_size - 2].D = u_state[kk * x1grid_size * x2grid_size + jj * x1grid_size +  x1grid_size - 3].D;

                u_state[kk * x1grid_size * x2grid_size + jj * x1grid_size + 0].S1               = - u_state[kk * x1grid_size * x2grid_size + jj * x1grid_size + 3].S1;
                u_state[kk * x1grid_size * x2grid_size + jj * x1grid_size + 1].S1               = - u_state[kk * x1grid_size * x2grid_size + jj * x1grid_size + 2].S1;
                u_state[kk * x1grid_size * x2grid_size + jj * x1grid_size + x1grid_size - 1].S1 =   u_state[kk * x1grid_size * x2grid_size + jj * x1grid_size + x1grid_size - 3].S1;
                u_state[kk * x1grid_size * x2grid_size + jj * x1grid_size + x1grid_size - 2].S1 =   u_state[kk * x1grid_size * x2grid_size + jj * x1grid_size + x1grid_size - 3].S1;

                u_state[kk * x1grid_size * x2grid_size + jj * x1grid_size + 0].S2               = u_state[kk * x1grid_size * x2grid_size + jj * x1grid_size + 3].S2;
                u_state[kk * x1grid_size * x2grid_size + jj * x1grid_size + 1].S2               = u_state[kk * x1grid_size * x2grid_size + jj * x1grid_size + 2].S2;
                u_state[kk * x1grid_size * x2grid_size + jj * x1grid_size + x1grid_size - 1].S2 = u_state[kk * x1grid_size * x2grid_size + jj * x1grid_size + x1grid_size - 3].S2;
                u_state[kk * x1grid_size * x2grid_size + jj * x1grid_size + x1grid_size - 2].S2 = u_state[kk * x1grid_size * x2grid_size + jj * x1grid_size + x1grid_size - 3].S2;

                u_state[kk * x1grid_size * x2grid_size + jj * x1grid_size + 0].S3               = u_state[kk * x1grid_size * x2grid_size + jj * x1grid_size + 3].S3;
                u_state[kk * x1grid_size * x2grid_size + jj * x1grid_size + 1].S3               = u_state[kk * x1grid_size * x2grid_size + jj * x1grid_size + 2].S3;
                u_state[kk * x1grid_size * x2grid_size + jj * x1grid_size + x1grid_size - 1].S3 = u_state[kk * x1grid_size * x2grid_size + jj * x1grid_size + x1grid_size - 3].S3;
                u_state[kk * x1grid_size * x2grid_size + jj * x1grid_size + x1grid_size - 2].S3 = u_state[kk * x1grid_size * x2grid_size + jj * x1grid_size + x1grid_size - 3].S3;

                u_state[kk * x1grid_size * x2grid_size + jj * x1grid_size + 0].tau                = u_state[kk * x1grid_size * x2grid_size + jj * x1grid_size + 3].tau;
                u_state[kk * x1grid_size * x2grid_size + jj * x1grid_size + 1].tau                = u_state[kk * x1grid_size * x2grid_size + jj * x1grid_size + 2].tau;
                u_state[kk * x1grid_size * x2grid_size + jj * x1grid_size + x1grid_size - 1].tau  = u_state[kk * x1grid_size * x2grid_size + jj * x1grid_size + x1grid_size - 3].tau;
                u_state[kk * x1grid_size * x2grid_size + jj * x1grid_size + x1grid_size - 2].tau  = u_state[kk * x1grid_size * x2grid_size + jj * x1grid_size + x1grid_size - 3].tau;

                // Fix the ghost zones at the angular boundaries
                
                if (jj < 2){
                    if (ii < x1grid_size){
                        if (jj == 0){
                            u_state[kk * x1grid_size * x2grid_size + jj * x1grid_size + ii].D   =   u_state[kk * x1grid_size * x2grid_size + 3 * x1grid_size + ii].D;
                            u_state[kk * x1grid_size * x2grid_size + jj * x1grid_size + ii].S1  =   u_state[kk * x1grid_size * x2grid_size + 3 * x1grid_size + ii].S1;
                            u_state[kk * x1grid_size * x2grid_size + jj * x1grid_size + ii].S2  =   u_state[kk * x1grid_size * x2grid_size + 3 * x1grid_size + ii].S2;
                            u_state[kk * x1grid_size * x2grid_size + jj * x1grid_size + ii].S3  =   u_state[kk * x1grid_size * x2grid_size + 3 * x1grid_size + ii].S3;
                            u_state[kk * x1grid_size * x2grid_size + jj * x1grid_size + ii].tau =   u_state[kk * x1grid_size * x2grid_size + 3 * x1grid_size + ii].tau;
                        } else {
                            u_state[kk * x1grid_size * x2grid_size + jj * x1grid_size + ii].D    =   u_state[kk * x1grid_size * x2grid_size + 2 * x1grid_size + ii].D;
                            u_state[kk * x1grid_size * x2grid_size + jj * x1grid_size + ii].S1   =   u_state[kk * x1grid_size * x2grid_size + 2 * x1grid_size + ii].S1;
                            u_state[kk * x1grid_size * x2grid_size + jj * x1grid_size + ii].S2   =   u_state[kk * x1grid_size * x2grid_size + 2 * x1grid_size + ii].S2;
                            u_state[kk * x1grid_size * x2grid_size + jj * x1grid_size + ii].S3   =   u_state[kk * x1grid_size * x2grid_size + 2 * x1grid_size + ii].S3;
                            u_state[kk * x1grid_size * x2grid_size + jj * x1grid_size + ii].tau  =   u_state[kk * x1grid_size * x2grid_size + 2 * x1grid_size + ii].tau;
                        }
                    }
                } else if (jj > x2grid_size - 3) {
                    if (ii < x1grid_size){
                        if (jj == x2grid_size - 1){
                            u_state[kk * x1grid_size * x2grid_size + jj * x1grid_size + ii].D   =   u_state[kk * x1grid_size * x2grid_size + (x2grid_size - 4) * x1grid_size + ii].D;
                            u_state[kk * x1grid_size * x2grid_size + jj * x1grid_size + ii].S1  =   u_state[kk * x1grid_size * x2grid_size + (x2grid_size - 4) * x1grid_size + ii].S1;
                            u_state[kk * x1grid_size * x2grid_size + jj * x1grid_size + ii].S2  =   u_state[kk * x1grid_size * x2grid_size + (x2grid_size - 4) * x1grid_size + ii].S2;
                            u_state[kk * x1grid_size * x2grid_size + jj * x1grid_size + ii].S3  =   u_state[kk * x1grid_size * x2grid_size + (x2grid_size - 4) * x1grid_size + ii].S3;
                            u_state[kk * x1grid_size * x2grid_size + jj * x1grid_size + ii].tau =   u_state[kk * x1grid_size * x2grid_size + (x2grid_size - 4) * x1grid_size + ii].tau;
                        } else {
                            u_state[kk * x1grid_size * x2grid_size + jj * x1grid_size + ii].D   =   u_state[kk * x1grid_size * x2grid_size + (x2grid_size - 3) * x1grid_size + ii].D;
                            u_state[kk * x1grid_size * x2grid_size + jj * x1grid_size + ii].S1  =   u_state[kk * x1grid_size * x2grid_size + (x2grid_size - 3) * x1grid_size + ii].S1;
                            u_state[kk * x1grid_size * x2grid_size + jj * x1grid_size + ii].S2  =   u_state[kk * x1grid_size * x2grid_size + (x2grid_size - 3) * x1grid_size + ii].S2;
                            u_state[kk * x1grid_size * x2grid_size + jj * x1grid_size + ii].S3  =   u_state[kk * x1grid_size * x2grid_size + (x2grid_size - 3) * x1grid_size + ii].S3;
                            u_state[kk * x1grid_size * x2grid_size + jj * x1grid_size + ii].tau =   u_state[kk * x1grid_size * x2grid_size + (x2grid_size - 3) * x1grid_size + ii].tau;
                        }
                    }
                }

                if (kk < 2){
                    if (ii < x1grid_size){
                        if (jj == 0){
                            u_state[kk * x1grid_size * x2grid_size + jj * x1grid_size + ii].D   =   u_state[3 * x1grid_size * x2grid_size + jj * x1grid_size + ii].D;
                            u_state[kk * x1grid_size * x2grid_size + jj * x1grid_size + ii].S1  =   u_state[3 * x1grid_size * x2grid_size + jj * x1grid_size + ii].S1;
                            u_state[kk * x1grid_size * x2grid_size + jj * x1grid_size + ii].S2  =   u_state[3 * x1grid_size * x2grid_size + jj * x1grid_size + ii].S2;
                            u_state[kk * x1grid_size * x2grid_size + jj * x1grid_size + ii].S3  =   u_state[3 * x1grid_size * x2grid_size + jj * x1grid_size + ii].S3;
                            u_state[kk * x1grid_size * x2grid_size + jj * x1grid_size + ii].tau =   u_state[3 * x1grid_size * x2grid_size + jj * x1grid_size + ii].tau;
                        } else {
                            u_state[kk * x1grid_size * x2grid_size + jj * x1grid_size + ii].D    =   u_state[2 * x1grid_size * x2grid_size + jj * x1grid_size + ii].D;
                            u_state[kk * x1grid_size * x2grid_size + jj * x1grid_size + ii].S1   =   u_state[2 * x1grid_size * x2grid_size + jj * x1grid_size + ii].S1;
                            u_state[kk * x1grid_size * x2grid_size + jj * x1grid_size + ii].S2   =   u_state[2 * x1grid_size * x2grid_size + jj * x1grid_size + ii].S2;
                            u_state[kk * x1grid_size * x2grid_size + jj * x1grid_size + ii].S3   =   u_state[2 * x1grid_size * x2grid_size + jj * x1grid_size + ii].S3;
                            u_state[kk * x1grid_size * x2grid_size + jj * x1grid_size + ii].tau  =   u_state[2 * x1grid_size * x2grid_size + jj * x1grid_size + ii].tau;
                        }
                    }
                } else if (kk > x3grid_size - 3) {
                    if (ii < x1grid_size){
                        if (kk == x3grid_size - 1){
                            u_state[kk * x1grid_size * x2grid_size + jj * x1grid_size + ii].D   =   u_state[(x3grid_size - 4) * x1grid_size * x2grid_size + jj * x1grid_size + ii].D;
                            u_state[kk * x1grid_size * x2grid_size + jj * x1grid_size + ii].S1  =   u_state[(x3grid_size - 4) * x1grid_size * x2grid_size + jj * x1grid_size + ii].S1;
                            u_state[kk * x1grid_size * x2grid_size + jj * x1grid_size + ii].S2  =   u_state[(x3grid_size - 4) * x1grid_size * x2grid_size + jj * x1grid_size + ii].S2;
                            u_state[kk * x1grid_size * x2grid_size + jj * x1grid_size + ii].S3  =   u_state[(x3grid_size - 4) * x1grid_size * x2grid_size + jj * x1grid_size + ii].S3;
                            u_state[kk * x1grid_size * x2grid_size + jj * x1grid_size + ii].tau =   u_state[(x3grid_size - 4) * x1grid_size * x2grid_size + jj * x1grid_size + ii].tau;
                        } else {
                            u_state[kk * x1grid_size * x2grid_size + jj * x1grid_size + ii].D   =   u_state[(x3grid_size - 3) * x1grid_size * x2grid_size + jj * x1grid_size + ii].D;
                            u_state[kk * x1grid_size * x2grid_size + jj * x1grid_size + ii].S1  =   u_state[(x3grid_size - 3) * x1grid_size * x2grid_size + jj * x1grid_size + ii].S1;
                            u_state[kk * x1grid_size * x2grid_size + jj * x1grid_size + ii].S2  =   u_state[(x3grid_size - 3) * x1grid_size * x2grid_size + jj * x1grid_size + ii].S2;
                            u_state[kk * x1grid_size * x2grid_size + jj * x1grid_size + ii].S3  =   u_state[(x3grid_size - 3) * x1grid_size * x2grid_size + jj * x1grid_size + ii].S3;
                            u_state[kk * x1grid_size * x2grid_size + jj * x1grid_size + ii].tau =   u_state[(x3grid_size - 3) * x1grid_size * x2grid_size + jj * x1grid_size + ii].tau;
                        }
                    }
                }
                
            }

        }
    };
}
