#include "helpers.hip.hpp"
#include "parallel_for.hpp"

//==================================
//              GPU HELPERS
//==================================
namespace simbi{
    void config_ghosts1DGPU(
        const ExecutionPolicy<> p, 
        SRHD *sim, 
        const int grid_size, 
        const bool first_order)
    {
        simbi::parallel_for(p, 0, 1, [=] GPU_LAMBDA (const int gid) {
            sr1d::Conserved *cons = (BuildPlatform == Platform::GPU) ? sim->gpu_cons : sim->cons.data();
            if (first_order){
                cons[0].D = cons[1].D;
                cons[grid_size - 1].D = cons[grid_size - 2].D;

                cons[0].S = - cons[1].S;
                cons[grid_size - 1].S = cons[grid_size - 2].S;

                cons[0].tau = cons[1].tau;
                cons[grid_size - 1].tau = cons[grid_size - 2].tau;
            } else {
                cons[0].D = cons[3].D;
                cons[1].D = cons[2].D;
                cons[grid_size - 1].D = cons[grid_size - 3].D;
                cons[grid_size - 2].D = cons[grid_size - 3].D;

                cons[0].S = - cons[3].S;
                cons[1].S = - cons[2].S;
                cons[grid_size - 1].S = cons[grid_size - 3].S;
                cons[grid_size - 2].S = cons[grid_size - 3].S;

                cons[0].tau = cons[3].tau;
                cons[1].tau = cons[2].tau;
                cons[grid_size - 1].tau = cons[grid_size - 3].tau;
                cons[grid_size - 2].tau = cons[grid_size - 3].tau;
            }
        });
    };
    void config_ghosts2DGPU(
        const ExecutionPolicy<> p,
        SRHD2D *sim, 
        const int x1grid_size, 
        const int x2grid_size, 
        const bool first_order,
        const bool bipolar)
    {
        
        const int extent = (BuildPlatform == Platform::GPU) ? p.gridSize.x * p.blockSize.x * p.gridSize.y * p.blockSize.y : x1grid_size * x2grid_size; 
        simbi::parallel_for(p, 0, extent, [=] GPU_LAMBDA (const int gid) {
            sr2d::Conserved *u_state = (BuildPlatform == Platform::GPU) ? sim->gpu_cons : sim->cons.data();
            const int ii = (BuildPlatform == Platform::GPU) ? blockDim.x * blockIdx.x + threadIdx.x: gid % x1grid_size;
            const int jj = (BuildPlatform == Platform::GPU) ? blockDim.y * blockIdx.y + threadIdx.y: gid / x1grid_size;
            if (first_order){
                if ((jj < x2grid_size) && (ii < x1grid_size))
                {
                    if (jj < 1){
                        u_state[ii + x1grid_size * jj].D   =   u_state[ii + x1grid_size].D;
                        u_state[ii + x1grid_size * jj].S1  =   u_state[ii + x1grid_size].S1;
                        u_state[ii + x1grid_size * jj].S2  =   u_state[ii + x1grid_size].S2;
                        u_state[ii + x1grid_size * jj].tau =   u_state[ii + x1grid_size].tau;
                        
                    } else if (jj > x2grid_size - 2) {
                        u_state[ii + x1grid_size * jj].D    =   u_state[(x2grid_size - 2) * x1grid_size + ii].D;
                        u_state[ii + x1grid_size * jj].S1   =   u_state[(x2grid_size - 2) * x1grid_size + ii].S1;
                        u_state[ii + x1grid_size * jj].S2   =   u_state[(x2grid_size - 2) * x1grid_size + ii].S2;
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
        });
    };
    
    void config_ghosts3DGPU(
        const ExecutionPolicy<> p,
        SRHD3D *sim, 
        const int x1grid_size, 
        const int x2grid_size,
        const int x3grid_size, 
        const bool first_order,
        const bool bipolar)
    {
        const int extent = (BuildPlatform == Platform::GPU) ? 
                            p.gridSize.z * p.gridSize.y * p.gridSize.x * p.blockSize.z * p.blockSize.y * p.blockSize.x 
                            : x1grid_size * x2grid_size * x3grid_size;
        simbi::parallel_for(p, 0, extent, [=] GPU_LAMBDA (const int gid) {
            const int kk = (BuildPlatform == Platform::GPU) ? blockDim.z * blockIdx.z + threadIdx.z : simbi::detail::get_height(gid, x1grid_size, x2grid_size);
            const int jj = (BuildPlatform == Platform::GPU) ? blockDim.y * blockIdx.y + threadIdx.y : simbi::detail::get_column(gid, x1grid_size, x2grid_size, kk);
            const int ii = (BuildPlatform == Platform::GPU) ? blockDim.x * blockIdx.x + threadIdx.x : simbi::detail::get_row(gid, x1grid_size, x2grid_size, kk);
            
            sr3d::Conserved *u_state = (BuildPlatform == Platform::GPU) ? sim->gpu_cons : sim->cons.data();
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

        });
        
    };
}
