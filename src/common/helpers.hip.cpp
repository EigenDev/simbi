#include "common/helpers.hip.hpp"
#include "util/parallel_for.hpp"
#include <thread>
//==================================
//              GPU HELPERS
//==================================
real gpu_theoretical_bw = 1;
namespace simbi{
    //==================================================
    //               1D
    //==================================================
    void config_ghosts1D(
        const ExecutionPolicy<> p, 
        SRHD::conserved_t *cons, 
        const int grid_size, 
        const bool first_order,
        const simbi::BoundaryCondition boundary_condition,
        const sr1d::Conserved * outer_zones)
    {
        simbi::parallel_for(p, 0, 1, [=] GPU_LAMBDA (const int gid) {
            if (first_order){
                cons[0] = cons[1];
                if (outer_zones)
                {
                    cons[grid_size - 1] = outer_zones[0];
                } else {
                    cons[grid_size - 1] = cons[grid_size - 2];
                }
                
                
                switch (boundary_condition)
                {
                case simbi::BoundaryCondition::INFLOW:
                    cons[0] = cons[1];
                    break;
                case simbi::BoundaryCondition::REFLECTING:
                    cons[0]   = cons[1];
                    cons[0].s = - cons[1].s;
                    break;
                default:
                    cons[0] = cons[1];
                    break;
                }
            } else {
                
                if (outer_zones)
                {
                    cons[grid_size - 1] = outer_zones[0];
                    cons[grid_size - 2] = outer_zones[0];
                } else {
                    cons[grid_size - 1] = cons[grid_size - 3];
                    cons[grid_size - 2] = cons[grid_size - 3];
                }
                
                switch (boundary_condition)
                {
                case simbi::BoundaryCondition::INFLOW:
                    cons[0] =     cons[2];
                    cons[1] =     cons[2];
                    cons[0].s = - cons[2].s;
                    cons[1].s = - cons[2].s;
                    break;
                case simbi::BoundaryCondition::REFLECTING:
                    cons[0] =     cons[3];
                    cons[1] =     cons[2];  
                    cons[0].s = - cons[3].s;
                    cons[1].s = - cons[2].s;
                    break;
                default:
                    cons[0] = cons[2];
                    cons[1] = cons[2];
                    break;
                }
            }
        });
    };

    void config_ghosts1D(
        const ExecutionPolicy<> p, 
        Newtonian1D::conserved_t *cons, 
        const int grid_size, 
        const bool first_order,
        const simbi::BoundaryCondition boundary_condition,
        const hydro1d::Conserved *outer_zones)
    {
        simbi::parallel_for(p, 0, 1, [=] GPU_LAMBDA (const int gid) {
            if (first_order){
                cons[0] = cons[1];
                cons[grid_size - 1] = cons[grid_size - 2];
                
                switch (boundary_condition)
                {
                case simbi::BoundaryCondition::INFLOW:
                    cons[0].m =   cons[1].m;
                    break;
                case simbi::BoundaryCondition::REFLECTING:
                    cons[0].m = - cons[1].m;
                    break;
                default:
                    break;
                }

            } else {
                cons[grid_size - 1] = cons[grid_size - 3];
                cons[grid_size - 2] = cons[grid_size - 3];

                switch (boundary_condition)
                {
                case simbi::BoundaryCondition::REFLECTING:
                    cons[0]   =   cons[3];
                    cons[1]   =   cons[2];
                    cons[0].m = - cons[3].m;
                    cons[1].m = - cons[2].m;
                    break;
                case simbi::BoundaryCondition::INFLOW:
                    cons[0]   =   cons[2];
                    cons[1]   =   cons[2];
                    cons[0].m = - cons[2].m;
                    cons[1].m = - cons[2].m;
                    break;
                default:
                    cons[0]   =   cons[2];
                    cons[1]   =   cons[2];
                    break;
                }
            }
        });
    };


    //==============================================
    //                  2D
    //==============================================
    void config_ghosts2D(
        const ExecutionPolicy<> p,
        SRHD2D::conserved_t *cons, 
        const int x1grid_size, 
        const int x2grid_size, 
        const bool first_order,
        const simbi::BoundaryCondition boundary_condition,
        const sr2d::Conserved *outer_zones,
        const bool reflecting_theta)
    {
        const int extent = (BuildPlatform == Platform::GPU) ? p.gridSize.x * p.blockSize.x * p.gridSize.y * p.blockSize.y : x1grid_size * x2grid_size; 
        const int sx     = (col_maj) ? 1 : x1grid_size;
        const int sy     = (col_maj) ? x2grid_size : 1;
        simbi::parallel_for(p, 0, extent, [=] GPU_LAMBDA (const int gid) {
            const int ii = (BuildPlatform == Platform::GPU) ? blockDim.x * blockIdx.x + threadIdx.x: gid % x1grid_size;
            const int jj = (BuildPlatform == Platform::GPU) ? blockDim.y * blockIdx.y + threadIdx.y: gid / x1grid_size;
            if (first_order){
                if(jj < x2grid_size)
                {
                    // Fix the ghost zones at the radial boundaries
                    if (outer_zones) {
                        cons[jj * sx +  (x1grid_size - 1) * sy] = outer_zones[jj];
                    } else {
                        cons[jj * sx +  (x1grid_size - 1) * sy] = cons[jj * sx +  (x1grid_size - 2) * sy];
                    }
                    
                    cons[jj * sx +  0 * sy]   = cons[jj * sx +  1 * sy];
                    switch (boundary_condition)
                    {
                    case simbi::BoundaryCondition::REFLECTING:
                        cons[jj * sx + 0 * sy].s1 = - cons[jj * sx + 1 * sy].s1;
                        break;
                    case simbi::BoundaryCondition::INFLOW:
                        cons[jj * sx + 0 * sy].s1 = - cons[jj * sx + 1 * sy].s1;
                        break;
                    default:
                        cons[jj * sx +  0 * sy]   = cons[jj * sx +  1 * sy];
                        break;
                    }

                    // Fix the ghost zones at the angular boundaries
                    if (ii < x1grid_size){
                        cons[0 * sx + ii * sy]  = cons[1 * sx + ii * sy];
                        cons[(x2grid_size - 1) * sx + ii * sy] = cons[(x2grid_size - 2) * sx + ii * sy];

                        if (reflecting_theta)
                        {
                            cons[(x2grid_size - 1) * sx + ii * sy].s2 = - cons[(x2grid_size - 2) * sx + ii * sy].s2;
                        }
                    }
                }
            } else {
                if(jj < x2grid_size)
                {
                    // Fix the ghost zones at the radial boundaries
                    if (outer_zones) {
                        cons[jj * sx +  (x1grid_size - 1) * sy] = outer_zones[jj];
                        cons[jj * sx +  (x1grid_size - 2) * sy] = outer_zones[jj];
                    } else {
                        cons[jj * sx +  (x1grid_size - 1) * sy] = cons[jj * sx +  (x1grid_size - 3) * sy];
                        cons[jj * sx +  (x1grid_size - 2) * sy] = cons[jj * sx +  (x1grid_size - 3) * sy];
                    }
                    switch (boundary_condition)
                    {
                    case simbi::BoundaryCondition::REFLECTING:
                        cons[jj * sx +  0 * sy]   = cons[jj * sx +  3 * sy];
                        cons[jj * sx +  1 * sy]   = cons[jj * sx +  2 * sy];

                        cons[jj * sx + 0 * sy].s1 = - cons[jj * sx + 3 * sy].s1;
                        cons[jj * sx + 1 * sy].s1 = - cons[jj * sx + 2 * sy].s1;
                        break;
                    case simbi::BoundaryCondition::INFLOW:
                        cons[jj * sx +  0 * sy]   = cons[jj * sx +  2 * sy];
                        cons[jj * sx +  1 * sy]   = cons[jj * sx +  2 * sy];

                        cons[jj * sx + 0 * sy].s1 = - cons[jj * sx + 2 * sy].s1;
                        cons[jj * sx + 1 * sy].s1 = - cons[jj * sx + 2 * sy].s1;
                        break;
                    default:
                        cons[jj * sx +  0 * sy]   = cons[jj * sx +  2 * sy];
                        cons[jj * sx +  1 * sy]   = cons[jj * sx +  2 * sy];
                        break;
                    }

                    // Fix the ghost zones at the angular boundaries
                    if (ii < x1grid_size){
                        cons[0 * sx + (ii + 0) * sy]  = cons[3 * sx + (ii + 0) * sy];
                        cons[1 * sx + (ii + 0) * sy]  = cons[2 * sx + (ii + 0) * sy];

                        cons[(x2grid_size - 1) * sx + ii * sy] = cons[(x2grid_size - 4) * sx + ii * sy];
                        cons[(x2grid_size - 2) * sx + ii * sy] = cons[(x2grid_size - 3) * sx + ii * sy];

                        if (reflecting_theta)
                        {
                            cons[(x2grid_size - 1) * sx + ii * sy].s2 = - cons[(x2grid_size - 4) * sx + ii * sy].s2;
                            cons[(x2grid_size - 2) * sx + ii * sy].s2 = - cons[(x2grid_size - 3) * sx + ii * sy].s2;
                        }
                    }
                }
            }
        });
    };

    void config_ghosts2D(
        const ExecutionPolicy<> p,
        Newtonian2D::conserved_t *cons, 
        const int x1grid_size, 
        const int x2grid_size, 
        const bool first_order,
        const simbi::BoundaryCondition boundary_condition,
        const hydro2d::Conserved *outer_zones,
        const bool reflecting_theta)
    {
        const int extent = (BuildPlatform == Platform::GPU) ? p.gridSize.x * p.blockSize.x * p.gridSize.y * p.blockSize.y : x1grid_size * x2grid_size; 
        simbi::parallel_for(p, 0, extent, [=] GPU_LAMBDA (const int gid) {
            const int ii = (BuildPlatform == Platform::GPU) ? blockDim.x * blockIdx.x + threadIdx.x: gid % x1grid_size;
            const int jj = (BuildPlatform == Platform::GPU) ? blockDim.y * blockIdx.y + threadIdx.y: gid / x1grid_size;

            const int sx = (col_maj) ? 1 : x1grid_size;
            const int sy = (col_maj) ? x2grid_size : 1;
            if (first_order){
                if(jj < x2grid_size)
                {
                    // Fix the ghost zones at the radial boundaries
                    cons[jj * sx +  (x1grid_size - 1) * sy] = cons[jj * sx +  (x1grid_size - 2) * sy];
                    cons[jj * sx +  0 * sy]   = cons[jj * sx +  1 * sy];
                    switch (boundary_condition)
                    {
                    case simbi::BoundaryCondition::REFLECTING:
                        cons[jj * sx + 0 * sy].m1 = - cons[jj * sx + 1 * sy].m1;
                        break;
                    case simbi::BoundaryCondition::INFLOW:
                        cons[jj * sx + 0 * sy].m1 = - cons[jj * sx + 1 * sy].m1;
                        break;
                    default:
                        break;
                    }

                    // Fix the ghost zones at the angular boundaries
                    if (ii < x1grid_size){
                        cons[0 * sx + ii * sy]  = cons[1 * sx + ii * sy];
                        cons[(x2grid_size - 1) * sx + ii * sy] = cons[(x2grid_size - 2) * sx + ii * sy];
                    }
                }

            } else {
                if(jj < x2grid_size)
                {
                    // Fix the ghost zones at the radial boundaries
                    cons[jj * sx +  (x1grid_size - 1) * sy] = cons[jj * sx +  (x1grid_size - 3) * sy];
                    cons[jj * sx +  (x1grid_size - 2) * sy] = cons[jj * sx +  (x1grid_size - 3) * sy];
                    switch (boundary_condition)
                    {
                    case simbi::BoundaryCondition::REFLECTING:
                        cons[jj * sx +  0 * sy]   = cons[jj * sx +  3 * sy];
                        cons[jj * sx +  1 * sy]   = cons[jj * sx +  2 * sy];

                        cons[jj * sx + 0 * sy].m1 = - cons[jj * sx + 3 * sy].m1;
                        cons[jj * sx + 1 * sy].m1 = - cons[jj * sx + 2 * sy].m1;
                        break;
                    case simbi::BoundaryCondition::INFLOW:
                        cons[jj * sx +  0 * sy]   = cons[jj * sx +  2 * sy];
                        cons[jj * sx +  1 * sy]   = cons[jj * sx +  2 * sy];

                        cons[jj * sx + 0 * sy].m1 = - cons[jj * sx + 2 * sy].m1;
                        cons[jj * sx + 1 * sy].m1 = - cons[jj * sx + 2 * sy].m1;
                        break;
                    default:
                        cons[jj * sx +  0 * sy]   = cons[jj * sx +  2 * sy];
                        cons[jj * sx +  1 * sy]   = cons[jj * sx +  2 * sy];
                        break;
                    }

                    // Fix the ghost zones at the angular boundaries
                    if (ii < x1grid_size) {
                        cons[0 * sx + ii * sy]  = cons[3 * sx + ii * sy];
                        cons[1 * sx + ii * sy]  = cons[2 * sx + ii * sy];

                        cons[(x2grid_size - 1) * sx + ii * sy] = cons[(x2grid_size - 4) * sx + ii * sy];
                        cons[(x2grid_size - 2) * sx + ii * sy] = cons[(x2grid_size - 3) * sx + ii * sy];

                        if (reflecting_theta)
                        {
                            cons[(x2grid_size - 1) * sx + ii * sy].m2 = - cons[(x2grid_size - 4) * sx + ii * sy].m2;
                            cons[(x2grid_size - 2) * sx + ii * sy].m2 = - cons[(x2grid_size - 3) * sx + ii * sy].m2;
                        }
                    }
                }
            }
        });
    };








    //============================================
    //                  3D
    //============================================
    
    void config_ghosts3D(
        const ExecutionPolicy<> p,
        SRHD3D::conserved_t *cons, 
        const int x1grid_size, 
        const int x2grid_size,
        const int x3grid_size, 
        const bool first_order,
        const simbi::BoundaryCondition boundary_condition,
        const bool reflecting_theta)
    {
        const int extent = (BuildPlatform == Platform::GPU) ? 
                            p.gridSize.z * p.gridSize.y * p.gridSize.x * p.blockSize.z * p.blockSize.y * p.blockSize.x 
                            : x1grid_size * x2grid_size * x3grid_size;
        simbi::parallel_for(p, 0, extent, [=] GPU_LAMBDA (const int gid) {
            const int kk = (BuildPlatform == Platform::GPU) ? blockDim.z * blockIdx.z + threadIdx.z : simbi::detail::get_height(gid, x1grid_size, x2grid_size);
            const int jj = (BuildPlatform == Platform::GPU) ? blockDim.y * blockIdx.y + threadIdx.y : simbi::detail::get_row(gid, x1grid_size, x2grid_size, kk);
            const int ii = (BuildPlatform == Platform::GPU) ? blockDim.x * blockIdx.x + threadIdx.x : simbi::detail::get_column(gid, x1grid_size, x2grid_size, kk);
            if (first_order){
                if ((jj < x2grid_size) && (ii < x1grid_size) && (kk < x3grid_size))
                {
                    if (jj < 1){
                        cons[ii + x1grid_size * jj + x1grid_size * x2grid_size * kk].d   =   cons[ii + x1grid_size * 1 + x1grid_size * x2grid_size * kk].d;
                        cons[ii + x1grid_size * jj + x1grid_size * x2grid_size * kk].s1  =   cons[ii + x1grid_size * 1 + x1grid_size * x2grid_size * kk].s1;
                        cons[ii + x1grid_size * jj + x1grid_size * x2grid_size * kk].s2  =   cons[ii + x1grid_size * 1 + x1grid_size * x2grid_size * kk].s2;
                        cons[ii + x1grid_size * jj + x1grid_size * x2grid_size * kk].s3  =   cons[ii + x1grid_size * 1 + x1grid_size * x2grid_size * kk].s3;
                        cons[ii + x1grid_size * jj + x1grid_size * x2grid_size * kk].tau =   cons[ii + x1grid_size * 1 + x1grid_size * x2grid_size * kk].tau;
                        
                    } else if (jj > x2grid_size - 2) {
                        cons[ii + x1grid_size * jj + x1grid_size * x2grid_size * kk].d    =   cons[x1grid_size * x2grid_size * kk + (x2grid_size - 2) * x1grid_size + ii].d;
                        cons[ii + x1grid_size * jj + x1grid_size * x2grid_size * kk].s1   =   cons[x1grid_size * x2grid_size * kk + (x2grid_size - 2) * x1grid_size + ii].s1;
                        cons[ii + x1grid_size * jj + x1grid_size * x2grid_size * kk].s2   =   cons[x1grid_size * x2grid_size * kk + (x2grid_size - 2) * x1grid_size + ii].s2;
                        cons[ii + x1grid_size * jj + x1grid_size * x2grid_size * kk].s3   =   cons[x1grid_size * x2grid_size * kk + (x2grid_size - 2) * x1grid_size + ii].s3;
                        cons[ii + x1grid_size * jj + x1grid_size * x2grid_size * kk].tau  =   cons[x1grid_size * x2grid_size * kk + (x2grid_size - 2) * x1grid_size + ii].tau;

                    } 
                    if (kk < 1){
                        cons[ii + x1grid_size * jj + x1grid_size * x2grid_size * kk].d   =   cons[ii + x1grid_size * jj + x1grid_size * x2grid_size * 1].d;
                        cons[ii + x1grid_size * jj + x1grid_size * x2grid_size * kk].s1  =   cons[ii + x1grid_size * jj + x1grid_size * x2grid_size * 1].s1;
                        cons[ii + x1grid_size * jj + x1grid_size * x2grid_size * kk].s2  =   cons[ii + x1grid_size * jj + x1grid_size * x2grid_size * 1].s2;
                        cons[ii + x1grid_size * jj + x1grid_size * x2grid_size * kk].s3  =   cons[ii + x1grid_size * jj + x1grid_size * x2grid_size * 1].s3;
                        cons[ii + x1grid_size * jj + x1grid_size * x2grid_size * kk].tau =   cons[ii + x1grid_size * jj + x1grid_size * x2grid_size * 1].tau;
                        
                    } else if (kk > x3grid_size - 2) {
                        cons[ii + x1grid_size * jj + x1grid_size * x2grid_size * kk].d    =   cons[x1grid_size * x2grid_size * (x3grid_size - 3) + jj * x1grid_size + ii].d;
                        cons[ii + x1grid_size * jj + x1grid_size * x2grid_size * kk].s1   =   cons[x1grid_size * x2grid_size * (x3grid_size - 3) + jj * x1grid_size + ii].s1;
                        cons[ii + x1grid_size * jj + x1grid_size * x2grid_size * kk].s2   =   cons[x1grid_size * x2grid_size * (x3grid_size - 3) + jj * x1grid_size + ii].s2;
                        cons[ii + x1grid_size * jj + x1grid_size * x2grid_size * kk].s3   =   cons[x1grid_size * x2grid_size * (x3grid_size - 3) + jj * x1grid_size + ii].s3;
                        cons[ii + x1grid_size * jj + x1grid_size * x2grid_size * kk].tau  =   cons[x1grid_size * x2grid_size * (x3grid_size - 3) + jj * x1grid_size + ii].tau;

                    } else {
                        cons[kk * x1grid_size * x2grid_size + jj * x1grid_size + 0].d               = cons[kk * x1grid_size * x2grid_size + jj * x1grid_size + 1].d;
                        cons[kk * x1grid_size * x2grid_size + jj * x1grid_size + x1grid_size - 1].d = cons[kk * x1grid_size * x2grid_size + jj * x1grid_size + x1grid_size - 2].d;

                        cons[kk * x1grid_size * x2grid_size + jj * x1grid_size + 0].s1               = - cons[kk * x1grid_size * x2grid_size + jj * x1grid_size + 1].s1;
                        cons[kk * x1grid_size * x2grid_size + jj * x1grid_size + x1grid_size - 1].s1 =   cons[kk * x1grid_size * x2grid_size + jj * x1grid_size + x1grid_size - 2].s1;

                        cons[kk * x1grid_size * x2grid_size + jj * x1grid_size + 0].s2                = cons[kk * x1grid_size * x2grid_size + jj * x1grid_size + 1].s2;
                        cons[kk * x1grid_size * x2grid_size + jj * x1grid_size + x1grid_size - 1].s2  = cons[kk * x1grid_size * x2grid_size + jj * x1grid_size + x1grid_size - 2].s2;

                        cons[kk * x1grid_size * x2grid_size + jj * x1grid_size + 0].s3                = cons[kk * x1grid_size * x2grid_size + jj * x1grid_size + 1].s3;
                        cons[kk * x1grid_size * x2grid_size + jj * x1grid_size + x1grid_size - 1].s3  = cons[kk * x1grid_size * x2grid_size + jj * x1grid_size + x1grid_size - 2].s3;

                        cons[kk * x1grid_size * x2grid_size + jj * x1grid_size + 0].tau               = cons[kk * x1grid_size * x2grid_size + jj * x1grid_size + 1].tau;
                        cons[kk * x1grid_size * x2grid_size + jj * x1grid_size + x1grid_size - 1].tau = cons[kk * x1grid_size * x2grid_size + jj * x1grid_size + x1grid_size - 2].tau;
                    }
                }

            } else {
                if((jj < x2grid_size) && (kk < x3grid_size))
                {

                    // Fix the ghost zones at the radial boundaries
                    cons[kk * x1grid_size * x2grid_size + jj * x1grid_size +  0].d               = cons[kk * x1grid_size * x2grid_size + jj * x1grid_size +  3].d;
                    cons[kk * x1grid_size * x2grid_size + jj * x1grid_size +  1].d               = cons[kk * x1grid_size * x2grid_size + jj * x1grid_size +  2].d;
                    cons[kk * x1grid_size * x2grid_size + jj * x1grid_size +  x1grid_size - 1].d = cons[kk * x1grid_size * x2grid_size + jj * x1grid_size +  x1grid_size - 3].d;
                    cons[kk * x1grid_size * x2grid_size + jj * x1grid_size +  x1grid_size - 2].d = cons[kk * x1grid_size * x2grid_size + jj * x1grid_size +  x1grid_size - 3].d;

                    cons[kk * x1grid_size * x2grid_size + jj * x1grid_size + 0].s1               = - cons[kk * x1grid_size * x2grid_size + jj * x1grid_size + 3].s1;
                    cons[kk * x1grid_size * x2grid_size + jj * x1grid_size + 1].s1               = - cons[kk * x1grid_size * x2grid_size + jj * x1grid_size + 2].s1;
                    cons[kk * x1grid_size * x2grid_size + jj * x1grid_size + x1grid_size - 1].s1 =   cons[kk * x1grid_size * x2grid_size + jj * x1grid_size + x1grid_size - 3].s1;
                    cons[kk * x1grid_size * x2grid_size + jj * x1grid_size + x1grid_size - 2].s1 =   cons[kk * x1grid_size * x2grid_size + jj * x1grid_size + x1grid_size - 3].s1;

                    cons[kk * x1grid_size * x2grid_size + jj * x1grid_size + 0].s2               = cons[kk * x1grid_size * x2grid_size + jj * x1grid_size + 3].s2;
                    cons[kk * x1grid_size * x2grid_size + jj * x1grid_size + 1].s2               = cons[kk * x1grid_size * x2grid_size + jj * x1grid_size + 2].s2;
                    cons[kk * x1grid_size * x2grid_size + jj * x1grid_size + x1grid_size - 1].s2 = cons[kk * x1grid_size * x2grid_size + jj * x1grid_size + x1grid_size - 3].s2;
                    cons[kk * x1grid_size * x2grid_size + jj * x1grid_size + x1grid_size - 2].s2 = cons[kk * x1grid_size * x2grid_size + jj * x1grid_size + x1grid_size - 3].s2;

                    cons[kk * x1grid_size * x2grid_size + jj * x1grid_size + 0].s3               = cons[kk * x1grid_size * x2grid_size + jj * x1grid_size + 3].s3;
                    cons[kk * x1grid_size * x2grid_size + jj * x1grid_size + 1].s3               = cons[kk * x1grid_size * x2grid_size + jj * x1grid_size + 2].s3;
                    cons[kk * x1grid_size * x2grid_size + jj * x1grid_size + x1grid_size - 1].s3 = cons[kk * x1grid_size * x2grid_size + jj * x1grid_size + x1grid_size - 3].s3;
                    cons[kk * x1grid_size * x2grid_size + jj * x1grid_size + x1grid_size - 2].s3 = cons[kk * x1grid_size * x2grid_size + jj * x1grid_size + x1grid_size - 3].s3;

                    cons[kk * x1grid_size * x2grid_size + jj * x1grid_size + 0].tau                = cons[kk * x1grid_size * x2grid_size + jj * x1grid_size + 3].tau;
                    cons[kk * x1grid_size * x2grid_size + jj * x1grid_size + 1].tau                = cons[kk * x1grid_size * x2grid_size + jj * x1grid_size + 2].tau;
                    cons[kk * x1grid_size * x2grid_size + jj * x1grid_size + x1grid_size - 1].tau  = cons[kk * x1grid_size * x2grid_size + jj * x1grid_size + x1grid_size - 3].tau;
                    cons[kk * x1grid_size * x2grid_size + jj * x1grid_size + x1grid_size - 2].tau  = cons[kk * x1grid_size * x2grid_size + jj * x1grid_size + x1grid_size - 3].tau;

                    // Fix the ghost zones at the angular boundaries
                    
                    if (jj < 2){
                        if (ii < x1grid_size){
                            if (jj == 0){
                                cons[kk * x1grid_size * x2grid_size + jj * x1grid_size + ii].d   =   cons[kk * x1grid_size * x2grid_size + 3 * x1grid_size + ii].d;
                                cons[kk * x1grid_size * x2grid_size + jj * x1grid_size + ii].s1  =   cons[kk * x1grid_size * x2grid_size + 3 * x1grid_size + ii].s1;
                                cons[kk * x1grid_size * x2grid_size + jj * x1grid_size + ii].s2  =   cons[kk * x1grid_size * x2grid_size + 3 * x1grid_size + ii].s2;
                                cons[kk * x1grid_size * x2grid_size + jj * x1grid_size + ii].s3  =   cons[kk * x1grid_size * x2grid_size + 3 * x1grid_size + ii].s3;
                                cons[kk * x1grid_size * x2grid_size + jj * x1grid_size + ii].tau =   cons[kk * x1grid_size * x2grid_size + 3 * x1grid_size + ii].tau;
                            } else {
                                cons[kk * x1grid_size * x2grid_size + jj * x1grid_size + ii].d    =   cons[kk * x1grid_size * x2grid_size + 2 * x1grid_size + ii].d;
                                cons[kk * x1grid_size * x2grid_size + jj * x1grid_size + ii].s1   =   cons[kk * x1grid_size * x2grid_size + 2 * x1grid_size + ii].s1;
                                cons[kk * x1grid_size * x2grid_size + jj * x1grid_size + ii].s2   =   cons[kk * x1grid_size * x2grid_size + 2 * x1grid_size + ii].s2;
                                cons[kk * x1grid_size * x2grid_size + jj * x1grid_size + ii].s3   =   cons[kk * x1grid_size * x2grid_size + 2 * x1grid_size + ii].s3;
                                cons[kk * x1grid_size * x2grid_size + jj * x1grid_size + ii].tau  =   cons[kk * x1grid_size * x2grid_size + 2 * x1grid_size + ii].tau;
                            }
                        }
                    } else if (jj > x2grid_size - 3) {
                        if (ii < x1grid_size){
                            if (jj == x2grid_size - 1){
                                cons[kk * x1grid_size * x2grid_size + jj * x1grid_size + ii].d   =   cons[kk * x1grid_size * x2grid_size + (x2grid_size - 4) * x1grid_size + ii].d;
                                cons[kk * x1grid_size * x2grid_size + jj * x1grid_size + ii].s1  =   cons[kk * x1grid_size * x2grid_size + (x2grid_size - 4) * x1grid_size + ii].s1;
                                cons[kk * x1grid_size * x2grid_size + jj * x1grid_size + ii].s2  =   cons[kk * x1grid_size * x2grid_size + (x2grid_size - 4) * x1grid_size + ii].s2;
                                cons[kk * x1grid_size * x2grid_size + jj * x1grid_size + ii].s3  =   cons[kk * x1grid_size * x2grid_size + (x2grid_size - 4) * x1grid_size + ii].s3;
                                cons[kk * x1grid_size * x2grid_size + jj * x1grid_size + ii].tau =   cons[kk * x1grid_size * x2grid_size + (x2grid_size - 4) * x1grid_size + ii].tau;
                            } else {
                                cons[kk * x1grid_size * x2grid_size + jj * x1grid_size + ii].d   =   cons[kk * x1grid_size * x2grid_size + (x2grid_size - 3) * x1grid_size + ii].d;
                                cons[kk * x1grid_size * x2grid_size + jj * x1grid_size + ii].s1  =   cons[kk * x1grid_size * x2grid_size + (x2grid_size - 3) * x1grid_size + ii].s1;
                                cons[kk * x1grid_size * x2grid_size + jj * x1grid_size + ii].s2  =   cons[kk * x1grid_size * x2grid_size + (x2grid_size - 3) * x1grid_size + ii].s2;
                                cons[kk * x1grid_size * x2grid_size + jj * x1grid_size + ii].s3  =   cons[kk * x1grid_size * x2grid_size + (x2grid_size - 3) * x1grid_size + ii].s3;
                                cons[kk * x1grid_size * x2grid_size + jj * x1grid_size + ii].tau =   cons[kk * x1grid_size * x2grid_size + (x2grid_size - 3) * x1grid_size + ii].tau;
                            }
                        }
                    }

                    if (kk < 2){
                        if (ii < x1grid_size){
                            if (jj == 0){
                                cons[kk * x1grid_size * x2grid_size + jj * x1grid_size + ii].d   =   cons[3 * x1grid_size * x2grid_size + jj * x1grid_size + ii].d;
                                cons[kk * x1grid_size * x2grid_size + jj * x1grid_size + ii].s1  =   cons[3 * x1grid_size * x2grid_size + jj * x1grid_size + ii].s1;
                                cons[kk * x1grid_size * x2grid_size + jj * x1grid_size + ii].s2  =   cons[3 * x1grid_size * x2grid_size + jj * x1grid_size + ii].s2;
                                cons[kk * x1grid_size * x2grid_size + jj * x1grid_size + ii].s3  =   cons[3 * x1grid_size * x2grid_size + jj * x1grid_size + ii].s3;
                                cons[kk * x1grid_size * x2grid_size + jj * x1grid_size + ii].tau =   cons[3 * x1grid_size * x2grid_size + jj * x1grid_size + ii].tau;
                            } else {
                                cons[kk * x1grid_size * x2grid_size + jj * x1grid_size + ii].d    =   cons[2 * x1grid_size * x2grid_size + jj * x1grid_size + ii].d;
                                cons[kk * x1grid_size * x2grid_size + jj * x1grid_size + ii].s1   =   cons[2 * x1grid_size * x2grid_size + jj * x1grid_size + ii].s1;
                                cons[kk * x1grid_size * x2grid_size + jj * x1grid_size + ii].s2   =   cons[2 * x1grid_size * x2grid_size + jj * x1grid_size + ii].s2;
                                cons[kk * x1grid_size * x2grid_size + jj * x1grid_size + ii].s3   =   cons[2 * x1grid_size * x2grid_size + jj * x1grid_size + ii].s3;
                                cons[kk * x1grid_size * x2grid_size + jj * x1grid_size + ii].tau  =   cons[2 * x1grid_size * x2grid_size + jj * x1grid_size + ii].tau;
                            }
                        }
                    } else if (kk > x3grid_size - 3) {
                        if (ii < x1grid_size){
                            if (kk == x3grid_size - 1){
                                cons[kk * x1grid_size * x2grid_size + jj * x1grid_size + ii].d   =   cons[(x3grid_size - 4) * x1grid_size * x2grid_size + jj * x1grid_size + ii].d;
                                cons[kk * x1grid_size * x2grid_size + jj * x1grid_size + ii].s1  =   cons[(x3grid_size - 4) * x1grid_size * x2grid_size + jj * x1grid_size + ii].s1;
                                cons[kk * x1grid_size * x2grid_size + jj * x1grid_size + ii].s2  =   cons[(x3grid_size - 4) * x1grid_size * x2grid_size + jj * x1grid_size + ii].s2;
                                cons[kk * x1grid_size * x2grid_size + jj * x1grid_size + ii].s3  =   cons[(x3grid_size - 4) * x1grid_size * x2grid_size + jj * x1grid_size + ii].s3;
                                cons[kk * x1grid_size * x2grid_size + jj * x1grid_size + ii].tau =   cons[(x3grid_size - 4) * x1grid_size * x2grid_size + jj * x1grid_size + ii].tau;
                            } else {
                                cons[kk * x1grid_size * x2grid_size + jj * x1grid_size + ii].d   =   cons[(x3grid_size - 3) * x1grid_size * x2grid_size + jj * x1grid_size + ii].d;
                                cons[kk * x1grid_size * x2grid_size + jj * x1grid_size + ii].s1  =   cons[(x3grid_size - 3) * x1grid_size * x2grid_size + jj * x1grid_size + ii].s1;
                                cons[kk * x1grid_size * x2grid_size + jj * x1grid_size + ii].s2  =   cons[(x3grid_size - 3) * x1grid_size * x2grid_size + jj * x1grid_size + ii].s2;
                                cons[kk * x1grid_size * x2grid_size + jj * x1grid_size + ii].s3  =   cons[(x3grid_size - 3) * x1grid_size * x2grid_size + jj * x1grid_size + ii].s3;
                                cons[kk * x1grid_size * x2grid_size + jj * x1grid_size + ii].tau =   cons[(x3grid_size - 3) * x1grid_size * x2grid_size + jj * x1grid_size + ii].tau;
                            }
                        }
                    }
                    
                }

            }

        });
        
    };

    void anyDisplayProps()
    {
        // Adapted from: https://stackoverflow.com/questions/5689028/how-to-get-card-specs-programmatically-in-cuda
        #if GPU_CODE 
            const int kb = 1024;
            const int mb = kb * kb;
            int devCount;
            anyGpuGetDeviceCount(&devCount);
            std::cout << std::string(80, '=')  << "\n";
            std::cout << "GPU Device(s): " << std::endl << std::endl;

            for(int i = 0; i < devCount; ++i)
            {
                anyGpuProp_t props;
                anyGpuGetDeviceProperties(&props, i);
                std::cout << "  Device number:   " << i << std::endl;
                std::cout << "  Device name:     " << props.name << ": " << props.major << "." << props.minor << std::endl;
                std::cout << "  Global memory:   " << props.totalGlobalMem / mb << "mb" << std::endl;
                std::cout << "  Shared memory:   " << props.sharedMemPerBlock / kb << "kb" << std::endl;
                std::cout << "  Constant memory: " << props.totalConstMem / kb << "kb" << std::endl;
                std::cout << "  Block registers: " << props.regsPerBlock << std::endl << std::endl;

                std::cout << "  Warp size:         " << props.warpSize << std::endl;
                std::cout << "  Threads per block: " << props.maxThreadsPerBlock << std::endl;
                std::cout << "  Max block dimensions: [ " << props.maxThreadsDim[0] << ", " << props.maxThreadsDim[1]  << ", " << props.maxThreadsDim[2] << " ]" << std::endl;
                std::cout << "  Max grid dimensions:  [ " << props.maxGridSize[0] << ", " << props.maxGridSize[1]  << ", " << props.maxGridSize[2] << " ]" << std::endl;
                std::cout << "  Memory Clock Rate (KHz): " <<  props.memoryClockRate << std::endl;
                std::cout << "  Memory Bus Width (bits): " <<  props.memoryBusWidth << std::endl;
                std::cout << "  Peak Memory Bandwidth (GB/s): " << 2.0*props.memoryClockRate*(props.memoryBusWidth/8)/1.0e6 << std::endl;
                std::cout << std::endl;
                gpu_theoretical_bw = 2.0*props.memoryClockRate*(props.memoryBusWidth/8)/1.0e6;
            }
        #else 
        const auto processor_count = std::thread::hardware_concurrency();
        std::cout << std::string(80, '=')  << "\n";
        std::cout << "CPU Compute Core(s): " << processor_count << std::endl;
        #endif
    }
}
