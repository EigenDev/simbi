#include "helpers.hip.hpp"
#include "util/parallel_for.hpp"

//==================================
//              GPU HELPERS
//==================================
namespace simbi{
    //==================================================
    //               1D
    //==================================================
    void config_ghosts1D(
        const ExecutionPolicy<> p, 
        SRHD *sim, 
        const int grid_size, 
        const bool first_order,
        const simbi::BoundaryCondition boundary_condition,
        const sr1d::Conserved * outer_zones)
    {
        simbi::parallel_for(p, 0, 1, [=] GPU_LAMBDA (const int gid) {
            #if GPU_CODE
            sr1d::Conserved *cons = sim->gpu_cons;
            #else 
            sr1d::Conserved *cons = sim->cons.data();
            #endif
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
        Newtonian1D *sim, 
        const int grid_size, 
        const bool first_order,
        const simbi::BoundaryCondition boundary_condition)
    {
        simbi::parallel_for(p, 0, 1, [=] GPU_LAMBDA (const int gid) {
            #if GPU_CODE
            hydro1d::Conserved *cons = sim->gpu_cons;
            #else 
            hydro1d::Conserved *cons = sim->cons.data();
            #endif
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
        SRHD2D *sim, 
        const int x1grid_size, 
        const int x2grid_size, 
        const bool first_order,
        const simbi::BoundaryCondition boundary_condition,
        const bool bipolar)
    {
        
        const int extent = (BuildPlatform == Platform::GPU) ? p.gridSize.x * p.blockSize.x * p.gridSize.y * p.blockSize.y : x1grid_size * x2grid_size; 
        simbi::parallel_for(p, 0, extent, [=] GPU_LAMBDA (const int gid) {
            const int ii = (BuildPlatform == Platform::GPU) ? blockDim.x * blockIdx.x + threadIdx.x: gid % x1grid_size;
            const int jj = (BuildPlatform == Platform::GPU) ? blockDim.y * blockIdx.y + threadIdx.y: gid / x1grid_size;

            const int sx = (col_maj) ? 1 : x1grid_size;
            const int sy = (col_maj) ? x2grid_size : 1;

            #if GPU_CODE
            sr2d::Conserved *u_state = sim->gpu_cons;
            #else 
            sr2d::Conserved *u_state = sim->cons.data();
            #endif 
            if (first_order){
                if(jj < x2grid_size)
                {
                    // Fix the ghost zones at the radial boundaries
                    u_state[jj * sx +  (x1grid_size - 1) * sy] = u_state[jj * sx +  (x1grid_size - 2) * sy];
                    u_state[jj * sx +  0 * sy]   = u_state[jj * sx +  1 * sy];
                    switch (boundary_condition)
                    {
                    case simbi::BoundaryCondition::REFLECTING:
                        u_state[jj * sx + 0 * sy].s1 = - u_state[jj * sx + 1 * sy].s1;
                        break;
                    case simbi::BoundaryCondition::INFLOW:
                        u_state[jj * sx + 0 * sy].s1 = - u_state[jj * sx + 1 * sy].s1;
                        break;
                    default:
                        break;
                    }

                    // Fix the ghost zones at the angular boundaries
                    if (ii < x1grid_size){
                        u_state[0 * sx + ii * sy]  = u_state[1 * sx + ii * sy];
                        u_state[(x2grid_size - 1) * sx + ii * sy] = u_state[(x2grid_size - 2) * sx + ii * sy];

                        if (bipolar)
                        {
                            u_state[(x2grid_size - 1) * sx + ii * sy].s2 = - u_state[(x2grid_size - 2) * sx + ii * sy].s2;
                        }
                    }
                }

            } else {
                if(jj < x2grid_size)
                {
                    // Fix the ghost zones at the radial boundaries
                    u_state[jj * sx +  (x1grid_size - 1) * sy] = u_state[jj * sx +  (x1grid_size - 3) * sy];
                    u_state[jj * sx +  (x1grid_size - 2) * sy] = u_state[jj * sx +  (x1grid_size - 3) * sy];
                    switch (boundary_condition)
                    {
                    case simbi::BoundaryCondition::REFLECTING:
                        u_state[jj * sx +  0 * sy]   = u_state[jj * sx +  3 * sy];
                        u_state[jj * sx +  1 * sy]   = u_state[jj * sx +  2 * sy];

                        u_state[jj * sx + 0 * sy].s1 = - u_state[jj * sx + 3 * sy].s1;
                        u_state[jj * sx + 1 * sy].s1 = - u_state[jj * sx + 2 * sy].s1;
                        break;
                    case simbi::BoundaryCondition::INFLOW:
                        u_state[jj * sx +  0 * sy]   = u_state[jj * sx +  2 * sy];
                        u_state[jj * sx +  1 * sy]   = u_state[jj * sx +  2 * sy];

                        u_state[jj * sx + 0 * sy].s1 = - u_state[jj * sx + 2 * sy].s1;
                        u_state[jj * sx + 1 * sy].s1 = - u_state[jj * sx + 2 * sy].s1;
                        break;
                    default:
                        u_state[jj * sx +  0 * sy]   = u_state[jj * sx +  2 * sy];
                        u_state[jj * sx +  1 * sy]   = u_state[jj * sx +  2 * sy];
                        break;
                    }

                    // Fix the ghost zones at the angular boundaries
                    if (ii < x1grid_size){
                        u_state[0 * sx + ii * sy]  = u_state[3 * sx + ii * sy];
                        u_state[1 * sx + ii * sy]  = u_state[2 * sx + ii * sy];

                        u_state[(x2grid_size - 1) * sx + ii * sy] = u_state[(x2grid_size - 4) * sx + ii * sy];
                        u_state[(x2grid_size - 2) * sx + ii * sy] = u_state[(x2grid_size - 3) * sx + ii * sy];

                        if (bipolar)
                        {
                            u_state[(x2grid_size - 1) * sx + ii * sy].s2 = - u_state[(x2grid_size - 4) * sx + ii * sy].s2;
                            u_state[(x2grid_size - 2) * sx + ii * sy].s2 = - u_state[(x2grid_size - 3) * sx + ii * sy].s2;
                        }
                    }
                }
            }
        });
    };

    void config_ghosts2D(
        const ExecutionPolicy<> p,
        Newtonian2D *sim, 
        const int x1grid_size, 
        const int x2grid_size, 
        const bool first_order,
        const simbi::BoundaryCondition boundary_condition,
        const bool bipolar)
    {
        
        const int extent = (BuildPlatform == Platform::GPU) ? p.gridSize.x * p.blockSize.x * p.gridSize.y * p.blockSize.y : x1grid_size * x2grid_size; 
        simbi::parallel_for(p, 0, extent, [=] GPU_LAMBDA (const int gid) {
            const int ii = (BuildPlatform == Platform::GPU) ? blockDim.x * blockIdx.x + threadIdx.x: gid % x1grid_size;
            const int jj = (BuildPlatform == Platform::GPU) ? blockDim.y * blockIdx.y + threadIdx.y: gid / x1grid_size;

            const int sx = (col_maj) ? 1 : x1grid_size;
            const int sy = (col_maj) ? x2grid_size : 1;

            #if GPU_CODE
            hydro2d::Conserved *u_state = sim->gpu_cons;
            #else 
            hydro2d::Conserved *u_state = sim->cons.data();
            #endif 
            if (first_order){
                if(jj < x2grid_size)
                {
                    // Fix the ghost zones at the radial boundaries
                    u_state[jj * sx +  (x1grid_size - 1) * sy] = u_state[jj * sx +  (x1grid_size - 2) * sy];
                    u_state[jj * sx +  0 * sy]   = u_state[jj * sx +  1 * sy];
                    switch (boundary_condition)
                    {
                    case simbi::BoundaryCondition::REFLECTING:
                        u_state[jj * sx + 0 * sy].m1 = - u_state[jj * sx + 1 * sy].m1;
                        break;
                    case simbi::BoundaryCondition::INFLOW:
                        u_state[jj * sx + 0 * sy].m1 = - u_state[jj * sx + 1 * sy].m1;
                        break;
                    default:
                        break;
                    }

                    // Fix the ghost zones at the angular boundaries
                    if (ii < x1grid_size){
                        u_state[0 * sx + ii * sy]  = u_state[1 * sx + ii * sy];
                        u_state[(x2grid_size - 1) * sx + ii * sy] = u_state[(x2grid_size - 2) * sx + ii * sy];
                    }
                }

            } else {
                if(jj < x2grid_size)
                {
                    // Fix the ghost zones at the radial boundaries
                    u_state[jj * sx +  (x1grid_size - 1) * sy] = u_state[jj * sx +  (x1grid_size - 3) * sy];
                    u_state[jj * sx +  (x1grid_size - 2) * sy] = u_state[jj * sx +  (x1grid_size - 3) * sy];
                    switch (boundary_condition)
                    {
                    case simbi::BoundaryCondition::REFLECTING:
                        u_state[jj * sx +  0 * sy]   = u_state[jj * sx +  3 * sy];
                        u_state[jj * sx +  1 * sy]   = u_state[jj * sx +  2 * sy];

                        u_state[jj * sx + 0 * sy].m1 = - u_state[jj * sx + 3 * sy].m1;
                        u_state[jj * sx + 1 * sy].m1 = - u_state[jj * sx + 2 * sy].m1;
                        break;
                    case simbi::BoundaryCondition::INFLOW:
                        u_state[jj * sx +  0 * sy]   = u_state[jj * sx +  2 * sy];
                        u_state[jj * sx +  1 * sy]   = u_state[jj * sx +  2 * sy];

                        u_state[jj * sx + 0 * sy].m1 = - u_state[jj * sx + 2 * sy].m1;
                        u_state[jj * sx + 1 * sy].m1 = - u_state[jj * sx + 2 * sy].m1;
                        break;
                    default:
                        u_state[jj * sx +  0 * sy]   = u_state[jj * sx +  2 * sy];
                        u_state[jj * sx +  1 * sy]   = u_state[jj * sx +  2 * sy];
                        break;
                    }

                    // Fix the ghost zones at the angular boundaries
                    if (ii < x1grid_size){
                        u_state[0 * sx + ii * sy]  = u_state[3 * sx + ii * sy];
                        u_state[1 * sx + ii * sy]  = u_state[2 * sx + ii * sy];

                        u_state[(x2grid_size - 1) * sx + ii * sy] = u_state[(x2grid_size - 4) * sx + ii * sy];
                        u_state[(x2grid_size - 2) * sx + ii * sy] = u_state[(x2grid_size - 3) * sx + ii * sy];

                        if (bipolar)
                        {
                            u_state[(x2grid_size - 1) * sx + ii * sy].m2 = - u_state[(x2grid_size - 4) * sx + ii * sy].m2;
                            u_state[(x2grid_size - 2) * sx + ii * sy].m2 = - u_state[(x2grid_size - 3) * sx + ii * sy].m2;
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
        SRHD3D *sim, 
        const int x1grid_size, 
        const int x2grid_size,
        const int x3grid_size, 
        const bool first_order,
        const simbi::BoundaryCondition boundary_condition,
        const bool bipolar)
    {
        const int extent = (BuildPlatform == Platform::GPU) ? 
                            p.gridSize.z * p.gridSize.y * p.gridSize.x * p.blockSize.z * p.blockSize.y * p.blockSize.x 
                            : x1grid_size * x2grid_size * x3grid_size;
        simbi::parallel_for(p, 0, extent, [=] GPU_LAMBDA (const int gid) {
            const int kk = (BuildPlatform == Platform::GPU) ? blockDim.z * blockIdx.z + threadIdx.z : simbi::detail::get_height(gid, x1grid_size, x2grid_size);
            const int jj = (BuildPlatform == Platform::GPU) ? blockDim.y * blockIdx.y + threadIdx.y : simbi::detail::get_row(gid, x1grid_size, x2grid_size, kk);
            const int ii = (BuildPlatform == Platform::GPU) ? blockDim.x * blockIdx.x + threadIdx.x : simbi::detail::get_column(gid, x1grid_size, x2grid_size, kk);

            #if GPU_CODE
            sr3d::Conserved *u_state = sim->gpu_cons;
            #else
            sr3d::Conserved *u_state = sim->cons.data();
            #endif
            if (first_order){
                if ((jj < x2grid_size) && (ii < x1grid_size) && (kk < x3grid_size))
                {
                    if (jj < 1){
                        u_state[ii + x1grid_size * jj + x1grid_size * x2grid_size * kk].d   =   u_state[ii + x1grid_size * 1 + x1grid_size * x2grid_size * kk].d;
                        u_state[ii + x1grid_size * jj + x1grid_size * x2grid_size * kk].s1  =   u_state[ii + x1grid_size * 1 + x1grid_size * x2grid_size * kk].s1;
                        u_state[ii + x1grid_size * jj + x1grid_size * x2grid_size * kk].s2  =   u_state[ii + x1grid_size * 1 + x1grid_size * x2grid_size * kk].s2;
                        u_state[ii + x1grid_size * jj + x1grid_size * x2grid_size * kk].s3  =   u_state[ii + x1grid_size * 1 + x1grid_size * x2grid_size * kk].s3;
                        u_state[ii + x1grid_size * jj + x1grid_size * x2grid_size * kk].tau =   u_state[ii + x1grid_size * 1 + x1grid_size * x2grid_size * kk].tau;
                        
                    } else if (jj > x2grid_size - 2) {
                        u_state[ii + x1grid_size * jj + x1grid_size * x2grid_size * kk].d    =   u_state[x1grid_size * x2grid_size * kk + (x2grid_size - 2) * x1grid_size + ii].d;
                        u_state[ii + x1grid_size * jj + x1grid_size * x2grid_size * kk].s1   =   u_state[x1grid_size * x2grid_size * kk + (x2grid_size - 2) * x1grid_size + ii].s1;
                        u_state[ii + x1grid_size * jj + x1grid_size * x2grid_size * kk].s2   =   u_state[x1grid_size * x2grid_size * kk + (x2grid_size - 2) * x1grid_size + ii].s2;
                        u_state[ii + x1grid_size * jj + x1grid_size * x2grid_size * kk].s3   =   u_state[x1grid_size * x2grid_size * kk + (x2grid_size - 2) * x1grid_size + ii].s3;
                        u_state[ii + x1grid_size * jj + x1grid_size * x2grid_size * kk].tau  =   u_state[x1grid_size * x2grid_size * kk + (x2grid_size - 2) * x1grid_size + ii].tau;

                    } 
                    if (kk < 1){
                        u_state[ii + x1grid_size * jj + x1grid_size * x2grid_size * kk].d   =   u_state[ii + x1grid_size * jj + x1grid_size * x2grid_size * 1].d;
                        u_state[ii + x1grid_size * jj + x1grid_size * x2grid_size * kk].s1  =   u_state[ii + x1grid_size * jj + x1grid_size * x2grid_size * 1].s1;
                        u_state[ii + x1grid_size * jj + x1grid_size * x2grid_size * kk].s2  =   u_state[ii + x1grid_size * jj + x1grid_size * x2grid_size * 1].s2;
                        u_state[ii + x1grid_size * jj + x1grid_size * x2grid_size * kk].s3  =   u_state[ii + x1grid_size * jj + x1grid_size * x2grid_size * 1].s3;
                        u_state[ii + x1grid_size * jj + x1grid_size * x2grid_size * kk].tau =   u_state[ii + x1grid_size * jj + x1grid_size * x2grid_size * 1].tau;
                        
                    } else if (kk > x3grid_size - 2) {
                        u_state[ii + x1grid_size * jj + x1grid_size * x2grid_size * kk].d    =   u_state[x1grid_size * x2grid_size * (x3grid_size - 3) + jj * x1grid_size + ii].d;
                        u_state[ii + x1grid_size * jj + x1grid_size * x2grid_size * kk].s1   =   u_state[x1grid_size * x2grid_size * (x3grid_size - 3) + jj * x1grid_size + ii].s1;
                        u_state[ii + x1grid_size * jj + x1grid_size * x2grid_size * kk].s2   =   u_state[x1grid_size * x2grid_size * (x3grid_size - 3) + jj * x1grid_size + ii].s2;
                        u_state[ii + x1grid_size * jj + x1grid_size * x2grid_size * kk].s3   =   u_state[x1grid_size * x2grid_size * (x3grid_size - 3) + jj * x1grid_size + ii].s3;
                        u_state[ii + x1grid_size * jj + x1grid_size * x2grid_size * kk].tau  =   u_state[x1grid_size * x2grid_size * (x3grid_size - 3) + jj * x1grid_size + ii].tau;

                    } else {
                        u_state[kk * x1grid_size * x2grid_size + jj * x1grid_size + 0].d               = u_state[kk * x1grid_size * x2grid_size + jj * x1grid_size + 1].d;
                        u_state[kk * x1grid_size * x2grid_size + jj * x1grid_size + x1grid_size - 1].d = u_state[kk * x1grid_size * x2grid_size + jj * x1grid_size + x1grid_size - 2].d;

                        u_state[kk * x1grid_size * x2grid_size + jj * x1grid_size + 0].s1               = - u_state[kk * x1grid_size * x2grid_size + jj * x1grid_size + 1].s1;
                        u_state[kk * x1grid_size * x2grid_size + jj * x1grid_size + x1grid_size - 1].s1 =   u_state[kk * x1grid_size * x2grid_size + jj * x1grid_size + x1grid_size - 2].s1;

                        u_state[kk * x1grid_size * x2grid_size + jj * x1grid_size + 0].s2                = u_state[kk * x1grid_size * x2grid_size + jj * x1grid_size + 1].s2;
                        u_state[kk * x1grid_size * x2grid_size + jj * x1grid_size + x1grid_size - 1].s2  = u_state[kk * x1grid_size * x2grid_size + jj * x1grid_size + x1grid_size - 2].s2;

                        u_state[kk * x1grid_size * x2grid_size + jj * x1grid_size + 0].s3                = u_state[kk * x1grid_size * x2grid_size + jj * x1grid_size + 1].s3;
                        u_state[kk * x1grid_size * x2grid_size + jj * x1grid_size + x1grid_size - 1].s3  = u_state[kk * x1grid_size * x2grid_size + jj * x1grid_size + x1grid_size - 2].s3;

                        u_state[kk * x1grid_size * x2grid_size + jj * x1grid_size + 0].tau               = u_state[kk * x1grid_size * x2grid_size + jj * x1grid_size + 1].tau;
                        u_state[kk * x1grid_size * x2grid_size + jj * x1grid_size + x1grid_size - 1].tau = u_state[kk * x1grid_size * x2grid_size + jj * x1grid_size + x1grid_size - 2].tau;
                    }
                }

            } else {
                if((jj < x2grid_size) && (kk < x3grid_size))
                {

                    // Fix the ghost zones at the radial boundaries
                    u_state[kk * x1grid_size * x2grid_size + jj * x1grid_size +  0].d               = u_state[kk * x1grid_size * x2grid_size + jj * x1grid_size +  3].d;
                    u_state[kk * x1grid_size * x2grid_size + jj * x1grid_size +  1].d               = u_state[kk * x1grid_size * x2grid_size + jj * x1grid_size +  2].d;
                    u_state[kk * x1grid_size * x2grid_size + jj * x1grid_size +  x1grid_size - 1].d = u_state[kk * x1grid_size * x2grid_size + jj * x1grid_size +  x1grid_size - 3].d;
                    u_state[kk * x1grid_size * x2grid_size + jj * x1grid_size +  x1grid_size - 2].d = u_state[kk * x1grid_size * x2grid_size + jj * x1grid_size +  x1grid_size - 3].d;

                    u_state[kk * x1grid_size * x2grid_size + jj * x1grid_size + 0].s1               = - u_state[kk * x1grid_size * x2grid_size + jj * x1grid_size + 3].s1;
                    u_state[kk * x1grid_size * x2grid_size + jj * x1grid_size + 1].s1               = - u_state[kk * x1grid_size * x2grid_size + jj * x1grid_size + 2].s1;
                    u_state[kk * x1grid_size * x2grid_size + jj * x1grid_size + x1grid_size - 1].s1 =   u_state[kk * x1grid_size * x2grid_size + jj * x1grid_size + x1grid_size - 3].s1;
                    u_state[kk * x1grid_size * x2grid_size + jj * x1grid_size + x1grid_size - 2].s1 =   u_state[kk * x1grid_size * x2grid_size + jj * x1grid_size + x1grid_size - 3].s1;

                    u_state[kk * x1grid_size * x2grid_size + jj * x1grid_size + 0].s2               = u_state[kk * x1grid_size * x2grid_size + jj * x1grid_size + 3].s2;
                    u_state[kk * x1grid_size * x2grid_size + jj * x1grid_size + 1].s2               = u_state[kk * x1grid_size * x2grid_size + jj * x1grid_size + 2].s2;
                    u_state[kk * x1grid_size * x2grid_size + jj * x1grid_size + x1grid_size - 1].s2 = u_state[kk * x1grid_size * x2grid_size + jj * x1grid_size + x1grid_size - 3].s2;
                    u_state[kk * x1grid_size * x2grid_size + jj * x1grid_size + x1grid_size - 2].s2 = u_state[kk * x1grid_size * x2grid_size + jj * x1grid_size + x1grid_size - 3].s2;

                    u_state[kk * x1grid_size * x2grid_size + jj * x1grid_size + 0].s3               = u_state[kk * x1grid_size * x2grid_size + jj * x1grid_size + 3].s3;
                    u_state[kk * x1grid_size * x2grid_size + jj * x1grid_size + 1].s3               = u_state[kk * x1grid_size * x2grid_size + jj * x1grid_size + 2].s3;
                    u_state[kk * x1grid_size * x2grid_size + jj * x1grid_size + x1grid_size - 1].s3 = u_state[kk * x1grid_size * x2grid_size + jj * x1grid_size + x1grid_size - 3].s3;
                    u_state[kk * x1grid_size * x2grid_size + jj * x1grid_size + x1grid_size - 2].s3 = u_state[kk * x1grid_size * x2grid_size + jj * x1grid_size + x1grid_size - 3].s3;

                    u_state[kk * x1grid_size * x2grid_size + jj * x1grid_size + 0].tau                = u_state[kk * x1grid_size * x2grid_size + jj * x1grid_size + 3].tau;
                    u_state[kk * x1grid_size * x2grid_size + jj * x1grid_size + 1].tau                = u_state[kk * x1grid_size * x2grid_size + jj * x1grid_size + 2].tau;
                    u_state[kk * x1grid_size * x2grid_size + jj * x1grid_size + x1grid_size - 1].tau  = u_state[kk * x1grid_size * x2grid_size + jj * x1grid_size + x1grid_size - 3].tau;
                    u_state[kk * x1grid_size * x2grid_size + jj * x1grid_size + x1grid_size - 2].tau  = u_state[kk * x1grid_size * x2grid_size + jj * x1grid_size + x1grid_size - 3].tau;

                    // Fix the ghost zones at the angular boundaries
                    
                    if (jj < 2){
                        if (ii < x1grid_size){
                            if (jj == 0){
                                u_state[kk * x1grid_size * x2grid_size + jj * x1grid_size + ii].d   =   u_state[kk * x1grid_size * x2grid_size + 3 * x1grid_size + ii].d;
                                u_state[kk * x1grid_size * x2grid_size + jj * x1grid_size + ii].s1  =   u_state[kk * x1grid_size * x2grid_size + 3 * x1grid_size + ii].s1;
                                u_state[kk * x1grid_size * x2grid_size + jj * x1grid_size + ii].s2  =   u_state[kk * x1grid_size * x2grid_size + 3 * x1grid_size + ii].s2;
                                u_state[kk * x1grid_size * x2grid_size + jj * x1grid_size + ii].s3  =   u_state[kk * x1grid_size * x2grid_size + 3 * x1grid_size + ii].s3;
                                u_state[kk * x1grid_size * x2grid_size + jj * x1grid_size + ii].tau =   u_state[kk * x1grid_size * x2grid_size + 3 * x1grid_size + ii].tau;
                            } else {
                                u_state[kk * x1grid_size * x2grid_size + jj * x1grid_size + ii].d    =   u_state[kk * x1grid_size * x2grid_size + 2 * x1grid_size + ii].d;
                                u_state[kk * x1grid_size * x2grid_size + jj * x1grid_size + ii].s1   =   u_state[kk * x1grid_size * x2grid_size + 2 * x1grid_size + ii].s1;
                                u_state[kk * x1grid_size * x2grid_size + jj * x1grid_size + ii].s2   =   u_state[kk * x1grid_size * x2grid_size + 2 * x1grid_size + ii].s2;
                                u_state[kk * x1grid_size * x2grid_size + jj * x1grid_size + ii].s3   =   u_state[kk * x1grid_size * x2grid_size + 2 * x1grid_size + ii].s3;
                                u_state[kk * x1grid_size * x2grid_size + jj * x1grid_size + ii].tau  =   u_state[kk * x1grid_size * x2grid_size + 2 * x1grid_size + ii].tau;
                            }
                        }
                    } else if (jj > x2grid_size - 3) {
                        if (ii < x1grid_size){
                            if (jj == x2grid_size - 1){
                                u_state[kk * x1grid_size * x2grid_size + jj * x1grid_size + ii].d   =   u_state[kk * x1grid_size * x2grid_size + (x2grid_size - 4) * x1grid_size + ii].d;
                                u_state[kk * x1grid_size * x2grid_size + jj * x1grid_size + ii].s1  =   u_state[kk * x1grid_size * x2grid_size + (x2grid_size - 4) * x1grid_size + ii].s1;
                                u_state[kk * x1grid_size * x2grid_size + jj * x1grid_size + ii].s2  =   u_state[kk * x1grid_size * x2grid_size + (x2grid_size - 4) * x1grid_size + ii].s2;
                                u_state[kk * x1grid_size * x2grid_size + jj * x1grid_size + ii].s3  =   u_state[kk * x1grid_size * x2grid_size + (x2grid_size - 4) * x1grid_size + ii].s3;
                                u_state[kk * x1grid_size * x2grid_size + jj * x1grid_size + ii].tau =   u_state[kk * x1grid_size * x2grid_size + (x2grid_size - 4) * x1grid_size + ii].tau;
                            } else {
                                u_state[kk * x1grid_size * x2grid_size + jj * x1grid_size + ii].d   =   u_state[kk * x1grid_size * x2grid_size + (x2grid_size - 3) * x1grid_size + ii].d;
                                u_state[kk * x1grid_size * x2grid_size + jj * x1grid_size + ii].s1  =   u_state[kk * x1grid_size * x2grid_size + (x2grid_size - 3) * x1grid_size + ii].s1;
                                u_state[kk * x1grid_size * x2grid_size + jj * x1grid_size + ii].s2  =   u_state[kk * x1grid_size * x2grid_size + (x2grid_size - 3) * x1grid_size + ii].s2;
                                u_state[kk * x1grid_size * x2grid_size + jj * x1grid_size + ii].s3  =   u_state[kk * x1grid_size * x2grid_size + (x2grid_size - 3) * x1grid_size + ii].s3;
                                u_state[kk * x1grid_size * x2grid_size + jj * x1grid_size + ii].tau =   u_state[kk * x1grid_size * x2grid_size + (x2grid_size - 3) * x1grid_size + ii].tau;
                            }
                        }
                    }

                    if (kk < 2){
                        if (ii < x1grid_size){
                            if (jj == 0){
                                u_state[kk * x1grid_size * x2grid_size + jj * x1grid_size + ii].d   =   u_state[3 * x1grid_size * x2grid_size + jj * x1grid_size + ii].d;
                                u_state[kk * x1grid_size * x2grid_size + jj * x1grid_size + ii].s1  =   u_state[3 * x1grid_size * x2grid_size + jj * x1grid_size + ii].s1;
                                u_state[kk * x1grid_size * x2grid_size + jj * x1grid_size + ii].s2  =   u_state[3 * x1grid_size * x2grid_size + jj * x1grid_size + ii].s2;
                                u_state[kk * x1grid_size * x2grid_size + jj * x1grid_size + ii].s3  =   u_state[3 * x1grid_size * x2grid_size + jj * x1grid_size + ii].s3;
                                u_state[kk * x1grid_size * x2grid_size + jj * x1grid_size + ii].tau =   u_state[3 * x1grid_size * x2grid_size + jj * x1grid_size + ii].tau;
                            } else {
                                u_state[kk * x1grid_size * x2grid_size + jj * x1grid_size + ii].d    =   u_state[2 * x1grid_size * x2grid_size + jj * x1grid_size + ii].d;
                                u_state[kk * x1grid_size * x2grid_size + jj * x1grid_size + ii].s1   =   u_state[2 * x1grid_size * x2grid_size + jj * x1grid_size + ii].s1;
                                u_state[kk * x1grid_size * x2grid_size + jj * x1grid_size + ii].s2   =   u_state[2 * x1grid_size * x2grid_size + jj * x1grid_size + ii].s2;
                                u_state[kk * x1grid_size * x2grid_size + jj * x1grid_size + ii].s3   =   u_state[2 * x1grid_size * x2grid_size + jj * x1grid_size + ii].s3;
                                u_state[kk * x1grid_size * x2grid_size + jj * x1grid_size + ii].tau  =   u_state[2 * x1grid_size * x2grid_size + jj * x1grid_size + ii].tau;
                            }
                        }
                    } else if (kk > x3grid_size - 3) {
                        if (ii < x1grid_size){
                            if (kk == x3grid_size - 1){
                                u_state[kk * x1grid_size * x2grid_size + jj * x1grid_size + ii].d   =   u_state[(x3grid_size - 4) * x1grid_size * x2grid_size + jj * x1grid_size + ii].d;
                                u_state[kk * x1grid_size * x2grid_size + jj * x1grid_size + ii].s1  =   u_state[(x3grid_size - 4) * x1grid_size * x2grid_size + jj * x1grid_size + ii].s1;
                                u_state[kk * x1grid_size * x2grid_size + jj * x1grid_size + ii].s2  =   u_state[(x3grid_size - 4) * x1grid_size * x2grid_size + jj * x1grid_size + ii].s2;
                                u_state[kk * x1grid_size * x2grid_size + jj * x1grid_size + ii].s3  =   u_state[(x3grid_size - 4) * x1grid_size * x2grid_size + jj * x1grid_size + ii].s3;
                                u_state[kk * x1grid_size * x2grid_size + jj * x1grid_size + ii].tau =   u_state[(x3grid_size - 4) * x1grid_size * x2grid_size + jj * x1grid_size + ii].tau;
                            } else {
                                u_state[kk * x1grid_size * x2grid_size + jj * x1grid_size + ii].d   =   u_state[(x3grid_size - 3) * x1grid_size * x2grid_size + jj * x1grid_size + ii].d;
                                u_state[kk * x1grid_size * x2grid_size + jj * x1grid_size + ii].s1  =   u_state[(x3grid_size - 3) * x1grid_size * x2grid_size + jj * x1grid_size + ii].s1;
                                u_state[kk * x1grid_size * x2grid_size + jj * x1grid_size + ii].s2  =   u_state[(x3grid_size - 3) * x1grid_size * x2grid_size + jj * x1grid_size + ii].s2;
                                u_state[kk * x1grid_size * x2grid_size + jj * x1grid_size + ii].s3  =   u_state[(x3grid_size - 3) * x1grid_size * x2grid_size + jj * x1grid_size + ii].s3;
                                u_state[kk * x1grid_size * x2grid_size + jj * x1grid_size + ii].tau =   u_state[(x3grid_size - 3) * x1grid_size * x2grid_size + jj * x1grid_size + ii].tau;
                            }
                        }
                    }
                    
                }

            }

        });
        
    };
}
