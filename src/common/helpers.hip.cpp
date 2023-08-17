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
        const simbi::BoundaryCondition* boundary_conditions,
        const sr1d::Conserved *outer_zones,
        const sr1d::Conserved *inflow_zones)
    {
        simbi::parallel_for(p, 0, 1, [=] GPU_LAMBDA (const int gid) {
            if (first_order){                
                switch (boundary_conditions[0])
                {
                case simbi::BoundaryCondition::INFLOW:
                    cons[0] = inflow_zones[0];
                    break;
                case simbi::BoundaryCondition::REFLECTING:
                    cons[0]   =   cons[1];
                    cons[0].s = - cons[1].s;
                    break;
                case simbi::BoundaryCondition::PERIODIC:
                    cons[0] = cons[grid_size - 2];
                    break;
                default:
                    cons[0] = cons[1];
                    break;
                }

                switch (boundary_conditions[1])
                {
                case simbi::BoundaryCondition::INFLOW:
                    cons[grid_size - 1] = inflow_zones[1];
                    break;
                case simbi::BoundaryCondition::REFLECTING:
                    cons[grid_size - 1]   =   cons[grid_size - 2];
                    cons[grid_size - 1].s = - cons[grid_size - 2].s;
                    break;
                case simbi::BoundaryCondition::PERIODIC:
                    cons[grid_size - 1] = cons[1];
                    break;
                default:
                    cons[grid_size - 1] = cons[grid_size - 2];
                    break;
                }

                if (outer_zones) {
                    cons[grid_size - 1] = outer_zones[0];
                }
            } else {
                
                switch (boundary_conditions[0])
                {
                case simbi::BoundaryCondition::INFLOW:
                    cons[0] = inflow_zones[0];
                    cons[1] = inflow_zones[1];
                    break;
                case simbi::BoundaryCondition::REFLECTING:
                    cons[0]   =   cons[3];
                    cons[1]   =   cons[2];
                    cons[0].s = - cons[3].s;
                    cons[1].s = - cons[2].s;
                    break;
                case simbi::BoundaryCondition::PERIODIC:
                    cons[0] = cons[grid_size - 4];
                    cons[1] = cons[grid_size - 3];
                    break;
                default:
                    cons[0] = cons[2];
                    cons[1] = cons[2];
                    break;
                }

                switch (boundary_conditions[1])
                {
                case simbi::BoundaryCondition::INFLOW:
                    cons[grid_size - 1] = inflow_zones[0];
                    cons[grid_size - 2] = inflow_zones[0];
                    break;
                case simbi::BoundaryCondition::REFLECTING:
                    cons[grid_size - 1]   =   cons[grid_size - 4];
                    cons[grid_size - 2]   =   cons[grid_size - 3];
                    cons[grid_size - 1].s = - cons[grid_size - 4].s;
                    cons[grid_size - 2].s = - cons[grid_size - 3].s;
                    break;
                case simbi::BoundaryCondition::PERIODIC:
                    cons[grid_size - 1] = cons[3];
                    cons[grid_size - 2] = cons[2];
                    break;
                default:
                    cons[grid_size - 1] = cons[grid_size - 3];
                    cons[grid_size - 2] = cons[grid_size - 3];
                    break;
                }

                if (outer_zones) {
                    cons[grid_size - 1] = outer_zones[0];
                    cons[grid_size - 2] = outer_zones[0];
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
                case simbi::BoundaryCondition::PERIODIC:
                    cons[grid_size - 1] = cons[3];
                    cons[grid_size - 2] = cons[2];
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
        const simbi::Geometry geometry,
        const simbi::BoundaryCondition* boundary_conditions,
        const sr2d::Conserved *outer_zones,
        const sr2d::Conserved *boundary_zones,
        const bool half_sphere)
    {
        const int extent = p.get_full_extent();
        const int sx = (col_maj) ? 1 : x1grid_size;
        const int sy = (col_maj) ? x2grid_size : 1;
        simbi::parallel_for(p, 0, extent, [=] GPU_LAMBDA (const int gid) {
            const int ii = (BuildPlatform == Platform::GPU) ? blockDim.x * blockIdx.x + threadIdx.x: gid % x1grid_size;
            const int jj = (BuildPlatform == Platform::GPU) ? blockDim.y * blockIdx.y + threadIdx.y: gid / x1grid_size;
            if (first_order){
                if(jj < x2grid_size - 2) {
                    switch (boundary_conditions[0])
                    {
                    case simbi::BoundaryCondition::REFLECTING:
                        cons[(jj + 1) * sx + 0 * sy]    =   cons[(jj + 1) * sx + 1 * sy];
                        cons[(jj + 1) * sx + 0 * sy].s1 = - cons[(jj + 1) * sx + 1 * sy].s1;
                        break;
                    case simbi::BoundaryCondition::INFLOW:
                        cons[(jj + 1) * sx + 0 * sy] =  boundary_zones[0];
                        break;
                    case simbi::BoundaryCondition::PERIODIC:
                        cons[(jj + 1) * sx + 0 * sy] =  cons[(jj + 1) * sx + (x1grid_size - 2) * sy];
                        break;
                    default:
                        cons[(jj + 1) * sx +  0 * sy] = cons[(jj + 1) * sx +  1 * sy];
                        break;
                    }

                    switch (boundary_conditions[1])
                    {
                    case simbi::BoundaryCondition::REFLECTING:
                        cons[(jj + 1) * sx + (x1grid_size - 1) * sy]    =   cons[(jj + 1) * sx + (x1grid_size - 2) * sy];
                        cons[(jj + 1) * sx + (x1grid_size - 1) * sy].s1 = - cons[(jj + 1) * sx + (x1grid_size - 2) * sy].s1;
                        break;
                    case simbi::BoundaryCondition::INFLOW:
                        cons[(jj + 1) * sx + (x1grid_size - 1) * sy] = boundary_zones[1];
                        break;
                    case simbi::BoundaryCondition::PERIODIC:
                        cons[(jj + 1) * sx + (x1grid_size - 1) * sy] =  cons[(jj + 1) * sx + 1 * sy];
                        break;
                    default:
                        cons[(jj + 1) * sx + (x1grid_size - 1) * sy] = cons[(jj + 1) * sx +  (x1grid_size - 2) * sy];
                        break;
                    }
                }
                // Fix the ghost zones at the angular boundaries
                if (ii < x1grid_size - 2) {
                    switch (geometry)
                    {
                    case simbi::Geometry::SPHERICAL:
                        cons[0 * sx + (ii + 1) * sy]  = cons[1 * sx + (ii + 1) * sy];
                        cons[(x2grid_size - 1) * sx + (ii + 1) * sy] = cons[(x2grid_size - 2) * sx + (ii + 1) * sy];
                        if (half_sphere) {
                            cons[(x2grid_size - 1) * sx + (ii + 2) * sy].s2 = - cons[(x2grid_size - 2) * sx + (ii + 2) * sy].s2;
                        }
                        break;
                    case simbi::Geometry::PLANAR_CYLINDRICAL:
                        cons[0 * sx + (ii + 1) * sy]  = cons[(x2grid_size - 2) * sx + (ii + 1) * sy];
                        cons[(x2grid_size - 1) * sx + (ii + 1) * sy] = cons[1 * sx + (ii + 1) * sy];
                        break;
                    default:
                        switch (boundary_conditions[2])
                            {
                            case simbi::BoundaryCondition::REFLECTING:
                                cons[0 * sx + (ii + 1) * sy]  = cons[1 * sx + (ii + 1) * sy];
                                cons[0 * sx + (ii + 1) * sy].s2  = - cons[1 * sx + (ii + 1) * sy].s2;
                                break;
                            case simbi::BoundaryCondition::INFLOW:
                                cons[0 * sx + (ii + 1) * sy] = boundary_zones[2];
                                break;
                            case simbi::BoundaryCondition::PERIODIC:
                                cons[0 * sx + (ii + 1) * sy] = cons[(x2grid_size - 2) * sx + (ii + 1) * sy];
                                break;
                            default:
                                cons[0 * sx + (ii + 1) * sy]  = cons[1 * sx + (ii + 1) * sy];
                                break;
                            }

                        switch (boundary_conditions[3])
                            {
                            case simbi::BoundaryCondition::REFLECTING:
                                cons[(x2grid_size - 1) * sx + (ii + 1) * sy]    =   cons[(x2grid_size - 2) * sx + (ii + 1) * sy];
                                cons[(x2grid_size - 1) * sx + (ii + 1) * sy].s2 = - cons[(x2grid_size - 2) * sx + (ii + 1) * sy].s2;
                                break;
                            case simbi::BoundaryCondition::INFLOW:
                                cons[(x2grid_size - 1) * sx + (ii + 1) * sy] = boundary_zones[3];
                                break;
                            case simbi::BoundaryCondition::PERIODIC:
                                cons[(x2grid_size - 1) * sx + (ii + 1) * sy] =  cons[1 * sx + (ii + 1) * sy];
                                break;
                            default:
                                // Fix the ghost zones at the radial boundaries
                                cons[(x2grid_size - 1) * sx + (ii + 1) * sy] = cons[(x2grid_size - 2) * sx +  (ii + 1) * sy];
                                break;
                            }
                        
                        break;
                    } // end switch
                }
            } else {
                if(jj < x2grid_size - 4) {
                    // Fix the ghost zones at the radial boundaries
                    cons[(jj + 2) * sx +  (x1grid_size - 1) * sy] = cons[(jj + 2) * sx +  (x1grid_size - 3) * sy];
                    cons[(jj + 2) * sx +  (x1grid_size - 2) * sy] = cons[(jj + 2) * sx +  (x1grid_size - 3) * sy];
                    switch (boundary_conditions[0]) {
                    case simbi::BoundaryCondition::REFLECTING:
                        cons[(jj + 2) * sx +  0 * sy]   = cons[(jj + 2) * sx +  3 * sy];
                        cons[(jj + 2) * sx +  1 * sy]   = cons[(jj + 2) * sx +  2 * sy];

                        cons[(jj + 2) * sx + 0 * sy].s1 = - cons[(jj + 2) * sx + 3 * sy].s1;
                        cons[(jj + 2) * sx + 1 * sy].s1 = - cons[(jj + 2) * sx + 2 * sy].s1;
                        break;
                    case simbi::BoundaryCondition::INFLOW:
                        cons[(jj + 2) * sx +  0 * sy]   = boundary_zones[0];
                        cons[(jj + 2) * sx +  1 * sy]   = boundary_zones[0];
                        break;
                    case simbi::BoundaryCondition::PERIODIC:
                        cons[(jj + 2) * sx +  0 * sy]   = cons[(jj + 2) * sx +  (x1grid_size - 4) * sy];
                        cons[(jj + 2) * sx +  1 * sy]   = cons[(jj + 2) * sx +  (x1grid_size - 3) * sy];
                        break;
                    default:
                        cons[(jj + 2) * sx +  0 * sy]   = cons[(jj + 2) * sx +  2 * sy];
                        cons[(jj + 2) * sx +  1 * sy]   = cons[(jj + 2) * sx +  2 * sy];
                        break;
                    }

                    switch (boundary_conditions[1]) {
                    case simbi::BoundaryCondition::REFLECTING:
                        cons[(jj + 2) * sx +  (x1grid_size - 1) * sy]   = cons[(jj + 2) * sx +  (x1grid_size - 4) * sy];
                        cons[(jj + 2) * sx +  (x1grid_size - 2) * sy]   = cons[(jj + 2) * sx +  (x1grid_size - 3) * sy];

                        cons[(jj + 2) * sx + (x1grid_size - 1) * sy].s1 = - cons[(jj + 2) * sx + (x1grid_size - 4) * sy].s1;
                        cons[(jj + 2) * sx + (x1grid_size - 2) * sy].s1 = - cons[(jj + 2) * sx + (x1grid_size - 3) * sy].s1;
                        break;
                    case simbi::BoundaryCondition::INFLOW:
                        cons[(jj + 2) * sx +  (x1grid_size - 1) * sy]   = boundary_zones[1];
                        cons[(jj + 2) * sx +  (x1grid_size - 2) * sy]   = boundary_zones[1];
                        break;
                    case simbi::BoundaryCondition::PERIODIC:
                        cons[(jj + 2) * sx +  (x1grid_size - 1) * sy]   = cons[(jj + 2) * sx +  3 * sy];
                        cons[(jj + 2) * sx +  (x1grid_size - 2) * sy]   = cons[(jj + 2) * sx +  2 * sy];
                        break;
                    default:
                        cons[(jj + 2) * sx +  (x1grid_size - 1) * sy] = cons[(jj + 2) * sx +  (x1grid_size - 3) * sy];
                        cons[(jj + 2) * sx +  (x1grid_size - 2) * sy] = cons[(jj + 2) * sx +  (x1grid_size - 3) * sy];
                        break;
                    }
                }

                // Fix the ghost zones at the x2 boundaries
                if (ii < x1grid_size - 4) {
                    switch (geometry) 
                    {
                    case simbi::Geometry::SPHERICAL:
                        cons[0 * sx + (ii + 2) * sy]  = cons[3 * sx + (ii + 2) * sy];
                        cons[1 * sx + (ii + 2) * sy]  = cons[2 * sx + (ii + 2) * sy];
                        cons[(x2grid_size - 1) * sx + (ii + 2) * sy] = cons[(x2grid_size - 4) * sx + (ii + 2) * sy];
                        cons[(x2grid_size - 2) * sx + (ii + 2) * sy] = cons[(x2grid_size - 3) * sx + (ii + 2) * sy];
                        if (half_sphere) {
                            cons[(x2grid_size - 1) * sx + (ii + 2) * sy].s2 = - cons[(x2grid_size - 4) * sx + (ii + 2) * sy].s2;
                            cons[(x2grid_size - 2) * sx + (ii + 2) * sy].s2 = - cons[(x2grid_size - 3) * sx + (ii + 2) * sy].s2;
                        }
                        break;
                    case simbi::Geometry::PLANAR_CYLINDRICAL:
                        cons[0 * sx + (ii + 2) * sy]  = cons[(x2grid_size - 4) * sx + (ii + 2) * sy];
                        cons[1 * sx + (ii + 2) * sy]  = cons[(x2grid_size - 3) * sx + (ii + 2) * sy];
                        cons[(x2grid_size - 1) * sx + (ii + 2) * sy] = cons[2 * sx + (ii + 2) * sy];
                        cons[(x2grid_size - 2) * sx + (ii + 2) * sy] = cons[3 * sx + (ii + 2) * sy];
                        break;
                    default:
                            switch (boundary_conditions[2]) {
                            case simbi::BoundaryCondition::REFLECTING:
                                cons[0 * sx + (ii + 2) * sy]  = cons[3 * sx + (ii + 2) * sy];
                                cons[1 * sx + (ii + 2) * sy]  = cons[2 * sx + (ii + 2) * sy];
                                cons[0 * sx + (ii + 2) * sy].s2  = - cons[3 * sx + (ii + 2) * sy].s2;
                                cons[1 * sx + (ii + 2) * sy].s2  = - cons[2 * sx + (ii + 2) * sy].s2;
                                break;
                            case simbi::BoundaryCondition::INFLOW:
                                cons[0 * sx +  (ii + 2) * sy] = boundary_zones[2];
                                cons[1 * sx +  (ii + 2) * sy] = boundary_zones[2];
                                break;
                            case simbi::BoundaryCondition::PERIODIC:
                                cons[0 * sx + (ii + 2) * sy]  = cons[(x2grid_size - 4) * sx + (ii + 2) * sy];
                                cons[1 * sx + (ii + 2) * sy]  = cons[(x2grid_size - 3) * sx + (ii + 2) * sy];
                                break;
                            default:
                                cons[0 * sx + (ii + 2) * sy]  = cons[2 * sx + (ii + 2) * sy];
                                cons[1 * sx + (ii + 2) * sy]  = cons[2 * sx + (ii + 2) * sy];
                                break;
                            }

                            switch (boundary_conditions[3]) {
                            case simbi::BoundaryCondition::REFLECTING:
                                cons[(x2grid_size - 1) * sx + (ii + 2) * sy]  = cons[(x2grid_size - 4) * sx + (ii + 2) * sy];
                                cons[(x2grid_size - 2) * sx + (ii + 2) * sy]  = cons[(x2grid_size - 3) * sx + (ii + 2) * sy];
                                cons[(x2grid_size - 1) * sx + (ii + 2) * sy].s2  = - cons[(x2grid_size - 4) * sx + (ii + 2) * sy].s2;
                                cons[(x2grid_size - 2) * sx + (ii + 2) * sy].s2  = - cons[(x2grid_size - 3) * sx + (ii + 2) * sy].s2;
                                break;
                            case simbi::BoundaryCondition::INFLOW:
                                cons[(x2grid_size - 1) * sx +  (ii + 2) * sy] = boundary_zones[3];
                                cons[(x2grid_size - 2) * sx +  (ii + 2) * sy] = boundary_zones[3];
                                break;
                            case simbi::BoundaryCondition::PERIODIC:
                                cons[(x2grid_size - 1) * sx + (ii + 2) * sy]  = cons[3 * sx + (ii + 2) * sy];
                                cons[(x2grid_size - 2) * sx + (ii + 2) * sy]  = cons[2 * sx + (ii + 2) * sy];
                                break;
                            default:
                                cons[(x2grid_size - 1) * sx + (ii + 2) * sy]  = cons[(x2grid_size - 3) * sx + (ii + 2) * sy];
                                cons[(x2grid_size - 2) * sx + (ii + 2) * sy]  = cons[(x2grid_size - 3) * sx + (ii + 2) * sy];
                                break;
                            }
                        break;
                    } // end switch
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
        const simbi::Geometry geometry,
        const simbi::BoundaryCondition *boundary_conditions,
        const hydro2d::Conserved *outer_zones,
        const hydro2d::Conserved *boundary_zones,
        const bool half_sphere)
    {
        const int extent = p.get_full_extent();
        const int sx = (col_maj) ? 1 : x1grid_size;
        const int sy = (col_maj) ? x2grid_size : 1;
        simbi::parallel_for(p, 0, extent, [=] GPU_LAMBDA (const int gid) {
            const int ii = (BuildPlatform == Platform::GPU) ? blockDim.x * blockIdx.x + threadIdx.x: gid % x1grid_size;
            const int jj = (BuildPlatform == Platform::GPU) ? blockDim.y * blockIdx.y + threadIdx.y: gid / x1grid_size;
            if (first_order){
                if(jj < x2grid_size - 2) {
                    switch (boundary_conditions[0])
                    {
                    case simbi::BoundaryCondition::REFLECTING:
                        cons[(jj + 1) * sx + 0 * sy]    =   cons[(jj + 1) * sx + 1 * sy];
                        cons[(jj + 1) * sx + 0 * sy].m1 = - cons[(jj + 1) * sx + 1 * sy].m1;
                        break;
                    case simbi::BoundaryCondition::INFLOW:
                        cons[(jj + 1) * sx + 0 * sy] =  boundary_zones[0];
                        break;
                    case simbi::BoundaryCondition::PERIODIC:
                        cons[(jj + 1) * sx +  0 * sy] = cons[(jj + 1) * sx +  (x1grid_size - 2) * sy];
                        break;
                    default:
                        cons[(jj + 1) * sx +  0 * sy] = cons[(jj + 1) * sx +  1 * sy];
                        break;
                    }

                    switch (boundary_conditions[1])
                    {
                    case simbi::BoundaryCondition::REFLECTING:
                        cons[(jj + 1) * sx + (x1grid_size - 1) * sy]    =   cons[(jj + 1) * sx + (x1grid_size - 2) * sy];
                        cons[(jj + 1) * sx + (x1grid_size - 1) * sy].m1 = - cons[(jj + 1) * sx + (x1grid_size - 2) * sy].m1;
                        break;
                    case simbi::BoundaryCondition::INFLOW:
                        cons[(jj + 1) * sx + (x1grid_size - 1) * sy] = boundary_zones[1];
                        break;
                    case simbi::BoundaryCondition::PERIODIC:
                        cons[(jj + 1) * sx + (x1grid_size - 1) * sy] = cons[(jj + 1) * sx +  1 * sy];
                        break;
                    default:
                        cons[(jj + 1) * sx + (x1grid_size - 1) * sy] = cons[(jj + 1) * sx +  (x1grid_size - 2) * sy];
                        break;
                    }
                }
                // Fix the ghost zones at the angular boundaries
                if (ii < x1grid_size - 2) {
                    switch (geometry)
                    {
                    case simbi::Geometry::SPHERICAL:
                        cons[0 * sx + (ii + 1) * sy]  = cons[1 * sx + (ii + 1) * sy];
                        cons[(x2grid_size - 1) * sx + (ii + 1) * sy] = cons[(x2grid_size - 2) * sx + (ii + 1) * sy];
                        if (half_sphere) {
                            cons[(x2grid_size - 1) * sx + (ii + 2) * sy].m2 = - cons[(x2grid_size - 2) * sx + (ii + 2) * sy].m2;
                        }
                        break;
                    case simbi::Geometry::PLANAR_CYLINDRICAL:
                        cons[0 * sx + (ii + 1) * sy]  = cons[(x2grid_size - 2) * sx + (ii + 1) * sy];
                        cons[(x2grid_size - 1) * sx + (ii + 1) * sy] = cons[1 * sx + (ii + 1) * sy];
                        break;
                    default:
                        switch (boundary_conditions[2])
                            {
                            case simbi::BoundaryCondition::REFLECTING:
                                cons[0 * sx + (ii + 1) * sy]  = cons[1 * sx + (ii + 1) * sy];
                                cons[0 * sx + (ii + 1) * sy].m1  = - cons[1 * sx + (ii + 1) * sy].m1;
                                break;
                            case simbi::BoundaryCondition::INFLOW:
                                cons[0 * sx + (ii + 1) * sy] = boundary_zones[2];
                                break;
                            case simbi::BoundaryCondition::PERIODIC:
                                cons[0 * sx + (ii + 1) * sy]  = cons[(x1grid_size - 2) * sx + (ii + 1) * sy];
                                break;
                            default:
                                cons[0 * sx + (ii + 1) * sy]  = cons[1 * sx + (ii + 1) * sy];
                                break;
                            }

                        switch (boundary_conditions[3])
                            {
                            case simbi::BoundaryCondition::REFLECTING:
                                cons[(x2grid_size - 1) * sx + (ii + 1) * sy]    =   cons[(x2grid_size - 2) * sx + (ii + 1) * sy];
                                cons[(x2grid_size - 1) * sx + (ii + 1) * sy].m1 = - cons[(x2grid_size - 2) * sx + (ii + 1) * sy].m1;
                                break;
                            case simbi::BoundaryCondition::INFLOW:
                                cons[(x2grid_size - 1) * sx + (ii + 1) * sy] = boundary_zones[3];
                                break;
                            case simbi::BoundaryCondition::PERIODIC:
                                cons[(x2grid_size - 1) * sx + (ii + 1) * sy] = cons[1 * sx + (ii + 1) * sy];
                                break;
                            default:
                                // Fix the ghost zones at the radial boundaries
                                cons[(x2grid_size - 1) * sx + (ii + 1) * sy] = cons[(x2grid_size - 2) * sx +  (ii + 1) * sy];
                                break;
                            }
                        
                        break;
                    } // end switch
                }
            } else {
                if(jj < x2grid_size - 4) {
                    // Fix the ghost zones at the radial boundaries
                    cons[(jj + 2) * sx +  (x1grid_size - 1) * sy] = cons[(jj + 2) * sx +  (x1grid_size - 3) * sy];
                    cons[(jj + 2) * sx +  (x1grid_size - 2) * sy] = cons[(jj + 2) * sx +  (x1grid_size - 3) * sy];
                    switch (boundary_conditions[0]) {
                    case simbi::BoundaryCondition::REFLECTING:
                        cons[(jj + 2) * sx +  0 * sy]   = cons[(jj + 2) * sx +  3 * sy];
                        cons[(jj + 2) * sx +  1 * sy]   = cons[(jj + 2) * sx +  2 * sy];

                        cons[(jj + 2) * sx + 0 * sy].m1 = - cons[(jj + 2) * sx + 3 * sy].m1;
                        cons[(jj + 2) * sx + 1 * sy].m1 = - cons[(jj + 2) * sx + 2 * sy].m1;
                        break;
                    case simbi::BoundaryCondition::INFLOW:
                        cons[(jj + 2) * sx +  0 * sy]   = boundary_zones[0];
                        cons[(jj + 2) * sx +  1 * sy]   = boundary_zones[0];
                        break;
                    case simbi::BoundaryCondition::PERIODIC:
                        cons[(jj + 2) * sx +  0 * sy]   = cons[(jj + 2) * sx +  (x1grid_size - 4) * sy];
                        cons[(jj + 2) * sx +  1 * sy]   = cons[(jj + 2) * sx +  (x1grid_size - 3) * sy];
                        break;
                    default:
                        cons[(jj + 2) * sx +  0 * sy]   = cons[(jj + 2) * sx +  2 * sy];
                        cons[(jj + 2) * sx +  1 * sy]   = cons[(jj + 2) * sx +  2 * sy];
                        break;
                    }

                    switch (boundary_conditions[1]) {
                    case simbi::BoundaryCondition::REFLECTING:
                        cons[(jj + 2) * sx +  (x1grid_size - 1) * sy]   = cons[(jj + 2) * sx +  (x1grid_size - 4) * sy];
                        cons[(jj + 2) * sx +  (x1grid_size - 2) * sy]   = cons[(jj + 2) * sx +  (x1grid_size - 3) * sy];

                        cons[(jj + 2) * sx + (x1grid_size - 1) * sy].m1 = - cons[(jj + 2) * sx + (x1grid_size - 4) * sy].m1;
                        cons[(jj + 2) * sx + (x1grid_size - 2) * sy].m1 = - cons[(jj + 2) * sx + (x1grid_size - 3) * sy].m1;
                        break;
                    case simbi::BoundaryCondition::INFLOW:
                        cons[(jj + 2) * sx +  0 * sy]   = boundary_zones[1];
                        cons[(jj + 2) * sx +  1 * sy]   = boundary_zones[1];
                        break;
                    case simbi::BoundaryCondition::PERIODIC:
                        cons[(jj + 2) * sx +  (x1grid_size - 1) * sy] = cons[(jj + 2) * sx +  3 * sy];
                        cons[(jj + 2) * sx +  (x1grid_size - 2) * sy] = cons[(jj + 2) * sx +  2 * sy];
                        break;
                    default:
                        cons[(jj + 2) * sx +  (x1grid_size - 1) * sy] = cons[(jj + 2) * sx +  (x1grid_size - 3) * sy];
                        cons[(jj + 2) * sx +  (x1grid_size - 2) * sy] = cons[(jj + 2) * sx +  (x1grid_size - 3) * sy];
                        break;
                    }
                }

                // Fix the ghost zones at the angular boundaries
                if (ii < x1grid_size - 4) {
                    switch (geometry) 
                    {
                    case simbi::Geometry::SPHERICAL:
                        cons[0 * sx + (ii + 2) * sy]  = cons[3 * sx + (ii + 2) * sy];
                        cons[1 * sx + (ii + 2) * sy]  = cons[2 * sx + (ii + 2) * sy];
                        cons[(x2grid_size - 1) * sx + (ii + 2) * sy] = cons[(x2grid_size - 4) * sx + (ii + 2) * sy];
                        cons[(x2grid_size - 2) * sx + (ii + 2) * sy] = cons[(x2grid_size - 3) * sx + (ii + 2) * sy];
                        if (half_sphere) {
                            cons[(x2grid_size - 1) * sx + (ii + 2) * sy].m2 = - cons[(x2grid_size - 4) * sx + (ii + 2) * sy].m2;
                            cons[(x2grid_size - 2) * sx + (ii + 2) * sy].m2 = - cons[(x2grid_size - 3) * sx + (ii + 2) * sy].m2;
                        }
                        break;
                    case simbi::Geometry::PLANAR_CYLINDRICAL:
                        cons[0 * sx + (ii + 2) * sy]  = cons[(x2grid_size - 4) * sx + (ii + 2) * sy];
                        cons[1 * sx + (ii + 2) * sy]  = cons[(x2grid_size - 3) * sx + (ii + 2) * sy];
                        cons[(x2grid_size - 1) * sx + (ii + 2) * sy] = cons[2 * sx + (ii + 2) * sy];
                        cons[(x2grid_size - 2) * sx + (ii + 2) * sy] = cons[3 * sx + (ii + 2) * sy];
                        break;
                    default:
                            switch (boundary_conditions[2]) {
                            case simbi::BoundaryCondition::REFLECTING:
                                cons[0 * sx + (ii + 2) * sy]  = cons[3 * sx + (ii + 2) * sy];
                                cons[1 * sx + (ii + 2) * sy]  = cons[2 * sx + (ii + 2) * sy];
                                cons[0 * sx + (ii + 2) * sy].m1  = - cons[3 * sx + (ii + 2) * sy].m1;
                                cons[1 * sx + (ii + 2) * sy].m1  = - cons[2 * sx + (ii + 2) * sy].m1;
                                break;
                            case simbi::BoundaryCondition::INFLOW:
                                cons[0 * sx +  (ii + 2) * sy] = boundary_zones[2];
                                cons[1 * sx +  (ii + 2) * sy] = boundary_zones[2];
                                break;
                            case simbi::BoundaryCondition::PERIODIC:
                                cons[0 * sx + (ii + 2) * sy]  = cons[(x2grid_size - 4) * sx + (ii + 2) * sy];
                                cons[1 * sx + (ii + 2) * sy]  = cons[(x2grid_size - 3) * sx + (ii + 2) * sy];
                                break;
                            default:
                                cons[0 * sx + (ii + 2) * sy]  = cons[2 * sx + (ii + 2) * sy];
                                cons[1 * sx + (ii + 2) * sy]  = cons[2 * sx + (ii + 2) * sy];
                                break;
                            }

                            switch (boundary_conditions[3]) {
                            case simbi::BoundaryCondition::REFLECTING:
                                cons[(x2grid_size - 1) * sx + (ii + 2) * sy]  = cons[(x2grid_size - 4) * sx + (ii + 2) * sy];
                                cons[(x2grid_size - 2) * sx + (ii + 2) * sy]  = cons[(x2grid_size - 3) * sx + (ii + 2) * sy];
                                cons[(x2grid_size - 1) * sx + (ii + 2) * sy].m1  = - cons[(x2grid_size - 4) * sx + (ii + 2) * sy].m1;
                                cons[(x2grid_size - 2) * sx + (ii + 2) * sy].m1  = - cons[(x2grid_size - 3) * sx + (ii + 2) * sy].m1;
                                break;
                            case simbi::BoundaryCondition::INFLOW:
                                cons[0 * sx +  (ii + 2) * sy] = boundary_zones[3];
                                cons[1 * sx +  (ii + 2) * sy] = boundary_zones[3];
                                break;
                            case simbi::BoundaryCondition::PERIODIC:
                                cons[(x2grid_size - 1) * sx + (ii + 2) * sy]  = cons[3 * sx + (ii + 2) * sy];
                                cons[(x2grid_size - 2) * sx + (ii + 2) * sy]  = cons[2 * sx + (ii + 2) * sy];
                                break;
                            default:
                                cons[(x2grid_size - 1) * sx + (ii + 2) * sy]  = cons[(x2grid_size - 3) * sx + (ii + 2) * sy];
                                cons[(x2grid_size - 2) * sx + (ii + 2) * sy]  = cons[(x2grid_size - 3) * sx + (ii + 2) * sy];
                                break;
                            }
                        break;
                    } // end switch
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
        const simbi::BoundaryCondition* boundary_conditions,
        const sr3d::Conserved* inflow_zones,
        const bool half_sphere,
        const simbi::Geometry geometry)
    {
        const int extent = p.get_full_extent();
        const int sx = x1grid_size;
        const int sy = x2grid_size;
        simbi::parallel_for(p, 0, extent, [=] GPU_LAMBDA (const int gid) {
            const int kk = (BuildPlatform == Platform::GPU) ? blockDim.z * blockIdx.z + threadIdx.z : simbi::detail::get_height(gid, x1grid_size, x2grid_size);
            const int jj = (BuildPlatform == Platform::GPU) ? blockDim.y * blockIdx.y + threadIdx.y : simbi::detail::get_row(gid, x1grid_size, x2grid_size, kk);
            const int ii = (BuildPlatform == Platform::GPU) ? blockDim.x * blockIdx.x + threadIdx.x : simbi::detail::get_column(gid, x1grid_size, x2grid_size, kk);

            
            if (first_order){
                if(jj < x2grid_size - 2 && kk < x3grid_size - 2) {
                    
                    switch (boundary_conditions[0])
                    {
                    case simbi::BoundaryCondition::REFLECTING:
                        cons[(kk + 1) * sx * sy + (jj + 1) * sx + 0] = cons[(kk + 1) * sx * sy + (jj + 1) * sx + 1];
                        cons[(kk + 1) * sx * sy + (jj + 1) * sx + 0].s1 = -cons[(kk + 1) * sx * sy + (jj + 1) * sx + 1].s1;
                        break;
                    case simbi::BoundaryCondition::INFLOW:
                        cons[(kk + 1) * sx * sy + (jj + 1) * sx + 0] = inflow_zones[0];
                        break;
                    case simbi::BoundaryCondition::PERIODIC:
                        cons[(kk + 1) * sx * sy + (jj + 1) * sx + 0] = cons[(kk + 1) * sx * sy + (jj + 1) * sx + (x1grid_size - 2)];
                        break;
                    default:
                        cons[(kk + 1) * sx * sy + (jj + 1) * sx + 0] = cons[(kk + 1) * sx * sy + (jj + 1) * sx + 1];
                        break;
                    }

                    switch (boundary_conditions[1])
                    {
                    case simbi::BoundaryCondition::REFLECTING:
                        cons[(kk + 1) * sx * sy + (jj + 1) * sx + (x1grid_size - 1)] = cons[(kk + 1) * sx * sy + (jj + 1) * sx + (x1grid_size - 2)];
                        cons[(kk + 1) * sx * sy + (jj + 1) * sx + (x1grid_size - 1)].s1 = - cons[(kk + 1) * sx * sy + (jj + 1) * sx + (x1grid_size - 2)].s1;
                        break;
                    case simbi::BoundaryCondition::INFLOW:
                        cons[(kk + 1) * sx * sy + (jj + 1) * sx + (x1grid_size - 1)] = inflow_zones[1];
                        break;
                    case simbi::BoundaryCondition::PERIODIC:
                        cons[(kk + 1) * sx * sy + (jj + 1) * sx + (x1grid_size - 1)] = cons[(kk + 1) * sx * sy + (jj + 1) * sx + 1];
                        break;
                    default:
                        cons[(kk + 1) * sx * sy + (jj + 1) * sx + (x1grid_size - 1)] = cons[(kk + 1) * sx * sy + (jj + 1) * sx + (x1grid_size - 2)];
                        break;
                    }
                }
                // Fix the ghost zones at the x2 boundaries
                if (ii < x1grid_size - 2 && kk < x3grid_size - 2) {
                    switch (geometry)
                    {
                    case simbi::Geometry::SPHERICAL:
                        cons[(kk + 1) * sx * sy + 0 * sx + (ii + 1)]                 = cons[(kk + 1) * sx * sy + 1 * sx + (ii + 1)];
                        cons[(kk + 1) * sx * sy + (x2grid_size - 1) * sx + (ii + 1)] = cons[(kk + 1) * sx * sy + (x2grid_size - 2) * sx + (ii + 1)];
                        
                        if (half_sphere) {
                            cons[(kk + 1) * sx * sy + (x2grid_size - 1) * sx + (ii + 1)].s2 = - cons[(kk + 1) * sx * sy + (x2grid_size - 2) * sx + (ii + 1)].s2;
                        }
                        break;
                    case simbi::Geometry::CYLINDRICAL:
                        cons[(kk + 1) * sx * sy + 0 * sx + (ii + 1)]                 = cons[(kk + 1) * sx * sy + (x2grid_size - 2) * sx + (ii + 1)];
                        cons[(kk + 1) * sx * sy + (x2grid_size - 1) * sx + (ii + 1)] = cons[(kk + 1) * sx * sy + 1 * sx + (ii + 1)];
                        break;
                    default:
                        switch (boundary_conditions[2])
                        {
                        case simbi::BoundaryCondition::REFLECTING:
                            cons[(kk + 1) * sx * sy + 0 * sx + (ii + 1)]    = cons[(kk + 1) * sx * sy + 1 * sx + (ii + 1)];
                            cons[(kk + 1) * sx * sy + 0 * sx + (ii + 1)].s2 = cons[(kk + 1) * sx * sy + 1 * sx + (ii + 1)].s2;
                            break;
                        case simbi::BoundaryCondition::INFLOW:
                            cons[(kk + 1) * sx * sy + 0 * sx + (ii + 1)] = inflow_zones[2];
                            break;
                        case simbi::BoundaryCondition::PERIODIC:
                            cons[(kk + 1) * sx * sy + 0 * sx + (ii + 1)] = cons[(kk + 1) * sx * sy + (x2grid_size - 1) * sx + (ii + 1)];
                            break;
                        default:
                            cons[(kk + 1) * sx * sy + 0 * sx + (ii + 1)] = cons[(kk + 1) * sx * sy + 1 * sx + (ii + 1)];
                            break;
                        }

                        switch (boundary_conditions[3])
                        {
                        case simbi::BoundaryCondition::REFLECTING:
                            cons[(kk + 1) * sx * sy + (x2grid_size - 1) * sx + (ii + 1)]    =   cons[(kk + 1) * sx * sy + (x2grid_size - 2) * sx + (ii + 1)];
                            cons[(kk + 1) * sx * sy + (x2grid_size - 1) * sx + (ii + 1)].s2 = - cons[(kk + 1) * sx * sy + (x2grid_size - 2) * sx + (ii + 1)].s2;
                            break;
                        case simbi::BoundaryCondition::INFLOW:
                            cons[(kk + 1) * sx * sy + (x2grid_size - 1) * sx + (ii + 1)] = inflow_zones[3];
                            break;
                        case simbi::BoundaryCondition::PERIODIC:
                            cons[(kk + 1) * sx * sy + (x2grid_size - 1) * sx + (ii + 1)] = cons[(kk + 1) * sx * sy + 1 * sx + (ii + 1)];
                            break;
                        default:
                            cons[(kk + 1) * sx * sy + (x2grid_size - 1) * sx + (ii + 1)] = cons[(kk + 1) * sx * sy + (x2grid_size - 1) * sx + (ii + 1)];
                            break;
                        }
                        break;
                    }
                }

                // Fix the ghost zones at the x3 boundaries
                if (jj < x2grid_size - 2 && ii < x1grid_size - 2) {
                    switch (geometry)
                    {
                    case simbi::Geometry::SPHERICAL:
                        cons[0 * sx * sy + (jj + 1) * sx + (ii + 1)]                 = cons[(x3grid_size - 2) * sx * sy + (jj + 1) * sx + (ii + 1)];
                        cons[(x3grid_size - 1) * sx * sy + (jj + 1) * sx + (ii + 1)] = cons[1 * sx * sy + (jj + 1) * sx + (ii + 1)];
                        break;
                    default:
                        switch (boundary_conditions[4])
                        {
                        case simbi::BoundaryCondition::REFLECTING:
                            cons[0 * sx * sy + (jj + 1) * sx + (ii + 1)]    =   cons[1 * sx * sy + (jj + 1) * sx + (ii + 1)];
                            cons[0 * sx * sy + (jj + 1) * sx + (ii + 1)].s3 = - cons[1 * sx * sy + (jj + 1) * sx + (ii + 1)].s3;
                            break;
                        case simbi::BoundaryCondition::INFLOW:
                            cons[0 * sx * sy + (jj + 1) * sx + (ii + 1)] = inflow_zones[4];
                            break;
                        case simbi::BoundaryCondition::PERIODIC:
                            cons[0 * sx * sy + (jj + 1) * sx + (ii + 1)] = cons[(x3grid_size - 2) * sx * sy + (jj + 1) * sx + (ii + 1)];
                            break;
                        default:
                            cons[0 * sx * sy + (jj + 1) * sx + (ii + 1)] = cons[1 * sx * sy + (jj + 1) * sx + (ii + 1)];
                            break;
                        }
                        switch (boundary_conditions[5])
                        {
                        case simbi::BoundaryCondition::REFLECTING:
                            cons[(x3grid_size - 1) * sx * sy + (jj + 1) * sx + (ii + 1)]    =   cons[(x2grid_size - 2) * sx * sy + (jj + 1) * sx + (ii + 1)];
                            cons[(x3grid_size - 1) * sx * sy + (jj + 1) * sx + (ii + 1)].s3 = - cons[(x2grid_size - 2) * sx * sy + (jj + 1) * sx + (ii + 1)].s3;
                            break;
                        case simbi::BoundaryCondition::INFLOW:
                            cons[(x3grid_size - 1) * sx * sy + (jj + 1) * sx + (ii + 1)] = inflow_zones[5];
                            break;
                        case simbi::BoundaryCondition::PERIODIC:
                            cons[(x3grid_size - 1) * sx * sy + (jj + 1) * sx + (ii + 1)] = cons[1 * sx * sy + (jj + 1) * sx + (ii + 1)];
                            break;
                        default:
                            cons[(x3grid_size - 1) * sx * sy + (jj + 1) * sx + (ii + 1)] = cons[(x3grid_size - 2) * sx * sy + (jj + 1) * sx + (ii + 1)];
                            break;
                        }

                        break;
                    }
                    
                }

            } else {
                if(jj < x2grid_size - 4 && kk < x3grid_size - 4) {
                    
                    switch (boundary_conditions[0])
                    {
                    case simbi::BoundaryCondition::REFLECTING:
                        cons[(kk + 2) * sx * sy + (jj + 2) * sx + 0] = cons[(kk + 2) * sx * sy + (jj + 2) * sx + 3];
                        cons[(kk + 2) * sx * sy + (jj + 2) * sx + 1] = cons[(kk + 2) * sx * sy + (jj + 2) * sx + 2];
                        cons[(kk + 2) * sx * sy + (jj + 2) * sx + 0].s1 = - cons[(kk + 2) * sx * sy + (jj + 2) * sx + 3].s1;
                        cons[(kk + 2) * sx * sy + (jj + 2) * sx + 1].s1 = - cons[(kk + 2) * sx * sy + (jj + 2) * sx + 2].s1;
                        break;
                    case simbi::BoundaryCondition::INFLOW:
                        cons[(kk + 2) * sx * sy + (jj + 2) * sx + 0] = inflow_zones[0];
                        cons[(kk + 2) * sx * sy + (jj + 2) * sx + 1] = inflow_zones[0];
                        break;
                    case simbi::BoundaryCondition::PERIODIC:
                        cons[(kk + 2) * sx * sy + (jj + 2) * sx + 0] = cons[(kk + 2) * sx * sy + (jj + 2) * sx + (x1grid_size - 4)];
                        cons[(kk + 2) * sx * sy + (jj + 2) * sx + 1] = cons[(kk + 2) * sx * sy + (jj + 2) * sx + (x1grid_size - 3)];
                        break;
                    default:
                        cons[(kk + 2) * sx * sy + (jj + 2) * sx + 0] = cons[(kk + 2) * sx * sy + (jj + 2) * sx + 2];
                        cons[(kk + 2) * sx * sy + (jj + 2) * sx + 1] = cons[(kk + 2) * sx * sy + (jj + 2) * sx + 2];
                        break;
                    }

                    switch (boundary_conditions[1])
                    {
                    case simbi::BoundaryCondition::REFLECTING:
                        cons[(kk + 2) * sx * sy + (jj + 2) * sx + (x1grid_size - 1)] = cons[(kk + 2) * sx * sy + (jj + 2) * sx + (x1grid_size - 4)];
                        cons[(kk + 2) * sx * sy + (jj + 2) * sx + (x1grid_size - 2)] = cons[(kk + 2) * sx * sy + (jj + 2) * sx + (x1grid_size - 3)];
                        cons[(kk + 2) * sx * sy + (jj + 2) * sx + (x1grid_size - 1)].s1 = - cons[(kk + 2) * sx * sy + (jj + 2) * sx + (x1grid_size - 4)].s1;
                        cons[(kk + 2) * sx * sy + (jj + 2) * sx + (x1grid_size - 2)].s1 = - cons[(kk + 2) * sx * sy + (jj + 2) * sx + (x1grid_size - 3)].s1;
                        break;
                    case simbi::BoundaryCondition::INFLOW:
                        cons[(kk + 2) * sx * sy + (jj + 2) * sx + (x1grid_size - 1)] = inflow_zones[1];
                        cons[(kk + 2) * sx * sy + (jj + 2) * sx + (x1grid_size - 2)] = inflow_zones[1];
                        break;
                    case simbi::BoundaryCondition::PERIODIC:
                        cons[(kk + 2) * sx * sy + (jj + 2) * sx + (x1grid_size - 1)] = cons[(kk + 2) * sx * sy + (jj + 2) * sx + 3];
                        cons[(kk + 2) * sx * sy + (jj + 2) * sx + (x1grid_size - 2)] = cons[(kk + 2) * sx * sy + (jj + 2) * sx + 2];
                        break;
                    default:
                        cons[(kk + 2) * sx * sy + (jj + 2) * sx + (x1grid_size - 1)] = cons[(kk + 2) * sx * sy + (jj + 2) * sx + (x1grid_size - 3)];
                        cons[(kk + 2) * sx * sy + (jj + 2) * sx + (x1grid_size - 2)] = cons[(kk + 2) * sx * sy + (jj + 2) * sx + (x1grid_size - 3)];
                        break;
                    }
                }
                // Fix the ghost zones at the x2 boundaries
                if (ii < x1grid_size - 4 && kk < x3grid_size - 4) {
                    switch (geometry)
                    {
                    case simbi::Geometry::SPHERICAL:
                        cons[(kk + 2) * sx * sy + 0 * sx + (ii + 2)]                 = cons[(kk + 2) * sx * sy + 3 * sx + (ii + 2)];
                        cons[(kk + 2) * sx * sy + 1 * sx + (ii + 2)]                 = cons[(kk + 2) * sx * sy + 2 * sx + (ii + 2)];
                        cons[(kk + 2) * sx * sy + (x2grid_size - 1) * sx + (ii + 2)] = cons[(kk + 2) * sx * sy + (x2grid_size - 4) * sx + (ii + 2)];
                        cons[(kk + 2) * sx * sy + (x2grid_size - 2) * sx + (ii + 2)] = cons[(kk + 2) * sx * sy + (x2grid_size - 3) * sx + (ii + 2)];
                        if (half_sphere) {
                            cons[(kk + 2) * sx * sy + (x2grid_size - 4) * sx + (ii + 2)].s2 *= - 1;
                            cons[(kk + 2) * sx * sy + (x2grid_size - 3) * sx + (ii + 2)].s2 *= - 1;
                        }
                        break;
                    case simbi::Geometry::CYLINDRICAL:
                        cons[(kk + 2) * sx * sy + 0 * sx + (ii + 2)]                 = cons[(kk + 2) * sx * sy + (x2grid_size - 3) * sx + (ii + 2)];
                        cons[(kk + 2) * sx * sy + 1 * sx + (ii + 2)]                 = cons[(kk + 2) * sx * sy + (x2grid_size - 2) * sx + (ii + 2)];                        
                        cons[(kk + 2) * sx * sy + (x2grid_size - 1) * sx + (ii + 2)] = cons[(kk + 2) * sx * sy + 3 * sx + (ii + 2)];
                        cons[(kk + 2) * sx * sy + (x2grid_size - 2) * sx + (ii + 2)] = cons[(kk + 2) * sx * sy + 2 * sx + (ii + 2)];
                        break;
                    default:
                        switch (boundary_conditions[2])
                        {
                        case simbi::BoundaryCondition::REFLECTING:
                            cons[(kk + 2) * sx * sy + 0 * sx + (ii + 2)]     =   cons[(kk + 2) * sx * sy + 3 * sx + (ii + 2)];
                            cons[(kk + 2) * sx * sy + 1 * sx + (ii + 2)]     =   cons[(kk + 2) * sx * sy + 2 * sx + (ii + 2)];
                            cons[(kk + 2) * sx * sy + 0 * sx + (ii + 2)].s2  = - cons[(kk + 2) * sx * sy + 3 * sx + (ii + 2)].s2;
                            cons[(kk + 2) * sx * sy + 1 * sx + (ii + 2)].s2  = - cons[(kk + 2) * sx * sy + 2 * sx + (ii + 2)].s2;
                            break;
                        case simbi::BoundaryCondition::INFLOW:
                            cons[(kk + 2) * sx * sy + 0 * sx + (ii + 2)] = inflow_zones[2];
                            cons[(kk + 2) * sx * sy + 1 * sx + (ii + 2)] = inflow_zones[2];
                            break;
                        case simbi::BoundaryCondition::PERIODIC:
                            cons[(kk + 2) * sx * sy + 0 * sx + (ii + 2)] = cons[(kk + 2)* sx * sy + (x2grid_size - 4) * sx + (ii + 2)];
                            cons[(kk + 2) * sx * sy + 1 * sx + (ii + 2)] = cons[(kk + 2)* sx * sy + (x2grid_size - 3) * sx + (ii + 2)];
                            break;
                        default:
                            cons[(kk + 2) * sx * sy + 0 * sx + (ii + 2)] = cons[(kk + 2)* sx * sy + 2 * sx + (ii + 2)];
                            cons[(kk + 2) * sx * sy + 1 * sx + (ii + 2)] = cons[(kk + 2)* sx * sy + 2 * sx + (ii + 2)];
                            break;
                        }

                        switch (boundary_conditions[3])
                        {
                        case simbi::BoundaryCondition::REFLECTING:
                            cons[(kk + 2) * sx * sy + (x2grid_size - 1) * sx + (ii + 2)]    =   cons[(kk + 2) * sx * sy + (x2grid_size - 4) * sx + (ii + 2)];
                            cons[(kk + 2) * sx * sy + (x2grid_size - 2) * sx + (ii + 2)]    =   cons[(kk + 2) * sx * sy + (x2grid_size - 3) * sx + (ii + 2)];
                            cons[(kk + 2) * sx * sy + (x2grid_size - 1) * sx + (ii + 2)].s2 = - cons[(kk + 2) * sx * sy + (x2grid_size - 4) * sx + (ii + 2)].s2;
                            cons[(kk + 2) * sx * sy + (x2grid_size - 2) * sx + (ii + 2)].s2 = - cons[(kk + 2) * sx * sy + (x2grid_size - 3) * sx + (ii + 2)].s2;
                            break;
                        case simbi::BoundaryCondition::INFLOW:
                            cons[(kk + 2) * sx * sy + (x2grid_size - 1) * sx + (ii + 2)] = inflow_zones[3];
                            cons[(kk + 2) * sx * sy + (x2grid_size - 2) * sx + (ii + 2)] = inflow_zones[3];
                            break;
                        case simbi::BoundaryCondition::PERIODIC:
                            cons[(kk + 2) * sx * sy + (x2grid_size - 1) * sx + (ii + 2)] = cons[(kk + 2) * sx * sy + 3 * sx + (ii + 2)];
                            cons[(kk + 2) * sx * sy + (x2grid_size - 2) * sx + (ii + 2)] = cons[(kk + 2) * sx * sy + 2 * sx + (ii + 2)];
                            break;
                        default:
                            cons[(kk + 2) * sx * sy + (x2grid_size - 1) * sx + (ii + 2)]    =   cons[(kk + 2) * sx * sy + (x2grid_size - 3) * sx + (ii + 2)];
                            cons[(kk + 2) * sx * sy + (x2grid_size - 2) * sx + (ii + 2)]    =   cons[(kk + 2) * sx * sy + (x2grid_size - 3) * sx + (ii + 2)];
                            break;
                        }
                        break;
                    }
                }

                // Fix the ghost zones at the x3 boundaries
                if (jj < x2grid_size - 4 && ii < x1grid_size - 4) {
                    switch (geometry)
                    {
                    case simbi::Geometry::SPHERICAL:
                        cons[0 * sx * sy + (jj + 2) * sx + (ii + 2)] = cons[(x3grid_size - 4) * sx * sy + (jj + 2) * sx + (ii + 2)];
                        cons[1 * sx * sy + (jj + 2) * sx + (ii + 2)] = cons[(x3grid_size - 3) * sx * sy + (jj + 2) * sx + (ii + 2)];
                        cons[(x3grid_size - 1) * sx * sy + (jj + 2) * sx + (ii + 2)] = cons[3 * sx * sy + (jj + 2) * sx + (ii + 2)];
                        cons[(x3grid_size - 2) * sx * sy + (jj + 2) * sx + (ii + 2)] = cons[2 * sx * sy + (jj + 2) * sx + (ii + 2)];
                        break;
                    default:
                        switch (boundary_conditions[4])
                        {
                        case simbi::BoundaryCondition::REFLECTING:
                            cons[0 * sx * sy + (jj + 2) * sx + (ii + 2)] = cons[3 * sx * sy + (jj + 2) * sx + (ii + 2)];
                            cons[1 * sx * sy + (jj + 2) * sx + (ii + 2)] = cons[2 * sx * sy + (jj + 2) * sx + (ii + 2)];
                            cons[0 * sx * sy + (jj + 2) * sx + (ii + 2)].s3 = - cons[3 * sx * sy + (jj + 2) * sx + (ii + 2)].s3;
                            cons[1 * sx * sy + (jj + 2) * sx + (ii + 2)].s3 = - cons[2 * sx * sy + (jj + 2) * sx + (ii + 2)].s3;
                            break;
                        case simbi::BoundaryCondition::INFLOW:
                            cons[0 * sx * sy + (jj + 2) * sx + (ii + 2)] = inflow_zones[4];
                            cons[1 * sx * sy + (jj + 2) * sx + (ii + 2)] = inflow_zones[4];
                            break;
                        case simbi::BoundaryCondition::PERIODIC:
                            cons[0 * sx * sy + (jj + 2) * sx + (ii + 2)] = cons[(x2grid_size - 4) * sx * sy + (jj + 2) * sx + (ii + 2)];
                            cons[1 * sx * sy + (jj + 2) * sx + (ii + 2)] = cons[(x2grid_size - 3) * sx * sy + (jj + 2) * sx + (ii + 2)];
                            break;
                        default:
                            cons[0 * sx * sy + (jj + 2) * sx + (ii + 2)] = cons[2 * sx * sy + (jj + 2) * sx + (ii + 2)];
                            cons[1 * sx * sy + (jj + 2) * sx + (ii + 2)] = cons[2 * sx * sy + (jj + 2) * sx + (ii + 2)];
                            break;
                        }
                        switch (boundary_conditions[5])
                        {
                        case simbi::BoundaryCondition::REFLECTING:
                            cons[(x3grid_size - 1) * sx * sy + (jj + 2) * sx + (ii + 2)]    =   cons[(x3grid_size - 4) * sx * sy + (jj + 2) * sx + (ii + 2)];
                            cons[(x3grid_size - 2) * sx * sy + (jj + 2) * sx + (ii + 2)]    =   cons[(x3grid_size - 3) * sx * sy + (jj + 2) * sx + (ii + 2)];
                            cons[(x3grid_size - 1) * sx * sy + (jj + 2) * sx + (ii + 2)].s3 = - cons[(x3grid_size - 4) * sx * sy + (jj + 2) * sx + (ii + 2)].s3;
                            cons[(x3grid_size - 2) * sx * sy + (jj + 2) * sx + (ii + 2)].s3 = - cons[(x3grid_size - 3) * sx * sy + (jj + 2) * sx + (ii + 2)].s3;
                            break;
                        case simbi::BoundaryCondition::INFLOW:
                            cons[(x3grid_size - 1) * sx * sy + (jj + 2) * sx + (ii + 2)] = inflow_zones[5];
                            cons[(x3grid_size - 2) * sx * sy + (jj + 2) * sx + (ii + 2)] = inflow_zones[5];
                            break;
                        case simbi::BoundaryCondition::PERIODIC:
                            cons[(x3grid_size - 1) * sx * sy + (jj + 2) * sx + (ii + 2)] = cons[3 * sx * sy + (jj + 2) * sx + (ii + 2)];
                            cons[(x3grid_size - 2) * sx * sy + (jj + 2) * sx + (ii + 2)] = cons[2 * sx * sy + (jj + 2) * sx + (ii + 2)];
                            break;
                        default:
                            cons[(x3grid_size - 1) * sx * sy + (jj + 2) * sx + (ii + 2)] = cons[(x3grid_size - 3) * sx * sy + (jj + 2) * sx + (ii + 2)];
                            cons[(x3grid_size - 2) * sx * sy + (jj + 2) * sx + (ii + 2)] = cons[(x3grid_size - 3) * sx * sy + (jj + 2) * sx + (ii + 2)];
                            break;
                        }
                        break;
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
            gpu::api::getDeviceCount(&devCount);
            std::cout << std::string(80, '=')  << "\n";
            std::cout << "GPU Device(s): " << std::endl << std::endl;

            for(int i = 0; i < devCount; ++i)
            {
                anyGpuProp_t props;
                gpu::api::getDeviceProperties(&props, i);
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
        std::cout << "CPU Compute Thread(s): " << processor_count << std::endl;
        #endif
    }
} // namespace simbi
