
#include "util/parallel_for.hpp"
namespace simbi{
    template<typename T, typename U>
    void config_ghosts1D_t(
        const ExecutionPolicy<> p,
        T &conserved_arr, 
        const int grid_size,
        const bool first_order, 
        const simbi::BoundaryCondition* boundary_conditions,
        const U *outer_zones,
        const hydro1d::Conserved* inflow_zones) 
    {
        auto *cons = conserved_arr.data();
        simbi::parallel_for(p, 0, 1, [=] GPU_LAMBDA (const int gid) {
            if (first_order){                
                switch (boundary_conditions[0])
                {
                case simbi::BoundaryCondition::INFLOW:
                    cons[0] = inflow_zones[0];
                    break;
                case simbi::BoundaryCondition::REFLECTING:
                    cons[0]   = cons[1];
                    cons[0].m = - cons[1].m;
                    break;
                case simbi::BoundaryCondition::PERIODIC:
                    cons[0] = cons[grid_size - 2];
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
                    cons[grid_size - 1].m = - cons[grid_size - 2].m;
                    break;
                case simbi::BoundaryCondition::PERIODIC:
                    cons[grid_size - 1] = cons[1];
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
                    cons[0]   = cons[3];
                    cons[1]   = cons[2];
                    cons[0].m = - cons[3].m;
                    cons[1].m = - cons[2].m;
                    break;
                case simbi::BoundaryCondition::PERIODIC:
                    cons[0] = cons[grid_size - 4];
                    cons[1] = cons[grid_size - 3];
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
                    cons[grid_size - 1]   = cons[grid_size - 4];
                    cons[grid_size - 2]   = cons[grid_size - 3];
                    cons[grid_size - 1].m = - cons[grid_size - 4].m;
                    cons[grid_size - 2].m = - cons[grid_size - 3].m;
                    break;
                case simbi::BoundaryCondition::PERIODIC:
                    cons[grid_size - 1] = cons[3];
                    cons[grid_size - 2] = cons[2];
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

    template<typename T, TIMESTEP_TYPE dt_type, typename U, typename V>
    GPU_LAUNCHABLE  typename std::enable_if<is_1D_primitive<T>::value>::type 
    compute_dt(U *self, const V* prim_buffer, real* dt_min)
    {
        #if GPU_CODE
        real vPlus, vMinus;
        int ii   = blockDim.x * blockIdx.x + threadIdx.x;
        int aid  = ii + self->idx_active;
        if (ii < self->active_zones)
        {
            const real rho = prim_buffer[aid].rho;
            const real p   = prim_buffer[aid].p;
            const real v   = prim_buffer[aid].get_v();
        
            if constexpr(is_relativistic<T>::value)
            {
                if constexpr(dt_type == TIMESTEP_TYPE::ADAPTIVE) {
                    real h   = 1 + self->gamma * p / (rho * (self->gamma - 1));
                    real cs  = std::sqrt(self->gamma * p / (rho * h));
                    vPlus  = (v + cs) / (1 + v * cs);
                    vMinus = (v - cs) / (1 - v * cs);
                } else {
                    vPlus  = 1;
                    vMinus = 1;
                }
            } else {
                const real cs = std::sqrt(self->gamma * p / rho );
                vPlus         = (v + cs);
                vMinus        = (v - cs);
            }
            const real x1l    = self->get_xface(ii, self->geometry, 0);
            const real x1r    = self->get_xface(ii, self->geometry, 1);
            const real dx1    = x1r - x1l;
            const real vfaceL = (self->geometry == simbi::Geometry::CARTESIAN) ? self->hubble_param : x1l * self->hubble_param;
            const real vfaceR = (self->geometry == simbi::Geometry::CARTESIAN) ? self->hubble_param : x1r * self->hubble_param;
            const real cfl_dt = dx1 / (helpers::my_max(std::abs(vPlus - vfaceR), std::abs(vMinus - vfaceL)));
            dt_min[ii]        = self->cfl * cfl_dt;
        }
        #endif
    }

    template<typename T, TIMESTEP_TYPE dt_type, typename U, typename V>
    GPU_LAUNCHABLE  typename std::enable_if<is_2D_primitive<T>::value>::type 
    compute_dt(U *self, 
    const V* prim_buffer,
    real* dt_min,
    const simbi::Geometry geometry)
    {
        #if GPU_CODE
        real cfl_dt, v1p, v1m, v2p, v2m;
        const real gamma = self->gamma;
        const luint ii  = blockDim.x * blockIdx.x + threadIdx.x;
        const luint jj  = blockDim.y * blockIdx.y + threadIdx.y;
        const luint ia  = ii + self->idx_active;
        const luint ja  = jj + self->idx_active;
        const luint aid = (col_maj) ? ia * self-> ny + ja : ja * self->nx + ia;
        if ((ii < self->xphysical_grid) && (jj < self->yphysical_grid))
        {
            real plus_v1 , plus_v2 , minus_v1, minus_v2;
            real rho  = prim_buffer[aid].rho;
            real p    = prim_buffer[aid].p;
            real v1   = prim_buffer[aid].get_v1();
            real v2   = prim_buffer[aid].get_v2();

            if constexpr(is_relativistic<T>::value)
            {
                if constexpr(dt_type == TIMESTEP_TYPE::ADAPTIVE) {
                    real h   = 1 + gamma * p / (rho * (gamma - 1));
                    real cs  = std::sqrt(gamma * p / (rho * h));
                    plus_v1  = (v1 + cs) / (1 + v1 * cs);
                    plus_v2  = (v2 + cs) / (1 + v2 * cs);
                    minus_v1 = (v1 - cs) / (1 - v1 * cs);
                    minus_v2 = (v2 - cs) / (1 - v2 * cs);
                } else {
                    plus_v1  = 1;
                    plus_v2  = 1;
                    minus_v1 = 1;
                    minus_v2 = 1;
                }
            } else {
                real cs  = std::sqrt(gamma * p / rho);
                plus_v1  = (v1 + cs);
                plus_v2  = (v2 + cs);
                minus_v1 = (v1 - cs);
                minus_v2 = (v2 - cs);
            }

            v1p = std::abs(plus_v1);
            v1m = std::abs(minus_v1);
            v2p = std::abs(plus_v2);
            v2m = std::abs(minus_v2);
            switch (geometry)
            {
                case simbi::Geometry::CARTESIAN:
                    cfl_dt = helpers::my_min(self->dx1 / (helpers::my_max(v1p, v1m)),
                                             self->dx2 / (helpers::my_max(v2m, v2m)));
                    break;
                
                case simbi::Geometry::SPHERICAL:
                {
                    // Compute avg spherical distance 3/4 *(rf^4 - ri^4)/(rf^3 - ri^3)
                    const real rl           = self->get_x1face(ii, geometry, 0);
                    const real rr           = self->get_x1face(ii, geometry, 1);
                    const real tl           = self->get_x2face(jj, 0);  
                    const real tr           = self->get_x2face(jj, 1); 
                    if (self->mesh_motion)
                    {
                        const real vfaceL   = rl * self->hubble_param;
                        const real vfaceR   = rr * self->hubble_param;
                        v1p = std::abs(plus_v1  - vfaceR);
                        v1m = std::abs(minus_v1 - vfaceL);
                    }
                    const real rmean        = 0.75 * (rr * rr * rr * rr - rl * rl * rl *rl) / (rr * rr * rr - rl * rl * rl);
                    cfl_dt = helpers::my_min((rr - rl) / (helpers::my_max(v1p, v1m)),
                                     rmean * (tr - tl) / (helpers::my_max(v2p, v2m)));
                    break;
                }
                case simbi::Geometry::PLANAR_CYLINDRICAL:
                {
                    // Compute avg spherical distance 3/4 *(rf^4 - ri^4)/(rf^3 - ri^3)
                    const real rl           = self->get_x1face(ii, geometry, 0);
                    const real rr           = self->get_x1face(ii, geometry, 1);
                    const real tl           = self->get_x2face(jj, 0);  
                    const real tr           = self->get_x2face(jj, 1); 
                    if (self->mesh_motion)
                    {
                        const real vfaceL   = rl * self->hubble_param;
                        const real vfaceR   = rr * self->hubble_param;
                        v1p = std::abs(plus_v1  - vfaceR);
                        v1m = std::abs(minus_v1 - vfaceL);
                    }
                    const real rmean        = (2.0 / 3.0) * (rr * rr * rr - rl * rl * rl) / (rr * rr - rl * rl);
                    cfl_dt = helpers::my_min((rr - rl) / (helpers::my_max(v1p, v1m)),
                                     rmean * (tr - tl) / (helpers::my_max(v2p, v2m)));
                    break;
                }
                case simbi::Geometry::AXIS_CYLINDRICAL:
                {
                    const real rl           = self->get_x1face(ii, geometry, 0);
                    const real rr           = self->get_x1face(ii, geometry, 1);
                    const real zl           = self->get_x2face(jj, 0);  
                    const real zr           = self->get_x2face(jj, 1); 
                    if (self->mesh_motion)
                    {
                        const real vfaceL   = rl * self->hubble_param;
                        const real vfaceR   = rr * self->hubble_param;
                        v1p = std::abs(plus_v1  - vfaceR);
                        v1m = std::abs(minus_v1 - vfaceL);
                    }
                    cfl_dt = helpers::my_min((rr - rl) / (helpers::my_max(v1p, v1m)),
                                             (zr - zl) / (helpers::my_max(v2p, v2m)));
                    break;
                }
                // TODO: Implement
            } // end switch
            dt_min[jj * self->xphysical_grid + ii] = self->cfl * cfl_dt;
        }
        #endif
    }

    template<typename T, TIMESTEP_TYPE dt_type, typename U, typename V>
    GPU_LAUNCHABLE  typename std::enable_if<is_3D_primitive<T>::value>::type 
    compute_dt(U *self, 
    const V* prim_buffer,
    real *dt_min,
    const simbi::Geometry geometry)
    {
        #if GPU_CODE
        real cfl_dt;
        const real gamma = self->gamma;
        const luint ii  = blockDim.x * blockIdx.x + threadIdx.x;
        const luint jj  = blockDim.y * blockIdx.y + threadIdx.y;
        const luint kk  = blockDim.z * blockIdx.z + threadIdx.z;
        const luint ia  = ii + self->idx_active;
        const luint ja  = jj + self->idx_active;
        const luint ka  = kk + self->idx_active;
        const luint aid = (col_maj) ? ia * self-> ny + ja : ka * self->nx * self->ny + ja * self->nx + ia;
        if ((ii < self->xphysical_grid) && (jj < self->yphysical_grid) && (kk < self->zphysical_grid))
        {
            real plus_v1 , plus_v2 , minus_v1, minus_v2, plus_v3, minus_v3;
            real rho  = prim_buffer[aid].rho;
            real p    = prim_buffer[aid].p;
            real v1   = prim_buffer[aid].get_v1();
            real v2   = prim_buffer[aid].get_v2();
            real v3   = prim_buffer[aid].get_v3();

            if constexpr(is_relativistic<T>::value)
            {
                if constexpr(dt_type == TIMESTEP_TYPE::ADAPTIVE) {
                    real h   = 1 + gamma * p / (rho * (gamma - 1));
                    real cs  = std::sqrt(gamma * p / (rho * h));
                    plus_v1  = (v1 + cs) / (1 + v1 * cs);
                    plus_v2  = (v2 + cs) / (1 + v2 * cs);
                    plus_v3  = (v3 + cs) / (1 + v3 * cs);
                    minus_v1 = (v1 - cs) / (1 - v1 * cs);
                    minus_v2 = (v2 - cs) / (1 - v2 * cs);
                    minus_v3 = (v3 - cs) / (1 - v3 * cs);
                } else {
                    plus_v1  = 1;
                    plus_v2  = 1;
                    plus_v3  = 1;
                    minus_v1 = 1;
                    minus_v2 = 1;
                    minus_v3 = 1;
                }
            } else {
                real cs  = std::sqrt(gamma * p / rho);
                plus_v1  = (v1 + cs);
                plus_v2  = (v2 + cs);
                plus_v3  = (v3 + cs);
                minus_v1 = (v1 - cs);
                minus_v2 = (v2 - cs);
                minus_v3 = (v3 - cs);
            }

            const auto x1l = self->get_x1face(ii, geometry, 0);
            const auto x1r = self->get_x1face(ii, geometry, 1);
            const auto dx1 = x1r - x1l; 
            const auto x2l = self->get_x2face(jj, 0);
            const auto x2r = self->get_x2face(jj, 1);
            const auto dx2 = x2r - x2l; 
            const auto x3l = self->get_x3face(kk, 0);
            const auto x3r = self->get_x3face(kk, 1);
            const auto dx3 = x3r - x3l; 
            switch (geometry)
            {
                case simbi::Geometry::CARTESIAN:
                {

                    cfl_dt = helpers::my_min3(dx1 / (helpers::my_max(std::abs(plus_v1), std::abs(minus_v1))),
                                              dx2 / (helpers::my_max(std::abs(plus_v2), std::abs(minus_v2))),
                                              dx3 / (helpers::my_max(std::abs(plus_v3), std::abs(minus_v3))));
                    break;
                }
                case simbi::Geometry::SPHERICAL:
                {     
                    const real rmean = static_cast<real>(0.75) * (x1r * x1r * x1r * x1r - x1l * x1l * x1l *x1l) / (x1r * x1r * x1r - x1l * x1l * x1l);
                    const real th    = static_cast<real>(0.5) * (x2l + x2r);
                    cfl_dt = helpers::my_min3(dx1 / (helpers::my_max(std::abs(plus_v1), std::abs(minus_v1))),
                                      rmean * dx2 / (helpers::my_max(std::abs(plus_v2), std::abs(minus_v2))),
                       rmean * std::sin(th) * dx3 / (helpers::my_max(std::abs(plus_v3), std::abs(minus_v3))));
                    break;
                 }
                case simbi::Geometry::CYLINDRICAL:
                {    
                    const real rmean = static_cast<real>(2.0 / 3.0) * (x1r * x1r * x1r - x1l * x1l * x1l) / (x1r * x1r - x1l * x1l);
                    const real th    = static_cast<real>(0.5) * (x2l + x2r);
                    cfl_dt = helpers::my_min3(dx1 / (helpers::my_max(std::abs(plus_v1), std::abs(minus_v1))),
                                      rmean * dx2 / (helpers::my_max(std::abs(plus_v2), std::abs(minus_v2))),
                                              dx3 / (helpers::my_max(std::abs(plus_v3), std::abs(minus_v3))));
                    break;
                 }
            } // end switch
            
            dt_min[kk * self->xphysical_grid * self->yphysical_grid + jj * self->xphysical_grid + ii] = self->cfl * cfl_dt;
        }
        #endif
    }

    template<int dim, typename T>
    GPU_LAUNCHABLE void deviceReduceKernel(T *self, real *dt_min, lint nmax) {
        #if GPU_CODE
        real min = INFINITY;
        luint ii   = blockIdx.x * blockDim.x + threadIdx.x;
        luint jj   = blockIdx.y * blockDim.y + threadIdx.y;
        luint kk   = blockIdx.z * blockDim.z + threadIdx.z;
        luint tid  = threadIdx.z * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
        luint bid  = blockIdx.z * gridDim.x * gridDim.y + blockIdx.y * gridDim.x + blockIdx.x;
        luint nt   = blockDim.x * blockDim.y * blockDim.z * gridDim.x * gridDim.y * gridDim.z;
        luint gid;
        if constexpr(dim == 1) {
            gid = ii;
        } else if constexpr(dim == 2) {
            gid  = self->xphysical_grid * jj + ii;
        } else if constexpr(dim == 3) {
            gid  = self->yphysical_grid * self->xphysical_grid * kk + self->xphysical_grid * jj + ii;
        }
        // reduce multiple elements per thread
        for (luint i = gid; i < nmax; i += nt) {
            min = helpers::my_min(dt_min[i], min);
        }
        min = blockReduceMin(min);
        if (tid==0) {
            dt_min[bid] = min;
            self->dt    = dt_min[0];
        }
        #endif 
    };

    template<int dim, typename T>
    GPU_LAUNCHABLE void deviceReduceWarpAtomicKernel(T *self, real *dt_min, lint nmax) {
        #if GPU_CODE
        real min = INFINITY;
        luint ii   = blockIdx.x * blockDim.x + threadIdx.x;
        luint tid  = threadIdx.z * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
        // luint bid  = blockIdx.z * gridDim.x * gridDim.y + blockIdx.y * gridDim.x + blockIdx.x;
        luint nt   = blockDim.x * blockDim.y * blockDim.z * gridDim.x * gridDim.y * gridDim.z;
        luint gid;
        if constexpr(dim == 1) {
            gid = ii;
        } else if constexpr(dim == 2) {
            luint jj   = blockIdx.y * blockDim.y + threadIdx.y;
            gid  = self->xphysical_grid * jj + ii;
        } else if constexpr(dim == 3) {
            luint jj   = blockIdx.y * blockDim.y + threadIdx.y;
            luint kk   = blockIdx.z * blockDim.z + threadIdx.z;
            gid  = self->yphysical_grid * self->xphysical_grid * kk + self->xphysical_grid * jj + ii;
        }
        // reduce multiple elements per thread
        for(auto i = gid; i < nmax; i += nt) {
            min = helpers::my_min(dt_min[i], min);
        }

        min = blockReduceMin(min);
        if (tid == 0) {
            self->dt = atomicMinReal(dt_min, min);
        }
        #endif 
    };
}

