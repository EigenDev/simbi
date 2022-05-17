
namespace simbi{
    template<unsigned int blockSize>
    GPU_DEV void warpReduceMin(volatile real* smem, unsigned int tid)
    {
        if (blockSize >= 64) smem[tid] = (smem[tid] < smem[tid + 32]) ? smem[tid] : smem[tid + 32];
        if (blockSize >= 32) smem[tid] = (smem[tid] < smem[tid + 16]) ? smem[tid] : smem[tid + 16];
        if (blockSize >= 16) smem[tid] = (smem[tid] < smem[tid +  8]) ? smem[tid] : smem[tid +  8];
        if (blockSize >=  8) smem[tid] = (smem[tid] < smem[tid +  4]) ? smem[tid] : smem[tid +  4];
        if (blockSize >=  4) smem[tid] = (smem[tid] < smem[tid +  2]) ? smem[tid] : smem[tid +  2];
        if (blockSize >=  2) smem[tid] = (smem[tid] < smem[tid +  1]) ? smem[tid] : smem[tid +  1];
    };

    template<typename T, typename N>
    GPU_LAUNCHABLE  typename std::enable_if<is_1D_primitive<N>::value>::type 
    compute_dt(T *s)
    {
        #if GPU_CODE
        __shared__  N prim_buff[BLOCK_SIZE];
        real vPlus, vMinus;
        const real gamma     = s->gamma;
        int tid  = threadIdx.x;
        int ii   = blockDim.x * blockIdx.x + threadIdx.x;
        int aid  = ii + s->idx_active;
        real val;
        if (ii < s->active_zones)
        {
            prim_buff[tid] = s->gpu_prims[aid];
            __syncthreads();
            
            real dr  = s->coord_lattice.gpu_dx1[ii];
            real rho = prim_buff[tid].rho;
            real p   = prim_buff[tid].p;
        
            if constexpr(is_relativistic<N>::value)
            {
                // real gb  = prim_buff[tid].v;
                // real w   = std::sqrt(1 + gb * gb);
                // real v   = gb / w;
                real v = prim_buff[tid].v;
                real h = 1. + gamma * p / (rho * (gamma - 1.));
                real cs = std::sqrt(gamma * p / (rho * h));
                vPlus  = (v + cs) / (1 + v * cs);
                vMinus = (v - cs) / (1 - v * cs);
            } else {
                real v  = prim_buff[tid].v;
                real cs = std::sqrt(gamma * p / rho );
                vPlus   = (v + cs);
                vMinus  = (v - cs);
            }

            real cfl_dt = dr / (my_max(std::abs(vPlus), std::abs(vMinus)));
            s->dt_min[ii] = s->cfl * cfl_dt;
        }
        #endif
    }
    template<typename T, typename N, unsigned int blockSize>
    GPU_LAUNCHABLE typename std::enable_if<is_1D_primitive<N>::value>::type 
    dtWarpReduce(T *s)
    {
        #if GPU_CODE
        real min = INFINITY;
        __shared__ volatile real dt_buff[BLOCK_SIZE];

        int tid = threadIdx.x;
        int ii  = blockDim.x * blockIdx.x + threadIdx.x;
        int aid = ii + s->idx_active;
        real val;
        if (ii < s->active_zones)
        {
            // tail part
            int bidx = 0;
            for(int i = 1; bidx + tid < s->active_zones; i++)
            {
                val = s->dt_min[tid + bidx];
                min = (val ==0 || val > min) ? min : val;
                bidx = i * blockDim.x;
            }

            // previously reduced MIN part
            bidx = 0;
            int i;
            // real min = dt_buff[tid];
            for(i = 1; bidx + tid < gridDim.x; i++)
            {
                val  = s->dt_min[tid + bidx];
                min  = (val == 0 || val > min) ? min : val;
                bidx = i * blockDim.x;
            }

            dt_buff[tid] = min;
            __syncthreads();

            if (tid < 32)
            {
                warpReduceMin<blockSize>(dt_buff, tid);
            }
            if(tid == 0) 
            {
                s->dt_min[blockIdx.x] = dt_buff[0]; // dt_min[0] == minimum
                s->dt                 = s->dt_min[0];
            }
        }
        #endif
    }; // end dtWarpReduce

    

    template<typename T, typename N>
    GPU_LAUNCHABLE  typename std::enable_if<is_2D_primitive<N>::value>::type 
    compute_dt(T *s, 
    const simbi::Geometry geometry, 
    luint bytes,
    real dx1, 
    real dx2 , 
    real rmin, 
    real rmax,
    real x2min,
    real x2max)
    {
        #if GPU_CODE
        extern __shared__ N prim_buff[];
        real cfl_dt;
        const real gamma = s->gamma;
        real val;

        const luint tx  = threadIdx.x;
        const luint ty  = threadIdx.y;
        const luint tid = blockDim.x * ty + tx;
        const luint ii  = blockDim.x * blockIdx.x + threadIdx.x;
        const luint jj  = blockDim.y * blockIdx.y + threadIdx.y;
        const luint ia  = ii + s->idx_active;
        const luint ja  = jj + s->idx_active;
        const luint aid = (col_maj) ? ia * s-> ny + ja : ja * s->nx + ia;
        if ((ii < s->xphysical_grid) && (jj < s->yphysical_grid))
        {
             prim_buff[tid] = s->gpu_prims[aid];
             __syncthreads();
            real plus_v1 , plus_v2 , minus_v1, minus_v2;

            real rho  = prim_buff[tid].rho;
            real p    = prim_buff[tid].p;
            real v1   = prim_buff[tid].v1;
            real v2   = prim_buff[tid].v2;

            if constexpr(is_relativistic<N>::value)
            {
                real h   = 1 + gamma * p / (rho * (gamma - 1));
                real cs  = sqrt(gamma * p / (rho * h));
                plus_v1  = (v1 + cs) / (static_cast<real>(1.0) + v1 * cs);
                plus_v2  = (v2 + cs) / (static_cast<real>(1.0) + v2 * cs);
                minus_v1 = (v1 - cs) / (static_cast<real>(1.0) - v1 * cs);
                minus_v2 = (v2 - cs) / (static_cast<real>(1.0) - v2 * cs);
            } else {
                real cs  = sqrt(gamma * p / rho);
                plus_v1  = (v1 + cs);
                plus_v2  = (v2 + cs);
                minus_v1 = (v1 - cs);
                minus_v2 = (v2 - cs);
            }
        
            switch (geometry)
            {
                case simbi::Geometry::CARTESIAN:
                    cfl_dt = my_min(dx1 / (my_max(std::abs(plus_v1), std::abs(minus_v1))),
                                    dx2 / (my_max(std::abs(plus_v2), std::abs(minus_v2))));
                    break;
                
                case simbi::Geometry::SPHERICAL:
                    // Compute avg spherical distance 3/4 *(rf^4 - ri^4)/(rf^3 - ri^3)
                    const real rl           = my_max(rmin * pow(10, (ii -static_cast<real>(0.5)) * dx1), rmin);
                    const real rr           = my_min(rl * pow(10, dx1 * (ii == 0 ? 0.5 : 1.0)), rmax);
                    const real tl           = my_max(x2min + (jj - static_cast<real>(0.5)) * dx2, x2min);
                    const real tr           = my_min(tl + dx2 * (jj == 0 ? 0.5 : 1.0), x2max); 
                    const real rmean        = 0.75 * (rr * rr * rr * rr - rl * rl * rl *rl) / (rr * rr * rr - rl * rl * rl);
                    cfl_dt = my_min((rr - rl) / (my_max(std::abs(plus_v1), std::abs(minus_v1))),
                            rmean * (tr - tl) / (my_max(std::abs(plus_v2), std::abs(minus_v2))));
                    break;
            } // end switch
            
            s->dt_min[jj * s->xphysical_grid + ii] = s->cfl * cfl_dt;
            // if (jj * s->xphysical_grid + ii == 21)
            //     printf("jj: %lu, ii: %lu, dt: %.2e\n", jj, ii, s->dt_min[jj * s->xphysical_grid + ii]);
        }
        #endif
    }

    template<typename T, typename N>
    GPU_LAUNCHABLE  typename std::enable_if<is_3D_primitive<N>::value>::type 
    compute_dt(T *s, 
    const simbi::Geometry geometry, 
    luint bytes,
    real dx1, 
    real dx2,
    real dx3, 
    real rmin, 
    real rmax,
    real x2min,
    real x2max,
    real x3min,
    real x3max)
    {
        #if GPU_CODE
        extern __shared__ N prim_buff[];
        real cfl_dt;
        const real gamma = s->gamma;
        real val;

        const luint tx  = threadIdx.x;
        const luint ty  = threadIdx.y;
        const luint tz  = threadIdx.z;
        const luint tid = blockDim.x * blockDim.y * tz + blockDim.x * ty + tx;
        const luint ii  = blockDim.x * blockIdx.x + tx;
        const luint jj  = blockDim.y * blockIdx.y + ty;
        const luint kk  = blockDim.z * blockIdx.z + tz;
        const luint ia  = ii + s->idx_active;
        const luint ja  = jj + s->idx_active;
        const luint ka  = kk + s->idx_active;
        const luint aid = (col_maj) ? ia * s-> ny + ja : ka * s->nx * s->ny + ja * s->nx + ia;
        if ((ii < s->xphysical_grid) && (jj < s->yphysical_grid) && (kk < s->zphysical_grid))
        {
             prim_buff[tid] = s->gpu_prims[aid];
             __syncthreads();
            real plus_v1 , plus_v2 , minus_v1, minus_v2, plus_v3, minus_v3;
            real rho  = prim_buff[tid].rho;
            real p    = prim_buff[tid].p;
            real v1   = prim_buff[tid].v1;
            real v2   = prim_buff[tid].v2;
            real v3   = prim_buff[tid].v3;

            if constexpr(is_relativistic<N>::value)
            {
                real h   = 1 + gamma * p / (rho * (gamma - 1));
                real cs  = sqrt(gamma * p / (rho * h));
                plus_v1  = (v1 + cs) / (static_cast<real>(1.0) + v1 * cs);
                plus_v2  = (v2 + cs) / (static_cast<real>(1.0) + v2 * cs);
                plus_v3  = (v3 + cs) / (static_cast<real>(1.0) + v3 * cs);
                minus_v1 = (v1 - cs) / (static_cast<real>(1.0) - v1 * cs);
                minus_v2 = (v2 - cs) / (static_cast<real>(1.0) - v2 * cs);
                minus_v3 = (v3 - cs) / (static_cast<real>(1.0) - v3 * cs);
            } else {
                real cs  = sqrt(gamma * p / rho);
                plus_v1  = (v1 + cs);
                plus_v2  = (v2 + cs);
                plus_v3  = (v3 + cs);
                minus_v1 = (v1 - cs);
                minus_v2 = (v2 - cs);
                minus_v3 = (v3 - cs);
            }
        
            switch (geometry)
            {
                case simbi::Geometry::CARTESIAN:
                    cfl_dt = my_min(dx1 / (my_max(std::abs(plus_v1), std::abs(minus_v1))),
                                    dx2 / (my_max(std::abs(plus_v2), std::abs(minus_v2))));
                    cfl_dt = my_min(cfl_dt, dx3 / (my_max(std::abs(plus_v3), std::abs(minus_v3))));
                    break;
                
                case simbi::Geometry::SPHERICAL:
                    // Compute avg spherical distance 3/4 *(rf^4 - ri^4)/(rf^3 - ri^3)
                    const real rl           = my_max(rmin * pow(10, (ii -static_cast<real>(0.5)) * dx1), rmin);
                    const real rr           = my_min(rl * pow(10, dx1 * (ii == 0 ? 0.5 : 1.0)), rmax);
                    const real tl           = my_max(x2min + (jj - static_cast<real>(0.5)) * dx2, x2min);
                    const real tr           = my_min(tl + dx2 * (jj == 0 ? 0.5 : 1.0), x2max); 
                    const real ql           = my_max(x3min + (kk - static_cast<real>(0.5)) * dx3, x3min);
                    const real qr           = my_min(ql + dx3 * (kk == 0 ? 0.5 : 1.0), x3max); 
                    const real rmean        = static_cast<real>(0.75) * (rr * rr * rr * rr - rl * rl * rl *rl) / (rr * rr * rr - rl * rl * rl);
                    const real th           = static_cast<real>(0.5) * (tl + tr);
                    cfl_dt = my_min((rr - rl) / (my_max(std::abs(plus_v1), std::abs(minus_v1))),
                            rmean * (tr - tl) / (my_max(std::abs(plus_v2), std::abs(minus_v2))));
                    cfl_dt = my_min(cfl_dt, rmean * std::sin(th) * (qr - ql)  / (my_max(std::abs(plus_v3), std::abs(minus_v3))));
                    break;
            } // end switch
            
            s->dt_min[kk * s->xphysical_grid * s->yphysical_grid + jj * s->xphysical_grid + ii] = s->cfl * cfl_dt;
        }
        #endif
    }

    template<typename T, typename N, unsigned int blockSize>
    GPU_LAUNCHABLE typename std::enable_if<is_2D_primitive<N>::value>::type
    dtWarpReduce(T *s)
    {
        #if GPU_CODE
        real val;
        const real gamma     = s->gamma;
        extern __shared__ volatile real dt_buff[];
        real min = INFINITY;
        const luint tx  = threadIdx.x;
        const luint ty  = threadIdx.y;
        const luint tid = blockDim.x * ty + tx;
        const luint ii  = blockDim.x * blockIdx.x + threadIdx.x;
        const luint jj  = blockDim.y * blockIdx.y + threadIdx.y;
        const luint zones = s->xphysical_grid * s->yphysical_grid;
        if ((ii < s->xphysical_grid) && (jj < s->yphysical_grid))
        {
            // tail part
            int bidx = 0;
            for(int i = 1; bidx + tid  < zones; i++)
            {
                val = s->dt_min[tid + bidx];
                // GPUs are a bit strange...If I don't include the erroneous if statement then val will occasionally be <= 0....
                if (val <= 0) 
                    printf("idx: %lu, val: %f, dt: %f", tid + bidx, val, s->dt_min[tid + bidx]);
                min = (val <= 0 || val > min) ? min : val;
                bidx = i * blockDim.x * blockDim.y;
            }
            // previously reduced MIN part
            bidx = 0;
            int i;
            for(i = 1; bidx + tid < gridDim.x*gridDim.y; i++)
            {
                val  = s->dt_min[tid + bidx];
                min  = (val <= 0 || val > min)  ? min : val;
                bidx = i * blockDim.x * blockDim.y;
            }

            dt_buff[tid] = min;
            __syncthreads();

            if (tid < 32)
            {
                warpReduceMin<blockSize>(dt_buff, tid);
            }
            if(tid == 0)
            {
                // if (dt_buff[0] < 1e-10)
                //     printf("dt: %f\n", dt_buff[0]);
                s->dt_min[blockIdx.x + blockIdx.y * gridDim.x] = dt_buff[0]; // dt_min[0] == minimum
                s->dt = s->dt_min[0];
            }
        } // end if
    #endif

    }; // end dtWarpReduce

    template<typename T, typename N, unsigned int blockSize>
    GPU_LAUNCHABLE typename std::enable_if<is_3D_primitive<N>::value>::type
    dtWarpReduce(T *s)
    {
        #if GPU_CODE
        real val;
        const real gamma     = s->gamma;
        extern __shared__ volatile real dt_buff[];
        real min = INFINITY;
        const luint tx  = threadIdx.x;
        const luint ty  = threadIdx.y;
        const luint tz  = threadIdx.z;
        const luint tid = blockDim.x * blockDim.y * tz + blockDim.x * ty + tx;
        const luint ii  = blockDim.x * blockIdx.x + threadIdx.x;
        const luint jj  = blockDim.y * blockIdx.y + threadIdx.y;
        const luint kk  = blockDim.z * blockIdx.z + threadIdx.z;
        const luint zones = s->zphysical_grid * s->xphysical_grid * s->yphysical_grid;
        if ((ii < s->xphysical_grid) && (jj < s->yphysical_grid) && (kk < s->zphysical_grid))
        {
            // tail part
            int bidx = 0;
            for(int i = 1; bidx + tid  < zones; i++)
            {
                val = s->dt_min[tid + bidx];
                min = (val <= 0 || val > min) ? min : val;
                bidx = i * blockDim.x * blockDim.y * blockDim.z;
            }
            // previously reduced MIN part
            bidx = 0;
            int i;
            for(i = 1; bidx + tid < gridDim.x*gridDim.y; i++)
            {
                val  = s->dt_min[tid + bidx];
                min  = (val <= 0 || val > min) ? min : val;
                bidx = i * blockDim.x * blockDim.y * blockDim.z;
            }

            dt_buff[tid] = min;
            __syncthreads();

            if (tid < 32)
            {
                warpReduceMin<blockSize>(dt_buff, tid);
            }
            if(tid == 0)
            {
                s->dt_min[blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y] = dt_buff[0]; // dt_min[0] == minimum
                s->dt = s->dt_min[0];
            }
        } // end if
    #endif

    }; // end dtWarpReduce
}

