
namespace simbi{
    template<unsigned int blockSize>
    GPU_DEV void warpReduceMin(volatile real smem[BLOCK_SIZE], int tid)
    {
        if (blockSize >= 64) smem[tid] = (smem[tid] < smem[tid + 32]) ? smem[tid] : smem[tid + 32];
        if (blockSize >= 32) smem[tid] = (smem[tid] < smem[tid + 16]) ? smem[tid] : smem[tid + 16];
        if (blockSize >= 16) smem[tid] = (smem[tid] < smem[tid +  8]) ? smem[tid] : smem[tid +  8];
        if (blockSize >=  8) smem[tid] = (smem[tid] < smem[tid +  4]) ? smem[tid] : smem[tid +  4];
        if (blockSize >=  4) smem[tid] = (smem[tid] < smem[tid +  2]) ? smem[tid] : smem[tid +  2];
        if (blockSize >=  2) smem[tid] = (smem[tid] < smem[tid +  1]) ? smem[tid] : smem[tid +  1];
    };

    template<unsigned int blockSize>
    GPU_DEV void warpReduceMin(volatile real smem[BLOCK_SIZE2D * BLOCK_SIZE2D], const int tx, const int ty)
    {
        const int tid = blockDim.x * ty + tx;
        if (blockSize >= 64) smem[tid] = (smem[tid] < smem[tid + 32]) ? smem[tid] : smem[tid + 32];
        if (blockSize >= 32) smem[tid] = (smem[tid] < smem[tid + 16]) ? smem[tid] : smem[tid + 16];
        if (blockSize >= 16) smem[tid] = (smem[tid] < smem[tid +  8]) ? smem[tid] : smem[tid +  8];
        if (blockSize >=  8) smem[tid] = (smem[tid] < smem[tid +  4]) ? smem[tid] : smem[tid +  4];
        if (blockSize >=  4) smem[tid] = (smem[tid] < smem[tid +  2]) ? smem[tid] : smem[tid +  2];
        if (blockSize >=  2) smem[tid] = (smem[tid] < smem[tid +  1]) ? smem[tid] : smem[tid +  1];
    };

    template<unsigned int blockSize>
    GPU_DEV void warpReduceMin(volatile real smem[BLOCK_SIZE2D * BLOCK_SIZE2D], const int tx, const int ty, const int tz)
    {
        const int tid = blockDim.x * blockDim.y * tz + blockDim.x * ty + tx;
        if (blockSize >= 64) smem[tid] = (smem[tid] < smem[tid + 32]) ? smem[tid] : smem[tid + 32];
        if (blockSize >= 32) smem[tid] = (smem[tid] < smem[tid + 16]) ? smem[tid] : smem[tid + 16];
        if (blockSize >= 16) smem[tid] = (smem[tid] < smem[tid +  8]) ? smem[tid] : smem[tid +  8];
        if (blockSize >=  8) smem[tid] = (smem[tid] < smem[tid +  4]) ? smem[tid] : smem[tid +  4];
        if (blockSize >=  4) smem[tid] = (smem[tid] < smem[tid +  2]) ? smem[tid] : smem[tid +  2];
        if (blockSize >=  2) smem[tid] = (smem[tid] < smem[tid +  1]) ? smem[tid] : smem[tid +  1];
    };

    template<typename T, typename N, unsigned int blockSize>
    GPU_LAUNCHABLE typename std::enable_if<is_1D_primitive<N>::value>::type 
    dtWarpReduce(T *s)
    {
        const real gamma     = s->gamma;
        __shared__ volatile real dt_buff[BLOCK_SIZE];
        __shared__  N prim_buff[BLOCK_SIZE];

        int tid = threadIdx.x;
        int ii  = blockDim.x * blockIdx.x + threadIdx.x;
        int aid = ii + s->idx_shift;
        if (ii < s->active_zones)
        {
            prim_buff[tid] = s->gpu_prims[aid];
            __syncthreads();
            
            real dr  = s->coord_lattice.gpu_dx1[ii];
            real rho = prim_buff[tid].rho;
            real p   = prim_buff[tid].p;
            real v   = prim_buff[tid].v;

            real h = 1. + gamma * p / (rho * (gamma - 1.));
            real cs = sqrt(gamma * p / (rho * h));

            real vPLus  = (v + cs) / (1 + v * cs);
            real vMinus = (v - cs) / (1 - v * cs);

            real cfl_dt = dr / (my_max(abs(vPLus), abs(vMinus)));

            dt_buff[tid] = s->CFL * cfl_dt;
            __syncthreads();

            // printf("[%d] dt_min: %f, cfl_dt: %f\n",blockIdx.x, s->dt_min[blockIdx.x], s->CFL * cfl_dt);

            for (unsigned int stride=blockDim.x/2; stride>32; stride>>=1) 
            {   
                if (tid < stride) dt_buff[tid] = dt_buff[tid] < dt_buff[tid + stride] ? dt_buff[tid] : dt_buff[tid + stride]; 
                __syncthreads();
            }

            if (tid < 32)
            {
                warpReduceMin<blockSize>(dt_buff, tid);
            }
            if(tid == 0)
            {
                s->dt_min[blockIdx.x] = dt_buff[tid]; // dt_min[0] == minimum
                s->dt = s->dt_min[0];
            }
        }
    }; // end dtWarpReduce

    template<typename T, typename N, unsigned int blockSize>
    GPU_LAUNCHABLE typename std::enable_if<is_2D_primitive<N>::value>::type
    dtWarpReduce(T *s, const simbi::Geometry geometry)
    {
        const real gamma     = s->gamma;
        __shared__ volatile real dt_buff[BLOCK_SIZE2D * BLOCK_SIZE2D];
        __shared__  N prim_buff[BLOCK_SIZE2D][BLOCK_SIZE2D];

        real cfl_dt, rmean;
        const int tx  = threadIdx.x;
        const int ty  = threadIdx.y;
        const int tid = blockDim.x * ty + tx;
        const int ii  = blockDim.x * blockIdx.x + threadIdx.x;
        const int jj  = blockDim.y * blockIdx.y + threadIdx.y;
        const int ia  = ii + s->idx_active;
        const int ja  = jj + s->idx_active;
        const int aid = ia * s-> ny + ja;
        const int nx  = s->nx;
        const CLattice2D *coord_lattice = &(s->coord_lattice);

        // printf("%d\n", aid);
        if (aid < s->active_zones)
        {
            prim_buff[ty][tx] = s->gpu_prims[aid];
            __syncthreads();

            real dx1  = s->coord_lattice.gpu_dx1[ii];
            real dx2  = s->coord_lattice.gpu_dx2[jj];
            real rho  = prim_buff[ty][tx].rho;
            real p    = prim_buff[ty][tx].p;
            real v1   = prim_buff[ty][tx].v1;
            real v2   = prim_buff[ty][tx].v2;

            real h  = 1. + gamma * p / (rho * (gamma - 1.));
            real cs = sqrt(gamma * p / (rho * h));

            real plus_v1  = (v1 + cs) / (1. + v1 * cs);
            real plus_v2  = (v2 + cs) / (1. + v2 * cs);
            real minus_v1 = (v1 - cs) / (1. - v1 * cs);
            real minus_v2 = (v2 - cs) / (1. - v2 * cs);

            switch (geometry)
            {
                case simbi::Geometry::CARTESIAN:
                    cfl_dt = my_min(dx1 / (my_max(abs(plus_v1), abs(minus_v1))),
                                    dx2 / (my_max(abs(plus_v2), abs(minus_v2))));
                    break;
                
                case simbi::Geometry::SPHERICAL:
                    // Compute avg spherical distance 3/4 *(rf^4 - ri^4)/(rf^3 - ri^3)
                    rmean = coord_lattice->gpu_x1mean[ii];
                    cfl_dt = my_min(dx1 / (my_max(abs(plus_v1), abs(minus_v1))),
                                    rmean * dx2 / (my_max(abs(plus_v2), abs(minus_v2))));
                    break;
            } // end switch

            dt_buff[tid] = s->CFL * cfl_dt;
            __syncthreads();

            for (unsigned int stride=(blockDim.x*blockDim.y)/2; stride>32; stride>>=1) 
            {   
                if (tid < stride) dt_buff[tid] = dt_buff[tid] < dt_buff[tid + stride] ? dt_buff[tid] : dt_buff[tid + stride]; 
                __syncthreads();
            }

            if ((threadIdx.x < BLOCK_SIZE2D / 2) && (threadIdx.y < BLOCK_SIZE2D / 2))
            {
                warpReduceMin<blockSize>(dt_buff, tx, ty);
            }
            if((threadIdx.x == 0) && (threadIdx.y == 0) )
            {
                // printf("min dt: %f", s->dt_min[0]);
                s->dt_min[blockIdx.x + blockIdx.y * blockDim.x] = dt_buff[tid]; // dt_min[0] == minimum
                // printf("min dt: %f\n", s->dt_min[0]);
                s->dt = s->dt_min[0];
            }
        } // end if

    }; // end dtWarpReduce

    template<typename T, typename N, unsigned int blockSize>
    GPU_LAUNCHABLE typename std::enable_if<is_3D_primitive<N>::value>::type 
    dtWarpReduce(T *s, const simbi::Geometry geometry)
    {
        const real gamma     = s->gamma;
        __shared__ volatile real dt_buff[BLOCK_SIZE3D * BLOCK_SIZE3D * BLOCK_SIZE3D];
        __shared__  N prim_buff[BLOCK_SIZE3D][BLOCK_SIZE3D][BLOCK_SIZE3D];

        real cfl_dt, rmean;
        const int tx  = threadIdx.x;
        const int ty  = threadIdx.y;
        const int tz  = threadIdx.z;
        const int tid = blockDim.x * blockDim.y * tz + blockDim.x * ty + tx;
        const int ii  = blockDim.x * blockIdx.x + threadIdx.x;
        const int jj  = blockDim.y * blockIdx.y + threadIdx.y;
        const int kk  = blockDim.z * blockIdx.z + threadIdx.z;
        const int ia  = ii + s->idx_active;
        const int ja  = jj + s->idx_active;
        const int ka  = kk + s->idx_active;
        const int nx  = s->nx;
        const int ny  = s->ny;
        const int aid = ka * ny * nx + ja * nx + ia;
        const real sint = s->coord_lattice.gpu_sin[jj];
        
        const CLattice3D *coord_lattice = &(s->coord_lattice);
        if (aid < s->active_zones)
        {
            real dx1  = s->coord_lattice.gpu_dx1[ii];
            real dx2  = s->coord_lattice.gpu_dx2[jj];
            real dx3  = s->coord_lattice.gpu_dx3[kk];
            real rho  = s->gpu_prims[ka * nx * ny + ja * nx + ia].rho;
            real p    = s->gpu_prims[ka * nx * ny + ja * nx + ia].p;
            real v1   = s->gpu_prims[ka * nx * ny + ja * nx + ia].v1;
            real v2   = s->gpu_prims[ka * nx * ny + ja * nx + ia].v2;
            real v3   = s->gpu_prims[ka * nx * ny + ja * nx + ia].v3;

            real h  = 1. + gamma * p / (rho * (gamma - 1.));
            real cs = sqrt(gamma * p / (rho * h));

            real plus_v1  = (v1 + cs) / (1. + v1 * cs);
            real plus_v2  = (v2 + cs) / (1. + v2 * cs);
            real plus_v3  = (v3 + cs) / (1. + v3 * cs);
            real minus_v1 = (v1 - cs) / (1. - v1 * cs);
            real minus_v2 = (v2 - cs) / (1. - v2 * cs);
            real minus_v3 = (v3 - cs) / (1. - v3 * cs);

            switch (geometry)
            {
                case simbi::Geometry::CARTESIAN:
                    cfl_dt = my_min(
                                my_min(dx1 / (my_max(abs(plus_v1), abs(minus_v1))),
                                       dx2 / (my_max(abs(plus_v2), abs(minus_v2)))),
                                       dx3 / (my_max(abs(plus_v3), abs(minus_v3)))
                            );
                    break;
                
                case simbi::Geometry::SPHERICAL:
                    // Compute avg spherical distance 3/4 *(rf^4 - ri^4)/(rf^3 - ri^3)
                    real rmean = coord_lattice->gpu_x1mean[ii];
                    real rproj = rmean * sint;
                    // check if in pure r,theta plane
                    if (rproj == 0)
                            cfl_dt = my_min(dx1         / (my_max(abs(plus_v1), abs(minus_v1))),
                                            rmean * dx2 / (my_max(abs(plus_v2), abs(minus_v2))) );
                    else
                        cfl_dt = my_min(
                                    my_min(dx1         / (my_max(abs(plus_v1), abs(minus_v1))),
                                        rmean * dx2 / (my_max(abs(plus_v2), abs(minus_v2))) 
                                    ),
                                    rproj * dx3 / (my_max(abs(plus_v3), abs(minus_v3)))
                                );
                    break;
            }

        dt_buff[tid] = s->CFL * cfl_dt;
        __syncthreads();

        for (unsigned int stride=(blockDim.x*blockDim.y*blockDim.z)/2; stride>32; stride>>=1) 
        {   
            if (tid < stride) dt_buff[tid] = dt_buff[tid] < dt_buff[tid + stride] ? dt_buff[tid] : dt_buff[tid + stride]; 
            __syncthreads();
        }

        if ((threadIdx.x < BLOCK_SIZE3D / 2) && (threadIdx.y < BLOCK_SIZE3D / 2) && (threadIdx.z < BLOCK_SIZE3D / 2))
        {
            warpReduceMin<blockSize>(dt_buff, tx, ty, tz);
        }
        if((threadIdx.x == 0) && (threadIdx.y == 0) && (threadIdx.z == 0))
        {
            s->dt_min[blockDim.x * blockDim.y * blockIdx.z + blockIdx.y * blockDim.x + blockIdx.x] = dt_buff[tid]; // dt_min[0] == minimum
            s->dt = s->dt_min[0];
        }
            
        }
    }; // end dtWarpReduce
}

