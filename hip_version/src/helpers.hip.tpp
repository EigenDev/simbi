
namespace simbi{
    template<unsigned int blockSize>
    __device__ void warpReduceMin(volatile real smem[BLOCK_SIZE], int tid)
    {
        if (blockSize >= 64) smem[tid] = (smem[tid] < smem[tid + 32]) ? smem[tid] : smem[tid + 32];
        if (blockSize >= 32) smem[tid] = (smem[tid] < smem[tid + 16]) ? smem[tid] : smem[tid + 16];
        if (blockSize >= 16) smem[tid] = (smem[tid] < smem[tid +  8]) ? smem[tid] : smem[tid +  8];
        if (blockSize >=  8) smem[tid] = (smem[tid] < smem[tid +  4]) ? smem[tid] : smem[tid +  4];
        if (blockSize >=  4) smem[tid] = (smem[tid] < smem[tid +  2]) ? smem[tid] : smem[tid +  2];
        if (blockSize >=  2) smem[tid] = (smem[tid] < smem[tid +  1]) ? smem[tid] : smem[tid +  1];
    };

    template<typename T, typename N, unsigned int blockSize>
    __global__ typename std::enable_if<is_1D_primitive<N>::value>::type 
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

            // printf("[%d] dt_min: %f, dt_buf: %f, cfl_dt: %f\n",blockIdx.x, s->dt_min[blockIdx.x], dt_buff[tid], s->CFL * cfl_dt);

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
                // printf("dt_buff[%d] = %f\n", tid + 0, dt_buff[tid + 0]);
                // printf("dt_buff[%d] = %f\n", tid + 1, dt_buff[tid + 1]);
                // printf("dt_buff[%d] = %f\n", tid + 2, dt_buff[tid + 2]);
                // printf("dt_buff[%d] = %f\n", tid + 3, dt_buff[tid + 3]);
                //printf("dt_min[%d] = %f", blockIdx.x, s->dt_min[blockIdx.x]);
            }
            
        }
    }; // end dtWarpReduce

    template <typename T>
    __global__ void config_ghosts1DGPU(T *dev_sim, int grid_size, bool first_order){
        sr1d::Conserved *u_state = dev_sim->gpu_cons;
        if (first_order){
            u_state[0].D = u_state[1].D;
            u_state[grid_size - 1].D = u_state[grid_size - 2].D;

            u_state[0].S = - u_state[1].S;
            u_state[grid_size - 1].S = u_state[grid_size - 2].S;

            u_state[0].tau = u_state[1].tau;
            u_state[grid_size - 1].tau = u_state[grid_size - 2].tau;
        } else {
            u_state[0].D = u_state[3].D;
            u_state[1].D = u_state[2].D;
            u_state[grid_size - 1].D = u_state[grid_size - 3].D;
            u_state[grid_size - 2].D = u_state[grid_size - 3].D;

            u_state[0].S = - u_state[3].S;
            u_state[1].S = - u_state[2].S;
            u_state[grid_size - 1].S = u_state[grid_size - 3].S;
            u_state[grid_size - 2].S = u_state[grid_size - 3].S;

            u_state[0].tau = u_state[3].tau;
            u_state[1].tau = u_state[2].tau;
            u_state[grid_size - 1].tau = u_state[grid_size - 3].tau;
            u_state[grid_size - 2].tau = u_state[grid_size - 3].tau;
        }
    };
}

