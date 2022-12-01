#ifndef HELPERS_HIP_HPP
#define HELPERS_HIP_HPP

#include "build_options.hpp"
#include "common/enums.hpp"
#include "common/hydro_structs.hpp"
#include "common/helpers.hpp"
#include "hydro/euler1D.hpp"
#include "hydro/euler2D.hpp"
#include "hydro/srhydro1D.hip.hpp"
#include "hydro/srhydro2D.hip.hpp"
#include "hydro/srhydro3D.hip.hpp"
namespace simbi
{
    //======================================
    //          GPU TEMPLATES
    //======================================
    template<typename T, typename U, typename V>
    GPU_LAUNCHABLE  typename std::enable_if<is_1D_primitive<T>::value>::type 
    compute_dt(U *s, const V* prim_buffer, real* dt_min);

    template<typename T, typename U, typename V>
    GPU_LAUNCHABLE  typename std::enable_if<is_2D_primitive<T>::value>::type 
    compute_dt(U *s, 
    const V* prim_buffer,
    real *dt_min,
    const simbi::Geometry geometry, 
    luint bytes,
    real dx1, 
    real dx2 , 
    real rmin = 0, 
    real rmax = 1,
    real x2min = 0,
    real x2max = 1);

    template<typename T, typename U, typename V>
    GPU_LAUNCHABLE  typename std::enable_if<is_3D_primitive<T>::value>::type 
    compute_dt(U *s, 
    const V* prim_buffer,
    real *dt_min,
    const simbi::Geometry geometry, 
    luint bytes,
    real dx1, 
    real dx2,
    real dx3, 
    real rmin  = 0, 
    real rmax  = 0,
    real x2min = 0,
    real x2max = 0,
    real x3min = 0,
    real x3max = 0);

    //======================================
    //              HELPER OVERLOADS
    //======================================
    GPU_CALLABLE_INLINE
    bool quirk_strong_shock(real pl, real pr)
    {
        return std::abs(pr - pl) / helpers::my_min(pl, pr) > QUIRK_THRESHOLD;
    }

    GPU_CALLABLE_INLINE
    constexpr unsigned int kronecker(luint i, luint j) { return (i == j ? 1 : 0); }

    GPU_CALLABLE_INLINE
    auto get_2d_idx(const luint ii, const luint jj, const luint nx, const luint ny){
        if constexpr(col_maj) {
            return  ii * ny + jj;
        }
        return jj * nx + ii;
    }
    
    void config_ghosts1D(
        const ExecutionPolicy<> p,
        SRHD::conserved_t *cons, 
        const int grid_size,
        const bool first_order, 
        const simbi::BoundaryCondition boundary_condition,
        const sr1d::Conserved *outer_zones = nullptr);

    void config_ghosts1D(
        const ExecutionPolicy<> p,
        Newtonian1D::conserved_t *cons, 
        const int grid_size,
        const bool first_order, 
        const simbi::BoundaryCondition boundary_condition,
        const hydro1d::Conserved *outer_zones = nullptr);
        
    void config_ghosts2D(
        const ExecutionPolicy<> p,
        SRHD2D::conserved_t *cons, 
        const int x1grid_size, 
        const int x2grid_size, 
        const bool first_order,
        const simbi::BoundaryCondition boundary_condition,
        const sr2d::Conserved *outer_zones = nullptr,
        const bool half_sphere = true);

    void config_ghosts2D(
        const ExecutionPolicy<> p,
        Newtonian2D::conserved_t *cons, 
        const int x1grid_size, 
        const int x2grid_size, 
        const bool first_order,
        const simbi::BoundaryCondition boundary_condition,
        const hydro2d::Conserved *outer_zones = nullptr,
        const bool half_sphere = true);

    void config_ghosts3D(
        const ExecutionPolicy<> p,
        SRHD3D::conserved_t *cons, 
        const int x1grid_size, 
        const int x2grid_size,
        const int x3grid_size,  
        const bool first_order,
        const simbi::BoundaryCondition boundary_condition,
        const bool half_sphere = true);

    inline GPU_DEV real warpReduceMin(real val) {
        #if CUDA_CODE
        int mask = __match_any_sync(__activemask(), val);
        for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
            real next_val = __shfl_down_sync(mask, val, offset);
            val           = (val < next_val) ? val : next_val;
        }
        return val;
        #elif HIP_CODE
        for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
            real next_val = __shfl_down(val, offset);
            val = (val < next_val) ? val : next_val;
        }
        return val;
        #else 
        return 0.0;
        #endif
    };

    inline GPU_DEV real blockReduceMin(real val) {
        #if GPU_CODE
        __shared__ real shared[WARP_SIZE]; // Shared mem for 32 (Nvidia) / 64 (AMD) partial mins
        const int tid = threadIdx.z * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
        const int bsz = blockDim.x * blockDim.y * blockDim.z;
        int lane      = tid % WARP_SIZE;
        int wid       = tid / WARP_SIZE;

        val = warpReduceMin(val);     // Each warp performs partial reduction

        if (lane==0) 
            shared[wid] = val; // Write reduced value to shared memory
        __syncthreads();       // Wait for all partial reductions

        //read from shared memory only if that warp existed
        val = (tid < bsz / WARP_SIZE) ? shared[lane] : val;

        if (wid==0) 
            val = warpReduceMin(val); //Final reduce within first warp
        return val;
        #else 
        return 0.0;
        #endif
    };

    template<typename T>
    GPU_LAUNCHABLE void deviceReduceKernel(T *self, lint nmax);

    void anyDisplayProps();

    /**
     * @brief Get the Flops countin GB / s
     * 
     * @tparam T Consrved type
     * @tparam U Primitive type
     * @param radius halo radius
     * @param total_zones total number of zones in sim
     * @param real_zones total number of active zones in mesh
     * @param delta_t time for event completion
     * @return float
     */
    template<typename T, typename U>
    inline real getFlops(
        const luint radius,
        const luint total_zones, 
        const luint real_zones,
        const float delta_t
    ) {
        const float advance_contr    = total_zones * radius * sizeof(T) * (1.0 + 4.0 * radius);
        const float cons2prim_contr  = total_zones * radius * sizeof(U);
        const float ghost_conf_contr = (total_zones - real_zones)  * radius * sizeof(T);
        return (advance_contr + cons2prim_contr + ghost_conf_contr) / (delta_t * 1e9);
    }


} // end simbi

#include "helpers.hip.tpp"
#endif