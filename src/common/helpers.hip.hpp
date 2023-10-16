#ifndef HELPERS_HIP_HPP
#define HELPERS_HIP_HPP

#include <vector>
#include <iostream>
#include <cmath>
#include <map>
#include <memory>
#include "build_options.hpp"
#include "common/enums.hpp"
#include "common/traits.hpp"
#include "util/exec_policy.hpp"


// Some useful global constants
constexpr real QUIRK_THRESHOLD = 1e-4;

// Calculation derived from: https://developer.nvidia.com/blog/how-implement-performance-metrics-cuda-cc/
extern real gpu_theoretical_bw; //  = 1875e6 * (192.0 / 8.0) * 2 / 1e9;

namespace simbi
{
    namespace helpers
    {

        class InterruptException : public std::exception
        {
            public:
            InterruptException(int s);
            const char* what() const noexcept;
            int status;
        };

        class SimulationFailureException : public std::exception
        {
            public:
            SimulationFailureException(const char* reason, const char* details);
            const char* what() const noexcept;
            const std::string reason;
            const std::string details;
        };
        
        /*
        Catch keyboard Ctrl+C and other signals and pass
        back to python to kill
        */
        void catch_signals();
        
        /*
        @param a
        @param b 
        @return maximum between values a and b
        */
        template<typename T>
        GPU_CALLABLE_INLINE
        constexpr T my_max(const T a, const T b) {
            return a > b ? a : b;
        }

        /*
        @param a
        @param b 
        @return minimum between values a and b
        */
        template<typename T>
        GPU_CALLABLE_INLINE
        constexpr T my_min(const T a, const T b) {
            return a < b ? a : b;
        }

        /*
        @param a
        @param b 
        @param c
        @return maximum between values a, b, cand c
        */
        template<typename T>
        GPU_CALLABLE_INLINE
        constexpr T my_max3(const T a, const T b, const T c) {
            return (a > b) ? (a > c ? a : c) : b > c ? b : c;
        }

        /*
        @param a
        @param b 
        @param c
        @return minimum between values a, b, and c
        */
        template<typename T>
        GPU_CALLABLE_INLINE
        constexpr T my_min3(const T a, const T b, const T c) {
            return (a < b) ? (a < c ? a : c) : b < c ? b : c;
        }
        
        /*
        @param index the index you want to shift to
        @param size the total size of the array
        @return the peridoically shifted index, accounting for negative values
        */
        GPU_CALLABLE_INLINE lint mod(const lint index, const lint size)
        {
            return (index % size + size) % size;
        }

        const std::map<std::string, simbi::Geometry> geometry_map = {
        { "spherical", simbi::Geometry::SPHERICAL },
        { "cartesian", simbi::Geometry::CARTESIAN},
        { "planar_cylindrical", simbi::Geometry::PLANAR_CYLINDRICAL},
        { "axis_cylindrical", simbi::Geometry::AXIS_CYLINDRICAL},
        { "cylindtical", simbi::Geometry::CYLINDRICAL}
        };

        const std::map<std::string, simbi::BoundaryCondition> boundary_cond_map = {
        { "inflow", simbi::BoundaryCondition::INFLOW},
        { "outflow", simbi::BoundaryCondition::OUTFLOW},
        { "reflecting", simbi::BoundaryCondition::REFLECTING},
        { "periodic", simbi::BoundaryCondition::PERIODIC}
        };

        const std::map<std::string, simbi::Solver> solver_map = {
        { "hllc", simbi::Solver::HLLC},
        { "hlle", simbi::Solver::HLLE}
        };
        //---------------------------------------------------------------------------------------------------------
        //  HELPER-TEMPLATES
        //---------------------------------------------------------------------------------------------------------
        //-------------Define Function Templates-------------------------
        template <typename T, size_t N>
        constexpr size_t array_size(T (&)[N]);

        template<typename T, typename U>
        typename std::enable_if<is_3D_primitive<U>::value>::type
        writeToProd(T *from, PrimData *to);

        //Handle 2D primitive arrays whether SR or Newtonian
        template<typename T, typename U>
        typename std::enable_if<is_2D_primitive<U>::value>::type
        writeToProd(T *from, PrimData *to);

        template<typename T, typename U>
        typename std::enable_if<is_1D_primitive<U>::value>::type
        writeToProd(T *from, PrimData *to);

        template<typename T , typename U, typename arr_type>
        typename std::enable_if<is_3D_primitive<U>::value, T>::type
        vec2struct(const arr_type &p);

        template<typename T , typename U, typename arr_type>
        typename std::enable_if<is_2D_primitive<U>::value, T>::type
        vec2struct(const arr_type &p);

        template<typename T , typename U, typename arr_type>
        typename std::enable_if<is_1D_primitive<U>::value, T>::type
        vec2struct(const arr_type &p);

        template <typename... Args>
        std::string string_format(const std::string &format, Args... args);

        template <typename T>
        GPU_CALLABLE_INLINE
        constexpr int sgn(T val) { return (T(0) < val) - (val < T(0)); }

        template<typename T, typename U, typename V, typename W, int X>
        void write_to_file(
            T *sim_state_host, 
            T *sim_state_dev, 
            W  &dual_memory_layer,
            DataWriteMembers &setup,
            const std::string data_directory,
            const real t, 
            const real t_interval, 
            const real chkpt_interval, 
            const luint chkpt_zone_label);

        //---------------------------------------------------------------------------------------------------------
        //  HELPER-METHODS
        //---------------------------------------------------------------------------------------------------------
        //----------------Define Methods-------------------------
        std::string create_step_str(real t_interval, std::string &tnow);
        void write_hdf5(
            const std::string data_directory, 
            const std::string filename, 
            const PrimData prims, 
            const DataWriteMembers system, 
            const int dim, 
            const int size);

        //-------------------Inline for Speed--------------------------------------
        GPU_CALLABLE_INLINE real minmod(const real x, const real y, const real z)
        {
            return static_cast<real>(0.25) * std::abs(sgn(x) + sgn(y)) * (sgn(x) + sgn(z)) * my_min3(std::abs(x), std::abs(y), std::abs(z));
        };

        GPU_CALLABLE_INLINE
        real calc_any_mean(const real a, const real b, const simbi::Cellspacing cellspacing) 
        {
            switch (cellspacing)
            {
            case simbi::Cellspacing::LOGSPACE:
                return std::sqrt(a * b);
            case simbi::Cellspacing::LINSPACE:
                return 0.5 * (a + b);
            default:
                return INFINITY;
            }
        }

        template<typename T>
        GPU_CALLABLE_INLINE typename std::enable_if<is_3D_primitive<T>::value, T>::type
        plm_gradient(const T &a, const T &b, const T &c, const real plm_theta)
        {
            const real rho = minmod((a - b).rho * plm_theta, (c - b).rho * static_cast<real>(0.5), (c - a).rho * plm_theta);
            const real v1  = minmod((a - b).v1  * plm_theta, (c - b).v1  * static_cast<real>(0.5), (c - a).v1  * plm_theta);
            const real v2  = minmod((a - b).v2  * plm_theta, (c - b).v2  * static_cast<real>(0.5), (c - a).v2  * plm_theta);
            const real v3  = minmod((a - b).v3  * plm_theta, (c - b).v3  * static_cast<real>(0.5), (c - a).v3  * plm_theta);
            const real pre = minmod((a - b).p   * plm_theta, (c - b).p   * static_cast<real>(0.5), (c - a).p   * plm_theta);
            const real chi = minmod((a - b).chi * plm_theta, (c - b).chi * static_cast<real>(0.5), (c - a).chi * plm_theta);
            return T{rho, v1, v2, v3, pre, chi};
        }

        template<typename T>
        GPU_CALLABLE_INLINE typename std::enable_if<is_2D_primitive<T>::value, T>::type
        plm_gradient(const T &a, const T &b, const T &c, const real plm_theta)
        {
            const real rho = minmod((a - b).rho * plm_theta, (c - b).rho * static_cast<real>(0.5), (c - a).rho * plm_theta);
            const real v1  = minmod((a - b).v1  * plm_theta, (c - b).v1  * static_cast<real>(0.5), (c - a).v1  * plm_theta);
            const real v2  = minmod((a - b).v2  * plm_theta, (c - b).v2  * static_cast<real>(0.5), (c - a).v2  * plm_theta);
            const real pre = minmod((a - b).p   * plm_theta, (c - b).p   * static_cast<real>(0.5), (c - a).p   * plm_theta);
            const real chi = minmod((a - b).chi * plm_theta, (c - b).chi * static_cast<real>(0.5), (c - a).chi * plm_theta);
            return T{rho, v1, v2, pre, chi};
        }

        template<typename T>
        GPU_CALLABLE_INLINE typename std::enable_if<is_1D_primitive<T>::value, T>::type
        plm_gradient(const T &a, const T &b, const T &c, const real plm_theta)
        {
            const real rho = minmod((a - b).rho * plm_theta, (c - b).rho * static_cast<real>(0.5), (c - a).rho * plm_theta);
            const real v   = minmod((a - b).v1  * plm_theta, (c - b).v1  * static_cast<real>(0.5), (c - a).v1  * plm_theta);
            const real pre = minmod((a - b).p   * plm_theta, (c - b).p   * static_cast<real>(0.5), (c - a).p   * plm_theta);
            const real chi = minmod((a - b).chi * plm_theta, (c - b).chi * static_cast<real>(0.5), (c - a).chi * plm_theta);
            return T{rho, v, pre, chi};
        }

        GPU_CALLABLE_INLINE 
        constexpr lint get_real_idx(const lint idx, const lint offset, const lint active_zones) 
        {
            if (idx > active_zones - 1 + offset) {
                return active_zones - 1;
            }
            return (idx - offset > 0) * (idx - offset);
        }

        inline double sigmoid(const double t, const double tduration, const double time_step, const bool constant_sources) {
            if (constant_sources) {
                return 1 / time_step;
            }
            return 1 / (1 + std::exp(static_cast<real>(10.0) * (t - tduration)));
        }

        GPU_CALLABLE_INLINE real newton_f(real gamma, real tau, real D, real S, real p) {
            const auto et    = tau + D + p;
            const auto v2    = S * S / (et * et);
            const auto W     = 1 / std::sqrt(1 - v2);
            const auto rho   = D / W;
            const auto eps   = (tau + (1 - W) * D + (1 - W * W) * p) / (D * W);
            return (gamma - 1) * rho * eps - p;
        }

        GPU_CALLABLE_INLINE real newton_g(real gamma, real tau, real D, real S, real p) {
            const auto et    = tau + D + p;
            const auto v2    = S * S / (et * et);
            const auto W     = 1 / std::sqrt(1 - v2);
            const auto eps   = (tau + (1 - W) * D + (1 - W * W) * p) / (D * W);
            const auto c2    = (gamma - 1) * gamma * eps / (1 + gamma * eps);
            return c2 * v2 - 1;
        }

        //======================================
        //          GPU TEMPLATES
        //======================================
        template<typename T, TIMESTEP_TYPE dt_type = TIMESTEP_TYPE::ADAPTIVE, typename U, typename V>
        GPU_LAUNCHABLE  typename std::enable_if<is_1D_primitive<T>::value>::type 
        compute_dt(U *s, const V* prim_buffer, real* dt_min);

        template<typename T, TIMESTEP_TYPE dt_type = TIMESTEP_TYPE::ADAPTIVE, typename U, typename V>
        GPU_LAUNCHABLE  typename std::enable_if<is_2D_primitive<T>::value>::type 
        compute_dt(U *s, 
        const V* prim_buffer,
        real *dt_min,
        const simbi::Geometry geometry);

        template<typename T, TIMESTEP_TYPE dt_type = TIMESTEP_TYPE::ADAPTIVE, typename U, typename V>
        GPU_LAUNCHABLE  typename std::enable_if<is_3D_primitive<T>::value>::type 
        compute_dt(U *s, 
        const V* prim_buffer,
        real *dt_min,
        const simbi::Geometry geometry);

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
        
        GPU_CALLABLE_INLINE
        auto get_cell_centroid(const real xr, const real xl, const simbi::Geometry geometry) {
            switch (geometry)
            {
            case Geometry::SPHERICAL:
                return static_cast<real>(0.75) * (xr * xr * xr * xr - xl * xl * xl * xl) / (xr * xr * xr - xl * xl * xl);
            default:
                return static_cast<real>(2.0/3.0) * (xr * xr * xr - xl * xl * xl) / (xr * xr - xl * xl);
            }
        }


        template<typename T, typename U>
        void config_ghosts1D(
            const ExecutionPolicy<> p,
            T *cons, 
            const int grid_size,
            const bool first_order, 
            const simbi::BoundaryCondition* boundary_conditions,
            const U *outer_zones = nullptr,
            const U *inflow_zones = nullptr);

        template<typename T, typename U>
        void config_ghosts2D(
            const ExecutionPolicy<> p,
            T *cons, 
            const int x1grid_size, 
            const int x2grid_size, 
            const bool first_order,
            const simbi::Geometry geometry,
            const simbi::BoundaryCondition *boundary_conditions,
            const U *outer_zones,
            const U *boundary_zones,
            const bool half_sphere);

        template<typename T, typename U>
        void config_ghosts3D(
            const ExecutionPolicy<> p,
            T *cons, 
            const int x1grid_size, 
            const int x2grid_size,
            const int x3grid_size,  
            const bool first_order,
            const simbi::BoundaryCondition* boundary_conditions,
            const U* inflow_zones,
            const bool half_sphere,
            const simbi::Geometry geometry);

        inline GPU_DEV real warpReduceMin(real val) {
            #if CUDA_CODE
            // Adapted from https://stackoverflow.com/a/59883722/13874039
            // to work with older cuda versions
            int mask;
            #if __CUDA_ARCH__ >= 700
                mask = __match_any_sync(__activemask(), val);
            #else
                unsigned tmask = __activemask();
                for (int i = 0; i < WARP_SIZE; i++){
                    unsigned long long tval = __shfl_sync(tmask, (unsigned long long)val, i);
                    unsigned my_mask = __ballot_sync(tmask, (tval == (unsigned long long)val));
                    if (i == (threadIdx.x & (WARP_SIZE-1))) 
                        mask = my_mask;
                }
            #endif
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

        template<typename T>
        GPU_LAUNCHABLE void deviceReduceWarpAtomicKernel(T *self, lint nmax);

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
            const float advance_contr    = total_zones * sizeof(T) * (1.0 + 4.0 * radius);
            const float cons2prim_contr  = total_zones * sizeof(U);
            const float ghost_conf_contr = (total_zones - real_zones) * sizeof(T);
            return (advance_contr + cons2prim_contr + ghost_conf_contr) / (delta_t * 1e9);
        }

        #if GPU_CODE
        __device__ __forceinline__ real atomicMinReal (real * addr, real value) {
            real old = __int_as_real(atomicMin((atomic_cast *)addr, __real_as_int(value)));
            return old;
        }
        #endif 
        
    } // namespace helpers
} // end simbi

#include "helpers.hip.tpp"
#endif