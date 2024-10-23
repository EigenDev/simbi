/**
 * ***********************(C) COPYRIGHT 2024 Marcus DuPont**********************
 * @file       helpers.hpp
 * @brief      home to all helper functions used throughout library
 *
 * @note
 * @history:
 *   Version   Date            Author          Modification    Email
 *   V0.8.0    Dec-03-2023     Marcus DuPont                   md4469@nyu.edu
 *
 * @verbatim
 * ==============================================================================
 *
 * ==============================================================================
 * @endverbatim
 * ***********************(C) COPYRIGHT 2024 Marcus DuPont**********************
 */
#ifndef HELPERS_HIP_HPP
#define HELPERS_HIP_HPP

#include "build_options.hpp"   // for real, STATIC, luint, lint
#include "common/enums.hpp"    // for Geometry, BoundaryCondition, Solver
#include "common/hydro_structs.hpp"
#include "common/traits.hpp"      // for is_1D_primitive, is_2D_primitive
#include "util/exec_policy.hpp"   // for ExecutionPolicy
#include <cmath>                  // for sqrt, exp, INFINITY
#include <cstdlib>                // for abs, size_t
#include <exception>              // for exception
#include <map>                    // for map
#include <string>                 // for string, operator<=>
#include <type_traits>            // for enable_if

std::unordered_map<std::string, simbi::Cellspacing> const str2cell = {
  {"log", simbi::Cellspacing::LOGSPACE},
  {"linear", simbi::Cellspacing::LINSPACE}
  // {"log-linear",Cellspacing},
  // {"linear-log",Cellspacing},
};

std::unordered_map<simbi::Cellspacing, std::string> const cell2str = {
  {simbi::Cellspacing::LOGSPACE, "log"},
  {simbi::Cellspacing::LINSPACE, "linear"}
  // {"log-linear",Cellspacing},
  // {"linear-log",Cellspacing},
};

// forward declaration
struct PrimData;

// Some useful global constants
constexpr real QUIRK_THRESHOLD = 1e-4;

// Calculation derived from:
// https://developer.nvidia.com/blog/how-implement-performance-metrics-cuda-cc/
extern real gpu_theoretical_bw;   //  = 1875e6 * (192.0 / 8.0) * 2 / 1e9;

namespace simbi {
    namespace helpers {
        /**
         * @brief Get the column index of an nd-array
         *
         * @tparam index_type
         * @tparam T
         * @param idx global index
         * @param width width of nd-array
         * @param length length of nd-array
         * @param k height of the nd-array
         * @return column index
         */
        template <typename index_type, typename T>
        STATIC index_type
        get_column(index_type idx, T width, T length = 1, index_type k = 0)
        {
            idx -= (k * width * length);
            return idx % width;
        }

        /**
         * @brief Get the row index of an nd-array
         *
         * @tparam index_type
         * @tparam T
         * @param idx global index
         * @param width width of nd-array
         * @param length length of nd-array
         * @param k height of nd-array
         * @return row index
         */
        template <typename index_type, typename T>
        STATIC index_type
        get_row(index_type idx, T width, T length = 1, index_type k = 0)
        {
            idx -= (k * width * length);
            return idx / width;
        }

        /**
         * @brief Get the height index of an nd-array
         *
         * @tparam index_type
         * @tparam T
         * @param idx global index
         * @param width width of nd-array
         * @param length length of nd-array
         * @return height index
         */
        template <typename index_type, typename T>
        STATIC index_type get_height(index_type idx, T width, T length)
        {
            return idx / width / length;
        }

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
            SimulationFailureException();
            const char* what() const noexcept;
        };

        /*
        Catch keyboard Ctrl+C and other signals and pass
        back to python to kill
        */
        void catch_signals();

        /**
         * @brief get maximum between two values
         *
         * @tparam T type of params
         * @param a
         * @param b
         * @return maximum
         */
        template <typename T>
        STATIC constexpr T my_max(const T a, const T b)
        {
            return a > b ? a : b;
        }

        /**
         * @brief get minimum between two values
         *
         * @tparam T type of params
         * @param a
         * @param b
         * @return minimum
         */
        template <typename T>
        STATIC constexpr T my_min(const T a, const T b)
        {
            return a < b ? a : b;
        }

        /**
         * @brief get maximum between three values
         *
         * @tparam T
         * @param a
         * @param b
         * @param c
         * @return maximum
         */
        template <typename T>
        STATIC constexpr T my_max3(const T a, const T b, const T c)
        {
            return (a > b) ? (a > c ? a : c) : b > c ? b : c;
        }

        /**
         * @brief get minimum between three values
         *
         * @tparam T
         * @param a
         * @param b
         * @param c
         * @return minimum
         */
        template <typename T>
        STATIC constexpr T my_min3(const T a, const T b, const T c)
        {
            return (a < b) ? (a < c ? a : c) : b < c ? b : c;
        }

        // map geometry string to simbi::Geometry enum class
        const std::map<std::string, simbi::Geometry> geometry_map = {
          {"spherical", simbi::Geometry::SPHERICAL},
          {"cartesian", simbi::Geometry::CARTESIAN},
          {"planar_cylindrical", simbi::Geometry::PLANAR_CYLINDRICAL},
          {"axis_cylindrical", simbi::Geometry::AXIS_CYLINDRICAL},
          {"cylindrical", simbi::Geometry::CYLINDRICAL}
        };

        // map boundary condition string to simbi::BoundaryCondition enum class
        const std::map<std::string, simbi::BoundaryCondition>
            boundary_cond_map = {
              {"dynamic", simbi::BoundaryCondition::DYNAMIC},
              {"outflow", simbi::BoundaryCondition::OUTFLOW},
              {"reflecting", simbi::BoundaryCondition::REFLECTING},
              {"periodic", simbi::BoundaryCondition::PERIODIC}
        };

        // map solver string to simbi::Solver enum class
        const std::map<std::string, simbi::Solver> solver_map = {
          {"hllc", simbi::Solver::HLLC},
          {"hlle", simbi::Solver::HLLE},
          {"hlld", simbi::Solver::HLLD}
        };
        //---------------------------------------------------------------------------------------------------------
        //  HELPER-TEMPLATES
        //---------------------------------------------------------------------------------------------------------
        //-------------Define Function Templates-------------------------
        // custom string formatter
        template <typename... Args>
        std::string string_format(const std::string& format, Args... args);

        /**
         * @brief get sign of class T
         *
         * @tparam T
         * @param val values
         * @return sign of val
         */
        template <typename T>
        STATIC constexpr int sgn(T val)
        {
            return (T(0) < val) - (val < T(0));
        }

        /**
         * @brief function template for writing checkpoint to hdf5 file
         *
         * @tparam Prim_type
         * @tparam Ndim
         * @tparam Sim_type
         * @param sim_state_host the host-side simulation state
         * @param setup the setup struct
         * @param t the current time in the simulation
         * @param t_interval the current time interval in the simulation
         * @param chkpt_interval the current checkpoint interval
         * @param chkpt_zone_label the zone label for the checkpoint
         * <zone>.chkpt.<time>.h5
         */
        template <typename Sim_type>
        void write_to_file(Sim_type& sim_state);

        //---------------------------------------------------------------------------------------------------------
        //  HELPER-METHODS
        //---------------------------------------------------------------------------------------------------------
        //----------------Define Methods-------------------------------
        /**
         * @brief formats a real number to a string in the format 000_000 etc
         *
         * @param value
         * @return std::string
         */
        std::string format_real(real value);

        /**
         * @brief write to the hdf5 file (serial)
         *
         * @param data_directory
         * @param filename
         * @param prims
         * @param system
         * @param dim
         * @param size
         */
        template <typename T>
        void write_hdf5(
            const std::string data_directory,
            const std::string filename,
            const T& state
        );

        /**
         * @brief set the riemann solver function pointer on
         * the device or the host depending on the platform
         *
         * @tparam T
         * @param hydro_class
         * @return void
         */
        template <typename T>
        KERNEL void hybrid_set_riemann_solver(T hydro_class)
        {
            hydro_class->set_riemann_solver();
        }

        //-------------------Inline for Speed -------------------------
        /**
         * @brief compute the minmod slope limiter
         *
         * @param x
         * @param y
         * @param z
         * @return minmod value between x, y, and z
         */
        STATIC real minmod(const real x, const real y, const real z)
        {
            return 0.25 * std::abs(sgn(x) + sgn(y)) * (sgn(x) + sgn(z)) *
                   my_min3(std::abs(x), std::abs(y), std::abs(z));
        };

        /**
         * @brief compute the minmod slope limiter
         *
         * @param x
         * @param y
         * @return minmod value between x and y
         */
        STATIC real minmod(const real x, const real y)
        {
            return 0.5 * std::abs(sgn(x) + sgn(y)) * sgn(x) *
                   my_min(std::abs(x), std::abs(y));
        };

        /**
         * @brief compute the van Leer slope limiter (van Leer 1977)
         *
         * @param x
         * @param y
         * @return van Leer value between x and y
         */
        STATIC real vanLeer(const real x, const real y)
        {
            if (x * y > 0.0) {
                return static_cast<real>(2.0) * (x * y) / (x + y);
            }
            return static_cast<real>(0.0);
        };

        /**
         * @brief calculate the mean between any two values based on cell
         * spacing
         *
         * @param a
         * @param b
         * @param cellspacing
         * @return arithmetic or geometric mean between a and b
         */
        STATIC
        real calc_any_mean(
            const real a,
            const real b,
            const simbi::Cellspacing cellspacing
        )
        {
            switch (cellspacing) {
                case simbi::Cellspacing::LOGSPACE:
                    return std::sqrt(a * b);
                case simbi::Cellspacing::LINSPACE:
                    return 0.5 * (a + b);
                default:
                    return INFINITY;
            }
        }

        // the plm gradient for generic hydro
        template <typename T>
        STATIC T
        plm_gradient(const T& a, const T& b, const T& c, const real plm_theta)
        {
            T result;
            constexpr auto count = T::nmem;
            for (auto qq = 0; qq < count; qq++) {
                result[qq] = minmod(
                    (a[qq] - b[qq]) * plm_theta,
                    (c[qq] - b[qq]) * 0.5,
                    (c[qq] - a[qq]) * plm_theta
                );
            }
            return result;
        }

        /**
         * @brief Get the real idx of an array object
         *
         * @param idx the global index
         * @param offset the halo radius
         * @param active_zones the number of real, active zones in the grid
         * @return the nearest active index corresponding to the global index
         * given
         */
        STATIC
        constexpr luint
        get_real_idx(const lint idx, const lint offset, const lint active_zones)
        {
            if (idx > active_zones - 1 + offset) {
                return active_zones - 1;
            }
            return (idx - offset > 0) * (idx - offset);
        }

        /**
         * @brief calculate relativistic f & df/dp from Mignone and Bodo (2005)
         * @param gamma adiabatic index
         * @param tau energy density minus rest mass energy
         * @param d lab frame density
         * @param S lab frame momentum density
         * @param p pressure
         */
        STATIC auto newton_fg(real gamma, real tau, real d, real s, real p)
        {
            const auto et  = tau + d + p;
            const auto v2  = s * s / (et * et);
            const auto w   = 1.0 / std::sqrt(1.0 - v2);
            const auto rho = d / w;
            const auto eps =
                (tau + (1.0 - w) * d + (1.0 - w * w) * p) / (d * w);
            const auto c2 = (gamma - 1) * gamma * eps / (1 + gamma * eps);
            return std::make_tuple((gamma - 1.0) * rho * eps - p, c2 * v2 - 1);
        }

        /**
         * @brief calculate the bracketing function described in Kastaun,
         * Kalinani, & Colfi (2021)
         *
         * @param mu minimization variable
         * @param beesq rescaled magnetic field squared
         * @param r vector of rescaled momentum
         * @param beedr inner product between rescaled magnetic field & momentum
         * @return Eq. (49)
         */
        STATIC real kkc_fmu49(
            const real mu,
            const real beesq,
            const real beedrsq,
            const real r
        )
        {
            // the minimum enthalpy is unity for non-relativistic flows
            constexpr real hlim = 1.0;

            // Equation (26)
            const real x = 1.0 / (1.0 + mu * beesq);

            // Equation (38)
            const real rbar_sq = r * r * x * x + mu * x * (1.0 + x) * beedrsq;

            return mu * std::sqrt(hlim * hlim + rbar_sq) - 1.0;
        }

        /**
         * @brief Returns the master function described in Kastaun, Kalinani, &
         * Colfi (2021)
         *
         * @param mu minimization variable
         * @param r vector of rescaled momentum
         * @param rparr parallel component of rescaled momentum vector
         * @param beesq rescaled magnetic field squared
         * @param beedr inner product between rescaled magnetic field & momentum
         * @param qterm rescaled gas energy density
         * @param dterm mass density
         * @param gamma adiabatic index
         * @return Eq. (44)
         */
        STATIC real kkc_fmu44(
            const real mu,
            const real r,
            const real rparr,
            const real rperp,
            const real beesq,
            const real beedrsq,
            const real qterm,
            const real dterm,
            const real gamma
        )
        {
            // Equation (26)
            const real x = 1.0 / (1.0 + mu * beesq);

            // Equation (38)
            const real rbar_sq = r * r * x * x + mu * x * (1.0 + x) * beedrsq;

            // Equation (39)
            const real qbar =
                qterm - 0.5 * (beesq + mu * mu * x * x * beesq * rperp * rperp);

            // Equation (32) inverted and squared
            const real vsq  = mu * mu * rbar_sq;
            const real gbsq = vsq / std::abs(1.0 - vsq);
            const real g    = std::sqrt(1.0 + gbsq);

            // Equation (41)
            const real rhohat = dterm / g;

            // Equation (42)
            const real epshat = g * (qbar - mu * rbar_sq) + gbsq / (1.0 + g);

            // Equation (43)
            const real phat = (gamma - 1.0) * rhohat * epshat;
            const real ahat = phat / (rhohat * (1.0 + epshat));

            // Equation (46) - (48)
            const real vhatA = (1.0 + ahat) * (1.0 + epshat) / g;
            const real vhatB = (1.0 + ahat) * (1.0 + qbar - mu * rbar_sq);
            const real vhat  = my_max(vhatA, vhatB);

            // Equation (45)
            const real muhat = 1.0 / (vhat + rbar_sq * mu);

            return mu - muhat;
        }

        //======================================
        //          GPU TEMPLATES
        //======================================
        // compute the discrete dts for 1D primitive array
        template <
            typename T,
            TIMESTEP_TYPE dt_type = TIMESTEP_TYPE::ADAPTIVE,
            typename U,
            typename V>
        KERNEL typename std::enable_if<is_1D_primitive<T>::value>::type
        compute_dt(U* s, const V* prim_buffer, real* dt_min);

        // compute the discrete dts for 2D primitive array
        template <
            typename T,
            TIMESTEP_TYPE dt_type = TIMESTEP_TYPE::ADAPTIVE,
            typename U,
            typename V>
        KERNEL typename std::enable_if<is_2D_primitive<T>::value>::type
        compute_dt(
            U* s,
            const V* prim_buffer,
            real* dt_min,
            const simbi::Geometry geometry
        );

        // compute the discrete dts for 3D primitive array
        template <
            typename T,
            TIMESTEP_TYPE dt_type = TIMESTEP_TYPE::ADAPTIVE,
            typename U,
            typename V>
        KERNEL typename std::enable_if<is_3D_primitive<T>::value>::type
        compute_dt(
            U* s,
            const V* prim_buffer,
            real* dt_min,
            const simbi::Geometry geometry
        );

        // compute the discrete dts for 1D MHD primitive array
        template <
            typename T,
            TIMESTEP_TYPE dt_type = TIMESTEP_TYPE::ADAPTIVE,
            typename U,
            typename V>
        KERNEL typename std::enable_if<is_1D_mhd_primitive<T>::value>::type
        compute_dt(U* s, const V* prim_buffer, real* dt_min);

        // compute the discrete dts for 2D MHD primitive array
        template <
            typename T,
            TIMESTEP_TYPE dt_type = TIMESTEP_TYPE::ADAPTIVE,
            typename U,
            typename V>
        KERNEL typename std::enable_if<is_2D_mhd_primitive<T>::value>::type
        compute_dt(
            U* s,
            const V* prim_buffer,
            real* dt_min,
            const simbi::Geometry geometry
        );

        // compute the discrete dts for 3D MHD primitive array
        template <
            typename T,
            TIMESTEP_TYPE dt_type = TIMESTEP_TYPE::ADAPTIVE,
            typename U,
            typename V>
        KERNEL typename std::enable_if<is_3D_mhd_primitive<T>::value>::type
        compute_dt(
            U* s,
            const V* prim_buffer,
            real* dt_min,
            const simbi::Geometry geometry
        );

        //======================================
        //              HELPER OVERLOADS
        //======================================
        /**
         * @brief check if left and right pressures meet the Quirk (1994)
         * criterion
         *
         * @param pl left pressure
         * @param pr right pressure
         * @return true if smoothing needed, false otherwise
         */
        STATIC
        bool quirk_strong_shock(const real pl, const real pr)
        {
            return std::abs(pr - pl) / my_min(pl, pr) > QUIRK_THRESHOLD;
        }

        /**
         * @brief kronecker delta
         *
         * @param i
         * @param j
         * @return 1 for identity, 0 otherwise
         */
        STATIC
        constexpr unsigned int kronecker(luint i, luint j) { return (i == j); }

        /**
         * @brief Get the 2d idx object depending on row-major or column-major
         * ordering
         *
         * @param ii column index
         * @param jj row index
         * @param nx number of columns
         * @param ny number of rows
         * @return row-major or column-major index for 2D array
         */
        STATIC
        auto
        idx2(const luint ii, const luint jj, const luint nx, const luint ny)
        {
            if constexpr (global::col_maj) {
                return ii * ny + jj;
            }
            return jj * nx + ii;
        }

        /**
         * @brief Get the 3D idx object depending on row-major or column-major
         * ordering
         *
         * @param ii column index
         * @param jj row index
         * @param kk height index
         * @param nx number of columns
         * @param ny number of rows
         * @param nz number of heights
         * @return row-major or column-major index for 3D array
         */
        STATIC
        auto idx3(
            const luint ii,
            const luint jj,
            const luint kk,
            const luint nx,
            const luint ny,
            const luint nk
        )
        {
            if constexpr (global::col_maj) {
                return ii * nk * ny + jj * nk + kk;
            }
            return kk * nx * ny + jj * nx + ii;
        }

        /**
         * @brief Get the geometric cell centroid for spherical or cylindrical
         * mesh
         *
         * @param xr left coordinates
         * @param xl right coordinate
         * @param geometry geometry of state
         * @return cell centroid
         */
        STATIC
        auto get_cell_centroid(
            const real xr,
            const real xl,
            const simbi::Geometry geometry
        )
        {
            switch (geometry) {
                case Geometry::SPHERICAL:
                    return 0.75 * (xr * xr * xr * xr - xl * xl * xl * xl) /
                           (xr * xr * xr - xl * xl * xl);
                default:
                    return (2.0 / 3.0) * (xr * xr * xr - xl * xl * xl) /
                           (xr * xr - xl * xl);
            }
        }

        /**
         * @brief the next permutation in the set {1,2} or {1, 2, 3}
         *
         * @param nhat normal component value
         * @param step permutation steps
         * @return next permutation
         */
        STATIC
        constexpr auto next_perm(const luint nhat, const luint step)
        {
            return ((nhat - 1) + step) % 3 + 1;
        };

        /**
         * @brief           get the index of the direction neighbor
         * @param[in/out/in,out]ii:
         * @param[in/out/in,out]jj:
         * @param[in/out/in,out]kk:
         * @param[in/out/in,out]ni:
         * @param[in/out/in,out]nj:
         * @param[in/out/in,out]nk:
         * @return
         * @retval
         */
        template <Plane P, Corner C, Dir s>
        DUAL lint cidx(lint ii, lint jj, lint kk, luint ni, luint nj, luint nk);

        /**
         * @brief check if a value is within a range
         *
         * @tparam IndexType
         * @param val
         * @param lower
         * @param upper
         * @return bool
         */
        template <class IndexType>
        DUAL bool in_range(IndexType val, IndexType lower, IndexType upper);

        // configure the ghost zones in 1D hydro
        template <typename sim_state_t>
        void config_ghosts1D(sim_state_t& sim_state);

        // configure the ghost zones in 2D hydro
        template <typename sim_state_t>
        void config_ghosts2D(sim_state_t& sim_state);

        // configure the ghost zones in 3D hydro
        template <typename sim_state_t>
        void config_ghosts3D(sim_state_t& sim_state);

        template <typename T>
        void config_ghosts(T& sim_state);

        /**
         * @brief perform the reduction within the warp
         *
         * @param val
         * @return reduced min in the warp
         */
        inline DEV real warpReduceMin(real val)
        {
#if CUDA_CODE
            // Adapted from https://stackoverflow.com/a/59883722/13874039
            // to work with older cuda versions
            int mask;
#if __CUDA_ARCH__ >= 700
            mask = __match_any_sync(__activemask(), val);
#else
            const int tid = threadIdx.z * blockDim.x * blockDim.y +
                            threadIdx.y * blockDim.x + threadIdx.x;
            unsigned tmask = __activemask();
            for (int i = 0; i < global::WARP_SIZE; i++) {
                unsigned long long tval =
                    __shfl_sync(tmask, (unsigned long long) val, i);
                unsigned my_mask =
                    __ballot_sync(tmask, (tval == (unsigned long long) val));
                if (i == (tid & (global::WARP_SIZE - 1))) {
                    mask = my_mask;
                }
            }
#endif
            for (int offset = global::WARP_SIZE / 2; offset > 0; offset /= 2) {
                real next_val = __shfl_down_sync(mask, val, offset);
                val           = (val < next_val) ? val : next_val;
            }
            return val;
#elif HIP_CODE
            for (int offset = global::WARP_SIZE / 2; offset > 0; offset /= 2) {
                real next_val = __shfl_down(val, offset);
                val           = (val < next_val) ? val : next_val;
            }
            return val;
#else
            return 0.0;
#endif
        };

        /**
         * @brief perform the reduction in the GPU block
         *
         * @param val
         * @return block reduced value
         */
        inline DEV real blockReduceMin(real val)
        {
#if GPU_CODE
            static __shared__ real
                shared[global::WARP_SIZE];   // Shared mem for 32 (Nvidia) / 64
                                             // (AMD) partial mins
            const int tid = threadIdx.z * blockDim.x * blockDim.y +
                            threadIdx.y * blockDim.x + threadIdx.x;
            const int bsz = blockDim.x * blockDim.y * blockDim.z;
            int lane      = tid % global::WARP_SIZE;
            int wid       = tid / global::WARP_SIZE;

            val = warpReduceMin(val);   // Each warp performs partial reduction
            if (lane == 0) {
                shared[wid] = val;   // Write reduced value to shared memory
            }
            __syncthreads();   // Wait for all partial reductions

            // printf("Lane[%d]: %f\n", lane, shared[lane]);
            // read from shared memory only if that warp existed
            val = (tid < bsz / global::WARP_SIZE) ? shared[lane] : val;

            if (wid == 0) {
                val = warpReduceMin(val);   // Final reduce within first warp
            }
            return val;
#else
            return 0.0;
#endif
        };

        template <typename T>
        KERNEL void deviceReduceKernel(T* self, lint nmax);

        template <typename T>
        KERNEL void deviceReduceWarpAtomicKernel(T* self, lint nmax);

        // display the CPU / GPU device properties
        void anyDisplayProps();

        /**
         * @brief Get the Flops count in GB / s
         *
         * @tparam T Conserved type
         * @tparam U Primitive type
         * @param radius halo radius
         * @param total_zones total number of zones in sim
         * @param real_zones total number of active zones in mesh
         * @param delta_t time for event completion
         * @return float
         */
        template <typename T, typename U>
        inline real getFlops(
            const luint dim,
            const luint radius,
            const luint total_zones,
            const luint real_zones,
            const float delta_t
        );

#if GPU_CODE
        __device__ __forceinline__ real atomicMinReal(real* addr, real value)
        {
            real old = __int_as_real(
                atomicMin((atomic_cast*) addr, __real_as_int(value))
            );
            return old;
        }
#endif

        // separate values in a string using custom delimiter
        template <const unsigned num, const char separator>
        void separate(std::string& input);

        // Cubic and Quartic algos adapted from
        // https://stackoverflow.com/a/50747781/13874039

        // solve the cubic equation
        template <typename T>
        DUAL T cubic(T b, T c, T d);

        // solve the quartic equation
        template <typename T>
        DUAL int quartic(T b, T c, T d, T e, T res[4]);

        // swap any two values
        template <typename T>
        DUAL void swap(T& a, T& b);

        // Partition the array and return the pivot index
        template <typename T, typename index_type>
        DUAL index_type partition(T arr[], index_type low, index_type high);

        // Quick sort implementation
        template <typename T, typename index_type>
        DUAL void recursiveQuickSort(T arr[], index_type low, index_type high);

        template <typename T, typename index_type>
        DUAL void iterativeQuickSort(T arr[], index_type low, index_type high);

        template <typename T, typename U>
        SHARED T* sm_proxy(const U object);

        template <typename T>
        SHARED auto sm_or_identity(const T* object);

        template <int dim, typename T, typename idx>
        DUAL void
        ib_modify(T& lhs, const T& rhs, const bool ib, const idx side);

        template <int dim, typename T, typename idx>
        DUAL bool ib_check(
            T& arr,
            const idx ii,
            const idx jj,
            const idx kk,
            const idx ni,
            const idx nj,
            const int side
        );

        template <int dim, BlkAx axis, typename T>
        DUAL T axid(T idx, T ni, T nj, T kk = T(0));

        template <typename T>
        DUAL bool goes_to_zero(T val)
        {
            return (val * val) < global::epsilon;
        }

        template <int dim, typename T, typename U, typename V>
        DEV void load_shared_buffer(
            const ExecutionPolicy<>& p,
            T& buffer,
            const U& data,
            const V ni,
            const V nj,
            const V nk,
            const V sx,
            const V sy,
            const V tx,
            const V ty,
            const V tz,
            const V txa,
            const V tya,
            const V tza,
            const V ia,
            const V ja,
            const V ka,
            const V radius
        );

    }   // namespace helpers
}   // namespace simbi

#include "helpers.ipp"
#endif