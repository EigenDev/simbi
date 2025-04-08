/**
 *  *=============================================================================
 *  *           SIMBI - Special Relativistic Magnetohydrodynamics Code
 *  *=============================================================================
 *  *
 *  * @file            helpers.hpp
 *  * @brief           assortment of helper functions and classes
 *  * @details
 *  *
 *  * @version         0.8.0
 *  * @date            2025-02-26
 *  * @author          Marcus DuPont
 *  * @email           marcus.dupont@princeton.edu
 *  *
 *  *==============================================================================
 *  * @build           Requirements & Dependencies
 *  *==============================================================================
 *  * @requires        C++20
 *  * @depends         CUDA >= 11.0, HDF5 >= 1.12, OpenMP >= 4.5
 *  * @platform        Linux, MacOS
 *  * @parallel        GPU (CUDA, HIP), CPU (OpenMP)
 *  *
 *  *==============================================================================
 *  * @documentation   Reference & Notes
 *  *==============================================================================
 *  * @usage
 *  * @note
 *  * @warning
 *  * @todo
 *  * @bug
 *  * @performance
 *  *
 *  *==============================================================================
 *  * @testing        Quality Assurance
 *  *==============================================================================
 *  * @test
 *  * @benchmark
 *  * @validation
 *  *
 *  *==============================================================================
 *  * @history        Version History
 *  *==============================================================================
 *  * 2025-02-26      v0.8.0      Initial implementation
 *  *
 *  *==============================================================================
 *  * @copyright (C) 2025 Marcus DuPont. All rights reserved.
 *  *==============================================================================
 */
#ifndef HELPERS_HIP_HPP
#define HELPERS_HIP_HPP

#include "build_options.hpp"            // for real, STATIC, luint, lint
#include "core/traits.hpp"              // for is_1D_primitive, is_2D_primitive
#include "core/types/alias/alias.hpp"   // for uarray
#include "core/types/utility/enums.hpp"   // for Geometry, BoundaryCondition, Solver
#include "core/types/utility/functional.hpp"   // for Function
#include "io/console/tabulate.hpp"             // for PrettyTable
#include "util/parallel/exec_policy.hpp"       // for ExecutionPolicy
#include <cmath>                               // for sqrt, exp, INFINITY
#include <cstdlib>                             // for abs, size_t
#include <exception>                           // for exception
#include <map>                                 // for map
#include <string>                              // for string, operator<=>
#include <type_traits>                         // for enable_if
#include <unordered_map>                       // for unordered_map

std::unordered_map<simbi::Cellspacing, std::string> const cell2str = {
  {simbi::Cellspacing::LOG, "log"},
  {simbi::Cellspacing::LINEAR, "linear"}
  // {"log-linear",Cellspacing},
  // {"linear-log",Cellspacing},
};

// Some useful global constants
constexpr real QUIRK_THRESHOLD = 1e-4;

// Calculation derived from:
// https://developer.nvidia.com/blog/how-implement-performance-metrics-cuda-cc/
extern real gpu_theoretical_bw;   //  = 1875e6 * (192.0 / 8.0) * 2 / 1e9;

namespace simbi {
    namespace helpers {
        // forward declarations
        STATIC real minmod(real a, real b, real c);
        STATIC real vanLeer(real a, real b);

        //==========================================================================
        // TEMPLATES
        //==========================================================================

        template <int dim>
        struct real_func {
            using type = int;
        };

        template <>
        struct real_func<1> {
            using type = simbi::function<void(real, real, real[])>;
        };

        template <>
        struct real_func<2> {
            using type = simbi::function<void(real, real, real, real[])>;
        };

        template <>
        struct real_func<3> {
            using type = simbi::function<void(real, real, real, real, real[])>;
        };

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

        //---------------------------------------------------------------------------------------------------------
        //  HELPER-METHODS

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

        /**
         * @brief set the face area function pointer on
         * the device or the host depending on the platform
         *
         * @tparam T
         * @param hydro_class
         * @return void
         */
        template <typename T>
        KERNEL void hybrid_set_mesh_funcs(T geom_class)
        {
            geom_class->initialize_function_pointers();
        }

        // the plm gradient for generic hydro
        template <typename T>
        STATIC T
        plm_gradient(const T& a, const T& b, const T& c, const real plm_theta)
        {
            T result;
            constexpr auto count = T::nmem;
            for (auto qq = 0; qq < count; qq++) {
                if constexpr (is_relativistic_mhd<T>::value) {
                    // this is checked at compile time
                    switch (comp_slope_limiter) {
                        case LIMITER::VAN_LEER:
                            result[qq] = vanLeer(c[qq] - a[qq], a[qq] - b[qq]);
                            break;
                        default:
                            result[qq] = minmod(
                                (a[qq] - b[qq]) * plm_theta,
                                (c[qq] - b[qq]) * 0.5,
                                (c[qq] - a[qq]) * plm_theta
                            );
                            break;
                    }
                }
                else {
                    result[qq] = minmod(
                        (a[qq] - b[qq]) * plm_theta,
                        (c[qq] - b[qq]) * 0.5,
                        (c[qq] - a[qq]) * plm_theta
                    );
                }
            }
            return result;
        }

        //=========================================================================
        //              HELPER OVERLOADS
        //==========================================================================

        template <int dim, typename T>
        KERNEL void
        deviceReduceWarpAtomicKernel(T* self, real* dt_min, lint nmax);

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
        // template <typename T>
        // DUAL void solve_cubic(T b, T c, T d, T res[3]);

        template <typename T>
        DUAL T solve_cubic(T a, T b, T c);

        // solve the quartic equation
        template <typename T>
        DUAL int solve_quartic(T b, T c, T d, T e, T res[4]);

        // Partition the array and return the pivot index
        template <typename T, typename index_type>
        DUAL index_type partition(T arr[], index_type low, index_type high);

        // Quick sort implementation
        template <typename T, typename index_type>
        DUAL void recursiveQuickSort(T arr[], index_type low, index_type high);

        template <typename T, typename index_type>
        DUAL void iterativeQuickSort(T arr[], index_type low, index_type high);

        template <int dim, BlockAx axis, typename T>
        DUAL T axid(T idx, T ni, T nj, T kk = T(0));

        template <typename T>
        DUAL bool goes_to_zero(T val)
        {
            return (val * val) < global::epsilon;
        }

        std::string getColorCode(Color color);

        // display the CPU / GPU device properties
        void anyDisplayProps();

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

        template <size_type Dims>
        DUAL constexpr auto unravel_idx(const luint idx, const auto& shape)
        {
            uarray<Dims> coords;
            auto stride = 1;
            if constexpr (global::col_major) {
                // Column major: shape=(nk,nj,ni)
                // Want [k,j,i] where i is fastest
                for (size_type ii = 0; ii < Dims; ++ii) {
                    coords[Dims - 1 - ii] =
                        (idx / stride) % shape[Dims - 1 - ii];
                    stride *= shape[Dims - 1 - ii];
                }
            }
            else {
                // Row major: shape=(nk,nj,ni)
                // Want [i,j,k] where i is fastest
                for (size_type ii = Dims - 1; ii < Dims; --ii) {
                    coords[Dims - 1 - ii] = (idx / stride) % shape[ii];
                    stride *= shape[ii];
                }
            }

            return coords;
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
            if constexpr (global::col_major) {
                return ii * nk * ny + jj * nk + kk;
            }
            return kk * nx * ny + jj * nx + ii;
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
            if constexpr (global::col_major) {
                return ii * ny + jj;
            }
            return jj * nx + ii;
        }

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

        STATIC constexpr std::tuple<luint, luint, luint>
        get_indices(const luint idx, const auto nx, const auto ny)
        {
            const auto kk = get_height(idx, nx, ny);
            const auto jj = get_row(idx, nx, ny, kk);
            const auto ii = get_column(idx, nx, ny, kk);
            return {ii, jj, kk};
        }

        STATIC constexpr std::tuple<luint, luint, luint> get_real_indices(
            const auto ii,
            const auto jj,
            const auto kk,
            const auto nx,
            const auto ny,
            const auto offset
        )
        {
            return {
              get_real_idx(ii, offset, nx),
              get_real_idx(jj, offset, ny),
              get_real_idx(kk, offset, 1)
            };
        }

        /**
         * @brief calculate relativistic f & df/dp from Mignone and Bodo (2005)
         * @param gamma adiabatic index
         * @param tau energy density minus rest mass energy
         * @param d lab frame density
         * @param S lab frame momentum density
         * @param p pressure
         */
        DEV std::tuple<real, real>
        newton_fg(real gamma, real tau, real d, real s, real p);

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
        DEV real kkc_fmu49(
            const real mu,
            const real beesq,
            const real beedrsq,
            const real r
        );

        /**
         * @brief Returns the master function described in Kastaun, Kalinani, &
         * Colfi (2021)
         *
         * @param mu minimization variable
         * @param r vector of rescaled momentum
         * @param beesq rescaled magnetic field squared
         * @param beedr inner product between rescaled magnetic field & momentum
         * @param qterm rescaled gas energy density
         * @param dterm mass density
         * @param gamma adiabatic index
         * @return Eq. (44)
         */
        DEV real kkc_fmu44(
            const real mu,
            const real r,
            const real rperp,
            const real beesq,
            const real beedrsq,
            const real qterm,
            const real dterm,
            const real gamma
        );

        DEV real
        find_mu_plus(const real beesq, const real beedrsq, const real r);

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
         * @brief formats a real number to a string in the format 000_000 etc
         *
         * @param value
         * @return std::string
         */
        std::string format_real(real value);

        template <size_type Dims>
        DUAL static auto
        memory_layout_coordinates(auto idx, const uarray<Dims>& shape)
            -> uarray<Dims>;

    }   // namespace helpers
}   // namespace simbi

#include "helpers.ipp"
#endif
