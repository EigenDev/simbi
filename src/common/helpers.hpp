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
              {"inflow", simbi::BoundaryCondition::INFLOW},
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

        // Handle 3D primitive arrays whether SR or Newtonian
        template <typename T, typename U>
        typename std::enable_if<is_3D_primitive<U>::value>::type
        writeToProd(T* from, PrimData* to);

        // Handle 2D primitive arrays whether SR or Newtonian
        template <typename T, typename U>
        typename std::enable_if<is_2D_primitive<U>::value>::type
        writeToProd(T* from, PrimData* to);

        // Handle 1D primitive arrays whether SR or Newtonian
        template <typename T, typename U>
        typename std::enable_if<is_1D_primitive<U>::value>::type
        writeToProd(T* from, PrimData* to);

        // Handle 3D primitive arrays whether RMHD or NMHD
        template <typename T, typename U>
        typename std::enable_if<is_3D_mhd_primitive<U>::value>::type
        writeToProd(T* from, PrimData* to);

        // Handle 2D primitive arrays whether RMHD or NMHD
        template <typename T, typename U>
        typename std::enable_if<is_2D_mhd_primitive<U>::value>::type
        writeToProd(T* from, PrimData* to);

        // Handle 1d primitive arrays whether RMHD or NMHD
        template <typename T, typename U>
        typename std::enable_if<is_1D_mhd_primitive<U>::value>::type
        writeToProd(T* from, PrimData* to);

        // Convert 3D vector of structs to struct of vectors
        template <typename T, typename U, typename arr_type>
        typename std::enable_if<is_3D_primitive<U>::value, T>::type
        vec2struct(const arr_type& p);

        // Convert 2D vector of structs to struct of vectors
        template <typename T, typename U, typename arr_type>
        typename std::enable_if<is_2D_primitive<U>::value, T>::type
        vec2struct(const arr_type& p);

        // Convert 1D vector of structs to struct of vectors
        template <typename T, typename U, typename arr_type>
        typename std::enable_if<is_1D_primitive<U>::value, T>::type
        vec2struct(const arr_type& p);

        // Convert 3D MHD vector of structs to struct of vectors
        template <typename T, typename U, typename arr_type>
        typename std::enable_if<is_3D_mhd_primitive<U>::value, T>::type
        vec2struct(const arr_type& p);

        // Convert 2D MHD vector of structs to struct of vectors
        template <typename T, typename U, typename arr_type>
        typename std::enable_if<is_2D_mhd_primitive<U>::value, T>::type
        vec2struct(const arr_type& p);

        // Convert 1D MHD vector of structs to struct of vectors
        template <typename T, typename U, typename arr_type>
        typename std::enable_if<is_1D_mhd_primitive<U>::value, T>::type
        vec2struct(const arr_type& p);

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
        //----------------Define Methods-------------------------
        /**
         * @brief Create a step str object
         *
         * @param current_time current simulation time
         * @param max_order_of_mag maximum order of magnitude in the simulation
         * step
         * @return the step string for the checkpoint filename
         */
        std::string
        create_step_str(const real current_time, const int max_order_of_mag);

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
        void write_hdf5(
            const std::string data_directory,
            const std::string filename,
            const PrimData prims,
            const DataWriteMembers system,
            const int dim,
            const int size
        );

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

        // the plm gradient in 3D hydro
        template <typename T>
        STATIC typename std::enable_if<is_3D_primitive<T>::value, T>::type
        plm_gradient(const T& a, const T& b, const T& c, const real plm_theta)
        {
            const real rho = minmod(
                (a - b).rho * plm_theta,
                (c - b).rho * 0.5,
                (c - a).rho * plm_theta
            );
            const real v1 = minmod(
                (a - b).v1 * plm_theta,
                (c - b).v1 * 0.5,
                (c - a).v1 * plm_theta
            );
            const real v2 = minmod(
                (a - b).v2 * plm_theta,
                (c - b).v2 * 0.5,
                (c - a).v2 * plm_theta
            );
            const real v3 = minmod(
                (a - b).v3 * plm_theta,
                (c - b).v3 * 0.5,
                (c - a).v3 * plm_theta
            );
            const real pre = minmod(
                (a - b).p * plm_theta,
                (c - b).p * 0.5,
                (c - a).p * plm_theta
            );
            const real chi = minmod(
                (a - b).chi * plm_theta,
                (c - b).chi * 0.5,
                (c - a).chi * plm_theta
            );
            return T{rho, v1, v2, v3, pre, chi};
        }

        // the plm gradient in 2D hydro
        template <typename T>
        STATIC typename std::enable_if<is_2D_primitive<T>::value, T>::type
        plm_gradient(const T& a, const T& b, const T& c, const real plm_theta)
        {
            const real rho = minmod(
                (a - b).rho * plm_theta,
                (c - b).rho * 0.5,
                (c - a).rho * plm_theta
            );
            const real v1 = minmod(
                (a - b).v1 * plm_theta,
                (c - b).v1 * 0.5,
                (c - a).v1 * plm_theta
            );
            const real v2 = minmod(
                (a - b).v2 * plm_theta,
                (c - b).v2 * 0.5,
                (c - a).v2 * plm_theta
            );
            const real pre = minmod(
                (a - b).p * plm_theta,
                (c - b).p * 0.5,
                (c - a).p * plm_theta
            );
            const real chi = minmod(
                (a - b).chi * plm_theta,
                (c - b).chi * 0.5,
                (c - a).chi * plm_theta
            );
            return T{rho, v1, v2, pre, chi};
        }

        // the plm gradient in 1D hydro
        template <typename T>
        STATIC typename std::enable_if<is_1D_primitive<T>::value, T>::type
        plm_gradient(const T& a, const T& b, const T& c, const real plm_theta)
        {
            const real rho = minmod(
                (a - b).rho * plm_theta,
                (c - b).rho * 0.5,
                (c - a).rho * plm_theta
            );
            const real v1 = minmod(
                (a - b).v1 * plm_theta,
                (c - b).v1 * 0.5,
                (c - a).v1 * plm_theta
            );
            const real pre = minmod(
                (a - b).p * plm_theta,
                (c - b).p * 0.5,
                (c - a).p * plm_theta
            );
            const real chi = minmod(
                (a - b).chi * plm_theta,
                (c - b).chi * 0.5,
                (c - a).chi * plm_theta
            );
            return T{rho, v1, pre, chi};
        }

        /**
         * @brief
         * @param[in/out/in,out]a: center
         * @param[in/out/in,out]b: left of center
         * @param[in/out/in,out]c: right of center
         * @param[in/out/in,out]plm_theta:
         * @return          STATIC
         * @retval
         */
        template <typename T>
        STATIC typename std::enable_if<is_3D_mhd_primitive<T>::value, T>::type
        plm_gradient(const T& a, const T& b, const T& c, const real plm_theta)
        {
            switch (comp_slope_limiter) {
                case LIMITER::MINMOD:
                    {
                        const real rho = minmod(
                            (a - b).rho * plm_theta,
                            (c - b).rho * 0.5,
                            (c - a).rho * plm_theta
                        );
                        const real v1 = minmod(
                            (a - b).v1 * plm_theta,
                            (c - b).v1 * 0.5,
                            (c - a).v1 * plm_theta
                        );
                        const real v2 = minmod(
                            (a - b).v2 * plm_theta,
                            (c - b).v2 * 0.5,
                            (c - a).v2 * plm_theta
                        );
                        const real v3 = minmod(
                            (a - b).v3 * plm_theta,
                            (c - b).v3 * 0.5,
                            (c - a).v3 * plm_theta
                        );
                        const real pre = minmod(
                            (a - b).p * plm_theta,
                            (c - b).p * 0.5,
                            (c - a).p * plm_theta
                        );
                        const real b1 = minmod(
                            (a - b).b1 * plm_theta,
                            (c - b).b1 * 0.5,
                            (c - a).b1 * plm_theta
                        );
                        const real b2 = minmod(
                            (a - b).b2 * plm_theta,
                            (c - b).b2 * 0.5,
                            (c - a).b2 * plm_theta
                        );
                        const real b3 = minmod(
                            (a - b).b3 * plm_theta,
                            (c - b).b3 * 0.5,
                            (c - a).b3 * plm_theta
                        );
                        const real chi = minmod(
                            (a - b).chi * plm_theta,
                            (c - b).chi * 0.5,
                            (c - a).chi * plm_theta
                        );
                        return T{rho, v1, v2, v3, pre, b1, b2, b3, chi};
                    }

                default:
                    {
                        const real rho = vanLeer((c - a).rho, -(b - a).rho);
                        const real v1  = vanLeer((c - a).v1, -(b - a).v1);
                        const real v2  = vanLeer((c - a).v2, -(b - a).v2);
                        const real v3  = vanLeer((c - a).v3, -(b - a).v3);
                        const real pre = vanLeer((c - a).p, -(b - a).p);
                        const real b1  = vanLeer((c - a).b1, -(b - a).b1);
                        const real b2  = vanLeer((c - a).b2, -(b - a).b2);
                        const real b3  = vanLeer((c - a).b3, -(b - a).b3);
                        const real chi = vanLeer((c - a).chi, -(b - a).chi);
                        return T{rho, v1, v2, v3, pre, b1, b2, b3, chi};
                    }
            }
        }

        // the plm gradient in 2D MHD
        template <typename T>
        STATIC typename std::enable_if<is_2D_mhd_primitive<T>::value, T>::type
        plm_gradient(const T& a, const T& b, const T& c, const real plm_theta)
        {
            const real rho = minmod(
                (a - b).rho * plm_theta,
                (c - b).rho * 0.5,
                (c - a).rho * plm_theta
            );
            const real v1 = minmod(
                (a - b).v1 * plm_theta,
                (c - b).v1 * 0.5,
                (c - a).v1 * plm_theta
            );
            const real v2 = minmod(
                (a - b).v2 * plm_theta,
                (c - b).v2 * 0.5,
                (c - a).v2 * plm_theta
            );
            const real v3 = minmod(
                (a - b).v3 * plm_theta,
                (c - b).v3 * 0.5,
                (c - a).v3 * plm_theta
            );
            const real pre = minmod(
                (a - b).p * plm_theta,
                (c - b).p * 0.5,
                (c - a).p * plm_theta
            );
            const real b1 = minmod(
                (a - b).b1 * plm_theta,
                (c - b).b1 * 0.5,
                (c - a).b1 * plm_theta
            );
            const real b2 = minmod(
                (a - b).b2 * plm_theta,
                (c - b).b2 * 0.5,
                (c - a).b2 * plm_theta
            );
            const real b3 = minmod(
                (a - b).b3 * plm_theta,
                (c - b).b3 * 0.5,
                (c - a).b3 * plm_theta
            );
            const real chi = minmod(
                (a - b).chi * plm_theta,
                (c - b).chi * 0.5,
                (c - a).chi * plm_theta
            );
            return T{rho, v1, v2, v3, pre, b1, b2, b3, chi};
        }

        // the plm gradient in 1D MHD
        template <typename T>
        STATIC typename std::enable_if<is_1D_mhd_primitive<T>::value, T>::type
        plm_gradient(const T& a, const T& b, const T& c, const real plm_theta)
        {
            const real rho = minmod(
                (a - b).rho * plm_theta,
                (c - b).rho * 0.5,
                (c - a).rho * plm_theta
            );
            const real v1 = minmod(
                (a - b).v1 * plm_theta,
                (c - b).v1 * 0.5,
                (c - a).v1 * plm_theta
            );
            const real v2 = minmod(
                (a - b).v2 * plm_theta,
                (c - b).v2 * 0.5,
                (c - a).v2 * plm_theta
            );
            const real v3 = minmod(
                (a - b).v3 * plm_theta,
                (c - b).v3 * 0.5,
                (c - a).v3 * plm_theta
            );
            const real pre = minmod(
                (a - b).p * plm_theta,
                (c - b).p * 0.5,
                (c - a).p * plm_theta
            );
            const real b1 = minmod(
                (a - b).b1 * plm_theta,
                (c - b).b1 * 0.5,
                (c - a).b1 * plm_theta
            );
            const real b2 = minmod(
                (a - b).b2 * plm_theta,
                (c - b).b2 * 0.5,
                (c - a).b2 * plm_theta
            );
            const real b3 = minmod(
                (a - b).b3 * plm_theta,
                (c - b).b3 * 0.5,
                (c - a).b3 * plm_theta
            );
            const real chi = minmod(
                (a - b).chi * plm_theta,
                (c - b).chi * 0.5,
                (c - a).chi * plm_theta
            );
            return T{rho, v1, v2, v3, pre, b1, b2, b3, chi};
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
         * @brief  Evaluate the sigmoid function at a time t
         * @param t current time
         * @param tduration drop-off location of function
         * @param time_step time step
         * @param constant_sources flag to check if the source terms are
         * constant
         */
        inline real sigmoid(
            const real t,
            const real tduration,
            const real time_step,
            const bool constant_sources
        )
        {
            if (constant_sources) {
                return 1.0 / time_step;
            }
            return 1.0 / (1.0 + std::exp(10.0 * (t - tduration)));
        }

        /**
         * @brief Calculate the RMHD Lorentz factor according to Mignone and
         * Bodo (2006)
         *
         * @param ssq s-squared
         * @param bsq b-squared
         * @param msq m-squared
         * @param qq  variational parameter
         * @return Lorentz factor from Eq.(18)
         */
        STATIC real calc_rmhd_lorentz(
            const real ssq,
            const real bsq,
            const real msq,
            const real qq
        )
        {
            return std::sqrt(
                1.0 - (ssq * (2.0 * qq + bsq) + msq * qq * qq) /
                          ((qq + bsq) * (qq + bsq) * qq * qq)
            );
        }

        /**
         * @brief Calculate the RMHD gas pressure according to Mignone and Bodo
         * (2006)
         *
         * @param gr reduced adiabatic index
         * @param d lab frame density
         * @param w lorentz factor
         * @param qq rho * h * g *g
         * @return gas pressure from Eq.(19)
         */
        STATIC real
        calc_rmhd_pg(const real gr, const real d, const real w, const real qq)
        {
            return (qq - d * w) / (gr * w * w);
        }

        /**
         * @brief calculate relativistic f(p) from Mignone and Bodo (2005)
         * @param gamma adiabatic index
         * @param tau energy density minus rest mass energy
         * @param d lab frame density
         * @param S lab frame momentum density
         * @param p pressure
         */
        STATIC real newton_f(real gamma, real tau, real d, real s, real p)
        {
            const auto et  = tau + d + p;
            const auto v2  = s * s / (et * et);
            const auto w   = 1.0 / std::sqrt(1.0 - v2);
            const auto rho = d / w;
            const auto eps =
                (tau + (1.0 - w) * d + (1.0 - w * w) * p) / (d * w);
            return (gamma - 1) * rho * eps - p;
        }

        /**
         * @brief calculate relativistic df/dp from Mignone and Bodo (2005)
         * @param gamma adiabatic index
         * @param tau energy density minus rest mass energy
         * @param d lab frame density
         * @param S lab frame momentum density
         * @param p pressure
         */
        STATIC real newton_g(real gamma, real tau, real d, real s, real p)
        {
            const auto et = tau + d + p;
            const auto v2 = s * s / (et * et);
            const auto w  = 1.0 / std::sqrt(1.0 - v2);
            const auto eps =
                (tau + (1.0 - w) * d + (1.0 - w * w) * p) / (d * w);
            const auto c2 = (gamma - 1) * gamma * eps / (1 + gamma * eps);
            return c2 * v2 - 1;
        }

        /**
         * @brief calculate relativistic mhd f(q) & df/dq from Mignone &
         * McKinney (2007)
         * @param gr adiabatic index coeff (gamma / (gamma - 1))
         * @param tau energy density
         * @param d lab frame density
         * @param ssq s-squared
         * @param bsq b-squared
         * @param msq m-squared
         * @param qq energy density
         * @return Eq.(20)
         */
        STATIC std::tuple<real, real> newton_fg_mhd(
            real gr,
            real tau,
            real d,
            real ssq,
            real bsq,
            real msq,
            real qq
        )
        {
            //==============================
            const auto qqd = qq + d;
            const auto q2  = qqd * qqd;
            const auto rat = ssq / q2;
            const auto y1  = 1.0 / (qqd + bsq);
            const auto y2  = y1 * y1;
            // Equation (A3)
            const auto v2  = rat * y1 * (y1 * qqd + 1.0) + msq * y2;
            const auto ig2 = 1.0 - v2;
            const auto g2  = 1.0 / ig2;
            const auto g   = std::sqrt(g2);
            const auto chi = qq / g2 - d * v2 / (g + 1.0);

            const auto dv2_dq =
                -2.0 * y2 * (3.0 * rat + y1 * (rat * bsq * bsq / qqd + msq));

            //===== kinematical and thermodynamics expressions terms section A2
            const auto dchi_dq = ig2 - 0.5 * g * (d + 2.0 * chi * g) * dv2_dq;
            const auto drho_dq = -0.5 * d * g * dv2_dq;

            //====== IDEAL EOS
            const auto dp_dchi = 1.0 / gr;
            const auto dp_drho = 0.0;
            const auto pg      = chi * dp_dchi;

            //========= TODO: include Taub Adiabat

            const auto dp = dp_dchi * dchi_dq + dp_drho * drho_dq;
            return {
              qq - (pg + tau) + 0.5 * (bsq + (bsq * msq - ssq) * y2),
              1.0 - dp - (bsq * msq - ssq) * y1 * y2
            };
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
        bool quirk_strong_shock(real pl, real pr)
        {
            return std::abs(pr - pl) / helpers::my_min(pl, pr) >
                   QUIRK_THRESHOLD;
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
        auto get_2d_idx(
            const luint ii,
            const luint jj,
            const luint nx,
            const luint ny
        )
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
         * @return          HD
         * @retval
         */
        template <Plane P, Corner C, Dir s>
        HD lint cidx(lint ii, lint jj, lint kk, luint ni, luint nj, luint nk);

        // configure the ghost zones in 1D hydro
        template <typename T, typename U>
        void config_ghosts1D(
            const ExecutionPolicy<> p,
            T* cons,
            const int grid_size,
            const bool first_order,
            const simbi::BoundaryCondition* boundary_conditions,
            const U* outer_zones  = nullptr,
            const U* inflow_zones = nullptr
        );

        // configure the ghost zones in 2D hydro
        template <typename T, typename U>
        void config_ghosts2D(
            const ExecutionPolicy<> p,
            T* cons,
            const int x1grid_size,
            const int x2grid_size,
            const bool first_order,
            const simbi::Geometry geometry,
            const simbi::BoundaryCondition* boundary_conditions,
            const U* outer_zones,
            const U* boundary_zones,
            const bool half_sphere
        );

        // configure the ghost zones in 3D hydro
        template <typename T, typename U>
        void config_ghosts3D(
            const ExecutionPolicy<> p,
            T* cons,
            const int x1grid_size,
            const int x2grid_size,
            const int x3grid_size,
            const bool first_order,
            const simbi::BoundaryCondition* boundary_conditions,
            const U* inflow_zones,
            const bool half_sphere,
            const simbi::Geometry geometry
        );

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
        HD T cubic(T b, T c, T d);

        // solve the quartic equation
        template <typename T>
        HD int quartic(T b, T c, T d, T e, T res[4]);

        // solve the cubic equation
        template <typename T>
        HD int cubicPluto(T b, T c, T d, T z[]);

        // solve the quartic equation
        template <typename T>
        HD int quarticPluto(T b, T c, T d, T e, T res[4]);

        // swap any two values
        template <typename T>
        HD void swap(T& a, T& b);

        // Partition the array and return the pivot index
        template <typename T, typename index_type>
        HD index_type partition(T arr[], index_type low, index_type high);

        // Quick sort implementation
        template <typename T, typename index_type>
        HD void recursiveQuickSort(T arr[], index_type low, index_type high);

        template <typename T, typename index_type>
        HD void iterativeQuickSort(T arr[], index_type low, index_type high);

        template <typename T, typename U>
        SHARED T* sm_proxy(const U object);

        template <typename T, typename U>
        SHARED T* identity(const U& object);

        template <int dim, typename T, typename idx>
        HD void ib_modify(T& lhs, const T& rhs, const bool ib, const idx side);

        template <int dim, typename T, typename idx>
        HD bool ib_check(
            T& arr,
            const idx ii,
            const idx jj,
            const idx kk,
            const idx ni,
            const idx nj,
            const int side
        );

        template <typename index_type, typename T>
        HD index_type flattened_index(
            index_type ii,
            index_type jj,
            index_type kk,
            T ni,
            T nj,
            T nk
        );

        template <int dim, BlkAx axis, typename T>
        HD T axid(T idx, T ni, T nj, T kk = T(0));

        template <typename T>
        HD bool limit_zero(T val)
        {
            return (val * val) < global::tol_scale;
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

#include "helpers.tpp"
#endif