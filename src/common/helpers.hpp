/* 
* Helper functions for computation across all dimensions
* of the hydrodyanmic simulations for better readability/organization
* of the code
*
* Marcus DuPont
* New York University
* 04/09/2020
*/

#ifndef HELPERS_HPP
#define HELPERS_HPP

#include <vector>
#include <iostream>
#include <cmath>
#include <map>
#include <memory>
#include "H5Cpp.h"
#include "hydro_structs.hpp"
#include "enums.hpp"
#include "build_options.hpp"
#include "traits.hpp"

// Some useful global constants
constexpr real QUIRK_THRESHOLD = 1e-4;
// Calculate a static PI
constexpr double PI = 3.14159265358979323846;
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
            const char* what();
            int status;
        };

        class SimulationFailureException : public std::exception
        {
            public:
            SimulationFailureException(const char* reason, const char* details);
            const std::string what();
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
        void pause_program();
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
            const real v   = minmod((a - b).v   * plm_theta, (c - b).v   * static_cast<real>(0.5), (c - a).v   * plm_theta);
            const real pre = minmod((a - b).p   * plm_theta, (c - b).p   * static_cast<real>(0.5), (c - a).p   * plm_theta);
            // const real chi = minmod((a - b).chi * plm_theta, (c - b).chi * static_cast<real>(0.5), (c - a).chi * plm_theta);
            return T{rho, v, pre};
        }

        GPU_CALLABLE_INLINE 
        constexpr lint get_real_idx(const lint idx, const lint offset, const lint active_zones) 
        {
            lint real_idx = (idx - offset > 0) * (idx - offset);
            if (idx > active_zones + 1) {
                real_idx = active_zones - 1;
            }
            return real_idx;
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
            const auto rho   = D / W;
            const auto eps   = (tau + (1 - W) * D + (1 - W * W) * p) / (D * W);
            const auto c2    = (gamma - 1) * gamma * eps / (1 + gamma * eps);
            return c2 * v2 - 1;
        }

        
    } // namespace helpers
    
    
} // namespace simmbi




#include "helpers.tpp"
#endif