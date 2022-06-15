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
#include <algorithm>
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

namespace simbi
{
    namespace helpers
    {
        template<typename T>
        GPU_CALLABLE_INLINE
        constexpr T my_max(const T a, const T b) {
            return a > b ? a : b;
        }

        template<typename T>
        GPU_CALLABLE_INLINE
        constexpr T my_min(const T a, const T b) {
            return a < b ? a : b;
        }

        template<typename T>
        GPU_CALLABLE_INLINE
        constexpr T my_max3(const T a, const T b, const T c) {
            return (a > b) ? (a > c ? a : c) : b > c ? b : c;
        }

        template<typename T>
        GPU_CALLABLE_INLINE
        constexpr T my_min3(const T a, const T b, const T c) {
            return (a < b) ? (a < c ? a : c) : b < c ? b : c;
        }
        
        GPU_CALLABLE_INLINE lint mod(const lint index, const lint size)
        {
            return (index % size + size) % size;
        }

        const std::map<std::string, simbi::Geometry> geometry_map = {
        { "spherical", simbi::Geometry::SPHERICAL },
        { "cartesian", simbi::Geometry::CARTESIAN},
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

        template<typename T, typename N>
        typename std::enable_if<is_2D_primitive<N>::value>::type
        writeToProd(T *from, PrimData *to);

        template<typename T, typename N>
        typename std::enable_if<is_1D_primitive<N>::value>::type
        writeToProd(T *from, PrimData *to);

        template<typename T , typename N>
        typename std::enable_if<is_2D_primitive<N>::value, T>::type
        vec2struct(const std::vector<N> &p);

        template<typename T , typename N>
        typename std::enable_if<is_1D_primitive<N>::value, T>::type
        vec2struct(const std::vector<N> &p);

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

        void config_gpu_space();
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
            return 0.25 * std::abs(sgn(x) + sgn(y)) * (sgn(x) + sgn(z)) * my_min(my_min(std::abs(x), std::abs(y)) , std::abs(z)) ;
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
        minmod(const T x, const T y, const T z)
        {
            const real rho  = 0.25 * std::abs(sgn(x.rho) + sgn(y.rho)) * (sgn(x.rho) + sgn(z.rho)) * my_min(my_min(std::abs(x.rho), std::abs(y.rho)) , std::abs(z.rho)); 
            const real v1   = 0.25 * std::abs(sgn(x.v1) + sgn(y.v1)) * (sgn(x.v1) + sgn(z.v1)) * my_min(my_min(std::abs(x.v1), std::abs(y.v1)) , std::abs(z.v1));          
            const real v2   = 0.25 * std::abs(sgn(x.v2) + sgn(y.v2)) * (sgn(x.v2) + sgn(z.v2)) * my_min(my_min(std::abs(x.v2), std::abs(y.v2)) , std::abs(z.v2));       
            const real v3   = 0.25 * std::abs(sgn(x.v3) + sgn(y.v3)) * (sgn(x.v3) + sgn(z.v3)) * my_min(my_min(std::abs(x.v3), std::abs(y.v3)) , std::abs(z.v3));   
            const real pre  = 0.25 * std::abs(sgn(x.p) + sgn(y.p)) * (sgn(x.p) + sgn(z.p)) * my_min(my_min(std::abs(x.p), std::abs(y.p)) , std::abs(z.p));  
            const real chi  = 0.25 * std::abs(sgn(x.chi) + sgn(y.chi)) * (sgn(x.chi) + sgn(z.chi)) * my_min(my_min(std::abs(x.chi), std::abs(y.chi)) , std::abs(z.chi)); 

            // const real rho = minmod(x.rho, y.rho, z.rho);
            // const real v1  = minmod(x.v1, y.v1, z.v1);
            // const real v2  = minmod(x.v2, y.v2, z.v2);
            // const real v3  = minmod(x.v3, y.v3, z.v3);
            // const real pre = minmod(x.p, y.p, z.p);
            // const real chi = minmod(x.chi, y.chi, z.chi);

            return T{rho, v1, v2, v3, pre, chi};
        }

        template<typename T>
        GPU_CALLABLE_INLINE typename std::enable_if<is_2D_primitive<T>::value, T>::type
        minmod(const T x, const T y, const T z)
        {
            const real rho  = 0.25 * std::abs(sgn(x.rho) + sgn(y.rho)) * (sgn(x.rho) + sgn(z.rho)) * my_min3(std::abs(x.rho), std::abs(y.rho) , std::abs(z.rho)); 
            const real v1   = 0.25 * std::abs(sgn(x.v1)  + sgn(y.v1)) * (sgn(x.v1) + sgn(z.v1)) * my_min3(std::abs(x.v1), std::abs(y.v1) , std::abs(z.v1));          
            const real v2   = 0.25 * std::abs(sgn(x.v2)  + sgn(y.v2)) * (sgn(x.v2) + sgn(z.v2)) * my_min3(std::abs(x.v2), std::abs(y.v2) , std::abs(z.v2));       
            const real pre  = 0.25 * std::abs(sgn(x.p)   + sgn(y.p)) * (sgn(x.p) + sgn(z.p)) * my_min3(std::abs(x.p), std::abs(y.p), std::abs(z.p));  
            const real chi  = 0.25 * std::abs(sgn(x.chi) + sgn(y.chi)) * (sgn(x.chi) + sgn(z.chi)) * my_min3(std::abs(x.chi), std::abs(y.chi) , std::abs(z.chi)); 

            // const real rho = minmod(x.rho, y.rho, z.rho);
            // const real v1  = minmod(x.v1, y.v1, z.v1);
            // const real v2  = minmod(x.v2, y.v2, z.v2);
            // const real pre = minmod(x.p, y.p, z.p);
            // const real chi = minmod(x.chi, y.chi, z.chi);

            return T{rho, v1, v2, pre, chi};
        }

        template<typename T>
        GPU_CALLABLE_INLINE typename std::enable_if<is_1D_primitive<T>::value, T>::type
        minmod(const T x, const T y, const T z)
        {
            const real rho = 0.25 * std::abs(sgn(x.rho) + sgn(y.rho)) * (sgn(x.rho) + sgn(z.rho)) * my_min3(std::abs(x.rho), std::abs(y.rho) , std::abs(z.rho)); 
            const real v   = 0.25 * std::abs(sgn(x.v) + sgn(y.v)) * (sgn(x.v) + sgn(z.v)) * my_min3(std::abs(x.v), std::abs(y.v) , std::abs(z.v));               
            const real pre = 0.25 * std::abs(sgn(x.p) + sgn(y.p)) * (sgn(x.p) + sgn(z.p)) * my_min3(std::abs(x.p), std::abs(y.p) , std::abs(z.p));    
            // const real rho = minmod(x.rho, y.rho, z.rho);
            // const real v   = minmod(x.v, y.v, z.v);
            // const real pre = minmod(x.p, y.p, z.p);           
            // const real chi = minmod(x.chi, y.chi, z.chi);

            return T{rho, v, pre};
        }

        GPU_CALLABLE_INLINE 
        constexpr luint get_real_idx(const luint idx, const luint offset, const luint active_zones) 
        {
            lint real_idx = (idx - offset > 0) * (idx - offset);
            if (idx > active_zones + 1) {
                real_idx = active_zones - 1;
            }
            return real_idx;
        }   
    } // namespace helpers
    
    
} // namespace simmbi




#include "helpers.tpp"
#endif