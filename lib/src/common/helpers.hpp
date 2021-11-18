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
#include "config.hpp"
#include "traits.hpp"

// Some useful global constants
constexpr real QUIRK_THRESHOLD = 1e-4;
// Calculate a static PI
constexpr double PI = 3.14159265358979323846;

//---------------------------------------------------------------------------------------------------------
//  HELPER-GLOBAL-STRUCTS
//---------------------------------------------------------------------------------------------------------

struct PrimData
{
    std::vector<real> rho, v1, v2, v3, p, v, chi;
};

struct MinMod
{
    PrimData prims;
    real theta, NX;
    int active_zones, i_bound, j_bound;

    std::vector<real> rhoL, rhoR, rhoT, rhoB;
    std::vector<real> v1L, v1R, v1T, v1B;
    std::vector<real> v2L, v2R, v2T, v2B;
    std::vector<real> pL, pR, pT, pB;

    void compute(int face);
};

struct DataWriteMembers
{
    real t, ad_gamma;
    real xmin, xmax, ymin, ymax, zmin, zmax, dt;
    int nx, ny, nz, xactive_zones, yactive_zones, zactive_zones;
    bool linspace, first_order;
    std::string coord_system;
};



extern std::map<std::string, simbi::Geometry> geometry;
void config_system();
//---------------------------------------------------------------------------------------------------------
//  HELPER-TEMPLATES
//---------------------------------------------------------------------------------------------------------
//-------------Define Function Templates-------------------------
template <typename T, size_t N>
constexpr size_t array_size(T (&)[N]);

template <typename T>
void config_ghosts1D(T &v, int, bool = true);

template <typename T>
void config_ghosts2D(T &v, int, int, bool = false);

template <typename T>
std::vector<real> calc_lorentz_gamma(T &v);

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

template<typename T>
GPU_CALLABLE_INLINE constexpr T roll(const T *v, unsigned int n, int size) { return v[n % size];}

template<typename T>
constexpr T roll(const std::vector<T>  &v, unsigned int n) { return v[n % v.size()];}

//---------------------------------------------------------------------------------------------------------
//  HELPER-METHODS
//---------------------------------------------------------------------------------------------------------

void config_gpu_space();
//----------------Define Methods-------------------------
std::string create_step_str(real t_interval, std::string &tnow);
void write_hdf5(
    const std::string data_directory, 
    const std::string filename, 
    const PrimData prims, 
    const DataWriteMembers system, 
    const int dim, 
    const int size);

void config_ghosts1D(std::vector<hydro1d::Conserved> &u_state, int grid_size, bool first_order);
    
real calc_intermed_wave(real, real, real, real);
real calc_intermed_pressure(real, real, real, real, real, real);
real pressure_func(real, real, real, real, float, real);
real dfdp(real, real, real, real, float, real);
void config_ghosts2D(
    std::vector<hydro2d::Conserved> &u_state, 
    int x1grid_size, 
    int x2grid_size, 
    bool first_order);

void config_ghosts2D(
    std::vector<sr2d::Conserved> &u_state, 
    int x1grid_size, 
    int x2grid_size, 
    bool first_order,
    bool bipolar = true);




//-------------------Inline for Speed--------------------------------------
GPU_CALLABLE_INLINE real minmod(const real x, const real y, const real z)
{
    return 0.25 * std::abs(sgn(x) + sgn(y)) * (sgn(x) + sgn(z)) * my_min(my_min(std::abs(x), std::abs(y)) , std::abs(z)) ;
};

template<typename T>
GPU_CALLABLE_INLINE typename std::enable_if<is_2D_primitive<T>::value, T>::type
minmod(const T x, const T y, const T z)
{
    const real rho = minmod(x.rho, y.rho, z.rho);
    const real v1  = minmod(x.v1, y.v1, z.v1);
    const real v2  = minmod(x.v2, y.v2, z.v2);
    const real pre = minmod(x.p, y.p, z.p);
    const real chi = minmod(x.chi, y.chi, z.chi);

    return T{rho, v1, v2, pre, chi};

}


#include "helpers.tpp"
#endif