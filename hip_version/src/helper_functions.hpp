/* 
* Helper functions for computation across all dimensions
* of the hydrodyanmic simulations for better readability/organization
* of the code
*
* Marcus DuPont
* New York University
* 04/09/2020
*/

#ifndef HELPER_FUNCTIONS_HPP
#define HELPER_FUNCTIONS_HPP

#include <vector>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <map>
#include <memory>
#include "hip/hip_runtime.h"
#include "H5Cpp.h"
#include "hydro_structs.hpp"
#include "config.hpp"
#include "traits.hpp"
#include "gpu_error_check.h"
#include "srhd_2d.hpp"


//---------------------------------------------------------------------------------------------------------
//  HELPER-GLOBAL-STRUCTS
//---------------------------------------------------------------------------------------------------------

struct PrimData
{
    std::vector<real> rho, v1, v2, p, v;
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
    real xmin, xmax, ymin, ymax, dt;
    int NX, NY, xactive_zones, yactive_zones;
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
__global__ void config_ghosts1DGPU(T *dev_sim, int, bool = true);

template <typename T>
void config_ghosts2D(T &v, int, int, bool = false);

template <typename T>
std::vector<real> calc_lorentz_gamma(T &v);

template <typename T>
void toWritePrim(T *from, PrimData *to, int ndim = 1);

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
GPU_CALLABLE_MEMBER
constexpr int sgn(T val) { return (T(0) < val) - (val < T(0)); }


template<typename T>
GPU_CALLABLE_MEMBER
T roll(const T &v, unsigned int n, int size);


template<typename T>
constexpr T roll(const std::vector<T>  &v, unsigned int n) { return v[n % v.size()];}

//---------------------------------------------------------------------------------------------------------
//  HELPER-METHODS
//---------------------------------------------------------------------------------------------------------

void config_gpu_space();
//----------------Define Methods-------------------------
sr2d::PrimitiveData vecs2struct(const std::vector<sr2d::Primitive> &p);
std::vector<real> rollVector(const std::vector<real> &, unsigned int);
real roll(std::vector<real> &, unsigned int);
// real roll(std::vector<std::vector<real>> &, unsigned int xpos, unsigned int ypos);
void toWritePrim(sr1d::PrimitiveArray *from, PrimData *to);
void toWritePrim(sr2d::PrimitiveData *from, PrimData *to);
std::string create_step_str(real t_interval, std::string &tnow);
void write_hdf5(
    const std::string data_directory, 
    const std::string filename, 
    const PrimData prims, 
    const DataWriteMembers system, 
    const int dim, 
    const int size);

__global__ void config_ghosts2DGPU(
    simbi::SRHD2D *d_sim, 
    int x1grid_size, 
    int x2grid_size, 
    bool first_order,
    bool bipolar = true);
    
GPU_CALLABLE_MEMBER real calc_intermed_wave(real, real, real, real);
GPU_CALLABLE_MEMBER real calc_intermed_pressure(real, real, real, real, real, real);
GPU_CALLABLE_MEMBER real pressure_func(real, real, real, real, float, real);
GPU_CALLABLE_MEMBER real dfdp(real, real, real, real, float, real);
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
GPU_CALLABLE_MEMBER inline real minmod(const real x, const real y, const real z)
{
    return 0.25 * abs(sgn(x) + sgn(y)) * (sgn(x) + sgn(z)) * fmin( fmin(abs(x), abs(y)) , abs(z)) ;
};

#include "helpers.tpp"
#endif