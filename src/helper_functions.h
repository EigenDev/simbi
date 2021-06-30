/* 
* Helper functions for computation across all dimensions
* of the hydrodyanmic simulations for better readability/organization
* of the code
*
* Marcus DuPont
* New York University
* 04/09/2020
*/

#ifndef HELPER_FUNCTIONS_H
#define HELPER_FUNCTIONS_H

#include <vector>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <map>
#include <memory>
#include "H5Cpp.h"
#include "hydro_structs.h"
#include "config.h"
#include "traits.h"

//---------------------------------------------------------------------------------------------------------
//  HELPER-GLOBAL-STRUCTS
//---------------------------------------------------------------------------------------------------------

struct PrimData
{
    std::vector<double> rho, v1, v2, p, v;
};

struct MinMod
{
    PrimData prims;
    double theta, NX;
    int active_zones, i_bound, j_bound;

    std::vector<double> rhoL, rhoR, rhoT, rhoB;
    std::vector<double> v1L, v1R, v1T, v1B;
    std::vector<double> v2L, v2R, v2T, v2B;
    std::vector<double> pL, pR, pT, pB;

    void compute(int face);
};

struct DataWriteMembers
{
    double t, ad_gamma;
    double xmin, xmax, ymin, ymax, dt;
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
void config_ghosts2D(T &v, int, int, bool = false);

template <typename T>
std::vector<double> calc_lorentz_gamma(T &v);

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
constexpr int sgn(T val) { return (T(0) < val) - (val < T(0)); }

template<typename T>
constexpr T roll(const std::vector<T>  &v, unsigned int n) { return v[n % v.size()];}

//---------------------------------------------------------------------------------------------------------
//  HELPER-METHODS
//---------------------------------------------------------------------------------------------------------

//----------------Define Methods-------------------------
sr2d::PrimitiveData vecs2struct(const std::vector<sr2d::Primitive> &p);
std::vector<double> rollVector(const std::vector<double> &, unsigned int);
double roll(std::vector<double> &, unsigned int);
double roll(std::vector<std::vector<double>> &, unsigned int xpos, unsigned int ypos);
void toWritePrim(sr1d::PrimitiveArray *from, PrimData *to);
void toWritePrim(sr2d::PrimitiveData *from, PrimData *to);
std::string create_step_str(double t_interval, std::string &tnow);
void write_hdf5(
    const std::string data_directory, 
    const std::string filename, 
    const PrimData prims, 
    const DataWriteMembers system, 
    const int dim, 
    const int size);

void write_data(std::vector<std::vector<double>> &sim_data, double time, std::string sim_type);
double calc_intermed_wave(double, double, double, double);
double calc_intermed_pressure(double, double, double, double, double, double);
double pressure_func(double, double, double, double, float, double);
double dfdp(double, double, double, double, float, double);
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
inline double minmod(const double x, const double y, const double z)
{
    return 0.25 * std::abs(sgn(x) + sgn(y)) * (sgn(x) + sgn(z)) * std::min({std::abs(x), std::abs(y), std::abs(z)});
};

#include "helpers.tpp"
#endif