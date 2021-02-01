/* 
* Helper functions for computation across all dimensions
* of the hydrodyanmic simulations for better readability/organization
* of the code
*
* Marcus DuPont
* New York University
* 09/04/2020
*/

#ifndef HELPER_FUNCTIONS_H
#define HELPER_FUNCTIONS_H

#include <vector>
#include <iostream>
#include <cmath>
#include <h5cpp/all>
#include "ustate.h"

//---------------------------------------------------------------------------------------------------------
//  HELPER-TEMPLATES 
//---------------------------------------------------------------------------------------------------------
struct PrimData
{
    std::vector<double> rho, v1, v2, p, v;
};

template <typename T, size_t N>
constexpr size_t array_size(T (&)[N]);


template <typename T, typename... Args>
double newton_raphson(T x, T (*f)(T, Args... args),  T (*g)(T, Args... args), 
                    double epsilon, Args... args);

template <typename T>
void config_ghosts1D(T &v, int, bool=true);

template <typename T>
void config_ghosts2D(T &v, int, int, bool=true, std::string kind="outflow");

template <typename T>
std::vector<double> calc_lorentz_gamma(T &v);

template <typename T>
std::vector<double> calc_lorentz_gamma(T &v1, T &v2, int nx, int ny);

template <typename T>
void toWritePrim(T *from, PrimData *to, int ndim = 1);



double findMax(double, double, double);
double findMin(double a, double b, double c);
double calc_sound_speed(float, double, double);
int sign(double);
double minmod(double, double, double);
std::vector<double> rollVector(const std::vector<double>&, unsigned int);
double roll(std::vector<double>&, unsigned int);
double roll(std::vector<std::vector<double>>&, unsigned int xpos, unsigned int ypos);
std::vector<std::vector<double> > transpose(std::vector<std::vector<double> > &);

void write_hdf5(std::string filename, PrimData prims, float t, double dt, int NX, int NY);
// void config_ghosts1D(std::vector<std::vector<double> > &, int, bool=true);
// void config_ghosts2D(std::vector<std::vector< std::vector< double> > > &, int, int, bool=true, char kind='outflow');
void write_data(std::vector<std::vector<double> > &sim_data,double time, std::string sim_type );
double calc_pressure(float, double, double, double);
double calc_energy(float, double, double, double);
double calc_enthalpy(float, double, double);
double epsilon_rel(double, double, double, double);
double central_difference(double, double (*f)(double), double);
double calc_velocity(double, double, double, double, double=1);
double calc_intermed_wave(double, double, double, double);
double calc_intermed_pressure(double, double, double, double, double, double);
int kronecker(int, int);
double calc_lorentz_gamma(double v);
// std::vector<double> calc_lorentz_gamma(std::vector<double> &v);
// std::vector<std::vector<double> > calc_lorentz_gamma(std::vector<std::vector<double> > &v1,
//                                          std::vector<std::vector<double> > &v2);
double calc_rel_sound_speed(double, double, double, double, float);
double pressure_func(double, double , double , double, float, double);
double dfdp(double, double, double, double, float, double);

#include "helper_functions.tpp"
#endif 