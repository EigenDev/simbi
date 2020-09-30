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

//---------------------------------------------------------------------------------------------------------
//  NEWTON-RAPHSON 
//---------------------------------------------------------------------------------------------------------
template <typename T, typename... Args>
double newton_raphson(T x, T (*f)(T, Args... args),  T (*g)(T, Args... args), 
                    double epsilon, Args... args);




double findMax(double, double, double);
double calc_sound_speed(float, double, double);
int sign(double);
double minmod(double, double, double);
std::vector<double> rollVector(const std::vector<double>&, unsigned int);
double roll(std::vector<double>&, unsigned int);
std::vector<std::vector<double> > transpose(std::vector<std::vector<double> > &);
void config_ghosts1D(std::vector<std::vector<double> > &, int, bool=true);
void config_ghosts2D(std::vector<std::vector< std::vector< double> > > &, int, int, bool=true);
double calc_pressure(float, double, double, double);
double calc_energy(float, double, double, double);
double calc_enthalpy(float, double, double);
double epsilon_rel(double, double, double, double);
double central_difference(double, double (*f)(double), double);
double calc_velocity(double, double, double, double, double=1);
int kronecker(int, int);
double calc_lorentz_gamma(double v);
std::vector<double> calc_lorentz_gamma(std::vector<double> &v);
std::vector<std::vector<double> > calc_lorentz_gamma(std::vector<std::vector<double> > &v1,
                                         std::vector<std::vector<double> > &v2);
double calc_rel_sound_speed(double, double, double, double, float);
double pressure_func(double, double , double , double, float, double);
double dfdp(double, double, double, double, float, double);

#include "root_finder.tpp"
#endif 