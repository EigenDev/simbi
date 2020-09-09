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
double newton_raphson(double, double (*f)(double),  double (*g)(double), double = 1.e-5);
double central_difference(double, double (*f)(double), double);
double calc_velocity(double, double, double, double, double=1);
int kronecker(int, int);
double lorentz_gamma(double v);
double calc_rel_sound_speed(float, double);
double f(float, double , double , double );
double dfdp(float, double, double, double);
#endif 