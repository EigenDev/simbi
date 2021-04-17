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
#include <algorithm>
#include <map>
#include "hydro_structs.h"

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

    std::vector<double> rhoL,rhoR, rhoT, rhoB;
    std::vector<double> v1L,v1R, v1T, v1B;
    std::vector<double> v2L,v2R, v2T, v2B;
    std::vector<double> pL,pR, pT, pB;

    void compute(int face);
};

struct DataWriteMembers
{
    float t;
    double xmin, xmax, ymin, ymax, dt;
    int NX, NY, xactive_zones, yactive_zones;

};

namespace waves {
    enum wave_loc {
        LEFT_WAVE  = 1, 
        LEFT_STAR  = 2, 
        RIGHT_STAR = 3, 
        RIGHT_WAVE = 4
    };
};

namespace simulation {
    enum coord_system {
        CARTESIAN,
        SPHERICAL,
    };

    enum accuracy {
        FIRST_ORDER,
        SECOND_ORDER,
    };

    enum dimensions {
        ONE_D,
        TWO_D,
    };

    enum solver {
        HLLC,
        HLLE,
    };

    enum wave_speeds {
        SCHNEIDER_ET_AL_93,
        MIGNONE_AND_BODO_05,
    };
};

extern std::map<std::string, simulation::coord_system> geometry;
extern std::map<bool, simulation::accuracy> order_acc;
extern std::map<bool, waves::wave_loc> wave_loc;
void config_system();
void get_spacetime_wave(double aL, double aR, double aStar);

//---------------------------------------------------------------------------------------------------------
//  HELPER-TEMPLATES 
//---------------------------------------------------------------------------------------------------------
//-------------Define Function Templates-------------------------
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

template <typename ...Args>
std::string string_format(const std::string& format, Args ...args );

template <typename T> int sgn(T val) ;


//---------------------------------------------------------------------------------------------------------
//  HELPER-METHODS 
//---------------------------------------------------------------------------------------------------------

//----------------Define Methods-------------------------
double findMax(double, double, double);
double findMin(double a, double b, double c);
double calc_sound_speed(float, double, double);
hydro2d::PrimitiveData vecs2struct(std::vector<hydro2d::Primitives> &p);
void compute_vertices(std::vector<double> &cz, std::vector<double> &xv, int lx, bool linspace=true);
std::vector<double> rollVector(const std::vector<double>&, unsigned int);
double roll(std::vector<double>&, unsigned int);
double roll(std::vector<std::vector<double>>&, unsigned int xpos, unsigned int ypos);
std::vector<std::vector<double> > transpose(std::vector<std::vector<double> > &);
void toWritePrim(hydro1d::PrimitiveArray *from, PrimData *to, int ndim);
std::string create_step_str(double t_interval, std::string &tnow);
void write_hdf5(std::string data_directory, std::string filename, PrimData prims, DataWriteMembers system, int dim);
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
double calc_rel_sound_speed(double, double, double, double, float);
double pressure_func(double, double , double , double, float, double);
double dfdp(double, double, double, double, float, double);

//-------------------Inline for Speed--------------------------------------
inline double minmod(const double &x, const double &y , const double &z){
    return 0.25*std::abs(sgn(x) + sgn(y))*(sgn(x) + sgn(z))*std::min({std::abs(x), std::abs(y), std::abs(z)});
};

#include "helper_functions.tpp"
#endif 