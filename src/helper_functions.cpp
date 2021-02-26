
/*
* helper_functions.cpp is where all of the universal functions that can be used
* for all N-Dim hydro calculations
*/

#include "helper_functions.h" 
#include "ustate.h"
#include <cmath>
#include <map>
#include <algorithm>
#include <cstdarg>


using namespace std;
using namespace states;

// =========================================================================================================
//        HELPER FUNCTIONS FOR COMPUTATION
// =========================================================================================================


// Find the max element
double findMax(double a, double b, double c ){
    //Find max b/w a & b first
    double inter_max = max(a, b);

    double max_val = max(inter_max, c);

    return max_val;
};

double findMin(double a, double b, double c ){
    //Find max b/w a & b first
    double inter_min = min(a, b);

    double min_val = min(inter_min, c);

    return min_val;
};

// Sound Speed Function
double calc_sound_speed(float gamma, double rho, double pressure){
    double c = sqrt(gamma*pressure/rho);
    return c;

};

// Roll a vector for use with periodic boundary conditions
vector<double> rollVector(const vector<double>& v, unsigned int n){
    auto b = v.begin() + (n % v.size());
    vector<double> ret(b, v.end());
    ret.insert(ret.end(), v.begin(), b);
    return ret;
};

// Roll a single vector index
double roll(vector<double>  &v, unsigned int n) {
   return v[n % v.size()];
};

// Roll a single vector index in y-direction of lattice
double roll(vector<vector<double>>  &v, unsigned int xpos, unsigned int ypos) {
   return v[ypos % v.size()][xpos % v[0].size()];
};
vector<vector<double> > transpose(vector<vector<double> > &mat){

    vector<vector<double> > trans_vec(mat[0].size(), vector<double>());

    int y_size = mat.size();
    int x_size = mat[0].size(); 

    for (int i = 0; i < x_size; i++)
    {
        for (int j = 0; j < y_size; j++)
        {
            if (trans_vec[j].size() != mat.size()){

                trans_vec[j].resize(mat.size());

            }
            
            trans_vec[j][i] = mat[i][j];
        }
    }

    return trans_vec;
};

//====================================================================================================
//                                  WRITE DATA TO FILE
//====================================================================================================

string create_step_str(double t_interval, string &tnow){

    // Convert the time interval into an int with 2 decimal displacements
    int t_interval_int = round( 1.e3 * t_interval );
    int a, b;

    string s = to_string(t_interval_int);

    // Pad the file string if size less than 6
    if (s.size() < tnow.size()) {

        int num_zeros = tnow.size() - s.size();
        string pad_zeros = string(num_zeros, '0');
        s.insert(0, pad_zeros);

    }

    for (int i = 0; i < 6; i++){
        a = tnow[i] - '0';
        b = s[i] - '0';
        s[i] = a + b + '0';
    }

    return s;


}
void write_hdf5(string filename, PrimData prims, DataWriteMembers setup)
{
    string filePath = "data/sr/";
    cout << "\n" <<  "Writing File...: " << filePath + filename << endl;
    h5::fd_t fd = h5::create(filePath + filename, H5F_ACC_TRUNC, h5::default_fcpl,
                    h5::libver_bounds({H5F_LIBVER_V18, H5F_LIBVER_V18}) );

    // Write the Primitives 
    h5::write(fd,"rho", prims.rho);
    h5::write(fd,"v1",  prims.v1);
    h5::write(fd,"v2",  prims.v2);
    h5::write(fd,"p",   prims.p);

    // Generate Dataset Object to Write Attributes
    vector<int> sim_data(10, 0);
    auto ds = h5::write(fd, "sim_info", sim_data);

    // Write Datset Attributes
    ds["current_time"]   = setup.t;
    ds["time_step"]      = setup.dt;
    ds["NX"]             = setup.NX;
    ds["NY"]             = setup.NY;
    ds["xmax"]           = setup.xmax;
    ds["xmin"]           = setup.xmin;
    ds["ymax"]           = setup.ymax;
    ds["ymin"]           = setup.ymin;
    ds["xactive_zones"]  = setup.xactive_zones;
    ds["yactive_zones"]  = setup.yactive_zones;
    
}



//                                      NEWTONIAN HYDRO
//=======================================================================================================

//----------------------------------------------------------------------------------------------------------
//  PRESSURE CALCULATIONS
//---------------------------------------------------------------------------------------------------------

double calc_pressure(float gamma, double rho, double energy, double v){
    double pressure = (gamma - 1.)*(energy - 0.5*rho*v*v);
    return pressure;
};

//------------------------------------------------------------------------------------------------------------
//  ENERGY CALCULATIONS
//------------------------------------------------------------------------------------------------------------

double calc_energy(float gamma, double rho, double pressure, double v){
        return pressure/(gamma-1.) + 0.5*rho*v*v;
};


//=======================================================================================================
//                                      RELATIVISITC HYDRO
//=======================================================================================================
int kronecker(int i, int j){
    if (i == j){
        return 1;
    } else{
        return 0;
    }
}


//------------------------------------------------------------------------------------------------------------
//  SPECIFIC ENTHALPY CALCULATIONS
//------------------------------------------------------------------------------------------------------------
double calc_enthalpy(float gamma, double rho, double pressure){
        return 1 + gamma*pressure/(rho*(gamma - 1));
};

double epsilon_rel(double pressure, double D, double tau, double lorentz_gamma){
    return ( tau + D*(1. - lorentz_gamma) + (1. - lorentz_gamma*lorentz_gamma)*pressure )/(D*lorentz_gamma);
}

double rho_rel(double D, double lorentz_gamma, double root_g){
    return D/(lorentz_gamma*root_g);
}
//------------------------------------------------------------------------------------------------------------
//  VELOCITY CALCULATION
//------------------------------------------------------------------------------------------------------------
double calc_velocity(double s, double tau, double pressure, double D, double root_g){
    // Compute the 3-velocity given relaitivistic quanrities
    return s/(tau + root_g*pressure + D);
}

double calc_intermed_wave(double energy_density, double momentum_density, 
                            double flux_momentum_density, 
                            double flux_energy_density)
{
    double a = flux_energy_density;
    double b = - (energy_density + flux_momentum_density);
    double c = momentum_density;
    double disc = sqrt( b*b - 4*a*c);
  
    double quad = -0.5*(b + sgn(b)*disc);
    return c/quad;
    
    
}


double calc_intermed_pressure(double a,double aStar, double energy, double norm_mom, double u, double p){

    double e, f, g;
    e = (a*energy - norm_mom)*aStar;
    f = norm_mom*(a - u) - p;
    g = 1 + a*aStar;

    return (e - f)/g;
}
//------------------------------------------------------------------------------------------------------------
//  LORENTZ FACTOR CALCULATION
//------------------------------------------------------------------------------------------------------------



double calc_rel_sound_speed(double pressure, double D, double tau, double lorentz_gamma, float gamma){
    double epsilon = epsilon_rel(pressure, D, tau, lorentz_gamma);

    return sqrt((gamma - 1)*gamma*epsilon/(1 + gamma*epsilon));
}
//------------------------------------------------------------------------------------------------------------
//  F-FUNCTION FOR ROOT FINDING: F(P)
//------------------------------------------------------------------------------------------------------------
double pressure_func(double pressure, double D, double tau, double lorentz_gamma, float gamma, double S){

    double rho = D/lorentz_gamma; 
    double epsilon = epsilon_rel(pressure, D, tau, lorentz_gamma);

    return (gamma - 1.)*rho*epsilon - pressure;
}

double dfdp(double pressure, double D, double tau, double lorentz_gamma, float gamma, double S){
    double rho = D/lorentz_gamma;
    double h = 1 + pressure*gamma/(rho*(gamma - 1.));
    double cs = sqrt(gamma*pressure/(h*rho) );
    double v = S/(tau + D + pressure);

    return v*v*cs*cs - 1.;
}
