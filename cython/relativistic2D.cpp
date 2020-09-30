/* 
* C++ Library to perform 2D hydro calculations
* Marcus DuPont
* New York University
* 07/15/2020
* Compressible Hydro Simulation
*/

#include "ustate.h" 
#include "helper_functions.h"
#include <cmath>
#include <map>
#include <algorithm>

using namespace std;
using namespace states;

// Default Constructor 
UstateSR2D::UstateSR2D () {}

// Overloaded Constructor
UstateSR2D::UstateSR2D(vector<vector< vector<double> > > u_state2D, float Gamma, vector<double> X1, 
                    vector<double> X2, double cfl, string Coord_system = "cartesian")
{
    state2D = u_state2D;
    gamma = Gamma;
    x1 = X1;
    x2 = X2;
    CFL = cfl;
    coord_system = Coord_system;
}

// Destructor 
UstateSR2D::~UstateSR2D() {}


//-----------------------------------------------------------------------------------------
//                          GET THE PRIMITIVES
//-----------------------------------------------------------------------------------------

// Return a 1D array containing (rho, pressure, v) at a *single grid point*
vector<double>  cons2primSR(float gamma, vector<double>  &u_state, double lorentz_gamma){
    /**
     * Return a vector containing the primitive
     * variables density (rho), pressure, and
     * velocity (v)
     */
    vector<double> prims(4);

    double D = u_state[0];
    double S1 = u_state[1];
    double S2 = u_state[2];
    double tau = u_state[3];
    
    double S = sqrt(S1*S1 + S2*S2);

    // cout << "D: " << D << endl;
    // cout << "S: " << S << endl;
    // cout << "Tau: " << tau << endl;


    double pmin = abs(S - tau - D);

    // cout << "Pmin: " << pmin << endl;
    
    double pressure = newton_raphson(pmin, pressure_func, dfdp, 1.e-7, D, tau, lorentz_gamma, gamma, S);

    // cout << "NR Pressure: " << pressure << endl;

    double v1 = S1/(tau + pressure + D);
    double v2 = S2/(tau + pressure + D);

    double vtot = sqrt(v1*v1 + v2*v2);

    double Wnew = calc_lorentz_gamma(vtot);

    

    double rho = D/Wnew;

    prims[0] = rho;
    prims[1] = pressure;
    prims[2] = v1;
    prims[3] = v2;

    return prims;
    
    
};


vector<vector< vector<double> > > UstateSR2D::cons2prim2D(vector<vector< vector<double> > > &u_state2D,
                                                            vector<vector<double> > &lorentz_gamma){
    /**
     * Return a 2D matrix containing the primitive
     * variables density (rho), pressure, and
     * velocity (v)
     */
    double rho, S1,S2, S, D, tau, pmin;
    double pressure, W;
    double v1, v2, vtot;
     
    int n_vars = u_state2D.size();
    int ny_gridpts = u_state2D[0].size();
    int nx_gridpts = u_state2D[0][0].size();

    vector<vector<vector<double> > > prims(n_vars, vector<vector<double> > 
                                            (ny_gridpts, vector<double>(nx_gridpts)));
   

    for (int jj=0; jj < ny_gridpts; jj++){
        for(int ii=0; ii< nx_gridpts; ii++){
            D = u_state2D[0][jj][ii];       // Relativist Density
            S1 = u_state2D[1][jj][ii];      // X1-Momentum Denity
            S2 = u_state2D[2][jj][ii];      // x2-Momentum Density
            tau = u_state2D[3][jj][ii];    // Energy Density
            W = lorentz_gamma[jj][ii];

            

            S = sqrt(S1*S1 + S2*S2);

            pmin = abs(S - tau - D);

            pressure = newton_raphson(pmin, pressure_func, dfdp, 1.e-6, D, tau, W, gamma, S);

            // cout << pressure << endl;
            // cin.get();

            v1 = S1/(tau + D + pressure);
            v2 = S2/(tau + D + pressure);

            vtot = sqrt( v1*v1 + v2*v2 );

            W = calc_lorentz_gamma(vtot);

            rho = D/W;

            
            prims[0][jj][ii] = rho;
            prims[1][jj][ii] = pressure;
            prims[2][jj][ii] = v1;
            prims[3][jj][ii] = v2;
            

        }
    }
    

    return prims;
};




//-----------------------------------------------------------------------------------------
//                              CALCULATE THE STATE TENSOR
//-----------------------------------------------------------------------------------------

// Get the 2-Dimensional (4, 1) state tensor for computation. 
// It is being doing pointwise in this case as opposed to over
// the entire array since we are in c++
 vector<double>  calc_stateSR2D(float gamma, double rho, double pressure, double vx, double vy)
{
    double D, S1, S2, tau, h, vtot, lorentz_gamma;

    vtot = sqrt(vx*vx + vy*vy);
    lorentz_gamma = calc_lorentz_gamma(vtot);

    h = calc_enthalpy(gamma, rho, pressure);
    D = rho*lorentz_gamma; 
    S1 = rho*h*lorentz_gamma*lorentz_gamma*vx;
    S2 = rho*h*lorentz_gamma*lorentz_gamma*vy;
    tau = rho*h*lorentz_gamma*lorentz_gamma - pressure - rho*lorentz_gamma;
    

    vector<double>  cons_state(4);
    
    
    
    
    cons_state[0] = D; 
    cons_state[1] = S1;
    cons_state[2] = S2;
    cons_state[3] = tau;
        
        
    return cons_state;
};


//----------------------------------------------------------------------------------------------------------
//                              EIGENVALUE CALCULATIONS
//----------------------------------------------------------------------------------------------------------

map<string, map<string, double > > calc_eigenvals(float gamma, vector<double> &left_state,
                                            vector<double> &right_state, double lorentz_left,
                                            double lorentz_right, string direction = "x")
{

    // Initialize your important variables
    double v1_r, v1_l, v2_l, v2_r, p_r, p_l, cs_r, cs_l, vtot_l, vtot_r, D_r, D_l ,tau_r, tau_l; 
    double rho_l, rho_r;
    map<string, map<string, double > > lambda;
    string default_direction = "x";
    string coord_system = "cartesian";
    vector<double> prims_l, prims_r;

    prims_l = cons2primSR(gamma, left_state, lorentz_left);
    prims_r = cons2primSR(gamma, right_state, lorentz_right);

    // Separate the left and right state components
    rho_l = prims_l[0];
    p_l = prims_l[1];
    v1_l = prims_l[2];
    v2_l = prims_l[3];

    rho_r = prims_r[0];
    p_r = prims_r[1];
    v1_r = prims_r[2];
    v2_r = prims_r[3];

    D_l = left_state[0];
    D_r = right_state[0];
    tau_l = left_state[3];
    tau_r = right_state[3];

    cs_r = calc_rel_sound_speed(p_r, D_r, tau_r, lorentz_right, gamma);
    cs_l = calc_rel_sound_speed(p_l, D_l, tau_l, lorentz_left, gamma);


    /**
    if ((sign(p_r) == -1) || (sign(p_l) == -1.) ){
        cout << "I Have Broken: " << endl;
        cout << "Energy: " << tau_r << endl;
        cout << "Kinetic E: " << 0.5*rho_r*vtot_r*vtot_r << endl;
        cout << "Vx: " << vx_r << endl;
        cout << "Vy: " << vy_r << endl;
        cout << "Vtot: " << vtot_r << endl;
        cout << "Momx: " << momx_r << endl; 
        cout << "Rho: " << rho_r << endl;
        cout << "P R: " << p_r << endl;
        string a;
        cin >> a;
    }
    */


    // Populate the lambda dictionary
    if (direction == default_direction){
        lambda["left"]["plus"] = v1_l + cs_l; 
        lambda["left"]["minus"] = v1_l - cs_l; 
        lambda["right"]["plus"] = v1_r + cs_r; 
        lambda["right"]["minus"] = v1_r - cs_r; 
    } else {
        lambda["left"]["plus"] = v2_l + cs_l; 
        lambda["left"]["minus"] = v2_l - cs_l; 
        lambda["right"]["plus"] = v2_r + cs_r; 
        lambda["right"]["minus"] = v2_r - cs_r; 
    }
        
    return lambda;

    
    
};

//---------------------------------------------------------------------
//                  ADAPT THE TIMESTEP
//---------------------------------------------------------------------


// Adapt the CFL conditonal timestep
double UstateSR2D::adapt_dt(vector<vector<vector<double> > > &prims,
                        bool linspace=true){

    double r_left, r_right, left_cell, right_cell, upper_cell, lower_cell;
    double dx1, cs, dx2, x2_right, x2_left, rho, pressure, v1, v2, volAvg, h;
    double delta_logr, log_rLeft, log_rRight, min_dt, cfl_dt, vtot, D, tau, W;
    int shift_i, shift_j, x1physical_grid, x2physical_grid;

    int x1grid_size = prims[0][0].size();
    int x2grid_size = prims[0].size();

    min_dt = 0;
    // Find the minimum timestep over all i
    x1physical_grid = x1grid_size - 4;
    x2physical_grid = x2grid_size - 4;


    // Compute the minimum timestep given CFL
    for (int jj = 0; jj < x2physical_grid; jj ++){
        for (int ii = 0; ii < x1physical_grid; ii++){
            shift_i = ii + 2;
            shift_j = jj + 2;

            // Find the left and right cell centers in one direction
            if (ii - 1 < 0){
                left_cell = x1[ii];
                right_cell = x1[ii + 1];
            }
            else if (ii + 1 > x1physical_grid - 1){
                right_cell = x1[ii];
                left_cell = x1[ii - 1];
            } else {
                right_cell = x1[ii + 1];
                left_cell = x1[ii];
            }

            if (jj - 1 < 0){
                lower_cell = x2[jj];
                upper_cell = x2[jj + 1];
            }
            else if (jj + 1 > x2physical_grid - 1){
                upper_cell = x2[jj];
                lower_cell = x2[jj - 1];
            } else {
                upper_cell = x2[jj + 1];
                lower_cell = x2[jj];
            }

            // Check if using linearly-spaced grid or logspace
            if (linspace){
                r_right = 0.5*(right_cell + x1[ii]);
                r_left = 0.5*(x1[ii] + left_cell);

                if (coord_system == "cartesian"){
                    x2_right = 0.5*(upper_cell + x2[jj]);
                    x2_left = 0.5*(lower_cell + x2[jj]);
                } else {
                    x2_right = atan2 ( sin(upper_cell) + sin(x2[jj]), cos(upper_cell) + cos(x2[jj]) );
                    x2_left = atan2( sin(lower_cell) + sin(x2[jj]), cos(lower_cell) + cos(x2[jj]) );      
                }

            } else {
                delta_logr = (log10(x1[x1physical_grid - 1]) - log10(x1[0]))/x1physical_grid;
                log_rLeft = log10(x1[0]) + ii*delta_logr;
                log_rRight = log_rLeft + delta_logr;
                r_left = pow(10, log_rLeft);
                r_right = pow(10, log_rRight);
            }

            dx1 = r_right - r_left;
            dx2 = x2_right - x2_left;
            rho = prims[0][shift_j][shift_i];
            v1 = prims[2][shift_j][shift_i];
            v2 = prims[3][shift_j][shift_i];
            pressure = prims[1][shift_j][shift_i];

            vtot = sqrt(v1*v1 + v2*v2);
            W = calc_lorentz_gamma(vtot);
            D = rho*W;
            h = calc_enthalpy(gamma, rho, pressure);
            tau = rho*h*W*W - pressure - rho*W;

            
            cs = calc_rel_sound_speed(pressure, D, tau, W, gamma);

            if (coord_system == "cartesian"){
                
                cfl_dt = min( dx1/(max(abs(v1 + cs), abs(v1 - cs))), dx2/(max(abs(v2 + cs), abs(v2 - cs))) );

            } else {
                volAvg = 0.75*( ( pow(r_right, 4) - pow(r_left, 4) )/ ( pow(r_right, 3) - pow(r_left, 3) ) );

                cfl_dt = min( dx1/(max(abs(v1 + cs), abs(v1 - cs))), volAvg*dx2/(max(abs(v2 + cs), abs(v2 - cs))) );

            }

            
            if ((ii > 0) | (jj > 0) ){
                min_dt = min(min_dt, cfl_dt);
            }
            else {
                min_dt = cfl_dt;
            }

            
        }
        
    }

    return CFL*min_dt;
};


//-----------------------------------------------------------------------------------------------------------
//                                            FLUX CALCULATIONS
//-----------------------------------------------------------------------------------------------------------


// Get the 2D Flux array (4,1). Either return F or G depending on directional flag
vector<double> calc_fluxSR2D(float gamma, double rho, double pressure, 
                                        double vx, double vy, bool x_direction=true){
    
    // The Flux Tensor
    vector<double> flux(4);

     // The Flux components
    double h, D, S1, S2, convect_12, tau, zeta;
    double mom1, mom2, energy_dens;

    double vtot = sqrt(vx*vx + vy*vy );
    double lorentz_gamma = calc_lorentz_gamma(vtot);

    h = calc_enthalpy(gamma, rho, pressure);
    D = rho*lorentz_gamma;
    S1 = rho*lorentz_gamma*lorentz_gamma*h*vx;
    S2 = rho*lorentz_gamma*lorentz_gamma*h*vy;
    tau = rho*h*lorentz_gamma*lorentz_gamma - pressure - rho*lorentz_gamma;

    


    // Check if we're calculating the x-direction flux. If not, calculate the y-direction
    if (x_direction){
        mom1 = D*vx;
        convect_12 = S2*vx;
        energy_dens = S1*vx + pressure;
        zeta = (tau + pressure)*vx;

        flux[0] = mom1;
        flux[1] = energy_dens;
        flux[2] = convect_12;
        flux[3] = zeta;
           
        return flux;
    } else {
        mom2 = D*vy;
        convect_12 = S1*vy;
        energy_dens = S2*vy + pressure;
        zeta = (tau + pressure)*vy;

        flux[0] = mom2;
        flux[1] = convect_12;
        flux[2] = energy_dens;
        flux[3] = zeta;
           
        return flux;
    }
    
};


vector<double> calc_hll_flux(float gamma, double lorentz_left, double lorentz_right,
                                        vector<double> &left_state,
                                        vector<double> &right_state,
                                        vector<double> &left_flux,
                                        vector<double> &right_flux,
                                        string direction = "x")
{
    map<string, map<string, double > > lambda; 
    vector<double> hll_flux(4);
    double alpha_plus, alpha_minus;  
    
    lambda = calc_eigenvals(gamma, left_state, right_state, lorentz_left, lorentz_right, direction);

    // Calculate /pm alphas
    alpha_plus = findMax(0, lambda["left"]["plus"], lambda["right"]["plus"]);
    alpha_minus = findMax(0 , -lambda["left"]["minus"], -lambda["right"]["minus"]);

    // Compute the HLL Flux component-wise
    hll_flux[0] = ( alpha_plus*left_flux[0] + alpha_minus*right_flux[0]
                            - alpha_minus*alpha_plus*(right_state[0] - left_state[0] ) )  /
                            (alpha_minus + alpha_plus);

    hll_flux[1] = ( alpha_plus*left_flux[1] + alpha_minus*right_flux[1]
                            - alpha_minus*alpha_plus*(right_state[1] - left_state[1] ) )  /
                            (alpha_minus + alpha_plus);

    hll_flux[2] = ( alpha_plus*left_flux[2] + alpha_minus*right_flux[2]
                            - alpha_minus*alpha_plus*(right_state[2] - left_state[2]) )  /
                            (alpha_minus + alpha_plus);

    hll_flux[3] = ( alpha_plus*left_flux[3] + alpha_minus*right_flux[3]
                            - alpha_minus*alpha_plus*(right_state[3] - left_state[3]) )  /
                            (alpha_minus + alpha_plus);

    return hll_flux;
};



//-----------------------------------------------------------------------------------------------------------
//                                            UDOT CALCULATIONS
//-----------------------------------------------------------------------------------------------------------

vector<vector<vector<double> > > UstateSR2D::u_dot2D(vector<vector<vector<double> > > &u_state, 
                                        vector<vector<double> > &lorentz_gamma, bool periodic = false,
                                        float theta = 1.5, bool linspace=true)
{

    int i_start, i_bound, j_start, j_bound, xcoordinate, ycoordinate;
    int xgrid_size = u_state[0][0].size();
    int ygrid_size = u_state[0].size();
    int xphysical_grid = xgrid_size - 4;
    int yphysical_grid = ygrid_size - 4;
    string default_coordinates = "cartesian";

    double Wx_l, Wx_r, Wy_l, Wy_r, vx_l, vx_r, vy_l, vy_r;
    int n_vars = u_state.size();

    double dx = (x1[xphysical_grid - 1] - x1[0])/xphysical_grid;
    double dy = (x2[yphysical_grid - 1] - x2[0])/yphysical_grid;

    vector<vector<vector<double> > > L(n_vars, vector<vector<double> > 
                                        (yphysical_grid, vector<double> (xphysical_grid, 0)) );

    vector<double>  ux_l(n_vars), ux_r(n_vars), uy_l(n_vars), uy_r(n_vars), f_l(n_vars), f_r(n_vars); 
    vector<double>  f1(n_vars), f2(n_vars), g1(n_vars), g2(n_vars), g_l(n_vars), g_r(n_vars);
    vector<double>   xprims_l(n_vars), xprims_r(n_vars), yprims_l(n_vars), yprims_r(n_vars);

    
    
    // Calculate the primitives for the entire state
    vector<vector<vector<double> > > prims(n_vars, vector<vector<double> > 
                                            (yphysical_grid, vector<double> (xphysical_grid)));

    vector<double> xleft_most(n_vars), xleft_mid(n_vars), xright_mid(n_vars), xright_most(n_vars);
    vector<double> yleft_most(n_vars), yleft_mid(n_vars), yright_mid(n_vars), yright_most(n_vars);
    vector<double> center(n_vars);

    vector<vector<double> > rho_transpose(prims[0].size(), vector<double> (prims[0][0].size()));
    vector<vector<double> > pressure_transpose(prims[0].size(), vector<double> (prims[0][0].size()));
    vector<vector<double> > vx_transpose(prims[0].size(), vector<double> (prims[0][0].size()));
    vector<vector<double> > vy_transpose(prims[0].size(), vector<double> (prims[0][0].size()));

    prims = cons2prim2D(u_state, lorentz_gamma);
    
    // The periodic BC doesn't require ghost cells. Shift the index
    // to the beginning.
    if (periodic){ 
        i_start = 0;
        i_bound = xgrid_size;

        j_start = 0;
        j_bound = ygrid_size;
    } else {
        int true_nxpts = xgrid_size - 2;
        int true_nypts = ygrid_size - 2;
        i_start = 2;
        i_bound = true_nxpts;
        j_start = 2;
        j_bound = true_nypts;
    }

    
    /**
    for (int jj = j_start; jj < j_bound; jj++){
        for (int ii = i_start; ii < i_bound; ii++){
            ycoordinate = jj - 2;
            xcoordinate = ii - 2;

            // i+1/2
            ux_l[0] = u_state[0][jj][ii];
            ux_l[1] = u_state[1][jj][ii];
            ux_l[2] = u_state[2][jj][ii];
            ux_l[3] = u_state[3][jj][ii];

            ux_r[0] = u_state[0][jj][ii + 1];
            ux_r[1] = u_state[1][jj][ii + 1];
            ux_r[2] = u_state[2][jj][ii + 1];
            ux_r[3] = u_state[3][jj][ii + 1];

            // j+1/2
            uy_l[0] = u_state[0][jj][ii];
            uy_l[1] = u_state[1][jj][ii];
            uy_l[2] = u_state[2][jj][ii];
            uy_l[3] = u_state[3][jj][ii];

            uy_r[0] = u_state[0][jj + 1][ii];
            uy_r[1] = u_state[1][jj + 1][ii];
            uy_r[2] = u_state[2][jj + 1][ii];
            uy_r[3] = u_state[3][jj + 1][ii];

            xprims_l = cons2prim(gamma, ux_l);
            xprims_r = cons2prim(gamma, ux_r);
            yprims_l = cons2prim(gamma, uy_l);
            yprims_r = cons2prim(gamma, uy_r);
            
            f_l = calc_fluxSR2D(gamma, xprims_l[0], xprims_l[1], xprims_l[2], xprims_l[3]);
            f_r = calc_fluxSR2D(gamma, xprims_r[0], xprims_r[1], xprims_r[2], xprims_r[3]);

            g_l = calc_fluxSR2D(gamma, yprims_l[0], yprims_l[1], yprims_l[2], yprims_l[3], false);
            g_r = calc_fluxSR2D(gamma, yprims_r[0], yprims_r[1], yprims_r[2], yprims_r[3], false);

            // Calc HLL Flux at i+1/2 interface
            f1 = calc_hll_flux(gamma, ux_l, ux_r, f_l, f_r, 1, 2, "x");
            g1 = calc_hll_flux(gamma, uy_l, uy_r, g_l, g_r, 1, 2, "y");

            // Set up the left and right state interfaces for i-1/2

            // i-1/2
            ux_l[0] = u_state[0][jj][ii - 1];
            ux_l[1] = u_state[1][jj][ii - 1];
            ux_l[2] = u_state[2][jj][ii - 1];
            ux_l[3] = u_state[3][jj][ii - 1];

            ux_r[0] = u_state[0][jj][ii];
            ux_r[1] = u_state[1][jj][ii];
            ux_r[2] = u_state[2][jj][ii];
            ux_r[3] = u_state[3][jj][ii];

            // j-1/2
            uy_l[0] = u_state[0][jj - 1][ii];
            uy_l[1] = u_state[1][jj - 1][ii];
            uy_l[2] = u_state[2][jj - 1][ii];
            uy_l[3] = u_state[3][jj - 1][ii];

            uy_r[0] = u_state[0][jj][ii];
            uy_r[1] = u_state[1][jj][ii];
            uy_r[2] = u_state[2][jj][ii];
            uy_r[3] = u_state[3][jj][ii];

            xprims_l = cons2prim(gamma, ux_l);
            xprims_r = cons2prim(gamma, ux_r);
            yprims_l = cons2prim(gamma, uy_l);
            yprims_r = cons2prim(gamma, uy_r);

            f_l = calc_fluxSR2D(gamma, xprims_l[0], xprims_l[1], xprims_l[2], xprims_l[3]);
            f_r = calc_fluxSR2D(gamma, xprims_r[0], xprims_r[1], xprims_r[2], xprims_r[3]);

            g_l = calc_fluxSR2D(gamma, yprims_l[0], yprims_l[1], yprims_l[2], yprims_l[3], false);
            g_r = calc_fluxSR2D(gamma, yprims_r[0], yprims_r[1], yprims_r[2], yprims_r[3], false);

            // Calc HLL Flux at i+1/2 interface
            f2 = calc_hll_flux(gamma, ux_l, ux_r, f_l, f_r, 1, 2, "x");
            g2 = calc_hll_flux(gamma, uy_l, uy_r, g_l, g_r, 1, 2, "y");
            

            L[0][ycoordinate][xcoordinate] = - (f1[0] - f2[0])/dx - (g1[0] - g2[0])/dy;
            L[1][ycoordinate][xcoordinate] = - (f1[1] - f2[1])/dx - (g1[1] - g2[1])/dy;
            L[2][ycoordinate][xcoordinate] = - (f1[2] - f2[2])/dx - (g1[2] - g2[2])/dy;
            L[3][ycoordinate][xcoordinate] = - (f1[3] - f2[3])/dx - (g1[3] - g2[3])/dy;

        }
    }
    */

    // cout << coord_system << endl;
    // string a;
    // cin >> a;
    if (coord_system == "cartesian"){
        // cout << "Am Cartesian" << endl;
        for (int jj = j_start; jj < j_bound; jj++){
            for (int ii = i_start; ii < i_bound; ii++){
                if (periodic){
                    xcoordinate = ii;
                    ycoordinate = jj;

                    // Declare the c[i-2],c[i-1],c_i,c[i+1], c[i+2] variables

                    // X Coordinate
                    xleft_most[0] = roll(prims[0][ii], ii - 2);
                    xleft_mid[0] = roll(prims[0][ii], ii - 1);
                    center[0] = prims[0][ii][jj];
                    xright_mid[0] = roll(prims[0][ii], ii + 1);
                    xright_most[0] = roll(prims[0][ii], ii + 2);

                    xleft_most[1] = roll(prims[1][ii], ii - 2);
                    xleft_mid[1] = roll(prims[1][ii], ii - 1);
                    center[1] = prims[1][ii][jj];
                    xright_mid[1] = roll(prims[1][ii], ii + 1);
                    xright_most[1] = roll(prims[1][ii], ii + 2);

                    xleft_most[2] = roll(prims[2][ii], ii - 2);
                    xleft_mid[2] = roll(prims[2][ii], ii - 1);
                    center[2] = prims[2][ii][jj];
                    xright_mid[2] = roll(prims[2][ii], ii + 1);
                    xright_most[2] = roll(prims[2][ii], ii + 2);

                    xleft_most[3] = roll(prims[3][ii], ii - 2);
                    xleft_mid[3] = roll(prims[3][ii], ii - 1);
                    center[3] = prims[3][ii][jj];
                    xright_mid[3] = roll(prims[3][ii], ii + 1);
                    xright_most[3] = roll(prims[3][ii], ii + 2);

                    // Transpose the prims matrix to compute the Y Sweep
                    rho_transpose = transpose(prims[0]);
                    pressure_transpose = transpose(prims[1]);
                    vx_transpose = transpose(prims[2]);
                    vy_transpose = transpose(prims[3]);

                    yleft_most[0] = roll(rho_transpose[ii], ii - 2);
                    yleft_mid[0] = roll(rho_transpose[ii], ii - 1);
                    yright_mid[0] = roll(rho_transpose[ii], ii + 1);
                    yright_most[0] = roll(rho_transpose[ii], ii + 2);

                    yleft_most[1] = roll(pressure_transpose[ii], ii - 2);
                    yleft_mid[1] = roll(pressure_transpose[ii], ii - 1);
                    yright_mid[1] = roll(pressure_transpose[ii], ii + 1);
                    yright_most[1] = roll(pressure_transpose[ii], ii + 2);

                    yleft_most[2] = roll(vx_transpose[ii], ii - 2);
                    yleft_mid[2] = roll(vx_transpose[ii], ii - 1);
                    yright_mid[2] = roll(vx_transpose[ii], ii + 1);
                    yright_most[2] = roll(vx_transpose[ii], ii + 2);

                    yleft_most[3] = roll(vy_transpose[ii], ii - 2);
                    yleft_mid[3] = roll(vy_transpose[ii], ii - 1);
                    yright_mid[3] = roll(vy_transpose[ii], ii + 1);
                    yright_most[3] = roll(vy_transpose[ii], ii + 2);

                } else {
                    // Adjust for beginning input of L vector
                    xcoordinate = ii - 2;
                    ycoordinate = jj - 2;

                    // Coordinate X
                    xleft_most[0] = prims[0][jj][ii - 2];
                    xleft_mid[0] = prims[0][jj][ii - 1];
                    center[0] = prims[0][jj][ii];
                    xright_mid[0] = prims[0][jj][ii + 1];
                    xright_most[0] = prims[0][jj][ii + 2];

                    xleft_most[1] = prims[1][jj][ii - 2];
                    xleft_mid[1] = prims[1][jj][ii - 1];
                    center[1] = prims[1][jj][ii];
                    xright_mid[1] = prims[1][jj][ii + 1];
                    xright_most[1] = prims[1][jj][ii + 2];

                    xleft_most[2] = prims[2][jj][ii - 2];
                    xleft_mid[2] = prims[2][jj][ii - 1];
                    center[2] = prims[2][jj][ii];
                    xright_mid[2] = prims[2][jj][ii + 1];
                    xright_most[2] = prims[2][jj][ii + 2];

                    xleft_most[3] = prims[3][jj][ii - 2];
                    xleft_mid[3] = prims[3][jj][ii - 1];
                    center[3] = prims[3][jj][ii];
                    xright_mid[3] = prims[3][jj][ii + 1];
                    xright_most[3] = prims[3][jj][ii + 2];

                    // Coordinate Y
                    yleft_most[0] = prims[0][jj - 2][ii];
                    yleft_mid[0] = prims[0][jj - 1][ii];
                    yright_mid[0] = prims[0][jj + 1][ii];
                    yright_most[0] = prims[0][jj + 2][ii];

                    yleft_most[1] = prims[1][jj - 2][ii];
                    yleft_mid[1] = prims[1][jj - 1][ii];
                    yright_mid[1] = prims[1][jj + 1][ii];
                    yright_most[1] = prims[1][jj + 2][ii];

                    yleft_most[2] = prims[2][jj - 2][ii];
                    yleft_mid[2] = prims[2][jj - 1][ii];
                    yright_mid[2] = prims[2][jj + 1][ii];
                    yright_most[2] = prims[2][jj + 2][ii];

                    yleft_most[3] = prims[3][jj - 2][ii];
                    yleft_mid[3] = prims[3][jj - 1][ii];
                    yright_mid[3] = prims[3][jj + 1][ii];
                    yright_most[3] = prims[3][jj + 2][ii];

                }
                
                // Reconstructed left X primitives vector at the i+1/2 interface
                xprims_l[0] = center[0] + 0.5*minmod(theta*(center[0] - xleft_mid[0]),
                                                    0.5*(xright_mid[0] - xleft_mid[0]),
                                                    theta*(xright_mid[0] - center[0]));

                
                xprims_l[1] = center[1] + 0.5*minmod(theta*(center[1] - xleft_mid[1]),
                                                    0.5*(xright_mid[1] - xleft_mid[1]),
                                                    theta*(xright_mid[1] - center[1]));

                xprims_l[2] = center[2] + 0.5*minmod(theta*(center[2] - xleft_mid[2]),
                                                    0.5*(xright_mid[2] - xleft_mid[2]),
                                                    theta*(xright_mid[2] - center[2]));

                xprims_l[3] = center[3] + 0.5*minmod(theta*(center[3] - xleft_mid[3]),
                                                    0.5*(xright_mid[3] - xleft_mid[3]),
                                                    theta*(xright_mid[3] - center[3]));

                // Reconstructed right primitives vector in x
                xprims_r[0] = xright_mid[0] - 0.5*minmod(theta*(xright_mid[0] - center[0]),
                                                    0.5*(xright_most[0] - center[0]),
                                                    theta*(xright_most[0] - xright_mid[0]));

                xprims_r[1] = xright_mid[1] - 0.5*minmod(theta*(xright_mid[1] - center[1]),
                                                    0.5*(xright_most[1] - center[1]),
                                                    theta*(xright_most[1] - xright_mid[1]));

                xprims_r[2] = xright_mid[2] - 0.5*minmod(theta*(xright_mid[2] - center[2]),
                                                    0.5*(xright_most[2] - center[2]),
                                                    theta*(xright_most[2] - xright_mid[2]));

                xprims_r[3] = xright_mid[3] - 0.5*minmod(theta*(xright_mid[3] - center[3]),
                                                    0.5*(xright_most[3] - center[3]),
                                                    theta*(xright_most[3] - xright_mid[3]));

                
                // Reconstructed right primitives vector in y-direction at j+1/2 interfce
                yprims_l[0] = center[0] + 0.5*minmod(theta*(center[0] - yleft_mid[0]),
                                                    0.5*(yright_mid[0] - yleft_mid[0]),
                                                    theta*(yright_mid[0] - center[0]));

                yprims_l[1] = center[1] + 0.5*minmod(theta*(center[1] - yleft_mid[1]),
                                                    0.5*(yright_mid[1] - yleft_mid[1]),
                                                    theta*(yright_mid[1] - center[1]));

                yprims_l[2] = center[2] + 0.5*minmod(theta*(center[2] - yleft_mid[2]),
                                                    0.5*(yright_mid[2] - yleft_mid[2]),
                                                    theta*(yright_mid[2] - center[2]));

                yprims_l[3] = center[3] + 0.5*minmod(theta*(center[3] - yleft_mid[3]),
                                                    0.5*(yright_mid[3] - yleft_mid[3]),
                                                    theta*(yright_mid[3] - center[3]));
                

                yprims_r[0] = yright_mid[0] - 0.5*minmod(theta*(yright_mid[0] - center[0]),
                                                    0.5*(yright_most[0] - center[0]),
                                                    theta*(yright_most[0] - yright_mid[0]));

                yprims_r[1] = yright_mid[1] - 0.5*minmod(theta*(yright_mid[1] - center[1]),
                                                    0.5*(yright_most[1] - center[1]),
                                                    theta*(yright_most[1] - yright_mid[1]));

                yprims_r[2] = yright_mid[2] - 0.5*minmod(theta*(yright_mid[2] - center[2]),
                                                    0.5*(yright_most[2] - center[2]),
                                                    theta*(yright_most[2] - yright_mid[2]));

                yprims_r[3] = yright_mid[3] - 0.5*minmod(theta*(yright_mid[3] - center[3]),
                                                    0.5*(yright_most[3] - center[3]),
                                                    theta*(yright_most[3] - yright_mid[3]));

            
                
                // Calculate the left and right states using the reconstructed PLM primitives
                ux_l = calc_stateSR2D(gamma, xprims_l[0], xprims_l[1], xprims_l[2], xprims_l[3]);
                ux_r = calc_stateSR2D(gamma, xprims_r[0], xprims_r[1], xprims_r[2], xprims_r[3]);

                uy_l = calc_stateSR2D(gamma, yprims_l[0], yprims_l[1], yprims_l[2], yprims_l[3]);
                uy_r = calc_stateSR2D(gamma, yprims_r[0], yprims_r[1], yprims_r[2], yprims_r[3]);

                f_l = calc_fluxSR2D(gamma, xprims_l[0], xprims_l[1], xprims_l[2], xprims_l[3]);
                f_r = calc_fluxSR2D(gamma, xprims_r[0], xprims_r[1], xprims_r[2], xprims_r[3]);

                g_l = calc_fluxSR2D(gamma, yprims_l[0], yprims_l[1], yprims_l[2], yprims_l[3], false);
                g_r = calc_fluxSR2D(gamma, yprims_r[0], yprims_r[1], yprims_r[2], yprims_r[3], false);

                vx_l = sqrt(xprims_l[2]*xprims_l[2] + xprims_l[3]*xprims_l[3]);
                vx_r = sqrt(xprims_r[2]*xprims_r[2] + xprims_r[3]*xprims_r[3]);

                vy_l = sqrt(yprims_l[2]*yprims_l[2] + yprims_l[3]*yprims_l[3]);
                vy_r = sqrt(yprims_r[2]*yprims_r[2] + yprims_r[3]*yprims_r[3]);

                Wx_l = calc_lorentz_gamma(vx_l);
                Wx_r = calc_lorentz_gamma(vx_r);

                Wy_l = calc_lorentz_gamma(vy_l);
                Wy_r = calc_lorentz_gamma(vy_r);

                f1 = calc_hll_flux(gamma, Wx_l, Wx_r, ux_l, ux_r, f_l, f_r, "x");
                g1 = calc_hll_flux(gamma, Wy_l, Wy_r,  uy_l, uy_r, g_l, g_r, "y");
                




                // Left side primitives in x
                xprims_l[0] = xleft_mid[0] + 0.5 *minmod(theta*(xleft_mid[0] - xleft_most[0]),
                                                        0.5*(center[0] - xleft_most[0]),
                                                        theta*(center[0] - xleft_mid[0]));

                xprims_l[1] = xleft_mid[1] + 0.5 *minmod(theta*(xleft_mid[1] - xleft_most[1]),
                                                        0.5*(center[1] -xleft_most[1]),
                                                        theta*(center[1] - xleft_mid[1]));
                
                xprims_l[2] = xleft_mid[2] + 0.5 *minmod(theta*(xleft_mid[2] - xleft_most[2]),
                                                        0.5*(center[2] - xleft_most[2]),
                                                        theta*(center[2] - xleft_mid[2]));
                
                xprims_l[3] = xleft_mid[3] + 0.5 *minmod(theta*(xleft_mid[3] - xleft_most[3]),
                                                        0.5*(center[3] - xleft_most[3]),
                                                        theta*(center[3] - xleft_mid[3]));

                    
                // Right side primitives in x
                xprims_r[0] = center[0] - 0.5 *minmod(theta*(center[0] - xleft_mid[0]),
                                                    0.5*(xright_mid[0] - xleft_mid[0]),
                                                    theta*(xright_mid[0] - center[0]));

                xprims_r[1] = center[1] - 0.5 *minmod(theta*(center[1] - xleft_mid[1]),
                                                    0.5*(xright_mid[1] - xleft_mid[1]),
                                                    theta*(xright_mid[1] - center[1]));

                xprims_r[2] = center[2] - 0.5 *minmod(theta*(center[2] - xleft_mid[2]),
                                                    0.5*(xright_mid[2] - xleft_mid[2]),
                                                    theta*(xright_mid[2] - center[2]));

                xprims_r[3] = center[3] - 0.5 *minmod(theta*(center[3] - xleft_mid[3]),
                                                    0.5*(xright_mid[3] - xleft_mid[3]),
                                                    theta*(xright_mid[3] - center[3]));


                // Left side primitives in y
                yprims_l[0] = yleft_mid[0] + 0.5 *minmod(theta*(yleft_mid[0] - yleft_most[0]),
                                                        0.5*(center[0] - yleft_most[0]),
                                                        theta*(center[0] - yleft_mid[0]));

                yprims_l[1] = yleft_mid[1] + 0.5 *minmod(theta*(yleft_mid[1] - yleft_most[1]),
                                                        0.5*(center[1] -yleft_most[1]),
                                                        theta*(center[1] - yleft_mid[1]));
                
                yprims_l[2] = yleft_mid[2] + 0.5 *minmod(theta*(yleft_mid[2] - yleft_most[2]),
                                                        0.5*(center[2] - yleft_most[2]),
                                                        theta*(center[2] - yleft_mid[2]));
                
                yprims_l[3] = yleft_mid[3] + 0.5 *minmod(theta*(yleft_mid[3] - yleft_most[3]),
                                                        0.5*(center[3] - yleft_most[3]),
                                                        theta*(center[3] - yleft_mid[3]));

                    
                // Right side primitives in y
                yprims_r[0] = center[0] - 0.5 *minmod(theta*(center[0] - yleft_mid[0]),
                                                    0.5*(yright_mid[0] - yleft_mid[0]),
                                                    theta*(yright_mid[0] - center[0]));

                yprims_r[1] = center[1] - 0.5 *minmod(theta*(center[1] - yleft_mid[1]),
                                                    0.5*(yright_mid[1] - yleft_mid[1]),
                                                    theta*(yright_mid[1] - center[1]));

                yprims_r[2] = center[2] - 0.5 *minmod(theta*(center[2] - yleft_mid[2]),
                                                    0.5*(yright_mid[2] - yleft_mid[2]),
                                                    theta*(yright_mid[2] - center[2]));

                yprims_r[3] = center[3]  - 0.5 *minmod(theta*(center[3] - yleft_mid[3]),
                                                    0.5*(yright_mid[3] - yleft_mid[3]),
                                                    theta*(yright_mid[3] - center[3])); 
                
            

                // Calculate the left and right states using the reconstructed PLM primitives
                ux_l = calc_stateSR2D(gamma,xprims_l[0], xprims_l[1], xprims_l[2], xprims_l[3]);
                ux_r = calc_stateSR2D(gamma,xprims_r[0], xprims_r[1], xprims_r[2], xprims_r[3]);

                uy_l = calc_stateSR2D(gamma, yprims_l[0], yprims_l[1], yprims_l[2], yprims_l[3]);
                uy_r = calc_stateSR2D(gamma, yprims_r[0], yprims_r[1], yprims_r[2], yprims_r[3]);

                f_l = calc_fluxSR2D(gamma, xprims_l[0], xprims_l[1], xprims_l[2], xprims_l[3]);
                f_r = calc_fluxSR2D(gamma, xprims_r[0], xprims_r[1], xprims_r[2], xprims_r[3]);

                g_l = calc_fluxSR2D(gamma, yprims_l[0], yprims_l[1], yprims_l[2], yprims_l[3], false);
                g_r = calc_fluxSR2D(gamma, yprims_r[0], yprims_r[1], yprims_r[2], yprims_r[3], false);


                vx_l = sqrt(xprims_l[2]*xprims_l[2] + xprims_l[3]*xprims_l[3]);
                vx_r = sqrt(xprims_r[2]*xprims_r[2] + xprims_r[3]*xprims_r[3]);

                vy_l = sqrt(yprims_l[2]*yprims_l[2] + yprims_l[3]*yprims_l[3]);
                vy_r = sqrt(yprims_r[2]*yprims_r[2] + yprims_r[3]*yprims_r[3]);

                Wx_l = calc_lorentz_gamma(vx_l);
                Wx_r = calc_lorentz_gamma(vx_r);

                Wy_l = calc_lorentz_gamma(vy_l);
                Wy_r = calc_lorentz_gamma(vy_r);

                f2 = calc_hll_flux(gamma, Wx_l, Wx_r, ux_l, ux_r, f_l, f_r, "x");
                g2 = calc_hll_flux(gamma, Wy_l, Wy_r,  uy_l, uy_r, g_l, g_r, "y");
                

                
                

                L[0][ycoordinate][xcoordinate] = - (f1[0] - f2[0])/dx - (g1[0] - g2[0])/dy;
                L[1][ycoordinate][xcoordinate] = - (f1[1] - f2[1])/dx - (g1[1] - g2[1])/dy;
                L[2][ycoordinate][xcoordinate] = - (f1[2] - f2[2])/dx - (g1[2] - g2[2])/dy;
                L[3][ycoordinate][xcoordinate] = - (f1[3] - f2[3])/dx - (g1[3] - g2[3])/dy;
                
            }

        }

        return L;

    } else {
        //==============================================================================================
        //                                  SPHERICAL 
        //==============================================================================================
        double right_cell, left_cell, upper_cell, lower_cell, ang_avg; 
        double r_left, r_right, volAvg, pc, rhoc, vc, uc, deltaV1, deltaV2;
        double log_rLeft, log_rRight;
        double theta_right, theta_left, tcoordinate, rcoordinate;
        double upper_tsurface, lower_tsurface, right_rsurface, left_rsurface;

        double delta_logr = (log10(x1[xphysical_grid - 1]) - log10(x1[0]))/xphysical_grid;

        double dr; 

        for (int jj = j_start; jj < j_bound; jj++){
            for (int ii = i_start; ii < i_bound; ii++){
                if (periodic){
                    rcoordinate = ii;
                    tcoordinate = jj;

                    // Declare the c[i-2],c[i-1],c_i,c[i+1], c[i+2] variables

                    // X Coordinate
                    xleft_most[0] = roll(prims[0][ii], ii - 2);
                    xleft_mid[0] = roll(prims[0][ii], ii - 1);
                    center[0] = prims[0][ii][jj];
                    xright_mid[0] = roll(prims[0][ii], ii + 1);
                    xright_most[0] = roll(prims[0][ii], ii + 2);

                    xleft_most[1] = roll(prims[1][ii], ii - 2);
                    xleft_mid[1] = roll(prims[1][ii], ii - 1);
                    center[1] = prims[1][ii][jj];
                    xright_mid[1] = roll(prims[1][ii], ii + 1);
                    xright_most[1] = roll(prims[1][ii], ii + 2);

                    xleft_most[2] = roll(prims[2][ii], ii - 2);
                    xleft_mid[2] = roll(prims[2][ii], ii - 1);
                    center[2] = prims[2][ii][jj];
                    xright_mid[2] = roll(prims[2][ii], ii + 1);
                    xright_most[2] = roll(prims[2][ii], ii + 2);

                    xleft_most[3] = roll(prims[3][ii], ii - 2);
                    xleft_mid[3] = roll(prims[3][ii], ii - 1);
                    center[3] = prims[3][ii][jj];
                    xright_mid[3] = roll(prims[3][ii], ii + 1);
                    xright_most[3] = roll(prims[3][ii], ii + 2);

                    // Transpose the prims matrix to compute the Y Sweep
                    rho_transpose = transpose(prims[0]);
                    pressure_transpose = transpose(prims[1]);
                    vx_transpose = transpose(prims[2]);
                    vy_transpose = transpose(prims[3]);

                    yleft_most[0] = roll(rho_transpose[ii], ii - 2);
                    yleft_mid[0] = roll(rho_transpose[ii], ii - 1);
                    yright_mid[0] = roll(rho_transpose[ii], ii + 1);
                    yright_most[0] = roll(rho_transpose[ii], ii + 2);

                    yleft_most[1] = roll(pressure_transpose[ii], ii - 2);
                    yleft_mid[1] = roll(pressure_transpose[ii], ii - 1);
                    yright_mid[1] = roll(pressure_transpose[ii], ii + 1);
                    yright_most[1] = roll(pressure_transpose[ii], ii + 2);

                    yleft_most[2] = roll(vx_transpose[ii], ii - 2);
                    yleft_mid[2] = roll(vx_transpose[ii], ii - 1);
                    yright_mid[2] = roll(vx_transpose[ii], ii + 1);
                    yright_most[2] = roll(vx_transpose[ii], ii + 2);

                    yleft_most[3] = roll(vy_transpose[ii], ii - 2);
                    yleft_mid[3] = roll(vy_transpose[ii], ii - 1);
                    yright_mid[3] = roll(vy_transpose[ii], ii + 1);
                    yright_most[3] = roll(vy_transpose[ii], ii + 2);

                } else {
                    // Adjust for beginning input of L vector
                    rcoordinate = ii - 2;
                    tcoordinate = jj - 2;

                    // Coordinate X
                    xleft_most[0] = prims[0][jj][ii - 2];
                    xleft_mid[0] = prims[0][jj][ii - 1];
                    center[0] = prims[0][jj][ii];
                    xright_mid[0] = prims[0][jj][ii + 1];
                    xright_most[0] = prims[0][jj][ii + 2];

                    xleft_most[1] = prims[1][jj][ii - 2];
                    xleft_mid[1] = prims[1][jj][ii - 1];
                    center[1] = prims[1][jj][ii];
                    xright_mid[1] = prims[1][jj][ii + 1];
                    xright_most[1] = prims[1][jj][ii + 2];

                    xleft_most[2] = prims[2][jj][ii - 2];
                    xleft_mid[2] = prims[2][jj][ii - 1];
                    center[2] = prims[2][jj][ii];
                    xright_mid[2] = prims[2][jj][ii + 1];
                    xright_most[2] = prims[2][jj][ii + 2];

                    xleft_most[3] = prims[3][jj][ii - 2];
                    xleft_mid[3] = prims[3][jj][ii - 1];
                    center[3] = prims[3][jj][ii];
                    xright_mid[3] = prims[3][jj][ii + 1];
                    xright_most[3] = prims[3][jj][ii + 2];

                    // Coordinate Y
                    yleft_most[0] = prims[0][jj - 2][ii];
                    yleft_mid[0] = prims[0][jj - 1][ii];
                    yright_mid[0] = prims[0][jj + 1][ii];
                    yright_most[0] = prims[0][jj + 2][ii];

                    yleft_most[1] = prims[1][jj - 2][ii];
                    yleft_mid[1] = prims[1][jj - 1][ii];
                    yright_mid[1] = prims[1][jj + 1][ii];
                    yright_most[1] = prims[1][jj + 2][ii];

                    yleft_most[2] = prims[2][jj - 2][ii];
                    yleft_mid[2] = prims[2][jj - 1][ii];
                    yright_mid[2] = prims[2][jj + 1][ii];
                    yright_most[2] = prims[2][jj + 2][ii];

                    yleft_most[3] = prims[3][jj - 2][ii];
                    yleft_mid[3] = prims[3][jj - 1][ii];
                    yright_mid[3] = prims[3][jj + 1][ii];
                    yright_most[3] = prims[3][jj + 2][ii];

                }
                
                // Reconstructed left X primitives vector at the i+1/2 interface
                xprims_l[0] = center[0] + 0.5*minmod(theta*(center[0] - xleft_mid[0]),
                                                    0.5*(xright_mid[0] - xleft_mid[0]),
                                                    theta*(xright_mid[0] - center[0]));

                
                xprims_l[1] = center[1] + 0.5*minmod(theta*(center[1] - xleft_mid[1]),
                                                    0.5*(xright_mid[1] - xleft_mid[1]),
                                                    theta*(xright_mid[1] - center[1]));

                xprims_l[2] = center[2] + 0.5*minmod(theta*(center[2] - xleft_mid[2]),
                                                    0.5*(xright_mid[2] - xleft_mid[2]),
                                                    theta*(xright_mid[2] - center[2]));

                xprims_l[3] = center[3] + 0.5*minmod(theta*(center[3] - xleft_mid[3]),
                                                    0.5*(xright_mid[3] - xleft_mid[3]),
                                                    theta*(xright_mid[3] - center[3]));

                // Reconstructed right primitives vector in x
                xprims_r[0] = xright_mid[0] - 0.5*minmod(theta*(xright_mid[0] - center[0]),
                                                    0.5*(xright_most[0] - center[0]),
                                                    theta*(xright_most[0] - xright_mid[0]));

                xprims_r[1] = xright_mid[1] - 0.5*minmod(theta*(xright_mid[1] - center[1]),
                                                    0.5*(xright_most[1] - center[1]),
                                                    theta*(xright_most[1] - xright_mid[1]));

                xprims_r[2] = xright_mid[2] - 0.5*minmod(theta*(xright_mid[2] - center[2]),
                                                    0.5*(xright_most[2] - center[2]),
                                                    theta*(xright_most[2] - xright_mid[2]));

                xprims_r[3] = xright_mid[3] - 0.5*minmod(theta*(xright_mid[3] - center[3]),
                                                    0.5*(xright_most[3] - center[3]),
                                                    theta*(xright_most[3] - xright_mid[3]));

                
                // Reconstructed right primitives vector in y-direction at j+1/2 interfce
                yprims_l[0] = center[0] + 0.5*minmod(theta*(center[0] - yleft_mid[0]),
                                                    0.5*(yright_mid[0] - yleft_mid[0]),
                                                    theta*(yright_mid[0] - center[0]));

                yprims_l[1] = center[1] + 0.5*minmod(theta*(center[1] - yleft_mid[1]),
                                                    0.5*(yright_mid[1] - yleft_mid[1]),
                                                    theta*(yright_mid[1] - center[1]));

                yprims_l[2] = center[2] + 0.5*minmod(theta*(center[2] - yleft_mid[2]),
                                                    0.5*(yright_mid[2] - yleft_mid[2]),
                                                    theta*(yright_mid[2] - center[2]));

                yprims_l[3] = center[3] + 0.5*minmod(theta*(center[3] - yleft_mid[3]),
                                                    0.5*(yright_mid[3] - yleft_mid[3]),
                                                    theta*(yright_mid[3] - center[3]));
                

                yprims_r[0] = yright_mid[0] - 0.5*minmod(theta*(yright_mid[0] - center[0]),
                                                    0.5*(yright_most[0] - center[0]),
                                                    theta*(yright_most[0] - yright_mid[0]));

                yprims_r[1] = yright_mid[1] - 0.5*minmod(theta*(yright_mid[1] - center[1]),
                                                    0.5*(yright_most[1] - center[1]),
                                                    theta*(yright_most[1] - yright_mid[1]));

                yprims_r[2] = yright_mid[2] - 0.5*minmod(theta*(yright_mid[2] - center[2]),
                                                    0.5*(yright_most[2] - center[2]),
                                                    theta*(yright_most[2] - yright_mid[2]));

                yprims_r[3] = yright_mid[3] - 0.5*minmod(theta*(yright_mid[3] - center[3]),
                                                    0.5*(yright_most[3] - center[3]),
                                                    theta*(yright_most[3] - yright_mid[3]));

            
                
                
                
                
                // Calculate the left and right states using the reconstructed PLM primitives
                ux_l = calc_stateSR2D(gamma, xprims_l[0], xprims_l[1], xprims_l[2], xprims_l[3]);
                ux_r = calc_stateSR2D(gamma, xprims_r[0], xprims_r[1], xprims_r[2], xprims_r[3]);

                uy_l = calc_stateSR2D(gamma, yprims_l[0], yprims_l[1], yprims_l[2], yprims_l[3]);
                uy_r = calc_stateSR2D(gamma, yprims_r[0], yprims_r[1], yprims_r[2], yprims_r[3]);

                f_l = calc_fluxSR2D(gamma, xprims_l[0], xprims_l[1], xprims_l[2], xprims_l[3]);
                f_r = calc_fluxSR2D(gamma, xprims_r[0], xprims_r[1], xprims_r[2], xprims_r[3]);

                g_l = calc_fluxSR2D(gamma, yprims_l[0], yprims_l[1], yprims_l[2], yprims_l[3], false);
                g_r = calc_fluxSR2D(gamma, yprims_r[0], yprims_r[1], yprims_r[2], yprims_r[3], false);

                vx_l = sqrt(xprims_l[2]*xprims_l[2] + xprims_l[3]*xprims_l[3]);
                vx_r = sqrt(xprims_r[2]*xprims_r[2] + xprims_r[3]*xprims_r[3]);

                vy_l = sqrt(yprims_l[2]*yprims_l[2] + yprims_l[3]*yprims_l[3]);
                vy_r = sqrt(yprims_r[2]*yprims_r[2] + yprims_r[3]*yprims_r[3]);

                Wx_l = calc_lorentz_gamma(vx_l);
                Wx_r = calc_lorentz_gamma(vx_r);

                Wy_l = calc_lorentz_gamma(vy_l);
                Wy_r = calc_lorentz_gamma(vy_r);
                
                f1 = calc_hll_flux(gamma, Wx_l, Wx_r, ux_l, ux_r, f_l, f_r, "x");
                g1 = calc_hll_flux(gamma, Wy_l, Wy_r,  uy_l, uy_r, g_l, g_r, "y");


                /**
                if ((tcoordinate == 0) && (rcoordinate == 0)){
                    cout << ii << endl;
                    cout << jj << endl;
                    cout << "Rho  (L): " << xprims_l[0] << endl;
                    cout << "Rho  (R): " << xprims_r[0] << endl;
                    cout << "P  (L): " << xprims_l[1] << endl;
                    cout << "P  (R): " << xprims_r[1] << endl;
                    cout << "V_r  (L): " << xprims_l[2] << endl;
                    cout << "V_r  (R): " << xprims_l[2] << endl;
                    cout << "V_t  (L): " << xprims_l[3] << endl;
                    cout << "V_t  (R): " << xprims_l[3] << endl;
                    cout << " " << endl;
                    cout << "Rho  (L): " << yprims_l[0] << endl;
                    cout << "Rho  (R): " << yprims_r[0] << endl;
                    cout << "P  (L): " << yprims_l[1] << endl;
                    cout << "P  (R): " << yprims_r[1] << endl;
                    cout << "V_r  (L): " << yprims_l[2] << endl;
                    cout << "V_r  (R): " << yprims_r[2] << endl;
                    cout << "V_t  (L): " << yprims_l[3] << endl;
                    cout << "V_t  (R): " << yprims_r[3] << endl;
                    string h;
                    cin >> h;

                }
                */
                
                
                



                // Do the same thing, but for the left side interface [i - 1/2]

                // Left side primitives in x
                xprims_l[0] = xleft_mid[0] + 0.5 *minmod(theta*(xleft_mid[0] - xleft_most[0]),
                                                        0.5*(center[0] - xleft_most[0]),
                                                        theta*(center[0] - xleft_mid[0]));

                xprims_l[1] = xleft_mid[1] + 0.5 *minmod(theta*(xleft_mid[1] - xleft_most[1]),
                                                        0.5*(center[1] -xleft_most[1]),
                                                        theta*(center[1] - xleft_mid[1]));
                
                xprims_l[2] = xleft_mid[2] + 0.5 *minmod(theta*(xleft_mid[2] - xleft_most[2]),
                                                        0.5*(center[2] - xleft_most[2]),
                                                        theta*(center[2] - xleft_mid[2]));
                
                xprims_l[3] = xleft_mid[3] + 0.5 *minmod(theta*(xleft_mid[3] - xleft_most[3]),
                                                        0.5*(center[3] - xleft_most[3]),
                                                        theta*(center[3] - xleft_mid[3]));

                    
                // Right side primitives in x
                xprims_r[0] = center[0] - 0.5 *minmod(theta*(center[0] - xleft_mid[0]),
                                                    0.5*(xright_mid[0] - xleft_mid[0]),
                                                    theta*(xright_mid[0] - center[0]));

                xprims_r[1] = center[1] - 0.5 *minmod(theta*(center[1] - xleft_mid[1]),
                                                    0.5*(xright_mid[1] - xleft_mid[1]),
                                                    theta*(xright_mid[1] - center[1]));

                xprims_r[2] = center[2] - 0.5 *minmod(theta*(center[2] - xleft_mid[2]),
                                                    0.5*(xright_mid[2] - xleft_mid[2]),
                                                    theta*(xright_mid[2] - center[2]));

                xprims_r[3] = center[3] - 0.5 *minmod(theta*(center[3] - xleft_mid[3]),
                                                    0.5*(xright_mid[3] - xleft_mid[3]),
                                                    theta*(xright_mid[3] - center[3]));


                // Left side primitives in y
                yprims_l[0] = yleft_mid[0] + 0.5 *minmod(theta*(yleft_mid[0] - yleft_most[0]),
                                                        0.5*(center[0] - yleft_most[0]),
                                                        theta*(center[0] - yleft_mid[0]));

                yprims_l[1] = yleft_mid[1] + 0.5 *minmod(theta*(yleft_mid[1] - yleft_most[1]),
                                                        0.5*(center[1] -yleft_most[1]),
                                                        theta*(center[1] - yleft_mid[1]));
                
                yprims_l[2] = yleft_mid[2] + 0.5 *minmod(theta*(yleft_mid[2] - yleft_most[2]),
                                                        0.5*(center[2] - yleft_most[2]),
                                                        theta*(center[2] - yleft_mid[2]));
                
                yprims_l[3] = yleft_mid[3] + 0.5 *minmod(theta*(yleft_mid[3] - yleft_most[3]),
                                                        0.5*(center[3] - yleft_most[3]),
                                                        theta*(center[3] - yleft_mid[3]));

                    
                // Right side primitives in y
                yprims_r[0] = center[0] - 0.5 *minmod(theta*(center[0] - yleft_mid[0]),
                                                    0.5*(yright_mid[0] - yleft_mid[0]),
                                                    theta*(yright_mid[0] - center[0]));

                yprims_r[1] = center[1] - 0.5 *minmod(theta*(center[1] - yleft_mid[1]),
                                                    0.5*(yright_mid[1] - yleft_mid[1]),
                                                    theta*(yright_mid[1] - center[1]));

                yprims_r[2] = center[2] - 0.5 *minmod(theta*(center[2] - yleft_mid[2]),
                                                    0.5*(yright_mid[2] - yleft_mid[2]),
                                                    theta*(yright_mid[2] - center[2]));

                yprims_r[3] = center[3]  - 0.5 *minmod(theta*(center[3] - yleft_mid[3]),
                                                    0.5*(yright_mid[3] - yleft_mid[3]),
                                                    theta*(yright_mid[3] - center[3])); 
                
            

                // Calculate the left and right states using the reconstructed PLM primitives
                ux_l = calc_stateSR2D(gamma,xprims_l[0], xprims_l[1], xprims_l[2], xprims_l[3]);
                ux_r = calc_stateSR2D(gamma,xprims_r[0], xprims_r[1], xprims_r[2], xprims_r[3]);

                uy_l = calc_stateSR2D(gamma, yprims_l[0], yprims_l[1], yprims_l[2], yprims_l[3]);
                uy_r = calc_stateSR2D(gamma, yprims_r[0], yprims_r[1], yprims_r[2], yprims_r[3]);

                f_l = calc_fluxSR2D(gamma, xprims_l[0], xprims_l[1], xprims_l[2], xprims_l[3]);
                f_r = calc_fluxSR2D(gamma, xprims_r[0], xprims_r[1], xprims_r[2], xprims_r[3]);

                g_l = calc_fluxSR2D(gamma, yprims_l[0], yprims_l[1], yprims_l[2], yprims_l[3], false);
                g_r = calc_fluxSR2D(gamma, yprims_r[0], yprims_r[1], yprims_r[2], yprims_r[3], false);

                vx_l = sqrt(xprims_l[2]*xprims_l[2] + xprims_l[3]*xprims_l[3]);
                vx_r = sqrt(xprims_r[2]*xprims_r[2] + xprims_r[3]*xprims_r[3]);

                vy_l = sqrt(yprims_l[2]*yprims_l[2] + yprims_l[3]*yprims_l[3]);
                vy_r = sqrt(yprims_r[2]*yprims_r[2] + yprims_r[3]*yprims_r[3]);

                Wx_l = calc_lorentz_gamma(vx_l);
                Wx_r = calc_lorentz_gamma(vx_r);

                Wy_l = calc_lorentz_gamma(vy_l);
                Wy_r = calc_lorentz_gamma(vy_r);
                
                f2 = calc_hll_flux(gamma, Wx_l, Wx_r, ux_l, ux_r, f_l, f_r, "x");
                g2 = calc_hll_flux(gamma, Wy_l, Wy_r,  uy_l, uy_r, g_l, g_r, "y");

                if (linspace){
                    right_cell = x1[rcoordinate + 1];
                    left_cell = x1[rcoordinate - 1];
                    upper_cell = x2[tcoordinate + 1];
                    lower_cell = x2[tcoordinate - 1];

                    // cout << "Theta Coordinate: " << tcoordinate << endl;
                    // cout << "R Coordinate: " << rcoordinate << endl;
                    
                    // Outflow the left/right boundaries
                    if (rcoordinate - 1 < 0){
                        left_cell = x1[rcoordinate];

                    } else if (rcoordinate == xphysical_grid - 1){
                        right_cell = x1[rcoordinate];

                    }

                    if (tcoordinate - 1 < 0){

                        lower_cell = x2[tcoordinate];
                        
                    }  else if(tcoordinate == yphysical_grid - 1){
                        upper_cell = x2[tcoordinate];
                    }

                    r_right = 0.5*(right_cell + x1[rcoordinate]);
                    r_left = 0.5*(x1[rcoordinate] + left_cell);

                    theta_right = atan2( sin(upper_cell) + sin(x2[tcoordinate]) , 
                                                cos(upper_cell) + cos(x2[tcoordinate]) );

                    theta_left = atan2( sin(lower_cell) + sin(x2[tcoordinate]), 
                                                cos(lower_cell) + cos(x2[tcoordinate]) );

                } else {
                    log_rLeft = log10(x1[0]) + rcoordinate*delta_logr;
                    log_rRight = log_rLeft + delta_logr;
                    r_left = pow(10, log_rLeft);
                    r_right = pow(10, log_rRight);

                    upper_cell = x2[tcoordinate + 1];
                    lower_cell = x2[tcoordinate - 1];
                    
                    // Outflow the left/right boundaries
                    if (tcoordinate - 1 < 0){
                        cout << "Am Here: " << endl;
                        cout << "X2: " << x2[tcoordinate] << endl;
                        lower_cell = x2[tcoordinate];
                        cout << "Lower: " << lower_cell << endl;

                    } else if(tcoordinate == yphysical_grid - 1){
                        upper_cell = x2[tcoordinate];
                    }

                    theta_right = 0.5*(upper_cell + x2[tcoordinate]);
                    theta_left = 0.5*(x2[tcoordinate] + lower_cell);
                }

                dr = r_right - r_left;
                rhoc = center[0];
                pc = center[1];
                uc = center[2];
                vc = center[3];

                
                ang_avg =  atan2(sin(theta_right) + sin(theta_left), cos(theta_right) + cos(theta_left) );
                // Compute the surface areas
                right_rsurface = r_right*r_right;
                left_rsurface = r_left*r_left;
                upper_tsurface = 0.5*(r_right*r_right - r_left*r_left)*sin(theta_right);
                lower_tsurface = 0.5*(r_right*r_right - r_left*r_left)*sin(theta_left);
                volAvg = 0.75*((pow(r_right, 4) - pow(r_left, 4))/ (pow(r_right, 3) - pow(r_left, 3)) );
                deltaV1 = pow(volAvg, 2)*dr;
                deltaV2 = deltaV1*(cos(theta_left) - cos(theta_right)); 

                // cout << "Theta [i + 1/2]: " << theta_right << endl;
                // cout << "Theta[i - 1/2]: " << theta_left << endl;
                // cout << "Ang Avg: " << ang_avg << endl;
                // // string a;
                // // cin >> a;
                // cout << "Ang Avg: " << ang_avg << endl;
                // string a;
                // cin >> a;

                
                /**
                cout << "Sub: " << (cos(theta_left) - cos(theta_right)) << endl;
                cout << "VolAvg:" << volAvg << endl;
                cout << "F1: "<< f1[0] << endl;
                cout << "F2: "<< f2[0] << endl;
                cout << "G1: "<< g1[0] << endl;
                cout << "G2: "<< g2[0] << endl;
                string a;
                cin >> a;
                */

                L[0][tcoordinate][rcoordinate] = - (f1[0]*right_rsurface - f2[0]*left_rsurface)/deltaV1
                                                    - (g1[0]*upper_tsurface - g2[0]*lower_tsurface)/deltaV2;

                L[1][tcoordinate][rcoordinate] = - (f1[1]*right_rsurface - f2[1]*left_rsurface)/deltaV1
                                                    - (g1[1]*upper_tsurface - g2[1]*lower_tsurface)/deltaV2 
                                                    + rhoc*vc*vc/volAvg + 2*pc/volAvg;

                L[2][tcoordinate][rcoordinate] = - (f1[2]*right_rsurface - f2[2]*left_rsurface)/deltaV1
                                                    - (g1[2]*upper_tsurface - g2[2]*lower_tsurface)/deltaV2
                                                    -(rhoc*uc*vc/volAvg - pc*cos(ang_avg)/(volAvg*sin(ang_avg)));

                L[3][tcoordinate][rcoordinate] = - (f1[3]*right_rsurface - f2[3]*left_rsurface)/deltaV1
                                                    - (g1[3]*upper_tsurface - g2[3]*lower_tsurface)/deltaV2;
                
            }

        }

        return L;
        
    }
    

};


//===================================
//           One Source Term
//===================================
vector<vector<vector<double> > > UstateSR2D::u_dot2D(vector<vector<vector<double> > > &u_state, 
                                        vector<vector<double> > &lorentz_gamma,
                                        vector<vector<double> > &source1,
                                        bool periodic = false,
                                        float theta = 1.5, bool linspace=true)
{

    int i_start, i_bound, j_start, j_bound, xcoordinate, ycoordinate;
    int xgrid_size = u_state[0][0].size();
    int ygrid_size = u_state[0].size();
    int xphysical_grid = xgrid_size - 4;
    int yphysical_grid = ygrid_size - 4;
    string default_coordinates = "cartesian";

    double Wx_l, Wx_r, Wy_l, Wy_r, vx_l, vx_r, vy_l, vy_r;
    int n_vars = u_state.size();

    double dx = (x1[xphysical_grid - 1] - x1[0])/xphysical_grid;
    double dy = (x2[yphysical_grid - 1] - x2[0])/yphysical_grid;

    vector<vector<vector<double> > > L(n_vars, vector<vector<double> > 
                                        (yphysical_grid, vector<double> (xphysical_grid, 0)) );

    vector<double>  ux_l(n_vars), ux_r(n_vars), uy_l(n_vars), uy_r(n_vars), f_l(n_vars), f_r(n_vars); 
    vector<double>  f1(n_vars), f2(n_vars), g1(n_vars), g2(n_vars), g_l(n_vars), g_r(n_vars);
    vector<double>   xprims_l(n_vars), xprims_r(n_vars), yprims_l(n_vars), yprims_r(n_vars);

    
    
    // Calculate the primitives for the entire state
    vector<vector<vector<double> > > prims(n_vars, vector<vector<double> > 
                                            (yphysical_grid, vector<double> (xphysical_grid)));

    vector<double> xleft_most(n_vars), xleft_mid(n_vars), xright_mid(n_vars), xright_most(n_vars);
    vector<double> yleft_most(n_vars), yleft_mid(n_vars), yright_mid(n_vars), yright_most(n_vars);
    vector<double> center(n_vars);

    vector<vector<double> > rho_transpose(prims[0].size(), vector<double> (prims[0][0].size()));
    vector<vector<double> > pressure_transpose(prims[0].size(), vector<double> (prims[0][0].size()));
    vector<vector<double> > vx_transpose(prims[0].size(), vector<double> (prims[0][0].size()));
    vector<vector<double> > vy_transpose(prims[0].size(), vector<double> (prims[0][0].size()));

    prims = cons2prim2D(u_state, lorentz_gamma);
    
    // The periodic BC doesn't require ghost cells. Shift the index
    // to the beginning.
    if (periodic){ 
        i_start = 0;
        i_bound = xgrid_size;

        j_start = 0;
        j_bound = ygrid_size;
    } else {
        int true_nxpts = xgrid_size - 2;
        int true_nypts = ygrid_size - 2;
        i_start = 2;
        i_bound = true_nxpts;
        j_start = 2;
        j_bound = true_nypts;
    }

    if (coord_system == "cartesian"){
        // cout << "Am Cartesian" << endl;
        for (int jj = j_start; jj < j_bound; jj++){
            for (int ii = i_start; ii < i_bound; ii++){
                if (periodic){
                    xcoordinate = ii;
                    ycoordinate = jj;

                    // Declare the c[i-2],c[i-1],c_i,c[i+1], c[i+2] variables

                    // X Coordinate
                    xleft_most[0] = roll(prims[0][ii], ii - 2);
                    xleft_mid[0] = roll(prims[0][ii], ii - 1);
                    center[0] = prims[0][ii][jj];
                    xright_mid[0] = roll(prims[0][ii], ii + 1);
                    xright_most[0] = roll(prims[0][ii], ii + 2);

                    xleft_most[1] = roll(prims[1][ii], ii - 2);
                    xleft_mid[1] = roll(prims[1][ii], ii - 1);
                    center[1] = prims[1][ii][jj];
                    xright_mid[1] = roll(prims[1][ii], ii + 1);
                    xright_most[1] = roll(prims[1][ii], ii + 2);

                    xleft_most[2] = roll(prims[2][ii], ii - 2);
                    xleft_mid[2] = roll(prims[2][ii], ii - 1);
                    center[2] = prims[2][ii][jj];
                    xright_mid[2] = roll(prims[2][ii], ii + 1);
                    xright_most[2] = roll(prims[2][ii], ii + 2);

                    xleft_most[3] = roll(prims[3][ii], ii - 2);
                    xleft_mid[3] = roll(prims[3][ii], ii - 1);
                    center[3] = prims[3][ii][jj];
                    xright_mid[3] = roll(prims[3][ii], ii + 1);
                    xright_most[3] = roll(prims[3][ii], ii + 2);

                    // Transpose the prims matrix to compute the Y Sweep
                    rho_transpose = transpose(prims[0]);
                    pressure_transpose = transpose(prims[1]);
                    vx_transpose = transpose(prims[2]);
                    vy_transpose = transpose(prims[3]);

                    yleft_most[0] = roll(rho_transpose[ii], ii - 2);
                    yleft_mid[0] = roll(rho_transpose[ii], ii - 1);
                    yright_mid[0] = roll(rho_transpose[ii], ii + 1);
                    yright_most[0] = roll(rho_transpose[ii], ii + 2);

                    yleft_most[1] = roll(pressure_transpose[ii], ii - 2);
                    yleft_mid[1] = roll(pressure_transpose[ii], ii - 1);
                    yright_mid[1] = roll(pressure_transpose[ii], ii + 1);
                    yright_most[1] = roll(pressure_transpose[ii], ii + 2);

                    yleft_most[2] = roll(vx_transpose[ii], ii - 2);
                    yleft_mid[2] = roll(vx_transpose[ii], ii - 1);
                    yright_mid[2] = roll(vx_transpose[ii], ii + 1);
                    yright_most[2] = roll(vx_transpose[ii], ii + 2);

                    yleft_most[3] = roll(vy_transpose[ii], ii - 2);
                    yleft_mid[3] = roll(vy_transpose[ii], ii - 1);
                    yright_mid[3] = roll(vy_transpose[ii], ii + 1);
                    yright_most[3] = roll(vy_transpose[ii], ii + 2);

                } else {
                    // Adjust for beginning input of L vector
                    xcoordinate = ii - 2;
                    ycoordinate = jj - 2;

                    // Coordinate X
                    xleft_most[0] = prims[0][jj][ii - 2];
                    xleft_mid[0] = prims[0][jj][ii - 1];
                    center[0] = prims[0][jj][ii];
                    xright_mid[0] = prims[0][jj][ii + 1];
                    xright_most[0] = prims[0][jj][ii + 2];

                    xleft_most[1] = prims[1][jj][ii - 2];
                    xleft_mid[1] = prims[1][jj][ii - 1];
                    center[1] = prims[1][jj][ii];
                    xright_mid[1] = prims[1][jj][ii + 1];
                    xright_most[1] = prims[1][jj][ii + 2];

                    xleft_most[2] = prims[2][jj][ii - 2];
                    xleft_mid[2] = prims[2][jj][ii - 1];
                    center[2] = prims[2][jj][ii];
                    xright_mid[2] = prims[2][jj][ii + 1];
                    xright_most[2] = prims[2][jj][ii + 2];

                    xleft_most[3] = prims[3][jj][ii - 2];
                    xleft_mid[3] = prims[3][jj][ii - 1];
                    center[3] = prims[3][jj][ii];
                    xright_mid[3] = prims[3][jj][ii + 1];
                    xright_most[3] = prims[3][jj][ii + 2];

                    // Coordinate Y
                    yleft_most[0] = prims[0][jj - 2][ii];
                    yleft_mid[0] = prims[0][jj - 1][ii];
                    yright_mid[0] = prims[0][jj + 1][ii];
                    yright_most[0] = prims[0][jj + 2][ii];

                    yleft_most[1] = prims[1][jj - 2][ii];
                    yleft_mid[1] = prims[1][jj - 1][ii];
                    yright_mid[1] = prims[1][jj + 1][ii];
                    yright_most[1] = prims[1][jj + 2][ii];

                    yleft_most[2] = prims[2][jj - 2][ii];
                    yleft_mid[2] = prims[2][jj - 1][ii];
                    yright_mid[2] = prims[2][jj + 1][ii];
                    yright_most[2] = prims[2][jj + 2][ii];

                    yleft_most[3] = prims[3][jj - 2][ii];
                    yleft_mid[3] = prims[3][jj - 1][ii];
                    yright_mid[3] = prims[3][jj + 1][ii];
                    yright_most[3] = prims[3][jj + 2][ii];

                }
                
                // Reconstructed left X primitives vector at the i+1/2 interface
                xprims_l[0] = center[0] + 0.5*minmod(theta*(center[0] - xleft_mid[0]),
                                                    0.5*(xright_mid[0] - xleft_mid[0]),
                                                    theta*(xright_mid[0] - center[0]));

                
                xprims_l[1] = center[1] + 0.5*minmod(theta*(center[1] - xleft_mid[1]),
                                                    0.5*(xright_mid[1] - xleft_mid[1]),
                                                    theta*(xright_mid[1] - center[1]));

                xprims_l[2] = center[2] + 0.5*minmod(theta*(center[2] - xleft_mid[2]),
                                                    0.5*(xright_mid[2] - xleft_mid[2]),
                                                    theta*(xright_mid[2] - center[2]));

                xprims_l[3] = center[3] + 0.5*minmod(theta*(center[3] - xleft_mid[3]),
                                                    0.5*(xright_mid[3] - xleft_mid[3]),
                                                    theta*(xright_mid[3] - center[3]));

                // Reconstructed right primitives vector in x
                xprims_r[0] = xright_mid[0] - 0.5*minmod(theta*(xright_mid[0] - center[0]),
                                                    0.5*(xright_most[0] - center[0]),
                                                    theta*(xright_most[0] - xright_mid[0]));

                xprims_r[1] = xright_mid[1] - 0.5*minmod(theta*(xright_mid[1] - center[1]),
                                                    0.5*(xright_most[1] - center[1]),
                                                    theta*(xright_most[1] - xright_mid[1]));

                xprims_r[2] = xright_mid[2] - 0.5*minmod(theta*(xright_mid[2] - center[2]),
                                                    0.5*(xright_most[2] - center[2]),
                                                    theta*(xright_most[2] - xright_mid[2]));

                xprims_r[3] = xright_mid[3] - 0.5*minmod(theta*(xright_mid[3] - center[3]),
                                                    0.5*(xright_most[3] - center[3]),
                                                    theta*(xright_most[3] - xright_mid[3]));

                
                // Reconstructed right primitives vector in y-direction at j+1/2 interfce
                yprims_l[0] = center[0] + 0.5*minmod(theta*(center[0] - yleft_mid[0]),
                                                    0.5*(yright_mid[0] - yleft_mid[0]),
                                                    theta*(yright_mid[0] - center[0]));

                yprims_l[1] = center[1] + 0.5*minmod(theta*(center[1] - yleft_mid[1]),
                                                    0.5*(yright_mid[1] - yleft_mid[1]),
                                                    theta*(yright_mid[1] - center[1]));

                yprims_l[2] = center[2] + 0.5*minmod(theta*(center[2] - yleft_mid[2]),
                                                    0.5*(yright_mid[2] - yleft_mid[2]),
                                                    theta*(yright_mid[2] - center[2]));

                yprims_l[3] = center[3] + 0.5*minmod(theta*(center[3] - yleft_mid[3]),
                                                    0.5*(yright_mid[3] - yleft_mid[3]),
                                                    theta*(yright_mid[3] - center[3]));
                

                yprims_r[0] = yright_mid[0] - 0.5*minmod(theta*(yright_mid[0] - center[0]),
                                                    0.5*(yright_most[0] - center[0]),
                                                    theta*(yright_most[0] - yright_mid[0]));

                yprims_r[1] = yright_mid[1] - 0.5*minmod(theta*(yright_mid[1] - center[1]),
                                                    0.5*(yright_most[1] - center[1]),
                                                    theta*(yright_most[1] - yright_mid[1]));

                yprims_r[2] = yright_mid[2] - 0.5*minmod(theta*(yright_mid[2] - center[2]),
                                                    0.5*(yright_most[2] - center[2]),
                                                    theta*(yright_most[2] - yright_mid[2]));

                yprims_r[3] = yright_mid[3] - 0.5*minmod(theta*(yright_mid[3] - center[3]),
                                                    0.5*(yright_most[3] - center[3]),
                                                    theta*(yright_most[3] - yright_mid[3]));

            
                
                // Calculate the left and right states using the reconstructed PLM primitives
                ux_l = calc_stateSR2D(gamma, xprims_l[0], xprims_l[1], xprims_l[2], xprims_l[3]);
                ux_r = calc_stateSR2D(gamma, xprims_r[0], xprims_r[1], xprims_r[2], xprims_r[3]);

                uy_l = calc_stateSR2D(gamma, yprims_l[0], yprims_l[1], yprims_l[2], yprims_l[3]);
                uy_r = calc_stateSR2D(gamma, yprims_r[0], yprims_r[1], yprims_r[2], yprims_r[3]);

                f_l = calc_fluxSR2D(gamma, xprims_l[0], xprims_l[1], xprims_l[2], xprims_l[3]);
                f_r = calc_fluxSR2D(gamma, xprims_r[0], xprims_r[1], xprims_r[2], xprims_r[3]);

                g_l = calc_fluxSR2D(gamma, yprims_l[0], yprims_l[1], yprims_l[2], yprims_l[3], false);
                g_r = calc_fluxSR2D(gamma, yprims_r[0], yprims_r[1], yprims_r[2], yprims_r[3], false);

                vx_l = sqrt(xprims_l[2]*xprims_l[2] + xprims_l[3]*xprims_l[3]);
                vx_r = sqrt(xprims_r[2]*xprims_r[2] + xprims_r[3]*xprims_r[3]);

                vy_l = sqrt(yprims_l[2]*yprims_l[2] + yprims_l[3]*yprims_l[3]);
                vy_r = sqrt(yprims_r[2]*yprims_r[2] + yprims_r[3]*yprims_r[3]);

                Wx_l = calc_lorentz_gamma(vx_l);
                Wx_r = calc_lorentz_gamma(vx_r);

                Wy_l = calc_lorentz_gamma(vy_l);
                Wy_r = calc_lorentz_gamma(vy_r);

                f1 = calc_hll_flux(gamma, Wx_l, Wx_r, ux_l, ux_r, f_l, f_r, "x");
                g1 = calc_hll_flux(gamma, Wy_l, Wy_r,  uy_l, uy_r, g_l, g_r, "y");
                




                // Left side primitives in x
                xprims_l[0] = xleft_mid[0] + 0.5 *minmod(theta*(xleft_mid[0] - xleft_most[0]),
                                                        0.5*(center[0] - xleft_most[0]),
                                                        theta*(center[0] - xleft_mid[0]));

                xprims_l[1] = xleft_mid[1] + 0.5 *minmod(theta*(xleft_mid[1] - xleft_most[1]),
                                                        0.5*(center[1] -xleft_most[1]),
                                                        theta*(center[1] - xleft_mid[1]));
                
                xprims_l[2] = xleft_mid[2] + 0.5 *minmod(theta*(xleft_mid[2] - xleft_most[2]),
                                                        0.5*(center[2] - xleft_most[2]),
                                                        theta*(center[2] - xleft_mid[2]));
                
                xprims_l[3] = xleft_mid[3] + 0.5 *minmod(theta*(xleft_mid[3] - xleft_most[3]),
                                                        0.5*(center[3] - xleft_most[3]),
                                                        theta*(center[3] - xleft_mid[3]));

                    
                // Right side primitives in x
                xprims_r[0] = center[0] - 0.5 *minmod(theta*(center[0] - xleft_mid[0]),
                                                    0.5*(xright_mid[0] - xleft_mid[0]),
                                                    theta*(xright_mid[0] - center[0]));

                xprims_r[1] = center[1] - 0.5 *minmod(theta*(center[1] - xleft_mid[1]),
                                                    0.5*(xright_mid[1] - xleft_mid[1]),
                                                    theta*(xright_mid[1] - center[1]));

                xprims_r[2] = center[2] - 0.5 *minmod(theta*(center[2] - xleft_mid[2]),
                                                    0.5*(xright_mid[2] - xleft_mid[2]),
                                                    theta*(xright_mid[2] - center[2]));

                xprims_r[3] = center[3] - 0.5 *minmod(theta*(center[3] - xleft_mid[3]),
                                                    0.5*(xright_mid[3] - xleft_mid[3]),
                                                    theta*(xright_mid[3] - center[3]));


                // Left side primitives in y
                yprims_l[0] = yleft_mid[0] + 0.5 *minmod(theta*(yleft_mid[0] - yleft_most[0]),
                                                        0.5*(center[0] - yleft_most[0]),
                                                        theta*(center[0] - yleft_mid[0]));

                yprims_l[1] = yleft_mid[1] + 0.5 *minmod(theta*(yleft_mid[1] - yleft_most[1]),
                                                        0.5*(center[1] -yleft_most[1]),
                                                        theta*(center[1] - yleft_mid[1]));
                
                yprims_l[2] = yleft_mid[2] + 0.5 *minmod(theta*(yleft_mid[2] - yleft_most[2]),
                                                        0.5*(center[2] - yleft_most[2]),
                                                        theta*(center[2] - yleft_mid[2]));
                
                yprims_l[3] = yleft_mid[3] + 0.5 *minmod(theta*(yleft_mid[3] - yleft_most[3]),
                                                        0.5*(center[3] - yleft_most[3]),
                                                        theta*(center[3] - yleft_mid[3]));

                    
                // Right side primitives in y
                yprims_r[0] = center[0] - 0.5 *minmod(theta*(center[0] - yleft_mid[0]),
                                                    0.5*(yright_mid[0] - yleft_mid[0]),
                                                    theta*(yright_mid[0] - center[0]));

                yprims_r[1] = center[1] - 0.5 *minmod(theta*(center[1] - yleft_mid[1]),
                                                    0.5*(yright_mid[1] - yleft_mid[1]),
                                                    theta*(yright_mid[1] - center[1]));

                yprims_r[2] = center[2] - 0.5 *minmod(theta*(center[2] - yleft_mid[2]),
                                                    0.5*(yright_mid[2] - yleft_mid[2]),
                                                    theta*(yright_mid[2] - center[2]));

                yprims_r[3] = center[3]  - 0.5 *minmod(theta*(center[3] - yleft_mid[3]),
                                                    0.5*(yright_mid[3] - yleft_mid[3]),
                                                    theta*(yright_mid[3] - center[3])); 
                
            

                // Calculate the left and right states using the reconstructed PLM primitives
                ux_l = calc_stateSR2D(gamma,xprims_l[0], xprims_l[1], xprims_l[2], xprims_l[3]);
                ux_r = calc_stateSR2D(gamma,xprims_r[0], xprims_r[1], xprims_r[2], xprims_r[3]);

                uy_l = calc_stateSR2D(gamma, yprims_l[0], yprims_l[1], yprims_l[2], yprims_l[3]);
                uy_r = calc_stateSR2D(gamma, yprims_r[0], yprims_r[1], yprims_r[2], yprims_r[3]);

                f_l = calc_fluxSR2D(gamma, xprims_l[0], xprims_l[1], xprims_l[2], xprims_l[3]);
                f_r = calc_fluxSR2D(gamma, xprims_r[0], xprims_r[1], xprims_r[2], xprims_r[3]);

                g_l = calc_fluxSR2D(gamma, yprims_l[0], yprims_l[1], yprims_l[2], yprims_l[3], false);
                g_r = calc_fluxSR2D(gamma, yprims_r[0], yprims_r[1], yprims_r[2], yprims_r[3], false);


                vx_l = sqrt(xprims_l[2]*xprims_l[2] + xprims_l[3]*xprims_l[3]);
                vx_r = sqrt(xprims_r[2]*xprims_r[2] + xprims_r[3]*xprims_r[3]);

                vy_l = sqrt(yprims_l[2]*yprims_l[2] + yprims_l[3]*yprims_l[3]);
                vy_r = sqrt(yprims_r[2]*yprims_r[2] + yprims_r[3]*yprims_r[3]);

                Wx_l = calc_lorentz_gamma(vx_l);
                Wx_r = calc_lorentz_gamma(vx_r);

                Wy_l = calc_lorentz_gamma(vy_l);
                Wy_r = calc_lorentz_gamma(vy_r);

                f2 = calc_hll_flux(gamma, Wx_l, Wx_r, ux_l, ux_r, f_l, f_r, "x");
                g2 = calc_hll_flux(gamma, Wy_l, Wy_r,  uy_l, uy_r, g_l, g_r, "y");
                

                
                

                L[0][ycoordinate][xcoordinate] = - (f1[0] - f2[0])/dx - (g1[0] - g2[0])/dy;
                L[1][ycoordinate][xcoordinate] = - (f1[1] - f2[1])/dx - (g1[1] - g2[1])/dy;
                L[2][ycoordinate][xcoordinate] = - (f1[2] - f2[2])/dx - (g1[2] - g2[2])/dy;
                L[3][ycoordinate][xcoordinate] = - (f1[3] - f2[3])/dx - (g1[3] - g2[3])/dy;
                
            }

        }

        return L;

    } else {
        //==============================================================================================
        //                                  SPHERICAL 
        //==============================================================================================
        double right_cell, left_cell, upper_cell, lower_cell, ang_avg; 
        double r_left, r_right, volAvg, pc, rhoc, vc, uc, deltaV1, deltaV2;
        double log_rLeft, log_rRight;
        double theta_right, theta_left, tcoordinate, rcoordinate;
        double upper_tsurface, lower_tsurface, right_rsurface, left_rsurface;

        double delta_logr = (log10(x1[xphysical_grid - 1]) - log10(x1[0]))/xphysical_grid;

        double dr; 

        for (int jj = j_start; jj < j_bound; jj++){
            for (int ii = i_start; ii < i_bound; ii++){
                if (periodic){
                    rcoordinate = ii;
                    tcoordinate = jj;

                    // Declare the c[i-2],c[i-1],c_i,c[i+1], c[i+2] variables

                    // X Coordinate
                    xleft_most[0] = roll(prims[0][ii], ii - 2);
                    xleft_mid[0] = roll(prims[0][ii], ii - 1);
                    center[0] = prims[0][ii][jj];
                    xright_mid[0] = roll(prims[0][ii], ii + 1);
                    xright_most[0] = roll(prims[0][ii], ii + 2);

                    xleft_most[1] = roll(prims[1][ii], ii - 2);
                    xleft_mid[1] = roll(prims[1][ii], ii - 1);
                    center[1] = prims[1][ii][jj];
                    xright_mid[1] = roll(prims[1][ii], ii + 1);
                    xright_most[1] = roll(prims[1][ii], ii + 2);

                    xleft_most[2] = roll(prims[2][ii], ii - 2);
                    xleft_mid[2] = roll(prims[2][ii], ii - 1);
                    center[2] = prims[2][ii][jj];
                    xright_mid[2] = roll(prims[2][ii], ii + 1);
                    xright_most[2] = roll(prims[2][ii], ii + 2);

                    xleft_most[3] = roll(prims[3][ii], ii - 2);
                    xleft_mid[3] = roll(prims[3][ii], ii - 1);
                    center[3] = prims[3][ii][jj];
                    xright_mid[3] = roll(prims[3][ii], ii + 1);
                    xright_most[3] = roll(prims[3][ii], ii + 2);

                    // Transpose the prims matrix to compute the Y Sweep
                    rho_transpose = transpose(prims[0]);
                    pressure_transpose = transpose(prims[1]);
                    vx_transpose = transpose(prims[2]);
                    vy_transpose = transpose(prims[3]);

                    yleft_most[0] = roll(rho_transpose[ii], ii - 2);
                    yleft_mid[0] = roll(rho_transpose[ii], ii - 1);
                    yright_mid[0] = roll(rho_transpose[ii], ii + 1);
                    yright_most[0] = roll(rho_transpose[ii], ii + 2);

                    yleft_most[1] = roll(pressure_transpose[ii], ii - 2);
                    yleft_mid[1] = roll(pressure_transpose[ii], ii - 1);
                    yright_mid[1] = roll(pressure_transpose[ii], ii + 1);
                    yright_most[1] = roll(pressure_transpose[ii], ii + 2);

                    yleft_most[2] = roll(vx_transpose[ii], ii - 2);
                    yleft_mid[2] = roll(vx_transpose[ii], ii - 1);
                    yright_mid[2] = roll(vx_transpose[ii], ii + 1);
                    yright_most[2] = roll(vx_transpose[ii], ii + 2);

                    yleft_most[3] = roll(vy_transpose[ii], ii - 2);
                    yleft_mid[3] = roll(vy_transpose[ii], ii - 1);
                    yright_mid[3] = roll(vy_transpose[ii], ii + 1);
                    yright_most[3] = roll(vy_transpose[ii], ii + 2);

                } else {
                    // Adjust for beginning input of L vector
                    rcoordinate = ii - 2;
                    tcoordinate = jj - 2;

                    // Coordinate X
                    xleft_most[0] = prims[0][jj][ii - 2];
                    xleft_mid[0] = prims[0][jj][ii - 1];
                    center[0] = prims[0][jj][ii];
                    xright_mid[0] = prims[0][jj][ii + 1];
                    xright_most[0] = prims[0][jj][ii + 2];

                    xleft_most[1] = prims[1][jj][ii - 2];
                    xleft_mid[1] = prims[1][jj][ii - 1];
                    center[1] = prims[1][jj][ii];
                    xright_mid[1] = prims[1][jj][ii + 1];
                    xright_most[1] = prims[1][jj][ii + 2];

                    xleft_most[2] = prims[2][jj][ii - 2];
                    xleft_mid[2] = prims[2][jj][ii - 1];
                    center[2] = prims[2][jj][ii];
                    xright_mid[2] = prims[2][jj][ii + 1];
                    xright_most[2] = prims[2][jj][ii + 2];

                    xleft_most[3] = prims[3][jj][ii - 2];
                    xleft_mid[3] = prims[3][jj][ii - 1];
                    center[3] = prims[3][jj][ii];
                    xright_mid[3] = prims[3][jj][ii + 1];
                    xright_most[3] = prims[3][jj][ii + 2];

                    // Coordinate Y
                    yleft_most[0] = prims[0][jj - 2][ii];
                    yleft_mid[0] = prims[0][jj - 1][ii];
                    yright_mid[0] = prims[0][jj + 1][ii];
                    yright_most[0] = prims[0][jj + 2][ii];

                    yleft_most[1] = prims[1][jj - 2][ii];
                    yleft_mid[1] = prims[1][jj - 1][ii];
                    yright_mid[1] = prims[1][jj + 1][ii];
                    yright_most[1] = prims[1][jj + 2][ii];

                    yleft_most[2] = prims[2][jj - 2][ii];
                    yleft_mid[2] = prims[2][jj - 1][ii];
                    yright_mid[2] = prims[2][jj + 1][ii];
                    yright_most[2] = prims[2][jj + 2][ii];

                    yleft_most[3] = prims[3][jj - 2][ii];
                    yleft_mid[3] = prims[3][jj - 1][ii];
                    yright_mid[3] = prims[3][jj + 1][ii];
                    yright_most[3] = prims[3][jj + 2][ii];

                }
                
                // Reconstructed left X primitives vector at the i+1/2 interface
                xprims_l[0] = center[0] + 0.5*minmod(theta*(center[0] - xleft_mid[0]),
                                                    0.5*(xright_mid[0] - xleft_mid[0]),
                                                    theta*(xright_mid[0] - center[0]));

                
                xprims_l[1] = center[1] + 0.5*minmod(theta*(center[1] - xleft_mid[1]),
                                                    0.5*(xright_mid[1] - xleft_mid[1]),
                                                    theta*(xright_mid[1] - center[1]));

                xprims_l[2] = center[2] + 0.5*minmod(theta*(center[2] - xleft_mid[2]),
                                                    0.5*(xright_mid[2] - xleft_mid[2]),
                                                    theta*(xright_mid[2] - center[2]));

                xprims_l[3] = center[3] + 0.5*minmod(theta*(center[3] - xleft_mid[3]),
                                                    0.5*(xright_mid[3] - xleft_mid[3]),
                                                    theta*(xright_mid[3] - center[3]));

                // Reconstructed right primitives vector in x
                xprims_r[0] = xright_mid[0] - 0.5*minmod(theta*(xright_mid[0] - center[0]),
                                                    0.5*(xright_most[0] - center[0]),
                                                    theta*(xright_most[0] - xright_mid[0]));

                xprims_r[1] = xright_mid[1] - 0.5*minmod(theta*(xright_mid[1] - center[1]),
                                                    0.5*(xright_most[1] - center[1]),
                                                    theta*(xright_most[1] - xright_mid[1]));

                xprims_r[2] = xright_mid[2] - 0.5*minmod(theta*(xright_mid[2] - center[2]),
                                                    0.5*(xright_most[2] - center[2]),
                                                    theta*(xright_most[2] - xright_mid[2]));

                xprims_r[3] = xright_mid[3] - 0.5*minmod(theta*(xright_mid[3] - center[3]),
                                                    0.5*(xright_most[3] - center[3]),
                                                    theta*(xright_most[3] - xright_mid[3]));

                
                // Reconstructed right primitives vector in y-direction at j+1/2 interfce
                yprims_l[0] = center[0] + 0.5*minmod(theta*(center[0] - yleft_mid[0]),
                                                    0.5*(yright_mid[0] - yleft_mid[0]),
                                                    theta*(yright_mid[0] - center[0]));

                yprims_l[1] = center[1] + 0.5*minmod(theta*(center[1] - yleft_mid[1]),
                                                    0.5*(yright_mid[1] - yleft_mid[1]),
                                                    theta*(yright_mid[1] - center[1]));

                yprims_l[2] = center[2] + 0.5*minmod(theta*(center[2] - yleft_mid[2]),
                                                    0.5*(yright_mid[2] - yleft_mid[2]),
                                                    theta*(yright_mid[2] - center[2]));

                yprims_l[3] = center[3] + 0.5*minmod(theta*(center[3] - yleft_mid[3]),
                                                    0.5*(yright_mid[3] - yleft_mid[3]),
                                                    theta*(yright_mid[3] - center[3]));
                

                yprims_r[0] = yright_mid[0] - 0.5*minmod(theta*(yright_mid[0] - center[0]),
                                                    0.5*(yright_most[0] - center[0]),
                                                    theta*(yright_most[0] - yright_mid[0]));

                yprims_r[1] = yright_mid[1] - 0.5*minmod(theta*(yright_mid[1] - center[1]),
                                                    0.5*(yright_most[1] - center[1]),
                                                    theta*(yright_most[1] - yright_mid[1]));

                yprims_r[2] = yright_mid[2] - 0.5*minmod(theta*(yright_mid[2] - center[2]),
                                                    0.5*(yright_most[2] - center[2]),
                                                    theta*(yright_most[2] - yright_mid[2]));

                yprims_r[3] = yright_mid[3] - 0.5*minmod(theta*(yright_mid[3] - center[3]),
                                                    0.5*(yright_most[3] - center[3]),
                                                    theta*(yright_most[3] - yright_mid[3]));

            
                
                
                
                
                // Calculate the left and right states using the reconstructed PLM primitives
                ux_l = calc_stateSR2D(gamma, xprims_l[0], xprims_l[1], xprims_l[2], xprims_l[3]);
                ux_r = calc_stateSR2D(gamma, xprims_r[0], xprims_r[1], xprims_r[2], xprims_r[3]);

                uy_l = calc_stateSR2D(gamma, yprims_l[0], yprims_l[1], yprims_l[2], yprims_l[3]);
                uy_r = calc_stateSR2D(gamma, yprims_r[0], yprims_r[1], yprims_r[2], yprims_r[3]);

                f_l = calc_fluxSR2D(gamma, xprims_l[0], xprims_l[1], xprims_l[2], xprims_l[3]);
                f_r = calc_fluxSR2D(gamma, xprims_r[0], xprims_r[1], xprims_r[2], xprims_r[3]);

                g_l = calc_fluxSR2D(gamma, yprims_l[0], yprims_l[1], yprims_l[2], yprims_l[3], false);
                g_r = calc_fluxSR2D(gamma, yprims_r[0], yprims_r[1], yprims_r[2], yprims_r[3], false);

                vx_l = sqrt(xprims_l[2]*xprims_l[2] + xprims_l[3]*xprims_l[3]);
                vx_r = sqrt(xprims_r[2]*xprims_r[2] + xprims_r[3]*xprims_r[3]);

                vy_l = sqrt(yprims_l[2]*yprims_l[2] + yprims_l[3]*yprims_l[3]);
                vy_r = sqrt(yprims_r[2]*yprims_r[2] + yprims_r[3]*yprims_r[3]);

                Wx_l = calc_lorentz_gamma(vx_l);
                Wx_r = calc_lorentz_gamma(vx_r);

                Wy_l = calc_lorentz_gamma(vy_l);
                Wy_r = calc_lorentz_gamma(vy_r);
                
                f1 = calc_hll_flux(gamma, Wx_l, Wx_r, ux_l, ux_r, f_l, f_r, "x");
                g1 = calc_hll_flux(gamma, Wy_l, Wy_r,  uy_l, uy_r, g_l, g_r, "y");


                /**
                if ((tcoordinate == 0) && (rcoordinate == 0)){
                    cout << ii << endl;
                    cout << jj << endl;
                    cout << "Rho  (L): " << xprims_l[0] << endl;
                    cout << "Rho  (R): " << xprims_r[0] << endl;
                    cout << "P  (L): " << xprims_l[1] << endl;
                    cout << "P  (R): " << xprims_r[1] << endl;
                    cout << "V_r  (L): " << xprims_l[2] << endl;
                    cout << "V_r  (R): " << xprims_l[2] << endl;
                    cout << "V_t  (L): " << xprims_l[3] << endl;
                    cout << "V_t  (R): " << xprims_l[3] << endl;
                    cout << " " << endl;
                    cout << "Rho  (L): " << yprims_l[0] << endl;
                    cout << "Rho  (R): " << yprims_r[0] << endl;
                    cout << "P  (L): " << yprims_l[1] << endl;
                    cout << "P  (R): " << yprims_r[1] << endl;
                    cout << "V_r  (L): " << yprims_l[2] << endl;
                    cout << "V_r  (R): " << yprims_r[2] << endl;
                    cout << "V_t  (L): " << yprims_l[3] << endl;
                    cout << "V_t  (R): " << yprims_r[3] << endl;
                    string h;
                    cin >> h;

                }
                */
                
                
                



                // Do the same thing, but for the left side interface [i - 1/2]

                // Left side primitives in x
                xprims_l[0] = xleft_mid[0] + 0.5 *minmod(theta*(xleft_mid[0] - xleft_most[0]),
                                                        0.5*(center[0] - xleft_most[0]),
                                                        theta*(center[0] - xleft_mid[0]));

                xprims_l[1] = xleft_mid[1] + 0.5 *minmod(theta*(xleft_mid[1] - xleft_most[1]),
                                                        0.5*(center[1] -xleft_most[1]),
                                                        theta*(center[1] - xleft_mid[1]));
                
                xprims_l[2] = xleft_mid[2] + 0.5 *minmod(theta*(xleft_mid[2] - xleft_most[2]),
                                                        0.5*(center[2] - xleft_most[2]),
                                                        theta*(center[2] - xleft_mid[2]));
                
                xprims_l[3] = xleft_mid[3] + 0.5 *minmod(theta*(xleft_mid[3] - xleft_most[3]),
                                                        0.5*(center[3] - xleft_most[3]),
                                                        theta*(center[3] - xleft_mid[3]));

                    
                // Right side primitives in x
                xprims_r[0] = center[0] - 0.5 *minmod(theta*(center[0] - xleft_mid[0]),
                                                    0.5*(xright_mid[0] - xleft_mid[0]),
                                                    theta*(xright_mid[0] - center[0]));

                xprims_r[1] = center[1] - 0.5 *minmod(theta*(center[1] - xleft_mid[1]),
                                                    0.5*(xright_mid[1] - xleft_mid[1]),
                                                    theta*(xright_mid[1] - center[1]));

                xprims_r[2] = center[2] - 0.5 *minmod(theta*(center[2] - xleft_mid[2]),
                                                    0.5*(xright_mid[2] - xleft_mid[2]),
                                                    theta*(xright_mid[2] - center[2]));

                xprims_r[3] = center[3] - 0.5 *minmod(theta*(center[3] - xleft_mid[3]),
                                                    0.5*(xright_mid[3] - xleft_mid[3]),
                                                    theta*(xright_mid[3] - center[3]));


                // Left side primitives in y
                yprims_l[0] = yleft_mid[0] + 0.5 *minmod(theta*(yleft_mid[0] - yleft_most[0]),
                                                        0.5*(center[0] - yleft_most[0]),
                                                        theta*(center[0] - yleft_mid[0]));

                yprims_l[1] = yleft_mid[1] + 0.5 *minmod(theta*(yleft_mid[1] - yleft_most[1]),
                                                        0.5*(center[1] -yleft_most[1]),
                                                        theta*(center[1] - yleft_mid[1]));
                
                yprims_l[2] = yleft_mid[2] + 0.5 *minmod(theta*(yleft_mid[2] - yleft_most[2]),
                                                        0.5*(center[2] - yleft_most[2]),
                                                        theta*(center[2] - yleft_mid[2]));
                
                yprims_l[3] = yleft_mid[3] + 0.5 *minmod(theta*(yleft_mid[3] - yleft_most[3]),
                                                        0.5*(center[3] - yleft_most[3]),
                                                        theta*(center[3] - yleft_mid[3]));

                    
                // Right side primitives in y
                yprims_r[0] = center[0] - 0.5 *minmod(theta*(center[0] - yleft_mid[0]),
                                                    0.5*(yright_mid[0] - yleft_mid[0]),
                                                    theta*(yright_mid[0] - center[0]));

                yprims_r[1] = center[1] - 0.5 *minmod(theta*(center[1] - yleft_mid[1]),
                                                    0.5*(yright_mid[1] - yleft_mid[1]),
                                                    theta*(yright_mid[1] - center[1]));

                yprims_r[2] = center[2] - 0.5 *minmod(theta*(center[2] - yleft_mid[2]),
                                                    0.5*(yright_mid[2] - yleft_mid[2]),
                                                    theta*(yright_mid[2] - center[2]));

                yprims_r[3] = center[3]  - 0.5 *minmod(theta*(center[3] - yleft_mid[3]),
                                                    0.5*(yright_mid[3] - yleft_mid[3]),
                                                    theta*(yright_mid[3] - center[3])); 
                
            

                // Calculate the left and right states using the reconstructed PLM primitives
                ux_l = calc_stateSR2D(gamma,xprims_l[0], xprims_l[1], xprims_l[2], xprims_l[3]);
                ux_r = calc_stateSR2D(gamma,xprims_r[0], xprims_r[1], xprims_r[2], xprims_r[3]);

                uy_l = calc_stateSR2D(gamma, yprims_l[0], yprims_l[1], yprims_l[2], yprims_l[3]);
                uy_r = calc_stateSR2D(gamma, yprims_r[0], yprims_r[1], yprims_r[2], yprims_r[3]);

                f_l = calc_fluxSR2D(gamma, xprims_l[0], xprims_l[1], xprims_l[2], xprims_l[3]);
                f_r = calc_fluxSR2D(gamma, xprims_r[0], xprims_r[1], xprims_r[2], xprims_r[3]);

                g_l = calc_fluxSR2D(gamma, yprims_l[0], yprims_l[1], yprims_l[2], yprims_l[3], false);
                g_r = calc_fluxSR2D(gamma, yprims_r[0], yprims_r[1], yprims_r[2], yprims_r[3], false);

                vx_l = sqrt(xprims_l[2]*xprims_l[2] + xprims_l[3]*xprims_l[3]);
                vx_r = sqrt(xprims_r[2]*xprims_r[2] + xprims_r[3]*xprims_r[3]);

                vy_l = sqrt(yprims_l[2]*yprims_l[2] + yprims_l[3]*yprims_l[3]);
                vy_r = sqrt(yprims_r[2]*yprims_r[2] + yprims_r[3]*yprims_r[3]);

                Wx_l = calc_lorentz_gamma(vx_l);
                Wx_r = calc_lorentz_gamma(vx_r);

                Wy_l = calc_lorentz_gamma(vy_l);
                Wy_r = calc_lorentz_gamma(vy_r);
                
                f2 = calc_hll_flux(gamma, Wx_l, Wx_r, ux_l, ux_r, f_l, f_r, "x");
                g2 = calc_hll_flux(gamma, Wy_l, Wy_r,  uy_l, uy_r, g_l, g_r, "y");

                if (linspace){
                    right_cell = x1[rcoordinate + 1];
                    left_cell = x1[rcoordinate - 1];
                    upper_cell = x2[tcoordinate + 1];
                    lower_cell = x2[tcoordinate - 1];

                    // cout << "Theta Coordinate: " << tcoordinate << endl;
                    // cout << "R Coordinate: " << rcoordinate << endl;
                    
                    // Outflow the left/right boundaries
                    if (rcoordinate - 1 < 0){
                        left_cell = x1[rcoordinate];

                    } else if (rcoordinate == xphysical_grid - 1){
                        right_cell = x1[rcoordinate];

                    }

                    if (tcoordinate - 1 < 0){

                        lower_cell = x2[tcoordinate];
                        
                    }  else if(tcoordinate == yphysical_grid - 1){
                        upper_cell = x2[tcoordinate];
                    }

                    r_right = 0.5*(right_cell + x1[rcoordinate]);
                    r_left = 0.5*(x1[rcoordinate] + left_cell);

                    theta_right = atan2( sin(upper_cell) + sin(x2[tcoordinate]) , 
                                                cos(upper_cell) + cos(x2[tcoordinate]) );

                    theta_left = atan2( sin(lower_cell) + sin(x2[tcoordinate]), 
                                                cos(lower_cell) + cos(x2[tcoordinate]) );

                } else {
                    log_rLeft = log10(x1[0]) + rcoordinate*delta_logr;
                    log_rRight = log_rLeft + delta_logr;
                    r_left = pow(10, log_rLeft);
                    r_right = pow(10, log_rRight);

                    upper_cell = x2[tcoordinate + 1];
                    lower_cell = x2[tcoordinate - 1];
                    
                    // Outflow the left/right boundaries
                    if (tcoordinate - 1 < 0){
                        cout << "Am Here: " << endl;
                        cout << "X2: " << x2[tcoordinate] << endl;
                        lower_cell = x2[tcoordinate];
                        cout << "Lower: " << lower_cell << endl;

                    } else if(tcoordinate == yphysical_grid - 1){
                        upper_cell = x2[tcoordinate];
                    }

                    theta_right = 0.5*(upper_cell + x2[tcoordinate]);
                    theta_left = 0.5*(x2[tcoordinate] + lower_cell);
                }

                dr = r_right - r_left;
                rhoc = center[0];
                pc = center[1];
                uc = center[2];
                vc = center[3];

                
                ang_avg =  atan2(sin(theta_right) + sin(theta_left), cos(theta_right) + cos(theta_left) );
                // Compute the surface areas
                right_rsurface = r_right*r_right;
                left_rsurface = r_left*r_left;
                upper_tsurface = 0.5*(r_right*r_right - r_left*r_left)*sin(theta_right);
                lower_tsurface = 0.5*(r_right*r_right - r_left*r_left)*sin(theta_left);
                volAvg = 0.75*((pow(r_right, 4) - pow(r_left, 4))/ (pow(r_right, 3) - pow(r_left, 3)) );
                deltaV1 = pow(volAvg, 2)*dr;
                deltaV2 = deltaV1*(cos(theta_left) - cos(theta_right)); 

                // cout << "Theta [i + 1/2]: " << theta_right << endl;
                // cout << "Theta[i - 1/2]: " << theta_left << endl;
                // cout << "Ang Avg: " << ang_avg << endl;
                // // string a;
                // // cin >> a;
                // cout << "Ang Avg: " << ang_avg << endl;
                // string a;
                // cin >> a;

                
                /**
                cout << "Sub: " << (cos(theta_left) - cos(theta_right)) << endl;
                cout << "VolAvg:" << volAvg << endl;
                cout << "F1: "<< f1[0] << endl;
                cout << "F2: "<< f2[0] << endl;
                cout << "G1: "<< g1[0] << endl;
                cout << "G2: "<< g2[0] << endl;
                string a;
                cin >> a;
                */

                L[0][tcoordinate][rcoordinate] = - (f1[0]*right_rsurface - f2[0]*left_rsurface)/deltaV1
                                                    - (g1[0]*upper_tsurface - g2[0]*lower_tsurface)/deltaV2;

                L[1][tcoordinate][rcoordinate] = - (f1[1]*right_rsurface - f2[1]*left_rsurface)/deltaV1
                                                    - (g1[1]*upper_tsurface - g2[1]*lower_tsurface)/deltaV2 
                                                    + rhoc*vc*vc/volAvg + 2*pc/volAvg;

                L[2][tcoordinate][rcoordinate] = - (f1[2]*right_rsurface - f2[2]*left_rsurface)/deltaV1
                                                    - (g1[2]*upper_tsurface - g2[2]*lower_tsurface)/deltaV2
                                                    -(rhoc*uc*vc/volAvg - pc*cos(ang_avg)/(volAvg*sin(ang_avg)));

                L[3][tcoordinate][rcoordinate] = - (f1[3]*right_rsurface - f2[3]*left_rsurface)/deltaV1
                                                    - (g1[3]*upper_tsurface - g2[3]*lower_tsurface)/deltaV2;
                
            }

        }

        return L;
        
    }
    

};

//===========================
//        2 SOURCE TERMS
//===========================
vector<vector<vector<double> > > UstateSR2D::u_dot2D(vector<vector<vector<double> > > &u_state, 
                                        vector<vector<double> > &lorentz_gamma, 
                                        vector<vector<double> > &source1,
                                        vector<vector<double> > &source2,
                                        bool periodic = false,
                                        float theta = 1.5, bool linspace=true)
{

    int i_start, i_bound, j_start, j_bound, xcoordinate, ycoordinate;
    int xgrid_size = u_state[0][0].size();
    int ygrid_size = u_state[0].size();
    int xphysical_grid = xgrid_size - 4;
    int yphysical_grid = ygrid_size - 4;
    double sourceD, sourceR;
    string default_coordinates = "cartesian";

    double Wx_l, Wx_r, Wy_l, Wy_r, vx_l, vx_r, vy_l, vy_r;
    int n_vars = u_state.size();

    double dx = (x1[xphysical_grid - 1] - x1[0])/xphysical_grid;
    double dy = (x2[yphysical_grid - 1] - x2[0])/yphysical_grid;

    vector<vector<vector<double> > > L(n_vars, vector<vector<double> > 
                                        (yphysical_grid, vector<double> (xphysical_grid, 0)) );

    vector<double>  ux_l(n_vars), ux_r(n_vars), uy_l(n_vars), uy_r(n_vars), f_l(n_vars), f_r(n_vars); 
    vector<double>  f1(n_vars), f2(n_vars), g1(n_vars), g2(n_vars), g_l(n_vars), g_r(n_vars);
    vector<double>   xprims_l(n_vars), xprims_r(n_vars), yprims_l(n_vars), yprims_r(n_vars);

    
    
    // Calculate the primitives for the entire state
    vector<vector<vector<double> > > prims(n_vars, vector<vector<double> > 
                                            (yphysical_grid, vector<double> (xphysical_grid)));

    vector<double> xleft_most(n_vars), xleft_mid(n_vars), xright_mid(n_vars), xright_most(n_vars);
    vector<double> yleft_most(n_vars), yleft_mid(n_vars), yright_mid(n_vars), yright_most(n_vars);
    vector<double> center(n_vars);

    vector<vector<double> > rho_transpose(prims[0].size(), vector<double> (prims[0][0].size()));
    vector<vector<double> > pressure_transpose(prims[0].size(), vector<double> (prims[0][0].size()));
    vector<vector<double> > vx_transpose(prims[0].size(), vector<double> (prims[0][0].size()));
    vector<vector<double> > vy_transpose(prims[0].size(), vector<double> (prims[0][0].size()));

    prims = cons2prim2D(u_state, lorentz_gamma);
    
    // The periodic BC doesn't require ghost cells. Shift the index
    // to the beginning.
    if (periodic){ 
        i_start = 0;
        i_bound = xgrid_size;

        j_start = 0;
        j_bound = ygrid_size;
    } else {
        int true_nxpts = xgrid_size - 2;
        int true_nypts = ygrid_size - 2;
        i_start = 2;
        i_bound = true_nxpts;
        j_start = 2;
        j_bound = true_nypts;
    }

    
    /**
    for (int jj = j_start; jj < j_bound; jj++){
        for (int ii = i_start; ii < i_bound; ii++){
            ycoordinate = jj - 2;
            xcoordinate = ii - 2;

            // i+1/2
            ux_l[0] = u_state[0][jj][ii];
            ux_l[1] = u_state[1][jj][ii];
            ux_l[2] = u_state[2][jj][ii];
            ux_l[3] = u_state[3][jj][ii];

            ux_r[0] = u_state[0][jj][ii + 1];
            ux_r[1] = u_state[1][jj][ii + 1];
            ux_r[2] = u_state[2][jj][ii + 1];
            ux_r[3] = u_state[3][jj][ii + 1];

            // j+1/2
            uy_l[0] = u_state[0][jj][ii];
            uy_l[1] = u_state[1][jj][ii];
            uy_l[2] = u_state[2][jj][ii];
            uy_l[3] = u_state[3][jj][ii];

            uy_r[0] = u_state[0][jj + 1][ii];
            uy_r[1] = u_state[1][jj + 1][ii];
            uy_r[2] = u_state[2][jj + 1][ii];
            uy_r[3] = u_state[3][jj + 1][ii];

            xprims_l = cons2prim(gamma, ux_l);
            xprims_r = cons2prim(gamma, ux_r);
            yprims_l = cons2prim(gamma, uy_l);
            yprims_r = cons2prim(gamma, uy_r);
            
            f_l = calc_fluxSR2D(gamma, xprims_l[0], xprims_l[1], xprims_l[2], xprims_l[3]);
            f_r = calc_fluxSR2D(gamma, xprims_r[0], xprims_r[1], xprims_r[2], xprims_r[3]);

            g_l = calc_fluxSR2D(gamma, yprims_l[0], yprims_l[1], yprims_l[2], yprims_l[3], false);
            g_r = calc_fluxSR2D(gamma, yprims_r[0], yprims_r[1], yprims_r[2], yprims_r[3], false);

            // Calc HLL Flux at i+1/2 interface
            f1 = calc_hll_flux(gamma, ux_l, ux_r, f_l, f_r, 1, 2, "x");
            g1 = calc_hll_flux(gamma, uy_l, uy_r, g_l, g_r, 1, 2, "y");

            // Set up the left and right state interfaces for i-1/2

            // i-1/2
            ux_l[0] = u_state[0][jj][ii - 1];
            ux_l[1] = u_state[1][jj][ii - 1];
            ux_l[2] = u_state[2][jj][ii - 1];
            ux_l[3] = u_state[3][jj][ii - 1];

            ux_r[0] = u_state[0][jj][ii];
            ux_r[1] = u_state[1][jj][ii];
            ux_r[2] = u_state[2][jj][ii];
            ux_r[3] = u_state[3][jj][ii];

            // j-1/2
            uy_l[0] = u_state[0][jj - 1][ii];
            uy_l[1] = u_state[1][jj - 1][ii];
            uy_l[2] = u_state[2][jj - 1][ii];
            uy_l[3] = u_state[3][jj - 1][ii];

            uy_r[0] = u_state[0][jj][ii];
            uy_r[1] = u_state[1][jj][ii];
            uy_r[2] = u_state[2][jj][ii];
            uy_r[3] = u_state[3][jj][ii];

            xprims_l = cons2prim(gamma, ux_l);
            xprims_r = cons2prim(gamma, ux_r);
            yprims_l = cons2prim(gamma, uy_l);
            yprims_r = cons2prim(gamma, uy_r);

            f_l = calc_fluxSR2D(gamma, xprims_l[0], xprims_l[1], xprims_l[2], xprims_l[3]);
            f_r = calc_fluxSR2D(gamma, xprims_r[0], xprims_r[1], xprims_r[2], xprims_r[3]);

            g_l = calc_fluxSR2D(gamma, yprims_l[0], yprims_l[1], yprims_l[2], yprims_l[3], false);
            g_r = calc_fluxSR2D(gamma, yprims_r[0], yprims_r[1], yprims_r[2], yprims_r[3], false);

            // Calc HLL Flux at i+1/2 interface
            f2 = calc_hll_flux(gamma, ux_l, ux_r, f_l, f_r, 1, 2, "x");
            g2 = calc_hll_flux(gamma, uy_l, uy_r, g_l, g_r, 1, 2, "y");
            

            L[0][ycoordinate][xcoordinate] = - (f1[0] - f2[0])/dx - (g1[0] - g2[0])/dy;
            L[1][ycoordinate][xcoordinate] = - (f1[1] - f2[1])/dx - (g1[1] - g2[1])/dy;
            L[2][ycoordinate][xcoordinate] = - (f1[2] - f2[2])/dx - (g1[2] - g2[2])/dy;
            L[3][ycoordinate][xcoordinate] = - (f1[3] - f2[3])/dx - (g1[3] - g2[3])/dy;

        }
    }
    */

    // cout << coord_system << endl;
    // string a;
    // cin >> a;
    if (coord_system == "cartesian"){
        // cout << "Am Cartesian" << endl;
        for (int jj = j_start; jj < j_bound; jj++){
            for (int ii = i_start; ii < i_bound; ii++){
                if (periodic){
                    xcoordinate = ii;
                    ycoordinate = jj;

                    // Declare the c[i-2],c[i-1],c_i,c[i+1], c[i+2] variables

                    // X Coordinate
                    xleft_most[0] = roll(prims[0][ii], ii - 2);
                    xleft_mid[0] = roll(prims[0][ii], ii - 1);
                    center[0] = prims[0][ii][jj];
                    xright_mid[0] = roll(prims[0][ii], ii + 1);
                    xright_most[0] = roll(prims[0][ii], ii + 2);

                    xleft_most[1] = roll(prims[1][ii], ii - 2);
                    xleft_mid[1] = roll(prims[1][ii], ii - 1);
                    center[1] = prims[1][ii][jj];
                    xright_mid[1] = roll(prims[1][ii], ii + 1);
                    xright_most[1] = roll(prims[1][ii], ii + 2);

                    xleft_most[2] = roll(prims[2][ii], ii - 2);
                    xleft_mid[2] = roll(prims[2][ii], ii - 1);
                    center[2] = prims[2][ii][jj];
                    xright_mid[2] = roll(prims[2][ii], ii + 1);
                    xright_most[2] = roll(prims[2][ii], ii + 2);

                    xleft_most[3] = roll(prims[3][ii], ii - 2);
                    xleft_mid[3] = roll(prims[3][ii], ii - 1);
                    center[3] = prims[3][ii][jj];
                    xright_mid[3] = roll(prims[3][ii], ii + 1);
                    xright_most[3] = roll(prims[3][ii], ii + 2);

                    // Transpose the prims matrix to compute the Y Sweep
                    rho_transpose = transpose(prims[0]);
                    pressure_transpose = transpose(prims[1]);
                    vx_transpose = transpose(prims[2]);
                    vy_transpose = transpose(prims[3]);

                    yleft_most[0] = roll(rho_transpose[ii], ii - 2);
                    yleft_mid[0] = roll(rho_transpose[ii], ii - 1);
                    yright_mid[0] = roll(rho_transpose[ii], ii + 1);
                    yright_most[0] = roll(rho_transpose[ii], ii + 2);

                    yleft_most[1] = roll(pressure_transpose[ii], ii - 2);
                    yleft_mid[1] = roll(pressure_transpose[ii], ii - 1);
                    yright_mid[1] = roll(pressure_transpose[ii], ii + 1);
                    yright_most[1] = roll(pressure_transpose[ii], ii + 2);

                    yleft_most[2] = roll(vx_transpose[ii], ii - 2);
                    yleft_mid[2] = roll(vx_transpose[ii], ii - 1);
                    yright_mid[2] = roll(vx_transpose[ii], ii + 1);
                    yright_most[2] = roll(vx_transpose[ii], ii + 2);

                    yleft_most[3] = roll(vy_transpose[ii], ii - 2);
                    yleft_mid[3] = roll(vy_transpose[ii], ii - 1);
                    yright_mid[3] = roll(vy_transpose[ii], ii + 1);
                    yright_most[3] = roll(vy_transpose[ii], ii + 2);

                } else {
                    // Adjust for beginning input of L vector
                    xcoordinate = ii - 2;
                    ycoordinate = jj - 2;

                    // Coordinate X
                    xleft_most[0] = prims[0][jj][ii - 2];
                    xleft_mid[0] = prims[0][jj][ii - 1];
                    center[0] = prims[0][jj][ii];
                    xright_mid[0] = prims[0][jj][ii + 1];
                    xright_most[0] = prims[0][jj][ii + 2];

                    xleft_most[1] = prims[1][jj][ii - 2];
                    xleft_mid[1] = prims[1][jj][ii - 1];
                    center[1] = prims[1][jj][ii];
                    xright_mid[1] = prims[1][jj][ii + 1];
                    xright_most[1] = prims[1][jj][ii + 2];

                    xleft_most[2] = prims[2][jj][ii - 2];
                    xleft_mid[2] = prims[2][jj][ii - 1];
                    center[2] = prims[2][jj][ii];
                    xright_mid[2] = prims[2][jj][ii + 1];
                    xright_most[2] = prims[2][jj][ii + 2];

                    xleft_most[3] = prims[3][jj][ii - 2];
                    xleft_mid[3] = prims[3][jj][ii - 1];
                    center[3] = prims[3][jj][ii];
                    xright_mid[3] = prims[3][jj][ii + 1];
                    xright_most[3] = prims[3][jj][ii + 2];

                    // Coordinate Y
                    yleft_most[0] = prims[0][jj - 2][ii];
                    yleft_mid[0] = prims[0][jj - 1][ii];
                    yright_mid[0] = prims[0][jj + 1][ii];
                    yright_most[0] = prims[0][jj + 2][ii];

                    yleft_most[1] = prims[1][jj - 2][ii];
                    yleft_mid[1] = prims[1][jj - 1][ii];
                    yright_mid[1] = prims[1][jj + 1][ii];
                    yright_most[1] = prims[1][jj + 2][ii];

                    yleft_most[2] = prims[2][jj - 2][ii];
                    yleft_mid[2] = prims[2][jj - 1][ii];
                    yright_mid[2] = prims[2][jj + 1][ii];
                    yright_most[2] = prims[2][jj + 2][ii];

                    yleft_most[3] = prims[3][jj - 2][ii];
                    yleft_mid[3] = prims[3][jj - 1][ii];
                    yright_mid[3] = prims[3][jj + 1][ii];
                    yright_most[3] = prims[3][jj + 2][ii];

                }
                
                // Reconstructed left X primitives vector at the i+1/2 interface
                xprims_l[0] = center[0] + 0.5*minmod(theta*(center[0] - xleft_mid[0]),
                                                    0.5*(xright_mid[0] - xleft_mid[0]),
                                                    theta*(xright_mid[0] - center[0]));

                
                xprims_l[1] = center[1] + 0.5*minmod(theta*(center[1] - xleft_mid[1]),
                                                    0.5*(xright_mid[1] - xleft_mid[1]),
                                                    theta*(xright_mid[1] - center[1]));

                xprims_l[2] = center[2] + 0.5*minmod(theta*(center[2] - xleft_mid[2]),
                                                    0.5*(xright_mid[2] - xleft_mid[2]),
                                                    theta*(xright_mid[2] - center[2]));

                xprims_l[3] = center[3] + 0.5*minmod(theta*(center[3] - xleft_mid[3]),
                                                    0.5*(xright_mid[3] - xleft_mid[3]),
                                                    theta*(xright_mid[3] - center[3]));

                // Reconstructed right primitives vector in x
                xprims_r[0] = xright_mid[0] - 0.5*minmod(theta*(xright_mid[0] - center[0]),
                                                    0.5*(xright_most[0] - center[0]),
                                                    theta*(xright_most[0] - xright_mid[0]));

                xprims_r[1] = xright_mid[1] - 0.5*minmod(theta*(xright_mid[1] - center[1]),
                                                    0.5*(xright_most[1] - center[1]),
                                                    theta*(xright_most[1] - xright_mid[1]));

                xprims_r[2] = xright_mid[2] - 0.5*minmod(theta*(xright_mid[2] - center[2]),
                                                    0.5*(xright_most[2] - center[2]),
                                                    theta*(xright_most[2] - xright_mid[2]));

                xprims_r[3] = xright_mid[3] - 0.5*minmod(theta*(xright_mid[3] - center[3]),
                                                    0.5*(xright_most[3] - center[3]),
                                                    theta*(xright_most[3] - xright_mid[3]));

                
                // Reconstructed right primitives vector in y-direction at j+1/2 interfce
                yprims_l[0] = center[0] + 0.5*minmod(theta*(center[0] - yleft_mid[0]),
                                                    0.5*(yright_mid[0] - yleft_mid[0]),
                                                    theta*(yright_mid[0] - center[0]));

                yprims_l[1] = center[1] + 0.5*minmod(theta*(center[1] - yleft_mid[1]),
                                                    0.5*(yright_mid[1] - yleft_mid[1]),
                                                    theta*(yright_mid[1] - center[1]));

                yprims_l[2] = center[2] + 0.5*minmod(theta*(center[2] - yleft_mid[2]),
                                                    0.5*(yright_mid[2] - yleft_mid[2]),
                                                    theta*(yright_mid[2] - center[2]));

                yprims_l[3] = center[3] + 0.5*minmod(theta*(center[3] - yleft_mid[3]),
                                                    0.5*(yright_mid[3] - yleft_mid[3]),
                                                    theta*(yright_mid[3] - center[3]));
                

                yprims_r[0] = yright_mid[0] - 0.5*minmod(theta*(yright_mid[0] - center[0]),
                                                    0.5*(yright_most[0] - center[0]),
                                                    theta*(yright_most[0] - yright_mid[0]));

                yprims_r[1] = yright_mid[1] - 0.5*minmod(theta*(yright_mid[1] - center[1]),
                                                    0.5*(yright_most[1] - center[1]),
                                                    theta*(yright_most[1] - yright_mid[1]));

                yprims_r[2] = yright_mid[2] - 0.5*minmod(theta*(yright_mid[2] - center[2]),
                                                    0.5*(yright_most[2] - center[2]),
                                                    theta*(yright_most[2] - yright_mid[2]));

                yprims_r[3] = yright_mid[3] - 0.5*minmod(theta*(yright_mid[3] - center[3]),
                                                    0.5*(yright_most[3] - center[3]),
                                                    theta*(yright_most[3] - yright_mid[3]));

            
                
                // Calculate the left and right states using the reconstructed PLM primitives
                ux_l = calc_stateSR2D(gamma, xprims_l[0], xprims_l[1], xprims_l[2], xprims_l[3]);
                ux_r = calc_stateSR2D(gamma, xprims_r[0], xprims_r[1], xprims_r[2], xprims_r[3]);

                uy_l = calc_stateSR2D(gamma, yprims_l[0], yprims_l[1], yprims_l[2], yprims_l[3]);
                uy_r = calc_stateSR2D(gamma, yprims_r[0], yprims_r[1], yprims_r[2], yprims_r[3]);

                f_l = calc_fluxSR2D(gamma, xprims_l[0], xprims_l[1], xprims_l[2], xprims_l[3]);
                f_r = calc_fluxSR2D(gamma, xprims_r[0], xprims_r[1], xprims_r[2], xprims_r[3]);

                g_l = calc_fluxSR2D(gamma, yprims_l[0], yprims_l[1], yprims_l[2], yprims_l[3], false);
                g_r = calc_fluxSR2D(gamma, yprims_r[0], yprims_r[1], yprims_r[2], yprims_r[3], false);

                vx_l = sqrt(xprims_l[2]*xprims_l[2] + xprims_l[3]*xprims_l[3]);
                vx_r = sqrt(xprims_r[2]*xprims_r[2] + xprims_r[3]*xprims_r[3]);

                vy_l = sqrt(yprims_l[2]*yprims_l[2] + yprims_l[3]*yprims_l[3]);
                vy_r = sqrt(yprims_r[2]*yprims_r[2] + yprims_r[3]*yprims_r[3]);

                Wx_l = calc_lorentz_gamma(vx_l);
                Wx_r = calc_lorentz_gamma(vx_r);

                Wy_l = calc_lorentz_gamma(vy_l);
                Wy_r = calc_lorentz_gamma(vy_r);

                f1 = calc_hll_flux(gamma, Wx_l, Wx_r, ux_l, ux_r, f_l, f_r, "x");
                g1 = calc_hll_flux(gamma, Wy_l, Wy_r,  uy_l, uy_r, g_l, g_r, "y");
                




                // Left side primitives in x
                xprims_l[0] = xleft_mid[0] + 0.5 *minmod(theta*(xleft_mid[0] - xleft_most[0]),
                                                        0.5*(center[0] - xleft_most[0]),
                                                        theta*(center[0] - xleft_mid[0]));

                xprims_l[1] = xleft_mid[1] + 0.5 *minmod(theta*(xleft_mid[1] - xleft_most[1]),
                                                        0.5*(center[1] -xleft_most[1]),
                                                        theta*(center[1] - xleft_mid[1]));
                
                xprims_l[2] = xleft_mid[2] + 0.5 *minmod(theta*(xleft_mid[2] - xleft_most[2]),
                                                        0.5*(center[2] - xleft_most[2]),
                                                        theta*(center[2] - xleft_mid[2]));
                
                xprims_l[3] = xleft_mid[3] + 0.5 *minmod(theta*(xleft_mid[3] - xleft_most[3]),
                                                        0.5*(center[3] - xleft_most[3]),
                                                        theta*(center[3] - xleft_mid[3]));

                    
                // Right side primitives in x
                xprims_r[0] = center[0] - 0.5 *minmod(theta*(center[0] - xleft_mid[0]),
                                                    0.5*(xright_mid[0] - xleft_mid[0]),
                                                    theta*(xright_mid[0] - center[0]));

                xprims_r[1] = center[1] - 0.5 *minmod(theta*(center[1] - xleft_mid[1]),
                                                    0.5*(xright_mid[1] - xleft_mid[1]),
                                                    theta*(xright_mid[1] - center[1]));

                xprims_r[2] = center[2] - 0.5 *minmod(theta*(center[2] - xleft_mid[2]),
                                                    0.5*(xright_mid[2] - xleft_mid[2]),
                                                    theta*(xright_mid[2] - center[2]));

                xprims_r[3] = center[3] - 0.5 *minmod(theta*(center[3] - xleft_mid[3]),
                                                    0.5*(xright_mid[3] - xleft_mid[3]),
                                                    theta*(xright_mid[3] - center[3]));


                // Left side primitives in y
                yprims_l[0] = yleft_mid[0] + 0.5 *minmod(theta*(yleft_mid[0] - yleft_most[0]),
                                                        0.5*(center[0] - yleft_most[0]),
                                                        theta*(center[0] - yleft_mid[0]));

                yprims_l[1] = yleft_mid[1] + 0.5 *minmod(theta*(yleft_mid[1] - yleft_most[1]),
                                                        0.5*(center[1] -yleft_most[1]),
                                                        theta*(center[1] - yleft_mid[1]));
                
                yprims_l[2] = yleft_mid[2] + 0.5 *minmod(theta*(yleft_mid[2] - yleft_most[2]),
                                                        0.5*(center[2] - yleft_most[2]),
                                                        theta*(center[2] - yleft_mid[2]));
                
                yprims_l[3] = yleft_mid[3] + 0.5 *minmod(theta*(yleft_mid[3] - yleft_most[3]),
                                                        0.5*(center[3] - yleft_most[3]),
                                                        theta*(center[3] - yleft_mid[3]));

                    
                // Right side primitives in y
                yprims_r[0] = center[0] - 0.5 *minmod(theta*(center[0] - yleft_mid[0]),
                                                    0.5*(yright_mid[0] - yleft_mid[0]),
                                                    theta*(yright_mid[0] - center[0]));

                yprims_r[1] = center[1] - 0.5 *minmod(theta*(center[1] - yleft_mid[1]),
                                                    0.5*(yright_mid[1] - yleft_mid[1]),
                                                    theta*(yright_mid[1] - center[1]));

                yprims_r[2] = center[2] - 0.5 *minmod(theta*(center[2] - yleft_mid[2]),
                                                    0.5*(yright_mid[2] - yleft_mid[2]),
                                                    theta*(yright_mid[2] - center[2]));

                yprims_r[3] = center[3]  - 0.5 *minmod(theta*(center[3] - yleft_mid[3]),
                                                    0.5*(yright_mid[3] - yleft_mid[3]),
                                                    theta*(yright_mid[3] - center[3])); 
                
            

                // Calculate the left and right states using the reconstructed PLM primitives
                ux_l = calc_stateSR2D(gamma,xprims_l[0], xprims_l[1], xprims_l[2], xprims_l[3]);
                ux_r = calc_stateSR2D(gamma,xprims_r[0], xprims_r[1], xprims_r[2], xprims_r[3]);

                uy_l = calc_stateSR2D(gamma, yprims_l[0], yprims_l[1], yprims_l[2], yprims_l[3]);
                uy_r = calc_stateSR2D(gamma, yprims_r[0], yprims_r[1], yprims_r[2], yprims_r[3]);

                f_l = calc_fluxSR2D(gamma, xprims_l[0], xprims_l[1], xprims_l[2], xprims_l[3]);
                f_r = calc_fluxSR2D(gamma, xprims_r[0], xprims_r[1], xprims_r[2], xprims_r[3]);

                g_l = calc_fluxSR2D(gamma, yprims_l[0], yprims_l[1], yprims_l[2], yprims_l[3], false);
                g_r = calc_fluxSR2D(gamma, yprims_r[0], yprims_r[1], yprims_r[2], yprims_r[3], false);


                vx_l = sqrt(xprims_l[2]*xprims_l[2] + xprims_l[3]*xprims_l[3]);
                vx_r = sqrt(xprims_r[2]*xprims_r[2] + xprims_r[3]*xprims_r[3]);

                vy_l = sqrt(yprims_l[2]*yprims_l[2] + yprims_l[3]*yprims_l[3]);
                vy_r = sqrt(yprims_r[2]*yprims_r[2] + yprims_r[3]*yprims_r[3]);

                Wx_l = calc_lorentz_gamma(vx_l);
                Wx_r = calc_lorentz_gamma(vx_r);

                Wy_l = calc_lorentz_gamma(vy_l);
                Wy_r = calc_lorentz_gamma(vy_r);

                f2 = calc_hll_flux(gamma, Wx_l, Wx_r, ux_l, ux_r, f_l, f_r, "x");
                g2 = calc_hll_flux(gamma, Wy_l, Wy_r,  uy_l, uy_r, g_l, g_r, "y");
                

                
                

                L[0][ycoordinate][xcoordinate] = - (f1[0] - f2[0])/dx - (g1[0] - g2[0])/dy;
                L[1][ycoordinate][xcoordinate] = - (f1[1] - f2[1])/dx - (g1[1] - g2[1])/dy;
                L[2][ycoordinate][xcoordinate] = - (f1[2] - f2[2])/dx - (g1[2] - g2[2])/dy;
                L[3][ycoordinate][xcoordinate] = - (f1[3] - f2[3])/dx - (g1[3] - g2[3])/dy;
                
            }

        }

        return L;

    } else {
        //==============================================================================================
        //                                  SPHERICAL 
        //==============================================================================================
        double right_cell, left_cell, upper_cell, lower_cell, ang_avg; 
        double r_left, r_right, volAvg, pc, rhoc, vc, uc, deltaV1, deltaV2;
        double log_rLeft, log_rRight;
        double theta_right, theta_left, tcoordinate, rcoordinate;
        double upper_tsurface, lower_tsurface, right_rsurface, left_rsurface;

        double delta_logr = (log10(x1[xphysical_grid - 1]) - log10(x1[0]))/xphysical_grid;

        double dr; 

        for (int jj = j_start; jj < j_bound; jj++){
            for (int ii = i_start; ii < i_bound; ii++){
                if (periodic){
                    rcoordinate = ii;
                    tcoordinate = jj;

                    // Declare the c[i-2],c[i-1],c_i,c[i+1], c[i+2] variables

                    // X Coordinate
                    xleft_most[0] = roll(prims[0][ii], ii - 2);
                    xleft_mid[0] = roll(prims[0][ii], ii - 1);
                    center[0] = prims[0][ii][jj];
                    xright_mid[0] = roll(prims[0][ii], ii + 1);
                    xright_most[0] = roll(prims[0][ii], ii + 2);

                    xleft_most[1] = roll(prims[1][ii], ii - 2);
                    xleft_mid[1] = roll(prims[1][ii], ii - 1);
                    center[1] = prims[1][ii][jj];
                    xright_mid[1] = roll(prims[1][ii], ii + 1);
                    xright_most[1] = roll(prims[1][ii], ii + 2);

                    xleft_most[2] = roll(prims[2][ii], ii - 2);
                    xleft_mid[2] = roll(prims[2][ii], ii - 1);
                    center[2] = prims[2][ii][jj];
                    xright_mid[2] = roll(prims[2][ii], ii + 1);
                    xright_most[2] = roll(prims[2][ii], ii + 2);

                    xleft_most[3] = roll(prims[3][ii], ii - 2);
                    xleft_mid[3] = roll(prims[3][ii], ii - 1);
                    center[3] = prims[3][ii][jj];
                    xright_mid[3] = roll(prims[3][ii], ii + 1);
                    xright_most[3] = roll(prims[3][ii], ii + 2);

                    // Transpose the prims matrix to compute the Y Sweep
                    rho_transpose = transpose(prims[0]);
                    pressure_transpose = transpose(prims[1]);
                    vx_transpose = transpose(prims[2]);
                    vy_transpose = transpose(prims[3]);

                    yleft_most[0] = roll(rho_transpose[ii], ii - 2);
                    yleft_mid[0] = roll(rho_transpose[ii], ii - 1);
                    yright_mid[0] = roll(rho_transpose[ii], ii + 1);
                    yright_most[0] = roll(rho_transpose[ii], ii + 2);

                    yleft_most[1] = roll(pressure_transpose[ii], ii - 2);
                    yleft_mid[1] = roll(pressure_transpose[ii], ii - 1);
                    yright_mid[1] = roll(pressure_transpose[ii], ii + 1);
                    yright_most[1] = roll(pressure_transpose[ii], ii + 2);

                    yleft_most[2] = roll(vx_transpose[ii], ii - 2);
                    yleft_mid[2] = roll(vx_transpose[ii], ii - 1);
                    yright_mid[2] = roll(vx_transpose[ii], ii + 1);
                    yright_most[2] = roll(vx_transpose[ii], ii + 2);

                    yleft_most[3] = roll(vy_transpose[ii], ii - 2);
                    yleft_mid[3] = roll(vy_transpose[ii], ii - 1);
                    yright_mid[3] = roll(vy_transpose[ii], ii + 1);
                    yright_most[3] = roll(vy_transpose[ii], ii + 2);

                } else {
                    // Adjust for beginning input of L vector
                    rcoordinate = ii - 2;
                    tcoordinate = jj - 2;

                    // Coordinate X
                    xleft_most[0] = prims[0][jj][ii - 2];
                    xleft_mid[0] = prims[0][jj][ii - 1];
                    center[0] = prims[0][jj][ii];
                    xright_mid[0] = prims[0][jj][ii + 1];
                    xright_most[0] = prims[0][jj][ii + 2];

                    xleft_most[1] = prims[1][jj][ii - 2];
                    xleft_mid[1] = prims[1][jj][ii - 1];
                    center[1] = prims[1][jj][ii];
                    xright_mid[1] = prims[1][jj][ii + 1];
                    xright_most[1] = prims[1][jj][ii + 2];

                    xleft_most[2] = prims[2][jj][ii - 2];
                    xleft_mid[2] = prims[2][jj][ii - 1];
                    center[2] = prims[2][jj][ii];
                    xright_mid[2] = prims[2][jj][ii + 1];
                    xright_most[2] = prims[2][jj][ii + 2];

                    xleft_most[3] = prims[3][jj][ii - 2];
                    xleft_mid[3] = prims[3][jj][ii - 1];
                    center[3] = prims[3][jj][ii];
                    xright_mid[3] = prims[3][jj][ii + 1];
                    xright_most[3] = prims[3][jj][ii + 2];

                    // Coordinate Y
                    yleft_most[0] = prims[0][jj - 2][ii];
                    yleft_mid[0] = prims[0][jj - 1][ii];
                    yright_mid[0] = prims[0][jj + 1][ii];
                    yright_most[0] = prims[0][jj + 2][ii];

                    yleft_most[1] = prims[1][jj - 2][ii];
                    yleft_mid[1] = prims[1][jj - 1][ii];
                    yright_mid[1] = prims[1][jj + 1][ii];
                    yright_most[1] = prims[1][jj + 2][ii];

                    yleft_most[2] = prims[2][jj - 2][ii];
                    yleft_mid[2] = prims[2][jj - 1][ii];
                    yright_mid[2] = prims[2][jj + 1][ii];
                    yright_most[2] = prims[2][jj + 2][ii];

                    yleft_most[3] = prims[3][jj - 2][ii];
                    yleft_mid[3] = prims[3][jj - 1][ii];
                    yright_mid[3] = prims[3][jj + 1][ii];
                    yright_most[3] = prims[3][jj + 2][ii];

                }
                
                // Reconstructed left X primitives vector at the i+1/2 interface
                xprims_l[0] = center[0] + 0.5*minmod(theta*(center[0] - xleft_mid[0]),
                                                    0.5*(xright_mid[0] - xleft_mid[0]),
                                                    theta*(xright_mid[0] - center[0]));

                
                xprims_l[1] = center[1] + 0.5*minmod(theta*(center[1] - xleft_mid[1]),
                                                    0.5*(xright_mid[1] - xleft_mid[1]),
                                                    theta*(xright_mid[1] - center[1]));

                xprims_l[2] = center[2] + 0.5*minmod(theta*(center[2] - xleft_mid[2]),
                                                    0.5*(xright_mid[2] - xleft_mid[2]),
                                                    theta*(xright_mid[2] - center[2]));

                xprims_l[3] = center[3] + 0.5*minmod(theta*(center[3] - xleft_mid[3]),
                                                    0.5*(xright_mid[3] - xleft_mid[3]),
                                                    theta*(xright_mid[3] - center[3]));

                // Reconstructed right primitives vector in x
                xprims_r[0] = xright_mid[0] - 0.5*minmod(theta*(xright_mid[0] - center[0]),
                                                    0.5*(xright_most[0] - center[0]),
                                                    theta*(xright_most[0] - xright_mid[0]));

                xprims_r[1] = xright_mid[1] - 0.5*minmod(theta*(xright_mid[1] - center[1]),
                                                    0.5*(xright_most[1] - center[1]),
                                                    theta*(xright_most[1] - xright_mid[1]));

                xprims_r[2] = xright_mid[2] - 0.5*minmod(theta*(xright_mid[2] - center[2]),
                                                    0.5*(xright_most[2] - center[2]),
                                                    theta*(xright_most[2] - xright_mid[2]));

                xprims_r[3] = xright_mid[3] - 0.5*minmod(theta*(xright_mid[3] - center[3]),
                                                    0.5*(xright_most[3] - center[3]),
                                                    theta*(xright_most[3] - xright_mid[3]));

                
                // Reconstructed right primitives vector in y-direction at j+1/2 interfce
                yprims_l[0] = center[0] + 0.5*minmod(theta*(center[0] - yleft_mid[0]),
                                                    0.5*(yright_mid[0] - yleft_mid[0]),
                                                    theta*(yright_mid[0] - center[0]));

                yprims_l[1] = center[1] + 0.5*minmod(theta*(center[1] - yleft_mid[1]),
                                                    0.5*(yright_mid[1] - yleft_mid[1]),
                                                    theta*(yright_mid[1] - center[1]));

                yprims_l[2] = center[2] + 0.5*minmod(theta*(center[2] - yleft_mid[2]),
                                                    0.5*(yright_mid[2] - yleft_mid[2]),
                                                    theta*(yright_mid[2] - center[2]));

                yprims_l[3] = center[3] + 0.5*minmod(theta*(center[3] - yleft_mid[3]),
                                                    0.5*(yright_mid[3] - yleft_mid[3]),
                                                    theta*(yright_mid[3] - center[3]));
                

                yprims_r[0] = yright_mid[0] - 0.5*minmod(theta*(yright_mid[0] - center[0]),
                                                    0.5*(yright_most[0] - center[0]),
                                                    theta*(yright_most[0] - yright_mid[0]));

                yprims_r[1] = yright_mid[1] - 0.5*minmod(theta*(yright_mid[1] - center[1]),
                                                    0.5*(yright_most[1] - center[1]),
                                                    theta*(yright_most[1] - yright_mid[1]));

                yprims_r[2] = yright_mid[2] - 0.5*minmod(theta*(yright_mid[2] - center[2]),
                                                    0.5*(yright_most[2] - center[2]),
                                                    theta*(yright_most[2] - yright_mid[2]));

                yprims_r[3] = yright_mid[3] - 0.5*minmod(theta*(yright_mid[3] - center[3]),
                                                    0.5*(yright_most[3] - center[3]),
                                                    theta*(yright_most[3] - yright_mid[3]));

            
                
                
                
                
                // Calculate the left and right states using the reconstructed PLM primitives
                ux_l = calc_stateSR2D(gamma, xprims_l[0], xprims_l[1], xprims_l[2], xprims_l[3]);
                ux_r = calc_stateSR2D(gamma, xprims_r[0], xprims_r[1], xprims_r[2], xprims_r[3]);

                uy_l = calc_stateSR2D(gamma, yprims_l[0], yprims_l[1], yprims_l[2], yprims_l[3]);
                uy_r = calc_stateSR2D(gamma, yprims_r[0], yprims_r[1], yprims_r[2], yprims_r[3]);

                f_l = calc_fluxSR2D(gamma, xprims_l[0], xprims_l[1], xprims_l[2], xprims_l[3]);
                f_r = calc_fluxSR2D(gamma, xprims_r[0], xprims_r[1], xprims_r[2], xprims_r[3]);

                g_l = calc_fluxSR2D(gamma, yprims_l[0], yprims_l[1], yprims_l[2], yprims_l[3], false);
                g_r = calc_fluxSR2D(gamma, yprims_r[0], yprims_r[1], yprims_r[2], yprims_r[3], false);

                vx_l = sqrt(xprims_l[2]*xprims_l[2] + xprims_l[3]*xprims_l[3]);
                vx_r = sqrt(xprims_r[2]*xprims_r[2] + xprims_r[3]*xprims_r[3]);

                vy_l = sqrt(yprims_l[2]*yprims_l[2] + yprims_l[3]*yprims_l[3]);
                vy_r = sqrt(yprims_r[2]*yprims_r[2] + yprims_r[3]*yprims_r[3]);

                Wx_l = calc_lorentz_gamma(vx_l);
                Wx_r = calc_lorentz_gamma(vx_r);

                Wy_l = calc_lorentz_gamma(vy_l);
                Wy_r = calc_lorentz_gamma(vy_r);
                
                f1 = calc_hll_flux(gamma, Wx_l, Wx_r, ux_l, ux_r, f_l, f_r, "x");
                g1 = calc_hll_flux(gamma, Wy_l, Wy_r,  uy_l, uy_r, g_l, g_r, "y");


                /**
                if ((tcoordinate == 0) && (rcoordinate == 0)){
                    cout << ii << endl;
                    cout << jj << endl;
                    cout << "Rho  (L): " << xprims_l[0] << endl;
                    cout << "Rho  (R): " << xprims_r[0] << endl;
                    cout << "P  (L): " << xprims_l[1] << endl;
                    cout << "P  (R): " << xprims_r[1] << endl;
                    cout << "V_r  (L): " << xprims_l[2] << endl;
                    cout << "V_r  (R): " << xprims_l[2] << endl;
                    cout << "V_t  (L): " << xprims_l[3] << endl;
                    cout << "V_t  (R): " << xprims_l[3] << endl;
                    cout << " " << endl;
                    cout << "Rho  (L): " << yprims_l[0] << endl;
                    cout << "Rho  (R): " << yprims_r[0] << endl;
                    cout << "P  (L): " << yprims_l[1] << endl;
                    cout << "P  (R): " << yprims_r[1] << endl;
                    cout << "V_r  (L): " << yprims_l[2] << endl;
                    cout << "V_r  (R): " << yprims_r[2] << endl;
                    cout << "V_t  (L): " << yprims_l[3] << endl;
                    cout << "V_t  (R): " << yprims_r[3] << endl;
                    string h;
                    cin >> h;

                }
                */
                
                
                



                // Do the same thing, but for the left side interface [i - 1/2]

                // Left side primitives in x
                xprims_l[0] = xleft_mid[0] + 0.5 *minmod(theta*(xleft_mid[0] - xleft_most[0]),
                                                        0.5*(center[0] - xleft_most[0]),
                                                        theta*(center[0] - xleft_mid[0]));

                xprims_l[1] = xleft_mid[1] + 0.5 *minmod(theta*(xleft_mid[1] - xleft_most[1]),
                                                        0.5*(center[1] -xleft_most[1]),
                                                        theta*(center[1] - xleft_mid[1]));
                
                xprims_l[2] = xleft_mid[2] + 0.5 *minmod(theta*(xleft_mid[2] - xleft_most[2]),
                                                        0.5*(center[2] - xleft_most[2]),
                                                        theta*(center[2] - xleft_mid[2]));
                
                xprims_l[3] = xleft_mid[3] + 0.5 *minmod(theta*(xleft_mid[3] - xleft_most[3]),
                                                        0.5*(center[3] - xleft_most[3]),
                                                        theta*(center[3] - xleft_mid[3]));

                    
                // Right side primitives in x
                xprims_r[0] = center[0] - 0.5 *minmod(theta*(center[0] - xleft_mid[0]),
                                                    0.5*(xright_mid[0] - xleft_mid[0]),
                                                    theta*(xright_mid[0] - center[0]));

                xprims_r[1] = center[1] - 0.5 *minmod(theta*(center[1] - xleft_mid[1]),
                                                    0.5*(xright_mid[1] - xleft_mid[1]),
                                                    theta*(xright_mid[1] - center[1]));

                xprims_r[2] = center[2] - 0.5 *minmod(theta*(center[2] - xleft_mid[2]),
                                                    0.5*(xright_mid[2] - xleft_mid[2]),
                                                    theta*(xright_mid[2] - center[2]));

                xprims_r[3] = center[3] - 0.5 *minmod(theta*(center[3] - xleft_mid[3]),
                                                    0.5*(xright_mid[3] - xleft_mid[3]),
                                                    theta*(xright_mid[3] - center[3]));


                // Left side primitives in y
                yprims_l[0] = yleft_mid[0] + 0.5 *minmod(theta*(yleft_mid[0] - yleft_most[0]),
                                                        0.5*(center[0] - yleft_most[0]),
                                                        theta*(center[0] - yleft_mid[0]));

                yprims_l[1] = yleft_mid[1] + 0.5 *minmod(theta*(yleft_mid[1] - yleft_most[1]),
                                                        0.5*(center[1] -yleft_most[1]),
                                                        theta*(center[1] - yleft_mid[1]));
                
                yprims_l[2] = yleft_mid[2] + 0.5 *minmod(theta*(yleft_mid[2] - yleft_most[2]),
                                                        0.5*(center[2] - yleft_most[2]),
                                                        theta*(center[2] - yleft_mid[2]));
                
                yprims_l[3] = yleft_mid[3] + 0.5 *minmod(theta*(yleft_mid[3] - yleft_most[3]),
                                                        0.5*(center[3] - yleft_most[3]),
                                                        theta*(center[3] - yleft_mid[3]));

                    
                // Right side primitives in y
                yprims_r[0] = center[0] - 0.5 *minmod(theta*(center[0] - yleft_mid[0]),
                                                    0.5*(yright_mid[0] - yleft_mid[0]),
                                                    theta*(yright_mid[0] - center[0]));

                yprims_r[1] = center[1] - 0.5 *minmod(theta*(center[1] - yleft_mid[1]),
                                                    0.5*(yright_mid[1] - yleft_mid[1]),
                                                    theta*(yright_mid[1] - center[1]));

                yprims_r[2] = center[2] - 0.5 *minmod(theta*(center[2] - yleft_mid[2]),
                                                    0.5*(yright_mid[2] - yleft_mid[2]),
                                                    theta*(yright_mid[2] - center[2]));

                yprims_r[3] = center[3]  - 0.5 *minmod(theta*(center[3] - yleft_mid[3]),
                                                    0.5*(yright_mid[3] - yleft_mid[3]),
                                                    theta*(yright_mid[3] - center[3])); 
                
            

                // Calculate the left and right states using the reconstructed PLM primitives
                ux_l = calc_stateSR2D(gamma,xprims_l[0], xprims_l[1], xprims_l[2], xprims_l[3]);
                ux_r = calc_stateSR2D(gamma,xprims_r[0], xprims_r[1], xprims_r[2], xprims_r[3]);

                uy_l = calc_stateSR2D(gamma, yprims_l[0], yprims_l[1], yprims_l[2], yprims_l[3]);
                uy_r = calc_stateSR2D(gamma, yprims_r[0], yprims_r[1], yprims_r[2], yprims_r[3]);

                f_l = calc_fluxSR2D(gamma, xprims_l[0], xprims_l[1], xprims_l[2], xprims_l[3]);
                f_r = calc_fluxSR2D(gamma, xprims_r[0], xprims_r[1], xprims_r[2], xprims_r[3]);

                g_l = calc_fluxSR2D(gamma, yprims_l[0], yprims_l[1], yprims_l[2], yprims_l[3], false);
                g_r = calc_fluxSR2D(gamma, yprims_r[0], yprims_r[1], yprims_r[2], yprims_r[3], false);

                vx_l = sqrt(xprims_l[2]*xprims_l[2] + xprims_l[3]*xprims_l[3]);
                vx_r = sqrt(xprims_r[2]*xprims_r[2] + xprims_r[3]*xprims_r[3]);

                vy_l = sqrt(yprims_l[2]*yprims_l[2] + yprims_l[3]*yprims_l[3]);
                vy_r = sqrt(yprims_r[2]*yprims_r[2] + yprims_r[3]*yprims_r[3]);

                Wx_l = calc_lorentz_gamma(vx_l);
                Wx_r = calc_lorentz_gamma(vx_r);

                Wy_l = calc_lorentz_gamma(vy_l);
                Wy_r = calc_lorentz_gamma(vy_r);
                
                f2 = calc_hll_flux(gamma, Wx_l, Wx_r, ux_l, ux_r, f_l, f_r, "x");
                g2 = calc_hll_flux(gamma, Wy_l, Wy_r,  uy_l, uy_r, g_l, g_r, "y");

                if (linspace){
                    right_cell = x1[rcoordinate + 1];
                    left_cell = x1[rcoordinate - 1];
                    upper_cell = x2[tcoordinate + 1];
                    lower_cell = x2[tcoordinate - 1];

                    // cout << "Theta Coordinate: " << tcoordinate << endl;
                    // cout << "R Coordinate: " << rcoordinate << endl;
                    
                    // Outflow the left/right boundaries
                    if (rcoordinate - 1 < 0){
                        left_cell = x1[rcoordinate];

                    } else if (rcoordinate == xphysical_grid - 1){
                        right_cell = x1[rcoordinate];

                    }

                    if (tcoordinate - 1 < 0){

                        lower_cell = x2[tcoordinate];
                        
                    }  else if(tcoordinate == yphysical_grid - 1){
                        upper_cell = x2[tcoordinate];
                    }

                    r_right = 0.5*(right_cell + x1[rcoordinate]);
                    r_left = 0.5*(x1[rcoordinate] + left_cell);

                    theta_right = atan2( sin(upper_cell) + sin(x2[tcoordinate]) , 
                                                cos(upper_cell) + cos(x2[tcoordinate]) );

                    theta_left = atan2( sin(lower_cell) + sin(x2[tcoordinate]), 
                                                cos(lower_cell) + cos(x2[tcoordinate]) );

                } else {
                    log_rLeft = log10(x1[0]) + rcoordinate*delta_logr;
                    log_rRight = log_rLeft + delta_logr;
                    r_left = pow(10, log_rLeft);
                    r_right = pow(10, log_rRight);

                    upper_cell = x2[tcoordinate + 1];
                    lower_cell = x2[tcoordinate - 1];
                    
                    // Outflow the left/right boundaries
                    if (tcoordinate - 1 < 0){
                        cout << "Am Here: " << endl;
                        cout << "X2: " << x2[tcoordinate] << endl;
                        lower_cell = x2[tcoordinate];
                        cout << "Lower: " << lower_cell << endl;

                    } else if(tcoordinate == yphysical_grid - 1){
                        upper_cell = x2[tcoordinate];
                    }

                    theta_right = 0.5*(upper_cell + x2[tcoordinate]);
                    theta_left = 0.5*(x2[tcoordinate] + lower_cell);
                }

                dr = r_right - r_left;
                rhoc = center[0];
                pc = center[1];
                uc = center[2];
                vc = center[3];

                
                ang_avg =  atan2(sin(theta_right) + sin(theta_left), cos(theta_right) + cos(theta_left) );
                // Compute the surface areas
                right_rsurface = r_right*r_right;
                left_rsurface = r_left*r_left;
                upper_tsurface = 0.5*(r_right*r_right - r_left*r_left)*sin(theta_right);
                lower_tsurface = 0.5*(r_right*r_right - r_left*r_left)*sin(theta_left);
                volAvg = 0.75*((pow(r_right, 4) - pow(r_left, 4))/ (pow(r_right, 3) - pow(r_left, 3)) );
                deltaV1 = pow(volAvg, 2)*dr;
                deltaV2 = deltaV1*(cos(theta_left) - cos(theta_right)); 

                // cout << "Theta [i + 1/2]: " << theta_right << endl;
                // cout << "Theta[i - 1/2]: " << theta_left << endl;
                // cout << "Ang Avg: " << ang_avg << endl;
                // // string a;
                // // cin >> a;
                // cout << "Ang Avg: " << ang_avg << endl;
                // string a;
                // cin >> a;

                
                /**
                cout << "Sub: " << (cos(theta_left) - cos(theta_right)) << endl;
                cout << "VolAvg:" << volAvg << endl;
                cout << "F1: "<< f1[0] << endl;
                cout << "F2: "<< f2[0] << endl;
                cout << "G1: "<< g1[0] << endl;
                cout << "G2: "<< g2[0] << endl;
                string a;
                cin >> a;
                */

               sourceD = source1[tcoordinate][rcoordinate];
               sourceR = source2[tcoordinate][rcoordinate];

                L[0][tcoordinate][rcoordinate] = - (f1[0]*right_rsurface - f2[0]*left_rsurface)/deltaV1
                                                    - (g1[0]*upper_tsurface - g2[0]*lower_tsurface)/deltaV2 + sourceD;

                L[1][tcoordinate][rcoordinate] = - (f1[1]*right_rsurface - f2[1]*left_rsurface)/deltaV1
                                                    - (g1[1]*upper_tsurface - g2[1]*lower_tsurface)/deltaV2 
                                                    + rhoc*vc*vc/volAvg + 2*pc/volAvg + sourceR;

                L[2][tcoordinate][rcoordinate] = - (f1[2]*right_rsurface - f2[2]*left_rsurface)/deltaV1
                                                    - (g1[2]*upper_tsurface - g2[2]*lower_tsurface)/deltaV2
                                                    -(rhoc*uc*vc/volAvg - pc*cos(ang_avg)/(volAvg*sin(ang_avg))) + sourceR;

                L[3][tcoordinate][rcoordinate] = - (f1[3]*right_rsurface - f2[3]*left_rsurface)/deltaV1
                                                    - (g1[3]*upper_tsurface - g2[3]*lower_tsurface)/deltaV2 + sourceR;
                
            }

        }

        return L;
        
    }
    

};

//===========================
//        3 SOURCE TERMS
//===========================
vector<vector<vector<double> > > UstateSR2D::u_dot2D(vector<vector<vector<double> > > &u_state, 
                                        vector<vector<double> > &lorentz_gamma, 
                                        vector<vector<double> > &source1,
                                        vector<vector<double> > &source2,
                                        vector<vector<double> > &source3,
                                        bool periodic = false,
                                        float theta = 1.5, bool linspace=true)
{

    int i_start, i_bound, j_start, j_bound, xcoordinate, ycoordinate;
    int xgrid_size = u_state[0][0].size();
    int ygrid_size = u_state[0].size();
    int xphysical_grid = xgrid_size - 4;
    int yphysical_grid = ygrid_size - 4;
    double sourceD, sourceR, source0;
    string default_coordinates = "cartesian";

    double Wx_l, Wx_r, Wy_l, Wy_r, vx_l, vx_r, vy_l, vy_r;
    int n_vars = u_state.size();

    double dx = (x1[xphysical_grid - 1] - x1[0])/xphysical_grid;
    double dy = (x2[yphysical_grid - 1] - x2[0])/yphysical_grid;

    vector<vector<vector<double> > > L(n_vars, vector<vector<double> > 
                                        (yphysical_grid, vector<double> (xphysical_grid, 0)) );

    vector<double>  ux_l(n_vars), ux_r(n_vars), uy_l(n_vars), uy_r(n_vars), f_l(n_vars), f_r(n_vars); 
    vector<double>  f1(n_vars), f2(n_vars), g1(n_vars), g2(n_vars), g_l(n_vars), g_r(n_vars);
    vector<double>   xprims_l(n_vars), xprims_r(n_vars), yprims_l(n_vars), yprims_r(n_vars);

    
    
    // Calculate the primitives for the entire state
    vector<vector<vector<double> > > prims(n_vars, vector<vector<double> > 
                                            (yphysical_grid, vector<double> (xphysical_grid)));

    vector<double> xleft_most(n_vars), xleft_mid(n_vars), xright_mid(n_vars), xright_most(n_vars);
    vector<double> yleft_most(n_vars), yleft_mid(n_vars), yright_mid(n_vars), yright_most(n_vars);
    vector<double> center(n_vars);

    vector<vector<double> > rho_transpose(prims[0].size(), vector<double> (prims[0][0].size()));
    vector<vector<double> > pressure_transpose(prims[0].size(), vector<double> (prims[0][0].size()));
    vector<vector<double> > vx_transpose(prims[0].size(), vector<double> (prims[0][0].size()));
    vector<vector<double> > vy_transpose(prims[0].size(), vector<double> (prims[0][0].size()));

    prims = cons2prim2D(u_state, lorentz_gamma);
    
    // The periodic BC doesn't require ghost cells. Shift the index
    // to the beginning.
    if (periodic){ 
        i_start = 0;
        i_bound = xgrid_size;

        j_start = 0;
        j_bound = ygrid_size;
    } else {
        int true_nxpts = xgrid_size - 2;
        int true_nypts = ygrid_size - 2;
        i_start = 2;
        i_bound = true_nxpts;
        j_start = 2;
        j_bound = true_nypts;
    }

    
    /**
    for (int jj = j_start; jj < j_bound; jj++){
        for (int ii = i_start; ii < i_bound; ii++){
            ycoordinate = jj - 2;
            xcoordinate = ii - 2;

            // i+1/2
            ux_l[0] = u_state[0][jj][ii];
            ux_l[1] = u_state[1][jj][ii];
            ux_l[2] = u_state[2][jj][ii];
            ux_l[3] = u_state[3][jj][ii];

            ux_r[0] = u_state[0][jj][ii + 1];
            ux_r[1] = u_state[1][jj][ii + 1];
            ux_r[2] = u_state[2][jj][ii + 1];
            ux_r[3] = u_state[3][jj][ii + 1];

            // j+1/2
            uy_l[0] = u_state[0][jj][ii];
            uy_l[1] = u_state[1][jj][ii];
            uy_l[2] = u_state[2][jj][ii];
            uy_l[3] = u_state[3][jj][ii];

            uy_r[0] = u_state[0][jj + 1][ii];
            uy_r[1] = u_state[1][jj + 1][ii];
            uy_r[2] = u_state[2][jj + 1][ii];
            uy_r[3] = u_state[3][jj + 1][ii];

            xprims_l = cons2prim(gamma, ux_l);
            xprims_r = cons2prim(gamma, ux_r);
            yprims_l = cons2prim(gamma, uy_l);
            yprims_r = cons2prim(gamma, uy_r);
            
            f_l = calc_fluxSR2D(gamma, xprims_l[0], xprims_l[1], xprims_l[2], xprims_l[3]);
            f_r = calc_fluxSR2D(gamma, xprims_r[0], xprims_r[1], xprims_r[2], xprims_r[3]);

            g_l = calc_fluxSR2D(gamma, yprims_l[0], yprims_l[1], yprims_l[2], yprims_l[3], false);
            g_r = calc_fluxSR2D(gamma, yprims_r[0], yprims_r[1], yprims_r[2], yprims_r[3], false);

            // Calc HLL Flux at i+1/2 interface
            f1 = calc_hll_flux(gamma, ux_l, ux_r, f_l, f_r, 1, 2, "x");
            g1 = calc_hll_flux(gamma, uy_l, uy_r, g_l, g_r, 1, 2, "y");

            // Set up the left and right state interfaces for i-1/2

            // i-1/2
            ux_l[0] = u_state[0][jj][ii - 1];
            ux_l[1] = u_state[1][jj][ii - 1];
            ux_l[2] = u_state[2][jj][ii - 1];
            ux_l[3] = u_state[3][jj][ii - 1];

            ux_r[0] = u_state[0][jj][ii];
            ux_r[1] = u_state[1][jj][ii];
            ux_r[2] = u_state[2][jj][ii];
            ux_r[3] = u_state[3][jj][ii];

            // j-1/2
            uy_l[0] = u_state[0][jj - 1][ii];
            uy_l[1] = u_state[1][jj - 1][ii];
            uy_l[2] = u_state[2][jj - 1][ii];
            uy_l[3] = u_state[3][jj - 1][ii];

            uy_r[0] = u_state[0][jj][ii];
            uy_r[1] = u_state[1][jj][ii];
            uy_r[2] = u_state[2][jj][ii];
            uy_r[3] = u_state[3][jj][ii];

            xprims_l = cons2prim(gamma, ux_l);
            xprims_r = cons2prim(gamma, ux_r);
            yprims_l = cons2prim(gamma, uy_l);
            yprims_r = cons2prim(gamma, uy_r);

            f_l = calc_fluxSR2D(gamma, xprims_l[0], xprims_l[1], xprims_l[2], xprims_l[3]);
            f_r = calc_fluxSR2D(gamma, xprims_r[0], xprims_r[1], xprims_r[2], xprims_r[3]);

            g_l = calc_fluxSR2D(gamma, yprims_l[0], yprims_l[1], yprims_l[2], yprims_l[3], false);
            g_r = calc_fluxSR2D(gamma, yprims_r[0], yprims_r[1], yprims_r[2], yprims_r[3], false);

            // Calc HLL Flux at i+1/2 interface
            f2 = calc_hll_flux(gamma, ux_l, ux_r, f_l, f_r, 1, 2, "x");
            g2 = calc_hll_flux(gamma, uy_l, uy_r, g_l, g_r, 1, 2, "y");
            

            L[0][ycoordinate][xcoordinate] = - (f1[0] - f2[0])/dx - (g1[0] - g2[0])/dy;
            L[1][ycoordinate][xcoordinate] = - (f1[1] - f2[1])/dx - (g1[1] - g2[1])/dy;
            L[2][ycoordinate][xcoordinate] = - (f1[2] - f2[2])/dx - (g1[2] - g2[2])/dy;
            L[3][ycoordinate][xcoordinate] = - (f1[3] - f2[3])/dx - (g1[3] - g2[3])/dy;

        }
    }
    */

    // cout << coord_system << endl;
    // string a;
    // cin >> a;
    if (coord_system == "cartesian"){
        // cout << "Am Cartesian" << endl;
        for (int jj = j_start; jj < j_bound; jj++){
            for (int ii = i_start; ii < i_bound; ii++){
                if (periodic){
                    xcoordinate = ii;
                    ycoordinate = jj;

                    // Declare the c[i-2],c[i-1],c_i,c[i+1], c[i+2] variables

                    // X Coordinate
                    xleft_most[0] = roll(prims[0][ii], ii - 2);
                    xleft_mid[0] = roll(prims[0][ii], ii - 1);
                    center[0] = prims[0][ii][jj];
                    xright_mid[0] = roll(prims[0][ii], ii + 1);
                    xright_most[0] = roll(prims[0][ii], ii + 2);

                    xleft_most[1] = roll(prims[1][ii], ii - 2);
                    xleft_mid[1] = roll(prims[1][ii], ii - 1);
                    center[1] = prims[1][ii][jj];
                    xright_mid[1] = roll(prims[1][ii], ii + 1);
                    xright_most[1] = roll(prims[1][ii], ii + 2);

                    xleft_most[2] = roll(prims[2][ii], ii - 2);
                    xleft_mid[2] = roll(prims[2][ii], ii - 1);
                    center[2] = prims[2][ii][jj];
                    xright_mid[2] = roll(prims[2][ii], ii + 1);
                    xright_most[2] = roll(prims[2][ii], ii + 2);

                    xleft_most[3] = roll(prims[3][ii], ii - 2);
                    xleft_mid[3] = roll(prims[3][ii], ii - 1);
                    center[3] = prims[3][ii][jj];
                    xright_mid[3] = roll(prims[3][ii], ii + 1);
                    xright_most[3] = roll(prims[3][ii], ii + 2);

                    // Transpose the prims matrix to compute the Y Sweep
                    rho_transpose = transpose(prims[0]);
                    pressure_transpose = transpose(prims[1]);
                    vx_transpose = transpose(prims[2]);
                    vy_transpose = transpose(prims[3]);

                    yleft_most[0] = roll(rho_transpose[ii], ii - 2);
                    yleft_mid[0] = roll(rho_transpose[ii], ii - 1);
                    yright_mid[0] = roll(rho_transpose[ii], ii + 1);
                    yright_most[0] = roll(rho_transpose[ii], ii + 2);

                    yleft_most[1] = roll(pressure_transpose[ii], ii - 2);
                    yleft_mid[1] = roll(pressure_transpose[ii], ii - 1);
                    yright_mid[1] = roll(pressure_transpose[ii], ii + 1);
                    yright_most[1] = roll(pressure_transpose[ii], ii + 2);

                    yleft_most[2] = roll(vx_transpose[ii], ii - 2);
                    yleft_mid[2] = roll(vx_transpose[ii], ii - 1);
                    yright_mid[2] = roll(vx_transpose[ii], ii + 1);
                    yright_most[2] = roll(vx_transpose[ii], ii + 2);

                    yleft_most[3] = roll(vy_transpose[ii], ii - 2);
                    yleft_mid[3] = roll(vy_transpose[ii], ii - 1);
                    yright_mid[3] = roll(vy_transpose[ii], ii + 1);
                    yright_most[3] = roll(vy_transpose[ii], ii + 2);

                } else {
                    // Adjust for beginning input of L vector
                    xcoordinate = ii - 2;
                    ycoordinate = jj - 2;

                    // Coordinate X
                    xleft_most[0] = prims[0][jj][ii - 2];
                    xleft_mid[0] = prims[0][jj][ii - 1];
                    center[0] = prims[0][jj][ii];
                    xright_mid[0] = prims[0][jj][ii + 1];
                    xright_most[0] = prims[0][jj][ii + 2];

                    xleft_most[1] = prims[1][jj][ii - 2];
                    xleft_mid[1] = prims[1][jj][ii - 1];
                    center[1] = prims[1][jj][ii];
                    xright_mid[1] = prims[1][jj][ii + 1];
                    xright_most[1] = prims[1][jj][ii + 2];

                    xleft_most[2] = prims[2][jj][ii - 2];
                    xleft_mid[2] = prims[2][jj][ii - 1];
                    center[2] = prims[2][jj][ii];
                    xright_mid[2] = prims[2][jj][ii + 1];
                    xright_most[2] = prims[2][jj][ii + 2];

                    xleft_most[3] = prims[3][jj][ii - 2];
                    xleft_mid[3] = prims[3][jj][ii - 1];
                    center[3] = prims[3][jj][ii];
                    xright_mid[3] = prims[3][jj][ii + 1];
                    xright_most[3] = prims[3][jj][ii + 2];

                    // Coordinate Y
                    yleft_most[0] = prims[0][jj - 2][ii];
                    yleft_mid[0] = prims[0][jj - 1][ii];
                    yright_mid[0] = prims[0][jj + 1][ii];
                    yright_most[0] = prims[0][jj + 2][ii];

                    yleft_most[1] = prims[1][jj - 2][ii];
                    yleft_mid[1] = prims[1][jj - 1][ii];
                    yright_mid[1] = prims[1][jj + 1][ii];
                    yright_most[1] = prims[1][jj + 2][ii];

                    yleft_most[2] = prims[2][jj - 2][ii];
                    yleft_mid[2] = prims[2][jj - 1][ii];
                    yright_mid[2] = prims[2][jj + 1][ii];
                    yright_most[2] = prims[2][jj + 2][ii];

                    yleft_most[3] = prims[3][jj - 2][ii];
                    yleft_mid[3] = prims[3][jj - 1][ii];
                    yright_mid[3] = prims[3][jj + 1][ii];
                    yright_most[3] = prims[3][jj + 2][ii];

                }
                
                // Reconstructed left X primitives vector at the i+1/2 interface
                xprims_l[0] = center[0] + 0.5*minmod(theta*(center[0] - xleft_mid[0]),
                                                    0.5*(xright_mid[0] - xleft_mid[0]),
                                                    theta*(xright_mid[0] - center[0]));

                
                xprims_l[1] = center[1] + 0.5*minmod(theta*(center[1] - xleft_mid[1]),
                                                    0.5*(xright_mid[1] - xleft_mid[1]),
                                                    theta*(xright_mid[1] - center[1]));

                xprims_l[2] = center[2] + 0.5*minmod(theta*(center[2] - xleft_mid[2]),
                                                    0.5*(xright_mid[2] - xleft_mid[2]),
                                                    theta*(xright_mid[2] - center[2]));

                xprims_l[3] = center[3] + 0.5*minmod(theta*(center[3] - xleft_mid[3]),
                                                    0.5*(xright_mid[3] - xleft_mid[3]),
                                                    theta*(xright_mid[3] - center[3]));

                // Reconstructed right primitives vector in x
                xprims_r[0] = xright_mid[0] - 0.5*minmod(theta*(xright_mid[0] - center[0]),
                                                    0.5*(xright_most[0] - center[0]),
                                                    theta*(xright_most[0] - xright_mid[0]));

                xprims_r[1] = xright_mid[1] - 0.5*minmod(theta*(xright_mid[1] - center[1]),
                                                    0.5*(xright_most[1] - center[1]),
                                                    theta*(xright_most[1] - xright_mid[1]));

                xprims_r[2] = xright_mid[2] - 0.5*minmod(theta*(xright_mid[2] - center[2]),
                                                    0.5*(xright_most[2] - center[2]),
                                                    theta*(xright_most[2] - xright_mid[2]));

                xprims_r[3] = xright_mid[3] - 0.5*minmod(theta*(xright_mid[3] - center[3]),
                                                    0.5*(xright_most[3] - center[3]),
                                                    theta*(xright_most[3] - xright_mid[3]));

                
                // Reconstructed right primitives vector in y-direction at j+1/2 interfce
                yprims_l[0] = center[0] + 0.5*minmod(theta*(center[0] - yleft_mid[0]),
                                                    0.5*(yright_mid[0] - yleft_mid[0]),
                                                    theta*(yright_mid[0] - center[0]));

                yprims_l[1] = center[1] + 0.5*minmod(theta*(center[1] - yleft_mid[1]),
                                                    0.5*(yright_mid[1] - yleft_mid[1]),
                                                    theta*(yright_mid[1] - center[1]));

                yprims_l[2] = center[2] + 0.5*minmod(theta*(center[2] - yleft_mid[2]),
                                                    0.5*(yright_mid[2] - yleft_mid[2]),
                                                    theta*(yright_mid[2] - center[2]));

                yprims_l[3] = center[3] + 0.5*minmod(theta*(center[3] - yleft_mid[3]),
                                                    0.5*(yright_mid[3] - yleft_mid[3]),
                                                    theta*(yright_mid[3] - center[3]));
                

                yprims_r[0] = yright_mid[0] - 0.5*minmod(theta*(yright_mid[0] - center[0]),
                                                    0.5*(yright_most[0] - center[0]),
                                                    theta*(yright_most[0] - yright_mid[0]));

                yprims_r[1] = yright_mid[1] - 0.5*minmod(theta*(yright_mid[1] - center[1]),
                                                    0.5*(yright_most[1] - center[1]),
                                                    theta*(yright_most[1] - yright_mid[1]));

                yprims_r[2] = yright_mid[2] - 0.5*minmod(theta*(yright_mid[2] - center[2]),
                                                    0.5*(yright_most[2] - center[2]),
                                                    theta*(yright_most[2] - yright_mid[2]));

                yprims_r[3] = yright_mid[3] - 0.5*minmod(theta*(yright_mid[3] - center[3]),
                                                    0.5*(yright_most[3] - center[3]),
                                                    theta*(yright_most[3] - yright_mid[3]));

            
                
                // Calculate the left and right states using the reconstructed PLM primitives
                ux_l = calc_stateSR2D(gamma, xprims_l[0], xprims_l[1], xprims_l[2], xprims_l[3]);
                ux_r = calc_stateSR2D(gamma, xprims_r[0], xprims_r[1], xprims_r[2], xprims_r[3]);

                uy_l = calc_stateSR2D(gamma, yprims_l[0], yprims_l[1], yprims_l[2], yprims_l[3]);
                uy_r = calc_stateSR2D(gamma, yprims_r[0], yprims_r[1], yprims_r[2], yprims_r[3]);

                f_l = calc_fluxSR2D(gamma, xprims_l[0], xprims_l[1], xprims_l[2], xprims_l[3]);
                f_r = calc_fluxSR2D(gamma, xprims_r[0], xprims_r[1], xprims_r[2], xprims_r[3]);

                g_l = calc_fluxSR2D(gamma, yprims_l[0], yprims_l[1], yprims_l[2], yprims_l[3], false);
                g_r = calc_fluxSR2D(gamma, yprims_r[0], yprims_r[1], yprims_r[2], yprims_r[3], false);

                vx_l = sqrt(xprims_l[2]*xprims_l[2] + xprims_l[3]*xprims_l[3]);
                vx_r = sqrt(xprims_r[2]*xprims_r[2] + xprims_r[3]*xprims_r[3]);

                vy_l = sqrt(yprims_l[2]*yprims_l[2] + yprims_l[3]*yprims_l[3]);
                vy_r = sqrt(yprims_r[2]*yprims_r[2] + yprims_r[3]*yprims_r[3]);

                Wx_l = calc_lorentz_gamma(vx_l);
                Wx_r = calc_lorentz_gamma(vx_r);

                Wy_l = calc_lorentz_gamma(vy_l);
                Wy_r = calc_lorentz_gamma(vy_r);

                f1 = calc_hll_flux(gamma, Wx_l, Wx_r, ux_l, ux_r, f_l, f_r, "x");
                g1 = calc_hll_flux(gamma, Wy_l, Wy_r,  uy_l, uy_r, g_l, g_r, "y");
                




                // Left side primitives in x
                xprims_l[0] = xleft_mid[0] + 0.5 *minmod(theta*(xleft_mid[0] - xleft_most[0]),
                                                        0.5*(center[0] - xleft_most[0]),
                                                        theta*(center[0] - xleft_mid[0]));

                xprims_l[1] = xleft_mid[1] + 0.5 *minmod(theta*(xleft_mid[1] - xleft_most[1]),
                                                        0.5*(center[1] -xleft_most[1]),
                                                        theta*(center[1] - xleft_mid[1]));
                
                xprims_l[2] = xleft_mid[2] + 0.5 *minmod(theta*(xleft_mid[2] - xleft_most[2]),
                                                        0.5*(center[2] - xleft_most[2]),
                                                        theta*(center[2] - xleft_mid[2]));
                
                xprims_l[3] = xleft_mid[3] + 0.5 *minmod(theta*(xleft_mid[3] - xleft_most[3]),
                                                        0.5*(center[3] - xleft_most[3]),
                                                        theta*(center[3] - xleft_mid[3]));

                    
                // Right side primitives in x
                xprims_r[0] = center[0] - 0.5 *minmod(theta*(center[0] - xleft_mid[0]),
                                                    0.5*(xright_mid[0] - xleft_mid[0]),
                                                    theta*(xright_mid[0] - center[0]));

                xprims_r[1] = center[1] - 0.5 *minmod(theta*(center[1] - xleft_mid[1]),
                                                    0.5*(xright_mid[1] - xleft_mid[1]),
                                                    theta*(xright_mid[1] - center[1]));

                xprims_r[2] = center[2] - 0.5 *minmod(theta*(center[2] - xleft_mid[2]),
                                                    0.5*(xright_mid[2] - xleft_mid[2]),
                                                    theta*(xright_mid[2] - center[2]));

                xprims_r[3] = center[3] - 0.5 *minmod(theta*(center[3] - xleft_mid[3]),
                                                    0.5*(xright_mid[3] - xleft_mid[3]),
                                                    theta*(xright_mid[3] - center[3]));


                // Left side primitives in y
                yprims_l[0] = yleft_mid[0] + 0.5 *minmod(theta*(yleft_mid[0] - yleft_most[0]),
                                                        0.5*(center[0] - yleft_most[0]),
                                                        theta*(center[0] - yleft_mid[0]));

                yprims_l[1] = yleft_mid[1] + 0.5 *minmod(theta*(yleft_mid[1] - yleft_most[1]),
                                                        0.5*(center[1] -yleft_most[1]),
                                                        theta*(center[1] - yleft_mid[1]));
                
                yprims_l[2] = yleft_mid[2] + 0.5 *minmod(theta*(yleft_mid[2] - yleft_most[2]),
                                                        0.5*(center[2] - yleft_most[2]),
                                                        theta*(center[2] - yleft_mid[2]));
                
                yprims_l[3] = yleft_mid[3] + 0.5 *minmod(theta*(yleft_mid[3] - yleft_most[3]),
                                                        0.5*(center[3] - yleft_most[3]),
                                                        theta*(center[3] - yleft_mid[3]));

                    
                // Right side primitives in y
                yprims_r[0] = center[0] - 0.5 *minmod(theta*(center[0] - yleft_mid[0]),
                                                    0.5*(yright_mid[0] - yleft_mid[0]),
                                                    theta*(yright_mid[0] - center[0]));

                yprims_r[1] = center[1] - 0.5 *minmod(theta*(center[1] - yleft_mid[1]),
                                                    0.5*(yright_mid[1] - yleft_mid[1]),
                                                    theta*(yright_mid[1] - center[1]));

                yprims_r[2] = center[2] - 0.5 *minmod(theta*(center[2] - yleft_mid[2]),
                                                    0.5*(yright_mid[2] - yleft_mid[2]),
                                                    theta*(yright_mid[2] - center[2]));

                yprims_r[3] = center[3]  - 0.5 *minmod(theta*(center[3] - yleft_mid[3]),
                                                    0.5*(yright_mid[3] - yleft_mid[3]),
                                                    theta*(yright_mid[3] - center[3])); 
                
            

                // Calculate the left and right states using the reconstructed PLM primitives
                ux_l = calc_stateSR2D(gamma,xprims_l[0], xprims_l[1], xprims_l[2], xprims_l[3]);
                ux_r = calc_stateSR2D(gamma,xprims_r[0], xprims_r[1], xprims_r[2], xprims_r[3]);

                uy_l = calc_stateSR2D(gamma, yprims_l[0], yprims_l[1], yprims_l[2], yprims_l[3]);
                uy_r = calc_stateSR2D(gamma, yprims_r[0], yprims_r[1], yprims_r[2], yprims_r[3]);

                f_l = calc_fluxSR2D(gamma, xprims_l[0], xprims_l[1], xprims_l[2], xprims_l[3]);
                f_r = calc_fluxSR2D(gamma, xprims_r[0], xprims_r[1], xprims_r[2], xprims_r[3]);

                g_l = calc_fluxSR2D(gamma, yprims_l[0], yprims_l[1], yprims_l[2], yprims_l[3], false);
                g_r = calc_fluxSR2D(gamma, yprims_r[0], yprims_r[1], yprims_r[2], yprims_r[3], false);


                vx_l = sqrt(xprims_l[2]*xprims_l[2] + xprims_l[3]*xprims_l[3]);
                vx_r = sqrt(xprims_r[2]*xprims_r[2] + xprims_r[3]*xprims_r[3]);

                vy_l = sqrt(yprims_l[2]*yprims_l[2] + yprims_l[3]*yprims_l[3]);
                vy_r = sqrt(yprims_r[2]*yprims_r[2] + yprims_r[3]*yprims_r[3]);

                Wx_l = calc_lorentz_gamma(vx_l);
                Wx_r = calc_lorentz_gamma(vx_r);

                Wy_l = calc_lorentz_gamma(vy_l);
                Wy_r = calc_lorentz_gamma(vy_r);

                f2 = calc_hll_flux(gamma, Wx_l, Wx_r, ux_l, ux_r, f_l, f_r, "x");
                g2 = calc_hll_flux(gamma, Wy_l, Wy_r,  uy_l, uy_r, g_l, g_r, "y");
                

                
                

                L[0][ycoordinate][xcoordinate] = - (f1[0] - f2[0])/dx - (g1[0] - g2[0])/dy;
                L[1][ycoordinate][xcoordinate] = - (f1[1] - f2[1])/dx - (g1[1] - g2[1])/dy;
                L[2][ycoordinate][xcoordinate] = - (f1[2] - f2[2])/dx - (g1[2] - g2[2])/dy;
                L[3][ycoordinate][xcoordinate] = - (f1[3] - f2[3])/dx - (g1[3] - g2[3])/dy;
                
            }

        }

        return L;

    } else {
        //==============================================================================================
        //                                  SPHERICAL 
        //==============================================================================================
        double right_cell, left_cell, upper_cell, lower_cell, ang_avg; 
        double r_left, r_right, volAvg, pc, rhoc, vc, uc, deltaV1, deltaV2;
        double log_rLeft, log_rRight;
        double theta_right, theta_left, tcoordinate, rcoordinate;
        double upper_tsurface, lower_tsurface, right_rsurface, left_rsurface;

        double delta_logr = (log10(x1[xphysical_grid - 1]) - log10(x1[0]))/xphysical_grid;

        double dr; 

        for (int jj = j_start; jj < j_bound; jj++){
            for (int ii = i_start; ii < i_bound; ii++){
                if (periodic){
                    rcoordinate = ii;
                    tcoordinate = jj;

                    // Declare the c[i-2],c[i-1],c_i,c[i+1], c[i+2] variables

                    // X Coordinate
                    xleft_most[0] = roll(prims[0][ii], ii - 2);
                    xleft_mid[0] = roll(prims[0][ii], ii - 1);
                    center[0] = prims[0][ii][jj];
                    xright_mid[0] = roll(prims[0][ii], ii + 1);
                    xright_most[0] = roll(prims[0][ii], ii + 2);

                    xleft_most[1] = roll(prims[1][ii], ii - 2);
                    xleft_mid[1] = roll(prims[1][ii], ii - 1);
                    center[1] = prims[1][ii][jj];
                    xright_mid[1] = roll(prims[1][ii], ii + 1);
                    xright_most[1] = roll(prims[1][ii], ii + 2);

                    xleft_most[2] = roll(prims[2][ii], ii - 2);
                    xleft_mid[2] = roll(prims[2][ii], ii - 1);
                    center[2] = prims[2][ii][jj];
                    xright_mid[2] = roll(prims[2][ii], ii + 1);
                    xright_most[2] = roll(prims[2][ii], ii + 2);

                    xleft_most[3] = roll(prims[3][ii], ii - 2);
                    xleft_mid[3] = roll(prims[3][ii], ii - 1);
                    center[3] = prims[3][ii][jj];
                    xright_mid[3] = roll(prims[3][ii], ii + 1);
                    xright_most[3] = roll(prims[3][ii], ii + 2);

                    // Transpose the prims matrix to compute the Y Sweep
                    rho_transpose = transpose(prims[0]);
                    pressure_transpose = transpose(prims[1]);
                    vx_transpose = transpose(prims[2]);
                    vy_transpose = transpose(prims[3]);

                    yleft_most[0] = roll(rho_transpose[ii], ii - 2);
                    yleft_mid[0] = roll(rho_transpose[ii], ii - 1);
                    yright_mid[0] = roll(rho_transpose[ii], ii + 1);
                    yright_most[0] = roll(rho_transpose[ii], ii + 2);

                    yleft_most[1] = roll(pressure_transpose[ii], ii - 2);
                    yleft_mid[1] = roll(pressure_transpose[ii], ii - 1);
                    yright_mid[1] = roll(pressure_transpose[ii], ii + 1);
                    yright_most[1] = roll(pressure_transpose[ii], ii + 2);

                    yleft_most[2] = roll(vx_transpose[ii], ii - 2);
                    yleft_mid[2] = roll(vx_transpose[ii], ii - 1);
                    yright_mid[2] = roll(vx_transpose[ii], ii + 1);
                    yright_most[2] = roll(vx_transpose[ii], ii + 2);

                    yleft_most[3] = roll(vy_transpose[ii], ii - 2);
                    yleft_mid[3] = roll(vy_transpose[ii], ii - 1);
                    yright_mid[3] = roll(vy_transpose[ii], ii + 1);
                    yright_most[3] = roll(vy_transpose[ii], ii + 2);

                } else {
                    // Adjust for beginning input of L vector
                    rcoordinate = ii - 2;
                    tcoordinate = jj - 2;

                    // Coordinate X
                    xleft_most[0] = prims[0][jj][ii - 2];
                    xleft_mid[0] = prims[0][jj][ii - 1];
                    center[0] = prims[0][jj][ii];
                    xright_mid[0] = prims[0][jj][ii + 1];
                    xright_most[0] = prims[0][jj][ii + 2];

                    xleft_most[1] = prims[1][jj][ii - 2];
                    xleft_mid[1] = prims[1][jj][ii - 1];
                    center[1] = prims[1][jj][ii];
                    xright_mid[1] = prims[1][jj][ii + 1];
                    xright_most[1] = prims[1][jj][ii + 2];

                    xleft_most[2] = prims[2][jj][ii - 2];
                    xleft_mid[2] = prims[2][jj][ii - 1];
                    center[2] = prims[2][jj][ii];
                    xright_mid[2] = prims[2][jj][ii + 1];
                    xright_most[2] = prims[2][jj][ii + 2];

                    xleft_most[3] = prims[3][jj][ii - 2];
                    xleft_mid[3] = prims[3][jj][ii - 1];
                    center[3] = prims[3][jj][ii];
                    xright_mid[3] = prims[3][jj][ii + 1];
                    xright_most[3] = prims[3][jj][ii + 2];

                    // Coordinate Y
                    yleft_most[0] = prims[0][jj - 2][ii];
                    yleft_mid[0] = prims[0][jj - 1][ii];
                    yright_mid[0] = prims[0][jj + 1][ii];
                    yright_most[0] = prims[0][jj + 2][ii];

                    yleft_most[1] = prims[1][jj - 2][ii];
                    yleft_mid[1] = prims[1][jj - 1][ii];
                    yright_mid[1] = prims[1][jj + 1][ii];
                    yright_most[1] = prims[1][jj + 2][ii];

                    yleft_most[2] = prims[2][jj - 2][ii];
                    yleft_mid[2] = prims[2][jj - 1][ii];
                    yright_mid[2] = prims[2][jj + 1][ii];
                    yright_most[2] = prims[2][jj + 2][ii];

                    yleft_most[3] = prims[3][jj - 2][ii];
                    yleft_mid[3] = prims[3][jj - 1][ii];
                    yright_mid[3] = prims[3][jj + 1][ii];
                    yright_most[3] = prims[3][jj + 2][ii];

                }
                
                // Reconstructed left X primitives vector at the i+1/2 interface
                xprims_l[0] = center[0] + 0.5*minmod(theta*(center[0] - xleft_mid[0]),
                                                    0.5*(xright_mid[0] - xleft_mid[0]),
                                                    theta*(xright_mid[0] - center[0]));

                
                xprims_l[1] = center[1] + 0.5*minmod(theta*(center[1] - xleft_mid[1]),
                                                    0.5*(xright_mid[1] - xleft_mid[1]),
                                                    theta*(xright_mid[1] - center[1]));

                xprims_l[2] = center[2] + 0.5*minmod(theta*(center[2] - xleft_mid[2]),
                                                    0.5*(xright_mid[2] - xleft_mid[2]),
                                                    theta*(xright_mid[2] - center[2]));

                xprims_l[3] = center[3] + 0.5*minmod(theta*(center[3] - xleft_mid[3]),
                                                    0.5*(xright_mid[3] - xleft_mid[3]),
                                                    theta*(xright_mid[3] - center[3]));

                // Reconstructed right primitives vector in x
                xprims_r[0] = xright_mid[0] - 0.5*minmod(theta*(xright_mid[0] - center[0]),
                                                    0.5*(xright_most[0] - center[0]),
                                                    theta*(xright_most[0] - xright_mid[0]));

                xprims_r[1] = xright_mid[1] - 0.5*minmod(theta*(xright_mid[1] - center[1]),
                                                    0.5*(xright_most[1] - center[1]),
                                                    theta*(xright_most[1] - xright_mid[1]));

                xprims_r[2] = xright_mid[2] - 0.5*minmod(theta*(xright_mid[2] - center[2]),
                                                    0.5*(xright_most[2] - center[2]),
                                                    theta*(xright_most[2] - xright_mid[2]));

                xprims_r[3] = xright_mid[3] - 0.5*minmod(theta*(xright_mid[3] - center[3]),
                                                    0.5*(xright_most[3] - center[3]),
                                                    theta*(xright_most[3] - xright_mid[3]));

                
                // Reconstructed right primitives vector in y-direction at j+1/2 interfce
                yprims_l[0] = center[0] + 0.5*minmod(theta*(center[0] - yleft_mid[0]),
                                                    0.5*(yright_mid[0] - yleft_mid[0]),
                                                    theta*(yright_mid[0] - center[0]));

                yprims_l[1] = center[1] + 0.5*minmod(theta*(center[1] - yleft_mid[1]),
                                                    0.5*(yright_mid[1] - yleft_mid[1]),
                                                    theta*(yright_mid[1] - center[1]));

                yprims_l[2] = center[2] + 0.5*minmod(theta*(center[2] - yleft_mid[2]),
                                                    0.5*(yright_mid[2] - yleft_mid[2]),
                                                    theta*(yright_mid[2] - center[2]));

                yprims_l[3] = center[3] + 0.5*minmod(theta*(center[3] - yleft_mid[3]),
                                                    0.5*(yright_mid[3] - yleft_mid[3]),
                                                    theta*(yright_mid[3] - center[3]));
                

                yprims_r[0] = yright_mid[0] - 0.5*minmod(theta*(yright_mid[0] - center[0]),
                                                    0.5*(yright_most[0] - center[0]),
                                                    theta*(yright_most[0] - yright_mid[0]));

                yprims_r[1] = yright_mid[1] - 0.5*minmod(theta*(yright_mid[1] - center[1]),
                                                    0.5*(yright_most[1] - center[1]),
                                                    theta*(yright_most[1] - yright_mid[1]));

                yprims_r[2] = yright_mid[2] - 0.5*minmod(theta*(yright_mid[2] - center[2]),
                                                    0.5*(yright_most[2] - center[2]),
                                                    theta*(yright_most[2] - yright_mid[2]));

                yprims_r[3] = yright_mid[3] - 0.5*minmod(theta*(yright_mid[3] - center[3]),
                                                    0.5*(yright_most[3] - center[3]),
                                                    theta*(yright_most[3] - yright_mid[3]));

            
                
                
                
                
                // Calculate the left and right states using the reconstructed PLM primitives
                ux_l = calc_stateSR2D(gamma, xprims_l[0], xprims_l[1], xprims_l[2], xprims_l[3]);
                ux_r = calc_stateSR2D(gamma, xprims_r[0], xprims_r[1], xprims_r[2], xprims_r[3]);

                uy_l = calc_stateSR2D(gamma, yprims_l[0], yprims_l[1], yprims_l[2], yprims_l[3]);
                uy_r = calc_stateSR2D(gamma, yprims_r[0], yprims_r[1], yprims_r[2], yprims_r[3]);

                f_l = calc_fluxSR2D(gamma, xprims_l[0], xprims_l[1], xprims_l[2], xprims_l[3]);
                f_r = calc_fluxSR2D(gamma, xprims_r[0], xprims_r[1], xprims_r[2], xprims_r[3]);

                g_l = calc_fluxSR2D(gamma, yprims_l[0], yprims_l[1], yprims_l[2], yprims_l[3], false);
                g_r = calc_fluxSR2D(gamma, yprims_r[0], yprims_r[1], yprims_r[2], yprims_r[3], false);

                vx_l = sqrt(xprims_l[2]*xprims_l[2] + xprims_l[3]*xprims_l[3]);
                vx_r = sqrt(xprims_r[2]*xprims_r[2] + xprims_r[3]*xprims_r[3]);

                vy_l = sqrt(yprims_l[2]*yprims_l[2] + yprims_l[3]*yprims_l[3]);
                vy_r = sqrt(yprims_r[2]*yprims_r[2] + yprims_r[3]*yprims_r[3]);

                Wx_l = calc_lorentz_gamma(vx_l);
                Wx_r = calc_lorentz_gamma(vx_r);

                Wy_l = calc_lorentz_gamma(vy_l);
                Wy_r = calc_lorentz_gamma(vy_r);
                
                f1 = calc_hll_flux(gamma, Wx_l, Wx_r, ux_l, ux_r, f_l, f_r, "x");
                g1 = calc_hll_flux(gamma, Wy_l, Wy_r,  uy_l, uy_r, g_l, g_r, "y");


                /**
                if ((tcoordinate == 0) && (rcoordinate == 0)){
                    cout << ii << endl;
                    cout << jj << endl;
                    cout << "Rho  (L): " << xprims_l[0] << endl;
                    cout << "Rho  (R): " << xprims_r[0] << endl;
                    cout << "P  (L): " << xprims_l[1] << endl;
                    cout << "P  (R): " << xprims_r[1] << endl;
                    cout << "V_r  (L): " << xprims_l[2] << endl;
                    cout << "V_r  (R): " << xprims_l[2] << endl;
                    cout << "V_t  (L): " << xprims_l[3] << endl;
                    cout << "V_t  (R): " << xprims_l[3] << endl;
                    cout << " " << endl;
                    cout << "Rho  (L): " << yprims_l[0] << endl;
                    cout << "Rho  (R): " << yprims_r[0] << endl;
                    cout << "P  (L): " << yprims_l[1] << endl;
                    cout << "P  (R): " << yprims_r[1] << endl;
                    cout << "V_r  (L): " << yprims_l[2] << endl;
                    cout << "V_r  (R): " << yprims_r[2] << endl;
                    cout << "V_t  (L): " << yprims_l[3] << endl;
                    cout << "V_t  (R): " << yprims_r[3] << endl;
                    string h;
                    cin >> h;

                }
                */
                
                
                



                // Do the same thing, but for the left side interface [i - 1/2]

                // Left side primitives in x
                xprims_l[0] = xleft_mid[0] + 0.5 *minmod(theta*(xleft_mid[0] - xleft_most[0]),
                                                        0.5*(center[0] - xleft_most[0]),
                                                        theta*(center[0] - xleft_mid[0]));

                xprims_l[1] = xleft_mid[1] + 0.5 *minmod(theta*(xleft_mid[1] - xleft_most[1]),
                                                        0.5*(center[1] -xleft_most[1]),
                                                        theta*(center[1] - xleft_mid[1]));
                
                xprims_l[2] = xleft_mid[2] + 0.5 *minmod(theta*(xleft_mid[2] - xleft_most[2]),
                                                        0.5*(center[2] - xleft_most[2]),
                                                        theta*(center[2] - xleft_mid[2]));
                
                xprims_l[3] = xleft_mid[3] + 0.5 *minmod(theta*(xleft_mid[3] - xleft_most[3]),
                                                        0.5*(center[3] - xleft_most[3]),
                                                        theta*(center[3] - xleft_mid[3]));

                    
                // Right side primitives in x
                xprims_r[0] = center[0] - 0.5 *minmod(theta*(center[0] - xleft_mid[0]),
                                                    0.5*(xright_mid[0] - xleft_mid[0]),
                                                    theta*(xright_mid[0] - center[0]));

                xprims_r[1] = center[1] - 0.5 *minmod(theta*(center[1] - xleft_mid[1]),
                                                    0.5*(xright_mid[1] - xleft_mid[1]),
                                                    theta*(xright_mid[1] - center[1]));

                xprims_r[2] = center[2] - 0.5 *minmod(theta*(center[2] - xleft_mid[2]),
                                                    0.5*(xright_mid[2] - xleft_mid[2]),
                                                    theta*(xright_mid[2] - center[2]));

                xprims_r[3] = center[3] - 0.5 *minmod(theta*(center[3] - xleft_mid[3]),
                                                    0.5*(xright_mid[3] - xleft_mid[3]),
                                                    theta*(xright_mid[3] - center[3]));


                // Left side primitives in y
                yprims_l[0] = yleft_mid[0] + 0.5 *minmod(theta*(yleft_mid[0] - yleft_most[0]),
                                                        0.5*(center[0] - yleft_most[0]),
                                                        theta*(center[0] - yleft_mid[0]));

                yprims_l[1] = yleft_mid[1] + 0.5 *minmod(theta*(yleft_mid[1] - yleft_most[1]),
                                                        0.5*(center[1] -yleft_most[1]),
                                                        theta*(center[1] - yleft_mid[1]));
                
                yprims_l[2] = yleft_mid[2] + 0.5 *minmod(theta*(yleft_mid[2] - yleft_most[2]),
                                                        0.5*(center[2] - yleft_most[2]),
                                                        theta*(center[2] - yleft_mid[2]));
                
                yprims_l[3] = yleft_mid[3] + 0.5 *minmod(theta*(yleft_mid[3] - yleft_most[3]),
                                                        0.5*(center[3] - yleft_most[3]),
                                                        theta*(center[3] - yleft_mid[3]));

                    
                // Right side primitives in y
                yprims_r[0] = center[0] - 0.5 *minmod(theta*(center[0] - yleft_mid[0]),
                                                    0.5*(yright_mid[0] - yleft_mid[0]),
                                                    theta*(yright_mid[0] - center[0]));

                yprims_r[1] = center[1] - 0.5 *minmod(theta*(center[1] - yleft_mid[1]),
                                                    0.5*(yright_mid[1] - yleft_mid[1]),
                                                    theta*(yright_mid[1] - center[1]));

                yprims_r[2] = center[2] - 0.5 *minmod(theta*(center[2] - yleft_mid[2]),
                                                    0.5*(yright_mid[2] - yleft_mid[2]),
                                                    theta*(yright_mid[2] - center[2]));

                yprims_r[3] = center[3]  - 0.5 *minmod(theta*(center[3] - yleft_mid[3]),
                                                    0.5*(yright_mid[3] - yleft_mid[3]),
                                                    theta*(yright_mid[3] - center[3])); 
                
            

                // Calculate the left and right states using the reconstructed PLM primitives
                ux_l = calc_stateSR2D(gamma,xprims_l[0], xprims_l[1], xprims_l[2], xprims_l[3]);
                ux_r = calc_stateSR2D(gamma,xprims_r[0], xprims_r[1], xprims_r[2], xprims_r[3]);

                uy_l = calc_stateSR2D(gamma, yprims_l[0], yprims_l[1], yprims_l[2], yprims_l[3]);
                uy_r = calc_stateSR2D(gamma, yprims_r[0], yprims_r[1], yprims_r[2], yprims_r[3]);

                f_l = calc_fluxSR2D(gamma, xprims_l[0], xprims_l[1], xprims_l[2], xprims_l[3]);
                f_r = calc_fluxSR2D(gamma, xprims_r[0], xprims_r[1], xprims_r[2], xprims_r[3]);

                g_l = calc_fluxSR2D(gamma, yprims_l[0], yprims_l[1], yprims_l[2], yprims_l[3], false);
                g_r = calc_fluxSR2D(gamma, yprims_r[0], yprims_r[1], yprims_r[2], yprims_r[3], false);

                vx_l = sqrt(xprims_l[2]*xprims_l[2] + xprims_l[3]*xprims_l[3]);
                vx_r = sqrt(xprims_r[2]*xprims_r[2] + xprims_r[3]*xprims_r[3]);

                vy_l = sqrt(yprims_l[2]*yprims_l[2] + yprims_l[3]*yprims_l[3]);
                vy_r = sqrt(yprims_r[2]*yprims_r[2] + yprims_r[3]*yprims_r[3]);

                Wx_l = calc_lorentz_gamma(vx_l);
                Wx_r = calc_lorentz_gamma(vx_r);

                Wy_l = calc_lorentz_gamma(vy_l);
                Wy_r = calc_lorentz_gamma(vy_r);
                
                f2 = calc_hll_flux(gamma, Wx_l, Wx_r, ux_l, ux_r, f_l, f_r, "x");
                g2 = calc_hll_flux(gamma, Wy_l, Wy_r,  uy_l, uy_r, g_l, g_r, "y");

                if (linspace){
                    right_cell = x1[rcoordinate + 1];
                    left_cell = x1[rcoordinate - 1];
                    upper_cell = x2[tcoordinate + 1];
                    lower_cell = x2[tcoordinate - 1];

                    // cout << "Theta Coordinate: " << tcoordinate << endl;
                    // cout << "R Coordinate: " << rcoordinate << endl;
                    
                    // Outflow the left/right boundaries
                    if (rcoordinate - 1 < 0){
                        left_cell = x1[rcoordinate];

                    } else if (rcoordinate == xphysical_grid - 1){
                        right_cell = x1[rcoordinate];

                    }

                    if (tcoordinate - 1 < 0){

                        lower_cell = x2[tcoordinate];
                        
                    }  else if(tcoordinate == yphysical_grid - 1){
                        upper_cell = x2[tcoordinate];
                    }

                    r_right = 0.5*(right_cell + x1[rcoordinate]);
                    r_left = 0.5*(x1[rcoordinate] + left_cell);

                    theta_right = atan2( sin(upper_cell) + sin(x2[tcoordinate]) , 
                                                cos(upper_cell) + cos(x2[tcoordinate]) );

                    theta_left = atan2( sin(lower_cell) + sin(x2[tcoordinate]), 
                                                cos(lower_cell) + cos(x2[tcoordinate]) );

                } else {
                    log_rLeft = log10(x1[0]) + rcoordinate*delta_logr;
                    log_rRight = log_rLeft + delta_logr;
                    r_left = pow(10, log_rLeft);
                    r_right = pow(10, log_rRight);

                    upper_cell = x2[tcoordinate + 1];
                    lower_cell = x2[tcoordinate - 1];
                    
                    // Outflow the left/right boundaries
                    if (tcoordinate - 1 < 0){
                        lower_cell = x2[tcoordinate];

                    } else if(tcoordinate == yphysical_grid - 1){
                        upper_cell = x2[tcoordinate];
                    }

                    theta_right = atan2( sin(upper_cell) + sin(x2[tcoordinate]) , 
                                                cos(upper_cell) + cos(x2[tcoordinate]) );

                    theta_left = atan2( sin(lower_cell) + sin(x2[tcoordinate]), 
                                                cos(lower_cell) + cos(x2[tcoordinate]) );
                }

                dr = r_right - r_left;
                rhoc = center[0];
                pc = center[1];
                uc = center[2];
                vc = center[3];

                
                ang_avg =  atan2(sin(theta_right) + sin(theta_left), cos(theta_right) + cos(theta_left) );
                // Compute the surface areas
                right_rsurface = r_right*r_right;
                left_rsurface = r_left*r_left;
                upper_tsurface = 0.5*(r_right*r_right - r_left*r_left)*sin(theta_right);
                lower_tsurface = 0.5*(r_right*r_right - r_left*r_left)*sin(theta_left);
                volAvg = 0.75*((pow(r_right, 4) - pow(r_left, 4))/ (pow(r_right, 3) - pow(r_left, 3)) );
                deltaV1 = pow(volAvg, 2)*dr;
                deltaV2 = deltaV1*(cos(theta_left) - cos(theta_right)); 
                
                /**
                cout << "Sub: " << (cos(theta_left) - cos(theta_right)) << endl;
                cout << "VolAvg:" << volAvg << endl;
                cout << "F1: "<< f1[0] << endl;
                cout << "F2: "<< f2[0] << endl;
                cout << "G1: "<< g1[0] << endl;
                cout << "G2: "<< g2[0] << endl;
                string a;
                cin >> a;
                */

               
                source0 = source1[tcoordinate][rcoordinate];
                sourceD = source2[tcoordinate][rcoordinate];
                sourceR = source3[tcoordinate][rcoordinate];

                // cout << "S0: " << source0 << endl;
                // cout << "SD: " << sourceD << endl;
                // cout << "SR: " << sourceR << endl;
                // cin.get();


                L[0][tcoordinate][rcoordinate] = - (f1[0]*right_rsurface - f2[0]*left_rsurface)/deltaV1
                                                    - (g1[0]*upper_tsurface - g2[0]*lower_tsurface)/deltaV2 + sourceD;

                L[1][tcoordinate][rcoordinate] = - (f1[1]*right_rsurface - f2[1]*left_rsurface)/deltaV1
                                                    - (g1[1]*upper_tsurface - g2[1]*lower_tsurface)/deltaV2 
                                                    + rhoc*vc*vc/volAvg + 2*pc/volAvg + sourceR;

                L[2][tcoordinate][rcoordinate] = - (f1[2]*right_rsurface - f2[2]*left_rsurface)/deltaV1
                                                    - (g1[2]*upper_tsurface - g2[2]*lower_tsurface)/deltaV2
                                                    -(rhoc*uc*vc/volAvg - pc*cos(ang_avg)/(volAvg*sin(ang_avg)));

                L[3][tcoordinate][rcoordinate] = - (f1[3]*right_rsurface - f2[3]*left_rsurface)/deltaV1
                                                    - (g1[3]*upper_tsurface - g2[3]*lower_tsurface)/deltaV2 + source0;
                
            }

        }

        return L;
        
    }
    

};


//-----------------------------------------------------------------------------------------------------------
//                                            SIMULATE 
//-----------------------------------------------------------------------------------------------------------
vector<vector<vector<double> > > UstateSR2D::simulate2D(vector<vector<double> > &lorentz_gamma, float tend = 0.1, bool periodic = false, 
                                                        double dt = 1.e-4, bool linspace=true){

    // Define the swap vector for the integrated state
    int xgrid_size = state2D[0][0].size();
    int ygrid_size = state2D[0].size();
    int n_vars = state2D.size();
    int xphysical_grid = xgrid_size - 4;
    int yphysical_grid = ygrid_size - 4;
    float t = 0;

    
    vector<vector<vector<double> > > u_p(n_vars, vector<vector<double> > 
                                                (ygrid_size, vector<double> (xgrid_size, 0)));

    vector<vector<vector<double> > > u(n_vars, vector<vector<double> > 
                                                (ygrid_size, vector<double> (xgrid_size, 0)));
    
    vector<vector<vector<double> > > udot1(n_vars, vector<vector<double> > 
                                                (ygrid_size, vector<double> (xgrid_size, 0)));

    vector<vector<vector<double> > > udot2(n_vars, vector<vector<double> > 
                                                (ygrid_size, vector<double> (xgrid_size, 0)));

    vector<vector<vector<double> > > udot(n_vars, vector<vector<double> > 
                                                (ygrid_size, vector<double> (xgrid_size, 0)));

    vector<vector<vector<double> > > u1(n_vars, vector<vector<double> > 
                                                (ygrid_size, vector<double> (xgrid_size, 0)));

    vector<vector<vector<double> > > u2(n_vars, vector<vector<double> > 
                                                (ygrid_size, vector<double> (xgrid_size, 0)));

    vector<vector<vector<double> > > prims(n_vars, vector<vector<double> > 
                                                (ygrid_size, vector<double> (xgrid_size, 0)));

    // Copy the state array into real & profile variables
    u = state2D;
    u_p = u;
    u1 = u; 
    u2 = u;
    
    while (t < tend){

        if (t == 0){
            config_ghosts2D(u, xphysical_grid, ygrid_size, false);
        }

        // cout << " " << endl;
        // cout << "U: " << endl;
        // for (int jj=0; jj <ygrid_size; jj++){
        //     for (int ii=0; ii < xgrid_size; ii++){
        //         cout << u[1][jj][ii] << ", ";
        //     }
        //     cout << endl;
        // }
        // string a;
        // cin >> a;
        

        udot = u_dot2D(u, lorentz_gamma, periodic, theta, linspace);

        for (int jj = 0; jj < yphysical_grid; jj++){
            // Get the non-ghost index 
            int j_real = jj + 2;
            for (int ii = 0; ii < xphysical_grid; ii++){
                int i_real = ii + 2; 

                u1[0][j_real][i_real] = u[0][j_real][i_real] + dt*udot[0][jj][ii];
                u1[1][j_real][i_real] = u[1][j_real][i_real] + dt*udot[1][jj][ii];
                u1[2][j_real][i_real] = u[2][j_real][i_real] + dt*udot[2][jj][ii];
                u1[3][j_real][i_real] = u[3][j_real][i_real] + dt*udot[3][jj][ii];
            }
        }
        
        
        /**
        cout << " " << endl;
        cout << "U1: " << endl;
        for (int jj=0; jj <ygrid_size; jj++){
            for (int ii=0; ii < xgrid_size; ii++){
                cout << u1[2][jj][ii] << ", ";
            }
            cout << endl;
        }
        string d;
        cin >> d;
        */
        

        
        config_ghosts2D(u1, xphysical_grid, ygrid_size, false);
        

        prims = cons2prim2D(u1, lorentz_gamma);
        lorentz_gamma = calc_lorentz_gamma(prims[2], prims[3]);

        udot1 = u_dot2D(u1, lorentz_gamma, periodic, theta, linspace);

        
        // cout << " " << endl;
        // cout << "UDot1: " << endl;
        // for (int jj=0; jj <yphysical_grid; jj++){
        //     for (int ii=0; ii < xphysical_grid; ii++){
        //         cout << prims[3][jj][ii] << ", ";
        //         //cout << udot1[0][jj][ii] << ", ";
        //     }
        //     cout << endl;
        // }
        // string p;
        // cin >> p;
        

        for (int jj = 0; jj < yphysical_grid; jj++){
            // Get the non-ghost index 
            int j_real =  jj + 2;
            for (int ii = 0; ii < xphysical_grid; ii++){
                int i_real = ii + 2;
                u2[0][j_real][i_real] = 0.75*u[0][j_real][i_real] + 0.25*u1[0][j_real][i_real] + 0.25*dt*udot1[0][jj][ii];
                u2[1][j_real][i_real] = 0.75*u[1][j_real][i_real] + 0.25*u1[1][j_real][i_real] + 0.25*dt*udot1[1][jj][ii];
                u2[2][j_real][i_real] = 0.75*u[2][j_real][i_real] + 0.25*u1[2][j_real][i_real] + 0.25*dt*udot1[2][jj][ii];
                u2[3][j_real][i_real] = 0.75*u[3][j_real][i_real] + 0.25*u1[3][j_real][i_real] + 0.25*dt*udot1[3][jj][ii];

            }
            

        }

        /**
        cout << " " << endl;
        cout << "U2: " << endl;
        for (int jj=0; jj <ygrid_size; jj++){
            for (int ii=0; ii < xgrid_size; ii++){
                cout << u2[0][jj][ii] << ", ";
            }
            cout << endl;
        }
        string c;
        cin >> c;
        */
        
       
        config_ghosts2D(u2, xphysical_grid, ygrid_size, false);

        prims = cons2prim2D(u2, lorentz_gamma);
        lorentz_gamma = calc_lorentz_gamma(prims[2], prims[3]);

        udot2 = u_dot2D(u2, lorentz_gamma, periodic, theta, linspace);

        for (int jj = 0; jj < yphysical_grid; jj++){
            // Get the non-ghost index 
            int j_real =  jj + 2;
            for (int ii = 0; ii < xphysical_grid; ii++){
                int i_real = ii + 2;
                u_p[0][j_real][i_real] = (1.0/3.0)*u[0][j_real][i_real] + (2.0/3.0)*u2[0][j_real][i_real] + (2.0/3.0)*dt*udot2[0][jj][ii];
                u_p[1][j_real][i_real] = (1.0/3.0)*u[1][j_real][i_real] + (2.0/3.0)*u2[1][j_real][i_real] + (2.0/3.0)*dt*udot2[1][jj][ii];
                u_p[2][j_real][i_real] = (1.0/3.0)*u[2][j_real][i_real] + (2.0/3.0)*u2[2][j_real][i_real] + (2.0/3.0)*dt*udot2[2][jj][ii];
                u_p[3][j_real][i_real] = (1.0/3.0)*u[3][j_real][i_real] + (2.0/3.0)*u2[3][j_real][i_real] + (2.0/3.0)*dt*udot2[3][jj][ii];

            }

        }
        
        config_ghosts2D(u_p, xphysical_grid, ygrid_size, false);

        prims = cons2prim2D(u_p, lorentz_gamma);
        lorentz_gamma = calc_lorentz_gamma(prims[2], prims[3]);
        
        /**
        cout << " " << endl;
        cout << "UP: " << endl;
        for (int jj=0; jj <ygrid_size; jj++){
            for (int ii=0; ii < xgrid_size; ii++){
                cout << u_p[1][jj][ii]/1.e-10 << ", ";
            }
            cout << endl;
        }
        string b;
        cin >> b;
        */
        
        
        if (t > 0){
            dt = adapt_dt(prims);
        }
        
        

        // Swap the arrays
        u.swap(u_p);
        
        t += dt;

        

    }

    prims = cons2prim2D(u, lorentz_gamma);

    return prims;

 };


 //=======================
 //     1 SOURCE TERM
 //=======================
 vector<vector<vector<double> > > UstateSR2D::simulate2D(vector<vector<double> > &lorentz_gamma, vector<vector<double> > &source1,
                                                        float tend = 0.1, bool periodic = false, 
                                                        double dt = 1.e-4, bool linspace=true){

    // Define the swap vector for the integrated state
    int xgrid_size = state2D[0][0].size();
    int ygrid_size = state2D[0].size();
    int n_vars = state2D.size();
    int xphysical_grid = xgrid_size - 4;
    int yphysical_grid = ygrid_size - 4;
    float t = 0;

    
    vector<vector<vector<double> > > u_p(n_vars, vector<vector<double> > 
                                                (ygrid_size, vector<double> (xgrid_size, 0)));

    vector<vector<vector<double> > > u(n_vars, vector<vector<double> > 
                                                (ygrid_size, vector<double> (xgrid_size, 0)));
    
    vector<vector<vector<double> > > udot1(n_vars, vector<vector<double> > 
                                                (ygrid_size, vector<double> (xgrid_size, 0)));

    vector<vector<vector<double> > > udot2(n_vars, vector<vector<double> > 
                                                (ygrid_size, vector<double> (xgrid_size, 0)));

    vector<vector<vector<double> > > udot(n_vars, vector<vector<double> > 
                                                (ygrid_size, vector<double> (xgrid_size, 0)));

    vector<vector<vector<double> > > u1(n_vars, vector<vector<double> > 
                                                (ygrid_size, vector<double> (xgrid_size, 0)));

    vector<vector<vector<double> > > u2(n_vars, vector<vector<double> > 
                                                (ygrid_size, vector<double> (xgrid_size, 0)));

    vector<vector<vector<double> > > prims(n_vars, vector<vector<double> > 
                                                (ygrid_size, vector<double> (xgrid_size, 0)));

    // Copy the state array into real & profile variables
    u = state2D;
    u_p = u;
    u1 = u; 
    u2 = u;
    
    while (t < tend){

        if (t == 0){
            config_ghosts2D(u, xphysical_grid, ygrid_size, false);
        }

        // cout << " " << endl;
        // cout << "U: " << endl;
        // for (int jj=0; jj <ygrid_size; jj++){
        //     for (int ii=0; ii < xgrid_size; ii++){
        //         cout << u[1][jj][ii] << ", ";
        //     }
        //     cout << endl;
        // }
        // string a;
        // cin >> a;
        

        udot = u_dot2D(u, lorentz_gamma, source1, periodic, theta, linspace);

        for (int jj = 0; jj < yphysical_grid; jj++){
            // Get the non-ghost index 
            int j_real = jj + 2;
            for (int ii = 0; ii < xphysical_grid; ii++){
                int i_real = ii + 2; 

                u1[0][j_real][i_real] = u[0][j_real][i_real] + dt*udot[0][jj][ii];
                u1[1][j_real][i_real] = u[1][j_real][i_real] + dt*udot[1][jj][ii];
                u1[2][j_real][i_real] = u[2][j_real][i_real] + dt*udot[2][jj][ii];
                u1[3][j_real][i_real] = u[3][j_real][i_real] + dt*udot[3][jj][ii];
            }
        }
        
        
        /**
        cout << " " << endl;
        cout << "U1: " << endl;
        for (int jj=0; jj <ygrid_size; jj++){
            for (int ii=0; ii < xgrid_size; ii++){
                cout << u1[2][jj][ii] << ", ";
            }
            cout << endl;
        }
        string d;
        cin >> d;
        */
        

        
        config_ghosts2D(u1, xphysical_grid, ygrid_size, false);
        

        prims = cons2prim2D(u1, lorentz_gamma);
        lorentz_gamma = calc_lorentz_gamma(prims[2], prims[3]);

        udot1 = u_dot2D(u1, lorentz_gamma, source1, periodic, theta, linspace);

        
        // cout << " " << endl;
        // cout << "UDot1: " << endl;
        // for (int jj=0; jj <yphysical_grid; jj++){
        //     for (int ii=0; ii < xphysical_grid; ii++){
        //         cout << prims[3][jj][ii] << ", ";
        //         //cout << udot1[0][jj][ii] << ", ";
        //     }
        //     cout << endl;
        // }
        // string p;
        // cin >> p;
        

        for (int jj = 0; jj < yphysical_grid; jj++){
            // Get the non-ghost index 
            int j_real =  jj + 2;
            for (int ii = 0; ii < xphysical_grid; ii++){
                int i_real = ii + 2;
                u2[0][j_real][i_real] = 0.75*u[0][j_real][i_real] + 0.25*u1[0][j_real][i_real] + 0.25*dt*udot1[0][jj][ii];
                u2[1][j_real][i_real] = 0.75*u[1][j_real][i_real] + 0.25*u1[1][j_real][i_real] + 0.25*dt*udot1[1][jj][ii];
                u2[2][j_real][i_real] = 0.75*u[2][j_real][i_real] + 0.25*u1[2][j_real][i_real] + 0.25*dt*udot1[2][jj][ii];
                u2[3][j_real][i_real] = 0.75*u[3][j_real][i_real] + 0.25*u1[3][j_real][i_real] + 0.25*dt*udot1[3][jj][ii];

            }
            

        }

        /**
        cout << " " << endl;
        cout << "U2: " << endl;
        for (int jj=0; jj <ygrid_size; jj++){
            for (int ii=0; ii < xgrid_size; ii++){
                cout << u2[0][jj][ii] << ", ";
            }
            cout << endl;
        }
        string c;
        cin >> c;
        */
        
       
        config_ghosts2D(u2, xphysical_grid, ygrid_size, false);

        prims = cons2prim2D(u2, lorentz_gamma);
        lorentz_gamma = calc_lorentz_gamma(prims[2], prims[3]);

        udot2 = u_dot2D(u2, lorentz_gamma, source1, periodic, theta, linspace);

        for (int jj = 0; jj < yphysical_grid; jj++){
            // Get the non-ghost index 
            int j_real =  jj + 2;
            for (int ii = 0; ii < xphysical_grid; ii++){
                int i_real = ii + 2;
                u_p[0][j_real][i_real] = (1.0/3.0)*u[0][j_real][i_real] + (2.0/3.0)*u2[0][j_real][i_real] + (2.0/3.0)*dt*udot2[0][jj][ii];
                u_p[1][j_real][i_real] = (1.0/3.0)*u[1][j_real][i_real] + (2.0/3.0)*u2[1][j_real][i_real] + (2.0/3.0)*dt*udot2[1][jj][ii];
                u_p[2][j_real][i_real] = (1.0/3.0)*u[2][j_real][i_real] + (2.0/3.0)*u2[2][j_real][i_real] + (2.0/3.0)*dt*udot2[2][jj][ii];
                u_p[3][j_real][i_real] = (1.0/3.0)*u[3][j_real][i_real] + (2.0/3.0)*u2[3][j_real][i_real] + (2.0/3.0)*dt*udot2[3][jj][ii];

            }

        }
        
        config_ghosts2D(u_p, xphysical_grid, ygrid_size, false);

        prims = cons2prim2D(u_p, lorentz_gamma);
        lorentz_gamma = calc_lorentz_gamma(prims[2], prims[3]);
        
        /**
        cout << " " << endl;
        cout << "UP: " << endl;
        for (int jj=0; jj <ygrid_size; jj++){
            for (int ii=0; ii < xgrid_size; ii++){
                cout << u_p[1][jj][ii]/1.e-10 << ", ";
            }
            cout << endl;
        }
        string b;
        cin >> b;
        */
        
        
        if (t > 0){
            dt = adapt_dt(prims);
        }
        
        

        // Swap the arrays
        u.swap(u_p);
        
        t += dt;

        

    }

    prims = cons2prim2D(u, lorentz_gamma);

    return prims;

 };


 //==================================
 //             2 SOURCE TERMS
 //==================================
  vector<vector<vector<double> > > UstateSR2D::simulate2D(vector<vector<double> > &lorentz_gamma, 
                                                        vector<vector<double> > &source1,
                                                        vector<vector<double> > &source2, 
                                                        float tend = 0.1, bool periodic = false, 
                                                        double dt = 1.e-4, bool linspace=true){

    // Define the swap vector for the integrated state
    int xgrid_size = state2D[0][0].size();
    int ygrid_size = state2D[0].size();
    int n_vars = state2D.size();
    int xphysical_grid = xgrid_size - 4;
    int yphysical_grid = ygrid_size - 4;
    float t = 0;

    
    vector<vector<vector<double> > > u_p(n_vars, vector<vector<double> > 
                                                (ygrid_size, vector<double> (xgrid_size, 0)));

    vector<vector<vector<double> > > u(n_vars, vector<vector<double> > 
                                                (ygrid_size, vector<double> (xgrid_size, 0)));
    
    vector<vector<vector<double> > > udot1(n_vars, vector<vector<double> > 
                                                (ygrid_size, vector<double> (xgrid_size, 0)));

    vector<vector<vector<double> > > udot2(n_vars, vector<vector<double> > 
                                                (ygrid_size, vector<double> (xgrid_size, 0)));

    vector<vector<vector<double> > > udot(n_vars, vector<vector<double> > 
                                                (ygrid_size, vector<double> (xgrid_size, 0)));

    vector<vector<vector<double> > > u1(n_vars, vector<vector<double> > 
                                                (ygrid_size, vector<double> (xgrid_size, 0)));

    vector<vector<vector<double> > > u2(n_vars, vector<vector<double> > 
                                                (ygrid_size, vector<double> (xgrid_size, 0)));

    vector<vector<vector<double> > > prims(n_vars, vector<vector<double> > 
                                                (ygrid_size, vector<double> (xgrid_size, 0)));

    // Copy the state array into real & profile variables
    u = state2D;
    u_p = u;
    u1 = u; 
    u2 = u;

    cout << "I Am Here: " << endl;
    
    while (t < tend){

        if (t == 0){
            config_ghosts2D(u, xphysical_grid, ygrid_size, false);
        }

        // cout << " " << endl;
        // cout << "U: " << endl;
        // for (int jj=0; jj <ygrid_size; jj++){
        //     for (int ii=0; ii < xgrid_size; ii++){
        //         cout << u[1][jj][ii] << ", ";
        //     }
        //     cout << endl;
        // }
        // string a;
        // cin >> a;
        

        udot = u_dot2D(u, lorentz_gamma, source1, source2, periodic, theta, linspace);


        for (int jj = 0; jj < yphysical_grid; jj++){
            // Get the non-ghost index 
            int j_real = jj + 2;
            for (int ii = 0; ii < xphysical_grid; ii++){
                int i_real = ii + 2; 
                u1[0][j_real][i_real] = u[0][j_real][i_real] + dt*udot[0][jj][ii];
                u1[1][j_real][i_real] = u[1][j_real][i_real] + dt*udot[1][jj][ii];
                u1[2][j_real][i_real] = u[2][j_real][i_real] + dt*udot[2][jj][ii];
                u1[3][j_real][i_real] = u[3][j_real][i_real] + dt*udot[3][jj][ii];
            }
        }
        
        
        cout << " " << endl;
        cout << "U1: " << endl;
        for (int jj=0; jj <ygrid_size; jj++){
            for (int ii=0; ii < xgrid_size; ii++){
                cout << u1[1][jj][ii] << ", ";
            }
            cout << endl;
        }
        string d;
        cin >> d;
        
        

        
        config_ghosts2D(u1, xphysical_grid, ygrid_size, false);
        

        prims = cons2prim2D(u1, lorentz_gamma);

        lorentz_gamma = calc_lorentz_gamma(prims[2], prims[3]);

        udot1 = u_dot2D(u1, lorentz_gamma, source1, source2, periodic, theta, linspace);

        
        // cout << " " << endl;
        // cout << "UDot1: " << endl;
        // for (int jj=0; jj <yphysical_grid; jj++){
        //     for (int ii=0; ii < xphysical_grid; ii++){
        //         cout << prims[3][jj][ii] << ", ";
        //         //cout << udot1[0][jj][ii] << ", ";
        //     }
        //     cout << endl;
        // }
        // string p;
        // cin >> p;
        
        cout << t << endl;
        for (int jj = 0; jj < yphysical_grid; jj++){
            // Get the non-ghost index 
            int j_real =  jj + 2;
            for (int ii = 0; ii < xphysical_grid; ii++){
                int i_real = ii + 2;
                u2[0][j_real][i_real] = 0.75*u[0][j_real][i_real] + 0.25*u1[0][j_real][i_real] + 0.25*dt*udot1[0][jj][ii];
                u2[1][j_real][i_real] = 0.75*u[1][j_real][i_real] + 0.25*u1[1][j_real][i_real] + 0.25*dt*udot1[1][jj][ii];
                u2[2][j_real][i_real] = 0.75*u[2][j_real][i_real] + 0.25*u1[2][j_real][i_real] + 0.25*dt*udot1[2][jj][ii];
                u2[3][j_real][i_real] = 0.75*u[3][j_real][i_real] + 0.25*u1[3][j_real][i_real] + 0.25*dt*udot1[3][jj][ii];

            }
            

        }

        /**
        cout << " " << endl;
        cout << "U2: " << endl;
        for (int jj=0; jj <ygrid_size; jj++){
            for (int ii=0; ii < xgrid_size; ii++){
                cout << u2[0][jj][ii] << ", ";
            }
            cout << endl;
        }
        string c;
        cin >> c;
        */
        
       
        config_ghosts2D(u2, xphysical_grid, ygrid_size, false);

        prims = cons2prim2D(u2, lorentz_gamma);
        lorentz_gamma = calc_lorentz_gamma(prims[2], prims[3]);

        udot2 = u_dot2D(u2, lorentz_gamma, source1, source2, periodic, theta, linspace);

        for (int jj = 0; jj < yphysical_grid; jj++){
            // Get the non-ghost index 
            int j_real =  jj + 2;
            for (int ii = 0; ii < xphysical_grid; ii++){
                int i_real = ii + 2;
                u_p[0][j_real][i_real] = (1.0/3.0)*u[0][j_real][i_real] + (2.0/3.0)*u2[0][j_real][i_real] + (2.0/3.0)*dt*udot2[0][jj][ii];
                u_p[1][j_real][i_real] = (1.0/3.0)*u[1][j_real][i_real] + (2.0/3.0)*u2[1][j_real][i_real] + (2.0/3.0)*dt*udot2[1][jj][ii];
                u_p[2][j_real][i_real] = (1.0/3.0)*u[2][j_real][i_real] + (2.0/3.0)*u2[2][j_real][i_real] + (2.0/3.0)*dt*udot2[2][jj][ii];
                u_p[3][j_real][i_real] = (1.0/3.0)*u[3][j_real][i_real] + (2.0/3.0)*u2[3][j_real][i_real] + (2.0/3.0)*dt*udot2[3][jj][ii];

            }

        }
        
        config_ghosts2D(u_p, xphysical_grid, ygrid_size, false);

        prims = cons2prim2D(u_p, lorentz_gamma);
        lorentz_gamma = calc_lorentz_gamma(prims[2], prims[3]);
        
        /**
        cout << " " << endl;
        cout << "UP: " << endl;
        for (int jj=0; jj <ygrid_size; jj++){
            for (int ii=0; ii < xgrid_size; ii++){
                cout << u_p[1][jj][ii]/1.e-10 << ", ";
            }
            cout << endl;
        }
        string b;
        cin >> b;
        */
        
        
        if (t > 0){
            dt = adapt_dt(prims);
        }
        
        

        // Swap the arrays
        u.swap(u_p);
        
        t += dt;
        

    }

    prims = cons2prim2D(u, lorentz_gamma);

    return prims;

 };


 //==================================================
 //                 3 SOURCE TERMS
 //==================================================
 vector<vector<vector<double> > > UstateSR2D::simulate2D(vector<vector<double> > &lorentz_gamma, 
                                                        vector<vector<double> > &source1,
                                                        vector<vector<double> > &source2,
                                                        vector<vector<double> > &source3, 
                                                        float tend = 0.1, bool periodic = false, 
                                                        double dt = 1.e-4, bool linspace=true){

    // Define the swap vector for the integrated state
    int xgrid_size = state2D[0][0].size();
    int ygrid_size = state2D[0].size();
    int n_vars = state2D.size();
    int xphysical_grid = xgrid_size - 4;
    int yphysical_grid = ygrid_size - 4;
    float t = 0;

    
    vector<vector<vector<double> > > u_p(n_vars, vector<vector<double> > 
                                                (ygrid_size, vector<double> (xgrid_size, 0)));

    vector<vector<vector<double> > > u(n_vars, vector<vector<double> > 
                                                (ygrid_size, vector<double> (xgrid_size, 0)));
    
    vector<vector<vector<double> > > udot1(n_vars, vector<vector<double> > 
                                                (ygrid_size, vector<double> (xgrid_size, 0)));

    vector<vector<vector<double> > > udot2(n_vars, vector<vector<double> > 
                                                (ygrid_size, vector<double> (xgrid_size, 0)));

    vector<vector<vector<double> > > udot(n_vars, vector<vector<double> > 
                                                (ygrid_size, vector<double> (xgrid_size, 0)));

    vector<vector<vector<double> > > u1(n_vars, vector<vector<double> > 
                                                (ygrid_size, vector<double> (xgrid_size, 0)));

    vector<vector<vector<double> > > u2(n_vars, vector<vector<double> > 
                                                (ygrid_size, vector<double> (xgrid_size, 0)));

    vector<vector<vector<double> > > prims(n_vars, vector<vector<double> > 
                                                (ygrid_size, vector<double> (xgrid_size, 0)));

    // Copy the state array into real & profile variables
    u = state2D;
    u_p = u;
    u1 = u; 
    u2 = u;
    
    while (t < tend){

        if (t == 0){
            config_ghosts2D(u, xphysical_grid, ygrid_size, false);
        }

        /**
        cout << " " << endl;
        cout << "U: " << endl;
        for (int jj=0; jj <ygrid_size; jj++){
            for (int ii=0; ii < xgrid_size; ii++){
                cout << u[1][jj][ii] << ", ";
            }
            cout << endl;
        }
        cin.get();
        */
        

        udot = u_dot2D(u, lorentz_gamma, source1, source2, source3, periodic, theta, linspace);


        for (int jj = 0; jj < yphysical_grid; jj++){
            // Get the non-ghost index 
            int j_real = jj + 2;
            for (int ii = 0; ii < xphysical_grid; ii++){
                int i_real = ii + 2; 
                u1[0][j_real][i_real] = u[0][j_real][i_real] + dt*udot[0][jj][ii];
                u1[1][j_real][i_real] = u[1][j_real][i_real] + dt*udot[1][jj][ii];
                u1[2][j_real][i_real] = u[2][j_real][i_real] + dt*udot[2][jj][ii];
                u1[3][j_real][i_real] = u[3][j_real][i_real] + dt*udot[3][jj][ii];
            }
        }
        
        
        
        /**
        cout << " " << endl;
        cout << "U1: " << endl;
        for (int jj=0; jj <ygrid_size; jj++){
            for (int ii=0; ii < xgrid_size; ii++){
                cout << u1[1][jj][ii] << ", ";
            }
            cout << endl;
        }
        cin.get();
        */
        
        config_ghosts2D(u1, xphysical_grid, ygrid_size, false);
        

        prims = cons2prim2D(u1, lorentz_gamma);

        lorentz_gamma = calc_lorentz_gamma(prims[2], prims[3]);

        udot1 = u_dot2D(u1, lorentz_gamma, source1, source2, source3, periodic, theta, linspace);

        
        // cout << " " << endl;
        // cout << "UDot1: " << endl;
        // for (int jj=0; jj <yphysical_grid; jj++){
        //     for (int ii=0; ii < xphysical_grid; ii++){
        //         cout << prims[3][jj][ii] << ", ";
        //         //cout << udot1[0][jj][ii] << ", ";
        //     }
        //     cout << endl;
        // }
        // string p;
        // cin >> p;
        
        if (t > 0){
            dt = adapt_dt(prims);
        }
        
        for (int jj = 0; jj < yphysical_grid; jj++){
            // Get the non-ghost index 
            int j_real =  jj + 2;
            for (int ii = 0; ii < xphysical_grid; ii++){
                int i_real = ii + 2;
                u2[0][j_real][i_real] = 0.75*u[0][j_real][i_real] + 0.25*u1[0][j_real][i_real] + 0.25*dt*udot1[0][jj][ii];
                u2[1][j_real][i_real] = 0.75*u[1][j_real][i_real] + 0.25*u1[1][j_real][i_real] + 0.25*dt*udot1[1][jj][ii];
                u2[2][j_real][i_real] = 0.75*u[2][j_real][i_real] + 0.25*u1[2][j_real][i_real] + 0.25*dt*udot1[2][jj][ii];
                u2[3][j_real][i_real] = 0.75*u[3][j_real][i_real] + 0.25*u1[3][j_real][i_real] + 0.25*dt*udot1[3][jj][ii];

            }
            

        }


        /**
        cout << " " << endl;
        cout << "U2: " << endl;
        for (int jj=0; jj <ygrid_size; jj++){
            for (int ii=0; ii < xgrid_size; ii++){
                cout << u2[0][jj][ii] << ", ";
            }
            cout << endl;
        }
        string c;
        cin >> c;
        */
        
       
        config_ghosts2D(u2, xphysical_grid, ygrid_size, false);

        prims = cons2prim2D(u2, lorentz_gamma);
        lorentz_gamma = calc_lorentz_gamma(prims[2], prims[3]);

        udot2 = u_dot2D(u2, lorentz_gamma, source1, source2, source3, periodic, theta, linspace);

        if (t > 0){
            dt = adapt_dt(prims);
        }

        for (int jj = 0; jj < yphysical_grid; jj++){
            // Get the non-ghost index 
            int j_real =  jj + 2;
            for (int ii = 0; ii < xphysical_grid; ii++){
                int i_real = ii + 2;
                u_p[0][j_real][i_real] = (1.0/3.0)*u[0][j_real][i_real] + (2.0/3.0)*u2[0][j_real][i_real] + (2.0/3.0)*dt*udot2[0][jj][ii];
                u_p[1][j_real][i_real] = (1.0/3.0)*u[1][j_real][i_real] + (2.0/3.0)*u2[1][j_real][i_real] + (2.0/3.0)*dt*udot2[1][jj][ii];
                u_p[2][j_real][i_real] = (1.0/3.0)*u[2][j_real][i_real] + (2.0/3.0)*u2[2][j_real][i_real] + (2.0/3.0)*dt*udot2[2][jj][ii];
                u_p[3][j_real][i_real] = (1.0/3.0)*u[3][j_real][i_real] + (2.0/3.0)*u2[3][j_real][i_real] + (2.0/3.0)*dt*udot2[3][jj][ii];

            }

        }
        
        config_ghosts2D(u_p, xphysical_grid, ygrid_size, false);

        prims = cons2prim2D(u_p, lorentz_gamma);
        lorentz_gamma = calc_lorentz_gamma(prims[2], prims[3]);

        /**
        cout << " " << endl;
        cout << "UP: " << endl;
        for (int jj=0; jj <ygrid_size; jj++){
            for (int ii=0; ii < xgrid_size; ii++){
                cout << u_p[1][jj][ii]/1.e-10 << ", ";
            }
            cout << endl;
        }
        string b;
        cin >> b;
        */
        
        
        if (t > 0){
            dt = adapt_dt(prims);
        }
        
        

        // Swap the arrays
        u.swap(u_p);
        
        t += dt;
        

    }

    prims = cons2prim2D(u, lorentz_gamma);

    return prims;

 };