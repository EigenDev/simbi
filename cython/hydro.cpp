/* 
* C++ Library to perform extensive hydro calculations
* to be later wrapped and plotted in Python
* Marcus DuPont
* New York University
* 07/15/2020
* Compressible Hydro Simulation
*/

#include "ustate.h" 
#include <cmath>
#include <map>
#include <algorithm>

using namespace std;
using namespace states;


// Default Constructor 
Ustate::Ustate () {}

// Overloaded Constructor
Ustate::Ustate(vector< vector<double> > u_state, float Gamma, vector<double> R = {0},
                string Coord_system = "cartesian")
{
    state = u_state;
    gamma = Gamma;
    r = R;
    coord_system = Coord_system;

}

// Destructor 
Ustate::~Ustate() {}

// Default Constructor 
Ustate2D::Ustate2D () {}

// Overloaded Constructor
Ustate2D::Ustate2D(vector<vector< vector<double> > > u_state2D, float Gamma)
{
    state2D = u_state2D;
    gamma = Gamma;
}

// Destructor 
Ustate2D::~Ustate2D() {}

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

// Sound Speed Function
double calc_sound_speed(float gamma, double rho, double pressure){
    double c = sqrt(gamma*pressure/rho);
    return c;

};

// Get the Sign of a Number
double sign(double x){
    if (-x/abs(x) == -1) { 
        return 1;
    } else if (-x/abs(x) == 1){
        return 1;
    } else {
        return 0;
    }
};

// The Minmod slope delimiter
double minmod(double x, double y, double z){
    // The intermediate minimum
    double intermin = min(abs(x), abs(y));
    double min_val = min(intermin, abs(z)); 

    return 0.25*abs(sign(x) + sign(y))*(sign(x) + sign(z))*min_val;

};

// Roll a vector for use with periodic boundary conditions
vector<double> rollVector(const vector<double>& v, unsigned int n){
    auto b = v.begin() + (n % v.size());
    vector<double> ret(b, v.end());
    ret.insert(ret.end(), v.begin(), b);
    return ret;
};

// Roll a single vector index
double roll(vector<double>& v, unsigned int n) {
   return v[n % v.size()];
};

// Compute the minmod flux limiter
vector<vector<double> > minmodLimiter(vector<vector<double> > prims, bool periodic = false){
    vector<vector<double> > left_most, left_mid, center, right_mid, right_most;

    return left_mid;

};

vector<vector<double> > tranpose(vector<vector<double> > &mat){

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

// void config_ghosts1D(vector<vector<double> >&, int grid_size);

void config_ghosts1D(vector<vector<double> > &u_state, int grid_size,bool first_order=true){
    if (first_order){
        u_state[0][0] = u_state[0][1];
        u_state[0][grid_size - 1] = u_state[0][grid_size - 2];

        u_state[1][0] = - u_state[1][1];
        u_state[1][grid_size - 1] = - u_state[1][grid_size - 2];

        u_state[2][0] = u_state[2][1];
        u_state[2][grid_size - 1] = u_state[2][grid_size - 2];
    } else {
        u_state[0][0] = u_state[0][2];
        u_state[0][1] = u_state[0][2];
        u_state[0][grid_size - 1] = u_state[0][grid_size - 3];
        u_state[0][grid_size - 2] = u_state[0][grid_size - 3];

        u_state[1][0] = - u_state[1][2];
        u_state[1][1] = - u_state[1][2];
        u_state[1][grid_size - 1] = - u_state[1][grid_size - 3];
        u_state[1][grid_size - 2] = - u_state[1][grid_size - 3];

        u_state[2][0] = u_state[2][2];
        u_state[2][1] = u_state[2][2];
        u_state[2][grid_size - 1] = u_state[2][grid_size - 3];
        u_state[2][grid_size - 2] = u_state[2][grid_size - 3];

    }
};


//----------------------------------------------------------------------------------------------------------
//  PRESSURE CALCULATIONS
//---------------------------------------------------------------------------------------------------------

double calc_pressure(float gamma, double rho, double energy, double v){
    double pressure = (gamma - 1.)*(energy - 0.5*rho*pow(v,2));
    return pressure;
};

//------------------------------------------------------------------------------------------------------------
//  ENERGY CALCULATIONS
//------------------------------------------------------------------------------------------------------------

double calc_energy(float gamma, double rho, double pressure, double v){
        return pressure/(gamma-1.) + 0.5*rho*pow(v, 2);
};

//--------------------------------------------------------------------------------------------------
//                          GET THE PRIMITIVE VECTORS
//--------------------------------------------------------------------------------------------------

// Return a 1D array containing (rho, pressure, v) at a *single grid point*
vector<double>  cons2prim(float gamma, vector<double>  u_state, bool twoD=false){
    /**
     * Return a vector containing the primitive
     * variables density (rho), pressure, and
     * velocity (v)
     */

    if (twoD == false){
        double rho, energy, mom;
        vector<double> prims(3);
        double v, pressure;

        rho = u_state[0];
        mom = u_state[1];
        energy = u_state[2];

        v = mom/rho;
        pressure = calc_pressure(gamma, rho, energy, v);
        
        prims[0] = rho;
        prims[1] = pressure;
        prims[2] = v;

        return prims;
        
    } else {
        double rho, energy, momx, momy;
        vector<double> prims(4);
        double vx, vy, vtot, pressure;

        rho = u_state[0];
        momx = u_state[1];
        momy = u_state[2];
        energy = u_state[3];

        vx = momx/rho;
        vy = momy/rho;

        vtot = sqrt(pow(vx, 2) + pow(vy, 2));

        pressure = calc_pressure(gamma, rho, energy, vtot);
        
        prims[0] = rho;
        prims[1] = pressure;
        prims[2] = vx;
        prims[3] = vy;

        return prims;

    }
    
};

vector < vector<double> > Ustate::cons2prim1D(vector<vector<double> > u_state){
    /**
     * Return a vector containing the primitive
     * variables density (rho), pressure, and
     * velocity (v)
     */

    double rho, energy, mom;
    double v, pressure;
    int n_vars = u_state.size();
    int n_gridpts = u_state[0].size();

    vector < vector<double> > prims(n_vars, vector<double> (n_gridpts, 0));
    
    
    for (int ii=0; ii < n_gridpts; ii++){

        rho = u_state[0][ii];
        mom = u_state[1][ii];
        energy = u_state[2][ii];

        v = mom/rho;
        pressure = calc_pressure(gamma, rho, energy, v);
        
        prims[0][ii] = rho;
        prims[1][ii] = pressure;
        prims[2][ii] = v;
    }

    return prims;
};

vector<vector< vector<double> > > Ustate2D::cons2prim2D(vector<vector< vector<double> > > u_state2D){
    /**
     * Return a 2D matrix containing the primitive
     * variables density (rho), pressure, and
     * velocity (v)
     */

    double rho, energy, momx, momy;
    double vx,vy, pressure, vtot;
     
    int n_vars = u_state2D.size();
    int ny_gridpts = u_state2D[0].size();
    int nx_gridpts = u_state2D[0][0].size();

    vector<vector<vector<double> > > prims(n_vars, vector<vector<double> > 
                                            (ny_gridpts, vector<double>(nx_gridpts)));
   

    for (int jj=0; jj < ny_gridpts; jj++){
        for(int ii=0; ii< nx_gridpts; ii++){
            rho = u_state2D[0][jj][ii];       // Density
            momx = u_state2D[1][jj][ii];      // X-Momentum
            momy = u_state2D[2][jj][ii];      // Y-Momentum
            energy = u_state2D[3][jj][ii];    // Energy

            vx = momx/rho;
            vy = momy/rho;

            vtot = sqrt( pow(vx, 2) + pow(vy, 2) );

            pressure = calc_pressure(gamma, rho, energy, vtot);

            
            prims[0][jj][ii] = rho;
            prims[1][jj][ii] = pressure;
            prims[2][jj][ii] = vx;
            prims[3][jj][ii] = vy;
            

        }
    }
    

    return prims;
};



//----------------------------------------------------------------------------------------------------
//              STATE TENSOR CALCULATIONS
//----------------------------------------------------------------------------------------------------


// Get the (3,1) state tensor for computation. Used for Higher Order Reconstruction
vector<double>  calc_state1D(float gamma, double rho, double pressure, double v)
{
    // int n_vars = 3;
    vector<double> cons_state(3);

    // cons_state.resize(n_vars, vector<double> (1, -1));
    
    double energy = calc_energy(gamma, rho, pressure, v);
    
    cons_state[0] = rho; 
    cons_state[1] = rho*v;
    cons_state[2] = energy;

    return cons_state;
};

// Get the 2-Dimensional (4, 1) state tensor for computation. 
// It is being doing pointwise in this case as opposed to over
// the entire array since we are in c++
 vector<double>  calc_state2D(float gamma, double rho, double pressure, double vx, double vy)
{
    double vtot = sqrt( pow(vx,2) + pow(vy, 2));
    vector<double>  cons_state(4);
    double energy = calc_energy(gamma, rho, pressure, vtot);
    
    
    
    cons_state[0] = rho; 
    cons_state[1] = rho*vx;
    cons_state[2] = rho*vy;
    cons_state[3] = energy;
        
        
    return cons_state;
};


//----------------------------------------------------------------------------------------------------------
//                              EIGENVALUE CALCULATIONS
//----------------------------------------------------------------------------------------------------------


map<string, map<string, double > > calc_eigenvals(float gamma, vector<double> left_state,
                                            vector<double> right_state, int ndim = 1, int quadL = 1, 
                                            int quadR = 1)
{
    if (ndim == 1){
        // Initialize your important variables
        double v_r, v_l, p_r, p_l, cs_r, cs_l; 
        double rho_l, rho_r,  mom_l, mom_r, energy_l, energy_r;
        map<string, map<string, double > > lambda;
        

        // Separate the left and right state components
        rho_l = left_state[0];
        mom_l = left_state[1];
        energy_l = left_state[2];

        rho_r = right_state[0];
        mom_r = right_state[1];
        energy_r = right_state[2];

        // Initialize and populate the dictionary vectors
        
        v_l = mom_l/rho_l;
        v_r = mom_r/rho_r;
        
        p_r = calc_pressure(gamma, rho_r, energy_r, v_r);
        p_l = calc_pressure(gamma, rho_l, energy_l ,v_l);

        cs_r = calc_sound_speed(gamma, rho_r, p_r);
        cs_l = calc_sound_speed(gamma, rho_l, p_l);

        // Populate the Dictionary
        lambda["left"]["plus"] = v_l + cs_l; 
        lambda["left"]["minus"] = v_l - cs_l; 
        lambda["right"]["plus"] = v_r + cs_r; 
        lambda["right"]["minus"] = v_r - cs_r; 

        return lambda;

    } else {

    // Initialize your important variables
    double vx_r, vx_l, vy_l, vy_r, p_r, p_l, cs_r, cs_l, vtot_l, vtot_r; 
    double rho_l, rho_r,  momx_l, momx_r, momy_l, momy_r, energy_l, energy_r;
    long vl, vr;
    map<string, map<string, double > > lambda;

    // Separate the left and right state components 
    rho_l = left_state[0];
    momx_l = left_state[1];
    momy_l = left_state[2];
    energy_l = left_state[3];

    rho_r = right_state[0];
    momx_r = right_state[1];
    momy_r = right_state[2];
    energy_r = right_state[3];

    vx_l = momx_l/rho_l;
    vx_r = momx_r/rho_r;

    vy_l = momy_l/rho_l;
    vy_r = momy_r/rho_r;
        
    vtot_l = sqrt(pow(vx_l, 2) + pow(vy_l, 2) );
    vtot_r = sqrt(pow(vx_r, 2) + pow(vy_r, 2) );

    p_r = calc_pressure(gamma, rho_r, energy_r, vtot_r);
    p_l = calc_pressure(gamma, rho_l, energy_l, vtot_l);

    if ((sign(p_r) == -1) || (sign(p_l) == -1.) ){
        cout << "I Have Broken: " << endl;
        cout << "Energy: " << energy_r << endl;
        cout << "Kinetic E: " << 0.5*rho_r*pow(vtot_r, 2) << endl;
        cout << "Vx: " << vx_r << endl;
        cout << "Vy: " << vy_r << endl;
        cout << "Vtot: " << vtot_r << endl;
        cout << "Momx: " << momx_r << endl; 
        cout << "Rho: " << rho_r << endl;
        string a;
        cin >> a;
    }

    cs_r = calc_sound_speed(gamma, rho_r, p_r);
    cs_l = calc_sound_speed(gamma, rho_l, p_l);

    // Check the quadrant we are in to see if the fluid is 
    // flowing inward or outward and apply the necessary sign to 
    // the total velocity vector

    //Check the quadrant of the left state
    if (quadL == 1){
        if ((sign(vx_l) == -1) && (sign(vy_l) == -1) ){
            vtot_l *= -1.;
        }
    } else if (quadL == 2){
        if ((sign(vx_l) == 1) && (sign(vy_l) == -1)){
            vtot_l *= -1.;
        }
    } else if (quadL == 3){
        if ((sign(vx_l) == 1) && (sign(vy_l) == 1)){
            vtot_l *= -1.;
        }
    } else if (quadL == 4){
        if ((sign(vx_l) == -1) && (sign(vy_l) ==0)){
            vtot_l *= -1.;
        }
    } else if (quadL == 0.5) {
        if ((sign(vx_l) == -1) && (sign(vy_l) == 0)){
            vtot_l *= -1.;
        }
    } else if (quadL == 1.5) {
        if ((sign(vx_l) == 0) && (sign(vy_l) == -1)){
            vtot_l *= -1.;
        }
    } else if (quadL == 2.5) {
        if ((sign(vx_l) == 1) && (sign(vy_l) == 0)){
            vtot_l *= -1.;
        }
    } else if (quadL == 3.5) {
        if ((sign(vx_l) == 0) && (sign(vy_l) == 1)){
            vtot_l *= -1.;
        }
    }

    // Check the quadrant of the right state
    if (quadR == 1){
        if ((sign(vx_r) == -1) && (sign(vy_r) == -1) ){
            vtot_r *= -1.;
        }
    } else if (quadR == 2){
        if ((sign(vx_r) == 1) && (sign(vy_r) == -1 )){
            vtot_r *= -1.;
        }
    } else if (quadR == 3){
        if ((sign(vx_r) == 1) && (sign(vy_r) == 1)){
            vtot_r *= -1.;
        }
    } else if (quadR == 4){
        if ((sign(vx_r) == -1) && (sign(vy_r) == 0)){
            vtot_r *= -1.;
        }
    } else if (quadR == 0.5) {
        if ((sign(vx_r) == -1) && (sign(vy_r) == 0)){
            vtot_r *= -1.;
        }
    } else if (quadR == 1.5) {
        if ((sign(vx_r) == 0) && (sign(vy_r) == -1)){
            vtot_r *= -1.;
        }
    } else if (quadR == 2.5) {
        if ((sign(vx_r) == 1) && (sign(vy_r) == 0)){
            vtot_r *= -1.;
        }
    } else if (quadR == 3.5) {
        if ((sign(vx_r) == 0) && (sign(vy_r) == 1)){
            vtot_r *= -1.;
        }
    }
    vl = vx_l + vy_l;
    vr = vx_r + vy_r;
    // Populate the lambda dictionary
    lambda["left"]["plus"] = vl + cs_l; 
    lambda["left"]["minus"] = vl - cs_l; 
    lambda["right"]["plus"] = vr + cs_r; 
    lambda["right"]["minus"] = vr - cs_r; 

    return lambda;

    }
    
};



//-----------------------------------------------------------------------------------------------------------
//                                            FLUX CALCULATIONS
//-----------------------------------------------------------------------------------------------------------

// Get the 1D Flux array (3,1)
vector<double> calc_flux1D(float gamma, double rho, double pressure, double v){
    
    // The Flux Tensor
    vector<double> flux(3);

    // The Flux components
    double mom, energy_dens, zeta;
    

    // flux.resize(n_vars, vector<double> (grid_size, -1));
    double energy = calc_energy(gamma, rho, pressure, v);
    
    mom = rho*v;
    energy_dens = rho*pow(v, 2) + pressure;
    zeta = (energy + pressure)*v;
    flux[0] = mom;
    flux[1] = energy_dens;
    flux[2] = zeta;

    return flux;
};

// Get the 2D Flux array (4,1). Either return F or G depending on directional flag
vector<double> calc_flux2D(float gamma, double rho, double pressure, 
                                        double vx, double vy, bool x_direction=true){
    
    // The Flux Tensor
    vector<double> flux(4);

     // The Flux components
    double momx, momy, convect_xy, energy_dens, zeta;

    double vtot = sqrt(pow(vx, 2) + pow(vy, 2) );
    double energy = calc_energy(gamma,rho, pressure, vtot);

    


    // Check if we're calculating the x-direction flux. If not, calculate the y-direction
    if (x_direction){
        momx = rho*vx;
        convect_xy = rho*vx*vy;
        energy_dens = rho*pow(vx, 2) + pressure;
        zeta = (energy + pressure)*vx;

        flux[0] = momx;
        flux[1] = energy_dens;
        flux[2] = convect_xy;
        flux[3] = zeta;
           
        return flux;
    } else {
        momy = rho*vy;
        convect_xy = rho*vx*vy;
        energy_dens = rho*pow(vy, 2) + pressure;
        zeta = (energy + pressure)*vy;

        flux[0] = momy;
        flux[1] = convect_xy;
        flux[2] = energy_dens;
        flux[3] = zeta;
           
        return flux;
    }
    
};

vector<double> calc_hll_flux1D(float gamma, vector<double> left_state,
                                        vector<double> right_state,
                                        vector<double> left_flux,
                                        vector<double> right_flux)
{
    map<string, map<string, double > > lambda; 
    vector<double> hll_flux(3);
    double alpha_plus, alpha_minus;  
    
    lambda = calc_eigenvals(gamma, left_state, right_state);

    // cout << "Lambda Size: " <<  lambda["left"]["plus"].size() << endl;
    // Crudely form the arrays to search for the maximums
    alpha_plus = findMax(0, lambda["left"]["plus"], lambda["right"]["plus"]);
    alpha_minus = findMax(0 , - lambda["left"]["minus"], - lambda["right"]["minus"]);

    // Compute the HLL Flux component-wise
    hll_flux[0] = ( ( alpha_plus*left_flux[0] + alpha_minus*right_flux[0]
                            - alpha_minus*alpha_plus*(right_state[0] - left_state[0] ) )  /
                            (alpha_minus + alpha_plus) );

    hll_flux[1] = ( alpha_plus*left_flux[1] + alpha_minus*right_flux[1]
                            - alpha_minus*alpha_plus*(right_state[1] - left_state[1] ) )  /
                            (alpha_minus + alpha_plus);

    hll_flux[2] = ( alpha_plus*left_flux[2] + alpha_minus*right_flux[2]
                            - alpha_minus*alpha_plus*(right_state[2] - left_state[2]) )  /
                            (alpha_minus + alpha_plus);


    return hll_flux;
};

vector<double> calc_hll_flux2D(float gamma, vector<double> left_state,
                                        vector<double> right_state,
                                        vector<double> left_flux,
                                        vector<double> right_flux, int quadL, int quadR)
{
    map<string, map<string, double > > lambda; 
    vector<double> hll_flux(4);
    double alpha_plus, alpha_minus;  
    
    lambda = calc_eigenvals(gamma, left_state, right_state, 2, quadL, quadR);

    // Calculate /pm alphas
    alpha_plus = findMax(0, lambda["left"]["plus"], lambda["right"]["plus"]);
    alpha_minus = findMax(0 , -lambda["left"]["minus"], -lambda["right"]["minus"]);

    // cout << "Alpha Plus: " << alpha_plus << endl;
    // cout << "Alpha Minus: " << alpha_minus << endl;

    // alpha_minus *= 7;
    // alpha_plus *= 7;
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

//----------------------------------------------------------------------------------------------------------
//                                  UDOT CALCULATIONS
//----------------------------------------------------------------------------------------------------------

vector<vector<double> > Ustate::u_dot1D(float gamma, vector<vector<double> > u_state, 
                                        bool first_order=true, bool periodic = false, float theta = 1.5,
                                        bool linspace = true){
    int i_start, i_bound, coordinate;
    int grid_size = u_state[0].size();
    int n_vars = u_state.size();
    double nx = u_state[0].size();
    string default_coordinates = "cartesian";
    
    double dx = 1/nx;
    vector<double> u_l(n_vars), u_r(n_vars), f_l(n_vars), f_r(n_vars);
    vector<double> prims_l(n_vars), prims_r(n_vars), f1(n_vars), f2(n_vars);
    
    if (first_order){
        int physical_grid = grid_size - 2;
        vector<vector<double> > L(n_vars, vector<double> (physical_grid, 0));

        if (periodic){
            i_start = 0;
            i_bound = grid_size;
        } else{
            int true_npts = grid_size - 1;
            i_start = 1;
            i_bound = true_npts;
        }

        if (coord_system == default_coordinates) {
            for (int ii= i_start; ii < i_bound; ii++){
                if (periodic){
                    coordinate = ii;
                    // Set up the left and right state interfaces for i+1/2
                    u_l[0] = u_state[0][ii];
                    u_l[1] = u_state[1][ii];
                    u_l[2] = u_state[2][ii];

                    u_r[0] = roll(u_state[0], ii + 1);
                    u_r[1] = roll(u_state[1], ii + 1);
                    u_r[2] = roll(u_state[2], ii + 1);

                } else {
                    // Shift the index for C++ [0] indexing
                    coordinate = ii - 1;
                    // Set up the left and right state interfaces for i+1/2
                    u_l[0] = u_state[0][ii];
                    u_l[1] = u_state[1][ii];
                    u_l[2] = u_state[2][ii];

                    

                    u_r[0] = u_state[0][ii + 1];
                    u_r[1] = u_state[1][ii + 1];
                    u_r[2] = u_state[2][ii + 1];

                }

                prims_l = cons2prim(gamma, u_l);
                prims_r = cons2prim(gamma, u_r);
                
                double rho_l = prims_l[0];
                double rho_r = prims_r[0];

                double p_l = prims_l[1];
                double p_r = prims_r[1];

                double v_l = prims_l[2];
                double v_r = prims_r[2];

                f_l = calc_flux1D(gamma, rho_l, p_l, v_l);
                f_r = calc_flux1D(gamma, rho_r, p_r, v_r);

                // Calc HLL Flux at i+1/2 interface
                f1 = calc_hll_flux1D(gamma, u_l, u_r, f_l, f_r);

                // Set up the left and right state interfaces for i-1/2
                if (periodic){
                    u_l[0] = roll(u_state[0], ii - 1);
                    u_l[1] = roll(u_state[1], ii - 1);
                    u_l[2] = roll(u_state[2], ii - 1);
                    
                    u_r[0] = u_state[0][ii];
                    u_r[1] = u_state[1][ii];
                    u_r[2] = u_state[2][ii];

                } else {
                    u_l[0] = u_state[0][ii - 1];
                    u_l[1] = u_state[1][ii - 1];
                    u_l[2] = u_state[2][ii - 1];
                    
                    u_r[0] = u_state[0][ii];
                    u_r[1] = u_state[1][ii];
                    u_r[2] = u_state[2][ii];

                }

                prims_l = cons2prim(gamma, u_l);
                prims_r = cons2prim(gamma, u_r);

                rho_l = prims_l[0];
                rho_r = prims_r[0];

                p_l = prims_l[1];
                p_r = prims_r[1];

                v_l = prims_l[2];
                v_r = prims_r[2];

                f_l = calc_flux1D(gamma, rho_l, p_l, v_l);
                f_r = calc_flux1D(gamma, rho_r, p_r, v_r);

                // Calc HLL Flux at i-1/2 interface
                f2 = calc_hll_flux1D(gamma, u_l, u_r, f_l, f_r);

                L[0][coordinate] = - (f1[0] - f2[0])/dx;
                L[1][coordinate] = - (f1[1] - f2[1])/dx;
                L[2][coordinate] = - (f1[2] - f2[2])/dx;
            }

            
        } 
        else {
            double r_left, r_right, volAvg, pc;
            double log_rLeft, log_rRight;

            double delta_logr = (log10(r[physical_grid - 1]) - log10(r[0]))/physical_grid;

            long double dr = (r[physical_grid - 1] - r[0])/physical_grid;

            for (int ii= i_start; ii < i_bound; ii++){
                if (periodic){
                    coordinate = ii;
                    // Set up the left and right state interfaces for i+1/2
                    u_l[0] = u_state[0][ii];
                    u_l[1] = u_state[1][ii];
                    u_l[2] = u_state[2][ii];

                    u_r[0] = roll(u_state[0], ii + 1);
                    u_r[1] = roll(u_state[1], ii + 1);
                    u_r[2] = roll(u_state[2], ii + 1);

                } else {
                    // Shift the index for C++ [0] indexing
                    coordinate = ii - 1;
                    // Set up the left and right state interfaces for i+1/2
                    u_l[0] = u_state[0][ii];
                    u_l[1] = u_state[1][ii];
                    u_l[2] = u_state[2][ii];

                    

                    u_r[0] = u_state[0][ii + 1];
                    u_r[1] = u_state[1][ii + 1];
                    u_r[2] = u_state[2][ii + 1];

                }

                prims_l = cons2prim(gamma, u_l);
                prims_r = cons2prim(gamma, u_r);
                
                double rho_l = prims_l[0];
                double rho_r = prims_r[0];

                double p_l = prims_l[1];
                double p_r = prims_r[1];

                double v_l = prims_l[2];
                double v_r = prims_r[2];

                f_l = calc_flux1D(gamma, rho_l, p_l, v_l);
                f_r = calc_flux1D(gamma, rho_r, p_r, v_r);

                // Calc HLL Flux at i+1/2 interface
                f1 = calc_hll_flux1D(gamma, u_l, u_r, f_l, f_r);

                // Get the central pressure
                pc = prims_l[1];

                // Set up the left and right state interfaces for i-1/2
                if (periodic){
                    u_l[0] = roll(u_state[0], ii - 1);
                    u_l[1] = roll(u_state[1], ii - 1);
                    u_l[2] = roll(u_state[2], ii - 1);
                    
                    u_r[0] = u_state[0][ii];
                    u_r[1] = u_state[1][ii];
                    u_r[2] = u_state[2][ii];

                } else {
                    u_l[0] = u_state[0][ii - 1];
                    u_l[1] = u_state[1][ii - 1];
                    u_l[2] = u_state[2][ii - 1];
                    
                    u_r[0] = u_state[0][ii];
                    u_r[1] = u_state[1][ii];
                    u_r[2] = u_state[2][ii];

                }

                prims_l = cons2prim(gamma, u_l);
                prims_r = cons2prim(gamma, u_r);

                rho_l = prims_l[0];
                rho_r = prims_r[0];

                p_l = prims_l[1];
                p_r = prims_r[1];

                v_l = prims_l[2];
                v_r = prims_r[2];

                f_l = calc_flux1D(gamma, rho_l, p_l, v_l);
                f_r = calc_flux1D(gamma, rho_r, p_r, v_r);

                // Calc HLL Flux at i-1/2 interface
                f2 = calc_hll_flux1D(gamma, u_l, u_r, f_l, f_r);

                if (linspace){
                    double right_cell = r[coordinate + 1];
                    double left_cell = r[coordinate - 1];

                    // Outflow the left/right boundaries
                    if (coordinate - 1 < 0){
                        left_cell = r[coordinate];
                    } else if (coordinate == physical_grid - 1){
                        right_cell = r[coordinate];
                    }

                    r_right = 0.5*(right_cell + r[coordinate]);
                    r_left = 0.5*(r[coordinate] + left_cell);

                } else {
                    log_rLeft = log10(r[0]) + coordinate*delta_logr;
                    log_rRight = log_rLeft + delta_logr;
                    r_left = pow(10, log_rLeft);
                    r_right = pow(10, log_rRight);
                }

                volAvg = 0.75*( ( pow(r_right, 4) - pow(r_left, 4) )/ ( pow(r_right, 3) - pow(r_left, 3) ) );
                

                L[0][coordinate] = - (pow(r_right,2)*f1[0] - pow(r_left,2)*f2[0] )/(pow(volAvg,2)*dr);
                L[1][coordinate] = - (pow(r_right,2)*f1[1] - pow(r_left,2)*f2[1] )/(pow(volAvg,2)*dr) + 2*pc/volAvg;
                L[2][coordinate] = - (pow(r_right,2)*f1[2] - pow(r_left,2)*f2[2] )/(pow(volAvg,2)*dr);
            }
        }

        
        return L;
    } else {
        int physical_grid = grid_size - 4;
        vector<vector<double> > L(n_vars, vector<double> (physical_grid, 0));
        
        // Calculate the primitives for the entire state
        vector<vector<double> > prims(n_vars, vector<double>(grid_size, 0));
        vector<double> left_most(n_vars), left_mid(n_vars), center(n_vars);
        vector<double> right_mid(n_vars), right_most(n_vars);


        prims = cons2prim1D(u_state);
        // The periodic BC doesn't require ghost cells. Shift the index
        // to the beginning since all of he.
        if (periodic){ 
            i_start = 0;
            i_bound = grid_size;
        } else {
            int true_npts = grid_size - 2;
            i_start = 2;
            i_bound = true_npts;
        }

        if (coord_system == default_coordinates){
            for (int ii = i_start; ii < i_bound; ii++){
                if (periodic){
                    coordinate = ii;
                    // Declare the c[i-2],c[i-1],c_i,c[i+1], c[i+2] variables
                    left_most[0] = roll(prims[0], ii - 2);
                    left_mid[0] = roll(prims[0], ii - 1);
                    center[0] = prims[0][ii];
                    right_mid[0] = roll(prims[0], ii + 1);
                    right_most[0] = roll(prims[0], ii + 2);

                    left_most[1] = roll(prims[1], ii - 2);
                    left_mid[1] = roll(prims[1], ii - 1);
                    center[1] = prims[1][ii];
                    right_mid[1] = roll(prims[1], ii + 1);
                    right_most[1] = roll(prims[1], ii + 2);

                    left_most[2] = roll(prims[2], ii - 2);
                    left_mid[2] = roll(prims[2], ii - 1);
                    center[2] = prims[2][ii];
                    right_mid[2] = roll(prims[2], ii + 1);
                    right_most[2] = roll(prims[2], ii + 2);

                } else {
                    // Adjust for beginning input of L vector
                    coordinate = ii - 2;
                    left_most[0] = prims[0][ii - 2];
                    left_mid[0] = prims[0][ii - 1];
                    center[0] = prims[0][ii];
                    right_mid[0] = prims[0][ii + 1];
                    right_most[0] = prims[0][ii + 2];

                    left_most[1] = prims[1][ii - 2];
                    left_mid[1] = prims[1][ii - 1];
                    center[1] = prims[1][ii];
                    right_mid[1] = prims[1][ii + 1];
                    right_most[1] = prims[1][ii + 2];

                    left_most[2] = prims[2][ii - 2];
                    left_mid[2] = prims[2][ii - 1];
                    center[2] = prims[2][ii];
                    right_mid[2] = prims[2][ii + 1];
                    right_most[2] = prims[2][ii + 2];

                }

                // Compute the reconstructed primitives at the i+1/2 interface

                // Reconstructed left primitives vector
                prims_l[0] = center[0] + 0.5*minmod(theta*(center[0] - left_mid[0]),
                                                    0.5*(right_mid[0] - left_mid[0]),
                                                    theta*(right_mid[0] - center[0]));

                prims_l[1] = center[1] + 0.5*minmod(theta*(center[1] - left_mid[1]),
                                                    0.5*(right_mid[1] - left_mid[1]),
                                                    theta*(right_mid[1] - center[1]));

                prims_l[2] = center[2] + 0.5*minmod(theta*(center[2] - left_mid[2]),
                                                    0.5*(right_mid[2] - left_mid[2]),
                                                    theta*(right_mid[2] - center[2]));

                // Reconstructed right primitives vector
                prims_r[0] = right_mid[0] - 0.5*minmod(theta*(right_mid[0] - center[0]),
                                                    0.5*(right_most[0] - center[0]),
                                                    theta*(right_most[0] - right_mid[0]));

                prims_r[1] = right_mid[1] - 0.5*minmod(theta*(right_mid[1] - center[1]),
                                                    0.5*(right_most[1] - center[1]),
                                                    theta*(right_most[1] - right_mid[1]));

                prims_r[2] = right_mid[2] - 0.5*minmod(theta*(right_mid[2] - center[2]),
                                                    0.5*(right_most[2] - center[2]),
                                                    theta*(right_most[2] - right_mid[2]));

                // Calculate the left and right states using the reconstructed PLM primitives
                u_l = calc_state1D(gamma, prims_l[0], prims_l[1], prims_l[2]);
                u_r = calc_state1D(gamma, prims_r[0], prims_r[1], prims_r[2]);

                f_l = calc_flux1D(gamma, prims_l[0], prims_l[1], prims_l[2]);
                f_r = calc_flux1D(gamma, prims_r[0], prims_r[1], prims_r[2]);

                f1 = calc_hll_flux1D(gamma, u_l, u_r, f_l, f_r);

                // Do the same thing, but for the right side interface [i - 1/2]
                prims_l[0] = left_mid[0] + 0.5 *minmod(theta*(left_mid[0] - left_most[0]),
                                                        0.5*(center[0] -left_most[0]),
                                                        theta*(center[0] - left_mid[0]));

                prims_l[1] = left_mid[1] + 0.5 *minmod(theta*(left_mid[1] - left_most[1]),
                                                        0.5*(center[1] -left_most[1]),
                                                        theta*(center[1] - left_mid[1]));
                
                prims_l[2] = left_mid[2] + 0.5 *minmod(theta*(left_mid[2] - left_most[2]),
                                                        0.5*(center[2] -left_most[2]),
                                                        theta*(center[2] - left_mid[2]));


                    
                prims_r[0] = center[0] - 0.5 *minmod(theta*(center[0] - left_mid[0]),
                                                    0.5*(right_mid[0] - left_mid[0]),
                                                    theta*(right_mid[0] - center[0]));

                prims_r[1] = center[1] - 0.5 *minmod(theta*(center[1] - left_mid[1]),
                                                    0.5*(right_mid[1] - left_mid[1]),
                                                    theta*(right_mid[1] - center[1]));

                prims_r[2] = center[2] - 0.5 *minmod(theta*(center[2] - left_mid[2]),
                                                    0.5*(right_mid[2] - left_mid[2]),
                                                    theta*(right_mid[2] - center[2]));

                // Calculate the left and right states using the reconstructed PLM primitives
                u_l = calc_state1D(gamma, prims_l[0], prims_l[1], prims_l[2]);
                u_r = calc_state1D(gamma, prims_r[0], prims_r[1], prims_r[2]);

                f_l = calc_flux1D(gamma, prims_l[0], prims_l[1], prims_l[2]);
                f_r = calc_flux1D(gamma, prims_r[0], prims_r[1], prims_r[2]);

                f2 = calc_hll_flux1D(gamma, u_l, u_r, f_l, f_r);

                L[0][coordinate] = - (f1[0] - f2[0])/dx;
                L[1][coordinate] = - (f1[1] - f2[1])/dx;
                L[2][coordinate] = - (f1[2] - f2[2])/dx;            
                                                                                                                         
            }

        } else {
            double r_left, r_right, volAvg, pc;
            double log_rLeft, log_rRight;

            double delta_logr = (log10(r[physical_grid - 1]) - log10(r[0]))/physical_grid;

            long double dr = (r[physical_grid - 1] - r[0])/physical_grid;
            for (int ii = i_start; ii < i_bound; ii++){
                if (periodic){
                    coordinate = ii;
                    // Declare the c[i-2],c[i-1],c_i,c[i+1], c[i+2] variables
                    left_most[0] = roll(prims[0], ii - 2);
                    left_mid[0] = roll(prims[0], ii - 1);
                    center[0] = prims[0][ii];
                    right_mid[0] = roll(prims[0], ii + 1);
                    right_most[0] = roll(prims[0], ii + 2);

                    left_most[1] = roll(prims[1], ii - 2);
                    left_mid[1] = roll(prims[1], ii - 1);
                    center[1] = prims[1][ii];
                    right_mid[1] = roll(prims[1], ii + 1);
                    right_most[1] = roll(prims[1], ii + 2);

                    left_most[2] = roll(prims[2], ii - 2);
                    left_mid[2] = roll(prims[2], ii - 1);
                    center[2] = prims[2][ii];
                    right_mid[2] = roll(prims[2], ii + 1);
                    right_most[2] = roll(prims[2], ii + 2);

                } else {
                    // Adjust for beginning input of L vector
                    coordinate = ii - 2;
                    left_most[0] = prims[0][ii - 2];
                    left_mid[0] = prims[0][ii - 1];
                    center[0] = prims[0][ii];
                    right_mid[0] = prims[0][ii + 1];
                    right_most[0] = prims[0][ii + 2];

                    left_most[1] = prims[1][ii - 2];
                    left_mid[1] = prims[1][ii - 1];
                    center[1] = prims[1][ii];
                    right_mid[1] = prims[1][ii + 1];
                    right_most[1] = prims[1][ii + 2];

                    left_most[2] = prims[2][ii - 2];
                    left_mid[2] = prims[2][ii - 1];
                    center[2] = prims[2][ii];
                    right_mid[2] = prims[2][ii + 1];
                    right_most[2] = prims[2][ii + 2];

                }

                // Compute the reconstructed primitives at the i+1/2 interface

                // Reconstructed left primitives vector
                prims_l[0] = center[0] + 0.5*minmod(theta*(center[0] - left_mid[0]),
                                                    0.5*(right_mid[0] - left_mid[0]),
                                                    theta*(right_mid[0] - center[0]));

                prims_l[1] = center[1] + 0.5*minmod(theta*(center[1] - left_mid[1]),
                                                    0.5*(right_mid[1] - left_mid[1]),
                                                    theta*(right_mid[1] - center[1]));

                prims_l[2] = center[2] + 0.5*minmod(theta*(center[2] - left_mid[2]),
                                                    0.5*(right_mid[2] - left_mid[2]),
                                                    theta*(right_mid[2] - center[2]));

                // Reconstructed right primitives vector
                prims_r[0] = right_mid[0] - 0.5*minmod(theta*(right_mid[0] - center[0]),
                                                    0.5*(right_most[0] - center[0]),
                                                    theta*(right_most[0] - right_mid[0]));

                prims_r[1] = right_mid[1] - 0.5*minmod(theta*(right_mid[1] - center[1]),
                                                    0.5*(right_most[1] - center[1]),
                                                    theta*(right_most[1] - right_mid[1]));

                prims_r[2] = right_mid[2] - 0.5*minmod(theta*(right_mid[2] - center[2]),
                                                    0.5*(right_most[2] - center[2]),
                                                    theta*(right_most[2] - right_mid[2]));

                // Calculate the left and right states using the reconstructed PLM primitives
                u_l = calc_state1D(gamma, prims_l[0], prims_l[1], prims_l[2]);
                u_r = calc_state1D(gamma, prims_r[0], prims_r[1], prims_r[2]);

                f_l = calc_flux1D(gamma, prims_l[0], prims_l[1], prims_l[2]);
                f_r = calc_flux1D(gamma, prims_r[0], prims_r[1], prims_r[2]);

                f1 = calc_hll_flux1D(gamma, u_l, u_r, f_l, f_r);

                // Do the same thing, but for the right side interface [i - 1/2]
                prims_l[0] = left_mid[0] + 0.5 *minmod(theta*(left_mid[0] - left_most[0]),
                                                        0.5*(center[0] -left_most[0]),
                                                        theta*(center[0] - left_mid[0]));

                prims_l[1] = left_mid[1] + 0.5 *minmod(theta*(left_mid[1] - left_most[1]),
                                                        0.5*(center[1] -left_most[1]),
                                                        theta*(center[1] - left_mid[1]));
                
                prims_l[2] = left_mid[2] + 0.5 *minmod(theta*(left_mid[2] - left_most[2]),
                                                        0.5*(center[2] -left_most[2]),
                                                        theta*(center[2] - left_mid[2]));


                    
                prims_r[0] = center[0] - 0.5 *minmod(theta*(center[0] - left_mid[0]),
                                                    0.5*(right_mid[0] - left_mid[0]),
                                                    theta*(right_mid[0] - center[0]));

                prims_r[1] = center[1] - 0.5 *minmod(theta*(center[1] - left_mid[1]),
                                                    0.5*(right_mid[1] - left_mid[1]),
                                                    theta*(right_mid[1] - center[1]));

                prims_r[2] = center[2] - 0.5 *minmod(theta*(center[2] - left_mid[2]),
                                                    0.5*(right_mid[2] - left_mid[2]),
                                                    theta*(right_mid[2] - center[2]));

                // Calculate the left and right states using the reconstructed PLM primitives
                u_l = calc_state1D(gamma, prims_l[0], prims_l[1], prims_l[2]);
                u_r = calc_state1D(gamma, prims_r[0], prims_r[1], prims_r[2]);

                f_l = calc_flux1D(gamma, prims_l[0], prims_l[1], prims_l[2]);
                f_r = calc_flux1D(gamma, prims_r[0], prims_r[1], prims_r[2]);

                f2 = calc_hll_flux1D(gamma, u_l, u_r, f_l, f_r);

                //Get Central Pressure
                pc = center[1];

                if (linspace){
                    double right_cell = r[coordinate + 1];
                    double left_cell = r[coordinate - 1];

                    // Outflow the left/right boundaries
                    if (coordinate - 1 < 0){
                        left_cell = r[coordinate];
                    } else if (coordinate == physical_grid - 1){
                        right_cell = r[coordinate];
                    }

                    r_right = 0.5*(right_cell + r[coordinate]);
                    r_left = 0.5*(r[coordinate] + left_cell);

                } else {
                    log_rLeft = log10(r[0]) + coordinate*delta_logr;
                    log_rRight = log_rLeft + delta_logr;
                    r_left = pow(10, log_rLeft);
                    r_right = pow(10, log_rRight);
                }
                
                volAvg = 0.75*( ( pow(r_right, 4) - pow(r_left, 4) )/ ( pow(r_right, 3) - pow(r_left, 3) ) );

                L[0][coordinate] = - (pow(r_right,2)*f1[0] - pow(r_left,2)*f2[0] )/(pow(volAvg,2)*dr);
                L[1][coordinate] = - (pow(r_right,2)*f1[1] - pow(r_left,2)*f2[1] )/(pow(volAvg,2)*dr) + 2*pc/volAvg;
                L[2][coordinate] = - (pow(r_right,2)*f1[2] - pow(r_left,2)*f2[2] )/(pow(volAvg,2)*dr);


            }
        
        }

        return L; 
    }
    
};

//==========================================
//                 2D
//==========================================
vector<vector<vector<double> > > Ustate2D::u_dot2D(float gamma, vector<vector<vector<double> > > u_state, 
                                        bool periodic = false, float theta = 1.5)
{

    int i_start, i_bound, j_start, j_bound, xcoordinate, ycoordinate, xcenter, ycenter;
    int quadxL, quadxR, quadyL, quadyR;
    int xgrid_size = u_state[0][0].size();
    int ygrid_size = u_state[0].size();
    int xphysical_grid = xgrid_size - 4;
    int yphysical_grid = ygrid_size - 4;
    int xl_coord[2], xr_coord[2], yl_coord[2], yr_coord[2];

    int n_vars = u_state.size();

    double dx = 2.0/xphysical_grid;
    double dy = 2.0/yphysical_grid;
    xcenter = xphysical_grid/2 + 2; 
    ycenter = yphysical_grid/2 + 2;

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

    prims = cons2prim2D(u_state);

    // cout << "Rho: " << prims[0][5][5] << endl;
    // cout << "Pressure: " << prims[1][5][5] << endl;
    // cout << "Vx: " << prims[2][5][5] << endl;
    // cout << "Vy: " << prims[3][5][5] << endl;
    // string a;
    // cin >> a;
    

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

            xprims_l = cons2prim(gamma, ux_l, true);
            xprims_r = cons2prim(gamma, ux_r, true);
            yprims_l = cons2prim(gamma, uy_l, true);
            yprims_r = cons2prim(gamma, uy_r, true);
            
            f_l = calc_flux2D(gamma, xprims_l[0], xprims_l[1], xprims_l[2], xprims_l[3]);
            f_r = calc_flux2D(gamma, xprims_r[0], xprims_r[1], xprims_r[2], xprims_r[3]);

            g_l = calc_flux2D(gamma, yprims_l[0], yprims_l[1], yprims_l[2], yprims_l[3], false);
            g_r = calc_flux2D(gamma, yprims_r[0], yprims_r[1], yprims_r[2], yprims_r[3], false);

            // Calc HLL Flux at i+1/2 interface
            f1 = calc_hll_flux2D(gamma, ux_l, ux_r, f_l, f_r);
            g1 = calc_hll_flux2D(gamma, uy_l, uy_r, g_l, g_r);

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

            xprims_l = cons2prim(gamma, ux_l, true);
            xprims_r = cons2prim(gamma, ux_r, true);
            yprims_l = cons2prim(gamma, uy_l, true);
            yprims_r = cons2prim(gamma, uy_r, true);

            f_l = calc_flux2D(gamma, xprims_l[0], xprims_l[1], xprims_l[2], xprims_l[3]);
            f_r = calc_flux2D(gamma, xprims_r[0], xprims_r[1], xprims_r[2], xprims_r[3]);

            g_l = calc_flux2D(gamma, yprims_l[0], yprims_l[1], yprims_l[2], yprims_l[3], false);
            g_r = calc_flux2D(gamma, yprims_r[0], yprims_r[1], yprims_r[2], yprims_r[3], false);

            // Calc HLL Flux at i+1/2 interface
            f2 = calc_hll_flux2D(gamma, ux_l, ux_r, f_l, f_r);
            g2 = calc_hll_flux2D(gamma, uy_l, uy_r, g_l, g_r);
            

            L[0][ycoordinate][xcoordinate] = - (f1[0] - f2[0])/dx - (g1[0] - g2[0])/dy;
            L[1][ycoordinate][xcoordinate] = - (f1[1] - f2[1])/dx - (g1[1] - g2[1])/dy;
            L[2][ycoordinate][xcoordinate] = - (f1[2] - f2[2])/dx - (g1[2] - g2[2])/dy;
            L[3][ycoordinate][xcoordinate] = - (f1[3] - f2[3])/dx - (g1[3] - g2[3])/dy;

        }
    }
    */

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
                rho_transpose = tranpose(prims[0]);
                pressure_transpose = tranpose(prims[1]);
                vx_transpose = tranpose(prims[2]);
                vy_transpose = tranpose(prims[3]);

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

            // Figure out the coordinates of the left/right 
            // primitives in each direction
            xl_coord[0] = ii;
            xl_coord[1] = jj;

            xr_coord[0] = ii + 1;
            xr_coord[1] = jj;

            yl_coord[0] = ii;
            yl_coord[1] = jj;

            yr_coord[0] = ii;
            yr_coord[1] = jj + 1;

            // Check which quadrant (L/R) we are calculating in X_L
            if ( (xl_coord[0] > xcenter) && (xl_coord[1] > ycenter ) ){
                quadxL = 1;
            } else if ( (xl_coord[0] < xcenter) && (xl_coord[1] > ycenter) ){
                quadxL = 2;
            } else if ( (xl_coord[0] < xcenter) && (xl_coord[1] < ycenter) ){
                quadxL = 3;
            } else if ( (xl_coord[0] > xcenter) && (xl_coord[1] < ycenter) ){
                quadxL = 4;
            } else {
                quadxL = 1;
            }

            // Check if one x-y axes or in center
            if ( (xl_coord[0] == xcenter) && (xl_coord[1] > ycenter) ) {
                quadxL = 1.5;
            } else if ( (xl_coord[0] == xcenter) && (xl_coord[1] < ycenter) ){
                quadxL = 3.5;
            } else if ( (xl_coord[1] == ycenter) && (xl_coord[0] < xcenter) ){
                quadxL = 2.5;
            } else if ( (xl_coord[1] == ycenter) && (xl_coord[1] > xcenter) ) {
                quadxL = 0.5;
            }

            // Check which quadrant (L/R) we are calculating in X_R
            if ( (xr_coord[0] > xcenter) && (xr_coord[1] > ycenter ) ){
                quadxR = 1;
            } else if ( (xr_coord[0] < xcenter) && (xr_coord[1] > ycenter) ){
                quadxR = 2;
            } else if ( (xr_coord[0] < xcenter) && (xr_coord[1] < ycenter) ){
                quadxR = 3;
            } else if ( (xr_coord[0] > xcenter) && (xr_coord[1] < ycenter) ){
                quadxR = 4;
            } else {
                quadxR = 1;
            }

            // Check if one x-y axes or in center
            if ( (xr_coord[0] == xcenter) && (xr_coord[1] > ycenter) ) {
                quadxR = 1.5;
            } else if ( (xr_coord[0] == xcenter) && (xr_coord[1] < ycenter) ){
                quadxR = 3.5;
            } else if ( (xr_coord[1] == ycenter) && (xr_coord[0] < xcenter) ){
                quadxR = 2.5;
            } else if ( (xr_coord[1] == ycenter) && (xr_coord[1] > xcenter) ) {
                quadxR = 0.5;
            }

            // Check which quadrant (L/R) we are calculating in Y_L
            if ( (yl_coord[0] > xcenter) && (yl_coord[1] > ycenter ) ){
                quadyL = 1;
            } else if ( (yl_coord[0] < xcenter) && (yl_coord[1] > ycenter) ){
                quadyL = 2;
            } else if ( (yl_coord[0] < xcenter) && (yl_coord[1] < ycenter) ){
                quadyL = 3;
            } else if ( (yl_coord[0] > xcenter) && (yl_coord[1] < ycenter) ){
                quadyL = 4;
            } else {
                quadyL = 1;
            }

            // Check if one x-y axes or in center
            if ( (yl_coord[0] == xcenter) && (yl_coord[1] > ycenter) ) {
                quadyL = 1.5;
            } else if ( (yl_coord[0] == xcenter) && (yl_coord[1] < ycenter) ){
                quadyL = 3.5;
            } else if ( (yl_coord[1] == ycenter) && (yl_coord[0] < xcenter) ){
                quadyL = 2.5;
            } else if ( (yl_coord[1] == ycenter) && (yl_coord[1] > xcenter) ) {
                quadyL = 0.5;
            }

            // Check which quadrant (L/R) we are calculating in Y_R
            if ( (yr_coord[0] > xcenter) && (yr_coord[1] > ycenter ) ){
                quadyR = 1;
            } else if ( (yr_coord[0] < xcenter) && (yr_coord[1] > ycenter) ){
                quadyR = 2;
            } else if ( (yr_coord[0] < xcenter) && (yr_coord[1] < ycenter) ){
                quadyR = 3;
            } else if ( (yr_coord[0] > xcenter) && (yr_coord[1] < ycenter) ){
                quadyR = 4;
            } else {
                quadyR = 1;
            }

            // Check if one x-y axes or in center
            if ( (yr_coord[0] == xcenter) && (yr_coord[1] > ycenter) ) {
                quadyR = 1.5;
            } else if ( (yr_coord[0] == xcenter) && (yr_coord[1] < ycenter) ){
                quadyR = 3.5;
            } else if ( (yr_coord[1] == ycenter) && (yr_coord[0] < xcenter) ){
                quadyR = 2.5;
            } else if ( (yr_coord[1] == ycenter) && (yr_coord[1] > xcenter) ) {
                quadyR = 0.5;
            }

            // Check if we are at the very center of the grid
            if ( (xl_coord[0] == xcenter) && (xl_coord[1] == ycenter)){
                quadxL = 0;
            } else if ( (xr_coord[0] == xcenter) && (xr_coord[1] == ycenter)){
                quadxR = 0;
            } else if ( (yl_coord[0] == xcenter) && (yl_coord[1] == ycenter)){
                quadyL = 0;
            } else if ( (yr_coord[0] == xcenter) && (yr_coord[1] == ycenter)){
                quadyR = 0;
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

            
            /**
            if ( (jj == 5) && (ii == 7)){
                cout << "Rho (R): " << xprims_r[0] << endl;
                cout << "Rho (L): " << xprims_l[0] << endl;
                cout << "P (R): " << yprims_r[1] << endl;
                cout << "P (L): " << yprims_l[1] << endl;
                cout << "Vx (R): " << yprims_r[2] << endl;
                cout << "Vx (L): " << yprims_l[2] << endl;
                cout << "Vy (R): " << yprims_r[3] << endl;
                cout << "Vy (L): " << yprims_l[3] << endl;
                string e;
                cin >> e;

            }
            */
            
            
            
            
            // Calculate the left and right states using the reconstructed PLM primitives
            ux_l = calc_state2D(gamma, xprims_l[0], xprims_l[1], xprims_l[2], xprims_l[3]);
            ux_r = calc_state2D(gamma, xprims_r[0], xprims_r[1], xprims_r[2], xprims_r[3]);

            uy_l = calc_state2D(gamma, yprims_l[0], yprims_l[1], yprims_l[2], yprims_l[3]);
            uy_r = calc_state2D(gamma, yprims_r[0], yprims_r[1], yprims_r[2], yprims_r[3]);

            f_l = calc_flux2D(gamma, xprims_l[0], xprims_l[1], xprims_l[2], xprims_l[3]);
            f_r = calc_flux2D(gamma, xprims_r[0], xprims_r[1], xprims_r[2], xprims_r[3]);

            g_l = calc_flux2D(gamma, yprims_l[0], yprims_l[1], yprims_l[2], yprims_l[3], false);
            g_r = calc_flux2D(gamma, yprims_r[0], yprims_r[1], yprims_r[2], yprims_r[3], false);

            f1 = calc_hll_flux2D(gamma, ux_l, ux_r, f_l, f_r, quadxL, quadxR);
            g1 = calc_hll_flux2D(gamma, uy_l, uy_r, g_l, g_r, quadyL, quadyR);

            
            /**
            if ( (jj == 5) && (ii == 7)){
                cout << "Top" << endl;
                cout << "i,j +1/2: " << endl;
                cout << "Ux (R): " << ux_r[3] << endl;
                cout << "Ux (L): " << ux_l[3] << endl;
                cout << "Uy (R): " << uy_r[3] << endl;
                cout << "Uy (L): " << uy_l[3] << endl;
                cout << "F (R): " << f_r[3] << endl;
                cout << "F (L): " << f_l[3] << endl;
                cout << "G (R): " << g_r[3] << endl;
                cout << "G (L): " << g_l[3] << endl;
                string a;
                cin >> a;

            } else if ( (jj == 9) && (ii == 7) ) {
                cout << endl;
                cout << "Bottom" << endl;
                cout << "i,j + 1/2: " << endl;
                cout << "Ux (R): " << ux_r[3] << endl;
                cout << "Ux (L): " << ux_l[3] << endl;
                cout << "Uy (R): " << uy_r[3] << endl;
                cout << "Uy (L): " << uy_l[3] << endl;
                cout << "F (R): " << f_r[3] << endl;
                cout << "F (L): " << f_l[3] << endl;
                cout << "G (R): " << g_r[3] << endl;
                cout << "G (L): " << g_l[3] << endl;
                string a;
                cin >> a;

            }
            */
            
            



            // Do the same thing, but for the left side interface [i - 1/2]

            // Figure out the coordinates of the left/right 
            // primitives in each direction
            xl_coord[0] = ii - 1;
            xl_coord[1] = jj;

            xr_coord[0] = ii;
            xr_coord[1] = jj;

            yl_coord[0] = ii;
            yl_coord[1] = jj - 1;

            yr_coord[0] = ii;
            yr_coord[1] = jj;

            // Check which quadrant (L/R) we are calculating in X_L
            if ( (xl_coord[0] > xcenter) && (xl_coord[1] > ycenter ) ){
                quadxL = 1;
            } else if ( (xl_coord[0] < xcenter) && (xl_coord[1] > ycenter) ){
                quadxL = 2;
            } else if ( (xl_coord[0] < xcenter) && (xl_coord[1] < ycenter) ){
                quadxL = 3;
            } else if ( (xl_coord[0] > xcenter) && (xl_coord[1] < ycenter) ){
                quadxL = 4;
            } else {
                quadxL = 1;
            }

            // Check if one x-y axes or in center
            if ( (xl_coord[0] == xcenter) && (xl_coord[1] > ycenter) ) {
                quadxL = 1.5;
            } else if ( (xl_coord[0] == xcenter) && (xl_coord[1] < ycenter) ){
                quadxL = 3.5;
            } else if ( (xl_coord[1] == ycenter) && (xl_coord[0] < xcenter) ){
                quadxL = 2.5;
            } else if ( (xl_coord[1] == ycenter) && (xl_coord[1] > xcenter) ) {
                quadxL = 0.5;
            }

            // Check which quadrant (L/R) we are calculating in X_R
            if ( (xr_coord[0] > xcenter) && (xr_coord[1] > ycenter ) ){
                quadxR = 1;
            } else if ( (xr_coord[0] < xcenter) && (xr_coord[1] > ycenter) ){
                quadxR = 2;
            } else if ( (xr_coord[0] < xcenter) && (xr_coord[1] < ycenter) ){
                quadxR = 3;
            } else if ( (xr_coord[0] > xcenter) && (xr_coord[1] < ycenter) ){
                quadxR = 4;
            } else {
                quadxR = 1;
            }

            // Check if one x-y axes or in center
            if ( (xr_coord[0] == xcenter) && (xr_coord[1] > ycenter) ) {
                quadxR = 1.5;
            } else if ( (xr_coord[0] == xcenter) && (xr_coord[1] < ycenter) ){
                quadxR = 3.5;
            } else if ( (xr_coord[1] == ycenter) && (xr_coord[0] < xcenter) ){
                quadxR = 2.5;
            } else if ( (xr_coord[1] == ycenter) && (xr_coord[1] > xcenter) ) {
                quadxR = 0.5;
            }

            // Check which quadrant (L/R) we are calculating in Y_L
            if ( (yl_coord[0] > xcenter) && (yl_coord[1] > ycenter ) ){
                quadyL = 1;
            } else if ( (yl_coord[0] < xcenter) && (yl_coord[1] > ycenter) ){
                quadyL = 2;
            } else if ( (yl_coord[0] < xcenter) && (yl_coord[1] < ycenter) ){
                quadyL = 3;
            } else if ( (yl_coord[0] > xcenter) && (yl_coord[1] < ycenter) ){
                quadyL = 4;
            } else {
                quadyL = 1;
            }

            // Check if one x-y axes or in center
            if ( (yl_coord[0] == xcenter) && (yl_coord[1] > ycenter) ) {
                quadyL = 1.5;
            } else if ( (yl_coord[0] == xcenter) && (yl_coord[1] < ycenter) ){
                quadyL = 3.5;
            } else if ( (yl_coord[1] == ycenter) && (yl_coord[0] < xcenter) ){
                quadyL = 2.5;
            } else if ( (yl_coord[1] == ycenter) && (yl_coord[1] > xcenter) ) {
                quadyL = 0.5;
            }

            // Check which quadrant (L/R) we are calculating in Y_R
            if ( (yr_coord[0] > xcenter) && (yr_coord[1] > ycenter ) ){
                quadyR = 1;
            } else if ( (yr_coord[0] < xcenter) && (yr_coord[1] > ycenter) ){
                quadyR = 2;
            } else if ( (yr_coord[0] < xcenter) && (yr_coord[1] < ycenter) ){
                quadyR = 3;
            } else if ( (yr_coord[0] > xcenter) && (yr_coord[1] < ycenter) ){
                quadyR = 4;
            } else {
                quadyR = 1;
            }

            // Check if one x-y axes or in center
            if ( (yr_coord[0] == xcenter) && (yr_coord[1] > ycenter) ) {
                quadyR = 1.5;
            } else if ( (yr_coord[0] == xcenter) && (yr_coord[1] < ycenter) ){
                quadyR = 3.5;
            } else if ( (yr_coord[1] == ycenter) && (yr_coord[0] < xcenter) ){
                quadyR = 2.5;
            } else if ( (yr_coord[1] == ycenter) && (yr_coord[1] > xcenter) ) {
                quadyR = 0.5;
            }

            // Check if we are at the very center of the grid
            if ( (xl_coord[0] == xcenter) && (xl_coord[1] == ycenter)){
                quadxL = 0;
            } else if ( (xr_coord[0] == xcenter) && (xr_coord[1] == ycenter)){
                quadxR = 0;
            } else if ( (yl_coord[0] == xcenter) && (yl_coord[1] == ycenter)){
                quadyL = 0;
            } else if ( (yr_coord[0] == xcenter) && (yr_coord[1] == ycenter)){
                quadyR = 0;
            }

            // Left side primitives in x
            xprims_l[0] = xleft_mid[0] - 0.5 *minmod(theta*(xleft_mid[0] - xleft_most[0]),
                                                    0.5*(center[0] - xleft_most[0]),
                                                    theta*(center[0] - xleft_mid[0]));

            xprims_l[1] = xleft_mid[1] - 0.5 *minmod(theta*(xleft_mid[1] - xleft_most[1]),
                                                    0.5*(center[1] -xleft_most[1]),
                                                    theta*(center[1] - xleft_mid[1]));
            
            xprims_l[2] = xleft_mid[2] - 0.5 *minmod(theta*(xleft_mid[2] - xleft_most[2]),
                                                    0.5*(center[2] - xleft_most[2]),
                                                    theta*(center[2] - xleft_mid[2]));
            
            xprims_l[3] = xleft_mid[3] - 0.5 *minmod(theta*(xleft_mid[3] - xleft_most[3]),
                                                    0.5*(center[3] - xleft_most[3]),
                                                    theta*(center[3] - xleft_mid[3]));

                
            // Right side primitives in x
            xprims_r[0] = center[0] + 0.5 *minmod(theta*(center[0] - xleft_mid[0]),
                                                0.5*(xright_mid[0] - xleft_mid[0]),
                                                theta*(xright_mid[0] - center[0]));

            xprims_r[1] = center[1] + 0.5 *minmod(theta*(center[1] - xleft_mid[1]),
                                                0.5*(xright_mid[1] - xleft_mid[1]),
                                                theta*(xright_mid[1] - center[1]));

            xprims_r[2] = center[2] + 0.5 *minmod(theta*(center[2] - xleft_mid[2]),
                                                0.5*(xright_mid[2] - xleft_mid[2]),
                                                theta*(xright_mid[2] - center[2]));

            xprims_r[3] = center[3] + 0.5 *minmod(theta*(center[3] - xleft_mid[3]),
                                                0.5*(xright_mid[3] - xleft_mid[3]),
                                                theta*(xright_mid[3] - center[3]));


            // Left side primitives in y
            yprims_l[0] = yleft_mid[0] - 0.5 *minmod(theta*(yleft_mid[0] - yleft_most[0]),
                                                    0.5*(center[0] - yleft_most[0]),
                                                    theta*(center[0] - yleft_mid[0]));

            yprims_l[1] = yleft_mid[1] - 0.5 *minmod(theta*(yleft_mid[1] - yleft_most[1]),
                                                    0.5*(center[1] -yleft_most[1]),
                                                    theta*(center[1] - yleft_mid[1]));
            
            yprims_l[2] = yleft_mid[2] - 0.5 *minmod(theta*(yleft_mid[2] - yleft_most[2]),
                                                    0.5*(center[2] - yleft_most[2]),
                                                    theta*(center[2] - yleft_mid[2]));
            
            yprims_l[3] = yleft_mid[3] - 0.5 *minmod(theta*(yleft_mid[3] - yleft_most[3]),
                                                    0.5*(center[3] - yleft_most[3]),
                                                    theta*(center[3] - yleft_mid[3]));

                
            // Right side primitives in y
            yprims_r[0] = center[0] + 0.5 *minmod(theta*(center[0] - yleft_mid[0]),
                                                0.5*(yright_mid[0] - yleft_mid[0]),
                                                theta*(yright_mid[0] - center[0]));

            yprims_r[1] = center[1] + 0.5 *minmod(theta*(center[1] - yleft_mid[1]),
                                                0.5*(yright_mid[1] - yleft_mid[1]),
                                                theta*(yright_mid[1] - center[1]));

            yprims_r[2] = center[2] + 0.5 *minmod(theta*(center[2] - yleft_mid[2]),
                                                0.5*(yright_mid[2] - yleft_mid[2]),
                                                theta*(yright_mid[2] - center[2]));

            yprims_r[3] = center[3]  + 0.5 *minmod(theta*(center[3] - yleft_mid[3]),
                                                0.5*(yright_mid[3] - yleft_mid[3]),
                                                theta*(yright_mid[3] - center[3])); 

            /**
            if ( (jj == 5) && (ii == 7)){
                cout << "U Part: " << u_state[2][jj][ii] << endl;
                cout << "Prims:  " << prims[3][jj][ii] << endl;
                cout << "Center: " << center[3] << endl;
                cout << "X Prims (R): " << xprims_r[3] << endl;
                cout << "X Prims (L): " << xprims_l[3] << endl;
                cout << "Y Prims (R): " << yprims_r[3] << endl;
                cout << "Y Prims (L): " << yprims_l[3] << endl;
                string p;
                cin >> p;

            }
            */
             
        

            // Calculate the left and right states using the reconstructed PLM primitives
            ux_l = calc_state2D(gamma,xprims_l[0], xprims_l[1], xprims_l[2], xprims_l[3]);
            ux_r = calc_state2D(gamma,xprims_r[0], xprims_r[1], xprims_r[2], xprims_r[3]);

            uy_l = calc_state2D(gamma, yprims_l[0], yprims_l[1], yprims_l[2], yprims_l[3]);
            uy_r = calc_state2D(gamma, yprims_r[0], yprims_r[1], yprims_r[2], yprims_r[3]);

            f_l = calc_flux2D(gamma, xprims_l[0], xprims_l[1], xprims_l[2], xprims_l[3]);
            f_r = calc_flux2D(gamma, xprims_r[0], xprims_r[1], xprims_r[2], xprims_r[3]);

            g_l = calc_flux2D(gamma, yprims_l[0], yprims_l[1], yprims_l[2], yprims_l[3], false);
            g_r = calc_flux2D(gamma, yprims_r[0], yprims_r[1], yprims_r[2], yprims_r[3], false);

            f2 = calc_hll_flux2D(gamma, ux_l, ux_r, f_l, f_r, quadxL, quadxR);
            g2 = calc_hll_flux2D(gamma, uy_l, uy_r, g_l, g_r, quadyL, quadyR);


            
            /**
            if ( (jj == 5) && (ii == 7)){
                cout << endl;
                cout << "i,j -1/2: " << endl;
                cout << "Ux (R): " << ux_r[3] << endl;
                cout << "Ux (L): " << ux_l[3] << endl;
                cout << "Uy (R): " << uy_r[3] << endl;
                cout << "Uy (L): " << uy_l[3] << endl;
                cout << "F (R): " << f_r[3] << endl;
                cout << "F (L): " << f_l[3] << endl;
                cout << "G (R): " << g_r[3] << endl;
                cout << "G (L): " << g_l[3] << endl;
                string e;
                cin >> e;

            } else if ( (jj == 9) && (ii == 7) ) {
                cout << endl;
                cout << "Bottom" << endl;
                cout << "i,j -1/2: " << endl;
                cout << "Ux (R): " << ux_r[3] << endl;
                cout << "Ux (L): " << ux_l[3] << endl;
                cout << "Uy (R): " << uy_r[3] << endl;
                cout << "Uy (L): " << uy_l[3] << endl;
                cout << "F (R): " << f_r[3] << endl;
                cout << "F (L): " << f_l[3] << endl;
                cout << "G (R): " << g_r[3] << endl;
                cout << "G (L): " << g_l[3] << endl;
                string a;
                cin >> a;

            }
            
            
            
            

            
            
            if ( (jj == 5) && (ii == 7 )){
                cout << "F1: " << f1[3] << endl;
                cout << "G1: " << g1[3] << endl;
                cout << "F2: " << f2[3] << endl;
                cout << "G2: " << g2[3] << endl;
                string h;
                cin >> h;
            } else if ( (jj == 9) && (ii == 7) ) {
                cout << "F1: " << f1[3] << endl;
                cout << "G1: " << g1[3] << endl;
                cout << "F2: " << f2[3] << endl;
                cout << "G2: " << g2[3] << endl;
                string h;

            }
            
            */
            

            
            

            L[0][ycoordinate][xcoordinate] = - (f1[0] - f2[0])/dx - (g1[0] - g2[0])/dy;
            L[1][ycoordinate][xcoordinate] = - (f1[1] - f2[1])/dx - (g1[1] - g2[1])/dy;
            L[2][ycoordinate][xcoordinate] = - (f1[2] - f2[2])/dx - (g1[2] - g2[2])/dy;
            L[3][ycoordinate][xcoordinate] = - (f1[3] - f2[3])/dx - (g1[3] - g2[3])/dy;

        }

    }
    
    
    
    
    return L;
};

 vector<vector<double> > Ustate::simulate1D(float tend = 0.1, float dt = 1.e-4, float theta=1.5, 
                                            bool first_order = true, bool periodic = false, bool linspace = true){

    // Define the swap vector for the integrated state
    int grid_size = state[0].size();
    int n_vars = state.size();

    vector<vector<double> > u_p(n_vars, vector<double>(grid_size, 0));
    vector<vector<double> > u(n_vars, vector<double>(grid_size, 0)); 
    vector<vector<double> > s(n_vars, vector<double>(grid_size, 0));
    float t = 0;

    // Copy the state array into real & profile variables
    u = state;
    u_p = u;

    if (first_order){
        int physical_grid = grid_size - 2;
        vector<vector<double> > udot(n_vars, vector<double>(physical_grid, 0));

        // cout << "E Init: " << u[2][1] << endl;
        while (t < tend){
            // Compute the REAL udot array, purging the ghost cells.
            udot = u_dot1D(gamma, u, true, periodic, theta, linspace);

            for (int ii = 0; ii < physical_grid; ii++){
                // Get the non-ghost index 
                int i_real =  ii + 1;
                u_p[0][i_real] = u[0][i_real] + dt*udot[0][ii];
                u_p[1][i_real] = u[1][i_real] + dt*udot[1][ii];
                u_p[2][i_real] = u[2][i_real] + dt*udot[2][ii];

            }

            // Readjust the ghost cells at i-1,i+1
            config_ghosts1D(u_p, physical_grid);
            
            // Swap the arrays
            u.swap(u_p);
            
            t += dt;

        }   

    } else {
        int physical_grid = grid_size - 4;
        vector<vector<double> > udot(n_vars, vector<double>(physical_grid, 0)); 

        vector<vector<double> > u1(n_vars, vector<double>(grid_size, 0));
        vector<vector<double> > u2(n_vars, vector<double>(grid_size, 0));
        vector<vector<double> > udot1(n_vars, vector<double>(physical_grid, 0));
        vector<vector<double> > udot2(n_vars, vector<double>(physical_grid, 0));

        u1 = u;
        u2 = u;
        while (t < tend){
            // Compute the REAL udot array, purging the ghost cells.
            udot = u_dot1D(gamma, u, false, periodic, theta, linspace);
            
            for (int ii = 0; ii < physical_grid; ii++){
                // Get the non-ghost index 
                int i_real =  ii + 2;
                u1[0][i_real] = u[0][i_real] + dt*udot[0][ii];
                u1[1][i_real] = u[1][i_real] + dt*udot[1][ii];
                u1[2][i_real] = u[2][i_real] + dt*udot[2][ii];

            }
            // Readjust the ghost cells at i-2,i-1,i+1,i+2
            config_ghosts1D(u1, physical_grid, false);

            udot1 = u_dot1D(gamma, u1, false, periodic, theta, linspace);

            for (int ii = 0; ii < physical_grid; ii++){
                // Get the non-ghost index 
                int i_real =  ii + 2;
                u2[0][i_real] = 0.75*u[0][i_real] + 0.25*u1[0][i_real] + 0.25*dt*udot1[0][ii];
                u2[1][i_real] = 0.75*u[1][i_real] + 0.25*u1[1][i_real] + 0.25*dt*udot1[1][ii];
                u2[2][i_real] = 0.75*u[2][i_real] + 0.25*u1[2][i_real] + 0.25*dt*udot1[2][ii];

            }

            config_ghosts1D(u2, physical_grid, false);
            
            udot2 = u_dot1D(gamma, u2, false, periodic, theta, linspace);

            for (int ii = 0; ii < physical_grid; ii++){
                // Get the non-ghost index 
                int i_real =  ii + 2;
                u_p[0][i_real] = (1.0/3.0)*u[0][i_real] + (2.0/3.0)*u2[0][i_real] + (2.0/3.0)*dt*udot2[0][ii];
                u_p[1][i_real] = (1.0/3.0)*u[1][i_real] + (2.0/3.0)*u2[1][i_real] + (2.0/3.0)*dt*udot2[1][ii];
                u_p[2][i_real] = (1.0/3.0)*u[2][i_real] + (2.0/3.0)*u2[2][i_real] + (2.0/3.0)*dt*udot2[2][ii];
                
            }

            // Readjust the ghost cells at i-2,i-1,i+1,i+2
            config_ghosts1D(u_p, physical_grid, false);
            
            // Swap the arrays
            u.swap(u_p);
            
            t += dt;

        }  

    }

    return u;

 };

  vector<vector<vector<double> > > Ustate2D::simulate2D(float tend = 0.1, bool periodic = false){

    // Define the swap vector for the integrated state
    int xgrid_size = state2D[0][0].size();
    int ygrid_size = state2D[0].size();
    int n_vars = state2D.size();
    int xphysical_grid = xgrid_size - 4;
    int yphysical_grid = ygrid_size - 4;
    float t = 0;
    float dt = 1.e-4;

    
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

    // Copy the state array into real & profile variables
    u = state2D;
    u_p = u;
    u1 = u; 
    u2 = u;
    
    while (t < tend){
        udot = u_dot2D(gamma, u);

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
        
        udot1 = u_dot2D(gamma, u1);

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
        
       
        udot2 = u_dot2D(gamma, u2);

        for (int jj = 0; jj < yphysical_grid; jj++){
            // Get the non-ghost index 
            int j_real =  jj + 2;
            for (int ii = 0; ii < xphysical_grid; ii++){
                int i_real = ii + 2;
                u_p[0][j_real][i_real] = u[0][j_real][i_real] + (2.0/3.0)*u2[0][j_real][i_real] + (2.0/3.0)*dt*udot2[0][jj][ii];
                u_p[1][j_real][i_real] = u[1][j_real][i_real] + (2.0/3.0)*u2[1][j_real][i_real] + (2.0/3.0)*dt*udot2[1][jj][ii];
                u_p[2][j_real][i_real] = u[2][j_real][i_real] + (2.0/3.0)*u2[2][j_real][i_real] + (2.0/3.0)*dt*udot2[2][jj][ii];
                u_p[3][j_real][i_real] = u[3][j_real][i_real] + (2.0/3.0)*u2[3][j_real][i_real] + (2.0/3.0)*dt*udot2[3][jj][ii];

            }

        }
        

       
        // Adjust the ghost cells at the boundaries of the new state tensor
        
        for (int jj = 0; jj < ygrid_size; jj++){
            for (int ii = 2; ii < xphysical_grid; ii++){
                if (jj < 2){
                    u_p[0][jj][ii] = u_p[0][2][ii];
                    u_p[1][jj][ii] = u_p[1][2][ii];
                    u_p[2][jj][ii] = u_p[2][2][ii];
                    u_p[3][jj][ii] = u_p[3][2][ii];
                    
                } else if (jj > ygrid_size - 3) {
                    u_p[0][jj][ii] = u_p[0][ygrid_size - 3][ii];
                    u_p[1][jj][ii] = u_p[1][ygrid_size - 3][ii];
                    u_p[2][jj][ii] = u_p[2][ygrid_size - 3][ii];
                    u_p[3][jj][ii] = u_p[3][ygrid_size - 3][ii];

                } else {
                    u_p[0][jj][0] = u_p[0][jj][2];
                    u_p[0][jj][1] = u_p[0][jj][2];
                    u_p[0][jj][xgrid_size - 1] = u_p[0][jj][xgrid_size - 3];
                    u_p[0][jj][xgrid_size - 2] = u_p[0][jj][xgrid_size - 3];

                    u_p[1][jj][0] = u_p[1][jj][2];
                    u_p[1][jj][1] = u_p[1][jj][2];
                    u_p[1][jj][xgrid_size - 1] = u_p[1][jj][xgrid_size - 3];
                    u_p[1][jj][xgrid_size - 2] = u_p[1][jj][xgrid_size - 3];

                    u_p[2][jj][0] = u_p[2][jj][2];
                    u_p[2][jj][1] = u_p[2][jj][2];
                    u_p[2][jj][xgrid_size - 1] = u_p[2][jj][xgrid_size - 3];
                    u_p[2][jj][xgrid_size - 2] = u_p[2][jj][xgrid_size - 3];

                    u_p[3][jj][0] = u_p[3][jj][2];
                    u_p[3][jj][1] = u_p[3][jj][2];
                    u_p[3][jj][xgrid_size - 1] = u_p[3][jj][xgrid_size - 3];
                    u_p[3][jj][xgrid_size - 2] = u_p[3][jj][xgrid_size - 3];
                }
            }
        }
        
        /**
        cout << " " << endl;
        cout << "U1: " << endl;
        for (int jj=0; jj <ygrid_size; jj++){
            for (int ii=0; ii < xgrid_size; ii++){
                cout << u1[3][jj][ii]/1.e-5 << ", ";
            }
            cout << endl;
        }
        string b;
        cin >> b;
        */
        

        // Swap the arrays
        u.swap(u_p);

        

        // u.swap(u1);
        
        t += dt;

        

    }

    return u;

 };