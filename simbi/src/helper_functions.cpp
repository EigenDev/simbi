
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

// Get the Sign of a Number
int sign(double x){
    if (x/abs(x) == 1) { 
        return 1;
    } else if (x/abs(x) == -1){
        return - 1;
    } else {
        return 0;
    }
};

// The Minmod slope delimiter
double minmod(double x, double y, double z){
    // The minimum
    double min_val = min(min(abs(x), abs(y)), abs(z)); 

    return 0.25*abs(sign(x) + sign(y))*(sign(x) + sign(z))*min_val;

};

MinMod minmod(PrimData &prims, double theta, int face, int i, int j, int NX){
    MinMod slope; 
    double x, y, z;

    switch (face)
    {
    // Compute flux limit at i,j+1/2 interface
    case 1:
        //------------ Left Cell Face X-Direction-------------------
        x = theta * (prims.rho[i + NX * j] - prims.rho[(i - 1) + NX * j]);
        y = 0.5 * (prims.rho[(i + 1) + NX * j] - prims.rho[(i - 1) + NX * j]);
        z = theta * (prims.rho[(i + 1) + NX * j] + prims.rho[i + NX * j]);
        slope.rhoL = 0.25 * abs(sign(x) + sign(y))*(sign(x) + sign(z))*min(min(abs(x), abs(y)), abs(z));

        x = theta * (prims.v1[i + NX * j] - prims.v1[(i - 1) + NX * j]);
        y = 0.5 * (prims.v1[(i + 1) + NX * j] - prims.v1[(i - 1) + NX * j]);
        z = theta * (prims.v1[(i + 1) + NX * j] + prims.v1[i + NX * j]);
        slope.v1L = 0.25 * abs(sign(x) + sign(y))*(sign(x) + sign(z))*min(min(abs(x), abs(y)), abs(z));

        x = theta * (prims.v2[i + NX * j] - prims.v2[(i - 1) + NX * j]);
        y = 0.5 * (prims.v2[(i + 1) + NX * j] - prims.v2[(i - 1) + NX * j]);
        z = theta * (prims.v2[(i + 1) + NX * j] + prims.v2[i + NX * j]);
        slope.v2L = 0.25 * abs(sign(x) + sign(y))*(sign(x) + sign(z))*min(min(abs(x), abs(y)), abs(z));

        x = theta * (prims.p[i + NX * j] - prims.p[(i - 1) + NX * j]);
        y = 0.5 * (prims.p[(i + 1) + NX * j] - prims.p[(i - 1) + NX * j]);
        z = theta * (prims.p[(i + 1) + NX * j] + prims.p[i + NX * j]);
        slope.pL = 0.25 * abs(sign(x) + sign(y))*(sign(x) + sign(z))*min(min(abs(x), abs(y)), abs(z));



        // -----------Right Cell Face X-Direction -------------------------
        x = theta * (prims.rho[(i + 1) + NX * j] - prims.rho[i + NX * j]);
        y = 0.5 * (prims.rho[(i + 2) + NX * j] - prims.rho[i + NX * j]);
        z = theta * (prims.rho[(i + 2) + NX * j] + prims.rho[(i + 1) + NX * j]);
        slope.rhoR = 0.25 * abs(sign(x) + sign(y))*(sign(x) + sign(z))*min(min(abs(x), abs(y)), abs(z));

        x = theta * (prims.v1[(i + 1) + NX * j] - prims.v1[i + NX * j]);
        y = 0.5 * (prims.v1[(i + 2) + NX * j] - prims.v1[i + NX * j]);
        z = theta * (prims.v1[(i + 2) + NX * j] + prims.v1[(i + 1) + NX * j]);
        slope.v1R = 0.25 * abs(sign(x) + sign(y))*(sign(x) + sign(z))*min(min(abs(x), abs(y)), abs(z));

        x = theta * (prims.v2[(i + 1) + NX * j] - prims.v2[i + NX * j]);
        y = 0.5 * (prims.v2[(i + 2) + NX * j] - prims.v2[i + NX * j]);
        z = theta * (prims.v2[(i + 2) + NX * j] + prims.v2[(i + 1) + NX * j]);
        slope.v2R = 0.25 * abs(sign(x) + sign(y))*(sign(x) + sign(z))*min(min(abs(x), abs(y)), abs(z));

        x = theta * (prims.p[(i + 1) + NX * j] - prims.p[i + NX * j]);
        y = 0.5 * (prims.p[(i + 2) + NX * j] - prims.p[i + NX * j]);
        z = theta * (prims.p[(i + 2) + NX * j] + prims.p[(i + 1) + NX * j]);
        slope.rhoR = 0.25 * abs(sign(x) + sign(y))*(sign(x) + sign(z))*min(min(abs(x), abs(y)), abs(z));



        //------------- Upper Cell Face ---------------------------------
        x = theta * (prims.rho[i + NX * j] - prims.rho[i + NX * (j - 1)]);
        y = 0.5   * (prims.rho[i + NX * (j + 1)] - prims.rho[i + NX * (j - 1)]);
        z = theta * (prims.rho[i + NX * (j + 1)] + prims.rho[i + NX * j]);
        slope.rhoT = 0.25 * abs(sign(x) + sign(y))*(sign(x) + sign(z))*min(min(abs(x), abs(y)), abs(z));

        x = theta * (prims.v1[i + NX * j] - prims.v1[i + NX * (j - 1)]);
        y = 0.5   * (prims.v1[i + NX * (j + 1)] - prims.v1[i + NX * (j - 1)]);
        z = theta * (prims.v1[i + NX * (j + 1)] + prims.v1[i + NX * j]);
        slope.v1T = 0.25 * abs(sign(x) + sign(y))*(sign(x) + sign(z))*min(min(abs(x), abs(y)), abs(z));

        x = theta * (prims.v2[i + NX * j] - prims.v2[i + NX * (j - 1)]);
        y = 0.5   * (prims.v2[i + NX * (j + 1)] - prims.v2[i + NX * (j - 1)]);
        z = theta * (prims.v2[i + NX * (j + 1)] + prims.v2[i + NX * j]);
        slope.v2T = 0.25 * abs(sign(x) + sign(y))*(sign(x) + sign(z))*min(min(abs(x), abs(y)), abs(z));

        x = theta * (prims.p[i + NX * j] - prims.p[i + NX * (j - 1)]);
        y = 0.5   * (prims.p[i + NX * (j + 1)] - prims.p[i + NX * (j - 1)]);
        z = theta * (prims.p[i + NX * (j + 1)] + prims.p[i + NX * j]);
        slope.pT = 0.25 * abs(sign(x) + sign(y))*(sign(x) + sign(z))*min(min(abs(x), abs(y)), abs(z));



        //---------------Lower Face--------------------------------------
        x = theta * (prims.rho[i + NX * (j + 1)] - prims.rho[i + NX * j]);
        y = 0.5   * (prims.rho[i + NX * (j + 2)] - prims.rho[i + NX * j]);
        z = theta * (prims.rho[i + NX * (j + 2)] + prims.rho[i + NX * (j + 1)]);
        slope.rhoB = 0.25 * abs(sign(x) + sign(y))*(sign(x) + sign(z))*min(min(abs(x), abs(y)), abs(z));

        x = theta * (prims.v1[i + NX * (j + 1)] - prims.v1[i + NX * j]);
        y = 0.5   * (prims.v1[i + NX * (j + 2)] - prims.v1[i + NX * j]);
        z = theta * (prims.v1[i + NX * (j + 2)] + prims.v1[i + NX * (j + 1)]);
        slope.v1B = 0.25 * abs(sign(x) + sign(y))*(sign(x) + sign(z))*min(min(abs(x), abs(y)), abs(z));

        x = theta * (prims.v2[i + NX * (j + 1)] - prims.v2[i + NX * j]);
        y = 0.5   * (prims.v2[i + NX * (j + 2)] - prims.v2[i + NX * j]);
        z = theta * (prims.v2[i + NX * (j + 2)] + prims.v2[i + NX * (j + 1)]);
        slope.v2B = 0.25 * abs(sign(x) + sign(y))*(sign(x) + sign(z))*min(min(abs(x), abs(y)), abs(z));

        x = theta * (prims.p[i + NX * (j + 1)] - prims.p[i + NX * j]);
        y = 0.5   * (prims.p[i + NX * (j + 2)] - prims.p[i + NX * j]);
        z = theta * (prims.p[i + NX * (j + 2)] + prims.p[i + NX * (j + 1)]);
        slope.pB = 0.25 * abs(sign(x) + sign(y))*(sign(x) + sign(z))*min(min(abs(x), abs(y)), abs(z));

        break;
    // Compute flux limiter at i,j-1/2 interface 
    case 2:
        //------------ Left Cell Face Y-Direction-------------------
        x = theta * (prims.rho[(i - 1) + NX * j] - prims.rho[(i - 2) + NX * j]);
        y = 0.5 * (prims.rho[i + NX * j] - prims.rho[(i - 2) + NX * j]);
        z = theta * (prims.rho[i + NX * j] + prims.rho[(i - 1) + NX * j]);
        slope.rhoL = 0.25 * abs(sign(x) + sign(y))*(sign(x) + sign(z))*min(min(abs(x), abs(y)), abs(z));

        x = theta * (prims.v1[(i - 1) + NX * j] - prims.v1[(i - 2) + NX * j]);
        y = 0.5 * (prims.v1[i + NX * j] - prims.v1[(i - 2) + NX * j]);
        z = theta * (prims.v1[i + NX * j] + prims.v1[(i - 1) + NX * j]);
        slope.v1L = 0.25 * abs(sign(x) + sign(y))*(sign(x) + sign(z))*min(min(abs(x), abs(y)), abs(z));

        x = theta * (prims.v2[(i - 1) + NX * j] - prims.v2[(i - 2) + NX * j]);
        y = 0.5 * (prims.v2[i + NX * j] - prims.v2[(i - 2) + NX * j]);
        z = theta * (prims.v2[i + NX * j] + prims.v2[(i - 1) + NX * j]);
        slope.v2L = 0.25 * abs(sign(x) + sign(y))*(sign(x) + sign(z))*min(min(abs(x), abs(y)), abs(z));

        x = theta * (prims.p[(i - 1) + NX * j] - prims.p[(i - 2) + NX * j]);
        y = 0.5 * (prims.p[i + NX * j] - prims.p[(i - 2) + NX * j]);
        z = theta * (prims.p[i + NX * j] + prims.p[(i - 1) + NX * j]);
        slope.pL = 0.25 * abs(sign(x) + sign(y))*(sign(x) + sign(z))*min(min(abs(x), abs(y)), abs(z));



        // -----------Right Cell Face X-Direction -------------------------
        x = theta * (prims.rho[i  + NX * j] - prims.rho[(i  - 1) + NX * j]);
        y = 0.5 * (prims.rho[(i + 1) + NX * j] - prims.rho[(i - 1) + NX * j]);
        z = theta * (prims.rho[(i + 1) + NX * j] + prims.rho[i + NX * j]);
        slope.rhoR = 0.25 * abs(sign(x) + sign(y))*(sign(x) + sign(z))*min(min(abs(x), abs(y)), abs(z));

        x = theta * (prims.v1[i + NX * j] - prims.v1[(i - 1) + NX * j]);
        y = 0.5 * (prims.v1[(i + 1) + NX * j] - prims.v1[(i - 1) + NX * j]);
        z = theta * (prims.v1[(i + 1) + NX * j] + prims.v1[i + NX * j]);
        slope.v1R = 0.25 * abs(sign(x) + sign(y))*(sign(x) + sign(z))*min(min(abs(x), abs(y)), abs(z));

        x = theta * (prims.v2[i + NX * j] - prims.v2[(i - 1) + NX * j]);
        y = 0.5 * (prims.v2[(i + 1) + NX * j] - prims.v2[(i - 1) + NX * j]);
        z = theta * (prims.v2[(i + 1) + NX * j] + prims.v2[i + NX * j]);
        slope.v2R = 0.25 * abs(sign(x) + sign(y))*(sign(x) + sign(z))*min(min(abs(x), abs(y)), abs(z));

        x = theta * (prims.p[i + NX * j] - prims.p[(i - 1) + NX * j]);
        y = 0.5   * (prims.p[(i + 1) + NX * j] - prims.p[(i - 1) + NX * j]);
        z = theta * (prims.p[(i + 1) + NX * j] + prims.p[i + NX * j]);
        slope.rhoR = 0.25 * abs(sign(x) + sign(y))*(sign(x) + sign(z))*min(min(abs(x), abs(y)), abs(z));



        //------------- Upper Cell Face ---------------------------------
        x = theta * (prims.rho[i + NX * (j - 1)] - prims.rho[i + NX * (j - 2)]);
        y = 0.5   * (prims.rho[i + NX * j] - prims.rho[i + NX * (j - 2)]);
        z = theta * (prims.rho[i + NX * j] + prims.rho[i + NX * (j - 1)]);
        slope.rhoT = 0.25 * abs(sign(x) + sign(y))*(sign(x) + sign(z))*min(min(abs(x), abs(y)), abs(z));

        x = theta * (prims.v1[i + NX * (j - 1)] - prims.v1[i + NX * (j - 2)]);
        y = 0.5   * (prims.v1[i + NX * j] - prims.v1[i + NX * (j - 2)]);
        z = theta * (prims.v1[i + NX * j] + prims.v1[i + NX * (j - 1)]);
        slope.v1T = 0.25 * abs(sign(x) + sign(y))*(sign(x) + sign(z))*min(min(abs(x), abs(y)), abs(z));

        x = theta * (prims.v2[i + NX * (j - 1)] - prims.v2[i + NX * (j - 2)]);
        y = 0.5   * (prims.v2[i + NX * j] - prims.v2[i + NX * (j - 2)]);
        z = theta * (prims.v2[i + NX * j] + prims.v2[i + NX * (j - 1)]);
        slope.v2T = 0.25 * abs(sign(x) + sign(y))*(sign(x) + sign(z))*min(min(abs(x), abs(y)), abs(z));

        x = theta * (prims.p[i + NX * (j - 1)] - prims.p[i + NX * (j - 2)]);
        y = 0.5   * (prims.p[i + NX * j] - prims.p[i + NX * (j - 2)]);
        z = theta * (prims.p[i + NX * j] + prims.p[i + NX * (j - 1)]);
        slope.pT = 0.25 * abs(sign(x) + sign(y))*(sign(x) + sign(z))*min(min(abs(x), abs(y)), abs(z));



        //---------------Bottom Face--------------------------------------
        x = theta * (prims.rho[i + NX * j] - prims.rho[i + NX * (j - 1)]);
        y = 0.5   * (prims.rho[i + NX * (j + 1)] - prims.rho[i + NX * (j - 1)]);
        z = theta * (prims.rho[i + NX * (j + 1)] + prims.rho[i + NX * j]);
        slope.rhoB = 0.25 * abs(sign(x) + sign(y))*(sign(x) + sign(z))*min(min(abs(x), abs(y)), abs(z));

        x = theta * (prims.v1[i + NX * j] - prims.v1[i + NX * (j - 1)]);
        y = 0.5   * (prims.v1[i + NX * (j + 1)] - prims.v1[i + NX * (j - 1)]);
        z = theta * (prims.v1[i + NX * (j + 1)] + prims.v1[i + NX * j]);
        slope.v1B = 0.25 * abs(sign(x) + sign(y))*(sign(x) + sign(z))*min(min(abs(x), abs(y)), abs(z));

        x = theta * (prims.v2[i + NX * j] - prims.v2[i + NX * (j - 1)]);
        y = 0.5   * (prims.v2[i + NX * (j + 1)] - prims.v2[i + NX * (j - 1)]);
        z = theta * (prims.v2[i + NX * (j + 1)] + prims.v2[i + NX * j]);
        slope.v2B = 0.25 * abs(sign(x) + sign(y))*(sign(x) + sign(z))*min(min(abs(x), abs(y)), abs(z));

        x = theta * (prims.p[i + NX * j] - prims.p[i + NX * (j - 1)]);
        y = 0.5   * (prims.p[i + NX * (j + 1)] - prims.p[i + NX * (j - 1)]);
        z = theta * (prims.p[i + NX * (j + 1)] + prims.p[i + NX * j]);
        slope.pB = 0.25 * abs(sign(x) + sign(y))*(sign(x) + sign(z))*min(min(abs(x), abs(y)), abs(z));

        break;
    }

    return slope;

}

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


// void config_ghosts1D(vector<vector<double> >&, int grid_size);
/**
void config_ghosts1D(vector<vector<double> > &u_state, int grid_size, bool first_order){
    if (first_order){
        u_state[0][0] = u_state[0][1];
        u_state[0][grid_size - 1] = u_state[0][grid_size - 2];

        u_state[1][0] = u_state[1][1];
        u_state[1][grid_size - 1] = u_state[1][grid_size - 2];

        u_state[2][0] = u_state[2][1];
        u_state[2][grid_size - 1] = u_state[2][grid_size - 2];
    } else {
        u_state[0][0] = u_state[0][2];
        u_state[0][1] = u_state[0][2];
        u_state[0][grid_size - 1] = u_state[0][grid_size - 3];
        u_state[0][grid_size - 2] = u_state[0][grid_size - 3];

        u_state[1][0] = u_state[1][2];
        u_state[1][1] = u_state[1][2];
        u_state[1][grid_size - 1] = u_state[1][grid_size - 3];
        u_state[1][grid_size - 2] = u_state[1][grid_size - 3];

        u_state[2][0] = u_state[2][2];
        u_state[2][1] = u_state[2][2];
        u_state[2][grid_size - 1] = u_state[2][grid_size - 3];
        u_state[2][grid_size - 2] = u_state[2][grid_size - 3];
    }
};

void config_ghosts2D(vector &u_state, 
                        int x1grid_size, int x2grid_size, bool first_order,
                        char kind){

    if (first_order){
        for (int jj = 0; jj < x2grid_size; jj++){
            for (int ii = 0; ii < x1grid_size; ii++){
                if (jj < 1){
                    u_state[0][jj][ii] =   u_state[0][1][ii];
                    u_state[1][jj][ii] =   u_state[1][1][ii];
                    u_state[2][jj][ii] = - u_state[2][1][ii];
                    u_state[3][jj][ii] =   u_state[3][1][ii];
                    
                } else if (jj > x2grid_size - 2) {
                    u_state[0][jj][ii] =   u_state[0][x2grid_size - 2][ii];
                    u_state[1][jj][ii] =   u_state[1][x2grid_size - 2][ii];
                    u_state[2][jj][ii] = - u_state[2][x2grid_size - 2][ii];
                    u_state[3][jj][ii] =   u_state[3][x2grid_size - 2][ii];

                } else {
                    u_state[0][jj][0] = u_state[0][jj][1];
                    u_state[0][jj][x1grid_size - 1] = u_state[0][jj][x1grid_size - 2];

                    u_state[1][jj][0] = - u_state[1][jj][1];
                    u_state[1][jj][x1grid_size - 1] = u_state[1][jj][x1grid_size - 2];

                    u_state[2][jj][0] = u_state[2][jj][1];
                    u_state[2][jj][x1grid_size - 1] = u_state[2][jj][x1grid_size - 2];

                    u_state[3][jj][0] = u_state[3][jj][1];
                    u_state[3][jj][x1grid_size - 1] = u_state[3][jj][x1grid_size - 2];
                }
            }
        }

    } else {
        int i_shift;
        for (int jj = 0; jj < x2grid_size; jj++){

            // Fix the ghost zones at the radial boundaries
            u_state[0][jj][0] = u_state[0][jj][3];
            u_state[0][jj][1] = u_state[0][jj][2];
            u_state[0][jj][x1grid_size - 1] = u_state[0][jj][x1grid_size - 3];
            u_state[0][jj][x1grid_size - 2] = u_state[0][jj][x1grid_size - 3];

            u_state[1][jj][0] = - u_state[1][jj][3];
            u_state[1][jj][1] = - u_state[1][jj][2];
            u_state[1][jj][x1grid_size - 1] = u_state[1][jj][x1grid_size - 3];
            u_state[1][jj][x1grid_size - 2] = u_state[1][jj][x1grid_size - 3];

            u_state[2][jj][0] = u_state[2][jj][3];
            u_state[2][jj][1] = u_state[2][jj][2];
            u_state[2][jj][x1grid_size - 1] = u_state[2][jj][x1grid_size - 3];
            u_state[2][jj][x1grid_size - 2] = u_state[2][jj][x1grid_size - 3];

            u_state[3][jj][0] = u_state[3][jj][3];
            u_state[3][jj][1] = u_state[3][jj][2];
            u_state[3][jj][x1grid_size - 1] = u_state[3][jj][x1grid_size - 3];
            u_state[3][jj][x1grid_size - 2] = u_state[3][jj][x1grid_size - 3];

            // Fix the ghost zones at the angular boundaries
            for (int ii = 0; ii < x1grid_size; ii++){
                if (jj < 2){
                    if (jj == 0){
                        u_state[0][jj][ii] =   u_state[0][3][ii];
                        u_state[1][jj][ii] =   u_state[1][3][ii];
                        u_state[2][jj][ii] = - u_state[2][3][ii];
                        u_state[3][jj][ii] =   u_state[3][3][ii];
                    } else {
                        u_state[0][jj][ii] =   u_state[0][2][ii];
                        u_state[1][jj][ii] =   u_state[1][2][ii];
                        u_state[2][jj][ii] = - u_state[2][2][ii];
                        u_state[3][jj][ii] =   u_state[3][2][ii];
                    }
                    
                } else if (jj > x2grid_size - 3) {
                    if (jj == x2grid_size - 1){
                        u_state[0][jj][ii] =   u_state[0][x2grid_size - 4][ii];
                        u_state[1][jj][ii] =   u_state[1][x2grid_size - 4][ii];
                        u_state[2][jj][ii] = - u_state[2][x2grid_size - 4][ii];
                        u_state[3][jj][ii] =   u_state[3][x2grid_size - 4][ii];
                    } else {
                        u_state[0][jj][ii] =   u_state[0][x2grid_size - 3][ii];
                        u_state[1][jj][ii] =   u_state[1][x2grid_size - 3][ii];
                        u_state[2][jj][ii] = - u_state[2][x2grid_size - 3][ii];
                        u_state[3][jj][ii] =   u_state[3][x2grid_size - 3][ii];
                    }

                }
            }
        }

    }
};
*/

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

    h5::ds_t ds = h5::write(fd,"Simulation Data", prims.rho);
    
    // Datset objects
    ds["rho"] = prims.rho;
    ds["v1"]  = prims.v1;
    ds["v2"]  = prims.v2;
    ds["p"]   = prims.p;

    // Dataset Attributes
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

    // Write the Current Simulation Conditions in a File
    // h5::write( filename, "rho",  prims.rho);
    // h5::write( filename, "v1",   prims.v1 );
    // h5::write( filename, "v2",   prims.v2 );
    // h5::write( filename, "p",    prims.p  );


    // h5::write( filename, "t" , t );
    // h5::write( filename, "dt", dt );
    // h5::write( filename, "NX", NX );
    // h5::write( filename, "NY", NY );
    
    
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
    return ( tau + D*(1 - lorentz_gamma) + (1- lorentz_gamma*lorentz_gamma)*pressure )/(D*lorentz_gamma);
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
  
    double quad = -0.5*(b + sign(b)*disc);
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

    double rho = rho_rel(D, lorentz_gamma, 1);
    double epsilon = epsilon_rel(pressure, D, tau, lorentz_gamma);

    return (gamma - 1)*rho*epsilon - pressure;
}

double dfdp(double pressure, double D, double tau, double lorentz_gamma, float gamma, double S){

    double cs = calc_rel_sound_speed(pressure, D, tau, lorentz_gamma, gamma);
    double v = S/(tau + D + pressure);

    return v*v*cs*cs - 1.;
}
