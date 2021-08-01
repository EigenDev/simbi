/*
 * C++ Source to perform 2D SRHD Calculations
 * Marcus DuPont
 * New York University
 * 07/15/2020
 * Compressible Hydro Simulation
 */

#include "helper_functions.h"
#include "srhd_2d.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>

using namespace std;
using namespace simbi;
using namespace chrono;

// Calculate a static PI
constexpr double pi() { return std::atan(1)*4; }
constexpr double K = 0.0;
constexpr double a = 1e-3;

bool strong_shock(double pl, double pr){
    return std::abs(pr - pl) / std::min(pl, pr) > a;
}
// Default Constructor
SRHD2D::SRHD2D() {}

// Overloaded Constructor
SRHD2D::SRHD2D(vector<vector<double>> state2D, int nx, int ny, double gamma,
               vector<double> x1, vector<double> x2, double Cfl,
               string coord_system = "cartesian")
{
    auto nzones = state2D[0].size();

    this->NX = nx;
    this->NY = ny;
    this->nzones = nzones;
    this->state2D = state2D;
    this->gamma = gamma;
    this->x1 = x1;
    this->x2 = x2;
    this->CFL = Cfl;
    this->coord_system = coord_system;
}

// Destructor
SRHD2D::~SRHD2D() {}

/* Define typedefs because I am lazy */
typedef sr2d::Primitive Primitive;
typedef sr2d::Conserved Conserved;
typedef sr2d::Eigenvals Eigenvals;

//-----------------------------------------------------------------------------------------
//                          GET THE Primitive
//-----------------------------------------------------------------------------------------

vector<Primitive> SRHD2D::cons2prim2D(const vector<Conserved> &u_state2D)
{
    /**
   * Return a 2D matrix containing the primitive
   * variables density , pressure, and
   * three-velocity
   */

    double S1, S2, S, D, tau, tol;
    double W, v1, v2;

    vector<Primitive> prims;
    prims.reserve(nzones);

    // Define Newton-Raphson Vars
    double etotal, c2, f, g, p, peq;
    double Ws, rhos, eps, h;

    int iter = 0;
    int maximum_iteration = 50;
    for (int jj = 0; jj < NY; jj++)
    {
        for (int ii = 0; ii < NX; ii++)
        {
            D   = u_state2D [ii + NX * jj].D;     // Relativistic Mass Density
            S1  = u_state2D [ii + NX * jj].S1;   // X1-Momentum Denity
            S2  = u_state2D [ii + NX * jj].S2;   // X2-Momentum Density
            tau = u_state2D [ii + NX * jj].tau; // Energy Density
            S = sqrt(S1 * S1 + S2 * S2);

            peq = (n != 0.0) ? pressure_guess[ii + NX * jj] : abs(S - D - tau);

            tol = D * 1.e-12;

            //--------- Iteratively Solve for Pressure using Newton-Raphson
            // Note: The NR scheme can be modified based on:
            // https://www.sciencedirect.com/science/article/pii/S0893965913002930
            iter = 0;
            do
            {
                p = peq;
                etotal = tau + p + D;
                v2 = S * S / (etotal * etotal);
                Ws = 1.0 / sqrt(1.0 - v2);
                rhos = D / Ws;
                eps = (tau + D * (1. - Ws) + (1. - Ws * Ws) * p) / (D * Ws);
                f = (gamma - 1.0) * rhos * eps - p;

                h = 1. + eps + p / rhos;
                c2 = gamma * p / (h * rhos);
                g = c2 * v2 - 1.0;
                peq = p - f / g;
                iter++;

                if (iter > maximum_iteration)
                {
                    cout << "\n";
                    cout << "p: " << p << endl;
                    cout << "S: " << S << endl;
                    cout << "tau: " << tau << endl;
                    cout << "D: " << D << endl;
                    cout << "et: " << etotal << endl;
                    cout << "Ws: " << Ws << endl;
                    cout << "v2: " << v2 << endl;
                    cout << "W: " << W << endl;
                    cout << "n: " << n << endl;
                    cout << "\n Cons2Prim Cannot Converge" << endl;
                    exit(EXIT_FAILURE);
                }

            } while (abs(peq - p) >= tol);
        

            v1 = S1 / (tau + D + peq);
            v2 = S2 / (tau + D + peq);
            Ws = 1.0 / sqrt(1.0 - (v1 * v1 + v2 * v2));

            // Update the Gamma array
            // lorentz_gamma[ii + NX * jj] = Ws;

            // Update the pressure guess for the next time step
            pressure_guess[ii + NX * jj] = peq;

            prims.push_back(Primitive(D / Ws, v1, v2, peq));
        }
    }

    return prims;
};

//----------------------------------------------------------------------------------------------------------
//                              EIGENVALUE CALCULATIONS
//----------------------------------------------------------------------------------------------------------
Eigenvals SRHD2D::calc_Eigenvals(const Primitive &prims_l,
                                 const Primitive &prims_r,
                                 const unsigned int nhat = 1)
{
    // Eigenvals lambda;

    // Separate the left and right Primitive
    const double rho_l = prims_l.rho;
    const double p_l = prims_l.p;
    const double h_l = 1. + gamma * p_l / (rho_l * (gamma - 1));

    const double rho_r = prims_r.rho;
    const double p_r = prims_r.p;
    const double h_r = 1. + gamma * p_r / (rho_r * (gamma - 1));

    const double cs_r = sqrt(gamma * p_r / (h_r * rho_r));
    const double cs_l = sqrt(gamma * p_l / (h_l * rho_l));

    switch (nhat)
    {
    case 1:
    {
        const double v1_l = prims_l.v1;
        const double v1_r = prims_r.v1;

        //-----------Calculate wave speeds based on Shneider et al. 1992
        const double vbar  = 0.5 * (v1_l + v1_r);
        const double cbar  = 0.5 * (cs_l + cs_r);
        const double bl    = (vbar - cbar)/(1. - cbar*vbar);
        const double br    = (vbar + cbar)/(1. + cbar*vbar);
        const double aL = min(bl, (v1_l - cs_l)/(1. - v1_l*cs_l));
        const double aR = max(br, (v1_r + cs_r)/(1. + v1_r*cs_r));

        return Eigenvals(aL, aR);

        //--------Calc the wave speeds based on Mignone and Bodo (2005)
        // const double sL = cs_l * cs_l * (1. / (gamma * gamma * (1 - cs_l * cs_l)));
        // const double sR = cs_r * cs_r * (1. / (gamma * gamma * (1 - cs_r * cs_r)));

        // Define temporaries to save computational cycles
        // const double qfL = 1. / (1. + sL);
        // const double qfR = 1. / (1. + sR);
        // const double sqrtR = sqrt(sL * (1 - v1_l * v1_l + sL));
        // const double sqrtL = sqrt(sR * (1 - v1_r * v1_r + sL));

        // const double lamLm = (v1_l - sqrtL) * qfL;
        // const double lamRm = (v1_r - sqrtR) * qfR;
        // const double lamRp = (v1_l + sqrtL) * qfL;
        // const double lamLp = (v1_r + sqrtR) * qfR;

        // const double aL = lamLm < lamRm ? lamLm : lamRm;
        // const double aR = lamLp > lamRp ? lamLp : lamRp;

        // return Eigenvals(aL, aR);
    }
    case 2:
        const double v2_r = prims_r.v2;
        const double v2_l = prims_l.v2;

        //-----------Calculate wave speeds based on Shneider et al. 1992
        const double vbar  = 0.5 * (v2_l + v2_r);
        const double cbar  = 0.5 * (cs_l + cs_r);
        const double bl    = (vbar - cbar)/(1. - cbar*vbar);
        const double br    = (vbar + cbar)/(1. + cbar*vbar);
        const double aL = min(bl, (v2_l - cs_l)/(1. - v2_l*cs_l));
        const double aR = max(br, (v2_r + cs_r)/(1. + v2_r*cs_r));

        return Eigenvals(aL, aR);

        // Calc the wave speeds based on Mignone and Bodo (2005)
        // double sL = cs_l * cs_l * (1.0 / (gamma * gamma * (1 - cs_l * cs_l)));
        // double sR = cs_r * cs_r * (1.0 / (gamma * gamma * (1 - cs_r * cs_r)));

        // Define some temporaries to save a few cycles
        // const double qfL = 1. / (1. + sL);
        // const double qfR = 1. / (1. + sR);
        // const double sqrtR = sqrt(sL * (1 - v2_l * v2_l + sL));
        // const double sqrtL = sqrt(sR * (1 - v2_r * v2_r + sL));

        // const double lamLm = (v2_l - sqrtL) * qfL;
        // const double lamRm = (v2_r - sqrtR) * qfR;
        // const double lamRp = (v2_l + sqrtL) * qfL;
        // const double lamLp = (v2_r + sqrtR) * qfR;
        // const double aL = lamLm < lamRm ? lamLm : lamRm;
        // const double aR = lamLp > lamRp ? lamLp : lamRp;

        // return Eigenvals(aL, aR);
    }
};

//-----------------------------------------------------------------------------------------
//                              CALCULATE THE STATE ARRAY
//-----------------------------------------------------------------------------------------

Conserved SRHD2D::calc_stateSR2D(const Primitive &prims)
{
    const double rho = prims.rho;
    const double vx = prims.v1;
    const double vy = prims.v2;
    const double pressure = prims.p;
    const double lorentz_gamma = 1. / sqrt(1 - (vx * vx + vy * vy));
    const double h = 1. + gamma * pressure / (rho * (gamma - 1.));

    return Conserved{
        rho * lorentz_gamma, rho * h * lorentz_gamma * lorentz_gamma * vx,
        rho * h * lorentz_gamma * lorentz_gamma * vy,
        rho * h * lorentz_gamma * lorentz_gamma - pressure - rho * lorentz_gamma};
};

Conserved SRHD2D::calc_intermed_statesSR2D(const Primitive &prims,
                                           const Conserved &state, double a,
                                           double aStar, double pStar,
                                           int nhat = 1)
{
    double Dstar, S1star, S2star, tauStar, Estar, cofactor;
    Conserved starStates;

    double pressure = prims.p;
    double v1 = prims.v1;
    double v2 = prims.v2;

    double D = state.D;
    double S1 = state.S1;
    double S2 = state.S2;
    double tau = state.tau;
    double E = tau + D;

    switch (nhat)
    {
    case 1:
        cofactor = 1. / (a - aStar);
        Dstar = cofactor * (a - v1) * D;
        S1star = cofactor * (S1 * (a - v1) - pressure + pStar);
        S2star = cofactor * (a - v1) * S2;
        Estar = cofactor * (E * (a - v1) + pStar * aStar - pressure * v1);
        tauStar = Estar - Dstar;

        starStates = Conserved(Dstar, S1star, S2star, tauStar);

        return starStates;
    case 2:
        cofactor = 1. / (a - aStar);
        Dstar = cofactor * (a - v2) * D;
        S1star = cofactor * (a - v2) * S1;
        S2star = cofactor * (S2 * (a - v2) - pressure + pStar);
        Estar = cofactor * (E * (a - v2) + pStar * aStar - pressure * v2);
        tauStar = Estar - Dstar;

        starStates = Conserved(Dstar, S1star, S2star, tauStar);

        return starStates;
    }

    return starStates;
}

//---------------------------------------------------------------------
//                  ADAPT THE TIMESTEP
//---------------------------------------------------------------------

// Adapt the CFL conditonal timestep
double SRHD2D::adapt_dt(const vector<Primitive> &prims)
{

    double r_left, r_right, left_cell, right_cell, lower_cell, upper_cell;
    double dx1, cs, dx2, x2_right, x2_left, rho, pressure, v1, v2, volAvg, h;
    double min_dt, cfl_dt;
    int shift_i, shift_j;
    double plus_v1, plus_v2, minus_v1, minus_v2;

    min_dt = 0;
    // Compute the minimum timestep given CFL
    for (int jj = 0; jj < yphysical_grid; jj++)
    {
        shift_j  = jj + idx_active;
        x2_right = coord_lattice.x2vertices[jj + 1];
        x2_left  = coord_lattice.x2vertices[jj];
        dx2 = x2_right - x2_left;
        for (int ii = 0; ii < xphysical_grid; ii++)
        {
            
            shift_i = ii + idx_active;

            r_right = coord_lattice.x1vertices[ii + 1];
            r_left = coord_lattice.x1vertices[ii];

            dx1 = r_right - r_left;
            rho = prims[shift_i + NX * shift_j].rho;
            v1  = prims[shift_i + NX * shift_j].v1;
            v2  = prims[shift_i + NX * shift_j].v2;
            pressure = prims[shift_i + NX * shift_j].p;

            h = 1. + gamma * pressure / (rho * (gamma - 1.));
            cs = sqrt(gamma * pressure / (rho * h));

            plus_v1 = (v1 + cs) / (1. + v1 * cs);
            plus_v2 = (v2 + cs) / (1. + v2 * cs);
            minus_v1 = (v1 - cs) / (1. - v1 * cs);
            minus_v2 = (v2 - cs) / (1. - v2 * cs);

            if (coord_system == "cartesian")
            {

                cfl_dt = min(dx1 / (max(abs(plus_v1), abs(minus_v1))),
                             dx2 / (max(abs(plus_v2), abs(minus_v2))));
            }
            else
            {
                // Compute avg spherical distance 3/4 *(rf^4 - ri^4)/(rf^3 - ri^3)
                volAvg = coord_lattice.x1mean[ii];
                // std::cout << volAvg << "\n";
                // std::cout << dx1 << "\n";
                // std::cout << dx2 << "\n";
                // std::cout << volAvg * dx2 << "\n";
                // std::cin.get();
                cfl_dt = min(dx1 / (max(abs(plus_v1), abs(minus_v1))),
                             volAvg * dx2 / (max(abs(plus_v2), abs(minus_v2))));
            }

            if ((ii > 0) || (jj > 0))
            {
                min_dt = min_dt < cfl_dt ? min_dt : cfl_dt;
            }
            else
            {
                min_dt = cfl_dt;
            }
        }
    }
    return CFL * min_dt;
};

//===================================================================================================================
//                                            FLUX CALCULATIONS
//===================================================================================================================

// Get the 2D Flux array (4,1). Either return F or G depending on directional
// flag
Conserved SRHD2D::calc_Flux(const Primitive &prims, unsigned int nhat = 1)
{

    const double rho = prims.rho;
    const double vx = prims.v1;
    const double vy = prims.v2;
    const double pressure = prims.p;
    const double lorentz_gamma = 1. / sqrt(1. - (vx * vx + vy * vy));

    const double h = 1. + gamma * pressure / (rho * (gamma - 1));
    const double D = rho * lorentz_gamma;
    const double S1 = rho * lorentz_gamma * lorentz_gamma * h * vx;
    const double S2 = rho * lorentz_gamma * lorentz_gamma * h * vy;
    const double tau =
                    rho * h * lorentz_gamma * lorentz_gamma - pressure - rho * lorentz_gamma;

    return (nhat == 1) ? Conserved(D * vx, S1 * vx + pressure, S2 * vx,
                                   (tau + pressure) * vx)
                       : Conserved(D * vy, S1 * vy, S2 * vy + pressure,
                                   (tau + pressure) * vy);
};

Conserved SRHD2D::calc_hll_flux(
    const Conserved &left_state, 
    const Conserved &right_state,
    const Conserved &left_flux, 
    const Conserved &right_flux,
    const Primitive &left_prims, 
    const Primitive &right_prims,
    const unsigned int nhat)
{
    Eigenvals lambda = calc_Eigenvals(left_prims, right_prims, nhat);

    const double aL = lambda.aL;
    const double aR = lambda.aR;

    // Calculate plus/minus alphas
    const double aLminus = aL < 0.0 ? aL : 0.0;
    const double aRplus  = aR > 0.0 ? aR : 0.0;

    // Compute the HLL Flux component-wise
    return (left_flux * aRplus - right_flux * aLminus 
                + (right_state - left_state) * aRplus * aLminus) /
                    (aRplus - aLminus);
};

Conserved SRHD2D::calc_hllc_flux(
    const Conserved &left_state,
    const Conserved &right_state,
    const Conserved &left_flux,
    const Conserved &right_flux,
    const Primitive &left_prims,
    const Primitive &right_prims,
    const unsigned int nhat = 1)
{

    Eigenvals lambda = calc_Eigenvals(left_prims, right_prims, nhat);

    const double aL = lambda.aL;
    const double aR = lambda.aR;

    //---- Check Wave Speeds before wasting computations
    if (0.0 <= aL)
    {
        return left_flux;
    }
    else if (0.0 >= aR)
    {
        return right_flux;
    }

    const double aLminus = aL < 0.0 ? aL : 0.0;
    const double aRplus  = aR > 0.0 ? aR : 0.0;

    //-------------------Calculate the HLL Intermediate State
    const auto hll_state = 
        (right_state * aR - left_state * aL - right_flux + left_flux) / (aR - aL);

    //------------------Calculate the RHLLE Flux---------------
    const auto hll_flux 
        = (left_flux * aRplus - right_flux * aLminus + (right_state - left_state) * aRplus * aLminus) 
            / (aRplus - aLminus);

    //------ Mignone & Bodo subtract off the rest mass density
    const double e  = hll_state.tau + hll_state.D;
    const double s  = hll_state.momentum(nhat);
    const double fe = hll_flux.tau + hll_flux.D;
    const double fs = hll_flux.momentum(nhat);

    //------Calculate the contact wave velocity and pressure
    const double a = fe;
    const double b = -(e + fs);
    const double c = s;
    const double quad = -0.5 * (b + sgn(b) * sqrt(b * b - 4.0 * a * c));
    const double aStar = c * (1.0 / quad);
    const double pStar = -aStar * fe + fs;

    // return Conserved(0.0, 0.0, 0.0, 0.0);
    if (-aL <= (aStar - aL))
    {
        const double pressure = left_prims.p;
        const double D = left_state.D;
        const double S1 = left_state.S1;
        const double S2 = left_state.S2;
        const double tau = left_state.tau;
        const double E = tau + D;
        const double cofactor = 1. / (aL - aStar);
        //--------------Compute the L Star State----------
        switch (nhat)
        {
        case 1:
        {
            const double v1 = left_prims.v1;
            // Left Star State in x-direction of coordinate lattice
            const double Dstar    = cofactor * (aL - v1) * D;
            const double S1star   = cofactor * (S1 * (aL - v1) - pressure + pStar);
            const double S2star   = cofactor * (aL - v1) * S2;
            const double Estar    = cofactor * (E * (aL - v1) + pStar * aStar - pressure * v1);
            const double tauStar  = Estar - Dstar;

            const auto interstate_left = Conserved(Dstar, S1star, S2star, tauStar);

            //---------Compute the L Star Flux
            return left_flux + (interstate_left - left_state) * aL;
        }

        case 2:
            const double v2 = left_prims.v2;
            // Start States in y-direction in the coordinate lattice
            const double Dstar   = cofactor * (aL - v2) * D;
            const double S1star  = cofactor * (aL - v2) * S1;
            const double S2star  = cofactor * (S2 * (aL - v2) - pressure + pStar);
            const double Estar   = cofactor * (E * (aL - v2) + pStar * aStar - pressure * v2);
            const double tauStar = Estar - Dstar;

            const auto interstate_left = Conserved(Dstar, S1star, S2star, tauStar);

            //---------Compute the L Star Flux
            return left_flux + (interstate_left - left_state) * aL;
        }
    }
    else
    {
        const double pressure = right_prims.p;
        const double D = right_state.D;
        const double S1 = right_state.S1;
        const double S2 = right_state.S2;
        const double tau = right_state.tau;
        const double E = tau + D;
        const double cofactor = 1. / (aR - aStar);

        /* Compute the L/R Star State */
        switch (nhat)
        {
        case 1:
        {
            const double v1 = right_prims.v1;
            const double Dstar = cofactor * (aR - v1) * D;
            const double S1star = cofactor * (S1 * (aR - v1) - pressure + pStar);
            const double S2star = cofactor * (aR - v1) * S2;
            const double Estar = cofactor * (E * (aR - v1) + pStar * aStar - pressure * v1);
            const double tauStar = Estar - Dstar;

            const auto interstate_right = Conserved(Dstar, S1star, S2star, tauStar);

            // Compute the intermediate right flux
            return right_flux + (interstate_right - right_state) * aR;
        }

        case 2:
            const double v2 = right_prims.v2;
            // Start States in y-direction in the coordinate lattice
            const double cofactor = 1. / (aR - aStar);
            const double Dstar = cofactor * (aR - v2) * D;
            const double S1star = cofactor * (aR - v2) * S1;
            const double S2star = cofactor * (S2 * (aR - v2) - pressure + pStar);
            const double Estar = cofactor * (E * (aR - v2) + pStar * aStar - pressure * v2);
            const double tauStar = Estar - Dstar;

            const auto interstate_right = Conserved(Dstar, S1star, S2star, tauStar);

            // Compute the intermediate right flux
            return right_flux + (interstate_right - right_state) * aR;
        }
    }
};

//===================================================================================================================
//                                            UDOT CALCULATIONS
//===================================================================================================================

vector<Conserved> SRHD2D::u_dot2D(const vector<Conserved> &u_state)
{
    int xcoordinate, ycoordinate;

    vector<Conserved> L;
    L.reserve(active_zones);

    Conserved ux_l, ux_r, uy_l, uy_r;
    Conserved f_l, f_r, f1, f2, g1, g2, g_l, g_r;
    Primitive xprims_l, xprims_r, yprims_l, yprims_r;

    Primitive xleft_most, xleft_mid, xright_mid, xright_most;
    Primitive yleft_most, yleft_mid, yright_mid, yright_most;
    Primitive center;

    // The periodic BC doesn't require ghost cells. Shift the index
    // to the beginning.
    const int i_start = idx_active;
    const int j_start = idx_active;
    const int i_bound = x_bound;
    const int j_bound = y_bound;

    switch (geometry[coord_system])
    {
    case Geometry::CARTESIAN:
    {
        double dx = (x1[xphysical_grid - 1] - x1[0]) / xphysical_grid;
        double dy = (x2[yphysical_grid - 1] - x2[0]) / yphysical_grid;
        if (first_order)
        {
            for (int jj = j_start; jj < j_bound; jj++)
            {
                for (int ii = i_start; ii < i_bound; ii++)
                {
                    ycoordinate = jj - 1;
                    xcoordinate = ii - 1;

                    // i+1/2
                    ux_l = u_state[ii + NX * jj];
                    ux_r = u_state[(ii + 1) + NX * jj];

                    // j+1/2
                    uy_l = u_state[ii + NX * jj];
                    uy_r = u_state[(ii + 1) + NX * jj];

                    xprims_l = prims[ii + jj * NX];
                    xprims_r = prims[(ii + 1) + jj * NX];

                    yprims_l = prims[ii + jj * NX];

                    yprims_r = prims[ii + (jj + 1) * NX];

                    f_l = calc_Flux(xprims_l);
                    f_r = calc_Flux(xprims_r);

                    g_l = calc_Flux(yprims_l, 2);
                    g_r = calc_Flux(yprims_r, 2);

                    // Calc HLL Flux at i+1/2 interface
                    f1 = calc_hll_flux(ux_l, ux_r, f_l, f_r, xprims_l, xprims_r, 1);
                    g1 = calc_hll_flux(uy_l, uy_r, g_l, g_r, yprims_l, yprims_r, 2);

                    // Set up the left and right state interfaces for i-1/2

                    // i-1/2
                    ux_l = u_state[(ii - 1) + NX * jj];
                    ux_r = u_state[ii + NX * jj];

                    // j-1/2
                    uy_l = u_state[(ii - 1) + NX * jj];
                    uy_r = u_state[ii + NX * jj];

                    xprims_l = prims[(ii - 1) + jj * NX];
                    xprims_r = prims[ii + jj * NX];

                    yprims_l = prims[ii + (jj - 1) * NX];
                    yprims_r = prims[ii + jj * NX];

                    f_l = calc_Flux(xprims_l);
                    f_r = calc_Flux(xprims_r);

                    g_l = calc_Flux(yprims_l, 2);
                    g_r = calc_Flux(yprims_r, 2);

                    // Calc HLL Flux at i+1/2 interface
                    f2 = calc_hll_flux(ux_l, ux_r, f_l, f_r, xprims_l, xprims_r, 1);
                    g2 = calc_hll_flux(uy_l, uy_r, g_l, g_r, yprims_l, yprims_r, 2);

                    L.push_back(Conserved{(f1 - f2) * -1.0 / dx - (g1 - g2) / dy});
                }
            }
        }
        else
        {
            for (int jj = j_start; jj < j_bound; jj++)
            {
                for (int ii = i_start; ii < i_bound; ii++)
                {
                    if (periodic)
                    {
                        xcoordinate = ii;
                        ycoordinate = jj;

                        // Declare the c[i-2],c[i-1],c_i,c[i+1], c[i+2] variables

                        /* TODO: Poplate this later */
                    }
                    else
                    {
                        // Adjust for beginning input of L vector
                        xcoordinate = ii - 2;
                        ycoordinate = jj - 2;

                        // Coordinate X
                        xleft_most = prims[(ii - 2) + NX * jj];
                        xleft_mid = prims[(ii - 1) + NX * jj];
                        center = prims[ii + NX * jj];
                        xright_mid = prims[(ii + 1) + NX * jj];
                        xright_most = prims[(ii + 2) + NX * jj];

                        // Coordinate Y
                        yleft_most = prims[ii + NX * (jj - 2)];
                        yleft_mid = prims[ii + NX * (jj - 1)];
                        yright_mid = prims[ii + NX * (jj + 1)];
                        yright_most = prims[ii + NX * (jj + 2)];
                    }

                    // Reconstructed left X Primitive vector at the i+1/2 interface
                    xprims_l.rho =
                        center.rho + 0.5 * minmod(theta * (center.rho - xleft_mid.rho),
                                                  0.5 * (xright_mid.rho - xleft_mid.rho),
                                                  theta * (xright_mid.rho - center.rho));

                    xprims_l.v1 =
                        center.v1 + 0.5 * minmod(theta * (center.v1 - xleft_mid.v1),
                                                 0.5 * (xright_mid.v1 - xleft_mid.v1),
                                                 theta * (xright_mid.v1 - center.v1));

                    xprims_l.v2 =
                        center.v2 + 0.5 * minmod(theta * (center.v2 - xleft_mid.v2),
                                                 0.5 * (xright_mid.v2 - xleft_mid.v2),
                                                 theta * (xright_mid.v2 - center.v2));

                    xprims_l.p =
                        center.p + 0.5 * minmod(theta * (center.p - xleft_mid.p),
                                                0.5 * (xright_mid.p - xleft_mid.p),
                                                theta * (xright_mid.p - center.p));

                    // Reconstructed right Primitive vector in x
                    xprims_r.rho =
                        xright_mid.rho -
                        0.5 * minmod(theta * (xright_mid.rho - center.rho),
                                     0.5 * (xright_most.rho - center.rho),
                                     theta * (xright_most.rho - xright_mid.rho));

                    xprims_r.v1 = xright_mid.v1 -
                                  0.5 * minmod(theta * (xright_mid.v1 - center.v1),
                                               0.5 * (xright_most.v1 - center.v1),
                                               theta * (xright_most.v1 - xright_mid.v1));

                    xprims_r.v2 = xright_mid.v2 -
                                  0.5 * minmod(theta * (xright_mid.v2 - center.v2),
                                               0.5 * (xright_most.v2 - center.v2),
                                               theta * (xright_most.v2 - xright_mid.v2));

                    xprims_r.p = xright_mid.p -
                                 0.5 * minmod(theta * (xright_mid.p - center.p),
                                              0.5 * (xright_most.p - center.p),
                                              theta * (xright_most.p - xright_mid.p));

                    // Reconstructed right Primitive vector in y-direction at j+1/2
                    // interfce
                    yprims_l.rho =
                        center.rho + 0.5 * minmod(theta * (center.rho - yleft_mid.rho),
                                                  0.5 * (yright_mid.rho - yleft_mid.rho),
                                                  theta * (yright_mid.rho - center.rho));

                    yprims_l.v1 =
                        center.v1 + 0.5 * minmod(theta * (center.v1 - yleft_mid.v1),
                                                 0.5 * (yright_mid.v1 - yleft_mid.v1),
                                                 theta * (yright_mid.v1 - center.v1));

                    yprims_l.v2 =
                        center.v2 + 0.5 * minmod(theta * (center.v2 - yleft_mid.v2),
                                                 0.5 * (yright_mid.v2 - yleft_mid.v2),
                                                 theta * (yright_mid.v2 - center.v2));

                    yprims_l.p =
                        center.p + 0.5 * minmod(theta * (center.p - yleft_mid.p),
                                                0.5 * (yright_mid.p - yleft_mid.p),
                                                theta * (yright_mid.p - center.p));

                    yprims_r.rho =
                        yright_mid.rho -
                        0.5 * minmod(theta * (yright_mid.rho - center.rho),
                                     0.5 * (yright_most.rho - center.rho),
                                     theta * (yright_most.rho - yright_mid.rho));

                    yprims_r.v1 = yright_mid.v1 -
                                  0.5 * minmod(theta * (yright_mid.v1 - center.v1),
                                               0.5 * (yright_most.v1 - center.v1),
                                               theta * (yright_most.v1 - yright_mid.v1));

                    yprims_r.v2 = yright_mid.v2 -
                                  0.5 * minmod(theta * (yright_mid.v2 - center.v2),
                                               0.5 * (yright_most.v2 - center.v2),
                                               theta * (yright_most.v2 - yright_mid.v2));

                    yprims_r.p = yright_mid.p -
                                 0.5 * minmod(theta * (yright_mid.p - center.p),
                                              0.5 * (yright_most.p - center.p),
                                              theta * (yright_most.p - yright_mid.p));

                    // Calculate the left and right states using the reconstructed PLM
                    // Primitive
                    ux_l = calc_stateSR2D(xprims_l);
                    ux_r = calc_stateSR2D(xprims_r);

                    uy_l = calc_stateSR2D(yprims_l);
                    uy_r = calc_stateSR2D(yprims_r);

                    f_l = calc_Flux(xprims_l);
                    f_r = calc_Flux(xprims_r);

                    g_l = calc_Flux(yprims_l, 2);
                    g_r = calc_Flux(yprims_r, 2);

                    f1 = calc_hll_flux(ux_l, ux_r, f_l, f_r, xprims_l, xprims_r, 1);
                    g1 = calc_hll_flux(uy_l, uy_r, g_l, g_r, yprims_l, yprims_r, 2);

                    // Left side Primitive in x
                    xprims_l.rho = xleft_mid.rho +
                                   0.5 * minmod(theta * (xleft_mid.rho - xleft_most.rho),
                                                0.5 * (center.rho - xleft_most.rho),
                                                theta * (center.rho - xleft_mid.rho));

                    xprims_l.v1 = xleft_mid.v1 +
                                  0.5 * minmod(theta * (xleft_mid.v1 - xleft_most.v1),
                                               0.5 * (center.v1 - xleft_most.v1),
                                               theta * (center.v1 - xleft_mid.v1));

                    xprims_l.v2 = xleft_mid.v2 +
                                  0.5 * minmod(theta * (xleft_mid.v2 - xleft_most.v2),
                                               0.5 * (center.v2 - xleft_most.v2),
                                               theta * (center.v2 - xleft_mid.v2));

                    xprims_l.p =
                        xleft_mid.p + 0.5 * minmod(theta * (xleft_mid.p - xleft_most.p),
                                                   0.5 * (center.p - xleft_most.p),
                                                   theta * (center.p - xleft_mid.p));

                    // Right side Primitive in x
                    xprims_r.rho =
                        center.rho - 0.5 * minmod(theta * (center.rho - xleft_mid.rho),
                                                  0.5 * (xright_mid.rho - xleft_mid.rho),
                                                  theta * (xright_mid.rho - center.rho));

                    xprims_r.v1 =
                        center.v1 - 0.5 * minmod(theta * (center.v1 - xleft_mid.v1),
                                                 0.5 * (xright_mid.v1 - xleft_mid.v1),
                                                 theta * (xright_mid.v1 - center.v1));

                    xprims_r.v2 =
                        center.v2 - 0.5 * minmod(theta * (center.v2 - xleft_mid.v2),
                                                 0.5 * (xright_mid.v2 - xleft_mid.v2),
                                                 theta * (xright_mid.v2 - center.v2));

                    xprims_r.p =
                        center.p - 0.5 * minmod(theta * (center.p - xleft_mid.p),
                                                0.5 * (xright_mid.p - xleft_mid.p),
                                                theta * (xright_mid.p - center.p));

                    // Left side Primitive in y
                    yprims_l.rho = yleft_mid.rho +
                                   0.5 * minmod(theta * (yleft_mid.rho - yleft_most.rho),
                                                0.5 * (center.rho - yleft_most.rho),
                                                theta * (center.rho - yleft_mid.rho));

                    yprims_l.v1 = yleft_mid.v1 +
                                  0.5 * minmod(theta * (yleft_mid.v1 - yleft_most.v1),
                                               0.5 * (center.v1 - yleft_most.v1),
                                               theta * (center.v1 - yleft_mid.v1));

                    yprims_l.v2 = yleft_mid.v2 +
                                  0.5 * minmod(theta * (yleft_mid.v2 - yleft_most.v2),
                                               0.5 * (center.v2 - yleft_most.v2),
                                               theta * (center.v2 - yleft_mid.v2));

                    yprims_l.p =
                        yleft_mid.p + 0.5 * minmod(theta * (yleft_mid.p - yleft_most.p),
                                                   0.5 * (center.p - yleft_most.p),
                                                   theta * (center.p - yleft_mid.p));

                    // Right side Primitive in y
                    yprims_r.rho =
                        center.rho - 0.5 * minmod(theta * (center.rho - yleft_mid.rho),
                                                  0.5 * (yright_mid.rho - yleft_mid.rho),
                                                  theta * (yright_mid.rho - center.rho));

                    yprims_r.v1 =
                        center.v1 - 0.5 * minmod(theta * (center.v1 - yleft_mid.v1),
                                                 0.5 * (yright_mid.v1 - yleft_mid.v1),
                                                 theta * (yright_mid.v1 - center.v1));

                    yprims_r.v2 =
                        center.v2 - 0.5 * minmod(theta * (center.v2 - yleft_mid.v2),
                                                 0.5 * (yright_mid.v2 - yleft_mid.v2),
                                                 theta * (yright_mid.v2 - center.v2));

                    yprims_r.p =
                        center.p - 0.5 * minmod(theta * (center.p - yleft_mid.p),
                                                0.5 * (yright_mid.p - yleft_mid.p),
                                                theta * (yright_mid.p - center.p));

                    // Calculate the left and right states using the reconstructed PLM
                    // Primitive
                    ux_l = calc_stateSR2D(xprims_l);
                    ux_r = calc_stateSR2D(xprims_r);

                    uy_l = calc_stateSR2D(yprims_l);
                    uy_r = calc_stateSR2D(yprims_r);

                    f_l = calc_Flux(xprims_l);
                    f_r = calc_Flux(xprims_r);

                    g_l = calc_Flux(yprims_l, 2);
                    g_r = calc_Flux(yprims_r, 2);

                    f2 = calc_hll_flux(ux_l, ux_r, f_l, f_r, xprims_l, xprims_r, 1);
                    g2 = calc_hll_flux(uy_l, uy_r, g_l, g_r, yprims_l, yprims_r, 2);

                    L.push_back(Conserved{(f1 - f2) * -1.0 / dx - (g1 - g2) / dy});
                }
            }
        }

        break;
    }
    case Geometry::SPHERICAL:
        //=======================================================================================================================================================
        //                                  SPHERICAL
        //=======================================================================================================================================================
        double right_cell, left_cell, lower_cell, upper_cell, ang_avg;
        double r_left, r_right, volAvg, pc, rhoc, vc, uc, deltaV1, deltaV2;
        double theta_right, theta_left, ycoordinate, xcoordinate, hc, wc2;
        double upper_tsurface, lower_tsurface, right_rsurface, left_rsurface;

        if (first_order)
        {
            for (int jj = j_start; jj < j_bound; jj++)
            {
                ycoordinate = jj - 1;
                upper_tsurface = coord_lattice.x2_face_areas[ycoordinate + 1];
                lower_tsurface = coord_lattice.x2_face_areas[ycoordinate];
                for (int ii = i_start; ii < i_bound; ii++)
                {
                    ycoordinate = jj - 1;
                    xcoordinate = ii - 1;

                    // i+1/2
                    ux_l = u_state[ii + NX * jj];
                    ux_r = u_state[(ii + 1) + NX * jj];

                    // j+1/2
                    uy_l = u_state[ii + NX * jj];
                    uy_r = u_state[ii + NX * (jj + 1)];

                    xprims_l = prims[ii + jj * NX];
                    xprims_r = prims[(ii + 1) + jj * NX];
                    yprims_l = prims[ii + jj * NX];
                    yprims_r = prims[ii + (jj + 1) * NX];

                    // Get central values for spherical source terms
                    rhoc = xprims_l.rho;
                    pc   = xprims_l.p;
                    uc   = xprims_l.v1;
                    vc   = xprims_l.v2;

                    f_l = calc_Flux(xprims_l);
                    f_r = calc_Flux(xprims_r);

                    g_l = calc_Flux(yprims_l, 2);
                    g_r = calc_Flux(yprims_r, 2);

                    // Calc HLL Flux at i,j +1/2 interface
                    if (hllc)
                    {
                        f1 = calc_hllc_flux(ux_l, ux_r, f_l, f_r, xprims_l, xprims_r, 1);
                        g1 = calc_hllc_flux(uy_l, uy_r, g_l, g_r, yprims_l, yprims_r, 2);
                    }
                    else
                    {
                        f1 = calc_hll_flux(ux_l, ux_r, f_l, f_r, xprims_l, xprims_r, 1);
                        g1 = calc_hll_flux(uy_l, uy_r, g_l, g_r, yprims_l, yprims_r, 2);
                    }

                    // Set up the left and right state interfaces for i-1/2

                    // i-1/2
                    ux_l = u_state[(ii - 1) + NX * jj];
                    ux_r = u_state[ii + NX * jj];

                    // j-1/2
                    uy_l = u_state[ii + NX * (jj - 1)];
                    uy_r = u_state[ii + NX * jj];

                    xprims_l = prims[(ii - 1) + jj * NX];
                    xprims_r = prims[ii + jj * NX];
                    yprims_l = prims[ii + (jj - 1) * NX];
                    yprims_r = prims[ii + jj * NX];

                    f_l = calc_Flux(xprims_l);
                    f_r = calc_Flux(xprims_r);
                    g_l = calc_Flux(yprims_l, 2);
                    g_r = calc_Flux(yprims_r, 2);

                    // Calc HLL Flux at i,j - 1/2 interface
                    if (hllc)
                    {
                        f2 = calc_hllc_flux(ux_l, ux_r, f_l, f_r, xprims_l, xprims_r, 1);
                        g2 = calc_hllc_flux(uy_l, uy_r, g_l, g_r, yprims_l, yprims_r, 2);
                    }
                    else
                    {
                        f2 = calc_hll_flux(ux_l, ux_r, f_l, f_r, xprims_l, xprims_r, 1);
                        g2 = calc_hll_flux(uy_l, uy_r, g_l, g_r, yprims_l, yprims_r, 2);
                    }

                    right_rsurface = coord_lattice.x1_face_areas[xcoordinate + 1];
                    left_rsurface  = coord_lattice.x1_face_areas[xcoordinate];
                    volAvg         = coord_lattice.x1mean[xcoordinate];
                    deltaV1        = coord_lattice.dV1[xcoordinate];
                    deltaV2        = volAvg * coord_lattice.dV2[ycoordinate];
                    hc             = 1.0 + gamma * pc /(rhoc * (gamma - 1.0));
                    wc2            = 1.0/(1.0 - (uc * uc + vc * vc));


                    L.push_back(Conserved{
                        // L(D)
                        -(f1.D * right_rsurface - f2.D * left_rsurface) / deltaV1 
                            - (g1.D * upper_tsurface - g2.D * lower_tsurface) / deltaV2 
                                + sourceD[xcoordinate + xphysical_grid * ycoordinate] * decay_const,

                        // L(S1)
                        -(f1.S1 * right_rsurface - f2.S1 * left_rsurface) / deltaV1 
                            - (g1.S1 * upper_tsurface - g2.S1 * lower_tsurface) / deltaV2 
                                + rhoc * hc * wc2 * vc * vc / volAvg + 2 * pc / volAvg +
                                     source_S1[xcoordinate + xphysical_grid * ycoordinate] * decay_const,

                        // L(S2)
                        -(f1.S2 * right_rsurface - f2.S2 * left_rsurface) / deltaV1
                             - (g1.S2 * upper_tsurface - g2.S2 * lower_tsurface) / deltaV2 
                                - (rhoc * hc * wc2 * uc * vc / volAvg - pc * coord_lattice.cot[ycoordinate] / (volAvg)) 
                                    + source_S2[xcoordinate + xphysical_grid * ycoordinate] * decay_const,

                        // L(tau)
                        -(f1.tau * right_rsurface - f2.tau * left_rsurface) / deltaV1 
                            - (g1.tau * upper_tsurface - g2.tau * lower_tsurface) / deltaV2 
                                + source_tau[xcoordinate + xphysical_grid * ycoordinate] * decay_const
                    });
                }
            }
        }
        else
        {
            bool at_north_pole = false; 
            bool at_south_pole = false; 
            bool at_adjn_pole  = false;
            bool at_adjs_pole  = false;
            bool zero_flux     = false;
            bool zero_flux_north     = false;
            bool zero_flux_south     = false;
            auto null_flux = Conserved{0.0, 0.0, 0.0, 0.0};

            // Left/Right artificial viscosity
            Conserved favl, favr;
            for (int jj = j_start; jj < j_bound; jj++)
            {
                ycoordinate = jj - 2;
                upper_tsurface = coord_lattice.x2_face_areas[(ycoordinate + 1)];
                lower_tsurface = coord_lattice.x2_face_areas[(ycoordinate + 0)];
                for (int ii = i_start; ii < i_bound; ii++)
                {
                    if (!periodic)
                    {
                        // Adjust for beginning input of L vector
                        xcoordinate = ii - 2;

                        // Coordinate X
                        xleft_most  = prims[(ii - 2) + NX * jj];
                        xleft_mid   = prims[(ii - 1) + NX * jj];
                        center      = prims[ ii      + NX * jj];
                        xright_mid  = prims[(ii + 1) + NX * jj];
                        xright_most = prims[(ii + 2) + NX * jj];

                        // Coordinate Y
                        yleft_most  = prims[ii + NX * (jj - 2)];
                        yleft_mid   = prims[ii + NX * (jj - 1)];
                        yright_mid  = prims[ii + NX * (jj + 1)];
                        yright_most = prims[ii + NX * (jj + 2)];
                    }
                    else
                    {
                        xcoordinate = ii;
                        ycoordinate = jj;

                        // Declare the c[i-2],c[i-1],c_i,c[i+1], c[i+2] variables

                        /* TODO: Fix this */
                    }
                    // Reconstructed left X Primitive vector at the i+1/2 interface
                    xprims_l.rho =
                        center.rho + 0.5 * minmod(theta * (center.rho - xleft_mid.rho),
                                                  0.5 * (xright_mid.rho - xleft_mid.rho),
                                                  theta * (xright_mid.rho - center.rho));

                    xprims_l.v1 =
                        center.v1 + 0.5 * minmod(theta * (center.v1 - xleft_mid.v1),
                                                 0.5 * (xright_mid.v1 - xleft_mid.v1),
                                                 theta * (xright_mid.v1 - center.v1));

                    xprims_l.v2 =
                        center.v2 + 0.5 * minmod(theta * (center.v2 - xleft_mid.v2),
                                                 0.5 * (xright_mid.v2 - xleft_mid.v2),
                                                 theta * (xright_mid.v2 - center.v2));

                    xprims_l.p =
                        center.p + 0.5 * minmod(theta * (center.p - xleft_mid.p),
                                                0.5 * (xright_mid.p - xleft_mid.p),
                                                theta * (xright_mid.p - center.p));

                    // Reconstructed right Primitive vector in x
                    xprims_r.rho =
                        xright_mid.rho -
                        0.5 * minmod(theta * (xright_mid.rho - center.rho),
                                     0.5 * (xright_most.rho - center.rho),
                                     theta * (xright_most.rho - xright_mid.rho));

                    xprims_r.v1 = xright_mid.v1 -
                                  0.5 * minmod(theta * (xright_mid.v1 - center.v1),
                                               0.5 * (xright_most.v1 - center.v1),
                                               theta * (xright_most.v1 - xright_mid.v1));

                    xprims_r.v2 = xright_mid.v2 -
                                  0.5 * minmod(theta * (xright_mid.v2 - center.v2),
                                               0.5 * (xright_most.v2 - center.v2),
                                               theta * (xright_most.v2 - xright_mid.v2));

                    xprims_r.p = xright_mid.p -
                                 0.5 * minmod(theta * (xright_mid.p - center.p),
                                              0.5 * (xright_most.p - center.p),
                                              theta * (xright_most.p - xright_mid.p));

                    // Reconstructed right Primitive vector in y-direction at j+1/2
                    // interfce
                    yprims_l.rho =
                        center.rho + 0.5 * minmod(theta * (center.rho - yleft_mid.rho),
                                                  0.5 * (yright_mid.rho - yleft_mid.rho),
                                                  theta * (yright_mid.rho - center.rho));

                    yprims_l.v1 =
                        center.v1 + 0.5 * minmod(theta * (center.v1 - yleft_mid.v1),
                                                 0.5 * (yright_mid.v1 - yleft_mid.v1),
                                                 theta * (yright_mid.v1 - center.v1));

                    yprims_l.v2 =
                        center.v2 + 0.5 * minmod(theta * (center.v2 - yleft_mid.v2),
                                                 0.5 * (yright_mid.v2 - yleft_mid.v2),
                                                 theta * (yright_mid.v2 - center.v2));

                    yprims_l.p =
                        center.p + 0.5 * minmod(theta * (center.p - yleft_mid.p),
                                                0.5 * (yright_mid.p - yleft_mid.p),
                                                theta * (yright_mid.p - center.p));

                    yprims_r.rho =
                        yright_mid.rho -
                        0.5 * minmod(theta * (yright_mid.rho - center.rho),
                                     0.5 * (yright_most.rho - center.rho),
                                     theta * (yright_most.rho - yright_mid.rho));

                    yprims_r.v1 = yright_mid.v1 -
                                  0.5 * minmod(theta * (yright_mid.v1 - center.v1),
                                               0.5 * (yright_most.v1 - center.v1),
                                               theta * (yright_most.v1 - yright_mid.v1));

                    yprims_r.v2 = yright_mid.v2 -
                                  0.5 * minmod(theta * (yright_mid.v2 - center.v2),
                                               0.5 * (yright_most.v2 - center.v2),
                                               theta * (yright_most.v2 - yright_mid.v2));

                    yprims_r.p = yright_mid.p -
                                 0.5 * minmod(theta * (yright_mid.p - center.p),
                                              0.5 * (yright_most.p - center.p),
                                              theta * (yright_most.p - yright_mid.p));

                    // Calculate the left and right states using the reconstructed PLM
                    // Primitive
                    ux_l = calc_stateSR2D(xprims_l);
                    ux_r = calc_stateSR2D(xprims_r);

                    uy_l = calc_stateSR2D(yprims_l);
                    uy_r = calc_stateSR2D(yprims_r);

                    f_l = calc_Flux(xprims_l);
                    f_r = calc_Flux(xprims_r);

                    g_l = calc_Flux(yprims_l, 2);
                    g_r = calc_Flux(yprims_r, 2);

                    favr = (uy_r - uy_l) * (-K);

                    if (hllc)
                    {
                        if (strong_shock(xprims_l.p, xprims_r.p) ){
                            f1 = calc_hll_flux(ux_l, ux_r, f_l, f_r, xprims_l, xprims_r, 1);
                        } else {
                            f1 = calc_hllc_flux(ux_l, ux_r, f_l, f_r, xprims_l, xprims_r, 1);
                        }
                        
                        if (strong_shock(yprims_l.p, yprims_r.p)){
                            g1 = calc_hll_flux(uy_l, uy_r, g_l, g_r, yprims_l, yprims_r, 2);
                        } else {
                            g1 = calc_hllc_flux(uy_l, uy_r, g_l, g_r, yprims_l, yprims_r, 2);
                        }
                        // f1 = calc_hllc_flux(ux_l, ux_r, f_l, f_r, xprims_l, xprims_r, 1);
                        // g1 = calc_hllc_flux(uy_l, uy_r, g_l, g_r, yprims_l, yprims_r, 2);
                    }
                    else
                    {
                        f1 = calc_hll_flux(ux_l, ux_r, f_l, f_r, xprims_l, xprims_r, 1);
                        g1 = calc_hll_flux(uy_l, uy_r, g_l, g_r, yprims_l, yprims_r, 2);
                    }

                    // Do the same thing, but for the left side interface [i - 1/2]

                    // Left side Primitive in x
                    xprims_l.rho = xleft_mid.rho +
                                   0.5 * minmod(theta * (xleft_mid.rho - xleft_most.rho),
                                                0.5 * (center.rho - xleft_most.rho),
                                                theta * (center.rho - xleft_mid.rho));

                    xprims_l.v1 = xleft_mid.v1 +
                                  0.5 * minmod(theta * (xleft_mid.v1 - xleft_most.v1),
                                               0.5 * (center.v1 - xleft_most.v1),
                                               theta * (center.v1 - xleft_mid.v1));

                    xprims_l.v2 = xleft_mid.v2 +
                                  0.5 * minmod(theta * (xleft_mid.v2 - xleft_most.v2),
                                               0.5 * (center.v2 - xleft_most.v2),
                                               theta * (center.v2 - xleft_mid.v2));

                    xprims_l.p =
                        xleft_mid.p + 0.5 * minmod(theta * (xleft_mid.p - xleft_most.p),
                                                   0.5 * (center.p - xleft_most.p),
                                                   theta * (center.p - xleft_mid.p));

                    // Right side Primitive in x
                    xprims_r.rho =
                        center.rho - 0.5 * minmod(theta * (center.rho - xleft_mid.rho),
                                                  0.5 * (xright_mid.rho - xleft_mid.rho),
                                                  theta * (xright_mid.rho - center.rho));

                    xprims_r.v1 =
                        center.v1 - 0.5 * minmod(theta * (center.v1 - xleft_mid.v1),
                                                 0.5 * (xright_mid.v1 - xleft_mid.v1),
                                                 theta * (xright_mid.v1 - center.v1));

                    xprims_r.v2 =
                        center.v2 - 0.5 * minmod(theta * (center.v2 - xleft_mid.v2),
                                                 0.5 * (xright_mid.v2 - xleft_mid.v2),
                                                 theta * (xright_mid.v2 - center.v2));

                    xprims_r.p =
                        center.p - 0.5 * minmod(theta * (center.p - xleft_mid.p),
                                                0.5 * (xright_mid.p - xleft_mid.p),
                                                theta * (xright_mid.p - center.p));

                    // Left side Primitive in y
                    yprims_l.rho = yleft_mid.rho +
                                   0.5 * minmod(theta * (yleft_mid.rho - yleft_most.rho),
                                                0.5 * (center.rho - yleft_most.rho),
                                                theta * (center.rho - yleft_mid.rho));

                    yprims_l.v1 = yleft_mid.v1 +
                                  0.5 * minmod(theta * (yleft_mid.v1 - yleft_most.v1),
                                               0.5 * (center.v1 - yleft_most.v1),
                                               theta * (center.v1 - yleft_mid.v1));

                    yprims_l.v2 = yleft_mid.v2 +
                                  0.5 * minmod(theta * (yleft_mid.v2 - yleft_most.v2),
                                               0.5 * (center.v2 - yleft_most.v2),
                                               theta * (center.v2 - yleft_mid.v2));

                    yprims_l.p =
                        yleft_mid.p + 0.5 * minmod(theta * (yleft_mid.p - yleft_most.p),
                                                   0.5 * (center.p - yleft_most.p),
                                                   theta * (center.p - yleft_mid.p));

                    // Right side Primitive in y
                    yprims_r.rho =
                        center.rho - 0.5 * minmod(theta * (center.rho - yleft_mid.rho),
                                                  0.5 * (yright_mid.rho - yleft_mid.rho),
                                                  theta * (yright_mid.rho - center.rho));

                    yprims_r.v1 =
                        center.v1 - 0.5 * minmod(theta * (center.v1 - yleft_mid.v1),
                                                 0.5 * (yright_mid.v1 - yleft_mid.v1),
                                                 theta * (yright_mid.v1 - center.v1));

                    yprims_r.v2 =
                        center.v2 - 0.5 * minmod(theta * (center.v2 - yleft_mid.v2),
                                                 0.5 * (yright_mid.v2 - yleft_mid.v2),
                                                 theta * (yright_mid.v2 - center.v2));

                    yprims_r.p =
                        center.p - 0.5 * minmod(theta * (center.p - yleft_mid.p),
                                                0.5 * (yright_mid.p - yleft_mid.p),
                                                theta * (yright_mid.p - center.p));

                    // Calculate the left and right states using the reconstructed PLM
                    // Primitive
                    ux_l = calc_stateSR2D(xprims_l);
                    ux_r = calc_stateSR2D(xprims_r);
                    uy_l = calc_stateSR2D(yprims_l);
                    uy_r = calc_stateSR2D(yprims_r);

                    f_l = calc_Flux(xprims_l);
                    f_r = calc_Flux(xprims_r);
                    g_l = calc_Flux(yprims_l, 2);
                    g_r = calc_Flux(yprims_r, 2);

                    favl = (uy_r - uy_l) * (-K);
                    
                    if (hllc)
                    {
                        if (strong_shock(xprims_l.p, xprims_r.p) ){
                            f2 = calc_hll_flux(ux_l, ux_r, f_l, f_r, xprims_l, xprims_r, 1);
                        } else {
                            f2 = calc_hllc_flux(ux_l, ux_r, f_l, f_r, xprims_l, xprims_r, 1);
                        }
                        
                        if (strong_shock(yprims_l.p, yprims_r.p)){
                            g2 = calc_hll_flux(uy_l, uy_r, g_l, g_r, yprims_l, yprims_r, 2);
                        } else {
                            g2 = calc_hllc_flux(uy_l, uy_r, g_l, g_r, yprims_l, yprims_r, 2);
                        }
                        // f2 = calc_hllc_flux(ux_l, ux_r, f_l, f_r, xprims_l, xprims_r, 1);
                        // g2 = calc_hllc_flux(uy_l, uy_r, g_l, g_r, yprims_l, yprims_r, 2);
                        
                    }
                    else
                    {
                        f2 = calc_hll_flux(ux_l, ux_r, f_l, f_r, xprims_l, xprims_r, 1);
                        g2 = calc_hll_flux(uy_l, uy_r, g_l, g_r, yprims_l, yprims_r, 2);
                    }

                    rhoc = center.rho;
                    pc   = center.p;
                    uc   = center.v1;
                    vc   = center.v2;

                    // Compute the surface areas
                    // upper_tsurface = coord_lattice.x2_face_areas[(ycoordinate + 1)*xphysical_grid + xcoordinate];
                    // lower_tsurface = coord_lattice.x2_face_areas[ycoordinate * xphysical_grid + xcoordinate];
                    // right_rsurface = coord_lattice.x1_face_areas[ycoordinate * (xphysical_grid + 1) + xcoordinate + 1];
                    // left_rsurface  = coord_lattice.x1_face_areas[ycoordinate * (xphysical_grid + 1) + xcoordinate];
                    right_rsurface = coord_lattice.x1_face_areas[xcoordinate + 1];
                    left_rsurface  = coord_lattice.x1_face_areas[xcoordinate];

                    volAvg         = coord_lattice.x1mean[xcoordinate];
                    deltaV1        = coord_lattice.dV1[xcoordinate];
                    deltaV2        = volAvg * coord_lattice.dV2[ycoordinate];

                    hc             = 1.0 + gamma * pc /(rhoc * (gamma - 1.0));
                    wc2            = 1.0/(1.0 - (uc * uc + vc * vc));

                    L.push_back(Conserved{
                        // L(D)
                        -(f1.D * right_rsurface - f2.D * left_rsurface) / deltaV1 
                            - (g1.D * upper_tsurface - g2.D * lower_tsurface) / deltaV2 
                                + sourceD[xcoordinate + xphysical_grid * ycoordinate] * decay_const,

                        // L(S1)
                        -(f1.S1 * right_rsurface - f2.S1 * left_rsurface) / deltaV1 
                            - (g1.S1 * upper_tsurface - g2.S1 * lower_tsurface) / deltaV2 
                                + rhoc * hc * wc2 * vc * vc / volAvg + 2.0 * pc / volAvg 
                                    + source_S1[xcoordinate + xphysical_grid * ycoordinate] * decay_const,

                        // L(S2)
                        -(f1.S2 * right_rsurface - f2.S2 * left_rsurface) / deltaV1 
                            - (g1.S2 * upper_tsurface - g2.S2 * lower_tsurface) / deltaV2
                                -(rhoc * uc * hc * wc2 * vc / volAvg - pc * coord_lattice.cot[ycoordinate] / volAvg) 
                                    + source_S2[xcoordinate + xphysical_grid * ycoordinate] * decay_const,

                        // L(tau)
                        -(f1.tau * right_rsurface - f2.tau * left_rsurface) / deltaV1 
                            - (g1.tau * upper_tsurface - g2.tau * lower_tsurface) / deltaV2
                                + source_tau[xcoordinate + xphysical_grid * ycoordinate] * decay_const
                                    });
                }
            }
        }

        break;
    }

    return L;
};

//===================================================================================================================
//                                            SIMULATE
//===================================================================================================================
vector<vector<double>> SRHD2D::simulate2D(
    vector<double> lorentz_gamma, 
    const vector<vector<double>> sources,
    float tstart = 0., 
    float tend = 0.1, 
    double dt = 1.e-4, 
    double theta = 1.5,
    double engine_duration = 10, 
    double chkpt_interval = 0.1,
    string data_directory = "data/", 
    bool first_order = true,
    bool periodic = false, 
    bool linspace = true, 
    bool hllc = false)
{

    int i_real, j_real;
    string tnow, tchunk, tstep;
    int total_zones = NX * NY;
    
    double round_place = 1 / chkpt_interval;
    double t = tstart;
    double t_interval =
        t == 0 ? floor(tstart * round_place + 0.5) / round_place
               : floor(tstart * round_place + 0.5) / round_place + chkpt_interval;

    string filename;

    this->sources = sources;
    this->first_order = first_order;
    this->periodic = periodic;
    this->hllc = hllc;
    this->linspace = linspace;
    this->lorentz_gamma = lorentz_gamma;
    this->theta = theta;

    if (first_order)
    {
        this->xphysical_grid = NX - 2;
        this->yphysical_grid = NY - 2;
        this->idx_active = 1;
        this->x_bound = NX - 1;
        this->y_bound = NY - 1;
    }
    else
    {
        this->xphysical_grid = NX - 4;
        this->yphysical_grid = NY - 4;
        this->idx_active = 2;
        this->x_bound = NX - 2;
        this->y_bound = NY - 2;
    }

    this->active_zones = xphysical_grid * yphysical_grid;
    this->xvertices.resize(x1.size() + 1);
    this->yvertices.resize(x2.size() + 1);

    //--------Config the System Enums
    config_system();
    if ((coord_system == "spherical") && (linspace))
    {
        this->coord_lattice = CLattice(x1, x2, simbi::Geometry::SPHERICAL);
        coord_lattice.config_lattice(simbi::Cellspacing::LINSPACE,
                                     simbi::Cellspacing::LINSPACE);
    }
    else if ((coord_system == "spherical") && (!linspace))
    {
        this->coord_lattice = CLattice(x1, x2, simbi::Geometry::SPHERICAL);
        coord_lattice.config_lattice(simbi::Cellspacing::LOGSPACE,
                                     simbi::Cellspacing::LINSPACE);
    }
    else
    {
        this->coord_lattice = CLattice(x1, x2, simbi::Geometry::CARTESIAN);
        coord_lattice.config_lattice(simbi::Cellspacing::LINSPACE,
                                     simbi::Cellspacing::LINSPACE);
    }

    if (coord_lattice.x2vertices[yphysical_grid] == pi()){
        bipolar = true;
    }
    // Write some info about the setup for writeup later
    DataWriteMembers setup;
    setup.xmax = x1[xphysical_grid - 1];
    setup.xmin = x1[0];
    setup.ymax = x2[yphysical_grid - 1];
    setup.ymin = x2[0];
    setup.NX = NX;
    setup.NY = NY;

    vector<Conserved> u, u1, udot, udot1;
    u.resize(nzones);
    u1.resize(nzones);
    udot.reserve(active_zones);
    udot1.resize(nzones);
    prims.reserve(nzones);

    // Define the source terms
    sourceD = sources[0];
    source_S1 = sources[1];
    source_S2 = sources[2];
    source_tau = sources[3];

    // Copy the state array into real & profile variables
    for (size_t i = 0; i < state2D[0].size(); i++)
    {
        u[i] =
            Conserved(state2D[0][i], state2D[1][i], state2D[2][i], state2D[3][i]);
    }

    Conserved L;
    n = 0;

    block_size = 4;

    high_resolution_clock::time_point t1, t2;

    // Using a sigmoid decay function to represent when the source terms turn off.
    decay_const = 1.0 / (1.0 + exp(10.0 * (tstart - engine_duration)));

    // Set the Primitive from the initial conditions and initialize the pressure
    // guesses
    pressure_guess.resize(nzones);
    prims = cons2prim2D(u);
    n++;

    // Test Viscous FLux 
    // std::cout << "testing AV Visc" << "\n";
    // aVisc.calc_artificial_visc(prims, coord_lattice);
    // std::cout << "test done" << "\n";

    // Declare I/O variables for Read/Write capability
    PrimData prods;
    sr2d::PrimitiveData transfer_prims;

    if (t == 0)
    {
        config_ghosts2D(u, NX, NY, false);
    }

    u1  = u;

    if (first_order)
    {
        while (t < tend)
        {
            /* Compute the loop execution time */
            high_resolution_clock::time_point t1 = high_resolution_clock::now();

            udot = u_dot2D(u);

            for (int jj = 0; jj < yphysical_grid; jj++)
            {
                j_real = jj + 1;
                for (int ii = 0; ii < xphysical_grid; ii++)
                {
                    i_real = ii + 1;
                    u[i_real + NX * j_real] += udot[ii + xphysical_grid * jj] * dt;
                }
            }

            config_ghosts2D(u, NX, NY, true);
            prims = cons2prim2D(u);

            t += dt;
            dt = adapt_dt(prims);

            /* Compute the loop execution time */
            high_resolution_clock::time_point t2 = high_resolution_clock::now();
            duration<double> time_span = duration_cast<duration<double>>(t2 - t1);

            cout << fixed << setprecision(3) << scientific;
            cout << "\r"
                 << "dt: " << setw(5) << dt << "\t"
                 << "t: " << setw(5) << t << "\t"
                 << "Zones per sec: " << total_zones / time_span.count() << flush;

            // cout << n << endl;
            n++;
        }
    }
    else
    {
        tchunk = "000000";
        int tchunk_order_of_mag = 2;
        int time_order_of_mag, num_zeros;
        while (t < tend)
        {
            /* Compute the loop execution time */
            t1 = high_resolution_clock::now();

            udot = u_dot2D(u);

            for (int jj = 0; jj < yphysical_grid; jj+=block_size)
            {
                for (int ii = 0; ii < xphysical_grid; ii+=block_size)
                {
                    for(int y = jj; y < std::min(jj + block_size, yphysical_grid); y++){
                        j_real = y + 2;
                        for(int x = ii; x < std::min(ii + block_size, xphysical_grid); x++){
                            i_real = x + 2;
                            u1[i_real + NX * j_real] += udot[x + xphysical_grid * y] * dt;
                        }
                    }
                }
            }

            config_ghosts2D(u1, NX, NY, false);
            prims = cons2prim2D(u1);
            udot1  = u_dot2D(u1);

            for (int jj = 0; jj < yphysical_grid; jj+= block_size)
            {
                j_real = jj + 2;
                for (int ii = 0; ii < xphysical_grid; ii+= block_size)
                {
                    for(int y = jj; y < std::min(jj + block_size, yphysical_grid); y++){
                        j_real = y + 2;
                        for(int x = ii; x < std::min(ii + block_size, xphysical_grid); x++){
                            i_real = x + 2;
                            u[i_real + NX * j_real] +=  udot [x + xphysical_grid * y] * 0.5 * dt + 
                                                        udot1[x + xphysical_grid * y] * 0.5 * dt;
                        }
                    }
                    
                }
            }

            config_ghosts2D(u, NX, NY, false);
            prims = cons2prim2D(u);
            u1 = u;

            t += dt;
            dt = adapt_dt(prims);

            /* Compute the loop execution time */
            
            t2 = high_resolution_clock::now();
            auto time_span = duration_cast<duration<double>>(t2 - t1);
            
            

            cout << fixed << setprecision(3) << scientific;
            cout << "\r"
                 << "dt: " << setw(5) << dt << "\t"
                 << "t: " << setw(5) << t << "\t"
                 << "Zones per sec: " << total_zones / time_span.count() << flush;

            n++;
            decay_const = 1.0 / (1.0 + exp(10.0 * (t - engine_duration)));

            /* Write to a File every nth of a second */
            if (t >= t_interval)
            {
                // Check if time order of magnitude exceeds 
                // the hundreds place set by the tchunk string
                time_order_of_mag = std::floor(std::log10(t));
                if (time_order_of_mag > tchunk_order_of_mag){
                    tchunk.insert(0, "0");
                    tchunk_order_of_mag += 1;
                }
                
                transfer_prims = vecs2struct(prims);
                toWritePrim(&transfer_prims, &prods);
                tnow = create_step_str(t_interval, tchunk);
                filename = string_format("%d.chkpt." + tnow + ".h5", yphysical_grid);
                setup.t = t;
                setup.dt = dt;
                write_hdf5(data_directory, filename, prods, setup, 2, total_zones);
                t_interval += chkpt_interval;
            }
        }
    }

    cout << "\n " << endl;

    prims = cons2prim2D(u);
    transfer_prims = vecs2struct(prims);

    vector<vector<double>> solution(4, vector<double>(nzones));

    solution[0] = transfer_prims.rho;
    solution[1] = transfer_prims.v1;
    solution[2] = transfer_prims.v2;
    solution[3] = transfer_prims.p;

    return solution;
};
