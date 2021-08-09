/*
 * C++ Source to perform 2D SRHD Calculations
 * Marcus DuPont
 * New York University
 * 07/15/2020
 * Compressible Hydro Simulation
 */

#include "helpers.hpp"
#include "srhydro2D.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>

using namespace simbi;
using namespace std::chrono;

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
SRHD2D::SRHD2D(std::vector<std::vector<double>> state2D, int nx, int ny, double gamma,
               std::vector<double> x1, std::vector<double> x2, double Cfl,
               std::string coord_system = "cartesian")
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
/**
 * Return a 2D matrix containing the primitive
 * variables density , pressure, and
 * three-velocity
*/
void SRHD2D::cons2prim2D()
{
    #pragma omp parallel 
    {
        double S1, S2, S, D, tau, tol;
        double W, v1, v2;

        // Define Newton-Raphson Vars
        double etotal, c2, f, g, p, peq;
        double Ws, rhos, eps, h;

        int iter, gid;
        for (int jj = 0; jj < NY; jj++)
        {
            #pragma omp for nowait
            for (int ii = 0; ii < NX; ii++)
            {   
                // write global index idx
                gid = ii + NX * jj;
                D   = cons[gid].D;     // Relativistic Mass Density
                S1  = cons[gid].S1;   // X1-Momentum Denity
                S2  = cons[gid].S2;   // X2-Momentum Density
                tau = cons[gid].tau; // Energy Density
                S = sqrt(S1 * S1 + S2 * S2);

                peq = (n != 0.0) ? pressure_guess[gid] : std::abs(S - D - tau);

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

                    if (iter > MAX_ITER)
                    {
                        std::cout << "\n";
                        std::cout << "p: " << p << "\n";
                        std::cout << "S: " << S << "\n";
                        std::cout << "tau: " << tau << "\n";
                        std::cout << "D: " << D << "\n";
                        std::cout << "et: " << etotal << "\n";
                        std::cout << "Ws: " << Ws << "\n";
                        std::cout << "v2: " << v2 << "\n";
                        std::cout << "W: " << W << "\n";
                        std::cout << "n: " << n << "\n";
                        std::cout << "\n Cons2Prim Cannot Converge" << "\n";
                        exit(EXIT_FAILURE);
                    }

                } while (std::abs(peq - p) >= tol);
            
                v1 = S1 / (tau + D + peq);
                v2 = S2 / (tau + D + peq);

                // Update the pressure guess for the next time step
                pressure_guess[gid] = peq;
                prims[gid]          = (Primitive(D * sqrt(1.0 - (v1 * v1 + v2 * v2)), v1, v2, peq));
            } // end ii loop
        } // end jj loop
    } // end parallel region
    
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
        const double aL = std::min(bl, (v1_l - cs_l)/(1. - v1_l*cs_l));
        const double aR = std::max(br, (v1_r + cs_r)/(1. + v1_r*cs_r));

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
        const double aL = std::min(bl, (v2_l - cs_l)/(1. - v2_l*cs_l));
        const double aR = std::max(br, (v2_r + cs_r)/(1. + v2_r*cs_r));

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

Conserved SRHD2D::prims2cons(const Primitive &prims)
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
void SRHD2D::adapt_dt()
{
    double min_dt = INFINITY;
    #pragma omp parallel 
    {
        double dx1, cs, dx2, rho, pressure, v1, v2, rmean, h;
        double cfl_dt;
        int shift_i, shift_j;
        double plus_v1, plus_v2, minus_v1, minus_v2;
        int aid; // active index id

        // Compute the minimum timestep given CFL
        for (int jj = 0; jj < yphysical_grid; jj++)
        {
            dx2 = coord_lattice.dx2[jj];

            #pragma omp for schedule(static)
            for (int ii = 0; ii < xphysical_grid; ii++)
            {
                shift_i  = ii + idx_active;
                aid      = shift_i + NX * shift_j;
                dx1      = coord_lattice.dx1[ii];
                rho      = prims[aid].rho;
                v1       = prims[aid].v1;
                v2       = prims[aid].v2;
                pressure = prims[aid].p;

                h = 1. + gamma * pressure / (rho * (gamma - 1.));
                cs = sqrt(gamma * pressure / (rho * h));

                plus_v1  = (v1 + cs) / (1. + v1 * cs);
                plus_v2  = (v2 + cs) / (1. + v2 * cs);
                minus_v1 = (v1 - cs) / (1. - v1 * cs);
                minus_v2 = (v2 - cs) / (1. - v2 * cs);

                if (coord_system == "cartesian")
                {

                    cfl_dt = std::min(dx1 / (std::max(std::abs(plus_v1), std::abs(minus_v1))),
                                dx2 / (std::max(std::abs(plus_v2), std::abs(minus_v2))));
                }
                else
                {
                    rmean = coord_lattice.x1mean[ii];
                    cfl_dt = std::min(dx1 / (std::max(std::abs(plus_v1), std::abs(minus_v1))),
                                rmean * dx2 / (std::max(std::abs(plus_v2), std::abs(minus_v2))));
                }

                min_dt = min_dt < cfl_dt ? min_dt : cfl_dt;
                
            } // end ii 
        } // end jj
    } // end parallel region
    dt = CFL * min_dt;
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

void SRHD2D::evolve()
{
    #pragma omp parallel 
    {
        int xcoordinate, ycoordinate;
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
        int aid;
        double dx, dy, rmean, dV1, dV2, s1L, s1R, s2L, s2R;
        double pc, rhoc, hc, wc2, uc, vc;

        if (first_order)
        {
            for (int jj = j_start; jj < j_bound; jj++)
            {
                ycoordinate = jj - 1;
                s2R = coord_lattice.x2_face_areas[ycoordinate + 1];
                s2L = coord_lattice.x2_face_areas[ycoordinate];

                #pragma omp for nowait
                for (int ii = i_start; ii < i_bound; ii++)
                {
                    aid = jj * NX + ii;
                    if(!periodic)
                    {
                        xcoordinate = ii - 1;

                        // i+1/2
                        ux_l = cons[(ii + 0) + NX * jj];
                        ux_r = cons[(ii + 1) + NX * jj];

                        // j+1/2
                        uy_l = cons[ii + NX * (jj + 0)];
                        uy_r = cons[ii + NX * (jj + 1)];

                        xprims_l = prims[(ii + 0) + jj * NX];
                        xprims_r = prims[(ii + 1) + jj * NX];

                        yprims_l = prims[ii + (jj + 0) * NX];
                        yprims_r = prims[ii + (jj + 1) * NX];
                    } else {
                        xcoordinate = ii;
                        ycoordinate = jj;
                        // i+1/2
                        ux_l = cons[(ii + 0) + NX * jj];
                        ux_r = roll(cons, (ii + 1) + NX * jj);

                        // j+1/2
                        uy_l = cons[ii + NX * (jj + 0)];
                        uy_r = roll(cons, ii + NX * (jj + 1));

                        xprims_l = prims[ii + jj * NX];
                        xprims_r = roll(prims, (ii + 1) + jj * NX);

                        yprims_l = prims[ii + jj * NX];
                        yprims_r = roll(prims, ii + (jj + 1) * NX);
                    }

                    f_l = calc_Flux(xprims_l, 1);
                    f_r = calc_Flux(xprims_r, 1);

                    g_l = calc_Flux(yprims_l, 2);
                    g_r = calc_Flux(yprims_r, 2);

                    // Calc HLL Flux at i+1/2 interface
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
                    } else {
                        f1 = calc_hll_flux(ux_l, ux_r, f_l, f_r, xprims_l, xprims_r, 1);
                        g1 = calc_hll_flux(uy_l, uy_r, g_l, g_r, yprims_l, yprims_r, 2);
                    }

                    // Set up the left and right state interfaces for i-1/2
                    if(!periodic)
                    {
                        // i-1/2
                        ux_l = cons[(ii - 1) + NX * jj];
                        ux_r = cons[(ii + 0) + NX * jj];

                        // j-1/2
                        uy_l = cons[ii + NX * (jj - 1)];
                        uy_r = cons[ii + NX * (jj - 0)];

                        xprims_l = prims[(ii - 1) + jj * NX];
                        xprims_r = prims[(ii + 0) + jj * NX];

                        yprims_l = prims[ii + (jj - 1) * NX];
                        yprims_r = prims[ii + (jj + 0) * NX];
                    } else {
                        // i-1/2
                        ux_l = roll(cons, (ii - 1) + NX * jj);
                        ux_r = cons[(ii + 0) + NX * jj];

                        // j-1/2
                        uy_l = roll(cons, ii + NX * (jj - 1));
                        uy_r = cons[ii + NX * jj];

                        xprims_l = roll(prims,  ii - 1 + jj * NX);
                        xprims_r = prims[ii + jj * NX];

                        yprims_l = roll(prims, ii + (jj - 1) * NX);
                        yprims_r = prims[ii + jj * NX];
                    }

                    f_l = calc_Flux(xprims_l, 1);
                    f_r = calc_Flux(xprims_r, 1);

                    g_l = calc_Flux(yprims_l, 2);
                    g_r = calc_Flux(yprims_r, 2);

                    // Calc HLL Flux at i+1/2 interface
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
                    } else {
                        f2 = calc_hll_flux(ux_l, ux_r, f_l, f_r, xprims_l, xprims_r, 1);
                        g2 = calc_hll_flux(uy_l, uy_r, g_l, g_r, yprims_l, yprims_r, 2);
                    }

                    int real_loc = ycoordinate * xphysical_grid + xcoordinate;
                    switch (geometry[coord_system])
                    {
                    case simbi::Geometry::CARTESIAN:
                        dx = coord_lattice.dx1[xcoordinate];
                        dy = coord_lattice.dx2[ycoordinate];
                        cons_n[aid].D   += dt * (- (f1.D - f2.D)     / dx - (g1.D - g2.D)     / dy + sourceD[real_loc]);
                        cons_n[aid].S1  += dt * (- (f1.S1 - f2.S1)   / dx - (g1.S1 - g2.S1)   / dy + source_S1[real_loc]);
                        cons_n[aid].S2  += dt * (- (f1.S2 - f2.S2)   / dx - (g1.S2 - g2.S2)   / dy + source_S2[real_loc]);
                        cons_n[aid].tau += dt * (- (f1.tau - f2.tau) / dx - (g1.tau - g2.tau) / dy + source_tau[real_loc]);
                        break;
                    
                    case simbi::Geometry::SPHERICAL:
                        s1R   = coord_lattice.x1_face_areas[xcoordinate + 1];
                        s1L   = coord_lattice.x1_face_areas[xcoordinate];
                        rmean = coord_lattice.x1mean[xcoordinate];
                        dV1   = coord_lattice.dV1[xcoordinate];
                        dV2   = rmean * coord_lattice.dV2[ycoordinate];

                        pc   = prims[aid].p;
                        rhoc = prims[aid].rho, 
                        uc   = prims[aid].v1;
                        vc   = prims[aid].v2;
                        
                        hc    = 1.0 + gamma * pc /(rhoc * (gamma - 1.0));
                        wc2   = 1.0/(1.0 - (uc * uc + vc * vc));


                        cons_n[aid] += Conserved{
                            // L(D)
                            -(f1.D * s1R - f2.D * s1L) / dV1 
                                - (g1.D * s2R - g2.D * s2L) / dV2 
                                    + sourceD[xcoordinate + xphysical_grid * ycoordinate] * decay_const,

                            // L(S1)
                            -(f1.S1 * s1R - f2.S1 * s1L) / dV1 
                                - (g1.S1 * s2R - g2.S1 * s2L) / dV2 
                                    + rhoc * hc * wc2 * vc * vc / rmean + 2 * pc / rmean +
                                        source_S1[xcoordinate + xphysical_grid * ycoordinate] * decay_const,

                            // L(S2)
                            -(f1.S2 * s1R - f2.S2 * s1L) / dV1
                                - (g1.S2 * s2R - g2.S2 * s2L) / dV2 
                                    - (rhoc * hc * wc2 * uc * vc / rmean - pc * coord_lattice.cot[ycoordinate] / rmean) 
                                        + source_S2[xcoordinate + xphysical_grid * ycoordinate] * decay_const,

                            // L(tau)
                            -(f1.tau * s1R - f2.tau * s1L) / dV1 
                                - (g1.tau * s2R - g2.tau * s2L) / dV2 
                                    + source_tau[xcoordinate + xphysical_grid * ycoordinate] * decay_const
                        } * dt;
                        break;
                    }
                } // end ii loop
            } // end jj loop
        }
        else
        {
            for (int jj = j_start; jj < j_bound; jj++)
            {
                ycoordinate = jj - 2;
                s2L         = coord_lattice.x2_face_areas[ycoordinate];
                s2R         = coord_lattice.x2_face_areas[ycoordinate + 1];

                #pragma omp parallel for
                for (int ii = i_start; ii < i_bound; ii++)
                {
                    aid = jj * NX + ii;
                    if (periodic)
                    {
                        xcoordinate = ii;
                        ycoordinate = jj;

                        // X Coordinate
                        xleft_most   = roll(prims, jj * NX + ii - 2);
                        xleft_mid    = roll(prims, jj * NX + ii - 1);
                        center       = prims[jj * NX + ii];
                        xright_mid   = roll(prims, jj * NX + ii + 1);
                        xright_most  = roll(prims, jj * NX + ii + 2);

                        yleft_most   = roll(prims, ii +  NX * (jj - 2) );
                        yleft_mid    = roll(prims, ii +  NX * (jj - 1) );
                        yright_mid   = roll(prims, ii +  NX * (jj + 1) );
                        yright_most  = roll(prims, ii +  NX * (jj + 2) );
                    }
                    else
                    {
                        // Adjust for beginning input of L vector
                        xcoordinate = ii - 2;

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
                        center.rho + 0.5 * minmod(plm_theta * (center.rho - xleft_mid.rho),
                                                    0.5 * (xright_mid.rho - xleft_mid.rho),
                                                    plm_theta * (xright_mid.rho - center.rho));

                    xprims_l.v1 =
                        center.v1 + 0.5 * minmod(plm_theta * (center.v1 - xleft_mid.v1),
                                                    0.5 * (xright_mid.v1 - xleft_mid.v1),
                                                    plm_theta * (xright_mid.v1 - center.v1));

                    xprims_l.v2 =
                        center.v2 + 0.5 * minmod(plm_theta * (center.v2 - xleft_mid.v2),
                                                    0.5 * (xright_mid.v2 - xleft_mid.v2),
                                                    plm_theta * (xright_mid.v2 - center.v2));

                    xprims_l.p =
                        center.p + 0.5 * minmod(plm_theta * (center.p - xleft_mid.p),
                                                0.5 * (xright_mid.p - xleft_mid.p),
                                                plm_theta * (xright_mid.p - center.p));

                    // Reconstructed right Primitive vector in x
                    xprims_r.rho =
                        xright_mid.rho -
                        0.5 * minmod(plm_theta * (xright_mid.rho - center.rho),
                                        0.5 * (xright_most.rho - center.rho),
                                        plm_theta * (xright_most.rho - xright_mid.rho));

                    xprims_r.v1 = xright_mid.v1 -
                                    0.5 * minmod(plm_theta * (xright_mid.v1 - center.v1),
                                                0.5 * (xright_most.v1 - center.v1),
                                                plm_theta * (xright_most.v1 - xright_mid.v1));

                    xprims_r.v2 = xright_mid.v2 -
                                    0.5 * minmod(plm_theta * (xright_mid.v2 - center.v2),
                                                0.5 * (xright_most.v2 - center.v2),
                                                plm_theta * (xright_most.v2 - xright_mid.v2));

                    xprims_r.p = xright_mid.p -
                                    0.5 * minmod(plm_theta * (xright_mid.p - center.p),
                                                0.5 * (xright_most.p - center.p),
                                                plm_theta * (xright_most.p - xright_mid.p));

                    // Reconstructed right Primitive vector in y-direction at j+1/2
                    // interfce
                    yprims_l.rho =
                        center.rho + 0.5 * minmod(plm_theta * (center.rho - yleft_mid.rho),
                                                    0.5 * (yright_mid.rho - yleft_mid.rho),
                                                    plm_theta * (yright_mid.rho - center.rho));

                    yprims_l.v1 =
                        center.v1 + 0.5 * minmod(plm_theta * (center.v1 - yleft_mid.v1),
                                                    0.5 * (yright_mid.v1 - yleft_mid.v1),
                                                    plm_theta * (yright_mid.v1 - center.v1));

                    yprims_l.v2 =
                        center.v2 + 0.5 * minmod(plm_theta * (center.v2 - yleft_mid.v2),
                                                    0.5 * (yright_mid.v2 - yleft_mid.v2),
                                                    plm_theta * (yright_mid.v2 - center.v2));

                    yprims_l.p =
                        center.p + 0.5 * minmod(plm_theta * (center.p - yleft_mid.p),
                                                0.5 * (yright_mid.p - yleft_mid.p),
                                                plm_theta * (yright_mid.p - center.p));

                    yprims_r.rho =
                        yright_mid.rho -
                        0.5 * minmod(plm_theta * (yright_mid.rho - center.rho),
                                        0.5 * (yright_most.rho - center.rho),
                                        plm_theta * (yright_most.rho - yright_mid.rho));

                    yprims_r.v1 = yright_mid.v1 -
                                    0.5 * minmod(plm_theta * (yright_mid.v1 - center.v1),
                                                0.5 * (yright_most.v1 - center.v1),
                                                plm_theta * (yright_most.v1 - yright_mid.v1));

                    yprims_r.v2 = yright_mid.v2 -
                                    0.5 * minmod(plm_theta * (yright_mid.v2 - center.v2),
                                                0.5 * (yright_most.v2 - center.v2),
                                                plm_theta * (yright_most.v2 - yright_mid.v2));

                    yprims_r.p = yright_mid.p -
                                    0.5 * minmod(plm_theta * (yright_mid.p - center.p),
                                                0.5 * (yright_most.p - center.p),
                                                plm_theta * (yright_most.p - yright_mid.p));

                    // Calculate the left and right states using the reconstructed PLM
                    // Primitive
                    ux_l = prims2cons(xprims_l);
                    ux_r = prims2cons(xprims_r);

                    uy_l = prims2cons(yprims_l);
                    uy_r = prims2cons(yprims_r);

                    f_l = calc_Flux(xprims_l);
                    f_r = calc_Flux(xprims_r);

                    g_l = calc_Flux(yprims_l, 2);
                    g_r = calc_Flux(yprims_r, 2);

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
                    } else {
                        f1 = calc_hll_flux(ux_l, ux_r, f_l, f_r, xprims_l, xprims_r, 1);
                        g1 = calc_hll_flux(uy_l, uy_r, g_l, g_r, yprims_l, yprims_r, 2);
                    }

                    // Left side Primitive in x
                    xprims_l.rho = xleft_mid.rho +
                                    0.5 * minmod(plm_theta * (xleft_mid.rho - xleft_most.rho),
                                                0.5 * (center.rho - xleft_most.rho),
                                                plm_theta * (center.rho - xleft_mid.rho));

                    xprims_l.v1 = xleft_mid.v1 +
                                    0.5 * minmod(plm_theta * (xleft_mid.v1 - xleft_most.v1),
                                                0.5 * (center.v1 - xleft_most.v1),
                                                plm_theta * (center.v1 - xleft_mid.v1));

                    xprims_l.v2 = xleft_mid.v2 +
                                    0.5 * minmod(plm_theta * (xleft_mid.v2 - xleft_most.v2),
                                                0.5 * (center.v2 - xleft_most.v2),
                                                plm_theta * (center.v2 - xleft_mid.v2));

                    xprims_l.p =
                        xleft_mid.p + 0.5 * minmod(plm_theta * (xleft_mid.p - xleft_most.p),
                                                    0.5 * (center.p - xleft_most.p),
                                                    plm_theta * (center.p - xleft_mid.p));

                    // Right side Primitive in x
                    xprims_r.rho =
                        center.rho - 0.5 * minmod(plm_theta * (center.rho - xleft_mid.rho),
                                                    0.5 * (xright_mid.rho - xleft_mid.rho),
                                                    plm_theta * (xright_mid.rho - center.rho));

                    xprims_r.v1 =
                        center.v1 - 0.5 * minmod(plm_theta * (center.v1 - xleft_mid.v1),
                                                    0.5 * (xright_mid.v1 - xleft_mid.v1),
                                                    plm_theta * (xright_mid.v1 - center.v1));

                    xprims_r.v2 =
                        center.v2 - 0.5 * minmod(plm_theta * (center.v2 - xleft_mid.v2),
                                                    0.5 * (xright_mid.v2 - xleft_mid.v2),
                                                    plm_theta * (xright_mid.v2 - center.v2));

                    xprims_r.p =
                        center.p - 0.5 * minmod(plm_theta * (center.p - xleft_mid.p),
                                                0.5 * (xright_mid.p - xleft_mid.p),
                                                plm_theta * (xright_mid.p - center.p));

                    // Left side Primitive in y
                    yprims_l.rho = yleft_mid.rho +
                                    0.5 * minmod(plm_theta * (yleft_mid.rho - yleft_most.rho),
                                                0.5 * (center.rho - yleft_most.rho),
                                                plm_theta * (center.rho - yleft_mid.rho));

                    yprims_l.v1 = yleft_mid.v1 +
                                    0.5 * minmod(plm_theta * (yleft_mid.v1 - yleft_most.v1),
                                                0.5 * (center.v1 - yleft_most.v1),
                                                plm_theta * (center.v1 - yleft_mid.v1));

                    yprims_l.v2 = yleft_mid.v2 +
                                    0.5 * minmod(plm_theta * (yleft_mid.v2 - yleft_most.v2),
                                                0.5 * (center.v2 - yleft_most.v2),
                                                plm_theta * (center.v2 - yleft_mid.v2));

                    yprims_l.p =
                        yleft_mid.p + 0.5 * minmod(plm_theta * (yleft_mid.p - yleft_most.p),
                                                    0.5 * (center.p - yleft_most.p),
                                                    plm_theta * (center.p - yleft_mid.p));

                    // Right side Primitive in y
                    yprims_r.rho =
                        center.rho - 0.5 * minmod(plm_theta * (center.rho - yleft_mid.rho),
                                                    0.5 * (yright_mid.rho - yleft_mid.rho),
                                                    plm_theta * (yright_mid.rho - center.rho));

                    yprims_r.v1 =
                        center.v1 - 0.5 * minmod(plm_theta * (center.v1 - yleft_mid.v1),
                                                    0.5 * (yright_mid.v1 - yleft_mid.v1),
                                                    plm_theta * (yright_mid.v1 - center.v1));

                    yprims_r.v2 =
                        center.v2 - 0.5 * minmod(plm_theta * (center.v2 - yleft_mid.v2),
                                                    0.5 * (yright_mid.v2 - yleft_mid.v2),
                                                    plm_theta * (yright_mid.v2 - center.v2));

                    yprims_r.p =
                        center.p - 0.5 * minmod(plm_theta * (center.p - yleft_mid.p),
                                                0.5 * (yright_mid.p - yleft_mid.p),
                                                plm_theta * (yright_mid.p - center.p));

                    // Calculate the left and right states using the reconstructed PLM
                    // Primitive
                    ux_l = prims2cons(xprims_l);
                    ux_r = prims2cons(xprims_r);

                    uy_l = prims2cons(yprims_l);
                    uy_r = prims2cons(yprims_r);

                    f_l = calc_Flux(xprims_l);
                    f_r = calc_Flux(xprims_r);

                    g_l = calc_Flux(yprims_l, 2);
                    g_r = calc_Flux(yprims_r, 2);

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
                    } else {
                        f2 = calc_hll_flux(ux_l, ux_r, f_l, f_r, xprims_l, xprims_r, 1);
                        g2 = calc_hll_flux(uy_l, uy_r, g_l, g_r, yprims_l, yprims_r, 2);
                    }

                    int real_loc = ycoordinate * xphysical_grid + xcoordinate;
                    switch (geometry[coord_system])
                    {
                    case simbi::Geometry::CARTESIAN:
                        dx = coord_lattice.dx1[xcoordinate];
                        dy = coord_lattice.dx2[ycoordinate];
                        cons_n[aid].D   += 0.5 * dt * (- (f1.D - f2.D)     / dx - (g1.D - g2.D)     / dy + sourceD[real_loc]);
                        cons_n[aid].S1  += 0.5 * dt * (- (f1.S1 - f2.S1)   / dx - (g1.S1 - g2.S1)   / dy + source_S1[real_loc]);
                        cons_n[aid].S2  += 0.5 * dt * (- (f1.S2 - f2.S2)   / dx - (g1.S2 - g2.S2)   / dy + source_S2[real_loc]);
                        cons_n[aid].tau += 0.5 * dt * (- (f1.tau - f2.tau) / dx - (g1.tau - g2.tau) / dy + source_tau[real_loc]);
                        break;
                    
                    case simbi::Geometry::SPHERICAL:
                        s1R   = coord_lattice.x1_face_areas[xcoordinate + 1];
                        s1L   = coord_lattice.x1_face_areas[xcoordinate];
                        rmean = coord_lattice.x1mean[xcoordinate];
                        dV1   = coord_lattice.dV1[xcoordinate];
                        dV2   = rmean * coord_lattice.dV2[ycoordinate];

                        pc   = prims[aid].p;
                        rhoc = prims[aid].rho, 
                        uc   = prims[aid].v1;
                        vc   = prims[aid].v2;

                        hc    = 1.0 + gamma * pc /(rhoc * (gamma - 1.0));
                        wc2   = 1.0/(1.0 - (uc * uc + vc * vc));


                        cons_n[aid] += Conserved{
                            // L(D)
                            -(f1.D * s1R - f2.D * s1L) / dV1 
                                - (g1.D * s2R - g2.D * s2L) / dV2 
                                    + sourceD[real_loc] * decay_const,

                            // L(S1)
                            -(f1.S1 * s1R - f2.S1 * s1L) / dV1 
                                - (g1.S1 * s2R - g2.S1 * s2L) / dV2 
                                    + rhoc * hc * wc2 * vc * vc / rmean + 2 * pc / rmean +
                                        source_S1[real_loc] * decay_const,

                            // L(S2)
                            -(f1.S2 * s1R - f2.S2 * s1L) / dV1
                                - (g1.S2 * s2R - g2.S2 * s2L) / dV2 
                                    - (rhoc * hc * wc2 * uc * vc / rmean - pc * coord_lattice.cot[ycoordinate] / rmean) 
                                        + source_S2[real_loc] * decay_const,

                            // L(tau)
                            -(f1.tau * s1R - f2.tau * s1L) / dV1 
                                - (g1.tau * s2R - g2.tau * s2L) / dV2 
                                    + source_tau[real_loc] * decay_const
                        } * dt * 0.5;
                        break;
                    }
                } // end ii loop
            } // end jj loop
        }

    } // end parallel region
    
};

//===================================================================================================================
//                                            SIMULATE
//===================================================================================================================
std::vector<std::vector<double>> SRHD2D::simulate2D(
    const std::vector<std::vector<double>> sources,
    double tstart, 
    double tend, 
    double init_dt, 
    double plm_theta,
    double engine_duration, 
    double chkpt_interval,
    std::string data_directory, 
    bool first_order,
    bool periodic, 
    bool linspace, 
    bool hllc)
{
    std::string tnow, tchunk, tstep;
    int total_zones = NX * NY;
    
    double round_place = 1 / chkpt_interval;
    double t = tstart;
    double t_interval =
        t == 0 ? floor(tstart * round_place + 0.5) / round_place
               : floor(tstart * round_place + 0.5) / round_place + chkpt_interval;

    std::string filename;

    this->sources = sources;
    this->first_order = first_order;
    this->periodic = periodic;
    this->hllc = hllc;
    this->linspace = linspace;
    this->lorentz_gamma = lorentz_gamma;
    this->plm_theta = plm_theta;
    this->dt        = init_dt;

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
    setup.xmax         = x1[xphysical_grid - 1];
    setup.xmin         = x1[0];
    setup.ymax         = x2[yphysical_grid - 1];
    setup.ymin         = x2[0];
    setup.NX           = NX;
    setup.NY           = NY;
    setup.coord_system = coord_system;

    cons.resize(nzones);
    prims.resize(nzones);

    // Define the source terms
    sourceD    = sources[0];
    source_S1  = sources[1];
    source_S2  = sources[2];
    source_tau = sources[3];

    // Copy the state array into real & profile variables
    for (size_t i = 0; i < state2D[0].size(); i++)
    {
        cons[i] =
            Conserved(state2D[0][i], state2D[1][i], state2D[2][i], state2D[3][i]);
    }
    cons_n = cons;
    n = 0;

    high_resolution_clock::time_point t1, t2;

    // Using a sigmoid decay function to represent when the source terms turn off.
    decay_const = 1.0 / (1.0 + exp(10.0 * (tstart - engine_duration)));

    // Set the Primitive from the initial conditions and initialize the pressure
    // guesses
    pressure_guess.resize(nzones);
    
    tchunk = "000000";
    int tchunk_order_of_mag = 2;
    int time_order_of_mag, num_zeros;

    // Declare I/O variables for Read/Write capability
    PrimData prods;
    sr2d::PrimitiveData transfer_prims;

    if (t == 0)
    {
        config_ghosts2D(cons, NX, NY, first_order);
    }

    if (first_order)
    {
        while (t < tend)
        {
            /* Compute the loop execution time */
            t1 = high_resolution_clock::now();

            cons2prim2D();
            evolve();
            config_ghosts2D(cons_n, NX, NY, true);
            cons = cons_n;
            t += dt;

            /* Compute the loop execution time */
            t2 = high_resolution_clock::now();
            duration<double> time_span = duration_cast<duration<double>>(t2 - t1);

            std::cout << std::fixed << std::setprecision(3) << std::scientific;
            std::cout << "\r"
                 << "dt: " << std::setw(5) << dt << "\t"
                 << "t: "  << std::setw(5) << t << "\t"
                 << "Zones per sec: " << total_zones / time_span.count() << std::flush;

            /* Write to a File every nth of a second */
            if (t >= t_interval)
            {
                // Check if time order of magnitude exceeds 
                // the hundreds place set by the tchunk std::string
                time_order_of_mag = std::floor(std::log10(t));
                if (time_order_of_mag > tchunk_order_of_mag){
                    tchunk.insert(0, "0");
                    tchunk_order_of_mag += 1;
                }
                
                transfer_prims = vecs2struct(prims);
                toWritePrim(&transfer_prims, &prods);
                tnow           = create_step_str(t_interval, tchunk);
                filename       = string_format("%d.chkpt." + tnow + ".h5", yphysical_grid);
                setup.t        = t;
                setup.dt       = dt;
                write_hdf5(data_directory, filename, prods, setup, 2, total_zones);
                t_interval += chkpt_interval;
            }
            adapt_dt();
            n++;
        }
    }
    else
    {
        while (t < tend)
        {
            /* Compute the loop execution time */
            t1 = high_resolution_clock::now();

            // First half step
            cons2prim2D();
            evolve();
            config_ghosts2D(cons_n, NX, NY, false);
            cons = cons_n;

            // Final half step
            cons2prim2D();
            evolve();
            config_ghosts2D(cons_n, NX, NY, false);
            cons = cons_n;

            t += dt;

            t2 = high_resolution_clock::now();
            auto time_span = duration_cast<duration<double>>(t2 - t1);
            
            std::cout << std::fixed << std::setprecision(3) << std::scientific;
            std::cout << "\r"
                 << "dt: " << std::setw(5) << dt << "\t"
                 << "t: "  << std::setw(5) << t << "\t"
                 << "Zones per sec: " << total_zones / time_span.count() << std::flush;

            
            decay_const = 1.0 / (1.0 + exp(10.0 * (t - engine_duration)));

            /* Write to a File every nth of a second */
            if (t >= t_interval)
            {
                // Check if time order of magnitude exceeds 
                // the hundreds place set by the tchunk std::string
                time_order_of_mag = std::floor(std::log10(t));
                if (time_order_of_mag > tchunk_order_of_mag){
                    tchunk.insert(0, "0");
                    tchunk_order_of_mag += 1;
                }
                
                transfer_prims = vecs2struct(prims);
                toWritePrim(&transfer_prims, &prods);
                tnow           = create_step_str(t_interval, tchunk);
                filename       = string_format("%d.chkpt." + tnow + ".h5", yphysical_grid);
                setup.t        = t;
                setup.dt       = dt;
                write_hdf5(data_directory, filename, prods, setup, 2, total_zones);
                t_interval += chkpt_interval;
            }
            adapt_dt();
            n++;
        }
    }

    std::cout << "\n ";
    cons2prim2D();
    transfer_prims = vecs2struct(prims);

    std::vector<std::vector<double>> solution(4, std::vector<double>(nzones));

    solution[0] = transfer_prims.rho;
    solution[1] = transfer_prims.v1;
    solution[2] = transfer_prims.v2;
    solution[3] = transfer_prims.p;

    return solution;
};
