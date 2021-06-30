/*
 * C++ Library to perform extensive hydro calculations
 * to be later wrapped and plotted in Python
 * Marcus DuPont
 * New York University
 * 07/15/2020
 * Compressible Hydro Simulation
 */

#include "helper_functions.h"
#include "srhd_1d.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <map>

using namespace std;
using namespace simbi;
using namespace chrono;

constexpr int MAX_ITER = 50;
// Default Constructor
SRHD::SRHD() {}

// Overloaded Constructor
SRHD::SRHD(vector<vector<double>> u_state, double gamma, double CFL,
           vector<double> r, string coord_system = "cartesian")
{
    this->state = u_state;
    this->gamma = gamma;
    this->r = r;
    this->coord_system = coord_system;
    this->CFL = CFL;
}

// Destructor
SRHD::~SRHD() {}

//================================================
//              DATA STRUCTURES
//================================================
typedef sr1d::Conserved Conserved;
typedef sr1d::Primitive Primitive;
typedef sr1d::Eigenvals Eigenvals;

//--------------------------------------------------------------------------------------------------
//                          GET THE PRIMITIVE VECTORS
//--------------------------------------------------------------------------------------------------

/**
 * Return a vector containing the primitive
 * variables density (rho), pressure, and
 * velocity (v)
 */
void SRHD::cons2prim1D(const std::vector<Conserved> &u_state)
{
    double rho, S, D, tau, pmin;
    double v, W, tol, f, g, peq, h;
    double eps, rhos, p, v2, et, c2;
    int iter = 0;

    for (int ii = 0; ii < Nx; ii++)
    {
        D   = u_state[ii].D;
        S   = u_state[ii].S;
        tau = u_state[ii].tau;

        peq = n != 0 ? pressure_guess[ii] : abs(abs(S) - tau - D);

        tol = D * 1.e-12;

        iter = 0;
        do
        {
            p = peq;
            et = tau + D + p;
            v2 = S * S / (et * et);
            W = 1.0 / sqrt(1.0 - v2);
            rho = D / W;

            eps = (tau + (1.0 - W) * D + (1. - W * W) * p) / (D * W);

            h = 1. + eps + p / rho;
            c2 = gamma * p / (h * rho);

            g = c2 * v2 - 1.0;
            f = (gamma - 1.0) * rho * eps - p;

            peq = p - f / g;
            iter++;
            if (iter >= MAX_ITER)
            {
                std::cout << "\n";
                std::cout << "Cons2Prim cannot converge"
                          << "\n";
                exit(EXIT_FAILURE);
            }

        } while (abs(peq - p) >= tol);

        v = S / (tau + D + peq);

        W = 1. / sqrt(1 - v * v);
        pressure_guess[ii] = peq;
        prims[ii] = Primitive{D / W, v, peq};
    }
};

//----------------------------------------------------------------------------------------------------------
//                              EIGENVALUE CALCULATIONS
//----------------------------------------------------------------------------------------------------------
Eigenvals SRHD::calc_eigenvals(const Primitive &prims_l,
                               const Primitive &prims_r)
{

    // Initialize your important variables
    double v_r, v_l, p_r, p_l, cs_r, cs_l;
    double rho_l, rho_r, h_l, h_r, aL, aR;
    double sL, sR, minlam_l, minlam_r, pluslam_l, pluslam_r;
    double vbar, cbar;
    Eigenvals lambda;

    // Compute L/R Sound Speeds
    rho_l = prims_l.rho;
    p_l = prims_l.p;
    v_l = prims_l.v;
    h_l = 1. + gamma * p_l / (rho_l * (gamma - 1.));
    cs_l = sqrt(gamma * p_l / (rho_l * h_l));

    rho_r = prims_r.rho;
    p_r = prims_r.p;
    v_r = prims_r.v;
    h_r = 1. + gamma * p_r / (rho_r * (gamma - 1.));
    cs_r = sqrt(gamma * p_r / (rho_r * h_r));

    // Compute waves based on Schneider et al. 1993 Eq(31 - 33)
    vbar = 0.5 * (v_l + v_r);
    cbar = 0.5 * (cs_r + cs_l);
    double br = (vbar + cbar) / (1 + vbar * cbar);
    double bl = (vbar - cbar) / (1 - vbar * cbar);

    lambda.aL = min(bl, (v_l - cs_l) / (1 - v_l * cs_l));
    lambda.aR = max(br, (v_r + cs_r) / (1 + v_l * cs_l));

    // Get Wave Speeds based on Mignone & Bodo Eqs. (21 - 23)
    // sL          = cs_l*cs_l/(gamma*gamma*(1 - cs_l*cs_l));
    // sR          = cs_r*cs_r/(gamma*gamma*(1 - cs_r*cs_r));
    // minlam_l    = (v_l - sqrt(sL*(1 - v_l*v_l + sL)))/(1 + sL);
    // minlam_r    = (v_r - sqrt(sR*(1 - v_r*v_r + sR)))/(1 + sR);
    // pluslam_l   = (v_l + sqrt(sL*(1 - v_l*v_l + sL)))/(1 + sL);
    // pluslam_r   = (v_r + sqrt(sR*(1 - v_r*v_r + sR)))/(1 + sR);

    // lambda.aL = (minlam_l < minlam_r)   ? minlam_l : minlam_r;
    // lambda.aR = (pluslam_l > pluslam_r) ? pluslam_l : pluslam_r;

    return lambda;
};

// Adapt the CFL conditonal timestep
double SRHD::adapt_dt(const std::vector<Primitive> &prims)
{

    double r_left, r_right, left_cell, right_cell, dr, cs;
    double min_dt, cfl_dt;
    double h, rho, p, v, vPLus, vMinus;

    min_dt = 0;

    // Compute the minimum timestep given CFL
    for (int ii = 0; ii < pgrid_size; ii++)
    {
        dr  = coord_lattice.dx1[ii];
        rho = prims[ii + idx_shift].rho;
        p   = prims[ii + idx_shift].p;
        v   = prims[ii + idx_shift].v;

        h = 1. + gamma * p / (rho * (gamma - 1.));
        cs = sqrt(gamma * p / (rho * h));

        vPLus  = (v + cs) / (1 + v * cs);
        vMinus = (v - cs) / (1 - v * cs);

        cfl_dt = dr / (max(abs(vPLus), abs(vMinus)));

        if (ii > 0)
        {
            min_dt = min(min_dt, cfl_dt);
        }
        else
        {
            min_dt = cfl_dt;
        }
    }

    return CFL * min_dt;
};

//----------------------------------------------------------------------------------------------------
//              STATE ARRAY CALCULATIONS
//----------------------------------------------------------------------------------------------------

// Get the (3,1) state array for computation. Used for Higher Order
// Reconstruction
Conserved SRHD::calc_state(double rho, double v, double pressure)
{

    Conserved state;
    double W, h;

    h = 1. + gamma * pressure / (rho * (gamma - 1.));
    W = 1. / sqrt(1 - v * v);

    state.D = rho * W;
    state.S = rho * h * W * W * v;
    state.tau = rho * h * W * W - pressure - rho * W;

    return state;
};

Conserved SRHD::calc_hll_state(const Conserved &left_state,
                               const Conserved &right_state,
                               const Conserved &left_flux,
                               const Conserved &right_flux,
                               const Primitive &left_prims,
                               const Primitive &right_prims)
{
    double aL, aR;
    Conserved hll_states;

    Eigenvals lambda = calc_eigenvals(left_prims, right_prims);

    aL = lambda.aL;
    aR = lambda.aR;

    hll_states.D =
        (aR * right_state.D - aL * left_state.D - right_flux.D + left_flux.D) /
        (aR - aL);

    hll_states.S =
        (aR * right_state.S - aL * left_state.S - right_flux.S + left_flux.S) /
        (aR - aL);

    hll_states.tau = (aR * right_state.tau - aL * left_state.tau -
                      right_flux.tau + left_flux.tau) /
                     (aR - aL);

    return hll_states;
}

Conserved SRHD::calc_intermed_state(const Primitive &prims,
                                    const Conserved &state, const double a,
                                    const double aStar, const double pStar)
{
    double pressure, v, S, D, tau, E, Estar;
    double DStar, Sstar, tauStar;
    Eigenvals lambda;
    Conserved star_state;

    pressure = prims.p;
    v = prims.v;

    D = state.D;
    S = state.S;
    tau = state.tau;
    E = tau + D;

    DStar = ((a - v) / (a - aStar)) * D;
    Sstar = (1. / (a - aStar)) * (S * (a - v) - pressure + pStar);
    Estar = (1. / (a - aStar)) * (E * (a - v) + pStar * aStar - pressure * v);
    tauStar = Estar - DStar;

    star_state.D = DStar;
    star_state.S = Sstar;
    star_state.tau = tauStar;

    return star_state;
}

//-----------------------------------------------------------------------------------------------------------
//                                            FLUX CALCULATIONS
//-----------------------------------------------------------------------------------------------------------

// Get the 1D Flux array (3,1)
Conserved SRHD::calc_flux(double rho, double v, double pressure)
{

    Conserved flux;

    // The Flux components
    double mom, energy_dens, zeta, D, S, tau, h, W;

    W = 1. / sqrt(1 - v * v);
    h = 1. + gamma * pressure / (rho * (gamma - 1.));
    D = rho * W;
    S = rho * h * W * W * v;
    tau = rho * h * W * W - pressure - W * rho;

    mom = D * v;
    energy_dens = S * v + pressure;
    zeta = (tau + pressure) * v;

    flux.D = mom;
    flux.S = energy_dens;
    flux.tau = zeta;

    return flux;
};

Conserved
SRHD::calc_hll_flux(const Primitive &left_prims, const Primitive &right_prims,
                    const Conserved &left_state, const Conserved &right_state,
                    const Conserved &left_flux, const Conserved &right_flux)
{
    Conserved hll_flux;
    double aLm, aRp;

    Eigenvals lambda = calc_eigenvals(left_prims, right_prims);

    // Grab the necessary wave speeds
    double aR = lambda.aR;
    double aL = lambda.aL;

    aLm = (aL < 0.0) ? aL : 0.0;
    aRp = (aR > 0.0) ? aR : 0.0;

    // Compute the HLL Flux component-wise
  return  (left_flux * aRp -  right_flux * aLm +
            (right_state - left_state) * aLm * aRp ) / (aRp - aLm) ;
};

Conserved
SRHD::calc_hllc_flux(const Primitive &left_prims, const Primitive &right_prims,
                     const Conserved &left_state, const Conserved &right_state,
                     const Conserved &left_flux, const Conserved &right_flux)
{

    Conserved hllc_flux;
    Conserved hll_flux;

    Conserved starState;
    Conserved hll_state;
    double aL, aR, aStar, pStar;

    Eigenvals lambda = calc_eigenvals(left_prims, right_prims);

    aL = lambda.aL;
    aR = lambda.aR;

    if (0.0 <= aL)
    {
        return left_flux;
    } else if (0.0 >= aR)
    {
        return right_flux;
    }

    hll_flux = calc_hll_flux(left_prims, right_prims, left_state, right_state,
                                left_flux, right_flux);

    hll_state = calc_hll_state(left_state, right_state, left_flux, right_flux,
                                left_prims, right_prims);

    double e = hll_state.tau + hll_state.D;
    double s = hll_state.S;
    double fs = hll_flux.S;
    double fe = hll_flux.tau + hll_flux.D;

    aStar = calc_intermed_wave(e, s, fs, fe);
    pStar = -fe * aStar + fs;

    if (aL < 0.0 && 0.0 <= aStar)
    {
        starState =
        calc_intermed_state(left_prims, left_state, aL, aStar, pStar);

        return left_flux + (starState - left_state) * aL;
    } else {
        starState =
        calc_intermed_state(right_prims, right_state, aR, aStar, pStar);

        return right_flux + (starState - right_state) * aR;
    }

    };

    //----------------------------------------------------------------------------------------------------------
    //                                  UDOT CALCULATIONS
    //----------------------------------------------------------------------------------------------------------

    std::vector<Conserved> SRHD::u_dot1D(std::vector<Conserved> & u_state)
    {

        int coordinate;
        Conserved u_l, u_r;
        Conserved f_l, f_r, f1, f2;
        Primitive prims_l, prims_r;
        std::vector<Conserved> L(Nx);

        switch (geometry[coord_system])
        {
        case simbi::Geometry::CARTESIAN:
            {   
                double dx;
                if (first_order)
                {
                    double rho_l, rho_r, v_l, v_r, p_l, p_r;
                    for (int ii = i_start; ii < i_bound; ii++)
                    {
                        if (periodic)
                        {
                            coordinate = ii;
                            // Set up the left and right state interfaces for i+1/2
                            u_l.D   = u_state[ii].D;
                            u_l.S   = u_state[ii].D;
                            u_l.tau = u_state[ii].tau;

                            u_r = roll(u_state, ii + 1);
                        }
                        else
                        {
                            coordinate = ii - 1;
                            // Set up the left and right state interfaces for i+1/2
                            u_l.D   = u_state[ii].D;
                            u_l.S   = u_state[ii].S;
                            u_l.tau = u_state[ii].tau;

                            u_r.D   = u_state[ii + 1].D;
                            u_r.S   = u_state[ii + 1].S;
                            u_r.tau = u_state[ii + 1].tau;
                        }

                        prims_l.rho = prims[ii].rho;
                        prims_l.v   = prims[ii].v;
                        prims_l.p   = prims[ii].p;

                        prims_r.rho = prims[ii + 1].rho;
                        prims_r.v   = prims[ii + 1].v;
                        prims_r.p   = prims[ii + 1].p;

                        f_l = calc_flux(prims_l.rho, prims_l.v, prims_l.p);
                        f_r = calc_flux(prims_r.rho, prims_r.v, prims_r.p);

                        // Calc HLL Flux at i+1/2 interface
                        if (hllc)
                        {
                            f1 = calc_hllc_flux(prims_l, prims_r, u_l, u_r, f_l, f_r);
                        }
                        else
                        {
                            f1 = calc_hll_flux(prims_l, prims_r, u_l, u_r, f_l, f_r);
                        }

                        // Set up the left and right state interfaces for i-1/2
                        if (periodic)
                        {
                            u_l = roll(u_state, ii - 1);

                            u_r.D   = u_state[ii].D;
                            u_r.S   = u_state[ii].S;
                            u_r.tau = u_state[ii].tau;
                        }
                        else
                        {
                            u_l.D   = u_state[ii - 1].D;
                            u_l.S   = u_state[ii - 1].S;
                            u_l.tau = u_state[ii - 1].tau;

                            u_r.D   = u_state[ii].D;
                            u_r.S   = u_state[ii].S;
                            u_r.tau = u_state[ii].tau;
                        }

                        prims_l.rho = prims[ii - 1].rho;
                        prims_l.v   = prims[ii - 1].v;
                        prims_l.p   = prims[ii - 1].p;

                        prims_r.rho = prims[ii].rho;
                        prims_r.v   = prims[ii].v;
                        prims_r.p   = prims[ii].p;

                        f_l = calc_flux(prims_l.rho, prims_l.v, prims_l.p);
                        f_r = calc_flux(prims_r.rho, prims_r.v, prims_r.p);

                        // Calc HLL Flux at i-1/2 interface
                        if (hllc)
                        {
                            f2 = calc_hllc_flux(prims_l, prims_r, u_l, u_r, f_l, f_r);
                        }
                        else
                        {
                            f2 = calc_hll_flux(prims_l, prims_r, u_l, u_r, f_l, f_r);
                        }

                        dx = coord_lattice.dx1[coordinate];
                        L[coordinate].D   = -(f1.D - f2.D)     / dx + sourceD[coordinate];
                        L[coordinate].S   = -(f1.S - f2.S)     / dx + sourceS[coordinate];
                        L[coordinate].tau = -(f1.tau - f2.tau) / dx + source0[coordinate];
                    }
                    return L;

                } else {
                    Primitive left_most, right_most, left_mid, right_mid, center;
                    for (int ii = i_start; ii < i_bound; ii++)
                    {
                        if (periodic)
                        {
                            // Declare the c[i-2],c[i-1],c_i,c[i+1], c[i+2] variables
                            coordinate = ii;
                            left_most  = roll(prims, ii - 2);
                            left_mid   = roll(prims, ii - 1);
                            center     = prims[ii];
                            right_mid  = roll(prims, ii + 1);
                            right_most = roll(prims, ii + 2);
                        }
                        else
                        {
                            coordinate = ii - 2;
                            left_most  = prims[ii - 2];
                            left_mid   = prims[ii - 1];
                            center     = prims[ii];
                            right_mid  = prims[ii + 1];
                            right_most = prims[ii + 2];
                        }

                        // Compute the reconstructed primitives at the i+1/2 interface

                        // Reconstructed left primitives vector
                        prims_l.rho =
                            center.rho + 0.5 * minmod(theta * (center.rho - left_mid.rho),
                                                    0.5 * (right_mid.rho - left_mid.rho),
                                                    theta * (right_mid.rho - center.rho));

                        prims_l.v = center.v + 0.5 * minmod(theta * (center.v - left_mid.v),
                                                            0.5 * (right_mid.v - left_mid.v),
                                                            theta * (right_mid.v - center.v));

                        prims_l.p = center.p + 0.5 * minmod(theta * (center.p - left_mid.p),
                                                            0.5 * (right_mid.p - left_mid.p),
                                                            theta * (right_mid.p - center.p));

                        // Reconstructed right primitives vector
                        prims_r.rho = right_mid.rho -
                                    0.5 * minmod(theta * (right_mid.rho - center.rho),
                                                0.5 * (right_most.rho - center.rho),
                                                theta * (right_most.rho - right_mid.rho));

                        prims_r.v =
                            right_mid.v - 0.5 * minmod(theta * (right_mid.v - center.v),
                                                    0.5 * (right_most.v - center.v),
                                                    theta * (right_most.v - right_mid.v));

                        prims_r.p =
                            right_mid.p - 0.5 * minmod(theta * (right_mid.p - center.p),
                                                    0.5 * (right_most.p - center.p),
                                                    theta * (right_most.p - right_mid.p));

                        // Calculate the left and right states using the reconstructed PLM
                        // primitives
                        u_l = calc_state(prims_l.rho, prims_l.v, prims_l.p);
                        u_r = calc_state(prims_r.rho, prims_r.v, prims_r.p);

                        f_l = calc_flux(prims_l.rho, prims_l.v, prims_l.p);
                        f_r = calc_flux(prims_r.rho, prims_r.v, prims_r.p);

                        if (hllc)
                        {
                            f1 = calc_hllc_flux(prims_l, prims_r, u_l, u_r, f_l, f_r);
                        }
                        else
                        {
                            f1 = calc_hll_flux(prims_l, prims_r, u_l, u_r, f_l, f_r);
                        }

                        // Do the same thing, but for the right side interface [i - 1/2]
                        prims_l.rho =
                            left_mid.rho + 0.5 * minmod(theta * (left_mid.rho - left_most.rho),
                                                        0.5 * (center.rho - left_most.rho),
                                                        theta * (center.rho - left_mid.rho));

                        prims_l.v =
                            left_mid.v + 0.5 * minmod(theta * (left_mid.v - left_most.v),
                                                    0.5 * (center.v - left_most.v),
                                                    theta * (center.v - left_mid.v));

                        prims_l.p =
                            left_mid.p + 0.5 * minmod(theta * (left_mid.p - left_most.p),
                                                    0.5 * (center.p - left_most.p),
                                                    theta * (center.p - left_mid.p));

                        prims_r.rho =
                            center.rho - 0.5 * minmod(theta * (center.rho - left_mid.rho),
                                                    0.5 * (right_mid.rho - left_mid.rho),
                                                    theta * (right_mid.rho - center.rho));

                        prims_r.v = center.v - 0.5 * minmod(theta * (center.v - left_mid.v),
                                                            0.5 * (right_mid.v - left_mid.v),
                                                            theta * (right_mid.v - center.v));

                        prims_r.p = center.p - 0.5 * minmod(theta * (center.p - left_mid.p),
                                                            0.5 * (right_mid.p - left_mid.p),
                                                            theta * (right_mid.p - center.p));

                        // Calculate the left and right states using the reconstructed PLM
                        // primitives
                        u_l = calc_state(prims_l.rho, prims_l.v, prims_l.p);
                        u_r = calc_state(prims_r.rho, prims_r.v, prims_r.p);

                        f_l = calc_flux(prims_l.rho, prims_l.v, prims_l.p);
                        f_r = calc_flux(prims_r.rho, prims_r.v, prims_r.p);

                        if (hllc)
                        {
                            f2 = calc_hllc_flux(prims_l, prims_r, u_l, u_r, f_l, f_r);
                        }
                        else
                        {
                            f2 = calc_hll_flux(prims_l, prims_r, u_l, u_r, f_l, f_r);
                        }

                        dx = coord_lattice.dx1[coordinate];

                        L[coordinate].D   = -(f1.D - f2.D)     / dx  + sourceD[coordinate];
                        L[coordinate].S   = -(f1.S - f2.S)     / dx  + sourceS[coordinate];
                        L[coordinate].tau = -(f1.tau - f2.tau) / dx  + source0[coordinate];
                    }
                    return L;
                }   
            }
        case simbi::Geometry::SPHERICAL:
            {
                double rmean, dV, sL, sR, pc;
                if (first_order)
                {
                    for (int ii = i_start; ii < i_bound; ii++)
                    {
                        if (periodic)
                        {
                            coordinate = ii;
                            // Set up the left and right state interfaces for i+1/2
                            u_l = u_state[ii];
                            u_r = roll(u_state, ii + 1);
                        }
                        else
                        {
                            // Shift the index for C++ [0] indexing
                            coordinate = ii - 1;
                            u_l = u_state[ii];
                            u_r = u_state[ii + 1];
                        }

                        prims_l = prims[ii];
                        prims_r = prims[ii + 1];


                        f_l = calc_flux(prims_l.rho, prims_l.p, prims_l.v);
                        f_r = calc_flux(prims_r.rho, prims_r.p, prims_r.v);

                        // Calc HLL Flux at i+1/2 interface
                        f1 = calc_hll_flux(prims_l, prims_r, u_l, u_r, f_l, f_r);

                        // Get the central pressure
                        pc = prims_l.p;

                        // Set up the left and right state interfaces for i-1/2
                        if (periodic)
                        {
                            u_l = roll(u_state, ii - 1);
                            u_r = u_state[ii];
                        }
                        else
                        {
                            u_l = u_state[ii - 1];
                            u_r = u_state[ii];
                        }

                        prims_l = prims[ii - 1];
                        prims_r = prims[ii];

                        f_l = calc_flux(prims_l.rho, prims_l.p, prims_l.v);
                        f_r = calc_flux(prims_r.rho, prims_r.p, prims_r.v);

                        // Calc HLL Flux at i-1/2 interface
                        f2 = calc_hll_flux(prims_l, prims_r, u_l, u_r, f_l, f_r);

                        // Outflow the left/right boundaries
                        sL    = coord_lattice.face_areas[coordinate + 0];
                        sR    = coord_lattice.face_areas[coordinate + 1];
                        dV    = coord_lattice.dV[coordinate];
                        rmean = coord_lattice.dx1[coordinate];

                        L[coordinate].D   = -(sR * f1.D   - sL * f2.D)   / dV + sourceD[coordinate];
                        L[coordinate].S   = -(sR * f1.S   - sL * f2.S)   / dV +  2 * pc / rmean + sourceS[coordinate];
                        L[coordinate].tau = -(sR * f1.tau - sL * f2.tau) / dV + source0[coordinate];
                    }
                    return L;
                } else {
                    Primitive left_most, right_most, left_mid, right_mid, center;
                    for (int ii = i_start; ii < i_bound; ii++)
                    {
                        if (periodic)
                        {
                            coordinate = ii;
                            // Declare the c[i-2],c[i-1],c_i,c[i+1], c[i+2] variables
                            left_most  = roll(prims, ii - 2);
                            left_mid   = roll(prims, ii - 1);
                            center     = prims[ii];
                            right_mid  = roll(prims, ii + 1);
                            right_most = roll(prims, ii + 2);

                        }
                        else
                        {
                            // Adjust for beginning input of L vector
                            coordinate = ii - 2;
                            left_most  = prims[ii - 2];
                            left_mid   = prims[ii - 1];
                            center     = prims[ii];
                            right_mid  = prims[ii + 1];
                            right_most = prims[ii + 2];
                        }

                        // Compute the reconstructed primitives at the i+1/2 interface

                        // Reconstructed left primitives vector
                        prims_l.rho =
                            center.rho + 0.5 * minmod(theta * (center.rho - left_mid.rho),
                                                    0.5 * (right_mid.rho - left_mid.rho),
                                                    theta * (right_mid.rho - center.rho));

                        prims_l.v = center.v + 0.5 * minmod(theta * (center.v - left_mid.v),
                                                            0.5 * (right_mid.v - left_mid.v),
                                                            theta * (right_mid.v - center.v));

                        prims_l.p = center.p + 0.5 * minmod(theta * (center.p - left_mid.p),
                                                            0.5 * (right_mid.p - left_mid.p),
                                                            theta * (right_mid.p - center.p));

                        // Reconstructed right primitives vector
                        prims_r.rho = right_mid.rho -
                                    0.5 * minmod(theta * (right_mid.rho - center.rho),
                                                0.5 * (right_most.rho - center.rho),
                                                theta * (right_most.rho - right_mid.rho));

                        prims_r.v =
                            right_mid.v - 0.5 * minmod(theta * (right_mid.v - center.v),
                                                    0.5 * (right_most.v - center.v),
                                                    theta * (right_most.v - right_mid.v));

                        prims_r.p =
                            right_mid.p - 0.5 * minmod(theta * (right_mid.p - center.p),
                                                    0.5 * (right_most.p - center.p),
                                                    theta * (right_most.p - right_mid.p));

                        // Calculate the left and right states using the reconstructed PLM
                        // primitives
                        u_l = calc_state(prims_l.rho, prims_l.v, prims_l.p);
                        u_r = calc_state(prims_r.rho, prims_r.v, prims_r.p);

                        f_l = calc_flux(prims_l.rho, prims_l.v, prims_l.p);
                        f_r = calc_flux(prims_r.rho, prims_r.v, prims_r.p);

                        if (hllc)
                        {
                            f1 = calc_hllc_flux(prims_l, prims_r, u_l, u_r, f_l, f_r);
                        }
                        else
                        {
                            f1 = calc_hll_flux(prims_l, prims_r, u_l, u_r, f_l, f_r);
                        }

                        // Do the same thing, but for the right side interface [i - 1/2]
                        prims_l.rho =
                            left_mid.rho + 0.5 * minmod(theta * (left_mid.rho - left_most.rho),
                                                        0.5 * (center.rho - left_most.rho),
                                                        theta * (center.rho - left_mid.rho));

                        prims_l.v =
                            left_mid.v + 0.5 * minmod(theta * (left_mid.v - left_most.v),
                                                    0.5 * (center.v - left_most.v),
                                                    theta * (center.v - left_mid.v));

                        prims_l.p =
                            left_mid.p + 0.5 * minmod(theta * (left_mid.p - left_most.p),
                                                    0.5 * (center.p - left_most.p),
                                                    theta * (center.p - left_mid.p));

                        prims_r.rho =
                            center.rho - 0.5 * minmod(theta * (center.rho - left_mid.rho),
                                                    0.5 * (right_mid.rho - left_mid.rho),
                                                    theta * (right_mid.rho - center.rho));

                        prims_r.v = center.v - 0.5 * minmod(theta * (center.v - left_mid.v),
                                                            0.5 * (right_mid.v - left_mid.v),
                                                            theta * (right_mid.v - center.v));

                        prims_r.p = center.p - 0.5 * minmod(theta * (center.p - left_mid.p),
                                                            0.5 * (right_mid.p - left_mid.p),
                                                            theta * (right_mid.p - center.p));

                        // Calculate the left and right states using the reconstructed PLM
                        // primitives
                        u_l = calc_state(prims_l.rho, prims_l.v, prims_l.p);
                        u_r = calc_state(prims_r.rho, prims_r.v, prims_r.p);

                        f_l = calc_flux(prims_l.rho, prims_l.v, prims_l.p);
                        f_r = calc_flux(prims_r.rho, prims_r.v, prims_r.p);

                        if (hllc)
                        {
                            f2 = calc_hllc_flux(prims_l, prims_r, u_l, u_r, f_l, f_r);
                        }
                        else
                        {
                            f2 = calc_hll_flux(prims_l, prims_r, u_l, u_r, f_l, f_r);
                        }

                        // Get Central Pressure
                        pc = center.p;

                        // Outflow the left/right boundaries
                        sL    = coord_lattice.face_areas[coordinate];
                        sR    = coord_lattice.face_areas[coordinate + 1];
                        dV    = coord_lattice.dV[coordinate];
                        rmean = coord_lattice.x1mean[coordinate];

                        L[coordinate].D  = -(sR * f1.D - sL * f2.D) / dV +
                                        sourceD[coordinate] * decay_constant;

                        L[coordinate].S  = -(sR * f1.S - sL * f2.S) / dV + 2 * pc / rmean +
                                         sourceS[coordinate] * decay_constant;

                        L[coordinate].tau = -(sR * f1.tau - sL * f2.tau) / dV +
                                            source0[coordinate] * decay_constant;
                    }

                    return L;
                }
            }
        }
    };

    vector<vector<double>>
    SRHD::simulate1D(vector<double> & lorentz_gamma, vector<vector<double>> & sources,
                     float tstart = 0.0, float tend = 0.1, float dt = 1.e-4,
                     double theta = 1.5, double engine_duration = 10,
                     double chkpt_interval = 0.1, string data_directory = "data/",
                     bool first_order = true, bool periodic = false,
                     bool linspace = true, bool hllc = false)
    {

        this->periodic = periodic;
        this->first_order = first_order;
        this->theta = theta;
        this->linspace = linspace;
        this->lorentz_gamma = lorentz_gamma;
        this->sourceD = sources[0];
        this->sourceS = sources[1];
        this->source0 = sources[2];
        this->hllc = hllc;
        this->engine_duration = engine_duration;
        this->t = tstart;
        // Define the swap vector for the integrated state
        this->Nx = lorentz_gamma.size();

        if (periodic)
        {
            this->idx_shift = 0;
            this->i_start   = 0;
            this->i_bound   = Nx;
        }
        else
        {
            if (first_order)
            {
                this->idx_shift  = 1;
                this->pgrid_size = Nx - 2;
                this->i_start    = 1;
                this->i_bound    = Nx - 1;
            }
            else
            {
                this->idx_shift  = 2;
                this->pgrid_size = Nx - 4;
                this->i_start    = 2;
                this->i_bound    = Nx - 2;
            }
        }
        config_system();
        int i_real;
        n = 0;
        std::vector<Conserved> u_p, u, u1, u2, udot;
        // Write some info about the setup for writeup later
        string filename, tnow, tchunk;
        PrimData prods;
        double round_place = 1 / chkpt_interval;
        double t_interval =
            t == 0 ? floor(tstart * round_place + 0.5) / round_place
                   : floor(tstart * round_place + 0.5) / round_place + chkpt_interval;
        DataWriteMembers setup;
        setup.xmax = r[pgrid_size - 1];
        setup.xmin = r[0];
        setup.xactive_zones = pgrid_size;
        setup.NX = Nx;

        // Create Structure of Vectors (SoV) for trabsferring 
        // data to files once ready
        sr1d::PrimitiveArray transfer_prims; 

        u.resize(Nx);
        prims.resize(Nx);
        pressure_guess.resize(Nx);
        // Copy the state array into real & profile variables
        for (size_t ii = 0; ii < Nx; ii++)
        {
            u[ii] = Conserved{state[0][ii],
                              state[1][ii],
                              state[2][ii]};
        }

        if ((coord_system == "spherical") && (linspace))
        {
            this->coord_lattice = CLattice1D(r, simbi::Geometry::SPHERICAL);
            coord_lattice.config_lattice(simbi::Cellspacing::LINSPACE);
        }
        else if ((coord_system == "spherical") && (!linspace))
        {
            this->coord_lattice = CLattice1D(r, simbi::Geometry::SPHERICAL);
            coord_lattice.config_lattice(simbi::Cellspacing::LOGSPACE);
        }
        else
        {
            this->coord_lattice = CLattice1D(r, simbi::Geometry::CARTESIAN);
            coord_lattice.config_lattice(simbi::Cellspacing::LINSPACE);
        }

        cons2prim1D(u);

        n++;

        if (first_order)
        {
            u_p = u;
            while (t < tend)
            {
                /* Compute the loop execution time */
                high_resolution_clock::time_point t1 = high_resolution_clock::now();
                if (t == 0)
                {
                    config_ghosts1D(u, Nx);
                }

                // Compute the L(u).
                udot = u_dot1D(u);

                for (int ii = 0; ii < pgrid_size; ii++)
                {
                    i_real = ii + idx_shift;
                    u_p[i_real] = u[i_real] + udot[ii] * dt;
                }

                // Readjust the ghost cells at i-1,i+1 if not periodic
                if (periodic == false)
                {
                    config_ghosts1D(u_p, Nx);
                }

                cons2prim1D(u_p);

                // Adjust the timestep
                if (t > 0)
                {
                    dt = adapt_dt(prims);
                }

                // Swap the arrays
                u.swap(u_p);
                t += dt;

                /* Compute the loop execution time */
                high_resolution_clock::time_point t2 = high_resolution_clock::now();
                duration<double> time_span = duration_cast<duration<double>>(t2 - t1);

                cout << fixed << setprecision(3) << scientific;
                cout << "\r"
                     << "dt: " << setw(5) << dt << "\t"
                     << "t: " << setw(5) << t << "\t"
                     << "Zones per sec: " << Nx / time_span.count() << flush;
            }
        }
        else
        {
            tchunk = "000000";
            int tchunk_order_of_mag = 2;
            int time_order_of_mag, num_zeros;

            u1 = u;
            u2 = u;
            u_p = u;
            while (t < tend)
            {
                high_resolution_clock::time_point t1 = high_resolution_clock::now();
                // Compute the REAL udot array, purging the ghost cells.
                if (t == 0)
                {
                    config_ghosts1D(u, Nx, first_order);
                }

                udot = u_dot1D(u);

                for (int ii = 0; ii < pgrid_size; ii++)
                {
                    i_real = ii + idx_shift;
                    u1[i_real] = u[i_real] + udot[ii] * dt;
                }

                // Readjust the ghost cells at i-2,i-1,i+1,i+2
                if (periodic == false)
                {
                    config_ghosts1D(u1, Nx, false);
                }

                cons2prim1D(u1);
                udot = u_dot1D(u1);

                for (int ii = 0; ii < pgrid_size; ii++)
                {
                    i_real = ii + idx_shift;
                    u2[i_real] = u[i_real] * 0.5 + u1[i_real] * 0.5 + udot[ii] * dt * 0.5;
                }

                cons2prim1D(u2);

                if (periodic == false)
                {
                    config_ghosts1D(u2, Nx, false);
                }

                // Adjust the timestep
                if (t > 0)
                {
                    dt = adapt_dt(prims);
                }

                // Swap the arrays
                u.swap(u2);

                t += dt;

                //--- Decay the source terms
                this->decay_constant = exp(-t / engine_duration);

                /* Compute the loop execution time */
                high_resolution_clock::time_point t2 = high_resolution_clock::now();
                duration<double> time_span = duration_cast<duration<double>>(t2 - t1);

                cout << fixed << setprecision(3) << scientific;
                cout << "\r"
                     << "dt: " << setw(5) << dt << "\t"
                     << "t: " << setw(5) << t << "\t"
                     << "Zones per sec: " << Nx / time_span.count() << flush;

                /* Write to a File every tenth of a second */
                if (t >= t_interval)
                {
                    time_order_of_mag = std::floor(std::log10(t));
                    if (time_order_of_mag > tchunk_order_of_mag){
                        tchunk.insert(0, "0");
                        tchunk_order_of_mag += 1;
                    }
                    transfer_prims = vec2struct<sr1d::PrimitiveArray, Primitive>(prims);
                    writeToProd<sr1d::PrimitiveArray, Primitive>(&transfer_prims, &prods);
                    tnow = create_step_str(t_interval, tchunk);
                    filename = string_format("%d.chkpt." + tnow + ".h5", pgrid_size);
                    setup.t = t;
                    setup.dt = dt;
                    write_hdf5(data_directory, filename, prods, setup, 1, Nx);
                    t_interval += chkpt_interval;
                }

                n++;
            }
        }
        cout << "\n";
        cons2prim1D(u);
        vector<vector<double>> final_prims(3, vector<double>(Nx, 0));
        for (size_t ii = 0; ii < Nx; ii++)
        {
            final_prims[0][ii] = prims[ii].rho;
            final_prims[1][ii] = prims[ii].v;
            final_prims[2][ii] = prims[ii].p;
        }
        
        

        return final_prims;
    };
