/* 
* C++ Library to perform extensive hydro calculations
* to be later wrapped and plotted in Python
* Marcus DuPont
* New York University
* 07/15/2020
* Compressible Hydro Simulation
*/

#include "euler1D.hpp" 
#include <cmath>
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <map>


using namespace simbi;
using namespace std::chrono;


// Default Constructor 
Newtonian1D::Newtonian1D () {}

// Overloaded Constructor
Newtonian1D::Newtonian1D(
    std::vector< std::vector<real> > init_state, 
    real gamma, 
    real CFL, 
    std::vector<real> r,
    std::string coord_system = "cartesian") :

    init_state(init_state),
    gamma(gamma),
    r(r),
    coord_system(coord_system),
    CFL(CFL) {}

// Destructor 
Newtonian1D::~Newtonian1D() {}


// Typedefs because I'm lazy
typedef hydro1d::Conserved Conserved;
typedef hydro1d::Primitive Primitive;
typedef hydro1d::Eigenvals Eigenvals;
//--------------------------------------------------------------------------------------------------
//                          GET THE PRIMITIVE VECTORS
//--------------------------------------------------------------------------------------------------
/**
 * Return a vector containing the primitive
 * variables density (rho), pressure, and
 * velocity (v)
 */
void Newtonian1D::cons2prim(){
    #pragma omp parallel
    {
        real rho, pre, v;
        #pragma omp for schedule(static)
        for (int ii = 0; ii < nx; ii++)
        {  
            rho = cons[ii].rho;
            v   = cons[ii].m/rho;
            pre = (gamma - (real)1.0)*(cons[ii].e_dens - (real)0.5 * rho * v * v);
            prims [ii] = Primitive{rho, v, pre};
        }
    }
};

//----------------------------------------------------------------------------------------------------------
//                              EIGENVALUE CALCULATIONS
//----------------------------------------------------------------------------------------------------------


Eigenvals Newtonian1D::calc_eigenvals(const Primitive &left_prim, const Primitive &right_prim)
{
    Eigenvals lambdas;
    // Separate the left and right state components
    real rho_l = left_prim.rho;
    real v_l   = left_prim.v;
    real p_l   = left_prim.p;

    real rho_r    = right_prim.rho;
    real v_r      = right_prim.v;
    real p_r      = right_prim.p;

    real cs_r = std::sqrt(gamma * p_r/rho_r);
    real cs_l = std::sqrt(gamma * p_l/rho_l);

    switch (sim_solver)
    {
    case SOLVER::HLLE:
        lambdas.aR = std::max({v_l + cs_l, v_r + cs_r, (real)0.0}); 
        lambdas.aL = std::min({v_l - cs_l, v_r - cs_r, (real)0.0});
        return lambdas;
    
    case SOLVER::HLLC:
        real cbar   = (real)0.5*(cs_l + cs_r);
        real rhoBar = (real)0.5*(rho_l + rho_r);
        real pStar  = (real)0.5*(p_l + p_r) + (real)0.5*(v_l - v_r)*cbar*rhoBar;

        // Steps to Compute HLLC as described in Toro et al. 2019
        real z      = (gamma - 1.)/(2.*gamma);
        real num    = cs_l + cs_r - ( gamma-1.)/2 *(v_r - v_l);
        real denom  = cs_l/pow(p_l,z) + cs_r/pow(p_r, z);
        real p_term = num/denom;
        real qL, qR;

        pStar = pow(p_term, (1./z));

        if (pStar <= p_l){
            qL = 1.;
        } else {
            qL = sqrt(1. + ( (gamma + 1.)/(2.*gamma))*(pStar/p_l - 1.));
        }

        if (pStar <= p_r){
            qR = 1.;
        } else {
            qR = sqrt(1. + ( (gamma + 1.)/(2.*gamma))*(pStar/p_r - 1.));
        }

        real aL = v_l - qL*cs_l;
        real aR = v_r + qR*cs_r;

        real aStar = ( (p_r - p_l + rho_l*v_l*(aL - v_l) - rho_r*v_r*(aR - v_r))/
                        (rho_l*(aL - v_l) - rho_r*(aR - v_r) ) );

        lambdas.aL = aL;
        lambdas.aR = aR;
        lambdas.aStar = aStar;
        lambdas.pStar = pStar;

        return lambdas;
    }


};

// Adapt the CFL conditonal timestep
void Newtonian1D::adapt_dt(){
    real min_dt = INFINITY;
    #pragma omp parallel 
    {
        real dx, cs, cfl_dt;
        real v, pre, rho;
        int shift_i;

        // Compute the minimum timestep given CFL
        #pragma omp for schedule(static)
        for (int ii = 0; ii < active_zones; ii++){
            shift_i = ii + idx_active;
            dx      = coord_lattice.dx1[ii];

            rho = prims[shift_i].rho;
            v   = prims[shift_i].v;
            pre = prims[shift_i].p;

            cs = std::sqrt(gamma * pre/rho);
            cfl_dt = dx/(std::max({std::abs(v + cs), std::abs(v - cs)}));

            min_dt = std::min(min_dt, cfl_dt);
    
        }
    }

    dt = CFL * min_dt;
};

//----------------------------------------------------------------------------------------------------
//              STATE TENSOR CALCULATIONS
//----------------------------------------------------------------------------------------------------


// Get the (3,1) state tensor for computation. Used for Higher Order Reconstruction
Conserved Newtonian1D::prims2cons(const Primitive &prim)
{
    real energy = prim.p/(gamma - (real)1.0) + (real)0.5 * prim.rho * prim.v * prim.v;

    return Conserved{prim.rho, prim.rho * prim.v, energy};
};

//-----------------------------------------------------------------------------------------------------------
//                                            FLUX CALCULATIONS
//-----------------------------------------------------------------------------------------------------------

// Get the 1D Flux array (3,1)
Conserved Newtonian1D::calc_flux(const Primitive &prim)
{
    real energy = prim.p/(gamma - (real)1.0) + (real)0.5 * prim.rho * prim.v * prim.v;

    return Conserved{
        prim.rho * prim.v,
        prim.rho * prim.v * prim.v + prim.p,
        (energy + prim.p)*prim.v

    };
};

Conserved Newtonian1D::calc_hll_flux(
    const Primitive &left_prims,
    const Primitive &right_prims,
    const Conserved &left_state,
    const Conserved &right_state,
    const Conserved &left_flux,
    const Conserved &right_flux)
{
    Eigenvals lambda;
    lambda = calc_eigenvals(left_prims, right_prims);
    real am = lambda.aL;
    real ap = lambda.aR;

    // Compute the HLL Flux component-wise
    return (left_flux * ap - right_flux * am + (right_state - left_state) * am * ap)  / (ap - am) ;

};

Conserved Newtonian1D::calc_hllc_flux(
    const Primitive &left_prims,
    const Primitive &right_prims,
    const Conserved &left_state,
    const Conserved &right_state,
    const Conserved &left_flux,
    const Conserved &right_flux)
{
    Eigenvals lambda = calc_eigenvals(left_prims, right_prims);

    real aL = lambda.aL; 
    real aR = lambda.aR; 
    real ap = std::max((real)0.0, aR);
    real am = std::min((real)0.0, aL);
    if (0.0 <= aL){
        return left_flux;
    } 
    else if (0.0 >= aR){
        return right_flux;
    }

    real aStar = lambda.aStar;
    real pStar = lambda.pStar;

    auto hll_flux = (left_flux * ap + right_flux * am - (right_state - left_state) * am * ap)  / (am + ap) ;

    auto hll_state = (right_state * aR - left_state * aL - right_flux + left_flux)/(aR - aL);
    
    if (- aL <= (aStar - aL)){
        real pressure = left_prims.p;
        real v        = left_prims.v;
        real rho      = left_state.rho;
        real m        = left_state.m;
        real energy   = left_state.e_dens;
        real cofac    = 1./(aL - aStar);

        real rhoStar = cofac * (aL - v)*rho;
        real mstar   = cofac * (m*(aL - v) - pressure + pStar);
        real eStar   = cofac * (energy*(aL - v) + pStar*aStar - pressure*v);

        auto star_state = Conserved{rhoStar, mstar, eStar};

        // Compute the intermediate left flux
        return left_flux + (star_state - left_state) * aL;
    } else {
        real pressure = right_prims.p;
        real v        = right_prims.v;
        real rho      = right_state.rho;
        real m        = right_state.m;
        real energy   = right_state.e_dens;
        real cofac    = 1./(aR - aStar);

        real rhoStar = cofac * (aR - v)*rho;
        real mstar   = cofac * (m*(aR - v) - pressure + pStar);
        real eStar   = cofac * (energy*(aR - v) + pStar*aStar - pressure*v);

        auto star_state = Conserved{rhoStar, mstar, eStar};

        // Compute the intermediate right flux
        return right_flux + (star_state - right_state) * aR;
    }
    
};

//----------------------------------------------------------------------------------------------------------
//                                  UDOT CALCULATIONS
//----------------------------------------------------------------------------------------------------------

void Newtonian1D::evolve()
{
    #pragma omp parallel 
    {
        int coordinate;
        Conserved u_l, u_r;
        Conserved f_l, f_r, f1, f2;
        Primitive prims_l, prims_r;

        real dx, rmean, dV, sL, sR, pc;
        if (first_order)
        {
            real rho_l, rho_r, v_l, v_r, p_l, p_r;
            #pragma omp for nowait
            for (int ii = i_start; ii < i_bound; ii++)
            {
                if (periodic)
                {
                    coordinate = ii;
                    // Set up the left and right state interfaces for i+1/2
                    u_l     = cons[ii];
                    u_r     = roll(cons, ii + 1);
                    prims_l = prims[ii];
                    prims_r = roll(prims, ii + 1);
                }
                else
                {
                    coordinate = ii - 1;
                    // Set up the left and right state interfaces for i+1/2
                    u_l     = cons[ii];
                    u_r     = cons[ii + 1];
                    prims_l = prims[ii];
                    prims_r = prims[ii + 1];
                }

                f_l = calc_flux(prims_l);
                f_r = calc_flux(prims_r);

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
                    u_l     = roll(cons, ii - 1);
                    u_r     = cons[ii];
                    prims_l = roll(prims, ii - 1);
                    prims_r = prims[ii];
                }
                else
                {
                    u_l     = cons[ii - 1];
                    u_r     = cons[ii];
                    prims_l = prims[ii - 1];
                    prims_r = prims[ii];
                }

                f_l = calc_flux(prims_l);
                f_r = calc_flux(prims_r);

                // Calc HLL Flux at i-1/2 interface
                if (hllc)
                {
                    f2 = calc_hllc_flux(prims_l, prims_r, u_l, u_r, f_l, f_r);
                }
                else
                {
                    f2 = calc_hll_flux(prims_l, prims_r, u_l, u_r, f_l, f_r);
                }

                switch (geometry[coord_system])
                {
                case simbi::Geometry::CARTESIAN:
                    dx = coord_lattice.dx1[coordinate];
                    cons_n[ii].rho    += dt * (-(f1.rho - f2.rho)       / dx + sourceRho[coordinate]);
                    cons_n[ii].m      += dt * (-(f1.m - f2.m)           / dx + sourceMom[coordinate]);
                    cons_n[ii].e_dens += dt * (-(f1.e_dens - f2.e_dens) / dx + sourceE[coordinate]);
                    break;

                case simbi::Geometry::SPHERICAL:
                    pc = prims[ii].p;
                    sL = coord_lattice.face_areas[coordinate + 0];
                    sR = coord_lattice.face_areas[coordinate + 1];
                    dV = coord_lattice.dV[coordinate];
                    rmean = coord_lattice.x1mean[coordinate];

                    cons_n[ii].rho    += dt * (-(sR * f1.rho - sL * f2.rho) / dV + sourceRho[coordinate] * decay_constant);

                    cons_n[ii].m      += dt * (-(sR * f1.m - sL * f2.m) / dV + 2 * pc / rmean + sourceMom[coordinate] * decay_constant);

                    cons_n[ii].e_dens += dt * (-(sR * f1.e_dens - sL * f2.e_dens) / dV + sourceE[coordinate] * decay_constant);
                    break;
                }
            }
        }
        else
        {
            Primitive left_most, right_most, left_mid, right_mid, center;
            #pragma omp for nowait
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
                    coordinate  = ii - 2;
                    left_most   = prims[ii - 2];
                    left_mid    = prims[ii - 1];
                    center      = prims[ii];
                    right_mid   = prims[ii + 1];
                    right_most  = prims[ii + 2];
                }

                // Compute the reconstructed primitives at the i+1/2 interface

                // Reconstructed left primitives vector
                prims_l.rho =
                    center.rho + (real)0.5 * minmod(plm_theta * (center.rho - left_mid.rho),
                                            (real)0.5 * (right_mid.rho - left_mid.rho),
                                            plm_theta * (right_mid.rho - center.rho));

                prims_l.v = center.v + (real)0.5 * minmod(plm_theta * (center.v - left_mid.v),
                                                    (real)0.5 * (right_mid.v - left_mid.v),
                                                    plm_theta * (right_mid.v - center.v));

                prims_l.p = center.p + (real)0.5 * minmod(plm_theta * (center.p - left_mid.p),
                                                    (real)0.5 * (right_mid.p - left_mid.p),
                                                    plm_theta * (right_mid.p - center.p));

                // Reconstructed right primitives vector
                prims_r.rho = right_mid.rho -
                            (real)0.5 * minmod(plm_theta * (right_mid.rho - center.rho),
                                        (real)0.5 * (right_most.rho - center.rho),
                                        plm_theta * (right_most.rho - right_mid.rho));

                prims_r.v =
                    right_mid.v - (real)0.5 * minmod(plm_theta * (right_mid.v - center.v),
                                            (real)0.5 * (right_most.v - center.v),
                                            plm_theta * (right_most.v - right_mid.v));

                prims_r.p =
                    right_mid.p - (real)0.5 * minmod(plm_theta * (right_mid.p - center.p),
                                            (real)0.5 * (right_most.p - center.p),
                                            plm_theta * (right_most.p - right_mid.p));

                // Calculate the left and right states using the reconstructed PLM
                // primitives
                u_l = prims2cons(prims_l);
                u_r = prims2cons(prims_r);

                f_l = calc_flux(prims_l);
                f_r = calc_flux(prims_r);

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
                    left_mid.rho + (real)0.5 * minmod(plm_theta * (left_mid.rho - left_most.rho),
                                                (real)0.5 * (center.rho - left_most.rho),
                                                plm_theta * (center.rho - left_mid.rho));

                prims_l.v =
                    left_mid.v + (real)0.5 * minmod(plm_theta * (left_mid.v - left_most.v),
                                            (real)0.5 * (center.v - left_most.v),
                                            plm_theta * (center.v - left_mid.v));

                prims_l.p =
                    left_mid.p + (real)0.5 * minmod(plm_theta * (left_mid.p - left_most.p),
                                            (real)0.5 * (center.p - left_most.p),
                                            plm_theta * (center.p - left_mid.p));

                prims_r.rho =
                    center.rho - (real)0.5 * minmod(plm_theta * (center.rho - left_mid.rho),
                                            (real)0.5 * (right_mid.rho - left_mid.rho),
                                            plm_theta * (right_mid.rho - center.rho));

                prims_r.v = center.v - (real)0.5 * minmod(plm_theta * (center.v - left_mid.v),
                                                    (real)0.5 * (right_mid.v - left_mid.v),
                                                    plm_theta * (right_mid.v - center.v));

                prims_r.p = center.p - (real)0.5 * minmod(plm_theta * (center.p - left_mid.p),
                                                    (real)0.5 * (right_mid.p - left_mid.p),
                                                    plm_theta * (right_mid.p - center.p));

                // Calculate the left and right states using the reconstructed PLM
                // primitives
                u_l = prims2cons(prims_l);
                u_r = prims2cons(prims_r);

                f_l = calc_flux(prims_l);
                f_r = calc_flux(prims_r);

                if (hllc)
                {
                    f2 = calc_hllc_flux(prims_l, prims_r, u_l, u_r, f_l, f_r);
                }
                else
                {
                    f2 = calc_hll_flux(prims_l, prims_r, u_l, u_r, f_l, f_r);
                }

                switch (geometry[coord_system])
                {
                case simbi::Geometry::CARTESIAN:
                    dx = coord_lattice.dx1[coordinate];
                    cons_n[ii].rho    += (real)0.5 * dt * (-(f1.rho - f2.rho)       / dx + sourceRho[coordinate]);
                    cons_n[ii].m      += (real)0.5 * dt * (-(f1.m - f2.m)           / dx + sourceMom[coordinate]);
                    cons_n[ii].e_dens += (real)0.5 * dt * (-(f1.e_dens - f2.e_dens) / dx + sourceE[coordinate]);
                    break;

                case simbi::Geometry::SPHERICAL:
                    pc    = prims[ii].p;
                    sL    = coord_lattice.face_areas[coordinate + 0];
                    sR    = coord_lattice.face_areas[coordinate + 1];
                    dV    = coord_lattice.dV[coordinate];
                    rmean = coord_lattice.x1mean[coordinate];

                    cons_n[ii].rho    += (real)0.5 * dt * (-(sR * f1.rho - sL * f2.rho) / dV + sourceRho[coordinate] * decay_constant);

                    cons_n[ii].m      += (real)0.5 * dt * (-(sR * f1.m - sL * f2.m) / dV + 2 * pc / rmean + sourceMom[coordinate] * decay_constant);

                    cons_n[ii].e_dens += (real)0.5 * dt * (-(sR * f1.e_dens - sL * f2.e_dens) / dV + sourceE[coordinate] * decay_constant);
                    break;
                }
            }
        }
    } // end parallel region
    
};


 std::vector<std::vector<real> > Newtonian1D::simulate1D(
    std::vector<std::vector<real>> &sources,
    real tstart,
    real tend,
    real init_dt,
    real plm_theta,
    real engine_duration,
    real chkpt_interval,
    std::string data_directory,
    bool first_order,
    bool periodic,
    bool linspace,
    bool hllc)
{
    this->periodic        = periodic;
    this->first_order     = first_order;
    this->plm_theta       = plm_theta;
    this->linspace        = linspace;
    this->sourceRho       = sources[0];
    this->sourceMom       = sources[1];
    this->sourceE         = sources[2];
    this->hllc            = hllc;
    this->engine_duration = engine_duration;
    this->t               = tstart;
    this->dt              = init_dt;
    // Define the swap vector for the integrated state
    this->nx = init_state[0].size();

    if (periodic){
        this->idx_active    = 0;
        this->active_zones = nx;
        this->i_start      = 0;
        this->i_bound      = nx;
    } else {
        if (first_order){
            this->idx_active = 1;
            this->i_start   = 1;
            this->i_bound   = nx - 1;
            this->active_zones = nx - 2;
        } else {
            this->idx_active = 2;
            this->i_start    = 2;
            this->i_bound    = nx - 2;
            this->active_zones = nx - 4; 
        }
    }
    if (hllc){
        this->sim_solver = simbi::SOLVER::HLLC;
    } else {
        this->sim_solver = simbi::SOLVER::HLLE;
    }

    config_system();
    n = 0;
    // Write some info about the setup for writeup later
    std::string filename, tnow, tchunk;
    PrimData prods;
    real round_place = 1 / chkpt_interval;
    real t_interval =
        t == 0 ? floor(tstart * round_place + (real)0.5) / round_place
               : floor(tstart * round_place + (real)0.5) / round_place + chkpt_interval;
    DataWriteMembers setup;
    setup.xmax          = r[active_zones - 1];
    setup.xmin          = r[0];
    setup.xactive_zones = active_zones;
    setup.nx            = nx;
    setup.linspace      = linspace;
    setup.coord_system  = coord_system;


    cons.resize(nx);
    prims.resize(nx);
    // Copy the state array into real & profile variables
    for (size_t ii = 0; ii < nx; ii++)
    {
        cons [ii] = Conserved{init_state[0][ii], init_state[1][ii], init_state[2][ii]};
    }
    cons_n = cons;
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
    // Create Structure of Vectors (SoV) for trabsferring
    // data to files once ready
    hydro1d::PrimitiveArray transfer_prims;

    // Tools for file string formatting
    tchunk = "000000";
    int tchunk_order_of_mag = 2;
    int time_order_of_mag, num_zeros;

    if (first_order)
    {
        while (t < tend)
        {
            /* Compute the loop execution time */
            high_resolution_clock::time_point t1 = high_resolution_clock::now();
            cons2prim();
            evolve();
            if (periodic == false)
            {
                config_ghosts1D(cons_n, nx, true);
            }
            cons = cons_n;
            t += dt;

            /* Compute the loop execution time */
            high_resolution_clock::time_point t2 = high_resolution_clock::now();
            duration<real> time_span = duration_cast<duration<real>>(t2 - t1);

            std::cout << std::fixed << std::setprecision(3) << std::scientific;
            std::cout << "\r"
                      << "dt: " << std::setw(5) << dt << "\t"
                      << "t: "  << std::setw(5) << t << "\t"
                      << "Zones per sec: " << nx / time_span.count() << std::flush;
            adapt_dt();
            n++;
        }
    }
    else
    {
        while (t < tend)
        {
            /* Compute the loop execution time */
            high_resolution_clock::time_point t1 = high_resolution_clock::now();
            // First Half Step
            cons2prim();
            evolve();
            if (periodic == false)
            {
                config_ghosts1D(cons_n, nx, false);
            }
            cons = cons_n;
            // Final Half Step
            cons2prim();
            evolve();
            if (periodic == false)
            {
                config_ghosts1D(cons_n, nx, false);
            }
            cons = cons_n;
            t += dt;

            /* Compute the loop execution time */
            high_resolution_clock::time_point t2 = high_resolution_clock::now();
            duration<real> time_span = duration_cast<duration<real>>(t2 - t1);

            std::cout << std::fixed << std::setprecision(3) << std::scientific;
            std::cout << "\r"
                      << "dt: " << std::setw(5) << dt << "\t"
                      << "t: "  << std::setw(5) << t << "\t"
                      << "Zones per sec: " << nx / time_span.count() << std::flush;

            //--- Decay the source terms
            decay_constant = std::exp(-t / engine_duration);

            /* Write to a File every tenth of a second */
            if (t >= t_interval)
            {
                time_order_of_mag = std::floor(std::log10(t));
                if (time_order_of_mag > tchunk_order_of_mag)
                {
                    tchunk.insert(0, "0");
                    tchunk_order_of_mag += 1;
                }
                transfer_prims = vec2struct<hydro1d::PrimitiveArray, Primitive>(prims);
                writeToProd<hydro1d::PrimitiveArray, Primitive>(&transfer_prims, &prods);
                tnow        = create_step_str(t_interval, tchunk);
                filename    = string_format("%d.chkpt." + tnow + ".h5", active_zones);
                setup.t     = t;
                setup.dt    = dt;
                write_hdf5(data_directory, filename, prods, setup, 1, nx);
                t_interval += chkpt_interval;
            }
            adapt_dt();
            n++;
        }
    }
    cons2prim();
    std::cout << "\n";
    std::vector<std::vector<real>> solution(3, std::vector<real>(nx));
    for (size_t ii = 0; ii < nx; ii++)
    {
        solution[0][ii] = prims[ii].rho;
        solution[1][ii] = prims[ii].v;
        solution[2][ii] = prims[ii].p;
    }
    
    // write_data(u, tend, "sod");
    return solution;

 };
