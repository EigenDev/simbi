/*
Header file for low-level synchrotron radiation calculations
*/
#ifndef RAD_DOUBLES_HPP
#define RAD_DOUBLES_HPP

#include <vector>
#include "constants.hpp"

namespace sogbo_rad
{
    struct sim_conditions
    {
        double dt, theta_obs, ad_gamma, current_time;
        std::vector<double> nus;
    };

    struct quant_scales
    {
        double time_scale, pre_scale, rho_scale, v_scale, length_scale;
    };
    

    /*
    The Doppler Boost Factor
    */
    const double calc_delta_doppler(const double lorentz_gamma, const double beta[3], const double nhat[3]);

    /*
    The velocity of the flow in dimensions of speed of ligt
    */
    constexpr double calc_beta(const double gamma_beta);

    /*
    The Lorentz factor
    */
    constexpr double calc_lorentz_gamma(const double gamma_beta);

    /*
    The magnetic field behind the shock
    */
    constexpr double calc_shock_bfield(const double rho_e, const double eps_b);

    /*
    The canonical gyration frequency for a particle in a magnetic field
    */
    constexpr double calc_gyration_frequency(const double bfield);

    /**
        Calc bolometric synhrotron power
        
        Params:
        --------------------------------------
        lorentz_gamma:   lorentz factor 
        ub:              magnetic field energy density 
        beta:            dimensionless flow veloccity 
        
        Return:
        --------------------------------------
        Total synchrotron power
    */
    constexpr double calc_total_synch_power(const double lorentz_gamma, const double ub, const double beta);

    /**
        Calculate the number of photons per energy (gamma_e) bin
        
        Params:
        ------------------------------------------
        volume:              volume of the cell
        n_e:                 number density of electrons
        nu_g:                gyration frequency of electrons
        gamma_e:             lorentz factor of electrons
        u_b:                 magnetic field energy density 
        dt:                  the checkpoint time interval 
        p:                   electron spectral index 
        
        Return:
        ------------------------------------------
        number of photons per energy bin
    */
    const double calc_nphotons_per_bin(
        double    volume, 
        double     n_e, 
        double nu_g, 
        double     ub, 
        double    dt,
        double gamma_e, 
        double beta,
        double p);

    /**
    Calculate the number of photons per electron in energy bin
    
    Params: 
    -------------------------------------
    power:                 the total synchrotron power per electron
    nu_c:                  the critical frequency of electrons
    dt:                    the time step size between checkpoints
    
    Return:
    -------------------------------------
    calculate the number of photons emitted per electron in some energy bin
    */
    constexpr double calc_nphotons_other(const double power, const double nu_c, const double dt);

    //  Power-law generator for pdf(x) ~ x^{g-1} for a<=x<=b
    // const double gen_random_from_powerlaw(double a, double b, double g, int size=1)
    // {
    //     const double rng = np.random.default_rng();
    //     const double r = rng.random(size=size)[0];
    //     const double ag = std::pow(a, g);
    //     const double bg = std::pow(b, g);
    //     return std::pow(ag + (bg - ag) * r, (1.0 / g));
    // }

    // calculate magnitude of vector
    const double vector_magnitude(const std::vector<double> a);
    
    // calculate vector dot product
    const double vector_dotproduct(const std::vector<double> a, const std::vector<double> b);

    /**
        Calculate the number density of electrons per lorentz facor 
        Params:
        --------------------------------------
        n_e:                       total number density of electrons
        gamma_e:                   the electron lorentz factor
        dgamma:                    the energy bin size 
        p:                         electron spectral index 
        
        Return:
        -------------------------------------
        number density of electrons at a given energy
    */ 
    constexpr double calc_nelectrons_at_gamma_e(const double n_e, const double gamma_e, const double p);

    // Calculate the synchrotron frequency as function of lorentz_factor
    constexpr double calc_nu(const double gamma_e, const double nu_g);
    
    // Calculate the critical Lorentz factor as function of time and magnetic field
    constexpr double calc_critical_lorentz(const double bfield, const double time);

    // Calculate the maximum power per frequency as Eq. (5) in Sari, Piran. Narayan(1999)
    constexpr double calc_max_power_per_frequency(double bfield);
    /**
     *  Calculate the peak emissivity per frequency per equation (A3) in
        https://iopscience.iop.org/article/10.1088/0004-637X/749/1/44/pdf
    */ 
    constexpr double calc_emissivity(const double bfield, const double n, const double p);

    /*
        Calculate the minimum lorentz factor of electrons in the distribution
        
        Params:
        ------------------------------
        eps_e:              fraction of internal energy due to shocked electrons
        p:                  spectral electron number index 
        
        Return:
        ------------------------------
        Minimum lorentz factor of electrons
    */
    constexpr double calc_minimum_lorentz(const double eps_e, const double e_thermal, const double n, const double p);

    /*
    ~---------------------------------------
    ~Compute the flux according to https://arxiv.org/abs/astro-ph/9712005
    ~---------------------------------------
    */
    const double calc_powerlaw_flux(
        const double &power_max, 
        const double p,
        const double nu_prime, 
        const double nu_c, 
        const double nu_m
    );

    const std::vector<double> calc_fnu_2d(
        const sim_conditions args,
        const quant_scales   qscales,
        std::vector<std::vector<double>> &fields,
        std::vector<std::vector<double>> &mesh,  
        std::vector<double> &tbin_edges,
        std::vector<double> &flux_array,
        const int chkpt_idx,
        const int data_dim = 2
    );
    
} // namespace sogbo_rad


#endif