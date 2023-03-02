/**
    A radiation module that will be used to calculate synchtron lights curves, spectra, and sky maps
    from high-energy physical processes
    @file rad.cpp
    @author Marcus DuPont
    @version 0.1 05/20/22
*/
#include <cmath>
#include "rad_doubles.hpp"

namespace sogbo_rad
{
    
    /*
    The Doppler Boost Factor
    */
    const double calc_delta_doppler(const double lorentz_gamma, const std::vector<double> beta_vec, const std::vector<double> nhat)
    {
        return 1.0 / (lorentz_gamma * (1.0 - vector_dotproduct(beta_vec, nhat)));
    }

    /*
    The velocity of the flow in dimensions of speed of ligt
    */
    constexpr double calc_beta(const double gamma_beta)
    {
        return std::sqrt(1.0 - 1.0 / (1.0 + gamma_beta * gamma_beta));
    }

    /*
    The Lorentz factor
    */
    constexpr double calc_lorentz_gamma(const double gamma_beta)
    {
        return std::sqrt(1.0 + gamma_beta * gamma_beta);
    }

    /*
    The magnetic field behind the shock
    */
    constexpr double calc_shock_bfield(const double rho_e, const double eps_b)
    {
        return std::sqrt(8.0 * M_PI * eps_b * rho_e);
    }

    /*
    The canonical gyration frequency for a particle in a magnetic field
    */
    constexpr double calc_gyration_frequency(const double bfield)
    {
        const auto frequency_for_unit_field = (3.0 / 4.0 / M_PI) * (constants::e_charge) / (constants::m_e * constants::c_light);
        return frequency_for_unit_field  * bfield;
    }

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
    constexpr double calc_total_synch_power(const double lorentz_gamma, const double ub, const double beta)
    {
        return (4.0 / 3.0) * constants::sigmaT * constants::c_light * beta * beta * lorentz_gamma * lorentz_gamma * ub;
    }

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
        double p)
    {
        const auto a = (8.0 * M_PI * volume / (3.0 * constants::h_planck * nu_g));
        const auto b = constants::sigmaT * constants::c_light * beta * beta * ub * n_e;
        const auto c = std::pow(gamma_e, -(p + 1.0));
        return a * b * c * dt;
    }

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
    constexpr double calc_nphotons_other(const double power, const double nu_c, const double dt)
    {
        return power * dt / (constants::h_planck * nu_c);
    }

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
    const double vector_magnitude(const std::vector<double> a)
    {
        double mag = 0;
        for (const auto val : a)
        {
            mag += val * val;
        }
        return std::sqrt(mag);
    }
    
    // calculate vector dot product
    const double vector_dotproduct(const std::vector<double> a, const std::vector<double> b)
    {
        double mag = 0;
        for (int ii = 0; ii < a.size(); ii++)
        {
            mag += a[ii] * b[ii];
        }
        return mag;
    }

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
    constexpr double calc_nelectrons_at_gamma_e(const double n_e, const double gamma_e, const double p)
    {
        return n_e * std::pow(gamma_e, -p );
    }

    // Calculate the synchrotron frequency as function of lorentz_factor
    constexpr double calc_nu(const double gamma_e, const double nu_g)
    {
        return nu_g * gamma_e * gamma_e;
    }
    
    // Calculate the critical Lorentz factor as function of time and magnetic field
    constexpr double calc_critical_lorentz(const double bfield, const double time)
    {
        return (6.0 * M_PI * constants::m_e * constants::c_light) / (constants::sigmaT * bfield * bfield * time);
    }

    // Calculate the maximum power per frequency as Eq. (5) in Sari, Piran. Narayan(1999)
    constexpr double calc_max_power_per_frequency(double bfield)
    {
        return (constants::m_e * constants::c_light * constants::c_light * constants::sigmaT) / (3.0 * constants::e_charge) * bfield;
    }
    /**
     *  Calculate the peak emissivity per frequency per equation (A3) in
        https://iopscience.iop.org/article/10.1088/0004-637X/749/1/44/pdf
    */ 
    constexpr double calc_emissivity(const double bfield, const double n, const double p)
    {
        return (
            (9.6323/ 8.0 / M_PI) * (p - 1.0) / (3.0 * p - 1.0) * std::sqrt(3.0) * 
            std::pow(constants::e_charge, 3) / (constants::m_e * constants::c_light * constants::c_light) * n * bfield
        );
    }

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
    constexpr double calc_minimum_lorentz(const double eps_e, const double e_thermal, const double n, const double p)
    {
         return eps_e * (p - 2.0) / (p - 1.0) * e_thermal / (n * constants::m_e * constants::c_light* constants::c_light);
    }

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
    )
    {
        const bool slow_cool   = nu_c > nu_m;
        auto power_with_breaks = power_max;
        if (slow_cool)
        {   
            const bool slow_break1  = nu_prime < nu_m;
            const bool slow_break2  = (nu_prime < nu_c) & (nu_prime > nu_m);
            if (slow_break1)
            {
                power_with_breaks *= std::pow(nu_prime / nu_m, (1.0 / 3.0));  
                
            } else if (slow_break2) {
                power_with_breaks *= std::pow(nu_prime / nu_m, -0.5 * (p - 1.0));
            } else {
                power_with_breaks *= std::pow(nu_c  / nu_m, -0.5 * (p - 1.0)) * std::pow(nu_prime / nu_c, -0.5 * p);
            }

        } else {
            const bool fast_break1  = nu_prime < nu_c;
            const bool fast_break2  = (nu_prime < nu_m) & (nu_prime > nu_c);
            if (fast_break1)
            {
                power_with_breaks *= std::pow(nu_prime / nu_c, (1.0 / 3.0));  
            } else if(fast_break2) {
                power_with_breaks *= std::pow(nu_prime / nu_c, -0.5);
            } else {
                power_with_breaks *= std::pow(nu_m  / nu_c, -0.5) * std::pow(nu_prime / nu_m, -0.5 * p);
            }
        }        
        return power_with_breaks;      
    }

    /**
     * Compute the spectral flux due to synchrotron emission
     * 
     * @param args       a struct containing the simulation conditions
     * @param qscales    a struct containing the relevant dimensionful scales of the problem
     * @param fields     a 2D array of the primitives vairables rho, gamma_beta, and pressure
     * @param mesh       a 2D array of dimensionles values for the mesh centroids
     * @param tbin_edges a 1D array of the time bin edges for the flux calculations
     * @param fbin_edges a 1D array of the frequency bin edges for the flux calculartions
     * @param flux_array a flattened 1D array in which the summed frequencies in each bin will live
     * @param chkpt_idx  the integer index of the checkpoint file  
     * 
     * @return the summed fluxes in each frequency / time bin declared
    */
    const std::vector<double> calc_fnu_2d(
        const sim_conditions args,
        const quant_scales   qscales,
        std::vector<std::vector<double>> &fields,
        std::vector<std::vector<double>> &mesh,  
        std::vector<double> &tbin_edges,
        std::vector<double> &flux_array,
        const int chkpt_idx,
        const int data_dim
    )
    {
        // Place observer along chosen axis
        const std::vector<double> obs_hat  = {std::sin(args.theta_obs), 0.0, std::cos(args.theta_obs)};

        const auto rho = fields[0];             // fluid frame density 
        const auto gb  = fields[1];             // four-velocity
        const auto pre = fields[2];             // pressure
        const auto nt  = tbin_edges.size();     // time bin size
        const auto nf  = args.nus.size();       // frequency bin size


        // Extract the geomtry of the mesh
        const auto x1      = mesh[0];
        const auto x2      = mesh[1];
        const auto ni      = x1.size();
        const auto nj      = x2.size();
        const auto x1max   = x1[ni - 1];
        const auto x1min   = x1[0];
        const auto dlogx1  = std::log10(x1max / x1min) / (ni - 1);
        const auto x2max   = x2[nj - 1];
        const auto x2min   = x2[0];
        const auto dx2     = (x2max - x2min) / (nj - 1);
        const bool on_axis = args.theta_obs == 0;
        
        int nk = 1;
        double sin_phi = 0;
        double cos_phi = 1.0;
        double dx3     = 2.0 * M_PI;
        // Check whether to do 3D (off-axis) or not
        std::vector<double> x3;
        double x3max, x3min;
        if (!on_axis)
        {
            x3     = mesh[2];
            nk     = x3.size();
            x3max  = x3[nk - 1];
            x3min  = x3[0];
            dx3    = (x3max - x3min) / (nk - 1);
        }

        int jreal = 0;
        int kreal = 0;
        for (int kk=0; kk < nk; kk++)
        {       
            if (!on_axis)
            {
                const double x3l     = (kk > 0 ) ? x2min + (kk - 0.5) * dx3 :  x3min;
                const double x3r     = (kk < nk - 1) ? x3l + dx3 * (kk == 0 ? 0.5 : 1.0) :  x3max; 
                sin_phi              = std::sin(x3[kk]);
                cos_phi              = std::cos(x3[kk]);
                dx3                  = x3r - x3l;

                // If the data is 3D, then there is a real k-space to pull data from
                kreal = (data_dim > 2) * kk;
            }     
            #pragma omp parallel
            for (int jj = 0; jj < nj; jj++)
            {
                const double x2l     = (jj > 0 ) ? x2min + (jj - 0.5) * dx2 :  x2min;
                const double x2r     = (jj < nj - 1) ? x2l + dx2 * (jj == 0 ? 0.5 : 1.0) :  x2max; 
                const double dcos    = std::cos(x2l) - std::cos(x2r);

                // radial unit vector   
                const std::vector<double> rhat = {std::sin(x2[jj]) * cos_phi, std::sin(x2[jj]) * sin_phi, std::cos(x2[jj])}; 

                // Data greater than 1D? Cool, there is a j space to pull data from
                const int jreal = (data_dim > 1) * jj;
                #pragma omp for nowait
                for (int ii = 0; ii < ni; ii++)
                {
                    const auto central_idx  = kreal * ni * nj + jreal * ni + ii;
                    const auto beta         = calc_beta(gb[central_idx]);
                    const auto w            = calc_lorentz_gamma(gb[central_idx]);
                    const auto t_prime      = args.current_time * qscales.time_scale ;
                    const auto t_emitter    = t_prime / w;
                    //================================================================
                    //                    HYDRO CONDITIONS
                    //================================================================
                    const double p     = 2.5;  // Electron number index
                    const double eps_b = 0.1;  // Magnetic field fraction of internal energy 
                    const double eps_e = 0.1;  // shocked electrons fraction of internal energy
                    
                    const auto rho_einternal = pre[central_idx] * qscales.pre_scale / (args.ad_gamma - 1.0);  // internal energy density
                    const auto bfield        = calc_shock_bfield(rho_einternal, eps_b);                       // magnetic field based on equipartition
                    const auto n_e_proper    = rho[central_idx] * qscales.rho_scale / constants::m_p;         // electron number density
                    const auto nu_g          = calc_gyration_frequency(bfield);                               // gyration frequency
                    const auto d             = 1e28;                                                          // distance to source
                    const auto gamma_min     = calc_minimum_lorentz(eps_e, rho_einternal, n_e_proper, p);     // Minimum Lorentz factor of electrons 
                    const auto gamma_crit    = calc_critical_lorentz(bfield, t_emitter);                      // Critical Lorentz factor of electrons

                    // step size between checkpoints
                    const auto dt = args.dt * qscales.time_scale ;       

                    // Calc cell volumes
                    const double x1l    = (ii > 0 ) ? x1min * std::pow(10, (ii - 0.5) * dlogx1) :  x1min;
                    const double x1r    = (ii < ni - 1) ? x1l * std::pow(10, dlogx1 * (ii == 0 ? 0.5 : 1.0)) : x1max;
                    const auto dvolume  = dx3 * dcos * (1.0 / 3.0) * (x1r * x1r * x1r - x1l * x1l * x1l) * qscales.length_scale * qscales.length_scale * qscales.length_scale;      

                    // observer time
                    const auto t_obs = t_prime - x1[ii] * qscales.length_scale * vector_dotproduct(rhat, obs_hat) / constants::c_light;

                    const std::vector<double> beta_vec = {beta * rhat[0], beta * rhat[1], beta * rhat[2]};
                    
                    // Calculate the maximum flux based on the average bolometric power per electron
                    const double nu_c         = calc_nu(gamma_crit, nu_g);                                   // Critical frequency
                    const double nu_m         = calc_nu(gamma_min, nu_g);                                    // Minimum frequency
                    const double delta_doppler          = calc_delta_doppler(w, beta_vec, obs_hat);                    // Doppler factor
                    const double eps_m       = calc_emissivity(bfield, n_e_proper, p);                      // Emissivity per cell 
                    const double power_prime = dvolume * eps_m * delta_doppler * delta_doppler;             // Total emitted power per unit frequency in each cell volume
                    
                    const auto t_obs_day = t_obs * constants::sec2day;
                    // loop through the given frequencies and put them in their respective locations in dictionary
                    for (int fidx = 0; fidx < nf; fidx++)
                    {
                        // The frequency we see is doppler boosted, so account for that
                        const double  nu_source  = args.nus[fidx] / delta_doppler;
                        const double power_cool  = calc_powerlaw_flux(power_prime, p, nu_source, nu_c, nu_m);
                        const double f_nu        = (power_cool / (4.0 * M_PI * d * d)) * constants::cgs2mJy;

                        // place the fluxes in the appropriate time bins
                        for (int tidx = 0; tidx < nt - 1; tidx++)
                        {
                            const double t1 = tbin_edges[tidx + 0];
                            const double t2 = tbin_edges[tidx + 1];
                            if ((t_obs_day - t1) <= (t2 - t1))
                            {
                                // the effective lifetime of the emitting cell must be accounted for
                                const auto dt_day = dt * constants::sec2day;
                                const auto dt_obs = t2 - t1;
                                const double trat = (chkpt_idx > 0) ? dt_day / dt_obs : 1.0;

                                // Sum the fluxes in the given time bin
                                flux_array[fidx * (nt - 1) + tidx] += trat * f_nu;
                                break;
                            }
                        }
                    }
                } // end inner parallel loop
                
            } // end outer parallel loop
        }

        return flux_array;
    }
    
} // namespace sogbo_rad