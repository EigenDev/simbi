/*
Header file for low-level synchrotron radiation calculations
*/
#ifndef RAD_UNITS_HPP
#define RAD_UNITS_HPP

#include "units/constants.hpp"
#include "units/units.hpp"
#include <vector>

namespace sogbo_rad {
    struct sim_conditions {
        double dt, theta_obs, adiabatic_index, current_time, p, z, eps_e, eps_b,
            d_L;
        std::vector<double> nus;
    };

    struct quant_scales {
        double time_scale, pre_scale, rho_scale, v_scale, length_scale;
    };

    /*
    The Doppler Boost Factor
    */
    double calc_delta_doppler(
        const double lorentz_factor,
        const std::vector<double>& beta,
        const std::vector<double>& nhat
    );

    /*
    The velocity of the flow in dimensions of speed of ligt
    */
    double calc_beta(const double gamma_beta);

    /*
    The Lorentz factor
    */
    double calc_lorentz_factor(const double gamma_beta);

    /*
    The magnetic field behind the shock
    */
    units::mag_field
    calc_shock_bfield(const units::edens rho_e, const double eps_b);

    /*
    The canonical gyration frequency for a particle in a magnetic field
    */
    units::frequency calc_gyration_frequency(const units::mag_field bfield);

    /**
        Calc bolometric synhrotron power

        Params:
        --------------------------------------
        lorentz_factor:   lorentz factor
        ub:              magnetic field energy density
        beta:            dimensionless flow veloccity

        Return:
        --------------------------------------
        Total synchrotron power
    */
    units::power calc_total_synch_power(
        const double lorentz_factor,
        const units::edens ub,
        const double beta
    );

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
    double calc_nphotons_per_bin(
        units::volume volume,
        units::ndens n_e,
        units::frequency nu_g,
        units::edens ub,
        units::mytime dt,
        double gamma_e,
        double beta,
        double p
    );

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
    double calc_nphotons_other(
        const units::power power,
        const units::frequency nu_c,
        const units::mytime dt
    );

    //  Power-law generator for pdf(x) ~ x^{g-1} for a<=x<=b
    // const double gen_random_from_powerlaw(double a, double b, double g, int
    // size=1)
    // {
    //     const double rng = np.random.default_rng();
    //     const double r = rng.random(size=size)[0];
    //     const double ag = std::pow(a, g);
    //     const double bg = std::pow(b, g);
    //     return std::pow(ag + (bg - ag) * r, (1.0 / g));
    // }

    // calculate magnitude of vector
    double vector_magnitude(const std::vector<double>& a);

    // calculate vector dot product
    double vector_dotproduct(
        const std::vector<double>& a,
        const std::vector<double>& b
    );

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
    units::ndens calc_nelectrons_at_gamma_e(
        const units::ndens n_e,
        const double gamma_e,
        const double p
    );

    // Calculate the synchrotron frequency as function of lorentz_factor
    units::frequency calc_nu(const double gamma_e, const units::frequency nu_g);

    // Calculate the critical Lorentz factor as function of time and magnetic
    // field
    double calc_critical_lorentz(
        const units::mag_field bfield,
        const units::mytime time
    );

    // Calculate the maximum power per frequency as Eq. (5) in Sari, Piran.
    // Narayan(1999)
    units::energy calc_max_power_per_frequency(units::mag_field bfield);
    /**
     *  Calculate the peak emissivity per frequency per equation (A3) in
        https://iopscience.iop.org/article/10.1088/0004-637X/749/1/44/pdf
    */
    units::emissivity calc_emissivity(
        const units::mag_field bfield,
        const units::ndens n,
        const double p
    );

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
    double calc_minimum_lorentz(
        const double eps_e,
        const units::edens e_thermal,
        const units::ndens n,
        const double p
    );
    std::vector<double>
    vector_multiply(const std::vector<double>& a, const std::vector<double>& b);
    std::vector<double>
    scale_vector(const std::vector<double>& a, const double scalar);
    std::vector<double>
    vector_subtract(const std::vector<double>& a, const std::vector<double>& b);
    std::vector<double>
    vector_add(const std::vector<double>& a, const std::vector<double>& b);

    void log_events(
        const sim_conditions args,
        const quant_scales qscales,
        std::vector<std::vector<double>>& fields,
        std::vector<std::vector<double>>& mesh,
        std::vector<double>& photon_distribution,
        std::vector<double>& four_position,
        const int data_dim
    );

    /*
    ~---------------------------------------
    ~Compute the flux according to https://arxiv.org/abs/astro-ph/9712005
    ~---------------------------------------
    */
    units::spec_power calc_powerlaw_flux(
        const units::spec_power& power_max,
        const double p,
        const units::frequency nu_prime,
        const units::frequency nu_c,
        const units::frequency nu_m
    );

    void calc_fnu(
        const sim_conditions args,
        const quant_scales qscales,
        const std::vector<double>& rho,
        const std::vector<double>& gb,
        const std::vector<double>& pre,
        const std::vector<std::vector<double>>& mesh,
        const std::vector<double>& tbin_edges,
        std::vector<double>& flux_array,
        const int checkpoint_idx,
        const int data_dim
    );

}   // namespace sogbo_rad

#endif