// =============================================================================
// rad.hpp
//
// low-level synchrotron radiation calculations.
// computes doppler factors, magnetic fields, synchrotron frequencies,
// emissivities, and spectral fluxes for afterglow modeling.
//
// uses compile-time dimensional analysis from units.hpp for type safety.
//
// usage:
//   auto bfield = calc_shock_bfield(energy_density, eps_b);
//   auto nu_g = calc_gyration_frequency(bfield);
//   auto power = calc_total_synch_power(gamma, u_b, beta);
//
// =============================================================================

#ifndef AFTERGLOW_RAD_HPP
#define AFTERGLOW_RAD_HPP

#include "units.hpp"

#include <cstdint>
#include <vector>

namespace simbi::afterglow {

    struct sim_conditions_t
    {
        double              dt, theta_obs, adiabatic_index, current_time, p, z, eps_e, eps_b, d_L;
        std::vector<double> nus;
    };

    struct quant_scales_t
    {
        double time_scale, pre_scale, rho_scale, v_scale, length_scale;
    };

    // doppler boost factor
    double calc_delta_doppler(
        double                     lorentz_factor,
        const std::vector<double>& beta,
        const std::vector<double>& nhat
    );

    // velocity in units of c
    double calc_beta(double gamma_beta);

    // lorentz factor from four-velocity
    double calc_lorentz_factor(double gamma_beta);

    // magnetic field behind shock (gauss)
    double calc_shock_bfield(energy_density_t rho_e, double eps_b);

    // gyration frequency for particle in magnetic field (gauss)
    frequency_t calc_gyration_frequency(double bfield);

    // bolometric synchrotron power per electron
    power_t calc_total_synch_power(double lorentz_factor, energy_density_t ub, double beta);

    // number of photons per energy bin
    double calc_nphotons_per_bin(
        volume_t         volume,
        number_density_t n_e,
        frequency_t      nu_g,
        energy_density_t ub,
        time_t           dt,
        double           gamma_e,
        double           beta,
        double           p
    );

    // number of photons per electron in energy bin
    double calc_nphotons_other(power_t power, frequency_t nu_c, time_t dt);

    // vector magnitude
    double vector_magnitude(const std::vector<double>& a);

    // vector dot product
    double vector_dotproduct(const std::vector<double>& a, const std::vector<double>& b);

    // number density of electrons at given lorentz factor
    number_density_t calc_nelectrons_at_gamma_e(number_density_t n_e, double gamma_e, double p);

    // synchrotron frequency as function of lorentz factor
    frequency_t calc_nu(double gamma_e, frequency_t nu_g);

    // critical lorentz factor as function of time and magnetic field (gauss)
    double calc_critical_lorentz(double bfield, time_t time);

    // maximum power per frequency (sari, piran, narayan 1999, eq. 5)
    energy_t calc_max_power_per_frequency(double bfield);

    // peak emissivity per frequency (gauss)
    spectral_emissivity_t calc_emissivity(double bfield, number_density_t n, double p);

    // minimum lorentz factor of electrons in distribution
    double
    calc_minimum_lorentz(double eps_e, energy_density_t e_thermal, number_density_t n, double p);

    // vector operations
    std::vector<double> vector_multiply(const std::vector<double>& a, const std::vector<double>& b);
    std::vector<double> scale_vector(const std::vector<double>& a, double scalar);
    std::vector<double> vector_subtract(const std::vector<double>& a, const std::vector<double>& b);
    std::vector<double> vector_add(const std::vector<double>& a, const std::vector<double>& b);

    // log photon events for light curve generation
    void log_events(
        sim_conditions_t                  args,
        quant_scales_t                    qscales,
        std::vector<std::vector<double>>& fields,
        std::vector<std::vector<double>>& mesh,
        std::vector<double>&              photon_distribution,
        std::vector<double>&              four_position,
        std::int64_t                      data_dim
    );

    // compute flux with power-law spectrum (sari et al. 1998)
    spectral_power_t calc_powerlaw_flux(
        const spectral_power_t& power_max,
        double                  p,
        frequency_t             nu_prime,
        frequency_t             nu_c,
        frequency_t             nu_m
    );

    // calculate spectral flux density
    void calc_fnu(
        sim_conditions_t                        args,
        quant_scales_t                          qscales,
        const std::vector<double>&              rho,
        const std::vector<double>&              gb,
        const std::vector<double>&              pre,
        const std::vector<std::vector<double>>& mesh,
        const std::vector<double>&              tbin_edges,
        std::vector<double>&                    flux_array,
        std::int64_t                            checkpoint_index,
        std::int64_t                            data_dim
    );

} // namespace simbi::afterglow

#endif // SIMBI_AFTERGLOW_RAD_HPP
