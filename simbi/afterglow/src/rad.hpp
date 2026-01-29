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
#pragma once

#include "units.hpp"

#include <cstdint>
#include <vector>

namespace simbi::afterglow {

    // photon event for monte carlo radiative transfer and post-processing
    // stores complete emission information in lab frame
    struct photon_event_t
    {
        // spacetime coordinates (lab frame)
        double t_emission; // emission time [s]
        double x, y, z;    // emission position [cm]

        // photon 4-momentum
        double energy;     // photon energy [erg]
        double px, py, pz; // propagation direction (unit vector)

        // polarization (stokes parameters in photon frame)
        // SRMHD: computed from B-field geometry
        // SRHD: zero (unpolarized, no B-field info)
        double stokes_I; // intensity (always > 0)
        double stokes_Q; // linear polarization
        double stokes_U; // linear polarization at 45 deg
        double stokes_V; // circular polarization

        // source properties
        double doppler_factor; // Doppler boost factor
        double lorentz_factor; // fluid lorentz factor
        double optical_depth;  // integrated \tau along path (for MCRT)

        // metadata
        std::uint32_t cell_id;   // cell index for debugging
        bool          absorbed;  // true if absorbed/scattered during MCRT
        std::uint32_t n_scatter; // number of scattering events
    };

    // simulation type flag
    enum class hydro_type_t {
        SRHD, // no magnetic field -> unpolarized emission
        SRMHD // magnetic field known -> polarized emission
    };

    struct sim_conditions_t
    {
        double              dt, theta_obs, adiabatic_index, current_time, p, z, eps_e, eps_b, d_L;
        std::vector<double> nus;
        hydro_type_t        hydro_type; // SRHD or SRMHD
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
    magnetic_field_t calc_shock_bfield(energy_density_t rho_e, double eps_b);

    // gyration frequency for particle in magnetic field (gauss)
    frequency_t calc_gyration_frequency(magnetic_field_t bfield);

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
    double calc_critical_lorentz(magnetic_field_t bfield, time_t time);

    // maximum power per frequency (sari, piran, narayan 1999, eq. 5)
    energy_t calc_max_power_per_frequency(magnetic_field_t bfield);

    // peak emissivity per frequency (gauss)
    spectral_emissivity_t calc_emissivity(magnetic_field_t bfield, number_density_t n, double p);

    // minimum lorentz factor of electrons in distribution
    double
    calc_minimum_lorentz(double eps_e, energy_density_t e_thermal, number_density_t n, double p);

    // vector operations
    std::vector<double> vector_multiply(const std::vector<double>& a, const std::vector<double>& b);
    std::vector<double> scale_vector(const std::vector<double>& a, double scalar);
    std::vector<double> vector_subtract(const std::vector<double>& a, const std::vector<double>& b);
    std::vector<double> vector_add(const std::vector<double>& a, const std::vector<double>& b);

    // generate photon events from hydro snapshot with monte carlo sampling
    // returns vector of photon events with full 4-position, 4-momentum, polarization
    //
    // SRHD: polarization set to zero (unpolarized)
    // SRMHD: polarization computed from magnetic field geometry
    //
    // max_events: limit on total photons (for memory management)
    // photons_per_cell: controls sampling density (auto if 0)
    std::vector<photon_event_t> generate_photon_events(
        const sim_conditions_t&                 args,
        const quant_scales_t&                   qscales,
        const std::vector<std::vector<double>>& fields,
        const std::vector<std::vector<double>>& mesh,
        std::int64_t                            data_dim,
        std::uint64_t                           max_events       = 1000000,
        std::uint64_t                           photons_per_cell = 0
    );

    // propagate photons through medium with absorption and scattering
    // modifies photon_event_t.absorbed, .optical_depth, .n_scatter in place
    //
    // processes:
    // - synchrotron self-absorption (\tau_SSA)
    // - thomson scattering (optically thick regions)
    // - pair production \gamma\gamma -> e^+e^- (optional, high energy)
    void monte_carlo_radiative_transfer(
        std::vector<photon_event_t>&            events,
        const sim_conditions_t&                 args,
        const quant_scales_t&                   qscales,
        const std::vector<std::vector<double>>& fields,
        const std::vector<std::vector<double>>& mesh,
        std::int64_t                            data_dim,
        bool                                    include_scattering      = true,
        bool                                    include_pair_production = false
    );

    // compute lightcurve for arbitrary observer direction
    struct observer_lightcurve_t
    {
        std::vector<double> times;       // observer times [day]
        std::vector<double> fluxes;      // flux densities [mJy]
        std::vector<double> frequencies; // observed frequencies [Hz]
    };

    observer_lightcurve_t compute_lightcurve_from_events(
        const std::vector<photon_event_t>& events,
        const std::vector<double>&         observer_direction, // unit vector
        const std::vector<double>&         frequencies,        // Hz
        double                             redshift,
        double                             luminosity_distance, // cm
        const std::vector<double>&         time_bins            // day
    );

    // compute sky map at specific observer time and energy band
    struct skymap_t
    {
        std::vector<std::vector<double>> intensity; // [n_theta][n_phi]
        std::vector<double>              theta;     // polar angles [rad]
        std::vector<double>              phi;       // azimuthal angles [rad]
    };

    skymap_t compute_skymap_from_events(
        const std::vector<photon_event_t>& events,
        const std::vector<double>&         observer_direction,  // unit vector toward observer
        double                             observer_time,       // day
        double                             energy_min,          // erg
        double                             energy_max,          // erg
        double                             redshift,            // cosmological redshift
        double                             luminosity_distance, // cm
        double                             time_window, // day (binning window around observer_time)
        std::uint32_t                      n_theta = 128,
        std::uint32_t                      n_phi   = 256
    );

    // compute polarization evolution for arbitrary observer
    struct polarization_curve_t
    {
        std::vector<double> times;               // observer times [day]
        std::vector<double> polarization_degree; // 0 to 1
        std::vector<double> polarization_angle;  // radians
        std::vector<double> stokes_Q;            // normalized
        std::vector<double> stokes_U;            // normalized
        std::vector<double> stokes_V;            // normalized
    };

    polarization_curve_t compute_polarization_from_events(
        const std::vector<photon_event_t>& events,
        const std::vector<double>&         observer_direction,
        const std::vector<double>&         time_bins,  // day
        double                             energy_min, // erg
        double                             energy_max  // erg
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

    // =============================================================================
    // array-based (numpy-native) compute functions
    // these accept raw pointers to contiguous arrays for zero-copy numpy interop
    // =============================================================================

    // compute lightcurve from columnar event arrays
    // all input arrays must have length n_events
    // output: fluxes array of size (n_time_bins - 1) * n_frequencies
    observer_lightcurve_t compute_lightcurve_from_arrays(
        std::size_t         n_events,
        const double*       t_emission,         // [n_events]
        const double*       x,                  // [n_events]
        const double*       y,                  // [n_events]
        const double*       z,                  // [n_events]
        const double*       energy,             // [n_events]
        const double*       px,                 // [n_events]
        const double*       py,                 // [n_events]
        const double*       pz,                 // [n_events]
        const double*       stokes_I,           // [n_events]
        const std::uint8_t* absorbed,           // [n_events]
        const double*       observer_direction, // [3]
        const double*       frequencies,        // [n_frequencies]
        std::size_t         n_frequencies,
        double              redshift,
        double              luminosity_distance,
        const double*       time_bins, // [n_time_bins]
        std::size_t         n_time_bins
    );

    // compute skymap from columnar event arrays
    skymap_t compute_skymap_from_arrays(
        std::size_t         n_events,
        const double*       t_emission,
        const double*       x,
        const double*       y,
        const double*       z,
        const double*       energy,
        const double*       stokes_I,
        const std::uint8_t* absorbed,
        const double*       observer_direction, // [3] unit vector toward observer
        double              observer_time,
        double              energy_min,
        double              energy_max,
        double              redshift,
        double              luminosity_distance,
        double              time_window,
        std::uint32_t       n_theta,
        std::uint32_t       n_phi
    );

    // compute polarization from columnar event arrays
    polarization_curve_t compute_polarization_from_arrays(
        std::size_t         n_events,
        const double*       t_emission,
        const double*       x,
        const double*       y,
        const double*       z,
        const double*       energy,
        const double*       px,
        const double*       py,
        const double*       pz,
        const double*       stokes_I,
        const double*       stokes_Q,
        const double*       stokes_U,
        const double*       stokes_V,
        const std::uint8_t* absorbed,
        const double*       observer_direction,
        const double*       time_bins,
        std::size_t         n_time_bins,
        double              energy_min,
        double              energy_max
    );

} // namespace simbi::afterglow
