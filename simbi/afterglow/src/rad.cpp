// =============================================================================
// rad.cpp
//
// synchrotron radiation calculations for afterglow modeling.
// computes doppler factors, magnetic fields, frequencies, emissivities,
// and spectral fluxes from relativistic shock physics.
//
// =============================================================================
#include "rad.hpp"

#include "constants.hpp"
#include "units.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iostream>
#include <numbers>
#include <random>
#include <vector>

namespace simbi::afterglow {

    using namespace units;

    double calc_delta_doppler(
        double                     lorentz_factor,
        const std::vector<double>& beta_vec,
        const std::vector<double>& nhat
    )
    {
        return 1.0 / (lorentz_factor * (1.0 - vector_dotproduct(beta_vec, nhat)));
    }

    double calc_beta(double gamma_beta)
    {
        return gamma_beta / std::sqrt(1 + gamma_beta * gamma_beta);
    }

    double calc_lorentz_factor(double gamma_beta)
    {
        return std::sqrt(1.0 + gamma_beta * gamma_beta);
    }
    magnetic_field_t calc_shock_bfield(energy_density_t rho_e, double eps_b)
    {
        auto b_squared = 8.0 * std::numbers::pi * eps_b * rho_e;
        return sqrt(b_squared);
    }

    frequency_t calc_gyration_frequency(magnetic_field_t bfield)
    {
        auto frequency_for_unit_field = (3.0 / 4.0 / std::numbers::pi) * (constants::e_charge) /
                                        (constants::m_e * constants::c_light);
        return frequency_for_unit_field * bfield;
    }

    power_t calc_total_synch_power(double lorentz_factor, energy_density_t ub, double beta)
    {
        return (4.0 / 3.0) * constants::sigma_thomson * constants::c_light * beta * beta *
               lorentz_factor * lorentz_factor * ub;
    }

    double calc_nphotons_per_bin(
        volume_t         volume,
        number_density_t n_e,
        frequency_t      nu_g,
        energy_density_t ub,
        time_t           dt,
        double           gamma_e,
        double           beta,
        double           p
    )
    {
        const auto a = (8.0 * std::numbers::pi * volume / (3.0 * constants::h_planck * nu_g));
        const auto b = constants::sigma_thomson * constants::c_light * beta * beta * ub * n_e;
        const auto c = std::pow(gamma_e, -(p + 1.0));
        return a * b * c * dt;
    }

    double calc_nphotons_other(power_t power, frequency_t nu_c, time_t dt)
    {
        return power * dt / (constants::h_planck * nu_c);
    }

    double gen_random_from_powerlaw(double a, double b, double p, double random_number)
    {
        const double g  = 1 - p;
        const double ag = std::pow(a, g);
        const double bg = std::pow(b, g);
        return std::pow(ag + (bg - ag) * random_number, (1.0 / g));
    }

    double vector_magnitude(const std::vector<double>& a)
    {
        double mag = 0;
        for (const auto val : a) {
            mag += val * val;
        }
        return std::sqrt(mag);
    }

    double vector_dotproduct(const std::vector<double>& a, const std::vector<double>& b)
    {
        double mag = 0;
        for (size_t ii = 0; ii < a.size(); ii++) {
            mag += a[ii] * b[ii];
        }
        return mag;
    }

    number_density_t calc_nelectrons_at_gamma_e(number_density_t n_e, double gamma_e, double p)
    {
        return n_e * std::pow(gamma_e, -p);
    }

    frequency_t calc_nu(double gamma_e, frequency_t nu_g)
    {
        return nu_g * gamma_e * gamma_e;
    }

    double calc_critical_lorentz(magnetic_field_t bfield, time_t time_emitter)
    {
        auto numerator   = 6.0 * std::numbers::pi * constants::m_e * constants::c_light;
        auto denominator = constants::sigma_thomson * bfield * bfield * time_emitter;
        return (numerator / denominator);
    }
    energy_t calc_max_power_per_frequency(magnetic_field_t bfield)
    {
        auto coeff =
            (constants::m_e * constants::c_light * constants::c_light * constants::sigma_thomson) /
            (3.0 * constants::e_charge);
        return coeff * bfield;
    }
    spectral_emissivity_t calc_emissivity(magnetic_field_t bfield, number_density_t n, double p)
    {
        double coeff =
            (9.6323 / 8.0 / std::numbers::pi) * (p - 1.0) / (3.0 * p - 1.0) * std::sqrt(3.0);
        auto e_cubed = pow<3>(constants::e_charge);
        auto denom   = constants::m_e * pow<2>(constants::c_light);
        return coeff * e_cubed / denom * n * bfield;
    }
    double
    calc_minimum_lorentz(double eps_e, energy_density_t e_thermal, number_density_t n, double p)
    {
        return eps_e * (p - 2.0) / (p - 1.0) * e_thermal /
               (n * constants::m_e * constants::c_light * constants::c_light);
    }

    std::vector<double> vector_multiply(const std::vector<double>& a, const std::vector<double>& b)
    {
        std::vector<double> v(a.size());
        std::transform(a.begin(), a.end(), b.begin(), v.begin(), std::multiplies<double>());
        return v;
    }

    std::vector<double> vector_subtract(const std::vector<double>& a, const std::vector<double>& b)
    {
        std::vector<double> v(a.size());
        std::transform(a.begin(), a.end(), b.begin(), v.begin(), std::minus<double>());
        return v;
    }

    std::vector<double> vector_add(const std::vector<double>& a, const std::vector<double>& b)
    {
        std::vector<double> v(a.size());
        std::transform(a.begin(), a.end(), b.begin(), v.begin(), std::plus<double>());
        return v;
    }

    std::vector<double> scale_vector(const std::vector<double>& a, double scalar)
    {
        std::vector<double> v = a;
        std::transform(v.begin(), v.end(), v.begin(), [&scalar](auto& c) { return c * scalar; });
        return v;
    }

    spectral_power_t calc_powerlaw_flux(
        const spectral_power_t& power_max,
        double                  p,
        frequency_t             nu_prime,
        frequency_t             nu_c,
        frequency_t             nu_m
    )
    {
        const bool slow_cool         = nu_c > nu_m;
        auto       power_with_breaks = power_max;
        if (slow_cool) {
            const bool slow_break1 = nu_prime < nu_m;
            const bool slow_break2 = (nu_prime < nu_c) && (nu_prime > nu_m);
            if (slow_break1) {
                power_with_breaks *= std::pow(nu_prime / nu_m, (1.0 / 3.0));
            }
            else if (slow_break2) {
                power_with_breaks *= std::pow(nu_prime / nu_m, -0.5 * (p - 1.0));
            }
            else {
                power_with_breaks *=
                    std::pow(nu_c / nu_m, -0.5 * (p - 1.0)) * std::pow(nu_prime / nu_c, -0.5 * p);
            }
        }
        else {
            const bool fast_break1 = nu_prime < nu_c;
            const bool fast_break2 = (nu_prime < nu_m) && (nu_prime > nu_c);
            if (fast_break1) {
                power_with_breaks *= std::pow(nu_prime / nu_c, (1.0 / 3.0));
            }
            else if (fast_break2) {
                power_with_breaks *= std::pow(nu_prime / nu_c, -0.5);
            }
            else {
                power_with_breaks *=
                    std::pow(nu_m / nu_c, -0.5) * std::pow(nu_prime / nu_m, -0.5 * p);
            }
        }
        return power_with_breaks;
    }

    // =============================================================================
    // monte carlo radiative transfer
    // =============================================================================

    // helper: compute synchrotron self-absorption optical depth
    double compute_ssa_optical_depth(
        energy_t         photon_energy, // erg
        number_density_t n_e,           // electron density [cm^-3]
        magnetic_field_t bfield,        // magnetic field [gauss]
        double           path_length,   // cm
        double           p              // spectral index
    )
    {
        // synchrotron self-absorption coefficient
        // \alpha_\nu \propto n_e B (\nu_g/\nu)^{(p+4)/2}

        auto nu_photon = photon_energy / constants::h_planck;
        auto nu_gyro   = constants::e_charge * bfield /
                       (2.0 * std::numbers::pi * constants::m_e * constants::c_light);
        double nu_ratio = (nu_gyro / nu_photon).value;

        if (nu_ratio < 1.0) {
            return 0.0; // above synchrotron peak, SSA negligible
        }

        // ssa coefficient (simplified, cgs units: cm^-1)
        // prefactor calibrated for typical grb afterglow conditions
        auto   alpha_ssa_typed = 3.3e-10 * n_e * bfield * std::pow(nu_ratio, (p + 4.0) / 2.0);
        double alpha_ssa       = alpha_ssa_typed.value;

        return alpha_ssa * path_length;
    }

    // helper: compute thomson scattering optical depth
    double compute_thomson_optical_depth(number_density_t n_e, double path_length)
    {
        // \tau_T = n_e \sigma_T L
        return (n_e * constants::sigma_thomson * path_length).value;
    }

    // helper: scatter photon (thomson scattering)
    void
    scatter_photon(photon_event_t& photon, std::mt19937& gen, std::uniform_real_distribution<>& dis)
    {
        // isotropic scattering in lab frame (simplified)
        double phi       = 2.0 * std::numbers::pi * dis(gen);
        double mu        = 2.0 * dis(gen) - 1.0;
        double sin_theta = std::sqrt(1.0 - mu * mu);

        photon.px = sin_theta * std::cos(phi);
        photon.py = sin_theta * std::sin(phi);
        photon.pz = mu;

        // depolarize: each scatter reduces polarization
        double depol_factor = std::exp(-1.0);
        photon.stokes_Q *= depol_factor;
        photon.stokes_U *= depol_factor;
        photon.stokes_V *= depol_factor;

        photon.n_scatter++;
    }

    void monte_carlo_radiative_transfer(
        std::vector<photon_event_t>&            events,
        const sim_conditions_t&                 args,
        const quant_scales_t&                   qscales,
        const std::vector<std::vector<double>>& fields,
        const std::vector<std::vector<double>>& mesh,
        std::int64_t                            data_dim [[maybe_unused]],
        bool                                    include_scattering,
        bool                                    include_pair_production
    )
    {
        std::random_device               rd;
        std::mt19937                     gen(rd());
        std::uniform_real_distribution<> dis(0.0, 1.0);

        const auto rho = fields[0];
        const auto gb  = fields[1];
        const auto pre = fields[2];

        const auto x1 = mesh[0];

        const double p     = args.p;
        const double eps_b = args.eps_b;

// process each photon in parallel
#pragma omp parallel for
        for (std::size_t i = 0; i < events.size(); i++) {
            auto& photon = events[i];

            // get source cell properties
            std::uint32_t cell_id = photon.cell_id;

            // compute physical properties at emission cell
            const auto rho_einternal = pre[cell_id] * qscales.pre_scale /
                                       (args.adiabatic_index - 1.0) * units::erg_per_cm3;
            const auto bfield = calc_shock_bfield(rho_einternal, eps_b);
            const auto n_e = rho[cell_id] * qscales.rho_scale * units::g_per_cm3 / constants::m_p;

            // estimate path length through cell (approximate as ~10% of radius)
            double path_length = x1[0] * qscales.length_scale * 0.1;

            // compute optical depths
            double tau_ssa =
                compute_ssa_optical_depth(energy_t{photon.energy}, n_e, bfield, path_length, p);

            double tau_thomson = compute_thomson_optical_depth(n_e, path_length);
            double tau_total   = tau_ssa + tau_thomson;

            photon.optical_depth = tau_total;

            // monte carlo absorption test
            if (dis(gen) > std::exp(-tau_total)) {
                photon.absorbed = true;

                // check if scattered rather than absorbed
                if (include_scattering && tau_thomson > 0.0) {
                    double scatter_probability = tau_thomson / tau_total;
                    if (dis(gen) < scatter_probability) {
                        scatter_photon(photon, gen, dis);
                        photon.absorbed = false; // photon survives after scatter
                    }
                }
            }

            // pair production (optional, high energy only)
            if (include_pair_production) {
                // \gamma\gamma -> e^+e^- threshold: E > m_e c^2 ~ 0.5 MeV ~ 8e-7 erg
                double threshold_energy = 8e-7; // erg (~0.5 MeV)
                if (photon.energy > threshold_energy) {
                    // simplified: mark as absorbed if above threshold
                    // proper treatment would compute \gamma\gamma opacity
                    photon.absorbed = true;
                }
            }
        }
    }

    // =============================================================================
    // legacy interface
    // =============================================================================

    void log_events(
        sim_conditions_t                  args,
        quant_scales_t                    qscales,
        std::vector<std::vector<double>>& fields,
        std::vector<std::vector<double>>& mesh,
        std::vector<double>&              photon_distribution,
        std::vector<double>&              four_position,
        std::int64_t                      data_dim
    )
    {
        std::random_device               rd;
        std::mt19937                     gen(rd());
        std::uniform_real_distribution<> dis(0.0, 1.0);

        std::uint64_t ng  = 100;
        const auto    rho = fields[0]; // fluid frame density
        const auto    gb  = fields[1]; // four-velocity
        const auto    pre = fields[2]; // pressure

        // extract the geometry of the mesh
        const auto   x1     = mesh[0];
        const auto   x2     = mesh[1];
        const auto   x3     = mesh[2];
        const auto   ni     = x1.size();
        const auto   nj     = x2.size();
        const auto   nk     = x3.size();
        const auto   x1max  = x1[ni - 1];
        const auto   x1min  = x1[0];
        const auto   dlogx1 = std::log10(x1max / x1min) / (ni - 1);
        const auto   x2max  = x2[nj - 1];
        const auto   x2min  = x2[0];
        const auto   dx2    = (x2max - x2min) / (nj - 1);
        const auto   x3max  = x3[nk - 1];
        const auto   x3min  = x3[0];
        const auto   dx3    = (x3max - x2min) / (nk - 1);
        const double p      = args.p;     // electron number index
        const double eps_b  = args.eps_b; // magnetic field fraction of internal energy
        const double eps_e  = args.eps_e; // shocked electrons fraction of internal energy

        const auto t_prime = args.current_time * qscales.time_scale * units::s;
        const auto dt      = args.dt * qscales.time_scale * units::s;
        for (std::size_t kk = 0; kk < x3.size(); kk++) {
            const double x3l     = (kk > 0) ? x2min + (kk - 0.5) * dx3 : x3min;
            const double x3r     = (kk < nk - 1) ? x3l + dx3 * (kk == 0 ? 0.5 : 1.0) : x3max;
            const double sin_phi = std::sin(x3[kk]);
            const double cos_phi = std::cos(x3[kk]);
            const double dx3     = x3r - x3l;

            // if the data is 3d, then there is a real k-space to pull data from
            const std::int64_t kreal = (data_dim > 2) * kk;
#pragma omp parallel
            for (std::size_t jj = 0; jj < x2.size(); jj++) {
                const double x2l  = (jj > 0) ? x2min + (jj - 0.5) * dx2 : x2min;
                const double x2r  = (jj < nj - 1) ? x2l + dx2 * (jj == 0 ? 0.5 : 1.0) : x2max;
                const double dcos = std::cos(x2l) - std::cos(x2r);

                // radial unit vector
                const std::vector<double> rhat =
                    {std::sin(x2[jj]) * cos_phi, std::sin(x2[jj]) * sin_phi, std::cos(x2[jj])};
                const std::int64_t jreal = (data_dim > 1) * jj;
#pragma omp for nowait
                for (std::size_t ii = 0; ii < x1.size(); ii++) {
                    const auto central_idx =
                        kreal * ni * nj + jreal * ni + ii;             // index for current zone
                    const auto beta      = calc_beta(gb[central_idx]); // velocity in units of c
                    const auto w         = calc_lorentz_factor(gb[central_idx]); // lorentz factor
                    const auto t_emitter = t_prime / w; // time in emitter frame

                    const double              phi_prime  = 2.0 * std::numbers::pi * dis(gen);
                    const double              mu_prime   = 2.0 * dis(gen) - 1.0;
                    const std::vector<double> nhat_prime = {
                        std::sin(std::acos(mu_prime)) * std::cos(phi_prime),
                        std::sin(std::acos(mu_prime)) * std::sin(phi_prime),
                        mu_prime
                    };

                    // cosine of the isotropic emission angle wrt to the
                    // propagation direction
                    const double mu_rhat_prime = vector_dotproduct(rhat, nhat_prime);
                    // cos of the resulting beamed angle in the plane of rhat
                    // and nhat prime
                    const double mu_rhat_beam =
                        (mu_rhat_prime + beta) / (1.0 + beta * mu_rhat_prime);
                    const double rot_angle =
                        std::acos(mu_rhat_prime) -
                        std::acos(mu_rhat_beam); // rotation angle from initial emission direction
                                                 // to beaming direction
                    const auto nhat_beamed = scale_vector(nhat_prime, std::cos(rot_angle));
                    const auto nvec_lab    = scale_vector(nhat_beamed, x1[ii]);
                    const std::vector<double> x_mu =
                        {t_prime.value, nvec_lab[0], nvec_lab[1], nvec_lab[2]};

                    // hydro conditions
                    const auto rho_einternal = pre[central_idx] * qscales.pre_scale /
                                               (args.adiabatic_index - 1.0) * units::erg_per_cm3;
                    const auto bfield = calc_shock_bfield(
                        rho_einternal,
                        eps_b
                    ); // magnetic field based on equipartition
                    const auto n_e_proper =
                        rho[central_idx] * qscales.rho_scale * units::g_per_cm3 / constants::m_p;
                    const auto nu_g      = calc_gyration_frequency(bfield);
                    const auto gamma_min = calc_minimum_lorentz(
                        eps_e,
                        rho_einternal,
                        n_e_proper,
                        p
                    ); // minimum lorentz factor of electrons
                    const auto gamma_crit = calc_critical_lorentz(bfield, t_emitter);

                    const auto gamma_max = std::max(gamma_min, gamma_crit);
                    const auto gamma_low = std::min(gamma_min, gamma_crit);
                    const auto dg        = (gamma_max - gamma_low) / (ng - 1);

                    // calc cell volumes
                    const double x1l =
                        (ii > 0) ? x1min * std::pow(10.0, (ii - 0.5) * dlogx1) : x1min;
                    const double x1r     = (ii < ni - 1)
                                               ? x1l * std::pow(10.0, dlogx1 * (ii == 0 ? 0.5 : 1.0))
                                               : x1max;
                    const auto   dvolume = dx3 * dcos * (1.0 / 3.0) *
                                         (x1r * x1r * x1r - x1l * x1l * x1l) *
                                         qscales.length_scale * qscales.length_scale *
                                         qscales.length_scale * units::cm3;

                    // each cell will have its own photons distribution.
                    // to account for this, we divide the gamma bins up
                    // and bin the photons in each cell with respect to the
                    // gamma bin
                    const auto n_e = n_e_proper * w;
                    const auto ub  = bfield * bfield / 8.0 / std::numbers::pi;
                    for (std::uint64_t qq = 0; qq < ng; qq++) {
                        const auto gamma_e = gamma_min + qq * dg;
                        const auto gamma_sample =
                            gen_random_from_powerlaw(gamma_e, gamma_e + dg, p, dis(gen));
                        const auto nu_c = calc_nu(gamma_sample, nu_g);
                        const auto nphot =
                            calc_nphotons_per_bin(dvolume, n_e, nu_g, ub, dt, gamma_e, beta, p) *
                            dg;
                        photon_distribution[kk * ni * nj * ng + jj * ni * ng + ii * ng + qq] =
                            (constants::h_planck * nu_c).value * nphot;
                    }

                    // log the four-position
                    for (std::uint64_t qq = 0; qq < 4; qq++) {
                        four_position[kk * ni * nj * 4 + jj * ni * 4 + ii * 4 + qq] = x_mu[qq];
                    }
                }
            }
        }
    }

    /**
     * Compute the spectral flux due to synchrotron emission
     *
     * @param args       a struct containing the simulation conditions
     * @param qscales    a struct containing the relevant dimensionful scales of
     * the problem
     * @param fields     a 2D array of the primitives vairables rho, gamma_beta,
     * and pressure
     * @param mesh       a 2D array of dimensionles values for the mesh
     * centroids
     * @param tbin_edges a 1D array of the time bin edges for the flux
     * calculations
     * @param fbin_edges a 1D array of the frequency_t bin edges for the flux
     * calculartions
     * @param flux_array a flattened 1D array in which the summed frequencies in
     * each bin will live
     * @param checkpoint_index  the integer index of the checkpoint file
     *
     */
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
    )
    {
        // place observer along chosen axis
        const std::vector<double> obs_hat =
            {std::sin(args.theta_obs), 0.0, std::cos(args.theta_obs)};

        const auto nt = tbin_edges.size(); // time bin size
        const auto nf = args.nus.size();   // frequency bin size

        // extract the geometry of the mesh
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
        const bool at_pole = std::abs(std::cos(args.theta_obs)) == 1;

        size_t nk      = 1;
        double sin_phi = 0;
        double cos_phi = 1.0;
        double dx3     = 2.0 * std::numbers::pi;
        // check whether to do 3d (off-axis) or not
        std::vector<double> x3;
        double              x3max = 0.0;
        double              x3min = 0.0;
        if (!at_pole) {
            x3    = mesh[2];
            nk    = x3.size();
            x3max = x3[nk - 1];
            x3min = x3[0];
            dx3   = (x3max - x3min) / (nk - 1);
        }

        const double p     = args.p;
        const double eps_b = args.eps_b;
        const double eps_e = args.eps_e;
        const auto   d     = args.d_L * units::cm; // luminosity distance

        const auto t_prime    = args.current_time * qscales.time_scale * units::s;
        const auto dt         = args.dt * qscales.time_scale * units::s;
        const auto flux_denom = 1.0 / (4.0 * std::numbers::pi * d * d);
        for (std::size_t kk = 0; kk < nk; kk++) {
            if (!at_pole) {
                const double x3l = (kk > 0) ? x2min + (kk - 0.5) * dx3 : x3min;
                const double x3r = (kk < nk - 1) ? x3l + dx3 * (kk == 0 ? 0.5 : 1.0) : x3max;
                sin_phi          = std::sin(x3[kk]);
                cos_phi          = std::cos(x3[kk]);
                dx3              = x3r - x3l;
            }

            // if the data is 3d, then there is a real k-space to pull data from
            const std::int64_t kreal = (data_dim > 2) * kk;
#pragma omp parallel
            for (std::uint64_t jj = 0; jj < nj; jj++) {
                const double x2l  = (jj > 0) ? x2min + (jj - 0.5) * dx2 : x2min;
                const double x2r  = (jj < nj - 1) ? x2l + dx2 * (jj == 0 ? 0.5 : 1.0) : x2max;
                const double dcos = std::cos(x2l) - std::cos(x2r);

                // radial unit vector
                const std::vector<double> rhat =
                    {std::sin(x2[jj]) * cos_phi, std::sin(x2[jj]) * sin_phi, std::cos(x2[jj])};

                // data greater than 1d? there is a j space to pull data from
                const std::uint64_t jreal = (data_dim > 1) * jj;
#pragma omp for nowait
                for (std::uint64_t ii = 0; ii < ni; ii++) {
                    const auto central_idx   = kreal * ni * nj + jreal * ni + ii;
                    const auto beta          = calc_beta(gb[central_idx]);
                    const auto w             = calc_lorentz_factor(gb[central_idx]);
                    const auto t_emitter     = t_prime / w;
                    const auto rho_einternal = pre[central_idx] * qscales.pre_scale /
                                               (args.adiabatic_index - 1.0) * units::erg_per_cm3;
                    const auto bfield = calc_shock_bfield(
                        rho_einternal,
                        eps_b
                    ); // magnetic field based on equipartition
                    const auto n_e_proper =
                        rho[central_idx] * qscales.rho_scale * units::g_per_cm3 / constants::m_p;
                    const auto nu_g = calc_gyration_frequency(
                        bfield
                    ); // gyration frequency_t // distance to source
                    const auto gamma_min = calc_minimum_lorentz(
                        eps_e,
                        rho_einternal,
                        n_e_proper,
                        p
                    ); // minimum lorentz factor of electrons
                    const auto gamma_crit = calc_critical_lorentz(
                        bfield,
                        t_emitter
                    ); // critical lorentz factor of electrons

                    // calc cell volumes
                    const double x1l =
                        (ii > 0) ? x1min * std::pow(10.0, (ii - 0.5) * dlogx1) : x1min;
                    const double x1r     = (ii < ni - 1)
                                               ? x1l * std::pow(10.0, dlogx1 * (ii == 0 ? 0.5 : 1.0))
                                               : x1max;
                    const auto   dvolume = dx3 * dcos * (1.0 / 3.0) *
                                         (x1r * x1r * x1r - x1l * x1l * x1l) *
                                         qscales.length_scale * qscales.length_scale *
                                         qscales.length_scale * units::cm3;

                    // observer time
                    const auto t_obs = t_prime - x1[ii] * qscales.length_scale *
                                                     vector_dotproduct(rhat, obs_hat) * units::cm /
                                                     constants::c_light;

                    const std::vector<double> beta_vec =
                        {beta * rhat[0], beta * rhat[1], beta * rhat[2]};

                    // calculate the maximum flux based on the average
                    // bolometric power per electron
                    const frequency_t nu_c = calc_nu(gamma_crit, nu_g); // critical frequency
                    const frequency_t nu_m = calc_nu(gamma_min, nu_g);  // minimum frequency
                    const double      delta_doppler = calc_delta_doppler(
                        w,
                        beta_vec,
                        obs_hat
                    ); // doppler factor
                    const spectral_emissivity_t eps_m = calc_emissivity(
                        bfield,
                        n_e_proper,
                        p
                    ); // emissivity per cell

                    // total emitted power per unit frequency in each cell volume
                    const spectral_power_t power_prime =
                        dvolume * eps_m * delta_doppler * delta_doppler;
                    const double t_obs_day = (t_obs / day);
                    // loop through the given frequencies and put them in their
                    // respective locations in dictionary
                    for (size_t fidx = 0; fidx < nf; fidx++) {
                        // the frequency we see is doppler boosted, so account for that
                        const frequency_t      nu_source = args.nus[fidx] * hz / delta_doppler;
                        const spectral_power_t power_cool =
                            calc_powerlaw_flux(power_prime, p, nu_source, nu_c, nu_m);
                        const spectral_flux_t f_nu = (power_cool * flux_denom);

                        // place the fluxes in the appropriate time bins
                        for (size_t tidx = 0; tidx < nt - 1; tidx++) {
                            const double t1 = tbin_edges[tidx + 0];
                            const double t2 = tbin_edges[tidx + 1];
                            if (t1 < t_obs_day && t_obs_day < t2) {
                                // the effective lifetime of the emitting cell
                                // must be accounted for
                                const auto   dt_day = (dt / day);
                                const auto   dt_obs = t2 - t1;
                                const double trat =
                                    (checkpoint_index > 0) ? dt_day.value / dt_obs : 1.0;
                                // sum the fluxes in the given time bin
                                flux_array[fidx * (nt - 1) + tidx] += trat * f_nu.value;
                                break;
                            }
                        } // end time bin loop
                    }
                } // end inner parallel loop

            } // end outer parallel loop
        }

        // return flux_array;
    }

    std::vector<photon_event_t> generate_photon_events(
        const sim_conditions_t&                 args,
        const quant_scales_t&                   qscales,
        const std::vector<std::vector<double>>& fields,
        const std::vector<std::vector<double>>& mesh,
        std::int64_t                            data_dim,
        std::uint64_t                           max_events,
        std::uint64_t                           photons_per_cell
    )
    {
        std::vector<photon_event_t> events;
        events.reserve(std::min(max_events, std::uint64_t(10000)));

        std::random_device               rd;
        std::mt19937                     gen(rd());
        std::uniform_real_distribution<> dis(0.0, 1.0);

        const auto rho = fields[0];
        const auto gb  = fields[1];
        const auto pre = fields[2];

        const auto x1 = (mesh.size() > 0) ? mesh[0] : std::vector<double>{};
        const auto x2 = (mesh.size() > 1) ? mesh[1] : std::vector<double>{};
        const auto x3 = (mesh.size() > 2) ? mesh[2] : std::vector<double>{0.0};
        const auto ni = x1.size();
        const auto nj = x2.size();
        const auto nk = (x3.size() > 0) ? x3.size() : 1;

        const double p     = args.p;
        const double eps_b = args.eps_b;
        const double eps_e = args.eps_e;

        const auto t_prime = args.current_time * qscales.time_scale * units::s;

        const std::uint64_t total_cells = ni * nj * nk;
        const std::uint64_t photons_target =
            (photons_per_cell > 0) ? photons_per_cell
                                   : std::max(std::uint64_t(10), max_events / total_cells);

        std::uint64_t events_generated = 0;

        // loop order: radius outermost, then theta, then phi
        // this ensures we sample all angles at each radius before moving outward
        for (std::size_t ii = 0; ii < ni; ii++) {
            if (events_generated >= max_events) {
                goto done;
            }

            const auto r_center = x1[ii] * qscales.length_scale * units::cm;

            // cell radial bounds (log-spaced grid)
            const double x1l =
                (ii > 0) ? x1[0] * std::pow(10.0, (ii - 0.5) * std::log10(x1[1] / x1[0])) : x1[0];
            const double x1r =
                (ii < ni - 1) ? x1l * std::pow(10.0, std::log10(x1[1] / x1[0])) : x1[ni - 1];

            for (std::size_t jj = 0; jj < nj; jj++) {
                const std::int64_t jreal = (data_dim > 1) * jj;
                const double       dx2   = (nj > 1) ? (x2[1] - x2[0]) : 2.0 * std::numbers::pi;
                const double       dcos =
                    (nj > 1) ? std::abs(std::cos(x2[jj]) - std::cos(x2[jj] + dx2)) : 2.0;

                for (std::size_t kk = 0; kk < nk; kk++) {
                    if (events_generated >= max_events) {
                        goto done;
                    }

                    const double       sin_phi = std::sin(x3[kk]);
                    const double       cos_phi = std::cos(x3[kk]);
                    const std::int64_t kreal   = (data_dim > 2) * kk;

                    const std::vector<double> rhat =
                        {std::sin(x2[jj]) * cos_phi, std::sin(x2[jj]) * sin_phi, std::cos(x2[jj])};

                    // numpy uses C-order (row-major): last index varies fastest
                    // for shape (ni, nj, nk), flat index = ii * nj * nk + jj * nk + kk
                    const auto central_idx = ii * nj * nk + jreal * nk + kreal;
                    const auto beta        = calc_beta(gb[central_idx]);
                    const auto w           = calc_lorentz_factor(gb[central_idx]);

                    const auto rho_einternal = pre[central_idx] * qscales.pre_scale /
                                               (args.adiabatic_index - 1.0) * units::erg_per_cm3;
                    const auto bfield = calc_shock_bfield(rho_einternal, eps_b);
                    const auto n_e_proper =
                        rho[central_idx] * qscales.rho_scale * units::g_per_cm3 / constants::m_p;
                    const auto gamma_min =
                        calc_minimum_lorentz(eps_e, rho_einternal, n_e_proper, p);

                    // cell volume for packet weighting
                    const double dx3 = (nk > 1) ? (x3[1] - x3[0]) : 2.0 * std::numbers::pi;

                    // volume in cm^3 (bare double from product of lengths)
                    const double dvolume_cgs =
                        dx3 * dcos * (1.0 / 3.0) * (x1r * x1r * x1r - x1l * x1l * x1l) *
                        qscales.length_scale * qscales.length_scale * qscales.length_scale;

                    // calculate total synchrotron power from this cell
                    // p_sync = (4/3) * sigma_T * c * beta^2 * u_B * gamma_e^2 * n_e * V
                    const auto u_B     = bfield * bfield / (8.0 * std::numbers::pi);
                    const auto dt      = args.dt * qscales.time_scale * units::s;
                    const auto dvolume = dvolume_cgs * units::cm3;

                    // for power-law distribution N(\gamma) \propto \gamma^{-p} between \gamma_min
                    // and \gamma_max:
                    // <\gamma^2> = \int \gamma^2 \gamma^{-p} d\gamma / \int \gamma^{-p} d\gamma
                    // for p > 3 and \gamma_max >> \gamma_min: <\gamma^2> ~ (p-1)/(p-3) *
                    // \gamma_min^2 for 2 < p < 3: integral diverges at high \gamma, use
                    // \gamma_min^2 as lower bound
                    const double power_law_factor = (p > 3.0) ? (p - 1.0) / (p - 3.0) : 1.0;
                    const double gamma_e_sq_avg   = gamma_min * gamma_min * power_law_factor;

                    // total radiated energy from cell [erg]
                    const auto total_energy_cell = (4.0 / 3.0) * constants::sigma_thomson *
                                                   constants::c_light * beta * beta * u_B *
                                                   n_e_proper * dvolume * dt * gamma_e_sq_avg;

                    // packet weight: total energy divided by number of packets per cell
                    const auto packet_weight = total_energy_cell / photons_target;

                    for (std::uint64_t pp = 0; pp < photons_target; pp++) {
                        if (events_generated >= max_events) {
                            goto done;
                        }

                        photon_event_t evt;

                        const double              phi_prime  = 2.0 * std::numbers::pi * dis(gen);
                        const double              mu_prime   = 2.0 * dis(gen) - 1.0;
                        const std::vector<double> nhat_prime = {
                            std::sin(std::acos(mu_prime)) * std::cos(phi_prime),
                            std::sin(std::acos(mu_prime)) * std::sin(phi_prime),
                            mu_prime
                        };

                        const double mu_rhat_prime = vector_dotproduct(rhat, nhat_prime);
                        const double mu_rhat_beam =
                            (mu_rhat_prime + beta) / (1.0 + beta * mu_rhat_prime);
                        const double rot_angle = std::acos(mu_rhat_prime) - std::acos(mu_rhat_beam);
                        const auto   nhat_beamed = scale_vector(nhat_prime, std::cos(rot_angle));

                        evt.t_emission = t_prime.value;
                        evt.x          = r_center.value * rhat[0];
                        evt.y          = r_center.value * rhat[1];
                        evt.z          = r_center.value * rhat[2];

                        evt.energy = packet_weight.value;
                        evt.px     = nhat_beamed[0];
                        evt.py     = nhat_beamed[1];
                        evt.pz     = nhat_beamed[2];

                        evt.stokes_I = 1.0;
                        evt.stokes_Q = 0.0;
                        evt.stokes_U = 0.0;
                        evt.stokes_V = 0.0;

                        evt.doppler_factor = calc_delta_doppler(
                            w,
                            {beta * rhat[0], beta * rhat[1], beta * rhat[2]},
                            nhat_beamed
                        );
                        evt.lorentz_factor = w;
                        evt.optical_depth  = 0.0;
                        evt.cell_id        = static_cast<std::uint32_t>(central_idx);
                        evt.absorbed       = false;
                        evt.n_scatter      = 0;

                        events.push_back(evt);
                        events_generated++;
                    }
                }
            }
        }

    done:
        return events;
    }

    // compute lightcurve from photon events
    observer_lightcurve_t compute_lightcurve_from_events(
        const std::vector<photon_event_t>& events,
        const std::vector<double>&         observer_direction,
        const std::vector<double>&         frequencies,
        double                             redshift,
        double                             luminosity_distance,
        const std::vector<double>&         time_bins
    )
    {
        observer_lightcurve_t result;
        result.times       = time_bins;
        result.frequencies = frequencies;

        const std::size_t n_times = time_bins.size();
        const std::size_t n_freqs = frequencies.size();
        result.fluxes.resize(n_times * n_freqs, 0.0);

        // convert time bins from days to seconds
        std::vector<double> time_bins_s(n_times);
        for (std::size_t i = 0; i < n_times; ++i) {
            time_bins_s[i] = time_bins[i] * 86400.0;
        }

        // normalize observer direction
        const double obs_mag = std::sqrt(
            observer_direction[0] * observer_direction[0] +
            observer_direction[1] * observer_direction[1] +
            observer_direction[2] * observer_direction[2]
        );
        const std::vector<double> obs_hat = {
            observer_direction[0] / obs_mag,
            observer_direction[1] / obs_mag,
            observer_direction[2] / obs_mag
        };

        // luminosity distance in cm
        const double d_L    = luminosity_distance;
        const double d_L_sq = d_L * d_L;

        // bin photon events by arrival time and frequency
        for (const auto& evt : events) {
            // skip absorbed photons
            if (evt.absorbed) {
                continue;
            }

            // check if photon is propagating toward observer
            // photon direction (px, py, pz) should align with observer direction
            const double cos_angle =
                evt.px * obs_hat[0] + evt.py * obs_hat[1] + evt.pz * obs_hat[2];
            if (cos_angle < 0.5) { // viewing angle > 60 degrees, likely not visible
                continue;
            }

            // compute observer arrival time: t_obs = t_emission + r/c
            const double r_emission = std::sqrt(evt.x * evt.x + evt.y * evt.y + evt.z * evt.z);
            const double t_arrival  = evt.t_emission + r_emission / constants::c_light.value;

            // find time bin
            std::size_t t_bin = n_times;
            for (std::size_t i = 0; i < n_times - 1; ++i) {
                if (t_arrival >= time_bins_s[i] && t_arrival < time_bins_s[i + 1]) {
                    t_bin = i;
                    break;
                }
            }
            if (t_bin >= n_times - 1) {
                continue; // outside time range
            }

            // photon frequency (assume monochromatic at packet representative frequency)
            // use photon energy: E = h*nu -> nu = E/h
            const double nu_photon = evt.energy / constants::h_planck.value;

            // apply redshift: observed frequency = emitted frequency / (1+z)
            const double nu_obs = nu_photon / (1.0 + redshift);

            // find frequency bin
            std::size_t f_bin = n_freqs;
            for (std::size_t j = 0; j < n_freqs - 1; ++j) {
                if (nu_obs >= frequencies[j] && nu_obs < frequencies[j + 1]) {
                    f_bin = j;
                    break;
                }
            }
            if (f_bin >= n_freqs - 1) {
                continue; // outside frequency range
            }

            // compute flux contribution: F_nu = L_nu / (4\pi d_L^2)
            // packet energy is total energy, distribute over frequency bin width
            const double dnu          = frequencies[f_bin + 1] - frequencies[f_bin];
            const double dt           = time_bins_s[t_bin + 1] - time_bins_s[t_bin];
            const double flux_contrib = evt.energy / (4.0 * std::numbers::pi * d_L_sq * dnu * dt);

            // accumulate flux (weighted by stokes I for intensity)
            result.fluxes[t_bin * n_freqs + f_bin] += flux_contrib * evt.stokes_I;
        }

        return result;
    }

    // compute skymap from photon events
    //
    // the skymap shows intensity as function of angular position on the sky.
    // for a spherically symmetric blast wave viewed on-axis, the image is a ring
    // due to the equal arrival time surface (EATS) geometry.
    //
    // coordinate system:
    //   - observer_direction n points FROM source TO observer (unit vector)
    //   - sky plane is perpendicular to n
    //   - theta_sky = angular distance from center (line of sight)
    //   - phi_sky = azimuthal angle in sky plane
    //
    // arrival time: t_obs = (1 + z) * (t_em - r dot n / c)
    //   photons from far side (positive r dot n) arrive later
    //   redshift dilates the observed time
    skymap_t compute_skymap_from_events(
        const std::vector<photon_event_t>& events,
        const std::vector<double>&         observer_direction,
        double                             observer_time, // day
        double                             energy_min,    // erg
        double                             energy_max,    // erg
        double                             redshift,
        double                             luminosity_distance, // cm
        double                             time_window,         // day
        std::uint32_t                      n_theta,
        std::uint32_t                      n_phi
    )
    {
        const double one_plus_z = 1.0 + redshift;
        skymap_t     result;

        // observer direction (unit vector toward observer)
        const double nx = observer_direction[0];
        const double ny = observer_direction[1];
        const double nz = observer_direction[2];

        // build orthonormal basis for sky plane
        // e1, e2 span the plane perpendicular to n
        // choose e1 perpendicular to n and to z-axis (unless n is along z)
        double e1x, e1y, e1z;
        if (std::abs(nz) < 0.99) {
            // n is not along z, cross with z
            e1x = -ny;
            e1y = nx;
            e1z = 0.0;
        }
        else {
            // n is along z, cross with x
            e1x = 0.0;
            e1y = -nz;
            e1z = ny;
        }
        double e1_norm = std::sqrt(e1x * e1x + e1y * e1y + e1z * e1z);
        e1x /= e1_norm;
        e1y /= e1_norm;
        e1z /= e1_norm;

        // e2 = n cross e1
        double e2x = ny * e1z - nz * e1y;
        double e2y = nz * e1x - nx * e1z;
        double e2z = nx * e1y - ny * e1x;

        // determine angular scale from event positions projected onto sky
        double max_sky_radius = 0.0;
        for (const auto& evt : events) {
            // project position onto sky plane
            double proj1   = evt.x * e1x + evt.y * e1y + evt.z * e1z;
            double proj2   = evt.x * e2x + evt.y * e2y + evt.z * e2z;
            double sky_r   = std::sqrt(proj1 * proj1 + proj2 * proj2);
            max_sky_radius = std::max(max_sky_radius, sky_r);
        }

        // angular size on sky: theta = R_sky / d_L
        const double max_angle_rad = (max_sky_radius / luminosity_distance) * 1.5; // 1.5x padding

        // create angular grid
        result.theta.resize(n_theta);
        result.phi.resize(n_phi);
        result.intensity.resize(n_theta, std::vector<double>(n_phi, 0.0));

        for (std::uint32_t ii = 0; ii < n_theta; ++ii) {
            result.theta[ii] = max_angle_rad * ii / (n_theta - 1.0);
        }
        for (std::uint32_t jj = 0; jj < n_phi; ++jj) {
            result.phi[jj] = 2.0 * std::numbers::pi * jj / n_phi;
        }

        // convert observer time from days to seconds
        const double t_obs_s       = observer_time * 86400.0;
        const double time_window_s = time_window * 86400.0;
        const double t_obs_min     = t_obs_s - 0.5 * time_window_s;
        const double t_obs_max     = t_obs_s + 0.5 * time_window_s;

        // bin photon events onto angular grid
        for (const auto& evt : events) {
            // skip absorbed photons
            if (evt.absorbed) {
                continue;
            }

            // filter by energy
            if (evt.energy < energy_min || evt.energy > energy_max) {
                continue;
            }

            // compute observer arrival time: t_obs = (1+z) * (t_em - r dot n / c)
            // r dot n = projection of emission position onto line of sight
            const double r_dot_n = evt.x * nx + evt.y * ny + evt.z * nz;
            const double t_arrival =
                one_plus_z * (evt.t_emission - r_dot_n / constants::c_light.value);

            // filter by observer time window
            if (t_arrival < t_obs_min || t_arrival > t_obs_max) {
                continue;
            }

            // project emission position onto sky plane
            const double proj1 = evt.x * e1x + evt.y * e1y + evt.z * e1z;
            const double proj2 = evt.x * e2x + evt.y * e2y + evt.z * e2z;

            // angular position on sky
            const double sky_r     = std::sqrt(proj1 * proj1 + proj2 * proj2);
            const double theta_sky = sky_r / luminosity_distance;
            const double phi_sky   = std::atan2(proj2, proj1);

            // wrap phi to [0, 2\pi]
            const double phi_wrapped = phi_sky < 0.0 ? phi_sky + 2.0 * std::numbers::pi : phi_sky;

            // find angular bin
            const std::uint32_t i_theta = std::min(
                static_cast<std::uint32_t>(theta_sky / max_angle_rad * (n_theta - 1)),
                n_theta - 1
            );
            const std::uint32_t i_phi = std::min(
                static_cast<std::uint32_t>(phi_wrapped / (2.0 * std::numbers::pi) * n_phi),
                n_phi - 1
            );

            // accumulate intensity (photon packet energy weighted by stokes I)
            result.intensity[i_theta][i_phi] += evt.energy * evt.stokes_I;
        }

        // normalize by solid angle per bin and convert to surface brightness
        const double dtheta = max_angle_rad / (n_theta - 1.0);
        const double dphi   = 2.0 * std::numbers::pi / n_phi;

        for (std::uint32_t ii = 0; ii < n_theta; ++ii) {
            // solid angle element: d\Omega = sin(\theta) d\theta d\phi
            // for small angles: sin(\theta) ~ \theta
            const double theta       = result.theta[ii];
            const double solid_angle = (theta > 0.0) ? theta * dtheta * dphi : dtheta * dphi;

            for (std::uint32_t jj = 0; jj < n_phi; ++jj) {
                if (solid_angle > 0.0) {
                    result.intensity[ii][jj] /= solid_angle;
                }
            }
        }

        return result;
    }

    // compute polarization curve from photon events
    polarization_curve_t compute_polarization_from_events(
        const std::vector<photon_event_t>& events,
        const std::vector<double>&         observer_direction,
        const std::vector<double>&         time_bins,
        double                             energy_min,
        double                             energy_max
    )
    {
        polarization_curve_t result;
        result.times = time_bins;

        const std::size_t n_times = time_bins.size();
        result.polarization_degree.resize(n_times, 0.0);
        result.polarization_angle.resize(n_times, 0.0);
        result.stokes_Q.resize(n_times, 0.0);
        result.stokes_U.resize(n_times, 0.0);
        result.stokes_V.resize(n_times, 0.0);

        // normalize observer direction
        const double obs_mag = std::sqrt(
            observer_direction[0] * observer_direction[0] +
            observer_direction[1] * observer_direction[1] +
            observer_direction[2] * observer_direction[2]
        );
        const std::vector<double> obs_hat = {
            observer_direction[0] / obs_mag,
            observer_direction[1] / obs_mag,
            observer_direction[2] / obs_mag
        };

        // convert time bins from days to seconds
        std::vector<double> time_bins_s(n_times);
        for (std::size_t i = 0; i < n_times; ++i) {
            time_bins_s[i] = time_bins[i] * 86400.0;
        }

        // accumulate stokes parameters in each time bin
        std::vector<double> stokes_I_total(n_times, 0.0);

        for (const auto& evt : events) {
            // skip absorbed photons
            if (evt.absorbed) {
                continue;
            }

            // filter by energy
            if (evt.energy < energy_min || evt.energy > energy_max) {
                continue;
            }

            // check if photon is propagating toward observer
            const double cos_angle =
                evt.px * obs_hat[0] + evt.py * obs_hat[1] + evt.pz * obs_hat[2];
            if (cos_angle < 0.5) { // viewing angle > 60 degrees
                continue;
            }

            // compute observer arrival time
            const double r_emission = std::sqrt(evt.x * evt.x + evt.y * evt.y + evt.z * evt.z);
            const double t_arrival  = evt.t_emission + r_emission / constants::c_light.value;

            // find time bin
            std::size_t t_bin = n_times;
            for (std::size_t i = 0; i < n_times - 1; ++i) {
                if (t_arrival >= time_bins_s[i] && t_arrival < time_bins_s[i + 1]) {
                    t_bin = i;
                    break;
                }
            }
            if (t_bin >= n_times - 1) {
                continue;
            }

            // accumulate stokes parameters
            stokes_I_total[t_bin] += evt.energy * evt.stokes_I;
            result.stokes_Q[t_bin] += evt.energy * evt.stokes_Q;
            result.stokes_U[t_bin] += evt.energy * evt.stokes_U;
            result.stokes_V[t_bin] += evt.energy * evt.stokes_V;
        }

        // compute polarization degree and angle from stokes parameters
        for (std::size_t i = 0; i < n_times; ++i) {
            if (stokes_I_total[i] > 0.0) {
                // normalize stokes parameters
                result.stokes_Q[i] /= stokes_I_total[i];
                result.stokes_U[i] /= stokes_I_total[i];
                result.stokes_V[i] /= stokes_I_total[i];

                // linear polarization degree: P_L = sqrt(Q^2 + U^2) / I
                const double Q                = result.stokes_Q[i];
                const double U                = result.stokes_U[i];
                result.polarization_degree[i] = std::sqrt(Q * Q + U * U);

                // polarization angle: \chi = 0.5 * atan2(U, Q)
                result.polarization_angle[i] = 0.5 * std::atan2(U, Q);
            }
        }

        return result;
    }

    // =============================================================================
    // array-based (numpy-native) implementations
    // =============================================================================

    observer_lightcurve_t compute_lightcurve_from_arrays(
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
        const std::uint8_t* absorbed,
        const double*       observer_direction,
        const double*       frequencies,
        std::size_t         n_frequencies,
        double              redshift,
        double              luminosity_distance,
        const double*       time_bins,
        std::size_t         n_time_bins
    )
    {
        observer_lightcurve_t result;
        result.times.assign(time_bins, time_bins + n_time_bins);
        result.frequencies.assign(frequencies, frequencies + n_frequencies);

        const std::size_t n_bins = n_time_bins - 1;
        result.fluxes.resize(n_bins * n_frequencies, 0.0);

        // convert time bins from days to seconds
        std::vector<double> time_bins_s(n_time_bins);
        for (std::size_t ii = 0; ii < n_time_bins; ++ii) {
            time_bins_s[ii] = time_bins[ii] * 86400.0;
        }

        // normalize observer direction
        const double obs_mag = std::sqrt(
            observer_direction[0] * observer_direction[0] +
            observer_direction[1] * observer_direction[1] +
            observer_direction[2] * observer_direction[2]
        );
        const double obs_hat[3] = {
            observer_direction[0] / obs_mag,
            observer_direction[1] / obs_mag,
            observer_direction[2] / obs_mag
        };

        const double d_L_sq = luminosity_distance * luminosity_distance;

        // process events
        for (std::size_t ii = 0; ii < n_events; ++ii) {
            // skip absorbed photons
            if (absorbed[ii]) {
                continue;
            }

            // check if photon is propagating toward observer
            const double cos_angle =
                px[ii] * obs_hat[0] + py[ii] * obs_hat[1] + pz[ii] * obs_hat[2];
            if (cos_angle < 0.5) {
                continue;
            }

            // compute observer arrival time: t_obs = t_emission + r/c
            const double r_emission = std::sqrt(x[ii] * x[ii] + y[ii] * y[ii] + z[ii] * z[ii]);
            const double t_arrival  = t_emission[ii] + r_emission / constants::c_light.value;

            // find time bin
            std::size_t t_bin = n_bins;
            for (std::size_t jj = 0; jj < n_bins; ++jj) {
                if (t_arrival >= time_bins_s[jj] && t_arrival < time_bins_s[jj + 1]) {
                    t_bin = jj;
                    break;
                }
            }
            if (t_bin >= n_bins) {
                continue;
            }

            // photon frequency: E = h*nu -> nu = E/h
            const double nu_photon = energy[ii] / constants::h_planck.value;
            const double nu_obs    = nu_photon / (1.0 + redshift);

            // find frequency bin
            std::size_t f_bin = n_frequencies;
            for (std::size_t jj = 0; jj < n_frequencies - 1; ++jj) {
                if (nu_obs >= frequencies[jj] && nu_obs < frequencies[jj + 1]) {
                    f_bin = jj;
                    break;
                }
            }
            if (f_bin >= n_frequencies - 1) {
                continue;
            }

            // compute flux contribution
            const double dnu          = frequencies[f_bin + 1] - frequencies[f_bin];
            const double dt           = time_bins_s[t_bin + 1] - time_bins_s[t_bin];
            const double flux_contrib = energy[ii] / (4.0 * std::numbers::pi * d_L_sq * dnu * dt);

            result.fluxes[t_bin * n_frequencies + f_bin] += flux_contrib * stokes_I[ii];
        }

        return result;
    }

    skymap_t compute_skymap_from_arrays(
        std::size_t         n_events,
        const double*       t_emission,
        const double*       x,
        const double*       y,
        const double*       z,
        const double*       energy,
        const double*       stokes_I,
        const std::uint8_t* absorbed,
        const double*       observer_direction,
        double              observer_time,
        double              energy_min,
        double              energy_max,
        double              redshift,
        double              luminosity_distance,
        double              time_window,
        std::uint32_t       n_theta,
        std::uint32_t       n_phi
    )
    {
        const double one_plus_z = 1.0 + redshift;
        skymap_t     result;

        // observer direction (unit vector toward observer)
        const double nx = observer_direction[0];
        const double ny = observer_direction[1];
        const double nz = observer_direction[2];

        // build orthonormal basis for sky plane
        double e1x, e1y, e1z;
        if (std::abs(nz) < 0.99) {
            e1x = -ny;
            e1y = nx;
            e1z = 0.0;
        }
        else {
            e1x = 0.0;
            e1y = -nz;
            e1z = ny;
        }
        double e1_norm = std::sqrt(e1x * e1x + e1y * e1y + e1z * e1z);
        e1x /= e1_norm;
        e1y /= e1_norm;
        e1z /= e1_norm;

        double e2x = ny * e1z - nz * e1y;
        double e2y = nz * e1x - nx * e1z;
        double e2z = nx * e1y - ny * e1x;

        // find max sky radius for angular scale
        double max_sky_radius = 0.0;
        for (std::size_t ii = 0; ii < n_events; ++ii) {
            double proj1   = x[ii] * e1x + y[ii] * e1y + z[ii] * e1z;
            double proj2   = x[ii] * e2x + y[ii] * e2y + z[ii] * e2z;
            double sky_r   = std::sqrt(proj1 * proj1 + proj2 * proj2);
            max_sky_radius = std::max(max_sky_radius, sky_r);
        }

        const double max_angle_rad = (max_sky_radius / luminosity_distance) * 1.5;

        // create angular grid
        result.theta.resize(n_theta);
        result.phi.resize(n_phi);
        result.intensity.resize(n_theta, std::vector<double>(n_phi, 0.0));

        for (std::uint32_t ii = 0; ii < n_theta; ++ii) {
            result.theta[ii] = max_angle_rad * ii / (n_theta - 1.0);
        }
        for (std::uint32_t jj = 0; jj < n_phi; ++jj) {
            result.phi[jj] = 2.0 * std::numbers::pi * jj / n_phi;
        }

        // time window in seconds
        const double t_obs_s       = observer_time * 86400.0;
        const double time_window_s = time_window * 86400.0;
        const double t_obs_min     = t_obs_s - 0.5 * time_window_s;
        const double t_obs_max     = t_obs_s + 0.5 * time_window_s;

        // bin events
        for (std::size_t ii = 0; ii < n_events; ++ii) {
            if (absorbed[ii]) {
                continue;
            }

            if (energy[ii] < energy_min || energy[ii] > energy_max) {
                continue;
            }

            // correct arrival time: t_obs = (1+z) * (t_em - (r dot n) / c)
            const double r_dot_n = x[ii] * nx + y[ii] * ny + z[ii] * nz;
            const double t_arrival =
                one_plus_z * (t_emission[ii] - r_dot_n / constants::c_light.value);

            if (t_arrival < t_obs_min || t_arrival > t_obs_max) {
                continue;
            }

            // project onto sky plane
            const double proj1 = x[ii] * e1x + y[ii] * e1y + z[ii] * e1z;
            const double proj2 = x[ii] * e2x + y[ii] * e2y + z[ii] * e2z;

            const double sky_r       = std::sqrt(proj1 * proj1 + proj2 * proj2);
            const double theta_sky   = sky_r / luminosity_distance;
            const double phi_sky     = std::atan2(proj2, proj1);
            const double phi_wrapped = phi_sky < 0.0 ? phi_sky + 2.0 * std::numbers::pi : phi_sky;

            const std::uint32_t i_theta = std::min(
                static_cast<std::uint32_t>(theta_sky / max_angle_rad * (n_theta - 1)),
                n_theta - 1
            );
            const std::uint32_t i_phi = std::min(
                static_cast<std::uint32_t>(phi_wrapped / (2.0 * std::numbers::pi) * n_phi),
                n_phi - 1
            );

            result.intensity[i_theta][i_phi] += energy[ii] * stokes_I[ii];
        }

        // normalize by solid angle
        const double dtheta = max_angle_rad / (n_theta - 1.0);
        const double dphi   = 2.0 * std::numbers::pi / n_phi;

        for (std::uint32_t ii = 0; ii < n_theta; ++ii) {
            const double theta       = result.theta[ii];
            const double solid_angle = (theta > 0.0) ? theta * dtheta * dphi : dtheta * dphi;

            for (std::uint32_t jj = 0; jj < n_phi; ++jj) {
                if (solid_angle > 0.0) {
                    result.intensity[ii][jj] /= solid_angle;
                }
            }
        }

        return result;
    }

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
    )
    {
        polarization_curve_t result;
        result.times.assign(time_bins, time_bins + n_time_bins);

        result.polarization_degree.resize(n_time_bins, 0.0);
        result.polarization_angle.resize(n_time_bins, 0.0);
        result.stokes_Q.resize(n_time_bins, 0.0);
        result.stokes_U.resize(n_time_bins, 0.0);
        result.stokes_V.resize(n_time_bins, 0.0);

        // normalize observer direction
        const double obs_mag = std::sqrt(
            observer_direction[0] * observer_direction[0] +
            observer_direction[1] * observer_direction[1] +
            observer_direction[2] * observer_direction[2]
        );
        const double obs_hat[3] = {
            observer_direction[0] / obs_mag,
            observer_direction[1] / obs_mag,
            observer_direction[2] / obs_mag
        };

        // convert time bins from days to seconds
        std::vector<double> time_bins_s(n_time_bins);
        for (std::size_t ii = 0; ii < n_time_bins; ++ii) {
            time_bins_s[ii] = time_bins[ii] * 86400.0;
        }

        std::vector<double> stokes_I_total(n_time_bins, 0.0);

        for (std::size_t ii = 0; ii < n_events; ++ii) {
            if (absorbed[ii]) {
                continue;
            }

            if (energy[ii] < energy_min || energy[ii] > energy_max) {
                continue;
            }

            const double cos_angle =
                px[ii] * obs_hat[0] + py[ii] * obs_hat[1] + pz[ii] * obs_hat[2];
            if (cos_angle < 0.5) {
                continue;
            }

            const double r_emission = std::sqrt(x[ii] * x[ii] + y[ii] * y[ii] + z[ii] * z[ii]);
            const double t_arrival  = t_emission[ii] + r_emission / constants::c_light.value;

            std::size_t t_bin = n_time_bins;
            for (std::size_t jj = 0; jj < n_time_bins - 1; ++jj) {
                if (t_arrival >= time_bins_s[jj] && t_arrival < time_bins_s[jj + 1]) {
                    t_bin = jj;
                    break;
                }
            }
            if (t_bin >= n_time_bins - 1) {
                continue;
            }

            stokes_I_total[t_bin] += energy[ii] * stokes_I[ii];
            result.stokes_Q[t_bin] += energy[ii] * stokes_Q[ii];
            result.stokes_U[t_bin] += energy[ii] * stokes_U[ii];
            result.stokes_V[t_bin] += energy[ii] * stokes_V[ii];
        }

        // compute polarization degree and angle
        for (std::size_t ii = 0; ii < n_time_bins; ++ii) {
            if (stokes_I_total[ii] > 0.0) {
                result.stokes_Q[ii] /= stokes_I_total[ii];
                result.stokes_U[ii] /= stokes_I_total[ii];
                result.stokes_V[ii] /= stokes_I_total[ii];

                const double Q                 = result.stokes_Q[ii];
                const double U                 = result.stokes_U[ii];
                result.polarization_degree[ii] = std::sqrt(Q * Q + U * U);
                result.polarization_angle[ii]  = 0.5 * std::atan2(U, Q);
            }
        }

        return result;
    }

} // namespace simbi::afterglow
