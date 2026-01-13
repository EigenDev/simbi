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
    double calc_shock_bfield(energy_density_t rho_e, double eps_b)
    {
        return std::sqrt((8.0 * std::numbers::pi * eps_b * rho_e).value);
    }

    frequency_t calc_gyration_frequency(double bfield)
    {
        auto frequency_for_unit_field = (3.0 / 4.0 / std::numbers::pi) * (constants::e_charge) /
                                        (constants::m_e * constants::c_light);
        return frequency_for_unit_field.value * bfield;
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

    double vector_magnitude(const std::vector<double> a)
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

    double calc_critical_lorentz(double bfield, time_t time_emitter)
    {
        auto numerator   = 6.0 * std::numbers::pi * constants::m_e * constants::c_light;
        auto denominator = constants::sigma_thomson * time_emitter;
        return (numerator / denominator).value / (bfield * bfield);
    }
    energy_t calc_max_power_per_frequency(double bfield)
    {
        auto coeff =
            (constants::m_e * constants::c_light * constants::c_light * constants::sigma_thomson) /
            (3.0 * constants::e_charge);
        return energy_t{coeff.value * bfield};
    }
    spectral_emissivity_t calc_emissivity(double bfield, number_density_t n, double p)
    {
        double coeff =
            (9.6323 / 8.0 / std::numbers::pi) * (p - 1.0) / (3.0 * p - 1.0) * std::sqrt(3.0);
        double e_cubed = std::pow(constants::e_charge.value, 3);
        double denom   = constants::m_e.value * pow<2>(constants::c_light).value;
        return spectral_emissivity_t{coeff * e_cubed / denom * n.value * bfield};
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
        std::transform(a.begin() + 1, a.end(), b.begin() + 1, v.begin(), std::multiplies<double>());
        return v;
    };

    std::vector<double> vector_subtract(const std::vector<double>& a, const std::vector<double>& b)
    {
        std::vector<double> v(a.size());
        std::transform(a.begin() + 1, a.end(), b.begin() + 1, v.begin(), std::minus<double>());
        return v;
    };

    std::vector<double> vector_add(const std::vector<double>& a, const std::vector<double>& b)
    {
        std::vector<double> v(a.size());
        std::transform(a.begin() + 1, a.end(), b.begin() + 1, v.begin(), std::plus<double>());
        return v;
    };

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
        double photon_energy, // erg
        double n_e,           // electron density [cm^-3]
        double bfield,        // magnetic field [gauss]
        double path_length,   // cm
        double p              // spectral index
    )
    {
        // synchrotron self-absorption coefficient
        // α_ν ∝ n_e B (ν_g/ν)^{(p+4)/2}

        double nu_photon = photon_energy / constants::h_planck.value;
        double nu_gyro   = constants::e_charge.value * bfield /
                         (2.0 * std::numbers::pi * constants::m_e.value * constants::c_light.value);
        double nu_ratio = nu_gyro / nu_photon;

        if (nu_ratio < 1.0) {
            return 0.0; // above synchrotron peak, SSA negligible
        }

        // SSA coefficient (simplified, CGS units: cm^-1)
        // prefactor calibrated for typical GRB afterglow conditions
        double alpha_ssa = 3.3e-10 * n_e * bfield * std::pow(nu_ratio, (p + 4.0) / 2.0);

        return alpha_ssa * path_length;
    }

    // helper: compute thomson scattering optical depth
    double compute_thomson_optical_depth(double n_e, double path_length)
    {
        // \tau_T = n_e \sigma_T L
        return n_e * constants::sigma_thomson.value * path_length;
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
            const auto n_e =
                rho[cell_id] * qscales.rho_scale * units::g_per_cm3.value / constants::m_p.value;

            // estimate path length through cell (approximate as ~10% of radius)
            double path_length = x1[0] * qscales.length_scale * 0.1;

            // compute optical depths
            double tau_ssa = compute_ssa_optical_depth(photon.energy, n_e, bfield, path_length, p);

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
                // γγ → e⁺e⁻ threshold: E > ~100 MeV
                double threshold_energy = 1e-4; // erg (~60 MeV)
                if (photon.energy > threshold_energy) {
                    // simplified: mark as absorbed if above threshold
                    // proper treatment would compute γγ opacity
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

        // Extract the geomtry of the mesh
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
        const double p      = args.p;     // Electron number index
        const double eps_b  = args.eps_b; // Magnetic field fraction of internal energy
        const double eps_e  = args.eps_e; // shocked electrons fraction of internal energy

        const auto t_prime = args.current_time * qscales.time_scale * units::s;
        const auto dt      = args.dt * qscales.time_scale * units::s;
        for (std::size_t kk = 0; kk < x3.size(); kk++) {
            const double x3l     = (kk > 0) ? x2min + (kk - 0.5) * dx3 : x3min;
            const double x3r     = (kk < nk - 1) ? x3l + dx3 * (kk == 0 ? 0.5 : 1.0) : x3max;
            const double sin_phi = std::sin(x3[kk]);
            const double cos_phi = std::cos(x3[kk]);
            const double dx3     = x3r - x3l;

            // If the data is 3D, then there is a real k-space to pull data from
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
                    const auto w         = calc_lorentz_factor(gb[central_idx]); // Lorentz factor
                    const auto t_emitter = t_prime / w; // time in emitter frame

                    const double              phi_prime  = 2.0 * std::numbers::pi * dis(gen);
                    const double              mu_prime   = 2.0 * dis(gen) - 1.0;
                    const std::vector<double> nhat_prime = {
                        std::sin(std::acos(mu_prime)) * std::cos(phi_prime),
                        std::sin(std::acos(mu_prime)) * std::sin(phi_prime),
                        mu_prime
                    };

                    // Cosine of the isotropic emission angle wrt to the
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

                    //================================================================
                    //                    HYDRO CONDITIONS
                    //================================================================
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
                    ); // Minimum Lorentz factor of electrons
                    const auto gamma_crit = calc_critical_lorentz(bfield, t_emitter);

                    const auto gamma_max = std::max(gamma_min, gamma_crit);
                    const auto gamma_low = std::min(gamma_min, gamma_crit);
                    const auto dg        = (gamma_max - gamma_low) / (ng - 1);

                    // Calc cell volumes
                    const double x1l =
                        (ii > 0) ? x1min * std::pow(10.0, (ii - 0.5) * dlogx1) : x1min;
                    const double x1r     = (ii < ni - 1)
                                               ? x1l * std::pow(10.0, dlogx1 * (ii == 0 ? 0.5 : 1.0))
                                               : x1max;
                    const auto   dvolume = dx3 * dcos * (1.0 / 3.0) *
                                         (x1r * x1r * x1r - x1l * x1l * x1l) *
                                         qscales.length_scale * qscales.length_scale *
                                         qscales.length_scale * units::cm3;

                    // Each cell will have its own photons distribution.
                    // To account for this, we divide the gamma bins up
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
        // Place observer along chosen axis
        const std::vector<double> obs_hat =
            {std::sin(args.theta_obs), 0.0, std::cos(args.theta_obs)};

        const auto nt = tbin_edges.size(); // time bin size
        const auto nf = args.nus.size();   // frequency_t bin size

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
        const bool at_pole = std::abs(std::cos(args.theta_obs)) == 1;

        size_t nk      = 1;
        double sin_phi = 0;
        double cos_phi = 1.0;
        double dx3     = 2.0 * std::numbers::pi;
        // Check whether to do 3D (off-axis) or not
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

            // If the data is 3D, then there is a real k-space to pull data from
            const std::int64_t kreal = (data_dim > 2) * kk;
#pragma omp parallel
            for (std::uint64_t jj = 0; jj < nj; jj++) {
                const double x2l  = (jj > 0) ? x2min + (jj - 0.5) * dx2 : x2min;
                const double x2r  = (jj < nj - 1) ? x2l + dx2 * (jj == 0 ? 0.5 : 1.0) : x2max;
                const double dcos = std::cos(x2l) - std::cos(x2r);

                // radial unit vector
                const std::vector<double> rhat =
                    {std::sin(x2[jj]) * cos_phi, std::sin(x2[jj]) * sin_phi, std::cos(x2[jj])};

                // Data greater than 1D? Cool, there is a j space to pull data
                // from
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
                    ); // Minimum Lorentz factor of electrons
                    const auto gamma_crit = calc_critical_lorentz(
                        bfield,
                        t_emitter
                    ); // Critical Lorentz factor of electrons

                    // Calc cell volumes
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

                    // Calculate the maximum flux based on the average
                    // bolometric power per electron
                    const frequency_t nu_c = calc_nu(gamma_crit, nu_g); // Critical frequency_t
                    const frequency_t nu_m = calc_nu(gamma_min, nu_g);  // Minimum frequency_t
                    const double      delta_doppler = calc_delta_doppler(
                        w,
                        beta_vec,
                        obs_hat
                    ); // Doppler factor
                    const spectral_emissivity_t eps_m = calc_emissivity(
                        bfield,
                        n_e_proper,
                        p
                    ); // Emissivity per cell

                    // Total emitted power per unit
                    // frequency_t in each cell volume
                    const spectral_power_t power_prime =
                        dvolume * eps_m * delta_doppler * delta_doppler;
                    const double t_obs_day = (t_obs / day);
                    // loop through the given frequencies and put them in their
                    // respective locations in dictionary
                    for (size_t fidx = 0; fidx < nf; fidx++) {
                        // The frequency_t we see is doppler boosted, so account
                        // for that
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
                                // Sum the fluxes in the given time bin
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

        for (std::size_t kk = 0; kk < nk; kk++) {
            const double       sin_phi = std::sin(x3[kk]);
            const double       cos_phi = std::cos(x3[kk]);
            const std::int64_t kreal   = (data_dim > 2) * kk;

            for (std::size_t jj = 0; jj < nj; jj++) {
                const std::vector<double> rhat =
                    {std::sin(x2[jj]) * cos_phi, std::sin(x2[jj]) * sin_phi, std::cos(x2[jj])};
                const std::int64_t jreal = (data_dim > 1) * jj;

                for (std::size_t ii = 0; ii < ni; ii++) {
                    if (events_generated >= max_events) {
                        goto done;
                    }

                    const auto central_idx = kreal * ni * nj + jreal * ni + ii;
                    const auto beta        = calc_beta(gb[central_idx]);
                    const auto w           = calc_lorentz_factor(gb[central_idx]);

                    const auto rho_einternal = pre[central_idx] * qscales.pre_scale /
                                               (args.adiabatic_index - 1.0) * units::erg_per_cm3;
                    const auto bfield = calc_shock_bfield(rho_einternal, eps_b);
                    const auto n_e_proper =
                        rho[central_idx] * qscales.rho_scale * units::g_per_cm3 / constants::m_p;
                    const auto nu_g = calc_gyration_frequency(bfield);
                    const auto gamma_min =
                        calc_minimum_lorentz(eps_e, rho_einternal, n_e_proper, p);

                    const auto r_center = x1[ii] * qscales.length_scale * units::cm;

                    for (std::uint64_t pp = 0; pp < photons_target; pp++) {
                        if (events_generated >= max_events) {
                            goto done;
                        }

                        photon_event_t evt;

                        const double gamma_sample =
                            gen_random_from_powerlaw(gamma_min, gamma_min * 10.0, p, dis(gen));
                        const auto nu_c     = calc_nu(gamma_sample, nu_g);
                        const auto E_photon = constants::h_planck * nu_c;

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

                        evt.energy = E_photon.value;
                        evt.px     = nhat_beamed[0];
                        evt.py     = nhat_beamed[1];
                        evt.pz     = nhat_beamed[2];

                        evt.stokes_I = evt.energy;
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

} // namespace simbi::afterglow
