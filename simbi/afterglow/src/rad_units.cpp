/**
    A radiation module that will be used to calculate synchtron lights curves,
   spectra, and sky maps from high-energy physical processes
    @file rad_units.cpp
    @author Marcus DuPont
    @version 0.1 05/20/22
*/
#include "rad_units.hpp"
#include <algorithm>
#include <functional>
#include <random>

namespace sogbo_rad {

    /*
    The Doppler Boost Factor
    */
    double calc_delta_doppler(
        const double lorentz_factor,
        const std::vector<double>& beta_vec,
        const std::vector<double>& nhat
    )
    {
        return 1.0 /
               (lorentz_factor * (1.0 - vector_dotproduct(beta_vec, nhat)));
    }

    /*
    The velocity of the flow in dimensions of speed of ligt
    */
    double calc_beta(const double gamma_beta)
    {
        return gamma_beta / std::sqrt(1 + gamma_beta * gamma_beta);
    }

    /*
    The Lorentz factor
    */
    double calc_lorentz_factor(const double gamma_beta)
    {
        return std::sqrt(1.0 + gamma_beta * gamma_beta);
    }

    /*
    The magnetic field behind the shock
    */
    units::mag_field
    calc_shock_bfield(const units::edens rho_e, const double eps_b)
    {
        return units::math::sqrt(8.0 * M_PI * eps_b * rho_e);
    }

    /*
    The canonical gyration frequency for a particle in a magnetic field
    */
    units::frequency calc_gyration_frequency(const units::mag_field bfield)
    {
        auto frequency_for_unit_field = (3.0 / 4.0 / M_PI) *
                                        (constants::e_charge) /
                                        (constants::m_e * constants::c_light);
        return frequency_for_unit_field * bfield;
    }

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
    )
    {
        return (4.0 / 3.0) * constants::sigmaT * constants::c_light * beta *
               beta * lorentz_factor * lorentz_factor * ub;
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
    double calc_nphotons_per_bin(
        units::volume volume,
        units::ndens n_e,
        units::frequency nu_g,
        units::edens ub,
        units::mytime dt,
        double gamma_e,
        double beta,
        double p
    )
    {
        const auto a =
            (8.0 * M_PI * volume / (3.0 * constants::h_planck * nu_g));
        const auto b =
            constants::sigmaT * constants::c_light * beta * beta * ub * n_e;
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
    double calc_nphotons_other(
        const units::power power,
        const units::frequency nu_c,
        const units::mytime dt
    )
    {
        return power * dt / (constants::h_planck * nu_c);
    }

    //  Power-law generator for pdf(x) ~ x^{g-1} for a<=x<=b
    double
    gen_random_from_powerlaw(double a, double b, double p, double random_number)
    {
        const double g  = 1 - p;
        const double ag = std::pow(a, g);
        const double bg = std::pow(b, g);
        return std::pow(ag + (bg - ag) * random_number, (1.0 / g));
    }

    // calculate magnitude of vector
    double vector_magnitude(const std::vector<double> a)
    {
        double mag = 0;
        for (const auto val : a) {
            mag += val * val;
        }
        return std::sqrt(mag);
    }

    // calculate vector dot product
    double vector_dotproduct(
        const std::vector<double>& a,
        const std::vector<double>& b
    )
    {
        double mag = 0;
        for (size_t ii = 0; ii < a.size(); ii++) {
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
    units::ndens calc_nelectrons_at_gamma_e(
        const units::ndens n_e,
        const double gamma_e,
        const double p
    )
    {
        return n_e * std::pow(gamma_e, -p);
    }

    // Calculate the synchrotron frequency as function of lorentz_factor
    units::frequency calc_nu(const double gamma_e, const units::frequency nu_g)
    {
        return nu_g * gamma_e * gamma_e;
    }

    // Calculate the critical Lorentz factor as function of time and magnetic
    // field
    double calc_critical_lorentz(
        const units::mag_field bfield,
        const units::mytime time_emitter
    )
    {
        return (6.0 * M_PI * constants::m_e * constants::c_light) /
               (constants::sigmaT * bfield * bfield * time_emitter);
    }

    // Calculate the maximum power per frequency as Eq. (5) in Sari, Piran.
    // Narayan(1999)
    units::energy calc_max_power_per_frequency(units::mag_field bfield)
    {
        return (constants::m_e * constants::c_light * constants::c_light *
                constants::sigmaT) /
               (3.0 * constants::e_charge) * bfield;
    }

    /**
     *  Calculate the peak emissivity per frequency per equation (A3) in
        https://iopscience.iop.org/article/10.1088/0004-637X/749/1/44/pdf
    */
    units::emissivity calc_emissivity(
        const units::mag_field bfield,
        const units::ndens n,
        const double p
    )
    {
        return (
            (9.6323 / 8.0 / M_PI) * (p - 1.0) / (3.0 * p - 1.0) *
            std::sqrt(3.0) *
            units::math::pow<std::ratio<3>>(constants::e_charge) /
            (constants::m_e * constants::c_light * constants::c_light) * n *
            bfield
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
    double calc_minimum_lorentz(
        const double eps_e,
        const units::edens e_thermal,
        const units::ndens n,
        const double p
    )
    {
        return eps_e * (p - 2.0) / (p - 1.0) * e_thermal /
               (n * constants::m_e * constants::c_light * constants::c_light);
    }

    std::vector<double>
    vector_multiply(const std::vector<double>& a, const std::vector<double>& b)
    {
        std::vector<double> v(a.size());
        std::transform(
            a.begin() + 1,
            a.end(),
            b.begin() + 1,
            v.begin(),
            std::multiplies<double>()
        );
        return v;
    };

    std::vector<double>
    vector_subtract(const std::vector<double>& a, const std::vector<double>& b)
    {
        std::vector<double> v(a.size());
        std::transform(
            a.begin() + 1,
            a.end(),
            b.begin() + 1,
            v.begin(),
            std::minus<double>()
        );
        return v;
    };

    std::vector<double>
    vector_add(const std::vector<double>& a, const std::vector<double>& b)
    {
        std::vector<double> v(a.size());
        std::transform(
            a.begin() + 1,
            a.end(),
            b.begin() + 1,
            v.begin(),
            std::plus<double>()
        );
        return v;
    };

    std::vector<double>
    scale_vector(const std::vector<double>& a, const double scalar)
    {
        std::vector<double> v = a;
        std::transform(v.begin(), v.end(), v.begin(), [&scalar](auto& c) {
            return c * scalar;
        });
        return v;
    }

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
    )
    {
        const bool slow_cool   = nu_c > nu_m;
        auto power_with_breaks = power_max;
        if (slow_cool) {
            const bool slow_break1 = nu_prime < nu_m;
            const bool slow_break2 = (nu_prime < nu_c) && (nu_prime > nu_m);
            if (slow_break1) {
                power_with_breaks *= std::pow(nu_prime / nu_m, (1.0 / 3.0));
            }
            else if (slow_break2) {
                power_with_breaks *=
                    std::pow(nu_prime / nu_m, -0.5 * (p - 1.0));
            }
            else {
                power_with_breaks *= std::pow(nu_c / nu_m, -0.5 * (p - 1.0)) *
                                     std::pow(nu_prime / nu_c, -0.5 * p);
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
                power_with_breaks *= std::pow(nu_m / nu_c, -0.5) *
                                     std::pow(nu_prime / nu_m, -0.5 * p);
            }
        }
        return power_with_breaks;
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
     * @param photon_distribution a 1D array of the time bin edges for the flux
     * calculations
     * @param four_position a 1D array of the frequency bin edges for the flux
     * calculartions
     *
     */
    void log_events(
        const sim_conditions args,
        const quant_scales qscales,
        std::vector<std::vector<double>>& fields,
        std::vector<std::vector<double>>& mesh,
        std::vector<double>& photon_distribution,
        std::vector<double>& four_position,
        const int data_dim
    )
    {
        std::random_device
            rd;   // Will be used to obtain a seed for the random number engine
        std::mt19937 gen(
            rd()
        );   // Standard mersenne_twister_engine seeded with rd()
        std::uniform_real_distribution<> dis(0.0, 1.0);

        int ng         = 100;
        const auto rho = fields[0];   // fluid frame density
        const auto gb  = fields[1];   // four-velocity
        const auto pre = fields[2];   // pressure

        // Extract the geomtry of the mesh
        const auto x1     = mesh[0];
        const auto x2     = mesh[1];
        const auto x3     = mesh[2];
        const auto ni     = x1.size();
        const auto nj     = x2.size();
        const auto nk     = x3.size();
        const auto x1max  = x1[ni - 1];
        const auto x1min  = x1[0];
        const auto dlogx1 = std::log10(x1max / x1min) / (ni - 1);
        const auto x2max  = x2[nj - 1];
        const auto x2min  = x2[0];
        const auto dx2    = (x2max - x2min) / (nj - 1);
        const auto x3max  = x3[nk - 1];
        const auto x3min  = x3[0];
        const auto dx3    = (x3max - x2min) / (nk - 1);
        const double p    = args.p;   // Electron number index
        const double eps_b =
            args.eps_b;   // Magnetic field fraction of internal energy
        const double eps_e =
            args.eps_e;   // shocked electrons fraction of internal energy

        // time in source frame
        const auto t_prime = args.current_time * qscales.time_scale * units::s;
        // step size between checkpoints
        const auto dt = args.dt * qscales.time_scale * units::s;
        for (size_t kk = 0; kk < x3.size(); kk++) {
            const double x3l = (kk > 0) ? x2min + (kk - 0.5) * dx3 : x3min;
            const double x3r =
                (kk < nk - 1) ? x3l + dx3 * (kk == 0 ? 0.5 : 1.0) : x3max;
            const double sin_phi = std::sin(x3[kk]);
            const double cos_phi = std::cos(x3[kk]);
            const double dx3     = x3r - x3l;

            // If the data is 3D, then there is a real k-space to pull data from
            const int kreal = (data_dim > 2) * kk;
#pragma omp parallel
            for (size_t jj = 0; jj < x2.size(); jj++) {
                const double x2l = (jj > 0) ? x2min + (jj - 0.5) * dx2 : x2min;
                const double x2r =
                    (jj < nj - 1) ? x2l + dx2 * (jj == 0 ? 0.5 : 1.0) : x2max;
                const double dcos = std::cos(x2l) - std::cos(x2r);

                // radial unit vector
                const std::vector<double> rhat = {
                  std::sin(x2[jj]) * cos_phi,
                  std::sin(x2[jj]) * sin_phi,
                  std::cos(x2[jj])
                };
                const int jreal = (data_dim > 1) * jj;
#pragma omp for nowait
                for (size_t ii = 0; ii < x1.size(); ii++) {
                    const auto central_idx = kreal * ni * nj + jreal * ni +
                                             ii;   // index for current zone
                    const auto beta =
                        calc_beta(gb[central_idx]);   // velocity in units of c
                    const auto w = calc_lorentz_factor(
                        gb[central_idx]
                    );   // Lorentz factor
                    const auto t_emitter =
                        t_prime / w;   // time in emitter frame
                    //================================================================
                    //                 DIRECTIONAL SAMPLING RULES
                    //================================================================

                    // ============================================================
                    // Source Trajectory
                    const double phi_prime = 2.0 * M_PI * dis(gen);
                    const double mu_prime  = 2.0 * dis(gen) - 1.0;
                    const std::vector<double> nhat_prime = {
                      std::sin(std::acos(mu_prime)) * std::cos(phi_prime),
                      std::sin(std::acos(mu_prime)) * std::sin(phi_prime),
                      mu_prime
                    };

                    // Cosine of the isotropic emission angle wrt to the
                    // propagation direction
                    const double mu_rhat_prime =
                        vector_dotproduct(rhat, nhat_prime);
                    // cos of the resulting beamed angle in the plane of rhat
                    // and nhat prime
                    const double mu_rhat_beam =
                        (mu_rhat_prime + beta) / (1.0 + beta * mu_rhat_prime);
                    const double rot_angle =
                        std::acos(mu_rhat_prime) -
                        std::acos(
                            mu_rhat_beam
                        );   // rotation angle from initial emission direction
                             // to beaming direction
                    const auto nhat_beamed =
                        scale_vector(nhat_prime, std::cos(rot_angle));
                    const auto nvec_lab = scale_vector(nhat_beamed, x1[ii]);
                    const std::vector<double> x_mu =
                        {t_prime.value, nvec_lab[0], nvec_lab[1], nvec_lab[2]};

                    //================================================================
                    //                    HYDRO CONDITIONS
                    //================================================================
                    const auto rho_einternal =
                        pre[central_idx] * qscales.pre_scale /
                        (args.adiabatic_index - 1.0) *
                        units::erg_per_cm3;   // internal energy density
                    const auto bfield = calc_shock_bfield(
                        rho_einternal,
                        eps_b
                    );   // magnetic field based on equipartition
                    const auto n_e_proper =
                        rho[central_idx] * qscales.rho_scale *
                        units::g_per_cm3 /
                        constants::m_p;   // electron number density
                    const auto nu_g = calc_gyration_frequency(
                        bfield
                    );   // gyration frequency // distance to source
                    const auto gamma_min = calc_minimum_lorentz(
                        eps_e,
                        rho_einternal,
                        n_e_proper,
                        p
                    );   // Minimum Lorentz factor of electrons
                    const auto gamma_crit =
                        calc_critical_lorentz(bfield, t_emitter);

                    const auto gamma_max = std::max(gamma_min, gamma_crit);
                    const auto gamma_low = std::min(gamma_min, gamma_crit);
                    const auto dg        = (gamma_max - gamma_low) / (ng - 1);

                    // Calc cell volumes
                    const double x1l =
                        (ii > 0) ? x1min * std::pow(10.0, (ii - 0.5) * dlogx1)
                                 : x1min;
                    const double x1r =
                        (ii < ni - 1)
                            ? x1l *
                                  std::pow(10.0, dlogx1 * (ii == 0 ? 0.5 : 1.0))
                            : x1max;
                    const auto dvolume = dx3 * dcos * (1.0 / 3.0) *
                                         (x1r * x1r * x1r - x1l * x1l * x1l) *
                                         qscales.length_scale *
                                         qscales.length_scale *
                                         qscales.length_scale * units::cm3;

                    // Each cell will have its own photons distribution.
                    // To account for this, we divide the gamma bins up
                    // and bin the photons in each cell with respect to the
                    // gamma bin
                    const auto n_e = n_e_proper * w;
                    const auto ub  = bfield * bfield / 8.0 / M_PI;
                    for (int qq = 0; qq < ng; qq++) {
                        const auto gamma_e      = gamma_min + qq * dg;
                        const auto gamma_sample = gen_random_from_powerlaw(
                            gamma_e,
                            gamma_e + dg,
                            p,
                            dis(gen)
                        );
                        const auto nu_c  = calc_nu(gamma_sample, nu_g);
                        const auto nphot = calc_nphotons_per_bin(
                                               dvolume,
                                               n_e,
                                               nu_g,
                                               ub,
                                               dt,
                                               gamma_e,
                                               beta,
                                               p
                                           ) *
                                           dg;
                        // photon energy in erg
                        photon_distribution
                            [kk * ni * nj * ng + jj * ni * ng + ii * ng + qq] =
                                constants::h_planck.value * nu_c.value * nphot;
                    }

                    // log the four-position
                    for (int qq = 0; qq < 4; qq++) {
                        four_position
                            [kk * ni * nj * 4 + jj * ni * 4 + ii * 4 + qq] =
                                x_mu[qq];
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
     * @param fbin_edges a 1D array of the frequency bin edges for the flux
     * calculartions
     * @param flux_array a flattened 1D array in which the summed frequencies in
     * each bin will live
     * @param checkpoint_index  the integer index of the checkpoint file
     *
     */
    void calc_fnu(
        const sim_conditions args,
        const quant_scales qscales,
        const std::vector<double>& rho,
        const std::vector<double>& gb,
        const std::vector<double>& pre,
        const std::vector<std::vector<double>>& mesh,
        const std::vector<double>& tbin_edges,
        std::vector<double>& flux_array,
        const int checkpoint_index,
        const int data_dim
    )
    {
        // Place observer along chosen axis
        const std::vector<double> obs_hat =
            {std::sin(args.theta_obs), 0.0, std::cos(args.theta_obs)};

        const auto nt = tbin_edges.size();   // time bin size
        const auto nf = args.nus.size();     // frequency bin size

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
        double dx3     = 2.0 * M_PI;
        // Check whether to do 3D (off-axis) or not
        std::vector<double> x3;
        double x3max = 0.0;
        double x3min = 0.0;
        if (!at_pole) {
            x3    = mesh[2];
            nk    = x3.size();
            x3max = x3[nk - 1];
            x3min = x3[0];
            dx3   = (x3max - x3min) / (nk - 1);
        }

        const double p = args.p;   // Electron number index
        const double eps_b =
            args.eps_b;   // Magnetic field fraction of internal energy
        const double eps_e =
            args.eps_e;   // shocked electrons fraction of internal energy
        const auto d = args.d_L * units::cm;   // Luminosity distance

        // time in source frame
        const auto t_prime = args.current_time * qscales.time_scale * units::s;
        // step size between checkpoints
        const auto dt = args.dt * qscales.time_scale * units::s;
        const auto flux_denom =
            units::math::pow<std::ratio<-1>>(4.0 * M_PI * d * d);
        for (size_t kk = 0; kk < nk; kk++) {
            if (!at_pole) {
                const double x3l = (kk > 0) ? x2min + (kk - 0.5) * dx3 : x3min;
                const double x3r =
                    (kk < nk - 1) ? x3l + dx3 * (kk == 0 ? 0.5 : 1.0) : x3max;
                sin_phi = std::sin(x3[kk]);
                cos_phi = std::cos(x3[kk]);
                dx3     = x3r - x3l;
            }

            // If the data is 3D, then there is a real k-space to pull data from
            const int kreal = (data_dim > 2) * kk;
#pragma omp parallel
            for (size_t jj = 0; jj < nj; jj++) {
                const double x2l = (jj > 0) ? x2min + (jj - 0.5) * dx2 : x2min;
                const double x2r =
                    (jj < nj - 1) ? x2l + dx2 * (jj == 0 ? 0.5 : 1.0) : x2max;
                const double dcos = std::cos(x2l) - std::cos(x2r);

                // radial unit vector
                const std::vector<double> rhat = {
                  std::sin(x2[jj]) * cos_phi,
                  std::sin(x2[jj]) * sin_phi,
                  std::cos(x2[jj])
                };

                // Data greater than 1D? Cool, there is a j space to pull data
                // from
                const int jreal = (data_dim > 1) * jj;
#pragma omp for nowait
                for (size_t ii = 0; ii < ni; ii++) {
                    const auto central_idx = kreal * ni * nj + jreal * ni +
                                             ii;   // index for current zone
                    const auto beta =
                        calc_beta(gb[central_idx]);   // velocity in units of c
                    const auto w = calc_lorentz_factor(
                        gb[central_idx]
                    );   // Lorentz factor
                    const auto t_emitter =
                        t_prime / w;   // time in emitter frame
                    //================================================================
                    //                    HYDRO CONDITIONS
                    //================================================================

                    const auto rho_einternal =
                        pre[central_idx] * qscales.pre_scale /
                        (args.adiabatic_index - 1.0) *
                        units::erg_per_cm3;   // internal energy density
                    const auto bfield = calc_shock_bfield(
                        rho_einternal,
                        eps_b
                    );   // magnetic field based on equipartition
                    const auto n_e_proper =
                        rho[central_idx] * qscales.rho_scale *
                        units::g_per_cm3 /
                        constants::m_p;   // electron number density
                    const auto nu_g = calc_gyration_frequency(
                        bfield
                    );   // gyration frequency // distance to source
                    const auto gamma_min = calc_minimum_lorentz(
                        eps_e,
                        rho_einternal,
                        n_e_proper,
                        p
                    );   // Minimum Lorentz factor of electrons
                    const auto gamma_crit = calc_critical_lorentz(
                        bfield,
                        t_emitter
                    );   // Critical Lorentz factor of electrons

                    // Calc cell volumes
                    const double x1l =
                        (ii > 0) ? x1min * std::pow(10.0, (ii - 0.5) * dlogx1)
                                 : x1min;
                    const double x1r =
                        (ii < ni - 1)
                            ? x1l *
                                  std::pow(10.0, dlogx1 * (ii == 0 ? 0.5 : 1.0))
                            : x1max;
                    const auto dvolume = dx3 * dcos * (1.0 / 3.0) *
                                         (x1r * x1r * x1r - x1l * x1l * x1l) *
                                         qscales.length_scale *
                                         qscales.length_scale *
                                         qscales.length_scale * units::cm3;

                    // observer time
                    const auto t_obs =
                        t_prime - x1[ii] * qscales.length_scale *
                                      vector_dotproduct(rhat, obs_hat) *
                                      units::cm / constants::c_light;

                    const std::vector<double> beta_vec =
                        {beta * rhat[0], beta * rhat[1], beta * rhat[2]};

                    // Calculate the maximum flux based on the average
                    // bolometric power per electron
                    const units::frequency nu_c =
                        calc_nu(gamma_crit, nu_g);   // Critical frequency
                    const units::frequency nu_m =
                        calc_nu(gamma_min, nu_g);   // Minimum frequency
                    const double delta_doppler = calc_delta_doppler(
                        w,
                        beta_vec,
                        obs_hat
                    );   // Doppler factor
                    const units::emissivity eps_m = calc_emissivity(
                        bfield,
                        n_e_proper,
                        p
                    );   // Emissivity per cell
                    const units::spec_power power_prime =
                        dvolume * eps_m * delta_doppler *
                        delta_doppler;   // Total emitted power per unit
                                         // frequency in each cell volume

                    const auto t_obs_day = t_obs.to(units::day).value;
                    // loop through the given frequencies and put them in their
                    // respective locations in dictionary
                    for (size_t fidx = 0; fidx < nf; fidx++) {
                        // The frequency we see is doppler boosted, so account
                        // for that
                        const units::frequency nu_source =
                            args.nus[fidx] * units::hz / delta_doppler;
                        const units::spec_power power_cool = calc_powerlaw_flux(
                            power_prime,
                            p,
                            nu_source,
                            nu_c,
                            nu_m
                        );
                        const units::spectral_flux f_nu =
                            (power_cool * flux_denom).to(units::mjy);

                        // place the fluxes in the appropriate time bins
                        for (size_t tidx = 0; tidx < nt - 1; tidx++) {
                            const double t1 = tbin_edges[tidx + 0];
                            const double t2 = tbin_edges[tidx + 1];
                            if (t1 < t_obs_day && t_obs_day < t2) {
                                // the effective lifetime of the emitting cell
                                // must be accounted for
                                const auto dt_day = dt.to(units::day);
                                const auto dt_obs = t2 - t1;
                                const double trat = (checkpoint_index > 0)
                                                        ? dt_day.value / dt_obs
                                                        : 1.0;
                                // Sum the fluxes in the given time bin
                                flux_array[fidx * (nt - 1) + tidx] +=
                                    trat * f_nu.value;
                                break;
                            }
                        }   // end time bin loop
                    }
                }   // end inner parallel loop

            }   // end outer parallel loop
        }

        // return flux_array;
    }

}   // namespace sogbo_rad
