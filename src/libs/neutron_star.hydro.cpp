
    #include <cmath>
extern "C" {
    constexpr double rho_wind = 1.0;
    constexpr double uwind   = 1.0;
    double lorentz = std::sqrt(1.0 + uwind*uwind);
    constexpr double rwind = 1.5;
    constexpr double sigma = 1.e-1;
    double beta  = std::sqrt(1.0 - 1.0/(lorentz*lorentz));
    constexpr double omega = 1.0;

    double dens_source(double r, double theta, double phi, double t) {
        return 0.0;
        // if (r < rwind && t == 0.0) {
        //     return omega * (lorentz * rho_wind / (r * r));
        // } else {
        //     return 0.0;
        // };
    }

    double b1_source(double r, double theta, double phi, double t) {
        return 0.0;
    }

    double b2_source(double r, double theta, double phi, double t) {
        return 0.0;
    }

    double b3_source(double r, double theta, double phi, double t) {
        return 0.0;
        // double sint = std::sin(theta);
        // if (r < rwind && t == 0.0) {
        //     double rho = dens_source(r, theta, phi, t) / lorentz;
        //     if (theta < M_PI/2.0) {
        //         return +omega * std::sqrt(sigma * rho * sint * sint);
        //     } else {
        //         return +omega * std::sqrt(sigma * rho * sint * sint);
        //     }
        // } else {
        //     return 0.0;
        // }
    }

    double mom1_source(double r, double theta, double phi, double t) {
        return 0.0;
        // if (r < rwind && t == 0.0) {
        //     double dens = dens_source(r, theta, phi, t);
        //     double bfield = b3_source(r, theta, phi, t);
        //     double bsq = bfield * bfield;
        //     double rho = dens / lorentz;
        //     double pwind = 1e-10 * rho;
        //     double enthalpy = 1.0 + (1.3333333333333333 * pwind) / (rho * (1.3333333333333333 - 1.0));
        //     return omega * (dens * enthalpy * uwind + bsq * beta);
        // } else {
        //     return 0.0;
        // };
    }

    double mom2_source(double r, double theta, double phi, double t) {
        return 0.0;
    }

    double mom3_source(double r, double theta, double phi, double t) {
        return 0.0;
    }

    double ener_source(double r, double theta, double phi, double t) {
        return 0.0;
        // if (r < rwind && t == 0.0) {
        //     double dens = dens_source(r, theta, phi, t);
        //     double bfield = b3_source(r, theta, phi, t);
        //     double bsq = bfield * bfield;
        //     double rho = dens / lorentz;
        //     double pwind = 1e-10 * rho;
        //     double enthalpy = 1.0 + (1.3333333333333333 * pwind) / (rho * (1.3333333333333333 - 1.0));
        //     return dens * lorentz * enthalpy - pwind - rho * lorentz + 0.5 * (bsq + beta * beta * bsq);
        // } else {
        //     return 0.0;
        // }
    }

    double chi_source(double r, double theta, double phi, double t) {
        return 0.0;
    }
}
