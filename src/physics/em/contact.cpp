#include "contact.hpp"
#include "base/concepts.hpp"
#include "config.hpp"
#include "containers/vector.hpp"

namespace simbi::em {
    // Constrained transport contact algorithm developed by Gardiner & Stone
    // https://ui.adsabs.harvard.edu/abs/2005JCoPh.205..509G/abstract

    // CT Contact formula (Gardiner & Stone Eq. 51) - unchanged
    real ct_contact_formula(
        const vector_t<real, 4>& face_e_fields,
        const vector_t<real, 4>& cell_e_fields,
        const vector_t<real, 4>& density_fluxes
    )
    {
        // Face E-fields: North, South, East, West
        const auto [en, es, ee, ew] = face_e_fields;

        // Cell E-fields: NorthEast, NorthWest, SouthEast, SouthWest
        const auto [ene, enw, ese, esw] = cell_e_fields;

        // Density fluxes: North, South, East, West
        const auto [fn, fs, fe, fw] = density_fluxes;

        // Average of face-centered electric fields
        const real eavg = static_cast<real>(0.25) * (es + en + ew + ee);
        constexpr real one_eighth = static_cast<real>(0.125);

        // Compute gradients with upwinding based on density flux directions

        // West side gradient
        const real de_dqjL = [&] {
            if (fw > 0.0) {
                return static_cast<real>(2.0) * (es - esw);
            }
            else if (fw < 0.0) {
                return static_cast<real>(2.0) * (en - enw);
            }
            return es - esw + en - enw;
        }();

        // East side gradient
        const real de_dqjR = [&] {
            if (fe > 0.0) {
                return static_cast<real>(2.0) * (ese - es);
            }
            else if (fe < 0.0) {
                return static_cast<real>(2.0) * (ene - en);
            }
            return ese - es + ene - en;
        }();

        // South side gradient
        const real de_dqkL = [&] {
            if (fs > 0.0) {
                return static_cast<real>(2.0) * (ew - esw);
            }
            else if (fs < 0.0) {
                return static_cast<real>(2.0) * (ee - ese);
            }
            return ew - esw + ee - ese;
        }();

        // North side gradient
        const real de_dqkR = [&] {
            if (fn > 0.0) {
                return static_cast<real>(2.0) * (enw - ew);
            }
            else if (fn < 0.0) {
                return static_cast<real>(2.0) * (ene - ee);
            }
            return enw - ew + ene - ee;
        }();

        // Final CT Contact formula (Gardiner & Stone Eq. 51)
        return eavg + one_eighth * (de_dqjL - de_dqjR + de_dqkL - de_dqkR);
    }
}   // namespace simbi::em
