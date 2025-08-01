#ifndef CT_CONTACT_HPP
#define CT_CONTACT_HPP

#include "config.hpp"
#include "containers/vector.hpp"

namespace simbi::em {
    // constrained transport contact algorithm developed by Gardiner & Stone
    // https://ui.adsabs.harvard.edu/abs/2005JCoPh.205..509G/abstract

    // CT Contact formula (Gardiner & Stone Eq. 51)
    DEV real ct_contact_formula(
        const vector_t<real, 4>& face_e_fields,
        const vector_t<real, 4>& cell_e_fields,
        const vector_t<real, 4>& density_fluxes
    );
}   // namespace simbi::em
#endif   // CT_CONTACT_HPP
