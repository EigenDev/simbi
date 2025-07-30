#ifndef SIMBI_CT_CONTACT_HPP
#define SIMBI_CT_CONTACT_HPP

#include "config.hpp"
#include "containers/vector.hpp"

namespace simbi::em {
    // Constrained transport contact algorithm developed by Gardiner & Stone
    // https://ui.adsabs.harvard.edu/abs/2005JCoPh.205..509G/abstract

    // ========================================================================
    // CT CONTACT EMF COMPUTATION
    // ========================================================================

    // CT Contact formula (Gardiner & Stone Eq. 51)
    DEV real ct_contact_formula(
        const vector_t<real, 4>& face_e_fields,
        const vector_t<real, 4>& cell_e_fields,
        const vector_t<real, 4>& density_fluxes
    );
}   // namespace simbi::em
#endif   // SIMBI_CT_CONTACT_HPP
