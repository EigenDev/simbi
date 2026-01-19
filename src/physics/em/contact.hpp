// =============================================================================
// contact.hpp
//
// constrained transport (ct) contact algorithm.
// implements the ct contact algorithm from gardiner & stone (2005) for
// computing electric fields at cell edges, which is necessary for preserving
// the divergence-free constraint of the magnetic field in mhd simulations.
//
// usage:
//   real ex = ct_contact_formula(face_fields, cell_fields, density_fluxes);
// =============================================================================
#pragma once

#include "build_config.hpp"
#include "containers/vector.hpp"
#include "decorators.hpp"

namespace simbi::em {
    // constrained transport contact algorithm developed by Gardiner & Stone
    // https://ui.adsabs.harvard.edu/abs/2005JCoPh.205..509G/abstract

    // CT Contact formula (Gardiner & Stone Eq. 51)
    DEV real ct_contact_formula(
        const vector_t<real, 4>& face_e_fields,
        const vector_t<real, 4>& cell_e_fields,
        const vector_t<real, 4>& density_fluxes
    );
} // namespace simbi::em
