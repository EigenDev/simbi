// =============================================================================
// zero.hpp
//
// constrained transport (ct) "zero" scheme.
// implements the ct "zero" algorithm from gardiner & stone (2005), a simpler
// method for computing edge-centered electric fields for divergence-free
// mhd, compared to the more complex contact algorithm.
//
// usage:
//   real ex = ct_zero_formula(face_fields, cell_fields);
// =============================================================================
#pragma once

#include "build_config.hpp"
#include "containers/vector.hpp"
#include "decorators.hpp"

namespace simbi::em {
    // the Constrained Transport "Zero" scheme
    // described in section 3.2, Eqn. (40)
    // of Gardiner & Stone (2005)

    // CT Zero formula (Gardiner & Stone Eq. 51)
    DEV real
    ct_zero_formula(const vector_t<real, 4>& face_e_fields, const vector_t<real, 4>& cell_e_fields);
} // namespace simbi::em
