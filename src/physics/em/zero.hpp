// =============================================================================
// zero.hpp
//
// [TODO: Add description of what this file does]
//
// usage:
//   [TODO: Add usage example]
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

