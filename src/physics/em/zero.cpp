#include "zero.hpp"
#include "config.hpp"
#include "containers/vector.hpp"

namespace simbi::em {
    DEV real ct_zero_formula(
        const vector_t<real, 4>& face_e_fields,
        const vector_t<real, 4>& cell_e_fields
    )
    {
        return 0.5 * (face_e_fields[0] + face_e_fields[1] + face_e_fields[2] +
                      face_e_fields[3]) -
               0.25 * (cell_e_fields[0] + cell_e_fields[1] + cell_e_fields[2] +
                       cell_e_fields[3]);
    }
}   // namespace simbi::em
