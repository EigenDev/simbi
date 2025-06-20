/**
 *  *=============================================================================
 *  *           SIMBI - Special Relativistic Magnetohydrodynamics Code
 *  *=============================================================================
 *  *
 *  * @file            vector_field.hpp
 *  * @brief           a class to represent a  left/rightvector field in cell
 *  * @details
 *  *
 *  * @version         0.8.0
 *  * @date            2025-02-26
 *  * @author          Marcus DuPont
 *  * @email           marcus.dupont@princeton.edu
 *  *
 *  *==============================================================================
 *  * @build           Requirements & Dependencies
 *  *==============================================================================
 *  * @requires        C++20
 *  * @depends         CUDA >= 11.0, HDF5 >= 1.12, OpenMP >= 4.5
 *  * @platform        Linux, MacOS
 *  * @parallel        GPU (CUDA, HIP), CPU (OpenMP)
 *  *
 *  *==============================================================================
 *  * @documentation   Reference & Notes
 *  *==============================================================================
 *  * @usage
 *  * @note
 *  * @warning
 *  * @todo
 *  * @bug
 *  * @performance
 *  *
 *  *==============================================================================
 *  * @testing        Quality Assurance
 *  *==============================================================================
 *  * @test
 *  * @benchmark
 *  * @validation
 *  *
 *  *==============================================================================
 *  * @history        Version History
 *  *==============================================================================
 *  * 2025-02-26      v0.8.0      Initial implementation
 *  *
 *  *==============================================================================
 *  * @copyright (C) 2025 Marcus DuPont. All rights reserved.
 *  *==============================================================================
 */
#ifndef VECTOR_FIELD_HPP
#define VECTOR_FIELD_HPP

#include "config.hpp"
#include "core/containers/vector.hpp"

namespace simbi {
    namespace vector_field {
        // a class to represent a vector field
        // that will be used to do vector calculus
        // operations
        template <typename T, size_type Dims>
        class vector_field_t
        {
          protected:
            // the vector field
            general_vector_t<T, Dims> left_field;
            general_vector_t<T, Dims> right_field;

          public:
            // constructor
            DEV vector_field_t(
                const general_vector_t<T, Dims>& left,
                const general_vector_t<T, Dims>& right
            )
                : left_field(left), right_field(right)
            {
            }

            // get the left field
            DEV const general_vector_t<T, Dims>& left() const
            {
                return left_field;
            }

            // get the right field
            DEV const general_vector_t<T, Dims>& right() const
            {
                return right_field;
            }
        };
    }   // namespace vector_field

}   // namespace simbi

#endif
