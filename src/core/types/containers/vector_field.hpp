#ifndef VECTOR_FIELD_HPP
#define VECTOR_FIELD_HPP

#include "build_options.hpp"
#include "core/types/containers/vector.hpp"

namespace simbi {
    namespace vector_field {
        // a class to represent a vector field
        // that will be used to do vector calculus
        // operations
        template <typename T, size_type Dims>
        class VectorField
        {
          protected:
            // the vector field
            general_vector_t<T, Dims> left_field;
            general_vector_t<T, Dims> right_field;

          public:
            // constructor
            VectorField(
                const general_vector_t<T, Dims>& left,
                const general_vector_t<T, Dims>& right
            )
                : left_field(left), right_field(right)
            {
            }

            // get the left field
            const general_vector_t<T, Dims>& left() const { return left_field; }

            // get the right field
            const general_vector_t<T, Dims>& right() const
            {
                return right_field;
            }
        };
    }   // namespace vector_field

}   // namespace simbi

#endif