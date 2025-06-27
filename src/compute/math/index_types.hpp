#ifndef SIMBI_COORD_TYPES_HPP
#define SIMBI_COORD_TYPES_HPP

// #include "data/containers/vector.hpp"
#include <cstdint>

namespace simbi {

    template <typename T, std::uint64_t Dims>
    struct vector_t;

    template <std::uint64_t Dims>
    using uarray = vector_t<std::uint64_t, Dims>;

    template <std::uint64_t Dims>
    struct index_space_t;

    enum class index_semantic_t {
        cell,
        face_x,
        face_y,
        face_z,
        corner
    };

    template <
        std::uint64_t Dims,
        index_semantic_t Semantic = index_semantic_t::cell>
    class semantic_space_t : public index_space_t<Dims>
    {
        // interhits all index_space_t functionality
        // add semantic ype information and transform methods
    };

    template <std::uint64_t Dims>
    class semantic_space_t<Dims, index_semantic_t::face_x>
    {
        // facce indices to left/right indices
        auto left(const uarray<Dims>& /*face_indices*/) const
        {
            // return left face indices
        }

        auto right(const uarray<Dims>& /*face_indices*/) const
        {
            // return right face indices
        }
    };

}   // namespace simbi

#endif   // SIMBI_COORD_TYPES_HPP
