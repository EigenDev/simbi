#ifndef STATE_VIEW_HPP
#define STATE_VIEW_HPP

#include "config.hpp"
#include "core/containers/array.hpp"
#include "core/containers/vector.hpp"
#include "core/index/global_index.hpp"
#include "core/memory/values/state_value.hpp"
#include "core/parallel/view.hpp"
#include "core/types/alias/alias.hpp"
#include "core/utility/enums.hpp"
#include "geometry/mesh/mesh.hpp"
#include "magnetic_view.hpp"
#include <cstdint>

namespace simbi::views {
    using namespace simbi::parallel;
    template <Regime R, size_type Dims>
        requires(R == Regime::NEWTONIAN || R == Regime::SRHD)
    struct primitive_view_t {
        using counterpart_t = typename values::conserved_value_t<R, Dims>;
        static constexpr size_type dimensions = Dims;
        static constexpr Regime regime        = R;
        const real& rho;
        const_spatial_vector_view_t<real, Dims> vel;
        const real& pre;
        const real& chi;

        explicit constexpr operator values::primitive_value_t<R, Dims>() const
        {
            return values::primitive_value_t<R, Dims>{
              rho,
              vel.to_vector(),
              pre,
              chi
            };
        }
    };

    template <Regime R, size_type Dims>
        requires(R == Regime::NEWTONIAN || R == Regime::SRHD)
    struct conserved_view_t {
        using counterpart_t = typename values::primitive_value_t<R, Dims>;
        static constexpr size_type dimensions = Dims;
        static constexpr Regime regime        = R;
        const real& den;
        const const_spatial_vector_view_t<real, Dims> mom;
        const real& nrg;
        const real& chi;

        explicit constexpr operator values::conserved_value_t<R, Dims>() const
        {
            return values::conserved_value_t<R, Dims>{
              den,
              mom.to_vector(),
              nrg,
              chi
            };
        }
    };

    template <Regime R, size_type Dims>
        requires(R == Regime::MHD || R == Regime::RMHD)
    struct mhd_primitive_view_t {
        using counterpart_t = typename values::mhd_conserved_value_t<R, Dims>;
        static constexpr size_type dimensions = Dims;
        static constexpr Regime regime        = R;
        const real& rho;
        const_spatial_vector_view_t<real, Dims> vel;
        const real& pre;
        const magnetic_vector_view_t<real, Dims> mag;
        const real& chi;

        explicit constexpr
        operator values::mhd_primitive_value_t<R, Dims>() const
        {
            return values::mhd_primitive_value_t<R, Dims>{
              rho,
              vel.to_vector(),
              pre,
              mag.to_vector(),
              chi
            };
        }
    };

    template <Regime R, size_type Dims>
        requires(R == Regime::MHD || R == Regime::RMHD)
    struct mhd_conserved_view_t {
        using counterpart_t = typename values::mhd_primitive_value_t<R, Dims>;
        static constexpr size_type dimensions = Dims;
        static constexpr Regime regime        = R;
        const real& den;
        const const_spatial_vector_view_t<real, Dims> mom;
        const real& nrg;
        const magnetic_vector_view_t<real, Dims> mag;
        const real& chi;

        explicit constexpr
        operator values::mhd_conserved_value_t<R, Dims>() const
        {
            return values::mhd_conserved_value_t<R, Dims>{
              den,
              mom.to_vector(),
              nrg,
              mag.to_vector(),
              chi
            };
        }
    };

    // template <Regime R, size_type Dims>
    // auto get_conserved_at(
    //     real* data,
    //     const array_t<size_type, Dims>& pos,
    //     size_type domain_size
    // )
    // {
    //     // convert multi-dimensional position to linear index
    //     size_type linear_pos = calculate_linear_position(pos);
    //     return get_conserved_at<R, Dims>(data, linear_pos, domain_size);
    // }

    // // for cell_index_t from index system
    // template <Regime R, size_type Dims>
    // auto get_conserved_at(
    //     real* data,
    //     const index::cell_index_t& idx,
    //     const Mesh<Dims>& mesh
    // )
    // {
    //     // Convert cell index to linear position
    //     size_type position =
    //     index::stagger_array_index<index::stagger_t::cell>(
    //         idx,
    //         mesh.dimensions()[0],
    //         mesh.dimensions()[1],
    //         mesh.dimensions()[2]
    //     );

    //     return get_conserved_at<R, Dims>(data, position, mesh.size());
    // }
}   // namespace simbi::views

#endif   // STATE_VIEW_HPP
