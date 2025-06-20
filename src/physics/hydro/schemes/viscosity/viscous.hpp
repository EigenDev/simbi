/**
 *=============================================================================
 *           SIMBI - Special Relativistic Magnetohydrodynamics Code
 *=============================================================================
 *
 * @file            viscous.hpp
 * @brief           implementation of the viscous stress tensor
 * @details
 *
 * @version         0.8.0
 * @date            2025-05-16
 * @author          Marcus DuPont
 * @email           marcus.dupont@princeton.edu
 *
 *==============================================================================
 * @build           Requirements & Dependencies
 *==============================================================================
 * @requires        C++20
 * @depends         CUDA >= 11.0, HDF5 >= 1.12, OpenMP >= 4.5
 * @platform        Linux, MacOS
 * @parallel        GPU (CUDA, HIP), CPU (OpenMP)
 *
 *==============================================================================
 * @documentation   Reference & Notes
 *==============================================================================
 * @usage
 * @note
 * @warning
 * @todo
 * @bug
 * @performance
 *
 *==============================================================================
 * @testing        Quality Assurance
 *==============================================================================
 * @test
 * @benchmark
 * @validation
 *
 *==============================================================================
 * @history        Version History
 *==============================================================================
 * 2025-05-16      v0.8.0      Initial implementation
 *
 *==============================================================================
 * @copyright (C) 2025 Marcus DuPont. All rights reserved.
 *==============================================================================
 */
#ifndef VISCOUS_HPP
#define VISCOUS_HPP

#include "config.hpp"
#include "core/containers/array.hpp"
#include "core/containers/vector.hpp"
#include "geometry/mesh/cell.hpp"
#include "geometry/vector_calculus.hpp"
#include "util/tools/helpers.hpp"

namespace simbi::visc {
    template <size_type Dir, typename primitive_type, size_type Dims>
    DEV std::pair<
        typename primitive_type::counterpart_t,
        typename primitive_type::counterpart_t>
    viscous_flux(
        const primitive_type& px1L,
        const primitive_type& px1R,
        const primitive_type& px2L,
        const primitive_type& px2R,
        const primitive_type& px3L,
        const primitive_type& px3R,
        const Cell<Dims>& cell,
        real nu
    )
    {
        using conserved_t = typename primitive_type::counterpart_t;
        conserved_t left_viscous_flux{};
        conserved_t right_viscous_flux{};

        if (helpers::goes_to_zero(nu)) {
            return {left_viscous_flux, right_viscous_flux};
        }

        // get velocities along x1 direction
        const real v1x1L = px1L.vcomponent(1);
        const real v1x1R = px1R.vcomponent(1);
        const real v2x1L = px1L.vcomponent(2);
        const real v2x1R = px1R.vcomponent(2);
        const real v3x1L = px1L.vcomponent(3);
        const real v3x1R = px1R.vcomponent(3);

        // x2 direction
        const real v1x2L = px2L.vcomponent(1);
        const real v1x2R = px2R.vcomponent(1);
        const real v2x2L = px2L.vcomponent(2);
        const real v2x2R = px2R.vcomponent(2);
        const real v3x2L = px2L.vcomponent(3);
        const real v3x2R = px2R.vcomponent(3);

        // x3 direction
        const real v1x3L = px3L.vcomponent(1);
        const real v1x3R = px3R.vcomponent(1);
        const real v2x3L = px3L.vcomponent(2);
        const real v2x3R = px3R.vcomponent(2);
        const real v3x3L = px3L.vcomponent(3);
        const real v3x3R = px3R.vcomponent(3);

        // diagonal terms
        const real dv1_dx1 = (v1x1R - v1x1L) / cell.width(0);
        const real dv2_dx2 = (v2x2R - v2x2L) / cell.width(1);
        const real dv3_dx3 = (v3x3R - v3x3L) / cell.width(2);
        const real div_v   = dv1_dx1 + dv2_dx2 + dv3_dx3;

        // off diagonal terms
        const real dv2_dx1 = (v2x1R - v2x1L) / cell.width(0);
        const real dv3_dx1 = (v3x1R - v3x1L) / cell.width(0);

        const real dv1_dx2 = (v1x2R - v1x2L) / cell.width(1);
        const real dv3_dx2 = (v3x2R - v3x2L) / cell.width(1);

        const real dv1_dx3 = (v1x3R - v1x3L) / cell.width(2);
        const real dv2_dx3 = (v2x3R - v2x3L) / cell.width(2);

        // calc the stress tensor components for this direction
        if constexpr (Dir == 1) {   // X1 direction
            // calc dynamic viscosity ($\mu = \rho * \nu$)
            real rhoL = px1L.rho();
            real rhoR = px1R.rho();
            real muL  = rhoL * nu;
            real muR  = rhoR * nu;

            // diag component $\sigma_{11} = 2\mu (\partial v_1/\partial x_1
            // - (1/3) \nabla \cdot v)$
            real sigma_11 = 2.0 * muL * (dv1_dx1 - div_v / 3.0);

            // off-diagonal components
            real sigma_12 = muL * (dv1_dx2 + dv2_dx1);

            // set momentum flux components
            left_viscous_flux.mcomponent(1) = sigma_11;
            left_viscous_flux.mcomponent(2) = sigma_12;
            if constexpr (Dims == 3) {
                real sigma_13                   = muL * (dv1_dx3 + dv3_dx1);
                left_viscous_flux.mcomponent(3) = sigma_13;
            }

            sigma_11 = 2.0 * muR * (dv1_dx1 - div_v / 3.0);

            // off-diagonal components
            sigma_12 = muR * (dv1_dx2 + dv2_dx1);

            // set momentum flux components
            right_viscous_flux.mcomponent(1) = sigma_11;
            right_viscous_flux.mcomponent(2) = sigma_12;
            if constexpr (Dims == 3) {
                real sigma_13                    = muR * (dv1_dx3 + dv3_dx1);
                right_viscous_flux.mcomponent(3) = sigma_13;
            }

            // calc interface velocity for energy flux
            // real vx1 = (face == 0) ? v1x1L : v1x1R;
            // real vx2 = (face == 0) ? v2x1L : v2x1R;
            // real vx3 = (face == 0) ? v3x1L : v3x1R;

            // energy flux: \vec{v} \cdot \sigma
            // TODO: implement
            // viscous_flux.nrg() =
            //     vx1 * sigma_11 + vx2 * sigma_12 + vx3 * sigma_13;
        }
        else if constexpr (Dir == 2) {
            // calc dynamic viscosity ($\mu = \rho * \nu$)
            real rhoL = px2L.rho();
            real rhoR = px2R.rho();
            real muL  = rhoL * nu;
            real muR  = rhoR * nu;

            // diag component $\sigma_{22} = 2\mu (\partial v_2/\partial x_2
            // -(1/3) \nabla \cdot v)$
            real sigma_22 = 2.0 * muL * (dv2_dx2 - div_v / 3.0);

            // off-diagonal components
            real sigma_21 = muL * (dv1_dx2 + dv2_dx1);

            // set momentum flux components
            left_viscous_flux.mcomponent(1) = sigma_21;
            left_viscous_flux.mcomponent(2) = sigma_22;
            if constexpr (Dims == 3) {
                real sigma_23                   = muL * (dv2_dx3 + dv3_dx2);
                left_viscous_flux.mcomponent(3) = sigma_23;
            }

            sigma_22 = 2.0 * muR * (dv2_dx2 - div_v / 3.0);

            // off-diagonal components
            sigma_21 = muR * (dv1_dx2 + dv2_dx1);

            // set momentum flux components
            right_viscous_flux.mcomponent(1) = sigma_21;
            right_viscous_flux.mcomponent(2) = sigma_22;
            if constexpr (Dims == 3) {
                real sigma_23                    = muR * (dv2_dx3 + dv3_dx2);
                right_viscous_flux.mcomponent(3) = sigma_23;
            }
        }
        else {
            static_assert(
                Dims == 3,
                "must be 3D run for full access to the viscous stress tensor"
            );
            // calc dynamic viscosity ($\mu = \rho * \nu$)
            real rhoL = px3L.rho();
            real rhoR = px3R.rho();
            real muL  = rhoL * nu;
            real muR  = rhoR * nu;

            // diag component $\sigma_{33} = 2\mu (\partial v_3/\partial x_3
            // (1/3) \nabla \cdot v)$
            real sigma_33 = 2.0 * muL * (dv3_dx3 - div_v / 3.0);

            // off-diagonal components
            real sigma_31 = muL * (dv1_dx3 + dv3_dx1);
            real sigma_32 = muL * (dv2_dx3 + dv3_dx2);

            // set momentum flux components
            left_viscous_flux.mcomponent(1) = sigma_31;
            left_viscous_flux.mcomponent(2) = sigma_32;
            left_viscous_flux.mcomponent(3) = sigma_33;

            sigma_33 = 2.0 * muR * (dv3_dx3 - div_v / 3.0);

            // off-diagonal components
            sigma_31 = muR * (dv1_dx3 + dv3_dx1);
            sigma_32 = muR * (dv2_dx3 + dv3_dx2);

            // set momentum flux components
            right_viscous_flux.mcomponent(1) = sigma_31;
            right_viscous_flux.mcomponent(2) = sigma_32;
            right_viscous_flux.mcomponent(3) = sigma_33;
        }

        return {left_viscous_flux, right_viscous_flux};
    }

    template <typename PolicyType, typename MeshType>
    real get_minimum_viscous_time(
        const PolicyType& policy,
        const MeshType& mesh,
        real visc
    )
    {
        // get the minimum viscous time from the simulation state
        const real min_viscous_time = std::numeric_limits<real>::infinity();
        return fp::reduce(
            policy,
            min_viscous_time,
            [mesh, visc] DEV(real acc, luint idx) -> real {
                auto cell  = mesh.get_cell_from_global(idx);
                auto dx1   = cell.width(0);
                auto dx2   = cell.width(1);
                auto dx3   = cell.width(2);
                auto tvisc = my_min3(dx1 * dx1, dx2 * dx2, dx3 * dx3) / visc;
                return my_min(acc, tvisc);
            }
        );
    }

}   // namespace simbi::visc

#endif
