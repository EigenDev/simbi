/**
 *  *=============================================================================
 *  *           SIMBI - Special Relativistic Magnetohydrodynamics Code
 *  *=============================================================================
 *  *
 *  * @file            srhd.hpp
 *  * @brief           Special Relativistic Hydrodynamics
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
#ifndef SRHD_HPP
#define SRHD_HPP

#include "base.hpp"                          // for HydroBase
#include "config.hpp"                        // for real, HD, lint, luint
#include "core/containers/ndarray.hpp"       // for ndarray
#include "core/functional/monad/maybe.hpp"   // for Maybe
#include "core/utility/enums.hpp"            // for TIMESTEP_TYPE
#include "physics/hydro/types/generic_structs.hpp"   // for Eigenvals, mag_four_vec
#include "util/tools/helpers.hpp"                    // for my_min, my_max, ...
#include <functional>                                // for function
#include <type_traits>                               // for conditional_t
#include <vector>                                    // for vector

namespace simbi {
    template <int dim>
    class SRHD : public HydroBase<SRHD<dim>, dim, Regime::SRHD>
    {
      private:
        using base_t = HydroBase<SRHD<dim>, dim, Regime::SRHD>;
        ndarray_t<real, dim> pressure_guesses_;

      protected:
        // type alias
        using base_t::gamma;

      public:
        using typename base_t::conserved_t;
        using typename base_t::eigenvals_t;
        using typename base_t::function_t;
        using typename base_t::primitive_t;

        static constexpr int dimensions          = dim;
        static constexpr int nvars               = dim + 3;
        static constexpr std::string_view regime = "srhd";

        template <typename T>
        using RiemannFuncPointer = conserved_t (T::*)(
            const primitive_t&,
            const primitive_t&,
            const luint,
            const real
        ) const;
        RiemannFuncPointer<SRHD<dim>> riemann_solve;

        /* Methods */
        SRHD();
        SRHD(
            auto&& init_cons,
            auto&& init_prim,
            InitialConditions& init_conditions
        );
        ~SRHD();

        DUAL eigenvals_t calc_eigenvals(
            const auto& primsL,
            const auto& primsR,
            const luint nhat
        ) const;

        DUAL conserved_t calc_hllc_flux(
            const auto& prL,
            const auto& prR,
            const luint nhat,
            const real vface = 0.0
        ) const;

        DUAL conserved_t calc_hlle_flux(
            const auto& prL,
            const auto& prR,
            const luint nhat,
            const real vface = 0.0
        ) const;

        DUAL void set_riemann_solver()
        {
            switch (this->solver_type()) {
                case Solver::HLLE:
                    this->riemann_solve = &SRHD<dim>::calc_hlle_flux;
                    break;
                default:
                    this->riemann_solve = &SRHD<dim>::calc_hllc_flux;
                    break;
            }
        }

        void init_riemann_solver()
        {
            SINGLE(helpers::hybrid_set_riemann_solver, this);
        }

        void init_simulation();

        void sync_all_to_device()
        {
            this->sync_to_device();
            pressure_guesses_.sync_to_device();
        }

        DUAL auto calc_star_state(
            const auto& uL,
            const auto& uR,
            const auto& fL,
            const auto& fR,
            const real aL,
            const real aR,
            const luint nhat
        ) const -> std::pair<real, real>;

      private:
        DUAL conserved_t compute_star_state(
            const auto& prim,
            const auto& cons,
            const real a,
            const real aStar,
            const real pStar,
            const luint nhat
        ) const
        {
            const auto& mom     = cons.momentum();
            const auto vnorm    = prim.proper_velocity(nhat);
            const auto& p       = prim.press();
            const auto& d       = cons.dens();
            const auto& chi     = cons.chi();
            const auto e        = cons.nrg() + d;
            const auto cofactor = 1.0 / (a - aStar);
            const auto kdvec    = unit_vectors::canonical_basis<dim>(nhat);

            const auto dStar   = cofactor * (a - vnorm) * d;
            const auto chiStar = cofactor * (a - vnorm) * chi;
            const auto momStar =
                (mom * (a - vnorm) + kdvec * (-p + pStar)) * cofactor;
            const auto eStar =
                cofactor * (e * (a - vnorm) + pStar * aStar - p * vnorm);

            return conserved_t{dStar, momStar, eStar - dStar, chiStar};
        }

        DUAL conserved_t apply_hllc(
            const auto& star_state,
            const auto& flux,
            const auto& cons,
            const real a,
            const real vface,
            const auto& prL,
            const auto& prR
        ) const
        {
            auto hllc_flux =
                flux + (star_state - cons) * a - star_state * vface;

            // Upwind concentration
            hllc_flux.chi() = (hllc_flux.dens() < 0.0)
                                  ? prR.chi() * hllc_flux.dens()
                                  : prL.chi() * hllc_flux.dens();
            return hllc_flux;
        }

      public:
        template <TIMESTEP_TYPE dt_type = TIMESTEP_TYPE::ADAPTIVE>
        DUAL auto get_wave_speeds(const Maybe<primitive_t>& prim) const
            -> wave_speeds_t
        {
            // if constexpr (dt_type == TIMESTEP_TYPE::MINIMUM) {
            //     return wave_speeds_t{
            //       .v1p = 1.0,
            //       .v1m = 1.0,
            //       .v2p = 1.0,
            //       .v2m = 1.0,
            //       .v3p = 1.0,
            //       .v3m = 1.0
            //     };
            // }
            const real cs = prim->sound_speed(gamma);
            const real v1 = prim->vcomponent(1);
            const real v2 = prim->vcomponent(2);
            const real v3 = prim->vcomponent(3);

            return wave_speeds_t{
              std::abs((v1 + cs) / (1 + v1 * cs)),
              std::abs((v1 - cs) / (1 - v1 * cs)),
              std::abs((v2 + cs) / (1 + v2 * cs)),
              std::abs((v2 - cs) / (1 - v2 * cs)),
              std::abs((v3 + cs) / (1 + v3 * cs)),
              std::abs((v3 - cs) / (1 - v3 * cs)),
            };
        }

        void cons2prim_impl();
        void advance_impl();

        void
        run(const std::function<real(real)>& scale_factor,
            const std::function<real(real)>& scale_factor_derivative)
        {
            this->simulate(scale_factor, scale_factor_derivative);
        }
    };
}   // namespace simbi

template <>
struct is_relativistic<simbi::SRHD<1>> {

    static constexpr bool value = true;
};

template <>
struct is_relativistic<simbi::SRHD<2>> {
    static constexpr bool value = true;
};

template <>
struct is_relativistic<simbi::SRHD<3>> {
    static constexpr bool value = true;
};

#include "srhd.ipp"
#endif
