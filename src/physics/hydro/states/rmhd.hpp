/**
 *  *=============================================================================
 *  *           SIMBI - Special Relativistic Magnetohydrodynamics Code
 *  *=============================================================================
 *  *
 *  * @file            rmhd.hpp
 *  * @brief           Special Relativistic Magnetohydrodynamics
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

#ifndef RMHD_HPP
#define RMHD_HPP

#include "base.hpp"                            // for HydroBase
#include "build_options.hpp"                   // for real, HD, lint, luint
#include "core/types/containers/ndarray.hpp"   // for ndarray
#include "core/types/monad/maybe.hpp"          // for Maybe
#include "core/types/utility/enums.hpp"        // for TIMESTEP_TYPE
#include "geometry/mesh/mesh.hpp"              // for Mesh
#include "physics/hydro/schemes/ct/ct_calculator.hpp"   // for anyPrimitive
#include "physics/hydro/types/generic_structs.hpp"   // for Eigenvals, mag_four_vec
#include "util/parallel/exec_policy.hpp"             // for ExecutionPolicy
#include "util/tools/helpers.hpp"                    // for my_min, my_max, ...
#include <functional>                                // for function
#include <optional>                                  // for optional
#include <type_traits>                               // for conditional_t
#include <vector>                                    // for vector

namespace simbi {
    template <int dim>
    class RMHD : public HydroBase<RMHD<dim>, dim, Regime::RMHD>
    {
      private:
        using base_t = HydroBase<RMHD<dim>, dim, Regime::RMHD>;

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
        static constexpr std::string_view regime = "srmhd";

        // set the primitive and conservative types at compile time
        // using primitive_t = anyPrimitive<dim, Regime::RMHD>;
        // using conserved_t = anyConserved<dim, Regime::RMHD>;
        // using eigenvals_t = Eigenvals<dim, Regime::RMHD>;
        // using function_t  = typename helpers::real_func<dim>::type;
        using ct_scheme_t = std::conditional_t<
            comp_ct_type == CTTYPE::MdZ,
            ct::CTMdZ,
            ct::CTContact>;

        // hydrodynamic source functions
        function_t hydro_source;

        // gravity source functions
        function_t gravity_source;

        // boundary source functions at x1 boundaries
        function_t bx1_inner_source;
        function_t bx1_outer_source;
        // boundary source functions at x2 boundaries
        function_t bx2_inner_source;
        function_t bx2_outer_source;
        // boundary source functions at x3 boundaries
        function_t bx3_inner_source;
        function_t bx3_outer_source;

        template <typename T>
        using RiemannFuncPointer = conserved_t (T::*)(
            const primitive_t&,
            const primitive_t&,
            const luint,
            const real,
            const real
        ) const;
        RiemannFuncPointer<RMHD<dim>> riemann_solve;

        std::vector<function_t> bsources;   // boundary sources
        std::vector<function_t> hsources;   // hydro sources
        std::vector<function_t> gsources;   // gravity sources

        /* Shared Data Members */
        ndarray<conserved_t, dim> fri, gri, hri;
        ndarray<real, dim> bstag1, bstag2, bstag3;

        RMHD();
        RMHD(
            std::vector<std::vector<real>>& state,
            InitialConditions& init_conditions
        );

        ~RMHD();

        /* Methods */

        DEV auto cons2prim_single(const auto& cons) const;
        void sync_flux_boundaries();
        void sync_magnetic_boundaries(const auto& bfield_man);
        void riemann_fluxes();
        void advance_conserved();
        void advance_magnetic_fields();

        template <int nhat>
        void update_magnetic_component(const ExecutionPolicy<>& policy);

        DUAL auto
        calc_max_wave_speeds(const auto& prims, const luint nhat) const;

        DUAL eigenvals_t calc_eigenvals(
            const auto& primsL,
            const auto& primsR,
            const luint nhat
        ) const;

        DUAL conserved_t calc_hlle_flux(
            const auto& prL,
            const auto& prR,
            const luint nhat,
            const real vface,
            const real bface
        ) const;

        DUAL conserved_t calc_hllc_flux(
            const auto& prL,
            const auto& prR,
            const luint nhat,
            const real vface,
            const real bface
        ) const;

        DUAL conserved_t calc_hlld_flux(
            const auto& prL,
            const auto& prR,
            const luint nhat,
            const real vface,
            const real bface
        ) const;

        DUAL real div_b(
            const auto b1L,
            const auto b1R,
            const auto b2L,
            const auto b2R,
            const auto b3L,
            const auto b3R,
            const auto& cell
        ) const;

        DUAL void set_riemann_solver()
        {
            switch (this->solver_type()) {
                case Solver::HLLE:
                    this->riemann_solve = &RMHD<dim>::calc_hlle_flux;
                    break;
                case Solver::HLLC:
                    this->riemann_solve = &RMHD<dim>::calc_hllc_flux;
                    break;
                default:
                    this->riemann_solve = &RMHD<dim>::calc_hlld_flux;
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
            bstag1.sync_to_device();
            bstag2.sync_to_device();
            bstag3.sync_to_device();
            fri.sync_to_device();
            gri.sync_to_device();
            hri.sync_to_device();
        }

        DUAL real hlld_vdiff(
            const real p,
            const conserved_t r[2],
            const real lam[2],
            const real bn,
            const luint nhat,
            auto& praL,
            auto& praR,
            auto& prC
        ) const;

      public:
        void cons2prim_impl();
        void advance_impl();

        template <TIMESTEP_TYPE dt_type = TIMESTEP_TYPE::ADAPTIVE>
        DUAL auto get_wave_speeds(const Maybe<primitive_t>& prim) const
            -> WaveSpeeds
        {
            if constexpr (dt_type == TIMESTEP_TYPE::MINIMUM) {
                return WaveSpeeds{
                  .v1p = 1.0,
                  .v1m = 1.0,
                  .v2p = 1.0,
                  .v2m = 1.0,
                  .v3p = 1.0,
                  .v3m = 1.0
                };
            }
            const real cs = prim->sound_speed(gamma);
            const real v1 = prim->vcomponent(1);
            const real v2 = prim->vcomponent(2);
            const real v3 = prim->vcomponent(3);

            return WaveSpeeds{
              .v1p = std::abs(v1 + cs / (1 + v1 * cs)),
              .v1m = std::abs(v1 - cs / (1 - v1 * cs)),
              .v2p = std::abs(v2 + cs / (1 + v2 * cs)),
              .v2m = std::abs(v2 - cs / (1 - v2 * cs)),
              .v3p = std::abs(v3 + cs / (1 + v3 * cs)),
              .v3m = std::abs(v3 - cs / (1 - v3 * cs)),
            };
        }

        void
        run(const std::function<real(real)>& scale_factor,
            const std::function<real(real)>& scale_factor_derivative)
        {
            this->simulate(scale_factor, scale_factor_derivative);
        }
    };
}   // namespace simbi

template <>
struct is_relativistic_mhd<simbi::RMHD<1>> {

    static constexpr bool value = true;
};

template <>
struct is_relativistic_mhd<simbi::RMHD<2>> {
    static constexpr bool value = true;
};

template <>
struct is_relativistic_mhd<simbi::RMHD<3>> {
    static constexpr bool value = true;
};

#include "rmhd.ipp"
#endif