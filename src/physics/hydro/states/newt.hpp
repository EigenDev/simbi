/**
 *  *=============================================================================
 *  *           SIMBI - Special Relativistic Magnetohydrodynamics Code
 *  *=============================================================================
 *  *
 *  * @file            newt.hpp
 *  * @brief           Newtonian Hydrodynamics
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

#ifndef NEWT_HPP
#define NEWT_HPP

#include "base.hpp"                            // for HydroBase
#include "build_options.hpp"                   // for real, DUAL, lint, luint
#include "core/types/containers/ndarray.hpp"   // for ndarray
#include "core/types/monad/maybe.hpp"          // for Maybe
#include "geometry/mesh/mesh.hpp"              // for Mesh
#include "physics/hydro/types/generic_structs.hpp"   // for Eigenvals, mag_four_vec
#include "util/parallel/exec_policy.hpp"             // for ExecutionPolicy
#include "util/tools/helpers.hpp"                    // for my_min, my_max, ...
#include <functional>                                // for function
#include <optional>                                  // for optional
#include <type_traits>                               // for conditional_t
#include <vector>                                    // for vector

namespace simbi {
    template <int dim>
    class Newtonian : public HydroBase<Newtonian<dim>, dim, Regime::NEWTONIAN>
    {
      private:
        using base_t = HydroBase<Newtonian<dim>, dim, Regime::NEWTONIAN>;

        // isothermal EOS
        bool isothermal_, locally_isothermal_, shakura_sunyaev_alpha_;
        real sound_speed_squared_;

      protected:
        // type aliases
        using base_t::gamma;

      public:
        using typename base_t::conserved_t;
        using typename base_t::eigenvals_t;
        using typename base_t::function_t;
        using typename base_t::primitive_t;

        static constexpr int dimensions          = dim;
        static constexpr int nvars               = dim + 3;
        static constexpr std::string_view regime = "classical";

        template <typename T>
        using RiemannFuncPointer =
            conserved_t (T::*)(const primitive_t&, const primitive_t&, const luint, const real, const conserved_t&, const conserved_t&) const;
        RiemannFuncPointer<Newtonian<dim>> riemann_solve;

        // Constructors
        Newtonian();

        // Overloaded Constructor
        Newtonian(
            std::vector<std::vector<real>>& state,
            InitialConditions& init_conditions
        );

        // Destructor
        ~Newtonian();

        /* Methods */
        DUAL eigenvals_t calc_eigenvals(
            const auto& primsL,
            const auto& primsR,
            const luint nhat
        ) const;

        DUAL conserved_t calc_hllc_flux(
            const auto& prL,
            const auto& prR,
            const luint nhat,
            const real vface  = 0.0,
            const auto& viscL = {},
            const auto& viscR = {}
        ) const;

        DUAL conserved_t calc_hlle_flux(
            const auto& prL,
            const auto& prR,
            const luint nhat,
            const real vface  = 0.0,
            const auto& viscL = {},
            const auto& viscR = {}
        ) const;

        DUAL void set_riemann_solver()
        {
            switch (this->solver_type()) {
                case Solver::HLLE:
                    this->riemann_solve = &Newtonian<dim>::calc_hlle_flux;
                    break;
                default:
                    this->riemann_solve = &Newtonian<dim>::calc_hllc_flux;
                    break;
            }
        }

        void init_riemann_solver()
        {
            SINGLE(helpers::hybrid_set_riemann_solver, this);
        }

        void init_simulation();

      public:
        void cons2prim_impl();
        void advance_impl();

        DUAL auto get_wave_speeds(const Maybe<primitive_t>& prim) const
            -> WaveSpeeds
        {
            const real cs = prim->sound_speed(gamma);
            const real v1 = prim->vcomponent(1);
            const real v2 = prim->vcomponent(2);
            const real v3 = prim->vcomponent(3);

            return WaveSpeeds{
              std::abs(v1 + cs),
              std::abs(v1 - cs),
              std::abs(v2 + cs),
              std::abs(v2 - cs),
              std::abs(v3 + cs),
              std::abs(v3 - cs),
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
struct is_relativistic<simbi::Newtonian<1>> {
    static constexpr bool value = false;
};

template <>
struct is_relativistic<simbi::Newtonian<2>> {
    static constexpr bool value = false;
};

template <>
struct is_relativistic<simbi::Newtonian<3>> {
    static constexpr bool value = false;
};

#include "newt.ipp"
#endif
