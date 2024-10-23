/**
 * ***********************(C) COPYRIGHT 2024 Marcus DuPont**********************
 * @file       newt.hpp
 * @brief      single header for 1, 2, adn 3D Newtonian calculations
 *
 * @note
 * @history:
 *   Version   Date            Author          Modification    Email
 *   V0.8.0    Dec-03-2023     Marcus DuPont                   md4469@nyu.edu
 *
 * @verbatim
 * ==============================================================================
 *
 * ==============================================================================
 * @endverbatim
 * ***********************(C) COPYRIGHT 2024 Marcus DuPont**********************
 */

#ifndef NEWT_HPP
#define NEWT_HPP

#include "base.hpp"                   // for HydroBase
#include "build_options.hpp"          // for real, DUAL, lint, luint
#include "common/helpers.hpp"         // for my_min, my_max, ...
#include "common/hydro_structs.hpp"   // for Conserved, Primitive
#include "util/exec_policy.hpp"       // for ExecutionPolicy
#include "util/functional.hpp"        // for Function
#include "util/ndarray.hpp"           // for ndarray
#include <functional>                 // for function
#include <optional>                   // for optional
#include <type_traits>                // for conditional_t
#include <vector>                     // for vector

namespace simbi {
    template <int dim>
    struct Newtonian : public HydroBase {
        constexpr static int dimensions          = dim;
        constexpr static int nvars               = dim + 3;
        constexpr static std::string_view regime = "euler";
        // set the primitive and conservative types at compile time
        using primitive_t = anyPrimitive<dim, Regime::NEWTONIAN>;
        using conserved_t = anyConserved<dim, Regime::NEWTONIAN>;
        using eigenvals_t = Eigenvals<dim, Regime::NEWTONIAN>;
        using function_t  = typename helpers::real_func<dim>::type;
        template <typename T>
        using RiemannFuncPointer = conserved_t (T::*)(
            const primitive_t&,
            const primitive_t&,
            const luint,
            const real
        ) const;
        RiemannFuncPointer<Newtonian<dim>> riemann_solve;

        // boundary condition functions for mesh motion
        std::vector<function_t> bsources;   // boundary sources
        std::vector<function_t> hsources;   // hydro sources
        std::vector<function_t> gsources;   // gravity sources

        /* Shared Data Members */
        ndarray<primitive_t> prims;
        ndarray<conserved_t> cons;
        ndarray<real> dt_min;
        bool scalar_all_zeros;

        /* Methods */
        Newtonian();
        Newtonian(
            std::vector<std::vector<real>>& state,
            const InitialConditions& init_conditions
        );
        ~Newtonian();

        void cons2prim();
        void advance();

        DUAL eigenvals_t calc_eigenvals(
            const primitive_t& primsL,
            const primitive_t& primsR,
            const luint nhat
        ) const;

        DUAL conserved_t calc_hllc_flux(
            const primitive_t& prL,
            const primitive_t& prR,
            const luint nhat,
            const real vface = 0.0
        ) const;

        DUAL conserved_t calc_hlle_flux(
            const primitive_t& prL,
            const primitive_t& prR,
            const luint nhat,
            const real vface = 0.0
        ) const;

        DUAL void set_riemann_solver()
        {
            switch (sim_solver) {
                case Solver::HLLE:
                    this->riemann_solve = &Newtonian<dim>::calc_hlle_flux;
                    break;
                default:
                    this->riemann_solve = &Newtonian<dim>::calc_hllc_flux;
                    break;
            }
        }

        void set_the_riemann_solver()
        {
            SINGLE(helpers::hybrid_set_riemann_solver, this);
        }

        void adapt_dt();
        void adapt_dt(const ExecutionPolicy<>& p);

        void simulate(
            std::function<real(real)> const& a,
            std::function<real(real)> const& adot,
            const std::vector<std::optional<function_t>>& boundary_sources,
            const std::vector<std::optional<function_t>>& hydro_sources,
            const std::vector<std::optional<function_t>>& gravity_sources
        );

        DUAL constexpr real get_x1face(const lint ii, const int side) const;

        DUAL constexpr real get_x2face(const lint ii, const int side) const;

        DUAL constexpr real get_x3face(const lint ii, const int side) const;

        DUAL constexpr real get_x1_differential(const lint ii) const;

        DUAL constexpr real get_x2_differential(const lint ii) const;

        DUAL constexpr real get_x3_differential(const lint ii) const;

        DUAL real get_cell_volume(
            const lint ii,
            const lint jj = 0,
            const lint kk = 0
        ) const;

        void emit_troubled_cells() const;

        void offload()
        {
            cons.copyToGpu();
            prims.copyToGpu();
            dt_min.copyToGpu();
            if constexpr (dim > 1) {
                object_pos.copyToGpu();
            }
            bcs.copyToGpu();
            troubled_cells.copyToGpu();
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