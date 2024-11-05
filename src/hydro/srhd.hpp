/**
 * ***********************(C) COPYRIGHT 2024 Marcus DuPont**********************
 * @file       srhd.hpp
 * @brief      single header for 1,2, and 3D SRHD calculations
 *
 * @note
 * @history:
 *   Version   Date            Author          Modification    Email
 *   V0.8.0    Dec-03-2023     Marcus DuPont marcus.dupont@princeton.edu
 *
 * @verbatim
 * ==============================================================================
 *
 * ==============================================================================
 * @endverbatim
 * ***********************(C) COPYRIGHT 2024 Marcus DuPont**********************
 */
#ifndef SRHD_HPP
#define SRHD_HPP

#include "base.hpp"                   // for HydroBase
#include "build_options.hpp"          // for real, HD, lint, luint
#include "common/enums.hpp"           // for TIMESTEP_TYPE
#include "common/helpers.hpp"         // for my_min, my_max, ...
#include "common/hydro_structs.hpp"   // for Conserved, Primitive
#include "common/mesh.hpp"            // for Mesh
#include "util/exec_policy.hpp"       // for ExecutionPolicy
#include "util/ndarray.hpp"           // for ndarray
#include <functional>                 // for function
#include <optional>                   // for optional
#include <type_traits>                // for conditional_t
#include <vector>                     // for vector

namespace simbi {
    template <int dim>
    struct SRHD : public HydroBase,
                  public Mesh<
                      SRHD<dim>,
                      dim,
                      anyConserved<dim, Regime::SRHD>,
                      anyPrimitive<dim, Regime::SRHD>> {
        static constexpr int dimensions          = dim;
        static constexpr int nvars               = dim + 3;
        static constexpr std::string_view regime = "srhd";
        // set the primitive and conservative types at compile time
        using primitive_t = anyPrimitive<dim, Regime::SRHD>;
        using conserved_t = anyConserved<dim, Regime::SRHD>;
        using eigenvals_t = Eigenvals<dim, Regime::SRHD>;
        using function_t  = typename helpers::real_func<dim>::type;
        template <typename T>
        using RiemannFuncPointer = conserved_t (T::*)(
            const primitive_t&,
            const primitive_t&,
            const luint,
            const real
        ) const;
        RiemannFuncPointer<SRHD<dim>> riemann_solve;

        ndarray<function_t> bsources;   // boundary sources
        ndarray<function_t> hsources;   // hydro sources
        ndarray<function_t> gsources;   // gravity sources

        /* Shared Data Members */
        ndarray<primitive_t> prims;
        ndarray<conserved_t> cons;
        ndarray<real> pressure_guess, dt_min;

        /* Methods */
        SRHD();
        SRHD(
            std::vector<std::vector<real>>& state,
            const InitialConditions& init_conditions
        );
        ~SRHD();

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

        template <TIMESTEP_TYPE dt_type = TIMESTEP_TYPE::ADAPTIVE>
        void adapt_dt();

        template <TIMESTEP_TYPE dt_type = TIMESTEP_TYPE::ADAPTIVE>
        void adapt_dt(const ExecutionPolicy<>& p);

        void simulate(
            std::function<real(real)> const& a,
            std::function<real(real)> const& adot,
            const std::vector<std::optional<function_t>>& boundary_sources,
            const std::vector<std::optional<function_t>>& hydro_sources,
            const std::vector<std::optional<function_t>>& gravity_sources
        );

        void offload()
        {
            cons.copyToGpu();
            prims.copyToGpu();
            pressure_guess.copyToGpu();
            dt_min.copyToGpu();
            if constexpr (dim > 1) {
                object_pos.copyToGpu();
            }
            bcs.copyToGpu();
        }

        DUAL conserved_t hydro_sources(const auto& cell) const;

        DUAL conserved_t
        gravity_sources(const primitive_t& prims, const auto& cell) const;

        void emit_troubled_cells() const;
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