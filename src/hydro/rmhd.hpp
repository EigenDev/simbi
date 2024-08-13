/**
 * ***********************(C) COPYRIGHT 2024 Marcus DuPont**********************
 * @file       rmhd.hpp
 * @brief      Single header for 1, 2, and 3D RMHD calculations
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

#ifndef RMHD_HPP
#define RMHD_HPP

#include "base.hpp"                   // for HydroBase
#include "build_options.hpp"          // for real, HD, lint, luint
#include "common/enums.hpp"           // for TIMESTEP_TYPE
#include "common/helpers.hpp"         // for my_min, my_max, ...
#include "common/hydro_structs.hpp"   // for Conserved, Primitive
#include "util/exec_policy.hpp"       // for ExecutionPolicy
#include "util/ndarray.hpp"           // for ndarray
#include <functional>                 // for function
#include <optional>                   // for optional
#include <type_traits>                // for conditional_t
#include <vector>                     // for vector

namespace simbi {
    template <int dim>
    struct RMHD : public HydroBase {

        // set the primitive and conservative types at compile time
        using primitive_t     = rmhd::AnyPrimitive<dim>;
        using conserved_t     = rmhd::AnyConserved<dim>;
        using primitive_soa_t = rmhd::PrimitiveSOA;
        using eigenvals_t     = rmhd::Eigenvals;
        using mag_fourvec_t   = rmhd::mag_four_vec<dim>;
        using function_t      = typename std::conditional_t<
                 dim == 1,
                 std::function<real(real)>,
                 std::conditional_t<
                     dim == 2,
                     std::function<real(real, real)>,
                     std::function<real(real, real, real)>>>;
        template <typename T>
        using RiemannFuncPointer = conserved_t (T::*)(
            const primitive_t&,
            const primitive_t&,
            const luint,
            const real,
            const luint
        ) const;
        RiemannFuncPointer<RMHD<dim>> riemann_solve;

        function_t dens_outer;
        function_t mom1_outer;
        function_t mom2_outer;
        function_t mom3_outer;
        function_t enrg_outer;
        function_t mag1_outer;
        function_t mag2_outer;
        function_t mag3_outer;

        const static int dimensions = dim;

        /* Shared Data Members */
        ndarray<primitive_t> prims;
        ndarray<conserved_t> cons, outer_zones, inflow_zones;
        ndarray<real> edens_guess, dt_min, bstag1, bstag2, bstag3;
        bool scalar_all_zeros;
        luint nzone_edges;

        /* Methods */
        RMHD();
        RMHD(
            std::vector<std::vector<real>>& state,
            const InitialConditions& init_conditions
        );
        ~RMHD();

        void cons2prim(const ExecutionPolicy<>& p);

        /**
         * Return the primitive
         * variables density , three-velocity, pressure
         *
         * @param  con conserved array at index
         * @param gid  current global index
         * @return none
         */
        DUAL primitive_t cons2prim(const conserved_t& cons) const;

        void advance(const ExecutionPolicy<>& p);

        DUAL void calc_max_wave_speeds(
            const primitive_t& prims,
            const luint nhat,
            real speeds[],
            real& cs2
        ) const;

        DUAL eigenvals_t calc_eigenvals(
            const primitive_t& primsL,
            const primitive_t& primsR,
            const luint nhat
        ) const;

        DUAL conserved_t prims2cons(const primitive_t& prims) const;

        DUAL conserved_t calc_hlle_flux(
            const primitive_t& prL,
            const primitive_t& prR,
            const luint nhat,
            const real vface,
            const luint gid
        ) const;

        DUAL conserved_t calc_hllc_flux(
            const primitive_t& prL,
            const primitive_t& prR,
            const luint nhat,
            const real vface,
            const luint gid
        ) const;

        DUAL conserved_t calc_hlld_flux(
            const primitive_t& prL,
            const primitive_t& prR,
            const luint nhat,
            const real vface,
            const luint gid
        ) const;

        DUAL void set_riemann_solver()
        {
            switch (sim_solver) {
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

        DUAL conserved_t
        prims2flux(const primitive_t& prims, const luint nhat) const;

        template <TIMESTEP_TYPE dt_type = TIMESTEP_TYPE::ADAPTIVE>
        void adapt_dt();

        template <TIMESTEP_TYPE dt_type = TIMESTEP_TYPE::ADAPTIVE>
        void adapt_dt(const ExecutionPolicy<>& p);

        void simulate(
            std::function<real(real)> const& a,
            std::function<real(real)> const& adot,
            std::optional<function_t> const& d_outer  = nullptr,
            std::optional<function_t> const& s1_outer = nullptr,
            std::optional<function_t> const& s2_outer = nullptr,
            std::optional<function_t> const& s3_outer = nullptr,
            std::optional<function_t> const& e_outer  = nullptr
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

        DUAL real curl_e(
            const luint nhat,
            const real ejl,
            const real ejr,
            const real ekl,
            const real ekr
        ) const;

        /**
         * @brief
         * @retval
         */
        template <Plane P, Corner C>
        DUAL real calc_edge_emf(
            const conserved_t& fw,
            const conserved_t& fe,
            const conserved_t& fs,
            const conserved_t& fn,
            const ndarray<real>& bstagp1,
            const ndarray<real>& bstagp2,
            const primitive_t* prims,
            const luint ii,
            const luint jj,
            const luint kk,
            const luint ia,
            const luint ja,
            const luint ka,
            const luint nhat
        ) const;

        void emit_troubled_cells() const;

        void offload()
        {
            cons.copyToGpu();
            prims.copyToGpu();
            edens_guess.copyToGpu();
            dt_min.copyToGpu();
            density_source.copyToGpu();
            m1_source.copyToGpu();
            m2_source.copyToGpu();
            m3_source.copyToGpu();
            object_pos.copyToGpu();
            energy_source.copyToGpu();
            inflow_zones.copyToGpu();
            bcs.copyToGpu();
            troubled_cells.copyToGpu();
            sourceG1.copyToGpu();
            sourceG2.copyToGpu();
            sourceG3.copyToGpu();
            sourceB1.copyToGpu();
            sourceB2.copyToGpu();
            sourceB3.copyToGpu();
            bstag1.copyToGpu();
            bstag2.copyToGpu();
            bstag3.copyToGpu();
        }

        DUAL std::tuple<real, primitive_t, primitive_t, primitive_t, bool>
        hlld_vdiff(
            const real p,
            const conserved_t r[2],
            const real lam[2],
            real bn,
            const luint nhat,
            const luint gid
        ) const;
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