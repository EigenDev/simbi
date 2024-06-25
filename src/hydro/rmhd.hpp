/**
 * ***********************(C) COPYRIGHT 2023 Marcus DuPont**********************
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
 * ***********************(C) COPYRIGHT 2023 Marcus DuPont**********************
 */

#ifndef RMHD_HPP
#define RMHD_HPP

#include "base.hpp"             // for HydroBase
#include "build_options.hpp"    // for real, GPU_CALLABLE_MEMBER, lint, luint
#include "common/enums.hpp"     // for TIMESTEP_TYPE
#include "common/helpers.hpp"   // for my_min, my_max, ...
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
        primitive_t cons2prim(const conserved_t& cons, const luint gid);

        void advance(const ExecutionPolicy<>& p);

        GPU_CALLABLE_MEMBER
        void calc_max_wave_speeds(
            const primitive_t& prims,
            const luint nhat,
            real speeds[],
            real& cs2
        ) const;

        GPU_CALLABLE_MEMBER
        eigenvals_t calc_eigenvals(
            const primitive_t& primsL,
            const primitive_t& primsR,
            const luint nhat
        ) const;

        GPU_CALLABLE_MEMBER
        conserved_t prims2cons(const primitive_t& prims) const;

        GPU_CALLABLE_MEMBER
        conserved_t calc_hll_flux(
            primitive_t& prL,
            primitive_t& prR,
            const real bface,
            const luint nhat,
            const real vface = 0.0
        ) const;

        GPU_CALLABLE_MEMBER
        conserved_t calc_hllc_flux(
            primitive_t& prL,
            primitive_t& prR,
            const real bface,
            const luint nhat,
            const real vface = 0.0
        ) const;

        GPU_CALLABLE_MEMBER
        conserved_t calc_hlld_flux(
            primitive_t& prL,
            primitive_t& prR,
            const real bface,
            const luint nhat,
            const real vface = 0.0
        ) const;

        GPU_CALLABLE_MEMBER
        conserved_t (RMHD<dim>::*riemann_solve)(
            primitive_t& prL,
            primitive_t& prR,
            const real bface,
            const luint nhat,
            const real vface
        ) const;

        GPU_CALLABLE_MEMBER
        conserved_t
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

        GPU_CALLABLE_MEMBER
        constexpr real get_x1face(const lint ii, const int side) const;

        GPU_CALLABLE_MEMBER
        constexpr real get_x2face(const lint ii, const int side) const;

        GPU_CALLABLE_MEMBER
        constexpr real get_x3face(const lint ii, const int side) const;

        GPU_CALLABLE_MEMBER
        constexpr real get_x1_differential(const lint ii) const;

        GPU_CALLABLE_MEMBER
        constexpr real get_x2_differential(const lint ii) const;

        GPU_CALLABLE_MEMBER
        constexpr real get_x3_differential(const lint ii) const;

        GPU_CALLABLE_MEMBER
        real
        get_cell_volume(const lint ii, const lint jj = 0, const lint kk = 0)
            const;

        GPU_CALLABLE_MEMBER real curl_e(
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
        GPU_CALLABLE_MEMBER real calc_edge_emf(
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
            if constexpr (dim > 1) {
                m2_source.copyToGpu();
            }
            if constexpr (dim > 2) {
                m3_source.copyToGpu();
            }
            if constexpr (dim > 1) {
                object_pos.copyToGpu();
            }
            energy_source.copyToGpu();
            inflow_zones.copyToGpu();
            bcs.copyToGpu();
            troubled_cells.copyToGpu();
            sourceG1.copyToGpu();
            if constexpr (dim > 1) {
                sourceG2.copyToGpu();
            }
            if constexpr (dim > 2) {
                sourceG3.copyToGpu();
            }
            sourceB1.copyToGpu();
            if constexpr (dim > 1) {
                sourceB2.copyToGpu();
            }
            if constexpr (dim > 2) {
                sourceB3.copyToGpu();
            }

            if constexpr (dim > 1) {
                bstag1.copyToGpu();
                bstag2.copyToGpu();
                if constexpr (dim > 2) {
                    bstag3.copyToGpu();
                }
            }
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

#include "rmhd.tpp"
#endif