/**
 * ***********************(C) COPYRIGHT 2024 Marcus DuPont**********************
 * @file       srhd.hpp
 * @brief      single header for 1,2, and 3D SRHD calculations
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
#ifndef SRHD_HPP
#define SRHD_HPP

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
    struct SRHD : public HydroBase {
        // set the primitive and conservative types at compile time
        using primitive_t     = anyPrimitive<dim, Regime::SRHD>;
        using conserved_t     = anyConserved<dim, Regime::SRHD>;
        using primitive_soa_t = typename std::conditional_t<
            dim == 1,
            sr1d::PrimitiveSOA,
            std::conditional_t<
                dim == 2,
                sr2d::PrimitiveSOA,
                sr3d::PrimitiveSOA>>;
        using eigenvals_t = typename std::conditional_t<
            dim == 1,
            sr1d::Eigenvals,
            std::conditional_t<dim == 2, sr2d::Eigenvals, sr3d::Eigenvals>>;

        using function_t = typename std::conditional_t<
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

        const static int dimensions = dim;

        /* Shared Data Members */
        ndarray<primitive_t> prims;
        ndarray<conserved_t> cons, outer_zones, inflow_zones;
        ndarray<real> pressure_guess, dt_min;
        bool scalar_all_zeros;

        /* Methods */
        SRHD();
        SRHD(
            std::vector<std::vector<real>>& state,
            const InitialConditions& init_conditions
        );
        ~SRHD();

        void cons2prim(const ExecutionPolicy<>& p);

        void advance(const ExecutionPolicy<>& p);

        DUAL eigenvals_t calc_eigenvals(
            const primitive_t& primsL,
            const primitive_t& primsR,
            const luint nhat
        ) const;

        DUAL conserved_t prims2cons(const primitive_t& prims) const;

        DUAL conserved_t calc_hllc_flux(
            const conserved_t& uL,
            const conserved_t& uR,
            const conserved_t& fL,
            const conserved_t& fR,
            const primitive_t& prL,
            const primitive_t& prR,
            const luint nhat,
            const real vface = 0.0
        ) const;

        DUAL conserved_t
        prims2flux(const primitive_t& prims, const luint nhat) const;

        DUAL conserved_t calc_hlle_flux(
            const conserved_t& uL,
            const conserved_t& uR,
            const conserved_t& fL,
            const conserved_t& fR,
            const primitive_t& prL,
            const primitive_t& prR,
            const luint nhat,
            const real vface = 0.0
        ) const;

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

        void offload()
        {
            cons.copyToGpu();
            prims.copyToGpu();
            pressure_guess.copyToGpu();
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
        }

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