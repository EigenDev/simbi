/* 
* Single header for 1, 2, and 3D RMHD Calculations
*/
#ifndef RMHD_HPP
#define RMHD_HPP

#include "base.hpp"                 // for HydroBase
#include <optional>                 // for optional
#include <vector>                   // for vector
#include <functional>               // for function
#include <type_traits>              // for conditional_t
#include "build_options.hpp"        // for real, GPU_CALLABLE_MEMBER, lint, luint
#include "common/enums.hpp"         // for TIMESTEP_TYPE
#include "util/exec_policy.hpp"     // for ExecutionPolicy
#include "util/ndarray.hpp"         // for ndarray
#include "common/hydro_structs.hpp" // for Conserved, Primitive
#include "common/helpers.hip.hpp"   // for my_min, my_max, ...

namespace simbi
{
    template<int dim>
    struct RMHD : public HydroBase
    {
        // set the primitive and conservative types at compile time
        using primitive_t = typename std::conditional_t<
        dim == 1,
        rmhd1d::Primitive,
        std::conditional_t<
        dim == 2,
        rmhd2d::Primitive,
        rmhd3d::Primitive>
        >;
        using conserved_t = typename std::conditional_t<
        dim == 1,
        rmhd1d::Conserved,
        std::conditional_t<
        dim == 2,
        rmhd2d::Conserved,
        rmhd3d::Conserved>
        >;
        using primitive_soa_t = typename std::conditional_t<
        dim == 1,
        rmhd1d::PrimitiveSOA,
        std::conditional_t<
        dim == 2,
        rmhd2d::PrimitiveSOA,
        rmhd3d::PrimitiveSOA>
        >;
        using eigenvals_t = typename std::conditional_t<
        dim == 1,
        rmhd1d::Eigenvals,
        std::conditional_t<
        dim == 2,
        rmhd2d::Eigenvals,
        rmhd3d::Eigenvals>
        >;

        using function_t = typename std::conditional_t<
        dim == 1,
        std::function<real(real)>,
        std::conditional_t<
        dim == 2,
        std::function<real(real, real)>,
        std::function<real(real, real, real)>>
        >;

        function_t dens_outer;
        function_t mom1_outer;
        function_t mom2_outer;
        function_t mom3_outer;
        function_t enrg_outer;

        const static int dimensions = dim;

        /* Shared Data Members */
        ndarray<primitive_t> prims;
        ndarray<conserved_t> cons, outer_zones, inflow_zones;
        ndarray<real> enthalpy_density_guess, dt_min;
        bool scalar_all_zeros;

        /* Methods */
        RMHD();
        RMHD(
            std::vector<std::vector<real>> &state,
            const InitialConditions &init_conditions);
        ~RMHD();

        void cons2prim(const ExecutionPolicy<> &p);

        void advance(
            const ExecutionPolicy<> &p,
            const luint xstride,
            const luint ystride);

        GPU_CALLABLE_MEMBER
        eigenvals_t calc_eigenvals(
            const primitive_t &primsL,
            const primitive_t &primsR,
            const luint nhat) const;

        GPU_CALLABLE_MEMBER
        conserved_t prims2cons(const primitive_t &prims) const;

        GPU_CALLABLE_MEMBER
        conserved_t calc_hllc_flux(
            const conserved_t &left_state,
            const conserved_t &right_state,
            const conserved_t &left_flux,
            const conserved_t &right_flux,
            const primitive_t &left_prims,
            const primitive_t &right_prims,
            const luint nhat,
            const real vface) const;

        GPU_CALLABLE_MEMBER
        conserved_t prims2flux(const primitive_t &prims, const luint nhat) const;

        GPU_CALLABLE_MEMBER
        conserved_t calc_hll_flux(
            const conserved_t &left_state,
            const conserved_t &right_state,
            const conserved_t &left_flux,
            const conserved_t &right_flux,
            const primitive_t &left_prims,
            const primitive_t &right_prims,
            const luint nhat,
            const real vface) const;  

        template<TIMESTEP_TYPE dt_type = TIMESTEP_TYPE::ADAPTIVE>
        void adapt_dt();

        template<TIMESTEP_TYPE dt_type = TIMESTEP_TYPE::ADAPTIVE>
        void adapt_dt(const ExecutionPolicy<> &p);

        void simulate(
            std::function<real(real)> const &a,
            std::function<real(real)> const &adot,
            std::optional<function_t> const &d_outer  = nullptr,
            std::optional<function_t> const &s1_outer = nullptr,
            std::optional<function_t> const &s2_outer = nullptr,
            std::optional<function_t> const &s3_outer = nullptr,
            std::optional<function_t> const &e_outer  = nullptr
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
        real get_cell_volume(const lint ii, const lint jj = 0, const lint kk = 0) const;

        void emit_troubled_cells();
    };

    namespace rm
    {
        // Primitive<dim> template alias
        template<int dim>
        using Primitive = typename RMHD<dim>::primitive_t;

        // Conservative template alias
        template<int dim>
        using Conserved = typename RMHD<dim>::conserved_t;

        // Eigenvalue template alias
        template<int dim>
        using Eigenvals = typename RMHD<dim>::eigenvals_t;

        // file writer template alias
        template<int dim>
        constexpr auto write2file = helpers::write_to_file<typename RMHD<dim>::primitive_soa_t, dim, RMHD<dim>>;
            
    } // namespace srhd
}

template<>
struct is_relativistic<simbi::RMHD<1>>
{
    static constexpr bool value = true;
};
template<>
struct is_relativistic<simbi::RMHD<2>>
{
    static constexpr bool value = true;
};
template<>
struct is_relativistic<simbi::RMHD<3>>
{
    static constexpr bool value = true;
};

#include "rmhd.tpp"
#endif