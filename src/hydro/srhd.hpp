/* 
* Single header for 1, 2, and 3D SRHD Calculations
*/
#ifndef SRHD_HPP
#define SRHD_HPP

#include <vector>
#include "common/hydro_structs.hpp"
#include "common/helpers.hip.hpp"
#include "base.hpp"

namespace simbi
{
    template<int dim, Platform build_mode = BuildPlatform>
    struct SRHD : public HydroBase
    {
        // set the primitive and conservative types at compile time
        using primitive_t = typename std::conditional_t<
        dim == 1,
        sr1d::Primitive,
        std::conditional_t<
        dim == 2,
        sr2d::Primitive,
        sr3d::Primitive>
        >;
        using conserved_t = typename std::conditional_t<
        dim == 1,
        sr1d::Conserved,
        std::conditional_t<
        dim == 2,
        sr2d::Conserved,
        sr3d::Conserved>
        >;//sr3d::Conserved;
        using primitive_soa_t = typename std::conditional_t<
        dim == 1,
        sr1d::PrimitiveSOA,
        std::conditional_t<
        dim == 2,
        sr2d::PrimitiveSOA,
        sr3d::PrimitiveSOA>
        >;
        using eigenvals_t = typename std::conditional_t<
        dim == 1,
        sr1d::Eigenvals,
        std::conditional_t<
        dim == 2,
        sr2d::Eigenvals,
        sr3d::Eigenvals>
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
        ndarray<primitive_t, build_mode> prims;
        ndarray<conserved_t, build_mode> cons, outer_zones, inflow_zones;
        ndarray<simbi::BoundaryCondition, build_mode> bcs;
        ndarray<int, build_mode> troubled_cells;
        ndarray<real, build_mode> sourceG1, sourceG2, sourceG3;
        ndarray<real, build_mode> density_source, m1_source, m2_source, m3_source, energy_source;
        ndarray<real, build_mode> pressure_guess, dt_min;
        ndarray<bool, build_mode> object_pos;
        
        bool scalar_all_zeros;

        /* Methods */
        SRHD();
        SRHD(
            std::vector<std::vector<real>> &state,
            InitialConditions &init_conditions);
        ~SRHD();

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
        
        conserved_t calc_hll_state(
            const conserved_t &left_state,
            const conserved_t &right_state,
            const conserved_t &left_flux,
            const conserved_t &right_flux,
            const primitive_t &left_prims,
            const primitive_t &right_prims,
            const luint nhat) const;

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

        std::vector<std::vector<real>> simulate(
            std::function<real(real)> a,
            std::function<real(real)> adot,
            function_t const &d_outer  = nullptr,
            function_t const &s1_outer = nullptr,
            function_t const &s2_outer = nullptr,
            function_t const &s3_outer = nullptr,
            function_t const &e_outer  = nullptr
        );

        GPU_CALLABLE_INLINE
        constexpr real get_x1face(const lint ii, const int side) const;

        GPU_CALLABLE_INLINE
        constexpr real get_x2face(const lint ii, const int side) const;

        GPU_CALLABLE_INLINE
        constexpr real get_x3face(const lint ii, const int side) const;

        GPU_CALLABLE_INLINE
        constexpr real get_x1_differential(const lint ii) const;

        GPU_CALLABLE_INLINE
        constexpr real get_x2_differential(const lint ii) const;

        GPU_CALLABLE_INLINE
        constexpr real get_x3_differential(const lint ii) const;

        GPU_CALLABLE_INLINE
        real get_cell_volume(const lint ii, const lint jj = 0, const lint kk = 0) const;

        // GPU_CALLABLE_INLINE
        // conserved_t get_geometric_source_terms(const primitive_t &prim, const real dV) const {
        //     // Grab central primitives
        //     const real rhoc = prim.rho;
        //     const real v1   = prim.v2omponent(1);
        //     const real v2   = prim.v2omponent(2);
        //     const real v3   = prim.v2omponent(3);
        //     const real pc   = prim.p;
        //     const real hc   = 1 + gamma * pc/(rhoc * (gamma - 1));
        //     const real gam2 = 1/(1 - (v1 * v1 + v2 * v2 + v3 * v3));

        //     switch (geometry)
        //     {
        //     case Geometry::SPHERICAL:
        //         {
        //             if constexpr(dim == 1) {
        //                 return = conserved_t{
        //                     0, 
        //                     pc * (s1R - s1L) / dV,
        //                     0
        //                 };

        //             } else if constexpr(dim == 2) {
        //                 return = conserved_t{
        //                     0, 
        //                     (rhoc * hc * gam2 * (v2 * v2)) / rmean + pc * (s1R - s1L) / dV1,
        //                     rhoc * hc * gam2 * (-v1 * v2) / rmean + pc * (s2R - s2L)/dV2,
        //                     0
        //                 };

        //             } else {
        //                 return = conserved_t{
        //                     0, 
        //                     (rhoc * hc * gam2 * (v2 * v2 + v3 * v3)) / rmean + pc * (s1R - s1L) / dV1,
        //                     rhoc * hc * gam2 * (v3 * v3 * cot - v1 * v2) / rmean + pc * (s2R - s2L)/dV2 , 
        //                     - rhoc * hc * gam2 * v3 * (v1 + v2 * cot) / rmean, 
        //                     0
        //                 };
        //             }
        //         }
            
        //     case Geometry::AXIS_CYLINDRICAL:
        //     {

        //         return = conserved_t{
        //             0, 
        //             pc * (s1R - s1L) / dV,
        //             0,
        //             0
        //         };
        //     }
        //     case Geometry::PLANAR_CYLINDRICAL:
        //     {
        //         return = conserved_t{
        //             0, 
        //             (rhoc * hc * gam2 * (v2 * v2)) / rmean + pc * (s1R - s1L) / dV,
        //             rhoc * hc * gam2 * (-v1 * v2) / rmean + pc * (s2R - s2L) / dV,
        //             0
        //         };
        //     }
        //     case Geometry::CYLINDRICAL:
        //     {
        //         if constexpr(dim == 1) {
        //             return = conserved_t{
        //                 0, 
        //                 pc * (s1R - s1L) / dV,
        //                 0
        //             };

        //         } 
        //         } else {
        //             return = conserved_t{
        //                 0, 
        //                 (rhoc * hc * gam2 * (v2 * v2)) / rmean + pc * (s1R - s1L) / dV,
        //                 - rhoc * hc * gam2 * (v1 * v2) / rmean, 
        //                 0, 
        //                 0
        //             };
        //         }
        //     }
        //     default:
        //         return conserved_t;
        //     }
        // }

        void emit_troubled_cells();
    };
}

template<>
struct is_relativistic<simbi::SRHD<1>>
{
    static constexpr bool value = true;
};
template<>
struct is_relativistic<simbi::SRHD<2>>
{
    static constexpr bool value = true;
};
template<>
struct is_relativistic<simbi::SRHD<3>>
{
    static constexpr bool value = true;
};

#include "srhd.tpp"
#endif