/* 
* Interface between python construction of the 2D SR state 
* and cpp. This is where the heavy lifting will occur when 
* computing the HLL derivative of the state vector
* given the state itself.
*/
#ifndef SRHYDRO3D_HIP_HPP
#define SRHYDRO3D_HIP_HPP

#include <vector>
#include "common/hydro_structs.hpp"
#include "common/helpers.hpp"
#include "util/ndarray.hpp"
#include "base.hpp"

namespace simbi
{
    struct SRHD3D : public HydroBase
    {
        using primitive_t     = sr3d::Primitive;
        using conserved_t     = sr3d::Conserved;
        using primitive_soa_t = sr3d::PrimitiveSOA;
        const static int dimensions = 3;

        /* Shared Data Members */
        ndarray<primitive_t> prims;
        ndarray<conserved_t> cons, outer_zones;
        ndarray<real> sourceD, sourceS1, sourceS2, sourceS3, sourceTau, pressure_guess, dt_min;
        ndarray<bool> object_pos;
        bool d_all_zeros, s1_all_zeros, s2_all_zeros, s3_all_zeros, e_all_zeros, scalar_all_zeros, quirk_smoothing;

        /* Methods */
        SRHD3D();
        SRHD3D(
            std::vector<std::vector<real>> state3D, 
            luint nx, 
            luint ny,
            luint nz, 
            real gamma, 
            std::vector<real> x1,
            std::vector<real> x2, 
            std::vector<real> x3,
            real cfl, 
            std::string coord_system);
        ~SRHD3D();

        void cons2prim(const ExecutionPolicy<> &p);

        void advance(
            const ExecutionPolicy<> &p,
            const luint xstride,
            const luint ystride,
            const luint zstride);

        GPU_CALLABLE_MEMBER
        sr3d::Eigenvals calc_Eigenvals(
            const sr3d::Primitive &primsL,
            const sr3d::Primitive &primsR,
            const luint nhat);

        GPU_CALLABLE_MEMBER
        sr3d::Conserved prims2cons(const sr3d::Primitive &prims);
        
        sr3d::Conserved calc_hll_state(
            const sr3d::Conserved &left_state,
            const sr3d::Conserved &right_state,
            const sr3d::Conserved &left_flux,
            const sr3d::Conserved &right_flux,
            const sr3d::Primitive &left_prims,
            const sr3d::Primitive &right_prims,
            const luint nhat);

        sr3d::Conserved calc_luintermed_statesSR2D(const sr3d::Primitive &prims,
                                                 const sr3d::Conserved &state,
                                                 real a,
                                                 real aStar,
                                                 real pStar,
                                                 luint nhat);

        GPU_CALLABLE_MEMBER
        sr3d::Conserved calc_hllc_flux(
            const sr3d::Conserved &left_state,
            const sr3d::Conserved &right_state,
            const sr3d::Conserved &left_flux,
            const sr3d::Conserved &right_flux,
            const sr3d::Primitive &left_prims,
            const sr3d::Primitive &right_prims,
            const luint nhat);

        GPU_CALLABLE_MEMBER
        sr3d::Conserved prims2flux(const sr3d::Primitive &prims, const luint nhat);

        GPU_CALLABLE_MEMBER
        sr3d::Conserved calc_hll_flux(
            const sr3d::Conserved &left_state,
            const sr3d::Conserved &right_state,
            const sr3d::Conserved &left_flux,
            const sr3d::Conserved &right_flux,
            const sr3d::Primitive &left_prims,
            const sr3d::Primitive &right_prims,
            const luint nhat);  

        template<TIMESTEP_TYPE dt_type = TIMESTEP_TYPE::ADAPTIVE>
        void adapt_dt();

        template<TIMESTEP_TYPE dt_type = TIMESTEP_TYPE::ADAPTIVE>
        void adapt_dt(const ExecutionPolicy<> &p, const luint bytes);

        std::vector<std::vector<real>> simulate3D(
            const std::vector<std::vector<real>> &sources,
            std::vector<bool> &object_cells,
            real tstart, 
            real tend, 
            real dlogt, 
            real plm_theta,
            real engine_duration, 
            real chkpt_interval,
            int  chkpt_idx,
            std::string data_directory, 
            std::string boundary_condition,
            bool first_order,
            bool linspace, 
            bool hllc,
            bool constant_sources);

        GPU_CALLABLE_INLINE
        constexpr real get_x1face(const lint ii, const simbi::Geometry geometry, const int side)
        {
            switch (geometry)
            {
            case simbi::Geometry::CARTESIAN:
                {
                    const real x1l = helpers::my_max(x1min  + (ii - static_cast<real>(0.5)) * dx1,  x1min);
                    if (side == 0) {
                        return x1l;
                    }
                    return helpers::my_min(x1l + dx1 * (ii == 0 ? 0.5 : 1.0), x1max);
                }
            case simbi::Geometry::SPHERICAL:
                {
                    const real rl = helpers::my_max(x1min * std::pow(10, (ii - static_cast<real>(0.5)) * dlogx1),  x1min);
                    if (side == 0) {
                        return rl;
                    }
                    return helpers::my_min(rl * std::pow(10, dlogx1 * (ii == 0 ? 0.5 : 1.0)), x1max);
                }
            case simbi::Geometry::CYLINDRICAL:
                {
                    const real rl = helpers::my_max(x1min * std::pow(10, (ii - static_cast<real>(0.5)) * dlogx1),  x1min);
                    if (side == 0) {
                        return rl;
                    }
                    return helpers::my_min(rl * std::pow(10, dlogx1 * (ii == 0 ? 0.5 : 1.0)), x1max);
                }
                break;
            }
        }


        GPU_CALLABLE_INLINE
        constexpr real get_x2face(const lint ii, const int side)
        {
            const real x2l = helpers::my_max(x2min  + (ii - static_cast<real>(0.5)) * dx2,  x2min);
            if (side == 0) {
                return x2l;
            } 
            return helpers::my_min(x2l + dx2 * (ii == 0 ? 0.5 : 1.0), x2max);
        }

        GPU_CALLABLE_INLINE
        constexpr real get_x3face(const lint ii, const int side)
        {

            const real x3l = helpers::my_max(x3min  + (ii - static_cast<real>(0.5)) * dx3,  x3min);
            if (side == 0) {
                return x3l;
            } 
            return helpers::my_min(x3l + dx3 * (ii == 0 ? 0.5 : 1.0), x3max);
        }

        GPU_CALLABLE_INLINE
        real get_cell_volume(const lint ii, const lint jj, const simbi::Geometry geometry, const real step)
        {
            const real x1l     = get_x1face(ii, geometry, 0);
            const real x1r     = get_x1face(ii, geometry, 1);
            // const real x1lf    = x1l * (1.0 + step * dt * hubble_param);
            // const real x1rf    = x1r * (1.0 + step * dt * hubble_param);
            const real tl     = helpers::my_max(x2min + (jj - static_cast<real>(0.5)) * dx2, x2min);
            const real tr     = helpers::my_min(tl + dx2 * (jj == 0 ? 0.5 : 1.0), x2max); 
            const real dcos   = std::cos(tl) - std::cos(tr);
            const real dV     = (2.0 * M_PI * (1.0 / 3.0) * (x1r * x1r * x1r - x1l * x1l * x1l) * dcos);
            return dV;
            
        }
    };
}

#endif