/* 
* interface between python construction of the 1D state 
* and cpp. This is where the heavy lifting will occur when 
* computing the HLL derivative of the state vector
* given the state itself.
*/
#ifndef EULER1D_HPP
#define EULER1D_HPP

#include <vector>
#include "common/hydro_structs.hpp"
#include "common/helpers.hpp"
#include "common/enums.hpp"
#include "build_options.hpp"
#include "util/exec_policy.hpp"
#include "util/ndarray.hpp"
#include "base.hpp"

namespace simbi {

    struct Newtonian1D : public HydroBase {
        using conserved_t = hydro1d::Conserved;
        using primitive_t = hydro1d::Primitive;
        using primitive_soa_t = hydro1d::PrimitiveSOA;
        const static int dimensions = 1;

        ndarray<conserved_t> cons, outer_zones; 
        ndarray<primitive_t> prims;
        ndarray<real> sourceRho, sourceMom, sourceE, dt_min;
        
        Newtonian1D() = default;
        Newtonian1D(
            std::vector<std::vector<real>> state, 
            real gamma, 
            real cfl,
            std::vector<real> x1, 
            std::string coord_system);
        ~Newtonian1D() {};

        // Calculate the wave speeds from the Jacobian Matrix formed by the Euler Eqns
        GPU_CALLABLE_MEMBER
        hydro1d::Eigenvals calc_eigenvals(const hydro1d::Primitive &left_state, const hydro1d::Primitive &right_state);

        void cons2prim(const ExecutionPolicy<> &p);
        void adapt_dt();
        void adapt_dt(luint blockSize, luint tblock);
        
        GPU_CALLABLE_MEMBER
        hydro1d::Conserved prims2cons(const hydro1d::Primitive &prims);

        GPU_CALLABLE_MEMBER
        hydro1d::Conserved prims2flux(const hydro1d::Primitive &prims);

        GPU_CALLABLE_INLINE
        constexpr real get_xface(const lint ii, const simbi::Geometry geometry, const int side) const
        {
            switch (geometry)
            {
            case simbi::Geometry::CARTESIAN:
                {
                        const real xl = helpers::my_max(x1min  + (ii - static_cast<real>(0.5)) * dx1,  x1min);
                        if (side == 0) {
                            return xl;
                        } else {
                            return helpers::my_min(xl + dx1 * (ii == 0 ? 0.5 : 1.0), x1max);
                        }
                }
            case simbi::Geometry::SPHERICAL:
                {
                        const real rl = helpers::my_max(x1min * std::pow(10, (ii - static_cast<real>(0.5)) * dlogx1),  x1min);
                        if (side == 0) {
                            return rl;
                        } else {
                            return helpers::my_min(rl * std::pow(10, dlogx1 * (ii == 0 ? 0.5 : 1.0)), x1max);
                        }
                }
            case simbi::Geometry::CYLINDRICAL:
            {
                //  TODO: Implement
                break;
            }
            }
        }

        GPU_CALLABLE_MEMBER
        hydro1d::Conserved calc_hll_flux(
            const hydro1d::Primitive &left_prims,
            const hydro1d::Primitive &right_prims,
            const hydro1d::Conserved &left_state,
            const hydro1d::Conserved &right_state,
            const hydro1d::Conserved &left_flux,
            const hydro1d::Conserved &right_flux);

        GPU_CALLABLE_MEMBER
        hydro1d::Conserved calc_hllc_flux(
            const hydro1d::Primitive &left_prims,
            const hydro1d::Primitive &right_prims,
            const hydro1d::Conserved &left_state,
            const hydro1d::Conserved &right_state,
            const hydro1d::Conserved &left_flux,
            const hydro1d::Conserved &right_flux);

        void advance(
            const ExecutionPolicy<> &p, 
            const luint xstride);

        std::vector<std::vector<real> > simulate1D(
            std::vector<std::vector<real>> &sources,
            real tstart,
            real tend,
            real dlogt,
            real plm_theta,
            real engine_duration,
            real chkpt_luinterval,
            int  chkpt_idx,
            std::string data_directory,
            std::string boundary_condition,
            bool first_order,
            bool linspace,
            bool hllc);
    };
}

#endif