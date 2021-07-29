/**
 * Housed here is a series
 * of function declarations 
 * used to compute the artificial
 * velocity acorss grid faces. Currently
 * in just 2D, but can be generalized.
 * 
 * Marcus DuPont
 * New York Univeristy
 * 06/05/2021
 */

#ifndef VISCOUS_DIFF_HPP
#define VISCOUS_DIFF_HPP

#include "hydro_structs.hpp"
#include "clattice.hpp"

namespace simbi
{
    enum class Face{LEFT, RIGHT, UPPER, LOWER};


    struct ArtificialViscosity
    {
        bool bite;
        ArtificialViscosity();
        ~ArtificialViscosity();
        std::vector<sr2d::Conserved> avFlux;
        std::vector<sr2d::Primitive> visc_prims; 

        void av_flux_at_face(
            sr2d::Conserved &flux,
            const std::vector<sr2d::Primitive> &prims,
            const CLattice &coord_lattice,
            const int ii, 
            const int jj,
            const simbi::Face cell_face
        );
        void calc_four_velocity(std::vector<sr2d::Primitive> &prims);
        void calc_artificial_visc(
            const std::vector<sr2d::Primitive> &prims,
            const simbi::CLattice &coord_lattice
        );
    };
    

}

#endif