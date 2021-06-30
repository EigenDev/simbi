/**
 * This is a coordinate lattice class instance 
 * that will be used to compute the 1D lattice
 * vertices as well as the cell face areas
 * which will then be cached away.
 * 
 * 
 * Marcus DuPont
 * New York University 
 * 25/06/2021
*/

#ifndef CLATTICE_1D_HPP
#define CLATTICE_1D_HPP

#include <vector> 
#include "config.h"
namespace simbi {

    struct CLattice1D{
            std::vector<double> face_areas; // X & Y cell face areas
            std::vector<double> x1vertices;       // X & Y cell vertices
            std::vector<double> x1ccenters;       // X & Y cell centers
            std::vector<double> dx1;                     // Generalized x1 & x2 cell widths
            std::vector<double> dV;                     // Generalized effectivee cell "volumes"
            std::vector<double> x1mean;                  // Cache some geometrical source term components
            int                 nzones    ;       // Number of zones in either direction

            simbi::Geometry _geom;
            simbi::Cellspacing _cell_space;
        
            CLattice1D ();
            CLattice1D(std::vector<double> &x1, simbi::Geometry geom);
            ~CLattice1D();

            void set_nzones();

            // Compute the x1 cell vertices based on 
            // the linear or logarithmic cell spacing
            void compute_x1_vertices(simbi::Cellspacing  cellspacing);

            // Compute the xface areas based on
            // a non-Cartesian simulation geometry
            void compute_face_areas();

            // Compute the directional cell widths
            void compute_dx1();

            // Compute the effective directional "volumes"

            void compute_dV();

            void compute_x1mean();

            void config_lattice(simbi::Cellspacing cellspacing);

            
    };

}

#endif