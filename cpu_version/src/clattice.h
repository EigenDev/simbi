/**
 * This is a coordinate lattice class instance 
 * that will be used to compute the lattice
 * vertices as well as the cell face areas
 * which will then be cached away.
 * 
 * 
 * Marcus DuPont
 * New York University 
 * 19/04/2021
*/

#ifndef CLATTICE_H
#define CLATTICE_H

#include <vector> 
#include <array> 
#include "config.h"
namespace simbi {

    struct CLattice{
            std::vector<double> x1_face_areas, x2_face_areas; // X & Y cell face areas
            std::vector<double> x1vertices, x2vertices;       // X & Y cell vertices
            std::vector<double> x1ccenters, x2ccenters;       // X & Y cell centers
            std::vector<double> dx1, dx2;                     // Generalized x1 & x2 cell widths
            std::vector<double> dV1, dV2, dVc;                     // Generalized effectivee cell "volumes"
            std::vector<double> x1mean, cot;                  // Cache some geometrical source term components
            int                 nx1zones, nx2zones    ;       // Number of zones in either direction
            std::vector<double> s1_face_areas, s2_face_areas;
            simbi::Geometry _geom;
            simbi::Cellspacing _cell_space;
        
            CLattice ();
            CLattice(std::vector<double> &x1, std::vector<double> &x2, simbi::Geometry geom);
            ~CLattice();

            void set_nx1_zones();
            void set_nx2_zones();

            // Compute the x1 cell vertices based on 
            // the linear or logarithmic cell spacing
            void compute_x1_vertices(simbi::Cellspacing  cellspacing);

            // Compute the x2 cell vertices based on 
            // the linear or logarithmic cell spacing
            void compute_x2_vertices(simbi::Cellspacing  cellspacing);

            // Compute the xface areas based on
            // a non-Cartesian simulation geometry
            void compute_x1face_areas();

            // Compute the yface areas based on
            // a non-Cartesian simulation geometry
            void compute_x2face_areas();

            void compute_s1face_areas();
            void compute_s2face_areas();

            // Compute the directional cell widths
            void compute_dx1();

            void compute_dx2();

            // Compute the effective directional "volumes"
            void compute_dV1();

            void compute_dV2();

            void compute_dV();

            void compute_x1mean();

            void compute_cot();

            void config_lattice(simbi::Cellspacing xcellspacing,
                                simbi::Cellspacing ycellspacing);

            
    };

}

#endif