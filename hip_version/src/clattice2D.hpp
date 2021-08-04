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

#ifndef CLATTICE2D_HPP
#define CLATTICE2D_HPP

#include <vector> 
#include <array> 
#include "config.hpp"
namespace simbi {

    struct CLattice2D{
            std::vector<real> x1_face_areas, x2_face_areas; // X & Y cell face areas
            std::vector<real> x1vertices, x2vertices;       // X & Y cell vertices
            std::vector<real> x1ccenters, x2ccenters;       // X & Y cell centers
            std::vector<real> dx1, dx2;                     // Generalized x1 & x2 cell widths
            std::vector<real> dV1, dV2, dVc;                // Generalized effectivee cell "volumes"
            std::vector<real> x1mean, cot;                  // Cache some geometrical source term components
            int               nx1zones, nx2zones    ;       // Number of zones in either direction
            std::vector<real> s1_face_areas, s2_face_areas;
            simbi::Geometry _geom;
            simbi::Cellspacing _cell_space;
            
            //=========== GPU MIRRORS
            real* gpu_x1_face_areas, *gpu_x2_face_areas;
            real* gpu_x1vertices, *gpu_x2vertices, *gpu_x1ccenters, *gpu_x2ccenters;
            real* gpu_dx1, *gpu_dx2, *gpu_dV1, *gpu_dV2, *gpu_x1mean, *gpu_cot;


            
        
             CLattice2D ();
             CLattice2D(std::vector<real> &x1, std::vector<real> &x2, simbi::Geometry geom);
            ~CLattice2D();

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