/**
 * This is a coordinate lattice class instance 
 * that will be used to compute the lattice
 * vertices as well as the cell face areas
 * which will then be cached away.
 * 
 * 
 * Marcus DuPont
 * New York University 
 * 08/08/2021
*/

#ifndef CLATTICE3D_HPP
#define CLATTICE3D_HPP

#include <vector> 
#include <array> 
#include "config.hpp"
namespace simbi {

    struct CLattice3D{
            std::vector<real> x1_face_areas, x2_face_areas, x3_face_areas; // X & Y cell face areas
            std::vector<real> x1vertices, x2vertices, x3vertices;       // X & Y cell vertices
            std::vector<real> x1ccenters, x2ccenters, x3ccenters;       // X & Y cell centers
            std::vector<real> dx1, dx2, dx3;                     // Generalized x1 & x2 cell widths
            std::vector<real> dV1, dV2, dV3, dVc;                // Generalized effectivee cell "volumes"
            std::vector<real> x1mean, cot, sin;                  // Cache some geometrical source term components
            int               nx1zones, nx2zones, nx3zones;       // Number of zones in either direction
            std::vector<real> s1_face_areas, s2_face_areas;
            simbi::Geometry _geom;
            simbi::Cellspacing _cell_space;
            
            //=========== GPU MIRRORS
            real* gpu_x1_face_areas, *gpu_x2_face_areas, *gpu_x3_face_areas;
            real* gpu_x1vertices, *gpu_x2vertices, *gpu_x3vertices, *gpu_x1ccenters, *gpu_x2ccenters, *gpu_x3ccenters;
            real* gpu_dx1, *gpu_dx2, *gpu_dx3, *gpu_dV1, *gpu_dV2, gpu_dV3, *gpu_x1mean, *gpu_cot, *gpu_sin;


            
        
             CLattice3D ();
             CLattice3D(
                 std::vector<real> &x1, 
                 std::vector<real> &x2, 
                 std::vector<real> &x3,
                 simbi::Geometry geom);
            ~CLattice3D();

            void set_nx1_zones();
            void set_nx2_zones();
            void set_nx3_zones();

            // Compute the x1 cell vertices based on 
            // the linear or logarithmic cell spacing
            void compute_x1_vertices(simbi::Cellspacing  cellspacing);

            // Compute the x2 cell vertices based on 
            // the linear or logarithmic cell spacing
            void compute_x2_vertices(simbi::Cellspacing  cellspacing);

            // Compute the x3 cell vertices based on 
            // the linear or logarithmic cell spacing
            void compute_x3_vertices(simbi::Cellspacing  cellspacing);

            // Compute the xface areas based on
            // a non-Cartesian simulation geometry
            void compute_x1face_areas();

            // Compute the yface areas based on
            // a non-Cartesian simulation geometry
            void compute_x2face_areas();

            // Compute the zface areas based on
            // a non-Cartesian simulation geometry
            void compute_x3face_areas();

            void compute_s1face_areas();
            void compute_s2face_areas();

            // Compute the directional cell widths
            void compute_dx1();

            void compute_dx2();

            void compute_dx3();

            // Compute the effective directional "volumes"
            void compute_dV1();

            void compute_dV2();

            void compute_dV3();

            void compute_dV();

            void compute_x1mean();

            void compute_cot();
            void compute_sin();

            void config_lattice(simbi::Cellspacing xcellspacing,
                                simbi::Cellspacing ycellspacing,
                                simbi::Cellspacing zcellspacing);

            
    };

}

#endif