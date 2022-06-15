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

#ifndef CLATTICE1D_HPP
#define CLATTICE1D_HPP

#include <vector> 
#include "enums.hpp"
#include "build_options.hpp"
namespace simbi {

    struct CLattice1D{
            //---------Host vars-------------
            std::vector<real> face_areas; // X & Y cell face areas
            std::vector<real> x1vertices;         // X & Y cell vertices
            std::vector<real> x1ccenters;         // X & Y cell centers
            std::vector<real> dx1;                // Generalized x1 & x2 cell widths
            std::vector<real> dV;                 // Generalized effectivee cell "volumes"
            std::vector<real> x1mean;             // Cache some geometrical source term components
            int                 nzones    ;         // Number of zones in either direction
            simbi::Geometry _geom;
            
            //-------- Device Vars------------      
            real* gpu_face_areas;         // X & Y cell face areas
            real* gpu_x1vertices;         // X & Y cell vertices
            real* gpu_x1ccenters;         // X & Y cell centers
            real* gpu_dx1;                // Generalized x1 & x2 cell widths
            real* gpu_dV;                 // Generalized effectivee cell "volumes"
            real* gpu_x1mean;             // Cache some geometrical source term components
        
            CLattice1D ();
            CLattice1D(std::vector<real> &x1, simbi::Geometry geom);
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

            // Compute new values for moving faces
            void compute_newx1(
                simbi::Cellspacing Cellspacing, 
                const real vinner_excision,
                const real vouter_excision);

            void config_lattice(simbi::Cellspacing cellspacing);

            
    };

}

#endif