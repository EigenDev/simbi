#ifndef DUAL_HPP
#define DUAL_HPP

#include "build_options.hpp"
#include "common/clattice1D.hpp"
#include "common/clattice2D.hpp"
#include "common/clattice3D.hpp"
#include "device_api.hpp"
#include "hydro/euler1D.hpp"
#include "hydro/euler2D.hpp"

namespace simbi
{
    namespace dual
    {
        template <typename T, typename C, typename U> 
        struct DualSpace1D
        {
            DualSpace1D() {};
            ~DualSpace1D()
            {
                // printf("\nFreeing Device Memory...\n");
                simbi::gpu::api::gpuFree(host_u0);
                simbi::gpu::api::gpuFree(host_prims);
                simbi::gpu::api::gpuFree(host_clattice);
                simbi::gpu::api::gpuFree(host_dV);
                simbi::gpu::api::gpuFree(host_dx1);
                simbi::gpu::api::gpuFree(host_fas);
                simbi::gpu::api::gpuFree(host_x1m);
                simbi::gpu::api::gpuFree(host_source0);
                simbi::gpu::api::gpuFree(host_sourceD);
                simbi::gpu::api::gpuFree(host_sourceS);
                simbi::gpu::api::gpuFree(host_pressure_guess);
                // printf("Memory Freed.\n");
            };

            T *host_prims;
            C *host_u0;
            real            *host_pressure_guess;
            real            *host_source0;
            real            *host_sourceD;
            real            *host_sourceS;
            real            *host_dtmin;
            real            *host_dx1; 
            real            *host_x1m; 
            real            *host_fas; 
            real            *host_dV;
            CLattice1D      *host_clattice;

            real host_dt;
            real host_xmin;
            real host_xmax;

            void copyHostToDev(const U &host, U *device);
            void copyDevToHost(const U *device, U &host);
        }; // End GPU/CPU 1D DualSpace

        template <typename T, typename C> 
        struct DualSpace1D<T, C, Newtonian1D>
        {
            DualSpace1D() {};
            ~DualSpace1D()
            {
                // printf("\nFreeing Device Memory...\n");
                simbi::gpu::api::gpuFree(host_u0);
                simbi::gpu::api::gpuFree(host_prims);
                simbi::gpu::api::gpuFree(host_clattice);
                simbi::gpu::api::gpuFree(host_dV);
                simbi::gpu::api::gpuFree(host_dx1);
                simbi::gpu::api::gpuFree(host_fas);
                simbi::gpu::api::gpuFree(host_x1m);
                // printf("Memory Freed.\n");
            };

            T *host_prims;
            C *host_u0;
            real            *host_pressure_guess;
            real            *host_source0;
            real            *host_sourceRho;
            real            *host_sourceM;
            real            *host_dtmin;
            real            *host_dx1; 
            real            *host_x1m; 
            real            *host_fas; 
            real            *host_dV;
            CLattice1D      *host_clattice;

            real host_dt;
            real host_xmin;
            real host_xmax;

            void copyHostToDev(const Newtonian1D &host,   Newtonian1D *device);
            void copyDevToHost(const Newtonian1D *device, Newtonian1D &host);
        }; // End GPU/CPU 1D DualSpace

        template <typename T, typename C, typename U>
        struct DualSpace2D
        {
            bool d_all_zeros, s1_all_zeros, s2_all_zeros, e_all_zeros;
            DualSpace2D() {
                d_all_zeros  = true;
                s1_all_zeros = true;
                s2_all_zeros = true;
                e_all_zeros  = true;
            };

            ~DualSpace2D()
            {
                // printf("\nFreeing Device Memory...\n");
                simbi::gpu::api::gpuFree(host_u0);
                simbi::gpu::api::gpuFree(host_prims);
                simbi::gpu::api::gpuFree(host_pressure_guess);
                simbi::gpu::api::gpuFree(host_clattice);
                if(!e_all_zeros)  simbi::gpu::api::gpuFree(host_source0);
                if(!d_all_zeros)  simbi::gpu::api::gpuFree(host_sourceD);
                if(!s1_all_zeros) simbi::gpu::api::gpuFree(host_sourceS1);
                if(!s2_all_zeros) simbi::gpu::api::gpuFree(host_sourceS2);
                
                // printf("Memory Freed Successfully.\n");
            };

            

            T *host_prims;
            C *host_u0;
            real            *host_pressure_guess;
            real            *host_source0;
            real            *host_sourceD;
            real            *host_sourceS1;
            real            *host_sourceS2;
            real            *host_dtmin;
            real            *host_dx1, *host_x1m, *host_fas1, *host_dV1;
            real            *host_dx2, *host_cot, *host_fas2, *host_dV2;
            CLattice2D      *host_clattice;

            real host_dt;
            real host_xmin;
            real host_xmax;
            real host_ymin;
            real host_ymax;
            real host_dx;

            void copyHostToDev(const U &host, U *device);
            void copyDevToHost(const U *device, U &host);
        }; // End GPU/CPU 2D DualSpace

        template <typename T, typename C>
        struct DualSpace2D<T, C, Newtonian2D>
        {
            bool rho_all_zeros, m1_all_zeros, m2_all_zeros, e_all_zeros;
            DualSpace2D() {
                rho_all_zeros = true;
                m1_all_zeros  = true;
                m2_all_zeros  = true;
                e_all_zeros   = true;
            };

            ~DualSpace2D()
            {
                simbi::gpu::api::gpuFree(host_u0);
                simbi::gpu::api::gpuFree(host_prims);
                simbi::gpu::api::gpuFree(host_pressure_guess);
                simbi::gpu::api::gpuFree(host_clattice);
                if(!e_all_zeros)  simbi::gpu::api::gpuFree(host_source0);
                if(!rho_all_zeros)  simbi::gpu::api::gpuFree(host_sourceRho);
                if(!m1_all_zeros) simbi::gpu::api::gpuFree(host_sourceM1);
                if(!m2_all_zeros) simbi::gpu::api::gpuFree(host_sourceM2);
            };

            

            T *host_prims;
            C *host_u0;
            real            *host_pressure_guess;
            real            *host_source0;
            real            *host_sourceRho;
            real            *host_sourceM1;
            real            *host_sourceM2;
            real            *host_dtmin;
            real            *host_dx1, *host_x1m, *host_fas1, *host_dV1;
            real            *host_dx2, *host_cot, *host_fas2, *host_dV2;
            CLattice2D      *host_clattice;

            real host_dt;
            real host_xmin;
            real host_xmax;
            real host_ymin;
            real host_ymax;
            real host_dx;

            void copyHostToDev(const Newtonian2D &host,   Newtonian2D *device);
            void copyDevToHost(const Newtonian2D *device, Newtonian2D &host);
        }; // End GPU/CPU 2D DualSpace

        template<typename T, typename U, typename V>
        struct DualSpace3D
        {
            DualSpace3D() {};
            ~DualSpace3D()
            {
                // printf("\nFreeing Device Memory...\n");
                simbi::gpu::api::gpuFree(host_u0);
                simbi::gpu::api::gpuFree(host_prims);
                simbi::gpu::api::gpuFree(host_clattice);
                simbi::gpu::api::gpuFree(host_dV1);
                simbi::gpu::api::gpuFree(host_dV2);
                simbi::gpu::api::gpuFree(host_dV3);
                simbi::gpu::api::gpuFree(host_dx1);
                simbi::gpu::api::gpuFree(host_dx2);
                simbi::gpu::api::gpuFree(host_dx3);
                simbi::gpu::api::gpuFree(host_fas1);
                simbi::gpu::api::gpuFree(host_fas2);
                simbi::gpu::api::gpuFree(host_fas3);
                simbi::gpu::api::gpuFree(host_x1m);
                simbi::gpu::api::gpuFree(host_cot);
                simbi::gpu::api::gpuFree(host_source0);
                simbi::gpu::api::gpuFree(host_sourceD);
                simbi::gpu::api::gpuFree(host_sourceS1);
                simbi::gpu::api::gpuFree(host_sourceS2);
                simbi::gpu::api::gpuFree(host_sourceS3);
                simbi::gpu::api::gpuFree(host_pressure_guess);
                // printf("Memory Freed.\n");
            };

            T *host_prims;
            U *host_u0;
            real            *host_pressure_guess;
            real            *host_source0;
            real            *host_sourceD;
            real            *host_sourceS1;
            real            *host_sourceS2;
            real            *host_sourceS3;
            real            *host_dtmin;
            real            *host_dx1, *host_x1m, *host_fas1, *host_dV1, *host_dx3, *host_sin;
            real            *host_dx2, *host_cot, *host_fas2, *host_dV2, *host_dV3, *host_fas3;
            CLattice2D      *host_clattice;

            real host_dt;
            real host_xmin;
            real host_xmax;
            real host_ymin;
            real host_ymax;
            real host_zmin;
            real host_zmax;

            void copyHostToDev(const V &host, V *device);
            void copyDevToHost(const V *device, V &host);
        }; // End 3D DualSpace

    } // namespace dual
    
} // namespace simbi


#include "dual.tpp"
#endif