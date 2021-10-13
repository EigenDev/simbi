#ifndef DUAL_HPP
#define DUAL_HPP

#include "build_options.hpp"
#include "common/clattice1D.hpp"
#include "common/clattice2D.hpp"
#include "common/clattice3D.hpp"
#include "device_api.hpp"


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
                printf("\nFreeing Device Memory...\n");
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
                printf("Memory Freed.\n");
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

        template <typename T, typename C, typename U>
        struct DualSpace2D
        {
            DualSpace2D() {};
            ~DualSpace2D()
            {
                printf("\nFreeing Device Memory...\n");
                simbi::gpu::api::gpuFree(host_u0);
                simbi::gpu::api::gpuFree(host_prims);
                simbi::gpu::api::gpuFree(host_clattice);
                simbi::gpu::api::gpuFree(host_dV1);
                simbi::gpu::api::gpuFree(host_dV2);
                simbi::gpu::api::gpuFree(host_dx1);
                simbi::gpu::api::gpuFree(host_dx2);
                simbi::gpu::api::gpuFree(host_fas1);
                simbi::gpu::api::gpuFree(host_fas2);
                simbi::gpu::api::gpuFree(host_x1m);
                simbi::gpu::api::gpuFree(host_cot);
                simbi::gpu::api::gpuFree(host_source0);
                simbi::gpu::api::gpuFree(host_sourceD);
                simbi::gpu::api::gpuFree(host_sourceS1);
                simbi::gpu::api::gpuFree(host_sourceS2);
                simbi::gpu::api::gpuFree(host_pressure_guess);
                printf("Memory Freed Successfully.\n");
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

        template<typename T, typename U, typename V>
        struct DualSpace3D
        {
            DualSpace3D() {};
            ~DualSpace3D()
            {
                printf("\nFreeing Device Memory...\n");
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
                printf("Memory Freed.\n");
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