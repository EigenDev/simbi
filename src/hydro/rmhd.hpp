/**
 * ***********************(C) COPYRIGHT 2024 Marcus DuPont**********************
 * @file       rmhd.hpp
 * @brief      Single header for 1, 2, and 3D RMHD calculations
 *
 * @note
 * @history:
 *   Version   Date            Author          Modification    Email
 *   V0.8.0    Dec-03-2023     Marcus DuPont marcus.dupont@princeton.edu
 *
 * @verbatim
 * ==============================================================================
 *
 * ==============================================================================
 * @endverbatim
 * ***********************(C) COPYRIGHT 2024 Marcus DuPont**********************
 */

#ifndef RMHD_HPP
#define RMHD_HPP

#include "base.hpp"                   // for HydroBase
#include "build_options.hpp"          // for real, HD, lint, luint
#include "common/enums.hpp"           // for TIMESTEP_TYPE
#include "common/helpers.hpp"         // for my_min, my_max, ...
#include "common/hydro_structs.hpp"   // for Conserved, Primitive
#include "common/mesh.hpp"            // for Mesh
#include "util/exec_policy.hpp"       // for ExecutionPolicy
#include "util/ndarray.hpp"           // for ndarray
#include <dlfcn.h>                    // for dlopen, dlclose, dlsym
#include <functional>                 // for function
#include <optional>                   // for optional
#include <type_traits>                // for conditional_t
#include <vector>                     // for vector

namespace simbi {
    template <int dim>
    struct RMHD : public HydroBase,
                  public Mesh<
                      RMHD<dim>,
                      dim,
                      anyConserved<dim, Regime::RMHD>,
                      anyPrimitive<dim, Regime::RMHD>> {
        static constexpr int dimensions          = dim;
        static constexpr int nvars               = dim + 3;
        static constexpr std::string_view regime = "srmhd";

        // set the primitive and conservative types at compile time
        using primitive_t   = anyPrimitive<dim, Regime::RMHD>;
        using conserved_t   = anyConserved<dim, Regime::RMHD>;
        using eigenvals_t   = Eigenvals<dim, Regime::RMHD>;
        using mag_fourvec_t = mag_four_vec<dim>;
        using function_t    = typename helpers::real_func<dim>::type;

        // hydrodynamic source functions
        function_t dens_source;
        function_t mom1_source;
        function_t mom2_source;
        function_t mom3_source;
        function_t ener_source;
        function_t b1_source;
        function_t b2_source;
        function_t b3_source;
        function_t chi_source;

        // gravity source functions
        function_t g1_func;
        function_t g2_func;
        function_t g3_func;

        // boundary source functions
        function_t bx1_inner_func;
        function_t bx1_outer_func;
        function_t bx2_inner_func;
        function_t bx2_outer_func;
        function_t bx3_inner_func;
        function_t bx3_outer_func;

        // library handles
        void* hsource_handle = nullptr;
        void* gsource_handle = nullptr;
        void* bsource_handle = nullptr;

        template <typename T>
        using RiemannFuncPointer = conserved_t (T::*)(
            const primitive_t&,
            const primitive_t&,
            const luint,
            const real
        ) const;
        RiemannFuncPointer<RMHD<dim>> riemann_solve;

        std::vector<function_t> bsources;   // boundary sources
        std::vector<function_t> hsources;   // hydro sources
        std::vector<function_t> gsources;   // gravity sources

        /* Shared Data Members */
        ndarray<primitive_t> prims;
        ndarray<conserved_t> cons, fri, gri, hri;
        ndarray<real> dt_min, bstag1, bstag2, bstag3;

        RMHD();
        RMHD(
            std::vector<std::vector<real>>& state,
            const InitialConditions& init_conditions
        );

        ~RMHD();

        /* Methods */
        void cons2prim();
        DUAL primitive_t cons2prim(const conserved_t& cons) const;
        void set_flux_and_fields();
        void riemann_fluxes();
        void advance();

        DUAL void calc_max_wave_speeds(
            const primitive_t& prims,
            const luint nhat,
            real speeds[4]
        ) const;

        DUAL eigenvals_t calc_eigenvals(
            const primitive_t& primsL,
            const primitive_t& primsR,
            const luint nhat
        ) const;

        DUAL conserved_t prims2cons(const primitive_t& prims) const;

        DUAL conserved_t calc_hlle_flux(
            const primitive_t& prL,
            const primitive_t& prR,
            const luint nhat,
            const real vface
        ) const;

        DUAL conserved_t calc_hllc_flux(
            const primitive_t& prL,
            const primitive_t& prR,
            const luint nhat,
            const real vface
        ) const;

        DUAL conserved_t calc_hlld_flux(
            const primitive_t& prL,
            const primitive_t& prR,
            const luint nhat,
            const real vface
        ) const;

        DUAL real div_b(
            const auto b1L,
            const auto b1R,
            const auto b2L,
            const auto b2R,
            const auto b3L,
            const auto b3R,
            const auto& cell
        ) const;

        DUAL void set_riemann_solver()
        {
            switch (sim_solver) {
                case Solver::HLLE:
                    this->riemann_solve = &RMHD<dim>::calc_hlle_flux;
                    break;
                case Solver::HLLC:
                    this->riemann_solve = &RMHD<dim>::calc_hllc_flux;
                    break;
                default:
                    this->riemann_solve = &RMHD<dim>::calc_hlld_flux;
                    break;
            }
        }

        void init_riemann_solver()
        {
            SINGLE(helpers::hybrid_set_riemann_solver, this);
        }

        DUAL conserved_t
        prims2flux(const primitive_t& prims, const luint nhat) const;

        template <TIMESTEP_TYPE dt_type = TIMESTEP_TYPE::ADAPTIVE>
        void adapt_dt();

        void simulate(
            std::function<real(real)> const& a,
            std::function<real(real)> const& adot,
            const std::vector<std::optional<function_t>>& boundary_sources,
            const std::vector<std::optional<function_t>>& hydro_sources,
            const std::vector<std::optional<function_t>>& gravity_sources
        );

        DUAL real curl_e(
            const luint nhat,
            const real ej[4],
            const real ek[4],
            const auto& cell,
            const int side
        ) const;

        /**
         * @brief
         * @retval
         */
        template <Plane P, Corner C>
        DUAL real calc_edge_emf(
            const conserved_t& fw,
            const conserved_t& fe,
            const conserved_t& fs,
            const conserved_t& fn,
            const primitive_t* prims,
            const luint ii,
            const luint jj,
            const luint kk,
            const luint ia,
            const luint ja,
            const luint ka,
            const luint nhat,
            const real bw = 0.0,
            const real be = 0.0,
            const real bs = 0.0,
            const real bn = 0.0
        ) const;

        void offload()
        {
            cons.copyToGpu();
            prims.copyToGpu();
            dt_min.copyToGpu();
            object_pos.copyToGpu();
            bcs.copyToGpu();
            troubled_cells.copyToGpu();
            bstag1.copyToGpu();
            bstag2.copyToGpu();
            bstag3.copyToGpu();
            fri.copyToGpu();
            gri.copyToGpu();
            hri.copyToGpu();
        }

        DUAL real hlld_vdiff(
            const real p,
            const conserved_t r[2],
            const real lam[2],
            const real bn,
            const luint nhat,
            primitive_t& praL,
            primitive_t& praR,
            primitive_t& prC
        ) const;

        DUAL conserved_t hydro_sources(const auto& cell) const;

        DUAL conserved_t
        gravity_sources(const primitive_t& prims, const auto& cell) const;

        void load_functions()
        {
            // Load the symbol based on the dimension
            using f2arg = real (*)(real, real);
            using f3arg = real (*)(real, real, real);
            using f4arg = real (*)(real, real, real, real);

            //=================================================================
            // Check if the hydro source library is set
            //=================================================================
            null_sources = true;
            if (!hydro_source_lib.empty()) {
                // Load the shared library
                hsource_handle = dlopen(hydro_source_lib.c_str(), RTLD_LAZY);
                if (!hsource_handle) {
                    std::cerr << "Cannot open library: " << dlerror() << '\n';
                    return;
                }

                // Clear any existing error
                dlerror();

                const std::vector<std::pair<const char*, function_t&>> symbols =
                    {{"dens_source", dens_source},
                     {"mom1_source", mom1_source},
                     {"mom2_source", mom2_source},
                     {"mom3_source", mom3_source},
                     {"ener_source", ener_source},
                     {"b1_source", b1_source},
                     {"b2_source", b2_source},
                     {"b3_source", b3_source},
                     {"chi_source", chi_source}};

                bool success = true;
                for (const auto& [symbol, func] : symbols) {
                    void* source            = dlsym(hsource_handle, symbol);
                    const char* dlsym_error = dlerror();
                    // if can't load symbol, print error and
                    // set null_sources to true
                    if (dlsym_error) {
                        std::cerr << "Cannot load symbol '" << symbol
                                  << "': " << dlsym_error << '\n';
                        success = false;
                        dlclose(hsource_handle);
                        break;
                    }

                    // Assign the function pointer based on the dimension
                    if constexpr (dim == 1) {
                        func = reinterpret_cast<f2arg>(source);
                    }
                    else if constexpr (dim == 2) {
                        func = reinterpret_cast<f3arg>(source);
                    }
                    else if constexpr (dim == 3) {
                        func = reinterpret_cast<f4arg>(source);
                    }
                }
                if (success) {
                    null_sources = false;
                }
            }

            //=================================================================
            // Check if the gravity source library is set
            //=================================================================
            null_gravity = true;
            if (!gravity_source_lib.empty()) {
                gsource_handle = dlopen(gravity_source_lib.c_str(), RTLD_LAZY);
                if (!gsource_handle) {
                    std::cerr << "Cannot open library: " << dlerror() << '\n';
                    return;
                }

                // Clear any existing error
                dlerror();

                // Load the symbol based on the dimension
                const std::vector<std::pair<const char*, function_t&>>
                    g_symbols = {
                      {"g1_func", g1_func},
                      {"g2_func", g2_func},
                      {"g3_func", g3_func}
                    };

                bool success = true;
                for (const auto& [symbol, func] : g_symbols) {
                    void* source            = dlsym(gsource_handle, symbol);
                    const char* dlsym_error = dlerror();
                    // if can't load symbol, print error
                    if (dlsym_error) {
                        std::cerr << "Cannot load symbol '" << symbol
                                  << "': " << dlsym_error << '\n';
                        success = false;
                        dlclose(gsource_handle);
                        break;
                    }

                    // Assign the function pointer based on the dimension
                    if constexpr (dim == 1) {
                        func = reinterpret_cast<f2arg>(source);
                    }
                    else if constexpr (dim == 2) {
                        func = reinterpret_cast<f3arg>(source);
                    }
                    else if constexpr (dim == 3) {
                        func = reinterpret_cast<f4arg>(source);
                    }
                }
                if (success) {
                    null_gravity = false;
                }
            }

            //=================================================================
            // Check if the boundary source library is set
            //=================================================================
            if (!boundary_source_lib.empty()) {
                bsource_handle = dlopen(boundary_source_lib.c_str(), RTLD_LAZY);
                if (!bsource_handle) {
                    std::cerr << "Cannot open library: " << dlerror() << '\n';
                    return;
                }

                // Clear any existing error
                dlerror();

                // Load the symbol based on the dimension
                const std::vector<std::pair<const char*, function_t&>>
                    b_symbols = {
                      {"bx1_inner_func", bx1_inner_func},
                      {"bx1_outer_func", bx1_outer_func},
                      {"bx2_inner_func", bx2_inner_func},
                      {"bx2_outer_func", bx2_outer_func},
                      {"bx3_inner_func", bx3_inner_func},
                      {"bx3_outer_func", bx3_outer_func}
                    };

                for (const auto& [symbol, func] : b_symbols) {
                    void* source            = dlsym(bsource_handle, symbol);
                    const char* dlsym_error = dlerror();
                    // if can't load symbol, print error
                    if (dlsym_error) {
                        // it is only unsuccessful if the boundary
                        // condition is set to dynamic
                        for (int i = 0; i < 6; ++i) {
                            if (symbol == b_symbols[i].first &&
                                bcs[i] == BoundaryCondition::DYNAMIC) {
                                std::cerr << "Cannot load symbol '" << symbol
                                          << "': " << dlsym_error << '\n';
                                bcs[i] = BoundaryCondition::OUTFLOW;
                                dlclose(bsource_handle);
                            }
                        }

                        // Assign the function pointer based on the dimension
                        if constexpr (dim == 1) {
                            func = reinterpret_cast<f2arg>(source);
                        }
                        else if constexpr (dim == 2) {
                            func = reinterpret_cast<f3arg>(source);
                        }
                        else if constexpr (dim == 3) {
                            func = reinterpret_cast<f4arg>(source);
                        }
                    }
                }
            }
        }
    };
}   // namespace simbi

template <>
struct is_relativistic_mhd<simbi::RMHD<1>> {

    static constexpr bool value = true;
};

template <>
struct is_relativistic_mhd<simbi::RMHD<2>> {
    static constexpr bool value = true;
};

template <>
struct is_relativistic_mhd<simbi::RMHD<3>> {
    static constexpr bool value = true;
};

#include "rmhd.ipp"
#endif