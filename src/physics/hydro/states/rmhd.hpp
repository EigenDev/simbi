/**
 * ***********************(C) COPYRIGHT 2025 Marcus DuPont**********************
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
 * ***********************(C) COPYRIGHT 2025 Marcus DuPont**********************
 */

#ifndef RMHD_HPP
#define RMHD_HPP

#include "build_options.hpp"        // for real, HD, lint, luint
#include "core/base.hpp"            // for HydroBase
#include "core/types/enums.hpp"     // for TIMESTEP_TYPE
#include "core/types/maybe.hpp"     // for Maybe
#include "core/types/ndarray.hpp"   // for ndarray
#include "geometry/mesh.hpp"        // for Mesh
#include "physics/hydro/schemes/ct/ct_calculator.hpp"   // for anyPrimitive
#include "physics/hydro/types/generic_structs.hpp"   // for Eigenvals, mag_four_vec
#include "util/parallel/exec_policy.hpp"             // for ExecutionPolicy
#include "util/tools/helpers.hpp"                    // for my_min, my_max, ...
#include <dlfcn.h>       // for dlopen, dlclose, dlsym
#include <functional>    // for function
#include <optional>      // for optional
#include <type_traits>   // for conditional_t
#include <vector>        // for vector

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
        using primitive_t = anyPrimitive<dim, Regime::RMHD>;
        using conserved_t = anyConserved<dim, Regime::RMHD>;
        using eigenvals_t = Eigenvals<dim, Regime::RMHD>;
        using function_t  = typename helpers::real_func<dim>::type;
        using ct_scheme_t = std::conditional_t<
            comp_ct_type == CTTYPE::MdZ,
            ct::CTMdZ,
            ct::CTContact>;

        // hydrodynamic source functions
        function_t hydro_source;

        // gravity source functions
        function_t gravity_source;

        // boundary source functions at x1 boundaries
        function_t bx1_inner_source;
        function_t bx1_outer_source;
        // boundary source functions at x2 boundaries
        function_t bx2_inner_source;
        function_t bx2_outer_source;
        // boundary source functions at x3 boundaries
        function_t bx3_inner_source;
        function_t bx3_outer_source;

        // library handles
        void* hsource_handle = nullptr;
        void* gsource_handle = nullptr;
        void* bsource_handle = nullptr;

        template <typename T>
        using RiemannFuncPointer = conserved_t (T::*)(
            const primitive_t&,
            const primitive_t&,
            const luint,
            const real,
            const real
        ) const;
        RiemannFuncPointer<RMHD<dim>> riemann_solve;

        std::vector<function_t> bsources;   // boundary sources
        std::vector<function_t> hsources;   // hydro sources
        std::vector<function_t> gsources;   // gravity sources

        /* Shared Data Members */
        ndarray<Maybe<primitive_t>, dim> prims;
        ndarray<conserved_t, dim> cons;
        ndarray<conserved_t, dim> fri, gri, hri;
        ndarray<real, dim> bstag1, bstag2, bstag3;
        ndarray<real> dt_min;

        RMHD();
        RMHD(
            std::vector<std::vector<real>>& state,
            InitialConditions& init_conditions
        );

        ~RMHD();

        /* Methods */
        void cons2prim();
        DEV auto cons2prim(const auto& cons) const;
        void sync_flux_boundaries(const auto& flux_man);
        void sync_magnetic_boundaries(const auto& bfield_man);
        void riemann_fluxes();
        void advance(const auto& man, const auto& bfield_man);
        void advance_conserved();
        void advance_magnetic_fields();

        template <int nhat>
        void update_magnetic_component(
            const ExecutionPolicy<>& policy,
            const auto& prim_region
        );

        DUAL auto
        calc_max_wave_speeds(const auto& prims, const luint nhat) const;

        DUAL eigenvals_t calc_eigenvals(
            const auto& primsL,
            const auto& primsR,
            const luint nhat
        ) const;

        DUAL conserved_t calc_hlle_flux(
            const auto& prL,
            const auto& prR,
            const luint nhat,
            const real vface,
            const real bface
        ) const;

        DUAL conserved_t calc_hllc_flux(
            const auto& prL,
            const auto& prR,
            const luint nhat,
            const real vface,
            const real bface
        ) const;

        DUAL conserved_t calc_hlld_flux(
            const auto& prL,
            const auto& prR,
            const luint nhat,
            const real vface,
            const real bface
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

        template <TIMESTEP_TYPE dt_type = TIMESTEP_TYPE::ADAPTIVE>
        void adapt_dt();

        void simulate(
            std::function<real(real)> const& a,
            std::function<real(real)> const& adot
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
            const auto& fw,
            const auto& fe,
            const auto& fs,
            const auto& fn,
            const auto* prims,
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

        template <size_type i, size_type j>
        DUAL void calc_emf_edges(
            real ei[],
            real ej[],
            const auto& cell,
            size_type ii,
            size_type jj,
            size_type kk
        ) const;

        void offload()
        {
            cons.sync_to_device();
            prims.sync_to_device();
            dt_min.sync_to_device();
            object_pos.sync_to_device();
            bcs.sync_to_device();
            troubled_cells.sync_to_device();
            bstag1.sync_to_device();
            bstag2.sync_to_device();
            bstag3.sync_to_device();
            fri.sync_to_device();
            gri.sync_to_device();
            hri.sync_to_device();
        }

        DUAL real hlld_vdiff(
            const real p,
            const conserved_t r[2],
            const real lam[2],
            const real bn,
            const luint nhat,
            auto& praL,
            auto& praR,
            auto& prC
        ) const;

        DUAL conserved_t hydro_sources(const auto& cell) const;

        DUAL conserved_t
        gravity_sources(const auto& prims, const auto& cell) const;

        void load_functions()
        {
            // Load the symbol based on the dimension
            using f2arg = void (*)(real, real, real[]);
            using f3arg = void (*)(real, real, real, real[]);
            using f4arg = void (*)(real, real, real, real, real[]);

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
                    {
                      {"hydro_source", hydro_source},
                    };

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
                      {"gravity_source", gravity_source},
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
                      {"bx1_inner_source", bx1_inner_source},
                      {"bx1_outer_source", bx1_outer_source},
                      {"bx2_inner_source", bx2_inner_source},
                      {"bx2_outer_source", bx2_outer_source},
                      {"bx3_inner_source", bx3_inner_source},
                      {"bx3_outer_source", bx3_outer_source},
                    };

                for (const auto& [symbol, func] : b_symbols) {
                    void* source            = dlsym(bsource_handle, symbol);
                    const char* dlsym_error = dlerror();
                    // if can't load symbol, print error
                    if (dlsym_error) {
                        // erro out  only  if the boundary
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
                    }
                    else {
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