/**
 * ***********************(C) COPYRIGHT 2024 Marcus DuPont**********************
 * @file       srhd.hpp
 * @brief      single header for 1,2, and 3D SRHD calculations
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
#ifndef SRHD_HPP
#define SRHD_HPP

#include "build_options.hpp"        // for real, HD, lint, luint
#include "core/base.hpp"            // for HydroBase
#include "core/types/enums.hpp"     // for TIMESTEP_TYPE
#include "core/types/maybe.hpp"     // for Maybe
#include "core/types/ndarray.hpp"   // for ndarray
#include "geometry/mesh.hpp"        // for Mesh
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
    struct SRHD : public HydroBase,
                  public Mesh<
                      SRHD<dim>,
                      dim,
                      anyConserved<dim, Regime::SRHD>,
                      anyPrimitive<dim, Regime::SRHD>> {
        static constexpr int dimensions          = dim;
        static constexpr int nvars               = dim + 3;
        static constexpr std::string_view regime = "srhd";
        // set the primitive and conservative types at compile time
        using primitive_t = anyPrimitive<dim, Regime::SRHD>;
        using conserved_t = anyConserved<dim, Regime::SRHD>;
        using eigenvals_t = Eigenvals<dim, Regime::SRHD>;
        using function_t  = typename helpers::real_func<dim>::type;
        template <typename T>
        using RiemannFuncPointer = conserved_t (T::*)(
            const primitive_t&,
            const primitive_t&,
            const luint,
            const real
        ) const;
        RiemannFuncPointer<SRHD<dim>> riemann_solve;

        ndarray<function_t> bsources;   // boundary sources
        ndarray<function_t> hsources;   // hydro sources
        ndarray<function_t> gsources;   // gravity sources

        /* Shared Data Members */
        ndarray<Maybe<primitive_t>, dim> prims;
        ndarray<conserved_t, dim> cons;
        ndarray<real, dim> pressure_guess, dt_min;

        // library handles
        void* hsource_handle = nullptr;
        void* gsource_handle = nullptr;
        void* bsource_handle = nullptr;

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

        /* Methods */
        SRHD();
        SRHD(
            std::vector<std::vector<real>>& state,
            InitialConditions& init_conditions
        );
        ~SRHD();

        void cons2prim();

        void advance();

        DUAL eigenvals_t calc_eigenvals(
            const auto& primsL,
            const auto& primsR,
            const luint nhat
        ) const;

        DUAL conserved_t calc_hllc_flux(
            const auto& prL,
            const auto& prR,
            const luint nhat,
            const real vface = 0.0
        ) const;

        DUAL conserved_t calc_hlle_flux(
            const auto& prL,
            const auto& prR,
            const luint nhat,
            const real vface = 0.0
        ) const;

        DUAL void set_riemann_solver()
        {
            switch (sim_solver) {
                case Solver::HLLE:
                    this->riemann_solve = &SRHD<dim>::calc_hlle_flux;
                    break;
                default:
                    this->riemann_solve = &SRHD<dim>::calc_hllc_flux;
                    break;
            }
        }

        void init_riemann_solver()
        {
            SINGLE(helpers::hybrid_set_riemann_solver, this);
        }

        template <TIMESTEP_TYPE dt_type = TIMESTEP_TYPE::ADAPTIVE>
        void adapt_dt();

        template <TIMESTEP_TYPE dt_type = TIMESTEP_TYPE::ADAPTIVE>
        void adapt_dt(const ExecutionPolicy<>& p);

        void simulate(
            std::function<real(real)> const& a,
            std::function<real(real)> const& adot
        );

        void offload()
        {
            cons.sync_to_device();
            prims.sync_to_device();
            pressure_guess.sync_to_device();
            dt_min.sync_to_device();
            if constexpr (dim > 1) {
                object_pos.sync_to_device();
            }
            bcs.sync_to_device();
        }

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
                        for (int i = 0; i < 2 * dim; ++i) {
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

        auto calc_star_state(
            const auto& uL,
            const auto& uR,
            const auto& fL,
            const auto& fR,
            const real aL,
            const real aR,
            const luint nhat
        ) const -> std::pair<real, real>;

      private:
        DUAL conserved_t compute_star_state(
            const auto& prim,
            const auto& cons,
            const real a,
            const real aStar,
            const real pStar,
            const luint nhat
        ) const
        {
            const auto& mom     = cons.momentum();
            const auto vnorm    = prim.proper_velocity(nhat);
            const auto& p       = prim.press();
            const auto& d       = cons.dens();
            const auto& chi     = cons.chi();
            const auto e        = cons.nrg() + d;
            const auto cofactor = 1.0 / (a - aStar);
            const auto kdvec    = unit_vectors::get<dim>(nhat);

            const auto dStar   = cofactor * (a - vnorm) * d;
            const auto chiStar = cofactor * (a - vnorm) * chi;
            const auto momStar =
                (mom * (a - vnorm) + kdvec * (-p + pStar)) * cofactor;
            const auto eStar =
                cofactor * (e * (a - vnorm) + pStar * aStar - p * vnorm);

            return conserved_t{dStar, momStar, eStar - dStar, chiStar};
        }

        DUAL conserved_t apply_hllc(
            const auto& star_state,
            const auto& flux,
            const auto& cons,
            const real a,
            const real vface,
            const auto& prL,
            const auto& prR
        ) const
        {
            auto hllc_flux =
                flux + (star_state - cons) * a - star_state * vface;

            // Upwind concentration
            hllc_flux.chi() = (hllc_flux.dens() < 0.0)
                                  ? prR.chi() * hllc_flux.dens()
                                  : prL.chi() * hllc_flux.dens();
            return hllc_flux;
        }
    };
}   // namespace simbi

template <>
struct is_relativistic<simbi::SRHD<1>> {

    static constexpr bool value = true;
};

template <>
struct is_relativistic<simbi::SRHD<2>> {
    static constexpr bool value = true;
};

template <>
struct is_relativistic<simbi::SRHD<3>> {
    static constexpr bool value = true;
};

#include "srhd.ipp"
#endif