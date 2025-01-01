/**
 * ***********************(C) COPYRIGHT 2024 Marcus DuPont**********************
 * @file       newt.hpp
 * @brief      single header for 1, 2, adn 3D Newtonian calculations
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

#ifndef NEWT_HPP
#define NEWT_HPP

#include "base.hpp"                   // for HydroBase
#include "build_options.hpp"          // for real, DUAL, lint, luint
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
    struct Newtonian : public HydroBase,
                       public Mesh<
                           Newtonian<dim>,
                           dim,
                           anyConserved<dim, Regime::NEWTONIAN>,
                           anyPrimitive<dim, Regime::NEWTONIAN>> {
        static constexpr int dimensions          = dim;
        static constexpr int nvars               = dim + 3;
        static constexpr std::string_view regime = "classical";
        // set the primitive and conservative types at compile time
        using primitive_t = anyPrimitive<dim, Regime::NEWTONIAN>;
        using conserved_t = anyConserved<dim, Regime::NEWTONIAN>;
        using eigenvals_t = Eigenvals<dim, Regime::NEWTONIAN>;
        using function_t  = typename helpers::real_func<dim>::type;
        template <typename T>
        using RiemannFuncPointer = conserved_t (T::*)(
            const primitive_t&,
            const primitive_t&,
            const luint,
            const real
        ) const;
        RiemannFuncPointer<Newtonian<dim>> riemann_solve;

        // boundary condition functions for mesh motion
        ndarray<function_t> bsources;   // boundary sources
        ndarray<function_t> hsources;   // hydro sources
        ndarray<function_t> gsources;   // gravity sources

        /* Shared Data Members */
        ndarray<primitive_t> prims;
        ndarray<conserved_t> cons;
        ndarray<real> dt_min;

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

        // Constructors
        Newtonian();

        // Overloaded Constructor
        Newtonian(
            std::vector<std::vector<real>>& state,
            const InitialConditions& init_conditions
        );

        // Destructor
        ~Newtonian();

        /* Methods */
        void cons2prim();
        void advance();

        DUAL eigenvals_t calc_eigenvals(
            const primitive_t& primsL,
            const primitive_t& primsR,
            const luint nhat
        ) const;

        DUAL conserved_t calc_hllc_flux(
            const primitive_t& prL,
            const primitive_t& prR,
            const luint nhat,
            const real vface = 0.0
        ) const;

        DUAL conserved_t calc_hlle_flux(
            const primitive_t& prL,
            const primitive_t& prR,
            const luint nhat,
            const real vface = 0.0
        ) const;

        DUAL void set_riemann_solver()
        {
            switch (sim_solver) {
                case Solver::HLLE:
                    this->riemann_solve = &Newtonian<dim>::calc_hlle_flux;
                    break;
                default:
                    this->riemann_solve = &Newtonian<dim>::calc_hllc_flux;
                    break;
            }
        }

        void init_riemann_solver()
        {
            SINGLE(helpers::hybrid_set_riemann_solver, this);
        }

        void adapt_dt();
        void adapt_dt(const ExecutionPolicy<>& p);

        void simulate(
            std::function<real(real)> const& a,
            std::function<real(real)> const& adot
        );

        void offload()
        {
            cons.copyToGpu();
            prims.copyToGpu();
            dt_min.copyToGpu();
            if constexpr (dim > 1) {
                object_pos.copyToGpu();
            }
            bcs.copyToGpu();
            troubled_cells.copyToGpu();
        }

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

        DUAL conserved_t hydro_sources(const auto& cell) const;

        DUAL conserved_t
        gravity_sources(const primitive_t& prims, const auto& cell) const;

        void check_sources()
        {
            // check if ~all~ boundary sources have been set.
            // if the user forgot one, the code will run with
            // and outflow outer boundary condition
            this->all_outer_bounds = std::all_of(
                this->bsources.begin(),
                this->bsources.end(),
                [](const auto& q) { return q != nullptr; }
            );

            this->null_gravity = std::all_of(
                this->gsources.begin(),
                this->gsources.end(),
                [](const auto& q) { return q == nullptr; }
            );

            this->null_sources = std::all_of(
                this->hsources.begin(),
                this->hsources.end(),
                [](const auto& q) { return q == nullptr; }
            );
        }
    };
}   // namespace simbi

template <>
struct is_relativistic<simbi::Newtonian<1>> {
    static constexpr bool value = false;
};

template <>
struct is_relativistic<simbi::Newtonian<2>> {
    static constexpr bool value = false;
};

template <>
struct is_relativistic<simbi::Newtonian<3>> {
    static constexpr bool value = false;
};

#include "newt.ipp"
#endif