/**
 *  *=============================================================================
 *  *           SIMBI - Special Relativistic Magnetohydrodynamics Code
 *  *=============================================================================
 *  *
 *  * @file            io_manager.hpp
 *  * @brief
 *  * @details
 *  *
 *  * @version         0.8.0
 *  * @date            2025-02-23
 *  * @author          Marcus DuPont
 *  * @email           marcus.dupont@princeton.edu
 *  *
 *  *==============================================================================
 *  * @build           Requirements & Dependencies
 *  *==============================================================================
 *  * @requires        C++20
 *  * @depends         CUDA >= 11.0, HDF5 >= 1.12, OpenMP >= 4.5
 *  * @platform        Linux, MacOS
 *  * @parallel        GPU (CUDA, HIP), CPU (OpenMP)
 *  *
 *  *==============================================================================
 *  * @documentation   Reference & Notes
 *  *==============================================================================
 *  * @usage
 *  * @note
 *  * @warning
 *  * @todo
 *  * @bug
 *  * @performance
 *  *
 *  *==============================================================================
 *  * @testing        Quality Assurance
 *  *==============================================================================
 *  * @test
 *  * @benchmark
 *  * @validation
 *  *
 *  *==============================================================================
 *  * @history        Version History
 *  *==============================================================================
 *  * 2025-02-23      v0.8.0      Initial implementation
 *  *
 *  *==============================================================================
 *  * @copyright (C) 2025 Marcus DuPont. All rights reserved.
 *  *==============================================================================
 */
#ifndef IO_MANAGER_HPP
#define IO_MANAGER_HPP

#include "core/managers/solver_manager.hpp"         // for SolverManager
#include "core/types/utility/init_conditions.hpp"   // for InitialConditions
#include <dlfcn.h>   // for dlopen, dlclose, dlsym
#include <string>
namespace simbi {

    class IOManager
    {
      private:
        // const SolverManager& solver_manager_;
        std::string data_directory_;
        std::string hydro_source_lib_;
        std::string gravity_source_lib_;
        std::string boundary_source_lib_;
        size_type current_iter_{0};
        size_type checkpoint_zones_;
        size_type checkpoint_idx_{0};

        // library handles
        void* hsource_handle = nullptr;
        void* gsource_handle = nullptr;
        void* bsource_handle = nullptr;

      public:
        IOManager(
            const SolverManager& solver_manager,
            const InitialConditions& init
        )
            :   // solver_manager_(solver_manager),
              data_directory_(init.data_directory),
              hydro_source_lib_(init.hydro_source_lib),
              gravity_source_lib_(init.gravity_source_lib),
              boundary_source_lib_(init.boundary_source_lib),
              checkpoint_zones_(determine_checkpoint_zones(init))
        {
        }

        ~IOManager()
        {
            if (hsource_handle) {
                dlclose(hsource_handle);
            }
            if (gsource_handle) {
                dlclose(gsource_handle);
            }
            if (bsource_handle) {
                dlclose(bsource_handle);
            }
        }

        void load_functions()
        {
            // Load the symbol based on the dimension
            // using f2arg = void (*)(real, real, real[]);
            // using f3arg = void (*)(real, real, real, real[]);
            // using f4arg = void (*)(real, real, real, real, real[]);

            // //=================================================================
            // // Check if the hydro source library is set
            // //=================================================================
            // if (!hydro_source_lib_.empty()) {
            //     // Load the shared library
            //     hsource_handle = dlopen(hydro_source_lib_.c_str(),
            //     RTLD_LAZY); if (!hsource_handle) {
            //         std::cerr << "Cannot open library: " << dlerror() <<
            //         '\n'; return;
            //     }

            //     // Clear any existing error
            //     dlerror();

            //     const std::vector<std::pair<const char*, function_t&>>
            //     symbols =
            //         {
            //           {"hydro_source", hydro_source},
            //         };

            //     bool success = true;
            //     for (const auto& [symbol, func] : symbols) {
            //         void* source            = dlsym(hsource_handle, symbol);
            //         const char* dlsym_error = dlerror();
            //         // if can't load symbol, print error and
            //         // set null_sources to true
            //         if (dlsym_error) {
            //             std::cerr << "Cannot load symbol '" << symbol
            //                       << "': " << dlsym_error << '\n';
            //             success = false;
            //             dlclose(hsource_handle);
            //             break;
            //         }

            //         // Assign the function pointer based on the dimension
            //         if constexpr (dim == 1) {
            //             func = reinterpret_cast<f2arg>(source);
            //         }
            //         else if constexpr (dim == 2) {
            //             func = reinterpret_cast<f3arg>(source);
            //         }
            //         else if constexpr (dim == 3) {
            //             func = reinterpret_cast<f4arg>(source);
            //         }
            //     }
            //     if (success) {
            //         solver_manager_.null_sources_ = false;
            //     }
            // }

            // //=================================================================
            // // Check if the gravity source library is set
            // //=================================================================
            // if (!gravity_source_lib_.empty()) {
            //     gsource_handle = dlopen(gravity_source_lib_.c_str(),
            //     RTLD_LAZY); if (!gsource_handle) {
            //         std::cerr << "Cannot open library: " << dlerror() <<
            //         '\n'; return;
            //     }

            //     // Clear any existing error
            //     dlerror();

            //     // Load the symbol based on the dimension
            //     const std::vector<std::pair<const char*, function_t&>>
            //         g_symbols = {
            //           {"gravity_source", gravity_source},
            //         };

            //     bool success = true;
            //     for (const auto& [symbol, func] : g_symbols) {
            //         void* source            = dlsym(gsource_handle, symbol);
            //         const char* dlsym_error = dlerror();
            //         // if can't load symbol, print error
            //         if (dlsym_error) {
            //             std::cerr << "Cannot load symbol '" << symbol
            //                       << "': " << dlsym_error << '\n';
            //             success = false;
            //             dlclose(gsource_handle);
            //             break;
            //         }

            //         // Assign the function pointer based on the dimension
            //         if constexpr (dim == 1) {
            //             func = reinterpret_cast<f2arg>(source);
            //         }
            //         else if constexpr (dim == 2) {
            //             func = reinterpret_cast<f3arg>(source);
            //         }
            //         else if constexpr (dim == 3) {
            //             func = reinterpret_cast<f4arg>(source);
            //         }
            //     }
            //     if (success) {
            //         solver_manaer_.null_gravity_ = false;
            //     }
            // }

            // //=================================================================
            // // Check if the boundary source library is set
            // //=================================================================
            // if (!boundary_source_lib_.empty()) {
            //     bsource_handle =
            //         dlopen(boundary_source_lib_.c_str(), RTLD_LAZY);
            //     if (!bsource_handle) {
            //         std::cerr << "Cannot open library: " << dlerror() <<
            //         '\n'; return;
            //     }

            //     // Clear any existing error
            //     dlerror();

            //     // Load the symbol based on the dimension
            //     const std::vector<std::pair<const char*, function_t&>>
            //         b_symbols = {
            //           {"bx1_inner_source", bx1_inner_source},
            //           {"bx1_outer_source", bx1_outer_source},
            //           {"bx2_inner_source", bx2_inner_source},
            //           {"bx2_outer_source", bx2_outer_source},
            //           {"bx3_inner_source", bx3_inner_source},
            //           {"bx3_outer_source", bx3_outer_source},
            //         };

            //     for (const auto& [symbol, func] : b_symbols) {
            //         void* source            = dlsym(bsource_handle, symbol);
            //         const char* dlsym_error = dlerror();
            //         // if can't load symbol, print error
            //         if (dlsym_error) {
            //             // erro out  only  if the boundary
            //             // condition is set to dynamic
            //             for (int i = 0; i < 2 * dim; ++i) {
            //                 if (symbol == b_symbols[i].first &&
            //                     bcs[i] == BoundaryCondition::DYNAMIC) {
            //                     std::cerr << "Cannot load symbol '" << symbol
            //                               << "': " << dlsym_error << '\n';
            //                     bcs[i] = BoundaryCondition::OUTFLOW;
            //                     dlclose(bsource_handle);
            //                 }
            //             }
            //         }
            //         else {
            //             // Assign the function pointer based on the dimension
            //             if constexpr (dim == 1) {
            //                 func = reinterpret_cast<f2arg>(source);
            //             }
            //             else if constexpr (dim == 2) {
            //                 func = reinterpret_cast<f3arg>(source);
            //             }
            //             else if constexpr (dim == 3) {
            //                 func = reinterpret_cast<f4arg>(source);
            //             }
            //         }
            //     }
            // }
        }

        void increment_iter() { current_iter_++; }
        void increment_checkpoint_idx() { checkpoint_idx_++; }

        // accessors
        auto& data_directory() const { return data_directory_; }
        auto& data_directory() { return data_directory_; }
        auto& hydro_source_lib() const { return hydro_source_lib_; }
        auto& hydro_source_lib() { return hydro_source_lib_; }
        auto& gravity_source_lib() const { return gravity_source_lib_; }
        auto& gravity_source_lib() { return gravity_source_lib_; }
        auto& boundary_source_lib() const { return boundary_source_lib_; }
        auto& boundary_source_lib() { return boundary_source_lib_; }
        auto current_iter() const { return current_iter_; }
        auto checkpoint_zones() const { return checkpoint_zones_; }
        auto checkpoint_idx() const { return checkpoint_idx_; }

      private:
        size_type determine_checkpoint_zones(const InitialConditions& init)
        {
            const auto [xag, yag, zag] = init.active_zones();
            return (zag > 1) ? zag : (yag > 1) ? yag : xag;
        }
    };
}   // namespace simbi

#endif