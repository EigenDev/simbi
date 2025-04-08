/**
 *  *=============================================================================
 *  *           SIMBI - Special Relativistic Magnetohydrodynamics Code
 *  *=============================================================================
 *  *
 *  * @file            io_manager.hpp
 *  * @brief           I/O manager for HDF5 file I/O and JIT compilation
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
#include "util/tools/helpers.hpp"                   // for real_func
#include <dlfcn.h>   // for dlopen, dlclose, dlsym
#include <string>
namespace simbi {

    template <size_type D>
    struct SourceParams {
        using type = std::conditional_t<
            D == 1,
            std::tuple<real, real, real*>,   // 1D: x1, t, res
            std::conditional_t<
                D == 2,
                std::tuple<real, real, real, real*>,   // 2D: x1, x2, t, res
                std::tuple<real, real, real, real, real*>>>;   // 3D: x1,
                                                               // x2, x3, t,
                                                               // res
    };

    template <size_type D, typename... Args>
    concept ValidSourceParams = requires(Args... args) {
        requires sizeof...(Args) == D + 2;
        requires(std::convertible_to<std::remove_reference_t<Args>, real> && ...
                ) ||
                    (std::is_pointer_v<std::tuple_element_t<
                         sizeof...(Args) - 1,
                         std::tuple<std::remove_reference_t<Args>...>>>);
    };

    template <size_type Dims>
    class IOManager
    {
      private:
        struct LibraryDeleter {
            using pointer = void*;

            void operator()(pointer p) const noexcept
            {
                if (p) {
                    dlclose(p);
                }
            }
        };

        SolverManager& solver_manager_;
        std::string data_directory_;
        std::string hydro_source_lib_;
        std::string gravity_source_lib_;
        std::string boundary_source_lib_;
        size_type current_iter_{0};
        size_type checkpoint_zones_;
        size_type checkpoint_idx_{0};

        // library handles
        std::shared_ptr<void> hsource_handle_;
        std::shared_ptr<void> gsource_handle_;
        std::shared_ptr<void> bsource_handle_;

        using function_t = typename helpers::real_func<Dims>::type;

        // hydrodynamic source functions
        function_t hydro_source_;

        // gravity source functions
        function_t gravity_source_;

        // boundary source functions at x1 boundaries
        function_t bx1_inner_source_;
        function_t bx1_outer_source_;
        // boundary source functions at x2 boundaries
        function_t bx2_inner_source_;
        function_t bx2_outer_source_;
        // boundary source functions at x3 boundaries
        function_t bx3_inner_source_;
        function_t bx3_outer_source_;

      public:
        // move constructor and assignment
        IOManager(IOManager&& other) noexcept            = default;
        IOManager& operator=(IOManager&& other) noexcept = default;

        // Delete copy operations to prevent double-free
        IOManager(const IOManager&)            = delete;
        IOManager& operator=(const IOManager&) = delete;

        IOManager(SolverManager& solver_manager, const InitialConditions& init)
            : solver_manager_(solver_manager),
              data_directory_(init.data_directory),
              hydro_source_lib_(init.hydro_source_lib),
              gravity_source_lib_(init.gravity_source_lib),
              boundary_source_lib_(init.boundary_source_lib),
              checkpoint_zones_(determine_checkpoint_zones(init)),
              checkpoint_idx_(init.checkpoint_index),
              hsource_handle_(nullptr, LibraryDeleter()),
              gsource_handle_(nullptr, LibraryDeleter()),
              bsource_handle_(nullptr, LibraryDeleter())
        {
        }

        // shared_ptr cleans up libraries
        ~IOManager() = default;

        void load_functions()
        {
            // Load the symbol based on the dimension
            using f2arg = void (*)(real, real, real[]);
            using f3arg = void (*)(real, real, real, real[]);
            using f4arg = void (*)(real, real, real, real, real[]);

            //=================================================================
            // Check if the hydro source library is set
            //=================================================================
            if (!hydro_source_lib_.empty()) {
                // Load the shared library
                void* handle = dlopen(hydro_source_lib_.c_str(), RTLD_LAZY);
                if (!handle) {
                    std::cerr << "Cannot open library: " << dlerror() << '\n';
                    return;
                }
                hsource_handle_ =
                    std::shared_ptr<void>(handle, LibraryDeleter());

                // Clear any existing error
                dlerror();

                const std::vector<std::pair<const char*, function_t&>> symbols =
                    {
                      {"hydro_source", hydro_source_},
                    };

                bool success = true;
                for (const auto& [symbol, func] : symbols) {
                    void* source            = dlsym(handle, symbol);
                    const char* dlsym_error = dlerror();
                    // if can't load symbol, print error and
                    // set null_sources to true
                    if (dlsym_error) {
                        std::cerr << "Cannot load symbol '" << symbol
                                  << "': " << dlsym_error << '\n';
                        success = false;
                        dlclose(handle);
                        break;
                    }

                    // Assign the function pointer based on the dimension
                    if constexpr (Dims == 1) {
                        func = reinterpret_cast<f2arg>(source);
                    }
                    else if constexpr (Dims == 2) {
                        func = reinterpret_cast<f3arg>(source);
                    }
                    else if constexpr (Dims == 3) {
                        func = reinterpret_cast<f4arg>(source);
                    }
                }
                if (success) {
                    solver_manager_.set_null_sources(false);
                }
            }

            //=================================================================
            // Check if the gravity source library is set
            //=================================================================
            if (!gravity_source_lib_.empty()) {
                void* handle = dlopen(gravity_source_lib_.c_str(), RTLD_LAZY);
                if (!handle) {
                    std::cerr << "Cannot open library: " << dlerror() << '\n';
                    return;
                }
                gsource_handle_ =
                    std::shared_ptr<void>(handle, LibraryDeleter());

                // Clear any existing error
                dlerror();

                // Load the symbol based on the dimension
                const std::vector<std::pair<const char*, function_t&>>
                    g_symbols = {
                      {"gravity_source", gravity_source_},
                    };

                bool success = true;
                for (const auto& [symbol, func] : g_symbols) {
                    void* source            = dlsym(handle, symbol);
                    const char* dlsym_error = dlerror();
                    // if can't load symbol, print error
                    if (dlsym_error) {
                        std::cerr << "Cannot load symbol '" << symbol
                                  << "': " << dlsym_error << '\n';
                        success = false;
                        dlclose(handle);
                        break;
                    }

                    // Assign the function pointer based on the dimension
                    if constexpr (Dims == 1) {
                        func = reinterpret_cast<f2arg>(source);
                    }
                    else if constexpr (Dims == 2) {
                        func = reinterpret_cast<f3arg>(source);
                    }
                    else if constexpr (Dims == 3) {
                        func = reinterpret_cast<f4arg>(source);
                    }
                }
                if (success) {
                    solver_manager_.set_null_gravity(false);
                }
            }

            //=================================================================
            // Check if the boundary source library is set
            //=================================================================
            if (!boundary_source_lib_.empty()) {
                void* handle = dlopen(boundary_source_lib_.c_str(), RTLD_LAZY);
                if (!handle) {
                    std::cerr << "Cannot open library: " << dlerror() << '\n';
                    return;
                }
                bsource_handle_ =
                    std::shared_ptr<void>(handle, LibraryDeleter());

                // Clear any existing error
                dlerror();

                // Load the symbol based on the dimension
                const std::vector<std::pair<const char*, function_t&>>
                    b_symbols = {
                      {"bx1_inner_source", bx1_inner_source_},
                      {"bx1_outer_source", bx1_outer_source_},
                      {"bx2_inner_source", bx2_inner_source_},
                      {"bx2_outer_source", bx2_outer_source_},
                      {"bx3_inner_source", bx3_inner_source_},
                      {"bx3_outer_source", bx3_outer_source_},
                    };

                for (const auto& [symbol, func] : b_symbols) {
                    void* source            = dlsym(handle, symbol);
                    const char* dlsym_error = dlerror();
                    // if can't load symbol, print error
                    if (dlsym_error) {
                        // erro out  only  if the boundary
                        // condition is set to dynamic
                        for (size_type ii = 0; ii < 2 * Dims; ++ii) {
                            if (symbol == b_symbols[ii].first &&
                                solver_manager_.boundary_conditions()[ii] ==
                                    BoundaryCondition::DYNAMIC) {
                                std::cerr << "Cannot load symbol '" << symbol
                                          << "': " << dlsym_error << '\n';
                                solver_manager_.boundary_conditions()[ii] =
                                    BoundaryCondition::OUTFLOW;
                                dlclose(handle);
                            }
                        }
                    }
                    else {
                        // Assign the function pointer based on the dimension
                        if constexpr (Dims == 1) {
                            func = reinterpret_cast<f2arg>(source);
                        }
                        else if constexpr (Dims == 2) {
                            func = reinterpret_cast<f3arg>(source);
                        }
                        else if constexpr (Dims == 3) {
                            func = reinterpret_cast<f4arg>(source);
                        }
                    }
                }
            }
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
        auto checkpoint_index() const { return checkpoint_idx_; }

        template <typename... Args>
            requires ValidSourceParams<Dims, Args...>
        auto call_hydro_source(Args&&... args) const
        {
            if (solver_manager_.null_sources()) {
                return;
            }
            hydro_source_(std::forward<Args>(args)...);
        }

        template <typename... Args>
            requires ValidSourceParams<Dims, Args...>
        auto call_gravity_source(Args&&... args) const
        {
            if (solver_manager_.null_gravity()) {
                return;
            }
            gravity_source_(std::forward<Args>(args)...);
        }

        template <typename... Args>
            requires ValidSourceParams<Dims, Args...>
        void call_boundary_source(BoundaryFace face, Args&&... args) const
        {
            switch (face) {
                case BoundaryFace::X1_INNER:
                    if (bx1_inner_source_) {
                        bx1_inner_source_(std::forward<Args>(args)...);
                    }
                    break;
                case BoundaryFace::X1_OUTER:
                    if (bx1_outer_source_) {
                        bx1_outer_source_(std::forward<Args>(args)...);
                    }
                    break;
                case BoundaryFace::X2_INNER:
                    if (bx2_inner_source_) {
                        bx2_inner_source_(std::forward<Args>(args)...);
                    }
                    break;
                case BoundaryFace::X2_OUTER:
                    if (bx2_outer_source_) {
                        bx2_outer_source_(std::forward<Args>(args)...);
                    }
                    break;
                case BoundaryFace::X3_INNER:
                    if (bx3_inner_source_) {
                        bx3_inner_source_(std::forward<Args>(args)...);
                    }
                    break;
                case BoundaryFace::X3_OUTER:
                    if (bx3_outer_source_) {
                        bx3_outer_source_(std::forward<Args>(args)...);
                    }
                    break;
            }
        }

      private:
        size_type determine_checkpoint_zones(const InitialConditions& init)
        {
            const auto [xag, yag, zag] = init.active_zones();
            return (zag > 1) ? zag : (yag > 1) ? yag : xag;
        }
    };   // namespace simbi
}   // namespace simbi

#endif
