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

#include "build_options.hpp"
#include "core/managers/solver_manager.hpp"         // for SolverManager
#include "core/types/utility/init_conditions.hpp"   // for InitialConditions
#include "util/jit/device_callable.hpp"
#include "util/jit/func_registry.hpp"   // for FunctionRegistry
#include "util/jit/load.hpp"
#include "util/jit/source_code.hpp"
#include <dlfcn.h>   // for dlopen, dlclose, dlsym
#include <string>

using namespace simbi::jit;

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

    template <size_type D>
    struct FunctionSignature {
        using type = std::conditional_t<
            D == 1,
            void(real, real, real*),   // 1D: x1, t, res
            std::conditional_t<
                D == 2,
                void(real, real, real, real*),           // 2D: x1, x2, t, res
                void(real, real, real, real, real*)>>;   // 3D: x1,
                                                         // x2, x3,
                                                         // t, res
    };

    template <size_type D>
    using function_signature_t = typename FunctionSignature<D>::type;

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

        jit::FunctionRegistry<Dims> function_registry_;
        DeviceCallable<Dims> hydro_source_function_;
        DeviceCallable<Dims> gravity_source_function_;
        std::array<DeviceCallable<Dims>, 2 * Dims> boundary_functions_;

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
              bsource_handle_(nullptr, LibraryDeleter()),
              function_registry_(),
              hydro_source_function_("hydro_source", nullptr),
              gravity_source_function_("gravity_source", nullptr)
        {
            setup_hydro_source(init.hydro_source_code);
            setup_gravity_source(init.gravity_source_code);
            setup_boundary_source(init.boundary_source_code);

            for (size_type ii = 0; ii < boundary_functions_.size(); ++ii) {
                std::string part = (ii % 2 == 0) ? "inner" : "outer";

                boundary_functions_[ii] = DeviceCallable<Dims>(
                    "bx" + std::to_string(ii) + "_" + part + "_source",
                    nullptr
                );
            }
        }

        // shared_ptr cleans up libraries
        ~IOManager() = default;

        void setup_hydro_source(const std::string& hydro_source_code)
        {
            if (hydro_source_code.empty()) {
                return;
            }

            auto source = jit::SourceCode(hydro_source_code, "hydro_source");
            auto result = jit::compile_and_load<Dims>(source, source.name);

            if (result.is_ok()) {
                // Store in registry using new model
                function_registry_ = function_registry_.with_function(
                    source.name,
                    DeviceCallable<Dims>(source.name, result.value())
                );
            }
            else {
                std::cerr << "Error compiling hydro source code: "
                          << result.error() << '\n';
                solver_manager_.set_null_sources(true);
            }
        }

        void setup_gravity_source(const std::string& gravity_source_code)
        {
            if (gravity_source_code.empty()) {
                return;
            }

            auto source =
                jit::SourceCode(gravity_source_code, "gravity_source");
            auto result = jit::compile_and_load<Dims>(source, source.name);

            if (result.is_ok()) {
                // Store in registry using new model
                function_registry_ = function_registry_.with_function(
                    source.name,
                    DeviceCallable<Dims>(source.name, result.value())
                );
            }
            else {
                std::cerr << "Error compiling gravity source code: "
                          << result.error() << '\n';
                solver_manager_.set_null_gravity(true);
            }
        }

        void setup_boundary_source(const std::string& boundary_source_code)
        {
            if (boundary_source_code.empty()) {
                return;
            }

            auto source =
                jit::SourceCode(boundary_source_code, "boundary_source");
            auto result = jit::compile_and_load<Dims>(source, source.name);

            if (result.is_ok()) {
                auto callable =
                    DeviceCallable<Dims>(source.name, result.value());

                // Register for all boundary faces
                const std::array<std::string, 6> boundary_names = {
                  "bx1_inner_source",
                  "bx1_outer_source",
                  "bx2_inner_source",
                  "bx2_outer_source",
                  "bx3_inner_source",
                  "bx3_outer_source"
                };

                for (const auto& name : boundary_names) {
                    function_registry_ =
                        function_registry_.with_function(name, callable);
                }
            }
            else {
                std::cerr << "Error compiling boundary source code: "
                          << result.error() << '\n';
            }
        }

        // After loading all functions, prepare GPU-accessible function cache
        void prepare_gpu_functions()
        {
            // Look up functions from registry and cache them
            hydro_source_function_ =
                function_registry_.get_function_or_noop("hydro_source");
            gravity_source_function_ =
                function_registry_.get_function_or_noop("gravity_source");

            // Cache boundary functions
            const std::array<std::string, 6> boundary_names = {
              "bx1_inner_source",
              "bx1_outer_source",
              "bx2_inner_source",
              "bx2_outer_source",
              "bx3_inner_source",
              "bx3_outer_source"
            };

            for (size_type i = 0;
                 i < std::min<size_type>(boundary_names.size(), 2 * Dims);
                 ++i) {
                boundary_functions_[i] =
                    function_registry_.get_function_or_noop(boundary_names[i]);
            }
        }
        void load_functions()
        {
            load_hydro_functions();
            load_gravity_functions();
            load_boundary_functions();

            // prepare gpu-accessible function cache
            prepare_gpu_functions();
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

        // call 'em
        template <typename... Args>
            requires(sizeof...(Args) == Dims + 2)
        DEV void call_hydro_source(Args&&... args) const
        {
            if (solver_manager_.null_sources()) {
                return;
            }
            hydro_source_function_(std::forward<Args>(args)...);
        }

        template <typename... Args>
            requires(sizeof...(Args) == Dims + 2)
        DEV void call_gravity_source(Args&&... args) const
        {
            if (solver_manager_.null_gravity()) {
                return;
            }
            gravity_source_function_(std::forward<Args>(args)...);
        }

        template <typename... Args>
            requires(sizeof...(Args) == Dims + 2)
        DEV void call_boundary_source(BoundaryFace face, Args&&... args) const
        {
            // Convert enum to array index
            int idx = static_cast<int>(face);
            if (idx >= 0 && idx < boundary_functions_.size()) {
                boundary_functions_[idx](std::forward<Args>(args)...);
            }
        }

      private:
        size_type determine_checkpoint_zones(const InitialConditions& init)
        {
            const auto [xag, yag, zag] = init.active_zones();
            return (zag > 1) ? zag : (yag > 1) ? yag : xag;
        }

        // helper methods to load functions from shared libraries
        void load_hydro_functions()
        {
            if (hydro_source_lib_.empty()) {
                return;
            }

            // load the shared library
            void* handle = dlopen(hydro_source_lib_.c_str(), RTLD_LAZY);
            if (!handle) {
                std::cerr << "Cannot open library: " << dlerror() << '\n';
                return;
            }
            hsource_handle_ = std::shared_ptr<void>(handle, LibraryDeleter());

            // clear any existing error
            dlerror();

            // load the function
            void* source            = dlsym(handle, "hydro_source");
            const char* dlsym_error = dlerror();

            if (dlsym_error) {
                std::cerr << "Cannot load symbol 'hydro_source': "
                          << dlsym_error << '\n';
                solver_manager_.set_null_sources(true);
                return;
            }

            // create a callable wrapped around the function pointer
            using FuncPtr = user_function_ptr_t<Dims>;
            auto func_ptr = reinterpret_cast<FuncPtr>(source);

            // store in registry
            function_registry_ = function_registry_.with_function(
                "hydro_source",
                DeviceCallable<Dims>(
                    "hydro_source",
                    [func_ptr](auto&&... args) {
                        func_ptr(std::forward<decltype(args)>(args)...);
                    }
                )
            );

            solver_manager_.set_null_sources(false);
        }

        void load_gravity_functions()
        {
            if (gravity_source_lib_.empty()) {
                return;
            }

            // load the shared library
            void* handle = dlopen(gravity_source_lib_.c_str(), RTLD_LAZY);
            if (!handle) {
                std::cerr << "Cannot open library: " << dlerror() << '\n';
                return;
            }
            gsource_handle_ = std::shared_ptr<void>(handle, LibraryDeleter());

            // clear any existing error
            dlerror();

            // load the function
            void* source            = dlsym(handle, "gravity_source");
            const char* dlsym_error = dlerror();

            if (dlsym_error) {
                std::cerr << "Cannot load symbol 'gravity_source': "
                          << dlsym_error << '\n';
                solver_manager_.set_null_gravity(true);
                return;
            }

            // create a callable wrapped around the function pointer
            using FuncPtr = user_function_ptr_t<Dims>;
            auto func_ptr = reinterpret_cast<FuncPtr>(source);

            // store in registry
            function_registry_ = function_registry_.with_function(
                "gravity_source",
                DeviceCallable<Dims>(
                    "gravity_source",
                    [func_ptr](auto&&... args) {
                        func_ptr(std::forward<decltype(args)>(args)...);
                    }
                )
            );

            solver_manager_.set_null_gravity(false);
        }

        void load_boundary_functions()
        {
            if (boundary_source_lib_.empty()) {
                return;
            }

            // Load the shared library
            void* handle = dlopen(boundary_source_lib_.c_str(), RTLD_LAZY);
            if (!handle) {
                std::cerr << "Cannot open boundary library: " << dlerror()
                          << '\n';
                return;
            }
            bsource_handle_ = std::shared_ptr<void>(handle, LibraryDeleter());

            // Clear any existing error
            dlerror();

            // Define all boundary function names
            const std::array<std::string, 6> boundary_function_names = {
              "bx1_inner_source",
              "bx1_outer_source",
              "bx2_inner_source",
              "bx2_outer_source",
              "bx3_inner_source",
              "bx3_outer_source"
            };

            // Only try to load functions for dimensions that exist
            const size_type num_faces = 2 * Dims;   // 2 faces per dimension

            // Track which functions were loaded
            std::unordered_map<std::string, bool> loaded_functions;

            // Load each boundary function
            for (size_type i = 0; i < num_faces; ++i) {
                const std::string& func_name = boundary_function_names[i];
                void* func_ptr               = dlsym(handle, func_name.c_str());
                const char* dlsym_error      = dlerror();

                if (dlsym_error) {
                    // Function not found, but only report error if this
                    // boundary uses DYNAMIC condition
                    if (solver_manager_.boundary_conditions()[i] ==
                        BoundaryCondition::DYNAMIC) {
                        std::cerr << "Cannot load boundary function '"
                                  << func_name << "': " << dlsym_error << '\n';
                        std::cerr << "Falling back to OUTFLOW boundary "
                                     "condition for this face."
                                  << '\n';

                        // Fall back to OUTFLOW for this boundary
                        solver_manager_.boundary_conditions()[i] =
                            BoundaryCondition::OUTFLOW;
                    }
                    loaded_functions[func_name] = false;
                }
                else {
                    // Successfully loaded function - create callable
                    using FuncPtr     = user_function_ptr_t<Dims>;
                    auto function_ptr = reinterpret_cast<FuncPtr>(func_ptr);

                    // Create callable with appropriate signature
                    auto callable = DeviceCallable<Dims>(
                        func_name,
                        [function_ptr](auto&&... args) {
                            function_ptr(std::forward<decltype(args)>(args)...);
                        }
                    );

                    // Store in registry
                    function_registry_ = function_registry_.with_function(
                        func_name,
                        std::move(callable)
                    );

                    loaded_functions[func_name] = true;
                }
            }

            // Check if we should try a generic "boundary_source" function
            bool need_generic_function = false;
            for (size_type i = 0; i < num_faces; ++i) {
                const std::string& func_name = boundary_function_names[i];
                if (solver_manager_.boundary_conditions()[i] ==
                        BoundaryCondition::DYNAMIC &&
                    !loaded_functions[func_name]) {
                    need_generic_function = true;
                    break;
                }
            }

            // Try loading a generic boundary source function if any DYNAMIC
            // boundaries don't have their specific function loaded
            if (need_generic_function) {
                void* generic_func      = dlsym(handle, "boundary_source");
                const char* dlsym_error = dlerror();

                if (dlsym_error) {
                    std::cerr
                        << "Cannot load generic 'boundary_source' function: "
                        << dlsym_error << '\n';

                    // Fall back to OUTFLOW for all DYNAMIC boundaries without a
                    // function
                    for (size_type i = 0; i < num_faces; ++i) {
                        const std::string& func_name =
                            boundary_function_names[i];
                        if (solver_manager_.boundary_conditions()[i] ==
                                BoundaryCondition::DYNAMIC &&
                            !loaded_functions[func_name]) {
                            std::cerr << "No boundary function for "
                                      << func_name
                                      << ". Falling back to OUTFLOW." << '\n';
                            solver_manager_.boundary_conditions()[i] =
                                BoundaryCondition::OUTFLOW;
                        }
                    }
                }
                else {
                    // Successfully loaded generic function
                    using FuncPtr     = user_function_ptr_t<Dims>;
                    auto function_ptr = reinterpret_cast<FuncPtr>(generic_func);

                    // Create callable
                    auto callable = DeviceCallable<Dims>(
                        "boundary_source",
                        [function_ptr](auto&&... args) {
                            function_ptr(std::forward<decltype(args)>(args)...);
                        }
                    );

                    // Register the generic function for any DYNAMIC boundaries
                    // without specific functions
                    for (size_type i = 0; i < num_faces; ++i) {
                        const std::string& func_name =
                            boundary_function_names[i];
                        if (solver_manager_.boundary_conditions()[i] ==
                                BoundaryCondition::DYNAMIC &&
                            !loaded_functions[func_name]) {

                            // Store the generic function under the specific
                            // name
                            function_registry_ =
                                function_registry_.with_function(
                                    func_name,
                                    callable
                                );

                            loaded_functions[func_name] = true;
                        }
                    }
                }
            }

            // Log what was loaded
            std::cout << "Boundary functions loaded:" << std::endl;
            for (size_type i = 0; i < num_faces; ++i) {
                const std::string& func_name = boundary_function_names[i];
                std::cout << "  " << func_name << ": "
                          << (loaded_functions[func_name] ? "Yes" : "No");

                if (solver_manager_.boundary_conditions()[i] ==
                    BoundaryCondition::DYNAMIC) {
                    std::cout << " (DYNAMIC)";
                }
                else {
                    std::cout << " ("
                              << solver_manager_.boundary_conditions_c_str()[i]
                              << ")";
                }
                std::cout << std::endl;
            }
        }
    };   // namespace simbi
}   // namespace simbi

#endif
