/**
 *  *=============================================================================
 *  *           SIMBI - Special Relativistic Magnetohydrodynamics Code
 *  *=============================================================================
 *  *
 *  * @file            exceptions.hpp
 *  * @brief           assortment of exception classes
 *  * @details
 *  *
 *  * @version         0.8.0
 *  * @date            2025-02-26
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
 *  * 2025-02-26      v0.8.0      Initial implementation
 *  *
 *  *==============================================================================
 *  * @copyright (C) 2025 Marcus DuPont. All rights reserved.
 *  *==============================================================================
 */
#ifndef EXCEPTIONS_HPP
#define EXCEPTIONS_HPP
#include "build_options.hpp"
#include <cstdint>
#include <exception>

namespace simbi {
    enum class ErrorCode : uint32_t {
        NONE                  = 0,
        NEGATIVE_PRESSURE     = 1 << 0,
        NON_FINITE_PRESSURE   = 1 << 1,
        NEGATIVE_DENSITY      = 1 << 2,
        SUPERLUMINAL_VELOCITY = 1 << 3,
        NEGATIVE_ENERGY       = 1 << 4,
        NEGATIVE_ENTROPY      = 1 << 5,
        NEGATIVE_MASS         = 1 << 6,
        NON_FINITE_ROOT       = 1 << 7,
        MAX_ITER              = 1 << 8,
        UNDEFINED             = 1 << 9,
    };

    namespace exception {
        class InterruptException : public std::exception
        {
          public:
            InterruptException(int s) : status(s) {};
            const char* what() const noexcept;
            int status;
        };

        class SimulationFailureException : public std::exception
        {
          public:
            SimulationFailureException() = default;
            const char* what() const noexcept;
        };

    }   // namespace exception

    inline DUAL constexpr ErrorCode operator|(ErrorCode lhs, ErrorCode rhs)
    {
        return static_cast<ErrorCode>(
            static_cast<std::underlying_type_t<ErrorCode>>(lhs) |
            static_cast<std::underlying_type_t<ErrorCode>>(rhs)
        );
    }

    inline DUAL constexpr ErrorCode operator&(ErrorCode lhs, ErrorCode rhs)
    {
        return static_cast<ErrorCode>(
            static_cast<std::underlying_type_t<ErrorCode>>(lhs) &
            static_cast<std::underlying_type_t<ErrorCode>>(rhs)
        );
    }

    inline DUAL constexpr bool has_error(ErrorCode code, ErrorCode error)
    {
        return (static_cast<uint32_t>(code) & static_cast<uint32_t>(error)) !=
               0;
    }
}   // namespace simbi

#endif
