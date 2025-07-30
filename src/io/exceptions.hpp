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

#include "base/concepts.hpp"
#include "config.hpp"
#include <atomic>
#include <cstdint>
#include <exception>
#include <string>
#include <type_traits>

namespace simbi {
    // forward declarations
    template <typename T, std::uint64_t Dims>
    struct vector_t;

    template <std::uint64_t Dims>
    using uarray = vector_t<std::uint64_t, Dims>;

    template <std::uint64_t Dims>
    using iarray = vector_t<std::int64_t, Dims>;

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

    namespace helpers {
        std::string error_code_to_string(ErrorCode code);
    }

    namespace exception {
        class InterruptException : public std::exception
        {
          public:
            InterruptException(std::int64_t s) : status(s) {};
            const char* what() const noexcept;
            std::int64_t status;
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

    struct error_budget_t {
        std::atomic<int> remaining{1};   // start with budget of 1 error
        std::atomic<bool> error_captured{false};

        // error details (only filled by the winning thread)
        std::atomic<ErrorCode> first_error_code{ErrorCode::NONE};
        std::atomic<std::uint64_t> first_error_index{0};

        // try to consume budget - returns true if this thread "wins"
        bool try_consume()
        {
            int expected = 1;
            return remaining.compare_exchange_strong(expected, 0);
        }

        // check if budget is exhausted
        bool is_exhausted() const { return remaining.load() <= 0; }

        void reset()
        {
            remaining.store(1);
            error_captured.store(false);
            first_error_code.store(ErrorCode::NONE);
            first_error_index.store(0);
        }
    };

    template <std::uint64_t Dims>
    std::string format_coord(const iarray<Dims>& coord)
    {
        std::string result = "(";
        for (std::uint64_t i = 0; i < Dims; ++i) {
            result += std::to_string(coord[i]);
            if (i < Dims - 1) {
                result += ", ";
            }
        }
        result += ")";
        return result;
    }

    template <std::uint64_t Dims, typename T>
    std::string format_position(const vector_t<T, Dims>& position)
    {
        std::string result = "(";
        for (std::uint64_t i = 0; i < Dims; ++i) {
            result += std::to_string(position[i]);
            if (i < Dims - 1) {
                result += ", ";
            }
        }
        result += ")";
        return result;
    }

    template <typename conserved_t>
    std::string format_conserved(const conserved_t& cons)
    {
        std::string result = "Conservative State: ";
        result += "den=" + std::to_string(cons.den) + ", ";
        result += "mom=(";
        for (std::uint64_t i = 0; i < cons.mom.size(); ++i) {
            result += std::to_string(cons.mom[i]);
            if (i < cons.mom.size() - 1) {
                result += ", ";
            }
        }

        if constexpr (concepts::is_mhd_conserved_c<conserved_t>) {
            result += "), nrg=" + std::to_string(cons.nrg) + ", mag=(";
            for (std::uint64_t i = 0; i < cons.mag.size(); ++i) {
                result += std::to_string(cons.mag[i]);
                if (i < cons.mag.size() - 1) {
                    result += ", ";
                }
            }
            result += "), chi=" + std::to_string(cons.chi);
        }
        else {
            result += "), nrg=" + std::to_string(cons.nrg) +
                      ", chi=" + std::to_string(cons.chi);
        }

        return result;
    }

    struct error_info_t {
        std::string coord_str;
        // std::string position_str;
        ErrorCode error_code;
        std::string message;   // additional message for context
    };

    class primitive_conversion_error_t : public std::exception
    {
      public:
        primitive_conversion_error_t(const error_info_t& error_info)
            : info_(error_info)
        {
        }

        const char* what() const noexcept override
        {
            if (what_message_.empty()) {
                what_message_ = format_error_message();
            }
            return what_message_.c_str();
        }

      private:
        error_info_t info_;
        mutable std::string what_message_;

        std::string format_error_message() const
        {
            return std::string("Primitive conversion failed:\n") + "  " +
                   info_.coord_str + "\n" +
                   "  "
                   "  Error: " +
                   helpers::error_code_to_string(info_.error_code) + "\n" +
                   (info_.message.empty() ? "" : "  Message: " + info_.message);
        }
    };

}   // namespace simbi

#endif
