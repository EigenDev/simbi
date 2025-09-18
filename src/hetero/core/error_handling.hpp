#ifndef HETERO_CORE_ERROR_HANDLING_HPP
#define HETERO_CORE_ERROR_HANDLING_HPP

#include "backend_traits.hpp"

#include <stdexcept>
#include <string>

namespace simbi::hetero {

    enum class status_t {
        success,
        out_of_memory,
        invalid_argument,
        device_error,
        runtime_error
    };

    class compute_error : public std::runtime_error
    {
        status_t status_;

      public:
        explicit compute_error(status_t status, const std::string& message)
            : std::runtime_error(message), status_(status)
        {
        }

        status_t status() const noexcept { return status_; }
    };

    template <typename backend_t>
    struct error_translator_t {
        static_assert(
            False<backend_t>{},
            "error translation not implemented for backend"
        );
    };

    template <>
    struct error_translator_t<cpu_backend_t> {
        static status_t translate(int) { return status_t::success; }

        static void check_and_throw(int, const char*)
        {
            // cpu operations don't typically fail in ways we can detect
        }
    };

#ifdef CUDA_ENABLED
    template <>
    struct error_translator_t<cuda_backend_t> {
        static status_t translate(cudaError_t error)
        {
            switch (error) {
                case cudaSuccess: return status_t::success;
                case cudaErrorMemoryAllocation: return status_t::out_of_memory;
                case cudaErrorInvalidValue:
                case cudaErrorInvalidDevicePointer:
                case cudaErrorInvalidConfiguration:
                    return status_t::invalid_argument;
                case cudaErrorDeviceUninitialized:
                    return status_t::device_error;
                default: return status_t::runtime_error;
            }
        }

        static void check_and_throw(cudaError_t error, const char* operation)
        {
            if (error != cudaSuccess) {
                auto status = translate(error);
                std::string message =
                    std::string(operation) + ": " + cudaGetErrorString(error);
                throw compute_error(status, message);
            }
        }
    };
#endif

#ifdef HIP_ENABLED
    template <>
    struct error_translator_t<hip_backend_t> {
        static status_t translate(hipError_t error)
        {
            switch (error) {
                case hipSuccess: return status_t::success;
                case hipErrorMemoryAllocation: return status_t::out_of_memory;
                case hipErrorInvalidValue:
                case hipErrorInvalidDevicePointer:
                case hipErrorInvalidConfiguration:
                    return status_t::invalid_argument;
                case hipErrorDeviceUninitialized:
                case hipErrorDeinitialized: return status_t::device_error;
                default: return status_t::runtime_error;
            }
        }

        static void check_and_throw(hipError_t error, const char* operation)
        {
            if (error != hipSuccess) {
                auto status = translate(error);
                std::string message =
                    std::string(operation) + ": " + hipGetErrorString(error);
                throw compute_error(status, message);
            }
        }
    };
#endif

    template <typename backend_t>
    void check_error(auto native_error, const char* operation)
    {
        error_translator_t<backend_t>::check_and_throw(native_error, operation);
    }

    template <typename backend_t>
    status_t translate_error(auto native_error)
    {
        return error_translator_t<backend_t>::translate(native_error);
    }

}   // namespace simbi::hetero

#endif   // HETERO_CORE_ERROR_HANDLING_HPP
