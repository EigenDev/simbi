#ifndef EXCEPTIONS_HPP
#define EXCEPTIONS_HPP
#include <exception>

namespace simbi {
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

}   // namespace simbi

#endif