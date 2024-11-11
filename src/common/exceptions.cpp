#include "exceptions.hpp"   // for InterruptException, SimulationFailureException

namespace simbi {
    namespace exception {
        const char* InterruptException::what() const noexcept
        {
            return "\033[1;37mSimulation interrupted. Saving last "
                   "checkpoint...\033[0m";
        }

        const char* SimulationFailureException::what() const noexcept
        {
            // crashed in bold red!
            return "\033[1;31mSimulation Crashed\033[0m";
        }
    }   // namespace exception

}   // namespace simbi