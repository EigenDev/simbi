#include "exceptions.hpp"   // for InterruptException, SimulationFailureException

namespace simbi {
    namespace exception {
        const char* InterruptException::what() const noexcept
        {
            return "++{Simulation interrupted. Saving last "
                   "checkpoint...}++";
        }

        const char* SimulationFailureException::what() const noexcept
        {
            return "++{Simulation Crashed}++";
        }
    }   // namespace exception

}   // namespace simbi