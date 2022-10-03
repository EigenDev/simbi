/**
 * Houses all of the system configuration enum classes
 * 
*/
#ifndef ENUMS_HPP
#define ENUMS_HPP

namespace simbi{
    enum class MemSide {
        Host,
        Dev,
    };

    enum class Cellspacing {
        LINSPACE,
        LOGSPACE,
    };

    enum class Geometry {
        CARTESIAN,
        SPHERICAL,
        CYLINDRICAL,
    };

    enum class Accuracy
    {
        FIRST_ORDER,
        SECOND_ORDER,
    };

    enum class Solver {
        HLLC,
        HLLE,
    };

    enum class BoundaryCondition {
        REFLECTING,
        OUTFLOW,
        INFLOW,
        PERIODIC,
    };

    enum class WaveSpeeds {
        SCHNEIDER_ET_AL_93,
        MIGNONE_AND_BODO_05,
        NAIVE,
    };
    constexpr auto comp_wave_speed = WaveSpeeds::MIGNONE_AND_BODO_05;
}
#endif