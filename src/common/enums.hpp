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
        AXIS_CYLINDRICAL,
        PLANAR_CYLINDRICAL,
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

    enum class TIMESTEP_TYPE {
        MINIMUM,
        ADAPTIVE,
    };

    enum class HLLCTYPE {
        CLASSICAL,
        // Apply the low-Mach HLLC fix found in Fleischmann et al 2020: 
        // https://www.sciencedirect.com/science/article/pii/S0021999120305362
        FLEISCHMANN,
    };

    enum class BoundaryCondition {
        REFLECTING,
        OUTFLOW,
        INFLOW,
        PERIODIC,
    };

    enum class HydroRegime {
        RELATIVISTC,
        NEWTONIAN,
    };

    enum class WaveSpeeds {
        SCHNEIDER_ET_AL_93,
        MIGNONE_AND_BODO_05,
        HUBER_AND_KISSMANN_2021,
        NAIVE,
    };
    constexpr auto comp_wave_speed = WaveSpeeds::MIGNONE_AND_BODO_05;
    constexpr auto comp_hllc_type  = HLLCTYPE::CLASSICAL;
}
#endif