/**
 * Houses all of the system configuration enum classes
 * 
*/

#ifndef CONFIG_H
#define CONFIG_H

namespace simbi{
    enum class Cellspacing
    {
        LINSPACE,
        LOGSPACE,
    };

    enum class Geometry
    {
        SPHERICAL,
        CARTESIAN,
    };

    enum class Accuracy
    {
        FIRST_ORDER,
        SECOND_ORDER,
    };

    enum class Dimensions
    {
        ONE_D,
        TWO_D,
    };

    enum class Solver
    {
        HLLC,
        HLLE,
    };

    enum class WaveSpeeds
    {
        SCHNEIDER_ET_AL_93,
        MIGNONE_AND_BODO_05,
    };
}

#endif