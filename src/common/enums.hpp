/**
 * ***********************(C) COPYRIGHT 2023 Marcus DuPont**********************
 * @file       enums.hpp
 * @brief A place to house compilation enums for context switching in a sim
 *
 * @note
 * @history:
 *   Version   Date            Author          Modification    Email
 *   V0.8.0    Dec-03-2023     Marcus DuPont                   md4469@nyu.edu
 *
 * @verbatim
 * ==============================================================================
 *
 * ==============================================================================
 * @endverbatim
 * ***********************(C) COPYRIGHT 2023 Marcus DuPont**********************
 */

#ifndef ENUMS_HPP
#define ENUMS_HPP

namespace simbi {
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

    enum class Accuracy {
        FIRST_ORDER,
        SECOND_ORDER,
    };

    enum class Solver {
        HLLD,
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

    enum class CONS2PRIMTYPE {
        VOLUMETRIC,
        CHARGES,
    };

    enum class BlkAx {
        K,
        J,
        I
    };

    constexpr auto comp_wave_speed = WaveSpeeds::MIGNONE_AND_BODO_05;
    constexpr auto comp_hllc_type  = HLLCTYPE::CLASSICAL;
}   // namespace simbi
#endif